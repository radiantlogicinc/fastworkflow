from typing import Callable, List, Optional
import random
import re
import time

import fastworkflow
import litellm

from fastworkflow.train.determinism import (
    PERSONA_ID_SEPARATOR,
    SEED_PERSONA_ID,
    UNRESOLVED_PERSONA_PREFIX,
    UtteranceProvenance,
    derived_seed,
    get_training_seed,
    record_provenance,
)
from fastworkflow.train.utterance_cache import (
    PRODUCTION_COMPLETION_BACKEND,
    PRODUCTION_PERSONA_SOURCE,
    Fingerprint,
    UtteranceCacheEntry,
    callable_identity,
    compute_fingerprint,
    get_utterance_cache,
    source_digest,
)
from fastworkflow.train.personas import (
    active_persona_source_label,
    active_persona_source_name,
    persona_source_needs_corpus,
    resolve_personas,
)
from fastworkflow.utils.logging import logger

# Conditional import of datasets - only required at runtime during training
try:
    from datasets import load_dataset
    _DATASETS_AVAILABLE = True
except ImportError:
    _DATASETS_AVAILABLE = False
    load_dataset = None

DEFAULT_MAX_RETRIES = 5
DEFAULT_RETRY_BASE_SECONDS = 2.0

# Transient failures worth retrying. Deliberately enumerated rather than catching
# litellm.exceptions.APIError, which is the base class of AuthenticationError and
# BadRequestError too - those are configuration or programming errors and retrying
# them just burns minutes before failing anyway.
RETRYABLE_LLM_EXCEPTIONS = (
    litellm.exceptions.RateLimitError,
    litellm.exceptions.Timeout,
    litellm.exceptions.APIConnectionError,
    litellm.exceptions.ServiceUnavailableError,
    litellm.exceptions.InternalServerError,
    litellm.exceptions.BadGatewayError,
)


def _resolve_generation_count(value: Optional[int], env_name: str) -> int:
    return value if value is not None else fastworkflow.get_env_var(env_name, int)


def call_with_retries(
    operation: Callable,
    *,
    description: str,
    max_retries: Optional[int] = None,
    base_delay: Optional[float] = None,
    rng: Optional[random.Random] = None,
):
    """Call *operation* retrying transient LLM failures with exponential backoff.

    Delay for attempt *n* is ``base_delay * 2**n`` plus a uniform jitter in
    ``[0, delay]``, so concurrent workers do not re-collide on the same window.
    Jitter is drawn from *rng* rather than the global `random` module: perturbing
    global state here would make every subsequent random draw in the process
    depend on how many retries happened, which is exactly the non-reproducibility
    this module exists to remove.

    Re-raises the last transient exception once the budget is exhausted. Anything
    outside `RETRYABLE_LLM_EXCEPTIONS` propagates immediately.
    """
    if max_retries is None:
        max_retries = DEFAULT_MAX_RETRIES
    if base_delay is None:
        base_delay = DEFAULT_RETRY_BASE_SECONDS
    if rng is None:
        rng = random.Random()

    total_attempts = max(0, int(max_retries)) + 1
    last_exception: Optional[BaseException] = None

    for attempt in range(total_attempts):
        try:
            return operation()
        except RETRYABLE_LLM_EXCEPTIONS as exc:
            last_exception = exc
            if attempt == total_attempts - 1:
                break
            delay = base_delay * (2 ** attempt)
            delay += rng.uniform(0.0, delay)
            logger.warning(
                f"{description}: {type(exc).__name__} on attempt "
                f"{attempt + 1}/{total_attempts}; retrying in {delay:.1f}s"
            )
            if delay > 0:
                time.sleep(delay)

    raise last_exception


def select_persona_indices(
    dataset_size: int,
    num_personas: int,
    seed: int,
    command_name: str,
) -> list[int]:
    """Pick PersonaHub row indices for *command_name*, reproducibly.

    Driven by a private `random.Random` seeded from `(seed, command_name)`, not the
    global `random` module. With the global module the result depends on how many
    random draws happened earlier in the process, so the same command sampled
    different personas depending on which contexts were trained before it - the
    free-running sampling of finding F2.
    """
    if not dataset_size or not num_personas or num_personas <= 0:
        return []
    rng = random.Random(derived_seed(seed, command_name))
    return rng.sample(range(dataset_size), min(num_personas, dataset_size))


def _persona_contributors(persona_id: str) -> tuple[set[str], bool]:
    """Normalize a provenance id into atomic contributors and unresolved state.

    Canonical unresolved ids prefix the whole composite (``__unresolved__:3+9``),
    but older or incrementally merged values can carry the prefix after a separator.
    Strip it from every atom so it can never become a phantom persona contributor.
    """
    contributors: set[str] = set()
    unresolved = False
    for contributor in persona_id.split(PERSONA_ID_SEPARATOR):
        while contributor.startswith(UNRESOLVED_PERSONA_PREFIX):
            unresolved = True
            contributor = contributor[len(UNRESOLVED_PERSONA_PREFIX):]
        if contributor:
            contributors.add(contributor)
    return contributors, unresolved


def _attribute_utterance(
    utterance_personas: dict[str, str],
    utterance: str,
    persona_id: str,
) -> None:
    """Attribute *utterance* to *persona_id*, merging on collision.

    `utterance_personas` is keyed by utterance text, so two personas producing the
    same wording collide. Overwriting would silently hand the text to whichever
    persona happened to come last, and the downstream evaluation split holds out
    WHOLE personas (R1/D1) - so a text produced by both a held-out and a training
    persona would look like a generalisation success while actually being trained
    on. Instead:

    * `SEED_PERSONA_ID` is absorbing: a generated utterance that duplicates a
      hand-written seed is a seed. Seeds are never held out.
    * Otherwise the ids are unioned into a sorted composite id joined by
      `PERSONA_ID_SEPARATOR`. A composite id means "produced by all of these";
      a whole-persona holdout must exclude such an utterance from the held-out set
      unless every contributor is held out.
    * If either attribution is unresolved, the normalized union remains unresolved.
      One prefix covers the full union; prefixes are never retained as persona atoms.
    """
    existing = utterance_personas.get(utterance)
    if existing is None:
        utterance_personas[utterance] = persona_id
        return
    if existing == persona_id:
        return
    if SEED_PERSONA_ID in (existing, persona_id):
        utterance_personas[utterance] = SEED_PERSONA_ID
        return
    existing_contributors, existing_unresolved = _persona_contributors(existing)
    new_contributors, new_unresolved = _persona_contributors(persona_id)
    merged = PERSONA_ID_SEPARATOR.join(
        sorted(existing_contributors | new_contributors)
    )
    if existing_unresolved or new_unresolved:
        merged = UNRESOLVED_PERSONA_PREFIX + merged
    utterance_personas[utterance] = merged


def _resolve_persona_id(
    echoed_name: str,
    batch_name_to_id: dict[str, str],
    batch_persona_ids: list[str],
) -> str:
    """Map the persona name echoed by the LLM back to a PersonaHub row id.

    The model is asked to repeat `[Persona_N]` above each block, but it can rename,
    re-case, or invent a header. What IS reliable is that everything in this
    response came from a persona in this batch. So: exact (case-insensitive) name
    match first, then the unambiguous single-persona batch, and otherwise a
    conservative `__unresolved__:` id naming every persona in the batch, which the
    holdout consumer must treat as belonging to all of them.
    """
    if persona_id := batch_name_to_id.get(echoed_name.strip().lower()):
        return persona_id
    if len(batch_persona_ids) == 1:
        return batch_persona_ids[0]
    return UNRESOLVED_PERSONA_PREFIX + PERSONA_ID_SEPARATOR.join(batch_persona_ids)


def _announce_fallback(command_name: str, reason: str, final_count: int) -> None:
    """Report a degraded command loudly enough to survive a tqdm-flooded log.

    F3's root cause was not the missing retry, it was that the one `logger.error`
    line was invisible among tens of thousands of progress-bar lines. The banner
    goes to stdout as well so it survives log-level configuration.
    """
    banner = "!" * 78
    logger.error(
        f"Synthetic utterance generation FELL BACK for command "
        f"'{command_name}': {reason}. Training on {final_count} utterances."
    )
    print(
        f"\n{banner}\n"
        f"!! DEGRADED TRAINING DATA: command '{command_name}'\n"
        f"!! reason: {reason}\n"
        f"!! kept the command name + hand-written seeds + whatever was generated\n"
        f"!! before the failure ({final_count} utterances total). Intent detection\n"
        f"!! for this command will be weak. See\n"
        f"!! ___command_info/training_provenance.json.\n"
        f"{banner}\n",
        flush=True,
    )


def load_persona_dataset():
    """Load PersonaHub. Isolated so callers can supply their own persona source."""
    return load_dataset("proj-persona/PersonaHub", data_files="persona.jsonl")['train']


# The system prompt below shows the model the response format using the literal word
# "utterance" as its placeholder, and models periodically copy that scaffolding through
# instead of filling it in. Neither existing filter catches it: "utterance" is nine
# characters, so `len(u) > 3` passes it, and it does not start with '['. The token was
# therefore trained as a POSITIVE example of whatever command was being generated --
# measured at 21 of 412 rows (5.1%) on retail, spread across seven real commands at once,
# so the classifier was shown one identical string carrying seven conflicting labels.
#
# Filtering here rather than rewording the prompt is deliberate. Rewording changes what the
# model is asked to produce, and so changes the training-data distribution for every command
# in every workflow in a way nobody has measured; dropping an echo removes a row that was
# never an utterance in the first place. bd fix-iy0.
_TEMPLATE_ECHOES = frozenset({
    "utterance",
    "utterances",
    "persona_name",
    "next_persona_name",
    "...",
})

# Leading list markers ("- utterance", "1. utterance", "* utterance") are how the echo
# usually arrives when the model half-formats its answer.
_LIST_MARKER = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s*")


def _is_template_echo(utterance: str) -> bool:
    """True if the model copied the prompt's format scaffolding instead of answering."""
    stripped = _LIST_MARKER.sub("", utterance).strip().strip("*_`<>[]").strip()
    return stripped.casefold() in _TEMPLATE_ECHOES


def generate_utterances_for_personas(
    seed_utterances: List[str],
    command_name,
    selected_personas: List[str],
    selected_indices: List[int],
    provenance: UtteranceProvenance,
    utterances_per_persona: Optional[int] = None,
    personas_per_batch: Optional[int] = None,
    model: Optional[str] = None,
    seed: Optional[int] = None,
    completion_fn: Optional[Callable] = None,
    _max_retries: int = DEFAULT_MAX_RETRIES,
    _retry_base_seconds: float = DEFAULT_RETRY_BASE_SECONDS,
) -> list[str]:
    """Run the batched LLM generation over already-selected personas.

    Split out from `generate_diverse_utterances_with_provenance` so the generation
    loop can be exercised against a supplied `completion_fn` without a network call
    and without a PersonaHub download. Mutates *provenance* in place: persona
    attribution, and `fell_back` / `fallback_reason` when a batch exhausts its
    retries. Returns only the generated utterances (no seeds, no command name).
    """
    utterances_per_persona = _resolve_generation_count(
        utterances_per_persona,
        "SYNTHETIC_UTTERANCE_GEN_UTTERANCES_PER_PERSONA",
    )
    personas_per_batch = _resolve_generation_count(
        personas_per_batch,
        "SYNTHETIC_UTTERANCE_GEN_PERSONAS_PER_BATCH",
    )
    if seed is None:
        seed = get_training_seed()
    if completion_fn is None:
        completion_fn = litellm.completion

    all_generated_responses = []
    used_personas = []

    # Extract common themes from seed utterances
    keywords = set()
    for utterance in seed_utterances:
        words = utterance.lower().split()
        keywords.update(words)

    # Sorted, not raw set order: set iteration order for strings varies with the
    # per-process hash salt, so an unsorted list would change the prompt text - and
    # therefore the generated utterances - between two otherwise identical runs.
    utterance_patterns = sorted(keywords)
    utterance_string = "\n".join(seed_utterances)

    # Jitter source for retries, derived so backoff never touches global RNG state.
    retry_rng = random.Random(derived_seed(seed, str(command_name), "retry-jitter"))

    # Process personas in batches
    for batch_start in range(0, len(selected_personas), personas_per_batch):
        batch_end = min(batch_start + personas_per_batch, len(selected_personas))
        batch_personas = selected_personas[batch_start:batch_end]

        # Map the persona names shown to the LLM back to PersonaHub row ids so the
        # echoed header in the response can be resolved to a real persona.
        batch_persona_ids = [str(i) for i in selected_indices[batch_start:batch_end]]
        batch_name_to_id: dict[str, str] = {}

        # Create combined prompt for all personas in batch
        batch_prompt = ""
        for idx, persona in enumerate(batch_personas):
            persona_name = f"Persona_{batch_start + idx + 1}"
            batch_name_to_id[persona_name.lower()] = batch_persona_ids[idx]
            used_personas.append({
                "name": persona_name,
                "description": persona,
                "persona_id": batch_persona_ids[idx]
            })
            batch_prompt += f"\n[{persona_name}]\n{persona}\n"

        messages = [
            {
                "role": "system",
                "content": f"""
                Generate {utterances_per_persona} unique utterances for each of the following personas.

                {batch_prompt}

                Use these seed utterances for style and intent:
                {utterance_string}

                Guidelines:
                - Generate exactly {utterances_per_persona} utterances per persona
                - Keep responses brief and natural
                - Maintain intent consistency with command: {command_name}
                - Avoid repeating the same structure
                - Use varied phrasing based on these themes: {', '.join(utterance_patterns)}

                Format your response exactly as follows:
                [Persona_Name]
                utterance
                utterance
                ...

                [Next_Persona_Name]
                utterance
                utterance
                ...
                """
            },
            {
                "role": "user",
                "content": f"Generate {utterances_per_persona} natural utterances for each persona listed above."
            }
        ]

        try:
            response = call_with_retries(
                lambda: completion_fn(
                    model=model,  # Corrected model name
                    messages=messages,
                    max_tokens=1000,
                    temperature=1.0,
                    top_p=0.9,
                    stop=["<|end_of_text|>"]
                ),
                description=(
                    f"Utterance generation for '{command_name}' "
                    f"(personas {batch_start + 1}-{batch_end})"
                ),
                max_retries=_max_retries,
                base_delay=_retry_base_seconds,
                rng=retry_rng,
            )
        except RETRYABLE_LLM_EXCEPTIONS as exc:
            # Partial-batch policy: keep everything the earlier batches produced and
            # abandon the remaining ones. The retry budget has already spent tens of
            # seconds against a limit that is account-wide and time-windowed, so the
            # next batch is overwhelmingly likely to fail the same way; continuing
            # multiplies wall clock for near-zero expected yield. What matters is
            # that the command still contributes rows - its label enters the
            # classifier either way, which is precisely what returning [] destroyed.
            provenance.fell_back = True
            provenance.fallback_reason = (
                f"{type(exc).__name__} after {_max_retries} retries on personas "
                f"{batch_start + 1}-{batch_end} of {len(selected_personas)}; "
                f"remaining batches abandoned"
            )
            break

        # Process responses
        content = response.choices[0].message.content.strip()

        # Split by persona sections
        sections = content.split('[')
        for section in sections[1:]:  # Skip first empty section
            try:
                # Extract persona name and utterances
                persona_name = section.split(']')[0].strip()
                utterances = section.split(']')[1].strip().split('\n')

                # Clean up utterances
                utterances = [u.strip() for u in utterances if u.strip()]
                utterances = [u for u in utterances if len(u) > 3 and not u.startswith('[')]
                utterances = [u for u in utterances if not _is_template_echo(u)]

                persona_id = _resolve_persona_id(
                    persona_name, batch_name_to_id, batch_persona_ids)
                for resp in utterances:
                    _attribute_utterance(
                        provenance.utterance_personas, resp, persona_id)

                all_generated_responses.extend([
                    {"utterance": resp, "persona": persona_name} for resp in utterances
                ])
            except IndexError:
                continue

    # Structure the output
    result = {
        "seed_utterances": seed_utterances,
        "generated_utterances": all_generated_responses,
        "personas": used_personas,
        "metadata": {
            "num_personas": len(selected_personas),
            "utterances_per_persona": utterances_per_persona,
            "personas_per_batch": personas_per_batch,
            "total_utterances": len(all_generated_responses)
        }
    }
    return [utt["utterance"] for utt in result["generated_utterances"]]


def utterance_fingerprint(
    seed_utterances: List[str],
    command_name,
    num_personas: int,
    utterances_per_persona: int,
    personas_per_batch: int,
    model: Optional[str],
    completion_fn: Optional[Callable] = None,
    persona_dataset_loader: Optional[Callable] = None,
) -> Fingerprint:
    """Fingerprint this command's generation configuration for the utterance cache.

    Everything the LLM sees is in here, including a digest of the source of
    `generate_utterances_for_personas` — that function IS the prompt, so an edit to
    it must invalidate every entry. See `utterance_cache.compute_fingerprint`.

    The proxy base is read with a code default so an absent one does not log a
    warning per command; `LITELLM_PROXY_API_BASE` is an env-file setting in this
    project, never a shell export.
    """
    api_base = fastworkflow.get_env_var("LITELLM_PROXY_API_BASE", str, default="")
    corpus_identity = callable_identity(
        persona_dataset_loader, PRODUCTION_PERSONA_SOURCE)
    # None for the default PersonaHub draw, which keeps this key byte-identical to what it
    # was before personas.py existed, so no existing workflow's cached utterances are
    # invalidated by that landing. Non-None makes an app-supplied or domain-conditioned run
    # a different cache variant -- without it, a workflow that adds a personas.json would
    # keep being served utterances written by the generic personas it just replaced.
    persona_label = active_persona_source_label()
    return compute_fingerprint(
        command_name=str(command_name),
        seed_utterances=list(seed_utterances),
        num_personas=num_personas,
        utterances_per_persona=utterances_per_persona,
        personas_per_batch=personas_per_batch,
        model=model,
        api_base=api_base or None,
        persona_source=(
            f"{corpus_identity}+{persona_label}" if persona_label else corpus_identity
        ),
        completion_backend=callable_identity(
            completion_fn, PRODUCTION_COMPLETION_BACKEND),
        # Both functions, because the post-filter decides which generated rows survive to
        # become training data just as surely as the prompt decides which rows exist. If only
        # the generator were digested, editing `_is_template_echo` would leave every cached
        # entry looking valid while the filter it was written under had changed.
        generator_source_digest="+".join((
            source_digest(generate_utterances_for_personas),
            source_digest(_is_template_echo),
        )),
    )


def _apply_cached_entry(
    provenance: UtteranceProvenance,
    entry: UtteranceCacheEntry,
) -> list[str]:
    """Rebuild *provenance*'s persona attribution from a cache hit.

    Re-runs `_attribute_utterance` rather than copying the stored map wholesale, so a
    reused utterance that happens to duplicate a hand-written seed is absorbed into
    `SEED_PERSONA_ID` exactly as it would have been on a fresh generation. Without
    that, a whole-persona holdout (R1/D1) would treat it as held-out-able and score a
    memorised row as a generalisation success.

    An entry written before attribution existed, or one whose attribution is missing
    a row, falls back to the conservative "produced by some persona in this set" id —
    the same shape `_resolve_persona_id` uses when the LLM's echoed header cannot be
    matched.
    """
    persona_ids = [str(p) for p in entry.persona_ids]
    provenance.persona_ids = persona_ids
    unresolved = (
        UNRESOLVED_PERSONA_PREFIX + PERSONA_ID_SEPARATOR.join(persona_ids)
        if persona_ids
        else UNRESOLVED_PERSONA_PREFIX
    )

    utterances = entry.usable_utterances()
    for utterance in utterances:
        persona_id = entry.utterance_personas.get(utterance) or unresolved
        _attribute_utterance(provenance.utterance_personas, utterance, persona_id)
    return utterances


def generate_diverse_utterances_with_provenance(
    seed_utterances: List[str],
    command_name,
    num_personas: Optional[int] = None,
    utterances_per_persona: Optional[int] = None,
    personas_per_batch: Optional[int] = None,
    seed: Optional[int] = None,
    completion_fn: Optional[Callable] = None,
    persona_dataset_loader: Optional[Callable] = None,
    _max_retries: int = DEFAULT_MAX_RETRIES,
    _retry_base_seconds: float = DEFAULT_RETRY_BASE_SECONDS,
) -> tuple[list[str], UtteranceProvenance]:
    """Generate utterances and return them together with their provenance record.

    Same behaviour and return list as `generate_diverse_utterances`; the extra
    return value is what the trainer persists so a run can be reproduced.

    When the trainer has installed an `UtteranceCache` (R6) and an entry matches this
    command's fingerprint at this seed, the LLM is not called at all — and neither is
    the PersonaHub download, which is why the cache lookup happens BEFORE the
    `datasets`-availability check. Reuse is what makes two runs at the same seed train
    on the same data; seeding alone provably does not (spec §11, M4).

    `completion_fn` and `persona_dataset_loader` exist so the whole path can be
    driven without a network call; production leaves them None. Both are named in the
    cache fingerprint, so an injected generator can never collide with the real one.
    """
    num_personas = _resolve_generation_count(
        num_personas,
        "SYNTHETIC_UTTERANCE_GEN_NUMOF_PERSONAS",
    )
    utterances_per_persona = _resolve_generation_count(
        utterances_per_persona,
        "SYNTHETIC_UTTERANCE_GEN_UTTERANCES_PER_PERSONA",
    )
    personas_per_batch = _resolve_generation_count(
        personas_per_batch,
        "SYNTHETIC_UTTERANCE_GEN_PERSONAS_PER_BATCH",
    )
    if seed is None:
        seed = get_training_seed()

    seed_utterances = list(seed_utterances)
    fallback_utterances = [command_name] + seed_utterances

    # Resolved before the cache lookup: the model is part of the fingerprint, because
    # the same seeds under a different model are different training data.
    model = fastworkflow.get_env_var("LLM_SYNDATA_GEN")

    provenance = UtteranceProvenance(
        command_name=str(command_name),
        seed=seed,
        generator_config={
            "num_personas": num_personas,
            "utterances_per_persona": utterances_per_persona,
            "personas_per_batch": personas_per_batch,
            "model": model,
            # Recorded HERE, before the cache is consulted, and never on the generation
            # path alone. A cache hit returns early, so a field set only after that point
            # is present on the run that generated and absent on the run that reused --
            # which makes two runs at the same seed produce different provenance and
            # defeats the byte-comparison R6 exists to make possible. Corpus-free by
            # construction: naming the source never loads or downloads it.
            "persona_source": active_persona_source_name(),
        },
        seed_utterance_count=len(seed_utterances),
    )
    # The command-name token and the hand-written seeds are authored, not sampled.
    for utterance in fallback_utterances:
        _attribute_utterance(provenance.utterance_personas, utterance, SEED_PERSONA_ID)

    cache = get_utterance_cache()
    fingerprint = (
        utterance_fingerprint(
            seed_utterances,
            command_name,
            num_personas,
            utterances_per_persona,
            personas_per_batch,
            model,
            completion_fn=completion_fn,
            persona_dataset_loader=persona_dataset_loader,
        )
        if cache is not None
        else None
    )

    if cache is not None and fingerprint is not None:
        if entry := cache.lookup(fingerprint, seed):
            logger.info(
                f"Reusing {len(entry.generated_utterances)} cached utterances for "
                f"command '{command_name}' (seed {seed}, variant "
                f"{fingerprint.variant_key})"
            )
            cached_utterances = _apply_cached_entry(provenance, entry)
            return _finalize(provenance, fallback_utterances, cached_utterances)

    # If datasets is not available, return only the seed utterances
    # An app-supplied persona set needs no corpus, so it needs neither the download nor the
    # optional `datasets` dependency.
    if (
        not _DATASETS_AVAILABLE
        and persona_dataset_loader is None
        and persona_source_needs_corpus()
    ):
        logger.warning(
            f"datasets package not available. Skipping synthetic utterance generation "
            f"for command '{command_name}'. Using only seed utterances."
        )
        provenance.fell_back = True
        provenance.fallback_reason = "datasets package not available"
        provenance.final_count = len(fallback_utterances)
        return fallback_utterances, provenance

    # Initialize LiteLLM with API key
    api_key = fastworkflow.get_env_var("LITELLM_API_KEY_SYNDATA_GEN")
    litellm.api_key = api_key

    # Choose personas. `resolve_personas` takes the loader rather than a loaded dataset, so
    # an app-supplied persona set never triggers the PersonaHub download. With no
    # personas.json and no SYNTHETIC_UTTERANCE_GEN_PERSONA_FILE this selects exactly what
    # `select_persona_indices` selects, which tests/test_personas.py pins against the real
    # function rather than asserting in prose.
    selection = resolve_personas(
        num_personas,
        seed,
        command_name,
        persona_dataset_loader or load_persona_dataset,
    )
    selected_personas = selection.personas
    # Ids, not row indices, since an app-supplied persona has no row. The generation loop
    # only ever calls str() on these, so the widening is safe.
    selected_indices = selection.persona_ids
    provenance.persona_ids = list(selection.persona_ids)

    all_utterances = generate_utterances_for_personas(
        seed_utterances,
        command_name,
        selected_personas,
        selected_indices,
        provenance,
        utterances_per_persona=utterances_per_persona,
        personas_per_batch=personas_per_batch,
        model=model,
        seed=seed,
        completion_fn=completion_fn,
        _max_retries=_max_retries,
        _retry_base_seconds=_retry_base_seconds,
    )

    # A fallen-back run is degraded data. Caching it would make F3 permanent: the
    # command would train on its truncated set on every subsequent run too, with no
    # rate limit in sight to explain why.
    if (
        cache is not None
        and fingerprint is not None
        and not provenance.fell_back
        and all_utterances
    ):
        cache.store(
            fingerprint,
            seed,
            all_utterances,
            utterance_personas=provenance.utterance_personas,
            persona_ids=provenance.persona_ids,
            persona_dataset_size=selection.pool_size,
        )

    return _finalize(provenance, fallback_utterances, all_utterances)


def _finalize(
    provenance: UtteranceProvenance,
    fallback_utterances: list[str],
    generated_utterances: list[str],
) -> tuple[list[str], UtteranceProvenance]:
    """Assemble one seed's generated data and its authored fallback rows.

    Historical note: this step once applied optional cross-seed aggregation. That
    mode was unreachable through the shipped trainer and is intentionally absent;
    reuse now returns only the exact seed entry selected by the cache lookup.
    """
    final_utterances = fallback_utterances + generated_utterances
    provenance.generated_count = len(generated_utterances)
    provenance.final_count = len(final_utterances)

    if provenance.fell_back:
        _announce_fallback(
            provenance.command_name, provenance.fallback_reason,
            provenance.final_count)

    return final_utterances, provenance


def generate_diverse_utterances(
    seed_utterances: List[str],
    command_name,
    num_personas: Optional[int] = None,
    utterances_per_persona: Optional[int] = None,
    personas_per_batch: Optional[int] = None,
    seed: Optional[int] = None
) -> list[str]:
    """Public API called from every workflow's command files. Signature is a contract.

    `seed` defaults to the configured `TRAINING_SEED`. Provenance is pushed into the
    recorder installed by the trainer rather than returned, so this signature and
    return type stay compatible with the user-authored call sites generated by
    `fastworkflow/build/command_file_template.py`.
    """
    utterances, provenance = generate_diverse_utterances_with_provenance(
        seed_utterances,
        command_name,
        num_personas=num_personas,
        utterances_per_persona=utterances_per_persona,
        personas_per_batch=personas_per_batch,
        seed=seed,
    )
    record_provenance(provenance)
    return utterances

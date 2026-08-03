"""Persist and reuse LLM-generated utterances across training runs (spec R6, D6).

Why this exists
---------------
`TRAINING_SEED` (R2) seeds `random`, `numpy`, `torch`, persona selection and the
train/test split. It cannot seed the LLM. Two runs of `examples/hello_world` at the
same seed, same code, same environment produced **0 of 5** commands with identical
utterance sets, differing in both text and row count (spec §11, M4). The training
DATA varies run to run, so seeding a shuffle over a different dataset does not make
a reproducible model. Determinism is therefore unreachable until generated
utterances are persisted and reused — which inverts D6's sequencing: R6 is a
prerequisite for R2, not a consumer of it.

Where the cache lives, and why
------------------------------
    <workflow>/___command_info/utterance_cache/<command-slug>.<variant-key>.json

Four constraints picked this location:

* **It must survive a training run.** Artifact *versions* are immutable snapshots
  (R4); the cache is mutable and is shared by every version, so it cannot live
  inside `versions/<id>/`. Automatic artifact retention must not delete the reuse
  that makes the next run reproducible.
* **It must travel with the workflow.** A cache keyed to a machine (`~/.cache/...`)
  would make "did this workflow train reproducibly?" a property of the laptop
  rather than of the workflow, and would not survive a container build.
* **It must not be committed by accident.** `___command_info*` is already in
  `.gitignore`, so a workflow author opts in deliberately if they ever want to.
* **It must be inspectable.** One JSON file per command variant, holding plain
  utterance text, so `cat` answers "what is this command actually trained on?".

Prune interaction: `_prune_stale_artifacts` in `train/__main__.py` skips names in
`artifact_versioning.RESERVED_TOPLEVEL_NAMES`, and `CACHE_DIRNAME` is listed there.
Nothing else in `___command_info` walks unknown top-level directories.

The cost of losing it is a slow, non-reproducible run — never a wrong one. That is
the safe direction, and it is why `rm -rf ___command_info` remains a legitimate (if
blunt) way to start over.

Two-part key
------------
An entry is addressed by **(variant key, seed)**:

* the **variant key** hashes everything that is not the seed — the exact seed
  utterance list, the command name, the three generation counts, the model string, a
  digest of the API base, the persona source, the completion backend, and a digest
  of the source text of the function that builds the prompt;
* the **seed** selects one entry inside that variant's file.

Historical note: this cache once exposed cross-seed aggregation, but that mode was
unreachable through the shipped trainer and is intentionally absent. Multiple seed
entries still share one file so each seed can be reused exactly without overwriting
another seed's generated data.

Conservative by construction
----------------------------
The failure mode that matters is a developer editing a command's seed utterances and
silently training on stale generated data. Every input above is in the key, so that
edit misses. The prompt-source digest is deliberately coarse: *any* edit to
`generate_utterances_for_personas`, including a comment, invalidates every entry.
Over-invalidation costs money; under-invalidation costs trust in the measurement,
and this project has already paid that bill once.

Never cached: a run that fell back (rate limit, missing `datasets`). Freezing
degraded data into a cache would make F3 permanent instead of transient.
"""

from __future__ import annotations

import contextlib
import hashlib
import inspect
import json
import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Callable, Optional

from pydantic import BaseModel, ValidationError

from fastworkflow.train.determinism import COMMAND_INFO_FOLDERNAME
from fastworkflow.utils.logging import logger

# Bumped by hand when the on-disk shape changes in a way older readers cannot
# interpret. A file with a different value is ignored (a miss), never migrated.
CACHE_FORMAT_VERSION: int = 1

CACHE_DIRNAME: str = "utterance_cache"
CACHE_README_FILENAME: str = "README.md"

MODE_REUSE: str = "reuse"
MODE_REGENERATE: str = "regenerate"

CACHE_MODES: frozenset[str] = frozenset({MODE_REUSE, MODE_REGENERATE})
DEFAULT_CACHE_MODE: str = MODE_REUSE

# Identifies the production persona source and completion backend in the key. Named
# constants rather than the callables' qualnames: `litellm.completion` is a decorated
# wrapper whose qualname moves between litellm releases, which would invalidate every
# cache on an unrelated dependency bump. A caller that injects its own loader or
# completion function DOES get keyed on that callable's qualified name, because then
# the substitution is the point and a stale hit across two different sources would be
# a wrong answer.
PRODUCTION_PERSONA_SOURCE: str = "proj-persona/PersonaHub#persona.jsonl"
PRODUCTION_COMPLETION_BACKEND: str = "litellm.completion"

# Recorded in place of a source digest where `inspect.getsource` cannot read the
# module (zipapp, frozen build). Reuse still works; it just stops noticing prompt
# edits, which is the best available behaviour when the source is not on disk.
UNAVAILABLE_SOURCE_DIGEST: str = "source-unavailable"

_CACHE_README = """\
# Generated utterance cache — DELETING THIS COSTS MONEY AND REPRODUCIBILITY

Each file here holds the synthetic utterances one command was trained on, as
produced by an LLM. They are kept so that two training runs with the trainer's fixed
seed train on the **same data**. Without them, seeding alone does not make
training reproducible: the generator is a live model, and two runs at the same seed
were measured producing different utterances *and* different row counts for every
command.

* One file per (command, generation-configuration) pair; the hex suffix is the
  fingerprint of that configuration.
* Inside, `entries` is keyed by the training seed, so generating at several seeds
  accumulates rather than overwrites.
* An entry is reused only when every fingerprint input matches — the seed utterance
  list, the command name, the persona/utterance counts, the model, and the text of
  the prompt-building code. Edit any of them and the command regenerates.
* Degraded runs (rate-limited, `datasets` missing) are never written here.

Force regeneration with `fastworkflow train ... --regenerate-utterances`.
"""

_source_digest_cache: dict[str, str] = {}
_source_digest_lock = threading.Lock()

_unknown_mode_warned: set[str] = set()


def _utc_now() -> str:
    """UTC timestamp, used only as human-facing metadata on an entry."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def source_digest(func: Callable) -> str:
    """Return a short digest of *func*'s source text.

    Used to put the generation prompt itself into the fingerprint: the prompt is the
    LLM's input, so a change to it changes what would be produced, and a cache that
    ignored it would hand a developer tuning the prompt their old results. Cached per
    function because `inspect.getsource` reads and parses the file.
    """
    key = f"{getattr(func, '__module__', '?')}.{getattr(func, '__qualname__', '?')}"
    with _source_digest_lock:
        if key in _source_digest_cache:
            return _source_digest_cache[key]
    try:
        source = inspect.getsource(func)
    except (OSError, TypeError) as exc:  # zipapp / frozen / builtin
        logger.debug(f"Cannot read source of {key} for the utterance cache key: {exc}")
        digest = UNAVAILABLE_SOURCE_DIGEST
    else:
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
    with _source_digest_lock:
        _source_digest_cache[key] = digest
    return digest


def callable_identity(func: Optional[Callable], production_label: str) -> str:
    """Name the backend behind *func* for the fingerprint.

    None means "the production default", which is labelled by a stable constant. An
    injected callable is named by its qualified name so two different injections
    cannot collide on one cache entry.

    A callable OBJECT is named by its class, because instances carry no
    `__qualname__` and `repr()` would embed a memory address — which would make
    every run a miss. The consequence a caller must know: two instances of the same
    callable class are one identity here. Injection is a testing and
    embedding affordance, so give behaviourally different backends different
    classes rather than different constructor arguments.
    """
    if func is None:
        return production_label
    if (qualname := getattr(func, "__qualname__", None)) is not None:
        module = getattr(func, "__module__", None) or "?"
        return f"{module}.{qualname}"
    cls = type(func)
    module = getattr(cls, "__module__", None) or "?"
    qualname = getattr(cls, "__qualname__", None) or cls.__name__
    return f"{module}.{qualname}"


def normalize_mode(mode: Optional[str]) -> str:
    """Coerce *mode* to a known mode, warning once per unrecognised value."""
    if mode is None:
        return DEFAULT_CACHE_MODE
    candidate = str(mode).strip().lower()
    if not candidate:
        return DEFAULT_CACHE_MODE
    if candidate in CACHE_MODES:
        return candidate
    if candidate not in _unknown_mode_warned:
        _unknown_mode_warned.add(candidate)
        logger.warning(
            f"Unknown utterance-cache mode {candidate!r}; expected one of "
            f"{sorted(CACHE_MODES)}. Falling back to {DEFAULT_CACHE_MODE!r}."
        )
    return DEFAULT_CACHE_MODE


def resolve_cache_mode() -> str:
    """Return the normal production mode.

    The CLI passes ``regenerate`` directly to the cache constructor when requested;
    cache policy is not workflow environment configuration.
    """
    return DEFAULT_CACHE_MODE


def slugify(command_name: str) -> str:
    """Filesystem-safe, human-recognisable stem for a command name.

    Only cosmetic: the variant key that follows it in the filename is what actually
    disambiguates, so two command names collapsing to the same slug is harmless.

    Public because the parameter-example cache (fix-czb) files its entries the same
    way. Sharing the function rather than copying it keeps the two caches' filenames
    readable in the same way, and there is nothing utterance-specific in here.
    """
    safe = "".join(
        ch if (ch.isalnum() or ch in {"-", "_", "."}) else "_"
        for ch in str(command_name)
    )
    safe = safe.strip("._") or "command"
    return safe[:80]


def atomic_write_model(path: str, payload: BaseModel) -> None:
    """Serialise *payload* to *path* via a same-directory temporary + replace.

    Public and model-agnostic because the parameter-example cache needs exactly this
    guarantee for a differently shaped payload, and duplicating an atomic-write
    routine is how one of the two copies ends up non-atomic.
    """
    directory = os.path.dirname(path)
    tmp = os.path.join(
        directory, f".{os.path.basename(path)}.tmp-{os.getpid()}-"
        f"{uuid.uuid4().hex[:8]}"
    )
    serialised = payload.model_dump()
    # Sorted keys so two runs that produce the same content produce the same
    # bytes; R2's acceptance test is byte comparison, and an unordered dict dump
    # would make it fail for no reason anyone could act on.
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(serialised, f, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except BaseException:
        # Leaving a `.tmp-*` behind would accumulate one file per failed write
        # and, being dot-prefixed, would never be noticed.
        with contextlib.suppress(OSError):
            os.remove(tmp)
        raise


class Fingerprint(BaseModel):
    """Identifies one generation configuration for one command.

    `variant_key` covers everything except the training seed; the seed selects an
    entry within the variant. `inputs` is stored alongside the entries so a human can
    see *why* two variants differ without recomputing anything.
    """

    command_name: str
    variant_key: str
    inputs: dict


def compute_fingerprint(
    *,
    command_name: str,
    seed_utterances: list[str],
    num_personas: Optional[int],
    utterances_per_persona: Optional[int],
    personas_per_batch: Optional[int],
    model: Optional[str],
    api_base: Optional[str] = None,
    persona_source: str = PRODUCTION_PERSONA_SOURCE,
    completion_backend: str = PRODUCTION_COMPLETION_BACKEND,
    generator_source_digest: str = UNAVAILABLE_SOURCE_DIGEST,
) -> Fingerprint:
    """Build the variant key for one command's generation configuration.

    Every argument is something that changes what the LLM would produce:

    * `seed_utterances` — injected verbatim into the prompt, and unioned word-wise
      into the "use varied phrasing based on these themes" bag. Order is significant
      because the verbatim block preserves it. **This is the input whose silent
      staleness would be worst**: editing seeds and reusing old generated text is
      worse than having no cache at all.
    * `command_name` — injected as "maintain intent consistency with command: X".
    * the three counts — how many utterances are asked for, and how personas are
      grouped into completions (a batch is jointly sampled, so regrouping changes the
      text).
    * `model` / `api_base` — a different model, or the same model name pointed at a
      different proxy, is a different generator.
    * `persona_source` / `completion_backend` — see `callable_identity`.
    * `generator_source_digest` — the prompt template itself.

    The API base is hashed rather than stored: it can name an internal host, and the
    fingerprint file is meant to be readable and shareable.
    """
    api_base_digest = (
        hashlib.sha256(str(api_base).encode("utf-8")).hexdigest()[:12]
        if api_base
        else ""
    )
    inputs = {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "command_name": str(command_name),
        "seed_utterances": [str(u) for u in seed_utterances],
        "num_personas": num_personas,
        "utterances_per_persona": utterances_per_persona,
        "personas_per_batch": personas_per_batch,
        "model": model,
        "api_base_digest": api_base_digest,
        "persona_source": persona_source,
        "completion_backend": completion_backend,
        "generator_source_digest": generator_source_digest,
    }
    canonical = json.dumps(inputs, sort_keys=True, separators=(",", ":"))
    variant_key = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]
    return Fingerprint(
        command_name=str(command_name), variant_key=variant_key, inputs=inputs
    )


class UtteranceCacheEntry(BaseModel):
    """One command's generated utterances at one seed.

    Holds ONLY the generated text. The `[command_name] + seed_utterances` prefix is
    reconstructed by the caller from live sources, so a cache hit can never resurrect
    a seed utterance that has since been deleted from the command file — and the
    prefix is the part the fingerprint already covers.
    """

    seed: int
    created_at: str = ""
    generated_utterances: list[str] = []
    utterance_personas: dict[str, str] = {}
    persona_ids: list[str] = []
    persona_dataset_size: Optional[int] = None

    def usable_utterances(self) -> list[str]:
        """Non-empty generated strings, in the order they were produced.

        Duplicates are deliberately KEPT. Generation does not de-duplicate, and a row
        that appears twice is two training rows; dropping them here would make a
        cached run train on fewer rows than the run that produced the entry, and R2's
        acceptance test is exactly that the two runs agree. The retired cross-seed
        aggregation path de-duplicated separately; exact-seed reuse must not.
        """
        return [
            utterance.strip()
            for utterance in self.generated_utterances
            if isinstance(utterance, str) and utterance.strip()
        ]

    def is_usable(self) -> bool:
        """A hit must never hand back an empty utterance list.

        An entry that survives JSON parsing but carries nothing usable is treated as
        a miss, so the caller regenerates instead of training a command on zero rows
        — the F3 outcome, reached through the cache instead of through a rate limit.
        """
        return bool(self.usable_utterances())


class UtteranceCacheFile(BaseModel):
    """On-disk shape: one variant, every seed generated for it."""

    cache_format_version: int = CACHE_FORMAT_VERSION
    command_name: str
    variant_key: str
    fingerprint_inputs: dict = {}
    entries: dict[str, UtteranceCacheEntry] = {}


class UtteranceCache:
    """Reads and writes the per-workflow utterance cache.

    Every method degrades to "regenerate" rather than raising: a cache is an
    optimisation plus a determinism aid, and it must never be the reason a training
    run fails. Corrupt files are counted and logged, not repaired — the next
    successful generation overwrites them.
    """

    def __init__(self, workflow_folderpath: str, mode: Optional[str] = None) -> None:
        self.workflow_folderpath = workflow_folderpath
        self.mode = normalize_mode(mode) if mode is not None else resolve_cache_mode()
        self._lock = threading.Lock()
        self._counters: dict[str, int] = {
            "hit": 0,
            "miss": 0,
            "stored": 0,
            "unreadable": 0,
            "write_failed": 0,
        }

    # -- configuration -------------------------------------------------

    @property
    def enabled(self) -> bool:
        """Always true: shipped cache policy is exact-seed reuse or regeneration."""
        return True

    @property
    def reads_enabled(self) -> bool:
        """`regenerate` still WRITES: it refreshes the cache rather than bypassing it."""
        return self.mode == MODE_REUSE

    # -- paths ---------------------------------------------------------

    @property
    def root(self) -> str:
        """`<workflow>/___command_info/utterance_cache`. Not created by reading."""
        return os.path.join(
            self.workflow_folderpath, COMMAND_INFO_FOLDERNAME, CACHE_DIRNAME
        )

    def entry_path(self, fingerprint: Fingerprint) -> str:
        """Path of the file holding every seed's entry for *fingerprint*."""
        stem = slugify(fingerprint.command_name)
        return os.path.join(self.root, f"{stem}.{fingerprint.variant_key}.json")

    # -- reading -------------------------------------------------------

    def _read_file(self, fingerprint: Fingerprint) -> Optional[UtteranceCacheFile]:
        """Load a variant file, or None when absent, unreadable, or the wrong shape."""
        path = self.entry_path(fingerprint)
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            cached = UtteranceCacheFile(**payload)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValidationError,
                TypeError) as exc:
            self._bump("unreadable")
            logger.warning(
                f"Ignoring unreadable utterance cache entry {path}: {exc}. "
                f"Utterances for '{fingerprint.command_name}' will be regenerated."
            )
            return None
        if cached.cache_format_version != CACHE_FORMAT_VERSION:
            logger.info(
                f"Utterance cache entry {path} is format v"
                f"{cached.cache_format_version}, this build reads v"
                f"{CACHE_FORMAT_VERSION}; regenerating."
            )
            return None
        if cached.variant_key != fingerprint.variant_key:
            # Only reachable if a file was renamed or hand-edited. The filename is
            # derived from the key, so disagreement means the content is not what the
            # name claims, and trusting it would be exactly the stale-reuse bug.
            logger.warning(
                f"Utterance cache entry {path} records variant key "
                f"{cached.variant_key!r} but is filed under "
                f"{fingerprint.variant_key!r}; ignoring it."
            )
            return None
        return cached

    def lookup(self, fingerprint: Fingerprint, seed: int) -> Optional[UtteranceCacheEntry]:
        """Return the entry for exactly (*fingerprint*, *seed*), or None."""
        if not self.reads_enabled:
            return None
        cached = self._read_file(fingerprint)
        if cached is None:
            self._bump("miss")
            return None
        entry = cached.entries.get(str(int(seed)))
        if entry is None or not entry.is_usable():
            self._bump("miss")
            return None
        self._bump("hit")
        return entry

    # Historical aggregation notes retained after deleting the unreachable mode:
    # The lowest contributing seed labels the union; the individual entries
    # remain on disk, so nothing is lost by picking one for the summary.

    # -- writing -------------------------------------------------------

    def store(
        self,
        fingerprint: Fingerprint,
        seed: int,
        generated_utterances: list[str],
        utterance_personas: Optional[dict[str, str]] = None,
        persona_ids: Optional[list[str]] = None,
        persona_dataset_size: Optional[int] = None,
    ) -> bool:
        """Write (or replace) the entry for (*fingerprint*, *seed*). Never raises.

        Refuses to store an empty set, so a degraded run cannot poison the cache.
        Concurrency: the variant file is re-read immediately before writing and the
        other seeds' entries are merged forward, then a uniquely named temporary in
        the same directory is `os.replace`d over the target. `os.replace` is atomic on
        POSIX and Windows, so a concurrent reader sees either the old file or the new
        one, never a half-written one. Two writers racing on the SAME variant can
        still lose one update — the loser regenerates next run, which is the safe
        direction — but neither can corrupt the file.
        """
        cleaned = [
            u.strip() for u in generated_utterances
            if isinstance(u, str) and u.strip()
        ]
        if not cleaned:
            logger.debug(
                f"Not caching an empty utterance set for "
                f"'{fingerprint.command_name}'."
            )
            return False

        # Attribution is stored only for the utterances actually being cached. The
        # caller's map also covers the seed utterances and the command-name token,
        # which the caller re-attributes itself on every run.
        cached_texts = set(cleaned)
        attribution = {
            str(text): str(persona_id)
            for text, persona_id in (utterance_personas or {}).items()
            if isinstance(text, str) and text in cached_texts
        }

        entry = UtteranceCacheEntry(
            seed=int(seed),
            created_at=_utc_now(),
            generated_utterances=cleaned,
            utterance_personas=attribution,
            persona_ids=[str(p) for p in (persona_ids or [])],
            persona_dataset_size=persona_dataset_size,
        )

        path = self.entry_path(fingerprint)
        try:
            with self._lock:
                self._ensure_root()
                existing = self._read_file(fingerprint)
                entries = dict(existing.entries) if existing is not None else {}
                entries[str(int(seed))] = entry
                payload = UtteranceCacheFile(
                    cache_format_version=CACHE_FORMAT_VERSION,
                    command_name=fingerprint.command_name,
                    variant_key=fingerprint.variant_key,
                    fingerprint_inputs=fingerprint.inputs,
                    entries=entries,
                )
                self._atomic_write(path, payload)
        except OSError as exc:
            self._bump("write_failed")
            logger.warning(
                f"Could not write the utterance cache entry for "
                f"'{fingerprint.command_name}' to {path}: {exc}. Training continues; "
                f"the next run will regenerate."
            )
            return False
        self._bump("stored")
        return True

    def _ensure_root(self) -> None:
        """Create the cache directory and drop the "this costs money" README."""
        os.makedirs(self.root, exist_ok=True)
        readme = os.path.join(self.root, CACHE_README_FILENAME)
        if not os.path.isfile(readme):
            with open(readme, "w", encoding="utf-8") as f:
                f.write(_CACHE_README)

    @staticmethod
    def _atomic_write(path: str, payload: UtteranceCacheFile) -> None:
        """Serialise *payload* to *path* via a same-directory temporary + replace."""
        atomic_write_model(path, payload)

    # -- reporting -----------------------------------------------------

    def _bump(self, counter: str) -> None:
        self._counters[counter] = self._counters.get(counter, 0) + 1

    @property
    def stats(self) -> dict[str, int]:
        """Counts of hits, misses, stores and failures for this run."""
        return dict(self._counters)

    def format_summary(self) -> str:
        """One line for the end of a training run.

        Deliberately NOT recorded in `training_provenance.json`: whether a set of
        utterances came from the cache or from the LLM is a property of the run, not
        of the data, and putting it in provenance would make two otherwise identical
        runs produce different provenance files — defeating the byte-comparison this
        cache exists to make possible.
        """
        stats = self.stats
        return (
            f"Utterance cache ({self.mode}) at {self.root}: "
            f"{stats['hit']} reused, {stats['miss']} generated, "
            f"{stats['stored']} written"
            + (f", {stats['unreadable']} unreadable" if stats["unreadable"] else "")
            + (f", {stats['write_failed']} write failures" if stats["write_failed"] else "")
        )


_active_cache_lock = threading.Lock()
_active_cache: Optional[UtteranceCache] = None


def set_utterance_cache(cache: Optional[UtteranceCache]) -> None:
    """Install (or clear, with None) the cache that generation consults.

    Same shape as `determinism.set_provenance_recorder`, and for the same reason:
    `generate_diverse_utterances` has a fixed public signature — it is called from
    user-authored command files in every workflow — so it cannot be handed a
    workflow path or a cache. The trainer installs one for the duration of a run.
    """
    global _active_cache
    with _active_cache_lock:
        _active_cache = cache


def get_utterance_cache() -> Optional[UtteranceCache]:
    """Return the installed cache, or None when nothing is caching.

    None is normal outside a training run — a workflow author calling
    `generate_diverse_utterances` directly gets the unchanged, uncached behaviour.
    """
    with _active_cache_lock:
        return _active_cache

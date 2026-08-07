"""Integration tests for two ways synthetic generation could lose a training run.

Both findings come from the `fix-k0i` adversarial review of the intent-training
pipeline, and both are about the *blast radius* of a local problem:

1. **fix-k0i.45** — `generate_utterances_for_personas` caught only
   `RETRYABLE_LLM_EXCEPTIONS`. A rejection that is a property of ONE command's prompt
   (its seed list overflows the context window; its wording trips a content filter)
   propagated and killed the whole run at that command. On a 160-command workflow the
   exception arrives hours in, and every command after it loses its generated
   utterances too. These tests pin that such a failure takes the R3 seed fallback for
   that one command — and, just as importantly, that an account- or
   configuration-scoped failure still aborts, because swallowing those per command
   would turn one bad model name into 160 silently degraded commands.

2. **fix-k0i.43** — the utterance cache's `generator_source_digest` covered the
   prompt-building and post-filter code but not the persona-SELECTION code. Editing
   `derived_seed`, `select_persona_indices`, `resolve_personas` or
   `PersonaSource.select` changed which personas write a command's utterances while
   leaving every existing cache entry fingerprint-valid, so a fresh run was served
   text written by personas it would no longer pick.

Per `.cursor/rules/testing_rules.mdc` these are integration tests: no Mock fixtures
and no patching of fastWorkflow internals. The real generation entry point is driven
with a locally defined completion backend and persona loader, so nothing here needs an
API key or the network, and the exceptions are real `litellm` exception instances
constructed locally rather than stand-ins.
"""

import os
from types import SimpleNamespace

import litellm
import pytest

import fastworkflow
from fastworkflow.train.determinism import COMMAND_INFO_FOLDERNAME, derived_seed
from fastworkflow.train.generate_synthetic import (
    CONTENT_SHAPED_LLM_EXCEPTIONS,
    _DIGESTED_GENERATION_SOURCES,
    _is_template_echo,
    generate_diverse_utterances_with_provenance,
    generate_utterances_for_personas,
    generation_source_digest,
    select_persona_indices,
    utterance_fingerprint,
)
from fastworkflow.train.personas import PersonaSource, resolve_personas
from fastworkflow.train.utterance_cache import (
    UtteranceCache,
    compute_fingerprint,
    get_utterance_cache,
    set_utterance_cache,
    source_digest,
)

COMMAND_NAME = "add_two_numbers"
SEED_UTTERANCES = ["add 2 and 3", "sum these numbers"]
MODEL = "mistral/mistral-small-latest"

LOCAL_PERSONAS = [
    {"persona": f"A person who is persona number {i}."} for i in range(64)
]


def _local_persona_dataset():
    """`len()` plus `[i]['persona']` is the whole interface generation uses."""
    return LOCAL_PERSONAS


@pytest.fixture(autouse=True)
def training_env():
    """Pin the model string so the fingerprint does not depend on the local env."""
    previous = dict(fastworkflow._env_vars)
    fastworkflow.init({
        **previous,
        "LLM_SYNDATA_GEN": MODEL,
        "LITELLM_API_KEY_SYNDATA_GEN": "test-key-not-used-by-the-injected-backend",
        "TRAINING_SEED": "42",
    })
    yield
    fastworkflow.init(previous)


@pytest.fixture(autouse=True)
def clean_cache_sink():
    """Guarantee the process-wide cache sink is restored after each test."""
    previous = get_utterance_cache()
    set_utterance_cache(None)
    yield
    set_utterance_cache(previous)


@pytest.fixture
def workflow_dir(tmp_path):
    path = tmp_path / "workflow"
    path.mkdir()
    return str(path)


def _llm_response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


class RaisingCompletion:
    """A completion backend that raises whatever the test hands it, and counts calls.

    One class rather than one per exception because the class name is part of the
    cache fingerprint; a second class would change the variant key and make two calls
    that should share an entry miss instead.
    """

    def __init__(self, exception_factory, succeed_before: int = 0) -> None:
        self._exception_factory = exception_factory
        self._succeed_before = succeed_before
        self.calls = 0

    def __call__(self, **kwargs):
        self.calls += 1
        if self.calls <= self._succeed_before:
            name = kwargs["messages"][0]["content"].split("[Persona_")[1].split("]")[0]
            return _llm_response(
                f"[Persona_{name}]\nkept from batch {self.calls} alpha\n"
                f"kept from batch {self.calls} beta\n"
            )
        raise self._exception_factory()


class ForbiddenCompletion:
    """Any call is a failure: used to prove a cache entry was (or was not) reused."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, **kwargs):
        self.calls += 1
        raise AssertionError(
            "the LLM was called even though a cache entry should have been reused"
        )


class StaleEntryCompletion:
    """Records that generation happened, and produces one usable utterance per batch."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, **kwargs):
        self.calls += 1
        name = kwargs["messages"][0]["content"].split("[Persona_")[1].split("]")[0]
        return _llm_response(f"[Persona_{name}]\nfreshly generated phrasing\n")


def _content_policy_violation():
    return litellm.exceptions.ContentPolicyViolationError(
        message="the prompt was rejected by the provider's content filter",
        model=MODEL,
        llm_provider="test",
    )


def _context_window_exceeded():
    return litellm.exceptions.ContextWindowExceededError(
        message="this model's maximum context length is 8192 tokens",
        model=MODEL,
        llm_provider="test",
    )


def _rejected_request():
    return litellm.exceptions.RejectedRequestError(
        message="blocked by a guardrail before the call was made",
        model=MODEL,
        llm_provider="test",
        request_data={},
    )


def _authentication_error():
    return litellm.exceptions.AuthenticationError(
        message="the API key is invalid", llm_provider="test", model=MODEL
    )


def _malformed_request():
    return litellm.exceptions.BadRequestError(
        message="unsupported parameter in the request body",
        model=MODEL,
        llm_provider="test",
    )


def _unknown_model():
    return litellm.exceptions.NotFoundError(
        message="model does not exist", model=MODEL, llm_provider="test"
    )


def _budget_exceeded():
    return litellm.exceptions.BudgetExceededError(current_cost=12.0, max_budget=10.0)


def _generate(**overrides):
    """Run the real generation path with local, network-free inputs."""
    kwargs = {
        "num_personas": 2,
        "utterances_per_persona": 2,
        "personas_per_batch": 1,
        "seed": 42,
        "persona_dataset_loader": _local_persona_dataset,
    }
    kwargs |= overrides
    seed_utterances = kwargs.pop("seed_utterances", SEED_UTTERANCES)
    command_name = kwargs.pop("command_name", COMMAND_NAME)
    return generate_diverse_utterances_with_provenance(
        seed_utterances, command_name, **kwargs
    )


# ---------------------------------------------------------------------------
# fix-k0i.45 — a per-command rejection must cost one command, not the run
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("factory", "expected_name"),
    [
        pytest.param(
            _content_policy_violation, "ContentPolicyViolationError", id="content-policy"
        ),
        pytest.param(
            _context_window_exceeded, "ContextWindowExceededError", id="context-window"
        ),
        pytest.param(_rejected_request, "RejectedRequestError", id="guardrail"),
    ],
)
def test_a_prompt_shaped_rejection_falls_back_instead_of_aborting_the_run(
    factory, expected_name
):
    """The difference between losing one command and losing a multi-hour run.

    The observable signal is the whole point: the call must RETURN the authored
    fallback rows, and the provenance must say it was degraded and why. A silent
    empty return would be finding F3 all over again.
    """
    backend = RaisingCompletion(factory)
    utterances, provenance = _generate(completion_fn=backend)

    assert utterances == [COMMAND_NAME] + SEED_UTTERANCES
    assert provenance.fell_back is True
    assert expected_name in provenance.fallback_reason
    assert provenance.generated_count == 0
    assert provenance.final_count == len(utterances)


@pytest.mark.parametrize(
    ("factory", "expected_name"),
    [
        pytest.param(
            _content_policy_violation, "ContentPolicyViolationError", id="content-policy"
        ),
        pytest.param(
            _context_window_exceeded, "ContextWindowExceededError", id="context-window"
        ),
    ],
)
def test_a_prompt_shaped_rejection_is_not_retried(factory, expected_name):
    """Retrying is pointless: the prompt is what was rejected and it does not change.

    Pinned because the cheapest wrong fix is to add these to
    `RETRYABLE_LLM_EXCEPTIONS`, which would spend a minute of exponential backoff per
    command re-sending a request the provider has already refused.
    """
    backend = RaisingCompletion(factory)
    _utterances, provenance = _generate(
        completion_fn=backend, _max_retries=5, _retry_base_seconds=0.0
    )
    assert backend.calls == 1
    assert provenance.fell_back is True


def test_the_rejection_reason_survives_into_the_degraded_training_banner(capsys):
    """A degraded command has to be visible in a tqdm-flooded log (F3's real cause)."""
    _utterances, provenance = _generate(
        completion_fn=RaisingCompletion(_content_policy_violation)
    )
    printed = capsys.readouterr().out

    assert "DEGRADED TRAINING DATA" in printed
    assert COMMAND_NAME in printed
    assert "ContentPolicyViolationError" in printed
    # The provider's own words are what tell a developer WHICH seed utterance to fix.
    assert "content filter" in provenance.fallback_reason


def test_a_late_prompt_shaped_rejection_keeps_the_earlier_batches(workflow_dir):
    """Same partial-batch policy a terminal rate limit gets, and the same no-cache rule.

    The command still contributes rows, so its label still enters the classifier; and
    the truncated set is never persisted, because freezing it would make one bad
    minute permanent.
    """
    cache = UtteranceCache(workflow_dir)
    set_utterance_cache(cache)

    backend = RaisingCompletion(_context_window_exceeded, succeed_before=1)
    utterances, provenance = _generate(num_personas=3, completion_fn=backend)

    assert provenance.fell_back is True
    assert "kept from batch 1 alpha" in utterances
    assert provenance.generated_count == 2
    assert cache.stats["stored"] == 0


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(_authentication_error, id="authentication"),
        pytest.param(_malformed_request, id="malformed-request"),
        pytest.param(_unknown_model, id="unknown-model"),
        pytest.param(_budget_exceeded, id="budget-exceeded"),
    ],
)
def test_account_and_configuration_failures_still_abort_the_run(factory):
    """These doom every command that would follow, so swallowing them is the worse bug.

    Without this, the fix for fix-k0i.45 would convert a mistyped model name or an
    expired key into 160 commands that each quietly trained on their seed utterances
    alone — a whole workflow of weak intent detection with no failure to point at.
    """
    with pytest.raises(type(factory())):
        _generate(completion_fn=RaisingCompletion(factory))


def test_the_content_shaped_set_excludes_the_base_request_error():
    """`BadRequestError` is the parent of all three, and must not be caught.

    A plain 400 means the request WE built is malformed. Catching the parent would
    make every configuration error look like a content rejection.
    """
    assert litellm.exceptions.BadRequestError not in CONTENT_SHAPED_LLM_EXCEPTIONS
    for excluded in (
        litellm.exceptions.AuthenticationError,
        litellm.exceptions.PermissionDeniedError,
        litellm.exceptions.NotFoundError,
    ):
        assert not issubclass(excluded, CONTENT_SHAPED_LLM_EXCEPTIONS)
    for included in CONTENT_SHAPED_LLM_EXCEPTIONS:
        assert issubclass(included, litellm.exceptions.BadRequestError)


def test_a_prompt_shaped_rejection_on_the_inner_loop_records_provenance_only():
    """`generate_utterances_for_personas` is also called directly (tests, embedders).

    It must degrade there too, and it must report the degradation through the
    provenance record rather than by raising, because its caller is what decides what
    to do about a degraded command.
    """
    from fastworkflow.train.determinism import UtteranceProvenance

    provenance = UtteranceProvenance(command_name=COMMAND_NAME, seed=42)
    generated = generate_utterances_for_personas(
        SEED_UTTERANCES,
        COMMAND_NAME,
        ["A person who returns things."],
        ["7"],
        provenance,
        utterances_per_persona=2,
        personas_per_batch=1,
        model=MODEL,
        seed=42,
        completion_fn=RaisingCompletion(_content_policy_violation),
    )
    assert generated == []
    assert provenance.fell_back is True
    assert "ContentPolicyViolationError" in provenance.fallback_reason


# ---------------------------------------------------------------------------
# fix-k0i.43 — the cache key must cover the persona-SELECTION code
# ---------------------------------------------------------------------------

def _fingerprint(**overrides):
    kwargs = {
        "seed_utterances": SEED_UTTERANCES,
        "command_name": COMMAND_NAME,
        "num_personas": 2,
        "utterances_per_persona": 2,
        "personas_per_batch": 1,
        "model": MODEL,
        "completion_fn": None,
        "persona_dataset_loader": None,
    }
    kwargs |= overrides
    return utterance_fingerprint(
        kwargs["seed_utterances"],
        kwargs["command_name"],
        kwargs["num_personas"],
        kwargs["utterances_per_persona"],
        kwargs["personas_per_batch"],
        kwargs["model"],
        completion_fn=kwargs["completion_fn"],
        persona_dataset_loader=kwargs["persona_dataset_loader"],
    )


@pytest.mark.parametrize(
    "selection_function",
    [
        pytest.param(derived_seed, id="derived_seed"),
        pytest.param(select_persona_indices, id="select_persona_indices"),
        pytest.param(resolve_personas, id="resolve_personas"),
        pytest.param(PersonaSource.select, id="PersonaSource.select"),
    ],
)
def test_persona_selection_code_is_inside_the_cache_fingerprint(selection_function):
    """Each of these decides WHICH personas write a command's utterances.

    Persona ids themselves are deliberately absent from the key — computing them
    would force the PersonaHub download the cache exists to avoid (decision D6) — so
    the selection code is the only thing that can notice a change in the draw.
    """
    digest = _fingerprint().inputs["generator_source_digest"]
    assert source_digest(selection_function) in digest


def test_the_prompt_and_filter_stay_inside_the_cache_fingerprint():
    """Regression guard: widening the digest must not drop what it already covered."""
    digest = _fingerprint().inputs["generator_source_digest"]
    assert source_digest(generate_utterances_for_personas) in digest
    assert source_digest(_is_template_echo) in digest


def test_the_digested_sources_do_not_collapse_onto_one_digest():
    """Six functions must contribute six digests, or the join is decorative."""
    digests = {source_digest(func) for func in _DIGESTED_GENERATION_SOURCES}
    assert len(digests) == len(_DIGESTED_GENERATION_SOURCES)
    assert generation_source_digest() == generation_source_digest()


def test_the_digest_names_the_function_it_came_from():
    """A stale-cache miss must be diagnosable without re-deriving six hashes by hand."""
    digest = _fingerprint().inputs["generator_source_digest"]
    for func in _DIGESTED_GENERATION_SOURCES:
        assert func.__qualname__ in digest


def test_an_entry_written_before_selection_code_was_digested_is_not_reused(
    workflow_dir,
):
    """The behaviour the widened digest buys, end to end.

    The stored entry is filed under the digest the PREVIOUS formula produced — prompt
    plus post-filter, no persona selection. A run that reused it would be training on
    utterances written by personas the current selection code may no longer pick,
    which is exactly the stale reuse fix-k0i.43 reports. So the real generation path
    must miss it and call the LLM.
    """
    cache = UtteranceCache(workflow_dir)
    stale_digest = "+".join((
        source_digest(generate_utterances_for_personas),
        source_digest(_is_template_echo),
    ))
    stale_fingerprint = compute_fingerprint(
        command_name=COMMAND_NAME,
        seed_utterances=SEED_UTTERANCES,
        num_personas=2,
        utterances_per_persona=2,
        personas_per_batch=1,
        model=MODEL,
        persona_source=(
            f"{_local_persona_dataset.__module__}.{_local_persona_dataset.__qualname__}"
        ),
        completion_backend=(
            f"{StaleEntryCompletion.__module__}.{StaleEntryCompletion.__qualname__}"
        ),
        generator_source_digest=stale_digest,
    )
    assert cache.store(stale_fingerprint, 42, ["written by yesterday's personas"])

    set_utterance_cache(cache)
    backend = StaleEntryCompletion()
    utterances, provenance = _generate(completion_fn=backend)

    assert backend.calls > 0, (
        "the pre-fix cache entry was reused, so an edit to persona selection would "
        "silently keep serving utterances written by personas a fresh run would not "
        "pick"
    )
    assert "written by yesterday's personas" not in utterances
    assert provenance.fell_back is False


def test_the_widened_digest_still_reuses_an_entry_it_wrote_itself(workflow_dir):
    """Over-invalidation costs money, so prove the key is still self-consistent."""
    cache = UtteranceCache(workflow_dir)
    set_utterance_cache(cache)

    first_backend = StaleEntryCompletion()
    first_utterances, _provenance = _generate(completion_fn=first_backend)
    assert first_backend.calls > 0
    assert cache.stats["stored"] == 1

    second_backend = ForbiddenCompletion()
    with pytest.raises(AssertionError, match="should have been reused"):
        # Different backend class, so a MISS is expected here for the fingerprint
        # reason; the shared-class reuse case is the assertion after this block.
        _generate(completion_fn=second_backend)

    reused_utterances, reused_provenance = _generate(
        completion_fn=StaleEntryCompletion()
    )
    assert reused_utterances == first_utterances
    assert reused_provenance.fell_back is False
    assert cache.stats["hit"] == 1
    assert not os.path.exists(
        os.path.join(workflow_dir, COMMAND_INFO_FOLDERNAME, "versions")
    )

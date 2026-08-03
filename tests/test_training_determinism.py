"""Integration tests for training determinism (R2) and generation robustness (R3a).

Covers the two findings in `docs/intent_training_improvements_spec.md`:

* **F2** — nothing was seeded, so persona selection was free-running and two
  identical training runs disagreed on 20.6% of held-out routing cases.
* **F3** — a rate-limited command returned `[]`, dropping even its hand-written
  seed utterances, so its label never entered the context's classifier.

Per `.cursor/rules/testing_rules.mdc` these are integration tests: no Mock
fixtures and no patching of fastWorkflow internals. The failure conditions are
INDUCED — real `litellm` exception objects raised by locally defined callables,
and a locally defined persona source — so nothing here needs an API key or the
network.
"""

import importlib.util
import json
import os
import random
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest

import litellm

import fastworkflow
from fastworkflow.command_context_model import CommandContextModel
from fastworkflow.command_directory import CommandDirectory
from fastworkflow.command_routing import RoutingDefinition, RoutingRegistry
from fastworkflow.model_pipeline_training import _get_cached_command_utterances
from fastworkflow.train.determinism import (
    ContextTrainingStatus,
    DEFAULT_TRAINING_SEED,
    PERSONA_ID_SEPARATOR,
    PROVENANCE_FILENAME,
    PROVENANCE_SCHEMA_VERSION,
    SEED_PERSONA_ID,
    UNRESOLVED_PERSONA_PREFIX,
    ProvenanceRecorder,
    UtteranceProvenance,
    derived_seed,
    get_provenance_recorder,
    get_training_seed,
    record_provenance,
    seed_everything,
    set_provenance_recorder,
)
from fastworkflow.train.generate_synthetic import (
    RETRYABLE_LLM_EXCEPTIONS,
    call_with_retries,
    generate_diverse_utterances_with_provenance,
    generate_utterances_for_personas,
    select_persona_indices,
)


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


COMMAND_NAME = "add_two_numbers"
SEED_UTTERANCES = ["add 2 and 3", "sum these numbers"]

# A stand-in for the PersonaHub dataset: `len()` plus `[i]['persona']` is the
# entire interface generate_synthetic uses. Real personas, no download.
LOCAL_PERSONAS = [
    {"persona": f"A person who is persona number {i}."} for i in range(64)
]


def _local_persona_dataset():
    return LOCAL_PERSONAS


def _rate_limit_error() -> litellm.exceptions.RateLimitError:
    """A real litellm RateLimitError, constructed locally rather than mocked."""
    return litellm.exceptions.RateLimitError(
        message="rate limited by the test", llm_provider="test", model="test/model"
    )


def _llm_response(content: str) -> SimpleNamespace:
    """The shape generate_synthetic reads off a completion result."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


@pytest.fixture
def fast_retry_options():
    """Use the retry helper's private test seam without exposing user configuration."""
    return {"_max_retries": 2, "_retry_base_seconds": 0.0}


@pytest.fixture
def clean_recorder():
    """Guarantee the process-wide recorder sink is restored after each test."""
    previous = get_provenance_recorder()
    set_provenance_recorder(None)
    yield
    set_provenance_recorder(previous)


# ---------------------------------------------------------------------------
# derived_seed
# ---------------------------------------------------------------------------

def test_derived_seed_is_stable_across_processes():
    """Two fresh interpreters must agree — proving builtin hash() is not used.

    `hash()` of a str is salted per process, so a derived seed built on it would
    differ here even though the inputs are identical.
    """
    script = textwrap.dedent(
        """
        from fastworkflow.train.determinism import derived_seed
        print(derived_seed(42, "AccessReview/bulk_decide"))
        print(derived_seed(42, "a", "b"))
        """
    )
    env = dict(os.environ)
    # Force a different hash salt in each child; only a hashlib-based
    # implementation survives this.
    outputs = []
    for salt in ("1", "2"):
        env["PYTHONHASHSEED"] = salt
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        outputs.append(completed.stdout.strip().splitlines())

    assert outputs[0] == outputs[1], (
        f"derived_seed differs between processes: {outputs}. "
        "This is the signature of Python's salted builtin hash()."
    )
    in_process = [
        str(derived_seed(42, "AccessReview/bulk_decide")),
        str(derived_seed(42, "a", "b")),
    ]
    assert outputs[0] == in_process


def test_derived_seed_varies_with_inputs():
    base = derived_seed(42, "cmd")
    assert derived_seed(42, "cmd") == base
    assert derived_seed(43, "cmd") != base
    assert derived_seed(42, "other_cmd") != base
    assert derived_seed(42, "cmd", "extra") != base
    assert derived_seed(42, "a", "b") != derived_seed(42, "b", "a")


def test_derived_seed_is_in_range_for_torch_and_numpy():
    for keys in (("a",), ("b", "c"), ("really/long/command_name",)):
        value = derived_seed(DEFAULT_TRAINING_SEED, *keys)
        assert 0 <= value < 2**31


# ---------------------------------------------------------------------------
# seed_everything / get_training_seed
# ---------------------------------------------------------------------------

def _sample_all_rngs():
    import numpy as np
    import torch

    return (
        [random.random() for _ in range(5)],
        np.random.rand(5).tolist(),
        torch.rand(5).tolist(),
    )


@pytest.mark.skipif(
    not (_module_available("numpy") and _module_available("torch")),
    reason="numpy and torch are required to verify seeding of all three RNGs.",
)
def test_seed_everything_makes_all_rngs_reproducible():
    assert seed_everything(1234) == 1234
    first = _sample_all_rngs()

    seed_everything(1234)
    second = _sample_all_rngs()
    assert first == second

    seed_everything(4321)
    third = _sample_all_rngs()
    assert first != third


def test_seed_everything_defaults_to_the_configured_seed():
    assert seed_everything() == get_training_seed()


def test_training_seed_is_fixed_even_when_env_file_requests_an_override():
    previous = dict(fastworkflow._env_vars)
    try:
        fastworkflow.init({k: v for k, v in previous.items() if k != "TRAINING_SEED"})
        fastworkflow.init({**previous, "TRAINING_SEED": "7"})
        assert get_training_seed() == DEFAULT_TRAINING_SEED
    finally:
        fastworkflow.init(previous)


def test_training_seed_ignores_shell_exports(monkeypatch):
    """Training seed is implementation policy, not process configuration."""
    previous = dict(fastworkflow._env_vars)
    try:
        fastworkflow.init({k: v for k, v in previous.items() if k != "TRAINING_SEED"})
        monkeypatch.setenv("TRAINING_SEED", "999")
        assert get_training_seed() == DEFAULT_TRAINING_SEED
    finally:
        fastworkflow.init(previous)


# ---------------------------------------------------------------------------
# Persona selection
# ---------------------------------------------------------------------------

def test_persona_selection_is_reproducible_for_the_same_seed():
    first = select_persona_indices(200_000, 8, 42, COMMAND_NAME)
    second = select_persona_indices(200_000, 8, 42, COMMAND_NAME)
    assert first == second
    assert len(first) == 8


def test_persona_selection_changes_with_seed_and_command():
    baseline = select_persona_indices(200_000, 8, 42, COMMAND_NAME)
    assert select_persona_indices(200_000, 8, 43, COMMAND_NAME) != baseline
    assert select_persona_indices(200_000, 8, 42, "some_other_command") != baseline


def test_persona_selection_is_immune_to_prior_global_random_use():
    """The regression that F2 is actually about.

    The old code called `random.sample` on the global module, so a command's
    personas depended on how many random draws the process had already made —
    i.e. on which contexts were trained before it.
    """
    baseline = select_persona_indices(200_000, 8, 42, COMMAND_NAME)

    random.seed(1)
    for _ in range(1000):
        random.random()
    after_churn = select_persona_indices(200_000, 8, 42, COMMAND_NAME)

    assert after_churn == baseline


def test_persona_selection_clamps_to_dataset_size():
    assert len(select_persona_indices(3, 10, 42, COMMAND_NAME)) == 3
    assert select_persona_indices(0, 10, 42, COMMAND_NAME) == []
    assert select_persona_indices(100, 0, 42, COMMAND_NAME) == []


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

def test_call_with_retries_recovers_from_transient_failures():
    attempts = {"count": 0}

    def flaky():
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise _rate_limit_error()
        return "generated"

    result = call_with_retries(
        flaky, description="test", max_retries=5, base_delay=0.0
    )
    assert result == "generated"
    assert attempts["count"] == 3


def test_call_with_retries_exhausts_budget_then_reraises():
    attempts = {"count": 0}

    def always_rate_limited():
        attempts["count"] += 1
        raise _rate_limit_error()

    with pytest.raises(litellm.exceptions.RateLimitError):
        call_with_retries(
            always_rate_limited, description="test", max_retries=3, base_delay=0.0
        )
    assert attempts["count"] == 4  # the initial call plus three retries


def test_call_with_retries_does_not_retry_configuration_errors():
    """An auth or bad-request failure is not transient; retrying just wastes time."""
    attempts = {"count": 0}

    def bad_request():
        attempts["count"] += 1
        raise litellm.exceptions.BadRequestError(
            message="bad model", model="test/model", llm_provider="test"
        )

    with pytest.raises(litellm.exceptions.BadRequestError):
        call_with_retries(
            bad_request, description="test", max_retries=5, base_delay=0.0
        )
    assert attempts["count"] == 1


def test_retryable_exceptions_exclude_the_generic_api_error():
    """APIError is the base of AuthenticationError/BadRequestError; never retry it."""
    assert litellm.exceptions.APIError not in RETRYABLE_LLM_EXCEPTIONS
    assert litellm.exceptions.RateLimitError in RETRYABLE_LLM_EXCEPTIONS


# ---------------------------------------------------------------------------
# Rate-limited generation must never return []
# ---------------------------------------------------------------------------

def test_rate_limited_command_falls_back_to_seed_utterances(fast_retry_options):
    def always_rate_limited(**_kwargs):
        raise _rate_limit_error()

    utterances, provenance = generate_diverse_utterances_with_provenance(
        SEED_UTTERANCES,
        COMMAND_NAME,
        num_personas=4,
        utterances_per_persona=3,
        personas_per_batch=1,
        seed=42,
        completion_fn=always_rate_limited,
        persona_dataset_loader=_local_persona_dataset,
        **fast_retry_options,
    )

    assert utterances, "a rate-limited command must never contribute zero rows"
    assert utterances == [COMMAND_NAME] + SEED_UTTERANCES
    assert provenance.fell_back is True
    assert "RateLimitError" in provenance.fallback_reason
    assert provenance.generated_count == 0
    assert provenance.final_count == len(utterances)
    assert provenance.seed_utterance_count == len(SEED_UTTERANCES)
    for utterance in utterances:
        assert provenance.utterance_personas[utterance] == SEED_PERSONA_ID


def test_partial_batch_failure_keeps_what_earlier_batches_produced(fast_retry_options):
    """Batches 1-2 succeed, batch 3 exhausts its retries: keep the first two."""
    attempted_batches = []

    def two_good_then_rate_limited(**kwargs):
        prompt = kwargs["messages"][0]["content"]
        index = int(prompt.split("[Persona_")[1].split("]")[0])
        attempted_batches.append(index)
        if index > 2:
            raise _rate_limit_error()
        return _llm_response(
            f"[Persona_{index}]\nutterance {index} alpha\nutterance {index} beta\n"
        )

    utterances, provenance = generate_diverse_utterances_with_provenance(
        SEED_UTTERANCES,
        COMMAND_NAME,
        num_personas=5,
        utterances_per_persona=2,
        personas_per_batch=1,
        seed=42,
        completion_fn=two_good_then_rate_limited,
        persona_dataset_loader=_local_persona_dataset,
        **fast_retry_options,
    )

    assert provenance.fell_back is True
    assert provenance.generated_count == 4
    assert utterances[: len(SEED_UTTERANCES) + 1] == [COMMAND_NAME] + SEED_UTTERANCES
    assert "utterance 1 alpha" in utterances
    assert "utterance 2 beta" in utterances
    # Batch 3 was tried three times (initial call plus two retries) and then the
    # run gave up: personas 4 and 5 were deliberately never attempted.
    assert attempted_batches == [1, 2, 3, 3, 3]


def test_successful_generation_records_persona_attribution(fast_retry_options):
    def echoing_completion(**kwargs):
        # Recover the persona header the prompt asked the model to echo back.
        prompt = kwargs["messages"][0]["content"]
        name = prompt.split("[Persona_")[1].split("]")[0]
        return _llm_response(
            f"[Persona_{name}]\nphrase one for {name}\nphrase two for {name}\n"
        )

    utterances, provenance = generate_diverse_utterances_with_provenance(
        SEED_UTTERANCES,
        COMMAND_NAME,
        num_personas=3,
        utterances_per_persona=2,
        personas_per_batch=1,
        seed=42,
        completion_fn=echoing_completion,
        persona_dataset_loader=_local_persona_dataset,
        **fast_retry_options,
    )

    assert provenance.fell_back is False
    assert len(provenance.persona_ids) == 3
    assert provenance.generated_count == 6
    assert provenance.final_count == len(utterances)

    for utterance in [COMMAND_NAME] + SEED_UTTERANCES:
        assert provenance.utterance_personas[utterance] == SEED_PERSONA_ID

    generated = [u for u in utterances if u.startswith("phrase ")]
    assert len(generated) == 6
    for utterance in generated:
        assert provenance.utterance_personas[utterance] in provenance.persona_ids


def test_unresolvable_persona_header_is_attributed_conservatively(fast_retry_options):
    """A hallucinated header must not be silently pinned to one persona."""
    def hallucinating_completion(**_kwargs):
        return _llm_response(
            "[Some_Invented_Name]\nfirst invented line\nsecond invented line\n"
        )

    _utterances, provenance = generate_diverse_utterances_with_provenance(
        SEED_UTTERANCES,
        COMMAND_NAME,
        num_personas=2,
        utterances_per_persona=2,
        personas_per_batch=2,
        seed=42,
        completion_fn=hallucinating_completion,
        persona_dataset_loader=_local_persona_dataset,
        **fast_retry_options,
    )

    attribution = provenance.utterance_personas["first invented line"]
    assert attribution.startswith(UNRESOLVED_PERSONA_PREFIX)
    for persona_id in provenance.persona_ids:
        assert persona_id in attribution


def test_duplicate_utterance_across_personas_gets_a_composite_id(fast_retry_options):
    """Whole-persona holdout would leak if one text belonged to two personas."""
    def duplicating_completion(**kwargs):
        prompt = kwargs["messages"][0]["content"]
        name = prompt.split("[Persona_")[1].split("]")[0]
        return _llm_response(
            f"[Persona_{name}]\nthe very same wording\nunique to {name}\n"
        )

    _utterances, provenance = generate_diverse_utterances_with_provenance(
        SEED_UTTERANCES,
        COMMAND_NAME,
        num_personas=2,
        utterances_per_persona=2,
        personas_per_batch=1,
        seed=42,
        completion_fn=duplicating_completion,
        persona_dataset_loader=_local_persona_dataset,
        **fast_retry_options,
    )

    composite = provenance.utterance_personas["the very same wording"]
    contributors = composite.split(PERSONA_ID_SEPARATOR)
    assert sorted(contributors) == sorted(provenance.persona_ids)


@pytest.mark.parametrize("unresolved_first", [False, True])
def test_resolved_unresolved_collision_keeps_normalized_union(
    fast_retry_options, unresolved_first
):
    """A mixed collision must retain every real contributor and no prefix atom."""
    def mixed_attribution_completion(**_kwargs):
        resolved = "[Persona_1]\nthe mixed collision wording\n"
        unresolved = "[Invented_Persona]\nthe mixed collision wording\n"
        content = unresolved + resolved if unresolved_first else resolved + unresolved
        return _llm_response(content)

    _utterances, provenance = generate_diverse_utterances_with_provenance(
        SEED_UTTERANCES,
        COMMAND_NAME,
        num_personas=2,
        utterances_per_persona=1,
        personas_per_batch=2,
        seed=42,
        completion_fn=mixed_attribution_completion,
        persona_dataset_loader=_local_persona_dataset,
        **fast_retry_options,
    )

    expected = (
        UNRESOLVED_PERSONA_PREFIX
        + PERSONA_ID_SEPARATOR.join(sorted(provenance.persona_ids))
    )
    attribution = provenance.utterance_personas["the mixed collision wording"]
    assert attribution == expected
    assert f"{PERSONA_ID_SEPARATOR}{UNRESOLVED_PERSONA_PREFIX}" not in attribution


def test_generated_duplicate_of_a_seed_stays_attributed_to_the_seed(fast_retry_options):
    """Seeds are authored ground truth and must never land in a held-out persona."""
    def seed_echoing_completion(**_kwargs):
        return _llm_response(f"[Persona_1]\n{SEED_UTTERANCES[0]}\nsomething else\n")

    _utterances, provenance = generate_diverse_utterances_with_provenance(
        SEED_UTTERANCES,
        COMMAND_NAME,
        num_personas=1,
        utterances_per_persona=2,
        personas_per_batch=1,
        seed=42,
        completion_fn=seed_echoing_completion,
        persona_dataset_loader=_local_persona_dataset,
        **fast_retry_options,
    )

    assert provenance.utterance_personas[SEED_UTTERANCES[0]] == SEED_PERSONA_ID


def test_generation_is_reproducible_for_the_same_seed(fast_retry_options):
    def echoing_completion(**kwargs):
        prompt = kwargs["messages"][0]["content"]
        name = prompt.split("[Persona_")[1].split("]")[0]
        return _llm_response(f"[Persona_{name}]\nline a for {name}\nline b for {name}\n")

    runs = []
    for _ in range(2):
        # Churn the global RNG between runs; a seeded run must not notice.
        random.random()
        runs.append(
            generate_diverse_utterances_with_provenance(
                SEED_UTTERANCES,
                COMMAND_NAME,
                num_personas=4,
                utterances_per_persona=2,
                personas_per_batch=2,
                seed=42,
                completion_fn=echoing_completion,
                persona_dataset_loader=_local_persona_dataset,
                **fast_retry_options,
            )
        )

    assert runs[0][0] == runs[1][0]
    assert runs[0][1].persona_ids == runs[1][1].persona_ids
    assert runs[0][1].utterance_personas == runs[1][1].utterance_personas


def test_generation_loop_returns_only_generated_utterances(fast_retry_options):
    """The extracted loop must not re-add seeds; its caller owns the assembly."""
    provenance = UtteranceProvenance(command_name=COMMAND_NAME, seed=42)

    def always_rate_limited(**_kwargs):
        raise _rate_limit_error()

    generated = generate_utterances_for_personas(
        SEED_UTTERANCES,
        COMMAND_NAME,
        ["persona one"],
        [11],
        provenance,
        utterances_per_persona=2,
        personas_per_batch=1,
        model="test/model",
        seed=42,
        completion_fn=always_rate_limited,
        **fast_retry_options,
    )
    assert generated == []
    assert provenance.fell_back is True


# ---------------------------------------------------------------------------
# ProvenanceRecorder
# ---------------------------------------------------------------------------


def test_inherited_command_generation_is_cached_by_fully_qualified_name(tmp_path):
    """A real inherited command is generated once, then reused in both contexts."""
    workflow_path = tmp_path / "multi_context_workflow"
    commands = workflow_path / "_commands"
    parent = commands / "Parent"
    parent.mkdir(parents=True)
    (commands / "context_inheritance_model.json").write_text(
        json.dumps({"Child": {"base": ["Parent"]}}),
        encoding="utf-8",
    )
    (parent / "shared_command.py").write_text(
        textwrap.dedent(
            """
            from pydantic import BaseModel

            class Signature:
                class Input(BaseModel):
                    value: str

                plain_utterances = ["shared seed"]

                @staticmethod
                def generate_utterances(workflow, command_name):
                    workflow.context["generation_calls"] = (
                        workflow.context.get("generation_calls", 0) + 1
                    )
                    return [command_name, *Signature.plain_utterances]
            """
        ),
        encoding="utf-8",
    )

    RoutingRegistry.clear_registry()
    context_model = CommandContextModel.load(str(workflow_path))
    command_directory = CommandDirectory.load(str(workflow_path))
    routing = RoutingDefinition.build(str(workflow_path))
    workflow = fastworkflow.Workflow.create(
        str(workflow_path), workflow_id_str="multi-context-provenance-test"
    )
    try:
        command_name = "Parent/shared_command"
        assert command_name in context_model.commands("Parent")
        assert command_name in context_model.commands("Child")
        assert command_name in routing.contexts["Parent"]
        assert command_name in routing.contexts["Child"]

        command_cache = {}
        parent_rows = _get_cached_command_utterances(
            workflow,
            str(workflow_path),
            command_directory,
            command_name,
            command_cache,
        )
        child_rows = _get_cached_command_utterances(
            workflow,
            str(workflow_path),
            command_directory,
            command_name,
            command_cache,
        )

        assert parent_rows == child_rows == [command_name, "shared seed"]
        assert workflow.context["generation_calls"] == 1
        assert list(command_cache) == [command_name]
    finally:
        workflow.close()
        RoutingRegistry.clear_registry()


def _example_provenance(command_name: str = COMMAND_NAME) -> UtteranceProvenance:
    return UtteranceProvenance(
        command_name=command_name,
        seed=42,
        persona_ids=["3", "17"],
        utterance_personas={
            command_name: SEED_PERSONA_ID,
            "add 2 and 3": SEED_PERSONA_ID,
            "please total these": "3",
            "what do these come to": "17",
        },
        generator_config={
            "num_personas": 2,
            "utterances_per_persona": 2,
            "personas_per_batch": 1,
            "model": "mistral/mistral-small-latest",
        },
        seed_utterance_count=1,
        generated_count=2,
        final_count=4,
    )


def test_provenance_recorder_round_trips(tmp_path):
    workflow = str(tmp_path / "workflow")
    os.makedirs(workflow)

    recorder = ProvenanceRecorder(workflow)
    recorder.record(_example_provenance())
    recorder.record(_example_provenance("delete_todo_list"))
    path = recorder.save()

    assert path == os.path.join(workflow, "___command_info", PROVENANCE_FILENAME)
    assert os.path.isfile(path)
    with open(path) as f:
        raw = json.load(f)
    assert raw["schema_version"] == PROVENANCE_SCHEMA_VERSION
    assert set(raw["commands"]) == {COMMAND_NAME, "delete_todo_list"}
    assert raw["context_training"] == {}

    loaded = ProvenanceRecorder.load(workflow)
    assert loaded == recorder.records


def test_one_generation_record_can_be_reused_in_multiple_contexts(tmp_path):
    recorder = ProvenanceRecorder(str(tmp_path))
    recorder.record(_example_provenance("TodoItem/assign_to"))
    recorder.record_context(
        context_name="TodoItem",
        command_name="TodoItem/assign_to",
        status=ContextTrainingStatus.INCLUDED,
        row_count=4,
    )
    recorder.record_context(
        context_name="TodoList",
        command_name="TodoItem/assign_to",
        status=ContextTrainingStatus.INCLUDED,
        row_count=4,
    )
    recorder.save()

    assert list(recorder.records) == ["TodoItem/assign_to"]
    loaded_contexts = ProvenanceRecorder.load_context_records(str(tmp_path))
    assert set(loaded_contexts) == {
        ("TodoItem", "TodoItem/assign_to"),
        ("TodoList", "TodoItem/assign_to"),
    }
    assert sum(record.row_count for record in loaded_contexts.values()) == 8


def test_schema_v2_save_is_byte_identical_for_the_same_records(tmp_path):
    recorder = ProvenanceRecorder(str(tmp_path))
    recorder.record(_example_provenance("TodoItem/assign_to"))
    for context_name in ("TodoList", "TodoItem"):
        recorder.record_context(
            context_name=context_name,
            command_name="TodoItem/assign_to",
            status=ContextTrainingStatus.INCLUDED,
            row_count=4,
        )

    path = recorder.save()
    with open(path, "rb") as file:
        first = file.read()
    recorder.save()
    with open(path, "rb") as file:
        second = file.read()

    assert first == second


def test_legacy_flat_provenance_remains_readable_without_rewrite(tmp_path):
    info = tmp_path / "___command_info"
    info.mkdir()
    path = info / PROVENANCE_FILENAME
    payload = {COMMAND_NAME: _example_provenance().model_dump()}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    before = path.read_bytes()

    assert ProvenanceRecorder.load(str(tmp_path))[COMMAND_NAME] == _example_provenance()
    assert ProvenanceRecorder.load_context_records(str(tmp_path)) == {}
    assert path.read_bytes() == before


def test_provenance_recorder_load_is_empty_when_nothing_was_saved(tmp_path):
    assert ProvenanceRecorder.load(str(tmp_path)) == {}


def test_recorder_does_not_let_a_later_success_hide_a_fallback(tmp_path):
    recorder = ProvenanceRecorder(str(tmp_path))

    degraded = _example_provenance()
    degraded.fell_back = True
    degraded.fallback_reason = "RateLimitError after 5 retries"
    recorder.record(degraded)
    recorder.record(_example_provenance())

    assert recorder.records[COMMAND_NAME].fell_back is True
    assert [r.command_name for r in recorder.fallback_summary()] == [COMMAND_NAME]


def test_recorder_sink_is_installable_and_clearable(tmp_path, clean_recorder):
    assert get_provenance_recorder() is None
    # No recorder installed: recording is a no-op, not an error.
    record_provenance(_example_provenance())

    recorder = ProvenanceRecorder(str(tmp_path))
    set_provenance_recorder(recorder)
    assert get_provenance_recorder() is recorder

    record_provenance(_example_provenance())
    assert COMMAND_NAME in recorder.records

    set_provenance_recorder(None)
    assert get_provenance_recorder() is None

"""Integration tests for two silent failures in the parameter-example pipeline.

Both come from the `fix-k0i` adversarial review, and both are about a signal that was
missing rather than a behaviour that was wrong:

* **fix-k0i.37** — (1) in `regenerate` mode `lookup()` returned before bumping its miss
  counter, and `format_summary` renders misses as "generated", so
  `fastworkflow train --regenerate-utterances` printed "0 reused, 0 generated, 12
  written" after twelve real LLM completions: a developer auditing token spend read
  zero calls. (2) A command whose generated examples were ALL rejected shipped an empty
  few-shot corpus — the runtime silently falls back to zero-shot parameter extraction
  for it — reported only by a `print` of "Validated 0 examples" among tens of thousands
  of progress-bar lines, indistinguishable from a healthy "13 of 15 accepted".

* **fix-k0i.46** — (1) `extract_field_details` accepted a STRINGIFIED `model_fields`,
  which is the type its caller declared. Under a string, `generate_dspy_examples`
  derived an empty `model_fields`, so every type check and every enum-membership check
  silently passed: the validation hole bd fix-b8h closed for dicts stayed open through
  the public signature. (2) `_canonical_digest` used `json.dumps(default=repr)`. Most
  objects inherit `object.__repr__`, whose text embeds `id()`, so any such value in a
  field-details record produced a different variant key every run — a permanent cache
  miss that is invisible by construction, because it looks exactly like a first run.

Per `.cursor/rules/testing_rules.mdc` these are integration tests: no Mock fixtures.
Real pydantic models, real files, the real `generate_dspy_examples`, and a locally
defined completion backend so nothing here needs an API key or the network. The
"observable signal" assertions read real `logging.LogRecord`s off the real
`fastWorkflow` logger through a stdlib handler, because the logger does not propagate
to root and so is invisible to `caplog`.
"""

import json
import logging
import os
import re
from decimal import Decimal
from enum import Enum
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, Field

import fastworkflow
from fastworkflow.train.param_example_cache import (
    ParamExampleCache,
    _canonical_digest,
    get_param_example_cache,
    set_param_example_cache,
)
from fastworkflow.train.utterance_cache import (
    MODE_REGENERATE,
    MODE_REUSE,
    UtteranceCache,
)
from fastworkflow.train.utterance_cache import (
    compute_fingerprint as utterance_compute_fingerprint,
)
from fastworkflow.utils.generate_param_examples import (
    extract_field_details,
    generate_dspy_examples,
    param_example_fingerprint,
)
from fastworkflow.utils.logging import logger as fastworkflow_logger

COMMAND_NAME = "add_two_numbers"
MODEL = "mistral/mistral-small-latest"


class AddUserInput(BaseModel):
    """A real parameter model of the shape `fastworkflow build` emits."""

    user_name: str = Field(
        json_schema_extra={"description": "who to add", "examples": ["bob"]}
    )
    is_premium_user: bool = Field(
        json_schema_extra={"description": "premium flag", "examples": ["True"]}
    )


class _OpaqueExample:
    """Inherits ``object.__repr__``, whose text embeds ``id(self)``.

    That is the whole defect in one class: `repr` of one of these is different in
    every process and for every instance, so digesting it produced a variant key that
    could never be hit again.
    """


class OpaqueExampleInput(BaseModel):
    """A parameter model whose field examples have no run-stable representation."""

    user_name: str = Field(
        json_schema_extra={"description": "who to add", "examples": [_OpaqueExample()]}
    )


class DeliveryMode(str, Enum):
    """A `str`-mixin enum, which is the shape shipped workflows actually use."""

    STANDARD = "standard"
    EXPRESS = "express"


class PlainMode(Enum):
    """A bare `Enum`, whose members JSON cannot encode without the hook."""

    STANDARD = "standard"
    EXPRESS = "express"


class OtherPlainMode(Enum):
    """Shares a VALUE with `PlainMode.EXPRESS` but is a different constraint."""

    EXPRESS = "express"


class _LogSink(logging.Handler):
    """Collects real `LogRecord`s off the real logger.

    The `fastWorkflow` logger sets ``propagate = False``, so pytest's `caplog`, whose
    handler lives on the root logger, never sees these records. Attaching a stdlib
    handler is not a mock: the records are the ones production emits.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def messages(self, level: int) -> list[str]:
        return [r.getMessage() for r in self.records if r.levelno == level]


@pytest.fixture
def log_sink():
    """Capture fastWorkflow log records for the duration of one test."""
    sink = _LogSink()
    previous_level = fastworkflow_logger.level
    fastworkflow_logger.setLevel(logging.DEBUG)
    fastworkflow_logger.addHandler(sink)
    try:
        yield sink
    finally:
        fastworkflow_logger.removeHandler(sink)
        fastworkflow_logger.setLevel(previous_level)


@pytest.fixture(autouse=True)
def training_env():
    """A fixed model string, so the fingerprint does not depend on the local env."""
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
    previous = get_param_example_cache()
    set_param_example_cache(None)
    yield
    set_param_example_cache(previous)


@pytest.fixture
def workflow_dir(tmp_path):
    path = tmp_path / "workflow"
    path.mkdir()
    return str(path)


def _llm_response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


class RecordingParamCompletion:
    """One injected backend, because the class name is part of the cache fingerprint."""

    def __init__(self, tag: str = "alpha", content: str = None) -> None:
        self.tag = tag
        self.content = content
        self.calls = 0

    def __call__(self, **kwargs):
        self.calls += 1
        if self.content is not None:
            return _llm_response(self.content)
        return _llm_response(
            "\n".join(
                f'dspy.Example(command="{self.tag} add user bob{i}", '
                f'user_name="bob{i}", is_premium_user=True).with_inputs("command")'
                for i in range(3)
            )
        )


def _generate(**overrides):
    """Run the real generation path with local, network-free inputs."""
    kwargs = {
        "field_annotations": AddUserInput.model_fields,
        "command_name": "add_user",
        "num_examples": 3,
        "validation_threshold": 0.3,
        "seed": 42,
        "completion_fn": RecordingParamCompletion(),
    }
    kwargs |= overrides
    return generate_dspy_examples(**kwargs)


# ---------------------------------------------------------------------------
# fix-k0i.37 (1) — the regenerate summary must not report zero LLM calls
# ---------------------------------------------------------------------------

def test_regenerate_mode_counts_the_llm_calls_it_forced(workflow_dir):
    """The reported symptom, as a number.

    In `regenerate` mode reads are disabled, so `lookup` returns early. Returning
    before the counter ran left the run summary claiming nothing was generated after
    every command had hit the LLM.
    """
    seeded = ParamExampleCache(workflow_dir, mode=MODE_REUSE)
    set_param_example_cache(seeded)
    _generate(completion_fn=RecordingParamCompletion(tag="alpha"))

    regenerating = ParamExampleCache(workflow_dir, mode=MODE_REGENERATE)
    set_param_example_cache(regenerating)
    backend = RecordingParamCompletion(tag="beta")
    _generate(completion_fn=backend)

    assert backend.calls == 1
    assert regenerating.stats["hit"] == 0
    assert regenerating.stats["miss"] == 1
    assert regenerating.stats["stored"] == 1


def test_the_regenerate_summary_states_what_was_generated(workflow_dir):
    """The line a developer auditing token spend actually reads."""
    cache = ParamExampleCache(workflow_dir, mode=MODE_REGENERATE)
    set_param_example_cache(cache)
    for index in range(3):
        _generate(
            command_name=f"add_user_{index}",
            completion_fn=RecordingParamCompletion(tag="alpha"),
        )

    summary = cache.format_summary()
    assert "0 reused" in summary
    assert "3 generated" in summary
    assert "3 written" in summary
    assert "0 generated" not in summary


def test_the_utterance_cache_reports_regenerate_misses_too(workflow_dir):
    """The same defect is in the sibling cache, in the same shape.

    Both summaries are printed by the same training run, so one of them telling the
    truth about token spend and the other not would be worse than neither.
    """
    fingerprint = utterance_compute_fingerprint(
        command_name=COMMAND_NAME,
        seed_utterances=["add 2 and 3"],
        num_personas=2,
        utterances_per_persona=2,
        personas_per_batch=1,
        model=MODEL,
    )
    seeded = UtteranceCache(workflow_dir, mode=MODE_REUSE)
    assert seeded.store(fingerprint, 42, ["the stale draw"])

    regenerating = UtteranceCache(workflow_dir, mode=MODE_REGENERATE)
    assert regenerating.lookup(fingerprint, 42) is None
    assert regenerating.stats["miss"] == 1

    regenerating.store(fingerprint, 42, ["the fresh draw"])
    summary = regenerating.format_summary()
    assert "1 generated" in summary
    assert "0 generated" not in summary


def test_reuse_mode_accounting_is_unchanged(workflow_dir):
    """The fix must not start double-counting the mode that was already correct."""
    cache = ParamExampleCache(workflow_dir, mode=MODE_REUSE)
    set_param_example_cache(cache)
    _generate(completion_fn=RecordingParamCompletion(tag="alpha"))
    _generate(completion_fn=RecordingParamCompletion(tag="alpha"))

    assert cache.stats["miss"] == 1
    assert cache.stats["hit"] == 1
    assert "1 reused" in cache.format_summary()
    assert "1 generated" in cache.format_summary()


# ---------------------------------------------------------------------------
# fix-k0i.37 (2) — a command with zero accepted examples must say so
# ---------------------------------------------------------------------------

_ALL_REJECTED = (
    'dspy.Example(command="add user alice", user_name="mallory", '
    'is_premium_user=True).with_inputs("command")'
)

_ONE_GOOD_ONE_BAD = "\n".join((
    'dspy.Example(command="add user alice", user_name="alice", '
    'is_premium_user=True).with_inputs("command")',
    _ALL_REJECTED,
))


def test_a_command_with_no_accepted_examples_warns_by_name(log_sink, tmp_path):
    """This command ships an empty few-shot corpus and extracts zero-shot at runtime.

    Nothing else in the run distinguishes it from a command that accepted 13 of 15, so
    the warning has to name the command and has to be at WARNING — an `info` line is
    invisible in a tqdm-flooded training log, which is exactly how finding F3 survived.
    """
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        valid, rejected = _generate(
            num_examples=1,
            completion_fn=RecordingParamCompletion(content=_ALL_REJECTED),
        )
    finally:
        os.chdir(original_cwd)

    assert valid == []
    assert len(rejected) == 1

    warnings = log_sink.messages(logging.WARNING)
    assert len(warnings) == 1, warnings
    assert "add_user" in warnings[0]
    assert "ZERO-SHOT" in warnings[0]
    assert "NO usable DSPy parameter examples" in warnings[0]


def test_a_partially_rejected_command_does_not_warn(log_sink, tmp_path):
    """The boundary that makes the warning worth reading.

    Some rejection is normal and healthy. Warning on every partial rejection would
    train a developer to ignore the line, and then the all-rejected case is silent
    again.
    """
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        valid, rejected = _generate(
            num_examples=2,
            completion_fn=RecordingParamCompletion(content=_ONE_GOOD_ONE_BAD),
        )
    finally:
        os.chdir(original_cwd)

    assert len(valid) == 1
    assert len(rejected) == 1
    assert log_sink.messages(logging.WARNING) == []


def test_an_unparsable_response_also_warns(log_sink, tmp_path):
    """Zero accepted is zero accepted, whether the model refused or was truncated."""
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        valid, rejected = _generate(
            completion_fn=RecordingParamCompletion(
                content="I'm sorry, I can't help with that request."
            ),
        )
    finally:
        os.chdir(original_cwd)

    assert valid == []
    assert rejected == []
    warnings = log_sink.messages(logging.WARNING)
    assert len(warnings) == 1, warnings
    assert "add_user" in warnings[0]


def test_the_all_rejected_warning_is_absent_on_a_cache_hit(log_sink, workflow_dir):
    """A reused entry is by construction non-empty, so there is nothing to warn about.

    Warning again on every subsequent run would report a degradation that is no longer
    happening.
    """
    cache = ParamExampleCache(workflow_dir, mode=MODE_REUSE)
    set_param_example_cache(cache)
    _generate(completion_fn=RecordingParamCompletion(tag="alpha"))
    log_sink.records.clear()

    reused, _rejected = _generate(completion_fn=RecordingParamCompletion(tag="alpha"))
    assert reused
    assert cache.stats["hit"] == 1
    assert log_sink.messages(logging.WARNING) == []


# ---------------------------------------------------------------------------
# fix-k0i.46 (1) — the stringified-annotations fail-open path is gone
# ---------------------------------------------------------------------------

def test_a_stringified_model_fields_is_rejected_rather_than_parsed():
    """The declared parameter type was `str`, and a string disabled all validation."""
    with pytest.raises(TypeError, match="model_fields mapping"):
        extract_field_details(str(AddUserInput.model_fields))


@pytest.mark.parametrize(
    "annotations",
    [
        pytest.param(str(AddUserInput.model_fields), id="stringified-model-fields"),
        pytest.param("user_name: str\nis_premium_user: bool", id="hand-written-string"),
        pytest.param(None, id="none"),
        pytest.param(["user_name"], id="list"),
    ],
)
def test_the_generator_refuses_annotations_it_cannot_validate_against(annotations):
    """Failing at build time beats shipping examples that were never validated.

    With a non-mapping, `generate_dspy_examples` derived `model_fields = {}` and every
    malformed-value check below it became a no-op that reported success.
    """
    with pytest.raises(TypeError, match="model_fields mapping"):
        _generate(field_annotations=annotations, num_examples=1)


def test_a_type_violating_value_is_still_rejected_when_annotations_are_a_mapping(
    tmp_path,
):
    """The check that a string used to switch off, exercised on the supported input."""
    source = (
        'dspy.Example(command="add user alice yes", user_name="alice", '
        'is_premium_user="yes").with_inputs("command")'
    )
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        valid, rejected = _generate(
            num_examples=1, completion_fn=RecordingParamCompletion(content=source)
        )
    finally:
        os.chdir(original_cwd)

    assert valid == []
    assert rejected[0]["reason"] == "Malformed parameter values"
    assert rejected[0]["invalid_params"][0]["param"] == "is_premium_user"


# ---------------------------------------------------------------------------
# fix-k0i.46 (2) — the variant key must be the same tomorrow
# ---------------------------------------------------------------------------

def test_two_instances_of_an_opaque_class_really_do_repr_differently():
    """Guard the premise: without this, the digest tests below prove nothing."""
    assert repr(_OpaqueExample()) != repr(_OpaqueExample())
    assert "0x" in repr(_OpaqueExample())


def test_an_address_bearing_repr_can_no_longer_reach_the_variant_key():
    """`default=repr` made the key a function of a memory address.

    The consequence was not a wrong answer, it was a permanent miss that looked
    identical to a first run: one LLM call per command per training run, forever, with
    nothing anywhere to point at.
    """
    with pytest.raises(TypeError, match="no run-stable representation"):
        _canonical_digest([{"name": "user_name", "examples": [_OpaqueExample()]}])


def test_the_raise_names_the_offending_type_and_where_to_fix_it():
    """A digest that cannot be computed must say what stopped it."""
    with pytest.raises(TypeError) as excinfo:
        _canonical_digest({"examples": [_OpaqueExample()]})
    message = str(excinfo.value)
    assert "_OpaqueExample" in message
    assert "_canonicalize_unserialisable" in message


@pytest.mark.parametrize(
    "make_payload",
    [
        pytest.param(lambda: {"enum": [DeliveryMode.EXPRESS]}, id="str-mixin-enum"),
        pytest.param(lambda: {"enum": [PlainMode.EXPRESS]}, id="plain-enum"),
        pytest.param(lambda: {"pattern": re.compile(r"^#W\d+$")}, id="compiled-pattern"),
        pytest.param(lambda: {"type": str}, id="type-object"),
        pytest.param(lambda: {"enum": {"a", "b", "c"}}, id="set-of-strings"),
        pytest.param(lambda: {"enum": {PlainMode.EXPRESS}}, id="set-of-enums"),
        pytest.param(lambda: {"examples": [b"raw"]}, id="bytes"),
        pytest.param(lambda: {"examples": [Decimal("1.10")]}, id="decimal"),
    ],
)
def test_the_types_a_parameter_model_can_realistically_hold_digest_stably(make_payload):
    """Two structurally identical field details must digest identically.

    Constructed twice on purpose: an identity-based representation would pass a
    single-object comparison and still miss on the next run.
    """
    assert _canonical_digest(make_payload()) == _canonical_digest(make_payload())


def test_a_set_digests_the_same_however_it_was_iterated():
    """Set iteration order for strings varies with the per-process hash salt.

    Digesting it in iteration order would make the variant key a function of the
    process, which is the same permanent-miss failure by another route.
    """
    assert _canonical_digest({"enum": {"a", "b", "c"}}) == _canonical_digest(
        {"enum": {"c", "b", "a"}}
    )


def test_plain_enums_that_share_a_value_are_not_the_same_constraint():
    """Digesting an enum by its value alone would let two constraints share an entry.

    A stale hit is the one direction this cache may never go: it hands a command the
    few-shot examples of a different parameter model. A `str`-mixin enum member is a
    `str`, so JSON encodes it as its value and `field_annotations_digest` is what keeps
    those apart; a bare `Enum` reaches the hook and is labelled by its class here.
    """
    assert _canonical_digest({"enum": [PlainMode.EXPRESS]}) != _canonical_digest(
        {"enum": [OtherPlainMode.EXPRESS]}
    )
    assert _canonical_digest({"enum": [PlainMode.EXPRESS]}) != _canonical_digest(
        {"enum": ["express"]}
    )


def test_a_json_only_payload_digests_exactly_as_it_did_before():
    """Existing caches must survive: the hook must never fire on ordinary field details.

    Every field detail a shipped example workflow produces is plain JSON, so making
    the encoder strict must not change a single existing variant key.
    """
    payload = extract_field_details(AddUserInput.model_fields)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    assert json.loads(canonical) == json.loads(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=repr)
    )
    # The digest is computable without the hook at all, which is the actual guarantee.
    assert _canonical_digest(payload)


def test_an_undigestable_field_detail_degrades_to_uncached_and_says_so(log_sink):
    """A cache must never be the reason a training run fails.

    So the raise is caught at the fingerprint boundary and turned into "this command
    is not cacheable this run", named and at WARNING — the opposite of the silent
    permanent miss it replaces.
    """
    fingerprint = param_example_fingerprint(
        OpaqueExampleInput.model_fields,
        "add_user",
        3,
        0.3,
        MODEL,
    )
    assert fingerprint is None

    warnings = log_sink.messages(logging.WARNING)
    assert len(warnings) == 1, warnings
    assert "add_user" in warnings[0]
    assert "Cannot compute a parameter-example cache key" in warnings[0]
    assert "_OpaqueExample" in warnings[0]


def test_an_undigestable_command_still_trains_just_without_reuse(workflow_dir):
    """End to end: the run completes, returns examples, and writes no cache entry."""
    cache = ParamExampleCache(workflow_dir, mode=MODE_REUSE)
    set_param_example_cache(cache)

    source = (
        'dspy.Example(command="add user alice", user_name="alice")'
        '.with_inputs("command")'
    )
    valid, _rejected = _generate(
        field_annotations=OpaqueExampleInput.model_fields,
        num_examples=1,
        completion_fn=RecordingParamCompletion(content=source),
    )
    assert valid
    assert cache.stats["stored"] == 0
    assert cache.stats["hit"] == 0
    assert not os.path.exists(cache.root)

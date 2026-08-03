"""Integration tests for the DSPy parameter-example cache (bd fix-czb).

The finding these cover: `generate_dspy_examples` is the SECOND unseeded LLM call in
training. R6 pinned the utterance path and left this one drawing fresh examples at
temperature 0.9 on every run, so `<command>_param_labeled.json` differed between two
runs at the same `TRAINING_SEED` and R2's "identical artifacts" claim stayed false.
Measured on `examples/hello_world` before this landed: **0 of 30** generated example
utterances were shared between two consecutive runs at seed 42.

The hazard that reuse introduces is the mirror image, and most of what follows is
about it: training on stale examples after a developer changes a command's parameter
model, which would leave the runtime few-shot prompting with examples that cannot
mention a field that now exists.

Per `.cursor/rules/testing_rules.mdc` these are integration tests: no Mock fixtures
and no patching of fastWorkflow internals. Real pydantic models, real files, real
`generate_dspy_examples`, with a locally defined completion backend so nothing here
needs an API key or the network. The end-to-end "two training runs agree" test lives
in `test_param_example_determinism.py`, which does need a key and skips without one.

One structural note, inherited from `test_utterance_cache.py`: the fingerprint records
the completion backend by qualified name precisely so an injected generator can never
share an entry with the real one. Any two calls here that are SUPPOSED to share an
entry must therefore go through the same `RecordingParamCompletion` class — hence one
backend class with a `forbidden` switch rather than a separate "must not be called"
class.
"""

import inspect
import json
import os
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import dspy
import pytest
from pydantic import BaseModel, Field

import fastworkflow
from fastworkflow.examples.retail_workflow._commands.cancel_pending_order import (
    Signature as CancelPendingOrderSignature,
)
from fastworkflow.examples.retail_workflow._commands.modify_pending_order_items import (
    Signature as ModifyPendingOrderItemsSignature,
)
from fastworkflow.train import artifact_versioning
from fastworkflow.train.determinism import COMMAND_INFO_FOLDERNAME
from fastworkflow.train.param_example_cache import (
    CACHE_DIRNAME,
    CACHE_FORMAT_VERSION,
    CACHE_README_FILENAME,
    ParamExampleCache,
    ParamExampleCacheEntry,
    get_param_example_cache,
    resolve_cache_mode,
    set_param_example_cache,
)
from fastworkflow.train.utterance_cache import (
    DEFAULT_CACHE_MODE,
    MODE_REGENERATE,
    MODE_REUSE,
    PRODUCTION_COMPLETION_BACKEND,
    callable_identity,
    source_digest,
)
from fastworkflow.utils import generate_param_examples
from fastworkflow.utils.generate_param_examples import (
    DSPY_EXAMPLE_TEMPERATURE,
    _UNDIGESTED_FUNCTIONS,
    _digested_functions,
    canonicalized,
    extract_field_details,
    generate_dspy_examples,
    param_example_fingerprint,
    prompt_source_digest,
    transform_examples_to_dict_format,
    validate_parameters,
)


COMMAND_NAME = "add_two_numbers"
MODEL = "mistral/mistral-small-latest"


class AddTwoNumbersInput(BaseModel):
    """A real parameter model, shaped like the ones `fastworkflow build` emits."""

    first_number: float = Field(
        json_schema_extra={
            "description": "the first addend",
            "examples": ["5", "3.2"],
        }
    )
    second_number: float = Field(
        json_schema_extra={
            "description": "the second addend",
            "examples": ["7", "1.8"],
        }
    )


class AddTwoNumbersInputWithComment(BaseModel):
    """`AddTwoNumbersInput` after a developer adds a field — the staleness hazard."""

    first_number: float = Field(
        json_schema_extra={
            "description": "the first addend",
            "examples": ["5", "3.2"],
        }
    )
    second_number: float = Field(
        json_schema_extra={
            "description": "the second addend",
            "examples": ["7", "1.8"],
        }
    )
    comment: str = Field(
        json_schema_extra={"description": "a note to attach", "examples": ["for tax"]}
    )


class AddUserInput(BaseModel):
    """A model whose examples do NOT come out in alphabetical field order.

    `transform_examples_to_dict_format` inserts `command`, then string fields, then
    booleans — so this yields `command, user_name, is_premium_user`, which is the
    ordering that exposed the cache round-trip bug. `AddTwoNumbersInput` cannot: its
    insertion order happens to be alphabetical already.
    """

    user_name: str = Field(
        json_schema_extra={"description": "who to add", "examples": ["bob"]}
    )
    is_premium_user: bool = Field(
        json_schema_extra={"description": "premium flag", "examples": ["True"]}
    )


class MixedFieldTypesInput(BaseModel):
    """Pins optional unwrapping without flattening unrelated generic types."""

    nickname: str | None = None
    tags: list[str]


class DeliveryMode(str, Enum):
    """A representative enum value shape for generated parameter examples."""

    STANDARD = "standard"
    EXPRESS = "express"


class StructuredFieldTypesInput(BaseModel):
    """The structured field families found while inventorying workflow Inputs."""

    item_ids: list[str]
    reference: str | int
    mode: DeliveryMode
    reason: str = Field(
        json_schema_extra={"enum": ["no longer needed", "ordered by mistake"]}
    )
    note: str | None = None


class LiteralOneInput(BaseModel):
    """A literal whose Python-equal values have different runtime types."""

    choice: Literal[1]


_UNSORTED_FIELD_EXAMPLES = "\n".join(
    f'dspy.Example(command="add user bob{i}", user_name="bob{i}", '
    f'is_premium_user=True).with_inputs("command")'
    for i in range(3)
)


def _llm_response(content: str) -> SimpleNamespace:
    """The shape `generate_dspy_examples` reads off a completion result."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def _examples_block(tag: str, count: int = 3) -> str:
    """`count` syntactically valid `dspy.Example` lines, tagged so drafts differ."""
    return "\n".join(
        f'dspy.Example(command="{tag} add {i} and {i + 1}", '
        f'first_number={i}, second_number={i + 1}).with_inputs("command")'
        for i in range(count)
    )


class RecordingParamCompletion:
    """The one completion backend these tests inject.

    Tags its output so two drafts are distinguishable, counts calls so a hit can be
    proved by absence, and `forbidden=True` turns any call into an assertion failure.
    The `forbidden` switch is a flag on this class rather than a second class because
    the class name is part of the cache fingerprint: a different class would change
    the key and turn a hit into a miss, which is the opposite of what the tests that
    use it are trying to observe.
    """

    def __init__(self, tag: str = "alpha", forbidden: bool = False,
                 content: str = None) -> None:
        self.tag = tag
        self.forbidden = forbidden
        self.content = content
        self.calls = 0

    def __call__(self, **kwargs):
        if self.forbidden:
            raise AssertionError(
                "the LLM was called even though a cache entry should have been reused"
            )
        self.calls += 1
        return _llm_response(
            self.content if self.content is not None else _examples_block(self.tag)
        )


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
    """Guarantee the process-wide cache sink is restored after each test."""
    previous = get_param_example_cache()
    set_param_example_cache(None)
    yield
    set_param_example_cache(previous)


@pytest.fixture
def workflow_dir(tmp_path):
    path = tmp_path / "workflow"
    path.mkdir()
    return str(path)


def _generate(**overrides):
    """Run the real generation path with local, network-free inputs."""
    kwargs = {
        "field_annotations": AddTwoNumbersInput.model_fields,
        "command_name": COMMAND_NAME,
        "num_examples": 3,
        "validation_threshold": 0.3,
        "seed": 42,
        "completion_fn": RecordingParamCompletion(),
    }
    kwargs |= overrides
    return generate_dspy_examples(**kwargs)


def _fingerprint(**overrides):
    """The fingerprint `_generate` would compute, for direct cache manipulation."""
    kwargs = {
        "field_annotations": AddTwoNumbersInput.model_fields,
        "command_name": COMMAND_NAME,
        "num_examples": 3,
        "validation_threshold": 0.3,
        "model": MODEL,
        "completion_fn": RecordingParamCompletion(),
    }
    kwargs |= overrides
    return param_example_fingerprint(
        kwargs["field_annotations"],
        kwargs["command_name"],
        kwargs["num_examples"],
        kwargs["validation_threshold"],
        kwargs["model"],
        completion_fn=kwargs["completion_fn"],
    )


def _commands(examples: list[dict]) -> list[str]:
    return [example["fields"].get("command") for example in examples]


# ---------------------------------------------------------------------------
# Validation before runtime transformation
# ---------------------------------------------------------------------------

def test_validation_filters_hallucinated_strings_and_preserves_runtime_schema(
    tmp_path,
):
    """Rejected source examples must not become runtime few-shot examples.

    Pydantic v2 exposes a string annotation as the class object ``str``. Its string
    representation is ``"<class 'str'>"``, so comparing that representation with
    the literal ``"str"`` made parameter extraction silently find no values and
    accept every parsed example. This drives the real generator with a real Pydantic
    model and the same local completion boundary used by the cache integration tests.
    """
    field_details = {
        field["name"]: field for field in extract_field_details(AddUserInput.model_fields)
    }
    assert field_details["user_name"]["type"] == "str"
    assert field_details["is_premium_user"]["type"] == "bool"

    accepted_source = (
        'dspy.Example(command="add user alice", user_name="alice", '
        'is_premium_user=True).with_inputs("command")'
    )
    rejected_source = (
        'dspy.Example(command="add user alice", user_name="mallory", '
        'is_premium_user=True).with_inputs("command")'
    )
    backend = RecordingParamCompletion(
        content="\n".join((accepted_source, rejected_source))
    )

    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        valid_examples, rejected_examples = _generate(
            field_annotations=AddUserInput.model_fields,
            command_name="add_user",
            num_examples=2,
            completion_fn=backend,
        )
    finally:
        os.chdir(original_cwd)

    expected_valid = canonicalized(
        transform_examples_to_dict_format([accepted_source])
    )
    assert valid_examples == expected_valid
    assert all(
        example["fields"].get("user_name") != "mallory"
        for example in valid_examples
    )
    assert len(rejected_examples) == 1
    assert rejected_examples[0]["params"]["user_name"] == "mallory"
    assert rejected_examples[0]["invalid_params"][0]["param"] == "user_name"
    # The returned/cacheable rejection record is the durable diagnostic. Training must
    # not also leave a command-overwriting debug file in whichever directory launched it.
    assert not (tmp_path / "rejected_examples.json").exists()

    # This is the exact schema consumed by signatures.get_trainset at runtime.
    runtime_example = dspy.Example(
        **valid_examples[0]["fields"]
    ).with_inputs(*valid_examples[0]["inputs"])
    assert runtime_example.command == "add user alice"
    assert runtime_example.labels().user_name == "alice"


def test_field_type_normalization_unwraps_only_optional_unions():
    """A generic such as list[str] must not be mislabeled as its element type."""
    field_details = {
        field["name"]: field
        for field in extract_field_details(MixedFieldTypesInput.model_fields)
    }

    assert field_details["nickname"]["type"] == "str"
    assert field_details["nickname"]["optional"] is True
    assert field_details["tags"]["type"] == "list[str]"
    assert field_details["tags"]["optional"] is False


def test_missing_required_field_is_rejected(tmp_path):
    """A syntactically valid example is unusable when its model requires more fields."""
    source = (
        'dspy.Example(command="add premium user alice", '
        'is_premium_user=True).with_inputs("command")'
    )
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        valid_examples, rejected_examples = _generate(
            field_annotations=AddUserInput.model_fields,
            command_name="add_user",
            num_examples=1,
            completion_fn=RecordingParamCompletion(content=source),
        )
    finally:
        os.chdir(original_cwd)

    assert valid_examples == []
    assert len(rejected_examples) == 1
    assert rejected_examples[0]["reason"] == "Missing required parameters"
    assert rejected_examples[0]["missing_required"] == ["user_name"]


def test_optional_field_may_be_omitted_from_structured_example(tmp_path):
    """Omitting a defaulted Optional field must not erase an otherwise valid example."""
    source = (
        'dspy.Example(command="modify A and B in express mode for order 42 because '
        'it was ordered by mistake", item_ids=["A", "B"], reference=42, '
        'mode="express", reason="ordered by mistake").with_inputs("command")'
    )
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        valid_examples, rejected_examples = _generate(
            field_annotations=StructuredFieldTypesInput.model_fields,
            command_name="modify_items",
            num_examples=1,
            completion_fn=RecordingParamCompletion(content=source),
        )
    finally:
        os.chdir(original_cwd)

    assert rejected_examples == []
    assert valid_examples == [
        {
            "fields": {
                "command": (
                    "modify A and B in express mode for order 42 because it was "
                    "ordered by mistake"
                ),
                "item_ids": ["A", "B"],
                "mode": "express",
                "reason": "ordered by mistake",
                "reference": 42,
            },
            "inputs": ["command"],
        }
    ]
    assert "note" not in valid_examples[0]["fields"]


def test_supported_structured_types_are_reported_from_real_pydantic_fields():
    """Lists, unions, enums, and enum metadata stay structured for validation."""
    details = {
        field["name"]: field
        for field in extract_field_details(StructuredFieldTypesInput.model_fields)
    }

    assert details["item_ids"]["type"] == "list[str]"
    assert details["reference"]["type"] in {"str | int", "int | str"}
    assert details["mode"]["type"] == "DeliveryMode"
    assert details["reason"]["enum"] == [
        "no longer needed",
        "ordered by mistake",
    ]
    assert details["note"]["optional"] is True
    assert details["note"]["required"] is False

    retail_list_details = {
        field["name"]: field
        for field in extract_field_details(
            ModifyPendingOrderItemsSignature.Input.model_fields
        )
    }
    retail_enum_details = {
        field["name"]: field
        for field in extract_field_details(CancelPendingOrderSignature.Input.model_fields)
    }
    assert retail_list_details["item_ids"]["type"] == "List[str]"
    assert retail_list_details["item_ids"]["required"] is False
    assert retail_enum_details["reason"]["enum"] == [
        "no longer needed",
        "ordered by mistake",
    ]


def test_malformed_structured_and_enum_values_are_rejected(tmp_path):
    """Quoted containers and values outside enum domains must not reach few-shot data."""
    source = (
        'dspy.Example(command="modify A overnight because I changed my mind", '
        'item_ids="A", reference={"bad": 1}, mode="overnight", '
        'reason="changed my mind").with_inputs("command")'
    )
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        valid_examples, rejected_examples = _generate(
            field_annotations=StructuredFieldTypesInput.model_fields,
            command_name="modify_items",
            num_examples=1,
            completion_fn=RecordingParamCompletion(content=source),
        )
    finally:
        os.chdir(original_cwd)

    assert valid_examples == []
    assert len(rejected_examples) == 1
    assert rejected_examples[0]["reason"] == "Malformed parameter values"
    assert {
        invalid["param"] for invalid in rejected_examples[0]["invalid_params"]
    } == {"item_ids", "reference", "mode", "reason"}


def test_generated_non_literal_values_are_rejected_without_execution(tmp_path):
    """Model text is parsed as data; a call expression must never run."""
    marker = tmp_path / "executed"
    source = (
        'dspy.Example(command="add user alice", '
        f'user_name=Path({str(marker)!r}).write_text("bad"), '
        'is_premium_user=True).with_inputs("command")'
    )
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        valid_examples, rejected_examples = _generate(
            field_annotations=AddUserInput.model_fields,
            command_name="add_user",
            num_examples=1,
            completion_fn=RecordingParamCompletion(content=source),
        )
    finally:
        os.chdir(original_cwd)

    assert valid_examples == []
    assert len(rejected_examples) == 1
    assert "must be a Python literal" in rejected_examples[0]["reason"]
    assert Path(marker).exists() is False


def test_unknown_generated_fields_are_rejected_and_named(tmp_path):
    """An LLM-invented field must not survive into the runtime example schema."""
    source = (
        'dspy.Example(command="add user alice", user_name="alice", '
        'is_premium_user=True, invented_role="admin").with_inputs("command")'
    )
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        valid_examples, rejected_examples = _generate(
            field_annotations=AddUserInput.model_fields,
            command_name="add_user",
            num_examples=1,
            completion_fn=RecordingParamCompletion(content=source),
        )
    finally:
        os.chdir(original_cwd)

    assert valid_examples == []
    assert len(rejected_examples) == 1
    assert rejected_examples[0]["unknown_fields"] == ["invented_role"]
    assert rejected_examples[0]["reason"] == "Unknown fields: 'invented_role'"


@pytest.mark.parametrize(
    "with_inputs",
    [
        pytest.param("", id="missing-command"),
        pytest.param('"user_name"', id="wrong-input"),
        pytest.param('"command", "user_name"', id="extra-input"),
        pytest.param('"command", "command"', id="duplicate-command"),
    ],
)
def test_with_inputs_requires_exactly_command(tmp_path, with_inputs):
    """Generated examples may expose only command as the DSPy input."""
    source = (
        'dspy.Example(command="add user alice", user_name="alice", '
        f'is_premium_user=True).with_inputs({with_inputs})'
    )
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        valid_examples, rejected_examples = _generate(
            field_annotations=AddUserInput.model_fields,
            command_name="add_user",
            num_examples=1,
            completion_fn=RecordingParamCompletion(content=source),
        )
    finally:
        os.chdir(original_cwd)

    assert valid_examples == []
    assert len(rejected_examples) == 1
    assert (
        rejected_examples[0]["reason"]
        == "with_inputs arguments must be exactly ['command']"
    )


@pytest.mark.parametrize("value", ["True", "1.0"])
def test_literal_membership_requires_matching_runtime_type(tmp_path, value):
    """Python equality must not let bool or float satisfy Literal[1]."""
    source = (
        f'dspy.Example(command="choose {value}", choice={value})'
        '.with_inputs("command")'
    )
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        valid_examples, rejected_examples = _generate(
            field_annotations=LiteralOneInput.model_fields,
            command_name="choose",
            num_examples=1,
            completion_fn=RecordingParamCompletion(content=source),
        )
    finally:
        os.chdir(original_cwd)

    assert valid_examples == []
    assert len(rejected_examples) == 1
    assert rejected_examples[0]["reason"] == "Malformed parameter values"
    assert rejected_examples[0]["invalid_params"][0]["param"] == "choice"


def test_literal_membership_accepts_same_value_and_type(tmp_path):
    """Literal[1] still accepts the integer value 1."""
    source = (
        'dspy.Example(command="choose 1", choice=1).with_inputs("command")'
    )
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        valid_examples, rejected_examples = _generate(
            field_annotations=LiteralOneInput.model_fields,
            command_name="choose",
            num_examples=1,
            completion_fn=RecordingParamCompletion(content=source),
        )
    finally:
        os.chdir(original_cwd)

    assert rejected_examples == []
    assert valid_examples[0]["fields"]["choice"] == 1


@pytest.mark.parametrize(
    ("command", "message"),
    [
        (
            "Can you notify all employees that the deadline for the quarterly report "
            "has been extended to Friday?",
            "The deadline for the quarterly report has been extended to Friday.",
        ),
        (
            "Remind all staff that the guest Wi-Fi password has been updated. Let them "
            "know the new one is 'Connect2024'.",
            "The guest Wi-Fi password has been updated. The new one is 'Connect2024'.",
        ),
    ],
)
def test_long_valid_messages_use_value_length_fuzzy_windows(command, message):
    """Real LLM examples must survive punctuation changes and light paraphrases."""
    result = validate_parameters(command, {"message": message}, threshold=0.3)

    assert result["message"]["valid"] is True
    assert result["message"]["confidence"] >= 0.7


def test_long_message_fix_does_not_relax_short_identifier_matching():
    """The measured long-text fix must not admit unrelated short identifiers."""
    result = validate_parameters(
        "add user alice",
        {"user_name": "mallory"},
        threshold=0.3,
    )

    assert result["user_name"]["valid"] is False


# ---------------------------------------------------------------------------
# Fingerprint: what must invalidate
# ---------------------------------------------------------------------------

def test_fingerprint_is_stable_for_identical_inputs():
    assert _fingerprint().variant_key == _fingerprint().variant_key


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param(
            {"field_annotations": AddTwoNumbersInputWithComment.model_fields},
            id="parameter-model",
        ),
        pytest.param({"command_name": "subtract_two_numbers"}, id="command-name"),
        pytest.param({"num_examples": 4}, id="num-examples"),
        pytest.param({"validation_threshold": 0.4}, id="validation-threshold"),
        pytest.param({"model": "openai/gpt-4o-mini"}, id="model"),
        pytest.param({"completion_fn": None}, id="completion-backend"),
    ],
)
def test_changing_any_generation_input_changes_the_variant_key(overrides):
    """Every input the LLM sees, plus every input that shapes what gets stored."""
    assert _fingerprint().variant_key != _fingerprint(**overrides).variant_key


def test_the_temperature_is_in_the_fingerprint():
    """A future decision to lower it must invalidate loudly, not reuse silently.

    The temperature is the one generation knob whose value is under active discussion
    (see the constant's comment in `generate_param_examples`), so it is named in the
    key explicitly rather than left to the prompt-source digest — which would stop
    covering it the moment the constant moved to a config file.
    """
    inputs = _fingerprint().inputs
    assert inputs["temperature"] == DSPY_EXAMPLE_TEMPERATURE
    assert inputs["temperature"] == 0.9


def test_the_api_base_is_hashed_not_stored():
    """The fingerprint file is meant to be readable and shareable."""
    previous = dict(fastworkflow._env_vars)
    try:
        fastworkflow.init(
            {**previous, "LITELLM_PROXY_API_BASE": "https://internal.example.invalid"}
        )
        with_proxy = _fingerprint()
    finally:
        fastworkflow.init(previous)
    without_proxy = _fingerprint()

    assert with_proxy.variant_key != without_proxy.variant_key
    assert "internal.example.invalid" not in json.dumps(with_proxy.inputs)
    assert with_proxy.inputs["api_base_digest"]


def test_the_api_key_is_not_in_the_fingerprint():
    """A key identifies a caller, not a generator. R6 excludes it for the same reason."""
    serialised = json.dumps(_fingerprint().inputs)
    assert "test-key-not-used-by-the-injected-backend" not in serialised
    assert "api_key" not in serialised


def test_the_production_backend_is_named_by_a_stable_constant():
    """`litellm.completion` is a decorated wrapper whose qualname moves between releases.

    Keying on it directly would invalidate every workflow's cache on an unrelated
    dependency bump.
    """
    assert _fingerprint(completion_fn=None).inputs["completion_backend"] == (
        PRODUCTION_COMPLETION_BACKEND
    )
    assert callable_identity(None, PRODUCTION_COMPLETION_BACKEND) == (
        PRODUCTION_COMPLETION_BACKEND
    )


# ---------------------------------------------------------------------------
# The prompt-source digest
# ---------------------------------------------------------------------------

def test_every_function_in_the_module_is_digested_or_explicitly_exempt():
    """The maintenance hazard of a hand-written digest list, closed by a test.

    A future contributor who adds a prompt-building or payload-shaping helper and
    forgets `_digested_functions` would silently create a cache that serves stale
    results after their own edits. This fails until the new function is classified one
    way or the other.
    """
    digested = {func.__name__ for func in _digested_functions()}
    defined = {
        name
        for name, obj in inspect.getmembers(
            generate_param_examples, inspect.isfunction
        )
        if obj.__module__ == generate_param_examples.__name__
    }
    unclassified = defined - digested - _UNDIGESTED_FUNCTIONS
    assert not unclassified, (
        "these functions are neither digested into the parameter-example cache key "
        f"nor listed as unable to affect it: {sorted(unclassified)}"
    )


def test_the_digest_combines_every_listed_function():
    """Not just the first one: all six must move the combined digest."""
    digests = {source_digest(func) for func in _digested_functions()}
    assert len(digests) == len(_digested_functions()), (
        "two digested functions hashed identically, so the combination is degenerate"
    )
    assert prompt_source_digest() == prompt_source_digest()
    assert prompt_source_digest() != source_digest(generate_dspy_examples)


def test_the_source_digest_is_sensitive_to_a_comment():
    """R6's coarseness rule: any edit, including a comment, must invalidate."""

    def with_a_comment():
        # this comment is the only difference
        return 1

    def without_a_comment():
        return 1

    assert source_digest(with_a_comment) != source_digest(without_a_comment)


# ---------------------------------------------------------------------------
# Reuse, and the staleness hazard reuse introduces
# ---------------------------------------------------------------------------

def test_a_second_run_reuses_the_entry_without_calling_the_llm(workflow_dir):
    """The whole point: same configuration, same seed, same examples, no LLM call."""
    cache = ParamExampleCache(workflow_dir, mode=MODE_REUSE)
    set_param_example_cache(cache)

    backend = RecordingParamCompletion(tag="alpha")
    first_examples, first_rejected = _generate(completion_fn=backend)
    assert backend.calls == 1
    assert cache.stats["stored"] == 1

    second_examples, second_rejected = _generate(
        completion_fn=RecordingParamCompletion(tag="beta", forbidden=True)
    )
    assert second_examples == first_examples
    assert second_rejected == first_rejected
    assert cache.stats["hit"] == 1
    assert _commands(second_examples) == _commands(first_examples)


def test_a_hit_and_a_miss_serialise_to_identical_bytes(workflow_dir):
    """The bug that only a second workflow revealed, pinned as a regression test.

    `<command>_param_labeled.json` is written with `json.dump(..., indent=2)`, which
    preserves insertion order, while the cache file is written with `sort_keys=True`,
    which sorts nested keys too. So a generated example and its cached twin compared
    EQUAL as dicts and serialised to DIFFERENT bytes. Measured on
    `examples/messaging_app_4`: every example identical, and 3 of 5 artifacts still
    differing between two runs. `examples/hello_world` alone would have passed, because
    `command` < `first_number` < `second_number` is already alphabetical — which is
    exactly why the measurement was repeated on a second workflow.
    """
    cache = ParamExampleCache(workflow_dir)
    set_param_example_cache(cache)

    generated, _rejected = _generate(
        field_annotations=AddUserInput.model_fields,
        command_name="add_user",
        completion_fn=RecordingParamCompletion(content=_UNSORTED_FIELD_EXAMPLES),
    )
    reused, _rejected = _generate(
        field_annotations=AddUserInput.model_fields,
        command_name="add_user",
        completion_fn=RecordingParamCompletion(forbidden=True),
    )

    assert generated == reused
    assert json.dumps(generated, indent=2) == json.dumps(reused, indent=2), (
        "the cached examples round-tripped with a different key ORDER; the two runs "
        "would write byte-different <command>_param_labeled.json files"
    )
    # And the canonical form is what actually landed, not merely a coincidence.
    for example in generated:
        assert list(example["fields"]) == sorted(example["fields"])


def test_editing_the_parameter_model_invalidates_the_entry(workflow_dir):
    """Stale reuse after a field is added is worse than no cache; prove it cannot happen.

    The runtime few-shot prompts parameter extraction with these examples. Serving the
    pre-edit draw would mean prompting with examples that cannot possibly mention the
    field that was just added.
    """
    set_param_example_cache(ParamExampleCache(workflow_dir))

    original = RecordingParamCompletion(tag="alpha")
    _generate(completion_fn=original)
    assert original.calls == 1

    after_edit = RecordingParamCompletion(tag="alpha", forbidden=True)
    with pytest.raises(AssertionError, match="cache entry should have been reused"):
        _generate(
            field_annotations=AddTwoNumbersInputWithComment.model_fields,
            completion_fn=after_edit,
        )


def test_a_different_seed_is_a_different_entry(workflow_dir):
    """A seed sweep must vary the whole pipeline, not hold this half of it fixed.

    Sharing one draw across seeds would make a multi-seed variance measurement
    understate the variance of "train this workflow", because one of the two LLM
    inputs would be pinned by accident.
    """
    cache = ParamExampleCache(workflow_dir)
    set_param_example_cache(cache)

    _generate(seed=42, completion_fn=RecordingParamCompletion(tag="alpha"))
    at_other_seed = RecordingParamCompletion(tag="beta")
    _generate(seed=7, completion_fn=at_other_seed)
    assert at_other_seed.calls == 1

    # Both live in one file, keyed by seed, exactly as R6 stores them.
    with open(cache.entry_path(_fingerprint()), encoding="utf-8") as f:
        payload = json.load(f)
    assert sorted(payload["entries"]) == ["42", "7"]


def test_the_seed_defaults_to_the_configured_training_seed(workflow_dir):
    """No explicit seed means `TRAINING_SEED`, which the fixture pinned at 42."""
    cache = ParamExampleCache(workflow_dir)
    set_param_example_cache(cache)
    _generate(seed=None)

    with open(cache.entry_path(_fingerprint()), encoding="utf-8") as f:
        payload = json.load(f)
    assert list(payload["entries"]) == ["42"]


# ---------------------------------------------------------------------------
# Degraded generations are never cached
# ---------------------------------------------------------------------------

def test_a_draw_that_parsed_to_nothing_is_not_cached(workflow_dir):
    """Freezing an empty set would make one bad minute permanent.

    A truncated response or a model that ignored the output format yields no parsable
    examples. Caching that would leave the command few-shot prompting with nothing on
    every subsequent run, with no failure left in sight to explain it.
    """
    cache = ParamExampleCache(workflow_dir)
    set_param_example_cache(cache)

    empty = RecordingParamCompletion(
        content="I'm sorry, I can't help with that request."
    )
    examples, _rejected = _generate(completion_fn=empty)
    assert examples == []
    assert cache.stats["stored"] == 0

    # The next run must try again rather than reuse the degraded set.
    retry = RecordingParamCompletion(tag="alpha")
    recovered, _rejected = _generate(completion_fn=retry)
    assert retry.calls == 1
    assert recovered


def test_an_entry_with_no_valid_examples_reads_as_a_miss(workflow_dir):
    """Defence in depth: a hand-edited or older empty entry must not be served."""
    cache = ParamExampleCache(workflow_dir)
    fingerprint = _fingerprint()
    assert cache.store(fingerprint, 42, []) is False

    # Written by hand, since `store` refuses to create one.
    os.makedirs(cache.root, exist_ok=True)
    with open(cache.entry_path(fingerprint), "w", encoding="utf-8") as f:
        json.dump(
            {
                "cache_format_version": CACHE_FORMAT_VERSION,
                "command_name": COMMAND_NAME,
                "variant_key": fingerprint.variant_key,
                "entries": {"42": {"seed": 42, "valid_examples": []}},
            },
            f,
        )
    assert cache.lookup(fingerprint, 42) is None
    assert ParamExampleCacheEntry(seed=42).is_usable() is False


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def test_removed_off_mode_falls_back_to_reuse(workflow_dir):
    cache = ParamExampleCache(workflow_dir, mode="off")
    set_param_example_cache(cache)

    first = RecordingParamCompletion(tag="alpha")
    _generate(completion_fn=first)
    second = RecordingParamCompletion(forbidden=True)
    _generate(completion_fn=second)

    assert first.calls == 1
    assert second.calls == 0
    assert cache.mode == MODE_REUSE
    assert os.path.exists(cache.root)


def test_regenerate_mode_refreshes_the_entry_rather_than_bypassing_it(workflow_dir):
    """`regenerate` still WRITES: the next `reuse` run gets the fresh draw."""
    set_param_example_cache(ParamExampleCache(workflow_dir, mode=MODE_REUSE))
    _generate(completion_fn=RecordingParamCompletion(tag="alpha"))

    regenerating = ParamExampleCache(workflow_dir, mode=MODE_REGENERATE)
    set_param_example_cache(regenerating)
    backend = RecordingParamCompletion(tag="beta")
    refreshed, _rejected = _generate(completion_fn=backend)
    assert backend.calls == 1
    assert regenerating.stats["stored"] == 1
    assert all("beta" in command for command in _commands(refreshed))

    set_param_example_cache(ParamExampleCache(workflow_dir, mode=MODE_REUSE))
    reused, _rejected = _generate(
        completion_fn=RecordingParamCompletion(forbidden=True)
    )
    assert _commands(reused) == _commands(refreshed)


def test_removed_aggregate_mode_falls_back_to_reuse(workflow_dir):
    """The removed mode must not reintroduce a cross-seed union through this cache.

    That is a behaviour change dressed up as a cache mode, and nothing has measured
    whether more few-shot examples help or hurt. Unknown modes follow the shared cache
    normalizer's conservative fallback to exact-seed reuse.
    """
    cache = ParamExampleCache(workflow_dir, mode="aggregate")
    set_param_example_cache(cache)
    first, _rejected = _generate(completion_fn=RecordingParamCompletion(tag="alpha"))
    reused, _rejected = _generate(
        completion_fn=RecordingParamCompletion(forbidden=True)
    )
    assert reused == first
    assert cache.mode == MODE_REUSE
    assert cache.reads_enabled is True


def test_cache_mode_is_not_workflow_environment_configuration():
    """`--regenerate-utterances` must refresh BOTH generated artifacts.

    Someone forcing regeneration means "stop reusing anything"; the train command passes
    that decision directly to both cache constructors.
    """
    previous = dict(fastworkflow._env_vars)
    try:
        fastworkflow.init({
            k: v for k, v in previous.items()
            if k not in ("PARAM_EXAMPLE_CACHE_MODE", "UTTERANCE_CACHE_MODE")
        })
        assert resolve_cache_mode() == DEFAULT_CACHE_MODE

        fastworkflow.init({**previous, "UTTERANCE_CACHE_MODE": MODE_REGENERATE})
        assert resolve_cache_mode() == DEFAULT_CACHE_MODE

        fastworkflow.init({
            **previous,
            "UTTERANCE_CACHE_MODE": MODE_REGENERATE,
            "PARAM_EXAMPLE_CACHE_MODE": "off",
        })
        assert resolve_cache_mode() == DEFAULT_CACHE_MODE
    finally:
        fastworkflow.init(previous)


def test_an_unknown_mode_falls_back_to_reuse():
    """A typo must not silently disable reuse and quietly cost money on every run."""
    previous = dict(fastworkflow._env_vars)
    try:
        fastworkflow.init({**previous, "PARAM_EXAMPLE_CACHE_MODE": "reyuse"})
        assert resolve_cache_mode() == DEFAULT_CACHE_MODE
    finally:
        fastworkflow.init(previous)


# ---------------------------------------------------------------------------
# Corrupt and foreign files degrade to a miss, never to a wrong answer
# ---------------------------------------------------------------------------

def test_an_unreadable_entry_is_a_miss(workflow_dir):
    cache = ParamExampleCache(workflow_dir)
    fingerprint = _fingerprint()
    os.makedirs(cache.root, exist_ok=True)
    with open(cache.entry_path(fingerprint), "w", encoding="utf-8") as f:
        f.write("{not json at all")

    assert cache.lookup(fingerprint, 42) is None
    assert cache.stats["unreadable"] == 1


def test_a_renamed_entry_is_ignored(workflow_dir):
    """The filename is derived from the key; disagreement means the content lies."""
    cache = ParamExampleCache(workflow_dir)
    fingerprint = _fingerprint()
    os.makedirs(cache.root, exist_ok=True)
    with open(cache.entry_path(fingerprint), "w", encoding="utf-8") as f:
        json.dump(
            {
                "cache_format_version": CACHE_FORMAT_VERSION,
                "command_name": COMMAND_NAME,
                "variant_key": "0123456789abcdef01234567",
                "entries": {
                    "42": {
                        "seed": 42,
                        "valid_examples": [{"fields": {"command": "stale"},
                                            "inputs": ["command"]}],
                    }
                },
            },
            f,
        )
    assert cache.lookup(fingerprint, 42) is None


def test_a_future_format_version_is_ignored_never_migrated(workflow_dir):
    cache = ParamExampleCache(workflow_dir)
    fingerprint = _fingerprint()
    os.makedirs(cache.root, exist_ok=True)
    with open(cache.entry_path(fingerprint), "w", encoding="utf-8") as f:
        json.dump(
            {
                "cache_format_version": CACHE_FORMAT_VERSION + 1,
                "command_name": COMMAND_NAME,
                "variant_key": fingerprint.variant_key,
                "entries": {
                    "42": {
                        "seed": 42,
                        "valid_examples": [{"fields": {"command": "stale"},
                                            "inputs": ["command"]}],
                    }
                },
            },
            f,
        )
    assert cache.lookup(fingerprint, 42) is None


# ---------------------------------------------------------------------------
# On-disk layout
# ---------------------------------------------------------------------------

def test_cache_lives_beside_the_versions_directory(workflow_dir):
    cache = ParamExampleCache(workflow_dir)
    assert cache.root == os.path.join(
        workflow_dir, COMMAND_INFO_FOLDERNAME, CACHE_DIRNAME
    )
    # Reading must not create anything: inspecting an unbuilt workflow stays
    # read-only, matching `artifact_versioning.command_info_root`.
    assert not os.path.exists(cache.root)


def test_cache_dirname_is_exempt_from_the_stale_artifact_prune():
    """`_prune_stale_artifacts` skips RESERVED_TOPLEVEL_NAMES; the cache is in it.

    The constant is duplicated in `artifact_versioning` to keep that module free of
    heavy imports, so the two must be asserted to agree. Without the exemption every
    training run would delete the cache it just wrote, and the second run would be
    just as non-reproducible as before.
    """
    assert CACHE_DIRNAME == artifact_versioning.PARAM_EXAMPLE_CACHE_DIRNAME
    assert CACHE_DIRNAME in artifact_versioning.RESERVED_TOPLEVEL_NAMES


def test_cache_is_never_a_valid_artifact_version_id():
    with pytest.raises(ValueError):
        artifact_versioning.version_dir("/tmp/nope", CACHE_DIRNAME)


def test_store_writes_the_expensive_to_regenerate_readme(workflow_dir):
    cache = ParamExampleCache(workflow_dir)
    cache.store(
        _fingerprint(), 42,
        [{"fields": {"command": "add 1 and 2"}, "inputs": ["command"]}],
    )
    readme = os.path.join(cache.root, CACHE_README_FILENAME)
    assert os.path.isfile(readme)
    with open(readme, encoding="utf-8") as f:
        assert "COSTS MONEY" in f.read()


def test_the_entry_file_is_written_with_sorted_keys(workflow_dir):
    """Two runs producing the same content must produce the same BYTES.

    R2's acceptance test is a byte comparison, and an unordered dict dump would fail
    it for a reason nobody could act on.
    """
    cache = ParamExampleCache(workflow_dir)
    fingerprint = _fingerprint()
    payload = [{"fields": {"command": "add 1 and 2"}, "inputs": ["command"]}]
    cache.store(fingerprint, 42, payload)
    with open(cache.entry_path(fingerprint), "rb") as f:
        first = f.read()

    body = json.loads(first)
    assert list(body) == sorted(body)


def test_rejected_examples_round_trip(workflow_dir):
    """Both halves are written to `<command>_param_labeled.json`, so both are cached.

    A cache that reproduced only the accepted half would still leave that file
    differing between two runs, which is the exact thing this exists to stop.
    """
    cache = ParamExampleCache(workflow_dir)
    fingerprint = _fingerprint()
    rejected = [{"example": "dspy.Example(...)", "reason": "Command extraction failed",
                 "command": None, "params": {}}]
    cache.store(
        fingerprint, 42,
        [{"fields": {"command": "add 1 and 2"}, "inputs": ["command"]}],
        rejected,
    )
    entry = cache.lookup(fingerprint, 42)
    assert entry is not None
    assert entry.rejected_examples == rejected


def test_generation_without_an_installed_cache_is_unchanged(workflow_dir):
    """Anything calling `generate_dspy_examples` directly gets the old behaviour."""
    assert get_param_example_cache() is None
    backend = RecordingParamCompletion(tag="alpha")
    examples, _rejected = _generate(completion_fn=backend)
    assert backend.calls == 1
    assert examples
    assert not os.path.exists(
        os.path.join(workflow_dir, COMMAND_INFO_FOLDERNAME, CACHE_DIRNAME)
    )


def test_cache_summary_reports_reuse(workflow_dir):
    cache = ParamExampleCache(workflow_dir)
    set_param_example_cache(cache)
    _generate(completion_fn=RecordingParamCompletion(tag="alpha"))
    _generate(completion_fn=RecordingParamCompletion(forbidden=True))

    summary = cache.format_summary()
    assert "1 reused" in summary
    assert "1 written" in summary
    assert cache.root in summary


def test_canonicalized_sorts_nested_mappings_and_preserves_list_order():
    """Key order is not meaningful; list order is. Only one of them may be touched."""
    payload = [
        {"fields": {"user_name": "bob", "command": "add bob"}, "inputs": ["command"]},
        {"b": 1, "a": {"z": [3, 1, 2], "y": None}},
    ]
    result = canonicalized(payload)
    assert list(result[0]) == ["fields", "inputs"]
    assert list(result[0]["fields"]) == ["command", "user_name"]
    assert list(result[1]) == ["a", "b"]
    assert list(result[1]["a"]) == ["y", "z"]
    assert result[1]["a"]["z"] == [3, 1, 2]
    assert result == payload


def test_extract_field_details_is_covered_by_the_field_details_digest():
    """The structured render is keyed separately from the raw annotation text.

    Both reach the prompt — the raw `model_fields` repr inside a ```python fence, and
    the extracted details as the "Fields to extract" block — and `extract_field_details`
    can change what it pulls out of an unchanged annotation.
    """
    inputs = _fingerprint().inputs
    assert inputs["field_annotations_digest"]
    assert inputs["field_details_digest"]
    assert inputs["field_annotations_digest"] != inputs["field_details_digest"]
    assert extract_field_details(AddTwoNumbersInput.model_fields) != (
        extract_field_details(AddTwoNumbersInputWithComment.model_fields)
    )

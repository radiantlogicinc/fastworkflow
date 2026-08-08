"""Integration tests for bd fix-r4p: no memory address may reach the LLM prompt.

`generate_dspy_examples` built its "Fields to extract" block with
``', '.join(repr(ex) for ex in field['examples'])``. Most objects inherit
``object.__repr__``, whose text embeds ``id(self)``, so a field declared with a
non-JSON-native example put ``<Foo object at 0x7f...>`` **inside the prompt sent to the
model**.

bd fix-k0i.46 fixed the neighbouring instance of this in the cache KEY: `_canonical_digest`
no longer uses ``default=repr``, because the embedded address made the variant key differ
every run. This one is worse in kind. An unstable key costs one wasted regeneration; an
address in the prompt is noise the model conditions on, it changes every run, and so the
examples it produced are irreproducible for a reason no diff shows.

The fix renders through `param_example_cache.run_stable_form` — the same canonicaliser the
digest reaches through ``json.dumps(default=...)`` — and then ``repr``s the result, so
Python-shaped output is preserved for the JSON-native examples every shipped workflow
actually has. A value with no run-stable representation renders as its type and is
reported at WARNING.

There are TWO routes into the prompt and both are covered here. The second one is not
named in the issue and was found while fixing it: the prompt also interpolates
``str(field_annotations)`` verbatim, and pydantic's ``FieldInfo`` repr nests the repr of
anything in ``json_schema_extra``. Fixing only the line the issue names would have left
the address in the prompt and changed nothing about reproducibility.

Per `.cursor/rules/testing_rules.mdc` these are integration tests: no Mock fixtures. Real
pydantic models, the real `generate_dspy_examples`, and a locally defined completion
backend that captures the prompt production actually sends it — which is the injection
point the generator declares for exactly this purpose, not a stand-in for one.
"""

import json
import logging
import os
import re
from decimal import Decimal
from enum import Enum, IntEnum
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, Field

import fastworkflow
from fastworkflow.train.param_example_cache import (
    _canonicalize_unserialisable,
    get_param_example_cache,
    run_stable_form,
    set_param_example_cache,
)
from fastworkflow.utils.generate_param_examples import (
    _address_free,
    _digested_functions,
    extract_field_details,
    generate_dspy_examples,
    render_field_examples,
)
from fastworkflow.utils.logging import logger as fastworkflow_logger

MODEL = "mistral/mistral-small-latest"
COMMAND_NAME = "add_user"

#: What `object.__repr__` puts in the prompt. Matched rather than searched for as the
#: literal "0x", because a legitimate example value could contain "0x" — a hex product
#: code, say — and a test that failed on that would be a test nobody trusts.
_ADDRESS_RE = re.compile(r"0x[0-9a-fA-F]{6,}")


class _OpaqueExample:
    """Inherits ``object.__repr__``, whose text embeds ``id(self)``.

    The whole defect in one class: `repr` of one of these differs between two instances
    and between two processes, so a prompt built from it is never the same twice.
    """


class DeliveryMode(str, Enum):
    """A `str`-mixin enum, which is the shape shipped workflows actually use."""

    STANDARD = "standard"
    EXPRESS = "express"


class Priority(IntEnum):
    """An `int`-mixin enum, for the numeric half of the same question."""

    LOW = 1
    HIGH = 9


class OpaqueExampleInput(BaseModel):
    """A parameter model whose field example has no run-stable representation.

    Declared through ``json_schema_extra`` rather than through the native
    ``Field(examples=[...])`` kwarg because pydantic rejects a non-serialisable value in
    the latter at class-definition time. That is worth knowing: the native kwarg IS the
    strict boundary the issue proposes, and it already exists — it just is not the only
    way `extract_field_details` receives examples.
    """

    user_name: str = Field(
        json_schema_extra={"description": "who to add", "examples": [_OpaqueExample()]}
    )


class MixedExampleInput(BaseModel):
    """One usable example and one that cannot be rendered, in one field."""

    user_name: str = Field(
        json_schema_extra={
            "description": "who to add",
            "examples": ["bob", _OpaqueExample()],
        }
    )


class JsonNativeInput(BaseModel):
    """The shape every shipped example workflow has: plain JSON examples."""

    user_name: str = Field(
        json_schema_extra={"description": "who to add", "examples": ["bob", "alice"]}
    )
    retries: int = Field(
        default=0, json_schema_extra={"description": "how many", "examples": [0, 3]}
    )
    verbose: bool = Field(
        default=False,
        json_schema_extra={"description": "chatty", "examples": [True, False]},
    )
    note: str = Field(
        default="", json_schema_extra={"description": "free text", "examples": [None]}
    )


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


class PromptCapturingCompletion:
    """The production `completion_fn`, keeping the prompt it was asked to answer.

    This is how the prompt is asserted on: the string this records is the one
    `litellm.completion` would have received, character for character.
    """

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def __call__(self, **kwargs):
        self.prompts.append(kwargs["messages"][1]["content"])
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            'dspy.Example(command="add user bob", user_name="bob")'
                            '.with_inputs("command")'
                        )
                    )
                )
            ]
        )


def _prompt_for(model_class, tmp_path) -> str:
    """Build one real prompt through `generate_dspy_examples` and return it.

    ``chdir`` because the generator prints debugging output relative to the working
    directory; nothing is written by these calls, but keeping them out of the repo root
    is the established habit in `tests/test_param_example_reporting.py`.
    """
    backend = PromptCapturingCompletion()
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        generate_dspy_examples(
            field_annotations=model_class.model_fields,
            command_name=COMMAND_NAME,
            num_examples=1,
            validation_threshold=0.3,
            seed=42,
            completion_fn=backend,
        )
    finally:
        os.chdir(original_cwd)
    assert len(backend.prompts) == 1, "the injected backend was not the one called"
    return backend.prompts[0]


# ---------------------------------------------------------------------------
# The premise
# ---------------------------------------------------------------------------

def test_two_instances_of_the_opaque_class_really_do_repr_differently():
    """Without this, every prompt assertion below proves nothing."""
    assert repr(_OpaqueExample()) != repr(_OpaqueExample())
    assert _ADDRESS_RE.search(repr(_OpaqueExample()))


def test_the_old_rendering_really_did_put_an_address_in_the_prompt():
    """State the defect as an executable fact, so the fix has something to be measured
    against. This is the exact expression `generate_dspy_examples` used."""
    examples = OpaqueExampleInput.model_fields["user_name"].json_schema_extra["examples"]
    assert _ADDRESS_RE.search(", ".join(repr(ex) for ex in examples))


# ---------------------------------------------------------------------------
# The load-bearing claim: no address reaches the prompt
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "model_class",
    [
        pytest.param(OpaqueExampleInput, id="only-example-is-opaque"),
        pytest.param(MixedExampleInput, id="one-of-two-examples-is-opaque"),
    ],
)
def test_the_prompt_sent_to_the_model_contains_no_memory_address(
    model_class, tmp_path
):
    """The whole of bd fix-r4p, asserted on the text the model receives.

    Deliberately a claim about the WHOLE prompt rather than about the "Examples:" line,
    because the address had two routes in and the issue names only one: the prompt also
    interpolates ``str(field_annotations)``, whose ``FieldInfo`` repr nests the repr of
    everything in ``json_schema_extra``.
    """
    prompt = _prompt_for(model_class, tmp_path)

    assert not _ADDRESS_RE.search(prompt), (
        f"a memory address reached the prompt: "
        f"{_ADDRESS_RE.search(prompt).group(0)!r} in "
        f"{[line for line in prompt.splitlines() if _ADDRESS_RE.search(line)]}"
    )
    assert "object at" not in prompt


def test_two_runs_build_the_same_prompt_from_the_same_parameter_model(tmp_path):
    """The consequence the fix exists for, and the reason it is worse than a stale key.

    The two models below are structurally identical and differ only in WHICH instance of
    `_OpaqueExample` they hold — which is exactly the difference between two runs of the
    same workflow. Under the old rendering their prompts differed, so the generated
    parameter examples were irreproducible with nothing in the diff to explain it.
    """
    class FirstRun(BaseModel):
        user_name: str = Field(
            json_schema_extra={"description": "who", "examples": [_OpaqueExample()]}
        )

    class SecondRun(BaseModel):
        user_name: str = Field(
            json_schema_extra={"description": "who", "examples": [_OpaqueExample()]}
        )

    first = _prompt_for(FirstRun, tmp_path)
    second = _prompt_for(SecondRun, tmp_path)
    assert first == second


def test_the_examples_line_names_the_type_instead_of_the_address(tmp_path):
    """A placeholder the model cannot mistake for a value, and cannot vary on."""
    prompt = _prompt_for(MixedExampleInput, tmp_path)
    examples_line = next(
        line for line in prompt.splitlines() if line.strip().startswith("Examples:")
    )

    assert "'bob'" in examples_line
    assert f"<{__name__}._OpaqueExample>" in examples_line


def test_the_annotations_block_keeps_its_shape_minus_the_address(tmp_path):
    """The second route, fixed textually rather than by re-rendering ``model_fields``.

    Re-rendering the block would change the prompt for every workflow in existence,
    which is a training-data QUALITY change wearing a determinism fix's clothes. So the
    ``FieldInfo`` repr is still there, with the address gone from inside it.
    """
    prompt = _prompt_for(OpaqueExampleInput, tmp_path)
    annotations_line = next(
        line for line in prompt.splitlines() if "FieldInfo" in line
    )

    assert "json_schema_extra=" in annotations_line
    assert f"<{__name__}._OpaqueExample object>" in annotations_line
    assert not _ADDRESS_RE.search(annotations_line)


def test_the_unrenderable_example_is_reported_by_field_and_command(log_sink, tmp_path):
    """A coercion is acceptable; a silent one is not.

    The prompt is invisible to the developer, so the log line is the only place they can
    learn that the model was shown a type name where they wrote an example.
    """
    _prompt_for(MixedExampleInput, tmp_path)

    warnings = [
        message
        for message in log_sink.messages(logging.WARNING)
        if "no run-stable representation" in message
    ]
    assert len(warnings) == 1, log_sink.messages(logging.WARNING)
    assert "user_name" in warnings[0]
    assert COMMAND_NAME in warnings[0]
    assert "_OpaqueExample" in warnings[0]
    assert "JSON-native" in warnings[0]


# ---------------------------------------------------------------------------
# What must NOT change: the prompt every shipped workflow gets
# ---------------------------------------------------------------------------

def test_a_json_native_field_renders_exactly_what_it_rendered_before():
    """Over-invalidation of prompt CONTENT is a training-data change, not a fix.

    Every field example in every shipped example workflow is plain JSON, so the new
    renderer must be character-for-character the old expression for all of them. The
    cache key moves regardless — editing `generate_dspy_examples` moves its source
    digest — but what the model is ASKED must be untouched.
    """
    for field in extract_field_details(JsonNativeInput.model_fields):
        assert render_field_examples(
            field["name"], field["examples"], COMMAND_NAME
        ) == ", ".join(repr(example) for example in field["examples"])


def test_a_field_with_no_examples_still_renders_none():
    """The pre-existing empty-list branch, which the old expression handled inline."""
    assert render_field_examples("user_name", [], COMMAND_NAME) == "None"


def test_the_address_strip_leaves_ordinary_annotation_text_alone():
    """It must match nothing in a prompt built from a JSON-native parameter model."""
    annotations = str(JsonNativeInput.model_fields)
    assert _address_free(annotations) == annotations


# ---------------------------------------------------------------------------
# The prompt and the cache key must agree on what a value IS
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "value",
    [
        pytest.param("bob", id="str"),
        pytest.param(3, id="int"),
        pytest.param(1.5, id="float"),
        pytest.param(True, id="bool"),
        pytest.param(None, id="none"),
        pytest.param(["a", "b"], id="list"),
        pytest.param({"b": 1, "a": 2}, id="dict"),
        pytest.param(DeliveryMode.EXPRESS, id="str-mixin-enum"),
        pytest.param(Priority.HIGH, id="int-mixin-enum"),
        pytest.param(Decimal("1.10"), id="decimal"),
        pytest.param(b"raw", id="bytes"),
        pytest.param(re.compile(r"^#W\d+$"), id="compiled-pattern"),
        pytest.param({"a", "b", "c"}, id="set"),
        pytest.param(("a", "b"), id="tuple"),
    ],
)
def test_the_prompt_renderer_and_the_digest_see_the_same_value(value):
    """"The same canonicaliser" has to mean the same RESULT, not merely the same import.

    The digest reaches `_canonicalize_unserialisable` lazily through ``json.dumps``;
    `run_stable_form` applies it eagerly. If the two disagreed, the prompt could show a
    value the key had never seen — and the sharing would be decorative.
    """
    assert json.dumps(run_stable_form(value), sort_keys=True) == json.dumps(
        value, sort_keys=True, default=_canonicalize_unserialisable
    )


def test_a_str_mixin_enum_renders_its_value_not_its_member_name():
    """`str(DeliveryMode.EXPRESS)` is ``'DeliveryMode.EXPRESS'`` on this interpreter.

    The cache key sees ``'express'``, because a `str`-mixin member IS a `str` and
    ``json.dumps`` writes its value. Narrowing through ``str()`` rather than through
    ``str.__str__`` would have shown the model a name the key had never recorded.
    """
    assert render_field_examples(
        "mode", [DeliveryMode.EXPRESS], COMMAND_NAME
    ) == repr("express")
    assert str(DeliveryMode.EXPRESS) != "express", (
        "the interpreter changed; this test no longer exercises the distinction"
    )


def test_a_value_with_no_run_stable_form_still_raises_for_the_digest():
    """The renderer's tolerance must not have loosened the digest.

    A digest that cannot be computed correctly must not be computed approximately: two
    configurations sharing one entry is a stale hit, the one direction this cache may
    never go (bd fix-k0i.46).
    """
    with pytest.raises(TypeError, match="no run-stable representation"):
        run_stable_form(_OpaqueExample())


# ---------------------------------------------------------------------------
# The renderer is inside the cache key
# ---------------------------------------------------------------------------

def test_the_two_new_prompt_functions_are_digested():
    """They decide the text of two parts of the prompt, so an edit must invalidate.

    `test_param_example_cache.test_every_function_in_the_module_is_digested_or_explicitly_exempt`
    would fail if they were classified nowhere; this states which side they are on.
    """
    digested = _digested_functions()
    assert render_field_examples in digested
    assert _address_free in digested

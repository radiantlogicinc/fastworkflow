"""A required argument is a plain required field (``name: str``, no default) (fix-3rsq.plain-required).

The developer writes no sentinel. The framework fills a missing required field with its
not-given marker (``NOT_FOUND`` for a str, ``INVALID_INT_VALUE`` / ``INVALID_FLOAT_VALUE`` for a
number, ``[]`` for a list) and asks the user for it, as it does for ``default="NOT_FOUND"``.
``default="NOT_FOUND"`` still works, and logs one WARNING at load.

Integration throughout: a real fixture workflow (``tests/plain_required_workflow``) with an
execution command, a planning command and a deprecated command, the real routing, the real
``CommandExecutor.perform_action``, the real ``ParameterExtraction``, the real validator. The only
double is the LLM extractor, and that is patched at the seam a live model would answer.
"""

from __future__ import annotations

import contextlib
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional
from unittest.mock import patch

import pytest
from pydantic import BaseModel, Field

import fastworkflow
from fastworkflow import ModuleType, planning_commands
from fastworkflow._workflows.command_metadata_extraction.parameter_extraction import (
    ParameterExtraction,
)
from fastworkflow.command_executor import CommandExecutor
from fastworkflow.command_metadata_api import CommandMetadataAPI
from fastworkflow.command_routing import RoutingDefinition
from fastworkflow.utils import signatures
from fastworkflow.utils.signatures import (
    INVALID_FLOAT_VALUE,
    INVALID_INT_VALUE,
    NOT_FOUND,
    InputForParamExtraction,
    not_given_value,
)

FOLDER = str(Path(__file__).parent.joinpath("plain_required_workflow").resolve())
RECORD = "record_values"
OLD = "old_style_values"
PLAN = "plan_values"
ALL_VALUES = {"name": "Bob", "count": 3, "ratio": 1.5, "ids": ["a", "b"]}
PLAIN_FIELDS = ("name", "count", "ratio", "ids")


@pytest.fixture(autouse=True)
def _framework(tmp_path):
    fastworkflow.init({"FASTWORKFLOW_STATE_ROOT": str(tmp_path / "workflow_contexts")})
    fastworkflow.RoutingRegistry.clear_registry()
    yield
    fastworkflow.RoutingRegistry.clear_registry()


@pytest.fixture
def workflow():
    fastworkflow.RoutingRegistry.get_definition(FOLDER, load_cached=False)
    return fastworkflow.Workflow.create(FOLDER, workflow_id_str=f"plain_{uuid.uuid4().hex[:8]}")


def _action(command_name: str, parameters: dict):
    """An Action as a direct caller builds it. ``Action.parameters`` is typed as scalars, so a
    list value is set without validation (the executor validates the parameters itself)."""
    return fastworkflow.Action.model_construct(
        command_name=command_name, command=command_name, parameters=parameters, workflow_id=None)


def _params_class(command_name: str):
    definition = fastworkflow.RoutingRegistry.get_definition(FOLDER)
    return definition.get_command_class(command_name, ModuleType.COMMAND_PARAMETERS_CLASS)


def _validate(wf, command_name: str, params):
    """(is_valid, message, missing-or-invalid field names)."""
    extraction = InputForParamExtraction(command=command_name)
    ok, message, _, fields = extraction.validate_parameters(wf, command_name, params)
    return ok, message, fields


# ---------------------------------------------------------------------------
# The shared helper: what a field holds when nothing gives it a value
# ---------------------------------------------------------------------------

class _Shapes(BaseModel):
    text: str = Field(description="text")
    number: int = Field(description="number")
    ratio: float = Field(description="ratio")
    items: List[str] = Field(description="items")
    maybe: Optional[str] = Field(default=None, description="optional text")
    maybe_unset: Optional[str] = Field(description="optional text, no default")
    kept: str = Field(default="kept", description="declared default")


@pytest.mark.parametrize("field, expected", [
    ("text", NOT_FOUND),
    ("number", INVALID_INT_VALUE),
    ("ratio", INVALID_FLOAT_VALUE),
    ("items", []),
    ("maybe", None),
    ("maybe_unset", None),
    ("kept", "kept"),
])
def test_not_given_value_per_field(field, expected):
    assert not_given_value(_Shapes.model_fields[field]) == expected


def test_a_plain_required_field_has_no_default_to_write():
    """The developer's spelling carries no default, so the framework cannot be taught one."""
    for name in PLAIN_FIELDS:
        assert _params_class(RECORD).model_fields[name].is_required()


# ---------------------------------------------------------------------------
# Execution command, direct action path (MCP / REST / "perform_action")
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("missing", PLAIN_FIELDS)
def test_direct_action_missing_plain_required_field_is_asked_not_raised(workflow, missing):
    given = {k: v for k, v in ALL_VALUES.items() if k != missing}
    action = _action(RECORD, given)

    with pytest.raises(ValueError) as exc_info:
        CommandExecutor.perform_action(workflow, action)

    message = str(exc_info.value)
    assert "Missing parameter values" in message
    assert missing in message


def test_direct_action_with_no_parameters_asks_for_every_plain_required_field(workflow):
    action = _action(RECORD, {})

    with pytest.raises(ValueError) as exc_info:
        CommandExecutor.perform_action(workflow, action)

    for name in PLAIN_FIELDS:
        assert name in str(exc_info.value)


def test_direct_action_with_every_plain_required_field_runs(workflow):
    action = _action(RECORD, ALL_VALUES)

    result = CommandExecutor.perform_action(workflow, action)

    assert result.success is True
    assert "count=3" in result.command_response.response


# ---------------------------------------------------------------------------
# Execution command, extraction path (the LLM answers without some values)
# ---------------------------------------------------------------------------

@pytest.fixture
def llm_answers(workflow):
    """Patch the LLM extractor at its seam: it returns the values it is given, unvalidated."""
    state = {"values": {}}

    def fake_llm(self, cls, command_name, folder):
        return cls.model_construct(**state["values"])

    with patch.object(InputForParamExtraction, "extract_parameters", fake_llm):
        def run(command_name: str, values: dict, command_text: str):
            state["values"] = values
            cme = SimpleNamespace(context={})
            extraction = ParameterExtraction(cme, workflow, command_name, command_text)
            output = extraction._extract_impl({})
            return output, cme
        yield run


def test_llm_leaving_plain_required_fields_out_asks_for_them(llm_answers):
    output, cme = llm_answers(RECORD, {"name": "Bob"}, f"{RECORD} for Bob")

    assert output.parameters_are_valid is False
    for name in ("count", "ratio", "ids"):
        assert name in output.error_msg
    stored = cme.context["stored_parameters"]
    assert stored.name == "Bob"
    for name in ("count", "ratio", "ids"):
        field = _params_class(RECORD).model_fields[name]
        assert ParameterExtraction._is_asked_field(field, getattr(stored, name)), name


def test_llm_answering_every_plain_required_field_is_valid(llm_answers):
    # The request writes the str value: the argument checks (fix-qjf4) refuse one it does not write.
    output, _ = llm_answers(RECORD, dict(ALL_VALUES), f"{RECORD} for Bob with all values")

    assert output.parameters_are_valid is True
    assert output.cmd_parameters.count == 3


def test_dspy_result_missing_a_plain_required_field_is_asked_not_passed_on(workflow, monkeypatch):
    """The DSPy path reads each field with ``getattr(result, name, default)``. A plain required
    field has no declared default, so the default must be the not-given marker, not pydantic's
    ``PydanticUndefined`` (which would pass validation as if the user had given a value)."""
    class _FakeBestOfN:
        def __init__(self, **_kwargs):
            pass

        def __call__(self, command=None):
            return SimpleNamespace(name="Bob")  # the model answered only for name

    class _FakeOptimizer:
        def __init__(self, **_kwargs):
            pass

        def compile(self, student, trainset):
            return student

    monkeypatch.setattr(signatures, "get_trainset", lambda *_a, **_k: [])
    monkeypatch.setattr(signatures.dspy_utils, "get_lm", lambda *_a, **_k: None)
    monkeypatch.setattr(signatures.dspy, "LabeledFewShot", _FakeOptimizer)
    monkeypatch.setattr(signatures.dspy, "BestOfN", _FakeBestOfN)
    monkeypatch.setattr(signatures.dspy, "context", lambda **_k: contextlib.nullcontext())

    params = InputForParamExtraction.extract_parameters(
        InputForParamExtraction(command="record for Bob"), _params_class(RECORD), RECORD, FOLDER)

    assert params.name == "Bob"
    assert params.count == INVALID_INT_VALUE
    assert params.ratio == INVALID_FLOAT_VALUE
    assert params.ids == []
    ok, message, fields = _validate(workflow, RECORD, params)
    assert ok is False
    assert set(fields) >= {"count", "ratio", "ids"}, message


# ---------------------------------------------------------------------------
# "/" deterministic key=value path
# ---------------------------------------------------------------------------

def test_key_value_text_leaving_plain_required_fields_out_is_asked(workflow):
    params = ParameterExtraction._extract_parameters_from_key_values(
        f"{RECORD} name=Bob count=3", _params_class(RECORD), RECORD)

    assert params is not None
    ok, _, fields = _validate(workflow, RECORD, params)
    assert ok is False
    assert set(fields) >= {"ratio", "ids"}


# ---------------------------------------------------------------------------
# Planning command: the same plain required fields, asked for as a refusal
# ---------------------------------------------------------------------------

def test_planning_command_plain_required_fields_are_asked_when_missing(workflow):
    planning = planning_commands.planning_command(FOLDER, PLAN)
    assert planning is not None and planning.input_model is not None
    params = planning.input_model.model_construct(name="Bob")

    ok, message, fields = _validate(workflow, PLAN, params)

    assert ok is False
    assert set(fields) >= {"count", "ratio", "ids"}, message


def test_planning_command_has_no_not_found_default(workflow):
    planning = planning_commands.planning_command(FOLDER, PLAN)
    for param in planning.params:
        assert param.required, param.name
        assert param.default != NOT_FOUND


# ---------------------------------------------------------------------------
# Command metadata (MCP tool schemas read "required" from here)
# ---------------------------------------------------------------------------

def test_metadata_marks_plain_required_fields_required(workflow):
    inputs = CommandMetadataAPI.get_params_for_all_commands(FOLDER)[RECORD]["inputs"]
    by_name = {p["name"]: p for p in inputs}

    for name in PLAIN_FIELDS:
        assert by_name[name].get("required") is True, name


# ---------------------------------------------------------------------------
# Deprecation: default="NOT_FOUND" still works and warns once at load
# ---------------------------------------------------------------------------

@pytest.fixture
def fresh_load(workflow, monkeypatch):
    """Load command classes afresh so the load-time warning is observed."""
    monkeypatch.setattr(RoutingDefinition, "_GLOBAL_COMMAND_CLASS_CACHE", {})
    warnings: list[str] = []
    definition = fastworkflow.RoutingRegistry.get_definition(FOLDER)
    monkeypatch.setattr(fastworkflow.command_routing.logger, "warning",
                        lambda msg, *a, **k: warnings.append(str(msg)))

    def load(command_name: str):
        return definition.get_command_class(
            command_name, ModuleType.INPUT_FOR_PARAM_EXTRACTION_CLASS)
    yield load, warnings


def test_default_not_found_logs_one_deprecation_warning_naming_command_and_field(fresh_load):
    load, warnings = fresh_load
    load(OLD)

    deprecated = [w for w in warnings if "NOT_FOUND" in w]
    assert len(deprecated) == 1, warnings
    assert OLD in deprecated[0]
    assert "name" in deprecated[0]
    assert "write a plain required field (no default)" in deprecated[0]


def test_plain_required_fields_do_not_warn(fresh_load):
    load, warnings = fresh_load
    load(RECORD)
    load(PLAN)

    assert [w for w in warnings if "NOT_FOUND" in w or "plain required" in w] == []


def test_default_not_found_still_asks_the_user_as_before(workflow):
    action = _action(OLD, {})

    with pytest.raises(ValueError) as exc_info:
        CommandExecutor.perform_action(workflow, action)

    assert "name" in str(exc_info.value)


# ---------------------------------------------------------------------------
# A backslash escape of punctuation in a key=value value (fix-variance)
# ---------------------------------------------------------------------------

class _LookupParams(BaseModel):
    account: str = Field(description="the account to look up")


@pytest.mark.parametrize("text, value", [
    (r"find_account account=ann\@example.org", "ann@example.org"),           # a model's escaped @
    (r"find_account account=ann\_lee", "ann_lee"),                           # a model's escaped _
    (r'find_account account="ann\@example.org"', "ann@example.org"),         # escaped inside quotes
    (r"find_account account=C:\new\path", "C:\\new\\path"),                  # letters after a backslash: kept
    (r"find_account account=a\\b", "a\\\\b"),                                # a doubled backslash: kept
    (r"find_account account=ann\ lee", "ann\\ lee"),                         # a backslash before a space: kept
])
def test_a_key_value_lookup_reads_escaped_punctuation_as_the_character(text, value):
    params = ParameterExtraction._extract_parameters_from_key_values(text, _LookupParams, "find_account")

    assert params is not None
    assert params.account == value

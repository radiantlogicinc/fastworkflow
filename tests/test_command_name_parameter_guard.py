"""A command's name written as a parameter value is treated as not given.

An extractor copies a command's name into a field (a skill's own name as its
``application``, another command's name as a lookup ``query``). The guard clears such a
value so the normal missing message is produced, with wording that names the command.
Matching is on the identifier form only, so an entity written with spaces is a value.

Integration: the real ``direct_action_validation_workflow`` (commands ``check_ready`` and
``use_payload``, plus the built-in ``IntentDetection`` commands) and the real
``validate_parameters``. No mocks.
"""

from __future__ import annotations

import uuid
from enum import Enum
from pathlib import Path
from typing import List, Literal, Optional

import pytest
from pydantic import BaseModel, Field

import fastworkflow
from fastworkflow.utils import signatures
from fastworkflow.utils.signatures import InputForParamExtraction, command_name_index

WORKFLOW_PATH = str(
    Path(__file__).parent.joinpath("direct_action_validation_workflow").resolve()
)


@pytest.fixture(autouse=True)
def _error_messages(monkeypatch):
    monkeypatch.setattr(signatures, "MISSING_INFORMATION_ERRMSG", "Missing parameter values: ")
    monkeypatch.setattr(signatures, "INVALID_INFORMATION_ERRMSG", "Invalid parameter values: ")
    monkeypatch.setattr(signatures, "NOT_FOUND", "NOT_FOUND")


@pytest.fixture
def workflow(tmp_path):
    fastworkflow.init({"FASTWORKFLOW_STATE_ROOT": str(tmp_path / "workflow_contexts")})
    fastworkflow.RoutingRegistry.clear_registry()
    fastworkflow.RoutingRegistry.get_definition(WORKFLOW_PATH, load_cached=False)
    app_workflow = fastworkflow.Workflow.create(
        WORKFLOW_PATH, workflow_id_str=f"cnpg_{uuid.uuid4().hex[:8]}")
    yield app_workflow
    fastworkflow.RoutingRegistry.clear_registry()


class NoteParams(BaseModel):
    note: str = Field(description="A note to attach to the payload")


class OptionalNoteParams(BaseModel):
    note: Optional[str] = Field(default=None, description="A note to attach")


class NamesParams(BaseModel):
    names: List[str] = Field(description="names")


class KindParams(BaseModel):
    kind: Literal["check_ready", "person"] = Field(description="kind")


class Kind(str, Enum):
    CHECK_READY = "check_ready"
    PERSON = "person"


class EnumParams(BaseModel):
    kind: Kind = Field(description="kind")


def _validate(app_workflow, params, subject="use_payload", diagnostics=None):
    extraction = InputForParamExtraction(
        command="use the payload", input_for_param_extraction_class=None)
    return extraction.validate_parameters(app_workflow, subject, params, diagnostics=diagnostics)


# --- the name index comes from the real workflow ----------------------------------------

def test_the_index_holds_the_workflows_command_names(workflow):
    index = command_name_index(workflow, "use_payload")
    assert index["check_ready"] == "check_ready"
    assert index["use_payload"] == "use_payload"
    assert index["intentdetection/go_up"] == "IntentDetection/go_up"
    assert index["go_up"] == "go_up"


def test_a_name_written_with_spaces_is_not_in_the_index(workflow):
    index = command_name_index(workflow, "use_payload")
    assert "use payload" not in index
    assert "tag record" not in index


# --- a command's own name --------------------------------------------------------------

def test_a_required_field_holding_the_called_commands_name_is_missing(workflow):
    diagnostics = {}
    is_valid, message, _, fields = _validate(
        workflow, NoteParams(note="use_payload"), diagnostics=diagnostics)

    assert not is_valid and fields == ["note"]
    assert "Missing parameter values: note" in message
    assert "'use_payload' is the name of this command, not a value" in message
    assert diagnostics["command_name_values"] == {"note": "use_payload"}


def test_the_called_commands_name_is_refused_in_any_case(workflow):
    for value in ("USE_PAYLOAD", "Use_Payload"):
        is_valid, message, _, fields = _validate(workflow, NoteParams(note=value))
        assert not is_valid and fields == ["note"], value
        assert "name of this command" in message, value


def test_an_optional_field_holding_the_called_commands_name_takes_its_default(workflow):
    params = OptionalNoteParams(note="use_payload")
    diagnostics = {}
    is_valid, _, _, _ = _validate(workflow, params, diagnostics=diagnostics)

    assert is_valid
    assert params.note is None
    assert diagnostics["command_name_values"] == {"note": "use_payload"}


# --- another command's name ------------------------------------------------------------

def test_another_commands_name_is_refused_with_its_name(workflow):
    is_valid, message, _, fields = _validate(workflow, NoteParams(note="check_ready"))

    assert not is_valid and fields == ["note"]
    assert "note 'check_ready' is the name of the command 'check_ready', not a value" in message


def test_a_qualified_command_name_is_refused(workflow):
    is_valid, message, _, fields = _validate(workflow, NoteParams(note="IntentDetection/go_up"))

    assert not is_valid and fields == ["note"]
    assert "the name of the command 'IntentDetection/go_up'" in message


def test_a_list_item_holding_a_command_name_makes_the_whole_list_missing(workflow):
    is_valid, message, _, fields = _validate(
        workflow, NamesParams(names=["Ledger", "check_ready"]))

    assert not is_valid and fields == ["names"]
    assert "Missing parameter values: names" in message
    assert "the name of the command 'check_ready'" in message


# --- values that are not command names -----------------------------------------------

@pytest.mark.parametrize("value", ["Tag Record", "Use Payload", "remember the ledger"])
def test_a_value_written_with_spaces_or_as_text_is_kept(workflow, value):
    params = NoteParams(note=value)
    is_valid, _, _, _ = _validate(workflow, params)
    assert is_valid and params.note == value


def test_a_declared_literal_choice_is_never_refused(workflow):
    params = KindParams(kind="check_ready")
    is_valid, _, _, _ = _validate(workflow, params)
    assert is_valid and params.kind == "check_ready"


def test_a_declared_enum_choice_is_never_refused(workflow):
    is_valid, _, _, fields = _validate(workflow, EnumParams(kind="check_ready"))
    assert is_valid and fields == []

"""The one validation hook and the one refusal idiom (fix-3rsq.one-hook, owner decision 2026-11-03).

``validate_extracted_parameters`` returns None to accept, may correct values in place, and refuses
only by raising ``fastworkflow.Refused(message, field=...)``. ``db_lookup`` is retired (a load error).
A ``ResponseGenerator.__call__`` takes three arguments and returns the Output, a str or a
CommandResponse; a hand-built CommandOutput, or success=False, is a call error. Every case runs on a
real workflow (``tests/one_hook_workflow``, ``tests/messaging_app_4`` copy), with no mocks.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

import fastworkflow
from fastworkflow import ModuleType
from fastworkflow.command_executor import CommandExecutor
from fastworkflow.observability.decision_signals import SLOT_BINDING_SOURCES, SIGNAL_CONTRACT_VERSION
from fastworkflow.observability.decision_signals import slot_binding_source  # noqa: F401 - the vocabulary
from fastworkflow._workflows.command_metadata_extraction.parameter_extraction import slot_binding_uncertainty
from fastworkflow import planning_commands as pc
from fastworkflow.planning_commands import PlanningCommandInputError
from fastworkflow.utils.signatures import InputForParamExtraction
from fastworkflow import workflow_agent

from tests.test_global_first_lookup import messaging_on_a_user  # noqa: F401 - fixture (a copy of messaging_app_4)
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401 - shared fixture
from tests.test_planning_command_files import _write as write_files  # noqa: F401 - helper

WORKFLOW = Path(__file__).parent / "one_hook_workflow"


@pytest.fixture
def workflow_path() -> str:
    return str(WORKFLOW.resolve())


@pytest.fixture
def initialized(tmp_path, workflow_path: str):
    fastworkflow.init({"FASTWORKFLOW_STATE_ROOT": str(tmp_path / "workflow_contexts")})
    fastworkflow.RoutingRegistry.clear_registry()
    fastworkflow.RoutingRegistry.get_definition(workflow_path, load_cached=False)
    yield
    fastworkflow.RoutingRegistry.clear_registry()


def _workflow(workflow_path: str) -> fastworkflow.Workflow:
    return fastworkflow.Workflow.create(workflow_path, workflow_id_str=f"oh_{uuid.uuid4().hex[:8]}")


def _command_class(workflow_path: str, command: str, module_type: ModuleType):
    return fastworkflow.RoutingRegistry.get_definition(workflow_path).get_command_class(command, module_type)


def _validate(workflow, workflow_path: str, command: str, text: str, **values):
    """validate_parameters for a command with *values* as its filled Input (as the NLU path does)."""
    extractor = InputForParamExtraction.create(workflow, command, text)
    input_obj = _command_class(workflow_path, command, ModuleType.COMMAND_PARAMETERS_CLASS)(**values)
    diagnostics: dict = {}
    result = extractor.validate_parameters(workflow, command, input_obj, diagnostics=diagnostics)
    return result, input_obj, diagnostics


# ---------------------------------------------------------------- Refused(field=) and Refused()


def test_refused_with_field_asks_for_that_field_only_and_keeps_the_others(initialized, workflow_path):
    workflow = _workflow(workflow_path)

    (is_valid, message, _suggestions, fields), input_obj, diagnostics = _validate(
        workflow, workflow_path, "pick_seat", "pick seat B9", seat="B9", note="window side")

    assert is_valid is False
    # the named field is the one the framework resets and asks for; the note is kept
    assert fields == ["seat"]
    assert diagnostics["invalid_fields"] == ["seat"]
    assert "seat B9 is taken; pick another" in message
    assert "Provide corrected parameter values in the exact order specified below, separated by commas:\nseat" in message
    assert "note" not in message.split("Provide corrected parameter values")[1]
    assert input_obj.note == "window side"


def test_refused_with_field_on_a_planning_command_asks_for_that_argument(tmp_path):
    text = '''"""pick_room: pick a room, for the test."""
from pydantic import BaseModel, Field

import fastworkflow
from fastworkflow.planning import run


class Signature:
    """Pick a room."""

    template = ["say the room"]

    class Input(BaseModel):
        room: str = Field(description="the room")

    @staticmethod
    def validate_extracted_parameters(workflow, command, cmd_parameters):
        if cmd_parameters.room == "attic":
            raise fastworkflow.Refused("the attic is closed", field="room")


class PlanGenerator:
    def steps(self, input):
        return ["say the room"]
'''
    folder = write_files(tmp_path, {"pick_room": text})
    command = workflow_agent.load_planning_command_map(folder)["pick_room"]

    with pytest.raises(PlanningCommandInputError, match="the attic is closed") as refused:
        pc._run_validate(command, {"room": "attic"}, "pick the attic", None)
    assert refused.value.fields == ("room",)
    question = pc.refusal_question(command, refused.value)
    assert "What should the room be?" in question


def test_refused_without_field_refuses_the_call_and_asks_for_nothing(initialized, workflow_path):
    workflow = _workflow(workflow_path)

    (is_valid, message, _suggestions, fields), _input, _diagnostics = _validate(
        workflow, workflow_path, "strict_check", "start 5 at once", count=5)

    assert is_valid is False
    assert fields == []
    assert message == "no more than 2 can start at once"


def test_refused_without_field_is_the_call_answer_with_success_false(initialized, workflow_path):
    workflow = _workflow(workflow_path)
    from tests.one_hook_workflow._commands.refuse import REFUSAL

    out = CommandExecutor.perform_action(
        workflow, fastworkflow.Action(command_name="refuse", command="open the vault", parameters={"word": "x"}))

    assert out.command_response.success is False
    assert out.command_response.response == REFUSAL


def test_refused_naming_an_unknown_field_is_an_error(tmp_path):
    text = '''"""ghost: names a field its Input does not have, for the test."""
from pydantic import BaseModel, Field

import fastworkflow
from fastworkflow.planning import run


class Signature:
    """Ghost."""

    template = ["say ghost"]

    class Input(BaseModel):
        room: str = Field(description="the room")

    @staticmethod
    def validate_extracted_parameters(workflow, command, cmd_parameters):
        raise fastworkflow.Refused("no such thing", field="nowhere")


class PlanGenerator:
    def steps(self, input):
        return ["say ghost"]
'''
    folder = write_files(tmp_path, {"ghost": text})
    command = workflow_agent.load_planning_command_map(folder)["ghost"]
    with pytest.raises(PlanningCommandInputError, match="no such thing") as refused:
        pc._run_validate(command, {"room": "x"}, "x", None)
    assert refused.value.fields == ("nowhere",)


# ---------------------------------------------------------------- the hook's return value


def test_a_hook_that_returns_a_tuple_fails_fast_naming_the_form(initialized, workflow_path):
    workflow = _workflow(workflow_path)

    with pytest.raises(TypeError) as exc_info:
        _validate(workflow, workflow_path, "tuple_hook", "old hook", name="x")

    message = str(exc_info.value)
    assert "must return None to accept" in message
    assert "raise fastworkflow.Refused(message, field=...)" in message
    assert "it returned tuple" in message


def test_a_planning_hook_that_returns_a_tuple_fails_fast(tmp_path):
    text = '''"""weigh_group: weigh one group, for the test."""
from pydantic import BaseModel, Field


class Signature:
    """Weigh one group."""

    template = ["say the weight"]

    class Input(BaseModel):
        name: str = Field(description="the group")

    @staticmethod
    def validate_extracted_parameters(workflow, command, cmd_parameters, command_text=None):
        return (False, "old form")


class PlanGenerator:
    def steps(self, input):
        return ["say the weight"]
'''
    folder = write_files(tmp_path, {"weigh_group": text})
    # the loader runs the hook on the template grid, so the old form fails at load, naming the form
    with pytest.raises(TypeError, match="must return None to accept and raise fastworkflow.Refused"):
        workflow_agent.load_planning_command_map(folder)


# ---------------------------------------------------------------- correction, write-back, observability


def test_a_hook_that_corrects_a_value_writes_it_back_and_records_a_validation_hook_binding(
        initialized, workflow_path):
    workflow = _workflow(workflow_path)

    (is_valid, message, _suggestions, fields), input_obj, diagnostics = _validate(
        workflow, workflow_path, "fix_color", "paint it Red", color="Red")

    assert (is_valid, fields) == (True, [])
    assert input_obj.color == "red"
    assert diagnostics["validation_hook"]["changed_fields"] == ["color"]
    assert diagnostics["validation_hook"]["is_valid"] is True

    # the slot is bound by the hook, where db_lookup used to be recorded
    uncertainty = slot_binding_uncertainty({"extraction_method": "xml_regex", **diagnostics})
    values = [signal.value for signal in uncertainty.signals if signal.kind == "slot-binding-source"]
    assert values == ["xml_regex", "validation_hook"]
    assert "validation_hook" in SLOT_BINDING_SOURCES
    assert "db_lookup" not in SLOT_BINDING_SOURCES
    assert SIGNAL_CONTRACT_VERSION == 2


def test_a_hook_that_leaves_values_alone_records_no_validation_hook_binding(initialized, workflow_path):
    workflow = _workflow(workflow_path)

    (is_valid, _m, _s, _f), _input, diagnostics = _validate(
        workflow, workflow_path, "fix_color", "paint it red", color="red")

    assert is_valid is True
    assert diagnostics["validation_hook"]["changed_fields"] == []
    uncertainty = slot_binding_uncertainty({"extraction_method": "xml_regex", **diagnostics})
    assert [signal.value for signal in uncertainty.signals if signal.kind == "slot-binding-source"] == ["xml_regex"]


# ---------------------------------------------------------------- db_lookup is retired


def test_a_db_lookup_json_schema_key_is_a_load_error_naming_the_hook(initialized, workflow_path):
    with pytest.raises(ValueError, match=r"db_lookup is retired: do the lookup in validate_extracted_parameters"):
        _command_class(workflow_path, "legacy_lookup", ModuleType.INPUT_FOR_PARAM_EXTRACTION_CLASS)


def test_a_signature_db_lookup_method_is_a_load_error_naming_the_hook(initialized, workflow_path):
    with pytest.raises(ValueError, match=r"Signature\.db_lookup: db_lookup is retired"):
        _command_class(workflow_path, "legacy_method", ModuleType.INPUT_FOR_PARAM_EXTRACTION_CLASS)


# ---------------------------------------------------------------- the three-argument __call__ and the short returns


def test_a_two_argument_call_is_a_load_error_naming_the_form(initialized, workflow_path):
    with pytest.raises(ValueError, match=r"must take three arguments, \(self, workflow, command, input\)"):
        _command_class(workflow_path, "two_arg", ModuleType.RESPONSE_GENERATION_INFERENCE)


def test_a_hand_built_commandoutput_return_is_a_call_error_naming_the_short_forms(initialized, workflow_path):
    workflow = _workflow(workflow_path)

    with pytest.raises(TypeError) as exc_info:
        CommandExecutor.perform_action(
            workflow, fastworkflow.Action(command_name="envelope", command="hand built", parameters={"word": "x"}))

    message = str(exc_info.value)
    assert message == ("envelope declares an Output but returned CommandOutput; return the Output, "
                       "or CommandResponse(response=..., output=<Output>)")


def test_a_commandresponse_with_success_false_is_a_call_error(initialized, workflow_path):
    workflow = _workflow(workflow_path)

    with pytest.raises(TypeError, match="success=False"):
        CommandExecutor.perform_action(
            workflow, fastworkflow.Action(command_name="refused_response", command="failed", parameters={"word": "x"}))


def test_the_output_return_is_the_response_and_the_framework_sets_success(initialized, workflow_path):
    workflow = _workflow(workflow_path)

    out = CommandExecutor.perform_action(
        workflow, fastworkflow.Action(command_name="pick_seat", command="pick seat A1",
                                      parameters={"seat": "A1", "note": "aisle"}))

    assert out.command_response.success is True
    assert out.command_response.response.startswith("{")
    assert '"seat":"A1"' in out.command_response.response.replace(" ", "")


# ---------------------------------------------------------------- messaging_app_4: the lookup moved into its hook


def test_messaging_app_4_set_current_user_corrects_a_near_name_and_refuses_one_it_cannot_place(
        messaging_on_a_user):
    app, _cme, root, _user = messaging_on_a_user
    from fastworkflow.examples.messaging_app_4.application.user import User

    root.add_user(User(root, "fred"))
    # the NLU path binds the record the command runs on (the chat room) before the hook runs
    app.command_context_for_response_generation = root
    command = "ChatRoom/set_current_user"
    params_cls = fastworkflow.RoutingRegistry.get_definition(app.folderpath).get_command_class(
        command, ModuleType.COMMAND_PARAMETERS_CLASS)

    # a near name is corrected to the user's name, written back to the Input
    extractor = InputForParamExtraction.create(app, command, "set fredd as the current user")
    near = params_cls(user_name="fredd")
    (is_valid, _message, _suggestions, fields) = extractor.validate_parameters(app, command, near)
    assert (is_valid, fields) == (True, [])
    assert near.user_name == "fred"

    # a name close to one in the chat room, but not a match, is refused by field with the suggestion
    refused = params_cls(user_name="frd")
    (is_valid, message, _suggestions, fields) = extractor.validate_parameters(app, command, refused)
    assert is_valid is False
    assert fields == ["user_name"]
    assert "Did you mean" in message

    # a name with no close match is declined by the hook (left as typed, accepted), as db_lookup did
    unknown = params_cls(user_name="qzxv")
    (is_valid, _message, _suggestions, fields) = extractor.validate_parameters(app, command, unknown)
    assert (is_valid, fields) == (True, [])
    assert unknown.user_name == "qzxv"


def test_messaging_app_4_set_current_user_refuses_a_near_tie_with_field_and_both_names(messaging_on_a_user):
    """fix-jd9o.10: 'Sarah Coen' is near both users, so the hook refuses by field with both names and
    sets nothing; the field is then asked again rather than silently taking one of them."""
    app, _cme, root, _user = messaging_on_a_user
    from fastworkflow.examples.messaging_app_4.application.user import User

    root.add_user(User(root, "Sarah Chen"))
    root.add_user(User(root, "Sarah Cohen"))
    app.command_context_for_response_generation = root
    command = "ChatRoom/set_current_user"
    params_cls = fastworkflow.RoutingRegistry.get_definition(app.folderpath).get_command_class(
        command, ModuleType.COMMAND_PARAMETERS_CLASS)

    extractor = InputForParamExtraction.create(app, command, "set sarah coen as the current user")
    tied = params_cls(user_name="Sarah Coen")
    (is_valid, message, _suggestions, fields) = extractor.validate_parameters(app, command, tied)

    assert is_valid is False
    assert fields == ["user_name"]
    assert "Sarah Cohen" in message and "Sarah Chen" in message
    assert "Did you mean" in message
    assert tied.user_name == "Sarah Coen"   # nothing was written back

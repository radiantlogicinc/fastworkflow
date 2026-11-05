"""A planning command's `Literal` input takes its member values, as a routed enum does.

The defect: `InputForParamExtraction.validate_parameters` (fastworkflow/utils/signatures.py)
checks each field's value against its annotation's concrete types. It handles `Enum`
subclasses, but a skill's `Literal["person", "login", ...]` input is `typing.Literal`, which
is neither a type nor an Enum, so no candidate type matches and every value is rejected,
members included. The live `inspect_entity` got `kind 'login'` rejected with
`typing.Literal['person', 'login', ...]` as the expected type.

Every test here is integration: a real library copy, its real `RoutingDefinition`, the real
skill loader (`planning_commands.planning_command`) and the real `validate_parameters` and
`CommandExecutor`. The only double is the classifier, which raises if it is ever consulted.
"""

import uuid
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, Field

import fastworkflow
from fastworkflow import planning_commands
from fastworkflow._workflows.command_metadata_extraction import intent_detection
from fastworkflow.command_executor import CommandExecutor
from fastworkflow.command_routing import RoutingRegistry
from fastworkflow.utils.signatures import InputForParamExtraction
from fastworkflow.workflow_execution_context import WorkflowExecutionContext
from tests.test_intent_detection_foreign_context import (
    _refusing_router as refusing_router,
)
from tests.test_step_helpers import LibraryRunner, _library_workflow
from tests.test_turn_result_capture import (  # noqa: F401 - shared fixture
    initialized_fastworkflow,
)

ROOT_PLANNING = "inspect_entity"
KIND_MEMBERS = ("person", "login", "right", "application", "feed", "group", "unit")
CONNECTED_MEMBERS = ("none", "login", "right", "person")

def _literal_args(members: tuple[str, ...]) -> str:
    return ", ".join(repr(m) for m in members)


#: A root planning command with the live `inspect_entity`'s two Literal inputs, `kind` and
#: `connected`. Its one phase is a guard (`do_not_run`), so no nested command runs.
INSPECT_ENTITY_SOURCE = f'''"""inspect_entity: the entity the request names, as one planned step."""
from typing import Literal

from pydantic import BaseModel, Field

from fastworkflow.planning import do_not_run


class Signature:
    """Inspect the entity the request names."""

    class Input(BaseModel):
        name: str = Field(description="the name exactly as the request writes it",
                          json_schema_extra={{"in_request": True}})
        kind: Literal[{_literal_args(KIND_MEMBERS)}] = Field(description="what the name is")
        connected: Literal[{_literal_args(CONNECTED_MEMBERS)}] = Field(
            default="none", description="one connected record to open")
        control_is: Literal["code", "name"] = Field(
            default="name", description="code when the request writes the control's code; name "
                                        "when it writes the control's title")


class PlanGenerator:
    def check(self, input):
        return [do_not_run("library_hours")]

    def __result__(self, input):
        return f"Inspected {{input.name}} as {{input.kind}}."
'''


class RoutedKind(str, Enum):
    """The enum a routed command declares for the same choice; the reference for the text."""

    person = "person"
    login = "login"
    right = "right"
    application = "application"
    feed = "feed"
    group = "group"
    unit = "unit"


class RoutedKindParams(BaseModel):
    kind: RoutedKind = Field(description="what the name is")


@pytest.fixture
def literal_env(tmp_path, initialized_fastworkflow, monkeypatch):
    """A library copy with a root planning command that has Literal inputs; the classifier is
    a double that raises if consulted."""
    folder = _library_workflow(tmp_path)
    Path(folder, "_commands", f"{ROOT_PLANNING}.py").write_text(
        INSPECT_ENTITY_SOURCE, encoding="utf-8")
    RoutingRegistry.clear_registry()
    RoutingRegistry.get_definition(folder, load_cached=False)
    wf = fastworkflow.Workflow.create(folder, workflow_id_str=f"pl_{uuid.uuid4().hex[:8]}")
    ctx = WorkflowExecutionContext(run_as_agent=True)
    ctx.bind_app_workflow(wf)
    ctx.push_active_workflow(wf)
    LibraryRunner(wf)(ctx, "startup")
    consulted: list[str] = []
    monkeypatch.setattr(intent_detection, "CommandRouter", refusing_router(consulted))
    yield SimpleNamespace(ctx=ctx, wf=wf, folder=folder, consulted=consulted)
    ctx.pop_active_workflow()
    RoutingRegistry.clear_registry()


def _planning_params(env, **values):
    """The planning command's own Input model holding *values*, built as extraction builds it
    (``model_construct``, no validation: the check under test is the one that runs later)."""
    planning = planning_commands.planning_command(env.folder, ROOT_PLANNING)
    assert planning is not None and planning.input_model is not None
    return planning.input_model.model_construct(name="Jesse Rodgers-ADM", **values)


def _validate(env, subject: str, params):
    """The real validation a turn runs on extracted arguments."""
    extraction = InputForParamExtraction(command=f"{ROOT_PLANNING} Jesse Rodgers-ADM")
    return extraction.validate_parameters(env.wf, subject, params)


# ---------------------------------------------------------------------------
# A planning command's Literal input: members pass, non-members are refused as a routed
# enum's are
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", KIND_MEMBERS)
def test_a_planning_literal_accepts_each_of_its_member_values(literal_env, kind):
    params = _planning_params(literal_env, kind=kind, connected="none")

    is_valid, message, _, invalid = _validate(literal_env, ROOT_PLANNING, params)

    assert is_valid is True, message
    assert invalid == []
    assert params.kind == kind


@pytest.mark.parametrize("connected", CONNECTED_MEMBERS)
def test_a_planning_literal_accepts_each_connected_member_value(literal_env, connected):
    params = _planning_params(literal_env, kind="login", connected=connected)

    is_valid, message, _, _ = _validate(literal_env, ROOT_PLANNING, params)

    assert is_valid is True, message
    assert params.connected == connected


def test_a_planning_literal_refuses_a_non_member_with_the_routed_enum_text(literal_env):
    """The refusal text of a `Literal` input is the text a routed enum's refusal has."""
    planning_params = _planning_params(literal_env, kind="robot", connected="none")
    planning_ok, planning_msg, _, planning_invalid = _validate(
        literal_env, ROOT_PLANNING, planning_params)

    routed_params = RoutedKindParams.model_construct(kind="robot")
    routed_ok, routed_msg, _, routed_invalid = _validate(
        literal_env, "Library/find_book", routed_params)

    assert planning_ok is False and routed_ok is False
    assert planning_invalid == routed_invalid == ["kind"]
    assert planning_msg == routed_msg
    assert "Valid values: 'person', 'login', 'right', 'application', 'feed', 'group', 'unit'" \
        in planning_msg


def test_a_planning_literal_accepts_a_member_in_another_case_as_the_member(literal_env):
    """A routed enum takes a member's name in any case; a Literal takes it the same way."""
    params = _planning_params(literal_env, kind="LOGIN", connected="none")

    is_valid, message, _, _ = _validate(literal_env, ROOT_PLANNING, params)

    assert is_valid is True, message
    assert params.kind == "login"


# ---------------------------------------------------------------------------
# End to end: the real executor, key=value arguments, run_planning_command
# ---------------------------------------------------------------------------

def test_a_literal_argument_reaches_the_planning_command_as_its_member(literal_env):
    ctx = literal_env.ctx
    literal_env.wf.context["run_as_agent"] = True
    ctx._cme_workflow.context["run_as_agent"] = True
    literal_env.wf.current_command_context = None

    out = CommandExecutor.invoke_command(
        ctx, f'{ROOT_PLANNING} name="Jesse Rodgers-ADM" kind=login')

    assert literal_env.consulted == []
    assert out.success is True
    assert out.command_parameters.kind == "login"
    assert out.command_response.response == "Inspected Jesse Rodgers-ADM as login."
    assert ctx._frame_stack()[1:] == []
# ---------------------------------------------------------------------------
# A declared Literal choice that looks like a placeholder word is a value; a real placeholder
# is still refused
# ---------------------------------------------------------------------------

#: The live 2.03 shape: the request never writes the choice's word, so the screen would call
#: an extracted `code` ungrounded, unless the declared choice counts as a value.
REQUEST_WITHOUT_CHOICE_WORD = "Explain the finding on Jesse Rodgers-ADM and say how risky it is."


class RoutedChoice(str, Enum):
    """A routed command's enum whose member looks like a placeholder word."""

    code = "code"
    name = "name"


class RoutedChoiceParams(BaseModel):
    control_is: RoutedChoice = Field(description="code or name")


@pytest.fixture
def agent_request(literal_env):
    """Agent mode with the request the agent got: the placeholder screen grounds values in it."""
    literal_env.wf.context["run_as_agent"] = True
    literal_env.wf.context["raw_user_message"] = REQUEST_WITHOUT_CHOICE_WORD
    literal_env.ctx._cme_workflow.context["run_as_agent"] = True
    return literal_env


@pytest.mark.parametrize("choice", ["code", "name"])
def test_a_planning_literal_member_that_looks_like_a_placeholder_is_accepted(agent_request, choice):
    params = _planning_params(agent_request, kind="login", connected="none", control_is=choice)

    is_valid, message, _, _ = _validate(agent_request, ROOT_PLANNING, params)

    assert is_valid is True, message
    assert params.control_is == choice


def test_a_literal_member_that_looks_like_a_placeholder_runs_end_to_end(agent_request):
    ctx = agent_request.ctx
    agent_request.wf.current_command_context = None

    out = CommandExecutor.invoke_command(
        ctx, f'{ROOT_PLANNING} name="Jesse Rodgers-ADM" kind=login control_is=code')

    assert agent_request.consulted == []
    assert out.success is True, out.command_response.response
    assert out.command_parameters.control_is == "code"
    assert ctx._frame_stack()[1:] == []


@pytest.mark.parametrize("value", ["<code>", "control_is"])
def test_a_real_placeholder_for_a_literal_field_is_still_refused(agent_request, value):
    """A bracketed placeholder, and the field's own name when it is not a member: both refused."""
    params = _planning_params(agent_request, kind="login", connected="none", control_is=value)

    is_valid, message, _, _ = _validate(agent_request, ROOT_PLANNING, params)

    assert is_valid is False
    assert "placeholder" in message


def test_a_routed_enum_member_that_looks_like_a_placeholder_is_accepted(agent_request):
    """The same screen runs for a routed command's Enum field, so the same rule holds there."""
    params = RoutedChoiceParams.model_construct(control_is="code")

    is_valid, message, _, _ = _validate(agent_request, "Library/find_book", params)

    assert is_valid is True, message
    assert params.control_is == RoutedChoice.code

"""Golden parity for every public shape (arch §12.2).

§12.2 is a promise about four models: `CommandResponse`, `CommandOutput` and
`TurnOutput` stay compatible, and `TurnResult` grows only additively. A promise
nothing checks is a promise a well-meant refactor breaks silently, and the cost
lands on a reader months later — by which time the records that lost a field
have already been written and cannot be repaired.

So every shape's serialized field set is pinned here, in declaration order.
Order rather than a bare set, because "additive" is exactly the claim: a field
appended at the end grows the record, a field inserted in the middle or renamed
does not, and only the ordered pin can tell those apart.

The pins are on NAMES and STRUCTURE, never on values. No turn key, timestamp,
duration, call id or digest is asserted, because a golden test that fails
whenever the clock moves gets deleted rather than read.

Three kinds of test, each catching something the others cannot:

* **Field-set pins** catch a rename, a removal, or a non-appended addition.
* **Round trips** catch a field that serializes but cannot be read back. The
  `ask_user` role inversion [A7] gets its own, because there
  `command_parameters` holds the agent's *question* as a `str` and the response
  holds the user's *answer*; so does the typed-model-to-dict asymmetry
  `CommandOutput`'s docstring calls [A10] honesty — in memory the value is a
  workflow's Pydantic params instance, in a record it is a dict, and restore
  must accept both without either shape lying about the other.
* **Cross-version compatibility** catches the two directions that actually
  break deployments: an older record (no `command_call_id`, no
  `execution_records`, no `routing_events`) read by today's model, and a newer
  record carrying a key today's model has never heard of.

Real components throughout, per `.cursor/rules/testing_rules.mdc`: the
parameters class comes from the real `todo_list_workflow` through the
framework's own `RoutingRegistry`, and the end-to-end pin runs a real
`WorkflowExecutionContext` against a real `TodoListManager`.
"""

from __future__ import annotations

import uuid
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path

import pytest

import fastworkflow
from fastworkflow import (
    CommandOutput,
    CommandResponse,
    TurnOutput,
    TurnResult,
    TurnStatus,
    mint_turn_key,
)
from fastworkflow.command_routing import ModuleType, RoutingRegistry
from fastworkflow.workflow_execution_context import WorkflowExecutionContext

from tests.todo_list_workflow.application.todo_manager import TodoListManager

# A real, parameterless command of the real workflow — parameterless so the
# direct-action path stays off the parameter-extraction machinery the untrained
# test workflow cannot run.
LIST_COMMAND = "TodoListManager/list_todo_lists"
CREATE_COMMAND = "TodoListManager/create_todo_list"


# ----------------------------------------------------------------------
# The golden shapes
# ----------------------------------------------------------------------
#
# Written out rather than derived, so that changing the model does not change
# the expectation in the same edit. A pin computed from the thing it pins
# reports agreement with itself.

COMMAND_RESPONSE_FIELDS = (
    "response",
    "success",
    "artifacts",
    "next_actions",
    "recommendations",
    # Appended by fix-hia1.1, optional with a None default: the command's
    # Signature.Output, archived as JSON beside the observation text.
    "output",
)

COMMAND_OUTPUT_FIELDS = (
    "command_response",
    "workflow_name",
    "context",
    "command_name",
    "command_parameters",
    "started_at",
    "duration_ms",
    # Appended by fix-ajv.3 (arch §12.0 delta 1), optional with a default.
    "command_call_id",
    # Appended by fix-ajv.17, optional with a default, and APPENDED rather than
    # inserted for the same §12.2 reason the TurnResult pin spells out: the
    # claim is that everything before it is untouched. `is_ask_user` used to be
    # `command_name == "ask_user"`, which fix-ajv.16 made forgeable — a failure
    # output carries the real routed name with success=False, and root-context
    # names are unqualified, so a workflow defining a root command called
    # `ask_user` would have its failures collected as unanswered questions.
    "ask_user_entry",
)

# `success` is a computed field: absent from `model_fields`, present in every
# dump. Both facts are pinned, because a consumer reads the dump and a
# maintainer reads the model.
TURN_OUTPUT_FIELDS = (
    "turn_key",
    "status",
    "failure_reason",
    "answer",
    "command_outputs",
)
TURN_OUTPUT_DUMP_FIELDS = TURN_OUTPUT_FIELDS + ("success",)

TURN_RESULT_FIELDS = (
    "turn_output",
    "channel_id",
    "conversation_id",
    "ordinal",
    "user_message",
    "refined_user_message",
    "conversation_summary",
    "conversation_traces",
    "entry_workflow_name",
    "entry_context",
    "continuation_of",
    "trajectory_ref",
    "started_at",
    "completed_at",
    "suspended_ms",
    "metadata",
    # The experiment container's labels (fix-bn1), appended on the 3.3 branch
    # before change 1's pair was ported onto it; that is the shipped order.
    "experiment_id",
    "task_id",
    "attempt",
    "claim_epoch",
    "server_incarnation",
    # §12.2's additive pair, appended last (fix-ajv.5).
    "execution_records",
    "routing_events",
)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def todo_workflow_path() -> str:
    return str(Path(__file__).parent.joinpath("todo_list_workflow").resolve())


@pytest.fixture
def initialized_fastworkflow():
    fastworkflow.init({})
    RoutingRegistry.clear_registry()
    yield
    RoutingRegistry.clear_registry()


@pytest.fixture
def real_params_class(initialized_fastworkflow, todo_workflow_path):
    """The real parameters model of a real command, resolved the way the
    framework resolves it — `CommandExecutor.perform_action` calls exactly this
    to build the typed instance that lands on `CommandOutput`."""
    routing_definition = RoutingRegistry.get_definition(todo_workflow_path)
    params_class = routing_definition.get_command_class(
        CREATE_COMMAND, ModuleType.COMMAND_PARAMETERS_CLASS
    )
    assert params_class is not None, "the real workflow lost its parameters class"
    return params_class


def _command_output(**overrides) -> CommandOutput:
    kwargs = dict(
        command_response=CommandResponse(
            response="3 lists", artifacts={"client_key": "a,b\n1,2"}
        ),
        workflow_name="todo_list_workflow",
        context="TodoListManager",
        command_name=LIST_COMMAND,
        started_at=datetime.now(timezone.utc),
        duration_ms=7,
    )
    kwargs.update(overrides)
    return CommandOutput(**kwargs)


def _turn_output(**overrides) -> TurnOutput:
    kwargs = dict(
        turn_key=mint_turn_key(),
        status=TurnStatus.COMPLETED,
        answer="here are your lists",
        command_outputs=[_command_output()],
    )
    kwargs.update(overrides)
    return TurnOutput(**kwargs)


def _turn_result(**overrides) -> TurnResult:
    kwargs = dict(
        turn_output=_turn_output(),
        channel_id="chan",
        conversation_id=1,
        user_message="list my todo lists",
        entry_workflow_name="todo_list_workflow",
        entry_context="TodoListManager",
    )
    kwargs.update(overrides)
    return TurnResult(**kwargs)


# ----------------------------------------------------------------------
# Field-set pins
# ----------------------------------------------------------------------


def test_command_response_field_set_is_pinned():
    assert tuple(CommandResponse.model_fields) == COMMAND_RESPONSE_FIELDS
    assert tuple(CommandResponse(response="x").model_dump()) == COMMAND_RESPONSE_FIELDS


def test_command_output_field_set_is_pinned():
    assert tuple(CommandOutput.model_fields) == COMMAND_OUTPUT_FIELDS
    assert tuple(_command_output().model_dump()) == COMMAND_OUTPUT_FIELDS


def test_turn_output_field_set_is_pinned():
    """The public projection old APIs return. §12.2: unchanged."""
    assert tuple(TurnOutput.model_fields) == TURN_OUTPUT_FIELDS
    assert tuple(_turn_output().model_dump()) == TURN_OUTPUT_DUMP_FIELDS


def test_turn_result_field_set_is_pinned_and_grows_only_by_appending():
    """The one shape §12.2 allows to grow, and only at the end.

    An insertion in the middle would still pass a set comparison while breaking
    the claim that everything before it is untouched, which is why this is
    ordered.
    """
    assert tuple(TurnResult.model_fields) == TURN_RESULT_FIELDS
    assert tuple(_turn_result().model_dump()) == TURN_RESULT_FIELDS
    assert TURN_RESULT_FIELDS[-2:] == ("execution_records", "routing_events")


def test_the_nested_structure_of_a_serialized_turn_is_pinned():
    """A top-level pin says nothing about what is inside `command_outputs`.

    The nesting is the part a reader actually walks — `observability_store`
    reaches `record["turn_output"]["command_outputs"][i]["command_response"]
    ["artifacts"]` — so each level is pinned rather than only the outermost.
    """
    dumped = _turn_result().model_dump()

    assert tuple(dumped["turn_output"]) == TURN_OUTPUT_DUMP_FIELDS
    nested_output = dumped["turn_output"]["command_outputs"][0]
    assert tuple(nested_output) == COMMAND_OUTPUT_FIELDS
    assert tuple(nested_output["command_response"]) == COMMAND_RESPONSE_FIELDS


def test_a_real_turn_produces_exactly_the_pinned_public_shape(
    initialized_fastworkflow, todo_workflow_path, tmp_path
):
    """The end-to-end version: a shape pinned only against hand-built objects
    can drift from what the runtime actually emits."""
    workflow = fastworkflow.Workflow.create(
        todo_workflow_path, workflow_id_str=f"parity-{uuid.uuid4().hex}"
    )
    ctx = WorkflowExecutionContext(run_as_agent=False)
    ctx.bind_app_workflow(workflow)
    workflow.root_command_context = TodoListManager(str(tmp_path / "todo_list.json"))
    try:
        turn_output = ctx.process_action_turn(
            fastworkflow.Action(command_name=LIST_COMMAND, command="list them")
        )
    finally:
        with suppress(Exception):
            ctx.close()

    assert isinstance(turn_output, TurnOutput)
    dumped = turn_output.model_dump()
    assert tuple(dumped) == TURN_OUTPUT_DUMP_FIELDS
    assert dumped["command_outputs"], "the real turn recorded no command output"
    assert tuple(dumped["command_outputs"][0]) == COMMAND_OUTPUT_FIELDS


# ----------------------------------------------------------------------
# Round trips
# ----------------------------------------------------------------------


def test_command_response_round_trips():
    original = CommandResponse(
        response="exported",
        success=False,
        artifacts={"client_key": {"nested": [1, "two", None]}},
        next_actions=[fastworkflow.Action(command_name=LIST_COMMAND)],
        recommendations=[fastworkflow.Recommendation(summary="try listing them")],
    )
    restored = CommandResponse.model_validate(original.model_dump(mode="json"))
    assert restored == original


def test_command_output_round_trips_including_the_appended_field():
    original = _command_output(command_call_id="0123456789abcdef")
    restored = CommandOutput.model_validate(original.model_dump(mode="json"))

    assert restored.command_call_id == "0123456789abcdef"
    assert restored.command_name == LIST_COMMAND
    assert restored.command_response.artifacts == {"client_key": "a,b\n1,2"}
    assert restored.success is True


def test_typed_parameters_round_trip_into_a_dict_and_back(real_params_class):
    """[A10] honesty, with the real workflow's real parameters model.

    In memory `command_parameters` is the typed instance; a record holds
    `model_dump()`. `Any` is what makes both legal — a `str` annotation would
    make the dumped form a lie and a typed annotation would reject the restored
    one, so the asymmetry is declared rather than hidden.
    """
    original = _command_output(
        command_name=CREATE_COMMAND,
        command_parameters=real_params_class(description="groceries"),
    )
    assert isinstance(original.command_parameters, real_params_class)

    dumped = original.model_dump(mode="json")
    assert dumped["command_parameters"] == {"description": "groceries"}

    restored = CommandOutput.model_validate(dumped)
    assert restored.command_parameters == {"description": "groceries"}
    # ...and the dict form survives a second pass, which is what a re-read of a
    # stored record does.
    assert (
        CommandOutput.model_validate(restored.model_dump(mode="json"))
        .command_parameters
        == {"description": "groceries"}
    )


def test_the_ask_user_role_inversion_round_trips():
    """[A7]: the question is in `command_parameters`, the answer is the response.

    The inversion has to survive serialization or a stored clarification reads
    backwards — as a command that was asked its own answer.
    """
    original = CommandOutput(
        command_name="ask_user",
        command_parameters="Which todo list did you mean?",
        command_response=CommandResponse(response="the urgent one", success=True),
    )
    restored = CommandOutput.model_validate(original.model_dump(mode="json"))

    assert restored.is_ask_user is True
    assert restored.question == "Which todo list did you mean?"
    assert restored.user_reply == "the urgent one"
    assert restored.success is True


def test_an_unanswered_ask_user_round_trips_as_unanswered():
    original = CommandOutput(
        command_name="ask_user",
        command_parameters="Which todo list did you mean?",
        command_response=CommandResponse(response="", success=False),
    )
    restored = CommandOutput.model_validate(original.model_dump(mode="json"))

    assert restored.question == "Which todo list did you mean?"
    assert restored.user_reply == ""
    assert restored.success is False


def test_turn_output_round_trips_with_its_computed_success():
    original = _turn_output(
        command_outputs=[
            _command_output(),
            _command_output(
                command_response=CommandResponse(response="nope", success=False)
            ),
        ]
    )
    assert original.success is False

    restored = TurnOutput.model_validate(original.model_dump(mode="json"))
    assert restored.turn_key == original.turn_key
    assert restored.status == TurnStatus.COMPLETED
    # Recomputed from the restored command outputs, not read from the dump: a
    # computed field that round-tripped as data could disagree with them.
    assert restored.success is False


def test_turn_result_round_trips_whole():
    original = _turn_result(
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
        suspended_ms=12,
        metadata={"anything": "the framework does not interpret"},
    )
    restored = TurnResult.model_validate(original.model_dump(mode="json"))

    assert restored.user_message == original.user_message
    assert restored.channel_id == "chan"
    assert restored.suspended_ms == 12
    assert restored.metadata == {"anything": "the framework does not interpret"}
    assert restored.turn_output.turn_key == original.turn_output.turn_key


# ----------------------------------------------------------------------
# Records written by other versions
# ----------------------------------------------------------------------


def test_a_record_written_before_command_call_id_still_validates():
    """`record_json` rows persisted by earlier releases have no key for it."""
    legacy = {
        "command_response": {"response": "ok", "success": True, "artifacts": {}},
        "workflow_name": "todo_list_workflow",
        "context": "TodoListManager",
        "command_name": LIST_COMMAND,
        "command_parameters": {"id": 1},
        "started_at": None,
        "duration_ms": 12,
    }
    restored = CommandOutput.model_validate(legacy)
    assert restored.command_call_id is None
    assert restored.command_name == LIST_COMMAND


def test_a_turn_result_written_before_the_additive_fields_still_validates():
    """The §12.2 compatibility claim, stated as the record it is about.

    Simulated by deleting the keys from a current dump rather than by writing a
    literal, so it stays a statement about *this* model as the rest of it
    changes.
    """
    aged = _turn_result().model_dump(mode="json")
    del aged["execution_records"]
    del aged["routing_events"]

    restored = TurnResult.model_validate(aged)
    assert restored.execution_records == ()
    assert restored.routing_events == ()
    assert restored.user_message == "list my todo lists"


def test_a_newer_record_validates_on_a_reader_that_does_not_know_its_keys():
    """The other direction, which is the one that strands deployed readers.

    `TurnResult` and `TurnOutput` ignore unknown keys, so a record written by a
    later version — one that added a field this code has never heard of — reads
    back as the subset this version understands instead of raising.
    """
    from_the_future = _turn_result().model_dump(mode="json")
    from_the_future["a_field_added_later"] = {"whatever": "it holds"}
    from_the_future["turn_output"]["another_one"] = [1, 2, 3]

    restored = TurnResult.model_validate(from_the_future)
    assert restored.user_message == "list my todo lists"
    assert restored.turn_output.answer == "here are your lists"
    assert not hasattr(restored, "a_field_added_later")


def test_the_pre_v3_command_responses_keyword_is_still_refused():
    """The compatibility guarantee is not "anything validates".

    A pre-v3.0 list keyword names a shape this model no longer has, and
    accepting it would silently drop every response but one.
    """
    with pytest.raises(ValueError, match="no longer accepts command_responses"):
        CommandOutput(command_responses=[CommandResponse(response="ok")])

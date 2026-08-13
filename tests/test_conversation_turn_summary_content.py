"""What the durable turn's "conversation summary" holds (bead fix-dzs.5).

Every durable turn is the 3-key shape ``{"conversation summary",
"conversation_traces", "feedback"}`` (see fastworkflow/conversation_history_io.py).
The deterministic and direct-action paths used to hardcode that field to the
constants ``"assistant_mode_command"`` and ``"process_action command"``, which
made it useless to the three consumers that read it and only it:
``ConversationStore.generate_topic_and_summary``,
``ConversationStore.get_conversation_summaries``, and
``WorkflowExecutionContext._refine_user_query``. ``POST /initialize`` routes
through exactly those two paths, so turn 0 of an initialize-only conversation
carried no information at all.

These tests lock in three things at once, because the field has to satisfy all
three or it is not fixed:

* it identifies what ran, on both paths;
* it stays small even when the command, its parameters, or its response are
  large -- the field lives in the in-memory history window and is fed verbatim
  to LLMs, so a request payload must never reach it;
* ``conversation_traces`` still carries the whole record, so the payload is
  still recoverable and a future change cannot quietly drop it.

The direct-action tests run the real CommandExecutor.perform_action against the
real todo_list_workflow. The deterministic tests replace
CommandExecutor.invoke_command with a plain classmethod, which is the seam the
rest of the suite uses for that path (tests/test_turn_result_capture.py,
tests/test_execution_context_concurrency.py): invoke_command runs the wildcard
NLU pipeline, and the test workflows ship no trained intent models.
"""

from __future__ import annotations

import json
import uuid
from contextlib import suppress
from pathlib import Path

import pytest

import fastworkflow
from fastworkflow.command_executor import CommandExecutor
from fastworkflow.workflow_execution_context import WorkflowExecutionContext

from tests.todo_list_workflow.application.todo_manager import TodoListManager

# The constants the two paths used to hardcode into the field.
OLD_ASSISTANT_CONSTANT = "assistant_mode_command"
OLD_ACTION_CONSTANT = "process_action command"

# Both call sites slice each half to 200 characters and join them with " -> ".
MAX_SUMMARY_CHARS = 404

# Big enough to prove the bound; the production payload that motivated the bead
# is a ~450 KB XML document, and the argument is the same at any size.
LARGE_VALUE = "x" * 40_000

# Direct actions address commands by their fully qualified name, the form the
# durable record already stores.
CREATE_LIST_COMMAND = "TodoListManager/create_todo_list"


@pytest.fixture
def todo_workflow_path() -> str:
    return str(Path(__file__).parent.joinpath("todo_list_workflow").resolve())


@pytest.fixture
def initialized_fastworkflow():
    fastworkflow.init({})
    fastworkflow.RoutingRegistry.clear_registry()
    yield
    fastworkflow.RoutingRegistry.clear_registry()


def _make_ctx(todo_workflow_path: str, label: str) -> WorkflowExecutionContext:
    """A deterministic (assistant-mode) context over the real todo workflow."""
    workflow = fastworkflow.Workflow.create(
        todo_workflow_path,
        workflow_id_str=f"turnsummary-{label}-{uuid.uuid4().hex}",
    )
    ctx = WorkflowExecutionContext(run_as_agent=False)
    ctx.bind_app_workflow(workflow)
    return ctx


@pytest.fixture
def action_ctx(initialized_fastworkflow, todo_workflow_path, tmp_path):
    """A context whose direct actions really execute.

    The manager is rooted at a tmp file rather than run through the startup
    command, whose ResponseGenerator hardcodes the workflow folder: these tests
    create lists with 40 KB descriptions and must not write them into the
    repository's todo_list.json fixture.
    """
    ctx = _make_ctx(todo_workflow_path, "action")
    ctx.app_workflow.root_command_context = TodoListManager(
        str(tmp_path / "todo_list.json")
    )
    yield ctx
    with suppress(Exception):
        ctx.close()


@pytest.fixture
def message_ctx(initialized_fastworkflow, todo_workflow_path):
    ctx = _make_ctx(todo_workflow_path, "message")
    yield ctx
    with suppress(Exception):
        ctx.close()


def _echoing_invoke(monkeypatch, response_text: str, parameters=None):
    """Stand in for the NLU pipeline, returning a real CommandOutput."""

    def fake_invoke(cls, session, command: str):
        return fastworkflow.CommandOutput(
            command_name=command.lstrip("/").split()[0] if command.strip() else "",
            command_parameters=parameters,
            command_response=fastworkflow.CommandResponse(response=response_text),
        )

    monkeypatch.setattr(CommandExecutor, "invoke_command", classmethod(fake_invoke))


def _last_turn(ctx: WorkflowExecutionContext) -> dict:
    messages = ctx.conversation_history.messages
    assert len(messages) == 1, "one turn should have been recorded"
    return messages[-1]


def _create_list_action(description: str) -> fastworkflow.Action:
    return fastworkflow.Action(
        command_name=CREATE_LIST_COMMAND,
        command="create a todo list",
        parameters={"description": description},
    )


# ---------------------------------------------------------------------------
# The field identifies what ran
# ---------------------------------------------------------------------------


def test_deterministic_turn_summary_names_the_command(message_ctx, monkeypatch):
    """A '/'-prefixed turn records the command, not "assistant_mode_command"."""
    _echoing_invoke(monkeypatch, "You have 2 todo lists.")

    message_ctx.process_turn("/list_todo_lists")

    summary = _last_turn(message_ctx)["conversation summary"]

    assert OLD_ASSISTANT_CONSTANT not in summary
    assert "list_todo_lists" in summary
    assert "You have 2 todo lists." in summary


def test_process_action_turn_summary_names_the_command(action_ctx):
    """A direct action records the command, not "process_action command"."""
    action_ctx.process_action(_create_list_action("groceries"))

    summary = _last_turn(action_ctx)["conversation summary"]

    assert OLD_ACTION_CONSTANT not in summary
    assert summary.startswith(f"{CREATE_LIST_COMMAND} -> ")
    # The response is the real command's, so the turn is recognizable from the
    # summary alone -- which is all generate_topic_and_summary gets.
    assert "groceries" in summary


def test_summary_is_a_single_line_even_when_the_response_is_not(action_ctx):
    """_refine_user_query renders one "key: value" line per field.

    create_todo_list's response is multi-line; left as-is its lines would show
    up in the refine prompt looking like fields of their own.
    """
    action_ctx.process_action(_create_list_action("trip preparation"))

    summary = _last_turn(action_ctx)["conversation summary"]

    assert "\n" not in summary
    assert "\r" not in summary


def test_refine_user_query_sees_the_command_instead_of_a_constant(action_ctx):
    """The field is not display-only: it is what refinement reads on the next turn."""
    action_ctx.process_action(_create_list_action("groceries"))
    summary = _last_turn(action_ctx)["conversation summary"]

    refined = action_ctx._refine_user_query(
        "what did that do?", action_ctx.conversation_history
    )

    assert refined.splitlines() == [
        f"conversation summary: {summary}",
        "feedback: None",
        "new_user_query: what did that do?",
    ]
    assert OLD_ACTION_CONSTANT not in refined
    assert "create_todo_list" in refined
    # Traces are excluded from the refine prompt, so the record never leaks in
    # through the back door.
    assert "conversation_traces" not in refined


# ---------------------------------------------------------------------------
# The field stays small
# ---------------------------------------------------------------------------


def test_action_summary_is_bounded_when_parameters_are_large(action_ctx):
    """The payload lives in the action's parameters; it must not reach the field.

    create_todo_list echoes its parameters into its response, so this bounds the
    response half too -- truncating only the parameters would not be enough.
    """
    action_ctx.process_action(_create_list_action(LARGE_VALUE))

    summary = _last_turn(action_ctx)["conversation summary"]

    assert len(summary) <= MAX_SUMMARY_CHARS
    assert LARGE_VALUE not in summary
    assert summary.startswith(f"{CREATE_LIST_COMMAND} -> ")


def test_deterministic_summary_is_bounded_when_the_command_is_large(
    message_ctx, monkeypatch
):
    """The deterministic path's message is caller-supplied and can be huge too."""
    _echoing_invoke(monkeypatch, f"stored {LARGE_VALUE}", parameters={"note": LARGE_VALUE})

    message_ctx.process_turn(f"/set_notes {LARGE_VALUE}")

    summary = _last_turn(message_ctx)["conversation summary"]

    assert len(summary) <= MAX_SUMMARY_CHARS
    assert LARGE_VALUE not in summary
    assert "set_notes" in summary


# ---------------------------------------------------------------------------
# conversation_traces still carries the whole record
# ---------------------------------------------------------------------------


def test_action_traces_still_carry_the_full_record(action_ctx):
    action_ctx.process_action(_create_list_action(LARGE_VALUE))

    turn = _last_turn(action_ctx)
    record = json.loads(turn["conversation_traces"])

    assert set(record) == {"command", "command_name", "parameters", "response"}
    assert record["command"] == "process_action"
    assert record["command_name"] == CREATE_LIST_COMMAND
    assert record["parameters"] == {"description": LARGE_VALUE}
    assert LARGE_VALUE in record["response"]
    assert turn["feedback"] is None


def test_deterministic_traces_still_carry_the_full_record(message_ctx, monkeypatch):
    _echoing_invoke(monkeypatch, f"stored {LARGE_VALUE}", parameters={"note": LARGE_VALUE})

    message_ctx.process_turn(f"/set_notes {LARGE_VALUE}")

    turn = _last_turn(message_ctx)
    record = json.loads(turn["conversation_traces"])

    assert set(record) == {"command", "command_name", "parameters", "response"}
    assert record["command"] == f"/set_notes {LARGE_VALUE}"
    assert record["command_name"] == "set_notes"
    assert record["parameters"] == {"note": LARGE_VALUE}
    assert LARGE_VALUE in record["response"]
    assert turn["feedback"] is None

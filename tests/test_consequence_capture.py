"""Context-before/after types on executed commands.

Architecture §12.0 deltas 2 and 4, §6.6.1, and FW-REQ-002's acceptance
criteria -- specifically the two that name this behavior directly:

* "An authorized navigation command records distinct context-before and
  context-after handles."
* "A non-navigation command records identical context-before and context-after
  handles."

The recorded value is the active context's TYPE name. fastWorkflow has no
framework-level identity for a context instance, so two equal values do not
prove the command stayed on the same object; the diagnosis reports that case as
unknown rather than unchanged.

"""

from __future__ import annotations

import uuid
from contextlib import suppress
from pathlib import Path

import pytest

import fastworkflow
from fastworkflow import tracing
from fastworkflow.command_executor import CommandExecutor
from fastworkflow.workflow_execution_context import WorkflowExecutionContext

from tests.todo_list_workflow.application.todo_manager import TodoListManager

# Moves the workflow's command context from TodoListManager down to the created
# TodoList (create_todo_list.py line 55).
NAVIGATING_COMMAND = "TodoListManager/create_todo_list"

# Reads and returns; never touches current_command_context.
NON_NAVIGATING_COMMAND = "TodoListManager/list_todo_lists"


@pytest.fixture
def todo_workflow_path() -> str:
    return str(Path(__file__).parent.joinpath("todo_list_workflow").resolve())


@pytest.fixture
def initialized_fastworkflow():
    fastworkflow.init({})
    from fastworkflow.command_routing import RoutingRegistry

    RoutingRegistry.clear_registry()
    yield
    RoutingRegistry.clear_registry()


class RecordingTraceSink:
    def __init__(self):
        self.spans: list[tracing.Span] = []

    def emit_span(self, span: tracing.Span) -> None:
        self.spans.append(span)

    def emit_turn_record(self, record) -> bool:
        return True

    def record_conversation_label(self, channel_id, conversation_id, topic, summary):
        pass

    def named(self, name: str) -> list[tracing.Span]:
        return [span for span in self.spans if span.name == name]


@pytest.fixture
def sink() -> RecordingTraceSink:
    return RecordingTraceSink()


@pytest.fixture
def ctx(initialized_fastworkflow, todo_workflow_path, tmp_path, sink):
    workflow = fastworkflow.Workflow.create(
        todo_workflow_path,
        workflow_id_str=f"consequence-{uuid.uuid4().hex}",
    )
    context = WorkflowExecutionContext(run_as_agent=False, trace_sink=sink)
    context.bind_app_workflow(workflow)
    workflow.root_command_context = TodoListManager(str(tmp_path / "todo_list.json"))
    yield context
    with suppress(Exception):
        context.close()


def _action(command_name: str, **parameters) -> fastworkflow.Action:
    return fastworkflow.Action(
        command_name=command_name, command="do it", parameters=parameters
    )


def _last_tool_call(sink: RecordingTraceSink) -> tracing.Span:
    return sink.named(tracing.SPAN_AGENT_TOOL_CALL)[-1]


# ----------------------------------------------------------------------
# FW-REQ-002 acceptance criteria
# ----------------------------------------------------------------------


def test_a_navigation_command_records_distinct_context_types(ctx, sink):
    """create_todo_list descends TodoListManager -> TodoList."""
    ctx.process_action_turn(_action(NAVIGATING_COMMAND, description="groceries"))

    span = _last_tool_call(sink)
    assert span.attributes[tracing.ATTR_CONTEXT_BEFORE] == "TodoListManager"
    assert span.attributes[tracing.ATTR_CONTEXT_AFTER] == "TodoList"


def test_a_non_navigation_command_records_identical_context_types(ctx, sink):
    """list_todo_lists reads and returns; the workflow does not move."""
    ctx.process_action_turn(_action(NON_NAVIGATING_COMMAND))

    span = _last_tool_call(sink)
    assert span.attributes[tracing.ATTR_CONTEXT_BEFORE] == "TodoListManager"
    assert span.attributes[tracing.ATTR_CONTEXT_AFTER] == "TodoListManager"


def test_the_prose_path_records_context_types_too(
    initialized_fastworkflow, todo_workflow_path, tmp_path, sink, monkeypatch
):
    """FW-REQ-002 clause 5: capture semantics are shared across paths.

    The CME hop is stood in for because the test workflow ships no trained intent
    models; the dispatch it stands in for is the real `perform_action`, so the
    context reads under test run for real.
    """
    workflow = fastworkflow.Workflow.create(
        todo_workflow_path, workflow_id_str=f"context-prose-{uuid.uuid4().hex}"
    )
    context = WorkflowExecutionContext(run_as_agent=False, trace_sink=sink)
    context.bind_app_workflow(workflow)
    workflow.root_command_context = TodoListManager(str(tmp_path / "todo_list.json"))

    real_perform_action = CommandExecutor.perform_action

    def cme_hop(cls, wf, action):
        command_output = real_perform_action(
            workflow, _action(NAVIGATING_COMMAND, description="groceries")
        )
        command_output.command_response.artifacts["command_handled"] = True
        command_output.command_name = NAVIGATING_COMMAND
        return command_output

    monkeypatch.setattr(CommandExecutor, "perform_action", classmethod(cme_hop))

    try:
        context.process_turn("make me a grocery list")

        execute = sink.named(tracing.SPAN_COMMAND_EXECUTE)[0]
        assert execute.attributes[tracing.ATTR_CONTEXT_BEFORE] == "TodoListManager"
        assert execute.attributes[tracing.ATTR_CONTEXT_AFTER] == "TodoList"
    finally:
        with suppress(Exception):
            context.close()

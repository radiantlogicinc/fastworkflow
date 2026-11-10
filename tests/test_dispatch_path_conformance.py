"""Dispatch-path conformance matrix for command_call_id + execution_records.

Architecture §12.1.1 requires the migrated paths to stamp the same correlation
skeleton. This matrix covers:

* prose — ``process_turn`` / ``invoke_command`` through the CME hop stand-in;
* direct action — ``process_action_turn``;
* agent tool — ``_execute_workflow_query`` on the trained hello_world workflow
  (no full agent training required).

Each row asserts ``command_call_id`` on the outcome, on the covering span(s),
and one ``ExecutionRecordRef`` row on ``TurnResult`` that joins them.
"""

from __future__ import annotations

import uuid
from contextlib import suppress
from pathlib import Path

import pytest
from dotenv import dotenv_values

import fastworkflow
from fastworkflow import tracing
from fastworkflow.command_executor import CommandExecutor
from fastworkflow.workflow_execution_context import WorkflowExecutionContext
from fastworkflow.workflow_agent import _execute_workflow_query

from tests.todo_list_workflow.application.todo_manager import TodoListManager

HELLO_WORLD = str(
    Path(__file__).parent.parent / "fastworkflow" / "examples" / "hello_world"
)
LIST_COMMAND = "TodoListManager/list_todo_lists"


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
        self.turn_records: list = []

    def emit_span(self, span: tracing.Span) -> None:
        self.spans.append(span)

    def emit_turn_record(self, record) -> bool:
        self.turn_records.append(record)
        return True

    def record_conversation_label(self, *args) -> None:
        pass

    def named(self, name: str) -> list[tracing.Span]:
        return [span for span in self.spans if span.name == name]


@pytest.fixture
def sink() -> RecordingTraceSink:
    return RecordingTraceSink()


def _make_todo_ctx(todo_workflow_path: str, tmp_path, sink):
    workflow = fastworkflow.Workflow.create(
        todo_workflow_path,
        workflow_id_str=f"conform-{uuid.uuid4().hex}",
    )
    ctx = WorkflowExecutionContext(run_as_agent=False, trace_sink=sink)
    ctx.bind_app_workflow(workflow)
    workflow.root = TodoListManager(str(tmp_path / "todo_list.json"))
    return ctx


def _action(command_name: str = LIST_COMMAND, **parameters) -> fastworkflow.Action:
    return fastworkflow.Action(
        command_name=command_name, command="do it", parameters=parameters
    )


def _nesting_cme_hop(monkeypatch, app_workflow, nested_command: str = LIST_COMMAND):
    real_perform_action = CommandExecutor.perform_action

    def cme_hop(cls, workflow, action):
        command_output = real_perform_action(
            app_workflow, _action(nested_command)
        )
        command_output.command_response.artifacts["command_handled"] = True
        command_output.command_name = nested_command
        return command_output

    monkeypatch.setattr(CommandExecutor, "perform_action", classmethod(cme_hop))


def _assert_path_conformance(
    *,
    sink: RecordingTraceSink,
    turn_outputs: list,
    turn_result,
    outer_span_name: str,
    inner_span_name: str | None = None,
):
    assert turn_result.execution_records, "execution_records stayed empty"
    assert turn_outputs[-1].command_call_id

    outer = sink.named(outer_span_name)[-1]
    call_id = outer.attributes[tracing.ATTR_COMMAND_CALL_ID]
    assert call_id == turn_outputs[-1].command_call_id

    primary_record = turn_result.execution_records[0]
    assert primary_record.command_call_id == call_id

    if inner_span_name is not None:
        inner = sink.named(inner_span_name)[-1]
        assert inner.attributes[tracing.ATTR_COMMAND_CALL_ID] == call_id
        assert primary_record.span_id == inner.span_id
    else:
        assert primary_record.span_id == outer.span_id


def test_prose_path_via_process_turn(
    initialized_fastworkflow, todo_workflow_path, tmp_path, sink, monkeypatch
):
    ctx = _make_todo_ctx(todo_workflow_path, tmp_path, sink)
    try:
        _nesting_cme_hop(monkeypatch, ctx.app_workflow)
        ctx.process_turn("list my todo lists")

        _assert_path_conformance(
            sink=sink,
            turn_outputs=ctx._turn_outputs,
            turn_result=sink.turn_records[-1],
            outer_span_name=tracing.SPAN_AGENT_TOOL_CALL,
            inner_span_name=tracing.SPAN_COMMAND_EXECUTE,
        )
    finally:
        with suppress(Exception):
            ctx.close()


def test_direct_action_path_via_process_action_turn(
    initialized_fastworkflow, todo_workflow_path, tmp_path, sink
):
    ctx = _make_todo_ctx(todo_workflow_path, tmp_path, sink)
    try:
        ctx.process_action_turn(_action())

        _assert_path_conformance(
            sink=sink,
            turn_outputs=ctx._turn_outputs,
            turn_result=sink.turn_records[-1],
            outer_span_name=tracing.SPAN_AGENT_TOOL_CALL,
            inner_span_name=None,
        )
    finally:
        with suppress(Exception):
            ctx.close()


@pytest.fixture(scope="module")
def hello_initialized():
    # `is_workflow_trained`, not "___command_info exists": an untrained
    # scaffold already has that directory (3.3 port adaptation).
    from fastworkflow.model_pipeline_training import is_workflow_trained

    if not is_workflow_trained(HELLO_WORLD)[0]:
        pytest.skip("hello_world is not trained on this machine")
    env = dotenv_values("fastworkflow/examples/fastworkflow.env")
    fastworkflow.init(dict(env))
    from fastworkflow.command_routing import RoutingRegistry

    RoutingRegistry.clear_registry()
    yield
    RoutingRegistry.clear_registry()


def test_agent_tool_path_via_execute_workflow_query(hello_initialized):
    sink = RecordingTraceSink()
    wf = fastworkflow.Workflow.create(
        HELLO_WORLD,
        workflow_id_str=f"conform-agent-{uuid.uuid4().hex}",
        workflow_context={"run_as_agent": True},
    )
    ctx = WorkflowExecutionContext(run_as_agent=True, trace_sink=sink)
    ctx.bind_app_workflow(wf)
    ctx._begin_turn("agent tool conformance")
    ctx.push_active_workflow(wf)

    _execute_workflow_query(
        "add_two_numbers <first_num>4</first_num><second_num>6</second_num>",
        ctx,
    )
    turn_result = ctx._build_turn_result(ctx._turn_outputs[-1])

    _assert_path_conformance(
        sink=sink,
        turn_outputs=ctx._turn_outputs,
        turn_result=turn_result,
        outer_span_name=tracing.SPAN_AGENT_TOOL_CALL,
        inner_span_name=tracing.SPAN_COMMAND_EXECUTE,
    )

    ctx.pop_active_workflow()


def test_each_path_produces_distinct_call_ids(
    initialized_fastworkflow, todo_workflow_path, tmp_path, sink, monkeypatch
):
    """Two turns on the same path are two calls, not one reused id."""
    ctx = _make_todo_ctx(todo_workflow_path, tmp_path, sink)
    try:
        ctx.process_action_turn(_action())
        ctx.process_action_turn(_action())

        ids = [r.command_call_id for r in sink.turn_records[-1].execution_records]
        prior_ids = [
            r.command_call_id for r in sink.turn_records[-2].execution_records
        ]
        assert ids
        assert prior_ids
        assert ids[0] != prior_ids[0]
    finally:
        with suppress(Exception):
            ctx.close()

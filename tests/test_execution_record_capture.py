"""TurnResult.execution_records join command_call_id to spans (arch §12.1–§12.2).

These tests measure the durable correlation skeleton on ``TurnResult``: each
``ExecutionRecordRef`` must name the ``command_call_id`` minted for one dispatch
and the ``span_id`` of the span that covers it, so a reader holding
``record_json`` can join outcomes to trace rows without guessing from ordering.

Fixtures follow ``tests/test_command_call_id.py`` and
``tests/test_turn_result_additive.py``: real ``todo_list_workflow``, real
``WorkflowExecutionContext``, real ``CommandExecutor``, and a real
``TraceSink`` implementation writing real SQLite in ``tmp_path``. The only
stand-in is the CME wildcard hop on the prose path, because the test workflow
ships no trained intent models — and it performs a real nested dispatch rather
than returning a canned result.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import suppress
from pathlib import Path

import pytest

import fastworkflow
from fastworkflow import TurnResult, tracing
from fastworkflow.observability import store as obs
from fastworkflow.command_executor import CommandExecutor
from fastworkflow.observability.execution_recorder import record_execution
from fastworkflow.workflow_execution_context import WorkflowExecutionContext

from tests.todo_list_workflow.application.todo_manager import TodoListManager

LIST_COMMAND = "TodoListManager/list_todo_lists"
CREATE_COMMAND = "TodoListManager/create_todo_list"


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
    """Real TraceSink implementation that records everything it receives."""

    def __init__(self):
        self.spans: list[tracing.Span] = []
        self.turn_records: list[TurnResult] = []

    def emit_span(self, span: tracing.Span) -> None:
        self.spans.append(span)

    def emit_turn_record(self, record) -> bool:
        self.turn_records.append(record)
        return True

    def record_conversation_label(self, channel_id, conversation_id, topic, summary):
        pass

    def named(self, name: str) -> list[tracing.Span]:
        return [span for span in self.spans if span.name == name]


def _make_ctx(todo_workflow_path: str, tmp_path, sink) -> WorkflowExecutionContext:
    workflow = fastworkflow.Workflow.create(
        todo_workflow_path,
        workflow_id_str=f"execrec-{uuid.uuid4().hex}",
    )
    ctx = WorkflowExecutionContext(run_as_agent=False, trace_sink=sink)
    ctx.bind_app_workflow(workflow)
    workflow.root_command_context = TodoListManager(str(tmp_path / "todo_list.json"))
    return ctx


@pytest.fixture
def sink() -> RecordingTraceSink:
    return RecordingTraceSink()


@pytest.fixture
def ctx(initialized_fastworkflow, todo_workflow_path, tmp_path, sink):
    context = _make_ctx(todo_workflow_path, tmp_path, sink)
    yield context
    with suppress(Exception):
        context.close()


def _action(command_name: str = LIST_COMMAND, **parameters) -> fastworkflow.Action:
    return fastworkflow.Action(
        command_name=command_name, command="do it", parameters=parameters
    )


def _nesting_cme_hop(monkeypatch, app_workflow, nested_command: str = LIST_COMMAND):
    """Stand in for the untrained CME wildcard hop, keeping the nesting real."""
    real_perform_action = CommandExecutor.perform_action

    def cme_hop(cls, workflow, action):
        command_output = real_perform_action(
            app_workflow, _action(nested_command)
        )
        command_output.command_response.artifacts["command_handled"] = True
        command_output.command_name = nested_command
        return command_output

    monkeypatch.setattr(CommandExecutor, "perform_action", classmethod(cme_hop))
    return real_perform_action


def _rows(path: str, sql: str) -> list[dict]:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in conn.execute(sql).fetchall()]
    finally:
        conn.close()


def test_direct_action_turn_populates_execution_records(ctx, sink):
    """One real dispatch yields one correlation ref joined to its span."""
    ctx.process_action_turn(_action())

    assert sink.turn_records, "the real turn emitted no turn record"
    turn_result = sink.turn_records[-1]
    assert turn_result.execution_records, "execution_records stayed empty"

    record = turn_result.execution_records[0]
    tool_call = sink.named(tracing.SPAN_AGENT_TOOL_CALL)[0]
    span_call_id = tool_call.attributes[tracing.ATTR_COMMAND_CALL_ID]

    assert record.command_call_id == span_call_id
    assert record.span_id == tool_call.span_id
    assert record.command_ordinal == 0
    assert record.parent_call_id is None
    assert ctx._turn_outputs[-1].command_call_id == span_call_id


def test_execution_records_survive_turn_record_serialization(ctx, sink):
    """The join must reach record_json, not only in-memory TurnResult."""
    ctx.process_action_turn(_action())

    turn_row, _artifacts = obs.serialize_turn_result(sink.turn_records[-1])
    persisted = json.loads(turn_row["record_json"])

    span_call_id = sink.named(tracing.SPAN_AGENT_TOOL_CALL)[0].attributes[
        tracing.ATTR_COMMAND_CALL_ID
    ]
    assert persisted["execution_records"] == [
        {
            "contract_version": persisted["execution_records"][0]["contract_version"],
            "command_call_id": span_call_id,
            "parent_call_id": None,
            "command_ordinal": 0,
            "span_id": sink.named(tracing.SPAN_AGENT_TOOL_CALL)[0].span_id,
        }
    ]


def test_execution_records_survive_sqlite_round_trip(
    initialized_fastworkflow, todo_workflow_path, tmp_path
):
    """Through real SQLite, because that is where a reader actually joins."""
    db_path = str(tmp_path / "observability.sqlite3")
    store_sink = obs.SQLiteTraceSink(db_path)
    context = _make_ctx(todo_workflow_path, tmp_path, store_sink)
    try:
        context.process_action_turn(_action())
        assert store_sink.flush()
    finally:
        with suppress(Exception):
            context.close()
        store_sink.close()

    stored = json.loads(_rows(db_path, "SELECT record_json FROM turns")[0]["record_json"])
    restored = TurnResult.model_validate(stored)

    assert restored.execution_records
    assert restored.execution_records[0].command_call_id
    assert restored.execution_records[0].span_id


def test_prose_path_records_nested_child_calls(
    initialized_fastworkflow, todo_workflow_path, tmp_path, sink, monkeypatch
):
    """Inner CME/core hops with no span of their own appear as child refs."""
    context = _make_ctx(todo_workflow_path, tmp_path, sink)
    try:
        _nesting_cme_hop(monkeypatch, context.app_workflow)
        context.process_turn("list my todo lists")

        turn_result = sink.turn_records[-1]
        assert len(turn_result.execution_records) >= 2

        parent = turn_result.execution_records[0]
        child = turn_result.execution_records[1]
        execute = sink.named(tracing.SPAN_COMMAND_EXECUTE)[0]

        assert parent.command_call_id == execute.attributes[tracing.ATTR_COMMAND_CALL_ID]
        assert parent.span_id == execute.span_id
        assert child.parent_call_id == parent.command_call_id
        assert child.command_call_id != parent.command_call_id
        assert child.span_id is None
    finally:
        with suppress(Exception):
            context.close()


def test_no_sink_means_no_execution_records(
    initialized_fastworkflow, todo_workflow_path, tmp_path
):
    """Capture projections are sink-gated; the id on CommandOutput is not.

    The no-op sink is passed explicitly: a context given no sink at all opens
    its workflow's own observability sink.
    """
    context = _make_ctx(todo_workflow_path, tmp_path, sink=tracing.NoOpTraceSink())
    try:
        context.process_action_turn(_action())
        assert context._turn_outputs[-1].command_call_id
        assert getattr(context, "_execution_recorder", None) is None
    finally:
        with suppress(Exception):
            context.close()


def test_record_execution_with_no_recorder_is_cheap():
    """FW-NFR-005 sanity: record_execution(recorder=None) is a no-op, not work.

    No strict timing budget — only that 100k calls finish in well under a second,
    matching the spirit of test_span_contract_versioning.py's stamping-cost test.
    """
    start = time.perf_counter()
    for _ in range(100_000):
        record_execution(
            None,
            command_call_id="deadbeef",
            parent_call_id=None,
            span_id=None,
            child_calls=[{"call_id": "c1", "parent_call_id": "deadbeef"}],
        )
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0, f"record_execution(None) took {elapsed:.2f}s for 100k calls"


# ----------------------------------------------------------------------
# The recorder must exist on every construction path (fix-ajv.20)
# ----------------------------------------------------------------------
#
# _build_turn_result reads `self._execution_recorder.records() if
# self._execution_recorder is not None else ()`. That guard is only a guard if
# the attribute exists: _begin_turn was the sole binder, and resume deliberately
# skips _begin_turn (it continues the same logical turn), so a cross-process
# resume reached the read on a context that had never bound it — and the null
# check raised AttributeError instead of returning (). A missing binding wearing
# the costume of a null check.


def test_a_fresh_context_has_the_attribute_before_any_turn_begins(
    initialized_fastworkflow, todo_workflow_path, tmp_path, sink
):
    """Construction, not _begin_turn, is what guarantees the attribute."""
    context = _make_ctx(todo_workflow_path, tmp_path, sink)
    try:
        # Not getattr(..., default): the point is that the attribute is bound.
        assert context.__dict__["_execution_recorder"] is None
    finally:
        with suppress(Exception):
            context.close()


def test_a_resumed_turn_finalizes_instead_of_raising(
    initialized_fastworkflow, todo_workflow_path, tmp_path, sink
):
    """The production shape of the bug: resume, then finalize, in a new process.

    `apply_serialized_state` rebuilds the turn accumulator without _begin_turn,
    which is exactly the path that used to reach the read unbound.
    """
    origin = _make_ctx(todo_workflow_path, tmp_path, sink)
    origin._begin_turn("original message")
    state = origin.serialize_state(channel_id="chan")
    with suppress(Exception):
        origin.close()

    resumed = _make_ctx(todo_workflow_path, tmp_path, sink)
    try:
        resumed.apply_serialized_state(state)
        turn_result = resumed._build_turn_result(
            fastworkflow.CommandOutput(
                command_name="x",
                command_response=fastworkflow.CommandResponse(response="ok"),
            )
        )
        assert turn_result is not None
    finally:
        with suppress(Exception):
            resumed.close()


def test_a_resumed_turn_can_still_record_its_post_resume_commands(
    initialized_fastworkflow, todo_workflow_path, tmp_path, sink
):
    """Not merely 'does not crash': the resumed half of the turn stays joinable.

    The pre-suspension process kept its own ledger and those records went durable
    with its spans; what the restore rebuilds is the accumulator for the commands
    the resumed turn is about to run. Without it every post-resume outcome would
    carry a command_call_id with no ExecutionRecordRef to join to — the same gap
    fix-ajv.16 closed for failures, reopened for resumptions.
    """
    origin = _make_ctx(todo_workflow_path, tmp_path, sink)
    origin._begin_turn("original message")
    state = origin.serialize_state(channel_id="chan")
    with suppress(Exception):
        origin.close()

    resumed = _make_ctx(todo_workflow_path, tmp_path, sink)
    try:
        resumed.apply_serialized_state(state)
        assert resumed._execution_recorder is not None, "sink installed → recorder"

        resumed.process_action_turn(_action())
        assert resumed._turn_outputs[-1].command_call_id
    finally:
        with suppress(Exception):
            resumed.close()

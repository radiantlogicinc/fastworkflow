"""An execution context given no sink records into its workflow's own DB.

Recording is always on, so a program that embeds the library and builds a
``WorkflowExecutionContext`` itself -- no fastWorkflow entry point involved --
gets the same owner-only, pruned observability record: binding the app
workflow opens that workflow's sink. A sink the caller chose, including an
explicit ``NoOpTraceSink``, is never replaced; an automatic one follows a
rebind to another workflow.

Every test runs under the per-test temporary state root from
``tests/conftest.py``. Deterministic turns only: no model, no network.
"""
from __future__ import annotations

import os
import sqlite3
import stat
import uuid
from contextlib import suppress
from pathlib import Path

import pytest

import fastworkflow
from fastworkflow import state_paths, tracing
from fastworkflow.observability import store as obs
from fastworkflow.workflow_execution_context import WorkflowExecutionContext

from tests.todo_list_workflow.application.todo_manager import TodoListManager

LIST_COMMAND = "TodoListManager/list_todo_lists"
TESTS = Path(__file__).parent


@pytest.fixture
def todo_workflow_path() -> str:
    return str(TESTS.joinpath("todo_list_workflow").resolve())


@pytest.fixture
def example_workflow_path() -> str:
    return str(TESTS.joinpath("example_workflow").resolve())


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
        self.turn_records: list = []

    def emit_span(self, span: tracing.Span) -> None:
        self.spans.append(span)

    def emit_turn_record(self, record) -> bool:
        self.turn_records.append(record)
        return True

    def record_conversation_label(self, channel_id, conversation_id, topic, summary):
        pass


def _workflow(path: str) -> fastworkflow.Workflow:
    return fastworkflow.Workflow.create(path, workflow_id_str=f"auto-{uuid.uuid4().hex}")


def _action() -> fastworkflow.Action:
    return fastworkflow.Action(command_name=LIST_COMMAND, command="do it", parameters={})


def _turn_rows(db_path: str) -> list[tuple[str, str]]:
    if not os.path.exists(db_path):
        return []
    with sqlite3.connect(db_path) as conn:
        return conn.execute("SELECT turn_key, channel_id FROM turns").fetchall()


def test_a_context_given_no_sink_records_into_its_workflows_db(
    initialized_fastworkflow, todo_workflow_path, tmp_path
):
    workflow = _workflow(todo_workflow_path)
    context = WorkflowExecutionContext(run_as_agent=False)
    try:
        context.bind_observability_identity(channel_id="embedded-channel")
        context.bind_app_workflow(workflow)
        workflow.root_command_context = TodoListManager(str(tmp_path / "todo_list.json"))

        db_path = state_paths.observability_db(todo_workflow_path)
        sink = context.trace_sink
        assert isinstance(sink, obs.SQLiteTraceSink)
        # The process-wide factory's sink: opened once, pruned on construction.
        assert obs.existing_observability_sink(todo_workflow_path) is sink
        assert sink.store.db_path == db_path

        turn = context.process_action_turn(_action())
        assert turn.success
        assert sink.flush()
        assert (turn.turn_key, "embedded-channel") in _turn_rows(db_path)
        assert stat.S_IMODE(os.stat(db_path).st_mode) == 0o600
        assert stat.S_IMODE(os.stat(os.path.dirname(db_path)).st_mode) == 0o700
    finally:
        with suppress(Exception):
            context.close()


def test_a_caller_supplied_sink_is_never_replaced(
    initialized_fastworkflow, todo_workflow_path, example_workflow_path, tmp_path
):
    recording = RecordingTraceSink()
    via_constructor = WorkflowExecutionContext(run_as_agent=False, trace_sink=recording)
    via_setter = WorkflowExecutionContext(run_as_agent=False)
    try:
        via_constructor.bind_app_workflow(_workflow(todo_workflow_path))
        assert via_constructor.trace_sink is recording
        via_constructor.bind_app_workflow(_workflow(example_workflow_path))
        assert via_constructor.trace_sink is recording

        via_setter.set_trace_sink(recording)
        via_setter.bind_app_workflow(_workflow(todo_workflow_path))
        assert via_setter.trace_sink is recording
        # Neither context opened a sink of its own for either workflow.
        assert obs.existing_observability_sink(todo_workflow_path) is None
        assert obs.existing_observability_sink(example_workflow_path) is None
    finally:
        for context in (via_constructor, via_setter):
            with suppress(Exception):
                context.close()


def test_an_explicit_no_op_sink_stays_no_op_and_records_nothing(
    initialized_fastworkflow, todo_workflow_path, tmp_path
):
    workflow = _workflow(todo_workflow_path)
    context = WorkflowExecutionContext(
        run_as_agent=False, trace_sink=tracing.NoOpTraceSink()
    )
    try:
        context.bind_observability_identity(channel_id="silent-channel")
        context.bind_app_workflow(workflow)
        workflow.root_command_context = TodoListManager(str(tmp_path / "todo_list.json"))

        assert isinstance(context.trace_sink, tracing.NoOpTraceSink)
        turn = context.process_action_turn(_action())
        assert turn.success
        assert isinstance(context.trace_sink, tracing.NoOpTraceSink)
        assert obs.existing_observability_sink(todo_workflow_path) is None
        assert _turn_rows(state_paths.observability_db(todo_workflow_path)) == []
    finally:
        with suppress(Exception):
            context.close()


def test_a_rebind_moves_an_automatic_sink_to_the_new_workflows_db(
    initialized_fastworkflow, todo_workflow_path, example_workflow_path
):
    context = WorkflowExecutionContext(run_as_agent=False)
    try:
        context.bind_app_workflow(_workflow(todo_workflow_path))
        first = context.trace_sink
        assert first.store.db_path == state_paths.observability_db(todo_workflow_path)

        context.bind_app_workflow(_workflow(example_workflow_path))
        second = context.trace_sink
        assert isinstance(second, obs.SQLiteTraceSink)
        assert second is not first
        assert second.store.db_path == state_paths.observability_db(example_workflow_path)

        # Rebinding to the same workflow keeps the sink it already has.
        context.bind_app_workflow(_workflow(example_workflow_path))
        assert context.trace_sink is second
    finally:
        with suppress(Exception):
            context.close()


def test_setting_no_sink_hands_the_choice_back_to_the_context(
    initialized_fastworkflow, todo_workflow_path
):
    """``set_trace_sink(None)`` returns to the automatic sink, not to silence."""
    context = WorkflowExecutionContext(run_as_agent=False, trace_sink=RecordingTraceSink())
    try:
        context.bind_app_workflow(_workflow(todo_workflow_path))
        context.set_trace_sink(None)
        assert context.trace_sink is obs.existing_observability_sink(todo_workflow_path)
        assert isinstance(context.trace_sink, obs.SQLiteTraceSink)
    finally:
        with suppress(Exception):
            context.close()


def test_the_internal_command_metadata_workflow_is_never_a_recording_target(
    initialized_fastworkflow,
):
    """The context's own CME workflow lives in the package; nothing records there."""
    internal = fastworkflow.get_internal_workflow_path("command_metadata_extraction")
    context = WorkflowExecutionContext(run_as_agent=False)
    try:
        context.bind_app_workflow(_workflow(internal))
        assert isinstance(context.trace_sink, tracing.NoOpTraceSink)
        assert obs.existing_observability_sink(internal) is None
    finally:
        with suppress(Exception):
            context.close()

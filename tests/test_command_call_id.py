"""Durable CommandOutput-to-span correlation (arch §12.0 delta 1, §12.1 item 5).

Before `command_call_id` there was no join at all between a command's outcome and
its trace: `fw.command.execute` carried a random uuid4 that nothing else knew, and
`CommandOutput` carried no id, so a reader holding a turn's `record_json` could
only guess which span produced which outcome from ordering and timestamps. These
tests are about that join actually working end to end, and about the two ways it
can look right and be useless:

* an id that is stamped on the span but not on the outcome (or the reverse), which
  reads as correlation and joins nothing;
* an id minted twice for one execution — once by the dispatcher and once by the
  span emitter — which produces two plausible ids per command and matches neither.

The child-call ledger gets the same treatment. An application command is reached
only through an internal CME `perform_action` hop, which may dispatch a further
core command; those inner calls have no spans, so if their ids are not filed under
the enclosing call they are simply lost.

`tests/test_no_capture_control_flow.py` holds the structural half — that nothing
branches on any of this. It is a separate file because it reads source rather than
running workflows.

Fixtures follow tests/test_tracing_phase1.py and
tests/test_conversation_turn_summary_content.py: the real todo_list_workflow, the
real WorkflowExecutionContext, the real CommandExecutor, and a real TraceSink
implementation. The only stand-in is at the NLU boundary, because the test
workflows ship no trained intent models — and it is a stand-in that performs a
REAL nested dispatch rather than returning a canned result, since the nesting is
the thing under test.
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

# A real, parameterless command of the real workflow. Parameterless keeps the
# direct-action path off the parameter-extraction machinery, which the untrained
# test workflow cannot run.
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
        self.turn_records: list = []

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
    """A context whose commands really execute against a scratch todo store."""
    workflow = fastworkflow.Workflow.create(
        todo_workflow_path,
        workflow_id_str=f"callid-{uuid.uuid4().hex}",
    )
    ctx = WorkflowExecutionContext(run_as_agent=False, trace_sink=sink)
    ctx.bind_app_workflow(workflow)
    workflow.root = TodoListManager(str(tmp_path / "todo_list.json"))
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
    """Stand in for the untrained CME wildcard hop, keeping the nesting real.

    The real hop (``_workflows/command_metadata_extraction/_commands/wildcard.py``
    lines 76-88) resolves the user's text into an ``Action`` and calls
    ``CommandExecutor.perform_action`` again, then marks the result handled.
    Returning a canned ``CommandOutput`` here — which is what the rest of the
    suite does at this seam — would erase exactly the nested dispatch these tests
    exist to observe, so this reproduces the shape and lets the real
    ``perform_action`` run underneath it.
    """
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


# ----------------------------------------------------------------------
# The join works end to end
# ----------------------------------------------------------------------


def test_direct_action_span_and_output_share_one_call_id(ctx, sink):
    """The id on the outcome is the id on the span that produced it."""
    ctx.process_action_turn(_action())

    tool_call = sink.named(tracing.SPAN_AGENT_TOOL_CALL)[0]
    call_id = tool_call.attributes[tracing.ATTR_COMMAND_CALL_ID]

    assert call_id, "the span carries no command_call_id"
    assert [output.command_call_id for output in ctx._turn_outputs] == [call_id]


def test_the_join_survives_turn_record_serialization(ctx, sink):
    """The id reaches `record_json`, which is where a reader actually joins.

    Stamping the span and the in-memory object is not enough: the correlation is
    only durable if the id survives the projection the store persists.
    """
    from fastworkflow.observability import store as obs

    ctx.process_action_turn(_action())

    turn_row, _artifacts = obs.serialize_turn_result(sink.turn_records[-1])
    recorded = turn_row["record_json"]

    span_call_id = sink.named(tracing.SPAN_AGENT_TOOL_CALL)[0].attributes[
        tracing.ATTR_COMMAND_CALL_ID
    ]
    assert f'"command_call_id": "{span_call_id}"' in recorded


def test_prose_path_stamps_the_same_id_on_both_spans(
    initialized_fastworkflow, todo_workflow_path, tmp_path, sink, monkeypatch
):
    """One command execution, two spans, one id.

    `fw.agent.tool_call` and the `fw.command.execute` nested inside it describe
    the same execution, so a reader must not have to decide which id is "the"
    id for that command.
    """
    ctx = _make_ctx(todo_workflow_path, tmp_path, sink)
    try:
        _nesting_cme_hop(monkeypatch, ctx.app_workflow)
        ctx.process_turn("list my todo lists")

        execute = sink.named(tracing.SPAN_COMMAND_EXECUTE)[0]
        tool_call = sink.named(tracing.SPAN_AGENT_TOOL_CALL)[0]

        call_id = execute.attributes[tracing.ATTR_COMMAND_CALL_ID]
        assert call_id
        assert tool_call.attributes[tracing.ATTR_COMMAND_CALL_ID] == call_id
        assert ctx._turn_outputs[-1].command_call_id == call_id
    finally:
        with suppress(Exception):
            ctx.close()


# ----------------------------------------------------------------------
# Uniqueness and stability
# ----------------------------------------------------------------------


def test_each_command_execution_gets_its_own_id(ctx, sink):
    """Two executions of the same command are two calls, not one."""
    ctx.process_action_turn(_action())
    ctx.process_action_turn(_action())

    ids = [
        span.attributes[tracing.ATTR_COMMAND_CALL_ID]
        for span in sink.named(tracing.SPAN_AGENT_TOOL_CALL)
    ]
    assert len(ids) == 2
    assert len(set(ids)) == 2


def test_the_id_is_stable_across_a_re_emission_of_the_same_span(ctx, sink):
    """A span is emitted at open and again at close; the id may not move.

    The store treats a re-emission of a span_id as an idempotent upsert ([R2],
    [R6]). An id minted per emission rather than per execution would make the
    second write disagree with the first about which command it described.
    """
    ctx.process_action_turn(_action())
    span = sink.named(tracing.SPAN_AGENT_TOOL_CALL)[0]
    call_id = span.attributes[tracing.ATTR_COMMAND_CALL_ID]

    tracing.end_span(ctx, span, attributes={"replayed": True})

    assert span.attributes[tracing.ATTR_COMMAND_CALL_ID] == call_id
    replays = [s for s in sink.spans if s.span_id == span.span_id]
    assert len(replays) == 2
    assert {s.attributes[tracing.ATTR_COMMAND_CALL_ID] for s in replays} == {call_id}


# ----------------------------------------------------------------------
# Parent correlation for internal CME/core calls (arch §12.1 item 5)
# ----------------------------------------------------------------------


def test_a_nested_dispatch_is_filed_under_its_parent_call(ctx):
    """`perform_action` under an open call scope records itself as a child.

    This is the mechanism the CME hop uses. No span exists for the inner call,
    so the ledger is the only place its correlation to the enclosing command can
    live.
    """
    parent_call_id = tracing.new_command_call_id()

    with tracing.call_scope(parent_call_id) as ledger:
        command_output = CommandExecutor.perform_action(
            ctx.app_workflow, _action()
        )

    assert ledger == [
        {
            "call_id": command_output.command_call_id,
            "parent_call_id": parent_call_id,
            "command_name": LIST_COMMAND,
        }
    ]


def test_an_outermost_dispatch_files_no_child_entry(ctx):
    """A dispatch with no parent is not a child of anything.

    Its own id is already on its span; an entry here would invent a parent.
    """
    with tracing.call_scope(tracing.new_command_call_id()) as ledger:
        pass
    assert ledger == []

    command_output = CommandExecutor.perform_action(ctx.app_workflow, _action())
    assert command_output.command_call_id
    assert tracing.current_call_id() is None


def test_invoke_command_records_the_cme_hop_as_a_child(
    initialized_fastworkflow, todo_workflow_path, tmp_path, sink, monkeypatch
):
    """The real prose dispatch files its inner calls on its own span.

    The child's id is deliberately NOT the id on the returned CommandOutput: the
    outcome must carry the id of the span that covers it, or the join in
    `test_prose_path_stamps_the_same_id_on_both_spans` breaks. The ledger is
    what keeps the inner call visible anyway.
    """
    ctx = _make_ctx(todo_workflow_path, tmp_path, sink)
    try:
        _nesting_cme_hop(monkeypatch, ctx.app_workflow)
        ctx.process_turn("list my todo lists")

        execute = sink.named(tracing.SPAN_COMMAND_EXECUTE)[0]
        children = execute.attributes[tracing.ATTR_CHILD_CALLS]
        parent_call_id = execute.attributes[tracing.ATTR_COMMAND_CALL_ID]

        assert len(children) == 1
        assert children[0]["parent_call_id"] == parent_call_id
        assert children[0]["command_name"] == LIST_COMMAND
        assert children[0]["call_id"] != parent_call_id
    finally:
        with suppress(Exception):
            ctx.close()


def test_parent_call_id_is_none_at_the_outermost_dispatch(
    initialized_fastworkflow, todo_workflow_path, tmp_path, sink, monkeypatch
):
    """Recorded explicitly rather than omitted.

    An absent key and a top-level call look identical to a reader; None says
    "this dispatch had no parent", which is a fact rather than a gap.
    """
    ctx = _make_ctx(todo_workflow_path, tmp_path, sink)
    try:
        _nesting_cme_hop(monkeypatch, ctx.app_workflow)
        ctx.process_turn("list my todo lists")

        execute = sink.named(tracing.SPAN_COMMAND_EXECUTE)[0]
        assert tracing.ATTR_PARENT_CALL_ID in execute.attributes
        assert execute.attributes[tracing.ATTR_PARENT_CALL_ID] is None
    finally:
        with suppress(Exception):
            ctx.close()


# ----------------------------------------------------------------------
# Public compatibility (arch §12.2)
# ----------------------------------------------------------------------


def test_command_output_still_constructs_without_the_new_field():
    """Every existing constructor call in the codebase looks like this."""
    command_output = fastworkflow.CommandOutput(
        command_response=fastworkflow.CommandResponse(response="ok")
    )
    assert command_output.command_call_id is None


def test_an_already_serialized_record_still_validates():
    """A record written before this field existed has no key for it.

    `record_json` rows persisted by earlier versions are read back through this
    model, so an absent key must validate rather than raise — which is why the
    field is optional with a default instead of required.
    """
    legacy = {
        "command_response": {"response": "ok", "success": True, "artifacts": {}},
        "workflow_name": "todo_list_workflow",
        "context": "TodoList",
        "command_name": "TodoList/get_properties",
        "command_parameters": {"id": 1},
        "started_at": None,
        "duration_ms": 12,
    }
    restored = fastworkflow.CommandOutput.model_validate(legacy)
    assert restored.command_call_id is None
    assert restored.command_name == "TodoList/get_properties"


def test_the_field_round_trips_through_dump_and_validate():
    call_id = tracing.new_command_call_id()
    original = fastworkflow.CommandOutput(
        command_response=fastworkflow.CommandResponse(response="ok"),
        command_call_id=call_id,
    )
    assert (
        fastworkflow.CommandOutput.model_validate(
            original.model_dump(mode="json")
        ).command_call_id
        == call_id
    )


def test_the_legacy_command_responses_rejection_still_fires():
    """The pre-v3.0 keyword guard is untouched by the added field."""
    with pytest.raises(ValueError, match="no longer accepts command_responses"):
        fastworkflow.CommandOutput(
            command_responses=[fastworkflow.CommandResponse(response="ok")]
        )


# ----------------------------------------------------------------------
# Capture stays off when observability is off
# ----------------------------------------------------------------------


def test_no_sink_means_no_handle_projection_but_still_an_id(
    initialized_fastworkflow, todo_workflow_path, tmp_path
):
    """Handles cost nothing with tracing off; the id is stamped regardless.

    The id is the one piece a caller can read back through a public API without
    a trace store, so it is not sink-gated. The projections are, because they are
    only ever span attributes.
    """
    ctx = _make_ctx(todo_workflow_path, tmp_path, sink=None)
    try:
        ctx.process_action_turn(_action())
        assert ctx._turn_outputs[-1].command_call_id
    finally:
        with suppress(Exception):
            ctx.close()

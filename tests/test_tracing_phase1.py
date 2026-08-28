"""Phase 1 observability: TraceSink protocol + v1 boundary span emission
(bead fix-kw7.2, docs/fastworkflow_observability_studio_design.md §3.1).

Mirrors the fixtures/patterns of tests/test_turn_result_capture.py (real
todo_list_workflow; fakes only at the agent/LLM/NLU boundaries). The
RecordingTraceSink below is a real implementation of the TraceSink protocol
(the pluggable seam the design defines), not a mock of a fastworkflow
component.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import fastworkflow
from fastworkflow import TurnStatus, metrics, tracing, workflow_agent
from fastworkflow.command_executor import CommandExecutor
from fastworkflow.utils.react import AskUserSuspend, fastWorkflowReAct
from fastworkflow.workflow_execution_context import WorkflowExecutionContext


# ----------------------------------------------------------------------
# Fixtures and helpers
# ----------------------------------------------------------------------


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
        self.labels: list[tuple] = []

    def emit_span(self, span: tracing.Span) -> None:
        self.spans.append(span)

    def emit_turn_record(self, record) -> None:
        self.turn_records.append(record)

    def record_conversation_label(
        self, channel_id, conversation_id, topic, summary
    ) -> None:
        self.labels.append((channel_id, conversation_id, topic, summary))

    def named(self, name: str) -> list[tracing.Span]:
        return [s for s in self.spans if s.name == name]


class RaisingTraceSink:
    """A sink whose every method raises — must never fail a turn."""

    def emit_span(self, span) -> None:
        raise RuntimeError("sink is broken")

    def emit_turn_record(self, record) -> None:
        raise RuntimeError("sink is broken")

    def record_conversation_label(self, *args) -> None:
        raise RuntimeError("sink is broken")


class RecordingMetricsSink:
    def __init__(self):
        self.counters: list[tuple] = []
        self.observations: list[tuple] = []

    def increment(self, name, value=1, **labels):
        self.counters.append((name, value, labels))

    def observe(self, name, value, **labels):
        self.observations.append((name, value, labels))


def _make_assistant_ctx(todo_workflow_path, monkeypatch, sink=None):
    """Deterministic-mode context; CommandExecutor.invoke_command is faked at
    the NLU boundary (the test workflow ships no trained intent models)."""
    wf = fastworkflow.Workflow.create(
        todo_workflow_path,
        workflow_id_str=f"tracing-assist-{uuid.uuid4().hex}",
    )
    ctx = WorkflowExecutionContext(run_as_agent=False, trace_sink=sink)
    ctx.bind_app_workflow(wf)

    def fake_invoke(cls, session, command: str):
        return fastworkflow.CommandOutput(
            command_name=command.split()[0] if command else "",
            command_response=fastworkflow.CommandResponse(response=f"ok:{command}"),
        )

    monkeypatch.setattr(CommandExecutor, "invoke_command", classmethod(fake_invoke))
    return ctx, wf


def _make_agent_ctx(todo_workflow_path, monkeypatch, sink=None):
    """Agent-mode context; only the LLM/agent boundary is faked (same seams
    as tests/test_execution_context_agent.py)."""
    ctx = WorkflowExecutionContext(run_as_agent=True, trace_sink=sink)
    wf = fastworkflow.Workflow.create(
        todo_workflow_path,
        workflow_id_str=f"tracing-agent-{uuid.uuid4().hex}",
    )
    ctx.bind_app_workflow(wf)

    monkeypatch.setattr(
        "fastworkflow.workflow_agent.build_query_with_next_steps",
        lambda user_query, session, **kwargs: user_query,
    )
    monkeypatch.setattr(
        "fastworkflow.workflow_agent._what_can_i_do", lambda session: "commands"
    )
    monkeypatch.setattr(ctx, "_ensure_agent_initialized", lambda: None)
    monkeypatch.setattr(
        ctx,
        "_extract_conversation_summary",
        lambda user_query, actions, final: ("summary", "{}"),
    )
    return ctx, wf


def _set_agents(ctx, agent):
    ctx._workflow_tool_agent = agent
    ctx._intent_clarification_agent = MagicMock()


# ----------------------------------------------------------------------
# Taxonomy, protocol, and helper basics
# ----------------------------------------------------------------------


class TestTaxonomyAndProtocol:
    def test_v1_span_names(self):
        assert tracing.V1_SPAN_NAMES == {
            "fw.turn",
            "fw.planner.plan",
            "fw.planner.replan",
            "fw.agent.tool_call",
            "fw.command.execute",
            "fw.ask_user",
        }

    def test_agent_loop_span_names(self):
        assert tracing.AGENT_LOOP_SPAN_NAMES == {"fw.agent.execute", "fw.agent.step"}
        # fw.agent.execute is the whole loop; fw.command.execute is one command
        # inside one tool call. Two levels, two names — collapsing them would
        # put the executor and a single command at the same depth.
        assert tracing.SPAN_AGENT_EXECUTE != tracing.SPAN_COMMAND_EXECUTE
        assert not (tracing.AGENT_LOOP_SPAN_NAMES & tracing.V1_SPAN_NAMES)
        assert not (tracing.AGENT_LOOP_SPAN_NAMES & tracing.RESERVED_V2_SPAN_NAMES)

    def test_reserved_v2_names_modeled_not_emitted(self):
        assert "fw.nlu.intent" in tracing.RESERVED_V2_SPAN_NAMES
        assert "fw.nlu.param_extraction" in tracing.RESERVED_V2_SPAN_NAMES
        assert "fw.llm.call" in tracing.RESERVED_V2_SPAN_NAMES
        assert not (tracing.V1_SPAN_NAMES & tracing.RESERVED_V2_SPAN_NAMES)

    def test_sinks_satisfy_protocols(self):
        assert isinstance(tracing.NoOpTraceSink(), tracing.TraceSink)
        assert isinstance(RecordingTraceSink(), tracing.TraceSink)
        assert isinstance(metrics.NoOpMetricsSink(), metrics.MetricsSink)
        assert isinstance(metrics.LoggingMetricsSink(), metrics.MetricsSink)
        assert isinstance(RecordingMetricsSink(), metrics.MetricsSink)

    def test_deterministic_span_ids(self):
        a0 = tracing.deterministic_span_id("K", tracing.SPAN_ASK_USER, 0)
        a1 = tracing.deterministic_span_id("K", tracing.SPAN_ASK_USER, 1)
        assert a0 == tracing.deterministic_span_id("K", tracing.SPAN_ASK_USER, 0)
        assert a0 != a1
        assert tracing.root_span_id("K") == tracing.deterministic_span_id(
            "K", tracing.SPAN_TURN, 0
        )
        assert tracing.root_span_id("K") != tracing.root_span_id("K2")

    def test_attr_cap_is_lossy_and_counted(self):
        big = "x" * 20_000
        capped = tracing.cap_attr_value(big)
        assert capped["truncated"] is True
        assert capped["original_length"] == 20_000
        assert len(capped["sha256"]) == 64
        assert tracing.cap_attr_value("small") == "small"
        assert tracing.cap_attr_value(42) == 42

    def test_helpers_no_op_without_sink_or_turn(self, todo_workflow_path):
        # No sink at all (default no-op): start_span declines, end_span of a
        # None span is silent — nothing raises.
        host = SimpleNamespace(trace_sink=tracing.NoOpTraceSink(), current_turn_key="K")
        assert tracing.start_span(host, tracing.SPAN_AGENT_TOOL_CALL) is None
        tracing.end_span(host, None)
        # Real sink but no open turn: also declines.
        host2 = SimpleNamespace(trace_sink=RecordingTraceSink(), current_turn_key=None)
        assert tracing.start_span(host2, tracing.SPAN_AGENT_TOOL_CALL) is None


# ----------------------------------------------------------------------
# Deterministic path: root span, tool_call span, record, metrics, identity
# ----------------------------------------------------------------------


class TestAssistantPathEmission:
    def test_turn_and_tool_call_spans(
        self, initialized_fastworkflow, todo_workflow_path, monkeypatch
    ):
        sink = RecordingTraceSink()
        ctx, _wf = _make_assistant_ctx(todo_workflow_path, monkeypatch, sink=sink)
        ctx.bind_observability_identity(channel_id="chan-1", conversation_id=7)

        turn_output = ctx.process_turn("add_todo buy milk")

        roots = sink.named(tracing.SPAN_TURN)
        assert len(roots) == 2  # emitted at open and at close
        open_root, closed_root = roots
        assert open_root.span_id == closed_root.span_id == tracing.root_span_id(
            turn_output.turn_key
        )
        assert open_root.trace_id == turn_output.turn_key
        assert open_root.attributes["user_message"] == "add_todo buy milk"
        assert open_root.attributes["channel_id"] == "chan-1"
        assert open_root.attributes["conversation_id"] == 7
        assert closed_root.end_ns is not None
        assert closed_root.status == TurnStatus.COMPLETED.value
        assert closed_root.attributes["success"] is True

        tool_calls = sink.named(tracing.SPAN_AGENT_TOOL_CALL)
        assert len(tool_calls) == 1
        tool_call = tool_calls[0]
        assert tool_call.trace_id == turn_output.turn_key
        assert tool_call.parent_span_id == closed_root.span_id
        assert tool_call.kind == tracing.KIND_TOOL
        assert tool_call.attributes["raw_command"] == "add_todo buy milk"
        assert tool_call.attributes["success"] is True
        assert tool_call.command_name == "add_todo"
        assert tool_call.end_ns is not None
        assert tool_call.channel_id == "chan-1"

    def test_turn_record_emitted_with_identity(
        self, initialized_fastworkflow, todo_workflow_path, monkeypatch
    ):
        sink = RecordingTraceSink()
        ctx, _wf = _make_assistant_ctx(todo_workflow_path, monkeypatch, sink=sink)
        ctx.bind_observability_identity(channel_id="chan-2", conversation_id=3)

        turn_output = ctx.process_turn("list_todos")

        assert len(sink.turn_records) == 1
        record = sink.turn_records[0]
        assert record.turn_output.turn_key == turn_output.turn_key
        assert record.channel_id == "chan-2"
        assert record.conversation_id == 3
        assert record.user_message == "list_todos"

    def test_metrics_emitted_at_finalize(
        self, initialized_fastworkflow, todo_workflow_path, monkeypatch
    ):
        sink = RecordingTraceSink()
        msink = RecordingMetricsSink()
        ctx, _wf = _make_assistant_ctx(todo_workflow_path, monkeypatch, sink=sink)
        ctx.set_metrics_sink(msink)

        ctx.process_turn("list_todos")

        assert ("fw_turns_total", 1, {"status": "completed"}) in msink.counters
        durations = [o for o in msink.observations if o[0] == "fw_turn_duration_seconds"]
        assert len(durations) == 1
        assert durations[0][2] == {"status": "completed"}

    def test_command_trace_events_carry_turn_key(
        self, initialized_fastworkflow, todo_workflow_path, monkeypatch
    ):
        from queue import Queue

        ctx, _wf = _make_assistant_ctx(todo_workflow_path, monkeypatch)
        trace_queue: Queue = Queue()
        ctx.set_transport_queues(command_trace_queue=trace_queue)

        turn_output = ctx.process_turn("list_todos")

        events = []
        while not trace_queue.empty():
            events.append(trace_queue.get())
        trace_events = [e for e in events if e is not None]  # drop sentinel
        assert len(trace_events) == 2  # AGENT_TO_WORKFLOW + WORKFLOW_TO_AGENT
        assert all(e.turn_key == turn_output.turn_key for e in trace_events)

    def test_default_sink_is_no_op_and_turn_unaffected(
        self, initialized_fastworkflow, todo_workflow_path, monkeypatch
    ):
        ctx, _wf = _make_assistant_ctx(todo_workflow_path, monkeypatch)
        turn_output = ctx.process_turn("list_todos")
        assert turn_output.success
        assert isinstance(ctx.trace_sink, tracing.NoOpTraceSink)

    def test_raising_sink_never_fails_the_turn(
        self, initialized_fastworkflow, todo_workflow_path, monkeypatch
    ):
        ctx, _wf = _make_assistant_ctx(
            todo_workflow_path, monkeypatch, sink=RaisingTraceSink()
        )
        turn_output = ctx.process_turn("list_todos")
        assert turn_output.success
        assert turn_output.status == TurnStatus.COMPLETED


# ----------------------------------------------------------------------
# fw.command.execute at the real invoke_command boundary
# ----------------------------------------------------------------------


class TestCommandExecuteSpan:
    def test_invoke_command_emits_nested_span(
        self, initialized_fastworkflow, todo_workflow_path, monkeypatch
    ):
        """Real invoke_command with only perform_action (the wildcard/NLU
        boundary) faked: the fw.command.execute span nests under the
        fw.agent.tool_call span via the WEC span stack."""
        sink = RecordingTraceSink()
        wf = fastworkflow.Workflow.create(
            todo_workflow_path,
            workflow_id_str=f"tracing-invoke-{uuid.uuid4().hex}",
        )
        ctx = WorkflowExecutionContext(run_as_agent=False, trace_sink=sink)
        ctx.bind_app_workflow(wf)

        def fake_perform_action(cls, workflow, action):
            return fastworkflow.CommandOutput(
                command_name=action.command_name,
                command_response=fastworkflow.CommandResponse(
                    response="handled",
                    artifacts={"command_handled": True},
                ),
            )

        monkeypatch.setattr(
            CommandExecutor, "perform_action", classmethod(fake_perform_action)
        )

        turn_output = ctx.process_turn("what_can_i_do")
        assert turn_output.success

        executes = sink.named(tracing.SPAN_COMMAND_EXECUTE)
        assert len(executes) == 1
        execute = executes[0]
        tool_call = sink.named(tracing.SPAN_AGENT_TOOL_CALL)[0]
        assert execute.parent_span_id == tool_call.span_id
        assert execute.trace_id == turn_output.turn_key
        assert execute.kind == tracing.KIND_TOOL
        assert execute.attributes["raw_command"] == "what_can_i_do"
        assert execute.attributes["success"] is True
        assert execute.end_ns is not None
        assert execute.end_ns >= execute.start_ns


# ----------------------------------------------------------------------
# fw.ask_user span pair (both topologies funnel through the accumulator)
# ----------------------------------------------------------------------


class TestAskUserSpans:
    def test_ask_user_open_and_close(
        self, initialized_fastworkflow, todo_workflow_path, monkeypatch
    ):
        sink = RecordingTraceSink()
        ctx, _wf = _make_assistant_ctx(todo_workflow_path, monkeypatch, sink=sink)
        ctx._begin_turn("do something")

        ctx.append_ask_user_entry("Which list?")
        opened = sink.named(tracing.SPAN_ASK_USER)
        assert len(opened) == 1
        assert opened[0].kind == tracing.KIND_HUMAN_WAIT
        assert opened[0].status == tracing.STATUS_OPEN
        assert opened[0].end_ns is None
        assert opened[0].span_id == tracing.deterministic_span_id(
            ctx.current_turn_key, tracing.SPAN_ASK_USER, 0
        )
        assert opened[0].parent_span_id == tracing.root_span_id(ctx.current_turn_key)
        assert opened[0].attributes["agent_query"] == "Which list?"

        ctx.complete_ask_user_entry("the groceries list")
        ask_spans = sink.named(tracing.SPAN_ASK_USER)
        assert len(ask_spans) == 2
        closed = ask_spans[1]
        assert closed.span_id == opened[0].span_id  # idempotent-upsert identity [R6]
        assert closed.end_ns is not None
        assert closed.status == tracing.STATUS_OK
        assert closed.attributes["user_response"] == "the groceries list"
        assert closed.attributes["agent_query"] == "Which list?"

    def test_second_ask_user_gets_new_attempt_id(
        self, initialized_fastworkflow, todo_workflow_path, monkeypatch
    ):
        sink = RecordingTraceSink()
        ctx, _wf = _make_assistant_ctx(todo_workflow_path, monkeypatch, sink=sink)
        ctx._begin_turn("do something")

        ctx.append_ask_user_entry("Q1?")
        ctx.complete_ask_user_entry("A1")
        ctx.append_ask_user_entry("Q2?")
        ctx.complete_ask_user_entry("A2")

        ids = {s.span_id for s in sink.named(tracing.SPAN_ASK_USER)}
        assert ids == {
            tracing.deterministic_span_id(ctx.current_turn_key, tracing.SPAN_ASK_USER, 0),
            tracing.deterministic_span_id(ctx.current_turn_key, tracing.SPAN_ASK_USER, 1),
        }


# ----------------------------------------------------------------------
# Suspension: root span stays open across ask_user, closes on resume
# ----------------------------------------------------------------------


class TestSuspensionRootSpan:
    def test_awaiting_then_resume_closes_same_root(
        self, initialized_fastworkflow, todo_workflow_path, monkeypatch
    ):
        sink = RecordingTraceSink()
        ctx, _wf = _make_agent_ctx(todo_workflow_path, monkeypatch, sink=sink)

        suspended = SimpleNamespace(suspended=True, clarification="Which task?")
        completed = SimpleNamespace(final_answer="All done")
        mock_agent = MagicMock()
        mock_agent.return_value = suspended
        mock_agent.resume.return_value = completed
        _set_agents(ctx, mock_agent)

        first = ctx.process_turn("clean up my tasks")
        assert first.status == TurnStatus.AWAITING_USER

        roots = sink.named(tracing.SPAN_TURN)
        # open + awaiting-update, same deterministic id, still open
        assert len(roots) == 2
        assert roots[1].status == TurnStatus.AWAITING_USER.value
        assert roots[1].end_ns is None
        # ask_user human-wait span opened at suspension
        assert len(sink.named(tracing.SPAN_ASK_USER)) == 1
        # the suspended turn's record is already visible [R2]
        assert len(sink.turn_records) == 1
        assert sink.turn_records[0].turn_output.status == TurnStatus.AWAITING_USER

        second = ctx.process_turn("the urgent one")
        assert second.status == TurnStatus.COMPLETED
        assert second.turn_key == first.turn_key  # same logical turn [A30.2]

        roots = sink.named(tracing.SPAN_TURN)
        assert len(roots) == 3
        final_root = roots[2]
        assert final_root.span_id == tracing.root_span_id(first.turn_key)
        assert final_root.status == TurnStatus.COMPLETED.value
        assert final_root.end_ns is not None
        # ask_user span closed with the user's answer
        ask_spans = sink.named(tracing.SPAN_ASK_USER)
        assert len(ask_spans) == 2
        assert ask_spans[1].end_ns is not None
        assert ask_spans[1].attributes["user_response"] == "the urgent one"
        # terminal record emitted for the same turn key
        assert len(sink.turn_records) == 2
        assert sink.turn_records[1].turn_output.status == TurnStatus.COMPLETED


# ----------------------------------------------------------------------
# Planner spans
# ----------------------------------------------------------------------


class TestPlannerSpans:
    def _plan(self, ctx, wf, monkeypatch, trace_trigger=None):
        import dspy

        from fastworkflow import workflow_agent
        from fastworkflow.command_metadata_api import CommandMetadataAPI

        monkeypatch.setattr(
            CommandMetadataAPI,
            "get_all_contexts_command_display_text",
            staticmethod(lambda **kwargs: "commands"),
        )
        monkeypatch.setattr(
            "fastworkflow.utils.dspy_utils.get_lm",
            lambda *a, **kw: SimpleNamespace(model="test-model"),
        )
        monkeypatch.setattr(
            dspy,
            "ChainOfThought",
            lambda signature: (
                lambda **kwargs: SimpleNamespace(
                    next_steps="1. step one\n2. step two", reasoning=""
                )
            ),
        )

        class _NullCtx:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        monkeypatch.setattr(dspy, "context", lambda **kwargs: _NullCtx())

        ctx.push_active_workflow(wf)
        try:
            return workflow_agent.build_query_with_next_steps(
                "do the thing", ctx, trace_trigger=trace_trigger
            )
        finally:
            ctx.pop_active_workflow()

    def test_initial_plan_span(
        self, initialized_fastworkflow, todo_workflow_path, monkeypatch
    ):
        sink = RecordingTraceSink()
        ctx, wf = _make_assistant_ctx(todo_workflow_path, monkeypatch, sink=sink)
        ctx._begin_turn("do the thing")

        result = self._plan(ctx, wf, monkeypatch)
        assert "step one" in result

        plans = sink.named(tracing.SPAN_PLANNER_PLAN)
        assert len(plans) == 1
        assert plans[0].kind == tracing.KIND_LLM
        assert plans[0].attributes["model"] == "test-model"
        assert plans[0].attributes["replan_trigger"] is None
        assert "step one" in plans[0].attributes["plan"]
        assert not sink.named(tracing.SPAN_PLANNER_REPLAN)

    def test_replan_span_carries_trigger(
        self, initialized_fastworkflow, todo_workflow_path, monkeypatch
    ):
        sink = RecordingTraceSink()
        ctx, wf = _make_assistant_ctx(todo_workflow_path, monkeypatch, sink=sink)
        ctx._begin_turn("do the thing")

        self._plan(ctx, wf, monkeypatch, trace_trigger="ask_user_response")

        replans = sink.named(tracing.SPAN_PLANNER_REPLAN)
        assert len(replans) == 1
        assert replans[0].attributes["replan_trigger"] == "ask_user_response"
        assert not sink.named(tracing.SPAN_PLANNER_PLAN)


# ----------------------------------------------------------------------
# fw.agent.execute + fw.agent.step: the agent loop's own structure
# ----------------------------------------------------------------------


class _NullDSPyContext:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class TestAgentLoopSpans:
    """The executor phase and its per-iteration steps, as real spans.

    Only ReAct's two LLM modules are scripted — the loop, the tool dispatch,
    the tool-call span and the parenting stack are all production code, which
    is the point: these spans exist so the turn's shape stops being something
    a reader has to infer from module names and timestamps.
    """

    def _agent(self, script, tools):
        agent = fastWorkflowReAct.__new__(fastWorkflowReAct)
        agent.iteration_counter = 0
        agent.max_iters = 5
        agent.inputs = {}
        agent.current_trajectory = {}
        agent._suspended = None
        agent.tools = tools
        pending = list(script)
        agent.react = lambda trajectory, **kw: SimpleNamespace(**pending.pop(0))
        agent.extract = lambda trajectory, **kw: {"final_answer": "42"}
        return agent

    def test_execute_wraps_steps_which_wrap_their_tool_calls(
        self, initialized_fastworkflow, todo_workflow_path, monkeypatch
    ):
        import dspy

        sink = RecordingTraceSink()
        ctx, _wf = _make_agent_ctx(todo_workflow_path, monkeypatch, sink=sink)
        ctx._begin_turn("add two numbers")
        monkeypatch.setattr(
            ctx,
            "_agent_dspy_context",
            lambda: (SimpleNamespace(model="test-model"), None),
        )
        monkeypatch.setattr(dspy, "context", lambda **kwargs: _NullDSPyContext())

        def fake_invoke(cls, session, command: str):
            return fastworkflow.CommandOutput(
                command_name=command.split()[0] if command else "",
                command_response=fastworkflow.CommandResponse(response=f"ok:{command}"),
            )

        monkeypatch.setattr(
            CommandExecutor, "invoke_command", classmethod(fake_invoke)
        )
        # _execute_workflow_query consults the CME workflow's NLU stage and the
        # active workflow's context after the command runs; an empty CME
        # context is the "nothing to clarify" case.
        ctx._cme_workflow = SimpleNamespace(context={})
        ctx.push_active_workflow(_wf)

        agent = self._agent(
            [
                {
                    "next_thought": "run the command",
                    "next_tool_name": "execute_workflow_query",
                    "next_tool_args": {"command": "add_todo milk"},
                },
                {"next_thought": "done", "next_tool_name": "finish",
                 "next_tool_args": {}},
            ],
            {
                "execute_workflow_query": lambda command: (
                    workflow_agent._execute_workflow_query(command, chat_session_obj=ctx)
                ),
                "finish": lambda: "Completed.",
            },
        )

        result = ctx._call_agent_with_retry(
            lambda: agent.forward(user_query="add two numbers"),
            trace_input="add two numbers",
        )
        assert result.final_answer == "42"

        executes = sink.named(tracing.SPAN_AGENT_EXECUTE)
        assert len(executes) == 1
        execute = executes[0]
        # The executor is a sibling of planning under the turn, not a child of
        # whatever span happened to be open.
        assert execute.parent_span_id == tracing.root_span_id(ctx.current_turn_key)
        assert execute.attributes["agent_input"] == "add two numbers"
        assert execute.attributes["resumed"] is False
        assert execute.attributes["final_answer"] == "42"
        assert execute.attributes["suspended"] is False
        assert execute.attributes["attempts"] == 1
        assert execute.end_ns is not None

        steps = sink.named(tracing.SPAN_AGENT_STEP)
        assert len(steps) == 2
        assert [s.parent_span_id for s in steps] == [execute.span_id, execute.span_id]
        assert steps[0].attributes["thought"] == "run the command"
        assert steps[0].attributes["tool_name"] == "execute_workflow_query"
        assert steps[0].attributes["tool_args"] == {"command": "add_todo milk"}
        assert steps[0].attributes["observation"] == "ok:add_todo milk"
        assert steps[0].attributes["step_index"] == 0
        assert steps[1].attributes["tool_name"] == "finish"
        assert all(s.end_ns is not None for s in steps)

        # The whole point of the step span: the tool call belongs to a step,
        # rather than being a flat sibling a reader has to re-associate.
        tool_calls = sink.named(tracing.SPAN_AGENT_TOOL_CALL)
        assert len(tool_calls) == 1
        assert tool_calls[0].parent_span_id == steps[0].span_id

        assert ctx.trace_span_stack == []

    def test_suspended_step_closes_rather_than_leaking(
        self, initialized_fastworkflow, todo_workflow_path, monkeypatch
    ):
        """A step that suspends may resume in a different process.

        Leaving its span open would leak it onto the parenting stack for the
        rest of the turn, so every later span would hang off a step that had
        already ended.
        """
        sink = RecordingTraceSink()
        ctx, _wf = _make_agent_ctx(todo_workflow_path, monkeypatch, sink=sink)
        ctx._begin_turn("ask me something")

        def raise_suspend(clarification_request):
            raise AskUserSuspend(clarification_request)

        agent = self._agent(
            [
                {
                    "next_thought": "need input",
                    "next_tool_name": "ask_user",
                    "next_tool_args": {"clarification_request": "Which one?"},
                }
            ],
            {"ask_user": raise_suspend},
        )

        with tracing.host_scope(ctx):
            result = agent._run_loop({}, 0, {"q": "x"}, max_iters=5, exception_count=0)

        assert result.suspended is True
        steps = sink.named(tracing.SPAN_AGENT_STEP)
        assert len(steps) == 1
        assert steps[0].status == tracing.STATUS_AWAITING_USER
        assert steps[0].end_ns is not None
        assert steps[0].attributes["clarification"] == "Which one?"
        assert ctx.trace_span_stack == []

    def test_step_records_a_failed_tool_selection(
        self, initialized_fastworkflow, todo_workflow_path, monkeypatch
    ):
        """A step that never picks a valid tool is still a recorded step."""
        sink = RecordingTraceSink()
        ctx, _wf = _make_agent_ctx(todo_workflow_path, monkeypatch, sink=sink)
        ctx._begin_turn("nonsense")

        agent = self._agent([], {"finish": lambda: "Completed."})
        agent.react = lambda trajectory, **kw: None  # no tool selected

        with tracing.host_scope(ctx):
            agent._run_loop({}, 0, {"q": "x"}, max_iters=5, exception_count=0)

        steps = sink.named(tracing.SPAN_AGENT_STEP)
        assert steps, "a step that failed to select a tool left no span"
        assert all(s.status == tracing.STATUS_ERROR for s in steps)
        assert all("Agent failed to select a valid tool" in s.attributes["observation"]
                   for s in steps)
        assert ctx.trace_span_stack == []


# ----------------------------------------------------------------------
# Identity plumbing
# ----------------------------------------------------------------------


class TestIdentityPlumbing:
    def test_bind_is_partial_and_sticky(self, initialized_fastworkflow):
        ctx = WorkflowExecutionContext()
        assert ctx.observability_channel_id is None
        ctx.bind_observability_identity(channel_id="chan-9")
        ctx.bind_observability_identity(conversation_id=42)
        assert ctx.observability_channel_id == "chan-9"
        assert ctx.observability_conversation_id == 42

    def test_chat_session_binds_synthetic_cli_channel(self, initialized_fastworkflow):
        session = fastworkflow.ChatSession()
        channel = session._core.observability_channel_id
        assert channel is not None
        assert channel.startswith("cli:")

    def test_turn_result_has_channel_field(self):
        # Additive TurnResult field [R1]; default None keeps old callers valid.
        from fastworkflow.turn import TurnResult

        assert "channel_id" in TurnResult.model_fields
        assert TurnResult.model_fields["channel_id"].default is None

    def test_bare_embedder_gets_a_conversation_minted_for_it(
        self, initialized_fastworkflow, tmp_path
    ):
        """FastAPI and the CLI bind an id; code embedding the core directly has
        no layer that would, and its turns must still group somewhere [R17]."""
        from fastworkflow.observability_store import SQLiteTraceSink

        sink = SQLiteTraceSink(str(tmp_path / "observability.sqlite3"))
        try:
            ctx = WorkflowExecutionContext()
            ctx.set_trace_sink(sink)
            ctx.bind_observability_identity(channel_id="embedder-channel")
            assert ctx.observability_conversation_id is None

            ctx._begin_turn("first message")
            assert ctx.observability_conversation_id == 1
            # One conversation per context, not one per turn.
            ctx._begin_turn("second message")
            assert ctx.observability_conversation_id == 1
        finally:
            sink.close()

    def test_a_bound_conversation_id_is_never_replaced(
        self, initialized_fastworkflow, tmp_path
    ):
        from fastworkflow.observability_store import SQLiteTraceSink

        sink = SQLiteTraceSink(str(tmp_path / "observability.sqlite3"))
        try:
            ctx = WorkflowExecutionContext()
            ctx.set_trace_sink(sink)
            ctx.bind_observability_identity(channel_id="chan-x", conversation_id=77)
            ctx._begin_turn("hello")
            assert ctx.observability_conversation_id == 77
        finally:
            sink.close()

    def test_no_conversation_is_minted_without_a_sink(self, initialized_fastworkflow):
        """Observability off must stay free: no DB is opened to mint an id."""
        ctx = WorkflowExecutionContext()
        ctx.bind_observability_identity(channel_id="chan-y")
        ctx._begin_turn("hello")
        assert ctx.observability_conversation_id is None


# ----------------------------------------------------------------------
# CommandTraceEvent additive field + live CLI rendering contract
# ----------------------------------------------------------------------


class TestCommandTraceEventAdditive:
    def test_turn_key_defaults_to_none(self):
        event = fastworkflow.CommandTraceEvent(
            direction=fastworkflow.CommandTraceEventDirection.AGENT_TO_WORKFLOW,
            raw_command="x",
            command_name=None,
            parameters=None,
            response_text=None,
            success=None,
            timestamp_ms=0,
        )
        assert event.turn_key is None


# ----------------------------------------------------------------------
# Final-review regression guards (fix-kw7 epic review, 2026-08-26)
# ----------------------------------------------------------------------


class TestStepSpanExceptionSafety:
    """A step span must close on EVERY exit — a leaked step on the parenting
    stack makes the retried attempt's whole subtree parent under a phantom
    span that is never emitted."""

    def _agent(self, react_fn, tools):
        agent = fastWorkflowReAct.__new__(fastWorkflowReAct)
        agent.iteration_counter = 0
        agent.max_iters = 5
        agent.inputs = {}
        agent.current_trajectory = {}
        agent._suspended = None
        agent.tools = tools
        agent.react = react_fn
        agent.extract = lambda trajectory, **kw: {"final_answer": "42"}
        return agent

    def test_non_valueerror_from_reasoning_closes_the_step(
        self, initialized_fastworkflow, todo_workflow_path, monkeypatch
    ):
        # AdapterParseError and provider errors are NOT ValueErrors; the
        # caller's retry loop re-enters _run_loop afterwards.
        sink = RecordingTraceSink()
        ctx, _wf = _make_agent_ctx(todo_workflow_path, monkeypatch, sink=sink)
        ctx._begin_turn("boom")

        def exploding_react(trajectory, **kw):
            raise RuntimeError("adapter parse failure")

        agent = self._agent(exploding_react, {"finish": lambda: "done"})
        with tracing.host_scope(ctx):
            with pytest.raises(RuntimeError):
                agent._run_loop({}, 0, {"q": "x"}, max_iters=5, exception_count=0)

        steps = sink.named(tracing.SPAN_AGENT_STEP)
        assert len(steps) == 1
        assert steps[0].status == tracing.STATUS_ERROR
        assert steps[0].end_ns is not None
        assert steps[0].attributes["error_type"] == "RuntimeError"
        assert ctx.trace_span_stack == []

    def test_control_signal_from_a_tool_closes_the_step(
        self, initialized_fastworkflow, todo_workflow_path, monkeypatch
    ):
        from fastworkflow.workflow_execution_context import CommandCancelledError

        sink = RecordingTraceSink()
        ctx, _wf = _make_agent_ctx(todo_workflow_path, monkeypatch, sink=sink)
        ctx._begin_turn("cancel me")

        script = [
            {
                "next_thought": "call it",
                "next_tool_name": "cancel_tool",
                "next_tool_args": {},
            }
        ]

        def cancel_tool():
            raise CommandCancelledError("user cancelled")

        agent = self._agent(
            lambda trajectory, **kw: SimpleNamespace(**script.pop(0)),
            {"cancel_tool": cancel_tool},
        )
        with tracing.host_scope(ctx):
            with pytest.raises(CommandCancelledError):
                agent._run_loop({}, 0, {"q": "x"}, max_iters=5, exception_count=0)

        steps = sink.named(tracing.SPAN_AGENT_STEP)
        assert len(steps) == 1
        assert steps[0].status == tracing.STATUS_CANCELLED
        assert steps[0].end_ns is not None
        assert ctx.trace_span_stack == []


class TestContextMutationEdgeCases:
    def test_mixed_type_removed_keys_never_fail_the_turn(
        self, initialized_fastworkflow, todo_workflow_path
    ):
        """workflow.context is app-authored: removing a str key and an int key
        in one turn made plain sorted() raise TypeError THROUGH finalize."""
        sink = RecordingTraceSink()
        wf = fastworkflow.Workflow.create(
            todo_workflow_path,
            workflow_id_str=f"ctxmix-{uuid.uuid4().hex}",
            workflow_context={"str_key": 1, 7: "int-keyed", "keep": True},
        )
        ctx = WorkflowExecutionContext(run_as_agent=False, trace_sink=sink)
        ctx.bind_app_workflow(wf)
        ctx._begin_turn("remove mixed keys")

        workflow_context = wf.context
        del workflow_context["str_key"]
        del workflow_context[7]
        wf.context = workflow_context

        ctx.finalize_turn_for_observability(
            fastworkflow.CommandOutput(
                command_name="noop",
                command_response=fastworkflow.CommandResponse(response="ok"),
            )
        )
        closes = [s for s in sink.named(tracing.SPAN_TURN) if s.end_ns is not None]
        assert closes, "turn failed to finalize"
        mutations = closes[-1].attributes["context_mutations"]
        assert set(map(str, mutations["removed"])) == {"str_key", "7"}


class TestEmbedderOwnedConversationSuppression:
    def test_fastapi_style_embedder_suppresses_self_mint(
        self, initialized_fastworkflow, tmp_path
    ):
        """Ruling C2: an embedder whose chokepoint mints with the legacy floor
        declares ownership; the WEC must NOT floor-lessly self-mint on its
        degraded path (which would alias legacy ids and split the session)."""
        from fastworkflow.observability_store import SQLiteTraceSink

        sink = SQLiteTraceSink(str(tmp_path / "observability.sqlite3"))
        try:
            ctx = WorkflowExecutionContext()
            ctx.set_trace_sink(sink)
            ctx.bind_observability_identity(
                channel_id="fastapi-chan", embedder_owns_conversations=True
            )
            ctx._begin_turn("first message")
            assert ctx.observability_conversation_id is None  # stays unbound
            # The embedder's own (floor-carrying) mint still binds normally.
            ctx.bind_observability_identity(conversation_id=5)
            assert ctx.observability_conversation_id == 5
        finally:
            sink.close()


class TestDisabledObservabilityDspyCost:
    def test_observe_dspy_host_is_inert_without_a_sink(
        self, initialized_fastworkflow, todo_workflow_path
    ):
        """FW_OBSERVABILITY=0 must cost ~nothing per LM call: with no live
        sink the DSPy callback is never bound, so on_lm_start's per-call
        prompt JSON projection never runs."""
        from fastworkflow.utils import dspy_logger

        wf = fastworkflow.Workflow.create(
            todo_workflow_path, workflow_id_str=f"nosink-{uuid.uuid4().hex}"
        )
        ctx = WorkflowExecutionContext(run_as_agent=False)  # NoOp sink
        ctx.bind_app_workflow(wf)
        with dspy_logger.observe_dspy_host(ctx):
            assert dspy_logger._active_observability_host.get() is None

        sink = RecordingTraceSink()
        ctx.set_trace_sink(sink)
        with dspy_logger.observe_dspy_host(ctx):
            assert dspy_logger._active_observability_host.get() is ctx
        assert dspy_logger._active_observability_host.get() is None

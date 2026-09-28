"""Which planner runs: one "finish check active" decision, shared with the agent.

A real todo_list_workflow session and a real tool agent built from settings
loaded out of the workflow's own ``fastworkflow.env`` (the way ``fastworkflow
run`` loads them). The planner LM is dspy's ``DummyLM``, which records every
prompt it is sent; nothing here makes a network call -- a finish checker is
built, never asked.
"""
from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from dotenv import dotenv_values
from dspy.utils import DummyLM

import fastworkflow
from fastworkflow import tracing
from fastworkflow.command_routing import RoutingRegistry
from fastworkflow.observation_offloading import jev_client
from fastworkflow.observation_offloading.finish_check import CHECK_ENV, KEY_ENV, MODEL_ENV
from fastworkflow.observation_offloading.state import snapshot_events
from fastworkflow.turn_plan import PlanStep, TurnPlan
from fastworkflow.utils.react import EVAL_FINISH_REMINDERS_ENV
from fastworkflow.workflow_agent import (
    build_query_with_next_steps,
    finish_check_active,
    initialize_workflow_tool_agent,
)
from fastworkflow.workflow_execution_context import WorkflowExecutionContext

needs_sdk = pytest.mark.skipif(jev_client.TypeSafeClient is None, reason="typesafe-sdk not installed")

TODO_WORKFLOW = str(Path(__file__).parent.joinpath("todo_list_workflow").resolve())
COMMANDS_TITLE = "Available execute_workflow_query tool commands"
STRUCTURED_MARK = "Return the plan as structured steps"
TEXT_MARK = "[[ ## next_steps ## ]]"


class RecordingTraceSink:
    def __init__(self):
        self.spans: list[tracing.Span] = []

    def emit_span(self, span: tracing.Span) -> None:
        self.spans.append(span)

    def emit_turn_record(self, record) -> None:
        pass

    def record_conversation_label(self, *args) -> None:
        pass

    def named(self, name: str) -> list[tracing.Span]:
        return [s for s in self.spans if s.name == name]


@pytest.fixture
def workflow_env(tmp_path, monkeypatch):
    """Load settings from a real fastworkflow.env / passwords file pair, process env cleared."""
    for name in (CHECK_ENV, KEY_ENV, MODEL_ENV, EVAL_FINISH_REMINDERS_ENV):
        monkeypatch.delenv(name, raising=False)
    RoutingRegistry.clear_registry()

    def load(**settings: str) -> None:
        env_file = tmp_path / "fastworkflow.env"
        passwords_file = tmp_path / "fastworkflow.passwords.env"
        env_file.write_text("".join(
            f"{name}={value}\n" for name, value in settings.items() if name != KEY_ENV))
        passwords_file.write_text(f"{KEY_ENV}={settings[KEY_ENV]}\n" if KEY_ENV in settings else "")
        fastworkflow.init(env_vars={**dotenv_values(env_file), **dotenv_values(passwords_file)})

    yield load
    RoutingRegistry.clear_registry()


def _session(sink):
    ctx = WorkflowExecutionContext(run_as_agent=True, trace_sink=sink)
    wf = fastworkflow.Workflow.create(TODO_WORKFLOW, workflow_id_str=f"planner-{uuid.uuid4().hex}")
    ctx.bind_app_workflow(wf)
    ctx._workflow_tool_agent = initialize_workflow_tool_agent(ctx)
    return ctx, wf


def _plan(ctx, wf, lm, **kwargs):
    ctx.push_active_workflow(wf)
    try:
        return build_query_with_next_steps("show my todo items", ctx, planner_lm=lm, **kwargs)
    finally:
        ctx.pop_active_workflow()


def _prompt(lm, index: int) -> str:
    return "\n".join(str(m.get("content", "")) for m in lm.history[index]["messages"])


def _structured_answer():
    return {
        "reasoning": "r",
        "subjects": [],
        "steps": [{"text": "Show all todo items", "commands": ["show_all_todos"], "optional": True}],
    }


def _text_answer():
    return {"reasoning": "r", "next_steps": "1. Show all todo items"}


@needs_sdk
def test_reminders_off_in_the_workflow_env_file_is_the_control_arm(workflow_env):
    workflow_env(**{CHECK_ENV: "jev", KEY_ENV: "k1", EVAL_FINISH_REMINDERS_ENV: "0"})
    sink = RecordingTraceSink()
    ctx, wf = _session(sink)
    agent = ctx.workflow_tool_agent
    assert agent.finish_checker is not None
    assert agent.finish_reminders_enabled is False
    assert not finish_check_active(agent)
    controls = [e for e in snapshot_events()
                if e["kind"] == "evaluation_controls" and e["scope_id"] == agent.continuation_scope_id]
    assert controls and controls[-1]["finish_reminders_enabled"] is False

    ctx._begin_turn("show my todo items")
    lm = DummyLM([_text_answer()])
    result = _plan(ctx, wf, lm)

    assert len(lm.history) == 1
    assert TEXT_MARK in _prompt(lm, 0) and STRUCTURED_MARK not in _prompt(lm, 0)
    assert "(optional)" not in result and "(needs the user)" not in result
    plan, = sink.named(tracing.SPAN_PLANNER_PLAN)
    assert plan.attributes["plan_source"] == "text"
    assert ctx._turn_plan is None


@needs_sdk
def test_with_reminders_on_the_text_planner_runs_and_its_plan_is_parsed(workflow_env):
    workflow_env(**{CHECK_ENV: "jev", KEY_ENV: "k1"})
    sink = RecordingTraceSink()
    ctx, wf = _session(sink)
    assert finish_check_active(ctx.workflow_tool_agent)

    ctx._begin_turn("show my todo items")
    lm = DummyLM([_text_answer()])
    result = _plan(ctx, wf, lm)

    assert len(lm.history) == 1
    assert TEXT_MARK in _prompt(lm, 0) and STRUCTURED_MARK not in _prompt(lm, 0)
    assert COMMANDS_TITLE in _prompt(lm, 0)
    assert "Show all todo items" in result
    plan, = sink.named(tracing.SPAN_PLANNER_PLAN)
    assert (plan.attributes["plan_source"], plan.attributes["subjects"]) == ("text", [])
    assert ctx._turn_plan is not None and ctx._turn_plan.source == "text"
    assert [step.text for step in ctx._turn_plan.steps] == ["Show all todo items"]
    assert ctx._turn_plan.subjects == [] and ctx._turn_plan_status == "planned"


@pytest.mark.skip(reason="structured planning disabled 2026-09-28 (owner decision)")
@needs_sdk
def test_with_reminders_on_the_structured_planner_runs(workflow_env):
    workflow_env(**{CHECK_ENV: "jev", KEY_ENV: "k1"})
    sink = RecordingTraceSink()
    ctx, wf = _session(sink)
    assert finish_check_active(ctx.workflow_tool_agent)

    ctx._begin_turn("show my todo items")
    lm = DummyLM([_structured_answer()])
    result = _plan(ctx, wf, lm)

    assert len(lm.history) == 1
    assert STRUCTURED_MARK in _prompt(lm, 0)
    assert COMMANDS_TITLE in _prompt(lm, 0)
    assert "Show all todo items (optional)" in result
    plan, = sink.named(tracing.SPAN_PLANNER_PLAN)
    assert plan.attributes["plan_source"] == "structured"
    assert ctx._turn_plan is not None and ctx._turn_plan.source == "structured"


@needs_sdk
def test_a_replan_uses_the_text_planner_and_keeps_the_initial_plan(workflow_env):
    workflow_env(**{CHECK_ENV: "jev", KEY_ENV: "k1"})
    sink = RecordingTraceSink()
    ctx, wf = _session(sink)
    assert finish_check_active(ctx.workflow_tool_agent)
    initial = TurnPlan(steps=[PlanStep(text="Show all todo items", commands=["show_all_todos"])])
    ctx._turn_plan = initial

    ctx._begin_turn("show my todo items")
    lm = DummyLM([_text_answer()])
    _plan(ctx, wf, lm, with_agent_inputs_and_trajectory=True, trace_trigger="ask_user_response")

    assert len(lm.history) == 1
    assert TEXT_MARK in _prompt(lm, 0) and STRUCTURED_MARK not in _prompt(lm, 0)
    replan, = sink.named(tracing.SPAN_PLANNER_REPLAN)
    assert replan.attributes["plan_source"] == "text"
    assert ctx._turn_plan is initial


@pytest.mark.skip(reason="structured planning disabled 2026-09-28 (owner decision)")
@needs_sdk
def test_an_unparseable_structured_reply_goes_straight_to_the_text_planner(workflow_env):
    workflow_env(**{CHECK_ENV: "jev", KEY_ENV: "k1"})
    sink = RecordingTraceSink()
    ctx, wf = _session(sink)

    ctx._begin_turn("show my todo items")
    lm = DummyLM([{"reasoning": "r", "unrelated": "not a plan"}, _text_answer(), _text_answer()])
    result = _plan(ctx, wf, lm)

    assert len(lm.history) == 2
    assert STRUCTURED_MARK in _prompt(lm, 0)
    assert TEXT_MARK in _prompt(lm, 1)
    assert all(COMMANDS_TITLE in _prompt(lm, i) for i in range(len(lm.history)))
    assert "Show all todo items" in result
    plan, = sink.named(tracing.SPAN_PLANNER_PLAN)
    assert plan.attributes["plan_source"] == "text_fallback"


def test_without_the_check_the_planner_is_text_only(workflow_env):
    workflow_env()
    sink = RecordingTraceSink()
    ctx, wf = _session(sink)
    assert ctx.workflow_tool_agent.finish_checker is None

    ctx._begin_turn("show my todo items")
    lm = DummyLM([_text_answer()])
    _plan(ctx, wf, lm)

    assert len(lm.history) == 1
    assert TEXT_MARK in _prompt(lm, 0) and COMMANDS_TITLE in _prompt(lm, 0)
    plan, = sink.named(tracing.SPAN_PLANNER_PLAN)
    assert plan.attributes["plan_source"] == "text"


def test_a_reminders_value_other_than_zero_in_the_env_file_is_refused(workflow_env):
    workflow_env(**{EVAL_FINISH_REMINDERS_ENV: "1"})
    ctx = WorkflowExecutionContext(run_as_agent=True)
    ctx.bind_app_workflow(fastworkflow.Workflow.create(
        TODO_WORKFLOW, workflow_id_str=f"planner-{uuid.uuid4().hex}"))
    with pytest.raises(ValueError, match=f"{EVAL_FINISH_REMINDERS_ENV} must be exactly 0"):
        initialize_workflow_tool_agent(ctx)

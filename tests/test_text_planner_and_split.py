"""Plain-text planner: signature choice and span attributes."""
from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from dotenv import dotenv_values
from dspy.utils import DummyLM

import fastworkflow
from fastworkflow import tracing
from fastworkflow.command_routing import RoutingRegistry
from fastworkflow.workflow_agent import (
    build_query_with_next_steps, initialize_workflow_tool_agent, run_progress_check)
from fastworkflow.workflow_execution_context import WorkflowExecutionContext

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
    RoutingRegistry.clear_registry()

    def load(**settings: str) -> None:
        env_file = tmp_path / "fastworkflow.env"
        passwords_file = tmp_path / "fastworkflow.passwords.env"
        env_file.write_text("".join(f"{name}={value}\n" for name, value in settings.items()))
        passwords_file.write_text("")
        fastworkflow.init(env_vars={**dotenv_values(env_file), **dotenv_values(passwords_file)})

    yield load
    RoutingRegistry.clear_registry()


def _session(sink, folder=TODO_WORKFLOW):
    ctx = WorkflowExecutionContext(run_as_agent=True, trace_sink=sink)
    wf = fastworkflow.Workflow.create(folder, workflow_id_str=f"planner-{uuid.uuid4().hex}")
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


def _text_answer():
    return {"reasoning": "r", "next_steps": "1. Show all todo items"}


def _replan_answer():
    return {"reasoning": "r", "whats_done": "Listed the todo items", "next_steps": "1. Show all todo items"}


def test_the_text_planner_runs_and_appends_next_steps(workflow_env):
    workflow_env()
    sink = RecordingTraceSink()
    ctx, wf = _session(sink)

    ctx._begin_turn("show my todo items")
    lm = DummyLM([_text_answer()])
    result = _plan(ctx, wf, lm)

    assert len(lm.history) == 1
    assert TEXT_MARK in _prompt(lm, 0) and STRUCTURED_MARK not in _prompt(lm, 0)
    assert COMMANDS_TITLE in _prompt(lm, 0)
    assert "Show all todo items" in result
    assert "Execute these next steps:" in result
    plan, = sink.named(tracing.SPAN_PLANNER_PLAN)
    assert plan.attributes["plan_source"] == "text"


def test_a_replan_uses_the_text_planner(workflow_env):
    workflow_env()
    sink = RecordingTraceSink()
    ctx, wf = _session(sink)

    ctx._begin_turn("show my todo items")
    lm = DummyLM([_replan_answer(), _replan_answer()])
    result = _plan(ctx, wf, lm, with_agent_inputs_and_trajectory=True, trace_trigger="ask_user_response")

    assert len(lm.history) == 1
    assert result == (
        "User Query:\nshow my todo items\n\n"
        "What's done:\nListed the todo items\n\n"
        "What remains (execute these next steps):\n1. Show all todo items"
    )
    assert TEXT_MARK in _prompt(lm, 0) and STRUCTURED_MARK not in _prompt(lm, 0)
    replan, = sink.named(tracing.SPAN_PLANNER_REPLAN)
    assert replan.attributes["plan_source"] == "text"
    assert not sink.named(tracing.SPAN_PLANNER_PLAN)


def test_the_progress_check_tool_replans(workflow_env):
    workflow_env()
    sink = RecordingTraceSink()
    ctx, wf = _session(sink)
    ctx._current_planner_lm = DummyLM([_replan_answer()])
    agent = ctx._workflow_tool_agent
    agent.iteration_counter = 24

    ctx._begin_turn("show my todo items")
    ctx.push_active_workflow(wf)
    try:
        result = run_progress_check(ctx, "progress_check")
    finally:
        ctx.pop_active_workflow()

    assert result.startswith("What's done:\nListed the todo items")
    assert "What remains (execute these next steps):\n1. Show all todo items" in result
    replan, = sink.named(tracing.SPAN_PLANNER_REPLAN)
    assert replan.attributes["replan_trigger"] == "progress_check"


def test_a_progress_check_with_nothing_remaining_says_to_finish(workflow_env):
    workflow_env()
    ctx, wf = _session(RecordingTraceSink())

    ctx._begin_turn("show my todo items")
    lm = DummyLM([{"reasoning": "r", "whats_done": "Listed the todo items", "next_steps": ""}])
    result = _plan(ctx, wf, lm, with_agent_inputs_and_trajectory=True, trace_trigger="progress_check")

    assert result.endswith("What's done:\nListed the todo items\n\nNothing left to do. Call finish now.")



def test_a_zero_argument_tool_ignores_a_stray_argument(workflow_env, monkeypatch):
    import fastworkflow.workflow_agent as workflow_agent

    workflow_env()
    ctx, wf = _session(RecordingTraceSink())
    monkeypatch.setattr(workflow_agent, "_what_can_i_do", lambda chat_session_obj: "the commands")
    tool = ctx._workflow_tool_agent.tools["what_can_i_do"]

    assert tool.args == {}
    assert tool(clarification_request="list the commands please") == "the commands"


def _reentry_session(monkeypatch, workflow_env):
    import fastworkflow.workflow_agent as workflow_agent
    from fastworkflow.command_executor import AlreadyInContextError

    workflow_env()
    ctx, wf = _session(RecordingTraceSink())
    calls = []
    state = {"inside": True}

    def fake_execute(command, chat_session_obj):
        calls.append(command)
        if command == "go_up":
            state["inside"] = False
            return "Context is now 'ItemExplorer'"
        if state["inside"]:
            raise AlreadyInContextError("You are already in context 'Account' (a1 Jane Roe)")
        return "Entered Account context."

    monkeypatch.setattr(workflow_agent, "_execute_workflow_query", fake_execute)
    monkeypatch.setattr(workflow_agent, "context_clause_for", lambda wf: "Account a1 Jane Roe")
    return ctx, wf, calls


def test_opening_another_instance_of_the_current_context_goes_up_first(workflow_env, monkeypatch):
    ctx, wf, calls = _reentry_session(monkeypatch, workflow_env)
    execute = ctx._workflow_tool_agent.tools["execute_workflow_query"]

    observation = execute(command="open_account_by_uid b2")

    assert calls == ["open_account_by_uid b2", "go_up", "open_account_by_uid b2"]
    assert "(Left the current Account with go_up first.)\nEntered Account context." in observation


def test_reopening_the_same_instance_is_refused_without_moving(workflow_env, monkeypatch):
    ctx, wf, calls = _reentry_session(monkeypatch, workflow_env)
    execute = ctx._workflow_tool_agent.tools["execute_workflow_query"]

    observation = execute(command="open_account_by_uid a1")

    assert calls == ["open_account_by_uid a1"]
    assert "You are already in context 'Account'" in observation


def _split(answer):
    from fastworkflow.workflow_agent import split_into_subtasks
    return split_into_subtasks("the request", planner_lm=DummyLM([answer]))


def test_a_numbered_answer_becomes_the_subtasks():
    assert _split({"reasoning": "r", "subtasks": "1. Audit right A.\n2) Audit right B.\n  3. Sweep Hubbard."}) == [
        "Audit right A.", "Audit right B.", "Sweep Hubbard."]


def test_an_unusable_or_too_long_split_leaves_the_request_whole():
    from fastworkflow.workflow_agent import MAX_SUBTASKS

    assert _split({"reasoning": "r", "subtasks": "no numbered lines here"}) == ["the request"]
    many = "\n".join(f"{n}. part {n}" for n in range(1, MAX_SUBTASKS + 2))
    assert _split({"reasoning": "r", "subtasks": many}) == ["the request"]


def test_a_failing_split_leaves_the_request_whole(monkeypatch):
    import dspy
    from fastworkflow.workflow_agent import split_into_subtasks

    def broken(*args, **kwargs):
        raise RuntimeError("provider down")

    monkeypatch.setattr(dspy, "ChainOfThought", broken)
    assert split_into_subtasks("the request", planner_lm=DummyLM([{}])) == ["the request"]


def test_a_visibly_broken_split_is_retried_once_then_left_whole():
    from fastworkflow.workflow_agent import split_into_subtasks

    duplicated = {"reasoning": "r", "subtasks": "1. Same thing.\n2. Same thing."}
    fixed = {"reasoning": "r", "subtasks": "1. Part A.\n2. Part B."}
    assert split_into_subtasks("the request", planner_lm=DummyLM([duplicated, fixed])) == [
        "Part A.", "Part B."]
    assert split_into_subtasks("the request", planner_lm=DummyLM([duplicated, duplicated])) == [
        "the request"]


def test_a_split_that_refers_back_to_the_request_is_retried_once():
    from fastworkflow.workflow_agent import split_into_subtasks

    referring = {"reasoning": "r", "subtasks": "1. Audit each of them.\n2. Sweep Hubbard."}
    named = {"reasoning": "r", "subtasks": "1. Audit Alice and Bob.\n2. Sweep Hubbard."}
    assert split_into_subtasks("the request", planner_lm=DummyLM([referring, named])) == [
        "Audit Alice and Bob.", "Sweep Hubbard."]
    # Not fixed by the retry: a split with a leftover reference beats no split.
    assert split_into_subtasks("the request", planner_lm=DummyLM([referring, referring])) == [
        "Audit each of them.", "Sweep Hubbard."]


def test_a_clean_split_is_not_retried():
    from fastworkflow.workflow_agent import split_into_subtasks

    clean = {"reasoning": "r", "subtasks": "1. Audit as well as sweep Alice.\n2. Sweep Hubbard."}
    retry_would_differ = {"reasoning": "r", "subtasks": "1. Other part.\n2. Another part."}
    assert split_into_subtasks("the request", planner_lm=DummyLM([clean, retry_would_differ])) == [
        "Audit as well as sweep Alice.", "Sweep Hubbard."]


def test_skills_are_read_as_plain_lines_and_all_are_shown(tmp_path):
    from fastworkflow.workflow_agent import _job_types_for, load_skills

    for name, text in {"leaver-sweep": "One person is leaving: their reach. One job per person.\nfind them\nopen them",
                       "unit-walk": "Walk a whole unit roster. The unit is one job.\nopen the unit"}.items():
        (tmp_path / "_skills" / name).mkdir(parents=True)
        (tmp_path / "_skills" / name / "SKILL.md").write_text(text + "\n\n")

    assert load_skills(str(tmp_path))[0] == (
        "leaver-sweep", "One person is leaving: their reach. One job per person.", ("find them", "open them"))
    assert _job_types_for(str(tmp_path)).splitlines() == [
        "- leaver-sweep: One person is leaving: their reach. One job per person.",
        "- unit-walk: Walk a whole unit roster. The unit is one job.",
    ]
    assert _job_types_for(str(tmp_path / "missing")) == "(none)"


def test_a_workflow_split_instructions_file_replaces_the_default_instructions(tmp_path):
    from fastworkflow.workflow_agent import SplitIntoSubtasks, _split_signature, load_skills

    (tmp_path / "_skills").mkdir()
    (tmp_path / "_skills" / "split_instructions.md").write_text("Split it my way.")
    other = tmp_path / "other"
    (other / "_skills").mkdir(parents=True)

    assert _split_signature(str(tmp_path)).instructions == "Split it my way."
    assert _split_signature(str(other)).instructions == SplitIntoSubtasks.instructions
    assert load_skills(str(tmp_path)) == ()


def _skill_folder(tmp_path):
    (tmp_path / "_skills" / "leaver").mkdir(parents=True)
    (tmp_path / "_skills" / "leaver" / "SKILL.md").write_text("Someone is leaving.\nfind them\nopen them\n")
    return str(tmp_path)


def _fake_matcher(value, probability):
    from types import SimpleNamespace

    def predict(**_inputs):
        choice = SimpleNamespace(value=value, probabilities={value: probability}, confidence=probability)
        return SimpleNamespace(job=choice)

    return predict, DummyLM([])


def test_a_plan_template_is_the_matched_skill_steps_in_a_fixed_format(monkeypatch):
    from fastworkflow import workflow_agent

    monkeypatch.setattr(workflow_agent, "match_skill", lambda text, folder: ("leaver", ("find them", "open them")))
    assert workflow_agent.plan_template_for("Two people are leaving", "unused") == (
        "A job like this is usually done as:\n- find them\n- open them\n"
        "Adapt this to the request: drop steps it does not need and add steps it asks for.")
    monkeypatch.setattr(workflow_agent, "match_skill", lambda text, folder: None)
    assert workflow_agent.plan_template_for("What is the weather in Paris?", "unused") == ""


def test_match_skill_only_returns_a_confident_skill_pick(tmp_path, monkeypatch):
    from fastworkflow import workflow_agent

    folder = _skill_folder(tmp_path)
    match = lambda: workflow_agent.match_skill("Two people are leaving", folder)  # noqa: E731

    monkeypatch.setattr(workflow_agent, "_skill_matcher", lambda f: _fake_matcher("leaver", 0.9))
    assert match() == ("leaver", ("find them", "open them"))
    monkeypatch.setattr(workflow_agent, "_skill_matcher", lambda f: _fake_matcher("none", 0.9))
    assert match() is None
    monkeypatch.setattr(workflow_agent, "_skill_matcher", lambda f: _fake_matcher("leaver", 0.49))
    assert match() is None

    def broken(folder):
        raise RuntimeError("provider down")

    monkeypatch.setattr(workflow_agent, "_skill_matcher", broken)
    assert match() is None


def test_no_jev_key_means_no_matcher_and_no_template(tmp_path, workflow_env, monkeypatch):
    from fastworkflow import workflow_agent

    monkeypatch.delenv("JEV_API_KEY", raising=False)
    workflow_env()
    folder = _skill_folder(tmp_path)
    assert workflow_agent._skill_matcher(folder) is None
    assert workflow_agent.plan_template_for("Two people are leaving", folder) == ""


def test_the_plan_template_reaches_the_planner_but_not_the_returned_query(workflow_env):
    workflow_env()
    ctx, wf = _session(RecordingTraceSink())

    ctx._begin_turn("show my todo items")
    lm = DummyLM([_text_answer()])
    result = _plan(ctx, wf, lm, plan_template="TEMPLATE TEXT")

    assert "TEMPLATE TEXT" in _prompt(lm, 0)
    assert "TEMPLATE TEXT" not in result


def test_a_workflow_file_is_read_from_its_skills_folder_only(tmp_path):
    from fastworkflow.workflow_agent import _skill_file_text

    (tmp_path / "_skills").mkdir()
    (tmp_path / "_skills" / "split_instructions.md").write_text("Split it my way.")

    assert _skill_file_text(str(tmp_path), "split_instructions.md") == "Split it my way."
    assert _skill_file_text(str(tmp_path / "missing"), "split_instructions.md") is None

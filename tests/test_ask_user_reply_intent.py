"""ask_user reply intent: how a reply is classified, and which tools it takes from the agent."""
from __future__ import annotations

import json
from types import SimpleNamespace

import dspy
import pytest
from dspy.utils import DummyLM

from fastworkflow import workflow_agent
from fastworkflow.utils.react import fastWorkflowReAct
from fastworkflow.workflow_agent import _apply_reply_intent, classify_user_reply
from fastworkflow.workflow_execution_context import WorkflowExecutionContext


def ask_user(clarification_request: str) -> str:
    """Ask the user."""
    return "asked"


def lookup(query: str) -> str:
    """Look something up."""
    return "found"


def _agent() -> fastWorkflowReAct:
    return fastWorkflowReAct(dspy.Signature("question -> answer"), tools=[ask_user, lookup])


def _listed_tools(agent) -> set[str]:
    return set(agent.react.signature.output_fields["next_tool_name"].annotation.__args__)


def _fake_intent_predictor(value, probability):
    def predict(**_inputs):
        choice = SimpleNamespace(value=value, probabilities={value: probability}, confidence=probability)
        return SimpleNamespace(intent=choice)

    return predict


def _use_predictor(monkeypatch, predict) -> None:
    monkeypatch.setattr(workflow_agent, "_jev_lm", lambda: DummyLM([]))
    monkeypatch.setattr(workflow_agent.dspy, "Predict", lambda signature: predict)


@pytest.mark.parametrize("value", ["answer", "stop_asking", "abort"])
def test_a_confident_intent_is_returned(monkeypatch, value):
    _use_predictor(monkeypatch, _fake_intent_predictor(value, 0.9))
    assert classify_user_reply("cancel it", "Which one?") == value


def test_an_unconfident_intent_is_an_answer(monkeypatch):
    _use_predictor(monkeypatch, _fake_intent_predictor("abort", 0.49))
    assert classify_user_reply("maybe", "Which one?") == "answer"


def test_a_failed_classification_is_an_answer(monkeypatch):
    def broken(**_inputs):
        raise RuntimeError("provider down")

    _use_predictor(monkeypatch, broken)
    assert classify_user_reply("cancel it", "Which one?") == "answer"


def test_without_a_jev_key_the_reply_is_an_answer_and_nothing_is_called(monkeypatch):
    def must_not_run(**_inputs):
        raise AssertionError("no Jev call without a key")

    monkeypatch.setattr(workflow_agent, "_jev_lm", lambda: None)
    monkeypatch.setattr(workflow_agent.dspy, "Predict", lambda signature: must_not_run)
    assert classify_user_reply("cancel it", "Which one?") == "answer"


def test_stop_asking_takes_ask_user_out_of_the_listed_tools():
    agent = _agent()
    _apply_reply_intent(agent, "stop_asking")
    assert _listed_tools(agent) == {"lookup", "finish"}


def test_abort_leaves_only_finish_and_the_prompt_omits_the_rest():
    agent = _agent()
    _apply_reply_intent(agent, "abort")
    assert _listed_tools(agent) == {"finish"}

    lm = DummyLM([{"next_thought": "done", "next_tool_name": "finish", "next_tool_args": {}}])
    with dspy.context(lm=lm):
        agent.react(question="q", trajectory="")
    prompt = "\n".join(str(m.get("content", "")) for m in lm.history[-1]["messages"])
    assert "ask_user" not in prompt and "lookup" not in prompt and "finish" in prompt


def test_answer_changes_nothing():
    agent = _agent()
    _apply_reply_intent(agent, "answer")
    assert agent.disabled_tools == frozenset()
    assert _listed_tools(agent) == {"ask_user", "lookup", "finish"}


def test_a_disabled_tool_called_anyway_is_refused():
    agent = _agent()
    agent.disabled_tools = {"ask_user"}
    steps = iter([
        SimpleNamespace(next_thought="ask", next_tool_name="ask_user", next_tool_args={"clarification_request": "q"}),
        SimpleNamespace(next_thought="done", next_tool_name="finish", next_tool_args={}),
    ])
    agent.react = lambda trajectory, **_input_args: next(steps)  # type: ignore[method-assign]
    trajectory: dict = {}

    result = agent._run_loop(trajectory, 0, {"question": "q"}, max_iters=5, exception_count=0)

    assert result is None
    assert trajectory["observation_0"] == "ask_user is not available now."


def test_disabled_tools_survive_a_suspension_round_trip():
    agent = _agent()
    agent.disabled_tools = {"ask_user"}
    agent._suspended = {
        "trajectory": {}, "idx": 0, "input_args": {"question": "q"},
        "max_iters": 5, "clarification": "Which one?",
    }
    state = json.loads(json.dumps(agent.export_suspended()))

    restored = _agent()
    restored.import_suspended(state)

    assert restored.disabled_tools == {"ask_user"}
    assert _listed_tools(restored) == {"lookup", "finish"}


def test_each_subtask_starts_with_every_tool_enabled(monkeypatch):
    ctx = WorkflowExecutionContext(run_as_agent=True)
    agent = _agent()
    ctx._workflow_tool_agent = agent
    ctx._subtasks = ["first", "second"]
    ctx._origins = [0, 1]
    seen = []

    def call_agent(query, planner_user_query, *, first_step=None, template_text=None):
        seen.append(set(agent.disabled_tools))
        agent.disabled_tools = {"ask_user"}
        return SimpleNamespace(final_answer="done", suspended=False)

    monkeypatch.setattr(ctx, "_call_agent_for_query", call_agent)
    ctx._run_subtasks_from(0)

    assert seen == [set(), set()]


def test_clearing_the_subtask_state_enables_every_tool_again():
    ctx = WorkflowExecutionContext(run_as_agent=True)
    agent = _agent()
    ctx._workflow_tool_agent = agent
    agent.disabled_tools = {"ask_user"}

    ctx._clear_subtask_state()

    assert agent.disabled_tools == frozenset()
    assert _listed_tools(agent) == {"ask_user", "lookup", "finish"}

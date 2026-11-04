"""ask_user reply intent: how a reply is classified, and which tools it takes from the agent."""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import get_args

import dspy
import pytest
from dspy.utils import DummyLM

from fastworkflow import workflow_agent
from fastworkflow.utils.react import fastWorkflowReAct
from fastworkflow.workflow_agent import (
    REPLY_INTENTS,
    _apply_reply_intent,
    _post_ask_user_response,
    classify_user_reply,
)
from fastworkflow.workflow_execution_context import WorkflowExecutionContext
from tests.test_jev_field_given import jev_env  # noqa: F401  (fixture: real Jev, skipped without a key)


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

    def call_agent(query, planner_user_query, *, first_step=None, template_text=None,
                   evidence_floor=None):
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


def _offered_values(signature) -> tuple[str, ...]:
    """The option values the intent field of the classifier signature offers to Jev."""
    choice_type = signature.output_fields["intent"].annotation
    return get_args(choice_type.model_fields["value"].annotation)


def test_reply_intents_are_the_three_measured_options_plus_wait_with_verbatim_texts():
    assert REPLY_INTENTS == (
        ("answer", "The reply answers the question or gives direction for the work to go on."),
        ("stop_asking", "The reply declines to answer or says not to ask again, but does not ask to stop the work."),
        ("abort", "The reply asks to stop, cancel or abort the work."),
        ("wait", "The reply asks for a moment or says the user is checking; it gives no value and asks nothing."),
    )


def test_a_confident_wait_is_returned(monkeypatch):
    _use_predictor(monkeypatch, _fake_intent_predictor("wait", 0.9))
    assert classify_user_reply("hmm, hold on", "Which repository should I search?") == "wait"


def test_an_unconfident_wait_is_an_answer(monkeypatch):
    _use_predictor(monkeypatch, _fake_intent_predictor("wait", 0.49))
    assert classify_user_reply("hmm", "Which repository should I search?") == "answer"


def _record_offered(monkeypatch, offered: list) -> None:
    """Record the options each classifier signature offers to Jev; Jev answers "answer" with confidence."""
    monkeypatch.setattr(workflow_agent, "_jev_lm", lambda: DummyLM([]))

    def predict_factory(signature):
        offered.append(_offered_values(signature))
        return _fake_intent_predictor("answer", 0.99)

    monkeypatch.setattr(workflow_agent.dspy, "Predict", predict_factory)


@pytest.mark.parametrize("setting", ["off", "OFF", "0", "false", " False "])
def test_with_the_wait_option_off_wait_is_not_offered_and_nothing_returns_wait(monkeypatch, setting):
    monkeypatch.setenv("FW_JEV_REPLY_WAIT", setting)
    offered = []
    _record_offered(monkeypatch, offered)
    assert classify_user_reply("hmm, hold on", "Which repository should I search?") == "answer"
    assert offered == [("answer", "stop_asking", "abort")]


def test_with_the_wait_option_on_jev_is_offered_all_four(monkeypatch):
    monkeypatch.setenv("FW_JEV_REPLY_WAIT", "on")
    offered = []
    _record_offered(monkeypatch, offered)
    classify_user_reply("the billing one", "Which repository should I search?")
    assert offered == [("answer", "stop_asking", "abort", "wait")]


def test_wait_changes_no_disabled_tools():
    agent = _agent()
    _apply_reply_intent(agent, "wait")
    assert agent.disabled_tools == frozenset()
    assert _listed_tools(agent) == {"ask_user", "lookup", "finish"}


class _Session:
    """The chat session surface _post_ask_user_response uses: the action log, the pending ask_user entry, the agent."""

    def __init__(self, agent):
        self.workflow_tool_agent = agent
        self._current_planner_lm = None
        self.actions = []
        self.completed = []

    def append_action_log(self, record):
        self.actions.append(record)

    def complete_ask_user_entry(self, answer):
        self.completed.append(answer)

    def get_active_workflow(self):
        return None


def test_post_ask_user_response_on_wait_asks_again_without_replanning(monkeypatch):
    question = "Which repository should I search?"
    agent = _agent()  # built before dspy.Predict is patched: the react step itself uses it
    session = _Session(agent)
    _use_predictor(monkeypatch, _fake_intent_predictor("wait", 0.9))

    def planner_must_not_run(*_args, **_kwargs):
        raise AssertionError("a wait reply must not replan")

    monkeypatch.setattr(workflow_agent, "build_query_with_next_steps", planner_must_not_run)

    observation = _post_ask_user_response(question, "hmm, hold on, let me check", session)

    assert observation.startswith("The user asked for a moment and has not answered yet.")
    assert observation.endswith(f"ask_user: {question}")
    assert session.completed == ["hmm, hold on, let me check"]
    assert session.actions == [{"agent_query": question, "user_response": "hmm, hold on, let me check"}]
    assert agent.disabled_tools == frozenset()


def test_real_jev_hesitation_is_wait_and_an_answer_is_answer(jev_env):
    question = "Which repository should I search?"
    assert classify_user_reply("hmm, hold on, let me check", question) == "wait"
    assert classify_user_reply("the billing one", question) == "answer"

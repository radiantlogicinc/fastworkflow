"""Unit tests for fastWorkflowReAct suspend/resume (Topology B ask_user)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from fastworkflow.utils.react import AskUserSuspend, fastWorkflowReAct


def _bare_react_agent(**tools):
    """Construct a fastWorkflowReAct without running Module.__init__ (no dspy Tool wiring)."""
    agent = fastWorkflowReAct.__new__(fastWorkflowReAct)
    agent.iteration_counter = 0
    agent.max_iters = 5
    agent.inputs = {}
    agent.current_trajectory = {}
    agent._suspended = None
    agent.tools = tools
    return agent


def test_run_loop_returns_suspended_prediction_without_observation():
    agent = _bare_react_agent(
        ask_user=lambda clarification_request: (_ for _ in ()).throw(
            AskUserSuspend(clarification_request)
        ),
    )
    agent.react = lambda trajectory, **input_args: SimpleNamespace(  # type: ignore[method-assign]
        next_thought="need input",
        next_tool_name="ask_user",
        next_tool_args={"clarification_request": "Which one?"},
    )

    result = agent._run_loop({}, 0, {"query": "hello"}, max_iters=5, exception_count=0)

    assert result is not None
    assert result.suspended is True
    assert result.clarification == "Which one?"
    assert agent._suspended is not None
    assert "observation_0" not in agent._suspended["trajectory"]


def test_resume_continues_after_observation():
    agent = _bare_react_agent(
        finish=lambda: "done",
    )
    trajectory = {"thought_0": "ask", "tool_name_0": "ask_user", "tool_args_0": {}}
    agent._suspended = {
        "trajectory": trajectory,
        "idx": 0,
        "input_args": {"query": "hello"},
        "max_iters": 5,
        "clarification": "Which one?",
    }
    agent.extract = lambda trajectory, **input_args: {"final_answer": "finished"}  # type: ignore[method-assign]

    calls: list[str] = []

    def react_after_resume(trajectory, **input_args):
        calls.append("react")
        return SimpleNamespace(
            next_thought="got answer",
            next_tool_name="finish",
            next_tool_args={},
        )

    agent.react = react_after_resume  # type: ignore[method-assign]

    result = agent.resume("user said B")

    assert calls == ["react"]
    assert result.final_answer == "finished"
    assert agent._suspended is None


def test_clear_suspension_drops_stash():
    agent = _bare_react_agent()
    agent._suspended = {"trajectory": {}, "idx": 0, "input_args": {}, "max_iters": 5}
    agent.clear_suspension()
    assert agent._suspended is None
    with pytest.raises(RuntimeError, match="No suspended"):
        agent.resume("too late")

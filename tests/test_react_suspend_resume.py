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
    agent.trajectory = {}
    agent._dropped_steps = 0
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


def test_run_loop_records_full_step_in_trajectory():
    """A completed tool step lands in the trajectory with thought, tool_name,
    tool_args, and the full observation: the planner and distillation read it."""
    agent = _bare_react_agent(
        do_it=lambda: "did it",
        finish=lambda: "done",
    )

    preds = iter([
        SimpleNamespace(next_thought="act", next_tool_name="do_it", next_tool_args={}),
        SimpleNamespace(next_thought="stop", next_tool_name="finish", next_tool_args={}),
    ])
    agent.react = lambda trajectory, **input_args: next(preds)  # type: ignore[method-assign]
    agent.extract = lambda trajectory, **input_args: {"final_answer": "ok"}  # type: ignore[method-assign]

    trajectory = {}
    result = agent._run_loop(trajectory, 0, {"query": "hello"}, max_iters=5, exception_count=0)

    assert result is None  # completed normally
    assert trajectory["thought_0"] == "act"
    assert trajectory["tool_name_0"] == "do_it"
    assert trajectory["observation_0"] == "did it"
    assert trajectory["tool_args_0"] == {}


def test_trajectory_resets_each_forward_turn():
    """trajectory is per-logical-turn: forward() must reset it at the
    start of each new turn so a later turn does not accumulate the prior turn's
    steps. (resume() must NOT reset — covered separately.)"""
    agent = _bare_react_agent(do_it=lambda: "did it", finish=lambda: "done")
    agent._exhausted_last_run = False
    agent._suspended = None
    agent.max_iters = 5
    # _bare_react_agent skips __init__; provide the submodule attrs that forward()
    # passes to _call_with_potential_trajectory_truncation (our mock ignores them).
    agent.react = object()
    agent.extract = object()

    def make_turn(num_tool_steps: int):
        # `num_tool_steps` tool calls then finish, per forward() call. Turn 1 runs
        # MORE steps than turn 2 so that, if the reset is missing, turn 1's higher-
        # index keys survive into turn 2 (detectable), rather than being overwritten.
        preds = iter(
            [
                SimpleNamespace(next_thought=f"act{i}", next_tool_name="do_it", next_tool_args={})
                for i in range(num_tool_steps)
            ]
            + [SimpleNamespace(next_thought="stop", next_tool_name="finish", next_tool_args={})]
        )

        def call(module, trajectory, **input_args):
            try:
                return next(preds)
            except StopIteration:
                return {"final_answer": "ok"}

        return call

    # Turn 1: 3 tool steps -> populates indices up to thought_3/observation_3.
    agent._call_with_potential_trajectory_truncation = make_turn(3)  # type: ignore[method-assign]
    agent.forward(query="first")
    assert "observation_3" in agent.trajectory  # deep turn

    # Turn 2: 1 tool step -> only indices 0 and 1. If forward() reset the mirror,
    # the leftover observation_3 from turn 1 must be GONE.
    agent._call_with_potential_trajectory_truncation = make_turn(1)  # type: ignore[method-assign]
    agent.forward(query="second")
    second_keys = set(agent.trajectory.keys())

    assert "thought_0" in second_keys
    # The load-bearing assertion: turn 1's deep keys did not survive into turn 2.
    assert "observation_3" not in second_keys
    assert "thought_2" not in second_keys


def test_resume_records_user_answer_in_trajectory():
    """The resumed observation (the user's ask_user answer) lands in the one
    trajectory the planner and distillation read."""
    agent = _bare_react_agent(finish=lambda: "done")
    trajectory = {"thought_0": "ask", "tool_name_0": "ask_user", "tool_args_0": {}}
    agent._suspended = {
        "trajectory": trajectory,
        "idx": 0,
        "input_args": {"query": "hello"},
        "max_iters": 5,
        "clarification": "Which one?",
    }
    agent.extract = lambda trajectory, **input_args: {"final_answer": "finished"}  # type: ignore[method-assign]
    agent.react = lambda trajectory, **input_args: SimpleNamespace(  # type: ignore[method-assign]
        next_thought="got answer", next_tool_name="finish", next_tool_args={}
    )

    agent.resume("user said B")

    assert agent.trajectory is trajectory
    assert trajectory["observation_0"] == "user said B"


def test_truncation_drops_oldest_steps_from_the_model_view_only():
    """The fallback shows the model a copy without the oldest step; the
    canonical trajectory keeps every step."""
    agent = _bare_react_agent()
    trajectory = {f"{name}_{i}": i for i in range(2) for name in ("thought", "tool_name", "tool_args", "observation")}
    before = dict(trajectory)

    agent.truncate_trajectory(trajectory)

    assert trajectory == before
    assert list(agent._model_view(trajectory)) == list(before)[4:]


def test_truncation_drops_whole_steps_when_a_recovered_step_has_fewer_keys():
    """A parse-error recovery writes observation_0 alone, then a thought/observation
    pair under index 1. Dropping one step must remove all of step 0 and nothing
    of step 1 or 2, not the first four keys by count."""
    agent = _bare_react_agent()
    trajectory = {
        "observation_0": "Agent failed to select a valid tool",
        "thought_1": "To execute a command, I should use one of the available tools",
        "observation_1": "Use the appropriate tool with proper arguments",
        "thought_2": "run it",
        "tool_name_2": "ask_user",
        "tool_args_2": {},
        "observation_2": "ok",
    }

    agent.truncate_trajectory(trajectory)

    view = agent._model_view(trajectory)
    assert list(view) == ["thought_1", "observation_1", "thought_2", "tool_name_2", "tool_args_2", "observation_2"]
    assert len(trajectory) == 7

    agent.truncate_trajectory(trajectory)
    assert list(agent._model_view(trajectory)) == ["thought_2", "tool_name_2", "tool_args_2", "observation_2"]

    agent.truncate_trajectory(trajectory)
    assert agent._model_view(trajectory) == {}
    with pytest.raises(ValueError, match="cannot be truncated"):
        agent.truncate_trajectory(trajectory)


def test_clear_suspension_drops_stash():
    from fastworkflow.utils.react import NoSuspendedAgentStateError

    agent = _bare_react_agent()
    agent._suspended = {"trajectory": {}, "idx": 0, "input_args": {}, "max_iters": 5}
    agent.clear_suspension()
    assert agent._suspended is None
    with pytest.raises(NoSuspendedAgentStateError, match="No suspended"):
        agent.resume("too late")

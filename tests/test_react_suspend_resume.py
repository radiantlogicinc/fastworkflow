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


def test_run_loop_mirrors_full_step_into_current_trajectory():
    """A completed tool step must populate current_trajectory with thought,
    tool_name, tool_args, and observation (not just an action summary), because
    the planner and distillation read current_trajectory as the agent trajectory."""
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

    result = agent._run_loop({}, 0, {"query": "hello"}, max_iters=5, exception_count=0)

    assert result is None  # completed normally
    ct = agent.current_trajectory
    assert ct["thought_0"] == "act"
    assert ct["tool_name_0"] == "do_it"
    assert ct["observation_0"] == "did it"
    assert ct["tool_args_0"] == {}


def test_current_trajectory_resets_each_forward_turn():
    """current_trajectory is per-logical-turn: forward() must reset it at the
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
    assert "observation_3" in agent.current_trajectory  # deep turn

    # Turn 2: 1 tool step -> only indices 0 and 1. If forward() reset the mirror,
    # the leftover observation_3 from turn 1 must be GONE.
    agent._call_with_potential_trajectory_truncation = make_turn(1)  # type: ignore[method-assign]
    agent.forward(query="second")
    second_keys = set(agent.current_trajectory.keys())

    assert "thought_0" in second_keys
    # The load-bearing assertion: turn 1's deep keys did not survive into turn 2.
    assert "observation_3" not in second_keys
    assert "thought_2" not in second_keys


def test_resume_mirrors_user_answer_into_current_trajectory():
    """The resumed observation (the user's ask_user answer) must land in
    current_trajectory, not only in the local working trajectory."""
    agent = _bare_react_agent(finish=lambda: "done")
    # Pre-suspend, current_trajectory already holds the pre-ask_user step.
    agent.current_trajectory = {
        "thought_0": "ask",
        "tool_name_0": "ask_user",
        "tool_args_0": {},
    }
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

    # The user's answer is recorded as observation_0 in current_trajectory.
    assert agent.current_trajectory["observation_0"] == "user said B"


def test_clear_suspension_drops_stash():
    from fastworkflow.utils.react import NoSuspendedAgentStateError

    agent = _bare_react_agent()
    agent._suspended = {"trajectory": {}, "idx": 0, "input_args": {}, "max_iters": 5}
    agent.clear_suspension()
    assert agent._suspended is None
    with pytest.raises(NoSuspendedAgentStateError, match="No suspended"):
        agent.resume("too late")


# ---------------------------------------------------------------------------
# ido-dpx / F15: both loops intercept the finish action the same way
# ---------------------------------------------------------------------------

class _Tool:
    """A tool the sync loop calls and the async loop awaits."""

    def __init__(self, text):
        self.text = text

    def __call__(self, **kwargs):
        return self.text

    async def acall(self, **kwargs):
        return self.text


class _FixedChecker:
    """Stands in for the finish check: always has the same thing to say."""

    def __init__(self, note):
        self.text = note
        self.calls = 0

    def note(self, agent, input_args, *, iterations_left):
        self.calls += 1
        return self.text


NOTE = "Before you finish: this turn's record shows no command carrying out step 3."


def _looping_agent(predictions, monkeypatch, note=NOTE):
    """A bare agent whose predictor replays *predictions* and whose finish
    check always has the same thing to say. The note's CONTENT is the check's
    business (tests/test_finish_check.py); what is under test here is what
    each loop does with it."""
    agent = _bare_react_agent(
        finish=_Tool("Completed."), a_tool=_Tool("observed more"))
    agent.finish_checker = _FixedChecker(note)
    agent.finish_reminders_enabled = True
    agent.max_iters = 12
    agent._finish_notes_fired = 0
    agent._exhausted_last_run = False
    agent.continuation_scope = None
    agent.observation_archive = None
    agent.react = object()

    pending = iter(predictions)
    agent._call_with_potential_trajectory_truncation = (  # type: ignore[method-assign]
        lambda module, trajectory, **kwargs: next(pending))

    async def _acall(module, trajectory, **kwargs):
        return next(pending)

    async def _aextract(trajectory, **kwargs):
        return {"final_answer": "ok"}

    agent._async_call_with_potential_trajectory_truncation = _acall  # type: ignore[method-assign]
    agent._async_extract_prediction = _aextract  # type: ignore[method-assign]
    return agent


def _pred(tool_name):
    return SimpleNamespace(
        next_thought="t", next_tool_name=tool_name, next_tool_args={})


SCRIPT = [_pred("finish"), _pred("a_tool"), _pred("finish")]


def test_the_async_loop_fires_the_finish_check_note_and_returns_control(monkeypatch):
    """`aforward` must not recognise finish and break out with no note and no
    `_finish_notes_fired` bookkeeping, which would leave the sync and async
    loops implementing the same rule differently."""
    import asyncio

    agent = _looping_agent(list(SCRIPT), monkeypatch)
    result = asyncio.run(agent.aforward(user_query="who holds it", max_iters=12))

    trajectory = result.trajectory
    assert trajectory["observation_0"] == NOTE
    assert trajectory["tool_name_1"] == "a_tool"
    assert trajectory["tool_name_2"] == "finish"
    assert trajectory["observation_2"] == "Completed."
    assert agent._finish_notes_fired == 1


def test_both_loops_agree_on_the_finish_action(monkeypatch):
    """The point of the shared helper: one rule, two loops, one behaviour."""
    import asyncio

    sync_agent = _looping_agent(list(SCRIPT), monkeypatch)
    sync_trajectory: dict = {}
    sync_agent._run_loop(
        sync_trajectory, 0, {"user_query": "who holds it"}, 12, 0)

    async_agent = _looping_agent(list(SCRIPT), monkeypatch)
    async_trajectory = asyncio.run(
        async_agent.aforward(user_query="who holds it", max_iters=12)).trajectory

    keys = ("observation_0", "tool_name_1", "observation_1", "tool_name_2",
            "observation_2")
    assert ({k: sync_trajectory[k] for k in keys}
            == {k: async_trajectory[k] for k in keys})
    assert sync_agent._finish_notes_fired == async_agent._finish_notes_fired == 1


def test_the_async_loop_notes_at_most_once_a_turn(monkeypatch):
    """The cap is the note's own (`_finish_check_note`); what the loop owes it is
    the per-turn reset, which `aforward` never did."""
    import asyncio

    agent = _looping_agent([_pred("finish")] * 4, monkeypatch)
    agent._finish_notes_fired = 7  # a previous turn's count, left behind
    trajectory = asyncio.run(
        agent.aforward(user_query="who holds it", max_iters=12)).trajectory

    assert trajectory["observation_0"] == NOTE
    assert trajectory["observation_1"] == "Completed."
    assert "tool_name_2" not in trajectory
    assert agent._finish_notes_fired == 1


def test_a_silent_finish_check_ends_the_async_loop_at_finish(monkeypatch):
    """No note is the ordinary case, and it must leave the loop as it was."""
    import asyncio

    agent = _looping_agent(list(SCRIPT), monkeypatch, note="")
    trajectory = asyncio.run(
        agent.aforward(user_query="who holds it", max_iters=12)).trajectory

    assert trajectory["observation_0"] == "Completed."
    assert "tool_name_1" not in trajectory
    assert agent._finish_notes_fired == 0


class _BrokenChecker:
    def note(self, agent, input_args, *, iterations_left):
        raise RuntimeError("decision model unreachable")


def test_a_failing_finish_check_ends_the_turn_as_if_there_were_none(monkeypatch):
    """The check fails open: an exception inside it is a finish with no note."""
    import asyncio

    agent = _looping_agent(list(SCRIPT), monkeypatch)
    agent.finish_checker = _BrokenChecker()
    trajectory = asyncio.run(
        agent.aforward(user_query="who holds it", max_iters=12)).trajectory

    assert trajectory["observation_0"] == "Completed."
    assert "tool_name_1" not in trajectory
    assert agent._finish_notes_fired == 0


def test_no_finish_check_attached_means_no_note(monkeypatch):
    import asyncio

    agent = _looping_agent(list(SCRIPT), monkeypatch)
    agent.finish_checker = None
    trajectory = asyncio.run(
        agent.aforward(user_query="who holds it", max_iters=12)).trajectory

    assert trajectory["observation_0"] == "Completed."
    assert agent._finish_notes_fired == 0

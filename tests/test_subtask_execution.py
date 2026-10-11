"""A split request runs its sub-tasks in order, in one turn, and resumes across an ask_user."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import dspy

import fastworkflow
from fastworkflow.turn import TurnStatus
from fastworkflow.utils.react import fastWorkflowReAct
from fastworkflow.workflow_execution_context import MAX_HANDOFFS
from fastworkflow.workflow_execution_context import WorkflowExecutionContext
from tests.test_turn_result_capture import (  # noqa: F401 - shared fixtures
    _make_agent_ctx,
    _set_agents,
    initialized_fastworkflow,
    todo_workflow_path,
)


def _agent_with_steps(answers, suspend_on=None, exhaust=lambda n: False, observations=("o",)):
    """Mock ReAct agent: each call writes steps from its first_step, then answers.

    ``exhaust(n)`` says whether the n-th call ends by hitting the step cap.
    The first call's steps carry ``observations``; later calls carry one plain step.
    """
    calls = []
    agent = MagicMock()

    def call(**kwargs):
        calls.append(kwargs)
        start = kwargs.get("first_step", 0)
        steps = observations if len(calls) == 1 else ("o",)
        agent.trajectory = {}
        for i, text in enumerate(steps):
            agent.trajectory[f"thought_{start + i}"] = "t"
            agent.trajectory[f"observation_{start + i + 1}"] = text
        if len(calls) == suspend_on:
            return SimpleNamespace(suspended=True, clarification="Which one?")
        return SimpleNamespace(final_answer=answers[len(calls) - 1], suspended=False,
                               exhausted=bool(exhaust(len(calls))))

    agent.side_effect = call
    agent.trajectory = {}
    agent.calls = calls
    return agent


def _fake_progress_check(monkeypatch, text="DONE SUMMARY"):
    """Stub the progress check; returns the list of trigger names it was called with."""
    triggers = []

    def fake(chat_session_obj, trigger, **_kwargs):
        triggers.append(trigger)
        return text

    monkeypatch.setattr("fastworkflow.workflow_agent.run_progress_check", fake)
    return triggers


def _split(monkeypatch, subtasks):
    monkeypatch.setattr(
        "fastworkflow.workflow_agent.split_into_subtasks",
        lambda *args, **kwargs: list(subtasks),
    )


def test_two_subtasks_complete_in_order_with_continuous_steps(
    initialized_fastworkflow, todo_workflow_path, monkeypatch
):
    ctx, _wf = _make_agent_ctx(todo_workflow_path, monkeypatch)
    _split(monkeypatch, ["find X", "find Y"])
    long_x = "X is here " + "x" * 400
    agent = _agent_with_steps([long_x, "Y is there"])
    _set_agents(ctx, agent)

    result = ctx.process_turn("find X and find Y")

    assert result.status == TurnStatus.COMPLETED
    assert result.answer.index(f"### 1. find X\n{long_x}") < result.answer.index(
        "### 2. find Y\nY is there"
    )
    assert len(agent.calls) == 2
    assert agent.calls[0].get("first_step") == 0
    # first call wrote thought_0 and observation_1, so the second starts at 2
    assert agent.calls[1]["first_step"] == 2
    # the second sub-task is sent as its own text only, with no earlier answer
    assert agent.calls[1]["user_query"] == "find Y"
    assert "X is here" not in agent.calls[1]["user_query"]


def test_suspended_subtask_resumes_and_runs_the_remaining_subtasks(
    initialized_fastworkflow, todo_workflow_path, monkeypatch
):
    ctx, _wf = _make_agent_ctx(todo_workflow_path, monkeypatch)
    _split(monkeypatch, ["find X", "find Y", "find Z"])
    agent = _agent_with_steps(["X is here", None, "Z is there"], suspend_on=2)
    agent.resume = MagicMock(
        side_effect=lambda observation: SimpleNamespace(final_answer="Y is there", suspended=False)
    )
    _set_agents(ctx, agent)

    first = ctx.process_turn("find X, Y and Z")

    assert first.status == TurnStatus.AWAITING_USER
    assert ctx.awaiting_user
    assert len(agent.calls) == 2

    second = ctx.process_turn("the blue one")

    assert second.status == TurnStatus.COMPLETED
    assert not ctx.awaiting_user
    agent.resume.assert_called_once()
    answer = second.answer
    assert answer.index("### 1. find X\nX is here") < answer.index("### 2. find Y\nY is there")
    assert answer.index("### 2. find Y") < answer.index("### 3. find Z\nZ is there")
    # the third sub-task ran after the resume, numbering past the resumed steps:
    # the mock's second sub-task ends at observation_3, so the third starts at 4
    assert len(agent.calls) == 3
    assert agent.calls[2]["first_step"] == 4


def test_state_round_trips_the_subtask_fields(
    initialized_fastworkflow, todo_workflow_path, monkeypatch
):
    ctx, _wf = _make_agent_ctx(todo_workflow_path, monkeypatch)
    ctx._subtasks = ["find X", "find Y", "find Y (continued)"]
    ctx._origins = [0, 1, 1]
    ctx._subtask_index = 1
    ctx._subtask_answers = ["X is here"]
    ctx._next_step = 7
    ctx._subtask_exhausted = True

    blob = ctx.serialize_state(channel_id="subtask-roundtrip")
    restored = WorkflowExecutionContext(run_as_agent=True)
    restored.apply_serialized_state(blob)

    assert restored._subtasks == ["find X", "find Y", "find Y (continued)"]
    assert restored._origins == [0, 1, 1]
    assert restored._subtask_index == 1
    assert restored._subtask_answers == ["X is here"]
    assert restored._next_step == 7
    assert restored._subtask_exhausted is True


def test_split_disabled_or_single_item_runs_the_old_single_call(
    initialized_fastworkflow, todo_workflow_path, monkeypatch
):
    ctx, _wf = _make_agent_ctx(todo_workflow_path, monkeypatch)
    _split(monkeypatch, ["find X and find Y"])
    agent = _agent_with_steps(["all of it"])
    _set_agents(ctx, agent)

    result = ctx.process_turn("find X and find Y")

    assert result.answer == "all of it"
    agent.assert_called_once()
    assert agent.calls[0] == {
        "user_query": "find X and find Y",
        "available_commands": "commands",
    }


def test_env_switch_off_never_splits(
    initialized_fastworkflow, todo_workflow_path, monkeypatch
):
    ctx, _wf = _make_agent_ctx(todo_workflow_path, monkeypatch)
    split = MagicMock(return_value=["find X", "find Y"])
    monkeypatch.setattr("fastworkflow.workflow_agent.split_into_subtasks", split)
    # get_env_var reads the loaded env table first, so the switch is set there
    monkeypatch.setitem(fastworkflow._env_vars, "FW_SPLIT_REQUESTS", "0")
    agent = _agent_with_steps(["all of it"])
    _set_agents(ctx, agent)

    result = ctx.process_turn("find X and find Y")

    split.assert_not_called()
    assert result.answer == "all of it"
    agent.assert_called_once()


def test_react_forward_starts_numbering_at_first_step(monkeypatch):
    def a_tool(command: str) -> str:
        """Runs a command."""
        return f"ran {command}"

    agent = fastWorkflowReAct("user_query -> answer", tools=[a_tool], max_iters=10)
    monkeypatch.setattr(
        agent, "_call_with_potential_trajectory_truncation",
        lambda *args, **kwargs: dspy.Prediction(
            next_thought="done", next_tool_name="finish", next_tool_args={}),
    )
    monkeypatch.setattr(
        agent, "_extract_prediction", lambda trajectory, **kwargs: {"answer": "ok"}
    )

    prediction = agent.forward(user_query="hi", available_commands="commands", first_step=5)

    assert {key.rsplit("_", 1)[1] for key in prediction.trajectory} == {"5"}
    assert "thought_5" in prediction.trajectory


def test_exhausted_subtask_hands_off_to_a_continuation_matched_on_its_original_text(
    initialized_fastworkflow, todo_workflow_path, monkeypatch
):
    ctx, _wf = _make_agent_ctx(todo_workflow_path, monkeypatch)
    triggers = _fake_progress_check(monkeypatch)
    templates = []
    monkeypatch.setattr(
        "fastworkflow.workflow_agent.plan_template_for",
        lambda text, folder: templates.append(text) or "",
    )
    _split(monkeypatch, ["find X", "find Y"])
    agent = _agent_with_steps(["x partial", "x rest", "Y is there"], exhaust=lambda n: n == 1)
    _set_agents(ctx, agent)

    result = ctx.process_turn("find X and find Y")

    assert result.status == TurnStatus.COMPLETED
    assert triggers == ["step_limit"]
    assert len(agent.calls) == 3
    continuation = agent.calls[1]["user_query"]
    assert continuation.startswith("find X\n\nAlready done in this sub-task")
    assert "DONE SUMMARY" in continuation
    # the fresh run continues numbering where the exhausted run stopped
    assert agent.calls[1]["first_step"] == 2
    assert templates == ["find X", "find X", "find Y"]
    assert agent.calls[2]["user_query"] == "find Y"


def test_a_subtask_that_keeps_exhausting_gets_max_handoffs_then_the_loop_moves_on(
    initialized_fastworkflow, todo_workflow_path, monkeypatch
):
    ctx, _wf = _make_agent_ctx(todo_workflow_path, monkeypatch)
    triggers = _fake_progress_check(monkeypatch)
    _split(monkeypatch, ["find X", "find Y"])
    agent = _agent_with_steps(["x1", "x2", "x3", "y"], exhaust=lambda n: n <= 3)
    _set_agents(ctx, agent)

    ctx.process_turn("find X and find Y")

    assert triggers == ["step_limit"] * MAX_HANDOFFS
    assert len(agent.calls) == MAX_HANDOFFS + 2
    assert agent.calls[-1]["user_query"] == "find Y"


def test_at_the_cap_the_last_capped_answer_is_kept(
    initialized_fastworkflow, todo_workflow_path, monkeypatch
):
    ctx, _wf = _make_agent_ctx(todo_workflow_path, monkeypatch)
    _fake_progress_check(monkeypatch)
    _split(monkeypatch, ["find X", "find Y"])
    agent = _agent_with_steps(["x1", "x2", "x3", "y"], exhaust=lambda n: n <= 3)
    _set_agents(ctx, agent)

    result = ctx.process_turn("find X and find Y")

    assert result.answer == "### 1. find X\nx3\n\n### 2. find Y\ny"


def test_handoff_continuation_carries_only_what_is_done(
    initialized_fastworkflow, todo_workflow_path, monkeypatch
):
    ctx, _wf = _make_agent_ctx(todo_workflow_path, monkeypatch)
    # The hand-off asks the progress check for the done part only (its whats_done field).
    _fake_progress_check(monkeypatch, "- did A")
    _split(monkeypatch, ["find X", "find Y"])
    agent = _agent_with_steps(["x partial", "x rest", "Y is there"], exhaust=lambda n: n == 1)
    _set_agents(ctx, agent)

    ctx.process_turn("find X and find Y")

    continuation = agent.calls[1]["user_query"]
    assert "Already done in this sub-task" in continuation
    assert "- did A" in continuation
    assert "What remains" not in continuation
    assert "What's done" not in continuation


def test_answers_of_a_subtask_and_its_continuations_share_one_heading(
    initialized_fastworkflow, todo_workflow_path, monkeypatch
):
    ctx, _wf = _make_agent_ctx(todo_workflow_path, monkeypatch)
    _fake_progress_check(monkeypatch)
    _split(monkeypatch, ["find X", "find Y"])
    agent = _agent_with_steps(["x1", "x2", "y"], exhaust=lambda n: n == 1)
    _set_agents(ctx, agent)

    result = ctx.process_turn("find X and find Y")

    # the capped run's answer x1 is replaced by its continuation's answer x2
    assert result.answer == "### 1. find X\nx2\n\n### 2. find Y\ny"


def test_unsplit_request_that_is_exhausted_continues_and_has_no_heading(
    initialized_fastworkflow, todo_workflow_path, monkeypatch
):
    ctx, _wf = _make_agent_ctx(todo_workflow_path, monkeypatch)
    triggers = _fake_progress_check(monkeypatch)
    _split(monkeypatch, ["find X and find Y"])
    agent = _agent_with_steps(["part one", "part two"], exhaust=lambda n: n == 1)
    _set_agents(ctx, agent)

    result = ctx.process_turn("find X and find Y")

    assert triggers == ["step_limit"]
    assert len(agent.calls) == 2
    assert agent.calls[1]["user_query"].startswith("find X and find Y\n\nAlready done")
    assert result.answer == "part two"


def test_suspension_inside_a_continuation_resumes_the_remaining_subtasks(
    initialized_fastworkflow, todo_workflow_path, monkeypatch
):
    ctx, _wf = _make_agent_ctx(todo_workflow_path, monkeypatch)
    _fake_progress_check(monkeypatch)
    _split(monkeypatch, ["find X", "find Y"])
    agent = _agent_with_steps(["x partial", None, "Y is there"],
                              suspend_on=2, exhaust=lambda n: n == 1)
    agent.resume = MagicMock(
        side_effect=lambda observation: SimpleNamespace(final_answer="x done", suspended=False)
    )
    _set_agents(ctx, agent)

    first = ctx.process_turn("find X and find Y")

    assert first.status == TurnStatus.AWAITING_USER
    assert ctx._origins == [0, 0, 1]

    second = ctx.process_turn("the blue one")

    assert second.status == TurnStatus.COMPLETED
    agent.resume.assert_called_once()
    assert second.answer == "### 1. find X\nx done\n\n### 2. find Y\nY is there"
    assert len(agent.calls) == 3
    assert agent.calls[2]["user_query"] == "find Y"


def test_disabled_tools_carry_into_a_continuation_and_reset_for_the_next_subtask(
    initialized_fastworkflow, todo_workflow_path, monkeypatch
):
    ctx, _wf = _make_agent_ctx(todo_workflow_path, monkeypatch)
    _fake_progress_check(monkeypatch)
    _split(monkeypatch, ["find X", "find Y"])
    agent = _agent_with_steps(["x1", "x2", "y"], exhaust=lambda n: n == 1)
    agent.disabled_tools = frozenset()
    inner = agent.side_effect
    disabled_seen = []

    def call(**kwargs):
        disabled_seen.append(frozenset(agent.disabled_tools))
        if len(disabled_seen) == 1:
            agent.disabled_tools = frozenset({"ask_user"})  # a stop_asking reply during run 1
        return inner(**kwargs)

    agent.side_effect = call
    _set_agents(ctx, agent)

    ctx.process_turn("find X and find Y")

    assert disabled_seen == [frozenset(), frozenset({"ask_user"}), frozenset()]


def test_react_step_cap_stops_the_loop_and_marks_the_run_exhausted(monkeypatch):
    def a_tool(command: str) -> str:
        """Runs a command."""
        return f"ran {command}"

    agent = fastWorkflowReAct("user_query -> answer", tools=[a_tool], max_iters=3)
    monkeypatch.setattr(
        agent, "_call_with_potential_trajectory_truncation",
        lambda *args, **kwargs: dspy.Prediction(
            next_thought="again", next_tool_name="a_tool", next_tool_args={"command": "x"}),
    )
    monkeypatch.setattr(
        agent, "_extract_prediction", lambda trajectory, **kwargs: {"answer": "partial"}
    )

    prediction = agent.forward(user_query="hi", available_commands="commands")

    assert prediction.exhausted is True
    assert prediction.answer == "partial"
    assert sorted(k for k in prediction.trajectory if k.startswith("thought_")) == [
        "thought_0", "thought_1", "thought_2"]


def test_handoff_continuation_lists_the_contexts_the_capped_run_opened(
    initialized_fastworkflow, todo_workflow_path, monkeypatch
):
    ctx, _wf = _make_agent_ctx(todo_workflow_path, monkeypatch)
    _fake_progress_check(monkeypatch)
    _split(monkeypatch, ["find X", "find Y"])
    observations = (
        "Observation O1 (execute_workflow_query ran in context 'Directory' and 1 Ann)\nfound",
        "Observation O2 (execute_workflow_query ran in prior context 'Directory' and 1 Ann; "
        "now in context 'Account' and 738e9c85 John Doe)\nopened",
        "Observation O3 (execute_workflow_query ran in prior context 'Account' and "
        "738e9c85 John Doe; now in context 'Account' and 738e9c85 John Doe)\nagain",
        "Observation O4 (execute_workflow_query ran in context 'Identity' and 28c5aeb5)\nbare",
    )
    agent = _agent_with_steps(["x partial", "x rest", "Y is there"], exhaust=lambda n: n == 1,
                              observations=observations)
    _set_agents(ctx, agent)

    ctx.process_turn("find X and find Y")

    continuation = agent.calls[1]["user_query"]
    assert continuation.endswith(
        "\n\nOpened in this sub-task:\n"
        "- Directory 1 (Ann)\n"
        "- Account 738e9c85 (John Doe)\n"
        "- Identity 28c5aeb5")


def test_handoff_continuation_has_no_opened_block_when_no_context_was_entered(
    initialized_fastworkflow, todo_workflow_path, monkeypatch
):
    ctx, _wf = _make_agent_ctx(todo_workflow_path, monkeypatch)
    _fake_progress_check(monkeypatch)
    _split(monkeypatch, ["find X", "find Y"])
    agent = _agent_with_steps(["x partial", "x rest", "Y is there"], exhaust=lambda n: n == 1,
                              observations=("Observation O1 (execute_workflow_query)\nplain",))
    _set_agents(ctx, agent)

    ctx.process_turn("find X and find Y")

    assert "Opened in this sub-task" not in agent.calls[1]["user_query"]


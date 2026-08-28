"""Phase 7 §2.7 / ruling I8: distillation reads the in-process action log.

The cwd `action.jsonl` mirror is retired. Distillation now compares teacher and
student passes off `WorkflowExecutionContext.action_log`, which is a single live
list cleared between passes. That makes two things load-bearing, and both are
covered here:

* **Snapshot discipline** — a pass's actions must be copied out (`list(...)`).
  An aliased reference would be emptied by the next pass's clear, silently
  making every student trajectory equal to the teacher's (no divergence ever
  detected, no insights ever extracted).
* **Clear-point discipline** — the log must be cleared at distillation entry
  and between passes, or the previous turn's actions leak into the teacher pass.

These drive a real WorkflowExecutionContext bound to a real workflow, and append
through the production `_append_action_record` path.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

import fastworkflow
from fastworkflow.distillation import DistillationSession, distill_message
from fastworkflow.workflow_agent import _append_action_record
from fastworkflow.workflow_execution_context import WorkflowExecutionContext


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


@pytest.fixture
def distillation_session(initialized_fastworkflow, todo_workflow_path):
    """A DistillationSession over a real WEC bound to the todo workflow."""
    ctx = WorkflowExecutionContext(run_as_agent=True)
    workflow = fastworkflow.Workflow.create(
        todo_workflow_path,
        workflow_id_str=f"distill-actionlog-{uuid.uuid4().hex}",
    )
    ctx.bind_app_workflow(workflow)
    # Distillation reads the workflow off the active-workflow stack, which the
    # run loop normally pushes.
    ctx.push_active_workflow(workflow)
    yield DistillationSession(ctx), ctx
    ctx.clear_workflow_stack()
    ctx.close()


def _action(command_name: str, **params) -> dict:
    return {
        "command": command_name,
        "command_name": command_name,
        "parameters": params,
        "response": f"ran {command_name}",
    }


def test_pass_snapshot_survives_the_next_pass_clear(distillation_session):
    """`list(ctx.action_log)` must detach from the live list (ruling I8)."""
    ds, ctx = distillation_session
    initial = ds.snapshot_workflow_state()

    # Teacher pass: two actions, snapshotted the way _run_agent_pass does.
    _append_action_record(ctx, _action("list_tasks"))
    _append_action_record(ctx, _action("complete_task", id="1"))
    teacher_actions = list(ctx.action_log)

    # Between passes distillation restores state, which clears the live log.
    ds.restore_workflow_state(initial)

    assert ctx.action_log == []
    # The snapshot is a copy: an alias would be empty here, and the student pass
    # would then refill it, making teacher == student for every comparison.
    assert len(teacher_actions) == 2
    assert [a["command_name"] for a in teacher_actions] == [
        "list_tasks",
        "complete_task",
    ]


def test_per_pass_action_counts_stay_disjoint(distillation_session):
    """Teacher and student snapshots hold only their own pass's actions."""
    ds, ctx = distillation_session
    initial = ds.snapshot_workflow_state()

    _append_action_record(ctx, _action("list_tasks"))
    _append_action_record(ctx, _action("complete_task", id="1"))
    teacher_actions = list(ctx.action_log)

    ds.restore_workflow_state(initial)

    _append_action_record(ctx, _action("list_tasks"))
    student_actions = list(ctx.action_log)

    assert len(teacher_actions) == 2
    assert len(student_actions) == 1
    # Divergence is detectable precisely because the passes did not merge.
    diverged, _summary = ds.compare_trajectories(teacher_actions, student_actions)
    assert diverged


def test_distill_message_sheds_prior_turn_actions_at_entry(
    distillation_session, monkeypatch
):
    """Entry clear: turn N-1's actions must not be attributed to the teacher pass.

    The distillation branch bypasses `_run_agent` (the only other clear point),
    so `distill_message` clears on the way in. The agent passes are scripted here
    to keep the test LLM-free; each records how many actions were already in the
    live log when it started, which is what the clear points govern.
    """
    ds, ctx = distillation_session
    entry_lengths: list[int] = []

    def scripted_pass(self, message, **kwargs):
        # Deliberately does NOT clear: the observed length reflects only what the
        # caller (distill_message / restore_workflow_state) left behind.
        entry_lengths.append(len(self.chat_session.action_log))
        _append_action_record(self.chat_session, _action("list_tasks"))
        response = fastworkflow.CommandResponse(response="done")
        return (
            fastworkflow.CommandOutput(command_response=response),
            {},
            list(self.chat_session.action_log),
            [],
        )

    monkeypatch.setattr(DistillationSession, "_run_agent_pass", scripted_pass)

    # Residue from the previous turn, as the live log would hold it.
    _append_action_record(ctx, _action("stale_previous_turn_action"))
    assert len(ctx.action_log) == 1

    result = distill_message(ctx, "list my tasks")

    assert result.command_output.command_response.response == "done"
    # Both passes started clean: entry clear before the teacher, restore-driven
    # clear before the student.
    assert entry_lengths == [0, 0]


class _ScriptedAgent:
    """Stands in for the DSPy ReAct agent, appending the actions its pass ran."""

    def __init__(self, chat_session, command_names: list[str]):
        self._chat_session = chat_session
        self._command_names = command_names
        self.current_trajectory: dict = {}

    def __call__(self, **_kwargs):
        for name in self._command_names:
            _append_action_record(self._chat_session, _action(name))
        self.current_trajectory = {"thought_0": "scripted"}
        return type("AgentResult", (), {"final_answer": "done"})()


def test_run_agent_pass_returns_a_detached_action_snapshot(
    distillation_session, monkeypatch
):
    """Covers the production snapshot in `_run_agent_pass` (ruling I8).

    Only the LLM boundaries are scripted — the clear at pass start and the
    `list(...)` snapshot at pass end are the real lines under test. Returning the
    live list instead of a copy would leave the teacher holding the student's
    actions, so the two passes below would compare equal.
    """
    ds, ctx = distillation_session
    pass_commands = [["list_tasks", "complete_task"], ["list_tasks"]]

    def scripted_agent(chat_session, **_kwargs):
        return _ScriptedAgent(chat_session, pass_commands.pop(0))

    monkeypatch.setattr(
        "fastworkflow.workflow_agent.initialize_workflow_tool_agent", scripted_agent
    )
    monkeypatch.setattr(
        "fastworkflow.workflow_agent.build_query_with_next_steps",
        lambda user_query, session, **kwargs: user_query,
    )
    monkeypatch.setattr(
        "fastworkflow.workflow_agent._what_can_i_do", lambda session: "commands"
    )
    monkeypatch.setattr(
        "fastworkflow.utils.dspy_utils.get_lm", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        ctx, "_call_agent_with_retry", lambda agent_call, lm=None: agent_call()
    )
    monkeypatch.setattr(
        ctx, "summarize_and_record_turn", lambda *args, **kwargs: ("summary", None)
    )

    initial = ds.snapshot_workflow_state()
    _, _, teacher_actions, _ = ds._run_agent_pass(
        "list my tasks",
        agent_lm_role="LLM_TEACHER_AGENT",
        agent_api_key_role="LITELLM_API_KEY_TEACHER_AGENT",
        planner_lm_role="LLM_TEACHER_PLANNER",
        planner_api_key_role="LITELLM_API_KEY_TEACHER_PLANNER",
    )
    ds.restore_workflow_state(initial)
    _, _, student_actions, _ = ds._run_agent_pass(
        "list my tasks",
        agent_lm_role="LLM_STUDENT_AGENT",
        agent_api_key_role="LITELLM_API_KEY_STUDENT_AGENT",
        planner_lm_role="LLM_STUDENT_PLANNER",
        planner_api_key_role="LITELLM_API_KEY_STUDENT_PLANNER",
    )

    assert [a["command_name"] for a in teacher_actions] == [
        "list_tasks",
        "complete_task",
    ]
    assert [a["command_name"] for a in student_actions] == ["list_tasks"]
    diverged, _summary = ds.compare_trajectories(teacher_actions, student_actions)
    assert diverged


def test_action_records_never_write_action_jsonl(
    distillation_session, tmp_path, monkeypatch
):
    """The cwd action.jsonl mirror is gone, including the no-log-available path."""
    _ds, ctx = distillation_session
    monkeypatch.chdir(tmp_path)

    _append_action_record(ctx, _action("list_tasks"))
    # An object exposing neither the WEC nor a core used to fall back to a file
    # append; it now no-ops.
    _append_action_record(object(), _action("list_tasks"))

    assert len(ctx.action_log) == 1
    assert not (tmp_path / "action.jsonl").exists()
    strays = list(tmp_path.iterdir())
    assert not strays, f"appending an action record wrote to the cwd: {strays}"

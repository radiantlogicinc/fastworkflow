"""Cross-process simulation for Topology-B trajectory serialization."""

from __future__ import annotations

import tempfile
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import fastworkflow
from fastworkflow.session_state_store import (
    SAVED_AT_KEY,
    SCHEMA_VERSION,
    DiskSessionStateStore,
    IncompatibleSessionState,
)
from fastworkflow.turn_plan import PlanStep, PlanSubject, TurnPlan
from fastworkflow.workflow_execution_context import WorkflowExecutionContext


@pytest.fixture
def todo_workflow_path() -> str:
    return str(Path(__file__).parent.joinpath("todo_list_workflow").resolve())


@pytest.fixture
def initialized_fastworkflow(tmp_path, monkeypatch):
    fastworkflow.init({"FASTWORKFLOW_STATE_ROOT": str(tmp_path / "workflow_contexts")})
    from fastworkflow.command_routing import RoutingRegistry

    monkeypatch.setattr(
        WorkflowExecutionContext,
        "_agent_dspy_context",
        lambda self: (SimpleNamespace(model="test-model"), None),
    )
    RoutingRegistry.clear_registry()
    yield tmp_path
    RoutingRegistry.clear_registry()


def _suspended_react_blob() -> dict:
    """The shape WorkflowToolAgent.export_suspended() actually returns.

    Has to be a real JSON-native blob: serialize_state is strict now, so a stub
    that returned an opaque object would be rejected rather than quietly
    stringified — which is the whole point, and is what this test used to do
    without noticing.
    """
    return {
        "trajectory": {"thought_0": "find the task", "observation_0": "two matches"},
        "idx": 1,
        "input_args": {"user_query": "list tasks"},
        "max_iters": 25,
        "clarification": "Which task?",
        "iteration_counter": 1,
    }


def _wire_mock_agent(ctx, suspended, completed):
    mock_agent = MagicMock()
    mock_agent.return_value = suspended
    mock_agent.resume.return_value = completed
    mock_agent.export_suspended.return_value = _suspended_react_blob()
    ctx._workflow_tool_agent = mock_agent
    ctx._intent_clarification_agent = MagicMock()


def test_serialize_restore_resume_across_contexts(
    initialized_fastworkflow,
    todo_workflow_path,
    monkeypatch,
):
    channel_id = f"ch-{uuid.uuid4().hex}"
    store_dir = initialized_fastworkflow / "session_state"
    store = DiskSessionStateStore(str(store_dir))

    ctx_a = WorkflowExecutionContext(run_as_agent=True, session_key=channel_id)
    wf = fastworkflow.Workflow.create(
        todo_workflow_path,
        workflow_id_str=channel_id,
    )
    ctx_a.bind_app_workflow(wf)

    monkeypatch.setattr(ctx_a, "_ensure_agent_initialized", lambda: None)
    monkeypatch.setattr(
        "fastworkflow.workflow_agent.build_query_with_next_steps",
        lambda user_query, session, with_agent_inputs_and_trajectory=False,
        planning_insights=None, planner_lm=None, **kwargs: user_query,
    )
    monkeypatch.setattr(
        "fastworkflow.workflow_agent._what_can_i_do",
        lambda session: "commands",
    )
    monkeypatch.setattr(
        ctx_a,
        "_extract_conversation_summary",
        lambda user_query, actions, final: ("summary", "{}"),
    )

    _wire_mock_agent(
        ctx_a,
        SimpleNamespace(suspended=True, clarification="Which task?"),
        SimpleNamespace(final_answer="Done"),
    )

    first = ctx_a._execute_message("list tasks")
    assert ctx_a.awaiting_user
    assert first.command_response.artifacts.get("awaiting_user")

    blob = ctx_a.serialize_state(channel_id=channel_id)
    store.save(channel_id, blob)
    ctx_a.close()

    ctx_b = WorkflowExecutionContext(run_as_agent=True, session_key=channel_id)
    wf_b = fastworkflow.Workflow.create(
        todo_workflow_path,
        workflow_id_str=channel_id,
    )
    ctx_b.bind_app_workflow(wf_b)
    monkeypatch.setattr(ctx_b, "_ensure_agent_initialized", lambda: None)
    monkeypatch.setattr(
        ctx_b,
        "_extract_conversation_summary",
        lambda user_query, actions, final: ("summary", "{}"),
    )

    loaded = store.load(channel_id)
    assert loaded is not None
    ctx_b.apply_serialized_state(loaded)

    _wire_mock_agent(
        ctx_b,
        SimpleNamespace(suspended=True, clarification="Which task?"),
        SimpleNamespace(final_answer="Done"),
    )
    if loaded.get("react") and ctx_b._workflow_tool_agent is not None:
        ctx_b._workflow_tool_agent.import_suspended(loaded["react"])

    assert ctx_b.awaiting_user
    second = ctx_b._execute_message("the urgent one")
    assert not ctx_b.awaiting_user
    assert "Done" in second.command_response.response
    store.clear(channel_id)
    ctx_b.close()


def test_disk_session_state_store_roundtrip(tmp_path):
    store = DiskSessionStateStore(str(tmp_path / "state"))
    state = {"schema_version": 1, "awaiting_user": True, "react": {"idx": 0}}
    store.save("user-1", state)
    assert store.exists("user-1")

    loaded = store.load("user-1")
    # Every saved field comes back unchanged. Not compared with == because the
    # store also stamps a save time for retention (fix-6b4); that is storage
    # metadata rather than session state, which is why it is not in the schema.
    assert {k: loaded[k] for k in state} == state
    assert loaded[SAVED_AT_KEY] <= time.time()

    store.clear("user-1")
    assert not store.exists("user-1")


@pytest.mark.parametrize(
    "found",
    [SCHEMA_VERSION + 1, SCHEMA_VERSION - 1, 0, None, "1"],
    ids=["newer", "older", "missing-as-zero", "null", "string"],
)
def test_unreadable_schema_version_applies_nothing(
    initialized_fastworkflow, todo_workflow_path, found
):
    """A blob this build cannot read must not be partly applied.

    Before this, the mismatch branch logged a warning and then fell through to
    apply every field anyway, so a blob written by a future build would be
    half-restored onto a live session.
    """
    channel_id = f"schema_{uuid.uuid4().hex[:8]}"
    ctx = WorkflowExecutionContext(run_as_agent=True, session_key=channel_id)
    workflow = fastworkflow.Workflow.create(
        todo_workflow_path, workflow_id_str=channel_id
    )
    ctx.bind_app_workflow(workflow)

    # Every field here is one apply_serialized_state would have written.
    hostile = {
        "schema_version": found,
        "awaiting_user": True,
        "suspended_user_message": "restored from the future",
        "pending_clarification_request": "which one?",
        "action_log": [{"command_name": "should_not_appear"}],
    }

    with pytest.raises(IncompatibleSessionState) as excinfo:
        ctx.apply_serialized_state(hostile)

    assert excinfo.value.found == found
    assert excinfo.value.expected == SCHEMA_VERSION

    assert not ctx.awaiting_user
    assert ctx._suspended_user_message is None
    assert ctx._pending_clarification_request is None
    assert ctx._action_log == []
    ctx.close()


def test_the_turn_plan_is_an_additive_json_native_key(
    initialized_fastworkflow, todo_workflow_path
):
    """No schema bump: a blob with the plan is still this version, and saves as it is."""
    channel_id = f"plan_{uuid.uuid4().hex[:8]}"
    ctx = WorkflowExecutionContext(run_as_agent=True, session_key=channel_id)
    ctx.bind_app_workflow(
        fastworkflow.Workflow.create(todo_workflow_path, workflow_id_str=channel_id)
    )
    blob = ctx.serialize_state(channel_id=channel_id)
    assert blob["schema_version"] == SCHEMA_VERSION
    assert (blob["turn_plan"], blob["turn_plan_status"]) == (None, "not_planned")

    ctx._turn_plan = TurnPlan(
        steps=[PlanStep(text="Show all todo items", commands=["show_all_todos"])],
        subjects=[PlanSubject(name="groceries", kind="list")],
    )
    ctx._turn_plan_status = "planned"
    store = DiskSessionStateStore(str(initialized_fastworkflow / "state"))
    store.save(channel_id, ctx.serialize_state(channel_id=channel_id))
    loaded = store.load(channel_id)
    assert TurnPlan.model_validate(loaded["turn_plan"]) == ctx._turn_plan
    assert loaded["turn_plan_status"] == "planned"
    ctx.close()


def test_a_plan_that_does_not_validate_is_lost_on_resume_not_a_failed_restore(
    initialized_fastworkflow, todo_workflow_path, jev_stub, monkeypatch
):
    from fastworkflow.observation_offloading import finish_check

    monkeypatch.setenv(finish_check.CHECK_ENV, "jev")
    monkeypatch.setenv(finish_check.KEY_ENV, "stub-key")
    finish_check._CHECKERS.clear()
    channel_id = f"plan_bad_{uuid.uuid4().hex[:8]}"
    ctx = WorkflowExecutionContext(run_as_agent=True, session_key=channel_id)
    ctx.bind_app_workflow(
        fastworkflow.Workflow.create(todo_workflow_path, workflow_id_str=channel_id)
    )
    ctx.apply_serialized_state({
        "schema_version": SCHEMA_VERSION,
        "awaiting_user": True,
        "react": _suspended_react_blob(),
        "turn_plan": {"steps": "not a list of steps"},
        "turn_plan_status": "planned",
    })
    assert ctx.awaiting_user
    agent = ctx.workflow_tool_agent
    assert agent.finish_checker is not None
    assert ctx._turn_plan is None and ctx._turn_plan_status == "lost_on_resume"
    # The mirror the finish check reads is seeded from the suspension.
    assert agent.current_trajectory == _suspended_react_blob()["trajectory"]
    finish_check._CHECKERS.clear()
    ctx.close()


def test_readable_schema_version_still_applies(
    initialized_fastworkflow, todo_workflow_path
):
    """The guard rejects on version, not on every blob it is handed.

    Without this, deleting the whole restore body would still pass the test
    above, so it is what makes that one about the version rather than about
    apply_serialized_state doing nothing at all.
    """
    channel_id = f"schema_ok_{uuid.uuid4().hex[:8]}"
    ctx = WorkflowExecutionContext(run_as_agent=True, session_key=channel_id)
    workflow = fastworkflow.Workflow.create(
        todo_workflow_path, workflow_id_str=channel_id
    )
    ctx.bind_app_workflow(workflow)

    ctx.apply_serialized_state(
        {
            "schema_version": SCHEMA_VERSION,
            "awaiting_user": True,
            "suspended_user_message": "the urgent one",
            "action_log": [{"command_name": "list_tasks"}],
        }
    )

    assert ctx.awaiting_user
    assert ctx._suspended_user_message == "the urgent one"
    assert len(ctx._action_log) == 1
    ctx.close()

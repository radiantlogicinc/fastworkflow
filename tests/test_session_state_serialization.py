"""Cross-process simulation for Topology-B trajectory serialization."""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import fastworkflow
from fastworkflow.session_state_store import DiskSessionStateStore
from fastworkflow.workflow_execution_context import WorkflowExecutionContext


@pytest.fixture
def todo_workflow_path() -> str:
    return str(Path(__file__).parent.joinpath("todo_list_workflow").resolve())


@pytest.fixture
def initialized_fastworkflow(tmp_path):
    fastworkflow.init({"SPEEDDICT_FOLDERNAME": str(tmp_path / "speedict")})
    from fastworkflow.command_routing import RoutingRegistry

    RoutingRegistry.clear_registry()
    yield tmp_path
    RoutingRegistry.clear_registry()


def _wire_mock_agent(ctx, suspended, completed):
    mock_agent = MagicMock()
    mock_agent.return_value = suspended
    mock_agent.resume.return_value = completed
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
        lambda user_query, session, with_agent_inputs_and_trajectory=False: user_query,
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

    first = ctx_a.process_message("list tasks")
    assert ctx_a.awaiting_user
    assert first.command_responses[0].artifacts.get("awaiting_user")

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
    second = ctx_b.process_message("the urgent one")
    assert not ctx_b.awaiting_user
    assert "Done" in second.command_responses[0].response
    store.clear(channel_id)
    ctx_b.close()


def test_disk_session_state_store_roundtrip(tmp_path):
    store = DiskSessionStateStore(str(tmp_path / "state"))
    state = {"schema_version": 1, "awaiting_user": True, "react": {"idx": 0}}
    store.save("user-1", state)
    assert store.exists("user-1")
    assert store.load("user-1") == state
    store.clear("user-1")
    assert not store.exists("user-1")

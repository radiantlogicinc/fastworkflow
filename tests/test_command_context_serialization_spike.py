"""Spike: current_command_context_name is captured in serialize_state."""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

import fastworkflow
from fastworkflow.workflow_execution_context import WorkflowExecutionContext


@pytest.fixture
def todo_workflow_path() -> str:
    return str(Path(__file__).parent.joinpath("todo_list_workflow").resolve())


@pytest.fixture
def initialized_fastworkflow(tmp_path):
    fastworkflow.init({"SPEEDDICT_FOLDERNAME": str(tmp_path / "speedict")})
    from fastworkflow.command_routing import RoutingRegistry

    RoutingRegistry.clear_registry()
    yield
    RoutingRegistry.clear_registry()


def test_serialize_includes_command_context_name(
    initialized_fastworkflow,
    todo_workflow_path,
):
    channel_id = f"ctx-{uuid.uuid4().hex}"
    ctx = WorkflowExecutionContext(run_as_agent=True, session_key=channel_id)
    wf = fastworkflow.Workflow.create(
        todo_workflow_path,
        workflow_id_str=channel_id,
    )
    ctx.bind_app_workflow(wf)

    class TodoListManager:
        pass

    wf.root_command_context = TodoListManager()
    wf.current_command_context = wf.root_command_context

    blob = ctx.serialize_state(channel_id=channel_id)
    assert blob.get("current_command_context_name") == "TodoListManager"
    ctx.close()

"""Concurrency isolation for WorkflowExecutionContext (NLU + execution paths)."""

from __future__ import annotations

import threading
import uuid
from pathlib import Path

import pytest

import fastworkflow
from fastworkflow.command_executor import CommandExecutor
from fastworkflow.workflow_execution_context import WorkflowExecutionContext


@pytest.fixture
def todo_workflow_path() -> str:
    return str(
        Path(__file__).parent.joinpath("todo_list_workflow").resolve()
    )


@pytest.fixture
def initialized_fastworkflow():
    fastworkflow.init({"SPEEDDICT_FOLDERNAME": "___workflow_contexts"})
    from fastworkflow.command_routing import RoutingRegistry

    RoutingRegistry.clear_registry()
    yield
    RoutingRegistry.clear_registry()


def test_two_contexts_do_not_cross_contaminate(
    initialized_fastworkflow,
    todo_workflow_path,
    monkeypatch,
):
    """Parallel process_message calls see only their own app workflow."""
    markers = ("session-A", "session-B")
    contexts = []
    workflows = []

    for marker in markers:
        wf = fastworkflow.Workflow.create(
            todo_workflow_path,
            workflow_id_str=f"concurrency-{marker}-{uuid.uuid4().hex}",
            workflow_context={"session_marker": marker},
        )
        ctx = WorkflowExecutionContext(run_as_agent=False)
        ctx.bind_app_workflow(wf)
        contexts.append(ctx)
        workflows.append(wf)

    def fake_invoke(cls, session, command: str):
        nlu_marker = session.cme_workflow.context["app_workflow"].context[
            "session_marker"
        ]
        exec_marker = session.get_active_workflow().context["session_marker"]
        return fastworkflow.CommandOutput(
            command_responses=[
                fastworkflow.CommandResponse(
                    response="ok",
                    artifacts={
                        "nlu_marker": nlu_marker,
                        "exec_marker": exec_marker,
                    },
                )
            ]
        )

    monkeypatch.setattr(
        CommandExecutor,
        "invoke_command",
        classmethod(fake_invoke),
    )

    captured: list[tuple[str, str] | None] = [None, None]
    barrier = threading.Barrier(2)

    def run_probe(index: int) -> None:
        barrier.wait()
        ctx = contexts[index]
        output = ctx.process_message("probe")
        artifacts = output.command_responses[0].artifacts
        captured[index] = (artifacts["nlu_marker"], artifacts["exec_marker"])

    threads = [
        threading.Thread(target=run_probe, args=(i,)) for i in range(2)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
        assert not t.is_alive()

    for i, expected in enumerate(markers):
        assert captured[i] == (expected, expected)

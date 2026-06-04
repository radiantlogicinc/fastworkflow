"""Agent-mode and ask_user timeout behavior on WorkflowExecutionContext."""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path
from queue import Queue
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import fastworkflow
from fastworkflow.workflow_execution_context import (
    CommandCancelledError,
    WorkflowExecutionContext,
)


def _set_agents(ctx, agent, clarification_agent=None):
    """
    Set the main workflow tool agent, always setting a clarification agent too.

    Mirrors the production invariant in
    WorkflowExecutionContext._initialize_agent_functionality, where the main agent
    and the intent clarification agent are initialized together. A MagicMock is a
    fine non-None default for tests that never exercise the clarification path.
    """
    ctx._workflow_tool_agent = agent
    ctx._intent_clarification_agent = (
        clarification_agent if clarification_agent is not None else MagicMock()
    )


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


def test_process_message_agent_mode_mocked_agent(
    initialized_fastworkflow,
    todo_workflow_path,
    monkeypatch,
):
    ctx = WorkflowExecutionContext(run_as_agent=True)
    wf = fastworkflow.Workflow.create(
        todo_workflow_path,
        workflow_id_str=f"agent-ctx-{uuid.uuid4().hex}",
    )
    ctx.bind_app_workflow(wf)

    mock_agent = MagicMock()
    mock_result = MagicMock()
    mock_result.final_answer = "Agent finished successfully"
    mock_agent.return_value = mock_result

    monkeypatch.setattr(
        "fastworkflow.workflow_agent.build_query_with_next_steps",
        lambda user_query, session: user_query,
    )
    monkeypatch.setattr(
        "fastworkflow.workflow_agent._what_can_i_do",
        lambda session: "commands",
    )
    monkeypatch.setattr(ctx, "_ensure_agent_initialized", lambda: None)
    monkeypatch.setattr(
        ctx,
        "_extract_conversation_summary",
        lambda user_query, actions, final: ("summary", "{}"),
    )
    _set_agents(ctx, mock_agent)

    output = ctx.process_message("list my tasks")

    assert output.success
    assert "Agent finished successfully" in output.command_responses[0].response
    mock_agent.assert_called_once()


def test_topology_b_ask_user_is_non_blocking_and_suspends_indefinitely(
    initialized_fastworkflow,
    todo_workflow_path,
    monkeypatch,
):
    """No queue: ask_user returns immediately as awaiting_user and never times out."""
    ctx, _wf = _make_agent_ctx(initialized_fastworkflow, todo_workflow_path, monkeypatch)

    mock_agent = MagicMock()
    mock_agent.return_value = SimpleNamespace(
        suspended=True, clarification="Which one?"
    )
    _set_agents(ctx, mock_agent)

    out = ctx.process_message("start")  # returns without blocking
    assert ctx.awaiting_user
    assert out.command_responses[0].artifacts.get("awaiting_user") is True

    # A suspended Topology B turn never expires on its own; it waits in memory.
    time.sleep(0.05)
    assert ctx.awaiting_user

    # The embedder abandons it explicitly via cancel_pending().
    assert ctx.cancel_pending() is True
    assert not ctx.awaiting_user


def test_process_message_converts_ask_user_cancel_to_output(
    initialized_fastworkflow,
    todo_workflow_path,
    monkeypatch,
):
    ctx = WorkflowExecutionContext(run_as_agent=True)
    wf = fastworkflow.Workflow.create(
        todo_workflow_path,
        workflow_id_str=f"cancel-turn-{uuid.uuid4().hex}",
    )
    ctx.bind_app_workflow(wf)

    def failing_agent(**kwargs):
        raise CommandCancelledError("ask_user requires a user_message_queue")

    monkeypatch.setattr(
        "fastworkflow.workflow_agent.build_query_with_next_steps",
        lambda user_query, session: user_query,
    )
    monkeypatch.setattr(
        "fastworkflow.workflow_agent._what_can_i_do",
        lambda session: "commands",
    )
    monkeypatch.setattr(ctx, "_ensure_agent_initialized", lambda: None)
    _set_agents(ctx, failing_agent)

    output = ctx.process_message("do something")

    assert not output.success
    assert "cancelled" in output.command_responses[0].response.lower()


def _make_agent_ctx(initialized_fastworkflow, todo_workflow_path, monkeypatch):
    ctx = WorkflowExecutionContext(run_as_agent=True)
    wf = fastworkflow.Workflow.create(
        todo_workflow_path,
        workflow_id_str=f"agent-ctx-{uuid.uuid4().hex}",
    )
    ctx.bind_app_workflow(wf)

    monkeypatch.setattr(
        "fastworkflow.workflow_agent.build_query_with_next_steps",
        lambda user_query, session, with_agent_inputs_and_trajectory=False: user_query,
    )
    monkeypatch.setattr(
        "fastworkflow.workflow_agent._what_can_i_do",
        lambda session: "commands",
    )
    monkeypatch.setattr(ctx, "_ensure_agent_initialized", lambda: None)
    monkeypatch.setattr(
        ctx,
        "_extract_conversation_summary",
        lambda user_query, actions, final: ("summary", "{}"),
    )
    return ctx, wf


def test_topology_b_ask_user_suspend_resume_round_trip(
    initialized_fastworkflow,
    todo_workflow_path,
    monkeypatch,
):
    ctx, _wf = _make_agent_ctx(initialized_fastworkflow, todo_workflow_path, monkeypatch)

    suspended = SimpleNamespace(suspended=True, clarification="Which task?")
    completed = SimpleNamespace(final_answer="All done")

    mock_agent = MagicMock()
    mock_agent.return_value = suspended
    mock_agent.resume.return_value = completed
    _set_agents(ctx, mock_agent)

    first = ctx.process_message("list my tasks")

    assert ctx.awaiting_user
    assert first.command_responses[0].artifacts.get("awaiting_user") is True
    assert first.command_responses[0].response == "Which task?"
    assert len(ctx.conversation_history.messages) == 0

    second = ctx.process_message("the urgent one")

    assert not ctx.awaiting_user
    assert "All done" in second.command_responses[0].response
    assert len(ctx.conversation_history.messages) == 1
    mock_agent.resume.assert_called_once()
    mock_agent.assert_called_once()


def test_topology_b_resume_parity_steps(
    initialized_fastworkflow,
    todo_workflow_path,
    monkeypatch,
):
    ctx, wf = _make_agent_ctx(initialized_fastworkflow, todo_workflow_path, monkeypatch)

    parity_log: dict[str, object] = {}

    def tracking_post(clarification_request, user_response, chat_session_obj):
        parity_log["iteration_counter"] = chat_session_obj._workflow_tool_agent.iteration_counter
        parity_log["clarification"] = clarification_request
        parity_log["user_response"] = user_response
        wf.context["raw_user_message"] = user_response
        chat_session_obj.append_action_log(
            {
                "agent_query": clarification_request,
                "user_response": user_response,
            }
        )
        parity_log["replan_called"] = True
        return "replanned observation"

    monkeypatch.setattr(
        "fastworkflow.workflow_agent._post_ask_user_response",
        tracking_post,
    )

    suspended = SimpleNamespace(suspended=True, clarification="Need detail?")
    completed = SimpleNamespace(final_answer="Finished")

    mock_agent = MagicMock()
    mock_agent.iteration_counter = 3
    mock_agent.return_value = suspended
    mock_agent.resume.return_value = completed
    _set_agents(ctx, mock_agent)

    ctx.process_message("start")
    ctx.process_message("user answer")

    assert parity_log["iteration_counter"] == -1
    assert parity_log["clarification"] == "Need detail?"
    assert parity_log["user_response"] == "user answer"
    assert parity_log["replan_called"] is True
    assert wf.context["raw_user_message"] == "user answer"
    assert len(ctx.action_log) == 1
    assert ctx.action_log[0]["user_response"] == "user answer"


def test_abort_resets_awaiting_user(
    initialized_fastworkflow,
    todo_workflow_path,
    monkeypatch,
):
    ctx, _wf = _make_agent_ctx(initialized_fastworkflow, todo_workflow_path, monkeypatch)

    suspended = SimpleNamespace(suspended=True, clarification="Which one?")
    completed = SimpleNamespace(final_answer="Done after fresh turn")

    mock_agent = MagicMock()
    mock_agent.return_value = suspended
    mock_agent.resume.side_effect = CommandCancelledError("aborted during resume")
    _set_agents(ctx, mock_agent)

    ctx.process_message("first turn")
    assert ctx.awaiting_user

    failed = ctx.process_message("user tries to answer")
    assert not ctx.awaiting_user
    assert not failed.success

    mock_agent.return_value = completed
    fresh = ctx.process_message("brand new turn")
    assert fresh.success
    assert "Done after fresh turn" in fresh.command_responses[0].response
    assert mock_agent.call_count == 2
    mock_agent.resume.assert_called_once()


def test_cancel_pending_clears_awaiting_user(
    initialized_fastworkflow,
    todo_workflow_path,
    monkeypatch,
):
    ctx, _wf = _make_agent_ctx(initialized_fastworkflow, todo_workflow_path, monkeypatch)

    mock_agent = MagicMock()
    mock_agent.return_value = SimpleNamespace(
        suspended=True, clarification="Still waiting?"
    )
    _set_agents(ctx, mock_agent)

    ctx.process_message("question me")
    assert ctx.awaiting_user
    assert ctx.cancel_pending() is True
    assert not ctx.awaiting_user
    assert ctx.cancel_pending() is False


def test_resolve_or_escalate_executes_clarified_command(
    initialized_fastworkflow,
    todo_workflow_path,
    monkeypatch,
):
    from fastworkflow.workflow_agent import _resolve_or_escalate

    ctx = WorkflowExecutionContext(run_as_agent=True)
    wf = fastworkflow.Workflow.create(
        todo_workflow_path,
        workflow_id_str=f"intent-resolve-{uuid.uuid4().hex}",
    )
    ctx.bind_app_workflow(wf)
    calls: list[str] = []

    def record_execute(cmd, chat_session_obj):
        calls.append(cmd)
        return f"executed:{cmd}"

    monkeypatch.setattr(
        "fastworkflow.workflow_agent._execute_workflow_query",
        record_execute,
    )

    result = SimpleNamespace(
        clarified_command="add_task <title>x</title>",
        needs_human=False,
        clarification_question="",
    )
    observation = _resolve_or_escalate(result, ctx, "ambiguous error")
    assert observation == "executed:add_task <title>x</title>"
    assert calls == ["add_task <title>x</title>"]


def test_resolve_or_escalate_escalates_to_outer_ask_user(
    initialized_fastworkflow,
    todo_workflow_path,
    monkeypatch,
):
    from fastworkflow.workflow_agent import _resolve_or_escalate

    ctx = WorkflowExecutionContext(run_as_agent=True)
    wf = fastworkflow.Workflow.create(
        todo_workflow_path,
        workflow_id_str=f"intent-escalate-{uuid.uuid4().hex}",
    )
    ctx.bind_app_workflow(wf)
    calls: list[str] = []

    def record_execute(cmd, chat_session_obj):
        calls.append(cmd)
        return "abort confirmed"

    monkeypatch.setattr(
        "fastworkflow.workflow_agent._execute_workflow_query",
        record_execute,
    )

    result = SimpleNamespace(
        clarified_command="",
        needs_human=True,
        clarification_question="Which list?",
    )
    observation = _resolve_or_escalate(result, ctx, "fallback error text")
    assert calls == ["abort"]
    assert "ask_user" in observation
    assert "Which list?" in observation


def test_topology_a_cli_ask_user_blocks_with_queue(
    initialized_fastworkflow,
    todo_workflow_path,
    monkeypatch,
):
    from fastworkflow.workflow_agent import _ask_user_tool

    ctx = WorkflowExecutionContext(run_as_agent=True)
    wf = fastworkflow.Workflow.create(
        todo_workflow_path,
        workflow_id_str=f"cli-ask-{uuid.uuid4().hex}",
    )
    ctx.bind_app_workflow(wf)
    user_queue: Queue = Queue()
    ctx.set_transport_queues(
        user_message_queue=user_queue,
        command_output_queue=Queue(),
    )

    monkeypatch.setattr(
        "fastworkflow.workflow_agent.build_query_with_next_steps",
        lambda user_query, session, with_agent_inputs_and_trajectory=False: (
            f"replanned:{user_query}"
        ),
    )

    def deliver_answer():
        time.sleep(0.05)
        user_queue.put("user picks option A")

    threading.Thread(target=deliver_answer, daemon=True).start()

    ctx.push_active_workflow(wf)
    try:
        observation = _ask_user_tool("Pick A or B?", ctx)
    finally:
        ctx.pop_active_workflow()

    assert observation == "replanned:user picks option A"
    assert wf.context["raw_user_message"] == "user picks option A"
    assert len(ctx.action_log) == 1

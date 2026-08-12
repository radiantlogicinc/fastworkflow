"""
Tests for PR #56: context-aware command visibility.

Covers:
- Workflow context-change listeners (notify / unsubscribe / exception isolation)
- ReAct resume keeping self.inputs aliased to the active run's input_args
- get_all_contexts_command_display_text multi-context sections + safe fallback
- _refresh_agent_available_commands updating available_commands on context change
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

import fastworkflow
from fastworkflow.command_metadata_api import CommandMetadataAPI
from fastworkflow.command_routing import RoutingRegistry
from fastworkflow.utils.react import fastWorkflowReAct
from fastworkflow.workflow_agent import _refresh_agent_available_commands


def _todo_list_path() -> str:
    return str((Path(__file__).parent / "todo_list_workflow").resolve())


def _cme_workflow_path() -> str:
    return fastworkflow.get_internal_workflow_path("command_metadata_extraction")


@pytest.fixture
def todo_list_env():
    RoutingRegistry.clear_registry()
    fastworkflow.init(
        {
            "NOT_FOUND": "NOT_FOUND",
        }
    )
    yield
    RoutingRegistry.clear_registry()


def test_context_change_notifies_and_unsubscribe(todo_list_env):
    workflow = fastworkflow.Workflow.create(
        _todo_list_path(), workflow_id_str="ctx-listener-notify"
    )
    seen: list[object] = []

    def on_change():
        seen.append(workflow.current_command_context)

    workflow.add_context_change_listener(on_change)
    marker_a = object()
    marker_b = object()

    workflow.current_command_context = marker_a
    assert seen == [marker_a]

    # Same object identity: no notification
    workflow.current_command_context = marker_a
    assert seen == [marker_a]

    workflow.current_command_context = marker_b
    assert seen == [marker_a, marker_b]

    workflow.remove_context_change_listener(on_change)
    workflow.current_command_context = marker_a
    assert seen == [marker_a, marker_b]


def test_context_change_listener_exception_swallowed(todo_list_env, caplog):
    workflow = fastworkflow.Workflow.create(
        _todo_list_path(), workflow_id_str="ctx-listener-exc"
    )
    good: list[int] = []

    def boom():
        raise RuntimeError("listener blew up")

    def ok():
        good.append(1)

    workflow.add_context_change_listener(boom)
    workflow.add_context_change_listener(ok)

    with caplog.at_level(logging.WARNING):
        workflow.current_command_context = object()

    assert good == [1]
    assert workflow.current_command_context is not None
    assert any("context-change listener failed" in r.message for r in caplog.records)


def test_bound_method_listener_can_be_removed(todo_list_env):
    workflow = fastworkflow.Workflow.create(
        _todo_list_path(), workflow_id_str="ctx-listener-bound"
    )
    hits: list[int] = []

    class Observer:
        def on_change(self):
            hits.append(1)

    obs = Observer()
    workflow.add_context_change_listener(obs.on_change)
    workflow.current_command_context = object()
    assert hits == [1]

    workflow.remove_context_change_listener(obs.on_change)
    workflow.current_command_context = object()
    assert hits == [1]


def test_react_resume_aliases_inputs_to_active_run_args():
    """resume must set self.inputs to the stashed input_args dict (same object)."""
    agent = fastWorkflowReAct.__new__(fastWorkflowReAct)
    agent.iteration_counter = 0
    agent.max_iters = 5
    agent.inputs = {"available_commands": "stale"}
    agent.current_trajectory = {}
    agent.tools = {"finish": lambda: "done"}
    agent.react = object()
    agent.extract = object()

    active_args = {"query": "hello", "available_commands": "from-stash"}
    agent._suspended = {
        "trajectory": {
            "thought_0": "ask",
            "tool_name_0": "ask_user",
            "tool_args_0": {},
        },
        "idx": 0,
        "input_args": active_args,
        "max_iters": 5,
        "clarification": "Which?",
    }

    def react_after_resume(trajectory, **input_args):
        # Mid-run refresh mutates agent.inputs; the loop must see it via input_args.
        assert agent.inputs is active_args
        assert input_args is active_args
        return SimpleNamespace(
            next_thought="finish",
            next_tool_name="finish",
            next_tool_args={},
        )

    agent.react = react_after_resume  # type: ignore[method-assign]
    agent.extract = lambda trajectory, **input_args: {"final_answer": "ok"}  # type: ignore[method-assign]
    agent._call_with_potential_trajectory_truncation = (  # type: ignore[method-assign]
        lambda module, trajectory, **kwargs: (
            module(trajectory, **kwargs)
            if module is agent.react
            else module(trajectory, **kwargs)
        )
    )

    # Simpler: patch _run_loop to assert alias then complete
    def fake_run_loop(trajectory, idx, input_args, max_iters, exception_count):
        assert agent.inputs is active_args
        assert input_args is active_args
        # Simulate a mid-run refresh mutating available_commands
        agent.inputs["available_commands"] = "refreshed"
        assert input_args["available_commands"] == "refreshed"
        return None

    agent._run_loop = fake_run_loop  # type: ignore[method-assign]
    agent._exhausted_last_run = False
    agent._call_with_potential_trajectory_truncation = (  # type: ignore[method-assign]
        lambda module, trajectory, **kwargs: module(trajectory, **kwargs)
    )

    result = agent.resume("answer")
    assert agent.inputs is active_args
    assert active_args["available_commands"] == "refreshed"
    assert result.final_answer == "ok"


def test_get_all_contexts_command_display_text_multi_sections(todo_list_env):
    subject = _todo_list_path()
    cme = _cme_workflow_path()

    text = CommandMetadataAPI.get_all_contexts_command_display_text(
        subject_workflow_path=subject,
        cme_workflow_path=cme,
        active_context_name="*",
        for_agents=True,
    )
    assert isinstance(text, str)
    assert "Commands available" in text
    # Non-active contexts should introduce their own sections
    assert "TodoListManager" in text or "after entering the TodoListManager" in text
    assert "TodoItem" in text or "TodoList" in text


def test_get_all_contexts_command_display_text_fallback_logs(
    todo_list_env, monkeypatch, caplog
):
    subject = _todo_list_path()
    cme = _cme_workflow_path()

    real_get = RoutingRegistry.get_definition
    real_display = CommandMetadataAPI.get_command_display_text
    armed = {"fail": False}

    def get_def(path):
        if armed["fail"]:
            raise RuntimeError("routing unavailable")
        return real_get(path)

    def display(*args, **kwargs):
        out = real_display(*args, **kwargs)
        # Fail the multi-context assembly that runs after base_text is built.
        armed["fail"] = True
        return out

    monkeypatch.setattr(RoutingRegistry, "get_definition", get_def)
    monkeypatch.setattr(
        CommandMetadataAPI, "get_command_display_text", staticmethod(display)
    )

    with caplog.at_level(logging.WARNING):
        text = CommandMetadataAPI.get_all_contexts_command_display_text(
            subject_workflow_path=subject,
            cme_workflow_path=cme,
            active_context_name="*",
            for_agents=True,
        )

    armed["fail"] = False
    monkeypatch.setattr(
        CommandMetadataAPI, "get_command_display_text", staticmethod(real_display)
    )
    base = CommandMetadataAPI.get_command_display_text(
        subject_workflow_path=subject,
        cme_workflow_path=cme,
        active_context_name="*",
        for_agents=True,
    )
    assert text == base
    assert any(
        "Failed to assemble command metadata" in r.message for r in caplog.records
    )


def test_refresh_agent_available_commands_on_context_change(todo_list_env):
    subject = _todo_list_path()
    workflow = fastworkflow.Workflow.create(
        subject, workflow_id_str="ctx-refresh-agent"
    )

    class FakeAgent:
        def __init__(self):
            self.inputs = {"available_commands": "INITIAL"}

    class FakeHost:
        def __init__(self, app_workflow):
            self._app_workflow = app_workflow
            self.workflow_tool_agent = FakeAgent()

        def get_active_workflow(self):
            return self._app_workflow

    host = FakeHost(workflow)
    workflow.add_context_change_listener(
        lambda: _refresh_agent_available_commands(host)
    )

    class TodoListManager:
        pass

    workflow.current_command_context = TodoListManager()
    refreshed = host.workflow_tool_agent.inputs["available_commands"]
    assert refreshed != "INITIAL"
    assert isinstance(refreshed, str)
    assert "TodoListManager" in refreshed or "create_todo_list" in refreshed.lower() or len(refreshed) > 10

    # No-op when agent has no available_commands key
    host.workflow_tool_agent.inputs = {"query": "x"}
    workflow.current_command_context = object()
    assert host.workflow_tool_agent.inputs == {"query": "x"}


def test_refresh_finds_wec_private_agent_and_app_workflow_fallback(todo_list_env):
    """WEC-shaped host: private _workflow_tool_agent + empty active stack → still refresh."""
    subject = _todo_list_path()
    workflow = fastworkflow.Workflow.create(
        subject, workflow_id_str="ctx-refresh-wec-shape"
    )

    class FakeAgent:
        def __init__(self):
            self.inputs = {"available_commands": "STALE"}

    class WECShapedHost:
        """Mirrors WorkflowExecutionContext attribute layout without a public agent field."""

        def __init__(self, app_workflow):
            self._app_workflow = app_workflow
            self._workflow_tool_agent = FakeAgent()

        @property
        def workflow_tool_agent(self):
            return self._workflow_tool_agent

        @property
        def app_workflow(self):
            return self._app_workflow

        def get_active_workflow(self):
            # Match WEC when the contextvar stack is empty.
            return None

    host = WECShapedHost(workflow)
    _refresh_agent_available_commands(host)
    refreshed = host._workflow_tool_agent.inputs["available_commands"]
    assert refreshed != "STALE"
    assert isinstance(refreshed, str)
    assert len(refreshed) > 0

    # Private-only host (no property): still resolve via _workflow_tool_agent.
    class PrivateOnlyHost:
        def __init__(self, app_workflow):
            self._app_workflow = app_workflow
            self._workflow_tool_agent = FakeAgent()

    private = PrivateOnlyHost(workflow)
    _refresh_agent_available_commands(private)
    assert private._workflow_tool_agent.inputs["available_commands"] != "STALE"


def test_wec_close_removes_context_change_listener(todo_list_env):
    """WEC.close() must unsubscribe its bound context-change listener."""
    from fastworkflow.workflow_execution_context import WorkflowExecutionContext

    subject = _todo_list_path()
    workflow = fastworkflow.Workflow.create(
        subject, workflow_id_str="ctx-wec-close"
    )

    hits: list[int] = []

    class MiniWEC:
        def __init__(self):
            self._app_workflow = workflow
            self._context_change_listener = None

        def _on_app_context_change(self):
            hits.append(1)

        def close(self):
            listener = self._context_change_listener
            if listener is not None and self._app_workflow is not None:
                self._app_workflow.remove_context_change_listener(listener)
                self._context_change_listener = None

    wec = MiniWEC()
    workflow.add_context_change_listener(wec._on_app_context_change)
    wec._context_change_listener = wec._on_app_context_change

    workflow.current_command_context = object()
    assert hits == [1]

    wec.close()
    workflow.current_command_context = object()
    assert hits == [1]

    # Real WEC.close path also clears the listener attribute when present
    real = WorkflowExecutionContext.__new__(WorkflowExecutionContext)
    real._app_workflow = workflow
    real._cme_workflow = None
    real._context_change_listener = wec._on_app_context_change
    workflow.add_context_change_listener(wec._on_app_context_change)
    assert real.close() is True
    assert real._context_change_listener is None

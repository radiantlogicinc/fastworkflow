"""A multi-command execute_workflow_query call is refused before anything runs.

Intent detection would execute only the first command of a bundle and drop the
rest, so _execute_workflow_query refuses any call holding two or more lines that
start with a known command name. Fixtures mirror tests/test_workflow_agent.py.
"""

from pathlib import Path

import pytest

import fastworkflow
from fastworkflow.chat_session import ChatSession
from fastworkflow.command_executor import CommandExecutor
from fastworkflow import workflow_agent


@pytest.fixture
def initialized_fastworkflow():
    from fastworkflow.command_routing import RoutingRegistry
    RoutingRegistry.clear_registry()
    fastworkflow.init(env_vars={})
    yield
    fastworkflow.chat_session = None
    RoutingRegistry.clear_registry()


@pytest.fixture
def session(initialized_fastworkflow):
    chat_session = ChatSession(run_as_agent=True)
    todo_path = str(Path(__file__).parent.joinpath("todo_list_workflow").resolve())
    workflow = fastworkflow.Workflow.create(todo_path, workflow_id_str="multi-cmd-test")
    chat_session.push_active_workflow(workflow)
    return chat_session


@pytest.fixture
def executed(monkeypatch):
    """Records every command that reaches the executor; runs nothing real."""
    calls = []

    def fake_invoke(cls, session, command: str):
        calls.append(command)
        return fastworkflow.CommandOutput(
            command_name=command.split()[0],
            command_response=fastworkflow.CommandResponse(response=f"ok:{command}"),
        )

    monkeypatch.setattr(CommandExecutor, "invoke_command", classmethod(fake_invoke))
    return calls


def test_a_bundle_of_commands_is_refused_and_nothing_runs(session, executed, monkeypatch):
    resolved = []
    monkeypatch.setattr(workflow_agent, "_explicit_agent_command",
                        lambda command, workflow: resolved.append(command) or command)

    result = workflow_agent._execute_workflow_query(
        "reset_context <context>\ngo_up\nwhat_can_i_do\nwhat_can_i_do",
        chat_session_obj=session,
    )

    assert result == ("You sent 4 commands in one call; nothing ran. "
                      "Send one command per call: reset_context <context> first.")
    assert resolved == []
    assert executed == []


def test_a_single_command_still_runs(session, executed):
    result = workflow_agent._execute_workflow_query("what_can_i_do", chat_session_obj=session)

    assert "nothing ran" not in result
    assert executed == ["what_can_i_do"]


def test_a_multiline_parameter_that_is_not_a_command_line_still_runs(session, executed):
    result = workflow_agent._execute_workflow_query(
        "what_can_i_do\nplease list everything you can do",
        chat_session_obj=session,
    )

    assert "nothing ran" not in result
    assert len(executed) == 1
    assert executed[0].startswith("what_can_i_do")

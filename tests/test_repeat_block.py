"""A command that repeats its own result is blocked for the next step only.

Same command text, same context, identical response as the previous call in
this executor run: the command is left out of the next step's available
commands and a call to it is refused unrun. Fixtures mirror
tests/test_multi_command_refusal.py.
"""

import pytest

import fastworkflow
from fastworkflow.command_executor import CommandExecutor
from fastworkflow import workflow_agent
from fastworkflow.workflow_agent import initialize_workflow_tool_agent
from tests.test_multi_command_refusal import initialized_fastworkflow, session  # noqa: F401

COMMAND = "get_all_children"
REFUSAL = ("get_all_children is unavailable for this step: it just returned the same "
           "result again. Use what it returned, or do something else.")


@pytest.fixture
def responses(monkeypatch):
    """Scripted executor: each call returns the next scripted response; records calls."""
    state = {"calls": [], "next": []}

    def fake_invoke(cls, session, command: str):
        state["calls"].append(command)
        text = state["next"].pop(0) if state["next"] else "same"
        return fastworkflow.CommandOutput(
            command_name=command.split()[0],
            command_response=fastworkflow.CommandResponse(response=text),
        )

    monkeypatch.setattr(CommandExecutor, "invoke_command", classmethod(fake_invoke))
    monkeypatch.setattr(workflow_agent, "_explicit_agent_command", lambda command, workflow: command)
    return state


@pytest.fixture
def agent(session):
    """The session's workflow agent, with the available_commands input a run would carry."""
    tool_agent = initialize_workflow_tool_agent(session)
    session._core._workflow_tool_agent = tool_agent
    tool_agent.inputs = {"available_commands": workflow_agent._what_can_i_do(session)}
    return tool_agent


def run_step(session, agent, command: str) -> str:
    """One ReAct step: the tool call, then the end-of-step hook the loop runs."""
    result = workflow_agent._execute_workflow_query(command, chat_session_obj=session)
    agent.on_step_end()
    return result


def test_identical_repeat_blocks_next_step_only(session, agent, responses):
    assert COMMAND in agent.inputs["available_commands"]
    run_step(session, agent, COMMAND)
    run_step(session, agent, COMMAND)  # same output: repeat

    assert COMMAND not in agent.inputs["available_commands"]
    assert run_step(session, agent, COMMAND) == REFUSAL
    assert len(responses["calls"]) == 2

    # The refusal step is over: the command is back, and it runs again.
    assert COMMAND in agent.inputs["available_commands"]
    run_step(session, agent, COMMAND)
    assert len(responses["calls"]) == 3


def test_different_output_is_not_a_repeat(session, agent, responses):
    responses["next"] = ["first", "second"]
    run_step(session, agent, COMMAND)
    run_step(session, agent, COMMAND)

    assert COMMAND in agent.inputs["available_commands"]
    assert len(responses["calls"]) == 2


def test_same_command_in_a_different_context_is_not_a_repeat(session, agent, responses, monkeypatch):
    workflow = session.get_active_workflow()
    context = {"name": workflow.current_command_context_name}
    monkeypatch.setattr(type(workflow), "current_command_context_name",
                        property(lambda self: context["name"]))
    run_step(session, agent, COMMAND)
    context["name"] = "TodoListManager"
    run_step(session, agent, COMMAND)

    assert COMMAND in agent.inputs["available_commands"]
    assert len(responses["calls"]) == 2

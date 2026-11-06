"""The agent's command tool refuses a call that names no workflow command (fix-wfxt).

A blank command ("", whitespace) and a call that names the tool itself
(``execute_workflow_query``) used to reach intent detection, where blank text
raises ValueError and the tool name was classified as an utterance. Both are
refused unrun, with an observation that tells the agent to name a workflow
command. Blank text is also an empty command at the executor. The executor is
spied on, not replaced, so the real dispatch runs for anything not refused.
"""

import pytest

from fastworkflow import workflow_agent
from fastworkflow.command_executor import CommandExecutor
from fastworkflow.workflow_agent import initialize_workflow_tool_agent
from tests.test_tracing_phase1 import (  # noqa: F401
    _make_agent_ctx,
    initialized_fastworkflow,
    todo_workflow_path,
)

TOOL = "execute_workflow_query"


@pytest.fixture
def ctx(initialized_fastworkflow, todo_workflow_path, monkeypatch):  # noqa: F811
    """An agent-mode context with an active turn over the todo workflow."""
    context, wf = _make_agent_ctx(todo_workflow_path, monkeypatch)
    context.push_active_workflow(wf)
    context._begin_turn("find Ann")
    return context


@pytest.fixture
def agent(ctx):
    """The agent's tool set on that context, with the available_commands input a run would carry."""
    tool_agent = initialize_workflow_tool_agent(ctx)
    ctx._workflow_tool_agent = tool_agent
    tool_agent.inputs = {"available_commands": workflow_agent._what_can_i_do(ctx)}
    return tool_agent


@pytest.fixture
def invoked(monkeypatch):
    """Records each command that reaches the executor, then runs it for real."""
    calls = []
    real = CommandExecutor.invoke_command.__func__

    def spy(cls, chat_session, command: str):
        calls.append(command)
        return real(cls, chat_session, command)

    monkeypatch.setattr(CommandExecutor, "invoke_command", classmethod(spy))
    return calls


@pytest.mark.parametrize("command", ["", " ", "   ", "\t\n"])
def test_a_blank_command_is_refused_unrun(agent, invoked, command):
    observation = agent.tools[TOOL](command=command)
    assert invoked == []
    assert observation.startswith("No command was given, so nothing ran.")
    assert "Name a workflow command" in observation


@pytest.mark.parametrize("command", ["execute_workflow_query", "/execute_workflow_query",
                                     "  Execute_Workflow_Query  "])
def test_the_tool_name_as_the_command_is_refused_unrun(agent, invoked, command):
    observation = agent.tools[TOOL](command=command)
    assert invoked == []
    assert observation.startswith("execute_workflow_query is this tool, not a workflow command")
    assert "nothing ran" in observation
    assert "Name a workflow command" in observation


def test_whitespace_text_is_an_empty_command_at_the_executor(ctx):
    """The executor answers blank text as it answers an empty string, without intent detection."""
    for command in ("", "   ", "\t\n"):
        output = CommandExecutor.invoke_command(ctx, command)
        assert "hit the <Enter> key" in output.command_response.response


def test_blank_text_answers_no_trained_head(ctx):
    """The sub-task gate's head lookup answers blank text with nothing, without calling a head's predict."""
    from fastworkflow import run_listing

    workflow = ctx.get_active_workflow()
    for text in ("", "   "):
        assert run_listing.head_answers(workflow, text) == {}
        assert run_listing.head_candidates(workflow, text).labels == frozenset()

"""
Tests for the new workflow_agent.py functionality.
"""

import json
import pytest
import fastworkflow
from fastworkflow.chat_session import ChatSession
from fastworkflow.workflow_agent import (
    initialize_workflow_tool_agent,
)


@pytest.fixture
def mock_env_vars():
    return {
        "SPEEDDICT_FOLDERNAME": "___workflow_contexts",
        "LLM_AGENT": "gpt-4",
        "LITELLM_API_KEY_AGENT": "test-key",
        "NOT_FOUND": "NOT_FOUND",
        "PARAMETER_EXTRACTION_ERROR_MSG": "Parameter extraction error: {error}",
        "MISSING_INFORMATION_ERRMSG": "Missing information: ",
        "INVALID_INFORMATION_ERRMSG": "Invalid information: ",
    }


@pytest.fixture
def initialized_fastworkflow(mock_env_vars):
    from fastworkflow.command_routing import RoutingRegistry
    RoutingRegistry.clear_registry()
    fastworkflow.init(env_vars=mock_env_vars)
    yield
    fastworkflow.chat_session = None
    RoutingRegistry.clear_registry()


def test_initialize_tool_agent(initialized_fastworkflow):
    chat_session = ChatSession(run_as_agent=True)
    from fastworkflow.mcp_server import FastWorkflowMCPServer
    mcp_server = FastWorkflowMCPServer(chat_session)
    agent = initialize_workflow_tool_agent(mcp_server)
    assert agent is not None


def test_what_can_i_do_tool_executes(initialized_fastworkflow):
    chat_session = ChatSession(run_as_agent=True)
    # Create a simple workflow and set active
    from pathlib import Path
    retail_path = str(Path(__file__).parent.parent.joinpath("fastworkflow", "examples", "retail_workflow").resolve())
    workflow = fastworkflow.Workflow.create(retail_path, workflow_id_str="agent-test")
    chat_session.push_active_workflow(workflow)

    from fastworkflow.mcp_server import FastWorkflowMCPServer
    mcp_server = FastWorkflowMCPServer(chat_session)
    agent = initialize_workflow_tool_agent(mcp_server)

    # Call the underlying helper to validate behavior without relying on agent internals
    from fastworkflow.workflow_agent import _what_can_i_do
    result = _what_can_i_do(chat_session)
    assert isinstance(result, str)
    # Should include header or JSON depending on mode detection; accept either
    try:
        data = json.loads(result)
        assert "commands" in data
    except Exception:
        assert "Commands available" in result

    chat_session.clear_workflow_stack()



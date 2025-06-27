from __future__ import annotations

import os
import uuid
from dotenv import dotenv_values
import pytest

import fastworkflow
from fastworkflow.command_executor import CommandExecutor
from fastworkflow.chat_session import ChatSession


@pytest.fixture(scope="module")
def retail_workflow_path() -> str:
    """Absolute path to the retail workflow example used by the integration tests."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "examples", "retail_workflow"))


@pytest.fixture(scope="function")
def chat_session(retail_workflow_path: str):
    """Spin up an in-memory chat session for each test so state cannot leak."""
    env_vars = {
        **dotenv_values("./env/.env"),
        **dotenv_values("./passwords/.env")
    }
    fastworkflow.init(env_vars)
    fastworkflow.CommandContextModel.load(retail_workflow_path)
    fastworkflow.RoutingRegistry.clear_registry()
    fastworkflow.RoutingRegistry.get_definition(retail_workflow_path)

    return ChatSession(
        retail_workflow_path,
        workflow_id_str=str(uuid.uuid4())
    )


class TestCommandExecutor:
    """Basic sanity checks for the refactored CommandExecutor.perform_action."""

    def test_perform_action_simple_command(self, chat_session: ChatSession):
        """Ensure a parameter-free command can be executed successfully."""
        action = fastworkflow.Action(
            command_name="list_all_product_types",
            command="List all the product categories you have.",
            parameters={},
        )

        result = CommandExecutor.perform_action(chat_session.app_workflow, action)

        assert isinstance(result, fastworkflow.CommandOutput)
        assert result.success is True
        assert any("product" in resp.response.lower() for resp in result.command_responses)

    def test_perform_action_with_parameters(self, chat_session: ChatSession):
        """Execute a command that expects parameters and verify validation passes."""
        action = fastworkflow.Action(
            command_name="find_user_id_by_email",
            command="Find the user id for john.doe@example.com",
            parameters={"email": "john.doe@example.com"},
        )

        result = CommandExecutor.perform_action(chat_session.app_workflow, action)

        assert isinstance(result, fastworkflow.CommandOutput)
        # Response text should contain a user id (pattern xyz_xyz_\d+)
        assert any("user id" in resp.response.lower() for resp in result.command_responses) 
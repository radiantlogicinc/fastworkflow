from __future__ import annotations

import os
import uuid
from dotenv import dotenv_values
import pytest

import fastworkflow
from fastworkflow.command_executor import CommandExecutor
from fastworkflow.workflow_session import WorkflowSession


@pytest.fixture(scope="module")
def retail_workflow_path() -> str:
    """Absolute path to the retail workflow example used by the integration tests."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "examples", "retail_workflow"))


@pytest.fixture(scope="module")
def command_executor() -> CommandExecutor:
    return CommandExecutor()


@pytest.fixture(scope="function")
def workflow_session(retail_workflow_path: str, command_executor: CommandExecutor):
    """Spin up an in-memory workflow session for each test so state cannot leak."""
    env_vars = {
        **dotenv_values("./env/.env"),
        **dotenv_values("./passwords/.env")
    }
    fastworkflow.init(env_vars)
    fastworkflow.CommandContextModel.load(retail_workflow_path)
    fastworkflow.RoutingRegistry.clear_registry()
    fastworkflow.RoutingRegistry.get_definition(retail_workflow_path)

    return WorkflowSession(
        command_executor,
        retail_workflow_path,
        session_id_str=str(uuid.uuid4())
    )


class TestCommandExecutor:
    """Basic sanity checks for the refactored CommandExecutor.perform_action."""

    def test_perform_action_simple_command(self, command_executor: CommandExecutor, workflow_session: WorkflowSession):
        """Ensure a parameter-free command can be executed successfully."""
        action = fastworkflow.Action(
            command_name="list_all_product_types",
            command="List all the product categories you have.",
            parameters={},
        )

        result = command_executor.perform_action(workflow_session.session, action)

        assert isinstance(result, fastworkflow.CommandOutput)
        assert result.success is True
        assert any("product" in resp.response.lower() for resp in result.command_responses)

    def test_perform_action_with_parameters(self, command_executor: CommandExecutor, workflow_session: WorkflowSession):
        """Execute a command that expects parameters and verify validation passes."""
        action = fastworkflow.Action(
            command_name="find_user_id_by_email",
            command="Find the user id for john.doe@example.com",
            parameters={"email": "john.doe@example.com"},
        )

        result = command_executor.perform_action(workflow_session.session, action)

        assert isinstance(result, fastworkflow.CommandOutput)
        # Response text should contain a user id (pattern xyz_xyz_\d+)
        assert any("user id" in resp.response.lower() for resp in result.command_responses) 
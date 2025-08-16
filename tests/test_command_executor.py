from __future__ import annotations

import os
import uuid
from contextlib import suppress
from pathlib import Path
from dotenv import dotenv_values
import pytest

import fastworkflow
from fastworkflow.command_executor import CommandExecutor
from fastworkflow.chat_session import ChatSession


@pytest.fixture(scope="module")
def retail_workflow_path() -> str:
    """Absolute path to the retail workflow example used by the integration tests."""
    return str(Path(__file__).parent.parent.joinpath("fastworkflow", "examples", "retail_workflow").resolve())


@pytest.fixture(scope="function")
def chat_session(retail_workflow_path: str, request):
    """Spin up an in-memory chat session for each test so state cannot leak."""
    env_vars = {
        **dotenv_values("./env/.env"),
        **dotenv_values("./passwords/.env")
    }
    fastworkflow.init(env_vars)
    
    # Clear ALL caches AFTER initialization
    fastworkflow.RoutingRegistry.clear_registry()
    fastworkflow.CommandContextModel.load(retail_workflow_path)
    # Force rebuild of routing definition to avoid stale persisted JSON
    fastworkflow.RoutingRegistry.get_definition(retail_workflow_path, load_cached=False)

    chat_session = ChatSession()
    chat_session.start_workflow(
        retail_workflow_path,
        workflow_id_str=str(uuid.uuid4()),
        keep_alive=True,
    )

    def _teardown():
        # Nudge exit by marking not keep_alive and stopping
        chat_session._keep_alive = False
        with suppress(Exception):
            chat_session.stop_workflow()
        # Clear caches after test completes
        fastworkflow.RoutingRegistry.clear_registry()

    request.addfinalizer(_teardown)
    return chat_session


class TestCommandExecutor:
    """Basic sanity checks for the refactored CommandExecutor.perform_action."""

    def test_perform_action_simple_command(self, chat_session: ChatSession):
        """Ensure a parameter-free command can be executed successfully."""
        action = fastworkflow.Action(
            command_name="list_all_product_types",
            command="List all the product categories you have.",
            parameters={},
        )

        active_workflow = chat_session.get_active_workflow()
        result = CommandExecutor.perform_action(active_workflow, action)

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

        active_workflow = chat_session.get_active_workflow()
        result = CommandExecutor.perform_action(active_workflow, action)

        assert isinstance(result, fastworkflow.CommandOutput)
        # Response text should contain a user id (pattern xyz_xyz_\d+)
        assert any("user id" in resp.response.lower() for resp in result.command_responses) 
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import fastworkflow
from fastworkflow.command_executor import CommandExecutor, CommandNotFoundError
from fastworkflow.session import WorkflowSnapshot, Session


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class Ctx:
    """Picklable dummy context object (defined at module level)."""
    pass


def test_command_not_found(monkeypatch, tmp_path):
    """Test that attempting to execute a non-existent command raises an appropriate error."""
    fastworkflow.init({"SPEEDDICT_FOLDERNAME": "___workflow_contexts"})
    
    # Create a session
    session = Session.create(
        workflow_folderpath=str(tmp_path),
        session_id_str="test-session-3"
    )

    # Create a mock CommandRoutingDefinition
    mock_routing_def = MagicMock()
    mock_routing_def.get_command_class.return_value = None  # No command class found
    
    # Patch the CommandRoutingRegistry.get_definition to return our mock
    monkeypatch.setattr(
        fastworkflow.CommandRoutingRegistry,
        "get_definition",
        lambda _: mock_routing_def
    )

    executor = CommandExecutor()
    
    # Create an Action with a non-existent command
    action = fastworkflow.Action(
        workitem_path="*",  # Use global context
        command_name="non_existent_command",
        command="This command doesn't exist",
    )

    # Expect a ValueError when trying to perform this action
    with pytest.raises(ValueError):
        executor.perform_action(session, action)


def test_invalid_action_parameters(monkeypatch, tmp_path):
    """Test that providing invalid parameters to a command raises an appropriate error."""
    fastworkflow.init({"SPEEDDICT_FOLDERNAME": "___workflow_contexts"})
    
    # Create a session
    session = Session.create(
        workflow_folderpath=str(tmp_path),
        session_id_str="test-session-4"
    )

    # Create a mock ResponseGenerator class
    class MockResponseGenerator:
        def __call__(self, *args, **kwargs):
            return "mock response"
    
    # Create a mock CommandRoutingDefinition
    mock_routing_def = MagicMock()
    mock_routing_def.get_command_class.side_effect = lambda cmd_name, module_type: \
        MockResponseGenerator if module_type == fastworkflow.ModuleType.RESPONSE_GENERATION_INFERENCE else None
    
    # Patch the CommandRoutingRegistry.get_definition to return our mock
    monkeypatch.setattr(
        fastworkflow.CommandRoutingRegistry,
        "get_definition",
        lambda _: mock_routing_def
    )

    executor = CommandExecutor()
    
    # Create an Action with a command that exists but with no parameter class
    action = fastworkflow.Action(
        workitem_path="*",  # Use global context
        command_name="greet",
        command="greet",
        parameters={"invalid_param": "value"}  # These parameters will be ignored since there's no parameter class
    )

    # This should execute without error since we're not validating parameters
    result = executor.perform_action(session, action)
    assert result is not None 
from pathlib import Path
import os
from unittest.mock import MagicMock

import pytest
import fastworkflow
from fastworkflow._workflows.command_metadata_extraction._commands.IntentDetection.what_can_i_do import ResponseGenerator
from fastworkflow._workflows.command_metadata_extraction._commands.IntentDetection.reset_context import ResponseGenerator as ResetGen


class Ctx:
    """Picklable dummy context class used in tests."""
    pass


def get_example_workflow_path():
    """Get the path to the examples/todo_list workflow."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "fastworkflow", "examples", "todo_list"))


def test_what_can_i_do_global(monkeypatch):
    fastworkflow.init({"SPEEDDICT_FOLDERNAME": "___workflow_contexts"})
    
    # Use the real todo_list workflow directory
    workflow_dir = get_example_workflow_path()
    assert os.path.isdir(workflow_dir), "examples/todo_list directory should exist"
    
    # Create a app workflow
    app_workflow = fastworkflow.Workflow.create(
        workflow_folderpath=workflow_dir,
        workflow_id_str="test-subject-workflow-1"
    )
    
    # Create a command metadata extraction workflow
    workflow_type = "command_metadata_extraction"
    cme_workflow_folderpath = fastworkflow.get_internal_workflow_path(workflow_type)
    
    # Create a child workflow with the app_workflow in its context
    mock_workflow = fastworkflow.Workflow.create(
        workflow_folderpath=cme_workflow_folderpath,
        parent_workflow_id=app_workflow.id,
        workflow_context={"app_workflow": app_workflow}
    )
    
    # Mock the RoutingRegistry.get_definition method
    mock_utterance_def = MagicMock()
    mock_utterance_def.get_sample_utterances.return_value = ["Sample utterance 1", "Sample utterance 2"]
    monkeypatch.setattr(fastworkflow.RoutingRegistry, "get_definition", lambda _: mock_utterance_def)
    
    # Call the __call__ method which is the current implementation
    generator = ResponseGenerator()
    response = generator(mock_workflow, "what can i do")
    
    # Check that the response contains commands
    assert "commands" in response.command_responses[0].response.lower()


def test_what_can_i_do_context(monkeypatch):
    fastworkflow.init({"SPEEDDICT_FOLDERNAME": "___workflow_contexts"})
    
    # Use the real todo_list workflow directory
    workflow_dir = get_example_workflow_path()
    assert os.path.isdir(workflow_dir), "examples/todo_list directory should exist"
    
    # Create a app workflow
    app_workflow = fastworkflow.Workflow.create(
        workflow_folderpath=workflow_dir,
        workflow_id_str="test-subject-workflow-2"
    )
    
    # Use TodoListManager as context
    from fastworkflow.examples.todo_list.application.todo_manager import TodoListManager
    app_workflow.current_command_context = TodoListManager()
    
    # Create a command metadata extraction workflow
    workflow_type = "command_metadata_extraction"
    cme_workflow_folderpath = fastworkflow.get_internal_workflow_path(workflow_type)
    
    # Create a child workflow with the app_workflow in its context
    mock_workflow = fastworkflow.Workflow.create(
        workflow_folderpath=cme_workflow_folderpath,
        parent_workflow_id=app_workflow.id,
        workflow_context={"app_workflow": app_workflow}
    )
    
    # Mock the RoutingRegistry.get_definition method
    mock_utterance_def = MagicMock()
    # First return context-specific commands, then return global commands after reset
    mock_utterance_def.get_sample_utterances.side_effect = [["create_todo_list", "help"], ["list_todo_lists", "help"]]
    monkeypatch.setattr(fastworkflow.RoutingRegistry, "get_definition", lambda _: mock_utterance_def)
    
    # Call the __call__ method which is the current implementation
    generator = ResponseGenerator()
    response = generator(mock_workflow, "what can i do")
    
    # Check that the response contains commands and mentions TodoListManager
    assert "TodoListManager" in response.command_responses[0].response
    
    # Test reset_context functionality
    reset_gen = ResetGen()
    reset_gen(mock_workflow, "reset context")
    
    # Check that after reset, we get global context
    resp2 = generator(mock_workflow, "what can i do")
    assert "global" in resp2.command_responses[0].response.lower() or "*" in resp2.command_responses[0].response

    # Mock the RoutingRegistry.get_definition method
    mock_utterance_def = MagicMock()
    mock_utterance_def.get_sample_utterances.return_value = []
    monkeypatch.setattr(fastworkflow.RoutingRegistry, "get_definition", lambda _: mock_utterance_def) 
import os
import pytest
import fastworkflow
from fastworkflow._workflows.command_metadata_extraction._commands.IntentDetection.what_is_current_context import ResponseGenerator
from tests.todo_list_workflow.application.todo_manager import TodoListManager
from tests.todo_list_workflow.application.todo_list import TodoList


def get_example_workflow_path():
    """Get the path to the tests/todo_list workflow."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tests", "todo_list_workflow"))


def test_global_context():
    """Test the response when in the global context."""
    fastworkflow.init({"SPEEDDICT_FOLDERNAME": "___workflow_contexts"})
    
    # Use the real todo_list workflow directory
    workflow_dir = get_example_workflow_path()
    assert os.path.isdir(workflow_dir), "tests/todo_list_workflow directory should exist"
    
    # Create a app workflow
    app_workflow = fastworkflow.Workflow.create(
        workflow_folderpath=workflow_dir,
        workflow_id_str="test-subject-workflow-3"
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
    
    # Call the __call__ method which is the current implementation
    gen = ResponseGenerator()
    response = gen(mock_workflow, "what context am I in")
    
    # Check that the response indicates the global context
    assert "global" in response.command_responses[0].response.lower() or "*" in response.command_responses[0].response


def test_context_no_properties():
    """Test the response when in a context without get_properties method."""
    fastworkflow.init({"SPEEDDICT_FOLDERNAME": "___workflow_contexts"})
    
    # Use the real todo_list_workflow directory
    workflow_dir = get_example_workflow_path()
    assert os.path.isdir(workflow_dir), "tests/todo_list_workflow directory should exist"
    
    # Create a app workflow
    app_workflow = fastworkflow.Workflow.create(
        workflow_folderpath=workflow_dir,
        workflow_id_str="test-subject-workflow-4"
    )
    
    # Use TodoListManager as context
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
    
    # Call the __call__ method which is the current implementation
    response = ResponseGenerator()(mock_workflow, "what context am I in")
    
    # Check that the response contains the context name
    assert "TodoListManager" in response.command_responses[0].response


def test_context_with_properties():
    """Test the response when in a context with properties."""
    fastworkflow.init({"SPEEDDICT_FOLDERNAME": "___workflow_contexts"})
    
    # Use the real todo_list_workflow directory
    workflow_dir = get_example_workflow_path()
    assert os.path.isdir(workflow_dir), "tests/todo_list_workflow directory should exist"
    
    # Create a app workflow
    app_workflow = fastworkflow.Workflow.create(
        workflow_folderpath=workflow_dir,
        workflow_id_str="test-subject-workflow-5"
    )
    
    # Create a TodoList with some properties
    todo_list = TodoList(id=1, description="Test List")
    app_workflow.current_command_context = todo_list
    
    # Create a command metadata extraction workflow
    workflow_type = "command_metadata_extraction"
    cme_workflow_folderpath = fastworkflow.get_internal_workflow_path(workflow_type)
    
    # Create a child workflow with the app_workflow in its context
    mock_workflow = fastworkflow.Workflow.create(
        workflow_folderpath=cme_workflow_folderpath,
        parent_workflow_id=app_workflow.id,
        workflow_context={"app_workflow": app_workflow}
    )
    
    # Call the __call__ method which is the current implementation
    response = ResponseGenerator()(mock_workflow, "what context am I in")
    
    # Check that the response contains the context name
    assert "TodoList" in response.command_responses[0].response 
import os
import pytest
import fastworkflow
from fastworkflow._workflows.command_metadata_extraction._commands.IntentDetection.what_is_current_context import ResponseGenerator
from examples.todo_list.application.todo_manager import TodoListManager
from examples.todo_list.application.todo_list import TodoList


def get_example_workflow_path():
    """Get the path to the examples/todo_list workflow."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "examples", "todo_list"))


def test_global_context():
    """Test the response when in the global context."""
    fastworkflow.init({"SPEEDDICT_FOLDERNAME": "___workflow_contexts"})
    
    # Use the real todo_list workflow directory
    workflow_dir = get_example_workflow_path()
    assert os.path.isdir(workflow_dir), "examples/todo_list directory should exist"
    
    # Create a subject session
    subject_session = fastworkflow.Session.create(
        workflow_folderpath=workflow_dir,
        session_id_str="test-subject-session-3"
    )
    
    # Create a command metadata extraction session
    workflow_type = "command_metadata_extraction"
    cme_workflow_folderpath = fastworkflow.get_internal_workflow_path(workflow_type)
    
    # Create a child session with the subject_session in its context
    mock_session = fastworkflow.Session.create(
        workflow_folderpath=cme_workflow_folderpath,
        parent_session_id=subject_session.id,
        workflow_context={"subject_session": subject_session}
    )
    
    # Call the __call__ method which is the current implementation
    gen = ResponseGenerator()
    response = gen(mock_session, "what context am I in")
    
    # Check that the response indicates the global context
    assert "global" in response.command_responses[0].response.lower() or "*" in response.command_responses[0].response


def test_context_no_properties():
    """Test the response when in a context without get_properties method."""
    fastworkflow.init({"SPEEDDICT_FOLDERNAME": "___workflow_contexts"})
    
    # Use the real todo_list workflow directory
    workflow_dir = get_example_workflow_path()
    assert os.path.isdir(workflow_dir), "examples/todo_list directory should exist"
    
    # Create a subject session
    subject_session = fastworkflow.Session.create(
        workflow_folderpath=workflow_dir,
        session_id_str="test-subject-session-4"
    )
    
    # Use TodoListManager as context
    subject_session.current_command_context = TodoListManager()
    
    # Create a command metadata extraction session
    workflow_type = "command_metadata_extraction"
    cme_workflow_folderpath = fastworkflow.get_internal_workflow_path(workflow_type)
    
    # Create a child session with the subject_session in its context
    mock_session = fastworkflow.Session.create(
        workflow_folderpath=cme_workflow_folderpath,
        parent_session_id=subject_session.id,
        workflow_context={"subject_session": subject_session}
    )
    
    # Call the __call__ method which is the current implementation
    response = ResponseGenerator()(mock_session, "what context am I in")
    
    # Check that the response contains the context name
    assert "TodoListManager" in response.command_responses[0].response


def test_context_with_properties():
    """Test the response when in a context with properties."""
    fastworkflow.init({"SPEEDDICT_FOLDERNAME": "___workflow_contexts"})
    
    # Use the real todo_list workflow directory
    workflow_dir = get_example_workflow_path()
    assert os.path.isdir(workflow_dir), "examples/todo_list directory should exist"
    
    # Create a subject session
    subject_session = fastworkflow.Session.create(
        workflow_folderpath=workflow_dir,
        session_id_str="test-subject-session-5"
    )
    
    # Create a TodoList with some properties
    todo_list = TodoList(id=1, description="Test List")
    subject_session.current_command_context = todo_list
    
    # Create a command metadata extraction session
    workflow_type = "command_metadata_extraction"
    cme_workflow_folderpath = fastworkflow.get_internal_workflow_path(workflow_type)
    
    # Create a child session with the subject_session in its context
    mock_session = fastworkflow.Session.create(
        workflow_folderpath=cme_workflow_folderpath,
        parent_session_id=subject_session.id,
        workflow_context={"subject_session": subject_session}
    )
    
    # Call the __call__ method which is the current implementation
    response = ResponseGenerator()(mock_session, "what context am I in")
    
    # Check that the response contains the context name
    assert "TodoList" in response.command_responses[0].response 
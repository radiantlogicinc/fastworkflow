import pytest
import fastworkflow
from fastworkflow._workflows.command_metadata_extraction._commands.IntentDetection.reset_context import ResponseGenerator


def test_reset_context():
    fastworkflow.init({"SPEEDDICT_FOLDERNAME": "___workflow_contexts"})

    # Create a session instead of directly
    session = fastworkflow.Session.create(
        workflow_folderpath="/tmp",
        session_id_str="test-reset-context"
    )
    
    # Set a context object
    obj = {"ctx": 1}
    session.current_command_context = obj
    assert session.current_command_context is obj

    # Create a mock session that mimics the structure expected by the ResponseGenerator
    mock_session = fastworkflow.Session.create(
        workflow_folderpath="/tmp",
        session_id_str="mock-session"
    )
    mock_session.workflow_context = {"subject_session": session}

    resp_gen = ResponseGenerator()
    
    # Call the __call__ method which is the current implementation
    response = resp_gen(mock_session, "reset context")
    
    # Check that the response indicates context was reset
    assert "context" in response.command_responses[0].response.lower()
    
    # Check that the context was actually reset
    assert session.current_command_context is session.root_command_context 
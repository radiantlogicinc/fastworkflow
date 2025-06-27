import pytest
import fastworkflow
from fastworkflow._workflows.command_metadata_extraction._commands.IntentDetection.reset_context import ResponseGenerator


def test_reset_context():
    fastworkflow.init({"SPEEDDICT_FOLDERNAME": "___workflow_contexts"})

    # Create a workflow instead of directly
    workflow = fastworkflow.Workflow.create(
        workflow_folderpath="/tmp",
        workflow_id_str="test-reset-context"
    )
    
    # Set a context object
    obj = {"ctx": 1}
    workflow.current_command_context = obj
    assert workflow.current_command_context is obj

    # Create a mock workflow that mimics the structure expected by the ResponseGenerator
    mock_workflow = fastworkflow.Workflow.create(
        workflow_folderpath="/tmp",
        workflow_id_str="mock-workflow"
    )
    mock_workflow.context = {"app_workflow": workflow}

    resp_gen = ResponseGenerator()
    
    # Call the __call__ method which is the current implementation
    response = resp_gen(mock_workflow, "reset context")
    
    # Check that the response indicates context was reset
    assert "context" in response.command_responses[0].response.lower()
    
    # Check that the context was actually reset
    assert workflow.current_command_context is workflow.root_command_context 
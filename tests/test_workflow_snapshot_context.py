import tempfile
import os

import fastworkflow
from fastworkflow.workflow import Workflow


def test_current_context_object_property(tmp_path):
    # sourcery skip: extract-duplicate-method
    """Verify getter and setter for current_command_context work as expected."""
    # Initialize FastWorkflow with a minimal set of env vars
    env_vars = {"SPEEDDICT_FOLDERNAME": "___workflow_contexts"}
    fastworkflow.init(env_vars=env_vars)

    # Get the path to the hello_world example directory
    hello_world_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples", "hello_world")
    
    # Create a workflow with the hello_world workflow
    workflow = Workflow.create(
        workflow_folderpath=hello_world_path,
        workflow_id_str="test-workflow-3"
    )

    # Initially, context should be None
    assert workflow.current_command_context is None

    # Set a dummy context object and verify getter returns it
    dummy_obj = object()
    workflow.current_command_context = dummy_obj
    assert workflow.current_command_context is dummy_obj

    # Reset context to None and verify
    workflow.current_command_context = None
    assert workflow.current_command_context is None

    # Validate setting context again
    obj2 = object()
    workflow.current_command_context = obj2
    assert workflow.current_command_context is obj2

    # Reset context again
    workflow.current_command_context = None
    assert workflow.current_command_context is None 
import tempfile
import os

import fastworkflow
from fastworkflow.session import Session


def test_current_context_object_property(tmp_path):
    # sourcery skip: extract-duplicate-method
    """Verify getter and setter for current_command_context work as expected."""
    # Initialize FastWorkflow with a minimal set of env vars
    env_vars = {"SPEEDDICT_FOLDERNAME": "___workflow_contexts"}
    fastworkflow.init(env_vars=env_vars)

    # Get the path to the hello_world example directory
    hello_world_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples", "hello_world")
    
    # Create a session with the hello_world workflow
    session = Session.create(
        workflow_folderpath=hello_world_path,
        session_id_str="test-session-3"
    )

    # Initially, context should be None
    assert session.current_command_context is None

    # Set a dummy context object and verify getter returns it
    dummy_obj = object()
    session.current_command_context = dummy_obj
    assert session.current_command_context is dummy_obj

    # Reset context to None and verify
    session.current_command_context = None
    assert session.current_command_context is None

    # Validate setting context again
    obj2 = object()
    session.current_command_context = obj2
    assert session.current_command_context is obj2

    # Reset context again
    session.current_command_context = None
    assert session.current_command_context is None 
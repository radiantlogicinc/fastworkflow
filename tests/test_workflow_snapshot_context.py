import tempfile
import os

import fastworkflow
from fastworkflow.session import Session


def test_current_context_object_property(tmp_path):
    """Verify getter and setter for current_command_context work as expected."""
    # Initialize FastWorkflow with a minimal set of env vars
    env_vars = {"SPEEDDICT_FOLDERNAME": "___workflow_contexts"}
    fastworkflow.init(env_vars=env_vars)

    # Create a temporary workflow directory
    workflow_dir = tmp_path / "sample_workflow"
    workflow_dir.mkdir()

    # Create a session instead of directly using WorkflowSnapshot
    session = Session.create(
        workflow_folderpath=str(workflow_dir),
        session_id_str="test-session-123"
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
import os
import fastworkflow
from fastworkflow.session import Session


def test_context_helpers_todo_list(tmp_path):
    """Integration test for context helpers using the examples/todo_list workflow."""
    env_vars = {"SPEEDDICT_FOLDERNAME": "___workflow_contexts"}
    fastworkflow.init(env_vars=env_vars)

    # Use the real workflow directory from the repo
    workflow_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "examples", "todo_list"))
    assert os.path.isdir(workflow_dir), "examples/todo_list directory should exist"

    # Create a session instead of directly using WorkflowSnapshot
    session = Session.create(
        workflow_folderpath=workflow_dir,
        session_id_str="test-session-999"
    )

    # Basic sanity checks
    assert session.current_command_context is None

    dummy_obj = object()
    session.current_command_context = dummy_obj
    assert session.current_command_context is dummy_obj

    session.current_command_context = None
    assert session.current_command_context is None 
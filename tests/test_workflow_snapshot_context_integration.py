import os
import fastworkflow
from fastworkflow.workflow import Workflow


def test_context_helpers_todo_list(tmp_path):
    """Integration test for context helpers using the tests/todo_list_workflow."""
    env_vars = {"SPEEDDICT_FOLDERNAME": "___workflow_contexts"}
    fastworkflow.init(env_vars=env_vars)

    # Use the real workflow directory from the repo
    workflow_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tests", "todo_list_workflow"))
    assert os.path.isdir(workflow_dir), "tests/todo_list_workflow directory should exist"

    # Create a workflow instead of directly
    workflow = Workflow.create(
        workflow_folderpath=workflow_dir,
        workflow_id_str="test-workflow-999"
    )

    # Basic sanity checks
    assert workflow.current_command_context is None

    dummy_obj = object()
    workflow.current_command_context = dummy_obj
    assert workflow.current_command_context is dummy_obj

    workflow.current_command_context = None
    assert workflow.current_command_context is None 
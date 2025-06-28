import os
import fastworkflow
from fastworkflow.workflow import Workflow


def test_context_helpers_todo_list(tmp_path):
    """Integration test for context helpers using the examples/todo_list workflow."""
    env_vars = {"SPEEDDICT_FOLDERNAME": "___workflow_contexts"}
    fastworkflow.init(env_vars=env_vars)

    # Use the real workflow directory from the repo
    workflow_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "fastworkflow", "examples", "todo_list"))
    assert os.path.isdir(workflow_dir), "fastworkflow/examples/todo_list directory should exist"

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
import os
import fastworkflow
import pytest


# Define a dummy class that mimics what ContextExpander would do
class DummyExpander:
    """Simple implementation that always clears context."""

    def get_parent_command_context(self, workflow: fastworkflow.Workflow):  # noqa: D401
        """Clear the context of the provided workflow."""
        workflow.current_command_context = None


def test_workflow_has_context_methods(tmp_path):
    env_vars = {"SPEEDDICT_FOLDERNAME": "___workflow_contexts"}
    fastworkflow.init(env_vars=env_vars)

    # Get the path to the hello_world example directory
    hello_world_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples", "hello_world")
    
    # Create a workflow with the hello_world workflow
    workflow = fastworkflow.Workflow.create(
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


def test_dummy_expander(tmp_path):
    """Test that DummyExpander can clear a workflow's context."""
    env_vars = {"SPEEDDICT_FOLDERNAME": "___workflow_contexts"}
    fastworkflow.init(env_vars=env_vars)

    # Get the path to the hello_world example directory
    hello_world_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples", "hello_world")
    
    # Create a workflow with the hello_world workflow
    workflow = fastworkflow.Workflow.create(
        workflow_folderpath=hello_world_path,
        workflow_id_str="test-workflow-3"
    )

    # Set some context (simple, picklable object)
    ctx = {"foo": "bar"}
    workflow.current_command_context = ctx
    assert workflow.current_command_context is ctx

    # Use the updated DummyExpander to clear the context
    expander = DummyExpander()
    expander.get_parent_command_context(workflow)
    assert workflow.current_command_context is None


# ---------------------------------------------------------------------------
# New test: context object that implements context navigation
# ---------------------------------------------------------------------------


class ParentCtx:  # noqa: D401
    """A parent context object."""
    pass


class ChildCtx:
    """A child context that knows how to navigate to its parent."""
    def __init__(self, parent):
        self._parent = parent

    def get_parent_command_context(self, workflow: fastworkflow.Workflow):  # noqa: D401
        """Set the workflow's context to this object's parent context."""
        workflow.current_command_context = self._parent


def test_object_level_move_to_parent(tmp_path):
    """Test that a context object can implement parent navigation."""
    env_vars = {"SPEEDDICT_FOLDERNAME": "___workflow_contexts"}
    fastworkflow.init(env_vars=env_vars)

    # Get the path to the hello_world example directory
    hello_world_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples", "hello_world")
    
    # Create a workflow with the hello_world workflow
    workflow = fastworkflow.Workflow.create(
        workflow_folderpath=hello_world_path,
        workflow_id_str="test-workflow-3"
    )

    parent = ParentCtx()
    child = ChildCtx(parent)

    # Set the context using the workflow
    workflow.current_command_context = child
    assert workflow.current_command_context is child

    # Use the child's get_parent_command_context method to navigate to parent
    child.get_parent_command_context(workflow)
    assert workflow.current_command_context is parent 
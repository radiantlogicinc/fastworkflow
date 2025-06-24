import os
import fastworkflow
from fastworkflow.session import WorkflowSnapshot, Session
import pytest


# Define a dummy class that mimics what ContextExpander would do
class DummyExpander:
    """Simple implementation that always clears context."""

    def get_parent_command_context(self, session: Session):  # noqa: D401
        """Clear the context of the provided session."""
        session.current_command_context = None


def test_workflow_snapshot_has_context_methods(tmp_path):
    env_vars = {"SPEEDDICT_FOLDERNAME": "___workflow_contexts"}
    fastworkflow.init(env_vars=env_vars)

    # Get the path to the hello_world example directory
    hello_world_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples", "hello_world")
    
    # Create a session with the hello_world workflow
    session = Session.create(
        workflow_folderpath=hello_world_path,
        session_id_str="test-session-3"
    )
    snapshot = session.workflow_snapshot

    # Check that we can set and get context via the session
    # Skip if set_context is available on the snapshot (old implementation)
    if hasattr(snapshot, "set_context"):
        pytest.skip("Using old implementation with set_context on WorkflowSnapshot")
    
    # Initially, context should be None
    assert session.current_command_context is None

    # Set a dummy context object and verify getter returns it
    dummy_obj = object()
    session.current_command_context = dummy_obj
    assert session.current_command_context is dummy_obj

    # Reset context to None and verify
    session.current_command_context = None
    assert session.current_command_context is None


def test_dummy_expander(tmp_path):
    """Test that DummyExpander can clear a session's context."""
    env_vars = {"SPEEDDICT_FOLDERNAME": "___workflow_contexts"}
    fastworkflow.init(env_vars=env_vars)

    # Get the path to the hello_world example directory
    hello_world_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples", "hello_world")
    
    # Create a session with the hello_world workflow
    session = Session.create(
        workflow_folderpath=hello_world_path,
        session_id_str="test-session-3"
    )

    # Set some context (simple, picklable object)
    ctx = {"foo": "bar"}
    session.current_command_context = ctx
    assert session.current_command_context is ctx

    # Use the updated DummyExpander to clear the context
    expander = DummyExpander()
    expander.get_parent_command_context(session)
    assert session.current_command_context is None


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

    def get_parent_command_context(self, session: Session):  # noqa: D401
        """Set the session's context to this object's parent context."""
        session.current_command_context = self._parent


def test_object_level_move_to_parent(tmp_path):
    """Test that a context object can implement parent navigation."""
    env_vars = {"SPEEDDICT_FOLDERNAME": "___workflow_contexts"}
    fastworkflow.init(env_vars=env_vars)

    # Get the path to the hello_world example directory
    hello_world_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples", "hello_world")
    
    # Create a session with the hello_world workflow
    session = Session.create(
        workflow_folderpath=hello_world_path,
        session_id_str="test-session-3"
    )

    parent = ParentCtx()
    child = ChildCtx(parent)

    # Set the context using the session
    session.current_command_context = child
    assert session.current_command_context is child

    # Use the child's get_parent_command_context method to navigate to parent
    child.get_parent_command_context(session)
    assert session.current_command_context is parent 
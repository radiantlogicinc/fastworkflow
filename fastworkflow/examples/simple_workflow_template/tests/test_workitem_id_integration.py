"""Integration tests for WorkItem ID functionality.

This module contains pytest tests for integrating the new ID functionality with the existing WorkItem class.
"""

import pytest
from fastworkflow._workflows.simple_workflow_template.application.workitem import WorkItem


class TestWorkItemId:
    """Test cases for WorkItem ID functionality."""

    def test_workitem_init_with_id(self):
        """Test WorkItem initialization with ID."""
        workitem = WorkItem(workitem_type="Task", id="task-123")
        assert workitem.id == "task-123"
        assert workitem._id == "task-123"

    def test_unique_ids_allowed(self):
        """Test that unique IDs are allowed among siblings."""
        parent = WorkItem(workitem_type="Project")
        child1 = WorkItem(workitem_type="Task", id="task-1")
        child2 = WorkItem(workitem_type="Task", id="task-2")
        
        parent.add_child(child1)
        parent.add_child(child2)
        
        assert parent.get_child_count() == 2
        assert parent.get_child(0).id == "task-1"
        assert parent.get_child(1).id == "task-2"

    def test_duplicate_id_raises_error(self):
        """Test that duplicate IDs raise ValueError."""
        parent = WorkItem(workitem_type="Project")
        child1 = WorkItem(workitem_type="Task", id="task-1")
        
        parent.add_child(child1)
        
        with pytest.raises(ValueError, match="Work item id 'task-1' is not unique among siblings of type 'Task'"):
            child2 = WorkItem(workitem_type="Task", id="task-1", parent=parent)

    def test_get_absolute_path_with_id(self):
        """Test getting absolute path for workitem with ID."""
        root = WorkItem(workitem_type="Project")
        child = WorkItem(workitem_type="Task", id="task-123")
        root.add_child(child)
        
        assert child.get_absolute_path() == "/Task[id=task-123]"

    def test_get_child_workitem_by_id_path(self):
        """Test getting child workitem by path with ID."""
        root = WorkItem(workitem_type="Project")
        task = WorkItem(workitem_type="Task", id="task-123")
        
        root.add_child(task)
        
        assert root._get_child_workitem("Task[id=task-123]") == task
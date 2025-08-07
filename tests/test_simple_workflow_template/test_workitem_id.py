"""Test suite for WorkItem ID functionality.

This module contains pytest tests for the new ID functionality in the WorkItem class, including:
- WorkItem initialization with ID
- ID uniqueness validation
- Path operations with IDs
- Serialization/deserialization with IDs
"""

import pytest
import json
from fastworkflow.examples.simple_workflow_template.application.workitem import WorkItem


class TestWorkItemIdBasics:
    """Test cases for basic WorkItem ID functionality."""

    def test_workitem_init_with_id(self):
        """Test WorkItem initialization with ID."""
        workitem = WorkItem(workitem_type="Task", id="task-123")
        assert workitem.id == "task-123"
        assert workitem._id == "task-123"

    def test_workitem_init_without_id(self):
        """Test WorkItem initialization without ID."""
        workitem = WorkItem(workitem_type="Task")
        assert workitem.id is None
        assert workitem._id is None

    def test_workitem_id_property(self):
        """Test WorkItem id property getter."""
        workitem = WorkItem(workitem_type="Task", id="task-123")
        assert workitem.id == "task-123"


class TestWorkItemIdUniqueness:
    """Test cases for WorkItem ID uniqueness validation."""

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

    def test_duplicate_id_in_add_child(self):
        """Test that adding a child with duplicate ID raises ValueError."""
        parent = WorkItem(workitem_type="Project")
        child1 = WorkItem(workitem_type="Task", id="task-1")
        child2 = WorkItem(workitem_type="Task", id="task-1")
        
        parent.add_child(child1)
        
        with pytest.raises(ValueError, match="Work item id 'task-1' is not unique among siblings of type 'Task'"):
            parent.add_child(child2)

    def test_duplicate_id_different_types_allowed(self):
        """Test that duplicate IDs are allowed among siblings of different types."""
        parent = WorkItem(workitem_type="Project")
        task = WorkItem(workitem_type="Task", id="item-1")
        bug = WorkItem(workitem_type="Bug", id="item-1")
        
        parent.add_child(task)
        parent.add_child(bug)
        
        assert parent.get_child_count() == 2
        assert parent.get_child(0, "Task").id == "item-1"
        assert parent.get_child(0, "Bug").id == "item-1"


class TestWorkItemIdPathOperations:
    """Test cases for WorkItem path operations with IDs."""

    def test_get_absolute_path_with_id(self):
        """Test getting absolute path for workitem with ID."""
        root = WorkItem(workitem_type="Project")
        child = WorkItem(workitem_type="Task", id="task-123")
        root.add_child(child)
        
        assert child.get_absolute_path() == "/Task[id=task-123]"

    def test_get_absolute_path_mixed_with_and_without_id(self):
        """Test getting absolute path with mixed ID and non-ID workitems."""
        root = WorkItem(workitem_type="Project")
        task_with_id = WorkItem(workitem_type="Task", id="task-123")
        subtask_without_id = WorkItem(workitem_type="Subtask")
        
        root.add_child(task_with_id)
        task_with_id.add_child(subtask_without_id)
        
        assert subtask_without_id.get_absolute_path() == "/Task[id=task-123]/Subtask[index=0]"

    def test_get_child_workitem_by_id_path(self):
        """Test getting child workitem by path with ID."""
        root = WorkItem(workitem_type="Project")
        task = WorkItem(workitem_type="Task", id="task-123")
        subtask = WorkItem(workitem_type="Subtask", id="subtask-456")
        
        root.add_child(task)
        task.add_child(subtask)
        
        # Test various path formats with IDs
        assert root._get_child_workitem("Task[id=task-123]") == task
        assert root._get_child_workitem("Task[id=task-123]/Subtask[id=subtask-456]") == subtask

    def test_get_child_workitem_by_mixed_path(self):
        """Test getting child workitem by mixed path with ID and index."""
        root = WorkItem(workitem_type="Project")
        task1 = WorkItem(workitem_type="Task", id="task-123")
        task2 = WorkItem(workitem_type="Task")  # No ID
        subtask1 = WorkItem(workitem_type="Subtask", id="subtask-456")
        subtask2 = WorkItem(workitem_type="Subtask")  # No ID
        
        root.add_child(task1)
        root.add_child(task2)
        task1.add_child(subtask1)
        task1.add_child(subtask2)
        
        # Test mixed path formats
        assert root._get_child_workitem("Task[id=task-123]/Subtask[index=1]") == subtask2
        assert root._get_child_workitem("Task[index=1]/Subtask[index=0]") is None  # task2 has no subtasks

    def test_get_workitem_by_absolute_path_with_id(self):
        """Test getting workitem by absolute path with ID."""
        root = WorkItem(workitem_type="Project")
        task = WorkItem(workitem_type="Task", id="task-123")
        subtask = WorkItem(workitem_type="Subtask", id="subtask-456")
        
        root.add_child(task)
        task.add_child(subtask)
        
        # Test from different nodes
        assert root.get_workitem("/Task[id=task-123]") == task
        assert task.get_workitem("/Task[id=task-123]/Subtask[id=subtask-456]") == subtask
        assert subtask.get_workitem("/Task[id=task-123]") == task

    def test_get_workitem_invalid_id_path(self):
        """Test getting workitem with invalid ID in path."""
        root = WorkItem(workitem_type="Project")
        task = WorkItem(workitem_type="Task", id="task-123")
        root.add_child(task)
        
        assert root.get_workitem("/Task[id=non-existent]") is None


class TestWorkItemIdSerialization:
    """Test cases for WorkItem serialization with IDs."""

    def test_to_dict_with_id(self):
        """Test WorkItem serialization to dictionary with ID."""
        workitem = WorkItem(workitem_type="Task", id="task-123")
        workitem["title"] = "Test Task"
        
        result = workitem._to_dict()
        
        assert result["workitem_type"] == "Task"
        assert result["id"] == "task-123"
        assert result["data_dict"]["title"] == "Test Task"

    def test_from_dict_with_id(self):
        """Test WorkItem deserialization from dictionary with ID."""
        data = {
            'workitem_type': 'Task',
            'is_complete': False,
            'data_dict': {'title': 'Test Task'},
            'id': 'task-123',
            'children': []
        }
        
        workitem = WorkItem._from_dict(data)
        assert workitem.type == 'Task'
        assert workitem.id == 'task-123'
        assert workitem["title"] == 'Test Task'

    def test_serialization_roundtrip_with_id(self):
        """Test serialization roundtrip with ID."""
        # Create a workitem with ID
        original = WorkItem(workitem_type="Task", id="task-123")
        original["title"] = "Test Task"
        
        # Add a child with ID
        child = WorkItem(workitem_type="Subtask", id="subtask-456")
        child["description"] = "Test Subtask"
        child.is_complete = True  # Set child to complete first
        original.add_child(child)
        
        # Now set parent to complete (will work because child is complete)
        original.is_complete = True
        
        # Serialize and deserialize
        serialized = original._to_dict()
        deserialized = WorkItem._from_dict(serialized)
        
        # Verify properties
        assert deserialized.type == "Task"
        assert deserialized.id == "task-123"
        assert deserialized["title"] == "Test Task"
        assert deserialized.is_complete is True
        assert deserialized.get_child_count() == 1
        
        # Verify child properties
        child_deserialized = deserialized.get_child(0)
        assert child_deserialized.type == "Subtask"
        assert child_deserialized.id == "subtask-456"
        assert child_deserialized["description"] == "Test Subtask"
"""Comprehensive test suite for WorkItem class.

This module contains pytest tests for the WorkItem class, including tests for:
- WorkItem initialization and basic properties
- ChildSchema functionality
- WorkflowSchema functionality  
- Child management with cardinality constraints
- Path operations and navigation
- Data dictionary operations
- Serialization/deserialization
- Error handling and edge cases
"""

import pytest
import json
import tempfile
import os
from typing import Dict, Any
from fastworkflow._workflows.simple_workflow_template.application.workitem import WorkItem


class TestChildSchema:
    """Test cases for WorkItem.ChildSchema class."""

    def test_child_schema_init_valid(self):
        """Test valid ChildSchema initialization."""
        schema = WorkItem.ChildSchema("TaskItem", 1, 5)
        assert schema.child_workitem_type == "TaskItem"
        assert schema.min_cardinality == 1
        assert schema.max_cardinality == 5

    def test_child_schema_init_defaults(self):
        """Test ChildSchema initialization with default values."""
        schema = WorkItem.ChildSchema("TaskItem")
        assert schema.child_workitem_type == "TaskItem"
        assert schema.min_cardinality == 0
        assert schema.max_cardinality is None

    def test_child_schema_init_empty_type(self):
        """Test ChildSchema initialization with empty workitem type."""
        with pytest.raises(ValueError, match="child_workitem_type cannot be empty or contain spaces"):
            WorkItem.ChildSchema("")

    def test_child_schema_init_type_with_spaces(self):
        """Test ChildSchema initialization with spaces in workitem type."""
        with pytest.raises(ValueError, match="child_workitem_type cannot be empty or contain spaces"):
            WorkItem.ChildSchema("Task Item")

    def test_child_schema_init_negative_min_cardinality(self):
        """Test ChildSchema initialization with negative min_cardinality."""
        with pytest.raises(ValueError, match="min_cardinality cannot be negative"):
            WorkItem.ChildSchema("TaskItem", -1)

    def test_child_schema_init_negative_max_cardinality(self):
        """Test ChildSchema initialization with negative max_cardinality."""
        with pytest.raises(ValueError, match="max_cardinality cannot be negative"):
            WorkItem.ChildSchema("TaskItem", 0, -1)

    def test_child_schema_init_min_greater_than_max(self):
        """Test ChildSchema initialization with min > max cardinality."""
        with pytest.raises(ValueError, match="min_cardinality cannot be greater than max_cardinality"):
            WorkItem.ChildSchema("TaskItem", 5, 3)

    def test_child_schema_to_dict(self):
        """Test ChildSchema serialization to dictionary."""
        schema = WorkItem.ChildSchema("TaskItem", 1, 5)
        expected = {
            "child_workitem_type": "TaskItem",
            "min_cardinality": 1,
            "max_cardinality": 5
        }
        assert schema.to_dict() == expected

    def test_child_schema_from_dict(self):
        """Test ChildSchema deserialization from dictionary."""
        data = {
            "child_workitem_type": "TaskItem",
            "min_cardinality": 1,
            "max_cardinality": 5
        }
        schema = WorkItem.ChildSchema.from_dict(data)
        assert schema.child_workitem_type == "TaskItem"
        assert schema.min_cardinality == 1
        assert schema.max_cardinality == 5

    def test_child_schema_from_dict_defaults(self):
        """Test ChildSchema deserialization with default values."""
        data = {"child_workitem_type": "TaskItem"}
        schema = WorkItem.ChildSchema.from_dict(data)
        assert schema.child_workitem_type == "TaskItem"
        assert schema.min_cardinality == 0
        assert schema.max_cardinality is None


class TestWorkflowSchema:
    """Test cases for WorkItem.WorkflowSchema class."""

    def test_workflow_schema_init_minimal(self):
        """Test WorkflowSchema initialization with minimal parameters."""
        schema = WorkItem.WorkflowSchema()
        assert schema.workflow_types is None
        assert schema.child_schema_dict is None

    def test_workflow_schema_init_with_types_dict(self):
        """Test WorkflowSchema initialization with workflow types as dictionary."""
        types_dict = {
            "Project": "A project that contains multiple tasks",
            "Task": "A specific task to be completed",
            "Subtask": "A smaller task within a larger task"
        }
        schema = WorkItem.WorkflowSchema(workflow_types=types_dict)
        assert schema.workflow_types == types_dict
        assert schema.child_schema_dict is None

    def test_workflow_schema_init_with_types_list(self):
        """Test WorkflowSchema initialization with workflow types as list (legacy)."""
        types_dict = {"Project": "desc", "Task": "desc", "Subtask": "desc"}
        schema = WorkItem.WorkflowSchema(workflow_types=types_dict)
        assert schema.workflow_types == types_dict
        assert schema.child_schema_dict is None

    def test_workflow_schema_init_with_child_schemas(self):
        """Test WorkflowSchema initialization with child schemas."""
        types_dict = {
            "Project": "A project that contains multiple tasks",
            "Task": "A specific task to be completed", 
            "Subtask": "A smaller task within a larger task"
        }
        child_schemas = {
            "Project": {"Task": {"min_cardinality": 1, "max_cardinality": 10}},
            "Task": {"Subtask": {"min_cardinality": 0, "max_cardinality": 5}},
            "Subtask": None
        }
        schema = WorkItem.WorkflowSchema(workflow_types=types_dict, child_schema_dict=child_schemas)
        assert schema.workflow_types == types_dict
        assert schema.child_schema_dict == child_schemas

    def test_workflow_schema_init_missing_workflow_type_in_child_schema(self):
        """Test WorkflowSchema initialization with child schema key not in workflow types."""
        types_dict = {
            "Project": "A project that contains multiple tasks",
            "Task": "A specific task to be completed"
        }
        child_schemas = {
            "Project": {"Task": {"min_cardinality": 1, "max_cardinality": 10}},
            "Subtask": {"Task": {"min_cardinality": 0, "max_cardinality": 5}}  # Subtask not in types
        }
        with pytest.raises(ValueError, match="Child schema key 'Subtask' not found in workflow_types"):
            WorkItem.WorkflowSchema(workflow_types=types_dict, child_schema_dict=child_schemas)

    def test_workflow_schema_init_invalid_child_workitem_type(self):
        """Test WorkflowSchema initialization with invalid child workitem type."""
        types_dict = {
            "Project": "A project that contains multiple tasks",
            "Task": "A specific task to be completed"
        }
        child_schemas = {
            "Project": {"InvalidType": {"min_cardinality": 1, "max_cardinality": 10}}  # InvalidType not in types
        }
        with pytest.raises(ValueError, match="Child workitem type 'InvalidType' not found in workflow_types"):
            WorkItem.WorkflowSchema(workflow_types=types_dict, child_schema_dict=child_schemas)

    def test_workflow_schema_auto_add_missing_types(self):
        """Test WorkflowSchema automatically adds missing workflow types to child_schema_dict."""
        types_dict = {
            "Project": "A project that contains multiple tasks",
            "Task": "A specific task to be completed",
            "Subtask": "A smaller task within a larger task"
        }
        child_schemas = {
            "Project": {"Task": {"min_cardinality": 1, "max_cardinality": 10}}
        }
        schema = WorkItem.WorkflowSchema(workflow_types=types_dict, child_schema_dict=child_schemas)
        assert "Task" in schema.child_schema_dict
        assert "Subtask" in schema.child_schema_dict
        assert schema.child_schema_dict["Task"] is None
        assert schema.child_schema_dict["Subtask"] is None

    def test_workflow_schema_to_dict(self):
        """Test WorkflowSchema serialization to dictionary."""
        types_dict = {
            "Project": "A project that contains multiple tasks",
            "Task": "A specific task to be completed"
        }
        child_schemas = {
            "Project": {"Task": {"min_cardinality": 1, "max_cardinality": 5}},
            "Task": None
        }
        schema = WorkItem.WorkflowSchema(workflow_types=types_dict, child_schema_dict=child_schemas)
        result = schema.to_dict()
        
        expected = {
            "workflow_types": types_dict,
            "child_schema_dict": {
                "Project": {"Task": {"min_cardinality": 1, "max_cardinality": 5}},
                "Task": None
            },
            "context_hierarchy_model": {}
        }
        assert result == expected

    def test_workflow_schema_from_dict(self):
        """Test WorkflowSchema deserialization from dictionary."""
        data = {
            "workflow_types": {
                "Project": "A project that contains multiple tasks",
                "Task": "A specific task to be completed"
            },
            "child_schema_dict": {
                "Project": {"Task": {"min_cardinality": 1, "max_cardinality": 5}},
                "Task": None
            }
        }
        schema = WorkItem.WorkflowSchema.from_dict(data)
        assert schema.workflow_types == data["workflow_types"]
        assert "Project" in schema.child_schema_dict
        assert "Task" in schema.child_schema_dict
        assert "Task" in schema.child_schema_dict["Project"]
        assert schema.child_schema_dict["Project"]["Task"]["min_cardinality"] == 1
        assert schema.child_schema_dict["Project"]["Task"]["max_cardinality"] == 5
        assert schema.child_schema_dict["Task"] is None

    def test_workflow_schema_create_workitem_valid(self):
        """Test WorkflowSchema.create_workitem with valid type."""
        types_dict = {"Project": "desc", "Task": "desc"}
        schema = WorkItem.WorkflowSchema(workflow_types=types_dict)
        workitem = schema.create_workitem("Project")
        assert workitem.type == "Project"
        assert workitem._workflow_schema == schema

    def test_workflow_schema_create_workitem_invalid_type(self):
        """Test WorkflowSchema.create_workitem with invalid type."""
        types_dict = {"Project": "desc", "Task": "desc"}
        schema = WorkItem.WorkflowSchema(workflow_types=types_dict)
        with pytest.raises(ValueError, match="workitem_type 'InvalidType' not found in workflow_types"):
            schema.create_workitem("InvalidType")

    def test_workflow_schema_create_workitem_no_types_defined(self):
        """Test WorkflowSchema.create_workitem when no workflow types are defined."""
        schema = WorkItem.WorkflowSchema()
        workitem = schema.create_workitem("AnyType")
        assert workitem.type == "AnyType"

    def test_workflow_schema_json_file_operations(self):
        """Test WorkflowSchema JSON file save/load operations."""
        types_dict = {"Project": "desc", "Task": "desc"}
        child_schemas = {
            "Project": {"Task": {"min_cardinality": 1, "max_cardinality": 5}},
            "Task": None
        }
        schema = WorkItem.WorkflowSchema(workflow_types=types_dict, child_schema_dict=child_schemas)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save to file
            schema.to_json_file(temp_path)
            
            # Load from file
            loaded_schema = WorkItem.WorkflowSchema.from_json_file(temp_path)
            
            # Verify loaded schema
            assert loaded_schema.workflow_types == types_dict
            assert "Project" in loaded_schema.child_schema_dict
            assert "Task" in loaded_schema.child_schema_dict["Project"]
            assert loaded_schema.child_schema_dict["Project"]["Task"]["min_cardinality"] == 1
            assert loaded_schema.child_schema_dict["Project"]["Task"]["max_cardinality"] == 5
        finally:
            os.unlink(temp_path)

    def test_workflow_schema_getitem(self):
        """Test WorkflowSchema __getitem__ method."""
        types_dict = {
            "Epic": "A very long task with lots of stories and bugs",
            "Story": "A story is a user story that is part of an epic",
            "Task": "A task is a subtask of a story",
            "Bug": "A bug is a bug that is part of a story",
            "Subtask": "A subtask is a subtask of a task"
        }
        child_schemas = {
            "Epic": {
                "Story": {"min_cardinality": 1, "max_cardinality": 10},
                "Bug": {"min_cardinality": 0, "max_cardinality": 5}
            },
            "Story": {
                "Task": {"min_cardinality": 1, "max_cardinality": 20},
                "Bug": {"min_cardinality": 0, "max_cardinality": 3}
            },
            "Task": {
                "Subtask": {"min_cardinality": 0, "max_cardinality": 5}
            },
            "Bug": None,
            "Subtask": None
        }
        schema = WorkItem.WorkflowSchema(workflow_types=types_dict, child_schema_dict=child_schemas)
        
        # Test getting Epic workflow type
        epic_info = schema["Epic"]
        assert epic_info["description"] == "A very long task with lots of stories and bugs"
        assert "Story" in epic_info["child_schema"]
        assert "Bug" in epic_info["child_schema"]
        assert epic_info["child_schema"]["Story"]["min_cardinality"] == 1
        assert epic_info["child_schema"]["Story"]["max_cardinality"] == 10
        assert epic_info["child_schema"]["Bug"]["min_cardinality"] == 0
        assert epic_info["child_schema"]["Bug"]["max_cardinality"] == 5
        
        # Test getting Story workflow type
        story_info = schema["Story"]
        assert story_info["description"] == "A story is a user story that is part of an epic"
        assert "Task" in story_info["child_schema"]
        assert "Bug" in story_info["child_schema"]
        assert story_info["child_schema"]["Task"]["min_cardinality"] == 1
        assert story_info["child_schema"]["Task"]["max_cardinality"] == 20
        
        # Test getting Bug workflow type (no children)
        bug_info = schema["Bug"]
        assert bug_info["description"] == "A bug is a bug that is part of a story"
        assert bug_info["child_schema"] == {}

    def test_workflow_schema_context_hierarchy_model_validation(self):
        """Test WorkflowSchema context_hierarchy_model validation."""
        types_dict = {"Epic": "desc", "Story": "desc", "Task": "desc"}
        
        # Valid context_hierarchy_model
        context_model = {
            "Epic": None,  # Root-only
            "Story": {"parent": ["Epic"]},  # Must have Epic parent
            "Task": {"parent": ["Story"]}   # Must have Story parent
        }
        schema = WorkItem.WorkflowSchema(
            workflow_types=types_dict, 
            parents_dict=context_model
        )
        assert schema.parents_dict == context_model
        
        # Invalid: key not in workflow_types
        invalid_context_model = {
            "Epic": None,
            "InvalidType": {"parent": ["Epic"]}
        }
        with pytest.raises(ValueError, match="Context hierarchy key 'InvalidType' not found in workflow_types"):
            WorkItem.WorkflowSchema(
                workflow_types=types_dict, 
                parents_dict=invalid_context_model
            )
        
        # Invalid: parent type not in workflow_types
        invalid_parent_model = {
            "Epic": None,
            "Story": {"parent": ["InvalidParent"]}
        }
        with pytest.raises(ValueError, match="Parent type 'InvalidParent' for 'Story' not found in workflow_types"):
            WorkItem.WorkflowSchema(
                workflow_types=types_dict, 
                parents_dict=invalid_parent_model
            )

    def test_workflow_schema_context_hierarchy_model_serialization(self):
        """Test WorkflowSchema context_hierarchy_model serialization."""
        types_dict = {"Epic": "desc", "Story": "desc", "Task": "desc"}
        context_model = {
            "Epic": None,
            "Story": {"parent": ["Epic"]},
            "Task": {"parent": ["Story"]}
        }
        schema = WorkItem.WorkflowSchema(
            workflow_types=types_dict, 
            parents_dict=context_model
        )
        
        # Test to_dict
        schema_dict = schema.to_dict()
        assert "context_hierarchy_model" in schema_dict
        assert schema_dict["context_hierarchy_model"] == context_model
        
        # Test from_dict
        loaded_schema = WorkItem.WorkflowSchema.from_dict(schema_dict)
        assert loaded_schema.parents_dict == context_model

    def test_workflow_schema_context_hierarchy_model_json_file_operations(self):
        """Test WorkflowSchema context_hierarchy_model JSON file save/load operations."""
        types_dict = {"Epic": "desc", "Story": "desc", "Task": "desc"}
        context_model = {
            "Epic": None,
            "Story": {"parent": ["Epic"]},
            "Task": {"parent": ["Story"]}
        }
        schema = WorkItem.WorkflowSchema(
            workflow_types=types_dict, 
            parents_dict=context_model
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save to file
            schema.to_json_file(temp_path)
            
            # Load from file
            loaded_schema = WorkItem.WorkflowSchema.from_json_file(temp_path)
            
            # Verify loaded schema
            assert loaded_schema.parents_dict == context_model
        finally:
            os.unlink(temp_path)

    def test_create_workitem_with_context_hierarchy_model_root_only(self):
        """Test create_workitem with context_hierarchy_model - root-only type."""
        types_dict = {"Epic": "desc", "Story": "desc"}
        context_model = {
            "Epic": None,  # Root-only
            "Story": {"parent": ["Epic"]}
        }
        schema = WorkItem.WorkflowSchema(
            workflow_types=types_dict, 
            parents_dict=context_model
        )
        
        # Epic should be created without parent (root-only)
        epic = schema.create_workitem("Epic")
        assert epic.type == "Epic"
        assert epic.parent is None
        
        # Epic should not accept parent
        with pytest.raises(ValueError, match="Epic is root-level only; parent given."):
            schema.create_workitem("Epic", parent=epic)

    def test_create_workitem_with_context_hierarchy_model_parent_required(self):
        """Test create_workitem with context_hierarchy_model - parent required."""
        types_dict = {"Epic": "desc", "Story": "desc", "Task": "desc"}
        context_model = {
            "Epic": None,
            "Story": {"parent": ["Epic"]},
            "Task": {"parent": ["Story"]}
        }
        schema = WorkItem.WorkflowSchema(
            workflow_types=types_dict, 
            parents_dict=context_model
        )
        
        # Create Epic first
        epic = schema.create_workitem("Epic")
        
        # Story requires Epic parent
        story = schema.create_workitem("Story", parent=epic)
        assert story.type == "Story"
        assert story.parent == epic
        
        # Story should not be created without parent
        with pytest.raises(ValueError, match="Story requires a parent of type \\['Epic'\\]."):
            schema.create_workitem("Story")
        
        # Task requires Story parent, not Epic
        with pytest.raises(ValueError, match="Invalid parent type 'Epic' for Task; allowed: \\['Story'\\]."):
            schema.create_workitem("Task", parent=epic)
        
        # Task with correct parent should work
        task = schema.create_workitem("Task", parent=story)
        assert task.type == "Task"
        assert task.parent == story

    def test_create_workitem_with_context_hierarchy_model_different_schema(self):
        """Test create_workitem with parent from different schema."""
        types_dict1 = {"Epic": "desc", "Story": "desc"}
        types_dict2 = {"Epic": "desc", "Story": "desc"}
        context_model = {
            "Epic": None,
            "Story": {"parent": ["Epic"]}
        }
        schema1 = WorkItem.WorkflowSchema(
            workflow_types=types_dict1, 
            parents_dict=context_model
        )
        schema2 = WorkItem.WorkflowSchema(
            workflow_types=types_dict2, 
            parents_dict=context_model
        )
        
        # Create Epic with schema1
        epic = schema1.create_workitem("Epic")
        
        # Try to create Story with schema2 but parent from schema1
        with pytest.raises(ValueError, match="Parent belongs to a different WorkflowSchema."):
            schema2.create_workitem("Story", parent=epic)

    def test_create_workitem_without_context_hierarchy_model_backwards_compatibility(self):
        """Test create_workitem without context_hierarchy_model for backwards compatibility."""
        types_dict = {"Epic": "desc", "Story": "desc"}
        schema = WorkItem.WorkflowSchema(workflow_types=types_dict)
        
        # Should work without parent (root)
        epic = schema.create_workitem("Epic")
        assert epic.type == "Epic"
        assert epic.parent is None
        
        # Should work with parent (no restrictions)
        story = schema.create_workitem("Story", parent=epic)
        assert story.type == "Story"
        assert story.parent == epic
        
        # Test getting non-existent workflow type
        with pytest.raises(KeyError, match="Workflow type 'NonExistent' not found in schema"):
            _ = schema["NonExistent"]
        
        # Test with schema that has no workflow_types
        empty_schema = WorkItem.WorkflowSchema()
        with pytest.raises(KeyError, match="Workflow type 'AnyType' not found in schema"):
            _ = empty_schema["AnyType"]


class TestWorkItemBasics:
    """Test cases for basic WorkItem functionality."""

    def test_workitem_init_minimal(self):
        """Test WorkItem initialization with minimal parameters."""
        workitem = WorkItem()
        assert workitem.type == "WorkItem"  # Should use class name as default
        assert not workitem.is_complete
        assert workitem.parent is None
        assert workitem._workflow_schema is None

    def test_workitem_init_with_type(self):
        """Test WorkItem initialization with workitem type."""
        workitem = WorkItem(workitem_type="Task")
        assert workitem.type == "Task"
        assert not workitem.is_complete

    def test_workitem_init_with_spaces_in_type(self):
        """Test WorkItem initialization with spaces in type should raise error."""
        # Note: There's a bug in the original code - it checks ' ' in type instead of workitem_type
        # The test reflects the intended behavior
        with pytest.raises(ValueError, match="workitem_type cannot contain spaces"):
            WorkItem(workitem_type="Task Item")

    def test_workitem_init_with_parent(self):
        """Test WorkItem initialization with parent."""
        parent = WorkItem(workitem_type="Project")
        child = WorkItem(workitem_type="Task", parent=parent)
        assert child.parent == parent

    def test_workitem_init_with_data_dict(self):
        """Test WorkItem initialization with data dictionary."""
        data = {"title": "Test Task", "priority": "High"}
        workitem = WorkItem(workitem_type="Task", data_dict=data)
        assert workitem["title"] == "Test Task"
        assert workitem["priority"] == "High"

    def test_workitem_properties(self):
        """Test WorkItem property getters and setters."""
        workitem = WorkItem(workitem_type="Task")
        
        # Test is_complete property
        assert not workitem.is_complete
        workitem.is_complete = True
        assert workitem.is_complete
        workitem.is_complete = False
        assert not workitem.is_complete

    def test_workitem_data_dict_operations(self):
        """Test WorkItem data dictionary operations."""
        workitem = WorkItem(workitem_type="Task")
        
        # Test setting and getting values
        workitem["title"] = "Test Task"
        workitem["priority"] = "High"
        assert workitem["title"] == "Test Task"
        assert workitem["priority"] == "High"
        
        # Test KeyError for non-existent key
        with pytest.raises(KeyError):
            _ = workitem["non_existent"]
        
        # Test clear_data_dict
        workitem.clear_data_dict()
        with pytest.raises(KeyError):
            _ = workitem["title"]


class TestWorkItemChildManagement:
    """Test cases for WorkItem child management functionality."""

    def test_add_child_no_schema(self):
        """Test adding child when no workflow schema is defined."""
        parent = WorkItem(workitem_type="Project")
        child = WorkItem(workitem_type="Task")
        
        parent.add_child(child)
        assert child.parent == parent
        assert parent.get_child_count() == 1
        assert parent.get_child(0) == child

    def test_add_child_with_schema_valid(self):
        """Test adding child with valid schema constraints."""
        schema = WorkItem.WorkflowSchema(
            workflow_types={"Project": "desc", "Task": "desc"},
            child_schema_dict={
                "Project": {"Task": {"min_cardinality": 0, "max_cardinality": 5}},
                "Task": None
            }
        )
        parent = WorkItem(workitem_type="Project", workflow_schema=schema)
        child = WorkItem(workitem_type="Task")
        parent.add_child(child)
        assert parent.get_child_count() == 1

    def test_add_child_invalid_type(self):
        """Test adding child with invalid type according to schema."""
        schema = WorkItem.WorkflowSchema(
            workflow_types={"Project": "desc", "Task": "desc"},
            child_schema_dict={
                "Project": {"Task": {"min_cardinality": 0, "max_cardinality": 5}},
                "Task": None
            }
        )
        parent = WorkItem(workitem_type="Project", workflow_schema=schema)
        child = WorkItem(workitem_type="InvalidType")
        with pytest.raises(ValueError, match="Child type 'InvalidType' not allowed for parent type 'Project'"):
            parent.add_child(child)

    def test_add_child_exceeds_max_cardinality(self):
        """Test adding child that exceeds max cardinality."""
        schema = WorkItem.WorkflowSchema(
            workflow_types={"Project": "desc", "Task": "desc"},
            child_schema_dict={
                "Project": {"Task": {"min_cardinality": 0, "max_cardinality": 2}},
                "Task": None
            }
        )
        parent = WorkItem(workitem_type="Project", workflow_schema=schema)
        # Add maximum allowed children
        parent.add_child(WorkItem(workitem_type="Task"))
        parent.add_child(WorkItem(workitem_type="Task"))
        # Try to add one more - should fail
        with pytest.raises(ValueError, match="Cannot add more than 2 children of type 'Task'"):
            parent.add_child(WorkItem(workitem_type="Task"))

    def test_remove_child_no_schema(self):
        """Test removing child when no schema is defined."""
        parent = WorkItem(workitem_type="Project")
        child = WorkItem(workitem_type="Task")
        
        parent.add_child(child)
        assert parent.get_child_count() == 1
        
        parent.remove_child(child)
        assert parent.get_child_count() == 0

    def test_remove_child_violates_min_cardinality(self):
        """Test removing child that violates min cardinality."""
        schema = WorkItem.WorkflowSchema(
            workflow_types={"Project": "desc", "Task": "desc"},
            child_schema_dict={
                "Project": {"Task": {"min_cardinality": 2, "max_cardinality": 5}},
                "Task": None
            }
        )
        parent = WorkItem(workitem_type="Project", workflow_schema=schema)
        # Parent should have 2 Task children due to min_cardinality=2
        assert parent.get_child_count("Task") == 2
        child = parent.get_child(0, "Task")
        with pytest.raises(ValueError, match="Cannot remove child; would violate min_cardinality=2"):
            parent.remove_child(child)

    def test_get_child_by_index(self):
        """Test getting child by index."""
        parent = WorkItem(workitem_type="Project")
        child1 = WorkItem(workitem_type="Task")
        child2 = WorkItem(workitem_type="Task")
        
        parent.add_child(child1)
        parent.add_child(child2)
        
        assert parent.get_child(0) == child1
        assert parent.get_child(1) == child2

    def test_get_child_by_index_and_type(self):
        """Test getting child by index and type filter."""
        parent = WorkItem(workitem_type="Project")
        task1 = WorkItem(workitem_type="Task")
        bug1 = WorkItem(workitem_type="Bug")
        task2 = WorkItem(workitem_type="Task")
        
        parent.add_child(task1)
        parent.add_child(bug1)
        parent.add_child(task2)
        
        # Get Task children by index
        assert parent.get_child(0, "Task") == task1
        assert parent.get_child(1, "Task") == task2
        
        # Get Bug children by index
        assert parent.get_child(0, "Bug") == bug1

    def test_get_child_negative_index(self):
        """Test getting child with negative index should raise error."""
        parent = WorkItem(workitem_type="Project")
        child = WorkItem(workitem_type="Task")
        parent.add_child(child)
        
        with pytest.raises(IndexError, match="child index cannot be negative"):
            parent.get_child(-1)

    def test_get_child_index_out_of_range(self):
        """Test getting child with index out of range."""
        parent = WorkItem(workitem_type="Project")
        child = WorkItem(workitem_type="Task")
        parent.add_child(child)
        
        with pytest.raises(IndexError, match="child index out of range"):
            parent.get_child(1)

    def test_get_child_type_filtered_index_out_of_range(self):
        """Test getting child with type filter and index out of range."""
        parent = WorkItem(workitem_type="Project")
        task = WorkItem(workitem_type="Task")
        parent.add_child(task)
        
        with pytest.raises(IndexError, match="child index out of range for filters: workitem_type='Task', is_complete=None"):
            parent.get_child(1, "Task")

    def test_get_child_no_children_of_type(self):
        """Test getting child when no children of specified type exist."""
        parent = WorkItem(workitem_type="Project")
        task = WorkItem(workitem_type="Task")
        parent.add_child(task)
        
        assert parent.get_child(0, "Bug") is None

    def test_get_child_count(self):
        """Test getting child count."""
        parent = WorkItem(workitem_type="Project")
        task1 = WorkItem(workitem_type="Task")
        task2 = WorkItem(workitem_type="Task")
        bug1 = WorkItem(workitem_type="Bug")
        
        parent.add_child(task1)
        parent.add_child(task2)
        parent.add_child(bug1)
        
        assert parent.get_child_count() == 3
        assert parent.get_child_count("Task") == 2
        assert parent.get_child_count("Bug") == 1
        assert parent.get_child_count("NonExistent") == 0

    def test_remove_all_children_no_constraints(self):
        """Test removing all children when no constraints exist."""
        parent = WorkItem(workitem_type="Project")
        child1 = WorkItem(workitem_type="Task")
        child2 = WorkItem(workitem_type="Task")
        
        parent.add_child(child1)
        parent.add_child(child2)
        assert parent.get_child_count() == 2
        
        parent.remove_all_children()
        assert parent.get_child_count() == 0

    def test_remove_all_children_violates_min_cardinality(self):
        """Test removing all children when it violates min cardinality."""
        schema = WorkItem.WorkflowSchema(
            workflow_types={"Project": "desc", "Task": "desc"},
            child_schema_dict={
                "Project": {"Task": {"min_cardinality": 1, "max_cardinality": 5}},
                "Task": None
            }
        )
        
        parent = WorkItem(workitem_type="Project", workflow_schema=schema)
        
        with pytest.raises(ValueError, match="Cannot remove all children; min_cardinality > 0 for types: Task"):
            parent.remove_all_children()

    def test_index_of(self):
        """Test getting index of a child."""
        parent = WorkItem(workitem_type="Project")
        child1 = WorkItem(workitem_type="Task")
        child2 = WorkItem(workitem_type="Task")
        
        parent.add_child(child1)
        parent.add_child(child2)
        
        assert parent.index_of(child1) == 0
        assert parent.index_of(child2) == 1

    def test_min_cardinality_children_created_automatically(self):
        """Test that children are created automatically to satisfy min cardinality."""
        schema = WorkItem.WorkflowSchema(
            workflow_types={"Project": "desc", "Task": "desc", "Bug": "desc"},
            child_schema_dict={
                "Project": {
                    "Task": {"min_cardinality": 2, "max_cardinality": 5},
                    "Bug": {"min_cardinality": 1, "max_cardinality": 3}
                },
                "Task": None,
                "Bug": None
            }
        )
        
        parent = WorkItem(workitem_type="Project", workflow_schema=schema)
        
        # Should have 2 Task children and 1 Bug child automatically created
        assert parent.get_child_count("Task") == 2
        assert parent.get_child_count("Bug") == 1
        assert parent.get_child_count() == 3


class TestWorkItemPathOperations:
    """Test cases for WorkItem path and navigation operations."""

    def test_get_absolute_path_root(self):
        """Test getting absolute path for root workitem."""
        root = WorkItem(workitem_type="Project")
        assert root.get_absolute_path() == "/"

    def test_get_absolute_path_child(self):
        """Test getting absolute path for child workitem."""
        root = WorkItem(workitem_type="Project")
        child = WorkItem(workitem_type="Task")
        root.add_child(child)
        
        assert child.get_absolute_path() == "/Task[index=0]"

    def test_get_absolute_path_nested(self):
        """Test getting absolute path for nested workitems."""
        root = WorkItem(workitem_type="Project")
        task = WorkItem(workitem_type="Task")
        subtask = WorkItem(workitem_type="Subtask")
        
        root.add_child(task)
        task.add_child(subtask)
        
        assert subtask.get_absolute_path() == "/Task[index=0]/Subtask[index=0]"

    def test_get_absolute_path_multiple_children_same_type(self):
        """Test getting absolute path with multiple children of same type."""
        root = WorkItem(workitem_type="Project")
        task1 = WorkItem(workitem_type="Task")
        task2 = WorkItem(workitem_type="Task")
        
        root.add_child(task1)
        root.add_child(task2)
        
        assert task1.get_absolute_path() == "/Task[index=0]"
        assert task2.get_absolute_path() == "/Task[index=1]"

    def test_get_child_workitem_by_path(self):
        """Test getting child workitem by relative path."""
        root = WorkItem(workitem_type="Project")
        task = WorkItem(workitem_type="Task")
        subtask = WorkItem(workitem_type="Subtask")
        
        root.add_child(task)
        task.add_child(subtask)
        
        # Test various path formats
        assert root._get_child_workitem("Task[index=0]") == task
        assert root._get_child_workitem("Task") == task  # Default index 0
        assert root._get_child_workitem("Task[index=0]/Subtask[index=0]") == subtask
        assert root._get_child_workitem("Task/Subtask") == subtask

    def test_get_child_workitem_empty_path(self):
        """Test getting child workitem with empty path returns self."""
        root = WorkItem(workitem_type="Project")
        assert root._get_child_workitem("") == root
        assert root._get_child_workitem("/") == root

    def test_get_child_workitem_invalid_path(self):
        """Test getting child workitem with invalid path."""
        root = WorkItem(workitem_type="Project")
        task = WorkItem(workitem_type="Task")
        root.add_child(task)
        
        # Invalid segment format
        assert root._get_child_workitem("Invalid-Path") is None
        
        # Non-existent child
        assert root._get_child_workitem("NonExistent[0]") is None
        
        # Index out of range
        assert root._get_child_workitem("Task[5]") is None

    def test_get_workitem_by_absolute_path(self):
        """Test getting workitem by absolute path from any node."""
        root = WorkItem(workitem_type="Project")
        task = WorkItem(workitem_type="Task")
        subtask = WorkItem(workitem_type="Subtask")
        
        root.add_child(task)
        task.add_child(subtask)
        
        # Test from different nodes
        assert root.get_workitem("/") == root
        assert root.get_workitem("/Task[index=0]") == task
        assert task.get_workitem("/Task[index=0]/Subtask[index=0]") == subtask
        assert subtask.get_workitem("/") == root

    def test_get_workitem_invalid_absolute_path(self):
        """Test getting workitem with invalid absolute path."""
        root = WorkItem(workitem_type="Project")
        
        assert root.get_workitem("") is None
        assert root.get_workitem("/NonExistent") is None

    def test_get_next_workitem_with_children(self):
        """Test getting next workitem when current has children."""
        root = WorkItem(workitem_type="Project")
        task = WorkItem(workitem_type="Task")
        subtask = WorkItem(workitem_type="Subtask")
        
        root.add_child(task)
        task.add_child(subtask)
        
        # Next workitem should be first child
        assert root.get_next_workitem() == task
        assert task.get_next_workitem() == subtask

    def test_get_next_workitem_no_children(self):
        """Test getting next workitem when current has no children."""
        root = WorkItem(workitem_type="Project")
        task1 = WorkItem(workitem_type="Task")
        task2 = WorkItem(workitem_type="Task")
        
        root.add_child(task1)
        root.add_child(task2)
        
        # Next workitem should be next sibling
        assert task1.get_next_workitem() == task2

    def test_get_next_workitem_with_type_filter(self):
        """Test getting next workitem with type filter."""
        root = WorkItem(workitem_type="Project")
        task1 = WorkItem(workitem_type="Task")
        bug1 = WorkItem(workitem_type="Bug")
        task2 = WorkItem(workitem_type="Task")
        
        root.add_child(task1)
        root.add_child(bug1)
        root.add_child(task2)
        
        # Next Task after task1 should be task2 (skipping bug1)
        assert task1.get_next_workitem("Task") == task2

    def test_get_next_workitem_no_next(self):
        """Test getting next workitem when none exists."""
        root = WorkItem(workitem_type="Project")
        task = WorkItem(workitem_type="Task")
        root.add_child(task)
        
        # No next workitem for the last child
        assert task.get_next_workitem() is None

    def test_find_child_workitems_exact_match(self):
        """Test find_child_workitems with exact field match."""
        root = WorkItem(workitem_type="Project")
        
        # Create children with different data
        task1 = WorkItem(workitem_type="Task", data_dict={"id": "task-001", "priority": "high"})
        task2 = WorkItem(workitem_type="Task", data_dict={"id": "task-002", "priority": "medium"})
        task3 = WorkItem(workitem_type="Task", data_dict={"id": "task-003", "priority": "high"})
        
        root.add_child(task1)
        root.add_child(task2)
        root.add_child(task3)
        
        # Test exact match on id field
        result = root.find_child_workitems("Task[id=task-002]")
        assert len(result) == 1
        assert result[0] == task2
        
        # Test exact match on priority field
        result = root.find_child_workitems("Task[priority=high]")
        assert len(result) == 2
        assert task1 in result
        assert task3 in result

    def test_find_child_workitems_fuzzy_match_single(self):
        """Test find_child_workitems with fuzzy match returning single result."""
        root = WorkItem(workitem_type="Project")
        
        # Create children with similar names
        task1 = WorkItem(workitem_type="Task", data_dict={"name": "invoice_processing"})
        task2 = WorkItem(workitem_type="Task", data_dict={"name": "payment_validation"})
        task3 = WorkItem(workitem_type="Task", data_dict={"name": "user_authentication"})
        
        root.add_child(task1)
        root.add_child(task2)
        root.add_child(task3)
        
        # Test fuzzy match that should return single result
        result = root.find_child_workitems("Task[name=invoice]")
        assert len(result) == 1
        assert result[0] == task1

    def test_find_child_workitems_fuzzy_match_multiple(self):
        """Test find_child_workitems with fuzzy match returning multiple results."""
        root = WorkItem(workitem_type="Project")
        
        # Create children with similar names
        task1 = WorkItem(workitem_type="Task", data_dict={"name": "invoice_processing"})
        task2 = WorkItem(workitem_type="Task", data_dict={"name": "invoice_validation"})
        task3 = WorkItem(workitem_type="Task", data_dict={"name": "payment_processing"})
        
        root.add_child(task1)
        root.add_child(task2)
        root.add_child(task3)
        
        # Test fuzzy match that should return multiple results
        result = root.find_child_workitems("Task[name=invoice]")
        assert len(result) == 2
        assert task1 in result
        assert task2 in result

    def test_find_child_workitems_no_match(self):
        """Test find_child_workitems when no matches are found."""
        root = WorkItem(workitem_type="Project")
        
        task1 = WorkItem(workitem_type="Task", data_dict={"id": "task-001"})
        task2 = WorkItem(workitem_type="Task", data_dict={"id": "task-002"})
        
        root.add_child(task1)
        root.add_child(task2)
        
        # Test non-existent field
        result = root.find_child_workitems("Task[nonexistent=value]")
        assert result == []
        
        # Test non-existent value - DatabaseValidator returns all values as suggestions
        # so this will actually return both tasks
        result = root.find_child_workitems("Task[id=task-999]")
        assert len(result) == 2  # Both tasks are returned as fuzzy matches
        assert task1 in result
        assert task2 in result

    def test_find_child_workitems_no_children_of_type(self):
        """Test find_child_workitems when no children of specified type exist."""
        root = WorkItem(workitem_type="Project")
        
        # Add children of different type
        bug = WorkItem(workitem_type="Bug", data_dict={"id": "bug-001"})
        root.add_child(bug)
        
        # Try to find Task children (none exist)
        result = root.find_child_workitems("Task[id=task-001]")
        assert result == []

    def test_find_child_workitems_invalid_format(self):
        """Test find_child_workitems with invalid format raises ValueError."""
        root = WorkItem(workitem_type="Project")
        
        # Test missing field=value part
        with pytest.raises(ValueError, match="relative_matching_path must be of the form 'Type\\[field=value\\]'"):
            root.find_child_workitems("Task")
        
        # Test malformed format
        with pytest.raises(ValueError, match="relative_matching_path must be of the form 'Type\\[field=value\\]'"):
            root.find_child_workitems("Task[id]")
        
        # Test missing closing bracket
        with pytest.raises(ValueError, match="relative_matching_path must be of the form 'Type\\[field=value\\]'"):
            root.find_child_workitems("Task[id=value")

    def test_find_child_workitems_with_leading_slash(self):
        """Test find_child_workitems accepts leading slash for consistency."""
        root = WorkItem(workitem_type="Project")
        
        task = WorkItem(workitem_type="Task", data_dict={"id": "task-001"})
        root.add_child(task)
        
        # Test with leading slash
        result = root.find_child_workitems("/Task[id=task-001]")
        assert len(result) == 1
        assert result[0] == task

    def test_find_child_workitems_empty_field_values(self):
        """Test find_child_workitems with empty or missing field values."""
        root = WorkItem(workitem_type="Project")
        
        task1 = WorkItem(workitem_type="Task", data_dict={"id": "task-001", "description": ""})
        task2 = WorkItem(workitem_type="Task", data_dict={"id": "task-002"})  # No description field
        
        root.add_child(task1)
        root.add_child(task2)
        
        # Test matching empty string - both tasks match (task1 has empty description, task2 has no description field)
        result = root.find_child_workitems("Task[description=]")
        assert len(result) == 2
        assert task1 in result  # task1 has empty description
        assert task2 in result  # task2 has no description field (returns empty string)
        
        # Test matching non-existent field (should return empty list)
        result = root.find_child_workitems("Task[description=some_value]")
        assert result == []

    def test_find_child_workitems_mixed_data_types(self):
        """Test find_child_workitems with different data types in fields."""
        root = WorkItem(workitem_type="Project")
        
        task1 = WorkItem(workitem_type="Task", data_dict={"id": "task-001", "priority": 1, "active": True})
        task2 = WorkItem(workitem_type="Task", data_dict={"id": "task-002", "priority": 2, "active": False})
        
        root.add_child(task1)
        root.add_child(task2)
        
        # Test matching numeric values (converted to string)
        result = root.find_child_workitems("Task[priority=1]")
        assert len(result) == 1
        assert result[0] == task1
        
        # Test matching boolean values (converted to string)
        result = root.find_child_workitems("Task[active=True]")
        assert len(result) == 1
        assert result[0] == task1

    def test_find_child_workitems_case_sensitivity(self):
        """Test find_child_workitems case sensitivity in field values."""
        root = WorkItem(workitem_type="Project")
        
        task1 = WorkItem(workitem_type="Task", data_dict={"status": "PENDING"})
        task2 = WorkItem(workitem_type="Task", data_dict={"status": "pending"})
        
        root.add_child(task1)
        root.add_child(task2)
        
        # Test exact case match
        result = root.find_child_workitems("Task[status=PENDING]")
        assert len(result) == 1
        assert result[0] == task1
        
        result = root.find_child_workitems("Task[status=pending]")
        assert len(result) == 1
        assert result[0] == task2

class TestWorkItemSerialization:
    """Test cases for WorkItem serialization functionality."""

    def test_to_dict(self):
        """Test WorkItem serialization to dictionary."""
        workitem = WorkItem(workitem_type="Task")
        workitem["title"] = "Test Task"
        workitem.is_complete = True
        
        result = workitem._to_dict()
        
        assert isinstance(result, dict)
        assert result["workitem_type"] == "Task"
        assert result["is_complete"] is True
        assert result["data_dict"]["title"] == "Test Task"
        assert "id" not in result  # No ID was set

    def test_from_dict(self):
        """Test WorkItem deserialization from dictionary."""
        data = {
            'workitem_type': 'Task',
            'is_complete': True,
            'data_dict': {'title': 'task-1'}
        }
        
        workitem = WorkItem._from_dict(data)
        assert isinstance(workitem, WorkItem)
        assert workitem.type == 'Task'
        assert workitem.is_complete is True
        assert workitem["title"] == "task-1"
        assert workitem.id is None  # No ID in the data


class TestWorkItemEdgeCases:
    """Test cases for edge cases and error conditions."""

    def test_workitem_with_complex_schema(self):
        """Test WorkItem with complex workflow schema."""
        schema = WorkItem.WorkflowSchema(
            workflow_types={"Epic": "desc", "Story": "desc", "Task": "desc", "Bug": "desc", "Subtask": "desc"},
            child_schema_dict={
                "Epic": {
                    "Story": {"min_cardinality": 1, "max_cardinality": 10},
                    "Bug": {"min_cardinality": 0, "max_cardinality": 5}
                },
                "Story": {
                    "Task": {"min_cardinality": 1, "max_cardinality": 20},
                    "Bug": {"min_cardinality": 0, "max_cardinality": 3}
                },
                "Task": {"Subtask": {"min_cardinality": 0, "max_cardinality": 5}},
                "Bug": None,
                "Subtask": None
            }
        )
        
        epic = WorkItem(workitem_type="Epic", workflow_schema=schema)
        
        # Should have 1 Story automatically created due to min_cardinality
        assert epic.get_child_count("Story") == 1
        assert epic.get_child_count("Bug") == 0
        
        # The Story should have 1 Task automatically created
        story = epic.get_child(0, "Story")
        assert story.get_child_count("Task") == 1
        assert story.get_child_count("Bug") == 0

    def test_remove_child_updates_positions(self):
        """Test that removing a child correctly updates positions of remaining children."""
        parent = WorkItem(workitem_type="Project")
        child1 = WorkItem(workitem_type="Task")
        child2 = WorkItem(workitem_type="Task")
        child3 = WorkItem(workitem_type="Task")
        
        parent.add_child(child1)
        parent.add_child(child2)
        parent.add_child(child3)
        
        # Remove middle child
        parent.remove_child(child2)
        
        # Verify positions are updated correctly
        assert parent.index_of(child1) == 0
        assert parent.index_of(child3) == 1
        assert parent.get_child(0) == child1
        assert parent.get_child(1) == child3

    def test_large_number_of_children(self):
        """Test WorkItem with large number of children."""
        parent = WorkItem(workitem_type="Project")
        children = []
        
        # Add 100 children
        for i in range(100):
            child = WorkItem(workitem_type="Task")
            child[f"task_id"] = f"task_{i}"
            children.append(child)
            parent.add_child(child)
        
        assert parent.get_child_count() == 100
        
        # Verify all children are accessible
        for i, child in enumerate(children):
            assert parent.get_child(i) == child
            assert parent.index_of(child) == i

    def test_deep_nesting(self):
        """Test WorkItem with deep nesting."""
        current = WorkItem(workitem_type="Level0")
        root = current
        
        # Create 10 levels of nesting
        for i in range(1, 11):
            child = WorkItem(workitem_type=f"Level{i}")
            current.add_child(child)
            current = child
        
        # Verify path
        expected_path = "/" + "/".join(f"Level{i}[index=0]" for i in range(1, 11))
        assert current.get_absolute_path() == expected_path
        
        # Verify we can navigate back to root
        assert current.get_workitem("/") == root

    def test_schema_validation_edge_cases(self):
        """Test schema validation edge cases."""
        # Test with max_cardinality = 0 (no children allowed)
        schema = WorkItem.WorkflowSchema(
            workflow_types={"Parent": "desc", "Child": "desc"},
            child_schema_dict={
                "Parent": {"Child": {"min_cardinality": 0, "max_cardinality": 0}},
                "Child": None
            }
        )
        
        parent = WorkItem(workitem_type="Parent", workflow_schema=schema)
        child = WorkItem(workitem_type="Child")
        
        with pytest.raises(ValueError, match="Cannot add more than 0 children"):
            parent.add_child(child)

    def test_concurrent_modifications(self):
        """Test that concurrent modifications don't break internal state."""
        parent = WorkItem(workitem_type="Project")
        children = []
        
        # Add children
        for i in range(10):
            child = WorkItem(workitem_type="Task")
            children.append(child)
            parent.add_child(child)
        
        # Remove every other child
        for i in range(0, 10, 2):
            parent.remove_child(children[i])
        
        # Verify remaining children have correct positions
        remaining_children = [children[i] for i in range(1, 10, 2)]
        assert parent.get_child_count() == 5
        
        for i, child in enumerate(remaining_children):
            assert parent.get_child(i) == child
            assert parent.index_of(child) == i


class TestWorkItemRecentFeatures:
    def test_get_status_and_description(self):
        # Setup
        schema = WorkItem.WorkflowSchema(
            workflow_types={"Project": "desc", "Task": "desc", "Bug": "desc"},
            child_schema_dict={
                "Project": {
                    "Task": {"min_cardinality": 0, "max_cardinality": 5},
                    "Bug": {"min_cardinality": 0, "max_cardinality": 5}
                },
                "Task": None,
                "Bug": None
            }
        )
        project = WorkItem(workitem_type="Project", workflow_schema=schema)
        task1 = WorkItem(workitem_type="Task", workflow_schema=schema)
        task2 = WorkItem(workitem_type="Task", workflow_schema=schema)
        bug1 = WorkItem(workitem_type="Bug", workflow_schema=schema)
        project.add_child(task1)
        project.add_child(task2)
        project.add_child(bug1)
        task1.is_complete = True
        task2.is_complete = False
        bug1.is_complete = True
        project.is_complete = False

        # Test get_status
        status = project.get_status()
        assert status["is_complete"] is False
        assert status["Task"] == "1/2 completed"
        assert status["Bug"] == "1/1 completed"

        # Test description property
        assert project.schema["description"] == "desc"
        assert task1.schema["description"] == "desc"
        assert bug1.schema["description"] == "desc"

    def test_navigation_with_is_complete_filter(self):
        parent = WorkItem(workitem_type="Parent")
        c1 = WorkItem(workitem_type="Child")
        c2 = WorkItem(workitem_type="Child")
        c3 = WorkItem(workitem_type="Child")
        parent.add_child(c1)
        parent.add_child(c2)
        parent.add_child(c3)
        c1.is_complete = True
        c2.is_complete = False
        c3.is_complete = True

        # get_child_count with is_complete
        assert parent.get_child_count("Child", is_complete=True) == 2
        assert parent.get_child_count("Child", is_complete=False) == 1

        # get_child with is_complete
        assert parent.get_child(0, "Child", is_complete=True) == c1
        assert parent.get_child(1, "Child", is_complete=True) == c3
        assert parent.get_child(0, "Child", is_complete=False) == c2

        # Navigation methods with is_complete
        assert c1.get_next_workitem("Child", is_complete=True) == c3
        assert c3.get_previous_workitem("Child", is_complete=True) == c1
        assert c2.get_next_workitem("Child", is_complete=False) is None
        assert c2.get_previous_workitem("Child", is_complete=False) == parent
        assert c1.get_first_workitem("Child", is_complete=True) == c1
        assert c1.get_last_workitem("Child", is_complete=True) == c3

    def test_get_first_and_last_workitem_no_siblings(self):
        solo = WorkItem(workitem_type="Solo")
        assert solo.get_first_workitem() == solo
        assert solo.get_last_workitem() == solo


class TestWorkItemIsComplete:
    """Test cases for WorkItem is_complete property with recursive propagation."""

    def test_leaf_workitem_is_complete_setter(self):
        """Test that leaf workitems can be set directly and propagate to parent."""
        parent = WorkItem(workitem_type="Project")
        child = WorkItem(workitem_type="Task")
        parent.add_child(child)
        
        # Initially all items should be incomplete
        assert not parent.is_complete
        assert not child.is_complete
        
        # Set leaf item to complete - should propagate to parent
        child.is_complete = True
        assert child.is_complete
        assert parent.is_complete
        
        # Set leaf item to incomplete - should propagate to parent
        child.is_complete = False
        assert not child.is_complete
        assert not parent.is_complete

    def test_non_leaf_workitem_is_complete_calculation(self):
        """Test that non-leaf workitems calculate completion based on children."""
        parent = WorkItem(workitem_type="Project")
        child1 = WorkItem(workitem_type="Task")
        child2 = WorkItem(workitem_type="Task")
        parent.add_child(child1)
        parent.add_child(child2)
        
        # Initially all items should be incomplete
        assert not parent.is_complete
        assert not child1.is_complete
        assert not child2.is_complete
        
        # Set one child to complete - parent should still be incomplete
        child1.is_complete = True
        assert child1.is_complete
        assert not child2.is_complete
        assert not parent.is_complete
        
        # Set both children to complete - parent should become complete
        child2.is_complete = True
        assert child1.is_complete
        assert child2.is_complete
        assert parent.is_complete
        
        # Set one child back to incomplete - parent should become incomplete
        child1.is_complete = False
        assert not child1.is_complete
        assert child2.is_complete
        assert not parent.is_complete

    def test_deep_nested_completion_propagation(self):
        """Test completion propagation through multiple levels of nesting."""
        # Create a 3-level hierarchy: root -> parent -> child
        root = WorkItem(workitem_type="Root")
        parent = WorkItem(workitem_type="Parent")
        child = WorkItem(workitem_type="Child")
        
        root.add_child(parent)
        parent.add_child(child)
        
        # Initially all items should be incomplete
        assert not root.is_complete
        assert not parent.is_complete
        assert not child.is_complete
        
        # Set leaf item to complete - should propagate up to root
        child.is_complete = True
        assert child.is_complete
        assert parent.is_complete
        assert root.is_complete
        
        # Set leaf item to incomplete - should propagate up to root
        child.is_complete = False
        assert not child.is_complete
        assert not parent.is_complete
        assert not root.is_complete

    def test_multiple_children_completion_propagation(self):
        """Test completion propagation with multiple children at each level."""
        root = WorkItem(workitem_type="Root")
        parent1 = WorkItem(workitem_type="Parent")
        parent2 = WorkItem(workitem_type="Parent")
        child1 = WorkItem(workitem_type="Child")
        child2 = WorkItem(workitem_type="Child")
        child3 = WorkItem(workitem_type="Child")
        child4 = WorkItem(workitem_type="Child")
        
        root.add_child(parent1)
        root.add_child(parent2)
        parent1.add_child(child1)
        parent1.add_child(child2)
        parent2.add_child(child3)
        parent2.add_child(child4)
        
        # Initially all items should be incomplete
        assert not root.is_complete
        assert not parent1.is_complete
        assert not parent2.is_complete
        assert not child1.is_complete
        assert not child2.is_complete
        assert not child3.is_complete
        assert not child4.is_complete
        
        # Complete one child - only its parent should become complete
        child1.is_complete = True
        assert not root.is_complete
        assert not parent1.is_complete  # parent1 still has child2 incomplete
        assert not parent2.is_complete
        assert child1.is_complete
        assert not child2.is_complete
        assert not child3.is_complete
        assert not child4.is_complete
        
        # Complete all children of parent1 - parent1 should become complete
        child2.is_complete = True
        assert not root.is_complete  # root still has parent2 incomplete
        assert parent1.is_complete
        assert not parent2.is_complete
        assert child1.is_complete
        assert child2.is_complete
        assert not child3.is_complete
        assert not child4.is_complete
        
        # Complete all children of parent2 - root should become complete
        child3.is_complete = True
        child4.is_complete = True
        assert root.is_complete
        assert parent1.is_complete
        assert parent2.is_complete
        assert child1.is_complete
        assert child2.is_complete
        assert child3.is_complete
        assert child4.is_complete
        
        # Set one child to incomplete - should propagate up
        child1.is_complete = False
        assert not root.is_complete
        assert not parent1.is_complete
        assert parent2.is_complete
        assert not child1.is_complete
        assert child2.is_complete
        assert child3.is_complete
        assert child4.is_complete

    def test_root_workitem_without_parent(self):
        """Test that root workitems without parents work correctly."""
        root = WorkItem(workitem_type="Root")
        child = WorkItem(workitem_type="Child")
        root.add_child(child)
        
        # Initially incomplete
        assert not root.is_complete
        assert not child.is_complete
        
        # Set child to complete - root should become complete
        child.is_complete = True
        assert root.is_complete
        assert child.is_complete
        
        # Set child to incomplete - root should become incomplete
        child.is_complete = False
        assert not root.is_complete
        assert not child.is_complete

    def test_leaf_workitem_without_parent(self):
        """Test that leaf workitems without parents work correctly."""
        leaf = WorkItem(workitem_type="Leaf")
        
        # Initially incomplete
        assert not leaf.is_complete
        
        # Can be set directly
        leaf.is_complete = True
        assert leaf.is_complete
        
        leaf.is_complete = False
        assert not leaf.is_complete

    def test_completion_propagation_with_mixed_types(self):
        """Test completion propagation with different workitem types."""
        project = WorkItem(workitem_type="Project")
        task1 = WorkItem(workitem_type="Task")
        task2 = WorkItem(workitem_type="Task")
        bug1 = WorkItem(workitem_type="Bug")
        bug2 = WorkItem(workitem_type="Bug")
        
        project.add_child(task1)
        project.add_child(task2)
        project.add_child(bug1)
        project.add_child(bug2)
        
        # Initially all incomplete
        assert not project.is_complete
        assert not task1.is_complete
        assert not task2.is_complete
        assert not bug1.is_complete
        assert not bug2.is_complete
        
        # Complete all tasks - project should still be incomplete due to bugs
        task1.is_complete = True
        task2.is_complete = True
        assert not project.is_complete
        assert task1.is_complete
        assert task2.is_complete
        assert not bug1.is_complete
        assert not bug2.is_complete
        
        # Complete all bugs - project should become complete
        bug1.is_complete = True
        bug2.is_complete = True
        assert project.is_complete
        assert task1.is_complete
        assert task2.is_complete
        assert bug1.is_complete
        assert bug2.is_complete
        
        # Set one task to incomplete - project should become incomplete
        task1.is_complete = False
        assert not project.is_complete
        assert not task1.is_complete
        assert task2.is_complete
        assert bug1.is_complete
        assert bug2.is_complete

    def test_completion_propagation_after_child_removal(self):
        """Test completion propagation behavior after removing children."""
        parent = WorkItem(workitem_type="Parent")
        child1 = WorkItem(workitem_type="Child")
        child2 = WorkItem(workitem_type="Child")
        
        parent.add_child(child1)
        parent.add_child(child2)
        
        # Complete both children
        child1.is_complete = True
        child2.is_complete = True
        assert parent.is_complete
        
        # Remove one complete child - parent should remain complete if remaining child is complete
        parent.remove_child(child1)
        assert parent.is_complete  # parent should still be complete because child2 is complete
        assert child2.is_complete  # remaining child should still be complete
        
        # Now set remaining child to incomplete - parent should become incomplete
        child2.is_complete = False
        assert not parent.is_complete
        assert not child2.is_complete

    def test_completion_propagation_after_child_addition(self):
        """Test completion propagation behavior after adding children."""
        parent = WorkItem(workitem_type="Parent")
        child1 = WorkItem(workitem_type="Child")
        child2 = WorkItem(workitem_type="Child")
        
        # Add and complete first child
        parent.add_child(child1)
        child1.is_complete = True
        assert parent.is_complete
        
        # Add incomplete child - parent should become incomplete
        parent.add_child(child2)
        assert not parent.is_complete
        assert child1.is_complete
        assert not child2.is_complete
        
        # Complete second child - parent should become complete again
        child2.is_complete = True
        assert parent.is_complete
        assert child1.is_complete
        assert child2.is_complete

    def test_completion_propagation_edge_cases(self):
        """Test edge cases in completion propagation."""
        # Test with single child
        parent = WorkItem(workitem_type="Parent")
        child = WorkItem(workitem_type="Child")
        parent.add_child(child)
        
        # Test multiple rapid changes
        child.is_complete = True
        child.is_complete = False
        child.is_complete = True
        child.is_complete = False
        
        assert not parent.is_complete
        assert not child.is_complete
        
        # Test with empty parent (no children)
        empty_parent = WorkItem(workitem_type="EmptyParent")
        assert not empty_parent.is_complete
        
        # Empty parent should be considered complete (no incomplete children)
        # This depends on the implementation - if empty parents are considered complete
        # by default, then this test should reflect that behavior


if __name__ == "__main__":
    pytest.main([__file__])

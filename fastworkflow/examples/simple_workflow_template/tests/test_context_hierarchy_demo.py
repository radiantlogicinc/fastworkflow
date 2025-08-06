"""Demonstration test for the new context_hierarchy_model functionality."""

import pytest
import tempfile
import os
from fastworkflow._workflows.simple_workflow_template.application.workitem import WorkItem


def test_sample_workflow_schema_with_context_hierarchy():
    """Test the sample workflow schema with context_hierarchy_model."""
    
    # Sample workflow schema matching the attached JSON
    workflow_types = {
        "Epic": "A very long task with lots of stories and bugs",
        "Story": "A story is a user story that is part of an epic", 
        "Bug": "A bug is a bug that is part of a story",
        "Task": "A task is a subtask of a story",
        "Subtask": "A subtask is a subtask of a task"
    }
    
    child_schema_dict = {
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
    
    # The new context_hierarchy_model from the attached JSON
    context_hierarchy_model = {
        "Epic": None,  # Root-only
        "Story": {
            "parent": ["Epic"]
        },
        "Bug": {
            "parent": ["Epic", "Story"]
        },
        "Task": {
            "parent": ["Story"]
        },
        "Subtask": {
            "parent": ["Task"]
        }
    }
    
    # Create schema with context hierarchy
    schema = WorkItem.WorkflowSchema(
        workflow_types=workflow_types,
        child_schema_dict=child_schema_dict,
        parents_dict=context_hierarchy_model
    )
    
    # Test 1: Epic can only be created as root
    epic = schema.create_workitem("Epic")
    assert epic.type == "Epic"
    assert epic.parent is None
    
    # Test 2: Story must have Epic parent
    story = schema.create_workitem("Story", parent=epic)
    assert story.type == "Story"
    assert story.parent == epic
    
    # Test 3: Bug can have either Epic or Story parent
    bug1 = schema.create_workitem("Bug", parent=epic)
    bug2 = schema.create_workitem("Bug", parent=story)
    assert bug1.type == "Bug"
    assert bug1.parent == epic
    assert bug2.type == "Bug"
    assert bug2.parent == story
    
    # Test 4: Task must have Story parent
    task = schema.create_workitem("Task", parent=story)
    assert task.type == "Task"
    assert task.parent == story
    
    # Test 5: Subtask must have Task parent
    subtask = schema.create_workitem("Subtask", parent=task)
    assert subtask.type == "Subtask"
    assert subtask.parent == task
    
    # Test 6: Validation errors
    with pytest.raises(ValueError, match="Epic is root-level only; parent given."):
        schema.create_workitem("Epic", parent=story)
    
    with pytest.raises(ValueError, match="Story requires a parent of type \\['Epic'\\]."):
        schema.create_workitem("Story")
    
    with pytest.raises(ValueError, match="Invalid parent type 'Epic' for Task; allowed: \\['Story'\\]."):
        schema.create_workitem("Task", parent=epic)
    
    with pytest.raises(ValueError, match="Invalid parent type 'Story' for Subtask; allowed: \\['Task'\\]."):
        schema.create_workitem("Subtask", parent=story)


def test_context_hierarchy_serialization():
    """Test that context_hierarchy_model is properly serialized and deserialized."""
    
    workflow_types = {"Epic": "desc", "Story": "desc"}
    context_hierarchy_model = {
        "Epic": None,
        "Story": {"parent": ["Epic"]}
    }
    
    schema = WorkItem.WorkflowSchema(
        workflow_types=workflow_types,
        parents_dict=context_hierarchy_model
    )
    
    # Test JSON file round-trip
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
    
    try:
        # Save to file
        schema.to_json_file(temp_path)
        
        # Load from file
        loaded_schema = WorkItem.WorkflowSchema.from_json_file(temp_path)
        
        # Verify context_hierarchy_model is preserved
        assert loaded_schema.parents_dict == context_hierarchy_model
        
        # Verify it still works
        epic = loaded_schema.create_workitem("Epic")
        story = loaded_schema.create_workitem("Story", parent=epic)
        assert story.parent == epic
        
    finally:
        os.unlink(temp_path)


def test_backwards_compatibility():
    """Test that existing code without context_hierarchy_model still works."""
    
    workflow_types = {"Epic": "desc", "Story": "desc"}
    
    # Create schema without context_hierarchy_model
    schema = WorkItem.WorkflowSchema(workflow_types=workflow_types)
    
    # Should work as before
    epic = schema.create_workitem("Epic")
    story = schema.create_workitem("Story", parent=epic)
    
    assert epic.type == "Epic"
    assert epic.parent is None
    assert story.type == "Story"
    assert story.parent == epic
    
    # context_hierarchy_model should be empty dict
    assert schema.parents_dict == {} 
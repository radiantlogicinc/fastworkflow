#!/usr/bin/env python3
"""Comprehensive test runner for WorkItem with advanced test cases."""

import sys
import traceback
import tempfile
import os
from workitem import WorkItem


def run_test(test_func, test_name):
    """Run a single test function and report results."""
    try:
        test_func()
        print(f"✓ {test_name}")
        return True
    except Exception as e:
        print(f"✗ {test_name}: {str(e)}")
        traceback.print_exc()
        return False


def test_complex_workflow_schema():
    """Test complex workflow schema with multiple child types."""
    schema = WorkItem.WorkflowSchema(
        workflow_types=["Epic", "Story", "Task", "Bug", "Subtask"],
        child_schema_dict={
            "Epic": [
                WorkItem.ChildSchema("Story", 1, 10),
                WorkItem.ChildSchema("Bug", 0, 5)
            ],
            "Story": [
                WorkItem.ChildSchema("Task", 1, 20),
                WorkItem.ChildSchema("Bug", 0, 3)
            ],
            "Task": [WorkItem.ChildSchema("Subtask", 0, 5)],
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
    
    # Test adding more children
    bug1 = WorkItem(workitem_type="Bug")
    epic.add_child(bug1)
    assert epic.get_child_count("Bug") == 1
    
    # Test deep nesting
    task = story.get_child(0, "Task")
    subtask = WorkItem(workitem_type="Subtask")
    task.add_child(subtask)
    assert task.get_child_count("Subtask") == 1


def test_json_serialization():
    """Test JSON file serialization and deserialization."""
    types = ["Project", "Task"]
    child_schemas = {
        "Project": [WorkItem.ChildSchema("Task", 1, 5)],
        "Task": None
    }
    schema = WorkItem.WorkflowSchema(workflow_types=types, child_schema_dict=child_schemas)
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
    
    try:
        # Save to file
        schema.to_json_file(temp_path)
        
        # Load from file
        loaded_schema = WorkItem.WorkflowSchema.from_json_file(temp_path)
        
        # Verify loaded schema
        assert loaded_schema.workflow_types == types
        assert "Project" in loaded_schema.child_schema_dict
        assert len(loaded_schema.child_schema_dict["Project"]) == 1
        assert loaded_schema.child_schema_dict["Project"][0].child_workitem_type == "Task"
        assert loaded_schema.child_schema_dict["Project"][0].min_cardinality == 1
        assert loaded_schema.child_schema_dict["Project"][0].max_cardinality == 5
    finally:
        os.unlink(temp_path)


def test_large_number_of_children():
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
        assert child["task_id"] == f"task_{i}"


def test_deep_nesting():
    """Test WorkItem with deep nesting."""
    current = WorkItem(workitem_type="Level0")
    root = current
    
    # Create 10 levels of nesting
    for i in range(1, 11):
        child = WorkItem(workitem_type=f"Level{i}")
        current.add_child(child)
        current = child
    
    # Verify path
    expected_path = "/" + "/".join(f"Level{i}[0]" for i in range(1, 11))
    assert current.get_absolute_path() == expected_path
    
    # Verify we can navigate back to root
    assert current.get_workitem("/") == root


def test_remove_child_updates_positions():
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


def test_typed_children_filtering():
    """Test filtering children by type."""
    parent = WorkItem(workitem_type="Project")
    task1 = WorkItem(workitem_type="Task")
    task2 = WorkItem(workitem_type="Task")
    bug1 = WorkItem(workitem_type="Bug")
    bug2 = WorkItem(workitem_type="Bug")
    
    parent.add_child(task1)
    parent.add_child(bug1)
    parent.add_child(task2)
    parent.add_child(bug2)
    
    # Test type-filtered access
    assert parent.get_child(0, "Task") == task1
    assert parent.get_child(1, "Task") == task2
    assert parent.get_child(0, "Bug") == bug1
    assert parent.get_child(1, "Bug") == bug2
    
    # Test counts
    assert parent.get_child_count() == 4
    assert parent.get_child_count("Task") == 2
    assert parent.get_child_count("Bug") == 2
    assert parent.get_child_count("NonExistent") == 0


def test_edge_case_paths():
    """Test edge cases in path operations."""
    root = WorkItem(workitem_type="Project")
    task = WorkItem(workitem_type="Task")
    root.add_child(task)
    
    # Test invalid paths
    assert root._get_child_workitem("Invalid-Path") is None
    assert root._get_child_workitem("NonExistent[0]") is None
    assert root._get_child_workitem("Task[5]") is None
    
    # Test empty path returns self
    assert root._get_child_workitem("") == root
    assert root._get_child_workitem("/") == root
    
    # Test invalid absolute paths
    assert root.get_workitem("") is None
    assert root.get_workitem("/NonExistent") is None


def test_next_workitem_navigation():
    """Test next workitem navigation functionality."""
    root = WorkItem(workitem_type="Project")
    task1 = WorkItem(workitem_type="Task")
    task2 = WorkItem(workitem_type="Task")
    bug1 = WorkItem(workitem_type="Bug")
    subtask = WorkItem(workitem_type="Subtask")
    
    root.add_child(task1)
    root.add_child(bug1)
    root.add_child(task2)
    task1.add_child(subtask)
    
    # Test navigation with children
    assert root.get_next_workitem() == task1
    assert task1.get_next_workitem() == subtask
    
    # Test navigation without children (next sibling)
    # subtask's parent is task1, task1's next sibling is bug1
    assert subtask.get_next_workitem() == bug1
    
    # Test type-filtered navigation
    assert task1.get_next_workitem("Task") == task2
    
    # Test no next workitem
    assert task2.get_next_workitem() is None


def test_schema_edge_cases():
    """Test schema validation edge cases."""
    # Test with max_cardinality = 0 (no children allowed)
    schema = WorkItem.WorkflowSchema(
        workflow_types=["Parent", "Child"],
        child_schema_dict={
            "Parent": [WorkItem.ChildSchema("Child", 0, 0)],
            "Child": None
        }
    )
    
    parent = WorkItem(workitem_type="Parent", workflow_schema=schema)
    child = WorkItem(workitem_type="Child")
    
    try:
        parent.add_child(child)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Cannot add more than 0 children" in str(e)


def test_remove_all_children_edge_cases():
    """Test remove_all_children with various constraints."""
    # Test with no constraints
    parent = WorkItem(workitem_type="Project")
    child1 = WorkItem(workitem_type="Task")
    child2 = WorkItem(workitem_type="Task")
    
    parent.add_child(child1)
    parent.add_child(child2)
    parent.remove_all_children()
    assert parent.get_child_count() == 0
    
    # Test with min cardinality constraint
    schema = WorkItem.WorkflowSchema(
        workflow_types=["Project", "Task"],
        child_schema_dict={
            "Project": [WorkItem.ChildSchema("Task", 1, 5)],
            "Task": None
        }
    )
    
    parent2 = WorkItem(workitem_type="Project", workflow_schema=schema)
    try:
        parent2.remove_all_children()
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Cannot remove all children; min_cardinality > 0 for types: Task" in str(e)


def test_workitem_serialization():
    """Test WorkItem serialization methods."""
    workitem = WorkItem(workitem_type="Task")
    workitem["title"] = "Test Task"
    workitem["priority"] = "High"
    workitem.is_complete = True
    
    # Test to_dict
    result = workitem._to_dict()
    expected = {
        'workitem_type': 'Task',
        'is_complete': True,
        'data_dict': {'title': 'Test Task', 'priority': 'High'}
    }
    assert result == expected
    
    # Test from_dict
    workitem2 = WorkItem._from_dict(result)
    assert workitem2.workitem_type == "Task"
    assert workitem2.is_complete == True
    assert workitem2["title"] == "Test Task"
    assert workitem2["priority"] == "High"


def test_concurrent_modifications():
    """Test that concurrent modifications don't break internal state."""
    parent = WorkItem(workitem_type="Project")
    children = []
    
    # Add children
    for i in range(10):
        child = WorkItem(workitem_type="Task")
        child["id"] = i
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
        assert child["id"] == i * 2 + 1


def test_multiple_child_types_same_parent():
    """Test parent with multiple different child types."""
    parent = WorkItem(workitem_type="Project")
    
    # Add different types of children
    for i in range(3):
        task = WorkItem(workitem_type="Task")
        bug = WorkItem(workitem_type="Bug")
        feature = WorkItem(workitem_type="Feature")
        
        task["id"] = f"task_{i}"
        bug["id"] = f"bug_{i}"
        feature["id"] = f"feature_{i}"
        
        parent.add_child(task)
        parent.add_child(bug)
        parent.add_child(feature)
    
    # Verify counts
    assert parent.get_child_count() == 9
    assert parent.get_child_count("Task") == 3
    assert parent.get_child_count("Bug") == 3
    assert parent.get_child_count("Feature") == 3
    
    # Verify type-specific access
    for i in range(3):
        task = parent.get_child(i, "Task")
        bug = parent.get_child(i, "Bug")
        feature = parent.get_child(i, "Feature")
        
        assert task["id"] == f"task_{i}"
        assert bug["id"] == f"bug_{i}"
        assert feature["id"] == f"feature_{i}"


def main():
    """Run all comprehensive tests."""
    tests = [
        (test_complex_workflow_schema, "Complex Workflow Schema"),
        (test_json_serialization, "JSON Serialization"),
        (test_large_number_of_children, "Large Number of Children"),
        (test_deep_nesting, "Deep Nesting"),
        (test_remove_child_updates_positions, "Remove Child Updates Positions"),
        (test_typed_children_filtering, "Typed Children Filtering"),
        (test_edge_case_paths, "Edge Case Paths"),
        (test_next_workitem_navigation, "Next WorkItem Navigation"),
        (test_schema_edge_cases, "Schema Edge Cases"),
        (test_remove_all_children_edge_cases, "Remove All Children Edge Cases"),
        (test_workitem_serialization, "WorkItem Serialization"),
        (test_concurrent_modifications, "Concurrent Modifications"),
        (test_multiple_child_types_same_parent, "Multiple Child Types Same Parent"),
    ]
    
    passed = 0
    failed = 0
    
    print("Running comprehensive WorkItem tests...")
    print("=" * 60)
    
    for test_func, test_name in tests:
        if run_test(test_func, test_name):
            passed += 1
        else:
            failed += 1
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed > 0:
        sys.exit(1)
    else:
        print("All comprehensive tests passed! ✓")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Simple test runner for WorkItem tests without pytest dependencies."""

import sys
import traceback
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


def test_child_schema_basic():
    """Test basic ChildSchema functionality."""
    # Test valid initialization
    schema = WorkItem.ChildSchema("TaskItem", 1, 5)
    assert schema.child_workitem_type == "TaskItem"
    assert schema.min_cardinality == 1
    assert schema.max_cardinality == 5
    
    # Test defaults
    schema2 = WorkItem.ChildSchema("TaskItem")
    assert schema2.min_cardinality == 0
    assert schema2.max_cardinality is None
    
    # Test serialization
    data = schema.to_dict()
    expected = {
        "child_workitem_type": "TaskItem",
        "min_cardinality": 1,
        "max_cardinality": 5
    }
    assert data == expected
    
    # Test deserialization
    schema3 = WorkItem.ChildSchema.from_dict(data)
    assert schema3.child_workitem_type == "TaskItem"
    assert schema3.min_cardinality == 1
    assert schema3.max_cardinality == 5


def test_child_schema_validation():
    """Test ChildSchema validation."""
    # Test empty type
    try:
        WorkItem.ChildSchema("")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "child_workitem_type cannot be empty or contain spaces" in str(e)
    
    # Test spaces in type
    try:
        WorkItem.ChildSchema("Task Item")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "child_workitem_type cannot be empty or contain spaces" in str(e)
    
    # Test negative min cardinality
    try:
        WorkItem.ChildSchema("TaskItem", -1)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "min_cardinality cannot be negative" in str(e)
    
    # Test negative max cardinality
    try:
        WorkItem.ChildSchema("TaskItem", 0, -1)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "max_cardinality cannot be negative" in str(e)
    
    # Test min > max
    try:
        WorkItem.ChildSchema("TaskItem", 5, 3)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "min_cardinality cannot be greater than max_cardinality" in str(e)


def test_workflow_schema_basic():
    """Test basic WorkflowSchema functionality."""
    # Test minimal initialization
    schema = WorkItem.WorkflowSchema()
    assert schema.workflow_types is None
    assert schema.child_schema_dict is None
    
    # Test with types
    types = ["Project", "Task", "Subtask"]
    schema2 = WorkItem.WorkflowSchema(workflow_types=types)
    assert schema2.workflow_types == types
    assert schema2.child_schema_dict is None
    
    # Test create_workitem
    workitem = schema2.create_workitem("Project")
    assert workitem.workitem_type == "Project"
    assert workitem._workflow_schema == schema2


def test_workflow_schema_validation():
    """Test WorkflowSchema validation."""
    types = ["Project", "Task"]
    
    # Test invalid child schema key
    try:
        child_schemas = {
            "Project": [WorkItem.ChildSchema("Task", 1, 10)],
            "InvalidType": [WorkItem.ChildSchema("Task", 0, 5)]
        }
        WorkItem.WorkflowSchema(workflow_types=types, child_schema_dict=child_schemas)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Child schema key 'InvalidType' not found in workflow_types" in str(e)
    
    # Test invalid child workitem type
    try:
        child_schemas = {
            "Project": [WorkItem.ChildSchema("InvalidType", 1, 10)]
        }
        WorkItem.WorkflowSchema(workflow_types=types, child_schema_dict=child_schemas)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Child workitem type 'InvalidType' not found in workflow_types" in str(e)


def test_workitem_basic():
    """Test basic WorkItem functionality."""
    # Test minimal initialization
    workitem = WorkItem()
    assert workitem.workitem_type == "WorkItem"
    assert not workitem.is_complete
    assert workitem.parent is None
    
    # Test with type
    workitem2 = WorkItem(workitem_type="Task")
    assert workitem2.workitem_type == "Task"
    assert not workitem2.is_complete
    
    # Test properties
    workitem2.is_complete = True
    assert workitem2.is_complete
    workitem2.is_complete = False
    assert not workitem2.is_complete
    
    # Test data dictionary
    workitem2["title"] = "Test Task"
    workitem2["priority"] = "High"
    assert workitem2["title"] == "Test Task"
    assert workitem2["priority"] == "High"
    
    # Test clear data dict
    workitem2.clear_data_dict()
    try:
        _ = workitem2["title"]
        assert False, "Should have raised KeyError"
    except KeyError:
        pass


def test_workitem_validation():
    """Test WorkItem validation."""
    # Test spaces in type
    try:
        WorkItem(workitem_type="Task Item")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "workitem_type cannot contain spaces" in str(e)


def test_child_management():
    """Test child management functionality."""
    parent = WorkItem(workitem_type="Project")
    child1 = WorkItem(workitem_type="Task")
    child2 = WorkItem(workitem_type="Task")
    
    # Test adding children
    parent.add_child(child1)
    parent.add_child(child2)
    
    assert parent.get_child_count() == 2
    assert parent.get_child(0) == child1
    assert parent.get_child(1) == child2
    assert child1.parent == parent
    assert child2.parent == parent
    
    # Test child index
    assert parent.index_of(child1) == 0
    assert parent.index_of(child2) == 1
    
    # Test removing children
    parent.remove_child(child1)
    assert parent.get_child_count() == 1
    assert parent.get_child(0) == child2
    assert parent.index_of(child2) == 0


def test_schema_enforcement():
    """Test schema enforcement for child management."""
    schema = WorkItem.WorkflowSchema(
        workflow_types=["Project", "Task"],
        child_schema_dict={
            "Project": [WorkItem.ChildSchema("Task", 0, 2)],
            "Task": None
        }
    )
    
    parent = WorkItem(workitem_type="Project", workflow_schema=schema)
    
    # Test adding valid children
    task1 = WorkItem(workitem_type="Task")
    task2 = WorkItem(workitem_type="Task")
    parent.add_child(task1)
    parent.add_child(task2)
    assert parent.get_child_count() == 2
    
    # Test exceeding max cardinality
    task3 = WorkItem(workitem_type="Task")
    try:
        parent.add_child(task3)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Cannot add more than 2 children of type 'Task'" in str(e)
    
    # Test invalid child type
    bug = WorkItem(workitem_type="Bug")
    try:
        parent.add_child(bug)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Child type 'Bug' not allowed for parent type 'Project'" in str(e)


def test_min_cardinality():
    """Test minimum cardinality enforcement."""
    schema = WorkItem.WorkflowSchema(
        workflow_types=["Project", "Task"],
        child_schema_dict={
            "Project": [WorkItem.ChildSchema("Task", 2, 5)],
            "Task": None
        }
    )
    
    # Should create 2 Task children automatically
    parent = WorkItem(workitem_type="Project", workflow_schema=schema)
    assert parent.get_child_count("Task") == 2
    
    # Try to remove a child - should fail due to min cardinality
    child = parent.get_child(0, "Task")
    try:
        parent.remove_child(child)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Cannot remove child; would violate min_cardinality=2" in str(e)


def test_path_operations():
    """Test path and navigation operations."""
    root = WorkItem(workitem_type="Project")
    task = WorkItem(workitem_type="Task")
    subtask = WorkItem(workitem_type="Subtask")
    
    root.add_child(task)
    task.add_child(subtask)
    
    # Test absolute paths
    assert root.get_absolute_path() == "/"
    assert task.get_absolute_path() == "/Task[0]"
    assert subtask.get_absolute_path() == "/Task[0]/Subtask[0]"
    
    # Test path navigation
    assert root._get_child_workitem("Task[0]") == task
    assert root._get_child_workitem("Task") == task  # Default index 0
    assert root._get_child_workitem("Task[0]/Subtask[0]") == subtask
    
    # Test absolute path navigation
    assert root.get_workitem("/") == root
    assert root.get_workitem("/Task[0]") == task
    assert subtask.get_workitem("/") == root


def main():
    """Run all tests."""
    tests = [
        (test_child_schema_basic, "ChildSchema Basic"),
        (test_child_schema_validation, "ChildSchema Validation"),
        (test_workflow_schema_basic, "WorkflowSchema Basic"),
        (test_workflow_schema_validation, "WorkflowSchema Validation"),
        (test_workitem_basic, "WorkItem Basic"),
        (test_workitem_validation, "WorkItem Validation"),
        (test_child_management, "Child Management"),
        (test_schema_enforcement, "Schema Enforcement"),
        (test_min_cardinality, "Min Cardinality"),
        (test_path_operations, "Path Operations"),
    ]
    
    passed = 0
    failed = 0
    
    print("Running WorkItem tests...")
    print("=" * 50)
    
    for test_func, test_name in tests:
        if run_test(test_func, test_name):
            passed += 1
        else:
            failed += 1
    
    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed > 0:
        sys.exit(1)
    else:
        print("All tests passed! ✓")


if __name__ == "__main__":
    main()
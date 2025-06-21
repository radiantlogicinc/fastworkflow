import os
import tempfile
import json
from pathlib import Path

import pytest

from fastworkflow.build.command_file_generator import verify_commands_against_context_model, EXCLUDE_DIRS
from fastworkflow.build.class_analysis_structures import ClassInfo, MethodInfo, PropertyInfo


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def commands_dir(temp_dir):
    """Create a commands directory with context subdirectories."""
    commands_dir = temp_dir / "_commands"
    commands_dir.mkdir(exist_ok=True)
    
    # Create context directories
    (commands_dir / "TodoList").mkdir(exist_ok=True)
    (commands_dir / "TodoItem").mkdir(exist_ok=True)
    (commands_dir / "User").mkdir(exist_ok=True)
    (commands_dir / "BaseList").mkdir(exist_ok=True)
    
    # Create some command files
    (commands_dir / "TodoList" / "get_properties.py").write_text("# Test command file")
    (commands_dir / "TodoItem" / "get_properties.py").write_text("# Test command file")
    (commands_dir / "User" / "get_properties.py").write_text("# Test command file")
    (commands_dir / "BaseList" / "get_properties.py").write_text("# Test command file")
    
    # Create special files that should be ignored
    (commands_dir / "README.md").write_text("# Test README")
    (commands_dir / "context_inheritance_model.json").write_text("{}")
    (commands_dir / "startup.py").write_text("# Test startup file")
    (commands_dir / "__init__.py").write_text("")
    
    return commands_dir


@pytest.fixture
def valid_context_model():
    """Create a valid context model with the new schema format."""
    return {
        "inheritance": {
            "TodoList": {"base": ["BaseList"]},
            "TodoItem": {"base": []},
            "User": {"base": []},
            "BaseList": {"base": []},
            "*": {"base": []}
        },
        "aggregation": {
            "TodoItem": {"container": ["TodoList"]}
        }
    }


@pytest.fixture
def classes_info():
    """Create ClassInfo objects for testing."""
    # Create TodoList class
    todo_list = ClassInfo("TodoList", "application/todo_list.py", bases=["BaseList"])
    todo_list.methods.append(MethodInfo("get_items", [], docstring="Get all items"))
    classes = {"TodoList": todo_list}
    # Create TodoItem class
    todo_item = ClassInfo("TodoItem", "application/todo_item.py")
    todo_item.methods.append(MethodInfo("complete", [], docstring="Mark as complete"))
    classes["TodoItem"] = todo_item

    # Create User class
    user = ClassInfo("User", "application/user.py")
    user.methods.append(MethodInfo("login", [], docstring="Login"))
    classes["User"] = user

    # Create BaseList class
    base_list = ClassInfo("BaseList", "application/base_list.py")
    base_list.methods.append(MethodInfo("base_method", [], docstring="Base method"))
    classes["BaseList"] = base_list

    return classes


def test_verify_valid_context_model(commands_dir, valid_context_model, classes_info):
    """Test that a valid context model passes verification."""
    errors = verify_commands_against_context_model(valid_context_model, commands_dir, classes_info)
    assert not errors, f"Expected no errors, but got: {errors}"


def test_missing_inheritance_block(commands_dir, classes_info):
    """Test that a context model without an inheritance block fails verification."""
    invalid_model = {
        "aggregation": {}
    }
    errors = verify_commands_against_context_model(invalid_model, commands_dir, classes_info)
    assert len(errors) == 1
    assert "missing 'inheritance' block" in errors[0].lower()


def test_missing_directory_for_context(commands_dir, valid_context_model, classes_info):
    """Test that a context in the model without a corresponding directory fails verification."""
    # Add a context to the model that doesn't have a directory
    valid_context_model["inheritance"]["MissingContext"] = {"base": []}
    
    # Add the missing context to classes_info so we only test the directory check
    classes_info["MissingContext"] = ClassInfo("MissingContext", "application/missing_context.py")
    
    errors = verify_commands_against_context_model(valid_context_model, commands_dir, classes_info)
    assert len(errors) == 1
    assert "MissingContext" in errors[0]
    assert "no directory" in errors[0].lower()


def test_directory_without_context(commands_dir, valid_context_model, classes_info):
    """Test that a directory without a corresponding context in the model fails verification."""
    # Create a directory that doesn't have a context in the model
    (commands_dir / "ExtraContext").mkdir(exist_ok=True)
    
    errors = verify_commands_against_context_model(valid_context_model, commands_dir, classes_info)
    assert len(errors) == 1
    assert "ExtraContext" in errors[0]
    assert "does not correspond to any known class" in errors[0].lower()


def test_invalid_base_class(commands_dir, valid_context_model, classes_info):
    """Test that a context with an invalid base class fails verification."""
    # Add InvalidBase to the model to avoid the directory check error
    valid_context_model["inheritance"]["InvalidBase"] = {"base": []}
    
    # Create a directory for InvalidBase to avoid the directory check error
    (commands_dir / "InvalidBase").mkdir(exist_ok=True)
    
    # Modify the model to include an invalid base class
    valid_context_model["inheritance"]["TodoList"]["base"] = ["InvalidBase"]
    
    # We expect two errors:
    # 1. InvalidBase is in the model but not in classes_info
    # 2. TodoList inherits from InvalidBase, but InvalidBase is not in classes_info
    errors = verify_commands_against_context_model(valid_context_model, commands_dir, classes_info)
    
    # Check that there are exactly two errors
    assert len(errors) == 2
    
    # Check that both errors mention InvalidBase
    assert all("InvalidBase" in error for error in errors)
    
    # Check that one error is about TodoList inheriting from InvalidBase
    assert any("TodoList" in error and "inherits from" in error for error in errors)


def test_context_not_in_classes_info(commands_dir, valid_context_model, classes_info):
    """Test that a context in the model that's not in classes_info fails verification."""
    # Add a context to the model that's not in classes_info
    valid_context_model["inheritance"]["UnknownClass"] = {"base": []}
    
    # Create a directory for UnknownClass to avoid the directory check error
    (commands_dir / "UnknownClass").mkdir(exist_ok=True)
    
    errors = verify_commands_against_context_model(valid_context_model, commands_dir, classes_info)
    assert len(errors) == 1  # Only the "not found in analyzed class information" error
    assert "UnknownClass" in errors[0]
    assert "not found in analyzed class information" in errors[0]


def test_global_context_special_handling(commands_dir, valid_context_model, classes_info):
    """Test that the global context (*) is handled specially."""
    # We don't need to create a directory for the global context
    # The test should pass because the global context doesn't need a directory
    errors = verify_commands_against_context_model(valid_context_model, commands_dir, classes_info)

    # Check that there are no errors related to the global context
    assert all(
        "'*'" not in error for error in errors
    ), f"Found errors related to global context: {errors}"


def test_excluded_directories_ignored(commands_dir, valid_context_model, classes_info):
    """Test that directories in EXCLUDE_DIRS are ignored."""
    for excluded_dir in EXCLUDE_DIRS:
        (commands_dir / excluded_dir).mkdir(exist_ok=True)
    
    # These should not cause errors because they're in EXCLUDE_DIRS
    errors = verify_commands_against_context_model(valid_context_model, commands_dir, classes_info)
    assert not errors, f"Expected no errors, but got: {errors}"


def test_special_files_ignored(commands_dir, valid_context_model, classes_info):
    """Test that special files like README.md, context_inheritance_model.json, etc. are ignored."""
    # Add more special files
    (commands_dir / "_special_file.py").write_text("# Should be ignored")
    
    # These should not cause errors because they're special files
    errors = verify_commands_against_context_model(valid_context_model, commands_dir, classes_info)
    assert not errors, f"Expected no errors, but got: {errors}" 
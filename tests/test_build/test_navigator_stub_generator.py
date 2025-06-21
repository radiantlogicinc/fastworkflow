import os
import json
import tempfile
import pytest
from pathlib import Path

from fastworkflow.build.navigator_stub_generator import NavigatorStubGenerator


def create_test_context_model(path, model_data):
    """Helper to create a test context model file."""
    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(model_data, f, indent=2)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def navigators_dir(temp_dir):
    """Create a navigators directory for testing."""
    navigators_dir = temp_dir / "navigators"
    navigators_dir.mkdir(exist_ok=True)
    return navigators_dir


@pytest.fixture
def basic_context_model(temp_dir):
    """Create a basic context model for testing."""
    model_path = temp_dir / "_commands/context_inheritance_model.json"
    model_data = {
        "inheritance": {
            "TodoList": {"base": []},
            "TodoItem": {"base": ["TodoList"]},
            "User": {"base": ["*"]},
            "Project": {"base": ["TodoList"]},
            "*": {"base": []}
        },
        "aggregation": {
            "TodoItem": {"container": ["TodoList"]},
            "TodoList": {"container": ["Project"]},
            "Project": {"container": ["User"]}
        }
    }
    create_test_context_model(model_path, model_data)
    return model_path


@pytest.fixture
def complex_context_model(temp_dir):
    """Create a more complex context model for testing."""
    model_path = temp_dir / "complex_context_model.json"
    model_data = {
        "inheritance": {
            "TodoList": {"base": []},
            "TodoItem": {"base": ["TodoList"]},
            "User": {"base": ["*"]},
            "Project": {"base": ["TodoList"]},
            "Task": {"base": ["TodoItem", "Project"]},  # Multiple inheritance
            "SubTask": {"base": ["Task"]},  # Nested inheritance
            "*": {"base": []}
        },
        "aggregation": {
            "TodoItem": {"container": ["TodoList"]},
            "TodoList": {"container": ["Project"]},
            "Project": {"container": ["User"]},
            "Task": {"container": ["Project", "User"]},  # Multiple containers
            "SubTask": {"container": ["Task"]}  # Nested containers
        }
    }
    create_test_context_model(model_path, model_data)
    return model_path


def test_get_parent_contexts(basic_context_model):
    """Test getting parent contexts from the model."""
    generator = NavigatorStubGenerator(
        model_path=basic_context_model
    )
    
    # Test global context (no parents)
    assert generator.get_parent_contexts('*') == {"inheritance": [], "aggregation": []}
    
    # Test context with inheritance to global
    assert generator.get_parent_contexts('User') == {"inheritance": ["*"], "aggregation": []}
    
    # Test context with inheritance to another context
    assert generator.get_parent_contexts('TodoItem') == {"inheritance": ["TodoList"], "aggregation": ["TodoList"]}
    
    # Test context with both inheritance and aggregation
    assert generator.get_parent_contexts('TodoList') == {"inheritance": [], "aggregation": ["Project"]}
    
    # Test context with complex relationships
    assert generator.get_parent_contexts('Project') == {"inheritance": ["TodoList"], "aggregation": ["User"]}


def test_get_navigator_file_path(navigators_dir):
    """Test getting the file path for a navigator."""
    generator = NavigatorStubGenerator(
        navigators_root=navigators_dir
    )
    
    # Test for TodoList context
    assert generator.get_navigator_file_path('TodoList') == navigators_dir / "todolist_navigator.py"
    
    # Test for camelCase context name
    assert generator.get_navigator_file_path('TodoItem') == navigators_dir / "todoitem_navigator.py"


def test_check_file_exists(temp_dir):
    """Test checking if a file exists."""
    # Create a file
    file_path = temp_dir / "test_file.py"
    with open(file_path, 'w') as f:
        f.write("# Test content")
    
    # Create an empty file
    empty_file_path = temp_dir / "empty_file.py"
    with open(empty_file_path, 'w') as f:
        pass
    
    # Create a directory with the same name as a file
    dir_path = temp_dir / "dir_file.py"
    dir_path.mkdir()
    
    generator = NavigatorStubGenerator()
    
    # Test file with content
    exists, reason = generator.check_file_exists(file_path)
    assert exists
    assert "with content" in reason
    
    # Test empty file
    exists, reason = generator.check_file_exists(empty_file_path)
    assert exists
    assert "empty" in reason
    
    # Test directory
    exists, reason = generator.check_file_exists(dir_path)
    assert exists
    assert "not a file" in reason
    
    # Test non-existent file
    exists, reason = generator.check_file_exists(temp_dir / "non_existent.py")
    assert not exists
    assert "does not exist" in reason


def test_generate_navigator_stub(navigators_dir, basic_context_model):
    """Test generating a navigator stub for a specific context."""
    generator = NavigatorStubGenerator(
        navigators_root=navigators_dir,
        model_path=basic_context_model
    )
    
    # Generate a navigator stub for TodoList context
    file_path = generator.generate_navigator_stub('TodoList')
    
    # Check that the file was created
    assert file_path.exists()
    
    # Check that the file contains the expected content
    content = file_path.read_text()
    assert 'class TodoListNavigator(ContextExpander):' in content
    assert 'def move_to_parent_context(self, snapshot: WorkflowSnapshot)' in content
    assert 'Project' in content  # Should mention container context
    
    # Test generating a stub for global context (should return None)
    assert generator.generate_navigator_stub('*') is None


def test_generate_navigator_stub_with_inheritance(navigators_dir, basic_context_model):
    """Test generating a navigator stub for a context with inheritance."""
    generator = NavigatorStubGenerator(
        navigators_root=navigators_dir,
        model_path=basic_context_model
    )
    
    # Generate a navigator stub for TodoItem context (inherits from TodoList)
    file_path = generator.generate_navigator_stub('TodoItem')
    
    # Check that the file was created
    assert file_path.exists()
    
    # Check that the file contains the expected content
    content = file_path.read_text()
    assert 'TodoList' in content  # Should mention base context


def test_generate_navigator_stub_with_global_inheritance(navigators_dir, basic_context_model):
    """Test generating a navigator stub for a context that inherits from global."""
    generator = NavigatorStubGenerator(
        navigators_root=navigators_dir,
        model_path=basic_context_model
    )
    
    # Generate a navigator stub for User context (inherits from *)
    file_path = generator.generate_navigator_stub('User')
    
    # Check that the file was created
    assert file_path.exists()
    
    # Check that the file contains the expected content
    content = file_path.read_text()
    assert 'snapshot.clear_context()' in content  # Should reset to global context


def test_generate_navigator_stub_existing_file(navigators_dir):
    """Test generating a navigator stub when the file already exists."""
    generator = NavigatorStubGenerator(
        navigators_root=navigators_dir
    )
    
    # Create the file first
    file_path = navigators_dir / "existing_navigator.py"
    with open(file_path, 'w') as f:
        f.write("# Existing content")
    
    # Try to generate a navigator stub
    result = generator.generate_navigator_stub('Existing')
    
    # Check that no file was returned
    assert result is None
    
    # Check that the file content was not changed
    content = file_path.read_text()
    assert content == "# Existing content"


def test_generate_navigator_stub_force_overwrite(navigators_dir):
    """Test generating a navigator stub with force overwrite."""
    generator = NavigatorStubGenerator(
        navigators_root=navigators_dir
    )
    
    # Create the file first
    file_path = navigators_dir / "force_navigator.py"
    with open(file_path, 'w') as f:
        f.write("# Existing content")
    
    # Try to generate a navigator stub with force=True
    result = generator.generate_navigator_stub('Force', force=True)
    
    # Check that a file was returned
    assert result is not None
    
    # Check that the file content was changed
    content = file_path.read_text()
    assert "# Existing content" not in content
    assert 'class ForceNavigator(ContextExpander):' in content


def test_generate_navigator_stubs(navigators_dir, basic_context_model):
    """Test generating navigator stubs for all contexts."""
    generator = NavigatorStubGenerator(
        navigators_root=navigators_dir,
        model_path=basic_context_model
    )
    
    # Generate navigator stubs for all contexts
    files = generator.generate_navigator_stubs()
    
    # Check that the files were created (excluding global context)
    assert len(files) == 4  # TodoList, TodoItem, User, Project
    
    # Check that the files are in the navigators directory
    for file_path in files.values():
        assert file_path.parent == navigators_dir


def test_complex_inheritance_and_aggregation(navigators_dir, complex_context_model):
    """Test generating navigator stubs for contexts with complex relationships."""
    generator = NavigatorStubGenerator(
        navigators_root=navigators_dir,
        model_path=complex_context_model
    )
    
    # Generate a navigator stub for Task context (multiple inheritance and containers)
    file_path = generator.generate_navigator_stub('Task')
    
    # Check that the file was created
    assert file_path.exists()
    
    # Check that the file contains the expected content
    content = file_path.read_text()
    assert 'TodoItem' in content  # Should mention first base context
    assert 'Project' in content  # Should mention second base context
    assert 'User' in content  # Should mention first container context
    
    # Generate a navigator stub for SubTask context (nested inheritance and containers)
    file_path = generator.generate_navigator_stub('SubTask')
    
    # Check that the file was created
    assert file_path.exists()
    
    # Check that the file contains the expected content
    content = file_path.read_text()
    assert 'Task' in content  # Should mention base context and container context 
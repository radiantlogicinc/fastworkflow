import os
import json
import tempfile
import pytest
from pathlib import Path

from fastworkflow.build.context_folder_generator import ContextFolderGenerator
from fastworkflow.context_model_loader import ContextModelLoaderError


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
def basic_context_model(temp_dir):
    """Create a basic context model for testing."""
    model_path = temp_dir / "_commands/context_inheritance_model.json"
    model_data = {
        "TodoList": {"base": []},
        "TodoItem": {"base": []},
        "User": {"base": []},
        "*": {"base": []}
    }
    create_test_context_model(model_path, model_data)
    return model_path


@pytest.fixture
def inheritance_context_model(temp_dir):
    """Create a context model with inheritance relationships for testing."""
    model_path = temp_dir / "_commands/context_inheritance_model.json"
    model_data = {
        "TodoList": {"base": ["BaseList"]},
        "TodoItem": {"base": ["BaseItem"]},
        "BaseList": {"base": []},
        "BaseItem": {"base": []},
        "*": {"base": []}
    }
    create_test_context_model(model_path, model_data)
    return model_path


@pytest.fixture
def commands_dir(temp_dir):
    """Create a commands directory for testing."""
    commands_dir = temp_dir / "_commands"
    commands_dir.mkdir(exist_ok=True)
    return commands_dir


def test_basic_folder_creation(temp_dir, basic_context_model, commands_dir):
    """Test basic folder creation functionality."""
    # Create the generator
    generator = ContextFolderGenerator(
        commands_root=commands_dir,
        model_path=basic_context_model
    )
    
    # Generate folders
    created_folders = generator.generate_folders()
    
    # Check that the expected folders were created
    assert len(created_folders) == 3  # TodoList, TodoItem, User
    assert (commands_dir / "TodoList").exists()
    assert (commands_dir / "TodoItem").exists()
    assert (commands_dir / "User").exists()
    
    # Verify the global context folder was not created
    assert not (commands_dir / "*").exists()


def test_handler_file_creation(temp_dir, basic_context_model, commands_dir):
    """Test that _<ContextName>.py handler files are created."""
    generator = ContextFolderGenerator(
        commands_root=commands_dir,
        model_path=basic_context_model
    )

    # Generate folders and handler files
    generator.generate_folders()

    # Check that handler files were created
    assert (commands_dir / "TodoList" / "_TodoList.py").exists()
    assert (commands_dir / "TodoItem" / "_TodoItem.py").exists()
    assert (commands_dir / "User" / "_User.py").exists()

    # Check content of handler files
    with open(commands_dir / "TodoList" / "_TodoList.py", 'r') as f:
        content = f.read()
        # Verify imports
        assert "from typing import Optional" in content
        assert "from ...application.todolist import TodoList" in content
        # Verify Context class with get_parent method
        assert "class Context:" in content
        assert "@classmethod" in content
        assert "def get_parent(cls, command_context_object: TodoList)" in content
        assert "return getattr(command_context_object, 'parent', None)" in content

    # Check TodoItem handler file
    with open(commands_dir / "TodoItem" / "_TodoItem.py", 'r') as f:
        content = f.read()
        assert "from typing import Optional" in content
        assert "from ...application.todoitem import TodoItem" in content
        assert "class Context:" in content
        assert "@classmethod" in content
        assert "def get_parent(cls, command_context_object: TodoItem)" in content
        assert "return getattr(command_context_object, 'parent', None)" in content


def test_handler_file_with_inheritance(temp_dir, inheritance_context_model, commands_dir):
    """Test that handler files correctly reference parent classes from inheritance."""
    generator = ContextFolderGenerator(
        commands_root=commands_dir,
        model_path=inheritance_context_model
    )
    
    # Generate folders and handler files
    generator.generate_folders()
    
    # Check TodoList handler file with inheritance
    with open(commands_dir / "TodoList" / "_TodoList.py", 'r') as f:
        content = f.read()
        # Verify imports include the parent class
        assert "from ...application.todolist import TodoList" in content
        assert "from ...application.baselist import BaseList" in content
        # Verify return type annotation includes parent class
        assert "def get_parent(cls, command_context_object: TodoList) -> Optional[BaseList]:" in content
        assert "return getattr(command_context_object, 'parent', None)" in content
    
    # Check TodoItem handler file with inheritance
    with open(commands_dir / "TodoItem" / "_TodoItem.py", 'r') as f:
        content = f.read()
        # Verify imports include the parent class
        assert "from ...application.todoitem import TodoItem" in content
        assert "from ...application.baseitem import BaseItem" in content
        # Verify return type annotation includes parent class
        assert "def get_parent(cls, command_context_object: TodoItem) -> Optional[BaseItem]:" in content
        assert "return getattr(command_context_object, 'parent', None)" in content


def test_idempotence(temp_dir, basic_context_model, commands_dir):
    """Test that running the generator multiple times doesn't cause errors."""
    generator = ContextFolderGenerator(
        commands_root=commands_dir,
        model_path=basic_context_model
    )
    
    # Run the generator twice
    generator.generate_folders()
    created_folders = generator.generate_folders()
    
    # Check that the expected folders still exist
    assert len(created_folders) == 3
    assert (commands_dir / "TodoList").exists()
    assert (commands_dir / "TodoItem").exists()
    assert (commands_dir / "User").exists()


def test_handler_file_idempotence(temp_dir, basic_context_model, commands_dir):
    """Test that handler files are not overwritten if they already exist."""
    generator = ContextFolderGenerator(
        commands_root=commands_dir,
        model_path=basic_context_model
    )
    
    # Generate folders and handler files
    generator.generate_folders()
    
    # Modify a handler file
    custom_content = """from typing import Optional
from ...application.todolist import TodoList

class Context:
    @classmethod
    def get_parent(cls, command_context_object: TodoList) -> Optional[object]:
        # Custom implementation
        return command_context_object.custom_parent
"""
    with open(commands_dir / "TodoList" / "_TodoList.py", 'w') as f:
        f.write(custom_content)
    
    # Run the generator again
    generator.generate_folders()
    
    # Check that the custom content was preserved
    with open(commands_dir / "TodoList" / "_TodoList.py", 'r') as f:
        content = f.read()
        assert "# Custom implementation" in content
        assert "return command_context_object.custom_parent" in content


def test_missing_model_file(temp_dir, commands_dir):
    """Test handling of a missing model file."""
    non_existent_model = temp_dir / "non_existent_model.json"
    
    generator = ContextFolderGenerator(
        commands_root=commands_dir,
        model_path=non_existent_model
    )
    
    # The generator should raise an exception when the model file doesn't exist
    with pytest.raises(Exception):
        generator.generate_folders()


def test_invalid_model_file(temp_dir, commands_dir):
    """Test handling of an invalid model file."""
    invalid_model_path = temp_dir / "invalid_model.json"
    
    # Create an invalid JSON file
    with open(invalid_model_path, 'w') as f:
        f.write("{invalid json")
    
    generator = ContextFolderGenerator(
        commands_root=commands_dir,
        model_path=invalid_model_path
    )
    
    # The generator should raise an exception when the model file is invalid
    with pytest.raises(Exception):
        generator.generate_folders()


def test_empty_model(temp_dir, commands_dir):
    """Test handling of an empty model."""
    empty_model_path = temp_dir / "empty_model.json"
    
    # Create an empty model with just the required structure
    create_test_context_model(empty_model_path, {
        "*": {"base": []}
    })
    
    generator = ContextFolderGenerator(
        commands_root=commands_dir,
        model_path=empty_model_path
    )
    
    # An empty model is valid in the new schema; no folders should be generated
    created = generator.generate_folders()
    assert created == {}


def test_special_characters_in_context_names(temp_dir):
    """Test handling of special characters in context names."""
    model_path = temp_dir / "special_chars_model.json"
    commands_dir = temp_dir / "_commands"
    
    # Create a model with special characters in context names
    model_data = {
        "Class-With-Hyphens": {"base": []},
        "Class_With_Underscores": {"base": []},
        "Class.With.Dots": {"base": []},
        "*": {"base": []}
    }
    create_test_context_model(model_path, model_data)
    
    generator = ContextFolderGenerator(
        commands_root=commands_dir,
        model_path=model_path
    )
    
    # Generate folders
    created_folders = generator.generate_folders()
    
    # Check that the expected folders were created
    assert len(created_folders) == 3
    assert (commands_dir / "Class-With-Hyphens").exists()
    assert (commands_dir / "Class_With_Underscores").exists()
    assert (commands_dir / "Class.With.Dots").exists()


def test_model_without_inheritance(temp_dir, commands_dir):
    """Test handling of a model without an inheritance block."""
    model_path = temp_dir / "no_inheritance_model.json"
    
    # Create a model without an inheritance block
    create_test_context_model(model_path, {
    })
    
    generator = ContextFolderGenerator(
        commands_root=commands_dir,
        model_path=model_path
    )
    
    # An empty model is valid in the new schema; no folders should be generated
    created = generator.generate_folders()
    assert created == {} 
import os
import json
import tempfile
import pytest
from pathlib import Path

from fastworkflow.build.context_folder_generator import ContextFolderGenerator
from fastworkflow.context_model_loader import ContextModelLoaderError


def create_test_context_model(path, model_data):
    """Helper to create a test context model file."""
    with open(path, 'w') as f:
        json.dump(model_data, f)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def basic_context_model(temp_dir):
    """Create a basic context model for testing."""
    model_path = temp_dir / "command_context_model.json"
    model_data = {
        "inheritance": {
            "TodoList": {"base": []},
            "TodoItem": {"base": []},
            "User": {"base": []},
            "*": {"base": []}
        },
        "aggregation": {
            "TodoItem": {"container": ["TodoList"]}
        }
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
        "inheritance": {
            "*": {"base": []}
        }
    })
    
    generator = ContextFolderGenerator(
        commands_root=commands_dir,
        model_path=empty_model_path
    )
    
    # The generator should not create any folders
    created_folders = generator.generate_folders()
    assert len(created_folders) == 0


def test_special_characters_in_context_names(temp_dir):
    """Test handling of special characters in context names."""
    model_path = temp_dir / "special_chars_model.json"
    commands_dir = temp_dir / "_commands"
    
    # Create a model with special characters in context names
    model_data = {
        "inheritance": {
            "Class-With-Hyphens": {"base": []},
            "Class_With_Underscores": {"base": []},
            "Class.With.Dots": {"base": []},
            "*": {"base": []}
        }
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
        "aggregation": {
            "TodoItem": {"container": ["TodoList"]}
        }
    })
    
    generator = ContextFolderGenerator(
        commands_root=commands_dir,
        model_path=model_path
    )
    
    # The generator should raise an exception when the model lacks an inheritance block
    with pytest.raises(ContextModelLoaderError):
        generator.generate_folders() 
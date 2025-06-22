import os
import json
import tempfile
import pytest
from pathlib import Path

from fastworkflow.build.inheritance_block_regenerator import InheritanceBlockRegenerator
from fastworkflow.build.class_analysis_structures import ClassInfo


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
def commands_dir(temp_dir):
    """Create a commands directory for testing."""
    commands_dir = temp_dir / "_commands"
    commands_dir.mkdir(exist_ok=True)
    return commands_dir


@pytest.fixture
def context_dirs(commands_dir):
    """Create context directories for testing."""
    (commands_dir / "TodoList").mkdir(exist_ok=True)
    (commands_dir / "TodoItem").mkdir(exist_ok=True)
    (commands_dir / "User").mkdir(exist_ok=True)
    return commands_dir


@pytest.fixture
def existing_model():
    """Create a temporary context model file with existing entries."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        commands_dir = Path(tmp_dir) / "_commands"
        commands_dir.mkdir(exist_ok=True)
        
        # Create a model file with the new flat structure
        model_data = {
            "OldClass": {"base": []},
            "AnotherOldClass": {"base": ["OldClass"]}
        }
        
        model_path = commands_dir / "context_inheritance_model.json"
        with open(model_path, "w") as f:
            json.dump(model_data, f, indent=2)
        
        yield model_path


@pytest.fixture
def mock_classes():
    """Create mock class information for testing."""
    # Create TodoList class
    todo_list = ClassInfo(
        name="TodoList",
        module_path="app/models/todo_list.py",
        bases=["BaseModel"]
    )
    classes = {"TodoList": todo_list}
    # Create TodoItem class
    todo_item = ClassInfo(
        name="TodoItem",
        module_path="app/models/todo_item.py",
        bases=["BaseModel"]
    )
    classes["TodoItem"] = todo_item

    # Create User class with inheritance
    user = ClassInfo(
        name="User",
        module_path="app/models/user.py",
        bases=["BaseModel", "TodoList"]
    )
    classes["User"] = user

    # Create BaseModel class
    base_model = ClassInfo(
        name="BaseModel",
        module_path="app/models/base.py",
        bases=[]
    )
    classes["BaseModel"] = base_model

    return classes


def test_scan_contexts(context_dirs):
    """Test scanning contexts from directory structure."""
    regenerator = InheritanceBlockRegenerator(commands_root=context_dirs)
    contexts = regenerator.scan_contexts()
    
    assert len(contexts) == 3
    assert "TodoList" in contexts
    assert "TodoItem" in contexts
    assert "User" in contexts


def test_load_existing_model(existing_model):
    """Test loading an existing context model."""
    regenerator = InheritanceBlockRegenerator(
        commands_root=existing_model.parent,
        model_path=existing_model
    )
    
    model = regenerator.load_existing_model()
    
    # Check that the model was loaded correctly
    assert "OldClass" in model
    assert "AnotherOldClass" in model
    assert model["AnotherOldClass"]["base"] == ["OldClass"]


def test_load_missing_model():
    """Test loading a non-existent context model."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = Path(tmp_dir) / "nonexistent.json"
        regenerator = InheritanceBlockRegenerator(
            commands_root=Path(tmp_dir),
            model_path=model_path
        )
        
        model = regenerator.load_existing_model()
        
        # Should return an empty dict
        assert model == {}


def test_load_invalid_model():
    """Test loading an invalid context model."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = Path(tmp_dir) / "invalid.json"
        with open(model_path, "w") as f:
            f.write("invalid json")
        
        regenerator = InheritanceBlockRegenerator(
            commands_root=Path(tmp_dir),
            model_path=model_path
        )
        
        model = regenerator.load_existing_model()
        
        # Should return an empty dict
        assert model == {}


def test_regenerate_inheritance_with_directory_scan(context_dirs):
    """Test regenerating inheritance block by scanning directories."""
    regenerator = InheritanceBlockRegenerator(
        commands_root=context_dirs,
        model_path=context_dirs / "context_inheritance_model.json"
    )
    
    model = regenerator.regenerate_inheritance()
    
    # Check that the model includes entries for each context directory
    assert "TodoList" in model
    assert "TodoItem" in model
    assert "User" in model
    
    # Each entry should have a base field
    assert "base" in model["TodoList"]
    assert "base" in model["TodoItem"]
    assert "base" in model["User"]


def test_regenerate_inheritance_with_classes():
    """Test regenerating inheritance block with class information."""
    # Create some class info objects
    base_model = ClassInfo("BaseModel", "base.py")
    todo_item = ClassInfo("TodoItem", "todo_item.py", bases=["BaseModel"])
    todo_list = ClassInfo("TodoList", "todo_list.py", bases=["BaseModel"])
    
    classes = {
        "BaseModel": base_model,
        "TodoItem": todo_item,
        "TodoList": todo_list
    }
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        commands_dir = Path(tmp_dir) / "_commands"
        commands_dir.mkdir(exist_ok=True)
        
        regenerator = InheritanceBlockRegenerator(
            commands_root=commands_dir,
            model_path=commands_dir / "context_inheritance_model.json"
        )
        
        model = regenerator.regenerate_inheritance(classes)
        
        # Check that the model includes entries for each class
        assert "BaseModel" in model
        assert "TodoItem" in model
        assert "TodoList" in model
        
        # Check inheritance relationships
        assert model["BaseModel"]["base"] == []
        assert model["TodoItem"]["base"] == ["BaseModel"]
        assert model["TodoList"]["base"] == ["BaseModel"]


def test_write_model(context_dirs):
    """Test writing the model to a file."""
    model_path = context_dirs / "context_inheritance_model.json"
    regenerator = InheritanceBlockRegenerator(
        commands_root=context_dirs,
        model_path=model_path
    )
    
    # Create a model to write
    model = {
        "TodoList": {"base": []},
        "TodoItem": {"base": ["BaseItem"]},
        "User": {"base": []}
    }
    
    # Write the model
    regenerator.write_model(model)
    
    # Check that the file was written
    assert model_path.exists()
    
    # Check that the file contains the expected content
    with open(model_path, "r") as f:
        written_model = json.load(f)
    
    assert written_model == model


def test_preserve_existing_entries(existing_model, context_dirs):
    """Test that existing entries are preserved during regeneration."""
    regenerator = InheritanceBlockRegenerator(
        commands_root=context_dirs,
        model_path=existing_model
    )
    
    model = regenerator.regenerate_inheritance()
    
    # Check that old entries are preserved
    assert "OldClass" in model
    assert "AnotherOldClass" in model
    assert model["AnotherOldClass"]["base"] == ["OldClass"]
    
    # Check that new entries were added
    assert "TodoList" in model
    assert "TodoItem" in model
    assert "User" in model


def test_preserve_aggregation(existing_model, context_dirs):
    """Test that the aggregation block is preserved during regeneration."""
    regenerator = InheritanceBlockRegenerator(
        commands_root=context_dirs,
        model_path=existing_model
    )

    model = regenerator.regenerate_inheritance()

    # Check that the model has been updated with TodoList
    # In the new flat structure, TodoList should be at the top level
    assert "TodoList" in model
    
    # Check that OldClass is preserved from the existing model
    assert "OldClass" in model
    assert "AnotherOldClass" in model
    assert model["AnotherOldClass"]["base"] == ["OldClass"]


def test_idempotence(existing_model, context_dirs):
    """Test that running the regenerator multiple times produces the same result."""
    regenerator = InheritanceBlockRegenerator(
        commands_root=context_dirs,
        model_path=existing_model
    )
    
    # Run the regenerator twice
    first_result = regenerator.regenerate_inheritance()
    second_result = regenerator.regenerate_inheritance()
    
    # Check that the results are the same
    assert first_result == second_result
    
    # Check that the file contents are the same
    with open(existing_model, 'r') as f:
        file_content = json.load(f)
    
    assert file_content == second_result 
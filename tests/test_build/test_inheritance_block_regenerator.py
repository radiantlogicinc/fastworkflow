import os
import json
import tempfile
import pytest
from pathlib import Path

from fastworkflow.build.inheritance_block_regenerator import InheritanceBlockRegenerator
from fastworkflow.build.class_analysis_structures import ClassInfo


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
def existing_model(temp_dir):
    """Create an existing model file with aggregation block."""
    model_path = temp_dir / "_commands/context_inheritance_model.json"
    model_data = {
        "inheritance": {
            "OldClass": {"base": []},
            "AnotherOldClass": {"base": ["OldClass"]},
            "*": {"base": []}
        },
        "aggregation": {
            "TodoItem": {"container": ["TodoList"]},
            "User": {"container": ["TodoList"]}
        }
    }
    create_test_context_model(model_path, model_data)
    return model_path


@pytest.fixture
def mock_classes():
    """Create mock class information for testing."""
    classes = {}
    
    # Create TodoList class
    todo_list = ClassInfo(
        name="TodoList",
        module_path="app/models/todo_list.py",
        bases=["BaseModel"]
    )
    classes["TodoList"] = todo_list
    
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
    """Test loading an existing model file."""
    regenerator = InheritanceBlockRegenerator(model_path=existing_model)
    model = regenerator.load_existing_model()
    
    assert "inheritance" in model
    assert "aggregation" in model
    assert "TodoItem" in model["aggregation"]
    assert "User" in model["aggregation"]
    assert "OldClass" in model["inheritance"]


def test_load_missing_model(temp_dir):
    """Test loading a non-existent model file."""
    non_existent_model = temp_dir / "non_existent_model.json"
    regenerator = InheritanceBlockRegenerator(model_path=non_existent_model)
    model = regenerator.load_existing_model()
    
    assert "inheritance" in model
    assert "aggregation" in model
    assert model["inheritance"] == {"*": {"base": []}}
    assert model["aggregation"] == {}


def test_load_invalid_model(temp_dir):
    """Test loading an invalid model file."""
    invalid_model_path = temp_dir / "invalid_model.json"
    
    # Create an invalid JSON file
    with open(invalid_model_path, 'w') as f:
        f.write("{invalid json")
    
    regenerator = InheritanceBlockRegenerator(model_path=invalid_model_path)
    model = regenerator.load_existing_model()
    
    assert "inheritance" in model
    assert "aggregation" in model
    assert model["inheritance"] == {"*": {"base": []}}
    assert model["aggregation"] == {}


def test_regenerate_inheritance_with_directory_scan(context_dirs, temp_dir):
    """Test regenerating inheritance block based on directory scan."""
    model_path = temp_dir / "_commands/context_inheritance_model.json"
    regenerator = InheritanceBlockRegenerator(
        commands_root=context_dirs,
        model_path=model_path
    )
    
    model = regenerator.regenerate_inheritance()
    
    assert "inheritance" in model
    assert "aggregation" in model
    assert "TodoList" in model["inheritance"]
    assert "TodoItem" in model["inheritance"]
    assert "User" in model["inheritance"]
    assert "*" in model["inheritance"]
    assert model["inheritance"]["TodoList"] == {"base": []}
    assert model["inheritance"]["TodoItem"] == {"base": []}
    assert model["inheritance"]["User"] == {"base": []}
    assert model["inheritance"]["*"] == {"base": []}


def test_regenerate_inheritance_with_classes(mock_classes, temp_dir):
    """Test regenerating inheritance block based on class information."""
    model_path = temp_dir / "_commands/context_inheritance_model.json"
    regenerator = InheritanceBlockRegenerator(model_path=model_path)
    
    model = regenerator.regenerate_inheritance(mock_classes)
    
    assert "inheritance" in model
    assert "aggregation" in model
    assert "TodoList" in model["inheritance"]
    assert "TodoItem" in model["inheritance"]
    assert "User" in model["inheritance"]
    assert "BaseModel" in model["inheritance"]
    assert "*" in model["inheritance"]
    assert model["inheritance"]["TodoList"] == {"base": ["BaseModel"]}
    assert model["inheritance"]["TodoItem"] == {"base": ["BaseModel"]}
    assert model["inheritance"]["User"] == {"base": ["BaseModel", "TodoList"]}
    assert model["inheritance"]["BaseModel"] == {"base": []}
    assert model["inheritance"]["*"] == {"base": []}


def test_preserve_aggregation(existing_model, context_dirs):
    """Test that the aggregation block is preserved during regeneration."""
    regenerator = InheritanceBlockRegenerator(
        commands_root=context_dirs,
        model_path=existing_model
    )
    
    model = regenerator.regenerate_inheritance()
    
    # Check that the inheritance block has been updated
    assert "TodoList" in model["inheritance"]
    assert "TodoItem" in model["inheritance"]
    assert "User" in model["inheritance"]
    assert "OldClass" not in model["inheritance"]
    
    # Check that the aggregation block has been preserved
    assert "aggregation" in model
    assert "TodoItem" in model["aggregation"]
    assert "User" in model["aggregation"]
    assert model["aggregation"]["TodoItem"] == {"container": ["TodoList"]}
    assert model["aggregation"]["User"] == {"container": ["TodoList"]}


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
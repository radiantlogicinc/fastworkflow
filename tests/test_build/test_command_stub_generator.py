import os
import json
import tempfile
import pytest
from pathlib import Path

from fastworkflow.build.command_stub_generator import CommandStubGenerator


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
def basic_context_model(temp_dir):
    """Create a basic context model for testing."""
    model_path = temp_dir / "_commands/context_inheritance_model.json"
    model_data = {
        "TodoList": {"base": []},
        "TodoItem": {"base": ["TodoList"]},
        "User": {"base": []},
        "*": {"base": []}
    }
    create_test_context_model(model_path, model_data)
    return model_path


def test_get_command_file_path(commands_dir):
    """Test getting the file path for a command."""
    generator = CommandStubGenerator(
        commands_root=commands_dir
    )
    
    # Test global context
    assert generator.get_command_file_path('*', 'test_command') == commands_dir / "test_command.py"
    
    # Test specific context
    assert generator.get_command_file_path('TodoList', 'test_command') == commands_dir / "TodoList" / "test_command.py"


def test_get_handlers_file_path(commands_dir):
    """Test getting the file path for a handlers file."""
    generator = CommandStubGenerator(
        commands_root=commands_dir
    )
    
    # Test global context
    assert generator.get_handlers_file_path('*') == commands_dir / "_fastworkflow_handlers.py"
    
    # Test specific context
    assert generator.get_handlers_file_path('TodoList') == commands_dir / "TodoList" / "_fastworkflow_handlers.py"


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
    
    generator = CommandStubGenerator()
    
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


def test_generate_command_stub(commands_dir, basic_context_model):
    """Test generating a command stub for a specific context."""
    generator = CommandStubGenerator(
        commands_root=commands_dir,
        model_path=basic_context_model
    )
    
    # Generate a command stub
    file_path = generator.generate_command_stub('*', 'test_command')
    
    # Check that the file was created
    assert file_path.exists()
    
    # Check that the file contains the expected content
    content = file_path.read_text()
    assert 'class Signature:' in content
    assert 'class ResponseGenerator:' in content


# Skip tests that depend on container-context handlers (feature removed)

@pytest.mark.skip(reason="Container context handlers no longer generated")
def test_generate_handlers_file(commands_dir, basic_context_model):
    pass


@pytest.mark.skip(reason="Container context handlers no longer generated")
def test_generate_command_stubs_for_context(commands_dir, basic_context_model):
    pass


@pytest.mark.skip(reason="Container context handlers no longer generated")
def test_generate_all_handlers_files(commands_dir, basic_context_model):
    pass


def test_generate_command_stub_existing_file(commands_dir):
    """Test generating a command stub when the file already exists."""
    generator = CommandStubGenerator(
        commands_root=commands_dir
    )
    
    # Create the file first
    file_path = commands_dir / "existing_command.py"
    with open(file_path, 'w') as f:
        f.write("# Existing content")
    
    # Try to generate a command stub
    result = generator.generate_command_stub('*', 'existing_command')
    
    # Check that no file was returned
    assert result is None
    
    # Check that the file content was not changed
    content = file_path.read_text()
    assert content == "# Existing content"


@pytest.mark.skip(reason="Container context handlers no longer generated")
def test_generate_handlers_file_existing_file(commands_dir, basic_context_model):
    pass


def test_generate_command_stub_force_overwrite(commands_dir):
    """Test generating a command stub with force overwrite."""
    generator = CommandStubGenerator(
        commands_root=commands_dir
    )
    
    # Create the file first
    file_path = commands_dir / "force_command.py"
    with open(file_path, 'w') as f:
        f.write("# Existing content")
    
    # Try to generate a command stub with force=True
    result = generator.generate_command_stub('*', 'force_command', force=True)
    
    # Check that a file was returned
    assert result is not None
    
    # Check that the file content was changed
    content = file_path.read_text()
    assert "# Existing content" not in content
    assert 'class Signature:' in content 
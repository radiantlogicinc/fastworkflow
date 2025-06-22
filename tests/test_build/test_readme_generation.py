import os
import tempfile
import json
from pathlib import Path

import pytest

from fastworkflow.build.documentation_generator import (
    collect_command_files_and_context_model,
    extract_command_metadata,
    generate_readme_content,
    write_readme_file,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_commands_dir(temp_dir):
    """Create a mock commands directory with a simple context model and command file."""
    # Create _commands directory
    commands_dir = temp_dir / "_commands"
    commands_dir.mkdir(exist_ok=True)
    
    # Create TodoItem directory
    todo_item_dir = commands_dir / "TodoItem"
    todo_item_dir.mkdir(exist_ok=True)
    
    # Create a simple context model
    context_model = {
        "TodoItem": {"base": []},
        "TodoList": {"base": ["TodoItem"]}
    }
    
    with open(commands_dir / "context_inheritance_model.json", "w") as f:
        json.dump(context_model, f, indent=2)
    
    # Create a simple command file
    command_content = """
from pydantic import BaseModel, Field

class Signature:
    class Output(BaseModel):
        success: bool = Field(description="Indicates successful execution.")
    
    plain_utterances = [
        "test command"
    ]
    
    template_utterances = []

class ResponseGenerator:
    def _process_command(self, session):
        return Signature.Output(success=True)
    
    def __call__(self, session, command):
        output = self._process_command(session)
        return output
"""
    
    with open(todo_item_dir / "test_command.py", "w") as f:
        f.write(command_content)
    
    return commands_dir


def test_readme_generation(mock_commands_dir):
    """Test that the README.md file is properly generated."""
    # Collect command files and context model
    command_files, context_model, error = collect_command_files_and_context_model(mock_commands_dir)
    
    assert error is None, f"Error collecting command files and context model: {error}"
    assert command_files, "No command files found"
    assert context_model, "No context model found"
    
    # Extract command metadata
    command_metadata = extract_command_metadata(command_files)
    
    assert command_metadata, "No command metadata extracted"
    
    # Generate README content
    readme_content = generate_readme_content(command_metadata, context_model, ".")
    
    assert readme_content, "No README content generated"
    assert "# FastWorkflow Commands" in readme_content, "README content should start with '# FastWorkflow Commands'"
    assert "## Available Commands" in readme_content, "README content should include '## Available Commands'"
    assert "### TodoItem Context" in readme_content, "README content should include '### TodoItem Context'"
    assert "test_command" in readme_content, "README content should include the command name"
    
    # Write README file
    readme_path = mock_commands_dir / "README.md"
    success = write_readme_file(mock_commands_dir, readme_content)
    
    assert success, "Failed to write README file"
    assert readme_path.exists(), "README.md file was not created"
    
    # Read the README file
    with open(readme_path, "r") as f:
        readme_file_content = f.read()
    
    assert readme_file_content == readme_content, "README file content does not match the generated content" 
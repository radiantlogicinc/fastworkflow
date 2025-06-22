import os
import tempfile
from pathlib import Path

import pytest

from fastworkflow.build.command_file_generator import generate_command_files
from fastworkflow.build.class_analysis_structures import ClassInfo, MethodInfo


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def todo_item_class():
    """Create a TodoItem class with a method for testing."""
    class_info = ClassInfo('TodoItem', 'application/todo_item.py')
    
    # Add a method
    class_info.methods.append(MethodInfo(
        name="complete",
        parameters=[],
        docstring="Mark the todo item as complete.",
        return_annotation="bool"
    ))
    
    return class_info


def test_no_access_comment(temp_dir, todo_item_class):
    """Test that the 'Access the application class instance:' comment is not present in generated files."""
    # Generate the command files
    files = generate_command_files(
        classes={'TodoItem': todo_item_class},
        output_dir=str(temp_dir),
        source_dir='.'
    )
    
    # Find the complete.py file
    complete_file = None
    for file_path in files:
        if os.path.basename(file_path) == 'complete.py':
            complete_file = file_path
            break
    
    assert complete_file is not None, "Could not find complete.py file"
    
    # Read the file
    with open(complete_file, 'r') as f:
        file_content = f.read()
    
    # Check that the comment is not present
    assert "# Access the application class instance:" not in file_content, \
        "The comment '# Access the application class instance:' should not be present in the generated file" 
import os
import tempfile
import ast
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
def todo_list_class():
    """Create a TodoList class with a method without parameters for testing."""
    class_info = ClassInfo('TodoList', 'application/todo_list.py')
    
    # Add a method without parameters
    class_info.methods.append(MethodInfo(
        name="mark_completed",
        parameters=[],  # No parameters
        docstring="Mark this TodoList and all children as complete.",
        return_annotation="None"
    ))
    
    return class_info


def test_no_parameters_no_input_class(temp_dir, todo_list_class):
    """Test that methods without parameters don't have an Input class."""
    # Generate the command files
    files = generate_command_files(
        classes={'TodoList': todo_list_class},
        output_dir=str(temp_dir),
        source_dir='.'
    )
    
    # Find the mark_completed.py file
    mark_completed_file = None
    for file_path in files:
        if os.path.basename(file_path) == 'mark_completed.py':
            mark_completed_file = file_path
            break
    
    assert mark_completed_file is not None, "Could not find mark_completed.py file"
    
    # Read the file
    with open(mark_completed_file, 'r') as f:
        file_content = f.read()
    
    # Parse the file
    tree = ast.parse(file_content)
    
    # Find the Signature class
    signature_class = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == 'Signature':
            signature_class = node
            break
    
    assert signature_class is not None, "Could not find Signature class"
    
    # Check that there is no Input class
    for node in signature_class.body:
        if isinstance(node, ast.ClassDef):
            assert node.name != 'Input', "There should be no Input class for methods without parameters"
    
    # The one __call__ takes input, which is None for a command with no Input (the three-argument form)
    call_method = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == '__call__':
            call_method = node
            break

    assert call_method is not None, "Could not find __call__ method"
    assert [a.arg for a in call_method.args.args] == ['self', 'workflow', 'command', 'input']
    assert ast.unparse(call_method.args.args[3].annotation) == 'None'
    assert call_method.args.defaults == [], "__call__ takes exactly three arguments, with no default" 
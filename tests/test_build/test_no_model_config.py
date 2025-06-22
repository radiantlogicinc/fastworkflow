import os
import tempfile
import ast
from pathlib import Path

import pytest

from fastworkflow.build.command_file_template import create_command_file
from fastworkflow.build.class_analysis_structures import ClassInfo, MethodInfo, PropertyInfo


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def todo_item_class():
    """Create a TodoItem class with methods for testing."""
    class_info = ClassInfo('TodoItem', 'application/todo_item.py')
    
    # Add a method that will be used for testing
    class_info.methods.append(MethodInfo(
        name="complete",
        parameters=[],
        docstring="Mark the todo item as complete.",
        return_annotation="bool"
    ))
    
    return class_info


def test_input_output_no_model_config(temp_dir, todo_item_class):
    """Test that generated command files do not include model_config in Input or Output classes."""
    # Generate the command file
    output_dir = temp_dir / "TodoItem"
    output_dir.mkdir(exist_ok=True)
    file_path = create_command_file(
        class_info=todo_item_class,
        method_info=todo_item_class.methods[0],
        output_dir=output_dir,
        file_name="complete.py",
        source_dir="."
    )
    
    # Read the generated file
    with open(file_path, 'r') as f:
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
    
    # Check for model_config in Input class (if present)
    input_class = None
    for node in signature_class.body:
        if isinstance(node, ast.ClassDef) and node.name == 'Input':
            input_class = node
            break
    
    if input_class:
        for node in input_class.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        assert target.id != 'model_config', "Input class should not have model_config"
    
    # Check for model_config in Output class
    output_class = None
    for node in signature_class.body:
        if isinstance(node, ast.ClassDef) and node.name == 'Output':
            output_class = node
            break
    
    assert output_class is not None, "Could not find Output class"
    
    for node in output_class.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assert target.id != 'model_config', "Output class should not have model_config"
    
    # Check that ConfigDict is not imported
    assert "from pydantic import ConfigDict" not in file_content, "ConfigDict should not be imported" 
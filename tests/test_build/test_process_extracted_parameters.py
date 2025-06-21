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


def test_process_extracted_parameters_is_instance_method(temp_dir, todo_item_class):
    """Test that process_extracted_parameters is defined as an instance method, not a static method."""
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
    
    # Find the process_extracted_parameters method
    process_method = None
    for node in signature_class.body:
        if isinstance(node, ast.FunctionDef) and node.name == 'process_extracted_parameters':
            process_method = node
            break
    
    assert process_method is not None, "Could not find process_extracted_parameters method"
    
    # Check if it's an instance method (first parameter is 'self')
    assert len(process_method.args.args) > 0, "process_extracted_parameters should have parameters"
    assert process_method.args.args[0].arg == 'self', "First parameter of process_extracted_parameters should be 'self'"
    
    # Check that it's not decorated with @staticmethod or @classmethod
    for decorator in process_method.decorator_list:
        if isinstance(decorator, ast.Name):
            assert decorator.id not in ('staticmethod', 'classmethod'), \
                "process_extracted_parameters should not be decorated with @staticmethod or @classmethod"


def test_generate_utterances_is_static_method(temp_dir, todo_item_class):
    """Test that generate_utterances is defined as a static method."""
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
    
    # Find the generate_utterances method
    generate_method = None
    for node in signature_class.body:
        if isinstance(node, ast.FunctionDef) and node.name == 'generate_utterances':
            generate_method = node
            break
    
    assert generate_method is not None, "Could not find generate_utterances method"
    
    # Check if it's a static method
    is_static_method = False
    for decorator in generate_method.decorator_list:
        if isinstance(decorator, ast.Name) and decorator.id == 'staticmethod':
            is_static_method = True
            break
    
    assert is_static_method, "generate_utterances should be decorated with @staticmethod" 
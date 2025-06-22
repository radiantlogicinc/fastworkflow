import os
import tempfile
import ast
from pathlib import Path

import pytest

from fastworkflow.build.command_file_generator import generate_command_files
from fastworkflow.build.class_analysis_structures import ClassInfo, MethodInfo, PropertyInfo


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def todo_item_class():
    """Create a TodoItem class with a settable property for testing."""
    class_info = ClassInfo('TodoItem', 'application/todo_item.py')
    
    # Add a property
    description_prop = PropertyInfo('description', docstring='Description of this todo item', type_annotation='str')
    class_info.properties.append(description_prop)
    
    # Add it to settable properties
    class_info.all_settable_properties.append(description_prop)
    
    # Add a setter method for the property
    class_info.methods.append(MethodInfo(
        name="description",  # Same name as the property
        parameters=[{'name': 'value', 'annotation': 'str'}],  # Parameter named 'value'
        docstring="Set the description of the todo item."
    ))
    
    return class_info


def test_property_setter_input_uses_property_name(temp_dir, todo_item_class):
    """Test that property setter input uses the property name, not 'value'."""
    # Generate the command files
    files = generate_command_files(
        classes={'TodoItem': todo_item_class},
        output_dir=str(temp_dir),
        source_dir='.'
    )
    
    # Find the description.py file
    description_file = None
    for file_path in files:
        if os.path.basename(file_path) == 'description.py':
            description_file = file_path
            break
    
    assert description_file is not None, "Could not find description.py file"
    
    # Read the file
    with open(description_file, 'r') as f:
        file_content = f.read()
    
    # Parse the file
    tree = ast.parse(file_content)
    
    # Find the Input class
    input_class = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == 'Signature':
            for child in node.body:
                if isinstance(child, ast.ClassDef) and child.name == 'Input':
                    input_class = child
                    break
            if input_class:
                break
    
    assert input_class is not None, "Could not find Input class"
    
    # Check that the input field is named 'description', not 'value'
    found_description_field = False
    for node in input_class.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == 'description':
                found_description_field = True
                break
    
    assert found_description_field, "Input field should be named 'description', not 'value'"
    
    # Also check that the assignment in _process_command uses input.description
    process_command = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == '_process_command':
            process_command = node
            break
    
    assert process_command is not None, "Could not find _process_command method"
    
    # Look for app_instance.description = input.description
    found_correct_assignment = False
    for node in ast.walk(process_command):
        if isinstance(node, ast.Assign):
            if (isinstance(node.targets[0], ast.Attribute) and 
                node.targets[0].attr == 'description' and
                isinstance(node.value, ast.Attribute) and
                node.value.attr == 'description'):
                found_correct_assignment = True
                break
    
    assert found_correct_assignment, "Property setter should use 'app_instance.description = input.description'" 
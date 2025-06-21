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


def test_response_uses_model_dump_json(temp_dir, todo_item_class):
    """Test that command files use model_dump_json() for responses."""
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
    
    # Find the __call__ method
    call_method = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == '__call__':
            call_method = node
            break
    
    assert call_method is not None, "Could not find __call__ method"
    
    # Look for CommandResponse with model_dump_json()
    found_model_dump_json = False
    for node in ast.walk(call_method):
        if isinstance(node, ast.Call):
            # Check if the call is to CommandResponse
            if isinstance(node.func, ast.Name) and node.func.id == 'CommandResponse':
                # Check for model_dump_json() in the arguments
                for keyword in node.keywords:
                    if keyword.arg == 'response':
                        if isinstance(keyword.value, ast.Call):
                            if isinstance(keyword.value.func, ast.Attribute) and keyword.value.func.attr == 'model_dump_json':
                                found_model_dump_json = True
                                break
    
    assert found_model_dump_json, "Command response should use model_dump_json() for the response"


def test_get_properties_response_uses_model_dump_json(temp_dir, todo_item_class):
    """Test that get_properties command uses model_dump_json() for responses."""
    # Add a property to the class
    todo_item_class.properties.append(PropertyInfo(
        name="description",
        docstring="Description of this todo item",
        type_annotation="str"
    ))
    
    # Generate the get_properties command file
    output_dir = temp_dir / "TodoItem"
    output_dir.mkdir(exist_ok=True)
    
    # Create a method info for get_properties
    get_properties_method = MethodInfo(
        name="GetProperties",
        parameters=[],
        docstring="Get all properties of the TodoItem class.",
        return_annotation="Dict[str, Any]"
    )
    
    file_path = create_command_file(
        class_info=todo_item_class,
        method_info=get_properties_method,
        output_dir=output_dir,
        file_name="get_properties.py",
        source_dir=".",
        is_get_all_properties=True,
        all_properties_for_template=todo_item_class.properties
    )
    
    # Read the generated file
    with open(file_path, 'r') as f:
        file_content = f.read()
    
    # Parse the file
    tree = ast.parse(file_content)
    
    # Find the __call__ method
    call_method = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == '__call__':
            call_method = node
            break
    
    assert call_method is not None, "Could not find __call__ method"
    
    # Look for CommandResponse with model_dump_json()
    found_model_dump_json = False
    for node in ast.walk(call_method):
        if isinstance(node, ast.Call):
            # Check if the call is to CommandResponse
            if isinstance(node.func, ast.Name) and node.func.id == 'CommandResponse':
                # Check for model_dump_json() in the arguments
                for keyword in node.keywords:
                    if keyword.arg == 'response':
                        if isinstance(keyword.value, ast.Call):
                            if isinstance(keyword.value.func, ast.Attribute) and keyword.value.func.attr == 'model_dump_json':
                                found_model_dump_json = True
                                break
    
    assert found_model_dump_json, "get_properties response should use model_dump_json() for the response"


def test_set_properties_response_uses_model_dump_json(temp_dir, todo_item_class):
    """Test that set_properties command uses model_dump_json() for responses."""
    # Add a settable property to the class
    prop = PropertyInfo(
        name="description",
        docstring="Description of this todo item",
        type_annotation="str"
    )
    todo_item_class.properties.append(prop)
    todo_item_class.all_settable_properties.append(prop)
    
    # Generate the set_properties command file
    output_dir = temp_dir / "TodoItem"
    output_dir.mkdir(exist_ok=True)
    
    # Create a method info for set_properties
    set_properties_method = MethodInfo(
        name="SetProperties",
        parameters=[{'name': 'description', 'annotation': 'Optional[str]', 'is_optional': True}],
        docstring="Sets one or more properties for an instance of TodoItem."
    )
    
    file_path = create_command_file(
        class_info=todo_item_class,
        method_info=set_properties_method,
        output_dir=output_dir,
        file_name="set_properties.py",
        source_dir=".",
        is_set_all_properties=True,
        settable_properties_for_template=todo_item_class.all_settable_properties
    )
    
    # Read the generated file
    with open(file_path, 'r') as f:
        file_content = f.read()
    
    # Parse the file
    tree = ast.parse(file_content)
    
    # Find the __call__ method
    call_method = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == '__call__':
            call_method = node
            break
    
    assert call_method is not None, "Could not find __call__ method"
    
    # Look for CommandResponse with model_dump_json()
    found_model_dump_json = False
    for node in ast.walk(call_method):
        if isinstance(node, ast.Call):
            # Check if the call is to CommandResponse
            if isinstance(node.func, ast.Name) and node.func.id == 'CommandResponse':
                # Check for model_dump_json() in the arguments
                for keyword in node.keywords:
                    if keyword.arg == 'response':
                        if isinstance(keyword.value, ast.Call):
                            if isinstance(keyword.value.func, ast.Attribute) and keyword.value.func.attr == 'model_dump_json':
                                found_model_dump_json = True
                                break
    
    assert found_model_dump_json, "set_properties response should use model_dump_json() for the response" 
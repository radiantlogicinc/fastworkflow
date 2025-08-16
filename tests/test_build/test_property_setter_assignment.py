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
    """Create a TodoItem class with properties for testing."""
    class_info = ClassInfo('TodoItem', 'application/todo_item.py')
    
    # Add a property that will be used for a setter command
    description_prop = PropertyInfo('description', docstring='Description of this todo item', type_annotation='str')
    class_info.properties.append(description_prop)
    
    # Add it to settable properties
    class_info.all_settable_properties.append(description_prop)
    
    return class_info


def test_property_setter_uses_assignment(temp_dir, todo_item_class):
    """Test that property setter commands use attribute assignment, not method calls."""
    # Create a method info for a property setter
    method_info = MethodInfo(
        name="description",
        parameters=[{'name': 'value', 'annotation': 'str'}],
        docstring="Set the description of the todo item."
    )
    
    # Generate the command file
    output_dir = temp_dir / "TodoItem"
    output_dir.mkdir(exist_ok=True)
    file_path = create_command_file(
        class_info=todo_item_class,
        method_info=method_info,
        output_dir=output_dir,
        file_name="description.py",
        source_dir=".",
        is_property_setter=True
    )
    
    # Read the generated file
    with open(file_path, 'r') as f:
        file_content = f.read()
    
    # Parse the file to find the property assignment
    tree = ast.parse(file_content)
    
    # Find the _process_command method
    process_command_method = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == '_process_command':
            process_command_method = node
            break
    
    assert process_command_method is not None, "Could not find _process_command method"
    
    # Look for attribute assignment (obj.prop = value)
    found_assignment = False
    for node in ast.walk(process_command_method):
        if isinstance(node, ast.Assign):
            # Check if the target is an attribute (obj.prop)
            for target in node.targets:
                if isinstance(target, ast.Attribute):
                    # Check if the attribute name is 'description'
                    if target.attr == 'description':
                        found_assignment = True
                        break
    
    assert found_assignment, "Property setter should use attribute assignment (obj.prop = value)"
    
    # Make sure there are no method calls for setting properties
    found_method_call = False
    for node in ast.walk(process_command_method):
        if isinstance(node, ast.Call):
            # Check if the call is to an attribute (obj.prop())
            if isinstance(node.func, ast.Attribute):
                # Check if the attribute name is 'description'
                if node.func.attr == 'description':
                    found_method_call = True
                    break
    
    assert not found_method_call, "Property setter should not use method calls (obj.prop(value=value))"


def test_set_properties_uses_assignment(temp_dir, todo_item_class):
    """Test that set_properties command uses attribute assignment for all properties."""
    # Generate the set_properties command file
    output_dir = temp_dir / "TodoItem"
    output_dir.mkdir(exist_ok=True)
    
    # Add another property to test multiple assignments
    status_prop = PropertyInfo('status', docstring='Status of this todo item', type_annotation='str')
    todo_item_class.properties.append(status_prop)
    todo_item_class.all_settable_properties.append(status_prop)
    
    # Create a method info for set_properties
    set_properties_method = MethodInfo(
        name="SetProperties",
        parameters=[
            {'name': 'description', 'annotation': 'Optional[str]', 'is_optional': True},
            {'name': 'status', 'annotation': 'Optional[str]', 'is_optional': True}
        ],
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
    
    # Parse the file to find the property assignments
    tree = ast.parse(file_content)
    
    # Find the _process_command method
    process_command_method = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == '_process_command':
            process_command_method = node
            break
    
    assert process_command_method is not None, "Could not find _process_command method"
    
    # Count attribute assignments (obj.prop = value)
    assignment_count = 0
    for node in ast.walk(process_command_method):
        if isinstance(node, ast.Assign):
            # Check if the target is an attribute (obj.prop)
            for target in node.targets:
                if isinstance(target, ast.Attribute):
                    # Check if the attribute name is one of our properties
                    if target.attr in ['description', 'status']:
                        assignment_count += 1
    
    # We should have conditional assignments for each property
    assert assignment_count >= 2, "set_properties should use attribute assignment for all properties"
    
    # Make sure there are no method calls for setting properties
    found_method_call = False
    for node in ast.walk(process_command_method):
        if isinstance(node, ast.Call):
            # Check if the call is to an attribute (obj.prop())
            if isinstance(node.func, ast.Attribute):
                # Check if the attribute name is one of our properties
                if node.func.attr in ['description', 'status']:
                    found_method_call = True
                    break
    
    assert not found_method_call, "set_properties should not use method calls for property setting"


def test_status_property_special_handling(temp_dir, todo_item_class):
    """Test that the status property is directly set using the status parameter."""
    # Add the status property
    status_prop = PropertyInfo('status', docstring='Status of this todo item', type_annotation='bool')
    todo_item_class.properties.append(status_prop)
    todo_item_class.all_settable_properties.append(status_prop)
    
    # Create a method info for the status command
    method_info = MethodInfo(
        name="status",
        parameters=[{'name': 'status', 'annotation': 'bool'}],
        docstring="Set the status of the todo item."
    )
    
    # Generate the command file
    output_dir = temp_dir / "TodoItem"
    output_dir.mkdir(exist_ok=True)
    file_path = create_command_file(
        class_info=todo_item_class,
        method_info=method_info,
        output_dir=output_dir,
        file_name="status.py",
        source_dir=".",
        is_property_setter=True
    )
    
    # Read the generated file
    with open(file_path, 'r') as f:
        file_content = f.read()
    
    # Parse the file
    tree = ast.parse(file_content)
    
    # Find the _process_command method
    process_command_method = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == '_process_command':
            process_command_method = node
            break
    
    assert process_command_method is not None, "Could not find _process_command method"
    
    # Look for any assignment to the status attribute
    found_status_assignment = False
    for node in ast.walk(process_command_method):
        if isinstance(node, ast.Assign):
            # Check if the target is the status attribute
            for target in node.targets:
                if isinstance(target, ast.Attribute) and target.attr == 'status':
                    found_status_assignment = True
                    break
    
    assert found_status_assignment, "Status setter should assign to the status attribute"
    
    # Check that the file content directly assigns status
    assert "app_instance.status = input.status" in file_content, "Status setter should directly assign status parameter" 
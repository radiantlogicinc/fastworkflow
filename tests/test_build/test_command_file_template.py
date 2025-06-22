import os
import pathlib
import tempfile
import ast

import pytest
from fastworkflow.build.command_file_template import create_command_file
from fastworkflow.build.class_analysis_structures import ClassInfo, MethodInfo, PropertyInfo

def make_class_and_method(class_name, method_name, params=None, docstring=None, module_path='application/module.py', return_annotation=None):
    class_info = ClassInfo(class_name, module_path)
    method_info = MethodInfo(method_name, params or [], docstring=docstring, return_annotation=return_annotation)
    return class_info, method_info

def test_create_command_file_simple():
    class_info, method_info = make_class_and_method('User', 'get_details', params=[{'name': 'user_id', 'annotation': 'int'}], docstring='Get user details.')
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = create_command_file(class_info, method_info, tmpdir, source_dir='.')
        assert os.path.exists(file_path)
        content = pathlib.Path(file_path).read_text()
        assert 'class Input(BaseModel):' in content
        assert 'class ResponseGenerator' in content
        assert 'plain_utterances =' in content
        
        # Verify new format for generate_utterances
        assert "command_name.split('/')[-1].lower().replace('_', ' ')" in content
        
        # Verify app_instance is retrieved from session.command_context_for_response_generation
        assert "app_instance = session.command_context_for_response_generation" in content
        
        # Verify response uses output.model_dump_json()
        assert "CommandResponse(response=output.model_dump_json())" in content

def test_create_command_file_with_type_annotations():
    params = [
        {'name': 'user_id', 'annotation': 'int'},
        {'name': 'verbose', 'annotation': 'bool'},
    ]
    class_info, method_info = make_class_and_method('User', 'fetch', params=params, docstring='Fetch user.')
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = create_command_file(class_info, method_info, tmpdir, source_dir='.')
        content = pathlib.Path(file_path).read_text()
        assert 'user_id: int = Field' in content
        assert 'verbose: bool = Field' in content
        assert 'Fetch user.' in content

def test_create_command_file_with_complex_types():
    params = [
        {'name': 'ids', 'annotation': 'List[int]'},
        {'name': 'options', 'annotation': 'Optional[dict]'},
    ]
    class_info, method_info = make_class_and_method('Group', 'batch_update', params=params, docstring='Batch update.')
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = create_command_file(class_info, method_info, tmpdir, source_dir='.')
        content = pathlib.Path(file_path).read_text()
        assert 'ids: List[int] = Field' in content
        assert 'options: Optional[dict] = Field' in content
        assert 'Batch update.' in content

def test_create_command_file_get_properties():
    class_name = "Product"
    class_info = ClassInfo(class_name, 'application/product.py')
    class_info.properties.append(PropertyInfo(name="product_id", type_annotation="int", docstring="The product ID."))
    class_info.properties.append(PropertyInfo(name="name", type_annotation="str", docstring="The product name."))
    class_info.properties.append(PropertyInfo(name="price", type_annotation="float"))

    method_info_get_props = MethodInfo(
        name="GetProperties", 
        parameters=[], 
        return_annotation="Dict[str, Any]",
        docstring=f"Get all properties of the {class_name} class."
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        check_created_command_file(
            class_info, method_info_get_props, tmpdir
        )


def check_created_command_file(class_info, method_info_get_props, tmpdir):
    file_path = create_command_file(
        class_info=class_info, 
        method_info=method_info_get_props, 
        output_dir=tmpdir, 
        source_dir='.', 
        is_get_all_properties=True,
        all_properties_for_template=class_info.properties
    )
    assert os.path.exists(file_path)
    content = pathlib.Path(file_path).read_text()

    # No Input class should be present for get_properties
    assert "class Input(BaseModel):" not in content
    assert "class Output(BaseModel):" in content
    assert "product_id: int = Field(description=\"The product ID.\")" in content
    assert "name: str = Field(description=\"The product name.\")" in content
    assert "price: float = Field(description=\"Value of property price\")" in content

    assert "# For get_properties, the primary logic is to gather attribute values" in content
    assert "return Signature.Output(product_id=app_instance.product_id, name=app_instance.name, price=app_instance.price)" in content
    assert "app_instance = session.command_context_for_response_generation" in content
    
    # Verify that _process_command doesn't expect an input parameter
    assert "def _process_command(self, session: Session) -> Signature.Output:" in content
    # Verify that __call__ doesn't expect command_parameters
    assert "def __call__(self, session: Session, command: str) -> CommandOutput:" in content

def test_create_command_file_set_properties():
    class_name = "InventoryItem"
    class_info = ClassInfo(class_name, 'application/inventory.py')
    
    settable_props = [
        PropertyInfo(name="sku", type_annotation="str", docstring="Stock Keeping Unit."),
        PropertyInfo(name="quantity", type_annotation="int", docstring="Current stock quantity."),
        PropertyInfo(name="location", type_annotation="Optional[str]")
    ]
    class_info.all_settable_properties = settable_props

    method_info_set_props = MethodInfo(
        name="SetProperties", 
        parameters=[
            {'name': 'sku', 'annotation': 'str', 'is_optional': True},
            {'name': 'quantity', 'annotation': 'int', 'is_optional': True},
            {'name': 'location', 'annotation': 'Optional[str]', 'is_optional': True}
        ],
        return_annotation="Dict[str, bool]",
        docstring=f"Sets one or more properties for an instance of {class_name}."
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = create_command_file(
            class_info=class_info, 
            method_info=method_info_set_props, 
            output_dir=tmpdir, 
            source_dir='.', 
            is_set_all_properties=True,
            settable_properties_for_template=settable_props
        )
        assert os.path.exists(file_path)
        content = pathlib.Path(file_path).read_text()

        assert "class Input(BaseModel):" in content
        assert "sku: Optional[str] = Field(default=None, description=\"Stock Keeping Unit.\")" in content
        assert "quantity: Optional[int] = Field(default=None, description=\"Current stock quantity.\")" in content
        assert "location: Optional[Optional[str]] = Field(default=None, description=\"Optional. New value for location.\")" in content
        
        # Verify model_config is NOT included for Optional fields (we've removed it)
        assert "model_config = ConfigDict" not in content

        assert "class Output(BaseModel):\n        success: bool = Field(description=\"True if properties update was attempted.\")" in content
        
        # Verify property setters use attribute assignment
        assert "if input.sku is not None:" in content
        assert "app_instance.sku = input.sku" in content
        assert "if input.quantity is not None:" in content
        assert "app_instance.quantity = input.quantity" in content
        assert "if input.location is not None:" in content
        assert "app_instance.location = input.location" in content
        
        # Verify is_complete handling
        assert "if input.is_complete is not None:" in content
        assert f"app_instance.status = {class_name}.COMPLETE if input.is_complete else {class_name}.INCOMPLETE" in content
        
        assert "return Signature.Output(success=True)" in content
        assert "app_instance = session.command_context_for_response_generation" in content

def test_property_setter_uses_attribute_assignment():
    # sourcery skip: extract-method, use-next
    """Test that property setters use attribute assignment instead of method calls."""
    class_name = "TodoItem"
    class_info = ClassInfo(class_name, 'application/todo_item.py')

    method_info = MethodInfo(
        name="assign_to",
        parameters=[{'name': 'assign_to', 'annotation': 'str'}],
        docstring="Set the person assigned to the todo item."
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = create_command_file(
            class_info=class_info,
            method_info=method_info,
            output_dir=tmpdir,
            source_dir='.',
            is_property_setter=True
        )

        # Parse the generated file to check for attribute assignment
        with open(file_path, 'r') as f:
            tree = ast.parse(f.read())

        # Find the _process_command method
        process_command = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == '_process_command':
                process_command = node
                break

        assert process_command is not None

        found_attribute_assignment = any(
            isinstance(node, ast.Assign)
            and (
                isinstance(node.targets[0], ast.Attribute)
                and isinstance(node.targets[0].value, ast.Name)
            )
            and (
                node.targets[0].value.id == 'app_instance'
                and node.targets[0].attr == 'assign_to'
            )
            for node in ast.walk(process_command)
        )
        assert found_attribute_assignment, "Property setter should use attribute assignment"
        
        # Verify that property setter uses property name as input parameter name
        content = pathlib.Path(file_path).read_text()
        assert "assign_to: str = Field" in content
        assert "app_instance.assign_to = input.assign_to" in content

def test_response_uses_model_dump_json():
    """Test that response uses output.model_dump_json()."""
    class_info, method_info = make_class_and_method('User', 'get_details', docstring='Get user details.')
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = create_command_file(class_info, method_info, tmpdir, source_dir='.')

        # Parse the generated file to check for model_dump_json
        with open(file_path, 'r') as f:
            tree = ast.parse(f.read())

        call_method = next(
            (
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef)
                and node.name == '__call__'
            ),
            None,
        )
        assert call_method is not None

        found_model_dump_json = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'model_dump_json'
            for node in ast.walk(call_method)
        )
        assert found_model_dump_json, "Response should use output.model_dump_json()"

def test_no_model_config_when_not_needed():
    """Test that model_config is not included when not needed."""
    class_info, method_info = make_class_and_method('User', 'get_details', docstring='Get user details.')
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = create_command_file(class_info, method_info, tmpdir, source_dir='.')
        content = pathlib.Path(file_path).read_text()
        
        # model_config should not be included for simple Input classes
        assert "model_config = ConfigDict" not in content

def test_no_input_class_for_parameterless_methods():
    """Test that Input class is not included for methods without parameters."""
    class_info, method_info = make_class_and_method('User', 'refresh', docstring='Refresh user data.')
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = create_command_file(class_info, method_info, tmpdir, source_dir='.')
        content = pathlib.Path(file_path).read_text()
        
        # Input class should not be included
        assert "class Input(BaseModel):" not in content
        
        # _process_command and __call__ should not have input parameters
        assert "def _process_command(self, session: Session) -> Signature.Output:" in content
        assert "def __call__(self, session: Session, command: str) -> CommandOutput:" in content

def test_no_unnecessary_comments():
    """Test that unnecessary comments are not included."""
    class_info, method_info = make_class_and_method('User', 'get_details', docstring='Get user details.')
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = create_command_file(class_info, method_info, tmpdir, source_dir='.')
        content = pathlib.Path(file_path).read_text()
        
        # The comment "# Access the application class instance:" should not be present
        assert "# Access the application class instance:" not in content 
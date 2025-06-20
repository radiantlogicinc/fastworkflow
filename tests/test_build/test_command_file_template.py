import os
import pathlib
import tempfile

import pytest
from fastworkflow.build.command_file_template import create_command_file
from fastworkflow.build.class_analysis_structures import ClassInfo, MethodInfo, PropertyInfo

def make_class_and_method(class_name, method_name, params=None, docstring=None, module_path='application/module.py', return_annotation=None):
    class_info = ClassInfo(class_name, module_path)
    method_info = MethodInfo(method_name, params or [], docstring=docstring, return_annotation=return_annotation)
    return class_info, method_info

def test_create_command_file_simple():
    class_info, method_info = make_class_and_method('User', 'get_details', docstring='Get user details.')
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = create_command_file(class_info, method_info, tmpdir, source_dir='.')
        assert os.path.exists(file_path)
        content = pathlib.Path(file_path).read_text()
        assert 'class Input(BaseModel):' in content
        assert 'class ResponseGenerator' in content
        assert 'plain_utterances =' in content

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

        assert "class Input(BaseModel):\n        pass" in content
        assert "class Output(BaseModel):" in content
        assert "product_id: int = Field(description=\"The product ID.\")" in content
        assert "name: str = Field(description=\"The product name.\")" in content
        assert "price: float = Field(description=\"Value of property price\")" in content
        
        assert "# For get_properties, the primary logic is to gather attribute values" in content
        assert "return Signature.Output(product_id=app_instance.product_id, name=app_instance.name, price=app_instance.price)" in content
        assert f"app_instance = session.workflow_snapshot.context_object  # type: {class_name}" in content

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

        assert "class Output(BaseModel):\n        success: bool = Field(description=\"True if properties update was attempted.\")" in content
        
        assert "if input.sku is not None:" in content
        assert "setattr(app_instance, 'sku', input.sku)" in content
        assert "if input.quantity is not None:" in content
        assert "setattr(app_instance, 'quantity', input.quantity)" in content
        assert "if input.location is not None:" in content
        assert "setattr(app_instance, 'location', input.location)" in content
        assert "return Signature.Output(success=True)" in content
        assert f"app_instance = session.workflow_snapshot.context_object  # type: {class_name}" in content 
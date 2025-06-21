import os
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any

import pytest

from fastworkflow.build.context_model_generator import generate_context_model
from fastworkflow.build.class_analysis_structures import ClassInfo, MethodInfo, PropertyInfo

def make_class_with_methods_and_properties():
    class_info = ClassInfo('User', 'application/user.py')
    class_info.methods.append(MethodInfo('GetDetails', [{'name': 'user_id', 'annotation': 'int'}], docstring='Get details.'))
    class_info.methods.append(MethodInfo('Update', [{'name': 'user_id', 'annotation': 'int'}, {'name': 'email', 'annotation': 'str'}], docstring='Update user.'))
    
    # This property will trigger get_properties.py
    email_prop = PropertyInfo('Email', docstring='User email', type_annotation='str')
    class_info.properties.append(email_prop)
    
    # Explicitly make 'Email' settable to trigger set_properties.py
    # In a real scenario, this would be populated by resolve_inherited_properties based on _property_setters
    class_info.all_settable_properties.append(PropertyInfo('Email', docstring='User email to set', type_annotation='str'))

    return {'User': class_info}

def test_generate_context_model():
    classes = make_class_with_methods_and_properties()
    with tempfile.TemporaryDirectory() as tmpdir:
        # generate_context_model now returns the model dictionary and writes the file.
        model_data = generate_context_model(classes, tmpdir)
        
        # Verify the file was also written
        expected_file_path = os.path.join(tmpdir, "_commands/context_inheritance_model.json")
        assert os.path.exists(expected_file_path), f"Expected context model file not found at {expected_file_path}"

        # Use the returned model_data for content assertions
        # New schema checks
        assert 'inheritance' in model_data
        inheritance_block = model_data['inheritance']

        # '*' must be present inside inheritance block
        assert '*' in inheritance_block
        assert inheritance_block['*']['base'] == []

        # Now User should appear in inheritance block even with no base classes
        assert 'User' in inheritance_block
        assert inheritance_block['User']['base'] == []

        # No "/" keys should exist anywhere in the JSON
        json_as_str = json.dumps(model_data)
        assert '"/"' not in json_as_str, "Found deprecated '/' key in context model JSON"

        # Aggregation block should exist but might be empty
        assert 'aggregation' in model_data

        # Legacy checks remain commented out as they are not applicable to the current context model structure.
        
        # method_entry = next(e for e in model_data if e['type'] == 'method' and e['method_or_property'] == 'GetDetails')
        # assert method_entry['command_file'] == 'user_getdetails.py'
        # assert method_entry['input_model'] == 'UserGetdetailsInput'
        # assert method_entry['output_model'] == 'UserGetdetailsOutput'
        # assert 'Get details.' in method_entry['description']
        # prop_entry = next(e for e in model_data if e['type'] == 'property')
        # assert prop_entry['command_file'] == 'get_user_email.py'
        # assert prop_entry['input_model'] == 'UserGet_EmailInput'
        # assert prop_entry['output_model'] == 'UserGet_EmailOutput'
        # assert 'User email' in prop_entry['description'] 

class TestContextModelGenerator:
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_classes(self):
        """Create mock classes for testing."""
        # Base class
        base_class = ClassInfo(
            name="BaseClass",
            module_path="application/base_class.py",
            docstring="A base class",
            bases=[]
        )
        base_class.methods = [
            MethodInfo(
                name="base_method",
                parameters=[],
                docstring="A base method"
            )
        ]
        classes = {"BaseClass": base_class}
        # Derived class
        derived_class = ClassInfo(
            name="DerivedClass",
            module_path="application/derived_class.py",
            docstring="A derived class",
            bases=["BaseClass"]
        )
        derived_class.methods = [
            MethodInfo(
                name="derived_method",
                parameters=[],
                docstring="A derived method"
            )
        ]
        classes["DerivedClass"] = derived_class

        # Another class with no inheritance
        standalone_class = ClassInfo(
            name="StandaloneClass",
            module_path="application/standalone_class.py",
            docstring="A standalone class",
            bases=[]
        )
        standalone_class.methods = [
            MethodInfo(
                name="standalone_method",
                parameters=[],
                docstring="A standalone method"
            )
        ]
        classes["StandaloneClass"] = standalone_class

        return classes
    
    def test_generate_context_model_new_format(self, temp_dir, mock_classes):
        # sourcery skip: class-extract-method, extract-duplicate-method
        """Test that generate_context_model creates a model in the new format."""
        # Generate the context model
        model = generate_context_model(mock_classes, temp_dir)
        
        # Check the model structure
        assert "inheritance" in model
        assert "aggregation" in model
        
        # Check inheritance block
        inheritance = model["inheritance"]
        assert "*" in inheritance  # Global context should always be present
        assert inheritance["*"] == {"base": []}  # Global context has no base classes
        
        # Check DerivedClass inherits from BaseClass
        assert "DerivedClass" in inheritance
        assert inheritance["DerivedClass"] == {"base": ["BaseClass"]}
        
        # Check BaseClass has no base classes
        assert "BaseClass" in inheritance
        assert inheritance["BaseClass"] == {"base": []}
        
        # StandaloneClass should be in the model even with no base classes
        assert "StandaloneClass" in inheritance
        assert inheritance["StandaloneClass"] == {"base": []}
        
        # Check that the file was written
        model_path = os.path.join(temp_dir, "_commands/context_inheritance_model.json")
        assert os.path.exists(model_path)
        
        # Read the file and verify its contents
        with open(model_path, "r") as f:
            file_model = json.load(f)
        
        assert file_model == model
    
    def test_generate_context_model_preserves_aggregation(self, temp_dir, mock_classes):
        # sourcery skip: extract-duplicate-method
        """Test that generate_context_model preserves the aggregation block."""
        # Create a pre-existing context model with an aggregation block
        commands_dir = os.path.join(temp_dir, "_commands")
        os.makedirs(commands_dir, exist_ok=True)
        
        existing_model = {
            "inheritance": {
                "*": {"base": []},
                "BaseClass": {"base": []}
            },
            "aggregation": {
                "DerivedClass": {
                    "container": ["BaseClass"]
                }
            }
        }
        
        model_path = os.path.join(commands_dir, "context_inheritance_model.json")
        with open(model_path, "w") as f:
            json.dump(existing_model, f, indent=2)
        
        # Generate the new context model
        model = generate_context_model(mock_classes, temp_dir)
        
        # Check that the aggregation block was preserved
        assert "aggregation" in model
        assert "DerivedClass" in model["aggregation"]
        assert model["aggregation"]["DerivedClass"] == {"container": ["BaseClass"]}
        
        # Check that the inheritance block was updated
        assert "inheritance" in model
        assert "DerivedClass" in model["inheritance"]
        assert model["inheritance"]["DerivedClass"] == {"base": ["BaseClass"]}
        
        # Read the file and verify its contents
        with open(model_path, "r") as f:
            file_model = json.load(f)
        
        assert file_model == model
    
    def test_generate_context_model_no_existing_file(self, temp_dir, mock_classes):
        """Test that generate_context_model creates a new file if none exists."""
        # Generate the context model
        model = generate_context_model(mock_classes, temp_dir)
        
        # Check that the file was written
        model_path = os.path.join(temp_dir, "_commands/context_inheritance_model.json")
        assert os.path.exists(model_path)
        
        # Read the file and verify its contents
        with open(model_path, "r") as f:
            file_model = json.load(f)
        
        assert file_model == model
        assert "inheritance" in file_model
        assert "aggregation" in file_model
        assert file_model["aggregation"] == {} 
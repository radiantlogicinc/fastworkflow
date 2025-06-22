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
        # New schema checks - flat structure
        assert 'User' in model_data
        assert 'base' in model_data['User']
        assert model_data['User']['base'] == []

        # No "/" keys should exist anywhere in the JSON
        json_as_str = json.dumps(model_data)
        assert '"/"' not in json_as_str, "Found deprecated '/' key in context model JSON"
        
        # No inheritance or aggregation keys should exist
        assert 'inheritance' not in model_data
        assert 'aggregation' not in model_data

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
        """Test that generate_context_model creates a model in the new flat format."""
        # Generate the context model
        model = generate_context_model(mock_classes, temp_dir)
        
        # Check the model structure - flat format, no inheritance/aggregation keys
        assert "DerivedClass" in model
        assert "BaseClass" in model
        assert "StandaloneClass" in model
        
        # Check DerivedClass inherits from BaseClass
        assert model["DerivedClass"] == {"base": ["BaseClass"]}
        
        # Check BaseClass has no base classes
        assert model["BaseClass"] == {"base": []}
        
        # StandaloneClass should be in the model even with no base classes
        assert model["StandaloneClass"] == {"base": []}
        
        # Check that the file was written
        model_path = os.path.join(temp_dir, "_commands/context_inheritance_model.json")
        assert os.path.exists(model_path)
        
        # Read the file and verify its contents
        with open(model_path, "r") as f:
            file_model = json.load(f)
        
        assert file_model == model
    
    def test_generate_context_model_preserves_existing_entries(self, temp_dir, mock_classes):
        """Test that generate_context_model preserves existing entries not in class analysis."""
        # Create a pre-existing context model
        commands_dir = os.path.join(temp_dir, "_commands")
        os.makedirs(commands_dir, exist_ok=True)
        
        existing_model = {
            "BaseClass": {"base": []},
            "ExistingClass": {"base": ["BaseClass"]},  # This class is not in mock_classes
        }
        
        model_path = os.path.join(commands_dir, "context_inheritance_model.json")
        with open(model_path, "w") as f:
            json.dump(existing_model, f, indent=2)
        
        # Generate the new context model
        model = generate_context_model(mock_classes, temp_dir)
        
        # Check that ExistingClass was preserved
        assert "ExistingClass" in model
        assert model["ExistingClass"] == {"base": ["BaseClass"]}
        
        # Check that the new classes were added
        assert "DerivedClass" in model
        assert model["DerivedClass"] == {"base": ["BaseClass"]}
        
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
        assert "DerivedClass" in file_model
        assert "BaseClass" in file_model
        assert "StandaloneClass" in file_model 
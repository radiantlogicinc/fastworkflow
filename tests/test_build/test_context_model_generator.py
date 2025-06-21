import os
import json
import tempfile
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

        # Since 'User' has no base classes, it should NOT appear in inheritance block
        assert 'User' not in inheritance_block

        # No "/" keys should exist anywhere in the JSON
        json_as_str = json.dumps(model_data)
        assert '"/"' not in json_as_str, "Found deprecated '/' key in context model JSON"

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
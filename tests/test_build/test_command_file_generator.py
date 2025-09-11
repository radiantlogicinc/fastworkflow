import os
import tempfile
from fastworkflow.build.command_file_generator import generate_command_files
from fastworkflow.build.class_analysis_structures import ClassInfo, MethodInfo, PropertyInfo
from fastworkflow.build.ast_class_extractor import resolve_inherited_properties
import pytest


def make_class_with_methods_and_properties():
    class_info = ClassInfo('User', 'application/user.py')
    class_info.methods.append(MethodInfo('GetDetails', [{'name': 'user_id', 'annotation': 'int'}], docstring='Get details.'))
    class_info.methods.append(MethodInfo('Update', [{'name': 'user_id', 'annotation': 'int'}, {'name': 'email', 'annotation': 'str'}], docstring='Update user.'))
    
    # This property will trigger get_properties.py
    email_prop = PropertyInfo('Email', docstring='User email', type_annotation='str')
    class_info.properties.append(email_prop)
    
    # Directly populate all_settable_properties to trigger set_properties.py
    # In a real scenario, this would be populated by resolve_inherited_properties based on _property_setters
    class_info.all_settable_properties.append(PropertyInfo('Email', docstring='User email to set', type_annotation='str'))
    # You might also want to add other settable properties if your class has more.
    # For this test, one is enough to trigger set_properties.py.

    return {'User': class_info}


def test_generate_command_files():
    classes_fixture = make_class_with_methods_and_properties()
    # Simulate what happens in __main__.py: resolve inherited properties AND settable properties
    # For this test, since we directly set up properties and all_settable_properties in the fixture,
    # and don't have complex inheritance, calling resolve_inherited_properties might be redundant
    # or could be used if the fixture was more complex.
    # For simplicity here, we assume the fixture provides ClassInfo objects in the state
    # generate_command_files expects them (i.e., .properties and .all_settable_properties are ready).

    with tempfile.TemporaryDirectory() as tmpdir:
        # Pass the prepared classes_fixture to generate_command_files
        files = generate_command_files(classes_fixture, tmpdir, source_dir='.') # Assuming source_dir is needed by create_command_file

        class_name_upper = 'User'
        class_specific_dir = os.path.join(tmpdir, class_name_upper)

        expected_files_map = {
            os.path.join(class_specific_dir, 'getdetails.py'): True,
            os.path.join(class_specific_dir, 'update.py'): True,
            os.path.join(class_specific_dir, 'get_properties.py'): True, # Changed from get_email.py
            os.path.join(class_specific_dir, 'set_properties.py'): True  # Added
        }

        assert os.path.isdir(class_specific_dir), f"Class specific directory {class_specific_dir} was not created."

        generated_file_names = {os.path.basename(f) for f in files}
        expected_file_names = {os.path.basename(k) for k in expected_files_map.keys()}

        assert generated_file_names == expected_file_names, \
            f"Expected files {expected_file_names}, but got {generated_file_names}"

        for f_path in files:
            assert os.path.exists(f_path), f"Expected file {f_path} does not exist."
            assert os.path.basename(f_path) == os.path.basename(f_path).lower(), \
                f"File name {os.path.basename(f_path)} is not lowercase."
        
        assert len(files) == len(expected_files_map), \
            f"Expected {len(expected_files_map)} files, but got {len(files)}"

# Removed dependency-graph-related tests and imports below to reflect feature deprecation. 
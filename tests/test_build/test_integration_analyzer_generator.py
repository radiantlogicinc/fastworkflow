import os
import shutil
import json
import pytest
from fastworkflow.build.ast_class_extractor import analyze_python_file, resolve_inherited_properties
from fastworkflow.build.command_file_generator import generate_command_files
from fastworkflow.build.context_model_generator import generate_context_model

INPUT_FOLDER = 'tests/example_workflow/application'
OUTPUT_DIR = 'tests/example_workflow/_commands'
CONTEXT_MODEL_FILE = 'tests/example_workflow/command_context_model.json'

@pytest.mark.integration
def test_analyzer_and_generators_on_all_files():
    # Clean output dir
    if os.path.exists(OUTPUT_DIR):
        # Ensure we delete subdirectories if they exist (e.g., from previous runs)
        for item in os.listdir(OUTPUT_DIR):
            item_path = os.path.join(OUTPUT_DIR, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            elif os.path.isfile(item_path):
                os.remove(item_path)
    else:
        os.makedirs(OUTPUT_DIR, exist_ok=True) # exist_ok=True in case of parallel test runs or other creation logic
    
    # Clean context model file
    if os.path.exists(CONTEXT_MODEL_FILE):
        os.remove(CONTEXT_MODEL_FILE)

    # Analyze all files in the input folder and merge classes
    classes = {}
    for fname in os.listdir(INPUT_FOLDER):
        if fname.endswith('.py') and fname != '__init__.py': # Exclude __init__.py if it doesn't contain relevant classes
            file_path = os.path.join(INPUT_FOLDER, fname)
            file_classes = analyze_python_file(file_path)
            for cname, cinfo in file_classes.items():
                if cname in classes:
                    # This might be too strict if classes are intentionally split for organization.
                    # For now, let's assume unique class names across files for simplicity of this test.
                    print(f"Warning: Duplicate class name {cname} found. Overwriting from {file_path}.")
                classes[cname] = cinfo
    assert classes, "No classes found in input folder."

    # Resolve inherited properties and identify all settable properties for each class
    resolve_inherited_properties(classes)

    # Generate command files
    # generate_command_files is expected to create subdirectories per class in OUTPUT_DIR
    generated_files_paths = generate_command_files(classes, OUTPUT_DIR, INPUT_FOLDER)

    all_class_names = set(classes.keys())
    for class_name, class_info in classes.items():
        class_specific_output_dir = os.path.join(OUTPUT_DIR, class_name)
        assert os.path.isdir(class_specific_output_dir), f"Class specific directory {class_specific_output_dir} was not created for class {class_name}"

        expected_simple_filenames = []
        for method_info in class_info.methods:
            expected_simple_filenames.append(f'{method_info.name.lower()}.py')
        
        if class_info.properties: # Check if there are any properties (could be direct or inherited)
            expected_simple_filenames.append('get_properties.py')
        if class_info.all_settable_properties: # Check if there are any settable properties
            expected_simple_filenames.append('set_properties.py')
        
        generated_in_subdir = os.listdir(class_specific_output_dir)
        assert sorted(generated_in_subdir) == sorted(expected_simple_filenames), \
            f"File list mismatch in {class_specific_output_dir}. Expected {sorted(expected_simple_filenames)}, got {sorted(generated_in_subdir)}"

        for esf in expected_simple_filenames:
            expected_file_path = os.path.join(class_specific_output_dir, esf)
            assert os.path.exists(expected_file_path), f"File {expected_file_path} does not exist for class {class_name}"

    # Generate context model
    # generate_context_model writes the file and returns the dictionary
    context_data = generate_context_model(classes, os.path.dirname(CONTEXT_MODEL_FILE), os.path.basename(CONTEXT_MODEL_FILE))
    assert os.path.exists(CONTEXT_MODEL_FILE), f"Context model file {CONTEXT_MODEL_FILE} was not created."
    
    # Now, context_data is the loaded dictionary, no need to open and json.load again

    # Context model schema v2 validations
    assert 'inheritance' in context_data, "Missing 'inheritance' block in context model"
    inheritance_block = context_data['inheritance']

    # '*' entry must exist
    assert '*' in inheritance_block and inheritance_block['*']['base'] == [], "'*' entry missing or malformed in inheritance block"

    for class_name, class_info in classes.items():
        expected_bases = [b for b in class_info.bases if b in all_class_names]

        if expected_bases:
            # Class should appear in inheritance block with correct bases
            assert class_name in inheritance_block, f"{class_name} missing from inheritance block"
            assert inheritance_block[class_name]['base'] == expected_bases, f"Base for {class_name} should be {expected_bases}, got {inheritance_block[class_name]['base']}"
        else:
            # Classes with no bases should not be included (to keep file small)
            assert class_name not in inheritance_block, f"{class_name} should not be in inheritance block as it has no bases"

    # Ensure no old '/' keys exist anywhere
    json_as_str = json.dumps(context_data)
    assert '"/"' not in json_as_str, "Deprecated '/' key found in context model JSON" 
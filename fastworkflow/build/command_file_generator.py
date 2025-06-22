import os
import ast
from typing import Dict, List, Tuple
from fastworkflow.build.class_analysis_structures import ClassInfo, MethodInfo, PropertyInfo, FunctionInfo
from fastworkflow.build.command_file_template import create_command_file, create_function_command_file
from fastworkflow.utils.python_utils import get_module_import_path, find_module_dependencies
from fastworkflow.command_context_model import CommandContextModel
import importlib.util
import traceback
import json
from .ast_class_extractor import parse_google_docstring

def generate_command_files(classes: Dict[str, ClassInfo], output_dir: str, source_dir: str, overwrite: bool = False, functions: Dict[str, FunctionInfo] = None) -> List[str]:
    """Generate command files for all public methods, properties, and top-level functions in the analyzed code."""
    os.makedirs(output_dir, exist_ok=True)
    generated_files = []

    # Create a map from class names to their module paths
    class_name_to_module_map = {name: c_info.module_path for name, c_info in classes.items()}
    
    # Generate command files for classes
    for class_info in classes.values():
        class_output_dir = os.path.join(output_dir, class_info.name)
        os.makedirs(class_output_dir, exist_ok=True)

        # Methods
        for method_info in class_info.methods:
            file_name = f"{method_info.name.lower()}.py"
            file_path = os.path.join(class_output_dir, file_name)
            
            # Check if this method is a property setter
            is_property_setter = False
            if method_info.name in [prop.name for prop in class_info.all_settable_properties]:
                is_property_setter = True
                # For property setters, ensure the parameter name matches the property name
                if method_info.parameters and len(method_info.parameters) > 0:
                    # Replace the parameter name (usually 'value') with the property name
                    method_info.parameters[0]['name'] = method_info.name
            
            if generated_file_path := create_command_file(
                class_info=class_info,
                method_info=method_info,
                output_dir=class_output_dir,
                file_name=file_name,
                source_dir=source_dir,
                overwrite=overwrite,
                class_name_to_module_map=class_name_to_module_map,
                is_property_setter=is_property_setter,
            ):
                generated_files.append(generated_file_path)

        # Properties (Getters) - REVISED FOR get_properties
        if class_info.properties: # If the class has any properties
            # Synthesize a MethodInfo for the 'get_properties' command
            get_all_method_info = MethodInfo(
                name="GetProperties", # Will be lowercased to get_properties.py by create_command_file or naming convention
                parameters=[], 
                docstring=f"Get all properties of the {class_info.name} class.",
                return_annotation="Dict[str, Any]", # Placeholder, actual Output model defined in template
                decorators=[],
                docstring_parsed=parse_google_docstring(f"Get all properties of the {class_info.name} class.")
            )
            file_name_get_all = "get_properties.py"
            if generated_get_all_path := create_command_file(
                method_info=get_all_method_info,
                class_info=class_info,
                output_dir=class_output_dir,
                file_name="get_properties.py",  # Explicitly set the filename
                class_name_to_module_map=class_name_to_module_map,
                is_get_all_properties=True,
                all_properties_for_template=class_info.properties,  # Pass all properties for the template
                source_dir=source_dir,  # Pass source_dir
                overwrite=overwrite,  # Pass overwrite
            ):
                generated_files.append(generated_get_all_path)

        # Generate 'set_properties' command if there are settable properties
        if class_info.all_settable_properties:
            set_properties_params = []
            parsed_doc_params = {}
            for prop in class_info.all_settable_properties:
                set_properties_params.append({
                    'name': prop.name,
                    'annotation': prop.type_annotation, # Template will handle making this Optional
                    'is_optional': True # Add this flag for the template
                })
                parsed_doc_params[prop.name] = f"Optional. New value for the '{prop.name}' property."

            set_properties_method_info = MethodInfo(
                name="SetProperties",
                parameters=set_properties_params,
                return_annotation="Dict[str, bool]", # e.g., {"success": True}
                docstring=f"Sets one or more properties for an instance of {class_info.name}.",
                docstring_parsed={
                    "summary": f"Sets one or more properties for an instance of {class_info.name}.",
                    "params": parsed_doc_params,
                    "returns": {"success": "True if the operation was successful (or attempted)."} # Simplified return doc
                }
            )

            if generated_set_all_path := create_command_file(
                method_info=set_properties_method_info,
                class_info=class_info,
                output_dir=class_output_dir,  # Use the existing class-specific output directory
                file_name="set_properties.py",  # Explicitly set the filename
                class_name_to_module_map=class_name_to_module_map,
                is_set_all_properties=True,  # New flag
                # Pass the list of settable properties for the template to use
                settable_properties_for_template=class_info.all_settable_properties,
                source_dir=source_dir,  # Pass source_dir
                overwrite=overwrite,  # Pass overwrite
            ):
                generated_files.append(generated_set_all_path)
    
    # Generate command files for top-level functions
    if functions:
        # Global commands should be placed directly in the `output_dir` (which is the _commands folder)
        commands_dir = output_dir  # Avoid creating nested _commands
        
        for function_info in functions.values():
            file_name = f"{function_info.name}.py"
            file_path = os.path.join(commands_dir, file_name)
            
            if generated_file_path := create_function_command_file(
                function_info=function_info,
                output_dir=commands_dir,
                file_name=file_name,
                source_dir=source_dir,
                overwrite=overwrite
            ):
                generated_files.append(generated_file_path)

    return generated_files

def validate_python_syntax_in_dir(directory: str) -> List[tuple]:
    """
    Validate Python syntax for all .py files in the given directory.
    Returns a list of (file_path, error_message) for files with syntax errors.
    Prints errors if any are found.
    """
    errors = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        source = f.read()
                    ast.parse(source, filename=file_path)
                except SyntaxError as e:
                    error_msg = f"Syntax error in {file_path} at line {e.lineno}: {e.msg}"
                    errors.append((file_path, error_msg))
                except Exception as e:
                    error_msg = f"Error parsing {file_path}: {str(e)}"
                    errors.append((file_path, error_msg))
    if errors:
        print("Python syntax validation errors found:")
        for file_path, error_msg in errors:
            print(f"  - {file_path}: {error_msg}")
    return errors

def validate_command_file_components_in_dir(directory: str) -> list:
    """
    Validate that each .py file in the directory contains required FastWorkflow command file components.
    Returns a list of (file_path, error_message) for files missing required components.
    Prints errors if any are found.
    
    Files starting with underscore (_) are skipped as they are not command files.
    Files named 'startup.py' are also skipped as they have a different structure.
    """
    import ast
    errors = []
    for root, _, files in os.walk(directory):
        for file in files:
            # Skip files starting with underscore, __init__.py, and startup.py
            if file.startswith('_') or file == '__init__.py' or file == 'startup.py':
                continue
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        source = f.read()
                    tree = ast.parse(source, filename=file_path)

                    # Track findings
                    has_signature = False
                    has_input = False
                    has_output = False
                    input_inherits_basemodel = False
                    output_inherits_basemodel = False
                    has_model_config = False
                    has_plain_utterances = False
                    has_generate_utterances = False
                    has_process_extracted_parameters = False
                    has_response_generator = False
                    has_response_call = False

                    for node in tree.body:
                        # Signature class
                        if isinstance(node, ast.ClassDef) and node.name == 'Signature':
                            has_signature = True
                            for subnode in node.body:
                                # Input class
                                if isinstance(subnode, ast.ClassDef) and subnode.name == 'Input':
                                    has_input = True
                                    # Check inheritance
                                    for base in subnode.bases:
                                        if (isinstance(base, ast.Name) and base.id == 'BaseModel') or \
                                           (isinstance(base, ast.Attribute) and base.attr == 'BaseModel'):
                                            input_inherits_basemodel = True
                                    # Check for model_config
                                    for item in subnode.body:
                                        if isinstance(item, ast.Assign):
                                            for target in item.targets:
                                                if isinstance(target, ast.Name) and target.id == 'model_config':
                                                    has_model_config = True
                                # Output class
                                if isinstance(subnode, ast.ClassDef) and subnode.name == 'Output':
                                    has_output = True
                                    for base in subnode.bases:
                                        if (isinstance(base, ast.Name) and base.id == 'BaseModel') or \
                                           (isinstance(base, ast.Attribute) and base.attr == 'BaseModel'):
                                            output_inherits_basemodel = True
                                # process_extracted_parameters method
                                if isinstance(subnode, ast.FunctionDef) and subnode.name == 'process_extracted_parameters':
                                    has_process_extracted_parameters = True
                            # plain_utterances and template_utterances
                            for subnode in node.body:
                                if isinstance(subnode, ast.Assign):
                                    for target in subnode.targets:
                                        if isinstance(target, ast.Name) and target.id == 'plain_utterances':
                                            has_plain_utterances = True
                            # generate_utterances static method
                            for subnode in node.body:
                                if isinstance(subnode, ast.FunctionDef) and subnode.name == 'generate_utterances':
                                    has_generate_utterances = True
                        # ResponseGenerator class
                        if isinstance(node, ast.ClassDef) and node.name == 'ResponseGenerator':
                            has_response_generator = True
                            for subnode in node.body:
                                if isinstance(subnode, ast.FunctionDef) and subnode.name == '__call__':
                                    has_response_call = True
                    # Collect errors
                    if not has_signature:
                        errors.append((file_path, 'Missing Signature class'))
                        continue
                    if not has_input:
                        errors.append((file_path, 'Missing Input class in Signature'))
                    elif not input_inherits_basemodel:
                        errors.append((file_path, 'Input class does not inherit from BaseModel'))
                    if not has_output:
                        errors.append((file_path, 'Missing Output class in Signature'))
                    elif not output_inherits_basemodel:
                        errors.append((file_path, 'Output class does not inherit from BaseModel'))
                    if not has_plain_utterances:
                        errors.append((file_path, 'Missing plain_utterances in Signature'))
                    if not has_generate_utterances:
                        errors.append((file_path, 'Missing generate_utterances static method in Signature'))
                    if not has_process_extracted_parameters:
                        errors.append((file_path, 'Missing process_extracted_parameters method in Signature'))
                    if not has_response_generator:
                        errors.append((file_path, 'Missing ResponseGenerator class'))
                    elif not has_response_call:
                        errors.append((file_path, 'ResponseGenerator missing __call__ method'))
                except Exception as e:
                    errors.append((file_path, f'Error during component validation: {str(e)}'))
    if errors:
        print("Command file component validation errors found:")
        for file_path, error_msg in errors:
            print(f"  - {file_path}: {error_msg}")
    return errors

EXCLUDE_DIRS = {"__pycache__", ".venv", ".git", ".vscode", ".DS_Store", "node_modules"}

def verify_commands_against_context_model(
    context_model: Dict, 
    commands_dir: str,
    classes_info: Dict[str, ClassInfo]
) -> list:
    """
    Verifies that the context model and directory structure are in sync according to the new schema.
    
    Validation rules:
    1. Every context in the model must have a corresponding directory
    2. Every context directory must be listed in the model
    3. Base classes referenced in the model must exist in the model
    4. Classes in the model should exist in the analyzed class information
    
    Args:
        context_model: The context model dictionary (flat structure with context classes at top level)
        commands_dir: Path to the commands directory
        classes_info: Dictionary mapping class names to ClassInfo objects
        
    Returns:
        list: List of error messages, empty if validation passes
    """
    errors = []
    
    # Get the context names from the model (excluding special entries)
    context_model_classes = set(context_model.keys())
    context_model_classes.discard("*")  # Ignore the global '*' context for directory checks
    
    # Get directories from the filesystem
    present_class_dirs = set()
    for item in os.listdir(commands_dir):
        if item in EXCLUDE_DIRS or item.startswith('_') or item.endswith('.py') or item.endswith('.json') or item.endswith('.md'):
            continue
        item_path = os.path.join(commands_dir, item)
        if os.path.isdir(item_path):  # Item is a potential class directory
            present_class_dirs.add(item)
    
    # Rule 1: Every context in the model must have a corresponding directory
    for class_name in context_model_classes:
        if class_name not in present_class_dirs:
            errors.append(f"Class '{class_name}' is in context model but has no directory in {commands_dir}")
        
        # Rule 4: Classes in model should exist in analyzed class information
        if class_name not in classes_info:
            errors.append(f"Class '{class_name}' is in context model but not found in analyzed class information")
    
    # Rule 2: Every context directory must be listed in the model
    for class_dir_name in present_class_dirs:
        if class_dir_name not in context_model_classes:
            # Check if this directory corresponds to a known class
            if class_dir_name in classes_info:
                errors.append(f"Directory '{class_dir_name}' exists in {commands_dir} but the class is not in the context model")
            else:
                errors.append(f"Directory '{class_dir_name}' exists in {commands_dir} but does not correspond to any known class")
    
    # Rule 3: Base classes referenced in the model must exist in the model
    for class_name, class_data in context_model.items():
        if class_name == "*":  # Skip global context
            continue
            
        base_classes = class_data.get("base", [])
        if not base_classes:
            continue  # No base classes to check
            
        for base_class in base_classes:
            if base_class not in context_model and base_class != "*":
                errors.append(f"Class '{class_name}' inherits from '{base_class}', but '{base_class}' is not in the context model")
            
            # Additional check for base class in class_info
            if base_class != "*" and base_class not in classes_info:
                errors.append(f"Class '{class_name}' inherits from '{base_class}', but '{base_class}' is not in the analyzed class information")
    
    return errors

def validate_command_imports(commands_dir: str) -> bool:
    """
    Validate that all command files can be imported without error.
    Returns True if all files import successfully, else False.
    """
    import os
    errors = []
    for root, _, files in os.walk(commands_dir):
        for file in files:
            if file.endswith('.py') and file != '__init__.py':
                file_path = os.path.join(root, file)
                module_name = os.path.splitext(file)[0]
                try:
                    spec = importlib.util.spec_from_file_location(module_name, file_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                except Exception as e:
                    errors.append((file_path, traceback.format_exc()))
    if errors:
        print("Import errors found in command files:")
        for file_path, tb in errors:
            print(f"  - {file_path}:\n{tb}")
    else:
        print("All command files imported successfully.")
    return not errors 
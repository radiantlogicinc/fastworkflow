import os
import ast
from typing import Dict, List
from fastworkflow.build.class_analysis_structures import ClassInfo, MethodInfo, PropertyInfo
from fastworkflow.build.command_file_template import create_command_file
from fastworkflow.utils.python_utils import get_module_import_path, find_module_dependencies
from fastworkflow.command_context_model import CommandContextModel
import importlib.util
import traceback
import json
from .ast_class_extractor import parse_google_docstring

def generate_command_files(classes: Dict[str, ClassInfo], output_dir: str, source_dir: str, overwrite: bool = False) -> List[str]:
    """Generate command files for all public methods and properties in the analyzed classes."""
    os.makedirs(output_dir, exist_ok=True)
    generated_files = []

    # Create a map from class names to their module paths
    class_name_to_module_map = {name: c_info.module_path for name, c_info in classes.items()}

    for class_info in classes.values():
        class_output_dir = os.path.join(output_dir, class_info.name)
        os.makedirs(class_output_dir, exist_ok=True)

        # Methods
        for method_info in class_info.methods:
            file_name = f"{method_info.name.lower()}.py"
            file_path = os.path.join(class_output_dir, file_name)
            # Pass the map to create_command_file
            generated_file_path = create_command_file(
                class_info=class_info, 
                method_info=method_info, 
                output_dir=class_output_dir, 
                file_name=file_name, 
                source_dir=source_dir, 
                overwrite=overwrite, 
                class_name_to_module_map=class_name_to_module_map
            )
            if generated_file_path:
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
            # Pass a new flag is_get_all_properties=True to create_command_file
            # Also need to pass all properties of the class_info to create_command_file so it can build the Output model.
            # Let's assume create_command_file can access class_info.properties directly when this flag is true.
            generated_get_all_path = create_command_file(
                method_info=get_all_method_info,
                class_info=class_info,
                output_dir=class_output_dir,
                file_name="get_properties.py", # Explicitly set the filename
                class_name_to_module_map=class_name_to_module_map,
                is_get_all_properties=True,
                all_properties_for_template=class_info.properties, # Pass all properties for the template
                source_dir=source_dir, # Pass source_dir
                overwrite=overwrite    # Pass overwrite
            )
            if generated_get_all_path:
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
            
            # Ensure the command file is placed in the correct class-specific directory
            # output_dir_for_class (now class_output_dir) is already defined earlier for the current class_info
            
            generated_set_all_path = create_command_file(
                method_info=set_properties_method_info,
                class_info=class_info,
                output_dir=class_output_dir, # Use the existing class-specific output directory
                file_name="set_properties.py", # Explicitly set the filename
                class_name_to_module_map=class_name_to_module_map,
                is_set_all_properties=True, # New flag
                # Pass the list of settable properties for the template to use
                settable_properties_for_template=class_info.all_settable_properties,
                source_dir=source_dir, # Pass source_dir
                overwrite=overwrite    # Pass overwrite
            )
            if generated_set_all_path:
                generated_files.append(generated_set_all_path)

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
    """
    import ast
    errors = []
    for root, _, files in os.walk(directory):
        for file in files:
            # Skip files starting with underscore and __init__.py
            if file.startswith('_') or file == '__init__.py':
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
                    elif not has_model_config:
                        errors.append((file_path, 'Input class missing model_config assignment'))
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

def _is_command_defined_in_class(command_name: str, class_info: ClassInfo) -> bool:
    """Helper to check if a command (method or property) is directly defined in a class."""
    # Check methods
    if command_name in [m.name.lower() for m in class_info.methods]:
        return True
    # Check properties (getters)
    if command_name.startswith("get_") and command_name[4:] in [p.name.lower() for p in class_info.properties]:
        return True
    # Check properties (setters) - assumes _property_setters stores original names
    if command_name.startswith("set_") and command_name[4:] in [p_name.lower() for p_name in getattr(class_info, '_property_setters', [])]:
        return True
    return False

def verify_commands_against_context_model(
    context_model: Dict, 
    commands_dir: str,
    classes_info: Dict[str, ClassInfo] # New parameter
) -> list:
    """
    Verifies that for every command listed in the context model for a class,
    a corresponding .py file exists in the expected class-specific command directory
    (or in a base class's directory for inherited commands).
    Also checks if there are any .py files in the command directories that are not
    listed in the context model.
    """
    errors = []
    
    context_model_classes = set(context_model.keys())
    context_model_classes.discard("*") # Ignore the global '*' context for this check

    filesystem_commands_by_class: Dict[str, set] = {}
    present_class_dirs = set()

    for item in os.listdir(commands_dir):
        if item in EXCLUDE_DIRS:
            continue
        item_path = os.path.join(commands_dir, item)
        if os.path.isdir(item_path): # Item is a potential class directory
            class_name_fs = item
            present_class_dirs.add(class_name_fs)
            filesystem_commands_by_class[class_name_fs] = set()
            for cmd_file in os.listdir(item_path):
                if cmd_file.endswith(".py") and cmd_file != "__init__.py":
                    command_name = cmd_file[:-3] # Remove .py
                    filesystem_commands_by_class[class_name_fs].add(command_name)

    # If context_model is in new schema (has 'inheritance' key and no per-class '/') we skip deep command checks for now.
    if 'inheritance' in context_model and not any('/' in k for k in context_model.keys()):
        # TODO: implement inheritance-only validation in future iteration
        return []

    # 1. Check: Commands in Context Model vs. Filesystem (Handles Inheritance)
    for class_name_ctx in context_model_classes:
        if class_name_ctx not in classes_info:
            errors.append(f"Class '{class_name_ctx}' is in context model but not found in analyzed class information.")
            continue

        current_class_info = classes_info[class_name_ctx]
        commands_in_model_for_class = set(context_model.get(class_name_ctx, {}).get('/', []))
        missing_files_for_class = []

        for command_name_in_model in commands_in_model_for_class:
            expected_command_file_path = None
            # Check if defined in the current class itself
            if _is_command_defined_in_class(command_name_in_model, current_class_info):
                expected_command_file_path = os.path.join(commands_dir, class_name_ctx, f"{command_name_in_model}.py")
            else:
                # Check base classes by MRO (already pre-calculated in ClassInfo.mro if available, or use base_classes)
                # For simplicity, we'll iterate base_classes. A full MRO check would be more robust.
                # Note: ClassInfo.base_classes lists immediate parents. A recursive search or MRO list is better.
                # Assuming ast_class_extractor provides base_classes in MRO order or ClassInfo has an MRO list.
                # Let's use current_class_info.bases which should be direct parents.
                # A proper MRO traversal is needed for multi-level inheritance.
                # For now, let's assume a simplified MRO if ClassInfo doesn't provide it directly.
                
                mro_paths_to_check = [class_name_ctx] # Own class first
                
                # Simple MRO: add base classes. For complex cases, ClassInfo should provide a resolved MRO list.
                processed_bases = set()
                bases_to_check = list(current_class_info.bases)
                while bases_to_check:
                    base_name = bases_to_check.pop(0)
                    if base_name in processed_bases or base_name not in classes_info:
                        continue
                    mro_paths_to_check.append(base_name)
                    processed_bases.add(base_name)
                    # Add grandparents etc.
                    # bases_to_check.extend(classes_info[base_name].bases) # Full MRO if needed (also changed here for consistency if uncommented)


                found_in_hierarchy = False
                for owner_class_name in mro_paths_to_check:
                    owner_class_info = classes_info.get(owner_class_name)
                    if owner_class_info and _is_command_defined_in_class(command_name_in_model, owner_class_info):
                        # Command is defined in this class (could be current_class_ctx or a base)
                        # The physical file should be in this owner_class_name's directory
                        path_to_check = os.path.join(commands_dir, owner_class_name, f"{command_name_in_model}.py")
                        if os.path.exists(path_to_check):
                            expected_command_file_path = path_to_check
                            found_in_hierarchy = True
                            break # Found the defining class and its command file
                
                if not found_in_hierarchy and not expected_command_file_path: # If not found after checking hierarchy
                     # Fallback: if not clearly defined, assume it should be in its own class dir (original behavior before fix)
                     # This might indicate an issue with _is_command_defined_in_class or context model generation
                     # For the problem at hand, if it's in the context model for class_name_ctx, and not in its own methods/props
                     # it MUST be inherited. The loop above should find it. If not, it's a deeper issue.
                     # Defaulting to its own path if not found in hierarchy to detect if command_file_generator missed it.
                    expected_command_file_path = os.path.join(commands_dir, class_name_ctx, f"{command_name_in_model}.py")


            if expected_command_file_path and not os.path.exists(expected_command_file_path):
                # Clarify if it was expected in its own dir or a base's
                if class_name_ctx != os.path.basename(os.path.dirname(expected_command_file_path)):
                    missing_files_for_class.append(f"{command_name_in_model} (expected in {os.path.basename(os.path.dirname(expected_command_file_path))})")
                else:
                    missing_files_for_class.append(command_name_in_model)
            elif not expected_command_file_path: # Should not happen if logic is correct
                 missing_files_for_class.append(f"{command_name_in_model} (unable to determine source class)")


        if missing_files_for_class:
            errors.append(f"Class '{class_name_ctx}': The following commands are in the context model but have no corresponding file: {sorted(list(missing_files_for_class))}")

    # 2. Check: Filesystem Commands vs. Context Model
    for class_name_fs, commands_on_fs_for_class in filesystem_commands_by_class.items():
        if class_name_fs not in context_model_classes:
            # Check if this class is known at all from AST analysis
            if class_name_fs in classes_info:
                 # It's a valid class, but not in context model (or has no commands listed)
                 if commands_on_fs_for_class: # Only error if there are actual command files
                    errors.append(f"Class directory '{class_name_fs}' exists with commands, but class is not in context model or has no commands listed.")
            # else: # class_name_fs is not a known class, could be an unrelated directory. Already handled by EXCLUDE_DIRS
            continue

        commands_in_model_for_class = set(context_model.get(class_name_fs, {}).get('/', []))
        
        # We only care about files directly in *this* class_name_fs's directory for this check.
        # Inherited commands are fine if their source files are in their base class's directory.
        # This check is for files in _commands/class_name_fs/ that shouldn't be there according to its *own* definition.
        
        extra_files_for_class = []
        for cmd_fs in commands_on_fs_for_class:
            # Is this command directly defined by class_name_fs?
            is_direct_member = False
            if class_name_fs in classes_info:
                class_info_fs = classes_info[class_name_fs]
                if _is_command_defined_in_class(cmd_fs, class_info_fs):
                    is_direct_member = True
            
            if is_direct_member and cmd_fs not in commands_in_model_for_class:
                extra_files_for_class.append(cmd_fs)
            elif not is_direct_member and cmd_fs in commands_in_model_for_class:
                # This is an odd case: command is in model for this class, file is in this class's dir,
                # but it's NOT a direct member. This implies it's an inherited command whose file
                # was mistakenly placed in the derived class's command dir.
                # This shouldn't happen if generate_command_files is correct.
                # For now, this check focuses on files that are truly "extra" / not in model for direct def.
                pass # This file might be an inherited one, correctly listed in model.
                     # The previous check (model vs filesystem) handles if this file *should* exist.
                     # This check is more about "is this file an orphan for THIS class's direct definition?"

        # Re-evaluating this check: if a file cmd_fs exists in _commands/class_name_fs/,
        # and class_name_fs is in the context model, then cmd_fs should be listed under
        # class_name_fs in the context model IF it's a command directly defined by class_name_fs.
        # If it's an inherited command, its file should NOT be in _commands/class_name_fs/.
        
        truly_extra_files = []
        if class_name_fs in classes_info:
            class_info_obj = classes_info[class_name_fs]
            for cmd_on_fs in commands_on_fs_for_class:
                is_direct = _is_command_defined_in_class(cmd_on_fs, class_info_obj)
                is_in_model_for_this_class = cmd_on_fs in commands_in_model_for_class

                if is_direct and not is_in_model_for_this_class:
                    # Directly defined, but not in its own class's model entry. Context model gen issue?
                    truly_extra_files.append(f"{cmd_on_fs} (defined here but not in its context model entry)")
                elif not is_direct and os.path.exists(os.path.join(commands_dir, class_name_fs, f"{cmd_on_fs}.py")):
                    # Not directly defined by this class, but a file exists in its command dir.
                    # This is an incorrectly placed file for an inherited command.
                    # generate_command_files should not do this.
                    # Check if it's in the model for this class (meaning it's an expected inherited command)
                    if is_in_model_for_this_class:
                         truly_extra_files.append(f"{cmd_on_fs} (inherited, but file incorrectly in this class's dir)")
                    # else: # Not direct, not in model for this class = truly an orphan file
                    #    truly_extra_files.append(f"{cmd_on_fs} (orphan file)")


        if truly_extra_files: # Use the refined list
             errors.append(f"Class '{class_name_fs}': The following command files are present but not expected or misplaced: {sorted(list(truly_extra_files))}")


    # 3. Check for class directories that exist but class is not in context model (already partially handled)
    fs_class_dirs_not_in_model = present_class_dirs - context_model_classes
    for class_dir_name in fs_class_dirs_not_in_model:
        # Further check: is class_dir_name a known class from AST analysis?
        if class_dir_name in classes_info:
            # If it has actual command files, it's an error. Empty dir might be an artifact.
            if filesystem_commands_by_class.get(class_dir_name): # Check if the set is non-empty
                 errors.append(f"Command directory for class '{class_dir_name}' exists with commands, but the class is not listed in the context model or has no commands.")
        # else: It's a directory not corresponding to any known class, and not in EXCLUDE_DIRS. Could be an issue.
        #    errors.append(f"Unknown directory '{class_dir_name}' found in commands directory.")


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
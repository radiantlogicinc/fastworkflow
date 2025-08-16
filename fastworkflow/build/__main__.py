import argparse
import os
import sys
import glob
import ast

import fastworkflow
from fastworkflow.build.command_file_generator import validate_python_syntax_in_dir, validate_command_file_components_in_dir, verify_commands_against_context_model, validate_command_imports
from fastworkflow.build import ast_class_extractor
from fastworkflow.build.command_file_generator import generate_command_files as real_generate_command_files
from fastworkflow.build.context_model_generator import generate_context_model as real_generate_context_model
from fastworkflow.build.context_folder_generator import ContextFolderGenerator
from fastworkflow.build.command_stub_generator import CommandStubGenerator
from fastworkflow.build.navigator_stub_generator import NavigatorStubGenerator
from fastworkflow.utils.logging import logger

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate FastWorkflow command files and context model from a Python application."
    )
    parser.add_argument('--app-dir', '-s', required=True, help='Path to the source code directory of the application')
    parser.add_argument('--workflow-folderpath', '-w', required=True, help='Path to the workflow folder where commands will be generated')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite files in output directory if present')
    parser.add_argument('--stub-commands', help='Comma-separated list of command names to generate stubs for')
    parser.add_argument('--no-startup', action='store_true', help='Skip generating the startup.py file')
    return parser.parse_args()

def validate_directories(args):
    if not os.path.isdir(args.app_dir):
        print(f"Error: Application directory '{args.app_dir}' does not exist or is not accessible.")
        sys.exit(1)
    if not os.access(args.app_dir, os.R_OK):
        print(f"Error: Application directory '{args.app_dir}' is not readable.")
        sys.exit(1)
    
    # Create workflow folder if it doesn't exist
    if not os.path.isdir(args.workflow_folderpath):
        try:
            os.makedirs(args.workflow_folderpath, exist_ok=True)
            print(f"Created workflow directory: {args.workflow_folderpath}")
        except Exception as e:
            print(f"Error: Could not create workflow directory '{args.workflow_folderpath}': {e}")
            sys.exit(1)
    
    # Ensure the _commands directory exists under the workflow folder
    commands_dir = os.path.join(args.workflow_folderpath, "_commands")
    try:
        os.makedirs(commands_dir, exist_ok=True)
        print(f"Ensured _commands directory exists: {commands_dir}")
    except Exception as e:
        print(f"Error: Could not create _commands directory '{commands_dir}': {e}")
        sys.exit(1)
    
    # Create __init__.py in _commands directory if it doesn't exist
    init_path = os.path.join(commands_dir, "__init__.py")
    if not os.path.exists(init_path):
        with open(init_path, "w") as f:
            f.write("")

def run_command_generation(args):
    # Ensure workflow_folderpath is a Python package
    init_path = os.path.join(args.workflow_folderpath, "__init__.py")
    if not os.path.exists(init_path):
        with open(init_path, "w") as f:
            f.write("")
    
    # Define the commands directory
    commands_dir = os.path.join(args.workflow_folderpath, "_commands")
    
    # Generate command files
    py_files = [f for f in glob.glob(os.path.join(args.app_dir, '**', '*.py'), recursive=True) if not f.endswith('__init__.py')]
    all_classes = {}
    all_functions = {}
    for py_file in py_files:
        classes, functions = ast_class_extractor.analyze_python_file(py_file)
        all_classes |= classes
        all_functions |= functions

    # Resolve inherited properties for all classes
    ast_class_extractor.resolve_inherited_properties(all_classes)

    # Generate context model
    context_model_data = real_generate_context_model(all_classes, args.workflow_folderpath)

    # Generate startup command file (unless --no-startup flag is used)
    if not args.no_startup:
        if generate_startup_command(args.workflow_folderpath, args.app_dir, args.overwrite):
            logger.info("Generated startup command file")
        else:
            logger.warning("Failed to generate startup command file")

    # Generate context folders based on the model
    context_model_path = os.path.join(commands_dir, 'context_inheritance_model.json')
    folder_generator = ContextFolderGenerator(
        commands_root=commands_dir,
        model_path=context_model_path
    )
    try:
        created_folders = folder_generator.generate_folders()
        logger.info(f"Created context folders: {', '.join(created_folders.keys())}")
    except Exception as e:
        logger.error(f"Error generating context folders: {e}")
        # Continue with command generation even if folder generation fails

    # Always generate command stubs if stub_commands is provided
    if args.stub_commands:
        try:
            generate_command_stubs_for_contexts(
                args, context_model_path, context_model_data
            )
        except Exception as e:
            logger.error(f"Error generating command stubs: {e}")
            # Continue with command generation even if stub generation fails

    # Always generate navigator stubs
    try:
        navigators_dir = os.path.join(args.workflow_folderpath, "navigators")
        navigator_generator = NavigatorStubGenerator(
            navigators_root=navigators_dir,
            model_path=context_model_path
        )

        # Generate navigators for all contexts
        generated_files = navigator_generator.generate_navigator_stubs(force=args.overwrite)

        # Log results
        logger.info(f"Generated {len(generated_files)} navigator stub files")
        for context, file_path in generated_files.items():
            logger.debug(f"  - {context}: {file_path}")
    except Exception as e:
        logger.error(f"Error generating navigator stubs: {e}")
        # Continue with command generation even if navigator generation fails

    # Generate command files
    real_generate_command_files(all_classes, commands_dir, args.app_dir, overwrite=args.overwrite, functions=all_functions)
    
    return all_classes, context_model_data


def generate_command_stubs_for_contexts(args, context_model_path, context_model_data):
    commands_dir = os.path.join(args.workflow_folderpath, "_commands")
    stub_generator = CommandStubGenerator(
        commands_root=commands_dir,
        model_path=context_model_path
    )

    # Parse command names
    command_names = [name.strip() for name in args.stub_commands.split(',')]

    # Generate stubs for each context
    contexts = list(context_model_data.get('inheritance', {}).keys())
    generated_files = {}

    for context in contexts:
        if context_files := stub_generator.generate_command_stubs_for_context(
            context, command_names, force=args.overwrite
        ):
            generated_files[context] = context_files

    # Log results
    total_files = sum(len(files) for files in generated_files.values())
    logger.info(f"Generated {total_files} command stub files across {len(generated_files)} contexts")
    for context, files in generated_files.items():
        logger.debug(f"  - {context}: {len(files)} files")

def run_validation(args, all_classes, context_model_dict):
    errors = []
    commands_dir = os.path.join(args.workflow_folderpath, '_commands')
    
    # Check if command files were generated
    if [
        f
        for f in glob.glob(
            os.path.join(commands_dir, '**', '*.py'), recursive=True
        )
        if not f.endswith('__init__.py') and not os.path.basename(f).startswith('_') and os.path.basename(f) != 'startup.py'
    ]:
        if syntax_errors := validate_python_syntax_in_dir(commands_dir):
            errors.append(f"Error: {len(syntax_errors)} syntax error(s) found in generated command files.")
        if component_errors := validate_command_file_components_in_dir(
            commands_dir
        ):
            errors.append(f"Error: {len(component_errors)} component error(s) found in generated command files.")
        # Validate command imports - commented out as it can be slow and error-prone
        # imports_ok = validate_command_imports(commands_dir)
        # if not imports_ok:
        #     errors.append("Error: Some command files could not be imported.")
    else:
        errors.append(f"Error: No command files were generated in {commands_dir}. Aborting.")
    
    # Check for context model file - ensure we're looking for context_inheritance_model.json
    context_model_path = os.path.join(commands_dir, 'context_inheritance_model.json')
    if not os.path.isfile(context_model_path):
        errors.append(f"Error: Context inheritance model JSON was not generated at {context_model_path}. Aborting.")
    else:
        commands_ok = verify_commands_against_context_model(
            context_model_dict,
            commands_dir,
            all_classes
        )
        if isinstance(commands_ok, list) and commands_ok:
            errors.extend(commands_ok)
    
    # Check for startup.py file (if not explicitly skipped)
    if not args.no_startup and not os.path.isfile(os.path.join(commands_dir, 'startup.py')):
        errors.append(f"Warning: startup.py file was not generated in {commands_dir}.")
    
    return errors

def run_documentation(args):
    from fastworkflow.build.documentation_generator import (
        collect_command_files_and_context_model,
        extract_command_metadata,
        generate_readme_content,
        write_readme_file,
    )
    # Use the _commands directory specifically
    commands_dir = os.path.join(args.workflow_folderpath, '_commands')
    command_files, context_model, doc_error = collect_command_files_and_context_model(commands_dir)
    
    if doc_error:
        logger.error(f"Documentation generation error: {doc_error}")
        return
    
    command_metadata = extract_command_metadata(command_files)
    readme_content = generate_readme_content(command_metadata, context_model, args.app_dir)
    
    # Write README.md directly to the _commands directory
    if write_readme_file(commands_dir, readme_content):
        logger.info(f"README.md generated in {commands_dir}")
    else:
        logger.error("Failed to write README.md.")

def generate_startup_command(workflow_folderpath: str, app_dir: str, overwrite: bool = False) -> bool:
    """Generate a startup command file in the _commands directory.
    
    Args:
        workflow_folderpath: Path to the workflow folder
        app_dir: Path to the application source directory
        overwrite: Whether to overwrite existing file
        
    Returns:
        bool: True if file was created or already exists, False on error
    """
    # Determine file path
    startup_path = os.path.join(workflow_folderpath, "_commands", "startup.py")

    # Check if file already exists and overwrite is False
    if os.path.exists(startup_path) and not overwrite:
        logger.debug(f"Startup file already exists at {startup_path}")
        return True

    # Get the name of the application module (last part of source_dir)
    app_module = os.path.basename(os.path.normpath(app_dir))

    # Find potential manager classes that could serve as root context
    manager_classes = []
    py_files = glob.glob(os.path.join(app_dir, "**", "*.py"), recursive=True)
    for py_file in py_files:
        if "manager" in py_file.lower():
            # This is a heuristic - files with "manager" in the name are likely to contain manager classes
            rel_path = os.path.relpath(py_file, app_dir)
            module_path = os.path.splitext(rel_path)[0].replace(os.path.sep, ".")
            manager_classes.append((module_path, os.path.basename(os.path.splitext(py_file)[0])))

    # Default manager class if none found
    manager_import = "# TODO: Replace with your application's root context class"
    manager_class = "YourRootContextClass"

    # Use the first manager class found, if any
    if manager_classes:
        module_path, module_name = manager_classes[0]
        # Try to find a class name that ends with "Manager"
        try:
            with open(os.path.join(app_dir, *module_path.split("."))) as f:
                tree = ast.parse(f.read())
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef) and "manager" in node.name.lower():
                        manager_class = node.name
                        break
        except Exception:
            # If parsing fails, use a default name based on the module
            manager_class = f"{module_name.capitalize()}Manager"

        manager_import = f"from ..{app_module}.{module_path} import {manager_class}"

    # Generate startup.py content
    startup_content = f'''import fastworkflow
from fastworkflow import CommandOutput, CommandResponse
{manager_import}

class ResponseGenerator:
    def __call__(self, workflow: fastworkflow.Workflow, command: str) -> CommandOutput:
        # Initialize your application's root context here
        # This is typically a manager class that provides access to all functionality
        filepath = (
            f'{{workflow.folderpath}}/'
            '{app_module}/'
            'data.json'  # Replace with your application's data file if needed
        )
        workflow.root_command_context = {manager_class}(filepath)
        
        response = {{
            "message": "Application initialized.",
            "context": workflow.current_command_context_name
        }}

        return CommandOutput(
            workflow_id=workflow.id,
            command_responses=[
                CommandResponse(response=str(response))
            ]
        )
'''

    # Write the file
    try:
        with open(startup_path, 'w') as f:
            f.write(startup_content)
        logger.info(f"Generated startup command file: {startup_path}")
        return True
    except Exception as e:
        logger.error(f"Error writing startup command file: {e}")
        return False

def main():  # sourcery skip: extract-method
    try:
        args = parse_args()
        fastworkflow.init(env_vars={})  # Initialize with empty environment variables
        validate_directories(args)
        all_classes_data, ctx_model_data = run_command_generation(args)
        if errors := run_validation(args, all_classes_data, ctx_model_data):
            print('\n'.join(errors))
            sys.exit(1)
        else:
            commands_dir = os.path.join(args.workflow_folderpath, "_commands")
            print(f"Successfully generated FastWorkflow commands in {commands_dir}")
        run_documentation(args)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# Add this function to be imported by cli.py
def build_main(args):  # sourcery skip: extract-method
    """Entry point for the CLI build command."""
    print("Building fastworkflow...\n")

    try:
        # Skip parsing args since they're provided by the CLI
        fastworkflow.init(env_vars={})  # Initialize with empty environment variables
        validate_directories(args)
        all_classes_data, ctx_model_data = run_command_generation(args)
        if errors := run_validation(args, all_classes_data, ctx_model_data):
            print('\n'.join(errors))
            sys.exit(1)
        else:
            commands_dir = os.path.join(args.workflow_folderpath, "_commands")
            print(f"Successfully generated FastWorkflow commands in {commands_dir}")
        run_documentation(args)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 
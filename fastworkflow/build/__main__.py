import argparse
import os
import sys
import re
import glob
import ast

from dotenv import dotenv_values

import fastworkflow
from fastworkflow.build.command_file_generator import validate_python_syntax_in_dir, validate_command_file_components_in_dir, verify_commands_against_context_model, validate_command_imports
from fastworkflow.build import ast_class_extractor
from fastworkflow.build.command_file_generator import generate_command_files as real_generate_command_files
from fastworkflow.build.class_analysis_structures import ClassInfo
from fastworkflow.build.context_model_generator import generate_context_model as real_generate_context_model
from fastworkflow.build.context_folder_generator import ContextFolderGenerator
from fastworkflow.build.command_stub_generator import CommandStubGenerator
from fastworkflow.build.navigator_stub_generator import NavigatorStubGenerator
from fastworkflow.utils.logging import logger

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate FastWorkflow command files and context model from a Python application."
    )
    parser.add_argument('--source-dir', '-s', required=True, help='Path to the source directory of the target application')
    parser.add_argument('--output-dir', '-o', required=True, help='Path to save the generated command files')
    parser.add_argument("--env_file_path", '-e', help="Path to the environment file")
    parser.add_argument("--passwords_file_path", '-p', help="Path to the passwords file")
    parser.add_argument('--dry-run', action='store_true', help='Do not write files, just print actions')
    parser.add_argument('--verbose', '-v', action='store_true', help='Print detailed logs')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite files in output directory if present')
    parser.add_argument('--context-model-dir', required=False, help='Directory to save the generated command context JSON (default: output directory)')
    parser.add_argument('--generate-stubs', action='store_true', help='Generate command stub files for contexts')
    parser.add_argument('--stub-commands', help='Comma-separated list of command names to generate stubs for')
    parser.add_argument('--generate-navigators', action='store_true', help='Generate navigator stub files for contexts')
    parser.add_argument('--navigators-dir', help='Directory to save the generated navigator files (default: "navigators" in the output directory)')
    parser.add_argument('--no-startup', action='store_true', help='Skip generating the startup.py file')
    return parser.parse_args()

def initialize_environment(args):
    env_vars = {
        **dotenv_values(args.env_file_path),
        **dotenv_values(args.passwords_file_path)
    }
    fastworkflow.init(env_vars=env_vars)

def validate_directories(args):
    if not os.path.isdir(args.source_dir):
        print(f"Error: Source directory '{args.source_dir}' does not exist or is not accessible.")
        sys.exit(1)
    if not os.access(args.source_dir, os.R_OK):
        print(f"Error: Source directory '{args.source_dir}' is not readable.")
        sys.exit(1)
    if not os.path.isdir(args.output_dir):
        print(f"Error: Output directory '{args.output_dir}' does not exist or is not accessible.")
        sys.exit(1)
    if not os.access(args.output_dir, os.W_OK):
        print(f"Error: Output directory '{args.output_dir}' is not writable.")
        sys.exit(1)
    
    # Ensure the _commands directory exists
    commands_dir = os.path.join(args.output_dir, "_commands")
    os.makedirs(commands_dir, exist_ok=True)
    
    # Create __init__.py in _commands directory if it doesn't exist
    init_path = os.path.join(commands_dir, "__init__.py")
    if not os.path.exists(init_path):
        with open(init_path, "w") as f:
            f.write("")

def run_command_generation(args):
    # Ensure output_dir and context_model_dir are Python packages
    for pkg_dir in {args.output_dir, args.context_model_dir or args.output_dir}:
        init_path = os.path.join(pkg_dir, "__init__.py")
        if not os.path.exists(init_path):
            with open(init_path, "w") as f:
                f.write("")
    # Generate command files
    py_files = [f for f in glob.glob(os.path.join(args.source_dir, '**', '*.py'), recursive=True) if not f.endswith('__init__.py')]
    all_classes = {}
    all_functions = {}
    for py_file in py_files:
        classes, functions = ast_class_extractor.analyze_python_file(py_file)
        all_classes |= classes
        all_functions |= functions

    # Resolve inherited properties for all classes
    ast_class_extractor.resolve_inherited_properties(all_classes)

    # Generate context model
    context_dir = args.context_model_dir or args.output_dir
    context_model_data = real_generate_context_model(all_classes, context_dir)

    # Generate startup command file (unless --no-startup flag is used)
    if not args.no_startup:
        if generate_startup_command(context_dir, args.source_dir, args.overwrite):
            logger.info("Generated startup command file")
        else:
            logger.warning("Failed to generate startup command file")

    # Generate context folders based on the model
    context_model_path = os.path.join(context_dir, '_commands/context_inheritance_model.json')
    folder_generator = ContextFolderGenerator(
        commands_root=os.path.join(context_dir, '_commands'),
        model_path=context_model_path
    )
    try:
        created_folders = folder_generator.generate_folders()
        logger.info(f"Created context folders: {', '.join(created_folders.keys())}")
    except Exception as e:
        logger.error(f"Error generating context folders: {e}")
        # Continue with command generation even if folder generation fails

    # Generate command stubs if requested
    if args.generate_stubs and args.stub_commands:
        try:
            generate_command_stubs_for_contexts(
                args, context_model_path, context_model_data
            )
        except Exception as e:
            logger.error(f"Error generating command stubs: {e}")
            # Continue with command generation even if stub generation fails

    # Generate handler files for contexts with container relationships
    # This is done separately to ensure handlers are generated even if no command stubs are requested
    try:
        stub_generator = CommandStubGenerator(
            commands_root=os.path.join(context_dir, '_commands'),
            model_path=context_model_path
        )

        if generated_handlers := stub_generator.generate_all_handlers_files(
            force=args.overwrite
        ):
            logger.info(f"Generated {len(generated_handlers)} _fastworkflow_handlers.py files")
            for context, file_path in generated_handlers.items():
                logger.debug(f"  - {context}: {file_path}")
    except Exception as e:
        logger.error(f"Error generating handler files: {e}")
        # Continue with command generation even if handler generation fails

    # Generate navigator stubs if requested
    if args.generate_navigators:
        try:
            navigators_dir = args.navigators_dir or os.path.join(args.output_dir, "navigators")
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
    real_generate_command_files(all_classes, os.path.join(context_dir, '_commands'), args.source_dir, overwrite=args.overwrite, functions=all_functions)

    return all_classes, context_model_data


def generate_command_stubs_for_contexts(args, context_model_path, context_model_data):
    stub_generator = CommandStubGenerator(
        commands_root=args.output_dir,
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
    commands_dir = os.path.join(args.context_model_dir or args.output_dir, '_commands')
    
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
    commands_dir = os.path.join(args.context_model_dir or args.output_dir, '_commands')
    command_files, context_model, doc_error = collect_command_files_and_context_model(commands_dir)
    
    if doc_error:
        logger.error(f"Documentation generation error: {doc_error}")
        return
    
    command_metadata = extract_command_metadata(command_files)
    readme_content = generate_readme_content(command_metadata, context_model, args.source_dir)
    
    # Write README.md directly to the _commands directory
    if write_readme_file(commands_dir, readme_content):
        logger.info(f"README.md generated in {commands_dir}")
    else:
        logger.error("Failed to write README.md.")

def generate_startup_command(output_dir: str, source_dir: str, overwrite: bool = False) -> bool:
    """Generate a startup command file in the _commands directory.
    
    Args:
        output_dir: Path to the output directory
        source_dir: Path to the source directory
        overwrite: Whether to overwrite existing file
        
    Returns:
        bool: True if file was created or already exists, False on error
    """
    # Determine file path
    startup_path = os.path.join(output_dir, "_commands", "startup.py")
    
    # Check if file already exists and overwrite is False
    if os.path.exists(startup_path) and not overwrite:
        logger.debug(f"Startup file already exists at {startup_path}")
        return True
    
    # Get the name of the application module (last part of source_dir)
    app_module = os.path.basename(os.path.normpath(source_dir))
    
    # Find potential manager classes that could serve as root context
    manager_classes = []
    py_files = glob.glob(os.path.join(source_dir, "**", "*.py"), recursive=True)
    for py_file in py_files:
        if "manager" in py_file.lower():
            # This is a heuristic - files with "manager" in the name are likely to contain manager classes
            rel_path = os.path.relpath(py_file, source_dir)
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
            with open(os.path.join(source_dir, *module_path.split("."))) as f:
                tree = ast.parse(f.read())
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef) and "manager" in node.name.lower():
                        manager_class = node.name
                        break
        except:
            # If parsing fails, use a default name based on the module
            manager_class = f"{module_name.capitalize()}Manager"
        
        manager_import = f"from ..{app_module}.{module_path} import {manager_class}"
    
    # Generate startup.py content
    startup_content = f'''import fastworkflow
from fastworkflow import CommandOutput, CommandResponse
{manager_import}

class ResponseGenerator:
    def __call__(self, session: fastworkflow.Session, command: str) -> CommandOutput:
        # Initialize your application's root context here
        # This is typically a manager class that provides access to all functionality
        filepath = (
            f'{{session.workflow_snapshot.workflow_folderpath}}/'
            '{app_module}/'
            'data.json'  # Replace with your application's data file if needed
        )
        session.root_command_context = {manager_class}(filepath)
        
        response = {{
            "message": "Application initialized.",
            "context": session.current_command_context_name
        }}

        return CommandOutput(
            session_id=session.id,
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

def main():
    args = parse_args()
    initialize_environment(args)
    validate_directories(args)
    all_classes_data, ctx_model_data = run_command_generation(args)
    if errors := run_validation(args, all_classes_data, ctx_model_data):
        print('\n'.join(errors))
        sys.exit(1)
    else:
        print(f"Successfully generated FastWorkflow commands in {args.output_dir}")
    run_documentation(args)

if __name__ == "__main__":
    main() 
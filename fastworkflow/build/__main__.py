import argparse
import os
import sys
import re
import glob

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
    for py_file in py_files:
        classes = ast_class_extractor.analyze_python_file(py_file)
        all_classes |= classes

    # Resolve inherited properties for all classes
    ast_class_extractor.resolve_inherited_properties(all_classes)

    # Generate context model
    context_dir = args.context_model_dir or args.output_dir
    context_model_data = real_generate_context_model(all_classes, context_dir)

    # Generate context folders based on the model
    context_model_path = os.path.join(context_dir, 'command_context_model.json')
    folder_generator = ContextFolderGenerator(
        commands_root=args.output_dir,
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
            commands_root=args.output_dir,
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
    real_generate_command_files(all_classes, args.output_dir, args.source_dir, overwrite=args.overwrite)

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
    if [
        f
        for f in glob.glob(
            os.path.join(args.output_dir, '**', '*.py'), recursive=True
        )
        if not f.endswith('__init__.py')
    ]:
        if syntax_errors := validate_python_syntax_in_dir(args.output_dir):
            errors.append(f"Error: {len(syntax_errors)} syntax error(s) found in generated command files.")
        if component_errors := validate_command_file_components_in_dir(
            args.output_dir
        ):
            errors.append(f"Error: {len(component_errors)} component error(s) found in generated command files.")
        # Validate command imports
        # imports_ok = validate_command_imports(args.output_dir)
        # if not imports_ok:
        #     errors.append("Error: Some command files could not be imported.")
    else:
        errors.append(f"Error: No command files were generated in {args.output_dir}. Aborting.")
    context_model_path = os.path.join(args.context_model_dir or args.output_dir, 'command_context_model.json')
    if not os.path.isfile(context_model_path):
        errors.append(f"Error: Command context model JSON was not generated at {context_model_path}. Aborting.")
    else:
        commands_ok = verify_commands_against_context_model(
            context_model_dict,
            args.output_dir,
            all_classes
        )
        if not commands_ok:
            errors.extend(commands_ok)
    return errors

def run_documentation(args):
    from fastworkflow.build.documentation_generator import (
        collect_command_files_and_context_model,
        extract_command_metadata,
        generate_readme_content,
        write_readme_file,
    )
    command_files, _, _ = collect_command_files_and_context_model(args.output_dir)
    context_model_dir = args.context_model_dir or args.output_dir
    _, context_model, doc_error = collect_command_files_and_context_model(context_model_dir)
    if doc_error:
        print(f"Documentation generation skipped: {doc_error}")
        return
    command_metadata = extract_command_metadata(command_files)
    readme_content = generate_readme_content(command_metadata, context_model, args.source_dir)
    if write_readme_file(context_model_dir, readme_content):
        print(f"README.md generated in {context_model_dir}")
    else:
        print("Error: Failed to write README.md.")

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
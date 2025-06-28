"""
Main CLI entry point for fastWorkflow.
"""

import argparse
import contextlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
import importlib.resources
from .build.__main__ import build_main
from .train.__main__ import train_main
from .run.__main__ import run_main

def find_examples_dir():
    """Finds the bundled examples directory using importlib.resources."""
    with contextlib.suppress(ModuleNotFoundError, FileNotFoundError):
        # Use files() for robust path handling with importlib.resources
        resources_path = importlib.resources.files('fastworkflow')
        examples_path = resources_path / 'examples'
        if examples_path.is_dir():
            return examples_path, True  # True indicates package examples
    # If not found in the package, look in the project root
    project_root = Path.cwd()
    examples_path = project_root / 'examples'
    return (examples_path, False) if examples_path.is_dir() else (None, False)

def list_examples(args):
    """List available bundled examples."""
    examples_dir, is_package = find_examples_dir()
    if not examples_dir:
        print("Error: Could not find the bundled 'examples' directory.", file=sys.stderr)
        sys.exit(1)

    print("Available examples:")
    for item in sorted(examples_dir.iterdir()):
        if item.is_dir() and not item.name.startswith('_'):
            print(f"- {item.name}")

def fetch_example(args):
    """Fetch a bundled example and copy it to the local filesystem."""
    examples_dir, is_package = find_examples_dir()
    if not examples_dir:
        print("Error: Could not find the bundled 'examples' directory.", file=sys.stderr)
        sys.exit(1)

    source_path = examples_dir / args.name
    if not source_path.is_dir():
        print(f"Error: Example '{args.name}' not found.", file=sys.stderr)
        print("Use 'fastworkflow examples list' to see available examples.")
        sys.exit(1)

    # If examples are only found locally (not in package), skip the fetch operation
    if not is_package:
        print(f"Note: Example '{args.name}' is already in the local examples directory.")
        return source_path

    target_root = Path("./examples")
    target_path = target_root / args.name

    if target_path.exists() and not getattr(args, 'force', False):
        # Ask user for confirmation before overwriting
        response = input(f"Target directory '{target_path}' already exists. Overwrite? [y/N] ")
        if response.lower() != 'y':
            print("Operation cancelled.")
            sys.exit(0)
        
    target_root.mkdir(exist_ok=True)

    # Ignore generated files during copy
    ignore_patterns = shutil.ignore_patterns('___command_info', '__pycache__', '*.pyc')

    try:
        shutil.copytree(source_path, target_path, ignore=ignore_patterns, dirs_exist_ok=True)
        print(f"✅ Example '{args.name}' copied to '{target_path}'")
        return target_path
    except Exception as e:
        print(f"Error copying example: {e}", file=sys.stderr)
        sys.exit(1)

def train_example(args):
    """Fetch and then train an example."""
    print(f"Fetching example '{args.name}'...")

    # Check if example already exists and handle accordingly
    examples_dir, is_package = find_examples_dir()
    if not examples_dir:
        print("Error: Could not find the bundled 'examples' directory.", file=sys.stderr)
        sys.exit(1)

    source_path = examples_dir / args.name
    if not source_path.is_dir():
        print(f"Error: Example '{args.name}' not found.", file=sys.stderr)
        print("Use 'fastworkflow examples list' to see available examples.")
        sys.exit(1)

    # If examples are only found locally, use the local path directly
    if not is_package:
        target_path = source_path
        print("Note: Using example from local examples directory.")
    else:
        # Otherwise, fetch the example from the package
        target_path = fetch_example(args)

    print(f"\nTraining example in '{target_path}'...")

    # Get the appropriate env files for this example workflow
    env_file_path, passwords_file_path = find_default_env_files(target_path)

    # Check if the files exist
    env_file = Path(env_file_path)
    passwords_file = Path(passwords_file_path)

    if not env_file.exists() or not passwords_file.exists():
        print("Warning: Default env files not found at expected paths:")
        print(f"  - {env_file}")
        print(f"  - {passwords_file}")
        print("Using empty files to proceed. Please ensure API keys are set if needed.")

        # Create empty files if they don't exist
        env_file = Path(".env")
        passwords_file = Path("passwords.env")
        env_file.touch()
        passwords_file.touch()

    # The `train` script needs the path to the workflow, and the env files
    cmd = [
        sys.executable,
        "-m", "fastworkflow.train",
        str(target_path),
        str(env_file),
        str(passwords_file)
    ]

    try:
        # We run the command from the current working directory
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')

        # Stream the output
        if process.stdout:
            for line in iter(process.stdout.readline, ''):
                print(line, end='')

        process.wait()

        if process.returncode == 0:
            print(f"\n✅ Successfully trained example '{args.name}'.")
            print(f"You can now run it with:\npython -m fastworkflow.run {target_path}")
        else:
            print(f"\n❌ Training failed with exit code {process.returncode}.", file=sys.stderr)
            sys.exit(1)

    except FileNotFoundError:
        print("Error: 'python' executable not found. Make sure your environment is set up correctly.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during training: {e}", file=sys.stderr)
        sys.exit(1)

def find_default_env_files(workflow_path=None):
    """Find the appropriate default env files based on context.
    
    If the workflow path is within the examples directory, use the bundled examples env files.
    Otherwise, use local .env files in the current directory.
    
    Args:
        workflow_path: Optional path to the workflow directory
        
    Returns:
        tuple: (env_file_path, passwords_file_path)
    """
    # Default to local env files for user workflows
    default_env = ".env"
    default_passwords = "passwords.env"
    
    # If workflow_path is provided and seems to be an example workflow
    if workflow_path:
        workflow_path = Path(workflow_path)
        examples_dir, _ = find_examples_dir()
        
        if examples_dir and (
            str(workflow_path).startswith(str(examples_dir)) or
            "/examples/" in str(workflow_path) or
            "\\examples\\" in str(workflow_path)
        ):
            # This appears to be an example workflow, use bundled env files
            return "fastworkflow/examples/fastworkflow.env", "fastworkflow/examples/passwords.env"
    
    # For user workflows, use local env files
    return default_env, default_passwords

def add_build_parser(subparsers):
    """Add subparser for the 'build' command."""
    parser_build = subparsers.add_parser("build", help="Generate FastWorkflow command files and context model from a Python application.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_build.add_argument('--source-dir', '-s', required=True, help='Path to the source directory of the target application')
    parser_build.add_argument('--output-dir', '-o', required=True, help='Path to save the generated command files')
    parser_build.add_argument("--env_file_path", '-e', required=False, default=".env", 
                             help="Path to the environment file (default: .env in current directory)")
    parser_build.add_argument("--passwords_file_path", '-p', required=False, default="passwords.env", 
                             help="Path to the passwords file (default: passwords.env in current directory)")
    parser_build.add_argument('--dry-run', action='store_true', help='Do not write files, just print actions')
    parser_build.add_argument('--verbose', '-v', action='store_true', help='Print detailed logs')
    parser_build.add_argument('--overwrite', action='store_true', help='Overwrite files in output directory if present')
    parser_build.add_argument('--context-model-dir', required=True, help='Directory to save the generated command context JSON')
    parser_build.add_argument('--generate-stubs', action='store_true', help='Generate command stub files for contexts')
    parser_build.add_argument('--stub-commands', help='Comma-separated list of command names to generate stubs for')
    parser_build.add_argument('--generate-navigators', action='store_true', help='Generate navigator stub files for contexts')
    parser_build.add_argument('--navigators-dir', help='Directory to save the generated navigator files (default: "navigators" in the output directory)')
    parser_build.add_argument('--no-startup', action='store_true', help='Skip generating the startup.py file')
    parser_build.set_defaults(func=build_main)

def add_train_parser(subparsers):
    """Add subparser for the 'train' command."""
    parser_train = subparsers.add_parser("train", help="Train the intent detection pipeline for a workflow.")
    parser_train.add_argument("workflow_folderpath", help="Path to the workflow folder")
    
    # Default env files will be determined at runtime based on the workflow path
    parser_train.add_argument("env_file_path", nargs='?', default=None,
                             help="Path to the environment file (default: .env in current directory, or bundled env file for examples)")
    parser_train.add_argument("passwords_file_path", nargs='?', default=None,
                             help="Path to the passwords file (default: passwords.env in current directory, or bundled env file for examples)")
    parser_train.set_defaults(func=lambda args: train_with_defaults(args))

def add_run_parser(subparsers):
    """Add subparser for the 'run' command."""
    parser_run = subparsers.add_parser("run", help="Run a workflow's interactive assistant.")
    parser_run.add_argument("workflow_path", help="Path to the workflow folder")
    
    # Default env files will be determined at runtime based on the workflow path
    parser_run.add_argument("env_file_path", nargs='?', default=None,
                           help="Path to the environment file (default: .env in current directory, or bundled env file for examples)")
    parser_run.add_argument("passwords_file_path", nargs='?', default=None,
                           help="Path to the passwords file (default: passwords.env in current directory, or bundled env file for examples)")
    parser_run.add_argument("--context_file_path", help="Optional context file path", default="")
    parser_run.add_argument("--startup_command", help="Optional startup command", default="")
    parser_run.add_argument("--startup_action", help="Optional startup action", default="")
    parser_run.add_argument("--keep_alive", help="Optional keep_alive", default=True)
    parser_run.set_defaults(func=lambda args: run_with_defaults(args))

def train_with_defaults(args):
    """Wrapper for train_main that sets default env file paths based on context."""
    if args.env_file_path is None or args.passwords_file_path is None:
        default_env, default_passwords = find_default_env_files(args.workflow_folderpath)
    if args.env_file_path is None:
        args.env_file_path = default_env
    if args.passwords_file_path is None:
        args.passwords_file_path = default_passwords

    return train_main(args)

def run_with_defaults(args):
    """Wrapper for run_main that sets default env file paths based on context."""
    if args.env_file_path is None or args.passwords_file_path is None:
        default_env, default_passwords = find_default_env_files(args.workflow_path)
    if args.env_file_path is None:
        args.env_file_path = default_env
    if args.passwords_file_path is None:
        args.passwords_file_path = default_passwords

    return run_main(args)

def main():
    """Main function for the fastworkflow CLI."""
    parser = argparse.ArgumentParser(
        description="fastWorkflow CLI tool for building, training, and running workflows."
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Sub-command help")

    # 'examples' command group
    parser_examples = subparsers.add_parser("examples", help="Manage bundled examples")
    examples_subparsers = parser_examples.add_subparsers(dest="action", required=True)

    # 'examples list' command
    parser_list = examples_subparsers.add_parser("list", help="List available examples")
    parser_list.set_defaults(func=list_examples)

    # 'examples fetch' command
    parser_fetch = examples_subparsers.add_parser("fetch", help="Fetch a specific example")
    parser_fetch.add_argument("name", help="The name of the example to fetch")
    parser_fetch.add_argument("--force", action="store_true", help="Force overwrite if example already exists")
    parser_fetch.set_defaults(func=fetch_example)
    
    # 'examples train' command
    parser_train_example = examples_subparsers.add_parser("train", help="Fetch and train a specific example")
    parser_train_example.add_argument("name", help="The name of the example to fetch and train")
    parser_train_example.add_argument("--force", action="store_true", help="Force overwrite if example already exists")
    parser_train_example.set_defaults(func=train_example)
    
    # Add top-level commands
    add_build_parser(subparsers)
    add_train_parser(subparsers)
    add_run_parser(subparsers)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main() 
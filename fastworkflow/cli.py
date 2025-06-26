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
        # Use files() context manager for robust path handling with importlib.resources
        with importlib.resources.files('fastworkflow') as resources_path:
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

    # Check for env files in the CWD, not the target path
    env_file = Path(".env")
    passwords_file = Path("passwords.env")

    if not env_file.exists() or not passwords_file.exists():
        print("Warning: '.env' or 'passwords.env' not found in the current directory.")
        print("Creating empty files to proceed. Please populate them if your workflow requires API keys.")
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
            print(f"You can now run it with:\npython -m fastworkflow.run {target_path} .env passwords.env --startup_command startup")
        else:
            print(f"\n❌ Training failed with exit code {process.returncode}.", file=sys.stderr)
            sys.exit(1)

    except FileNotFoundError:
        print("Error: 'python' executable not found. Make sure your environment is set up correctly.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during training: {e}", file=sys.stderr)
        sys.exit(1)

def add_build_parser(subparsers):
    """Add subparser for the 'build' command."""
    parser_build = subparsers.add_parser("build", help="Generate FastWorkflow command files and context model from a Python application.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_build.add_argument('--source-dir', '-s', required=True, help='Path to the source directory of the target application')
    parser_build.add_argument('--output-dir', '-o', required=True, help='Path to save the generated command files')
    parser_build.add_argument("--env_file_path", '-e', required=True, help="Path to the environment file")
    parser_build.add_argument("--passwords_file_path", '-p', required=True, help="Path to the passwords file")
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
    parser_train.add_argument("env_file_path", help="Path to the environment file")
    parser_train.add_argument("passwords_file_path", help="Path to the passwords file")
    parser_train.set_defaults(func=train_main)

def add_run_parser(subparsers):
    """Add subparser for the 'run' command."""
    parser_run = subparsers.add_parser("run", help="Run a workflow's interactive assistant.")
    parser_run.add_argument("workflow_path", help="Path to the workflow folder")
    parser_run.add_argument("env_file_path", help="Path to the environment file")
    parser_run.add_argument("passwords_file_path", help="Path to the passwords file")
    parser_run.add_argument("--context_file_path", help="Optional context file path", default="")
    parser_run.add_argument("--startup_command", help="Optional startup command", default="")
    parser_run.add_argument("--startup_action", help="Optional startup action", default="")
    parser_run.add_argument("--keep_alive", help="Optional keep_alive", default=True)
    parser_run.set_defaults(func=run_main)

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
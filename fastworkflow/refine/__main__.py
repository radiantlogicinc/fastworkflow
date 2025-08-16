import argparse
import os
import sys

import fastworkflow
from fastworkflow.utils.logging import logger
from fastworkflow.build.genai_postprocessor import run_genai_postprocessor
from fastworkflow.utils.command_dependency_graph import generate_dependency_graph


def parse_args():
    parser = argparse.ArgumentParser(
        description="Refine a FastWorkflow by enhancing command metadata and generating dependency graph."
    )
    parser.add_argument('--workflow-folderpath', '-w', required=True, help='Path to the workflow folder to refine')
    return parser.parse_args()


def _validate_workflow_folder(workflow_folderpath: str) -> None:
    if not os.path.isdir(workflow_folderpath):
        print(f"Error: Workflow directory '{workflow_folderpath}' does not exist or is not accessible.")
        sys.exit(1)
    commands_dir = os.path.join(workflow_folderpath, "_commands")
    if not os.path.isdir(commands_dir):
        print(f"Error: Commands directory not found at '{commands_dir}'. Please run 'fastworkflow build' first.")
        sys.exit(1)


def _prompt_for_action(non_interactive_choice: int | None = None) -> int:
    """Prompt user to choose refine action. Returns 1, 2, or 3. Default is 1."""
    # If a non-interactive choice is provided (e.g., tests), validate and use it
    if non_interactive_choice in {1, 2, 3}:
        return int(non_interactive_choice)
    print("Select refine action:")
    print("  1) Generate dependency graph (default)")
    print("  2) Refine command metadata")
    print("  3) Do both")
    while True:
        choice = input("Enter choice [1/2/3]: ").strip()
        if choice == "":
            return 1
        if choice in {"1", "2", "3"}:
            return int(choice)
        print("Invalid choice. Please enter 1, 2, or 3.")


def refine_main(args):
    """Entry point for the CLI refine command (invoked from fastworkflow.cli)."""
    try:
        # Initialize environment
        fastworkflow.init(env_vars={})

        workflow_folderpath = args.workflow_folderpath
        _validate_workflow_folder(workflow_folderpath)

        # Ask user which actions to run (support non-interactive via env var for tests/automation)
        env_choice = os.environ.get("FASTWORKFLOW_REFINE_CHOICE")
        try:
            env_choice_val = int(env_choice) if env_choice is not None else None
        except ValueError:
            env_choice_val = None
        action = _prompt_for_action(non_interactive_choice=env_choice_val)

        # Prepare args-like object for post-processor when needed
        class _Dummy:
            pass
        _dummy = _Dummy()
        _dummy.workflow_folderpath = workflow_folderpath

        # Execute selected actions
        if action in (2, 3):
            logger.info("Running GenAI metadata refinement with LibCST...")
            run_genai_postprocessor(_dummy, classes={}, functions={})

        if action in (1, 3):
            logger.info("Generating parameter dependency graph...")
            graph_path = generate_dependency_graph(workflow_folderpath)
            print(f"Generated parameter dependency graph at {graph_path}")

        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    args = parse_args()
    code = refine_main(args)
    sys.exit(code)


if __name__ == "__main__":
    main()

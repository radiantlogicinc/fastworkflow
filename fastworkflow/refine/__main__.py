import argparse
import os
import sys

import fastworkflow
from fastworkflow.utils.logging import logger
from fastworkflow.build.genai_postprocessor import run_genai_postprocessor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Refine a FastWorkflow by enhancing command metadata."
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


def refine_main(args):
    """Entry point for the CLI refine command (invoked from fastworkflow.cli)."""
    try:
        # Initialize environment
        fastworkflow.init(env_vars={})

        workflow_folderpath = args.workflow_folderpath
        _validate_workflow_folder(workflow_folderpath)

        # Prepare args-like object for post-processor when needed
        class _Dummy:
            pass
        _dummy = _Dummy()
        _dummy.workflow_folderpath = workflow_folderpath
        logger.info("Running GenAI metadata refinement with LibCST...")
        run_genai_postprocessor(_dummy, classes={}, functions={})

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

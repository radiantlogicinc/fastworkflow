import argparse
import os

from colorama import Fore, Style

from fastworkflow.start_workflow import start_workflow

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Assistant for workflow processing")
    parser.add_argument("workflow_path", help="Path to the workflow folder")
    parser.add_argument(
        "--startup_command", help="Optional startup command", default=""
    )
    args = parser.parse_args()

    if not os.path.isdir(args.workflow_path):
        print(
            f"{Fore.RED}Error: The specified workflow path '{args.workflow_path}' is not a valid directory.{Style.RESET_ALL}"
        )
        exit(1)

    print(
        f"{Fore.GREEN}{Style.BRIGHT}AI>{Style.RESET_ALL}{Fore.GREEN} AI Assistant is running with workflow: {args.workflow_path}{Style.RESET_ALL}"
    )
    print(
        f"{Fore.GREEN}{Style.BRIGHT}AI>{Style.RESET_ALL}{Fore.GREEN} Type 'exit' to quit.{Style.RESET_ALL}"
    )

    command_output = start_workflow(
        args.workflow_path, startup_command=args.startup_command
    )

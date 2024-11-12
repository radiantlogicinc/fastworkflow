import argparse
import json
import os
import random
from dotenv import dotenv_values

from colorama import Fore, Style

from fastworkflow.session import Session
from fastworkflow.start_workflow import start_workflow


parser = argparse.ArgumentParser(description="AI Assistant for workflow processing")
parser.add_argument("workflow_path", help="Path to the workflow folder")
parser.add_argument("env_file_path", help="Path to the environment file")
parser.add_argument(
    "--context_file_path", help="Path to the context file", default=""
)
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

context = {}
if args.context_file_path:
    with open(args.context_file_path, "r") as f:
        context = json.load(f)

session = Session(
    random.randint(1, 100000000), 
    args.workflow_path, 
    env_vars={**dotenv_values(args.env_file_path)}, 
    context=context
)

command_output = start_workflow(
    session, startup_command=args.startup_command
)

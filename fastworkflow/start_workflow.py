import argparse
import os
import random
from typing import Optional

from colorama import Fore, Style, init
from semantic_router import RouteLayer

from fastworkflow.command_executor import CommandResponse
from fastworkflow.command_router import CommandRouter

from fastworkflow.session import Session


def start_workflow(
    session: Session,
    startup_command: str = "",
    caller_session: Optional[Session] = None,
    keep_alive=True,
) -> list[CommandResponse]:
    # Initialize colorama
    init(autoreset=True)

    if caller_session:
        session.caller_session = caller_session

    command_router = CommandRouter(session)

    if startup_command:
        command_output: list[CommandResponse] = command_router.route_command(startup_command)
        for command_response in command_output:
            if not command_response.response:
                continue
            print(
                f"{Fore.GREEN}{Style.BRIGHT}{session.root_workitem_type.upper()} AI>{Style.RESET_ALL}{Fore.GREEN} {command_response.response}{Style.RESET_ALL}"
            )
            for artifact_name, artifact_value in command_response.artifacts.items():
                if artifact_name == "abort_command":
                    continue
                print(
                    f"{Fore.CYAN}{Style.NORMAL}Artifact: {artifact_name}={artifact_value}{Style.RESET_ALL}"
                )
            for recommendation in command_response.recommendations:
                print(
                    f"{Fore.MAGENTA}{Style.NORMAL}Recommendation: {recommendation}{Style.RESET_ALL}"
                )


    while not session.workflow.is_complete or keep_alive:
        user_command = input(
            f"{Fore.YELLOW}{Style.BRIGHT}User>{Style.RESET_ALL}{Fore.YELLOW} "
        )
        command_output: list[CommandResponse] = command_router.route_command(user_command)
        if session.workflow.is_complete and not keep_alive:
            break

        abort_command = (
            command_output[0].artifacts.get("abort_command", False)
            if command_output[0].artifacts
            else False
        )
        if abort_command:
            break

        for command_response in command_output:
            if not command_response.response:
                continue
            print(
                f"{Fore.GREEN}{Style.BRIGHT}{session.root_workitem_type.upper()} AI>{Style.RESET_ALL}{Fore.GREEN} {command_response.response}{Style.RESET_ALL}"
            )
            for artifact_name, artifact_value in command_response.artifacts.items():
                if artifact_name == "abort_command":
                    continue
                print(
                    f"{Fore.CYAN}{Style.NORMAL}Artifact: {artifact_name}={artifact_value}{Style.RESET_ALL}"
                )
            for recommendation in command_response.recommendations:
                print(
                    f"{Fore.MAGENTA}{Style.NORMAL}Recommendation: {recommendation}{Style.RESET_ALL}"
                )

    if not keep_alive:
        session.close_session()

    return command_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Assistant for workflow processing")
    parser.add_argument("workflow_path", help="Path to the workflow folder")
    parser.add_argument("env_file_path", help="Path to the environment file")
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

    session = Session(random.randint(1, 100000000), args.workflow_path, args.env_file_path)

    command_output = start_workflow(
        session,
        startup_command=args.startup_command,
    )

import argparse
import os
import random
from typing import Optional

from colorama import Fore, Style, init
from semantic_router import RouteLayer
from semantic_router.encoders import HuggingFaceEncoder

from fastworkflow.command_executor import CommandOutput
from fastworkflow.command_router import CommandRouter
from fastworkflow.semantic_router_definition import SemanticRouterDefinition
from fastworkflow.session import Session


def start_workflow(
    workflow_path: str,
    startup_command: str = "",
    payload: Optional[dict] = None,
    keep_alive=True,
) -> CommandOutput:
    # Initialize colorama
    init(autoreset=True)

    session_id = random.randint(1, 100000000)

    session = Session(session_id, workflow_path)
    encoder = HuggingFaceEncoder()
    command_router = CommandRouter(session)

    semantic_router = SemanticRouterDefinition(session, encoder)

    map_workitem_type_2_route_layer: dict[str, RouteLayer] = {}
    for workitem_type in session.workflow_definition.types:
        route_layer = semantic_router.get_route_layer(workitem_type)
        map_workitem_type_2_route_layer[workitem_type] = route_layer

    if startup_command:
        command_output: CommandOutput = command_router.route_command(
            map_workitem_type_2_route_layer, startup_command, payload
        )
        print(
            f"{Fore.GREEN}{Style.BRIGHT}{session.root_workitem_type.upper()} AI>{Style.RESET_ALL}{Fore.GREEN} {command_output.response}{Style.RESET_ALL}"
        )

    while not session.workflow.is_complete or keep_alive:
        user_command = input(
            f"{Fore.YELLOW}{Style.BRIGHT}User>{Style.RESET_ALL}{Fore.YELLOW} "
        )
        command_output: CommandOutput = command_router.route_command(
            map_workitem_type_2_route_layer, user_command
        )

        abort_command = (
            command_output.payload.get("abort_command", False)
            if command_output.payload
            else False
        )
        if abort_command:
            break

        if command_output.response:
            print(
                f"{Fore.GREEN}{Style.BRIGHT}{session.root_workitem_type.upper()} AI>{Style.RESET_ALL}{Fore.GREEN} {command_output.response}{Style.RESET_ALL}"
            )

    return command_output


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

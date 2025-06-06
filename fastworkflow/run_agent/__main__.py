import argparse
import contextlib
import json
import os
import queue
from typing import Optional
from dotenv import dotenv_values
from colorama import Fore, Style, init

import fastworkflow
from fastworkflow.command_executor import CommandExecutor
from .agent_module import initialize_dspy_agent

# Initialize colorama
init(autoreset=True)

def print_command_output(command_output):
    for command_response in command_output.command_responses:
        session_id = "UnknownSession"
        with contextlib.suppress(Exception):
            session_id = fastworkflow.WorkflowSession.get_active_session_id()

        if command_response.response:
            print(
                f"{Fore.GREEN}{Style.BRIGHT}{session_id} AI>{Style.RESET_ALL}{Fore.GREEN} {command_response.response}{Style.RESET_ALL}"
            )

        for artifact_name, artifact_value in command_response.artifacts.items():
            print(
                f"{Fore.CYAN}{Style.BRIGHT}{session_id} AI>{Style.RESET_ALL}{Fore.CYAN} Artifact: {artifact_name}={artifact_value}{Style.RESET_ALL}"
            )
        for action in command_response.next_actions:
            print(
                f"{Fore.BLUE}{Style.BRIGHT}{session_id} AI>{Style.RESET_ALL}{Fore.BLUE} Next Action: {action}{Style.RESET_ALL}"
            )
        for recommendation in command_response.recommendations:
            print(
                f"{Fore.MAGENTA}{Style.BRIGHT}{session_id} AI>{Style.RESET_ALL}{Fore.MAGENTA} Recommendation: {recommendation}{Style.RESET_ALL}"
            )

def main():
    parser = argparse.ArgumentParser(description="AI Assistant for workflow processing")
    parser.add_argument("workflow_path", help="Path to the workflow folder")
    parser.add_argument("env_file_path", help="Path to the environment file")
    parser.add_argument("passwords_file_path", help="Path to the passwords file")
    parser.add_argument(
        "--context_file_path", help="Optional context file path", default=""
    )
    parser.add_argument(
        "--startup_command", help="Optional startup command", default=""
    )
    parser.add_argument(
        "--startup_action", help="Optional startup action", default=""
    )
    parser.add_argument(
        "--keep_alive", help="Optional keep_alive", default=True
    )

    args = parser.parse_args()

    if not os.path.isdir(args.workflow_path):
        print(
            f"{Fore.RED}Error: The specified workflow path '{args.workflow_path}' is not a valid directory.{Style.RESET_ALL}"
        )
        exit(1)

    print(
        f"{Fore.GREEN}{Style.BRIGHT}AI>{Style.RESET_ALL}{Fore.GREEN} Running fastWorkflow: {args.workflow_path}{Style.RESET_ALL}"
    )
    print(
        f"{Fore.GREEN}{Style.BRIGHT}AI>{Style.RESET_ALL}{Fore.GREEN} Type 'exit' to quit the application.{Style.RESET_ALL}"
    )

    if args.startup_command and args.startup_action:
        raise ValueError("Cannot provide both startup_command and startup_action")

    env_vars = {
        **dotenv_values(args.env_file_path),
        **dotenv_values(args.passwords_file_path)
    }
    fastworkflow.init(env_vars=env_vars)

    LLM_AGENT = fastworkflow.get_env_var("LLM_AGENT")
    if not LLM_AGENT:
        print(f"{Fore.RED}Error: DSPy Language Model not provided. Set LLM_AGENT environment variable.{Style.RESET_ALL}")
        exit(1)

    # this could be None
    LITELLM_API_KEY_AGENT = fastworkflow.get_env_var("LITELLM_API_KEY_AGENT")

    startup_action: Optional[fastworkflow.Action] = None
    if args.startup_action:
        with open(args.startup_action, 'r') as file:
            startup_action_dict = json.load(file)
        startup_action = fastworkflow.Action(**startup_action_dict)

    context_dict = {}
    if args.context_file_path:
        with open(args.context_file_path, 'r') as file:
            context_dict = json.load(file)

    workflow_session = fastworkflow.WorkflowSession(
        CommandExecutor(),
        args.workflow_path,
        session_id_str=f"run_{args.workflow_path}",
        context=context_dict,
        startup_command=args.startup_command, 
        startup_action=startup_action, 
        keep_alive=args.keep_alive
    )

    try:
        react_agent = initialize_dspy_agent(
            workflow_session, 
            LLM_AGENT, 
            LITELLM_API_KEY_AGENT,
            clear_cache=True
        )
    except (EnvironmentError, RuntimeError) as e:
        print(f"{Fore.RED}Failed to initialize DSPy agent: {e}{Style.RESET_ALL}")
        exit(1)

    workflow_session.start()
    with contextlib.suppress(queue.Empty):
        if command_output := workflow_session.command_output_queue.get(
            timeout=0.1
        ):
            print(f"{Fore.WHITE}{Style.DIM}--- Startup Command Output ---{Style.RESET_ALL}")
            print_command_output(command_output)
            print(f"{Fore.WHITE}{Style.DIM}--- End Startup Command Output ---{Style.RESET_ALL}")
    
    while True: 
        if not args.keep_alive and workflow_session.workflow_is_complete:
            print(f"{Fore.BLUE}Workflow complete and keep_alive is false. Exiting...{Style.RESET_ALL}")
            break

        user_input_str = input(
            f"{Fore.YELLOW}{Style.BRIGHT}User>{Style.RESET_ALL}{Fore.YELLOW} "
        )
        if user_input_str.lower() == "exit":
            print(f"{Fore.BLUE}User requested exit. Exiting...{Style.RESET_ALL}")
            break

        try:
            agent_response = react_agent(user_query=user_input_str)
            print(f"{Fore.GREEN}{Style.BRIGHT}Agent>{Style.RESET_ALL}{Fore.GREEN} {agent_response.final_answer}{Style.RESET_ALL}")
        except Exception as e: # pylint: disable=broad-except
            print(f"{Fore.RED}{Style.BRIGHT}Agent Error>{Style.RESET_ALL}{Fore.RED} An error occurred during agent processing: {e}{Style.RESET_ALL}")

if __name__ == "__main__":
    main()

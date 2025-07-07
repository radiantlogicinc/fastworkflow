import argparse
import contextlib
import json
import os
import queue
from typing import Optional
from dotenv import dotenv_values

# Third-party CLI prettification libraries
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.console import Group

# Progress bar helper (light import)
from fastworkflow.utils.startup_progress import StartupProgress

# NOTE: heavy fastworkflow / DSPy imports moved into main()

# Instantiate a global console for consistent styling
console = Console()

def _build_artifact_table(artifacts: dict[str, str]) -> Table:
    """Return a rich.Table representation for artifact key-value pairs."""
    table = Table(show_header=True, header_style="bold cyan", box=None)
    table.add_column("Name", style="cyan", overflow="fold")
    table.add_column("Value", style="white", overflow="fold")
    for name, value in artifacts.items():
        table.add_row(str(name), str(value))
    return table

def print_command_output(command_output):
    """Pretty-print workflow output using rich panels and tables."""
    for command_response in command_output.command_responses:
        workflow_id = "UnknownSession"
        with contextlib.suppress(Exception):
            workflow_id = fastworkflow.ChatSession.get_active_workflow_id()

        # Collect body elements for the panel content
        body_renderables = []

        if command_response.response:
            body_renderables.append(Text(command_response.response, style="green"))

        if command_response.artifacts:
            body_renderables.extend(
                (
                    Text("Artifacts", style="bold cyan"),
                    _build_artifact_table(command_response.artifacts),
                )
            )
        if command_response.next_actions:
            actions_table = Table(show_header=False, box=None)
            for act in command_response.next_actions:
                actions_table.add_row(Text(str(act), style="blue"))
            body_renderables.extend(
                (Text("Next Actions", style="bold blue"), actions_table)
            )
        if command_response.recommendations:
            rec_table = Table(show_header=False, box=None)
            for rec in command_response.recommendations:
                rec_table.add_row(Text(str(rec), style="magenta"))
            body_renderables.extend(
                (Text("Recommendations", style="bold magenta"), rec_table)
            )

        panel_title = f"[bold yellow]Workflow {workflow_id}[/bold yellow]"
        # Group all renderables together
        group = Group(*body_renderables)
        # Use the group in the panel
        panel = Panel.fit(group, title=panel_title, border_style="green")
        console.print(panel)

def check_workflow_trained(workflow_path: str) -> bool:
    """
    Check if a workflow has been trained by looking for the tiny_ambiguous_threshold.json file
    in the ___command_info/global folder.
    
    Args:
        workflow_path: Path to the workflow folder
    
    Returns:
        bool: True if the workflow appears to be trained, False otherwise
    """
    # Path to the global command info directory
    global_cmd_info_path = os.path.join(workflow_path, "___command_info", "global")
    
    # Path to the tiny_ambiguous_threshold.json file
    threshold_file_path = os.path.join(global_cmd_info_path, "tiny_ambiguous_threshold.json")
    
    # Check if the file exists
    return os.path.exists(threshold_file_path)

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
        console.print(f"[bold red]Error:[/bold red] The specified workflow path '{args.workflow_path}' is not a valid directory.")
        exit(1)

    # ------------------------------------------------------------------
    # Startup progress bar – start early
    # ------------------------------------------------------------------
    StartupProgress.begin(total=4)  # coarse steps; will add more later

    console.print(Panel(f"Running fastWorkflow: [bold]{args.workflow_path}[/bold]", title="[bold green]fastworkflow[/bold green]", border_style="green"))
    console.print("[bold green]Tip:[/bold green] Type 'exit' to quit the application.")

    # Heavy imports AFTER progress bar
    global fastworkflow
    import fastworkflow  # noqa: F401
    from fastworkflow.command_executor import CommandExecutor  # noqa: F401

    StartupProgress.advance("Imported fastworkflow modules")

    env_vars = {
        **dotenv_values(args.env_file_path),
        **dotenv_values(args.passwords_file_path)
    }

    fastworkflow.init(env_vars=env_vars)
    StartupProgress.advance("fastworkflow.init complete")

    LLM_AGENT = fastworkflow.get_env_var("LLM_AGENT")
    if not LLM_AGENT:
        console.print("[bold red]Error:[/bold red] DSPy Language Model not provided. Set LLM_AGENT environment variable.")
        exit(1)

    # Check if the workflow has been trained
    if not check_workflow_trained(args.workflow_path):
        # Extract workflow name for the error message
        workflow_name = os.path.basename(args.workflow_path)
        console.print(Panel(
            f"To train this workflow, run:\n"
            f"[bold white]fastworkflow train {args.workflow_path}[/bold white]",
            title="[bold red]Workflow '{workflow_name}' has not been trained[/bold red]", 
            border_style="red"
        ))
        exit(1)

    # this could be None
    LITELLM_API_KEY_AGENT = fastworkflow.get_env_var("LITELLM_API_KEY_AGENT")

    startup_action: Optional[fastworkflow.Action] = None
    if args.startup_action:
        with open(args.startup_action, 'r') as file:
            startup_action_dict = json.load(file)
        startup_action = fastworkflow.Action(**startup_action_dict)

    context_dict = None
    if args.context_file_path:
        with open(args.context_file_path, 'r') as file:
            context_dict = json.load(file)

    chat_session = fastworkflow.ChatSession(
        args.workflow_path,
        workflow_context=context_dict,
        startup_command=args.startup_command, 
        startup_action=startup_action, 
        keep_alive=args.keep_alive
    )

    StartupProgress.advance("ChatSession ready")

    # Import DSPy-related agent lazily (may be heavy)
    from .agent_module import initialize_dspy_agent

    StartupProgress.add_total(1)
    StartupProgress.advance("Initializing DSPy agent")

    try:
        react_agent = initialize_dspy_agent(
            chat_session, 
            LLM_AGENT, 
            LITELLM_API_KEY_AGENT,
            clear_cache=True
        )
    except (EnvironmentError, RuntimeError) as e:
        console.print(f"[bold red]Failed to initialize DSPy agent:[/bold red] {e}")
        StartupProgress.end()
        exit(1)

    StartupProgress.end()

    chat_session.start()
    with contextlib.suppress(queue.Empty):
        if command_output := chat_session.command_output_queue.get(
            timeout=0.1
        ):
            console.print(Panel("Startup Command Output", border_style="dim"))
            print_command_output(command_output)
            console.print(Panel("End Startup Command Output", border_style="dim"))
    
    while True: 
        if not args.keep_alive and chat_session.workflow_is_complete:
            console.print("[blue]Workflow complete and keep_alive is false. Exiting...[/blue]")
            break

        user_input_str = console.input("[bold yellow]User > [/bold yellow]")
        if user_input_str.lower() == "exit":
            console.print("[blue]User requested exit. Exiting...[/blue]")
            break

        try:
            agent_response = react_agent(user_query=user_input_str)
            console.print(Panel(agent_response.final_answer, title="[bold green]Agent Response[/bold green]", border_style="green"))
        except Exception as e: # pylint: disable=broad-except
            console.print(f"[bold red]Agent Error:[/bold red] An error occurred during agent processing: {e}")

if __name__ == "__main__":
    main()

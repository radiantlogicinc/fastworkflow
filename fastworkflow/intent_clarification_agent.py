"""
Intent detection error state agent module for fastWorkflow.
Specialized agent for handling intent detection errors.
"""

import json
import dspy

import fastworkflow
from fastworkflow.utils.react import fastWorkflowReAct
from fastworkflow.command_metadata_api import CommandMetadataAPI


class IntentClarificationAgentSignature(dspy.Signature):
    """
    Handle intent detection errors by clarifying user intent.
    You are provided with:
    1. The workflow agent's inputs and trajectory - showing what the agent has been trying to do
    2. Suggested commands metadata (for intent ambiguity) or empty (for intent misunderstanding - use show_available_commands tool)

    Review the agent trajectory to understand the context and what led to this error.
    If suggested_commands_metadata is provided, review it carefully to understand each command's purpose, parameters, and usage.
    If suggested_commands_metadata is empty, use the show_available_commands tool to get the full list of available commands.
    IMPORTANT: When clarifying intent, preserve ALL parameters from the original command.
    Use available tools to resolve ambiguous or misunderstood commands. Use the ask_user tool ONLY as a last resort. Return the complete clarified command with the correct name and all original parameters.
    """
    original_command = dspy.InputField(desc="The original command with all parameters that caused the error.")
    error_message = dspy.InputField(desc="The intent detection error message from the workflow.")
    agent_inputs = dspy.InputField(desc="The original inputs to the workflow agent.")
    agent_trajectory = dspy.InputField(desc="The workflow agent's trajectory showing all actions taken so far leading to this error.")
    clarified_command = dspy.OutputField(desc="The complete command with correct command name AND all original parameters preserved.")


# def _show_available_commands(chat_session: fastworkflow.ChatSession) -> str:
#     """
#     Show available commands to help resolve intent detection errors.

#     Args:
#         chat_session: The chat session instance

#     Returns:
#         List of available commands
#     """

#     current_workflow = chat_session.get_active_workflow()
#     return CommandMetadataAPI.get_command_display_text(
#         subject_workflow_path=current_workflow.folderpath,
#         cme_workflow_path=fastworkflow.get_internal_workflow_path("command_metadata_extraction"),
#         active_context_name=current_workflow.current_command_context_name,
#     )


def _ask_user_for_clarification(
    clarification_request: str,
    chat_session: fastworkflow.ChatSession
) -> str:
    """
    Ask user for clarification when intent is unclear.

    Args:
        clarification_request: The question to ask the user
        chat_session: The chat session instance

    Returns:
        User's response
    """
    command_output = fastworkflow.CommandOutput(
        command_responses=[fastworkflow.CommandResponse(response=clarification_request)],
        workflow_name=chat_session.get_active_workflow().folderpath.split('/')[-1]
    )
    chat_session.command_output_queue.put(command_output)

    user_response = chat_session.user_message_queue.get()

    # Log to action.jsonl (shared with main agent)
    with open("action.jsonl", "a", encoding="utf-8") as f:
        agent_user_dialog = {
            "intent_clarification_agent": True,
            "agent_query": clarification_request,
            "user_response": user_response
        }
        f.write(json.dumps(agent_user_dialog, ensure_ascii=False) + "\n")

    return user_response


def initialize_intent_clarification_agent(
    chat_session: fastworkflow.ChatSession,
    max_iters: int = 20
):
    """
    Initialize a specialized agent for handling intent detection errors.
    This agent has a limited tool set and shares traces with the main execution agent.

    Args:
        chat_session: The chat session instance
        max_iters: Maximum iterations for the agent (default: 10)

    Returns:
        DSPy ReAct agent configured for intent detection error handling
    """
    if not chat_session:
        raise ValueError("chat_session cannot be null")

    # def show_available_commands() -> str:
    #     """
    #     Show all available commands to help resolve intent ambiguity.
    #     """
    #     return _show_available_commands(chat_session)

    def ask_user(clarification_request: str) -> str:
        """
        Ask the user for clarification when the intent is unclear.
        Use this as a last resort when you cannot determine the correct command.

        Args:
            clarification_request: Clear question to ask the user
        """
        return _ask_user_for_clarification(clarification_request, chat_session)

    # Limited tool set for intent detection errors
    tools = [
        # show_available_commands,
        ask_user,
    ]

    return fastWorkflowReAct(
        IntentClarificationAgentSignature,
        tools=tools,
        max_iters=max_iters,
    )

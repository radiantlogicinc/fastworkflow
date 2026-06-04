"""
Intent detection error state module for fastWorkflow.
Tool-free predictor for handling intent detection errors in agent mode.
"""

import dspy

import fastworkflow


class IntentClarificationAgentSignature(dspy.Signature):
    """
    Handle intent detection errors by clarifying user intent.
    You are provided with:
    1. The workflow agent's inputs and trajectory - showing what the agent has been trying to do
    2. Suggested commands metadata (for intent ambiguity) injected via available_commands when present

    Review the agent trajectory to understand the context and what led to this error.
    If available_commands metadata is provided, review it carefully to understand each command's
    purpose, parameters, and usage.
    IMPORTANT: When clarifying intent, preserve ALL parameters from the original command.
    Return the complete clarified command with the correct name and all original parameters.
    If the command cannot be resolved confidently from the provided context, set needs_human=True
    and provide a plain-text clarification_question for the outer agent to ask the user.
    """
    original_command = dspy.InputField(desc="The original command with all parameters that caused the error.")
    error_message = dspy.InputField(desc="The intent detection error message from the workflow.")
    agent_inputs = dspy.InputField(desc="The original inputs to the workflow agent.")
    agent_trajectory = dspy.InputField(desc="The workflow agent's trajectory showing all actions taken so far leading to this error.")
    clarified_command = dspy.OutputField(desc="The complete command with correct command name AND all original parameters preserved.")
    needs_human = dspy.OutputField(
        desc="True only if the command cannot be resolved from the provided context and a human end user is required."
    )
    clarification_question = dspy.OutputField(
        desc="If needs_human, a plain-text question to ask the user; else empty."
    )


def initialize_intent_clarification_agent(
    chat_session: fastworkflow.ChatSession,
):
    """
    Initialize a tool-free predictor for handling intent detection errors.

    Args:
        chat_session: The chat session instance (used for validation; predictor is stateless)

    Returns:
        dspy.ChainOfThought predictor configured for intent detection error handling
    """
    if not chat_session:
        raise ValueError("chat_session cannot be null")

    return dspy.ChainOfThought(IntentClarificationAgentSignature)

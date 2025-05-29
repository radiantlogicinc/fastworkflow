# fastworkflow/run_agent/agent_module.py
import functools
import os
from typing import Any, Optional

import dspy
from colorama import Fore, Style # For logging within the agent tool

import fastworkflow # For WorkflowSession type hint and get_env_var

# DSPy Signature for the Agent
class DialogueWithWorkflow(dspy.Signature):
    """
    "Understand the user's request, interact with the WorkflowAssistant tool to get information, "
    "or perform actions, and then provide a final answer to the user. "
    "If the assistant misunderstands your intent and executes the wrong action, you can respond with 'thats not what i meant'. "
    "If the assistant asks for information/clarification that you cannot provide using the available assistant tools, stop and ask the user as a final resort. " 
    "BUT FIRST, get a good understanding of assistant skills by asking the assistant: 'What can you do?'. "
    "And if none of the skills are relevant, use your internal knowledge to answer the user's question."
    """
    user_query = dspy.InputField(desc="The user's full input or question.")
    final_answer = dspy.OutputField(desc="The agent's comprehensive response to the user after interacting with the workflow.")

def _format_workflow_output_for_agent(command_output: Any) -> str:
    """
    Formats the structured CommandOutput from the workflow into a single string for the agent.
    """
    output_parts = []
    if not hasattr(command_output, 'command_responses') or not command_output.command_responses:
        return "Workflow produced no command responses or the response structure is unexpected."

    for command_response in command_output.command_responses:
        if response_text := getattr(command_response, 'response', None):
            output_parts.append(f"AI Response: {response_text}")

        artifacts = getattr(command_response, 'artifacts', {})
        output_parts.extend(
            f"Artifact: {artifact_name}={artifact_value}"
            for artifact_name, artifact_value in artifacts.items()
        )
        next_actions = getattr(command_response, 'next_actions', [])
        output_parts.extend(f"Next Action: {action}" for action in next_actions)
        
        recommendations = getattr(command_response, 'recommendations', [])
        output_parts.extend(
            f"Recommendation: {recommendation}"
            for recommendation in recommendations
        )
    
    if not output_parts:
        return "Workflow executed but produced no specific output, actions, or recommendations."
    return "\n".join(output_parts)

def _execute_workflow_command_tool(query: str, *, workflow_session_obj: fastworkflow.WorkflowSession) -> str:
    """
    Sends the query to the WorkflowSession and returns a formatted string of the output.
    This function is intended to be used as a tool by a DSPy agent.
    """
    print(f"{Fore.CYAN}{Style.BRIGHT}Agent -> Workflow>{Style.RESET_ALL}{Fore.CYAN} {query}{Style.RESET_ALL}")
    workflow_session_obj.user_message_queue.put(query)
    command_output = workflow_session_obj.command_output_queue.get()
    formatted_output = _format_workflow_output_for_agent(command_output)
    # Log the truncated workflow response to the agent
    print(f"{Fore.CYAN}{Style.BRIGHT}Workflow -> Agent>{Style.RESET_ALL}{Fore.CYAN} {formatted_output[:200].replace(os.linesep, ' ')}...{Style.RESET_ALL}")
    return formatted_output

def initialize_dspy_agent(workflow_session: fastworkflow.WorkflowSession, LLM_AGENT: str, LITELLM_API_KEY_AGENT: Optional[str] = None, max_iters: int = 10):
    """
    Configures and returns a DSPy ReAct agent.
    Raises:
        EnvironmentError: If LLM_AGENT is not set.
        RuntimeError: If there's an error configuring the DSPy LM.
    """
    if not LLM_AGENT:
        # This check might be redundant if LLM_AGENT is checked before calling this function,
        # but good for encapsulation.
        print(f"{Fore.RED}Error: DSPy Language Model name not provided.{Style.RESET_ALL}")
        raise EnvironmentError("DSPy Language Model name not provided.")

    try:
        lm = dspy.LM(model=LLM_AGENT, api_key=LITELLM_API_KEY_AGENT)
        dspy.settings.configure(lm=lm)
        print(f"{Fore.BLUE}DSPy LM Configured: {LLM_AGENT}{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Error configuring DSPy LM: {e}{Style.RESET_ALL}")
        raise RuntimeError(f"Error configuring DSPy LM: {e}") from e

    # --- Initialize DSPy Agent Tool (Function-based) ---
    workflow_assistant_tool = functools.partial(_execute_workflow_command_tool, workflow_session_obj=workflow_session)
    workflow_assistant_tool.__name__ = "WorkflowAssistant"
    workflow_assistant_tool.__doc__ = (
        "Use this tool to send commands or queries to the underlying fastWorkflow application. "
        "Input should be a natural language command or question for the workflow. "
        "Output will be the structured response from the workflow. "
        "Args: query (str): The command or query to send to the workflow system."
    )

    return dspy.ReAct(
        DialogueWithWorkflow,
        tools=[workflow_assistant_tool],
        max_iters=max_iters,
    ) 
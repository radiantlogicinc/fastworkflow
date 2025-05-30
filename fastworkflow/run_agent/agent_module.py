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
    "If the assistant asks for information that you do not have, abort the command and find a skill that provides this information. "
    "If the assistant misunderstands your intent and executes the wrong action, you can abort the command. "
    "If none of the workflowassistant skills are relevant, use your internal knowledge to answer the user's question."
    """
    user_query = dspy.InputField(desc="The user's full input or question.")
    final_answer = dspy.OutputField(desc="The agent's comprehensive response to the user after interacting with the workflow.")

    # "BUT FIRST, get a good understanding of assistant skills by asking the assistant: 'What can you do?'. "

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

def _ask_user_tool(prompt: str) -> str:
    """
    Allows the agent to ask the user for clarification or additional information via CLI.
    This function is intended to be used as a tool by a DSPy agent.
    
    Args:
        prompt (str): The question or request for clarification to present to the user.
        
    Returns:
        str: The user's response.
    """
    print(f"\n{Fore.YELLOW}{Style.BRIGHT}Agent needs clarification:{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{prompt}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Please provide your response: {Style.RESET_ALL}", end="")
    
    try:
        user_response = input().strip()
        print(f"{Fore.GREEN}User response received: {user_response}{Style.RESET_ALL}")
        return user_response
    except (EOFError, KeyboardInterrupt):
        print(f"\n{Fore.RED}User input interrupted. Returning empty response.{Style.RESET_ALL}")
        return "User input was interrupted or unavailable."

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
        "Use this tool to send commands or queries to the WorkflowAssistant. "
        "Input should be a natural language command or question for the workflow. "
        "Output will be the structured response from the workflow. "
        "Args: query (str): The command or query to send to the workflow system."
        "The workflow assistant has the following skills: "
        "1. Cancel pending orders. Example: 'Cancel my order <order_id>. <reason>'"
        "2. Exchange delivered order items. Example: 'Exchange items <item_id1>, <item_id2>, ... with <item_id3>, ... in order <order_id>. My payment id is <payment_id>'"
        "3. Find user id by email. Example: 'Find the user id for email <email_address>.' "
        "4. Find user id by name and zip. Example: 'Find the user id for name <name> and zip <zip>.' "
        "5. Get order details. Example: 'Get details of order <order_id>.'"
        "6. Get product details. Example: 'Get details of product <product_id>.'"
        "7. Get user details. Example: 'Get details of user <user_id>.'"
        "8. List all product types. Example: 'List all product types.'"
        "9. Modify pending order address. Example: 'Modify the address of order #W123456.'"
        "10. Modify pending order items. Example: 'Modify the items of order #W123456.'"
        "11. Modify pending order payment. Example: 'Modify the payment of order #W123456.'"
        "12. Modify user address. Example: 'Modify the address of user #U123456.'"
        "13. Return delivered order items. Example: 'Return the items of order #W123456.'"
        "14. Transfer to human agent. Example: 'Transfer to human agent.'"
        "15. Abort command. Example: 'Abort command'"
    )

    # workflow_assistant_tool.__doc__ += f"\n{workflow_session.workflow_description}"
    # workflow_assistant_tool.__doc__ += f"\n{workflow_session.workflow_skills}"

    # --- Initialize User Communication Tool ---
    ask_user_tool = _ask_user_tool
    ask_user_tool.__name__ = "AskUser"
    ask_user_tool.__doc__ = (
        "Use this tool to ask the user for clarification, additional information, or guidance when needed. "
        "This is especially useful when the workflow assistant requests information that only the user can provide, "
        "or when you need to resolve ambiguity in the user's request. "
        "Args: prompt (str): A clear, specific question or request for the user."
        "USE THIS TOOL SPARINGLY AND ONLY AFTER YOU HAVE CONFIRMED THAT AVAILABLE ASSISTANT SKILLS A"
    )

    return dspy.ReAct(
        DialogueWithWorkflow,
        tools=[workflow_assistant_tool, ask_user_tool],
        max_iters=max_iters,
    ) 
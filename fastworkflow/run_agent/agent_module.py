# fastworkflow/run_agent/agent_module.py
"""
High-level planning agent module for fastWorkflow.
Uses the integrated workflow tool agent from ChatSession.
"""
import functools
import os
from typing import Any, Optional, List, Dict

import dspy
from colorama import Fore, Style

import fastworkflow
from fastworkflow.mcp_server import FastWorkflowMCPServer


# DSPy Signature for the High-Level Planning Agent
class PlanningAgentSignature(dspy.Signature):
    """
    Prepare and execute a plan for building the final answer using the WorkflowAssistant tool.
    """
    user_query = dspy.InputField(desc="The user's full input or question.")
    final_answer = dspy.OutputField(desc="The agent's comprehensive response to the user after interacting with the workflow.")


def _format_workflow_output_for_agent(command_output: Any) -> str:
    """
    Formats the structured CommandOutput from the workflow into a single string for the agent.
    Handles both regular command responses and MCP tool results.
    """
    # Check if this is an MCP result converted to CommandOutput
    if hasattr(command_output, '_mcp_source'):
        return _format_mcp_result_for_agent(command_output._mcp_source)
    
    # Otherwise use existing logic for regular command responses
    output_parts = []
    if not hasattr(command_output, 'command_responses') or not command_output.command_responses:
        return "Workflow produced no command responses or the response structure is unexpected."

    for command_response in command_output.command_responses:
        if response_text := getattr(command_response, 'response', None):
            output_parts.append(f"{response_text}")

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


def _format_mcp_result_for_agent(mcp_result) -> str:
    """Format MCPToolResult specifically for agent consumption"""
    if mcp_result.isError:
        return f"Error: {mcp_result.content[0].text}"
    else:
        return mcp_result.content[0].text


def _build_assistant_tool_documentation(available_tools: List[Dict]) -> str:
    """Build simplified tool documentation for the main agent's WorkflowAssistant tool."""
    
    # Guidance for the MAIN AGENT on how to call WorkflowAssistant
    main_agent_guidance = """
    Use the WorkflowAssistant to interact with a suite of underlying tools to assist the user.
    It takes a natural language query as input and delegates to an internal agent 
    that will try to understand the request, select the most appropriate tool, and execute it.
    Example tool_args: {"tool_request": "<A single tool request with tool description and all required input parameter names and values>"}

    Available tools that WorkflowAssistant can access:
    """

    tool_docs = []
    for tool_def in available_tools:
        tool_name = tool_def['name']
        tool_desc = tool_def['description']

        # Main agent does not need the detailed input schema, only name, description and parameters.
        tool_docs.append(
            f"\nTool Name: \"{tool_name}\""
            f"\nDescription: {tool_desc}"
            f"\nRequired Parameters: {tool_def['inputSchema']['required']}"
        ) 

    return main_agent_guidance + "\n".join(tool_docs)


def _execute_workflow_command_tool_with_delegation(tool_request: str,
                                                   *, 
                                                   chat_session: fastworkflow.ChatSession) -> str:
    """
    Delegate tool requests to the workflow via queues.
    This is used by the high-level planning agent in run_agent.
    """
    print(f"{Fore.CYAN}{Style.BRIGHT}Agent -> Workflow>{Style.RESET_ALL}{Fore.CYAN} {tool_request}{Style.RESET_ALL}")

    # Send the request through the user message queue
    chat_session.user_message_queue.put(tool_request)
    
    # Get the response from the command output queue
    command_output = chat_session.command_output_queue.get()
    
    # Format the output for the agent
    result = _format_workflow_output_for_agent(command_output)

    print(f"{Fore.BLUE}{Style.BRIGHT}Workflow -> Agent>{Style.RESET_ALL}{Fore.BLUE} {result.replace(os.linesep, ' ')}{Style.RESET_ALL}")
    return result


def _ask_user_tool(prompt: str) -> str:
    """
    Allows the agent to ask the user for clarification or additional information via CLI.
    
    Args:
        prompt (str): The question or request for clarification to present to the user.
        
    Returns:
        str: The user's response.
    """
    print(f"{Fore.YELLOW}{Style.BRIGHT}Agent -> User> Agent needs clarification!{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{prompt}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}User -> Agent> {Style.RESET_ALL}", end="")
    
    user_response = input().strip()
    print(f"{Fore.GREEN}User response received: {user_response}{Style.RESET_ALL}")
    return user_response


def initialize_dspy_agent(chat_session: fastworkflow.ChatSession, LLM_AGENT: str, LITELLM_API_KEY_AGENT: Optional[str] = None, max_iters: int = 25, clear_cache: bool = False):
    """
    Configures and returns a high-level DSPy ReAct planning agent.
    The workflow tool agent is already integrated in the ChatSession.
    
    Args:
        chat_session: ChatSession instance (should be in agent mode)
        LLM_AGENT: Language model name
        LITELLM_API_KEY_AGENT: API key for the language model
        max_iters: Maximum iterations for the ReAct agent
        clear_cache: If True, clears DSPy cache before initialization
    
    Raises:
        EnvironmentError: If LLM_AGENT is not set.
        RuntimeError: If there's an error configuring the DSPy LM.
    """
    if not LLM_AGENT:
        print(f"{Fore.RED}Error: DSPy Language Model name not provided.{Style.RESET_ALL}")
        raise EnvironmentError("DSPy Language Model name not provided.")

    # Configure DSPy LM for the high-level agent
    lm = dspy.LM(model=LLM_AGENT, api_key=LITELLM_API_KEY_AGENT)
    dspy.settings.configure(lm=lm)

    # Get available tools for documentation
    mcp_server = FastWorkflowMCPServer(chat_session)
    available_tools = mcp_server.list_tools()

    # WorkflowAssistant Tool - delegates to the integrated workflow tool agent
    _workflow_assistant_partial_func = functools.partial(
        _execute_workflow_command_tool_with_delegation,
        chat_session=chat_session
    )
    # Set the docstring for the partial object
    _workflow_assistant_partial_func.__doc__ = _build_assistant_tool_documentation(available_tools)

    workflow_assistant_instance = dspy.Tool(
        name="WorkflowAssistant",
        func=_workflow_assistant_partial_func
    )

    # AskUser Tool
    _ask_user_tool.__doc__ = (
        "Use this tool to get information from the user. "
        "Use it as the last resort if information is not available via any of the other tools. "
        "Args: prompt (str): A clear specific request with helpful context based on the information already gathered."
    )

    ask_user_instance = dspy.Tool(
        name="AskUser",
        func=_ask_user_tool
    )

    return dspy.ReAct(
        PlanningAgentSignature,
        tools=[workflow_assistant_instance, ask_user_instance],
        max_iters=max_iters,
    )
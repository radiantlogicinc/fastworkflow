# fastworkflow/run_agent/agent_module.py
import functools
import os
import json
from typing import Any, Optional, List, Dict
import traceback # Keep for now, might be used by other parts or future debugging

import dspy
from colorama import Fore, Style # For logging within the agent tool

import fastworkflow # For WorkflowSession type hint and get_env_var
from fastworkflow.mcp_server import FastWorkflowMCPServer

def clear_dspy_cache():
    """
    Clear DSPy LLM cache completely by disabling both disk and memory cache.
    Call this before initializing agents if you want fresh LLM calls every time.
    """
    dspy.configure_cache(
        enable_disk_cache=False,
        enable_memory_cache=False,
        enable_litellm_cache=False
    )

def configure_dspy_cache(enable_cache: bool = True, cache_dir: Optional[str] = None):
    """
    Configure DSPy caching behavior.
    
    Args:
        enable_cache: Whether to enable caching (True) or disable it completely (False)
        cache_dir: Optional custom cache directory
    """
    if not enable_cache:
        clear_dspy_cache()
        return
    
    cache_config = {
        "enable_disk_cache": True,
        "enable_memory_cache": True,
        "enable_litellm_cache": False
    }
    
    if cache_dir:
        cache_config["disk_cache_dir"] = cache_dir
    
    print(f"{Fore.BLUE}{Style.BRIGHT}🔧 Configuring DSPy cache: {cache_config}{Style.RESET_ALL}")
    dspy.configure_cache(**cache_config)
    print(f"{Fore.GREEN}✅ DSPy cache configured{Style.RESET_ALL}")

# DSPy Signature for the Agent
class PlanningAgentSignature(dspy.Signature):
    """
    'Prepare and execute a plan for building the final answer using the WorkflowAssistant tool"
    """
    user_query = dspy.InputField(desc="The user's full input or question.")
    final_answer = dspy.OutputField(desc="The agent's comprehensive response to the user after interacting with the workflow.")

# DSPy Signature for the MCP Tool Agent
class WorkflowAssistantSignature(dspy.Signature):
    """
    "Understand the tool request. Based on this tool request, select the most appropriate tool from the available list. "
    "Then, construct a complete and valid specially formatted query string for that chosen tool, including its specific arguments. "
    "Finally, invoke the chosen tool by passing this query string as its argument. "
    "Tool execution may return an error message because of missing or invalid parameter values. "
    "Retry with a corrected query string if: "
    "1. Missing parameters can be found in the tool request "
    "2. Error message indicates parameter values are improperly formatted and the formatting errors can be corrected using your internal knowledge. "
    "Otherwise, return the error message back to the user and finish. "
    "Invoke only one tool. DO NOT invoke any other tools except the one explicitly requested"
    """
    tool_request = dspy.InputField(desc="The natural language tool request.")
    tool_result = dspy.OutputField(desc="Result or information request after invoking the tool.")

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
    """Build simplified tool documentation for the main agent's WorkflowAssistant tool.
    This documentation is used as the __doc__ string for the WorkflowAssistant tool function.
    It should guide the main LLM on how to call WorkflowAssistant, primarily by passing the user's query.
    It lists available tools by name and description only.
    """
    
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

# def _create_individual_mcp_tool(tool_def: Dict, workflow_session_obj: fastworkflow.WorkflowSession):
#     """Create a DSPy tool function for a specific MCP tool.
#     This tool expects a single string argument 'mcp_json_payload' containing the full MCP JSON for the specific tool.
#     It then passes this payload to _execute_workflow_command_tool for processing by the WorkflowSession.
#     """
#     tool_name = tool_def['name']
#     tool_desc = tool_def['description']

#     example_args = {}
#     if tool_def.get('inputSchema', {}).get('properties'):
#         for param_name in tool_def.get('inputSchema', {}).get('properties', {}):
#             if param_name == "command":
#                 example_args[param_name] = '<the natural language query (tool_request)>'
#             else:
#                 example_args[param_name] = f"value_for_{param_name}" 

#     example_mcp_for_this_tool = {
#         "type": "mcp_tool_call",
#         "tool_call": {
#             "name": tool_name,
#             "arguments": example_args
#         }
#     }
#     # For the docstring, we need to show what the mcp_json_payload *string itself* should look like,
#     # and then how it fits into the tool_args for the ReAct agent.
#     # json.dumps(example_mcp_for_this_tool) gives the string content of mcp_json_payload.
#     # json.dumps(json.dumps(example_mcp_for_this_tool)) gives an escaped string suitable for embedding in another JSON string (like tool_args value).
#     string_content_of_mcp_json_payload = json.dumps(example_mcp_for_this_tool, indent=2)
#     example_tool_args_value = json.dumps(string_content_of_mcp_json_payload) # This is what ReAct should put in tool_args

#     tool_docstring = (
#         f"Executes the '{tool_name}' tool. Tool description: {tool_desc}.\\n"
#         f"To use this tool, you MUST provide a single string argument named 'mcp_json_payload'.\\n"
#         f"The value for 'mcp_json_payload' MUST be a formatted string specifically for the '{tool_name}' tool.\\n"
#         f"Example of the MCP JSON string that should be the VALUE of 'mcp_json_payload':\\n"
#         f"{string_content_of_mcp_json_payload}\\n"
#         f"Therefore, when invoking this tool, your 'tool_args' field should be a JSON object like this (ensure the mcp_json_payload value is a correctly escaped JSON string):\\n"
#         f'{{ "mcp_json_payload": {example_tool_args_value} }}'
#     )

#     def individual_tool(mcp_json_payload: str) -> str:
#         """Receives a complete MCP JSON string for the '{tool_name}' tool and passes it to the core workflow execution."""
#         return _execute_workflow_mcp_tool(mcp_json_payload=mcp_json_payload, workflow_session_obj=workflow_session_obj)
    
#     individual_tool.__name__ = tool_name 
#     individual_tool.__doc__ = tool_docstring
    
#     return individual_tool

def _create_individual_query_tool(tool_def: Dict, workflow_session_obj: fastworkflow.WorkflowSession):
    """Create a DSPy tool function for a specific MCP tool.
    This tool expects a single string argument 'query' for the specific tool.
    It then passes this command to _execute_workflow_command_tool for processing by the WorkflowSession.
    """
    tool_name = tool_def['name']
    tool_desc = tool_def['description']

    example_query = f'@{tool_name}'
    param_name_value_list = []
    for param_name, param_dict in tool_def['inputSchema']['properties'].items():
        param_name_value_list.append((param_name, f"<replace with {param_dict.get('description', 'value')} or '{param_dict.get('default', 'null')}'>"))

    if param_name_value_list:
        param_values = ', '.join(f'{name}={value}' for name, value in param_name_value_list)
        example_query = f'{example_query} {param_values}'

    tool_docstring = (
        f"Executes the '{tool_name}' tool. Tool description: {tool_desc}.\\n"
        f"To use this tool, you MUST provide a single string argument named 'query'.\\n"
        f"The value for 'query' MUST be a string specifically formatted for the '{tool_name}' tool.\\n"
        f"Example of a query string that should be the VALUE of 'query':\\n"
        f"{example_query}\\n"
        f"Therefore, when invoking this tool, your 'tool_args' field should be a JSON object like this:\\n"
        f'{{ "query": "{example_query}" }}'
        f'If there is a PARAMETER EXTRACTION ERROR, ignore the instructions above and just follow the instructions returned in the tool response'
    )

    def individual_tool(query: str) -> str:
        """Receives a query string for the '{tool_name}' tool and passes it to the core workflow execution."""
        return _execute_workflow_query_tool(query=query, workflow_session_obj=workflow_session_obj)
    
    individual_tool.__name__ = tool_name 
    individual_tool.__doc__ = tool_docstring
    
    return individual_tool

def initialize_workflow_tool_agent(mcp_server: FastWorkflowMCPServer, max_iters: int = 5):
    """
    Initialize and return a DSPy ReAct agent that exposes individual MCP tools.
    Each tool expects a single MCP JSON string payload for its specific tool.
    """
    available_tools = mcp_server.list_tools()
    
    if not available_tools:
        print(f"{Fore.YELLOW}Warning: No MCP tools available for individual tool agent{Style.RESET_ALL}")
        return None
    
    workflow_session_obj = mcp_server.workflow_session # Get workflow_session from mcp_server
    if not workflow_session_obj:
        # This should ideally not happen if mcp_server is correctly initialized
        print(f"{Fore.RED}Error: WorkflowSession not found in MCP Server. Cannot create individual tools.{Style.RESET_ALL}")
        return None

    individual_tools = []
    for tool_def in available_tools:
        if tool_def['name'] == "transfer_to_human_agents":
            continue
        # tool_func = _create_individual_mcp_tool(tool_def, workflow_session_obj) # Pass workflow_session_obj
        tool_func = _create_individual_query_tool(tool_def, workflow_session_obj) # Pass workflow_session_obj
        individual_tools.append(tool_func)
    
    return dspy.ReAct(
        WorkflowAssistantSignature,
        tools=individual_tools,
        max_iters=max_iters,
    )

# def _execute_workflow_mcp_tool(mcp_json_payload: str, *, workflow_session_obj: fastworkflow.WorkflowSession) -> str:
#     """
#     Process JSON MCP tool call query.
#     This function is intended to be used as a tool by a DSPy agent.
#     """
#     print(f"{Fore.CYAN}{Style.BRIGHT}Workflow Assistant -> Workflow>{Style.RESET_ALL}{Fore.CYAN} {mcp_json_payload}{Style.RESET_ALL}")
    
#     workflow_session_obj.user_message_queue.put(mcp_json_payload)
    
#     # Get response and format
#     command_output = workflow_session_obj.command_output_queue.get()
#     formatted_output = _format_workflow_output_for_agent(command_output)
    
#     # Log the truncated workflow response to the agent
#     print(f"{Fore.CYAN}{Style.BRIGHT}Workflow -> Workflow Assistant>{Style.RESET_ALL}{Fore.CYAN} {formatted_output.replace(os.linesep, ' ')}{Style.RESET_ALL}")
#     return formatted_output

def _execute_workflow_query_tool(query: str, *, workflow_session_obj: fastworkflow.WorkflowSession) -> str:
    """
    Process plain utterance query.
    This function is intended to be used as a tool by a DSPy agent.
    """
    print(f"{Fore.CYAN}{Style.BRIGHT}Workflow Assistant -> Workflow>{Style.RESET_ALL}{Fore.CYAN} {query}{Style.RESET_ALL}")
    
    workflow_session_obj.user_message_queue.put(query)
    
    # Get response and format
    command_output = workflow_session_obj.command_output_queue.get()
    formatted_output = _format_workflow_output_for_agent(command_output)
    
    # Log the truncated workflow response to the agent
    print(f"{Fore.CYAN}{Style.BRIGHT}Workflow -> Workflow Assistant>{Style.RESET_ALL}{Fore.CYAN} {formatted_output.replace(os.linesep, ' ')}{Style.RESET_ALL}")
    return formatted_output

def _execute_workflow_command_tool_with_delegation(tool_request: str,
                                                   *, 
                                                   workflow_tool_agent) -> str:
    """
    Delegate JSON MCP tool requests to MCP Tool Agent
    This function is intended to be used as a tool by the Workflow DSPy agent.
    """
    print(f"{Fore.CYAN}{Style.BRIGHT}Agent -> Workflow Assistant>{Style.RESET_ALL}{Fore.CYAN} {tool_request}{Style.RESET_ALL}")

    agent_result = workflow_tool_agent(tool_request=tool_request)

    result = (
        agent_result.tool_result
        if hasattr(agent_result, 'tool_result')
        else str(agent_result)
    )
    print(f"{Fore.BLUE}{Style.BRIGHT}Workflow Assistant -> Agent>{Style.RESET_ALL}{Fore.BLUE} {result.replace(os.linesep, ' ')}{Style.RESET_ALL}")
    return result

def _ask_user_tool(prompt: str) -> str:
    """
    Allows the agent to ask the user for clarification or additional information via CLI.
    This function is intended to be used as a tool by a DSPy agent.
    
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

def initialize_dspy_agent(workflow_session: fastworkflow.WorkflowSession, LLM_AGENT: str, LITELLM_API_KEY_AGENT: Optional[str] = None, max_iters: int = 25, clear_cache: bool = False):
    """
    Configures and returns a DSPy ReAct agent.
    
    Args:
        workflow_session: WorkflowSession instance
        LLM_AGENT: Language model name
        LITELLM_API_KEY_AGENT: API key for the language model
        max_iters: Maximum iterations for the ReAct agent
        clear_cache: If True, clears DSPy cache before initialization
    
    Raises:
        EnvironmentError: If LLM_AGENT is not set.
        RuntimeError: If there's an error configuring the DSPy LM.
    """
    if not LLM_AGENT:
        # This check might be redundant if LLM_AGENT is checked before calling this function,
        # but good for encapsulation.
        print(f"{Fore.RED}Error: DSPy Language Model name not provided.{Style.RESET_ALL}")
        raise EnvironmentError("DSPy Language Model name not provided.")

    # 🗑️ Clear cache if requested
    if clear_cache:
        clear_dspy_cache()
    else:
        # Configure cache with defaults (enabled)
        configure_dspy_cache(enable_cache=True)

    lm = dspy.LM(model=LLM_AGENT, api_key=LITELLM_API_KEY_AGENT)
    dspy.settings.configure(lm=lm)

    # --- Initialize MCP Server and get available tools ---
    mcp_server = FastWorkflowMCPServer(workflow_session)
    available_tools = mcp_server.list_tools()

    # --- Initialize MCP Tool Agent ---
    workflow_tool_agent = initialize_workflow_tool_agent(mcp_server, max_iters=5)

    # WorkflowAssistant Tool
    _workflow_assistant_partial_func = functools.partial(
        _execute_workflow_command_tool_with_delegation,
        workflow_tool_agent=workflow_tool_agent
    )
    # Set the docstring for the partial object, which ReAct will use as the description.
    _workflow_assistant_partial_func.__doc__ = _build_assistant_tool_documentation(available_tools)

    workflow_assistant_instance = dspy.Tool(
        name="WorkflowAssistant",
        func=_workflow_assistant_partial_func
    )

    # AskUser Tool
    # Ensure _ask_user_tool itself has the correct docstring, as it's used directly.
    _ask_user_tool.__doc__ = (
        "Use this tool to get information from the user. "
        "Use it as the last resort if information is not available via any of the other tools. "
        "Args: prompt (str): A clear specific request with helpful context based on the information already gathered."
    ) # Appended Args to docstring for clarity

    ask_user_instance = dspy.Tool(
        name="AskUser",
        func=_ask_user_tool # Direct function reference
        # Removed description and InputField kwargs
    )

    return dspy.ReAct(
        PlanningAgentSignature,
        tools=[workflow_assistant_instance, ask_user_instance], # Use instances of dspy.Tool
        max_iters=max_iters,
    )

"""
Agent integration module for fastWorkflow.
Provides workflow tool agent functionality for intelligent tool selection.
"""


import contextlib
import os
import json
from typing import Any, List, Dict

import dspy

import fastworkflow
from fastworkflow.mcp_server import FastWorkflowMCPServer


class WorkflowAssistantSignature(dspy.Signature):
    """
    Understand the tool request. Based on this tool request, select the most appropriate tool from the available list.
    Then, construct a complete and valid specially formatted query string for that chosen tool, including its specific arguments.
    Finally, invoke the chosen tool by passing this query string as its argument.
    Tool execution may return an error message because of missing or invalid parameter values.
    Retry with a corrected query string if:
    1. Missing parameters can be found in the tool request
    2. Error message indicates parameter values are improperly formatted and the formatting errors can be corrected using your internal knowledge.
    Otherwise, return the error message back to the user and finish.
    Invoke only one tool. DO NOT invoke any other tools except the one explicitly requested.
    """
    tool_request = dspy.InputField(desc="The natural language tool request.")
    tool_result = dspy.OutputField(desc="Result or information request after invoking the tool.")





def _create_individual_query_tool(tool_def: Dict, chat_session_obj: fastworkflow.ChatSession):
    """Create a DSPy tool function for a specific MCP tool.
    This tool expects a single string argument 'query' for the specific tool.
    """
    tool_name = tool_def['name']
    tool_desc = tool_def['description']

    example_query = f'{tool_name}'
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
        """Receives a query string for the tool and passes it to the core workflow execution."""
        return _execute_workflow_query_tool(query=query, chat_session_obj=chat_session_obj)
    
    individual_tool.__name__ = tool_name 
    individual_tool.__doc__ = tool_docstring
    
    return individual_tool


def _execute_workflow_query_tool(query: str, *, chat_session_obj: fastworkflow.ChatSession) -> str:
    """
    Process plain utterance query.
    This function is intended to be used as a tool by a DSPy agent.
    """
    # Directly invoke the command without going through queues
    # This allows the agent to synchronously call workflow tools
    from fastworkflow.command_executor import CommandExecutor
    command_output = CommandExecutor.invoke_command(chat_session_obj, query)

    # Format output - extract text from command response
    if hasattr(command_output, 'command_responses') and command_output.command_responses:
        response_parts = []
        response_parts.extend(
            cmd_response.response
            for cmd_response in command_output.command_responses
            if hasattr(cmd_response, 'response') and cmd_response.response
        )
        return "\n".join(response_parts) if response_parts else "Command executed successfully."

    return "Command executed but produced no output."



def initialize_workflow_tool_agent(mcp_server: FastWorkflowMCPServer, max_iters: int = 5):
    """
    Initialize and return a DSPy ReAct agent that exposes individual MCP tools.
    Each tool expects a single query string for its specific tool.
    
    Args:
        mcp_server: FastWorkflowMCPServer instance
        max_iters: Maximum iterations for the ReAct agent
        
    Returns:
        DSPy ReAct agent configured with workflow tools
    """
    # Configure DSPy if not already configured
    if not dspy.settings.lm:
        if LLM_AGENT := fastworkflow.get_env_var("LLM_AGENT"):
            LITELLM_API_KEY_AGENT = fastworkflow.get_env_var("LITELLM_API_KEY_AGENT")
            lm = dspy.LM(model=LLM_AGENT, api_key=LITELLM_API_KEY_AGENT)
            dspy.settings.configure(lm=lm)

    available_tools = mcp_server.list_tools()

    if not available_tools:
        return None

    chat_session_obj = mcp_server.chat_session
    if not chat_session_obj:
        return None

    individual_tools = []
    for tool_def in available_tools:
        if tool_def['name'] == "transfer_to_human_agents":
            continue
        tool_func = _create_individual_query_tool(tool_def, chat_session_obj)
        individual_tools.append(tool_func)

    return dspy.ReAct(
        WorkflowAssistantSignature,
        tools=individual_tools,
        max_iters=max_iters,
    )





def get_enhanced_what_can_i_do_output(chat_session: fastworkflow.ChatSession) -> Dict[str, Any]:
    """
    Get enhanced command information for agent mode.
    Returns structured JSON with context info, command details, etc.
    
    Args:
        chat_session: The active ChatSession
    
    Returns:
        Dictionary with enhanced command and context information
    """
    workflow = fastworkflow.ChatSession.get_active_workflow()
    if not workflow:
        return {"error": "No active workflow"}

    # Get the current context information
    context_info = {
        "name": workflow.current_command_context_name,
        "display_name": workflow.current_command_context_displayname,
        "description": "",  # Could be extracted from context class docstring
        "inheritance": [],  # Could be extracted from context_inheritance_model.json
        "containment": []   # Could be extracted from context_containment_model.json
    }

    # Get command information
    from fastworkflow.command_routing import RoutingRegistry
    from fastworkflow.command_directory import CommandDirectory

    subject_crd = RoutingRegistry.get_definition(workflow.folderpath)
    cme_crd = RoutingRegistry.get_definition(
        fastworkflow.get_internal_workflow_path("command_metadata_extraction")
    )

    # Get available commands
    cme_command_names = cme_crd.get_command_names('IntentDetection')
    subject_command_names = subject_crd.get_command_names(workflow.current_command_context_name)

    candidate_commands = set(cme_command_names) | set(subject_command_names)

    # Filter and build command details
    commands = []
    for fq_cmd in candidate_commands:
        if fq_cmd == "wildcard":
            continue

        # Check if command has utterances
        utterance_meta = (
            subject_crd.command_directory.get_utterance_metadata(fq_cmd) or
            cme_crd.command_directory.get_utterance_metadata(fq_cmd)
        )

        if not utterance_meta:
            continue

        cmd_name = fq_cmd.split("/")[-1]

        # Get command signature information if available
        signature_info = {}
        with contextlib.suppress(Exception):
            # Try to get command class for signature extraction
            cmd_module = subject_crd.get_command_module(fq_cmd)
            if hasattr(cmd_module, 'Signature') and hasattr(cmd_module.Signature, 'Input'):
                input_class = cmd_module.Signature.Input
                signature_info = {
                    "inputs": [
                        {
                            "name": field_name,
                            "type": str(field_info.annotation),
                            "description": field_info.description or "",
                            "examples": getattr(field_info, 'examples', [])
                        }
                        for field_name, field_info in input_class.model_fields.items()
                    ]
                }
        # Get docstring and plain_utterances from utterance metadata if available
        docstring = ""
        plain_utterances = []
        if hasattr(utterance_meta, 'docstring'):
            docstring = utterance_meta.docstring or ""
        if hasattr(utterance_meta, 'plain_utterances'):
            plain_utterances = utterance_meta.plain_utterances or []

        commands.append({
            "qualified_name": fq_cmd,
            "name": cmd_name,
            "signature_docstring": docstring,
            "plain_utterances": plain_utterances,
            **signature_info
        })

    return {
        "context": context_info,
        "commands": sorted(commands, key=lambda x: x["name"])
    }




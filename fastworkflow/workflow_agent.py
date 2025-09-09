"""
Agent integration module for fastWorkflow.
Provides workflow tool agent functionality for intelligent tool selection.
"""

import json
import time
import dspy

import fastworkflow
from fastworkflow.utils.logging import logger
from fastworkflow.utils import dspy_utils
from fastworkflow.command_metadata_api import CommandMetadataAPI
from fastworkflow.mcp_server import FastWorkflowMCPServer


class WorkflowAgentSignature(dspy.Signature):
    """
    Carefully review the user request and conversation_history, then execute the todo list using available tools for building the final answer.
    All the tasks in the todo list must be completed before returning the final answer.
    """
    user_query = dspy.InputField(desc="The natural language user query.")
    conversation_history: dspy.History = dspy.InputField()
    final_answer = dspy.OutputField(desc="Comprehensive final answer with supporting evidence to demonstrate that all the tasks in the todo list have been completed.")

def _what_can_i_do(chat_session_obj: fastworkflow.ChatSession) -> str:
    """
    Returns a list of available commands, including their names and parameters.
    """
    current_workflow = chat_session_obj.get_active_workflow()
    return CommandMetadataAPI.get_command_display_text(
        subject_workflow_path=current_workflow.folderpath,
        cme_workflow_path=fastworkflow.get_internal_workflow_path("command_metadata_extraction"),
        active_context_name=current_workflow.current_command_context_name,
    )

def _clarify_ambiguous_intent(
        correct_command_name: str,
        chat_session_obj: fastworkflow.ChatSession) -> str:
    """
    Call this tool ONLY in the intent detection error state (ambiguous or misunderstood intent) to provide the exact command name.
    The intent detection error message will list the command names to pick from.
    """
    return _execute_workflow_query(correct_command_name, chat_session_obj = chat_session_obj)

def _provide_missing_or_corrected_parameters(
        missing_or_corrected_parameter_values: list[str|int|float|bool],
        chat_session_obj: fastworkflow.ChatSession) -> str:
    """
    Call this tool ONLY in the parameter extraction error state to provide missing or corrected parameter values.
    Missing parameter values may be found in the user query, or information already available, or by aborting and executing a different command (refer to the optional 'available_from' hint for guidance on appropriate commands to use to get the information).
    If the error message indicates parameter values are improperly formatted, correct using your internal knowledge and command metadata information.
    """
    if missing_or_corrected_parameter_values:
        command = ', '.join(missing_or_corrected_parameter_values)
    else:
        return "Provide missing or corrected parameter values or abort"
    
    return _execute_workflow_query(command, chat_session_obj = chat_session_obj)

def _abort_current_command_to_exit_parameter_extraction_error_state(
        chat_session_obj: fastworkflow.ChatSession) -> str:
    """
    Call this tool ONLY in the parameter extraction error state when you want to get out of the parameter extraction error state and execute a different command.
    """
    return _execute_workflow_query('abort', chat_session_obj = chat_session_obj)

def _intent_misunderstood(
        chat_session_obj: fastworkflow.ChatSession) -> str:
    """
    Shows the full list of available command names so you can specify the command name you really meant
    Call this tool when your intent is misunderstood (i.e. the wrong command name is executed).
    """
    return _execute_workflow_query('you misunderstood', chat_session_obj = chat_session_obj)

def _execute_workflow_query(command: str, chat_session_obj: fastworkflow.ChatSession) -> str:
    """
    Executes the command and returns either a response, or a clarification request.
    Use the "what_can_i_do" tool to get details on available commands, including their names and parameters. Fyi, values in the 'examples' field are fake and for illustration purposes only.
    Commands must be formatted using plain text for command name followed by XML tags enclosing parameter values (if any) as follows: command_name <param1_name>param1_value</param1_name> <param2_name>param2_value</param2_name> ...
    Don't use this tool to respond to a clarification requests in PARAMETER EXTRACTION ERROR state
    """
    # Emit trace event before execution
    chat_session_obj.command_trace_queue.put(fastworkflow.CommandTraceEvent(
        direction=fastworkflow.CommandTraceEventDirection.AGENT_TO_WORKFLOW,
        raw_command=command,
        command_name=None,
        parameters=None,
        response_text=None,
        success=None,
        timestamp_ms=int(time.time() * 1000),
    ))

    # Directly invoke the command without going through queues
    # This allows the agent to synchronously call workflow tools
    from fastworkflow.command_executor import CommandExecutor
    command_output = CommandExecutor.invoke_command(chat_session_obj, command)

    # Emit trace event after execution
    # Extract command name and parameters from command_output
    name = command_output.command_name
    params = command_output.command_parameters
    
    # Handle parameter serialization
    params_dict = params.model_dump() if params else None
    
    # Extract response text
    resp_text = ""
    if command_output.command_responses:
        response_parts = [
            cmd_response.response
            for cmd_response in command_output.command_responses
            if cmd_response.response
        ]
        resp_text = "\n".join(response_parts)
    
    chat_session_obj.command_trace_queue.put(fastworkflow.CommandTraceEvent(
        direction=fastworkflow.CommandTraceEventDirection.WORKFLOW_TO_AGENT,
        raw_command=None,
        command_name=name,
        parameters=params_dict,
        response_text=resp_text or "",
        success=bool(command_output.success),
        timestamp_ms=int(time.time() * 1000),
    ))

    # Append executed action to action.json for external consumers (agent mode only)
    record = {
        "command" if command_output.success else "failing command": command,
        "command_name": name,
        "parameters": params_dict,
        "response": resp_text if command_output.success else ""
    }
    with open("action.json", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Format output - extract text from command response
    if command_output.command_responses:
        response_parts = []
        response_parts.extend(
            cmd_response.response
            for cmd_response in command_output.command_responses
            if cmd_response.response
        )
        return "\n".join(response_parts) if response_parts else "Command executed successfully."

    return "Command executed but produced no output."

def _missing_information_guidance_tool(
        how_to_find_request: str, 
        chat_session_obj: fastworkflow.ChatSession) -> str:
    """
    Request guidance on finding missing information. 
    The how_to_find_request must be plain text without any formatting.
    """
    lm = dspy_utils.get_lm("LLM_AGENT", "LITELLM_API_KEY_AGENT")
    with dspy.context(lm=lm):
        guidance_func = dspy.ChainOfThought(
            "command_info, missing_information_guidance_request -> guidance: str")
        prediction = guidance_func(
            command_info=_what_can_i_do(chat_session_obj), 
            missing_information_guidance_request=how_to_find_request)
        return prediction.guidance

def _ask_user_tool(clarification_request: str, chat_session_obj: fastworkflow.ChatSession) -> str:
    """
    As a last resort, request clarification for missing information (only after using the missing_information_guidance_tool) or error correction from the human user. 
    The clarification_request must be plain text without any formatting.
    """
    command_output = fastworkflow.CommandOutput(
        command_responses=[fastworkflow.CommandResponse(response=clarification_request)]
    )
    chat_session_obj.command_output_queue.put(command_output)
    user_query = chat_session_obj.user_message_queue.get()
    return _think_and_plan(user_query, chat_session_obj)

def initialize_workflow_tool_agent(mcp_server: FastWorkflowMCPServer, max_iters: int = 25):
    """
    Initialize and return a DSPy ReAct agent that exposes individual MCP tools.
    Each tool expects a single query string for its specific tool.
    
    Args:
        mcp_server: FastWorkflowMCPServer instance
        max_iters: Maximum iterations for the ReAct agent
        
    Returns:
        DSPy ReAct agent configured with workflow tools
    """
    chat_session_obj = mcp_server.chat_session
    if not chat_session_obj:
        return None

    def what_can_i_do() -> str:
        """
        Returns a list of available commands, including their names and parameters
        """
        return _what_can_i_do(chat_session_obj=chat_session_obj)

    def clarify_ambiguous_intent(
            correct_command_name: str) -> str:
        """
        Call this tool ONLY in the intent detection error state to provide the exact command name.
        The intent detection error message will list the command names to pick from.
        """
        return _clarify_ambiguous_intent(correct_command_name, chat_session_obj = chat_session_obj)

    def provide_missing_or_corrected_parameters(
            missing_or_corrected_parameter_values: list[str|int|float|bool]) -> str:
        """
        Call this tool ONLY in the parameter extraction error state to provide missing or corrected parameter values.
        Missing parameter values may be found in the user query, or information already available, or by executing a different command (refer to the optional 'available_from' hint for guidance on appropriate commands to use to get the information).
        If the error message indicates parameter values are improperly formatted, correct using your internal knowledge.
        """
        return _provide_missing_or_corrected_parameters(missing_or_corrected_parameter_values, chat_session_obj=chat_session_obj)

    def abort_current_command_to_exit_parameter_extraction_error_state() -> str:
        """
        Call this tool ONLY when you need to execute a different command to get missing parameters.
        DO NOT execute the same failing command over and over. Either provide_missing_or_corrected_parameters or abort 
        """
        return _abort_current_command_to_exit_parameter_extraction_error_state(
            chat_session_obj=chat_session_obj)

    def intent_misunderstood() -> str:
        """
        Shows the full list of available command names so you can specify the command name you really meant
        Call this tool when your intent is misunderstood (i.e. the wrong command name is executed).
        Do not use this tool if its a missing or invalid parameter issue. Use the provide_missing_or_corrected_parameters tool instead
        """
        return _intent_misunderstood(chat_session_obj = chat_session_obj)

    def execute_workflow_query(command: str) -> str:
        """
        Executes the command and returns either a response, or a clarification request.
        Use the "what_can_i_do" tool to get details on available commands, including their names and parameters. Fyi, values in the 'examples' field are fake and for illustration purposes only.
        Commands must be formatted using plain text for command name followed by XML tags enclosing parameter values (if any) as follows: command_name <param1_name>param1_value</param1_name> <param2_name>param2_value</param2_name> ...
        Don't use this tool to respond to a clarification requests in PARAMETER EXTRACTION ERROR state
        """
        # Retry logic for workflow execution
        max_retries = 2
        for attempt in range(max_retries):
            try:
                return _execute_workflow_query(command, chat_session_obj=chat_session_obj)
            except Exception as e:
                if attempt == max_retries - 1:  # Last attempt
                    message = f"Terminate immediately! Exception processing {command}: {str(e)}"
                    logger.critical(message)                    
                    return message
                # Continue to next attempt
                logger.warning(f"Attempt {attempt + 1} failed for command '{command}': {str(e)}")

    def missing_information_guidance(how_to_find_request: str) -> str:
        """
        Request guidance on finding missing information. 
        The how_to_find_request must be plain text without any formatting.
        """
        return _missing_information_guidance_tool(how_to_find_request, chat_session_obj=chat_session_obj)

    def ask_user(clarification_request: str) -> str:
        """
        As a last resort, request clarification for missing information (only after using the missing_information_guidance_tool) or error correction from the human user. 
        The clarification_request must be plain text without any formatting.
        """
        return _ask_user_tool(clarification_request, chat_session_obj=chat_session_obj)

    tools = [
        what_can_i_do,
        execute_workflow_query,
        missing_information_guidance,
        clarify_ambiguous_intent,
        intent_misunderstood,
        ask_user,
        provide_missing_or_corrected_parameters,
        abort_current_command_to_exit_parameter_extraction_error_state,
    ]

    return dspy.ReAct(
        WorkflowAgentSignature,
        tools=tools,
        max_iters=max_iters,
    )

def _think_and_plan(user_query: str, chat_session_obj: fastworkflow.ChatSession) -> str:
    """
    Returns a refined plan by breaking down a user_query into simpler tasks based only on available commands and returns a todo list.
    """
    class TaskPlannerSignature(dspy.Signature):
        """
        Break down a user_query into simpler tasks based only on available commands and return a todo list.
        If user_query is simple, return a single todo that is the user_query as-is
        """
        user_query: str = dspy.InputField()
        available_commands: list[str] = dspy.InputField()
        todo_list: list[str] = dspy.OutputField(desc="task descriptions as short sentences")

    current_workflow = chat_session_obj.get_active_workflow()
    available_commands = CommandMetadataAPI.get_command_display_text(
        subject_workflow_path=current_workflow.folderpath,
        cme_workflow_path=fastworkflow.get_internal_workflow_path("command_metadata_extraction"),
        active_context_name=current_workflow.current_command_context_name,
    )

    planner_lm = dspy_utils.get_lm("LLM_PLANNER", "LITELLM_API_KEY_PLANNER")
    with dspy.context(lm=planner_lm):
        task_planner_func = dspy.ChainOfThought(TaskPlannerSignature)
        prediction = task_planner_func(user_query=user_query, available_commands=available_commands)

        if not prediction.todo_list or (len(prediction.todo_list) == 1 and prediction.todo_list[0] == user_query):
            return user_query

        return f"{user_query}\nNext steps:\n{'\n'.join([f'{i + 1}. {task}' for i, task in enumerate(prediction.todo_list)])}"

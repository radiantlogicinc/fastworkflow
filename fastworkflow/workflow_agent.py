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
from fastworkflow.utils.react import fastWorkflowReAct


class WorkflowAgentSignature(dspy.Signature):
    """
    Carefully review the user request, then execute the next steps using available tools for building the final answer.
    Every user intent must be fully addressed before returning the final answer.
    """
    user_query = dspy.InputField(desc="The natural language user query.")
    final_answer = dspy.OutputField(desc="Comprehensive final answer with supporting evidence to demonstrate that every user intent has been fully addressed.")

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

# def _clarify_ambiguous_intent(
#         correct_command_name: str,
#         chat_session_obj: fastworkflow.ChatSession) -> str:
#     """
#     Call this tool ONLY in the intent detection error state (ambiguous or misunderstood intent) to provide the exact command name.
#     The intent detection error message will list the command names to pick from.
#     """
#     return _execute_workflow_query(correct_command_name, chat_session_obj = chat_session_obj)

# def _provide_missing_or_corrected_parameters(
#         missing_or_corrected_parameter_values: list[str|int|float|bool],
#         chat_session_obj: fastworkflow.ChatSession) -> str:
#     """
#     Call this tool ONLY in the parameter extraction error state to provide missing or corrected parameter values.
#     Missing parameter values may be found in the user query, or information already available, or by aborting and executing a different command (refer to the optional 'available_from' hint for guidance on appropriate commands to use to get the information).
#     If the error message indicates parameter values are improperly formatted, correct using your internal knowledge and command metadata information.
#     """
#     if missing_or_corrected_parameter_values:
#         command = ', '.join(missing_or_corrected_parameter_values)
#     else:
#         return "Provide missing or corrected parameter values or abort"
    
#   return _execute_workflow_query(command, chat_session_obj = chat_session_obj)

# def _abort_current_command_to_exit_parameter_extraction_error_state(
#         chat_session_obj: fastworkflow.ChatSession) -> str:
#     """
#     Call this tool ONLY in the parameter extraction error state when you want to get out of the parameter extraction error state and execute a different command.
#     """
#     return 

def _intent_misunderstood(
        chat_session_obj: fastworkflow.ChatSession) -> str:
    """
    Shows the full list of available command names so you can specify the command name you really meant
    Call this tool when your intent is misunderstood (i.e. the wrong command name is executed).
    """
    return _what_can_i_do(chat_session_obj = chat_session_obj)

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
    response_text = ""
    if command_output.command_responses:
        response_parts = []
        response_parts.extend(
            cmd_response.response
            for cmd_response in command_output.command_responses
            if cmd_response.response
        )
        response_text = "\n".join(response_parts) \
            if response_parts else "Command executed successfully but produced no output."

    chat_session_obj.command_trace_queue.put(fastworkflow.CommandTraceEvent(
        direction=fastworkflow.CommandTraceEventDirection.WORKFLOW_TO_AGENT,
        raw_command=None,
        command_name=name,
        parameters=params_dict,
        response_text=response_text,
        success=bool(command_output.success),
        timestamp_ms=int(time.time() * 1000),
    ))

    # Append executed action to action.jsonl for external consumers (agent mode only)
    record = {
        "command": command,
        "command_name": name,
        "parameters": params_dict,
        "response": response_text
    }
    with open("action.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Check workflow context to determine if we're in an error state that needs specialized handling
    cme_workflow = chat_session_obj.cme_workflow
    nlu_stage = cme_workflow.context.get("NLU_Pipeline_Stage")

    # Handle intent ambiguity clarification state with specialized agent
    if nlu_stage == fastworkflow.NLUPipelineStage.INTENT_AMBIGUITY_CLARIFICATION:
        if intent_agent := chat_session_obj.intent_clarification_agent:
            # Get suggested commands from intent detection system
            from fastworkflow._workflows.command_metadata_extraction.intent_detection import CommandNamePrediction
            predictor = CommandNamePrediction(cme_workflow)
            suggested_commands = predictor._get_suggested_commands(predictor.path)

            suggested_commands = list(suggested_commands) if suggested_commands is not None else []

            # Get metadata for only the suggested commands
            current_workflow = chat_session_obj.get_active_workflow()
            suggested_commands_metadata = CommandMetadataAPI.get_suggested_commands_metadata(
                subject_workflow_path=current_workflow.folderpath,
                cme_workflow_path=fastworkflow.get_internal_workflow_path("command_metadata_extraction"),
                active_context_name=current_workflow.current_command_context_name,
                suggested_command_names=suggested_commands
            )

            # Get the workflow agent's trajectory and inputs for context
            workflow_tool_agent = chat_session_obj.workflow_tool_agent
            agent_inputs = workflow_tool_agent.inputs if workflow_tool_agent else {}
            agent_trajectory = workflow_tool_agent.current_trajectory if workflow_tool_agent else {}

            lm = dspy_utils.get_lm("LLM_AGENT", "LITELLM_API_KEY_AGENT")
            with dspy.context(lm=lm):
                result = intent_agent(
                    original_command=command,
                    error_message=response_text,
                    agent_inputs=agent_inputs,
                    agent_trajectory=agent_trajectory,
                    suggested_commands_metadata=suggested_commands_metadata
                )
            # The clarified command should have the correct name with all original parameters
            clarified_cmd = result.clarified_command if hasattr(result, 'clarified_command') else str(result)
            # Execute the clarified command
            return _execute_workflow_query(clarified_cmd, chat_session_obj=chat_session_obj)
        else:
            # No intent clarification agent available, fall back to abort
            abort_confirmation = _execute_workflow_query('abort', chat_session_obj=chat_session_obj)
            return f'{response_text}\n{abort_confirmation}'

    # Handle intent misunderstanding clarification state with specialized agent
    if nlu_stage == fastworkflow.NLUPipelineStage.INTENT_MISUNDERSTANDING_CLARIFICATION:
        if intent_agent := chat_session_obj.intent_clarification_agent:
            # Get the workflow agent's trajectory and inputs for context
            workflow_tool_agent = chat_session_obj.workflow_tool_agent
            agent_inputs = workflow_tool_agent.inputs if workflow_tool_agent else {}
            agent_trajectory = workflow_tool_agent.current_trajectory if workflow_tool_agent else {}

            lm = dspy_utils.get_lm("LLM_AGENT", "LITELLM_API_KEY_AGENT")
            with dspy.context(lm=lm):
                result = intent_agent(
                    original_command=command,
                    error_message=response_text,
                    agent_inputs=agent_inputs,
                    agent_trajectory=agent_trajectory,
                    suggested_commands_metadata=""
                )

            clarified_cmd = result.clarified_command if hasattr(result, 'clarified_command') else str(result)

            return _execute_workflow_query(clarified_cmd, chat_session_obj=chat_session_obj)
        else:
            # No intent clarification agent available, fall back to abort
            abort_confirmation = _execute_workflow_query('abort', chat_session_obj=chat_session_obj)
            return f'{response_text}\n{abort_confirmation}'

    # Handle parameter extraction errors with abort
    if nlu_stage == fastworkflow.NLUPipelineStage.PARAMETER_EXTRACTION and not command_output.success:
        abort_confirmation = _execute_workflow_query('abort', chat_session_obj=chat_session_obj)
        return f'{response_text}\n{abort_confirmation}'

    return response_text

# def _missing_information_guidance_tool(
#         how_to_find_request: str, 
#         chat_session_obj: fastworkflow.ChatSession) -> str:
#     """
#     Request guidance on finding missing information. 
#     The how_to_find_request must be plain text without any formatting.
#     """
#     class MissingInfoGuidanceSignature(dspy.Signature):
#         """
#         Carefully review the command info 'available_from' hints to see if the missing information can be found by executing a different command.
#         You may have to walk the graph of commands based on the 'available_from' hints to find the most appropriate command
#         Note that using the wrong command name can produce missing information errors. The requestor should double-check that the command name is correct. 
#         """
#         command_info: str = dspy.InputField()
#         missing_information_guidance_request: str = dspy.InputField()
#         guidance: str = dspy.OutputField()


#     lm = dspy_utils.get_lm("LLM_AGENT", "LITELLM_API_KEY_AGENT")
#     with dspy.context(lm=lm):
#         guidance_func = dspy.ChainOfThought(MissingInfoGuidanceSignature)
#         prediction = guidance_func(
#             command_info=_what_can_i_do(chat_session_obj), 
#             missing_information_guidance_request=how_to_find_request)
#         return prediction.guidance

def _ask_user_tool(clarification_request: str, chat_session_obj: fastworkflow.ChatSession) -> str:
    """
    If the missing_information_guidance_tool does not help and only as the last resort, request clarification for missing information from the human user. 
    The clarification_request must be plain text without any formatting.
    Note that using the wrong command name can produce missing information errors. Double-check with the missing_information_guidance_tool to verify that the correct command name is being used 
    """
    command_output = fastworkflow.CommandOutput(
        command_responses=[fastworkflow.CommandResponse(response=clarification_request)],
        workflow_name = chat_session_obj.get_active_workflow().folderpath.split('/')[-1]
    )    
    chat_session_obj.command_output_queue.put(command_output)

    user_query = chat_session_obj.user_message_queue.get()

    # add the agent user dialog to the log
    with open("action.jsonl", "a", encoding="utf-8") as f:
        agent_user_dialog = {
            "agent_query": clarification_request,
            "user_response": user_query
        }
        f.write(json.dumps(agent_user_dialog, ensure_ascii=False) + "\n")

    return build_query_with_next_steps(user_query, chat_session_obj, with_agent_inputs_and_trajectory = True)

def initialize_workflow_tool_agent(chat_session: fastworkflow.ChatSession, max_iters: int = 25):
    """
    Initialize and return a DSPy ReAct agent that exposes individual MCP tools.
    Each tool expects a single query string for its specific tool.
    
    Args:
        chat_session: fastworkflow.ChatSession instance
        max_iters: Maximum iterations for the ReAct agent
        
    Returns:
        DSPy ReAct agent configured with workflow tools
    """
    chat_session_obj = chat_session
    if not chat_session_obj:
        raise ValueError("chat session cannot be null")

    def what_can_i_do() -> str:
        """
        Returns a list of available commands, including their names and parameters
        """
        return _what_can_i_do(chat_session_obj=chat_session_obj)

    def intent_misunderstood() -> str:
        """
        Shows the full list of available command names so you can specify the command name you really meant
        Call this tool when your intent is misunderstood (i.e. the wrong command name is executed).
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

    # def missing_information_guidance(how_to_find_request: str) -> str:
    #     """
    #     Request guidance on finding missing information. 
    #     The how_to_find_request must be plain text without any formatting.
    #     """
    #     return _missing_information_guidance_tool(how_to_find_request, chat_session_obj=chat_session_obj)

    def ask_user(clarification_request: str) -> str:
        """
        Only as the last resort, request clarification for missing information from the human user. 
        The clarification_request must be plain text without any formatting.
        Note that using the wrong command name can produce missing information errors. Double-check with the what_can_i_do tool to verify that the correct command name is being used 
        """
        return _ask_user_tool(clarification_request, chat_session_obj=chat_session_obj)

    tools = [
        what_can_i_do,
        execute_workflow_query,
        # missing_information_guidance,
        intent_misunderstood,
        ask_user,
    ]

    return fastWorkflowReAct(
        WorkflowAgentSignature,
        tools=tools,
        max_iters=max_iters,
    )


def build_query_with_next_steps(user_query: str, 
    chat_session_obj: fastworkflow.ChatSession, with_agent_inputs_and_trajectory: bool = False) -> str:
    """
    Generate a todo list.
    Return a string that combine the user query and todo list
    """
    class TaskPlannerSignature(dspy.Signature):
        """
        Carefully review the user_query and generate a next steps sequence based only on available commands.
        Walk the graph of commands based on the 'available_from' hints to build the most appropriate command sequence
        Avoid specifying 'ask user' because 9 times out of 10, you can find the information via available commands. 
        """
        user_query: str = dspy.InputField()
        available_commands: list[str] = dspy.InputField()
        next_steps: list[str] = dspy.OutputField(desc="task descriptions as short sentences")

    class TaskPlannerWithTrajectoryAndAgentInputsSignature(dspy.Signature):
        """
        Carefully review agent inputs, agent trajectory and user response and generate a next steps sequence based only on available commands.
        Walk the graph of commands based on the 'available_from' hints to build the most appropriate command sequence
        Avoid specifying 'ask user' because 9 times out of 10, you can find the information via available commands. 
        """
        agent_inputs: dict = dspy.InputField()
        agent_trajectory: dict = dspy.InputField()
        user_response: str = dspy.InputField()
        available_commands: list[str] = dspy.InputField()
        next_steps: list[str] = dspy.OutputField(desc="task descriptions as short sentences")

    current_workflow = chat_session_obj.get_active_workflow()
    available_commands = CommandMetadataAPI.get_command_display_text(
        subject_workflow_path=current_workflow.folderpath,
        cme_workflow_path=fastworkflow.get_internal_workflow_path("command_metadata_extraction"),
        active_context_name=current_workflow.current_command_context_name,
    )

    planner_lm = dspy_utils.get_lm("LLM_PLANNER", "LITELLM_API_KEY_PLANNER")
    with dspy.context(lm=planner_lm):
        if with_agent_inputs_and_trajectory:
            workflow_tool_agent = chat_session_obj.workflow_tool_agent
            task_planner_func = dspy.ChainOfThought(TaskPlannerWithTrajectoryAndAgentInputsSignature)
            prediction = task_planner_func(
                agent_inputs = workflow_tool_agent.inputs,
                agent_trajectory = workflow_tool_agent.current_trajectory,
                user_response = user_query,
                available_commands=available_commands)
        else:
            task_planner_func = dspy.ChainOfThought(TaskPlannerSignature)
            prediction = task_planner_func(
                user_query=user_query,
                available_commands=available_commands)

        if not prediction.next_steps:
            return user_query

        steps_list = '\n'.join([f'{i + 1}. {task}' for i, task in enumerate(prediction.next_steps)])
        user_query_and_next_steps = f"{user_query}\n\nNext steps:\n{steps_list}"
        return (
            f'{available_commands}\n\nUser Query:\n{user_query_and_next_steps}'
            if with_agent_inputs_and_trajectory else
            user_query_and_next_steps
        )

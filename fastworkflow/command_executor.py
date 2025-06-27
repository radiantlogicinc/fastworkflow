import fastworkflow
from fastworkflow.command_interfaces import CommandExecutorInterface

from fastworkflow import Action, CommandOutput, ChatSession
from fastworkflow import ModuleType
from fastworkflow.utils.signatures import InputForParamExtraction
from pathlib import Path
from fastworkflow.command_routing import RoutingDefinition
from typing import Optional
from fastworkflow.command_context_model import CommandContextModel
from fastworkflow.command_directory import CommandDirectory


# ------------------------------------------------------------------
# Module-level delegation configuration and exceptions
# ------------------------------------------------------------------

MAX_DELEGATION_DEPTH: int = 10  # Safety limit for delegation hops


class CommandNotFoundError(Exception):
    """Raised when a command cannot be resolved in any accessible context."""


class CommandExecutor(CommandExecutorInterface):
    @classmethod
    def invoke_command(
        cls,
        chat_session: 'fastworkflow.ChatSession',
        command: str,
    ) -> fastworkflow.CommandOutput:
        if not command:
            return CommandOutput(
                command_responses=[
                    fastworkflow.CommandResponse(
                        response="You just hit the <Enter> key. How about a command or some feedback instead?"
                    )
                ]
            )

        command_output = cls.perform_action(
            chat_session.cme_workflow, 
            Action(
                command_name = "wildcard",
                command = command)
        )

        if command_output.command_handled or not command_output.success:
            return command_output

        command = command_output.command_responses[0].artifacts["command"]
        command_name = command_output.command_responses[0].artifacts["command_name"]
        input_obj = command_output.command_responses[0].artifacts["cmd_parameters"]

        workflow_folderpath = chat_session.app_workflow.folderpath
        command_routing_definition = fastworkflow.RoutingRegistry.get_definition(
            workflow_folderpath
        )

        response_generation_class = command_routing_definition.get_command_class(
            command_name,
            ModuleType.RESPONSE_GENERATION_INFERENCE,
        )
        if not response_generation_class:
            raise ValueError(
                f"Response generation class not found for command name '{command_name}' "
            )
        response_generation_object = response_generation_class()

        if command_parameters_class := (
            command_routing_definition.get_command_class(
                command_name, ModuleType.COMMAND_PARAMETERS_CLASS
            )
        ):
            return response_generation_object(chat_session.app_workflow, command, input_obj)
        else:
            return response_generation_object(chat_session.app_workflow, command)

    @classmethod
    def perform_action(
        cls,
        workflow: fastworkflow.Workflow,
        action: fastworkflow.Action,
    ) -> fastworkflow.CommandOutput:  # sourcery skip: extract-method
        workflow.command_context_for_response_generation = \
            workflow.current_command_context

        workflow_folderpath = workflow.folderpath
        command_routing_definition = fastworkflow.RoutingRegistry.get_definition(workflow_folderpath)

        response_generation_class = (
            command_routing_definition.get_command_class(
                action.command_name,
                ModuleType.RESPONSE_GENERATION_INFERENCE,
            )
        )
        if not response_generation_class:
            raise ValueError(
                f"Response generation class not found for command name '{action.command_name}'"
            )

        response_generation_object = response_generation_class()

        command_parameters_class = (
            command_routing_definition.get_command_class(
                action.command_name, ModuleType.COMMAND_PARAMETERS_CLASS
            )
        )
        if not command_parameters_class:
            return response_generation_object(workflow, action.command)

        if action.parameters:
            input_obj = command_parameters_class(**action.parameters)

            input_for_param_extraction = InputForParamExtraction(command=action.command)
            is_valid, error_msg, _ = input_for_param_extraction.validate_parameters(
                workflow, action.command_name, input_obj
            )
            if not is_valid:
                raise ValueError(f"Invalid action parameters for command '{action.command_name}'\n{error_msg}")
        else:
            input_obj = command_parameters_class()

        return response_generation_object(workflow, action.command, input_obj)

    # MCP-compliant methods
    @classmethod
    def perform_mcp_tool_call(
        cls,
        workflow: fastworkflow.Workflow,
        tool_call: fastworkflow.MCPToolCall,
        command_context: str = '*'
    ) -> fastworkflow.MCPToolResult:
        """
        MCP-compliant tool execution method.
        
        Args:
            workflow: FastWorkflow workflow
            tool_call: MCP tool call request
            workitem_path: The context in which to execute the command. If None, it must be in the arguments.
            
        Returns:
            MCPToolResult: MCP-compliant result format
        """
        try:
            context = tool_call.arguments.get('workitem_path', command_context)
            if not context:
                raise ValueError("Context ('workitem_path') must be provided for an MCP tool call.")

            # Convert MCP tool call to FastWorkflow Action using helper method
            action = fastworkflow.Action(
                command_name=tool_call.name,
                command=tool_call.arguments.get('command', ''),
                parameters=dict(tool_call.arguments.items()),
            )

            # Execute using existing perform_action method
            command_output = cls.perform_action(workflow, action)

            # Convert to MCP format
            return command_output.to_mcp_result()

        except Exception as e:
            # Return error in MCP format
            return fastworkflow.MCPToolResult(
                content=[fastworkflow.MCPContent(type="text", text=f"Error: {str(e)}")],
                isError=True
            )

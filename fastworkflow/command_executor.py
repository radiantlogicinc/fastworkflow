import fastworkflow
from fastworkflow.command_interfaces import CommandExecutorInterface

from fastworkflow import Action, CommandOutput, WorkflowSession
from fastworkflow import ModuleType
from fastworkflow.utils.signatures import InputForParamExtraction
from pathlib import Path
from fastworkflow.command_router import CommandRouter


# ------------------------------------------------------------------
# Module-level delegation configuration and exceptions
# ------------------------------------------------------------------

MAX_DELEGATION_DEPTH: int = 10  # Safety limit for delegation hops


class CommandNotFoundError(Exception):
    """Raised when a command cannot be resolved in any accessible context."""


class CommandExecutor(CommandExecutorInterface):
    def invoke_command(
        self,
        workflow_session: 'fastworkflow.WorkflowSession',
        command: str,
    ) -> fastworkflow.CommandOutput:
        if not command:
            raise ValueError("Command cannot be none or empty.")

        command_output = self._invoke_command_metadata_extraction_workflow(
            workflow_session,
            command
        )
        if command_output.command_handled or not command_output.success:
            return command_output

        command_name = command_output.command_responses[0].artifacts["command_name"]
        input_obj = command_output.command_responses[0].artifacts["cmd_parameters"]

        snapshot = workflow_session.session.workflow_snapshot
        workflow_folderpath = snapshot.workflow_folderpath
        command_routing_definition = fastworkflow.CommandRoutingRegistry.get_definition(
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
            return response_generation_object(workflow_session.session, command, input_obj)
        else:
            return response_generation_object(workflow_session.session, command)

    def perform_action(
        self,
        session: fastworkflow.Session,
        action: fastworkflow.Action,
    ) -> fastworkflow.CommandOutput:  # sourcery skip: extract-method
        session.command_context_for_response_generation = \
            session.current_command_context

        workflow_folderpath = session.workflow_snapshot.workflow_folderpath
        command_routing_definition = fastworkflow.CommandRoutingRegistry.get_definition(workflow_folderpath)

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
            return response_generation_object(session, action.command)

        if action.parameters:
            input_obj = command_parameters_class(**action.parameters)

            input_for_param_extraction = InputForParamExtraction(command=action.command)
            is_valid, error_msg, _ = input_for_param_extraction.validate_parameters(
                session.workflow_snapshot, action.command_name, input_obj
            )
            if not is_valid:
                raise ValueError(f"Invalid action parameters for command '{action.command_name}'\n{error_msg}")
        else:
            input_obj = command_parameters_class()

        return response_generation_object(session, action.command, input_obj)

    # MCP-compliant methods
    def perform_mcp_tool_call(
        self,
        session: fastworkflow.Session,
        tool_call: fastworkflow.MCPToolCall,
        command_context: str = '*'
    ) -> fastworkflow.MCPToolResult:
        """
        MCP-compliant tool execution method.
        
        Args:
            session: FastWorkflow session
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
                workitem_path=context,
                command_name=tool_call.name,
                command=tool_call.arguments.get('command', ''),
                parameters=dict(tool_call.arguments.items()),
            )

            # Execute using existing perform_action method
            command_output = self.perform_action(session, action)

            # Convert to MCP format
            return command_output.to_mcp_result()

        except Exception as e:
            # Return error in MCP format
            return fastworkflow.MCPToolResult(
                content=[fastworkflow.MCPContent(type="text", text=f"Error: {str(e)}")],
                isError=True
            )

    def _invoke_command_metadata_extraction_workflow(
        self,
        workflow_session: fastworkflow.WorkflowSession,
        command: str,
    ) -> CommandOutput:
        startup_action = Action(
            workitem_path="/command_metadata_extraction",
            command_name="wildcard",
            command=command,
        )

        workflow_context = {
            "subject_session": workflow_session.session
        }

        # Use the utility function to get the internal workflow path
        workflow_type = "command_metadata_extraction"
        cme_workflow_folderpath = fastworkflow.get_internal_workflow_path(workflow_type)

        child_session_id_str = f"{workflow_session.session.id}{workflow_type}"
        child_session_id = fastworkflow.get_session_id(child_session_id_str)

        # Attempt to load an existing session.
        existing_session = fastworkflow.Session.get_session(child_session_id)

        # Use the existing session **only** if it is not yet complete.
        if existing_session and not existing_session.workflow_snapshot.is_complete:
            session = existing_session
        else:
            # Either no previous session exists OR the previous session has
            # finished.  Create a fresh session (which will implicitly
            # overwrite the prior one if it was complete).
            session = fastworkflow.Session.create(
                cme_workflow_folderpath,
                session_id_str=None,
                parent_session_id=workflow_session.session.id,
                context=workflow_context,
            )

        command_executor = CommandExecutor()
        return command_executor.perform_action(session, startup_action)

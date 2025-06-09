import fastworkflow
from fastworkflow.command_interfaces import CommandExecutorInterface

from fastworkflow import Action, CommandOutput
from fastworkflow.command_routing_definition import ModuleType
from fastworkflow.utils.signatures import InputForParamExtraction


class CommandExecutor(CommandExecutorInterface):
    def invoke_command(
        self,
        workflow_session: 'fastworkflow.WorkflowSession',
        command: str,
    ) -> fastworkflow.CommandOutput:
        if not command:
            raise ValueError("Command cannot be none or empty.")

        cme_workflow_session_id, command_output = self._extract_command_metadata(
            workflow_session,
            command
        )
        current_session_id = fastworkflow.WorkflowSession.get_active_session_id()
        if current_session_id == cme_workflow_session_id:
            return command_output
        if command_output.command_aborted:
            return command_output

        command_name = command_output.command_responses[0].artifacts["command_name"]
        input_obj = command_output.command_responses[0].artifacts["cmd_parameters"]

        active_workitem_type = workflow_session.session.workflow_snapshot.active_workitem.path
        workflow_folderpath = workflow_session.session.workflow_snapshot.workflow.workflow_folderpath
        command_routing_definition = fastworkflow.CommandRoutingRegistry.get_definition(workflow_folderpath)
        if response_generation_object := (
            command_routing_definition.get_command_class_object(
                active_workitem_type,
                command_name,
                ModuleType.RESPONSE_GENERATION_INFERENCE,
            )
        ):
            return (
                response_generation_object(
                    workflow_session.session, command, input_obj
                )
                if input_obj
                else response_generation_object(workflow_session.session, command)
            )
        else:
            raise ValueError(
                f"Response generation object not found for workitem type '{active_workitem_type}' and command name '{command_name}'"
            )

    def perform_action(
        self,
        session: fastworkflow.Session,
        action: fastworkflow.Action,
    ) -> fastworkflow.CommandOutput:  # sourcery skip: extract-method
        workflow_folderpath = session.workflow_snapshot.workflow.workflow_folderpath
        command_routing_definition = fastworkflow.CommandRoutingRegistry.get_definition(workflow_folderpath)

        response_generation_object = (
            command_routing_definition.get_command_class_object(
                action.workitem_path,
                action.command_name,
                ModuleType.RESPONSE_GENERATION_INFERENCE,
            )
        )
        if not response_generation_object:
            raise ValueError(
                f"Response generation object not found for workitem type '{action.workitem_path}' and command name '{action.command_name}'"
            )

        command_parameters_class = (
            command_routing_definition.get_command_class(
                action.workitem_path, action.command_name, ModuleType.COMMAND_PARAMETERS_CLASS
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
        workitem_path: str = "/"
    ) -> fastworkflow.MCPToolResult:
        """
        MCP-compliant tool execution method.
        
        Args:
            session: FastWorkflow session
            tool_call: MCP tool call request
            workitem_path: Default workitem path if not specified in arguments
            
        Returns:
            MCPToolResult: MCP-compliant result format
        """
        try:
            # Convert MCP tool call to FastWorkflow Action using helper method
            action = fastworkflow.Action(
                workitem_path=tool_call.arguments.get('workitem_path', workitem_path),
                command_name=tool_call.name,
                command=tool_call.arguments.get('command', ''),
                parameters={k: v for k, v in tool_call.arguments.items() if k != 'command'}
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

    def _extract_command_metadata(
        self,
        workflow_session: fastworkflow.WorkflowSession,
        command: str,
    ) -> tuple[int, CommandOutput]:
        startup_action = Action(
            workitem_path="/command_metadata_extraction",
            command_name="wildcard",
            command=command,
        )

        # if we are already in the command_metadata_extraction workflow, we can just perform the action
        if workflow_session.session.workflow_snapshot.workflow.path == "/command_metadata_extraction":
            command_executor = CommandExecutor()
            command_output = command_executor.perform_action(workflow_session.session, startup_action)
            if len(command_output.command_responses) > 1:
                raise ValueError("Multiple command responses returned from command_metadata_extraction workflow")    
            return (workflow_session.session.id, command_output)    

        # Use the utility function to get the internal workflow path
        command_metadata_extraction_workflow_folderpath = fastworkflow.get_internal_workflow_path("command_metadata_extraction")

        context = {
            "subject_workflow_snapshot": workflow_session.session.workflow_snapshot
        }

        cme_workflow_session = fastworkflow.WorkflowSession(
            workflow_session.command_executor,
            command_metadata_extraction_workflow_folderpath,
            parent_session_id=workflow_session.session.id, 
            context=context,
            startup_action=startup_action, 
            user_message_queue=workflow_session.user_message_queue,
            command_output_queue=workflow_session.command_output_queue,
        )

        command_output = cme_workflow_session.start()
        return (cme_workflow_session.session.id, command_output)
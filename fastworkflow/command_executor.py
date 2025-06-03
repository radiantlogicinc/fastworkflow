import fastworkflow
from fastworkflow.command_interfaces import CommandExecutorInterface
from fastworkflow.command_routing_definition import ModuleType as CommandModuleType
from fastworkflow.utils.signatures import InputForParamExtraction

# import torch
# from transformers import BitsAndBytesConfig
# from outlines.generate import json as outlines_generate_json
# from outlines.models import Transformers as outlines_models_Transformers, transformers as outlines_models_transformers


class CommandExecutor(CommandExecutorInterface):
    def __init__(self):
        pass

        # self._model: outlines_models_Transformers = outlines_models_transformers(
        #     "microsoft/Phi-3.5-mini-instruct",
        #     model_kwargs={
        #         'device_map': "cuda",
        #         'torch_dtype': "auto",
        #         'trust_remote_code': True,
        #     }
        # model_kwargs={
        #     'quantization_config':BitsAndBytesConfig(
        #         # Load the model in 4-bit mode
        #         load_in_4bit=True,
        #         bnb_4bit_use_double_quant=True,
        #         bnb_4bit_quant_type="nf4",
        #         bnb_4bit_compute_dtype=torch.bfloat16,
        #     )
        # }
        # )

    def invoke_command(
        self,
        workflow_session: 'fastworkflow.WorkflowSession',
        command_name: str,
        command: str,
    ) -> fastworkflow.CommandOutput:
        if not command_name:
            raise ValueError("Command name cannot be None.")

        workflow_folderpath = workflow_session.session.workflow_snapshot.workflow.workflow_folderpath
        command_routing_definition = fastworkflow.CommandRoutingRegistry.get_definition(workflow_folderpath)

        active_workitem_type = workflow_session.session.workflow_snapshot.active_workitem.path
        response_generation_object = (
            command_routing_definition.get_command_class_object(
                active_workitem_type,
                command_name,
                CommandModuleType.RESPONSE_GENERATION_INFERENCE,
            )
        )
        if not response_generation_object:
            raise ValueError(
                f"Response generation object not found for workitem type '{active_workitem_type}' and command name '{command_name}'"
            )

        input_obj = None
        input_for_param_extraction_class = (
            command_routing_definition.get_command_class(
                active_workitem_type,
                command_name,
                CommandModuleType.INPUT_FOR_PARAM_EXTRACTION_CLASS,
            )
        )
        command_parameters_class = (
            command_routing_definition.get_command_class(
                active_workitem_type, command_name, CommandModuleType.COMMAND_PARAMETERS_CLASS
            )
        )
        if input_for_param_extraction_class and command_parameters_class:
            # lazy import to avoid circular dependency
            from fastworkflow.parameter_extraction import extract_command_parameters

            command_output = extract_command_parameters(
                workflow_session,
                command_name,
                command
            )
            if command_output.command_aborted:
                return command_output

            input_obj = command_output.command_responses[0].artifacts["cmd_parameters"]
            command_name_from_param_extraction = command_output.command_responses[0].artifacts["command_name"]
            if command_name_from_param_extraction != command_name:
                command_name = command_name_from_param_extraction
                response_generation_object = (
                    command_routing_definition.get_command_class_object(
                        active_workitem_type,
                        command_name,
                        CommandModuleType.RESPONSE_GENERATION_INFERENCE,
                    )
                )
                if not response_generation_object:
                    raise ValueError(
                        f"Response generation object not found for workitem type '{active_workitem_type}' and command name '{command_name}'"
                    )

        if input_obj:
            return response_generation_object(workflow_session.session, command, input_obj)
        else:
            return response_generation_object(workflow_session.session, command)

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
                CommandModuleType.RESPONSE_GENERATION_INFERENCE,
            )
        )
        if not response_generation_object:
            raise ValueError(
                f"Response generation object not found for workitem type '{action.workitem_path}' and command name '{action.command_name}'"
            )

        command_parameters_class = (
            command_routing_definition.get_command_class(
                action.workitem_path, action.command_name, CommandModuleType.COMMAND_PARAMETERS_CLASS
            )
        )
        if not command_parameters_class:
            return response_generation_object(session, action.command)

        if action.parameters:
            input_obj = command_parameters_class(**action.parameters)

            previous_context = session.workflow_snapshot.context
            session.workflow_snapshot.context =  {
                "subject_command_name": action.command_name,
                "param_extraction_sws": session.workflow_snapshot
            }
            input_for_param_extraction = InputForParamExtraction.create(session.workflow_snapshot, action.command)
            session.workflow_snapshot.context = previous_context

            is_valid, error_msg, _ = input_for_param_extraction.validate_parameters(session.workflow_snapshot, input_obj)
            if not is_valid:
                raise ValueError(f"Invalid action parameters: {error_msg}")
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
            action = self.action_from_mcp_tool_call(tool_call, workitem_path)
            
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

    @staticmethod
    def action_from_mcp_tool_call(
        tool_call: fastworkflow.MCPToolCall,
        default_workitem_path: str = "/"
    ) -> fastworkflow.Action:
        """
        Convert MCP tool call to FastWorkflow Action.
        
        Args:
            tool_call: MCP tool call request
            default_workitem_path: Default workitem path
            
        Returns:
            Action: FastWorkflow action object
        """
        return fastworkflow.Action(
            workitem_path=tool_call.arguments.get('workitem_path', default_workitem_path),
            command_name=tool_call.name,
            command=tool_call.arguments.get('command', ''),
            parameters={k: v for k, v in tool_call.arguments.items() if k != 'command'}
        )

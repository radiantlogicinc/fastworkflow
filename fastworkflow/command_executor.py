import fastworkflow
from fastworkflow.command_interfaces import CommandExecutorInterface
from fastworkflow.command_routing_definition import ModuleType as CommandModuleType

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

        active_workitem_type = workflow_session.session.workflow_snapshot.active_workitem.type
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

        if input_obj:
            return response_generation_object(workflow_session.session, command, input_obj)
        else:
            return response_generation_object(workflow_session.session, command)

    def perform_action(
        self,
        session: fastworkflow.Session,
        action: fastworkflow.Action,
    ) -> fastworkflow.CommandOutput:
        workflow_folderpath = session.workflow_snapshot.workflow.workflow_folderpath
        command_routing_definition = fastworkflow.CommandRoutingRegistry.get_definition(workflow_folderpath)

        response_generation_object = (
            command_routing_definition.get_command_class_object(
                action.workitem_type,
                action.command_name,
                CommandModuleType.RESPONSE_GENERATION_INFERENCE,
            )
        )
        if not response_generation_object:
            raise ValueError(
                f"Response generation object not found for workitem type '{action.workitem_type}' and command name '{action.command_name}'"
            )

        command_parameters_class = (
            command_routing_definition.get_command_class(
                action.workitem_type, action.command_name, CommandModuleType.COMMAND_PARAMETERS_CLASS
            )
        )
        if command_parameters_class:
            if action.parameters:
                input_obj = command_parameters_class(**action.parameters)

                input_for_param_extraction_class = (
                    command_routing_definition.get_command_class(
                        action.workitem_type,
                        action.command_name,
                        CommandModuleType.INPUT_FOR_PARAM_EXTRACTION_CLASS,
                    )
                )
                is_valid, error_msg = input_for_param_extraction_class.validate_parameters(session.workflow_snapshot, input_obj)
                if not is_valid:
                    raise ValueError(f"Invalid action parameters: {error_msg}")
            else:
                input_obj = command_parameters_class()

            command_output = response_generation_object(session, action.command, input_obj)
        else:
            command_output = response_generation_object(session, action.command)

        return command_output

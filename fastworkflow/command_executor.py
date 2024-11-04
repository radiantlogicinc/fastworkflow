from typing import Optional

from pydantic import BaseModel

from fastworkflow.command_routing_definition import ModuleType as CommandModuleType
from fastworkflow.parameter_extraction import extract_command_parameters
from fastworkflow.session import Session

# import torch
# from transformers import BitsAndBytesConfig
# from outlines.generate import json as outlines_generate_json
# from outlines.models import Transformers as outlines_models_Transformers, transformers as outlines_models_transformers


class CommandOutput(BaseModel):
    success: bool = True
    response: str
    payload: Optional[dict] = None


class CommandExecutor:
    def __init__(self, session: Session):
        self._session = session

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

    @property
    def session(self):
        return self._session

    def invoke_command(
        self,
        workitem_type: str,
        command_name: str,
        command: str,
        payload: Optional[dict] = None,
    ) -> CommandOutput:
        if not workitem_type:
            raise ValueError("Workitem type cannot be None.")
        if not command_name:
            raise ValueError("Command name cannot be None.")
        if not command:
            raise ValueError("Command cannot be None.")

        response_generation_object = (
            self._session.command_routing_definition.get_command_class_object(
                workitem_type,
                command_name,
                CommandModuleType.RESPONSE_GENERATION_INFERENCE,
            )
        )
        if not response_generation_object:
            raise ValueError(
                f"Response generation object not found for workitem type '{workitem_type}' and command name '{command_name}'"
            )

        abort_command = False
        input_obj = None
        input_for_param_extraction_class = (
            self._session.command_routing_definition.get_command_class(
                workitem_type,
                command_name,
                CommandModuleType.INPUT_FOR_PARAM_EXTRACTION_CLASS,
            )
        )
        command_parameters_class = (
            self._session.command_routing_definition.get_command_class(
                workitem_type, command_name, CommandModuleType.COMMAND_PARAMETERS_CLASS
            )
        )
        if input_for_param_extraction_class and command_parameters_class:
            input_for_parameter_extraction = input_for_param_extraction_class.create(
                session=self._session, command=command, payload=payload
            )
            abort_command, input_obj = extract_command_parameters(
                self._session,
                input_for_parameter_extraction,
                command_parameters_class,
                "parameter_extraction",
            )

        if abort_command:
            return CommandOutput(success=False, response="Command aborted")

        if input_obj:
            return response_generation_object(
                self._session, command, input_obj, payload
            )
        else:
            return response_generation_object(self._session, command, payload)

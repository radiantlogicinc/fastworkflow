from typing import Optional, Type

from pydantic import BaseModel

import dspy

import fastworkflow
from fastworkflow.command_routing_definition import ModuleType
from fastworkflow.utils.logging import logger

from fastworkflow.utils.pydantic_model_2_dspy_signature_class import (
    TypedPredictorSignature,
)


class OutputOfProcessCommand(BaseModel):
    parameter_is_valid: bool
    cmd_parameters: Optional[BaseModel] = None
    error_msg: Optional[str] = None

def process_command(
    session: fastworkflow.Session, command: str
) -> OutputOfProcessCommand:
    sws = session.workflow_snapshot.context["subject_workflow_snapshot"]

    subject_workflow_folderpath = sws.workflow.workflow_folderpath
    subject_command_routing_definition = fastworkflow.CommandRoutingRegistry.get_definition(subject_workflow_folderpath)

    active_workitem_type = sws.active_workitem.type
    subject_command_name = session.workflow_snapshot.context["subject_command_name"]
    input_for_param_extraction_class = subject_command_routing_definition.get_command_class(
        active_workitem_type, subject_command_name, ModuleType.INPUT_FOR_PARAM_EXTRACTION_CLASS)
    input_for_param_extraction = input_for_param_extraction_class.create(
        workflow_snapshot=sws, command=command
    )

    command_parameters_class = subject_command_routing_definition.get_command_class(
        active_workitem_type, subject_command_name, ModuleType.COMMAND_PARAMETERS_CLASS)
    input_obj = extract_command_parameters_from_input(
        input_for_param_extraction, command_parameters_class
    )

    is_valid, error_msg = input_for_param_extraction.validate_parameters(sws, input_obj)
    if not is_valid:
        error_msg += "Enter 'abort' if you want to abort the command."
        return OutputOfProcessCommand(parameter_is_valid=False, error_msg=error_msg)

    return OutputOfProcessCommand(parameter_is_valid=True, cmd_parameters=input_obj)


def extract_command_parameters_from_input(
    input_for_param_extraction: BaseModel,
    command_parameters_class: Type[BaseModel],
) -> BaseModel:
    try:
        dspy_signature_class = TypedPredictorSignature.create(
            input_for_param_extraction,
            command_parameters_class,
            prefix_instructions=input_for_param_extraction.__doc__,
        )

        DSPY_LM_MODEL = fastworkflow.get_env_var("DSPY_LM_MODEL")
        lm = dspy.LM(DSPY_LM_MODEL)
        with dspy.context(lm=lm):
            extract_cmd_params = dspy.TypedChainOfThought(dspy_signature_class)
            prediction = extract_cmd_params(**input_for_param_extraction.model_dump())
            command_parameters_obj = command_parameters_class(**prediction)
    except ValueError as e:
        logger.error(f"DSPy error extracting command parameters: {e}")
        command_parameters_obj = command_parameters_class()

    return command_parameters_obj

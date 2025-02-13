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

    active_workitem_type = sws.active_workitem.path
    subject_command_name = session.workflow_snapshot.context["subject_command_name"]
    command_parameters_class = subject_command_routing_definition.get_command_class(
        active_workitem_type, subject_command_name, ModuleType.COMMAND_PARAMETERS_CLASS)

    input_for_param_extraction_class = subject_command_routing_definition.get_command_class(
        active_workitem_type, subject_command_name, ModuleType.INPUT_FOR_PARAM_EXTRACTION_CLASS)
    input_for_param_extraction = input_for_param_extraction_class.create(
        sws, command
    )

    # if we are already in the parameter extraction workflow, no need to predict using DSPy
    # require parameters to be passed in comma delimited format
    if session.workflow_snapshot.context.get("in_param_extraction_workflow", None):
        param_values = [item.strip() for item in command.split(',')]
        fields = list(command_parameters_class.__fields__.keys())
        data = dict(zip(fields, param_values))
        input_obj = command_parameters_class(**data)
    else:
        input_obj = extract_command_parameters_from_input(
            input_for_param_extraction, command_parameters_class
        )

    is_valid, error_msg = input_for_param_extraction.validate_parameters(sws, input_obj)
    if not is_valid:
        session.workflow_snapshot.context["in_param_extraction_workflow"] = True
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

    # Clean up the values of all fields, replacing "\_" with "_"
    for field_name, value in command_parameters_obj.model_dump().items():
        if isinstance(value, str):
            cleaned_value = value.replace('\\_', '_')
            setattr(command_parameters_obj, field_name, cleaned_value)

    return command_parameters_obj

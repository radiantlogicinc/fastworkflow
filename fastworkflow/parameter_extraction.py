import os
from typing import Optional, Tuple, Type

import dspy
from pydantic import BaseModel

from fastworkflow.session import Session
from fastworkflow.utils.env import get_env_variable
from fastworkflow.utils.logging import logger
from fastworkflow.utils.pydantic_model_2_dspy_signature_class import (
    TypedPredictorSignature,
)


def extract_command_parameters(
    session: Session,
    input_for_param_extraction: BaseModel,
    command_parameters_class: Type[BaseModel],
    extraction_failure_workflow: Optional[str] = None,
) -> Tuple[bool, BaseModel]:
    try:
        dspy_signature_class = TypedPredictorSignature.create(
            input_for_param_extraction,
            command_parameters_class,
            prefix_instructions=input_for_param_extraction.__doc__,
        )

        DSPY_LM_MODEL = get_env_variable("DSPY_LM_MODEL")
        lm = dspy.LM(DSPY_LM_MODEL)
        with dspy.context(lm=lm):
            extract_cmd_params = dspy.TypedChainOfThought(dspy_signature_class)
            prediction = extract_cmd_params(**input_for_param_extraction.model_dump())
            command_parameters_obj = command_parameters_class(**prediction)
    except ValueError as e:
        logger.error(f"DSPy error extracting command parameters: {e}")
        command_parameters_obj = command_parameters_class()

    abort_command = False
    if not extraction_failure_workflow:
        return (abort_command, command_parameters_obj)

    is_valid, error_msg = input_for_param_extraction.validate_parameters(
        session, command_parameters_obj
    )
    if is_valid:
        return (abort_command, command_parameters_obj)

    # lazy import to avoid circular dependency
    from fastworkflow.start_workflow import start_workflow

    fastworkflow_folder = os.path.dirname(os.path.abspath(__file__))
    parameter_extraction_workflow_folderpath = os.path.join(
        fastworkflow_folder, "_workflows", extraction_failure_workflow
    )
    command_output = start_workflow(
        parameter_extraction_workflow_folderpath,
        startup_command="extract parameter",
        payload={
            "error_msg": error_msg,
            "session": session,
            "input_for_param_extraction_class": type(input_for_param_extraction),
            "command_parameters_class": command_parameters_class,
            "parameter_extraction_func": extract_command_parameters,
            "parameter_validation_func": input_for_param_extraction.validate_parameters,
        },
        keep_alive=False,
    )

    abort_command = (
        command_output.payload["abort_command"]
        if "abort_command" in command_output.payload
        else False
    )
    if abort_command:
        return (abort_command, None)

    command_parameters_obj = command_output.payload["cmd_parameters"]

    return (False, command_parameters_obj)

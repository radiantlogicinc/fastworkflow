import os
import random
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
    extraction_failure_workflow: Optional[str] = None
) -> Tuple[bool, BaseModel]:
    """
    This function is called when the command parameters are invalid.
    It extracts the command parameters from the command using DSPy.
    If the extraction fails, it starts the extraction failure workflow.
    Note: this function is called from the regular workflow as well as the extraction failure workflow.
    If it is called from the extraction failure workflow, the session is the source workflow session which is basically what we want.
    """
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

    session.parameter_extraction_info = {
        "error_msg": error_msg,
        "input_for_param_extraction_class": type(input_for_param_extraction),
        "command_parameters_class": command_parameters_class,
        "parameter_extraction_func": extract_command_parameters,
        "parameter_validation_func": input_for_param_extraction.validate_parameters,
    }

    wf_session = Session(-random.randint(1, 100000000), 
                         parameter_extraction_workflow_folderpath, 
                         session.env_file_path)
    
    command_output = start_workflow(
        wf_session,
        startup_command="extract parameter",
        caller_session=session,
        keep_alive=False,
    )

    session.parameter_extraction_info = None

    abort_command = (
        command_output.payload["abort_command"]
        if "abort_command" in command_output.payload
        else False
    )
    if abort_command:
        return (abort_command, None)

    command_parameters_obj = command_output.payload["cmd_parameters"]

    return (False, command_parameters_obj)

import os
import random
from typing import Optional, Tuple, Type

from pydantic import BaseModel

from fastworkflow.session import Session
from fastworkflow._commands.get_command_name.parameter_extraction import (
    signatures as pes,
)


def extract_commmand_name_from_command_parameters(
    caller_session: Session,
    input_for_param_extraction: pes.InputForParamExtraction,
    _: Type[BaseModel],
) -> Tuple[bool, BaseModel]:
    """
    This function is called when the command parameters are invalid.
    It extracts the command name from the command using the semantic router.
    Note: This function is called from the extraction failure workflow, so the source workflow folderpath and active workitem type are needed.
    """
    route_layer = caller_session.get_route_layer(caller_session.get_active_workitem().type)
    # Use semantic router to decipher the command name
    command_name = route_layer(input_for_param_extraction.command).name
    cmd_parameters = pes.CommandParameters(command_name=command_name)
    return (False, cmd_parameters)


def extract_command_name(
    session: Session,
    command: str,
    extraction_failure_workflow: Optional[str],
) -> Tuple[bool, str]:
    active_workitem_type = session.get_active_workitem().type
    route_layer = session.get_route_layer(active_workitem_type)
    # Use semantic router to decipher the command name
    command_name = route_layer(command).name
    if not command_name:
        # if extraction_failure_workflow is provided, default command is "NOT_FOUND"
        # otherwise, we assume we are in the extraction failure workflow and default command is "extract_parameters"
        command_name = (
            "NOT_FOUND" if extraction_failure_workflow else "extract_parameters"
        )

    abort_command = False
    cmd_parameters = pes.CommandParameters(command_name=command_name)
    input_for_param_extraction = pes.InputForParamExtraction.create(session, command)
    is_valid, error_msg = input_for_param_extraction.validate_parameters(
        session, cmd_parameters
    )
    if is_valid:
        return (abort_command, command_name)

    # lazy import to avoid circular dependency
    from fastworkflow.start_workflow import start_workflow

    fastworkflow_folder = os.path.dirname(os.path.abspath(__file__))
    parameter_extraction_workflow_folderpath = os.path.join(
        fastworkflow_folder, "_workflows", extraction_failure_workflow
    )

    session.parameter_extraction_info = {
        "error_msg": error_msg,
        "input_for_param_extraction_class": pes.InputForParamExtraction,
        "command_parameters_class": pes.CommandParameters,
        "parameter_extraction_func": extract_commmand_name_from_command_parameters,
        "parameter_validation_func": pes.InputForParamExtraction.validate_parameters,
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

    if len(command_output) > 1:
        raise ValueError("Multiple command responses returned from parameter extraction workflow")

    session.parameter_extraction_info = None

    abort_command = command_output[0].artifacts["abort_command"]
    if abort_command:
        return (abort_command, None)

    cmd_parameters = command_output[0].artifacts["cmd_parameters"]
    return (False, cmd_parameters.command_name)

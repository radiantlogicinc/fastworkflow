import os
from typing import Optional, Tuple, Type
from pydantic import BaseModel

from semantic_router import RouteLayer

from fastworkflow.session import Session
from fastworkflow._commands.get_command_name.parameter_extraction import signatures as pes


class ExtractCommandNameFromCommandParameters:
    def __init__(self, map_workitem_type_2_route_layer: dict[str, RouteLayer]):
        self.map_workitem_type_2_route_layer = map_workitem_type_2_route_layer

    def extract_commmand_name(
        self,
        session: Session,
        input_for_param_extraction: pes.InputForParamExtraction,
        _: Type[BaseModel]) -> Tuple[bool, BaseModel]:
        if not self.map_workitem_type_2_route_layer:
            raise ValueError("map_workitem_type_2_route_layer is not set")

        active_workitem_type = session.get_active_workitem().type
        route_layer = self.map_workitem_type_2_route_layer[active_workitem_type]
        # Use semantic router to decipher the command name
        command_name = route_layer(input_for_param_extraction.command).name
        cmd_parameters = pes.CommandParameters(command_name=command_name)
        return (False, cmd_parameters)


def extract_command_name(
    session: Session,
    map_workitem_type_2_route_layer: dict[str, RouteLayer],
    command: str,
    extraction_failure_workflow: Optional[str]) -> Tuple[bool, str]:
    active_workitem_type = session.get_active_workitem().type
    route_layer = map_workitem_type_2_route_layer[active_workitem_type]
    # Use semantic router to decipher the command name
    command_name = route_layer(command).name
    if not command_name:
        # if extraction_failure_workflow is provided, default command is "NOT_FOUND"
        # otherwise, we assume we are in the extraction failure workflow and default command is "extract_parameters"
        command_name = "NOT_FOUND" if extraction_failure_workflow else "extract_parameters"

    abort_command = False
    cmd_parameters = pes.CommandParameters(command_name=command_name)
    input_for_param_extraction = pes.InputForParamExtraction.create(session, command)
    is_valid, error_msg = input_for_param_extraction.validate_parameters(session, cmd_parameters)
    if is_valid:
        return (abort_command, command_name)

    # lazy import to avoid circular dependency
    from fastworkflow.start_workflow import start_workflow

    fastworkflow_folder = os.path.dirname(os.path.abspath(__file__))
    parameter_extraction_workflow_folderpath = os.path.join(
        fastworkflow_folder, '_workflows', extraction_failure_workflow)
    
    extract_command_name_from_command_parameters = ExtractCommandNameFromCommandParameters(
        map_workitem_type_2_route_layer)

    command_output = start_workflow(
        parameter_extraction_workflow_folderpath,
        startup_command=f"extract parameter",
        payload={
            "error_msg": error_msg,
            "session": session,
            "input_for_param_extraction_class": pes.InputForParamExtraction,
            "command_parameters_class": pes.CommandParameters,
            "parameter_extraction_func": extract_command_name_from_command_parameters.extract_commmand_name,
            "parameter_validation_func": pes.InputForParamExtraction.validate_parameters,
        },
        keep_alive=False)

    abort_command = command_output.payload["abort_command"]
    if abort_command:
        return (abort_command, None)

    cmd_parameters = command_output.payload["cmd_parameters"]
    return (False, cmd_parameters.command_name)

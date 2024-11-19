from typing import Optional

from pydantic import BaseModel

import fastworkflow
from fastworkflow.session import WorkflowSnapshot


class CommandParameters(BaseModel):
    command_name: Optional[str] = None

class OutputOfProcessCommand(BaseModel):
    command_name: Optional[str] = None
    error_msg: Optional[str] = None

def process_command(
    session: fastworkflow.Session, command: str
) -> OutputOfProcessCommand:
    sws = session.workflow_snapshot.context["subject_workflow_snapshot"]
    sws_workflow_folderpath = sws.workflow.workflow_folderpath
    sws_active_workitem_type = sws.active_workitem.type
    sws_route_layer = fastworkflow.RouteLayerRegistry.get_route_layer(sws_workflow_folderpath, sws_active_workitem_type)   

    current_workflow_folderpath = session.workflow_snapshot.workflow.workflow_folderpath
    current_active_workitem_type = session.workflow_snapshot.active_workitem.type
    current_route_layer = fastworkflow.RouteLayerRegistry.get_route_layer(current_workflow_folderpath, current_active_workitem_type)

    route_list = [sws_route_layer.routes, current_route_layer.routes]
    rl = fastworkflow.RouteLayerRegistry.build_route_layer_from_routelayers(route_list)

    command_name = rl(command).name

    command_parameters = CommandParameters(command_name=command_name)
    is_valid, error_msg = validate_command_name(sws, command_parameters)
    if not is_valid:
        return OutputOfProcessCommand(error_msg=error_msg)

    return OutputOfProcessCommand(command_name=command_parameters.command_name)   

def validate_command_name(
    sws: WorkflowSnapshot,
    command_parameters: CommandParameters
) -> tuple[bool, str]:
    valid_command_names = {'abort'}

    sws_workflow_folderpath = sws.workflow.workflow_folderpath
    sws_command_routing_definition = fastworkflow.CommandRoutingRegistry.get_definition(sws_workflow_folderpath)
    valid_command_names |= set(sws_command_routing_definition.get_command_names(
        sws.active_workitem.type
    ))

    if command_parameters.command_name in valid_command_names:
        return (True, None)

    if not command_parameters.command_name and "*" in valid_command_names:
        command_parameters.command_name = "*"
        return (True, None)

    command_list = "\n".join(valid_command_names)
    return (
        False,
        "The command is ambiguous. Use one of the intents below in wording your command:\n"
        f"{command_list}",
    )

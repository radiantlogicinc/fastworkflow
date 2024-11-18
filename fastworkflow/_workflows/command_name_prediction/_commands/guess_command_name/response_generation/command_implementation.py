from typing import Optional

from pydantic import BaseModel

from semantic_router import RouteLayer

import fastworkflow
from fastworkflow.session import WorkflowSnapshot


class OutputOfProcessCommand(BaseModel):
    command_name: Optional[str] = None
    error_msg: Optional[str] = None

def process_command(
    session: fastworkflow.Session, command: str
) -> OutputOfProcessCommand:
    sws = session.workflow_snapshot.context["subject_workflow_snapshot"]

    workflow_folderpath = sws.workflow.workflow_folderpath
    active_workitem_type = sws.get_active_workitem().type
    route_layer = fastworkflow.RouteLayerRegistry.get_route_layer(workflow_folderpath, active_workitem_type)

    command_name = guess_commmand_name_from_command(route_layer, command)

    is_valid, error_msg = validate_command_name(sws, command_name)
    if not is_valid:
        return OutputOfProcessCommand(error_msg=error_msg)

    return OutputOfProcessCommand(command_name=command_name)

def guess_commmand_name_from_command(
    route_layer: RouteLayer,
    command: str
) -> str:
    """
    This function is called when the command parameters are invalid.
    It extracts the command name from the command using the semantic router.
    Note: This function is called from the extraction failure workflow, so the source workflow folderpath and active workitem type are needed.
    """
    # Use semantic router to decipher the command name
    command_name = route_layer(command).name
    if not command_name:
        command_name = "NOT_FOUND"
    return command_name

def validate_command_name(
    workflow_snapshot: WorkflowSnapshot,
    command_name: str
) -> tuple[bool, str]:
    workflow_folderpath = workflow_snapshot.workflow.workflow_folderpath
    command_routing_definition = fastworkflow.CommandRoutingRegistry.get_definition(workflow_folderpath)
    valid_command_names = command_routing_definition.get_command_names(
        workflow_snapshot.get_active_workitem().type
    )
    if command_name in valid_command_names:
        return (True, None)

    valid_command_names.insert(0, "abort")
    command_list = "\n".join(valid_command_names)
    return (
        False,
        f"Invalid command name: {command_name}.\n"
        f"Valid command names are:\n"
        f"{command_list}",
    )

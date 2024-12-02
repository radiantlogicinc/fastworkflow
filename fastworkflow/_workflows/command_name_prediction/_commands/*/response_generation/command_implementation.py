from typing import Optional

from pydantic import BaseModel
from semantic_router.schema import RouteChoice

import fastworkflow
from fastworkflow.session import WorkflowSnapshot


class CommandParameters(BaseModel):
    command_name: Optional[str] = None

class OutputOfProcessCommand(BaseModel):
    command_name: Optional[str] = None
    command: Optional[str] = None
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

    valid_command_names = get_valid_command_names(sws)

    command_name = None

    # Check if the entire command is a valid command name
    normalized_command = command.replace(" ", "_").lower()
    for name in valid_command_names:
        if normalized_command == name.lower():
            command_name = name
            break

    if not command_name:
        if "@" in command:
            # Check if the command has a valid @command_name
            tentative_command_name = command.split("@")[1].split()[0]
            normalized_command_name = tentative_command_name.lower()
            for name in valid_command_names:
                if normalized_command_name == name.lower():
                    command_name = name
                    command = command.replace(f"@{tentative_command_name}", "").strip().replace("  ", " ")
                    break

    if not command_name:
        # Check if the command has a valid route
        route_choice_list = rl.retrieve_multiple_routes(command)
        if route_choice_list:
            if len(route_choice_list) > 1:
                route_choice_list = sorted(
                    route_choice_list,
                    key=lambda x: x.similarity_score,
                    reverse=True
                )   # get the top route choices sorted by similarity_score
                score_difference = abs(route_choice_list[0].similarity_score - route_choice_list[1].similarity_score)
                if score_difference < 0.09 and len(route_choice_list) <= 2:
                    error_msg = formulate_ambiguous_command_error_message(route_choice_list)
                    return OutputOfProcessCommand(error_msg=error_msg)
                else:
                    command_name = route_choice_list[0].name
            else:
                command_name = route_choice_list[0].name

    command_parameters = CommandParameters(command_name=command_name)
    is_valid, error_msg = validate_command_name(valid_command_names, command_parameters)
    if not is_valid:
        return OutputOfProcessCommand(error_msg=error_msg)

    return OutputOfProcessCommand(
        command_name=command_parameters.command_name,
        command=command
    )   

def get_valid_command_names(sws: WorkflowSnapshot) -> set[str]:
    valid_command_names = {'abort'}
    sws_workflow_folderpath = sws.workflow.workflow_folderpath
    sws_command_routing_definition = fastworkflow.CommandRoutingRegistry.get_definition(sws_workflow_folderpath)
    valid_command_names |= set(sws_command_routing_definition.get_command_names(
        sws.active_workitem.type
    ))
    return valid_command_names

def validate_command_name(
    valid_command_names: set[str],
    command_parameters: CommandParameters
) -> tuple[bool, str]:
    if command_parameters.command_name in valid_command_names:
        return (True, None)

    if not command_parameters.command_name and "*" in valid_command_names:
        command_parameters.command_name = "*"
        return (True, None)

    command_list = "\n".join(f"@{name}" for name in valid_command_names)
    return (
        False,
        "The command is ambiguous. Prefix your command with an appropriate tag from the list below:\n"
        f"{command_list}"
    )

def formulate_ambiguous_command_error_message(route_choice_list: list[RouteChoice]) -> str:
    command_list = (
        "\n".join([
            f"@{route_choice.name}" 
            for route_choice in route_choice_list
        ])
    )

    return (
        "The command is ambiguous. Prefix your command with an appropriate tag from the list below:\n"
        f"{command_list}"
    )

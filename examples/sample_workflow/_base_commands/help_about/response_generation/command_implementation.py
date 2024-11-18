from pydantic import BaseModel

import fastworkflow
from fastworkflow.session import Session

from ..parameter_extraction.signatures import CommandParameters


class CommandProcessorOutput(BaseModel):
    help_info: dict


def process_command(
    session: Session, input: CommandParameters
) -> CommandProcessorOutput:
    """
    Provides helpful information about this type of work-item.
    If the workitem_type is not provided, it provides information about the current work-item.

    :param input: The input parameters for the function.
    """

    workitem_type = input.workitem_type
    if not workitem_type:
        workitem = session.workflow_snapshot.get_active_workitem()
        workitem_type = workitem.type

    help_info = {"workitem_type": workitem_type, "allowable_child_types": {}}

    workflow_folderpath = session.workflow_snapshot.workflow.workflow_folderpath
    workflow_definition = fastworkflow.WorkflowRegistry.get_definition(workflow_folderpath)

    for (
        child_type,
        child_size_metadata,
    ) in workflow_definition.allowable_child_types[workitem_type].items():
        help_info["allowable_child_types"][child_type] = {
            "min_size": child_size_metadata.min,
            "max_size": child_size_metadata.max,
        }

    return CommandProcessorOutput(help_info=help_info)

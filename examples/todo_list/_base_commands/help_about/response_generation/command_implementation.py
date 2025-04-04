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
    If the workitem_path is not provided, it provides information about the current work-item.

    :param input: The input parameters for the function.
    """

    workitem_path = input.workitem_path
    if not workitem_path:
        workitem = session.workflow_snapshot.active_workitem
        workitem_path = workitem.path

    help_info = {"workitem_path": workitem_path, "allowable_child_types": {}}

    workflow_folderpath = session.workflow_snapshot.workflow.workflow_folderpath
    workflow_definition = fastworkflow.WorkflowRegistry.get_definition(workflow_folderpath)

    for (
        child_path,
        child_size_metadata,
    ) in workflow_definition.paths_2_allowable_child_paths_2_sizemetadata[workitem_path].items():
        help_info["allowable_child_types"][child_path] = {
            "min_size": child_size_metadata.min,
            "max_size": child_size_metadata.max,
        }

    return CommandProcessorOutput(help_info=help_info)

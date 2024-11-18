from pydantic import BaseModel

import fastworkflow
from fastworkflow.session import Session

from ...get_status.parameter_extraction.signatures import (
    CommandParameters as GetStatusCommandParameters,
)
from ...get_status.response_generation.command_implementation import (
    process_command as get_status,
)
from ..parameter_extraction.signatures import CommandParameters


class CommandProcessorOutput(BaseModel):
    target_workitem_found: bool
    status_of_target_workitem: str


def process_command(
    session: Session, input: CommandParameters
) -> CommandProcessorOutput:
    """
    Move to the work-item specified by the given path and optional id.

    :param input: The input parameters for the function.
    """
    workflow_folderpath = session.workflow_snapshot.workflow.workflow_folderpath
    workflow_definition = fastworkflow.WorkflowRegistry.get_definition(workflow_folderpath)

    input.workitem_path = "".join(input.workitem_path.split()).strip(" \"'")
    relative_to_root = False
    if input.workitem_path in [
        workitem_type for workitem_type in workflow_definition.types
    ]:
        relative_to_root = True
        input.workitem_path = f"//{input.workitem_path}"

    workitem = session.workflow_snapshot.workflow.find_workitem(
        input.workitem_path, input.workitem_id, relative_to_root
    )

    target_workitem_found = workitem is not None

    if target_workitem_found:
        active_workitem = workitem
        session.workflow_snapshot.set_active_workitem(active_workitem)
    else:
        active_workitem = session.workflow_snapshot.get_active_workitem()

    get_status_tool_output = get_status(
        session,
        GetStatusCommandParameters(
            workitem_path=active_workitem.path, workitem_id=active_workitem.id
        ),
    )
    return CommandProcessorOutput(
        target_workitem_found=target_workitem_found,
        status_of_target_workitem=get_status_tool_output.status,
    )


if __name__ == "__main__":
    # create a session id
    session_id = 1234
    session = Session(session_id, "shared/tests/lighthouse/workflows/accessreview")

    command_input = CommandParameters(workitem_path="leavers", workitem_id=None)
    output = process_command(session, command_input)
    print(output)

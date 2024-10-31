from typing import Optional
from pydantic import BaseModel

from fastworkflow.session import Session
from fastworkflow.workflow_definition import NodeType
from ..parameter_extraction.signatures import CommandParameters


class CommandProcessorOutput(BaseModel):
    status: str

def process_command(session: Session, input: CommandParameters, payload: Optional[dict] = None) -> CommandProcessorOutput:
    """
    get the review status of the entitlements in this workitem.

    :param input: The input parameters for the function.

    return the review status of the entitlements for the current workitem if workitem_path and workitem_id are not provided.
        if workitem_id is specified, the workitem_path must be specified.
    """
    workitem = session.get_active_workitem()
    if workitem.node_type == NodeType.Workflow:
        status=(
            f"workitem_path: {workitem.path}, workitem_id: {workitem.id}\n"
            f"started: {workitem.has_started}, Complete: {workitem.is_complete}"
        )
    else:
        status=(
            f"workitem_path: {workitem.path}, workitem_id: {workitem.id}\n"
            f"Complete: {workitem.is_complete}"
        )

    return CommandProcessorOutput(status=status)


if __name__ == "__main__":
    # create a session id
    session_id = 1234
    session = Session(session_id, "shared/tests/lighthouse/workflows/accessreview")

    tool_input = CommandParameters(workitem_path=None, workitem_id=None)
    tool_output = process_command(session, tool_input)
    print(tool_output)

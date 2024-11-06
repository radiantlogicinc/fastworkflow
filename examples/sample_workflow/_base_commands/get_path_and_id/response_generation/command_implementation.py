from typing import Optional

from pydantic import BaseModel

from fastworkflow.session import Session

from ..parameter_extraction.signatures import CommandParameters


class CommandProcessorOutput(BaseModel):
    workitem_path: Optional[str]
    workitem_id: Optional[str]


def process_command(
    session: Session, input: CommandParameters
) -> CommandProcessorOutput:
    """
    Get the path and id of the current or next work-item.

    :param input: The input parameters for the function.
    """
    active_workitem = session.get_active_workitem()
    if not input.for_next_workitem:
        return CommandProcessorOutput(
            workitem_path=active_workitem.path, workitem_id=active_workitem.id
        )

    next_workitem = active_workitem.next_workitem()
    if input.skip_completed:
        while next_workitem and next_workitem.is_complete:
            next_workitem = next_workitem.next_workitem()

    if not next_workitem:
        return CommandProcessorOutput(workitem_path=None, workitem_id=None)

    return CommandProcessorOutput(
        workitem_path=next_workitem.path, workitem_id=next_workitem.id
    )


if __name__ == "__main__":
    # create a session id
    session_id = 1234
    session = Session(session_id, "shared/tests/lighthouse/workflows/accessreview")

    tool_input = CommandParameters(for_next_workitem=False, skip_completed=True)
    tool_output = process_command(session, tool_input)
    print(tool_output)

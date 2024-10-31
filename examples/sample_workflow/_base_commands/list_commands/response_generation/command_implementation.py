from typing import Optional
from pydantic import BaseModel

from fastworkflow.session import Session


class CommandProcessorOutput(BaseModel):
    workitem_type: str
    commands: list[str]

def process_command(session: Session, payload: Optional[dict] = None) -> CommandProcessorOutput:
    """
        Provides helpful information about this type of work-item.
        If the workitem_type is not provided, it provides information about the current work-item.
        
        :param input: The input parameters for the function.
    """
    # Get the current workitem type
    current_workitem_type = session.get_active_workitem().type

    # Get the list of commands for the current workitem type
    commands = session.command_routing_definition.get_command_names(current_workitem_type)

    return CommandProcessorOutput(workitem_type=current_workitem_type, commands=commands)


if __name__ == "__main__":
    # create a session id
    session_id = 1234
    session = Session(session_id, "shared/tests/lighthouse/workflows/accessreview")

    tool_output = process_command(session, payload=None)
    print(tool_output)

from typing import Optional
from pydantic import BaseModel

from fastworkflow.session import Session
from ..parameter_extraction.signatures import CommandParameters
from ...get_status.parameter_extraction.signatures import CommandParameters as GetStatusCommandParameters
from ...get_status.response_generation.command_implementation import process_command as get_status

class CommandProcessorOutput(BaseModel):
    next_workitem_found: bool
    status_of_next_workitem: str

def process_command(session: Session, input: CommandParameters, payload: Optional[dict] = None) -> CommandProcessorOutput:
    """
    Move to the next work-item. Skip next work-items that are completed if the skip_completed flag is set to True.
    
    :param input: The input parameters for the function.
    """
    workitem = session.get_active_workitem()
    next_workitem = workitem.next_workitem(input.skip_completed)

    next_workitem_found = next_workitem is not None
    if next_workitem_found:
        active_workitem = next_workitem
        session.set_active_workitem(active_workitem)
    else:
        active_workitem = workitem

    get_status_tool_output = get_status(session, 
                                        GetStatusCommandParameters(
                                            workitem_path=active_workitem.path,
                                            workitem_id=active_workitem.id
                                        ))    
    return CommandProcessorOutput(
        next_workitem_found=next_workitem_found,
        status_of_next_workitem=get_status_tool_output.status
    )


if __name__ == "__main__":
    import os

    # create a session
    session_id = 1234

    workflow_folderpath = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    if not os.path.isdir(workflow_folderpath):
        raise ValueError(f"The provided folderpath '{workflow_folderpath}' is not valid. Please provide a valid directory.")

    session = Session(session_id, workflow_folderpath)

    command_input = CommandParameters(skip_completed=True)
    command_output = process_command(session, command_input)
    print(command_output)

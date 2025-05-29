from pydantic import BaseModel
from fastworkflow.session import Session
from ..parameter_extraction.signatures import CommandParameters
from ....retail_data import load_data
from ....tools.get_user_details import GetUserDetails

class CommandProcessorOutput(BaseModel):
    status: str

def process_command(
    session: Session, input: CommandParameters
) -> CommandProcessorOutput:
    """Process get user details command"""
    data = load_data()
    
    result = GetUserDetails.invoke(
        data=data,
        user_id=input.user_id
    )
    
    return CommandProcessorOutput(status=result)
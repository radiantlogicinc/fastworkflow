from pydantic import BaseModel
from fastworkflow.session import Session
from ..parameter_extraction.signatures import CommandParameters
from ....retail_data import load_data
from ....tools.find_user_id_by_email import FindUserIdByEmail

class CommandProcessorOutput(BaseModel):
    status: str

def process_command(
    session: Session, input: CommandParameters
) -> CommandProcessorOutput:
    """Process find user ID by email command"""
    # Load data
    data = load_data()
    
    # Call FindUserIdByEmail's invoke method
    result = FindUserIdByEmail.invoke(
        data=data,
        email=input.email
    )
    
    return CommandProcessorOutput(status=result)
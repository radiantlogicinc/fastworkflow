from pydantic import BaseModel
from fastworkflow.session import Session
from ..parameter_extraction.signatures import CommandParameters
from ....retail_data import load_data
from ....tools.find_user_id_by_name_zip import FindUserIdByNameZip

class CommandProcessorOutput(BaseModel):
    status: str

def process_command(
    session: Session, input: CommandParameters
) -> CommandProcessorOutput:
    """Process find user ID by name and zip command"""
    data = load_data()
    
    result = FindUserIdByNameZip.invoke(
        data=data,
        first_name=input.first_name,
        last_name=input.last_name,
        zip=input.zip
    )
    
    return CommandProcessorOutput(status=result)
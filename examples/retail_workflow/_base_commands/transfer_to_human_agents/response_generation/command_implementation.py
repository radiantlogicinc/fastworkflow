from pydantic import BaseModel
from fastworkflow.session import Session
from ..parameter_extraction.signatures import CommandParameters
from ....retail_data import load_data
from ....tools.transfer_to_human_agents import TransferToHumanAgents

class CommandProcessorOutput(BaseModel):
    status: str

def process_command(
    session: Session, input: CommandParameters
) -> CommandProcessorOutput:
    """Process transfer to human agents command"""
    data = load_data()
    
    result = TransferToHumanAgents.invoke(
        data=data,
        summary=input.summary
    )
    
    return CommandProcessorOutput(status=result)
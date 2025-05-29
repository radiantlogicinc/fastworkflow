from pydantic import BaseModel
from fastworkflow.session import Session
from ..parameter_extraction.signatures import CommandParameters
from ....retail_data import load_data
from ....tools.get_order_details import GetOrderDetails

class CommandProcessorOutput(BaseModel):
    status: str

def process_command(
    session: Session, input: CommandParameters
) -> CommandProcessorOutput:
    """Process get order details command"""
    data = load_data()
    
    result = GetOrderDetails.invoke(
        data=data,
        order_id=input.order_id
    )
    
    return CommandProcessorOutput(status=result)
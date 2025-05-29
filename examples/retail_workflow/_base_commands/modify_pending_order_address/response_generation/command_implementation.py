from pydantic import BaseModel
from fastworkflow.session import Session
from ..parameter_extraction.signatures import CommandParameters
from ....retail_data import load_data
from ....tools.modify_pending_order_address import ModifyPendingOrderAddress

class CommandProcessorOutput(BaseModel):
    status: str

def process_command(
    session: Session, input: CommandParameters
) -> CommandProcessorOutput:
    """Process modify pending order address command"""
    data = load_data()
    
    result = ModifyPendingOrderAddress.invoke(
        data=data,
        order_id=input.order_id,
        address1=input.address1,
        address2=input.address2,
        city=input.city,
        state=input.state,
        country=input.country,
        zip=input.zip
    )
    
    return CommandProcessorOutput(status=result)
from pydantic import BaseModel
from fastworkflow.session import Session
from ..parameter_extraction.signatures import CommandParameters
from ....retail_data import load_data
from ....tools.return_delivered_order_items import ReturnDeliveredOrderItems

class CommandProcessorOutput(BaseModel):
    status: str

def process_command(
    session: Session, input: CommandParameters
) -> CommandProcessorOutput:
    """Process return delivered order items command"""
    data = load_data()
    
    result = ReturnDeliveredOrderItems.invoke(
        data=data,
        order_id=input.order_id,
        item_ids=input.item_ids,
        payment_method_id=input.payment_method_id
    )
    
    return CommandProcessorOutput(status=result)
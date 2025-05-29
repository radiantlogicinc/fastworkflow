from pydantic import BaseModel
from fastworkflow.session import Session
from ..parameter_extraction.signatures import CommandParameters
from ....retail_data import load_data
from ....tools.exchange_delivered_order_items import ExchangeDeliveredOrderItems

class CommandProcessorOutput(BaseModel):
    status: str

def process_command(
    session: Session, input: CommandParameters
) -> CommandProcessorOutput:
    """Process exchange order command"""
    # Load data
    data = load_data()
    
    # Call ExchangeDeliveredOrderItems's invoke method
    result = ExchangeDeliveredOrderItems.invoke(
        data=data,
        order_id=input.order_id,
        item_ids=input.item_ids,
        new_item_ids=input.new_item_ids,
        payment_method_id=input.payment_method_id
    )
    
    return CommandProcessorOutput(status=result)

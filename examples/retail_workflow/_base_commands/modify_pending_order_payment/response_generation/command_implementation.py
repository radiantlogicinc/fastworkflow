from pydantic import BaseModel
from fastworkflow.session import Session
from ..parameter_extraction.signatures import CommandParameters
from ....retail_data import load_data
from ....tools.modify_pending_order_payment import ModifyPendingOrderPayment

class CommandProcessorOutput(BaseModel):
    status: str

def process_command(
    session: Session, input: CommandParameters
) -> CommandProcessorOutput:
    """Process modify pending order payment command"""
    data = load_data()
    
    result = ModifyPendingOrderPayment.invoke(
        data=data,
        order_id=input.order_id,
        payment_method_id=input.payment_method_id
    )
    
    return CommandProcessorOutput(status=result)
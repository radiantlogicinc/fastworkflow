from pydantic import BaseModel
from fastworkflow.session import Session
from ..parameter_extraction.signatures import CommandParameters
from ....retail_data import load_data
from ....tools.get_product_details import GetProductDetails

class CommandProcessorOutput(BaseModel):
    status: str

def process_command(
    session: Session, input: CommandParameters
) -> CommandProcessorOutput:
    """Process get product details command"""
    data = load_data()
    
    result = GetProductDetails.invoke(
        data=data,
        product_id=input.product_id
    )
    
    return CommandProcessorOutput(status=result)
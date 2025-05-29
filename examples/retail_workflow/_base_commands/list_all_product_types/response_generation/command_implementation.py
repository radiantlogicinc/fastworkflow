from pydantic import BaseModel
from fastworkflow.session import Session
from ....retail_data import load_data
from ....tools.list_all_product_types import ListAllProductTypes

class CommandProcessorOutput(BaseModel):
    status: str

def process_command(
    session: Session
) -> CommandProcessorOutput:
    """Process list all product types command"""
    data = load_data()
    
    result = ListAllProductTypes.invoke(data=data)
    
    return CommandProcessorOutput(status=result)
from typing import Annotated
from pydantic import BaseModel, Field, ConfigDict
from fastworkflow.session import WorkflowSnapshot
from fastworkflow.utils.signatures import InputForParamExtraction

class CommandParameters(BaseModel):
    order_id: Annotated[
        str,
        Field(
            default="NOT_FOUND",
            description="The order ID to modify (must start with #)",
            pattern=r"^(#W\d+|NOT_FOUND)$",
            examples=["#W0000000"]
        )
    ]
    
    payment_method_id: Annotated[
        str,
        Field(
            default="NOT_FOUND",
            description="Payment method ID to switch to",
            pattern=r"^((gift_card|credit_card)_\d+|NOT_FOUND)$",
            examples=["gift_card_0000000", "credit_card_0000000"]
        )
    ]

    model_config = ConfigDict(arbitrary_types_allowed=True)
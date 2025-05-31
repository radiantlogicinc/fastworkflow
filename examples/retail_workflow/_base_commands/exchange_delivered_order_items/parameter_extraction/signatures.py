from typing import Annotated, List
from pydantic import BaseModel, Field, ConfigDict
from fastworkflow.session import WorkflowSnapshot

class CommandParameters(BaseModel):
    """Returns status (whether exchange succeeded)"""
    order_id: Annotated[
        str,
        Field(
            default="NOT_FOUND",
            description="The order ID to exchange (must start with #)",
            pattern=r"^(#[\w\d]+|NOT_FOUND)$",
            examples=["#W0000000"]
        )
    ]
    
    item_ids: Annotated[
        List[str],
        Field(
            default_factory=list,
            description="The item IDs to be exchanged",
            examples=["1008292230"]
        )
    ]
    
    new_item_ids: Annotated[
        List[str],
        Field(
            default_factory=list,
            description="The new item IDs to exchange for",
            examples=["1008292230"]
        )
    ]
    
    payment_method_id: Annotated[
        str,
        Field(
            default="NOT_FOUND",
            description="Payment method ID for price difference. You can get this from order details->payment_history",
            examples=["gift_card_0000000", "credit_card_0000000"]
        )
    ]

    model_config = ConfigDict(arbitrary_types_allowed=True)

class InputForParamExtraction(BaseModel):
    command: str
    workflow_snapshot: WorkflowSnapshot
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def create(cls, workflow_snapshot: WorkflowSnapshot, command: str):
        return cls(command=command, workflow_snapshot=workflow_snapshot)
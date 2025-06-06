from typing import Annotated, List
from pydantic import BaseModel, Field, ConfigDict
from fastworkflow.session import WorkflowSnapshot

class CommandParameters(BaseModel):
    """Returns status (whether order items modification succeeded)"""
    order_id: Annotated[
        str,
        Field(
            default="NOT_FOUND",
            description="The order ID to modify (must start with #)",
            pattern=r"^(#W\d+|NOT_FOUND)$",
            examples=["#W0000000"]
        )
    ]
    
    item_ids: Annotated[
        List[str],
        Field(
            default_factory=list,
            description="List of item IDs to be modified",
            examples=["1008292230", "2468135790"]
        )
    ]
    
    new_item_ids: Annotated[
        List[str],
        Field(
            default_factory=list,
            description="List of new item IDs to replace with",
            examples=["1008292231", "2468135791"]
        )
    ]
    
    payment_method_id: Annotated[
        str,
        Field(
            default="NOT_FOUND",
            description="Payment method ID for price differences",
            pattern=r"^((gift_card|credit_card)_\d+|NOT_FOUND)$",
            examples=["gift_card_0000000", "credit_card_0000000"]
        )
    ]

    model_config = ConfigDict(arbitrary_types_allowed=True)

class InputForParamExtraction(BaseModel):
    command: str
    workflow_snapshot: WorkflowSnapshot
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def create(cls, workflow_snapshot: WorkflowSnapshot, _: str, command: str):
        return cls(command=command, workflow_snapshot=workflow_snapshot)
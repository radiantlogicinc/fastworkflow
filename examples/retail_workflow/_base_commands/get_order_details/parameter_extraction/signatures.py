from typing import Annotated
from pydantic import BaseModel, Field, ConfigDict
from fastworkflow.session import WorkflowSnapshot

class CommandParameters(BaseModel):
    """Returns details of items ordered, order delivery address, fulfillments, status, and payment history"""
    order_id: Annotated[
        str,
        Field(
            default="NOT_FOUND",
            description="The order ID to get details for (must start with #). You can get order id's from get_user_details, given a user id",
            pattern=r"^(#[\w\d]+|NOT_FOUND)$",
            examples=["#W0000000"]
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
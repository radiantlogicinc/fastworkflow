from typing import Annotated
from pydantic import BaseModel, Field, ConfigDict
from fastworkflow.session import WorkflowSnapshot

class CommandParameters(BaseModel):
    """Returns product name and details of a list variants including variant item_id, variant attributes such as color, size, material and style, whether it is available and the price"""
    product_id: Annotated[
        str,
        Field(
            default="NOT_FOUND",
            description="The product ID (numeric string)",
            pattern=r"^(\d{10}|NOT_FOUND)$",
            examples=["6086499569"]
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
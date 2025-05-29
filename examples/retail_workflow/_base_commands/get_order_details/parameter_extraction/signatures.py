from typing import Annotated
from pydantic import BaseModel, Field, ConfigDict
from fastworkflow.session import WorkflowSnapshot

class CommandParameters(BaseModel):
    order_id: Annotated[
        str,
        Field(
            default="NOT_FOUND",
            description="The order ID to get details for (must start with #)",
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
    def create(cls, workflow_snapshot: WorkflowSnapshot, command: str):
        return cls(command=command, workflow_snapshot=workflow_snapshot)
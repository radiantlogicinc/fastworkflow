from typing import Annotated
from pydantic import BaseModel, Field, ConfigDict
from fastworkflow.session import WorkflowSnapshot

class CommandParameters(BaseModel):
    "Returns user's first and last name, address, email, payment methods, and all the order id's"
    user_id: Annotated[
        str,
        Field(
            default="NOT_FOUND",
            description="The user ID to get details for",
            pattern=r"^([a-z]+_[a-z]+_\d+|NOT_FOUND)$",
            examples=["sara_doe_496"]
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
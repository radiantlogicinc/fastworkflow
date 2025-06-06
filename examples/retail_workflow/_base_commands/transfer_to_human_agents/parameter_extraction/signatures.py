from typing import Annotated
from pydantic import BaseModel, Field, ConfigDict
from fastworkflow.session import WorkflowSnapshot

class CommandParameters(BaseModel):
    """Returns status (whether transfer succeeded)"""
    summary: Annotated[
        str,
        Field(
            default="NOT_FOUND",
            description="A summary of the user's issue",
            examples=["Customer needs help with complex return process"]
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
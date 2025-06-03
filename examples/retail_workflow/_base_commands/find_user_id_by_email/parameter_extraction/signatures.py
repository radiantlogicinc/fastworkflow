from typing import Annotated
from pydantic import BaseModel, Field, ConfigDict
from fastworkflow.session import WorkflowSnapshot

class CommandParameters(BaseModel):
    """Returns the user_id"""
    email: Annotated[
        str,
        Field(
            default="NOT_FOUND",
            description="The email address to search for. If email is not available, you can use find_user_id_by_name_zip",
            pattern=r"^(NOT_FOUND|[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})$",
            examples=["user@example.com"]
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
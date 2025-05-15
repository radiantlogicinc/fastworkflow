from typing import Tuple

from pydantic import BaseModel

from fastworkflow.session import WorkflowSnapshot


class CommandParameters(BaseModel):
    for_next_workitem: bool = False
    skip_completed: bool = True


class InputForParamExtraction(BaseModel):
    """Extract the command parameters from the command"""

    command: str

    @classmethod
    def create(cls, workflow_snapshot: WorkflowSnapshot, command: str):
        return cls(
            command=command,
        )

    class Config:
        arbitrary_types_allowed = True

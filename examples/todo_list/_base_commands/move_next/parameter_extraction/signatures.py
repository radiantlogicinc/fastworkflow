from typing import Tuple

from pydantic import BaseModel

from fastworkflow.session import WorkflowSnapshot


class CommandParameters(BaseModel):
    skip_completed: bool = True


class InputForParamExtraction(BaseModel):
    """Extract the command parameters from the command"""

    command: str

    @classmethod
    def create(cls, workflow_snapshot: WorkflowSnapshot, _: str, command: str):
        return cls(
            command=command,
        )


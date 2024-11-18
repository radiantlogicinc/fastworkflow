from typing import Tuple

from pydantic import BaseModel, Field

import fastworkflow
from fastworkflow.session import WorkflowSnapshot


class CommandParameters(BaseModel):
    command_name: str = Field(default="NOT_FOUND", description="The command name")


class InputForParamExtraction(BaseModel):
    """Extract the command name from the command."""

    command: str

    @classmethod
    def create(cls, workflow_snapshot: WorkflowSnapshot, command: str):
        return cls(
            command=command,
        )

    @classmethod

    class Config:
        arbitrary_types_allowed = True

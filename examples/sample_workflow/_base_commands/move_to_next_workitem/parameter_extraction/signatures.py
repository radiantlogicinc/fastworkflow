from typing import Tuple

from pydantic import BaseModel

from fastworkflow.session import WorkflowSnapshot


class CommandParameters(BaseModel):
    skip_completed: bool = True


class InputForParamExtraction(BaseModel):
    """Extract the command parameters from the command"""

    command: str

    @classmethod
    def create(cls, workflow_snapshot: WorkflowSnapshot, command: str):
        return cls(
            command=command,
        )

    @classmethod
    def validate_parameters(
        cls, workflow_snapshot: WorkflowSnapshot, cmd_parameters: CommandParameters
    ) -> Tuple[bool, str]:
        """Nothing to validate"""
        return (True, None)

    class Config:
        arbitrary_types_allowed = True

from typing import Optional, Tuple
from pydantic import BaseModel

from fastworkflow.session import Session

class CommandParameters(BaseModel):
    for_next_workitem: bool = False
    skip_completed: bool = True

class InputForParamExtraction(BaseModel):
    """Extract the command parameters from the command"""
    command: str

    @classmethod
    def create(cls, session: Session, command: str, payload: Optional[dict] = None):
        return cls(
            command=command,
        )

    @classmethod
    def validate_parameters(cls, session: Session, cmd_parameters: CommandParameters) -> Tuple[bool, str]:
        """Nothing to validate"""
        return (True, None)

    class Config:
        arbitrary_types_allowed = True

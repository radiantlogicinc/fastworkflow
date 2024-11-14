from typing import Tuple

from pydantic import BaseModel, Field

from fastworkflow.session import Session


class CommandParameters(BaseModel):
    command_name: str = Field(default="NOT_FOUND", description="The command name")


class InputForParamExtraction(BaseModel):
    """Extract the command name from the command."""

    command: str

    @classmethod
    def create(cls, session: Session, command: str):
        return cls(
            command=command,
        )

    @classmethod
    def validate_parameters(
        cls, session: Session, cmd_parameters: CommandParameters
    ) -> Tuple[bool, str]:
        """
        Check if the parameters are valid in the current context.
        Parameter is a single field pydantic model.
        Return a tuple with a boolean indicating success or failure.
        And a string with a message indicating the error and suggested fixes.
        """
        active_workitem_type = session.get_active_workitem().type
        valid_command_names = session.command_routing_definition.get_command_names(
            active_workitem_type
        )
        if cmd_parameters.command_name in valid_command_names:
            return (True, None)

        command_list = "\n".join(valid_command_names)
        return (
            False,
            f"Invalid command name: {cmd_parameters.command_name}.\n"
            f"Valid command names are:\n"
            f"{command_list}",
        )

    class Config:
        arbitrary_types_allowed = True

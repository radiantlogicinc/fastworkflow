from typing import Optional

from fastworkflow.command_executor import CommandOutput
from fastworkflow.session import Session

from .command_implementation import process_command


class ResponseGenerator:
    def __call__(
        self, session: Session, command: str
    ) -> CommandOutput:
        output = process_command(session, payload)

        # Format the list of commands
        command_list = "\n".join([f"- {cmd}" for cmd in output.commands])

        # Create the response
        response = (
            f"Available commands for this task ({output.workitem_type}) are:\n"
            f"{command_list}\n"
            f"Enter a command that falls within the scope of these tools."
        )

        return CommandOutput(success=True, response=response)

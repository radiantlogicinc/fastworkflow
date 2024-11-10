from typing import Optional

from fastworkflow.command_executor import CommandResponse
from fastworkflow.session import Session

from .command_implementation import process_command


class ResponseGenerator:
    def __call__(
        self, session: Session, command: str
    ) -> list[CommandResponse]:
        output = process_command(session)

        # Format the list of commands
        utterance_list = "\n".join([f"- {cmd}" for cmd in output.utterances])

        # Create the response
        response = (
            f"Available utterances for this task ({output.workitem_type}) are:\n"
            f"{utterance_list}\n"
            f"Your chat message must fall within the scope of these utterances."
        )

        return [
            CommandResponse(success=True, response=response)
        ]

from fastworkflow import CommandOutput, CommandResponse
from fastworkflow.session import Session

from .command_implementation import process_command


class ResponseGenerator:
    def __call__(
        self, session: Session, command: str
    ) -> CommandOutput:
        output = process_command(session)


        # Create the response
        response = (
            f"Succesfully loaded {output.num_of_todos} todos"
        )

        return CommandOutput(
            command_responses=[
                CommandResponse(response=response)
            ]
        )

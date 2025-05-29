from fastworkflow import CommandOutput, CommandResponse
from fastworkflow.session import Session

from ..parameter_extraction.signatures import CommandParameters
from .command_implementation import process_command


class ResponseGenerator:
    def __call__(
        self,
        session: Session,
        command: str,
        command_parameters: CommandParameters
    ) -> CommandOutput:
        output = process_command(session, command_parameters)
        return CommandOutput(
            session_id=session.id,
            command_responses=[
                CommandResponse(response=f"The user id is: {output.status}")
            ]
        )


# if __name__ == "__main__":

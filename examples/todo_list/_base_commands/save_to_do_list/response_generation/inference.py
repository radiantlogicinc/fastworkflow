from fastworkflow import CommandOutput, CommandResponse
from fastworkflow.session import Session

from .command_implementation import process_command


class ResponseGenerator:
    def __call__(
        self, session: Session, command: str
    ) -> CommandOutput:
        output = process_command(session)

        # Format the list of commands
        utterance_list = "\n".join([f"- {cmd}" for cmd in output.utterances])

        # Create the response
        response = (
            f"Here are some example commands available in this task ({output.workitem_path}):\n"
            f"{utterance_list}\n"
            f"Your chat message must fall within the scope of these utterances."
        )

        return CommandOutput(
            session_id=session.id,
            command_responses=[
                CommandResponse(response=response)
            ]
        )

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
                CommandResponse(
                    response=(
                        f"was the next workitem found: {output.next_workitem_found}\n"
                        f"status of the new workitem: {output.status_of_next_workitem}"
                    )
                )
            ]
        )


# if __name__ == "__main__":

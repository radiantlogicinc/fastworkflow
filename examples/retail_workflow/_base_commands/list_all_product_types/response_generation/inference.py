from fastworkflow import CommandOutput, CommandResponse
from fastworkflow.session import Session
from .command_implementation import process_command


class ResponseGenerator:
    def __call__(
        self,
        session: Session,
        command: str
    ) -> CommandOutput:
        output = process_command(session)
        return CommandOutput(
            session_id=session.id,
            command_responses=[
                CommandResponse(response=f"list of product types: {output.status}")
            ]
        )


# if __name__ == "__main__":

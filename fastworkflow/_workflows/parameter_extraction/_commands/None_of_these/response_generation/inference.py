from fastworkflow import CommandOutput, CommandResponse
from fastworkflow.session import Session

from .command_implementation import process_command


class ResponseGenerator:
    def __call__(
        self, session: Session, command: str
    ) -> CommandOutput:
        session.workflow_snapshot.workflow.is_complete = True
        return CommandOutput(
            session_id=session.id,
            command_responses=[
                CommandResponse(
                    response="Command ambiguous",
                    artifacts={"command_name": "None_of_these", "None": True},
                )
            ]
        )

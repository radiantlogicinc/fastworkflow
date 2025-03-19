import fastworkflow
from fastworkflow import CommandOutput, CommandResponse
from fastworkflow.session import Session


class ResponseGenerator:
    def __call__(
        self, session: Session, command: str
    ) -> CommandOutput:
        session.workflow_snapshot.workflow.is_complete = True
        return CommandOutput(
            session_id=session.id,
            command_responses=[
                CommandResponse(
                    response="Command aborted",
                    artifacts={"abort": True},
                )
            ]
        )

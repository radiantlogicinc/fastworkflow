import fastworkflow
from fastworkflow import CommandOutput, CommandResponse
from fastworkflow.session import Session


class ResponseGenerator:
    def __call__(
        self, session: Session, command: str
    ) -> CommandOutput:
        session.workflow_snapshot.workflow.is_complete = True
        return CommandOutput(
            command_responses=[
                CommandResponse(
                    response="command aborted",
                    artifacts={
                        "command": command,
                        "command_name": "abort",
                    },
                )
            ]
        )

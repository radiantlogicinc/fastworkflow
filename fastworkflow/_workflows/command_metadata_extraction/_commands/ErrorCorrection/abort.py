from fastworkflow import CommandOutput, CommandResponse
from fastworkflow.session import Session
from fastworkflow.train.generate_synthetic import generate_diverse_utterances
from pydantic import BaseModel, ConfigDict


class Signature:
    class Output(BaseModel):
        command: str
        command_name: str

    plain_utterances = [
        "cancel",
        "stop",
        "quit",
        "terminate",
        "end",
        "never mind",
        "exit",
    ]
    template_utterances = []

    @staticmethod
    def generate_utterances(session: Session, command_name: str) -> list[str]:
        return generate_diverse_utterances(Signature.plain_utterances, command_name)


class ResponseGenerator:
    def _process_command(self, session: Session, command: str) -> Signature.Output:
        session.workflow_snapshot.is_complete = True
        return Signature.Output(command=command, command_name="abort")

    def __call__(self, session: Session, command: str) -> CommandOutput:
        output = self._process_command(session, command)
        return CommandOutput(
            session_id=session.id,
            command_responses=[
                CommandResponse(
                    response="command aborted",
                    artifacts=output.model_dump(),
                )
            ],
        ) 
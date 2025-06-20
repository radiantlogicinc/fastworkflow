import fastworkflow
from fastworkflow import CommandOutput, CommandResponse
from fastworkflow.session import Session
from fastworkflow.train.generate_synthetic import generate_diverse_utterances
from pydantic import BaseModel, ConfigDict, Field


class Signature:
    plain_utterances = [
        "That is not what I meant",
        "Not what I asked",
        "You misunderstood",
        "None of these commands",
        "Incorrect command",
        "Wrong command",
        "Change command",
        "Different command",
    ]
    template_utterances = []

    class Output(BaseModel):
        command_name: str
        is_none: bool = Field(..., alias="None")

    @staticmethod
    def generate_utterances(session: Session, command_name: str) -> list[str]:
        return generate_diverse_utterances(Signature.plain_utterances, command_name)


class ResponseGenerator:
    def _process_command(self, session: Session) -> Signature.Output:
        session.workflow_snapshot.is_complete = True
        return Signature.Output(command_name="misunderstood_intent", is_none=True)

    def __call__(self, session: Session, command: str) -> CommandOutput:
        output = self._process_command(session)
        return CommandOutput(
            session_id=session.id,
            command_responses=[
                CommandResponse(
                    response="Ambiguous command",
                    artifacts=output.model_dump(by_alias=True),
                )
            ],
        ) 
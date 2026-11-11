"""Command with parameters whose validate_extracted_parameters requires context."""

from __future__ import annotations

import fastworkflow
from fastworkflow.workflow import Workflow
from pydantic import BaseModel, Field, model_validator

HOOK_MISSING_PAYLOAD = (
    "payload not found in context. Run the initializing command first."
)


class Signature:
    class Input(BaseModel):
        note: str = Field(description="A note to attach to the payload")

        @model_validator(mode="after")
        def refuse_without_payload(self, info):
            workflow = info.context.get("workflow")
            if not workflow.context.get("payload"):
                raise fastworkflow.Refused(HOOK_MISSING_PAYLOAD)
            return self

    class Output(BaseModel):
        result: str = Field(description="Echo of payload and note")

    plain_utterances = [
        "use the payload",
        "apply payload with note",
    ]

    template_utterances = []

    @staticmethod
    def generate_utterances(workflow: Workflow, command_name: str) -> list[str]:
        return [command_name.split("/")[-1].lower().replace("_", " ")]

class ResponseGenerator:
    def __call__(self, workflow: Workflow, command: str, input: Signature.Input) -> fastworkflow.CommandResponse:
        payload = workflow.context.get("payload")
        response = f"payload={payload}; note={input.note}"
        return fastworkflow.CommandResponse(
            response=response, output=Signature.Output(result=response))

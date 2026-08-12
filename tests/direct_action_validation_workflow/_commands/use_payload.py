"""Command with parameters whose validate_extracted_parameters requires context."""

from __future__ import annotations

import fastworkflow
from fastworkflow import CommandOutput, CommandResponse
from fastworkflow.workflow import Workflow
from pydantic import BaseModel, Field

HOOK_MISSING_PAYLOAD = (
    "payload not found in context. Run the initializing command first."
)


class Signature:
    class Input(BaseModel):
        note: str = Field(description="A note to attach to the payload")

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

    @staticmethod
    def validate_extracted_parameters(
        workflow: fastworkflow.Workflow,
        command: str,
        cmd_parameters: "Signature.Input",
    ) -> tuple[bool, str]:
        if not workflow.context.get("payload"):
            return (False, HOOK_MISSING_PAYLOAD)
        return (True, "")


class ResponseGenerator:
    def __call__(
        self,
        workflow: Workflow,
        command: str,
        command_parameters: Signature.Input,
    ) -> CommandOutput:
        payload = workflow.context.get("payload")
        response = f"payload={payload}; note={command_parameters.note}"
        return CommandOutput(
            command_response=CommandResponse(response=response)
        )

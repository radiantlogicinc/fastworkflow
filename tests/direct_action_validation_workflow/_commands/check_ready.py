"""Parameter-free command whose validate_extracted_parameters requires context.

Used to lock down the former `if action.parameters:` guard: an empty/absent
parameter dict must still run the hook.
"""

from __future__ import annotations

from typing import Optional

import fastworkflow
from fastworkflow import CommandOutput, CommandResponse
from fastworkflow.workflow import Workflow
from pydantic import BaseModel, Field

HOOK_NOT_READY = "workflow is not ready. Set context['ready'] first."


class Signature:
    class Input(BaseModel):
        """All fields optional so callers can invoke with no parameters."""

        unused: Optional[str] = Field(
            default=None,
            description="Optional unused field so Signature.Input exists",
        )

    class Output(BaseModel):
        status: str = Field(description="Ready status")

    plain_utterances = [
        "check ready",
        "am I ready",
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
        if not workflow.context.get("ready"):
            return (False, HOOK_NOT_READY)
        return (True, "")


class ResponseGenerator:
    def __call__(
        self,
        workflow: Workflow,
        command: str,
        command_parameters: Signature.Input,
    ) -> CommandOutput:
        return CommandOutput(
            command_response=CommandResponse(response="ready=true")
        )

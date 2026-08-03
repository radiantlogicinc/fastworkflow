"""Half of the deliberate duplicate pair. See the workflow README."""

from typing import List

import fastworkflow
from pydantic import BaseModel, Field

from fastworkflow import CommandOutput, CommandResponse
from fastworkflow.train.generate_synthetic import generate_diverse_utterances
from fastworkflow.workflow import Workflow


class Signature:
    """List the control findings recorded for a scope."""

    class Input(BaseModel):
        scope: str = Field(default="NOT_FOUND", description="Scope to list findings for")
        severity: str = Field(default="NOT_FOUND", description="Severity filter")

    class Output(BaseModel):
        findings: str = Field(description="The findings that were found.")

    plain_utterances: List[str] = [
        "show me the control findings",
        "list the findings for this scope",
        "what control findings do we have open",
        "pull up the findings, high severity only",
        "I need to see the control findings for the finance scope",
        "give me the open findings list",
    ]
    template_utterances: List[str] = []

    @staticmethod
    def generate_utterances(workflow: Workflow, command_name: str) -> List[str]:
        return [
            command_name.split("/")[-1].lower().replace("_", " ")
        ] + generate_diverse_utterances(Signature.plain_utterances, command_name)

    @staticmethod
    def validate_extracted_parameters(
        workflow: fastworkflow.Workflow, command: str, cmd_parameters: "Signature.Input"
    ) -> tuple[bool, str]:
        return (True, "")


class ResponseGenerator:
    def _process_command(
        self, workflow: Workflow, input: Signature.Input
    ) -> Signature.Output:
        return Signature.Output(findings=f"findings for {input.scope}")

    def __call__(
        self, workflow: Workflow, command: str, command_parameters: Signature.Input
    ) -> CommandOutput:
        output = self._process_command(workflow, command_parameters)
        return CommandOutput(
            command_responses=[CommandResponse(response=output.findings)]
        )

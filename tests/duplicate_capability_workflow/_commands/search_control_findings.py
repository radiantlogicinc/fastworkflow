"""The other half of the deliberate duplicate pair. See the workflow README.

Written as a second author would write it: same capability, no shared sentence with
`list_findings`, but the same vocabulary because the domain only has one vocabulary.
"""

from typing import List

import fastworkflow
from pydantic import BaseModel, Field

from fastworkflow import CommandOutput, CommandResponse
from fastworkflow.train.generate_synthetic import generate_diverse_utterances
from fastworkflow.workflow import Workflow


class Signature:
    """Search the control findings for a scope."""

    class Input(BaseModel):
        scope: str = Field(default="NOT_FOUND", description="Scope to search findings in")
        severity: str = Field(default="NOT_FOUND", description="Severity filter")

    class Output(BaseModel):
        findings: str = Field(description="The findings that matched.")

    plain_utterances: List[str] = [
        "find the control findings for this scope",
        "search the findings we have open",
        "which control findings are recorded here",
        "look up the findings with high severity",
        "I want the control findings for the finance scope",
        "get me a list of the open findings",
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
        return Signature.Output(findings=f"findings matching {input.scope}")

    def __call__(
        self, workflow: Workflow, command: str, command_parameters: Signature.Input
    ) -> CommandOutput:
        output = self._process_command(workflow, command_parameters)
        return CommandOutput(
            command_responses=[CommandResponse(response=output.findings)]
        )

"""Hard negative: same subject matter as the duplicate pair, different capability."""

from typing import List

import fastworkflow
from pydantic import BaseModel, Field

from fastworkflow import CommandOutput, CommandResponse
from fastworkflow.train.generate_synthetic import generate_diverse_utterances
from fastworkflow.workflow import Workflow


class Signature:
    """Acknowledge a control finding so it stops being reported as open."""

    class Input(BaseModel):
        finding_id: str = Field(default="NOT_FOUND", description="Finding to acknowledge")
        justification: str = Field(default="NOT_FOUND", description="Why it is accepted")

    class Output(BaseModel):
        status: str = Field(description="Whether the acknowledgement succeeded.")

    plain_utterances: List[str] = [
        "acknowledge finding F-1201",
        "mark this control finding as accepted risk",
        "sign off on the finding, we have a compensating control",
        "dismiss finding F-88 with a justification",
        "accept this finding, it is a known exception",
        "close out the finding I just reviewed",
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
        return Signature.Output(status=f"acknowledged {input.finding_id}")

    def __call__(
        self, workflow: Workflow, command: str, command_parameters: Signature.Input
    ) -> CommandOutput:
        output = self._process_command(workflow, command_parameters)
        return CommandOutput(
            command_responses=[CommandResponse(response=output.status)]
        )

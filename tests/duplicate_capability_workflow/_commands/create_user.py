"""Easy negative: an unrelated capability."""

from typing import List

import fastworkflow
from pydantic import BaseModel, Field

from fastworkflow import CommandOutput, CommandResponse
from fastworkflow.train.generate_synthetic import generate_diverse_utterances
from fastworkflow.workflow import Workflow


class Signature:
    """Create a new user account in the directory."""

    class Input(BaseModel):
        email: str = Field(default="NOT_FOUND", description="Email of the new user")
        display_name: str = Field(default="NOT_FOUND", description="Display name")

    class Output(BaseModel):
        status: str = Field(description="Whether the account was created.")

    plain_utterances: List[str] = [
        "create a new user account",
        "onboard someone with this email address",
        "add a person to the directory",
        "I need to provision an account for a new hire",
        "set up a login for jane@example.com",
        "register a new employee in the system",
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
        return Signature.Output(status=f"created {input.email}")

    def __call__(
        self, workflow: Workflow, command: str, command_parameters: Signature.Input
    ) -> CommandOutput:
        output = self._process_command(workflow, command_parameters)
        return CommandOutput(
            command_responses=[CommandResponse(response=output.status)]
        )

"""ProposalReview/add: insert one new step."""

from __future__ import annotations

from typing import Optional

import fastworkflow
from fastworkflow.train.generate_synthetic import generate_diverse_utterances
from fastworkflow.workflow import Workflow
from pydantic import BaseModel, ConfigDict, Field


class Signature:
    """Add a new step to the plan."""

    class Input(BaseModel):
        model_config = ConfigDict(extra="forbid")
        goal: str = Field(description="what the new step should do")
        after: Optional[int] = Field(
            default=None,
            description="insert after this step number, or at the end when omitted",
        )

    class Output(BaseModel):
        model_config = ConfigDict(extra="forbid")
        goal: str
        after: Optional[int] = None

    plain_utterances = [
        "add a step to verify the address",
        "insert a step after 2 to confirm the phone number",
        "also check that the membership is active",
        "before finishing, add a step to print a receipt",
        "include one more step to notify the member",
        "put a new step before the last one",
        "append a step after the first one",
        "add another step that looks up the hold status",
        "slide in a step between 1 and 2 to verify identity",
        "we need an extra step at the end to send a reminder",
        "can you add a step about updating the address",
        "insert one more action after step 3",
    ]

    @staticmethod
    def generate_utterances(workflow: Workflow, command_name: str) -> list[str]:
        return [
            command_name.split("/")[-1].lower().replace("_", " ")
        ] + generate_diverse_utterances(Signature.plain_utterances, command_name)

    @staticmethod
    def validate_extracted_parameters(workflow, command, cmd_parameters) -> None:
        proposal = workflow.context.get("consultation_proposal")
        if proposal is None or cmd_parameters.after is None:
            return None
        count = len(proposal.items)
        if not (0 <= cmd_parameters.after <= count):
            raise fastworkflow.Refused(
                f"There are only {count} step(s)." if count != 1 else "There is only 1 step.",
                fields=["after"],
            )
        return None


class ResponseGenerator:
    def __call__(
        self,
        workflow: Workflow,
        command: str,
        input: Signature.Input,
    ) -> fastworkflow.CommandResponse:
        output = Signature.Output(goal=input.goal, after=input.after)
        return fastworkflow.CommandResponse(
            response="",
            artifacts=output.model_dump(),
            output=output,
        )

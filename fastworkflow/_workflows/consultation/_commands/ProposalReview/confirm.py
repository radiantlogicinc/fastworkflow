"""ProposalReview/confirm: the user accepts the proposed plan."""

from __future__ import annotations

import fastworkflow
from fastworkflow.train.generate_synthetic import generate_diverse_utterances
from fastworkflow.workflow import Workflow
from pydantic import BaseModel, ConfigDict, Field


class Signature:
    """Accept the plan as shown."""

    class Output(BaseModel):
        model_config = ConfigDict(extra="forbid")
        decision: str = Field(default="confirm", description="accept the plan")

    plain_utterances = [
        "yes",
        "yep",
        "yeah",
        "sure",
        "ok",
        "okay",
        "sounds good",
        "looks good",
        "go ahead",
        "that works",
        "accept",
        "approve",
        "run it",
        "do it",
        "perfect",
        "fine by me",
        "agreed",
    ]

    @staticmethod
    def generate_utterances(workflow: Workflow, command_name: str) -> list[str]:
        return [
            command_name.split("/")[-1].lower().replace("_", " ")
        ] + generate_diverse_utterances(Signature.plain_utterances, command_name)


class ResponseGenerator:
    def __call__(
        self,
        workflow: Workflow,
        command: str,
        input: None,
    ) -> fastworkflow.CommandResponse:
        output = Signature.Output()
        return fastworkflow.CommandResponse(
            response="",
            artifacts=output.model_dump(),
            output=output,
        )

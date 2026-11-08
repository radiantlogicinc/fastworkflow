"""ProposalReview/decline: the user refuses the proposed plan."""

from __future__ import annotations

import fastworkflow
from fastworkflow.train.generate_synthetic import generate_diverse_utterances
from fastworkflow.workflow import Workflow
from pydantic import BaseModel, ConfigDict, Field


class Signature:
    """Decline the plan."""

    class Output(BaseModel):
        model_config = ConfigDict(extra="forbid")
        decision: str = Field(default="decline", description="decline the plan")

    plain_utterances = [
        "no",
        "nope",
        "never mind",
        "cancel",
        "stop",
        "don't",
        "do not",
        "forget it",
        "not now",
        "leave it",
        "abort",
        "quit",
        "decline",
        "reject",
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

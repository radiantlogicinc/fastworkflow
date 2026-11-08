"""ProposalReview/look_up: re-read or re-draft one step the user named."""

from __future__ import annotations

import fastworkflow
from fastworkflow.train.generate_synthetic import generate_diverse_utterances
from fastworkflow.workflow import Workflow
from pydantic import BaseModel, ConfigDict, Field

from fastworkflow._workflows.consultation import _item_validation


class Signature:
    """Look up one step."""

    class Input(BaseModel):
        model_config = ConfigDict(extra="forbid")
        item: str = Field(
            description="step number or goal phrase to look up",
            examples=["2", "the renewal step", "first step"],
        )

    class Output(BaseModel):
        model_config = ConfigDict(extra="forbid")
        item: str

    plain_utterances = [
        "what does step 2 do",
        "look up step 2",
        "just look that up",
        "only check it, don't change anything",
        "read it without updating",
        "find the record instead of modifying it",
        "look up step 1 first",
        "check the current value on step 2",
        "inspect it rather than editing",
        "query it, don't write",
        "just fetch the information for step 2",
        "search for it instead",
    ]

    @staticmethod
    def generate_utterances(workflow: Workflow, command_name: str) -> list[str]:
        return [
            command_name.split("/")[-1].lower().replace("_", " ")
        ] + generate_diverse_utterances(Signature.plain_utterances, command_name)

    @staticmethod
    def validate_extracted_parameters(workflow, command, cmd_parameters) -> None:
        proposal = workflow.context.get("consultation_proposal")
        if proposal is None:
            return None
        cmd_parameters.item = _item_validation.resolve_proposal_step(
            proposal, str(cmd_parameters.item), field_name="item", slot_retry=True)
        return None


class ResponseGenerator:
    def __call__(
        self,
        workflow: Workflow,
        command: str,
        input: Signature.Input,
    ) -> fastworkflow.CommandResponse:
        output = Signature.Output(item=input.item)
        return fastworkflow.CommandResponse(
            response="",
            artifacts=output.model_dump(),
            output=output,
        )

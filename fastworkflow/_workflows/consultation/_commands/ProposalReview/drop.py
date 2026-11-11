"""ProposalReview/drop: remove one or more steps the user named."""

from __future__ import annotations

import fastworkflow
from fastworkflow.train.generate_synthetic import generate_diverse_utterances
from fastworkflow.workflow import Workflow
from pydantic import BaseModel, ConfigDict, Field, model_validator

from fastworkflow._workflows.consultation import _item_validation


class Signature:
    """Drop steps from the plan."""

    class Input(BaseModel):
        model_config = ConfigDict(extra="forbid")
        items: list[str] = Field(
            description="step numbers and/or goal phrases the user wants removed",
            examples=[["2"], ["3", "check membership"], ["the last step"]],
        )

        @model_validator(mode="after")
        def drop_items_name_proposal_steps(self, info):
            workflow = info.context.get("workflow")
            proposal = workflow.context.get("consultation_proposal")
            if proposal is None:
                return self
            if not self.items:
                raise fastworkflow.Refused("Which steps should I drop?", fields=["items"])
            normalized: list[str] = []
            for entry in self.items:
                normalized.append(
                    _item_validation.resolve_proposal_step(proposal, str(entry), field_name="items", slot_retry=True)
                )
            self.items = normalized
            return self

    class Output(BaseModel):
        model_config = ConfigDict(extra="forbid")
        items: list[str] = Field(description="steps to drop")

    plain_utterances = [
        "drop step 2",
        "skip the second step",
        "leave out step 1",
        "cut step 3",
        "don't do the third step",
        "remove steps 2 and 4",
        "omit the last step",
        "take step 2 out of the plan",
        "delete the first step",
        "we don't need step 2",
        "without step 2",
        "get rid of step 3",
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
        input: Signature.Input,
    ) -> fastworkflow.CommandResponse:
        output = Signature.Output(items=list(input.items))
        return fastworkflow.CommandResponse(
            response="",
            artifacts=output.model_dump(),
            output=output,
        )

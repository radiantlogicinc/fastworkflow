"""review_plan: review a plan (goals with bound steps) with the user, then return the accepted plan.

A framework planning command (global scope, lowest precedence). Plans are goals with bound steps, never a bare
command list (arch 19.0 invariant 1). One phase makes one ``review_proposal`` call; the accepted plan is this
command's typed Output.
"""
import json

from pydantic import BaseModel, Field

from fastworkflow.planning import run
from fastworkflow._framework_commands.review_proposal import Signature as ReviewProposal
from fastworkflow.review_proposals import PlanGoal


class Signature:
    """Review a plan of goals, each with its bound steps, before it runs."""

    plain_utterances = ["let me approve the plan first", "show me the plan before you run it"]

    class Input(BaseModel):
        goals: str = Field(description="the goals, each as 'goal = step | step', separated by semicolons")

    class Output(BaseModel):
        items: list[PlanGoal] = Field(description="the goals and their steps the user accepted")


class PlanGenerator:
    class Review(BaseModel):
        proposal: ReviewProposal.Output

        @property
        def accepted(self) -> str:
            return self.proposal.accepted

    def review(self, input) -> Review:
        return [run("review_proposal", round=2, items=input.goals)]

    def __result__(self, input, review):
        return Signature.Output(items=[PlanGoal(**item) for item in json.loads(review.accepted)["items"]])

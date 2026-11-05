"""review_subtasks: review the sub-tasks of a request with the user, then return the accepted ones.

A framework planning command (global scope, lowest precedence). Its one phase builds the round-1 proposal
from the sub-tasks the request names, and makes one ``review_proposal`` call: the framework's review loop asks
the user, applies their edits, and returns only on accept or decline. The accepted proposal is this command's
typed Output.
"""
import json

from pydantic import BaseModel, Field

from fastworkflow.planning import fact_from, run
from fastworkflow.review_proposals import SubTask


class Signature:
    """Review the sub-tasks of a request with the user before they run."""

    plain_utterances = ["review the sub-tasks with me first", "split this request and let me check it"]

    class Input(BaseModel):
        subtasks: str = Field(description="the sub-tasks of the request, separated by semicolons (or the draft token "
                                          "the sub-task gate passes, whose draft it holds)")

    class Output(BaseModel):
        items: list[SubTask] = Field(description="the sub-tasks the user accepted")


class PlanGenerator:
    class Review(BaseModel):
        accepted: str = fact_from("review_proposal.accepted")

    def review(self, input) -> Review:
        return [run("review_proposal", round=1, items=input.subtasks)]

    def __result__(self, input, review):
        return Signature.Output(items=[SubTask(**item) for item in json.loads(review.accepted)["items"]])

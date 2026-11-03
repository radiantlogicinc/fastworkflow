"""review_proposal: put a typed proposal to the user and return the accepted one (fix-3rsq.plan-review.1).

A framework execution command (fastworkflow/_framework_commands; system machinery, not an agent or user tool):
the review loop, called once by a review planning command
(review_subtasks, review_plan). The first call opens the review and suspends the agent on the prompt
(ask_user): the user's replies edit the proposal, and only confirm or decline ends the review. On accept
the framework runs this command again under the accepted record (review_engine.approved_review), and the
command then returns the accepted proposal as its typed Output. The agent never produces an accepted
proposal itself.
"""
import fastworkflow
from pydantic import BaseModel, Field

from fastworkflow import review_engine


class Signature:
    """Put a proposal to the user for review and return what the user accepted."""

    class Input(BaseModel):
        round: int = Field(description="the review round: 1 for sub-tasks, 2 for a plan")
        items: str = Field(default="", description="the proposal's items, as the proposal reads them")

    class Output(BaseModel):
        round: int = Field(description="the review round that was accepted: 1 or 2")
        accepted: str = Field(description="the accepted items, as JSON")

    plain_utterances = ["review the sub-tasks with me", "let me approve the plan first"]


class ResponseGenerator:
    def __call__(self, workflow: fastworkflow.Workflow, command: str,
                 command_parameters: Signature.Input) -> Signature.Output:
        from fastworkflow.review_proposals import PROPOSAL_KINDS

        cls = PROPOSAL_KINDS.get(command_parameters.round)
        if cls is None:
            raise ValueError(f"no review round {command_parameters.round!r}: use 1 (sub-tasks) or 2 (plan)")
        accepted = review_engine.approved_review_proposal(workflow, cls.spec)
        if accepted is not None:
            return Signature.Output(round=command_parameters.round, accepted=accepted.model_dump_json())
        proposal = cls.from_text(command_parameters.items)
        review_engine.open_for_workflow(workflow, proposal,
                                        items_text=command_parameters.items)   # raises AskUserSuspend
        raise AssertionError("open_for_workflow must suspend")

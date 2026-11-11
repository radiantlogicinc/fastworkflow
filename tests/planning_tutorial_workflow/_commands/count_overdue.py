"""count_overdue: how many loans are overdue, in one branch or in all."""
import fastworkflow
from pydantic import BaseModel, Field

from ..application import library


class Signature:
    """Count the overdue loans, in one branch or across the library."""

    class Input(BaseModel):
        branch: str = Field(default="", description="one branch's name; empty for every branch")

    class Output(BaseModel):
        total: int = Field(description="how many loans are overdue")

    plain_utterances = ["how many loans are overdue", "count the overdue loans at Central"]


class ResponseGenerator:
    def __call__(self, workflow: fastworkflow.Workflow, command: str, input: Signature.Input) -> fastworkflow.CommandResponse:
        output = Signature.Output(total=library.overdue_count(input.branch))
        return fastworkflow.CommandResponse(response=f"Overdue loans: {output.total}", output=output)

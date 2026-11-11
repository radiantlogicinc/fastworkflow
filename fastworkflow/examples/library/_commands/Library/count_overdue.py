"""count_overdue: how many loans are overdue, in one branch or in all."""
import fastworkflow
from pydantic import BaseModel, Field

from ...application.library import Library


class Signature:
    """Count the overdue loans, in one branch or across the library."""

    class Input(BaseModel):
        branch: str = Field(default="", description="one branch's name; empty for every branch")

    class Output(BaseModel):
        total: int = Field(description="how many loans are overdue")

    plain_utterances = [
        "how many loans are overdue",
        "count the overdue loans at Central",
        "how many books are past their due date",
        "number of overdue loans at Riverside",
        "tally the late loans across all branches",
        "how many of our loans are late",
        "give me the late-loan count for Riverside",
        "are there overdue books at Central",
    ]


class ResponseGenerator:
    def __call__(self, workflow: fastworkflow.Workflow, command: str,
                 input: Signature.Input) -> Signature.Output:
        library: Library = workflow.record
        return Signature.Output(total=library.overdue_count(input.branch))

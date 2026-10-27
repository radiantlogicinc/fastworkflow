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

    plain_utterances = ["how many loans are overdue", "count the overdue loans at Central"]


class ResponseGenerator:
    def __call__(self, workflow: fastworkflow.Workflow, command: str,
                 command_parameters: Signature.Input) -> Signature.Output:
        library: Library = workflow.command_context_for_response_generation
        return Signature.Output(total=library.overdue_count(command_parameters.branch))

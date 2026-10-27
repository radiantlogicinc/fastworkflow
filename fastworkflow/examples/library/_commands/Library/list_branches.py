"""list_branches: the library's branches."""
import fastworkflow
from pydantic import BaseModel, Field

from ...application.library import Library


class Signature:
    """List the library's branches."""

    class Output(BaseModel):
        names: list[str] = Field(description="the branch names")
        total: int = Field(description="how many branches there are")

    plain_utterances = ["which branches does the library have"]


class ResponseGenerator:
    def __call__(self, workflow: fastworkflow.Workflow, command: str,
                 command_parameters: None) -> Signature.Output:
        library: Library = workflow.command_context_for_response_generation
        return Signature.Output(names=list(library.branches), total=len(library.branches))

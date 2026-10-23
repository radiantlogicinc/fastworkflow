"""list_branches: the library's branches."""
import fastworkflow
from pydantic import BaseModel, Field

from ..application import library


class Signature:
    """List the library's branches."""

    class Output(BaseModel):
        names: list[str] = Field(description="the branch names")
        total: int = Field(description="how many branches there are")

    plain_utterances = ["which branches does the library have"]


class ResponseGenerator:
    def __call__(self, workflow: fastworkflow.Workflow, command: str) -> fastworkflow.CommandOutput:
        output = Signature.Output(names=list(library.BRANCHES), total=len(library.BRANCHES))
        return fastworkflow.CommandOutput(command_response=fastworkflow.CommandResponse(
            response=f"Branches: {', '.join(output.names)}", output=output))

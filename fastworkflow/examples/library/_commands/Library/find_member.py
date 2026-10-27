"""find_member: the members whose name holds a query."""
import fastworkflow
from pydantic import BaseModel, Field

from ...application.library import Library


class Signature:
    """Find library members by name."""

    class Input(BaseModel):
        query: str = Field(description="words of the member's name", examples=["Jane Doe"])

    class Output(BaseModel):
        member_uids: list[str] = Field(description="the uid of each member found, in row order; "
                                                   "open_member takes one")
        names: list[str] = Field(description="the name of each member found, in row order")
        total: int = Field(description="how many members match")

    plain_utterances = ["find the member Ann Lee"]


class ResponseGenerator:
    def __call__(self, workflow: fastworkflow.Workflow, command: str,
                 command_parameters: Signature.Input) -> Signature.Output:
        library: Library = workflow.command_context_for_response_generation
        found = library.find_members(command_parameters.query)
        return Signature.Output(member_uids=found,
                                names=[library.members[u].name for u in found], total=len(found))

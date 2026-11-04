"""find_member: the members whose name holds a query."""
import fastworkflow
from pydantic import BaseModel, Field

from ..application import library


class Signature:
    """Find library members by name."""

    class Input(BaseModel):
        query: str = Field(description="words of the member's name")

    class Output(BaseModel):
        member_uids: list[str] = Field(description="the uid of each member found, in row order")
        names: list[str] = Field(description="the name of each member found, in row order")
        total: int = Field(description="how many members match")

    plain_utterances = ["find the member Ann Lee"]


class ResponseGenerator:
    def __call__(self, workflow: fastworkflow.Workflow, command: str, input: Signature.Input) -> Signature.Output:
        found = library.find_members(input.query)
        output = Signature.Output(member_uids=found, names=[library.MEMBERS[u] for u in found],
                                  total=len(found))
        return output

"""find_record: how many books or members match a name, and which.

One find phase whose facts model depends on the kind the request asks about: a book search
fills BookMatch, a member search MemberMatch. Both have labels and total, so the answer
reads either.
"""
from typing import Literal

from pydantic import BaseModel, Field

from fastworkflow.planning import listed, phase, run

from .find_book import Signature as FindBook
from .find_member import Signature as FindMember


class Signature:
    """Say how many books or members match the name the request gives, and name them."""

    plain_utterances = ["do you have a book called Emma", "is there a member named Lee"]

    class Input(BaseModel):
        name: str = Field(description="the name to look for",
                          json_schema_extra={"in_request": True})
        kind: Literal["book", "member"] = Field(
            default="book", description="whether the request is about a book or a member")


class BookMatch(BaseModel):
    find: FindBook.Output

    @property
    def labels(self) -> list[str]:
        return self.find.titles

    @property
    def total(self) -> int:
        return self.find.total


class MemberMatch(BaseModel):
    find: FindMember.Output

    @property
    def labels(self) -> list[str]:
        return self.find.names

    @property
    def total(self) -> int:
        return self.find.total


class PlanGenerator:
    @phase(output=lambda input: BookMatch if input.kind == "book" else MemberMatch)
    def find(self, input):
        return [run(FIND[input.kind], query=input.name)]

    def __result__(self, input, find):
        return f"{input.name} matches {listed(find.labels, find.total, input.kind)}."


FIND = {"book": "find_book", "member": "find_member"}

"""find_record: how many books or members match a name, and which.

One find phase whose facts model depends on the kind the request asks about: a book search
fills BookMatch, a member search MemberMatch. Both have labels and total, so the answer
reads either.
"""
from typing import Literal

from pydantic import BaseModel, Field

from fastworkflow.skill import fact_from, listed, phase


class Signature:
    """Say how many books or members match the name the request gives, and name them."""

    plain_utterances = ["do you have a book called Emma", "is there a member named Lee"]

    class Input(BaseModel):
        name: str = Field(description="the name exactly as the request writes it")
        kind: Literal["book", "member"] = Field(
            default="book", description="whether the request is about a book or a member")


class BookMatch(BaseModel):
    labels: list[str] = fact_from("find_book.titles")
    total: int = fact_from("find_book.total")


class MemberMatch(BaseModel):
    labels: list[str] = fact_from("find_member.names")
    total: int = fact_from("find_member.total")


class PlanGenerator:
    @phase(output=lambda input: BookMatch if input.kind == "book" else MemberMatch)
    def find(self, input):
        return [f"{FIND[input.kind]} query={input.name}"]

    def answer(self, input, find):
        return [f"say that {input.name} matches {listed(find.labels, find.total, input.kind)}"]


FIND = {"book": "find_book", "member": "find_member"}

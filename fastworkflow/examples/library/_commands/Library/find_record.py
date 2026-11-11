"""find_record: whether a book or a member goes by a name, and which.

One find phase runs find_book or find_member, by the kind the request asks about, and
hands on the Output of whichever ran; __result__ reads its names either way.
"""
from typing import Literal

from pydantic import BaseModel, Field

from fastworkflow.planning import count_and_names_read, run

from .find_book import Signature as FindBook
from .find_member import Signature as FindMember


class Signature:
    """Say whether a book or a member goes by a name, and name each match."""

    plain_utterances = [
        "do you have a book called Emma",
        "is there a member named Lee",
        "do we have any member called Patel",
        "is there a book called Dune Messiah",
        "does anyone go by the name Mia",
        "is a title called Persuasion in the catalogue",
        "is there a patron named Raj",
        "check whether we have a book titled Dune and name the matches",
    ]

    class Input(BaseModel):
        name: str = Field(description="the name to look for",
                          json_schema_extra={"copied_from_command_text": True})
        kind: Literal["book", "member"] = Field(
            default="book", description="whether to look for a book or a member")


class PlanGenerator:
    def find(self, input) -> FindBook.Output | FindMember.Output:
        return [run(FIND[input.kind], query=input.name)]

    def __result__(self, input, find):
        names = find.titles if input.kind == "book" else find.names
        return f"{input.name} matches {count_and_names_read(names, find.total, input.kind)}."


FIND = {"book": "find_book", "member": "find_member"}

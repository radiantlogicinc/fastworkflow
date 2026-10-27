"""count_books: how many books match the words the request gives.

The find phase runs find_book and hands its Output on; __result__ writes the answer from
that Output's total.
"""
from pydantic import BaseModel, Field

from fastworkflow.planning import count, run

from .find_book import Signature as FindBook


class Signature:
    """Count the books whose title or author holds some words."""

    plain_utterances = ["how many books by Jane Austen do you have"]

    class Input(BaseModel):
        query: str = Field(description="words of the title or the author's name",
                           json_schema_extra={"in_request": True})


class PlanGenerator:
    def find(self, input) -> FindBook.Output:
        return [run("find_book", query=input.query)]

    def __result__(self, input, find):
        return f"{input.query} matches {count(find.total, 'book')}."

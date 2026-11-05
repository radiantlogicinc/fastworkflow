"""count_books: how many books match the words the request gives.

The find phase runs find_book and keeps its total as a fact; __result__ writes the
answer from that fact.
"""
from pydantic import BaseModel, Field

from fastworkflow.planning import run

from .find_book import Signature as FindBook


class Signature:
    """Say how many books have the request's words in their title or author."""

    plain_utterances = ["how many books by Jane Austen do you have"]

    class Input(BaseModel):
        query: str = Field(description="the title or author words",
                        json_schema_extra={"in_request": True})


class PlanGenerator:
    def find(self, input) -> FindBook.Output:
        return [run("find_book", query=input.query)]

    def __result__(self, input, find):
        return f"{find.total} books match {input.query}."

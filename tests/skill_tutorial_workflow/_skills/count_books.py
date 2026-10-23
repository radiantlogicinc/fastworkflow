"""count_books: how many books match the words the request gives.

The find phase runs find_book and keeps its total as a fact; the answer phase writes the
answer from that fact.
"""
from pydantic import BaseModel, Field

from fastworkflow.skill import fact_from


class Signature:
    """Say how many books have the request's words in their title or author."""

    plain_utterances = ["how many books by Jane Austen do you have"]

    class Input(BaseModel):
        query: str = Field(description="the title or author words exactly as the request "
                                       "writes them")


class PlanGenerator:
    class Found(BaseModel):
        total: int = fact_from("find_book.total")

    def find(self, input) -> Found:
        return [f"find_book query={input.query}"]

    def answer(self, input, find):
        return [f"say that {find.total} books match {input.query}"]

"""list_books: how many books match the words the request gives, and their titles.

count_books with its answer worded by listed(): "2 books (Dune, Dune Messiah)", "none (0
books)", or "12 books (...; 10 of 12 shown)" when the page read holds fewer than all.
"""
from pydantic import BaseModel, Field

from fastworkflow.planning import fact_from, listed, run


class Signature:
    """Say how many books have the request's words in their title or author, and name them."""

    plain_utterances = ["which books by Jane Austen do you have"]

    class Input(BaseModel):
        query: str = Field(description="the title or author words")


class PlanGenerator:
    class Found(BaseModel):
        titles: list[str] = fact_from("find_book.titles")
        total: int = fact_from("find_book.total")

    def find(self, input) -> Found:
        return [run("find_book", query=input.query)]

    def answer(self, input, find):
        return [f"say that {input.query} matches {listed(find.titles, find.total, 'book')}"]

"""book_details: the first book matching the request's words, or that none matches.

find lists the matches; book opens the first one, and is skipped when nothing matched;
answer says what book found, or that no book matches.
"""
from pydantic import BaseModel, Field

from fastworkflow.planning import fact_from, run


class Signature:
    """Show the author and copies of the book the request names."""

    plain_utterances = ["who wrote Persuasion", "how many copies of Emma are there"]

    class Input(BaseModel):
        title: str = Field(description="the book's title")


class PlanGenerator:
    class Found(BaseModel):
        book_uids: list[str] = fact_from("find_book.book_uids")

    def find(self, input) -> Found:
        return [run("find_book", query=input.title)]

    class Book(BaseModel):
        name: str = fact_from("show_book.title")
        author: str = fact_from("show_book.author")
        copies: int = fact_from("show_book.copies")

    def book(self, find) -> Book:
        if not find.book_uids:
            return []
        return [run("show_book", book_uid=find.book_uids[0])]

    def answer(self, input, book=None):
        if book is None:
            return [f"say that no book matches {input.title}"]
        return [f"say that {book.name} is by {book.author}; the library owns "
                f"{book.copies} copies"]

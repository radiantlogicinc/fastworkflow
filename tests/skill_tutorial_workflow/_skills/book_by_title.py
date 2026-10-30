"""book_by_title: the book whose title the request names, picked by its title.

find lists the matches; book opens the one whose title is the request's title (pick), and
is skipped when nothing matched; answer says what book found, or that no book matches.
"""
from pydantic import BaseModel, Field

from fastworkflow.planning import fact_from, pick, run


class Signature:
    """Show the author and copies of the book whose title the request names."""

    plain_utterances = ["who wrote Dune", "how many copies of Dune Messiah are there"]

    class Input(BaseModel):
        title: str = Field(description="the book's title")


class PlanGenerator:
    class Found(BaseModel):
        book_uids: list[str] = fact_from("find_book.book_uids")
        titles: list[str] = fact_from("find_book.titles")

    def find(self, input) -> Found:
        return [run("find_book", query=input.title)]

    class Book(BaseModel):
        uid: str = fact_from("show_book.book_uid")
        name: str = fact_from("show_book.title")
        author: str = fact_from("show_book.author")
        copies: int = fact_from("show_book.copies")

    def book(self, input, find) -> Book:
        if not find.book_uids:
            return []
        return [pick("show_book", named=input.title)]

    def answer(self, input, book=None):
        if book is None:
            return [f"say that no book matches {input.title}"]
        return [f"say that {book.name} is by {book.author}; the library owns "
                f"{book.copies} copies"]

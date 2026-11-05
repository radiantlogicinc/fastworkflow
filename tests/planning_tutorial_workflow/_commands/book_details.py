"""book_details: the first book matching the request's words, or that none matches.

find lists the matches; book opens the first one, and is skipped when nothing matched;
__result__ says what book found, or that no book matches.
"""
from pydantic import BaseModel, Field

from fastworkflow.planning import run

from .find_book import Signature as FindBook
from .show_book import Signature as ShowBook


class Signature:
    """Show the author and copies of the book the request names."""

    plain_utterances = ["who wrote Persuasion", "how many copies of Emma are there"]

    class Input(BaseModel):
        title: str = Field(description="the book's title")


class PlanGenerator:
    def find(self, input) -> FindBook.Output:
        return [run("find_book", query=input.title)]

    class Book(BaseModel):
        show: ShowBook.Output

        @property
        def name(self) -> str:
            return self.show.title

        @property
        def author(self) -> str:
            return self.show.author

        @property
        def copies(self) -> int:
            return self.show.copies

    def book(self, find) -> Book:
        if not find.book_uids:
            return []
        return [run("show_book", book_uid=find.book_uids[0])]

    def __result__(self, input, book=None):
        if book is None:
            return f"No book matches {input.title}."
        return (f"{book.name} is by {book.author}; the library owns "
                f"{book.copies} copies.")

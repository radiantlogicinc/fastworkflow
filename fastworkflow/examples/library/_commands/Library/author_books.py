"""author_books: each book by the author the request names, opened one by one.

find lists the author's books; book runs once per book found (at most five), each its own
sub-plan that opens that book; __result__ names every book with its copies.
"""
from pydantic import BaseModel, Field

from fastworkflow.planning import phase, run

from ._wording import copies
from .find_book import Signature as FindBook
from .open_book import Signature as OpenBook


class Signature:
    """Name each book by an author, with how many copies the library owns."""

    plain_utterances = [
        "what do you have by Frank Herbert, and how many copies of each",
        "list every Austen book we own with its copies",
        "what Herbert titles are on the shelves, and how many of each",
        "give me each book by Jane Austen with the copy count",
        "all the books by Frank Herbert, with copies",
        "name each title by Austen and how many copies it has",
        "which books by Jane Austen do we hold, copies included",
        "catalogue of Frank Herbert's books and copies",
    ]

    class Input(BaseModel):
        author: str = Field(description="the author's name",
                            json_schema_extra={"in_request": True})


class PlanGenerator:
    def find(self, input) -> FindBook.Output:
        return [run("find_book", query=input.author)]

    @phase(over="find.book_uids", max_items=5, noun="books")
    def book(self, uid) -> OpenBook.Output:
        return [run("open_book", book_uid=uid)]

    def __result__(self, input, book):
        if not book:
            return f"No book by {input.author} was found."
        each = "; ".join(f"{b.title} ({copies(b.copies)})" for b in book)
        return f"The books by {input.author}: {each}."

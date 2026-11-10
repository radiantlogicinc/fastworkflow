"""author_books: each book by the author the request names, opened one by one.

find lists the author's books; book runs once per book found (at most five), each its own
sub-plan; __result__ names every book with its copies.
"""
from pydantic import BaseModel, Field

from fastworkflow.planning import count, phase, run

from .find_book import Signature as FindBook
from .show_book import Signature as ShowBook


class Signature:
    """Name each book by the author the request names, with how many copies the library owns."""

    plain_utterances = ["what do you have by Frank Herbert, and how many copies of each"]

    class Input(BaseModel):
        author: str = Field(description="the author's name",
                            json_schema_extra={"copied_from_command_text": True})


class PlanGenerator:
    def find(self, input) -> FindBook.Output:
        return [run("find_book", query=input.author)]

    @phase(over="find.book_uids", max_items=5, noun="books", bind=False)
    def book(self, uid) -> ShowBook.Output:
        return [run("show_book", book_uid=uid)]

    def __result__(self, input, book=None):
        return self._books_by(input.author, book)

    def _books_by(self, author, book):
        if not book:
            return f"No book by {author} was found."
        copies = "; ".join(f"{b.title} ({count(b.copies, 'copy', 'copies')})" for b in book)
        return f"The books by {author}: {copies}."

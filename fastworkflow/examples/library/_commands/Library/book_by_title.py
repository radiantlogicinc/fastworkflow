"""book_by_title: the book whose title the request names, picked by its title.

find lists the matches; book opens the one whose title is the request's title (run(..., named=...)), and
is skipped when nothing matched; __result__ says the author and copies, or that no book
matches, and reports the book to a caller in its Output. The call ends on the book it
opened.
"""
from typing import Optional

from pydantic import BaseModel, Field

from fastworkflow import CommandResponse
from fastworkflow.planning import run

from ._wording import copies
from .find_book import Signature as FindBook
from .open_book import Signature as OpenBook


class Signature:
    """Show who wrote a book, and how many copies the library owns, given its title."""

    plain_utterances = [
        "who wrote Dune",
        "how many copies of Dune Messiah are there",
        "who is the author of Persuasion, and how many copies do we have",
        "tell me about the book Emma",
        "what is Dune Messiah by, and how many copies",
        "give me the author and copy count for the title Emma",
        "does the library own Persuasion, and who wrote it",
        "how many copies of Emma do we own and who wrote it",
    ]

    class Input(BaseModel):
        title: str = Field(description="the book's title",
                           json_schema_extra={"copied_from_command_text": True})

    class Output(BaseModel):
        found: bool = Field(description="whether a book was opened for the title")
        book_uid: Optional[str] = Field(default=None, description="the uid of the book opened")
        title: Optional[str] = Field(default=None, description="its title, as the library has it")
        author: Optional[str] = Field(default=None, description="its author")
        copies: int = Field(default=0, description="how many copies the library owns")


class PlanGenerator:
    def find(self, input) -> FindBook.Output:
        return [run("find_book", query=input.title)]

    def book(self, input, find) -> OpenBook.Output:
        if not find.book_uids:
            return []
        return [run("open_book", named=input.title)]

    def __result__(self, input, book):
        if book is None:
            return CommandResponse(response=f"No book matches {input.title}.",
                                   output=Signature.Output(found=False))
        return CommandResponse(
            response=f"{book.title} is by {book.author}; the library owns {copies(book.copies)}.",
            output=Signature.Output(found=True, **book.model_dump()))

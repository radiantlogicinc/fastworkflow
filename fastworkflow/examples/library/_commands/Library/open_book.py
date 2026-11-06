"""open_book: make one book, by its uid, the current record."""
import fastworkflow
from fastworkflow import from_
from pydantic import BaseModel, Field

from ...application.library import Library


class Signature:
    """Open one book by its uid, so that the book's own commands act on it."""

    class Input(BaseModel):
        book_uid: str = from_("find_book.book_uids", description="the uid of the book to open",
                               pattern=r"^b[0-9]+$")

    class Output(BaseModel):
        book_uid: str = Field(description="the uid of the book now open")
        title: str = Field(description="the book's title")
        author: str = Field(description="the book's author")
        copies: int = Field(description="how many copies the library owns")

    plain_utterances = [
        "open book b1",
        "go to the book Dune",
        "open the book with uid b3",
        "make b2 the current book",
        "select b2 so I can act on it",
        "I want to act on b3 now",
        "make the Dune Messiah book the open one",
        "load uid b4 as the current book",
    ]


class ResponseGenerator:
    descends_to = "Book"

    def __call__(self, workflow: fastworkflow.Workflow, command: str,
                 input: Signature.Input) -> Signature.Output:
        library: Library = workflow.command_context_for_response_generation
        book = library.books.get(input.book_uid)
        if book is None:
            raise fastworkflow.Refused(
                f"no book has the uid {input.book_uid}; find_book lists the uids")
        workflow.current_command_context = book
        return Signature.Output(book_uid=book.uid, title=book.title, author=book.author,
                                copies=book.copies)

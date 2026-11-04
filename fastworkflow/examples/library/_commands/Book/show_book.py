"""show_book: the open book's details."""
import fastworkflow
from pydantic import BaseModel, Field

from ...application.library import Book


class Signature:
    """Show the open book: its title, author, branch, copies and how many are on loan."""

    class Output(BaseModel):
        book_uid: str = Field(description="the book's uid")
        title: str = Field(description="the book's title")
        author: str = Field(description="the book's author")
        branch: str = Field(description="the branch that holds the book")
        copies: int = Field(description="how many copies the library owns")
        on_loan: int = Field(description="how many copies are on loan")

    plain_utterances = [
        "show this book",
        "which branch holds it",
        "give me the details of this book",
        "who wrote this one, and how many do we own",
        "what is this book's branch and copy count",
        "how many copies of this one are out right now",
        "describe the book we have open",
        "show me the record for this title",
    ]


class ResponseGenerator:
    def __call__(self, workflow: fastworkflow.Workflow, command: str,
                 input: None) -> Signature.Output:
        book: Book = workflow.command_context_for_response_generation
        return Signature.Output(book_uid=book.uid, title=book.title, author=book.author,
                                branch=book.branch, copies=book.copies,
                                on_loan=len(book.library.loans_of_book(book.uid)))

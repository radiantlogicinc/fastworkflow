"""list_loans: who has the open book on loan, and until when."""
from typing import Optional

import fastworkflow
from pydantic import BaseModel, Field

from ...application.library import Book, page


class Signature:
    """List the loans of the open book: each borrower and the due date."""

    class Input(BaseModel):
        start: Optional[int] = Field(default=None, description="the first row of the page")

    class Output(BaseModel):
        borrowers: list[str] = Field(description="each borrower's name, in row order")
        due_dates: list[str] = Field(description="each loan's due date, in row order")
        total: int = Field(description="how many loans the book has, on every page")

    plain_utterances = [
        "list this book's loans",
        "give me page two of this title's loans",
        "how many loans does this book have",
        "give me the full loan list for this title",
        "read out the borrowers and due dates of this book, row by row",
        "list the loans on this one, starting from row 11",
        "every loan this title has, please",
        "what loans are open on this book",
    ]


class ResponseGenerator:
    def __call__(self, workflow: fastworkflow.Workflow, command: str,
                 input: Signature.Input) -> Signature.Output:
        book: Book = workflow.record
        loans = book.library.loans_of_book(book.uid)
        rows = page(loans, input.start)
        return Signature.Output(borrowers=[m for m, _ in rows], due_dates=[d for _, d in rows],
                                total=len(loans))

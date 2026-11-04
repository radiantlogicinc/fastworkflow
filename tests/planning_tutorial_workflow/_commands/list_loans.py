"""list_loans: who has one book on loan, and until when."""
from typing import Optional

import fastworkflow
from pydantic import BaseModel, Field

from ..application import library


class Signature:
    """List the loans of one book: each borrower and the due date."""

    class Input(BaseModel):
        book_uid: str = Field(description="the book's uid, as find_book lists it")
        start: Optional[int] = Field(default=None, description="the first row of the page")

    class Output(BaseModel):
        borrowers: list[str] = Field(description="each borrower's name, in row order")
        due_dates: list[str] = Field(description="each loan's due date, in row order")
        total: int = Field(description="how many loans the book has, on every page")

    plain_utterances = ["who has borrowed book b1"]


class ResponseGenerator:
    def __call__(self, workflow: fastworkflow.Workflow, command: str, input: Signature.Input) -> Signature.Output:
        loans = library.loans_of_book(input.book_uid)
        rows = library.page(loans, input.start)
        output = Signature.Output(borrowers=[m for m, _ in rows], due_dates=[d for _, d in rows],
                                  total=len(loans))
        return output

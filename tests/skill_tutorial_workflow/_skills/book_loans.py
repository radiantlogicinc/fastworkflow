"""book_loans: the book the request names and who has it on loan.

Reuses book_by_title's phases by subclassing its PlanGenerator: __call__ places them and the
new loans phase in order, and answer extends book_by_title's answer with super().
"""
from pydantic import BaseModel, Field

from fastworkflow.skill import fact_from, listed

from .book_by_title import PlanGenerator as BookByTitle


class Signature:
    """Show the book whose title the request names and who has it on loan."""

    plain_utterances = ["who has borrowed Dune", "is anyone reading Emma right now"]

    class Input(BaseModel):
        title: str = Field(description="the book's title exactly as the request writes it")


class PlanGenerator(BookByTitle):
    def __call__(self, input):
        return [self.find, self.book, self.loans, self.answer]

    class Loans(BaseModel):
        borrowers: list[str] = fact_from("list_loans.borrowers")
        total: int = fact_from("list_loans.total")

    def loans(self, book=None) -> Loans:
        if book is None:
            return []
        return [f"list_loans book_uid={book.uid} (full list)"]

    def answer(self, input, book=None, loans=None):
        steps = super().answer(input, book)
        if loans is None:
            return steps
        return [*steps, f"say that it is on loan to "
                        f"{listed(loans.borrowers, loans.total, 'borrower')}"]

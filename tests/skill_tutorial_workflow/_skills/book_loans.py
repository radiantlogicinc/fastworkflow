"""book_loans: who has the book the request names on loan.

Reuses book_by_title's find and book phases by inheriting its PlanGenerator, adds a loans
phase and writes its own answer.
"""
from pydantic import BaseModel, Field

from fastworkflow.skill import fact_from, listed

from .book_by_title import PlanGenerator as BookByTitle


class Signature:
    """Say who has the book whose title the request names on loan."""

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
        if book is None or loans is None:
            return [f"say that no book matches {input.title}"]
        return [f"say that {book.name} is on loan to "
                f"{listed(loans.borrowers, loans.total, 'borrower')}"]

"""book_loans: the book the request names and who has it on loan.

Reuses the phases book_by_title shares (_book_phases.BookPhases): __phases__ places them and the
new loans phase in order, and __result__ extends the shared answer with super().
"""
from pydantic import BaseModel, Field

from fastworkflow.planning import fact_from, full_list, listed

from ._book_phases import BookPhases


class Signature:
    """Show the book whose title the request names and who has it on loan."""

    plain_utterances = ["who has borrowed Dune", "is anyone reading Emma right now"]

    class Input(BaseModel):
        title: str = Field(description="the book's title",
                           json_schema_extra={"in_request": True})


class PlanGenerator(BookPhases):
    def __phases__(self, input):
        return [self.find, self.book, self.loans]

    class Loans(BaseModel):
        borrowers: list[str] = fact_from("list_loans.borrowers")
        total: int = fact_from("list_loans.total")

    def loans(self, book=None) -> Loans:
        if book is None:
            return []
        return [full_list("list_loans", book_uid=book.uid)]

    def __result__(self, input, book=None, loans=None):
        answer = super().__result__(input, book)
        if loans is None:
            return answer
        return (f"{answer}\nIt is on loan to "
                f"{listed(loans.borrowers, loans.total, 'borrower')}.")

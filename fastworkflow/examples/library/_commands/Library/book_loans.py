"""book_loans: who has the book the request names on loan.

book calls book_by_title, a planning command, as one step, and hands its Output on; loans
lists the loans of the book that call left open, and is skipped when it found none;
__result__ names the borrowers.
"""
from pydantic import BaseModel, Field

from fastworkflow.planning import count_and_names_read, run

from ..Book.list_loans import Signature as ListLoans
from .book_by_title import Signature as BookByTitle


class Signature:
    """Say who has a book on loan, given its title."""

    plain_utterances = [
        "who has borrowed Dune",
        "is anyone reading Emma right now",
        "who is reading Persuasion at the moment",
        "which patrons have Dune Messiah out",
        "is anyone holding a copy of Emma",
        "names of everyone who borrowed Persuasion",
        "has anybody got Dune out, and until when",
        "who currently holds the title Dune Messiah",
    ]

    class Input(BaseModel):
        title: str = Field(description="the book's title",
                           json_schema_extra={"copied_from_command_text": True})


class PlanGenerator:
    def book(self, input) -> BookByTitle.Output:
        return [run("book_by_title", title=input.title)]

    def loans(self, book) -> ListLoans.Output:
        if not book.found:
            return []
        return [run("list_loans", all_pages=True)]

    def __result__(self, input, book, loans):
        if loans is None:
            return f"No book matches {input.title}."
        return f"{book.title} is on loan to {count_and_names_read(loans.borrowers, loans.total, 'borrower')}."

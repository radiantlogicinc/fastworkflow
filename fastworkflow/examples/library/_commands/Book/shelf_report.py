"""shelf_report: where the open book is held, and who has it on loan.

A planning command of the Book context. Its one phase runs show_book and list_loans on the
open book, and hands on both Outputs in a facts model with one field per command;
__result__ reads both.
"""
from pydantic import BaseModel

from fastworkflow.planning import count, full_list, listed, run

from .list_loans import Signature as ListLoans
from .show_book import Signature as ShowBook


class Signature:
    """Say where this book is held, how many copies there are, and who has it on loan."""

    plain_utterances = ["where is this book kept, and who has it", "is this book in"]


class Shelf(BaseModel):
    book: ShowBook.Output
    loans: ListLoans.Output


class PlanGenerator:
    def shelf(self) -> Shelf:
        return [run("show_book"), full_list("list_loans")]

    def __result__(self, shelf):
        book, loans = shelf.book, shelf.loans
        held = f"{book.title} is held at {book.branch}, {count(book.copies, 'copy', 'copies')}"
        if not loans.borrowers:
            return f"{held}; none is on loan."
        return f"{held}; on loan to {listed(loans.borrowers, loans.total, 'borrower')}."

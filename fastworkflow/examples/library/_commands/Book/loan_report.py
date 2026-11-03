"""loan_report: who has the open book on loan, and until when.

A planning command of the Book context: it is found only while the workflow stands on a
book. loans lists the book's loans; __result__ names each borrower with the due date.
"""
from fastworkflow.planning import full_list

from .list_loans import Signature as ListLoans


class Signature:
    """Say who has this book on loan and when each loan is due back."""

    plain_utterances = [
        "who has this book",
        "when is it due back",
        "is this book out, and until when",
        "who is holding this one right now and what's the return date",
        "tell me who borrowed this book and when it comes back",
        "is this one on loan to anyone",
        "when will this copy be returned, and to whom",
        "who's got this book at the moment",
    ]


class PlanGenerator:
    def loans(self) -> ListLoans.Output:
        return [full_list("list_loans")]

    def __result__(self, loans):
        if not loans.borrowers:
            return "No one has it on loan."
        due = (f"{b} has it until {d}" for b, d in zip(loans.borrowers, loans.due_dates))
        return "; ".join(due) + "."

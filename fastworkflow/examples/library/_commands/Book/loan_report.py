"""loan_report: who has the open book on loan, and until when.

A planning command of the Book context: it is found only while the workflow stands on a
book. loans lists the book's loans; __result__ names each borrower with the due date.
"""
from fastworkflow.planning import full_list

from .list_loans import Signature as ListLoans


class Signature:
    """Say who has this book on loan and when each loan is due back."""

    plain_utterances = ["who has this book", "when is it due back"]


class PlanGenerator:
    def loans(self) -> ListLoans.Output:
        return [full_list("list_loans")]

    def __result__(self, loans):
        if not loans.borrowers:
            return "No one has it on loan."
        due = (f"{b} has it until {d}" for b, d in zip(loans.borrowers, loans.due_dates))
        return "; ".join(due) + "."

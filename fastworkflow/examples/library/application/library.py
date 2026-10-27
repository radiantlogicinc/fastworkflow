"""A small in-memory library: the application behind the planning-command tutorial.

Two branches, four books, three members and their loans. A Book and a Member each hold
the Library they belong to, which is how the workflow climbs from a record to the library
(``Context.get_parent``). Notes are the only thing the workflow writes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

PAGE_SIZE = 10

#: Opening hours by branch.
HOURS = {
    "Central": "9:00-18:00, Monday to Saturday",
    "Riverside": "10:00-16:00, Tuesday to Saturday",
}


def page(rows: list, start: int | None) -> list:
    begin = start or 0
    return rows[begin:begin + PAGE_SIZE]


@dataclass(eq=False)
class Book:
    library: Library = field(repr=False)
    uid: str
    title: str
    author: str
    copies: int
    branch: str
    notes: list[str] = field(default_factory=list)


@dataclass(eq=False)
class Member:
    library: Library = field(repr=False)
    uid: str
    name: str
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Loan:
    book_uid: str
    member_uid: str
    due: str
    overdue: bool


class Library:
    """The root object of the workflow: every book, member and loan."""

    def __init__(self) -> None:
        self.branches = list(HOURS)
        self.books = {book.uid: book for book in [
            Book(self, "b1", "Dune", "Frank Herbert", 3, "Central",
                 notes=["one copy is a large-print edition"]),
            Book(self, "b2", "Dune Messiah", "Frank Herbert", 1, "Central"),
            Book(self, "b3", "Emma", "Jane Austen", 2, "Riverside"),
            Book(self, "b4", "Persuasion", "Jane Austen", 1, "Riverside"),
        ]}
        self.members = {member.uid: member for member in [
            Member(self, "m1", "Ann Lee"),
            Member(self, "m2", "Raj Patel", notes=["prefers email reminders"]),
            Member(self, "m3", "Mia Chen"),
        ]}
        self.loans = [
            Loan("b1", "m1", "2026-11-02", overdue=False),
            Loan("b1", "m2", "2026-10-20", overdue=True),
            Loan("b3", "m2", "2026-11-05", overdue=False),
            Loan("b4", "m3", "2026-10-18", overdue=True),
        ]

    def find_books(self, query: str) -> list[str]:
        """Uids of the books whose title or author holds the query (case ignored)."""
        q = query.casefold()
        return [uid for uid, book in self.books.items()
                if q in book.title.casefold() or q in book.author.casefold()]

    def find_members(self, query: str) -> list[str]:
        """Uids of the members whose name holds the query (case ignored)."""
        q = query.casefold()
        return [uid for uid, member in self.members.items() if q in member.name.casefold()]

    def loans_of_book(self, book_uid: str) -> list[tuple[str, str]]:
        """(member name, due date) of each loan of the book."""
        return [(self.members[loan.member_uid].name, loan.due)
                for loan in self.loans if loan.book_uid == book_uid]

    def loans_of_member(self, member_uid: str) -> list[tuple[str, str]]:
        """(book title, due date) of each loan of the member."""
        return [(self.books[loan.book_uid].title, loan.due)
                for loan in self.loans if loan.member_uid == member_uid]

    def overdue_count(self, branch: str = "") -> int:
        """Overdue loans in one branch (case ignored), or in every branch when empty."""
        return sum(1 for loan in self.loans if loan.overdue and (
            not branch or self.books[loan.book_uid].branch.casefold() == branch.casefold()))

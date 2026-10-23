"""A tiny in-memory library: the application behind the skill tutorials' commands.

Two branches, four books, three members and their loans. Every command of the tutorial
workflow reads this module; nothing is written.
"""
from __future__ import annotations

PAGE_SIZE = 10

BRANCHES = ["Central", "Riverside"]

BOOKS = {
    "b1": {"title": "Dune", "author": "Frank Herbert", "copies": 3, "branch": "Central"},
    "b2": {"title": "Dune Messiah", "author": "Frank Herbert", "copies": 1, "branch": "Central"},
    "b3": {"title": "Emma", "author": "Jane Austen", "copies": 2, "branch": "Riverside"},
    "b4": {"title": "Persuasion", "author": "Jane Austen", "copies": 1, "branch": "Riverside"},
}

MEMBERS = {"m1": "Ann Lee", "m2": "Raj Patel", "m3": "Mia Chen"}

#: (book uid, member uid, due date, overdue)
LOANS = [
    ("b1", "m1", "2026-11-02", False),
    ("b1", "m2", "2026-10-20", True),
    ("b3", "m2", "2026-11-05", False),
    ("b4", "m3", "2026-10-18", True),
]


def page(rows: list, start: int | None) -> list:
    begin = start or 0
    return rows[begin:begin + PAGE_SIZE]


def find_books(query: str) -> list[str]:
    """Uids of the books whose title or author holds the query (case ignored)."""
    q = query.casefold()
    return [uid for uid, book in BOOKS.items()
            if q in book["title"].casefold() or q in book["author"].casefold()]


def find_members(query: str) -> list[str]:
    q = query.casefold()
    return [uid for uid, name in MEMBERS.items() if q in name.casefold()]


def loans_of_book(book_uid: str) -> list[tuple[str, str]]:
    """(member name, due date) of each loan of the book."""
    return [(MEMBERS[m], due) for b, m, due, _ in LOANS if b == book_uid]


def loans_of_member(member_uid: str) -> list[tuple[str, str]]:
    """(book title, due date) of each loan of the member."""
    return [(BOOKS[b]["title"], due) for b, m, due, _ in LOANS if m == member_uid]


def overdue_count(branch: str = "") -> int:
    return sum(1 for b, _, _, overdue in LOANS
               if overdue and (not branch or BOOKS[b]["branch"].casefold() == branch.casefold()))

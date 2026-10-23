"""book_report: the book the request names, and who has it on loan when asked.

With brief, one plan: find the book and say whether the library has it. Otherwise find,
book (skipped when nothing matched), loans (only when the request asks who has it, and
skipped when no book was opened) and answer.
"""
from pydantic import BaseModel, Field

from fastworkflow.skill import fact_from, listed, pick


class Signature:
    """Show the book whose title the request names: its author and copies, and who has it on
    loan when asked. With brief, only whether the library has it."""

    plain_utterances = ["tell me about Dune and who has it", "do you have Emma at all"]

    class Input(BaseModel):
        title: str = Field(description="the book's title exactly as the request writes it")
        loans: bool = Field(default=False,
                            description="the request asks who has the book on loan")
        brief: bool = Field(default=False,
                            description="the request only asks whether the library has the book")


class PlanGenerator:
    def __call__(self, input):
        if input.brief:
            return [self.check]
        loans = [self.loans] if input.loans else []
        return [self.find, self.book, *loans, self.answer]

    def check(self, input):
        return [f"find_book query={input.title}",
                "say whether the library has the book"]

    class Found(BaseModel):
        book_uids: list[str] = fact_from("find_book.book_uids")
        titles: list[str] = fact_from("find_book.titles")

        def only_uid(self, title):
            """The uid of the one book titled *title* (case ignored), else None."""
            hits = [uid for uid, t in zip(self.book_uids, self.titles)
                    if t.casefold() == title.casefold()]
            return hits[0] if len(hits) == 1 else None

    def find(self, input) -> Found:
        return [f"find_book query={input.title}"]

    class Book(BaseModel):
        uid: str = fact_from("show_book.book_uid")
        name: str = fact_from("show_book.title")
        author: str = fact_from("show_book.author")
        copies: int = fact_from("show_book.copies")

    def book(self, input, find) -> Book:
        if not find.book_uids:
            return []
        return [pick("show_book", book_uid=find.only_uid(input.title), named=input.title)]

    class Loans(BaseModel):
        borrowers: list[str] = fact_from("list_loans.borrowers")
        total: int = fact_from("list_loans.total")

    def loans(self, book=None) -> Loans:
        if book is None:
            return []
        return [f"list_loans book_uid={book.uid} (full list)"]

    def answer(self, input, book=None, loans=None):
        if book is None:
            return [f"say that no book matches {input.title}"]
        lines = [f"say that {book.name} is by {book.author}; the library owns "
                 f"{book.copies} copies"]
        if loans is not None:
            lines.append(f"say that it is on loan to "
                         f"{listed(loans.borrowers, loans.total, 'borrower')}")
        return lines

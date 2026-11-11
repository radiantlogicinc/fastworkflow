"""book_report: the book the request names, and who has it on loan when asked.

With brief, one phase: find the book, and __result__ says whether the library has it. Otherwise
find, book (skipped when nothing matched), loans (only when the request asks who has it, and
skipped when no book was opened) and __result__ writes the answer.
"""
from pydantic import BaseModel, Field

from fastworkflow.planning import count_and_names_read, run

from .find_book import Signature as FindBook
from .list_loans import Signature as ListLoans
from .show_book import Signature as ShowBook


class Signature:
    """Show the book whose title the request names: its author and copies, and who has it on
    loan when asked. With brief, only whether the library has it."""

    plain_utterances = ["tell me about Dune and who has it", "do you have Emma at all"]

    class Input(BaseModel):
        title: str = Field(description="the book's title")
        loans: bool = Field(default=False,
                            description="the request asks who has the book on loan")
        brief: bool = Field(default=False,
                            description="the request only asks whether the library has the book")


class PlanGenerator:
    def __phases__(self, input):
        if input.brief:
            return [self.find]
        loans = [self.loans] if input.loans else []
        return [self.find, self.book, *loans]

    def find(self, input) -> FindBook.Output:
        return [run("find_book", query=input.title)]

    class Book(BaseModel):
        show: ShowBook.Output

        @property
        def uid(self) -> str:
            return self.show.book_uid

        @property
        def name(self) -> str:
            return self.show.title

        @property
        def author(self) -> str:
            return self.show.author

        @property
        def copies(self) -> int:
            return self.show.copies

    def book(self, input, find) -> Book:
        if not find.book_uids:
            return []
        return [run("show_book", named=input.title)]

    def loans(self, book=None) -> ListLoans.Output:
        if book is None:
            return []
        return [run("list_loans", book_uid=book.uid, all_pages=True)]

    def __result__(self, input, find=None, book=None, loans=None):
        if input.brief:
            if find.book_uids:
                return f"Yes, the library has {input.title}."
            return f"No, the library does not have {input.title}."
        if book is None:
            return f"No book matches {input.title}."
        lines = [f"{book.name} is by {book.author}; the library owns {book.copies} copies."]
        if loans is not None:
            lines.append(f"It is on loan to {count_and_names_read(loans.borrowers, loans.total, 'borrower')}.")
        return "\n".join(lines)

"""_book_phases: the phases book_by_title and book_loans share.

find lists the matches; book opens the one whose title is the request's title (run(..., named=...)), and
is skipped when nothing matched; __result__ says what book found, or that no book matches. A
command that subclasses BookPhases keeps these phases and adds its own (book_loans adds loans).
"""
from pydantic import BaseModel

from fastworkflow.planning import run

from .find_book import Signature as FindBook
from .show_book import Signature as ShowBook


class BookPhases:
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

    def __result__(self, input, book=None):
        if book is None:
            return f"No book matches {input.title}."
        return (f"{book.name} is by {book.author}; the library owns "
                f"{book.copies} copies.")

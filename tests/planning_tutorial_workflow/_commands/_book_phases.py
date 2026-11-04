"""_book_phases: the phases book_by_title and book_loans share.

find lists the matches; book opens the one whose title is the request's title (pick), and
is skipped when nothing matched; __result__ says what book found, or that no book matches. A
command that subclasses BookPhases keeps these phases and adds its own (book_loans adds loans).
"""
from pydantic import BaseModel

from fastworkflow.planning import fact_from, pick, run


class BookPhases:
    class Found(BaseModel):
        book_uids: list[str] = fact_from("find_book.book_uids")
        titles: list[str] = fact_from("find_book.titles")

    def find(self, input) -> Found:
        return [run("find_book", query=input.title)]

    class Book(BaseModel):
        uid: str = fact_from("show_book.book_uid")
        name: str = fact_from("show_book.title")
        author: str = fact_from("show_book.author")
        copies: int = fact_from("show_book.copies")

    def book(self, input, find) -> Book:
        if not find.book_uids:
            return []
        return [pick("show_book", named=input.title)]

    def __result__(self, input, book=None):
        if book is None:
            return f"No book matches {input.title}."
        return (f"{book.name} is by {book.author}; the library owns "
                f"{book.copies} copies.")

"""author_books: each book by the author the request names, opened one by one.

find lists the author's books; book runs once per book found (at most five), each its own
sub-plan; answer names every book with its copies.
"""
from pydantic import BaseModel, Field

from fastworkflow.skill import count, fact_from, phase


class Signature:
    """Name each book by the author the request names, with how many copies the library owns."""

    plain_utterances = ["what do you have by Frank Herbert, and how many copies of each"]

    class Input(BaseModel):
        author: str = Field(description="the author's name exactly as the request writes it")


class PlanGenerator:
    class Found(BaseModel):
        book_uids: list[str] = fact_from("find_book.book_uids")

    def find(self, input) -> Found:
        return [f"find_book query={input.author}"]

    class Book(BaseModel):
        name: str = fact_from("show_book.title")
        copies: int = fact_from("show_book.copies")

    @phase(over="find.book_uids", max_items=5, noun="books", bind=False)
    def book(self, uid) -> Book:
        return [f"show_book book_uid={uid}"]

    def answer(self, input, book=None):
        return [f"say that {self._books_by(input.author, book)}"]

    def _books_by(self, author, book):
        if not book:
            return f"no book by {author} was found"
        copies = "; ".join(f"{b.name} ({count(b.copies, 'copy', 'copies')})" for b in book)
        return f"the books by {author}: {copies}"

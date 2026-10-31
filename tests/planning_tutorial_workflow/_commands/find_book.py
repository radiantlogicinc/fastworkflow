"""find_book: the books whose title or author holds a query."""
from typing import Optional

import fastworkflow
from pydantic import BaseModel, Field

from ..application import library


class Signature:
    """Find books by title or author."""

    class Input(BaseModel):
        query: str = Field(description="words of the title or the author's name")
        start: Optional[int] = Field(default=None, description="the first row of the page")

    class Output(BaseModel):
        book_uids: list[str] = Field(description="the uid of each book found, in row order")
        titles: list[str] = Field(description="the title of each book found, in row order")
        total: int = Field(description="how many books match, on every page")

    plain_utterances = ["find the book Dune", "look up books by Jane Austen"]


class ResponseGenerator:
    def __call__(self, workflow: fastworkflow.Workflow, command: str,
                 command_parameters: Signature.Input) -> fastworkflow.CommandOutput:
        found = library.find_books(command_parameters.query)
        rows = library.page(found, command_parameters.start)
        output = Signature.Output(book_uids=rows, titles=[library.BOOKS[u]["title"] for u in rows],
                                  total=len(found))
        listing = ", ".join(f"{t} ({u})" for u, t in zip(output.book_uids, output.titles))
        return fastworkflow.CommandOutput(command_response=fastworkflow.CommandResponse(
            response=f"{output.total} books: {listing or 'none'}", output=output))

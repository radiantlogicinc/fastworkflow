"""find_books: the books whose title or author holds the words the request gives."""
from pydantic import BaseModel, Field


class Signature:
    """Find the books whose title or author holds the request's words, and name them."""

    plain_utterances = ["find books by Jane Austen", "do you have Dune"]

    class Input(BaseModel):
        query: str = Field(description="the title or author words exactly as the request "
                                       "writes them")


class PlanGenerator:
    def search(self, input):
        return [f"find_book query={input.query}",
                "say which books were found"]

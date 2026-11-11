"""book_by_title: the book whose title the request names, picked by its title.

find lists the matches; book opens the one whose title is the request's title (run(..., named=...)), and
is skipped when nothing matched; __result__ says what book found, or that no book matches.
"""
from pydantic import BaseModel, Field

from ._book_phases import BookPhases


class Signature:
    """Show the author and copies of the book whose title the request names."""

    plain_utterances = ["who wrote Dune", "how many copies of Dune Messiah are there"]

    class Input(BaseModel):
        title: str = Field(description="the book's title",
                           json_schema_extra={"copied_from_command_text": True})


class PlanGenerator(BookPhases):
    """find, book and __result__ are BookPhases' phases; this command is their whole job."""

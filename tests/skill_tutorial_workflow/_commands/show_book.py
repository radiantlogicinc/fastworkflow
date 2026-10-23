"""show_book: one book's details, by its uid."""
import fastworkflow
from pydantic import BaseModel, Field

from ..application import library


class Signature:
    """Show one book: its title, author, copies and how many are on loan."""

    class Input(BaseModel):
        book_uid: str = Field(description="the book's uid, as find_book lists it")

    class Output(BaseModel):
        book_uid: str = Field(description="the book's uid")
        title: str = Field(description="the book's title")
        author: str = Field(description="the book's author")
        copies: int = Field(description="how many copies the library owns")
        on_loan: int = Field(description="how many copies are on loan")

    plain_utterances = ["show book b1"]


class ResponseGenerator:
    def __call__(self, workflow: fastworkflow.Workflow, command: str,
                 command_parameters: Signature.Input) -> fastworkflow.CommandOutput:
        uid = command_parameters.book_uid
        book = library.BOOKS.get(uid)
        if book is None:
            raise fastworkflow.Refused(f"no book has the uid {uid}; find_book lists the uids")
        output = Signature.Output(book_uid=uid, title=book["title"], author=book["author"],
                                  copies=book["copies"], on_loan=len(library.loans_of_book(uid)))
        return fastworkflow.CommandOutput(command_response=fastworkflow.CommandResponse(
            response=output.model_dump_json(), output=output))

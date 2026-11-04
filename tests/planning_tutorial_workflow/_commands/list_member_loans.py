"""list_member_loans: the books one member has on loan."""
from typing import Optional

import fastworkflow
from pydantic import BaseModel, Field

from ..application import library


class Signature:
    """List the books one member has on loan, with their due dates."""

    class Input(BaseModel):
        member_uid: str = Field(description="the member's uid, as find_member lists it")
        start: Optional[int] = Field(default=None, description="the first row of the page")

    class Output(BaseModel):
        titles: list[str] = Field(description="each borrowed book's title, in row order")
        due_dates: list[str] = Field(description="each loan's due date, in row order")
        total: int = Field(description="how many books the member has on loan, on every page")

    plain_utterances = ["what has member m1 borrowed"]


class ResponseGenerator:
    def __call__(self, workflow: fastworkflow.Workflow, command: str, input: Signature.Input) -> Signature.Output:
        loans = library.loans_of_member(input.member_uid)
        rows = library.page(loans, input.start)
        output = Signature.Output(titles=[t for t, _ in rows], due_dates=[d for _, d in rows],
                                  total=len(loans))
        return output

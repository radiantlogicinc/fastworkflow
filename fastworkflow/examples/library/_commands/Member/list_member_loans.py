"""list_member_loans: the books the open member has on loan."""
from typing import Optional

import fastworkflow
from pydantic import BaseModel, Field

from ...application.library import Member, page


class Signature:
    """List the books the open member has on loan, with their due dates."""

    class Input(BaseModel):
        start: Optional[int] = Field(default=None, description="the first row of the page")

    class Output(BaseModel):
        titles: list[str] = Field(description="each borrowed book's title, in row order")
        due_dates: list[str] = Field(description="each loan's due date, in row order")
        total: int = Field(description="how many books the member has on loan, on every page")

    plain_utterances = ["what has this member borrowed"]


class ResponseGenerator:
    def __call__(self, workflow: fastworkflow.Workflow, command: str,
                 command_parameters: Signature.Input) -> Signature.Output:
        member: Member = workflow.command_context_for_response_generation
        loans = member.library.loans_of_member(member.uid)
        rows = page(loans, command_parameters.start)
        return Signature.Output(titles=[t for t, _ in rows], due_dates=[d for _, d in rows],
                                total=len(loans))

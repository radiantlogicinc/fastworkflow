"""open_member: make one member, by their uid, the current record."""
import fastworkflow
from fastworkflow import from_
from pydantic import BaseModel, Field

from ...application.library import Library


class Signature:
    """Open one member by their uid, so that the member's own commands act on them."""

    class Input(BaseModel):
        member_uid: str = from_("find_member.member_uids",
                                description="the uid of the member to open")

    class Output(BaseModel):
        member_uid: str = Field(description="the uid of the member now open")
        name: str = Field(description="the member's name")
        on_loan: int = Field(description="how many books the member has on loan")

    plain_utterances = [
        "open member m1",
        "go to Ann Lee's record",
        "make patron m3 the open record",
        "set patron m2 as the current record",
        "select m3 so I can act on the patron",
        "I want to work on Raj Patel's account now",
        "pull up Mia Chen's record",
        "load uid m2 as the open member",
    ]


class ResponseGenerator:
    descends_to = "Member"

    def __call__(self, workflow: fastworkflow.Workflow, command: str,
                 input: Signature.Input) -> Signature.Output:
        library: Library = workflow.command_context_for_response_generation
        member = library.members.get(input.member_uid)
        if member is None:
            raise fastworkflow.Refused(
                f"no member has the uid {input.member_uid}; find_member lists the uids")
        workflow.current_command_context = member
        return Signature.Output(member_uid=member.uid, name=member.name,
                                on_loan=len(library.loans_of_member(member.uid)))

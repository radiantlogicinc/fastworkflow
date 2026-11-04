"""show_member: the open member's details."""
import fastworkflow
from pydantic import BaseModel, Field

from ...application.library import Member


class Signature:
    """Show the open member: their name and how many books they have on loan."""

    class Output(BaseModel):
        member_uid: str = Field(description="the member's uid")
        name: str = Field(description="the member's name")
        on_loan: int = Field(description="how many books the member has on loan")

    plain_utterances = [
        "show this member",
        "show me this patron's details",
        "who is this member, and how many books do they have out",
        "give me their name and loan count",
        "what's this member's name",
        "display the open member's card",
        "show the member record I'm on",
        "tell me about the member we have open",
    ]


class ResponseGenerator:
    def __call__(self, workflow: fastworkflow.Workflow, command: str,
                 input: None) -> Signature.Output:
        member: Member = workflow.command_context_for_response_generation
        return Signature.Output(member_uid=member.uid, name=member.name,
                                on_loan=len(member.library.loans_of_member(member.uid)))

"""member_loans: the books one named member has on loan.

A request about every member is refused before any step runs. find lists the members by
name; member reads the loans of the one picked by name (skipped when nothing matched);
__result__ names the books.
"""
import fastworkflow
from pydantic import BaseModel, Field

from fastworkflow.planning import listed, pick, run

from .find_member import Signature as FindMember
from .list_member_loans import Signature as ListMemberLoans


class Signature:
    """List the books one named member has on loan. One job per member."""

    plain_utterances = ["what has Ann Lee borrowed", "which books does Raj Patel have out"]

    class Input(BaseModel):
        member: str = Field(description="the member's name")

    @staticmethod
    def validate_extracted_parameters(workflow, command, input, command_text=None):
        if input.member.casefold() in EVERYONE:
            raise fastworkflow.Refused(
                f"member is {input.member!r}, which names no one member; this skill "
                "reads one member's loans: give that member's name", field="member")
        return None


class PlanGenerator:
    def find(self, input) -> FindMember.Output:
        return [run("find_member", query=input.member)]

    def member(self, input, find) -> ListMemberLoans.Output:
        if not find.member_uids:
            return []
        return [pick("list_member_loans", named=input.member)]

    def __result__(self, input, member=None):
        if member is None:
            return f"No member is named {input.member}."
        return f"{input.member} has {listed(member.titles, member.total, 'book')} on loan."


#: Words that name every member, not one.
EVERYONE = {"everyone", "everybody", "all", "all members", "every member"}

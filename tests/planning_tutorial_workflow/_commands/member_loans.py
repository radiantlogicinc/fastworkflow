"""member_loans: the books one named member has on loan.

A request about every member is refused before any step runs. find lists the members by
name; member reads the loans of the one picked by name (skipped when nothing matched);
answer names the books.
"""
import fastworkflow
from pydantic import BaseModel, Field

from fastworkflow.planning import fact_from, listed, pick, run


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
    class Found(BaseModel):
        member_uids: list[str] = fact_from("find_member.member_uids")
        names: list[str] = fact_from("find_member.names")

    def find(self, input) -> Found:
        return [run("find_member", query=input.member)]

    class Loans(BaseModel):
        titles: list[str] = fact_from("list_member_loans.titles")
        total: int = fact_from("list_member_loans.total")

    def member(self, input, find) -> Loans:
        if not find.member_uids:
            return []
        return [pick("list_member_loans", named=input.member)]

    def answer(self, input, member=None):
        if member is None:
            return [f"say that no member is named {input.member}"]
        return [f"say that {input.member} has {listed(member.titles, member.total, 'book')} "
                "on loan"]


#: Words that name every member, not one.
EVERYONE = {"everyone", "everybody", "all", "all members", "every member"}

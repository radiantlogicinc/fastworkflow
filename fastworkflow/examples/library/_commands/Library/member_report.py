"""member_report: the member the request names, with their loans or notes when asked.

find lists the members by name; member opens the one named (skipped when nothing
matched); loans and notes run only when the request asks for them, on the member that
member opened; __result__ says what the phases found. A request about every member is
refused before any step runs.
"""
import fastworkflow
from pydantic import BaseModel, Field, model_validator

from fastworkflow.planning import count, count_and_names_read, run

from ..Member.list_member_loans import Signature as ListMemberLoans
from ..Record.show_notes import Signature as ShowNotes
from .find_member import Signature as FindMember
from .open_member import Signature as OpenMember


class Signature:
    """Show a member's loans, and the notes on their record when asked. One member per call."""

    plain_utterances = [
        "tell me about Raj Patel",
        "what has Ann Lee borrowed",
        "any notes on Mia Chen's record",
        "give me the summary for Ann Lee, loans and notes included",
        "how many books does Raj Patel have out, and what does his record say",
        "report on Mia Chen's account",
        "what is on Ann Lee's record, including her loans",
        "full picture for Mia Chen please",
    ]

    class Input(BaseModel):
        member: str = Field(description="the member's name",
                            json_schema_extra={"copied_from_command_text": True})
        loans: bool = Field(default=False,
                            description="whether to name the books the member has on loan")
        notes: bool = Field(default=False,
                            description="whether to show the notes on the member's record")

        @model_validator(mode="after")
        def refuse_everyone(self, info):
            if self.member.casefold() in EVERYONE:
                raise fastworkflow.Refused(
                    f"member is {self.member!r}, which names no one member; this "
                    "command reports on one member: give that member's name", field="member")
            return self


class PlanGenerator:
    def __phases__(self, input):
        loans = [self.loans] if input.loans else []
        notes = [self.notes] if input.notes else []
        return [self.find, self.member, *loans, *notes]

    def find(self, input) -> FindMember.Output:
        return [run("find_member", query=input.member)]

    def member(self, input, find) -> OpenMember.Output:
        if not find.member_uids:
            return []
        return [run("open_member", named=input.member)]

    def loans(self, member) -> ListMemberLoans.Output:
        if member is None:
            return []
        return [run("list_member_loans", all_pages=True)]

    def notes(self, member) -> ShowNotes.Output:
        if member is None:
            return []
        return [run("show_notes")]

    def __result__(self, input, member, loans, notes):
        if member is None:
            return f"No member is named {input.member}."
        on_loan = (count(member.on_loan, "book") if loans is None
                   else count_and_names_read(loans.titles, loans.total, "book"))
        lines = [f"{member.name} has {on_loan} on loan."]
        if notes is not None:
            lines.append(f"{member.name}'s record holds "
                         f"{count_and_names_read(notes.notes, notes.total, 'note')}.")
        return "\n".join(lines)


#: Words that name every member, not one.
EVERYONE = {"everyone", "everybody", "all", "all members", "every member"}

# 9. Refusing a call

**Goal:** "what has Raj Patel borrowed?" lists one member's loans. "What has everyone
borrowed?" is not this skill's job, and the planner should be told so before any step runs.

## The skill file

<!-- file: tests/skill_tutorial_workflow/_skills/member_loans.py -->
```python
"""member_loans: the books one named member has on loan.

A request about every member is refused before any step runs. find lists the members by
name; member reads the loans of the one picked by name (skipped when nothing matched);
answer names the books.
"""
import fastworkflow
from pydantic import BaseModel, Field

from fastworkflow.skill import fact_from, listed, pick


class Signature:
    """List the books one named member has on loan. One job per member."""

    plain_utterances = ["what has Ann Lee borrowed", "which books does Raj Patel have out"]

    class Input(BaseModel):
        member: str = Field(description="the member's name exactly as the request writes it")

    @staticmethod
    def validate_extracted_parameters(workflow, command, cmd_parameters, command_text=None):
        if cmd_parameters.member.casefold() in EVERYONE:
            raise fastworkflow.Refused(
                f"member is {cmd_parameters.member!r}, which names no one member; this skill "
                "reads one member's loans: give that member's name")
        return True, ""


class PlanGenerator:
    class Found(BaseModel):
        member_uids: list[str] = fact_from("find_member.member_uids")
        names: list[str] = fact_from("find_member.names")

        def only_uid(self, name):
            """The uid of the one member named *name* (case ignored), else None."""
            hits = [uid for uid, n in zip(self.member_uids, self.names)
                    if n.casefold() == name.casefold()]
            return hits[0] if len(hits) == 1 else None

    def find(self, input) -> Found:
        return [f"find_member query={input.member}"]

    class Loans(BaseModel):
        titles: list[str] = fact_from("list_member_loans.titles")
        total: int = fact_from("list_member_loans.total")

    def member(self, input, find) -> Loans:
        if not find.member_uids:
            return []
        return [pick("list_member_loans", member_uid=find.only_uid(input.member),
                     named=input.member)]

    def answer(self, input, member=None):
        if member is None:
            return [f"say that no member is named {input.member}"]
        return [f"say that {input.member} has {listed(member.titles, member.total, 'book')} "
                "on loan"]


#: Words that name every member, not one.
EVERYONE = {"everyone", "everybody", "all", "all members", "every member"}
```

## Line by line

- **`validate_extracted_parameters`** is the hook a command's `Signature` has, with the
  same signature. It gets the checked arguments as an `Input` (`cmd_parameters`) after the
  planner filled them, before the `PlanGenerator` is called.
- **`raise fastworkflow.Refused(message)`** refuses the call. The message goes back to the
  planner, which fills the call once more; a second refusal drops the skill call and the
  planner plans from the template. Write the message as what to fix. Returning
  `(False, message)` refuses the same way; `(True, "")` accepts.
- Any other exception the hook raises fails the call (the planner falls back to the
  template); `Refused` is the one that means "these arguments are wrong".
- A rule that ties fields together, or a field to the request, can instead be a pydantic
  validator on `Input` raising `ValueError` (see "Rules on the arguments" in
  [skill_file_layout.md](../../skill_file_layout.md)).
- **`EVERYONE`** is a module-level table, written after `PlanGenerator` as the layout
  asks. The hook reads it when it is called, so the order does not matter to Python.
- **`pick("list_member_loans", member_uid=...)`**: `pick` works on any command whose one
  unwritten parameter is a uid.

## What the planner sees

<!-- template: member_loans -->
```text
find_member query=<member>
list_member_loans with the uid of the <member> match (only when find.member_uids)
say that <member> has <member.total> books (<member.titles>) on loan
```

The template is rendered from placeholder inputs that go through the hook too: a
combination the hook refuses is left out of the template.

## At run time

A refused call never reaches the `PlanGenerator`:

<!-- refusal: member_loans {"member": "everyone"} -->
```text
validate_extracted_parameters: member is 'everyone', which names no one member; this skill reads one member's loans: give that member's name
```

A call it accepts:

<!-- dry-run: member_loans {"member": "Raj Patel"} -->
```text
Phase find
  1. find_member query=Raj Patel
  code runs: find_member query=Raj Patel
  facts: Found(member_uids=['m2'], names=['Raj Patel'])
Phase member
  1. list_member_loans member_uid=m2
  code runs: list_member_loans member_uid=m2
  facts: Loans(titles=['Dune', 'Emma'], total=2)
Phase answer
  1. say that Raj Patel has 2 books (Dune, Emma) on loan
  answer written by code:
    Raj Patel has 2 books (Dune, Emma) on loan.
```

## What you learned

`validate_extracted_parameters` checks a call's arguments; `raise fastworkflow.Refused(...)`
sends the planner back with what to fix.

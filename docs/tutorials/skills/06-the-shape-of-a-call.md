# 6. The shape of a call

**Goal:** one command for three kinds of request about a member:

- "tell me about Raj Patel": find, open, answer;
- "what has Ann Lee borrowed": the same, plus a phase that lists the loans;
- "any notes on Mia Chen's record": the same, plus a phase that reads the notes.

Which phases a call runs depends on its input. `PlanGenerator.__phases__` says so.

## The planning command file

<!-- file: fastworkflow/examples/library/_commands/Library/member_report.py -->
```python
"""member_report: the member the request names, with their loans or notes when asked.

find lists the members by name; member opens the one named (skipped when nothing
matched); loans and notes run only when the request asks for them, on the member that
member opened; __result__ says what the phases found. A request about every member is
refused before any step runs.
"""
import fastworkflow
from pydantic import BaseModel, Field

from fastworkflow.planning import count, full_list, listed, pick, run

from ..Member.list_member_loans import Signature as ListMemberLoans
from ..Record.show_notes import Signature as ShowNotes
from .find_member import Signature as FindMember
from .open_member import Signature as OpenMember


class Signature:
    """Show a member's loans, and the notes on their record when asked. One member per call."""

    plain_utterances = ["tell me about Raj Patel", "what has Ann Lee borrowed",
                        "any notes on Mia Chen's record"]

    class Input(BaseModel):
        member: str = Field(description="the member's name",
                            json_schema_extra={"in_request": True})
        loans: bool = Field(default=False,
                            description="whether to name the books the member has on loan")
        notes: bool = Field(default=False,
                            description="whether to show the notes on the member's record")

    @staticmethod
    def validate_extracted_parameters(workflow, command, cmd_parameters, command_text=None):
        if cmd_parameters.member.casefold() in EVERYONE:
            raise fastworkflow.Refused(
                f"member is {cmd_parameters.member!r}, which names no one member; this "
                "command reports on one member: give that member's name")
        return True, ""


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
        return [pick("open_member", named=input.member)]

    def loans(self, member) -> ListMemberLoans.Output:
        if member is None:
            return []
        return [full_list("list_member_loans")]

    def notes(self, member) -> ShowNotes.Output:
        if member is None:
            return []
        return [run("show_notes")]

    def __result__(self, input, member, loans, notes):
        if member is None:
            return f"No member is named {input.member}."
        on_loan = (count(member.on_loan, "book") if loans is None
                   else listed(loans.titles, loans.total, "book"))
        lines = [f"{member.name} has {on_loan} on loan."]
        if notes is not None:
            lines.append(f"{member.name}'s record holds "
                         f"{listed(notes.notes, notes.total, 'note')}.")
        return "\n".join(lines)


#: Words that name every member, not one.
EVERYONE = {"everyone", "everybody", "all", "all members", "every member"}
```

`validate_extracted_parameters` and `EVERYONE` refuse a request about every member; they are
page 9's subject.

## Line by line

- **`def __phases__(self, input)`** returns the phases this call runs, in order, as the
  methods themselves (`self.find`, ...). Without it, a call runs every phase. It is written
  first in the class, so a reader sees the shape before the phases.
- **`loans: bool` and `notes: bool`**: each adds its phase only when the request asks. A
  shape that depends on the input belongs in `__phases__`, decided before anything runs; a
  skip that depends on facts belongs in the phase (`return []`).
- **Both default to false**, so a call reads the loans and the notes only when the request
  asks for them. Their descriptions say what asking means; to have a classifier decide each
  again from the request, add the `verify` option, which needs `JEV_API_KEY` (page 9).
- **`pick("open_member", named=input.member)`** opens the member named, from the listing the
  `find` phase read, as `book_by_title` opens its book (page 4).
- **Each phase hands on the `Output` of the command it runs.** The commands of other
  contexts are imported from their folders: `list_member_loans` from `../Member/`,
  `show_notes` from `../Record/`.
- **`def loans(self, member)`**: `member` is `None` when the `member` phase was skipped (no
  match), so `loans` skips itself too. `full_list("list_member_loans")` writes the step
  `list_member_loans (full list)`, which code runs once per page until the list is read
  (page 7).
- **`list_member_loans` and `show_notes` take no uid.** They run on the member that the
  `member` phase opened, because that is where the workflow stands. `show_notes` is a
  command of `Record`, the base of `Member`, so a member has it too (page 14).
- **`def __result__(self, input, member, loans, notes)`**: `loans` is `None` both when the
  call has no `loans` phase and when it was skipped, so the loans are named only when they
  were read. Its answer is two lines when the notes were read.
- **`listed(...)`** writes "2 books (Dune, Emma)"; page 7 has the wording helpers.
- **A call of one phase** runs as one plan, as page 2's does, whatever `__phases__` returns
  for it.

## What the agent plans from

The template covers every combination of the inputs, and a line some combinations do not
write says when it is written:

```text
find_member query=<member>
open_member with the uid of the <member> match (only when find.member_uids)
list_member_loans (full list) (only when loans)
show_notes (only when notes)
```

The preview names the input that decides each phase, renders each plan once, and shows the
answer `__result__` writes:

```markdown
## Phases

| phase | output | runs | fans out |
|---|---|---|---|
| `find` | `Library/find_member`'s `Output` | always | no |
| `member` | `Library/open_member`'s `Output` | only when `find.member_uids` | no |
| `loans` | `Member/list_member_loans`'s `Output` | only when loans = true | no |
| `notes` | `Record/show_notes`'s `Output` | only when notes = true | no |

## Answer

Written by `__result__`, when loans = true and notes = true:

<member.name> has <loans.total> books (<loans.titles>) on loan.
<member.name>'s record holds <notes.total> notes (<notes.notes>).
```

## At run time

Both asked for:

```text
Phase find
  1. find_member query=Raj Patel
  code runs: find_member query=Raj Patel
  facts: find_member.Output(member_uids=['m2'], names=['Raj Patel'], total=1)
Phase member
  1. open_member with the uid of the Raj Patel match
  code runs: open_member member_uid=m2
  now in: Member: Raj Patel (m2)
  facts: open_member.Output(member_uid='m2', name='Raj Patel', on_loan=2)
Phase loans
  1. list_member_loans (full list)
  code runs: list_member_loans
  facts: list_member_loans.Output(titles=['Dune', 'Emma'], due_dates=['2026-10-20', '2026-11-05'], total=2)
Phase notes
  1. show_notes
  code runs: show_notes
  facts: show_notes.Output(notes=['prefers email reminders'], total=1)
answer written by __result__:
  Raj Patel has 2 books (Dune, Emma) on loan.
  Raj Patel's record holds 1 note (prefers email reminders).
```

Neither asked for:

```text
Phase find
  1. find_member query=Mia Chen
  code runs: find_member query=Mia Chen
  facts: find_member.Output(member_uids=['m3'], names=['Mia Chen'], total=1)
Phase member
  1. open_member with the uid of the Mia Chen match
  code runs: open_member member_uid=m3
  now in: Member: Mia Chen (m3)
  facts: open_member.Output(member_uid='m3', name='Mia Chen', on_loan=1)
answer written by __result__:
  Mia Chen has 1 book on loan.
```

No match, loans asked for: both later phases skip themselves.

```text
Phase find
  1. find_member query=Ulysses Grant
  code runs: find_member query=Ulysses Grant
  facts: find_member.Output(member_uids=[], names=[], total=0)
Phase member: skipped (its method returned [])
Phase loans: skipped (its method returned [])
answer written by __result__:
  No member is named Ulysses Grant.
```

## What you learned

`__phases__(self, input)` returns the phases a call runs. Shape by input in `__phases__`,
skips by facts in the phase; a later phase runs on the record an earlier one opened, and a
phase that did not run reads as `None`.

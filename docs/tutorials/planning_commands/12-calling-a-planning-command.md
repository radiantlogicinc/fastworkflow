# 12. Calling a planning command

**Goal:** "who has borrowed Dune?" `book_by_title` (pages 4 and 11) already finds the book
titled Dune and opens it, and says what it opened in its `Output`. `book_loans` calls it as
one step, then lists the loans of the book it left open.

## The planning command file

<!-- file: fastworkflow/examples/library/_commands/Library/book_loans.py -->
```python
"""book_loans: who has the book the request names on loan.

book calls book_by_title, a planning command, as one step, and hands its Output on; loans
lists the loans of the book that call left open, and is skipped when it found none;
__result__ names the borrowers.
"""
from pydantic import BaseModel, Field

from fastworkflow.planning import full_list, listed, run

from ..Book.list_loans import Signature as ListLoans
from .book_by_title import Signature as BookByTitle


class Signature:
    """Say who has a book on loan, given its title."""

    plain_utterances = [
        "who has borrowed Dune",
        "is anyone reading Emma right now",
        "who is reading Persuasion at the moment",
        "which patrons have Dune Messiah out",
        "is anyone holding a copy of Emma",
        "names of everyone who borrowed Persuasion",
        "has anybody got Dune out, and until when",
        "who currently holds the title Dune Messiah",
    ]

    class Input(BaseModel):
        title: str = Field(description="the book's title",
                           json_schema_extra={"in_request": True})


class PlanGenerator:
    def book(self, input) -> BookByTitle.Output:
        return [run("book_by_title", title=input.title)]

    def loans(self, book) -> ListLoans.Output:
        if not book.found:
            return []
        return [full_list("list_loans")]

    def __result__(self, input, book, loans):
        if loans is None:
            return f"No book matches {input.title}."
        return f"{book.title} is on loan to {listed(loans.borrowers, loans.total, 'borrower')}."
```

## Line by line

- **`run("book_by_title", title=input.title)`** is a step that names a planning command. It
  is written, checked at load and run like any command step: its command is a command of
  the workflow, and every keyword must be a field of its `Input`.
- **`-> BookByTitle.Output`**: the `book` phase hands on `book_by_title`'s `Output`, the
  planning command's public result (page 11), exactly as a phase hands on an execution
  command's. `BookByTitle` is `book_by_title`'s `Signature`, imported from its file.
- **`title`** describes this command's own input. `book_by_title` takes a `title` too, and
  the preview shows its description beside this field, because the step passes `title` on.
  Both carry `in_request` (page 2).
- **The callee is a black box.** It contributes exactly one run to this phase: its name, its
  checked arguments and its `Output`. The runs inside it (`find_book`, `open_book`) are not
  this phase's runs, so the phase could not hand on `open_book`'s `Output` here. That is what
  the callee's `Output` is for.
- **`def loans(self, book)`**: `book` is never skipped here. When the callee found nothing,
  `book.found` is false and `loans` skips itself.
- **`full_list("list_loans")`** names no book. The callee left the workflow on the book it
  opened (page 5), so `list_loans` runs on that book. The observation of the
  `book_by_title` step carries the note `now in context 'Book'`, as it would after
  `open_book`. The library's `Book` context declares no instance label, so the note names the
  context alone.
- **`__result__`** reads both phases: `book` for the title as the library has it, `loans`
  for the borrowers, and `None` for `loans` when it was skipped.

## How the call runs

When the `book` phase's step is run:

1. Its arguments are extracted from the step text, as any command's are, and checked with the
   step text as the request (page 9): `book_by_title`'s `title` carries `in_request`, and
   `Dune` is written in `book_by_title title=Dune`, so the check passes.
2. `book_by_title` runs in a frame of its own, with its own phases and its own executor
   runs. Its step numbers follow on from the caller's, so every step of the turn has its
   own number.
3. Its answer and its `Output` come back as the result of the step, and its frame is
   released.

**A call that fails returns to its caller.** A refused argument, a callee that fails, or a
callee that runs out of steps gives the caller a failed step with the reason, and the
caller carries on. A nested call never hands the turn off.

**A command never calls itself, even through others.** A cycle in the steps the templates
name (`a` calls `b`, `b` calls `a`) is a load error for each command on it: the load reports
it, and `command preview --check` fails. The commands still load. A step written in open
words at run time can still name a command that is already running; that step is refused
with `Library/book_loans is already running; call its steps directly.` and the agent carries
on with execution commands.

**Nested calls share one step budget per turn.** Every model step taken inside a nested
call counts against `FW_TURN_MAX_STEPS` (optional; 400 by default), in
`fastworkflow.env`, and so does a planning command that is one step of a longer plan
(page 14). A planning command that is the agent's whole plan for a request runs right after
the plan, with no other model step; its own steps are bounded by its own caps, not by this
budget, and the calls it makes count. Once the budget is spent, the nested call that is running stops with
`The step budget for nested commands is spent; continue with the remaining steps
directly.`, and every later nested call in the turn fails the same way.

**A question inside a callee works as anywhere else.** When `book_by_title`'s agent asks the
user which book was meant, the whole call waits, `book_loans` included, and the user's
answer resumes it where it stopped, in this process or another.

The template the planner reads for `book_loans`:

<!-- template: book_loans -->
```text
find_book query=<title>
show_book with the uid of the <title> match (only when find.book_uids)
list_loans book_uid=<book.uid> (full list)
say that <book.name> is by <book.author>; the library owns <book.copies> copies
say that it is on loan to <loans.total> borrowers (<loans.borrowers>)
```

## What the agent plans from

```text
book_by_title title=<title>
list_loans (full list) (only when book.found)
```

```markdown
Source: `_commands/Library/book_loans.py`.

Calls: `Book/list_loans`, `Library/book_by_title`.
Depth: 2 (lower bound; open steps name no command)

## Answer

Written by `__result__`: <book.title> is on loan to <loans.total> borrowers (<loans.borrowers>).
```

Depth 2: `book_loans` calls a planning command, `book_by_title`, which calls only execution
commands.

The preview: the phases `book_loans` runs, one of them through the nested `book_by_title`'s sub-task.

<!-- preview: book_loans -->
```markdown
| phase | output | runs | fans out |
|---|---|---|---|
| `find` | `Found` | always | no |
| `book` | `Book` | only when `find.book_uids` | no |
| `loans` | `Loans` | always | no |
| `answer` | none | always | no |
```

## At run time

```text
Phase book
  1. book_by_title title=Dune
  runs the planning command book_by_title (its own frame):
    Phase find: code runs find_book query=Dune
    Phase book: open_book with the uid of the Dune match; code runs open_book book_uid=b1; now in: Book: Dune (b1)
    answer written by __result__: Dune is by Frank Herbert; the library owns 3 copies.
    output: Output(found=True, book_uid='b1', title='Dune', author='Frank Herbert', copies=3)
  now in: Book: Dune (b1)
  facts: book_by_title.Output(found=True, book_uid='b1', title='Dune', author='Frank Herbert', copies=3)
Phase loans
  1. list_loans (full list)
  code runs: list_loans
  facts: list_loans.Output(borrowers=['Ann Lee', 'Raj Patel'], due_dates=['2026-11-02', '2026-10-20'], total=2)
answer written by __result__:
  Dune is on loan to 2 borrowers (Ann Lee, Raj Patel).
```

With no match, the callee reports `found=False`, `loans` is skipped, and the answer says no
book matches Ulysses.

## Who else can call it

Planning commands are listed beside the execution commands of their context, with their
`Signature`, on every list of commands the agent is given, and with their template too on the
list the planner plans from (page 14), so a step that names one can be written in the plan or
later in the agent's run. They are not offered as MCP tools, and a direct action call
(`perform_action`, an MCP tool call) that names one is refused: such a call runs one
command, with no agent to run a plan. Their results still reach the workflow's clients:
every call is an entry of the turn's `command_outputs`, with its `Output` (page 11).

## What you learned

A step may name a planning command. The callee runs in its own frame, is checked like any
call, returns one run with its typed `Output`, and leaves the workflow where it ended. A
phase hands on a planning command's `Output` exactly as it hands on an execution command's.

# 10. Facts from more than one command

**Goal:** hand on facts that are not one command's `Output`. Two jobs show the two cases:

- "is there a member named Lee?" and "do you have a book called Emma?": one search or the
  other runs, and the answer reads whichever did;
- "where is this book kept, and who has it?": one phase runs two commands, and the answer
  reads both.

Pages 3 to 8 hand on one command's `Output` per phase. A phase's facts are always typed by
the commands that produce them: one command's `Output`, a union of `Output`s when one of
several runs, or a **facts model** with one field per command when several run.

## One search or the other: a union

<!-- file: fastworkflow/examples/library/_commands/Library/find_record.py -->
```python
"""find_record: whether a book or a member goes by a name, and which.

One find phase runs find_book or find_member, by the kind the request asks about, and
hands on the Output of whichever ran; __result__ reads its names either way.
"""
from typing import Literal

from pydantic import BaseModel, Field

from fastworkflow.planning import count_and_names_read, run

from .find_book import Signature as FindBook
from .find_member import Signature as FindMember


class Signature:
    """Say whether a book or a member goes by a name, and name each match."""

    plain_utterances = [
        "do you have a book called Emma",
        "is there a member named Lee",
        "do we have any member called Patel",
        "is there a book called Dune Messiah",
        "does anyone go by the name Mia",
        "is a title called Persuasion in the catalogue",
        "is there a patron named Raj",
        "check whether we have a book titled Dune and name the matches",
    ]

    class Input(BaseModel):
        name: str = Field(description="the name to look for",
                          json_schema_extra={"copied_from_command_text": True})
        kind: Literal["book", "member"] = Field(
            default="book", description="whether to look for a book or a member")


class PlanGenerator:
    def find(self, input) -> FindBook.Output | FindMember.Output:
        return [run(FIND[input.kind], query=input.name)]

    def __result__(self, input, find):
        names = find.titles if input.kind == "book" else find.names
        return f"{input.name} matches {count_and_names_read(names, find.total, input.kind)}."


FIND = {"book": "find_book", "member": "find_member"}
```

- **`kind: Literal["book", "member"]`** is an enum input; `default="book"` makes it optional.
- **`-> FindBook.Output | FindMember.Output`** hands on the `Output` of whichever of the two
  commands the phase ran: `find_book`'s for a book, `find_member`'s for a member. Each is
  imported from its file, as on page 3.
- **`FIND[input.kind]`** looks the command up in a module-level table, so the step names
  `find_book` or `find_member` by the kind.
- **`__result__` reads either.** `find_book`'s `Output` calls its names `titles`, and
  `find_member`'s calls them `names`; one line of Python picks the field, and `total` is
  in both.

The template merges the lines of an enum's values, each difference written once with the
input that decides it:

```text
<find_book or find_member, by kind> query=<name>
```

```text
Phase find
  1. find_member query=Lee
  code runs: find_member query=Lee
  facts: find_member.Output(member_uids=['m1'], names=['Ann Lee'], total=1)
answer written by __result__:
  Lee matches 1 member (Ann Lee).
```

```text
Phase find
  1. find_book query=Emma
  code runs: find_book query=Emma
  facts: find_book.Output(book_uids=['b3'], titles=['Emma'], total=1)
answer written by __result__:
  Emma matches 1 book (Emma).
```

## Two commands in one phase: a facts model

<!-- file: fastworkflow/examples/library/_commands/Book/shelf_report.py -->
```python
"""shelf_report: where the open book is held, and who has it on loan.

A planning command of the Book context. Its one phase runs show_book and list_loans on the
open book, and hands on both Outputs in a facts model with one field per command;
__result__ reads both.
"""
from pydantic import BaseModel

from fastworkflow.planning import count, count_and_names_read, run

from .list_loans import Signature as ListLoans
from .show_book import Signature as ShowBook


class Signature:
    """Say where this book is held, how many copies there are, and who has it on loan."""

    plain_utterances = [
        "where is this book kept, and who has it",
        "is this book in",
        "which branch keeps this book, how many copies, and who is borrowing it",
        "where do we keep this title and is any copy out",
        "shelf location and loan status for this one",
        "is there a copy in the building right now",
        "tell me where this book sits and who holds the copies",
        "give me the shelf status of this title",
    ]


class Shelf(BaseModel):
    book: ShowBook.Output
    loans: ListLoans.Output


class PlanGenerator:
    def shelf(self) -> Shelf:
        return [run("show_book"), run("list_loans", all_pages=True)]

    def __result__(self, shelf):
        book, loans = shelf.book, shelf.loans
        held = f"{book.title} is held at {book.branch}, {count(book.copies, 'copy', 'copies')}"
        if not loans.borrowers:
            return f"{held}; none is on loan."
        return f"{held}; on loan to {count_and_names_read(loans.borrowers, loans.total, 'borrower')}."
```

- **It lives in `_commands/Book/`**, beside the two commands it runs, so it runs on the book
  the workflow stands on, as `list_loans` does (page 5). Page 14 says how a request finds
  it.
- **`class Shelf(BaseModel)`** is a facts model: an ordinary pydantic model whose every field
  is typed with the `Output` of a command the phase runs. After the phase, code fills `book`
  with the `Output` of its `show_book` run and `loans` with that of its `list_loans` run,
  with no model call. A field no run fills fails the phase.
- **`def shelf(self) -> Shelf`**: the phase runs both commands, and hands on the model.
- **`def __result__(self, shelf)`** reads both, `shelf.book.branch` and
  `shelf.loans.borrowers`.
- **Other fields a facts model can have:**
  - a field typed `X.Output | None = None` is `None` when the phase ran no `X`;
  - a field typed with a command's `Input` gets the arguments that command was called with;
  - a field typed `A.Output | B.Output` takes whichever ran, as the union above does;
  - nothing else: a value of your own is a property or a method of the model, computed from
    its fields. A field of any other type fails the load.
- **A facts model is ordinary Python.** It may have properties, methods and validators; a
  validator that refuses fails the phase. It may live in a `_` module and be shared by
  several commands, and each of its fields names its command by the import, so a command
  that several contexts define is never ambiguous.

```text
Phase shelf
  1. show_book
  2. list_loans (full list)
  code runs: show_book
  code runs: list_loans
  facts: Shelf(book=show_book.Output(book_uid='b1', title='Dune', author='Frank Herbert', branch='Central', copies=3, on_loan=2), loans=list_loans.Output(borrowers=['Ann Lee', 'Raj Patel'], due_dates=['2026-11-02', '2026-10-20'], total=2))
answer written by __result__:
  Dune is held at Central, 3 copies; on loan to 2 borrowers (Ann Lee, Raj Patel).
```

And on Dune Messiah, which no one has borrowed:

```text
answer written by __result__:
  Dune Messiah is held at Central, 1 copy; none is on loan.
```

The template the planner reads for `find_record`:

<!-- template: find_record -->
```text
<find_book or find_member, by kind> query=<name>
```

## What the agent plans from

The preview's phase table names each phase's output, and its facts section lists each
field with the command its value comes from:

```markdown
## Phases

| phase | output | runs | fans out |
|---|---|---|---|
| `shelf` | `Shelf` | always | no |
```

The preview lists the phases its facts models come from.

<!-- preview: find_record -->
```markdown
| phase | output | runs | fans out |
|---|---|---|---|
| `find` | when kind = book: `BookMatch`<br>when kind = member: `MemberMatch` | always | no |
```

## Which form to write

Write the phase's return annotation when its facts type is fixed: one `Output`, a union for
"whichever ran", or a facts model, as on this page. Use `@phase(output=lambda input: ...)` only
when the type depends on the input. A phase with both a return annotation and `output=` does
not load: the error names the phase.

## What you learned

A phase's facts are typed by the commands that produce them: a union of `Output`s hands on
whichever ran, and a facts model holds one field per command, typed with its `Output` (or
its `Input`). Anything else is ordinary Python, in the model or in what reads it.

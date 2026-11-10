# 5. Where a command leaves the workflow

**Goal:** understand the `now in:` lines of page 4. `book_by_title` ran `open_book`, which
moved the workflow onto the book, and the call ended there. This page shows the command that
moves it, the context it moves to, and what that means for the steps and requests that come
after.

## The command that moves the workflow

<!-- file: fastworkflow/examples/library/_commands/Library/open_book.py -->
```python
"""open_book: make one book, by its uid, the current record."""
import fastworkflow
from pydantic import BaseModel, Field

from ...application.library import Library


class Signature:
    """Open one book by its uid, so that the book's own commands act on it."""

    class Input(BaseModel):
        book_uid: str = Field(
            description="the uid of the book to open",
            json_schema_extra={"from_command_output": "find_book.book_uids"},
            pattern=r"^b[0-9]+$",
        )

    class Output(BaseModel):
        book_uid: str = Field(description="the uid of the book now open")
        title: str = Field(description="the book's title")
        author: str = Field(description="the book's author")
        copies: int = Field(description="how many copies the library owns")

    plain_utterances = [
        "open book b1",
        "go to the book Dune",
        "open the book with uid b3",
        "make b2 the current book",
        "select b2 so I can act on it",
        "I want to act on b3 now",
        "make the Dune Messiah book the open one",
        "load uid b4 as the current book",
    ]


@fastworkflow.descends_to("Book")
class ResponseGenerator:

    def __call__(self, workflow: fastworkflow.Workflow, command: str,
                 input: Signature.Input) -> Signature.Output:
        library: Library = workflow.record
        book = library.books.get(input.book_uid)
        if book is None:
            raise fastworkflow.Refused(
                f"no book has the uid {input.book_uid}; find_book lists the uids")
        workflow.record = book
        return Signature.Output(book_uid=book.uid, title=book.title, author=book.author,
                                copies=book.copies)
```

- **`workflow.record = book`** is the move. From here on, the workflow
  stands on that book, and the book's own commands act on it.
- **`@descends_to("Book")`** declares the move, so the framework knows without running it
  that `open_book` leads from `Library` into `Book`. It is the one place this is declared:
  the agent is shown the way into a context it is not in, the list it plans from (page 14)
  uses these declarations to reach the commands of contexts a plan has not entered yet, and
  code checks each plan against them, so a step planned in a context the plan never enters
  gets a note. A command that moves the workflow into a context it does not declare gets a
  warning the first time it does it, naming the command and the context, so a missing
  `@descends_to` shows up in the first run.
- **`json_schema_extra={"from_command_output": "find_book.book_uids"}`** points this field at
  another command's `Output` field: the place its value comes from. On an `Input` field,
  `from_command_output` is a **producer** hint: "call `find_book` first; one of its `book_uids` is this value". The
  agent reads it as `Library/find_book.book_uids`, field and all, as it plans and as it
  runs; code checks each plan against it, so a plan that opens a book before any
  `find_book` gets a note (page 14); and the message for a missing `book_uid` says "use the
  find_book command(s) to get book_uid information". The types are checked: a `str` filled
  from one item of a `list[str]` is allowed. The key sits beside the field's other `Field`
  arguments, such as the description.
- **A producer hint is only a hint.** It does not look the value up or check it. `open_book`
  refuses a uid no book has itself, with `fastworkflow.Refused`; the agent reads the message
  and can run `find_book`.
- **`from_command_output` is only for an `Input` field.** It names the `Output` field a value comes from,
  and that is all it does. A phase's facts never need it: they are typed with the commands
  they come from (pages 3 and 10).

## The context it moves to

<!-- file: fastworkflow/examples/library/_commands/Book/_Book.py -->
```python
from ...application.library import Book, Library


class Context:
    @classmethod
    def get_parent(cls, record: Book) -> Library:
        return record.library

    @classmethod
    def get_displayname(cls, record: Book) -> str:
        return f"Book: {record.title} ({record.uid})"
```

`_Book.py` describes the `Book` context's object. `get_parent` is the record that encloses a
book, the library; `get_displayname` is how a book is named to the agent. The context class
says nothing about how to enter it: that is the navigation command's `@descends_to`. And here
is one of the book's own commands:

<!-- file: fastworkflow/examples/library/_commands/Book/list_loans.py -->
```python
"""list_loans: who has the open book on loan, and until when."""
from typing import Optional

import fastworkflow
from pydantic import BaseModel, Field

from ...application.library import Book, page


class Signature:
    """List the loans of the open book: each borrower and the due date."""

    class Input(BaseModel):
        start: Optional[int] = Field(default=None, description="the first row of the page")

    class Output(BaseModel):
        borrowers: list[str] = Field(description="each borrower's name, in row order")
        due_dates: list[str] = Field(description="each loan's due date, in row order")
        total: int = Field(description="how many loans the book has, on every page")

    plain_utterances = [
        "list this book's loans",
        "give me page two of this title's loans",
        "how many loans does this book have",
        "give me the full loan list for this title",
        "read out the borrowers and due dates of this book, row by row",
        "list the loans on this one, starting from row 11",
        "every loan this title has, please",
        "what loans are open on this book",
    ]


class ResponseGenerator:
    def __call__(self, workflow: fastworkflow.Workflow, command: str,
                 input: Signature.Input) -> Signature.Output:
        book: Book = workflow.record
        loans = book.library.loans_of_book(book.uid)
        rows = page(loans, input.start)
        return Signature.Output(borrowers=[m for m, _ in rows], due_dates=[d for _, d in rows],
                                total=len(loans))
```

`list_loans` takes no `book_uid`. It reads the book it acts on from
`workflow.record`: the book the workflow stands on. It is
also **paged**, by its integer `start`; page 7 says what that asks of its `Output`.

## Every command leaves the workflow where it ends

This holds for both kinds of command, and nothing restores the context a command started
in. `book_by_title` opened Dune, so after the call the workflow stands on Dune, exactly as
it would after the user ran `open_book book_uid=b1` themselves. A conversation:

```text
user:  who wrote Dune?
agent: Dune is by Frank Herbert; the library owns 3 copies.      (book_by_title; now in Book: Dune)
user:  which branch holds it?
agent: Dune is held at Central.                                  (Book/show_book, on Dune)
user:  go back
agent: You are in the Library.                                   (go_up)
```

"Which branch holds it?" is answered by `show_book`, the open book's own command, with no
search: the workflow is already on the book.

**The agent is told where it now is.** When a step moves the workflow, its observation ends
with the context it left the agent in, the same note for an execution command and for a
planning command:

```text
Observation O2 (execute_workflow_query ran in prior context 'Library'; now in context 'Book' and Dune b1)
```

Without that note a weak model reads the context a command ran in as the one it is still in.

**A step that needs another record navigates to it.** A planning command does not get its
starting context back after a step that moved: its next step starts where the last one left
off. That is why `book_by_title` hands on `open_book`'s own `Output`, the record actually
opened, and why a command that must start on the library says so in a step (`go_up`, or a
command that opens what it needs).

**Facts are read where the phase ended.** A phase hands on its runs on the record it ended
on. Its facts name each command by the imported `Signature`, so a command that several
contexts define (two contexts that each have a `list_accounts`, say) is never ambiguous: the
import says which one.

## What you learned

A navigation command sets `workflow.record` and is decorated `@descends_to`, the one place the
move is declared. `from_command_output` on an `Input` field names the `Output` field a value comes
from: a producer hint. Every command, of either kind, leaves the workflow where it
ends; the agent's observation says where that is, and nothing restores the context a command
started in.

# 11. A typed Output

**Goal:** let other commands, and the clients of the workflow, use what `book_by_title` (page
4) found: the book's uid, title, author and copies, as typed fields rather than a sentence.
An execution command does this by returning its `Signature.Output` from `__call__`. A
planning command declares a `Signature.Output` too, and returns it from `__result__`, beside
its answer or instead of it.

## The planning command file

The same file as page 4, read for the parts that page skipped:

<!-- file: fastworkflow/examples/library/_commands/Library/book_by_title.py -->
```python
"""book_by_title: the book whose title the request names, picked by its title.

find lists the matches; book opens the one whose title is the request's title (pick), and
is skipped when nothing matched; __result__ says the author and copies, or that no book
matches, and reports the book to a caller in its Output. The call ends on the book it
opened.
"""
from typing import Optional

from pydantic import BaseModel, Field

from fastworkflow import CommandResponse
from fastworkflow.planning import pick, run

from ._wording import copies
from .find_book import Signature as FindBook
from .open_book import Signature as OpenBook


class Signature:
    """Show who wrote a book, and how many copies the library owns, given its title."""

    plain_utterances = ["who wrote Dune", "how many copies of Dune Messiah are there"]

    class Input(BaseModel):
        title: str = Field(description="the book's title",
                           json_schema_extra={"in_request": True})

    class Output(BaseModel):
        found: bool = Field(description="whether a book was opened for the title")
        book_uid: Optional[str] = Field(default=None, description="the uid of the book opened")
        title: Optional[str] = Field(default=None, description="its title, as the library has it")
        author: Optional[str] = Field(default=None, description="its author")
        copies: int = Field(default=0, description="how many copies the library owns")


class PlanGenerator:
    def find(self, input) -> FindBook.Output:
        return [run("find_book", query=input.title)]

    def book(self, input, find) -> OpenBook.Output:
        if not find.book_uids:
            return []
        return [pick("open_book", named=input.title)]

    def __result__(self, input, book):
        if book is None:
            return CommandResponse(response=f"No book matches {input.title}.",
                                   output=Signature.Output(found=False))
        return CommandResponse(
            response=f"{book.title} is by {book.author}; the library owns {copies(book.copies)}.",
            output=Signature.Output(found=True, **book.model_dump()))
```

## Line by line

- **`class Output(BaseModel)` inside `Signature`** is the command's public result, written
  as an execution command's is: typed fields, each with a description. It is what a caller
  reads (page 12) and what API clients receive (below).
- **No `Output` field has a `from_`**, at either kind of command: an `Output` is built by
  code, not copied from runs.
- **The phases' facts are working memory.** Here they are the `Output`s of the commands the
  phases ran; on page 10 one is a facts model. Keeping them apart from this command's
  `Output` lets the command change how it works without changing what it promises.
- **`def __result__(self, input, book)`** returns what an execution command's `__call__`
  returns, in one of three forms:
  - a `str`: the answer, with no `Output` (page 3);
  - the command's `Signature.Output` alone: the `Output`, with its JSON as the response, so
    the agent that reads the call words it, as it does after an execution command;
  - a `CommandResponse` holding both, as here: `response` is the answer a person reads, and
    `output` the `Output` callers and clients read.
- **Its parameters bind by name to the phases' facts**, as a phase's do. A phase that was
  skipped, or filled no facts, arrives as `None`. It runs once, after the last phase of the
  call, as plain code: it adds no step and runs no command or model.
- **`**book.model_dump()`**: `book` is `open_book`'s `Output`, and its four fields have the
  names of four of this command's, so `__result__` passes them through.
- **It returns an `Output` on every path.** When no book matched, the `Output` says so,
  `found=False`, because `book_loans` (page 12) reads it: a command that declares an
  `Output` and returns none gets the warning `book_by_title declares an Output but returned
  none; phases and from_ references cannot read it`, and its callers get no facts.
- **The `Output` may span phases or be computed.** `__result__` is ordinary code: it can take
  one field from each of several phases, or compare two of them.

The load checks the hook's shape without running it: a `Signature.Output` with no
`__result__` to build it, and a parameter (other than `input`) that names no phase, each fail
the load.

**One rule differs from an execution command.** An execution command whose `__call__` fails
fails the call. A planning command's runs took many steps and still answer the request, so a
`__result__` that raises, or returns something other than its three forms (an `Output` of
another class included), keeps the call: a model writes the answer from the phases' runs,
`output` is `None`, and the warning names the command and the exception.

## What the agent plans from

`__result__` writes no step, so the template is unchanged. The preview adds the `Output`:

```markdown
Source: `_commands/Library/book_by_title.py`.

Calls: `Library/find_book`, `Library/open_book`.
Depth: 1 (lower bound; open steps name no command)

## Output

Output: built by `__result__`.

| field | type | description |
|---|---|---|
| `found` | `bool` | whether a book was opened for the title |
| `book_uid` | `Optional[str]` | the uid of the book opened |
| `title` | `Optional[str]` | its title, as the library has it |
| `author` | `Optional[str]` | its author |
| `copies` | `int` | how many copies the library owns |
```

Its answer is shown as decided at run time, because `__result__` builds the `Output` from
facts that are placeholders while the template is built (page 4).

## At run time

The call's answer is what the user reads, as written when this command is the whole plan for
a request (page 14); its `Output` is stored beside it, as an execution command's is
(`CommandResponse.output`), where callers and later steps read it. The agent's observation
shows the answer, not the `Output`.

```text
Phase find
  1. find_book query=Dune
  code runs: find_book query=Dune
  facts: find_book.Output(book_uids=['b1', 'b2'], titles=['Dune', 'Dune Messiah'], total=2)
Phase book
  1. open_book with the uid of the Dune match
  code runs: open_book book_uid=b1
  now in: Book: Dune (b1)
  facts: open_book.Output(book_uid='b1', title='Dune', author='Frank Herbert', copies=3)
answer written by __result__:
  Dune is by Frank Herbert; the library owns 3 copies.
output: Output(found=True, book_uid='b1', title='Dune', author='Frank Herbert', copies=3)
```

With no match, `book` is `None` in `__result__`:

```text
output: Output(found=False, book_uid=None, title=None, author=None, copies=0)
```

## Who reads it

- **A caller.** A planning command that runs this one as a step hands its `Output` on, as it
  would an execution command's (page 12).
- **The clients of the workflow.** A turn's result, the body every `invoke_agent`,
  `invoke_assistant` and `perform_action` call returns, lists every command the turn ran in
  `command_outputs`, each with its `command_name` and its `command_response`, whose `output`
  is that command's `Output`. A planning command that answered the request is one of them,
  with the `Output` its `__result__` returned. A client finds each command's `Input` and
  `Output` schemas by qualified name with `GET /command_schemas`, so it can read `output` as
  the type it is. An MCP tool for an execution command declares its `Output` as the tool's
  `outputSchema` and returns it as `structuredContent`.

The shape of that result never depends on which command ran: the framework builds every
entry of it, from what each command returned.

## What you learned

A planning command's `Signature.Output` is its public, typed result, returned by
`__result__` from the phases' facts, beside its answer in a `CommandResponse` or on its own,
exactly as an execution command's `__call__` returns one. A command that others read from
should always return an `Output`, saying "not found" in a field when it found nothing;
callers and clients read it next to the answer.

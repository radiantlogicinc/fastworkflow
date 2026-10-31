# 3. A fact passed on, and an answer written in code

**Goal:** "how many books by Jane Austen do you have?" answered by code, with no model writing
the answer. Page 1's phases passed nothing on; here the `find` phase runs the search and hands
on what it found, `find_book`'s `Output`, and `__result__` writes the answer from it.

## The planning command file

<!-- file: fastworkflow/examples/library/_commands/Library/count_books.py -->
```python
"""count_books: how many books match the words the request gives.

The find phase runs find_book and hands its Output on; __result__ writes the answer from
that Output's total.
"""
from pydantic import BaseModel, Field

from fastworkflow.planning import count, run

from .find_book import Signature as FindBook


class Signature:
    """Count the books whose title or author holds some words."""

    plain_utterances = ["how many books by Jane Austen do you have"]

    class Input(BaseModel):
        query: str = Field(description="words of the title or the author's name",
                           json_schema_extra={"in_request": True})


class PlanGenerator:
    def find(self, input) -> FindBook.Output:
        return [run("find_book", query=input.query)]

    def __result__(self, input, find):
        return f"{input.query} matches {count(find.total, 'book')}."
```

## Line by line

- **One phase**, `find`. Nothing passes from one phase to the next, or to the answer, but
  the workflow's position and the **facts**, so `find` says what it hands on.
- **`-> FindBook.Output`** (the return annotation) says it: the `Output` of the phase's
  `find_book` run, as it is. After the phase, code takes that run's `Output` and hands it on,
  with no model call. If the phase ends without a successful `find_book` run, the phase
  fails.
- **`from .find_book import Signature as FindBook`**: `FindBook` is `find_book`'s
  `Signature`, imported from the file beside this one. The reference is plain Python, so a
  command that is renamed or moved fails at the import, and your editor follows it to the
  command. A phase annotated with a command's `Output` must run that command: a phase whose
  steps never name `find_book` fails the load.
- **`def __result__(self, input, find)`** writes the call's result in code. A parameter named
  after a phase gets that phase's facts: here `find_book`'s `Output`, so `find.total` is its
  `total`. A misspelt name fails the load. It runs once, after the last phase.
- **It returns a `str`**, as an execution command's `__call__` returns a `str` when it has no
  `Output`: that text is the call's answer, as it is. When this command is the whole plan
  for a request, the user reads it as written: no model rewrites it. `__result__` can also
  return the command's `Output`, or both (page 11).
- **`count(find.total, 'book')`** writes "1 book" or "2 books" (page 7 has the other
  helpers).
- **`query`** describes what this command takes, in its own words. Here that happens to be
  what `find_book` takes too. Its `in_request` option makes it words the request writes
  (page 2).

A command without `__result__` has its answer written by its last phase's agent, as on
pages 1 and 2. When the facts a later phase or `__result__` needs come from more than one
command, page 10 shows how to hand them on.

## The same job as an execution command

No step of `count_books` needs the agent, so the job is shorter as an execution command
that calls the app (page 1, Which kind to write). A sketch, not a file of the sample:

```python
class ResponseGenerator:
    def __call__(self, workflow, command, command_parameters: Signature.Input) -> str:
        library: Library = workflow.command_context_for_response_generation
        found = library.find_books(command_parameters.query)
        return f"{command_parameters.query} matches {count(len(found), 'book')}."
```

It answers the same, runs in assistant mode and through direct API and MCP calls too, and
needs none of the planning idioms. The sample keeps `count_books` as a planning command to
show a phase handing on a command's `Output`.

The template the planner reads for `count_books` (its steps with the query unknown):

<!-- template: count_books -->
```text
find_book query=<query>
say that <find.total> books match <query>
```

## What the agent plans from

The agent's list of commands shows the phases' steps joined, every fact a placeholder named
by where it is read:

```text
find_book query=<query>
```

The preview adds a table of the phases and of what each hands on, and the answer
`__result__` writes with every fact a placeholder (`count` on a placeholder writes the plural,
since the number is not known yet):

```markdown
## Phases

| phase | output | runs | fans out |
|---|---|---|---|
| `find` | `Library/find_book`'s `Output` | always | no |

## Facts

What each phase's output holds, filled in code from the commands the phase ran.

### `find` -> `Library/find_book`'s `Output`

| fact | type | from |
|---|---|---|
| `book_uids` | `list[str]` | not filled from a command |
| `titles` | `list[str]` | not filled from a command |
| `total` | `int` | not filled from a command |

## Answer

Written by `__result__`: <query> matches <find.total> books.
```

The planning command's preview: its phases and its facts.

<!-- preview: count_books -->
```markdown
| phase | output | runs | fans out |
|---|---|---|---|
| `find` | `Found` | always | no |
| `answer` | none | always | no |
```

## At run time

For "how many Dune books do you have" (`query=Dune`):

```text
Phase find
  1. find_book query=Dune
  code runs: find_book query=Dune
  facts: Output(book_uids=['b1', 'b2'], titles=['Dune', 'Dune Messiah'], total=2)
answer written by __result__:
  Dune matches 2 books.
```

And for `query=Persuasion`:

```text
Phase find
  1. find_book query=Persuasion
  code runs: find_book query=Persuasion
  facts: Output(book_uids=['b4'], titles=['Persuasion'], total=1)
answer written by __result__:
  Persuasion matches 1 book.
```

Code ran `find_book`, handed its `Output` on, called `__result__` with it, and the text it
returned is the answer: no model wrote a word of it.

Page 1's overview could say its branches the same way: annotate its `branches` phase with
`ListBranches.Output`, imported from `list_branches.py`, and read it through a parameter
named `branches`.

Run offline, `find` hands its fact on and `answer` writes the sentence from it:

<!-- dry-run: count_books {"query": "Dune"} -->
```text
Phase find
  1. find_book query=Dune
  code runs: find_book query=Dune
  facts: Found(total=2)
Phase answer
  1. say that 2 books match Dune
  answer written by code:
    2 books match Dune.
```

## What you learned

A phase annotated with a command's `Output` hands that `Output` on, and whatever reads it
names the phase as a parameter. `__result__` returns the call's result in code, from the
input and the phases' facts; a `str` it returns is the answer, as written. A job that needs
no agent at all is shorter as an execution command.

# 4. Skipping a phase, picking a row

**Goal:** "who wrote Dune?" The command finds the book, opens it, and says its author and
copies. Two things can go wrong. Nothing matches, so there is nothing to open: the second
phase is skipped and the answer says so. Or several books match (`find_book query=Dune`
lists Dune and Dune Messiah): the command opens the one *titled* Dune, with `pick`.

## The planning command file

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

    plain_utterances = [
        "who wrote Dune",
        "how many copies of Dune Messiah are there",
        "who is the author of Persuasion, and how many copies do we have",
        "tell me about the book Emma",
        "what is Dune Messiah by, and how many copies",
        "give me the author and copy count for the title Emma",
        "does the library own Persuasion, and who wrote it",
        "how many copies of Emma do we own and who wrote it",
    ]

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

`__result__` returns two things here, in a `CommandResponse`: the answer (`response`), and
the `Output` that a planning command calling this one reads. The `Output` is page 11's
subject; read the answer part for now.

`copies` comes from a helper file beside it:

<!-- file: fastworkflow/examples/library/_commands/Library/_wording.py -->
```python
"""Wording the library's planning commands share.

A file whose name starts with "_" is never a command: planning commands import it.
"""
from fastworkflow.planning import count


def copies(number):
    """How many copies the library owns, in words: "1 copy", "3 copies"."""
    return count(number, "copy", "copies")
```

## Line by line

- **`from ._wording import copies`**: a file whose name starts with `_` is never a command,
  so `_wording.py` is an ordinary module that commands of its folder import. `copies` is a
  plain function over `count` (page 7) that says "1 copy" or "3 copies". `author_books`
  (page 8) calls it too.
- **Each phase hands on the `Output` of the command it runs**, as on page 3: `find` hands on
  `find_book`'s, `book` hands on `open_book`'s, the record actually opened, with its
  `book_uid`, `title`, `author` and `copies`.
- **`if not find.book_uids: return []`**: a phase that returns no steps is **skipped**. The
  method is called once, when the run reaches the phase, with the real facts; the decision
  is never made earlier. A skipped phase runs nothing and hands nothing on.
- **`pick("open_book", named=input.title)`** writes the step that opens one row of a
  listing, in the words a row is picked by: `open_book with the uid of the Dune match`. Code
  looks for the listing among the runs of this call, wherever it read it: the latest run
  whose arguments hold the name and whose uid list fits `open_book`'s `book_uid`. Here that
  is `find_book`'s `book_uids`, read in the phase before. Then:
  - when exactly one row is titled as asked (case and spacing aside), or the listing has
    only one row, code runs `open_book book_uid=...` on it;
  - otherwise the agent picks the row: the step is written in words it can read.

  When a rule of your own chooses the row, give the uid yourself:
  `pick("open_book", named=input.title, book_uid=uid)`.
- **`def __result__(self, input, book)`**: `book` is `None` when the `book` phase was
  skipped, so `__result__` says that no book matches. A phase that was skipped always
  arrives as `None`, here and in a later phase's parameters; no default is needed.
- **`__result__` runs after whatever phases ran**, so a call always has its result, however
  many phases it skipped.

The template the planner reads for `book_by_title`:

<!-- template: book_by_title -->
```text
find_book query=<title>
show_book with the uid of the <title> match (only when find.book_uids)
```

## What the agent plans from

In the template every fact is unknown, so every test on one reads as true and is noted. The
`book` phase's line ends with the test its method made, and the `pick` is in pick words:

```text
find_book query=<title>
open_book with the uid of the <title> match (only when find.book_uids)
```

The preview shows the phases:

```markdown
## Phases

| phase | output | runs | fans out |
|---|---|---|---|
| `find` | `Library/find_book`'s `Output` | always | no |
| `book` | `Library/open_book`'s `Output` | only when `find.book_uids` | no |

## Answer

Written by `__result__`: decided at run time (TypeError while the template was built, with every fact unknown).
```

The answer is decided at run time in the preview because `__result__` also builds the
`Output` from `book`, and an `Output` cannot hold the placeholders the template is built
with. Nothing at run time depends on it, and the runs below show the answer on real facts.

The preview lists the phases, and `book` runs only when `find` found a book uid.

<!-- preview: book_by_title -->
```markdown
| phase | output | runs | fans out |
|---|---|---|---|
| `find` | `Found` | always | no |
| `book` | `Book` | only when `find.book_uids` | no |
```

## At run time

The exact title is one row of two: code opens it.

```text
Phase find
  1. find_book query=Dune
  code runs: find_book query=Dune
  facts: Output(book_uids=['b1', 'b2'], titles=['Dune', 'Dune Messiah'], total=2)
Phase book
  1. open_book with the uid of the Dune match
  code runs: open_book book_uid=b1
  now in: Book: Dune (b1)
  facts: Output(book_uid='b1', title='Dune', author='Frank Herbert', copies=3)
answer written by __result__:
  Dune is by Frank Herbert; the library owns 3 copies.
output: Output(found=True, book_uid='b1', title='Dune', author='Frank Herbert', copies=3)
```

No title is exactly "Dune Messia", but the listing has only one row, so code opens it:

```text
Phase find
  1. find_book query=Dune Messia
  code runs: find_book query=Dune Messia
  facts: Output(book_uids=['b2'], titles=['Dune Messiah'], total=1)
Phase book
  1. open_book with the uid of the Dune Messia match
  code runs: open_book book_uid=b2
  now in: Book: Dune Messiah (b2)
  facts: Output(book_uid='b2', title='Dune Messiah', author='Frank Herbert', copies=1)
answer written by __result__:
  Dune Messiah is by Frank Herbert; the library owns 1 copy.
output: Output(found=True, book_uid='b2', title='Dune Messiah', author='Frank Herbert', copies=1)
```

"Dun" lists two books, and neither is titled "Dun", so code cannot pick: the agent runs the
step, with the listing in front of it:

```text
Phase find
  1. find_book query=Dun
  code runs: find_book query=Dun
  facts: Output(book_uids=['b1', 'b2'], titles=['Dune', 'Dune Messiah'], total=2)
Phase book
  1. open_book with the uid of the Dun match
  step 1 is open: the agent would run it and the rest
```

No match:

```text
Phase find
  1. find_book query=Ulysses
  code runs: find_book query=Ulysses
  facts: Output(book_uids=[], titles=[], total=0)
Phase book: skipped (its method returned [])
answer written by __result__:
  No book matches Ulysses.
output: Output(found=False, book_uid=None, title=None, author=None, copies=0)
```

Each run that opened a book ends with `now in:`: the call moved the workflow onto that book
and leaves it there. Page 5 is about that. The `output:` lines are the `Output` page 11 is
about.

Run offline for the title Dune, the pick opens the one exact match:

<!-- dry-run: book_by_title {"title": "Dune"} -->
```text
Phase find
  1. find_book query=Dune
  code runs: find_book query=Dune
  facts: Found(book_uids=['b1', 'b2'], titles=['Dune', 'Dune Messiah'])
Phase book
  1. show_book with the uid of the Dune match
  code runs: show_book book_uid=b1
  facts: Book(uid='b1', name='Dune', author='Frank Herbert', copies=3)
answer written by __result__:
  Dune is by Frank Herbert; the library owns 3 copies.
```

## What you learned

`return []` skips a phase, decided when it is reached; whatever reads a skipped phase gets
`None`; the template marks the skipped lines "(only when ...)". `pick(command, named=...)`
opens the row of a listing the call read, in any phase: code opens the row titled as asked,
or the only row, and otherwise the agent picks. A `_` file holds what several commands share.

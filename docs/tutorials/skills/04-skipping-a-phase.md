# 4. Skipping a phase

**Goal:** "who wrote Persuasion?" The skill finds the book, opens it, and says its author
and copies. When nothing matches, there is nothing to open: the middle phase is skipped and
the answer says no book matches.

## The skill file

<!-- file: tests/skill_tutorial_workflow/_skills/book_details.py -->
```python
"""book_details: the first book matching the request's words, or that none matches.

find lists the matches; book opens the first one, and is skipped when nothing matched;
answer says what book found, or that no book matches.
"""
from pydantic import BaseModel, Field

from fastworkflow.skill import fact_from


class Signature:
    """Show the author and copies of the book the request names."""

    plain_utterances = ["who wrote Persuasion", "how many copies of Emma are there"]

    class Input(BaseModel):
        title: str = Field(description="the book's title exactly as the request writes it")


class PlanGenerator:
    class Found(BaseModel):
        book_uids: list[str] = fact_from("find_book.book_uids")

    def find(self, input) -> Found:
        return [f"find_book query={input.title}"]

    class Book(BaseModel):
        name: str = fact_from("show_book.title")
        author: str = fact_from("show_book.author")
        copies: int = fact_from("show_book.copies")

    def book(self, find) -> Book:
        if not find.book_uids:
            return []
        return [f"show_book book_uid={find.book_uids[0]}"]

    def answer(self, input, book=None):
        if book is None:
            return [f"say that no book matches {input.title}"]
        return [f"say that {book.name} is by {book.author}; the library owns "
                f"{book.copies} copies"]
```

## Line by line

- **`if not find.book_uids: return []`**: a phase that returns no steps is **skipped**. The
  method is called once, when the run reaches the phase, with the real facts; the decision
  is never made earlier. A skipped phase runs nothing and fills no facts.
- **`show_book book_uid={find.book_uids[0]}`** writes the uid the find returned, so code
  runs the step. It opens the *first* match, whatever its title (page 5 picks by title).
- **`Book`** holds three facts of `show_book`'s `Output`. A field name is yours to choose:
  `name` is filled from `show_book.title`.
- **`def answer(self, input, book=None)`**: a phase that may have been skipped is read
  through a parameter with the default `None`. `book` is `None` when the `book` phase was
  skipped, so `answer` says that no book matches. Without the default, a skipped `book`
  would fail the `answer` phase.
- **Both branches of `answer` return steps**, so the answer phase always runs. Keep the
  answer phase unskippable: when the last phase is skipped, the answer is the last
  phase's that ran.

## What the planner sees

In the template every fact is unknown, so every test on one reads as true and is noted.
The `book` phase's line ends with the test its method made:

<!-- template: book_details -->
```text
find_book query=<title>
show_book book_uid=<find.book_uids[0]> (only when find.book_uids)
say that <book.name> is by <book.author>; the library owns <book.copies> copies
```

`book is None` is not a test a placeholder can see (a placeholder is never `None`), so the
template shows the answer for a book found. The preview's phase table says when each phase
runs:

<!-- preview: book_details -->
```markdown
## Phases

| phase | output | runs | fans out |
|---|---|---|---|
| `find` | `Found` | always | no |
| `book` | `Book` | only when `find.book_uids` | no |
| `answer` | none | always | no |
```

## At run time

A match:

<!-- dry-run: book_details {"title": "Persuasion"} -->
```text
Phase find
  1. find_book query=Persuasion
  code runs: find_book query=Persuasion
  facts: Found(book_uids=['b4'])
Phase book
  1. show_book book_uid=b4
  code runs: show_book book_uid=b4
  facts: Book(name='Persuasion', author='Jane Austen', copies=1)
Phase answer
  1. say that Persuasion is by Jane Austen; the library owns 1 copies
  answer written by code:
    Persuasion is by Jane Austen; the library owns 1 copies.
```

No match:

<!-- dry-run: book_details {"title": "Ulysses"} -->
```text
Phase find
  1. find_book query=Ulysses
  code runs: find_book query=Ulysses
  facts: Found(book_uids=[])
Phase book: skipped (its method returned [])
Phase answer
  1. say that no book matches Ulysses
  answer written by code:
    No book matches Ulysses.
```

## What you learned

`return []` skips a phase, decided when it is reached; a later phase reads a phase that may
be skipped through `=None`; the template marks the skipped lines "(only when ...)".

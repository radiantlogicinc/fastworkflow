# 7. The shape of a call

**Goal:** one skill for three kinds of request about a book:

- "tell me about Dune": find, open, answer (page 5's phases);
- "tell me about Dune and who has it": the same, plus a phase that lists its loans;
- "do you have Emma at all?": just a search, in one plan, no phases.

Which phases a call runs depends on its input. `PlanGenerator.__call__` says so.

## The skill file

<!-- file: tests/skill_tutorial_workflow/_skills/book_report.py -->
```python
"""book_report: the book the request names, and who has it on loan when asked.

With brief, one plan: find the book and say whether the library has it. Otherwise find,
book (skipped when nothing matched), loans (only when the request asks who has it, and
skipped when no book was opened) and answer.
"""
from pydantic import BaseModel, Field

from fastworkflow.skill import fact_from, listed, pick


class Signature:
    """Show the book whose title the request names: its author and copies, and who has it on
    loan when asked. With brief, only whether the library has it."""

    plain_utterances = ["tell me about Dune and who has it", "do you have Emma at all"]

    class Input(BaseModel):
        title: str = Field(description="the book's title exactly as the request writes it")
        loans: bool = Field(default=False,
                            description="the request asks who has the book on loan")
        brief: bool = Field(default=False,
                            description="the request only asks whether the library has the book")


class PlanGenerator:
    def __call__(self, input):
        if input.brief:
            return [self.check]
        loans = [self.loans] if input.loans else []
        return [self.find, self.book, *loans, self.answer]

    def check(self, input):
        return [f"find_book query={input.title}",
                "say whether the library has the book"]

    class Found(BaseModel):
        book_uids: list[str] = fact_from("find_book.book_uids")
        titles: list[str] = fact_from("find_book.titles")

        def only_uid(self, title):
            """The uid of the one book titled *title* (case ignored), else None."""
            hits = [uid for uid, t in zip(self.book_uids, self.titles)
                    if t.casefold() == title.casefold()]
            return hits[0] if len(hits) == 1 else None

    def find(self, input) -> Found:
        return [f"find_book query={input.title}"]

    class Book(BaseModel):
        uid: str = fact_from("show_book.book_uid")
        name: str = fact_from("show_book.title")
        author: str = fact_from("show_book.author")
        copies: int = fact_from("show_book.copies")

    def book(self, input, find) -> Book:
        if not find.book_uids:
            return []
        return [pick("show_book", book_uid=find.only_uid(input.title), named=input.title)]

    class Loans(BaseModel):
        borrowers: list[str] = fact_from("list_loans.borrowers")
        total: int = fact_from("list_loans.total")

    def loans(self, book=None) -> Loans:
        if book is None:
            return []
        return [f"list_loans book_uid={book.uid} (full list)"]

    def answer(self, input, book=None, loans=None):
        if book is None:
            return [f"say that no book matches {input.title}"]
        lines = [f"say that {book.name} is by {book.author}; the library owns "
                 f"{book.copies} copies"]
        if loans is not None:
            lines.append(f"say that it is on loan to "
                         f"{listed(loans.borrowers, loans.total, 'borrower')}")
        return lines
```

## Line by line

- **`def __call__(self, input)`** returns the phases this call runs, in order, as the
  methods themselves (`self.find`, ...). Without it, a call runs every phase. It is
  written first in the class, so a reader sees the shape before the phases.
- **`loans: bool`**: a `[self.loans]` list that is empty unless the request asks who has
  the book on loan. Input-dependent shape belongs in `__call__`, decided by the input
  before anything runs; a skip that depends on facts belongs in the phase (`return []`).
- **`if input.brief: return [self.check]`**: a call of **one phase that reads and fills no
  facts runs flat**, as one plan, exactly as page 2's skill does. `check` reads only
  `input` and has no return annotation.
- **`def loans(self, book=None)`**: `book` may have been skipped (no match), so `loans`
  reads it through `=None` and skips itself too. `list_loans ... (full list)`: a note
  asking for the full list makes code read every page of `list_loans` (up to 20).
- **`def answer(self, input, book=None, loans=None)`**: `loans` is `None` both when the
  call has no `loans` phase and when it was skipped, so the loans line is written only
  when it ran.

## What the planner sees

The template covers every combination of the inputs, and a line some combinations do not
write says when it is written:

<!-- template: book_report -->
```text
find_book query=<title>
say whether the library has the book (only when brief)
show_book with the uid of the <title> match (only when find.book_uids and not brief)
list_loans book_uid=<book.uid> (full list) (only when loans and not brief)
say that <book.name> is by <book.author>; the library owns <book.copies> copies (only when not brief)
say that it is on loan to <loans.total> borrowers (<loans.borrowers>) (only when loans and not brief)
```

The preview names the input that decides each plan and renders each plan once:

<!-- preview: book_report -->
```markdown
## Phases

The call runs as phases only when brief = false; for other inputs it runs flat, as one plan. The table is about the phased calls.

| phase | output | runs | fans out |
|---|---|---|---|
| `find` | `Found` | always | no |
| `book` | `Book` | only when `find.book_uids` | no |
| `loans` | `Loans` | only when loans = true | no |
| `answer` | none | always | no |
```

<!-- preview: book_report -->
```markdown
### Plan when brief = true

For these inputs the call runs flat: one phase that reads no facts, as one plan.

1. `find_book query=<title>`
2. `say whether the library has the book`

Pages: step 1 reads page 1 only.
```

## At run time

<!-- dry-run: book_report {"title": "Dune", "brief": true} -->
```text
Runs flat: one plan, one executor run.
Phase book_report
  1. find_book query=Dune
  2. say whether the library has the book
  code runs: find_book query=Dune
  the agent writes the answer from these runs
```

<!-- dry-run: book_report {"title": "Dune", "loans": true} -->
```text
Phase find
  1. find_book query=Dune
  code runs: find_book query=Dune
  facts: Found(book_uids=['b1', 'b2'], titles=['Dune', 'Dune Messiah'])
Phase book
  1. show_book book_uid=b1
  code runs: show_book book_uid=b1
  facts: Book(uid='b1', name='Dune', author='Frank Herbert', copies=3)
Phase loans
  1. list_loans book_uid=b1 (full list)
  code runs: list_loans book_uid=b1
  facts: Loans(borrowers=['Ann Lee', 'Raj Patel'], total=2)
Phase answer
  1. say that Dune is by Frank Herbert; the library owns 3 copies
  2. say that it is on loan to 2 borrowers (Ann Lee, Raj Patel)
  answer written by code:
    Dune is by Frank Herbert; the library owns 3 copies.
    It is on loan to 2 borrowers (Ann Lee, Raj Patel).
```

No match, loans asked for: both later phases skip themselves.

<!-- dry-run: book_report {"title": "Ulysses", "loans": true} -->
```text
Phase find
  1. find_book query=Ulysses
  code runs: find_book query=Ulysses
  facts: Found(book_uids=[], titles=[])
Phase book: skipped (its method returned [])
Phase loans: skipped (its method returned [])
Phase answer
  1. say that no book matches Ulysses
  answer written by code:
    No book matches Ulysses.
```

## What you learned

`__call__(self, input)` returns the phases a call runs; a call of one phase that reads and
fills no facts runs flat.

# 11. Composing skills

**Goal:** "who has borrowed Dune?" needs exactly what page 5's `book_by_title` does (find
the book, open the one titled Dune), then one more phase that lists its loans, and its own
answer. Rather than copy those phases, `book_loans` **extends** `book_by_title`'s
`PlanGenerator` with ordinary Python inheritance.

Phase inheritance is read by `fastworkflow.skill.phase_methods`, which walks the class's
bases; a fastworkflow older than that change collects only the subclass's own methods.

## The skill file

<!-- file: tests/skill_tutorial_workflow/_skills/book_loans.py -->
```python
"""book_loans: who has the book the request names on loan.

Reuses book_by_title's find and book phases by inheriting its PlanGenerator, adds a loans
phase and writes its own answer.
"""
from pydantic import BaseModel, Field

from fastworkflow.skill import fact_from, listed

from .book_by_title import PlanGenerator as BookByTitle


class Signature:
    """Say who has the book whose title the request names on loan."""

    plain_utterances = ["who has borrowed Dune", "is anyone reading Emma right now"]

    class Input(BaseModel):
        title: str = Field(description="the book's title exactly as the request writes it")


class PlanGenerator(BookByTitle):
    def __call__(self, input):
        return [self.find, self.book, self.loans, self.answer]

    class Loans(BaseModel):
        borrowers: list[str] = fact_from("list_loans.borrowers")
        total: int = fact_from("list_loans.total")

    def loans(self, book=None) -> Loans:
        if book is None:
            return []
        return [f"list_loans book_uid={book.uid} (full list)"]

    def answer(self, input, book=None, loans=None):
        if book is None or loans is None:
            return [f"say that no book matches {input.title}"]
        return [f"say that {book.name} is on loan to "
                f"{listed(loans.borrowers, loans.total, 'borrower')}"]
```

## Line by line

- **`from .book_by_title import PlanGenerator as BookByTitle`**: a skill file is imported
  as a module of the workflow's package, so a package-relative import reads another skill
  file. `book_by_title` still loads as a skill of its own; importing it changes nothing
  about it.
- **`class PlanGenerator(BookByTitle)`** inherits the base's phases, as Python inherits
  methods:
  - the base's phases come first, in their order: `find`, `book`, `answer`;
  - a method of the same name **overrides** a base phase **in its place**: this `answer`
    replaces `BookByTitle.answer` as the third phase;
  - the new phases follow: `loans` comes fourth.
- It also inherits the base's facts models (`Found`, `Book`, with `Found.only_uid`) and
  its helpers. The inherited `book` phase still fills `Book`, so `loans` and `answer` read
  `book.uid` and `book.name` as page 5 wrote them.
- **`__call__`** puts `loans` before `answer`. Inherited order alone would run `answer`
  third and `loans` last, too late for the answer to read it:

<!-- doctest -->
```pycon
>>> from tests.skill_tutorial_workflow._skills import book_loans
>>> [name for name, _ in phase_methods(book_loans.PlanGenerator)]
['find', 'book', 'answer', 'loans']
```

  A subclass that only adds phases after the base's, or only overrides some, needs no
  `__call__`. A `__call__` is inherited too, like any method: a base's `__call__` that
  names its own phases leaves a subclass's new ones out until the subclass writes its own.
- **`Signature` is not inherited.** A skill's `Signature` is its own job, so `book_loans`
  writes its own description and examples. Its `Input` must hold every field the
  inherited methods read: `book_by_title`'s `find` and `book` read `input.title`.
- **`def loans(self, book=None)`** and **`def answer(self, input, book=None, loans=None)`**
  read the inherited `book` phase by its name, as any phase does.

Two other ways to share between skills work without inheritance: facts models and data
shared by several skills go in a helper module, `_skills/_<name>.py` (a file starting with
`_` is never loaded as a skill; skill files import it with `from ._name import ...`); and
one skill's `__call__` chooses among its own phases per input (page 7).

## What the planner sees

<!-- template: book_loans -->
```text
find_book query=<title>
show_book with the uid of the <title> match (only when find.book_uids)
list_loans book_uid=<book.uid> (full list)
say that <book.name> is on loan to <loans.total> borrowers (<loans.borrowers>)
```

The preview's phase table shows the inherited phases with the new one:

<!-- preview: book_loans -->
```markdown
## Phases

| phase | output | runs | fans out |
|---|---|---|---|
| `find` | `Found` | always | no |
| `book` | `Book` | only when `find.book_uids` | no |
| `loans` | `Loans` | always | no |
| `answer` | none | always | no |
```

## At run time

<!-- dry-run: book_loans {"title": "Dune"} -->
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
  1. say that Dune is on loan to 2 borrowers (Ann Lee, Raj Patel)
  answer written by code:
    Dune is on loan to 2 borrowers (Ann Lee, Raj Patel).
```

<!-- dry-run: book_loans {"title": "Ulysses"} -->
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

A `PlanGenerator` that subclasses another skill's runs the base's phases first, in their
order; a same-named method overrides one in its place; new phases follow, and `__call__`
reorders them when needed.

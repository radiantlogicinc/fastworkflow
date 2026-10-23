# 11. Composing skills

**Goal:** "who has borrowed Dune?" needs exactly what page 5's `book_by_title` does (find
the book, open the one titled Dune, say its author and copies), then one more phase that
lists its loans, and one more line in the answer. Rather than copy those phases,
`book_loans` **extends** `book_by_title`'s `PlanGenerator` with ordinary Python
inheritance, and says explicitly, as Python does, where the base's work goes:

- **`__call__` places the base's phases** among the new ones;
- **`super()` reuses a base phase's steps** inside an override.

## The skill file

<!-- file: tests/skill_tutorial_workflow/_skills/book_loans.py -->
```python
"""book_loans: the book the request names and who has it on loan.

Reuses book_by_title's phases by subclassing its PlanGenerator: __call__ places them and the
new loans phase in order, and answer extends book_by_title's answer with super().
"""
from pydantic import BaseModel, Field

from fastworkflow.skill import fact_from, listed

from .book_by_title import PlanGenerator as BookByTitle


class Signature:
    """Show the book whose title the request names and who has it on loan."""

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
        steps = super().answer(input, book)
        if loans is None:
            return steps
        return [*steps, f"say that it is on loan to "
                        f"{listed(loans.borrowers, loans.total, 'borrower')}"]
```

## Line by line

- **`from .book_by_title import PlanGenerator as BookByTitle`**: a skill file is imported
  as a module of the workflow's package, so a package-relative import reads another skill
  file. `book_by_title` still loads as a skill of its own; importing it changes nothing
  about it.
- **`class PlanGenerator(BookByTitle)`**: `self` now reaches the base's phases (`find`,
  `book`, `answer`), its facts models (`Found`, `Book`, with `Found.only_uid`) and its
  helpers, as Python resolves any method. The inherited `book` phase still fills `Book`,
  so `loans` reads `book.uid` as page 5 wrote it.
- **`__call__` says the order**, base phases included, each placed where this job needs
  it: `find`, `book`, the new `loans`, then `answer`. A base phase can go anywhere in the
  list, or be left out when no phase reads it. Nothing is guessed: a subclass that **adds** a phase must write
  `__call__`, or the load fails, naming the phases it can order:

<!-- doctest -->
```pycon
>>> from tests.skill_tutorial_workflow._skills.book_by_title import PlanGenerator as BookByTitle
>>> class Unordered(BookByTitle):
...     def loans(self, book=None):
...         return [f"list_loans book_uid={book.uid} (full list)"]
>>> planned(Unordered)
Traceback (most recent call last):
...
fastworkflow.skill_phases.PhaseError: Unordered adds the phase(s) loans to those of book_by_title.PlanGenerator: write __call__(self, input) returning the phases a call runs, in order (from: self.find, self.book, self.answer, self.loans)
```

  A subclass that only **overrides** base phases adds nothing to order: it keeps the base's
  plan (the base's `__call__`, or its phases in the order written), each override in its
  base phase's place.
- **`def answer(self, input, book=None, loans=None)`** overrides `BookByTitle.answer`. Its
  parameters are its own: they say what *this* method reads (`loans` too), whatever the
  base's method read.
- **`super().answer(input, book)`** calls the base's `answer` with the arguments it takes
  and gives back its steps, as any Python override reuses its base: the "no book matches"
  step, or the author-and-copies step. The override adds its own line after them. It works
  the same while the template is built: the base's method then gets the placeholder facts.
- **`Signature` is not inherited.** A skill's `Signature` is its own job, so `book_loans`
  writes its own description and examples. Its `Input` must hold every field the
  inherited methods read: `book_by_title`'s `find`, `book` and `answer` read `input.title`.

Two other ways to share between skills work without inheritance: facts models and data
shared by several skills go in a helper module, `_skills/_<name>.py` (a file starting with
`_` is never loaded as a skill; skill files import it with `from ._name import ...`); and
one skill's `__call__` chooses among its own phases per input (page 7).

## What the planner sees

The base's answer step and the override's line, one after the other:

<!-- template: book_loans -->
```text
find_book query=<title>
show_book with the uid of the <title> match (only when find.book_uids)
list_loans book_uid=<book.uid> (full list)
say that <book.name> is by <book.author>; the library owns <book.copies> copies
say that it is on loan to <loans.total> borrowers (<loans.borrowers>)
```

The preview's phase table shows the base's phases where `__call__` placed them:

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
  1. say that Dune is by Frank Herbert; the library owns 3 copies
  2. say that it is on loan to 2 borrowers (Ann Lee, Raj Patel)
  answer written by code:
    Dune is by Frank Herbert; the library owns 3 copies.
    It is on loan to 2 borrowers (Ann Lee, Raj Patel).
```

With no match, `book` and `loans` are skipped and the override returns the base's steps
alone:

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

A `PlanGenerator` that subclasses another skill's reaches the base's phases through `self`.
`__call__` places them, explicitly, among the new ones (a subclass that adds a phase must
write it); an override reuses a base phase's steps with `super()` and reads what its own
parameters name.

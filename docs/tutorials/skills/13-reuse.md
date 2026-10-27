# 13. Reuse without a call

**Goal:** know the three ways one planning command can use another's work, and choose. Page
12 called `book_by_title` as a step. The other two reuse *code*: a `_` helper module, and a
base class kept in one.

## Three ways

| | a nested call (page 12) | a `_` helper module (page 4) | a base class in a `_` module |
|---|---|---|---|
| what is shared | the whole job, its checks and its `Output` | functions, tables, facts models | phases, hooks, helpers |
| what the caller sees | one run and a typed `Output` | ordinary Python | the base's phases, as its own |
| run-time cost | a frame, an argument extraction and check, the callee's navigation | none | none |
| the boundary | the callee's `Input` and `Output` | none | none |

Choose a nested call when the caller wants the callee's whole job, done as the callee does
it, checks included, and should not depend on how the callee does it: `book_loans` reads
`book_by_title`'s `Output`, not its facts. Choose shared code when you want parts of a job,
or when a call's cost buys nothing. Nothing on the nested path is needed to keep using
helpers; a workflow can reuse entirely through `_` modules.

## A `_` helper module

`_commands/Library/_wording.py` (page 4) holds `copies`, the function `book_by_title` and
`author_books` both call. Any file whose name starts with `_` is a module, never a command,
and the commands of its folder import it (`from ._wording import copies`). Shared functions,
tables and facts models go there.

A facts model in a helper is checked once, however many commands use it, and a problem with
it is reported once, naming every command that uses it.

## A base class in a `_` module

A `PlanGenerator` may subclass a class kept in a `_` module, and reach the base's phases,
hooks and helpers through `self`, as Python resolves any method. A sketch, not a file of the
sample:

```python
from ..Book.list_loans import Signature as ListLoans
from ._shelf import BookPhases            # a _ module: phases find and book, and a __result__


class Signature:
    """Show a book and who has it on loan, given its title."""

    class Input(BaseModel):
        title: str = Field(description="the book's title")


class PlanGenerator(BookPhases):
    def __phases__(self, input):
        return [self.find, self.book, self.loans]

    def loans(self, book) -> ListLoans.Output:
        return [] if book is None else [full_list("list_loans")]

    def __result__(self, input, book, loans):
        said = super().__result__(input, book)
        if loans is None:
            return said
        return f"{said} It is on loan to {listed(loans.borrowers, loans.total, 'borrower')}."
```

The rules:

- **`__phases__` says the order**, base phases included, each placed where this job needs
  it. A subclass that **adds** a phase must write `__phases__`, or the load fails, naming the
  phases it can order. A subclass that only **overrides** keeps the base's plan.
- **`super().__result__(input, book)`** reuses the base's answer inside an override, as
  `super().book(...)` would reuse a base phase's steps. The override's parameters are its
  own: they say what *this* method reads.
- **`Signature` is not inherited.** A command's `Signature` is its own job, so the subclass
  writes its own description and examples. Its `Input` must hold every field the inherited
  methods read.
- **A base class is never another command's `PlanGenerator`.** That would tie this command
  to the other command's private phases and facts, which it may change at any time, so the
  load fails, naming both ways out:

  ```text
  _commands/Library/book_and_loans.py: PlanGenerator inherits from book_by_title's PlanGenerator; to reuse its whole job, call book_by_title in a step; to share phases, move them to a base class in a _ module
  ```

- **Hooks a base defines are inherited like any method**, and checked as if the file wrote
  them. A `__result__` that returns an `Output` belongs with the `Signature.Output` it
  builds, so keep that one in the command: inherited by a command whose `Output` is another
  class, it fails at run time, and a model writes the answer (page 11). A command that
  declares an `Output` and has no `__result__`, its own or a base's, fails the load.

## What you learned

Call a planning command to reuse its whole job behind its `Input` and `Output`; share code
through a `_` module, or a base class kept there, to reuse parts of a job at no run-time
cost. A `PlanGenerator` never subclasses another command's.

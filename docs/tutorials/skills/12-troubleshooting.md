# 12. Troubleshooting

A skill file that does not load is skipped: the workflow's other skills still load, the
error is logged, and `fastworkflow skill preview` fails naming the file and its error. So run
the preview after every change. Each error below was produced by saving the file shown as
`_skills/<name>.py` in a copy of the tutorial workflow (the test does this on every run);
the preview prints the same message with the file's full path.

## A step written in a form the runtime would misread: `StepFormError`

`pick` takes exactly one `..._uid=` parameter and a `named=` name. Here the parameter is
`uid=`:

<!-- broken: bad_pick -->
```python
from pydantic import BaseModel, Field

from fastworkflow.skill import fact_from, pick


class Signature:
    """Show the book whose title the request names."""

    class Input(BaseModel):
        title: str = Field(description="the book's title exactly as the request writes it")


class PlanGenerator:
    class Found(BaseModel):
        book_uids: list[str] = fact_from("find_book.book_uids")

    def find(self, input) -> Found:
        return [f"find_book query={input.title}"]

    def book(self, input, find):
        return [pick("show_book", uid=find.book_uids[0], named=input.title)]
```

<!-- load-error -->
```text
SkillFormatError: _skills/bad_pick.py: phase 'book' fails while its template is built (every fact unknown): StepFormError: pick('show_book'): give exactly one uid parameter (..._uid=...), got ['uid']
```

**Fix:** name the command's uid parameter: `pick("show_book", book_uid=..., named=...)`.

## A method that fails on unknown facts: `TemplateError`

While the template is built, every fact is a placeholder, which is text. Arithmetic on it
fails, and so does the load:

<!-- broken: bad_arithmetic -->
```python
from pydantic import BaseModel, Field

from fastworkflow.skill import fact_from


class Signature:
    """Say how many other books match the request's words."""

    class Input(BaseModel):
        query: str = Field(description="the title or author words as the request writes them")


class PlanGenerator:
    class Found(BaseModel):
        total: int = fact_from("find_book.total")

    def find(self, input) -> Found:
        return [f"find_book query={input.query}"]

    def answer(self, input, find):
        return [f"say that {find.total - 1} other books match {input.query}"]
```

<!-- load-error -->
```text
SkillFormatError: _skills/bad_arithmetic.py: phase 'answer' fails while its template is built (every fact unknown): TypeError: unsupported operand type(s) for -: 'Placeholder' and 'int'
```

**Fix:** compute from facts in a helper (`self._others(find)`): a helper called with an
unknown fact is not run while the template is built, it reads as its name (`<others>`). At
run time it gets the real facts.

## A method that writes `None` into a step

A value the method formats is `None` for some input: here a table without an entry for one
of the enum's values. The template is rendered for every value of `kind`, so the load
catches it:

<!-- broken: bad_none -->
```python
from typing import Literal

from pydantic import BaseModel, Field

from fastworkflow.skill import fact_from


class Signature:
    """Say how many books, members or branches have the request's words in their name."""

    class Input(BaseModel):
        name: str = Field(description="the name exactly as the request writes it")
        kind: Literal["book", "member", "branch"] = Field(
            default="book", description="what the request is about")


class PlanGenerator:
    class Found(BaseModel):
        total: int = fact_from("find_book.total")

    def find(self, input) -> Found:
        return [f"{FIND.get(input.kind)} query={input.name}"]

    def answer(self, input, find):
        return [f"say that {find.total} match {input.name}"]


FIND = {"book": "find_book", "member": "find_member"}
```

<!-- load-error -->
```text
SkillFormatError: _skills/bad_none.py: phase 'find' writes None into a step ('None query=<name>'): a value it formats is None for some input
```

**Fix:** give every value its entry (`FIND = {..., "branch": ...}`), or drop the value
from the `Literal`. (This check is made on a phased skill's methods; a flat skill's
template shows the `None` as written, so read its preview.)

## A step naming an unknown command or parameter

Every command step of every input combination is checked against the workflow's commands
and each command's `Signature.Input`:

<!-- broken: bad_command -->
```python
from pydantic import BaseModel, Field


class Signature:
    """Find the books whose title the request names."""

    class Input(BaseModel):
        title: str = Field(description="the book's title exactly as the request writes it")


class PlanGenerator:
    def search(self, input):
        return [f"find_books query={input.title}",
                f"find_book title={input.title}",
                "say which books were found"]
```

<!-- load-error -->
```text
_skills/bad_command.py: a step runs 'find_books', which is not a command of the workflow
_skills/bad_command.py: a step passes title to 'find_book', whose Signature.Input has no such field
```

**Fix:** use the command's name and its `Input` fields (`find_book query=...`). Text that
is not a command belongs after the command in parentheses (`find_book query=Dune (the
first page)`), which is never sent.

## A parameter that names no earlier phase

A phase's parameters bind by name to earlier phases. `found` is not a phase (`find` is):

<!-- broken: bad_binding -->
```python
from pydantic import BaseModel, Field

from fastworkflow.skill import fact_from


class Signature:
    """Say how many books match the request's words."""

    class Input(BaseModel):
        query: str = Field(description="the title or author words as the request writes them")


class PlanGenerator:
    class Found(BaseModel):
        total: int = fact_from("find_book.total")

    def find(self, input) -> Found:
        return [f"find_book query={input.query}"]

    def answer(self, input, found):
        return [f"say that {found.total} books match {input.query}"]
```

<!-- load-error -->
```text
SkillFormatError: _skills/bad_binding.py: plan(): phase 'answer' takes 'found', which is not an earlier phase with an output; give the parameter a default or build the phase without it
```

**Fix:** name the phase (`def answer(self, input, find)`). A phase that may be skipped is
read with a default: `book=None`.

## A subclass that adds a phase without saying where

A `PlanGenerator` that subclasses another and adds a phase must write `__call__` with the
order (page 11). Here `Overview` adds `overdue` to `Branches`' phase:

<!-- broken: bad_order -->
```python
class Signature:
    """List the library's branches, then say how many loans are overdue."""


class Branches:
    def branches(self):
        return ["list_branches"]


class PlanGenerator(Branches):
    def overdue(self):
        return ["count_overdue", "say how many loans are overdue across the library"]
```

<!-- load-error -->
```text
SkillFormatError: _skills/bad_order.py: PlanGenerator adds the phase(s) overdue to those of bad_order.Branches: write __call__(self, input) returning the phases a call runs, in order (from: self.branches, self.overdue)
```

**Fix:** write the order: `def __call__(self, input): return [self.branches, self.overdue]`.

## A fact source the workflow does not have

`fact_from("find_book.matches")` names a field `find_book` does not have. The skill still
loads (a workflow may add the command's field later) with this warning, and
`fastworkflow skill preview --check` fails on it:

<!-- broken: bad_source -->
```python
from pydantic import BaseModel, Field

from fastworkflow.skill import fact_from


class Signature:
    """Say how many books match the request's words."""

    class Input(BaseModel):
        query: str = Field(description="the title or author words as the request writes them")


class PlanGenerator:
    class Found(BaseModel):
        matches: int = fact_from("find_book.matches")

    def find(self, input) -> Found:
        return [f"find_book query={input.query}"]

    def answer(self, input, find):
        return [f"say that {find.matches} books match {input.query}"]
```

<!-- check-error -->
```text
_skills/bad_source.py: Found.matches (from 'find_book.matches'): find_book has no Output or Input field 'matches'
```

**Fix:** use a field of the command's `Output` (or `Input`): `fact_from("find_book.total")`.
At run time a required fact no run holds fails its phase.

## A fact named like a `str` method

While the template is built, a phase's facts are one placeholder, which is a `str`. A fact
named like a `str` method (`title`, `count`, `index`, `format`, `split`, ...) reads as
that method, not as the fact. The skill loads and its preview looks right (the preview
fills each fact separately), but the template, what the planner sees, reads:

<!-- broken: bad_fact_name -->
```python
from pydantic import BaseModel, Field

from fastworkflow.skill import fact_from


class Signature:
    """Show the title of the book with the uid the request gives."""

    class Input(BaseModel):
        uid: str = Field(description="the book's uid exactly as the request writes it")


class PlanGenerator:
    class Book(BaseModel):
        title: str = fact_from("show_book.title")

    def book(self, input) -> Book:
        return [f"show_book book_uid={input.uid}"]

    def answer(self, book):
        return [f"say that the book is {book.title}"]
```

<!-- template-line -->
```text
say that the book is <built-in method title of Placeholder object at 0x
```

**Fix:** name the fact otherwise: `name: str = fact_from("show_book.title")`, as these
pages do.

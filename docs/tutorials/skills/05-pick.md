# 5. Picking a row by name

**Goal:** "who wrote Dune?" `find_book query=Dune` lists two books, Dune and Dune Messiah.
Page 4 would open the first row; this skill opens the row *titled* Dune, with `pick`.

## The skill file

<!-- file: tests/skill_tutorial_workflow/_skills/book_by_title.py -->
```python
"""book_by_title: the book whose title the request names, picked by its title.

find lists the matches; book opens the one whose title is the request's title (pick), and
is skipped when nothing matched; answer says what book found, or that no book matches.
"""
from pydantic import BaseModel, Field

from fastworkflow.skill import fact_from, pick


class Signature:
    """Show the author and copies of the book whose title the request names."""

    plain_utterances = ["who wrote Dune", "how many copies of Dune Messiah are there"]

    class Input(BaseModel):
        title: str = Field(description="the book's title exactly as the request writes it")


class PlanGenerator:
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

    def answer(self, input, book=None):
        if book is None:
            return [f"say that no book matches {input.title}"]
        return [f"say that {book.name} is by {book.author}; the library owns "
                f"{book.copies} copies"]
```

## Line by line

- **`Found` now keeps `titles`** too, and has a method, `only_uid(title)`: the uid of the
  one row whose title is the request's title, else `None`. A facts model is an ordinary
  pydantic model; methods on it are plain Python.
- **`pick("show_book", book_uid=..., named=input.title)`** writes the step that opens one
  row of the listing. It takes the command, exactly one `..._uid=` parameter, and the name
  the row is picked by:
  - when the uid is known, it writes `show_book book_uid=b1`, which code runs;
  - when it is `None` (no single exact title), it writes `show_book with the uid of the
    Dune Messia match`, the wording the agent (or code, right after the listing in the same
    phase) picks a row by.
- **`uid` is filled from `show_book.book_uid`**, the record actually opened.

## What the planner sees

`find.only_uid(...)` on a placeholder is unknown, so the template shows the pick wording:

<!-- template: book_by_title -->
```text
find_book query=<title>
show_book with the uid of the <title> match (only when find.book_uids)
say that <book.name> is by <book.author>; the library owns <book.copies> copies
```

## At run time

The exact title is one row: code opens it.

<!-- dry-run: book_by_title {"title": "Dune"} -->
```text
Phase find
  1. find_book query=Dune
  code runs: find_book query=Dune
  facts: Found(book_uids=['b1', 'b2'], titles=['Dune', 'Dune Messiah'])
Phase book
  1. show_book book_uid=b1
  code runs: show_book book_uid=b1
  facts: Book(uid='b1', name='Dune', author='Frank Herbert', copies=3)
Phase answer
  1. say that Dune is by Frank Herbert; the library owns 3 copies
  answer written by code:
    Dune is by Frank Herbert; the library owns 3 copies.
```

Case does not matter to `only_uid`:

<!-- dry-run: book_by_title {"title": "dune messiah"} -->
```text
Phase find
  1. find_book query=dune messiah
  code runs: find_book query=dune messiah
  facts: Found(book_uids=['b2'], titles=['Dune Messiah'])
Phase book
  1. show_book book_uid=b2
  code runs: show_book book_uid=b2
  facts: Book(uid='b2', name='Dune Messiah', author='Frank Herbert', copies=1)
Phase answer
  1. say that Dune Messiah is by Frank Herbert; the library owns 1 copies
  answer written by code:
    Dune Messiah is by Frank Herbert; the library owns 1 copies.
```

No title is exactly "Dune Messia", so `only_uid` gives `None` and the step is written in
pick words. Each phase is a fresh run, so code has no listing in this phase to pick from;
the agent runs the step:

<!-- dry-run: book_by_title {"title": "Dune Messia"} -->
```text
Phase find
  1. find_book query=Dune Messia
  code runs: find_book query=Dune Messia
  facts: Found(book_uids=['b2'], titles=['Dune Messiah'])
Phase book
  1. show_book with the uid of the Dune Messia match
  step 1 is open: the agent runs it and the rest (the dry run stops here)
```

## What you learned

`pick(command, x_uid=..., named=...)` opens the row a listing named: by its uid when the
facts give one, else in words the agent (or code) picks by.

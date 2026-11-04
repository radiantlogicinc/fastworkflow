# 8. Once per item

**Goal:** "what do you have by Frank Herbert, and how many copies of each?" Find the author's
books, then open **each one**, each in its own small sub-plan, and name them all in the
answer.

## The planning command file

<!-- file: fastworkflow/examples/library/_commands/Library/author_books.py -->
```python
"""author_books: each book by the author the request names, opened one by one.

find lists the author's books; book runs once per book found (at most five), each its own
sub-plan that opens that book; __result__ names every book with its copies.
"""
from pydantic import BaseModel, Field

from fastworkflow.planning import phase, run

from ._wording import copies
from .find_book import Signature as FindBook
from .open_book import Signature as OpenBook


class Signature:
    """Name each book by an author, with how many copies the library owns."""

    plain_utterances = [
        "what do you have by Frank Herbert, and how many copies of each",
        "list every Austen book we own with its copies",
        "what Herbert titles are on the shelves, and how many of each",
        "give me each book by Jane Austen with the copy count",
        "all the books by Frank Herbert, with copies",
        "name each title by Austen and how many copies it has",
        "which books by Jane Austen do we hold, copies included",
        "catalogue of Frank Herbert's books and copies",
    ]

    class Input(BaseModel):
        author: str = Field(description="the author's name",
                            json_schema_extra={"in_request": True})


class PlanGenerator:
    def find(self, input) -> FindBook.Output:
        return [run("find_book", query=input.author)]

    @phase(over="find.book_uids", max_items=5, noun="books")
    def book(self, uid) -> OpenBook.Output:
        return [run("open_book", book_uid=uid)]

    def __result__(self, input, book):
        if not book:
            return f"No book by {input.author} was found."
        each = "; ".join(f"{b.title} ({copies(b.copies)})" for b in book)
        return f"The books by {input.author}: {each}."
```

## Line by line

- **`@phase(over="find.book_uids")`** makes `book` a **fan-out**: it runs once per item of
  the `find` phase's `book_uids` list. Empty values and repeats are dropped.
- **`def book(self, uid)`**: the one parameter that is neither `input` nor a phase gets the
  item. Each item is its own executor run, with its own step budget.
- **`-> OpenBook.Output`**: each item hands on the `Output` of the `open_book` run in its own
  sub-plan.
- **`max_items=5`** caps the items run; the answer then ends with a note saying only the
  first 5 were covered. **`noun="books"`** is the word that note uses.
- **Each item is bound to the record it opens.** That is the default (`bind=True`): an
  item's sub-plan must open the item's own record, here the book `open_book` opens, and its
  facts are read on that record. An item whose run never opens its record fails with
  `item b3 was never opened`. Write `bind=False` only when the items are not records.
- **The second item starts on the first item's book.** Nothing moves the workflow back
  between items (page 5), so item `b4`'s `open_book` is run while the workflow stands on
  Emma. `open_book` is a `Library` command, and the library encloses every book, so the
  step finds it there and runs it on the library (page 14). The workflow ends on the last
  book opened.
- **`def __result__(self, input, book)`**: named after the fan-out, `book` is the **list**
  of the items' facts, in item order: here, one `open_book` `Output` per book. It is empty
  when the find listed nothing, so `if not book` says that no book was found. `copies` is
  the `_wording.py` helper `book_by_title` uses too (page 4).

The template the planner reads for `author_books`:

<!-- template: author_books -->
```text
find_book query=<author>
show_book book_uid=<uid>
```

## What the agent plans from

The fan-out's steps are written once, with the item as a placeholder:

```text
find_book query=<author>
open_book book_uid=<uid>
```

The preview renders the fan-out for two items, and the answer with one item standing for
each:

```markdown
## Phases

| phase | output | runs | fans out |
|---|---|---|---|
| `find` | `Library/find_book`'s `Output` | always | no |
| `book` | `Library/open_book`'s `Output` | always | per item of `find.book_uids` as `uid` (books); at most 5 |
```

```markdown
### Phase 2: `book`

Reads no earlier facts. Output: `Output`. Settings: `over='find.book_uids'`, `item='uid'`, `max_items=5`, `noun='books'`.

For the first item, `uid` = `<find.book_uids[0]>`:

1. `open_book book_uid=<find.book_uids[0]>`

For each later item (shown for the second), `uid` = `<find.book_uids[1]>`:

1. `open_book book_uid=<find.book_uids[1]>`

## Answer

Written by `__result__`: The books by <author>: <book[*].title> (<book[*].copies> copies).
```

The preview: `book` fans out once per book the search found.

<!-- preview: author_books -->
```markdown
| phase | output | runs | fans out |
|---|---|---|---|
| `find` | `Found` | always | no |
| `book` | `Book` | always | per item of `find.book_uids` as `uid` (books); at most 5; not bound to a record |
```

## At run time

```text
Phase find
  1. find_book query=Jane Austen
  code runs: find_book query=Jane Austen
  facts: find_book.Output(book_uids=['b3', 'b4'], titles=['Emma', 'Persuasion'], total=2)
Phase book (item b3)
  1. open_book book_uid=b3
  code runs: open_book book_uid=b3
  now in: Book: Emma (b3)
  facts: open_book.Output(book_uid='b3', title='Emma', author='Jane Austen', copies=2)
Phase book (item b4)
  1. open_book book_uid=b4
  code runs: open_book book_uid=b4
  now in: Book: Persuasion (b4)
  facts: open_book.Output(book_uid='b4', title='Persuasion', author='Jane Austen', copies=1)
answer written by __result__:
  The books by Jane Austen: Emma (2 copies); Persuasion (1 copy).
```

```text
Phase find
  1. find_book query=Tolstoy
  code runs: find_book query=Tolstoy
  facts: find_book.Output(book_uids=[], titles=[], total=0)
Phase book: no items, so it runs nothing
answer written by __result__:
  No book by Tolstoy was found.
```

Run offline for Jane Austen, `book` runs once for each of the two books:

<!-- dry-run: author_books {"author": "Jane Austen"} -->
```text
Phase find
  1. find_book query=Jane Austen
  code runs: find_book query=Jane Austen
  facts: Found(book_uids=['b3', 'b4'])
Phase book (item b3)
  1. show_book book_uid=b3
  code runs: show_book book_uid=b3
  facts: Book(name='Emma', copies=2)
Phase book (item b4)
  1. show_book book_uid=b4
  code runs: show_book book_uid=b4
  facts: Book(name='Persuasion', copies=1)
answer written by __result__:
  The books by Jane Austen: Emma (2 copies); Persuasion (1 copy).
```

## What you learned

`@phase(over="phase.list_field")` runs a phase once per item; its item parameter gets each
item, and whatever reads it later gets the list of the items' facts. Each item is bound to
the record it opens, and each item starts where the last one ended.

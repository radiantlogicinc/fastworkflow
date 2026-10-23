# 8. Once per item

**Goal:** "what do you have by Frank Herbert, and how many copies of each?" Find the
author's books, then open **each one**, each in its own small sub-plan, and name them all in
the answer.

## The skill file

<!-- file: tests/skill_tutorial_workflow/_skills/author_books.py -->
```python
"""author_books: each book by the author the request names, opened one by one.

find lists the author's books; book runs once per book found (at most five), each its own
sub-plan; answer names every book with its copies.
"""
from pydantic import BaseModel, Field

from fastworkflow.skill import count, fact_from, phase


class Signature:
    """Name each book by the author the request names, with how many copies the library owns."""

    plain_utterances = ["what do you have by Frank Herbert, and how many copies of each"]

    class Input(BaseModel):
        author: str = Field(description="the author's name exactly as the request writes it")


class PlanGenerator:
    class Found(BaseModel):
        book_uids: list[str] = fact_from("find_book.book_uids")

    def find(self, input) -> Found:
        return [f"find_book query={input.author}"]

    class Book(BaseModel):
        name: str = fact_from("show_book.title")
        copies: int = fact_from("show_book.copies")

    @phase(over="find.book_uids", max_items=5, noun="books", bind=False)
    def book(self, uid) -> Book:
        return [f"show_book book_uid={uid}"]

    def answer(self, input, book=None):
        return [f"say that {self._books_by(input.author, book)}"]

    def _books_by(self, author, book):
        if not book:
            return f"no book by {author} was found"
        copies = "; ".join(f"{b.name} ({count(b.copies, 'copy', 'copies')})" for b in book)
        return f"the books by {author}: {copies}"
```

## Line by line

- **`@phase(over="find.book_uids")`** makes `book` a **fan-out**: it runs once per item of
  the `find` phase's `book_uids` list. Empty values and repeats are dropped.
- **`def book(self, uid)`**: the one parameter that is neither `input` nor a phase gets the
  item. Each item is its own executor run, with its own step budget.
- **`max_items=5`** caps the items run; the answer then ends with a note saying only the
  first 5 were covered. **`noun="books"`** is the word that note uses.
- **`bind=False`**: by default each item is bound to the record its run opens (a context
  whose label names the item). This toy workflow has no record contexts, so its items are
  not records; in a workflow where `show_book` opens a Book context, leave `bind` out.
- **`def answer(self, input, book=None)`**: a later phase named after the fan-out gets the
  **list** of the items' facts, in item order.
- **`self._books_by(...)`** is a helper (a `_name` method is not a phase). The test
  `if not book` lives in the helper, so the answer phase itself always runs. While the
  template is built, a helper called with an unknown fact is not run; it reads as its name.

## What the planner sees

The fan-out's steps are written once, with the item as a placeholder, and the helper reads
as its name:

<!-- template: author_books -->
```text
find_book query=<author>
show_book book_uid=<uid>
say that <books by>
```

The preview renders the fan-out for two items and the helper on placeholder facts:

<!-- preview: author_books -->
```markdown
## Phases

| phase | output | runs | fans out |
|---|---|---|---|
| `find` | `Found` | always | no |
| `book` | `Book` | always | per item of `find.book_uids` as `uid` (books); at most 5; not bound to a record |
| `answer` | none | always | no |
```

<!-- preview: author_books -->
```markdown
### Phase 2: `book`

Reads no earlier facts. Output: `Book`. Settings: `over='find.book_uids'`, `item='uid'`, `max_items=5`, `bind=False`, `noun='books'`.

For the first item, `uid` = `<find.book_uids[0]>`:

1. `show_book book_uid=<find.book_uids[0]>`

For each later item (shown for the second), `uid` = `<find.book_uids[1]>`:

1. `show_book book_uid=<find.book_uids[1]>`

### Phase 3: `answer`

Reads: `book`. Output: none.

_Steps below are rendered with placeholder facts: `<phase.fact>` stands for that fact as read at run time, and 999999 (or 999,999) for a number fact the code converted; values computed from facts (counts, comparisons, matches) are not real._

1. `say that the books by <author>: <book[0].name> (<book[0].copies> copy); <book[1].name> (<book[1].copies> copy)`

Answer: written by code, word for word (step 1 verbatim).
```

The "(... copy)" there is `count` reading a placeholder number as 1: values computed from
placeholder facts are not real, as the note above the steps says.

## At run time

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
Phase answer
  1. say that the books by Jane Austen: Emma (2 copies); Persuasion (1 copy)
  answer written by code:
    The books by Jane Austen: Emma (2 copies); Persuasion (1 copy).
```

<!-- dry-run: author_books {"author": "Tolstoy"} -->
```text
Phase find
  1. find_book query=Tolstoy
  code runs: find_book query=Tolstoy
  facts: Found(book_uids=[])
Phase book: no items, so it runs nothing
Phase answer
  1. say that no book by Tolstoy was found
  answer written by code:
    No book by Tolstoy was found.
```

## What you learned

`@phase(over="phase.list_field")` runs a phase once per item; its item parameter gets each
item, and later phases get the list of the items' facts.

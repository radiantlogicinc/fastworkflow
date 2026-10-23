# 6. Wording the answer

**Goal:** answers that read well for any number of rows: "2 books (Emma, Persuasion)",
"1 book", "none (0 books)". `fastworkflow.skill` has three small text helpers for answer
steps: `listed`, `count` and `num`.

## The skill file

<!-- file: tests/skill_tutorial_workflow/_skills/list_books.py -->
```python
"""list_books: how many books match the words the request gives, and their titles.

count_books with its answer worded by listed(): "2 books (Dune, Dune Messiah)", "none (0
books)", or "12 books (...; 10 of 12 shown)" when the page read holds fewer than all.
"""
from pydantic import BaseModel, Field

from fastworkflow.skill import fact_from, listed


class Signature:
    """Say how many books have the request's words in their title or author, and name them."""

    plain_utterances = ["which books by Jane Austen do you have"]

    class Input(BaseModel):
        query: str = Field(description="the title or author words exactly as the request "
                                       "writes them")


class PlanGenerator:
    class Found(BaseModel):
        titles: list[str] = fact_from("find_book.titles")
        total: int = fact_from("find_book.total")

    def find(self, input) -> Found:
        return [f"find_book query={input.query}"]

    def answer(self, input, find):
        return [f"say that {input.query} matches {listed(find.titles, find.total, 'book')}"]
```

## Line by line

- **`listed(find.titles, find.total, 'book')`** writes a list fact with its total:
  "N books (a, b)", "none (0 books)", and "K of N shown" when the page read holds fewer
  rows than the total.
- It is imported from `fastworkflow.skill`, with `fact_from`. A skill imports what it uses
  from there: `fact_from`, `pick`, `phase`, `num`, `count`, `listed`.

The three helpers, as they behave:

<!-- doctest -->
```pycon
>>> listed(["Emma", "Persuasion"], 2, "book")
'2 books (Emma, Persuasion)'
>>> listed([], 0, "book")
'none (0 books)'
>>> listed(["Dune"], 12, "book")
'12 books (Dune; 1 of 12 shown)'
>>> count(1, "book"), count(3, "copy", "copies")
('1 book', '3 copies')
>>> num(4.0), num(2.5), num(5000000.0, grouped=True), num(None)
('4', '2.5', '5,000,000', 'none recorded')
```

`count` fixes page 3's "1 books"; `num` writes a number as read (4.0 is "4"), with
`grouped=True` for thousands, and "none recorded" for a missing value.

## What the planner sees

On a placeholder each helper writes the placeholder in its place:

<!-- template: list_books -->
```text
find_book query=<query>
say that <query> matches <find.total> books (<find.titles>)
```

## At run time

<!-- dry-run: list_books {"query": "Austen"} -->
```text
Phase find
  1. find_book query=Austen
  code runs: find_book query=Austen
  facts: Found(titles=['Emma', 'Persuasion'], total=2)
Phase answer
  1. say that Austen matches 2 books (Emma, Persuasion)
  answer written by code:
    Austen matches 2 books (Emma, Persuasion).
```

<!-- dry-run: list_books {"query": "Tolstoy"} -->
```text
Phase find
  1. find_book query=Tolstoy
  code runs: find_book query=Tolstoy
  facts: Found(titles=[], total=0)
Phase answer
  1. say that Tolstoy matches none (0 books)
  answer written by code:
    Tolstoy matches none (0 books).
```

## What you learned

Write answer facts with `listed`, `count` and `num`, so the sentence is right for zero,
one or many rows.

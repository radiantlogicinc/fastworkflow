# 3. A fact passed from one phase to the next

**Goal:** "how many books by Jane Austen do you have?" answered by code, with no model
writing the answer. Page 1's phases passed nothing on; here the `find` phase runs the search
and keeps its total as a **fact**, and the `answer` phase writes the answer from that fact.

## The skill file

<!-- file: tests/skill_tutorial_workflow/_skills/count_books.py -->
```python
"""count_books: how many books match the words the request gives.

The find phase runs find_book and keeps its total as a fact; the answer phase writes the
answer from that fact.
"""
from pydantic import BaseModel, Field

from fastworkflow.skill import fact_from


class Signature:
    """Say how many books have the request's words in their title or author."""

    plain_utterances = ["how many books by Jane Austen do you have"]

    class Input(BaseModel):
        query: str = Field(description="the title or author words exactly as the request "
                                       "writes them")


class PlanGenerator:
    class Found(BaseModel):
        total: int = fact_from("find_book.total")

    def find(self, input) -> Found:
        return [f"find_book query={input.query}"]

    def answer(self, input, find):
        return [f"say that {find.total} books match {input.query}"]
```

## Line by line

- **Two phases**, as on page 1: `find`, then `answer`. Nothing passes from one phase to the
  next but the workflow's position and the facts, so `find` declares the fact it hands on.
- **`class Found(BaseModel)`** is a **facts model**: what the `find` phase hands on. It is
  nested right above the method that fills it.
- **`total: int = fact_from("find_book.total")`** says where the fact comes from: the
  `total` field of `find_book`'s `Output` (else its `Input`). After the phase, code fills it
  from the `find_book` run of that phase, with no model call. A required fact no run holds
  fails the phase.
- **`-> Found`** (the return annotation) is how `find` says it fills `Found`.
- **`def answer(self, input, find)`**: a parameter named after an earlier phase gets that
  phase's facts. `find.total` is the number `find_book` returned. A misspelt name fails the
  load.
- **`f"say that {find.total} books match {input.query}"`**: a `say that ...` step. A last
  phase made only of `say that` steps (and guards such as `apply no change`) runs no
  command, so it is **answer-only**: code writes the answer, each `say that P` as the
  sentence `P.`

## What the planner sees

The planner sees the phases' steps joined, every fact a placeholder named by where it comes
from, `<find.total>`:

<!-- template: count_books -->
```text
find_book query=<query>
say that <find.total> books match <query>
```

The preview adds a table of the phases and of the facts each fills, with their sources:

<!-- preview: count_books -->
```markdown
## Phases

| phase | output | runs | fans out |
|---|---|---|---|
| `find` | `Found` | always | no |
| `answer` | none | always | no |

## Facts

What each phase's output holds, filled in code from the commands the phase ran.

### `find` -> `Found`

| fact | type | from |
|---|---|---|
| `total` | `int` | `find_book.total` |
```

and, for the answer phase, who writes the answer:

<!-- preview: count_books -->
```markdown
### Phase 2: `answer`

Reads: `find`. Output: none.

_Steps below are rendered with placeholder facts: `<phase.fact>` stands for that fact as read at run time, and 999999 (or 999,999) for a number fact the code converted; values computed from facts (counts, comparisons, matches) are not real._

1. `say that <find.total> books match <query>`

Answer: written by code, word for word (step 1 verbatim).
```

## At run time

For "how many Dune books do you have" (`query=Dune`):

<!-- dry-run: count_books {"query": "Dune"} -->
```text
Phase find
  1. find_book query=Dune
  code runs: find_book query=Dune
  facts: Found(total=2)
Phase answer
  1. say that 2 books match Dune
  answer written by code:
    2 books match Dune.
```

Code ran `find_book`, filled `Found(total=2)` from its output, called `answer` with it, and
wrote the answer from the `say that` step: no model wrote a word of it. With one match the
same code writes "1 books"; page 6 fixes that wording.

Page 1's overview could say its branches the same way: a `branches` phase that fills
`total: int = fact_from("list_branches.total")`, read by `overdue` as a parameter named
`branches`.

## What you learned

A facts model with `fact_from("command.field")` is what a phase hands on, and a later phase
reads it by naming the phase as a parameter.

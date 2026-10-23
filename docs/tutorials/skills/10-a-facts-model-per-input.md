# 10. A facts model chosen by the input

**Goal:** one skill for "is there a member named Lee?" and "do you have a book called
Emma?". The search command, and so the facts model, depend on what the request is about.

## The skill file

<!-- file: tests/skill_tutorial_workflow/_skills/find_record.py -->
```python
"""find_record: how many books or members match a name, and which.

One find phase whose facts model depends on the kind the request asks about: a book search
fills BookMatch, a member search MemberMatch. Both have labels and total, so the answer
reads either.
"""
from typing import Literal

from pydantic import BaseModel, Field

from fastworkflow.skill import fact_from, listed, phase


class Signature:
    """Say how many books or members match the name the request gives, and name them."""

    plain_utterances = ["do you have a book called Emma", "is there a member named Lee"]

    class Input(BaseModel):
        name: str = Field(description="the name exactly as the request writes it")
        kind: Literal["book", "member"] = Field(
            default="book", description="whether the request is about a book or a member")


class BookMatch(BaseModel):
    labels: list[str] = fact_from("find_book.titles")
    total: int = fact_from("find_book.total")


class MemberMatch(BaseModel):
    labels: list[str] = fact_from("find_member.names")
    total: int = fact_from("find_member.total")


class PlanGenerator:
    @phase(output=lambda input: BookMatch if input.kind == "book" else MemberMatch)
    def find(self, input):
        return [f"{FIND[input.kind]} query={input.name}"]

    def answer(self, input, find):
        return [f"say that {input.name} matches {listed(find.labels, find.total, input.kind)}"]


FIND = {"book": "find_book", "member": "find_member"}
```

## Line by line

- **`kind: Literal["book", "member"]`** is an enum input; `default="book"` makes it
  optional.
- **`BookMatch` and `MemberMatch`** fill the same field names (`labels`, `total`) from
  different commands, so a later phase reads either the same way. Shared facts models are
  written at module level, above `PlanGenerator`.
- **`@phase(output=lambda input: ...)`**: a return annotation names one model, so a model
  that depends on the input goes in `output=`, a function of `input` alone. It is decided
  when the call is planned, before anything runs. Any `@phase` option may be written this
  way.
- **`FIND[input.kind]`** looks the command up in a module-level table.

## What the planner sees

The lines of an enum's values are merged, each difference written once with the input
that decides it:

<!-- template: find_record -->
```text
<find_book or find_member, by kind> query=<name>
say that <name> matches <find.total> <books or members, by kind> (<find.labels>)
```

The preview's phase table lists the model with the input that gives it:

<!-- preview: find_record -->
```markdown
## Phases

| phase | output | runs | fans out |
|---|---|---|---|
| `find` | when kind = book: `BookMatch`<br>when kind = member: `MemberMatch` | always | no |
| `answer` | none | always | no |
```

## At run time

<!-- dry-run: find_record {"name": "Lee", "kind": "member"} -->
```text
Phase find
  1. find_member query=Lee
  code runs: find_member query=Lee
  facts: MemberMatch(labels=['Ann Lee'], total=1)
Phase answer
  1. say that Lee matches 1 member (Ann Lee)
  answer written by code:
    Lee matches 1 member (Ann Lee).
```

<!-- dry-run: find_record {"name": "Emma"} -->
```text
Phase find
  1. find_book query=Emma
  code runs: find_book query=Emma
  facts: BookMatch(labels=['Emma'], total=1)
Phase answer
  1. say that Emma matches 1 book (Emma)
  answer written by code:
    Emma matches 1 book (Emma).
```

## What you learned

`@phase(output=lambda input: Model)` chooses a phase's facts model per call, from the
input.

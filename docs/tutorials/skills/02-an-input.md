# 2. An input

**Goal:** a skill for "find books by Jane Austen". It takes one value from the request, the
words to search for, and writes them into its steps.

## The skill file

<!-- file: tests/skill_tutorial_workflow/_skills/find_books.py -->
```python
"""find_books: the books whose title or author holds the words the request gives."""
from pydantic import BaseModel, Field


class Signature:
    """Find the books whose title or author holds the request's words, and name them."""

    plain_utterances = ["find books by Jane Austen", "do you have Dune"]

    class Input(BaseModel):
        query: str = Field(description="the title or author words exactly as the request "
                                       "writes them")


class PlanGenerator:
    def search(self, input):
        return [f"find_book query={input.query}",
                "say which books were found"]
```

## Line by line

- **`class Input(BaseModel)`** declares what the skill takes, as a command's `Input` does.
  The planner fills it from the request in one typed call, and the arguments are checked
  before any step is written.
- **`query: str`** with no default is **required**. A field is a `str`, `bool`, `int`, a
  `Literal[...]` of text (an enum), or `Optional[str] = None`. A `bool` always defaults to
  `False`.
- **The `Field` description is the planner's instruction** for filling the field. Write it
  as the rule: "exactly as the request writes them" keeps the planner from rewording a
  name.
- **`def search(self, input)`**: a phase method that names `input` gets the checked
  arguments as an `Input` instance; `input.query` is the value.
- **`f"find_book query={input.query}"`** writes the value into the step. A step written
  `command key=value` with every required parameter given is one **code runs** itself,
  before the agent takes a step.
- **`"say which books were found"`** is an answer step that needs the agent: only a step
  starting `say that` is copied into the answer word for word (page 3).

One method that reads only `input` makes a **flat** skill: the whole call is one plan, run
by one executor run.

## What the planner sees

The template is the steps written with every input a placeholder, `<query>`:

<!-- template: find_books -->
```text
find_book query=<query>
say which books were found
```

The preview lists the inputs, with the sample value each is rendered with:

<!-- preview: find_books -->
```markdown
Source: `_skills/find_books.py`. Kind: flat: one list of steps.

## Inputs

| input | type | default | sample | verify | subject | description |
|---|---|---|---|---|---|---|
| `query` | str | required | `<query>` |  |  | the title or author words exactly as the request writes them |

## Example requests

- find books by Jane Austen
- do you have Dune

## Steps

For the sample inputs; no input changes them.

1. `find_book query=<query>`
2. `say which books were found`

Pages: step 1 reads page 1 only.
```

"Pages: step 1 reads page 1 only": `find_book` takes a page (`start`), and a step without a
note asking for the full list reads the first page ("Previews" in
[skill_file_layout.md](../../skill_file_layout.md) has the paging rules).

## At run time

For "which books by Austen do you have", the planner fills `query=Austen`. A dry run:

<!-- dry-run: find_books {"query": "Austen"} -->
```text
Runs flat: one plan, one executor run.
Phase find_books
  1. find_book query=Austen
  2. say which books were found
  code runs: find_book query=Austen
  the agent writes the answer from these runs
```

Code runs step 1 (`find_book query=Austen`, every value written). Step 2 is an answer step
that is not `say that ...`, so the agent writes the answer from the run's output.

## What you learned

`Input` fields are the skill's arguments; their descriptions tell the planner how to fill
them; a method reading `input` writes them into its steps.

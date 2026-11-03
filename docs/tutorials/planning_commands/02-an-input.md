# 2. An input

**Goal:** "note on Raj Patel's record that he prefers email reminders". The command takes two
values from the request, the member's name and the note, and writes them into its steps.

## The planning command file

<!-- file: fastworkflow/examples/library/_commands/Library/note_member.py -->
```python
"""note_member: add the request's note to the record of the member it names.

One phase, so the call is one plan, run by one executor run. It finds the member, opens
the one named, and adds the note there.
"""
from pydantic import BaseModel, Field

from fastworkflow.planning import pick, run


class Signature:
    """Add a note to a member's record."""

    plain_utterances = [
        "note on Raj Patel's record that he prefers email reminders",
        "put a note on Ann Lee: card renewed",
        "put on Mia Chen's record that she wants large print",
        "add to Raj Patel's record: moved address",
        "leave a note for Ann Lee saying her card is due for renewal",
        "write a note about Mia Chen: prefers pickup at Riverside",
        "record for Raj Patel that he asked for a call back",
        "attach to Ann Lee's record a note about the damaged card",
    ]

    class Input(BaseModel):
        member: str = Field(description="the member's name",
                            json_schema_extra={"in_request": True})
        text: str = Field(description="the note to add")


class PlanGenerator:
    def note(self, input):
        return [run("find_member", query=input.member),
                pick("open_member", named=input.member),
                run("add_note", text=input.text),
                "say whose record the note was added to"]
```

## Line by line

- **`class Input(BaseModel)`** declares what the command takes, as an execution command's
  `Input` does. For a request the command covers, one typed call fills it from the request
  (page 14); a step that calls the command fills it by the parameter extractor, as it does
  any command's arguments (page 9). The values are then checked before any step is written,
  by the options their fields carry.
- **`json_schema_extra={"in_request": True}`** is such an option: the member's name must
  appear in the request word for word (case and dashes aside), so nothing rewords a name or
  invents one. `text` carries no option, so the extractor may tidy the note's words. A field
  with no option is checked by its type alone. Page 9 lists the options.
- **Both fields are required** (no default). A field is a `str`, a `bool`, an `int` or a
  `Literal[...]` of text values.
- **Each description says what this command takes, for a person.** `member` is "the
  member's name". `find_member`, which the first step passes it to, describes its own
  `query` as "words of the member's name", which is right for a search and wrong here: this
  command wants the member's whole name. A field of a planning command describes its own
  value, even when a callee takes the same value; the preview lists each input with its
  own description (below).
- **Leave instructions to the model out of descriptions.** The framework adds the sentence
  of each option a planning command's field carries when it asks the parameter extractor to
  fill the field. For `in_request` on a `str` that is "Copy it exactly as the request
  writes it.", so the extractor reads "the member's name. Copy it exactly as the request
  writes it."
- **`def note(self, input)`**: a phase method that names `input` gets the checked arguments
  as an `Input` instance.
- **`run("add_note", text=input.text)`** writes the step `add_note text=...`, with the value
  written so that the step parser reads it whole: a note that ends in brackets, or holds an
  `=`, is quoted. Code runs a step `run` wrote when every required value is given. Do not
  build a step with an f-string: `f"add_note text={input.text}"` would lose "(not phone)"
  from the end of "prefers email (not phone)", with no error.
- **`pick("open_member", named=input.member)`** writes the step that opens the row of the
  `find_member` listing named that way: `open_member with the uid of the Raj Patel match`.
  Code finds that row in the listing the call read, and opens it. Page 4 is about `pick`,
  and page 5 about the record it moves the workflow to.
- **`"say whose record the note was added to"`** is a plain string, so it is the agent's to
  do: here, writing the answer.

This job is one plan with no facts to hand on, so it is one method: one phase, run as one
plan by one executor run, with the agent's whole step budget.

## What the agent plans from

The template is the steps written with every input a placeholder:

```text
find_member query=<member>
open_member with the uid of the <member> match
add_note text=<text>
say whose record the note was added to
```

The preview lists the inputs: each one's type, default, sample value, its options (here
`in_request`, as a sentence under the table) and its description:

```markdown
Source: `_commands/Library/note_member.py`.

Calls: `Library/find_member`, `Library/open_member`, `Record/add_note`.
Depth: 1 (lower bound; open steps name no command)

## Inputs

| input | type | default | sample | verify | subject | description |
|---|---|---|---|---|---|---|
| `member` | str | required | `<member>` |  |  | the member's name |
| `text` | str | required | `<text>` |  |  | the note to add |

- `member` must be written in the request (`in_request`).
```

## At run time

For "add a note to Raj Patel: card renewed", the typed call fills `member=Raj Patel` and
`text=card renewed` from the request:

```text
Phase note
  1. find_member query=Raj Patel
  2. open_member with the uid of the Raj Patel match
  3. add_note text=card renewed
  4. say whose record the note was added to
  code runs: find_member query=Raj Patel
  code runs: open_member member_uid=m2
  now in: Member: Raj Patel (m2)
  code runs: add_note text=card renewed
  the agent writes the answer from these runs
```

Code runs the first three steps. Step 2 is written in pick words, and code finds the one row
named "Raj Patel" in the listing step 1 read, and opens it.
That moves the workflow onto Raj Patel's record, so step 3's `add_note`, a command every
record has, runs there. Step 4 is a plain string, so the agent writes the answer from the
runs.

The call ends where its last step left the workflow: on Raj Patel. Page 5 is about that.

## What you learned

`Input` fields are a command's arguments, filled from the request, and checked by the options
they carry, such as `in_request` on a planning command's field; their descriptions say, for a
person, what each value is. `run` writes a step with its values
written whole; a step that opens a listing's row by name is written with `pick`.

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

    plain_utterances = ["note on Raj Patel's record that he prefers email reminders",
                        "add a note to Ann Lee: card renewed"]

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
  `Input` does, and its values are filled the same way: the parameter extractor fills the
  `Input` from the step that calls the command, which for a request the command covers is
  the request itself (page 9). The values are then checked before any step is written, by
  the options their fields carry.
- **`json_schema_extra={"in_request": True}`** is such an option: the member's name must
  appear in the request word for word (case and dashes aside), so nothing rewords a name or
  invents one. `text` carries no option, so the extractor may tidy the note's words. A field
  with no option is checked by its type alone. The options mean the same on an execution
  command's field: `find_member`'s `query` could carry `in_request` too. Page 9 lists them.
- **Both fields are required** (no default). An `Input` field may have any type the
  parameter extractor supports.
- **Each description says what this command takes, for a person.** `member` is "the
  member's name". `find_member`, which the first step passes it to, describes its own
  `query` as "words of the member's name", which is right for a search and wrong here: this
  command wants the member's whole name. A field of a planning command describes its own
  value, even when a callee takes the same value; the preview shows the callee's description
  beside it (below), so you can compare the two while you write.
- **Leave instructions to the model out of descriptions.** The framework adds the sentence
  of each option a field carries when it asks the parameter extractor to fill the field, for
  either kind of command. For `in_request` on a `str` that is "Copy it exactly as the request
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

The preview lists the inputs, and for each one the field of the command the template passes
it to, with that field's description:

```markdown
Source: `_commands/Library/note_member.py`.

Calls: `Library/find_member`, `Library/open_member`, `Record/add_note`.
Depth: 1 (lower bound; open steps name no command)

## Inputs

| input | type | default | sample | options | description | passed to |
|---|---|---|---|---|---|---|
| `member` | str | required | `<member>` | `in_request` | the member's name | `Library/find_member.query`: words of the member's name |
| `text` | str | required | `<text>` |  | the note to add | `Record/add_note.text`: the note to add |
```

## At run time

For "add a note to Raj Patel: card renewed", the parameter extractor fills `member=Raj Patel`
and `text=card renewed` from the request:

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

`Input` fields are a command's arguments, filled from the request the way any command's are,
and checked by the options they carry, such as `in_request`, on either kind of command;
their descriptions say, for a person, what each value is. `run` writes a step with its values
written whole; a step that opens a listing's row by name is written with `pick`.

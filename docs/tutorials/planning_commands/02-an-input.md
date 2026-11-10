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
import re

from pydantic import BaseModel, Field, model_validator

from fastworkflow.planning import run


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
                            json_schema_extra={"copied_from_command_text": True})
        text: str = Field(description="what the note says, without the request around it "
                                      "(to note that someone called: 'someone called')",
                          json_schema_extra={"copied_from_command_text": True})

        @model_validator(mode="after")
        def note_without_the_request(self):
            # A note on Mia Chen's record does not name Mia Chen: a text that does holds the
            # request ("Note on Mia Chen's record that ..."), not the note.
            if re.search(rf"\b{re.escape(self.member)}\b", self.text, re.IGNORECASE):
                raise ValueError(f"text is {self.text!r}, which holds the member's name: that is the "
                                 "request around the note; give only the words the note says")
            return self


class PlanGenerator:
    def note(self, input):
        return [run("find_member", query=input.member),
                run("open_member", named=input.member),
                run("add_note", text=input.text),
                "say whose record the note was added to"]
```

## Line by line

- **`class Input(BaseModel)`** declares what the command takes, as an execution command's
  `Input` does. For a request the command covers, one typed call fills it from the request
  (page 14); a step that calls the command fills it by the parameter extractor, as it does
  any command's arguments (page 9). The values are then checked before any step is written,
  by the options their fields carry.
- **`json_schema_extra={"copied_from_command_text": True}`** is such an option: the value must
  appear in the command text word for word (case and dashes aside), so nothing rewords a name
  or invents one. `text` carries it too: a note is the user's words, and a note the extractor
  tidied, shortened or made up ("Requested extension for upcoming project deadline" for "she
  asked for an extension") is refused, and the extractor is asked once more. A field with no
  option is checked by its type alone; leave the option off only a value the request may
  legitimately write another way (a date written "next Monday"). Page 9 lists the options.
- **The `model_validator`** refuses what `copied_from_command_text` cannot see: a note that is
  written in the command text but holds more than the note. When the extractor takes the whole
  instruction ("Note on Mia Chen's record that she asked for an extension"), the text names the
  member, which a note on her own record does not. The validator's message says what to fix, and
  the call is filled once more with it (page 9). It is a plain `ValueError`, so it refuses with
  no field named (page 9 has the difference from `fastworkflow.Refused`).
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
  fill the field. For `copied_from_command_text` on a `str` that is "Copy it exactly as the
  request writes it.", so the extractor reads "the member's name. Copy it exactly as the request
  writes it." For `record_named_in_command_text` the sentence also says the value must name a
  record the command text names. The description still only says what the field holds. The
  check matches the extractor's value against names the command text already writes; it does
  not read the description. A model chooses among those names again only when the field also
  has `classifier_confirms` and `JEV_API_KEY` is set (page 9).
- **`def note(self, input)`**: a phase method that names `input` gets the checked arguments
  as an `Input` instance.
- **`run("add_note", text=input.text)`** writes the step `add_note text=...`, with the value
  written so that the step parser reads it whole: a note that ends in brackets, or holds an
  `=`, is quoted. Code runs a step `run` wrote when every required value is given. Do not
  build a step with an f-string: `f"add_note text={input.text}"` would lose "(not phone)"
  from the end of "prefers email (not phone)", with no error.
- **`run("open_member", named=input.member)`** writes the step that opens the row of the
  `find_member` listing named that way: `open_member with the uid of the Raj Patel match`.
  Code finds that row in the listing the call read, and opens it. Page 4 is about `named=`,
  and page 5 about the record it moves the workflow to.
- **`"say whose record the note was added to"`** is a plain string, so it is the agent's to
  do: here, writing the answer.

This job is one plan with no facts to hand on, so it is one method: one phase, run as one
plan by one executor run, with the agent's whole step budget.

## A known false positive

The validator refuses a note that repeats the member's own name, because it cannot tell the
note from the request around it. "Mia Chen to call back", written as the note for Mia Chen, is
refused, with this message:

```text
validate_extracted_parameters: text is 'Mia Chen to call back', which holds the member's name: that is the request around the note; give only the words the note says
```

The name is the member of the record the note goes on, so the words of the note are "to call
back", and the member's name in them reads as the request. The call is filled once more. Write the note without the name ("to call back"), which is what the record holds anyway,
since the record is already Mia Chen's. The rule is not narrowed on this page: the cost is
shown here, so the next author can judge it.

## What the agent plans from

The template is the steps written with every input a placeholder:

```text
find_member query=<member>
open_member with the uid of the <member> match
add_note text=<text>
say whose record the note was added to
```

The preview lists the inputs: each one's type, default, sample value, its options (here
`copied_from_command_text`, as a sentence under the table) and its description:

```markdown
Source: `_commands/Library/note_member.py`.

Calls: `Library/find_member`, `Library/open_member`, `Record/add_note`.
Depth: 1 (lower bound; open steps name no command)

## Inputs

| input | type | default | sample | classifier_confirms / classifier_decides | copied_from_command_text / record_named_in_command_text | description |
|---|---|---|---|---|---|---|
| `member` | str | required | `<member>` |  | yes | the member's name |
| `text` | str | required | `<text>` |  | yes | what the note says, without the request around it (to note that someone called: 'someone called') |

- `member` must be written in the command text (`copied_from_command_text`).
- `text` must be written in the command text (`copied_from_command_text`).

## Example requests

- note on Raj Patel's record that he prefers email reminders
- put a note on Ann Lee: card renewed
- put on Mia Chen's record that she wants large print
- add to Raj Patel's record: moved address
- leave a note for Ann Lee saying her card is due for renewal
- write a note about Mia Chen: prefers pickup at Riverside
- record for Raj Patel that he asked for a call back
- attach to Ann Lee's record a note about the damaged card

## Steps

For the sample inputs; no input changes them.

1. `find_member query=<member>`
2. `open_member with the uid of the <member> match`
3. `add_note text=<text>`
4. `say whose record the note was added to`
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

Code runs the first three steps. Step 2 is written in named-row words, and code finds the one row
named "Raj Patel" in the listing step 1 read, and opens it.
That moves the workflow onto Raj Patel's record, so step 3's `add_note`, a command every
record has, runs there. Step 4 is a plain string, so the agent writes the answer from the
runs.

The call ends where its last step left the workflow: on Raj Patel. Page 5 is about that.

## What you learned

`Input` fields are a command's arguments, filled from the request, and checked by the options
they carry, such as `copied_from_command_text` on a planning command's field; their descriptions
say, for a person, what each value is. `run` writes a step with its values written whole; a step
that opens a listing's row by name is written with `named=`.

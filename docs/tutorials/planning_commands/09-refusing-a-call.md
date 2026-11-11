# 9. Refusing a call, and a missing argument

**Goal:** "what has Raj Patel borrowed?" is `member_report`'s job. "What has everyone
borrowed?" is not, and the caller should be told so before any step runs. And "tell me about
the member" leaves out the one thing the command needs.

## The checks a field can ask for

`member_report`'s `member` carries `json_schema_extra={"copied_from_command_text": True}` (page 6).
Each check is one key of a field's `json_schema_extra`, and means the same on an execution
command's field as on a planning command's. The key names the behaviour; its value is `True`, or
a dict of that key's own options:

| key | on a field of type | what it checks |
|---|---|---|
| `copied_from_command_text: True` | `str`, `int`, `list[str]` | a `str` is written in the command text (case and dashes aside); an `int` is a number the command text writes; each item of a `list[str]` is written |
| `copied_from_command_text: {"ignore": [...]}` | `str` | as above, after the named words are removed from either end of the value (the cleanup the retired `strip` option did) |
| `record_named_in_command_text: True` | `str` | as `copied_from_command_text`, and the value names a record the command text already names |
| `record_named_in_command_text: {"ignore": [...]}` | `str` | as above, with the named words removed from either end first |
| `classifier_confirms: True` | `bool`, `Literal`, or a `str` with `record_named_in_command_text` alongside | a classifier decides the value again from the command text; it needs `JEV_API_KEY`, and without it checks nothing |
| `classifier_confirms: {"meanings": {value: meaning, ...}}` | `Literal` | the same check, and gives each value its meaning (every key one of the enum's values) |
| `classifier_decides: True` | `bool` | the classifier alone decides the value from the command text; the typed call and the parameter extractor never fill it, and it keeps its default when the classifier is not sure or there is no `JEV_API_KEY` |

`record_named_in_command_text` checks a value the parameter extractor already proposed. The
extractor reads the field's description, which says what the field holds (a person's name, a
product, a code) and does not say how a record is found. The check matches that value against
names the command text already writes; it does not read the description. The description and the
sentence the framework appends to it are what the extractor reads. A model chooses among those
names again only when the same field also has `classifier_confirms`, and only when `JEV_API_KEY`
is set. Without the key the string check still runs.

The keys are not the whole list: `command_text_after_command_name`, `from_command_output`
and `converted_from_command_text` are the other field keys, and the
[planning command file layout](../../planning_command_file_layout.md) has them under "Field
options". A field with no check is cleaned in code: call `strip_words(text, words)` from
`fastworkflow.planning_commands` in a `field_validator` on that field. A validator's returned value
is written back, so a validator that cleans a value changes the argument. A validator cannot
replace "required" or `ignore`: a missing required field fails before any validator runs, and
`ignore` strips the value before the verbatim match.

A field with no key is checked by its type alone, and a key on a type it does not apply to fails
the load. For each key a field carries, the framework adds a sentence to the field's description
when it asks the parameter extractor to fill it (page 2).

## The validator

`member_report` (page 6) refuses in a model validator on its `Input`:

```python
class Input(BaseModel):
    member: str = Field(description="the member's name",
                        json_schema_extra={"copied_from_command_text": True})
    ...

    @model_validator(mode="after")
    def refuse_everyone(self, info):
        if self.member.casefold() in EVERYONE:
            raise fastworkflow.Refused(
                f"member is {self.member!r}, which names no one member; this "
                "command reports on one member: give that member's name", field="member")
        return self

...

#: Words that name every member, not one.
EVERYONE = {"everyone", "everybody", "all", "all members", "every member"}
```

## Line by line

- **`@model_validator(mode="after")`** is the one place a cross-field rule is written, for both
  kinds of command. It runs on the `Input` after the arguments were filled and the field options
  were checked, and before the `PlanGenerator` is called. A validator on several fields reads
  them all as `self`, so there is no hook to set up: the rule is a method of the model.
- **`info.context`** is the validation context: `command_text` (the text the arguments were
  filled from), `workflow` (the app's workflow, or `None` while the template is rendered) and
  `command` (the qualified name, such as `Library/member_report`). Both kinds of command get the
  same context, so a validator that reads `info.context["command_text"]` needs no guard.
  `refuse_everyone` does not read it.
- **`raise fastworkflow.Refused(message, field="member")`** refuses the call, and it is the
  refusal that names an input. Write the message as what to fix. `field` names the input the
  user can fix: on a typed call in an attended turn, that input is asked for. Leave `field` out
  when no answer could fix the call, because the command does not fit the request ("the owner is
  read only for an application"): the request is then planned without the command, and the user
  is not asked. The `fields` of the refusal reach the caller intact, as the `member` above does.
- **A plain `ValueError`** in a validator is a refusal with no field named, worded as the hook's:
  `validate_extracted_parameters: <text>`. The extractor is asked once more, and if the call is still
  refused the planner's typed fill takes over. The note validator on page 2 is one: its refusal names
  no input.
- **A correction** is an assignment and a return. A validator that sets `self.<field>` and returns
  `self` writes the value back, and the notes of the call record what changed, attributed to the
  validator. Its message says what to fix in the same way.
- **Several validators** run in the order they are declared, one after another, each on the
  values the one before left.
- **A placeholder combination** (the template and the preview build the call with `command_text=""`
  and `workflow=None`) runs the validators too. A combination a validator refuses or changes is left
  out of the template and the preview, as the old hook's were.
- **`EVERYONE`** is a module-level table, written after `PlanGenerator`. The validator reads it
  when it is called, so the order does not matter to Python.

## How the arguments arrive

A planning command's arguments arrive one of two ways, and the same checks run on both:

1. **A typed call.** When `JEV_API_KEY` is set and the command fits the request (or the part
   of it the command covers), the planner fills the command's `Input` as a typed call: "what
   has Ann Lee borrowed" becomes `member=Ann Lee` (page 14).
2. **A step.** A planning command that a plan or another command names is called by that step
   (page 12). A step written `member_report member=Ann Lee loans=true` is bound in code; a
   step in words goes to the parameter extractor, which reads the field descriptions and the
   sentences of their options (page 2), as it does for an execution command.

Then **the checks run**, in this order, for either kind of command: the field options' checks
(here, that `member` is written in the command text), then the `Input`'s model validators, in
the order declared. Last, **the `PlanGenerator` is called** with the checked `Input`.

The request the checks read is the user's request for a typed call. For a step, it is the
step's text, with one exception: when the planner wrote the step, its arguments are checked
against the user's request and what the turn's earlier steps found, not the plan's own
words, so a value the planner made up is refused (page 14).

**Who gets a refusal.** For a typed call, the planner, which fills the call once more with
the message in front of it; refused again, the call is not made. An attended turn asks the
user when the refusal names an input (a missing argument, a value the request does not
write, or `Refused(..., field=...)`; see below). Every other refusal, and every refusal in an
unattended turn, plans the request from the template instead. For a step, the
agent that ran it, as a failed step whose text is the message. It can rewrite the step, run
another command first, or tell the user.

## A missing argument

"Tell me about the member" names no member, so `member` cannot be filled.

**On a typed call**, an attended turn asks the user, in a question code writes from the
command and the field:

```text
I can't run member report yet: your request does not say which member to use. Which member should I use?
```

The reply is the next message, and it completes the call. An unattended turn never asks: the
planner plans the request from the command's template instead.

**On a step**, the step's observation is the message any command gets for a missing
argument. It names the field, and the commands a producer `from_command_output` points it to,
when the field has one:

```text
Missing parameter values: member

Provide corrected parameter values in the exact order specified below, separated by commas:
member
```

The agent then finds the value, with the producer the message named, or asks the user in its
own words with `ask_user`, and runs the step again with the answer. An unattended turn never
asks: the agent goes on without the value, or says what it could not do.

Nothing about a missing argument is declared. The question is written from the command's
name and the field's name. The field's description is what the parameter extractor and the
agent read, which is one more reason to write it for a person.

A call that names no one member is refused, with the reason the planner reads:

<!-- refusal: member_loans {"member": "everyone"} -->
```text
validate_extracted_parameters: member is 'everyone', which names no one member; this skill reads one member's loans: give that member's name
```

## At run time

A refused call never reaches the `PlanGenerator`:

```text
validate_extracted_parameters: member is 'everyone', which names no one member; this command reports on one member: give that member's name
```

`member=Ann Lee` passes, and the call runs as on page 6.

Run offline for Raj Patel, the call passes its check and finds the member:

<!-- dry-run: member_loans {"member": "Raj Patel"} -->
```text
Phase find
  1. find_member query=Raj Patel
  code runs: find_member query=Raj Patel
  facts: Output(member_uids=['m2'], names=['Raj Patel'], total=1)
Phase member
  1. list_member_loans with the uid of the Raj Patel match
  code runs: list_member_loans member_uid=m2
  facts: Output(titles=['Dune', 'Emma'], due_dates=['2026-10-20', '2026-11-05'], total=2)
answer written by __result__:
  Raj Patel has 2 books (Dune, Emma) on loan.
```

## What you learned

A planning command's arguments are checked the way any command's are, whether a typed call or
a step filled them: by the options their fields carry (`copied_from_command_text`,
`classifier_confirms`), then the `Input`'s model validators;
`raise fastworkflow.Refused(...)` sends the caller back with what to fix. A missing argument
on a typed call is asked for in a question code writes, in an attended turn; on a step it
comes back as the same message an execution command's does, and the agent asks for it.

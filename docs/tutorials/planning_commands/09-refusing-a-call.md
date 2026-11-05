# 9. Refusing a call, and a missing argument

**Goal:** "what has Raj Patel borrowed?" is `member_report`'s job. "What has everyone
borrowed?" is not, and the caller should be told so before any step runs. And "tell me about
the member" leaves out the one thing the command needs.

## The checks a field can ask for

`member_report`'s `member` carries `json_schema_extra={"in_request": True}` (page 6). Each
check is such an option of an `Input` field, and means the same on an execution command's
field as on a planning command's:

| option | on a field of type | what it checks |
|---|---|---|
| `in_request: True` | `str`, `int`, `list[str]` | a `str` is written in the request (case and dashes aside); an `int` is a number the request writes; each item of a `list[str]` is written |
| `in_request: "subject"` | `str` | as `in_request`, and the value names a record the request names |
| `in_request: {"ignore": [...]}` | `str` | as `in_request`, after the named words are removed from either end of the value (the cleanup the retired `strip` option did) |
| `verify: True` | `bool`, `Literal`, or a `str` with `in_request: "subject"` | a classifier decides the value again from the request; it needs `JEV_API_KEY`, and without it checks nothing |
| `verify: {value: meaning, ...}` | `Literal` | the same check, and gives each value its meaning (every key one of the enum's values) |

Two options is the whole list: `in_request` (with `"subject"` and the `ignore` words inside it) and
`verify` (with `"alone"` for a bool only the classifier decides, such as a part of the job the
request may leave out; see "Field options" in
[planning_command_file_layout.md](../../planning_command_file_layout.md)). A field with no check is cleaned in code: call `strip_words(text, words)` from
`fastworkflow.planning_commands` in its validator or in the `validate_extracted_parameters` hook. An
`Input` validator's returned value is written back, so a validator that cleans a value changes the
argument. The retired `subject`, `means` and `strip` options fail the load and name the new spelling.

A field with no option is checked by its type alone, and an option on a type it does not
apply to fails the load. For each option it finds, the framework adds a sentence to the
field's description when it asks the parameter extractor to fill it (page 2).

## The hook

`member_report` (page 6) refuses in its `Signature`:

```python
class Signature:
    ...
    @staticmethod
    def validate_extracted_parameters(workflow, command, input, command_text=None):
        if input.member.casefold() in EVERYONE:
            raise fastworkflow.Refused(
                f"member is {input.member!r}, which names no one member; this "
                "command reports on one member: give that member's name", field="member")
        return None

...

#: Words that name every member, not one.
EVERYONE = {"everyone", "everybody", "all", "all members", "every member"}
```

## Line by line

- **`validate_extracted_parameters`** is the hook an execution command's `Signature` has,
  with the same signature. It gets the arguments as an `Input` (`input`) after they
  were filled and checked, before the `PlanGenerator` is called. `command_text` is the
  request the arguments were filled from.
- **`raise fastworkflow.Refused(message, field="member")`** refuses the call. Write the message
  as what to fix. `field` names the input the user can fix: on a typed call in an attended
  turn, that input is asked for. Leave `field` out when no answer could fix the call, because
  the command does not fit the request ("the owner is read only for an application"): the
  request is then planned without the command, and the user is not asked. Return `None` to
  accept; a tuple such as `(False, message)` is a call error.
- **Any other exception** the hook raises fails the call; `Refused` is the one that means
  "these arguments are wrong".
- **The hook may correct** as well as refuse: set a field of `input` (a name in the
  catalogue's spelling, say) and return `None`. It runs once per call, after the
  options' checks, so a value it corrects is not checked again against the request.
- **A rule that ties fields together**, or a field to the request, can instead be a
  pydantic validator on `Input` raising `ValueError` (see "Rules on the arguments" in
  [planning_command_file_layout.md](../../planning_command_file_layout.md)). It is checked after the hook, with the
  request in the validation context. That context carries the request on a planning command;
  an execution command's validators get none.
- **`EVERYONE`** is a module-level table, written after `PlanGenerator`. The hook reads it
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
(here, that `member` is written in the request), then this hook, once, then the `Input`'s
validators. Last, **the `PlanGenerator` is called** with the checked `Input`.

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
argument. It names the field, and the commands a producer `from_` points it to, when the
field has one:

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
a step filled them: by the options their fields carry (`in_request`, `verify`), then
`validate_extracted_parameters`, once, then the `Input`'s validators;
`raise fastworkflow.Refused(...)` sends the caller back with what to fix. A missing argument
on a typed call is asked for in a question code writes, in an attended turn; on a step it
comes back as the same message an execution command's does, and the agent asks for it.

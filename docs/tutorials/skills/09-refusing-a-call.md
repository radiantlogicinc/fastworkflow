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
| `in_request` | `str`, `int` | a `str` is written in the request (case and dashes aside); an `int` is a number the request writes |
| `subject` | `str` | as `in_request`, and the value names a record the request names |
| `strip` | `str` | removes these words from either end before the check (`["group", "unit"]`) |
| `verify` | `bool`, `Literal`, or a `subject` str | a classifier decides the value again from the request; it needs `JEV_API_KEY`, and without it checks nothing |
| `means` | `Literal` with `verify` | gives each value its meaning, for `verify` |

A field with no option is checked by its type alone, and an option on a type it does not
apply to fails the load. For each option it finds, the framework adds a sentence to the
field's description when it asks the parameter extractor to fill it (page 2).

## The hook

`member_report` (page 6) refuses in its `Signature`:

```python
class Signature:
    ...
    @staticmethod
    def validate_extracted_parameters(workflow, command, cmd_parameters, command_text=None):
        if cmd_parameters.member.casefold() in EVERYONE:
            raise fastworkflow.Refused(
                f"member is {cmd_parameters.member!r}, which names no one member; this "
                "command reports on one member: give that member's name")
        return True, ""

...

#: Words that name every member, not one.
EVERYONE = {"everyone", "everybody", "all", "all members", "every member"}
```

## Line by line

- **`validate_extracted_parameters`** is the hook an execution command's `Signature` has,
  with the same signature. It gets the arguments as an `Input` (`cmd_parameters`) after they
  were filled and checked, before the `PlanGenerator` is called. `command_text` is the
  request the arguments were filled from.
- **`raise fastworkflow.Refused(message)`** refuses the call. Write the message as what to
  fix. Returning `(False, message)` refuses the same way; `(True, "")` accepts.
- **Any other exception** the hook raises fails the call; `Refused` is the one that means
  "these arguments are wrong".
- **The hook may correct** as well as refuse: set a field of `cmd_parameters` (a name in the
  catalogue's spelling, say) and return `(True, "")`. It runs once per call, after the
  options' checks, so a value it corrects is not checked again against the request.
- **A rule that ties fields together**, or a field to the request, can instead be a
  pydantic validator on `Input` raising `ValueError` (see "Rules on the arguments" in
  [skill_file_layout.md](../../skill_file_layout.md)). It is checked after the hook, with the
  request in the validation context, on either kind of command.
- **`EVERYONE`** is a module-level table, written after `PlanGenerator`. The hook reads it
  when it is called, so the order does not matter to Python.

## How the arguments arrive

A planning command's arguments arrive one way, wherever the call comes from, and it is the
way an execution command's do:

1. **The step that calls the command is extracted.** For a request the command covers, the
   agent writes that step in its plan, from the request or from the part of it the command
   covers: "what has Ann Lee borrowed" becomes the step that calls `member_report` (page 14). A
   planning command that another one names is called by that line of the plan (page 12). A
   step written `member_report member=Ann Lee loans=true` is bound in code; a step in words
   goes to the parameter extractor, which reads the field descriptions and the sentences of
   their options (page 2).
2. **The checks run**, in this order, for either kind of command: the field options' checks
   (here, that `member` is written in the request), then this hook, once, then the `Input`'s
   validators.
3. **The `PlanGenerator` is called** with the checked `Input`.

The request the checks read is the step's text, with one exception: when the agent wrote the
step, as it does in its plan for a request the command covers, its arguments are checked
against the user's request and what the turn's earlier steps found, not the agent's own
words, so a value the agent made up is refused (page 14).

**Who gets a refusal.** The agent that ran the step, as a failed step whose text is the
message. It can rewrite the step, run another command first, or tell the user.

## A missing argument

"Tell me about the member" names no member. The extractor cannot fill `member`, and the
step's observation is the message any command gets for a missing argument. It names the
field, and the commands a producer `from_` points it to, when the field has one:

```text
Missing parameter values: member

Provide corrected parameter values in the exact order specified below, separated by commas:
member
```

The agent then finds the value, with the producer the message named, or asks the user in its
own words with `ask_user`, and runs the step again with the answer. An unattended turn never
asks: the agent goes on without the value, or says what it could not do.

So a missing argument works the same for both kinds of command, and nothing about it is
declared. The field's description is what the agent reads when it decides what to ask,
which is one more reason to write it for a person.

## At run time

A refused call never reaches the `PlanGenerator`:

```text
validate_extracted_parameters: member is 'everyone', which names no one member; this command reports on one member: give that member's name
```

`member=Ann Lee` passes, and the call runs as on page 6.

## What you learned

A planning command's arguments are filled and checked the way any command's are: by the
options their fields carry (`in_request`, `subject`, `strip`, `verify`, `means`), then
`validate_extracted_parameters`, once, then the `Input`'s validators;
`raise fastworkflow.Refused(...)` sends the caller back with what to fix. A missing argument
comes back as the same message an execution command's does, and the agent asks for it.

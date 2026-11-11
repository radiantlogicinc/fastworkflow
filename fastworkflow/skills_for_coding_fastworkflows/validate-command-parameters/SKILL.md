---
name: validate-command-parameters
description: >-
  Write the model validators of a fastWorkflow command's `Input` so whole-call rules, cross-field
  constraints, context preconditions and value lookups are checked and corrected before the command
  body runs, and so a bad call stays a conversation instead of becoming an exception. Covers the exact
  contract (`@model_validator(mode="after")` that corrects by assigning `self` or refuses with
  `fastworkflow.Refused(message, field=...)`), the validation context (`info.context["command_text"]`,
  `["workflow"]`, `["command"]`), where it sits in the validation order, writing error messages an
  agent can act on, and what belongs here versus in the field declaration. Use when a command has rules
  spanning more than one field, needs a precondition on the current record, must normalise input before
  executing, or when authoring or reviewing any command `Signature`.
---

# Validating extracted parameters

Pydantic answers one field at a time: right type, matches the pattern, was supplied. It cannot answer
the questions that only exist once the whole call is on the table: is this end date before its own
start date, is anything actually selected for this bulk command to operate on, does this decision
value exist in the backend's vocabulary. Those rules are model validators on the command's `Input`.
That is the one validation mechanism a command has. The `validate_extracted_parameters` hook is
removed: a `Signature` that still defines it fails the load, and the load error shows the validator form.

## The contract

A `@model_validator(mode="after")` on the command's `Input`, written as an instance method:

```python
from pydantic import BaseModel, model_validator

import fastworkflow


class Signature:
    class Input(BaseModel):
        begin_date: str
        end_date: str

        @model_validator(mode="after")
        def end_is_not_before_begin(self, info):
            ...
            return self
```

- Correct a value by assigning to `self.<field>` and returning `self`. The corrected value is what the
  command body reads.
- Refuse by raising `fastworkflow.Refused(message, field=...)`. `field=` resets the named field(s) to
  missing and asks the user for them again with the message. A plain `ValueError` is a refusal with no
  field. Do not raise any other exception: it is caught and becomes a call error that names the exception.

```python
        @model_validator(mode="after")
        def decision_is_accepted(self, info):
            if self.decision not in ("approve", "revoke", "defer"):
                raise fastworkflow.Refused(
                    "decision 'approve-all' is not one of the accepted values. Use one of: approve, revoke, defer.",
                    field="decision")
            return self
```

The validator reads the validation context from `info.context`. It has the same keys for both kinds of
command (an execution command and a planning command):

- `info.context["command_text"]`: the user's command text (the request), `""` when a template renders.
- `info.context["workflow"]`: the workflow, or `None` while a planning command's template renders.
- `info.context["command"]`: the qualified name of the command.

The framework always validates an `Input` with this context, so a validator does not need a `None` guard
on `info.context`. Code that builds an `Input` directly (in a test, or other author code) has no context.
Use `Signature.Input.model_construct(...)`, or `model_validate(values, context={...})`, or the framework
reports a clear error. Several model validators run in declaration order.

## Where it runs in the order

The framework validates a call in this sequence, and **stops at the first failure**:

```
1. extraction                          the value is bound (from the text, by the classifier, or by the typed call)
2. option checks                       copied_from_command_text, record_named_in_command_text, classifier_confirms, ...
3. the Input's model validators        after pydantic's type, pattern and required checks
4. role checks                         the same value for two required free-text inputs
```

**Migrating a `validate_extracted_parameters` hook (upgrade step).** The old order was option checks, then the
hook (which could correct fields), then the Input's own model validators, then roles. Pydantic runs model
validators in declaration order, so when you move a hook body into a `@model_validator(mode="after")`, declare
that validator **first**, before any model validators the `Input` already has. Otherwise an existing check sees
the uncorrected value. (In one workflow, a kind-fit check refused before the kind resolver ran.)

Pydantic checks types, patterns and required-ness on the fields first, so a model validator may assume every
field is **present and of the declared type**. Do not re-check types, required-ness or regex here: declare
those on the field and let pydantic do it. What is left is exactly what field declarations cannot express.
Value lookups against a live key set also live here, see the `resolve-parameter-values` skill.

A value that fails its type or a field validator is dropped at binding: a `name=value` binding whose value
does not check is no binding, and the LLM extracts the value instead. A model validator runs later, with
the context, on the value that was bound.

Before validation, in agent mode, the values are bound in code when the call's form allows it: XML tags,
`name=value` pairs, or `command_name value`. The last form binds to the field declared
`json_schema_extra={"command_text_after_command_name": True}` (several words allowed), or else to the
class's only `str` field (one token only). It declines on anything doubtful and the LLM extracts instead.
Mark at most one field, and not a free-text field of a command that changes state. A lone `command_name value`
that binds to no field is refused with a message naming the parameters, so the command never runs on its
defaults instead. Whatever bound the values, the sequence above still runs.

## It is a correction hook, not just a check

The `self` the validator receives is the `Input` the command body will read, so writing to it is the
intended way to normalise:

```python
        @model_validator(mode="after")
        def order_id_has_hash(self):
            if not self.order_id.startswith('#'):
                self.order_id = f'#{self.order_id}'
            return self
```

Prefer normalising to rejecting whenever the intent is unambiguous. Every rejection costs a round trip; a
value that can only have meant one thing should just be fixed.

## Refuse with Refused, never with a bare exception

Exceptions other than `fastworkflow.Refused` (and a plain `ValueError`) are caught, logged at critical, and
converted into a call error whose message names the command and the exception. That is strictly worse than
a message written for the caller: the caller gets a stack-trace-flavoured string instead of an instruction.

## Write the message for an agent

The message is the only thing the caller has to work from. Every message should name the field, say what is
wrong with the value that arrived, and say where an acceptable one comes from.

```python
raise fastworkflow.Refused(
    "decision 'approve-all' is not one of the accepted values. Use one of: approve, revoke, defer.",
    field="decision")
```

Vague messages ("invalid input") cause the agent to retry the same value.

## Accumulate errors, report once

Check everything, then raise once, so one round trip tells the caller all of it. `Refused` takes one message,
so collect the messages and raise the joined text. Pass `field=` only when the named fields are the ones to
re-ask:

```python
def refuse_if_any(errors, fields=()):
    """Raise one Refused for every error string, or return None when there are none."""
    message = "\n".join(error for error in errors if error)
    if message:
        raise fastworkflow.Refused(message, field=list(fields) or None)
```

Three helper shapes are worth distinguishing at the call site:

| Shape | Returns | Use for |
|---|---|---|
| normalising | `(value, error)`: store the value back | a code to upper-case, a uid to strip |
| relational | `error` | two or more fields that must agree |
| reading | `value` | turning an intent the caller can express into the value the body wants |

```python
        @model_validator(mode="after")
        def dates_and_decision(self, info):
            errors = []
            self.decision, error = one_of(self.decision, DECISIONS)
            errors.append(error)
            errors.append(in_order(self.begin_date, self.end_date))
            refuse_if_any(errors)
            return self
```

## Keep it local and cheap

The validator runs on **every** attempt at the command, including the ones an agent is guessing at, and
including retries after a correction. Do not call a backend from here. Check against values already in the
workflow context, or against a constant vocabulary. If a check genuinely needs a live key set, do the
`DatabaseValidator.fuzzy_match` lookup here: see the `resolve-parameter-values` skill.

## Context preconditions belong here

This is the only place that sees the workflow before the body runs, so it is where "is this command
meaningful right now" goes. The workflow's record is `workflow.record`, the record this command acts on:

```python
        @model_validator(mode="after")
        def something_is_selected(self, info):
            workflow = info.context["workflow"]
            selection = getattr(workflow.record, 'selection', None)
            if not selection:
                raise fastworkflow.Refused("Nothing is selected. Use select_items first, then retry.")
            return self
```

Note the message names the producing command, the same job the `from_command_output` edge does for a
parameter that never arrived (the runtime message reads `use the <producers> command(s) to get <field>
information`).

## Do not treat it as a security boundary

The validator runs on both the NLU path and the direct-action path, but it is a command's own declaration of
what it needs, not an enforcement layer: a caller reaching the body another way bypasses it. Invariants that
must hold for correctness or safety belong in the command body as well.

## Declare the format of id inputs

An input that takes an identifier (a uid, a record number, a key) must say what an identifier looks like, on
the field, with the format the application really mints, and name its producer:

```python
class Input(BaseModel):
    book_uid: str = Field(description="the uid of the book to open (b1, b2, ...)",
                          json_schema_extra={"from_command_output": "find_book.book_uids"},
                          pattern=r"^b[0-9]+$")
```

Without it, a `str` id field accepts anything, and a name or title is a valid value. The extractor, given "open
the book Dune Messiah" and no uid in the text, fills `book_uid="Dune Messiah"`; the sub-task review then shows
that title as the uid, and the run either fails on it or quietly opens something else. With the pattern, the
value breaks a declared rule at step 1 above: the draft leaves the field unbound (the review shows no value),
and the run finds the id the way it would have without a draft (the producer command, or the agent), so what
the review shows is what runs. The check is the field's own declaration, so it holds on every path: drafted,
extracted, typed by the user, or written by the agent.

Keep the pattern compatible with the field's producers: every id the `from_command_output` source command
returns must match it. Match the whole value (`^...$`), and match the ids the application creates, not a guess:
read where the application mints them. Do not move the format into a validator; a declared `pattern` is checked
before it.

## Authoring checklist

```
- [ ] Type, pattern and required-ness are declared on the field, not re-checked here
- [ ] Every id input declares the format of the application's ids (`pattern`), so a name or title is never a valid id
- [ ] The rules are one @model_validator(mode="after") that returns self; no @staticmethod validate_extracted_parameters
- [ ] Unambiguous values are normalised by assigning to self.<field>, not rejected
- [ ] Every refusal is raise fastworkflow.Refused(message, field=...) (or a plain ValueError); no other exceptions escape
- [ ] Each message names the field, the bad value, and where a good one comes from
- [ ] Errors accumulate into one Refused
- [ ] No backend calls
- [ ] Safety-critical invariants are also enforced in the command body
```

## Related

- Handle parameters that never arrived: the `declare-parameter-producers` skill.
- Values that arrived nearly right and should be fuzzy-resolved: the `resolve-parameter-values` skill.

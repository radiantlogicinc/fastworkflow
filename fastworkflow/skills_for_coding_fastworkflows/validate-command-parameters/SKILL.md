---
name: validate-command-parameters
description: >-
  Implement the `Signature.validate_extracted_parameters` hook on a fastWorkflow command so
  whole-call rules, cross-field constraints, context preconditions and value lookups are checked and
  corrected before the command body runs, and so a bad call stays a conversation instead of becoming
  an exception. Covers the exact contract (return None to accept, write corrections to `input`,
  refuse with `fastworkflow.Refused(message, field=...)`), where it sits in the validation order,
  writing error messages an agent can act on, and what belongs here versus in the field declaration.
  Use when a command has rules spanning more than one field, needs a precondition on the current
  context, must normalise input before executing, or when authoring or reviewing any command `Signature`.
---

# Validating extracted parameters

Pydantic answers one field at a time: right type, matches the pattern, was supplied. It cannot answer
the questions that only exist once the whole call is on the table — is this end date before its own
start date, is anything actually selected for this bulk command to operate on, does this decision
value exist in the backend's vocabulary. `validate_extracted_parameters` is where those go. It is the
one validation hook a command has.

## The contract

A static method on the command's `Signature` class:

```python
class Signature:
    class Input(BaseModel):
        ...

    @staticmethod
    def validate_extracted_parameters(
        workflow: fastworkflow.Workflow,
        command: str,
        input: "Signature.Input",
    ) -> None:
        ...
```

- Return `None` to accept the call. Do not return a tuple: `(True, '')` or `(False, 'msg')` is a call error.
- Correct a value by assigning to `input.<field>`. The corrected value is what the command body reads.
- Refuse only by raising `fastworkflow.Refused(message)`. Pass `field=` to reset the named field(s) to
  missing and ask the user for them again with the message:

```python
raise fastworkflow.Refused("decision 'approve-all' is not one of the accepted values. Use one of: approve, revoke, defer.",
                           field="decision")
```

The hook takes an optional `command_text` parameter if it needs the raw utterance. It is optional: a
command without the hook simply skips this stage.

## Where it runs in the order

The framework validates in this sequence, and **stops at the first failure**:

```
1. type coercion per field   str/int/float/bool/Enum/list, plus any `pattern`
2. missing required fields   sentinel values count as missing; empty list counts as missing
3. validate_extracted_parameters      <- only reached if 1-2 all passed
```

So this hook may assume every field is **present and of the declared type**. Do not re-check types,
required-ness or regex here — declare those on the field and let stage 1 do it. What is left is
exactly what the field declarations cannot express. Value lookups against a live key set also live
here now, see the `resolve-parameter-values` skill.

Before validation, in agent mode, the values are bound in code when the call's form allows it: XML
tags, `name=value` pairs, or `command_name value` (positional). The positional form binds to the
field declared `json_schema_extra={'positional': True}` (several words allowed), or else to the
class's only `str` field (one token only); it declines on anything doubtful and the LLM extracts
instead. Mark at most one field, and not a free-text field of a command that changes state. A lone
`command_name value` that binds to no field is refused with a message naming the parameters, so the
command never runs on its defaults instead. Whatever bound the values, the sequence above still runs.

## It is a correction hook, not just a check

The `input` object it receives is the one the command body will read, so writing to it is
the intended way to normalise:

```python
    @staticmethod
    def validate_extracted_parameters(workflow, command, input):
        if not input.order_id.startswith('#'):
            input.order_id = f'#{input.order_id}'
```

Prefer normalising to rejecting whenever the intent is unambiguous. Every rejection costs a round
trip; a value that can only have meant one thing should just be fixed.

## Refuse with Refused, never with a bare exception

Exceptions other than `fastworkflow.Refused` are caught, logged at critical, and converted into a
call error whose message names the command and the exception. That is strictly worse than a message
written for the caller: the caller gets a stack-trace-flavoured string instead of an instruction.
Refusals are `raise fastworkflow.Refused(...)`, with a message written for the user or agent.

## Write the message for an agent

The message is the only thing the caller has to work from. Every message should name the field, say
what is wrong with the value that arrived, and say where an acceptable one comes from.

```python
raise fastworkflow.Refused(
    "decision 'approve-all' is not one of the accepted values. Use one of: approve, revoke, defer.",
    field="decision")
```

Vague messages ("invalid input") cause the agent to retry the same value.

## Accumulate errors, report once

Check everything, then raise once, so one round trip tells the caller all of it. `Refused` takes one
message, so collect the messages and raise the joined text. Pass `field=` only when the named fields
are the ones to re-ask:

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
| normalising | `(value, error)` — store the value back | a code to upper-case, a uid to strip |
| relational | `error` | two or more fields that must agree |
| reading | `value` | turning an intent the caller can express into the value the body wants |

```python
    @staticmethod
    def validate_extracted_parameters(workflow, command, input):
        errors = []
        input.decision, error = one_of(input.decision, DECISIONS)
        errors.append(error)
        errors.append(in_order(input.begin_date, input.end_date))
        refuse_if_any(errors)
```

## Keep it local and cheap

The hook runs on **every** attempt at the command, including the ones an agent is guessing at, and
including retries after a correction. Do not call a backend from here. Check against values already
in the workflow context, or against a constant vocabulary. If a check genuinely needs a live key set,
do the `DatabaseValidator.fuzzy_match` lookup here: see the `resolve-parameter-values` skill.

## Context preconditions belong here

This is the only hook that sees the workflow before the body runs, so it is where "is this command
meaningful right now" goes:

```python
        selection = getattr(workflow.command_context_for_response_generation, 'selection', None)
        if not selection:
            raise fastworkflow.Refused("Nothing is selected. Use select_items first, then retry.")
```

Note the message names the producing command, the same job the `from_` edge does for a parameter
that never arrived (the runtime message reads `use the <producers> command(s) to get <field> information`).

## Do not treat it as a security boundary

The hook runs on both the NLU path and the direct-action path, but it is a command's own declaration
of what it needs, not an enforcement layer — a caller reaching the body another way bypasses it.
Invariants that must hold for correctness or safety belong in the command body as well.

## Authoring checklist

```
- [ ] Type, pattern and required-ness are declared on the field, not re-checked here
- [ ] The hook returns None; it never returns a tuple
- [ ] Unambiguous values are normalised by assigning to input.<field>, not rejected
- [ ] Every refusal is raise fastworkflow.Refused(message, field=...); no other exceptions escape
- [ ] Each message names the field, the bad value, and where a good one comes from
- [ ] Errors accumulate into one Refused
- [ ] No backend calls
- [ ] Safety-critical invariants are also enforced in the command body
```

## Related

- Handle parameters that never arrived: the `declare-parameter-producers` skill.
- Values that arrived nearly right and should be fuzzy-resolved: the `resolve-parameter-values` skill.

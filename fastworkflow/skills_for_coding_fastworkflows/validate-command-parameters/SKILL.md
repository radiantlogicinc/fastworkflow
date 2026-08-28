---
name: validate-command-parameters
description: >-
  Implement the `Signature.validate_extracted_parameters` hook on a fastWorkflow command so
  whole-call rules, cross-field constraints and context preconditions are checked and corrected
  before the command body runs, and so a bad call stays a conversation instead of becoming an
  exception. Covers the exact contract and where it sits in the validation order, normalising values
  in place, writing error messages an agent can act on, and what belongs here versus in the field
  declaration. Use when a command has rules spanning more than one field, needs a precondition on
  the current context, must normalise input before executing, or when authoring or reviewing any
  command `Signature`.
---

# Validating extracted parameters

Pydantic answers one field at a time: right type, matches the pattern, was supplied. It cannot answer
the questions that only exist once the whole call is on the table — is this end date before its own
start date, is anything actually selected for this bulk command to operate on, does this decision
value exist in the backend's vocabulary. `validate_extracted_parameters` is where those go.

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
        cmd_parameters: "Signature.Input",
    ) -> tuple[bool, str]:
        ...
        return (True, '')
```

Return `(True, '')` to proceed, or `(False, message)` to send `message` back to the caller. The hook
is optional — a command without it simply skips this stage.

## Where it runs in the order

The framework validates in this sequence, and **stops at the first failure**:

```
1. type coercion per field   str/int/float/bool/Enum/list, plus any `pattern`
2. missing required fields   sentinel values count as missing; empty list counts as missing
3. db_lookup per field       fuzzy resolution against a live key set
4. validate_extracted_parameters      <- only reached if 1-3 all passed
```

So this hook may assume every field is **present and of the declared type**. Do not re-check types,
required-ness or regex here — declare those on the field and let stage 1 do it. What is left is
exactly what the field declarations cannot express.

## It is a correction hook, not just a check

The `cmd_parameters` object it receives is the one the command body will read, so writing to it is
the intended way to normalise:

```python
    @staticmethod
    def validate_extracted_parameters(workflow, command, cmd_parameters):
        if not cmd_parameters.order_id.startswith('#'):
            cmd_parameters.order_id = f'#{cmd_parameters.order_id}'
        return (True, '')
```

Prefer normalising to rejecting whenever the intent is unambiguous. Every rejection costs a round
trip; a value that can only have meant one thing should just be fixed.

## Return, never raise

Exceptions are caught, logged at critical, and converted into a failure whose message names the
command and the exception. That is strictly worse than a message written for the caller: the caller
gets a stack-trace-flavoured string instead of an instruction. Return `(False, message)`.

## Write the message for an agent

The message is the only thing the caller has to work from. Every message should name the field, say
what is wrong with the value that arrived, and say where an acceptable one comes from.

```python
        return (False,
                "decision 'approve-all' is not one of the accepted values. "
                "Use one of: approve, revoke, defer.")
```

Vague messages ("invalid input") cause the agent to retry the same value.

## Accumulate errors, report once

Check everything, then report together, so one round trip tells the caller all of it:

```python
def report(errors):
    """The (is_valid, message) pair the framework expects, from a list of
    error strings of which most are normally empty."""
    message = "\n".join(error for error in errors if error)
    return (False, message) if message else (True, "")
```

Three helper shapes are worth distinguishing at the call site:

| Shape | Returns | Use for |
|---|---|---|
| normalising | `(value, error)` — store the value back | a code to upper-case, a uid to strip |
| relational | `error` | two or more fields that must agree |
| reading | `value` | turning an intent the caller can express into the value the body wants |

```python
    @staticmethod
    def validate_extracted_parameters(workflow, command, cmd_parameters):
        errors = []
        cmd_parameters.decision, error = one_of(cmd_parameters.decision, DECISIONS)
        errors.append(error)
        errors.append(in_order(cmd_parameters.begin_date, cmd_parameters.end_date))
        return report(errors)
```

## Keep it local and cheap

The hook runs on **every** attempt at the command, including the ones an agent is guessing at, and
including retries after a correction. Do not call a backend from here. Check against values already
in the workflow context, or against a constant vocabulary. If a check genuinely needs a live key set,
that is what `db_lookup` is for — see the `resolve-parameter-values` skill.

## Context preconditions belong here

This is the only hook that sees the workflow before the body runs, so it is where "is this command
meaningful right now" goes:

```python
        selection = getattr(workflow.command_context_for_response_generation, 'selection', None)
        if not selection:
            errors.append("Nothing is selected. Use select_items first, then retry.")
```

Note the message names the producing command, the same job `available_from` does for a parameter
that never arrived.

## Do not treat it as a security boundary

The hook runs on both the NLU path and the direct-action path, but it is a command's own declaration
of what it needs, not an enforcement layer — a caller reaching the body another way bypasses it.
Invariants that must hold for correctness or safety belong in the command body as well.

## Authoring checklist

```
- [ ] Type, pattern and required-ness are declared on the field, not re-checked here
- [ ] Unambiguous values are normalised in place rather than rejected
- [ ] Every failure returns (False, message); nothing raises
- [ ] Each message names the field, the bad value, and where a good one comes from
- [ ] Errors accumulate into one message
- [ ] No backend calls
- [ ] Safety-critical invariants are also enforced in the command body
```

## Related

- Handle parameters that never arrived: the `declare-parameter-producers` skill.
- Values that arrived nearly right and should be fuzzy-resolved: the `resolve-parameter-values` skill.

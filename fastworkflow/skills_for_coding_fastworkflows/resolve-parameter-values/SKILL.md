---
name: resolve-parameter-values
description: >-
  Resolve a value the user typed against a live key set inside `validate_extracted_parameters`, so
  typos are corrected automatically (the hook writes the corrected value back) and near misses come
  back as "did you mean" suggestions (the hook raises `fastworkflow.Refused(..., field=...)`) instead
  of a not-found. Covers the `DatabaseValidator.fuzzy_match` contract, the write-back and refusal,
  the three-stage matcher and its thresholds, tuning auto-apply versus suggest, the key-set design,
  and the failure modes of a badly chosen key set. Use when a parameter names a real-world entity the
  user types by hand (a person, product, list, room, tag), or when a command keeps failing because the
  user's spelling does not match stored data. `db_lookup` is retired; this is its replacement.
---

# Resolving parameter values in the validation hook

A parameter the user types from memory will not match stored data exactly. The resolution step runs
inside `Signature.validate_extracted_parameters`, the one validation hook for a command. An exact
match passes, a close match is corrected in place, an ambiguous one is refused with suggestions, and a
genuine miss is accepted for the hook to judge (or for the command body to reject).

`db_lookup` is retired. A `json_schema_extra={'db_lookup': True}` key or a `Signature.db_lookup`
method is a load error. Move the lookup into `validate_extracted_parameters` as shown below.

## Declaring the field

No special key is needed. Declare the field as a plain parameter:

```python
class Signature:
    class Input(BaseModel):
        user_name: str = Field(
            description="Name of a person",
            examples=['John', 'Jane Doe'],
        )
```

## The hook

A static method on the same `Signature`. It takes `(workflow, command, input)`, and an optional
`command_text` parameter if it needs the raw utterance. It returns `None` to accept. It corrects
values by assigning to `input.<field>`. It refuses only by raising `fastworkflow.Refused`.

```python
import fastworkflow
from fastworkflow.utils.signatures import DatabaseValidator

class Signature:
    class Input(BaseModel):
        user_name: str = Field(description="Name of a person", examples=['John', 'Jane Doe'])

    @staticmethod
    def validate_extracted_parameters(workflow: fastworkflow.Workflow,
                                      command: str,
                                      input: "Signature.Input") -> None:
        chatroom = workflow.command_context_for_response_generation
        key_values = chatroom.list_users()
        matched, corrected, suggestions = DatabaseValidator.fuzzy_match(input.user_name, key_values)
        if matched:
            input.user_name = corrected              # write the corrected value back
        elif suggestions:
            raise fastworkflow.Refused(
                f"No user named '{input.user_name}'. Did you mean {', '.join(suggestions)}?",
                field="user_name")                   # resets user_name to missing and asks again
        # else: accept; absence from the key set is not by itself an error
```

Fetch the key set **at call time** from the live context, `workflow.command_context_for_response_generation`,
never from a module-level constant. The valid set depends on where the user is.

If the hook validates several fields, dispatch on the fields you own and leave the rest alone. Test an
optional field for `None` before you match on it.

## The three outcomes of `fuzzy_match`

`matched, corrected, suggestions = DatabaseValidator.fuzzy_match(value, key_values)`, and the three
outcomes are genuinely different:

| `fuzzy_match` result | What the hook does | Effect |
|---|---|---|
| `matched` is `True` | `input.<field> = corrected` | the value is **overwritten** and the rest of the pipeline sees it; the diagnostic records a `changed_fields` entry |
| `matched` is `False`, `suggestions` non-empty | `raise fastworkflow.Refused(msg, field="<field>")` | the field is reset to missing and the user is asked again with the message; the diagnostic records a `refused_fields` entry |
| `matched` is `False`, `suggestions` empty | return `None` | **accepted**: absence from the key set is not by itself an error |

That last row is the one that surprises people. A non-empty suggestion list is what refuses the field,
so an over-eager matcher rejects values that are merely absent from this particular list. If a value
must exist, enforce that in the command body or in the same hook with an explicit `Refused`, not by
hoping the matcher rejects it.

Do not return a tuple from the hook. A `(True, '')` or `(False, 'msg')` return is a call error.

## Where it runs

```
1. type coercion per field
2. missing required fields
3. validate_extracted_parameters        <- here: lookup, write-back, refusal
```

The hook runs only after the earlier checks pass, so it sees the coerced value. Because it runs before
the command executes, the command body sees the **corrected** value.

## The matcher and its thresholds

`DatabaseValidator.fuzzy_match(value, key_values, threshold=0.6, suggest_threshold=0.7,
auto_apply_threshold=0.2)` runs three stages in order, each short-circuiting:

1. case-insensitive exact match → matched
2. Levenshtein within `suggest_threshold` → a **unique** candidate within `auto_apply_threshold` is
   matched; otherwise the candidates are returned as suggestions
3. `difflib` at `threshold` → suggestions only

What the three knobs actually control:

- **`auto_apply_threshold` (0.2)** — how confident the matcher must be to overwrite the user's value
  without saying so. Deliberately strict: the metric is edits divided by length, so a one-edit typo
  on a value shorter than about five characters exceeds it and is offered as a suggestion instead.
  Being generous here silently substitutes non-members. Raise it per command only when an extra
  clarification turn costs more than a wrong substitution.
- **`suggest_threshold` (0.7)** — how far a candidate can be and still be considered at all.
- **`threshold` (0.6)** — the `difflib` cutoff for the last stage, which only sees values no
  candidate came close to. Loosening it produces unrelated suggestions, and since any non-empty
  suggestion list refuses the field, those unrelated suggestions reject values that were simply not in
  the list.

Pass a custom key set and keep the defaults unless you have a measured reason:

```python
matched, corrected, suggestions = DatabaseValidator.fuzzy_match(
    input.user_name, key_values, auto_apply_threshold=0.35)
```

You may also implement matching yourself. The hook only needs the three-tuple from your own logic.

## Choosing the key set

The key set is the whole design decision.

- **Scope it to the current context.** Listing every entity in the system makes near misses more
  likely and makes a correction more likely to be wrong.
- **Use the labels the user says, not the ids they never see.** Matching on a uid is pointless; the
  point is to map a human name onto a stored one.
- **Keep it cheap.** The hook runs on every attempt at the command, including retries. If the set can
  only come from a network call, cache it on the context object rather than fetching per attempt.
- **Do not let it grow unbounded.** A key set of thousands of similar strings will produce
  suggestions for everything and reject a lot of valid input.

## When not to use it

- The parameter is an **opaque handle** the user never types. Use `from_` to point at the
  command that produces it — see the `declare-parameter-producers` skill.
- The valid values are a **fixed vocabulary**. Use an `Enum`, which the framework already coerces and
  reports valid values for.
- The check is **relational** — this field must agree with that one. Keep it in the same hook; see the
  `validate-command-parameters` skill.

## Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| Valid values rejected with irrelevant suggestions | `threshold` too loose, or key set too small/wrong | tighten `threshold`; check the key set is the one the user is naming |
| The command silently acts on the wrong entity | `auto_apply_threshold` too generous | lower it; let the near miss become a suggestion |
| A typo the user obviously meant needs a clarification turn | short values exceed the ratio metric | raise `auto_apply_threshold` for that command |
| Load error: `db_lookup is retired` | the field still carries `json_schema_extra={'db_lookup': True}` or the Signature still defines `db_lookup` | delete the key and the method; put the lookup in `validate_extracted_parameters` |
| Hook exception reported as a call error | the hook raised something other than `Refused`, or returned a tuple | raise `fastworkflow.Refused` for a refusal; return `None` otherwise |
| Every attempt is slow | key set fetched over the network per attempt | cache it on the context object |

## Authoring checklist

```
- [ ] No db_lookup key on the field and no Signature.db_lookup method (both are load errors)
- [ ] validate_extracted_parameters(workflow, command, input) is a staticmethod returning None
- [ ] fuzzy_match result: matched -> input.<field> = corrected; suggestions -> Refused(..., field="<field>")
- [ ] Key set is read from the live context, scoped to it, and cheap to obtain
- [ ] Key set holds the labels the user actually says
- [ ] Thresholds left at defaults unless there is a measured reason
- [ ] Existence is enforced in the hook or the body if absence must be an error
```

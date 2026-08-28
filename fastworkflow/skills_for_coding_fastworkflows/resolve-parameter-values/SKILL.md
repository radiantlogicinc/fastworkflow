---
name: resolve-parameter-values
description: >-
  Implement `db_lookup` on a fastWorkflow command parameter so a value the user typed is resolved
  against a live key set, typos are corrected automatically and near misses come back as "did you
  mean" suggestions instead of a not-found. Covers the field declaration, the
  `Signature.db_lookup` contract, the three-stage fuzzy matcher and its thresholds, tuning
  auto-apply versus suggest, and the failure modes of a badly chosen key set. Use when a parameter
  names a real-world entity the user types by hand — a person, product, list, room, tag — or when a
  command keeps failing because the user's spelling does not match stored data.
---

# Resolving parameter values with `db_lookup`

A parameter the user types from memory will not match stored data exactly. `db_lookup` puts a
resolution step between extraction and execution: an exact match passes, a close match is corrected
silently, an ambiguous one comes back as suggestions, and a genuine miss falls through.

## Declaring it

```python
class Signature:
    class Input(BaseModel):
        user_name: str = Field(
            description="Name of a person",
            examples=['John', 'Jane Doe'],
            json_schema_extra={'db_lookup': True},
        )
```

## The hook

A static method on the same `Signature`, dispatching on field name:

```python
from fastworkflow.utils.signatures import DatabaseValidator

    @staticmethod
    def db_lookup(workflow: fastworkflow.Workflow,
                  field_name: str,
                  field_value: str) -> tuple[bool, str | None, list[str]]:
        if field_name == 'user_name':
            chatroom = workflow.command_context_for_response_generation
            key_values = chatroom.list_users()
            return DatabaseValidator.fuzzy_match(field_value, key_values)
        return (False, '', [])
```

One hook serves every `db_lookup` field on the command, so **always dispatch on `field_name`** and
always return the neutral `(False, '', [])` for anything unrecognised.

Fetch the key set **at call time** from the live context — `workflow.command_context_for_response_generation`
— never from a module-level constant. The valid set depends on where the user is.

## The return contract

`(matched, corrected_value, suggestions)`, and the three outcomes are genuinely different:

| Return | Effect |
|---|---|
| `(True, value, [])` | the parameter is **overwritten** with `value` and validation continues |
| `(False, None, [s1, s2])` | the parameter is marked invalid; the caller is asked "Did you mean one of these …?" |
| `(False, '', [])` | validation **passes** — absence from the key set is not by itself an error |

That last row is the one that surprises people. A non-empty suggestion list is what fails the
parameter, so an over-eager matcher rejects values that are merely absent from this particular list.
If a value must exist, enforce that in the command body or in
`validate_extracted_parameters`, not by hoping the matcher rejects it.

## Where it runs

```
1. type coercion per field
2. missing required fields
3. db_lookup                            <- here
4. validate_extracted_parameters
```

It never sees a missing value — fields holding the not-found sentinel or `None` are skipped — so the
hook can assume a non-empty string. And because it runs *before*
`validate_extracted_parameters`, that hook sees the **corrected** value.

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
  suggestion list fails validation, those unrelated suggestions reject values that were simply not in
  the list.

Pass a custom key set and keep the defaults unless you have a measured reason:

```python
            return DatabaseValidator.fuzzy_match(
                field_value, key_values, auto_apply_threshold=0.35)
```

You may also implement matching yourself — the framework only requires the three-tuple.

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

- The parameter is an **opaque handle** the user never types. Use `available_from` to point at the
  command that produces it — see the `declare-parameter-producers` skill.
- The valid values are a **fixed vocabulary**. Use an `Enum`, which the framework already coerces and
  reports valid values for.
- The check is **relational** — this field must agree with that one. Use
  `validate_extracted_parameters`; see the `validate-command-parameters` skill.

## Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| Valid values rejected with irrelevant suggestions | `threshold` too loose, or key set too small/wrong | tighten `threshold`; check the key set is the one the user is naming |
| The command silently acts on the wrong entity | `auto_apply_threshold` too generous | lower it; let the near miss become a suggestion |
| A typo the user obviously meant needs a clarification turn | short values exceed the ratio metric | raise `auto_apply_threshold` for that command |
| `ValueError: input_for_param_extraction_class is not set` | the hook could not be resolved on the calling path | the `Signature` must define `db_lookup`; confirm the command is being invoked through a path that resolves the signature class |
| Every attempt is slow | key set fetched over the network per attempt | cache it on the context object |

## Authoring checklist

```
- [ ] Field declares json_schema_extra={'db_lookup': True}
- [ ] Signature.db_lookup dispatches on field_name and returns (False, '', []) for the rest
- [ ] Key set is read from the live context, scoped to it, and cheap to obtain
- [ ] Key set holds the labels the user actually says
- [ ] Thresholds left at defaults unless there is a measured reason
- [ ] Existence is enforced in the body if absence must be an error
```

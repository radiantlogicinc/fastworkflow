---
paths:
  - "**/_commands/**"
  - "**/context_inheritance_model.json"
---

# Authoring fastWorkflow Commands

## Workflow Directory Structure

```
my_workflow/
├── application/                     # Your app code (untouched)
├── _commands/                       # Command implementations (generated + edited)
│   ├── context_inheritance_model.json  # Context/command hierarchy
│   └── <command_name>.py            # Single-file command (preferred)
├── ___command_info/                 # Generated at train-time (gitignore)
├── ___workflow_contexts/            # Session state at run-time (gitignore)
└── ___convo_info/                   # Conversation logs (gitignore)
```

Add to `.gitignore`: `___workflow_contexts`, `___command_info`, `___convo_info`

## Command File Structure (Single-File Pattern)

New commands use a **single `.py` file** in `_commands/`. Legacy commands with subdirectories (`parameter_extraction/`, `response_generation/`, `utterances/`) are being migrated to this pattern. When you encounter both, use the single file.

```python
# _commands/<command_name>.py

class Signature:
    plain_utterances: list[str] = [...]   # Seed utterances; the framework adds the synthetic variants

    class Input(BaseModel):               # Pydantic params; no default = required (asked for if missing)
        some_param: str = Field(description="...", examples=[...])
        note: Optional[str] = None        # Optional[...] = None = optional

        @model_validator(mode="after")     # Optional; see the validation section below
        def some_rule(self, info):         # info.context: "command_text", "workflow", "command"
            ...                            # assign self.<field> to correct, then return self;
                                           # raise fastworkflow.Refused(message, field=...) to refuse

    class Output(BaseModel):              # Structured result
        ...


class ResponseGenerator:
    def __call__(self, workflow: fastworkflow.Workflow, command: str,
                 input: Signature.Input) -> Signature.Output:
        # input is None when the command has no Signature.Input.
        # Return the Output (the framework fills the envelope), a str (response text),
        # or fastworkflow.CommandResponse (artifacts or next actions only).
        # Refuse with raise fastworkflow.Refused(message).
        ...
```

A two-argument `__call__` is a load error. `CommandOutput` is not a return value.

## `plain_utterances` — the seed list is the highest-leverage thing you write

`plain_utterances` is not decoration. Training expands it with an LLM into the synthetic
corpus that the intent classifier learns from, so seed count and seed variety are the
single largest measured input to whether the right command gets picked at run time —
larger than the persona/utterance-count dials, larger than anything in the training loop.

Guidance, with its provenance stated honestly:

- **Aim for roughly eight seed utterances per command.** On one 160-command workflow
  scored against a 446-case hand-written benchmark, everything else held constant,
  held-out routing top-1 went 46.2% at 3.2 seeds/command → 70.4% at 8.0 → 73.8% at 9.3:
  steep, then flat. **That is an observation from a single workflow, not a universal
  constant** — the shape (steep early, flattening) is what transfers; the exact knee will
  differ for you. Two or three seeds is the starved case worth fixing first.
- **Vary the phrasing family, not the wording.** Imperative, question, colloquial, terse,
  synonym-heavy, and value-bearing phrasings each buy something. Six paraphrases of one
  sentence buy almost nothing.
- **Never paste a failing benchmark case into `plain_utterances` to "fix" it.** That turns
  your benchmark into a memorisation test. Keep the seeds disjoint from
  `<workflow>/intent_benchmark.json` (see `docs/intent_benchmark_format.md`);
  `heldout_evaluation.assert_benchmark_disjoint_from_seeds` is called during training
  (`fix-eia`, fixed) and aborts the run with `BenchmarkLeakError`, so a leak fails the
  build instead of quietly inflating the score.
- **Count what you actually have** after a run: `seed_utterance_count` per command in
  `<workflow>/___command_info/training_provenance.json`. The same file's `fell_back` flag
  tells you whether a command's utterances degraded to seeds-only because generation was
  rate-limited.

Judging whether adding seeds actually helped is a measurement problem with its own
pitfalls (two training runs are not reproducible): see the
`fastworkflow-intent-training-convergence` skill before comparing two runs.

## Validating parameters: model validators on `Input`

Cross-field rules, context preconditions and value lookups are `@model_validator(mode="after")` methods on the
command's `Input`. That is the one validation mechanism: the `validate_extracted_parameters` hook is removed, and a
`Signature` that still defines it fails the load with the validator form in the message. Pydantic checks types,
patterns and required fields first, so a validator can assume every field is present and of its type.

- Correct by assigning `self.<field>` and returning `self`.
- Refuse with `raise fastworkflow.Refused(message, field=...)`. A plain `ValueError` is a refusal with no field.
  Any other exception becomes a call error naming the command, so raise `Refused` for any refusal you mean the user to see.
- `info.context` has `"command_text"` (the user's text, `""` while a planning template renders), `"workflow"`
  (`None` while a planning template renders) and `"command"` (the qualified name), for both kinds of command.
  Code that builds an `Input` directly, with no context, gets a clear error: use `Input.model_validate(values, context={...})`
  or `Input.model_construct(...)` in tests.

**Upgrade step (migrating a hook).** The old order was option checks, then the hook (which could correct fields), then
the Input's own model validators, then roles. Pydantic runs model validators in declaration order, so when you move a
hook body into a `@model_validator(mode="after")`, declare that validator **first**, before any model validators the
`Input` already has. Otherwise an existing check sees the uncorrected value. (In one workflow, a kind-fit check refused
before the kind resolver ran.)

A value lookup against a live set of legal values goes in a validator too. `db_lookup` is retired: a
`json_schema_extra={"db_lookup": True}` key or a `Signature.db_lookup` method is a load error.

```python
class Signature:
    class Input(BaseModel):
        user_name: str = Field(description="Name of a person")

        @model_validator(mode="after")
        def user_name_is_known(self, info):
            workflow = info.context["workflow"]
            if workflow is None:               # a planning command's template is rendering
                return self
            candidates = workflow.record.list_users()
            matched, corrected, suggestions = DatabaseValidator.fuzzy_match(self.user_name, candidates)
            if matched:
                self.user_name = corrected
            elif suggestions:
                raise fastworkflow.Refused(
                    f"No user named '{self.user_name}'. Did you mean {', '.join(suggestions)}?",
                    field="user_name")
            return self                        # else: accept
```

`DatabaseValidator` is `fastworkflow.utils.signatures.DatabaseValidator`. `workflow.record` is the record this command acts on. The three outcomes:

| `fuzzy_match` result | what the validator does | what the framework records |
|---|---|---|
| `matched` | assigns `self.<field> = corrected` | `changed_fields` lists the field |
| `suggestions` non-empty | `raise fastworkflow.Refused(msg, field="<field>")` | `refused_fields` lists the field; the field is reset to missing and asked again with the message |
| neither | returns `self` | nothing: the value is accepted |

The last outcome means "I have no opinion, leave the extracted value alone", and it is what you want
whenever a field can legitimately hold a value that is not in your candidate list. Any non-empty
suggestion list refuses the field, so feeding real uids to a field whose candidates are display names
rejects valid input. Decline instead, before the lookup:

```python
    if _looks_like_a_uid(self.user_name):
        return self                    # not a name; not mine to judge
```

**Assign the value the field holds, not the value you matched on.** A lookup that returns a display
label is right for a field holding a name. On a `*_uid` field, map the label back to the uid yourself:
`self.user_uid = uid_of[corrected]`.

**Tuning the match, per command.** `DatabaseValidator.fuzzy_match` runs a
case-insensitive exact match, then Levenshtein, then `difflib`, and each stage
short-circuits. Because the validator is your code, you tune it by passing arguments:

- `auto_apply_threshold` (default `0.2`): how far off a *unique* match may be before
  the framework rewrites the user's value. Deliberately strict: at `0.7`, `"Batman"`
  scored 0.333 against a customer list and was silently applied. Distance is
  edits ÷ length, so a one-edit typo on a value under ~5 characters exceeds `0.2` and
  is offered as a suggestion instead. Raise it if an extra clarification turn costs
  more than a wrong substitution; `0.0` accepts only exact and containing matches.
- `suggest_threshold` (default `0.7`): how far off a candidate may be to be offered
  at all.
- `threshold` (default `0.6`): `difflib` cutoff for the final suggestion-only stage.
  Only values that nothing came within `suggest_threshold` of reach it, so loosening
  this yields unrelated suggestions, which refuse the value.

## `command_text_after_command_name`: binding `command value` without the LLM

In agent mode, parameters are parsed in code before the LLM is asked: XML tags, then
`name=value` pairs, then the form `command_name value`. That last form binds the text after the command
name to one target field:

- the field marked `json_schema_extra={"command_text_after_command_name": True}`, or, with no mark,
- the class's only `str` field (an `int` paging field does not count).

An unmarked field takes a single token only (`open_by_uid 9f2c…`); a marked field also
takes several words (`find_item Ann Lee 2`, where the trailing `2` stays in the value).
Mark the field a bare value most naturally fills, when the class has more than one `str`
field or the value can contain spaces:

```python
class Input(BaseModel):
    query: Optional[str] = Field(default=None, description="Name or name fragment",
                                 json_schema_extra={"command_text_after_command_name": True})
    start: Optional[int] = Field(default=None, description="Paging offset")
```

Mark at most one field per class; two marks disable the form. Do not mark a free-text
field of a command that changes state, where prose sent by mistake would become the
value. The form declines (and the LLM extracts as before) when the first token is not
the command's own name, when the text holds `=` or XML, when the first word names
another field, or when the value holds `*`, `` ` ``, `?`, `\`, ` / `, `<>{}` or no letter
or digit. A leading word naming the target field itself (`uid 9f2c…`) is dropped, and
matching surrounding quotes are removed. The usual validation runs after.

A lone value that binds to no field (`command_name value` where the LLM fills nothing)
is refused with a message naming the parameters that could take it, instead of running
the command with its defaults.


## Context Model (`context_inheritance_model.json`)

Each context entry has exactly one key, `base` (required): the list of parent context names whose commands are inherited. A `/` key is rejected; a context's own commands come from the filesystem (`_commands/<Context>/<command_name>.py`, or `_commands/<command_name>.py` for the global context `*`).

To add a new command: create its `.py` file in the context's folder; add the context to the JSON only to give it a `base`.

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
    plain_utterances: list[str] = [...]   # Seed utterances for training
    template_utterances: list[str] = [...] # Optional parameterized patterns

    class Input(BaseModel):               # Pydantic params; use NOT_FOUND default
        model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)
        some_param: Annotated[str, Field(default="NOT_FOUND", description="...", examples=[...])]

    class Output(BaseModel):              # Structured result
        ...

    @staticmethod
    def generate_utterances(...): ...     # Optional: diverse utterance generation

    @staticmethod                          # Optional; see the section below
    def db_lookup(workflow, field_name, field_value
                  ) -> tuple[bool, str | None, list[str]]: ...
    @staticmethod
    def process_extracted_parameters(...): ...  # Optional post-extraction hook


class ResponseGenerator:
    def __call__(self, workflow_snapshot, command: str, cmd_parameters: Signature.Input) -> fastworkflow.CommandOutput:
        return self._process_command(workflow_snapshot, command, cmd_parameters)

    @staticmethod
    def _process_command(workflow_snapshot, command, cmd_parameters) -> fastworkflow.CommandOutput:
        # Your business logic here
        ...
```

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

## `db_lookup` — validating a parameter against a live set of legal values

Mark the field, then implement the hook:

```python
class Input(BaseModel):
    user_name: str = Field(description="Name of a person",
                           json_schema_extra={'db_lookup': True})

@staticmethod
def db_lookup(workflow: fastworkflow.Workflow, field_name: str, field_value: str
              ) -> tuple[bool, str | None, list[str]]:
    if field_name == 'user_name':
        candidates = workflow.command_context_for_response_generation.list_users()
        return DatabaseValidator.fuzzy_match(field_value, candidates)
    return (False, None, [])
```

`messaging_app_4`'s `ChatRoom/set_current_user` is the working example.

**The return value has three states, not two, and the third is the one people miss:**

| return | what the framework does |
|---|---|
| `(True, corrected_value, [])` | overwrites the field with `corrected_value` |
| `(False, None, [suggestions])` | fails validation and shows the suggestions |
| `(False, None, [])` | **nothing** — no overwrite, no failure |

The third state means "I have no opinion, leave the extracted value alone", and it is
what you want whenever a field can legitimately hold a value that is not in your
candidate list. Feeding real uids to a field whose candidates are display names
rejected 34 of 40 valid uids before this was understood, because *any* non-empty
suggestion list fails validation. Decline instead:

```python
    if _looks_like_a_uid(field_value):
        return (False, None, [])       # not a name; not mine to judge
```

**Return the value the field holds, not the value you matched on.** `set_current_user`
returns the matched *label*, which is right for a field holding a name. Copy that shape
onto a `*_uid` field and you leave a display name in the uid field, and the command then
queries the backend by display name. Map back yourself: `return (True, uid_of[label], [])`.

**Tuning the match, per command.** `DatabaseValidator.fuzzy_match` runs a
case-insensitive exact match, then Levenshtein, then `difflib`, and each stage
short-circuits. Because the hook is your code, you tune it by passing arguments — there
is no `json_schema_extra` key for this:

- `auto_apply_threshold` (default `0.2`) — how far off a *unique* match may be before
  the framework rewrites the user's value. Deliberately strict: at `0.7`, `"Batman"`
  scored 0.333 against a customer list and was silently applied. Distance is
  edits ÷ length, so a one-edit typo on a value under ~5 characters exceeds `0.2` and
  is offered as a suggestion instead. Raise it if an extra clarification turn costs
  more than a wrong substitution; `0.0` accepts only exact and containing matches.
- `suggest_threshold` (default `0.7`) — how far off a candidate may be to be offered
  at all.
- `threshold` (default `0.6`) — `difflib` cutoff for the final suggestion-only stage.
  Only values that nothing came within `suggest_threshold` of reach it, so loosening
  this yields unrelated suggestions — which, per the table above, *reject* the value.

Errors are not caught. Unlike `validate_extracted_parameters`, the `db_lookup` call is
unguarded, so an exception propagates out of parameter validation.

## Context Model (`context_inheritance_model.json`)

Each context entry has exactly two possible keys:
- `/` — list of command names available in that context
- `base` — list of parent context names whose commands are inherited

To add a new command: update the JSON, then create `_commands/<command_name>.py`. `CommandRoutingDefinition` validates that every declared command has an implementation.

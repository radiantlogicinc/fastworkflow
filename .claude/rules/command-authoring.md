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

    @staticmethod
    def db_lookup(workflow_snapshot, command) -> list[str]: ...     # Optional
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

## Context Model (`context_inheritance_model.json`)

Each context entry has exactly two possible keys:
- `/` — list of command names available in that context
- `base` — list of parent context names whose commands are inherited

To add a new command: update the JSON, then create `_commands/<command_name>.py`. `CommandRoutingDefinition` validates that every declared command has an implementation.

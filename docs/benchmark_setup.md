# Benchmark setup

Select a workflow in `run_chatbot`, then open **Benchmark setup** in the header.
Create a benchmark with a title, an optional description, and tasks with optional
prompts. Benchmark IDs, task IDs and versions are generated automatically. Saving
edits creates the next immutable version, retaining IDs for existing tasks. An
experiment keeps the exact benchmark version it was created against.

Open a benchmark to see its tasks and experiments across versions. **New
experiment** returns a unique experiment ID for the displayed version. It does
not execute tasks. Copy that ID and the displayed task IDs into your harness.

An experiment is named by its ID everywhere it is listed — in the left tree, the
breadcrumbs and its own heading. Its **description** is optional free text you
write on the experiment page, and it is editable for as long as no runner has
claimed the registration; a harness started through
`from_benchmark_experiment` records whatever the description said at that point.
Once the experiment is bound to an evidence store the description belongs to the
recorded run and the page shows it read-only.
Recorded experiments drill down through tasks, attempts and turns. Analysis is
optional and collapsed by default. Historical runs without a benchmark pin
remain in the ordinary Experiments/conversation views; they are not guessed into
a benchmark by name.

For an in-process harness (after your normal runtime initialization):

```python
from fastworkflow.experiment.runner import ExperimentHarness, ExperimentTask

harness = ExperimentHarness.from_benchmark_experiment(
    workflow_folderpath, experiment_id="exp-..."
)
result = harness.run([
    ExperimentTask(task_id="task_...", messages=["The prompt to execute"]),
])
```

The task IDs must exactly cover the registered version. Prompts are optional
setup data; the harness supplies executable messages. The existing run path
binds each attempt's conversation and turns to the experiment and task IDs.

External controllers use the same mechanism: pass `workflow_folderpath` to
`ExperimentController` (or to its `create_experiment` call), then call
`create_experiment` with the UI's experiment ID, the registered task IDs and the
usual attempt declarations. The controller looks up the immutable benchmark pin,
refuses contradictory pins/task IDs, and binds the registration to its actual
store. Continue with the existing register/claim/start/finish lifecycle. IDs are
labels, not substitutes for attempt bootstrap credentials or execution context
binding. Supplying an experiment ID alone to a raw conversation call does not
perform that lifecycle.

Benchmark versions remain in `<workflow>/benchmarks/<id>/vN.json`. Pending
experiment registrations live in `<workflow>/benchmarks/.experiments/`; they
need no execution database. Once a controller declares execution, its store path
and identity let the browser drill into that evidence even when it is outside
the default state root. One experiment identity binds to one store. Multiple
sealed stores continue to use the existing workspace viewer.

The existing `ExperimentHarness(...)`, unregistered controller experiments and
ordinary chat conversations continue to work without benchmark setup.

HTTP authoring uses the existing localhost/session authentication:

- `POST /api/benchmark-setup`: `{title, description?, tasks:[{prompt?}]}`.
  Edits also include `benchmark_id`, `expected_version`, and existing task IDs.
- `POST /api/benchmarks/<id>/experiments`: `{version, description?}` → unique experiment ID.
- `PATCH /api/benchmark-experiments/<id>`: `{description}` → the author's description,
  refused with 409 once a runner has bound the registration.
- `GET /api/benchmarks/<id>/experiments`: registered and recorded experiments.
- `GET /api/benchmark-experiments/<id>`: registration, pinned tasks and execution availability.

Authoring does not modify old evidence or migrate incompatible stores. A missing
or incompatible evidence source is reported separately from benchmark contents.


## Human feedback and analysis

In an experiment, open a task attempt and select a turn. The **Human feedback**
card accepts free-form comments. Click into planning, execution, a step, intent
detection, parameter extraction, summary, or any recorded call to comment on that
component. Each save appends a timestamped comment; earlier comments stay visible.
Synthetic groups use their recorded descendant span IDs as their anchor. A
component with no recorded spans directs feedback to the turn instead.

Comments live in the same `observability.sqlite3`, in `human_feedback`, separate
from agent-memory `feedback`. They reference the turn and recorded span IDs,
travel with database snapshots, and are removed when their turn is deleted.
They do not change turn records, spans, scores, or agent memory. Workspace snapshots
show previously captured comments read-only; add new comments in the working
experiment database. Formal blinded assignments retain their separate review UI.

Benchmark and experiment analysis editors start blank and accept arbitrary text,
including Markdown or pasted JSON, without requiring JSON syntax. Existing
structured analysis remains readable. The HTTP analysis endpoints also accept
JSON-native structured values inside an `{"analysis": ...}` envelope.

This change introduces observability schema v4. The repository's fresh-schema
policy applies: v3 evidence is not migrated. Opening an older-schema database
with the writer deletes it and starts a fresh store in its place (the read-only
viewer refuses it and leaves it untouched), so copy or seal anything you need
before running the new build against it. Benchmark definitions and
registrations do not need recreating; create a new experiment and record it into
a fresh database to use feedback. Restart `run_chatbot` to load the new UI. Do not
reuse the old experiment ID or overwrite a sealed collection to refresh its schema.


## Hierarchy navigation

The debug sidebar lists benchmarks directly, with experiments underneath.
Expand an experiment to see its conversations. The left tree stops there.
Select a conversation to see its turns on the right, then drill into a turn and
its trace components in that pane. Breadcrumbs appear only on the right and
retain the complete path from **Benchmarks** through the selected component.
The conversation stays highlighted in the left tree during deeper inspection.

Conversations outside experiments appear under **ad-hoc conversations**, grouped
by UTC date. A conversation spanning dates appears under each relevant date with
only that date's turns. Experiments without a benchmark are kept under
**Experiments without a benchmark**. Empty registered experiments remain visible.
The tree preserves source identity, including when conversation IDs repeat in
different evidence stores. Sidebar filters and the old Experiments/Benchmarks
buttons have been removed. Use the header's Benchmark setup action to create a
benchmark and select benchmark nodes to review or edit their definitions.


## Delete an unused experiment

An experiment that has not been handed to a runner offers **Delete empty experiment** on
its overview. Confirm the deletion to return to its benchmark; the benchmark version and
tasks remain unchanged. Once a runner binds an evidence store, deletion is refused even if
that store is temporarily unavailable. Deletion and runner binding are serialized, and a
retired identity cannot be reused by a delayed runner. Workspace snapshots remain read-only.
The authenticated API is `DELETE /api/benchmark-experiments/<id>`.

Actions report success or failure in dismissible notifications. They disappear automatically
and pause while hovered or focused; form errors remain beside the action. Navigation brings
keyboard focus to the new view heading, adding a task focuses its prompt, and reduced-motion
preferences disable transitions. The responsive layout keeps the left tree at conversations
and preserves complete right-pane breadcrumbs.

# Runner handoff and optional pre-run review

Use this reference when connecting an application's runner to UI-authored benchmarks. Source
contracts are `fastworkflow/benchmark/setup.py`, `benchmark/catalog.py`, `experiment/runner.py`, and
`experiment/setup.py` in the installed framework checkout.

## Resolve the frozen input

```python
from fastworkflow.benchmark.setup import experiment_manifest

registration, manifest = experiment_manifest(workflow_folderpath, experiment_id)
# Verifies the registered version's byte digest, not whatever version is newest.
```

Preserve `task_id` exactly. Resolve fixture and operator policy from the driver's own contract.
A prompt is optional metadata; if neither it nor driver data supplies runnable input, report an
incomplete setup before execution. Never invent the missing task or omit it from the denominator.

## In-process example

This example explicitly adopts a single-prompt driver convention. It runs models/backend commands
when `run()` is reached; use only after runtime configuration, fixtures and execution authorization
are in place. It is not a dry-run snippet.

```python
from fastworkflow.experiment.runner import ExperimentHarness, ExperimentTask
from fastworkflow.benchmark.setup import experiment_manifest

registration, manifest = experiment_manifest(workflow_folderpath, experiment_id)
tasks = []
for task in manifest["tasks"]:
    prompt = task.get("prompt", "")
    if not prompt.strip():
        raise ValueError(f"Driver input required for task {task['task_id']}")
    tasks.append(ExperimentTask(task_id=task["task_id"], messages=[prompt]))

harness = ExperimentHarness.from_benchmark_experiment(
    workflow_folderpath, experiment_id,
    max_workers=1,
)
# grade_attempt is the application's outcome checker (see Grader in experiment/runner.py).
result = harness.run(tasks, attempts=1, grader=grade_attempt)
```

`from_benchmark_experiment` carries the registration's `description` — the optional prose its
author wrote in the studio — into the recorded experiment. Pass `description=` explicitly to
override it.

`ExperimentTask.messages` is an in-process harness input, not a native benchmark manifest field.
Multi-turn or conditional-response drivers should map their own payload to it or use their existing
collector. The harness accepts one experiment per instance. Constructing it opens/initializes the
working evidence store even before model execution; do not instantiate it to inspect archives.

A grader receives `AttemptRun` and returns `(outcome, outcome_source, reward, detail)`;
returning `None` uses the framework's derived fallback. That fallback only measures whether
commands reported failure, and can mark a recovered command error as a failed attempt. Supply
a real application grader for task-success claims, with its identity and evidence in the result.
A task grader must check the expected application outcome. Runtime completion and all commands
succeeding do not prove that the user received the right answer. An unanswered `awaiting_user`
attempt is incomplete; do not count it as a verified success or erase it to make a run finish.
For interactive tasks, define who answers, which inputs are valid, the question/turn budget and
a timeout. Record Q/A and responder identity in the driver's evidence. A simulated responder is
not a human rater, and a saved review decision does not implement this responder.

## External / HTTP collectors

Use `ExperimentController(workflow_folderpath, expected_store_identity, ...)`; it records into
the workflow's one live DB, `state_paths.observability_db(workflow_folderpath)`, which the server
and the UI resolve too. Do not give the run its own `FASTWORKFLOW_STATE_ROOT`: a different root is
a different DB, so a registered experiment raises `ExperimentNotRegisteredHere` (naming the path
and both location variables) and an ad-hoc one logs a WARNING. External controllers open an
already initialized compatible store with `migrate=False`. Pass the UI's `experiment_id`, the
complete task set and `registered=True` to `create_experiment`:
`declarations` contains `(task_id, attempt_number, channel_id)` tuples; `declared_tasks` is the
number of unique tasks and `declared_attempts` is the repetitions per task.

The controller resolves the registration, validates the pin and task set, declares attempts and
binds the registration to that store. Continue through the existing register/bootstrap, claim/bind,
invoke, terminalize and evidence-drain lifecycle. Merely inserting experiment labels after sending
HTTP turns loses attempt provenance. Read the controller and the app's collector before adapting
this path; channel IDs, claims and attempt fencing are not interchangeable with UI labels.

For a server-backed run, examine the bound attempt's `runtime_snapshot` to see what actually ran.
It can contain feature settings, workflow/skill fingerprints, model version, deadlines, output
limits and observability regime. A null snapshot means it was not recorded, not that the intended
settings were used. `/probes/readyz?runtime=true` describes the current server; it is not historical
proof of a prior attempt's configuration. For measured external runs, pruning suppression must be
set in the server environment before startup (`FW_OBS_SUPPRESS_PRUNE=1`).

## Optional reviewable experiment setup

When a project uses pre-run review, create a `fastworkflow-experiment-setup/1` spec via
`ExperimentSetups(workflow_folderpath).save(spec, expected_revision=0)` for a new setup. The
required spec fields are:

| Fields | Content |
|---|---|
| `experiment_id` | Nonempty simple identifier (up to 120 characters); same name as the studio experiment |
| `description` | Optional free-text string; empty if the author wrote nothing |
| `configuration`, `model_routes`, `budgets` | Nonempty JSON objects describing the intended execution |
| `estimated_cost` | `currency`, `basis`, and nonnegative `low`/`high` with `high >= low` |
| `repetitions` | Positive integer |
| `tasks` | Nonempty list with unique `task_id`, nonempty `description`, `split` (`tuning` or `held_out`), `input`, and nonempty `expected_outcomes` |
| Each expected outcome | Nonempty `description` and `evidence` strings |

Use benchmark task IDs in these setup tasks. The setup is a review record separate from the
benchmark catalog; it does not automatically resolve benchmark pins or drive the runner. Keep
input/configuration consistent explicitly.

Saving edits requires the revision read and creates a new revision needing review. Decisions
(`approved` or `changes_requested`) name an exact revision and digest. Use
`approved(experiment_id, revision, digest)` to retrieve an exact current review receipt for a driver
that supports this check. The reviewer name is self-declared; a receipt is not authenticated
execution authority and does not prove the runner used the settings. Do not record the owner's
approval on their behalf or treat a review receipt as permission for additional paid calls.

## Local UI API equivalents

The browser supplies the local session authentication; use that same authenticated session for
programmatic UI requests. URL-encode IDs. These are `run_chatbot` APIs, not `/invoke_agent` APIs:

| Action | Method and route |
|---|---|
| Save simple benchmark | `POST /api/benchmark-setup` with the `save_benchmark` body |
| Register experiment | `POST /api/benchmarks/<benchmark_id>/experiments` with `{"version":"v1"}` |
| Read registration and pinned tasks | `GET /api/benchmark-experiments/<experiment_id>` |
| Read version | `GET /api/benchmarks/<benchmark_id>/versions/<version>` |
| Read/write benchmark analysis | `GET` / `PUT /api/benchmarks/<benchmark_id>/analysis`; PUT body `{"analysis": value}` |
| Read/write experiment notes | `GET /api/experiment/<experiment_id>` / `PATCH /api/experiment/<experiment_id>` with `{"notes": "..."}` |

Execution reads/writes go to the workflow's one live database, so a registered experiment's
evidence is visible only once it is recorded there. For workspace reads use the appropriate
`store_id`; workspace writes are refused. Do not confuse a registration with a recorded experiment:
a registration can exist before there are attempts or conversations to show.

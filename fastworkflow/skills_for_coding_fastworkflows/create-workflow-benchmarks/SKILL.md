---
name: create-workflow-benchmarks
description: >-
  Create and revise fastWorkflow benchmarks and their tasks through run_chatbot or Python,
  register experiments pinned to a benchmark version, and hand those identities to a runner.
  Use when adding regression cases, authoring a workflow test corpus, or connecting a harness
  to the benchmark and experiment UI.
---

# Create workflow benchmarks and tasks

A benchmark defines repeatable work; its tasks have stable IDs. An experiment records attempts
against a specific benchmark version. Creating either in the UI does not execute the workflow.

## Author in run_chatbot

Start `fastworkflow run_chatbot`, select the application workflow, then open **Benchmark setup**.
Use the printed authenticated local URL. Selecting a workflow normally starts its server; reuse
an existing session when only authoring or reviewing. Follow the project's runtime permissions.

1. Choose **New benchmark**, enter a title and optional description, and add task rows.
   Each task's prompt is optional: a runner can supply the actual messages later.
2. **Save benchmark** assigns the benchmark ID, `v1`, and new task IDs. Keep those identities
   when adding execution code; do not derive task IDs from conversation topics.
3. **Edit benchmark** saves a new immutable version. Existing task IDs are retained; new rows
   receive new IDs. Removing a row removes it only from the new version.
4. Select the intended version, then **New experiment**. Copy the generated experiment ID
   and task IDs to the driver. This creates a registration, initially without an evidence store.
5. Once the driver records execution, use **Open execution records** to inspect it.

The left navigation is **benchmark → experiment → conversation**. Ad-hoc conversations appear
under **ad-hoc conversations → UTC date → conversation**. Node clicks show details on the right
and expand/collapse the left tree. Turns and trace components are inspected on the right;
only the right pane shows the complete breadcrumb. Tasks are listed in benchmark/experiment
details, not as another required left-tree level. Sidebar filters and the old Experiments/
Benchmarks navigation buttons are no longer part of this flow.

## Author the same objects from Python

For the UI's simple title/prompt model, use the setup API:

```python
from fastworkflow.benchmark import setup as benchmark_setup

version = benchmark_setup.save_benchmark(workflow_folderpath, {
    "title": "Order support regressions",
    "description": "Lookup, ambiguity and recovery using disposable test orders",
    "tasks": [{"prompt": "Show the delivery status of my most recent order."}],
})
registration = benchmark_setup.create_experiment(
    workflow_folderpath, version["benchmark_id"], version["version"]
)
experiment_id = registration["experiment_id"]
task_ids = registration["task_ids"]
```

When editing, supply `benchmark_id`, `expected_version` from the version you read, and the
**complete desired task list**. Include existing `task_id` values for retained rows; omit IDs
for new rows. A stale version raises `BenchmarkSetupConflict`: reload and reconcile instead of
silently overwriting another edit. Existing descriptions and opaque payloads survive this editor;
new simple-editor tasks have empty descriptions and payloads.

For explicit IDs or richer driver data, use `fastworkflow.benchmark.catalog.write_version`:

```python
from fastworkflow.benchmark.catalog import write_version

version = write_version(workflow_folderpath, {
    "benchmark_id": "order-support",
    "version": "v1",
    "title": "Order support regressions",
    "description": "Delivery status with a fixed disposable fixture",
    "tasks": [{
        "task_id": "delivery-status",
        "prompt": "Where is my order?",
        "description": "Resolve the customer's order and report the recorded delivery status",
        "payload": {
            "fixture": "single-test-order",
            "expected_outcomes": ["Answer agrees with the fixture's delivery status"],
        },
    }],
})
```

`fixture` and `expected_outcomes` above are **example driver conventions**, not framework fields
or automatic graders. Use the conventions implemented by the application's runner.

## Storage and identity contract

| Object | Location / contract |
|---|---|
| Benchmark version | `<workflow>/benchmarks/<benchmark_id>/vN.json`; schema `fastworkflow-benchmark/1` |
| Benchmark analysis | Sibling `analysis.json`; mutable, outside the version digest |
| UI experiment registration | `experiment_registrations` table of the live `observability.sqlite3`; pin, task IDs, and bound/deleted state |
| Execution evidence | `observability.sqlite3`; experiments, attempts, conversations, turns, spans and annotations |
| Pre-run review, if used | `<workflow>/experiment_setups/reviews.sqlite3`; configuration revisions and review decisions |

The manifest contains `schema`, `benchmark_id`, `version`, `description`, `tasks`, and optional
`title`. Each task has a unique nonempty `task_id`, `description` (default empty), a required
object `payload`, and optional text `prompt`. Tasks must be nonempty as a collection; a prompt
may be empty. JSON values must be native and finite; convert dataclasses explicitly.

Use `load_version`, `list_benchmarks`, and `list_versions` to read the catalog. `load_version`
returns `digest_sha256`, computed from the file bytes. Never edit or reformat a published version;
write a new version instead. Benchmark task definitions stay in files, not evidence tables.

The experiment pin is the complete triple `benchmark_id`, `benchmark_version`,
`benchmark_digest_sha256`. Editing a benchmark never changes an existing experiment's pin.
Keep unchanged task IDs across versions for traceability, while reporting any changed task input
when comparing results. Do not silently replace a pinned corpus with the latest version.

## Connect a runner

Read [runner-reference.md](runner-reference.md) when implementing execution or pre-run review.
The driver supplies messages, fixtures, an outcome grader and any ask-user response policy;
FastWorkflow does not interpret arbitrary `payload` keys or run a task just because it has a prompt.
The controller validates the registered task set and pin and binds the registration to its real
store when declaring attempts. One registration cannot be rebound to a different evidence store.

Use the app's existing HTTP collector when that is its measured execution path; use
`ExperimentHarness` when in-process execution is appropriate. Do not switch transports merely
to fit an example. Establish execution authorization, budget and stopping conditions before a run;
creating a registration is neither a launch nor an approval.

## Close the improvement loop

Use [debug-workflow-conversations](../debug-workflow-conversations/SKILL.md) to inspect the failing
trace and [optimize-workflow-with-feedback](../optimize-workflow-with-feedback/SKILL.md) to turn
observations into a controlled comparison. For complex multi-turn test design, read
[build-task-benchmarks](../build-task-benchmarks/SKILL.md); a small regression need not become a
20–100-step suite.

Record run conclusions in experiment notes; record cross-experiment conclusions in benchmark
analysis (`write_analysis`). Benchmark analysis accepts JSON-native values, including plain text
from the UI. Keep original evidence, human comments, description and notes in their respective
fields; notes can cite them and summarize metrics but do not replace scoring or change recorded
outcomes.
Workspace snapshots are read-only: author and annotate in the working workflow/store.

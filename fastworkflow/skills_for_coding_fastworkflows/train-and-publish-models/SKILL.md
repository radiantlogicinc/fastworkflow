---
name: train-and-publish-models
description: >-
  Run, interpret and recover a `fastworkflow train` run, given that training is cached,
  incremental and versioned: deterministic utterance and parameter-example caches, per-command
  fingerprints that decide which contexts retrain, and an atomically published artifact version
  with one previous recovery point. Covers what invalidates a cache, what forces a full retrain
  instead of a selective one, what a failed run leaves behind, how to roll back, and which files
  must never be hand-deleted. Use before or after running `fastworkflow train`, when a run
  regenerated more than expected, when it retrained everything unexpectedly, when it reports
  artifacts are already up to date, or when trained models need to be rolled back.
---

# Training and publishing models

## The command

```bash
fastworkflow train <workflow_folderpath> [env_file_path] [passwords_file_path] [--regenerate-utterances]
```

The two env-file arguments are positional and optional; they default to files in the current
directory, or to the bundled ones for a packaged example. `--regenerate-utterances` is the only
training-policy override, and it is expensive — see Rule 2.

Training recurses into `<workflow>/_workflows/*` first, so a child workflow is trained before its
parent, each with its own persona source and caches.

## What one run does, in order

Knowing the order is what lets you read a failure: everything before the first paid call is a
gate, and everything after publication is cleanup.

```
1. Build routing artifacts        command_directory.json, routing_definition.json
2. Duplicate-capability scan      lexical, seeds only          <- free, reports
3. Benchmark preflight            leak + defect checks         <- free, CAN FAIL THE RUN
4. Migrate legacy artifacts       into versions/ if unversioned
5. DSPy parameter examples        param_example_cache          <- costs money on a miss
6. Compute the training plan      selective or full
   -> if the plan is empty: republish, apply retention, stop ("already up to date")
7. Generate utterances + train    utterance_cache              <- costs money on a miss
8. Carry forward untrained contexts from the previous version
9. Router-confusion scan          needs every context present
10. Write signature + manifest
11. Safety gate on training data  CAN REFUSE TO PUBLISH
12. Publish + retain current and previous   <- the commit point
13. Prune orphaned artifacts
```

Steps 2 and 3 sit before every paid call deliberately: a leaked benchmark used to abort a run
that had already spent money on every command with parameters.

## Rule 1 — the seed is fixed, so a cache miss is the only source of drift

`get_training_seed()` returns **42**, always. It is not configurable, and a `TRAINING_SEED`
environment variable is **not read** — seed selection is part of the trainer, not workflow
configuration.

That matters because seeding alone does not make training reproducible. The utterance generator
is a live LLM: two runs at the same seed were measured producing different utterances *and*
different row counts for every command. Reproducibility comes from the **caches**, not the seed.

So the practical rule is: with caches warm, two runs train on identical data and produce
byte-identical models. The moment something misses, that guarantee is gone for the commands that
missed.

## Rule 2 — know what invalidates a cache before you blame the cache

Two caches, both under `___command_info/`, both keyed on a fingerprint of every input that could
change what the LLM produces:

| Cache | Holds | Directory |
|---|---|---|
| utterance cache | the synthetic utterances one command was trained on | `utterance_cache/` |
| parameter-example cache | the DSPy few-shot examples for one command's parameters | `param_example_cache/` |

A miss is **reported at INFO, naming the input that moved** — up to three of them, with before
and after values. Read that line instead of hand-diffing JSON. The inputs include the command
name, its seed utterances, the persona counts, the generation model and API base, the persona
source, and a digest of the generation prompt's own source code. Full list in
[reference.md](reference.md).

The failure mode all of this exists to prevent is a developer editing a command's seed utterances
and silently training on stale generated data. The cost direction is deliberate: a lost cache
entry produces a slow, non-reproducible run — never a wrong one.

**`--regenerate-utterances` ignores both caches.** It calls the LLM again for everything, costs
money, and breaks reproducibility against previous runs. Use it after you suspect the cache is
wrong, not as a habit — a seed-utterance edit already invalidates exactly the commands it
touched, which is both cheaper and more precise. It also forces a **full** retrain (Rule 3).

## Rule 3 — selective is the default; a global change is what forces full

`fastworkflow train` retrains only what it must. A command is dirty if its **source bytes** or
its **seed utterances** changed; dirty commands are closed upward over the `base` and `parent`
axes to the contexts that must retrain; the rest are carried forward from the previous version.

The design bias is stated in the code and worth internalising: *every path that cannot answer
"this context is provably unchanged" retrains.* A false skip means a silently stale model, so
uncertainty always resolves toward doing the work.

A **full** retrain is forced when any of these is true:

- there is no published version to carry forward from, or no usable previous signature;
- `--regenerate-utterances` was passed;
- any **global header** input changed — the artifact format version, the seed, the tiny or large
  intent model, the synthetic-data model, **the persona source**, or a source digest of the
  generator, the trainer, or the class-balance module;
- commands shared between retrained and carried-forward contexts have no cache entries, which
  would mix cached and freshly generated utterances in one model.

Two consequences people trip over:

- **Adding or editing `personas.json` retrains everything and regenerates everything.** It is a
  global input on both keys. See the `supply-training-personas` skill.
- **Upgrading the framework can force a full retrain**, because the generator and trainer module
  source digests are global inputs. That is intended: their behaviour decides the models.

There are no `--changed-only` / `--only-contexts` CLI flags. Those parameters exist on
`compute_training_plan` and are deliberately not exposed.

## Rule 4 — only a successful run becomes current

Model artifacts live in `___command_info/versions/<version_id>/`. A pointer file `current.json`
says which version is live; a `current` symlink and one compatibility entry per context point at
it so every existing reader keeps working unchanged.

`publish_version` writes `current.json` **last**, after preparing every reader path. That makes
it the commit point:

- A run that **fails** — at training, at carry-forward, or at the training-data safety gate —
  never publishes. The previous version stays current and complete, with the partial version
  beside it rather than on top of it.
- Pruning runs only **after** a successful publish, so a failed run leaves the previous
  `___command_info` intact and runnable.
- Retention keeps the **current version and one previous**. That previous version is your
  recovery point, and it is the only one.

Because retention is that shallow, do not run training twice to "get back" to a known-good set.
The second run discards it.

## Rule 5 — never hand-delete `___command_info`

It is not a build directory. In one place it holds:

- the trained models, at several gigabytes for a large workflow and hours of LLM calls to rebuild;
- the **only** rollback point;
- both caches, which are the only thing making two runs train on the same data;
- the selective-training baseline, whose loss forces a full retrain.

A previous incident destroyed a complete trained set because nothing said so; the `README.md`
files inside `versions/` and `utterance_cache/` exist because of it.

If you genuinely need to force work to happen, prefer the narrow instrument:

| Want | Do | Not |
|---|---|---|
| regenerate one command's utterances | edit its seed utterances | delete the cache |
| regenerate everything | `--regenerate-utterances` | `rm -rf ___command_info` |
| retrain one context | touch a command in it | delete its model folder |
| go back to the previous models | republish the previous version (Rule 6) | retrain and hope |

The trainer's own pruning is conservative by comparison: it removes an artifact only when the
command or context it belongs to no longer exists, and it never touches `versions/`, `current`,
or either cache.

## Rule 6 — rollback is a Python call, not a flag

There is no rollback CLI. The procedure:

```python
from fastworkflow.train import artifact_versioning

for version in artifact_versioning.list_versions(workflow_folderpath):   # newest first
    print(version)

artifact_versioning.publish_version(workflow_folderpath, "<older version id>")
```

Republishing rewires the compatibility entries and rewrites `current.json`. It also restores that
version's `training_signature.json`, so the selective-training baseline rolls back with the
models — the next run compares against the artifacts you actually restored, not the ones you
discarded.

Promotion is gated: `publish_version` re-derives the recorded retraining closure from the target
manifest alone and refuses to advance `current` if the manifest contradicts itself or the version
its carried-forward models came from.

## Reading the output

| Line | Means |
|---|---|
| `Training plan: full retrain of N context(s).` + reasons | something global moved; the reasons name it |
| a changed-command list, then contexts to retrain with per-context reasons | the selective path |
| `Carried forward N context(s) from version <id>: ...` | those contexts were **not** retrained this run |
| `Utterance cache (reuse) at <path>: X reused, Y generated, Z written` | `Y` is what this run paid for |
| `Training artifacts are already up to date.` | the plan was empty; artifacts were republished and retention applied, nothing retrained |
| `MODEL CONFUSION (N)` | diagnostic only, after publication — see `detect-duplicate-capabilities` |
| `Training complete.` | published; `current.json` now names this version |

`Training artifacts are already up to date.` is a success, not a no-op skip. It still repairs
reader paths and applies retention, which is what makes it safe to re-run training after a
partial failure.

## Pre-flight checklist

```
- [ ] Seed-utterance and command-source edits are complete — both invalidate caches
- [ ] Persona changes are batched with other full-retrain-forcing changes, not sprinkled
- [ ] intent_benchmark.json is disjoint from seeds (the preflight will fail the run otherwise)
- [ ] Time budgeted for a full retrain if any global input moved
- [ ] The previous version is the one you want as a recovery point
- [ ] --regenerate-utterances is being passed for a stated reason, not by habit
```

## Additional resources

- Exact cache fingerprint inputs, the global-header input list, the on-disk layout under
  `___command_info`, and a failure-mode catalogue: [reference.md](reference.md).

## Related

- Interpreting the numbers a run reports at the end: the `evaluate-intent-routing` skill.
- The persona configuration that forces a full retrain: the `supply-training-personas` skill.
- The two scans a run prints: the `detect-duplicate-capabilities` skill.

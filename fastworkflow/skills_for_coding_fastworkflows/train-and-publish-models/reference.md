# Reference: cache keys, layout and failure modes

Companion to [SKILL.md](SKILL.md). Read when a specific input, path or symptom is needed.

## Utterance cache fingerprint inputs

Every input below is hashed into a 24-character `variant_key`. Change any one of them and the
command regenerates. The miss report names up to three that moved.

| Input | Moves when |
|---|---|
| `cache_format_version` | the cache format itself changes |
| `command_name` | the command is renamed or moved between contexts |
| `seed_utterances` | any seed is edited, added, removed or **reordered** |
| `num_personas` | `SYNTHETIC_UTTERANCE_GEN_NUMOF_PERSONAS` changes |
| `utterances_per_persona` | the corresponding setting changes |
| `personas_per_batch` | the corresponding setting changes |
| `model` | `LLM_SYNDATA_GEN` changes |
| `api_base_digest` | the same model is pointed at a different proxy (hashed, since it can name an internal host) |
| `persona_source` | a `personas.json` is added, edited or removed, **or** the framework's persona-pool code changes |
| `completion_backend` | a caller injects its own completion function |
| `generator_source_digest` | the generation prompt's source text changes — including its comments |

The persona component is itself three parts, `<name>#<content fingerprint>#<pool-code digest>`,
so a miss report naming `persona_source` can be read further: a changed third component with the
first two intact says the framework's persona code was edited, not your persona file.

`generator_source_digest` records `source-unavailable` when `inspect.getsource` cannot read the
module (zipapp, frozen build). Reuse still works; it just stops noticing prompt edits.

**Retention:** three variants per command survive a write, so flipping a prompt edit back and
forth, or training the same workflow under two model strings, costs regeneration only once.

**Files:** `___command_info/utterance_cache/<slug>.<variant_key>.json`, plus a `README.md`
warning. The `(variant_key, seed)` pair addresses one entry: the variant key covers everything
except the seed, and the seed selects an entry inside the file.

## Training-signature global inputs

Any difference here forces a **full** retrain. Recorded per version in
`versions/<id>/training_signature.json`.

| Input | Source |
|---|---|
| `format_version` | the artifact format constant |
| `seed` | `get_training_seed()` — fixed at 42 |
| `tiny_model` | `INTENT_DETECTION_TINY_MODEL` (default `google/bert_uncased_L-4_H-128_A-2`) |
| `large_model` | `INTENT_DETECTION_LARGE_MODEL` (default `distilbert-base-uncased`) |
| `syndata_model` | `LLM_SYNDATA_GEN` |
| `persona_source` | the active persona source label |
| `generator_source_digest` | source of the synthetic-generation module |
| `trainer_source_digest` | source of the model-pipeline-training module |
| `class_balance_source_digest` | source of the class-balance module |
| `parameter_value_placeholders_sha256` | the reserved parameter-value placeholder set |

`INTENT_DETECTION_TINY_MODEL` and `INTENT_DETECTION_LARGE_MODEL` are read with code defaults, so a
shell `export` of either is **silently ignored** — set them in the workflow's `fastworkflow.env`
or they will not take effect, and the retrain you expected will not happen.

Per-command staleness is decided by two hashes: the command's **source file bytes** and its
**seed utterances**. A command whose metadata cannot be hydrated is always treated as dirty.

Per-context signature covers the context's `label_space`, `ancestors`, `wildcard_sources` and
whether it expects a wildcard label — so a context retrains when its label space moves even if
none of its own commands changed. A context missing any of its six required artifacts is
retrained rather than carried forward.

## Layout under `___command_info`

```
<workflow>/___command_info/
    command_directory.json          # NOT versioned — rewritten whenever a command's mtime moves
    routing_definition.json         # NOT versioned — same
    <command>_param_labeled.json    # NOT versioned — read at runtime from the top level
    training_provenance.json        # this run's provenance (a copy is stamped into the version)
    heldout_evaluation.json         # this run's evaluation report
    duplicate_capabilities.json     # this run's duplicate scan
    command_fingerprints.json       # ephemeral selective-training input
    utterance_cache/                # shared across versions, never pruned as an orphan
    param_example_cache/            # same
    current.json                    # AUTHORITATIVE pointer to the live version
    current       -> versions/<id>/            # convenience symlink, best effort
    <Context>     -> versions/<id>/<Context>/  # compatibility entry per context
    global        -> versions/<id>/global/     # the "*" context's folder is named `global`
    versions/
        README.md
        <version_id>/                          # e.g. 20260802T144233Z-a1b2c3
            manifest.json
            training_signature.json
            training_provenance.json
            global/{tinymodel.pth, largemodel.pth, threshold.json, ...}
            <Context>/{tinymodel.pth, largemodel.pth, threshold.json,
                       tiny_ambiguous_threshold.json, large_ambiguous_threshold.json,
                       label_encoder.pkl}
```

Only the **per-context model directories** belong to a version. The JSON snapshots are build
artifacts rewritten by merely importing a workflow, so versioning them would manufacture versions
on import.

Compatibility entries point **directly** at `versions/<id>/<Context>`, not through `current`, so
losing the `current` symlink cannot break every context at once and `os.path.realpath` on any
context entry names its version in one hop. Where symlinks are unavailable the entries are
materialised as hardlink farms carrying a `.fastworkflow_compat` marker.

Version ids sort lexicographically into chronological order; `list_versions` orders by the
manifest's `created_at` with the id as a tiebreaker, because two runs can start in the same
second.

`versions`, `current`, `utterance_cache`, `param_example_cache` and `__pycache__` are reserved
top-level names, skipped by publication's stale-entry sweep, by legacy migration, and by orphan
pruning.

## Failure-mode catalogue

### The run regenerated everything and charged for it

**Cause.** A global input moved. The commonest are a new or edited `personas.json`, a changed
`LLM_SYNDATA_GEN`, and a framework upgrade that changed the generator or trainer source digest.

**Diagnose.** Read the plan output — a full retrain prints its reasons — and the per-command miss
lines, which name the fingerprint input that moved.

### One command regenerated and nothing else did

**Cause.** Its seed utterances or its source file changed. Reordering seeds counts: the list is
hashed in order.

**Not a bug.** This is the cache working precisely.

### A context shows old evaluation numbers after a selective run

**Cause.** It was carried forward, not retrained. Its entry in `heldout_evaluation.json` is
re-inserted from the previous report and tagged `carried_forward` with the version it came from.
Those numbers still describe the models this version ships; they were just not measured on this
run.

**Diagnose.** Cross-check the `Carried forward N context(s)` line and the manifest's
`contexts_carried_forward`.

### The run said "already up to date" but something is visibly wrong at runtime

**Cause.** Model-context fingerprints do not cover a deleted or corrupt
`<command>_param_labeled.json`, which is why parameter-example refresh runs *before* the no-op
check. If intent routing is right but parameter extraction is wrong, that file is the suspect.

**Fix.** `--regenerate-utterances` also bypasses the parameter-example cache.

### Training failed partway; what state is the workflow in

**Answer.** The previous version is still current and complete. The partial version sits beside
it under `versions/` and will be pruned by retention on the next successful publish. Nothing was
un-trained. Re-running training is safe and will re-plan from the still-current version.

### Publication refused with a training-data or consistency error

**Cause.** Either the training-data safety gate found structurally incomplete data, or the
version's manifest contradicted itself or the version its carried-forward models came from.

**Meaning.** Deliberate. Models were trained but are not published, so the live workflow is
unchanged. Fix the reported commands and re-run; do not force-publish the version by hand.

### Rolling back did not change behaviour

**Check.** `current.json` is authoritative — read it rather than trusting the `current` symlink,
which is best-effort and skipped entirely on filesystems without symlink support. Also confirm
the target version still exists: a pointer naming a version that is gone is treated as absent,
and retention keeps only current plus one previous.

## Cost and time notes

- One version of a large workflow is roughly 276 MB per context, several gigabytes in total.
- Rebuilding one version costs hours of LLM calls plus fine-tuning time, and the utterances are
  not reproducible byte-for-byte across runs once regenerated.
- With both caches warm, two independent full retrains of a 32-context workflow produced
  byte-identical models. That is the property the caches exist to provide, and the reason a
  single paired build is interpretable — see the `evaluate-intent-routing` skill.

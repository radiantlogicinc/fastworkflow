# One Live Evidence DB per Workflow — Target Layout and Invariants (Design)

> **Status: APPROVED (2026-10-04, owner Dhar Rawal)** (beads `fix-10vj.1`, epic
> `fix-10vj`). The owner's resolutions are recorded in §11.
> Binding inputs: the eight owner decisions of 2026-10-04 recorded on the epic.
> This doc specifies the NEW layout only. Nothing here reads, moves, adopts or
> migrates any of the retired files (decision 3); the owner retires them by hand
> (`fix-10vj.7`). Where a child issue's text and this doc disagree, this doc
> wins and the disagreement is listed under §10 "Scope changes".
> Author: Claude (with Dhar Rawal), 2026-10-04.

## 0. Goal

Each workflow has exactly one live evidence DB:
`<state_root>/workflows/<workflow_id>/observability.sqlite3` (schema v8 since
`fix-0gh0` dropped the capture-policy columns; v7 when this design was approved). It
holds evidence, and it also holds every judgement made about that evidence:
registrations, contests, winners and pair-review marks.
A few things stay outside it:

- authored benchmark files;
- logs;
- the per-channel runtime blobs that already existed before this epic;
- sealed archives, which are immutable copies made from the live DB.

Non-goals:

- `ReviewSidecar` (blinded review, `fastworkflow/review/sidecar.py`) is out of scope.
- Workspace-viewer reading of sealed archives is kept.
- No `SCHEMA_VERSION` bump.

> **Update (`fix-0gh0`):** the first two non-goals no longer hold. The blinded
> review sidecar (`fastworkflow/review/`) and the workspace viewer
> (`--workspace-manifest`, `/api/workspace/*`, `observability/workspace.py`)
> were removed outright, and with the viewer went `sealed_turn_comments`
> (§2.5). Experiment sealing is kept: `runner.seal_workspace_evidence`,
> `store.record_workspace_archive`, `sealed_archives`,
> `experiments.workspace_archive_sha256`, `ReadOnlyObservabilityStore` and
> `selection.store_for`, which reads a member's sealed archive in live mode.

Principles applied to every decision below:

- No regression.
- No new I/O on a GET.
- Delete machinery rather than adapt it.
- One small module owns the control tables.

## 1. Runtime file list

### 1.1 Per workflow, under `<state_root>/workflows/<workflow_id>/`

| Path | Written by | Purpose |
|---|---|---|
| `observability.sqlite3` (+ `-wal`, `-shm`) | every recorder; control writers | evidence (v8 tables) + control tables (§2) |
| `server.log`, `server.log.1` | `run_chatbot` launcher | stdio of the spawned FastAPI server, rotated |
| `chatbot_train.log` | `run_chatbot` launcher | stdio of the detached `fastworkflow train` child |
| `session_state/` | `run_fastapi_mcp` | suspended `awaiting_user` trajectories, keyed by channel |
| `checkpoints/` | `run_fastapi_mcp` | resume checkpoints, keyed by deployment / fingerprint / channel |
| `function_cache/` | `@enablecache` | per-function app cache; only exists if a workflow uses it |

Rationale per row:

- **`chatbot_train.log` stays a file.** A detached child's stdio must be a file descriptor. Sending it into `server.log` would interleave two processes and let one process's rotation cut the other's output.
- **`session_state/`, `checkpoints/` and `function_cache/` are unchanged.** They are transport and runtime blobs with their own per-channel lifecycle, not evidence or judgements, and the epic does not list them.
- **Training artifacts stay in the workflow's `___command_info/`.** Unchanged.

**Never created again:**

- `selection.control.sqlite3`, `pairreview.control.sqlite3`
- `*.selection.sqlite3`, `*.pairreview.sqlite3`, `*.feedback.sqlite3`
- `*.offload-handles.sqlite3`
- `selection.sources.json` / `.lock`
- `consistency-vectors/`
- `chatbot_train.pid`
- `conversations/`

`backups/` is not created by any code; the owner made it by hand (`fix-10vj.7`).

### 1.2 Per project, under `<project>/benchmarks/`

| Path | Kind | Verdict |
|---|---|---|
| `<benchmark_id>/vN.json` | authored manifest, immutable | stays |
| `<benchmark_id>/analysis.json` | **authored** | stays |

- **`analysis.json` is AUTHORED, not derived.** Its only writer is `catalog.write_analysis`, reached through `PUT /api/benchmarks/<id>/analysis` from the analysis textarea (`170-comparison-detail.js`, "Analysis saved"). Nothing computes it.
- **`.experiments/` (including `.deleted/`) goes.** It moves to the `experiment_registrations` table (§2).
- **`.setup.lock` goes.** `save_benchmark` no longer locks. `catalog.write_version` already creates `vN.json` with an exclusive `open(..., "xb")`, so two concurrent saves of the same next version cannot both win. The loser's `BenchmarkAlreadyExistsError` is mapped to `BenchmarkSetupConflict` (409), which is exactly what the `expected_version` check returned before.

## 2. Control tables in the live DB

### 2.1 Placement, marker, creation

- **New module `fastworkflow/observability/control.py`, about 180 lines.** It holds `CONTROL_SCHEMA` (the DDL below), `CONTROL_TABLES` (the table names), `FEATURE_CONTROL_V1 = "control_v1"`, and four helpers:
  - `ensure(conn)`
  - `write(store)` — a `BEGIN IMMEDIATE` context manager
  - `present(store)` — reports whether the marker is recorded
  - the sealed-comment merge reader (§2.5)

  It imports nothing from `store.py`, so `store.py` can import it without a cycle.
- **One marker for every control table, landed in one commit (the first commit of `.3`).** A feature marker means a fixed set of tables. Adding a table later would need a second marker.
- **Additive, with no `SCHEMA_VERSION` bump (decision 5).** This follows the existing convention for `FEATURE_*` markers. `user_version` stays 7.

**When tables are created** (the only two places):

1. A writable open with `migrate=True` (`_ensure_schema_once`), in the same transaction that records the other feature markers. This covers recording processes: the server, the harness, `train`.
2. A control writer whose store was opened with `migrate=False` and finds the marker missing calls `control.ensure(conn)` inside its own `BEGIN IMMEDIATE`, before its first insert.

**Reconciling with "READING never creates anything":**

- Every GET reads through `ReadOnlyObservabilityStore` (`mode=ro`).
- If the DB file is absent, or `control_v1` is not in `schema_features`, every control read returns the empty answer: no registrations, no groups, no marks. It does not raise, create or write.
- Control **writers** open with `migrate=False`, so a click in the UI never rewrites the capture-regime diagnostic or the `schema_opened` probe.
- Control writers refuse if the live DB file does not exist. The one exception is the training row (§2.7), because a POST that starts training creates the live DB anyway when `train` records its run.

`state_paths.workflow_state_dir`'s `makedirs` behaviour is unchanged.

### 2.2 Registrations (deletion is a row state)

```sql
CREATE TABLE IF NOT EXISTS experiment_registrations (
    experiment_id TEXT PRIMARY KEY,
    benchmark_id TEXT NOT NULL,
    benchmark_version TEXT NOT NULL,
    benchmark_digest_sha256 TEXT NOT NULL,
    description TEXT NOT NULL,
    task_ids_json TEXT NOT NULL,
    runs_per_task INTEGER NOT NULL,
    source_experiment_id TEXT,
    changed_fields_json TEXT NOT NULL DEFAULT '[]',
    state TEXT NOT NULL DEFAULT 'registered'
        CHECK (state IN ('registered', 'bound', 'deleted')),
    created_at TEXT NOT NULL,
    bound_at TEXT,
    deleted_at TEXT);
CREATE INDEX IF NOT EXISTS idx_registrations_benchmark
    ON experiment_registrations(benchmark_id, state);
```

- **`state` replaces the record's `store` field and the `.deleted/` folder.**
  - `bound` means the runner has started recording.
  - `deleted` is the tombstone. `load_experiment` raises `ExperimentDeleted` on it.
- **The JSON record served to the UI keeps its key set.**
  - `store` becomes `{"store_id": <live identity>}` once the registration is bound, else `null`.
  - The `db_path` key is gone, so no path ever crosses the API (decision 6).
- **Registering joins the contest in the same transaction (§2.3).** Deleting a registration retires its member in the same transaction as well. The multi-file compensation logic in `_delete_locked` is therefore deleted, not ported.

### 2.3 Selection: groups, members, winner pointers, decision history

These tables are copied from `selection.py` unchanged, except that `source_id` is dropped from members.

```sql
CREATE TABLE IF NOT EXISTS comparison_groups (
    group_id TEXT PRIMARY KEY, group_kind TEXT NOT NULL,
    workflow_name TEXT, benchmark_id TEXT,
    created_at TEXT NOT NULL, created_from_experiment_id TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS comparison_group_members (
    group_id TEXT NOT NULL, experiment_id TEXT NOT NULL,
    benchmark_version TEXT, benchmark_digest_sha256 TEXT,
    created_at TEXT, joined_at TEXT NOT NULL,
    PRIMARY KEY (group_id, experiment_id));
CREATE UNIQUE INDEX IF NOT EXISTS idx_group_member_experiment
    ON comparison_group_members(experiment_id);
CREATE TABLE IF NOT EXISTS selection_pointers (
    scope_kind TEXT NOT NULL, group_id TEXT NOT NULL, scope_key TEXT NOT NULL,
    experiment_id TEXT NOT NULL, task_id TEXT, attempt INTEGER,
    selection_id TEXT NOT NULL, decision TEXT NOT NULL,
    decision_seq INTEGER NOT NULL, decided_at TEXT NOT NULL,
    PRIMARY KEY (scope_kind, group_id, scope_key));
CREATE TABLE IF NOT EXISTS selection_decisions (
    decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
    scope_kind TEXT NOT NULL, group_id TEXT NOT NULL, scope_key TEXT NOT NULL,
    seq INTEGER NOT NULL, decision TEXT NOT NULL,
    previous_experiment_id TEXT, previous_task_id TEXT,
    previous_attempt INTEGER, previous_selection_id TEXT,
    candidate_experiment_id TEXT, candidate_task_id TEXT, candidate_attempt INTEGER,
    new_experiment_id TEXT, new_task_id TEXT, new_attempt INTEGER,
    new_selection_id TEXT,
    actor TEXT NOT NULL, actor_kind TEXT NOT NULL, provenance TEXT NOT NULL,
    rationale TEXT, created_at TEXT NOT NULL,
    UNIQUE (scope_kind, group_id, scope_key, seq));
CREATE INDEX IF NOT EXISTS idx_selection_decisions_scope
    ON selection_decisions(scope_kind, group_id, scope_key, seq);
CREATE TRIGGER IF NOT EXISTS selection_decisions_no_update
    BEFORE UPDATE ON selection_decisions
    BEGIN SELECT RAISE(ABORT, 'decision history is append-only'); END;
CREATE TRIGGER IF NOT EXISTS selection_decisions_no_delete
    BEFORE DELETE ON selection_decisions
    BEGIN SELECT RAISE(ABORT, 'decision history is append-only'); END;
```

- **Group identity is unchanged.** `comparison_group_identity` is a deterministic sha256 over (kind, workflow_name[, benchmark_id]). The decision constants (`initial`, `promote`, `keep`, `undecided`, `retire`) and scopes (`experiment`, `task_best`) are also unchanged.
- **Membership is explicit.** An experiment joins when it is registered or when `create_experiment(initialize_winner=True)` records it, in the same transaction.
  - The first member of a group is elected `initial`, in the same transaction as the join. "Oldest member" now needs no cross-store view.
  - **Enrolment by explicit request.** An experiment already in the live DB's `experiments` table from before this build joins its group only when a decide, keep or promote request from the UI or API names it. It joins through the same path and election rules as any member, inside that request's `BEGIN IMMEDIATE`. It reuses the existing join path, so no new mechanism is added.
  - **No automatic adoption or bootstrap.** A GET never enrols, and nothing scans existing history and adopts it. `.3` deletes the "automatic first experiment" adoption.

### 2.4 Pair-review marks

Copied verbatim from `pair_review.py`.

```sql
CREATE TABLE IF NOT EXISTS pair_reviews (
    pair_key TEXT NOT NULL, reviewer TEXT NOT NULL, reviewer_kind TEXT NOT NULL,
    left_ref_id TEXT NOT NULL, right_ref_id TEXT NOT NULL,
    left_ref_json TEXT NOT NULL, right_ref_json TEXT NOT NULL,
    state TEXT NOT NULL, seq INTEGER NOT NULL, note TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (pair_key, reviewer));
CREATE INDEX IF NOT EXISTS idx_pair_reviews_left
    ON pair_reviews(left_ref_id, reviewer);
CREATE TABLE IF NOT EXISTS pair_review_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    pair_key TEXT NOT NULL, reviewer TEXT NOT NULL, reviewer_kind TEXT NOT NULL,
    seq INTEGER NOT NULL, previous_state TEXT, state TEXT NOT NULL,
    note TEXT, created_at TEXT NOT NULL,
    UNIQUE (pair_key, reviewer, seq));
```

- **Pair keys stay stable across sealing.** `ExecutionRef.store_id` is always the live store's identity, and sealed copies carry that same identity.

### 2.5 Comments on sealed-archive turns (removed)

Removed with the workspace viewer (`fix-0gh0`). The viewer was the only way to
read a turn from a sealed archive, so it was the only writer of the
`sealed_turn_comments` table this section specified. A comment on any turn the
chatbot reads goes to the live DB's `human_feedback` (decision 7), including a
turn of a sealed member, whose rows stay in the live DB until its evidence is
released.

### 2.6 Known sealed archives

```sql
CREATE TABLE IF NOT EXISTS sealed_archives (
    experiment_id TEXT PRIMARY KEY,
    archive_sha256 TEXT NOT NULL UNIQUE,
    path TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    sealed_at TEXT NOT NULL);
```

- **`store.record_workspace_archive` inserts the row in the same transaction** that already writes `experiments.workspace_archive_sha256` and `evidence_sealed_at`. Nothing else writes it, so no user-supplied path is ever trusted.
- **This replaces `selection.sources.json`.** With one live DB, the only "other source" a contest can have is its own members' seals.
- **The store for an experiment is resolved by `store_for(experiment_id)`:**
  - If the experiment has a `sealed_archives` row whose file exists with the recorded size, the answer is the `mode=ro` archive.
  - Otherwise the answer is the live DB.

  Sealed evidence is the frozen truth, and preferring it keeps a sealed member readable after live pruning releases its spans (§7). The sha is verified at seal time, not on every GET, to avoid hashing a file per request.

### 2.7 The training-process row

```sql
CREATE TABLE IF NOT EXISTS training_process (
    slot INTEGER PRIMARY KEY CHECK (slot = 1),
    pid INTEGER NOT NULL,
    proc_start_ticks INTEGER NOT NULL,
    server_incarnation TEXT,
    log_path TEXT NOT NULL,
    started_at TEXT NOT NULL);
```

- **One row per workflow.** The launcher writes it right after `Popen` and deletes it when it observes the child gone.
- **Liveness check is unchanged.** It is pid plus `/proc/<pid>/stat` start ticks, the same proc start time `_read_pid_record` checked before.
- **`find_live_train` opens each `workflows/*/observability.sqlite3` read-only.** A missing file or table means no training process.

### 2.8 Experiment-evidence release (for §7)

```sql
CREATE TABLE IF NOT EXISTS evidence_releases (
    experiment_id TEXT PRIMARY KEY,
    released_at TEXT NOT NULL, actor TEXT NOT NULL, reason TEXT);
```

### 2.9 The consistency cache: no table

**Decision:** the `VectorCache` files are replaced by a process-wide in-memory LRU with the same `get(fingerprint, digest)` / `put` interface.

- It is capped at 2,048 entries.
- Vectors are stored as `array('f')`: 384 floats is about 1.5 KB, so the cap is about 3 MB.

Rationale:

- It is a cache: losing it on restart costs only re-embedding at most about 30 runs per view with MiniLM.
- A table would make GETs write to the live DB, contend with recorders, grow the DB, and need exclusion from seals.

### 2.10 Dropped

- `control_meta` and `evidence_sources` (both copies) are dropped.
- So are `CONTROL_SCHEMA_VERSION`, `PRIMARY_SOURCE_ID` and the single/shared modes. §4 explains why nothing replaces them.

## 3. Sealed copies (`archive_to`)

**Rule 1 — sealed copies contain no control table, ever.** Every table in `control.CONTROL_TABLES` is absent from the copy. The copy's `schema_features` row is rewritten without `control_v1`.

**Rule 2 — the digest is the sha256 of the sealed file's bytes.** It is fixed once, at seal time. Rule 1 means judgements live only in the live DB, so no later decision, mark or comment can change a sealed file or its digest. Comments on any turn go to `human_feedback` in the live DB (§2.5). Live-turn `human_feedback` written after the seal stays in the live DB.

**Two paths, one function:** `archive_to(destination, *, experiment_id=None, quiesce_live_writer=True)`.

- **Experiment seal (`experiment_id` given), new.**
  1. Create the destination's temp file with `_SCHEMA_STATEMENTS` only. The control DDL is never run there, so Rule 1 holds by construction.
  2. `ATTACH` the live DB as `file:...?mode=ro`.
  3. In ONE read transaction, `INSERT ... SELECT` the experiment's rows:
     - `experiments`, `experiment_attempts`, `experiment_attempt_declarations`, `experiment_attempt_claims` and `experiment_evidence_runs`, filtered by `experiment_id`;
     - `conversations`, `turns`, `spans` and `artifacts`, filtered by `experiment_id`, plus spans whose `trace_id` is a kept turn;
     - `human_feedback` and `offload_*`, filtered to kept turns;
     - `diagnostics`, except `schema_features` and `writer_health*`.
  4. Stamp `user_version=7` and features minus the control marker.
  5. `VACUUM INTO` the final file, then verify integrity, identity and the absence of sidecars, then chmod `0444`, as today.

  Rationale:
  - It copies kilobytes to megabytes rather than the whole live DB (about 630 MB on ido) on every seal.
  - One read transaction is a consistent snapshot, however many other processes are writing.
- **Whole-store archive (`experiment_id=None`), used by `archive.py` and operator archiving.** The existing backup, `VACUUM INTO` and verify path is kept, with one added step: drop the control tables from the backup temp file before `VACUUM INTO`.
  - It keeps the "refuse while another writer is alive" check. That check is correct for archiving a whole store, and wrong for the shared live DB's experiment seals, which no longer use this path.

**Seal preconditions:**

- These are unchanged in `runner.seal_workspace_evidence`: status `complete` or `capture_complete`, no open in-process sink, and `begin_workspace_seal`.
- The file-bytes-unchanged check and the other-writer refusal apply only to the whole-store path. Late writes to a closed experiment are already rejected by claim-epoch fencing (`_reject_stale_claim_in_txn`).
- The result dict reports `consistent_snapshot: True` instead of `source_bytes_verified_unchanged`.

## 4. What replaces `control_mode` single/shared and `evidence_sources`

**Nothing replaces them.** Both existed because judgements and evidence lived in different files, and possibly in several evidence files:

- `single` versus `shared` chose WHICH control file.
- `evidence_sources` authorised which evidence files a control file could read.

With one live DB:

- There is one control location: the tables in §2.
- Authorisation reduces to a closed set: the live DB, plus the `sealed_archives` rows of that same DB, which only the seal transaction writes.

Deleted with them:

- `authorize_source`, `declare_bound_source`, `attach_source` and `source_for_identity`
- `_resolve_source` and `_source_resolution`
- `_unadopted_older_peers` and `adopt_existing_experiments`
- `ensure_selection_bootstrap`, `bind_runner_evidence`, `authorize_evidence_store`, `_admit_default_store` and `_seed_registrations`
- every `source_id` parameter in `best_run` and `selected_runs`

`store_id` stays in payloads, valued as the live identity.

The workspace viewer that read sealed archives through a manifest was removed (`fix-0gh0`).

## 5. Concurrency: many processes writing one WAL DB

The writers are `run_chatbot`, `run_fastapi_mcp` servers, harnesses, `train`, and UI control writes. The existing store discipline is kept, and it is sufficient:

- WAL mode, `synchronous=NORMAL`
- per-call connections with a 30 s busy timeout
- `BEGIN IMMEDIATE` for every write
- the sink's busy-retry queue

Rules for control writes:

- **One short `BEGIN IMMEDIATE` per user action, each a few milliseconds.** This replaces `selection.sources.lock` and `.setup.lock`, both fcntl locks. Read-modify-write sequences (`seq` increments, election, tombstone and retire) sit inside that one transaction.
- **Busy past 30 s surfaces as HTTP 503, retryable.** It is never swallowed.
- **Long transactions stay batched as today.** Prune runs up to 5,000 rows per transaction. A whole-store archive holds only a read transaction, which does not block writers.
- **Local filesystems only.** WAL over NFS is unsupported. The docs already say so for the live DB; nothing changes.

**Scope addition (`.6`) — per-incarnation writer health.** `writer_health` is ONE diagnostics row with an incarnation stamp. Concurrent writers overwrite each other's stamp, so `health_delta` would report `writer_restarted` and invalidate every evidence run captured while the interactive server was up.

Fix:

- Each writer persists to diagnostics key `writer_health/<incarnation id>`. A diagnostics row needs no DDL.
- `health_delta` compares snapshots of the incarnation that owns the run: the in-process sink, or the claim's `server_incarnation`.
- `merge_writer_health` keeps its floor semantics within one incarnation.
- Rows of dead incarnations are pruned with the store's other diagnostics housekeeping.

## 6. Second-live-DB guard and run isolation

**Guard decision: refuse for registered experiments, and log a loud warning for ad-hoc runs.**

- **Controllers resolve the DB themselves.** `ExperimentController` and `ExperimentRunner` drop their `db_path` argument, take `workflow_folderpath`, and resolve `state_paths.observability_db(workflow_folderpath)`. No caller can point them elsewhere.
- **A registered experiment must be found in the resolved DB.** Since the registration lives in the live DB, `create_experiment` for a registered id that is absent there raises `ExperimentNotRegisteredHere`. The message names:
  - the resolved path;
  - `FASTWORKFLOW_STATE_ROOT`;
  - `FASTWORKFLOW_WORKFLOW_ID`.

  The 2026-10-04 Nemotron failure becomes impossible rather than detectable.
- **Ad-hoc runs only warn.** When an ad-hoc (unregistered) run starts with `FASTWORKFLOW_STATE_ROOT` or `FASTWORKFLOW_WORKFLOW_ID` overridden, it logs one WARNING naming the resolved DB. A blanket refusal is impossible: the test suite and embedders legitimately use private state roots.

**Isolation without a separate state root:**

- **LM cache.** `FW_LM_CACHE=0` already makes `dspy_utils` build LMs with `cache=False`. `DSPY_CACHEDIR` is DSPy's own variable: the harness may set it per run, and fastWorkflow neither reads nor needs it.
- **Session and checkpoints.** `runner.channel_for(experiment_id, task_id, attempt)` is already unique per attempt, and `session_state/` and `checkpoints/` are keyed by channel. Harness-spawned servers additionally set `FASTWORKFLOW_DEPLOYMENT_ID=<experiment_id>`, so checkpoint partitions never mix with the interactive server's `default`.
- **Evidence.** Rows are tagged `experiment_id` / `task_id` / `attempt`, fenced by claims, and exempt from pruning while bound (§7).

## 7. Pruning exemption for experiment-bound evidence (`fix-10vj.2`)

An experiment is **bound** while:

```sql
SELECT experiment_id FROM experiments
 WHERE evidence_sealed_at IS NULL
   AND experiment_id NOT IN (SELECT experiment_id FROM evidence_releases)
```

The release table is guarded by `control.present` and treated as empty when absent.

Rules:

- **Exempt from both the horizon and the size cap.** Every prune delete gains `AND (experiment_id IS NULL OR experiment_id NOT IN <bound>)`. This covers spans, artifacts and the conversationless-turn sweep. `offload_*` deletes gain the same predicate through their turn's `experiment_id`. SQLite materialises the subquery once, and the bound set is small: no index and no measurable cost.
- **Released by sealing, or explicitly.** Sealing releases by setting `evidence_sealed_at`; the archive now carries the evidence. Explicit release goes through `store.release_experiment_evidence(experiment_id, actor, reason)`, which inserts into `evidence_releases`. Invalid experiments are not auto-released, so their evidence stays inspectable.
- **The size cap may be exceeded.** If bound evidence alone is over `max_bytes`, prune evicts everything unbound and records `prune_over_cap_bound_bytes` in diagnostics. It never deletes bound rows.
- **`pruning_suppressed()` is unchanged.**

## 8. Older DB found by the writable store (`fix-10vj.9`)

**Writable open:**

- A DB with `user_version < 7` that has tables raises `OlderObservabilityStore`. The message gives the path, the found version, and "fastWorkflow never deletes evidence; move it aside (see fix-10vj.7)".
- An empty file (version 0, no tables) is initialised.
- A newer DB is refused, as today.
- The caller handles it exactly as it handles a newer-schema DB today.
- **`_delete_older_store` and `_OlderPopulatedStore` are deleted.** No code path deletes a user DB.

**Read-only open:** it requires exactly v7. `MIN_READABLE_SCHEMA_VERSION` goes, along with:

- the v6 shape branches in `_records_feedback_taxonomy` and `list_human_feedback`;
- the `handler_feedback` v6 writer branch;
- `archive.py`'s acceptance of pre-v7 stores.

`FEEDBACK_TAXONOMY_SCHEMA_VERSION` stays only as the expected-version constant if anything still names it, and is otherwise deleted.

**Also deleted:** the `?benchmark_experiment=` source in

- `handler_feedback.py`, `handler_navigation.py`, `navigation.py`, `server.py`, `handler_experiment.py`;
- `handler_benchmark._registered_store`;
- the four JS DOM harnesses.

## 9. Per-child implementation plan

**Order:** `.9` → `.3` → `.4` → `.8` → `.5` → `.2` → `.6`, then `.7` by hand.

- `.9` touches only `store.py`'s open path and the handlers' source parameter.
- `.3` lands `control.py` with the full DDL that every later child uses.
- `.4` and `.8` sit on the same tables but in disjoint files.
- `.5` is independent and is placed where it least conflicts with `selection_api.py` edits.
- `.2` needs `evidence_releases`.
- `.6` needs `.2` and `.3`, per its issue.

Net line estimates are for production code; tests are given in parentheses.

**`.9` — remove old-evidence reading.** About −400 (tests about −1,500).

- `store.py`:
  - delete `MIN_READABLE_SCHEMA_VERSION`, `_delete_older_store`, `_OlderPopulatedStore` and the v6 branches (lines around 1917, 3070, 3105, 3160 and 5771);
  - add `OlderObservabilityStore`;
  - make `ReadOnlyObservabilityStore` and `open_for_annotation` require v7.
- `archive.py`: v7 only.
- `handler_feedback.py`: drop the v6 writer branch.
- `navigation.py`, `handler_navigation.py`, `handler_benchmark.py`, `handler_experiment.py`, `server.py`: drop `benchmark_experiment`.
- JS: delete the `benchmark_experiment` plumbing.
- Tests:
  - delete the v6-only cases in `test_task_feedback`, `test_historical_archive`, `test_observability_store` and `test_experiment_container`;
  - delete `chatbot_finder_registered_source_dom.cjs` and `chatbot_source_switch_dom.cjs`;
  - port the "older DB is refused, file untouched" assertion.

**`.3` — control tables; selection, pair review and feedback move in.** About −2,300 (tests about −3,500, ports about +600).

- **New:** `observability/control.py`, about +180 lines (§2.1). `store.py` runs `control.ensure` in `_ensure_schema_once` and filters control tables in `archive_to` (§3 Rule 1).
- **`selection.py` (2,560 lines → about 1,500):**
  - `SelectionControlStore` takes an `ObservabilityStore` and uses `control.write` / read connections;
  - decision, pointer, election and history logic is kept verbatim;
  - `record_decision` / `apply_scoped_decision` call the existing join (`_register`) first when the named experiment exists in `experiments` but is not yet a member, inside the same transaction (§2.3);
  - delete the modes, the evidence sources, bootstrap, adoption, `control_meta`, its own `_connect` / `_ensure_schema`, the module-level `control_mode_of`, `initialize_winner_for`, `selection_control_for`, `open_shared_control`, `SHARED_CONTROL_FILENAME`, `control_db_path_for`, `shared_control_db_path_for`, and the source exceptions;
  - `store_for(experiment_id)` replaces `store_for_source` (§2.6).
- **`pair_review.py` (721 lines → about 250):** the same treatment. Delete `SHARED_PAIR_REVIEW_FILENAME`, `pair_review_db_path_for`, `shared_pair_review_db_path_for` and `_require_authorized`. The roughly 150 duplicated lines of `fix-90i7` go with them.
- **Delete `feedback_sidecar.py`** (579 lines). The sealed-comment writer and merge reader live in `control.py`, and `handler_feedback` routes by the read store (§2.5). Both were later removed with the workspace viewer (`fix-0gh0`).
- **`benchmark/setup.py`:** delete `open_workflow_control`, `ensure_selection_bootstrap`, `bind_runner_evidence`, `authorize_evidence_store`, `_admit_default_store`, `_seed_registrations`, `_authorize_recorded_store`, `_declare_unreadable` and `workflow_control_db_path`.
- **`selection_api.py`:** `_open_control` → `SelectionControlStore(live store)`. Its 409 "no selection control sidecar at …" becomes an empty contest. `_AuthorizedReader` and `_authorized_sources` are deleted. `_pair_review` uses the live DB.
- **Other callers:** `best_run.py` and `selected_runs.py` drop `source_id` resolution. `runner.py` drops `selection_control_db_path`. `static/src/160-comparison.js` gets new message text.
- **Tests:**
  - port winner, decision, pair-review and feedback behaviour in `test_experiment_winner_selection`, `test_selection_api`, `test_execution_comparison`, `test_task_feedback`, `test_human_feedback` and `test_task_best_run` to one live store;
  - delete the mode, source, bootstrap and sidecar tests (decision 4);
  - add one test that a GET on a store without `control_v1` creates nothing;
  - add one test that a pre-existing, unenrolled experiment is enrolled and elected by an explicit promote, while GETs on it enrol nothing;
  - add one test that a sealed copy contains no control table and that its sha is unchanged after a later promote.

**`.4` — registrations and known archives.** About −350 (tests about −600).

- `setup.py`: `create_experiment`, `duplicate_experiment`, `update_experiment_description`, `load_experiment`, `registered_experiments`, `experiment_manifest`, `bind_experiment` and `delete_empty_experiment` become SQL on `experiment_registrations`.
- Also in `setup.py`, delete:
  - `_registration_path`, the `.deleted/` handling and `_atomic_json` use for registrations;
  - `_delete_locked`'s compensation;
  - `_lock` / `.setup.lock`;
  - `_sources_path`, `_sources_lock`, `known_evidence_sources` and `_source_resolver`.
- `store.record_workspace_archive` inserts `sealed_archives`.
- Handlers: `handler_benchmark`, `navigation` / `handler_navigation` and `selection_api` read registrations through the read-only store.
- Tests:
  - port `test_benchmark_setup`, `test_experiment_registration_and_repeat` and `test_experiment_winner_deletion` to the table;
  - delete the JSON-file and sources-map tests.

**`.8` — training row and legacy leftovers.** About −140 (tests about −300).

- `launcher.py`: `TRAIN_PID_FILENAME`, `write_pid_file`, `_read_pid_record` and the pid-file scan become the `training_process` row (§2.7). `chatbot_train.log` is kept.
- `store.py`: delete `LEGACY_OFFLOAD_SIDECAR_SUFFIX` and `_remove_legacy_offload_sidecar`.
- Delete the `conversations/` handling in `http_common.py`, `server.py`, `run_fastapi_mcp/__main__.py`, `utils.get_channelconversations_dir` and `state_paths.conversations_dir`.
- Tests: port `test_run_chatbot_train` and `test_training_history`; delete the conversations-dir and legacy-offload cases in `test_state_paths`, `test_run_chatbot_server`, `test_offload_*` and `test_context_runtime_manifest`; update `soak/memory_soak.py`.

**`.5` — consistency cache.** About −45 (tests about −150).

- `consistency.py`: `VectorCache` becomes a bounded in-memory LRU (§2.9).
- `selection_api.py`: drop `_CONSISTENCY_CACHE_DIRNAME` and the per-workflow directory.
- Tests: delete the file-layout tests in `test_consistency`; add LRU eviction and a hit test.

**`.2` — pruning exemption.** About +70 (tests about +120).

- `store.prune` gets the bound predicate (§7).
- `release_experiment_evidence` is added.
- Tests: the acceptance test in the issue, plus "seal releases" and "explicit release".

**`.6` — runners record into the live DB.** About +40 (tests: port about 40 constructor sites).

- `runner.py`: `ExperimentController(workflow_folderpath, ...)` and the `ExperimentNotRegisteredHere` guard (§6). Harness servers set `FASTWORKFLOW_DEPLOYMENT_ID`.
- `store.archive_to(experiment_id=...)`: the scoped seal (§3), with `seal_workspace_evidence` passing the id.
- `evidence_run.py`: archive the experiment scope.
- `store.py`: per-incarnation writer health (§5).
- Tests: port the controller constructions in `test_experiment_declarations`, `test_task_best_run`, `test_experiment_controller`, `test_benchmark_setup`, `test_experiment_sealing` and others to `workflow_folderpath` plus a tmp state root. Add a two-process test: an interactive writer and a harness writing one DB, with the evidence run still valid.

**Totals:** about −3,100 production lines and about −5,000 test lines net, with one new module (`control.py`).

## 10. Scope changes relative to the child issues

- **`.6` gains per-incarnation writer health (§5).** Without it, a shared DB invalidates evidence runs whenever the interactive server is up.
- **`.6` gains the experiment-scoped seal (§3).** A whole-store seal of a shared, multi-hundred-MB, concurrently written DB is neither cheap nor possible under the old byte-stability check.
- **`.9` also refuses workspace manifest stores in `mode: "live"`**, and `archive.py` stops accepting pre-v7 stores. (The manifest refusal is moot since `fix-0gh0` removed the workspace viewer.)
- **`.3` lands the DDL for all control tables**, including those `.4`, `.8` and `.2` use, because one feature marker must name one table set.
- **`.4` drops `benchmarks/.setup.lock` too.** It is runtime state in the project folder, and SQLite transactions plus `vN.json`'s exclusive create make it unnecessary.
- **`.8`: `analysis.json` stays** (authored, §1.2). The epic's "benchmarks/ contains only vN.json" acceptance should read "only authored files: `vN.json` and `analysis.json`".

## 11. Owner resolutions (2026-10-04)

1. **Experiment seals contain only that experiment's rows** (§3).
2. **Workspace-mode comments are refused** with a clear message when the archive's workflow has no live DB on this machine. (Superseded: `fix-0gh0` removed workspace mode.)
3. **The consistency cache is an in-memory LRU**, not a table (§2.9).
4. **Evidence release is API-only for now** (`release_experiment_evidence`), plus a diagnostics counter of bound bytes; no UI button yet.
5. **Complete-but-never-sealed experiments are never auto-released.**
6. **Workspace manifest stores with `mode: "live"` are refused.** (Superseded: `fix-0gh0` removed workspace manifests.)
7. **Per-incarnation writer health is in scope for `.6`** (§5).
8. **Pre-existing experiments are enrolled only by an explicit decide, keep or promote request** that names them, in the same transaction (§2.3). Nothing enrols them automatically.
9. **`chatbot_train.log` stays a separate file** (§1.1).

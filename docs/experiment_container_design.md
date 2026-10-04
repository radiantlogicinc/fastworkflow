# fastWorkflow Experiment Container — Design Doc (Phase 0)

Status: **DECIDED — Phase-0 gate for epic `fix-bn1`, revision 2 (post adversarial
review).** Implementation of `fix-bn1.2`…`fix-bn1.8` may begin against the rulings
in §11. §13 is the review record.
Author: Claude (with Dhar Rawal), 2026-08-29.
Baseline verified against: `ido-complex-tasks-project` @ `89ba8ba` (post-`fix-sb8`
merge `7a0340e`). Every file:line citation below was read against that tree.
Parent platform design: `docs/fastworkflow_observability_studio_design.md`
(rulings `[R1]`–`[R28]`). Sibling Phase-0 gate: `docs/distillation_observability_design.md`
(rulings `[DR1]`–`[DR55]`).

**Ruling namespace.** This document uses a fresh `[XR#]` namespace. It is peer to
`[R#]` and `[DR#]` and binding on the same terms. Where an `[XR]` ruling depends
on a `[DR]` or `[R]` ruling it cites it; no `[XR]` ruling supersedes one.

**Revision 2** incorporates six adversarial reviews (§13): 38 findings survived
independent verification, eight of them blocking. Two rulings were **reversed**
(`[XR7]`, `[XR8]`), one table was **added** (`experiment_attempts`), and the
scoring section was rewritten around the finding that the derived pass predicate
of revision 1 scores a tau2 retail attempt as a pass whether or not the task was
accomplished.

**Amendment, 2026-09-08 (owner decision, `fix-1ay`).** `hypothesis` is
**withdrawn**, and with it the write-once half of `[XR12]`: the column, the
`set_experiment_hypothesis` writer, `HypothesisIsWriteOnce`, the harness and
controller keyword, the `PATCH` 409 and the read-only block in the studio are
all gone. `label` takes its place as the experiment's one piece of author prose
— free text, optional, and editable through the studio only while the
registration is still unbound — and is **renamed `description`**, which is what
it now is. Everything §7.3 says about **terminal `invalid`** stands unchanged;
only the pre-registration half was removed.

`SCHEMA_VERSION` was deliberately **not** bumped, on the owner's instruction.
The one live store (`~/.local/state/fastworkflow/workflows/ido_workflow/
observability.sqlite3`) was rewritten in place on 2026-09-08: `label`
became `description` and `hypothesis` was dropped; later the same day
`analysis_json` was dropped too, leaving `notes` as the one post-run
annotation. Frozen collection stores
under `evaluation/collections/` are provenance, not compatibility, and
were left as written. `_ensure_schema` carries no rename or drop; a store
that still has the old columns is not opened by this build.

---

## 0. What does not exist

An **experiment** is a labelled set of tasks, each run one or more times, whose
turns can be found, scored and compared as a unit. Nothing in the tree is that
object. Verified 2026-08-29: the string `experiment` does not occur in any
`fastworkflow/*.py` outside two unrelated prose comments under `train/`, and
occurs nowhere in `fastworkflow/run_chatbot/static/index.html`. (Since
2026-09-30 that page's source lives as ordered parts under
`fastworkflow/run_chatbot/static/src/`; `load_index_html()` assembles them into
the same single page.)

What exists, and why each is not this:

| What exists | Why it is not an experiment container |
|---|---|
| `observability.sqlite3` is three-level — channel → conversation → turn (`observability/store.py:1050,1058`), with `idx_turns_conv` and the shipped `/api/channels`, `/api/conversations`, `/api/turns` — and the SPA already nests those three levels (`index.html:736-746,767-787` as of 2026-08-29; today the rail is `renderHierarchy` in `static/src/120-header-rail.js`, fed by `run_chatbot/navigation.py`) | The read shape generalises for free. The missing thing is a **grouping dimension**, not a viewer. |
| `observability/evidence_run.py` mints a `run_id` per measured run (`:267`) and asserts zero-drop, prune suppression, archival and provenance | That `run_id` exists **only** inside the dict `EvidenceRun.as_record()` returns (`:141-158`). **No column anywhere joins a turn back to the run it belonged to.** That is the gap in one sentence. |
| `distillation_runs` (`:1095-1125`) is one row per compared **message** | A single trial, one level *below* a task attempt. |
| `train_runs` (`:1087`) | A training publication, unrelated to task execution. |
| `fix-9eg.3`'s aggregate views | Aggregate over whatever turns are in the DB, with no experiment boundary to `GROUP BY`. That bead depends on `fix-bn1.2` for exactly this reason. |

`evidence_run()` has **no production callers today** — only `tests/test_observability/evidence_run.py`.
So `fix-bn1.4`'s harness is the first real consumer, and there is no existing
bundle-writing code whose `as_record()` → column mapping can be copied.

---

## 1. Decisions, in one table

| # | Question | Ruling |
|---|---|---|
| `[XR1]` | Is `experiment_id` the `EvidenceRun.run_id`? | **No.** Separate id, `exp-<32 hex>`. Evidence runs are recorded against it, one row per segment in a child table. |
| `[XR2]` | Where does `task_id` come from? | Caller-supplied stable string, required, never derived from `conversations.topic`. |
| `[XR3]` | What is `attempt`? | A caller-supplied 1-based counter, unique within `(experiment_id, task_id)` and enforced by a PRIMARY KEY, not by convention. |
| `[XR4]` | Denormalise onto `conversations` only, or also `turns`? | **Both**, following `channel_id`. Not onto `spans`. |
| `[XR5]` | Schema version | No bump. Columns in **both** the `CREATE TABLE` literal and a guarded `ALTER TABLE`; `FEATURE_EXPERIMENTS_V1` marker. Per `[DR28]`. |
| `[XR6]` | Capture policy for the experiment surface | **Scrub-only, all of it**, on the `set_diagnostic` and `_POLICY_EXEMPT_TURN_COLUMNS` precedents. No `POLICY_PATH_*` constants — the `spans.channel_id` convention. |
| `[XR7]` | *(reversed in rev 2)* Capture policy for `task_id` | **Scrub-only**, not policed. Policing withheld nothing (the plaintext is in `record_json`) and broke every equality lookup. |
| `[XR8]` | *(reversed in rev 2)* Evidence provenance | Its own child table, scrub-only, one row per segment. Not a policed JSON array on `experiments`. |
| `[XR9]` | URL noun split with `/api/distillation` | `run` stays distillation's. Experiments use `experiment` / `task` / `attempt`. Cross-links extend the shipped routes. |
| `[XR10]` | Id-prefix collision | `EvidenceRun`'s default prefix changes `run-` → `evr-`. |
| `[XR11]` | How are the new rows written? | **Direct store methods**, not the sink record queue. The `record_train_run` shape. |
| `[XR12]` | Write-once `hypothesis` | Enforced in one store transaction; `invalid` is likewise terminal in SQL. |
| `[XR13]` | The attempt verdict | An **explicit per-attempt row** the harness writes, with a named `outcome_source`. Never derived silently from turn columns. |
| `[XR14]` | The denominator | `declared_tasks` × `declared_attempts`, `NOT NULL`, checked **by the store** at completion — not asserted by the caller. |
| `[XR15]` | Erasure and retention | `forget_channel` invalidates and deletes its own rows; `clear_conversations` deletes; `prune` exempts; every store method raises on a 0-row update. |
| `[XR16]` | Repeat-attempt determinism | Two contaminants, not one: the DSPy response cache **and** the shared utterance/clarification cache. Both blocking for `fix-bn1.4`. |
| `[XR17]` | Binding | Through `bind_observability_identity`, validated there. `channel_id` semantics unchanged. |
| `[XR18]` | One channel per attempt | `exp:<experiment_id>:<task_id>:<attempt>`, one WEC per attempt, unique `workflow_id_str`. |
| `[XR19]` | Comparability | Equal task-id **sets**, equal declarations, equal capture profile. Cardinality is not comparability. |
| `[XR20]` | Read-modify-write | A policed column may never be read-modify-written. General ruling; it is why `[XR8]` reversed. |

---

## 2. Rejected: `channel_id` as the experiment label

The obvious mapping — channel = experiment, conversation = task — is rejected.
Recorded here because it is the first thing anyone proposes.

1. **It is the concurrency key.** Every FastAPI route runs under
   `leased_session(channel_id)`, and a second turn on a busy channel gets `409
   ChannelBusyError`. Tasks in one experiment could never run in parallel.
   *(Mechanism correction, verified: `leased_session`
   (`run_fastapi_mcp/utils.py:1107-1157`) is an eviction refcount, **not** mutual
   exclusion — the serialisation is the registry pointer. The conclusion stands;
   the mechanism named in the bead does not.)*
2. **It is the suspended-state key.** One `ask_user` pending blob per channel:
   the ABC at `session_state_store.py:146,150,154` and the disk implementation at
   `:343,375,392`. All tasks would share one slot.
3. **Topic uniquification is channel-scoped.** `_unique_topic_in_txn`
   (`observability/store.py:1672`) suffixes collisions across the channel, so
   repeating a task set inside one channel yields `Task 3`, `Task 3 1`,
   `Task 3 2` — titles stop being join keys exactly when they are needed as
   join keys.
4. **`conversation_id` is a per-channel INTEGER** from a monotonic counter
   (`:1508`), so task identity would have to live in `topic`: LLM-generated and
   uniquified, the worst available join key.
5. **Two levels cannot express experiment → task → attempt**, and repeats are
   what `pass^k` is.

`chat_session.py:127` already mints `channel_id=f"cli:{timestamp}"` — one channel
per session. **Channel is a session.** This epic adds a labelling dimension
beside it and leaves channel semantics untouched (`[XR17]`).

---

## 3. Identity `[XR1]` `[XR2]` `[XR3]` `[XR4]`

### 3.1 `experiment_id` is not the evidence run id `[XR1]`

`experiment_id = "exp-" + uuid4().hex` — minted by the harness at creation,
before any task runs.

It is **not** `EvidenceRun.run_id`, for one reason that cannot be worked around:

> An experiment may span more than one evidence run. `evidence_run()` is a
> context manager whose `run_id` is minted at entry (`observability/evidence_run.py:267`) and
> whose verdict is only meaningful after the block ends (`:123-134` —
> `delta is None` ⇒ invalid). A harness that crashes and resumes opens a
> **second** `evidence_run()` block and gets a **second** `run_id`. If
> `experiment_id == run_id`, the resumed half of a 45-attempt sweep is a
> different experiment from the first half, and the epic's whole purpose is
> defeated by a crash.

Evidence runs are recorded in a **child table**, `experiment_evidence_runs`, one
row per segment (§4.1). Revision 1 put a JSON array in an
`experiments.provenance_json` column; §6.4 records why that was wrong twice over.

**Store the whole `as_record()`, not `record["observability"]`.** The bead's
phrase "the `EvidenceRun.as_record()` observability block" is ambiguous:
`record["observability"]` alone is the `ObservabilityProvenance` sub-dict and
carries neither the run id, nor `valid`, nor `problems`, nor the archive digest.
The full dict is already JSON-round-trippable (asserted at
`tests/test_observability/evidence_run.py:422`).

Validity across segments is a conjunction: an experiment can be `complete` only
if **every** segment row has `valid = 1`.

### 3.2 `task_id` `[XR2]`

Caller-supplied, stable, opaque to the framework. A tau2 task name, a corpus
message id, a row key in a CSV — the framework does not interpret it.

Explicitly **not** derived from `conversations.topic`: that value is
LLM-generated *and* channel-uniquified (§2 item 3), which makes it the one string
in the system guaranteed not to be stable.

Required, not optional: binding an experiment without a `task_id` is refused at
the binding chokepoint (§5.1). A NULL `task_id` under a non-NULL `experiment_id`
would produce turns that belong to an experiment and to no task — rows that
contribute to a numerator and to no denominator.

### 3.3 `attempt` `[XR3]`

A 1-based integer, supplied by the harness, unique within `(experiment_id,
task_id)`. Not `conversation_id`: that is a per-channel counter (`:1508`) and
`[XR18]` puts each attempt on its **own** channel, so every attempt's
`conversation_id` would be `1`.

Uniqueness is **enforced**, not asserted: it is the PRIMARY KEY of
`experiment_attempts` (§4.1) and a UNIQUE partial index on `conversations`.
Revision 1 stated the invariant and enforced it nowhere, and its own resume rule
violated it.

Retained per attempt and never collapsed. `pass^k` is not a function of a
per-task pass *rate*: an agent that passes each task 80% of the time has a very
different `pass^3` depending on whether its failures are correlated across
attempts. Storing a rate at write time destroys exactly the information the
metric is about.

### 3.4 Denormalisation: `conversations` **and** `turns`, not `spans` `[XR4]`

`channel_id` is already denormalised onto `turns`, `spans` and `artifacts` so
filters need no join. The same argument applies here, with one exclusion:

* **`conversations`** — the attempt *is* a conversation; the labels are its
  identity.
* **`turns`** — every score query filters turns. Without the column, every one
  of them is a join through `conversations` on `(channel_id, conversation_id)`.
* **`spans` — no.** Spans are the highest-volume table by an order of magnitude,
  they are best-effort and droppable, and `spans.trace_id == turns.turn_key`
  (`[DR1]`) already resolves any span to its turn and therefore to its experiment
  in one join. Three columns on the largest table, to save a join nothing
  performance-sensitive makes, is not a trade this epic gets to make. Recorded
  rather than omitted, so a later slice that wants it argues for it.

---

## 4. Schema `[XR5]`

Inherited from `fix-sb8`, **not re-litigated**. `[DR28]` (recorded at
`observability/store.py:1385-1399`) abandoned `user_version` gating: the premise
"schema v1 was never shipped" expired at v3.2.0, so two released builds already
disagree about shape at the same `user_version`. Readers feature-detect.

Concretely — and this is **two halves, not one**, which revision 1 got wrong:

1. **The `CREATE TABLE` literals.** `experiment_id TEXT, task_id TEXT, attempt
   INTEGER` are added to the `CREATE TABLE IF NOT EXISTS conversations`
   (`:1050-1054`) and `... turns` (`:1057-1068`) literals in `_SCHEMA_STATEMENTS`,
   and the three new tables join the list. **On a fresh DB this is the only thing
   that creates the columns**, because `PRAGMA table_info` returns nothing and the
   guarded ALTER below is skipped. The shipped precedent works exactly this way:
   `distillation_pass TEXT` is inside the `CREATE TABLE IF NOT EXISTS spans`
   literal at `:1079` *as well as* in the ALTER, and the comment at `:1385-1390`
   says so in as many words.
2. **The guarded `ALTER TABLE` pair**, for existing DBs, copying `:1398-1402`:

   ```python
   turn_cols = {r[1] for r in conn.execute("PRAGMA table_info(turns)").fetchall()}
   if turn_cols and "experiment_id" not in turn_cols:
       conn.execute("ALTER TABLE turns ADD COLUMN experiment_id TEXT")
       conn.execute("ALTER TABLE turns ADD COLUMN task_id TEXT")
       conn.execute("ALTER TABLE turns ADD COLUMN attempt INTEGER")
   # …the same shape for conversations
   ```

   The `X_cols and` half is load-bearing in **both** directions: without it the
   ALTER runs before `CREATE TABLE turns` exists on a fresh DB and raises
   `no such table`; with it and without half 1, a fresh DB gets no columns at all
   and `CREATE INDEX idx_turns_experiment` raises `no such column`. Both branches
   were reproduced against a real SQLite file during review.
3. **Ordering**, already learned once at `:1385-1396`: the ALTER pair runs
   **before** the `_SCHEMA_STATEMENTS` loop, because the loop contains the
   `CREATE INDEX` statements (`:1181-1200`) and `idx_turns_experiment` names the
   new columns — the same constraint as `idx_spans_trace_pass`. The
   `conversations.updated_at` ALTER sits *after* the loop only because no index
   names it; do not copy that one.
4. `FEATURE_EXPERIMENTS_V1 = "experiments_v1"` beside `FEATURE_DISTILLATION_V1`
   (`:124`), appended to the `_merge_schema_features(conn, [...])` call
   (`:1426`). That helper **merges** (`:1439-1461`) — writing
   `json.dumps(["experiments_v1"])` straight into diagnostics would drop
   `distillation_v1` and break `has_feature` for every existing reader.
5. `_load_features`' PRAGMA fallback (`:1477-1485`) gains an arm: a DB whose
   `turns` table has `experiment_id` but whose marker is missing is
   `experiments_v1`.
6. `SCHEMA_VERSION` **stays 1**. `tests/test_observability/store.py:1117`
   asserts it, and a bump would also change every `provenance.db_schema_version`
   and every `archive["schema_version"]` already recorded.

### 4.1 DDL

```sql
-- 1. the container
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id          TEXT PRIMARY KEY,      -- 'exp-<32 hex>'
    description            TEXT NOT NULL,         -- author prose, optional (amendment)
    notes                  TEXT,                  -- freely editable
    arm                    TEXT,                  -- 'baseline' | 'treatment' | NULL
    baseline_experiment_id TEXT,
    status                 TEXT NOT NULL,         -- 'running' | 'complete' | 'invalid'
    invalid_reason         TEXT,                  -- CLOSED vocabulary, §6.3
    invalid_detail         TEXT,                  -- free text, append-only, scrubbed
    declared_tasks         INTEGER NOT NULL,      -- the denominator [XR14]
    declared_attempts      INTEGER NOT NULL,      -- k                 [XR14]
    workflow_name          TEXT,
    capture_profile        TEXT NOT NULL,         -- [XR19]: comparability
    capture_policy_version TEXT NOT NULL,         -- [XR19]
    created_at             TEXT NOT NULL,
    completed_at           TEXT);

-- 2. the unit of scoring [XR13]
CREATE TABLE IF NOT EXISTS experiment_attempts (
    experiment_id   TEXT NOT NULL,
    task_id         TEXT NOT NULL,
    attempt         INTEGER NOT NULL,
    channel_id      TEXT NOT NULL,     -- 'exp:<eid>:<task>:<attempt>' [XR18]
    conversation_id INTEGER,
    outcome         TEXT,              -- 'pass'|'fail'|'error'|'incomplete'; NULL while running
    outcome_source  TEXT,              -- who decided; NOT NULL whenever outcome is
    reward          REAL,              -- the grader's raw number, when it has one
    restarts        INTEGER NOT NULL DEFAULT 0,
    started_at      TEXT NOT NULL,
    finished_at     TEXT,
    detail_json     TEXT,
    PRIMARY KEY (experiment_id, task_id, attempt));

-- 3. the evidence segments [XR1] [XR8]
CREATE TABLE IF NOT EXISTS experiment_evidence_runs (
    experiment_id   TEXT NOT NULL,
    seq             INTEGER NOT NULL,  -- 1-based, in order
    evidence_run_id TEXT NOT NULL,     -- 'evr-…' [XR10]
    valid           INTEGER NOT NULL,
    started_at      TEXT, completed_at TEXT,
    record_json     TEXT NOT NULL,     -- the whole EvidenceRun.as_record()
    PRIMARY KEY (experiment_id, seq));

-- 4. the denormalised labels: in the CREATE TABLE literals AND as guarded ALTERs
--    conversations += experiment_id TEXT, task_id TEXT, attempt INTEGER
--    turns         += experiment_id TEXT, task_id TEXT, attempt INTEGER

CREATE INDEX IF NOT EXISTS idx_turns_experiment
    ON turns(experiment_id, task_id, attempt) WHERE experiment_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS idx_conv_experiment_attempt
    ON conversations(experiment_id, task_id, attempt) WHERE experiment_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_experiments_baseline
    ON experiments(baseline_experiment_id) WHERE baseline_experiment_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_experiments_status
    ON experiments(status, created_at);
CREATE INDEX IF NOT EXISTS idx_experiment_attempts_channel
    ON experiment_attempts(channel_id);
```

The two label indexes are **partial**, following `idx_spans_command` (`:1182`)
and `idx_distill_runs_replay`: an ordinary chatbot turn is not in an experiment
and must cost nothing.

`idx_conv_experiment_attempt` is **UNIQUE**. It is what makes `[XR3]` an
invariant rather than a hope, and it is what forces §9.2's resume to be explicit
about the crashed attempt's rows instead of quietly minting a second conversation
under the same three labels.

Additions to the bead's DDL, each argued below rather than assumed:
`experiment_attempts` (§8), `experiment_evidence_runs` (§6.4), `declared_*`
(`[XR14]`), `invalid_reason`/`invalid_detail` (§6.3), `capture_profile` and
`capture_policy_version` (`[XR19]`). `provenance_json` from the bead's sketch is
**removed** — §6.4 says why.

---

## 5. Plumbing `[XR17]`

### 5.1 The binding chokepoint

`WorkflowExecutionContext.bind_observability_identity(channel_id=None,
conversation_id=None, embedder_owns_conversations=None)`
(`workflow_execution_context.py:251-277`) is the sole public identity setter; it
is called before the first turn by every embedder (`chat_session.py:126`,
`run/__main__.py:230`, `run_fastapi_mcp/utils.py:586,670,1495`,
`run_fastapi_mcp/__main__.py:2232`), and it already carries the "a `None`
argument leaves the binding unchanged" contract.

The labels bind there. Three new keyword arguments, three new attributes beside
`_channel_id` / `_conversation_id` (`:168-170`).

**Validation at the chokepoint**, since there is nowhere else it can happen
cheaply: a non-NULL `experiment_id` requires a non-empty `task_id` and an
`attempt` that `isinstance(attempt, int)` and `> 0`; anything else raises
`ValueError`. The **type** check is load-bearing and not decorative: SQLite's
`INTEGER` is an *affinity*, not a constraint — a string bound to it that cannot
be losslessly converted is stored as TEXT — so the column's declared type
protects nothing on its own. Every store method that accepts `attempt` also
coerces with `int(attempt)` so the claim is true rather than assumed (§6.3).

**`channel_id` semantics do not change.** It stays the session key: the
`leased_session(channel_id)` refcount and the `SessionStateStore` pending-blob
key. One channel remains one live session, which is what lets attempts run
concurrently.

Related but orthogonal: `fix-9eg.8` unpins `run_chatbot`'s hard-coded
`channel_id = "chatbot"` (`run_chatbot/server.py:429`). That is about **who**;
this is about **which experiment**. They touch the same field and must land aware
of each other.

### 5.2 The write path

`_build_turn_result` (`:1229`, literal at `:1282`) is the **only**
`TurnResult(...)` construction site in the package. Three fields appended to
`TurnResult` (`turn.py:405-450`, whose docstring requires "appended, never
reordered", with defaults — the `execution_records`/`routing_events` precedent at
`:449-450`), passed at `:1282`, projected in `serialize_turn_result`'s `turn_row`
literal (`observability/store.py:1009-1037`).

`upsert_turn_row` needs **no change**: it derives its column list from
`turn_row.keys()` (`:1955-1963`). But that cuts both ways —

> A dict key with no matching column raises `sqlite3.OperationalError` on
> `_sync_write`, which trips the sync breaker (`:4062`) and degrades **every**
> subsequent turn to the queued path for 60 s. The DDL and the projection must
> land in the same commit.

The three columns join the `DO UPDATE` set (unlike `ordinal`, deliberately
excluded at `:1961-1963`): an `awaiting_user` row is upserted again at terminal
finalize and the labels must survive. They cannot *change* across the upsert,
because one WEC instance carries one binding for the life of a logical turn.

`experiment_id` and `task_id` join `upsert_turn_row`'s credential-scrub tuple
(`:1911-1923`) — see `[XR7]`.

`conversations` rows are labelled where they are minted: `mint_conversation_id`
(`:1508`) gains the three optional keyword arguments and writes them into its
`INSERT INTO conversations`. `_assign_ordinal`'s fallback insert (`:2016-2025`) —
the "conversation row not minted here" path — copies them off the turn row it is
inserting for.

### 5.3 What `fix-bn1.3` no longer needs to do

**Verified 2026-08-29: `DistillationResult.run_id` already exists**
(`distillation.py:627`, comment citing `fix-sb8.2`), is set on both return paths,
is accumulated in `WEC._distillation_run_ids`, and is printed by the `//exit`
summary at `run/__main__.py:288-296`. The bead's third bullet is **already
shipped**; re-implementing it would create a second, colliding path — the exact
"work shipped under a neighbouring issue" pattern `CLAUDE.md` warns about.

What remains of that bullet is the naming half, `[XR10]`.

### 5.4 Id namespaces `[XR10]`

Four ids, three of them spelled `run_id`:

| Id | Format | Where |
|---|---|---|
| distillation run | `run-<12 hex>` | `distillation_runs.run_id` |
| evidence run | `run-<YYYYmmddTHHMMSSZ>-<8 hex>` | `observability/evidence_run.py:267`, no table today |
| train run | `<YYYYmmddTHHMMSS>-<8 hex>` | `train_runs.run_id` |
| replay | `rpl-…` | `distillation_runs.run_id` under replay |

The first two share the literal prefix `run-` and differ only in body shape.
They can never collide as primary keys, so this is a legibility defect — until
`experiment_evidence_runs` holds one beside a `distillation_runs` join, at which
point `run_id` is ambiguous in SQL.

Ruling: **`EvidenceRun`'s default prefix becomes `evr-`**, and the column is
`evidence_run_id`. Cheap and safe: no DB row is keyed on it, no id is derived
from it, and every test passes an explicit `run_id`. The archive filename
(`observability/evidence_run.py:388`) changes shape, which no shipped code parses.

Do **not** touch `distillation._new_run_id()`'s output: `insight_id` and
`divergence_id` are seeded from it (`distillation.py:100`,
`distillation_alignment.py:410`), so changing its value would orphan every stored
citation and every `fw.distill.*` span attribute.

`turn_key`, `trace_id` and `distillation_runs.turn_key` are the **same string
under three names** (`[DR1]`), except under replay. No fourth spelling is
introduced.

---

## 6. Capture policy `[XR6]` `[XR7]` `[XR8]` `[XR20]`

`FW-REQ-002` clause 3 requires every captured field to have a **declared**
policy. The module comment at `observability/store.py:276-290` is explicit that
each non-`TurnResult` surface is "decided here rather than by omission —
including the three that are deliberately scrub-only". This section is that
decision for the fourth surface, and revision 2 reverses two of revision 1's
rulings on it.

### 6.1 The constraint that decides it

Under the `evidence` profile, `user-text` is **omitted**, not bounded
(`observability/capture_policy.py:109-110`) — "a bounded prefix of arbitrary text is still
arbitrary text". `opaque-payload` is **omitted** too (`:104-106`).
`controlled-vocabulary` bounds at 256 bytes (`:130`). `identifier` refuses
`bounded-text` outright at policy-construction time (`:200-206`).

So no *classification* produces "bounded but readable". Only an explicitly
declared `CaptureFieldPolicy` does — and **`resolve_capture_policy()`
(`observability/store.py:232-245`) calls `policy_for_profile(name)` with no
`field_policies` at all.** There is today no runtime surface through which a
deployment can declare one. Verified.

### 6.2 The ruling: the whole experiment surface is scrub-only `[XR6]` `[XR7]`

`experiments`, `experiment_attempts` and `experiment_evidence_runs` are
**scrub-only** in every text column: `redactor.redact(value)` with no
`policy.apply` call, the `spans.channel_id` code shape at `:1758-1760`. So are
`turns.experiment_id`, `turns.task_id`, `conversations.experiment_id` and
`conversations.task_id`.

The governing precedent is **`set_diagnostic`** (`:2286-2302`) together with
**`_POLICY_EXEMPT_TURN_COLUMNS`** (`:248-261`), not `spans.channel_id`'s erasure
argument:

> `_POLICY_EXEMPT_TURN_COLUMNS`: "These two are not evidence, they are
> operational state… Withholding them would not reduce what a bundle exposes — it
> would make the agent forget, which is a behavior change."

The same shape holds one level up, and `set_diagnostic` is the closer analogue
still: it is the record of *whether the workload can be trusted*, and its
docstring refuses a capture policy for exactly that reason even though
`writer_health.last_error` is a raw `repr(exc)`. The experiment surface is that
record. An `EvidenceRun`'s `valid`/`problems`, an attempt's `outcome`, and a
pre-registered `hypothesis` are not evidence *about a tenant*; they are the frame
that makes the evidence interpretable and the verdict on whether it may be used
at all. Withholding them reduces nothing a tenant would care about and destroys
the record's own readability — and `[XR1]`'s `complete` rule and §10.1's
"evidence validity and problems" both become unreadable from the DB under
precisely the profile an evidence-grade run uses.

Note what this argument is **not**. It is not "these fields are developer-authored,
therefore safe": the bead itself observes that policy classifies by content type,
not author, and that is right. The claim is a **dataflow** claim, and it is
testable: *no code path exists by which workflow, model or user content can reach
these tables except through `task_id`, which the caller supplies from its own
task-set file.* `fix-bn1.8` asserts this directly, and a future writer that
violates it fails that test.

The residual risk — an operator pasting a credential into `notes`, an exception
repr landing in `record_json.problems` — is exactly what the scrub catches, which
is why the ruling is scrub-only rather than untouched.

**No `POLICY_PATH_EXPERIMENT_*` constants are declared.** Revision 1 declared
four "so a deployment that disagrees can spell them", and that was wrong twice:
`CapturePolicy.policy_for` is consulted only from inside `apply`, so a constant
never passed to `apply` is inert **by construction of the scrub-only ruling
itself**, not merely by the missing config surface; and it would invert the
codebase's convention, where all six shipped `POLICY_PATH_*` constants are passed
to `policy.apply` at a real write site and the one genuinely scrub-only column,
`spans.channel_id`, deliberately has none. The decision is recorded as a comment
at each write site, matching `spans.channel_id` (`:1721-1730`). Re-admitting
these fields to the policy is a code change, not a configuration change; §12
item 2 records that.

**`[XR7]` is a reversal.** Revision 1 policed `turns.task_id` as `identifier`.
Three independent findings killed it:

1. **It withholds nothing.** `[XR17]` puts `task_id` on `TurnResult`, and
   `serialize_turn_result` writes the raw `model_dump()` into `turns.record_json`
   **before** the `_POLICED_TURN_COLUMNS` loop runs; `_apply_capture_policy` walks
   only `turn_output.command_outputs` and never touches top-level `TurnResult`
   fields. The same row would hold a digest badge in `task_id` and the plaintext
   in `record_json`.
2. **It breaks selection.** The digest disposition stores a ~250-byte serialized
   `CapturedValue` envelope, not a digest string. `GROUP BY` survives (the
   envelope is deterministic — unkeyed SHA-256, no HMAC), but every equality
   lookup this design is built on returns zero rows: `experiment_attempts`' own
   primary key, `GET /api/experiment/<id>/attempts?task=…`, and the new `task=`
   param on `GET /api/turns`. `_POLICED_TURN_COLUMNS` is *defined* as "columns
   that are pure evidence — nothing operational reads them", and `task_id` is
   read operationally by three of this design's surfaces.
3. **It splits the write order.** `turns.task_id` would go through
   `_policed_column` (police, no scrub) while `conversations.task_id` goes
   through `_protected_text` (scrub then police, `:299-336`). Two routes
   producing two different envelopes for one value is the exact failure
   `_protected_text`'s docstring exists to prevent — and §6.5 states the opposite
   rule.

`turns.attempt` and `conversations.attempt` are unpoliced because the chokepoint
(§5.1) admits only a positive `int` and every store method coerces with
`int(attempt)` — **not** because the column is declared INTEGER, which under
SQLite's affinity rules guarantees nothing.

### 6.3 `invalid_reason` is a closed vocabulary `[XR15]`

`forget_channel`'s invalidation (§9.1) would otherwise write a free-text string
interpolating a `channel_id` — which under `[XR18]` embeds a raw caller-supplied
`task_id`, and for a non-experiment channel is a raw session identifier that
every shipped write path scrubs.

So `invalid_reason` holds a **code** from a closed set —
`attempt_shortfall`, `evidence_run_invalid`, `turns_erased`, `never_completed`,
`operator` — and the human detail goes in `invalid_detail`, which is scrubbed and
**append-only** (newline-joined), so a second cause does not erase the first.

### 6.4 Evidence provenance is a child table, not a policed column `[XR8]` `[XR20]`

Revision 1 put a JSON array in `experiments.provenance_json` classified
`opaque-payload` and policed. Two blocking findings:

1. **`opaque-payload` maps to `omit` under `evidence`, not to a digest**
   (`observability/capture_policy.py:104-106`). The entire `as_record()` array — `valid`,
   `problems`, `writer_health_delta`, `archive` — would be erased under the one
   profile an evidence-grade experiment runs in. (The stated *reason* was wrong
   too: `ObservabilityProvenance.config` is a hard-coded tuple of eleven named
   variables holding booleans, integers and a profile name, and holds no DB
   paths. The genuinely un-enumerable parts are `problems`, which embeds
   exception reprs, and the two `writer_health_*` blocks, which carry turn keys —
   which is precisely what `set_diagnostic` refuses to withhold.)
2. **A policed column cannot be appended to.** `[XR20]`, general ruling: *a
   policed column may never be read-modify-written.* Appending segment 2 means
   `SELECT` → `json.loads` → append → rewrite, and under `evidence` the stored
   value is a capture envelope dict, not a list — so segment 1 is unrecoverable
   and the crash-resume story that `[XR1]` exists for cannot work. A `digest`
   disposition breaks it identically, so no reclassification saves it.

`experiment_evidence_runs` fixes both by construction: each segment is an
independent `INSERT`, nothing is inverted, and `valid` is a queryable column
rather than a field inside a blob. `record_json` is scrub-only per `[XR6]`.

### 6.5 Ordering constraints inherited

* **Scrub before police**, always, on non-`TurnResult` paths: `_protected_text`
  (`:299-336`) does it in that order because the scrub is idempotent and a digest
  that depends on which route wrote the value is a digest nobody can compare.
  With `[XR6]` and `[XR7]` there is no policing left on this surface, so the rule
  reduces to "scrub, on both routes, identically" — which is what makes
  `turns.task_id` and `conversations.task_id` byte-identical and joinable.
* **Police last** where a value is post-processed (`apply_label_txn` at
  `:1626-1632`).
* `_protected_text` and `Redactor.redact` both return falsy input unchanged
  (`:329-330`). A NULL column stays NULL. The upserts in §7 depend on it.

---

## 7. The store API `[XR11]` `[XR12]` `[XR14]`

### 7.1 Direct methods, not the record queue `[XR11]`

The three new tables are written by **direct `ObservabilityStore` methods** with
their own short-lived `BEGIN IMMEDIATE` connection — the shape of
`mint_conversation_id` (`:1508`), `record_conversation_label` (`:1559`) and
`record_train_run` (`:2229`).

Not through `SQLiteTraceSink`'s record queue. The grounds are `[R13]`'s two-queue
writer discipline (studio design `:251-266`, `:525`), which exists to keep the
*turn path* off the writer's back — and these are a handful of writes per
experiment that nothing is waiting on. Routing them through the queue would
require a `tracing.py` kind registration, a `NoOpTraceSink` method, an
`_apply_batch` dispatch arm and a `_requeue_records` retry tuple: four new places
to lose a row silently, since an unrecognised kind falls through `_apply_batch`
(`:4523-4539`) with no counter bumped and no log line.

*(Correction to revision 1: it justified this on `[R14]`, which says the
opposite — the shipped sink deliberately opens a store connection on the turn
thread on every terminal turn. The ruling survives the correction; that argument
did not.)*

Consequence, stated because it is a real trade: an experiment-row write is
**synchronous and can raise**. That is correct here. A failed `create_experiment`
must fail the run *before* any task executes.

### 7.2 Methods

```
create_experiment(experiment_id, description, *, declared_tasks, declared_attempts,
                  hypothesis=None, arm=None, baseline_experiment_id=None,
                  workflow_name=None) -> None            # status='running'
record_evidence_segment(experiment_id, seq, evidence_run_id, record: dict) -> None
start_attempt(experiment_id, task_id, attempt, channel_id) -> None
finish_attempt(experiment_id, task_id, attempt, *, outcome, outcome_source,
               reward=None, detail=None) -> None
restart_attempt(experiment_id, task_id, attempt) -> None   # §9.2
complete_experiment(experiment_id, *, force_invalid=None) -> str   # returns the status
update_experiment_notes(experiment_id, notes) -> None
set_experiment_hypothesis(experiment_id, hypothesis) -> None        # write-once
get_experiment(experiment_id) -> Optional[dict]
list_experiments(status=None, arm=None, limit=100, offset=0) -> list[dict]
experiment_tasks(experiment_id) -> list[dict]
experiment_attempts(experiment_id, task_id=None) -> list[dict]
experiment_scores(experiment_id) -> dict                            # §8
compare_experiments(experiment_id, baseline_experiment_id) -> dict  # §8.2
```

`declared_tasks` and `declared_attempts` are **required keyword arguments**, both
`NOT NULL` in the DDL, both checked `> 0`. `[XR14]` is the denominator ruling and
revision 1 made the columns nullable and the arguments optional, which silently
degraded it into the failure it names.

**Every write method raises `ExperimentNotFound` on a 0-row `UPDATE`.** This is
the mid-run contract for `clear_conversations` (§9.1): a whole-DB erase landing
while a harness is running fails the harness loudly, rather than leaving turns
labelled against a container that no longer exists.

Reads live on `ObservabilityStore` (and so on `ReadOnlyObservabilityStore`, which
subclasses it) so the chatbot's read-only handle serves them without a writable
handle ever existing — `[DR53]`'s posture.

### 7.3 Write-once `hypothesis`, terminal `invalid` `[XR12]`

> **Withdrawn in part, 2026-09-08:** the `hypothesis` half of this section no
> longer describes the code (see the amendment at the head of this document).
> The terminal-`invalid` half below is current.

One enforcement point, one transaction, the `apply_label_txn` shape:

```
BEGIN IMMEDIATE
SELECT hypothesis FROM experiments WHERE experiment_id=?
if stored is not None and incoming != stored: raise HypothesisIsWriteOnce
UPDATE experiments SET hypothesis=? WHERE experiment_id=?
COMMIT
```

* Rejects `non-NULL → different value` and `non-NULL → NULL` (erasing a
  pre-registration is the same defeat as rewriting it). Accepts `NULL → value`
  and `value → identical value` (idempotent retry, the `upsert_turn_row`
  precedent at `:1930-1934`).
* A dedicated exception type, so `fix-bn1.5`'s `PATCH` returns a clear 4xx rather
  than a 500 — the bead asks for this explicitly.

**`invalid` is terminal, in SQL.** Revision 1 left two writers able to clear it:

```sql
UPDATE experiments SET status=?, completed_at=?, ...
 WHERE experiment_id=? AND status <> 'invalid'
```

…and `status`, `invalid_reason`, `invalid_detail` and `hypothesis` are all
excluded from `create_experiment`'s `ON CONFLICT DO UPDATE` set, so a resume
cannot launder `invalid` → `running` and then to `complete`. A 0-row update on
an `invalid` experiment raises rather than passing silently.

This is enforced **at the store**, not in the UI. The point of these fields is
that the record is trustworthy later, and the failure they must prevent —
reporting a score for a run that lost its evidence — is not an honest mistake.

`notes` is freely editable, by design and by contrast: one mutable description
beside one immutable one is what makes the immutable one mean something.

---

## 8. Scoring `[XR13]` `[XR14]` `[XR19]`

### 8.1 The verdict is written, not derived `[XR13]`

**Revision 1 derived the attempt verdict from turn columns. That was wrong, and
badly so.** The predicate was
`MIN(CASE WHEN status='completed' AND success=1 THEN 1 ELSE 0 END)`, and against
the workflow this epic targets:

* `TurnOutput.success` is `all(command_output.success ...)` (`turn.py:194-215`)
  over per-command flags that default to True, and `all([]) is True`.
* The only statuses reachable in this tree are `completed`, `awaiting_user` and
  `failed` on agent max-iters. `cancelled` and `abandoned` are dead vocabulary —
  nothing in `fastworkflow/` sets them — and a *real* cancellation is recorded as
  `completed` with `success=1`, because the cancelled `CommandOutput` never
  reaches the turn accumulator.

So for a tau2 retail attempt the predicate was true whenever the agent did not
exhaust its iterations and left no question hanging — **regardless of whether the
task was accomplished**. `pass@1` would have been ≈1.0 by construction. And the
one thing that *does* flip `success` to 0 is a tool that returned a failure the
agent then recovered from — so a normal, successful tau2 trajectory (probe a
wrong id, recover) scored as a *failed* attempt. The predicate was wrong in both
directions at once.

The ruling: **an attempt's outcome is an explicit row**.
`experiment_attempts.outcome` ∈ `{pass, fail, error, incomplete}` with a
`outcome_source` naming who decided (`tau2_reward`, `contract_evaluator`,
`operator`, or the literal `derived`). `reward` carries the grader's raw number
where it has one.

A derived fallback still exists, but it is **named for what it measures** and
never called `passed`:

```sql
-- diagnostic only: 'no command in this attempt reported a failure code'
SELECT experiment_id, task_id, attempt,
       MIN(CASE WHEN status='completed' AND success=1 THEN 1 ELSE 0 END)
         AS no_command_reported_failure
FROM turns WHERE experiment_id = ? GROUP BY experiment_id, task_id, attempt
```

A harness with no grader may write `outcome_source='derived'` from it — but that
is an explicit, recorded choice, not a silent default: `outcome_source` carries
the string `derived` into the attempt row, `experiment_scores()` returns the
distinct sources it saw, and the UI renders an explicit warning that a derived
outcome "reports only that no command returned a failure code" and "is not a
judgement that the task was accomplished".

**What is NOT enforced**, stated because a draft of this section promised it and
something weaker shipped: nothing refuses to score a workflow whose commands
never set `success=False`. That check needs a per-workflow notion of "reports
failure honestly" which this container does not have. The warning is visible;
the refusal is not implemented. See §12 item 6.

**Attempt lifecycle.** `start_attempt` writes the row before the first turn;
`finish_attempt` stamps `finished_at` and the outcome. An attempt with
`finished_at IS NULL` is unfinished, full stop — which is what makes the two
holes revision 1 had unreachable:

* An attempt that crashed after turn 3 of 10 has turn rows, so revision 1
  counted it as observed and could mark the experiment `complete`. Now it has no
  `finished_at` and cannot.
* §9.2's resume selector looked for pairs with "no terminal turn", so it skipped
  that attempt. Now it selects on `finished_at IS NULL`.

An unanswered `awaiting_user` attempt is `outcome='incomplete'`, written by the
harness. Revision 1's §9.2 claimed such an attempt was "counted as neither pass
nor fail" while its own SQL counted it as a fail; and its remedy, `cancel_pending()`,
writes no turn record at all, so the stored row would stay `awaiting_user`
forever. Both are corrected: the harness must call `finish_attempt(...,
outcome='incomplete', outcome_source='harness_timeout')`, and an experiment
containing an `incomplete` attempt cannot be `complete`.

### 8.2 The denominator is declared, and the store checks it `[XR14]`

`pass@1` is the fraction of attempts with `outcome='pass'`; `pass^k` is the
fraction of tasks all of whose attempts passed. Both are computed over
`declared_tasks × declared_attempts`, **never** over surviving rows — the same
argument `EvidenceRun` makes one layer down (`observability/evidence_run.py:5-9`).

`complete_experiment` **computes the verdict itself** in one `BEGIN IMMEDIATE`;
the caller may request completion or force `invalid`, and nothing else:

```
finished = SELECT COUNT(*) FROM experiment_attempts
            WHERE experiment_id=? AND finished_at IS NOT NULL AND outcome IS NOT NULL
segments_ok = NOT EXISTS(SELECT 1 FROM experiment_evidence_runs
                          WHERE experiment_id=? AND valid=0)
complete iff finished == declared_tasks * declared_attempts
            AND no outcome='incomplete'
            AND segments_ok
otherwise -> 'invalid' with invalid_reason in
            {attempt_shortfall, evidence_run_invalid, never_completed}
```

Revision 1 took `status` as a caller argument, which let the harness certify its
own completeness while §7.3 made exactly the opposite argument for the strictly
weaker `hypothesis` invariant.

**A score is reportable only when `status='complete'`.** `experiment_scores()`
returns per-task and per-attempt detail for a `running` or `invalid` experiment
but refuses a headline `pass@1`/`pass^k`, returning the status and reason in
their place. A provisional number in a UI becomes a quoted number in a document.

### 8.3 Arm comparison `[XR19]`

`compare_experiments(treatment, baseline)` returns per-task deltas and the two
flip sets (`fail→pass`, `pass→fail`). It reports flip counts and sample size; it
does **not** claim significance — the `fastworkflow-proof-and-analysis-toolkit`
recipes own that, and a query layer that emits a p-value is a query layer that
will be quoted as if it had run the protocol.

Refuses with a 409 unless **all** of:

* both experiments are `status='complete'`;
* their `declared_tasks`/`declared_attempts` match;
* their **DISTINCT `task_id` sets are equal** — the 409 body reports the
  symmetric difference. Revision 1 checked cardinality only, which admitted two
  15×3 runs over *disjoint* task sets and reported "0 regressions" for two runs
  sharing no task;
* their `capture_profile` and `capture_policy_version` match. Two arms captured
  under different profiles are not measuring the same columns.

### 8.4 One layer, three consumers

Built once against `experiment_id`, consumed by `fix-bn1.7`, by `fix-9eg.3`
(which formally depends on `fix-bn1.2` for the boundary it lacks), and by
`fix-sb8.10`'s corpus view.

---

## 9. Erasure, retention and the harness

### 9.1 Erasure and retention `[XR15]`

`PRAGMA foreign_keys` is off and the schema declares no `REFERENCES`, so nothing
cascades; the delete order is the consistency mechanism (`[DR44]`). An experiment
spans **many channels** by construction, which makes it the first object in this
DB not owned by a channel.

* **`forget_channel(channel_id)`** does not delete `experiments` rows — 44 of 45
  attempts may live elsewhere. In the same transaction, with the id set collected
  **before** any delete (`[DR44]`), it: deletes `experiment_attempts WHERE
  channel_id=?`, and sets `status='invalid'`,
  `invalid_reason='turns_erased'` (a closed code, §6.3) with the channel recorded
  in the scrubbed `invalid_detail`, on every experiment that had a turn in that
  channel. Because `invalid` is terminal (`[XR12]`), a later resume cannot undo
  it. This is the only honest outcome: after erasure the denominator is
  unreconstructable, and an erased experiment must never render a headline score.
* **`clear_conversations()`** deletes all three new tables outright, alongside the
  conversations, turns, spans, artifacts and six distillation tables it already
  deletes. Mid-run, the §7.2 `ExperimentNotFound` contract makes the running
  harness fail loudly on its next write rather than accumulate orphan labels.
* **`prune()`** exempts all three tables, following `turns` and `conversations`,
  which are already exempt from retention (`[R16]`). Pruning a container out from
  under attempts that still exist would produce exactly the orphan shape
  `[DR44]` prevents.

### 9.2 The harness `[XR18]` `[XR16]`

`fix-bn1.4` drives `WorkflowExecutionContext.process_turn(message)` (`:1188`)
directly, in threads, one WEC per attempt.

**Why not `ChatSession`** — restated, because revision 1's reason was wrong
(`ChatSession.start_workflow` *does* accept and forward `workflow_id_str`; the
folder-basename fallback is the CLI caller's choice):

* `fastworkflow.chat_session` is a **write-once process global**
  (`__init__.py:228-241`) that raises on a second set, so N attempts cannot each
  own one.
* `ChatSession` installs transport queues (`chat_session.py:130-136`), which
  switches `ask_user` to the Topology-A **blocking** path — a harness attempt
  would hang instead of returning `AWAITING_USER`.

**Independent of which vehicle drives it**, every attempt must pass a unique
`workflow_id_str` to `Workflow.create`, because a colliding id returns the
registry's existing live object and overwrites its context (`workflow.py:97-112`).
`run_fastapi_mcp/utils.py:566-694` (`_create_user_runtime`) is the assembly order
to copy.

**`[XR18]` One channel per attempt:** `channel_id =
f"exp:{experiment_id}:{task_id}:{attempt}"`, so attempts never share a
`SessionStateStore` pending slot and never share a topic namespace.

**`[XR16]` Two determinism contaminants, not one.** Both are blocking for
`fix-bn1.4`; either one alone makes `pass^k` measure nothing.

1. **The DSPy response cache.** On by default (disk + memory); `get_lm`
   (`utils/dspy_utils.py:9`) sets neither `cache=False` nor a rollout id. `k`
   repeated attempts on one task with the same prompt hit the cache and produce
   byte-identical trajectories, so `pass^k == pass@1` by construction and the run
   looks like a spectacular result. `get_lm` already forwards `**kwargs` to
   `dspy.LM`, but no caller threads anything through it, so **the reachable lever
   is process-wide**: `get_lm` reads an env var and passes `cache=False`. Revision
   1 offered "disable **or** per-attempt-vary"; the second option has no seam and
   is withdrawn. The harness sets that var by **merging into**
   `fastworkflow._env_vars`, never by calling `fastworkflow.init({...})`: `init`
   REPLACES the process env dict (`__init__.py:253`), so setting two keys through
   it deletes `LLM_AGENT`, every `LITELLM_API_KEY_*` and every threshold the run
   needs — and the failure surfaces as "DSPy Language Model not provided" from
   somewhere unrelated. (`fastworkflow/utils/dspy_cache_utils.py` must be fixed or
   deleted — a reader will reach for it first, and it has no callers and does not
   run: `clear_dspy_cache_completely` passes `enable_litellm_cache=` to
   `dspy.configure_cache`, which in the pinned dspy 3.3.0 accepts no such
   parameter and raises `TypeError`. Verified 2026-08-29.)
2. **The shared utterance/clarification cache.** `CommandNamePrediction.__init__`
   (`_workflows/command_metadata_extraction/intent_detection.py:286-294`) builds
   two paths under `<app_workflow_folderpath>/___convo_info`:
   `_get_cache_path(workflow_id, …)` is sharded per workflow id (so `[XR18]`'s
   unique `workflow_id_str` already isolates it), but
   `_get_cache_path_cache(convo_path)` (`:595-604`) is the **fixed literal**
   `cache.sqlite3`, keyed on nothing. It is read at `cache_match` (`:482`) and
   written at `store_utterance_cache` (`:577`) on the runtime turn path. So
   attempt 2 inherits attempt 1's disambiguation decisions — correlated attempts,
   which is exactly what `pass^k` must not have — and, worse, the **treatment arm
   inherits the baseline arm's**, because both run against the same workflow
   folder.

   `fix-bn1.4` must shard this file per attempt (give `_get_cache_path_cache` the
   `workflow_id` treatment its sibling already has, behind an env flag so ordinary
   runs keep their cache), and `fix-bn1.8` asserts that a second attempt of a task
   does not hit `matcher_layer="embedding_cache"` on a turn where the first hit
   `matcher_layer="classifier"`.

Whatever is chosen for both is recorded in the segment's `record_json`.

**Resume.** A crashed harness leaves `status='running'`. Resume opens a second
`evidence_run()` segment (§3.1), records it, and selects the attempts with
`finished_at IS NULL`. For each, `restart_attempt` deletes that attempt's
conversations and turns **in one transaction**, increments `restarts`, and
re-runs it under the same `(task_id, attempt)`. The deletion is deliberate: the
abandoned partial trajectory is evidence of nothing, `idx_conv_experiment_attempt`
is UNIQUE so a second conversation under the same labels is refused, and leaving
the rows would pin every resumed attempt's derived diagnostic to 0 forever. The
`restarts` counter makes a task that keeps crashing visible rather than silently
retried. (§12 item 6 records the alternative that was not chosen.)

**Other harness constraints**, verified and recorded so they are not
rediscovered:

* **The DSPy memory policy is the entry point's to install, not the library's**
  (`install_memory_policy=True`, default **False**). A 45-attempt agentic sweep
  in one process is exactly the shape that fills DSPy's retention structures, on
  a box `CLAUDE.md` already documents OOM-killing, and this is a place where a
  bare-WEC harness does **not** match FastAPI: `server_memory.install_policy` is
  called only from the server entrypoint. But it claims DSPy's config-owner
  **thread** process-wide and has **no uninstall**, so a library that called it
  on construction would poison every process that ever built a harness —
  including a test process, where the next FastAPI app's lifespan then fails its
  own `claim_async_owner()`. Observed exactly that way during implementation;
  hence opt-in. Either way the answer is recorded in the evidence segment, so a
  sweep that ran without the policy is visible in its own evidence rather than
  silently different from one that did.
* `fastworkflow.init()` installs **one process-global env dict**
  (`__init__.py:244-286`). Attempts in one process cannot vary `LLM_AGENT` or any
  other knob. **An arm that changes configuration needs a separate process** —
  which is also why `arm` and `baseline_experiment_id` are columns rather than a
  runtime switch.
* `CommandRouter._instances_cache` / `ModelPipeline._instances_cache`
  (`model_pipeline_training.py:378-393,469-485`) are unlocked dicts. Launching N
  cold attempts at once can build N copies of the same BERT pipeline. Warm the
  routers once before releasing the attempts (`chat_session.py:217-241`).
* A suspended turn returns `AWAITING_USER` and never blocks. See §8.1 for the
  outcome it must be given.
* `evidence_run()`'s in-process verdict requires the harness to hold the sink
  (`observability/evidence_run.py:265-280`); driving attempts against a spawned server silently
  downgrades the bundle to the persisted-health path.

---

## 10. Read API and UI `[XR9]`

### 10.1 The noun split

`/api/distillation/*` keeps **`run`** — one row per compared message, already
shipped (`[DR55]`). The experiment surface never uses that word:

| Route | Returns |
|---|---|
| `GET /api/experiments` | list: `experiment_id`, `description`, `status`, `arm`, `baseline_experiment_id`, declared vs finished attempt counts, `invalid_reason`, `capture_profile`, `created_at`. **No score**: computing one per row would be a query per experiment, and `status` plus the two counts already say whether a score exists and whether it is reportable. `/score` serves the number for one experiment. |
| `GET /api/experiment/<id>` | detail: `hypothesis`, `notes`, `arm`, `baseline_experiment_id`, every evidence segment with its `valid`/`problems`, `invalid_reason`/`invalid_detail` |
| `GET /api/experiment/<id>/tasks` | one row per `task_id` with its attempts' outcomes |
| `GET /api/experiment/<id>/attempts?task=<task_id>` | one row per attempt: `outcome`, `outcome_source`, `reward`, `restarts`, `channel_id`, `conversation_id`, timestamps. **Not turn keys**: those come from `GET /api/turns?experiment=&task=&attempt=`, which is the shipped route the UI already drills through, rather than a second projection of the same rows. |
| `GET /api/experiment/<id>/score` | §8; refuses a headline number unless `complete` |
| `GET /api/experiment/<id>/compare` | §8.3; 409 with the symmetric difference unless comparable |
| `PATCH /api/experiment/<id>` | `notes` only. `hypothesis` returns **409** with the write-once reason, never a 500 |

**Filtering extends the shipped route.** `GET /api/turns` gains `experiment=`,
`task=`, `attempt=` alongside its existing filters, with matching clauses in
`store.list_turns` (`:2365-2414`) — whose `SELECT` names its columns explicitly
(`:2406-2410`), so the new columns do not appear until they are added there.

**Both directions are reachable, without a second "run" concept:**

* experiment → distillation: `GET /api/distillation/runs?experiment=<id>`, one new
  query param on the shipped route, filtering `turn_key IN (SELECT turn_key FROM
  turns WHERE experiment_id=?)`.
* distillation → experiment: `GET /api/distillation/run/<run_id>` gains an
  `experiment` block resolved through its own `turn_key` — the shape `[DR55]` used
  for `retention`.

Every route 404s with a human-readable reason on a DB without
`FEATURE_EXPERIMENTS_V1` (`[DR29]`'s posture), and `/api/experiments` joins the
`store is None` early branch (`server.py:1014-1023`) returning
`{"experiments": []}`, so a cold start renders an empty state rather than
"observability DB not found".

`PATCH` is currently `_refuse_write` (`server.py:984`). Admitting it needs the
argument `[DR30]` made for `POST /api/distillation/verdict`: the invariant is that
recorded observability data stays read-only over HTTP (studio design §3.4, the
`[R5]`/`[R18]` access-control section — **not** `[R12]`, which is the
maintenance/vacuum ruling and is mis-tagged in `[DR30]` and in `server.py`'s own
comment), and `notes` is an annotation that cannot alter any span, turn, artifact
or score. Implemented through `ObservabilityStore.open_for_annotation` (`:1295`),
never a migrating handle. **A `do_PATCH` must repeat `_host_origin_allowed()` and
`_token_valid(query)` and carry its own path allow-list**, exactly as `do_POST`
does at `:870-893` — the server applies those gates per verb method, with no
shared chokepoint, so a `do_PATCH` written without them is an ungated
cross-origin write.

### 10.2 UI

Extends the existing rail rather than adding a second tree: experiment → task →
attempt is the same three-level grouping `groupTurnsByChannel`
(`index.html:736-746`) and `renderConversationGroups` (`:767-787`) already build.
(Both were removed as dead code on 2026-09-29, fix-hzux.13; the rail now renders
the server-built tree from `run_chatbot/navigation.py` in `renderHierarchy`,
`static/src/120-header-rail.js`.)
An attempt row links into the trace view through `openTurnInDebug()`
(`:3352-3371`; now in `static/src/260-chat-live.js`), which already tolerates the
async writer lag — not `selectTurn` directly.

Hard constraints, all asserted at the byte level by
`tests/test_run_chatbot_server.py`, not by a browser:

* **No `innerHTML`** (`:377`; now `test_page_never_uses_innerhtml`). Build nodes with `el()` / `createTextNode`.
* **No non-loopback URL anywhere in the file**, `https://` included (`:668-679`;
  now `test_spa_ships_as_package_data`).
* A new panel must be added to `setTopMode`'s explicit id list (`:2457-2469`;
  now in `static/src/230-mode-picker.js`) —
  it assigns `className` wholesale, so an unlisted panel never hides and any class
  set in the HTML on a listed element is wiped.
* `server.py` stays **stdlib-only** (`[R23]`, asserted at `:709`; now
  `test_server_module_is_stdlib_only`, plus a subprocess check that importing the
  chatbot server loads no dspy/litellm/fastapi/starlette/uvicorn).
* The SPA stays **one file** — `pyproject.toml:24` lists exactly
  `run_chatbot/static/index.html` as package data. Since 2026-09-30
  (fix-hzux.12) it is still served as one page with one inline script, but its
  source is ordered parts under `run_chatbot/static/src/` and `pyproject.toml`
  includes `run_chatbot/static/src/*`. A new part is a new file in that
  directory; `tests/test_run_chatbot_spa_parts.py` guards the assembly.

An **invalid experiment must be visually unmistakable**, and must show its
`invalid_reason`. The hypothesis renders read-only with an affordance saying *why*
— otherwise the first person who tries to edit it files a bug.

---

## 11. Decision log

| Ruling | Decision |
|---|---|
| `[XR1]` | `experiment_id` = `exp-<32 hex>`, distinct from `EvidenceRun.run_id`. Segments live in `experiment_evidence_runs`, one row each, `valid` a column. |
| `[XR2]` | `task_id` is caller-supplied, stable, required whenever `experiment_id` is set, never derived from `conversations.topic`. |
| `[XR3]` | `attempt` is a caller-supplied positive int, unique within `(experiment_id, task_id)`, **enforced** by a PRIMARY KEY and a UNIQUE partial index. |
| `[XR4]` | Labels denormalised onto `conversations` **and** `turns`; not onto `spans`. |
| `[XR5]` | No `SCHEMA_VERSION` bump. Columns in **both** the `CREATE TABLE` literal and a guarded `ALTER TABLE`; ALTER before the statement loop; `FEATURE_EXPERIMENTS_V1` via `_merge_schema_features`. |
| `[XR6]` | The whole experiment surface is **scrub-only**, on the `set_diagnostic` + `_POLICY_EXEMPT_TURN_COLUMNS` precedents, justified by a testable dataflow claim. **No `POLICY_PATH_*` constants** — the `spans.channel_id` convention. |
| `[XR7]` | *(reversed)* `task_id` is scrub-only, not policed: policing withheld nothing (`record_json` holds the plaintext), broke every equality lookup, and split the write order between two helpers. `attempt` is unpoliced because the chokepoint admits only a positive int, not because of SQLite's INTEGER affinity. |
| `[XR8]` | *(reversed)* Evidence provenance is a scrub-only child table, one row per segment. `opaque-payload` maps to **omit** under `evidence`, and a policed column cannot be appended to. |
| `[XR9]` | `run` stays distillation's noun. Cross-links are query params and a detail block on the shipped routes. `do_PATCH` repeats the host/origin and token gates. |
| `[XR10]` | `EvidenceRun`'s default prefix `run-` → `evr-`. `distillation._new_run_id()`'s output is untouchable. |
| `[XR11]` | Direct store methods, not the sink queue, on `[R13]` grounds. Synchronous and allowed to raise. |
| `[XR12]` | `hypothesis` write-once and `invalid` terminal, both in SQL, both excluded from the create upsert's `DO UPDATE` set. |
| `[XR13]` | The attempt verdict is a written row with a named `outcome_source`. The turn-derived signal is renamed `no_command_reported_failure` and is a diagnostic, never a score. `finished_at` is the completion marker. |
| `[XR14]` | `declared_tasks`/`declared_attempts` are `NOT NULL` and required; `complete_experiment` **computes** the verdict, the caller may only request completion or force `invalid`. |
| `[XR15]` | `forget_channel` invalidates terminally and deletes its own attempt rows; `clear_conversations` deletes; `prune` exempts; every write method raises `ExperimentNotFound` on a 0-row update. |
| `[XR16]` | Two determinism contaminants — the DSPy response cache (process-wide flag; the per-attempt option has no seam) and the shared `___convo_info/cache.sqlite3` utterance cache (must be sharded per attempt). Both blocking for `fix-bn1.4`. |
| `[XR17]` | Labels bind through `bind_observability_identity`, type- and value-validated there. `channel_id` semantics unchanged. |
| `[XR18]` | One channel per attempt; one WEC per attempt; a unique `workflow_id_str` whatever drives it. |
| `[XR19]` | Comparability requires equal task-id **sets**, equal declarations, and equal `capture_profile`/`capture_policy_version`. |
| `[XR20]` | A policed column may never be read-modify-written. |

---

## 12. What this gate does not settle

1. **There is no configuration surface for a `CaptureFieldPolicy`.**
   `resolve_capture_policy()` (`:232-245`) passes no `field_policies`, so the six
   shipped `POLICY_PATH_*` constants are unreachable by any deployment today.
   `[XR6]` sidesteps it by declaring none, and records that re-admitting the
   experiment surface to the policy is a code change. The gap is `fix-ajv`'s.
2. **Top-level `TurnResult` fields are not withheld by the capture pipeline.**
   `_apply_capture_policy` walks only `turn_output.command_outputs`, so
   `user_message`, `answer` and now `task_id` appear in plaintext in
   `record_json` regardless of what the column beside them says. `[XR7]` rests on
   this being true; if `fix-ajv` closes it, `[XR7]` should be revisited.
3. **What "passed" means for a real benchmark.** `[XR13]` gives the container a
   place to store a verdict and a name for who decided it; it does not supply a
   grader. tau2's reward function is `fix-bn1.4`'s consumer's problem.
4. **Cross-process experiments.** `fastworkflow.init()`'s global env dict and
   `[XR16]`'s process-wide cache flag together mean an arm that varies
   configuration needs its own process, and nothing here coordinates two
   processes writing one experiment. The schema permits it
   (`experiment_evidence_runs` is a list); no code does it.
5. **Retention of a very large experiment.** `[XR15]` exempts the three tables,
   and turns are already exempt from the size cap (`[R16]`) — so the *denominator*
   is safe. What a long sweep can lose is its own early **spans**, i.e. trace
   detail, while its rows and score survive: the `[DR52]` pinning problem one
   level up, unsolved here.
6. **A derived score is warned about, not refused.** §8.1 records that the
   turn-derived fallback is meaningless for a workflow whose commands never
   report failure. The UI and the score payload both say so; nothing prevents
   it.
7. **Restart destroys the abandoned trajectory.** `restart_attempt` deletes the
   crashed attempt's conversations and turns. The alternative — allocating a
   fresh `attempt` and marking the abandoned one — keeps the partial trajectory
   but makes `declared_attempts` arithmetic no longer exact and the UNIQUE index
   no longer expressible at this grain. If a post-mortem of crashed attempts ever
   matters, that is the trade to reopen.

---

## 13. Review record

Four review passes across six lenses (correctness, capture policy,
concurrency/harness, scoring semantics, erasure/retention, scope/precedent). 46
findings were raised; each was handed to an independent verifier instructed to
**refute** it and to default to rejection when the claim could not be confirmed
from source. **38 survived; 8 were refuted** and are not recorded here.

**The eight blocking findings, and what happened to each:**

1. *The six denormalised columns were specified only as `ALTER TABLE`.* On a
   fresh DB the guarded ALTER is skipped by design and `CREATE INDEX
   idx_turns_experiment` raises `no such column` — reproduced against a real
   SQLite file, in both branches. §4 now states both halves; the shipped
   `distillation_pass` precedent has always been both.
2. *`provenance_json` classified `opaque-payload` and policed* would be **omitted**
   under `evidence`, erasing `valid`/`problems`/`archive` under the one profile an
   evidence run uses. Reversed → `[XR8]`, child table, scrub-only.
3. *A policed column cannot be read-modify-written*, so appending segment 2 was
   impossible under any classification. Generalised into `[XR20]` and designed
   away by the child table.
4. *Policing `task_id` withheld nothing* — the plaintext is in `record_json` —
   *and broke every equality lookup.* Reversed → `[XR7]`.
5. *One channel per attempt does not give attempt independence*: the
   `___convo_info/cache.sqlite3` utterance/clarification cache is keyed on
   nothing and is read and written on the turn path, so attempt 2 inherits
   attempt 1's disambiguations and the treatment arm inherits the baseline's.
   Added as the second half of `[XR16]`.
6. *The derived pass predicate was pass-by-default* for the workflow this epic
   targets, and inverted for the one case that flips `success`. §8.1 rewritten;
   the predicate is demoted to a diagnostic and renamed.
7. *§12 said the container "stores per-attempt outcomes" and the DDL stored none.*
   `experiment_attempts` added; it also closed the completion-marker and
   resume-selector gaps.
8. *The declared denominator checked existence, not completion*, so an attempt
   that crashed mid-way counted as observed, was skipped by resume, and could
   score as a pass. `finished_at` + a store-computed verdict (`[XR14]`).

**What else changed as a result:** `invalid` made terminal in SQL; `declared_*`
made `NOT NULL` and required; comparability tightened from cardinality to task-id
set identity plus capture profile; `invalid_reason` split into a closed code and
a scrubbed append-only detail; the `POLICY_PATH_EXPERIMENT_*` constants dropped as
inert; the `[R14]` justification for `[XR11]` replaced with `[R13]`; the
`ChatSession` rejection restated on the two reasons that hold; the DSPy
memory-policy constraint added; the `[R12]` mis-tag on the read-only-over-HTTP
invariant noted; and seven file:line citations corrected.

### Round 2: the implementation

A second adversarial pass ran against the shipped code across eight lenses
(store correctness, concurrency, schema/migration, plumbing, HTTP API, SPA,
design conformance, test quality). It was **partial**: the verification stage
was cut short by a session limit, so six findings were independently verified
and the rest are recorded as reviewer claims that were acted on but not
double-checked by a second agent. Stated plainly rather than presented as a
clean bill of health.

The three that mattered most, all fixed:

1. **`_prepare_process` wiped the process configuration.** It called
   `fastworkflow.init({FW_LM_CACHE: "0", ...})`, and `init` REPLACES the env
   dict — so a real run would have lost `LLM_AGENT` and every API key and failed
   with "DSPy Language Model not provided" from somewhere unrelated. The tests
   missed it because they fake the command executor and need no LLM. Now merged
   into `fastworkflow._env_vars`, with a regression test that asserts the
   configuration survives.
2. **`list_turns` projected the three new columns unconditionally**, so the base
   `/api/turns` route — the whole point of the debug UI — raised
   `no such column: experiment_id` on any DB this build had not migrated, which
   is exactly the read-only viewer case `[R12]` exists for. Now feature-gated,
   with the `list_turns` case added to the degradation test.
3. **The harness installed the DSPy memory policy on construction**, which
   claims DSPy's config-owner thread process-wide with no uninstall and broke
   `tests/test_fastapi_turn_output_contract.py` outright. Now opt-in (§9.2).

Also fixed: `create_experiment` could rewrite a completed experiment's declared
denominator; `_ensure_observability_conversation`'s blanket `except` swallowed
the UNIQUE-index refusal the index exists to raise; `complete_experiment` wrote
`invalid_detail` on the `complete` branch; `restart_attempt` left the deleted
turns' distillation closure orphaned; `reserve_turn_ordinal` could mint an
unlabelled attempt conversation on the degraded path; both binding paths
mutated before validating; the SPA's `api()` discarded the 409 body that the
compare refusal exists to deliver, making `renderComparison`'s incomparable
branch dead code; `PATCH /api/experiment/<id>/<anything>` was a silent alias for
the notes route; `dspy_cache_utils` still could not run; and the
`/api/distillation/runs?experiment=` cross-link §10.1 promised was missing in
both directions. Four tests were strengthened, including one that claimed to
assert a 409 and never did.

### Round 3: the implementation, completely

Round 2 was cut short. Round 3 re-ran the whole thing against the shipped tree:
**8 lenses, 41 findings, every one independently verified by an agent instructed
to refute it.** 24 were upheld, 17 refuted (several of those because round 2 had
already fixed them). No agent failed. This is the complete pass.

One finding was reported independently by three lenses and is the most serious
defect the epic produced:

> **`complete_experiment` compared a COUNT against the declared denominator
> rather than checking the declared SHAPE.** A resume whose task list gained two
> tasks and lost one reaches `finished == expected` with a declared task that
> never ran, so the experiment was stamped `complete` — and `experiment_scores`
> then divided 4 scored attempts by a declared denominator of 3 and returned
> **`pass@1 = 1.33` with `reportable: True`**. Reproduced directly before fixing.
> Completion now requires every row finished, exactly `declared_tasks ×
> declared_attempts` rows, and exactly `declared_tasks` distinct tasks; and
> `experiment_scores` refuses outright when the rows do not match the
> declaration, because a score that can exceed 1.0 is worse than no score.

The other upheld findings, all fixed:

* **The evidence gate discarded `flush()`'s durability bool.** `_apply_batch`'s
  generic failure arm rolls back a whole batch, counts one `write_error` and
  requeues nothing — while `records_dropped` stays 0 and `evidence_valid` reads
  only the drop counters. A run that lost an entire batch of turn records, or
  whose writer thread died, certified itself zero-drop. False from the barrier is
  the only signal, and it is now a recorded problem.
* **`_warm_routers` warmed the wrong cache.** It called
  `RoutingRegistry.get_definition`, which fills `_definitions`; the caches that
  matter are `CommandRouter`/`ModelPipeline._instances_cache`, whose `__new__`
  publishes an uninitialised instance, so N cold threads each ran the full
  constructor — N concurrent TinyBERT+DistilBERT loads, the exact spike the
  docstring claimed to prevent.
* **A closed experiment accepted new attempts.** Re-invoking a driver script
  that pins its `experiment_id` silently overwrote every stored verdict of a
  `complete` run whose numbers had been quoted. `run()` had no guard; the store
  now refuses (`ExperimentIsClosed`).
* **Segment invalidity was rewritable**: a segment recorded `valid=0` could be
  replaced by `valid=1` at the same seq. Now monotone, like `status <> 'invalid'`.
* **A re-create under a different capture profile silently kept the first one**,
  which is the column `compare_experiments` gates on. Now refused
  (`CaptureRegimeChanged`).
* **The migration guard keyed on one column of three**, so an interruption
  between ALTERs left a state it could never repair (DDL autocommits). Now
  per-column and self-healing.
* **`experiment_id` was scrubbed on the label routes and stored raw in the
  container tables** — scrubbing a join key in some tables and not others is what
  makes a join silently return nothing. It is now uniformly raw, like every other
  machine-minted join key in the file; `task_id` is scrubbed on all four routes,
  including `_assign_ordinal`, which the queued path reaches without the caller's
  scrub.
* HTTP: ids are now percent-decoded (after the sub-path split, so `%2F` cannot
  invent a segment); a missing experiment on `/compare` 404s instead of 400ing
  about a baseline; a non-integer `attempt` is rejected rather than silently
  dropped, which had *widened* the filter.
* SPA: the ≥500-turn guard was dead code — its warning was wiped by the very next
  line — and is now a stop with an explicit choice; the three experiment views
  and `selectTurn` now hold a navigation token so a slow response cannot repaint
  a view the user left; the attempts breadcrumb shows the label rather than the
  raw id; and the list says when it is truncated instead of silently showing 200.

**Six tests were strengthened.** The two flip-expectation tests asserted only
`> 0` and now pin 0.984 and 0.375; the erasure test used one channel, which could
not tell a channel-scoped delete from an experiment-scoped one, and now uses two;
a signature lint that claimed to prove `[XR6]`'s dataflow property was replaced
by a behavioural test that drives a real attempt whose command raises with a
credential and whose grader returns one; the aggregate test never issued an HTTP
request; the crash test wrote no turns despite its docstring; and the closed
vocabularies, the `outcome_sources` surface, the positive feature-detection arm,
and the routes the SPA actually calls had no coverage at all.

**Two doc-vs-code mismatches were fixed in the doc**, because the code was right:
`GET /api/experiments` carries no per-row score (one query per row, and status
plus the counts already say whether a score exists), and
`/api/experiment/<id>/attempts` does not project turn keys (those come from the
shipped `/api/turns` filter the UI already drills through).

**What survived a direct attack:** `[XR1]` (separate `experiment_id`), `[XR2]`,
`[XR4]` (not on spans), `[XR5]`'s no-bump inheritance, `[XR6]`'s scrub-only
conclusion (though not revision 1's constants), `[XR9]`'s noun split, `[XR10]`,
`[XR11]`'s conclusion, `[XR12]`'s write-once mechanism, `[XR18]`'s one-channel
rule, and §2's rejection of `channel_id` as the label.

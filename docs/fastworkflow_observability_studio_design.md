# fastWorkflow Observability & Studio — Design Doc (v2)

Status: **REVISED after adversarial review round 1 (28/28 findings resolved)** —
implementation may begin per §7 phasing.
Author: Claude (with Dhar Rawal), 2026-08-25 (v1); revised 2026-08-25 (v2).
Baseline verified against: v3.1.2 (main @ 27e8d91).
Review artifacts: `docs/fastworkflow_observability_studio_design_review.md`
(findings R1–R28 + resolution log); rulings recorded in §9 Decision Log.
Prior art this document implements/reconciles: `docs/turn_result_design_final.md`
§1, §11, §14, §15; `docs/fastworkflow_turns_async_execution_design.md` (fix-85g);
`docs/agent_mode_live_command_traces.md`.

Where this document conflicts with `turn_result_design_final.md`, that spec's
decisions stand unless a §9 ruling explicitly supersedes them with rationale.

---

## 0. Origin: the problem

fastWorkflow has no end-to-end observability of a user turn. A developer
debugging "why did the agent do that?" must stitch together five fragmented,
lossy surfaces:

| Surface | Loss mode |
|---|---|
| `action.jsonl` (CLI cwd mirror) | deleted at every turn start — only ever holds the LAST turn |
| FastAPI `traces` response field | destructively drained — read-once (fix-85g.10) |
| `CommandTraceEvent` queues | in-memory; silently no-op'ed for queue-less embedders |
| Train-time model metrics | stdout only, never persisted |
| CLI conversation view | shows neither past conversations nor artifacts |

There is also no manual-testing UX beyond the CLI.

## 1. Decisions (owner, 2026-08-25; review rulings in §9)

- **D1 — Single source of truth.** One store holds turn records, spans, and
  artifacts — the `ConversationTurnStore` role from the final spec, realized on
  SQLite, reached via the staged consolidation in §3.3.
- **D2 — End-to-end traceability with fastWorkflow's design abstractions
  observable**: planner, executor (agent loop + command execution), context,
  ask_user — as first-class named spans.
- **D3 — v1 instrumentation depth: agent↔workflow boundary only.** NLU/LLM
  internals are v2; the v1 schema models them so v2 adds emitters, not
  migrations (schema-touching changes are governed by the §3.2 versioning rule).
  *Amendment (owner ruling, 2026-08-26, fix-kw7.15):* the NLU emitters land —
  `fw.nlu.intent` (per prediction attempt: context, stage, matching layer
  [exact prefix / fuzzy pre-match / embedding cache / classifier], classifier
  confidence + ambiguity threshold + model tier via
  `CommandRouter.predict_with_details`, candidate set and discarded escalation
  labels on an ambiguity) and `fw.nlu.param_extraction` (extraction method,
  NOT_FOUND retry round, structured missing/invalid fields, per-field
  `db_lookup` events with the three-state outcome, and the
  `validate_extracted_parameters` verdict — via an additive `diagnostics`
  out-param on `validate_parameters`). The root `fw.turn` span additionally
  records `context_mutations` (shallow diff of the app workflow context across
  the turn). Purpose: make conversation logs a first-class diagnosis surface
  for coding agents (the shipped `debug-workflow-conversations` skill is the
  consumer contract). The sink is reached via a `tracing.host_scope`
  ContextVar bound in `CommandExecutor.invoke_command` — still the WEC, never
  a transport queue [R28]. `fw.llm.call` landed in the same window via a DSPy
  callback emitter (`utils/dspy_logger.DSPyObservabilityCallback`: one span
  per LM invocation with module chain, messages, usage/cost, and `cache_hit`);
  only `fw.train.*` remains reserved.
- **D4 — No observability-platform dependency.** OTel-aligned records in
  stdlib-SQLite; export is an external translation script (§3.2 translation
  table); the framework never imports an OTel SDK.
- **D5 — One UI ("Studio") for trace debugging and manual testing.**
- **D6 — Process**: T2 gate. Round-1 adversarial review completed 2026-08-25;
  further schema/system-of-record changes re-enter review.

## 2. Baseline reality (v3.1.2)

- `state_paths.py`: single absolute state root, per-workflow namespace
  `<root>/workflows/<workflow-id>/`. The observability DB slots in here.
- SQLite is the house storage tech (v2.31.0); per-channel
  `conversations/<channel_id>.sqlite3` conversation stores exist (FastAPI only).
- v3.0 shipped the wire collapse (singular `command_response`); the spec's
  `stores/` package, `turn_serializer.py`, `metrics.py` remain unbuilt —
  this design builds their observability-relevant subset.

## 3. Architecture: flight recorder → black box → viewer

```
┌──────────────────────────────────────────────────────────────┐
│ fastWorkflow core (per process)                              │
│  WEC.process_turn ─┬─ planner ─ agent loop ─ command exec    │
│                    └────────── TraceSink protocol ────────── │
│                     (no-op default; SQLite sink per §5)      │
└────────────────────────────────┬─────────────────────────────┘
                                 │ background writer (two queues)
                                 ▼
        <state-root>/workflows/<workflow-id>/observability.sqlite3
                                 │ read-only, token-gated
                                 ▼
│ fastWorkflow Studio (SPA):  debug mode │ test mode │
```

### 3.1 The flight recorder: `TraceSink` + identity plumbing + span emission

```python
class TraceSink(Protocol):                       # all methods never raise to caller
    def emit_span(self, span: Span) -> None: ...
    def emit_turn_record(self, record: TurnRecord) -> None: ...
    def record_conversation_label(self, channel_id, conversation_id,
                                  topic, summary) -> None: ...   # [R15]
```

**Identity plumbing `[R1]`.** The embedder binds `channel_id` and
`conversation_id` onto the WEC **before** the turn (new bind/begin-turn
parameters); conversation-id reservation moves ahead of the turn, and from
Phase A the observability DB is the sole id-minting authority (the old store
consumes the same ids during dual-write). WEC stamps
`channel_id`/`conversation_id`/store-assigned `ordinal` onto `TurnResult`.
**CLI identity `[R17]`:** the CLI binds a synthetic channel
(`cli:<session-start-timestamp>`) and mints a new conversation per `//new`.
*Amendment (2026-08-26):* WEC mints its own conversation id at turn start when
no embedder bound one, so bare embedders group like every other caller and the
turns-first view they needed is no longer required. A turn stays
conversation-less only when minting itself fails; those rows remain reachable
through their channel's conversation-less group.

**v1 emission sites** (D3: agent↔workflow boundary only):

| Span name | Where emitted | Key attributes |
|---|---|---|
| `fw.turn` (root) | `process_turn` begin/finalize | turn_key, channel_id, conversation_id, user_message, status, success, failure_reason, suspended_ms |
| `fw.planner.plan` / `.replan` | around the task-planner calls (`workflow_agent.py`) | plan text (capped), replan trigger, model name |
| `fw.agent.tool_call` | the CommandTraceEvent emission sites — **outside** the `command_trace_queue is not None` guards; the sink is reached via WEC/config, not the transport-queue contract `[R28]` | raw agent command text |
| `fw.command.execute` | `CommandExecutor.invoke_command` boundary | context, command_name, parameters, response text (capped), success, duration_ms |
| `fw.ask_user` | ask_user suspend/resume (A7 semantics) | agent_query, user_response, human wait |

**Cross-process span identity `[R6]`.** Root and ask_user span ids are
deterministic — `span_id = hash(turn_key + span_name + attempt)` — and span
close is an idempotent upsert of `end_ns`/`status`, so a turn suspended in
process A closes cleanly in process B after rehydration/restart.

*Amendment (owner request, 2026-08-26) — the agent loop's own structure.* Two
names join the emitted taxonomy so the levels a developer reasons in stop being
inferred:

| Span name | Where emitted | Key attributes |
|---|---|---|
| `fw.agent.execute` | `WEC._call_agent_with_retry` — the one choke point both the fresh forward and the resume pass through | agent_input, resumed, model, attempts, final_answer, suspended, clarification, exhausted |
| `fw.agent.step` | each iteration of `fastWorkflowReAct._run_loop` | step_index, thought, tool_name, tool_args, observation, clarification, tool_error |

`fw.agent.execute` is the executor as a phase, sibling to `fw.planner.plan`
under `fw.turn`; it is **not** `fw.command.execute`, which is one command inside
one tool call. `fw.agent.step` parents both the ReAct reasoning `fw.llm.call`
and the `fw.agent.tool_call` it produced, which is the association that
previously existed only as DSPy module names plus timestamps. ReAct's extract
call stays a child of `fw.agent.execute` and not of any step: it runs inside
`forward()` after the loop and before the agent returns, so it is the tail of
execution even though its output becomes the turn's answer verbatim.
`_call_agent_with_retry` binds `tracing.host_scope`, so the step spans — several
frames below the WEC — reach the sink without a transport queue `[R28]`. A
suspended step closes with `awaiting_user` rather than staying open, so nothing
leaks onto the parenting stack across a resume in another process. Studio reads
these spans when present and keeps its pre-amendment reconstruction (module
chain plus sibling ordering) only for turns recorded before them.

Reserved-for-v2 names as originally scoped: `fw.nlu.intent`,
`fw.nlu.param_extraction`, `fw.llm.call`, `fw.train.*`. *Per the D3 amendment
(§1), the first three now EMIT* — the NLU pair inside the CME pipeline, and
`fw.llm.call` at the DSPy callback level (cache hits produce spans with
`cache_hit=true`); only `fw.train.*` remains reserved. `CommandTraceEvent`
gains `turn_key` (additive); live CLI trace rendering is unchanged.

### 3.2 The black box: `observability.sqlite3`

One DB per workflow at `state_paths.observability_db()`. Schema (v1):

```sql
PRAGMA user_version = 1;          -- schema version [R11]
PRAGMA auto_vacuum = INCREMENTAL; -- set at creation, before any table [R12]

CREATE TABLE conversations (
  channel_id TEXT NOT NULL, conversation_id INTEGER NOT NULL,
  topic TEXT, summary TEXT, status TEXT, next_ordinal INTEGER,   -- [R11]
  started_at TEXT, last_turn_at TEXT,
  PRIMARY KEY (channel_id, conversation_id));

CREATE TABLE turns (
  turn_key TEXT PRIMARY KEY,      -- trace key (translation table below)
  channel_id TEXT NOT NULL, conversation_id INTEGER, ordinal INTEGER,
  user_message TEXT NOT NULL, refined_user_message TEXT,
  entry_workflow_name TEXT, entry_context TEXT,
  status TEXT NOT NULL, success INTEGER NOT NULL,
  failure_reason TEXT, answer TEXT,
  conversation_summary TEXT, conversation_traces TEXT,           -- [R3]
  started_at TEXT, completed_at TEXT, suspended_ms INTEGER,
  continuation_of TEXT, record_version INTEGER NOT NULL,
  record_json TEXT NOT NULL);

CREATE TABLE feedback (           -- mutable by design [R3]
  turn_key TEXT PRIMARY KEY, feedback_json TEXT NOT NULL,
  updated_at TEXT NOT NULL);

CREATE TABLE spans (
  span_id TEXT PRIMARY KEY, trace_id TEXT NOT NULL,   -- = turn_key
  parent_span_id TEXT, name TEXT NOT NULL,
  kind TEXT NOT NULL,             -- internal | llm | human_wait | tool
  channel_id TEXT,                -- erasure join key [R21]
  command_name TEXT, context TEXT,                    -- queryable [R27]
  start_ns INTEGER NOT NULL, end_ns INTEGER,
  status TEXT NOT NULL, attributes TEXT NOT NULL);

CREATE TABLE artifacts (
  artifact_id TEXT PRIMARY KEY, turn_key TEXT NOT NULL,
  channel_id TEXT,                                    -- [R21]
  span_id TEXT, key TEXT NOT NULL, content_type TEXT,
  size_bytes INTEGER, sha256 TEXT,                    -- hash of the ORIGINAL value
  inline_value BLOB, error TEXT);

CREATE TABLE train_runs (
  run_id TEXT PRIMARY KEY, workflow_fingerprint TEXT, started_at TEXT,
  completed_at TEXT, metrics_json TEXT NOT NULL);

CREATE INDEX idx_spans_trace   ON spans(trace_id);            -- [R27]
CREATE INDEX idx_spans_command ON spans(command_name) WHERE command_name IS NOT NULL;
CREATE INDEX idx_turns_conv    ON turns(channel_id, conversation_id, ordinal);
CREATE INDEX idx_turns_status  ON turns(status);
CREATE INDEX idx_artifacts_turn ON artifacts(turn_key);
```

**Schema versioning `[R11]`:** `PRAGMA user_version` stamped at creation;
readers refuse DBs newer than they know; writers migrate forward. Row payloads
keep strict `record_version` dispatch per the final spec §6.

**Turn-row lifecycle `[R2]`:** INSERT at first emission (e.g.
`awaiting_user`); exactly one guarded status-transition UPDATE to a terminal
status (`completed`/`failed`/`abandoned` — including the spec's ABANDONED
filing of stale suspensions). Write-once applies to rows already terminal;
identical-content retry verifies and claims idempotent success.

**Size policy — one mechanism `[R10]`:** at serialization time, any artifact
value over `FW_OBS_INLINE_ARTIFACT_BYTES` is replaced **inside `record_json`**
by a placeholder/ref envelope (per final spec `[A10]`); the `artifacts` table
is the only value holder. Individual span attributes are capped at
`FW_OBS_MAX_ATTR_BYTES` (truncation is lossy-and-counted: truncated attrs
carry `truncated: true` + original length + sha256 — no silent truncation).
`record_json` stores the internal `TurnResult` (full capture is the
observability value), post-envelope and post-redaction.

**Redaction `[R20]`:** a sink-boundary pass scrubs known credential shapes and
the values of every loaded `*_API_KEY`/`*_TOKEN` env var from all persisted
text; traceback artifacts persist only when `FW_OBS_CAPTURE_TRACEBACKS=1`.

**Writer discipline `[R7][R8][R13]`:**
- **Two queues:** a small dedicated turn-record/feedback queue
  (bounded-timeout put, then drop-with-log — the only case a turn record may
  drop in v1) and a droppable span/artifact queue (`FW_OBS_QUEUE_MAX`). This
  replaces the unsatisfiable never-block/never-drop combination.
- Connection: WAL, `synchronous=NORMAL`, `timeout=30.0` (house precedent:
  `kvstore.py`), `BEGIN IMMEDIATE`, short batched transactions; SQLITE_BUSY on
  a turn record → requeue with bounded retries (spans: drop-and-count).
  Multi-process writers are supported on **local filesystems only** — the
  state root must not be NFS (documented; WAL constraint).
- Lifecycle: daemon writer plus an explicit `close()` (sentinel enqueue,
  bounded join, final commit) wired into the CLI exit path and `atexit`, so
  the last turn of a session is never lost.
- Writer health is user-visible: writer errors/drops are recorded in a
  self-diagnostic row in the DB and surfaced as a Studio banner (not only in
  the unbuilt metrics counter) `[R13]`.
- Durability classes `[R14]`: v1/Phase A — everything is best-effort as
  above. **Phase B gate:** turn-record and feedback writes become
  synchronous-or-acked in the turn path (honoring the turns engine's
  persist-before-DONE); spans/artifacts stay fire-and-forget.

**File posture and capture default `[R4]`:** DB directory 0700, DB files 0600,
set explicitly. `FW_OBSERVABILITY` defaults ON only under fastWorkflow's own
entry points (`fastworkflow run`, `studio`, `run_fastapi_mcp`, `train`);
library embedders are opt-in (`FW_OBSERVABILITY=1`).

**Erasure `[R21]`:** `channel_id` on spans/artifacts plus a first-class
forget-channel operation (store API + `fastworkflow studio --forget-channel
<id>`) that deletes across all tables, then `PRAGMA
wal_checkpoint(TRUNCATE)` + `incremental_vacuum`.

*Amendment (2026-08-26):* the `--forget-channel`/`--prune` CLI flags were
retired with the chatbot UI rework. Erasure is the store API plus the
chatbot's confirmed **Clear conversations** action (which also deletes the
legacy per-channel DB files, Phase-7 ruling C1); pruning runs automatically
at sink startup only. The R21/R12 mechanisms themselves are unchanged.

**Maintenance ownership `[R12]`:** opportunistic bounded prune at sink
startup + `fastworkflow studio --prune`; `incremental_vacuum` in the same
path; the `-wal` sidecar counts toward `FW_OBS_DB_MAX_BYTES`; Studio's read
layer uses per-request connections (no held cursors) so checkpointing never
starves.

**OTel translation table `[R26]`** (the export script's contract — records
are OTel-*aligned*, not wire-conformant): `trace_id` = first 16 bytes of
SHA-256(turn_key) (reverse mapping kept by the script); `span_id` = first 8
bytes of SHA-256(stored span_id); `kind` maps `internal|llm|tool` →
`SpanKind.INTERNAL`/`CLIENT` + `gen_ai.*` attributes, `human_wait` →
INTERNAL + `fw.human_wait=true`; TEXT timestamps parse RFC3339.

### 3.3 Single source of truth: consolidation path (D1)

- **Phase A (v1):** turn records + spans written to `observability.sqlite3`;
  the new DB mints conversation ids `[R1]`; the FastAPI labeling hook calls
  `record_conversation_label` so topic/summary land in the new DB from day
  one `[R15]`; the old conversation store keeps running (dual-write,
  explicitly temporary), consuming the same ids so the stores cannot diverge
  on identity.
- **Phase B** — LANDED in v3.2.0 (`fix-24f.8`); slice design and its deviations
  in `docs/observability_phase7_consolidation_design.md`. Gates, all normative:
  1. Memory rebuild reads `conversation_summary`/`conversation_traces`/
     `feedback` from the new store `[R3]`.
  2. Turn-record/feedback writes go synchronous-or-acked `[R14]`.
  3. Retention exempts conversation rows + turn records — pruning applies to
     spans/artifacts only; conversation pruning is operator opt-in `[R16]`.
  4. Insights distillation is ported off `action.jsonl` onto the in-process
     `ctx.action_log` (no file, no writer race) and the diagnostics skill's
     `trace_turn.py` is updated `[R25]`.

  `run_fastapi_mcp/conversation_store.py` and the `action.jsonl` mirror are
  gone; the module was deleted under bead `fix-gxr` and its tests ported to
  store equivalents. The per-channel DB **files** are left untouched on disk
  (readable by older builds, erased with the channel by `run_forget_channel`).

### 3.4 The viewer: fastWorkflow Studio

Static SPA + a small localhost read-only HTTP layer, launched by
`fastworkflow studio <workflow_path>`.

**Access control `[R5][R18]`:** a per-launch random bearer token embedded in
the printed URL (Jupyter pattern) is required on every API call, plus a
strict Host/Origin allowlist (`127.0.0.1:<port>`, `localhost:<port>`). These
are invariants, not implementation details.
*Amendment (2026-08-25, recorded post-implementation):* the allowlist admits
any **loopback** authority — `127.0.0.1` / `localhost` / `[::1]`, on any
port — and nothing else. Pinning the exact port broke legitimate access
through port forwarders (WSL localhost relay, IDE port forwards), whose
re-exposed local port lands in the browser's Host header. Loopback-only is
the property that defeats DNS rebinding (a rebound request carries the
attacker's hostname); the bearer token remains the authentication on every
request, so a loopback origin without the token still gets 401. The absorbed server endpoint
`GET /turns/{turn_key}` authorizes the record's channel against the caller's
JWT — no bare-handle reads, per final spec `[A39]`.

**Rendering safety `[R22]`:** all record-derived text renders as text (no
innerHTML); HTML-ish artifacts render only inside sandboxed iframes with
`default-src 'none'`; the SPA carries a restrictive CSP (`connect-src
'self'`); the read layer uses parameterized queries only.
*Amendment (Phase 5, recorded post-implementation):* test mode requires the
chat pane to call the local FastAPI server, so the shipped CSP widens
`connect-src` to `'self' http://127.0.0.1:* http://localhost:*` — loopback
origins only, never a routable host; scripts remain sha256-hash-sourced with
no `unsafe-inline` (`style-src` alone allows inline styles for the page's own
stylesheet block).

**Spawned-server posture `[R19]`:** a Studio-started FastAPI server always
gets `--host 127.0.0.1`, CORS pinned to the Studio origin, and refuses to
start in unsigned-JWT mode without an explicit CLI flag; any wider bind is a
command-line decision, never a UI one.
*Amendment (owner ruling, 2026-08-25, recorded post-implementation):* the
server is now spawned **automatically** whenever a workflow is known
(`--server-port` opts out and names an existing server), and the auto-spawned loopback server runs unsigned
dev JWTs **by default** — the chatbot mints its own tokens via `/initialize`,
the bind stays 127.0.0.1-only, and CORS is pinned to loopback origins
(`--cors_loopback_only`, any port: port forwarders re-expose the UI on other
local ports; never a routable origin). `--expect-encrypted-jwt` restores
signed mode. The invariants that survive unchanged: loopback-only bind, no
UI path to a wider bind, no wildcard CORS. Additionally, launched without a
workflow argument the chatbot opens a workflow picker (bundled examples + a
directory browser) served by a token-gated control plane whose ONLY write is
`POST /api/select_workflow`; recorded observability data stays read-only over
HTTP. The chat session's `channel_id` is chatbot-managed
(`chatbot-<hex>` per launch) — a single-developer tool exposes no channel
concept in its UI.
*Amendment (owner request, 2026-08-26):* the chatbot channel is now the fixed
string `chatbot`, not `chatbot-<hex>` per launch. A per-launch channel filed
every restart's turns under a new top-level group in the debug rail, splitting
one developer's history by process for a reason invisible in the UI;
conversations already separate sessions. Consequence, by design rather than
accident: since `/initialize` restores a channel's last conversation and any
pending suspended turn, a restart rejoins that history instead of starting
blank — the cross-process resume `[R6]` exists for, now reachable from the
chatbot. "New conversation" remains the way to start clean.
*Amendment (owner request, 2026-08-26):* `run_chatbot` no longer accepts a
workflow path or env-file paths; workflow selection is browser-owned. Missing
env files are installed through token-gated `POST /api/configure_env`, either
by copying browser-selected text into owner-only workflow-local files or by
creating files from the bundled templates. The control plane also admits one
explicit destructive action, `POST /api/clear_conversations`, guarded by an
exact confirmation phrase. It deletes conversations, turns, spans, artifacts,
feedback, and legacy per-channel DB files while preserving training runs,
writer diagnostics, and monotonic conversation counters. The fourth and final
control-plane write is `POST /api/train` (the picker's Train button): it
spawns a DETACHED `fastworkflow train` for a local workflow — same token/host
gates, refused for bundled examples, one run at a time, survives chatbot
exit, stdio to a log beside a pid file under the workflow's state dir. All
other observability routes remain read-only.

**Debug mode** (offline, post-mortem): conversations with nested turns
(including a conversation-less group `[R17]`) → span waterfall (planner /
per-DSPy `fw.llm.call` / tool calls / command executions / ask_user human-wait
distinct) → rendered artifacts → raw record JSON; filters by
status/success/command/context; plus the writer-health banner `[R13]`. LLM
spans expose redacted/capped module input, formatted messages or prompt,
output, provider response, usage, cache status, and provider-native reasoning
when supplied.

**Test mode:** chat pane against the FastAPI server (`/invoke_agent`,
`/invoke_assistant`, `/`-prefix deterministic via `/invoke_assistant`).
**CLI parity `[R24]`:** `/initialize` gains additive
`startup_command`/`startup_action`/`context` request fields so per-session
variation is testable; insights distillation is explicitly out of Studio
scope. A CLI-parity table ships in the Studio docs listing anything still
CLI-only.

**Packaging `[R23]`:** the SPA bundle is built in CI and included in the
wheel via explicit `pyproject.toml` `include` entries, with a wheel-content
assertion test so a missing bundle fails the build; debug mode's HTTP layer
is stdlib-only (works on a base install); test mode requires the `[server]`
extra (guard via `_require_server_extra`-style check with an updated
message).

## 4. Absorbed open work

| Item | How absorbed |
|---|---|
| fix-85g.9 — `GET /turns/{turn_key}` + real 202 | the endpoint consults the in-memory TurnRegistry first, falling back to the DB; the 202 body returns the logical turn_key alongside the execution key, and the execution↔logical mapping is recorded, resolving the key-identity split `[R9]` |
| fix-85g.10 — non-destructive trace replay | spans table is the replay buffer; live queue drain remains for streaming |
| Train metrics unpersisted | `train_runs` written at publication time (tests train into temp copies only — fix-0hb rule) |
| `action.jsonl` last-turn-only | retired at Phase B after the `[R25]` distillation/tooling port |

Not absorbed: fix-85g.11 (backpressure/TTL), fix-85g.13 (distributed store).

## 5. Configuration

| Variable | Default | Meaning |
|---|---|---|
| `FW_OBSERVABILITY` | `1` under fastworkflow entry points; `0` for embedders `[R4]` | master switch for the SQLite sink |
| `FW_OBS_DB_MAX_BYTES` | `1073741824` | soft cap incl. `-wal` `[R12]` |
| `FW_OBS_RETENTION_DAYS` | `30` | prune horizon — spans/artifacts; conversations exempt post-Phase B `[R16]` |
| `FW_OBS_INLINE_ARTIFACT_BYTES` | `262144` | inline vs envelope-in-artifacts-table `[R10]` |
| `FW_OBS_MAX_ATTR_BYTES` | `16384` | per-attribute cap, lossy-and-counted `[R10]` |
| `FW_OBS_QUEUE_MAX` | `10000` | span/artifact queue bound (turn-record queue is separate and small) `[R13]` |
| `FW_OBS_SYNC_WRITE_TIMEOUT_S` | `5` | busy timeout for in-request synchronous store writes (conversation-id minting; Phase-7 ruling C9's fail-fast principle) |
| `FW_OBS_CAPTURE_TRACEBACKS` | `0` | persist traceback artifacts `[R20]` |

## 6. Test strategy

Per final spec §16: no mocks for stores/serialization; scripted-agent doubles
drive real `invoke_command`; store contract suite against real SQLite in
`tmp_path`; fault injection (chmod/full-disk on the writer) asserts turns
complete AND the turn-record queue's bounded-timeout path is exercised
`[R13]`; suspend/resume-across-process tests for `[R2][R6]`; wheel-content
assertion for the SPA `[R23]`; training tests on temp copies only (fix-0hb).

## 7. Phasing

| Phase | Slice | Notes |
|---|---|---|
| 0 | Adversarial design review | **DONE** 2026-08-25, 28/28 resolved |
| 1 | TraceSink + identity plumbing `[R1]` + v1 boundary spans | minor release |
| 2 | SQLite store + two-queue writer + lifecycle/maintenance | same minor; §3.2 invariants are acceptance criteria |
| 3 | Serve from store: `GET /turns` (registry-first `[R9]`) + trace replay | closes fix-85g.9/.10 (owner approval) |
| 4 | Studio debug mode (token auth, CSP, packaging) | minor |
| 5 | Studio test mode (+ additive `/initialize` fields `[R24]`) | minor |
| 6 | Train-metrics persistence | minor |
| 7 | Phase B consolidation | own reviewed slice; gates in §3.3 |

## 8. Resolved review questions (was: open questions)

Q1 multi-writer concurrency → `[R8]` (busy_timeout, BEGIN IMMEDIATE, retries,
local-fs constraint; per-workflow single DB retained). Q2 dual-write
divergence → `[R1][R15]` (single id-minting authority + shared labeling
removes the divergence source). Q3 record_json content → internal
`TurnResult`, post-envelope and post-redaction `[R10][R20]`. Q4 artifact
retention → envelope-in-record + artifacts-table-as-value-holder `[R10]`.
Q5 Studio auth → launch token + scoped GET `[R5]`. Q6 pruning owner →
sink-startup + `studio --prune` `[R12]`.

## 9. Decision Log (review round 1, owner rulings 2026-08-25)

Rows below record the rulings AS MADE on 2026-08-25. Where later owner
amendments supersede details, the inline amendments above govern — notably:
"Studio" shipped as `fastworkflow run_chatbot` (the fastWorkflow Chatbot);
the `--prune`/`--forget-channel` CLI flags became chatbot-UI/store-API
operations (§3.2 amendment); R17/R18/R19/R22/D3 carry inline amendments; the
implemented schema additionally holds `conversation_counters`, `diagnostics`,
and `conversations.updated_at` (Phase-7 rulings C2/R13/C7). `fw.llm.call`
usage/cost capture reads DSPy history: `run_fastapi_mcp --keep_dspy_history`
enables a bounded (20-entry) history for it — off by default; the chatbot's
spawned server always passes it.

| Finding | Ruling |
|---|---|
| R1 | Plumb channel/conversation ids into WEC pre-turn; new DB mints ids from Phase A |
| R2 | Status-transition upsert; write-once for terminal rows only |
| R3 | summary/traces columns on turns + separate mutable feedback table |
| R4 | 0700/0600 always; default ON for fastworkflow entry points; embedders opt-in |
| R5 | Per-launch bearer token + Host/Origin allowlist; JWT-scoped GET /turns |
| R6 | Deterministic root/ask_user span ids + idempotent close upsert |
| R7 | Daemon writer + explicit close() on CLI exit/atexit |
| R8 | timeout=30, BEGIN IMMEDIATE, bounded turn-record retries; local-fs only |
| R9 | Registry-first GET /turns; 202 returns logical key; mapping recorded |
| R10 | Single envelope mechanism inside record_json; FW_OBS_MAX_ATTR_BYTES; lossy-and-counted |
| R11 | PRAGMA user_version + refuse-newer/migrate-forward; spec conversation columns added |
| R12 | auto_vacuum INCREMENTAL at creation; startup prune + studio --prune; count -wal |
| R13 | Two queues; bounded-timeout turn-record put; writer health visible in DB/Studio |
| R14 | Sync-or-acked turn records at Phase B (gate); spans stay best-effort |
| R15 | record_conversation_label sink method called by the labeling hook from Phase A |
| R16 | Retention exempts conversations/turn records post-Phase B (gate) |
| R17 | Synthetic CLI channel + per-//new conversations; WEC self-mints for bare embedders (2026-08-26), leaving a conversation-less group only for failed mints |
| R18 | Token + Host/Origin allowlist as §3.4 invariants |
| R19 | Spawned server pinned 127.0.0.1, CORS pinned, unsigned-JWT refused |
| R20 | Sink-boundary redaction; tracebacks behind FW_OBS_CAPTURE_TRACEBACKS |
| R21 | channel_id on spans/artifacts + forget-channel op with checkpoint+vacuum |
| R22 | Text-as-text, sandboxed iframes, restrictive CSP, parameterized SQL |
| R23 | CI-built bundle + wheel assertion; stdlib debug mode; [server] gates test mode |
| R24 | Additive /initialize startup fields; distillation out of scope; parity table |
| R25 | Distillation ported to ctx.action_log as a Phase B gate; trace_turn.py updated |
| R26 | OTel-aligned claim + documented translation table (derived ids, kind mapping) |
| R27 | Secondary indexes added; command_name/context promoted to span columns |
| R28 | Span emission outside trace-queue guards; sink reached via WEC/config |

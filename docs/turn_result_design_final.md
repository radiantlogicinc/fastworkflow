# fastWorkflow TurnResult — Final Design Specification

**Status: authoritative implementation spec.** This document consolidates
`docs/turn_result_design.md` (original design + Amendments A1–A47),
`docs/turn_result_design_review.md` (48 resolved findings), and
`docs/turn_result_architecture_review.md` (cross-cutting fixes X1–X12, all adopted
2026-06-11). Where this document conflicts with any of those, **this document wins**; they
remain the rationale archive. Traceability tags like `[A7]`/`[X3]` link back to decisions.

---

## 1. Overview

Every user interaction with a workflow is a **logical turn**. A turn is captured in full
(every command execution, every clarification exchange, failures included), returned to the
caller as a `TurnResult`, and persisted as an immutable **turn record** in a unified
conversation store. Records are the system of record for conversations, the source the
agent's cross-turn memory is rebuilt from, and the substrate for review/observability.
Large values (payloads, trajectories) are offloaded to a turn-scoped payload store and
referenced by envelope. One durable record write per turn; the live response is served from
RAM.

---

## 2. Final types

```python
class TurnStatus(str, Enum):                                # [A3]
    COMPLETED = "completed"
    AWAITING_USER = "awaiting_user"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ABANDONED = "abandoned"

class CommandResponse(BaseModel):                           # unchanged value type
    response: str
    success: bool = True
    artifacts: dict[str, Any] = {}
    next_actions: list[Action] = []        # never populated by the agent path [A9]
    recommendations: list[Recommendation] = []

class CommandOutput(BaseModel):                             # one command execution
    command_response: CommandResponse      # collapsed from list at v3.0 [A13/A15]
    command_name: str = ""
    command_parameters: Any = None         # typed model in memory; model_dump() in records [A10.4]
    workflow_name: str = ""
    context: str = ""
    started_at: datetime | None = None     # [A38]
    duration_ms: int | None = None         # [A38]
    nested_turns: list["TurnResult"] = []  # zero producers today; field ships, machinery deferred [A32][X6]

    # properties (never serialized; success already on the wire via command_response) [A42]
    success / command_handled / command_aborted / not_what_i_meant   # unchanged [A3]

class TurnResult(BaseModel):                                # one logical turn
    turn_key: str                          # minted at turn start [A22]
    conversation_id: int                   # [A2]
    ordinal: int                           # stored, authoritative ordering [A24]
    status: TurnStatus                     # [A3]
    failure_reason: str | None = None      # "extraction_error" | "max_iters_exhausted" |
                                           # "agent_error" | "command_error" | "expired" [X10b]
    user_message: str                      # raw original only [A7.3]
    refined_user_message: str | None = None  # agent turns where refinement ran [A36]
    entry_workflow_name: str = ""
    entry_context: str = ""
    answer: CommandResponse                # agent: synthesized; assistant/action: aliases
                                           # command_outputs[-1].command_response [A33-invariant]
    command_outputs: list[CommandOutput] = []  # chronological; includes ask_user entries [A7]
    continuation_of: str | None = None     # field ships at v3.0; writer deferred [A33][X6]
    trajectory_ref: str | None = None      # field ships at v3.0; writer deferred [A37][X6]
    started_at: datetime
    completed_at: datetime | None = None
    suspended_ms: int = 0                  # [A38]
    metadata: dict[str, Any] = {}          # extension point; A10 serializability rules [A25]
                                           # reserved: metadata["tokens"][role] [A38],
                                           # metadata["otel"] = {trace_id, span_id} [A40]

    @computed_field
    def success(self) -> bool: ...         # = answer.success; the ONLY wire predicate [A6][A42]

    # convenience (properties, not serialized) [X10c]
    gallery -> list[CommandOutput]         # payload-bearing outputs, turn order [A9/A20]
```

```python
FW_PAYLOAD_REF_KEY = "__fw_payload_ref__"                   # [A10][A47]
# Envelope (replaces an offloaded value in place):
# {FW_PAYLOAD_REF_KEY: <scoped handle>, "size": int, "content_type": str,
#  "content_encoding": str|None,          # reserved [A27]
#  "error": str|None}                     # set for placeholder envelopes [X3b]
```

**ask_user entries** are ordinary `CommandOutput`s: `command_name="ask_user"`,
`command_parameters` = the agent's question, `command_response.response` = the user's answer
(`""` + `success=False` while unanswered) `[A7]`. The role inversion is documented in the
`CommandOutput` docstring, and helpers `CommandOutput.is_ask_user`, `.question`,
`.user_reply` ship with the types `[X10c]`. The exchange's `duration_ms` is the user's think
time `[A38]`.

**Aliasing rule:** in assistant/action turns `answer` *is* `command_outputs[-1].command_response`
(same object). The serializer never mutates it (selective copy `[A20]`); an A44 test asserts
the alias `[X11]`.

**Exports** from `fastworkflow/__init__` (re-exported from `fastworkflow.turn`): `TurnResult`,
`TurnStatus`, `CommandOutput`, `CommandResponse`, `FW_PAYLOAD_REF_KEY` `[A47]`.

---

## 3. Turn lifecycle

```
                    ┌────────────── resume (answer arrives) ───────────────┐
                    ▼                                                       │
  [turn start] ── RUNNING ──── ask_user ──► AWAITING_USER ─────────────────┘
   atomic [A30/A22/A24]:│                        │
   reset accumulator,   │                        ├─ /cancel_pending, conv switch ──► CANCELLED [A4/A2]
   mint key, assign     │                        ├─ TTL exceeded on next touch ────► ABANDONED [A5.3]
   ordinal, stamp       │                        └─ (pod restart: rehydrate, stay AWAITING)
   started_at           │
                        ├─ finish ──────────────► COMPLETED  (success = answer.success [A6];
                        │                          max_iters → success=False,
                        │                          failure_reason="max_iters_exhausted")
                        └─ fatal error ─────────► FAILED (partial record + re-raise [A5.2])
```

Rules:

- **One logical turn = one key = one record**, across any number of suspensions `[A22]`.
- A new turn begins on `process_turn()` with no awaiting state, or `process_action()`
  (which **requires** no awaiting state — rejected with a clear error during suspension;
  escape hatch `/cancel_pending`) `[A30]`. A message arriving while awaiting is the answer —
  never a reset `[A30.2]`.
- A message arriving **after** `FW_PENDING_TURN_TTL_SECONDS` does **not** execute as a fresh
  turn silently: the stale turn is filed `ABANDONED` and the user receives "your previous
  question expired — please re-ask"; the new message is then processed fresh `[A5.3][X9e]`.
  The pending blob stamps `suspended_at` for this comparison.
- Every terminal transition runs the same finalize-and-clear; terminal records own all
  suspend-offloaded payloads (no orphan class) `[A4/A5/R24]`.
- Nested suspension always **escalates** to the top-level turn; nested `TurnResult`s attach
  only when complete — partial-inside-partial pending blobs never exist `[A32]`.
- Conversation switches (`/new_conversation`, `/activate_conversation`) auto-cancel a
  suspended turn (recorded under its original conversation) and run under the per-session
  lock `[A2]`.

---

## 4. Capture

- **Choke point:** every `CommandExecutor.invoke_command` result is appended to the
  accumulator — app commands, CME/clarification executions, everything `[§3.1][A33-invariant]`.
- **Failed tool calls** append `CommandOutput(success=False)` with error type, message,
  truncated traceback in artifacts (developer projection only); the wrapper is
  exception-safe (`repr` fallbacks) and passes suspension signals (`AskUserSuspend`,
  `CommandCancelledError`) through untouched `[A5.1][X9f]`.
- **Eager artifact validation** `[X3a]`: at the `invoke_command` return boundary, a cheap
  type-walk validates artifact serializability and **raises at the author's stack frame**
  (on by default; disable with `FW_EAGER_ARTIFACT_VALIDATION=0` for production hot paths).
  Turn-filing rejection remains the backstop.
- ask_user exchanges append at ask/suspend time (unanswered convention) and are completed in
  `_post_ask_user_response` `[A7]`.
- `action_log` is retired; the summary text view (and the dspy.History projection fields) are
  **derived at read/finalize time from `command_outputs`** — never stored redundantly in the
  record `[A7.4][X12]`.

---

## 5. Keyspace

All keys for one conversation embed the identical **hash-tag** segment so Redis Cluster
colocates them `[X1]`; `UNLINK` is used for deletes.

```
fw:conv:{ch/cv}                  conversation metadata record
fw:turn:{ch/cv}:<turn_key>       turn record
fw:payload:{ch/cv}:<turn_key>:<sha256-hex32>   payload blob          [A8][A27]
fw:feedback:{ch/cv}:<turn_key>   feedback card                        [A18]
fw:turnidx:{ch/cv}               ZSET ordinal → turn_key  (write-time index) [X1]
fw:convidx:{<channel>}           ZSET created_ts → conv_id            [X1]
fw:lease:{<channel>}             writer lease (detection)             [X9d]
```

- `{ch/cv}` is the literal hash-tag form of `<channel_id>/<conv_id>`.
- **Turn key grammar:** `YYYYMMDDTHHMMSS.ffffffZ-<uuid-hex-12>` — colon-free, sortable;
  timestamp = logical-turn start `[A22][A24][A26]`.
- Disk layout mirrors the keyspace as directories; **every path component passes the shared
  allowlist sanitizer** (`[A-Za-z0-9._-]`, reject empty/`.`/`..`); the pending store is
  retrofitted onto the same sanitizer `[A26]`.
- **Conversation metadata record** holds: topic, summary, status, timestamps, schema version,
  `next_ordinal` (updated with each turn write — restore never scans `[X2]`),
  `total_turns`, `approx_bytes` (size accounting `[X5]`).
- All reads go through the index (`ZRANGE` + pipelined `MGET`); `SCAN` survives only inside
  retention deletes `[X1]`.

---

## 6. Stores and serializer

**`ConversationTurnStore`** (disk/redis; factory mirrors `get_session_state_store()`) owns
conversation metadata, turn records, feedback cards, and the two indexes. Its docstring
names all record types `[X11]`. Listing returns `(turn_key, summary-card)` pages
(ordinal, timestamps, status, success, truncated user_message, summary, command count) with
`limit`/`before`/`after`, newest-first `[A23]`.

**`PayloadStore`** (disk/redis): `put(channel, conv, turn_key, data) -> handle`,
`get(handle) -> bytes|None`. UTF-8 encode strings; SHA-256 hex-32 algorithm-prefixed leaf;
raw bytes only (metadata lives in the envelope); atomic disk writes (temp +
`os.replace`) `[A8][A27]`. **Atomic temp+replace applies to ALL disk writes in every
store** `[X9a]`.

**`TurnSerializer`** — single owner of serialization and reading at both boundaries
(suspend partial and every terminal write); transport-free, lives beside the stores
`[A21]`:

- Selective copy — never mutates live objects, never blind-deepcopies `[A20]`.
- Offload rule: any `str`/`bytes` artifact value > `FW_PAYLOAD_OFFLOAD_THRESHOLD_BYTES`
  → `PayloadStore` + envelope `[A10]`.
- **Error classification `[X3]`:** non-serializable values found at a persistence boundary
  do **not** abort the write — the offending value is replaced by a **placeholder envelope**
  (`error` field set, key+command named) and the record is always written; the event
  increments `fw_serialization_rejections_total`. Hard-raising is the *eager dev-time*
  validator's job (§4). Suspend-boundary serialization failure follows the same rule: write
  what can be written, alarm loudly; the turn is never failed by filing.
- Stamps the record format version; strict reader dispatch (unknown/missing version =
  explicit error) — but the **memory-rebuild reader skips-and-counts** corrupt records
  (alarm metric) instead of failing the session `[A25][X9/Reliability-3]`.
- Per-turn payload budget: `FW_MAX_TURN_PAYLOADS` / `FW_MAX_TURN_PAYLOAD_BYTES`; beyond it,
  payloads become not-retained placeholder envelopes (warning, never turn failure) `[X12]`.
- Owns the reader: envelope detection, lazy resolution, tolerate-missing-payload.

**Write-once:** records and index entries are written once per key (`SET NX`/`O_EXCL`).
A collision during retry **verifies content** (parseable, version present) before claiming
idempotent success; unparseable existing content is overwritten via atomic replace and
alarmed `[A22][X9a]`.

---

## 7. Memory, conversations, restore

- The agent's `dspy.History` is a per-session cache **projected from records**:
  `status=COMPLETED` records only `[A4]`, **newest-first, capped at
  `FW_MEMORY_PROJECTION_TURNS` (default 10)** `[X2]`, joined with feedback cards in the same
  indexed read. Projection fields: summary, derived traces view, feedback.
- Restore reads the conversation metadata record for `next_ordinal` — never a full scan
  `[X2]`. Ordinal **holes are legal** (a best-effort-lost record leaves a gap; the stored
  ordinal remains authoritative; the original §7.7/§8.1 ordinal-by-sort language is void)
  `[X9b]`.
- **Auto-rotation:** a conversation rotates (finalize topic/summary, reserve next id) when it
  exceeds `FW_CONVERSATION_MAX_TURNS` (default 200; 0 disables), bounding restore cost,
  History size, and GC granularity for implicit-conversation deployments `[X2]`.
- Restore failure policy: store unreachable → session starts with **empty memory + alarm**
  (`fw_memory_rebuild_failures_total`), never a failed session `[X9]`.
- **Writer lease (detection only):** at session creation the pod takes
  `fw:lease:{channel}` (`SET NX EX`, TTL-refreshed); a conflicting holder increments
  `fw_writer_conflicts_total` and logs loudly. Single-writer-per-channel (sticky routing)
  remains the deployment requirement `[A31][X9d]`.

---

## 8. Live response path

- Served **from RAM**; offload happens on the serialized copy only; zero store reads —
  except envelope entries in a rehydrated (post-restart) partial, which fetch by handle
  server-side `[A16][A17]`.
- The **headline never carries payloads**; the gallery is all payload-bearing outputs in
  turn order; the UI picks the featured payload `[A20]`. Gallery provenance: full in
  `TurnResult`; xray embeds provenance in response text initially (ResponseTuple change is
  xray-repo scope) `[A9]`.
- Bundled-server bodies inline payloads up to `FW_MAX_INLINE_PAYLOAD_BYTES` (default 10 MB);
  beyond it, the envelope rides instead `[A16]`.
- **Queue contract (v3.0):** `command_output_queue` carries status-stamped `TurnResult`s
  only; every mid-turn awaiting enqueue is paired with a trace sentinel (fixes `fix-5fv`);
  the trace queue and dim CLI ticker are unchanged `[A19][A34]`.

---

## 9. Failure semantics

| Event | Behavior |
|---|---|
| Record write fails (I/O) | 1 retry / 250 ms cap, then best-effort: turn succeeds for the user (side effects already committed), `fw_record_write_failures_total` increments, prominent log `[A28][X12-retry-budget]` |
| Serialization rejection at filing | Placeholder envelope, record still written, `fw_serialization_rejections_total` `[X3]` |
| Author bug in dev/test | Eager validator raises at the offending frame `[X3a]` |
| Crash windows | Bounded leaks: retry adopts (stable key + hashed leaves) or retention removes `[A28]` |
| Stale pending blob | Record write precedes pending clear; at restore, a blob whose turn key already has a record is discarded `[A28.3]` |
| Pending-store save/clear raises | Same best-effort treatment as record writes — never fails a committed turn `[X9f]` |
| Transport retry (client) | Optional `Idempotency-Key` header maps to the turn key; replay returns the existing record `[X9c]` |
| Old-schema pending blob (upgrade) | Graceful expiry + "please re-ask"; **2.21.x expires *future* schemas symmetrically** for rollback `[A14][X4]` |

---

## 10. Projections and read side

- **One record, two read-time views** via a projection function beside the reader: *user*
  (non-internal executions + exchanges + answer + gallery; error messages, no tracebacks) and
  *developer* (everything: internal CME executions, parameters, tracebacks, trajectory,
  refined message). Internal detection derives from `workflow_name`, predicate centralized
  in exactly one place `[A39]`.
- **Record-first authorization:** every read path authorizes the turn record, then resolves
  its references; no bare-handle endpoints, ever. End users: JWT channel-bound, user
  projection. Admin/developer: the `/dump_conversations`-style admin path `[A41]`.
- The projection function and envelope reader are **public API at v3.0** (not "later") —
  clients get a shipped way to read records `[X10d]`.

---

## 11. Observability

- **`MetricsSink` protocol** (counter/histogram; no-op default, log-emitting fallback)
  `[X7]`. Core metrics: `fw_turn_duration_seconds{status}`, `fw_turns_total{status}`,
  `fw_record_write_failures_total`, `fw_serialization_rejections_total`,
  `fw_execution_failures_total`, `fw_payload_fetch_misses_total`,
  `fw_memory_rebuild_failures_total`, `fw_writer_conflicts_total`,
  `fw_tokens_total{role,kind}` (best-effort).
- **Log correlation:** `channel_id`/`conv_id`/`turn_key` bound via logging contextvar;
  every framework log line during a turn carries them; record *contents* never logged
  `[A12][X7]`. `CommandTraceEvent` gains a `turn_key` field (additive) `[X7]`.
- **In-flight registry:** per-pod gauge of running + suspended turns, exposed alongside the
  existing probes `[X7]`.
- **OTel:** documented mapping (turn=root span, execution=child span, ask_user=human-wait
  span, nested=sub-spans, tokens=`gen_ai.usage.*`); `metadata["otel"]` join keys best-effort;
  exporter out of scope `[A40]`.
- Timing: per-execution `started_at`/`duration_ms`; per-turn `started_at`/`completed_at`/
  `suspended_ms` `[A38]`.

---

## 12. Configuration inventory `[X5a]`

Store-selection vars mirror the existing unprefixed convention; new feature knobs are
`FW_`-namespaced. All defaults chosen so **a 2.20→2.21 upgrade needs zero config changes**.

| Variable | Default | Meaning |
|---|---|---|
| `SESSION_STATE_STORE` | `disk` | pending store backend (existing) |
| `CONVERSATION_TURN_STORE` | `disk` | unified store backend (v3.0) |
| `PAYLOAD_STORE` | = `CONVERSATION_TURN_STORE` | payload backend |
| `FW_ALLOW_DISK_STORES` | unset | bundled server refuses `disk` stores in prod unless `1` `[X12]` |
| `FW_PAYLOAD_OFFLOAD_THRESHOLD_BYTES` | `4096` | inline vs offload `[A10]` |
| `FW_MAX_INLINE_PAYLOAD_BYTES` | `10485760` | response inline cap `[A16]` |
| `FW_MAX_TURN_PAYLOADS` / `FW_MAX_TURN_PAYLOAD_BYTES` | `64` / `52428800` | per-turn budget `[X12]` |
| `FW_MAX_TRAJECTORY_BYTES` | `262144` | trajectory truncation (when writer ships) `[X12]` |
| `FW_PENDING_TURN_TTL_SECONDS` | `604800` (7 d) | abandonment TTL `[A5.3]` |
| `FW_MEMORY_PROJECTION_TURNS` | `10` | restore projection cap `[X2]` |
| `FW_CONVERSATION_MAX_TURNS` | `200` (`0` = off) | auto-rotation `[X2]` |
| `FW_EAGER_ARTIFACT_VALIDATION` | `1` | dev-time validator `[X3a]` |

`fastworkflow/examples/fastworkflow.env` gains a commented persistence section listing all
of the above (including the previously undocumented `SESSION_STATE_STORE`).

---

## 13. Operations `[X5]`

- **`fastworkflow admin` CLI:** `pending list|show|cancel [--all-channels]` (makes the
  upgrade drain real and verifiable), `store stats [--channel]`, `turn show <key>
  [--developer]`, `verify` (dangling handles, recordless prefixes, stale blobs), and
  **`retention apply --older-than Nd [--dry-run]`** — a cron-invocable reference
  implementation; the infra contract reduces to "schedule this."
- **Startup preflight:** verify Redis `maxmemory-policy=noeviction` on the store DB; warn on
  disk backends with multi-pod signals; log the resolved config inventory.
- **Readiness** pings the unified store (it is load-bearing post-A1).
- **Security contract** `[A12]`: stores hold PII/entitlement-grade data; deployments owe
  encryption at rest, TLS, least privilege; framework owes record-mediated access and no
  record contents in logs. Retention = age-based conversation-prefix deletes; quota named as
  an infra responsibility. Runbook section in the migration guide: Redis-full behavior,
  wedged-blob repair, backup/restore (Redis RDB/AOF; disk tree), broken-stickiness symptoms,
  upgrade-day drain.

---

## 14. Release train `[A14][X4][X6]`

**v2.21 (minor, non-breaking — the xray payload fix; ~6 files):**
`fastworkflow/turn.py` types + exports; WEC accumulator with the A30 lifecycle;
`process_turn()` returning `TurnResult` (built from the *existing* `command_responses[0]` —
no model collapse); A5.1 failure capture; A7 ask_user entries; eager artifact validation;
generators emit new-style `command_response=`; `process_message` gains a
`DeprecationWarning` pointing at `process_turn` `[X10a]`. Standalone bugfix PRs off the
train: conversation-switch lock acquisition `[A2.3]` and the ask_user trace-sentinel pairing
(`fix-5fv`).

**v2.21.x (patch, rollback enabler):** pending blobs with *future* schema versions expire
gracefully, symmetric to A14 `[X4]`.

**v3.0 (major — big-bang cutover, mixed fleets forbidden):** `command_responses` collapse
with the constructor shim introduced here (removed v4.0); wire hard-break (endpoints +
SSE return `TurnResult.model_dump()`; MCP `isError = not success`); `ConversationTurnStore`
consolidation (lift `generate_topic_and_summary`, `_ensure_unique_topic`,
`restore_history_from_turns` verbatim; Rdict data loss announced via a one-time synthesized
per-channel notice on first touch `[X12]`; Rdict files left in place with a tombstone marker
for rollback detection `[X4]`); `action_log` retired; queue contract switch `[A19]`;
`process_message` **removed** (loud `AttributeError`, migration guide points to
`process_turn`) `[X10a]`; projections + envelope reader public; metrics/logging/lease/
preflight/admin CLI; in-repo command files migrate here `[X6]`.

**Fast-follows (post-3.0, fields already in schema):** trajectory writer `[A37]`, token
capture `[A38]`, `continuation_of` writer `[A33]`, nesting machinery + synthetic producer
`[A32]`, the A41 read API, OTel exporter `[A40]`.

**Cutover runbook (ordered):** announce → provision Redis (dedicated DB, noeviction, TLS) →
set env vars → drain (`admin pending list/cancel`) → stop **all** old pods → start new →
cut clients → enable retention cron + alarms. **Rollback:** roll back to ≥2.21.x only;
3.0-era conversations are unrecoverable on rollback (documented, bidirectional loss);
clients roll back in lockstep `[X4]`.

---

## 15. Module layout `[X11]`

```
fastworkflow/
  turn.py                # TurnResult, TurnStatus, CommandOutput, CommandResponse,
                         # FW_PAYLOAD_REF_KEY (one module: solves the circular ref)
  turn_accumulator.py    # A30 lifecycle: start/append/suspend/resume/finalize;
                         # key mint, ordinal, timing. WEC delegates to it.
  turn_serializer.py     # serializer + envelope + reader + A39 projections
  stores/
    base.py              # shared keyed-blob base extracted from session_state_store.py
    sanitize.py          # A26 shared sanitizer
    pending.py           # SessionStateStore (retrofitted onto base + sanitizer)
    conversation_turn.py # ConversationTurnStore + indexes
    payload.py           # PayloadStore
  metrics.py             # MetricsSink protocol + default sinks
```

`run_fastapi_mcp/conversation_store.py` is deleted at v3.0; lifted logic moves into core.

---

## 16. Test strategy `[A44][X8]`

- **LLM-boundary rule:** "no mocks" applies to stores/serializer/workflows; agent loops are
  driven by a first-class **`ScriptedToolAgent`** test double that calls real
  `invoke_command` per a fixed script (codifying the existing test pattern).
- **Determinism seams (specified now):** injectable `Clock` (WEC, serializer, stores;
  default real time) and injectable `mint_turn_key()`.
- **Store contracts:** one parametrized suite run against `disk`, `fakeredis` (every PR), and
  real Redis (CI service-container job) — SCAN/NX/prefix/UNLINK parity is structural.
- **Fault injection:** the A28 counter is an inspectable `MetricsSink` attribute; disk
  failures via chmod/`O_EXCL`; Redis failures via a permitted raising-store subclass.
- **Matrix:** the A44 22 items **plus**: A6 max-iters stamping, A11 action records, A16
  inline cap, A17.3 rehydration fetch, A25 strict version dispatch, A42 wire shape, A3
  awaiting-artifact-removal regression, the answer-alias assertion, concurrent same-channel
  accumulator integrity, and the A31 last-writer-wins demonstration (~32 items). A32
  escalation tests are explicitly deferred until a nested runtime exists (tracked).

---

## 17. Documentation deliverables

Migration guide (schema mapping old→new for HTTP/SSE/MCP + constructor migration `[A13]`;
upgrade runbook + rollback `[X4]`; ops runbook `[X5]`; zero-config 2.21 guarantee);
status×success matrix; "reading a turn record" cookbook (gallery iteration, ask_user
inversion, envelopes, projections) `[X10e]`; updated example workflows at v2.21.

---

## 18. Traceability

| Source | Where it landed |
|---|---|
| A1–A47 | §§2–14 (all decisions carried; A14/A13 train re-cut by X6 with user approval 2026-06-11) |
| X1 index | §5, §6, §7 |
| X2 caps/rotation | §7, §12 |
| X3 error classification | §4, §6, §9 |
| X4 rollback | §9, §14 |
| X5 ops/config | §12, §13 |
| X6 minimal 2.21 + defer list | §14 (user-approved, supersedes parts of A14/A37/A38/A33/A32 timing) |
| X7 metrics/logging | §11 |
| X8 test seams | §16 |
| X9 reliability hardening | §3, §6, §7, §9 |
| X10 ergonomics | §2, §10, §14, §17 |
| X11 spec/layout | this document, §15 |
| X12 sundry | §6, §8, §12, §14 |

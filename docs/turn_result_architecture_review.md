# Architecture review: TurnResult design (sections 0–13 + Amendments A1–A47)

Ten-concern architectural review conducted 2026-06-11 by parallel review agents, each grounded
in `docs/turn_result_design.md` (amendments authoritative), `docs/turn_result_design_review.md`,
and the codebase. This review evaluates the *finalized* design — after all 48 review findings
were resolved — and asks what the amendments themselves missed or under-priced.

## Executive summary

The design's **write side is sound and cheap**: per-turn hot-path additions are negligible
against multi-second LLM turns and are a net *win* over today's whole-conversation Rdict
rewrite. The type algebra, lifecycle model, co-GC keyspace, and failure-capture semantics
survived scrutiny. The exposure concentrates in four places the amendments deferred rather
than priced:

1. **The read side is unindexed and unbounded** (SCAN-based listing, full-conversation restore
   projection) — the single most-flagged issue, independently raised by the performance,
   scalability, and observability reviews.
2. **A new contradiction surfaced: A10 (strict rejection) × A28 (best-effort writes)** — an
   author's bad artifact deterministically fails every record-write retry, silently amputating
   conversation history and agent memory. The two amendments never met each other.
3. **Operational delegation without artifacts** — retention, quota, alarms, and drain are
   contracts in prose with no tooling, no config inventory, no named metrics, and no preflight;
   the default deployment outcome is that none of them happen.
4. **No rollback story** — v3.0 is an undocumented big-bang cutover; 3.0→2.21 rollback
   produces split-brain history and unreadable pending blobs; canary/blue-green is structurally
   impossible under the current plan.

None of these invalidate the architecture. All are addressable with a final amendment round
(A48+) before implementation. The code-change review additionally proposes a **much smaller
v2.21** than A14 currently schedules.

## Top cross-cutting findings (ranked)

### X1 — Build the per-conversation index now (A23.1 reversal) — HIGH
Redis `SCAN MATCH` iterates the *entire keyspace*, not the matching prefix. Every listing,
session restore, feedback join, retention sweep, and co-GC delete pays O(total keys) — at
months of multi-tenant scale, every `/initialize` walks millions of keys. Three reviews
independently called A23's "no write-time index, reserved as future optimization" the
worst under-priced decision. **Fix (~20 lines):** one `ZADD fw:turnidx:{channel}:{conv}
ordinal→turn_key` pipelined with each record write (plus a per-channel conversation ZSET);
reads become `ZRANGE`+`MGET`; SCAN survives only inside retention deletes. Also hash-tag keys
(`{channel/conv}`) for future Redis Cluster compatibility and use `UNLINK` for co-GC.

### X2 — Cap the restore-time memory projection (A1.2 amendment) — HIGH
The A1 rebuild reads **all** of a conversation's records (plus feedback) at session restore,
while `_refine_user_query` consumes only the last ~5 turns. Deployments that never rotate
conversations (xray today) degrade monotonically forever. **Fix:** newest-first, last-K
(≈5–10) completed records; persist the ordinal counter in the conversation metadata record
(updated on each turn write) so restore never needs a full read; auto-rotate conversations at
a turn-count/age threshold for implicit-conversation deployments.

### X3 — Reconcile A10 × A28: classify errors at persistence boundaries — HIGH
Strict rejection (author bug) and store I/O failure (infra) are different animals getting the
same best-effort treatment. A bad artifact fails identically on every retry → permanent,
near-silent record loss; and the error surfaces at *turn filing*, an entire agent turn after
the offending `return`. **Fix:** (a) eager dev-time artifact validation at the
`invoke_command` return boundary (cheap type-walk; moves the error to the author's stack
frame); (b) at persistence boundaries, degrade — replace the offending value with an error
placeholder envelope and *always* write the record; (c) define suspend-boundary
(`serialize_state`) failure semantics, currently unspecified anywhere.

### X4 — Design rollback; document v3.0 as a big-bang cutover — HIGH
Nothing addresses 3.0→2.21: newer-schema pending blobs (A14 expiry is forward-only), 3.0
conversations invisible to 2.21 (while pre-3.0 Rdict history *reappears* — split-brain), wire
rollback requiring synchronized client rollback. Mixed-version fleets are impossible
(divergent stores + wire shapes; sticky sessions break on failover). **Fix:** a rollback
amendment — 2.21.x patch release that expires future-schema blobs symmetrically; Rdict
tombstone marker at 3.0; documented bidirectional-loss contract; an ordered cutover runbook
that explicitly forbids mixed fleets.

### X5 — Ship operational artifacts, not contracts (A12/A28/A29 hardening) — HIGH
The "infra will do it" stack (retention, quota, alarm, drain) is unenforced; the failure
chain is concrete: no retention → `noeviction` Redis fills → all writes fail → A28 silently
drops every turn record. **Fix:** (a) a **Configuration Inventory** amendment naming every env
var with defaults (`CONVERSATION_TURN_STORE`, `PAYLOAD_STORE`,
`FW_PAYLOAD_OFFLOAD_THRESHOLD_BYTES=4096`, `FW_MAX_INLINE_PAYLOAD_BYTES=10485760`,
`FW_PENDING_TURN_TTL_SECONDS`, …) + an updated `fastworkflow.env` template; (b) a
`fastworkflow admin` CLI: `pending list|cancel` (makes the A14 drain real and verifiable),
`store stats`, `turn show`, `verify`, and **`retention apply --older-than Nd --dry-run`** as
a cron-invocable reference implementation; (c) startup preflight (verify
`maxmemory-policy=noeviction`, warn on disk backends in multi-pod signals, log resolved
config); (d) readiness probe pings the now-load-bearing store.

### X6 — Shrink v2.21 to the bug fix; move the collapse to v3.0 — HIGH (process)
A14 currently schedules ~96 in-repo command-file migrations plus the constructor shim into
the "quick-fix minor," ballooning it to ~2.5k lines/120 files — and the shim covers
constructors only, while the collapse breaks ~20 read sites scheduled for 3.0 anyway.
**Minimal v2.21 (~500 lines / ~6 files):** `TurnResult`/`TurnStatus` + exports, the WEC
accumulator (A30 reset), `process_turn()` building `answer` from the existing
`command_responses[0]`, A5.1 failure capture, A7 ask_user entries, generators emitting new
style. **Defer to post-3.0** (explicit costs accepted): A37 trajectory persistence, A38 token
slots (keep timestamps), A33 `continuation_of`, A32 nesting machinery (keep the empty field).
Ship A2.3 (lock acquisition in switch endpoints) and A19.3/`fix-5fv` (sentinel pairing) as
standalone bugfix PRs off the train.

### X7 — Name the metrics; minimal pipeline — MED
A28's "monitoring counter" has no concrete form (the codebase has a stdlib logger only).
**Fix:** a minimal `MetricsSink` protocol (counter/histogram; no-op default, log fallback);
name the metrics (`fw_record_write_failures_total`, turn-duration histogram, turns-by-status,
payload-fetch misses, token usage); stamp `turn_key` on `CommandTraceEvent` and into a logging
contextvar so every framework log line carries `channel_id`/`turn_key`; an in-flight
gauge/registry of running+suspended turns per pod.

### X8 — Testability seams before implementation — MED
A44 is not executable as written: "no mocks" is already violated by existing agent tests; zero
Redis test infrastructure exists; time/uuid are called inline. **Fix:** amend A44 with an
LLM-boundary rule (codify a `ScriptedToolAgent` double that drives real `invoke_command`);
add injectable `Clock` and `mint_turn_key()` seams to A22/A38 *now*; add `fakeredis` +
parametrized store-contract suite (disk/fakeredis/real-redis CI job); expand the matrix
~22→~30 items (A6 max-iters, A11 action records, A16 cap, A17.3 rehydration fetch, A25
version dispatch, A42 wire shape, A3 regression).

### X9 — Reliability hardening details — MED
(a) Extend A27.3 atomic-write (temp+`os.replace`) to **all** disk writes, and make A22's
write-once collision verify content before claiming idempotent success (truncated-JSON
records are otherwise unrepairable by design). (b) Declare ordinal holes legal (a best-effort
loss leaves N-1, N+1) and strike the stale 7.7/8.1 ordinal-by-sort text. (c) Optional client
idempotency key mapping to the turn key (A22 mints early) to close the transport-retry
double-execution edge. (d) Per-channel lease fencing (`SET NX EX`) at restore so A31
violations are *detected*. (e) On lazy abandonment, return "expired — please re-ask" instead
of executing the stale answer as a fresh turn; stamp `suspended_at` in the blob. (f) Make the
A5.1 capture wrapper exception-safe; give pending-store save/clear the A28 treatment.

### X10 — Author/embedder ergonomics — MED
(a) Deprecate `process_message` at 2.21 and make 3.0 an alias-or-removal — never a silent
return-type swap (the one "fails subtly" path A45 missed). (b) Add
`TurnResult.failure_reason` (`extraction_error | max_iters | …`) so `completed+success=False`
is diagnosable on the wire. (c) Ship three convenience properties with 2.21 —
`TurnResult.gallery`, `CommandOutput.is_ask_user`, `question`/`user_reply` aliases — killing
the gallery-iteration and role-inversion footguns. (d) Promote the A39 projection + envelope
reader to the public surface at 3.0 (clients otherwise have ~10 concepts and no shipped
reader). (e) Docs: status×success matrix; "reading a turn record" cookbook.

### X11 — Consolidated spec + module layout before coding — MED
47 amendments overriding scattered originals (and each other) is a rationale archive, not an
implementation spec — drift is guaranteed. **Fix:** a ~300-line consolidated spec (final
types, key grammar, store ABCs, lifecycle state machine, conventions table). Proposed layout:
`fastworkflow/turn.py` (types; solves the TurnResult↔CommandOutput circular ref),
`turn_accumulator.py` (A30 lifecycle, key mint, ordinal, timing — WEC delegates),
`turn_serializer.py` (serializer + envelope + projections), `stores/` package (`pending.py`,
`conversation_turn.py`, `payload.py`, `sanitize.py`) — built on a shared keyed-blob base
extracted from `session_state_store.py`; lift `generate_topic_and_summary`,
`_ensure_unique_topic`, and `restore_history_from_turns` verbatim rather than rewriting.
Document the A7 role inversion on the `CommandOutput` class itself.

### X12 — Sundry (LOW)
Per-turn payload budget (count + total bytes) to bound RAM/OOM; size-cap the trajectory blob
(developer-only); specify that the traces text view is *derived at read time*, never stored
(avoids double-carrying text in records); enable A27's compression slot for CSV before
launch; design the A1.5 loss announcement (one-time synthesized notice per channel +
`/list_conversations` epoch marker); assert the 2.21 zero-config guarantee in the migration
guide; at 3.0 refuse `disk` stores in the bundled server unless `FW_ALLOW_DISK_STORES=1`.

---

## Per-concern findings

### 1. Performance
**Strengths:** net win vs today's O(n) per-turn Rdict rewrite; live path zero store reads
(A16/A20); O(1) accumulation; zero new LLM calls (`_refine_user_query` is string concat;
summary extraction already runs today); honest memory accounting.
**Risks:** (H) unindexed SCAN everywhere → X1; (M) full-history restore → X2; (M) in-lock
write sequence unquantified retry budget — during a Redis stall, retries × (K payloads +
trajectory + record) serialize behind the lock; (M) RAM = payloads × concurrent turns × 2–3
transient copies (~1–1.5 GB spikes at 50×10 MB); (L) records may double-carry traces text
(A1.2 vs A7.4 ambiguity); (L) switch endpoints gain lock + cancel writes (rare, correct).
**Recommendations:** X1, X2; quantify A28's retry budget (e.g. 2 retries / 250 ms cap);
trajectory size cap; per-turn payload budget; resolve the traces-text ambiguity as
derived-at-read.

### 2. Reliability
**Strengths:** crash-ordering mostly right (mint-at-start, record-then-clear, stale-discard,
ownership transfer, co-GC); every terminal path records; single serializer prevents format
drift.
**Risks:** (H) A10×A28 deterministic permanent record loss → X3; (H) disk record corruption
unrepairable (atomicity specified only for payloads; collision check doesn't verify content);
(H) memory-rebuild failure/latency policy undefined (store down at `/initialize` — fail or
empty memory?); (M) ordinal holes after best-effort loss + stale 7.7/8.1 text; (M) no
client-retry idempotency; (M) A31 violations undetected; (M) disk failover loses turns with
no terminal record; (M) TTL-boundary answer executed as fresh turn; (L) capture wrapper can
itself raise; (L) pending-store I/O failure mid-request unspecified.
**Recommendations:** X3, X9; rebuild policy: skip-and-count corrupt records, degrade to empty
History with alarm, cap the scan.

### 3. Scalability
**Strengths:** one write per turn, light records, structural ownership, sound single-pod
semantics.
**Risks:** (H) SCAN O(keyspace) → X1; (H) unbounded implicit conversation → X2 + rotation;
(H) retention delegated but un-indexed; failure mode = silent data loss via `noeviction` →
X5; (H) single-writer is both throughput ceiling and correctness cliff (no fencing) → X9(d);
(M) aggregate RAM uncapped → X12; (M) no Redis Cluster story (no hash tags; MGET/prefix ops
break) → X1; (M) multi-MB single values (compression deferred); (L) Redis becomes shared SPOF
vs local Rdict.
**Recommendations:** X1 (incl. hash tags, UNLINK), X2, X5; Redis `INCR` ordinal + lease
epoch; per-turn payload budget; enable compression for CSV.

### 4. Minimizing code changes
**Strengths:** evidence-driven deletions; single choke point; shim+train policy right; A7
avoided a type explosion.
**Risks:** (H) v2.21 balloons to ~120 files via in-repo migration that the shim exists to
defer; (H) constructor-only shim vs read-site breaks; (H) A1 absorption ≈40% of the 3.0 diff
with planned rewrites of liftable code; (M) A37/A32/A44 disproportionate cost; (M) A22/A2
conversation-id awareness leaks into the transport-free core; (L) A38 tokens, A33.
**Recommendations:** X6 (minimal 2.21 + defer list); reuse mandates (shared keyed-blob base
from `session_state_store.py`; lift conversation_store logic verbatim); fallback if 3.0 must
shrink: original decision 11 (separate store, temporary dual-write) is the lower-blast-radius
option A1 rejected.

### 5. Readability / modularity
**Strengths:** clean type algebra; TurnSerializer as single owner; structural co-GC; unusual
evidence-confidence for deletions.
**Risks:** (H) amendments-over-original doc will cause implementation drift → X11; (H) WEC
grows past ~1,200 mixed-concern lines; (M) answer↔gallery aliasing unenforced in memory; (M)
A7 role inversion + success-overload are two field-meaning lies in one convention; (M)
`success != all-succeeded` naming trap; (M) `ConversationTurnStore` undersells three record
types; `process_turn` fate unpinned; (M) circular type ref needs one-module placement; (L)
kept magic strings; (L) workflow_name-derived internal detection; (L) envelope duck-typing
must not leak outside the serializer (+ A17 fetch path).
**Recommendations:** X11 (spec + layout); alias safety (computed property or deep-copy at
finalize + test); `is_ask_user` helper + docstring at the definition site; rename or
document the store; pin `process_turn` deprecation.

### 6. Observability
**Verdict: records-complete, pipeline-incomplete.**
**Strengths:** single choke point; full timing incl. `suspended_ms`; honest status/success;
failures recorded not shredded; trajectory; projections; OTel mapping; early-minted key;
immutable records + feedback.
**Gaps:** (H) no read-side API — the bottleneck gating everything; (H) A28 counter has no
concrete form (stdlib logger only — the contract is unfalsifiable); (H) no aggregation story
(p95 latency = scan everything); (M) no log-correlation convention; (M) no live in-flight
visibility (suspended/running turns fleet-wide); (M) one alarm, no taxonomy
(failed-turn rate, abandon rate, payload-miss rate, expiry, strict-rejection); (L)
trace-event/record duplication without a shared key; (L) "best-effort" tokens undefined.
**Recommendations:** X7 (turn_key on trace events + contextvar; MetricsSink; 4–6 core
metrics; in-flight gauge); then the A41 read API as the prerequisite for the observability
tool; defer exporter/search/warehouse.

### 7. Manageability
**Strengths:** one-command retention shape; thought-through failure semantics; security
contract; A44 coverage of state transitions.
**Gaps:** (H) unenforced infra stack with silent-failure chain → X5; (H) no config inventory
(no variable even named; `fastworkflow.env` has no persistence section) → X5(a); (H) zero
day-2 tooling (backup/restore, wedged-blob repair, size accounting, admin visibility) →
X5(b); (M) drain unverifiable; (M) counter unnamed/in-process; (M) sticky violations silent;
(M) reaper deferred + retention delegated = pending blobs live forever by default; (L)
`noeviction` never verified.
**Recommendations:** X5 in full (config inventory amendment, admin CLI, reference retention
command, named metrics, preflight, ops runbook in the migration guide).

### 8. Testability
**Strengths:** traceable matrix; clock-free-by-design expiry test; isolatable serializer;
shipped synthetic producer; crisp invariants.
**Gaps:** (H) "no mocks" already false for agent paths — matrix not executable as stated;
(H) zero Redis test infrastructure while Redis-only semantics are where the bugs live; (H) no
time/uuid seams; (M) A28 failure-path testing needs an inspectable counter + permitted
fault-injection fake; (M) A32 escalation untestable until a nested runtime exists (flag it);
(M) A31 deterministic half untested and unwaived; (M) no concurrent same-channel accumulator
test; (M) ~8 matrix omissions; (L) FastAPI tests skip silently without env.
**Recommendations:** X8 in full (LLM-boundary rule + ScriptedToolAgent, Clock/key seams,
fakeredis + CI redis job + parametrized store-contract suite, matrix → ~30 items, A31 demo or
waiver).

### 9. Ease of use
**Strengths:** teachable algebra; flat TurnResult; near-free author migration (shim +
inventory + new-style generators); exported marker constant; uniform headline invariant.
**Friction:** (H) A10×A28 author-bug demotion + error-site distance → X3; (H)
`process_message` silent return-type swap at 3.0 → X10(a); (M) `completed+success=False`
ambiguity (extraction error vs max-iters) → X10(b); (M) headline-never-carries-payloads makes
naive clients silently lose data (the original bug's symptom, by design) → X10(c); (M)
ask_user role inversion misleads generic readers → X10(c); (M) ~10 concepts for record
readers with no shipped reader/projection → X10(d); (L) shim doesn't cover author read-sites.
**Recommendations:** X10 in full + docs (status×success matrix, record-reading cookbook,
updated examples at 2.21).

### 10. Ease of deployment
**Strengths:** loss not hidden in a minor; loud version coupling; graceful forward expiry;
zero-init bootstrap; one-command retention shape.
**Risks:** (H) no rollback story (split-brain history, forward-only expiry, client lockstep)
→ X4; (H) canary/blue-green impossible — 3.0 is an undocumented big-bang → X4; (M) drain
unverifiable (pending store deliberately non-enumerable) → X5(b); (M) loss announcement
mechanism undesigned → X12; (M) readiness ignores the load-bearing store; no preflight →
X5(c,d); (L/M) zero-config 2.21 unasserted; disk default at 3.0 contradicts A29; (L)
operator deliverables are prose.
**Recommendations:** X4 (rollback amendment + cutover runbook), X5, X12 (loss announcement,
`FW_ALLOW_DISK_STORES` guard, env template updates).

---

## Suggested disposition

1. **Adopt as a final amendment round (A48+)** before implementation: X1, X2, X3, X4, X5(a)
   (config inventory), X9(a,b,e) — these change contracts and are an order of magnitude
   cheaper to fix on paper than in code.
2. **Fold into the implementation plan** (work items, not design changes): X5(b,c,d) admin
   CLI/preflight/probe, X6 minimal-2.21 cut + defer list, X7 metrics pipeline, X8 test seams
   and fixtures, X10 ergonomics, X11 consolidated spec + module layout.
3. **Explicitly accept** (document, don't build): the remaining X12 items chosen against.
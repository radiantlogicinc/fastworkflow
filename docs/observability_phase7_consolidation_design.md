# Observability Phase 7 — Conversation-Store Consolidation (Design Slice)

> **Status (2026-08-27): LANDED in v3.2.0** (beads `fix-24f.8`, `fix-gxr`). The
> observability DB is the single source of truth. `conversation_store.py` is
> **deleted** — ruling C5 deferred that to a separately approved change, which
> was granted. Its tests were ported to observability-store equivalents rather
> than dropped (topic uniquification, windowed reads, cross-process WAL); only
> the pre-sqlite-migration RocksDB artifact test was retired outright, having no
> surviving behaviour. Deviations from the design as written, all deliberate:
>
> - **`legacy_floor` at mint is gone, not ported.** It existed to stop a fresh
>   observability DB re-issuing an id that already named a per-channel-DB
>   conversation, which is only a hazard while BOTH stores are written (an
>   aliased id appended new turns into the user's oldest legacy conversation).
>   Post-cutover nothing reads those files, so the per-channel counter alone is
>   authoritative — §3's posture is "legacy-only conversations become
>   invisible", not "misattributed". The one-shot per-channel topic dedupe from
>   ruling I9 is likewise unlanded, for the same reason: it reconciled two live
>   stores.
> - **`finalize_conversations_on_shutdown` is deleted, not re-scoped.** Ruling
>   C5 said its turn-append job was obsolete and its labeling job stayed, but
>   labeling had already been removed from shutdown before this slice, so
>   nothing was left.
> - **A suspended-then-resumed exchange is now ONE conversation turn**, where
>   the legacy store recorded each half separately. Falls out of one logical
>   turn = one row.
> - **With `FW_OBSERVABILITY=0` the server has no conversation persistence at
>   all** (in-memory history only, nothing survives a restart), because the
>   turn record IS the conversation record now. `run_fastapi_mcp` is an entry
>   point, so the sink defaults on.

> Errata (2026-08-26): CLI names herein predate the chatbot UI rework —
> `fastworkflow studio` shipped as `fastworkflow run_chatbot`, and the
> `--prune`/`--forget-channel`/`--include-conversations` flags became the
> chatbot's confirmed Clear-conversations action plus automatic
> startup pruning (see the parent design doc's §3.2/§3.4 amendments).

Status: **REVISED after adversarial review round 1** (two independent lenses,
22 findings; rulings in §6 — where §6 conflicts with §2/§3, §6 wins).
Implementation may begin per the amended design. Rulings were made
autonomously per the standing "proceed" instruction and are subject to Dhar's
review before commit; the two flagged decisions needing his explicit approval
are marked **[DHAR]**.
Author: Claude (with Dhar Rawal), 2026-08-25.
Parent design: `docs/fastworkflow_observability_studio_design.md` §3.3 (Phase B
gates, all normative). Prior art: `docs/turn_result_design_final.md` §7, §14.

## 0. Goal

End the Phase-A dual-write period. After this slice, the per-workflow
`observability.sqlite3` is the **single source of truth** (decision D1) for
conversations, turn records, memory rebuild, and feedback. Retired:
`run_fastapi_mcp/conversation_store.py`, the per-channel
`conversations/<channel_id>.sqlite3` DBs, and the cwd `action.jsonl` mirror.

## 1. Current state (post Phases 1–6, this tree)

- Turn records, spans, and artifacts land in `observability.sqlite3` via
  `SQLiteTraceSink` (best-effort, background writer).
- Conversation ids are minted by the observability DB
  (`reserve_conversation_id` chokepoint); the legacy store consumes the same
  ids (`sync_conversation_id_floor`). Labels are mirrored via
  `record_conversation_label` [R15].
- The legacy `ConversationStore` still owns: memory restore
  (`get_conversation_window` → `restore_history_from_turns`), incremental turn
  persistence (`append_conversation_turns` + `durable_turn_count` high-water
  mark), feedback rewrite (`update_last_conversation_turn`), label-due state
  (`get_conversation_label_state`), topic lookup (`get_conversation_by_topic`),
  listings (`list_conversations`, `get_all_conversations_for_dump`), and
  topic-uniqueness (`_ensure_unique_topic`).
- The **turns table's `conversation_summary`/`conversation_traces` columns are
  never populated** (serializer writes NULL) — the 3-key memory shape lives
  only in the in-memory history and in legacy turn dicts.
- Insights distillation (`distillation.py`) reads/deletes cwd `action.jsonl`;
  `ChatSession` constructs its WEC with `mirror_action_log_to_file=True`.

## 2. Design

### 2.1 Populate the memory columns at finalize (prerequisite for gate 1)

`WEC._build_turn_result` runs after the turn's conversation-history entry is
appended (agent path: `_finalize_agent_output` → `summarize_and_record_turn`;
deterministic path: `append_conversation_turn` inside `_process_message` /
`_process_action`). At finalize, stamp the newest history entry's
`"conversation summary"` and `"conversation_traces"` onto the `TurnResult`
(new optional fields `conversation_summary` / `conversation_traces`, additive)
**only when that entry was appended during this turn** (guard: capture
`len(history.messages)` at `_begin_turn`; stamp only if it grew).
`serialize_turn_result` writes them into the existing columns. An
`awaiting_user` emission leaves them NULL; the terminal upsert fills them.

### 2.2 Feedback goes to the `feedback` table

The FastAPI feedback path (`save_last_turn_feedback` /
`update_last_conversation_turn`) additionally upserts
`ObservabilityStore.upsert_feedback(turn_key, feedback_json)` where `turn_key`
is the newest turn of the active conversation (query: max ordinal). The
feedback table is mutable by design [R3]; turn records stay write-once.
During the deprecation window both writes happen; the legacy write is deleted
with the store.

### 2.3 Memory rebuild from the new store (gate 1, [R3])

New read: `ObservabilityStore.get_memory_window(channel_id, conversation_id,
max_turns)` → newest `max_turns` turns (ordered by ordinal, `LEFT JOIN
feedback`) projected to the canonical 3-key dicts
(`{"conversation summary", "conversation_traces", "feedback"}`), oldest-first.
`_create_channel_runtime` restore switches from
`conversation_store.get_conversation_window` to this; conversation to restore
= `MAX(conversation_id)` for the channel (replaces
`get_last_conversation_id`, including its step-back-one fallback: if the
newest conversation has no turns, restore the previous one).

`durable_turn_count` (high-water mark for incremental legacy saves) is
**deleted**: every turn is durably recorded by the sink keyed by `turn_key`,
so there is nothing incremental to track. The label-due milestone check
(`_label_is_due`) counts turns rows
(`SELECT COUNT(*) FROM turns WHERE channel_id=? AND conversation_id=?`).

### 2.4 Turn-record writes: synchronous-or-acked in the turn path (gate 2, [R14])

`SQLiteTraceSink.emit_turn_record` becomes **synchronous by default**: the
turn row + artifact rows are written on the caller thread in one short
`BEGIN IMMEDIATE` transaction with a dedicated short busy timeout
(`FW_OBS_SYNC_WRITE_TIMEOUT_S`, default 5s). On timeout/failure it falls back
to the existing record queue (acked-or-queued-with-warning; counted in writer
health). Spans/artifacts emitted via `emit_span` stay fire-and-forget on the
background writer. This honors the turns engine's persist-before-DONE without
new plumbing: by the time `process_turn` returns, the record is durable (or
counted as degraded). Typical cost: ~1 ms/turn on local SSD.

Trade-off reviewed: a wedged DB can add up to the sync timeout to a turn once
(then the sink circuit-breaks to queued mode for
`FW_OBS_SYNC_BREAKER_COOLDOWN_S`, default 60s, so a broken disk degrades to
Phase-A behavior instead of taxing every turn).

### 2.5 Labeling and topic lookup on the new store

- `generate_topic_and_summary` (pure LLM call) moves from
  `conversation_store.py` to `fastworkflow/conversation_labeling.py`
  unchanged, with its timeout plumbing.
- Turn summaries for labeling read from the turns table
  (`conversation_summary` column) instead of legacy turn records.
- Topic uniqueness ports to the store:
  `ObservabilityStore.ensure_unique_topic(channel_id, topic,
  exclude_conversation_id)` — same case/whitespace-insensitive collision
  suffixing, same blank-exemption, one indexed scan of the channel's
  conversations. `record_conversation_label` applies it (topic writes keep the
  blank-preserving COALESCE policy).
- `/activate_conversation` topic lookup:
  `SELECT ... FROM conversations WHERE channel_id=? AND lower(trim(topic)) =
  lower(trim(?))`.
- Listings (`/conversations`, admin dump) read from
  `list_conversations` / turns rows.

### 2.6 Retention (gate 3, [R16])

Already true in Phase 2: `prune()` touches spans/artifacts only. This slice
adds the operator opt-in: `fastworkflow studio --prune --include-conversations
--older-than-days N` deletes conversation rows AND their turns/feedback beyond
the horizon. Default prune remains exempt.

### 2.7 Distillation and the action.jsonl mirror (gate 4, [R25])

- `distillation.py` reads the in-process `ctx.action_log` (already the source
  the file mirrors) instead of reading/deleting cwd `action.jsonl`;
  `_reset_action_log` becomes `ctx.clear_action_log()`.
- `ChatSession` stops passing `mirror_action_log_to_file=True`; the mirror
  code in `append_action_log` and the file-append fallback in
  `workflow_agent._append_action_record` are removed. (The WEC parameter stays
  one release for external consumers, defaulting False, documented as
  deprecated.)
- The diagnostics skill's `trace_turn.py`
  (`.claude/skills/fastworkflow-diagnostics-and-tooling/scripts/`, team-private,
  untracked) is updated to read spans/turns from `observability.sqlite3`.

### 2.8 Retirement

Delete `run_fastapi_mcp/conversation_store.py` and every import. Per-channel
`conversations/*.sqlite3` files are **left on disk untouched** (readable by
older builds; deletable by the operator). `state_paths.conversations_dir` stays
(documented as legacy) until the next major.

## 3. Migration and compatibility (the breaking-adjacent part)

- **One-way cutover, no data migration.** A deployment upgrading across this
  slice starts reading conversations from `observability.sqlite3`, which for
  pre-existing channels contains only what Phase A dual-wrote since the
  upgrade to Phase A. Conversations that exist only in legacy per-channel DBs
  become invisible to `/conversations` and to memory restore (the files
  remain on disk). This mirrors the accepted v3.0 posture ("rollback loses
  3.0-era conversations", final spec §14) and MUST be called out in release
  notes.
- Optional (explicitly out of scope unless review demands it): a one-shot
  `fastworkflow studio --import-legacy-conversations` backfill.
- Mixed operation (some processes Phase A, some Phase B) is forbidden for a
  workflow, same as the v3.0 mixed-fleet rule.

## 4. Test plan

- Memory-window parity: seed turns via the sink, restore a WEC through
  `_create_channel_runtime`, assert the rebuilt `dspy.History` matches the
  3-key shape the legacy path produced (including feedback join and the
  step-back-one empty-conversation fallback).
- Sync-or-acked: a turn's row is durable before `process_turn` returns; a
  locked DB degrades to queued mode without failing the turn and the breaker
  cools down.
- Labeling: milestone due-ness off turn counts; topic uniqueness collision
  suffixing on the store; `/activate_conversation` by topic.
- Feedback: `/feedback` lands in the feedback table and in the restored
  memory window.
- Distillation drives a scripted session and reads `ctx.action_log` with no
  `action.jsonl` in cwd before/after.
- Retirement: no import of `conversation_store` anywhere;
  FastAPI suites green with the module gone.

## 5. Review questions for the adversarial pass

1. Does stamping memory columns at finalize race with anything on the resume
   path (history appended by `_finalize_agent_output` vs `_build_turn_result`
   ordering, distillation path included)?
2. Is the sync-write circuit breaker the right [R14] realization, or should
   the turns engine explicitly await a sink ack?
3. Any consumer of `durable_turn_count` semantics beyond incremental save +
   label milestones?
4. Does deleting the legacy store break `/admin/dump_all_conversations`
   consumers that expect the hydrated turn shape?
5. Is silently hiding legacy conversations acceptable, or does the backfill
   move into scope?
6. CLI parity: the CLI (Topology A) never had conversation persistence — after
   consolidation its turns ARE persisted (turn records). Does anything assume
   CLI statelessness?

## 6. Review round 1 — findings and rulings (normative; supersede §2/§3 where they conflict)

Reviews: integrity lens (I1–I10), compatibility/operations lens (C1–C12),
2026-08-25. Core insight accepted in full: the legacy store's
`durable_turn_count` was three correctness mechanisms (retry cursor, trim
gate, feedback position guard), and the consolidation must inherit each
explicitly.

| # | Finding (abridged) | Ruling |
|---|---|---|
| I1+I2 (CRIT) | Best-effort emit has no retry cursor; retiring the incremental save orphans the trim gate → silent permanent memory holes | `emit_turn_record` becomes **ack-returning** ("stored" / degraded). Failed terminal records enter a bounded sink-side pending-retry ring (keyed by turn_key, idempotent), retried on writer heartbeat and on subsequent emits. History trimming is **gated on stored**: un-acked turns defer the trim (window growth bounded by the ring). |
| I3+C4 (HIGH) | Max-ordinal feedback keying attaches feedback to suspended/missing turns | Feedback is keyed by the **actual turn_key of the last completed turn**, tracked on the WEC (`last_completed_turn_key`, set at terminal finalize and serialized with session state). No SQL inference. Feedback rows may be written before the turn row lands (join reunites them). Mismatch/no-key → skip with warning (ports the legacy guard). |
| I4+C3 (HIGH) | Turns-table rows ≠ history entries (cancelled/failed/NULL-summary/abandoned rows corrupt window, milestone count, labeling, step-back) | Invariant: every conversation-memory read (`get_memory_window`, milestone counts, labeling summaries, step-back emptiness) filters `status IN ('completed','failed') AND conversation_summary IS NOT NULL` ("usable rows"). Parity tests seed failed/cancelled/suspended/dropped turns. |
| I5 (HIGH) | §2.1 growth-guard baseline not serialized → cross-process resume can stamp the previous turn's summary onto a write-once row | Baseline is reconstructed at restore: `_apply_turn_accumulator` sets it to `len(restored history)` (history is applied before the accumulator). Also captured at `_begin_turn`. The guard re-reads `self._conversation_history` through the property (distillation swaps the object). |
| I6+C8 (HIGH/MED) | Sync/queue path mixing reorders ordinals and splits one turn across paths (spurious refused-terminal-write health noise; DONE→404 window) | ALL turn-record emissions go sync-first (awaiting_user and terminal both); the queue is only the degraded fallback. On fallback of a conversation-bound record, the **ordinal is reserved synchronously in a tiny transaction before enqueue**. Same-key identical-status refusals are not counted as anomalies. The breaker-mode DONE→404 window on GET /turns is documented behavior. |
| I7+C2-adj (MED) | Step-back-one restore leaves active-conversation binding unspecified; obs store mints rows at reserve (emptiness ≠ row absence) | On step-back, the stepped-back conversation becomes both the restored AND the active/bound conversation (legacy parity). Emptiness = zero usable turns rows. Checkpoint-restored `active_conversation_id` keeps precedence. |
| I8+C11 (MED) | Distillation port: in-place `clear()` aliasing + file-reset points not converted → cross-contaminated teacher/student trajectories | Port discipline: reads snapshot (`list(ctx.action_log)`); `ctx.clear_action_log()` at both `_reset_action_log` sites AND at distillation entry. Test asserts per-pass action counts. Both skill docs (run-and-operate, diagnostics) updated, not just trace_turn.py. |
| I9+C6 (MED/HIGH) | Topic uniqueness: TOCTOU across the async label queue; SQLite `lower()` is ASCII-only; **Phase A already mirrors the UNSUFFIXED topic** (live bug) | Uniquification runs inside the writer's `_apply_label` transaction (single enforcement point), with **Python-side casefold** comparison (scan channel topics, compare in Python — legacy semantics incl. blank exemption, self-exclusion, candidate renormalization). Live-bug fix: `update_conversation_topic_summary` returns the stored unique topic; the Phase-A mirror passes it. One-shot per-channel dedupe at first Phase-7 store open (tracked via diagnostics row). |
| I10+C5.iv (LOW) | Milestone check races the queued write; `turns_appended` replacement unspecified | `turns_appended` comes from the emit ack (1 if stored, 0 if degraded). Accepted deferral otherwise (blank-topic retry self-heals). |
| C1 (CRIT) | Erasure [R21] never touches legacy per-channel DBs — live compliance hole | The operator forget-channel command (studio `--forget-channel` / store API caller) also deletes `conversations/<channel_id>.sqlite3` (+`-wal`/`-shm`) while the legacy dir exists. Release notes state the Phase-A window's gap. |
| C2 (HIGH) | MAX-based minting aliases ids across cutover/rollback and reuses ids after forget/prune | Minting moves to a **per-channel counter row** (`conversation_counters` table; schema is still pre-release so user_version stays 1). Seeded at first mint per channel from `max(MAX(obs conversations), legacy meta.last_conversation_id)` — the legacy floor is passed in by the embedder chokepoint (`reserve_conversation_id` / restore), keeping the store decoupled from the legacy module. Counter never decreases (forget/prune cannot cause reuse). |
| C5 (HIGH) | Deletion list misses consumers; test plan under-scopes; **repo rule: tests cannot be deleted without approval** | Disposition: every production call site is switched to the new store; `run_fastapi_mcp/conversation_store.py` and its dedicated tests are **retired from production paths but NOT deleted** — physical deletion is a follow-up bead requiring **[DHAR]** approval. `trim_conversation_window` re-homed behind the emit ack (I1 ruling). `finalize_conversations_on_shutdown` re-scoped: its turn-append job is obsolete under sync-first writes; its labeling job stays. Checkpoints: `durable_turn_count` still serialized (written as 0) and ignored on read, so old checkpoints deserialize. Legacy-behavior tests are ported to obs-store equivalents where the behavior survives. |
| C7 (MED) | /conversations + admin dump shapes break; reserve-time conversation rows create visible phantoms | Projections specified: topic/summary NULL→""; timestamps ISO→ms epoch; `updated_at` column added to conversations (bumped on label and turn writes) for legacy ordering parity. Listings exclude conversations with zero usable turns (kills the phantom). Admin dump reconstructs the hydrated legacy shape (conversations ⋈ turns ⋈ feedback, 3-key dicts) with a shape parity test, plus `legacy_stores_present: true` marker when legacy files exist beside the DB. |
| C9 (MED) | "~1 ms" understates p99 (span-batch + multi-process contention); breaker re-arms blind; sync path inherits timeout=30 | Sync path uses its own connection with a short busy timeout (`FW_OBS_SYNC_WRITE_TIMEOUT_S`); breaker re-arms only after a cheap off-turn-path write probe succeeds; sync-write latency counter added to writer health. p99 documented as bounded by the busy timeout. |
| C10 (MED) | Conversation-less (CLI) turns survive every retention knob | The §2.6 opt-in prune also covers conversation-less turns older than the horizon. Default prune stays exempt per [R16]. |
| C12 (LOW) | §2.5 ports a dead path (topic lookup; activation is by id) | `get_conversation_by_topic` is NOT ported and no topic-lookup endpoint is added. §2.5's lookup SQL is struck. Topic uniqueness remains (listing display + future lookup). |
| Q4/Q5/Q6 | — | Q4: dump shape must be reconstructed + marker (see C7). Q5: backfill stays OUT of scope **provided** C1 + C2 land; without id seeding the posture would be "misidentified", worse than v3.0's "hidden". **[DHAR]** may still order a backfill importer. Q6: no code assumes CLI statelessness; amend retention (C10) instead. |

### Implementation order (amended)

1. Store: `conversation_counters` + seeded minting (C2); `updated_at` column;
   usable-rows read filters (I4); writer-side uniquification + casefold (I9);
   ack-returning sync-first emit + pending-retry ring + ordinal reservation
   (I1/I6/C8/C9); opt-in prune extension (C10).
2. Live Phase-A bug fixes: suffixed-topic mirror (I9), erasure of legacy files
   in forget-channel (C1).
3. WEC: memory-column stamping with serialized/reconstructed baseline (§2.1 +
   I5); `last_completed_turn_key` (I3).
4. FastAPI: restore/memory-window switch with step-back binding (I7); feedback
   by turn_key (I3/C4); labeling reads + `turns_appended` from ack (I10);
   trim gating (I2); shutdown finalizer re-scope, checkpoint compat (C5);
   endpoint projections + dump parity + phantom exclusion (C7).
5. Distillation + action.jsonl retirement with clear-point discipline (I8);
   skill-doc updates.
6. Retire production imports of conversation_store; module + its tests remain
   pending [DHAR] deletion approval.
7. Test plan per §4 amended by the rulings above (seeded failure-mode parity
   tests, dump shape test, ack/trim gating tests, dedupe test).

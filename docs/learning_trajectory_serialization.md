# Understanding: Trajectory Serialization + Topology B

A living checklist of what you should deeply understand. We update this together as we go.
Legend: [ ] not yet · [~] partial · [x] mastered

## Stage 1 — The Problem (the "why does this even exist")
- [x] 1.1 What "Topology A" is and how a turn flows through it (worker thread + queues)
- [x] 1.2 What `ask_user` does, and why it *blocks a thread* in Topology A
- [x] 1.3 What "Topology B" is and how it differs (suspend/resume, no blocked thread)
- [x] 1.4 Why "pins the session to one process" was the blocker for horizontal scale
- [x] 1.5 What was already durable (workflow context on disk) vs what was volatile (the ReAct trajectory)

STAGE 1 COMPLETE.

## Stage 2 — The Solution (the "what and how")
- [x] 2.1 The three "blockers fixed first": stable cme id, per-channel app workflow id, action.jsonl
- [x] 2.2 What exactly gets serialized (the SuspendedSessionState blob) and why each field
- [x] 2.3 `export_suspended` / `import_suspended` on the ReAct module
- [x] 2.4 `serialize_state` / `apply_serialized_state` on WorkflowExecutionContext
- [x] 2.5 The pluggable SessionStateStore (disk vs redis) and the factory
- [x] 2.6 "Cold rehydration, not full object serialization" — why startup is re-run
- [x] 2.7 The new ChannelSessionManager: LRU cache + cold rehydrate + eviction save
- [x] 2.8 Endpoint rewrite: process_message in executor, persist/clear on awaiting_user
- [x] 2.9 Trace streaming reimplementation for Topology B
- [x] 2.10 cancel_pending lifecycle

STAGE 2 COMPLETE.

## Stage 3 — Edge Cases & Design Decisions
- [x] 3.1 Why sticky routing per channel is still required (RocksDB single writer)
- [x] 3.2 Why JSON (not pickle) for the disk store; the MagicMock pickling lesson
- [x] 3.3 The command-context navigation-depth limitation (best-effort name only)
- [x] 3.4 Nested intent-clarification: how it now escalates instead of aborting
- [x] 3.5 action_log mirroring flag (CLI vs server)

STAGE 3 COMPLETE.

## Stage 4 — Broader Context (the "why it matters")
- [x] 4.1 What this unlocks operationally (many pods, mostly-idle sessions)
- [x] 4.2 What is still NOT solved / future work
- [x] 4.3 Backward compatibility: CLI ChatSession still Topology A

STAGE 4 COMPLETE.

## Key corrections internalized
- "Stateless" = pods hold no PERMANENT state, so a channel can be RELOCATED to any pod.
  It does NOT mean a channel can run on multiple pods concurrently (single-writer RocksDB).
- SESSION_STATE_STORE=redis only swaps the suspended-trajectory blob; domain context still
  persists in embedded RocksDB -> sticky routing still required.
- Sticky routing is fundamentally about "one owner for the live mutable session" (concurrency
  control + per-process live objects), deeper than the storage backend choice.

## Still-open / future work
- Sticky routing required but NOT enforced by code (infra must guarantee it).
- Orphaned pending blobs never expire (no TTL/reaper; eviction only saves; cancel_pending clears).
- Command-context navigation depth is best-effort (name only).
- schema_version present but no migration logic (mismatch only warns).

ALL STAGES COMPLETE.

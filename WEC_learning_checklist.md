# Learning Checklist: How `run_fastapi_mcp` uses `WorkflowExecutionContext` (Topology B), not `ChatSession`

> Living doc. We check items off only after you've *demonstrated* understanding (explained it back or answered a quiz). Nothing is "done" just because we discussed it.

Legend: `[ ]` not yet · `[~]` in progress · `[x]` mastered (you explained/quizzed it correctly)

---

## Part 1 — The Problem (why Topology B had to exist)

- [x] 1.1 What `ChatSession` (Topology A) actually is: queues + a `ChatWorker` thread + a blocking REPL loop.
- [x] 1.2 How `ask_user` worked in Topology A: it **blocks** on `user_message_queue.get()` until a human types.
- [x] 1.3 Why blocking `ask_user` is fine for a CLI but a disaster for a web service (thread held hostage, hang risk).
- [x] 1.4 Why one-thread-per-session does not scale to thousands of HTTP clients.
- [x] 1.5 Why transport (queues/threads) being entangled with execution logic blocked cross-process resume.

### Notes — Part 1
- Topology A: ChatSession = WEC + 3 queues + ChatWorker thread + blocking REPL loop.
- `ask_user` blocks on `user_message_queue.get()`; thread parked holding the whole trajectory on its stack.
- Three failures for web: (A) thread exhaustion, (B) unbounded hang / no timeout, (C) state trapped in stack frames → no serialize/move/restart.

## Part 2 — The Solution (Topology B: WEC at the edge)

- [x] 2.1 What `WorkflowExecutionContext` is: transport-free, synchronous, one-per-session execution core.
- [x] 2.2 The relationship: `ChatSession` *composes* a WEC (`self._core`); FastAPI uses a WEC **directly** and never touches `ChatSession`.
- [x] 2.3 The non-blocking `ask_user`: suspend-in-memory + `awaiting_user` output + resume on next `process_message`.
- [x] 2.4 How FastAPI runs a turn: `run_in_executor` (thread pool) + `asyncio.wait_for` timeout + per-channel `lock`.
- [x] 2.5 Which queues FastAPI wires (only `command_trace_queue`) and why the other two are `None`.
- [x] 2.6 `ChannelRuntime` holds a WEC; `.chat_session` is only a backward-compat alias returning `execution_context`.
- [x] 2.7 Durable suspended state: `serialize_state` / `apply_serialized_state` + `SessionStateStore` for cold rehydrate.
- [x] 2.8 `cancel_pending` and why a suspended Topology-B turn never "hangs."

### Notes — Part 2
- WEC docstring = the design in 3 phrases: transport-free synchronous core / non-blocking ask_user / ChatSession composes it.
- FastAPI builds a raw WEC (never imports ChatSession); wires ONLY command_trace_queue (SSE); user/output queues are None.
- A turn runs via `loop.run_in_executor(None, ...)` + `asyncio.wait_for(timeout)`; pooled thread borrowed per-call, returned at suspend.
- Two independent suspension states: (1) agent/trajectory (`_workflow_tool_agent`, `_awaiting_user`, `_pending_clarification_request`) cleared by `_reset_agent_suspension`; (2) turn accumulator (`_turn_outputs`, `_turn_key`, timings) reset by `_begin_turn`. Resume skips `_begin_turn` so Q+A = one logical turn.
- `cancel_pending` = deliberate, embedder-driven abandonment (no implicit timeout); default suspended state is "harmlessly waiting."

## Part 3 — The Broader Context (why this matters / what it impacts)

- [x] 3.1 Scaling: sessions become plain data in an LRU `OrderedDict`, evictable, no dedicated threads.
- [x] 3.2 Resilience: process can restart / evict and still resume an in-flight clarification.
- [x] 3.3 Concurrency safety: per-channel lock → 409 on concurrent turns; ContextVar isolates active workflow per thread.
- [x] 3.4 Separation of concerns: transport lives in embedders (FastAPI/CLI), execution lives in WEC.
- [x] 3.5 What you could now build/change because of this design.

### Notes — Part 3
- LRU `max_live_sessions` cap; evicting an `awaiting_user` session serializes it first → unbounded suspended sessions on disk.
- Cold rehydrate from `SessionStateStore` on next request, possibly on a different worker → true distributed scaling.
- Concurrency: per-runtime `asyncio.Lock` (→ 409) serializes a channel; ContextVar stack isolates active workflow across pool threads.
- Engine/chassis: WEC = engine; ChatSession (CLI) and FastAPI (HTTP) = two chassis. New embedders need no execution changes.

### The transport contract (what an embedder owes the WEC)
WEC owns exactly one thing: "execute one message correctly" (`process_message(msg) -> CommandOutput`). The embedder must provide:
1. Thread-pool offload (`run_in_executor`) — WEC is synchronous/blocking.
2. Session management — create/bind/look-up the per-channel WEC across requests.
3. Cleanup of dead/abandoned sessions — eviction, `close()`, `cancel_pending()`. WEC has no timeout/cleanup policy.
4. Concurrency control — per-channel lock + 409; WEC isn't safe against two overlapping turns on one session.
5. Timeout enforcement (`asyncio.wait_for`) + persistence/rehydration of suspended state.

**Unifying principle:** WEC owns "execute one message." Who/when/which-thread/how-long/abandoned-state = the embedder.

---

## STATUS: ALL ITEMS MASTERED ✅ (verified via explanation + quizzes)

---

## Notes & evidence (filled in as we go)
```


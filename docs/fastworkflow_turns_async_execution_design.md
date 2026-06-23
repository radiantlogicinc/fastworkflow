# Turns-Based Async Execution for `run_fastapi_mcp` — Final Design & Implementation Handoff

**Status:** Final, ready to implement
**Scope:** `fastworkflow/run_fastapi_mcp/` (FastAPI + MCP server). The motivating
workload lives in `talk_to_ido`, but **all code changes here are in the fastWorkflow repo**.
**Verified against:** `fastworkflow==2.21.6` (pinned by `talk_to_ido`, installed under `.venv`).
All line references below are from that release and were checked against the `.venv` source.
**Date:** 2026-06-23

> **Backward compatibility is NOT required** (request/response schemas may change).
> The synchronous response *shape* is preserved in Step 1 only to minimize client churn,
> not as a contract.

---

## 1. Executive summary

`POST /initialize` (and `/invoke_agent`, `/invoke_assistant`, `/perform_action`) executes
potentially very long LLM work **synchronously inside the HTTP request**. When the work is
slow (observed **~10 min**, requirement **up to 15 min**), the request outlives client/proxy
timeouts, the caller retries, and the retry returns **without the result**. The result is
computed and persisted server-side but never delivered.

The fix is one unifying idea:

> **A long operation must never live or die with an HTTP request/response cycle.**
> Run every unit of work as a **turn execution** owned by an in-process **turn registry**.
> Each endpoint *submits* a turn and waits a short, bounded window: if the turn finishes in
> time, return it inline (feels synchronous); otherwise the request returns while the
> execution keeps running, recoverable by polling.

### Implementation strategy (the key decision)

Build the **turns engine first**, then make the existing endpoints **thin wrappers** over it.
This is smaller than the alternative (it builds the mechanism once instead of a throwaway
stopgap) and funnels **all** per-channel state mutation through **one gated path**, which is
what actually fixes the concurrency bugs in §3.

- **Step 1 — Engine + thin synchronous wrappers (fixes the bug, scales vertically).**
  Implement the `TurnRegistry` + `submit_turn(...)` "wait-or-defer" helper + locked
  `_run_turn(...)`. Convert `/invoke_agent`, `/invoke_assistant`, `/perform_action`, and the
  `/initialize` startup into thin callers. Keep today's response shape on the fast path; on
  timeout **defer** (return `{turn_key, running}`) instead of aborting. Migrate off the
  **deprecated** `process_message` onto `process_turn`.
- **Step 2 — Expose the async surface (additive).** Add `GET /turns/{turn_key}` polling, real
  `202` deferred responses, a per-execution **trace replay buffer**, a **sized executor**, and
  **global backpressure**. Make `/initialize` strictly non-blocking (startup is just the first
  turn).
- **Step 3 — Distributed (arbitrary horizontal scale).** Durable turn store + external worker
  pool + sticky channel routing. Deferred until load/durability demands it.

### Critical correctness rule (do not get this wrong)

"Synchronous with timeout" must mean **wait-or-defer**, never **wait-or-abort**. The request's
short `wait_seconds` waits on a `done_event` wrapped in `asyncio.shield`; when it elapses, the
**request returns but the execution keeps running, still owned by the registry**. A retry must
rejoin the *same* execution via the per-channel **active-execution pointer**, never start a
second racing one. Today's `asyncio.wait_for(...)` → `504` path is exactly the bug (see §3.1).

---

## 2. What the framework already gives us (reuse, don't reinvent)

fastWorkflow 2.21.6 already ships the turn vocabulary. Do **not** invent a parallel status enum.

- `fastworkflow/turn.py`:
  - `TurnStatus` (`COMPLETED | AWAITING_USER | FAILED | CANCELLED | ABANDONED`, lines 83–91).
  - `TurnOutput` (line 168) — consumer-facing slice with `turn_key`, `status`, `failure_reason`,
    `answer`, `command_outputs`, computed `success`.
  - `TurnResult` (line 240) — internal system-of-record.
  - `mint_turn_key()` (line 93) — colon-free, lexicographically sortable id.
- `fastworkflow/workflow_execution_context.py`:
  - `process_turn(message) -> TurnOutput` (line 485) — **non-deprecated** path; same dispatch as
    `process_message` plus it builds the `TurnResult`.
  - `process_message(message) -> CommandOutput` (line 470) — **deprecated** (emits
    `DeprecationWarning`, lines 477–483).
  - `process_action(action) -> CommandOutput` (line 579) — **NOT deprecated**; calls `_begin_turn`
    but returns a bare `CommandOutput` (it does **not** build a `TurnResult`/`TurnOutput`).
- `fastworkflow/__init__.py` exports `TurnOutput`/`TurnStatus` (lines 286–297) — currently
  **unused by the server**.

**Two status axes — keep them separate:**

- **Turn outcome** — `TurnStatus` (already on `TurnOutput.status`). *What happened in the turn.*
- **Execution lifecycle** — a small new enum. *Where the async work is.*

```
ExecState = QUEUED | RUNNING | DONE | LOST
```

`DONE` means "a `TurnOutput` (or error) is available"; read the outcome from `TurnOutput.status`.
`LOST` is the in-process-only "process restarted, record gone" state (Step 1/2; Step 3 removes it).

---

## 3. Verified findings to fix (current `.venv` behavior)

### 3.1 Timeout breaks single-flight for the turn endpoints (HIGH)

Each turn endpoint holds `runtime.lock`, but `run_process_message`/`run_process_action` wrap the
executor call in `asyncio.wait_for`. On timeout the `HTTPException(504)` unwinds **through**
`async with runtime.lock`, releasing the lock — while the orphaned executor thread **keeps running
the full ~10-min call**, still mutating `ctx` and persisting after the response.

```390:402:.venv/lib/python3.12/site-packages/fastworkflow/run_fastapi_mcp/utils.py
    try:
        output = await asyncio.wait_for(
            loop.run_in_executor(None, _run),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError as exc:
```

Consequence: with `timeout_seconds=60` (default) and 10-min calls, the lock is free for ~9 min
while the thread runs, so a retry passes the `if runtime.lock.locked(): 409` guard, acquires the
lock, and starts a **second** execution mutating the **same** `ctx` — a real data race. The doc's
earlier claim that the in-process lock is sufficient single-flight holds **only absent a timeout**.

**Fix:** `submit_turn` defers instead of aborting (shield + `done_event`); the per-channel
active-execution pointer makes a retry rejoin the same execution.

### 3.2 Startup runs unlocked and without a timeout (HIGH)

In `initialize()` the runtime is created first, then the startup action runs via
`run_in_executor` with **no** `async with runtime.lock` and **no** `asyncio.wait_for`:

```615:622:.venv/lib/python3.12/site-packages/fastworkflow/run_fastapi_mcp/__main__.py
            try:
                loop = asyncio.get_running_loop()
                startup_output = await loop.run_in_executor(
                    None,
                    lambda: _run_startup_sync(
                        ctx, startup_command_str, startup_action
                    ),
                )
```

The moment the runtime exists, a concurrent/retried `/initialize` hits the "already exists" branch
(`__main__.py:563–575`) and returns **tokens with no `startup_output`**, while the original startup
is still running unlocked. A client holding those tokens can call `/invoke_agent` and race the
unlocked startup thread on the shared `ctx`.

**Fix:** startup becomes a normal `submit_turn(..., kind="initialize_startup")` under the same lock.

### 3.3 The "already exists" branch returns a silently-empty result (HIGH — the reported bug)

```563:575:.venv/lib/python3.12/site-packages/fastworkflow/run_fastapi_mcp/__main__.py
        existing_runtime = await session_manager.get_session(channel_id)
        if existing_runtime:
            logger.info(f"Session for channel_id {channel_id} already exists, generating new tokens")
            access_token = create_access_token(channel_id, user_id)
            refresh_token = create_refresh_token(channel_id, user_id)
            return InitializeResponse(
                access_token=access_token,
                refresh_token=refresh_token,
                token_type="bearer",
                expires_in=JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            )
```

A pure read-back of `workflow.context["defect_summary"]` does **not** fix the motivating trace: the
retry arrived ~15s **before** startup finished, so the summary did not exist yet. The branch must
return a **three-state** answer: `done` (with output) / `running` (with `turn_key`) / `failed`.
Never a normal response with an empty result.

### 3.4 Mutating endpoints that bypass `runtime.lock` (HIGH)

`new_conversation`, `post_feedback`, and `activate_conversation` mutate
`runtime.execution_context` (conversation history) with **no** `async with runtime.lock`
(`cancel_pending` *does* take it). These run on the event loop thread and can race a turn running
on an executor thread.

```1213:1219:.venv/lib/python3.12/site-packages/fastworkflow/run_fastapi_mcp/__main__.py
            runtime.active_conversation_id = next_id
            runtime.execution_context.clear_conversation_history()

            logger.info(f"Ready for new conversation {runtime.active_conversation_id} for session {channel_id}")
            return {"status": "ok"}
```

**Fix:** these endpoints must acquire `runtime.lock` (and reject with `409` if the channel has an
active execution) so all `ctx` mutation is serialized per channel. The `409` decision keys off the
**registry active-execution pointer** (`registry.has_active(channel_id)`), **not** `lock.locked()` —
see §4.4: with wait-or-defer the lock is released while a request defers and across `AWAITING_USER`,
so `lock.locked()` is no longer a reliable "is something in flight" signal.

### 3.5 Non-atomic session creation (MEDIUM)

`ensure_user_runtime_exists` does `get_session` (manager lock) → build `ctx` → `create_session`
(manager lock) as **two separate** lock acquisitions, and `create_session` overwrites
unconditionally. Concurrent cold requests for the same channel can both build and one overwrites
the other (wasted work; via the `get_session_and_ensure_runtime` dependency path it can even
double-run startup → double LLM spend).

**Fix:** single-flight session creation (idempotency key + per-channel creation guard).

### 3.6 Eviction can `close()` a live `ctx` mid-turn (MEDIUM, steady-state — missed by prior reviews)

`_evict_oldest_if_needed` (triggered inside `create_session`) evicts the LRU channel and calls
`runtime.execution_context.close()` **without checking for an active turn**:

```642:651:.venv/lib/python3.12/site-packages/fastworkflow/run_fastapi_mcp/utils.py
    async def _evict_oldest_if_needed(self) -> None:
        while len(self._sessions) > self._max_live_sessions:
            channel_id, runtime = self._sessions.popitem(last=False)
            if runtime.execution_context.awaiting_user:
                self.session_state_store.save(
                    channel_id,
                    runtime.execution_context.serialize_state(channel_id=channel_id),
                )
            runtime.execution_context.close()
```

With `max_live_sessions=2000` under load, a long-running turn's session can be evicted and its
`ctx` closed while a thread is still mutating it — worse than the restart-only `LOST` case because
it happens at steady state.

**Fix:** eviction must **skip channels with an active execution pointer** (never close a live
turn's `ctx`).

### 3.7 Graceful drain is meaningless for long turns (LOW, note)

`lifespan` waits only 30s for active turns (`wait_for_active_turns_to_complete(max_wait_seconds=30)`,
`__main__.py:335`) — irrelevant against 15-min turns. In Step 1/2 a restart yields `LOST`; Step 3
(durable executions) is the real fix. Make submits idempotent so a post-restart resubmit is safe.

---

## 4. Target architecture

### 4.1 Components (new file: `run_fastapi_mcp/turns.py`)

```
ExecState        # QUEUED | RUNNING | DONE | LOST   (new, small)

TurnExecution
  turn_key        # reuse TurnOutput.turn_key (mint_turn_key()); opaque handle
  channel_id
  kind            # initialize_startup | invoke_agent | invoke_assistant | perform_action
  idempotency_key # hash(channel_id + kind + normalized args); dedupes retried submissions
  exec_state      # ExecState
  result          # TurnOutput when DONE+ok (carries its own TurnStatus)
  error           # reason string when DONE+error
  task            # asyncio.Task running the turn
  done_event      # asyncio.Event set when exec_state -> DONE
  created_at, started_at, finished_at, ttl_expires_at

TurnRegistry
  _by_key: dict[str, TurnExecution]                 # keyed by turn_key
  _active_by_channel: dict[str, str]                # channel_id -> turn_key (the pointer)
  start_or_get_active(channel_id, kind, idempotency_key, run_turn) -> TurnExecution
  has_active(channel_id) -> bool                    # is there a live execution? (the 409 basis)
  get(turn_key) -> TurnExecution | None
  evict_terminal(now)                               # TTL eviction of DONE/LOST entries
```

The **per-channel active-execution pointer** (`_active_by_channel`) is the heart of single-flight:
it outlives `runtime.lock` (which is released at each suspension boundary) and makes retries
idempotent. It is also the seam Step 3 replaces with a cross-process guard.

**Construction-order contract (do not get this wrong — see §4.2).** `start_or_get_active` is the
sole owner of `TurnExecution` creation and task launch, and it must do them in this order, all under
the registry's own lock:

1. If `_active_by_channel[channel_id]` already points at a live execution whose `idempotency_key`
   matches, return that **existing** execution (the retry rejoins it). If it exists but the
   idempotency key differs, that is a `409`-worthy conflict (see `has_active`).
2. Otherwise build a fresh `TurnExecution` — mint `turn_key`, allocate `done_event`, set
   `exec_state=QUEUED` — and insert it into `_by_key` and `_active_by_channel` **before** launching
   any task.
3. Only then call `run_turn(execn)` to create the `asyncio.Task`, passing the **fully-built**
   execution, and assign it to `execn.task`.

This guarantees a concurrent waiter that observes the pointer always sees an execution with a valid
`done_event`; there is no window in which a caller can await on a half-built execution.

### 4.2 The one helper, four thin callers

```python
async def submit_turn(runtime, registry, work_fn, *, wait_seconds, kind, idempotency_key):
    # Single-flight: one active execution per channel. A retry with the same
    # idempotency_key rejoins the SAME execution rather than starting a new one.
    #
    # Construction order matters (§4.1): the REGISTRY owns TurnExecution creation
    # and task launch. It builds the execution (turn_key + done_event) and inserts
    # the pointer FIRST, then calls run_turn(execn) with the fully-built object.
    # The factory takes execn as a parameter, so there is no caller-side forward
    # reference (no `execn` used before assignment) and no half-built-execution race.
    execn = registry.start_or_get_active(
        runtime.channel_id,
        kind=kind,
        idempotency_key=idempotency_key,
        run_turn=lambda execn: asyncio.create_task(
            _run_turn(runtime, registry, execn, work_fn)
        ),
    )
    try:
        # shield: the request's wait window timing out must NEVER cancel the execution.
        await asyncio.wait_for(asyncio.shield(execn.done_event.wait()), wait_seconds)
        return execn          # fast path: caller renders inline (200)
    except asyncio.TimeoutError:
        return execn          # deferred: caller returns {turn_key, running} (202 in Step 2)
```

`_run_turn` (the only place that touches `ctx`):
1. acquire `runtime.lock` (per attempt — released on terminal `TurnStatus` **or** `AWAITING_USER`),
2. run the blocking `work_fn` in the **sized executor**,
3. collect traces (append to the replay buffer in Step 2),
4. run persistence (`save_conversation_incremental`, `persist_pending_after_turn`) **before** `DONE`,
5. set `exec_state = DONE`, then `done_event.set()`.

Each endpoint differs only in `work_fn`, and all return a `TurnOutput`:

- `/invoke_agent`     → `lambda: ctx.process_turn(user_query.lstrip("/"))`
- `/invoke_assistant` → `lambda: ctx.process_turn(prefixed_assistant_message)`
- `/perform_action`   → `lambda: ctx.process_action_turn(action)`  *(see §4.3)*
- `/initialize` startup → `lambda: ctx.process_action_turn(startup_action)` or `process_turn(cmd)`

### 4.3 Small framework change: `process_action` → `TurnOutput`

`process_action` returns a bare `CommandOutput`, not a `TurnOutput`, so the registry would store two
result types. Add a `process_action_turn(action) -> TurnOutput` to `WorkflowExecutionContext`
(mirror of `process_turn`: call `_begin_turn`, `_process_action`, then `_build_turn_result`). This is
a small, isolated addition in the framework so the registry stores exactly one result type.

### 4.4 Lock discipline (read this twice)

- `runtime.lock` is held **per run/resume attempt**, released on terminal `TurnStatus` **or** when
  the turn suspends to `AWAITING_USER`. **Never** hold it across `AWAITING_USER` — the resume is a
  new request that must acquire the lock, so holding it would deadlock.
- The **registry pointer**, not the lock, carries the execution across the deferring request and
  across suspension.
- **The `409` "busy" guard keys off the registry active-execution pointer, not `lock.locked()`.**
  Under the old synchronous model `if lock.locked(): 409` worked because the lock was held for the
  whole turn. With wait-or-defer that is no longer true: the lock is **released while a request
  defers** (the execution keeps running with the lock free) and **across `AWAITING_USER`**. So
  `lock.locked()` would report "free" while a turn is very much in flight, reintroducing the §3.1
  race. All "is the channel busy?" decisions (the mutating endpoints in §3.4, and single-flight for
  turn endpoints) must therefore use `registry.has_active(channel_id)` / the active pointer. The
  pointer is the single source of truth for both *liveness* and *lifetime/idempotency*.
- Reading the pointer and reacting is **atomic enough** in a single event loop: registry mutations
  happen under the registry's own lock and there is no `await` between checking `has_active(...)` and
  acquiring/declining, so no new atomic primitive is needed for single-flight *correctness*.
- Cross-**process** single-flight (multiple replicas) is **not** solved in-process — deferred to Step 3.

---

## 5. API surface

### 5.1 `/initialize` (non-blocking; startup is the first turn)

`/initialize` mints tokens + ensures the session and **never blocks on long work**. If a
`startup_action`/`startup_command` is present it is submitted via `submit_turn(kind="initialize_startup")`
and handled with wait-or-defer. The **"already exists" branch returns the active/last startup
execution** (three states), never a silently-empty result.

```
POST /initialize
  -> 200 {tokens, startup_turn_key, exec_state, startup_output?}        # startup finished in window
  -> 202 {tokens, startup_turn_key, exec_state:"running"}              # startup still in flight
  (already-exists behaves identically: returns the SAME startup execution's state)
```

`InitializeResponse` gains: `startup_turn_key: Optional[str]`, `startup_exec_state: Optional[str]`,
`startup_error: Optional[str]` (keep existing `startup_output: Optional[CommandOutput]`).

`InitializationRequest` gains `timeout_seconds: int = 60` — the startup-turn wait window, with the
**exact same shape and default as the turn endpoints** (`InvokeRequest`/`PerformActionRequest.timeout_seconds`).
There is no separate server-side startup-wait constant: all four submitting endpoints take the wait
window from `request.timeout_seconds`. Like the turn endpoints in Step 1, this single knob doubles as
`wait_seconds`; the clean `wait_seconds`/`deadline_seconds` split (§5.2) is a Step 2 refinement applied
uniformly across all submitting endpoints.

> **Why `/initialize` does *not* reuse the turn endpoints' `_turn_json_response` renderer:** the two
> responses are different envelopes. `/initialize` returns the typed `InitializeResponse`
> (`response_model`) carrying **JWT tokens** on every reply and a **typed `startup_output:
> CommandOutput`** (so clients can read `command_responses`/artifacts); a failed startup is
> `200 + startup_error` (the session still initialized). The turn endpoints return a bare turn dict
> with a *synthesized* top-level `command_responses`, no tokens, and map a failed execution to `500`.
> The shared piece — the execution→three-state mapping (running→202, done→200+result, error) — is
> mirrored in `_initialize_response_from_execution`, the `/initialize` analog of `render_turn_response`.
> The *pattern* is shared; the *function* is not, because the envelope differs.

### 5.2 Turn endpoints

```
POST /invoke_agent      -> 200 {turn_key, exec_state, status, result?} | 202 {turn_key, exec_state:"running"}
POST /invoke_assistant  -> 200 {turn_key, exec_state, status, result?} | 202 {turn_key, exec_state:"running"}
POST /perform_action    -> 200 {turn_key, exec_state, status, result?} | 202 {turn_key, exec_state:"running"}
GET  /turns/{turn_key}   -> 200 {turn_key, exec_state, status?, result?, error?, traces?}   # Step 2
GET  /turns/{turn_key}/events -> SSE/NDJSON over the replay buffer (optional, Step 2)
```

**HTTP semantics:** `200` = finished in window, `202` = deferred (keep polling). `GET /turns`
transport `200` means "execution state read successfully"; the *turn's* outcome lives in
`TurnStatus`. **Do not** map a failed turn to HTTP 5xx.

**Two distinct, separately-named knobs** (replace the single `timeout_seconds`):
- `wait_seconds` — how long the *request* blocks before deferring (short; per-surface default).
- `deadline_seconds` — the *execution* ceiling (LLM-level timeout + turn TTL). Defer ≠ abort.

> **Deferred decision (point 3 — hard cancellation):** in **Step 1/2** `deadline_seconds` is
> **advisory**, not a hard kill. `asyncio.wait_for` cannot cancel an executor thread (§7.1), so a
> runaway turn is bounded only by an LLM-level timeout plus the turn TTL — it is not forcibly
> terminated. True deadline enforcement (a deadline that cancels the in-flight LLM HTTP request)
> requires `litellm.acompletion` and lands in **Step 3**. Accepted limitation for Step 1/2.

**Auth scoping:** `GET /turns/{turn_key}` MUST verify the JWT channel/user owns `turn_key`. The key
is unguessable but must not be the sole authz (avoid cross-tenant leakage).

### 5.3 MCP surface (decide explicitly)

`mcp_specific.py` currently **excludes** `rest_initialize`, `perform_action`, `rest_invoke_agent`,
`refresh_token`; the primary MCP tool is streaming `/invoke_agent_stream` (operation_id
`invoke_agent`). Decide explicitly whether `GET /turns/{turn_key}` is exposed as an MCP tool and
give MCP callers a **longer default `wait_seconds`** (they may not poll gracefully).

> **Deferred decision (point 4 — MCP surface):** whether `GET /turns/{turn_key}` becomes an MCP tool,
> and the exact longer MCP `wait_seconds` default, are **left open and deferred to before Step 2
> wiring** (the async surface). Step 1 does not change the MCP mount. Resolve this decision when
> `GET /turns/{turn_key}` is actually built.

---

## 6. Implementation steps & acceptance criteria

### Step 1 — Turns engine + thin synchronous wrappers

**Build**
- `run_fastapi_mcp/turns.py`: `ExecState`, `TurnExecution`, `TurnRegistry`, `submit_turn`, `_run_turn`.
- A module-level `TurnRegistry` instance (alongside `session_manager`).
- `process_action_turn() -> TurnOutput` in `WorkflowExecutionContext` (§4.3).

**Wire**
- `/invoke_agent`, `/invoke_assistant`, `/perform_action` → call `submit_turn`; migrate
  `process_message` → `process_turn`.
- `/initialize` startup → `submit_turn(kind="initialize_startup")`; "already exists" branch returns
  the three-state startup execution.
- `new_conversation`, `post_feedback`, `activate_conversation` → wrap mutations in `runtime.lock`
  and reject with `409` when `registry.has_active(channel_id)` (the active-execution pointer, **not**
  `lock.locked()`) (fix §3.4, see §4.4).
- Eviction skips channels with an active execution pointer (fix §3.6).
- Single-flight/idempotent session creation (fix §3.5).

**Acceptance**
- A turn that exceeds `wait_seconds` returns a deferred response **without** orphaning the execution;
  a retry with the same args returns the **same** `turn_key` and never starts a second LLM call
  (assert single `dspy.Predict` invocation via the existing `[initialize_defect_info] ... took` log).
- No `ctx` mutation happens outside `runtime.lock` (audit all endpoints).
- `DeprecationWarning` for `process_message` no longer emitted by the server.
- The §3.3 trace is fixed: a retry that arrives before startup completes receives
  `exec_state:"running"` + `startup_turn_key`, never an empty `startup_output`.
- `tests/load/load_test_talk_to_ido.py --mode initialize --unique-payload` still passes (fast path
  returns `startup_output` inline within `wait_seconds` when the LLM is fast).

### Step 2 — Async surface (additive)

**Build**: `GET /turns/{turn_key}` (auth-scoped); real `202` deferred responses; per-execution
**trace replay buffer** (replace the destructive `get_nowait` drain so a poller and a streamer can
both read by offset — `utils.py:487/507/532/538/566`); a **dedicated sized executor**
(`loop.set_default_executor` or a private pool) separate from the default pool; a **global
semaphore/queue** returning `429`/`queued` when saturated; TTL eviction of terminal executions.

**Acceptance**: a 10-min turn is fully recoverable via `GET /turns/{turn_key}` across many polls;
streaming and polling read the same record; saturation returns `429` rather than exhausting threads;
terminal results are TTL-evicted (registry does not grow unbounded).

### Step 3 — Distributed (when load/durability demands)

Durable turn store + queue (Redis/Arq/RQ/Celery/SQS); external worker pool sized to safe LLM
concurrency; stateless API replicas; sticky channel→worker routing or rehydratable `ctx`
(`serialize_state`/`apply_serialized_state`/`SessionStateStore`/`ConversationStore` exist but not for
live mid-turn state); leasing + heartbeat (visibility timeout > 15 min); prefer `litellm.acompletion`
so a deadline cancels the in-flight HTTP request instead of orphaning a thread. Eliminates `LOST`.

---

## 7. Things to watch out for

1. **`asyncio.wait_for` does not cancel executor threads.** In wait-or-defer this is *intended*
   (execution outlives the request) — use `shield` + `done_event`. A runaway turn can only be bounded
   by an LLM-level timeout (`deadline_seconds`) + a turn TTL, not truly killed.
2. **Never hold `runtime.lock` across `AWAITING_USER`** — deadlocks the resume. Hold per attempt; the
   registry pointer carries the execution (§4.4).
3. **Persist before `DONE`** — `save_conversation_incremental`/`persist_pending_after_turn` run inside
   the turn-completion path, under the lock, **before** `exec_state=DONE`; otherwise a poll sees "done"
   while the conversation is unsaved.
4. **Result source for startup is a string, not a typed output.** `workflow.context["defect_summary"]`
   is a plain `str`, and `ConversationStore` persists summaries/traces, not a `CommandOutput`. The
   registry's `TurnOutput` (from `process_action_turn`) is the clean source of truth — don't try to
   reconstruct `startup_output`'s type from the conversation store.
5. **Idempotency key is mandatory.** Without it, client/proxy retries spawn duplicate LLM calls — the
   exact behavior that produced three ~10-min calls in the trace. Key on
   `hash(channel_id + kind + normalized args)` (or a client-supplied key).
6. **`wait_seconds` defaults per surface.** HTTP: short (1–3s) then poll. MCP: longer (clients may not
   poll gracefully).
7. **`LOST` after restart/deploy.** In Step 1/2 the in-process registry answers post-restart polls with
   `exec_state=LOST`; make resubmits idempotent and return a clear "execution lost, resubmit."
8. **Two-level concurrency.** Per-channel single-flight does not bound *global* concurrency (≈ active
   channels). Add the global semaphore/queue (Step 2) or you reintroduce thread exhaustion.

---

## Appendix A — Verified code references (fastWorkflow 2.21.6, under `.venv`)

- `fastworkflow/turn.py` — `TurnStatus` (83–91), `mint_turn_key()` (93), `TurnOutput` (168),
  `TurnResult` (240).
- `fastworkflow/workflow_execution_context.py` — `process_message()` deprecated (470, warns 477–483),
  `process_turn()` (485), `_build_turn_result()` (525), `process_action()` (579, no `TurnResult`).
- `fastworkflow/__init__.py` — exports `TurnOutput`/`TurnStatus` (286–297), unused by server.
- `fastworkflow/run_fastapi_mcp/__main__.py` —
  - `initialize()` already-exists branch returns no `startup_output` (563–575); startup via
    `run_in_executor`, **no** `wait_for`, **not** under `runtime.lock` (615–622); startup persisted
    only after the call (632).
  - `invoke_agent` (757) `rest_invoke_agent`; `invoke_agent_stream` (832) op_id `invoke_agent`;
    `invoke_assistant` (997); `perform_action` (1072) — all use the `if lock.locked(): 409` + `async with lock` pattern (778/859/1017/1091).
  - `new_conversation` (1174), `post_feedback` (1286), `activate_conversation` (1349) — **no**
    `runtime.lock`; `cancel_pending` (1142) — **has** the lock.
  - `lifespan` drain `wait_for_active_turns_to_complete(max_wait_seconds=30)` (335).
- `fastworkflow/run_fastapi_mcp/utils.py` —
  - `_run_startup_sync()` (243) calls `process_action`/`process_message`.
  - `ensure_user_runtime_exists()` (258) — non-atomic get-then-create; in `/initialize` it is called
    with `startup_*=None` (creation only), startup runs in the endpoint afterward.
  - `run_process_message()` (376) / `run_process_action()` (408) — `asyncio.wait_for(run_in_executor(...))`
    over the **deprecated** `process_message`/`process_action`.
  - destructive trace drain `get_nowait` (487/507/532/538/566).
  - `ChannelSessionManager` (610), `_evict_oldest_if_needed()` closes live `ctx` (642–651),
    `max_live_sessions=2000` (621).
- `fastworkflow/run_fastapi_mcp/mcp_specific.py` — MCP mount excludes `rest_initialize`,
  `perform_action`, `rest_invoke_agent`, `refresh_token` (48–56).

## Appendix B — `talk_to_ido` trigger (workload, not the fix site)

`talk_to_ido/_commands/initialize_defect_info.py::_process_command` runs
`dspy.Predict(SummaryGenerationSignature)` against `LLM_RESPONSE_GEN` (`litellm_proxy/large-model-name`),
sets `workflow.context["defect_summary"]` (line 187), and returns it. Observed durations:
619.9s / 614.8s / 34.2s. The summary is recoverable server-side, but the fix belongs in
`run_fastapi_mcp`, not in this command.

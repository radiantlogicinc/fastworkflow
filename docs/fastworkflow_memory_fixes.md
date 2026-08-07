# fastworkflow memory fixes (for upstream submission)

Changes applied to a local copy of **fastworkflow 2.24.1**, to be fixed in the fastworkflow repository. Every change below is
implemented and verified; measurements are from a production workload whose
command payloads are ~450 KB.

## Problem

`run_fastapi_mcp` grows resident memory monotonically with the *number of
requests processed*, with **no concurrency involved**, until the container is
OOM-killed. Measured at **1.76 MB per request**. Growth is roughly **4x the
request payload size**, so payload size — not request rate — sets the OOM
horizon: at a 1 GB limit the process died after ~200 sequential requests.

Four unbounded retainers were found. Two are in fastworkflow's own state
management; two are DSPy globals a long-running server should switch off.
Bounding them exposed a fifth issue — eviction silently dropping workflow
context — which had to be fixed before the new bounds could ship.

| # | Retainer | Location | Share of growth |
|---|---|---|---|
| 1 | `TurnRegistry._by_key` | `run_fastapi_mcp/turns.py` | ~260 MB / 300 req |
| 2 | `_creation_locks` | `run_fastapi_mcp/utils.py` | small, unbounded |
| 3 | Session cache (cap 2000, no TTL) | `run_fastapi_mcp/utils.py` | ~135 MB / 300 req |
| 4 | `dspy` `GLOBAL_HISTORY` + `settings.trace` | `run_fastapi_mcp/__main__.py` | ~130 MB / 300 req + ~0.45 MB/req |
| 5 | Workflow context lost on eviction | `run_fastapi_mcp/utils.py` | correctness, not memory |

---

## 1. `TurnRegistry` retains every execution forever

**File:** `fastworkflow/run_fastapi_mcp/turns.py`

`_by_key` is never pruned. `clear_active()` drops only the *active pointer*, and
`evict_terminal()` was a documented no-op because `ttl_expires_at` was never
assigned ("TTL eviction is Step 2"). Every completed turn therefore pins its
action payload, the `raw_command` string built from it in
`workflow_execution_context.py` (a second full copy), and its result, for the
lifetime of the process.

Two bounds are now applied. The count cap is the one that actually guarantees
memory is bounded — under sustained load turns complete faster than any TTL —
and it is derived from `MAX_LIVE_SESSIONS` so a deployment has a single knob for
overall footprint.

```python
TURN_RETENTION_SECONDS = 300.0
TERMINAL_TURNS_PER_SESSION = 2


class TurnRegistry:
    def __init__(
        self,
        retention_seconds: Optional[float] = None,
        max_terminal_turns: Optional[int] = None,
    ) -> None:
        ...
        self._retention_seconds = retention_seconds
        self._max_terminal_turns = max_terminal_turns
        self._limits_resolved = (
            retention_seconds is not None and max_terminal_turns is not None
        )

    def _resolve_limits(self) -> None:
        """Resolve retention limits on first use.

        Deferred because the registry is constructed at module import time,
        before ``fastworkflow.init()`` has loaded the env file.
        """
        if self._limits_resolved:
            return
        if self._retention_seconds is None:
            self._retention_seconds = TURN_RETENTION_SECONDS
        if self._max_terminal_turns is None:
            self._max_terminal_turns = (
                get_max_live_sessions() * TERMINAL_TURNS_PER_SESSION
            )
        self._limits_resolved = True
```

The retention clock starts when an execution goes terminal, and the sweep runs
right there — opportunistic eviction on completion keeps `_by_key` bounded
without a background task:

```python
    async def clear_active(self, channel_id: str, turn_key: str) -> None:
        async with self._lock:
            if self._active_by_channel.get(channel_id) == turn_key:
                self._active_by_channel.pop(channel_id, None)

            self._resolve_limits()
            execn = self._by_key.get(turn_key)
            if execn is not None and execn.ttl_expires_at is None:
                execn.ttl_expires_at = _now() + timedelta(
                    seconds=self._retention_seconds
                )
            self.evict_terminal()

    def evict_terminal(self, now: Optional[datetime] = None) -> int:
        """Evict expired terminal (DONE/LOST) executions; return the count.

        Non-terminal executions are never evicted: a live turn must stay
        reachable through its ``turn_key``.

        Safe to call with ``self._lock`` held: it performs no awaits.
        """
        self._resolve_limits()
        now = now or _now()

        expired = [
            key
            for key, execn in self._by_key.items()
            if execn.is_terminal
            and execn.ttl_expires_at is not None
            and execn.ttl_expires_at <= now
        ]
        for key in expired:
            self._by_key.pop(key, None)
        evicted = len(expired)

        terminal = [
            (execn.finished_at or execn.created_at, key)
            for key, execn in self._by_key.items()
            if execn.is_terminal
        ]
        if (overflow := len(terminal) - self._max_terminal_turns) > 0:
            terminal.sort(key=lambda item: item[0])
            for _, key in terminal[:overflow]:
                self._by_key.pop(key, None)
            evicted += overflow

        return evicted
```

Because the clock starts only at completion, a long-running turn is never at
risk mid-flight. Evicting a terminal execution costs a very late poller its
cached result; `/initialize` already degrades gracefully to a tokens-only
response when `TurnRegistry.get()` returns `None`.

## 2. `_creation_locks` is never pruned

**File:** `fastworkflow/run_fastapi_mcp/utils.py`

`ChannelSessionManager._creation_locks` is keyed by `channel_id` and grew for the
lifetime of the process; it still held 300 entries after 300 requests even once
the session cache was capped at 50.

```python
    def _prune_creation_lock(self, channel_id: str) -> None:
        """Drop a channel's creation lock once the channel is no longer live.

        A lock that is currently held is left alone: removing it would let a
        concurrent cold request build a second runtime for the same channel.
        """
        lock = self._creation_locks.get(channel_id)
        if lock is not None and not lock.locked():
            self._creation_locks.pop(channel_id, None)
```

## 3. Session cache bounds

**File:** `fastworkflow/run_fastapi_mcp/utils.py`

`max_live_sessions` defaulted to **2000**, far too high when each session pins a
`WorkflowExecutionContext` holding the workflow context (with whatever payload
commands stashed there) plus conversation history. There was also no idle expiry.

```python
DEFAULT_MAX_LIVE_SESSIONS = 50
DEFAULT_SESSION_IDLE_TTL_SECONDS = 900.0


def get_max_live_sessions() -> int:
    """Resolve the live-session cap. Single source of truth for MAX_LIVE_SESSIONS.

    The turn registry sizes its own retention off this too, so the deployment
    has one knob controlling overall in-memory footprint.
    """
    return int(
        fastworkflow.get_env_var("MAX_LIVE_SESSIONS", int, DEFAULT_MAX_LIVE_SESSIONS)
    )
```

`ChannelRuntime` gained `last_used_at`, maintained by `_touch()`, which drives an
idle sweep. `_sessions` is LRU-ordered, so the scan stops at the first session
still inside its TTL:

```python
    def _touch(self, channel_id: str) -> None:
        if channel_id in self._sessions:
            self._sessions[channel_id].last_used_at = time.time()
            self._sessions.move_to_end(channel_id)

    def _sweep_idle_locked(self) -> None:
        """Retire sessions idle for longer than the TTL. Caller holds the lock.

        Sweeping opportunistically on access avoids a background task; memory
        only grows on activity, so that is exactly when reclamation is needed.
        """
        self._resolve_limits()
        if self._session_idle_ttl_seconds <= 0:
            return
        cutoff = time.time() - self._session_idle_ttl_seconds
        for channel_id in list(self._sessions.keys()):
            runtime = self._sessions.get(channel_id)
            if runtime is None or runtime.last_used_at > cutoff:
                break
            # Never retire a channel mid-turn (§3.6): closing its ctx would race
            # the executor thread running the turn.
            if self.is_channel_busy and self.is_channel_busy(channel_id):
                continue
            self._retire_locked(channel_id)
```

The sweep is called from `create_session()` and from `get_session()` — in the
latter *after* `_touch`, so the channel being requested can never be its own
victim. All retirement paths share one helper, which `_evict_oldest_if_needed()`
and `remove_session()` now both use:

```python
    def _retire_locked(self, channel_id: str) -> None:
        """Drop a session from the live cache, persisting suspended state first.

        Caller must hold ``self._lock``.
        """
        runtime = self._sessions.pop(channel_id, None)
        if runtime is None:
            return
        self._save_workflow_context(channel_id, runtime)
        if runtime.execution_context.awaiting_user:
            self.session_state_store.save(
                channel_id,
                runtime.execution_context.serialize_state(channel_id=channel_id),
            )
        runtime.execution_context.close()
        self._prune_creation_lock(channel_id)
```

Both eviction paths keep the existing rule that a channel with an active turn is
never retired — closing its context would race the executor thread.

## 4. Two DSPy globals are unbounded for server workloads

**File:** `fastworkflow/run_fastapi_mcp/__main__.py`

Both are bounded by **entry count**, which is meaningless when prompts are large.
At a 450 KB prompt either cap is several GB, so in a long-running server both
behave as unbounded leaks that scale with request count:

- `dspy.clients.base_lm.GLOBAL_HISTORY` — records every LLM call (full prompt,
  messages, `ModelResponse`), capped at `MAX_HISTORY_SIZE = 10_000`. Exists for
  interactive `dspy.inspect_history()`.
- `dspy.settings.trace` — defaults to `[]`, i.e. **on**, and
  `Predict._forward_postprocess` appends `(self, {**kwargs}, pred)` for every
  prediction, where `kwargs` holds the entire prompt. Capped at
  `max_trace_size = 10_000`. Exists for optimizer runs.

```python
def configure_dspy_memory_limits() -> None:
    """Disable DSPy's two global, effectively-unbounded accumulators.

    Both are switched off unconditionally rather than behind a flag: neither is
    read on the request path, and this is the long-running server entrypoint --
    the optimizer workflows that need the trace run under ``fastworkflow train``,
    in a separate process. A developer who wants either back can call
    ``dspy.settings.configure()`` directly.
    """
    dspy.settings.configure(
        disable_history=True,
        # max_trace_size=0 is the documented off switch: Predict checks
        # `settings.max_trace_size > 0` before appending.
        max_trace_size=0,
        trace=[],
    )
    logger.info("DSPy history and predictor trace disabled (server memory bounds)")
```

Called from `initialize_fastworkflow_on_startup()`. These two are arguably DSPy
issues as well: server-oriented defaults, or size-based rather than count-based
caps, would help every DSPy server user.

## 5. Session eviction silently dropped the workflow context

**Files:** `fastworkflow/run_fastapi_mcp/utils.py`,
`fastworkflow/session_state_store.py`

Lowering the session cap from 2000 to 50 and adding an idle TTL made an existing,
latent behaviour routine, so it had to be resolved before the new defaults could
ship.

When a session is retired, conversation turns have already been written to the
`ConversationStore` incrementally and `awaiting_user` state is serialized to the
`SessionStateStore`; `_create_user_runtime()` rehydrates both. **The app
workflow's `context` dict was not among them.** `_WORKFLOW_REGISTRY` is a
`weakref.WeakValueDictionary`, and the only strong reference to the app workflow
is `ctx._app_workflow`; `WorkflowExecutionContext.close()` explicitly closes only
the `cme_workflow` ("caller owns that lifecycle"). So retiring a session drops
the app workflow, and the next `Workflow.create()` for that `channel_id` builds a
fresh one with an empty context.

Any command that stashes state in `workflow.context` and any later command that
reads it back are therefore silently decoupled by eviction. The reader does not
error — it simply sees nothing. A command's own `validate_extracted_parameters`
hook would normally catch a missing precondition like this, but that hook does
not run on the direct-action path, so nothing intercepts it. (That skipped
validation is a separate bug, tracked independently of this work.)

The context is now persisted on retirement and restored on rehydration:

```python
    @property
    def workflow_context_store(self) -> SessionStateStore:
        """Durable store for retired sessions' app-workflow context.

        Separate from ``session_state_store``: that one is cleared whenever a
        turn ends without ``awaiting_user``, which would wipe the workflow
        context on nearly every turn.
        """
        if self._workflow_context_store is None:
            self._workflow_context_store = get_session_state_store(
                base_folder=get_channel_workflow_context_dir(),
                key_prefix="fw:session:workflow_context:",
            )
        return self._workflow_context_store
```

The write is **conditional on the context having changed**. A context can be
large and retirement happens roughly once per request at steady state, so a
rehydrate/retire cycle that never touches it must not rewrite it:

```python
    def _save_workflow_context(
        self, channel_id: str, runtime: "ChannelRuntime"
    ) -> None:
        """Persist the app workflow's context so retirement is a pure cache miss.

        Best-effort: a workflow context may hold values that do not round-trip
        through JSON, and losing the cache is preferable to failing eviction.
        """
        workflow = runtime.execution_context.app_workflow
        context = getattr(workflow, "context", None) if workflow else None
        if not context:
            return
        payload = serialize_workflow_context(context)
        if payload is None:
            return
        digest = workflow_context_digest(payload)
        if digest == runtime.saved_context_digest:
            logger.debug(
                f"Workflow context unchanged for channel {channel_id}; skipping write"
            )
            return
        try:
            self.workflow_context_store.save_serialized(channel_id, payload)
            runtime.saved_context_digest = digest
        except Exception as exc:  # noqa: BLE001 - never let eviction fail
            logger.warning(
                f"Could not persist workflow context for channel {channel_id}: {exc}"
            )
```

`ChannelRuntime.saved_context_digest` is **seeded at session creation** — without
that, the comparison would never hit, because a runtime is retired exactly once:

```python
    # Seed the digest with what the workflow starts from, so retirement only
    # writes if a turn actually changed the context.
    if starting_context := getattr(app_workflow, "context", None):
        if payload := serialize_workflow_context(starting_context):
            runtime.saved_context_digest = workflow_context_digest(payload)
```

Restoration happens in `_create_user_runtime()`, under any caller-supplied
context so a fresh `http_bearer_token` still wins:

```python
    try:
        if saved_context := session_manager.workflow_context_store.load(channel_id):
            context = {**saved_context, **(context or {})}
            logger.info(f"Restored workflow context for channel_id {channel_id}")
    except Exception as exc:  # noqa: BLE001 - a bad cache must not block creation
        logger.warning(
            f"Could not restore workflow context for channel {channel_id}: {exc}"
        )
```

A naive digest check would have been a *pessimization*, because
`SessionStateStore.save()` re-encodes the dict — hashing plus saving would encode
twice on the common changed path. `SessionStateStore` therefore gained
`save_serialized()`, so the context is encoded once and handed through already
serialized:

```python
    def save_serialized(self, channel_id: str, payload: str) -> None:
        """Persist an already-JSON-encoded blob.

        Lets a caller that must encode anyway -- to hash the blob and decide
        whether the write is even needed -- hand the encoded form straight
        through instead of paying a second encode inside ``save()``. Backends
        that can write a string directly should override this.
        """
        self.save(channel_id, json.loads(payload))
```

with direct overrides on the disk backend (`f.write(payload)`) and the redis
backend (`self._client.set(self._key(channel_id), payload)`). The unchanged path
now costs one encode plus a hash and no write; the changed path does no more work
than before. `get_session_state_store()` also gained an optional `key_prefix` so
the two stores cannot collide on the same `channel_id` under redis (the disk
backend separates them by folder).

---

## New configuration

Deliberately kept to **two**:

| Env var | Default | Purpose |
|---|---:|---|
| `MAX_LIVE_SESSIONS` | 50 | Live session cache cap (was hardcoded 2000). Also sizes turn-registry retention, at 2 completed turns per session. |
| `SESSION_IDLE_TTL_SECONDS` | 900 | Idle expiry for cached sessions |

Everything else is a module constant (`TURN_RETENTION_SECONDS`,
`TERMINAL_TURNS_PER_SESSION`) or unconditional (the two DSPy settings). Four
further knobs were considered and rejected — turn TTL, turn cap, and the two DSPy
escape hatches — because a caller who needs to tune them is better served by
fixing the default than by carrying deployment config forever.

### Turn retention: what it bounds, and why neither knob is exposed

A `TurnExecution` is one unit of *async work*, not a conversational turn: one is
created per request that runs work, so the wait-or-defer path can hand back a
result after the HTTP request has already returned.

When the cap is exceeded, the **oldest completed** entries are dropped. The blast
radius is bounded by two properties: running executions are never evicted, and
the only thing lost is a finished turn's cached result. The sole consumer of a
retained terminal entry is the `/initialize` "already exists" branch; if the
entry is gone that path falls through to `_tokens_only_response()`, so the caller
gets HTTP 200 with tokens but no `startup_output` and must resubmit. Conversation
state is unaffected — it lives on the session, not the registry.

(For `/perform_action` and `/invoke_agent` the idempotency rejoin only applies
while an execution is still *active*: `_active_execution()` returns `None` once
terminal. Those endpoints gain nothing from terminal retention today.)

Which bound binds first depends on request rate. The count cap guarantees memory
is bounded at any rate; the TTL governs how long a caller can come back for a
result, and at the observed production rate (~65 requests/hour) the 300 s TTL
expires long before 100 further turns accumulate.

**Recommendation made:** of the two, `TURN_RETENTION_SECONDS` is the more
defensible knob, because "how long should a completed turn stay collectable?" is
a product question, whereas the memory cap only has to be big enough not to bind
during a burst.

**Decision (fastworkflow maintainer): rejected — keep both as constants.** 300 s
is ample against observed client behaviour (retries arrived ~3 s after a timeout),
and adding a knob invites deployments to diverge on a value that should be a
property of the framework. Revisit only if a real client needs a collection
window longer than 300 s.

## Results

Verified over 2000 sequential requests at a 450 KB payload, all successful:

| | before | after |
|---|---:|---:|
| slope | 1.758 MB/req | **0.032 MB/req (−98%)** |
| OOM horizon at 1 GB | ~204 requests | ~40,000 requests |

RSS holds flat from request ~100 onward with every retainer pinned at its cap.
The small residual (~0.024 MB/req, near measurement noise) is covered by ordinary
worker recycling, worth configuring as a backstop regardless.

Per-retainer contribution, measured by disabling each independently at a 450 KB
payload over 300 requests:

| arm | MB/request |
|---|---:|
| baseline (nothing bounded) | 1.763 |
| turn registry bounded | 0.885 |
| session cache bounded | 0.829 |
| DSPy history disabled | 0.880 |
| all of the above | 0.440 |
| + DSPy trace disabled (final) | **0.032** |

---

## Tests

Both tests are workflow-agnostic and need only a small fixture workflow, not any
particular application workflow.

### Fixture

Two trivial commands are enough to exercise everything:

- **`store_payload`** — takes a `payload: str` input, writes it to
  `workflow.context["payload"]`, and runs one `dspy.Predict` call with the
  payload as an input field. The LLM call matters: the DSPy retainers only
  accumulate if a prediction actually happens.
- **`read_payload`** — takes no meaningful input and reports whether
  `workflow.context.get("payload")` is present, plus whether the value reached
  the model.

The LLM must be stubbed at `dspy.LM.forward`, which is the narrowest useful seam:
`dspy.LM` overrides only `forward()`, so `BaseLM.__call__` →
`_process_lm_response` → `update_history` and `Predict._forward_postprocess` →
`settings.trace` still run for real. Stubbing any higher (at `Predict`, or at the
command) would bypass exactly the retainers under test and the memory test would
pass vacuously.

Drive the service in-process over ASGI against the real FastAPI app, importing
`run_fastapi_mcp.__main__` with `sys.argv` patched (it parses args at import) and
entering `app.router.lifespan_context(app)`.

### Test A — sequential memory growth

Issue N requests **strictly one at a time** (await each before the next), each
with a unique `channel_id` and a unique payload, via `POST /initialize` carrying
a `store_payload` startup action. Uniqueness matters twice: it defeats the
DSPy/litellm prompt cache so every request does real work, and it mirrors
production where each request carries distinct data.

Sample RSS after a forced `gc.collect()` every K requests, and fit a slope over
the second half of the samples so cache warm-up is excluded. Report alongside RSS
the size of each bounded structure — live sessions, `_creation_locks`,
`len(TurnRegistry._by_key)`, `len(GLOBAL_HISTORY)` — so a regression names its own
cause instead of just showing a number.

Growth alone only demonstrates correlation, so the test should support disabling
each retainer independently (constructor args on `TurnRegistry`, the session cap,
the two DSPy settings) and comparing slopes. An arm that flattens the slope is a
confirmed cause.

Two things make this test easy to get wrong:

- **Warm up first.** One request before measurement pays the one-time model
  loading and lazy-cache costs that would otherwise read as a leak.
- **Scale the payload.** Retention is ~4x payload size, so a small fixture
  payload produces a slope indistinguishable from noise. Parameterise the payload
  size and run the assertion at a production-representative value.

### Test B — session eviction and context rehydration

Initialize a channel with `store_payload`, force the session out of the live
cache, then call `read_payload` on the same channel and assert the context
survived. Four scenarios:

| scenario | how the session leaves the cache | asserts |
|---|---|---|
| `none` | not evicted | sanity: the read works at all |
| `lru` | create more channels than `MAX_LIVE_SESSIONS` | context rehydrated |
| `idle-ttl` | set a sub-second idle TTL, wait, then trigger the sweep via any session lookup | context rehydrated |
| `revisit` | evict, rehydrate via a read, then retire again | persisted blob's mtime is **unchanged** |

Assert two things after the read, not one:

1. **The invariant** — the rehydrated session's `workflow.context` still holds
   the stored value. Inspect `ctx.app_workflow.context`; `get_active_workflow()`
   is only populated while a turn is executing.
2. **The behaviour** — the reading command actually saw the value.

The second assertion needs care. Because `validate_extracted_parameters` does not
run on the direct-action path, a lost context does not raise: the command runs
with a missing value and returns HTTP 200. A stub that ignores its inputs will
score that silent degradation as a pass. Have the stub emit a distinguishable
sentinel when the expected value is **absent** from the prompt, and treat that
sentinel as a failure.

The `revisit` scenario covers the conditional write and is the only one that
needs filesystem inspection: compare `st_mtime_ns` of the persisted blob across
the second retirement. It is non-vacuous because the context assertion already
proves the blob existed and was read.

### Acceptance criteria

Gate on the **slope**, not on surviving a fixed-size run — a run of N requests
only proves N was below the current ceiling, and a fix that halves the slope
would pass while still OOMing later. At a production-representative payload,
MB/request should be within noise of zero over a few hundred requests, and every
bounded structure should hold flat at its cap.

## Note on diagnosis

An earlier hypothesis — that the residual was a C/Rust extension leaking a Python
reference (suspecting `pydantic-core`) — was **wrong**. A minimal repro showed
`json.loads`, pydantic validation and `fastworkflow.Action` construction all leak
exactly nothing; bisecting the execution path by layer found `dspy.settings.trace`
instead. The misleading signal was that a `gc` sweep did not surface the trace
list's payload references, which sent the investigation toward native code.
Bisecting by layer, rather than reasoning from the allocation site, is what
actually located it.

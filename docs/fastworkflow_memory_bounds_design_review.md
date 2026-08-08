# Adversarial review — bounded memory for `run_fastapi_mcp`

**Reviews:** `docs/fastworkflow_memory_bounds_design.md` (revision 1, as it stood at review time)
**Background:** `docs/fastworkflow_memory_fixes.md` (the production problem report and measurement
record; held privately, not in this repository)
**Verified against:** fastWorkflow `2.24.1`, commit `c206a813af1f48e09b63f7972cfcb16ee2262d2a`,
working tree as of this review; DSPy `3.2.1` in `.venv` (Python 3.12.2)
**Date:** 2026-08-05
**Tracking:** review of `fix-9uk` (closed)

> Historical record. Findings R1–R15 are resolved in revision 3 of the design; see its §21
> traceability table.

---

## 1. Method and standing

This is a **second-order** review. `fix-9uk` already absorbed one review round directly into the
design ("Integrated recommendations directly into the design; no separate adversarial-review
section"), so the question here is not "what did the author miss first time" but the Recipe-3
step-4 question: **what did the amendments themselves miss?**

Every finding below was verified against the working tree at the pinned commit, not against the
design's own citations. Where the design's evidence is correct I say so explicitly — a review that
only lists complaints is not calibrated. Four claims were re-derived by running code rather than
reading it.

The acceptance bar applied: **one mechanism must explain all observations, including the
negatives.** Two findings (R1, R2) fail that bar in the design's favour — they describe cases the
design's own stated invariants promise to cover but cannot.

**Severity scale:** S1 blocking (design cannot ship as written) · S2 major (resolve before
implementation) · S3 moderate (resolve before release) · S4 minor / bookkeeping.

### Findings index

| # | Severity | Finding | Design section |
|---|---|---|---|
| R1 | **S1** | Pin-on-object-context makes the session-cache bound inoperative for the framework's canonical workflow shape | §8.1, §7.1, §13.2 |
| R2 | **S1** | Invariant 6 is unachievable as wired: `/invoke_agent_stream` is invisible to `is_channel_busy` | §4.6, §7.1, §8.7 |
| R3 | S2 | Startup re-execution on top of restored state is a new corruption path | §6.3, §8.5 |
| R4 | S2 | `MAX_LIVE_SESSIONS` cannot be set in the deployment shape that motivated the change, so the stated rollback does not work | §5.1, §14 |
| R5 | S2 | Trades unbounded RSS for unbounded disk, with no acceptance gate on the growth it introduces | §8.9, §13.5 |
| R6 | S2 | Restore precedence is right for application state and wrong for operator configuration | §8.5, decision 12 |
| R7 | S2 | §9's policy is correct (confirmed), but fails **silently** on the supported DSPy floor, and §13.4 omits the production shape | §9.2, §9.3, §13.4 |
| R8 | S3 | The latency gate measures the one path the production workload never takes | §13.6, §11 |
| R9 | S3 | The new post-turn trimming trigger has unspecified lock ordering | §7.1, §6.2 |
| R10 | S3 | Invariant 7 overstates what the code delivers: half the "atomic" write is lossy-but-successful | §4.7, §8.4, §8.7 |
| R11 | S3 | Redis namespace separation is a data-loss guard reduced to one clause | §8.2, §12 |
| R12 | S3 | The fixed cap of 100 repeats the count-vs-byte criticism the source study makes of DSPy | §5.2, §11 |
| R13 | S4 | §6.2's step list double-specifies active-pointer removal (nested-lock trap) | §6.2 |
| R14 | S4 | Shutdown drain uses `lock.locked()`, so §7.3's snapshot can race a QUEUED turn | §7.3 |
| R15 | S4 | Evidence bookkeeping: interpreter pin, and a dead lifecycle API treated as contract surface | §2.1, §7.3 |

---

## 2. What the design gets right (independently re-verified)

Stated first because it is load-bearing for the recommendation: the diagnosis is sound and the
design's code citations are accurate. I re-verified each and found no errors.

| Design claim | Verdict | Evidence |
|---|---|---|
| Terminal turns are retained forever | **Confirmed** | Every execution enters `_by_key` (`turns.py:202-214`); `clear_active` touches only `_active_by_channel` (`:216-224`); `ttl_expires_at` is never assigned, so `evict_terminal` is a no-op (`:226-243`) |
| Creation locks grow with unique channel IDs | **Confirmed** | `_creation_locks` is a plain dict (`utils.py:689`) with an insert-only accessor (`:695-701`) and no removal path anywhere |
| The session cache can lose app state | **Confirmed** | `max_live_sessions: int = 2000` (`utils.py:677`); `_evict_oldest_if_needed` persists only `awaiting_user` state (`:715-743`); `close()` closes only the CME workflow (`workflow_execution_context.py:476-490`) |
| Only `/initialize` consumes a retained terminal record | **Confirmed** | `turn_registry.get(...)` has exactly one call site in the whole server (`__main__.py:697`) |
| `_update_http_bearer_token` is a no-op today | **Confirmed** | It reads a ContextVar (`workflow_execution_context.py:496-497`) pushed only inside execution (`:551`, `:630`, i.e. in the executor thread), but is called from the event loop in `ensure_user_runtime_exists` (`utils.py:290`, `:304`). Tokens on warm sessions are already stale; §8.5's fix is a real bug fix |
| `get_env_var` precedence makes the workflow env file the reliable override | **Confirmed** | `default` is returned *before* `os.getenv` is consulted (`__init__.py:215-219`) — see R4 for the consequence the design draws too weakly |
| `max_trace_size=0` is the trace off switch; `disable_history` is separate from `max_history_size` | **Confirmed** | `predict.py:228` gates on `settings.max_trace_size > 0`; `base_lm.py:98` and `:223` gate on `settings.disable_history` |
| Configuring DSPy from an async task is unsafe | **Confirmed** | `Settings._ensure_configure_allowed` pins both an owner thread id and an owner async task, raising on mismatch |
| §13.5's "do not sum ablation deltas as independent shares" | **Correct and important** | The source study's own arms sum to 2.70 MB/req against a 1.76 MB/req baseline; the design is right to refuse the addition |

Two implementation details the design asserts were **confirmed by execution**, not just reading:

1. **§7.2's weak creation locks work.** `asyncio.Lock` is weak-referenceable; a
   `WeakValueDictionary` entry survives while any holder or waiter has a strong reference
   (`async with` keeps one for the block) and drops to zero entries once all are released. The
   design's argument for *not* manually pruning an apparently-unlocked lock is sound.

2. **§9's DSPy policy survives the production shape** — and this is a stronger result than the
   design claims for itself. fastWorkflow never calls `dspy.configure()` on the request path (only
   `dspy.context(...)` and `configure_cache(...)`; verified by grep across `fastworkflow/`), and
   `Settings.context` builds its overlay as `{**main_thread_config, **original_overrides,
   **kwargs}` — so a policy installed by `configure()` on the main thread **is inherited** into
   `dspy.context` blocks running in executor threads. Measured, `Predict` inside `dspy.context`
   inside a `ThreadPoolExecutor`, stubbing only `LM.forward`:

   | Arm | `GLOBAL_HISTORY` | `lm.history` | `settings.trace` |
   |---|---:|---:|---:|
   | Baseline, 4 calls | 4 | 4 | 4 |
   | §9.1 policy, 4 calls | **0** | **0** | **0** |

   Worker threads observed `disable_history=True, max_trace_size=0`. §9's mechanism is correct.
   R7 concerns only its *failure mode* and its test plan, not its correctness.

---

## 3. Findings

### R1 — [S1] Pin-on-object-context makes the session-cache bound inoperative for the framework's canonical workflow shape

**The design says.** §8.1: a session is evictable "only when: no root/current/response-generation
command-context object is live; no live child workflow state exists; the app workflow's `context`
passes strict JSON projection." §7.1: if no safe candidate exists, "remain over target and emit one
rate-limited warning… Never discard state merely to satisfy the number."

**What the tree says.** `Workflow.__init__` starts with `_root_command_context = None`
(`workflow.py:160`), so eligibility depends entirely on whether the application assigns one. It
does, in the startup command that `/initialize` runs as its first turn:

```12:12:fastworkflow/examples/simple_workflow_template/_commands/startup.py
        workflow.root_command_context = workflow_schema.create_workitem("Epic")
```

```42:42:fastworkflow/examples/messaging_app_2/_commands/startup.py
        workflow.root_command_context = User(command_parameters.name)
```

Also `messaging_app_4` and `tests/todo_list_workflow`. `simple_workflow_template` is the scaffold
new workflows are copied from.

The negative case matters for calibration: `retail_workflow`, `hello_world`, `messaging_app_1` and
`messaging_app_3` never assign `root_command_context`, so they **are** evictable and the bound works
for them. This is precisely why the defect is dangerous rather than obvious — the workflows most
likely to be used for measurement are the ones where the bound holds.

**Why it matters.** For any workflow that uses object command contexts — the framework's headline
feature, "AI-enable your existing Python classes" — **every channel becomes permanently
un-evictable the moment startup completes.** `_sessions` then grows without bound, the LRU never
fires, and per-request memory growth returns. The source study attributes ~135 MB / 300 requests
(≈0.45 MB/req) to the session cache: **9× the design's own ≤0.05 MB/req shipping gate.** Fix 2
(2000 → 50) becomes cosmetic for these workflows, and the correctness fix it was gated on (fix 4)
never runs.

The design cannot detect this, and in one place asserts it as correct:

- §13.3 runs the state matrix on "a real small workflow and direct actions" — no object-context arm
  in the *soak*, only a pin assertion in the matrix.
- §13.5's shipping target says "all structurally bounded collections stay at/below their caps" —
  but §7.1 explicitly licenses `_sessions` to sit over cap, so the gate is satisfiable while the
  bound is inoperative.
- §13.2's test **"Unsupported object-context sessions stay live"** encodes the vacuous outcome as
  the expected result.

**Ask.** The executive decision in §1 must state plainly that the memory bound is **conditional on
workflow shape**, and one of:

- (a) add an object-context arm to the §13.5 soak with a pre-registered expected slope, and accept
  in §16 that object-context workflows are unbounded pending an explicit workflow-owned serializer
  (§16.3's trigger has therefore already fired — it is not hypothetical demand); or
- (b) bound the pinned set itself: cap pinned sessions and shed load (503) rather than allocate
  without limit. This preserves "never lose state" while refusing to promise a bound the code
  cannot keep.

Option (a) is honest; (b) is honest and bounded. Shipping §8.1 as written with §13's current gates
would ship a memory fix that does not apply to `simple_workflow_template`.

---

### R2 — [S1] Invariant 6 is unachievable as wired: `/invoke_agent_stream` is invisible to `is_channel_busy`

**The design says.** Invariant 6: "A channel with an active execution is never removed from the
live-session cache." §7.1 step 1: "skip a channel with an active execution." §8.7: verify "the
channel is not busy" before snapshotting.

**What the tree says.** Busy-ness is defined solely by the turn registry:

```194:194:fastworkflow/run_fastapi_mcp/__main__.py
session_manager.is_channel_busy = turn_registry.has_active
```

But `/invoke_agent_stream` (`__main__.py:920-1085`) never creates a `TurnExecution`. It guards with
the lock and holds it directly:

```963:974:fastworkflow/run_fastapi_mcp/__main__.py
            if runtime.lock.locked():
                await emit_error(
                    f"A turn is already in progress for user: {channel_id}"
                )
                return
            # ...
            async with runtime.lock:
```

So throughout a streaming turn `has_active(channel_id)` is **False**, `_evict_oldest_if_needed`
(`utils.py:715-743`) treats the channel as a valid victim, pops it, and calls
`execution_context.close()` — closing the CME workflow's store while the turn is mutating state in
an executor thread. The design never mentions streaming (zero matches for "stream" in the doc).

**Why it matters.** Today this needs 2000 concurrent live channels — unreachable at the reported
65 requests/hour. The design makes it reachable three ways at once: the target drops 40×, §7.1 adds
a *new* trimming trigger on every turn completion, and §8.7 now **serializes the concurrently
mutating context first** — so the failure is upgraded from "lose the cache" to "persist a torn
snapshot as authoritative, then close the context under a running turn." That torn snapshot then
wins on the next cold creation (§8.5), making the corruption durable.

**The trap to avoid.** Invariant 2 says "never replace it with `runtime.lock.locked()`," which
reads as forbidding the fix. Invariant 2 is about the **409 idempotency truth source**, where using
the lock re-opens the v2.22.0 double-execution race. **Eviction safety is a different question**
and correctly uses the union. Scope invariant 2 to the 409 guard and define eviction busy-ness as
`has_active(cid) or (runtime is not None and runtime.lock.locked())`.

**Ask.** Add the union predicate; add a §13.2 test that a channel with a held `runtime.lock` and no
registry execution is never evicted; state in §4 that invariant 6 depends on the union, not on the
registry alone. Note this also closes the pre-existing `/invoke_agent_stream` inconsistency flagged
in the fix-85g scope discussion.

---

### R3 — [S2] Startup re-execution on top of restored state is a new corruption path

**The design says.** §6.3 reuses a retained startup record when one matches; §6.5: "If the terminal
record expires, a cold session may submit startup again, **matching current cold-session
behavior**." §8.5: a cold runtime is seeded from the saved context.

**Why the "matching current behavior" claim does not hold.** Today a cold session always starts
from launch-time context, so a re-run startup action operates on a clean slate — non-idempotent
startup actions are harmless *by construction*. After this change startup re-runs **on top of
restored state**. A `store_payload`-style action that appends to a list or increments a counter now
doubles. The two mechanisms are individually safe: bounded retention alone re-runs startup on a
clean slate; state restore alone never re-runs startup. Only the combination corrupts — which is
exactly the interaction an atomic five-fix release must own.

Worse, the trigger is **wall-clock**: reuse depends on the 300-second window and the 100-record cap
(§5.2), so an identical request sequence against identical state either re-runs startup or does
not, depending on timing. That is a Heisenbug, and §13.1's "Retained startup is reused after its
`ChannelRuntime` is evicted" only tests the *reuse* branch, never the expired branch against
restored state.

**Ask.** Make startup reuse a function of persisted state, not of in-process retention: record the
startup idempotency key (and completion) **inside the §8.3 envelope**. Then a cold session with a
saved record knows startup already ran regardless of registry retention. This is strictly simpler —
it removes §6.3's bounded registry lookup entirely, removes the timing dependence, and makes
§16.4 ("terminal retention is in-process; restart loses results") a reporting limitation rather
than a correctness one. Add a test: expire the retained record, cold-create, assert startup does
**not** re-run and the context is not doubled.

---

### R4 — [S2] `MAX_LIVE_SESSIONS` cannot be set in the deployment shape that motivated the change

**The design says.** §5.1 correctly notes "the reliable override location is the workflow's
`fastworkflow.env`, not merely a shell `export`." §14 then says: "Raising `MAX_LIVE_SESSIONS` is
the operational rollback for excessive cold churn."

**What the tree says.** The precedence is env-file → **default** → `os.environ`:

```215:219:fastworkflow/__init__.py
    value = _env_vars.get(var_name)
    if value is None:
        if default is not None:
            return default
        value = os.getenv(var_name)
```

Because a default is always supplied, `os.environ` is **never consulted**.

**Why it matters.** The motivating deployment is a container that was OOM-killed at a 1 GB limit.
In that shape the natural knob is a Kubernetes/Docker environment variable — which this code
silently ignores. So the single stated rollback lever requires editing a file inside the mounted
workflow directory and restarting the process. §5.1 states the mechanism but §14 draws the wrong
operational conclusion from it, and the two sections are 9 pages apart.

**Ask.** Either resolve the limit with `default=None` and apply `DEFAULT_MAX_LIVE_SESSIONS`
explicitly (so `os.environ` works and container operators get the knob they expect), or rewrite §14
to say rollback requires a workflow-env-file edit plus restart. Separately: §5.1 requires
documenting the variable in `fastworkflow/examples/fastworkflow.env`; it is **not there today**
(only `SPEEDDICT_FOLDERNAME` at line 36), so this is a real open item, not a formality.

---

### R5 — [S2] Trades unbounded RSS for unbounded disk, with no acceptance gate on the growth it introduces

**The design says.** §8.9 acknowledges "Unique channels with changed 450 KB contexts can create
substantial disk/Redis growth" and calls it "an accepted limitation of the first release." §13.5
asks to record "durable-store counts/bytes."

**Why that is not sufficient.** The motivating workload is **unique-channel-per-request** (source
study Test A: "each with a unique `channel_id` and a unique payload"; production ≈65 req/hour at
450 KB). In that workload every request cold-creates, evicts, and writes a ~450 KB snapshot that
**will never be read again** — the channel is never revisited. That is ≈1560 requests/day ×
450 KB ≈ **700 MB/day of durable growth, unbounded**, with no reaper (fix-6b4 open) and §8.9
forbidding a TTL on principle. For the exact workload that motivated the change, the persistence
machinery is pure cost at zero benefit.

Two aggravating details:

- §13.5's shipping target has six bullets and **none of them bound durable growth**, so the design
  can pass its own gate while replacing an OOM in ~200 requests with disk exhaustion in N days.
- The disk key is only lightly sanitised — `_json_path` replaces `os.sep` and `/` only
  (`session_state_store.py:50-52`). §16.8 calls collisions "pre-existing and separate," but this
  change makes a collision **durable and authoritative** rather than transient, so its blast radius
  is materially different.

**Ask.** Add a durable-growth acceptance number to §13.5 (records and bytes per 1000 requests at
the representative payload), and gate broad rollout on fix-6b4 landing rather than on "monitor."
Consider recording whether a channel has ever been revisited so single-shot channels can be
excluded from snapshotting — a bound the design currently has no mechanism to express.

---

### R6 — [S2] Restore precedence is right for application state and wrong for operator configuration

**The design says.** §8.5: "Do not merge launch-time application context over saved state. That
would overwrite mutations and resurrect keys the application deleted." Decision 12 calls
launch-time state "stale."

**What the tree says.** The workflow context is **not** client-supplied. `InitializationRequest`
has no `context` field (`utils.py:28-37`); the context is the process-wide CLI argument:

```724:724:fastworkflow/run_fastapi_mcp/__main__.py
            context=json.loads(ARGS.context) if ARGS.context else None,
```

The only per-request addition is `http_bearer_token` (`utils.py:239-247`).

**Why it matters.** `--context` is **operator deployment configuration** — API base URLs, tenant
identifiers, feature flags — not stale application state. For those keys the precedence inverts:
launch-time is the *fresh* value and the snapshot is the stale copy. So an operator who changes
`--context` and redeploys finds the change **silently ignored for every channel that has a
snapshot**, with no TTL (§8.9), no deletion API (fix-6b4), and no warning. The only recovery is
manually deleting files under `SPEEDDICT_FOLDERNAME`. §13.3's test "saved application state wins
over stale launch-time state" locks this in as intended behaviour.

The design's reasoning is correct for application-mutated keys and it has no way to tell the two
kinds apart, because both arrive in the same flat dict.

**Ask.** Record the launch-context digest in the §8.3 envelope. On cold creation, if the current
launch context differs from the one recorded at snapshot time, re-overlay the **changed launch
keys** and log at WARNING; application-only keys keep the saved value. Failing that, document a
supported reset procedure in §14 and emit a warning on digest mismatch — silence is the part that
must not ship.

---

### R7 — [S2] §9's policy is correct, but fails silently on the supported DSPy floor, and §13.4 omits the production shape

§9's mechanism is confirmed correct (see §2 above, including the executor-thread result the design
does not claim). Two gaps remain.

**(a) The failure mode is silent, and §9.2's guard does not cover it.** `Settings.configure` does no
key validation — it is `main_thread_config[k] = v`. I set `totally_bogus_key_xyz=True` and it was
accepted without error. `pyproject.toml:50` allows `dspy = "^3.0.1"`, so on any 3.0.x that does not
read `disable_history` / `max_trace_size`, the policy is a **silent no-op** and the server runs with
the leak while logging "dspy_history=off, dspy_trace=off" per §14. §9.2's guard fails startup when a
*different thread or task* owns incompatible settings — it cannot detect a version that ignores the
keys. §9.3's "verify against the minimum allowed DSPy 3.x before release" is a one-time human step
guarding a per-deployment runtime property.

**(b) §13.4 does not test the shape production uses.** Its steps make "a real `Predict` call with
default settings" and then "unique calls" — never inside `dspy.context(...)`, never in an executor
thread. Since *every* fastWorkflow LLM call is `dspy.context` inside `run_in_executor`, a test that
omits both would pass even if `context` did not inherit `main_thread_config`. It happens to inherit
(I verified), but the test as specified does not establish that.

**Ask.** Replace "configure, then log the policy" with a **positive structural assertion** at
startup: after installing the policy, make one throwaway `Predict` call through a stub boundary and
assert all three structures stay empty; fail readiness (invariant 14) if not. Add the
`dspy.context`-inside-executor-thread arm to §13.4. Minor test-plan note: the §13.3/§13.4 stub must
return a real `litellm` `ModelResponse` — a plain dict raises in
`BaseLM._process_completion` (`response.choices`), which will cost the implementer a cycle.

---

### R8 — [S3] The latency gate measures the one path the production workload never takes

**The design says.** §13.6: "The no-eviction p50/p95 must regress by less than 5% or measurement
noise, whichever is larger," with eviction latency "reported separately."

**Why that is the wrong primary gate.** With unique channels per request, **every** request
cold-creates and evicts, so the eviction path *is* the steady state, not an exception. And the work
lands inside the manager critical section: `_evict_oldest_if_needed` is awaited from
`create_session` while holding `self._lock` (`utils.py:762`, `:775`), and `get_session` needs the
same lock (`:746`). A ~450 KB encode + SHA-256 + store write therefore blocks the event loop and
every other channel's session lookups. §11 anticipates this and defers a `RETIRING` reservation to
"if measured eviction latency or event-loop stalls are material" — but no §13.6 threshold can ever
make it material, because the gated measurement excludes it by construction.

**Ask.** Promote the unique-channel / evict-every-request configuration to a **gated** p50/p95
number, and add an event-loop-stall measure (e.g. scheduling delay percentiles under load), not
just isolated eviction latency. Then §11's deferral has a trigger that can actually fire.

---

### R9 — [S3] The new post-turn trimming trigger has unspecified lock ordering

§7.1 adds trimming "after an execution clears its active pointer," without saying whether that runs
inside or outside the registry lock. It matters: `clear_active` does its work under `registry._lock`
(`turns.py:222-224`) and is called from `_run_turn`'s `finally` (`turns.py:317`). If trimming is
invoked there, a synchronous ~450 KB store write executes while `registry._lock` is held — blocking
`start_or_get_active` and therefore **every turn submission on every channel**, which is strictly
worse than R8's manager-lock case.

§6.2 is admirably explicit for the turn sweep ("The sweep contains no `await`"). §7.1 needs the
mirror-image statement.

**Ask.** State that post-turn trimming is scheduled **after** the registry lock is released, and
that no store I/O ever occurs under `registry._lock`. Add it to §4 as an invariant, since it is the
kind of detail that reads as a free optimisation to a later maintainer.

---

### R10 — [S3] Invariant 7 overstates what the code delivers

Invariant 7 permits removal "only after every state item required for supported cold rehydration
has been persisted **successfully**," and §8.4 forbids `default=str` precisely because it makes "a
lossy write appear successful." But §8.7 step 4 persists suspended state through the existing
store, and that path is:

```61:64:fastworkflow/session_state_store.py
    def save(self, channel_id: str, state: dict[str, Any]) -> None:
        path = self._json_path(channel_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, default=str)
```

So the exact failure mode §8.4 bans survives *inside* the transaction §8.7 calls atomic — and
lowering the cap 40× makes that path routine rather than rare. Note `serialize_state` carries the
suspended ReAct trajectory and conversation turns, so a non-JSON artifact degrades to its `repr()`
and the write reports success.

**Ask.** Either apply the strict projection to the suspended-state write too (preferred — it is the
same helper), or scope invariant 7 and §8.4 explicitly to the context namespace and add the
asymmetry to §16 as a named accepted limitation. As written, the invariant promises more than the
implementation map delivers.

---

### R11 — [S3] Redis namespace separation is a data-loss guard reduced to one clause

§8.2 names the target prefix (`fw:session:workflow_context:`) and §12 reduces the work to
"namespaced factory support." The factory cannot express it today:

```112:135:fastworkflow/session_state_store.py
def get_session_state_store(
    *,
    base_folder: Optional[str] = None,
) -> SessionStateStore:
    # ...
    if backend == "redis":
        # ...
        return RedisSessionStateStore(url)
```

There is no `key_prefix` parameter, and for `SESSION_STATE_STORE=redis` **`base_folder` is ignored
entirely** — both stores would receive the default prefix `fw:session:pending:`
(`session_state_store.py:83`). Disk separates by folder, so the collision is Redis-only, which
makes it the kind of defect that passes local testing and fails in the multi-pod deployment.

If missed, the failure is silent and destructive in both directions: `_persist_after_turn` calls
`session_state_store.clear(channel_id)` after every non-awaiting turn (`turns.py:266`), deleting the
workflow-context snapshot on essentially every turn; and `apply_serialized_state` would be handed a
context envelope, where a schema mismatch only warns and continues (fix-4od). The source study is
more explicit about this than the design of record is.

**Ask.** Promote to a §4 invariant and add an explicit test asserting the two stores' Redis keys
differ for the same `channel_id`. One clause in an implementation map is too little weight for a
data-loss guard.

---

### R12 — [S3] The fixed cap of 100 repeats the count-vs-byte criticism the source study makes of DSPy

The source study rejects DSPy's caps because "Both are bounded by **entry count**, which is
meaningless when prompts are large. At a 450 KB prompt either cap is several GB." §5.2 then adopts a
fixed count cap of 100 retained startups, and §11 states the bound is "count-based, not byte-based."

At the motivating payload the arithmetic is unflattering: the turn registry retained ~260 MB /
300 requests ≈ 0.87 MB per record, so 100 records ≈ **87 MB** — roughly 4× the 50-session live
budget — and it scales linearly with payload, exactly as criticised. Meanwhile the cap's only
consumer is the single `/initialize` lookup (`__main__.py:697`), which needs coverage for 300
seconds at ≈65 req/hour ≈ **6 records**. The cap is over-provisioned by more than an order of
magnitude against its own justification.

§2.3 does disclose "It does not provide a byte budget," so this is not concealed — but the constant
is unjustified and the contradiction with the source study's reasoning is unreconciled.

**Ask.** Derive the cap from its consumer (10–20 covers the observed rate with margin, at ≈9–17 MB),
or state the byte cost at the representative payload in §11 **and** in the §14 startup log line, so
the number is auditable by an operator rather than folklore.

---

### R13 — [S4] §6.2's step list double-specifies active-pointer removal

Step 3 is "call `clear_active(channel_id, turn_key)`"; step 4 is "under the registry lock, remove
the matching active pointer." Step 4 **is** what `clear_active` does (`turns.py:216-224`). Read
literally, an implementer either performs the removal twice or re-acquires `registry._lock` inside
it — and `asyncio.Lock` is not reentrant, so the second reading deadlocks the turn's `finally`
block, which is about the worst place to put a deadlock.

**Ask.** Collapse steps 3 and 4, and state that steps 5–8 execute inside `clear_active`'s existing
lock block (which is also where §6.2's "no `await`" guarantee comes from).

---

### R14 — [S4] Shutdown drain uses `lock.locked()`, so §7.3's snapshot can race a QUEUED turn

§7.3 adds "persist supported workflow state before releasing live runtimes" to shutdown. The drain
that is supposed to make that safe uses the lock, not the registry:

```285:286:fastworkflow/run_fastapi_mcp/__main__.py
            if rt and rt.lock.locked():
                active.append(channel_id)
```

A QUEUED execution whose task has not yet acquired `runtime.lock` reports not-busy, so
`wait_for_active_turns_to_complete` returns "All turns completed" and shutdown proceeds to finalize
conversations and close contexts (`__main__.py:341-342`). Today that loses a turn; with §7.3 it also
writes a snapshot taken *before* the queued turn mutates the context, then closes the context under
it — making the stale snapshot authoritative on next creation.

**Ask.** Use the union predicate from R2 in the drain. Same root cause, same one-line fix.

---

### R15 — [S4] Evidence bookkeeping

- **Interpreter pin.** The header's "installed DSPy 3.2.1" holds for `.venv` only; the ambient
  interpreter on this machine (`/home/drawal/miniconda3/bin/python`) has DSPy **2.6.27**. Since §9
  is version-sensitive and §13.4 mandates isolated subprocesses, the plan should pin
  `.venv/bin/python` explicitly rather than inherit `python`, or a §13.4 run can silently exercise
  a 2.x DSPy and prove nothing about the shipped policy.
- **Dead lifecycle API.** §7.3 preserves "Remove/terminate… existing explicit lifecycle semantics
  until a public channel deletion contract is designed." `remove_session` and `evict_live_session`
  have **no production callers** — the only caller in the tree is
  `tests/test_fastapi_topology_b.py:114`. Worth saying so, so a later reader does not treat them as
  live contract surface constraining the design.
- **Baseline is green.** `tests/test_fastapi_topology_b.py`,
  `tests/test_fastapi_turns_async.py`, `tests/test_session_state_serialization.py`: 8 passed in
  21.7 s on `.venv`. §13.7's "full suite twice with zero failures" starts from a clean base for the
  files this change touches.

---

## 4. Cross-cutting: the verification plan cannot detect the design's two largest failure modes

§13 is unusually strong on structure — deterministic registry and manager tests, a state round-trip
matrix, an isolated-subprocess DSPy test, a slope-gated RSS soak with an ablation warning. The gap
is not rigor; it is **coverage of the workflow shapes and endpoints where the design's guarantees
lapse**:

| Failure mode | Would §13 catch it? | Why not |
|---|---|---|
| R1 object-context pin ⇒ unbounded `_sessions` | **No** | §13.3/§13.5 fixtures are function-style; §13.2 asserts the pin as correct; §13.5's "collections at/below caps" is satisfiable because §7.1 licenses being over cap |
| R2 streaming turn evicted mid-mutation | **No** | No §13 test exercises `/invoke_agent_stream` at all |
| R3 startup re-run on restored state | **No** | §13.1 tests the reuse branch only, never the expired branch against a restored context |
| R5 durable growth | **Partially** | §13.5 records bytes but no shipping target bounds them |
| R7 DSPy policy silently ignored on the version floor | **No** | §13.4 pins one version and omits the `dspy.context`/executor shape |
| R8 eviction-path latency | **No** | §13.6 gates the no-eviction path, which the production workload never takes |

Three additions close most of it: an **object-context arm** and a **streaming arm** in the soak, and
promoting the **evict-every-request** configuration to a gated latency measurement.

---

## 5. Recommendation

**Do not implement §8.1 and §7.1 as written, and do not treat §13 as sufficient evidence.** The
diagnosis is sound, the five mechanisms are individually well-chosen, and §9 is verified correct
including a case the design does not claim. But two stated invariants do not hold against the
current tree:

- **R1** — the session-cache bound, which the source study measures as roughly half the growth, is
  inoperative for object-context workflows, including the template new workflows are copied from.
- **R2** — invariant 6 cannot hold while `is_channel_busy` is the registry pointer alone, and the
  design turns a previously unreachable race into a routine one that now writes a torn snapshot and
  makes it authoritative.

Both are cheap to resolve. R2 is a one-line predicate change plus a scoping sentence on invariant 2.
R1 requires a decision, not code: either state the bound as conditional on workflow shape and let
§16.3's trigger fire, or bound the pinned set and shed load. R3 arguably *simplifies* the design by
moving startup-completion into the state envelope and deleting §6.3's registry lookup.

Suggested gating, consistent with §12.1's landing order:

1. Resolve R1 as an explicit decision-log entry before any code lands (it changes what §1 promises).
2. Fold R2 and R14 into step 2 of the landing order (transactional retirement) — they are the same
   predicate.
3. Fold R3, R6, R10, R11 into step 1 (state envelope and store primitives) — all four are envelope
   or store-contract changes and are cheapest there.
4. Add R7(a)'s structural assertion to step 5, and the three §13 arms from section 4 to step 6.
5. Treat R4's env-file documentation and R12's cap derivation as release blockers, not polish: both
   are numbers an operator will rely on.

The design's own §17 warns against copying shortcuts that "flatten RSS while introducing races,
stale-state resurrection, credential persistence, or silent state loss." R1, R2, R3 and R6 are that
same warning turned on the design of record: an RSS graph that flattens on a function-style fixture
while object-context workflows stay unbounded, streaming turns get torn snapshots, and a stale
snapshot silently outranks a redeployed configuration.

# Bounded memory for `run_fastapi_mcp`

**Status:** All three releases implemented and verified — A in v2.25.0, B in v2.26.0, C in v2.27.0 (`fix-g03`)  
**Revision:** 3 — absorbs the round-2 adversarial review (findings R2-1–R2-22)  
**Scope:** `fastworkflow/run_fastapi_mcp/` and its session-state storage interface  
**Source study:** `docs/fastworkflow_memory_fixes.md` (the production problem report and measurement
record; held privately, not in this repository)  
**Adversarial reviews:** `docs/fastworkflow_memory_bounds_design_review.md` (round 1, findings R1–R15);
`docs/fastworkflow_memory_bounds_design_review_round_2.md` (round 2, findings R2-1–R2-22)  
**Verified against:** fastWorkflow `2.24.1`, commit
`c206a813af1f48e09b63f7972cfcb16ee2262d2a`; DSPy `3.2.1` **in `.venv`** (Python 3.12.2)  
**Date:** 2026-08-05  
**Tracking:** `fix-9uk` (design), `fix-p3l` (round-2 review), `fix-uy3` (this revision),
`fix-g03` (implementation)

> All three releases have shipped: A in v2.25.0, B (the checkpoint protocol) in v2.26.0, C (the
> lowered live-session cap) in v2.27.0. Measured results are in §16.5.1, §16.5.2 and §16.6.1.
>
> Two sections are **superseded by the serialization-hooks design** and are retained for provenance
> rather than as instructions: §11.1's "nothing is durable by default" and §11.2's declaration-based
> eligibility. Decision 25 was reversed on 2026-08-05 — JSON-native workflow context persists by
> default, and a workflow becomes evictable by implementing `get_state`/`from_state` on its context
> class. See `docs/fastworkflow_serialization_hooks_design.md`, which governs projection semantics.

This is the design of record for the memory-bounds change. The source study remains the problem
report and production measurement record. Where its illustrative patch snippets differ from this
document, this document governs.

**Provenance convention.** Revision 1 absorbed a first review round inline. Revision 2 absorbed the
retained round-1 review (findings `R1`–`R15`). Revision 3 absorbs the round-2 review, whose findings
are numbered `R2-1`–`R2-22` precisely so they cannot be confused with round 1's `R1`–`R15`.

- `[R#]` tags mark constraints that came from round 1 and **still stand**.
- `[R2-#]` tags mark amendments forced by round 2. Every round-2 finding lives in review §4; where
  this document draws on non-finding review material it names the section directly
  (review §3 is the round-1 resolution audit, review §5 the cross-cutting conclusion and
  decomposition, review §6 the captured runtime evidence).
- `[R2-# · absorption]` tags mark places where implementing the review's required resolution
  surfaced a *further* constraint that the review did not state. These are recorded in §2.2 with the
  evidence, because they change what the resolution can be.

**Sections were renumbered in revision 3** to follow the release decomposition adopted in §1.1. The
traceability table in §21 maps each round-2 finding to its section here, so the review's citations of
revision 2 section numbers remain followable.

Revision 3 is not a patch on revision 2. The review's verdict was **"do not implement revision 2 as
written,"** with 14 blocking findings, and its central diagnosis — that revision 2 treated a new
system-of-record protocol as a cache implementation detail (review §5) — is accepted. §1 therefore
changes what this design proposes, not merely how it is implemented.

---

## 1. Executive decision

`run_fastapi_mcp` retains request-sized objects in process-lifetime structures and uses a
live-session default that is too large for payload-heavy workflows. Lowering that cache limit exposes
a correctness problem: evicting a runtime can silently drop application state.

Revision 2 proposed to fix all of that as **one behaviorally atomic release with five modular
fixes**. That packaging is **withdrawn** `[R2-1 … R2-14, review §5]`. The four diagnosed retainers are
real and the direction of the fixes is right, but the state-preservation mechanism that the lower
cache limit depends on is a session checkpoint protocol, and round 2 showed that protocol is not
behavior-preserving, crash-atomic, or security-complete as specified.

### 1.1 Three releases, not one `[review §5]`

The work is split so that the low-risk memory wins are not held hostage to an immature persistence
protocol:

| Release | Contents | Depends on |
|---|---|---|
| **A — universal memory bounds** | Bound terminal `TurnExecution` records; store creation locks weakly; bound per-conversation in-memory history `[R2-11]`; disable DSPy history and predictor trace; make an explicit DSPy response-cache decision `[R2-15]`. No change to `MAX_LIVE_SESSIONS`, no new persistence. | nothing |
| **B — session checkpoint protocol** | Runtime leases and a unified turn lifecycle for streaming `[R2-1, R2-2, R2-3, R2-21]`; an explicit workflow-declared snapshot capability `[R2-5, R2-9, R2-12]`; complete or pinned continuation state `[R2-8]`; a generation-stamped, identity-bound composite commit `[R2-4, R2-6, R2-7, R2-10, R2-13]`; request-scoped credential handling `[R2-14]`; namespace lifecycle and a fleet-version protocol `[R2-17, R2-18]`. | A |
| **C — lower the live-session default** | `MAX_LIVE_SESSIONS` default 2,000 → 50, with the operator-reachable resolver `[R2-20]`. | B |

Release A is behaviorally atomic on its own: every fix in it is a pure retention bound with no new
durable state, no new eviction path, and no wire change. Releases B and C are separately gated.

The ordering is not preference, it is dependency. Lowering the cache limit is what makes eviction
routine; eviction is what makes state preservation necessary; state preservation is where all 14
blocking findings live. Shipping C before B would convert a memory problem into a state-loss problem,
which is the outcome the first non-negotiable rule below exists to prevent.

Three rules are non-negotiable across all three releases:

- A session that cannot be checkpointed safely is **not evicted**. The LRU may remain over target —
  *indefinitely*, not merely temporarily, for the workflow shapes in §1.4 `[R1]`; the framework must
  not trade an allocation problem for silent application-state loss.
- Request-scoped credentials are never written to durable channel state, and never installed into
  shared workflow state outside an accepted turn's lifecycle `[R2-12, R2-14]`.
- DSPy memory controls are installed before the server event loop and request worker threads start,
  not from the FastAPI lifespan task — and their being *in force* is asserted, not assumed
  `[R7]`, and re-asserted against post-readiness override `[R2-19]`.

Release A is a minor release (default DSPy diagnostic behavior and conversation retention change).
Releases B and C change persistent state semantics and the default cache behavior, and must not be
hidden in a patch release.

### 1.2 What Release A does **not** fix, stated up front `[R2-11, R2-22]`

Revision 2's executive decision would have let a reader believe the release bounded process memory.
Two limits must be visible before §2:

**Release A does not close the motivating production OOM.** The motivating deployment issues a unique
`channel_id` per request at a 450 KB payload and was OOM-killed at a 1 GB container limit after
roughly 200 requests. At the unchanged 2,000-session default the live-session cache never reaches its
limit within that window, so it retains **every** channel — its context and its conversation history
— for the entire pre-OOM period. Only Release C reclaims them.

The source study's per-retainer arms cannot be recombined to predict Release A's slope: they sum to
≈2.70 MB/request against a 1.76 MB/request baseline, because the retainers share references to the
same payload objects. Release A's slope is therefore a **measurement**, not a derivation. §16.5 arm
A0 measures it, and the pre-registered prediction is that it exceeds the 0.05 MB/request target by
roughly an order of magnitude. If arm A0 falsifies that prediction, Release C's urgency drops and
this design should be re-scoped rather than continued out of momentum.

**Release C closed it.** With the cap at 50 and eviction writing a durable checkpoint, the same
unique-channel workload measures **+0.011 MB/request** (§16.5.2) — bounded, where Release A left it at
+0.495. The rest of this section is the Release A measurement that established why Release C was
needed, retained because §1.2 exists to stop the release train overstating itself.

**Measured. The prediction is confirmed, so Release C's urgency stands.** Arm A0 as shipped, at 300
requests × 3 replicates and a 450 KB payload against a real Uvicorn server in its own process:

```text
arm A0, unpatched baseline: +1.33860, +1.33993, +1.33841 MB/request  (mean +1.33898)
arm A0, Release A:          +0.49320, +0.49283, +0.49809 MB/request  (mean +0.49471,
                                                                      upper 95% bound +0.49966)
```

Release A removes 0.844 MB/request, 63.1% of the baseline slope, and what remains is **9.9× the
0.05 MB/request target** — the predicted order of magnitude. The residual is attributed rather than
assumed: at 1,000 requests the process holds 440.1 MB of in-memory conversation bytes across 1,001
live sessions, i.e. 0.4397 MB per session, which accounts for 88% of the measured slope. That is the
live-session cache retaining every unique channel at the unchanged 2,000 default, exactly as this
section predicts, and only Release C reclaims it. Arm A0 also shows zero downward RSS steps and a
slope stable across horizons (+0.49471 at 300 requests, +0.49740 at 1,000), which is the signature of
retention rather than allocator high-water.

Until Release C lands, worker recycling remains the operator's mitigation for the unique-channel
workload. That is stated as a supported interim posture, not a footnote.

**Goal 1 is false today for a hot single channel, and Release A is what makes it true `[R2-11]`.**
`append_conversation_turn()` appends unconditionally
(`workflow_execution_context.py:437-451`) while only the last five entries are ever read
(`:1046-1059`). A single channel issuing bounded 450 KB direct actions grows one request-sized entry
per completed request even with the session count at one, terminal executions removed immediately,
and DSPy history and trace off. This is a different workload shape from the unique-channel OOM and it
is not fixed by any cache bound — which is why the conversation window is in Release A rather than
deferred with the checkpoint work.

### 1.3 The checkpoint protocol is a system of record, not a cache detail `[review §5]`

Release B decides which application state is durable, which credentials cross the at-rest boundary,
whether startup may replay, which conversation receives a resumed turn, whether two records form one
state generation, what old and new workers may read, and when a live runtime may be detached. Those
are system-of-record semantics.

Revision 2 tried to establish them with three negative eligibility checks and two JSON files. §11
replaces that with an explicit, workflow-declared capability: **nothing is durable unless the
workflow declares it**, and a session whose state is not fully declared is pinned. That single
principle resolves R2-5, R2-9 and R2-12 together, preserves today's semantics for every workflow that
does not opt in, and makes Release C's bound a property a workflow author opts into rather than one
the framework infers.

### 1.4 Conditional scope: pinned workflows `[R1]`, with the census corrected `[R2-1 audit, review §3]`

The first non-negotiable rule has a consequence that must be stated here rather than discovered in
§11: **Release C's bound is conditional on workflow shape.** A session holding a live command-context
object cannot be checkpointed under §11.2, so it is pinned, so it is never evicted. Workflows that
assign a command-context object therefore retain the pre-change unbounded live-session behavior.

Revision 2's census was wrong in one row and imprecise in another; the review caught the first
(review §3, R1 audit) and the corrected measurement is:

| Workflow | Assigns a command-context object? | Where | Pinned from |
|---|---|---|---|
| `simple_workflow_template` | root | `_commands/startup.py:12` | its first turn |
| `messaging_app_2` | root | `_commands/startup.py:42` | its first turn |
| `tests/todo_list_workflow` | root | `_commands/startup.py:12` | its first turn |
| `messaging_app_4` | root | `_commands/set_root_context.py:15` | when `set_root_context` runs |
| `messaging_app_3` | current | `_commands/initialize_user.py:46` | when `initialize_user` runs |
| `messaging_app_1`, `retail_workflow`, `hello_world` | none | — | never — evictable |

Two corrections to revision 2. `messaging_app_3` was listed as evictable; it is not — it assigns
`workflow.current_command_context`, and §11.2's first eligibility clause fails on the current context
just as it does on the root. `messaging_app_4` was listed as pinned from its first turn; it has no
`startup.py` at all, so it is pinned only once an explicit `set_root_context` command runs. The
distinction matters for §16.5 arm B, whose fixture must pin from turn 1 to be the adversarial case.

`Workflow.__init__` starts with `_root_command_context = None` (`fastworkflow/workflow.py:160`), so
eligibility is entirely determined by whether the application assigns one, and the root setter also
populates `_current_command_context` (`workflow.py:216-217`), so both clauses fail together.

Consequences accepted by this release train:

- Release A applies to every workflow. Release C applies only to workflows that declare a durable
  state projection under §11.2 and hold no live command-context object.
- For pinned workflows the source study's session-cache share of growth — approximately
  0.45 MB/request at a 450 KB payload for context alone — remains. A pinned workflow **cannot** meet
  the §16.5 shipping gate and must not be described as bounded.
- `simple_workflow_template` is the scaffold new workflows are copied from, so the pinned case is the
  *default* for newly authored workflows, not an exotic one.

§16.5 therefore requires an object-context soak arm with its own pre-registered expectation, and
§19.3's follow-up trigger (a workflow-owned serializer) is treated as **already fired**. The
alternative considered and rejected for this release train — capping the pinned set and shedding load
with 503 — is decision 18.

---

## 2. Evidence status

### 2.1 Verified in this working tree (carried forward from revision 2)

**VERIFIED — terminal turns are retained forever.**

- `TurnRegistry._by_key` receives every execution at `fastworkflow/run_fastapi_mcp/turns.py:202-213`.
- `clear_active()` removes only `_active_by_channel` (`turns.py:216-224`).
- `ttl_expires_at` is never assigned, so `evict_terminal()` is a no-op (`turns.py:226-243`).
- A fresh local reproduction inserted 300 terminal executions; eviction removed 0 and all 300
  remained.

**VERIFIED — creation locks grow with unique channel IDs.**

- `ChannelSessionManager._creation_locks` is a strong dictionary with no removal path
  (`run_fastapi_mcp/utils.py:689-701`).
- A fresh local reproduction requested locks for 300 unique IDs and retained 300.

**VERIFIED — the session cache can lose app state.**

- The manager defaults to 2,000 live sessions and reclaims only by count-based LRU
  (`utils.py:674-743`).
- The live runtime strongly owns the app workflow; the process registry is weak
  (`workflow.py:14-45`).
- `WorkflowExecutionContext.close()` closes only the CME workflow
  (`workflow_execution_context.py:476-490`).
- A fresh max-one-session reproduction stored `{"payload": "retained-state"}`, evicted the runtime,
  confirmed the old workflow was collected, and recreated the channel with `{}`.

**VERIFIED — DSPy diagnostic structures retain calls by default**, and the §9.1 policy suppresses
them in the production call shape (`Predict` inside `dspy.context` inside a `ThreadPoolExecutor`,
stubbing only `LM.forward`): baseline reached 4/4/4 entries in `GLOBAL_HISTORY`, the LM's history and
`settings.trace`; with the policy all three stayed at 0/0/0 `[R7]`. Read sites confirmed at
`dspy/clients/base_lm.py:98` and `:223` (`disable_history`) and `dspy/predict/predict.py:228`
(`max_trace_size > 0`).

**VERIFIED — lifespan-time DSPy configuration is unsafe in repeated app lifecycles.** DSPy assigns
global configuration ownership to the first thread and async task that calls `configure()`; calling
the proposed configuration in one `asyncio.run()` and then from a second async task raised DSPy's
owner-task `RuntimeError`.

**VERIFIED — `asyncio.Lock` supports the §7 weak-mapping argument.** A `WeakValueDictionary` entry
persists while any holder or waiter has a strong reference and drops to zero entries once all
release.

**VERIFIED — `get_env_var` never consults `os.environ` when a default is supplied** `[R4]`.
`fastworkflow/__init__.py:215-219` returns `default` before reaching `os.getenv`.

**Interpreter pin** `[R15]`. The DSPy facts hold for `.venv/bin/python` (DSPy 3.2.1). The ambient
interpreter on the development host resolves to DSPy 2.6.27, so §16.4's isolated subprocesses must
invoke `.venv/bin/python` explicitly.

### 2.2 Verified while absorbing round 2

Round 2's mechanism probes were re-run independently against the pinned tree rather than taken on
trust, and absorbing the required resolutions surfaced four constraints the review did not state.

**REPRODUCED — DSPy's response cache survives the proposed history/trace policy** `[R2-15, review
§6.1]`. Four unique `Predict` calls under the §9.1 policy, with `litellm_completion` replaced below
`LM.forward` and no network call:

```text
DSPy response-cache entries: 4
GLOBAL_HISTORY:              0
LM history:                  0
settings.trace:              0
```

Installed defaults are `memory_max_entries=1000000` and `disk_size_limit_bytes=30000000000` (30 GB),
and `dspy.LM` defaults `cache=True` while `get_lm()` never overrides it
(`fastworkflow/utils/dspy_utils.py:42-69`). The server therefore runs with a one-million-entry
in-memory response cache that the revision-2 policy does not touch.

**REPRODUCED — the DSPy policy can be silently disabled after the startup assertion** `[R2-19, review
§6.2]`. Synchronous `configure()` claims the owner thread but leaves the async owner unset
(`dspy/dsp/utils/settings.py:117-145`), so the first async task on that thread may reconfigure:

```text
after synchronous startup configure: disable_history=True,  max_trace_size=0, async_owner=None
after first async-task configure:     disable_history=False, max_trace_size=10000, async_owner=set
```

`Settings.configure` also performs no key validation — an unknown key is accepted and reads back —
and `dspy.settings.<attr> = value` routes through `configure()` (`settings.py:87-91`).

**VERIFIED — the lossy coercion is in `WorkflowExecutionContext`, not the store** `[R2-7]`.
`serialize_state()` returns `json.loads(json.dumps(payload, default=str))`
(`workflow_execution_context.py:365`), so unsupported objects have already become strings before any
store-level strictness could see them. `DiskSessionStateStore.save` applies `default=str` a second
time (`session_state_store.py:61-64`).

**VERIFIED — disk channel IDs are non-injective** `[R2-13, review §6.3]`.
`safe_id = channel_id.replace(os.sep, "_").replace("/", "_")` (`session_state_store.py:50-52`) maps
`tenant/a` and `tenant_a` to the same file. Redis keys use the raw channel ID under a fixed prefix
with no workflow or deployment namespace (`:83-94`).

**VERIFIED — the factory cannot express a second namespace** `[R11]`. `get_session_state_store` takes
only `base_folder`, and for `SESSION_STATE_STORE=redis` `base_folder` is ignored entirely
(`session_state_store.py:112-142`).

**VERIFIED — no `dspy.configure()` or `dspy.settings.configure()` call exists anywhere in the
`fastworkflow` package.** This strengthens §9.2's ownership claim: installing a process-global policy
at the entrypoint cannot collide with request-path code, which uses only `dspy.context(...)`.

Four absorption-time constraints, each of which changes a required resolution:

1. **`[R2-14 · absorption]` The bearer token in `workflow.context` is a documented public contract.**
   `fastworkflow/run_fastapi_mcp/README.md:86-103` tells workflow authors to read
   `workflow_context['http_bearer_token']`. The review's "carry request credentials on the execution
   or in a ContextVar" cannot mean removing the key, or every authenticated workflow breaks. §11.7
   therefore keeps the read contract and moves only the *write* under the accepted turn's lifecycle.
2. **`[R2-11 · absorption]` An in-memory conversation window truncates durable history unless
   persistence changes first.** `save_conversation_incremental` extracts all turns from in-memory
   history and `save_conversation_turns` assigns `conv["turns"] = turns`
   (`conversation_store.py:264-269`) — a full replace, not an append. Trimming memory first would
   delete durable turns. It is also O(n²) in write volume: turn *n* rewrites all *n* turns. §8
   therefore makes durable append the prerequisite step, not an afterthought.
3. **`[R2-10 · absorption]` Pinning awaiting sessions removes the controlled-eviction half of the
   two-commit problem but not the crash half, and does not help multi-pod.** If a session that is
   `awaiting_user` is never evicted, the context and pending records are never both live for the same
   channel at eviction time. A crash while awaiting still leaves a pending record with a stale or
   absent context record, and a *different pod* still cold-rehydrates from the store, so the
   completeness work in R2-8 remains required for multi-pod deployments regardless of pinning.
4. **`[R2-15 · absorption]` The in-tree cache utility is already broken against DSPy 3.2.1, and the
   cache probe requires a function, not a `Mock`.** `fastworkflow/utils/dspy_cache_utils.py:21-25`
   passes `enable_litellm_cache`, which `configure_cache` no longer accepts
   (`TypeError: configure_cache() got an unexpected keyword argument 'enable_litellm_cache'`); it has
   no in-process callers, so nothing depends on it, but it cannot be reused by the server helper and
   is not evidence that cache control works. Separately, the cache wrapper computes
   `f"{fn.__module__}.{fn.__qualname__}"` (`dspy/clients/cache.py:246`), so a `unittest.mock.Mock`
   patch of `litellm_completion` raises `AttributeError` before reaching the cache — §16.3's probe
   must patch with a real function carrying `__module__` and `__qualname__`.

### 2.3 Supplied by the source study

The following production numbers are credible design inputs but were not independently rerun because
the treatment patch is not in this working tree:

- approximately 1.758 MB/request before the patch at a 450 KB payload;
- approximately 0.032 MB/request after the full local patch;
- a successful 2,000-request soak;
- the reported per-mitigation ablation arms (baseline 1.763; registry bounded 0.885; session bounded
  0.829; history disabled 0.880; all three 0.440; plus trace 0.032).

The numeric treatment claim is **INCONCLUSIVE locally**, not rejected. It must be reproduced with
baseline and treatment artifacts before appearing in release notes. The arms are non-additive and
must never be presented as independent shares.

### 2.4 Scope of the guarantee

Release A bounds growth with the number of completed turns, unique channels seen, and turns within
one conversation, provided each individual payload value is itself bounded. Release C additionally
bounds growth with the number of safely evictable live sessions.

None of them provides a byte budget:

- one arbitrarily large active request can still exhaust memory;
- active sessions are never evicted, so peak memory scales with legitimate peak concurrency;
- sessions with undeclared or object-valued state are pinned rather than corrupted, so for those
  workflows the live-session count is **not** bounded at all — §1.4 `[R1]`;
- durable storage is not bounded by this release train; §11.10 and §16.5 carry an explicit growth
  number and a lifecycle dependency rather than an instruction to monitor `[R5, R2-17]`.

Two honesty notes about the count-based approach:

- The source study rejects DSPy's caps because they are "bounded by entry count, which is meaningless
  when prompts are large." This design's caps are also count-based, so the same criticism applies.
  §5.2 and §14 state the **byte cost at the representative payload** for every count cap `[R12]`, and
  §9.4 applies the same discipline to the DSPy cache cap `[R2-15]`.
- A count bound only holds where eviction can actually occur. §1.4 is the boundary condition.

Global admission control and backpressure remain Step 2 of
`docs/fastworkflow_turns_async_execution_design.md`.

---

## 3. Goals and non-goals

### 3.1 Goals

1. Completed requests must not produce monotonic process-memory growth — **including on a single hot
   channel**, which requires the conversation bound in §8 `[R2-11]`.
2. Inactive retention must have small, explicit count bounds, each with a stated byte cost at the
   representative payload.
3. Eviction must preserve behavior for every session it actually evicts, where "behavior" is defined
   by an explicit enumerated capability rather than inferred from negative checks `[R2-9]`.
4. Active work must never be closed mid-mutation, where "active" is established by a lease held
   across the whole window, not sampled `[R2-2]`.
5. Existing HTTP/MCP wire behavior must remain unchanged, including the documented
   `workflow_context['http_bearer_token']` read contract `[R2-14 · absorption]`.
6. The normal live-session hit path must receive effectively no new latency.
7. No network LLM call may be added.
8. Configuration must remain small.
9. Tests must use real framework components and workflows and never touch trained example artifacts.
10. Durable state must not silently change a deployment's data-classification boundary `[R2-12]`.

### 3.2 Non-goals

- A byte-accurate process memory budget.
- Global turn backpressure or a sized worker pool.
- `GET /turns/{turn_key}` or a distributed/durable turn store. Release A therefore drops revision 2's
  claim that startup output stays collectable across runtime eviction `[R2-16]`.
- General-purpose Python object persistence, pickle, or cloudpickle.
- Full command-context navigation and object restoration (`fix-cgs`).
- Direct-action validation.
- A hidden expiry policy for durable application state.
- Eliminating allocator high-water behavior; worker recycling remains a valid backstop and is the
  supported interim mitigation until Release C `[R2-22]`.
- **Exactly-once semantics for arbitrary startup side effects.** No framework envelope can provide
  them for effects outside framework-managed state; §11.5 scopes the guarantee explicitly and
  requires application-owned idempotency beyond it `[R2-5]`.

---

## 4. Load-bearing invariants

Grouped by the release that must establish them. Invariants 1–18 are carried forward from revision 2
with the scope corrections named; 19–31 are new in revision 3. Numbers are stable identifiers, not
positions, so they are non-contiguous within each group.

### 4.1 Always (unchanged by this design)

- **1.** A `QUEUED` or `RUNNING` execution is never age- or count-evicted.
- **2.** *Scoped* `[R2]`. The registry's active pointer is the source of truth for **409 idempotency
  behavior**; never replace it with `runtime.lock.locked()` *for that purpose*, because the lock is
  released while a request defers and across `AWAITING_USER`, which re-opens the v2.22.0
  double-execution race. This invariant does **not** govern eviction safety; see invariants 6 and 19.
- **3.** Conversation and suspended-trajectory persistence completes before `DONE`.
- **4.** Request wait timeout never cancels the independently owned execution task.
- **5.** Registry insertion still precedes task launch; a launch failure rolls the insertion back.
- **15.** Single-writer-per-channel remains required.

### 4.2 Release A

- **13.** DSPy policy applies only to `run_fastapi_mcp`.
- **14.** *Extended* `[R7, R2-19]`. The server is not ready until configuration is valid and memory
  controls are **proven** active — proven by a structural assertion, not by having called
  `configure()`. After readiness, the policy is **owned**: a post-readiness change to DSPy global
  settings is rejected or fails the process, because a one-time probe cannot protect a mutable global.
- **17.** `[R9]` No store I/O ever executes while `TurnRegistry._lock` is held. Post-turn trimming is
  scheduled after that lock is released.
- **23.** *New* `[R2-11]`. In-memory conversation history is bounded, and the durable conversation
  record is append-only with respect to it. A turn is never dropped from memory before it is durably
  recorded, and trimming memory never shortens the durable record.
- **24.** *New* `[R2-15]`. Every DSPy structure that retains request-sized data across calls —
  diagnostic history, predictor trace, and the response cache — has an explicit server policy with a
  stated byte cost. "Not configured" is not a policy.

### 4.3 Release B

- **6.** *Corrected* `[R2]`, *superseded in part* `[R2-2]`. A channel with work in flight is never
  removed from the live-session cache, where "in flight" is at minimum the **union** of a live registry
  execution and a held `runtime.lock`. The registry pointer alone is insufficient because
  `/invoke_agent_stream` runs a full turn without ever creating a `TurnExecution`. The union is
  necessary but **not sufficient** — see invariant 19.
- **7.** *Scoped* `[R10]`, *corrected* `[R2-7]`. Runtime removal occurs only after every state item
  required for supported cold rehydration has been persisted successfully, where "successfully" means
  the write both returned without error **and** was lossless. Losslessness is established at the
  *first* serializer — `WorkflowExecutionContext.serialize_state()` — not at the store, because a
  `default=str` round-trip upstream makes a store-level strictness check vacuous.
- **8.** Serialization or store failure leaves the runtime live and produces a rate-limited warning.
- **9.** Persisted state is strict, versioned JSON. No `default=str`, anywhere in the path.
- **10.** Empty saved state is distinct from absent saved state; clearing context cannot resurrect old
  or launch-time values.
- **11.** *Replaced* `[R2-12]`. No state is durable unless the workflow has declared it durable.
  Credentials and unclassified values are never persisted; a session whose state is not fully declared
  is pinned rather than partially written. This supersedes revision 2's "exclude one top-level key"
  rule, which persisted everything else by default.
- **12.** Context encoding, hashing and writing occurs only during retirement or graceful shutdown, not
  after every turn — **except** the semantic-metadata commit required by invariant 25.
- **16.** *Narrowed* `[R11]`, subsumed by invariant 26. No two record kinds share a key space on any
  backend. With invariant 26 in force this is a property of the composite record's key derivation
  rather than a separate namespace-discipline rule, but it remains testable and is tested (§16.3).
- **18.** `[R3]` Whether startup has already run for a channel is derived from persisted state, never
  from in-process turn retention, so it cannot depend on wall-clock retention windows.
- **19.** *New* `[R2-2]`. Eviction safety is established by a **lease**, not a sample. A runtime lease
  is acquired atomically with the manager-lock lookup that returns the runtime, and is held until turn
  registration or lock acquisition takes over and then until the executor actually exits. A boolean
  predicate evaluated after the manager lock is released cannot make lookup and admission atomic.
- **20.** *New* `[R2-1]`. A runtime under construction holds an initialization lease and is never an
  eviction candidate until startup admission or completion establishes its final eligibility. A manager
  must never evict the runtime it is in the middle of creating.
- **21.** *New* `[R2-3]`. Every path that mutates a `WorkflowExecutionContext` is registered with the
  turn registry for its whole duration, streaming included. Registry ownership and the runtime lease
  outlive client timeout and disconnect, and end only when the executor exits.
- **25.** *New* `[R2-4]`. Any channel fact whose loss would change behavior after restart — startup
  completion above all — is committed durably before the fact becomes observable, and its durability
  does not depend on whether `workflow.context` happened to change.
- **26.** *New* `[R2-10]`. The context record and the continuation record form **one state
  generation**. They are written under a generation stamp with a commit record, and a restore that
  observes mismatched generations fails closed and quarantines the channel rather than combining them.
- **27.** *New* `[R2-13]`. A durable record is bound to its identity: record type, namespace,
  deployment, workflow fingerprint, collision-resistant channel key, and session incarnation are stored
  inside the record and validated on read. Channel-key encoding is injective.
- **28.** *New* `[R2-6]`. Restore reconciliation is only permitted where the information required to
  perform it is actually stored. A single aggregate digest supports detection of change, not
  attribution of change, so reconciliation stores the prior canonical launch projection.
- **29.** *New* `[R2-14]`. Request-scoped credentials are installed into shared workflow state only
  under an accepted turn's lifecycle, immediately before executor dispatch, and cleared after. A
  rejected or deferred request never mutates shared state.
- **30.** *New* `[R2-21]`. Shutdown closes admission before it drains, and after its deadline expires
  it **never** snapshots or closes a runtime that is still busy. A deadline is a bound on waiting, not
  a license to write.
- **31.** *New* `[R2-18]`. Durable records carry a protocol version, and a deployment declares a fleet
  version floor. A node that cannot read the current protocol version must not silently ignore the
  namespace and write around it.

### 4.4 Release C

- **22.** *New* `[R2-20]`. `MAX_LIVE_SESSIONS` resolution gives the process environment precedence over
  workflow env files, and the effective value **and its source** are logged. An operator control whose
  documented default outranks the container override is not an operator control.

---

## 5. Configuration

### 5.1 One new environment variable

```text
MAX_LIVE_SESSIONS=50
```

Requirements:

- integer greater than zero;
- resolved after `fastworkflow.init()` loads workflow env files;
- validated before readiness becomes true;
- resolved with `default=None` and the fallback applied explicitly, so that a process environment
  variable is reachable at all `[R4]`;
- resolved with **explicit OS-first precedence** in a dedicated resolver, and the effective value plus
  its source logged `[R2-20]`;
- documented in `fastworkflow/examples/fastworkflow.env` **as a commented line, not an active
  assignment** `[R2-20]`.

The first `default=None` requirement exists because `get_env_var` returns the supplied default
**before** consulting `os.environ` (`fastworkflow/__init__.py:215-219`):

```python
value = _env_vars.get(var_name)
if value is None:
    if default is not None:
        return default          # os.getenv is never reached
    value = os.getenv(var_name)
```

Round 2 showed that fixing only this is insufficient `[R2-20]`. `default=None` reaches `os.environ`
only when `_env_vars` does not contain the key — and every existing default in
`fastworkflow/examples/fastworkflow.env` is an **active assignment**
(`SPEEDDICT_FOLDERNAME=___workflow_contexts`, `NOT_FOUND="NOT_FOUND"`, …), with commented lines
reserved for genuinely optional overrides such as `# INTENT_DETECTION_TINY_MODEL=…`. If
`MAX_LIVE_SESSIONS=50` follows the active-assignment pattern and an operator copies the example file,
a Kubernetes or Docker `MAX_LIVE_SESSIONS=500` is silently ignored and 50 wins. Revision 2's R4 fix
would have been defeated by revision 2's own R4 documentation requirement.

The resolver therefore reads, in order: process environment, then workflow env file, then
`DEFAULT_MAX_LIVE_SESSIONS`; and logs `max_live_sessions=<value> (source=env|file|default)`. This is a
deliberate, documented exception to the framework's usual file-first precedence, justified because the
motivating deployment is a container that was OOM-killed at a 1 GB limit and the natural knob there is
an orchestrator environment variable. §17 depends on this being real rather than nominal.

### 5.2 Private constants

```python
DEFAULT_MAX_LIVE_SESSIONS = 50          # Release C; 2000 until then
TURN_RETENTION_SECONDS = 300.0
MAX_RETAINED_STARTUP_TURNS = 20
MAX_CONVERSATION_TURNS_IN_MEMORY = 20   # Release A, R2-11
SERVER_DSPY_MEMORY_CACHE_ENTRIES = 200  # Release A, R2-15
```

There are deliberately no environment variables for terminal-turn count or age, DSPy history, DSPy
predictor trace, DSPy cache size, conversation window, or live-session idle age.

Turn-result collection and live-session capacity are different policies. Raising `MAX_LIVE_SESSIONS`
must not silently make terminal request-payload retention grow into gigabytes, so the terminal cap is
fixed rather than derived from the session cap.

**Every count cap states its byte cost** `[R12]`, because a count cap is meaningless without the
payload size. At the representative 450 KB payload:

| Cap | Value | Bytes at 450 KB | Derivation |
|---|---:|---:|---|
| `MAX_RETAINED_STARTUP_TURNS` | 20 | ≈17 MB | ≈0.87 MB/record measured (source study: ~260 MB / 300 requests); sole consumer needs ≈6 records to cover 300 s at ≈65 requests/hour; 20 is ≈3× margin. Revision 1's 100 would have cost ≈87 MB — ≈4× the entire live-session budget. |
| `MAX_CONVERSATION_TURNS_IN_MEMORY` | 20 | ≈9 MB/channel | Only the last 5 entries are read (`workflow_execution_context.py:1052`); 20 is 4× margin. Per-channel, so the aggregate scales with live sessions `[R2-11]`. |
| `SERVER_DSPY_MEMORY_CACHE_ENTRIES` | 200 | ≈90 MB worst case | Bounds a structure whose installed default is 1,000,000 entries. 200 is a starting figure that §16.6 must confirm against measured entry bytes, not a derived one `[R2-15]`. |
| `MAX_LIVE_SESSIONS` (Release C) | 50 | ≈23 MB | ≈0.45 MB/session for context. Excludes conversation history, which invariant 23 bounds separately. |

The DSPy cache row is the one whose derivation is weakest, and it is marked as such rather than
presented as settled: entry cost depends on prompt and response size, and §16.6 measures bytes rather
than trusting the count.

### 5.3 Why there is no live idle TTL

The count-based LRU is the request-count bound. A live idle TTL would add another configuration axis,
wall-clock sweep behavior, more cold starts and state writes, and another way to alter startup and
idempotency behavior. Add idle expiry only if production evidence shows material value after the
count bound ships.

---

## 6. Release A — terminal turn retention

### 6.1 Retain only currently collectable terminal work

Today only `/initialize` looks up a terminal execution by `turn_key`
(`run_fastapi_mcp/__main__.py:689-705`). Other turn endpoints use registry state only while an
execution is active; after completion, retries do not consult the retained record.

Therefore:

- retain terminal `kind == "initialize_startup"` executions;
- remove other terminal kinds from `_by_key` during finalization;
- preserve the local `TurnExecution` reference held by the completing request so response rendering is
  unaffected.

**The recoverability claim is withdrawn** `[R2-16]`. Revision 2 said startup output "remains
recoverable within the bounded window across live-runtime eviction." No such access path exists:
there is no `GET /turns/{turn_key}` route, `InitializationRequest` carries no prior `turn_key`, and
cold `/initialize` returns tokens only. Retention is therefore justified solely by the **live-runtime
re-poll** path — a client that calls `/initialize` again while the runtime is still live and its
startup turn is still retained — which is real and is what §6.4 describes.

**The cap binds before the window in a burst, and that is accepted** `[R2-16]`. Twenty-one startup
completions in quick succession evict the oldest seconds after it completed, despite almost all of its
300 s TTL remaining; raising `TURN_RETENTION_SECONDS` cannot help when the count binds. The design
accepts this rather than sizing for peak bursts, because the only consumer is a same-runtime re-poll
that a client performs immediately. If a real client is later found to re-poll after a 20-startup
burst, the correct response is a bounded replay surface in Step 2, not a larger in-process cap.

When `GET /turns/{turn_key}` is implemented, Step 2 must broaden retained kinds, re-evaluate the
capacity policy, and size retention from peak bursts rather than average request rate.

### 6.2 Terminal transition

In `_run_turn`'s `finally` path:

1. stamp `finished_at`;
2. set `exec_state=DONE`;
3. call `clear_active(channel_id, turn_key)`, which already removes the matching active pointer under
   the registry lock (`run_fastapi_mcp/turns.py:216-224`), and **inside that same lock block**:
   a. remove the execution immediately if its kind is not collectable;
   b. otherwise assign `ttl_expires_at = finished_at + 300 seconds` once;
   c. remove expired retained startup executions;
   d. if more than `MAX_RETAINED_STARTUP_TURNS` retained startup executions remain, remove the oldest
      by `(finished_at or created_at, turn_key)`;
4. set `done_event`;
5. **after the registry lock has been released**, request live-session trimming (Release C, §12).

Revision 1 listed the `clear_active` call and "remove the matching active pointer" as two separate
steps `[R13]`. They are one step. The distinction mattered: an implementer following the list
literally would either remove the pointer twice or re-enter `registry._lock` inside `clear_active`,
and `asyncio.Lock` is not reentrant, so the second reading deadlocks inside a turn's `finally` block.

The sweep contains no `await`. The just-completed retained startup is newest and unexpired, so normal
overflow cannot remove it before the original waiter renders.

Step 5 is ordered explicitly because trimming performs store I/O. Doing it inside the lock block would
hold `registry._lock` across a synchronous write and block `start_or_get_active` — that is, every turn
submission on every channel, not just this one. Invariant 17 states this as a rule `[R9]`.

Age cleanup is opportunistic. An expired record may remain while the process is entirely idle, but the
count cap remains the hard completed-record bound.

### 6.3 Task-launch rollback

`start_or_get_active()` inserts the execution and active pointer before invoking `run_turn(execution)`.
Preserve that construction order, but if task creation raises synchronously, remove both inserted
pointers before re-raising. Otherwise a rare launch failure creates an immortal non-terminal entry
that retention must never evict.

### 6.4 Cold-session startup, and why it is not a Release A problem

Revision 2 solved cold-session startup re-execution by moving the startup-completion fact into a
durable envelope `[R3]`, replacing a registry lookup that made re-execution depend on a wall-clock
retention window. That resolution is retained, but it belongs to Release B, because it requires the
durable envelope that Release B introduces (§11.5), and round 2 showed it also requires a commit point
that revision 2 never specified `[R2-4]`.

In Release A the live-session default is unchanged at 2,000, so cold-session startup remains as rare as
it is today and behaves exactly as it does today. Release A therefore changes nothing about startup.
Release C must not ship before §11.5 does, because lowering the cache limit is precisely what makes the
cold path routine.

Do not generalize terminal deduplication to ordinary completed turns. Two identical user queries may be
intentional without a client-supplied request ID.

### 6.5 Observable behavior

- No response shape or status code changes.
- A client re-polling `/initialize` against a still-live runtime sees its startup result within the
  bounded window, exactly as today.
- If the terminal record has been evicted but the runtime is live, `/initialize` returns the existing
  tokens-only response; resubmitting does not rerun startup.
- Startup output is **not** claimed to be recoverable after the runtime is gone `[R2-16]`.

---

## 7. Release A — creation locks

Replace the strong mapping with:

```python
weakref.WeakValueDictionary[str, asyncio.Lock]
```

`asyncio.Lock` is weak-referenceable on supported Python. The caller and queued waiters hold strong
references while the lock matters; after all release it, the weak mapping drops the entry.

Do not manually pop an apparently unlocked lock. A waiter may have been awakened but not yet resumed;
deleting the shared lock in that window lets a new caller create a second lock and defeats
single-flight creation. Weak values also clean failed creations that never reach an eviction path.

Review §2.3 confirms this is the right primitive; it is unchanged from revision 2.

---

## 8. Release A — conversation-history bounds `[R2-11]`

This section is new in revision 3. It exists because goal 1 is false today on a supported workload
that no cache bound touches.

### 8.1 The retainer

`append_conversation_turn()` appends unconditionally to `dspy.History.messages`
(`workflow_execution_context.py:437-451`). Only the last five entries are read, by
`_refine_user_query` (`:1046-1059`). Nothing trims older entries; history is cleared only by an
explicit `/new_conversation`, and is persisted and restored wholesale.

A hot channel issuing bounded 450 KB direct actions therefore grows one request-sized conversation
entry per completed request even when terminal executions are removed immediately, the session count
is one, DSPy history and trace are disabled, and no object context pins additional channels. The
direct-action path appends the record itself (`:1037-1039`), so the growth is proportional to payload,
not to a summary.

### 8.2 Durable append comes first `[R2-11 · absorption]`

An in-memory window cannot be applied until persistence stops depending on in-memory completeness.
`save_conversation_incremental` extracts **all** turns from in-memory history
(`run_fastapi_mcp/utils.py:798`) and `save_conversation_turns` assigns `conv["turns"] = turns`
(`conversation_store.py:264-269`). Trimming memory first would therefore delete durable turns on the
next save — converting a memory leak into data loss, which is the exact trade this design refuses
everywhere else.

The same code is O(n²) in write volume: turn *n* rewrites all *n* turns, so a 450 KB-per-turn
conversation writes ≈`n × 450 KB` on every turn. That is a latency and durable-growth defect
independent of memory, and it is why this step is worth doing on its own.

Order of work:

1. give the conversation store a true append (`append_conversation_turns(conversation_id, new_turns)`)
   plus a per-runtime high-water mark of turns already durably recorded;
2. change `save_conversation_incremental` to append only turns above the high-water mark;
3. only then apply the in-memory window.

### 8.3 The window

After a turn is durably recorded, trim `conversation_history.messages` to the newest
`MAX_CONVERSATION_TURNS_IN_MEMORY` entries. Invariant 23 governs the ordering: a turn is never dropped
from memory before it is durably recorded.

Reads are unaffected: `_refine_user_query` uses the last five. Restore is unaffected in shape:
`restore_history_from_turns` is handed the same windowed tail rather than the full conversation.

Two consequences to state rather than discover:

- **Topic and summary generation sees the window, not the conversation.**
  `finalize_conversations_on_shutdown` and `generate_topic_and_summary` operate on
  `extract_turns_from_history` (`__main__.py:308`). Once history is windowed they must read the
  durable conversation record instead, or topics silently become "topic of the last 20 turns." This is
  a required part of the change, not an optional polish.
- **`/activate_conversation` already replaces history wholesale**
  (`__main__.py:1471-1472`); it must load the windowed tail and set the high-water mark accordingly, or
  the next incremental save re-appends the whole conversation.

### 8.4 What this does and does not buy

It bounds the hot-single-channel workload, which is a supported shape and is currently unbounded.

It does **not** help the motivating unique-channel workload, where each channel has exactly one
conversation turn. There the conversation entry is retained because the *session* is retained, and only
Release C reclaims it. Saying otherwise would overstate Release A, which §1.2 exists to prevent.

---

## 9. Release A — DSPy server memory controls

### 9.1 History and trace policy

```python
dspy.settings.configure(
    disable_history=True,
    max_trace_size=0,
    trace=[],
)
```

`disable_history=True` is required: `max_history_size=0` alone still permits DSPy's separate
`GLOBAL_HISTORY` append. `max_trace_size=0` is the predictor-trace off switch; a fresh empty trace also
releases any pre-start trace references. Clear `GLOBAL_HISTORY` immediately before installing the
policy — the helper cannot find arbitrary old per-LM histories, so ordering is part of the contract.

### 9.2 Timing and ownership

Do **not** make the first configure call inside FastAPI's async lifespan. Install the policy
synchronously from the dedicated server entrypoint before `uvicorn` creates the event loop and before
any server LM is created.

The helper is idempotent: if the desired settings already hold, return without another configure call;
if a different thread or task owns incompatible DSPy global settings, fail startup with an actionable
error instead of advertising a known leak.

Application code inside executor threads must use `dspy.context(...)` for temporary overrides, not
process-global `configure()`. This is already true of every call site in the tree: **there is no
`dspy.configure()` or `dspy.settings.configure()` call anywhere in the `fastworkflow` package**
(§2.2). That is what makes claiming process-global ownership at the entrypoint safe. Revision 2's
phrasing implied fastWorkflow calls `configure_cache(...)` on the request path; it does not — the only
in-tree caller is a standalone maintenance script with no in-process callers, and that script is
itself broken against DSPy 3.2.1 (§2.2 item 4).

### 9.3 Assert the policy, then keep owning it `[R7, R2-19]`

Configuring is not evidence that the policy is in force. `Settings.configure` performs **no key
validation** — it is `main_thread_config[k] = v` — so on any DSPy 3.0.x that does not read
`disable_history` or `max_trace_size`, the policy is a silent no-op while §17 cheerfully logs
`dspy_history=off`. `pyproject.toml:50` allows `dspy = "^3.0.1"`.

So invariant 14 requires a **positive structural assertion** at startup, immediately after installing
the policy and before readiness becomes true:

1. clear `GLOBAL_HISTORY`;
2. install the policy;
3. make one throwaway `Predict` call through a stub `LM.forward` boundary — no network call, no
   provider credentials, no cost;
4. assert `GLOBAL_HISTORY`, the stub LM's history, and `settings.trace` are all still empty;
5. on failure, fail readiness with an error naming the installed DSPy version and the settings.

**A one-time probe is not sufficient, which revision 2 missed** `[R2-19]`. Synchronous `configure()`
claims the owner thread but leaves the async owner unset, so the first async task on that thread — the
lifespan task, or any endpoint coroutine — may reconfigure global settings and the startup probe stays
green in the logs. Measured (§2.2):

```text
after synchronous startup configure: disable_history=True,  max_trace_size=0, async_owner=None
after first async-task configure:     disable_history=False, max_trace_size=10000, async_owner=set
```

Attribute assignment (`dspy.settings.x = y`) also routes through `configure()`
(`settings.py:87-91`), so the hazard is not limited to explicit calls.

The policy is therefore **owned**, not merely installed:

- claim the async owner deliberately before accepting traffic, so any later async `configure()` raises
  DSPy's own ownership error rather than succeeding;
- re-assert the policy at readiness-probe time so a drift becomes an unready pod rather than a silent
  leak;
- add a negative test that a post-readiness override is rejected (§16.4 step 8).

**State the supported worker and reload model** `[R2-19]`. Spawned Uvicorn workers do not inherit the
parent's globals, so either each worker process runs the bootstrap or multi-worker and `--reload` modes
are explicitly refused. The entrypoint must not silently support a configuration in which half the
processes have no policy.

### 9.4 Response cache policy `[R2-15]`

Revision 2 disabled DSPy's diagnostic structures and left its response cache untouched. Reproduced in
§2.2: four unique calls under the §9.1 policy leave history and trace at zero and the response cache at
four entries. Installed defaults are 1,000,000 in-memory entries and a 30 GB disk cache, every
`dspy.LM` defaults `cache=True`, and `get_lm()` never overrides it.

Decision: **bound the memory cache explicitly and disable the disk cache in the server entrypoint.**

```python
dspy.configure_cache(
    enable_memory_cache=True,
    memory_max_entries=SERVER_DSPY_MEMORY_CACHE_ENTRIES,
    enable_disk_cache=False,
)
```

Reasoning, and the trade-off it accepts:

- The retainer under scrutiny is process memory, so `memory_max_entries` is the load-bearing knob. One
  million entries of request-sized prompts is unbounded for practical purposes.
- The disk cache is off because 30 GB is not a sane default for a container, this design already has a
  durable-growth problem (§11.9), and DSPy's disk cache uses pickle
  (`configure_cache(restrict_pickle=False)` by default), which conflicts with decision 10's no-pickle
  stance for anything this server owns.
- The cost is real and must be measured, not waved away: in the motivating unique-channel workload the
  cache never hits, so disabling disk costs nothing; in a repeated-prompt workload it means more
  provider calls after a restart. §16.6 measures LLM call counts with and without the policy, and §17
  logs the effective cache configuration.
- The 200-entry figure is a starting point sized to keep worst-case retention comparable to the
  live-session budget, not a derived number. §16.6 measures entry bytes and the figure is revisited
  against that measurement.

The alternative — disabling the response cache entirely — was rejected because parameter extraction and
NLU issue structurally similar prompts within a session, so a small cache is likely to pay for itself;
but that is an assumption, and §16.6 is what tests it.

Note for whoever implements this: `configure_cache` **replaces** the global cache object, so it runs at
the same entrypoint and in the same ordering as §9.2, before any LM is created. It cannot reuse
`fastworkflow/utils/dspy_cache_utils.py`, which passes a parameter DSPy 3.2.1 no longer accepts (§2.2).

### 9.5 Compatibility

- Training, build, refine, and CLI entrypoints are unchanged.
- `dspy.inspect_history()` intentionally has no server-request history.
- FastWorkflow's bounded command traces and logs remain the server diagnostic surface.
- Verify the helper against the minimum allowed DSPy 3.x dependency as well as the installed 3.2.1
  before release. §9.3's runtime assertion protects deployments where that verification was not
  repeated; the two are complements, not alternatives `[R7]`.
- The policy propagates into `dspy.context(...)` blocks running in executor threads, which is the only
  shape this server uses. Any future change to `Settings.context`'s overlay order would break that,
  which is a further reason the assertion must run at startup rather than being assumed from this
  document.

---

## 10. Release B — runtime leases and turn lifecycle

Release B's correctness rests on knowing, without a race, whether anything is touching a runtime.
Revision 2 answered that with a boolean predicate. Round 2 showed a predicate is a sample.

### 10.1 The union predicate is necessary `[R2]`

`is_channel_busy` is wired directly to the registry (`run_fastapi_mcp/__main__.py:194`):

```python
session_manager.is_channel_busy = turn_registry.has_active
```

But `/invoke_agent_stream` (`__main__.py:920-1085`) runs an entire turn without ever creating a
`TurnExecution` — it guards with `runtime.lock.locked()` (`:963`) and holds `async with runtime.lock`
(`:974`). So `has_active` returns `False` for the whole streaming turn and the channel looks like a
valid victim. Eviction safety must therefore use at minimum:

```python
def _has_work_in_flight(channel_id: str) -> bool:
    if turn_registry.has_active(channel_id):
        return True
    runtime = self._sessions.get(channel_id)
    return runtime is not None and runtime.lock.locked()
```

Invariant 2 forbids `runtime.lock.locked()` as the **409 idempotency** truth source, and that
prohibition stands; eviction safety asks a different question, which the lock answers correctly and the
registry pointer does not.

### 10.2 The union predicate is not sufficient `[R2-2]`

`get_session()` returns a raw `ChannelRuntime` after releasing the manager lock
(`utils.py:745-750`). A normal endpoint then awaits registry admission; a streaming endpoint returns a
lazy `StreamingResponse` before acquiring `runtime.lock`. During either interval **both halves of the
union are false**:

1. Request A obtains runtime X and releases the manager lock.
2. Before A inserts a turn pointer or acquires X's lock, request B creates another channel.
3. The LRU samples the predicate, pops X, and closes it.
4. A continues using the detached X.
5. A cold recreation calls `Workflow.create()` while X still strongly owns the old app workflow. The
   weak global registry returns that same workflow object and **overwrites its context**
   (`workflow.py:101-104`), producing two runtimes and two locks around one mutable workflow.

Step 5 is the part that makes this more than a lost cache entry: single-writer-per-channel
(invariant 15) is violated by construction.

**Resolution.** A manager-owned **lease** (refcount) is acquired atomically with the manager-lock
lookup that returns the runtime, and released only after the work that lookup enabled has finished —
turn registration or lock acquisition takes over, and the lease persists until the executor exits.
Eviction requires a zero lease count in addition to the union predicate. `get_session()` returns a
leased handle, or an explicit `acquire`/`release` pair with a context manager; a raw runtime reference
that outlives the manager lock is the defect.

### 10.3 Creation must not evict its own runtime `[R2-1]`

`create_session()` inserts the new runtime and calls `_evict_oldest_if_needed()` before returning
(`utils.py:752-776`); `/initialize` submits startup only after creation returns
(`__main__.py:718-769`). Before startup, the new runtime has no registry pointer, its lock is free, and
its workflow has no root or current object.

Counterexample at the Release C default:

1. Fifty older sessions are pinned by object contexts.
2. Channel 51 is inserted and moved to the MRU end.
3. The LRU skips all fifty pinned candidates.
4. Channel 51 is the **only** apparently safe candidate, precisely because its startup has not yet
   created the object context that will pin it.
5. The manager evicts channel 51 inside its own `create_session()`.
6. `/initialize`'s second lookup fails and returns 500 (`__main__.py:735-740`). With no startup action
   it would instead return tokens for a runtime that no longer exists.

This is the negative observation §16.5 arm B misses by construction: below capacity, or with any older
evictable victim, creation works. The defect appears exactly when the new default begins to matter,
which is why it must be fixed in Release B and not discovered in Release C.

**Resolution.** Invariant 20: a runtime under construction holds an initialization lease from insertion
until startup admission or completion establishes final eligibility, and a leased runtime is never an
eviction candidate. Test with a cap-one manager whose single existing session is pinned, creating a
second channel both with and without a startup action.

### 10.4 Streaming joins the turn lifecycle `[R2-3]`

Adding the lock to the eviction predicate covers the *held-lock* window. It does not fix streaming's
position outside the turn lifecycle:

- `/invoke_agent_stream` captures the runtime before returning `StreamingResponse` and acquires the
  lock only when body iteration begins (`__main__.py:953-974`, `:1073-1085`).
- A normal endpoint can register a turn while a stream owns the lock, because the registry sees no
  active stream (`turns.py:195-214`).
- `run_process_message_with_trace_stream()` raises 504 at its own deadline **without awaiting or
  cancelling the executor future** (`utils.py:550-554`). The route catches that HTTP exception, emits an
  error, and exits the lock while the executor thread can still mutate WEC and app state.

Failure schedules:

1. A stream suspends on `ask_user` while an unrelated normal turn has queued behind it. When the stream
   releases the lock, the unrelated query is interpreted as the clarification answer.
2. A stream times out; its lock is released, the registry remains empty, and eviction or shutdown
   snapshots and closes the context while the detached executor thread is still running.
3. The runtime is evicted after response construction but before Starlette begins consuming the body
   generator.

**Resolution.** Route streaming through the same turn-admission and lifecycle owner as every other turn
(invariant 21). Client timeout or disconnect may stop *delivery*, but registry ownership and the runtime
lease remain until the executor exits. Tests: stream→normal, normal→stream, suspension, timeout,
disconnect, and eviction attempted before first body iteration.

### 10.5 Shutdown quiescence `[R14, R2-21]`

The shutdown drain must use the invariant-6 union predicate and the invariant-19 lease. Today it uses
neither: `_active_turn_channel_ids` tests `rt.lock.locked()` only (`__main__.py:285-286`), so a
`QUEUED` execution whose task has not yet acquired `runtime.lock` reports not-busy.

That fix alone is insufficient `[R2-21]`. Shutdown waits at most 30 seconds and then **always**
finalizes conversations and closes every runtime (`__main__.py:289-342`). After the deadline the union
can truthfully report remaining work while shutdown snapshots and closes it anyway — and with Release B
persistence added, it would write a snapshot taken before the queued turn mutates the context and then
close the context under it, making the stale snapshot authoritative on next creation. Detached streaming
executor work from §10.4 makes this worse. There is also no atomic "admission closed" state, so a turn
can be registered after an empty scan.

Shutdown therefore:

1. closes admission under registry and manager coordination, atomically with respect to submission;
2. drains queued, running and leased work;
3. after the deadline, **never** snapshots or closes a still-busy runtime — it logs at high severity and
   leaves the process to its host;
4. states explicitly whether queued-but-unstarted work is cancelled before execution or left to host
   termination.

### 10.6 Keep lifecycle operations distinct

- **Evict:** remove only the live cache entry after preserving declared durable state.
- **Remove/terminate:** keep existing explicit lifecycle semantics until a public channel-deletion
  contract is designed. Note for scope calibration `[R15]`: `remove_session` and `evict_live_session`
  have **no production callers** — the only caller in the tree is
  `tests/test_fastapi_topology_b.py:114`. They are preserved to avoid an unrelated behavior change, not
  because live contract surface depends on them.
- **Shutdown:** as §10.5.

---

## 11. Release B — the session checkpoint protocol

Revision 2 called this "supported workflow-state snapshot" and specified it as a cache detail. It is a
system of record (§1.3), and this section is rewritten accordingly.

### 11.1 Declared state, not inferred eligibility `[R2-5, R2-9, R2-12]`

Revision 2 inferred safety from three negative checks: no command-context object, no live child, and a
JSON-projectable `workflow.context`. Round 2 showed that is not a state model. Concrete state it omits:

- `Workflow.is_complete`, which is settable and included in the workflow's own `_to_dict()`
  (`workflow.py:282-289`, `:383-390`);
- `ChannelRuntime.active_conversation_id` and `stream_format` (`utils.py:644-658`);
- current conversation selection, which can differ from "latest conversation"
  (`__main__.py:1439-1473`);
- repeated mutable-container identity: `{"a": shared, "b": shared}` is acyclic and JSON-native but
  restores as two independent lists;
- stale child IDs: `_children` can outlive a weakly registered child, so testing `bool(_children)`
  either pins forever or ignores a genuinely live descendant.

The active-conversation omission is corrupting rather than cosmetic: suspend in older conversation 1,
evict, cold-create restoring the latest conversation 2, overlay conversation-1 pending history, resume,
and the turn is saved under conversation 2. SSE likewise silently becomes NDJSON after eviction.

Revision 2's persistence policy was also a silent data-classification change `[R2-12]`. It excluded one
exact top-level key, `http_bearer_token`, and otherwise persisted every JSON-native value from an
unrestricted `Workflow.context`, plus pending ReAct trajectory, inputs, action log, and conversation
turns, in plaintext. That permits `api_key`, refresh tokens, cookies, signed URLs, database
credentials, a nested `http_bearer_token`, and secrets supplied as an `ask_user` answer and retained in
the trajectory. Before this change, arbitrary application context could remain process-local.

**Resolution — one principle replaces all three findings.** Durable state is **workflow-declared**:

- A workflow declares a durable projection: which context keys are durable, which are ephemeral, and
  which runtime fields participate. The declaration is versioned and part of the workflow's contract.
- **Nothing is durable by default.** An undeclared workflow behaves exactly as it does today: its
  sessions are pinned, nothing is written, and no data-classification boundary moves.
- The framework enumerates the runtime and workflow fields it can restore — `is_complete`,
  `active_conversation_id`, `stream_format`, current conversation selection, current context name — and
  a field outside that enumeration pins the session rather than being silently dropped.
- Identity-preserving structures (shared mutable containers) either round-trip with identity preserved
  or pin.

Release C's bound is therefore a property a workflow opts into, which is the honest form of §1.4's
disclosure: instead of "most workflows happen to be pinned," the rule becomes "a workflow is bounded
when its author has said what is safe to persist."

### 11.2 Snapshot eligibility

A session is evictable only when all hold:

- the workflow declares a durable projection under §11.1;
- no root, current, or response-generation command-context object is live;
- no live child workflow state exists;
- the declared projection passes the strict encoding of §11.4;
- the session is not `awaiting_user` and holds no CME continuation state (§11.6);
- no lease is held and the union predicate is false (§10.1, §10.2).

Otherwise the runtime is pinned. Pinning is a first-class outcome with a metric (§17), not an error.

### 11.3 Record shape and identity binding `[R2-13, R2-18]`

```json
{
  "protocol_version": 3,
  "record_type": "channel_checkpoint",
  "generation": 41,
  "deployment_id": "…",
  "workflow_fingerprint": "…",
  "channel_key": "…",
  "channel_id_hash": "…",
  "session_incarnation": "…",
  "declaration_version": 2,
  "context": {},
  "runtime": {"active_conversation_id": 7, "stream_format": "ndjson", "is_complete": false},
  "startup": {"state": "succeeded", "idempotency_key": "…", "epoch": 3},
  "launch_context": {"prior_projection": {}, "digest": "…"}
}
```

Every identity field is validated on read; a mismatch quarantines rather than applies
(invariant 27). This is required because the current disk mapping is non-injective — `tenant/a` and
`tenant_a` collide, reproduced in §2.2 — and Redis keys carry no workflow or deployment namespace.
Revision 2 acknowledged the collision in a limitations list while making the record durable and
authoritative, which changes the blast radius of a pre-existing defect rather than inheriting it.

Concretely:

- `channel_key` uses an injective, collision-resistant encoding (for example percent-encoding or a hash
  with the raw ID retained inside the record), never `replace(os.sep, "_")`;
- keys are namespaced by deployment and workflow fingerprint;
- disk records are written with private directory and file modes and symlink-safe publication;
- `protocol_version` supports the fleet gate in §11.10.

### 11.4 One strict serializer, at the first boundary `[R2-7, R10]`

Revision 2 banned `default=str` in the store. The store is not the first serializer.
`WorkflowExecutionContext.serialize_state()` already returns
`json.loads(json.dumps(payload, default=str))` (`workflow_execution_context.py:365`), so unsupported
trajectory and artifact objects have become ordinary strings before any strict projector could see
them, and strict store validation then cheerfully accepts the lossy result. `DiskSessionStateStore.save`
applies `default=str` again (`session_state_store.py:61-64`). Revision 2's implementation map changed the
store and never touched the actual coercion boundary.

Resolution: **remove the WEC round-trip** and pass raw typed state through one strict, shared serializer.
That serializer:

1. requires a top-level dictionary;
2. includes only declared-durable keys (§11.1); ephemeral and undeclared keys are omitted or pin;
3. requires string dictionary keys recursively;
4. permits JSON-native scalar, list, and object values;
5. rejects cycles, unsupported values, non-finite floats, and identity-significant sharing;
6. encodes canonically with sorted keys, compact separators, and `allow_nan=False`;
7. computes SHA-256 over the canonical bytes.

Any failure keeps the runtime live (invariant 8). Tests must exercise this **through
`ctx.serialize_state()`**, not by calling the store helper directly, or they prove nothing — that is
precisely how revision 2's strictness requirement passed review while being unimplementable.

Do not include a changing `saved_at` field in the canonical payload used for digest comparison; it would
force a write on every retirement.

### 11.5 Startup: an explicit state machine with a commit point `[R2-4, R2-5]`

Revision 2 moved the startup-completion fact into the envelope `[R3]`, which was the right direction and
is retained. Two defects remain.

**There is no durable commit point** `[R2-4]`. Revision 2 simultaneously required that startup completion
be written when the turn completes, that workflow context be written only at retirement or shutdown
(invariant 12, decision 8), and that the write be skipped when the context digest is unchanged. Those
cannot all hold. A startup that performs an external side effect or read-only initialization without
mutating `workflow.context` completes successfully, changes only in-memory `startup.completed`, leaves the
digest unchanged, has no retirement yet, and is skipped by the no-write path. A crash then loses the fact
and restart replays startup. Current completion persists conversation and pending state before `DONE`
(`turns.py:302-318`) with no context-envelope commit, and revision 2 added none to that ordering.

**The record cannot express the states that occur** `[R2-5]`. `{"idempotency_key": null, "completed":
false}` cannot represent attempted, suspended, succeeded, command-level failure, executor exception,
partial mutation, or whether a result remains collectable. `_run_turn` transitions to `DONE` even after an
exception (`turns.py:307-318`), and `TurnStatus.COMPLETED` is orthogonal to command success.

Resolution:

- **Explicit states:** `not_attempted`, `in_progress`, `suspended`, `succeeded`, `failed`, with a stated
  success predicate that does not equate `DONE` with success, and an `epoch` so a deliberate reset is
  expressible.
- **Commit point:** the semantic-envelope commit is synchronous and completes **before** startup success
  becomes observable (invariant 25). It digests the complete envelope, not just `context`, so a
  metadata-only change still writes. Store failure at this point has a defined response and retry policy
  rather than being silently skipped.
- **Scope of the guarantee:** exactly-once applies to framework-managed effects only. External side
  effects require application-owned idempotency, stated as a non-goal in §3.2.

**The object-context contradiction is resolved by §11.1, not by the envelope** `[R2-5]`. For
`simple_workflow_template`, startup creates `root_command_context`; §11.2 declares that unsnapshottable
and pins the runtime. If shutdown wrote only `startup.succeeded`, restart would suppress startup while
being unable to reconstruct the root object, and the workflow would resume without required state; if
shutdown refused the metadata write, restart would rerun startup and contradict invariant 18. Under
§11.1 neither arises: an undeclared workflow is pinned, is never checkpointed, writes no startup record,
and reruns startup on a genuinely fresh runtime exactly as today.

Tests: mutate-then-raise, `success=False`, suspension during startup, a read-only side effect, process
death after command completion but before terminal publication, store failure at the commit point, and
restart of `simple_workflow_template`.

### 11.6 Continuation state: complete it or pin `[R2-8]`

The existing pending snapshot stores suspension flags, ReAct state, NLU stage, action log, and
conversation turns. It does **not** store the WEC logical-turn accumulator initialized at
`workflow_execution_context.py:104-114`: `_turn_outputs`, `_turn_key`, `_turn_started_at`, the original
and refined message, suspended duration and suspension start, entry workflow and context, and the agent
result. Resume deliberately skips `_begin_turn()` (`:547-549`) precisely because it expects the old
accumulator, so after rehydration the logical turn takes a new fallback key and loses pre-suspension
command outputs, ask-user entry, artifacts, and timing.

The deterministic CME continuation is also incomplete: `serialize_state()` stores only `nlu_stage`, while
clarification and parameter extraction depend on the CME context keys `command`, `command_name`, and
`stored_parameters`
(`fastworkflow/_workflows/command_metadata_extraction/_commands/wildcard.py:45`, `:141-144`;
`fastworkflow/_workflows/command_metadata_extraction/parameter_extraction.py:72`, `:155-165`).
`Workflow.end_command_processing()` deletes `command` and `stored_parameters` and resets the stage
(`workflow.py:291-303`), so a mid-extraction snapshot that omits them cannot be distinguished from a
completed one after restore.

This is a **pre-existing defect** in today's suspended-state restore, not something Release B introduces —
but Release C makes the path routine, so Release B must resolve it.

Release B v1 resolution: **pin awaiting and continuation sessions.** A session that is `awaiting_user` or
holds CME continuation state is not evictable (§11.2). Completing the serialization is the tracked
follow-up, not the v1 requirement, because getting the typed logical-turn and CME records right is a
larger change than the memory bound needs.

Two consequences must be stated `[R2-10 · absorption]`:

- Pinning removes the controlled-eviction half of the two-record consistency problem (§11.8) but not the
  crash half.
- Pinning does **not** help multi-pod deployments: another pod still cold-rehydrates from the store, so
  the incomplete restore remains reachable. Redis-backed multi-pod deployments therefore need the
  completeness work before they can rely on suspension across pods, which is true today and is not made
  true by this design.

Tests when the completeness work lands: exact `TurnOutput` equivalence across
suspend→evict→rehydrate→resume, and missing-parameter→evict→answer.

### 11.7 Request-scoped credentials `[R2-14]`

Revision 2 proposed to fix today's no-op updater by having `_update_http_bearer_token()` mutate
`app_workflow.context` directly. The call occurs in `ensure_user_runtime_exists()` **before** endpoint
turn admission (`utils.py:286-304`), so:

1. Turn A runs under `runtime.lock` using token A.
2. Request B for the same channel reaches the dependency with token B.
3. The dependency writes token B into shared workflow context.
4. B is later rejected with 409 — but A can already read B.

The fix would have repaired a no-op by introducing cross-request credential contamination.

**Resolution, constrained by a contract the review did not have** `[R2-14 · absorption]`.
`fastworkflow/run_fastapi_mcp/README.md:86-103` documents `workflow_context['http_bearer_token']` as the
supported way for a workflow to read the caller's token, so the key cannot simply be removed without
breaking every authenticated workflow (goal 5). Therefore:

- the token is carried on the submitted execution (or a request/turn `ContextVar`), never written by a
  dependency;
- it is installed into `workflow.context` under the **accepted** turn's lifecycle, inside `runtime.lock`
  and immediately before executor dispatch, and cleared or restored after the turn;
- a rejected, deferred, or streaming-guarded request never mutates shared state;
- the key is ephemeral by declaration (§11.1) and never reaches a durable record — including nested
  occurrences, which §16.3 tests by inspecting raw stored bytes rather than the top-level key.

Barrier test: B is rejected with 409 while A reads the credential after B's dependency ran; A must see
token A.

### 11.8 One state generation, one commit `[R2-10]`

Revision 2 wrote the context record and the pending record as separate operations and called the pair
transactional. `os.replace()` makes one disk file atomic; Redis uses independent `SET`s. Neither creates
one transaction across two records. Crash schedule:

1. A tool mutates application context and suspends on `ask_user`.
2. Context generation N+1 is published.
3. Pending publication fails or the process dies.
4. Restore combines context N+1 with pending N, or with none.

The reverse ordering is also unsafe: a durable pending trajectory can resume against rolled-back
application state and repeat effects. "Persist before pop" protects controlled eviction only.

Resolution:

- a monotonic per-channel `generation` stamped on every record that participates in a checkpoint;
- a commit record that names the generation considered complete;
- restore requires all participating records to match the committed generation, and on mismatch **fails
  closed**: the channel is quarantined, its records are preserved for inspection, it starts from launch
  configuration, and a WARNING plus a metric is emitted (never a silent merge);
- Redis multi-key atomicity requires same-slot key design; disk requires a manifest or directory-level
  generation protocol.

With §11.6's pinning, the eviction path never has both records live for one channel, which materially
narrows the exposure; the crash path is what this machinery is for. Fault-inject every boundary and
reconstruct in a fresh process.

### 11.9 Restore, reconciliation, and launch-time configuration `[R6, R2-6]`

Absence and explicitly saved empty context are different:

```text
saved record absent:
    durable context = configured initial context

saved record present, including {"context": {}}:
    durable context = saved context

then:
    reconcile changed launch-time keys (below)
    install request-scoped credentials under the turn (§11.7)
```

Do not merge launch-time application context over saved state wholesale; that would overwrite mutations
and resurrect keys the application deleted.

Round 1 established that launch-time context is **operator configuration**, not stale application state
`[R6]`: `InitializationRequest` has no `context` field (`utils.py:28-37`) and the value comes from a
process-wide CLI argument (`__main__.py:241`, `:724`), so `--context` carries API base URLs, tenant
identifiers, and feature flags. An operator who changed `--context` and redeployed would otherwise find
the change silently ignored for every channel with a snapshot.

**Round 2 showed revision 2's mechanism was information-theoretically unimplementable** `[R2-6]`. The
envelope stored a single aggregate `launch_context_digest` while the rule required overlaying "only those
launch-time keys whose values differ from the launch-time values recorded when the snapshot was written."
Given:

```text
old launch = {url: A, tenant: T1}
saved app  = {url: B, tenant: T1}
new launch = {url: C}          # tenant intentionally removed
```

a digest can say *something* changed. It cannot identify `url`, know that `tenant` was removed, or
distinguish the application's A→B mutation from an operator A→C change.

Resolution: store the **prior canonical launch projection** itself (`launch_context.prior_projection`),
not only its digest, and define a true three-way merge over
`(prior launch, saved application, current launch)` with explicit add, change, delete, and conflict
semantics:

- key added to launch since the snapshot → apply;
- key changed in launch since the snapshot → apply, and log at WARNING naming the channel and key;
- key removed from launch since the snapshot → remove, unless the application has since written it;
- key unchanged in launch but changed in the saved application state → keep the application's value;
- key changed in both → conflict: operator value wins and the conflict is logged and counted.

The digest is retained as a fast-path equality check so the common case costs one hash.

**The cleaner alternative is recorded and preferred long-term** `[R2-6]`: separate operator-owned
configuration from mutable application context so provenance is structural rather than inferred from one
flat dictionary. It is deferred only because it changes `Workflow.context`'s shape, which is a wider
framework change than this design should carry.

Also fix `_update_http_bearer_token()`'s addressing defect per §11.7; its current `get_active_workflow()`
lookup occurs outside a turn and can return no workflow, making refresh a no-op.

### 11.10 Namespace lifecycle, growth, and fleet version `[R5, R2-17, R2-18]`

**Durable growth is quantified and gated, not monitored.** The motivating workload is
unique-channel-per-request at ≈65 requests/hour and 450 KB, so every request would cold-create, evict, and
write a ~450 KB snapshot **that is never read again**:

| Quantity | Value |
|---|---|
| Requests/day at the observed rate | ≈1,560 |
| Context-only durable growth/day at 450 KB/record | ≈700 MB/day, unbounded |
| Conversation records | additional, and can duplicate request-sized data |
| Reaper | none |
| TTL | none, and this design forbids a hidden one on principle |

Revision 2 claimed its §13.5 "now carries a durable-growth acceptance number." It did not `[R2-17]`: the gate
said only "does not exceed the pre-registered figure," with no figure and no plateau criterion, and any
positive bytes-per-1,000-requests satisfies a rate while cumulative storage grows forever. §16.5 now
gates **total physical bytes per namespace against a steady-state plateau**, not a rate.

**The rollout dependency was also mis-scoped** `[R2-17]`. `fix-6b4` is "Add TTL/reaper for orphaned
suspended-session blobs" and covers the pending namespace and abandoned `ask_user` state; it does not
define lifecycle for checkpoint or conversation records. Release C is therefore gated on **`fix-jtr`**,
opened for this namespace, which must deliver TTL or reaper, inspect, quarantine, delete, reset, and
generation-safe channel reuse.

A future revision should record whether a channel has ever been revisited, so single-shot channels can be
excluded from checkpointing entirely. This design has no mechanism to express that, which is itself worth
stating: for the exact workload that motivated the change, the persistence machinery is pure cost at zero
benefit.

**Mixed-version rollout** `[R2-18]`. Schema evolution was called "forward-only" while §17 treated raising
`MAX_LIVE_SESSIONS` as rollback — that is tuning, not binary compatibility. Old `2.24.1` code does not read
the checkpoint namespace on cold creation, can mutate application state while leaving the snapshot
untouched, and only warns on pending-state schema mismatch before continuing to apply fields
(`workflow_execution_context.py:367-400`). Sequence: new code writes generation A; old code receives the
channel, ignores A, starts from launch context, produces state B; new code later restores stale A as
authoritative. Rolling upgrades sharing Redis have the same problem.

Therefore: a fleet protocol marker and a declared version floor; either reject mixed versions or provide
dual read/write migration; and an explicit statement that **downgrade after the first checkpoint is
unsupported**. Tests: new→old→new, concurrent N−1 and new readers and writers, raw source-study blobs, and
rollback after a real eviction.

### 11.11 Transactional retirement

While holding the manager retirement critical section, after confirming zero leases and a false union
predicate:

1. verify snapshot eligibility (§11.2);
2. serialize the declared projection through the single strict serializer (§11.4);
3. skip the write if the envelope digest is unchanged;
4. otherwise publish the composite generation atomically (§11.8);
5. only after required writes succeed, pop the runtime;
6. close and release the runtime.

On serialization or store failure: leave the runtime in `_sessions`, try another candidate, warn with
channel ID, backend, and exception class, and never log payloads, context values, or credentials.
Popping first and treating persistence as best effort is rejected: it loses state on the exact path where
preservation failed.

`save_serialized()` writes a sibling temporary file and publishes with `os.replace()` so a crash leaves
either the old complete record or the new complete record, never truncated JSON. The same primitive may
improve existing pending-state writes without changing their interface.

---

## 12. Release C — live-session cache target

`_sessions` remains an `OrderedDict`; successful lookup or creation moves a channel to the MRU end. The
default becomes 50 via the §5.1 resolver.

When over target, inspect candidates oldest-first:

1. skip any channel with a non-zero lease (§10.2) or an initialization lease (§10.3);
2. skip any channel with work in flight, using the union predicate (§10.1);
3. ask §11.2 whether the session is checkpointable;
4. transactionally persist per §11.11;
5. remove the first successfully persisted victim;
6. continue until at or below target or no safe candidate remains.

If every candidate is leased, busy, or pinned, remain over target and emit one rate-limited warning with
counts and reasons. Never discard state to satisfy the number. For workflows that do not declare a durable
projection this is the steady state rather than a transient (§1.4), which is why §17 makes pinned count a
metric.

Re-run trimming after session creation, and after an execution clears its active pointer **once the
registry lock has been released** (§6.2 step 5, invariant 17).

This is cache reclamation, not admission control. Peak active channels may exceed 50.

---

## 13. Request and eviction flows

### 13.1 Live-session hit

1. Look up the runtime and acquire a lease atomically (§10.2).
2. Move it to the MRU end.
3. Execute under existing per-channel turn rules; install request credentials under the accepted turn
   (§11.7).
4. Persist conversation state incrementally by append (§8.2) and suspension state as today.
5. Perform no checkpoint serialization.
6. Release the lease after the executor exits.

Only terminal-registry cleanup and the conversation window are added when a turn completes.

### 13.2 Cold creation

1. Acquire the weakly held per-channel creation lock.
2. Re-check the live cache.
3. Load and validate the checkpoint record: identity binding (§11.3), committed generation (§11.8),
   protocol version (§11.10). Quarantine on any mismatch.
4. Use the saved declared projection when present; otherwise use configured initial context.
5. Three-way reconcile launch-time keys against `launch_context.prior_projection` (§11.9).
6. Create and bind WEC and app workflow; restore enumerated runtime fields including conversation
   selection and stream format (§11.1).
7. Restore conversation and continuation state, subject to §11.6.
8. Register the runtime **holding an initialization lease** (§10.3), then enforce the LRU target.
9. For startup, consult the envelope's startup state machine; submit only if it does not already record
   the same idempotency key as succeeded for the current epoch (§11.5).
10. Release the initialization lease once startup admission or completion establishes eligibility.

### 13.3 LRU eviction

1. Select the oldest candidate with zero leases and a false union predicate (§10.1, §10.2, §10.3).
2. Verify checkpoint eligibility (§11.2).
3. Encode the declared projection once through the single strict serializer (§11.4).
4. Skip the write if the envelope digest is unchanged.
5. Publish the composite generation atomically (§11.8).
6. Pop and close only after successful publication.
7. Try another candidate on candidate-specific failure.

---

## 14. Performance model

Ignoring active work and allocator high-water behavior:

```text
retained memory
  ≈ N_evictable_live × average live-runtime footprint     (N ≤ 50 only where §11.2 allows eviction)
  + N_pinned × average live-runtime footprint             (UNBOUNDED for undeclared/object workflows)
  + 20 × average retained-startup-execution footprint
  + N_live × min(conversation turns, 20) × turn footprint
  + bounded DSPy response cache
  + fixed framework/model caches
```

The bound is count-based, not byte-based. At the representative 450 KB payload:

| Structure | Cap | Bytes at 450 KB | Release |
|---|---:|---:|---|
| Retained startup executions | 20 | ≈17 MB | A |
| In-memory conversation turns, per channel | 20 | ≈9 MB | A |
| DSPy response cache | 200 entries | ≈90 MB worst case, to be measured | A |
| Live sessions, declared workflows | 50 | ≈23 MB | C |
| Live sessions, undeclared/object workflows | none `[R1]` | unbounded | — |

Expected overhead:

- terminal cleanup sorts at most 20 retained records;
- the conversation window is a slice on a list already being written;
- LRU victim search examines roughly 50 cached sessions — but for pinned workflows it examines *every*
  cached session on every trim and finds nothing, so the search is O(live sessions) with no upper bound
  `[R1]`;
- weak creation-lock cleanup uses reference counting and GC;
- checkpoint encoding, hashing, and I/O occurs only on eviction and shutdown, plus the §11.5 metadata
  commit at startup completion;
- an unchanged envelope performs no store write;
- three-way launch reconciliation costs one hash on the common path and a key-set diff otherwise `[R6]`;
- DSPy controls remove append and pop work; the bounded cache adds LRU eviction work.

**The deferral of asynchronous retirement needs a trigger that can actually fire** `[R8, R2-22]`. The I/O
runs inside the manager critical section: `_evict_oldest_if_needed` is awaited from `create_session` while
holding `self._lock` (`utils.py:762`, `:775`), and `get_session` needs the same lock (`:746`). A ~450 KB
encode plus SHA-256 plus store write therefore blocks the event loop and every other channel's session
lookup. In the unique-channel workload that is **every request**. §16.6 gates the evict-every-request
configuration and an event-loop scheduling-delay measure, with absolute numeric limits rather than
placeholders. If those are exceeded, add a per-channel `RETIRING` reservation before moving I/O outside the
manager lock; do not release the lock and allow a second runtime to appear without such coordination.
Invariant 17 separately forbids holding `registry._lock` across the same I/O.

---

## 15. Implementation map

**Release A**

- `fastworkflow/run_fastapi_mcp/turns.py`
  - startup-terminal retention constants;
  - TTL and count pruning, inside `clear_active`'s existing lock block `[R13]`;
  - task-launch rollback;
  - post-turn work scheduled **after** the registry lock is released `[R9]`;
  - constructor injection for deterministic tests.
- `fastworkflow/run_fastapi_mcp/utils.py`
  - weak creation-lock mapping;
  - `save_conversation_incremental` appends above a high-water mark `[R2-11]`.
- `fastworkflow/run_fastapi_mcp/conversation_store.py`
  - `append_conversation_turns`, replacing full-list rewrite `[R2-11 · absorption]`.
- `fastworkflow/workflow_execution_context.py`
  - in-memory conversation window after durable record `[R2-11]`.
- `fastworkflow/run_fastapi_mcp/__main__.py`
  - topic/summary generation reads the durable record, not windowed memory `[R2-11]`;
  - `/activate_conversation` sets the high-water mark `[R2-11]`;
  - install DSPy policy and cache policy before event-loop startup, assert both, claim async ownership
    `[R7, R2-15, R2-19]`.
- `fastworkflow/run_fastapi_mcp/server_memory.py` (new, focused module)
  - server-only DSPy history/trace/cache policy helper;
  - the §9.3 startup structural assertion and the post-readiness ownership guard `[R7, R2-19]`.

**Release B**

- `fastworkflow/run_fastapi_mcp/utils.py`
  - manager-owned runtime leases and leased lookup `[R2-2]`;
  - initialization lease across `create_session` `[R2-1]`;
  - union work-in-flight predicate `[R2]`;
  - checkpoint load, validate, restore, digest;
  - three-way launch reconciliation `[R6, R2-6]`;
  - transactional retirement over one generation `[R2-10]`;
  - credential install under the accepted turn `[R2-14]`.
- `fastworkflow/workflow_execution_context.py`
  - remove the `default=str` round-trip; single strict serializer at this boundary `[R2-7]`;
  - declared-projection API and enumerated restorable fields `[R2-9, R2-12]`;
  - complete logical-turn and CME continuation records, or pin `[R2-8]`.
- `fastworkflow/session_state_store.py`
  - `key_prefix` on the factory `[R11]`;
  - injective channel-key encoding and identity binding `[R2-13]`;
  - generation stamp and commit record `[R2-10]`;
  - `save_serialized` with atomic, symlink-safe, private-mode publication;
  - strict encoding for the pending path too `[R10]`.
- `fastworkflow/run_fastapi_mcp/__main__.py`
  - streaming routed through turn admission `[R2-3]`;
  - shutdown closes admission and never writes past its deadline `[R14, R2-21]`;
  - envelope-driven startup state machine `[R3, R2-4, R2-5]`;
  - fleet version gate `[R2-18]`.

**Release C**

- `fastworkflow/run_fastapi_mcp/utils.py`
  - OS-first `MAX_LIVE_SESSIONS` resolver logging value and source `[R4, R2-20]`;
  - default 2,000 → 50.
- `fastworkflow/examples/fastworkflow.env`
  - document `MAX_LIVE_SESSIONS` as a **commented** line `[R2-20]`.

Do not put memory policy into commands or endpoint response models.

### 15.1 Landing order

**Release A**

1. conversation durable append and high-water mark, then the in-memory window, then topic/summary and
   `/activate_conversation` follow-ups `[R2-11]`;
2. terminal registry bounds and task-launch rollback;
3. weak creation locks;
4. DSPy history, trace, and cache policy plus the startup assertion and ownership guard
   `[R7, R2-15, R2-19]`;
5. focused tests, full suite twice, soak arm A0 and the latency matrix.

**Release B**

6. leases: lookup lease, initialization lease, union predicate `[R2-1, R2-2]`;
7. streaming lifecycle unification and shutdown quiescence `[R2-3, R2-21]`;
8. single strict serializer at the WEC boundary, removing the `default=str` round-trip `[R2-7]`;
9. declared-projection capability and enumerated restorable fields `[R2-9, R2-12]`;
10. record identity, generation, commit record, atomic publication `[R2-10, R2-13]`;
11. startup state machine and commit point `[R2-4, R2-5]`;
12. three-way launch reconciliation `[R2-6]`;
13. credential lifecycle `[R2-14]`;
14. namespace lifecycle (`fix-jtr`) and fleet gate `[R2-17, R2-18]`.

**Release C**

15. resolver and default change, only after the full Release B verification matrix passes.

Steps 6 and 7 must precede any cap reduction: lowering the target while streaming turns are invisible to
the eviction path is the R2/R2-3 defect, and step 15 is what makes it reachable.

**R1 is a decision, not a step.** Decision 18 must be resolved before Release B code lands, because it
changes what §1 promises rather than how §1 is implemented.

---

## 16. Verification plan

### 16.1 Deterministic turn tests (Release A)

- More than `MAX_RETAINED_STARTUP_TURNS` completed startup executions retains exactly the newest 20.
- Expired retained startups are removed.
- Non-startup terminal executions are removed during finalization.
- Running and queued executions survive age and count sweeps.
- Long-running work receives TTL only after completion.
- Task-launch failure rolls back both registry pointers.
- Ordinary identical completed turns are not deduplicated.
- No store I/O occurs while `TurnRegistry._lock` is held `[R9]`.
- Existing single-flight and 409 tests remain unchanged.

### 16.2 Conversation-bound tests (Release A) `[R2-11]`

- A durable conversation retains **all** turns after the in-memory window has trimmed older ones. This
  test must fail against a naive window that trims before append.
- Incremental save writes only new turns; total bytes written over N turns is O(N), not O(N²).
- Topic and summary generation at shutdown sees the full durable conversation, not the window.
- `/activate_conversation` restores the window and sets the high-water mark so the next save does not
  re-append the conversation.
- Same-channel in-memory history bytes plateau across several hundred request-sized turns.

### 16.3 State and store tests (Release B)

Use a real small workflow and direct actions so no trained model is required:

- no-eviction control; LRU eviction and rehydration; unchanged revisit produces no second write; nested
  in-place mutation produces one new write; empty context persists `{}` and does not resurrect keys.
- **Three-way launch reconciliation** `[R2-6]`: with a snapshot in place, exercise key added, key changed,
  key removed, application-mutated key, and both-changed conflict; assert each outcome and that a WARNING
  names the reconciled keys. A digest-only implementation cannot pass this test, which is the point.
- **Strictness through the real boundary** `[R2-7]`: drive an opaque object, a cycle, a non-finite float,
  and a nested artifact through `ctx.serialize_state()` — not through the store helper — and assert the
  runtime stays live. A test that calls the new store helper directly proves nothing.
- **Identity binding** `[R2-13]`: separator aliases (`tenant/a` vs `tenant_a`), Unicode, control
  characters, oversized IDs, deliberately swapped records, two workflows sharing one Redis, and channel-ID
  reuse across incarnations. Assert quarantine, not silent application.
- **Generation atomicity** `[R2-10]`: fault-inject at every write boundary, then reconstruct in a fresh
  process; a partial generation must fail closed rather than combine.
- **Credentials at rest** `[R2-12, R2-14]`: inspect raw disk and Redis bytes for a nested
  `http_bearer_token`, an `api_key` in application context, and a secret supplied as an `ask_user` answer
  and carried in the trajectory. Top-level-key assertions are insufficient. Plus the §11.7 barrier test in
  which B is 409-rejected while A reads its own token.
- **Startup semantics** `[R2-4, R2-5]`: no-context-mutation startup survives process death; mutate-then-raise;
  `success=False`; suspension during startup; store failure at the commit point; restart of
  `simple_workflow_template`.
- **Pinning** `[R2-8, R2-9]`: awaiting session, CME continuation, object context, undeclared workflow,
  shared-container identity, `is_complete`, conversation selection, and stream format each pin or restore
  exactly; none is silently dropped.
- Disk failure preserves the live runtime and the prior complete record.
- Redis round-trip when available, asserting that record kinds do not collide for one `channel_id` and
  that a normal turn's pending `clear()` does not delete the checkpoint `[R11]`.

Stub only `dspy.LM.forward` where an LLM call is required; the real `BaseLM` history path and
`Predict._forward_postprocess` must execute or the DSPy test is vacuous. The stub must return a real
`litellm` `ModelResponse` — a plain dict raises inside `BaseLM._process_completion` on `response.choices`
`[R7]`. For cache assertions the patch must go **below** `LM.forward` and must be a real function carrying
`__module__` and `__qualname__`, because the cache wrapper computes `f"{fn.__module__}.{fn.__qualname__}"`
and a `Mock` raises `AttributeError` before the cache is reached `[R2-15 · absorption]`.

### 16.4 Deterministic manager and DSPy structural tests

Manager (Release B):

- LRU never evicts a channel with a live registry execution.
- **LRU never evicts a channel whose `runtime.lock` is held but which has no registry execution** — the
  `/invoke_agent_stream` shape `[R2]`. Must fail against the registry-only predicate.
- **LRU never evicts a leased runtime between lookup and admission** `[R2-2]`, with barriers for the normal
  and streaming intervals. Must fail against a predicate-only implementation.
- **A cap-one manager whose single session is pinned does not evict the channel it is creating**, with and
  without a startup action `[R2-1]`. Must fail against revision 2's ordering.
- Streaming: stream→normal, normal→stream, suspension, timeout, disconnect, and eviction attempted before
  first body iteration `[R2-3]`.
- An all-busy burst may exceed target, then converges after turns finish.
- Creation locks disappear after successful and failed creation; concurrent cold requests create one
  runtime.
- Undeclared/object-context sessions stay live **and the manager reports the pinned count** `[R1]`.
- Serialization or store failure leaves the candidate live, for context and continuation failures alike
  `[R10]`.
- Shutdown: queued and running work beyond a short deadline, a detached stream worker, and a submission
  racing the final empty scan; assert no snapshot or close of still-busy runtimes past the deadline
  `[R2-21]`.

DSPy structural (Release A), in an isolated subprocess invoked as `.venv/bin/python` because
configuration ownership is process-global and the ambient interpreter may carry DSPy 2.x `[R15]`:

1. make a real `Predict` call with default settings;
2. prove global history, LM history, and trace each gain an entry;
3. start a fresh server-policy subprocess;
4. make unique calls;
5. prove all three structures remain empty;
6. exercise repeated app imports and lifespans without a second owner-task configure call;
7. assert the installed DSPy actually *reads* both keys — the same assertion §9.3 runs at startup, so the
   test and the runtime guarantee cannot drift `[R7]`;
8. **assert a post-readiness async `configure()` is rejected**, and that the readiness probe reports
   unready if the policy has drifted `[R2-19]`;
9. **assert response-cache entries stay at or below `SERVER_DSPY_MEMORY_CACHE_ENTRIES` and that the disk
   cache is not written**, patching below `LM.forward` `[R2-15]`.

Steps 1–5 must run in the **production call shape** — `Predict` inside `dspy.context(lm=…, adapter=…)`
inside a `ThreadPoolExecutor` worker — because that is the only shape this server uses. A test that
configures and calls on the main thread would pass even if `Settings.context` did not inherit
`main_thread_config` `[R7]`.

### 16.5 RSS soak

**Run the real server** `[R2-22]`. Revision 2's in-process ASGI harness put client and server in one
process and could bypass the dedicated pre-event-loop entrypoint where the DSPy controls must be
installed — that is, it could not observe the thing it was meant to gate. The soak therefore launches the
actual CLI/Uvicorn server in a separate fresh process and drives it over HTTP.

Method:

- strictly sequential requests; unique channel and unique ~450 KB payload per request; direct startup
  action; one warm-up request;
- both natural-GC and forced-`gc.collect()` sample series, reported separately, because forced collection
  can hide production allocator behavior;
- several hundred measured requests minimum; second-half least-squares slope;
- record RSS **and USS and cgroup memory**, plus live-heap diagnostics;
- record session, lock, retained-turn, DSPy history, DSPy trace, **DSPy cache entries and bytes**,
  **in-memory conversation bytes**, and durable-store record counts and bytes;
- **multiple fresh-process replicates**, reported with an upper confidence bound rather than a single
  slope.

Capture unpatched baseline and treatment with identical Python, lockfile, payloads, sample count, warm-up,
host, and replicate count.

Required arms:

| Arm | Fixture | Release | Pre-registered expectation |
|---|---|---|---|
| **A0 — Release A only** `[R2-11, R2-22]` | function-style workflow, cache limit unchanged at 2,000 | A | Slope materially below baseline but **above** the 0.05 MB/request target, because the session cache still retains every unique channel (§1.2). Recorded, not gated. Falsifying this re-scopes Release C. |
| **A1 — hot single channel** `[R2-11]` | one channel, several hundred request-sized direct actions | A | In-memory conversation bytes plateau; slope at or below target. This arm fails without §8 and is the only arm that tests goal 1 on this workload shape. |
| **A — evictable** | function-style workflow, no command-context object | C | Slope ≤ 0.05 MB/request; all caps held. |
| **B — pinned** `[R1]` | workflow assigning `root_command_context` **in startup**, i.e. the `simple_workflow_template` shape | C | Slope materially above target; live sessions grow without bound; pinned count rises monotonically. Expected to "fail" the slope gate; its purpose is to **quantify** §1.4 so the limitation ships with a number. |
| **C — streaming** `[R2-3]` | arm A plus concurrent `/invoke_agent_stream` turns spanning more than `MAX_LIVE_SESSIONS` subsequent creations | C | Zero context-loss sentinels, zero torn snapshots, no streaming turn's context closed, no detached executor writes. |

Shipping targets:

- **Release A** gates on arm A1 (plateau, slope ≤ 0.05 MB/request) and on all Release A caps holding in
  A0. Arm A0's slope is recorded and quoted verbatim in §1.2 and the release notes, not gated.
- **Release C** gates on arm A and arm C; arm B is recorded and quoted verbatim in §1.4.
- All structurally bounded collections stay at or below their caps; no declared state is lost; all
  requests succeed; the context-loss sentinel never appears in arms A, A1 or C.
- **Durable storage reaches a plateau.** Total physical bytes per namespace must stop growing under a
  stated retention policy `[R2-17]`. A bytes-per-1,000-requests rate is not a bound: any positive rate
  grows forever, and ≈700 MB/day is what the motivating workload would produce (§11.10).
- The slope target itself is a **leak budget, not a plateau** `[R2-22]`: 0.05 MB/request permits ≈78 MB/day
  at 65 requests/hour. It is retained as a screening threshold, but the acceptance claim is the plateau —
  a slope whose upper confidence bound is compatible with zero, or with an explicitly stated
  worker-recycle budget.

Report raw samples and slopes. Do not replace them with "survived N requests," and do not sum ablation
deltas as independent shares: the source study's own arms sum to ≈2.70 MB/request against a
1.76 MB/request baseline.

#### 16.5.2 Measured results — Release C

Same harness and method as §16.5.1. `MAX_LIVE_SESSIONS` default 50.

| Arm | Requests × replicates | Slope (upper 95% bound) | Verdict |
|---|---|---|---|
| A — evictable, unique channel | 300 × 3 | **+0.01127** | **PASS** (gate ≤ 0.05) |
| C — streaming | 700 × 3 | **+0.01808** | **PASS** |
| C — streaming | 300 × 3 | +0.05746 | measurement artifact, see below |
| B — pinned (no `get_state` hook) | 300 × 3 | +0.49769 | recorded, not gated |

**Arm A passes, and this is the release's headline.** The unique-channel workload
that motivated the whole design measured +1.339 MB/request unpatched and +0.495
after Release A; with the cap at 50 and eviction writing a checkpoint it is
**+0.011**. 251 of 301 channels were retired per replicate, zero over-target
warnings, zero context-loss sentinels, and every structural cap held.

**Arm C passes on all three of its criteria** — streaming invariants, structural
caps, and slope. No streaming channel was retired mid-turn across 705 retirements
per replicate.

**Arm C's 300-request "failure" was the §16.5.1 estimator artifact, reproduced.**
+0.05746 at 300 requests became +0.01808 at 700 on the same code. This is the
second independent confirmation that 300 requests is below this estimator's
resolution; **size gated arms at 500 requests minimum**, as §16.5.1 already
concluded from arm A1.

**The durable-storage plateau gate is met.** §16.5 gates Release C on total
physical bytes per namespace reaching a plateau, and rejects any positive rate as
a bound. At 3,500 requests — long enough to span two 300 s reap intervals and to
exceed `RetentionPolicy.max_channels`, which shorter runs are not — the namespace
settled at **56.0 MB physical / 5.3 MB apparent across 1,592 channels**, with
three observed byte *decreases* as reclamation fired: **plateau observed**, not a
slower rate of growth. The store's own `stats()` agreed with the harness's
independent walk of the files. Slope on that run was +0.00865 with 3,451 of 3,501
channels retired and zero context-loss sentinels.

Shorter runs cannot show this and should not be read as showing its absence: at
300 requests the run spans zero reap intervals and puts fewer channels on disk
than the count cap, so no pass could reclaim anything even if one fired. The
harness reports that as a measurement-window result rather than a finding.

**Arm B quantifies §1.4, and its meaning has changed.** A workflow whose context
class has no `get_state` hook grows to 301 live sessions against a cap of 50 with
zero retirements, 251 over-target warnings, a monotonically rising pinned count,
and a slope of +0.498 — i.e. the cap does nothing for it, exactly as predicted.
What changed is the census: `simple_workflow_template` and the other four bundled
workflows now implement the hooks and are evictable, so "pinned" no longer means
"this workflow shape cannot be bounded". It means **its author has not written
`get_state` yet**, which is a state the author controls and the server warns
about. Arm B's fixture is therefore a deliberately un-hooked copy of
`tests/todo_list_workflow`, not `simple_workflow_template`.

#### 16.5.1 Measured results — Release A

Harness: `tests/soak/memory_soak.py`. A real Uvicorn server in its own process per replicate, driven
over HTTP, sequential requests, unique 450 KB payload per request, one excluded warm-up, per-request
RSS/USS, second-half least-squares slope. The unpatched baseline is a `git worktree` of the
pre-Release-A commit measured through the identical harness, so payload, request count, warm-up,
replicate count, interpreter and host match by construction. Slopes in MB/request.

| Arm | Requests × replicates | Baseline | Release A | Change |
|---|---|---|---|---|
| A1 — hot single channel | 300 × 3 | +1.49430 | +0.15123 (upper bound +0.30990) | −89.9% |
| A1 — hot single channel | 1,000 × 3 | not run (see below) | **+0.00040 (upper bound +0.01387)** | — |
| A0 — unique channel | 300 × 3 | +1.33898 | +0.49471 (upper bound +0.49966) | −63.1% |
| A0 — unique channel | 1,000 × 1 | +1.33935 | +0.49740 | −62.9% |

**Arm A1 passes, and the acceptance claim is the plateau.** In-memory conversation bytes plateau at
8.8 MB with a second-half slope of exactly +0.00000, and `conversations.turns` stays pinned at 20 for
the whole run. The RSS slope's upper confidence bound is +0.01387 ≤ 0.05, its mean is +0.00040, and
one of three replicates is negative — compatible with zero in this section's own terms.

**The 300-request A1 slope is the estimator's noise floor, not retention.** Refitting the same
estimator on truncated prefixes of one identical 1,000-request sample series reproduces it: +0.14410
at a 300-request window, +0.06208 at 500, +0.01158 at 800, +0.00040 at 1,000. Essentially all RSS
growth happens in the first ~100 requests, so at N=300 the "second half" is still on the warm-up
shoulder. Drawing 150-sample windows from the provably flat region (requests 500–1,000) reports
|slope| > 0.05 in 35.2% of windows containing no growth at all; at a 500-request run length that
false-positive rate is 0.0%. Arm A1 also shows 44 downward RSS steps against 64 upward — retention
never gives memory back, so the shape is allocator oscillation, which §3.2 lists as a non-goal.
**Several hundred requests is not enough on this host; size A1 at 500 requests minimum.**

Baseline A1 was not run at 1,000 requests: its O(n²) save path would write ≈225 GB and take over two
hours per replicate. The matched 300 × 3 pair is what supplies the attribution; the A1 gate is an
absolute threshold on the treatment.

**Arm A0's residual is real retention, and it is the live-session cache.** Unlike A1, arm A0 shows
zero downward RSS steps, a spread of 0.005 across replicates, and a slope stable across horizons.
See §1.2 for the attribution.

**The durable store is ruled out as the source of A1's residual, twice.** Making the payload
compressible cut durable bytes 9.4× (141.1 MB → 15.0 MB) while the slope moved only +0.15123 →
+0.10279, with overlapping replicate ranges. More cleanly: at 1,000 requests both trees write
identical durable volumes in arm A0 (530.0 MB baseline, 530.1 MB treatment) while their RSS slopes
differ by 0.842 MB/request.

### 16.6 Latency acceptance

Measure separately: no-eviction live hit; turn completion without overflow; conversation append at depth;
unchanged-context eviction; changed 450 KB context to local disk; changed 450 KB context to real Redis;
**unique-channel / evict-every-request end to end**; **event-loop scheduling-delay percentiles under that
load**; and **LLM call counts with and without the §9.4 cache policy**.

Gates, all with pre-registered absolute numbers rather than placeholders `[R8, R2-22]`:

- no-eviction p50/p95 regresses by less than 5% or measurement noise, whichever is larger;
- evict-every-request p50/p95 within its pre-registered absolute budget;
- event-loop scheduling delay p99 within its pre-registered absolute budget;
- the cache policy's additional provider calls within its pre-registered budget, or the 200-entry figure
  is revised.

The second and third gates matter more than the first: in the workload that motivated this design every
request is a unique channel, so eviction is the steady state, and revision 2 gated only the no-eviction
path — excluding the dominant path by construction. Because the write happens inside the manager critical
section (`utils.py:762`, `:775`, with `get_session` contending at `:746`), a ~450 KB encode-hash-write
blocks the event loop and every other channel's lookup. Without these numbers §14's `RETIRING`-reservation
deferral has no trigger that can ever fire.

Report absolute and relative eviction latency separately instead of hiding it under LLM duration. No LLM
call may be added.

#### 16.6.1 Measured results — Release A

`tests/soak/memory_soak.py --latency`, 450 KB payload, depth 200 turns on one channel, same
worktree-based baseline as §16.5.1. The eviction, Redis and scheduling-delay rows are Release C
scope; the four measured here are Release A's.

| Measurement | Baseline p50 / p95 | Release A p50 / p95 |
|---|---|---|
| no-eviction live-session hit | 490.7 / 891.3 ms | **38.1 / 46.4 ms** |
| turn completion without overflow | 76.3 / 88.3 ms | 58.9 / 75.3 ms |
| conversation append, shallow (turn 1) | 178.4 / 292.3 ms | 38.7 / 47.7 ms |
| conversation append at depth (turn 200) | 3,226.7 / 3,408.9 ms | 39.7 / 46.5 ms |
| append-at-depth ÷ shallow, p50 | **18.09×** | **1.03×** |

The gated row improves by 12.9× at p50 and 19.2× at p95 rather than regressing, so the 5% gate is met
with a wide margin. The last row is the O(n²) durable rewrite made visible: on the baseline, appending
turn 200 costs eighteen times what appending turn 1 costs, because turn *n* rewrites turns 1..*n*−1.
Under §8.2's true append it is flat. This is a latency and durable-growth defect independent of
memory, which is why §8.2 says that step is worth doing on its own.

Treatment figures are the median of three repeat runs (turn completion: 64.5 / 58.7 / 58.9 ms p50). An
earlier single reading of 147.5 ms was taken while the baseline run occupied the host and is excluded.

**Not measured: LLM call counts with and without the §9.4 cache policy.** The soak drives direct
actions and makes no provider call, so it cannot produce that number, and no LLM call may be added
here. What is verified structurally instead, in `tests/test_server_dspy_memory.py`: the response cache
stays at or below its cap under more unique requests than the cap, a repeated request still serves
from cache and makes zero provider calls, and the disk cache is never written. The 200-entry figure
therefore remains a starting point sized against the live-session budget, not a measured one, and
§9.4's revision trigger is still open.

### 16.7 Repository quality gates

- Activate `.venv`.
- Run affected files first.
- Run the full pytest suite twice with zero failures.
- Never train in or delete `fastworkflow/examples/*/___command_info`.
- Do not remove, skip, or weaken existing tests.

---

## 17. Observability and rollout

Emit one startup INFO record:

```text
memory bounds active:
max_live_sessions=2000 (source=default),
max_retained_startup_turns=20 (~17 MB at 450 KB payload),
turn_retention_seconds=300,
max_conversation_turns_in_memory=20,
dspy_history=off (asserted), dspy_trace=off (asserted),
dspy_memory_cache=200 entries (asserted), dspy_disk_cache=off,
dspy_policy_owner=claimed
```

The byte annotations and `(asserted)` markers are deliberate `[R7, R12]`: a count cap is unauditable
without the payload size, and revision 1 would have logged `dspy_trace=off` even on a DSPy version that
silently ignored the setting. `source=` on the session cap is required by `[R2-20]` — an operator must be
able to see whether the container variable took effect.

Runtime logs:

- DEBUG: session and turn eviction count and reason;
- WARNING: cache over target because all candidates are busy, leased, or pinned;
- WARNING: candidate pinned because state is undeclared, unsupported, or persistence failed;
- WARNING: record identity or generation mismatch, channel quarantined `[R2-10, R2-13]`;
- WARNING: launch-context reconciliation, naming changed and conflicting keys `[R6, R2-6]`;
- WARNING: a channel was skipped for eviction because a lease was held or `runtime.lock` was held with no
  registry execution `[R2, R2-2]`;
- ERROR: shutdown deadline expired with work still in flight; runtimes left unclosed `[R2-21]`;
- ERROR: DSPy policy drift detected after readiness `[R2-19]`.

Rate-limit warnings per channel and reason, and never log state or credentials.

**Metrics, not just logs.** Pinned-session count, quarantined-channel count, durable-store bytes per
namespace, DSPy cache entries and bytes, and in-memory conversation bytes must be metrics rather than log
lines, because for undeclared workflows "pinned" is the steady state (§1.4) and a rate-limited warning is
suppressed exactly when it matters most `[R1, R5, R2-17]`.

Rollout:

1. reproduce the unpatched baseline;
2. **Release A**: pass §16.1, §16.2 and §16.4 DSPy tests plus two full-suite runs; run soak arms A0 and A1
   and the latency matrix; canary while monitoring RSS, conversation bytes, DSPy cache bytes, and LLM call
   counts;
3. keep worker recycling as the supported interim mitigation for the unique-channel workload until
   Release C `[R2-22]`;
4. **Release B**: pass §16.3 and the §16.4 manager matrix; land `fix-jtr` for the checkpoint namespace and
   the fleet-version gate; verify new→old→new and rollback-after-eviction; do not enable checkpointing for
   any workflow that has not declared a durable projection;
5. **Release C**: only after arms A and C pass and the durable-storage plateau is demonstrated; canary
   while monitoring RSS, store growth, pinned and quarantined counts, eviction failures, cold latency, and
   duplicate startup calls.

Raising `MAX_LIVE_SESSIONS` is the operational rollback for excessive cold churn after Release C. It does
not re-enable unbounded turn, conversation, or DSPy retention.

**Rollback caveats.** The lever works only if §5.1's OS-first resolver ships *and* the example env file
documents the value as a comment rather than an active assignment `[R4, R2-20]`. There is **no**
operational rollback for the §1.4 pinned case: raising the cap does not help when nothing is evictable, so
worker recycling is the only lever. And after the first checkpoint is written, **binary downgrade is
unsupported** `[R2-18]` — raising a cap is tuning, not a version rollback.

---

## 18. Decision log

1. **Count cap plus 300-second age window:** accepted.
2. **Retain only startup terminal records for now:** accepted, **narrowed** `[R2-16]`. Justified solely by
   the live-runtime re-poll path; the cross-eviction recoverability claim is withdrawn and the burst case
   is an accepted limitation rather than a sizing target.
3. **Fixed terminal cap of 20:** accepted `[R12]`. 20 records ≈17 MB at a 450 KB payload against an
   observed need of ≈6; revision 1's 100 would have cost ≈87 MB.
4. **Default live-session target of 50:** accepted **only as Release C**, gated on the full Release B
   protocol `[review §5]`.
5. **Only `MAX_LIVE_SESSIONS` as new config:** accepted.
6. **No live idle TTL:** accepted.
7. **Weak creation locks:** accepted.
8. **Persist only at retirement/shutdown:** accepted, **amended** `[R2-4]`. The semantic-metadata commit
   (startup state) is synchronous at turn completion, because a fact whose loss changes restart behavior
   cannot wait for a context write that may never happen.
9. **Separate namespaces per record kind:** accepted, **subsumed** `[R2-10]` by the generation-stamped
   composite commit; namespace separation remains testable but is no longer the mechanism.
10. **Strict versioned JSON, never `default=str`/pickle:** accepted, **relocated** `[R2-7]`. Strictness is
    enforced at `WorkflowExecutionContext.serialize_state()`, the first serializer, not at the store.
11. **Persist-before-pop:** accepted; correctness wins over forcing a soft target.
12. **Saved application context is authoritative:** accepted, **corrected** `[R6, R2-6]`. See decision 20.
13. **Empty context is a real snapshot:** accepted.
14. **Reuse retained startup after runtime eviction:** **superseded** `[R3]`, and the replacement itself
    amended `[R2-4, R2-5]`. Startup authority is durable, expressed as a state machine with an explicit
    commit point, not a boolean.
15. **Configure DSPy before event loop, server-only, and assert it:** accepted, **extended** `[R2-19]`.
    A one-time probe is insufficient because the first async task on the owning thread can silently
    reconfigure; ownership is claimed and drift fails readiness.
16. **No hidden durable-state TTL:** accepted, **amended** `[R5, R2-17]`. Durable growth is gated on a
    plateau, and Release C waits on `fix-jtr` — not on `fix-6b4`, which covers only orphaned suspended-session
    blobs.
17. **Production slope remains supplied until reproduced:** accepted.
18. **Fix 2's bound is conditional on workflow shape, disclosed rather than mitigated:** accepted `[R1]`,
    with the census corrected `[review §3]`. Alternatives considered: evicting without a snapshot
    (rejected — reintroduces silent state loss); capping the pinned set and shedding load with 503
    (rejected for this release train, needs the admission-control work in
    `docs/fastworkflow_turns_async_execution_design.md` Step 2 to be coherent; still the preferred
    long-term answer).
19. **Eviction safety uses the union of registry pointer and held lock; 409 idempotency does not:**
    accepted `[R2]`, **and found insufficient** `[R2-2]`. See decision 24.
20. **Launch-time context is reconciled, not discarded:** accepted `[R6]`, **mechanism replaced**
    `[R2-6]`. A single aggregate digest can detect change but cannot attribute it, so the envelope stores
    the prior canonical launch projection and performs a real three-way merge. Separating operator
    configuration from application context is the preferred long-term shape and is deferred as a wider
    framework change.
21. **Strictness applies to the continuation write too:** accepted `[R10]`, **relocated** `[R2-7]`.
22. **Namespace separation is an invariant with a test:** accepted `[R11]`, subsumed by decision 9.
23. **The single atomic release is withdrawn in favor of three gated releases:** accepted
    `[review §5, R2-1 … R2-14]`. The universal memory bounds do not depend on the checkpoint protocol and
    should not wait for it; the lower cache limit does depend on it and must not precede it. Rejected
    alternative: keep one release and resolve all 14 blocking findings at once — rejected because it
    repeats revision 2's error of shipping a system-of-record protocol on a cache schedule, and because
    §1.2 shows Release A has measurable standalone value.
24. **Eviction safety is a lease, not a predicate:** accepted `[R2-1, R2-2, R2-3]`. `get_session()`
    returns a raw runtime after releasing the manager lock, so any boolean sampled later cannot make lookup
    and admission atomic; and a runtime under construction is the *only* apparently safe victim precisely
    when every older session is pinned, so creation can evict itself. Leases cover the lookup interval,
    the initialization interval, and the streaming interval with one mechanism.
25. **Durable state is workflow-declared; nothing is durable by default:** accepted
    `[R2-5, R2-9, R2-12]`. Negative eligibility checks cannot enumerate behaviorally mutable state, and
    persist-all-JSON silently moves a deployment's data-classification boundary. Declaration makes
    Release C's bound an opt-in property and preserves today's semantics for every workflow that does not
    opt in. Rejected alternative: an allowlist of framework-known-safe keys — rejected because it still
    guesses on application state.
26. **Awaiting and continuation sessions are pinned in Release B v1:** accepted `[R2-8]`. Today's
    suspended-state restore already loses the logical-turn accumulator and the CME continuation keys;
    pinning avoids making a pre-existing defect routine without pretending to have fixed it. Recorded
    limitation: pinning does not help multi-pod, where another pod still cold-rehydrates.
27. **Conversation history is bounded, and durable append lands first:** accepted `[R2-11]`. Goal 1 is
    false today on a hot single channel; and because `save_conversation_turns` replaces the full list, an
    in-memory window applied first would delete durable turns. The append change also removes O(n²) write
    amplification.
28. **The DSPy response cache gets an explicit policy: bounded memory, disk off:** accepted `[R2-15]`.
    Defaults are 1,000,000 in-memory entries and a 30 GB pickle-backed disk cache, and the revision-2
    policy left both untouched. The cost is more provider calls in repeated-prompt workloads, which
    §16.6 measures rather than assumes. Rejected alternative: disable the cache entirely — rejected
    because NLU and parameter extraction issue structurally similar prompts within a session.
29. **The verification plan runs the real server and reports a plateau, not a slope:** accepted
    `[R2-22]`. The in-process ASGI harness could bypass the pre-event-loop entrypoint it exists to gate;
    0.05 MB/request is a leak budget (≈78 MB/day) rather than a bound; and every numeric threshold is
    pre-registered before treatment is measured.

---

## 19. Accepted limitations and follow-up triggers

1. Active concurrency remains unbounded; trigger the existing backpressure work when active channels
   dominate RSS.
2. Durable checkpoint records have no lifecycle policy yet. **Quantified** `[R5]`: ≈700 MB/day at the
   observed rate and payload. Release C waits on `fix-jtr` `[R2-17]`; `fix-6b4` does not cover this
   namespace.
3. Undeclared or object-context application state pins a runtime. **This trigger has already fired**
   `[R1]`: five bundled workflows including `simple_workflow_template` assign a command-context object, so
   a workflow-owned serializer (never pickle) is a known prerequisite for a general live-session bound, not
   a conditional follow-up.
4. Terminal retention is in-process; restart loses results, and eviction makes them unreachable
   `[R2-16]`. The startup-completion *fact* is durable `[R3]`.
5. The 300-second age window is opportunistic and, in a burst of more than 20 startups, does not bind at
   all `[R2-16]`.
6. Persistence is synchronous in the first implementation. §16.6's evict-every-request and
   scheduling-delay gates are what make "material" measurable `[R8, R2-22]`.
7. Single-writer-per-channel remains required — and is violated today by the §10.2 detached-runtime race,
   which Release B fixes rather than inherits `[R2-2]`.
8. Disk channel-ID collision is pre-existing, but this design would make it **durable and authoritative**,
   so §11.3 fixes it rather than recording it `[R2-13]`.
9. `remove_session`/`evict_live_session` are preserved with existing semantics despite having no production
   callers `[R15]`.
10. The §9 DSPy policy depends on `Settings.context` inheriting `main_thread_config`, which is internal
    behavior rather than a documented contract. §9.3's assertion plus ownership converts a future DSPy
    change from a silent leak into a refused or unready startup `[R7, R2-19]`.
11. Multi-pod suspension across pods remains incomplete until §11.6's continuation serialization lands;
    pinning bounds only the local eviction path `[R2-8]`.
12. Exactly-once startup is scoped to framework-managed effects; external side effects need
    application-owned idempotency `[R2-5]`.

---

## 20. Recommendation

Ship **Release A** as an independent minor release. It is behaviorally atomic, adds no durable state, no
new eviction path, and no wire change, and it fixes a real unbounded retainer on a supported workload
(§8) that no cache bound touches. Record arm A0's slope honestly: Release A does not close the motivating
unique-channel OOM, and worker recycling remains the interim mitigation.

Do **not** implement Releases B and C as one package with A, and do not implement Release C before B. The
round-2 verdict stands: revision 2's checkpoint mechanism was not behavior-preserving, crash-atomic, or
security-complete, and lowering the cache limit is exactly what makes those defects routine.

Do not copy these shortcuts from the source-study patch or from revision 2:

- manual pruning of an apparently unlocked creation lock;
- returning early for an empty context;
- lossy `default=str` serialization — anywhere in the path, and note the first offender is
  `WorkflowExecutionContext.serialize_state()`, not the store `[R2-7]`;
- evicting after a best-effort failed state write;
- persisting credentials, or persisting everything except one named key `[R2-12]`;
- configuring DSPy for the first time inside the async lifespan task;
- treating a successful `configure()` call as proof the policy is active `[R7]`, or a one-time probe as
  proof it stays active `[R2-19]`;
- deriving startup-already-ran from in-process turn retention `[R3]`, or writing the fact only when the
  context digest happens to change `[R2-4]`;
- using any sampled boolean as the eviction-safety predicate `[R2, R2-2]`;
- letting launch-time configuration changes be silently outranked by a snapshot `[R6]`, or claiming a
  reconciliation the stored information cannot support `[R2-6]`;
- inferring snapshot eligibility from negative checks `[R2-9]`;
- treating two independent writes as one transaction `[R2-10]`;
- gating on a slope while durable storage grows without bound `[R2-17, R2-22]`.

**And one thing this design must not do to itself.** Revision 1 contained versions of four of those
shortcuts and revision 2 contained versions of nine. The round-2 reviewer's summary is the standard this
release train is held to: an RSS graph that flattens on a function-style fixture while object-context
workflows stay unbounded, streaming turns get torn snapshots, a stale snapshot silently outranks a
redeployed configuration, and a hot channel keeps leaking the whole time. Arms A0, A1 and B of §16.5 exist
so that none of that can happen quietly.

Sequencing that follows from the review:

1. Resolve decisions 18, 23, 24 and 25 before any Release B code lands — they change what §1 promises.
2. Ship Release A on its own evidence, with arm A0's number published rather than elided.
3. Land leases and the streaming lifecycle (§10) before any cap reduction; the cap reduction is what makes
   those races reachable.
4. Fold the record-shape findings (R2-4, R2-6, R2-7, R2-10, R2-13) into one envelope-and-store slice, where
   they are cheapest.
5. Treat R2-20's env precedence and R2-17's namespace lifecycle as Release C blockers: both are things an
   operator will rely on.
6. Per review §5, open a review-only beads child per accepted finding before implementation begins;
   `fix-jtr` is the first.

---

## 21. Traceability

Round-2 amendments, each traceable to the finding that forced it. Severity is from the review's index:
S1 blocking, S2 major, S3 moderate. All findings live in review §4 unless noted. `§4 inv. N` names an
invariant by its stable number, and `§19.N` names item N of §19's list.

| Finding | Sev | Sections amended | Resolution |
|---|---|---|---|
| R2-1 creation-time trimming evicts the runtime being initialized | S1 | §4 inv. 20, §10.3, §13.2, §15, §16.4, decision 24 | Initialization lease from insertion until startup establishes eligibility; cap-one failing-first test |
| R2-2 the union predicate is a sample, not a lease | S1 | §4 inv. 6 + 19, §10.2, §13.1, §15, §16.4, §19.7, decision 24 | Manager-owned lease acquired atomically with lookup, held through executor exit |
| R2-3 streaming is outside the turn lifecycle and can outlive its lock | S1 | §4 inv. 21, §10.4, §15, §16.4, §16.5 arm C | Streaming routed through turn admission; ownership survives timeout and disconnect |
| R2-4 startup completion has no durable commit point | S1 | §4 inv. 25, §11.5, §15, §16.3, decision 8 + 14 | Synchronous semantic-envelope commit before success is observable; envelope-wide digest |
| R2-5 the startup record cannot represent replay semantics | S1 | §3.2, §11.5, §16.3, §19.12, decision 14 + 25 | Explicit state machine with epoch; guarantee scoped to framework-managed effects; object-context case resolved by declaration |
| R2-6 one aggregate digest cannot perform the promised merge | S1 | §4 inv. 28, §11.3, §11.9, §13.2, §16.3, §17, decision 20 | Store the prior canonical launch projection; real three-way merge with conflict semantics |
| R2-7 strict validation occurs after lossy coercion | S1 | §4 inv. 7 + 9, §11.4, §15, §16.3, §20, decision 10 + 21 | Remove the WEC `default=str` round-trip; one strict serializer at the first boundary; test through `serialize_state()` |
| R2-8 suspended restore omits the logical turn and CME continuation | S1 | §11.2, §11.6, §16.3, §19.11, decision 26 | Pin awaiting and continuation sessions in v1; completion tracked, with the multi-pod caveat stated |
| R2-9 eligibility does not cover behaviorally mutable state | S1 | §3.1, §11.1, §11.2, §16.3, decision 25 | Declared projection plus an enumerated restorable-field list; anything outside pins |
| R2-10 "transactional eviction" is two independent commits | S1 | §4 inv. 26, §11.8, §13.3, §16.3, decision 9 | Generation stamp with a commit record; restore fails closed and quarantines |
| R2-11 one hot channel grows conversation memory per request | S1 | §1.2, §3.1, §4 inv. 23, §8 (new), §14, §15, §16.2, §16.5 arm A1, decision 27 | Durable append and high-water mark first, then a 20-turn in-memory window |
| R2-12 persist-all JSON creates a credential-at-rest boundary | S1 | §1, §3.1, §4 inv. 11, §11.1, §16.3, decision 25 | Nothing durable unless declared; raw-bytes credential tests |
| R2-13 records are not bound to channel, workflow, deployment | S1 | §4 inv. 27, §11.3, §15, §16.3, §19.8 | Injective channel key, identity fields validated on read, private modes |
| R2-14 bearer refresh can overwrite a running turn's credential | S1 | §3.1, §4 inv. 29, §11.7, §15, §16.3 | Credential carried on the execution, installed under the accepted turn; documented read contract preserved |
| R2-15 the DSPy response cache is a large process-lifetime retainer | S2 | §2.2, §4 inv. 24, §5.2, §9.4 (new), §14, §16.4, §17, decision 28 | Bounded memory cache, disk cache off; measured below `LM.forward` |
| R2-16 retained startup output is not collectable after eviction | S2 | §3.2, §6.1, §6.5, §19.4–5, decision 2 | Claim withdrawn; retention justified by the live-runtime re-poll only; burst limitation accepted |
| R2-17 durable-growth gate and rollout dependency non-executable | S2 | §11.10, §16.5, §17, §19.2, decision 16 | Plateau gate on total bytes; `fix-jtr` opened; `fix-6b4` dependency corrected |
| R2-18 mixed-version rollout can resurrect stale state | S2 | §4 inv. 31, §11.3, §11.10, §17 | Protocol version and fleet floor; downgrade after first checkpoint unsupported |
| R2-19 the DSPy policy can be disabled after the assertion | S2 | §2.2, §4 inv. 14, §9.3, §16.4 step 8, §17, decision 15 | Claim async ownership, re-assert at readiness, negative test, worker/reload model stated |
| R2-20 the documented env default defeats the container override | S2 | §4 inv. 22, §5.1, §15, §17 | OS-first resolver, commented example line, effective value and source logged |
| R2-21 the shutdown union does not make shutdown quiescent | S2 | §4 inv. 30, §10.5, §16.4, §17 | Close admission, drain leases, never write or close past the deadline |
| R2-22 the gates prove neither plateau nor production behavior | S3 | §1.2, §14, §16.5, §16.6, §17, decision 29 | Real server subprocess, USS and cgroup, replicates with confidence bounds, all thresholds pre-registered |

Round-1 findings R1–R15 remain resolved as recorded in revision 2, with these round-2 corrections
(review §3): R1's census fixed and its scope moved to Release C (§1.4); R2's predicate found necessary but
insufficient (§10.2); R3's envelope given a commit point and a state machine (§11.5); R4's mechanism kept
but its documentation requirement corrected (§5.1); R5's growth given a plateau gate (§16.5); R6's
mechanism replaced (§11.9); R7 extended past startup (§9.3); R8's gates given numbers (§16.6); R10
relocated to the first serializer (§11.4); R11 subsumed by the composite commit (§11.8); R12's cap
retained with the byte discipline extended to the DSPy cache (§5.2). R9, R13, R14 and R15 are unchanged and
were confirmed resolved by the review.

**Findings this revision declined to act on: none.** Three were resolved differently than the review
proposed:

- **R2-8** — pinning awaiting sessions rather than serializing the complete logical turn now, because the
  pin is sufficient for the memory bound and the serialization is a larger change that should not be rushed
  into it (decision 26). The review's completeness requirement is retained as a tracked prerequisite for
  multi-pod suspension.
- **R2-9/R2-12** — resolved together by workflow declaration rather than by enumerating a framework-owned
  capability plus a separate security classification, because one opt-in principle covers both and
  preserves today's behavior for workflows that do not opt in (decision 25).
- **R2-16** — the recoverability claim is withdrawn rather than implemented, because authenticated turn
  polling is a Step-2 non-goal (decision 2).

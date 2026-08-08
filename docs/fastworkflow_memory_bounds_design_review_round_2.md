# Adversarial review, round 2 — bounded memory for `run_fastapi_mcp`

**Reviews:** `docs/fastworkflow_memory_bounds_design.md` revision 2  
**Background:** `docs/fastworkflow_memory_fixes.md` (held privately, not in this repository)  
**Previous review:** `docs/fastworkflow_memory_bounds_design_review.md`  
**Verified against:** fastWorkflow `2.24.1`, commit
`c206a813af1f48e09b63f7972cfcb16ee2262d2a`; DSPy `3.2.1` in `.venv`
(Python 3.12.2)  
**Date:** 2026-08-05  
**Tracking:** `fix-p3l`, discovered from `fix-9uk`

> Historical record. Findings R2-1–R2-22 are resolved in revision 3 of the design; see its §21
> traceability table.

---

## 1. Method, terminology, and verdict

This is **round 2 relative to the retained review document**. Chronologically it is the
third review pass because the original `fix-9uk` review was integrated directly into revision 1.
The target is revision 2, which claims to absorb all R1–R15 findings from the retained review.

Seven independent concern passes attacked:

1. runtime and eviction concurrency;
2. startup idempotency and terminal retention;
3. state-envelope fidelity and restore precedence;
4. disk/Redis durability, isolation, and credential handling;
5. DSPy configuration and cache behavior;
6. memory-proof and acceptance methodology;
7. the R1–R15 resolution trace itself.

Every accepted finding below was then rechecked against the pinned tree. Small local probes were
used where reading alone was insufficient:

- a DSPy call-shape probe below `LM.forward`;
- a DSPy configuration-ownership probe;
- a disk channel-ID collision reproduction.

No treatment implementation exists, so this is a design and baseline-code review, not a
post-implementation audit.

### Verdict

**Do not implement revision 2 as written.** The retainer diagnosis remains sound, but the revised
state protocol is not behavior-preserving, crash-atomic, or security-complete. More importantly,
two monotonic process retainers remain outside the proposed bound: same-channel conversation
history and DSPy's response cache.

Round 2 records:

- **14 S1 blocking findings** — the release can corrupt state, cross credential/conversation
  boundaries, fail at the new capacity threshold, or remain unbounded on a supported request shape;
- **7 S2 major findings** — operational promises, rollout, or recovery are not implementable as
  specified;
- **1 S3 moderate finding** — the evidence plan cannot establish the claim it is intended to gate.

Severity scale: **S1** blocking (design cannot ship as written) · **S2** major (resolve before
implementation) · **S3** moderate (resolve before release) · **S4** minor/bookkeeping.

Round-2 findings use `R2-#` so they cannot be confused with the previous review's R1–R15.
None is marked resolved in this document.

### Findings index

| # | Severity | Finding | Primary sections |
|---|---|---|---|
| R2-1 | **S1** | Creation-time trimming can evict the runtime being initialized | §7.1, §10.2, §13.5 |
| R2-2 | **S1** | The union busy predicate is a sample, not a runtime lease | §4.6, §7.1, §10 |
| R2-3 | **S1** | Streaming remains outside the turn lifecycle and can outlive its lock | §4.2/6, §7.1/7.3, §13.5 |
| R2-4 | **S1** | Startup completion has no durable commit point | §4.12/18, §6.3, §8.3/6 |
| R2-5 | **S1** | The startup record cannot represent safe replay/suppression semantics | §6.3/5, §8.1/3, §16.4 |
| R2-6 | **S1** | One aggregate digest cannot perform the promised three-way merge | §8.3, §8.5.1 |
| R2-7 | **S1** | Strict pending-state validation occurs after lossy coercion | §4.7/9, §8.4, §12 |
| R2-8 | **S1** | Suspended-state restore omits the logical turn and deterministic continuation | §8.1/4/7, §13.3 |
| R2-9 | **S1** | Snapshot eligibility does not cover all behaviorally mutable state | §3.1, §8.1/3/5 |
| R2-10 | **S1** | “Transactional eviction” is two independent commits | §4.3/7, §8.7/8 |
| R2-11 | **S1** | One hot channel still grows conversation memory per completed request | §2.3, §3.1, §11, §13.5 |
| R2-12 | **S1** | Persist-all JSON silently creates a credential-at-rest boundary | §1, §8.4/7/8 |
| R2-13 | **S1** | State records are not safely bound to channel, workflow, and deployment | §8.2/3/8, §16.8 |
| R2-14 | **S1** | Bearer refresh can overwrite a credential used by a running turn | §1, §8.5.1, §12 |
| R2-15 | S2 | DSPy's response cache remains a large process-lifetime retainer | §9, §11, §13.3–5 |
| R2-16 | S2 | Retained startup output is not collectable after runtime eviction | §5.2, §6.1/3/5 |
| R2-17 | S2 | The durable-growth gate and its rollout dependency are non-executable | §8.9, §13.5, §14, §16.2 |
| R2-18 | S2 | Mixed-version rollout and downgrade can resurrect stale state | §8.3, §14 |
| R2-19 | S2 | The DSPy policy can be disabled after the startup assertion | §9.2/2.1, §13.4 |
| R2-20 | S2 | The documented env default can defeat the container override | §5.1, §12, §14 |
| R2-21 | S2 | The shutdown union does not make timeout or admission quiescent | §7.3, §12, §13.2 |
| R2-22 | S3 | The verification gates do not prove a plateau or production behavior | §13.5–7 |

---

## 2. What revision 2 gets right

The following should survive the next revision:

1. **The four original retainer diagnoses are real.** Terminal executions and creation locks have
   no removal path; the 2,000-session default is unsafe for payload-heavy sessions; DSPy history
   and trace retain calls by default.
2. **The source-study treatment number remains labeled inconclusive locally.** Revision 2 does not
   pretend the 0.032 MB/request result was reproduced in this tree.
3. **Weak creation locks are the right primitive.** `asyncio.Lock` is weak-referenceable, and
   holders/waiters retain a strong reference while identity matters.
4. **R9 and R13 are correctly resolved.** Store I/O is ordered after
   `TurnRegistry._lock` release, and active-pointer removal is no longer double-specified.
5. **R11's direct pending/context Redis-prefix collision is correctly elevated to an invariant.**
   The factory really cannot express the second prefix today, and normal-turn cleanup would delete
   context state if the prefixes matched.
6. **The R7 history/trace mechanism works in the executor shape.** With
   `disable_history=True`, `max_trace_size=0`, and `trace=[]`, calls inside `dspy.context` in a
   worker thread leave `GLOBAL_HISTORY`, LM history, and predictor trace empty.
7. **The object-context limitation is now disclosed early.** That is materially more honest than
   revision 1, even though R2-1 and the incorrect workflow census make the stated runtime behavior
   wrong.
8. **Strict JSON and persist-before-pop are the correct direction.** The defects below are about
   where strictness is applied and what constitutes one transaction, not an argument for pickle,
   `default=str`, or best-effort eviction.

---

## 3. Audit of the previous R1–R15 resolutions

This table distinguishes “the requested sentence/test was added” from “the revised mechanism is
actually implementable.”

| Prior finding | Round-2 status | Reason |
|---|---|---|
| R1 | **Partially resolved** | Conditional scope is disclosed and arm B exists, but creation ordering produces self-eviction rather than the stated monotonic pinned growth (R2-1). The census also incorrectly marks `messaging_app_3` evictable even though it assigns an object to `current_command_context`. |
| R2 | **Partially resolved** | The held-lock streaming case is covered by the union predicate, but lookup-to-admission and lazy-stream intervals have no lease (R2-2/R2-3). |
| R3 | **Partially resolved** | Authority moved to the envelope, but no durable completion commit or coherent replay policy exists (R2-4/R2-5). |
| R4 | **Mechanism resolved; operation incomplete** | `default=None` reaches `os.environ`, but an active default copied into `fastworkflow.env` outranks it (R2-20). |
| R5 | **Partially resolved** | Growth is quantified, but no actual acceptance number is present and `fix-6b4` covers another namespace (R2-17). |
| R6 | **Contradicted** | The envelope stores only an aggregate digest, while reconciliation requires prior per-key launch values (R2-6). |
| R7 | **Resolved for history/trace** | The structural assertion and executor-context test are specified. DSPy cache and post-start override are separate round-2 findings (R2-15/R2-19). |
| R8 | **Partially resolved** | The relevant latency path is named, but the absolute latency and scheduler-delay limits are still placeholders (R2-22). |
| R9 | **Resolved** | Store I/O is explicitly outside `TurnRegistry._lock`. |
| R10 | **Contradicted** | The proposed strict store sees state only after `serialize_state()` has already applied `default=str` (R2-7). |
| R11 | **Resolved for the direct prefix collision** | Pending/context prefixes are separated; broader identity and channel-key issues remain (R2-13). |
| R12 | **Resolved for derivation/bookkeeping** | The cap is 20 and its representative byte cost is stated. It still provides no minimum collection window under a burst (R2-16). |
| R13 | **Resolved** | Pointer removal and pruning now share `clear_active`'s existing lock block. |
| R14 | **Resolved for queued visibility** | The union sees queued registry work; shutdown still closes after its deadline and does not quiesce admission (R2-21). |
| R15 | **Resolved** | The `.venv` interpreter and dead lifecycle methods are identified. |

---

## 4. Findings

### R2-1 — [S1] Creation-time trimming can evict the runtime being initialized

**The design says.** §10.2 registers the runtime and enforces the LRU target at step 9, then
consults/submits startup at step 10. §7.1 skips busy or unsupported sessions and permits remaining
over target when none can be removed. §13.5 arm B expects object-context sessions to accumulate
above the target.

**What the tree says.** `ChannelSessionManager.create_session()` inserts the new runtime and calls
`_evict_oldest_if_needed()` before returning (`utils.py:752-776`). `/initialize` submits startup
only after creation returns (`__main__.py:718-769`). Before startup, the new runtime has no registry
pointer, its lock is free, and its workflow has no root/current object.

**Counterexample.**

1. Fifty older sessions are pinned by object contexts.
2. Channel 51 is inserted.
3. The LRU skips all fifty pinned candidates.
4. Channel 51 is the only apparently safe candidate because its startup has not yet created the
   object context that will pin it.
5. The manager evicts channel 51 inside its own `create_session()`.
6. Startup's second lookup fails and `/initialize` returns 500
   (`__main__.py:735-740`). With no startup action, the endpoint can return tokens for a runtime
   that no longer exists.

This is the negative observation arm B misses: below capacity, or with an older evictable victim,
creation works. The defect appears exactly when the new default begins to matter.

**Required resolution.** Give a newly created runtime a `CREATING`/`INITIALIZING` lease that
disqualifies it from eviction until startup admission/completion establishes final eligibility.
Test a cap-one manager with the existing session pinned, then create a second channel both with and
without startup.

---

### R2-2 — [S1] The union busy predicate is a sample, not a runtime lease

**The design says.** Invariant 6 and §7.1 define in-flight work as
`has_active(channel_id) or runtime.lock.locked()`.

**What the tree says.** `get_session()` returns a raw `ChannelRuntime` after releasing the manager
lock (`utils.py:745-750`). A normal endpoint then awaits registry admission; a streaming endpoint
returns a lazy response before acquiring `runtime.lock`. During either interval both halves of the
union are false.

**Race.**

1. Request A obtains runtime X and releases the manager lock.
2. Before A inserts a turn pointer or acquires X's lock, request B creates another channel.
3. The LRU snapshots, pops, and closes X.
4. A continues using the detached X.
5. A cold recreation can call `Workflow.create()` while X still strongly owns the old app workflow.
   The weak global registry can return that same workflow and overwrite its context
   (`workflow.py:101-104`), yielding two runtimes/locks around one mutable workflow.

The union correctly protects work *after* one signal becomes visible. It does not make lookup and
admission atomic.

**Required resolution.** Add a manager-owned runtime lease/refcount acquired atomically with
lookup and held through turn registration or lock acquisition, then through actual executor
completion. A boolean sampled later is insufficient. Add barrier-controlled tests for the normal
and streaming lookup-to-admission intervals.

---

### R2-3 — [S1] Streaming remains outside the turn lifecycle and can outlive its lock

**The design says.** R2 is resolved by adding `runtime.lock.locked()` to eviction safety while
leaving the registry as the 409 source.

**What the tree says.**

- `/invoke_agent_stream` never creates a `TurnExecution`; it captures the runtime before returning
  `StreamingResponse` and acquires the lock only when body iteration begins
  (`__main__.py:953-974`, `:1073-1085`).
- A normal endpoint can therefore register a turn while a stream owns the lock, because the
  registry sees no active stream (`turns.py:195-214`).
- `run_process_message_with_trace_stream()` raises 504 at its own deadline without awaiting or
  cancelling the executor future (`utils.py:532-554`). The route catches that HTTP exception,
  emits an error, and exits the lock while the thread can still mutate WEC/app state.

**Failure schedules.**

1. A stream suspends on `ask_user`; an unrelated normal turn has already queued behind it. When the
   stream releases the lock, the unrelated query is interpreted as the clarification answer.
2. A stream times out; its lock is released, registry remains empty, and eviction/shutdown writes a
   snapshot while the detached executor thread is still running.
3. The runtime is evicted after response construction but before Starlette begins consuming the
   body generator.

**Required resolution.** Route streaming through the same turn-admission/lifecycle owner as every
other turn. Client timeout or disconnect may stop delivery, but registry ownership and the runtime
lease must remain until the executor exits. Test stream→normal, normal→stream, suspension, timeout,
disconnect, and eviction before first body iteration.

---

### R2-4 — [S1] Startup completion has no durable commit point

**The design says.**

- §6.3/§8.3: write startup completion when the turn completes.
- Invariant 12 and decision 8: write workflow context only at retirement/shutdown.
- §8.6: skip the envelope write when the context digest is unchanged.
- §16.4: restart no longer loses the startup-completion fact.

Those statements cannot all hold.

**Counterexample.** A startup performs an external side effect or read-only initialization but does
not mutate `workflow.context`. The turn returns success. Its in-memory `startup.completed` changes,
but the context digest does not. There is no retirement yet, and the no-write path can skip the
future envelope write. A process crash then loses the completion fact and restart replays startup.

Current completion persists conversation/pending state before `DONE`, but has no context-envelope
commit (`turns.py:302-318`). The design does not add one to that ordering.

**Required resolution.** Define a synchronous semantic-envelope commit before startup success/DONE
is observable. Digest the complete envelope, not just context, or maintain an explicit metadata
generation/dirty bit. Store failure at this point must have a specified response and retry policy.
Test a no-context-mutation startup, process death after command completion, and store failure before
terminal publication.

---

### R2-5 — [S1] The startup record cannot represent safe replay/suppression semantics

The proposed record is only:

```json
{"idempotency_key": null, "completed": false}
```

It cannot represent attempted, suspended, succeeded, command-level failure, executor exception,
partial mutation, or whether the original result remains collectable. `_run_turn` transitions to
`DONE` even after an exception (`turns.py:307-318`), and `TurnStatus.COMPLETED` is orthogonal to
command success.

There is a deeper contradiction for the workflow shape revision 2 explicitly accepts:

1. `simple_workflow_template` startup creates `root_command_context`.
2. §8.1 declares that object unsnapshottable and pins the runtime.
3. If shutdown writes only `startup.completed`, restart suppresses startup but cannot reconstruct
   the root object; the workflow resumes without required state.
4. If shutdown refuses the metadata write because the full state is unsupported, restart must rerun
   startup, contradicting invariant 18 and §16.4.

No framework envelope can provide exactly-once semantics for an arbitrary startup's external side
effects without application idempotency.

**Required resolution.** Scope the guarantee to fully restorable framework-managed effects.
Define explicit startup states and success predicate, include a source/migration epoch, and specify
retry/reset behavior. For object-context workflows, either implement the workflow serializer first
or retain/replay startup as a reconstruction step with application-owned idempotency. Test
mutate-then-raise, `success=False`, suspension, read-only side effect, and restart of
`simple_workflow_template`.

---

### R2-6 — [S1] One aggregate digest cannot perform the promised three-way merge

**The design says.** The envelope stores one `launch_context_digest`; on mismatch restore overlays
only launch-time keys whose values differ from the launch-time values recorded when the snapshot
was written.

**Information actually stored.** No prior values or per-key hashes are recorded—only one SHA-256
digest (§8.3).

Given:

```text
old launch = {url: A, tenant: T1}
saved app   = {url: B, tenant: T1}
new launch = {url: C}          # tenant intentionally removed
```

the digest can say “something changed.” It cannot identify `url`, know that `tenant` was removed,
or distinguish the application's A→B mutation from an operator A→C change. The algorithm is
information-theoretically unimplementable.

**Required resolution.** Persist the prior canonical launch projection (or a per-key baseline with
tombstones) and define add/change/delete/conflict semantics. The cleaner alternative is to separate
operator-owned configuration from mutable application context instead of inferring provenance in
one flat dict. Add true three-way-merge tests.

---

### R2-7 — [S1] Strict pending-state validation occurs after lossy coercion

Revision 2 correctly bans `default=str` in the store, but the store is not the first serializer.
`WorkflowExecutionContext.serialize_state()` already returns:

```python
json.loads(json.dumps(payload, default=str))
```

at `workflow_execution_context.py:343-365`. Unsupported trajectory/artifact objects have become
ordinary strings before the proposed strict projector sees them. Strict store validation then
accepts the lossy result.

The implementation map changes `session_state_store.py` but omits the actual coercion boundary in
`workflow_execution_context.py`.

**Required resolution.** Remove the WEC round-trip and pass raw typed state through one strict,
shared serializer before any coercion. Test through `ctx.serialize_state()`, not by calling the new
store helper directly, with an opaque object, cycle, non-finite float, and nested artifact. Any
failure must keep the runtime live.

---

### R2-8 — [S1] Suspended-state restore omits the logical turn and deterministic continuation

The existing pending snapshot stores suspension flags, ReAct state, NLU stage, action log, and
conversation turns. It does **not** store WEC's logical-turn accumulator:

- `_turn_outputs`;
- `_turn_key`;
- `_turn_started_at`;
- original/refined message;
- suspended duration and suspension start;
- entry workflow/context;
- agent result.

Those fields are initialized at `workflow_execution_context.py:104-114`. A resume deliberately
skips `_begin_turn()` (`:547-549`), so it requires the old accumulator. After rehydration, the
logical turn receives a new fallback key and loses pre-suspension command outputs, ask-user entry,
artifacts, and timing.

The deterministic CME continuation is also incomplete. `serialize_state()` stores only
`nlu_stage`, while clarification/parameter extraction depends on CME context keys `command`,
`command_name`, and `stored_parameters` (`wildcard.py:43-46`, `:135-150`;
`parameter_extraction.py:72-82`, `:132-165`).

**Required resolution.** Either serialize the complete typed logical-turn and CME continuation
records, or pin every awaiting/continuation session. Add exact TurnOutput-equivalence tests across
suspend→evict→rehydrate→resume and missing-parameter→evict→answer.

---

### R2-9 — [S1] Snapshot eligibility does not cover all behaviorally mutable state

§8.1 infers safety from three negative checks: no command-context object, no live child, and
JSON-projectable `workflow.context`. That is not a complete state model.

Concrete omitted mutable state:

- `Workflow.is_complete`, which is settable and included in the workflow's own `_to_dict()`
  (`workflow.py:283-290`, `:383-390`);
- `ChannelRuntime.active_conversation_id` and `stream_format` (`utils.py:644-658`);
- current conversation selection, which can differ from “last conversation”
  (`__main__.py:1439-1473`);
- repeated mutable-container identity: `{"a": shared, "b": shared}` is acyclic and JSON-native but
  restores as two independent lists;
- stale child IDs: `_children` can outlive a weakly registered child, so `bool(_children)` either
  pins forever or ignoring it loses a genuinely live descendant.

The active-conversation omission is corrupting, not cosmetic: suspend in older conversation 1,
evict, cold-create restores latest conversation 2, pending state overlays conversation-1 history,
then resume saves it under conversation 2. SSE likewise silently becomes NDJSON after eviction.

**Required resolution.** Replace inferred eligibility with an explicit, versioned
workflow/runtime snapshot capability that enumerates every supported mutable field and preserves
identity where required. Until that exists, pin states whose equivalence is not proven. Add
conversation-selection, stream-format, `is_complete`, alias, live-child, collected-child, and
grandchild tests.

---

### R2-10 — [S1] “Transactional eviction” is two independent commits

§8.7 writes changed workflow context and pending suspended state as separate operations. §8.8 makes
one disk file atomic; Redis likewise uses independent `SET`s. Neither creates one transaction across
the two namespaces.

**Crash schedule.**

1. A tool mutates application context and suspends on `ask_user`.
2. Context generation N+1 is published.
3. Pending-state publication fails or the process dies.
4. Restore combines context N+1 with pending N (or none).

The reverse ordering is also unsafe: durable pending trajectory can resume against rolled-back app
state and repeat effects. “Persist-before-pop” protects controlled eviction only; it does not make
the pair crash-consistent.

**Required resolution.** Use a generation-stamped composite snapshot with a commit record, or one
atomic combined record. Redis transactions require same-slot key design; disk requires a manifest
or directory-level generation protocol. Fail closed on generation mismatch. Fault-inject every
boundary and reconstruct in a fresh process.

---

### R2-11 — [S1] One hot channel still grows conversation memory per completed request

Goal 1 says completed requests must not produce monotonic process-memory growth. That is false for
a supported single-channel workload.

`append_conversation_turn()` unconditionally appends to `dspy.History.messages`
(`workflow_execution_context.py:434-451`). Only the last five entries are read for query refinement
(`:1046-1059`), but nothing trims older entries. History is cleared only by explicit
`/new_conversation`; it is persisted and restored wholesale.

A hot channel issuing bounded 450 KB direct actions therefore grows one request-sized conversation
entry per completed request even when:

- terminal executions are immediately removed;
- the session count is one;
- DSPy history/trace are disabled;
- no object context pins additional channels.

Incremental conversation persistence also rewrites an ever-growing turn list.

**Required resolution.** Define a per-conversation turn/byte window and archival/summarization
semantics. Add a same-channel soak with request-sized actions and assert in-memory history bytes
plateau independently of unique-channel eviction.

---

### R2-12 — [S1] Persist-all JSON silently creates a credential-at-rest boundary

The non-negotiable rule excludes one exact top-level key, `http_bearer_token`. The proposed policy
otherwise persists every JSON-native value from unrestricted `Workflow.context`, plus pending ReAct
trajectory, inputs, action log, and conversation turns, in plaintext.

That permits:

- `api_key`, `refresh_token`, cookies, signed URLs, database credentials;
- nested `http_bearer_token`;
- Authorization/Cookie values copied by application code;
- secrets supplied as an `ask_user` answer and retained in trajectory/history.

Before this change, arbitrary application context could remain process-local. The design silently
changes its data-classification and retention boundary.

**Required resolution.** Durable application state must be workflow-declared: allowlisted keys or
an explicit serializer with ephemeral/redacted paths. Unclassified state pins rather than persists.
Define encryption/access/retention expectations. Tests must inspect raw disk/Redis bytes for nested
and trajectory-carried secrets, not only the top-level bearer key.

---

### R2-13 — [S1] State records are not safely bound to channel, workflow, and deployment

The current disk mapping is non-injective:

```python
safe_id = channel_id.replace(os.sep, "_").replace("/", "_")
```

(`session_state_store.py:50-52`). A fresh reproduction showed `tenant/a` and `tenant_a` map to the
same file; after writing the second, loading either returned the second channel's state.

Revision 2 acknowledges this in §16.8 but makes the record durable and authoritative. Redis keys use
raw channel ID under a fixed prefix and have no workflow/deployment namespace
(`session_state_store.py:83-94`). The proposed envelope contains no channel hash, workflow identity,
or deployment epoch to detect a misplaced record.

**Required resolution.**

- use an injective/collision-resistant channel-key encoding;
- namespace by deployment and workflow fingerprint;
- bind record type, namespace, workflow, channel hash, and session incarnation inside the envelope
  and validate on read;
- require private disk directory/file modes and symlink-safe publication.

Test separator aliases, Unicode/control/oversized IDs, swapped records, two workflows sharing Redis,
and channel-ID reuse.

---

### R2-14 — [S1] Bearer refresh can overwrite a credential used by a running turn

Revision 2 changes `_update_http_bearer_token()` to mutate
`app_workflow.context` directly. The call occurs in `ensure_user_runtime_exists()` before endpoint
turn admission (`utils.py:286-304`).

**Race.**

1. Turn A runs under `runtime.lock` using token A.
2. Request B for the same channel reaches the dependency with token B.
3. The dependency writes token B into shared workflow context.
4. B is later rejected with 409, but A can already read B.

The fix repairs today's no-op updater by introducing cross-request credential contamination.

**Required resolution.** Carry request credentials on the submitted execution or in a
request/turn ContextVar. Install and clear them under the accepted turn's lifecycle immediately
before executor dispatch. A rejected request must not mutate shared state. Add a barrier test where
B is rejected while A reads the credential after B's dependency ran.

---

### R2-15 — [S2] DSPy's response cache remains a large process-lifetime retainer

Revision 2 disables DSPy diagnostic history and predictor trace but leaves DSPy's separate response
cache untouched.

Installed DSPy 3.2.1 defaults:

- in-memory LRU: **1,000,000 entries** (`dspy/clients/__init__.py:63-74`);
- disk cache: **30 GB** (`:16-25`, `:63-73`);
- every fastWorkflow `dspy.LM` defaults `cache=True` (`dspy/clients/lm.py:33-40`);
- `get_lm()` does not override it (`fastworkflow/utils/dspy_utils.py:42-69`).

A fresh probe patched `litellm_completion` below `LM.forward` and made four unique `Predict` calls
under the proposed policy. Result:

```text
DSPy response-cache entries: 4
GLOBAL_HISTORY:              0
LM history:                  0
settings.trace:              0
```

The §13 test's prescribed `LM.forward` stub bypasses the cache wrapper, which lives *inside*
`LM.forward`; it can prove history/trace are off while never exercising this retainer.

**Required resolution.** Make an explicit server cache decision: disable it, or set a justified
byte/count budget and include it in observability. Test below `LM.forward`, assert cache entries and
bytes, and add this structure to the RSS soak.

---

### R2-16 — [S2] Retained startup output is not collectable after runtime eviction

§6.5 claims startup output remains recoverable within the bounded window across live-runtime
eviction. The proposed access path does not exist:

- there is no implemented `GET /turns/{turn_key}` route;
- `InitializationRequest` carries no prior `turn_key`;
- the schema-2 startup envelope stores no `turn_key`;
- cold `/initialize` sees `startup.completed` and returns tokens only;
- revision 2 explicitly deletes the registry lookup from cold creation.

Therefore retaining the terminal execution does not make it reachable after the runtime holding
`startup_turn_key` is gone. §6.3's sentence “a caller that already holds a turn_key can collect”
describes Step-2 behavior that the design lists as a non-goal.

The count policy also has no minimum window: 21 startup completions in a burst evict the oldest
seconds after completion despite almost all of its 300-second TTL remaining. Raising TTL cannot
help when count binds.

**Required resolution.** Either implement authenticated polling/bounded replay, or remove the
cross-eviction recoverability claim and the unreachable retention. If retained, persist result
availability/turn key and size admission+retention from peak bursts, not average requests/hour.

---

### R2-17 — [S2] The durable-growth gate and its rollout dependency are non-executable

Revision 2 says §13.5 “now carries a durable-growth acceptance number.” It does not. The gate says
only “does not exceed the pre-registered figure,” with no figure, derivation, or plateau criterion.
Any positive bytes/1,000 requests can satisfy a chosen rate while cumulative storage grows forever.

The rollout dependency is also mis-scoped. `fix-6b4` is titled “Add TTL/reaper for orphaned
suspended-session blobs” and covers the pending namespace plus abandoned `ask_user` state. It does
not define lifecycle for workflow-context snapshots or conversation records.

At the motivating unique-channel rate, every snapshot is written and never read. Conversation
turns can duplicate request-sized data, so the design's 700 MB/day context-only arithmetic is not a
total-store bound.

**Required resolution.** Give the new namespace an actual lifecycle before enabling the lower
default broadly, or make persistence/lower-cap behavior opt-in. Gate total physical bytes by
namespace to a steady-state plateau, and add inspect/quarantine/delete/reset operations with
generation-safe channel reuse.

---

### R2-18 — [S2] Mixed-version rollout and downgrade can resurrect stale state

Schema 2 is called “forward-only,” while §14 treats raising `MAX_LIVE_SESSIONS` as rollback.
That is tuning, not binary compatibility.

Old `2.24.1` code:

- does not read the new workflow-context namespace on cold creation;
- can mutate application state while leaving the old context snapshot untouched;
- only warns on pending-state schema mismatch and then continues applying fields
  (`workflow_execution_context.py:367-400`).

Deployment sequence:

1. new code writes context generation A;
2. old code receives the channel, ignores A, starts from launch context, and produces state B;
3. new code later receives the channel and restores stale A as authoritative.

Rolling upgrades have the same issue if old and new pods share Redis.

**Required resolution.** Define a fleet protocol marker and either reject mixed versions or
provide dual-read/write migration. State whether downgrade after the first snapshot is unsupported.
Test new→old→new, concurrent N−1/new readers and writers, raw source-study blobs, and rollback after
real eviction.

---

### R2-19 — [S2] The DSPy policy can be disabled after the startup assertion

Synchronous `dspy.settings.configure()` claims the owner thread but leaves
`config_owner_async_task=None` (`settings.py:117-145`). The first async task on that same thread is
therefore allowed to configure global settings.

A fresh process demonstrated:

```text
after synchronous startup configure: disable_history=True,  max_trace_size=0
after first async-task configure:     disable_history=False, max_trace_size=10000
```

The one-time startup probe remains green in logs even after the policy is changed. Attribute
assignment to `dspy.settings` also calls `configure()` (`settings.py:87-91`).

**Required resolution.** Define and enforce exclusive server ownership after readiness, or pin an
async owner before accepting traffic and reject subsequent global changes. Add a negative
post-readiness override test. Also state the supported worker/reload model: spawned Uvicorn workers
do not inherit a parent's globals, so each process needs the bootstrap or workers/reload must be
refused.

---

### R2-20 — [S2] The documented env default can defeat the container override

`default=None` allows `os.environ` only when `_env_vars` does not contain the key. Workflow env-file
values still win (`fastworkflow/__init__.py:215-219`).

Revision 2 also requires documenting `MAX_LIVE_SESSIONS=50` in
`fastworkflow/examples/fastworkflow.env`. Existing defaults in that file are active assignments,
not commented examples. If the new line follows that pattern and users copy it, Kubernetes/Docker
`MAX_LIVE_SESSIONS=500` is silently ignored and 50 wins.

The R4 fix therefore works only if the documentation does not install the default it documents.

**Required resolution.** Make the example commented, or give this operator control explicit
OS-first precedence in a dedicated resolver. Test default, OS-only, file-only, conflict, blank,
invalid, zero, and negative values through the real entrypoint. Log the effective value **and its
source**.

---

### R2-21 — [S2] The shutdown union does not make timeout or admission quiescent

Revision 2 correctly changes shutdown's busy test to the union predicate. Current shutdown,
however, waits at most 30 seconds and then always finalizes conversations and closes every runtime
(`__main__.py:289-342`).

After the deadline, the union can truthfully report remaining work while shutdown still snapshots
and closes it. Detached streaming executor work from R2-3 makes this especially unsafe. There is
also no atomic “admission closed” state: a turn can be registered after an empty scan unless host
ordering is assumed.

**Required resolution.**

1. close admission under registry/manager coordination;
2. drain queued/running/leased work;
3. after timeout, never snapshot or close a still-busy runtime;
4. define whether queued work is cancelled before execution or left to host termination.

Test queued and running work beyond a short deadline, a detached stream worker, and submission
racing the final empty scan.

---

### R2-22 — [S3] The verification gates do not prove a plateau or production behavior

The plan is structurally stronger than revision 1, but its headline proof remains non-falsifiable:

- `0.05 MB/request` permits about **78 MB/day** at 65 requests/hour; that is a leak budget, not a
  plateau;
- “a repeat run shows the same plateau” supplies no independent replicate count, confidence
  interval, or plateau definition;
- forced `gc.collect()` can hide production GC/allocator behavior;
- RSS alone cannot distinguish live private memory from allocator high-water; USS/cgroup memory are
  absent;
- the in-process ASGI harness includes client and server in one process and can bypass the dedicated
  pre-event-loop entrypoint where DSPy controls must be installed;
- durable growth, eviction p95, and scheduler-delay gates still have no numeric limits;
- the soak omits conversation-history and DSPy-cache counts/bytes, the two round-2 retainers;
- arm B is explicitly expected to violate the slope and is not a shipping gate.

**Required resolution.** Run the actual CLI/Uvicorn server in a separate fresh process; report RSS,
USS, cgroup memory, live-heap diagnostics, cache/history/store sizes, and natural versus forced-GC
samples. Use multiple fresh-process replicates and an upper confidence bound compatible with zero
or with a stated worker-recycle budget. Put every numeric threshold in the design before treatment
is measured.

---

## 5. Cross-cutting conclusion

Revision 2's central mistake is no longer “a cache might lose `workflow.context`.” It is treating a
new **session checkpoint protocol** as if it were still a cache implementation detail.

The protocol now decides:

- which application state is durable;
- which credentials cross the at-rest boundary;
- whether startup may replay;
- which conversation receives a resumed turn;
- whether two records form one state generation;
- what old/new workers may read;
- when a live runtime may be detached.

Those are system-of-record semantics. Negative eligibility checks plus two JSON files are not enough
to establish them.

### Recommended decomposition

Do not hold the lower session default hostage to an all-or-nothing five-fix package. Split the work:

1. **Universal, independently useful memory slice**
   - bound terminal registry records;
   - use weak creation locks;
   - disable DSPy history/trace;
   - make an explicit DSPy response-cache decision;
   - add same-channel conversation-history bounds.
2. **Session checkpoint slice**
   - runtime leases and unified streaming lifecycle;
   - explicit workflow/runtime snapshot capability;
   - complete logical-turn/CME continuation state;
   - versioned identity-bound composite commit;
   - security classification and lifecycle;
   - mixed-version/rollback protocol.
3. **Only then lower `MAX_LIVE_SESSIONS`.**

This preserves the low-risk memory wins without making an immature persistence protocol the price
of shipping them.

### Minimum gate before implementation

At minimum, resolve R2-1 through R2-14 in the design of record, then update:

- the invariant list;
- the record schema;
- the implementation map;
- deterministic tests;
- rollout/rollback;
- the traceability table.

Per the repository's design-review process, each accepted finding should receive its own review-only
beads child and explicit decision. No code should land while the design still calls R2-6 or R2-7
implementable: both are contradicted directly by the pinned tree.

---

## 6. Runtime evidence captured in round 2

These are small mechanism checks, not treatment benchmarks.

### 6.1 DSPy cache survives the proposed history/trace policy

Environment: `.venv/bin/python`, DSPy 3.2.1. Four unique `Predict` calls, fake completion patched
below `LM.forward`, no network call:

```text
cache_entries 4
global_history 0
lm_history 0
trace 0
```

### 6.2 Synchronous DSPy configuration does not claim async ownership

```text
after_sync  True 0     async_owner=None
after_async False 10000 async_owner=set
```

### 6.3 Disk channel IDs collide

```text
tenant/a  -> tenant_a_pending.json
tenant_a  -> tenant_a_pending.json

after writing tenant_a:
load(tenant/a) = {"owner": "underscore"}
load(tenant_a) = {"owner": "underscore"}
```

### 6.4 Evidence not reproduced

- source-study baseline/treatment RSS slopes;
- 2,000-request production soak;
- Redis failure/transaction behavior;
- proposed serializer, lease, migration, and checkpoint code (not implemented);
- total physical durable growth under the production payload.

Those remain open evidence obligations rather than implicit support for or against the mechanism.

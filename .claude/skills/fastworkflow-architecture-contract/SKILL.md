---
name: fastworkflow-architecture-contract
description: >
  Load this skill when you need to understand or preserve fastWorkflow's load-bearing
  runtime design before changing core code: anything touching WorkflowExecutionContext,
  ChatSession, the CME/wildcard NLU pipeline, ask_user suspension/resume, AskUserSuspend /
  CommandCancelledError, SessionStateStore, TurnResult/TurnOutput/process_turn, the
  TurnRegistry turns engine, context models, or session serialization. Trigger phrases:
  "why is this a BaseException", "Topology A vs Topology B", "where does the turn get
  finalized", "is it safe to run two turns on one channel", "what breaks if I rename a
  command", "what is the cme workflow", "context_hierarchy_model". Do NOT load it for
  step-by-step debugging of a failure (use fastworkflow-debugging-playbook), for running
  the server/CLI (fastworkflow-run-and-operate), for model/threshold details of the NLU
  stack (fastworkflow-nlu-pipeline-reference), or for env-var/flag lookups
  (fastworkflow-config-and-flags).
---

# fastWorkflow Architecture Contract

The design decisions that must survive any refactor, the invariants the code assumes
(some enforced, some merely assumed — labeled), and the known-weak points, each with
file:line evidence. Facts verified 2026-07-09 against v2.22.2 (commit c33b9a5).

## When to use / when NOT to use

| Situation | Skill |
|---|---|
| Changing core runtime code; reviewing a PR that touches WEC/ChatSession/turns; "why is it built this way?" | THIS skill |
| A runtime/build/train failure to triage right now | fastworkflow-debugging-playbook |
| Historical investigations, reverts, dead ends | fastworkflow-failure-archaeology |
| Intent-model internals, thresholds, DSPy signatures | fastworkflow-nlu-pipeline-reference |
| Env vars and CLI flags | fastworkflow-config-and-flags |
| Running the CLI / FastAPI server, artifact locations | fastworkflow-run-and-operate |
| What change classes are gated and why | fastworkflow-change-control |
| tau-bench / tau2 mechanics and the reliability campaign | fastworkflow-taubench-reference, tau2-reliability-campaign |

## Vocabulary (defined once)

| Term | Meaning |
|---|---|
| **WEC** | `WorkflowExecutionContext` (`fastworkflow/workflow_execution_context.py`) — the transport-free, synchronous execution core. One per session. |
| **ChatSession** | CLI/REPL wrapper that *composes* a WEC and adds Queues + a daemon `ChatWorker` thread (`fastworkflow/chat_session.py:27-48`). |
| **Topology A** | ChatSession mode: `ask_user` **blocks** a worker thread on `user_message_queue.get()` until a human answers (`workflow_agent.py:354-385`). |
| **Topology B** | WEC mode (FastAPI, embedders): no queues; `ask_user` **suspends** the agent trajectory in memory and the call returns; the next message resumes it. |
| **CME** | The *command metadata extraction* internal workflow at `fastworkflow/_workflows/command_metadata_extraction/` — a real fastWorkflow workflow that IS the NLU pipeline. |
| **wildcard** | The pseudo-command `_workflows/command_metadata_extraction/_commands/wildcard.py` whose single `ResponseGenerator` runs intent detection + parameter extraction. |
| **Logical turn** | One user interaction end-to-end, across any number of ask_user suspensions. One turn = one `turn_key` = one record (`turn.py:240-247`). |
| **Embedder** | Any host that owns a WEC's lifecycle (FastAPI server, ChatSession, a future Slack bot). |
| **DSPy ReAct loop** | The forked reason/act agent loop in `fastworkflow/utils/react.py` (`fastWorkflowReAct`), a vendored modification of DSPy's ReAct. |
| **[A#] / [R#] / [X#] / Topic-n** | Traceability tags into the TurnResult docs: design-doc Amendment / adversarial-review finding / architecture-review finding / feedback-session topic (`docs/turn_result_design*.md`; process in `fastworkflow-change-control`). |

## 1. The three-phase shape

```
Build-time  →  Train-time  →  Run-time
```

| Phase | Lives in | Input | Output |
|---|---|---|---|
| Build | `fastworkflow/build/` (AST introspection: `ast_class_extractor.py`, `command_file_generator.py`, `context_model_generator.py`, …) | Your Python app | `_commands/*.py` + `context_inheritance_model.json` |
| Train | `fastworkflow/train/` (`__main__.py`, `generate_synthetic.py`) + `fastworkflow/model_pipeline_training.py` | `_commands/` + utterances | `___command_info/` (intent models, thresholds) inside the workflow dir |
| Run | `fastworkflow/workflow_execution_context.py`, `command_executor.py`, the CME workflow, `run_fastapi_mcp/` | user message | `TurnOutput` / `CommandOutput` |

Doc-rot warning: CLAUDE.md says intent models are trained "via scikit-learn". False —
training is torch/transformers (`model_pipeline_training.py:3-9` imports
`transformers.AutoModelForSequenceClassification`, `torch`, `AdamW`); sklearn supplies only
`LabelEncoder`/`train_test_split`/metrics/PCA. Trust the code.

## 2. The run-time core, precisely

### 2.1 WEC vs ChatSession: engine and chassis

- **WEC is the engine.** It owns exactly one thing: "execute one message correctly,
  synchronously" (`workflow_execution_context.py:1-16` module docstring). It holds the
  CME workflow, the bound app workflow (`bind_app_workflow`, `:428-431`), agent state,
  conversation history, and the turn accumulator. No threads, no queues, no timeouts.
- **ChatSession is the CLI chassis (Topology A).** `ChatSession.__init__` constructs a
  WEC (`chat_session.py:118-121`); `keep_alive` root workflows get a daemon `ChatWorker`
  thread running `_run_workflow_loop` (`chat_session.py:27-48`); child workflows
  (`parent_workflow_id` set) run on the caller's thread with `keep_alive` forced False
  (`chat_session.py:189`). Queues are *injected into* the WEC via
  `set_transport_queues` (`workflow_execution_context.py:246-257`).
- **FastAPI is the server chassis (Topology B).** One `ChannelRuntime` per channel holds a
  live WEC + an `asyncio.Lock` (`run_fastapi_mcp/utils.py:645-664`); blocking WEC work runs
  in `loop.run_in_executor` threads; a `ContextVar` tuple-stack isolates the active workflow
  per thread/asyncio-task (`fastworkflow/active_workflow.py:16-20`) — this is what makes
  many concurrent channels safe in one process.

**The embedder contract** (verbatim intent from `WEC_learning_checklist.md:56-62` — anyone
adding a new transport owes the WEC all five):
1. thread-pool offload (WEC is blocking), 2. session create/bind/lookup per channel,
3. cleanup of abandoned sessions (`close()`, `cancel_pending()`), 4. per-channel lock +
409-busy guard — **the WEC is not safe under two overlapping turns on one session**,
5. timeout enforcement + persistence/rehydration of suspended state.

### 2.2 The CME workflow and the wildcard pseudo-command

Every deterministic command execution funnels through
`CommandExecutor.invoke_command` (`command_executor.py:27-100`), which executes
`Action(command_name="wildcard")` **on the CME workflow** (`command_executor.py:41-46`).
The wildcard `ResponseGenerator` (`_workflows/command_metadata_extraction/_commands/wildcard.py:27-179`)
runs the whole NLU state machine:

1. **Intent detection** — `CommandNamePrediction.predict` (`intent_detection.py`):
   exact first-token match → Levenshtein fuzzy match (threshold 0.3) → embedding
   cache match (0.85 cosine, `cache_matching.py`) → transformer classifier
   (`intent_detection.py:99-134`). Ambiguity flips the CME context's
   `NLU_Pipeline_Stage` (enum at `fastworkflow/__init__.py:13-18`).
2. **Parameter extraction** — `ParameterExtraction.extract` (`wildcard.py:150-151`);
   XML-regex first, DSPy LLM fallback (see fastworkflow-nlu-pipeline-reference).
3. On success it returns artifacts `{"command_name", "cmd_parameters"}`, which
   `invoke_command` unpacks (`command_executor.py:56-57`) to run the app command's own
   `ResponseGenerator` (**stage 3**, `command_executor.py:86-88`).

So stages 1+2 of the 3-stage pipeline are literally ONE ResponseGenerator on an internal
workflow. **Why it is a workflow and not plain framework code** (verified mechanics):
- The NLU state machine persists in `cme_workflow.context["NLU_Pipeline_Stage"]` between
  turns — workflow context IS the state store; `Workflow.end_command_processing` resets it
  (`workflow.py:291-303`).
- Core commands every app inherits (`ErrorCorrection/abort`, `ErrorCorrection/you_misunderstood`,
  `IntentDetection/what_can_i_do`, `go_up`, `reset_context`, `what_is_current_context` —
  `ls fastworkflow/_workflows/command_metadata_extraction/_commands/`) execute via the exact
  same `CommandExecutor.perform_action` machinery as app commands (`wildcard.py:77-89`).
- The CME has its own trainable intent models (`_workflows/command_metadata_extraction/___command_info/`),
  so the NLU front door itself is trained with the same pipeline it serves.

**OPEN QUESTION (unresolved, do not assume either way):** whether CME-as-workflow is
load-bearing beyond core-command reuse and stage persistence — e.g. planned trainability/
self-improvement of the CME itself. No doc records the ruling. If you are about to flatten
the CME into plain functions, stop and ask Dhar first (see fastworkflow-change-control).

`wildcard` is excluded from user-visible command lists (`command_routing.py:79-87`,
`get_command_names` filters it), and `/`-prefixed messages set the CME context flag
`is_assistant_mode_command` forcing the deterministic path
(`workflow_execution_context.py:616-632`).

### 2.3 Suspension: BaseException tunneling through the ReAct loop

Two deliberate `BaseException` subclasses:

| Exception | Raised where | Caught where | Meaning |
|---|---|---|---|
| `AskUserSuspend` (`utils/react.py:18-28`) | the `ask_user` tool closure when no `user_message_queue` exists (`workflow_agent.py:444-457`) | `fastWorkflowReAct._run_loop` (`react.py:250-262`) | Topology-B suspend: stash `{trajectory, idx, input_args, max_iters, clarification}` in `self._suspended`, return `Prediction(suspended=True)` |
| `CommandCancelledError` (`workflow_execution_context.py:42-49`) | e.g. Topology-A blocking ask_user reached with no queue (`workflow_agent.py:362-363`) | `WEC._execute_message` (`:516-518`) | convert to a failed `CommandOutput` |

**WHY BaseException:** the ReAct loop wraps every tool call in `except Exception`
(`react.py:263-266`) to convert tool failures into observations the agent can react to.
A suspension signal must tunnel *through* that handler untouched — subclassing
`BaseException` makes `except Exception` structurally unable to swallow it (stated in both
docstrings, `react.py:22-24`, `workflow_execution_context.py:47-48`; also
`workflow_agent.py:147-151` re-raises `CommandCancelledError` explicitly before its own
`except Exception`). If you "fix" these to inherit `Exception`, ask_user silently turns
into an agent-visible error string and suspension dies. Do not.

Resume: `WEC._execute_message` with `_awaiting_user=True` routes the next message to
`_resume_agent_message` (`:512-513`, `:817-838`) → `fastWorkflowReAct.resume(observation)`
(`react.py:171-196`) which appends the answer as `observation_<idx>` and continues the
same trajectory. Full sequence: [references/runtime-walkthrough.md](references/runtime-walkthrough.md).

### 2.4 SessionStateStore: serialize-before-evict

`WEC.serialize_state` (`workflow_execution_context.py:312-356`) exports a JSON blob:
`schema_version` (=1, `session_state_store.py:20`), `awaiting_user`, the suspended ReAct
blob (`fastWorkflowReAct.export_suspended`, `react.py:120-131` — includes
`iteration_counter`), `nlu_stage`, `current_command_context_name` (name only — see
weak-point fix-cgs), `action_log`, and conversation turns.
`apply_serialized_state` (`:358-405`) restores it.

Backends (`session_state_store.py:112-142`, factory keyed on env `SESSION_STATE_STORE`):
disk = one JSON per channel under `SPEEDDICT_FOLDERNAME/channel_session_state`
(default); redis = `RedisSessionStateStore` for multi-pod.

**Serialize-before-evict invariant:** the FastAPI LRU (`max_live_sessions=2000`,
`utils.py:677`) never closes a busy channel's ctx, and saves an `awaiting_user` session's
state to the store *before* `close()` (`utils.py:715-745`). Same rule after every turn:
persist-or-clear happens inside the turn-completion path before DONE
(`turns.py:246-266`, `:302-306`).

### 2.5 Turn accumulator, finalize chokepoint, TurnResult/TurnOutput

- `_begin_turn` (`workflow_execution_context.py:145-169`) atomically resets the
  accumulator and mints a `turn_key` (`turn.py:93-109`, sortable
  `YYYYMMDDTHHMMSS.ffffffZ-<uuid12>`). **Resume never resets** [A30.2]: a message during
  `awaiting_user` skips `_begin_turn` and continues the same logical turn (`:504-506`).
- Every command execution appends via `append_turn_output` (`:171-174`, which also runs
  the warn-only artifact serializability check, `turn.py:147-165`). ask_user exchanges are
  role-inverted `CommandOutput` entries [A7]: `command_parameters` = the agent's QUESTION,
  the response = the user's ANSWER (`""` + `success=False` while unanswered;
  `duration_ms` on completion = user think time [A38]) (`:176-212`, `__init__.py:67-118`).
- **Finalize chokepoint:** `_finalize_agent_output` (`:753-802`) synthesizes the single
  answer `CommandOutput` and — Topic 5 — merges every artifact-bearing tool response of
  the turn into that one response's `artifacts` dict, suffixing `_<n>` only on key
  collision (`:786-791`, `turn.py:39-57`). This is the ONLY place turn artifacts reach
  the user-facing agent path; bypass it and you recreate the original payload-loss bug.
- **Types:** `TurnResult` (internal system-of-record, `turn.py:240-264`) composes the
  public `TurnOutput` (`turn.py:168-237`) returned by `process_turn`
  (`workflow_execution_context.py:484-495`). `TurnOutput.success` is *computed* as
  `all(command_outputs succeeded)` — deliberately **orthogonal** to `status`
  (`TurnStatus`, `turn.py:83-90`) and `failure_reason` (agent exhaustion at
  `max_iters=25`, `workflow_agent.py:387`, yields `status=FAILED`,
  `failure_reason="max_iters_exhausted"`, `:545-551`).
- `process_message` is **deprecated** (DeprecationWarning, `:469-482`) but both transports
  still route through the shared `_execute_message`; `process_turn` and
  `process_action_turn` (`:595-610`, added for the turns engine) are the current API.
  `TurnResult.conversation_id`/`ordinal` are still `None` — no durable consumer exists yet
  (section 5 below).

## 3. Invariants — and what breaks when violated

| # | Invariant | Enforced? | Evidence | Violation consequence |
|---|---|---|---|---|
| I1 | **Single writer per channel** (one pod/process serves a channel) | **ASSUMED, NOT enforced** — R27 ruling (`docs/turn_result_design_review.md:1272-1286`), open **fix-5ka** (P1) | `session_state_store.py:4-6` docstring; bd show fix-5ka | Two pods on one channel: RocksDB lock conflict/corruption, pending-state last-writer-wins, interleaved ordinals |
| I2 | **`iteration_counter <= 0` means "originated from the user"** — dual-purposed | Convention only, self-acknowledged fragile | Set to -1 on every ask_user (`workflow_agent.py:450-454`) and on Topology-B resume (`workflow_execution_context.py:823`); `resume` increments it back to 0 (`react.py:184`); read at `workflow_agent.py:424-429`, published as `workflow.context["is_user_command"]` for command authors' `validate_extracted_parameters` hooks. **No in-tree command reads the flag today (verified: `grep -rn is_user_command fastworkflow/` finds only workflow_agent.py)** | Refactoring counter semantics silently changes the parameter-validation trust signal AND grants extra iterations (exhaustion check is `iteration_counter >= max_iters`, `react.py:273`) |
| I3 | **Command names are unique process-wide** | Structurally assumed | `_GLOBAL_COMMAND_CLASS_CACHE` keyed ONLY `command_name:module_type`, no workflow path (`command_routing.py:62`, `:89-107`, docstring: "command names are unique") | Two workflows in one process with a same-named command: whichever loads first wins for BOTH — silent cross-workflow class leakage |
| I4 | **Artifact merge happens only at finalize** | By construction | `_finalize_agent_output:786-791` is the sole merge site | Merging earlier double-surfaces artifacts; skipping it drops tool artifacts (the original v2.20 xray bug) |
| I5 | **Embedders provide lock + 409; WEC is not safe under overlapping turns** | Enforced per-embedder, not in WEC | Contract: `WEC_learning_checklist.md:56-62`; FastAPI: per-channel `asyncio.Lock` + registry pointer as 409 source (`turns.py:14-28`, `:155-162`) | Interleaved turns corrupt the accumulator/suspension state mid-mutation |
| I6 | **Resume never resets the turn** [A30.2] | Enforced | `workflow_execution_context.py:504-506` | Resetting on resume splits one logical turn into two records and loses suspended_ms/ask_user pairing |
| I7 | **TurnRegistry construction-order contract**: build execution + insert pointer BEFORE launching the task; pointer (not `lock.locked()`) is 409 truth; wait-or-defer never wait-or-abort; persist before DONE | Enforced | `turns.py:13-28` (header), `:171-214`, `:269-318`, `:354-359` (`asyncio.shield`) | Reordering lets a waiter observe a half-built execution (no `done_event`); using the lock as busy-truth re-opens the v2.22.0 504-race (double execution) |
| I8 | **BaseException for suspension signals** (2.3) | Enforced by class hierarchy | `react.py:18`, `workflow_execution_context.py:42` | Downgrade to Exception ⇒ ask_user becomes a swallowed tool error |
| I9 | `workflow_id_str` XOR `parent_workflow_id`; `root_command_context` set exactly once; `Workflow.create` returns the existing live registry object per id | Enforced (raises) | `workflow.py:82-83`, `:211-217`, `:101-104`; weakref registry `workflow.py:41-45` | n/a (raises) — but note the registry is weak: dropping all strong refs evicts the session |

## 4. The dual context model (docs omit half of it)

fastWorkflow has **two** context model files plus optional per-context callback classes.
Only the first is documented in CLAUDE.md / `.claude/rules/command-authoring.md`:

1. **`_commands/context_inheritance_model.json`** — command inheritance. Loader:
   `command_context_model.py:70-140` (`'base'` key required per entry; `'/'` lists are
   filesystem-derived, and a `'/'` key in the JSON is *rejected* at `:93-97` — the
   command-authoring rule's claim that entries have "two possible keys" describes the
   merged in-memory model, not the file).
2. **`context_hierarchy_model.json`** (workflow ROOT, not `_commands/`) — parent/containment
   ancestry with cycle detection. Loader: `command_context_model.py:151-160`
   (`_load_context_hierarchy`), resolution + cycle error `:162-216`; the per-entry key is
   `"parent"` (list of context names).
3. **`_<ContextName>.py` Context callback classes** — a file like
   `_commands/TodoItem/_TodoItem.py` exposing `class Context` with `get_parent`
   (classmethod). Loaded via `ModuleType.CONTEXT_CLASS` (`__init__.py:144-149`), resolved
   by `CommandContextModel.get_context_class` (`command_context_model.py:272-290`), and
   consumed by `Workflow.get_parent` for context navigation (`workflow.py:219-234`) and
   `current_command_context_displayname` (`:187-196`). The wildcard command walks parents
   when a command isn't found in the current context (`wildcard.py:99-107`).

**In-tree proof:** `tests/todo_list_workflow/` has all three —
`context_hierarchy_model.json` (TodoItem→TodoList→TodoListManager→`*`),
`_commands/TodoItem/_TodoItem.py` (`get_parent` returns `command_context_object.parent`),
and `_commands/context_inheritance_model.json` (TodoList inherits TodoItem's commands).
This is real, load-bearing runtime behavior, used by 10+ test files. It is a documented-
nowhere-else feature; PRD at `fastworkflow/docs/context_modules_prd.txt`.

Also note: the CME workflow dir contains a root-level `command_context_model.json` that no
package code reads (`grep -rn 'command_context_model.json' fastworkflow/ --include='*.py'`
returns nothing) — candidate stale file, do not imitate it.

## 5. Known-weak points (stated plainly)

| Weak point | Evidence | Tracking |
|---|---|---|
| `apply_serialized_state` restores only the context *name*; navigation depth is lost — a session suspended inside a nested context resumes at the wrong depth (only a debug log fires) | `workflow_execution_context.py:393-405` | **fix-cgs** (P3, open) |
| TurnRegistry TTL eviction is a no-op (`ttl_expires_at` never set) ⇒ terminal executions accumulate in `_by_key`; trace collection is a destructive queue drain | `turns.py:226-243`, `:293-296` | **fix-85g** Step 2 (.9-.13 open) |
| `schema_version=1` with **no migration path** — mismatch only logs a warning and ploughs on | `workflow_execution_context.py:360-363` | **fix-4od** (P3, open) |
| No TTL/reaper for orphaned suspended-session blobs — eviction saves, nothing ever deletes abandoned ones | `session_state_store.py` (no delete-by-age API exists) | **fix-6b4** (P2, open) |
| Latent `UnboundLocalError` (a `NameError` subclass): `get_last_conversation_id` assigns `db` inside `try` but `finally` calls `db.close()` — if `Rdict(...)` raises, the finally masks the real error | `run_fastapi_mcp/conversation_store.py:47-54` | no issue filed (verified 2026-07-09) |
| Dead `ChatSessionDescriptor`: the set-once guard never fires — `from .chat_session import ChatSession` (`__init__.py:269`) rebinds `fastworkflow.chat_session` to the SUBMODULE, shadowing the descriptor instance (`__init__.py:151-164`). Verified live: `type(fastworkflow.chat_session)` is `module`; double assignment succeeds | `fastworkflow/__init__.py:151-164`, `:269` | no issue filed |
| CLI hang on agent-mode ask_user: Topology-A clarification enqueued without the trace sentinel the CLI drains for (flagged "needs runtime verification" — treat as unreproduced) | `workflow_agent.py:371-372` vs sentinel sites `workflow_execution_context.py:656-658` | **fix-5fv** (P2, open) |
| `/invoke_agent_stream` bypasses the TurnRegistry (guards with `lock.locked()`, no idempotency/202) — inconsistent with I7 | `run_fastapi_mcp/__main__.py:963` (grep `locked()`) | noted in fix-85g scope discussion |

## 6. The v3.0 boundary: designed but UNBUILT

`docs/turn_result_design_final.md` is the authoritative spec ("where this conflicts with
any of those, this document wins"). Everything below is **roadmap, not code** — verified
absent on main (`ls fastworkflow/turn_accumulator.py fastworkflow/turn_serializer.py
fastworkflow/stores fastworkflow/metrics.py` → all missing):

- `ConversationTurnStore` + `ArtifactBlobStore` + `stores/` package + `TurnSerializer`
  (module layout: final.md §15).
- The Redis keyspace with hash-tagged keys and write-time ZSET indexes
  (`fw:turn:{ch/cv}:<turn_key>`, `fw:turnidx:{ch/cv}` — final.md §5).
- The **wire hard-break** (final.md §14, v3.0 release train): endpoints return
  `TurnOutput.model_dump()`, MCP `isError = not success`, `command_responses` collapses to
  singular `command_response` (the accepting shim already exists:
  `__init__.py:85-97`), `process_message` REMOVED, `run_fastapi_mcp/conversation_store.py`
  deleted, mixed fleets forbidden, rollback loses 3.0-era conversations.
- Detail + what this means for changes you make today:
  [references/v3-boundary.md](references/v3-boundary.md).

No beads epic for v3.0 exists (fix-vof was review-only and is fully closed); the endpoint
response-shape cutover is separately tracked as open epic **fix-qtq** (9 open children).
Whether v3.0 is scheduled at all is an open question — do not claim it is.

## 7. Discipline rules that bind architecture work

- **Never `git commit`/`push` without the developer'sexplicit request in that turn** (rule
  established 2026-07-08 after a private doc was auto-pushed to the public repo).
- **One bd write at a time; verify `.beads/issues.jsonl` after each.** `bd close --reason`
  silently failed to persist on an older bd (2026-06-11); it persists on 1.1.2, retested
  2026-08-13 — see change-control Rule 2.
- **Never let tests/experiments write into `fastworkflow/examples/*/___command_info`**
  (fix-0hb incident, commit fa97b48) — train into temp copies.
- Big core changes follow the paper-first pattern of the TurnResult saga (design doc →
  adversarial review → final spec → minimal slice); see fastworkflow-change-control.

## Provenance and maintenance

Facts verified 2026-07-09 against v2.22.2 (commit c33b9a5). All file:line cites were read
directly; behavioral claims (descriptor shadowing) were verified by executing code.
Re-verification one-liners for every volatile fact:

| Fact | Re-verify with |
|---|---|
| Version / commit | `grep '^version' pyproject.toml; git log --oneline -1` |
| BaseException subclasses + line anchors | `grep -n 'class CommandCancelledError\|class AskUserSuspend' fastworkflow/workflow_execution_context.py fastworkflow/utils/react.py` |
| wildcard dispatch lines | `grep -n "command_name = \"wildcard\"\|command_name = 'wildcard'\|wildcard" fastworkflow/command_executor.py` |
| iteration_counter sites | `grep -rn iteration_counter fastworkflow/ --include='*.py' \| grep -v __pycache__` |
| is_user_command still has no in-tree reader | `grep -rn is_user_command fastworkflow/ --include='*.py' \| grep -v __pycache__` |
| Global class cache key shape | `grep -n '_GLOBAL_COMMAND_CLASS_CACHE\|cache_key = ' fastworkflow/command_routing.py` |
| SCHEMA_VERSION and store backends | `grep -n 'SCHEMA_VERSION\|SESSION_STATE_STORE' fastworkflow/session_state_store.py` |
| Serialize-before-evict + LRU cap | `grep -n 'max_live_sessions\|_evict_oldest_if_needed' fastworkflow/run_fastapi_mcp/utils.py` |
| TurnRegistry invariants + TTL no-op | `sed -n '13,28p;226,243p' fastworkflow/run_fastapi_mcp/turns.py` |
| process_message deprecation | `grep -n DeprecationWarning fastworkflow/workflow_execution_context.py` |
| v3.0 modules still absent | `ls fastworkflow/turn_accumulator.py fastworkflow/stores 2>&1` |
| Dual context model loaders | `grep -n 'context_hierarchy_model.json\|context_inheritance_model.json' fastworkflow/command_context_model.py` |
| Context callback example | `cat tests/todo_list_workflow/_commands/TodoItem/_TodoItem.py` |
| Descriptor shadowing | `python -c "import fastworkflow; print(type(fastworkflow.chat_session))"` |
| conversation_store finally bug | `sed -n '47,54p' fastworkflow/run_fastapi_mcp/conversation_store.py` |
| Open issue statuses (fix-5ka/cgs/4od/6b4/5fv/85g/qtq/7kp) | `bd list --status open \| grep -E 'fix-(5ka\|cgs\|4od\|6b4\|5fv\|85g\|qtq\|7kp)'` |
| max_iters=25 default | `grep -n 'max_iters: int = 25' fastworkflow/workflow_agent.py` |
| Training stack is torch/transformers | `sed -n '1,10p' fastworkflow/model_pipeline_training.py` |

# Run-time walkthrough: one message, file:line by file:line

Companion to SKILL.md section 2. Facts verified 2026-07-09 against v2.22.2 (commit c33b9a5).
All paths relative to repo root; all line numbers from that commit — re-verify with the
one-liners in SKILL.md "Provenance and maintenance" before trusting them after any core change.

## 1. Entry: `_execute_message`

`WorkflowExecutionContext._execute_message` (`fastworkflow/workflow_execution_context.py:497-522`):

```
if self._app_workflow is None: raise RuntimeError(...)        # :499-502
if not self._awaiting_user: self._begin_turn(message)         # :504-506  (resume never resets)
self.push_active_workflow(self._app_workflow)                 # :508  ContextVar stack
try:
    self._prepare_message_routing(message)                    # :510  '/' prefix flag
    if self._should_run_agent_for_message(message):           # :511  agent unless '/' flag
        if self._awaiting_user: return self._resume_agent_message(message)   # :512-513
        return self._process_agent_message(message)           # :514
    return self._process_message(message)                     # :515  deterministic path
except CommandCancelledError as exc:                          # :516-518
    self._reset_agent_suspension(); return self._command_cancelled_output(str(exc))
finally:
    self.pop_active_workflow(); self._app_workflow.flush()    # :519-522
```

Both public entry points share it:
- `process_message` (`:469-482`) — DEPRECATED, emits `DeprecationWarning`, returns `CommandOutput`.
- `process_turn` (`:484-495`) — current API; wraps the same call in `_build_turn_result`
  (`:524-576`) and returns `turn_result.turn_output` (public `TurnOutput`).
- `process_action` / `process_action_turn` (`:578-610`) — direct-action mirror; each direct
  action is its own logical turn [A30].

## 2. Deterministic path (assistant mode / `/`-prefixed)

`_process_message` (`:844-909`):
1. optional trace event to `command_trace_queue` (CLI only)
2. `CommandExecutor.invoke_command(self, message)` (`:859`) — see section 3
3. stamps `started_at`/`duration_ms`, `append_turn_output` (`:860-864`)
4. appends an `assistant_mode_command` entry to conversation history (`:898-904`)
5. `_maybe_enqueue_output` — enqueues only when failed OR `keep_alive` (`:649-654`);
   `_maybe_enqueue_trace_sentinel` puts `None` on the trace queue (`:656-658`).

## 3. Stage 1+2: `invoke_command` → wildcard on the CME workflow

`CommandExecutor.invoke_command` (`fastworkflow/command_executor.py:27-100`):
- empty command → friendly nudge (`:32-39`)
- `perform_action(chat_session.cme_workflow, Action(command_name="wildcard", command=command))` (`:41-46`)
- if `command_output.command_handled` → a CME/core command already ran; clear
  `is_assistant_mode_command` and return (`:48-52`)
- if `not success` → NLU error surfaced to caller (ambiguity / param-extraction error) (`:53-54`)
- else unpack `artifacts["command_name"]` + `artifacts["cmd_parameters"]` (`:56-57`) and run
  the app command's ResponseGenerator — **stage 3** (`:67-88`), then stamp
  `workflow_name/context/command_name/command_parameters` (`:90-94`).

Inside wildcard's ResponseGenerator
(`fastworkflow/_workflows/command_metadata_extraction/_commands/wildcard.py:27-179`):

| Step | Lines | Behavior |
|---|---|---|
| Predict command name | `:39-40` | `CommandNamePrediction.predict(context_name, command, nlu_stage)` |
| Ambiguity | `:42-57` | set stage `INTENT_AMBIGUITY_CLARIFICATION`, store `command`, return `success=False` |
| Core/CME command | `:65-89` | `perform_action` on the CME workflow itself; mark `command_handled` for INTENT_DETECTION-stage hits and `abort` |
| Context-parent walk | `:99-107` | if no command found, retry prediction while walking `app_workflow.get_parent(...)` up the hierarchy (this is where `context_hierarchy_model.json` / `_<Context>.py` earn their keep) |
| Out-of-scope | `:109-124` | force `ErrorCorrection/you_misunderstood` |
| Param extraction | `:140-164` | `ParameterExtraction(...).extract()`; invalid → `PARAMETER EXTRACTION ERROR ...`, `success=False`, stage stays `PARAMETER_EXTRACTION` |
| Success | `:166-179` | `workflow.end_command_processing()` (resets stage to INTENT_DETECTION, clears `command`/`stored_parameters` — `workflow.py:291-303`), return artifacts `{command, command_name, cmd_parameters}` |

Intent-detection resolution order (`intent_detection.py:99-134`):
exact first-token match against `command_name_dict` → Levenshtein fuzzy
(`find_best_matches`, threshold 0.3) → embedding `cache_match` at 0.85 cosine →
transformer `command_router.predict`; multiple predictions → ambiguity error + suggested
commands stored. `abort` and `you_misunderstood` are matched ONLY by exact/fuzzy plain
utterance, never by the classifier (`intent_detection.py:79-97`). Model/threshold detail
belongs to fastworkflow-nlu-pipeline-reference.

The NLU state machine value lives in `cme_workflow.context["NLU_Pipeline_Stage"]`
(`NLUPipelineStage`, `fastworkflow/__init__.py:13-18`: INTENT_DETECTION,
INTENT_AMBIGUITY_CLARIFICATION, INTENT_MISUNDERSTANDING_CLARIFICATION, PARAMETER_EXTRACTION).

## 4. Agent path (Topology B)

`_process_agent_message` (`workflow_execution_context.py:804-815`):
1. `_ensure_agent_initialized` (`:675-677`) builds the `fastWorkflowReAct` tool agent
   (`workflow_agent.py:387-471`, `max_iters=25`, tools: `what_can_i_do`,
   `execute_workflow_query`, `ask_user`; `intent_misunderstood` and `whats_done_and_what_remains`
   were removed later, and `search_memory` is disabled since 2026-10-09). (It also built an
   intent-clarification agent until that was removed on 2026-10-05.)
2. `_run_agent` (`:709-734`): clears the action log, refines the query with the last 5
   conversation-history entries (`_refine_user_query`, `:981-991`), prepends an
   LLM-generated todo list (`build_query_with_next_steps`, `workflow_agent.py:474-534`),
   then calls the agent with up to 2 retries on `AdapterParseError`
   (`_call_agent_with_retry`, `:696-707`).
3. Tool calls land in `_execute_workflow_query` (`workflow_agent.py:122-259`), which calls
   the SAME `CommandExecutor.invoke_command` as the deterministic path, captures failures
   as `CommandOutput(success=False)` without masking the exception (`:152-181`), and after
   each call inspects `NLU_Pipeline_Stage`: AMBIGUITY/MISUNDERSTANDING auto-`abort` and
   return the NLU reply unchanged for the agent to choose from, and a PARAMETER_EXTRACTION
   error state auto-`abort`s and replans (`:237-252`).
4. If the result has `suspended=True` → set `_awaiting_user`, remember the message and
   clarification, `_note_agent_suspension` (appends the role-inverted ask_user entry,
   `:214-231`), return `_awaiting_user_output` (a CommandOutput whose response is the
   clarification, with `artifacts["awaiting_user"]=True`, `:741-751`).
5. Else `_reset_agent_suspension` + `_finalize_agent_output` (`:753-802`): final answer
   text + conversation summary (LLM) + **Topic-5 artifact merge** (`:786-791`).

## 5. Suspension and resume, exactly

Suspend (Topology B):
```
ask_user tool closure (workflow_agent.py:444-457)
  sets workflow_tool_agent.iteration_counter = -1      # :454 (dual-purpose flag)
  user_message_queue is None → raise AskUserSuspend    # :455-457
fastWorkflowReAct._run_loop catches it (react.py:250-262)
  self._suspended = {trajectory, idx, input_args, max_iters, clarification}
  return Prediction(suspended=True, clarification=..., exhausted=False)
WEC._process_agent_message sees suspended → _awaiting_user = True (wec:808-813)
```

Resume (next `process_turn` while `_awaiting_user`):
```
WEC._resume_agent_message (wec:817-838)
  _note_agent_resume folds elapsed time into suspended_ms   # :233-240
  workflow_tool_agent.iteration_counter = -1                # :823
  observation = _post_ask_user_response(question, answer)   # workflow_agent.py:321-351
    → _complete_ask_user_entry fills the pending A7 entry (answer + user think time)
    → builds a new todo list from trajectory + user response
  fastWorkflowReAct.resume(observation) (react.py:171-196)
    trajectory[f"observation_{idx}"] = observation; idx += 1; iteration_counter += 1
    _run_loop continues the SAME trajectory
```
Multiple suspensions per turn are fine: each re-suspension appends another ask_user entry
and the turn_key never changes (I6).

Topology A difference: `ask_user` sees a `user_message_queue`, so `_ask_user_tool`
(`workflow_agent.py:354-385`) enqueues the clarification to the output queue, appends the
unanswered ask_user entry itself (`:376`), and **blocks** on `user_queue.get()` (`:381`) —
the ChatWorker daemon thread (`chat_session.py:27-48`) is what makes indefinite blocking
tolerable. Known open bug fix-5fv: this path does not emit the trace-queue sentinel,
suspected CLI spinner hang (unreproduced — "needs runtime verification").

Cross-process resume: `serialize_state` (`wec:312-356`) exports
`{schema_version, channel_id, session_key, app/cme workflow ids, workflow_folderpath,
awaiting_user, suspended_user_message, pending_clarification_request,
react (export_suspended blob incl. iteration_counter — react.py:120-131), nlu_stage,
current_command_context_name, action_log, conversation_history_turns}` — round-tripped
through `json.dumps(default=str)` so it is JSON-portable. `apply_serialized_state`
(`wec:358-405`) restores everything except navigation depth (fix-cgs) and only warns on
schema mismatch (fix-4od).

## 6. FastAPI turns engine wrapping (v2.22.0)

`run_fastapi_mcp/turns.py`:
- `submit_turn` (`:321-360`): registry-owned single-flight per channel; idempotency key =
  sha256(channel_id + kind + normalized args) (`:84-96`) so a proxy retry REJOINS the same
  execution; bounded wait on `done_event` behind `asyncio.shield` (`:354-359`).
- `_run_turn` (`:269-318`) is "the only place that touches ctx for a turn": acquires
  `runtime.lock`, runs the blocking work_fn (`ctx.process_turn`/`process_action_turn`) in
  the executor, drains traces (destructively — Step-2 gap), persists conversation +
  suspended state BEFORE setting `exec_state=DONE` and firing `done_event`.
- `ExecState` (`:62-74`: QUEUED/RUNNING/DONE/LOST) is the *execution lifecycle*, orthogonal
  to `TurnStatus` (the turn outcome).
- Known inconsistency: `/invoke_agent_stream` bypasses all of this (guards with
  `runtime.lock.locked()`, `run_fastapi_mcp/__main__.py:963` — grep `locked()` to re-verify).

## 7. Identity and globals worth knowing before you refactor

- `workflow_id = mmh3.hash(workflow_id_str)` signed int (`__init__.py:265-266`);
  `Workflow.create` returns the existing live object for the same id (`workflow.py:101-104`).
- Live sessions are a process-global `weakref.WeakValueDictionary` (`workflow.py:41-45`);
  the live `Workflow` object IS the store entry; GC evicts abandoned sessions. The CLI
  intentionally lost cross-restart context resume when this replaced RocksDB (v2.21.4).
- `Workflow.create` inserts the project path into `sys.path` (`workflow.py:96-99`,
  "THIS IS IMPORTANT") and `close()` removes it — process-global mutation.
- `fastworkflow.init(env_vars)` must run before framework classes exist: it late-binds the
  `CommandContextModel`/`RoutingDefinition`/`RoutingRegistry`/`ModelPipelineRegistry`
  module globals (`__init__.py:174-209`).
- Cache invalidation (v2.22.1): `command_directory.json`/`routing_definition.json` carry a
  sha256 fingerprint over all `_commands/**/*.py` + both context-model JSONs;
  `RoutingRegistry` trusts persisted JSON only on fingerprint match
  (`command_directory.py:627` `compute_commands_source_fingerprint`; freshness check
  `command_routing.py:385-405`; stamped at `:201-204`).

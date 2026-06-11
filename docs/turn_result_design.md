# TurnResult, `CommandOutput` redesign, and durable turn-output persistence

Design note for fastWorkflow (framework-owned) plus its xray integration. This document
captures the **complete** investigation, the full design debate, and the **minutiae of every
decision** that produced the design below. Nothing is omitted; minor and "obvious" decisions
are recorded deliberately so the rationale survives.

Status: **design complete, not yet implemented.** No code has been changed. No bd epic filed
(by request).

---

## 0. Origin: the bug that started this

### 0.1 Symptom

`POST /invoke` (xray FastAPI) always returns `payload = None`, even for natural-language data
questions that clearly produced a table or chart.

### 0.2 The producer is correct

`NLQueryTool.process_query_results` (`shared/src/tools/nlquery_tool.py`) builds a `ResponseTuple`
with a real payload:

- `payload_hint` is computed as `"chart"` / `"table"` / `"text"`.
- `payload` is a CSV string: `df.write_csv(...)` for charts, or
  `df.head(MAXROWS_TABLEPAYLOAD).write_csv(...)` for tables.
- Returns `ResponseTuple(request_id, response=data_summary, payload_hint, payload, actions=None)`.

### 0.3 The mapping layer is correct

`shared/src/fastworkflows/access_review/_commands/_response_mapping.py` round-trips the tuple
through fastWorkflow's `CommandOutput`:

- `response_tuple_to_command_output()` packs `payload_hint`, `payload`, `actions`, `request_id`
  into `CommandOutput.command_responses[0].artifacts`.
- `command_output_to_response_tuples()` reads them back out, iterating
  `command_output.command_responses` and returning a `List[ResponseTuple]`.

The module docstring explicitly notes this is "intentional framework glue": the table/chart
payload travels through `artifacts` because `CommandResponse` only carries a text `response`.

### 0.4 The command handler is correct

`answer_data_question.ResponseGenerator.__call__` calls `process_sql_query`, gets the
`ResponseTuple`, and returns `response_tuple_to_command_output(workflow.id, response)`. The
payload is present in the `CommandOutput` it returns.

So every xray-owned component preserves the payload. The loss is inside the framework.

---

## 1. Root-cause investigation (agent mode drops artifacts)

The xray runner constructs each per-session `WorkflowExecutionContext` (WEC) with
`run_as_agent=True` (`shared/src/application/fastworkflow_runner.py`, `_create_runtime`). In agent
mode, `WorkflowExecutionContext.process_message` does **not** return the command's `CommandOutput`.
It routes:

```
process_message
  -> _should_run_agent_for_message == True (run_as_agent and not assistant '/'-command)
  -> _process_agent_message            (or _resume_agent_message when awaiting_user)
  -> _finalize_agent_output
```

Two places destroy the payload:

### 1.1 The ReAct tool boundary is text-only

The agent invokes commands as a DSPy ReAct tool, `execute_workflow_query`, implemented by
`_execute_workflow_query` in `fastworkflow/workflow_agent.py`. It calls
`CommandExecutor.invoke_command(...)`, which returns the full `CommandOutput` (artifacts intact),
but then extracts **only the text**:

```python
response_text = ""
if command_output.command_responses:
    response_parts = [cr.response for cr in command_output.command_responses if cr.response]
    response_text = "\n".join(response_parts) or "Command executed successfully but produced no output."
...
return response_text
```

The `artifacts` dict (`payload`, `payload_hint`, `request_id`, `actions`) is never read. Only the
`response` text enters the agent's reasoning. This is by design: DSPy ReAct tools return string
observations, so the agent reasons over text only.

### 1.2 The agent synthesizes a fresh `CommandOutput`

After the ReAct loop, `_finalize_agent_output` builds a **brand-new** `CommandOutput` from the
agent's `final_answer`, attaching only a `conversation_summary` artifact:

```python
command_response = fastworkflow.CommandResponse(response=result_text)
if self._action_log:
    conversation_summary, conversation_traces = self._extract_conversation_summary(...)
    command_response.artifacts["conversation_summary"] = conversation_summary
command_output = fastworkflow.CommandOutput(command_responses=[command_response])
```

So the `CommandOutput` that reaches the runner's `command_output_to_response_tuples` has artifacts
`{conversation_summary, maybe awaiting_user}` — **no `payload` key**. Therefore:

- `artifacts.get("payload")` -> `None`
- `payload_hint` falls back to its default `"text"`
- `request_id` falls back to `-1`

### 1.3 Confirming signature

In agent mode the FastAPI response shows **not just** `payload=None` but also `payload_hint="text"`
and `request_id=-1`, regardless of the query, and `response` is the LLM-rephrased `final_answer`
rather than the exact `data_summary`. By contrast, the deterministic `/`-prefixed assistant-mode
path (`_process_message`) returns the command's **actual** `CommandOutput` verbatim, so payload
*would* survive there. That divergence is the tell: the agent path is the culprit, not the mapping
module.

### 1.4 Why this is architectural, not a mapping bug

`_response_mapping.py` assumes the `CommandOutput` round-trips intact. That holds for direct command
execution but is violated by ReAct agent mode, where the framework deliberately collapses tool
outputs to text (the agent reasons over text observations) and emits its own final answer.
Artifacts are framework-internal and are not part of the agent's tool -> observation -> final-answer
contract.

---

## 10. Runner and xray integration

### 10.1 Runner: write review on the completion branch

`shared/src/application/fastworkflow_runner.py` `_persist_pending_after_turn` currently does:

```python
if runtime.ctx.awaiting_user or _output_is_awaiting_user(command_output):
    self._store.save(key, runtime.ctx.serialize_state(channel_id=key))   # pending (light partial)
else:
    self._store.clear(key)                                               # completed
```

**Change:** the `else` (completed, not-awaiting) branch must ALSO write the completed `TurnResult` to
the `TurnReviewStore` (after offloading payloads). This runs **inside the per-session lock**
(`run_turn` holds `runtime.lock`), so per-session review writes are serialized and safe. It must NOT
be wired only into the suspend path. The pending-store `save` continues to carry the light partial
across suspends.

### 10.2 `process_message` return type

`run_turn` will receive a `TurnResult` from `ctx.process_message` (section 5.6) rather than a
`CommandOutput`. The runner maps it to the xray API (`List[ResponseTuple]`).

### 10.3 xray `command_output_to_response_tuples` change

Today it iterates `command_output.command_responses`. After the redesign it consumes a `TurnResult`:

- The **headline** `ResponseTuple` is built from `TurnResult.answer` (the narrative; `payload` may be
  `None`).
- Then one `ResponseTuple` per **payload-bearing** `CommandOutput` in `TurnResult.command_outputs`
  (the gallery), fetching payloads lazily from the `PayloadStore` by handle.
- **The UI decides which payload is the headline, if any** (user decision); the framework/mapping does
  not designate a primary payload. The gallery is the raw payload-bearing outputs in turn order.

This realizes "single answer + gallery of data outputs," which the existing list-returning shape of
`command_output_to_response_tuples` already accommodates.

---

## 11. Mechanical changes (no design debate, recorded for completeness)

- **`CommandOutput.to_mcp_result()`** currently iterates `command_responses` to build MCP content;
  update for the single `command_response` collapse.
- **`CommandExecutor`** accessors `command_output.command_responses[0]` (e.g. `command_executor.py`
  lines 56-57, the `artifacts["command_name"]` / `artifacts["cmd_parameters"]` reads) update to
  `command_output.command_response`.
- **`workflow_execution_context.py`** construction sites (lines 419, 522, 558) and
  **`workflow_agent.py:276`** switch from `command_responses=[r]` to `command_response=r`.
- **All command authors / examples** that return `CommandOutput(command_responses=[r])` migrate to
  `command_response=r` (mechanical; section 4.1 shows all are single-element).
- **`SCHEMA_VERSION`** bump in `session_state_store.py` for the new serialized shape.
- **Old-schema migration is out of scope** (user decision): handled by a separate migration tool if
  deemed necessary. In-flight suspended sessions serialized under the old shape are not auto-migrated.

---

## 12. Decision log (quick reference)

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | Patch the framework (not arc side-channel / agent bypass) | User owns fastWorkflow; most correct |
| 2 | Accumulate ALL tool-call outputs per turn | Full fidelity; last-wins/attributed rejected |
| 3 | Unit of accumulation is `CommandOutput`, not `CommandResponse` | Preserves provenance; avoids poisoning predicates |
| 4 | Collapse `CommandOutput.command_responses` -> single `command_response` | Provably-unused list; removes competing multiplicity axis |
| 5 | Recursion seam is `CommandOutput.nested_turn`, modeled now | Recursion belongs on execution node; nested agents imminent |
| 6 | `TurnResult.answer` is a `CommandResponse` | Answer is a value, not an execution; str too lossy |
| 7 | Turn metadata on `TurnResult`, not on `answer` | Answer spans contexts; entry context is a turn property |
| 8 | `process_message` returns `TurnResult` (breaking) | Explicit > hidden mutable state; avoids stale-read footgun |
| 9 | Streaming stays text-only via existing `command_trace_queue` | No live payloads needed; no durable mid-turn writes |
| 10 | Durability = option D (light record + payload offload), once per logical turn | Preserves `action_log`-level latency; meets review |
| 11 | ~~Separate `TurnReviewStore` (enumerable) from pending `SessionStateStore`~~ **Superseded by Amendment A1**: pending `SessionStateStore` + unified `ConversationTurnStore` (absorbs `ConversationStore`) | Different lifecycle/access pattern |
| 12 | Enumeration only on `TurnReviewStore` | Pending store holds one in-flight turn; never lists |
| 13 | ~~`PayloadStore` content-addressed, disk/redis mirror~~ **Superseded by Amendment A8**: turn-scoped keys; content hash survives only as the leaf segment (within-turn idempotency) | Idempotent, concurrency-safe; matches existing split |
| 14 | Key = channel + sortable ISO-8601 GMT ts + uuid; ordinal by sorting | Backends don't preserve insertion order |
| 15 | Summary is value metadata, not key | Unsafe/long/collision-prone; couples to LLM call |
| 16 | Offload payloads at every persistence boundary (suspend + completion) | Keeps suspend blob light; resume needs no payload fetch |
| 17 | Review write on the completion branch, inside per-session lock | Serialized + safe; not the suspend path |
| 18 | Retire `action_log`; derive summary text view from `command_outputs` | Zero regression; removes divergence |
| 19 | Retention/GC out of scope; infra co-GC record+payload under one lifecycle key — **rewritten by Amendment A8**: the lifecycle key is literally the conversation prefix | Plus reader tolerates missing payload |
| 20 | Nested turns embedded-only; top-level keys for user-message turns | Serializer recurses; nested addressed by path |
| 21 | UI decides headline payload | Framework surfaces raw gallery in turn order |
| 22 | Old-schema migration out of scope (separate tool) | Bump `SCHEMA_VERSION` |

---

## 13. Open questions

None remaining. The capture side (types, single-producer/three-sinks, `action_log` retirement) and
the review side (key schema, two stores, payload offload, persistence boundaries, recursion,
GC contract) are all resolved per the decision log above. Remaining work is implementation, tracked
separately from this document.

### 1.5 Fix options considered for the symptom (before going deeper)

Three options were enumerated for getting payload back to the API:

1. **Side-channel through the `arc`.** `process_query_results` already holds the
   `AccessReviewContext` and `request_id`; stash the last `(payload_hint, payload, actions)` on the
   arc, read it in the runner after `run_turn`. Least invasive, framework-agnostic.
2. **Bypass the agent for pure data questions.** Route data-query turns through the deterministic
   `_process_message` path so the real `CommandOutput` survives.
3. **Patch the framework** to carry artifacts through the agent path. Most correct, but the user
   owns fastWorkflow, so this is viable — and it raised the central question below.

The user owns fastWorkflow and chose to patch the framework. That immediately surfaced the design
question that drives the rest of this document.

---

## 2. The central question: multiple tool calls per turn

> "Since an agent is involved, the output could include command artifacts from multiple tool calls.
> Is fastWorkflow designed to handle this scenario?"

### 2.1 Where multiplicity is *supported* by the data model

- `CommandOutput.command_responses` is a **list** of `CommandResponse`.
- Each `CommandResponse` carries its own `artifacts: dict`, plus `next_actions: list[Action]` and
  `recommendations: list[Recommendation]`.
- xray's `command_output_to_response_tuples` already **iterates** that list and returns a
  `List[ResponseTuple]`, and `FastWorkflowRunner.invoke` returns `List[ResponseTuple]`.

So the plumbing from `CommandOutput` outward is already built to surface N results with N payloads.

### 2.2 Where multiplicity is *dropped* by the agent

1. The tool boundary returns `str` (section 1.1).
2. The only per-turn accumulator, `action_log`, stores text-only records
   `{command, command_name, parameters, response}` — no artifacts.
3. The turn collapses to a single synthesized `CommandResponse` (section 1.2).

### 2.3 Conclusion

- **Transport/model layer:** yes, structurally supports multiple results, each with artifacts.
- **Agent layer:** no, deliberately not. ReAct's contract is *N text observations -> 1 text final
  answer*. Artifacts never enter the agent's reasoning or output.

Therefore a patch is not a one-liner: the semantics of "which artifacts, from which of several tool
calls" must be defined first, because the synthesized `final_answer` is a single narrative while
payloads come from individual, possibly multiple, tool calls.

### 2.4 Semantic options enumerated

- **Last-payload-wins:** track the most recent artifact-bearing output; wrong for multi-part answers.
- **Accumulate all:** collect every tool call's result for the turn and surface them.
- **Attributed:** surface only payloads the `final_answer` actually used; requires the agent to emit
  references/citations (a signature/prompt change).

The user selected **accumulate all** for full fidelity.

---

## 3. Unit of accumulation: `CommandOutput`, not `CommandResponse`

An initial proposal was to emit each tool call's artifacts as **additional `CommandResponse`
entries** in the final `CommandOutput`. The user rejected this, and was correct. Critique:

1. **It corrupts derived predicates.** `CommandOutput` has `success = all(r.success ...)`,
   `command_aborted = any(r.artifacts.get("command_name") == "abort" ...)`,
   `command_handled = any(r.artifacts.get("command_handled") ...)`, and
   `not_what_i_meant = any(r.artifacts.get("command_name") == "misunderstood_intent" ...)`.
   Injecting intermediate tool outputs into `command_responses` poisons all of these aggregates.
2. **`CommandResponse` is the wrong unit.** A tool call returns a whole `CommandOutput`:
   `command_name`, `command_parameters`, `workflow_name`, `context`, `success`, **and**
   `command_responses[*].artifacts`. Flattening to a bare `CommandResponse` discards provenance.
3. **Execution unit == storage unit.** Each `execute_workflow_query` -> `invoke_command` produces
   exactly one `CommandOutput`. A `list[CommandOutput]` per `process_message` is a faithful 1:1
   transcript of the turn.

**Decision:** the turn stores a `list[CommandOutput]`. The synthesized final answer stays a separate,
pristine object; per-tool-call outputs live in their own container.

### 3.1 Single clean insertion point

Every command execution — including nested intent-clarification commands and the
`PARAMETER_EXTRACTION` abort — funnels through one call in `_execute_workflow_query`:

```python
command_output = CommandExecutor.invoke_command(chat_session_obj, command)
```

Appending `command_output` to a WEC-held turn list right here captures everything with no
path-by-path plumbing. The agent helper tools `what_can_i_do`, `intent_misunderstood`, and
`ask_user` return strings and never reach `invoke_command`, so they are naturally excluded.

---

## 4. Empirical findings that grounded the type redesign

### 4.1 `list[CommandResponse]` is never used with more than one element

A search of the entire installed framework **and** its bundled examples
(`fastworkflow/examples/retail_workflow/...`) found:

- Every `CommandOutput(command_responses=[...])` construction uses a **single-element** list.
- **Zero** `command_responses.append(...)` or `.extend(...)` sites.

Construction sites observed: all retail example commands, `workflow_execution_context.py` (lines
419, 522, 558 — the cancelled/awaiting/finalize paths), `workflow_agent.py:276` (ask_user tool),
`command_executor.py:34` (empty-command guard). All single-element.

Conclusion: the `list[CommandResponse]` is **speculative generality**, never exercised. Collapsing
it to a single `CommandResponse` is lossless in practice.

### 4.2 The command `ResponseGenerator` returns the `CommandOutput`

`CommandExecutor.perform_action` calls `response_generation_object(workflow, ...)` and type-checks
that the result is a `CommandOutput` (raises `TypeError` otherwise). `invoke_command` then sets
`workflow_name`, `context`, `command_name`, `command_parameters` on it. So the only multiplicity that
exists is author-controlled and unused.

### 4.3 There is no nested-workflow execution loop today

There is no separate "run a child workflow as its own turn" mechanism. Nesting today is **context
switching within one WEC** (the `active_workflow` ContextVar push/pop and
`current_command_context`), and each step still yields one flat `CommandOutput` tagged with
`workflow_name` / `context`. `command_executor.py` references `MAX_DELEGATION_DEPTH` and "delegation
hops," but no recursion into a sub-turn is produced at runtime. So today the turn is a flat list;
recursion is a future concern (see section 5.3 and the user's "nested agents almost right away").

---

## 5. The type algebra (final)

Three levels of granularity were identified: a command's result *value*, a command *execution*, and
a *turn*. The redesign assigns one type to each.

### 5.1 `CommandResponse` — the result value (unchanged)

```python
class CommandResponse(BaseModel):
    response: str
    success: bool = True
    artifacts: dict[str, Any] = {}
    next_actions: list[Action] = []
    recommendations: list[Recommendation] = []
```

A pure value: human-facing text + structured artifacts + follow-ups. **No execution provenance.**

### 5.2 `CommandOutput` — one command execution (collapsed + recursion seam)

```python
class CommandOutput(BaseModel):
    command_response: CommandResponse          # was: command_responses: list[CommandResponse]
    command_name: str = ""
    command_parameters: str = ""
    workflow_name: str = ""
    context: str = ""
    nested_turn: Optional["TurnResult"] = None # set only when this command ran a sub-agent/workflow

    @property
    def success(self) -> bool:
        return self.command_response.success
    @property
    def command_aborted(self) -> bool:
        return self.command_response.artifacts.get("command_name") == "abort"
    @property
    def command_handled(self) -> bool:
        return self.command_response.artifacts.get("command_handled", False) is True
    @property
    def not_what_i_meant(self) -> bool:
        return self.command_response.artifacts.get("command_name") == "misunderstood_intent"
```

**Decisions captured:**

- Collapse `command_responses: list` -> `command_response: CommandResponse` (section 4.1 proves it
  is lossless). This removes the *competing multiplicity axis*: multiplicity now lives in exactly one
  place — `TurnResult.command_outputs` ("the commands this turn ran") — instead of being split between
  "multiple responses per command" and "multiple commands per turn."
- The predicates simplify from `all(...)`/`any(...)` folds to single reads.
- `nested_turn` is the **recursion seam** (section 5.3).

### 5.3 Recursion belongs on the execution node, not the value

The question "could a `CommandResponse` be a `TurnResult`?" was rejected: it conflates a *leaf value*
with an *aggregate*, and would make the simplest type the recursive one. When a command delegates to a
sub-agent over a child workflow, the thing that recurses is the **execution**. So the seam is
`CommandOutput.nested_turn: Optional[TurnResult]`.

- The common (flat) case stays flat; a tree appears only where real nesting happens.
- The user foresees nested agents "almost right away," so `nested_turn` is modeled **now** (not just
  reserved), and the serializer is designed to recurse from the start (section 8).
- Today's context-switch nesting is still captured by `workflow_name`/`context` on flat outputs;
  `nested_turn` is for true sub-agent turns.

### 5.4 `TurnResult` — one logical turn (new)

```python
class TurnResult(BaseModel):
    user_message: str
    entry_workflow_name: str = ""
    entry_context: str = ""
    answer: CommandResponse              # synthesized agent answer (NOT a command execution)
    command_outputs: list[CommandOutput] = []
```

**Decisions captured:**

- **`answer` is a `CommandResponse`, not a `CommandOutput`.** The agent's answer is *not* a command
  execution: it has no `command_name`/`command_parameters`, and predicates like `command_aborted`
  are meaningless for a synthesized narrative. Typing it as `CommandOutput` would force fabricated
  empty provenance and carry never-true predicates — that was the "smell." A `CommandResponse` is the
  right shape: text + artifacts + next_actions + recommendations, without provenance.
- **`answer` is not plain `str`.** Plain text is too lossy: the answer legitimately wants artifacts
  (a headline payload reference), `next_actions`, and `recommendations`.

  > **[Corrected by Amendment A9]** The `next_actions`/`recommendations` part of this
  > motivation was circular — nothing populates them on the agent answer, and they stay empty
  > by decision. The answer is typed `CommandResponse` for its text+artifacts shape; buttons
  > belong on gallery entries where provenance is clear.
- **Turn metadata (`user_message`, `entry_workflow_name`, `entry_context`) lives on `TurnResult`,
  not on the answer.** The answer can span multiple workflows/contexts (the agent called commands
  across contexts), so "the workflow of the answer" is ill-defined; "the workflow the turn entered
  in" is well-defined and is a turn property. Per-command workflow/context already live on each
  `CommandOutput`.

### 5.5 Uniformity across the three resolution paths

`process_message` resolves a turn in three ways; all must populate `command_outputs` identically so
consumers never branch on mode:

- **Agent** (`_finalize_agent_output`): `answer` = synthesized `final_answer` wrapped as
  `CommandResponse` (+ `conversation_summary` artifact when available); `command_outputs` = the
  accumulated executions.
- **Deterministic `/`-command** (`_process_message`): `command_outputs == [the one CommandOutput]`
  and `answer == command_outputs[-1].command_response`. No mode-branching for consumers.
- **Suspend/resume** (`_resume_agent_message`): accumulation spans the **logical** turn (section 7).

### 5.6 `process_message` return contract

**Decision:** break the `process_message` signature to return a `TurnResult` (or
`(answer, command_outputs)`), rather than exposing the accumulated list as hidden mutable WEC state.

Rationale: a side-channel property (`ctx.last_turn_command_outputs`) creates temporal coupling — the
value is valid only until the next turn clears it, and a caller who forgets to read-before-next-message
gets silently stale data. The user owns the framework and accepted the breaking change, so the
explicit typed return is preferred and removes the footgun.

---

## 6. Streaming investigation (does durability need to be mid-turn?)

The user streams agent activity in the UX (CLI / Chat UI) so the user sees progress. Question: does
that require **durable** mid-turn writes?

### 6.1 Streaming uses a separate in-memory channel

Streaming rides the `command_trace_queue` carrying `CommandTraceEvent`s — **not** persistence. The
producer emits two events per command execution at the same `invoke_command` choke point:
`AGENT_TO_WORKFLOW` (before) and `WORKFLOW_TO_AGENT` (after), in `_process_message`,
`_process_action`, and `_execute_workflow_query`.

Consumers drain it live:

- **CLI** `fastworkflow/run/__main__.py` (`command_trace_queue.get_nowait()` / `get(timeout=0.1)`).
- **Bundled FastAPI MCP server** `fastworkflow/run_fastapi_mcp/utils.py`: runs `process_message` in
  an executor and polls the queue every 50 ms, firing `on_trace` callbacks while the turn runs, then
  calls `persist_pending_after_turn` **once after** the turn. Streaming and persistence are already
  cleanly separated.

### 6.2 The trace event is text-only

```python
@dataclass
class CommandTraceEvent:
    direction: CommandTraceEventDirection
    raw_command: str | None
    command_name: str | None
    parameters: dict | str | None
    response_text: str | None
    success: bool | None
    timestamp_ms: int
```

No artifacts/payloads. So today the UI can stream "ran answer_data_question ✓" but cannot stream the
table/chart into a gallery mid-turn. Streaming live payloads would require **enriching the trace
carrier** with the `CommandOutput`/payload-handle — still an in-memory queue change, zero durable
hot-path I/O.

### 6.3 Decisions

- **Mid-turn durable writes (option C below) are NOT required for streaming.** Streaming is an
  ephemeral in-memory queue; durable `command_outputs` is for post-hoc review and is written once per
  (logical) turn.
- **Live payloads in the gallery are NOT needed** (user decision): text-trace streaming is enough.
  Therefore `CommandTraceEvent` is **left unchanged** and the producer does not need a fourth sink.

### 6.4 Gap noted (not in scope to fix here)

xray's own `shared/src/application/main.py` wires **none** of the streaming machinery — no queues, no
SSE — it is pure request/response (only a lifespan `yield`). The runner never calls
`set_transport_queues`. So the Chat UI streaming runs through the bundled fastWorkflow MCP server or a
separate frontend, **not** xray's `/invoke`. If the Chat UI is meant to stream through xray, that
wiring does not exist in this repo yet. Recorded as a known gap.

### 6.5 Architecture: one producer, three sinks

The same `CommandOutput` from the `invoke_command` choke point fans out to three sinks differing only
in lifetime/transport:

1. **Live `command_trace_queue`** -> streaming UX (text-only today; unchanged per 6.3).
2. **In-memory `command_outputs` list** -> turn accumulation (subsumes `action_log`, section 9).
3. **Durable store, once per logical turn** -> review (section 7, option D).

---

## 7. Durability for review

### 7.1 Current persistence mechanism (baseline)

`fastworkflow/session_state_store.py` defines `SessionStateStore` (ABC) with
`load`/`save`/`clear`/`exists`, keyed by `channel_id`, with two backends:

- `DiskSessionStateStore` — one JSON file per channel (`{safe_id}_pending.json`), `json.dump(..., default=str)`.
- `RedisSessionStateStore` — one JSON string per channel under `fw:session:pending:{channel_id}`.

Factory `get_session_state_store()` selects via `SESSION_STATE_STORE=disk|redis`.

In the xray runner, durable I/O fires **only at suspend/eviction**:

```python
if runtime.ctx.awaiting_user or _output_is_awaiting_user(command_output):
    self._store.save(key, runtime.ctx.serialize_state(channel_id=key))
else:
    self._store.clear(key)
```

On a normal completed turn it `clear`s. So today there is **no per-turn or per-command durable
write** on the happy path.

> **[Correction — review finding R37, see Amendments A1]** This baseline holds only for the
> xray runner. The bundled FastAPI server *does* durably persist a per-turn record on the happy
> path (`save_conversation_incremental` → Rdict `ConversationStore`) after every turn.

### 7.2 The reframing

"Can durable `command_outputs` match `action_log`'s latency?" decomposes into two independent
questions:

1. **In-memory turn accumulation** (subsumes `action_log`): `command_outputs` is also just an
   in-memory list with O(1) append. **Zero regression.** Unconditionally safe.
2. **Durability for review** (new requirement): `action_log` never persisted on completed turns, so
   there is no baseline to regress against. Cost depends entirely on when/where/what we persist.

### 7.3 Options enumerated with latency profiles

- **A — In-memory + whole-blob at suspend only (status quo extended).** Hot-path cost identical to
  `action_log` (zero on completed turns), but completed turns are cleared, so it does **not** satisfy
  "available for review." Rejected as the primary mechanism.
- **B — Persist `TurnResult` once per turn (whole-blob).** One write at turn end (~ms disk / sub-ms
  Redis), trivial behind a multi-second LLM turn. Meets review. Risk: large values if payloads are
  inlined.
- **C — Incremental per-command append (append-only log).** N writes per turn; the only option that
  reintroduces hot-path I/O comparable to CLI's `action.jsonl` mirror (esp. disk `fsync`). Needed
  only for streaming/mid-turn durability — and section 6 shows streaming does **not** need it.
- **D — Split storage: light metadata fast-path + payload offload (CHOSEN).** Keep in-memory
  `command_outputs`; persist a **light** `TurnResult` (records + artifact keys + payload **handles**)
  into the store, and offload large payloads (CSV) to a separate content-addressed payload store,
  fetched lazily on review. Keeps the suspend blob and the per-turn record `action_log`-sized,
  preserving latency; "single answer + gallery" lists cheaply and hydrates payloads on demand.

### 7.4 Decisions

- **Persist timing:** once per **logical** turn (B-style timing), payloads offloaded (D). No
  per-command durable writes.
- **Payloads are offloaded** to a `PayloadStore`; the `TurnResult` carries handles. This also keeps
  the **suspend** blob light (section 7.6).

### 7.5 Two stores, two animals: introduce `TurnReviewStore`

The existing `SessionStateStore` is the **pending/suspend** store: one transient blob per channel,
**cleared on completion**, single-key `load`/`save`/`clear`/`exists`. The **review** store is
**write-once, keep-many, enumerable**. Different access patterns and lifecycles.

**Decision:** do **not** overload `SessionStateStore`. Introduce a separate **`TurnReviewStore`**,
mirroring the disk/redis split, so the pending path stays single-key get/set/clear and review gets
prefix-scan semantics. **Enumeration is added only to `TurnReviewStore`**, never to the pending store
(the pending store only ever holds the single in-flight suspended turn, so it never needs to list).

### 7.6 Persistence boundaries (suspend AND completion)

Retiring `action_log` (section 9) means `serialize_state` must carry the **partial** `TurnResult`
(outputs produced before an `ask_user`) so a suspended turn can resume. To keep the suspend blob
light, payloads are offloaded to the `PayloadStore` **at the suspend boundary too**, not only at
completion. Rule: **offload payloads at every persistence boundary (suspend and completion)**, with
the logical-turn accumulation spanning them.

Cheap consequences (confirmed desirable):

- **Resume needs no payload fetch.** The agent reasons only over text, so on rehydrate we restore the
  light records (text + handles) and continue; payloads stay in the `PayloadStore` and are only
  referenced by the final `TurnResult`. No read-back on the hot path.
- **`TurnReviewStore` write fires once, at logical-turn completion** (the `else`/not-awaiting branch).
  Suspends write only the light partial to the **pending** store. A turn that suspended three times
  still produces exactly **one** review record.
- A suspended turn's "answer" is the clarification `CommandResponse` (`awaiting_user` artifact) and is
  **not** written to review — correct by construction.

### 7.7 Turn key schema

**Decision (corrected from the user's first proposal):**

- Key = `channel_id` + **lexicographically-sortable** GMT timestamp (ISO-8601 UTC, fixed-width,
  zero-padded, ms/µs precision, e.g. `2026-06-10T17:04:03.123456Z`) + a short disambiguator
  (`uuid4` suffix, or µs precision) to avoid collisions.
- **Ordinal is derived by sorting keys by the timestamp prefix**, **not** by store iteration order.
  Rationale: neither durable backend preserves insertion order — `os.listdir`/`os.scandir` directory
  order is arbitrary, and Redis `SCAN`/`KEYS` is explicitly unordered. The user's original
  "infer ordinal from position in the store, assuming Python dict ordering" only holds for an
  in-memory dict and was rejected for that reason.
- **The conversation summary is NOT part of the key.** It is an LLM output (long, may contain
  newlines/slashes/unicode, >255-byte filename risk, collision-prone) and is produced by
  `_extract_conversation_summary` (a `ChainOfThought` call) **only** for agent turns with a non-empty
  `action_log`. Deterministic `/`-command turns have no summary. Keying on it would (a) be unsafe as a
  filesystem/Redis key, (b) couple persistence to an LLM call, and (c) leave deterministic turns
  without a key. The summary is stored as **searchable metadata inside the value** when present.
- **Namespacing by `channel_id`** prevents turns from different sessions interleaving in one keyspace.

### 7.8 Retention / GC — out of scope (with a required contract)

Retention/GC is **out of scope** and handled at the infrastructure/deployment level (user decision).
But because the light `TurnResult` references payloads by handle in a **separate** payload store,
independent GC schedules would produce dangling handles (record -> deleted payload) or orphans
(payload kept, record gone).

**Documented contract:** infrastructure MUST GC the review record and its referenced payloads **under
one lifecycle key** (co-GC). Additionally, the review reader SHOULD tolerate a missing payload
gracefully (render "payload expired") as defense in depth.

### 7.9 Nested-turn addressing in review — embedded-only

Nested turns are **embedded** in the parent `CommandOutput.nested_turn` (the serializer recurses,
section 8). **Top-level `TurnReviewStore` keys are for user-message turns only**; nested turns are
addressed by **path within the parent record**, not as first-class top-level keys. So the
timestamp+uuid key scheme applies only to top-level turns.

---

## 8. Store interfaces and serializer

### 8.1 `TurnReviewStore` (new)

```python
class TurnReviewStore(ABC):
    @abstractmethod
    def put(self, channel_id: str, turn_key: str, turn: dict[str, Any]) -> None:
        """Write-once persist of a completed (light) TurnResult."""
    @abstractmethod
    def get(self, channel_id: str, turn_key: str) -> Optional[dict[str, Any]]: ...
    @abstractmethod
    def list(self, channel_id: str) -> list[str]:
        """Return this channel's turn keys, sorted by the ISO-8601 timestamp prefix
        (NOT by store iteration order). Ordinal == index in this sorted list."""
```

- `DiskTurnReviewStore` — directory per `channel_id`, one JSON file per `turn_key`; `list` sorts
  filenames by the timestamp prefix.
- `RedisTurnReviewStore` — keys `fw:turn:{channel_id}:{turn_key}`; `list` does a `SCAN MATCH` on the
  channel prefix then sorts by the timestamp prefix.
- Factory mirrors `get_session_state_store()` (`TURN_REVIEW_STORE=disk|redis`, defaulting consistent
  with the pending store).

### 8.2 `PayloadStore` (new, content-addressed, disk/redis mirror)

```python
class PayloadStore(ABC):
    @abstractmethod
    def put(self, data: bytes | str) -> str:
        """Store a payload, return an opaque handle (content-addressed hash recommended)."""
    @abstractmethod
    def get(self, handle: str) -> Optional[bytes]:
        """Fetch a payload by handle, or None if GC'd (reader tolerates None, section 7.8)."""
```

- `DiskPayloadStore` — content-addressed blob directory (single-node).
- `RedisPayloadStore` — `fw:payload:{handle}` (multi-pod). TTL is an infra concern (section 7.8).
- Content addressing makes writes idempotent and concurrency-safe (same bytes -> same handle).
- **User decision:** payload store mirrors the existing disk/redis split for now. No external object
  store (e.g. S3) targeted yet; revisit if the xray deployment provides one.

### 8.3 Recursive serialization (designed for `nested_turn` from the start)

The (light) `TurnResult` serializer MUST recurse: serializing a `TurnResult` serializes its
`command_outputs`, and each `CommandOutput` that has a `nested_turn` serializes that nested
`TurnResult` the same way. Payload offload applies at **every** level: any `CommandResponse.artifacts`
payload (top-level or nested) is replaced by a `PayloadStore` handle during serialization, and the
text/records are kept inline. Deserialization restores the tree; payloads are fetched lazily by handle
only when a reviewer opens that node.

---

## 9. Retiring `action_log`

### 9.1 How `action_log` is used today

- **Lifecycle:** `clear_action_log()` at the start of each agent turn (`_run_agent`); appended in
  `_execute_workflow_query` (`workflow_agent.py`) and `_post_ask_user_response`.
- **Sole data consumer:** `_finalize_agent_output` passes it to `_extract_conversation_summary` (an
  LLM `ChainOfThought`) when non-empty. That consumer needs only the **text projection**
  `{command, command_name, parameters, response}` — derivable from `command_outputs` by dropping
  artifacts/payloads.
- **Plumbing:** serialized in `serialize_state` (`"action_log"`), restored in `apply_serialized_state`.
- **No external consumer.** xray never reads `action_log` or `action.jsonl` (verified by repo grep:
  the only `_store.*` and `serialize_state` references are the runner's pending-store calls).

  > **[Correction — review finding R37, see Amendments A1]** The "sole consumer" framing is
  > incomplete. The full chain is: `action_log` → `(summary, traces)` →
  > `conversation_history` (dspy.History) → (a) the agent's cross-turn memory
  > (`_refine_user_query`, session restore), (b) durable `ConversationStore` persistence in the
  > bundled server, and (c) the `/post_feedback` target. Retiring `action_log` touches all three.
- **`action.jsonl` mirroring is CLI-only** (`ChatSession` sets `mirror_action_log_to_file=True`;
  the WEC/FastAPI path leaves it `False`), so in serving there is **zero file I/O** for `action_log`.
- **Perf profile:** in-memory `list[dict]`, O(1) append, read once per turn, serialized only inside
  the suspend blob (fires only on `ask_user`). Effectively free.

### 9.2 Decision

**Retire `action_log`** unconditionally; make in-memory `command_outputs` the single source of truth
and **derive the LLM-summary text view** from it. Zero hot-path regression (both are in-memory O(1)
append), and it removes a divergence risk (two parallel turn logs). The dependency from "review
durability" does not gate this: the in-memory subsumption is independent of how/where review persists.

### 9.3 Consequence for `serialize_state`

`serialize_state`/`apply_serialized_state` replace the `"action_log"` field with the **light partial
`TurnResult`** (records + payload handles), per section 7.6. The CLI `action.jsonl` mirror, if kept
for debugging, derives from `command_outputs`; otherwise it is dropped (CLI-only, no external
contract).

---

## Amendments from the critical review

Resolutions adopted after the systematic review of `docs/turn_result_design_review.md`
(beads epic `fix-vof`). Each amendment supersedes the referenced sections/decisions above.

### A1 — Unified `ConversationTurnStore` (resolves R37; supersedes decision 11; corrects 7.1/7.2 and 9.1) — 2026-06-10

The review discovered that the framework already persists per-turn records durably: every
resolution path appends `{conversation summary, conversation_traces, feedback}` to
`conversation_history` (dspy.History), and the bundled FastAPI server persists it after every
turn via `save_conversation_incremental` into the Rdict-backed `ConversationStore`
(conversation ids, LLM topics/summaries, `/list_conversations`, `/activate_conversation`,
`/post_feedback`, admin dump, session-restore of agent memory).

**Decision: full absorption.** `ConversationStore` is eliminated as a subsystem and
`TurnReviewStore` is not built under that name. A single unified store —
**`ConversationTurnStore`** — owns two record types:

- **Conversation metadata records:** key `fw:conv:{channel_id}:{conv_id}`; value
  `{topic, summary, status (active|closed), created_at, updated_at, schema_version, metadata}`.
- **Turn records:** key `fw:turn:{channel_id}:{conv_id}:{sortable-ts}-{uuid}`; value = the
  light `TurnResult`.

Consequences:

1. `save_conversation_incremental` is deleted; the once-per-logical-turn record write is the
   only durable turn write (one write per turn, not two).
2. All consumers re-point: session restore and `/activate_conversation` rebuild `dspy.History`
   by **projection from turn records** (turn records are the source of truth for agent
   cross-turn memory; the in-memory History is a per-session cache). Turn records must always
   carry the projection fields (conversation summary, traces-equivalent text view, feedback
   reference). `/list_conversations` reads metadata records; `/new_conversation` and the
   shutdown hook finalize the metadata record and open the next; `/dump_conversations`
   iterates the keyspace; `/post_feedback` attaches per R38.
3. The conversation id becomes a structural component of every turn key (resolves R9's
   namespace question); with turn-scoped payload keys (R2, pending resolution), deleting a
   conversation prefix co-GCs metadata, turns, and payloads under one lifecycle.
4. Backend: disk/redis split, factory mirroring `get_session_state_store()`. Rdict exits the
   conversation subsystem (remains only for `Workflow` state).
5. Migration: **accept loss** — existing Rdict conversation data is neither read nor migrated;
   no legacy read path (consistent with decision 22).

### A2 — Conversation scoping and switch semantics (resolves R9) — 2026-06-10

1. **Turn namespace:** per A1, the conversation id is structural in every turn key
   (`fw:turn:{channel_id}:{conv_id}:{sortable-ts}-{uuid}`). Per-conversation review listing is
   a prefix scan; channel-wide scans use the channel prefix.
2. **Eager conversation-id reservation:** `active_conversation_id` is guaranteed at session
   creation (restore-last or reserve-new). Required because the turn key is minted at
   logical-turn start (R16). Deployments that never call `/new_conversation` operate in a
   single implicit conversation.
3. **Auto-cancel on conversation switch:** `/new_conversation` and `/activate_conversation`,
   when a turn is suspended (`awaiting_user` or a durable pending blob), first cancel it —
   recording the partial turn under its *original* conversation with `status=cancelled` and
   cleaning up the pending blob plus suspend-offloaded payloads — then proceed with the
   switch. (Today neither endpoint checks suspension at all; the pending turn silently
   survives the switch and the next message resumes a clarification from the previous
   conversation.) The cancel-then-switch sequence runs under the per-session lock, which
   these endpoints currently do not acquire.

### A3 — First-class `TurnStatus`; `awaiting_user` artifact removed (resolves R11) — 2026-06-10

1. **`TurnResult.status: TurnStatus`** with five values: `completed`, `awaiting_user`,
   `failed`, `cancelled`, `abandoned`. `cancelled` is written by `/cancel_pending` and the A2
   auto-cancel paths; `abandoned` is reserved (written only if a stale-pending sweep is built);
   `failed` covers turns whose agent loop raises (R6, pending resolution of capture shape).
2. **The `artifacts["awaiting_user"]` protocol is removed in the same release** that
   introduces `TurnResult`. `_awaiting_user_output` no longer stamps the artifact; the runner
   and the bundled server branch on `turn.status == AWAITING_USER`. No dual-publish window:
   every consumer of the old signal is rewritten for the `TurnResult` return type in the same
   release anyway.
3. **Everything else carries unchanged (user decision: no further cleanup).** The four
   `CommandOutput` predicates and the NLU-internal artifact handshake
   (`command_handled` / `command_name` / `cmd_parameters` between
   `wildcard.py` and `CommandExecutor`) survive the redesign exactly as section 5.2 specifies.
   Reviewer's note for the record: `command_aborted` and `not_what_i_meant` were verified to
   have zero consumers in the framework and tests; they are retained deliberately as public
   API. Formalizing the handshake protocol is possible future work, out of scope here.

### A4 — Cancelled turns are recorded; memory projects completed turns only (resolves R39) — 2026-06-10

1. **Record, don't shred.** `/cancel_pending` and the A2 auto-cancel-on-switch paths write a
   turn record with `status=cancelled` under the turn's *original* conversation: the partial
   event sequence (commands executed so far + the unanswered clarification question; event
   shape per R1) and the payload handles already offloaded at suspend. Payload ownership
   transfers from the pending blob to the cancelled record — no cleanup step; the R24 orphan
   concern is dissolved for the cancel path. Sequence under the per-session lock: serialize
   partial `TurnResult` (status=cancelled) → write record → clear pending blob.
   (`cancel_pending()` today only resets in-memory state; it gains the record write.)
2. ~~**R7 interplay:** when review persistence is disabled by deployment config, cancel falls
   back to shredding, and only that mode performs an explicit suspend-payload delete.~~
   **Superseded by A12:** there is no persistence switch; cancel always records.
3. **Agent-memory projection rule (clarifies A1.2):** the rebuilt `dspy.History` projects from
   `status=completed` records **only**. Cancelled / failed / abandoned records are
   review-and-observability-only and never enter agent working memory — matching today's
   behavior, where a turn that never reached `_finalize_agent_output` never appends to
   conversation history.

### A5 — Failure and abandonment capture (resolves R6; closes R24's remaining path) — 2026-06-10

1. **Failed tool calls are captured with detail.** `_execute_workflow_query` wraps
   `invoke_command` in try/except; on failure it appends
   `CommandOutput(success=False)` with error type, message, and truncated traceback in
   artifacts (traceback visible only in the developer projection), then re-raises so the
   ReAct loop formats the observation for the agent as today. Suspension signals are passed
   through: `AskUserSuspend` subclasses `BaseException` (immune to the wrapper);
   `CommandCancelledError` is explicitly re-raised without capture.
2. **Failed turns: record + re-raise.** `process_message` wraps the agent loop; a fatal error
   writes a partial turn record (`status=failed`, executions so far, exception summary) and
   re-raises. Caller-visible error behavior (HTTP 500s, transient-error retry semantics) is
   unchanged.
3. **Abandoned turns: TTL + lazy filing.** A pending suspended turn older than a configurable
   TTL is not resumed on the channel's next touch: it is filed as `status=abandoned` (partial
   record; payload handles transfer as in A4) and the incoming message starts a fresh turn.
   No background reaper is required for active channels; backlog item `fix-6b4` covers the
   never-touched-again residue as optional future work. UX note: answers arriving after the
   TTL start a fresh turn rather than resuming the stale question.
4. **R24 closure:** with A4 (cancel) and A5.3 (abandon), every suspend-offloaded payload ends
   up owned by a terminal turn record; no orphan class remains. ~~(except review-persistence-off
   deployments, which delete payloads at the terminal transition instead)~~ **Superseded by
   A12:** there is no persistence switch.
5. **R24 residuals (added 2026-06-11 at final confirmation):** (a) never-touched channels —
   the `fix-6b4` reaper deletes the stale pending blob *plus the in-flight turn's payload
   prefix* (trivial under A8's turn-scoped keys); (b) upgrade day orphans nothing — A14's
   graceful expiry clears pre-3.0 pending blobs, which predate the stores (suspend-time
   offload begins at v3.0), so no payload copies exist to orphan.

### A6 — Turn-level success semantics (resolves R40) — 2026-06-11

1. **`TurnResult.success = answer.success`.** Assistant mode: the executed command's own
   success flag (extraction errors / business-logic failures → `False`). Agent mode: `True`
   when an answer is delivered (crashes are `status=failed` per A5). An agent that recovers
   from failed intermediate tool calls reads `True`; per-execution success stays visible in
   the gallery. Documented: `success != all commands succeeded` — the old
   `all(...)` fold is not lifted to the turn level.
2. **Iteration exhaustion:** when the ReAct loop ends by `max_iters` rather than `finish`,
   the synthesized best-effort answer yields `completed` + `success=False`. The loop surfaces
   an exhausted flag to `_finalize_agent_output`; the answer still enters records and the
   completed-only memory projection (A4).
3. **Wire exposure:** `success` is a serialized `computed_field` on `TurnResult` (HTTP/SSE
   visible; today's property never serialized). MCP: `isError = not success`. This decides
   the `success` half of R33; the remaining predicates stay property-only pending R33.

### A7 — ask_user exchanges captured as command executions (resolves R1) — 2026-06-11

Supersedes the reviewer-proposed discriminated-union event sequence with a simpler design
(Dhar): **`ask_user` is modeled as a command execution**, not a new event type.

1. Each exchange is a `CommandOutput` with `command_name="ask_user"`,
   `command_parameters=` the clarification question, `command_response.response=` the user's
   answer — appended **in chronological order** into the same `command_outputs` list as real
   command executions. `TurnResult` keeps its original shape (answer + `command_outputs`);
   interleaving needs no union types. Precedent: `_ask_user_tool` already constructs such a
   `CommandOutput` for the output queue.
2. **Unanswered-question convention:** appended at ask/suspend time with `response=""` and
   `success=False`; filled (and `success=True`) on answer via `_post_ask_user_response` (the
   choke point in both topologies). Cancelled/abandoned records (A4/A5) therefore end with
   the unanswered question; suspended partials show the pending question.
3. **`user_message` holds the original request only**; clarification replies live solely in
   their ask_user entries. (R44 will decide raw-vs-refined capture separately.)
4. **Role inversion documented:** in ask_user entries, `command_parameters` is the agent's
   utterance and `response` is the user's. The summary text projection maps today's
   `{"agent_query", "user_response"}` to `parameters`/`response` directly, so the derived
   LLM-summary view loses nothing and decision 18's zero-regression claim is restored.
5. Knock-ons: per-`CommandOutput` timestamps (R29) cover exchanges uniformly; R28's
   trajectory remains an offloaded blob; R30's user projection includes ask_user entries via
   `command_name` filter.

### A8 — Turn-scoped payload keys; record-mediated access (resolves R2; supersedes decisions 13 and 19) — 2026-06-11

1. **Payload keys are turn-scoped:**
   `fw:payload:{channel_id}:{conv_id}:{turn_key}:{content_hash}`. Each payload is owned by
   exactly one turn; sharing is impossible by construction. `PayloadStore.put` takes scope
   parameters (channel, conversation, turn key) and returns the full scoped key as the opaque
   handle. Content-addressing survives only as the leaf segment — within-turn retry
   idempotency, which is the only idempotency that matters. Cross-turn dedup is deliberately
   abandoned; tenant isolation becomes structural.
2. **Co-GC is literal:** deleting a conversation prefix removes conversation metadata (A1),
   turn records, and payloads in one stroke (disk: directory tree; Redis: prefix SCAN+DEL).
   The "one lifecycle key" of decision 19 is the conversation prefix — implementable as
   delegated. Readers still tolerate missing payloads as defense in depth.
3. **Record-mediated access contract:** payload reads always authorize the referencing turn
   record first, then resolve handles found inside it; bare-handle fetch endpoints are
   forbidden. Closes the hash-oracle / cross-tenant-probe exposure.

### A9 — `next_actions`/`recommendations` policy; gallery provenance (resolves R42) — 2026-06-11

Empirical basis: `next_actions` and `recommendations` have **zero producers and zero
consumers** in the framework, examples, and tests; xray's action buttons travel via
`artifacts`, which the design preserves. Post-A7, durable records keep complete
`CommandResponse`s, so these fields are automatically preserved if they ever gain producers.

1. **Gallery rule stays payload-bearing only** (conservative; user decision). Documented and
   accepted: an output bearing only buttons/recommendations would appear in the durable
   record but not the live gallery — currently unreachable (zero producers), recoverable from
   the record if it occurs.
2. **The agent answer's `next_actions`/`recommendations` stay empty.** Section 5.4's
   motivation is corrected in place (see the bracketed note there).
3. **Gallery provenance:** `TurnResult` preserves full per-entry provenance
   (`command_name`, `command_parameters`, `workflow_name`, `context`), but xray's
   `ResponseTuple` has no command-identity slot, so provenance dies at that wire format. xray
   will initially embed provenance in response text; extending `ResponseTuple` is xray-repo
   scope, out of scope for fastWorkflow — recorded as a note on the section 10.3 mapping.

### A10 — Artifact serialization and offload contract (resolves R5) — 2026-06-11

1. **Offload rule — size threshold:** any `str`/`bytes` artifact value above a configurable
   threshold (default ~4 KB) is offloaded to the turn-scoped `PayloadStore` (A8) and replaced
   in place by the **envelope** `{"__fw_payload_ref__": <scoped handle>, "size": <bytes>,
   "content_type": <best-effort>}`; smaller values stay inline. The marker key is reserved.
2. **Strict rejection for non-serializable values:** record serialization raises — with an
   error naming the offending artifact key and command — on any non-JSON-serializable
   artifact value, replacing today's silent `default=str` mangling
   (`session_state_store.py:64`). Verified safe: `Action`/`Recommendation` are typed fields,
   not artifacts, and unused (A9); xray's mapping packs `ResponseTuple.actions` into
   artifacts but that value is `None` in the verified flow; the framework's own
   `cmd_parameters` object in the NLU handshake never reaches serialization. Apps that pack
   objects into artifacts must serialize them first in their own mapping code.
3. **`command_parameters` honesty:** the in-memory field keeps the typed Pydantic instance;
   the declared type is corrected from the current lie (`str`); record serialization emits
   `model_dump()` as a dict.
4. Store-level details (hash algorithm, compression slot, atomic disk writes) remain with
   R21.

### A11 — `process_action` invocations are full turns (resolves R8) — 2026-06-11

`process_action` (the external `/perform_action` door; MCP rides the same endpoints) returns
a `TurnResult` and writes a turn record like any other turn:

1. `command_outputs = [the one execution]`; `answer = command_outputs[0].command_response`
   (assistant-path aliasing; copy-on-serialize applies); `status` per A3; `success` per A6;
   `user_message = ""` (no user words existed — provenance lives on the command entry).
2. Rationale: the state-mutating path must never be the unrecorded one (audit); A1 rebuilds
   agent memory from records, and `_process_action` writes a memory note today
   (`workflow_execution_context.py:715`) that would otherwise vanish on restart; one wire
   shape across all endpoints.
3. No record opt-out flag — the audit guarantee is unconditional. The internal
   `CommandExecutor.perform_action` helper (plumbing inside a turn) is unaffected.

### A12 — No persistence switches; retention is the compliance story (resolves R7) — 2026-06-11

1. **Records and payloads always persist — no opt-out configuration exists.** Post-A1 the
   turn records are load-bearing (conversation system of record, agent-memory source);
   payload durability is the product. The "review persistence disabled" fallback clauses in
   A4.2 and A5.4 are superseded (struck through above): cancel and abandon always record,
   payloads always transfer to their terminal record.
2. **Compliance = age-based retention, infra-executed (decision 19 affirmed).** Deployments
   remove conversations older than their chosen age via conversation-prefix deletes — A8
   makes this a single-operation co-GC (disk: directory-tree removal; Redis: prefix
   SCAN+DEL). The framework ships no retention code or schedule; the documented contract is
   the deliverable.
3. **Security contract (extends section 7.8):** both stores are declared to hold
   PII/entitlement-grade data. Deployment obligations: encryption at rest, TLS to Redis,
   least-privilege access for the service identity. Framework obligations: record-mediated
   payload access only (A8); record contents never written to logs.

### A13 — Wire-contract break policy (resolves R3) — 2026-06-11

1. **Constructor shim for command authors:** `CommandOutput` gains a
   `model_validator(mode="before")` accepting the legacy `command_responses=[x]` keyword,
   mapping it to `command_response` with a `DeprecationWarning`. Hand-written workflows keep
   running across the upgrade; the legacy keyword is removed at the major release following
   its introduction (train per R46).
2. **HTTP/SSE/MCP hard-break at the major:** endpoints return the `TurnResult` shape; no
   `/v2` endpoints, no dual-shape responses, no reverse-mapping. Consistent with A3's
   no-dual-publish stance; the bundled server's clients are org-controlled, and xray embeds
   the framework behind its own API.
3. **Deliverable: a schema-migration guide** covering HTTP bodies, the SSE final event, the
   MCP result (`isError = not success`, A6), and the author-side constructor migration.

### A14 — Release train and upgrade-day behavior (resolves R46) — 2026-06-11

1. **Quick-fix minor + one major** (the three-stage sketch was retired: A1's accept-loss
   store consolidation is user-visible data loss and must not hide in a minor):
   - **~v2.21 (minor, non-breaking):** `TurnResult`, the capture machinery (A5, A7), a new
     `process_turn()` returning `TurnResult`, and the A13 shim with deprecation warnings.
     `process_message` unchanged. xray switches to `process_turn()` — the original payload
     bug is fixed in this release.
   - **v3.0 (major):** `process_message` returns `TurnResult` (A3's artifact removal lands
     with it), wire hard-break (A13), `ConversationTurnStore` consolidation with announced
     data loss (A1), `action_log` retired. The shim continues accepting the legacy keyword.
   - **v4.0:** shim removed. (A13's window pinned: introduced 2.21, removed 4.0.)
2. **Upgrade day — graceful expiry:** a pending blob with an old `SCHEMA_VERSION` is cleared
   on first touch and the user is told "your previous question expired, please re-ask." No
   record is synthesized. The migration guide documents this and recommends a pre-upgrade
   drain (answer or cancel pending turns).

### A15 — Complete `command_responses` migration inventory (resolves R4; completes section 11) — 2026-06-11

Verified sites missing from section 11, with migration releases per A14:

- **v2.21** (generators emit new style; in-repo code migrates under the shim):
  `build/command_file_template.py:134,364`; `build/command_stub_generator.py:285`;
  `build/__main__.py:310`; `_workflows/command_metadata_extraction/_commands/` (~8 files,
  incl. `wildcard.py`'s artifact mutations at lines 86/123); all bundled examples (45
  files); the test workflows (`tests/example_workflow`, `tests/todo_list_workflow`,
  `tests/hello_world_workflow`, ~43 command files).
- **v3.0** (migrate with the surfaces they exercise): `run/__main__.py` CLI printer (R43);
  `run_fastapi_mcp/utils.py:355-357` awaiting check (A3); ~15 test modules asserting on
  `.command_responses[0]` / `artifacts["awaiting_user"]` (notably
  `test_execution_context_agent.py`, `test_fastapi_service.py`,
  `test_session_state_serialization.py`).
- **`refine` pipeline: audited clean** — zero references.

### A16 — Live response path serves from RAM; inline cap (resolves R41) — 2026-06-11

1. **Live path = in-memory.** The runner/mapping builds the user's response from the
   in-memory `TurnResult`; payload offload happens on a serialized copy at the persistence
   boundary (R10) and exists purely for review. Zero store reads on the hot path. Section
   10.3's "fetching payloads lazily by handle" applies **only to the review reader**.
2. **Bundled-server response bodies inline payloads up to a configurable cap** (generous
   default, ~10 MB); above it the A10 envelope is returned instead, marked
   not-inlined-available-via-review-record (depends on the future R32 read API; documented).
3. **R45 consequence:** payloads remain in RAM until the response is built, so eager offload
   provides no memory relief; R45 resolves to boundary-offload with a documented memory
   profile (finalized in its own pass).

### A17 — Boundary offload; memory profile; rehydration fetch fallback (resolves R45) — 2026-06-11

1. **Boundary offload** (eager offload rejected as moot — A16 pins the live path to RAM).
2. **Memory profile, documented and accepted:** peak RAM per turn ≈ sum of the turn's
   payloads, held from capture until the response is built. Tables are producer-capped;
   chart payloads are full-frame (xray-side); multiply by concurrent in-flight turns.
3. **Rehydration exception:** after a pod restart mid-suspension, restored partial turns hold
   envelopes only; the live mapping treats envelope entries as **fetch-by-handle** — the one
   legitimate live-path store read (server-side; A8 record-mediation holds). This is the
   correct scope of 10.3's "lazily by handle" wording.

### A18 — Feedback as separate records in the unified keyspace (resolves R38) — 2026-06-11

1. **Feedback record type:** `fw:feedback:{channel_id}:{conv_id}:{turn_key}` holding score,
   comment, timestamp. Turn records remain strictly write-once (R16 enforcement and audit
   immutability intact, no carve-outs).
2. **Read-time join by prefix:** the review reader and the A1 memory rebuild fetch turns and
   feedback in the same conversation-prefix scan; A8/A12 retention deletes cards with their
   envelopes.
3. **Conventions (match today):** re-posting overwrites (last-write-wins);
   `/post_feedback` targets the latest completed turn of the active conversation
   (arbitrary-turn feedback = future R32 territory); feedback enters the agent-memory
   projection for completed turns, as the dspy.History `feedback` slot does today.

### A19 — Queue contract: status-stamped `TurnResult`s (resolves R43) — 2026-06-11

1. **`command_output_queue` carries `TurnResult`s only.** Mid-turn ask_user questions are
   partial `TurnResult`s with `status=awaiting_user` (synthesized from in-memory state);
   completion is `status=completed/failed`. Consumers branch on `status` — the same branch
   Topology B callers use on `process_message`'s return, extending 5.5's no-mode-branching
   promise to the queue transport.
2. **The trace queue is untouched** (decision 9 stands): same events, same emission sites,
   same `None` sentinel. The CLI's dim ticker is byte-identical before and after; panel
   rendering verified pixel-identical under the new contract.
3. **Sentinel pairing rule (fixes bug `fix-5fv`):** every mid-turn awaiting_user enqueue is
   paired with a trace sentinel. Today's Topology A ask_user enqueues the clarification
   without one, and the CLI reads the output queue only after a sentinel — an apparent
   agent-mode mid-turn hang, filed for runtime verification and fixed by this rule.
4. **Timing:** lands at v3.0 with the CLI surface (A14/A15).

### A20 — Selective copy-on-serialize; headline never carries payloads (resolves R10) — 2026-06-11

1. **Selective copy-on-serialize:** the serializer never mutates live
   `TurnResult`/`CommandOutput` objects and never blind-deep-copies them (live objects in
   artifacts; typed model in `command_parameters`). It builds a new structure — small fields
   copied, values converted per A10 — leaving originals untouched, as A16's
   pristine-in-RAM live path requires.
2. **Headline never carries payloads:** the headline `ResponseTuple` is narrative text +
   metadata, always; the gallery contains **all** payload-bearing outputs in turn order,
   always. One invariant for every turn type; decision 21 honored (the UI picks the featured
   payload); deterministic-turn duplication eliminated by construction.

### A21 — The `TurnSerializer` component (resolves R25) — 2026-06-11

1. **Single owner of the light-record pass:** selective copy (A20), threshold offload +
   envelope substitution via scoped `PayloadStore` puts (A10/A8), `model_dump()` for
   `command_parameters`, strict rejection with key-and-command-named errors (A10),
   schema-version stamping (fields per R19), recursive `nested_turn` descent (arity per R12).
2. **Both persistence boundaries call it** — suspend (`serialize_state` partial) and every
   terminal write (completed / cancelled / abandoned / failed). One implementation; the
   suspend-blob and turn-record formats cannot drift.
3. **Owns the reader too:** envelope detection, lazy payload resolution for the review
   reader, tolerate-missing-payload.
4. **Placement:** transport-free core beside the stores; callable from WEC, runner, and
   bundled server; no FastAPI dependencies.

### A22 — Turn-key minting (resolves R16) — 2026-06-11

1. **Minted once at logical-turn start, by the WEC** (when accumulation starts); rides on the
   `TurnResult`; runner and stores read it, never re-mint. Stable across retries and across
   suspend/resume — one logical turn = one key = one record (makes 7.6 mechanical). A2's
   eager conversation-id reservation exists to enable this.
2. **Write-once enforced where free:** Redis `SET NX`; disk `O_EXCL`. Collision during a
   retry = idempotent success (debug log); collision otherwise = loud error (key-minting bug).
3. **Mid-flight referenceability** is a deliberate benefit: the turn id exists from turn
   start for logs, traces, and observability metadata (R28/R29/R31).

### A23 — Turn listing: summary cards, pagination, no sharding (resolves R17) — 2026-06-11

1. **Listing returns (key, summary-card) pairs:** ordinal, started/completed timestamps,
   status, success, truncated `user_message`, summary-if-present, command count (extensible
   via R19 metadata). Redis: `SCAN` + pipelined `MGET` — no write-time secondary index
   (reserved as future optimization). Disk: directory read.
2. **Pagination:** `limit` + `before`/`after` time-range params; newest-first default.
3. **No date sharding:** A1's `{channel}/{conv_id}/` layout bounds per-directory file counts
   by conversation length; the original concern predates the keyspace.

### A24 — Stored ordinal; key timestamp = turn start (resolves R18) — 2026-06-11

1. **Per-conversation ordinal stored in each record** (incremented under the session lock);
   authoritative for ordering — clock-skewed timestamps never override it.
2. **Counter restored for free:** the A1 memory-rebuild pass takes `max(ordinal) + 1` while
   reading the conversation's records; no separate counter storage.
3. **Key timestamp = logical-turn start** (implied by A22's mint-at-start); used for range
   scans (A23) and readability, not ordering.

### A25 — Record format versioning; `TurnResult.metadata` (resolves R19) — 2026-06-11

1. **Every durable record type embeds a format version** (turn records, conversation
   metadata, feedback cards). The `TurnSerializer` stamps its format version on every
   serialized `TurnResult` — in turn records and in the partial inside pending blobs alike;
   the pending blob keeps its separate envelope `SCHEMA_VERSION` (A14 expiry dispatch).
2. **Strict reader dispatch:** unknown future version → explicit error; missing version →
   corrupt, error.
3. **`TurnResult.metadata: dict[str, Any] = {}`** on the in-memory type, serialized as-is
   under A10's contract — the home for R29 timing/token slots, R31 trace ids, and A23 card
   extensions.

### A26 — Key grammar and path sanitization (resolves R20) — 2026-06-11

1. **Turn-key grammar:** `<compact-ts>-<uuid-hex-12>`, timestamp `YYYYMMDDTHHMMSS.ffffffZ`
   — colon-free (NTFS-legal), sortable, human-readable.
2. **Allowlist sanitization for every disk path component** (`[A-Za-z0-9._-]`; encode the
   rest; reject empty/`.`/`..`) applied to `channel_id`, `conv_id`, `turn_key` — defense in
   depth. One shared sanitizer; the pending store's weaker separator-only `safe_id`
   (`session_state_store.py:51`) is retrofitted onto it at v3.0.

### A27 — PayloadStore conventions (resolves R21) — 2026-06-11

1. **Hashing:** UTF-8-encode `str` payloads; SHA-256 hex truncated to 32 chars; leaf segment
   algorithm-prefixed (`sha256-<hex32>`). `get()` returns bytes; the envelope's
   `content_type` governs decoding.
2. **Metadata lives in the A10 envelope** (`size`, `content_type`); the store is
   raw-bytes-only. **Compression reserved:** optional `content_encoding` envelope slot; not
   built now.
3. **Atomic disk writes:** temp file + `os.replace()`.
4. **No delete surface on the ABC** — dissolved by A8 (prefix delete), A12 (infra-executed
   retention), A5.5 (reaper turn-prefix variant).

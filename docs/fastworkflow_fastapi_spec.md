### FastWorkflow FastAPI Service — Specification

#### 1. Overview
- **Goal**: Expose FastWorkflow workflows as a FastAPI web service, enabling clients to initialize for a given workflow (per channel), then interact in agent mode (forced). If a query starts with `/`, all leading slashes are stripped before processing. The service supports explicit actions, resetting conversations, listing conversations, dumping all conversations to JSONL, and posting feedback. `invoke_agent` returns the turn's `TurnOutput` projection (see §6a) and, when enabled, includes collected trace events in the final response. Streaming is supported via NDJSON or SSE at `/invoke_agent_stream`, and MCP tools map to the same NDJSON-based streaming implementation.
- **Source parity**: Behavior mirrors the CLI runner in `fastworkflow/run/__main__.py` while replacing its interactive loop with synchronous and streaming HTTP endpoints.

#### 2. Non‑Goals
- No UI; only REST with OpenAPI/Swagger at `/docs`.
- No WebSocket support. For streaming, use NDJSON or SSE at `/invoke_agent_stream`. MCP Streamable HTTP is mounted on the same FastAPI app and maps to the same server-side streaming (NDJSON).

#### 3. References
- CLI runner (`fastworkflow/run/__main__.py`) uses **Topology A**: a `ChatSession` chassis around a `WorkflowExecutionContext`, with queues and (for `keep_alive`) a worker thread that **blocks** on `ask_user`.
- This FastAPI service uses **Topology B**: no ChatSession worker, no request-path queues. The server embeds a transport-free `WorkflowExecutionContext` per channel and drives it with `process_turn` / `process_action_turn`. `ask_user` **suspends** the agent trajectory and returns; the next message resumes it.
- Parity with the CLI that still matters: environment loading, startup command/action handling, agent vs assistant (`/`-prefix) routing, and the public `TurnOutput` shape (§6a).

#### 4. Architecture Summary

**Topology B (current).** Implementation lives in `fastworkflow/run_fastapi_mcp/` (`__main__.py`, `utils.py`, `turns.py`).

- FastAPI app with a process-wide `ChannelSessionManager`: an in-memory LRU of live `ChannelRuntime` objects keyed by `channel_id`, plus durable stores for conversations and suspended session state.
- For each live `channel_id`, a `ChannelRuntime` holds:
  - A `WorkflowExecutionContext` (`run_as_agent=True`) — synchronous, transport-free; no `user_message_queue` / `command_output_queue` on the FastAPI path.
  - A per-channel `asyncio.Lock` held for the duration of one turn *attempt* (released on terminal outcome **or** `awaiting_user`, never held across a suspension).
  - A `ConversationStore` (Rdict, one DB file per channel): `conversation_id`, `topic`, `summary`, timestamps, per-turn history, optional feedback.
  - Session metadata: stream format, startup state / idempotency, session incarnation, durable turn high-water mark.
- **Turns engine** (`TurnRegistry` in `turns.py`): every unit of work is a registered `TurnExecution`. Endpoints call `submit_turn` (wait-or-defer): wait up to `timeout_seconds`; if still running, return **202** while the execution keeps going. The registry's per-channel **active-execution pointer** is the source of truth for liveness and the 409 busy guard (not `lock.locked()`).
- Blocking WEC work runs in `loop.run_in_executor`; a `ContextVar` stack isolates the active workflow per thread/task so concurrent channels are safe in one process.
- Trace events: collected into the non-streaming response when enabled; for `/invoke_agent_stream`, emitted live as NDJSON/`SSE` `trace` records, then a terminal `output` carrying bare `TurnOutput`.
- Suspended (`awaiting_user`) state is persisted via `SessionStateStore` so a worker can cold-rehydrate after eviction or restart.

**Not Topology A.** Do not enqueue a message and wait on `command_output_queue`. Do not assume a ChatSession daemon worker. The terminal streaming event is `output` (TurnOutput), not `command_output`.

**Not yet shipped (fix-85g Step 2).** `GET /turns/{turn_key}`, a durable poll/replay surface, sized global executor backpressure (429), and TTL eviction of retained terminal executions are **design**, not current API. Today a deferred client **retries the same request**; same idempotency key rejoins the same execution. Do not document Step 2 as live.

#### 5. Channel and Conversation Lifecycle
1) Client calls `POST /initialize` with `channel_id` (required) and optional `user_id`, plus optional startup command/action and `stream_format`.
2) Server:
   - Loads environment from `env_file_path` and `passwords_file_path` only; calls `fastworkflow.init(env_vars=...)`.
   - Ensures a Topology-B runtime via `ensure_user_runtime_exists` / `ChannelSessionManager` (single-flight per channel): builds a `WorkflowExecutionContext`, binds the app workflow, restores conversation history and any durable pending suspension / checkpoint when present.
   - If a startup command/action is provided, submits it as `submit_turn(..., kind="initialize_startup")` (wait-or-defer).
   - Returns JWT tokens (`access_token`, `refresh_token`, `token_type`, `expires_in`). When startup was requested: `startup_output` (TurnOutput, §6a) on completion; or **202** with `startup_turn_key` + `startup_exec_state: "running"` while it is still running. The "already exists" branch never returns a silently empty `startup_output` for a still-running startup — it reports the same three-state status.
3) Client uses the JWT access token with `/invoke_agent`, `/invoke_agent_stream`, `/invoke_assistant`, `/perform_action`, `/new_conversation`, etc. Trusted mode may omit encryption of JWTs. Additional endpoints: `/conversations`, `/activate_conversation`, `/abandon_clarification`. Admin: `/admin/dump_all_conversations`.
4) On `new_conversation` (and similarly at shutdown persistence paths):
   - Generate `topic` (unique per channel; case- and whitespace-insensitive; integer suffix if needed) and `summary` via `dspy.ChainOfThought()`.
   - If generation succeeds, persist to Rdict and rotate to a new internal conversation; if it fails, log critically, do **not** persist, do **not** rotate.
5) Conversation histories and suspended pending state can survive process restart (Rdict + session-state store). Cold rehydrate does not re-run startup when startup already completed durably.
6) `ask_user` (Topology B): the turn ends with HTTP 200 / stream `output` and `status: awaiting_user` (clarification in `answer`). The next user message on the same channel resumes the same **logical** turn; it is a new **execution** in the registry.

#### 6. Endpoints

1) POST `/initialize`
- Purpose: Create or resume a Topology-B `WorkflowExecutionContext` for a `channel_id` and start the workflow. Optionally execute a startup command/action as the channel's first logical turn and return its `TurnOutput` (§6a).
- Request (InitializationRequest):
```json
{
  "channel_id": "channel-123",
  "user_id": "user-9",               
  "conversation_id": null,
  "stream_format": "ndjson",
  "startup_command": "load_workflow ...",
  "startup_action": { "command_name": "User/get_details", "arguments": {"channel_id": "u-42"} }
}
```
- Rules:
  - `channel_id` is required.
  - Exactly one of `startup_command` or `startup_action` may be provided (or neither).
  - If startup is provided, `user_id` is required.
  - `stream_format`: `ndjson` (default) or `sse` for `/invoke_agent_stream`.
- Response:
  - TokenResponse + optional `startup_output` (the startup turn's `TurnOutput` projection, §6a). Absent while the startup turn is still running (202, poll via `startup_turn_key`).
- Notes:
  - `channel_id` is embedded in JWT `sub`; `user_id` (when provided) is embedded in `uid`.
  - Workflow definition can be obtained by calling the `what_can_i_do` command (IntentDetection context)
- Errors:
  - 500 initialization failure (details logged)

2) POST `/refresh_token`
- Purpose: Exchange a valid refresh token for a new access token (and rotated refresh token).
- Request: Header `Authorization: Bearer <refresh_token>`
- Response (TokenResponse): same shape as `/initialize` (without `workflow_info`).
- Errors:
  - 401 invalid or expired refresh token
  - 404 session not found

3) POST `/invoke_agent`
- Purpose: Submit a natural language query to an agentic session for a channel (synchronous response). Leading `/` characters are permitted and stripped for compatibility; assistant semantics remain exclusive to `/invoke_assistant`.
- Headers:
  - `Authorization: Bearer <access_token>` (contains `sub` and optional `uid`)
- Request:
```json
{ "user_query": "find orders for channel 42", "timeout_seconds": 60 }
```
- Behavior:
  - Validate the channel session exists. Agent mode is always enabled.
  - If `user_query` begins with `/`, strip all leading slashes before processing (compatibility path).
  - When the turn completes, return the turn's `TurnOutput` projection including collected `traces`.
- Response: turn response (§6a).
- Errors:
  - 404 channel not found
  - 409 concurrent turn already in progress for this channel (different idempotency key)
  - 202 deferred: wait window elapsed; execution still running — retry the same request to rejoin (see §6a `turn_key`)
  - 500 unexpected error

4) POST `/invoke_agent_stream`
- Purpose: Submit a natural language query to an agentic session and stream trace events in real-time, followed by the turn's final `TurnOutput`. Leading `/` characters are permitted and stripped.
- Headers: Same as `/invoke_agent`.
- Request:
```json
{ "user_query": "find orders for channel 42", "timeout_seconds": 60 }
```
- Behavior:
  - Validate the channel session exists. Agent mode is always enabled.
  - If `user_query` begins with `/`, strip all leading slashes before processing.
  - Emit streaming records as they are available:
    - NDJSON: `{ "type": "trace", "data": <trace_json> }` (multiple), then `{ "type": "output", "data": <TurnOutput_json> }` (final)
    - SSE: `event: trace` (multiple) and final `event: output` with JSON payloads
  - Only the final output record is streamed if no traces were produced.
  - The terminal `output` event carries the bare `TurnOutput` (§6a) — no
    `exec_state` (a stream has no deferral state to report).
  - A turn that fails, or that suspends to ask the user something, still arrives as an `output` event (with `status` `failed`/`awaiting_user`). The `error` event is reserved for transport failures.
- Response: HTTP 200 with `Content-Type: application/x-ndjson` (NDJSON) or `text/event-stream` (SSE).
- Errors:
  - 404 channel not found
  - 409 concurrent turn already in progress for this channel
  - On transport failure, send a terminal record `{ "type": "error", "data": { "detail": "..." } }` then close the connection (a turn with `status: failed` or `awaiting_user` is still an `output` event, not `error`)

5) POST `/invoke_assistant`
- Purpose: Deterministic/assistant invocation for a channel. The server accepts plain queries; clients need not prefix `/`.
- Headers: Same as `/invoke_agent`.
- Request:
```json
{ "user_query": "load_workflow file='...'" }
```
- Behavior: Same execution path as agent, but uses assistant path (no planning). The `/`-prefixing that selects the assistant path affects routing inside the execution context, not the response type.
- Response: turn response (§6a).
- Errors: as above.

6) POST `/perform_action`
- Purpose: Execute a specific workflow action chosen by the client (e.g., from `next_actions`).
- Headers: Same as `/invoke_agent`.
- Request:
```json
{ "action": { "command_name": "User/get_details", "arguments": { "channel_id": "u-42" } }, "timeout_seconds": 60 }
```
- Behavior:
  - Validate session exists.
  - Invoke through the same single‑turn path used for NL queries, but bypass parameter extraction (directly execute the provided `Action`). Each direct action is its own logical turn.
  - Wait for the turn (or defer) and return it.
- Response: turn response (§6a).
- Errors: 404/409/202/500 as above; 422 invalid action shape.

#### 6a. Turn response contract (`TurnOutput`)

Every turn surface answers with the same shape: `/invoke_agent`,
`/invoke_agent_stream`, `/invoke_assistant`, `/perform_action`, and
`/initialize`'s `startup_output`. This replaced a per-endpoint mix in which some
returned the public `TurnOutput` and others returned the older `CommandOutput`,
forcing integrators to handle both. See
`docs/turn_result_design_final.md` sections 1a, 8 and 14.

**Breaking wire change (v3.0):** the CommandOutput-shaped top level is gone, and
inside each `command_outputs` entry the field is now singular
`command_response` (not `command_responses`). The keys that moved off the top
level are `workflow_name`, `context`, `command_name`, `command_parameters` and
the response payload — they live per command under `command_outputs[*]`.

The public projection (`fastworkflow/turn.py`, `TurnOutput`):

```json
{
  "turn_key": "20260807T193000.123456Z-a1b2c3d4e5f6",
  "status": "completed",
  "failure_reason": null,
  "answer": "The sum is 5.",
  "command_outputs": [ /* one CommandOutput per command executed in the turn */ ],
  "success": true
}
```

- `status` (`TurnStatus`): `completed` | `awaiting_user` | `failed` | `cancelled` | `abandoned`.
- `failure_reason`: elaboration of a failure status (e.g. `max_iters_exhausted`), else `null`.
- `answer`: the turn's final answer text — the agent's synthesized answer, or the deterministic command's response text. When `status` is `awaiting_user`, this is the clarification question.
- `command_outputs`: per-command provenance; each entry has a singular `command_response` with `response` / `success` / `artifacts` / timing.
- `success`: a computed field, `all(command_outputs succeeded)`. Deliberately **orthogonal** to `status` — the agent phrases its answer as if it succeeded, so this is the framework's signal that some command returned a failure code even when the agent recovered from it or masked it in prose.

The non-streaming endpoints add two keys to that projection:

- `exec_state`: the transport's own lifecycle (`running` | `done`), not the turn's outcome. A deferred turn returns `202 {turn_key, exec_state: "running"}` and nothing else; retrying the same request rejoins the same execution.
- `traces`: present when trace events were collected.

**Migration (v3.0):** clients that previously read top-level `command_responses`
or nested `command_outputs[i].command_responses[0]` must switch to `answer`
and/or `command_outputs[i].command_response`. The Python constructor accepts
only `command_response=...` — `command_responses=[...]` is rejected.

##### Two `turn_key` meanings (deliberately distinct)

The JSON field name `turn_key` is used in **two different key spaces** on this API.
They are not interchangeable. Renaming one (e.g. to `execution_key`) is a
breaking wire change and is **not** done here; clients must treat the meanings
as deliberately distinct.

| Where you see `turn_key` | Key space | Minted by | Survives `ask_user`? | Use it to |
|---|---|---|---|---|
| Non-streaming body (`/invoke_agent`, `/invoke_assistant`, `/perform_action`) and `/initialize`'s `startup_turn_key` | **Execution key** | `TurnRegistry` (`TurnExecution.turn_key`) | No — each HTTP submission / resume attempt gets a new execution | Identify the in-flight registry execution; rejoin after **202** by **retrying the same request** (same args → same idempotency key). Today there is **no** `GET /turns/{turn_key}` (that is fix-85g Step 2, not shipped). |
| Streaming terminal `output` event (`/invoke_agent_stream`); also `startup_output.turn_key` when `/initialize` returns a full `TurnOutput` | **Logical turn key** | `WorkflowExecutionContext` (`TurnOutput.turn_key`) | Yes — one logical turn spans suspension and resume | Correlate conversation / TurnResult identity across clarifications. **Never** use this as a 202 poll / rejoin handle. |

On a non-streaming **200** that flattens the turn into the HTTP body, the top-level
`turn_key` field is the **execution** key (see `render_turn_response`). The
logical key is not separately exposed on that flattened body. On `/initialize`
**200** with `startup_output`, both appear: `startup_turn_key` (execution) and
`startup_output.turn_key` (logical) — and they usually differ.

**Worked correlation example**

1. Client posts `POST /invoke_agent` with a long-running query. Wait window
   elapses → **202** `{ "turn_key": "20260808T120000.000001Z-aaaaaaaaaaaa", "exec_state": "running" }`.
   That string is an **execution** key. Store it only as a label for logs; to
   recover the result, **retry the same POST** (same `user_query`). The server
   rejoins that execution via idempotency and eventually returns **200** with
   the same execution `turn_key` plus `status` / `answer` / `command_outputs`.
2. Separately, client posts `POST /invoke_agent_stream` for another query. The
   stream ends with `event: output` / NDJSON `{"type":"output","data":{...,"turn_key":"20260808T120100.000002Z-bbbbbbbbbbbb", "status":"awaiting_user", ...}}`.
   That `turn_key` is the **logical** turn key. If the client later polls or
   retries using it as if it were an execution key, the registry will not find
   that execution (**404** / no match) — or, worse, a client that stores
   "the turn_key" from a stream and later compares it to a non-streaming body's
   `turn_key` will silently mismatch across an `ask_user` resume, because the
   next non-streaming attempt mints a **new** execution key while the logical
   key in a subsequent stream `output` stays the same.
3. Correct pairing after a clarification: send the user's answer as a **new**
   `/invoke_agent` or `/invoke_agent_stream` on the same channel (resume). Do
   not look up the previous stream's logical `turn_key` in the registry. If you
   need both identities after `/initialize`, read `startup_turn_key` (execution)
   and `startup_output.turn_key` (logical) from the same 200 payload.

**HTTP status is not the turn outcome.** A turn that fails, or that suspends to
ask the user something, is a *successful call*: HTTP 200 with the outcome in
`status`/`failure_reason`/`success`. HTTP codes describe the transport — the
session was missing (404), another turn holds the channel (409), the request's
wait window elapsed without a terminal result (**202** deferred — retry to
rejoin; not a hard abort), the server broke (500). Collapsing transport and
outcome would leave a client unable to tell "the workflow could not finish"
from "the server is broken".

**MCP `isError` is deliberately not mapped** to `not success` (fix-qtq.6).
`fastapi-mcp` 0.4.0 exposes no hook: it answers from a private
`_execute_api_tool` returning `list[TextContent]`, and the MCP SDK marks a result
`isError` only when the handler raises — which that function does solely for HTTP
4xx/5xx. The only available lever is therefore the HTTP status code, and using it
would destroy the transport/outcome separation above. The streaming
`invoke_agent` tool could not read `success` anyway: its body is a stream, so
there is no top-level field to map. MCP clients read the outcome from the
`TurnOutput` in the tool result body. See the TODO in
`fastworkflow/run_fastapi_mcp/mcp_specific.py`.

7) POST `/new_conversation`
- Purpose: Persist and close the current conversation (topic + summary via GenAI), then reset history and start a new internal conversation.
- Request:
```json
{ "channel_id": "channel-123" }
```
- Behavior:
  - Generate `topic` (unique per channel; append integer suffix if needed) and `summary` synchronously using `dspy.ChainOfThought()`.
  - If generation succeeds, persist conversation `{topic, summary, history}` in Rdict and rotate; if it fails, log critical, return 500, and do not rotate.
- Response: `{ "status": "ok" }`.
- Errors: 404 if channel missing.

8) POST `/post_feedback`
- Purpose: Attach optional feedback to the latest turn in the current conversation for a channel.
- Request:
```json
{
  "channel_id": "channel-123",
  "binary_or_numeric_score": true,
  "nl_feedback": null
}
```
- Rules:
  - A conversation is a list of turns: `[ {"conversation summary": str, "conversation_traces": str, "feedback": dict|null}, ... ]`.
  - At least one of `binary_or_numeric_score` or `nl_feedback` must be provided. Both may be provided.
  - Feedback always applies to the latest (most recent) turn in the active conversation.
- Behavior:
  - Validate presence (reject only when both are null); store feedback on the latest turn in `ConversationStore` with a timestamp.
  - Feedback is optional per turn; multiple feedback updates overwrite the previous entry for that turn.
- Response: `{ "status": "ok" }`.
- Errors: 404 channel missing; 422 invalid input (both fields null).

9) GET `/` (root)
- Simple HTML page with a link to `/docs`. Serves also as a health check (no dedicated `/healthz`).

#### 7. Data Models

Pydantic model sketches (for reference; actual code will import FastWorkflow types where available):

```python
class InitializationRequest(BaseModel):
    channel_id: str
    user_id: str | None = None  # required if startup provided
    conversation_id: int | None = None
    stream_format: Literal["ndjson", "sse"] | None = None
    startup_command: str | None = None
    startup_action: Action | None = None

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str
    expires_in: int

class InvokeRequest(BaseModel):
    user_query: str
    timeout_seconds: int = 60

class PerformActionRequest(BaseModel):
    action: Action
    timeout_seconds: int = 60

class PostFeedbackRequest(BaseModel):
    channel_id: str
    binary_or_numeric_score: bool | float | None = None
    nl_feedback: str | None = None

class Action(BaseModel):
    command_name: str
    arguments: dict[str, Any] | None = None

class CommandResponse(BaseModel):
    response: str | None = None
    artifacts: dict[str, Any] | None = None
    next_actions: list[Action] | None = None
    recommendations: list[str] | None = None

class CommandOutput(BaseModel):
    success: bool | None = None
    workflow_name: str | None = None
    context: str | None = None
    command_name: str | None = None
    command_parameters: dict[str, Any] | None = None
    command_response: CommandResponse
    traces: list[dict[str, Any]] | None = None

class TurnOutput(BaseModel):
    turn_key: str
    status: TurnStatus              # completed | awaiting_user | failed | cancelled | abandoned
    failure_reason: str | None = None
    answer: str = ""
    command_outputs: list[CommandOutput] = []
    # success is a computed field: all(command_outputs succeeded)
```

Notes:
- `TurnOutput` is the response type of every turn surface (§6a). `CommandOutput` is no longer a top-level response shape — it appears only inside `TurnOutput.command_outputs`.
- Align `CommandOutput` and `CommandResponse` fields with FastWorkflow’s canonical definitions to avoid divergence. If Pydantic models exist in FastWorkflow, import them instead of redefining.
- `Action` mirrors the runtime execution object consumed by `CommandExecutor`.
- `user_id` is extracted on authenticated endpoints from JWT `uid` and included in traces alongside `raw_command`.

#### 8. Error Handling
- 404 Not Found: Missing `channel_id` / session.
- 409 Conflict: A different turn is already in progress for the same `channel_id` (registry active-execution pointer; retry with the same args rejoins instead).
- 422 Unprocessable Entity: Validation failures (invalid paths/action schema/channel input) and XOR violation in `/post_feedback`.
- 500 Internal Server Error: Unexpected errors (log with stack trace; avoid broad except without logging).
- 202 Accepted: Wait window elapsed; execution still running (wait-or-defer). Retry the same request to rejoin. **Not** a hard abort of the work.

A turn that reports `status: failed` or `success: false` is NOT an error status —
see §6a. These codes describe the transport only.

Error body format (example):
```json
{
  "detail": "Internal error in invoke_agent() for channel_id: channel-123"
}
```

#### 9. Concurrency & Timeouts
- Only one active turn **execution** per channel (registry pointer). Same-args retries rejoin; different work → 409.
- Per-channel `asyncio.Lock` serializes WEC mutation for one attempt; released on terminal status or `awaiting_user`.
- Default `timeout_seconds=60` is the **wait** window for wait-or-defer, not a kill switch for the execution.
- Live-session cache is bounded (`MAX_LIVE_SESSIONS`); busy channels (active execution or eviction lease) are not evicted out from under a turn.

#### 10. CORS & Security
- CORS: Allow configured origins; default to `*` for development only.
- Restrict `workflow_path` to an allow‑list of directories via config.

- Log each call with `channel_id`, action/command, and timing.
- Keep file logging to `action.jsonl` unchanged inside FastWorkflow when mirroring is enabled.

#### 10. Storage (Rdict) and Limits

- Conversations are stored under `SPEEDDICT_FOLDERNAME/user_conversations`; one Rdict DB file per channel (`<channel_id>.rdb`).
- Per-channel DB schema (keys/values):
  - Key: `meta` → { "last_conversation_id": int }
  - Key: `conv:<id>` → {
      "topic": str,
      "summary": str,
      "created_at": int,
      "updated_at": int,
      "turns": [ { "conversation summary": str, "conversation_traces": str (JSON), "feedback": { "binary_or_numeric_score": bool|float|null, "nl_feedback": str|null, "timestamp": int } | null } ]
    }
- Functional constraint: one active conversation per channel to avoid write concurrency.
- `/conversations` accepts `limit` (default `20`) controlling the max conversations returned (latest N by `updated_at`).
- Shutdown waits up to 30 seconds for active turns before persistence.
- Suspended Topology-B state: `SessionStateStore` under the session-state folder (injective channel-key encoding); not the conversation Rdict.
- `LLM_CONVERSATION_STORE`: LiteLLM model string for conversation topic/summary generation (e.g., `mistral/mistral-small-latest`).
- `LITELLM_API_KEY_CONVERSATION_STORE`: API key for the `LLM_CONVERSATION_STORE` model.

#### 11. Current implementation (Topology B)

Code of record: `fastworkflow/run_fastapi_mcp/` (package entry: `python -m fastworkflow.run_fastapi_mcp`). The older path names `services.run_fastapi.main` / `fastworkflow/run_fastapi/main.py` are obsolete.

What ships today (Step 1 of the turns design — see `docs/fastworkflow_turns_async_execution_design.md`):

1) **`ChannelSessionManager` + `ChannelRuntime`** (`utils.py`)
   - Live map `{ channel_id → ChannelRuntime }` with eviction leases, creation single-flight, busy-channel skip, optional pending-state reaper.
   - `ChannelRuntime.execution_context` is a `WorkflowExecutionContext` (agent mode). Property `chat_session` is a **backward-compatible alias** for that WEC — not a Topology-A `ChatSession`.

2) **`POST /initialize`**
   - Env from files → `fastworkflow.init`.
   - `ensure_user_runtime_exists` builds/binds WEC, restores conversation + durable pending/checkpoint when present.
   - Optional startup via `submit_turn(kind="initialize_startup")`; three-state already-exists / running / done responses (`startup_turn_key`, `startup_exec_state`, `startup_output` / `startup_error`).

3) **Non-streaming turn endpoints** (`/invoke_agent`, `/invoke_assistant`, `/perform_action`)
   - Thin wrappers: lease session → `submit_turn` → `render_turn_response` (200 done / 202 deferred / 409 busy).
   - Work calls `process_turn` or `process_action_turn` on the WEC (not queue put/get).
   - Top-level `turn_key` in the body is the **execution** key (§6a).

4) **`POST /invoke_agent_stream`**
   - Registers an owned turn (`run_owned_turn`) so disconnect stops *reading*, not *execution*.
   - Emits NDJSON or SSE `trace` records, then terminal `output` with bare `TurnOutput` (**logical** `turn_key`).
   - Never emits Topology-A `event: command_output`.

5) **Conversation admin surfaces**
   - `/new_conversation`, `/conversations`, `/activate_conversation`, `/post_feedback`, `/admin/dump_all_conversations`, `/abandon_clarification` — pointer/lock guarded where they touch the live WEC.

6) **Persistence**
   - Incremental conversation save before `exec_state=DONE`; pending suspend blobs via `SessionStateStore`; startup outcome committed durably before it becomes observable.

**Explicitly not implemented yet (fix-85g Step 2 — do not treat as current):**
- `GET /turns/{turn_key}` and client poll-by-GET
- Non-destructive per-execution trace replay buffer for reconnect
- Sized executor + global admission 429 + TTL eviction of terminal registry entries
- Step 3 distributed/durable turn store

Until Step 2 ships, deferred clients **retry the same HTTP request** to rejoin.

Type hints follow FastWorkflow's `TurnOutput` / `CommandOutput` models; import them rather than forking shapes.

#### 12. Testing Strategy
- Unit / focused integration
  - SessionManager: concurrency, leases, lifecycle, busy-channel eviction skip.
  - Env loading from files only.
  - Validation: reject both `startup_command` and `startup_action` together.
  - Wait-or-defer: 202 when work exceeds wait window; same-args retry rejoins one execution (`tests/test_fastapi_turns_async.py`).
  - Stream formatting: `trace` then terminal `output` (TurnOutput), not `command_output`.
- Integration
  - Spin up FastAPI app via `TestClient`.
  - Initialize with a sample workflow (fixture) and perform one agent turn; assert the `TurnOutput` projection fields (§6a). Covered by `tests/test_fastapi_turn_output_contract.py`.
  - Perform action path using a known command; verify response.
  - New conversation persists old history and resets runtime; validate prior conversation is stored and later appears in `/conversations`.
  - Test `/invoke_agent_stream` by parsing SSE/NDJSON: trace events before final `output`; validate TurnOutput JSON.
  - Test streaming with traces disabled: only the `output` event.
  - Test stream transport failure: terminal `error` record; `awaiting_user` / `failed` still arrive as `output`.

#### 13. Deployment Notes
- Run: `python -m fastworkflow.run_fastapi_mcp --workflow_path <dir> --port 8000` (see README). Do not use the deleted `services.run_fastapi.main` module path.
- Consider setting lifespan hooks only when using the server's startup/shutdown drain (active turns + pending reaper).

#### 14. Future Enhancements
- WebSocket support as an alternative to SSE for bidirectional communication.
- fix-85g Step 2: `GET /turns/{turn_key}`, trace replay, sized executor / 429 backpressure, TTL on terminal executions (not shipped — see §4 / §11).
- Richer observability: correlate CLI trace colors to structured HTTP traces.
- Security hardening: workflow/path allow‑list, authn/z.

#### 15. JWT Token-Based Auth

##### 15.1 Overview
Replace integer-based `session_id` with JWT (JSON Web Token) tokens using asymmetric cryptography (RS256) to enable:
- **Token expiration** with configurable TTL
- **Stateless verification** (no server-side session lookup needed)
- **Enhanced security** via public key cryptography (tokens cannot be forged)
- **Token refresh** mechanism for seamless session continuation

##### 15.2 Goals
1. Replace integer `session_id` with signed JWT tokens
2. Support configurable token expiration (default: 1 hour)
3. Use RS256 (RSA + SHA-256) for asymmetric signing
4. Add token refresh endpoint for active sessions
5. Maintain backward compatibility with existing `ConversationStore` (still uses integer session_id internally)

##### 15.3 Dependencies
Add to `pyproject.toml`:
```toml
python-jose = {extras = ["cryptography"], version = "^3.3.0"}
```

##### 15.4 JWT Token Structure

**Claims:**
```json
{
  "sub": "john_doe",           # subject (channel_id)
  "iat": 1234567890,           # issued at (Unix timestamp)
  "exp": 1234571490,           # expires at (iat + TTL)
  "jti": "uuid-v4-string",     # unique token ID (prevents replay attacks)
  "type": "access",            # token type ("access" or "refresh")
  "iss": "fastworkflow-api",   # issuer
  "aud": "fastworkflow-client" # audience
}
```

**Token Format:**
```
eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJqb2huX2RvZSIsInNlc3Npb25faWQiOjEyMzQ1Njc4OTAsImlhdCI6MTIzNDU2Nzg5MCwiZXhwIjoxMjM0NTcxNDkwLCJqdGkiOiJ1dWlkLXY0LXN0cmluZyIsInR5cGUiOiJhY2Nlc3MiLCJpc3MiOiJmYXN0d29ya2Zsb3ctYXBpIiwiYXVkIjoiZmFzdHdvcmtmbG93LWNsaWVudCJ9.signature
```

##### 15.5 Configuration

**Environment Variables / CLI Args:**
```bash
JWT_TOKEN_EXPIRE_MINUTES=60          # Access token TTL (default: 1 hour)
JWT_REFRESH_TOKEN_EXPIRE_MINUTES=10080  # Refresh token TTL (default: 7 days)
JWT_ALGORITHM=RS256                  # Signing algorithm
JWT_ISSUER=fastworkflow-api          # Token issuer
JWT_AUDIENCE=fastworkflow-client     # Token audience
JWT_PRIVATE_KEY_PATH=.jwt_keys/private_key.pem  # Private key location
JWT_PUBLIC_KEY_PATH=.jwt_keys/public_key.pem    # Public key location
```

**Configuration Structure:**
```python
JWT_CONFIG = {
    "algorithm": "RS256",
    "access_token_expire_minutes": 60,
    "refresh_token_expire_minutes": 10080,  # 7 days
    "issuer": "fastworkflow-api",
    "audience": "fastworkflow-client",
    "private_key_path": ".jwt_keys/private_key.pem",
    "public_key_path": ".jwt_keys/public_key.pem",
}
```

##### 15.6 Key Management

**Key Generation:**
- Generate 2048-bit RSA key pair on first startup (or load existing)
- Store keys in: `./jwt_keys/` (relative to project root)
  - `private_key.pem` - Server only, never share
  - `public_key.pem` - Can be shared for external verification

**Key Storage:**
```python
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

# Generate key pair
private_key = rsa.generate_private_key(
    public_exponent=65537,
    key_size=2048
)
public_key = private_key.public_key()

# Save as PEM files with proper permissions (600 for private key)
```

**Key Rotation (Future Enhancement):**
- Generate new key pair periodically
- Keep old public key for verification during grace period
- Sign new tokens with new private key

##### 15.7 New Module: `jwt_manager.py`

Create `services/run_fastapi/jwt_manager.py`:

**Functions:**
```python
def load_or_generate_keys() -> tuple[RSAPrivateKey, RSAPublicKey]:
    """Load existing keys or generate new pair on first run"""

def create_access_token(channel_id: str, expires_delta: timedelta) -> str:
    """Create and sign JWT access token with RS256"""

def create_refresh_token(channel_id: str, expires_delta: timedelta) -> str:
    """Create and sign JWT refresh token with RS256"""

def verify_and_decode_token(token: str, token_type: str = "access") -> dict:
    """Verify signature, check expiration, and decode token claims"""
```

##### 15.8 Data Model Changes

**New Models in `utils.py`:**
```python
class SessionData(BaseModel):
    """Decoded JWT session data"""
    channel_id: str
    issued_at: int
    expires_at: int
    token_id: str
    token_type: str

class TokenResponse(BaseModel):
    """Token response from /initialize"""
    access_token: str
    refresh_token: str
    token_type: str = "Bearer"
    expires_in: int  # seconds until access_token expires
    workflow_info: dict[str, Any] | None = None
```

**Updated Request Models:**
Remove `session_id` field from all request models (extracted from JWT header):
- `InvokeRequest`
- `PerformActionRequest`
- `PostFeedbackRequest`
- `ActivateConversationRequest`

Add docstrings noting: "Requires JWT access token in Authorization header"

##### 15.9 API Changes

**Before (Integer Session ID):**
```bash
# Initialize
POST /initialize
Request: {"channel_id": "john_doe"}
Response: {"session_id": 1234567890, "workflow_info": {...}}

# Use endpoint
POST /invoke_agent
Headers: Authorization: 1234567890
Body: {"user_query": "..."}
```

**After (JWT Tokens):**
```bash
# Initialize
POST /initialize
Request: {"channel_id": "john_doe"}
Response: {
  "access_token": "eyJhbGci...",
  "refresh_token": "eyJhbGci...",
  "token_type": "Bearer",
  "expires_in": 3600
}
# Notes:
# - channel_id is in the JWT's "sub" claim
# - Workflow definition available via what_can_i_do command

# Use endpoint
POST /invoke_agent
Headers: Authorization: Bearer eyJhbGci...
Body: {"user_query": "..."}
```

##### 15.10 Endpoint Changes

**1. `POST /initialize` (Modified)**
- Generate JWT tokens instead of returning integer session_id
- Return `TokenResponse` with both access and refresh tokens
- The `channel_id` is embedded in the JWT (sub claim), not returned separately

**2. All authenticated endpoints (Modified)**
- Change dependency from: `session_id: int = Depends(get_session_id_from_header)`
- To: `session: SessionData = Depends(get_session_from_jwt)`
- Use `session.channel_id` for session lookups and logging

**3. `POST /refresh_token` (New)**
```python
@app.post("/refresh_token")
async def refresh_token(
    session: SessionData = Depends(get_session_from_jwt)
) -> TokenResponse:
    """
    Generate new access token for existing session.
    Requires valid refresh token in Authorization header.
    """
```

**Request:**
```json
# Headers: Authorization: Bearer <refresh_token>
{}
```

**Response:**
```json
{
  "access_token": "eyJhbGci...",
  "refresh_token": "eyJhbGci...",
  "token_type": "Bearer",
  "expires_in": 3600
}
```

**Behavior:**
- Accept valid refresh token (not expired)
- Generate new access token with fresh TTL
- Generate new refresh token (token rotation)
- Return both new tokens

##### 15.11 New Dependency Function in `utils.py`

**Replace:**
```python
def get_session_id_from_header(authorization: str = Header(...)) -> int:
```

**With:**
```python
def get_session_from_jwt(
    authorization: str = Header(
        ...,
        description="JWT Bearer token (format: 'Bearer <token>')"
    )
) -> SessionData:
    """
    FastAPI dependency to extract and verify JWT from Authorization header.
    
    Args:
        authorization: JWT token with Bearer scheme
        
    Returns:
        SessionData: Decoded and validated session information
        
    Raises:
        HTTPException 400: Missing or malformed Authorization header
        HTTPException 401: Invalid token, expired, or verification failed
        
    Example:
        Authorization: Bearer eyJhbGci...
    """
    # Extract token from "Bearer <token>" format
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=400,
            detail="Invalid Authorization header format (expected: Bearer <token>)"
        )
    
    token = authorization.replace("Bearer ", "")
    
    try:
        claims = verify_and_decode_token(token, token_type="access")
        return SessionData(
            channel_id=claims["sub"],
            session_id=claims["session_id"],
            issued_at=claims["iat"],
            expires_at=claims["exp"],
            token_id=claims["jti"],
            token_type=claims["type"]
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=401,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"}
        )
    except jwt.InvalidTokenError as e:
        raise HTTPException(
            status_code=401,
            detail=f"Invalid token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"}
        )
```

##### 15.12 Error Handling

**New HTTP Status Codes:**
```
400 Bad Request:
  - Missing Authorization header
  - Malformed Authorization header (not "Bearer <token>")
  - Invalid token format

401 Unauthorized:
  - Token expired
  - Invalid signature
  - Token claims invalid (wrong issuer/audience)
  - Token type mismatch (access vs refresh)
  
403 Forbidden:
  - Token revoked (future enhancement)
```

**Error Response Format:**
```json
{
  "detail": "Token has expired",
  "error_code": "TOKEN_EXPIRED",
  "expires_at": 1234567890
}
```

##### 15.13 Security Considerations

**✅ Implemented:**
- Asymmetric crypto (RS256) prevents token forgery
- Token expiration limits lifetime exposure
- JTI (unique token ID) prevents replay attacks
- Stateless verification (no DB lookup per request)
- Separate access and refresh tokens
- Token rotation on refresh

**🔒 Best Practices:**
- Always use HTTPS in production
- Store private key securely (not in git)
- Add `.jwt_keys/private_key.pem` to `.gitignore`
- Use environment variables for production key paths
- Set proper file permissions (600) on private key
- Monitor for abnormal token usage patterns

**⚠️ Future Enhancements:**
- Token revocation list (Redis-based) for logout functionality
- Rate limiting on token refresh endpoint
- Key rotation with grace period
- Audit logging for token issuance and verification failures

##### 15.14 File Structure

```
services/run_fastapi/
├── jwt_manager.py          # NEW: JWT creation, verification, key management
├── utils.py                # MODIFIED: JWT models, get_session_from_jwt dependency
├── main.py                 # MODIFIED: Endpoints use JWT
└── mcp_specific.py         # NO CHANGE (still uses channel_id directly)

.jwt_keys/                  # NEW: Key storage (project root)
├── private_key.pem         # Server only, mode 600
├── public_key.pem          # Can be shared
└── .gitignore             # Ignore private key
```

##### 15.15 Implementation Steps

**Phase 1: JWT Infrastructure**
1. Add `python-jose[cryptography]` to dependencies
2. Create `jwt_manager.py` with:
   - Key generation and loading functions
   - Token creation functions (access and refresh)
   - Token verification and decoding function
3. Add key management CLI support (generate, rotate)

**Phase 2: Data Models**
4. Add `SessionData` and `TokenResponse` models to `utils.py`
5. Update request models to remove `session_id` field
6. Add docstrings noting JWT requirement

**Phase 3: Endpoints**
7. Update `get_session_id_from_header` → `get_session_from_jwt` in `utils.py`
8. Modify `/initialize` to return `TokenResponse` with JWT tokens
9. Update all authenticated endpoints to use `get_session_from_jwt` dependency
10. Add `/refresh_token` endpoint

**Phase 4: Error Handling**
11. Add 401 error handlers for token expiration
12. Add 400 error handlers for malformed tokens
13. Update OpenAPI documentation with Bearer security scheme

**Phase 5: Testing**
14. Test token generation and verification
15. Test token expiration handling
16. Test token refresh flow
17. Test invalid token rejection
18. Update Swagger UI with Bearer token authentication
19. Load test: Verify performance impact of token verification

**Phase 6: Security Hardening**
20. Add HTTPS enforcement middleware
21. Implement proper CORS configuration
22. Add rate limiting on `/refresh_token`
23. Add audit logging for token operations

##### 15.16 Backward Compatibility

**Breaking Change Approach (Recommended):**
- Clean break from integer session_id to JWT
- Update all clients to use new authentication flow
- Better long-term maintainability

**Migration Steps:**
1. Deploy new version with JWT support
2. Update client applications to:
   - Call `/initialize` to get JWT tokens
   - Use `Authorization: Bearer <token>` header
   - Implement token refresh logic
3. Remove old integer-based code after migration

##### 15.17 Testing Strategy

**Unit Tests:**
- Key generation and loading
- Token creation with correct claims
- Token verification (valid, expired, invalid signature)
- SessionData extraction from valid tokens
- Error handling for malformed tokens

**Integration Tests:**
- Full flow: initialize → invoke with JWT → refresh → invoke again
- Token expiration: verify 401 after TTL
- Invalid token: verify 401 with various invalid tokens
- Concurrent requests with same token
- Token refresh with expired access token but valid refresh token

**Performance Tests:**
- Token verification latency (should be < 1ms)
- Compare to previous integer-based auth
- Load test with many concurrent authenticated requests

##### 15.18 Swagger UI Integration

**OpenAPI Security Scheme:**
```python
app = FastAPI(
    title="FastWorkflow API",
    # ... other params
)

# Add security scheme to OpenAPI spec
app.openapi_components = {
    "securitySchemes": {
        "bearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "JWT token obtained from /initialize endpoint"
        }
    }
}

# Apply to endpoints
@app.post("/invoke_agent", 
    security=[{"bearerAuth": []}]
)
```

**Swagger UI Usage:**
1. Click "Authorize" button in Swagger UI
2. Enter JWT token (without "Bearer " prefix)
3. Swagger automatically adds "Bearer " prefix to requests
4. Token is remembered for session duration

##### 15.19 Configuration Example

**Production `.env`:**
```bash
# JWT Configuration
JWT_TOKEN_EXPIRE_MINUTES=60
JWT_REFRESH_TOKEN_EXPIRE_MINUTES=10080
JWT_ALGORITHM=RS256
JWT_ISSUER=fastworkflow-api
JWT_AUDIENCE=fastworkflow-client
JWT_PRIVATE_KEY_PATH=/secure/path/private_key.pem
JWT_PUBLIC_KEY_PATH=/secure/path/public_key.pem

# Force HTTPS
FORCE_HTTPS=true

# CORS
ALLOWED_ORIGINS=https://app.example.com,https://admin.example.com
```

##### 15.20 MCP Client Token Setup

**Generating Long-Lived Tokens for MCP:**

MCP clients (e.g., Claude Desktop) use pre-configured access tokens instead of dynamically obtaining them via `/initialize`. To set up an MCP client:

1. **Generate MCP Token (Admin):**
```bash
POST /admin/generate_mcp_token
{
  "channel_id": "claude_desktop_user",
  "expires_days": 365
}

Response:
{
  "access_token": "eyJhbGci...",  # Long-lived token (1 year)
  "refresh_token": "",
  "token_type": "bearer",
  "expires_in": 31536000  # 365 days in seconds
}
```

2. **Configure MCP Client:**

Add to Claude Desktop's `mcp.json`:
```json
{
  "mcpServers": {
    "fastworkflow": {
      "url": "http://localhost:8000/mcp",
      "headers": {
        "Authorization": "Bearer eyJhbGci..."
      }
    }
  }
}
```

3. **MCP Tool Usage:**
- MCP client calls tools like `invoke_agent`, `invoke_assistant`, etc.
- Authorization header is automatically included by the MCP client
- No need to call `initialize` (excluded from MCP tools)
- Token is long-lived (default 1 year) so no refresh needed

**Security Notes:**
- Store the generated token securely in MCP client config
- Tokens are tied to a specific `channel_id`
- All MCP tool calls are authenticated and tracked per channel
- Token expiration can be customized (e.g., 30 days, 180 days, etc.)

##### 15.21 Future Enhancements (Not in Initial Implementation)

**Token Revocation:**
- Maintain Redis-based revocation list
- Store token JTI when channel logs out
- Check revocation list in `verify_and_decode_token()`
- Implement `/logout` endpoint that adds token to revocation list

**Advanced Key Management:**
- Automatic key rotation every 90 days
- Multiple public keys for verification (during rotation)
- Key versioning in JWT header (`kid` claim)

**Additional Features:**
- OAuth2 scopes for fine-grained permissions
- Multi-factor authentication support
- Token introspection endpoint
- JWKS (JSON Web Key Set) endpoint for public key distribution



### FastWorkflow FastAPI Service — Specification

#### 1. Overview
- **Goal**: Expose FastWorkflow workflows as a FastAPI web service, enabling clients to initialize for a given workflow (per user), then interact in agent mode (forced). If a query starts with `/`, all leading slashes are stripped before processing. The service supports explicit actions, resetting conversations, listing conversations, dumping all conversations to JSONL, and posting feedback. `invoke_agent` returns a synchronous `CommandOutput` and, when enabled, includes collected trace events in the final response. `invoke_agent_stream` provides real-time streaming of trace events and the final `CommandOutput` via SSE (Server-Sent Events).
- **Source parity**: Behavior mirrors the CLI runner in `fastworkflow/run/__main__.py` while replacing its interactive loop with synchronous and streaming HTTP endpoints.

#### 2. Non‑Goals
- No UI; only REST with OpenAPI/Swagger at `/docs`.
- No WebSocket support in MVP; SSE (Server‑Sent Events) is used for streaming.

#### 3. References
- CLI runner setup (initialization and single ChatSession orchestration):
  - Initializes env vars, validates workflow folder, constructs `ChatSession`, calls `start_workflow(...)`, and prints outputs.
  - Parity we need to preserve: environment loading, startup command/action handling, deterministic vs agentic execution, command output shape.

#### 4. Architecture Summary
- FastAPI app with a process‑wide in‑memory `UserSessionManager` that manages per‑user runtime state keyed by `user_id`.
- For each `user_id`, maintain:
  - An active `ChatSession` (always agent mode) bound to the current conversation (internal id only).
  - A persistent `ConversationStore` backed by `Rdict` (one DB file per user) storing: `conversation_id` (internal), `topic` (unique per user), `summary`, timestamps, per‑turn history, and optional feedback per turn.
- For requests, enqueue a user message and synchronously wait for a `CommandOutput` on the `command_output_queue` (with timeout and single‑turn serialization per user).
- If `show_agent_traces=true` (initialization flag), trace events are collected and included in the final response payload.

#### 5. User and Conversation Lifecycle
1) Client calls `POST /initialize` with workflow location and setup options plus a `user_id`.
2) Server:
   - Loads environment from `env_file_path` and `passwords_file_path` only.
   - Calls `fastworkflow.init(env_vars=<file_based_env_dict>)`.
   - Creates a new `ChatSession` with `run_as=RUN_AS_AGENT` (forced agent mode), bound to `user_id` and a current conversation (internal id only).
   - If the user has a prior conversation, resume it by default (restore last conversation history); otherwise create a new conversation.
   - Starts the workflow via `chat_session.start_workflow(...)` with provided context/startup parameters.
   - Stores runtime in `UserSessionManager` and returns `{user_id}`.
3) Client uses `user_id` with `/invoke_agent`, `/invoke_assistant`, `/perform_action`, or `/new_conversation`. Additional endpoints: `/conversations`. Admin-only endpoint: `/admin/dump_all_conversations`.
4) On `new_conversation` or process shutdown:
   - Generate `topic` (guaranteed unique per user via case-insensitive and whitespace-insensitive comparison; append an incrementing integer if needed) and `summary` synchronously via `dspy.ChainOfThought()`.
   - If generation succeeds, persist the conversation to Rdict and rotate to a new internal conversation; if it fails, log a critical error, do NOT persist, and do NOT rotate.
5) Conversation histories persist across restarts; users resume from their last conversation by default.

#### 6. Endpoints

1) POST `/initialize`
- Purpose: Create or resume a FastWorkflow `ChatSession` for a `user_id` and start the workflow.
- Request (InitializationRequest):
```json
{
  "user_id": "user-123",
  "workflow_path": "/abs/path/to/workflow",            
  "env_file_path": "/abs/path/to/.env",                
  "passwords_file_path": "/abs/path/to/passwords.env", 
  "context": {"...": "..."},                         
  "startup_command": "",                               
  "startup_action": {"command_name":"...","parameters":{}} ,
  "show_agent_traces": true,
  "conversation_id": null
}
```
- Notes:
  - Load env only from files: `env_file_path` and `passwords_file_path`.
  - Service always runs in agent mode; assistant behavior is exposed only via `/invoke_assistant`.
  - `startup_action` matches FastWorkflow `Action` (see models below). Optional. Must include `command_name` if provided; don't send empty object `{}`.
  - `startup_command` and `startup_action` are mutually exclusive.
  - `conversation_id`: Optional. If `null` or omitted, restores last conversation (or starts new if none exist). Provide specific ID to restore that conversation.
  - `context`: workflow context dict (instead of `--context_file_path` in CLI).
- Response (InitializationResponse):
```json
{ "user_id": "user-123" }
```
- Errors:
  - 422 invalid paths or missing env vars
  - 400 both `startup_command` and `startup_action` provided
  - 500 initialization failure (details logged)

- 2) POST `/invoke_agent`
- Purpose: Submit a natural language query to an agentic session for a user (synchronous response). Leading `/` characters are permitted and stripped for compatibility; assistant semantics remain exclusive to `/invoke_assistant`.
- Request:
```json
{ "user_id": "user-123", "user_query": "find orders for user 42", "timeout_seconds": 60 }
```
- Behavior:
  - Validate the user session exists. Agent mode is always enabled.
  - If `user_query` begins with `/`, strip all leading slashes before processing (compatibility path).
  - When the turn completes, return a `CommandOutput` JSON; if `show_agent_traces=true`, include collected `traces`.
- Response: `CommandOutput` (JSON).
- Errors:
  - 404 user not found
  - 409 concurrent turn already in progress for this user
  - 504 turn timed out (no output on queue within `timeout_seconds`)
  - 500 unexpected error

3) POST `/invoke_agent_stream`
- Purpose: Submit a natural language query to an agentic session and stream trace events in real-time via SSE (Server-Sent Events), followed by the final `CommandOutput`. Leading `/` characters are permitted and stripped.
- Request:
```json
{ "user_id": "user-123", "user_query": "find orders for user 42", "timeout_seconds": 60 }
```
- Behavior:
  - Validate the user session exists. Agent mode is always enabled.
  - If `user_query` begins with `/`, strip all leading slashes before processing.
  - Stream trace events as they are emitted by the workflow via SSE format (text/event-stream).
  - When the turn completes, emit the final `CommandOutput` as a `data:` event with `event: command_output`.
  - If no traces are enabled (`show_agent_traces=false`), only the final `CommandOutput` is streamed.
- Response: SSE stream with events:
  - `event: trace` with `data: <trace_json>` for each trace event
  - `event: command_output` with `data: <CommandOutput_json>` for the final result
  - `event: error` with `data: <error_json>` if an error occurs
- SSE Message Format:
```
event: trace
data: {"timestamp": "...", "event_type": "...", "details": {...}}

event: trace
data: {"timestamp": "...", "event_type": "...", "details": {...}}

event: command_output
data: {"success": true, "workflow_name": "...", "command_responses": [...]}

```
- Errors:
  - 404 user not found
  - 409 concurrent turn already in progress for this user
  - 504 turn timed out (no output within `timeout_seconds`)
  - Stream will emit `event: error` followed by connection close on unexpected errors

4) POST `/invoke_assistant`
- Purpose: Deterministic/assistant invocation for a user. The server accepts plain queries; clients need not prefix `/`.
- Request:
```json
{ "user_id": "user-123", "user_query": "load_workflow file='...'" }
```
- Behavior: Same execution path as agent, but uses assistant path (no planning). Returns `CommandOutput`.
- Response: `CommandOutput`.
- Errors: as above.

5) POST `/perform_action`
- Purpose: Execute a specific workflow action chosen by the client (e.g., from `next_actions`).
- Request:
```json
{ "user_id": "user-123", "action": { "command_name": "User/get_details", "arguments": { "user_id": "u-42" } }, "timeout_seconds": 60 }
```
- Behavior:
  - Validate session exists.
  - Invoke through the same single‑turn path used for NL queries, but bypass parameter extraction (directly execute the provided `Action`).
  - Wait for `CommandOutput` (or timeout) and return it.
- Response: `CommandOutput`.
- Errors: 404/409/504/500 as above; 422 invalid action shape.

6) POST `/new_conversation`
- Purpose: Persist and close the current conversation (topic + summary via GenAI), then reset history and start a new internal conversation.
- Request:
```json
{ "user_id": "user-123" }
```
- Behavior:
  - Generate `topic` (unique per user; append integer suffix if needed) and `summary` synchronously using `dspy.ChainOfThought()`.
  - If generation succeeds, persist conversation `{topic, summary, history}` in Rdict and rotate; if it fails, log critical, return 500, and do not rotate.
- Response: `{ "status": "ok" }`.
- Errors: 404 if user missing.

7) POST `/post_feedback`
- Purpose: Attach optional feedback to the latest turn in the current conversation for a user.
- Request:
```json
{
  "user_id": "user-123",
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
- Errors: 404 user missing; 422 invalid input (both fields null).

8) GET `/` (root)
- Simple HTML page with a link to `/docs`. Serves also as a health check (no dedicated `/healthz`).

#### 7. Data Models

Pydantic model sketches (for reference; actual code will import FastWorkflow types where available):

```python
class InitializationRequest(BaseModel):
    user_id: str
    workflow_path: str
    env_file_path: str | None = None
    passwords_file_path: str | None = None
    context: dict[str, Any] | None = None
    startup_command: str | None = None
    startup_action: Action | dict | None = None
    show_agent_traces: bool = True

class InitializationResponse(BaseModel):
    user_id: str

class InvokeRequest(BaseModel):
    user_id: str
    user_query: str
    timeout_seconds: int = 60

class PerformActionRequest(BaseModel):
    user_id: str
    action: Action
    timeout_seconds: int = 60

class NewConversationRequest(BaseModel):
    user_id: str

class PostFeedbackRequest(BaseModel):
    user_id: str
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
    command_responses: list[CommandResponse]
    traces: list[dict[str, Any]] | None = None  # included when show_agent_traces=true
```

Notes:
- Align `CommandOutput` and `CommandResponse` fields with FastWorkflow’s canonical definitions to avoid divergence. If Pydantic models exist in FastWorkflow, import them instead of redefining.
- `Action` mirrors the runtime execution object consumed by `CommandExecutor`.

#### 8. Error Handling
- 404 Not Found: Missing `user_id`.
- 409 Conflict: A turn is already in progress for the same `user_id` (serialize turns per user).
- 422 Unprocessable Entity: Validation failures (invalid paths/action schema/user input) and XOR violation in `/post_feedback`.
- 500 Internal Server Error: Unexpected errors (log with stack trace; avoid broad except without logging).
- 504 Gateway Timeout: No `CommandOutput` received before `timeout_seconds`.

Error body format (example):
```json
{
  "detail": "Internal error in invoke_agent() for user_id: user-123"
}
```

#### 9. Concurrency & Timeouts
- Only one in‑flight turn per user. Use a per‑user asyncio Lock or queue state flag.
- Default `timeout_seconds=60` per request; configurable per call.
- Consider a global `MAX_CONCURRENT_SESSIONS` guard if needed.

#### 10. CORS & Security
- CORS: Allow configured origins; default to `*` for development only.
- Restrict `workflow_path` to an allow‑list of directories via config.

- Log each call with `user_id`, action/command, and timing.
- Keep file logging to `action.jsonl` unchanged inside FastWorkflow.

#### 10. Storage (Rdict) and Limits

- `USER_CONVERSATIONS_SPEEDDICT_FOLDERNAME`: base folder from env file; one Rdict DB file per user (`<user_id>.rdb`).
- Per-user DB schema (keys/values):
  - Key: `meta` → { "last_conversation_id": int }
  - Key: `conv:<id>` → {
      "topic": str,  // unique per user
      "summary": str,
      "created_at": int,  // epoch ms
      "updated_at": int,
      "turns": [ { "conversation summary": str, "conversation_traces": str (JSON), "feedback": { "binary_or_numeric_score": bool|float|null, "nl_feedback": str|null, "timestamp": int } | null } ]
    }
- Functional constraint: one active conversation per user to avoid write concurrency.
- `FASTWORKFLOW_CONVERSATIONS_LIST_LIMIT`: environment variable controlling the max conversations returned by `/conversations` (latest N by `updated_at`).
- `FASTWORKFLOW_SHUTDOWN_MAX_WAIT_SECONDS`: max wait to finish active turns before shutdown persistence.
- `LLM_CONVERSATION_STORE`: LiteLLM model string for conversation topic/summary generation (e.g., `mistral/mistral-small-latest`).
- `LITELLM_API_KEY_CONVERSATION_STORE`: API key for the `LLM_CONVERSATION_STORE` model.

#### 11. Implementation Plan

Minimal edits in `fastworkflow/run_fastapi/main.py`:
1) Replace legacy imports and undefined helpers with FastWorkflow runtime imports.
2) Implement a `UserSessionManager` holding `{ user_id: { active_conversation_id, chat_session, lock } }`.
3) `POST /initialize`:
   - Validate `workflow_path` exists and contains `_commands/` (optional warning).
   - Load env from files only; call `fastworkflow.init(env_vars=env)`.
   - Create `chat_session = fastworkflow.ChatSession(run_as_agent=True)` (keep_alive=True internally).
   - `chat_session.start_workflow(workflow_path, workflow_context=context, startup_command=..., startup_action=...)`.
   - If `conversation_id` provided and exists, restore its history; else restore last; else start new.
   - Store and return `{user_id}`.
4) `POST /invoke_agent`:
   - Serialize turns per user: acquire user lock.
   - Put `user_query` on `user_message_queue`; wait for `command_output_queue.get(timeout=timeout_seconds)`.
   - If `show_agent_traces=True`, drain `command_trace_queue` and include events in `traces` array of the final response.
5) `POST /invoke_agent_stream`:
   - Serialize turns per user: acquire user lock.
   - Put `user_query` on `user_message_queue`.
   - Set response content-type to `text/event-stream`.
   - If `show_agent_traces=True`, continuously drain `command_trace_queue` and emit each trace as an SSE event (`event: trace`).
   - Wait for `command_output_queue.get(timeout=timeout_seconds)`.
   - Emit the final `CommandOutput` as an SSE event (`event: command_output`).
   - Handle errors by emitting `event: error` with error details before closing the stream.
6) `POST /perform_action`:
   - Similar to above, but bypass parameter extraction by directly invoking the action path (either put a special message on the queue or a helper on `CommandExecutor`).
7) `POST /new_conversation`:
   - Persist current history to `Rdict`; schedule background job to generate topic/summary via `dspy.ChainOfThought()` and update the record.
   - Rotate internal conversation id and clear history.
8) `GET /conversations`:
   - Read from `Rdict` and return list of at most `FASTWORKFLOW_CONVERSATIONS_LIST_LIMIT` items: `{conversation_id, topic, summary}` ordered by `updated_at` desc.
9) `POST /admin/dump_all_conversations`:
   - Iterate all users and conversations in `Rdict` and write JSONL file at provided folder, return file path.
10) `POST /post_feedback`:
   - Validate presence (at least one field); attach to latest turn of active conversation in `Rdict`.
11) `POST /activate_conversation`:
   - Body: { user_id, conversation_id }. Find conversation by ID for user and set as active; if not found 404.
12) Root endpoint doubles as health check; remove `/healthz`.

Type hints & structure should follow existing FastWorkflow dataclasses/Pydantic models for compatibility.

#### 12. Testing Strategy
- Unit
  - SessionManager: concurrency and lifecycle.
  - Env loading from files only.
  - Validation: reject both `startup_command` and `startup_action` together.
  - Timeout behavior: ensure 504 when queue has no output.
  - SSE formatting: verify correct event structure for trace and command_output events.
- Integration
  - Spin up FastAPI app via `TestClient`.
  - Initialize with a sample workflow (fixture) and perform one agent turn; assert `CommandOutput` fields.
  - Perform action path using a known command; verify response.
  - New conversation persists old history and resets runtime; validate prior conversation is stored and later appears in `/conversations`.
  - Test `/invoke_agent_stream` by parsing SSE events: verify trace events arrive before final command_output event; validate JSON structure of each event.
  - Test streaming with `show_agent_traces=false`: verify only command_output event is emitted.
  - Test streaming error handling: verify `event: error` emission when turn fails.

#### 13. Deployment Notes
- Run via Uvicorn: `uvicorn services.run_fastapi.main:app --host 0.0.0.0 --port 8000`.
- Consider setting `lifespan="on"` only if using startup/shutdown hooks.

#### 14. Future Enhancements
- WebSocket support as an alternative to SSE for bidirectional communication.
- Session TTL and eviction policy; persistence layer for sessions if required.
- Richer observability: correlate CLI trace colors to structured HTTP traces.
- Security hardening: workflow/path allow‑list, authn/z.



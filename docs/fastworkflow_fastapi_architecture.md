### FastWorkflow FastAPI Service — Architecture & Design

Based on: [fastworkflow_fastapi_spec.md](mdc:docs/fastworkflow_fastapi_spec.md)

#### 1. Scope and Goals
- **Purpose**: Provide a minimal, modular FastAPI service that exposes FastWorkflow workflows over HTTP with behavior matching the CLI runner, replacing its interactive loop with synchronous endpoints.
- **Strict adherence**: This document mirrors the referenced spec and does not introduce functionality beyond what is explicitly stated.

#### 2. Non‑Goals
- **UI**: No user interface. Only REST endpoints with OpenAPI/Swagger at `/docs`.
- **WebSockets**: No WebSocket support; SSE (Server‑Sent Events) is used for streaming.

#### 3. High‑Level Architecture
- **Process‑wide app**: A single FastAPI app hosts an in‑memory `UserSessionManager` keyed by `user_id`.
- **Per‑user runtime**:
  - Active `ChatSession` (always agent mode) bound to a current conversation (internal id only).
  - Persistent `ConversationStore` per user backed by Rdict (one DB file per user).
- **Synchronous turn execution**: For each request, enqueue user input and synchronously await a `CommandOutput` from the `command_output_queue` with timeout and single‑turn serialization.
- **Optional traces**: If `show_agent_traces=true` (set during initialization), trace events are collected and included in the final response payload.

#### 4. Minimal Module Decomposition
- **API Layer (FastAPI routes)**
  - Endpoint handlers for all specified routes (Section 6 below).
  - Request validation via Pydantic models mirroring FastWorkflow types (prefer importing canonical FastWorkflow models where available).

- **Session Layer (`UserSessionManager`)**
  - In‑memory registry: `{ user_id: UserRuntime }`.
  - `UserRuntime` fields: `active_conversation_id`, `chat_session`, `lock`, `show_agent_traces`.
  - Serializes turns per user using an asyncio lock.

- **Workflow Adapter**
  - Bridges HTTP inputs to `ChatSession` by placing user messages on the `user_message_queue` (or executing an `Action`) and reading `CommandOutput` from `command_output_queue`.
  - When enabled, drains trace events from `command_trace_queue` into the response `traces` field.

- **Persistence Adapter (`ConversationStore`)**
  - Rdict database per user under `USER_CONVERSATIONS_SPEEDDICT_FOLDERNAME`.
  - Schema and functional rules described in Section 8.

- **Environment Loader**
  - Loads environment variables strictly from `env_file_path` and `passwords_file_path` files passed to `/initialize`.
  - Calls `fastworkflow.init(env_vars=<file_based_env_dict>)`.

#### 5. Request Lifecycle (Per Endpoint)

- **POST `/initialize`**
  - Validate `workflow_path` exists (optionally warn if `_commands/` missing).
  - Load env from the two files only; call `fastworkflow.init(...)`.
  - Create `ChatSession` with forced agent mode. Restore prior conversation when available; otherwise start a new one.
  - Start the workflow via `chat_session.start_workflow(...)` using provided `context`, `startup_command`, or `startup_action` (mutually exclusive).
  - Store `UserRuntime` in `UserSessionManager` and return `{ user_id }`.

- **POST `/invoke_agent`**
  - Require existing session; enforce single in‑flight turn per user (acquire user lock).
  - If the query begins with `/`, strip all leading slashes before processing.
  - Enqueue the user message and wait for `CommandOutput` up to `timeout_seconds`.
  - If `show_agent_traces=true`, include collected traces in `CommandOutput.traces`.

- **POST `/invoke_agent_stream`**
  - Require existing session; enforce single in‑flight turn per user (acquire user lock).
  - If the query begins with `/`, strip all leading slashes before processing.
  - Set response content-type to `text/event-stream` for SSE.
  - Enqueue the user message.
  - If `show_agent_traces=true`, continuously drain `command_trace_queue` and emit each trace as an SSE event (`event: trace`).
  - Wait for `CommandOutput` up to `timeout_seconds`.
  - Emit the final `CommandOutput` as an SSE event (`event: command_output`).
  - Handle errors by emitting `event: error` before closing the stream.

- **POST `/invoke_assistant`**
  - Same path as agent but deterministic/assistant execution (no planning).
  - Returns `CommandOutput`.

- **POST `/perform_action`**
  - Validate session; accept an explicit `Action` and execute it directly (bypass parameter extraction).
  - Serialize per user; return `CommandOutput` or timeout.

- **POST `/new_conversation`**
  - Persist current conversation to Rdict with generated `topic` (unique per user) and `summary`, both produced synchronously via `dspy.ChainOfThought()`.
  - On success, rotate to a new internal conversation id and clear runtime history; on failure, log critical, return 500, do not rotate.

- **GET `/conversations`**
  - Return up to `FASTWORKFLOW_CONVERSATIONS_LIST_LIMIT` latest conversations for the user by `updated_at` desc, each including `{ conversation_id, topic, summary }`.

- **POST `/post_feedback`**
  - Attach optional feedback to the latest turn of the active conversation.
  - Validate that at least one of `binary_or_numeric_score` or `nl_feedback` is provided; both may be provided.

- **POST `/admin/dump_all_conversations`**
  - Iterate all users and conversations in Rdict; write a JSONL file to the provided folder and return the file path.

- **POST `/activate_conversation`**
  - Body: `{ user_id, conversation_id }`. Locate the conversation by ID for the user and set it as active. 404 if not found.

- **GET `/`**
  - Simple HTML page linking to `/docs`; also serves as a health check.

#### 6. Data Models (Alignment with FastWorkflow)
- **Import first**: When FastWorkflow provides canonical Pydantic/dataclass models for these types, import and use them to avoid divergence.
- **Otherwise mirror the spec** (sketches only):
  - `InitializationRequest`, `InitializationResponse`
  - `InvokeRequest`, `PerformActionRequest`, `NewConversationRequest`, `PostFeedbackRequest`
  - `Action`, `CommandResponse`, `CommandOutput`
- **Fields and semantics**:
  - `CommandOutput.traces` is included only when `show_agent_traces=true`.
  - `Action` equals the runtime execution object consumed by `CommandExecutor`.

#### 7. Error Handling
- **Status codes**
  - 404 Not Found: Missing `user_id` / session not found.
  - 409 Conflict: Concurrent turn already in progress for the same `user_id`.
  - 422 Unprocessable Entity: Validation failures, invalid paths, invalid action schema, XOR violations (e.g., both `startup_command` and `startup_action` provided; or neither feedback field provided in `/post_feedback`).
  - 500 Internal Server Error: Unexpected errors (log with stack trace; avoid broad except without logging).
  - 504 Gateway Timeout: No `CommandOutput` before `timeout_seconds`.
- **Error body**: `{ "detail": "<human‑readable message>" }`.

#### 8. Persistence Model (Rdict)
- **Database layout** (one file per user under `USER_CONVERSATIONS_SPEEDDICT_FOLDERNAME`):
  - Key: `meta` → `{ last_conversation_id: int }`
  - Key: `conv:<id>` →
    - `topic: str` (unique per user; enforce uniqueness via case‑insensitive and whitespace‑insensitive comparison; on collision append incrementing integer)
    - `summary: str`
    - `created_at: int` (epoch ms)
    - `updated_at: int`
    - `turns: [ { "conversation summary": str, "conversation_traces": str (JSON), feedback: { binary_or_numeric_score: bool|float|null, nl_feedback: str|null, timestamp: int } | null } ]`
- **Constraints**
  - Single active conversation per user to avoid write concurrency.
  - Histories persist across restarts; default resume to the last conversation.

#### 9. Concurrency & Timeouts
- **Per‑user serialization**: Exactly one in‑flight turn per `user_id` via an asyncio lock stored in `UserRuntime`.
- **Request timeout**: Default `timeout_seconds=60` (per request, configurable per call).

#### 10. CORS & Security
- **CORS**: Allow configured origins; default `*` in development only.
- **Workflow path restrictions**: Enforce that `workflow_path` is valid. If a path allow‑list is configured, restrict resolution accordingly.
- **Logging**: Log each call with `user_id`, action/command, outcome, and timing. Keep FastWorkflow file logging to `action.jsonl` unchanged.

#### 11. Configuration & Environment Loading
- **Env sources**: Load environment variables strictly from `env_file_path` and `passwords_file_path` supplied to `/initialize`.
- **Initialization**: Call `fastworkflow.init(env_vars=<file_based_env_dict>)`.
- **Operational environment variables**:
  - `USER_CONVERSATIONS_SPEEDDICT_FOLDERNAME`: base folder for per‑user Rdict files
  - `FASTWORKFLOW_CONVERSATIONS_LIST_LIMIT`: max conversations returned by `/conversations`
  - `FASTWORKFLOW_SHUTDOWN_MAX_WAIT_SECONDS`: max wait to finish active turns before shutdown persistence
  - `LLM_CONVERSATION_STORE`: LiteLLM model string for conversation topic/summary generation
  - `LITELLM_API_KEY_CONVERSATION_STORE`: API key for the `LLM_CONVERSATION_STORE` model

#### 12. Deployment
- **Uvicorn**: Run with `uvicorn services.run_fastapi.main:app --host 0.0.0.0 --port 8000`.
- **Lifespan**: Use default unless startup/shutdown hooks are added; only enable `lifespan="on"` if explicitly needed.

#### 13. Minimal File/Type Layout
- **Code**: `services/run_fastapi/main.py` contains:
  - FastAPI app and route handlers
  - `UserSessionManager` and `UserRuntime` (in‑memory)
  - Wiring to `ChatSession` queues and optional trace collection
  - SSE streaming logic for `/invoke_agent_stream`
- **Code**: `services/conversation_store.py` contains:
  - Rdict adapters for `ConversationStore`
- **Docs**: This file and the source spec live under `docs/`.
- **Core FastWorkflow**: No changes required. The service uses existing APIs:
  - `ChatSession.user_message_queue`, `command_output_queue`, `command_trace_queue`
  - `ChatSession.start_workflow(...)`, `clear_conversation_history()`
  - Canonical models: `Action`, `CommandResponse`, `CommandOutput` from `fastworkflow.__init__`

#### 14. Testing Guidance (from spec)
- **Unit**
  - SessionManager: concurrency and lifecycle
  - Env loading from files only
  - Validation rules: XOR handling and feedback presence
  - Timeout behavior: return 504 when no output arrives
  - SSE formatting: verify correct event structure for trace and command_output events
- **Integration**
  - Spin up the FastAPI app via `TestClient`
  - Initialize with a sample workflow and perform a single agent turn; assert `CommandOutput` shape
  - Exercise `/perform_action` using a known command; verify response
  - Validate `/new_conversation` persistence and `/conversations` listing behavior
  - Test `/invoke_agent_stream` by parsing SSE events: verify trace events arrive before final command_output event; validate JSON structure of each event
  - Test streaming with `show_agent_traces=false`: verify only command_output event is emitted
  - Test streaming error handling: verify `event: error` emission when turn fails

#### 15. Future Enhancements (as enumerated in the spec)
- WebSocket support as an alternative to SSE for bidirectional communication
- Session TTL and eviction policy; persistence layer for sessions if required
- Richer observability: correlate CLI trace colors to structured HTTP traces
- Security hardening: workflow/path allow‑list, authn/z

#### 16. Design Principles Recap
- **Minimal**: Only implement explicitly specified features and behaviors.
- **Modular**: Keep API, session management, workflow bridging, persistence, and env loading as separate concerns with clear interfaces.
- **Parity**: Mirror the CLI runner’s semantics for initialization and single‑turn orchestration.
- **Observability**: Include traces in responses only when enabled; always log with sufficient context.



### FastWorkflow FastAPI Service — Architecture & Design

Based on: [fastworkflow_fastapi_spec.md](mdc:docs/fastworkflow_fastapi_spec.md)

#### 1. Scope and Goals
- **Purpose**: Provide a minimal, modular FastAPI service that exposes FastWorkflow workflows over HTTP with behavior matching the CLI runner, replacing its interactive loop with synchronous endpoints.
- **Strict adherence**: This document mirrors the referenced spec and does not introduce functionality beyond what is explicitly stated.

#### 2. Non‑Goals
- **UI**: No user interface. Only REST endpoints with OpenAPI/Swagger at `/docs`.
- **Streaming**: Streamable HTTP at `/invoke_agent_stream` (NDJSON by default; SSE also supported per session preference). MCP Streamable HTTP is also available when MCP is mounted and maps to the same streaming implementation.

#### 3. High‑Level Architecture
- **Process‑wide app**: A single FastAPI app hosts an in‑memory `UserSessionManager` keyed by `user_id`.
- **Per‑user runtime**:
  - Active `ChatSession` (always agent mode) bound to a current conversation (internal id only).
  - Persistent `ConversationStore` per user backed by Rdict (one DB file per user).
- **Synchronous turn execution**: For each request, enqueue user input and synchronously await a `CommandOutput` from the `command_output_queue` with timeout and single‑turn serialization.
- **Traces**: Trace events are collected by default and included in the synchronous response, or emitted incrementally for the streaming endpoint.

#### 4. Minimal Module Decomposition
- **API Layer (FastAPI routes)**
  - Endpoint handlers for all specified routes (Section 6 below).
  - Request validation via Pydantic models mirroring FastWorkflow types (prefer importing canonical FastWorkflow models where available).

- **Session Layer (`UserSessionManager`)**
  - In‑memory registry: `{ user_id: UserRuntime }`.
  - `UserRuntime` fields: `active_conversation_id`, `chat_session`, `lock`, `stream_format`.

- **Workflow Adapter**
  - Bridges HTTP inputs to `ChatSession` by placing user messages on the `user_message_queue` (or executing an `Action`) and reading `CommandOutput` from `command_output_queue`.
  - When enabled, drains trace events from `command_trace_queue` into the response `traces` field (REST). Streaming is handled by NDJSON or SSE for REST; MCP tools mirror the same server-side streaming.

- **Persistence Adapter (`ConversationStore`)**
  - Rdict database per user under `SPEEDDICT_FOLDERNAME/user_conversations`.
  - Schema and functional rules described in Section 8.

- **Startup Configuration Loader**
  - Parse CLI args at process startup (e.g., `--workflow_path`, `--env_file_path`, `--passwords_file_path`, `--context`, `--startup_command`, `--startup_action`).
  - Load env strictly from provided files and call `fastworkflow.init(env_vars=...)`.
  - Store configuration in process-wide variables for use by `/initialize`.

#### 5. Request Lifecycle (Per Endpoint)

- **POST `/initialize`**
  - Create `ChatSession` with forced agent mode. Restore prior conversation when available; otherwise start a new one.
  - Start the workflow via `chat_session.start_workflow(...)` using configuration loaded at process startup (CLI args/env).
  - Store `UserRuntime` in `UserSessionManager` and return a JWT `TokenResponse` containing `access_token`, `refresh_token`, `token_type`, `expires_in`, `user_id`, and `workflow_info`.

- **POST `/invoke_agent`**
  - Require existing session; enforce single in‑flight turn per user (acquire user lock).
  - If the query begins with `/`, strip all leading slashes before processing.
  - Enqueue the user message and wait for `CommandOutput` up to `timeout_seconds`.
  - Include collected traces in `CommandOutput.traces`.

- **POST `/invoke_agent_stream`**
  - Streamable HTTP (NDJSON) and SSE: emits `{ "type": "trace" }` records and a final `{ "type": "output" }` record.

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
  - Return up to `limit` (default `20`) latest conversations for the user by `updated_at` desc, each including `{ conversation_id, topic, summary }`.

- **POST `/post_feedback`**
  - Attach optional feedback to the latest turn of the active conversation.
  - Validate that at least one of `binary_or_numeric_score` or `nl_feedback` is provided; both may be provided.

- **POST `/admin/dump_all_conversations`**
  - Iterate all users and conversations in Rdict; write a JSONL file to the provided folder and return the file path.

- **POST `/activate_conversation`**
  - Body: `{ user_id, conversation_id }`. Locate the conversation by ID for the user and set it as active. 404 if not found.

- **POST `/admin/generate_mcp_token`**
  - Generate long-lived access tokens for MCP client configuration. Default expiration: 365 days. Returns TokenResponse with access_token (no refresh_token).

- **GET `/`**
  - Simple HTML page linking to `/docs`; also serves as a health check.

#### 6. Data Models (Alignment with FastWorkflow)
- **Import first**: When FastWorkflow provides canonical Pydantic/dataclass models for these types, import and use them to avoid divergence.
- **Otherwise mirror the spec** (sketches only):
  - `InitializationRequest`, `InitializationResponse`
  - `InvokeRequest`, `PerformActionRequest`, `PostFeedbackRequest`
  - `Action`, `CommandResponse`, `CommandOutput`
- **Fields and semantics**:
  - `CommandOutput.traces` is included by default (if any traces were produced during the turn).
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
- **Startup**: Configuration is provided via CLI args and environment on process start; do not accept these in `/initialize`.
- **Operational environment variables**:
  - `SPEEDDICT_FOLDERNAME`: base folder for per‑user Rdict files
  - `LLM_CONVERSATION_STORE`: LiteLLM model string for conversation topic/summary generation
  - `LITELLM_API_KEY_CONVERSATION_STORE`: API key for the `LLM_CONVERSATION_STORE` model

#### 12. Deployment
- **Uvicorn**: Run with `uvicorn services.run_fastapi.main:app --host 0.0.0.0 --port 8000`.
- **Lifespan**: Use default unless startup/shutdown hooks are added; only enable `lifespan="on"` if explicitly needed.

#### 13. Minimal File/Type Layout
- **Code**: `services/run_fastapi/main.py` contains:
  - FastAPI app and route handlers
  - `UserSessionManager` and `UserRuntime` (in‑memory)
  - Wiring to `ChatSession` queues and optional trace collection (REST)
  - OpenAPI security for JWT Bearer (all endpoints except `/initialize`, `/refresh_token`)
- **Code**: `services/run_fastapi_mcp/mcp_specific.py` contains:
  - MCP mounting using `fastapi_mcp` that exposes FastAPI endpoints as MCP tools
  - Excludes admin-only and non-applicable endpoints
  - Maps the REST streaming endpoint `/invoke_agent_stream` (operation_id `invoke_agent`) for MCP streaming
- **Code**: `services/conversation_store.py` contains:
  - Rdict adapters for `ConversationStore`

#### 14. Testing Guidance
- Parse NDJSON on `/invoke_agent_stream`: verify trace records arrive before the final `command_output` record; validate structure.
- Verify `/invoke_agent` returns `traces` array populated when produced.
- Concurrency: ensure 409 when a turn is in progress.
- Timeouts: ensure 504 when no `CommandOutput` arrives within `timeout_seconds`.



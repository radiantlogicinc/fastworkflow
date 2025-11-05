### FastWorkflow FastAPI Service — Architecture & Design

Based on: [fastworkflow_fastapi_spec.md](mdc:docs/fastworkflow_fastapi_spec.md)

#### 1. Scope and Goals
- **Purpose**: Provide a minimal, modular FastAPI service that exposes FastWorkflow workflows over HTTP with behavior matching the CLI runner, replacing its interactive loop with synchronous endpoints.
- **Strict adherence**: This document mirrors the referenced spec and does not introduce functionality beyond what is explicitly stated.

#### 2. Non‑Goals
- **UI**: No user interface. Only REST endpoints with OpenAPI/Swagger at `/docs`.
- **Streaming**: Streamable HTTP at `/invoke_agent_stream` (NDJSON by default; SSE also supported per session preference). MCP Streamable HTTP is also available when MCP is mounted and maps to the same streaming implementation.

#### 3. High‑Level Architecture
- **Process‑wide app**: A single FastAPI app hosts an in‑memory `ChannelSessionManager` keyed by `channel_id`.
- **Per‑channel runtime**:
  - Active `ChatSession` (always agent mode) bound to a current conversation (internal id only).
  - Persistent `ConversationStore` per channel backed by Rdict (one DB file per channel).
- **Synchronous turn execution**: For each request, enqueue channel input and synchronously await a `CommandOutput` from the `command_output_queue` with timeout and single‑turn serialization.
- **Traces**: Trace events are collected by default and included in the synchronous response, or emitted incrementally for the streaming endpoint.
- **Auth modes**:
  - JWT access/refresh tokens are created at `/initialize` and required on authenticated endpoints. Tokens include `sub` (channel_id) and `uid` (user_id, if provided).
  - Trusted mode (default): tokens are unencrypted.
  - Secure mode: tokens are encrypted

#### 4. Minimal Module Decomposition
- **API Layer (FastAPI routes)**
  - Endpoint handlers for all specified routes (Section 6 below).
  - Request validation via Pydantic models mirroring FastWorkflow types (prefer importing canonical FastWorkflow models where available).

- **Session Layer (`ChannelSessionManager`)**
  - In‑memory registry: `{ channel_id: ChannelRuntime }`.
  - `ChannelRuntime` fields: `active_conversation_id`, `chat_session`, `lock`, `stream_format`.

- **Workflow Adapter**
  - Bridges HTTP inputs to `ChatSession` by placing channel messages on the `user_message_queue` (or executing an `Action`) and reading `CommandOutput` from `command_output_queue`.
  - When enabled, drains trace events from `command_trace_queue` into the response `traces` field (REST). Streaming is handled by NDJSON or SSE for REST; MCP tools mirror the same server-side streaming.

- **Persistence Adapter (`ConversationStore`)**
  - Rdict database per channel under `SPEEDDICT_FOLDERNAME/channel_conversations`.
  - Schema and functional rules described in Section 8.

- **Startup Configuration Loader**
  - Parse CLI args at process startup (e.g., `--workflow_path`, `--env_file_path`, `--passwords_file_path`, `--context`, `--startup_command`, `--startup_action`).
  - Load env strictly from provided files and call `fastworkflow.init(env_vars=...)`.
  - Store configuration in process-wide variables for use by `/initialize`.

#### 5. Request Lifecycle (Per Endpoint)

- **POST `/initialize`**
  - Create `ChatSession` with forced agent mode. Restore prior conversation when available; otherwise start a new one.
  - Accept optional `startup_command` or `startup_action` (mutually exclusive). If provided, immediately execute and capture a `CommandOutput` which is returned in the response and persisted to `ConversationStore` as the first turn.
  - Accept optional `user_id` (required if startup is provided) for attribution and trace enrichment.
  - Create and return JWT access/refresh tokens in both trusted and secure modes. In trusted mode, the tokens are unencrypted
  - Store `ChannelRuntime` in `ChannelSessionManager` and return response including tokens and optional `startup_output`.

- **POST `/invoke_agent`**
  - Require existing session; enforce single in‑flight turn per channel (acquire channel lock).
  - If the query begins with `/`, strip all leading slashes before processing.
  - Enqueue the channel message and wait for `CommandOutput` up to `timeout_seconds`.
  - Include collected traces in `CommandOutput.traces`.
  - `user_id` is extracted from the Authorization JWT (`uid` claim) in secure mode and trusted modes.

- **POST `/invoke_agent_stream`**
  - Streamable HTTP (NDJSON) and SSE: emits `{ "type": "trace" }` records and a final `{ "type": "output" }` record.

- **POST `/invoke_assistant`**
  - Same path as agent but deterministic/assistant execution (no planning).
  - Returns `CommandOutput`.
  - `user_id` is extracted from the Authorization JWT (`uid` claim) in secure mode and trusted modes.

- **POST `/perform_action`**
  - Validate session; accept an explicit `Action` and execute it directly (bypass parameter extraction).
  - Serialize per channel; return `CommandOutput` or timeout.
  - `user_id` is extracted from the Authorization JWT (`uid` claim) in secure mode and trusted modes.

- **POST `/new_conversation`**
  - Persist current conversation to Rdict with generated `topic` (unique per channel) and `summary`, both produced synchronously via `dspy.ChainOfThought()`.
  - On success, rotate to a new internal conversation id and clear runtime history; on failure, log critical, return 500, do not rotate.

- **GET `/conversations`**
  - Return up to `limit` (default `20`) latest conversations for the channel by `updated_at` desc, each including `{ conversation_id, topic, summary }`.

- **POST `/post_feedback`**
  - Attach optional feedback to the latest turn of the active conversation.
  - Validate that at least one of `binary_or_numeric_score` or `nl_feedback` is provided; both may be provided.

- **POST `/admin/dump_all_conversations`**
  - Iterate all channels and conversations in Rdict; write a JSONL file to the provided folder and return the file path.

- **POST `/activate_conversation`**
  - Body: `{ channel_id, conversation_id }`. Locate the conversation by ID for the channel and set it as active. 404 if not found.

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
  - Traces are enriched with `user_id` (when provided) and `raw_command`.

#### 7. Error Handling
- **Status codes**
  - 404 Not Found: Missing `channel_id` / session not found.
  - 409 Conflict: Concurrent turn already in progress for the same `channel_id`.
  - 422 Unprocessable Entity: Validation failures, invalid paths, invalid action schema, XOR violations (e.g., both `startup_command` and `startup_action` provided; or neither feedback field provided in `/post_feedback`).
  - 500 Internal Server Error: Unexpected errors (log with stack trace; avoid broad except without logging).
  - 504 Gateway Timeout: No `CommandOutput` before `timeout_seconds`.
- **Error body**: `{ "detail": "<human‑readable message>" }`.

#### 8. Persistence Model (Rdict)
- **Database layout** (one file per channel under `USER_CONVERSATIONS_SPEEDDICT_FOLDERNAME`):
  - Key: `meta` → `{ last_conversation_id: int }`
  - Key: `conv:<id>` →
    - `topic: str` (unique per channel; enforce uniqueness via case‑insensitive and whitespace‑insensitive comparison; on collision append incrementing integer)
    - `summary: str`
    - `created_at: int` (epoch ms)
    - `updated_at: int`
    - `turns: [ { "conversation summary": str, "conversation_traces": str (JSON), feedback: { binary_or_numeric_score: bool|float|null, nl_feedback: str|null, timestamp: int } | null } ]`
- **Constraints**
  - Single active conversation per channel to avoid write concurrency.
  - Histories persist across restarts; default resume to the last conversation.

#### 9. Concurrency & Timeouts
- **Per‑channel serialization**: Exactly one in‑flight turn per `channel_id` via an asyncio lock stored in `ChannelRuntime`.
- **Request timeout**: Default `timeout_seconds=60` (per request, configurable per call).

#### 10. CORS & Security
- **CORS**: Allow configured origins; default `*` in development only.
- **Workflow path restrictions**: Enforce that `workflow_path` is valid. If a path allow‑list is configured, restrict resolution accordingly.
- **Logging**: Log each call with `channel_id`, `user_id` (when available), action/command, outcome, and timing. Keep FastWorkflow file logging to `action.jsonl` unchanged.

#### 11. Configuration & Environment Loading
- **Startup**: Configuration is provided via CLI args and environment on process start; do not accept these in `/initialize`.
- **Operational environment variables**:
  - `SPEEDDICT_FOLDERNAME`: base folder for per‑channel Rdict files
  - `LLM_CONVERSATION_STORE`: LiteLLM model string for conversation topic/summary generation
  - `LITELLM_API_KEY_CONVERSATION_STORE`: API key for the `LLM_CONVERSATION_STORE` model
  - `--expect_encrypted_jwt`: when set, secure mode is enabled; otherwise trusted mode (no encryption) is used.

#### 12. Deployment
- **Uvicorn**: Run with `uvicorn services.run_fastapi.main:app --host 0.0.0.0 --port 8000`.
- **Lifespan**: Use default unless startup/shutdown hooks are added; only enable `lifespan="on"` if explicitly needed.

#### 13. Minimal File/Type Layout
- **Code**: `services/run_fastapi/main.py` contains:
  - FastAPI app and route handlers
  - `ChannelSessionManager` and `ChannelRuntime` (in‑memory)
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
- Initialize with startup action/command returns `startup_output` and persists the first turn.
- Trusted mode returns unencrypted tokens; secure mode returns encrypted tokens. In both cases, the tokens contain `sub` (channel_id) and `uid` (user_id when provided).



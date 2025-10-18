### FastWorkflow MCP Server — Comprehensive Specification (HTTP-only)

This document specifies an HTTP-only Model Context Protocol (MCP) server for FastWorkflow. It is intended for client agents (e.g., Claude Desktop) to interrogate and use FastWorkflow workflows via a stable, sessioned API surface.

The spec balances two interaction styles:
- Coarse-grained tools mirroring the FastAPI behavior (initialize, invoke_agent/assistant, conversations, feedback).
- A fine-grained explicit `execute_command` tool for clients that prefer to perform their own planning and parameter formatting.

No agent-internal tools (e.g., ask_user, intent_misunderstood) are exposed. Current context is not returned by `get_workflow_info`; clients should use the `what_is_current_context` command (discoverable via `get_commands`) and read the `context` field in every `CommandOutput`.


## 1) Scope and Goals

- Expose an HTTP-only MCP server (streamable HTTP transport) for a single, pre-configured workflow loaded at server startup.
- Provide sessioned interactions: initialize, NL agentic turns, deterministic assistant turns, explicit command execution, conversation lifecycle, and feedback.
- Default `user_id` handling: if omitted, use `"default_user"`.
- Offer discoverability: workflow high-level info (purpose/contexts) and command metadata for the active context.
- Optionally include live trace events in responses when enabled at server startup.

Non-goals:
- Dynamic workflow loading at runtime.
- Admin endpoints (e.g., dump_all_conversations) over MCP.
- Exposing action logs as resources (use traces instead).
- Returning current context in `get_workflow_info` (clients should call the dedicated command or inspect `CommandOutput.context`).


## 2) Protocol and Transport

- Transport: HTTP-only via MCP Streamable HTTP (mount the MCP server into the existing FastAPI app using `fastapi_mcp`). MCP tools map to the same REST streaming implementation (`/invoke_agent_stream` exposed as `invoke_agent`).
- Sessions: Use the protocol/session mechanism provided by the MCP implementation. The server maintains a runtime keyed by the MCP session id; tools are session-scoped.
- Statefulness: Server-side state per MCP session; persistent conversation history per `user_id`.
- Implementation note: Prefer `fastapi_mcp` to mount MCP directly on the same FastAPI app (ASGI transport) for a unified deployment and zero extra HTTP hops. See references: [fastapi_mcp (GitHub)](https://github.com/tadata-org/fastapi_mcp), [fastapi_mcp (intro)](https://dev.to/auden/introducing-fastapi-mcp-effortless-ai-integration-for-your-fastapi-apis-2c8c).


## 3) Startup Configuration and Lifecycle (Server-Side Only)

All configuration is provided at server startup (Ops-managed). Clients do not provide environment paths or workflow paths when calling `initialize`.

Recommended environment variables (aligning with CLI runner behavior):
- `WORKFLOW_PATH`: Absolute path to the workflow folder.
- `ENV_FILE_PATH`: Path to `.env` with base variables.
- `PASSWORDS_FILE_PATH`: Path to `passwords.env` with secrets.
- `CONTEXT`: Optional JSON string for initial workflow context.
- `STARTUP_COMMAND`: Optional startup command string.
- `STARTUP_ACTION`: Optional startup action JSON (stringified) matching `fastworkflow.Action`.
- `PROJECT_FOLDERPATH`: Optional project folder path for workflow context.
- Conversation store and LLM config:
  - `SPEEDDICT_FOLDERNAME`
  - `LLM_CONVERSATION_STORE`, `LITELLM_API_KEY_CONVERSATION_STORE`

Startup sequence (parity with CLI `run/__main__.py`):
1) Load and merge env files using `dotenv_values(ENV_FILE_PATH)` and `dotenv_values(PASSWORDS_FILE_PATH)`; validate required keys.
2) `fastworkflow.init(env_vars=env_vars)`
3) Store workflow configuration in `UserSessionManager` or module-level variables: `WORKFLOW_PATH`, `CONTEXT`, `STARTUP_COMMAND`, `STARTUP_ACTION`, `PROJECT_FOLDERPATH`.
4) Initialize the `UserSessionManager` (empty state initially; sessions created on `initialize` calls).

**Important:** `ChatSession` instances are created per-user when `initialize` is called, NOT at server startup. Each user gets their own `ChatSession` with isolated queues and state.

Reference (CLI parity): see `fastworkflow/run/__main__.py` for env loading and startup orchestration.


## 4) Session Model and Persistence

- Session Key: MCP session id (managed by transport). The server binds runtime state to this id.
- Per-session runtime:
  - `user_id`: provided on `initialize` (defaults to `"default_user"`).
  - `chat_session`: reference to the process-wide `ChatSession` (agent-capable; one workflow per process). Per-session queues (message/output/trace) live on the `chat_session` instance.
  - `active_conversation_id`: current conversation for the `user_id`.
  - `lock`: single in-flight turn per session to serialize requests (409 on concurrent turn attempts).
- Persistence: `ConversationStore` backed by Rdict, one DB file per `user_id` under `USER_CONVERSATIONS_SPEEDDICT_FOLDERNAME`. The last conversation is restored by default, or a specific `conversation_id` if provided.
- Timeouts: Per-call `timeout_seconds` (default 60). Return a 504-equivalent MCP error if no output in time.


## 5) Tool Surface and Contracts

All tools are session-scoped (require an active MCP session). Unless specified, omit `user_id` on subsequent calls; the session owns it.

### 5.1 initialize
- Purpose: Create or resume a user session bound to the pre-configured workflow.
- Params:
  ```json
  { "user_id": "string (optional)", "conversation_id": "string|null (optional)" }
  ```
- Behavior:
  - If `user_id` is omitted, use `"default_user"`.
  - Restore the specified conversation if provided and exists; else restore last; else start new.
- Returns:
  ```json
  { "workflow_info": { /* metadata or null until implemented */ } }
  ```
- Errors: 422 (invalid), 500 (init failure)

### 5.2 get_workflow_info
- Purpose: Provide workflow-level metadata so clients understand the workflow’s purpose.
- Params: none
- Returns:
  ```json
  { "workflow_name": "string", "description": "string", "purpose": "string", "available_contexts": ["string", ...] }
  ```
- Notes: Do NOT include `current_context` here. Clients can call the `what_is_current_context` command (discoverable via `get_commands`) and read `CommandOutput.context` on every turn.
- Errors: 404 (no session)

### 5.3 get_commands
- Purpose: Discover available commands in the active context (both human-readable and structured).
- Params: none
- Returns:
  ```json
  { "display_text": "string", "commands": [ { "name": "qualified/name", "description": "string", "parameters": [ { "name": "string", "type": "string", "required": true, "description": "string" } ], "examples": ["qualified/name <param>value</param>"] } ] }
  ```
- Source: `CommandMetadataAPI` for the currently active context.
- Errors: 404 (no session)

### 5.4 invoke_agent
- Purpose: Agentic NL turn with server-side planning. Strip leading `/` if present.
- Params:
  ```json
  { "user_query": "string", "timeout_seconds": 60 }
  ```
- Returns: streaming NDJSON events `{ "type": "trace" }` (multiple) and final `{ "type": "output" }` that includes `CommandOutput` (without `traces`), mirroring REST `/invoke_agent_stream`.
- Errors: 404 (no session), 409 (turn in progress), 504 (timeout), 500 (internal)

### 5.5 invoke_assistant
- Purpose: Deterministic (no planning) turn; treat input as an imperative assistant instruction.
- Params:
  ```json
  { "user_query": "string", "timeout_seconds": 60 }
  ```
- Returns: `CommandOutput` (JSON)
- Errors: 404/409/504/500

### 5.6 execute_command
- Purpose: Execute a precise command string for clients that format parameters themselves.
- Params:
  ```json
  { "command": "QualifiedName <param1>value</param1> <param2>value</param2> ..." }
  ```
- Returns:
  ```json
  { "response_text": "string", "success": true, "traces": [ { ... } ] }
  ```
- Errors: 422 (malformed command), 500
- Notes: Same semantics as the internal `_execute_workflow_query` agent tool. Traces are emitted when enabled.

### 5.7 new_conversation
- Purpose: Persist current history (topic + summary via LLM) and rotate to a new conversation.
- Params: none
- Returns:
  ```json
  { "status": "ok", "new_conversation_id": "string" }
  ```
- Errors: 404 (no session), 500 (LLM failure — do not rotate)

### 5.8 list_conversations
- Purpose: List latest N conversations by `updated_at` desc for the session’s user.
- Params:
  ```json
  { "limit": 10 }
  ```
- Returns:
  ```json
  { "conversations": [ { "conversation_id": "string", "topic": "string", "summary": "string", "updated_at": 0 } ] }
  ```

### 5.9 activate_conversation
- Purpose: Switch to a different conversation for this session’s user.
- Params:
  ```json
  { "conversation_id": "string" }
  ```
- Returns:
  ```json
  { "status": "ok" }
  ```
- Errors: 404 (not found)

### 5.10 post_feedback
- Purpose: Attach optional feedback to the latest turn in the active conversation.
- Params:
  ```json
  { "binary_or_numeric_score": true, "nl_feedback": "string|null" }
  ```
- Returns:
  ```json
  { "status": "ok" }
  ```
- Errors: 404 (no session), 422 (both fields null)

Out of scope for MCP tools: `perform_action`, admin operations (dump/export), action log resources.


## 6) Events and Streaming (Streamable HTTP)

Tools may emit incremental outputs using MCP Streamable HTTP.

### 6.1 Streaming Semantics

- MCP transport: Prefer NDJSON for partial delivery; SSE is available to REST clients.
- Tools SHOULD yield partial outputs while the turn is executing, and finally yield the completed `CommandOutput` payload wrapped in a `{ "type": "output" }` event.
- For Streamable HTTP (NDJSON), emit structured partials:
  - `{ "type": "trace", "data": <trace_event> }` (multiple)
  - `{ "type": "output", "data": <CommandOutput_without_traces> }` (final)
- Do NOT include a `traces` array in the final `CommandOutput` when streaming is used; for non-streaming calls, attach collected `traces` to the final `CommandOutput`.

### 6.2 Implementation Notes with fastapi_mcp

- Mount MCP onto the FastAPI app using `fastapi_mcp`; the `invoke_agent` tool maps directly to the REST `/invoke_agent_stream` endpoint (operation_id `invoke_agent`).
  - NDJSON stream: `{ "type": "trace", "data": <trace_event> }` (multiple), then `{ "type": "output", "data": <CommandOutput_without_traces> }` (final).
  - SSE is supported for REST clients; MCP clients should prefer NDJSON.

### 6.3 Trace Event Shape

Each trace event (aligned with `CommandTraceEvent` in FastWorkflow):
```json
{
  "timestamp": 1712345678901,
  "direction": "agent_to_workflow" | "workflow_to_agent",
  "raw_command": "string|null",
  "command_name": "string|null",
  "parameters": {"...": "..."} | null,
  "response_text": "string|null",
  "success": true | false | null
}
```


## 7) Prompts (MCP prompts/list and prompts/get)

Prompts are server-advertised templates for client convenience; they do not trigger server actions.

- `format-command`: Given command metadata and a user intent, format a single executable command with XML-tagged parameters.
  - Args: `intent (string)`, `metadata (string|object)`
- `clarify-params`: Compose a concise clarification question for missing parameters.
  - Args: `error_message (string)`, `metadata (string|object)`

**Not included:**
- `name-conversation` and `summarize-conversation`: FastWorkflow handles conversation naming and summarization automatically via LLM; clients do not need these prompts.
- `plan-next-steps`: Redundant with `invoke_agent` server-side planning; may be added as future enhancement if clients request local planning assistance.


## 8) Resources

None required. Do not expose workflow folders or action logs. Commands metadata is provided by `get_commands` (resource duplicate is out of scope). Conversation export resources are out of scope.


## 9) Data Contracts

Prefer importing FastWorkflow canonical types where available (e.g., `CommandOutput`, `CommandResponse`, `Action`). Otherwise, match their structure:

```json
// CommandResponse (sketch)
{
  "response": "string|null",
  "artifacts": {"...": "..."} | null,
  "next_actions": [ {"command_name": "string", "arguments": {"...": "..."}} ] | null,
  "recommendations": ["string"] | null
}
```

```json
// CommandOutput (sketch)
{
  "success": true | false | null,
  "workflow_name": "string|null",
  "context": "string|null",
  "command_name": "string|null",
  "command_parameters": {"...": "..."} | null,
  "command_responses": [ CommandResponse, ... ],
  "traces": [ { ...trace event... } ] | null
}
```


## 10) Errors and Validation

Map failures to MCP/JSON-RPC error codes/messages analogous to HTTP semantics:
- 404: Session/user/conversation not found.
- 409: Concurrent turn in progress for this session.
- 422: Validation failures (e.g., both feedback fields null, malformed command).
- 504: Timeout waiting for `CommandOutput`.
- 500: Unexpected error (log stack trace; do not swallow errors).

Always log errors with sufficient context; avoid broad catch-and-ignore patterns.


## 11) Server Structure (Implementation Notes)

This section outlines a concrete, modular structure for an HTTP-only MCP server using FastMCP. It mirrors the CLI runner’s startup logic and the FastAPI architecture while complying with the constraints in this spec.

### 11.1 Modules

- `server.py` (entrypoint):
  - Loads server configuration from environment.
  - Performs FastWorkflow startup (env merge, `fastworkflow.init`, `ChatSession` creation, `start_workflow`).
  - Instantiates the MCP server and registers tools.
  - Holds a process-wide `UserSessionManager` instance.

- `session_manager.py`:
  - `UserRuntime`: dataclass storing `user_id`, `active_conversation_id`, `chat_session`, `lock`, `show_agent_traces`.
  - `UserSessionManager`: keyed by MCP session id; provides `get_or_create(session_id, user_id)`, `get(session_id)`, and lifecycle helpers.

- `conversation_store.py`:
  - Rdict adapters: load/save list, rotate conversation, fetch latest N, attach feedback.
  - Enforce uniqueness of `topic` per user (case/whitespace-insensitive; append suffix when needed).

- `workflow_adapter.py`:
  - Bridges tool calls to `ChatSession`: places messages on `user_message_queue`, reads `command_output_queue`, and drains `command_trace_queue` when enabled.
  - Helper for explicit `execute_command` invoking the internal path.

### 11.2 Startup Orchestration (parity with CLI)

```python
import os, json
from dotenv import dotenv_values
import fastworkflow

class WorkflowConfig:
    """Stores workflow configuration for all sessions."""
    def __init__(self):
        self.workflow_path = None
        self.context = None
        self.startup_command = None
        self.startup_action = None
        self.project_folderpath = None
        # no show_agent_traces flag; traces are emitted/attached by default

workflow_config = WorkflowConfig()

def _server_startup():
    """
    Initialize FastWorkflow and store configuration.
    Does NOT create ChatSession (that's per-user).
    """
    env_vars = {
        **dotenv_values(os.environ.get("ENV_FILE_PATH", "")),
        **dotenv_values(os.environ.get("PASSWORDS_FILE_PATH", "")),
    }
    # Validate required keys (example parity checks)
    if not env_vars.get("SPEEDDICT_FOLDERNAME"):
        raise ValueError("ENV_FILE_PATH missing SPEEDDICT_FOLDERNAME")
    if not env_vars.get("LITELLM_API_KEY_SYNDATA_GEN"):
        raise ValueError("PASSWORDS_FILE_PATH missing LITELLM_API_KEY_SYNDATA_GEN")

    fastworkflow.init(env_vars=env_vars)

    # Store workflow configuration for use when creating user sessions
    workflow_config.workflow_path = os.environ["WORKFLOW_PATH"]
    workflow_config.context = json.loads(os.environ.get("CONTEXT", "{}")) or None
    workflow_config.startup_command = os.environ.get("STARTUP_COMMAND", "")
    workflow_config.project_folderpath = os.environ.get("PROJECT_FOLDERPATH")
    # traces are implicit; no SHOW_AGENT_TRACES toggle
    
    if os.environ.get("STARTUP_ACTION"):
        workflow_config.startup_action = fastworkflow.Action(**json.loads(os.environ["STARTUP_ACTION"]))

def _create_user_chat_session() -> fastworkflow.ChatSession:
    """
    Create a new ChatSession for a user using stored workflow config.
    Called during initialize() for each new user.
    """
    chat_session = fastworkflow.ChatSession(run_as_agent=True)
    
    chat_session.start_workflow(
        workflow_config.workflow_path,
        workflow_context=workflow_config.context,
        startup_command=workflow_config.startup_command,
        startup_action=workflow_config.startup_action,
        keep_alive=True,
        project_folderpath=workflow_config.project_folderpath
    )

    return chat_session
```

### 11.3 Session Manager

```python
from dataclasses import dataclass
from typing import Optional
import asyncio

@dataclass
class UserRuntime:
    user_id: str
    active_conversation_id: Optional[str]
    chat_session: "fastworkflow.ChatSession"
    lock: asyncio.Lock
    # no show_agent_traces flag here; traces are implicit

class UserSessionManager:
    def __init__(self):
        self._sessions = {}

    def get(self, session_id: str) -> Optional[UserRuntime]:
        return self._sessions.get(session_id)

    def create(self, session_id: str, user_id: str, chat_session) -> UserRuntime:
        ur = UserRuntime(user_id=user_id, active_conversation_id=None, chat_session=chat_session, lock=asyncio.Lock())
        self._sessions[session_id] = ur
        return ur
```

### 11.4 MCP Server Mount (fastapi_mcp)

```python
from fastapi import FastAPI
from fastapi_mcp import FastApiMCP

app = FastAPI()

# ... existing FastAPI routes (initialize/invoke/etc.) remain for REST (sync) use

# Mount MCP on the same FastAPI app for Streamable HTTP (incremental streaming)
mcp = FastApiMCP(
    app,
    mount_path="/mcp",
    title="FastWorkflow MCP",
    # Exclude admin endpoints from MCP exposure
    exclude_operations=["dump_all_conversations"],
)

# Register MCP prompts (discoverable via prompts/list)
mcp.add_prompt(
    name="format-command",
    description="Format a single executable command with XML-tagged parameters",
    arguments=[{"name": "intent", "required": True}, {"name": "metadata", "required": True}],
    # Implementation returns MCP messages (user/system) the client can render
    handler=lambda intent, metadata: [
        {"role": "user", "content": {"type": "text", "text": f"Intent: {intent}\nMetadata: {metadata}"}}
    ],
)

mcp.add_prompt(
    name="clarify-params",
    description="Compose a concise clarification question for missing parameters",
    arguments=[{"name": "error_message", "required": True}, {"name": "metadata", "required": True}],
    handler=lambda error_message, metadata: [
        {"role": "user", "content": {"type": "text", "text": f"{error_message}\n\nMetadata: {metadata}"}}
    ],
)
```

Notes:
- Prefer mounting MCP with `fastapi_mcp` to leverage ASGI transport and share DI/auth with FastAPI. See [fastapi_mcp (GitHub)](https://github.com/tadata-org/fastapi_mcp), [intro article](https://dev.to/auden/introducing-fastapi-mcp-effortless-ai-integration-for-your-fastapi-apis-2c8c).
- Exclude the admin endpoint `/admin/dump_all_conversations` from MCP exposure (security principle: do not expose admin through MCP tools).

### 11.5 Trace Collection and Streaming

**Streamable HTTP (MCP) clients:**
- Implement tools as async generators
- As traces arrive on `command_trace_queue`, yield them immediately: `yield {"type": "trace", "data": trace_event}`
- When `command_output_queue` receives the final output, yield it: `yield {"type": "output", "data": command_output}`
- Do NOT include `traces` array in the final `CommandOutput` (already streamed)

**Non-streaming (plain REST) clients:**
- Collect all events from `command_trace_queue` in-memory while waiting for output
- When `command_output_queue` receives the final output, attach collected `traces` array to `CommandOutput`
- Return the complete `CommandOutput` with embedded traces

**Implementation pattern:**
```python
async def invoke_agent(user_query: str, timeout_seconds: int = 60):
    client_supports_streaming = _check_accept_header()
    
    if client_supports_streaming and workflow_config.show_agent_traces:
        # Stream mode: yield traces as they arrive
        async for event in _execute_and_stream_traces(user_query, timeout_seconds):
            yield event
    else:
        # Non-stream mode: collect traces and return complete response
        return await _execute_and_collect_traces(user_query, timeout_seconds)
```

**Note:** `execute_command` uses the same pattern, leveraging `_execute_workflow_query` which already emits trace events to the queue.

### 11.6 FastAPI Alignment

- REST: `/invoke_agent` is synchronous and includes `traces` when produced. `/invoke_agent_stream` streams either NDJSON (Streamable HTTP style) or SSE, controlled by a REST-only `stream_format` set at REST `/initialize`.
- MCP: Tools stream via Streamable HTTP only (no SSE). MCP `initialize` has no `stream_format`.

### 11.6 Error Mapping

- Map concurrency errors to 409-equivalent MCP error.
- Map validation to 422; timeouts to 504.
- Include `code`, `message`, and `data` with `user_id`, `conversation_id` where relevant. Log stack traces.


## 12) Testing Strategy

Unit tests:
- Session lifecycle: initialize with/without `user_id`, per-session serialization (409 on concurrent turns), timeout behavior (504), and error mapping.
- Validation: 422 for malformed `execute_command`, feedback XOR rule (at least one non-null field).
- Discovery: `get_workflow_info` (no current_context), `get_commands` returns both text and structured metadata.

Integration tests:
- End-to-end: `initialize` → `invoke_agent` → assert `CommandOutput` shape and presence/absence of `traces` based on `SHOW_AGENT_TRACES`.
- Deterministic path: `initialize` → `invoke_assistant` → assert output.
- Fine-grained path: `initialize` → `get_commands` → `execute_command`.
- Conversations: `new_conversation` → `list_conversations` → `activate_conversation` → verify persistence and restoration.
- Feedback path: `post_feedback` after a turn, verify it is stored on the latest turn.


## 13) Future Enhancements (Non-Blocking)

- Add `plan-next-steps` prompt if clients request local planning assistance.
- Stream trace events incrementally over MCP if the transport provides a portable streaming interface for tool invocations.
- Add session TTL and eviction policies.
- Security hardening: workflow allow-list, per-user authn/z if exposed beyond trusted environments.


## 14) Alignment Notes

- Startup parity: Mirrors CLI runner (`fastworkflow/run/__main__.py`) for env loading, `fastworkflow.init(...)`, `ChatSession(run_as_agent=True)`, and `start_workflow(...)` sequencing.
- Context handling: Current context is accessible via the command `what_is_current_context` (listed by `get_commands`) and on every `CommandOutput.context`. It is intentionally not duplicated in `get_workflow_info`.
- No dynamic workflow loading and no admin/export tools over MCP.



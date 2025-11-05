# FastWorkflow FastAPI + MCP Service

HTTP + MCP interface for FastWorkflow workflows with synchronous and streaming execution.

## Overview

This service exposes FastWorkflow workflows as REST endpoints and as MCP tools, enabling clients to:
- Initialize workflow sessions per channel
- Submit natural language queries (agent mode)
- Execute deterministic commands (assistant mode)
- Perform explicit actions
- Manage conversations with persistent history
- Collect feedback on interactions

## Architecture

- **Session Management**: In-memory `ChannelSessionManager` with per-channel `ChatSession` instances
- **Persistence**: Rdict-backed conversation storage (one DB file per channel)
- **Execution**: Synchronous turn-based processing with queue-based communication
- **Tracing**: Traces are collected by default and included in synchronous responses or emitted incrementally during streaming
- **Streaming (REST)**: `/invoke_agent_stream` supports Streamable HTTP via NDJSON by default and SSE when requested in REST initialize
- **Streaming (MCP)**: MCP transport mounted at `/mcp`; tools stream partials via MCP transport by default or SSE when requested in MCP initialize

See [`docs/fastworkflow_fastapi_spec.md`](../../docs/fastworkflow_fastapi_spec.md) and [`docs/fastworkflow_fastapi_architecture.md`](../../docs/fastworkflow_fastapi_architecture.md) for complete specification and design.

## Running the Service

### Start Server (REST + MCP)

```bash
uvicorn services.run_fastapi.main:app --host 0.0.0.0 --port 8000

# MCP (auto-mounted via fastapi_mcp) will be available at `/mcp`.
```

### Access Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/

## Environment Variables

Configure in your environment (loaded at process startup via CLI args or env load):

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `SPEEDDICT_FOLDERNAME` | Base folder for workflow contexts and conversation storage | Yes | - |
| `--expect_encrypted_jwt` | Enable full JWT signature verification (pass flag to require signed tokens) | No | False (no verification by default) |

Notes:
- Conversation DBs are stored under `SPEEDDICT_FOLDERNAME/channel_conversations` (directory is auto-created).
- `/conversations` now accepts a `limit` query parameter (default `20`).
- Shutdown waits up to 30 seconds for active turns (hard-coded).

## Auth Modes

### Trusted Mode (default): No Signature Verification
When `--expect_encrypted_jwt` is NOT set (trusted environments), the service still creates and returns JWT tokens from `/initialize`, but signature verification is disabled. Clients must include `Authorization: Bearer <access_token>` on subsequent requests.

```bash
uvicorn services.run_fastapi.main:app --workflow_path /path/to/workflow
```

Notes (trusted mode):
- `/initialize` returns access/refresh tokens and, if a startup command/action is provided, also returns the startup `CommandOutput`.
- Subsequent endpoints require `Authorization: Bearer <access_token>`.
- Traces include `user_id` when available (from JWT `uid` claim).

### Secure Mode: Signed JWTs with Verification
For production deployments requiring full RSA signature verification:

```bash
uvicorn services.run_fastapi.main:app --workflow_path /path/to/workflow --expect_encrypted_jwt
```

**Secure mode** (with `--expect_encrypted_jwt` flag):
- `/initialize` issues access/refresh tokens. Subsequent endpoints require `Authorization: Bearer <token>`.
- JWT claims include `sub` (channel_id) and `uid` (user_id when provided).
- Tokens are verified (signature, expiry, audience/issuer). Invalid or expired tokens are rejected.
- Recommended for production deployments in untrusted environments

### Token Access in Workflow Context

In secure mode, JWT tokens are passed to workflows via `workflow_context['http_bearer_token']` to support authenticated upstream calls. In trusted mode, tokens are not created/returned and `http_bearer_token` is absent.

**Important notes:**
- The token is **only available to authenticated endpoints** (those using `get_session_and_ensure_runtime` dependency)
- The token is stored in the workflow context dictionary under the key `http_bearer_token`
- Token is **automatically updated** on every authenticated request, ensuring workflows always have the current valid token
- Token expiration is **automatically verified** by `verify_token()` in both secure mode (`--expect_encrypted_jwt` flag) and trusted network mode
- In secure mode: Full cryptographic signature verification + expiration checking
- In trusted network mode: Expiration checking is performed (signature verification disabled)
- Tokens should be treated as sensitive data and handled securely in workflows
- The `/initialize` endpoint is unauthenticated and does NOT provide a token to the workflow context; tokens are only available after calling `/initialize` and using the returned token in subsequent requests

**Example usage in workflow:**

```python
# In workflow code
workflow_context = self._context
bearer_token = workflow_context.get('http_bearer_token')

# Use token for API calls
headers = {"Authorization": f"Bearer {bearer_token}"}
response = requests.get("https://api.example.com/data", headers=headers)
```

## API Endpoints (REST)

### `POST /initialize`
Initialize a session for a channel. Workflow configuration is loaded at server startup from CLI args/env.

**Request:**
```json
{
  "channel_id": "channel-123",
  "user_id": "user-9",
  "stream_format": "ndjson",
  "startup_command": "load_workflow ...",
  "startup_action": {
    "command_name": "find_orders",
    "parameters": {"channel_id": 42}
  }
}
```

**Rules:**
- `channel_id` is required.
- Exactly one of `startup_command` or `startup_action` may be provided (or neither).
- If startup is provided, `user_id` is required and recorded in the initial trace.
- `stream_format` controls REST streaming format for `/invoke_agent_stream` (NDJSON default, SSE optional).

**Response:**
```json
{
  "access_token": "eyJhbGci...",
  "refresh_token": "eyJhbGci...",
  "token_type": "bearer",
  "expires_in": 3600,
  "startup_output": { /* CommandOutput, present only if startup was executed */ }
}
```

### `POST /invoke_agent`
Submit a natural language query for agentic processing.

**Headers:**
- `Authorization: Bearer <access_token>` (JWT contains `sub` for channel_id and optional `uid` for user_id)

**Request:**
```json
{
  "user_query": "find orders for channel 42",
  "timeout_seconds": 60
}
```

**Response:**
```json
{
  "command_responses": [
    {
      "response": "Found 3 orders for channel 42",
      "success": true,
      "artifacts": {},
      "next_actions": [],
      "recommendations": []
    }
  ],
  "workflow_name": "default_workflow",
  "context": "Order management context",
  "command_name": "find_orders",
  "command_parameters": "channel_id=42",
  "traces": [...]
}
```

### `POST /invoke_assistant`
Submit a query for deterministic/assistant execution (no planning).

Same request/response format as `/invoke_agent`.

### `POST /perform_action`
Execute a specific workflow action directly (bypasses parameter extraction).

**Headers:**
- `Authorization: Bearer <access_token>` (JWT contains `sub` for channel_id and optional `uid` for user_id)

**Request:**
```json
{
  "action": {
    "command_name": "find_orders",
    "parameters": {"channel_id": 42}
  },
  "timeout_seconds": 60
}
```

**Response:**
Same format as `/invoke_agent` (CommandOutput with traces).

### `POST /invoke_agent_stream`
Stream trace events and final `CommandOutput` via Streamable HTTP:
- NDJSON (default; `Content-Type: application/x-ndjson`)
- SSE (when REST `stream_format` is set to `sse`; `Content-Type: text/event-stream`)

**Headers:** Same as `/invoke_agent`.

## Conversation Management (REST)

### `POST /new_conversation`
Persist current conversation and start a new one.

**Request:**
```json
{}
```

**Response:**
```json
{
  "status": "ok"
}
```

### `GET /conversations?channel_id={channel_id}&limit=20`
List conversations for a channel (most recent first). `limit` is optional; defaults to `20`.

**Response:**
```json
[
  {
    "conversation_id": 1,
    "topic": "Order Management",
    "summary": "...",
    "created_at": 1234567890000,
    "updated_at": 1234567890000
  }
]
```

### `POST /activate_conversation`
Switch to a different conversation by ID.

**Request:**
```json
{
  "conversation_id": 1
}
```

### `POST /post_feedback`
Attach feedback to the latest turn.

**Request:**
```json
{
  "binary_or_numeric_score": true,
  "nl_feedback": "Helpful response"
}
```

At least one of `binary_or_numeric_score` or `nl_feedback` must be provided.

## Admin Endpoints (REST-only; not exposed via MCP)

### `POST /admin/dump_all_conversations`
Export all conversations to JSONL.

**Request:**
```json
{
  "output_folder": "/path/to/export"
}
```

**Response:**
```json
{
  "file_path": "/path/to/export/all_conversations_1234567890.jsonl"
}
```

## Usage Examples

### REST initialize
```python
import requests

resp = requests.post("http://localhost:8000/initialize", json={
    "channel_id": "alice",
    "stream_format": "ndjson"
})
print(resp.json())  # {"access_token": "...", "refresh_token": "...", "token_type": "bearer", "expires_in": 3600}
```

### REST sync invoke
```python
# First get a token from /initialize
init_resp = requests.post("http://localhost:8000/initialize", json={
    "channel_id": "alice",
    "stream_format": "ndjson"
})
token_data = init_resp.json()
access_token = token_data["access_token"]

# Then use the token for authenticated requests
resp = requests.post("http://localhost:8000/invoke_agent", 
    headers={"Authorization": f"Bearer {access_token}"},
    json={
        "channel_query": "list all channels",
        "timeout_seconds": 30
    }
)
result = resp.json()
print(result["command_responses"])
print(result.get("traces"))
```

### MCP Quickstart
- Mount is available at `/mcp` (auto-exposed by fastapi_mcp).
- In secure mode, MCP clients use pre-configured long-lived access tokens (generated via `/admin/generate_mcp_token`).
- In trusted mode, clients must send `Authorization: Bearer <token>`; the JWT `sub` claim carries the `channel_id`.
- No need for initialize or refresh_token tools in MCP context.

MCP invoke agent (streaming):

```bash
# NDJSON (default for MCP)
curl -N -X POST http://localhost:8000/mcp/invoke_agent \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/x-ndjson' \
  -H 'Authorization: Bearer <your-mcp-token>' \
  -d '{"channel_query":"find orders for channel 42","timeout_seconds":60}'

# SSE (if stream_format was set to "sse" during MCP setup)
curl -N -X POST http://localhost:8000/mcp/invoke_agent \
  -H 'Content-Type: application/json' \
  -H 'Accept: text/event-stream' \
  -H 'Authorization: Bearer <your-mcp-token>' \
  -d '{"channel_query":"find orders for channel 42","timeout_seconds":60}'
```

### MCP Prompts
The MCP mount registers two prompts for client discovery:
- `format-command(intent, metadata)`
- `clarify-params(error_message, metadata)`

## Testing

See [`tests/`](../../tests/) for unit and integration tests.

**Key test scenarios:**
- SessionManager concurrency and lifecycle
- Env loading from files only
- CLI argument validation (startup_command vs startup_action)
- Timeout behavior (504 on no output)
- Conversation persistence and listing
- Feedback attachment
 - Initialize with startup command/action returns `startup_output` and records conversation
 - Both trusted mode return tokens; secure mode tokens are encrypted
 - Traces include `user_id` and `raw_command`

## Future Enhancements

- Session TTL and eviction policy
- Richer observability and structured logging
- Security: workflow path allow-list, authn/z
- Conversation history restoration on session resume


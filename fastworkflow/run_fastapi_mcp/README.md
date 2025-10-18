# FastWorkflow FastAPI + MCP Service

HTTP + MCP interface for FastWorkflow workflows with synchronous and streaming execution.

## Overview

This service exposes FastWorkflow workflows as REST endpoints and as MCP tools, enabling clients to:
- Initialize workflow sessions per user
- Submit natural language queries (agent mode)
- Execute deterministic commands (assistant mode)
- Perform explicit actions
- Manage conversations with persistent history
- Collect feedback on interactions

## Architecture

- **Session Management**: In-memory `UserSessionManager` with per-user `ChatSession` instances
- **Persistence**: Rdict-backed conversation storage (one DB file per user)
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

Notes:
- Conversation DBs are stored under `SPEEDDICT_FOLDERNAME/user_conversations` (directory is auto-created).
- `/conversations` now accepts a `limit` query parameter (default `20`).
- Shutdown waits up to 30 seconds for active turns (hard-coded).

## API Endpoints (REST)

### `POST /initialize`
Initialize a session for a user. Workflow configuration is loaded at server startup from CLI args/env.

**Request:**
```json
{
  "user_id": "user-123",
  "stream_format": "ndjson"  // optional: "ndjson" | "sse" (default "ndjson")
}
```

**Important Notes:**
- `stream_format` controls REST streaming format for `/invoke_agent_stream` (NDJSON default, SSE optional).
- Workflow configuration is loaded at server startup from CLI args/env, not from the request.

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}

```

### `POST /invoke_agent`
Submit a natural language query for agentic processing.

**Request:**
```json
{
  "user_id": "user-123",
  "user_query": "find orders for user 42",
  "timeout_seconds": 60
}
```

**Response:**
```json
{
  "command_responses": [
    {
      "response": "Found 3 orders for user 42",
      "success": true,
      "artifacts": {},
      "next_actions": [],
      "recommendations": []
    }
  ],
  "workflow_name": "default_workflow",
  "context": "Order management context",
  "command_name": "find_orders",
  "command_parameters": "user_id=42",
  "traces": [...]
}
```

### `POST /invoke_assistant`
Submit a query for deterministic/assistant execution (no planning).

Same request/response format as `/invoke_agent`.

### `POST /perform_action`
Execute a specific workflow action directly (bypasses parameter extraction).

**Request:**
```json
{
  "action": {
    "command_name": "find_orders",
    "parameters": {"user_id": 42}
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

### `GET /conversations?user_id={user_id}&limit=20`
List conversations for a user (most recent first). `limit` is optional; defaults to `20`.

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
    "user_id": "alice",
    "stream_format": "ndjson"
})
print(resp.json())  # {"access_token": "...", "refresh_token": "...", "token_type": "bearer", "expires_in": 3600}
```

### REST sync invoke
```python
# First get a token from /initialize
init_resp = requests.post("http://localhost:8000/initialize", json={
    "user_id": "alice",
    "stream_format": "ndjson"
})
token_data = init_resp.json()
access_token = token_data["access_token"]

# Then use the token for authenticated requests
resp = requests.post("http://localhost:8000/invoke_agent", 
    headers={"Authorization": f"Bearer {access_token}"},
    json={
        "user_query": "list all users",
        "timeout_seconds": 30
    }
)
result = resp.json()
print(result["command_responses"])
print(result.get("traces"))
```

### MCP Quickstart
- Mount is available at `/mcp` (auto-exposed by fastapi_mcp).
- MCP clients use pre-configured long-lived access tokens (generated via `/admin/generate_mcp_token`).
- No need for initialize or refresh_token tools in MCP context.

MCP invoke agent (streaming):

```bash
# NDJSON (default for MCP)
curl -N -X POST http://localhost:8000/mcp/invoke_agent \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/x-ndjson' \
  -H 'Authorization: Bearer <your-mcp-token>' \
  -d '{"user_query":"find orders for user 42","timeout_seconds":60}'

# SSE (if stream_format was set to "sse" during MCP setup)
curl -N -X POST http://localhost:8000/mcp/invoke_agent \
  -H 'Content-Type: application/json' \
  -H 'Accept: text/event-stream' \
  -H 'Authorization: Bearer <your-mcp-token>' \
  -d '{"user_query":"find orders for user 42","timeout_seconds":60}'
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

## Future Enhancements

- Session TTL and eviction policy
- Richer observability and structured logging
- Security: workflow path allow-list, authn/z
- Conversation history restoration on session resume


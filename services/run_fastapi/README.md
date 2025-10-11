# FastWorkflow FastAPI Service

HTTP interface for FastWorkflow workflows with synchronous execution.

## Overview

This service exposes FastWorkflow workflows as REST endpoints, enabling clients to:
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
- **Tracing**: Optional agent trace collection included in responses

See [`docs/fastworkflow_fastapi_spec.md`](../../docs/fastworkflow_fastapi_spec.md) and [`docs/fastworkflow_fastapi_architecture.md`](../../docs/fastworkflow_fastapi_architecture.md) for complete specification and design.

## Running the Service

### Start Server

```bash
uvicorn services.run_fastapi.main:app --host 0.0.0.0 --port 8000
```

### Access Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/

## Environment Variables

Configure in your environment file (loaded per-user during initialization):

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `SPEEDDICT_FOLDERNAME` | Base folder for workflow contexts | Yes | - |
| `USER_CONVERSATIONS_SPEEDDICT_FOLDERNAME` | Base folder for conversation DBs | No | `___user_conversations` |
| `FASTWORKFLOW_CONVERSATIONS_LIST_LIMIT` | Max conversations returned by `/conversations` | No | `50` |
| `FASTWORKFLOW_SHUTDOWN_MAX_WAIT_SECONDS` | Max wait for active turns during shutdown | No | `30` |

## API Endpoints

### Core Workflow Endpoints

#### `POST /initialize`
Initialize a session for a user with a specific workflow.

**Request:**
```json
{
  "user_id": "user-123",
  "workflow_path": "/abs/path/to/workflow",
  "env_file_path": "/abs/path/to/.env",
  "passwords_file_path": "/abs/path/to/passwords.env",
  "context": {},
  "startup_command": "",
  "startup_action": null,
  "show_agent_traces": true,
  "conversation_id": null
}
```

**Important Notes:**
- `startup_action`: If provided, must include `command_name` field. Don't send empty object `{}` - omit the field or use `null`.
- `conversation_id`: Optional. If `null` or omitted, restores last conversation. If no conversations exist, starts new. Provide specific ID to restore that conversation.
- `startup_command` and `startup_action` are mutually exclusive.

**Response:**
```json
{
  "user_id": "user-123"
}
```

#### `POST /invoke_agent`
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
  "command_responses": [...],
  "workflow_name": "...",
  "context": "...",
  "command_name": "...",
  "command_parameters": "...",
  "success": true,
  "traces": [...]  // if show_agent_traces=true
}
```

#### `POST /invoke_assistant`
Submit a query for deterministic/assistant execution (no planning).

Same request/response format as `/invoke_agent`.

#### `POST /perform_action`
Execute a specific workflow action directly.

**Request:**
```json
{
  "user_id": "user-123",
  "action": {
    "command_name": "User/get_details",
    "parameters": {"user_id": "u-42"}
  },
  "timeout_seconds": 60
}
```

### Conversation Management

#### `POST /new_conversation`
Persist current conversation and start a new one.

**Request:**
```json
{
  "user_id": "user-123"
}
```

**Response:**
```json
{
  "status": "ok"
}
```

#### `GET /conversations?user_id={user_id}`
List conversations for a user (most recent first).

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

#### `POST /activate_conversation`
Switch to a different conversation by ID.

**Request:**
```json
{
  "user_id": "user-123",
  "conversation_id": 1
}
```

#### `POST /post_feedback`
Attach feedback to the latest turn.

**Request:**
```json
{
  "user_id": "user-123",
  "binary_or_numeric_score": true,
  "nl_feedback": "Helpful response"
}
```

At least one of `binary_or_numeric_score` or `nl_feedback` must be provided.

### Admin Endpoints

#### `POST /admin/dump_all_conversations`
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

## Error Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad request (e.g., both startup_command and startup_action provided) |
| 404 | User session or conversation not found |
| 409 | Concurrent turn already in progress for user |
| 422 | Validation error (invalid paths, missing fields, etc.) |
| 500 | Internal server error |
| 504 | Command execution timeout |

## Usage Example

### Python Client

```python
import requests

# Initialize session (minimal)
resp = requests.post("http://localhost:8000/initialize", json={
    "user_id": "alice",
    "workflow_path": "/path/to/my_workflow",
    "env_file_path": "/path/to/.env",
    "passwords_file_path": "/path/to/passwords.env"
})
print(resp.json())  # {"user_id": "alice"}

# Initialize with all options
resp = requests.post("http://localhost:8000/initialize", json={
    "user_id": "alice",
    "workflow_path": "/path/to/my_workflow",
    "env_file_path": "/path/to/.env",
    "passwords_file_path": "/path/to/passwords.env",
    "context": {"key": "value"},
    "startup_command": "/add_two_numbers first_num=5 second_num=3",
    "show_agent_traces": True,
    "conversation_id": None  # Restore last conversation (or omit this field)
})
print(resp.json())  # {"user_id": "alice"}

# Submit query
resp = requests.post("http://localhost:8000/invoke_agent", json={
    "user_id": "alice",
    "user_query": "list all users",
    "timeout_seconds": 30
})
result = resp.json()
print(result["command_responses"])
print(result["traces"])  # Agent execution trace

# Start new conversation
resp = requests.post("http://localhost:8000/new_conversation", json={
    "user_id": "alice"
})

# List conversations
resp = requests.get("http://localhost:8000/conversations?user_id=alice")
print(resp.json())
```

## Common Mistakes & Troubleshooting

### ❌ Don't Send String "null"
```json
{
  "conversation_id": "null"  // WRONG - string "null"
}
```

✅ **Correct**: Use JSON `null` or omit the field
```json
{
  "conversation_id": null  // Correct - JSON null
}
// OR simply omit it:
{
  // conversation_id not included - this is fine
}
```

### ❌ Don't Send Empty startup_action
```json
{
  "startup_action": {}  // WRONG - empty object
}
```

✅ **Correct**: Omit it or use `null`, or provide complete action
```json
{
  "startup_action": null  // Correct
}
// OR
{
  "startup_action": {  // Correct - complete action
    "command_name": "add_two_numbers",
    "parameters": {"first_num": 10, "second_num": 20}
  }
}
```

### ❌ Don't Provide Both startup_command and startup_action
```json
{
  "startup_command": "/some_command",
  "startup_action": {"command_name": "other_command"}  // WRONG - mutually exclusive
}
```

✅ **Correct**: Use only one or neither
```json
{
  "startup_command": "/some_command"  // Correct
}
// OR
{
  "startup_action": {"command_name": "some_command", "parameters": {}}  // Correct
}
```

## Behavior Notes

### Leading Slashes
- `/invoke_agent`: Leading `/` characters are stripped (e.g., `/find users` → `find users`)
- `/invoke_assistant`: Server prepends `/` internally to force deterministic mode

### Conversation Lifecycle
1. On `POST /initialize`, resume last conversation if available
2. Submit queries via `/invoke_agent` or `/invoke_assistant`
3. Optional: `POST /post_feedback` after each turn
4. Call `POST /new_conversation` to persist and rotate
5. Use `GET /conversations` to list past conversations
6. Use `POST /activate_conversation` to resume a prior conversation

### Topic Uniqueness
Topics are guaranteed unique per user via case-insensitive and whitespace-insensitive comparison. If a collision occurs, an incrementing integer is appended (e.g., "Order Management 2").

## Testing

See [`tests/`](../../tests/) for unit and integration tests.

**Key test scenarios:**
- SessionManager concurrency and lifecycle
- Env loading from files only
- XOR validation (startup_command vs startup_action)
- Timeout behavior (504 on no output)
- Conversation persistence and listing
- Feedback attachment

## Future Enhancements

- Server-side streaming of agent traces (SSE/WebSockets)
- Session TTL and eviction policy
- Richer observability and structured logging
- Security: workflow path allow-list, authn/z
- Conversation history restoration on session resume

## References

- [FastWorkflow FastAPI Specification](../../docs/fastworkflow_fastapi_spec.md)
- [FastWorkflow FastAPI Architecture](../../docs/fastworkflow_fastapi_architecture.md)
- [FastWorkflow Core](../)


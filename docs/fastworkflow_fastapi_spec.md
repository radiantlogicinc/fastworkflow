### FastWorkflow FastAPI Service — Specification

#### 1. Overview
- **Goal**: Expose FastWorkflow workflows as a FastAPI web service, enabling clients to initialize for a given workflow (per user), then interact in agent mode (forced). If a query starts with `/`, all leading slashes are stripped before processing. The service supports explicit actions, resetting conversations, listing conversations, dumping all conversations to JSONL, and posting feedback. `invoke_agent` returns a synchronous `CommandOutput` and, when enabled, includes collected trace events in the final response. Streaming is supported via NDJSON or SSE at `/invoke_agent_stream`, and MCP tools map to the same NDJSON-based streaming implementation.
- **Source parity**: Behavior mirrors the CLI runner in `fastworkflow/run/__main__.py` while replacing its interactive loop with synchronous and streaming HTTP endpoints.

#### 2. Non‑Goals
- No UI; only REST with OpenAPI/Swagger at `/docs`.
- No WebSocket support. For streaming, use NDJSON or SSE at `/invoke_agent_stream`. MCP Streamable HTTP is mounted on the same FastAPI app and maps to the same server-side streaming (NDJSON).

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
- Trace events are collected and included in responses by default for REST streaming and collected into the synchronous response for `/invoke_agent`.

#### 5. User and Conversation Lifecycle
1) Client calls `POST /initialize` with workflow location and setup options plus a `user_id`.
2) Server:
   - Loads environment from `env_file_path` and `passwords_file_path` only.
   - Calls `fastworkflow.init(env_vars=<file_based_env_dict>)`.
   - Creates a new `ChatSession` with `run_as=RUN_AS_AGENT` (forced agent mode), bound to `user_id` and a current conversation (internal id only).
   - If the user has a prior conversation, resume it by default (restore last conversation history); otherwise create a new conversation.
   - Starts the workflow via `chat_session.start_workflow(...)` with provided context/startup parameters.
   - Stores runtime in `UserSessionManager` and returns a JWT TokenResponse containing `access_token`, `refresh_token`, `token_type`, `expires_in`, `user_id`, and `workflow_info`.
3) Client uses the JWT access token with `/invoke_agent`, `/invoke_assistant`, `/perform_action`, or `/new_conversation`. Additional endpoints: `/conversations`. Admin-only endpoint: `/admin/dump_all_conversations`.
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
  "conversation_id": null,
  "stream_format": "ndjson"
}
```
- Notes:
  - Workflow path, env, passwords, startup command/action, and initial context are loaded at server startup from CLI args/environment (ops-managed), not from the initialize call.
  - Service always runs in agent mode; assistant behavior is exposed via `/invoke_assistant`.
  - `conversation_id`: Optional. If `null` or omitted, restores last conversation (or starts new if none exist). Provide specific ID to restore that conversation.
  - `stream_format`: Optional preference for `/invoke_agent_stream` response format. Supported: `ndjson` (default) or `sse`. This setting is REST-only and does not apply to MCP.
- Response (TokenResponse):
```json
{
  "access_token": "<JWT>",
  "refresh_token": "<JWT>",
  "token_type": "bearer",
  "expires_in": 3600
}
```
- Notes:
  - `user_id` is embedded in the JWT token itself (as the "sub" claim) and can be extracted by decoding the token
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
- Purpose: Submit a natural language query to an agentic session for a user (synchronous response). Leading `/` characters are permitted and stripped for compatibility; assistant semantics remain exclusive to `/invoke_assistant`.
- Request:
```json
{ "user_query": "find orders for user 42", "timeout_seconds": 60 }
```
- Behavior:
  - Validate the user session exists. Agent mode is always enabled.
  - If `user_query` begins with `/`, strip all leading slashes before processing (compatibility path).
  - When the turn completes, return a `CommandOutput` JSON including collected `traces`.
- Response: `CommandOutput` (JSON).
- Errors:
  - 404 user not found
  - 409 concurrent turn already in progress for this user
  - 504 turn timed out (no output on queue within `timeout_seconds`)
  - 500 unexpected error

4) POST `/invoke_agent_stream`
- Purpose: Submit a natural language query to an agentic session and stream trace events in real-time, followed by the final `CommandOutput`. Leading `/` characters are permitted and stripped.
- Request:
```json
{ "user_query": "find orders for user 42", "timeout_seconds": 60 }
```
- Behavior:
  - Validate the user session exists. Agent mode is always enabled.
  - If `user_query` begins with `/`, strip all leading slashes before processing.
  - Emit streaming records as they are available:
    - NDJSON: `{ "type": "trace", "data": <trace_json> }` (multiple), then `{ "type": "output", "data": <CommandOutput_json> }` (final)
    - SSE: `event: trace` (multiple) and final `event: output` with JSON payloads
  - Only the final output record is streamed if no traces were produced.
- Response: HTTP 200 with `Content-Type: application/x-ndjson` (NDJSON) or `text/event-stream` (SSE).
- Errors:
  - 404 user not found
  - 409 concurrent turn already in progress for this user
  - 504 turn timed out (no output within `timeout_seconds`)
  - On error, send a terminal record `{ "type": "error", "data": { "detail": "..." } }` then close the connection

5) POST `/invoke_assistant`
- Purpose: Deterministic/assistant invocation for a user. The server accepts plain queries; clients need not prefix `/`.
- Request:
```json
{ "user_query": "load_workflow file='...'" }
```
- Behavior: Same execution path as agent, but uses assistant path (no planning). Returns `CommandOutput`.
- Response: `CommandOutput`.
- Errors: as above.

6) POST `/perform_action`
- Purpose: Execute a specific workflow action chosen by the client (e.g., from `next_actions`).
- Request:
```json
{ "action": { "command_name": "User/get_details", "arguments": { "user_id": "u-42" } }, "timeout_seconds": 60 }
```
- Behavior:
  - Validate session exists.
  - Invoke through the same single‑turn path used for NL queries, but bypass parameter extraction (directly execute the provided `Action`).
  - Wait for `CommandOutput` (or timeout) and return it.
- Response: `CommandOutput`.
- Errors: 404/409/504/500 as above; 422 invalid action shape.

7) POST `/new_conversation`
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

8) POST `/post_feedback`
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

9) GET `/` (root)
- Simple HTML page with a link to `/docs`. Serves also as a health check (no dedicated `/healthz`).

#### 7. Data Models

Pydantic model sketches (for reference; actual code will import FastWorkflow types where available):

```python
class InitializationRequest(BaseModel):
    user_id: str | None = None
    conversation_id: int | None = None

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
    traces: list[dict[str, Any]] | None = None
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

- Conversations are stored under `SPEEDDICT_FOLDERNAME/user_conversations`; one Rdict DB file per user (`<user_id>.rdb`).
- Per-user DB schema (keys/values):
  - Key: `meta` → { "last_conversation_id": int }
  - Key: `conv:<id>` → {
      "topic": str,
      "summary": str,
      "created_at": int,
      "updated_at": int,
      "turns": [ { "conversation summary": str, "conversation_traces": str (JSON), "feedback": { "binary_or_numeric_score": bool|float|null, "nl_feedback": str|null, "timestamp": int } | null } ]
    }
- Functional constraint: one active conversation per user to avoid write concurrency.
- `/conversations` accepts `limit` (default `20`) controlling the max conversations returned (latest N by `updated_at`).
- Shutdown waits up to 30 seconds for active turns before persistence.
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
  "sub": "john_doe",           # subject (user_id)
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

def create_access_token(user_id: str, expires_delta: timedelta) -> str:
    """Create and sign JWT access token with RS256"""

def create_refresh_token(user_id: str, expires_delta: timedelta) -> str:
    """Create and sign JWT refresh token with RS256"""

def verify_and_decode_token(token: str, token_type: str = "access") -> dict:
    """Verify signature, check expiration, and decode token claims"""
```

##### 15.8 Data Model Changes

**New Models in `utils.py`:**
```python
class SessionData(BaseModel):
    """Decoded JWT session data"""
    user_id: str
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
Request: {"user_id": "john_doe"}
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
Request: {"user_id": "john_doe"}
Response: {
  "access_token": "eyJhbGci...",
  "refresh_token": "eyJhbGci...",
  "token_type": "Bearer",
  "expires_in": 3600
}
# Notes:
# - user_id is in the JWT's "sub" claim
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
- The `user_id` is embedded in the JWT (sub claim), not returned separately

**2. All authenticated endpoints (Modified)**
- Change dependency from: `session_id: int = Depends(get_session_id_from_header)`
- To: `session: SessionData = Depends(get_session_from_jwt)`
- Use `session.user_id` for session lookups and logging

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
            user_id=claims["sub"],
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
└── mcp_specific.py         # NO CHANGE (still uses user_id directly)

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
  "user_id": "claude_desktop_user",
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
- Tokens are tied to a specific `user_id`
- All MCP tool calls are authenticated and tracked per user
- Token expiration can be customized (e.g., 30 days, 180 days, etc.)

##### 15.21 Future Enhancements (Not in Initial Implementation)

**Token Revocation:**
- Maintain Redis-based revocation list
- Store token JTI when user logs out
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



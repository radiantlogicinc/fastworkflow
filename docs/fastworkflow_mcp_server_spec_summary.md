# FastWorkflow MCP Server Specification - Summary of Key Decisions

This document summarizes the key architectural decisions and answers to questions raised during specification development.

## 1. Session Lifecycle - ChatSession Per User, Not Per Server

**Question:** Where should ChatSession be created?

**Answer:** 
- ❌ **NOT at server startup** (one shared ChatSession)
- ✅ **Per user during `initialize()` call** (isolated ChatSession per user)

**Rationale:**
- Each user needs isolated queues (`user_message_queue`, `command_output_queue`, `command_trace_queue`)
- One ChatSession per process would cause message/trace collisions between users
- Workflow config (path, context, startup command) is stored at server startup and reused when creating each user's ChatSession

**Implementation:**
```python
# Server startup: store config, don't create ChatSession
def _server_startup():
    fastworkflow.init(env_vars=env_vars)
    workflow_config.workflow_path = os.environ["WORKFLOW_PATH"]
    workflow_config.context = json.loads(os.environ.get("CONTEXT", "{}"))
    # ... store other config

# Per-user initialization: create ChatSession
def initialize(user_id: str | None = None):
    chat_session = fastworkflow.ChatSession(run_as_agent=True)
    chat_session.start_workflow(
        workflow_config.workflow_path,
        workflow_context=workflow_config.context,
        # ... use stored config
    )
    # Store chat_session in session manager
```

## 2. Incremental Streaming - Confirmed and Detailed

**Question:** Does MCP streamable-http support incremental streaming? How do clients signal support?

**Answer:**
✅ **Yes, via MCP Streamable HTTP; optionally support SSE too**

**Client Selection (MCP):**
- Client sets `stream_format` in MCP `initialize` to `ndjson` (Streamable HTTP) or `sse`.
  - Default: `ndjson`.

**Server Behavior:**

**If client supports streaming:**
```python
@mcp.tool()
async def invoke_agent(user_query: str, timeout_seconds: int = 60):
    # Yield trace events as they occur
    async for trace in _drain_trace_queue_async():
        yield {"type": "trace", "data": trace}
    
    # Yield final output WITHOUT traces array (already streamed)
    yield {"type": "output", "data": command_output}
```

**If client does NOT support streaming:**
```python
@mcp.tool()
def invoke_agent(user_query: str, timeout_seconds: int = 60):
    # Collect all traces in memory
    traces = _collect_all_traces()
    
    # Return complete output WITH traces array
    return CommandOutput(..., traces=traces)
```

**Key Insight:** Streaming clients get real-time trace events and a lighter final response. Non-streaming clients get everything in one shot.

## 3. Conversation Generation - Removed from Prompts

**Question:** Why aren't `name-conversation` and `summarize-conversation` listed in the prompts section?

**Answer:**
❌ **Not needed** - FastWorkflow handles this automatically

**Rationale:**
- FastWorkflow uses `dspy.ChainOfThought()` with configured LLM (`LLM_CONVERSATION_STORE`) to generate topics and summaries
- This happens server-side during `new_conversation` tool execution
- Clients don't need prompts to guide this process
- Generation is synchronous and blocks conversation rotation if it fails (fail-safe)

**Removed from spec:**
- `name-conversation` prompt
- `summarize-conversation` prompt

**Kept in spec:**
- `format-command` - helps clients format XML-tagged parameters
- `clarify-params` - helps clients compose clarification questions

## 4. Streaming and Traces in CommandOutput

**Question:** If the client supports streaming, why would traces be in CommandOutput?

**Answer:**
✅ **They wouldn't be!**

**Clarification:**

**Streaming client flow (MCP):**
1. MCP exposes the REST streaming endpoint (`/invoke_agent_stream`) as the `invoke_agent` tool.
   - Streaming uses NDJSON events: `{"type":"trace"}` partials then final `{"type":"output"}`.
   - SSE is also supported by REST; MCP clients should prefer NDJSON.

**Non-streaming client flow:**
1. Client sends request with `Accept: application/json`
2. Server collects all traces in memory
3. Server returns: `{"success": true, ..., "traces": [...]}`

**Benefits of streaming:**
- Real-time feedback (users see progress)
- Lower memory footprint (don't buffer all traces)
- Lighter final response (traces already delivered)

**Benefits of non-streaming:**
- Simpler client implementation
- Single atomic response
- Better for testing/debugging

## 5. Tool Surface Decisions

### Included Tools:
- ✅ `invoke_agent` - agentic turn with planning (streaming)
- ✅ `invoke_assistant` - deterministic turn without planning
- ✅ `new_conversation` - persist and rotate conversation
- ✅ `get_all_conversations` - view conversation history
- ✅ `activate_conversation` - switch conversations
- ✅ `post_feedback` - attach feedback to latest turn

### Excluded:
- ❌ `rest_initialize` - MCP clients use pre-configured long-lived tokens (via /admin/generate_mcp_token)
- ❌ `perform_action` - low-level; use invoke_agent/invoke_assistant instead
- ❌ `rest_invoke_agent` - non-streaming; use invoke_agent instead
- ❌ `refresh_token` - not needed for MCP (long-lived tokens)
- ❌ `dump_all_conversations` - admin-only, not for MCP clients
- ❌ `generate_mcp_token` - admin-only token generation

### MCP Client Setup:
1. Admin generates long-lived token: `POST /admin/generate_mcp_token {"user_id": "mcp_client", "expires_days": 365}`
2. Admin configures token in MCP client settings (e.g., Claude Desktop's `mcp.json`)
3. MCP client uses token in Authorization header for all tool calls
4. Token lasts 1 year by default (no refresh needed)

### Prompts:
- ✅ `format-command` - format commands with XML tags
- ✅ `clarify-params` - compose clarification questions
- ❌ `name-conversation` - handled by FastWorkflow
- ❌ `summarize-conversation` - handled by FastWorkflow
- ❌ `plan-next-steps` - redundant with `invoke_agent`

## 6. Key Implementation Notes

### Server Startup Sequence:
1. Load env files (`ENV_FILE_PATH`, `PASSWORDS_FILE_PATH`)
2. Call `fastworkflow.init(env_vars=env_vars)`
3. **Store** workflow config (don't create ChatSession yet)
4. Initialize empty `UserSessionManager`

### Per-User Initialize Sequence:
1. Receive `user_id` from client
2. Create new `ChatSession(run_as_agent=True)` for this user
3. Call `chat_session.start_workflow(...)` with stored config
4. Restore or create conversation for this user
5. Store session in manager keyed by MCP session ID

### Streaming Implementation:
- Use async generators for tools that support streaming
- Check `Accept` header to determine client capability
- Yield events for streaming clients
- Return complete response for non-streaming clients

### Error Handling:
- 404: Session/user/conversation not found
- 409: Concurrent turn in progress
- 422: Validation failures
- 504: Timeout waiting for output
- 500: Unexpected errors (always log with stack trace)

## 7. Testing Strategy

### Unit Tests:
- Session lifecycle (initialize with/without user_id)
- Concurrent turn rejection (409)
- Timeout behavior (504)
- Streaming vs non-streaming response formats
- Validation errors (422)

### Integration Tests:
- Full flow: initialize → get_workflow_info → get_commands → invoke_agent
- Streaming: verify trace events arrive before final output
- Non-streaming: verify traces in final CommandOutput
- Conversations: new → list → activate → verify persistence
- Feedback: attach feedback to latest turn

## 8. References

**Full specification:** [fastworkflow_mcp_server_spec.md](./fastworkflow_mcp_server_spec.md)

**Related specs:**
- [fastworkflow_fastapi_spec.md](./fastworkflow_fastapi_spec.md) - REST API counterpart
- FastWorkflow CLI runner: `fastworkflow/run/__main__.py`
- Agent implementation: `fastworkflow/workflow_agent.py`

---

**Last Updated:** Based on feedback addressing:
- Session lifecycle (ChatSession per user)
- Streaming confirmation and details
- Conversation generation (automatic)
- Streaming behavior and CommandOutput traces


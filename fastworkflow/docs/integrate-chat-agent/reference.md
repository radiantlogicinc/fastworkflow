# fastWorkflow Chat Integration Reference

Detailed contracts for the FastAPI service (`fastworkflow.run_fastapi_mcp`) and the workflow file
formats. Read this when wiring the chat UI to the backend or fixing generated command files.

## Hosting the service

```bash
python -m fastworkflow.run_fastapi_mcp \
  --workflow_path <workflow_dir> \
  --env_file_path <workflow_dir>/fastworkflow.env \
  --passwords_file_path <workflow_dir>/fastworkflow.passwords.env \
  --port 8000
```

Other CLI flags: `--host` (default `0.0.0.0`), `--context` (JSON string), `--startup_command`,
`--startup_action` (JSON file path), `--expect_encrypted_jwt` (enable JWT signature verification;
off by default for trusted networks). Requires `pip install "fastworkflow[server]"`.

Interactive docs are served at `/docs` (Swagger) and `/redoc`. CORS is open (`*`) by default —
tighten `allow_origins` for production.

## Authentication flow

1. `POST /initialize` → returns `access_token` + `refresh_token`.
2. Send `Authorization: Bearer <access_token>` on all authenticated endpoints.
3. On 401, call `POST /refresh_token` with the **refresh** token in the `Authorization` header.

Tokens are scoped to a `channel_id` (a session/user key you choose) and optional `user_id`.

## Endpoints

### POST /initialize  (public)
Create or resume a session and obtain tokens.
```json
{
  "channel_id": "user-123",
  "user_id": "user-123",
  "stream_format": "ndjson",        // "ndjson" (default) or "sse"
  "startup_command": null,          // optional; mutually exclusive with startup_action
  "startup_action": null            // optional dict; mutually exclusive with startup_command
}
```
Response:
```json
{
  "access_token": "…", "refresh_token": "…", "token_type": "bearer",
  "expires_in": 3600,
  "startup_output": null            // CommandOutput if a startup command/action ran
}
```
Notes: if `startup_command`/`startup_action` is provided, `user_id` is required. Set
`stream_format` here — it controls the framing of `/invoke_agent_stream` for the session.

### POST /refresh_token  (public)
Header: `Authorization: Bearer <refresh_token>`. Returns a new `TokenResponse`
(`access_token`, `refresh_token`, `token_type`, `expires_in`).

### POST /invoke_agent_stream  (auth) — primary chat endpoint
Body:
```json
{ "user_query": "cancel my most recent order", "timeout_seconds": 60 }
```
Streams the internal workflow↔assistant conversation as it happens, then the final output. Framing
depends on the session's `stream_format`:

- **NDJSON** (`application/x-ndjson`), one JSON object per line:
  ```json
  {"type":"trace","data": { /* trace event */ }}
  {"type":"trace","data": { /* … */ }}
  {"type":"output","data": { /* CommandOutput */ }}
  {"type":"error","data": {"detail":"…"}}   // only on failure
  ```
- **SSE** (`text/event-stream`):
  ```
  event: trace
  data: { /* trace event */ }

  event: output
  data: { /* CommandOutput */ }

  event: error
  data: {"detail":"…"}
  ```

UI guidance: render every `trace` event live (this reproduces the `fastWorkflow run` CLI streaming
UX), then render the `output` event as the final human-readable answer.

### POST /invoke_agent  (auth)
Non-streaming agent turn. Returns a `CommandOutput` JSON with an extra `traces` array. Use only if
streaming is not feasible.

### POST /invoke_assistant  (auth)
Deterministic / non-agentic turn (no planner). Same body as `/invoke_agent`. The service prefixes
the query with `/` automatically when needed.

### POST /perform_action  (auth)
Execute a specific command directly, bypassing intent + parameter extraction.
```json
{ "action": { "command_context": "Order", "command_name": "cancel_order",
              "command_parameters": { "order_id": "W123" } },
  "timeout_seconds": 60 }
```

### Conversation management (auth)
- `POST /new_conversation` — persist the current conversation (generates topic/summary) and start fresh. Use for the **New chat** button.
- `GET /conversations?limit=20` — list past conversations (`ConversationSummary[]`, newest first). Use to render the **history list**.
- `POST /activate_conversation` `{ "conversation_id": 7 }` — restore a past conversation into the active session. Use for **continue previous chat**.
- `POST /post_feedback` `{ "binary_or_numeric_score": 1.0, "nl_feedback": "…" }` — feedback on the latest turn (optional thumbs up/down).
- `POST /cancel_pending` — abandon a suspended `ask_user` clarification turn.

### Health probes (public)
- `GET /probes/healthz` → `{"status":"alive"}` (liveness).
- `GET /probes/readyz` → `200 {"status":"ready", …}` or `503 {"status":"not_ready", …}` (readiness).

### Admin (public; restrict in production)
- `POST /admin/dump_all_conversations` `{ "output_folder": "…" }` → dumps all conversations to JSONL.
- `POST /admin/generate_mcp_token` `{ "channel_id":"…", "user_id":"…", "expires_days":365 }` → long-lived token for MCP clients.

## CommandOutput shape (the `output` event / final answer)

```json
{
  "command_responses": [
    { "response": "Your order W123 was cancelled.",
      "success": true, "artifacts": {}, "next_actions": [], "recommendations": [] }
  ],
  "workflow_name": "", "context": "", "command_name": "", "command_parameters": ""
}
```
Render `command_responses[*].response` as the assistant's final message(s). When `command_name` is
`"ask_user"`, roles invert: `command_parameters` holds the agent's question and the response holds
the user's answer; `success=false` means the question is still open.

## Workflow file formats

### Command file (single-file pattern, preferred) — `_commands/<command_name>.py`
```python
import fastworkflow
from fastworkflow.train.generate_synthetic import generate_diverse_utterances
from pydantic import BaseModel, ConfigDict, Field
from typing import Annotated
from ..application.<module> import <callable_or_class>

class Signature:
    plain_utterances: list[str] = ["cancel order W123", "please cancel my latest order"]

    class Input(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)
        order_id: Annotated[str, Field(default="NOT_FOUND", description="…", examples=["W123"])]

    class Output(BaseModel):
        status: str

    @staticmethod
    def generate_utterances(workflow: fastworkflow.Workflow, command_name: str) -> list[str]:
        return [command_name.split('/')[-1].lower().replace('_', ' ')] + \
               generate_diverse_utterances(Signature.plain_utterances, command_name)

class ResponseGenerator:
    def __call__(self, workflow: fastworkflow.Workflow, command: str,
                 command_parameters: "Signature.Input") -> fastworkflow.CommandOutput:
        # call the app's real business logic here
        result = <callable_or_class>(...)
        return fastworkflow.CommandOutput(
            workflow_id=workflow.id,
            command_responses=[fastworkflow.CommandResponse(response=str(result))],
        )
```
- Use `default="NOT_FOUND"` for parameters so missing values are detected, not hallucinated.
- Use `Field` `description` + `examples` (and `pattern`/`min_length` where helpful) to drive accurate parameter extraction.
- Optional hooks: `db_lookup(workflow_snapshot, command)`, `process_extracted_parameters(...)`.
- For LLM-generated responses, use `fastworkflow.utils.dspy_utils.dspySignature(Signature.Input, Signature.Output)` with `dspy.Predict`.

### Context model — `_commands/context_inheritance_model.json`
Each context entry has at most two keys:
- `"/"` — list of command names available in that context.
- `"base"` — list of parent context names whose commands are inherited.

To add a command: add its name under the relevant context's `"/"`, then create the matching
`_commands/<command_name>.py`. Every declared command must have an implementation file or routing
validation fails.

## Environment variables (set in fastworkflow.env)

LLM model strings (all default to `mistral/mistral-small-latest`):
`LLM_SYNDATA_GEN`, `LLM_PARAM_EXTRACTION`, `LLM_RESPONSE_GEN`, `LLM_PLANNER`, `LLM_AGENT`,
`LLM_CONVERSATION_STORE`. Matching keys live in `fastworkflow.passwords.env` as
`LITELLM_API_KEY_<ROLE>`.

LiteLLM Proxy: prefix model names with `litellm_proxy/`, set `LITELLM_PROXY_API_BASE` in the env
file and `LITELLM_PROXY_API_KEY` in the passwords file (per-role keys are then ignored).

## Troubleshooting

- **PARAMETER EXTRACTION ERROR** — the command's `Field` descriptions/examples are too weak, or the user query lacks a required value. Improve the signature or ask the user.
- **Crash on run** — a corrupted `___workflow_contexts` folder; delete it and rerun.
- **Command not recognized** — import/syntax error in the command file; it failed to load. Check logs.
- **Missing API keys** — keys absent from `fastworkflow.passwords.env`.

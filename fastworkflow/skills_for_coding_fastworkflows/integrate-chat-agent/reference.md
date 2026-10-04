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
Streams the internal workflow↔assistant conversation as it happens, then the final output.

Response headers (read these first; both are CORS-exposed, so a browser client on another origin
can see them):

- `X-FW-Turn-Key` — the execution key of this turn, sent before the first frame. It is the handle
  for recovery (below).
- `X-FW-Stream-Format` — `ndjson` or `sse`, i.e. which framing this body actually uses. It follows
  the session's `stream_format` (fixed at `/initialize`); read the header rather than assuming.

**Event types.** `trace` — one public agent↔workflow interaction. `timeout` — NON-terminal: this
turn passed the `timeout_seconds` you asked for, but the deadline governs *delivery*, not
ownership: the turn is still running, its remaining traces and its `output` still arrive, and
resubmitting would start a second turn. `output` — TERMINAL, the turn's `TurnOutput`; a failed or
`awaiting_user` turn arrives here too. `error` — TERMINAL, the turn produced no output at all.
**Exactly one terminal frame ends the body and nothing follows it.**

- **NDJSON** (`application/x-ndjson`), one JSON object per line. The whole envelope is on the line:
  ```json
  {"type":"trace","seq":0,"turn_key":"…exec…","logical_turn_key":"…","data": { /* trace event */ }}
  {"type":"trace","seq":1,"turn_key":"…exec…","logical_turn_key":"…","data": { /* … */ }}
  {"type":"timeout","seq":2,"turn_key":"…exec…","logical_turn_key":"…",
   "data":{"detail":"Command execution timed out after 60 seconds","timeout_seconds":60,"still_running":true}}
  {"type":"output","seq":3,"turn_key":"…exec…","logical_turn_key":"…","data": { /* TurnOutput */ }}
  ```
- **SSE** (`text/event-stream`) — same events, same payloads; `seq` is the event id and `data` is
  the payload alone, so identity comes from the response header:
  ```
  id: 0
  event: trace
  data: { /* trace event */ }

  id: 1
  event: output
  data: { /* TurnOutput */ }
  ```

**Order and deduplication.** `seq` starts at 0 and is gapless and unique per stream; drop a `seq`
you have already rendered. A chunk can split a frame, so buffer until the delimiter (`\n` for
NDJSON, a blank line for SSE).

**Recovery — read, never resubmit.** If the body dies mid-turn, the turn keeps running server-side:
poll `GET /turns/{key}` and, for the interactions you missed, `GET /turns/{key}/trace` (non
destructive and repeatable). Resubmitting the query would be a second turn. *Which key:* start from
`X-FW-Turn-Key`, but keep the `logical_turn_key` that the NDJSON frames and every `/turns` answer
carry — the execution key only resolves while the turn is live (a chat execution is not retained
after it retires), whereas the store is keyed by the logical one. Where a partially read stream and
the stored record disagree, the record wins.

**HTTP 202 — the same query is already in flight.** A retried or duplicated submission is deduped
onto the running execution and answered `202 {"turn_key":…, "exec_state":"running",
"logical_turn_key":…, "reason":"duplicate_submission"}` with no body to read: the frames belong to
the consumer already reading them. Poll that key. A *different* query on a busy channel is
`409 {"detail":…, "reason":"channel_busy", "turn_key":…}`.

**Artifacts.** The terminal `output` carries `command_outputs[*].command_response.artifacts`
alongside the answer. A value may be inline, or an offloaded reference
`{"__fw_artifact_ref__":"<id>","size":…,"content_type":"…"}` to be fetched from the store that
produced it. Render them beside the final answer, not only in a debug view.

UI guidance: render every `trace` event live (this reproduces the `fastWorkflow run` CLI streaming
UX), show a `timeout` as "still working" rather than a failure, then render the `output` event as
the final human-readable answer.

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
- `POST /cancel_pending` — abandon a suspended `ask_user` clarification turn.

### Feedback (auth)
`POST /post_feedback` records ONE free-form comment about recorded evidence. It
is not a thumbs up/down and carries no score: the old
`{binary_or_numeric_score, nl_feedback}` body and the agent-memory table behind
it were removed, and a request in that shape is rejected rather than scored.

The comment is anchored explicitly — a turn, or a component within it — and
classified with a category and one of its own subcategories:

| `category` | `subcategory` |
| --- | --- |
| `observations_analysis` | `observation`, `analysis` |
| `conclusions` | `what_went_right`, `what_went_wrong` |
| `recommendations` | `what_to_do`, `what_not_to_do` |

```json
POST /post_feedback?turn_key=<turn_key>
{ "target_kind": "turn", "span_ids": [], "target_label": "Turn",
  "provenance": "coding_agent",
  "category": "recommendations", "subcategory": "what_to_do",
  "comment": "Plan all three items before answering." }
```

`comment` is arbitrary text and is stored as written. A comparison comment adds
a `paired` object naming the other execution (`store_id`, `turn_keys`,
`experiment_id`, `task_id`, `attempt`, plus the same target fields), and the one
row is then visible from both tasks.

Reads are separate GETs, never a POST: `GET /api/feedback-notes?turn_key=…` for
one turn, `GET /api/task-feedback?experiment=…&task=…` for a whole task, and
`GET /api/feedback-taxonomy` for the categories above. Evidence that cannot be
written to — a sealed archive, or an older store — still accepts a comment: it
is recorded beside the evidence and the evidence file is not modified.

### Health probes (public)
- `GET /probes/healthz` → `{"status":"alive"}` (liveness).
- `GET /probes/readyz` → `200 {"status":"ready", …}` or `503 {"status":"not_ready", …}` (readiness).

### Admin (off by default; `--enable_admin_endpoints` + auth)
Both routes 404 unless the server was started with `--enable_admin_endpoints`,
and require a Bearer token when it was. Neither is exposed as an MCP tool.
- `POST /admin/dump_all_conversations` `{ "output_folder": "…" }` → dumps all conversations to JSONL.
- `POST /admin/generate_mcp_token` `{ "channel_id":"…", "user_id":"…", "expires_days":365 }` → long-lived token for MCP clients.

## CommandOutput shape (inside `command_outputs[*]` on a turn response)

```json
{
  "command_response":
    { "response": "Your order W123 was cancelled.",
      "success": true, "artifacts": {}, "next_actions": [], "recommendations": [] },
  "workflow_name": "", "context": "", "command_name": "", "command_parameters": ""
}
```
Prefer the turn's top-level `answer` for the assistant's final message. Per-command
text and artifacts live at `command_outputs[*].command_response`. When `command_name`
is `"ask_user"`, roles invert: `command_parameters` holds the agent's question and
the response holds the user's answer; `success=false` means the question is still open.

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
            command_response=fastworkflow.CommandResponse(response=str(result)),
        )
```
- Use `default="NOT_FOUND"` for parameters so missing values are detected, not hallucinated.
- Use `Field` `description` + `examples` (and `pattern`/`min_length` where helpful) to drive accurate parameter extraction.
- Optional hooks: `db_lookup(workflow, field_name, field_value) -> tuple[bool, str | None, list[str]]`,
  `process_extracted_parameters(...)`. `db_lookup` runs for fields marked
  `json_schema_extra={'db_lookup': True}` and has three return states: `(True, value, [])`
  overwrites the field, `(False, None, [suggestions])` fails validation, and
  `(False, None, [])` does neither — use that last one for a value your hook does not own.
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

# run_fastapi_mcp endpoint contract (verified 2026-07-09, v2.22.2, commit c33b9a5)

Source of truth: `fastworkflow/run_fastapi_mcp/__main__.py` (routes),
`fastworkflow/run_fastapi_mcp/utils.py` (request/response models),
`fastworkflow/run_fastapi_mcp/turns.py` (turns engine),
`fastworkflow/run_fastapi_mcp/jwt_manager.py` (auth),
`fastworkflow/run_fastapi_mcp/mcp_specific.py` (MCP mount).

Do NOT trust `fastworkflow/run_fastapi_mcp/README.md` (stale: wrong launch command, obsolete 504
semantics, false MCP-prompts claim) or the shipped
`fastworkflow/skills_for_coding_fastworkflows/integrate-chat-agent/reference.md` (untested
contract; will drift on the fix-qtq wire change).

## Launch

```sh
# Trusted mode (JWT signatures NOT verified) — the only mode reachable from the CLI:
fastworkflow run_fastapi_mcp <workflow> <env> <passwords> --port 8000 --host 0.0.0.0 \
  [--context '<json>'] [--startup_command "..."] [--startup_action '<json>'] [--project_folderpath ...]

# Verified mode — must bypass the CLI (cli.py never forwards the flag):
python -m fastworkflow.run_fastapi_mcp --workflow_path <workflow> \
  --env_file_path <env> --passwords_file_path <passwords> \
  [--context '<json>'] [--startup_command "..."] [--startup_action <path-or-json>] \
  [--project_folderpath ...] [--port 8000] [--host 0.0.0.0] --expect_encrypted_jwt
```

Note the asymmetry: for the CLI, `--startup_action` is forwarded as given; for the module,
`load_args()` treats `--startup_action` as either a request-level JSON string (CLI `--context`
is a JSON string too) — the `/initialize` handler accepts a request-body `startup_action` dict
(takes precedence) or reads `ARGS.startup_action` as a JSON FILE path (`__main__.py`, initialize
handler: `with open(ARGS.startup_action)`).

Startup order caveat: `jwt_keys/` and `SPEEDDICT_FOLDERNAME` resolution are CWD/relative-path
sensitive — always launch from the same working directory.

Interactive docs once running: `http://localhost:<port>/docs` (Swagger, persists the Bearer
token) and `/redoc`.

## Auth model

- `/initialize` mints an RS256 (or `alg=none` in trusted mode) access token (60 min) + refresh
  token (30 days). Claims: `sub`=channel_id, `uid`=user_id, `type`, `jti`, `iss`
  `fastworkflow-api`, `aud` `fastworkflow-client`.
- All non-public endpoints require `Authorization: Bearer <access_token>`. Public (no auth):
  `/`, `/initialize`, `/refresh_token`, `/probes/*`, and — deliberately but dangerously — both
  `/admin/*` endpoints.
- Trusted mode (default): `verify_token` decodes with `verify_signature: False`, manually checks
  only `exp` and `type` (jwt_manager.py:277-299). Any peer can forge tokens.
- The bearer token is propagated into workflows as `workflow_context['http_bearer_token']` and
  refreshed per authenticated request.

## Turn endpoints — shared semantics (turns engine, v2.22.0)

All of `/initialize` (startup turn), `/invoke_agent`, `/invoke_assistant`, `/perform_action`
submit through `submit_turn()` → `TurnRegistry.start_or_get_active()`:

1. Work runs as its own asyncio.Task with the blocking `work_fn` in an executor thread under
   `runtime.lock`.
2. The request waits `timeout_seconds` (body field, default 60) on `done_event` under
   `asyncio.shield`; the shield + separate-task design means a request timeout NEVER cancels
   the work.
3. Finished in-window → `200` with the full turn body. Not finished → `202
   {"turn_key": ..., "exec_state": "running"}`.
4. Identical retry (same channel_id + kind + args → same sha256 idempotency key) REJOINS the
   running execution and waits again. This is the ONLY 202 recovery path: **there is no
   `GET /turns/{turn_key}` endpoint** (Step 2 of the design, unbuilt — verified against every
   route decorator). For startup turns you may also re-POST `/initialize`.
5. A different concurrent turn on the same channel → `409` with the active turn_key in `detail`.
   The busy check is `turn_registry.has_active(channel_id)` — never `runtime.lock.locked()`.
6. Conversation + suspended state persist BEFORE `exec_state=DONE` (never observe
   done-with-unsaved-state).
7. `TurnRegistry.evict_terminal` is a no-op in Step 1 (`ttl_expires_at` never set,
   turns.py:226-243): terminal executions accumulate in process memory until restart.
8. A server restart loses all in-flight executions (`LOST` is an in-process placeholder state;
   Step 3 durability unbuilt).

### 200 turn body (render_turn_response, turns.py:386-424)

```json
{
  "turn_key": "...",
  "exec_state": "done",
  "status": "<TurnStatus>",
  "success": true,
  "answer": "plain text final answer",
  "command_responses": [ ... ],
  "command_outputs": [ ... ],
  "traces": [ ... ]          // only if trace events were collected
}
```

`command_responses` semantics differ by kind (`_command_responses_for`): for `invoke_agent` it is
`[{"response": <synthesized agent answer>, "success": ...}]`; for assistant/action turns it is
the LAST command's responses with artifacts preserved. Error case: 200 with
`{turn_key, exec_state, error}` which `_turn_json_response` converts to HTTP 500 for turn
endpoints.

**fix-qtq warning (open):** these shapes serialize `CommandOutput` internals; the planned
cutover to the `TurnOutput` public projection is a declared breaking wire change (pre-v3.0
shapes are explicitly NOT compatibility-protected). Re-verify this section after fix-qtq lands.

## Endpoint-by-endpoint

### POST /initialize (public)
Body: `{channel_id, user_id?, stream_format? ("ndjson"|"sse", default ndjson — controls
/invoke_agent_stream framing for this session), startup_command?, startup_action? (dict),
timeout_seconds?=60}`. `startup_command` XOR `startup_action` (400 if both); `user_id` required
when a startup is provided. Response `InitializeResponse`: tokens + `startup_output`
(CommandOutput, present if the startup turn finished in-window), `startup_turn_key`,
`startup_exec_state` (queued|running|done|lost), `startup_error`. Startup runs as the FIRST turn
under the registry. Re-calling `/initialize` for an existing session returns fresh tokens plus
the SAME startup execution's three-state status — never a silently-empty result (that silent
emptiness was the original fix-85g bug).

### POST /refresh_token (public; needs refresh token in body/header per model)
Returns a new `TokenResponse`.

### POST /invoke_agent (auth)
Body: `{user_query, timeout_seconds?=60}` (`InvokeRequest`). Agent-mode turn via the registry.

### POST /invoke_agent_stream (auth) — DIFFERENT concurrency guard
Streams NDJSON (default) or SSE per the session's `stream_format`. NOT integrated with the
TurnRegistry: it checks `runtime.lock.locked()` (__main__.py:963) and emits an error event when
busy; no idempotency key, no 202 deferral, and the trace drain is destructive (a disconnected
client cannot replay). It strips leading `/` from `user_query`, so no deterministic-prefix
override here. Owns operation_id `invoke_agent`, which is what the MCP tool of that name maps
to. Known inconsistency — candidate for registry migration; unverified whether intentional.

### POST /invoke_assistant (auth)
Same body as `/invoke_agent`. Deterministic: inside the turn it prepends `/` to the query
(`f"/{request.user_query.lstrip('/')}"`, __main__.py:1123-1129) unless the session is already in
an assistant-mode command state.

### POST /perform_action (auth)
Body: `{action: {...}, timeout_seconds?=60}`; the dict is converted to `fastworkflow.Action` —
422 on bad format. Bypasses intent detection AND parameter extraction entirely.

### POST /cancel_pending (auth)
Abandons a suspended Topology-B `ask_user` turn: calls `execution_context.cancel_pending()` and
clears the durable session-state blob. Returns `{"status": "ok", "cleared": <bool>}`.

### POST /new_conversation (auth) — rejected with 409 while a turn is active (`_reject_if_busy`).
### GET /conversations?limit=20 (auth) — list past conversations for the channel.
### POST /post_feedback (auth) — also busy-guarded
Body: `{binary_or_numeric_score?: float, nl_feedback?: str}` — at least one required (validator);
booleans coerce to 1.0/0.0. Applies to the latest turn.
### POST /activate_conversation (auth) — resume an archived conversation by ID; also busy-guarded.

(Busy guard verified: `_reject_if_busy` is called by exactly three endpoints —
new_conversation, post_feedback, activate_conversation.)

### GET /probes/healthz, GET /probes/readyz (public)
Kubernetes-style probes; a middleware suppresses access logs for 200 responses.

### POST /admin/dump_all_conversations (UNAUTHENTICATED)
Body includes `output_folder`; scans ALL `.rdb` files under
`SPEEDDICT_FOLDERNAME/channel_conversations` (active or not) and writes
`all_conversations_<ts>.jsonl`. Returns `{"file_path": ...}`.

### POST /admin/generate_mcp_token (UNAUTHENTICATED)
Body: `{channel_id, user_id?, expires_days?=365}`. Mints a long-lived access token for MCP
client config (e.g. Claude Desktop). Docstring: "should be restricted to administrators only in
production."

## MCP mount

`setup_mcp()` mounts fastapi-mcp 0.4.0 via `FastApiMCP(app, exclude_operations=[...]).mount_http()`
at `/mcp`, AFTER all endpoints are defined (ordering requirement — fastapi-mcp discovers routes
at mount time). Excluded operations: `root`, `dump_all_conversations`, `generate_mcp_token`,
`rest_initialize`, `perform_action`, `rest_invoke_agent`, `refresh_token`. Resulting MCP tools:
`invoke_agent` (→ /invoke_agent_stream), `invoke_assistant`, `new_conversation`,
`get_all_conversations`, `post_feedback`, `activate_conversation`. Custom prompts are NOT
supported by fastapi-mcp 0.4.0 (mcp_specific.py:60-64) despite the package README's claim.

## Shutdown

Lifespan shutdown waits up to 30 s for active turns (`wait_for_active_turns_to_complete(30)`,
__main__.py:340), finalizes conversations, stops chat sessions. Anything still running after
30 s is lost (Step 3 durability unbuilt).

## Related but different: fastworkflow/mcp_server.py

`fastworkflow/mcp_server.py` is NOT this server. It is a library-level example wrapper
(`FastWorkflowMCPServer`) exposing per-command JSON-RPC 2.0 `tools/list` / `tools/call` directly
over `CommandExecutor.perform_mcp_tool_call`; it also fails fast if the workflow is untrained.
The FastAPI server's MCP surface is the coarse invoke_agent/invoke_assistant tool set above.

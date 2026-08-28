# `fastworkflow run_chatbot` — CLI parity table [R24]

The chatbot's chat (test) mode drives the same FastAPI server surface the CLI
drives (`/initialize`, `/invoke_agent`, `/invoke_assistant`,
`/new_conversation`). This table lists what remains **CLI-only** and why.

The chatbot manages the connection itself: it spawns the workflow's server,
mints its session via `/initialize`, and uses one fixed `channel_id`
(`chatbot`) so history accumulates in one place across launches — the
developer never types a URL, token, or channel. The Advanced panel exists
only for connecting to an externally managed server.

Startup commands, startup actions, and per-session context JSON are
deliberately **not** exposed by the chat UI: the session is driven by what the
user types, so pre-seeding it from a form makes no sense in an interactive UX.
They remain server-launch decisions (`run_fastapi_mcp` flags) and programmatic
`/initialize` request fields.

| Capability | CLI | Chatbot chat mode | Notes |
|---|---|---|---|
| Agent-mode chat | `fastworkflow run <wf>` | Yes — messages go to `/invoke_agent` | Auto-connected on launch |
| Deterministic execution | `/`-prefixed message at the prompt, or `--assistant` | Yes — a message starting with `/` goes to `/invoke_assistant` | Same `/` convention as the CLI |
| New conversation | `//new` at the CLI prompt | Yes — the **New conversation** header button calls `POST /new_conversation` | |
| Workflow selection | `run`/`train`/`build` commands take a positional workflow path; `run_chatbot` does not | Yes — pick from the browser (bundled examples + directory browser); switchable at runtime | Workflow selection is deliberately browser-owned for `run_chatbot` |
| Per-session startup command | `--startup_command` | **Not offered** — no chat-UI input | Set it on the server (`run_fastapi_mcp --startup_command`), or POST `/initialize` yourself. If the server has one, the chatbot renders that first turn rather than hiding it |
| Per-session startup action | `--startup_action <file.json>` | **Not offered** — no chat-UI input | Same: server flag, or the `/initialize` `startup_action` field via curl/Swagger (mutually exclusive with `startup_command`, 400 if both) |
| Per-session workflow context | `--context '<json>'` | **Not offered** — no chat-UI input | The `/initialize` `context` field still exists for programmatic callers; it applies only on the call that creates the session |
| Insights distillation | `fastworkflow run --generate_insights` | **CLI-only — explicitly out of scope** per the design decision log [R24] | Teacher/student comparison needs the interactive CLI loop |
| Environment/passwords selection | Other workflow commands retain their existing env-file arguments; `run_chatbot` does not | Auto-detected from the workflow directory, then the bundled `examples/` parent for bundled workflows. If either file is missing, choose files in the browser or create workflow-local copies from the bundled templates | Selected file contents are copied to owner-only workflow-local files; browser paths are never sent to the server |
| Bind address / port | `run_fastapi_mcp --host/--port` | **CLI-only** — a chatbot-spawned server is always `127.0.0.1` [R19]; any wider bind is a command-line decision on `run_fastapi_mcp` itself | Never a chatbot UI option |
| JWT posture | `run_fastapi_mcp --expect_encrypted_jwt` | The auto-spawned loopback server runs unsigned dev JWTs by default (owner decision amending [R19] — the chatbot mints its own tokens via `/initialize`); `run_chatbot --expect-encrypted-jwt` restores signed mode (paste a token in the Advanced panel) | |
| Training | `fastworkflow train` | Yes — the picker's **Train** button (`POST /api/train`) spawns a detached `fastworkflow train` for a local workflow (refused for bundled examples; one run at a time; survives chatbot exit; log + pid under the workflow's state dir) | Train metrics land in the observability store (Phase 6) |
| Build / refine | `fastworkflow build/refine` | **CLI-only** | Out of scope |
| Store maintenance | `run_chatbot` has no maintenance flags | **Clear conversations** removes all conversation, turn, span, artifact, and feedback records after explicit confirmation | Training runs, diagnostics, and monotonic conversation counters are preserved |

## Launching

Everything automatic (server spawned, chat auto-connected):

```bash
fastworkflow run_chatbot                 # pick the workflow in the browser
```

Pin the spawned server's port, or require signed tokens:

```bash
fastworkflow run_chatbot --server-port 8000
fastworkflow run_chatbot --expect-encrypted-jwt
```

The spawned server always binds `127.0.0.1`, gets CORS pinned to loopback
origins (`--cors_loopback_only` — any-port because WSL/IDE port forwarders
re-expose the UI on other local ports; never a routable origin), and is
terminated when the chatbot exits, including on SIGTERM [R19 as amended].

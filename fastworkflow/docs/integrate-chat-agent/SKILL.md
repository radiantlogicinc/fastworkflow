---
name: integrate-fastworkflow-chat-agent
description: >-
  Integrate a fastWorkflow-based agentic chat agent into an existing application so users can
  execute the app's business logic and ask questions in natural language, with a popup chat UI
  that streams internal workflow/assistant conversations before the final answer. Use when a
  developer asks to add a chat agent, conversational UI, natural-language interface, or
  fastWorkflow backend to their app, or mentions "AI-enabling" their application.
---

# Integrate a fastWorkflow Chat Agent

## Mission

> Your mission is to integrate agentic chat UI within the application's UI that allows natural
> language conversations to execute the business logic functionality and provide human readable
> answers to questions. All information that is exposed via the UI should also be available via
> the chat interface. Use fastWorkflow (`uv add fastworkflow`) to implement the backend for this
> chat functionality.
>
> This chat window should pop up when an overlaid chat icon is clicked in the web UI. Users should
> be able to start new chats and also continue previous chats. The chat window should stream
> internal fastWorkflow conversations between workflow and assistant (same UX as fastWorkflow cli
> invoked via `fastWorkflow run`) before giving the final answer.

fastWorkflow AI-enables an existing **Python** application by wrapping its classes and methods with
an intent-detection + parameter-extraction + execution pipeline. The chat backend is hosted as the
fastWorkflow **FastAPI service**, which exposes streaming HTTP endpoints the chat UI consumes.

## Track all work in beads

Use **beads (`bd`)** for issue tracking — not markdown TODOs. Install it if missing
(`npm install -g beads` or see beads docs). Create one epic for this integration and tasks under it
that mirror the workflow phases below. Mark tasks `in_progress` / `closed` as you go.

```bash
bd create "Integrate fastWorkflow chat agent" -t epic -p 1 --json
bd create "Build & train fastWorkflow from app business logic" -t task -p 1 --deps discovered-from:<epic-id> --json
bd create "Host fastWorkflow FastAPI streaming service" -t task -p 1 --deps discovered-from:<epic-id> --json
bd create "Build popup chat UI with streaming + conversation history" -t task -p 1 --deps discovered-from:<epic-id> --json
```

## Workflow

Copy this checklist and track progress:

```
- [ ] Step 1: Discover the application's business logic to expose
- [ ] Step 2: Install fastWorkflow and scaffold the workflow directory
- [ ] Step 3: Write command files for each app capability
- [ ] Step 4: Set up env files — PAUSE for the user to add API keys
- [ ] Step 5: Train the workflow's intent models
- [ ] Step 6: Smoke-test the agent from the CLI
- [ ] Step 7: Host the fastWorkflow FastAPI streaming service
- [ ] Step 8: Build the popup chat UI (new/continue chats + live trace streaming)
- [ ] Step 9: End-to-end verification
```

### Step 1: Discover the business logic to expose

Enumerate every capability the app's UI exposes (and any read-only "questions" users may ask). The
hard requirement: **everything available in the UI must also be reachable through chat.** Map each
UI action / data view to a Python class + method or function in the app's codebase. Group related
operations into classes (contexts) that hold state. This mapping becomes the command set.

If the app's backend is **not** Python, fastWorkflow still runs as a separate Python sidecar
service; the commands call into the app via its existing API/SDK/DB client instead of in-process
imports.

### Step 2: Install fastWorkflow and scaffold

```bash
uv add fastworkflow            # or: pip install "fastworkflow[server]"
fastworkflow examples fetch hello_world   # provides fastworkflow.env + fastworkflow.passwords.env templates
```

Create the workflow directory next to the app code:

```
<app>/chat_workflow/
├── application/        # symlink or thin wrappers calling the app's real business logic
├── _commands/          # generated + hand-edited command files
├── fastworkflow.env
└── fastworkflow.passwords.env
```

Add `___workflow_contexts`, `___command_info`, `___convo_info` to `.gitignore`.

### Step 3: Write the command files

For each capability mapped in Step 1, create a command file in `_commands/<command_name>.py` that
invokes the app's business logic, and declare it in `_commands/context_inheritance_model.json`.
Group commands into contexts (classes) that hold state. Use Pydantic `Input`/`Output` signatures
with strong `Field` descriptions, examples, and a `default="NOT_FOUND"` so missing parameters are
detected rather than hallucinated. See [reference.md](reference.md) for the command-file structure
and context-model format.

### Step 4: Set up env files — PAUSE for API keys

Copy `fastworkflow.env` and `fastworkflow.passwords.env` into the workflow directory. The service
needs LLM API keys to run. **Stop here, tell the user the absolute path of
`fastworkflow.passwords.env`, and ask them to add the keys.** Do not attempt to invent or commit
keys. Minimum keys (Mistral small / OpenRouter free tiers work):

```
LITELLM_API_KEY_SYNDATA_GEN=...
LITELLM_API_KEY_PARAM_EXTRACTION=...
LITELLM_API_KEY_RESPONSE_GEN=...
LITELLM_API_KEY_PLANNER=...
LITELLM_API_KEY_AGENT=...
LITELLM_API_KEY_CONVERSATION_STORE=...   # FastAPI conversation topic/summary
```

### Step 5: Train

```bash
fastworkflow train <app>/chat_workflow <app>/chat_workflow/fastworkflow.env <app>/chat_workflow/fastworkflow.passwords.env
```

Training generates synthetic utterances and trains intent models into `___command_info/`. It takes
several minutes and the first run also downloads HuggingFace models.

### Step 6: Smoke-test from the CLI

```bash
fastworkflow run <app>/chat_workflow <app>/chat_workflow/fastworkflow.env <app>/chat_workflow/fastworkflow.passwords.env
```

Ask "what can you do?" and exercise a few real commands. Prefix a command with `/` to force
deterministic (non-agentic) execution. Confirm the agent reaches every UI capability from Step 1.

### Step 7: Host the FastAPI streaming service

The chat backend is the bundled FastAPI-MCP service (requires the `server` extra). It exposes JWT
auth, conversation persistence, and the streaming endpoint that powers the live trace UX.

```bash
python -m fastworkflow.run_fastapi_mcp \
  --workflow_path <app>/chat_workflow \
  --env_file_path <app>/chat_workflow/fastworkflow.env \
  --passwords_file_path <app>/chat_workflow/fastworkflow.passwords.env \
  --port 8000
```

Run this alongside the app's backend (e.g. as a sidecar/separate process) and proxy it from the
app's server, or call it directly from the frontend. Full endpoint contracts (request/response
shapes, auth flow, streaming format) are in [reference.md](reference.md).

### Step 8: Build the popup chat UI

Required UX (matches the `fastWorkflow run` CLI experience):

1. **Overlaid chat icon** in the app UI that toggles a chat window.
2. **New chat** button → `POST /new_conversation`.
3. **Continue previous chats** → `GET /conversations` lists past conversations;
   `POST /activate_conversation` restores one into the window.
4. **Stream internal conversation** → send each user message to `POST /invoke_agent_stream` and
   render the `trace` events live (the back-and-forth between workflow and assistant) *before*
   rendering the final `output` event. This is the core differentiating UX — do not collapse it
   into a single final answer.

Auth flow: call `POST /initialize` once per user/session to get a JWT, send it as
`Authorization: Bearer <token>` on subsequent calls, and refresh via `POST /refresh_token`.

### Step 9: End-to-end verification

- Click the chat icon → window opens.
- Send a natural-language command → trace events stream, then a human-readable final answer.
- Verify every UI capability from Step 1 is reachable through chat.
- Start a new chat, send messages, then reopen a previous conversation and confirm history restores.
- Close any open `bd` tasks with a reason.

## Key references

- HTTP API contract, auth, streaming format, command-file & context-model structure: [reference.md](reference.md)
- fastWorkflow concepts, CLI, env vars: project `README.md`

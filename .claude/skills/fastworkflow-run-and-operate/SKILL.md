---
name: fastworkflow-run-and-operate
description: >-
  Operate fastWorkflow at runtime: use when you need to run the CLI (examples/build/refine/
  train/run/run_fastapi_mcp), fetch or pick a bundled example workflow, do the quickstart,
  understand agent vs assistant vs "/" deterministic mode, identify or clean runtime artifacts
  (___command_info, ___workflow_contexts, ___convo_info, observability.sqlite3, jwt_keys), launch or call
  the FastAPI+MCP server (endpoints, 202/409 turns semantics, JWT/security posture), or when you
  see symptoms like "workflow is not trained", HTTP 409 "turn already in progress", a hanging 202
  response, or "server extra not installed". Do NOT use for training internals or intent-model
  tuning (fastworkflow-nlu-pipeline-reference), env-var semantics (fastworkflow-config-and-flags),
  dev-environment setup (fastworkflow-build-and-env), or tau-bench benchmark runs
  (fastworkflow-taubench-reference / tau2-reliability-campaign).
---

# fastWorkflow: Run and Operate

Runbook for operating fastWorkflow v2.22.2: the `fastworkflow` CLI, the bundled examples, the
runtime artifacts on disk, and the FastAPI+MCP server. Every command, flag, path, and endpoint
below was verified against the code on 2026-07-09 (see Provenance at the end).

## When to use / when NOT to use

Use this skill when you are **running** things: fetching examples, training-then-running a
workflow, choosing agent vs assistant mode, launching the HTTP/MCP server, interpreting its
status codes, or deciding which `___*` directory is safe to delete.

Use a sibling instead when:

| Need | Sibling skill |
|---|---|
| Why intent detection / param extraction behaves as it does; model internals | `fastworkflow-nlu-pipeline-reference` |
| Every env var's default, consumers, sharp edges | `fastworkflow-config-and-flags` |
| Recreate the dev environment, secrets, poetry/venv | `fastworkflow-build-and-env` |
| tau-bench / tau2 mechanics, benchmark parity rules | `fastworkflow-taubench-reference` |
| Running the E0-E25 reliability experiments | `tau2-reliability-campaign` |
| A runtime failure you cannot triage from this page | `fastworkflow-debugging-playbook` |
| Whether a change to server/CLI code is allowed | `fastworkflow-change-control` |
| Architecture invariants behind the turns engine | `fastworkflow-architecture-contract` |
| Measuring instead of eyeballing (diagnostic scripts) | `fastworkflow-diagnostics-and-tooling` |

Jargon used below, defined once:
- **Workflow** — a directory with a `_commands/` folder; the unit you build/train/run.
- **Command** — one Python file in `_commands/` wrapping a piece of app logic; the unit of intent.
- **Agent mode** — an LLM ReAct agent plans and calls workflow commands as tools (default for `run`).
- **Assistant mode** — deterministic: your message goes straight through intent detection → parameter extraction → execution, no agent loop.
- **Turn** — one user message processed end-to-end, producing a `TurnOutput`.
- **Channel** — the server's session identity (`channel_id`); one conversation stream per channel.
- **MCP** — Model Context Protocol; the server auto-exposes selected endpoints as MCP tools at `/mcp`.

## 1. CLI anatomy

Entry point: `fastworkflow = "fastworkflow.cli:main"` (pyproject.toml:31). Six subcommands
(verified via `fastworkflow --help`): `examples`, `build`, `refine`, `train`, `run`,
`run_fastapi_mcp`. Heavy imports are lazy, so `--help` is fast.

| Subcommand | Arguments / flags (all verified in `fastworkflow/cli.py`) |
|---|---|
| `examples list` | none |
| `examples fetch <name>` | `--force` (skip the overwrite prompt) |
| `build` | `--app-dir/-s` (required), `--workflow-folderpath/-w` (required), `--overwrite`, `--stub-commands <csv>`, `--no-startup` |
| `refine` | `--workflow-folderpath/-w` (required) |
| `train` | `workflow_folderpath [env_file_path] [passwords_file_path]` |
| `run` | `workflow_path [env_file_path] [passwords_file_path]`, `--context_file_path`, `--startup_command`, `--startup_action <json file>`, `--keep_alive` (default True), `--project_folderpath`, `--assistant` |
| `run_fastapi_mcp` | `workflow_path [env_file_path] [passwords_file_path]`, `--context <json string>`, `--startup_command`, `--startup_action`, `--project_folderpath`, `--port 8000`, `--host 0.0.0.0` |

Sharp edges:

- **Default env files resolve to the workflow directory**, not your CWD. When you omit the two
  positional env paths, `find_default_env_files()` (cli.py:177-191) returns
  `<workflow_path>/fastworkflow.env` and `<workflow_path>/fastworkflow.passwords.env`. The CLI
  `--help` text saying "default: .env in current directory" is **wrong** (doc rot inside argparse
  help; code is authoritative). For fetched examples, always pass `./examples/fastworkflow.env
  ./examples/fastworkflow.passwords.env` explicitly — the error message tells you the exact
  command if you forget.
- `build` and `refine` load **no env files at all** — their LLM config
  (`LLM_COMMAND_METADATA_GEN`, `LITELLM_API_KEY_COMMANDMETADATA_GEN`) must be in the OS
  environment. See `fastworkflow-config-and-flags`.
- `run_fastapi_mcp` requires the server extra: `pip install "fastworkflow[server]"`. The CLI
  fails fast with that instruction if fastapi/uvicorn/fastapi-mcp/pyjwt are missing
  (`_require_server_extra`, cli.py:384-413).
- `train` requires the `training` extra (`datasets`).

## 2. The examples system

`fastworkflow examples fetch <name>` copies a bundled example from the installed package to
`./examples/<name>` (root `examples/` is gitignored), skipping `___command_info`, `__pycache__`,
`*.pyc`, prompting before overwrite unless `--force`, and dropping template `fastworkflow.env` +
`fastworkflow.passwords.env` into `./examples/` (stub passwords file created if missing).

Eight bundled examples (verified via `fastworkflow examples list`), a deliberate learning
progression — details and per-example run commands in
[references/examples-catalog.md](references/examples-catalog.md):

| Order | Example | Teaches |
|---|---|---|
| 1 | `hello_world` | Minimal: one global command wrapping `add_two_numbers` |
| 2 | `messaging_app_1` | Smallest hand-written workflow: one global command over a plain function |
| 3 | `messaging_app_2` | Class-based command context (`_commands/User/`) + `startup.py` binding an instance |
| 4 | `messaging_app_3` | Context inheritance (`PremiumUser` extends `User` via `context_inheritance_model.json`) |
| 5 | `messaging_app_4` | Container context + navigation (`ChatRoom`, `set_root_context.py`, `startup_action.json`) |
| 6 | `retail_workflow` | **tau-bench retail domain**: 15 global e-commerce commands + `retail_data/` fixtures + `tools/` |
| 7 | `simple_workflow_template` | JSON-driven WorkItem-hierarchy template |
| 8 | `extended_workflow_example` | Workflow inheritance (`workflow_inheritance_model.json`): override, add, and wrap commands |

**Discipline rule (tau-bench parity is sacred):** `retail_workflow` mirrors the tau-bench retail
tools. Never modify its tools/tasks (or the tau-bench fork's) for benchmark runs; any nonstandard
trade must be disclosed, never silent. See `fastworkflow-taubench-reference` and
`fastworkflow-change-control` before touching it.

## 3. Verified quickstart

```sh
pip install fastworkflow                     # Python 3.11+; Windows -> WSL
fastworkflow examples fetch hello_world      # creates ./examples/hello_world + env templates
nano ./examples/fastworkflow.passwords.env   # paste LITELLM_API_KEY_* (one free Mistral key works for all roles)
fastworkflow train ./examples/hello_world ./examples/fastworkflow.env ./examples/fastworkflow.passwords.env
fastworkflow run   ./examples/hello_world ./examples/fastworkflow.env ./examples/fastworkflow.passwords.env
```

What happens and what lands where:

- `train` first (re)trains fastWorkflow's own internal `command_metadata_extraction` workflow if
  stale, then yours. "Train" = generate synthetic utterances via LLM + fit two small BERT-class
  classifiers per context. ~5 min on CPU for hello_world; no GPU at runtime.
- Artifacts appear in `./examples/hello_world/___command_info/`: `command_directory.json`,
  `routing_definition.json`, `<command>_param_labeled.json`, and per-context dirs (`global/`)
  containing `tinymodel.pth`, `largemodel.pth`, `label_encoder.pkl`, `threshold.json`,
  `tiny_ambiguous_threshold.json`, `large_ambiguous_threshold.json`.
- `run` fails fast if untrained ("Workflow 'X' is not trained (missing model artifacts for
  context(s): ...)") and prints the exact train command (run/__main__.py:156-172).
- At the `User >` prompt try "what can you do?" or "add 49 + 51".

Server variant: `pip install "fastworkflow[server]"` then
`fastworkflow run_fastapi_mcp ./examples/hello_world ./examples/fastworkflow.env ./examples/fastworkflow.passwords.env --port 8000`.

## 4. Agent mode vs assistant mode vs "/" prefix

| Mechanism | Effect | Where verified |
|---|---|---|
| `fastworkflow run ...` (default) | Agent mode: `run_as_agent = not args.assistant` | run/__main__.py:185 |
| `fastworkflow run ... --assistant` | Deterministic assistant mode for the whole session | cli.py run parser |
| Message starting with `/` at the prompt | Forces deterministic execution of that ONE command even in agent mode: sets `is_assistant_mode_command` during INTENT_DETECTION, which makes `_should_run_agent_for_message()` bypass the agent | workflow_execution_context.py:617-632 |
| `//exit` | Quits the CLI loop | run/__main__.py:222-223 |
| `//new` | Clears conversation history, starts a new conversation | run/__main__.py:224-227 |

The CLI loop prints `Agent >` / `Workflow >` trace lines while processing, then a pretty
`CommandOutput` panel (context, command, parameters, responses, artifacts, next_actions,
recommendations).

Server equivalents: `/invoke_agent` = agent turn; `/invoke_assistant` = deterministic (it
prepends `/` inside the turn, __main__.py:1125-1129); `/perform_action` = direct `Action`,
bypassing NLU entirely.

## 5. Artifact conventions: what writes what, what is safe to delete

All `___*` dirs are gitignored, developer-local state (.gitignore:2-5). Paths are relative to
the workflow dir unless marked "CWD" (created wherever the process was launched).

| Artifact | Written by | Contents | Safe to delete? |
|---|---|---|---|
| `___command_info/` | `fastworkflow train` | Trained intent models per context, `command_directory.json`, `routing_definition.json`, param-labeled JSONs | Yes — retrain to regenerate. **NEVER let tests/experiments wipe the bundled examples' copies** (`fastworkflow/examples/*/___command_info`): the fix-0hb incident (commit fa97b48) — a test fixture rmtree'd hello_world's trained model and poisoned the next full-suite run. Train into temp copies. |
| `___command_info/command_directory.json` | train / cache refresh | Includes `source_fingerprint` — a hash over the `_commands` source set; on mismatch the cache is rebuilt (v2.22.1 fix, commit b5747df; `compute_commands_source_fingerprint`, command_directory.py:627) | Covered above |
| `___workflow_contexts/` (i.e. `SPEEDDICT_FOLDERNAME`, template default) | server + runtime | `channel_conversations/<channel_id>.rdb` (conversation history), `channel_session_state/` (suspended ask_user turns), `function_cache/` | Deleting loses conversation history and suspended turns. Safe only if you accept that. |
| `___convo_info/` | CME intent pipeline at runtime | **Live** utterance-correction cache (1-shot adaptation of intent detection from user corrections; intent_detection.py:31-33, cache_matching.py) | Deleting loses learned intent corrections; otherwise safe. NOT a stale dir despite the name. |
| `observability.sqlite3` (state root, per workflow) | `SQLiteTraceSink`, every topology incl. CLI | Turn records, spans, artifacts — the single source of truth for what a turn did (state_paths.py:`observability_db`) | Deleting loses all turn history/traces; the workflow still runs. Prune instead. |
| `action.jsonl` (CWD) | **nothing — retired in Phase 7** | Was a CLI-only debug mirror of the agent action log; superseded by `observability.sqlite3` (post-mortem) and the in-process `ctx.action_log` (live) | Yes, always — any copy on disk is pre-Phase-7 residue. |
| `___user_conversations/` (repo root) | nothing current | Legacy pre-rename conversation dir; code only writes `channel_conversations` now (utils.py:400) | Yes — stale residue. |
| `speedict/` (repo root) | nothing current | Empty stale dir; tests use `tmp_path/'speedict'` | Yes — stale residue. |
| `jwt_keys/` (CWD) | run_fastapi_mcp on first use | RSA-2048 keypair (jwt_manager.py:59) in `./jwt_keys` (jwt_manager.py:29-31), `private_key.pem` chmod 600 (jwt_manager.py:94) | Regenerated on next launch; deleting invalidates previously issued **signed** tokens (only matters with `--expect_encrypted_jwt`). CWD-relative — launch the server from a stable directory. |

Since v2.21.4 the **CLI does not resume workflow context across process restarts** (disk backend
deliberately dropped); resumable durability exists only in the FastAPI server via SessionStateStore
and the conversation records. Note the Phase-7 nuance: CLI turns are now *recorded* in
`observability.sqlite3` under a synthetic `cli:<timestamp>` channel, so they are readable
after the fact — but recording is not resumption, and a new CLI process still starts cold.

## 6. The FastAPI+MCP server

Two launch commands with one **critical** difference:

```sh
# 1) CLI wrapper — trusted mode ONLY. It re-execs python -m ... but NEVER forwards
#    --expect_encrypted_jwt (verified: cli.py:454-471 builds the subprocess argv without it).
fastworkflow run_fastapi_mcp ./examples/hello_world ./examples/fastworkflow.env ./examples/fastworkflow.passwords.env --port 8000

# 2) Direct module — the ONLY way to enable JWT signature verification.
python -m fastworkflow.run_fastapi_mcp --workflow_path ./examples/hello_world \
  --env_file_path ./examples/fastworkflow.env --passwords_file_path ./examples/fastworkflow.passwords.env \
  --port 8000 --expect_encrypted_jwt
```

Full endpoint list (every route decorator in `run_fastapi_mcp/__main__.py`, verified by grep;
request/response detail in [references/fastapi-endpoints.md](references/fastapi-endpoints.md)):

| Endpoint | Method | Auth | Purpose |
|---|---|---|---|
| `/` | GET | none | HTML landing page |
| `/probes/healthz`, `/probes/readyz` | GET | none | K8s liveness/readiness; logs suppressed on 200 |
| `/initialize` | POST | none (mints tokens) | Create session + JWTs; runs `startup_command`/`startup_action` as the FIRST turn (200 done / 202 still running) |
| `/refresh_token` | POST | refresh token | New access token |
| `/invoke_agent` | POST | Bearer | Synchronous agent turn (turns engine) |
| `/invoke_agent_stream` | POST | Bearer | Streaming turn; NDJSON default or SSE per session `stream_format` set at `/initialize` |
| `/invoke_assistant` | POST | Bearer | Deterministic turn (prepends `/`) |
| `/perform_action` | POST | Bearer | Direct `Action` execution, bypasses param extraction; 422 on bad format |
| `/cancel_pending` | POST | Bearer | Abandon a suspended ask_user turn, clear durable pending state |
| `/new_conversation` | POST | Bearer | Archive current, start fresh |
| `/conversations?limit=20` | GET | Bearer | List past conversations |
| `/post_feedback` | POST | Bearer | Record feedback on a turn |
| `/activate_conversation` | POST | Bearer | Resume an archived conversation |
| `/admin/dump_all_conversations` | POST | **NONE** | Dump every channel's conversations to a JSONL file |
| `/admin/generate_mcp_token` | POST | **NONE** | Mint long-lived (default 365-day) tokens for MCP clients |

There is **no `GET /turns` endpoint** — see below.

### v2.22.0 turns engine semantics (what callers must know)

Shipped in commit cf3eeae ("fix-85g Step 1"); invariants pinned in the `turns.py` module
docstring. Behavior contract:

- **Single-flight per channel.** One turn at a time per `channel_id`. The busy signal is the
  TurnRegistry's per-channel active-execution pointer — deliberately NOT `runtime.lock.locked()`
  (the lock is released while a request defers and across ask_user suspension).
- **409** — a *different* concurrent turn on a busy channel gets HTTP 409 with the active
  `turn_key` in the detail.
- **202 deferral, wait-or-defer never wait-or-abort.** Each request waits a bounded window
  (`timeout_seconds`, default 60; utils.py:37) on the execution's `done_event` under
  `asyncio.shield`. Timeout returns `202 {"turn_key": ..., "exec_state": "running"}` and the
  work KEEPS RUNNING — a request timeout never cancels execution.
- **Idempotent retry-rejoin.** Idempotency key = sha256(channel_id + kind + normalized args)
  (turns.py:84-100). Retrying the IDENTICAL request rejoins the same execution (no duplicate LLM
  spend) and waits again. That is the only recovery path for a 202 today, because:
- **NO `/turns` polling endpoint exists — Step 2 is unbuilt.** The `/initialize` docstring says
  "poll /initialize or /turns", but no such route exists (verified against all route
  decorators). Also unbuilt from Step 2: TTL eviction (`evict_terminal` is a structural no-op —
  `ttl_expires_at` is never set, turns.py:226-243, so terminal executions accumulate in memory)
  and non-destructive trace replay. Step 3 (durability) too: a server restart loses all running
  turns (`LOST` state is a placeholder).
- **Persist before DONE.** Conversation + suspended state are saved under the lock before
  `exec_state=DONE`, so a rejoiner never observes done-with-unsaved-state.
- **`/initialize` three-state contract.** Re-calling `/initialize` for an existing session
  returns the SAME startup execution's status (200 done / 202 running) or plain tokens if no
  startup turn exists.
- **Graceful shutdown** waits up to 30 s for active turns (__main__.py:340).

**Exception: `/invoke_agent_stream` uses a different, older concurrency guard.** It checks
`runtime.lock.locked()` (__main__.py:963) and emits an error event if busy; it does NOT go
through the TurnRegistry — no idempotency, no 202 deferral, destructive trace drain. Known
inconsistency (open; candidate for migration onto the registry). It also strips leading `/` from
the query, so you cannot force assistant mode through the streaming endpoint.

### MCP surface

`setup_mcp()` (mcp_specific.py:46-58) mounts fastapi-mcp via `mount_http()` at `/mcp` AFTER all
endpoints are defined. Exposed MCP tools: `invoke_agent` (maps to `/invoke_agent_stream` — that
endpoint owns the `invoke_agent` operation_id), `invoke_assistant`, `new_conversation`,
`get_all_conversations`, `post_feedback`, `activate_conversation`. Excluded: root,
dump_all_conversations, generate_mcp_token, rest_initialize, perform_action, rest_invoke_agent,
refresh_token. fastapi-mcp 0.4.0 does **not** support custom prompts (mcp_specific.py:60-64) —
any doc claiming two prompts are registered is wrong. MCP clients authenticate with long-lived
tokens from `/admin/generate_mcp_token`.

## 7. Security posture — stated plainly

The default server is **trusted-network only**. Do not expose it to the internet as-is.

| Fact | Evidence |
|---|---|
| Default is trusted mode: tokens are minted and required as Bearer headers, expiry and type are checked, but **signatures are NOT verified**; unsigned tokens use `alg=none` | `--expect_encrypted_jwt` defaults False (__main__.py:357-358), applied at startup (__main__.py:268); `jwt.decode(token, options={"verify_signature": False})` (jwt_manager.py:278-281); `alg="none"` encode (jwt_manager.py:199-204) |
| The CLI wrapper can NEVER enable verification | cli.py:454-471 builds the `python -m` argv without the flag |
| `/admin/generate_mcp_token` is unauthenticated and mints 365-day tokens; the code's own docstring says it "should be restricted to administrators only in production" | __main__.py:1543-1588 (no auth dependency; excluded from the Bearer scheme in `custom_openapi`, __main__.py:400) |
| `/admin/dump_all_conversations` is unauthenticated and exfiltrates every channel's conversation history to a file path of the caller's choosing | __main__.py:1488-1541 |
| CORS is `allow_origins=["*"]` with credentials allowed | __main__.py:413-419 |
| JWT constants (RS256, 60-min access, 30-day refresh, issuer/audience) are hardcoded, "can be made configurable via env vars" | jwt_manager.py:21-26 |
| Dependency CVEs: 21/22 fixed; ecdsa CVE-2024-23342 (Minerva timing attack) has NO patch. The report cites bead `fastworkflow-d8f` (line 86), but that ID is absent from `.beads/issues.jsonl` as of 2026-07-09 (report rot — see `fastworkflow-docs-and-positioning`), so the item is effectively **untracked**; filing a real bead is an owner decision for Dhar | `SECURITY_VULNERABILITY_REPORT.md` (repo root; note it covers dependency CVEs, not the server-posture items above — those are code-level design trade-offs) |

Net: anyone with network reach can forge a token (trusted mode), mint a year-long token, or dump
all conversations. Whether this is accepted risk for the actual deployment topology is an **open
question** recorded for Dhar; treat any hardening work as a change-control matter
(`fastworkflow-change-control`).

## 8. The shipped integrate-chat-agent skill

`fastworkflow/skills_for_coding_fastworkflows/integrate-chat-agent/` (SKILL.md + reference.md;
skill name `integrate-fastworkflow-chat-agent`; moved 2026-08-25 from
`fastworkflow/docs/integrate-chat-agent/` into the new coding-skills library, content revised in
the move) is a **wheel-shipped** coding-agent skill that
walks an application developer through AI-enabling their app: hand-written commands, training,
hosting run_fastapi_mcp, and building a popup streaming chat UI. README.md positions it as the
recommended integration path for non-trivial apps.

Caveats (verified 2026-07-09):
- Its `reference.md` documents the HTTP endpoint contract, but **nothing in tests/ verifies that
  contract against the actual routes** (grep for it in tests/ returns nothing). The planned
  fix-qtq wire change (endpoint responses cutting over from serialized `CommandOutput` to the
  `TurnOutput` public projection — a declared pre-v3.0 breaking change) **will silently break
  this shipped skill** unless someone updates reference.md in the same change. If you touch the
  wire shape, put reference.md on your checklist.
- reference.md also lists `LLM_RESPONSE_GEN`, which is dead config (zero consumers in the
  codebase; see `fastworkflow-config-and-flags`).

## 9. Known-stale operator docs — do not trust these

| Doc | What is wrong |
|---|---|
| `fastworkflow/run_fastapi_mcp/README.md` | **Materially stale.** Launch command `uvicorn services.run_fastapi.main:app` refers to a package layout deleted in v2.17 (line 31; real launch is the CLI or `python -m fastworkflow.run_fastapi_mcp`); documents obsolete "504 on timeout" semantics (line 360; v2.22.0 replaced that with 202 deferral); claims two MCP prompts are registered (line 348; fastapi-mcp 0.4.0 cannot). Trust this skill and the code, not that README. |
| `redoc.html` (repo root) + `run_fastapi_mcp/redoc_2_standalone_html.py` | Snapshot predates the turns rewrite; the generator imports `services.run_fastapi.main` and is broken. |
| CLI `--help` env-file default text | Says ".env in current directory"; code resolves `<workflow_path>/fastworkflow.env` (cli.py:177-191). |
| CLAUDE.md | Says intent models are trained "via scikit-learn" — actually torch/transformers (sklearn is only LabelEncoder/split/metrics). Doc rot, evidence in `fastworkflow-nlu-pipeline-reference`. |

## Provenance and maintenance

Facts verified 2026-07-09 against v2.22.2 (commit c33b9a5). Re-verification one-liners for every
volatile fact:

```sh
# CLI subcommands + flags
fastworkflow --help && fastworkflow run --help && fastworkflow run_fastapi_mcp --help
# Bundled examples list
fastworkflow examples list
# Default env-file resolution (workflow dir, not CWD)
grep -n -A4 "def find_default_env_files" fastworkflow/cli.py
# CLI does not forward --expect_encrypted_jwt
grep -n -A16 "cmd = \[" fastworkflow/cli.py | grep -c expect_encrypted   # expect 0
# Full endpoint list (compare with the table above)
grep -n -A2 "@app.get\|@app.post" fastworkflow/run_fastapi_mcp/__main__.py | grep '"/'
# No /turns polling endpoint yet (Step 2)
grep -rn '"/turns' fastworkflow/run_fastapi_mcp/ ; echo "expect no route hits"
# Turns engine: 202 body, idempotency, evict no-op
grep -n "202\|compute_idempotency_key\|ttl_expires_at" fastworkflow/run_fastapi_mcp/turns.py
# Streaming endpoint's divergent lock guard
grep -n "lock.locked()" fastworkflow/run_fastapi_mcp/__main__.py   # line ~963 inside invoke_agent_stream
# JWT trusted-mode default + alg=none + unverified decode
grep -n "expect_encrypted_jwt\|verify_signature\|algorithm=\"none\"" fastworkflow/run_fastapi_mcp/__main__.py fastworkflow/run_fastapi_mcp/jwt_manager.py
# Unauthenticated admin endpoints (excluded from Bearer scheme)
grep -n "admin/dump_all_conversations\|admin/generate_mcp_token" fastworkflow/run_fastapi_mcp/__main__.py
# CORS
grep -n -A2 "allow_origins" fastworkflow/run_fastapi_mcp/__main__.py
# MCP tool exclusions + no-prompts note
sed -n '40,65p' fastworkflow/run_fastapi_mcp/mcp_specific.py
# '/' prefix + //exit //new
grep -n "is_assistant_mode_command" fastworkflow/workflow_execution_context.py; grep -n '"//' fastworkflow/run/__main__.py
# Artifact writers
grep -n "def observability_db" -A 8 fastworkflow/state_paths.py
grep -rn "action.jsonl" fastworkflow/ --include=*.py   # expect only retirement comments
grep -n "___convo_info" fastworkflow/_workflows/command_metadata_extraction/intent_detection.py
grep -n "channel_conversations\|channel_session_state" fastworkflow/run_fastapi_mcp/utils.py
# Fingerprint-based cache invalidation (v2.22.1)
grep -n "source_fingerprint\|compute_commands_source_fingerprint" fastworkflow/command_directory.py
# Trained-artifact inventory for one example
ls fastworkflow/examples/hello_world/___command_info fastworkflow/examples/hello_world/___command_info/global
# Shipped integrate-chat-agent skill + its untested contract
ls fastworkflow/skills_for_coding_fastworkflows/integrate-chat-agent/; grep -rl "integrate-chat-agent" tests/ ; echo "expect no test hits"
# Stale README claims
grep -n "uvicorn services.run_fastapi\|504\|two prompts\|registers two" fastworkflow/run_fastapi_mcp/README.md
# Open-issue statuses cited here (read-only)
bd show fix-85g; bd show fix-qtq
# ecdsa bead reference is report rot: ID only in the report, not the tracker
grep -n d8f SECURITY_VULNERABILITY_REPORT.md; bd search ecdsa --json   # expect report hits only; empty search
```

Volatility notes: fix-85g Step 2 (a real `/turns` endpoint, TTL eviction, trace replay) and the
fix-qtq wire cutover are open work that will invalidate Section 6's "no polling" and Section 8's
contract-breakage warnings when they land — re-run the checks above before trusting either.

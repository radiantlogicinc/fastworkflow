---
name: fastworkflow-config-and-flags
description: >
  Load this skill whenever you touch fastWorkflow configuration: any LLM_* / LITELLM_API_KEY_* /
  SPEEDDICT_FOLDERNAME / INTENT_DETECTION_* / SESSION_STATE_* env var, the fastworkflow.env or
  fastworkflow.passwords.env files, a fastworkflow CLI subcommand or flag, LiteLLM proxy routing,
  or JWT server settings. Trigger symptoms: "env var not found" warnings, "SPEEDDICT_FOLDERNAME
  env var not found!", "DSPy Language Model not provided", shell exports being mysteriously
  ignored, refine failing with a missing-LM ValueError, or needing to add a new config knob.
  Do NOT use it for how to run/operate workflows end to end (fastworkflow-run-and-operate),
  recreating the dev environment and secrets (fastworkflow-build-and-env), or debugging
  non-config runtime failures (fastworkflow-debugging-playbook).
---

# fastWorkflow Configuration and Flags Catalog

Facts verified 2026-07-09 against v2.22.2 (commit c33b9a5). Every file:line below was
checked against the working tree on that date. If you are reading this later, run the
re-verification commands in "Provenance and maintenance" before trusting line numbers.

Jargon, defined once:
- **dotenv file** — a `KEY=value` text file parsed by `python-dotenv`'s `dotenv_values()`. It is
  NOT automatically exported to the process environment.
- **LiteLLM** — the multi-provider LLM client library; model strings look like
  `mistral/mistral-small-latest`. **LiteLLM Proxy** — a separate gateway server; model strings
  get the `litellm_proxy/` prefix and calls go to `LITELLM_PROXY_API_BASE`.
- **DSPy** — the LLM-programming library fastWorkflow uses for agent/planner/param-extraction
  calls; `dspy.LM` wraps a LiteLLM model string.
- **LLM role** — fastWorkflow assigns a separate model + API key pair per job (agent, planner,
  parameter extraction, synthetic data generation, ...), so each role can use a different model.

## When to use / when NOT to use

Use this skill when you need to:
- Look up any env var: its default, consumer, and sharp edges (Section 3).
- Understand why a shell `export` is ignored (Section 2 — the precedence trap).
- Look up a CLI subcommand/flag, including the ones the `--help` text lies about (Section 5).
- Add a new env var or CLI flag correctly (Section 7 checklist).

Use a sibling skill instead when you need:
- **fastworkflow-run-and-operate** — actually running workflows/the server; artifact layout; endpoints.
- **fastworkflow-build-and-env** — creating the dev environment from scratch; provisioning secrets.
- **fastworkflow-nlu-pipeline-reference** — what the LLM roles and intent models *do*.
- **fastworkflow-debugging-playbook** — symptom-first triage when you don't yet know it's config.
- **fastworkflow-change-control** — whether you are *allowed* to change a default.

## 1. The two-dotenv-file contract

Every workflow is configured by exactly two dotenv files, conventionally placed inside the
workflow folder:

| File | Contains | Template |
|---|---|---|
| `fastworkflow.env` | Model strings, framework settings, tuning knobs | `fastworkflow/examples/fastworkflow.env` |
| `fastworkflow.passwords.env` | `LITELLM_API_KEY_*` secrets only | `fastworkflow/examples/fastworkflow.passwords.env` |

Loading (verified in `run/__main__.py:125-128`, `train/__main__.py:269-271`,
`run_fastapi_mcp/__main__.py:259-264`):

```python
env_vars = {**dotenv_values(env_file_path), **dotenv_values(passwords_file_path)}
fastworkflow.init(env_vars=env_vars)
```

Rules that follow from this code:
- Passwords are merged AFTER the env file, so a key in the passwords file wins on conflict.
- `dotenv_values` never touches `os.environ`; config lives in an in-process dict
  (`fastworkflow._env_vars`, set at `fastworkflow/__init__.py:174-176`).
- `fastworkflow.init()` must run before using framework classes (module globals like
  `CommandContextModel` are `None` until then) and before importing modules that read env vars
  at import time (see Section 4 traps).
- `init()` also re-applies `LOG_LEVEL` from the dotenv dict (`__init__.py:180-182`).

When the CLI's env-file arguments are omitted, defaults resolve to
`<workflow_path>/fastworkflow.env` and `<workflow_path>/fastworkflow.passwords.env`
(`cli.py:177-191`). The `--help` text saying ".env in current directory" is WRONG — see Section 5.

For fetched examples, `fastworkflow examples fetch` copies the two templates to `./examples/`
(not into each example folder), so you must pass them explicitly:
`fastworkflow train ./examples/<name> ./examples/fastworkflow.env ./examples/fastworkflow.passwords.env`.

## 2. get_env_var precedence — the code default SHADOWS the OS environment

`fastworkflow.get_env_var(var_name, var_type=str, default=None)` at
`fastworkflow/__init__.py:211-227` resolves in this exact order:

1. `_env_vars` dict (the two dotenv files passed to `init()`)
2. the `default=` argument at the call site, **if one was provided**
3. `os.getenv(var_name)` — consulted ONLY when no code default exists
4. `None` + a logged warning

**Consequence (the trap):** for any var whose call site passes a non-None `default=`, a shell
`export VAR=...` is silently ignored — step 2 returns before `os.getenv` is ever consulted.
You can only override such vars via the env FILES. The shell-unoverridable vars as of v2.22.2
(call sites verified): `SESSION_STATE_STORE` (default `"disk"`, `session_state_store.py:123`),
`INTENT_DETECTION_TINY_MODEL` and `INTENT_DETECTION_LARGE_MODEL`
(`model_pipeline_training.py:892-895`). Put overrides for these in `fastworkflow.env`, never
in the shell.

Call sites that pass `default=None` (e.g. `SESSION_STATE_REDIS_URL`/`REDIS_URL` at
`session_state_store.py:127-129`, `LITELLM_PROXY_API_KEY` at `utils/dspy_utils.py:58`) do NOT
trigger the shadow — the guard is `if default is not None` — so those fall through to the OS
environment normally.

Type coercion: `var_type=bool` accepts only `true/1/false/0` (case-insensitive), else
`ValueError` (`__init__.py:232-238`). A provided `default` is returned as-is, uncoerced.

**Masking hazard:** importing `litellm` runs `load_dotenv()` over `./.env` in the CWD
(verified: `site-packages/litellm/__init__.py:29`). The repo root has a generated `.env`
(`make gen-env`), so import-time env bugs that would crash a customer deployment are invisible
when developing in-repo. This masked the v2.21.0-2.21.3 regression fixed in 79e6986.

## 3. Which subcommands load env files (and which ignore them)

| Subcommand | Env files loaded? | Evidence | Consequence |
|---|---|---|---|
| `build` | **NO** — `fastworkflow.init(env_vars={})` | `build/__main__.py:326,349` | LLM vars must be OS-environment exports (step 3 fallback works because call sites pass no default) |
| `refine` | **NO** — `init(env_vars={})` | `refine/__main__.py:32` | Same; no env-file CLI args exist at all |
| `train` | YES (both files) | `train/__main__.py:269-271` | — |
| `run` | YES (both files) | `run/__main__.py:125-128` | — |
| `run_fastapi_mcp` | YES (both files, in lifespan) | `run_fastapi_mcp/__main__.py:259-264` | Plus a pre-lifespan read of `LOG_LEVEL` for uvicorn (`:1612-1617`) |

So before `fastworkflow build` / `fastworkflow refine` you must:

```bash
export LLM_COMMAND_METADATA_GEN=mistral/mistral-small-latest
export LITELLM_API_KEY_COMMANDMETADATA_GEN=<your key>   # note: no underscore in COMMANDMETADATA
```

Nothing in `--help` tells you this; the failure mode is
`ValueError: DSPy Language Model not provided. Set LLM_COMMAND_METADATA_GEN environment variable.`
(raised from `utils/dspy_utils.py:44-45` via `build/genai_postprocessor.py:239`).

**File-presence probes at startup:**
- `run` HARD-FAILS if `LITELLM_API_KEY_SYNDATA_GEN` is absent (`run/__main__.py:131-132`) even
  though run never generates synthetic data — it is used purely as a "did the passwords file
  load?" probe. This breaks Bedrock/proxy users who legitimately have no such key; workaround:
  put a dummy value in the passwords file.
- `train` only WARNS for the same missing key ("OK if this is Bedrock",
  `train/__main__.py:276-277`) but hard-fails on missing `SPEEDDICT_FOLDERNAME` (`:273-275`),
  as does `run` (`run/__main__.py:129-130`).

## 4. Env var catalog

Legend: **prod** = production, load-bearing. **dead** = zero consumers in code.
"Default" = code default at the consumer (template value shown separately).

### 4a. LLM role variables (all resolved via `dspy_utils.get_lm` unless noted)

| Model var (template value: `mistral/mistral-small-latest`) | API key var | Consumer file:line | Role | Sharp edge |
|---|---|---|---|---|
| `LLM_AGENT` | `LITELLM_API_KEY_AGENT` | `workflow_agent.py:270,309`; `workflow_execution_context.py:691` | prod | — |
| `LLM_PLANNER` | `LITELLM_API_KEY_PLANNER` | `workflow_agent.py:507`; `workflow_execution_context.py:1019` | prod | Key var missing from the `cli.py:136-139` fallback passwords stub |
| `LLM_PARAM_EXTRACTION` | `LITELLM_API_KEY_PARAM_EXTRACTION` | `utils/signatures.py:252-255` | prod | Bypasses `get_lm`: direct `dspy.LM(model, api_key=...)` at `signatures.py:255`, values cached in module globals after first read. Because no `api_base` is passed, `litellm_proxy/` routing very likely does NOT work for this role (code-read inference, not runtime-verified) |
| `LLM_SYNDATA_GEN` | `LITELLM_API_KEY_SYNDATA_GEN` | `train/generate_synthetic.py:35-36`; `utils/generate_param_examples.py:333-334` (both call `litellm.completion` directly) | prod (train-time) | Key doubles as run-time file-presence probe (Section 3) |
| `LLM_CONVERSATION_STORE` | `LITELLM_API_KEY_CONVERSATION_STORE` | `run_fastapi_mcp/conversation_store.py:331` | prod (server) | Key var missing from the `cli.py` fallback stub |
| `LLM_COMMAND_METADATA_GEN` | `LITELLM_API_KEY_COMMANDMETADATA_GEN` | `build/genai_postprocessor.py:239` (used by `build` and `refine`) | prod (build-time) | (1) NOT in the packaged templates — only in `docs/genai_postprocessor_readme.md` and repo-local `env/.env` + `passwords/.env`; (2) key spelling is `COMMANDMETADATA` (no underscore) unlike every other key; (3) must be an OS export (Section 3) |
| `LLM_RESPONSE_GEN` | `LITELLM_API_KEY_RESPONSE_GEN` | **none** (rg over `fastworkflow/` + `tests/` `*.py`: zero hits) | **dead** | Templated at `examples/fastworkflow.env:8`, `examples/fastworkflow.passwords.env:9`, and written by the `cli.py:138` fallback stub — pure drift. Whether it is reserved for a future response-gen stage is an open question; do not delete without change control |

LiteLLM Proxy routing (`utils/dspy_utils.py:42-69`): if a model string starts with
`litellm_proxy/`, `get_lm` requires `LITELLM_PROXY_API_BASE` (raises `ValueError` if unset,
`:50-55`), optionally uses `LITELLM_PROXY_API_KEY` (default `None`, no-auth proxies allowed,
`:58`), and IGNORES the per-role `LITELLM_API_KEY_*`. The `[server]` extra is NOT needed for
proxy routing. Full recipe: [references/litellm-proxy-and-local-dev.md](references/litellm-proxy-and-local-dev.md).

### 4b. Framework and pipeline variables

| Var | Code default | Template value | Consumer file:line | Role | Sharp edge |
|---|---|---|---|---|---|
| `SPEEDDICT_FOLDERNAME` | none | `___workflow_contexts` (`fastworkflow.env:36`) | `workflow.py:360` (function cache); `session_state_store.py:138`; `run_fastapi_mcp/utils.py:388,399` | prod | Roots ALL disk state; hard-required by `run` and `train` (Section 3) |
| `SESSION_STATE_STORE` | `"disk"` | not templated | `session_state_store.py:123` | prod (server) | `redis` value can ONLY be set via env file (default shadows shell, Section 2) |
| `SESSION_STATE_REDIS_URL` / `REDIS_URL` | `None` | not templated | `session_state_store.py:127-129` | prod (server, redis only) | `ValueError` if `SESSION_STATE_STORE=redis` and neither is set |
| `INTENT_DETECTION_TINY_MODEL` | `google/bert_uncased_L-4_H-128_A-2` | commented (`fastworkflow.env:30`) | `model_pipeline_training.py:892-893` | prod (train-time) | Env-file-only override (Section 2). Defaults chosen for transformers 4.48+/5.x compat |
| `INTENT_DETECTION_LARGE_MODEL` | `distilbert-base-uncased` | commented (`fastworkflow.env:31`) | `model_pipeline_training.py:894-895` | prod (train-time) | Same |
| `SYNTHETIC_UTTERANCE_GEN_NUMOF_PERSONAS` | none (int) | `4` | `train/generate_synthetic.py:14` | prod (train-time) | **Import-time read trap**: all three are read as module globals when `generate_synthetic` is first imported. Import before `fastworkflow.init()` captures `None` (a warning is logged; downstream failure inferred, not runtime-verified). Never import train modules before `init()` |
| `SYNTHETIC_UTTERANCE_GEN_UTTERANCES_PER_PERSONA` | none (int) | `5` | `train/generate_synthetic.py:15` | prod (train-time) | Same trap |
| `SYNTHETIC_UTTERANCE_GEN_PERSONAS_PER_BATCH` | none (int) | `1` | `train/generate_synthetic.py:16` | prod (train-time) | Same trap |
| `MISSING_INFORMATION_ERRMSG` | none | `"Missing parameter values: "` | `_workflows/command_metadata_extraction/parameter_extraction.py:19` (import-time); `utils/signatures.py:324` | prod | Parameter-error handling string-matches on these values — changing them mid-deployment changes behavior |
| `INVALID_INFORMATION_ERRMSG` | none | `"Invalid parameter values: "` | `parameter_extraction.py:20`; `signatures.py:326` | prod | Same |
| `NOT_FOUND` | none | `"NOT_FOUND"` | `parameter_extraction.py:22` (import-time); `signatures.py:74,185,328`; `mcp_server.py:57`; example command files | prod | Sentinel value meaning "parameter not extracted"; commands compare against it |
| `INVALID` | none | `"INVALID"` | `parameter_extraction.py:23` (import-time) | prod | Sentinel |
| `PARAMETER_EXTRACTION_ERROR_MSG` | none | `"Error in parameter extraction: {error}"` | `parameter_extraction.py:263`; `signatures.py:249` (lazy, cached) | prod | Must keep the `{error}` placeholder |
| `LOG_LEVEL` | `INFO` | not templated | THREE paths: `utils/logging.py:53` (OS env at import, invalid value raises `ValueError`); `__init__.py:180-182` (dotenv, via `reconfigure_log_level`); `run_fastapi_mcp/__main__.py:1612-1617` (dotenv pre-read for uvicorn) | prod | Put it in `fastworkflow.env`; that covers paths 2 and 3. Shell export covers path 1 only |
| `FW_MODEL_CONTEXT_TOKENS` | model metadata, else 131072 | not templated | `context_budget.py` (env file first, then `os.environ`) | prod | **The one byte-budget input**: the model's context window in tokens. Every `FW_*_MAX_BYTES` budget is a fraction of it — see `docs/context_budget.md`. Unset, it is read from the `LLM_AGENT` model via `litellm.get_model_info`. |
| `FW_TRAJECTORY_MAX_BYTES`, `FW_ANSWER_REHYDRATION_MAX_BYTES`, `FW_RESULT_PAGE_MAX_BYTES`, `FW_SEARCH_ANSWER_MAX_BYTES`, `FW_OFFLOAD_HOT_MAX_BYTES`, `FW_RESULT_HANDLE_HOT_MAX_BYTES`, `FW_OFFLOAD_MIN_SAVING_BYTES` | derived from the window | not templated | `context_budget.py` | prod | **Tuning overrides, not the interface.** Below their floor or unparseable, the derived budget stands with a warning. |
| `FW_SEARCH_OBSERVATION_MAX_BYTES` | a quarter of the *search* window, capped at 131,072 (`SEARCH_OBSERVATION_CEILING_BYTES`) | commented (`fastworkflow.env`) | `context_budget.py` (`SEARCH_OBSERVATION`, `search_observation_max_bytes`; env file first, then `os.environ`) | prod | Added 2026-09-27 (fix-deus); before it the bound had no override. Bytes of one observation handed to the search model. **May exceed the ceiling**; below the 4,096 floor or unparseable, the derived value (not the floor) stands with a warning. The search window is the smaller of `FW_MODEL_CONTEXT_TOKENS` and `LLM_OBSERVATION_SEARCH`'s litellm metadata (the setting no longer wins outright; with `LLM_OBSERVATION_SEARCH` unset it is the agent's window). Reported in `budget_provenance()["overrides"]` when it moved the value. The ceiling halves the bound on the example `mistral/mistral-small-latest` (262,144-token metadata) |
| `FW_MAX_FORCED_REPLANS` | `3` (`MAX_FORCED_REPLANS`, so 4 segments) | commented (`fastworkflow.env`) | `observation_offloading/continuation.py` (`max_forced_replans_from_env`, read when each agent is built; env file first, then `os.environ`) | prod | Clamped to `[0, 10]`; a non-integer is ignored. Each correction warns once per raw value. The default rests on one ido task; it raises every workflow's worst-case step ceiling from 75 to 100. `forced_replan` / `forced_replan_wall` events carry `reached_limit`; `agent_installed` records the value in force |
| `FW_SEARCH_ROUTER` | off | commented (`fastworkflow.env`) | `observation_offloading/search_router.py:40`, via `jev_client.requested_key` | prod (opt-in) | Only the value `jev` turns it on, and only with `JEV_API_KEY` set and `typesafe-sdk` installed (`fastworkflow[jev]`; `[router]` is a deprecated alias). **Activation warnings**: a set flag that cannot take effect (other value / no SDK / no key) logs ONE warning per cause per process and stays off; unset is silent. **Data egress**: sends the search question, the agent's reasoning and the observation's first 4 lines to TypeSafe with only credential patterns scrubbed by default (nothing with `FW_OFFLOAD_EVIDENCE_REDACTION=off`). One attempt, 2 s timeout, fails open to the search model. **Since 2026-09-27**: the 2 s is a hard wall-clock cutoff; at most 3 routing calls per turn and 10 s of vendor time shared with the finish check, after which `router_budget` (no call, search model answers); same extra activation causes and registered-provider names as `FW_FINISH_CHECK` |
| `FW_SEARCH_ROUTER_MODEL` | `jev-1.13.0` | commented | `search_router.py:43` | prod (opt-in) | Router client cached per (workflow path, model) and key fingerprint |
| `FW_FINISH_CHECK` | off | commented (`fastworkflow.env`) | `observation_offloading/finish_check.py:66`, via `jev_client.requested_key`; also switches the planner to the structured planner (`workflow_agent.py`, `structured = checker_from_env() is not None`) | prod (opt-in) | Same activation rule and warnings as `FW_SEARCH_ROUTER`. **Data egress**: sends the user request, the plan and a per-step ledger (commands, contexts, subject names, output heads) with only credential patterns scrubbed. Turning it on also makes planning slower (structured planner: median 12.1 s vs 3.7 s plain text, three todo-list requests). 4 s per call, 8 s best-effort budget (checked before each call, so it can overrun by one call). Its tokens are outside `fw.llm.call` cost totals, as are the router's. `FW_EVAL_FINISH_REMINDERS=0` keeps it attached but records `reason: disabled` instead of checking. **Since 2026-09-27** (supersedes the planner and budget notes in this cell): the planner reads `workflow_agent.finish_check_active(agent)` (checker attached and reminders on), not `checker_from_env()`, and the structured planner runs only for the initial plan; the 4 s / 8 s bounds are hard wall-clock cutoffs, the 8 s include building the ledger, and the turn shares 10 s of vendor time with the router (module constants in `jev_client.py`, no settings). More activation causes (see `FW_JEV_BASE_URL`, and a capture profile that withholds command output or `FW_OFFLOAD_EVIDENCE_REDACTION=off`). The value may also name a decision provider registered from code (`decision.register_decision_provider`); then `JEV_API_KEY`, `FW_JEV_BASE_URL` and `FW_FINISH_CHECK_MODEL` are not read. **Since 2026-09-28** (supersedes every structured-planner note in this cell): structured planning is disabled -- its code is commented out, not deleted, and no flag re-enables it. With the check on the planner is plain text, exactly as with it off (so no planning slowdown), and the check verifies the plan `parse_text_plan` recovers from it, which has no subjects |
| `FW_FINISH_CHECK_MODEL` | `jev-1.13.0` | commented | `finish_check.py:69` | prod (opt-in) | Checker cached per model and key fingerprint (and, since 2026-09-27, endpoint). A value other than the default warns once per process: `FLAG_MIN`/`ASK_MIN` and the wording were calibrated on ido with the default (`CALIBRATION = "ido-v7-2026-09"`, on every `finish_check` event) |
| `FW_EVAL_FINISH_REMINDERS` | unset (reminders on) | commented (`fastworkflow.env`, since 2026-09-27) | `utils/react.py` (`_evaluation_control`, via `context_budget.env_value`: env file first, then `os.environ`) | eval control | Only `0` is accepted: it turns off everything the finish check adds (no note, plain-text planner) while the checker stays attached and records `reason: disabled`. Any other non-empty value makes the agent build raise `ValueError(... must be exactly 0 when set)`; empty counts as unset, whitespace is stripped. Until 2026-09-27 it was read from `os.environ` only and an empty value raised. Since 2026-09-28 (structured planning disabled) the planner is plain text either way; `0` still stops the text plan being parsed, persisted and checked |
| `FW_JEV_BASE_URL` | `https://api.typesafe.ai` (`jev_client.DEFAULT_BASE_URL`, equal to the SDK's default) | commented (`fastworkflow.env`) | `observation_offloading/jev_client.py` (`base_url()`; env file first, then `os.environ`) | prod (opt-in) | Added 2026-09-27 (fix-2so9). Receives the key and every payload of both Jev features. Must be `https`, or plain `http` only to a loopback address; no credentials, query, fragment or malformed port. A rejected value warns once (without the URL) and **both** features stay off; a non-default host is logged once at INFO. Not read when a flag names a registered provider |
| `TYPESAFE_BASE_URL` | — | not templated (mentioned as ignored) | `jev_client.base_url()` (`os.environ` only) | **ignored** | The SDK's own endpoint variable. Since 2026-09-27 fastWorkflow always passes the endpoint, so this is never used; when set, one warning names `FW_JEV_BASE_URL` instead |
| `JEV_API_KEY` | none | commented (`fastworkflow.passwords.env`) | `observation_offloading/jev_client.py:31` (`KEY_ENV`) | prod (opt-in) | Shared by both Jev features. **Deliberately not** `LITELLM_API_KEY_<ROLE>`: it is the vendor SDK's key, not a litellm role's, and nothing routes it through `get_lm`/`litellm_proxy/`. Never logged or stored; caches hold a 16-char SHA-256 fingerprint, so a changed key builds a new client |
| `PYTEST_RUNNING` | — | — | set by `tests/conftest.py:16`; **zero readers** in `fastworkflow/` | **dead** | Safe to ignore; do not build logic on it |

Unconditional behaviour, with no setting to change it: observability
recording is always on, for fastWorkflow's entry points and library embedders alike
(`get_observability_sink(workflow_path)` returns `None` only when the store cannot be
opened; the DB is owner-only, 0600 file / 0700 dir, under `FASTWORKFLOW_STATE_ROOT`, pruned
by `FW_OBS_RETENTION_DAYS` / `FW_OBS_DB_MAX_BYTES`). Offloading/search/answer-time events
are recorded in process (a ring of a fixed 2,000) and in the `offload_events` table, read
with `ObservabilityStore.offload_events(turn_key=..., channel_id=..., kind=...)`. The
unserializable-artifact validator always runs and only warns (a hard rejection in v3.0 per
the `warn_on_unserializable_artifacts` docstring). The `search_memory` input bound is
derived from the search model's context window only; correct it with
`FW_MODEL_CONTEXT_TOKENS`. (History since 2026-09-27: it now has its own
override, `FW_SEARCH_OBSERVATION_MAX_BYTES`, and a 131,072-byte ceiling on the
derived value — see its row above.) The `typesafe-sdk` DEBUG body logging is
always suppressed (a filter on the `typesafe_sdk` logger; no opt-in), and the
`[jev]` extra pins `typesafe-sdk` to exactly `0.7.2`.

Not fastWorkflow config, despite appearances: repo-root `config.yaml` is a Dolt SQL server
config for the beads issue tracker; repo-root `.env`, `env/.env`, `passwords/.env` are the
local dev convention (see [references/litellm-proxy-and-local-dev.md](references/litellm-proxy-and-local-dev.md)).

### 4c. Known drift summary (candidates for cleanup, gated by change control)

- `LLM_RESPONSE_GEN` + key: templated but unconsumed (dead).
- `LLM_COMMAND_METADATA_GEN` + key: consumed but untemplated, inconsistent key spelling.
- `cli.py:136-139` fallback passwords stub lists only 4 keys (SYNDATA_GEN, PARAM_EXTRACTION,
  RESPONSE_GEN, AGENT) — omits PLANNER, CONVERSATION_STORE, COMMANDMETADATA_GEN; includes the
  dead RESPONSE_GEN.
- `run`'s hard requirement on `LITELLM_API_KEY_SYNDATA_GEN` (probe misuse, Section 3).
- `JEV_API_KEY` breaks the `LITELLM_API_KEY_<ROLE>` naming on purpose (vendor SDK key, not a
  litellm role); kept as is — do not rename without change control.

## 5. CLI subcommands and flags

Entry point: `fastworkflow = "fastworkflow.cli:main"` (`pyproject.toml:31`). Six subcommands
(verified live with `fastworkflow --help` on 2026-07-09).

| Subcommand | Positional args | Flags | Notes |
|---|---|---|---|
| `examples list` | — | — | Lists bundled examples |
| `examples fetch` | `name` | `--force` | Copies example to `./examples/<name>`, copies/creates the two env templates in `./examples/` (`cli.py:104-139`) |
| `build` | — | `--app-dir/-s` (required), `--workflow-folderpath/-w` (required), `--overwrite`, `--stub-commands <a,b>`, `--no-startup` | Ignores env files (Section 3) |
| `refine` | — | `--workflow-folderpath/-w` (required) | Ignores env files; no env-file args exist |
| `train` | `workflow_folderpath [env_file_path] [passwords_file_path]` | — | Env defaults: `<workflow>/fastworkflow.env|.passwords.env` |
| `run` | `workflow_path [env_file_path] [passwords_file_path]` | `--context_file_path`, `--startup_command`, `--startup_action`, `--keep_alive` (default `True`), `--project_folderpath`, `--assistant` | Agent mode by default; `--assistant` = deterministic |
| `run_fastapi_mcp` | `workflow_path [env_file_path] [passwords_file_path]` | `--context <JSON>`, `--startup_command`, `--startup_action <JSON>`, `--project_folderpath`, `--port 8000`, `--host 0.0.0.0` | Re-execs `python -m fastworkflow.run_fastapi_mcp` as a subprocess (`cli.py:453-472`); requires the `[server]` extra (`cli.py:384-413`) |

**Help-text lie:** the env-file positionals for `train`/`run`/`run_fastapi_mcp` claim
"(default: .env in current directory, or bundled env file for examples)"
(`cli.py:239,245,259,265,288,294`). The actual default is
`<workflow_path>/fastworkflow.env` + `<workflow_path>/fastworkflow.passwords.env`
(`find_default_env_files`, `cli.py:177-191`). Trust the code.

**`--keep_alive` sharp edge:** defined with `default=True` and no `type=` (`cli.py:270`), so
any value you pass arrives as a string — `--keep_alive False` yields the truthy string
`"False"`. Effectively this flag cannot be turned off from the CLI (empty-string workaround
untested). Downstream signature is `keep_alive: bool` (`chat_session.py:150`).

**`--expect_encrypted_jwt` is unreachable from the wrapper:** it exists only on the module
server (`run_fastapi_mcp/__main__.py:358-359`, `action="store_true"`, default `False`), and the
CLI wrapper never forwards it (`cli.py:454-469`). Therefore:

```bash
# JWT signature verification DISABLED (trusted-network mode) — the only mode the wrapper offers:
fastworkflow run_fastapi_mcp ./examples/hello_world ./examples/fastworkflow.env ./examples/fastworkflow.passwords.env

# JWT signature verification ENABLED — must bypass the wrapper:
python -m fastworkflow.run_fastapi_mcp --workflow_path ./examples/hello_world \
  --env_file_path ./examples/fastworkflow.env \
  --passwords_file_path ./examples/fastworkflow.passwords.env \
  --port 8000 --expect_encrypted_jwt
```

Note the argument style changes: the module form uses `--workflow_path`/`--env_file_path`
flags (`load_args`, `run_fastapi_mcp/__main__.py:346-360`), not positionals.

## 6. JWT constants and jwt_keys/ — hardcoded, not env vars

`run_fastapi_mcp/jwt_manager.py:22-26` hardcodes (comment admits "can be made configurable via
env vars" — acknowledged unfinished work):

| Constant | Value |
|---|---|
| `JWT_ALGORITHM` | `RS256` |
| `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` | `60` |
| `JWT_REFRESH_TOKEN_EXPIRE_DAYS` | `30` |
| `JWT_ISSUER` / `JWT_AUDIENCE` | `fastworkflow-api` / `fastworkflow-client` |

- Keys live at `./jwt_keys/private_key.pem` (mode 600) and `public_key.pem` (mode 644),
  **relative to CWD** (`jwt_manager.py:29-31`, save at `:88-100`) — launch the server from a
  stable directory or you'll silently generate a fresh keypair. `jwt_keys/.gitignore` excludes
  the PEMs; never commit them.
- Module default `EXPECT_ENCRYPTED_JWT = True` (`jwt_manager.py:38`) is ALWAYS overwritten at
  startup by `set_jwt_verification_mode(ARGS.expect_encrypted_jwt)`
  (`run_fastapi_mcp/__main__.py:267-268`), i.e. effective default is False/unverified.
  Unsigned mode issues `alg="none"` tokens (`jwt_manager.py:203`).
- Deployment security posture (unauthenticated /admin endpoints, open CORS) is covered in
  **fastworkflow-run-and-operate**.

## 7. How to add a config axis (checklist)

Work through ALL of these; a var added to code but not the templates (or vice versa) becomes
the next `LLM_RESPONSE_GEN` / `LLM_COMMAND_METADATA_GEN` drift entry.

1. [ ] **Consumer**: read via `fastworkflow.get_env_var("MY_VAR", ...)` — never a scattered
   `os.environ` read (sanctioned exceptions: `LOG_LEVEL` at logger import, and
   `context_budget.env_value`, which checks the env file first and then `os.environ`
   precisely because `get_env_var` short-circuits on its default). Never call it at module import time — read lazily inside
   functions/properties (the 79e6986 lesson; litellm's cwd-`.env` load will hide the bug in-repo).
2. [ ] **Decide the precedence tier consciously**: passing `default=` to `get_env_var` makes
   the var shell-unoverridable (Section 2). If ops must be able to override via shell, pass no
   default and enforce requiredness at the CLI entry instead.
3. [ ] **LLM role?** Add BOTH `LLM_<ROLE>` and `LITELLM_API_KEY_<ROLE>` and resolve through
   `dspy_utils.get_lm("LLM_<ROLE>", "LITELLM_API_KEY_<ROLE>")` so `litellm_proxy/` routing
   works. Keep the `_` spelling consistent (do not imitate `COMMANDMETADATA`).
4. [ ] **Templates**: add to `fastworkflow/examples/fastworkflow.env` (settings) or
   `fastworkflow/examples/fastworkflow.passwords.env` (secrets), with a comment block.
5. [ ] **CLI fallback stub**: if it's an API key, add it to the stub writer at `cli.py:134-139`.
6. [ ] **build/refine reachability**: if the var is consumed on the build/refine path, document
   that it must be an OS export (those subcommands init with `env_vars={}`).
7. [ ] **This catalog**: add a row to Section 4 (or 4a) with consumer file:line and sharp edges,
   plus a re-verification grep in the Provenance section.
8. [ ] **Docs**: update `docs/genai_postprocessor_readme.md` / relevant doc of record if applicable
   (see fastworkflow-docs-and-positioning).
9. [ ] **Tests**: remember tests init from repo-local `./env/.env` + `./passwords/.env`
   (`tests/test_command_executor.py:24-27`); add your var there if tests need it, and re-run
   `make gen-env` mentally — it merges every `*.env` in the tree into root `.env`.
10. [ ] **Change control**: renaming/removing an existing var is a breaking change
    (deployed env files reference old names — e.g. the `COMMANDMETADATA` spelling is frozen by
    existing installs until proven otherwise). Route through fastworkflow-change-control.
    Do NOT commit or push any of this without the developer'sexplicit request in that turn.

Run `scripts/audit_env_catalog.sh` (read-only) after any config change: it re-derives consumed
vs templated vars and prints drift.

## Provenance and maintenance

Facts verified 2026-07-09 against v2.22.2 (commit c33b9a5). Re-verification one-liners (run from repo root):

- Full consumer catalog: `grep -rn "get_env_var(" fastworkflow/ --include="*.py" | grep -v "def get_env_var"`
- Direct OS reads (should stay ~3): `grep -rnE "os\.environ|os\.getenv" fastworkflow/ --include="*.py" | grep -v examples/`
- LLM role call sites: `grep -rn "get_lm(" fastworkflow/ --include="*.py" | grep -v "def get_lm"`
- Precedence logic: `sed -n '211,227p' fastworkflow/__init__.py`
- build/refine empty init: `grep -n "init(env_vars={})" fastworkflow/build/__main__.py fastworkflow/refine/__main__.py`
- run/train startup probes: `grep -n "SPEEDDICT_FOLDERNAME\|LITELLM_API_KEY_SYNDATA_GEN" fastworkflow/run/__main__.py fastworkflow/train/__main__.py`
- LLM_RESPONSE_GEN still dead: `grep -rn "LLM_RESPONSE_GEN" fastworkflow/ tests/ --include="*.py"` (expect zero hits)
- COMMANDMETADATA spelling: `grep -rn "COMMANDMETADATA" fastworkflow/ --include="*.py"`
- Import-time reads: `sed -n '14,16p' fastworkflow/train/generate_synthetic.py; sed -n '19,23p' fastworkflow/_workflows/command_metadata_extraction/parameter_extraction.py`
- Intent model defaults: `sed -n '890,896p' fastworkflow/model_pipeline_training.py`
- CLI flags + help-text lie + env defaults: `sed -n '177,302p' fastworkflow/cli.py` and `fastworkflow <subcommand> --help`
- `--expect_encrypted_jwt` reachability: `grep -n "expect_encrypted_jwt" fastworkflow/cli.py fastworkflow/run_fastapi_mcp/__main__.py` (expect: absent from cli.py)
- JWT constants + CWD-relative keys: `sed -n '21,40p' fastworkflow/run_fastapi_mcp/jwt_manager.py`
- litellm dotenv masking: `grep -n "load_dotenv" .venv/lib/python*/site-packages/litellm/__init__.py`
- PYTEST_RUNNING still dead: `grep -rn "PYTEST_RUNNING" fastworkflow/ tests/ --include="*.py"` (expect conftest.py only)
- Templates: `cat fastworkflow/examples/fastworkflow.env fastworkflow/examples/fastworkflow.passwords.env`
- Or run everything at once: `bash .claude/skills/fastworkflow-config-and-flags/scripts/audit_env_catalog.sh`

---
name: fastworkflow-debugging-playbook
description: >
  Symptom-to-triage playbook for fastWorkflow runtime, build, and train failures.
  Load this when you see: the wrong command being matched, "threshold.json not found" /
  FileNotFoundError in ___command_info, the internal CME workflow retraining on every
  `fastworkflow train`, parameter extraction looping on NOT_FOUND, the CLI hanging or
  never printing an agent question, FastAPI 409/202 surprises or "poll /turns" advice,
  DeprecationWarning about process_message, commands with too few training utterances,
  stale/frozen LLM outputs (DSPy cache), RocksDB/speedict lock errors, env vars that
  "don't take effect", or tests that mysteriously pass/fail/skip. Do NOT load this for
  writing new tests (fastworkflow-validation-and-qa), env-var reference
  (fastworkflow-config-and-flags), historical post-mortems (fastworkflow-failure-archaeology),
  or tau2 benchmark work (tau2-reliability-campaign).
---

> TEAM-PRIVATE: embeds content from uncommitted internal docs. Do not commit or publish this skill without the developer'sexplicit approval.

# fastWorkflow Debugging Playbook

Runbook for diagnosing fastWorkflow failures. Written against **v2.22.2 (commit c33b9a5)**.
Every file:line below was verified against that tree; re-verification one-liners are in
[Provenance and maintenance](#provenance-and-maintenance).

## When to use / when NOT to use

**Use** when something is broken and you need symptom → cause → discriminating experiment → fix.

**Do NOT use — go to a sibling instead:**

| Need | Sibling skill |
|---|---|
| Full incident chronicles (root cause + evidence + status) | `fastworkflow-failure-archaeology` |
| What every env var / CLI flag does | `fastworkflow-config-and-flags` |
| How to run CLI/examples/FastAPI server normally | `fastworkflow-run-and-operate` |
| Measurement scripts and interpretation guides | `fastworkflow-diagnostics-and-tooling` |
| Intent/param-extraction internals (models, thresholds, DSPy) | `fastworkflow-nlu-pipeline-reference` |
| Adding tests, what counts as evidence | `fastworkflow-validation-and-qa` |
| Change gating / non-negotiables before you fix anything | `fastworkflow-change-control` |
| tau2/tau-bench experiment work (E0–E25) | `tau2-reliability-campaign` |

## Jargon (defined once)

- **Workflow**: a directory with `_commands/*.py` command files; fastWorkflow wraps your app with it.
- **CME**: the internal `command_metadata_extraction` workflow (in `fastworkflow/_workflows/`) that runs intent detection + parameter extraction as a pseudo-command named `wildcard`. Every session owns a private CME workflow instance.
- **WEC**: `WorkflowExecutionContext` (`fastworkflow/workflow_execution_context.py`) — the transport-free execution core. One user message in, one turn out.
- **Topology A / B**: A = CLI `ChatSession` with queues + a worker thread (blocking `ask_user`); B = bare WEC embedded by FastAPI (`ask_user` suspends and resumes on the next message).
- **`___command_info/`**: per-workflow trained artifacts — `command_directory.json`, `routing_definition.json`, `<cmd>_param_labeled.json`, and per-context model folders (`tinymodel.pth/`, `largemodel.pth/`, `label_encoder.pkl`, `threshold.json`).
- **`___convo_info/`**: per-workflow runtime NLU caches (learned utterance→command mappings, suggested commands) in RocksDB (`speedict.Rdict`) files.
- **Fingerprint**: sha256 over the set of `(path, size, mtime_ns)` of all command sources, stamped into the two JSON artifacts (v2.22.1) so stale snapshots are rebuilt.
- **NOT_FOUND**: sentinel string (from env var `NOT_FOUND`) meaning "parameter not extracted yet"; extraction control-flow branches on it.
- **DSPy**: LLM-programming library used for parameter extraction and the agent; it caches LLM calls on disk (`~/.dspy_cache` for dspy 3.2.1).

## Universal first moves

1. `source .venv/bin/activate` — always (AGENTS.md rule; wrong interpreter = phantom failures).
2. Set `LOG_LEVEL=DEBUG` **in the workflow's `fastworkflow.env` file**, not the shell — `get_env_var` defaults shadow the OS environment (see trap T11).
3. Know which phase you are in: **build** (AST codegen, no LLM), **train** (utterance gen + BERT fine-tune → `___command_info/`), **run** (intent → params → execute). A "run" failure is often a train-phase artifact problem.
4. Read `fastworkflow-change-control` before touching anything under `fastworkflow/examples/*/___command_info` or running git/bd commands.

## Master triage table

| # | Symptom | Most likely cause | Discriminating experiment | Fix / workaround |
|---|---|---|---|---|
| T1 | Wrong command matched, or import error naming a command file that no longer exists | (a) stale persisted JSON snapshots (pre-v2.22.1 artifacts), (b) in-process `lru_cache` staleness in a long-running process, (c) a learned bad mapping in `___convo_info`, (d) genuine misclassification | Compare `compute_commands_source_fingerprint()` output vs the `source_fingerprint` field inside both JSONs; check process uptime vs file edit time; grep the utterance in the `___convo_info` cache | (a) auto-fixed on next load in ≥2.22.1; (b) restart the process or call `RoutingRegistry.clear_registry()`; (c) delete `___convo_info/`; (d) retrain / add plain_utterances |
| T2 | Crash or FileNotFoundError on `___command_info/<ctx>/threshold.json`; first message fails | Untrained workflow (or a test/experiment wiped the trained model — fix-0hb) | `python -c` call to `is_workflow_trained()` (below) | `fastworkflow train <wf> <env> <passwords>`; for the bundled hello_world: `fastworkflow train ./fastworkflow/examples/hello_world ./env/.env ./passwords/.env` |
| T3 | `fastworkflow train <your-workflow>` also retrains fastWorkflow's internal CME workflow, every single time | `is_fast_workflow_trained` checks `___command_info/ErrorCorrection/largemodel.pth`, but training has excluded the `ErrorCorrection` context since v2.7.0 — the file is never produced, so the check can never pass | `ls fastworkflow/_workflows/command_metadata_extraction/___command_info/` → only `IntentDetection` and `global` model folders exist | Known mismatch; verified in code 2026-07-09 (see T3 detail). No fix on main. Workaround: none safe; do NOT hand-create the folder. Budget the extra CME train time |
| T4 | Parameter extraction loops: keeps replying "missing information" with fields stuck at NOT_FOUND | Field metadata too weak for the LLM extractor, agent parroting `examples` values (deliberately rejected), or the sentinel env vars changed | Try the same command with explicit `<param>value</param>` XML (agent mode parses this by regex before any LLM call); check `MISSING_INFORMATION_ERRMSG`/`NOT_FOUND` env values are the template defaults | Improve `Field(description=, examples=, pattern=)` in the command's `Signature.Input`; in deterministic mode type `abort` to escape the state |
| T5 | CLI spinner hangs forever in agent mode when the agent asks the user a question | **OPEN, NEEDS RUNTIME VERIFICATION** — fix-5fv hypothesis: blocking `ask_user` enqueues the question on the output queue with **no trace sentinel**, and the CLI drains the trace queue for a sentinel before reading output | Run any agent-mode CLI session and force a clarification (omit a required param). If it hangs: hypothesis confirmed — update bd fix-5fv | No shipped fix. Workaround: `--assistant` mode, or `/`-prefix the command (deterministic path emits its sentinel at turn end) |
| T6 | FastAPI returns 409 on a retry; or 202 with a `turn_key` and docs say "poll /turns" but GET /turns 404s | 409 = a *different* turn is already active on the channel (by design). 202 = deferred execution; **GET /turns does not exist — fix-85g Step 2 is unbuilt** | `grep -n "@app.get" fastworkflow/run_fastapi_mcp/__main__.py` — routes are `/`, `/probes/*`, `/conversations` only | Recover a 202 by **re-sending the byte-identical request** — the idempotency key rejoins the same execution. Never treat `runtime.lock.locked()` as the busy signal (see T6 detail) |
| T7 | `DeprecationWarning: WorkflowExecutionContext.process_message() is deprecated` in CLI runs | Expected: `ChatSession` itself still calls `process_message` internally | `grep -n "process_message" fastworkflow/chat_session.py` → lines 361, 374, 452, 456 | Harmless; not your bug. New embedder code should call `process_turn()` |
| T8 | An intent model is weak for one specific command; that command has almost no training utterances | `generate_diverse_utterances` returns `[]` on litellm `RateLimitError` — **dropping even the seed utterances** for that command, silently (one `logger.error` line) | grep the train log for `LiteLLM Rate limiting error!`; inspect utterance counts per command in `___command_info/command_directory.json` | Re-run `fastworkflow train` when not rate-limited; consider a slower model/key. No retry logic exists on main |
| T9 | LLM outputs look frozen — refine/agent/param-extraction returns identical answers despite prompt/code changes | DSPy disk/memory cache serving stale completions | Delete or disable the cache and re-run; if output changes, it was the cache | `rm -rf ~/.dspy_cache` (dspy 3.2.1 location) or `dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)`. WARNING: `docs/DSPY_CACHE_GUIDE.md` is partly rotten (see T9 detail) |
| T10 | RocksDB errors ("IO error ... LOCK", "Resource temporarily unavailable") or mysterious `LOG`/`.sst` files appearing | Two processes/threads opened the same `Rdict` (RocksDB is single-writer), or you are looking at normal RocksDB internals | `fuser`/`lsof` on the `.db` dir; check for a second server/test process on the same workflow | Ensure one writer per channel/workflow; kill the other process. `LOG`/`.sst`/`MANIFEST` files inside `___convo_info` and `___workflow_contexts` are normal RocksDB internals, all gitignored — never commit, safe to delete only with the owning process stopped |
| T11 | An env var you exported in the shell is silently ignored (e.g. `SESSION_STATE_STORE`, `INTENT_DETECTION_TINY_MODEL`, `LOG_LEVEL`) | `get_env_var(var, type, default)` returns the code default **without consulting `os.environ`** whenever a default is supplied | Put the var in the workflow's `fastworkflow.env` and re-run: if it now takes effect, you hit the precedence quirk | Always configure via the env files. Shell exports only work for vars whose call sites pass no default |
| T12 | JWT weirdness: tokens accepted that shouldn't be, or `alg=none` in decoded tokens | Default server mode is *trusted*: signatures NOT verified, unsigned tokens use JWT `alg="none"`; the `fastworkflow run_fastapi_mcp` CLI **cannot** enable verification (flag not forwarded) | Decode a minted token; check server launch command | For signed JWTs run `python -m fastworkflow.run_fastapi_mcp ... --expect_encrypted_jwt` directly. Treat default deployments as trusted-network only |
| T13 | Test suite: tests "pass" on a fresh clone but are mostly skips; or 4 FastAPI tests fail with missing threshold.json; or the suite is ~4 min slower than it should be | Env-gated skips (need `./env/.env` + `./passwords/.env` + locally pre-trained hello_world); fix-0hb-class model wipe; deliberate 0.5 s sleep after every test | `python -m pytest -rs` and read skip reasons; check `fastworkflow/examples/hello_world/___command_info/global/threshold.json` exists | Provision env files, pre-train hello_world (T2 command). Never "fix" the sleep in `tests/conftest.py:111-118` without solving ChatWorker thread drain |

## Trap details

### T1 — Wrong command matched / stale command caches

Intent detection tries, in order (`fastworkflow/_workflows/command_metadata_extraction/intent_detection.py:99-133`):
exact first-token match → Levenshtein fuzzy match (threshold 0.3) → embedding cache match at 0.85 cosine
(`cache_match`, `fastworkflow/cache_matching.py:131`) → transformer `CommandRouter.predict`. Distinguish four causes:

1. **Stale persisted snapshots** (fixed in v2.22.1, commit b5747df): before the fix, `routing_definition.json`
   had *no* staleness check and `command_directory.json` used newest-mtime (blind to deletions/renames);
   add/remove/rename of a command caused import errors until artifacts were hand-deleted.
   Now `compute_commands_source_fingerprint` (`fastworkflow/command_directory.py:627`) stamps both JSONs and
   `RoutingRegistry._persisted_definition_is_fresh` (`fastworkflow/command_routing.py:385`) forces rebuild on mismatch.
   *Experiment:* check the `source_fingerprint` field in both JSONs matches a fresh
   `compute_commands_source_fingerprint(<wf>)` call. A missing field = pre-2.22.1 artifact → rebuilt on next load.
2. **In-process cache staleness** (still open): `get_cached_command_directory` is `@lru_cache(maxsize=32)`
   (`fastworkflow/command_directory.py:657-658`) and `RoutingRegistry.get_definition` returns
   `cls._definitions[path]` when present (`fastworkflow/command_routing.py:357-360`) — **the fingerprint is only
   checked on cache miss**. Editing commands while a server/CLI process is running is never picked up.
   *Experiment:* restart the process; if behavior changes, this was it.
   *Fix:* restart, or `fastworkflow.RoutingRegistry.clear_registry()` (clears definitions, the global command-class
   cache, the command-directory lru_cache, and the module import cache — `command_routing.py:410-425`).
   Note the fingerprint embeds absolute paths + `mtime_ns`, so moving a workflow dir or a fresh checkout triggers a
   harmless JSON rebuild (not a retrain).
3. **Learned clarification cache**: a successful "you misunderstood"-style clarification stores the ORIGINAL
   utterance → resolved command with its embedding (`intent_detection.py:151-163` calling `store_utterance_cache`),
   so future messages ≥0.85 similar skip the classifier. One bad learned mapping = persistently wrong command.
   *Experiment/fix:* delete the workflow's `___convo_info/` directory (regenerated empty; you lose learned mappings).
4. **Genuine misclassification**: only after 1–3 are excluded. Then it is an NLU-quality problem —
   see `fastworkflow-nlu-pipeline-reference` (thresholds, wildcard negative class, utterance economics).

### T2 — Missing threshold.json / untrained-workflow late failures

`threshold.json` per context folder is the marker of a trained intent model. Guards exist at two entry points
(`fastworkflow/run/__main__.py:156-172` and `fastworkflow/mcp_server.py:249-250` — both call
`is_workflow_trained`, `fastworkflow/model_pipeline_training.py:624`), added in v2.19.0 (commit d7853a1) after
untrained workflows crashed on the *first message* instead of at startup. If you still see a *late* failure, you
came in through a path without the guard (e.g. embedding WEC directly, or a test).
*Story:* the fix-0hb incident (commit fa97b48) — a training test's teardown `rmtree`'d the real
`fastworkflow/examples/hello_world/___command_info`, poisoning the NEXT suite run with exactly this error.
**Discipline rule (non-negotiable):** never let tests/experiments train into or delete the bundled examples'
`___command_info` — copytree into a temp dir first (pattern: `tests/test_train_modern_stack.py:129-139`).

Quick check:
```bash
source .venv/bin/activate && python -c "
from fastworkflow.model_pipeline_training import is_workflow_trained
print(is_workflow_trained('<workflow_path>'))"
```

### T3 — CME workflow retrains on every `fastworkflow train`

Verified state on c33b9a5 (2026-07-09):
- `is_fast_workflow_trained` (`fastworkflow/train/__main__.py:225-256`) requires
  `_workflows/command_metadata_extraction/___command_info/ErrorCorrection/largemodel.pth`.
- `train()` excludes `ErrorCorrection` from CME training (`fastworkflow/model_pipeline_training.py:821`,
  literally `- {'ErrorCorrection', 'ErrorCorrection'}` — note the duplicate literal), so that folder is never produced;
  the repo's CME `___command_info/` contains only `IntentDetection/` and `global/`.
- Therefore the check returns False and `train_main` (`train/__main__.py:283-287`) retrains the internal CME
  workflow before your workflow **whenever the target path does not contain the substring `"fastworkflow"`**
  (the guard is a substring test — it also *skips* CME training for any user path that happens to contain
  "fastworkflow", a separate sharp edge).
This mismatch dates to v2.7.0 and is **not fixed on main**; whether it is an accepted cost or an unnoticed
regression is an open question for Dhar (also flagged in discovery). Do not "fix" it ad hoc — the correct check
is a design decision (fingerprint vs mtime vs context list); file/attach a bd issue first (one bd write at a
time, verify `.beads/issues.jsonl` after — bd's `close --reason` silently failed to persist on an older bd, observed 2026-06-11; it persists on 1.1.2, retested 2026-08-13).

### T4 — Parameter extraction NOT_FOUND loops

Mechanics: partial params are stored with NOT_FOUND sentinels via `model_construct` (no validation)
(`fastworkflow/_workflows/command_metadata_extraction/parameter_extraction.py:93-104`); validation error messages
embed the `MISSING_INFORMATION_ERRMSG` / `INVALID_INFORMATION_ERRMSG` env-var text (`fastworkflow/utils/signatures.py:634`),
so changing those env values changes user-facing errors AND any code that matches on them.
Landmine: the string-splitting recovery helper `_extract_missing_fields` (`parameter_extraction.py:140-160`) is
**dead code with an arity bug** (3-tuple unpack of `validate_parameters`' 4-tuple return) — it raises ValueError
if revived; do not call it.
Agent mode first tries XML-regex extraction (`_extract_parameters_from_xml`, `parameter_extraction.py:296+`) and
**deliberately rejects** the result when a value equals one of the field's `examples` (anti-parroting,
`parameter_extraction.py:330-337`); rejected → falls back to DSPy LLM extraction, whose `BestOfN(N=3)` reward
also scores 0.0 for echoed example values (`fastworkflow/utils/signatures.py`). So an example value that is also a
*legitimate* input can never be extracted in agent mode — pick example values that are not real data.
Loop-breaker in deterministic mode: type `abort` (or `you misunderstood`) — these escape hatches are matched by
exact/fuzzy utterance only, never by the classifier. Agent exhaustion caps at `max_iters=25`
(`fastworkflow/workflow_agent.py:387`) → `failure_reason="max_iters_exhausted"`.

### T5 — ask_user suspension oddities (incl. the OPEN fix-5fv CLI hang)

Two topologies, two behaviors:
- **Topology B (FastAPI/WEC):** `ask_user` raises `AskUserSuspend` (`fastworkflow/utils/react.py:18`) — a
  **BaseException** subclass so the ReAct loop's `except Exception` cannot swallow it; WEC returns an
  `awaiting_user` output and the *next* message resumes the same logical turn. A message during suspension is the
  answer — it never starts a new turn. `CommandCancelledError` (`workflow_execution_context.py:42`) is BaseException
  for the same reason. **Never convert these to Exception subclasses or add broad `except BaseException`.**
- **Topology A (CLI):** `_ask_user_tool` (`fastworkflow/workflow_agent.py:354-385`) puts the question on
  `command_output_queue` and blocks on `user_message_queue.get()`.

**fix-5fv (P2, OPEN, code-reading evidence only — NEEDS RUNTIME VERIFICATION, never runtime-reproduced):**
the blocking path emits **no trace sentinel**; the only sentinel emitter is `_maybe_enqueue_trace_sentinel`
(`workflow_execution_context.py:656`), called only at turn-END sites. The CLI (`fastworkflow/run/__main__.py:231-260`)
drains the trace queue until a `None` sentinel BEFORE reading the output queue → predicted deadlock: spinner spins,
question never prints. Filed 2026-06-11 during the R43 design review. If you hit T5, you are the runtime
verification — record the result in bd fix-5fv. (The bead cites pre-refactor line numbers; current ones are above.)
Related invariant: every mid-turn clarification enqueue must be paired with a trace sentinel (design rule A19).

### T6 — FastAPI 409 / 202 semantics (turns engine, v2.22.0)

Contract (module docstring, `fastworkflow/run_fastapi_mcp/turns.py:1-30` — treat as law):
- **wait-or-defer, never wait-or-abort**: a request timeout never cancels the execution (it runs as its own asyncio.Task).
- **200** = finished within the wait window; **202** = still running, body carries `turn_key`; **409** = a *different*
  turn is active on this channel (`ChannelBusyError` → 409 at `run_fastapi_mcp/__main__.py:607-614`).
- Idempotency key = sha256(channel_id + kind + normalized args) (`turns.py:~84-96`): a byte-identical retry
  **rejoins** the same execution. That is the ONLY recovery path today: **GET /turns does not exist** (docstrings
  at `__main__.py:652` say "poll /initialize or /turns" — Step 2 of fix-85g is unbuilt; the only GET routes are
  `/`, `/probes/healthz`, `/probes/readyz`, `/conversations`). `TurnRegistry.evict_terminal` is a documented no-op
  (`turns.py:226-243`), so terminal executions accumulate in memory until restart.
- The busy signal is the registry's **per-channel active-execution pointer, never `runtime.lock.locked()`** — the
  lock is released while deferring and across AWAITING_USER. Story: the fix-85g 504 race (commit cf3eeae) — a 504
  unwound `async with runtime.lock`, a retry passed the lock-based guard, and two executions raced / tokens returned
  with silently-empty startup output. Known survivor of the old pattern: `/invoke_agent_stream` still checks
  `runtime.lock.locked()` (`__main__.py:963`) and bypasses the registry — expect inconsistent busy behavior on that
  endpoint (open question whether intentional).
- Restart loses all in-flight executions (Step 3 durability unbuilt).

### T8 — Silent utterance drops on RateLimitError during train

`fastworkflow/train/generate_synthetic.py:120-122`: on `litellm.exceptions.RateLimitError` the function logs one
error line and `return []` — the command trains with **neither synthetic nor seed utterances** (contrast the
no-`datasets` path at `:26-32`, which correctly returns `[command_name] + seed_utterances`). Result: a silently
underrepresented class in the intent model; the train run still "succeeds". Always grep train logs for
`LiteLLM Rate limiting error!` before trusting a freshly trained model.

### T9 — DSPy cache confusion

`docs/DSPY_CACHE_GUIDE.md` is the doc of record but **partly rotten** (verified 2026-07-09): it imports
`fastworkflow.run_agent.agent_module` (module deleted) and invokes `dspy_cache_utils.py` by bare filename —
the utility itself **does exist**, at `fastworkflow/utils/dspy_cache_utils.py`, so the guide's commands fail
as written from repo root. Run it as:
```bash
.venv/bin/python -m fastworkflow.utils.dspy_cache_utils status   # or clear-disk
```
(matches `fastworkflow-diagnostics-and-tooling` and `fastworkflow-nlu-pipeline-reference`). As a CLI only
`status` and `clear-disk` are useful — `clear`/`reset` call `dspy.configure_cache` inside a fresh process
and cannot affect any other running process.
What still works from the guide:
```python
import dspy
dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)
```
```bash
rm -rf ~/.dspy_cache   # verified location for the installed dspy 3.2.1
```
Symptom signature: you changed a prompt/signature/field description but outputs are byte-identical across runs.
Also remember tests and the flaky "LLM-training routing test" (accepted-flaky per commit 033132f message) can be
cache-sensitive.

### T10 — speedict/RocksDB notes

Where Rdict still lives (post v2.21.4 hot-path removal, commit 6880b0a): the `enablecache` decorator
(`fastworkflow/workflow.py:57`), NLU caches (`fastworkflow/cache_matching.py:67,148`,
`intent_detection.py:200-274`), the conversation store (`run_fastapi_mcp/conversation_store.py:45`).
All open/close per operation because **RocksDB is single-writer** — two concurrent writers on one DB error out
(hence the "one writer per channel / sticky routing" rule; enforcement is open issue fix-5ka).
`LOG`, `*.sst`, `MANIFEST-*`, `CURRENT`, `LOCK` files inside `___convo_info/`, `___workflow_contexts/` are normal
RocksDB internals (you will find such droppings under `tests/todo_list_workflow/___workflow_contexts/` — on-disk
only, gitignored, untracked). Do not commit them; delete only when no process has the DB open. A leftover `LOCK`
after a crash: confirm no live process (`lsof +D <dbdir>`) before removing the directory.

### T11 — env sharp edges

`get_env_var` (`fastworkflow/__init__.py:211-219`): checks the init()-loaded dict first; if absent **and a default
was supplied, returns the default without consulting `os.environ`**. Consequences:
- Vars with code defaults (`SESSION_STATE_STORE`, `INTENT_DETECTION_TINY_MODEL`, `INTENT_DETECTION_LARGE_MODEL`)
  can only be overridden via the env files.
- `build`/`refine` call `fastworkflow.init(env_vars={})` — they read **no env files**; their LLM vars
  (`LLM_COMMAND_METADATA_GEN`, `LITELLM_API_KEY_COMMANDMETADATA_GEN` — note the no-underscore spelling) must be OS
  exports, unlike train/run.
- Never read env vars at import time in new code: the v2.21.3 regression (commit 79e6986) crashed the server when
  `SPEEDDICT_FOLDERNAME` lived only in the env file, and was *masked in dev* by litellm's `load_dotenv` of the cwd `.env`.
- `run` hard-fails if `LITELLM_API_KEY_SYNDATA_GEN` is missing (`run/__main__.py:131-132`) — it is used as a
  file-presence probe, so Bedrock/proxy users hit it even though run does no synthetic generation.

### T13 — test-suite gotchas (debugging context only; authoring → `fastworkflow-validation-and-qa`)

- No pytest config anywhere; run `python -m pytest` from repo root in the venv. It takes the better part of an hour — and the per-test thread drain below accounts for minutes of that — so a run you think has hung usually has not. Test count and the current full-run baseline: `fastworkflow-validation-and-qa`.
- Key-gated skips: FastAPI/topology-B/probes/MCP tests skip unless `./env/.env` AND `./passwords/.env` exist
  (`tests/test_fastapi_service.py:26-36` pattern) — a green run on a fresh clone is mostly skips.
- 4 FastAPI tests additionally require a locally pre-trained hello_world example (T2 command). fix-0hb
  (commit fa97b48) is the incident; the temp-copy training pattern is the law.
- `tests/conftest.py:111-118` sleeps 0.5 s after EVERY test (ChatWorker thread drain) — ~4 min fixed overhead; a
  "hung" suite is usually just this.
- `tests/test_streaming_endpoint.py` silently passes with no server running (bare `return` on connection error) —
  zero signal; ignore its "pass".
- Bundled `fastworkflow/examples/fastworkflow.passwords.env` contains a dead placeholder key that litellm rejects
  with AuthenticationError; only repo-local `./passwords/.env` has real keys. Placeholder heuristic: values
  containing `<` or `your-` are not keys.

## Cache-layer map (what to clear for which symptom)

| Layer | Scope | Clear with | Symptom it explains |
|---|---|---|---|
| `RoutingRegistry._definitions` + `_GLOBAL_COMMAND_CLASS_CACHE` + `get_cached_command_directory` lru | process | `RoutingRegistry.clear_registry()` or restart | edits to `_commands/` invisible in a live process (T1b). Note: command-class cache is keyed `command_name:module_type` with NO workflow path — command names must be unique per process |
| `command_directory.json` / `routing_definition.json` | disk, fingerprint-guarded | delete files (auto-rebuilt) — safe | stale routing pre-2.22.1 (T1a) |
| `___convo_info/` Rdict caches | disk, per workflow | delete directory | learned wrong intent mapping (T1c), stale suggested-commands |
| `___command_info/` model folders | disk, per workflow | **re-run `fastworkflow train`** (never hand-delete parts; deleting the whole folder forces full retrain) | wrong-era models; NOTE: failed retrain deliberately keeps prior artifacts runnable (prune happens only after success) |
| DSPy disk/memory cache | user-global | `rm -rf ~/.dspy_cache` / `dspy.configure_cache(...)` | frozen LLM outputs (T9) |
| `_WORKFLOW_REGISTRY` weakref session registry | process | drop references / restart; holding a reference IS the session | "session survived longer/shorter than expected"; CLI has NO cross-restart resume since v2.21.4 |

## Discipline rules that bind you while debugging

1. **Never `git commit`/`push` without the developer'sexplicit request in that turn** — rule established 2026-07-08 after a
   private doc was auto-pushed to this PUBLIC repo, forcing a history rewrite. No "session-close protocol" overrides it.
2. **One bd write at a time**; verify `.beads/issues.jsonl` changed after each. `bd close --reason`
   silently failed to persist on an older bd (2026-06-11); it persists on 1.1.2 (persisted 12/12 on bd 1.1.2, retested 2026-08-13 — see change-control Rule 2; verify the JSONL regardless).
3. **Never wipe/train-into `fastworkflow/examples/*/___command_info`** (fix-0hb, commit fa97b48) — temp copies only.
4. **tau-bench parity is sacred**: never modify tau-bench tools/tasks for benchmark runs; disclose any nonstandard
   trade (e.g. pinned simulator temperature) — never silent. (Benchmark debugging → `tau2-reliability-campaign`.)
5. Doc-rot precedent: trust code over docs. Known rot: CLAUDE.md says intent models are "via scikit-learn"
   (actually torch/transformers fine-tuning; sklearn is only LabelEncoder/split/metrics) and omits
   `tests/todo_list_workflow`; `docs/DSPY_CACHE_GUIDE.md` cites the deleted `run_agent.agent_module` and
   mislocates `dspy_cache_utils.py` (really at `fastworkflow/utils/`); `run_fastapi_mcp/README.md`
   documents the pre-turns 504 behavior.

## Provenance and maintenance

**Facts verified 2026-07-09 against v2.22.2 (commit c33b9a5).** fix-5fv and the T3 CME-retrain finding were
re-verified in source on that date; both remain code-level findings without a runtime reproduction (T5) or an
owner ruling (T3). Re-verify volatile facts before relying on them:

```bash
# T1: fingerprint + lru cache + freshness check still where cited
grep -n "def compute_commands_source_fingerprint\|@lru_cache" fastworkflow/command_directory.py   # expect 627, 657
grep -n "_persisted_definition_is_fresh\|def clear_registry" fastworkflow/command_routing.py       # expect 368/385, 410
# T2: fail-fast guard
grep -n "is_workflow_trained" fastworkflow/run/__main__.py fastworkflow/model_pipeline_training.py
# T3: ErrorCorrection mismatch still present?
grep -n "ErrorCorrection" fastworkflow/train/__main__.py fastworkflow/model_pipeline_training.py
ls fastworkflow/_workflows/command_metadata_extraction/___command_info/                            # ErrorCorrection absent?
# T4: sentinel parsing + anti-parroting
grep -n "MISSING_INFORMATION_ERRMSG\|examples and extracted_value" fastworkflow/_workflows/command_metadata_extraction/parameter_extraction.py
# T5: fix-5fv still open? sentinel emitter unchanged?
bd show fix-5fv | head -5
grep -n "_maybe_enqueue_trace_sentinel\|def _ask_user_tool" fastworkflow/workflow_execution_context.py fastworkflow/workflow_agent.py
# T6: GET /turns still missing? stream endpoint still on lock.locked()?
grep -n "@app.get\|lock.locked()" fastworkflow/run_fastapi_mcp/__main__.py
# T7: ChatSession still on process_message?
grep -n "self._core.process_message" fastworkflow/chat_session.py
# T8: RateLimitError still drops seeds?
grep -n -A2 "RateLimitError" fastworkflow/train/generate_synthetic.py
# T9: dspy cache dir for the installed version; cache utility still where cited
python -c "from dspy.clients import DISK_CACHE_DIR; print(DISK_CACHE_DIR)"
ls fastworkflow/utils/dspy_cache_utils.py
# T11: get_env_var precedence
sed -n '211,220p' fastworkflow/__init__.py
# T12: CLI still not forwarding the JWT flag?
grep -n "expect_encrypted_jwt" fastworkflow/cli.py || echo "still not forwarded"
# T13: per-test sleep
sed -n '111,118p' tests/conftest.py
# open-issue statuses cited here
bd show fix-5fv fix-85g fix-5ka 2>/dev/null | grep -E "OPEN|CLOSED"
```

When any command above stops matching, fix this file in the same change — a wrong runbook is worse than none.

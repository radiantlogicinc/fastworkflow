---
name: fastworkflow-diagnostics-and-tooling
description: >
  Load this skill when you need to MEASURE a fastWorkflow workflow instead of
  eyeballing it — inspect ___command_info trained artifacts, check whether
  cached command snapshots are stale (fingerprint FRESH/STALE), capture or read
  a turn's command traces / turn records from observability.sqlite3,
  smoke-test intent-classifier accuracy,
  configure LOG_LEVEL, inspect the DSPy LLM cache, or hit the server probe
  endpoints. Trigger phrases/symptoms: "why did it rebuild", "is this workflow
  trained", "what did the agent actually do", "traces are empty", "stale cache",
  "threshold.json", "confusion between commands", "cache hit hides the LLM
  call". Do NOT load for diagnosing a specific failure end-to-end (use
  fastworkflow-debugging-playbook), for NLU/model theory (use
  fastworkflow-nlu-pipeline-reference), or for statistics like pass^k variance
  (use fastworkflow-proof-and-analysis-toolkit).
---

# fastWorkflow diagnostics and tooling — measure, don't eyeball

Runbook for observing what a fastWorkflow workflow actually did and what state
its artifacts are actually in. Ships four tested, read-only scripts in
`scripts/` plus a map of every observability surface the framework already
has. All paths are repo-relative to the fastWorkflow repo root; all facts
verified against v2.22.2 (commit c33b9a5).

## When to use / when NOT to use

Use this skill when you need to:
- dump/inventory a workflow's `___command_info/` (commands, utterance counts,
  thresholds, model files, DSPy few-shot artifacts)
- decide "rebuild vs stale vs fine" for `command_directory.json` /
  `routing_definition.json` (the v2.22.1 fingerprint)
- capture or read a turn's command trace: live CLI traces, the per-workflow
  `observability.sqlite3` (turns + spans + artifacts), the server response's
  `traces` field, `TurnOutput`
- get numbers on a trained intent classifier (per-command accuracy on seeds,
  confusion pairs, ambiguity/fallback rates)
- control logging verbosity, inspect the DSPy cache, or use the K8s probes

Do NOT use this skill for — go to the sibling instead:

| Need | Sibling skill |
|---|---|
| symptom → root-cause triage of a failure | fastworkflow-debugging-playbook |
| how the BERT two-tier/threshold/DSPy pipeline works | fastworkflow-nlu-pipeline-reference |
| env var / CLI flag catalog and defaults | fastworkflow-config-and-flags |
| starting the CLI / FastAPI+MCP server, artifact locations | fastworkflow-run-and-operate |
| what counts as evidence, adding tests | fastworkflow-validation-and-qa |
| pass^k / variance / attribution math | fastworkflow-proof-and-analysis-toolkit |
| running tau2 experiments E0–E25 | tau2-reliability-campaign |
| whether a change is allowed at all | fastworkflow-change-control |

## Ground rules (non-negotiable)

- **Everything here is read-only.** The shipped scripts never write. When a
  diagnosis leads you to retrain, train into a **temp copy** of the workflow —
  never let anything (tests, experiments, teardown) touch
  `fastworkflow/examples/*/___command_info`. A training test once destroyed
  the pre-trained example models other tests silently depended on (the fix-0hb
  incident, fixed in commit fa97b48).
- Never `git commit`/`push` diagnostics output or anything else without Dhar's
  explicit request in that turn (team rule since 2026-07-08).
- Use the repo venv interpreter: `.venv/bin/python`. `import fastworkflow`
  takes ~5–10 s (pulls dspy/torch); scripts note where that cost applies.

## The four scripts

All live in `.claude/skills/fastworkflow-diagnostics-and-tooling/scripts/`.
Each supports `--json` for machine-readable output (except trace_turn, whose
inputs are already JSON). Full real observed outputs with healthy-vs-sick
walkthroughs: [references/observed-outputs.md](references/observed-outputs.md).

| Script | Question it answers | Cost |
|---|---|---|
| `inspect_command_info.py <wf>` | what is in `___command_info/`: commands, utterance counts, parameterized?, `*_param_labeled.json` valid/rejected counts, per-context model files present/missing, trained threshold values, fingerprint FRESH/STALE | instant with `--no-fingerprint`, else +~7 s import |
| `check_cache_freshness.py <wf>` | will the JSON snapshots be trusted or rebuilt, and why — recomputes the v2.22.1 fingerprint, prints verdict + the 5 most recently modified command sources (the usual culprit) | ~7 s (imports fastworkflow) |
| `trace_turn.py {turns,turn,response,explain}` | list recorded turns / print one turn's span tree + artifacts from `observability.sqlite3`; read a saved server turn-response body; `explain` prints the capture recipes per topology | instant (stdlib only) |
| `collect_model_metrics.py <wf> [--context C]` | per-command top-1 accuracy, confusion pairs, ambiguity rate, DistilBERT-fallback rate of a TRAINED context, scored on stored seed utterances with the exact runtime decision rule | ~10–60 s/context (loads 2 transformer models, CPU ok) |

Copy-paste smoke test (works offline, no API keys, uses the bundled internal
workflow which is trained on any machine that has ever run `fastworkflow
train`):

```bash
cd /path/to/fastworkflow
S=.claude/skills/fastworkflow-diagnostics-and-tooling/scripts
.venv/bin/python $S/inspect_command_info.py  fastworkflow/_workflows/command_metadata_extraction
.venv/bin/python $S/check_cache_freshness.py fastworkflow/_workflows/command_metadata_extraction
.venv/bin/python $S/trace_turn.py explain
.venv/bin/python $S/collect_model_metrics.py fastworkflow/_workflows/command_metadata_extraction
```

### Interpretation: inspect_command_info.py

| Output pattern | Meaning | Next action |
|---|---|---|
| `UNTRAINED contexts: ... global` | built but never trained here (trained models are never shipped in the wheel and `___command_info` is not committed for examples) | `fastworkflow train` (temp copy if bundled example) |
| `IntentDetection: NO MODEL FOLDER` on an app workflow | normal — internal CME contexts are trained inside fastworkflow's bundled command_metadata_extraction workflow, not per app | nothing |
| `ErrorCorrection: NO MODEL FOLDER` | normal — ErrorCorrection commands are matched only by exact/fuzzy utterance match, never the classifier | nothing |
| `param_labeled ... MISSING` on a `params=yes` command | DSPy few-shot examples absent → parameter extraction runs zero-shot, silently worse | retrain; check `LITELLM_API_KEY_SYNDATA_GEN` |
| `MISSING: largemodel.pth` (or any model file) in a context marked complete elsewhere | partial artifacts — interrupted train or manual deletion; `CommandRouter` will crash loading this context | retrain that workflow |
| thresholds look weird (`large_amb=0.0`) | legitimate on tiny label sets: ambiguous threshold = mean confidence of that model's misclassifications; zero misclassifications → 0.0 | nothing, unless paired with bad metrics |
| `plain: 0` across app commands | command files declare empty `plain_utterances` | add seeds / run `fastworkflow refine` |

### Interpretation: check_cache_freshness.py

| Verdict | Meaning | Next action |
|---|---|---|
| FRESH | snapshots trusted as-is on next load | nothing |
| STALE | some command source changed/moved/re-stat'ed since stamping; next load rebuilds JSONs (cheap, automatic) | check the newest-sources list; if commands actually changed since last train, the classifiers have label drift → retrain |
| STALE but sources byte-identical | fingerprint embeds absolute paths + mtime_ns — fresh checkout, moved dir, Docker COPY, `touch` | spurious rebuild, harmless; expected on every fresh clone |
| LEGACY | pre-v2.22.1 snapshot without a fingerprint | will rebuild; fine |
| MISSING | never built here / info dir deleted | `fastworkflow train` |

Key asymmetry to keep straight: the fingerprint guards only the two JSON
snapshots. The trained models (`*.pth`, thresholds) are **not**
fingerprint-guarded and only change on an explicit `fastworkflow train`
(which prunes orphaned artifacts only AFTER success, so a failed retrain
leaves the previous models runnable).

### Interpretation: trace_turn.py (and turn records generally)

Trace vocabulary, defined once:
- **CommandTraceEvent** — in-memory event emitted around every agent tool call
  (`workflow_agent.py:130-139, 209-218`): AGENT_TO_WORKFLOW (raw command text
  the agent sent) then WORKFLOW_TO_AGENT (resolved command_name, parameters,
  response_text, success). A `None` sentinel terminates each turn's stream.
- **observability.sqlite3** — the per-workflow single source of truth since
  Phase 7, at `$FASTWORKFLOW_STATE_ROOT/workflows/<workflow-id>/`. Holds
  `turns` (one row per logical turn, keyed by `turn_key`), `spans`
  (`trace_id` == `turn_key`), and `artifacts`. Written by `SQLiteTraceSink`
  for every topology, CLI included, and it **keeps every turn** — read it with
  `trace_turn.py turns` / `trace_turn.py turn <turn_key>`.
- **`ctx.action_log`** — the in-process, per-turn list of tool-call
  `{command, command_name, parameters, response}` and ask_user
  `{agent_query, user_response}` records. Cleared at turn start and between
  distillation passes; live-only, so use it in-process and use the DB for
  anything post-mortem. Its old CWD mirror, `action.jsonl`, was **retired in
  Phase 7** — a copy still on disk is pre-Phase-7 residue, not current data.
- **TurnOutput** — returned by `ctx.process_turn(message)`; carries `turn_key`
  (the developer observability handle for one logical turn, stable across
  ask_user suspensions), `status`, `failure_reason`, `answer`,
  `command_outputs`. `success` = all commands succeeded and is deliberately
  orthogonal to `status` (turn.py:196-216).
- **Server `traces` field** — the FastAPI turns engine drains the trace queue
  destructively into the turn response (`run_fastapi_mcp/turns.py:293-296`);
  key only present when non-empty.

| Pattern | Meaning | Next action |
|---|---|---|
| `success: false` + `status: completed` + confident answer | a command failed and the agent masked it in prose | find the failing `command_outputs` entry; fastworkflow-debugging-playbook |
| `failure_reason: max_iters_exhausted` | agent hit max_iters=25 | read traces for the loop |
| empty `traces` on a retry | destructive drain already consumed them (fix-85g Step 2 open) | treat traces as read-once; persist the first response |
| ask_user entry with `success: false` | question still unanswered, NOT an error (A7 role inversion: parameters=agent's question, response=user's answer, duration_ms=user think time) | resume the turn |
| `action.jsonl` present in a CWD | pre-Phase-7 residue; the mirror is retired and nothing writes it now | ignore it; read `observability.sqlite3` via `trace_turn.py turns` |
| `trace_turn.py turn` finds spans but no turn row | the turn-record write degraded, or the turn is still in flight | check writer health; re-read after the turn completes |

### Interpretation: collect_model_metrics.py

Scores are on SEED utterances — training data, so near-perfect is the
*expected baseline*. Observed healthy reference (CME workflow): global context
top1=1.0; IntentDetection top1=0.9677 with one real confusion
(`go_up -> reset_context`).

| Pattern | Meaning | Next action |
|---|---|---|
| top1 ≥ ~0.95, few confusions | healthy | nothing |
| top1 < ~0.9 on seeds | broken/mismatched artifacts or label drift | `check_cache_freshness.py`; retrain |
| recurring confusion pair | genuinely overlapping seed phrasings; at runtime becomes the ambiguity-clarification flow | differentiate `plain_utterances`, retrain |
| high ambiguous_rate | thresholds above typical confidence → frequent clarification prompts | compare thresholds vs mean_confidence |
| everything → `wildcard` | out-of-scope class swallowed commands; artifacts likely from a different command set | retrain |
| sklearn `InconsistentVersionWarning` on LabelEncoder | version skew between training sklearn and current venv (observed: 1.7.0 pickle in 1.9.0 — benign here) | if labels look shuffled, retrain |

**Honest gap:** true held-out F1 is not cheaply computable. Synthetic training
utterances (litellm + PersonaHub) are generated at train time and never
persisted; training's own test-split metrics (score = f1 × ndcg × (1 − 0.15 ×
distil_usage%/100), model_pipeline_training.py:222-258) exist only in
train-time stdout. If you need them, capture train logs. No target quality
bars are documented anywhere — treat acceptable-F1 thresholds as an **open**
question (candidate work item for the tau2 program).

## The observability map (what already exists, where)

### Logging

- Logger name `fastWorkflow`, configured at import in
  `fastworkflow/utils/logging.py` (nanosecond timestamps).
- Set `LOG_LEVEL` (DEBUG/INFO/WARNING/ERROR/CRITICAL) in the workflow's
  `fastworkflow.env` — `fastworkflow.init()` reconfigures the logger from the
  dotenv dict (`__init__.py:180-182`), fixing the old trap where import-time
  config ignored env-file values. `run_fastapi_mcp` additionally pre-reads the
  env file to set uvicorn's log level (`run_fastapi_mcp/__main__.py:1611-1617`).
- Noise policy: DSPy/LiteLLM/httpx/etc. loggers are force-quieted at
  `utils/logging.py:104-114` (LiteLLM to CRITICAL). If you need LLM-call
  detail, raise those explicitly in your harness.
- `FW_EAGER_ARTIFACT_VALIDATION=0` (OS env, read at `turn.py:155`) silences
  the v3.0 artifact-serializability deprecation warnings.

### Live CLI traces (agent mode)

`fastworkflow run` renders CommandTraceEvents while the spinner runs — dim
yellow `Agent >` (raw command), dim yellow/green `Workflow >` (resolved
command + response), dim orange for failed steps. Methodology and event model:
`docs/agent_mode_live_command_traces.md` — the design doc for this feature;
its event model and rendering rules match v2.22.2, but note the drift: the
proposed `--no_agent_traces` flag and `FASTWORKFLOW_SHOW_AGENT_TRACES`
override were never implemented (traces are always on; verified by grep), and
action-log records are appended by `workflow_agent.py`, not
`command_executor.py` as §5 claims (§5 also describes the action.jsonl mirror,
retired in Phase 7). Consumer loop:
`fastworkflow/run/__main__.py:229-267`.
Known open bug fix-5fv (code-reading evidence only, never reproduced): the CLI
spinner may hang on agent-mode ask_user because the clarification is enqueued
without the trace sentinel.

### In-process capture (the pattern for experiment harnesses)

```python
import queue, fastworkflow
fastworkflow.init(env_vars={...})                     # required before framework classes
ctx = fastworkflow.WorkflowExecutionContext(run_as_agent=True)
trace_q = queue.Queue()
ctx.set_transport_queues(command_trace_queue=trace_q) # queues are injected, not ctor args
# ... start workflow ...
out = ctx.process_turn("user message")                # TurnOutput
out.turn_key, out.status, out.success, out.command_outputs
ctx.action_log                                        # this turn's records (live only)
# drain trace_q until the None sentinel for live events
# post-mortem, by turn_key: trace_turn.py turn <out.turn_key>
```

Duck-typing contract: anything passed to CommandExecutor/workflow_agent as
`chat_session_obj` must expose `get_active_workflow` / `cme_workflow` /
`command_trace_queue` / `append_action_log` / `append_turn_output` /
`append_ask_user_entry` / `complete_ask_user_entry`; ChatSession and
WorkflowExecutionContext both do. Missing members are gracefully no-op'ed —
i.e. turn capture is **silently dropped**, so a custom embedder that "works"
but shows empty traces is missing this surface.

### FastAPI server

- Turn endpoints (`/invoke_agent`, `/invoke_assistant`, `/initialize`,
  `/perform_action`) return `turn_key`, `exec_state`, `status`, `success`,
  `answer`, `command_responses`, `command_outputs`, and `traces` (when
  non-empty) — shapes in `run_fastapi_mcp/turns.py:render_turn_response`.
  202 `{turn_key, exec_state:"running"}` = deferred; re-POST the identical
  body to rejoin; different body → 409.
- `/invoke_agent_stream` streams trace events as NDJSON (or SSE) but is NOT
  integrated with the turns registry (no idempotent rejoin) — inconsistent by
  known omission.
- Probes: `GET /probes/healthz` → `{"status": "alive"}`; `GET /probes/readyz`
  → `{"status": "ready"|"not_ready", "checks": {...}}` (fastworkflow
  initialized + workflow path valid). Both are access-log-suppressed when 200
  (`__main__.py:93-122`), so absence from logs is normal.

### DSPy cache

Default: disk+memory cache at `~/.dspy_cache` (observed), LiteLLM cache off.
Inspect: `.venv/bin/python -m fastworkflow.utils.dspy_cache_utils status`.
Cache hits skip the API **and** `dspy.inspect_history()` — clear before
debugging LLM behavior. Details, real output, and the doc-rot warning about
`docs/DSPY_CACHE_GUIDE.md` (references a nonexistent `fastworkflow.run_agent`
module): [references/observed-outputs.md](references/observed-outputs.md) §5.

### Known observability gaps (open at v2.22.2 — do not design around them silently)

| Gap | Status |
|---|---|
| traces are read-once (destructive drain); no replay buffer | fix-85g Step 2, open |
| no `GET /turns/{turn_key}` polling endpoint despite docstring references | fix-85g Step 2, open |
| TurnResult persistence/metrics layer (ConversationTurnStore, MetricsSink, admin CLI) | designed (docs/turn_result_design_final.md), unbuilt |
| train-time model metrics not persisted anywhere | gap; capture stdout |
| CLI ask_user spinner hang | fix-5fv, open, unverified |

## Provenance and maintenance

Facts verified 2026-07-09 against v2.22.2 (commit c33b9a5), by reading the
cited sources and by executing every script in this skill against
`fastworkflow/_workflows/command_metadata_extraction`,
`fastworkflow/examples/hello_world` (read-only), and
`tests/example_workflow`. Observed outputs in references/ are real except the
one labeled SYNTHETIC.

Re-verification one-liners for volatile facts:

| Fact | Re-verify with |
|---|---|
| fingerprint fields/coverage | `sed -n '603,655p' fastworkflow/command_directory.py` |
| freshness gate | `sed -n '384,407p' fastworkflow/command_routing.py` |
| snapshot JSON keys | `python -c "import json;print(list(json.load(open('tests/example_workflow/___command_info/command_directory.json'))))"` |
| model artifact filenames per context | `ls fastworkflow/_workflows/command_metadata_extraction/___command_info/global/` |
| param_labeled naming | `grep -n "param_labeled" fastworkflow/train/__main__.py` |
| CommandRouter decision rule / thresholds | `sed -n '289,335p' fastworkflow/model_pipeline_training.py` |
| trained-check semantics | `grep -n "def is_workflow_trained" -A 15 fastworkflow/model_pipeline_training.py` |
| action-log lifecycle + clear points | `grep -rn "clear_action_log\|append_action_log" fastworkflow/*.py` |
| observability DB path + schema | `grep -n "def observability_db" -A 8 fastworkflow/state_paths.py; grep -n "_SCHEMA_STATEMENTS" -A 45 fastworkflow/observability_store.py` |
| trace event emission sites | `grep -n "CommandTraceEvent" fastworkflow/workflow_agent.py fastworkflow/workflow_execution_context.py` |
| server trace dict keys | `grep -n "_format_trace_event" -A 14 fastworkflow/run_fastapi_mcp/utils.py` |
| turn response body keys | `grep -n "def render_turn_response" -A 40 fastworkflow/run_fastapi_mcp/turns.py` |
| destructive drain still unfixed | `grep -n "Destructive trace drain" fastworkflow/run_fastapi_mcp/turns.py` |
| LOG_LEVEL reconfigure path | `grep -n "reconfigure_log_level" fastworkflow/__init__.py fastworkflow/utils/logging.py` |
| probe endpoints | `grep -n "probes/" fastworkflow/run_fastapi_mcp/__main__.py \| head` |
| DSPy cache utility | `.venv/bin/python -m fastworkflow.utils.dspy_cache_utils status` |
| open-issue statuses (fix-85g, fix-5fv) | `bd show fix-85g; bd show fix-5fv` (read-only) |
| version/commit baseline | `grep -n '^version' pyproject.toml; git log --oneline -1` |

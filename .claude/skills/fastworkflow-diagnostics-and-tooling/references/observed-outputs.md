# Observed outputs: healthy vs sick

Every output below was actually produced on 2026-07-09 against v2.22.2
(commit c33b9a5) on the maintainer's machine, using only bundled/test
workflows and locally present trained artifacts. No training was run, no API
keys were used, nothing was written. Where an output is synthetic (one case,
labeled), the shape was verified against the rendering code.

Run everything with the repo venv interpreter: `.venv/bin/python`.
`import fastworkflow` costs ~5-10 s (dspy/torch); the metrics script also
loads two transformer checkpoints per context (~10-60 s on CPU).

---

## 1. inspect_command_info.py

### Healthy trained workflow (internal CME workflow)

```
$ .venv/bin/python .claude/skills/fastworkflow-diagnostics-and-tooling/scripts/inspect_command_info.py \
    fastworkflow/_workflows/command_metadata_extraction

Commands (7):
  command                                       plain  tmpl params  param_labeled
  ErrorCorrection/abort                             8     0     no  n/a
  ErrorCorrection/you_misunderstood                 9     0     no  n/a
  IntentDetection/go_up                             7     0     no  n/a
  IntentDetection/reset_context                     2     0     no  n/a
  IntentDetection/what_can_i_do                    12     0     no  n/a
  IntentDetection/what_is_current_context           3     0     no  n/a
  wildcard                                          7     0     no  n/a
  core commands: IntentDetection/go_up, IntentDetection/reset_context, ...

Routing contexts:
  ErrorCorrection: 7 commands
  *: 5 commands
  IntentDetection: 5 commands

Model artifacts per context ('global' == the '*' context):
  ErrorCorrection: NO MODEL FOLDER (declared_in_routing=True)
  IntentDetection: complete | thresholds: conf=0.42909760276476544 tiny_amb=0.42909760276476544 large_amb=0.0
  global: complete | thresholds: conf=0.4128608295792027 tiny_amb=0.40101778507232666 large_amb=0.5467640906572342

  UNTRAINED contexts (declared but no threshold.json): ErrorCorrection

Fingerprint: command_directory.json=FRESH routing_definition.json=FRESH
  live fingerprint: 1d1f44b8b38e1764256b9a5a44dcb4df84570867878f7a9e44fce9bee46f01d9
```

Why this is healthy despite two oddities:

- `ErrorCorrection: NO MODEL FOLDER` is **by design**. ErrorCorrection commands
  (`abort`, `you_misunderstood`) are matched only by exact/fuzzy plain-utterance
  match, never by the classifier (intent_detection.py:79-97), and train()
  explicitly excludes ErrorCorrection when training the CME workflow
  (model_pipeline_training.py:821). This script only flags it here because the
  target IS the CME workflow itself; for app workflows the script suppresses
  internal CME contexts from the UNTRAINED list.
- `large_amb=0.0` for IntentDetection: the large-model ambiguous threshold is
  the mean confidence of DistilBERT's misclassifications on the test split
  (analyze_model_confidence); if DistilBERT misclassified nothing, the
  threshold is 0.0 and every DistilBERT prediction counts as confident.
  Legitimate for tiny label sets.

### Healthy trained app workflow (hello_world, locally trained)

```
$ ... inspect_command_info.py fastworkflow/examples/hello_world --no-fingerprint
  add_two_numbers                                   3     0    yes  15 valid / 0 rejected
  ...
Model artifacts per context ('global' == the '*' context):
  IntentDetection: NO MODEL FOLDER (declared_in_routing=True)   <- normal: internal CME context
  global: complete | thresholds: conf=0.5216... tiny_amb=0.5216... large_amb=0.6365...
```
No `UNTRAINED contexts` line is printed — that is the healthy signature for an
app workflow. `15 valid / 0 rejected` is a healthy DSPy few-shot artifact
(train targets num_examples=15; validation rejects examples whose values don't
fuzzy-match the utterance).

### Sick: built but never trained (tests/example_workflow, as found in repo)

```
  TodoItem/assign_to                                0     0    yes  MISSING
  ...
Model artifacts per context ('global' == the '*' context):
  IntentDetection: NO MODEL FOLDER (declared_in_routing=True)
  TodoItem: NO MODEL FOLDER (declared_in_routing=True)
  ...
  UNTRAINED contexts (declared but no threshold.json): TodoItem, TodoList, TodoListManager, global
```

Three independent sick signals here, with different meanings:

| Signal | Meaning | Next action |
|---|---|---|
| `UNTRAINED contexts ... global` | never trained (or models deleted); chat will fail fast via `is_workflow_trained`, or crash opening threshold.json if invoked directly | `fastworkflow train <wf> <env> <passwords>` — into a TEMP COPY if this is a bundled example (fix-0hb) |
| `param_labeled ... MISSING` on a parameterized command | DSPy few-shot examples never generated (train not run, or LLM_SYNDATA_GEN failed) — parameter extraction runs zero-shot, quality drops silently | retrain; check LITELLM_API_KEY_SYNDATA_GEN |
| `plain: 0` on app commands | command files declare empty `plain_utterances` lists — the classifier for that command trains on synthetic utterances only, or nothing | genuinely empty in tests/example_workflow (verified in its `_commands/TodoItem/*.py`); for a real workflow, add seeds or run `fastworkflow refine` |

---

## 2. check_cache_freshness.py

### STALE (real, found in-repo on 2026-07-09)

```
$ ... check_cache_freshness.py tests/example_workflow
Workflow: /home/drawal/rl/fastworkflow/tests/example_workflow
Live fingerprint : ab4c00b5f7ec131f...
Covers 22 source files; 5 most recently modified:
  2026-07-08T18:18:19  .../tests/example_workflow/_commands/TodoList/get_all_children.py
  2026-06-25T15:50:18  .../tests/example_workflow/_commands/context_inheritance_model.json
  ...
command_directory.json: STALE
  stamped: 6e99bab0787f78a8...
routing_definition.json: STALE
  stamped: 6e99bab0787f78a8...
```

Reading it: the newest-source list is the tell — `get_all_children.py` was
modified 2026-07-08, after the snapshots were stamped. Next fastworkflow load
of this workflow silently rebuilds both JSONs (seconds, harmless). This is the
exact failure class v2.22.1 (commit b5747df) exists to catch: before the
fingerprint, a renamed/deleted command left the stale snapshot trusted and
imports failed at runtime.

### FRESH

```
$ ... check_cache_freshness.py fastworkflow/_workflows/command_metadata_extraction
command_directory.json: FRESH
routing_definition.json: FRESH
Verdict: snapshots are trusted as-is; no rebuild on next load.
```

### Other verdicts (produced by construction, logic verified against command_routing.py:384-407)

- `LEGACY` — snapshot has no `source_fingerprint` field (written pre-v2.22.1):
  treated as not-fresh, forced rebuild.
- `MISSING` — never built here, or `___command_info` deleted.
- STALE with byte-identical sources — fingerprint embeds ABSOLUTE paths and
  `mtime_ns` (command_directory.py:646), so moving the workflow dir, a fresh
  `git clone`, Docker COPY, or `touch` all invalidate. Spurious but harmless.

---

## 3. trace_turn.py

### Real turn + span tree (a deterministic turn recorded into a temp observability DB)

```
$ ... trace_turn.py turns --db /tmp/fwdb_npt1zpjk/observability.sqlite3
/tmp/fwdb_npt1zpjk/observability.sqlite3 — 1 turn(s), newest first:

20260825T204226.820492Z-a5390537d187  [completed/OK]  no conversation (CLI/one-off)
    channel : (unbound)
    started : 2026-08-25T20:42:26.820510+00:00  completed: 2026-08-25T20:42:26.820701+00:00
    message : list_tasks please
    answer  : ok:list_tasks please

$ ... trace_turn.py turn 20260825T204226.820492Z-a5390537d187 --db ...
turn_key : 20260825T204226.820492Z-a5390537d187
status   : completed / OK
channel  : (unbound)  no conversation (CLI/one-off)
message  : list_tasks please
answer   : ok:list_tasks please

spans (2):
- fw.turn [completed, 0.2ms]  context=*
    turn_key: 20260825T204226.820492Z-a5390537d187
    user_message: list_tasks please
    status: completed
    success: True
  - fw.agent.tool_call [ok, 0.1ms]  command=list_tasks
      raw_command: list_tasks please
      response_text: ok:list_tasks please
      success: True
```

`channel : (unbound)` and "no conversation" are normal for a bare
`WorkflowExecutionContext`: the CLI binds a synthetic `cli:<timestamp>` channel
and the server binds the real one. Pre-Phase-7 this data came from a CWD
`action.jsonl` holding only the last turn; that mirror is retired, and any copy
still on disk (there is one in this repo root from 2026-06-04) is residue.

### Server turn response (SYNTHETIC body; field shape verified against turns.py render_turn_response and utils.py _format_trace_event)

```
$ ... trace_turn.py response /tmp/turn.json
turn_key       : 20260610T120000Z-ab12cd34ef56
exec_state     : done
status         : completed
success        : True
answer         : Added user jane to the room.

traces (2):
[0] Agent -> Workflow: add_user <name>jane</name>
[1] Workflow -> Agent [OK]: add_user, {'name': 'jane'}
      User jane added

command_outputs (1):
[0] add_user params="name='jane'" duration_ms=812
```

Sick patterns when reading turn responses:

| Pattern | Meaning | Next action |
|---|---|---|
| `success: False` but `status: completed` and a confident `answer` | some command returned failure and the agent masked it in prose — the three signals are deliberately orthogonal (turn.py:196-216) | find the failing entry in `command_outputs` |
| `status: failed`, `failure_reason: max_iters_exhausted` | agent hit max_iters=25 without finishing | inspect traces for a loop; see fastworkflow-debugging-playbook |
| `traces` empty on a retried request | destructive trace drain — traces were consumed by the first poll of the same turn_key (turns.py:293-296, fix-85g Step 2 open) | treat traces as read-once; save the first response |
| `[FAIL]` Workflow->Agent lines in dim orange (CLI) | that tool call's CommandOutput.success was False (param-extraction error, misunderstood intent, validation error) | read the response text of that step |
| 202 `{turn_key, exec_state: "running"}` | turn deferred past the wait window, still executing | re-POST the identical request to rejoin (idempotency key = hash(channel_id+kind+args)); a DIFFERENT request gets 409 |

---

## 4. collect_model_metrics.py

### Healthy (CME workflow; ~40 s CPU, models loaded read-only)

```
$ ... collect_model_metrics.py fastworkflow/_workflows/command_metadata_extraction
Scores are on SEED utterances (training data) — health check, not held-out F1.

=== ErrorCorrection: SKIPPED — no threshold.json (context untrained)

=== * (31 utterances, 5 commands)
  top1_accuracy=1.0  ambiguous_rate=0.0  distil_fallback_rate=0.129  mean_confidence=0.6531
  thresholds: conf=0.4129 tiny_amb=0.4010 large_amb=0.5468
    IntentDetection/go_up                         n=7   top1=7   confident=7   amb_topk_hit=0
    ...

=== IntentDetection (31 utterances, 5 commands)
  top1_accuracy=0.9677  ambiguous_rate=0.0  distil_fallback_rate=0.129  mean_confidence=0.5891
    IntentDetection/go_up                         n=7   top1=6   confident=6   amb_topk_hit=0
    ...
  confusions (expected -> predicted):
    IntentDetection/go_up -> IntentDetection/reset_context  x1
```

Also observed (stderr, repeated): sklearn `InconsistentVersionWarning: Trying
to unpickle estimator LabelEncoder from version 1.7.0 when using version
1.9.0`. That is real version skew between the sklearn that trained these local
artifacts and the current venv. For LabelEncoder it has been benign in
practice here, but if predictions ever look like shuffled labels, suspect the
pickle first and retrain (into a temp copy).

Interpretation table:

| Pattern | Meaning | Next action |
|---|---|---|
| top1_accuracy ≥ ~0.95 on seeds, few confusions | healthy artifacts; seeds are training data, so near-perfect is the EXPECTED baseline, not an achievement | nothing |
| top1_accuracy well below ~0.9 on seeds | broken/mismatched artifacts, label drift (commands changed since train), or wrong model folder | run check_cache_freshness; retrain |
| one recurring confusion pair (like go_up -> reset_context above) | two commands with genuinely overlapping seed phrasings | differentiate the plain_utterances; retrain; at runtime this surfaces as the ambiguity/clarification flow |
| high ambiguous_rate | ambiguous thresholds sit above typical confidence → many turns will trigger INTENT_AMBIGUITY_CLARIFICATION | often over-tight thresholds on tiny label sets; compare thresholds vs mean_confidence |
| distil_fallback_rate high (>~0.3) | TinyBERT rarely confident; every fallback pays the DistilBERT latency | threshold tuning already penalizes distil usage (alpha=0.15); consider more/better seed utterances |
| everything predicted `wildcard` | out-of-scope class swallowed the commands — classic symptom of training-set imbalance or artifacts from a different command set | retrain; verify fingerprint freshness |
| `SKIPPED — no stored seed utterances` | commands declare empty plain_utterances (e.g. tests/example_workflow) | nothing to measure; add seeds |

Known measurement gap (deliberate, do not pretend otherwise): synthetic
training utterances are generated at train time (litellm + PersonaHub) and
never persisted, so offline held-out F1 is impossible without an LLM key.
Training's own internal test-split metrics (f1 * ndcg * distil-usage penalty,
model_pipeline_training.py:222-258) exist only in train-time stdout — they are
not written to any artifact. If you need them, capture train logs.

---

## 5. DSPy cache status (real output)

```
$ .venv/bin/python -m fastworkflow.utils.dspy_cache_utils status
📊 Current DSPy Cache Status:
  • Disk cache enabled: True
  • Memory cache enabled: True
  • Cache directory: /home/drawal/.dspy_cache
  • LiteLLM cache enabled: False
```

Healthy default. Sharp edges:
- Cached LLM calls return instantly AND do not appear in
  `dspy.inspect_history()` — if you are debugging "why did the LLM answer
  this", a cache hit hides the call. Clear or disable first. Note the CLI's
  `clear`/`reset` subcommands call `dspy.configure_cache(...)` in THAT
  process only, so they are no-ops for your workflow process — as a CLI, only
  `status` and `clear-disk` (deletes the cache directory) are useful. To
  disable caching inside your own process, call
  `fastworkflow.utils.dspy_cache_utils.clear_dspy_cache_completely()` or
  `dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False,
  enable_litellm_cache=False)` before the calls you want fresh.
- docs/DSPY_CACHE_GUIDE.md is PARTIALLY STALE: it references
  `fastworkflow.run_agent.agent_module` and a repo-root `dspy_cache_utils.py`,
  neither of which exists in v2.22.2. The real module is
  `fastworkflow/utils/dspy_cache_utils.py` (`python -m
  fastworkflow.utils.dspy_cache_utils {clear,clear-disk,reset,status}`), and
  there is no `show_dspy_traces` helper — use `dspy.inspect_history(n=...)`.

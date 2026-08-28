---
name: debug-workflow-conversations
description: >-
  Diagnose fastWorkflow task failures from the conversation logs in a workflow's
  observability.sqlite3: locate and safely read the database, walk a turn's span
  trace to the failing pipeline stage (intent routing, parameter extraction,
  db_lookup, validation, planning, ask_user, or the app's own code), and route
  each diagnosis to the fastWorkflow feature and companion skill that fixes it.
  Covers the span taxonomy and structured attributes (classifier confidence and
  thresholds, matcher layers, per-field db_lookup outcomes, validation-hook
  verdicts, context mutations), the failure-triage decision tree, and the
  read-only access rules. Use when asked why a workflow task or conversation
  failed, when a chat turn did the wrong thing, when deciding which fastWorkflow
  feature would prevent a failure, or before recommending changes to commands,
  contexts, seeds or signatures based on observed behavior.
---

# Debugging workflows from conversation logs

Every turn a fastWorkflow workflow executes is recorded — the user's message, the
agent's plan, every intent-detection attempt with the classifier's confidence,
every extracted parameter, every db_lookup correction, every validation verdict,
every ask_user exchange, and the final answer. When a task fails, the log tells
you **which pipeline stage failed**, and each stage has a specific fastWorkflow
feature that fixes it. This skill is the map from log to fix.

## 1. Locate the database

One SQLite database per workflow, under the fastWorkflow state root:

```python
from fastworkflow import state_paths
db_path = state_paths.observability_db("<workflow_folder>")
# typically ~/.local/state/fastworkflow/workflows/<workflow-name>/observability.sqlite3
```

If the file does not exist, the workflow has never run with observability on
(the default under `fastworkflow run`, `run_chatbot`, and `run_fastapi_mcp`).
Reproduce the failure first — `fastworkflow run_chatbot <workflow>` is the
fastest way; every chat turn lands in this database immediately.

## 2. Read it — read-only, always

```python
from fastworkflow.observability_store import ReadOnlyObservabilityStore
store = ReadOnlyObservabilityStore(db_path)
```

**Never instantiate `ObservabilityStore` to inspect a database.** That class is
the writer: constructing it creates the file if missing, runs schema migration,
and write-probes it. `ReadOnlyObservabilityStore` opens `mode=ro` connections
and cannot mutate anything. Raw SQL is equally fine (the schema is documented
in [reference.md](reference.md)):

```bash
sqlite3 "file:$DB?mode=ro" "SELECT turn_key, status, success, user_message FROM turns ORDER BY turn_key DESC LIMIT 20"
```

## 3. Find the failing turn

```python
store.list_turns(status="failed")                 # turn-level failures
store.list_turns(success=False)                   # any command in the turn failed
store.list_turns(command_name="cancel_order")     # turns that executed a command
store.list_turns(context="TodoList")              # substring match on entry context
turn  = store.get_turn(turn_key)                  # full row incl. record_json
spans = store.get_spans(turn_key)                 # the trace, ordered by start_ns
```

Two orthogonal outcome fields, both worth reading:

- `status` is the turn lifecycle: `completed` / `failed` (agent ran out of
  iterations — `failure_reason: max_iters_exhausted`) / `awaiting_user` (still
  suspended) / `cancelled` / `abandoned`.
- `success` means every command in the turn succeeded. **`completed` with
  `success = 0` is the case to hunt**: a command failed and the agent wrote a
  confident answer over it.

## 4. Walk the trace to the failing stage

A turn's spans nest like this (`parent_span_id` links them; names and full
attribute catalogs in [reference.md](reference.md)):

```
fw.turn                          the whole logical turn (+ context_mutations diff)
├── fw.planner.plan/.replan      the agent's plan; replans carry their trigger
│   └── fw.llm.call              the planner's LM call
└── fw.agent.execute             the ReAct loop as a whole (attempts, final answer)
    └── fw.agent.step            one reasoning step: thought, tool choice, observation
        ├── fw.llm.call          the step's reasoning LM call (cache_hit exposes
        │                        stale-cached completions)
        └── fw.agent.tool_call   the agent invoking a command (raw_command)
            └── fw.command.execute       resolution + execution
                ├── fw.nlu.intent            matcher layer, confidence, candidates
                └── fw.nlu.param_extraction  extraction + validation, structured
                    └── fw.llm.call          the LM extraction call, when one ran
fw.ask_user                      a clarifying question + the human wait

Deterministic "/"-mode turns skip the planner/agent layers (fw.command.execute
directly under fw.turn); assistant-mode tool_call/command.execute pairs sit
under fw.turn without agent.step.
```

## 5. The triage tree

Work through these checks **in order** — earlier stages corrupt everything
downstream, so the first failing stage is the diagnosis.

**A. Did routing pick the right command?** Compare `fw.command.execute`'s
`raw_command` (what was asked) against its `command_name` (what ran), then read
the `fw.nlu.intent` spans:

| Observation | Diagnosis | Fix with |
|---|---|---|
| `resolved: false` on every attempt (the walk climbed contexts and gave up) | The utterance routes nowhere — vocabulary gap or command missing from the context's surface | Seed utterances (`plain_utterances`, ~8 varied phrasings) · `design-context-models` (is the command reachable from this context?) |
| `ambiguous: true` with a `candidates` list | Classifier confidence below threshold — check `classifier.confidence` vs `classifier.ambiguous_threshold`; near-misses mean starved or colliding seeds | `detect-duplicate-capabilities` (are two candidates the same capability?) · seeds · `design-context-models` |
| Wrong `command_name`, `matcher_layer: classifier` | A confident mis-route: the wrong command's training set claims this phrasing | `detect-duplicate-capabilities` · `evaluate-intent-routing` (measure before/after) · seeds |
| Wrong `command_name`, `matcher_layer: fuzzy_prematch` or `embedding_cache` | A pre-classifier layer matched — the utterance lexically resembles another command's name, or a stale cache entry | Rename the colliding command, or clear the workflow's `___convo_info` cache |
| `escalation_labels_discarded` present | The command likely lives in an ancestor context but the local prompt hid that | `design-context-models` (context surfaces / `base` inheritance) |

**B. Were the parameters extracted correctly?** Read `fw.nlu.param_extraction`:

| Observation | Diagnosis | Fix with |
|---|---|---|
| `missing_fields` non-empty | Extraction could not find the value in the utterance | Sharpen `Field(description=…, examples=…)` on `Signature.Input`; if the value is an opaque handle the user cannot know, declare its producer — `declare-parameter-producers` (`available_from`) |
| `db_lookup` event with `outcome: rejected` + `suggestions` | The typed value missed the live key set | `resolve-parameter-values` — check the key set and thresholds; a rejection of a *valid* value usually means the wrong candidate list |
| `db_lookup` event with `outcome: applied`, `corrected: true`, but wrong result downstream | The fuzzy matcher rewrote the value incorrectly (auto-apply too loose, or label/uid mixup) | `resolve-parameter-values` — `auto_apply_threshold`, and return the value the *field* holds, not the label matched on |
| `validation_hook.is_valid: false` | The command's own `validate_extracted_parameters` rejected the call — `message` says why; `raised` means the hook itself crashed | `validate-command-parameters` |
| `retry_round: true` on successive turns | The user is stuck in the NOT_FOUND correction loop — count the rounds; more than two means the error message is not actionable | `validate-command-parameters` (error-message quality) + field descriptions |
| `extraction_method: llm` and the nested `fw.llm.call` shows `cache_hit: true` with a wrong completion | A stale DSPy-cache replay, not a live extraction failure | Clear the DSPy cache and re-test before changing anything |
| A nested `fw.llm.call` with `status: error` and an `exception` (auth, timeout) | Environment problem — keys/env files — not workflow design | Fix the env/passwords files; nothing to change in the workflow |

**C. Did the agent plan a workable sequence?** `fw.planner.replan` spans carry
their trigger: repeated `parameter_extraction_error` replans mean the agent
cannot discover where a handle comes from → `declare-parameter-producers`.
Repeated `ask_user_response` replans point at ambiguous command surfaces or
missing context navigation → `design-context-models`.

**D. Did the conversation stall on questions?** Multiple `fw.ask_user` spans in
one turn (each records `agent_query` and the reply): the agent is asking for
things the workflow should resolve itself — usually the same fixes as B.

**E. Was state stored and used?** The `fw.turn` close carries
`context_mutations` (`added` / `changed` / `removed` keys with brief values).
A command that should have stored a handle but shows no mutation, or a later
turn that re-asks for stored information, is a storing-information-in-context
bug in the command's own code.

**F. Everything above clean?** Then the failure is the command implementation:
`fw.command.execute` has `status: error` or `success: false` with the
`response_text`; the full `CommandResponse` (and, when
`FW_OBS_CAPTURE_TRACEBACKS=1` was set, the traceback artifact) is in the
turn's `record_json`. Read the command's `_commands/<name>.py` source next.

## 6. Recommending fixes

State the diagnosis with its evidence (span attribute values, not paraphrase),
name the feature, and load the companion skill before writing the change:

| Feature | Companion skill |
|---|---|
| Seed utterances, context layout, `base` inheritance | `design-context-models` |
| Near-duplicate commands | `detect-duplicate-capabilities` |
| Measuring whether a routing change helped | `evaluate-intent-routing` |
| `available_from` producer hints | `declare-parameter-producers` |
| `db_lookup` value resolution | `resolve-parameter-values` |
| `validate_extracted_parameters` | `validate-command-parameters` |
| Training utterance realism | `supply-training-personas` |
| Retraining mechanics after any of the above | `train-and-publish-models` |
| Turning the failing conversation into regression coverage | `build-task-benchmarks` |

## Honesty notes

- Everything persisted is **redacted** (credential shapes and secret env
  values) and **size-capped**: an oversized attribute becomes a
  `{"truncated": true, "original_length": …, "sha256": …, "value": prefix}`
  envelope; oversized artifacts become `__fw_artifact_ref__` envelopes whose
  content lives in the `artifacts` table.
- Spans are best-effort: under load they can be dropped (counted in the
  `writer_health` diagnostics row — check it before concluding "no spans means
  nothing ran"). Turn records are near-lossless.
- A turn resumed in a different process finalizes without `context_mutations`
  (the baseline is not serialized), and `--generate_insights` CLI runs emit
  teacher *and* student passes into the same trace (roughly double the tool
  calls).
- Rows with `conversation_id NULL` are real turns from conversation-less
  embedders; query by `turn_key`/`channel_id` instead of the conversation
  drill-down.

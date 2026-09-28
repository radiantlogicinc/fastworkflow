---
name: debug-workflow-conversations
description: >-
  Diagnose fastWorkflow failures from run_chatbot traces and human comments, locate the
  correct evidence store, and trace wrong routing, extraction, planning, execution or answers
  to a concrete workflow fix. Use when a conversation behaves incorrectly or review feedback
  needs a trace-supported diagnosis. Read recorded evidence without mutating it.
---

# Debugging workflows from conversation logs

Use recorded turns and their span trees to identify the first wrong decision and its
consequences. Capture depends on the runtime profile, redaction, limits and writer health;
not every run contains every input or response.

## 1. Locate the conversation and its evidence store

In an existing `run_chatbot` session, navigate **benchmark → experiment → conversation**, or
**ad-hoc conversations → UTC date → conversation**. The left tree stops at conversations.
Select turns and drill into phases, steps and calls on the right; the right breadcrumb shows
the complete path. Read the **Human feedback** history at the relevant component as well as
its raw evidence. Use [optimize-workflow-with-feedback](../optimize-workflow-with-feedback/SKILL.md)
when the task is to improve and compare the workflow, rather than diagnose one failure.

For ordinary runs, resolve the database through the framework:

```python
from fastworkflow import state_paths
db_path = state_paths.observability_db("<workflow_folder>")
```

For registered experiments, resolve the registration's `store` via
`benchmark_setup.load_experiment(workflow_folderpath, experiment_id)`. A null store means
execution has not bound the registration yet. Workspace evidence may span multiple stores;
retain `store_id` with every turn/span reference. Conversation IDs are local to channels/stores.

A missing default database does not prove the workflow never ran: check the selected state root,
experiment registration and workspace sources first. If reproduction is needed, prepare it under
the project's execution permissions. The supported picker command is `fastworkflow run_chatbot`;
selecting a workflow can start a server, so it is not merely an offline file viewer.

## 2. Read it — read-only, always

```python
from fastworkflow.observability.store import ReadOnlyObservabilityStore
store = ReadOnlyObservabilityStore(db_path)
```

**Never instantiate `ObservabilityStore` to inspect a database.** That class is
the writer: constructing it can create the file and write-probe it, and it DELETES a populated
database from an older schema version and recreates it empty (there is no migration). A
database from a newer version is refused. Preserve older evidence by copying it first, and read
it with the matching framework version rather than altering its schema.
`ReadOnlyObservabilityStore` opens `mode=ro` connections
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
comments = store.list_human_feedback(turn_key)    # all human annotation anchors for this turn
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

Inspect the recorded causal order. The checks below help locate the first wrong decision;
a bad plan can precede routing, and a provider or backend failure may be environmental.
Treat a human comment as a lead to verify, not proof of the cause.

**A. Did routing pick the right command?** Compare `fw.command.execute`'s
`raw_command` (what was asked) against its `command_name` (what ran), then read
the `fw.nlu.intent` spans:

| Observation | Diagnosis | Fix with |
|---|---|---|
| `resolved: false` on every attempt (the walk climbed contexts and gave up) | The utterance routes nowhere — vocabulary gap or command missing from the context's surface | Seed utterances (`plain_utterances`, ~8 varied phrasings) · `design-context-models` (is the command reachable from this context?) |
| `ambiguous: true` with a `candidates` list | Classifier confidence below threshold — check `classifier.confidence` vs `classifier.ambiguous_threshold`; near-misses mean starved or colliding seeds | `detect-duplicate-capabilities` (are two candidates the same capability?) · seeds · `design-context-models` |
| Wrong `command_name`, `matcher_layer: classifier` | A confident mis-route: the wrong command's training set claims this phrasing | `detect-duplicate-capabilities` · `evaluate-intent-routing` (measure before/after) · seeds |
| Wrong `command_name`, `matcher_layer: fuzzy_prematch` or `embedding_cache` | A pre-classifier layer matched — the utterance lexically resembles another command's name, or a stale cache entry | Rename the colliding command, or investigate the matched cache entry and its provenance |
| `escalation_labels_discarded` present | The command likely lives in an ancestor context but the local prompt hid that | `design-context-models` (context surfaces / `base` inheritance) |

**B. Were the parameters extracted correctly?** Read `fw.nlu.param_extraction`:

| Observation | Diagnosis | Fix with |
|---|---|---|
| `missing_fields` non-empty | Extraction could not find the value in the utterance | Sharpen `Field(description=…, examples=…)` on `Signature.Input`; if the value is an opaque handle the user cannot know, declare its producer — `declare-parameter-producers` (`available_from`) |
| `db_lookup` event with `outcome: rejected` + `suggestions` | The typed value missed the live key set | `resolve-parameter-values` — check the key set and thresholds; a rejection of a *valid* value usually means the wrong candidate list |
| `db_lookup` event with `outcome: applied`, `corrected: true`, but wrong result downstream | The fuzzy matcher rewrote the value incorrectly (auto-apply too loose, or label/uid mixup) | `resolve-parameter-values` — `auto_apply_threshold`, and return the value the *field* holds, not the label matched on |
| `validation_hook.is_valid: false` | The command's own `validate_extracted_parameters` rejected the call — `message` says why; `raised` means the hook itself crashed | `validate-command-parameters` |
| `retry_round: true` on successive turns | The user is stuck in the NOT_FOUND correction loop — count the rounds; more than two means the error message is not actionable | `validate-command-parameters` (error-message quality) + field descriptions |
| `extraction_method: llm` and the nested `fw.llm.call` shows `cache_hit: true` with a wrong completion | Cached output; the cache hit alone does not prove staleness | Verify request/model/cache provenance; use an isolated fresh cache for an authorized comparison |
| A nested `fw.llm.call` with `status: error` and an `exception` (auth, timeout) | Environment problem — keys/env files — not workflow design | Fix the env/passwords files; nothing to change in the workflow |

**C. Did the agent plan a workable sequence?** `fw.planner.replan` spans carry
their trigger: repeated `parameter_extraction_error` replans mean the agent
cannot discover where a handle comes from → `declare-parameter-producers`.
Repeated `ask_user_response` replans point at ambiguous command surfaces or
missing context navigation → `design-context-models`.

**D. Did the conversation stall on questions?** Multiple `fw.ask_user` spans in
one turn (each records `agent_query` and the reply): review whether each question was needed.
Separate repeated asks after a valid answer from
necessary clarification or retries after invalid answers; inspect state retention as well as B.

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
| Publishing a regression task and creating a comparison experiment | [create-workflow-benchmarks](../create-workflow-benchmarks/SKILL.md) |
| Designing multi-turn regression content | [build-task-benchmarks](../build-task-benchmarks/SKILL.md) |
| Using human feedback to verify an improvement | [optimize-workflow-with-feedback](../optimize-workflow-with-feedback/SKILL.md) |

## Human feedback and output-quality failures

Read [reference.md](reference.md#human-feedback) for annotation anchors and read APIs.
`human_feedback` comments are distinct from agent-memory `feedback` and formal review ratings.
Do not write an annotation merely to inspect a trace or present an agent judgment as a human one.

If routing and parameters are correct, compare the final answer to command responses and the
stated expected outcome. Inspect `fw.llm.call` request limits (`call_kwargs.max_tokens`), output,
usage and errors for omitted or cut answers. Separate time spent waiting for a user from model
and command time. A cached call is not inherently stale; missing usage is not zero cost.

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

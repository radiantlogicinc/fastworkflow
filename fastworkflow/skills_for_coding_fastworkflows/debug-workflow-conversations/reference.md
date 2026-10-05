# observability.sqlite3 — the read contract

Reference for `debug-workflow-conversations` (see SKILL.md for the triage
method). The current source uses `SCHEMA_VERSION = 4` in `observability/store.py`.
Readers check compatibility; older evidence is not migrated by this version. Preserve it and
use its writer's framework version. Inspect `PRAGMA user_version` read-only when diagnosing a
mismatch, rather than forcing a version number or adding columns.

## Location and safe access

```python
from fastworkflow import state_paths
db_path = state_paths.observability_db("<workflow_folder>")

from fastworkflow.observability.store import ReadOnlyObservabilityStore
store = ReadOnlyObservabilityStore(db_path)   # mode=ro; cannot create/migrate/write
```

Raw SQL: open read-only (`sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)`
or `sqlite3 "file:...?mode=ro"`). The database is WAL-mode and safe to read
while the workflow runs. **Never** open it writable to inspect it —
`ObservabilityStore` (no `ReadOnly` prefix) is the writer and creates/probes
the file on construction.

## Relevant tables and joins

Use the installed `observability/store.py` for the exact schema; this is a navigation map, not
DDL to recreate or upgrade an evidence database.

| Table | Read purpose and identity |
|---|---|
| `conversations` | `(channel_id, conversation_id)`, topic, timestamps, experiment/task/attempt labels |
| `turns` | `turn_key`, channel/conversation, ordinal, lifecycle status, command success, failure reason, answer, `record_json`, experiment/task/attempt labels |
| `spans` | `span_id`, `trace_id = turn_key`, `parent_span_id`, name/kind, command/context, start/end, status, JSON `attributes` |
| `artifacts` | `artifact_id`, turn/span anchor, content type, byte size/digest, `inline_value`, capture error |
| `experiments` | Experiment identity, description, notes, benchmark pin, capture regime, status |
| `experiment_attempts` | `(experiment_id, task_id, attempt)`, channel, outcome/lifecycle evidence, `runtime_snapshot_json` |
| `human_feedback` | Append-only timestamped review notes anchored to a turn or component spans, with `category`/`subcategory`, `feedback_uid`, frozen `anchors_json` (v7; an older store is refused, not read) |
| `train_runs` | Training metadata and `metrics_json` |
| `diagnostics` | Writer health and store/capture markers |

Use `(store_id, turn_key)` when combining sources, and retain channel identity when looking up a
conversation. Parse `record_json` and span `attributes` as JSON. `end_ns IS NULL` means a span
has not closed, not zero duration. Turns reference immutable catalog tasks by their experiment
labels; the catalog corpus itself lives under the workflow's `benchmarks/` directory.

## Span catalog

`trace_id = turn_key` links every span to its turn; `parent_span_id` builds the
tree. Attribute values over the cap (`tracing.MAX_ATTR_BYTES`, 16 KiB)
are replaced by `{"truncated": true, "original_length", "sha256", "value"}`.

### `fw.turn` (root; kind `internal`)
One per logical turn; stays open across ask_user suspensions
(`status: awaiting_user`, `end_ns` NULL) and closes at terminal finalize.
Attributes: `turn_key`, `channel_id`, `conversation_id`, `user_message`,
`status`, `success`, `failure_reason`, `suspended_ms`, and
**`context_mutations`** — a shallow diff of the app workflow's context across
the turn: `{"added": {key: value_repr}, "changed": {key: {"from", "to"}},
"removed": [keys]}`, or `null` when nothing changed (also `null` after a
cross-process resume — the baseline is not serialized).

### `fw.planner.plan` / `fw.planner.replan` (kind `llm`)
Around the agent's task-planner calls. Attributes: `model`, the plan text
(capped) and, on replans, `replan_trigger`: `parameter_extraction_error` or
`ask_user_response`. Since contract v2 (aggregate span contract 8) also
`plan_source` — `structured` (the finish check's structured planner),
`text` (the plain-text planner, the default with the check off),
`text_fallback` (the structured call failed to parse or returned no steps, so
the plain-text planner ran), or `none` (no plan) — and `subjects`, the
structured plan's subject names redacted by the capture policy (`[]` for a
text plan and for continuation replans). With `plan_source: structured` the
plan text is a numbered list whose steps may carry `(optional)` /
`(needs the user)` flags. Spans written before v2 lack both keys.
Structured planning is disabled since 2026-09-28: new spans carry only
`plan_source` `text` or `none` and `subjects` `[]`, with the finish check on
or off; `structured` and `text_fallback` appear only in records written
before then.

### `fw.agent.execute` (kind `internal`) — the ReAct loop as a phase
Sibling of `fw.planner.plan` under the turn; NOT `fw.command.execute` (that is
one command inside a tool call — this is the whole loop). Attributes include
`agent_input`, `resumed`, `attempts`, `suspended`, `final_answer`.

### `fw.agent.step` (kind `internal`) — one reasoning step
Child of `fw.agent.execute`; the step's reasoning `fw.llm.call` and its
`fw.agent.tool_call` nest under it. Attributes: `step_index`, `thought`,
`tool_name`, `tool_args`, `observation`; on failures `error_type`/`tool_error`
and `recovered`; on suspension `clarification` (status `awaiting_user`);
`finish_check_note` on the one `finish` step whose observation was replaced by
the finish-time execution check's note (that observation is not a tool result).

### `fw.agent.tool_call` (kind `tool`)
One per agent → workflow invocation; `raw_command` is the exact command text
the agent sent, with `response_text`/`success` (and `error_type` on failure)
added at close.

### `fw.command.execute` (kind `tool`)
Wraps command resolution + execution. Attributes: `raw_command` (what was
asked), `parameters` (the extracted dict), `response_text`, `success`; columns
`command_name` and `context` hold what actually ran and where. Comparing
`raw_command` to `command_name` is the first routing check.

### `fw.nlu.intent` (kind `internal`) — one per prediction attempt
The wildcard pipeline may predict several times per command (walking up the
context chain), so read ALL of a trace's intent spans in `start_ns` order.

| Attribute | Meaning |
|---|---|
| `context`, `stage`, `utterance` | Where/when the attempt ran (`stage` ∈ INTENT_DETECTION, INTENT_AMBIGUITY_CLARIFICATION, INTENT_MISUNDERSTANDING_CLARIFICATION) |
| `matcher_layer` | Which layer decided: `exact_prefix`, `fuzzy_prematch`, `embedding_cache`, `classifier`, `clarification_default` |
| `classifier` | Present when the classifier ran: `{model_tier: tiny\|large, confidence, ambiguous_threshold, confident, top_label, topk_labels}` |
| `ambiguous` + `candidates` | Low-confidence prediction: the candidate list shown to the user/agent |
| `escalation_labels_discarded` | Escalation labels ranked in top-k but suppressed from the prompt |
| `fuzzy_prematch_tie` | Commands that tied at the fuzzy layer (deferred to the classifier) |
| `command_name`, `resolved`, `is_cme_command` | The outcome; `resolved: false` means no local prediction — the caller walks to the parent context |
| `cache_similarity_threshold` | Present on `embedding_cache` hits (0.85) |

### `fw.nlu.param_extraction` (kind `internal`)
Wraps parameter extraction + validation for the resolved command.

| Attribute | Meaning |
|---|---|
| `command_name`, `extraction_method` | `xml_regex` (agent format), `llm` (DSPy), `stored_merge` (a NOT_FOUND retry round merging user corrections) |
| `retry_round` | true when this turn continues a parameter-correction loop |
| `parameters_valid` | The overall verdict (span `status` stays `ok`; only exceptions mark `error`) |
| `missing_fields` / `invalid_fields` | Structured field lists (no prose parsing needed) |
| `db_lookup` | List of per-field events: `{field, input_value, outcome: applied\|rejected\|declined, corrected_value, corrected, suggestions}` — the hook's three-state contract, recorded |
| `validation_hook` | `{ran, is_valid, message, raised?}` from the command's `validate_extracted_parameters` |

### `fw.ask_user` (kind `human_wait`)
One per clarifying question (deterministic per-attempt span ids). Attributes:
`agent_query`, `user_response`, `attempt`, `human_wait_ms`; the wall-clock
wait also equals `end_ns - start_ns`.

### `fw.llm.call` (kind `llm`) — one per DSPy LM invocation
Emitted by a DSPy callback, so it appears under whichever stage made the call
(planner, LLM parameter extraction, summarization…).

| Attribute | Meaning |
|---|---|
| `module`, `module_chain`, `module_input` | Which DSPy module ran and with what inputs |
| `model`, `messages`, `prompt`, `call_kwargs` | The exact request sent to the LM |
| `output`, `provider_response`, `reasoning` | What came back (incl. provider-native reasoning when present) |
| `usage`, `cost`, `cache_hit`, `response_model`, `history_uuid`, `capture_source` | Cost accounting; **`cache_hit: true` means the completion came from the DSPy cache** — the classic "stale/frozen LLM output" tell |
| `usage_capture` | Present when usage was unavailable (DSPy history disabled in that process) |
| `exception` | The LM call failed (span `status: error`) — auth errors, timeouts |

Reserved, not yet emitted: `fw.train.*`.

## record_json (turns.record_json)

The full internal `TurnResult`, post-redaction:

```
{ "turn_output": {
    "turn_key", "status", "failure_reason", "answer",
    "command_outputs": [                  // every command the turn executed, in order
      { "command_name", "context",
        "command_parameters": {...},      // typed params dumped to a dict
        "command_response": { "response", "success", "artifacts": {...} },
        "started_at", "duration_ms" } ],
    "success" },
  "channel_id", "conversation_id", "ordinal",
  "user_message", "refined_user_message",
  "entry_workflow_name", "entry_context",
  "started_at", "completed_at", "suspended_ms", "continuation_of" }
```

- **ask_user entries invert roles**: when `command_name == "ask_user"`,
  `command_parameters` is the agent's QUESTION and the response is the user's
  ANSWER (`success: false` = still unanswered).
- Artifact values over `FW_OBS_INLINE_ARTIFACT_BYTES` (256 KiB) are replaced by
  `{"__fw_artifact_ref__": <artifact_id>, "size", "content_type",
  "content_encoding", "error"}` — fetch the content from the `artifacts` table
  by id.
- The sketch above shows the diagnosis-relevant fields; rows may carry further
  additive fields (e.g. `workflow_name`, `next_actions`, `recommendations` on
  responses) — treat unknown keys as informational.
- Non-JSON values become `{"__fw_unserializable__": <type>, "repr": ...}`.
- Tracebacks are persisted only when the run had `FW_OBS_CAPTURE_TRACEBACKS=1`;
  otherwise the artifact holds a suppression notice.

## Read API (`ReadOnlyObservabilityStore`)

| Method | Returns |
|---|---|
| `list_turns(channel_id=, conversation_id=, status=, success=, command_name=, context=, experiment_id=, task_id=, attempt=, limit=, offset=)` | Turn rows newest-first, without `record_json` (`context` is a substring match; `command_name` matches via spans) |
| `get_turn(turn_key)` | The full row incl. `record_json` (parse it yourself) |
| `get_spans(...)` | Span rows for one turn, ordered by `start_ns` (`attributes` is a JSON string). Pass the turn key POSITIONALLY — the parameter is named `trace_id` |
| `list_conversations(channel_id=, limit=, offset=)` / `list_channels()` | Navigation |
| `get_artifact(artifact_id)` | Offloaded artifact row (`inline_value` is bytes) |
| `list_train_runs(limit=)` | Training-run metrics rows, newest first (`metrics_json`) |
| `list_human_feedback(turn_key)` | All component and turn notes, oldest first; decoded `span_ids`, `category`/`subcategory`, decoded `anchors` |
| `list_task_feedback(experiment_id=, task_id=)` | Every note about one task, across attempts and turns, plus notes whose frozen pair anchor names it |
| `get_experiment(experiment_id)` / `experiment_attempt_rows(experiment_id, task_id=)` | Pin/configuration and attempt records; attempt `runtime_snapshot` is decoded or null |
| `store_identity()` / `capture_regime()` | Evidence source and capture profile/policy identity |
| `writer_health()` | The writer's drop/error counters — read this before trusting span completeness |
| `db_size_bytes()` | File + WAL size |

Additional conversation-memory reads exist (`get_memory_window`,
`count_usable_turns`, `conversation_summaries`, `list_conversation_summaries`,
`dump_all_conversations`) — Phase-7 consolidation surface, usable but not
needed for failure diagnosis. Note `ReadOnlyObservabilityStore` inherits the
writer's method NAMES too; any accidental write raises on the `mode=ro`
connection rather than mutating anything.

## Query recipes

```sql
-- Confidently wrong routing: what was asked vs what ran
SELECT s.trace_id, json_extract(s.attributes,'$.raw_command') AS asked,
       s.command_name AS ran
FROM spans s WHERE s.name='fw.command.execute'
ORDER BY s.start_ns DESC LIMIT 20;

-- Ambiguity hot spots per context, with the classifier's numbers
SELECT s.context,
       json_extract(s.attributes,'$.classifier.confidence')  AS conf,
       json_extract(s.attributes,'$.classifier.ambiguous_threshold') AS thr,
       json_extract(s.attributes,'$.candidates') AS candidates
FROM spans s
WHERE s.name='fw.nlu.intent' AND json_extract(s.attributes,'$.ambiguous');

-- Parameter-correction loops (users stuck re-entering values)
SELECT trace_id, COUNT(*) AS rounds FROM spans
WHERE name='fw.nlu.param_extraction'
  AND json_extract(attributes,'$.retry_round')
GROUP BY trace_id HAVING rounds > 1;

-- db_lookup rejections with what was offered instead
SELECT trace_id, json_extract(value,'$.field') AS field,
       json_extract(value,'$.input_value') AS typed,
       json_extract(value,'$.suggestions') AS offered
FROM spans, json_each(json_extract(spans.attributes,'$.db_lookup'))
WHERE spans.name='fw.nlu.param_extraction'
  AND json_extract(value,'$.outcome')='rejected';

-- Turns that "completed" over a failed command (the quiet failures)
SELECT turn_key, user_message, answer FROM turns
WHERE status='completed' AND success=0 ORDER BY turn_key DESC;

-- What a turn stored into workflow context
SELECT json_extract(attributes,'$.context_mutations') FROM spans
WHERE name='fw.turn' AND trace_id=:turn_key AND end_ns IS NOT NULL;
```

## Trust notes

- Capture profile/policy can withhold inputs, outputs or attributes. Inspect `capture_regime()`
  and recorded envelopes; missing or withheld text is not proof the runtime lacked that data.
- `call_kwargs` is flat: the completion cap is `call_kwargs.max_tokens`. Some mapping attributes
  can be JSON strings; decode them before reading fields. Compare usage, output and request limits
  together; a cap match alone is not a task-failure verdict.

- All persisted text passed the redaction pass (credential shapes + loaded
  secret env values become `[REDACTED]`).
- Turn records are near-lossless; spans are best-effort under load — check
  `writer_health()` (`spans_dropped`, `records_dropped`, `write_errors`)
  before reading absence as evidence.
- `--generate_insights` CLI turns contain teacher AND student passes in one
  trace (duplicate-looking tool calls are expected there).

## Human feedback

In the working `run_chatbot` session, **Save feedback** appends a comment; it does not edit the
turn, train a model or alter scores. The UI anchors a component using its recorded span IDs.
Supported `target_kind` values are `turn`, `phase`, `step`, `span`:

- A turn anchor has `span_ids=[]`.
- Other anchors require at least one span belonging to that turn. For grouping nodes the UI
  gathers the component's recorded spans. Do not invent IDs or use labels as unique anchors.
- If there is no recorded span, annotate the turn instead.
- Each row includes `feedback_id`, `turn_key`, `target_kind`, decoded `span_ids`, `target_label`,
  `comment` and `created_at`. Comments are credential-scrubbed and retained chronologically.

```python
import json
from fastworkflow.observability.store import ReadOnlyObservabilityStore

store = ReadOnlyObservabilityStore(db_path)
turn = store.get_turn(turn_key)
spans = {row["span_id"]: row for row in store.get_spans(turn_key)}
for comment in store.list_human_feedback(turn_key):
    anchors = [spans[span_id] for span_id in comment["span_ids"] if span_id in spans]
    # Inspect anchors and json.loads(anchor["attributes"]) alongside the original comment.
```

For authorized annotation automation, the write endpoint is
`POST /post_feedback?turn_key=<encoded-key>` with exactly
`{"target_kind":"turn","span_ids":[],"target_label":"Turn","comment":"...",`
`"provenance":"coding_agent","category":"conclusions","subcategory":"what_went_wrong"}`.
Category and subcategory are enums and must pair; `GET /api/feedback-taxonomy` lists all six
pairs with the watermark text the composer shows. Reads are separate routes:
`GET /api/feedback-notes?turn_key=<encoded-key>` for one turn, and
`GET /api/task-feedback?experiment=<id>&task=<id>` for every note about a task across attempts,
turns and components (optional `category`, `subcategory`, `provenance`, `target_kind`,
`component`, `attempt`, `limit`, `offset`; no filter is applied by default).

Preserve the UI's authentication and source selection: reads go to the workflow's one live
database, or add `store_id=<id>` for workspace reads.

A POST is accepted even when the evidence must not be written. For a workspace store and for a
database file this process cannot write, the note is appended to an annotation sidecar,
`<stem>.feedback.sqlite3` beside the evidence, and the evidence file is byte-identical
afterwards; the response carries `"annotated": true` and the reads return the union. Otherwise
the server uses the narrow `ObservabilityStore.open_for_annotation` path; that is not a reason to
open a writer during analysis.

The agent-memory `feedback` table and the `/api/feedback` read routes it backed were removed with
their `dspy.History` injection (fix-9eg.16); those paths now 404. Formal human review assignments
have separate rubric/capability controls and are not these comments.

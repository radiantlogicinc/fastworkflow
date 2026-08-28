# observability.sqlite3 — the read contract

Reference for `debug-workflow-conversations` (see SKILL.md for the triage
method). Everything here is the shipped contract for READING a workflow's
conversation logs; schema version is `PRAGMA user_version = 1` and readers
must refuse a database with a higher version.

## Location and safe access

```python
from fastworkflow import state_paths
db_path = state_paths.observability_db("<workflow_folder>")

from fastworkflow.observability_store import ReadOnlyObservabilityStore
store = ReadOnlyObservabilityStore(db_path)   # mode=ro; cannot create/migrate/write
```

Raw SQL: open read-only (`sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)`
or `sqlite3 "file:...?mode=ro"`). The database is WAL-mode and safe to read
while the workflow runs. **Never** open it writable to inspect it —
`ObservabilityStore` (no `ReadOnly` prefix) is the writer and creates/probes
the file on construction.

## Schema (v1)

```sql
CREATE TABLE conversations (
  channel_id TEXT NOT NULL, conversation_id INTEGER NOT NULL,
  topic TEXT, summary TEXT, status TEXT, next_ordinal INTEGER,
  started_at TEXT, last_turn_at TEXT, updated_at TEXT,
  PRIMARY KEY (channel_id, conversation_id));

CREATE TABLE conversation_counters (          -- id minting; never read for debugging
  channel_id TEXT PRIMARY KEY, next_id INTEGER NOT NULL);

CREATE TABLE turns (
  turn_key TEXT PRIMARY KEY,                  -- logical turn key = spans.trace_id
  channel_id TEXT NOT NULL, conversation_id INTEGER, ordinal INTEGER,
  user_message TEXT NOT NULL, refined_user_message TEXT,
  entry_workflow_name TEXT, entry_context TEXT,
  status TEXT NOT NULL,                       -- completed|failed|awaiting_user|cancelled|abandoned
  success INTEGER NOT NULL,                   -- 1 = every command in the turn succeeded
  failure_reason TEXT, answer TEXT,
  conversation_summary TEXT, conversation_traces TEXT,
  started_at TEXT, completed_at TEXT, suspended_ms INTEGER,
  continuation_of TEXT, record_version INTEGER NOT NULL,
  record_json TEXT NOT NULL);                 -- full TurnResult (see below)

CREATE TABLE feedback (
  turn_key TEXT PRIMARY KEY, feedback_json TEXT NOT NULL, updated_at TEXT NOT NULL);

CREATE TABLE spans (
  span_id TEXT PRIMARY KEY, trace_id TEXT NOT NULL,   -- trace_id = turn_key
  parent_span_id TEXT, name TEXT NOT NULL,
  kind TEXT NOT NULL,                         -- internal|llm|human_wait|tool
  channel_id TEXT, command_name TEXT, context TEXT,
  start_ns INTEGER NOT NULL, end_ns INTEGER,  -- epoch ns; end_ns NULL = still open
  status TEXT NOT NULL,                       -- open|ok|error|cancelled|awaiting_user
  attributes TEXT NOT NULL);                  -- JSON object

CREATE TABLE artifacts (
  artifact_id TEXT PRIMARY KEY, turn_key TEXT NOT NULL, channel_id TEXT,
  span_id TEXT, key TEXT NOT NULL, content_type TEXT,
  size_bytes INTEGER, sha256 TEXT, inline_value BLOB, error TEXT);

CREATE TABLE train_runs (
  run_id TEXT PRIMARY KEY, workflow_fingerprint TEXT, started_at TEXT,
  completed_at TEXT, metrics_json TEXT NOT NULL);

CREATE TABLE diagnostics (                    -- writer health, schema markers
  key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL);
```

Indexes exist on `spans(trace_id)`, `spans(command_name)` (partial — only
rows with a command_name),
`turns(channel_id, conversation_id, ordinal)`, `turns(status)`,
`artifacts(turn_key)`. `turn_key` is
`YYYYMMDDTHHMMSS.ffffffZ-<12hex>` — lexicographic order is chronological
order, so `ORDER BY turn_key` sorts by time.

## Span catalog

`trace_id = turn_key` links every span to its turn; `parent_span_id` builds the
tree. Attribute values over the cap (`FW_OBS_MAX_ATTR_BYTES`, default 16 KiB)
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
`ask_user_response`.

### `fw.agent.execute` (kind `internal`) — the ReAct loop as a phase
Sibling of `fw.planner.plan` under the turn; NOT `fw.command.execute` (that is
one command inside a tool call — this is the whole loop). Attributes include
`agent_input`, `resumed`, `attempts`, `suspended`, `final_answer`.

### `fw.agent.step` (kind `internal`) — one reasoning step
Child of `fw.agent.execute`; the step's reasoning `fw.llm.call` and its
`fw.agent.tool_call` nest under it. Attributes: `step_index`, `thought`,
`tool_name`, `tool_args`, `observation`; on failures `error_type`/`tool_error`
and `recovered`; on suspension `clarification` (status `awaiting_user`).

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
| `list_turns(channel_id=, conversation_id=, status=, success=, command_name=, context=, limit=, offset=)` | Turn rows newest-first, without `record_json` (`context` is a substring match; `command_name` matches via spans) |
| `get_turn(turn_key)` | The full row incl. `record_json` (parse it yourself) |
| `get_spans(...)` | Span rows for one turn, ordered by `start_ns` (`attributes` is a JSON string). Pass the turn key POSITIONALLY — the parameter is named `trace_id` |
| `list_conversations(channel_id=, limit=, offset=)` / `list_channels()` | Navigation |
| `get_artifact(artifact_id)` | Offloaded artifact row (`inline_value` is bytes) |
| `list_train_runs(limit=)` | Training-run metrics rows, newest first (`metrics_json`) |
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

- All persisted text passed the redaction pass (credential shapes + loaded
  secret env values become `[REDACTED]`).
- Turn records are near-lossless; spans are best-effort under load — check
  `writer_health()` (`spans_dropped`, `records_dropped`, `write_errors`)
  before reading absence as evidence.
- `--generate_insights` CLI turns contain teacher AND student passes in one
  trace (duplicate-looking tool calls are expected there).

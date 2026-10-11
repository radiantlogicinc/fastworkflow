# Observation offloading and the archive

Large, older command results can be saved outside the ReAct prompt. The text is
kept in the workflow's archive and the prompt keeps a short label in its place.
This page covers that offloading, the labels, the archive and answer-time
rehydration.

## Status of `search_memory` (disabled 2026-10-09)

`search_memory` was an agent tool that answered a question about one earlier
`execute_workflow_query` observation, named by its alias (for example `O8`). It
read the archived text with a separate model call. It is **disabled**: in live
runs it retrieved nothing useful.

- Code: `fastworkflow/observation_offloading/search.py` (kept as is). The tool
  wrapper is `search_memory` inside `initialize_workflow_tool_agent` in
  `fastworkflow/workflow_agent.py`.
- The agent's tools are `what_can_i_do`, `execute_workflow_query` and `ask_user`.
- To re-enable it, uncomment `search_memory,` in the `tools` list of
  `initialize_workflow_tool_agent`. The search model is then read from
  `LLM_OBSERVATION_SEARCH`, with `LITELLM_API_KEY_OBSERVATION_SEARCH` as its key.
  Nothing else changes; offloading, labels and the archive work the same with or
  without the tool.

Without the tool, the agent cannot read an offloaded observation directly. It is
told to run the command again if it needs a value from one (see
[Answer-time rehydration](#answer-time-rehydration) for the final answer).

## Offloading

Offloading is triggered when the whole executor ReAct prompt (instructions,
inputs, command list and trajectory) exceeds `FW_TRAJECTORY_MAX_BYTES` (28,000 B
at the reference 131,072-token window). Observations are then offloaded
oldest-first until the prompt is back under the target. Two rules limit this:

- The **five newest execute observations are never offloaded**
  (`RECENT_OBSERVATIONS_PROTECTED`). A protected observation is never priced, so
  no label is built for it.
- An observation is offloaded only when replacing it with its label frees at
  least `FW_OFFLOAD_MIN_SAVING_BYTES` (1,024 B), and only when the label is
  shorter than the original in both characters and UTF-8 bytes.

The saving is measured on the command response alone, with the printed alias
line excluded:

```
utf8_bytes(command response, alias line stripped)
    - utf8_bytes(that step's actual offload label)  >=  1024
```

Decision records report `reason: below_min_saving` with `offload_saving_bytes`
and `label_size`. `FW_OFFLOAD_MIN_SAVING_BYTES=0` offloads whenever the label is
smaller; a value that is not a non-negative integer logs a warning and uses the
default.

If irreducible evidence still exceeds the target after compaction, the runtime
records the overage and continues. The continuation byte measure counts
observation bytes only, so the prompt can run a few hundred bytes over the
target.

### Labels

An offloaded observation is replaced by a label of this form:

> Offloaded observation O8 returned by show_holders. It contains identity_uid: identities holding this permission; label: their display names. Normally restored for the final answer.

The label carries the step's alias, the full command argument and the command's
authored `Output` descriptions. When those descriptions are unavailable, it says
it describes the beginning of the command output. The closing sentence is the
per-label restore reminder (`labels.LABEL_RESTORE_MARK`). The promise itself
("normally restored", and that the agent re-runs a command when it needs a value
from an offloaded observation) is stated once, in the `WorkflowAgentSignature`
docstring, not in every label. Labels in the earlier wordings still parse, so
recorded trajectories resume.

### Canonical observation handles

Every `execute_workflow_query` observation is printed with its handle on the first
line, inline results included:

> Observation O42 (execute_workflow_query ran in global)
> 477 holder(s).
> ...

`O{n}` is the ReAct step index of the execute observation: step 0 prints `O0`.
The handle line, the offload label and the archive row use the same alias. Other
tool outputs (`ask_user`, `what_can_i_do`) get no handle.

#### The context the command ran in

When the command ran inside a non-root command context, the line also names that
context and, where the workflow declares one, its instance identity:

> Observation O22 (execute_workflow_query ran in Account e8a0c3a1-… Jane Roe)
> permission_uid  label
> 85cde168  Item Catalog_Cloud Administrator
> ...

The context is the one the command **ran in**, read before the dispatch. A
command that moved the context says so with `; and resulted in a context change`,
and the command's own response states where it moved to. The instance identity is
declared by the context class (a classmethod `instance_label(...)` or an
`instance_label_attr`, see `fastworkflow/context_identity.py`), never derived; a
context with no declaration prints its name alone. Backend text is never
inspected, and there is no fallback to another step's alias.

The clause is presentation only. The archived text, its digest and the offload
label all describe the command response without the handle line. The clause is
stored in the `offload_evidence` row (`context_clause`, `context_changed`), so a
resumed turn prints the same line.

## Archive

Persistence does not wait for an offload decision. When a step completes, every
`execute_workflow_query` response is archived once under its `O{step_index}`
alias in the workflow's scoped SQLite archive (the `offload_evidence` table of
`observability.sqlite3`). Offloading is therefore only a decision about whether
the text stays in the prompt; the text is always stored.

- What is stored is the raw command response, with the presentation line removed,
  and its `text_sha256`. Writes are insert-or-nothing; a repeated write keeps one
  row.
- Each first write is recorded as an `observation_archived` event.
- A failed write records `archive_refused` (`reason: persistence_failed_original_retained`),
  leaves the observation inline and raises nothing into the agent loop. An
  observation that failed to persist is never replaced by a label.
- An alias with no stored row is an explicit miss, never a guess at a nearby step.

## Answer-time rehydration

The extract step that writes the final answer has no tools, so an offload label
is all it would see unless the text is put back. Answer-time rehydration puts
offloaded observations back in full on the extractor's copy of the trajectory,
within the answer evidence budget (`FW_ANSWER_REHYDRATION_MAX_BYTES`, about 250 KB
at the reference window). See [`answer_rehydration.md`](answer_rehydration.md).
When the budget is reached, the oldest observations stay as labels, and the
answer names them. The line naming them is appended after the budget, so it is
not counted against it.

## Configuration

Budgets are derived from the model's context window in
`fastworkflow/context_budget.py`; see
[`docs/context_budget.md`](context_budget.md). The values below are what a
131,072-token window (`cerebras/gpt-oss-120b`, the reference main agent model)
produces. Each name remains an optional tuning override.

| override | derived at a 131,072-token window | meaning |
|---|---|---|
| `FW_OFFLOAD_MIN_SAVING_BYTES` | 1024 | minimum UTF-8 bytes an offload must free |
| `FW_TRAJECTORY_MAX_BYTES` | 28000 | executor prompt target: bounds the whole executor ReAct prompt; observations are offloaded oldest-first when it is over |

Offloading has no on/off switch. A turn runs in one ReAct loop until the agent
selects `finish` or the iteration ceiling is reached (default 25 steps,
`DEFAULT_MAX_ITERS` in `observation_offloading/agent.py`). The one evidence
setting is `FW_OFFLOAD_EVIDENCE_REDACTION`: `on` (the default) or `off`. See
*Retention, redaction and known limits*.

## Retention, redaction and known limits

**Where the evidence lives.** In the workflow's own observability database,
`<FASTWORKFLOW_STATE_ROOT>/workflows/<workflow-id>/observability.sqlite3`, in the
`offload_evidence` and `offload_events` tables. Every row is keyed by the turn that
produced it and carries that turn's `channel_id`. The database is created
owner-only (the file `0600`, its directory `0700`).

**Evidence lives and dies with its turn.** `forget_channel`, Clear conversations
and retention pruning delete a turn's evidence and events in the same transactions
that delete its turn record.

**Redaction happens when the evidence is written.** With
`FW_OFFLOAD_EVIDENCE_REDACTION=on` (the default), a command response is stored as
the trace sink's credential scrub leaves it: it redacts, it does not truncate.
Event text is protected the same way. `off` stores responses and events verbatim,
and each row records which mode produced it. Answer-time rehydration reads the
stored, redacted text, so a resumed turn sees what it saw.

**Subject clauses are not redacted.** `context_clause` holds a context name and an
instance label, stored in the clear. Treat that column as unredacted if labels can
carry anything sensitive.

**Older evidence files are deleted, not imported.** Earlier builds kept evidence
in a separate `.offload-handles.sqlite3` file beside the database. Opening the
store deletes that file and nothing in it is carried over.

**Pruning runs once per process start.** Evidence is pruned on the store's age
horizon and size cap (`FW_OBS_RETENTION_DAYS`, `FW_OBS_DB_MAX_BYTES`), one whole
turn at a time. A long-lived process does not prune again while it runs.

**A program that embeds the library gets the same record.** A
`WorkflowExecutionContext` opens the bound app workflow's own sink when
`bind_app_workflow()` runs. Passing `tracing.NoOpTraceSink()` records no spans, but
it does not turn offloading off: evidence and events are still written to the
workflow's `observability.sqlite3`.

**Worst-case agent work in one turn.** A turn runs at most four segments of 25
decisions each, plus the three continuation-planner calls that open the later
segments: 100 tool-or-finish decisions and three extra model calls before the turn
is forced to answer.

**A garbled model reply after a tool has run fails the turn.** The call is retried
only while the turn has executed nothing. Once any observation exists the turn
fails, because replaying the trajectory would re-run commands that already ran.

**The rehydration control note is added after the budget.** It costs a few hundred
bytes at most, and it is worth more than the evidence those bytes would have
bought.

**An evicted suspended session keeps a little memory until the process exits.**
The in-process event buffer is capped at 2,000 events, so the residue is small and
bounded.

**The finish check's thresholds were calibrated on one workflow.** `FLAG_MIN`,
`ASK_MIN`, the question wording and the published precision and recall come from
attempts on a single workflow with the default model. Another workflow, model or
provider is unmeasured.

## Validation

- `tests/test_observation_offloading.py` covers the 1 KB rule (exact savings at
  1,023 / 1,024 / 1,025 B), the recency protection, the printed handle and context
  line, the eager archive and the environment overrides.
- `tests/test_answer_rehydration.py` covers answer-time rehydration.
- `tests/test_observation_search.py` covers the kept search code. Its provider
  tests are opt-in:

```bash
FW_TEST_OBSERVATION_SEARCH_LIVE=1 python -m pytest \
  tests/test_observation_offloading.py tests/test_observation_search.py
```

Without the opt-in, deterministic checks run and paid cases skip.

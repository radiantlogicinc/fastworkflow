# Changelog

Releases before 3.4.0 were announced in their merge-commit subjects
(`feat: v3.2.0 — observability store, chatbot debug UI, …`) and are recoverable
with `git tag` and `git log --first-parent main`. This file starts at 3.4.0; it
does not backfill them.

## 3.4.0 — observation offloading and search

**Observation offloading and answer-time rehydration become the framework's
behaviour for every workflow.** They are no longer modes a deployment opts into.
A fastWorkflow tool agent compacts its trajectory, keeps every observation
reachable, and answers over the evidence behind its labels rather than over the
labels themselves.

### Added

- **Observation offloading** (`fastworkflow.observation_offloading`): canonical
  `O{n}` aliases on every execute observation, an eager SQLite archive, a packed
  trajectory target with offload labels, `search_memory` over one stored
  observation, and segmented continuation with forced replans.
- **Context-instance line** (`fastworkflow.context_identity`): every execute
  observation names the context instance the command ran in, so a listing
  produced inside a context is still attributable to that instance by a reader
  that cannot use the order of the commands.
- **Answer-time rehydration** (`fastworkflow.answer_rehydration`): the extract
  call is given its own copy of the trajectory with the evidence behind offload
  labels put back, under a byte budget, so the answer is written over evidence
  rather than over pointers.
- **Finish reminder**: a `finish` action that never opened a named item of the
  request goes back to the loop once, and only while iterations remain.
  (Replaced in this release by the finish-time execution check below; the
  roster nudge that implemented it, `fastworkflow.answer_coverage`, is removed.)
- **Finish-time execution check** (`observation_offloading/finish_check.py`,
  fix-4dsr): when the agent chooses `finish`, a decision model judges, for
  every step of the turn's initial plan and every subject the request names,
  whether the turn's record shows the step carried out; unexecuted steps are
  named in one note and the agent returns to the loop -- once per turn, only
  with iterations left, and it may finish anyway. It checks execution, not
  whether the request was answered. Off unless `FW_FINISH_CHECK=jev` and
  `JEV_API_KEY` are both set (it sends the request, the plan and a redacted
  summary of every step to TypeSafe); fails open to an unchecked finish; one
  `finish_check` offload event per finish and an `fw.finish_check` span.
  Measured offline on 82 recorded ido attempts it was not tuned on (910
  labelled step x subject pairs, pre-registered): precision 0.77, recall 0.89.
  Install with the `jev` extra.
- **Structured plan** (`fastworkflow.turn_plan`): with the finish check on, the
  planner returns its steps (commands, sub-steps, optional and user-gated
  flags) and the request's subjects as data, and the agent still reads a
  numbered list. A plan that does not parse falls back to the plain-text
  planner. Without the check the planner is unchanged, because the structured
  call is slower (median 12.1 s against 3.7 s on three todo-list requests with
  `cerebras/gpt-oss-120b`).
- **Context budgets** (`fastworkflow.context_budget`): one input — the model's
  context window — and every byte budget derived from it as a fixed fraction.
  `budget_provenance()` returns the input, its source and every budget.
- **Searches answered without the search model.** An archived observation of at
  most 256 bytes is returned verbatim with up to three better-matching handles
  of the turn (status `short_verbatim`). A request for every row of a listing is
  answered by copying the rows (`observation_offloading/listing.py`: aligned,
  markdown and tab-separated listings, served only when the parse is provably
  complete), cut at the answer bound with a line saying how many were shown
  (status `rows_served`).
- **Optional search router** (`observation_offloading/search_router.py`, the
  `router` extra): a decision-model call (TypeSafe Jev) that decides whether a
  search over a listing wants every row. Off unless `FW_SEARCH_ROUTER=jev` and
  `JEV_API_KEY` are set; what it sends is redacted by the capture policy; one
  attempt, a 2-second timeout, and it fails open to the search model. Every
  search event records its verdict. A workflow may add examples in
  `search_router_examples.json`.
- **`fw.search.route` span** (contract v1) for each routing call under a traced
  turn: the verdict, latency and tokens, never the question or the evidence.

### Changed

- **Known-name guard**: a known command name is never answered by a context
  that does not own it; the declining prediction carries a hint naming where the
  command lives. When no context on the chain owns it, that hint is the whole
  response and the next message goes through ordinary intent detection, rather
  than `you_misunderstood` and its clarification stage, which only matches the
  current context's commands and so could not route the command the hint names.
  The guard also covers a reply to `you_misunderstood`, which is matched against
  the same full command set; the ambiguity stage, which matches a short
  suggestion list, is still excluded. With several owners, the hint names each
  entering command beside the context it enters.
- **Threshold separation**: `write_ambiguity_thresholds` is the single writer
  for both ambiguity files and establishes a non-empty ambiguity band where the
  artifacts are produced, with `TIER_AMBIGUITY_MIN_SEPARATION` and
  `SINGLE_LABEL_RESOLUTION_FLOOR`. A tier threshold of exactly 1.0, which a
  float32-saturated sweep can pick, publishes the flat 0.99 cap instead of
  aborting the workflow's training run.
- **Workflow fingerprint scope rule v2**: a root `benchmarks/` tree and the
  runtime observability store leave `workflow_content_entries`.
  `WORKFLOW_SCOPE_RULE_VERSION` 1 → 2, so a v1 declaration reads as
  `incomparable`, not stale.
- **Synthetic utterance generation** sends `temperature` only; `top_p` is gone,
  because current Bedrock Claude models reject both together and every
  generation call was a hard `BadRequestError`.
- `fw.nlu.intent` span contract v2 → v3.
- `fw.command.execute` span contract v2 → v3: the four auto-navigation
  attributes are gone with the two-step dispatch that wrote them.
- The aggregate span-contract version is 5. Version 4's note named
  auto-navigation keys that `SPAN_CONTRACTS` does not contain; 5 is the number
  that matches those contracts. It is now 6: `fw.search.route` was added and no
  existing span changed.
- **Offload labels say the evidence comes back.** A label reads "Offloaded
  observation O… returned by … It is restored in full when the final answer is
  written, so search it with search_memory only for a value you need for your
  next step", and the agent's instructions say the same. The earlier wording,
  which opened with an instruction to search, is still recognised. In recorded
  runs this cut requests to search whole tables by 70-95%.
- **Answers about an incomplete listing name a real narrowing input.** The
  search signature gains a `narrowing` input listing the producing command's
  optional, non-selecting inputs; an answer that needs rows not shown ends with
  `To reach them: <command> <input>=<value>`, or says a narrower re-run is
  needed when the command declares none.
- **Continuation allows 3 forced replans** (4 segments, a 100-step ceiling),
  up from 2 (3 segments, 75 steps).
- **Intent `signal_version`** no longer carries a threshold-semantics segment.
  It now reads `intent-classifier/<artifact version>/...`, so a version string
  identifies the artifact behind a signal and nothing else.
- **Offloading evidence lives in the observability store.** Archived
  observations, their subjects and the offloading runtime's diagnostic events
  are rows of `offload_evidence`, `offload_subjects` and `offload_events` in the
  workflow's `observability.sqlite3` (feature markers `offload_evidence_v1`,
  `offload_events_v1`; no schema-version bump). They are keyed by turn, and
  `forget_channel`, Clear conversations and retention pruning delete them in
  the same transactions that delete the turn record. Events are read back with
  `ObservabilityStore.offload_events(...)`.
- **Evidence is redacted when it is written.** With
  `FW_OFFLOAD_EVIDENCE_REDACTION=on` (the default) responses and event text are
  stored as the trace sink's credential scrub and capture policy leave them;
  `off` stores them verbatim for development. The process running a turn keeps
  the raw text of its redacted observations in memory until the turn is over,
  so the agent's own reads stay exact; a turn resumed in another process reads
  the redacted text.
- **Observability recording is always on**, for fastWorkflow's entry points and
  for programs that embed the library alike. The database is owner-only (0600
  file, 0700 directory) and pruned by `FW_OBS_RETENTION_DAYS` and
  `FW_OBS_DB_MAX_BYTES`. `get_observability_sink()` no longer takes
  `entry_point`, and returns `None` only when the store cannot be opened.
- **An execution context opens its own sink.** A `WorkflowExecutionContext`
  given no sink opens the bound app workflow's sink in `bind_app_workflow()`,
  and follows a rebind to another workflow's database. A sink passed to the
  constructor or to `set_trace_sink()` is never replaced, and an explicit
  `tracing.NoOpTraceSink()` records no spans or turn records;
  `set_trace_sink(None)` returns the context to its automatic sink. Offloading
  evidence is written to the workflow's database whatever the sink, and the
  offloading archive prunes that database once per process when it opens it.
- The `search_memory` input bound has no tuning override; it is derived from the
  search model's context window only. It is a quarter of that window (131,072
  bytes at the reference window), up from 3/128 (12,288 bytes).
- **An observability database from an older schema version is replaced, not
  refused.** The store has not shipped in a release before, so such a file can
  only be a local development database: opening it with the writer deletes it
  (and its `-wal`/`-shm`) and creates a fresh store in place, with a warning
  naming the path and both versions. A database from a newer build is still
  refused and never touched, and the read-only viewer refuses an older one
  without deleting it.

### Fixed

- A malformed model reply that arrives after a tool has already run now fails
  the turn instead of silently re-running the whole trajectory. The re-run could
  repeat the side effects of the commands already executed and leave the
  archived evidence out of step with the answer.
- A known command name followed by a newline or a tab is now recognised by the
  known-name guard and by the owning context's exact match, not only when the
  name is followed by a space.
- A command whose name has capital letters is matched by its own context's
  exact match whatever case it is typed in, and is never refused by the
  known-name guard as belonging to another context. The guard compared a
  lowercased name against the context's command names as spelled, so it named
  the current context as the foreign owner.
- A failure while releasing the previous turn's process-local evidence no
  longer aborts the turn that is starting.
- An observability database that no trace sink ever opens -- a context built
  with `tracing.NoOpTraceSink()` -- is now pruned too: the offloading archive
  prunes each database once per process when it first opens it. Its offload
  tables used to grow without bound.
- The finish-time roster nudge no longer fires in a workflow whose contexts
  declare no instance identity, where every named item looked untouched, and
  it no longer tells the agent that nothing about those items was retrieved;
  it says only that they were never the subject of a command. (The roster
  nudge was then removed in favour of the finish-time execution check.)
- The foreign-context hint no longer ends with "Then run it there" when no
  entry command is declared; it says to move into the owning context first.
- Replanning and answer-time rehydration apply the agent's own execute
  numbering to a handle line or offload label, as compaction already did, so a
  command response shaped like one on a step that was never annotated is kept
  as response and never resolved to another observation's evidence. Each such
  line is recorded as a `foreign_line_ignored` event.
- The replan skeleton's policy label is `greedy_trajectory_budget`; it named
  a fixed 28 KB bound it does not use outside the reference window.
- A reply to `you_misunderstood` that matches none of the current context's
  commands now lists what can be done there. It used to raise
  `KeyError: 'what can i do?'`, because the fallback it substitutes was
  registered only for the ambiguity clarification stage.

### Migration

- `LLM_OBSERVATION_SEARCH` is the recommended setting for the model that answers
  `search_memory`. When it is unset, search runs on `LLM_AGENT`, which also
  fixes the budget the evidence is cut to, since that budget is sized from the
  search model's own context window.
- `LITELLM_API_KEY_OBSERVATION_SEARCH` is the recommended credential for that
  role. When it is unset, search uses the credential configured for `LLM_AGENT`.
- Offloading evidence now lives in `observability.sqlite3`. An evidence file
  left by an earlier build beside it (`observability.sqlite3.offload-handles.sqlite3`,
  its write-ahead-log files and any `.preserve` marker) is deleted the first time
  the store opens; its contents are not migrated.
- Experiment evidence is no longer preserved: Clear conversations, and
  forgetting an experiment run's channel, erase its offloading evidence like any
  other turn's.
- Recording can no longer be turned off. A program that embeds the library gets
  the same owner-only, pruned record as the entry points; to keep it elsewhere,
  set `FASTWORKFLOW_STATE_ROOT`. A `WorkflowExecutionContext` built without a
  sink now records into its workflow's database on its own; pass
  `trace_sink=tracing.NoOpTraceSink()` where a context must record no spans or
  turn records. That does not stop observation offloading, which still writes
  its evidence, subjects and events to the same database.
- The offloading runtime's events are not written to a separate file; read them
  with `ObservabilityStore.offload_events(turn_key=..., channel_id=..., kind=...)`.
- To correct the `search_memory` input bound, set `FW_MODEL_CONTEXT_TOKENS` or
  point `LLM_OBSERVATION_SEARCH` at the intended model.
- The roster nudge is gone: without `FW_FINISH_CHECK=jev` and `JEV_API_KEY` a
  `finish` is never interrupted. `FW_EVAL_FINISH_REMINDERS=0` still turns the
  note off when the check is on. `fastworkflow.answer_coverage` no longer
  exists, and a step span marks a note with `finish_check_note` where it used to
  say `roster_nudge` (`fw.agent.step` v3; span contract 7).

### Known limits

Offloading and search ship with documented limits rather than silent ones: what
a turn resumed in another process reads, which part of the evidence is stored
unredacted, when pruning runs, the worst-case agent work one turn can cost, and several places where a disclosure
line can push a bounded input a few hundred bytes past the budget it reports.
They are listed in
[Retention, redaction and known limits](docs/observation_search.md#retention-redaction-and-known-limits);
read that section before sizing a deployment or writing a retention policy.

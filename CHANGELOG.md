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
  `jev` extra; `router` is kept as a deprecated alias of it): a decision-model call (TypeSafe Jev) that decides whether a
  search over a listing wants every row. Off unless `FW_SEARCH_ROUTER=jev` and
  `JEV_API_KEY` are set; what it sends is redacted by the capture policy; one
  attempt, a 2-second timeout, and it fails open to the search model. Every
  search event records its verdict. A workflow may add examples in
  `search_router_examples.json`.
- **`fw.search.route` span** (contract v1) for each routing call under a traced
  turn: the verdict, latency and tokens, never the question or the evidence.

### Changed

- **Structured planning is disabled** (2026-09-28, owner decision; supersedes
  the "Structured plan" entry under Added and the later structured-planner
  notes in this release). The planner always makes a plain-text plan, with the
  finish check on or off; with it on, the check's plan is `parse_text_plan`'s
  (`source="text"`, no subjects). The structured signatures, the
  `STRUCTURED_PLAN_GUIDE` prompt, `turn_plan.render`, the structured adapter,
  the zero-steps / parse-error fallback and the planner span's subject
  redaction are commented out, not deleted, and not behind a flag: nothing
  re-enables them. `fw.planner.plan` / `.replan` keep contract v2 (and
  `SPAN_CONTRACT_VERSION` 8): `plan_source` is only `text` or `none`, and
  `subjects` is always `[]`. The `TurnPlan` / `PlanStep` / `PlanSubject`
  models, the plan's persistence across a resume (fix-ju1v) and the
  adapter's `use_json_adapter_fallback` parameter (fix-6jzi) are unchanged.
  The tests of the structured path are skipped, not removed.
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
  that matches those contracts. It became 6 when `fw.search.route` was added
  (no existing span changed), 7 with `fw.finish_check` and `fw.agent.step` v3,
  and is now 8: `fw.planner.plan` / `fw.planner.replan` v2 (fix-3x76).
- **Offload labels say the evidence comes back.** A label reads "Offloaded
  observation O… returned by … It is restored in full when the final answer is
  written, so search it with search_memory only for a value you need for your
  next step", and the agent's instructions say the same. The earlier wording,
  which opened with an instruction to search, is still recognised. In recorded
  runs this cut requests to search whole tables by 70-95%. (Since fix-2hxv a
  label ends with the short "Restored in full for the final answer."; the
  promise and the search guidance are stated once, in the agent's instructions
  and the `search_memory` description. Both earlier wordings still parse.)
- **Answers about an incomplete listing name a real narrowing input.** The
  search signature gains a `narrowing` input listing the producing command's
  optional, non-selecting inputs; an answer that needs rows not shown ends with
  `To reach them: <command> <input>=<value>`, or says a narrower re-run is
  needed when the command declares none.
- **Continuation allows 3 forced replans** (4 segments, a 100-step ceiling),
  up from 2 (3 segments, 75 steps). `FW_MAX_FORCED_REPLANS` moves it (see
  below).
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
  bytes at the reference window), up from 3/128 (12,288 bytes). (Superseded
  in part by fix-deus below: the derived value is now capped at 131,072 bytes
  and `FW_SEARCH_OBSERVATION_MAX_BYTES` overrides it.)
- **An observability database from an older schema version is replaced, not
  refused.** The store has not shipped in a release before, so such a file can
  only be a local development database: opening it with the writer deletes it
  (and its `-wal`/`-shm`) and creates a fresh store in place, with a warning
  naming the path and both versions. A database from a newer build is still
  refused and never touched, and the read-only viewer refuses an older one
  without deleting it.
- **The Jev features share one client module** (`observation_offloading/jev_client.py`,
  fix-xeg1): the optional SDK import, `JEV_API_KEY`, the default model,
  redaction of what is sent, one-attempt clients and a per-process cache, for
  both the search router and the finish check.
- **A Jev feature that is asked for but cannot start says so** (fix-jdcm).
  `FW_SEARCH_ROUTER` or `FW_FINISH_CHECK` set to anything other than `jev`, set
  without `typesafe-sdk` installed, or set without `JEV_API_KEY`, logs one
  warning per cause per process and the feature stays off; an unset flag is
  still silent. Clients are cached by a fingerprint of the key, never the key,
  so a changed key builds a new client instead of reusing the old one. (More
  causes since fix-2so9 and fix-7tz5, and a flag may name a registered
  provider since fix-5uva; see below.)
- **The forced-replan bound is a setting** (fix-vd5c): `FW_MAX_FORCED_REPLANS`
  overrides the default of 3, clamped to `[0, 10]`; a non-integer is ignored,
  with a warning either way. `forced_replan` and `forced_replan_wall` events
  carry `reached_limit`, and `agent_installed` records the bound in force. The
  default rests on one ido task, not on a measured tail across workflows.
- **Planner spans say which planner produced the plan** (fix-3x76):
  `fw.planner.plan` / `fw.planner.replan` v2 add `plan_source` (`structured`,
  `text`, `text_fallback`, `none`) and the structured plan's `subjects`,
  redacted by the capture policy. A `finish_check` event also carries a score
  table, one entry per checked step x subject (x part), capped at 200.
- **`finish_check` events are recorded for every finish an attached check
  sees** (fix-7pba), with a reason even when nothing was asked: `disabled`
  (`FW_EVAL_FINISH_REMINDERS=0`), `cap reached`, `nothing to check` (every step
  optional or user-gated), `no plan`, `no room to act`, `error`. The step count
  the note states includes the room later forced-replan segments still hold
  (`shown_iterations_left`); the two-iteration gate is still the current
  segment's.
- **Stored `finish_check` events name steps and subjects by index**
  (fix-7scj): `flagged_steps` holds the step index, the subject's index into the
  plan's subjects, its kind and `p_unmet`, never plan-step text or subject
  names.
- **Search events say which path a search took** (fix-wheb): `router_enabled`,
  `listing_parsed`, `listing_shape`, `listing_skip_reason` and `for_report` on
  every search event, so a `router` of `None` is never ambiguous;
  `related_scores` and `subject_recorded` on `short_verbatim`;
  `served_over_bound` and `trailing_lines_dropped` on `rows_served`. The
  search-observation budget moves into `context_budget` as
  `SEARCH_OBSERVATION`, and `budget_provenance()` reports it as
  `search_observation_max_bytes` beside `search_window_tokens` and
  `search_window_source`, the window it is cut from.
- **The `search_memory` description matches the path the agent has**
  (fix-8uj0): it is built with the agent, states the read bound as the byte
  count the search will apply, and, when a router is attached, says an all-rows
  search may be answered by copying rows instead of forbidding such searches.
  The router's `for_report` verdict is recorded and does not change the path.
- **Offload labels and served listings are shorter** (fix-2hxv): both end with
  "Restored in full for the final answer." instead of the full restore promise,
  which cost ~100 bytes per label and made fewer observations worth offloading.
  (Since fix-94m9 the mark is "Normally restored for the final answer."; see
  below.)
- **Narrowing inputs include defaulted ones** (fix-swqi): an input with a
  non-None default (`limit: int = 50`) is offered as a way to narrow a listing,
  not only one typed `Optional`. A required field's sentinel default
  (`NOT_FOUND`, `INVALID_INT_VALUE`, `INVALID_FLOAT_VALUE`) does not count.
- **Fewer archive reads** (fix-314g): `list_summaries` measures stored rows
  with `length()` on the BLOB, without reading their content, and an alias the
  archive has no subject for is remembered (bounded) instead of re-read on every
  related-handle lookup.
- The settings above, `FW_SEARCH_ROUTER_MODEL`, `FW_FINISH_CHECK_MODEL` and
  `JEV_API_KEY` are in the example env templates, commented out, with a
  data-egress warning (fix-9l9e); stale docs, comments and this changelog's
  span-contract number are corrected (fix-42ig).
- **Vendor calls have hard wall-clock bounds** (fix-i94q): the router's 2 s,
  the finish check's 4 s per call and 8 s per check are enforced during the
  call, not checked before it, and the check's 8 s now include building the
  ledger. Each call runs on its own daemon thread, at most 4 process-wide; a
  call with every worker busy fails at once as `VendorBusy`, one cut off at its
  cap is `CallTimedOut`, one cut off by a deadline or budget is `OutOfTime`. An
  abandoned request is never retried and never delays interpreter exit, but
  may still be billed vendor-side. The SDK's own timeout is the cap plus 0.5 s.
- **One vendor budget per turn** (fix-sotm): 10 s of time actually spent on
  vendor calls, router and finish check together, and at most 3 routing calls.
  Past either, `route` returns `router_budget` with no call and no span, the
  search model answers, and the search records `listing_skip_reason`
  `router_budget`. `vendor_ms` (the turn's running total) is on every router
  record and on a checked `finish_check` event.
- **The Jev endpoint is `FW_JEV_BASE_URL`** (fix-2so9): default
  `https://api.typesafe.ai`; https only, or plain http to a loopback address;
  no credentials, query or fragment. A rejected value warns once, without the
  URL, and turns both features off; a non-default host is logged once. The
  SDK's `TYPESAFE_BASE_URL` is ignored with one warning, so it can no longer
  redirect the key and payloads.
- **`typesafe-sdk` is pinned exactly to 0.7.2** (fix-em0l), with an SDK
  contract test on a loopback stand-in as the review gate for a bump.
- **Jev failures are diagnosable** (fix-conv): an `error` `finish_check` event
  and a router error record carry `error_status`, `error_request_id`,
  `error_code` (one machine-readable token from the body, never its text) and
  `error_stage` (`ledger`, `request`, `budget`, `check`, `note` on the check;
  `redaction`, `request`, `budget` on the router). A failure is warned about at
  most once per 5 minutes per (feature, error type, HTTP status), counting the
  ones not logged, instead of once per process; a ledger-build failure is
  logged with its traceback. No span-contract change.
- **The Jev features stay off when the capture policy would withhold what they
  send** (fix-7tz5): under a capture profile that withholds command output
  (e.g. `evidence`), an unknown profile, or
  `FW_OFFLOAD_EVIDENCE_REDACTION=off`, one warning per cause. A declared field
  policy that keeps observations whole is still allowed. A value the policy
  withholds at call time sends nothing: the finish check records the new
  reason `policy_withheld` (unwarned, `requests` 0) and the router returns the
  error `policy_withheld` (`error_stage` `redaction`).
- **One decision says whether the finish check is on for a plan** (fix-rm98):
  `workflow_agent.finish_check_active(agent)`, read by the planner. With
  `FW_EVAL_FINISH_REMINDERS=0` nothing the check adds runs -- no note and no
  structured plan -- while the checker stays attached and records `disabled`.
  The switch is read from the workflow's `fastworkflow.env` first, then the
  process; any other set value still fails the agent build, and an empty value
  now counts as unset.
- **The structured planner runs only for a turn's initial plan** (fix-5vtw);
  replans after a parameter-extraction error or an `ask_user` reply use the
  plain-text planner, and the check keeps checking the initial plan. **Its
  call no longer takes DSPy's hidden JSON-adapter retry** (fix-7eu6), which
  dropped the available-commands prelude: an unparseable reply goes straight
  to the text planner, at most 3 calls instead of 6.
  `CommandsSystemPreludeAdapter` gains `use_json_adapter_fallback` (default
  `True`, unchanged for the agent loop and intent clarification). (Since
  fix-6jzi that retry, where it stays on, keeps the command list; see Fixed.)
- **Large finish checks are bounded** (fix-az9q): a plan naming more than 12
  subjects is checked step by step only (reason `subjects capped`, no subject
  name sent); the ledger is kept under 96 KiB by dropping rows' `head`, then
  `refers_to`, oldest first; an over-long ledger is halved at most once, and a
  half still too large is an `error`. Checked events gain `subjects_capped`,
  `ledger_bytes` and `ledger_rows_trimmed`. No recorded turn is trimmed (max
  52 KB).
- **The plan survives a cross-process `ask_user` resume** (fix-ju1v): with a
  checker attached, `turn_plan` and `turn_plan_status` are session state and
  the resumed agent's step record is seeded from the suspension. A "no plan"
  event carries `no_plan_cause` (`planner_empty`, `plan_unreadable`,
  `lost_on_resume`, `not_planned`); a resume from a suspension the
  context-window fallback had cut is skipped as `ledger incomplete`.
- **The finish-check note says to skip what the user declined, and not to
  repeat or bypass** (fix-hfbr, partial; fix-03lt). Its tail now reads "Run any
  that are still needed, or say in your answer why not. Do not repeat a change
  this turn's record shows was already made, and do not make a change the user
  has not confirmed: ask them instead. Skip any step the user has since
  declined or changed. You have N steps left." Every `finish_check` event
  carries `user_replies`, the turn's `ask_user` step count. The checked plan is
  still the initial one. fix-hfbr stays open on a framework-owned approval
  gate (fix-47a9). (Since 2026-09-28 that gate is deferred and fix-hfbr no
  longer waits on it; see the next entry.)
- **The finish check only asks about provably read-only steps** (fix-hfbr).
  A step is checked only if every command it and its parts name is declared
  `read_only` in the workflow's runtime manifest (the metadata registered at
  startup, else `workflow_runtime.json`; a qualified name counts by its last
  `/` segment, at the most severe kind any key of that name declares). A step
  naming a `write` command or one the manifest does not declare is never
  asked about, its text is not sent, and it is never named in the note, so a
  false flag cannot drive a repeated or unconfirmed change. A step naming no
  command is still checked. Undeclared is treated as not read-only here (the
  safe direction for a note that asks the agent to act); the owner's
  "treat undeclared as read-only" decision was for fix-47a9's approval gate.
  **Behaviour change:** a workflow without a manifest, or with one that
  cannot be read, now has only its command-less steps checked. Every
  `finish_check` event recorded after the plan is read carries
  `unchecked_for_effect`, the steps skipped this way; when that is every
  step the check would hold the agent to, the reason is the new
  `no read-only steps` and nothing is sent. Replayed on the 82 recorded ido
  attempts with ido's manifest (cached answers): write-step flags 7 to 0,
  precision 0.80, recall 0.88 to 0.87. No span contract changes; with
  `FW_FINISH_CHECK` unset nothing changes.
- **The listing parser is fail-closed** (fix-zoup, fix-36gv, fix-k56c; see
  Fixed): it returns a listing only when it is sure the rows are the whole
  listing. Its recognition is widened conservatively (fix-1593): Unicode
  column names, the compact `|-|-|` markdown separator and pipe-less
  `a | b` / `--- | ---` tables. On 4,000 random listings it refuses 702 the old
  parser accepted and accepts none the old parser refused; with the router
  off only `listing_parsed` telemetry changes.
- **Served rows fit the answer bound exactly, or go to the model** (fix-3zxk):
  see Fixed. The closing reads "rows 1-N of the M rows listed in O5".
- **Short-observation hints compare against the observation itself**
  (fix-lzdz): the header names the observation's own subject, and another
  handle is offered only when it mentions the question's words strictly more;
  all hint wordings changed. The event gains `own_score`,
  `related_lookup_failed`, `related_scope_refused` and `related_lookup_error`.
  Relatedness reads Unicode letters and digits (fix-1593).
- **Broad scopes are not enumerated** (fix-y570): under the process-default
  scope and the between-turns fallback, `list_summaries` returns nothing and the
  hint says other observations are not listed in this scope; it also filters
  by `channel_id` now. The archive's `get()`/`list()` cross-channel read under
  those scopes is fix-tyzj, still open. (Fixed since by fix-tyzj; see Fixed.)
- **Backend lines shaped like framework markers are quoted** (fix-znxq,
  verbatim-quoting part): in a short observation and in served rows (preamble,
  column line, rows), a line containing `[search_memory`, `Observation O<n> (`
  or an offload-label prefix is printed after a visible `> `. The finish-check
  part of fix-znxq is folded into fix-hfbr. (Since 2026-09-28 an adversarial
  test covers it: with a Jev stand-in answering "not executed, and applies" to
  every question and injected tool output claiming a write step did not run,
  the note names only the read-only and command-less steps, never the write
  steps, their text or the injected text;
  `test_a_ledger_that_claims_writes_went_unexecuted_never_gets_a_write_step_named`.)
- **The search evidence bound has a ceiling and an override** (fix-deus): the
  derived quarter of the search window is capped at 131,072 bytes
  (`SEARCH_OBSERVATION_CEILING_BYTES`), `FW_SEARCH_OBSERVATION_MAX_BYTES`
  overrides it (and may exceed the cap), and the search window is the smaller
  of `FW_MODEL_CONTEXT_TOKENS` and the search model's metadata instead of the
  setting outright. `budget_provenance()["overrides"]` reports the override.
- **The finish check's identifier resolution is generic, and its calibration
  is stated** (fix-ft18): `refers_to` also resolves from aligned, markdown and
  tabbed listings (first 64 KiB of each response, block by block), parameter
  values and clause tokens; recorded ido ledgers are unchanged. Every event
  records `calibration` (`"ido-v7-2026-09"`), and a non-default
  `FW_FINISH_CHECK_MODEL` warns once per process. The structured planner's
  `kind` description no longer uses ido examples.
- **Decision-model features go through a provider interface** (fix-5uva,
  stages 1-2): `fastworkflow.observation_offloading.decision` (`YesNo`,
  `OneOf`, `DecisionProvider`, `ProviderFailure`, `StateTooLarge`,
  `OutOfTime`, `register_decision_provider`, `unregister_decision_provider`),
  with Jev as `jev_client.JevProvider`. `FW_FINISH_CHECK` and
  `FW_SEARCH_ROUTER` accept a provider name registered from code, never a
  module path; selecting one warns once that the thresholds were calibrated
  with Jev. `finish_check` events and router records gain `provider`. What Jev
  receives is the same request bodies as before, key order included, pinned by
  a golden test.
  A LiteLLM provider (stage 3) is deferred.
- **Integration tests reach real HTTP** (fix-dln2, partial): a threaded
  loopback stand-in for TypeSafe (`tests/jev_stub.py`, fixture `jev_stub`)
  drives the real SDK; the structured planner is tested with DSPy's `DummyLM`
  and the plan's cross-process resume through real `process_turn` turns.
- **The restore promise says "normally"** (fix-94m9). Default path,
  agent-visible, and **not re-measured**: the offload-label and served-rows
  mark (`labels.LABEL_RESTORE_MARK`) is now "Normally restored for the final
  answer." (was "Restored in full for the final answer."; +1 byte per label);
  the `search_memory` tool description and the `WorkflowAgentSignature`
  docstring (agent and extractor instructions) say every observation "is
  normally restored in full" and add "If the answer's evidence limit is
  reached, the oldest observations are not restored and the answer names
  them."; and the rehydration note's prefix (`NOT_REHYDRATED_PREFIX`) is now
  "Not rehydrated for the answer (evidence exists under these observations,
  but the answer's evidence limit was reached; say in the final answer that
  the rows of these observations are not included in it): ", so the extractor
  states the omission. The note appears only when the evidence budget binds.
  Labels in every earlier wording still parse.
- **`search_memory`'s narrowing inputs come from the command dispatched**
  (fix-v9jy): `_execute_workflow_query` files the qualified command each
  execute alias ran (`remember_dispatched_command`, on
  `agent.dispatched_commands`, current turn's scope only; CME commands such as
  `abort` and `go_up` are not filed), and `describe_command_inputs` uses it
  when its bare name matches, so two contexts' same-named commands are told
  apart. Otherwise, and after a cross-process resume, it falls back to the
  previous rule. No prompt, schema or event field changes.
- **`search.py` is split** (fix-vpe3): the relatedness score, related-handle
  suggestions and short-observation answer move to
  `observation_offloading/related.py`; `search.py` re-exports every name. The
  search event is built once from a shared base, so its keys come out in
  another order with the same values (stored with `sort_keys`). No behaviour
  change.
- **The decision-model wording is pinned to its measurement** (fix-8ee5):
  `tests/fixtures/decision_measurements.json` records the offline measurement
  behind the finish check's and router's published figures (calibration
  `ido-v7-2026-09`, model, thresholds; finish check precision 0.767
  [0.722, 0.810], recall 0.888 [0.832, 0.936] on 910 pairs over 82 attempts;
  router all-rows precision 1.0 [0.970, 1.0] on 126, recall 0.962
  [0.914, 0.984] on 131, at threshold 0.5; each with n and 95% interval) and
  the sha256 of every evidence file. `tests/test_decision_measurements.py`
  re-renders every finish-check and router question template with fixed
  arguments and fails when a digest changes, even if the golden request bodies
  were regenerated to match. The evidence itself is not in the repo; see
  Migration.

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
- The finish check asks about the whole step when a step concerns a subject
  but none of its sub-steps does; such a step was never asked about, so it
  could not be flagged (fix-ba7k).
- Text plans parse a plan written on one numbered line ("1. a. 2. b. 3. c")
  into its steps, and read "optional" and "needs the user" from the whole step
  rather than its first 40 characters, without taking "with optional filters"
  for an optional step (fix-nzmy).
- A structured planner call that returns no steps falls back to the plain-text
  planner instead of leaving the turn unchecked, and a text plan that cannot be
  parsed leaves the check without a plan instead of failing the turn. With the
  check off no text plan is parsed at all (fix-j8yn).
- The async agent loop runs the finish check off the event loop, so its HTTP
  calls no longer block other coroutines (fix-13lt).
- `FinishChecker.note()` never raises and always closes its `fw.finish_check`
  span, also when the check fails after the span opens (fix-re11).
- A step output whose first sentence opens by saying it is empty ("No accounts
  found for Alan.", "Identity not found") still counts as empty, but one that
  goes on to a counted clause or picks from a counted set is a result: "0
  errors, 12 updated" and "None of the 5 are overdue". Subject
  names are found in an output as whole words, so "Al" is not found in "Alan"
  (fix-tu2g).
- Finish-check tests assert structure and key facts; one golden test pins the
  measured note wording (fix-mbcb).
- An over-long finish-check ledger is recognised from the error body's
  `max_tokens_exceeded` code, not from `str(error)`, so a refusal worded for
  humans still halves the ledger instead of failing the check (fix-oivq).
- The TypeSafe SDK's DEBUG logging no longer writes whole, unredacted request
  and response bodies: a filter on the `typesafe_sdk` logger drops those
  records whatever the log level and keeps its INFO status line (fix-qgeu).
- Finish-check subject names and kinds, `refers_to` keys and
  `names_in_output` now pass the same outbound filter as every other value
  sent; a credential used as a subject name or listing id never reaches the
  wire. The note shown to the agent keeps the raw names (fix-hu4f).
- A step the NLU stage stopped before any command ran (parameter extraction,
  an ambiguous or misunderstood command) is recorded at dispatch and shows in
  the ledger as outcome `error`, instead of reading as executed (fix-lnzw).
- A short-observation search no longer blocks for 30 s and raises when the
  related-handle lookup cannot read the archive: `list_summaries` waits at most
  0.5 s, any failure is caught, and the hint says other observations could not
  be listed (fix-kvq0).
- A refused listing parse is final: `parse_table` no longer retries from every
  later line, so a 104 KB listing refused at its end takes ~2 ms instead of
  2.5-4.6 s, and a short observation is not parsed at all (fix-zoup,
  fix-n4z9).
- An aligned listing no longer ends on an ordinary row -- a wrapped label, a
  row with separators collapsed to one space, or empty trailing cells -- while
  claiming to be complete (fix-36gv).
- The first block of a listing is no longer presented as the whole
  observation: a second group or table after it, a `total`/`sum`/`count` last
  row, or text saying there is more (`remaining=`, `has_more=true`, `N of M`,
  `page=2`, `next cursor`, `truncated`, …) refuses the parse (fix-k56c).
- Served rows never exceed the answer bound: `served_rows` picks the largest
  row count whose closing line fits and returns `None` when none fits, and the
  search goes to the model with `listing_skip_reason` `rows_do_not_fit`. A
  large preamble used to give a 5,410 B answer against a 3,000 B bound
  (fix-3zxk).
- **Cold-resume archive location.** An agent built while a context restores a
  suspended turn has no active workflow yet, so it opened the offload archive
  from an empty workflow path: a database named after the working directory's
  basename (unless `FASTWORKFLOW_WORKFLOW_ID` was set). This hit the default
  path. Evidence archived before the suspension was unreachable by
  `search_memory` and answer rehydration, and everything archived after the
  resume went to the stray database for the context's life, where channel
  erasure never reached it. `build_tool_agent` now reads the session's bound
  app workflow first, then the active one; the router's examples and the
  finish check's effect lookup use the same path. Warm turns open the same
  database as before.
- The `ask_user` replan after a cross-process resume plans from the request
  and trajectory again (fix-ksdu): it ran before `resume()` restored them and
  got `{}` for both. `ContinuationReAct.planner_view()` falls back to a copy of
  the suspended stash's `input_args` and `trajectory` (offload labels, no
  `action_N` keys); a same-process replan prompt is byte-identical. Session
  state is unchanged.
- DSPy's JSON-adapter retry after an unparseable chat-format reply keeps the
  available-commands list (fix-6jzi), in the agent loop, intent clarification
  and the plain-text planner; call counts are unchanged. It relies on DSPy's
  **private** `ChatAdapter._make_json_adapter_fallback()` hook (checked on
  DSPy 3.3.0; the pin still allows `^3.0.1`), overridden to return the new
  `CommandsSystemPreludeJSONAdapter`; a guard in
  `tests/test_chat_adapter_commands.py` fails if DSPy renames it. On an
  installed DSPy without the hook, importing `fastworkflow.utils.chat_adapter`
  logs one warning that the JSON retry will drop the available commands list;
  behaviour is otherwise unchanged.
- The offload archive no longer reads across channels (fix-tyzj): `get`,
  `list`, `get_subject`, `capture_record`, `forget_subject` and the
  stored-digest check filter on `channel_id` as well as the turn key, and
  `put_subject` updates only a row of the same channel. A scope pairing another
  channel with this turn's key reads nothing, cannot replace or forget this
  channel's subjects, and still collides on persist. The collision's
  `PersistenceError` now reads "runtime handle alias is already stored for this
  turn (different text or another channel)"; it said "collides with different
  text", which was wrong for identical text from another channel.
- A labelled last row -- first cell ending with `:`, or one word then `:` and a
  space (`Note:  2 items`) -- is a footer, not a row (fix-4riz): the strict
  parse refuses the listing, as for a `total`/`sum`/`count` row, and the lax
  read drops it. Over 3,832 recorded ido texts, no accept/refuse decision
  changed in either mode.
- Test isolation (fix-4fw3): the SDK timeout contract asserts the private
  `_config.timeout` the pinned 0.7.2 SDK actually has, and tests that register a
  litellm model restore litellm's model tables and caches afterwards
  (`restore_litellm_model_cost` fixture).

### Migration

- **Structured planning is disabled** (2026-09-28). Its code is commented
  out, not deleted, and there is no setting or flag to turn it back on. With
  `FW_FINISH_CHECK` on, the finish check now gets text plans only: those have
  no subjects, so the per-subject questions are never asked of a live plan,
  only per-step ones. Planning with the check on is no slower than without it
  (a single plain-text planner call). A reader of `fw.planner.plan` spans sees
  `plan_source` `text` / `none` and `subjects` `[]`; records written before
  this change may still hold `structured` or `text_fallback`.
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
  point `LLM_OBSERVATION_SEARCH` at the intended model. (Since fix-deus, set
  `FW_SEARCH_OBSERVATION_MAX_BYTES` to move the bound itself; a
  `FW_MODEL_CONTEXT_TOKENS` larger than the search model's own window no longer
  raises it.)
- The roster nudge is gone: without `FW_FINISH_CHECK=jev` and `JEV_API_KEY` a
  `finish` is never interrupted. `FW_EVAL_FINISH_REMINDERS=0` still turns the
  note off when the check is on (and, since fix-rm98, the structured planner
  too). `fastworkflow.answer_coverage` no longer
  exists, and a step span marks a note with `finish_check_note` where it used to
  say `roster_nudge` (`fw.agent.step` v3; span contract 7).
- The aggregate span contract is 8 (was 7): `fw.planner.plan` and
  `fw.planner.replan` are v2 and gain `plan_source` and `subjects`; with the
  finish check on, `plan` is the structured plan rendered as a numbered list
  with "(optional)" / "(needs the user)" flags. Spans recorded before read with
  both keys absent. A reader that pins the aggregate number must accept 8.
- `finish_check` events recorded before this change carry step text and
  subject names in `flagged_steps`; new ones carry indices. Events are now also
  recorded for skipped finishes, so count checks by `reason`, not by event.
- Search events gain the path fields above; `listing_shape` is no longer only
  on `rows_served` events.
- Install the Jev features with `fastworkflow[jev]`. `fastworkflow[router]`
  still works as a deprecated alias.
- `JEV_API_KEY` keeps its name; it does not follow `LITELLM_API_KEY_<ROLE>`
  because it is the TypeSafe SDK's own key, not a litellm role's.
- Turning on `FW_SEARCH_ROUTER` or `FW_FINISH_CHECK` sends turn content to
  TypeSafe with only credential patterns scrubbed by default (since fix-7tz5,
  neither turns on at all under a withholding capture profile or with
  redaction off), and the tokens
  those calls use are not in `fw.llm.call` cost totals. The finish check's
  8-second budget is best-effort: it is checked before each call, so one call
  can run up to its 4-second timeout past it. (No longer true since fix-i94q:
  every bound is a hard wall-clock cutoff; see the migration notes below.)
- **Session state gains two additive keys**, `turn_plan` (the plan as JSON, or
  `null`) and `turn_plan_status` (`planned`, `planner_empty`,
  `plan_unreadable`, `lost_on_resume`, `not_planned`); there is no
  `SCHEMA_VERSION` bump. A state written before this change resumes with the
  plan lost (`no_plan_cause` `lost_on_resume`) rather than failing. With the
  finish check off every blob carries `turn_plan: null` and
  `turn_plan_status: "not_planned"`. A react suspension may carry an optional
  `dispatch_outcomes` key.
- **`typesafe-sdk` is pinned to exactly 0.7.2.** `pip install
  'fastworkflow[jev]'` no longer resolves a later 0.7.x; an environment that
  already has another version installed is downgraded or conflicts. A bump
  must rerun `tests/test_jev_client.py::test_the_sdk_surface_fastworkflow_uses`
  and the golden-body tests in `tests/test_decision_provider.py`.
- **`TYPESAFE_BASE_URL` is ignored.** A deployment that pointed the SDK at
  another endpoint with it must set `FW_JEV_BASE_URL` instead (https, or http
  only to loopback); until then the features call the default endpoint and log
  one warning naming the variable.
- **Event shapes gain fields; none is removed or renamed.** `finish_check`
  events: `provider`, `calibration` and `user_replies` on every event;
  `no_plan_cause` on "no plan"; `vendor_ms`, `subjects_capped`, `ledger_bytes`
  and `ledger_rows_trimmed` on a checked finish; `error_status`,
  `error_request_id`, `error_code` and `error_stage` on "error"; new reasons
  `policy_withheld`, `subjects capped` and `ledger incomplete` (a capped check
  reports `subjects capped` where it used to report `every step executed` or
  `unexecuted steps`; read `fired` for the outcome). `search_memory` events:
  the `router` record gains `vendor_ms`, `provider` and the four `error_*`
  fields, with new error values `router_budget`, `policy_withheld`,
  `CallTimedOut`, `OutOfTime` and `VendorBusy`; `listing_skip_reason` gains
  `router_budget` and `rows_do_not_fit`; `short_verbatim` gains `own_score`,
  `related_lookup_failed`, `related_scope_refused` and `related_lookup_error`,
  and now always has `listing_parsed: false`. Span attributes and the span
  contract are unchanged.
- **Later on 2026-09-28**: `finish_check` events and router records gain
  `vendor_calls_in_flight` (vendor worker slots held, only when above 0);
  `listing_skip_reason` gains `policy_withheld` (until then counted as
  `router_error`), and a withheld route no longer uses one of the turn's 3
  routing calls. Every vendor call now passes the SDK a timeout of its cutoff
  plus 0.5 s (the client's own if shorter), and an abandoned call gives its
  worker slot back at most 0.5 s after its cutoff even if a trickled body keeps
  it running (`jev_client.calls_orphaned`); while 8 such requests still run
  (`jev_client.ORPHANED_CALLS_MAX`) every new call fails at once as
  `VendorBusy` and sends nothing, so a trickling endpoint cannot pile up
  threads and sockets, and `finish_check` events and router records carry
  `vendor_calls_orphaned` (only when above 0). The finish check's ledger reads
  labels from a page or one group of a longer listing
  (`parse_table(text, require_complete=False)`; rows are still served only from
  a complete one), its archive reads wait at most 0.5 s for a locked database
  and a failed read now ends the check as `error` at stage `ledger` (until
  then the inline text was used). Its context-clause reads share the process
  caches every other clause read uses, the "no subject" answer included, so a
  second check on a turn reads no alias from the archive again. With `FW_EVAL_FINISH_REMINDERS=0` a check
  that is attached no longer seeds a resumed trajectory, records dispatch
  outcomes or restores the suspended plan. The async entry (`aforward`) now
  resets the vendor budget and dispatch outcomes per turn, as `forward` does.
  `calibration` stays on events only: it is not a `fw.finish_check` span
  attribute, and the span contract is unchanged.
- **`FW_EVAL_FINISH_REMINDERS` is read from the workflow's `fastworkflow.env`
  first**, then the process, and `0` now also turns off the structured planner:
  a control arm run with it gets the plain-text planner and no plan markers. A
  value in the env file other than `0` now fails the agent build, as one in the
  process did; an empty value is unset.
- **The search evidence bound halves on the example model.** The example
  configuration's `mistral/mistral-small-latest` reports a 262,144-token
  window, which gave a 262,144-byte bound; the 131,072-byte ceiling now applies,
  so observations of 128-256 KB are searched as a disclosed prefix. Set
  `FW_SEARCH_OBSERVATION_MAX_BYTES=262144` to keep the old bound. No recorded
  ido observation is over 5.4 KB.
- **`listing.served_rows` returns `Optional[tuple]`**: `None` when not one row
  fits the bound. A caller outside the framework must handle it.
  `parse_table` keeps its return shape and returns `None` more often.
- **Vendor-call limits are module constants**, not settings
  (`jev_client.VENDOR_WORKERS`, `TURN_VENDOR_SECONDS`,
  `ROUTER_CALLS_PER_TURN`; `finish_check.SUBJECTS_MAX`, `LEDGER_MAX_BYTES`,
  `LISTING_PARSE_MAX_BYTES`). A turn resumed in another process starts a fresh
  vendor budget.
- **Agent-visible wording changed on the default path without re-measurement**
  (fix-94m9): the offload-label mark, the `search_memory` description, the
  `WorkflowAgentSignature` docstring and the rehydration note prefix (texts
  under Changed). A deployment that compared agent runs against a baseline
  taken before this change is comparing different prompts. A reader that
  matched the exact old mark "Restored in full for the final answer." must
  accept "Normally restored for the final answer."; `labels.LABEL_RE` parses
  both.
- **fix-6jzi depends on a private DSPy hook**
  (`ChatAdapter._make_json_adapter_fallback`). A DSPy upgrade must rerun
  `tests/test_chat_adapter_commands.py`; if the hook is gone the retry silently
  loses the command list again.
- **Changing a finish-check or router question now fails
  `tests/test_decision_measurements.py`** (fix-8ee5). Update the digests in
  `tests/fixtures/decision_measurements.json` only together with a new
  measurement recorded there. **The measurement evidence lives only in local
  directories** -- `~/rl/4dsr-offline` (with the manifest-gate replay scripts
  and the manifest generator in `~/rl/4dsr-offline/replay/`) and
  `~/rl/ufot-offline` -- and must be archived somewhere durable; the manifest
  holds only their sha256 and those paths.
- `fastworkflow.observation_offloading.related` is a new module (fix-vpe3);
  every moved name is still importable from `observation_offloading.search`.

### Known limits

Offloading and search ship with documented limits rather than silent ones: what
a turn resumed in another process reads, which part of the evidence is stored
unredacted, when pruning runs, the worst-case agent work one turn can cost, and several places where a disclosure
line can push a bounded input a few hundred bytes past the budget it reports.
They are listed in
[Retention, redaction and known limits](docs/observation_search.md#retention-redaction-and-known-limits);
read that section before sizing a deployment or writing a retention policy.

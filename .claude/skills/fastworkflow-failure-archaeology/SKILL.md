---
name: fastworkflow-failure-archaeology
description: >
  Load this when you are about to investigate, revert, re-attempt, or "clean up" something in
  fastWorkflow and suspect a battle may already have been fought: symptoms like "this bug looks
  old", "why does this dead code / legacy file exist", "should we just delete X", "the revert
  didn't revert", "bd close didn't stick", "payload is None in agent mode", "why was feature Y
  removed", or "did we already try this approach". Also load before proposing any feature
  that resembles: command dependency graphs, a standalone FastAPI service, probabilistic
  response generation, rich CLI, ensemble intent voting, or RLM-style memory layers — all
  have prior art documented here.
  Do NOT load for live triage of a current failure (use fastworkflow-debugging-playbook), for
  the rules governing changes (fastworkflow-change-control), or for how the architecture is
  supposed to work (fastworkflow-architecture-contract).
---

# fastWorkflow Failure Archaeology

> TEAM-PRIVATE: embeds content from uncommitted internal docs. Do not commit or publish this skill without the developer'sexplicit approval.

The chronicle of every major investigation, dead end, revert, and abandoned feature in this
repo — so no one re-fights settled battles. Each entry: **symptom → root cause → evidence →
status**. Every commit hash, path, and line number below was verified against the working tree
at v2.22.2 (commit c33b9a5) on 2026-07-09.

**Prime directive of this skill:** before you delete "dead code", revive an "obvious" feature,
or diagnose a "new" bug, check the index below. A spec doc in `docs/` is NOT proof a feature
exists; an open beads issue is NOT proof work remains (tracker drift is real, see entry 11);
and a commit titled "Revert" is NOT proof anything was reverted (entry 4).

## When to use / when NOT to use

| Situation | Use |
|---|---|
| "Has this been tried/broken/fixed before?" — history lookup | **This skill** |
| A test/runtime failure is happening NOW and you need triage steps | `fastworkflow-debugging-playbook` |
| You want the non-negotiable rules and the incidents behind them, to gate a change | `fastworkflow-change-control` |
| You need the intended design and its invariants | `fastworkflow-architecture-contract` |
| tau-bench / tau2 mechanics, pass@1 vs pass^k, our benchmark claims | `fastworkflow-taubench-reference` |
| Running the E0–E25 reliability campaign | `tau2-reliability-campaign` |
| Statistics for judging whether a "fix" is real (variance, pass^k math) | `fastworkflow-proof-and-analysis-toolkit` |
| Open research problems built on these failures | `fastworkflow-research-frontier` |

## Index

| # | Entry | Status |
|---|---|---|
| 1 | The payload-loss bug (xray `/invoke` payload=None) | settled (v2.21/v2.21.1) |
| 2 | The TurnResult saga: what shipped (v2.21 X6 slice) vs v3.0 roadmap | partly shipped; v3.0 open |
| 3 | fix-85g: the 504-retry double-execution race (v2.22.0 turns engine) | Step 1 settled; Steps 2–3 open |
| 4 | The command_router.py revert trio (the revert that didn't) | moot since v2.6.0 |
| 5 | fix-0hb: the test suite that ate its own trained model | settled (fa97b48) |
| 6 | The RocksDB/speedict removal arc (v2.21.3–v2.21.6) | hot path settled; fix-7kp open |
| 7 | The fingerprint cache-invalidation bug (v2.22.1) | settled (b5747df) |
| 8 | The beads flakiness incident (2026-06-11) | workaround discipline, permanent |
| 9 | The private-doc auto-push incident (2026-07-08) | rule established, permanent |
| 10 | The RLM/tau2 failure taxonomy (Articles 1–3): the variance lesson | open — drives the tau2 campaign |
| 11 | Tracker drift: open beads issues describing shipped work | live caveat |
| 12 | The graveyard: abandoned features and dead code | reference table |
| 13 | Doc-rot register | live caveat |

Jargon used throughout, defined once:
**bd / beads** = the git-backed issue tracker (`.beads/issues.jsonl` is source of truth).
**WEC** = `WorkflowExecutionContext`, the transport-free execution core (Topology B).
**CME** = the internal `command_metadata_extraction` workflow that runs intent detection +
parameter extraction. **DSPy ReAct** = the LLM agent loop whose tools return *text only*.
**RLM** = Recursive/Reasoning Language Model — an LLM-written-code memory-retrieval layer used
in the team's tau2 experiments. **tau2 (τ²-bench)** = the retail/airline conversational-agent
benchmark; pass^k = task passes k independent runs in a row.

---

## 1. The payload-loss bug (xray `/invoke` payload=None)

- **Symptom:** xray's `POST /invoke` returned `payload=None`, `payload_hint="text"`,
  `request_id=-1` for every agent-mode data question, even when a table/chart was produced.
  Deterministic `/`-prefixed commands returned the payload fine — that divergence was the tell
  (`docs/turn_result_design.md` §0–§1.3).
- **Root cause:** NOT the mapping layer. Two framework drop points:
  1. **The DSPy ReAct tool boundary is text-only.** `_execute_workflow_query`
     (`fastworkflow/workflow_agent.py`) gets the full `CommandOutput` back from
     `CommandExecutor.invoke_command` but joins only the `.response` strings and discards
     `artifacts` — by design, because ReAct tools return string observations.
  2. **`_finalize_agent_output` synthesized a fresh answer-only `CommandOutput`** from the
     agent's `final_answer`, so every tool call's artifacts vanished from the user-facing path.
- **Evidence:** `docs/turn_result_design.md` lines ~18–129 (origin narrative, code excerpts,
  "confirming signature"); fix commits afcbe01 (v2.21.0) and aa20c1b (v2.21.1).
- **Status: settled.** Fixed at the chokepoint both transports flow through:
  `_finalize_agent_output` now merges every artifact-bearing tool response into the single
  answer `CommandResponse.artifacts` via `merge_artifact_responses_into`
  (`fastworkflow/workflow_execution_context.py`, call at ~line 789; helper in
  `fastworkflow/turn.py`). Key-collision rule: incoming key unchanged when absent, suffixed
  `_<n>` on collision. Wire shape kept identical to the v2.20 baseline (one-element
  `command_responses`).
- **Near-miss worth remembering:** v2.21 as first staged only *captured* turns into a dormant
  `TurnResult` and left `_finalize_agent_output` unchanged — the bug would have shipped
  "fixed" while still user-visible. Caught in a post-implementation teaching session
  (2026-06-12, `docs/turn_result_design_feedback.md` Topic 5: "Don't merge v2.21 believing the
  payload bug is fixed"). Lesson: re-verify the shipped code against the symptom, not the spec.

## 2. The TurnResult saga: what shipped vs the v3.0 roadmap

The payload bug triggered the largest design effort in repo history. Full detail:
[references/turn-result-saga.md](references/turn-result-saga.md). The short version:

- **Process:** design doc (`docs/turn_result_design.md`, with a decision log and amendments
  A1–A47) → adversarial 48-finding review (`docs/turn_result_design_review.md`, R1–R48; beads
  epic fix-vof with one child per finding, all 48 closed; one docs commit per resolution, from
  42e9b3a "resolve R1" to 920d0de "epic fix-vof complete (48/48)") → ten-concern architecture
  review (9031942, findings X1–X12) → authoritative spec `docs/turn_result_design_final.md`
  ("where this conflicts with any of those, this document wins").
- **The X6 slice — what v2.21 actually shipped:** the architecture review's X6 shrank the plan
  from ~2.5k lines / ~120 files (96 command-file migrations) to ~500 lines / ~6 files
  (`docs/turn_result_architecture_review.md` lines 85–95). Shipped in afcbe01 + aa20c1b +
  5a63790: `fastworkflow/turn.py` (TurnResult/TurnOutput/TurnStatus, `mint_turn_key`, artifact
  merge helpers), the WEC turn accumulator, `process_turn()` returning the slim public
  `TurnOutput` (`workflow_execution_context.py:484`), `process_message` deprecated, artifact
  merge at finalize. `TurnOutput.success` is computed = `all(co.success)` — because v2.20
  hard-coded the synthesized agent answer to `success=True`, masking every tool failure.
- **What is NOT implemented (v3.0 roadmap, `turn_result_design_final.md` §14–15):** no
  `stores/` package, no ConversationTurnStore/ArtifactBlobStore, no Redis keyspace/ZSET
  indexes, no MetricsSink, no `fastworkflow admin` CLI, no retention tooling;
  `run_fastapi_mcp/conversation_store.py` still exists (slated for deletion at v3.0);
  `TurnResult.conversation_id`/`ordinal` are Optional and left None. The
  `command_responses → command_response` collapse is deferred to a v3.0 big-bang. Do not
  assume any of these exist because the spec describes them.
- **Collateral found by the review, still open:** bug **fix-5fv** — CLI hangs on agent-mode
  `ask_user` because the clarification `CommandOutput` is enqueued without a trace sentinel
  (predicted by code-reading during R43; marked "NEEDS RUNTIME VERIFICATION"; `bd show fix-5fv`).

## 3. fix-85g: the 504-retry double-execution race (v2.22.0)

- **Symptom:** retrying `/initialize` after a 504 returned tokens with a silently-empty
  `startup_output`; occasionally two executions raced on the same context.
- **Root cause:** long LLM work (~10 min observed) ran *inside* the HTTP request.
  `asyncio.wait_for` timeout raised HTTPException(504), which unwound `async with
  runtime.lock`, releasing the lock while the orphaned executor thread kept mutating the WEC —
  so a retry passed the `lock.locked()` 409 guard and started a second racing execution.
- **Evidence:** commit cf3eeae; `bd show fix-85g`; design doc
  `docs/fastworkflow_turns_async_execution_design.md`; invariants pinned in the module
  docstring of `fastworkflow/run_fastapi_mcp/turns.py`.
- **Status:** Step 1 settled (TurnRegistry, wait-or-defer never wait-or-abort, per-channel
  active-execution pointer — never `lock.locked()` — as the single liveness/idempotency
  source, persist-before-DONE, idempotency-key rejoin). Steps 2 (GET /turns polling, 429
  backpressure, TTL eviction, trace replay) and 3 (durable distributed turn store) are OPEN
  (fix-85g.9–.13). Note `/invoke_agent_stream` still contains a `runtime.lock.locked()` check
  (`run_fastapi_mcp/__main__.py`, grep for it) — a survivor of the old pattern.

## 4. The command_router.py revert trio (the revert that didn't)

- **Symptom:** debug cruft on an external contributor's parameter-extraction branch (Apr
  2025): a hardcoded RocksDB cache path `./examples/sample_workflow/___convo_info/-694349230.db`,
  speedict `Rdict` helper functions, a commented-out `not_what_i_meant` retry block, and a
  broad wildcard-fallback try/except.
- **Root cause:** the contributor (Sanchit Satija, all three commits 2025-04-08) ping-ponged
  while trying to strip it: d2e3387 "Revert command_router.py changes" removed 58 lines;
  aa751e7 "Revert changes" added 1 blank line; 1386f88 "revert changes" re-added 31 lines of
  the same cruft. Net effect: the cruft shipped to main in the PR #10 merge.
- **Evidence:** `git show f9e2227:fastworkflow/command_router.py` still contains
  `from speedict import Rdict`, the dead helpers, and the commented `-694349230.db` block
  (verified 2026-07-09).
- **Status: moot.** `command_router.py` was dissolved in the v2.6.0 consolidation (a6d0f43,
  "Now we have just 2 files - command directory and command routing"). The cruft survived on
  main for roughly a year until then. Lessons: (a) verify a revert by diffing the file, not by
  reading commit titles; (b) the deeper speedict addiction this cruft hinted at took until
  v2.21.3/v2.21.4 to excise from the hot path (entry 6, where the still-open fix-7kp awaits
  owner adjudication).

## 5. fix-0hb: the test suite that ate its own trained model

- **Symptom:** 4 tests in `tests/test_fastapi_service.py` failed with FileNotFoundError on
  `fastworkflow/examples/hello_world/___command_info/global/threshold.json` — but only on the
  run AFTER a green run (v2.21.5 run: 4 passed; v2.21.6 run: 4 failed).
- **Root cause:** the module-scoped `trained_hello_world` fixture in
  `tests/test_train_modern_stack.py` trained against the REAL bundled example and rmtree'd its
  `___command_info` at setup AND teardown, destroying the pre-trained model the FastAPI tests
  depend on. The fixture only activated when `datasets` + real API keys were present, so it
  was invisible in keyless CI — a local-only, every-other-run poison.
- **Evidence:** commit fa97b48; `bd show fix-0hb` (full forensic narrative, "ROOT CAUSE
  CONFIRMED"); the fix at `tests/test_train_modern_stack.py` (copytree into
  `tmp_path_factory.mktemp(...)`, ignoring `___command_info`/`___workflow_contexts`/
  `___convo_info`/`__pycache__`).
- **Status: settled.** Suite verified 478 passed / 0 failed and idempotent. **Standing rule
  (cite fix-0hb / fa97b48):** never let tests or experiments train into or wipe
  `fastworkflow/examples/*/___command_info` — always train into a temp copy. Bonus lore
  recorded in the issue: the BUNDLED `fastworkflow/examples/fastworkflow.passwords.env` has a
  dead placeholder Mistral key (litellm AuthenticationError); the repo-local
  `./passwords/.env` has valid keys; local recipe:
  `fastworkflow train ./fastworkflow/examples/hello_world ./env/.env ./passwords/.env`.

## 6. The RocksDB/speedict removal arc (v2.21.3–v2.21.6)

Three fixes in rapid succession, two of them on the same day (2026-06-15):

- **v2.21.3 (79e6986) — env read at import time.** `run_fastapi_mcp` crashed at startup when
  `SPEEDDICT_FOLDERNAME` lived only in `--env_file`: a module-level
  `session_manager = ChannelSessionManager()` constructed the SessionStateStore in `__init__`
  at import, BEFORE `fastworkflow.init()` loaded the env file. The bug was *masked in-repo*
  because litellm calls `load_dotenv` on the cwd `.env`. Fix: lazy store construction.
  Standing rule: never read env vars at module import time.
- **v2.21.4 (6880b0a) — hot-path removal.** `Workflow.create` paid ~6 RocksDB `Rdict`
  open/close cycles per new session (~90–270 ms of pure churn) for trivial payloads. Replaced
  with a process-global dict + `threading.RLock`. **Accepted regression, stated in the commit
  body:** "the CLI no longer resumes workflow context across process restarts." Durable state
  is now exclusively SessionStateStore + ConversationStore (FastAPI only).
- **v2.21.5 (033132f, fix-04r) — the leak the fix introduced, ~90 minutes later.** Embedders
  never call `close()`, and a unique channel_id per request accumulated entries unboundedly.
  Fix: the registry became a `weakref.WeakValueDictionary` (`fastworkflow/workflow.py`,
  comment block at lines 14–40 tells the whole story) — GC auto-evicts abandoned sessions;
  holding a strong reference IS the session.
- **v2.21.6 (3fabdb0):** security dependency updates rounding out the patch train.
- **Status:** hot path settled. speedict (`from speedict import Rdict`) is still used,
  intentionally, at: `cache_matching.py`, the `enablecache` decorator in `workflow.py`, the
  NLU clarification cache (`_workflows/command_metadata_extraction/intent_detection.py`), and
  `run_fastapi_mcp/conversation_store.py`; `session_state_store.py` stores plain JSON but
  still keys its disk folder off `SPEEDDICT_FOLDERNAME`. Open-issue note — verified facts
  only: **fix-7kp** (created 2026-06-16 00:32:40Z, OPEN) and **fix-2jo** (00:32:49Z, nine
  seconds later, closed when v2.21.4 shipped) have byte-identical titles ("Remove
  speedict/RocksDB from the Workflow hot path (Blueprint/Session-State split)") and
  near-identical descriptions — both say "Ship as v2.21.4" and "speedict KEPT elsewhere".
  Whether fix-7kp is a duplicate from the flaky-bd era (entry 8) or an intentional extension
  is UNADJUDICATED — that call is reserved for the owner (see the stale-epic table in
  `fastworkflow-change-control`): Ask Dhar; do not close fix-7kp yourself.

## 7. The fingerprint cache-invalidation bug (v2.22.1)

- **Symptom:** after adding/removing/renaming a command, runtime import errors pointed at
  command sources that no longer existed, until the user manually deleted `___command_info`
  JSON artifacts.
- **Root cause:** `routing_definition.json` had NO staleness check at all, and
  `command_directory.json` used a newest-mtime check — which is blind to deletions and
  renames (nothing bumps a surviving file's mtime when another file disappears).
- **Evidence:** commit b5747df; `compute_commands_source_fingerprint()` at
  `fastworkflow/command_directory.py:627` (sha256 over the SET of `(path, size, mtime_ns)` of
  `_commands/**/*.py` incl. base workflows + context model JSONs), stamped into both
  artifacts; `fastworkflow/command_routing.py` rebuild-on-mismatch; orphaned
  `*_param_labeled.json` pruned only AFTER a successful train so a failed retrain leaves the
  prior artifacts runnable (`fastworkflow/train/__main__.py`).
- **Status: settled.** Doctrine: invalidate caches by content fingerprint of the full source
  set, never by newest-mtime. Pre-2.22.1 workflows may carry unstamped artifacts (missing
  fingerprint forces rebuild).

## 8. The beads flakiness incident (2026-06-11)

- **Symptom:** `bd close --reason=...` reported success but silently failed to persist to
  `.beads/issues.jsonl` (bare `bd close` worked). Separately, the Dolt server wedged into a
  state where every bd command re-imported issues.jsonl into an empty database and async
  exports raced/clobbered each other.
- **Evidence:** bd memory `beads-flakiness-observed-2026-06-11-in-the` (read via
  `bd memories --json` — note `bd memories show <name>` is not a valid invocation).
- **Status: partly superseded 2026-08-13.** (1) One bd write at a time; (2) verify
  `.beads/issues.jsonl` after each write — both still permanent; (3) the `--reason`
  workaround is retired: it persisted 12/12 on bd 1.1.2 when retested 2026-08-13 (the
  symptom above was on an older bd and may return if Dolt wedges again); (4) when the Dolt server wedges, patching issues.jsonl directly is safe because
  every command re-imports it as source of truth. Possible collateral: the fix-2jo/fix-7kp
  identical-title pair (entry 6 — duplicate vs intentional extension unadjudicated).

## 9. The private-doc auto-push incident (2026-07-08)

- **Symptom:** a private planning document was auto-committed and pushed to the PUBLIC
  fastworkflow repo by an agent following a "session-close protocol". Cleanup required a git
  history rewrite.
- **Evidence:** bd memory `never-git-commit-or-push-anything-without-dhar` (established
  2026-07-08).
- **Status: permanent rule.** NEVER `git commit` or `git push` without the developer'sexplicit request
  *in that turn*; no session-close protocol, skill, or workflow overrides this. Documents,
  plans, and analyses may be team-private even when written into the repo tree — the repo is
  public. This skill library itself is covered: it stays uncommitted until Dhar decides.

## 10. The RLM/tau2 failure taxonomy: the variance lesson

Source: three team-private PDFs in `docs/` ("Article 1 - The Setup", "Article 2 - The Failure
Taxonomy", "Article 3 - Mitigations and a Path Forward", Jul 2026, principally Rakshit Pandhi
with Sanchit Satija and Sharvil Oza; all roles gpt-oss-20b on Groq; raw traces at
https://github.com/Programiz-007/fastworkflow_RLM_Traces/). System: an RLM memory layer for
τ²-bench retail — tool outputs over a token threshold τ are archived; retrieval is LLM-written
Python ending in `SUBMIT()`.

- **Headline numbers, on 15 hand-selected (enriched, non-random) retail tasks:** pre-memory
  ~0/15 (asserted, never measured); initial memory-layer evaluation 2/15; after four bundled
  mitigations 5/15. **The variance lesson (Article 3, "Reproducibility"):** the two tasks that
  passed initially "were not among the five tasks that passed after the mitigations" — the
  pass sets are DISJOINT. With n=15, single runs, and unfixed temperature, 2→5 is
  statistically indistinguishable from noise. This is the founding argument for treating
  pass^k reliability engineering as its own discipline and for E0 (harness + k=5 paired runs
  + McNemar) being a hard prerequisite in the tau2 plan.
- **The three failure pillars (Article 2):**
  1. **Intrinsic RLM failures** — printed the answer instead of calling `SUBMIT`, hitting the
     20-iteration cap; repeated "not found" loops when the data was demonstrably present
     (task with order #W1994898: the RLM hallucinated a 3-key index format and looped).
  2. **Cross-layer failures** — planner set `new_item_id` equal to the original item
     (9494281769→9494281769, a self-exchange the benchmark rejects); incomplete execution
     reported as success ("illusion of success"); goal substitution (full cancellation
     instead of `modify_pending_order_items` — the one bucket that got WORSE after
     mitigations).
  3. **Schema-awareness issues** — queried a non-existent `product_type_id` instead of
     `product_id`.
  Headline quote: moving retrieval into generated code "doesn't remove errors, it relocates
  them" — into silently-wrong code that runs cleanly and SUBMITs a wrong answer.
- **Known defects in the study itself** (per the critical review folded into
  `docs/tau2_retail_reliability_implementation_plan.md` §2.4): attribution was single-rater,
  unblinded, forced single-label, and the counts don't sum (14 attributions vs 13 failures);
  the four mitigations were bundled with no ablation; mitigation 4 (full tool-output view for
  the planner) silently reintroduces the context bloat the memory layer exists to solve; the
  self-exchange gold label is itself contestable. Twist verified against main: the current
  retail example's exchange command already rejects self-exchanges (old∩new overlap check in
  `fastworkflow/examples/retail_workflow/_commands/exchange_delivered_order_items.py`,
  ~lines 98–104) — implying the experiment fork bypassed `validate_extracted_parameters` or
  ran stale wrappers. That plumbing audit is experiment card E1.
- **Status: open.** The whole tau2 retail reliability program (experiment cards E0–E25,
  "Option N v2 / Governed Determinism") exists to turn this into measured science. See
  `tau2-reliability-campaign` for execution and `fastworkflow-taubench-reference` for
  benchmark mechanics. Also note the citation-hygiene sub-incident: Article 1 cites the
  fastWorkflow paper DOI as 10.1145/3786335.3813158 (matching README.md:101) while Articles 2
  and 3 cite 10.1145/3738331.3738402 — one is wrong (unresolved; verify externally before
  citing either).

## 11. Tracker drift: open beads issues describing shipped work

`process_turn()`/`TurnOutput` are live on main (`workflow_execution_context.py:484`,
`turn.py`), yet epics **fix-yy1** (v2.21 TurnResult capture) and **fix-qtq** (migrate the
run_fastapi_mcp transport edge to process_turn/TurnOutput) remain OPEN with open children.
This is partly deliberate ("NO COMMITS until user approval" epics whose approval/close loop
never ran) and partly real residue (`run_fastapi_mcp/utils.py` still has
`run_process_message*` helpers — the endpoint-level migration is genuinely incomplete).
**Rule: reconcile beads state against git history before trusting `bd ready` or `bd list
--status open` as a work queue.** Stale cohort as of 2026-07-09: fix-5ka, fix-cgs, fix-4od
(all created 2026-06-04, untouched 35 days).

## 12. The graveyard: abandoned features and dead code

Do not resurrect, "finish", or silently delete anything here without checking the detail file:
[references/graveyard.md](references/graveyard.md). Summary:

| Item | Built | Killed / state | Why (or "unknown") |
|---|---|---|---|
| Command dependency graph (GenAI backward-chaining engine) | v2.10.0 (55a8668), refine built its JSON in v2.13.0 (ce5ada5) | tests removed v2.15.8 (ed21d77); sentencetransformers removed v2.17.19 (ad68d9b) | never adopted by the agent; superseded by injecting command info into the agent's system prompt. Orphan remnants still on main: `fastworkflow/build/dependency_manager.py`, `fastworkflow/build/command_dependency_resolver.py` (zero external callers) |
| Standalone FastAPI service (`services/run_fastapi`, 1092-line main.py) | v2.16.0, 2025-10-11 (447db70) | replaced wholesale by `fastworkflow/run_fastapi_mcp` in v2.17.0, 2025-10-18 (299739e) | lived one week; v2.17 line absorbed 35 patch releases and became the product |
| Probabilistic response generation | branch `origin/cursor/implement-probabilistic-response-generation-and-pass-tests-7586` (7c7d1cf, a4d3d4d; forked from ddab25c, Aug 2025) | never merged; `fastworkflow/probabilistic_response.py` does not exist on main | why-abandoned unknown. Trap: the spec `docs/probabilistic_response_generation.md` IS on main — a spec doc is not proof the feature exists |
| Rich CLI (replace colorama with rich) | branch `cursor/enhance-fastworkflow-cli-aesthetics-1752` (4d2d9a7, c4cf167; forked from v2.5.2) | never merged; main still depends on colorama (pyproject.toml:49) | why-abandoned unknown |
| 43-agents cloud-agent experiment ("agent categorization system with 43 agents in 4 collections") | branches `origin/cursor/new-cloud-agent-1d34` / `-af26` (2a0214f etc., Feb 2026, forked from v2.17.33) | never merged; nothing on main | Cursor cloud-agent experiment; why-abandoned unknown |
| `majority_vote_predictions` (5-way parallel intent voting) | present at `fastworkflow/_workflows/command_metadata_extraction/intent_detection.py:304` | dead code: sole call site commented out (line 122) | its own TODOs (lines 302–303) admit generation is deterministic so all 5 votes return the same answer; needs temperature support first. Open question: revive or delete |
| `ambiguous_threshold.json` legacy files | pre-split threshold artifact | linger in trained folders (e.g. `fastworkflow/_workflows/command_metadata_extraction/___command_info/*/`); the loader reads only `tiny_ambiguous_threshold.json` / `large_ambiguous_threshold.json` (`model_pipeline_training.py:299-300`) | harmless orphans from a rename; safe to ignore, not proof of a live code path |

## 13. Doc-rot register

Documented contradictions where code wins over docs (evidence attached; do not "fix" docs here
without change control):

- **CLAUDE.md** says intent models are trained "via scikit-learn". Actually torch/transformers
  fine-tuning of a two-tier TinyBERT/DistilBERT pipeline; sklearn supplies only
  LabelEncoder/split/metrics (see `fastworkflow/model_pipeline_training.py` imports). CLAUDE.md
  also omits `tests/todo_list_workflow` from the test-workflow list.
- **README/Article DOI disagreement** — see entry 10.
- **A spec doc in `docs/` is not proof of implementation** — `probabilistic_response_generation.md`
  (entry 12) and the TurnResult v3.0 sections (entry 2) are the standing examples.
- **`run_fastapi_mcp/README.md`** still references the deleted `services.run_fastapi` layout
  (entry 12's one-week service).

---

## Provenance and maintenance

Facts verified 2026-07-09 against v2.22.2 (commit c33b9a5). Team-private sources embedded:
the three Article PDFs, `docs/tau2_retail_reliability_implementation_plan.md`,
`docs/rsi_harness_agent_report.md` (all untracked in `docs/`). Re-verification one-liners for
every volatile fact:

| Fact | Re-verify with |
|---|---|
| Commit hashes/subjects cited anywhere above | `git log --oneline --no-decorate -1 <hash>` |
| Revert-trio cruft shipped to main | `git show f9e2227:fastworkflow/command_router.py \| grep -n "speedict\|-694349230"` |
| command_router.py dissolution | `git log --follow --oneline -- fastworkflow/command_router.py \| head -3` |
| Payload fix wired at finalize | `grep -n merge_artifact_responses_into fastworkflow/workflow_execution_context.py` |
| process_turn line number | `grep -n "def process_turn" fastworkflow/workflow_execution_context.py` |
| v3.0 pieces still absent | `ls fastworkflow/stores fastworkflow/turn_serializer.py 2>&1; ls fastworkflow/run_fastapi_mcp/conversation_store.py` |
| fix-85g / fix-7kp / fix-5fv / fix-yy1 / fix-qtq open status | `bd show <id> --json` (read-only) |
| fix-vof child count (48) | `python3 -c "import json;print(sum(1 for l in open('.beads/issues.jsonl') if json.loads(l).get('id','').startswith('fix-vof.')))"` |
| fix-2jo/fix-7kp identical titles + 9 s timestamps | `grep -E '"id":"fix-(2jo\|7kp)"' .beads/issues.jsonl \| python3 -c "import sys,json;[print(json.loads(l)['id'],json.loads(l)['created_at'],json.loads(l)['title']) for l in sys.stdin]"` |
| bd memories (flakiness, push rule) | `bd memories --json` |
| fix-0hb temp-copy fix in place | `grep -n "copytree\|tmp_path_factory" tests/test_train_modern_stack.py` |
| speedict remaining sites | `grep -rln speedict fastworkflow/ --include=*.py` |
| Fingerprint function | `grep -n "def compute_commands_source_fingerprint" fastworkflow/command_directory.py` |
| workflow.py weakref registry + CLI-resume trade-off comment | `sed -n '14,45p' fastworkflow/workflow.py` |
| majority_vote dead code + TODOs | `grep -n majority_vote fastworkflow/_workflows/command_metadata_extraction/intent_detection.py` |
| ambiguous_threshold.json orphans | `find fastworkflow -name ambiguous_threshold.json` |
| Dep-graph orphan modules have no callers | `grep -rn "dependency_manager\|resolve_command_dependencies" fastworkflow/ --include=*.py \| grep -v "build/dependency_manager.py\|build/command_dependency_resolver.py"` |
| Abandoned branches unmerged | `git branch -r --merged main \| grep cursor` (expect empty) |
| Self-exchange overlap check on main | `sed -n '98,104p' fastworkflow/examples/retail_workflow/_commands/exchange_delivered_order_items.py` |
| Article facts (0→2→5, disjoint sets, pillars) | extract text: `PYTHONPATH=/tmp/pdflib .venv/bin/python -c "import pypdf;print(''.join(p.extract_text() for p in pypdf.PdfReader('docs/Article 3 - Mitigations and a Path Forward.pdf').pages))" \| grep -i "not among"` (install pypdf to a temp target first) |
| DOI disagreement | grep `10.1145` in extracted Article texts vs `grep -n 10.1145 README.md` |
| Sanity-run the whole table | `bash .claude/skills/fastworkflow-failure-archaeology/scripts/verify_archaeology.sh` |

Maintenance protocol: when a new investigation closes, append an entry here in the same
symptom → root cause → evidence → status shape, with verified hashes, and add its
re-verification one-liner to the table. Never rewrite an existing entry's history — append a
dated correction (the amendments-over-edits discipline from the TurnResult saga).

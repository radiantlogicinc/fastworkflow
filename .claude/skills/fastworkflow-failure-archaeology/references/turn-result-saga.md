# The TurnResult saga — full chronicle

> TEAM-PRIVATE: embeds content from uncommitted internal docs. Do not commit or publish this skill without the developer'sexplicit approval.

Companion to SKILL.md entries 1–3. Facts verified 2026-07-09 against v2.22.2 (commit c33b9a5).
This is the deepest design effort in repo history and the house model for how large changes are
made. Read it before touching `fastworkflow/turn.py`, `workflow_execution_context.py`,
`workflow_agent.py`, or `run_fastapi_mcp/turns.py`.

## Timeline at a glance

| Date (2026) | Event | Artifact |
|---|---|---|
| pre-06-10 | xray payload=None bug investigated; design doc written | `docs/turn_result_design.md` (1400+ lines: origin, type algebra, 22-entry decision log, amendments A1–A47) |
| 06-10/11 | 48-finding adversarial review; epic fix-vof, one child per finding, one docs commit per resolution | `docs/turn_result_design_review.md`; commits 42e9b3a (R1) … 920d0de (48/48) |
| 06-11 | Ten-concern architecture review of the finalized design (X1–X12) | commit 9031942; `docs/turn_result_architecture_review.md` |
| 06-11 | Authoritative consolidated spec cut; X6 minimal slice approved by Dhar | `docs/turn_result_design_final.md` ("where this conflicts... this document wins") |
| 06-12 | v2.21.0 implementation | afcbe01 |
| 06-12 | Post-implementation teaching session catches the near-miss (Topic 5) | `docs/turn_result_design_feedback.md` |
| 06-12/13 | v2.21.1 artifact-merge shape; v2.21.2 public TurnOutput + two latent bugs | aa20c1b, 5a63790 |
| 06-15 | v2.21.3/.4/.5 speedict arc (SKILL.md entry 6) | 79e6986, 6880b0a, 033132f |
| 06-23 | v2.22.0 turns-based async execution (fix-85g Step 1) | cf3eeae; `fastworkflow/run_fastapi_mcp/turns.py` |

## The origin bug in full (design.md §0–§1)

Symptom: xray `POST /invoke` always returned `payload=None`, `payload_hint="text"`,
`request_id=-1` in agent mode. The producer (`NLQueryTool.process_query_results`) and the
mapping layer were verified correct. Two framework drop points:

1. **ReAct text-only tool boundary.** `_execute_workflow_query` (workflow_agent.py) receives
   the full `CommandOutput` but returns only joined `.response` text to DSPy ReAct — string
   observations by design. Artifacts (`payload`, `payload_hint`, `request_id`, `actions`)
   never enter the agent's reasoning or output.
2. **Synthesized final output.** `_finalize_agent_output` built a brand-new `CommandOutput`
   from the agent's `final_answer` with only a `conversation_summary` artifact.

Diagnostic tell: the deterministic `/`-prefixed assistant path returned the command's actual
`CommandOutput` verbatim — payload survived there. When agent mode and deterministic mode
disagree on artifact presence, suspect the agent finalize path first.

Fix option chosen: option 3, "patch the framework" (vs side-channel or agent-bypass), because
the team owns fastWorkflow (design.md §"Why this is architectural, not a mapping bug").

## The review process (the house change-control model)

1. Design doc with numbered decision log; **amendments-over-edits**: ratified sections are
   never rewritten, tagged amendments (A1–A47) supersede them and the original stays as
   rationale archive.
2. Adversarial review against the ACTUAL working tree: 48 findings R1–R48, severity-indexed.
   The original doc's §13 "Open questions: none remaining" was proven wrong (four internal
   contradictions) and publicly rewritten as an admission. Lesson institutionalized: a design
   is not done until adversarially reviewed against the codebase.
3. Beads epic fix-vof, 48 children (fix-vof.1–.48), review-only, one finding at a time with
   Dhar; every resolution is one docs commit shaped
   `docs: resolve R# — <decision>` with body listing amendment tag, review-doc mark, beads
   closure, epic counter. ELI5-before-decision-questions is the developer'sstanding review protocol
   (bd memory, established during R11).
4. Second-order review: ten parallel per-concern agents (performance, reliability,
   scalability, code size, readability, observability, manageability, testability,
   ease-of-use, deployment) over the FINALIZED design — "what did the amendments themselves
   miss" — producing X1–X12 (commit 9031942).
5. Consolidation into `turn_result_design_final.md` with a supremacy clause and a §18
   traceability table (R# → fix-vof.N → commit → A# → final §).
6. Post-implementation human teaching session re-verifying shipped code against the spec —
   which caught the near-miss below.

## Findings worth knowing by name

- **R37 — the design's factual baseline was wrong.** A second review pass discovered the
  framework ALREADY persisted per-turn records (`save_conversation_incremental` into the
  Rdict-backed ConversationStore); the design was about to build a THIRD overlapping turn
  store. Resolved by "full absorption" into a unified ConversationTurnStore (v3.0), resolved
  FIRST because it changed every later step. Lesson: review pass 2 must re-check the design's
  factual claims about the existing code, not just its logic.
- **R1/A7 — ask_user role inversion.** Each ask_user exchange is an ordinary `CommandOutput`
  with `command_name='ask_user'`, `command_parameters` = the agent's QUESTION,
  `command_response.response` = the user's ANSWER ('' + success=False while unanswered),
  duration = user think time. Anyone rendering history misreads ask_user entries without this.
- **X3 — the A10×A28 contradiction.** Strict serialization rejection × best-effort record
  writes = a bad artifact deterministically fails every retry = silent permanent record loss.
  Fix: eager dev-time validation at the author's stack frame (warn-only in v2.21,
  unconditional since `ido-pyw.1`/3.4.0, hard rejection at v3.0) + placeholder envelopes at persistence
  boundaries so the record is ALWAYS written.
- **X1 — reversed A23.1.** "No write-time index" meant Redis SCAN MATCH over the entire
  keyspace; independently flagged by three of the ten review agents as "the worst
  under-priced decision". v3.0 fix: ZADD ordinal→turn_key pipelined with each write.
- **X6 — the minimal slice.** A14's plan put ~96 command-file migrations + a constructor shim
  into the "quick-fix minor" (~2.5k lines / ~120 files). X6 cut v2.21 to ~500 lines / ~6
  files, user-approved 2026-06-11; the `command_responses → command_response` collapse moved
  to a v3.0 big-bang with a rollback runbook and mixed-fleet prohibition. House pattern:
  "field ships, machinery deferred" — schema fields may exist dormant.

## What shipped, precisely

- **v2.21.0 (afcbe01):** `fastworkflow/turn.py` (TurnResult/TurnOutput/TurnStatus,
  `mint_turn_key` with injectable now/uuid seams, `collect_artifact_responses`,
  `merge_artifact_responses_into`, `FW_ARTIFACT_REF_KEY`, warn-only eager artifact
  validation); WEC accumulator (`_begin_turn`, `append_turn_output`); artifact-agnostic
  renames (gallery → command_outputs_with_artifacts, FW_PAYLOAD_REF_KEY → FW_ARTIFACT_REF_KEY);
  tests/test_turn_result_capture.py.
- **v2.21.1 (aa20c1b):** finalize MERGES collected artifacts into the single answer response's
  artifacts dict (instead of appending responses) — one-element `command_responses`, shape
  identical to the v2.20 baseline; collision rule `_<n>` suffix only on collision.
- **v2.21.2 (5a63790):** `process_turn()` returns slim public `TurnOutput`
  (`workflow_execution_context.py:484`; `process_action_turn` at :595); `process_message`
  deprecated with DeprecationWarning; `success` computed = `all(co.success)`, orthogonal to
  `status`/`failure_reason` (two interim semantics were rejected on the way — see
  feedback.md). Two latent bugs found while validating the suite: PyJWT>=2 rejects HS256 with
  an empty key (trusted-network tokens moved to the JWT 'none' algorithm) and `model_dump()`
  leaked datetime into JSONResponse (fixed with `mode='json'`).

## The near-miss (feedback.md Topic 5) — the most important lesson

During the 2026-06-12 teaching session, the reviewer verified against code that
`_finalize_agent_output` was UNCHANGED from the buggy version: v2.21 as staged only CAPTURED
turns into a dormant TurnResult; the original payload bug would have shipped "fixed" while
still user-visible. Recorded warning: "Don't merge v2.21 believing the payload bug is fixed."
Resolved same day by wiring the artifact merge at the WEC finalize chokepoint — the single
point both CLI and FastAPI flow through — with the cleaner transport-edge projection
consciously deferred to v3.0. House pattern: **fix at the chokepoint now, migrate to the
right layer at the major; and always re-verify shipped code against the original symptom.**

## The v3.0 roadmap (NOT implemented — do not assume it exists)

Per `turn_result_design_final.md` §14–15: `stores/` package, ConversationTurnStore +
ArtifactBlobStore (turn-scoped keys, size-threshold offload ~4KB), Redis ZSET ordinal
indexes with hash-tagged keys, MetricsSink, `fastworkflow admin` CLI, projections/read API,
retention tooling, deletion of `run_fastapi_mcp/conversation_store.py`, the
command_response collapse. `TurnResult.conversation_id`/`ordinal` are Optional/None today.
Config defaults were chosen so a 2.20→2.21 upgrade needs ZERO config changes
(FW_ARTIFACT_OFFLOAD_THRESHOLD_BYTES, FW_PENDING_TURN_TTL_SECONDS, etc. — see
`fastworkflow-config-and-flags`).

## fix-85g: the turns engine (v2.22.0) — the saga's concurrency sequel

Root cause and fix are in SKILL.md entry 3. The invariants, verbatim (they were each violated
by the original bug and are pinned in `fastworkflow/run_fastapi_mcp/turns.py`'s docstring):

1. **Wait-or-defer, never wait-or-abort** — a request's wait window expiring must never
   cancel the execution (asyncio.shield around the done_event wait as defense-in-depth).
2. **The per-channel active-execution pointer in TurnRegistry — never
   `runtime.lock.locked()` — is the single source of truth** for liveness, idempotency, and
   the 409 busy guard (the lock is deliberately released while deferring and across
   AWAITING_USER).
3. **Persist before DONE** — conversation + suspended state saved under runtime.lock BEFORE
   `exec_state=DONE` / `done_event.set()`, so a poller never sees "done" with unsaved state.
4. **Construction-order contract** — turn_key + done_event built and the pointer inserted
   BEFORE the asyncio.Task launches; no waiter can observe a half-built execution.
5. Idempotency key = hash(channel_id + kind + args): a retry with identical args rejoins the
   SAME execution.

Open residue: fix-85g Steps 2 (GET /turns polling, 429 backpressure, TTL eviction, trace
replay buffer) and 3 (durable distributed turn store, removing the LOST state) — `bd show
fix-85g`. Known survivor of the old pattern: `/invoke_agent_stream` still checks
`runtime.lock.locked()` in `run_fastapi_mcp/__main__.py`.

## Tracker caveat

fix-yy1 and fix-qtq describe this shipped work yet remain open (see SKILL.md entry 11). The
genuinely-unfinished part is the endpoint-level TurnOutput wire cutover (fix-qtq — explicitly
"a BREAKING WIRE CHANGE", pre-v3.0 wire shapes are NOT compatibility-protected) and
`run_fastapi_mcp/utils.py`'s `run_process_message*` helpers.

## Re-verification

- `git log --oneline --no-decorate -1 afcbe01 aa20c1b 5a63790 cf3eeae 9031942 42e9b3a 920d0de` (one hash at a time)
- `grep -n "def process_turn\|def process_action_turn\|merge_artifact_responses_into" fastworkflow/workflow_execution_context.py`
- `sed -n '1,30p' fastworkflow/run_fastapi_mcp/turns.py` (invariant docstring)
- `grep -n "locked()" fastworkflow/run_fastapi_mcp/__main__.py` (surviving old-pattern check)
- `head -20 docs/turn_result_design.md` (supersession banner); `ls docs/turn_result_*.md`

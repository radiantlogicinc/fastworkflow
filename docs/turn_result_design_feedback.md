# TurnResult — Design Feedback & Open Questions (to revise design/implementation)

Captured during a review/teaching session on 2026-06-12. This document records a
substantive pushback on the **turn identity** design (`turn_key` vs
`(conversation_id, ordinal)`) plus the counterarguments, so the design/implementation
can be revised deliberately.

---

## Topic 1 — Is the global `turn_key` redundant given `conversation_id` + `ordinal` (+ `started_at`)?

**Current design.** A turn carries three identity-ish fields:
- `turn_key: str` — global, unique, minted at turn start as
  `YYYYMMDDTHHMMSS.ffffffZ-<uuid4[:12]>` (coordination-free, time-sortable).
- `conversation_id: Optional[int]` — which conversation the turn belongs to.
- `ordinal: Optional[int]` — position of the turn within its conversation.
- (`started_at` also exists as timing data.)

In v2.21, `conversation_id` and `ordinal` are present on the type but **left `None`**
(the WEC's `_build_turn_result` only populates `turn_key`, `status`, `answer`, timing).

### Reviewer's position (pushback)

The global `turn_key` looks redundant. A turn only has meaning inside a conversation;
`(conversation_id, ordinal)` should be the real key, and `started_at + ordinal` already
give ordering. Specifically:

1. **Uniqueness is not a requirement** unless someone can name a consumer that queries
   turns *outside the context of their conversation*. If every access path is
   "within conversation X, give me turns", a globally-unique handle buys nothing.
2. **Coordination is only an issue** if the same user drives the same conversation via
   two completely different sessions concurrently. Absent that, an in-process counter
   assigns ordinals with no contention.
3. **"They aren't populated yet" is not an argument.** That's a v2.21 scoping choice,
   not a fundamental one — perhaps `ordinal` should simply be implemented now.
4. **Ordinal holes should be rare, logged, and ideally designed out.** Even if holes
   occur, that's acceptable as long as ordinals are unique within the conversation.
5. **Layering:** turns don't stand on their own; they only make sense inside a
   conversation. Nobody will cache/query a turn by some global UUID.

### Counterarguments to evaluate before revising

These are the existing-design arguments that survive the pushback best — the revision
should explicitly answer each, not ignore them:

- **A. Identity is needed at turn START, before persistence exists.** `turn_key` is
  minted in-memory at `_begin_turn`, and it doubles as the **pending-blob name** while a
  turn is suspended (`awaiting_user`) and not yet filed. To use `(conversation_id,
  ordinal)` instead, the **ordinal must be assignable at turn start** (read/increment a
  persisted per-conversation counter on `_begin_turn`). Feasible, but it pulls a
  persistence-assigned number into the transport-free core and onto the crash-ordering
  path. Decision needed: assign ordinal at start (and persist the counter), or keep a
  coordination-free start identity.

- **B. Single-writer assumption is the linchpin.** Point 2 is correct *only if* the
  design's single-writer-per-channel assumption (A27/A31) holds and is *enforced*
  (lease/fencing). The review flagged that A31 violations are currently undetected. If
  two writers ever touch one conversation (failover, sticky-session break), an in-process
  counter double-assigns an ordinal → silent collision/overwrite. A uuid suffix is the
  cheap insurance against that. Decision needed: enforce single-writer with a fence, or
  keep uuid as a collision backstop.

- **C. Cross-turn references.** The type already has `continuation_of: Optional[str]`
  (links a turn to a prior one) and `trajectory_ref`. Cross-turn / cross-conversation
  references (and potential client idempotency keys for transport-retry dedup) want a
  stable handle. This works with a composite `(conv, ordinal)` too, but it must then be
  encoded consistently everywhere a reference is stored.

- **D. Global maintenance scans exist regardless.** Retention sweeps ("delete turns older
  than N days"), co-GC, feedback joins, and fleet-wide observability (p95 latency across
  all turns) operate across conversations. These don't *require* a global key (iterate
  conversations → turns), but the store key grammar should make them efficient
  (the review's X1 proposes per-conversation index ZSETs precisely so global SCAN is
  avoided).

### Resolution options (for the revision)

- **Option 1 — keep `turn_key` as the store key**, populate `conversation_id`/`ordinal`
  now as grouping+ordering metadata. (Closest to current design; answers C/D cheaply;
  concedes points 1/5 are "fine, but uuid is cheap insurance for B.")
- **Option 2 — make `(conversation_id, ordinal)` the store key**, assign `ordinal` at
  `_begin_turn` from a persisted counter, enforce single-writer with a lease/fence, and
  drop or demote `turn_key` to a non-identity timestamp. (Embraces the pushback; must
  resolve A + B explicitly.)
- **Option 3 — hybrid:** composite key for storage/grouping, plus a short uuid *suffix*
  only as a collision backstop under writer-failover (keeps B's insurance without a
  fully global opaque key).

### Open questions to settle

- [ ] Name the actual read/query paths. Are there ANY that are not conversation-scoped?
      (retention, observability, feedback, continuation_of resolution)
- [ ] Is single-writer-per-channel *enforced* (fenced), or only assumed? (Gates point 2.)
- [ ] Should `ordinal` be implemented in v2.21 (assigned where — `_begin_turn` or at
      persistence)? Who owns the counter and how does it survive restart/crash?
- [ ] Are holes truly acceptable, or should the write path guarantee dense ordinals?
- [ ] If composite key wins, how are `continuation_of` / references encoded?

---

## Evidence gathered (2026-06-12) — single-writer is NOT enforced

Searched the full runtime path for any lock/lease/fence/owner mechanism:
- `session_state_store.py`: no matches (no lock/owner/lease/concurrency).
- `workflow_execution_context.py`: only "unlock" in a docstring about releasing the
  speedict store — no mutex/lease.
- `chat_session.py`: only a docstring + a metrics *label string*
  `('lock_contention', ['lock','acquire','release'], 'time')` — not a real lock.

Conclusion: single-writer-per-channel is an **unenforced assumption**, confirming the
architecture review.

### Topology split (decisive for the identity question)
- **Topology A (CLI / ChatSession):** one persistent worker thread per session is a de
  facto single-writer lock → in-process ordinal counter would never collide → pushback
  holds fully here.
- **Topology B (FastAPI / per-request WEC):** the deployment the original bug came from.
  Multiple pods, sticky sessions, retries, failover, NO fence → two writers can touch one
  conversation → an in-process ordinal counter could double-assign. UUID suffix is the
  cheap insurance here.

### Sharpest point for v2.21 specifically
The v2.21 minimal slice has **no persistence**: `process_turn` → `_build_turn_result`
constructs the TurnResult in memory and returns it; nothing is keyed by `turn_key`. So in
the code that exists today, `turn_key`'s justification is entirely forward-looking (v3.0).
For v2.21 in isolation, "redundant" is essentially unanswerable.

### The decision reduces to a fork
1. **Enforce single-writer** (Redis `SET NX EX` lease per channel at restore, or formalize
   the Topology-A worker as THE writer) → assign `ordinal` at `_begin_turn` from a persisted
   counter → drop `turn_key`. Critique wins outright.
2. **Don't enforce** → keep UUID as the Topology-B double-write backstop; treat
   `conversation_id`/`ordinal` as grouping+ordering metadata.

=> The real question is not "uuid vs composite key" but **"will single-writer-per-channel
be enforced?"** Everything else follows from that answer.

## Topic 2 — Collapse CommandOutput to a single command_response (reviewer agrees with v3.0)

Reviewer argued CommandOutput should carry a single `command_response`, not a list, now that
TurnResult exists. **This is already the ratified v3.0 design** (`turn_result_design_final.md`
line 43: `command_response: CommandResponse  # collapsed from list at v3.0 [A13/A15]`). The list
survives in v2.21 ONLY for backward-compat (`_response_mapping.py` + ~20 read sites iterate it;
X6 keeps the minor non-breaking). The `_map_singular_command_response` validator is the bridge,
and the build generators already emit `command_response=` (singular).

- OPEN: collapse removes per-command multiplicity — a single command can no longer emit two
  distinct payload-bearing responses (e.g., chart + table). Confirm xray `NLQueryTool` never
  relies on multiple responses-per-command before the v3.0 collapse; otherwise model both
  payloads in one response's artifacts or split into two commands.

## Topic 3 — Is deprecating process_message actually necessary / breaking?

Reviewer: TurnResult is internal (project it for the user), so process_message can keep
returning CommandOutput → not breaking.

Resolution: For the END USER, correct — TurnResult never hits their wire; they get a projection
(answer + gallery). BUT the reviewer's own Topic-2 collapse forces process_message's deprecation:
- Collapsed CommandOutput holds ONE response/payload.
- A turn can produce MANY payloads (multi tool-call).
- So one CommandOutput can no longer represent a whole turn → returning it re-drops the gallery
  (the original bug).
- The only type that carries the full turn is TurnResult → the turn entry point must return it.
process_turn returns TurnResult to the TRANSPORT layer (internal infra), which projects to the
client wire — so "TurnResult is internal" and "process_message must be deprecated" are both true.
The breaking change is on the WEC embedding API (process_message is a public method embedders
call directly), NOT on the end-user wire. Clean migration = deprecate + add process_turn
(no silent return-type swap, X10a).

## Topic 4 — Writer conflict detection exists in v3.0 (not just hope)
`turn_result_design_final.md` line 254: v3.0 increments `fw_writer_conflicts_total` and logs
loudly on single-writer-per-channel violations (detection), though still NOT a full lease.
Reviewer's "add a lease, delete the UUID" remains a stronger open alternative to mere detection.

## Topic 5 — v2.21 captures but does NOT surface payloads (original bug still user-visible)

Reviewer proposed: keep TurnResult internal, and project ONLY the extracted payload values into
the (legacy-shaped) user-facing CommandOutput.artifacts — no nested object serialization, so no
recursion. This is the natural projection because xray's `command_output_to_response_tuples`
already reads payload/payload_hint/request_id from `command_responses[0].artifacts`.

Verified against code: `_finalize_agent_output` (workflow_execution_context.py ~L719-755) is
UNCHANGED from the buggy version — it synthesizes a single `CommandResponse(response=result_text)`
with only `conversation_summary` in artifacts, NO payload/gallery. `process_message` (the path
FastAPI + CLI actually call) returns through it.

=> CONCLUSION: v2.21 only CAPTURES the turn into TurnResult (via the dormant process_turn). It
does NOT surface payloads to the client. The original xray payload-loss bug is therefore STILL
user-visibly unfixed in v2.21. The projection step (transport reads TurnResult, projects gallery
payloads into the legacy CommandOutput/ResponseTuple shape) is deferred/unwired.

ACTION/RISK: Don't merge v2.21 believing the payload bug is fixed. Decide where the projection
lives (transport edge in run_fastapi_mcp, deriving from process_turn's TurnResult) and whether to
ship it in v2.21 or v3.0. Reviewer's payload-only-into-artifacts projection is the recommended
shape for the existing xray client.

### RESOLUTION (2026-06-12) — projection wired in v2.21 at the WEC finalize chokepoint

Decision: ship the payload projection in **v2.21**, located in the WEC agent finalize step
(`_finalize_agent_output`) rather than at the run_fastapi_mcp transport edge.

Rationale for the placement (vs. the transport-edge option above):
- `_finalize_agent_output` is the exact buggy chokepoint and the **single point both
  transports flow through**: CLI (`ChatSession`) and FastAPI both reach it via
  `process_message` → `_execute_message`, and `process_turn` reuses the same dispatch.
  Fixing it once surfaces payloads everywhere in v2.21.
- The transport-edge option requires switching the transport from `process_message` to
  `process_turn` and projecting `TurnResult.gallery` there. That is the cleaner **v3.0**
  shape (transport owns projection from the internal TurnResult) but is a larger change
  (streaming path included) and is deferred — `process_message` stays the transport call
  in v2.21.

Shape (artifact-agnostic projection, no recursion):
- The framework knows nothing about client-specific artifact keys (no "payload",
  "payload_hint", "request_id", etc.). It only knows "this response carries artifacts"
  and preserves the dict verbatim. **What** a key means and which outputs are worth
  rendering richly (a gallery, a chart, a download) is entirely the consuming client's
  concern.
- New helper `fastworkflow.turn.collect_artifact_responses(command_outputs)` flattens the
  turn's `CommandResponse`s whose `artifacts` dict is non-empty, in turn order, returning
  the **original** response objects. No nested TurnResult/object serialization.
- `_finalize_agent_output` keeps the synthesized agent answer at `command_responses[0]`
  and **appends** the artifact-bearing responses after it. A client that reads artifacts
  off each `command_response` (e.g. xray's `command_output_to_response_tuples`) therefore
  receives every tool-call's structured output in the legacy CommandOutput shape.
- `TurnResult.answer` is unaffected: `_build_turn_result` still aliases
  `command_responses[0]` (the textual answer), and `TurnResult.command_outputs` / the
  `command_outputs_with_artifacts` property still read from `self._turn_outputs`, so there
  is no double-count. (The former `gallery` property was renamed to
  `command_outputs_with_artifacts` and its predicate made key-agnostic — "has artifacts"
  rather than "has a non-empty `payload`".)

Scope note: projection is applied only on turn completion (`_finalize_agent_output`). A turn
suspended on `ask_user` returns the clarification without its partial pre-suspension artifacts;
those surface together when the resumed turn finalizes (the whole turn's `_turn_outputs` are
projected then). The deterministic/assistant path already surfaces its own command's artifacts
(its `command_responses[0].artifacts`), so no change was needed there.

## Notes
- Reviewer demonstrated mastery of the `answer` vs `command_outputs` split, the
  "one logical turn = one key, across suspensions" invariant, and `success = answer.success`
  (agent self-correction rationale) before raising this critique. Also independently re-derived
  the v3.0 CommandOutput collapse and reasoned correctly about projection vs internal record.

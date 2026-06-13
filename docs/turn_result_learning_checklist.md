# TurnResult — Learning Checklist

A running guide to make sure you *deeply* understand the v2.21 `TurnResult` capture
change (uncommitted work: `fastworkflow/turn.py` + edits to `__init__.py`,
`workflow_execution_context.py`, `workflow_agent.py`, `utils/react.py`,
`chat_session.py`, the `build/` generators, and `tests/test_turn_result_capture.py`).

Legend: `[ ]` not yet demonstrated · `[~]` partial · `[x]` mastered

---

## Stage 1 — The Problem (why this work exists)

- [x] 1.1 Restate the original bug symptom (what xray's `/invoke` returned, and when)
- [x] 1.2 Explain why the producer, mapping layer, and command handler were all *correct*
- [x] 1.3 The two places the agent path destroys the payload (ReAct text boundary + synthesized final output)
- [x] 1.4 Why the deterministic (`/`-prefixed) path did NOT have the bug — the "tell"
- [x] 1.5 Why this is architectural, not a mapping bug
- [x] 1.6 The deeper question: multiple tool calls per turn — where multiplicity is supported vs dropped

## Stage 2 — The Solution (what was built and why this way)

- [x] 2.1 What a "logical turn" is, and the core invariant (one turn = one key = one record, across suspensions)
- [x] 2.2 `TurnResult` shape: `answer` vs `command_outputs`, and what `success` means (the single wire predicate)
      NOTE: reviewer raised a substantive critique of turn_key vs (conversation_id, ordinal) —
      see docs/turn_result_design_feedback.md. Mastery demonstrated via the critique itself.
- [~] 2.3 The turn accumulator on the WEC: `_begin_turn` / `append_turn_output` / lifecycle (A30 reset)
- [x] 2.4 `process_turn()` vs deprecated `process_message()`; staged-but-dormant (tracer-bullet);
      no-silent-return-type-swap; TurnResult is INTERNAL (clients get a projection) — reviewer found this unprompted
- [ ] 2.5 ask_user role inversion (A7): question in `command_parameters`, answer in the response; `success=False` = unanswered
- [ ] 2.6 Suspend/resume capture: same turn key, ordering, `suspended_ms`, the two-topology double-append avoidance
- [ ] 2.7 Failure capture in `_execute_workflow_query` (capture-then-reraise, never mask; `CommandCancelledError` passthrough)
- [ ] 2.8 `exhausted` flow from ReAct → agent_result → `failure_reason="max_iters_exhausted"`, `success=False`, status still COMPLETED
- [x] 2.9 Forward-compat shim: `command_response=` (singular) → `command_responses` list; build generators emit new style.
      Reviewer independently re-derived the v3.0 collapse and connected it to why process_message must be deprecated.
- [ ] 2.10 Eager artifact validation (warn-only in v2.21) and the `gallery` property

## Stage 3 — Broader Context (why it matters / impact)

- [ ] 3.1 Why v2.21 is intentionally a *minimal* slice (X6) and what was deferred to v3.0
- [ ] 3.2 Forward-compatibility strategy: non-breaking now, the v3.0 cutover, deprecation of `process_message`
- [ ] 3.3 What this unblocks (durable per-turn persistence, observability, agent memory) and the big risks (read side, A10×A28)
- [ ] 3.4 Who/what is impacted: framework consumers, xray, command authors, the build pipeline

---

## Session log

- Stage 1 COMPLETE. Restated symptom + agent-vs-assistant divergence unprompted; identified
  both drop points after a nudge; nailed the text-token "why"; reached the multiplicity/provenance
  insight independently. Quiz: 4/4.

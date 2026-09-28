# The v3.0 boundary: what is designed but unbuilt, and how not to collide with it

Companion to SKILL.md section 6. Source of truth: `docs/turn_result_design_final.md`
("the authoritative implementation spec ... where this conflicts with any of those, this
document wins", lines 3-8). Facts verified 2026-07-09 against v2.22.2 (commit c33b9a5).

Everything in this file is ROADMAP. Verified absent on main:
`ls fastworkflow/turn_accumulator.py fastworkflow/turn_serializer.py fastworkflow/stores fastworkflow/metrics.py`
→ all "No such file or directory". No beads epic for v3.0 exists (fix-vof was
review-only, 48/48 closed). Whether v3.0 is scheduled at all is an OPEN question.

## What shipped (v2.21–v2.22.2) vs what did not

| Shipped (in code today) | Where | Unbuilt (spec only) | Spec section |
|---|---|---|---|
| `TurnResult`/`TurnOutput`/`TurnStatus` types, `mint_turn_key`, artifact collect/merge, warn-only eager artifact validation | `fastworkflow/turn.py` (264 lines) | `turn_accumulator.py` as a separate module (accumulator lives inline in WEC today) | §15 |
| WEC turn accumulator + `process_turn`/`process_action_turn` | `workflow_execution_context.py:145-240, 484-495, 595-610` | `turn_serializer.py` (serializer + envelope + reader + projections) | §6, §15 |
| Topic-5 artifact merge at finalize | `wec:786-791` | `stores/` package: `base.py`, `sanitize.py`, `pending.py`, `conversation_turn.py` (**ConversationTurnStore**), `artifact_blob.py` (**ArtifactBlobStore**) | §6, §15 |
| `process_message` DeprecationWarning | `wec:469-482` | `process_message` REMOVAL (loud AttributeError + migration guide) | §14 v3.0 |
| Forward-compat shim: `CommandOutput(command_response=...)` accepted, mapped to the list | `fastworkflow/__init__.py:85-97` | `command_responses` → `command_response` COLLAPSE (constructor shim introduced at 3.0, removed 4.0) | §14 |
| Turns engine Step 1 (TurnRegistry, submit_turn, idempotency) | `run_fastapi_mcp/turns.py` | Step 2 (GET /turns polling, real 202s, trace replay buffer, sized executor + 429, TTL eviction) and Step 3 (durable distributed turn store) — open **fix-85g.9–.13** | `docs/fastworkflow_turns_async_execution_design.md` |
| Disk/Redis SessionStateStore for suspended sessions | `session_state_store.py` | Redis keyspace with hash-tags + write-time ZSET indexes; writer lease (detection-only, `fw_writer_conflicts_total`); retention tooling; `fastworkflow admin` CLI; MetricsSink; projections/read API | §5, §10, §11, §13 |

`TurnResult.conversation_id` and `ordinal` exist on the model (`turn.py:253-254`) but are
left `None` — the capture is dormant until a `ConversationTurnStore` consumes it.

## The v3.0 keyspace (spec §5) — for orientation only

```
fw:conv:{ch/cv}                                conversation metadata record
fw:turn:{ch/cv}:<turn_key>                     turn record
fw:artifact:{ch/cv}:<turn_key>:<sha256-hex32>  artifact blob
fw:feedback:{ch/cv}:<turn_key>                 feedback card
fw:turnidx:{ch/cv}                             ZSET ordinal → turn_key (write-time index, X1)
fw:convidx:{<channel>}                         ZSET created_ts → conv_id
fw:lease:{<channel>}                           writer lease (detection)
```
`{ch/cv}` is a literal Redis hash-tag so Redis Cluster colocates a conversation's keys;
all reads go through the index (`ZRANGE` + pipelined `MGET`); `SCAN` survives only inside
retention deletes. This design exists because review finding X1 reversed the original
"no write-time index" decision (three of ten review agents independently flagged it).

## The wire hard-break (spec §14, v3.0 release train)

Big-bang cutover, mixed fleets forbidden:
- Endpoints + SSE return `TurnOutput.model_dump()` — the slim public projection, NOT the
  internal `TurnResult`. (Endpoint cutover is separately tracked NOW as open epic
  **fix-qtq**: `run_fastapi_mcp` responses still serialize `CommandOutput`.)
- MCP `isError = not success`.
- `run_fastapi_mcp/conversation_store.py` DELETED; its logic (`generate_topic_and_summary`,
  `_ensure_unique_topic`, `restore_history_from_turns`) lifted into core; Rdict data loss
  announced via a one-time synthesized per-channel notice; Rdict files left in place with a
  tombstone marker for rollback detection.
- `action_log` retired; queue contract switch [A19].
- Rollback only to ≥2.21.x; 3.0-era conversations are unrecoverable on rollback
  (documented, bidirectional loss); clients roll back in lockstep.
- Cutover runbook (ordered): announce → provision Redis (dedicated DB, noeviction, TLS) →
  set env vars → drain (`admin pending list/cancel`) → stop ALL old pods → start new →
  cut clients → enable retention cron + alarms.

## Config inventory already reserved (spec §12)

Defaults were chosen so a 2.20→2.21 upgrade needs ZERO config changes. Names reserved by
the spec (none of them is consumed in code today; the artifact validator is unconditional):
`FW_ARTIFACT_OFFLOAD_THRESHOLD_BYTES=4096`, `FW_MAX_INLINE_ARTIFACT_BYTES=10485760`,
`FW_MAX_TURN_ARTIFACT_BLOBS=64` / `FW_MAX_TURN_ARTIFACT_BYTES=52428800`,
`FW_MAX_TRAJECTORY_BYTES=262144`, `FW_PENDING_TURN_TTL_SECONDS=604800`,
`FW_MEMORY_PROJECTION_TURNS=10`, `FW_CONVERSATION_MAX_TURNS=200` (0 = off),
`FW_ALLOW_DISK_STORES`. Do not repurpose these names.
(Verified at 3.4.0: none of the reserved names above is read anywhere in `fastworkflow/`.)

## Open forks deliberately left standing (do not silently resolve them)

1. **turn_key vs (conversation_id, ordinal) identity**: will single-writer-per-channel be
   enforced (lease + fence, drop the UUID) or does the UUID stay as insurance? Both
   positions are recorded (`docs/turn_result_design_feedback.md:154-155, 189-192`); no
   final ruling. Related open issue: fix-5ka.
2. **command_responses collapse risk**: unconfirmed whether xray's NLQueryTool ever relies
   on multiple responses per command — must be verified before the collapse
   (feedback doc, Topic 2).
3. **ChatSession migration**: `process_message` is deprecated while the CLI transport still
   calls through it internally; no issue tracks migrating ChatSession to `process_turn`
   before v3.0 removes `process_message`.

## Rules of engagement for changes today

- Anything you add to `CommandResponse.artifacts` must be `None/str/int/float/bool` or
  dict/list/tuple compositions — v3.0 REJECTS other types at record filing; v2.21+ already
  warns (`turn.py:147-165`).
- New code should construct `CommandOutput(command_response=...)` (singular) — the shim
  maps it; this is the forward-compatible spelling.
- New endpoint work should return `TurnOutput`-shaped bodies (see `render_turn_response`,
  `turns.py:386-424`) rather than raw `CommandOutput` — that is the fix-qtq direction.
- Do not build ad-hoc turn persistence; if you need durable turns, that IS the v3.0
  ConversationTurnStore work — raise it with Dhar rather than routing around the spec.

Re-verify before relying on any of this:
`ls fastworkflow/stores 2>&1`, `bd list --status open | grep -E 'fix-(qtq|85g|5ka)'`,
`grep -n 'command_response' fastworkflow/__init__.py`, and reread
`docs/turn_result_design_final.md` §14-15.

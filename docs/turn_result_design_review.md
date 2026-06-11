# Critical review: TurnResult / `CommandOutput` redesign and durable turn-output persistence

Review of `docs/turn_result_design.md`. All code references were verified against the working
tree at the time of review (v2.20.0, commit `dc250b3`). Findings R1–R36 are from the first
review pass; **R37–R47 are from a second, deeper pass** that uncovered — among other things —
an existing turn-persistence subsystem the design does not account for.

**Verdict:** the diagnosis is correct and the core design is sound. The type algebra
(`CommandResponse` = value, `CommandOutput` = execution, `TurnResult` = turn) is the right
factoring; accumulate-all at the `invoke_command` choke point is the right capture strategy;
option D (light record + payload offload) is the right durability shape. However, the review
found **four internal contradictions** (R1, R2, R37, R42), several **blocking gaps**, an
**architectural consolidation question the design never asks** (R37), and a long tail of
refinements. "Open questions: none remaining" (section 13 of the design doc) is an overclaim.

## Severity index

| # | Finding | Severity | Design sections affected |
|---|---------|----------|--------------------------|
| R1 | ask_user exchanges lost; turn transcript not interleaved | **RESOLVED 2026-06-11** — ask_user modeled as a `CommandOutput` in the ordered list (Dhar's simplification) | 3.1, 5.4, 7.6, 9, decision 18 |
| R2 | Content-addressing makes co-GC incoherent; cross-tenant handle sharing | **RESOLVED 2026-06-11** — turn-scoped payload keys; record-mediated access | 7.8, 8.2, decisions 13, 19 |
| R37 | Framework already persists per-turn records (`ConversationStore`); design premises false; third overlapping store | **RESOLVED 2026-06-10** — full absorption into unified `ConversationTurnStore` | 7.1, 7.2, 9.1, decisions 11, 18 |
| R42 | `next_actions`/`recommendations` still die at the agent boundary — the original bug class survives the fix | **RESOLVED 2026-06-11** — fields verified unused; payload-only gallery kept; answer stays bare; provenance via text (xray scope) | 2.4, 5.4, 10.3, decision 2 |
| R3 | HTTP/MCP wire-contract break undocumented; external authors break | Blocking | 11, decisions 4, 8 |
| R4 | Build-time generators and internal `_workflows` commands missing from change list | Blocking | 11 |
| R5 | Artifact serialization/offload contract unspecified | **RESOLVED 2026-06-11** — size-threshold offload; reserved-key envelope; strict rejection; honest `command_parameters` | 8.2, 8.3, decision 16 |
| R6 | Failed executions and failed turns invisible to review | **RESOLVED 2026-06-10** — capture-with-detail; record+re-raise; TTL+lazy abandon | 3.1, 5.5, 7.6 |
| R7 | Sensitive data flips from ephemeral to durably-persisted-by-default | **RESOLVED 2026-06-11** — no switches; infra-executed age-based retention; security contract documented | 7, 7.8 |
| R8 | `process_action` / MCP paths not covered | **RESOLVED 2026-06-11** — full turns: TurnResult + record, empty user_message | 5.5, 5.6, 10 |
| R40 | `TurnResult` has no turn-level success semantics | **RESOLVED 2026-06-11** — success=answer.success; max-iters → success=False; serialized | 5.4, 5.5, 11 |
| R41 | Live response path ambiguous: in-memory payloads vs put-then-get round trip | High | 10.3, 8.2 |
| R38 | `/post_feedback` vs write-once review records | High | 7.5, 8.1 |
| R39 | `/cancel_pending` (cancelled turns) unaccounted for | **RESOLVED 2026-06-10** — cancelled turns recorded; memory projects completed-only | 7.6, 10.1 |
| R43 | CLI / `command_output_queue` contract undefined after redesign | High | 5.6, 11 |
| R9 | `channel_id` vs conversation boundary | **RESOLVED 2026-06-10** — structural via A1; eager id reservation; auto-cancel on switch | 7.7, 8.1 |
| R10 | Answer aliasing: copy-on-serialize hazard + headline/gallery duplication | High | 5.5, 8.3, 10.3 |
| R11 | `awaiting_user` artifact-sniffing → first-class turn status | **RESOLVED 2026-06-10** — 5-value `TurnStatus`; artifact removed immediately; no further cleanup | 5.5, 7.6, 10.1 |
| R46 | Big-bang release sequencing; no staged migration path | High (process) | 11, decisions 4, 8, 22 |
| R12 | `nested_turn`: speculative, singular, suspension semantics undefined | Medium | 5.2, 5.3, 7.9, 8.3 |
| R13 | Assistant-mode clarification spans N unlinked records | Medium | 5.5, 7.6 |
| R45 | Payload accumulation changes the per-turn memory profile | Medium | 3.1, 7.3 |
| R44 | Raw vs refined user message — capture both | Medium | 5.4 |
| R47 | Conversation summary stored in three places | Low | 5.5, 7.7, 9 |
| R14–R27 | Store/durability refinements | Medium/Low | 7, 8 |
| R28–R33 | Observability & roadmap alignment | Medium | 6, roadmap |
| R34–R36 | Editorial & process | Low | — |

---

## 1. Blocking gaps and design contradictions

### R1. ask_user exchanges are silently dropped, and the turn transcript loses interleaving

This is an internal contradiction between three parts of the design:

- Section 3.1 states that `ask_user` "returns strings and never reach[es] `invoke_command`, so
  they are naturally excluded" from `command_outputs` — framed as a feature.
- Section 9.1 records that today's `action_log` is appended in **both** `_execute_workflow_query`
  **and** `_post_ask_user_response`. Verified: `workflow_agent.py:232-249` appends
  `{"agent_query": clarification_request, "user_response": user_response}` to the action log,
  **in sequence with** the command records.
- Decision 18 claims retiring `action_log` and deriving the summary view from `command_outputs`
  has "zero regression."

All three cannot be true. Consequences of implementing as written:

1. **The review record omits clarification dialogs entirely.** A reviewer sees
   `user_message → commands A, B, C → answer`, when the actual turn was
   `user_message → A → agent asked Q1 → user answered → B → Q2 → answer → C → final answer`.
   For a review/observability feature, the human-in-the-loop exchanges are among the most
   important events to capture — they explain *why* the agent's trajectory bent.
2. **The LLM conversation summary regresses.** `_extract_conversation_summary` today receives
   the ask_user dialog records; the proposed text projection of `command_outputs` will not
   contain them. "Zero regression" is false as stated.
3. **Interleaving order is unrepresentable.** Even if exchanges were bolted on as a separate
   list, `TurnResult` has no ordering mechanism across the two lists.
4. **`user_message: str` (singular) is wrong for logical turns.** A turn that suspended three
   times has four user inputs (original + three clarification answers). The design stores one.
   Note this also applies to **Topology A**, where `_ask_user_tool`
   (`workflow_agent.py:262-289`) blocks on the queue *inside* one `process_message` call — so a
   single call spans multiple user inputs even without suspend/resume.

Also: in Topology A, `_ask_user_tool` constructs a clarification `CommandOutput` and puts it
**directly on the output queue**, bypassing `invoke_command` — a second user-visible output
path the accumulator misses.

**Recommendation:** make the turn record an *ordered event sequence*, not a bare list of
command outputs. Either:

- (preferred) `TurnResult.events: list[TurnEvent]` where `TurnEvent` is a discriminated union:
  `CommandExecutionEvent(command_output)` | `UserExchangeEvent(question, answer)` — with
  `command_outputs` retained as a derived convenience view; or
- minimally, `TurnResult.user_exchanges: list[UserExchange]` where each entry carries a
  sequence index into the unified ordering, plus `user_messages: list[str]`.

The event-sequence shape also gives a natural extension point for trajectory events later (R28).
Capture point: `_post_ask_user_response` is the single choke point for completed exchanges (both
topologies route through it), exactly parallel to the `invoke_command` choke point.

**RESOLVED 2026-06-11 (with Dhar) — superseding the reviewer's recommendation with a simpler
design from Dhar.** Decisions:

1. **No `UserExchangeEvent`, no discriminated union: `ask_user` is modeled as a command
   execution.** It is already a tool taking a single text parameter, so each exchange becomes a
   `CommandOutput` with `command_name="ask_user"`, `command_parameters=` the clarification
   question, `command_response.response=` the user's answer — appended **in order** into the
   existing `command_outputs` list. The design doc's original `TurnResult` shape (answer +
   `command_outputs`) survives intact; interleaving is solved with zero new types. Precedent:
   `_ask_user_tool` (`workflow_agent.py:276`) already constructs exactly such a `CommandOutput`
   for the output queue.
2. **Unanswered-question convention (satisfies A4/A5):** the ask_user entry is appended at
   ask/suspend time with `response=""` and `success=False`; when the answer arrives
   (`_post_ask_user_response`, both topologies), the response is filled and `success` flips to
   `True`. Cancelled/abandoned records therefore end with the unanswered question; a suspended
   partial blob naturally shows what is pending.
3. **`user_message` holds the original request only.** Clarification replies live solely in
   their ask_user entries — each utterance stored once, in context, in order. (R44 separately
   decides capturing the refined form of the message.)
4. **Documented role inversion:** for ask_user entries, `command_parameters` is what the
   *agent* said and `response` is what the *user* said — projections and readers must account
   for this. The summary text projection maps cleanly (today's
   `{"agent_query": q, "user_response": a}` becomes `parameters=q, response=a`).
5. **Knock-on effects:** R29 timestamps attach uniformly per `CommandOutput`; R28's trajectory
   remains an offloaded blob (no inline event type needed); R30's user-facing projection keeps
   ask_user entries visible via `command_name` filter; the derived LLM-summary view includes
   exchanges, so the decision-18 "zero regression" claim becomes true again.

Recorded in `docs/turn_result_design.md`, Amendments A7.

### R2. Content-addressing and "co-GC under one lifecycle key" are jointly incoherent

Decision 13 makes `PayloadStore` content-addressed (same bytes → same handle, dedup across
writes). Decision 19 requires infrastructure to "GC the review record and its referenced
payloads **under one lifecycle key**." These conflict:

- A handle produced by content addressing is **shared by every record whose payload hashed to
  it** — across turns in a channel, and across channels/tenants. Identical payloads are not
  rare in this domain: re-running the same access-review query, empty-result CSVs, header-only
  CSVs all collide *by design* (that is what dedup means).
- Therefore deleting "turn A's payloads" when GC'ing turn A's record can dangle turn B's live
  record. There is no "one lifecycle key" that owns a shared handle. The contract as written
  cannot be implemented by the infra team it is delegated to.

There is also a **tenancy/security wrinkle**: a content-addressed handle is a global key. If any
future read API fetches payloads by bare handle, possession (or guessing) of a hash lets one
channel read another channel's payload — and even without read access, idempotent `put`
behavior makes the store a **hash oracle** (you can confirm whether specific content exists in
some other tenant's turns). For access-review data this matters.

Options, with a recommendation:

- **(a) Turn-scoped payload keys (recommended).** Key payloads as
  `{channel_id}/{turn_key}/{content_hash}` (disk: directory per turn; Redis:
  `fw:payload:{channel}:{turn_key}:{hash}`). Co-GC becomes a trivial prefix delete; tenancy
  isolation is structural; writes remain idempotent *within* a turn (which is where retry
  idempotency actually matters). The sacrifice is cross-turn dedup, whose benefit is speculative
  and likely small. This keeps every property the design actually needs and discards only the
  one causing the incoherence.
- **(b) Reference counting.** Correct in principle; in practice racy without atomic ops (the
  disk backend has none), refcounts themselves need GC, and put/decrement races across pods are
  real. Do not delegate this to "infra."
- **(c) Mark-and-sweep.** Scan all records, collect live handles, delete the rest.
  Operationally heavy; nobody will build it.
- **(d) TTL-everything with refresh-on-put.** Give payloads TTL ≥ record retention; a
  content-addressed re-`put` refreshes TTL (Redis `SET` does; disk can touch mtime). Combined
  with the reader-tolerates-missing rule this works, but the disk backend then needs an
  mtime sweeper, and the hash-oracle/tenancy issue remains.

Whichever is chosen, decision 13/19 must be rewritten — currently the design hands infra an
impossible contract and calls it out-of-scope.

**RESOLVED 2026-06-11 (with Dhar).** Decisions:

1. **Turn-scoped payload keys** (option a):
   `fw:payload:{channel_id}:{conv_id}:{turn_key}:{content_hash}`. Every payload is owned by
   exactly one turn — sharing is impossible by construction, so the co-GC contradiction
   dissolves. `PayloadStore.put` gains scope parameters (channel, conversation, turn key); the
   returned handle is the full scoped key, opaque to callers. Content-addressing survives only
   as the leaf segment, providing **within-turn** retry idempotency (the only place idempotency
   matters). Cross-turn dedup is deliberately abandoned — storage is cheap, the benefit was
   speculative, and tenant isolation becomes structural.
2. **Co-GC becomes literal:** with A1's keyspace, deleting a conversation prefix removes
   conversation metadata, turn records, and payloads in one stroke (disk: directory tree;
   Redis: prefix SCAN+DEL). Decision 19's "one lifecycle key" is now simply the conversation
   prefix — a contract infra can actually implement. The reader still tolerates missing
   payloads as defense in depth. R21's missing-delete-surface concern largely dissolves
   (prefix delete is the natural operation).
3. **Record-mediated access rule (adopted as contract):** any read API authorizes access to
   the *turn record* first (JWT channel binding; admin path per R32), then resolves handles
   found inside it. Bare-handle fetch endpoints are forbidden. This closes the
   hash-oracle/cross-tenant-probe class entirely.

Supersedes design decisions 13 (content-addressed global store) and 19 (co-GC wording).
Recorded in `docs/turn_result_design.md`, Amendments A8.

### R37. The framework already persists per-turn records durably — the design's baseline is wrong, and it builds a third overlapping turn store

This is the largest second-pass finding. The design's durability analysis (sections 7.1/7.2)
claims "on a normal completed turn … there is **no per-turn or per-command durable write** on
the happy path," and section 9.1 claims `action_log` has a "sole data consumer"
(`_extract_conversation_summary`) and "no external consumer." Both claims are **false for the
framework's own bundled FastAPI server**, which the design never examined on this axis:

- `WorkflowExecutionContext` maintains a **second per-turn accumulator** the design never
  mentions: `conversation_history` (a `dspy.History`). Every resolution path appends a per-turn
  record `{conversation summary, conversation_traces, feedback}`:
  - agent path: `_finalize_agent_output` (`workflow_execution_context.py:549-554`), where
    `conversation_traces` is the JSON projection of `action_log`;
  - assistant path: `_process_message` (`:651-657`), traces = the single command record;
  - action path: `_process_action` (`:715-721`).
- The bundled server **durably persists these records after every turn** via
  `save_conversation_incremental` (`run_fastapi_mcp/utils.py:693-708`), called on the happy path
  of `/invoke_agent` (`__main__.py:794`), `/invoke_agent_stream` (`:901`), and
  `/invoke_assistant` (`:1038`), into **`ConversationStore`**
  (`run_fastapi_mcp/conversation_store.py`) — an Rdict-backed, per-channel store with
  conversation ids, AI-generated topics/summaries (`LLM_CONVERSATION_STORE`), turn lists, and a
  dump facility.
- The `action_log` consumer chain is therefore:
  `action_log → (summary, traces) → conversation_history → (a) the agent's cross-turn memory
  (restored on session rebuild, `utils.py:298-302`, and consumed by `_refine_user_query`),
  (b) durable `ConversationStore`, (c) the `/post_feedback` target (R38).` Retiring `action_log`
  (design section 9) touches all three, not one.

Consequences:

1. **Decision 11's framing ("two stores, two animals") undercounts.** With `TurnReviewStore` +
   `PayloadStore` the framework would run **three turn-shaped persistence subsystems**
   (`ConversationStore` turns, pending `SessionStateStore` partials, review records) plus a
   third backend technology (Rdict alongside disk-JSON and Redis), with overlapping content:
   `conversation summary` ≈ the review record's summary metadata; `conversation_traces` ≈ the
   text projection of `command_outputs`; `feedback` ≈ review metadata. Divergence between these
   is the exact failure mode decision 18 cites when retiring `action_log` ("removes a
   divergence risk: two parallel turn logs") — the design removes one parallel log and builds
   another.
2. **The latency baseline argument changes.** Option D's "preserve `action_log`-level latency
   (zero durable writes on completed turns)" was already not the bundled server's reality — it
   writes Rdict per turn today. The review write is therefore not a new class of cost there;
   conversely, xray (which wires none of this) genuinely goes from zero to one write per turn.
3. **`ConversationStore` is single-pod** (Rdict on local disk), which the multi-pod review
   design must not inherit by accident.

**Recommendation: consolidate rather than add.** Make the (light) `TurnResult` record the
single per-turn persistence unit:

- `TurnReviewStore` becomes the system of record for turns. The `{summary, traces, feedback}`
  triple becomes fields of (or derivations from) the review record.
- `ConversationStore` retains only conversation-*level* concerns: conversation ids,
  topic/summary, and an ordered list of turn keys (which also resolves R9 cleanly — the
  conversation id comes from here).
- `conversation_history` (the agent's cross-turn dspy memory) is rebuilt from review records
  (`restore_history_from_turns` already shows the required projection is tiny).
- Backend unification (Rdict → the disk/redis pair, or vice versa) can be staged, but the
  design must at minimum *decide and document* the relationship between the three stores. The
  current design is silent because it never discovered `ConversationStore`.

**RESOLVED 2026-06-10 (with Dhar) — full absorption.** Decisions:

1. **`ConversationStore` is eliminated as a subsystem.** A single unified store (working name
   **`ConversationTurnStore`**) replaces both it and the previously proposed `TurnReviewStore`,
   owning two record types:
   - *Conversation metadata records* — key `fw:conv:{channel_id}:{conv_id}`; value
     `{topic, summary, status (active|closed), created_at, updated_at, schema_version, metadata}`.
   - *Turn records* — key `fw:turn:{channel_id}:{conv_id}:{sortable-ts}-{uuid}`; value = the
     light `TurnResult` (serialization per R5/R25).
2. **`save_conversation_incremental` is deleted.** The once-per-logical-turn record write is the
   *only* durable turn write (the unamended design would have made the bundled server write
   twice per turn: Rdict + review record).
3. **All eight consumers re-point to the unified store:** session restore and
   `/activate_conversation` project the agent's `dspy.History` from turn records
   (`restore_history_from_turns` already shows the projection shape); `/list_conversations`
   reads conversation metadata records; `/new_conversation` and the shutdown hook finalize the
   metadata record (LLM topic/summary) and open the next; `/dump_conversations` iterates the
   keyspace; `/post_feedback` attaches per R38 (now unblocked).
4. **Source of truth for agent memory = turn records.** `conversation_history` becomes a
   per-session in-memory cache rebuilt by projection on restore/activation. Constraint: turn
   records must always carry the projection fields (conversation summary, traces-equivalent
   text view, feedback reference).
5. **Backend:** the unified store ships on the existing disk/redis split (factory mirroring
   `get_session_state_store()`). Rdict exits the conversation subsystem entirely (it remains
   only for `Workflow` state, out of scope).
6. **Migration: accept loss.** Existing Rdict conversation data is neither read nor migrated;
   agent memory starts fresh at upgrade. No legacy read path (consistent with decision 22's
   stance for pending blobs).
7. **Structural side effects:** R9 is resolved structurally (conversation id is a key component
   of every turn record); with R2's turn-scoped payload keys, deleting a conversation prefix
   co-GCs metadata, turns, and payloads under one lifecycle; design decision 11 is superseded
   (the store pair is now *pending store + unified conversation-turn store*) and the
   `TurnReviewStore` name is retired. Recorded in `docs/turn_result_design.md`, Amendments A1.

### R42. `next_actions` and `recommendations` still die at the agent boundary — the design fixes payloads but not the rest of the original bug class

The root cause (section 1 of the design) is "structured outputs collapse to text at the ReAct
tool boundary." The design fixes `artifacts`. But `CommandResponse` carries two more structured
fields — `next_actions: list[Action]` and `recommendations: list[Recommendation]` — and:

1. Verified: `_finalize_agent_output` (`workflow_execution_context.py:530-560`) builds a bare
   `CommandResponse(response=result_text)`; nothing populates `next_actions` or
   `recommendations` on the agent's answer. Section 5.4 *motivates* typing `answer` as
   `CommandResponse` partly by "the answer legitimately wants … `next_actions`, and
   `recommendations`" — but no mechanism in the design ever sets them. The motivation is
   circular as written.
2. The xray mapping (section 10.3) surfaces one `ResponseTuple` per **payload-bearing**
   `CommandOutput`. A command that returns `next_actions` or `recommendations` but no payload
   is **not** in the gallery and not in the answer — its structured output is dropped exactly
   the way payloads are dropped today. The original bug class survives the redesign for two of
   the three structured channels.

**Recommendation:** define the policy explicitly. Reasonable options: (a) the gallery includes
outputs bearing *any* structured content (payload, next_actions, recommendations, or
non-trivial artifacts), not just payloads; (b) `answer.next_actions`/`recommendations` are
aggregated from the final command's output (last-command-wins, documented); (c) explicitly
declare them dead on the agent path and remove the 5.4 motivation. Any of these is defensible;
silence is not — this is the same "which structured outputs, from which tool calls" question
the whole design exists to answer (section 2.4), asked only about payloads.

**RESOLVED 2026-06-11 (with Dhar).** New empirical finding first: **`next_actions` and
`recommendations` have zero producers and zero consumers** anywhere in the framework, bundled
examples, or tests (verified by grep; only the field declarations exist). Even xray — the one
application with real action buttons — ships them through `artifacts`
(`_response_mapping.py` packs `actions` there), a channel the design already preserves. And
post-R1/A7, the durable record keeps complete `CommandResponse`s for every execution, so these
fields are preserved in records automatically if they ever gain producers. Decisions:

1. **Gallery rule: payload-bearing only** (user decision, conservative). The live gallery
   filter stays as the design states. Accepted consequence, documented: a command returning
   buttons/recommendations without a payload appears in the durable record but not the live
   response — a drop scenario that is currently unreachable (zero producers) and recoverable
   from the record if it ever occurs.
2. **The agent answer's `next_actions`/`recommendations` stay empty**, and section 5.4's
   circular motivation is corrected: the answer is typed `CommandResponse` for its
   text+artifacts shape; buttons belong on gallery entries where provenance is clear.
3. **Gallery provenance** (raised during review — `ResponseTuple` has no command-identity
   slot, so per-entry provenance dies at xray's wire format even though `TurnResult`
   preserves `command_name`/`command_parameters`/`workflow_name`/`context` per entry):
   xray will initially embed provenance in the response text; extending `ResponseTuple` with
   structured provenance fields is **xray-repo scope and out of scope here** — documented as a
   note on the section 10.3 mapping. fastWorkflow's obligation is met by preserving full
   provenance in `TurnResult`.

Recorded in `docs/turn_result_design.md`, Amendments A9.

### R3. The breaking change is bigger than section 11 admits — it breaks the wire contract and every external workflow author

Section 11 lists internal accessors and `to_mcp_result()`, but:

- The bundled FastAPI server returns `command_output.model_dump()` **directly as the HTTP
  response body** in four places (`run_fastapi_mcp/__main__.py:798, 904, 1041, 1115`), covering
  `/invoke_agent`, the SSE final event of `/invoke_agent_stream`, `/invoke_assistant`, and
  `/perform_action`. `command_responses: list` is the JSON schema every HTTP and MCP client
  depends on. Collapsing the field and/or returning `TurnResult` changes the wire contract for
  all of them — undocumented in the design.
- `run_fastapi_mcp/utils.py:355-357` reads `output.command_responses[0].artifacts` for the
  awaiting-user check (see also R11).
- fastWorkflow is a published framework: every external user's hand-written `_commands/*.py`
  returning `CommandOutput(command_responses=[...])` breaks, not just bundled examples.

**Recommendation:** add a one-release compatibility shim — a `model_validator(mode="before")`
that accepts `command_responses=[x]`, maps it to `command_response`, and emits a
`DeprecationWarning` — plus an explicit, versioned statement of the HTTP schema change (or an
API version field in responses). If a hard break is the deliberate choice, the design must say
so and enumerate the client-facing schema delta. See also R46 (release sequencing).

### R4. Build-time generators and framework-internal commands are missing from the mechanical-changes list

Verified `command_responses` emission/construction/mutation sites **not** in section 11:

- `fastworkflow/build/command_file_template.py:134, 364`
- `fastworkflow/build/command_stub_generator.py:285`
- `fastworkflow/build/__main__.py:310`
- `fastworkflow/_workflows/command_metadata_extraction/_commands/` (~8 files). Note
  `wildcard.py` not only constructs but **mutates**:
  `command_output.command_responses[0].artifacts["command_handled"] = True` at lines 86 and 123.
  (Verified all internal sites are single-element constructions — the design's "provably
  unused" claim in section 4.1 does hold for `_workflows/` as well.)
- `fastworkflow/run/__main__.py` (CLI output printer)
- `fastworkflow/run_fastapi_mcp/utils.py:355-357`

If the build templates are missed, `fastworkflow build` generates broken commands the day the
change lands. The `refine` pipeline should be audited for the same reason.

### R5. The artifact serialization/offload contract is unspecified — section 8.3 is not implementable as written

`artifacts` is `dict[str, Any]`; current persistence punts with `json.dump(default=str)`. The
design says payloads are "replaced by a `PayloadStore` handle during serialization" but never
defines:

1. **What gets offloaded.** Every artifact value? Only known keys (`payload`)? A size threshold?
   (Recommended: any `str`/`bytes` value above a configurable threshold, e.g. 4 KB — offloading
   a 50-byte CSV adds a fetch round-trip for nothing.)
2. **How a handle is distinguished from a legitimate string artifact on read.** An envelope is
   required, e.g. `{"__fw_payload_ref__": handle, "content_type": "text/csv", "size": 12345}`.
   Without it, deserialization is ambiguous.
3. **What happens to non-JSON-serializable artifact values.** xray stashes `actions` objects in
   artifacts today; `default=str` mangles them irreversibly. Define: JSON-serializable values
   pass through; everything else is either rejected at the boundary (loud) or stringified with a
   marker (documented lossy).
4. **`command_parameters` lies about its type.** Declared `str` on `CommandOutput`
   (`fastworkflow/__init__.py:71`), but at runtime it carries a Pydantic model instance —
   `workflow_agent.py:103-106` calls `params.model_dump()` on it. The "light record" serializer
   must define how this field serializes (recommend: fix the declared type to `str | dict` and
   serialize via `model_dump()` explicitly).

This is the actual heart of option D and is currently a hand-wave.

**RESOLVED 2026-06-11 (with Dhar).** Decisions:

1. **Offload rule: size threshold.** Any `str`/`bytes` artifact value larger than a
   configurable threshold (default ~4 KB, env var named in the config inventory) is offloaded
   to the `PayloadStore` (turn-scoped per A8) and replaced by the envelope; smaller values
   stay inline. App-agnostic — no key-naming convention required.
2. **Envelope convention:** the offloaded value is replaced in place by
   `{"__fw_payload_ref__": <scoped handle>, "size": <bytes>, "content_type": <best-effort>}`.
   The marker key is reserved; readers detect it to distinguish references from literals.
3. **Non-serializable values: strict rejection.** Record serialization raises (a clear error
   naming the offending artifact key and command) on any non-JSON-serializable artifact
   value. Clarifications established during review: (a) `Action`/`Recommendation` are typed
   fields on `CommandResponse`, distinct from artifacts, and unused (R42) — unaffected;
   (b) the design doc's 0.3 "packs actions into artifacts" refers to xray's mapping shuttling
   its `ResponseTuple.actions` through the artifacts dict — and in the verified flow that
   value is `None` (0.2), so strict rejection breaks nothing existing; (c) the framework's
   own object-in-artifacts (the `cmd_parameters` Pydantic instance in the NLU handshake)
   never reaches serialization — consumed in-memory inside `invoke_command`; (d) the contract
   obligates any app packing objects into artifacts to serialize them first, in its own
   mapping code. Replaces today's silent `json.dump(..., default=str)` mangling.
4. **`command_parameters` honesty convention (stated, low-stakes):** the in-memory
   `CommandOutput` keeps the typed Pydantic instance (useful to consumers; the trace path
   already calls `model_dump()` on it); the declared type becomes honest
   (`str | BaseModel | None`-shaped); record serialization emits `model_dump()` as a dict.

Recorded in `docs/turn_result_design.md`, Amendments A10. Remaining store-level details
(hash algorithm, compression slot, atomic disk writes) stay with R21.

### R6. Failed executions and failed turns are invisible to review — the opposite of what observability needs

Three distinct holes:

1. **Failed tool calls never enter `command_outputs`.** Verified: `_execute_workflow_query`
   (`workflow_agent.py:76-98`) has no try/except around `invoke_command`. If the command raises,
   the proposed append never happens; DSPy surfaces an error observation to the agent, and the
   review record shows a gap where the failure occurred. Capture failed executions as
   `CommandOutput(success=False)` with error detail (exception type + message in artifacts).
2. **A turn that errors mid-loop produces no review record at all.** If the ReAct loop or an LLM
   call raises, nothing is finalized, nothing is written. The turns a developer most wants to
   inspect are precisely these. Wrap the agent loop; on the error path persist a partial
   `TurnResult` with `status="failed"` and the exception summary.
3. **Abandoned suspended turns are invisible.** A turn suspended on `ask_user` and never resumed
   sits in the pending store forever and never reaches review. Acceptable only if deliberate —
   decide, and either document it or add a sweep that converts stale pending blobs into
   `status="abandoned"` review records.

See R11 for the `status` field that unifies all three, and R39 for the fourth terminal state
(cancelled).

**RESOLVED 2026-06-10 (with Dhar).** Decisions (all three holes):

1. **Failed tool calls: capture with detail.** The capture point in `_execute_workflow_query`
   wraps `invoke_command` in try/except; on failure it appends a `CommandOutput` with
   `success=False` carrying the error type, message, and a truncated traceback in artifacts
   (traceback surfaced only in the developer projection, R30), then lets the error flow to the
   agent exactly as today (the ReAct loop at `utils/react.py:252` already formats it as an
   observation). Suspension signals pass through untouched — `AskUserSuspend` already
   subclasses `BaseException` for exactly this reason, and `CommandCancelledError` must be
   explicitly re-raised by the wrapper.
2. **Failed turns: record + re-raise.** `process_message` wraps the agent loop; on a fatal
   error it writes a partial turn record (`status=failed`, the executions completed so far,
   the exception summary) and then **re-raises**. Caller behavior is unchanged (HTTP 500s,
   retry semantics for transient LLM/infra errors); the evidence now exists for review. The
   never-raise alternative (always return `status=failed`) was rejected: it blurs
   transient-infra retry semantics and changes error handling in every caller.
3. **Abandoned turns: TTL + lazy filing.** A suspended turn older than a configurable TTL
   (order of days; named env var to be fixed in the config inventory) is *not* resumed when
   the channel is next touched: it is filed as `status=abandoned` (partial record; payload
   handles transfer as in A4 cancellation) and the incoming message starts a fresh turn. No
   background process is required for active channels; the pre-existing backlog item
   `fix-6b4` (TTL/reaper for orphaned suspended-session blobs) remains optional future work
   for channels that are never touched again, now scoped to that residue. Documented UX
   change: an answer arriving after the TTL starts a fresh turn instead of resuming a stale
   question.

This also closes R24's remaining open path: cancel-path orphans were dissolved by A4, and
abandoned-path payloads now transfer to the abandoned record at lazy-filing time. Recorded in
`docs/turn_result_design.md`, Amendments A5.

### R7. The design silently flips sensitive data from ephemeral to durably-persisted-by-default

Today a completed turn leaves zero durable trace of payloads; after this change every turn —
including payloads that are access-review CSVs, i.e. identity/entitlement data — persists
indefinitely, with retention declared out of scope. For the xray domain this is a **compliance
posture change**, not a deployment detail. (R37 softens this slightly — text traces already
persist in `ConversationStore` — but payloads do not, and payloads are the sensitive bulk.)

**Recommendation:** (a) make review persistence configurable (`TURN_REVIEW_ENABLED` or
equivalent; decide the default deliberately); (b) extend the section 7.8 contract beyond co-GC
to name encryption-at-rest and access-control expectations for both stores; (c) consider a
per-workflow or per-command opt-out for payload persistence (record persists, payload marked
"not retained") for classified data.

**RESOLVED 2026-06-11 (with Dhar) — no switches; retention is the compliance story.**

1. **No persistence switches at all.** Turn records *and* payloads always persist. The
   reviewer's opt-out recommendation was rejected: post-A1 the records are load-bearing
   (conversation system of record, agent-memory source), and the deliberate posture is that
   payload durability is the product, governed by retention rather than opt-out.
   Consequently the "review persistence disabled" fallback clauses in A4.2 and A5.4 are
   **dead and superseded** — cancel and abandon always record, payloads always transfer.
2. **Compliance = age-based retention, infra-executed (decision 19 affirmed).** Conversations
   older than a deployment-chosen age are removed by conversation-prefix deletes — the A8
   co-GC mechanism makes this a single-operation contract (disk: directory tree removal;
   Redis: prefix SCAN+DEL). The framework ships no retention code and no schedule; the
   documented contract is the deliverable. (The review's original concern — "the contract
   nobody implements" — is accepted as a deployment responsibility with eyes open.)
3. **Security expectations documented in the 7.8 contract (stated):** both stores are
   declared to hold PII/entitlement-grade data; deployments must provide encryption at rest
   and TLS to Redis with least-privilege access for the service identity; the framework's
   side is record-mediated payload access (A8) and never logging record contents.

Recorded in `docs/turn_result_design.md`, Amendments A12.

### R8. The `process_action` and MCP paths are not covered by the design

`WorkflowExecutionContext.process_action` exists (`workflow_execution_context.py:377`,
`_process_action` at `:664`) and is exposed via `/perform_action`; it appends to
`conversation_history` like the other paths (`:715-721`), so it is turn-shaped in every way
that matters. The design specifies the `TurnResult` contract only for `process_message`.
Undefined: does `process_action` return a `TurnResult`? Do action invocations produce review
records? What is their `user_message`? If excluded, say so explicitly — otherwise the review
timeline has invisible state mutations interleaved between recorded turns, which corrupts the
very audit trail the feature exists to provide. (MCP tool calls ride the same endpoints, so R3's
wire-schema resolution covers MCP transport; this finding is about record coverage.)

**RESOLVED 2026-06-11 (with Dhar) — full turns.** `process_action` returns a `TurnResult` and
writes a turn record like any other turn:

- `command_outputs = [the one execution]` (full provenance: command name, parameters,
  workflow, context); `answer = command_outputs[0].command_response` (same aliasing as the
  assistant path, R10's copy-on-serialize applies); `status` per A3; `success` = the command's
  own success per A6; **`user_message = ""`** (stated convention — no user words existed; the
  action is fully described by its command entry).
- Rationale recorded: (a) closes the audit hole — the state-*mutating* path can never be the
  unrecorded one; (b) preserves today's agent-memory behavior under A1's rebuild-from-records
  (verified: `_process_action` appends to `conversation_history` today,
  `workflow_execution_context.py:715` — without a record that note would vanish on restart);
  (c) MCP and `/perform_action` clients get the same wire shape as the other endpoints (one
  format, feeding R3).
- The opt-out flag variant was rejected to keep the audit guarantee unconditional. The
  *internal* `CommandExecutor.perform_action` (plumbing inside a turn) is unaffected.

Recorded in `docs/turn_result_design.md`, Amendments A11.

### R40. `TurnResult` has no turn-level success semantics

`CommandOutput.success` exists (and section 5.2 keeps it per-execution), but the proposed
`TurnResult` has neither a `success` field nor a defined derivation — while consumers need one:
`to_mcp_result()` sets `isError=not self.success` (`fastworkflow/__init__.py:96-99`); the CLI
and any HTTP client want a turn-level verdict. The agent mode makes this genuinely non-trivial:
an agent can deliver a perfectly good final answer after two failed and one successful tool
call. `all(outputs.success)` is wrong (penalizes recovered turns); `answer.success` is undefined
(nothing sets it on the agent path).

**Recommendation:** define explicitly — `TurnResult.status` (R11) carries lifecycle;
`answer.success` carries "did the turn produce a usable answer" (agent path: `True` unless the
loop failed; deterministic path: the command's success); per-execution success lives in the
gallery. Document that `success != all commands succeeded`, and update `to_mcp_result`'s
replacement accordingly.

**RESOLVED 2026-06-11 (with Dhar).** Decisions:

1. **`TurnResult.success = answer.success`.** Assistant mode: the executed command's own
   success flag (extraction errors and business-logic failures read `False`). Agent mode:
   `True` when the agent delivers an answer — crashes are already `status=failed` (A5). The
   recovery case (failed intermediate tool calls, correct final answer) reads `True`;
   per-execution successes remain individually visible in the gallery. `success != all
   commands succeeded` is documented explicitly.
2. **Iteration exhaustion is a quality signal:** when the ReAct loop hits `max_iters` and a
   best-effort answer is synthesized from the trajectory (today indistinguishable from a
   confident finish — `utils/react.py` runs "until finish, max_iters, or AskUserSuspend"),
   the turn is stamped `completed` + `success=False`. The answer still enters records and the
   completed-only memory projection; the loop must surface an *exhausted* flag to
   `_finalize_agent_output` (small, contained change).
3. **`success` goes on the wire:** serialized as a `computed_field` on `TurnResult` — visible
   in HTTP/SSE responses for the first time (today's property never serializes). The MCP
   result derives `isError = not success`. This pre-decides the `success` portion of R33;
   R33's remaining scope is the other predicates (`command_handled`, `command_aborted`,
   `not_what_i_meant`).

Recorded in `docs/turn_result_design.md`, Amendments A6.

### R41. The live response path is ambiguous — and one reading puts a put-then-get round trip on the hot path

Section 10.1 has the runner offload payloads and write the review record at persistence time;
section 10.3 says the xray mapping builds gallery `ResponseTuple`s "fetching payloads **lazily
from the `PayloadStore` by handle**." If the mapping operates on the serialized record, every
turn pays `PayloadStore.put` + immediate `get` for bytes the runner already holds in memory.
If it operates on the in-memory `TurnResult` (payloads inline), no fetch is needed — but then
the design's "lazily by handle" language is wrong for the live path, and the bundled server's
`model_dump()` responses (R3) will inline **all** gallery payloads into the JSON/SSE body,
ballooning agent-mode responses that today are text-only.

**Recommendation:** specify the rule: the live path maps from the in-memory `TurnResult`
(inline payloads, possibly size-capped per response), handles exist for the review reader only;
or deliberately serve the live path from handles for response-size control and accept the
round trip. Either is workable; the design currently implies both.

### R39. `/cancel_pending` exists — cancelled turns are a fourth terminal state the design doesn't model

Verified: `run_fastapi_mcp/__main__.py:1133-1160` — a user can abandon a suspended ask_user
turn; it calls `execution_context.cancel_pending()` and clears the pending store. Under the
redesign this path must decide: does a cancelled turn write a review record
(`status="cancelled"` with the partial event sequence — recommended, it is real history)? Who
cleans up the payloads offloaded at the suspend boundary (ties to R24 — they are now referenced
by nothing)? The design has no cancellation concept at all.

**RESOLVED 2026-06-10 (with Dhar).** Decisions:

1. **Cancelled turns are recorded, not shredded.** `/cancel_pending` and the A2
   auto-cancel-on-switch paths write a turn record with `status=cancelled` under the turn's
   *original* conversation, carrying the partial event sequence — commands executed so far plus
   the unanswered clarification question (exact event shape per R1, pending) — and the payload
   handles already offloaded at the suspend boundary. Payload ownership simply transfers from
   the pending blob to the cancelled record: no special cleanup, and R24's orphan concern is
   dissolved for the cancel path (the abandoned path remains tied to R6.3). Sequence, under the
   per-session lock: serialize partial `TurnResult` (status=cancelled) → write record → clear
   pending blob. Today's `cancel_pending()` (`workflow_execution_context.py:289`) only resets
   in-memory state — it gains the record write.
2. **R7 interplay:** if review persistence is disabled by deployment config, cancel falls back
   to today's shred — and only in that mode does an explicit suspend-payload delete step exist.
3. **Agent-memory projection rule (clarifies A1.2):** the rebuilt `dspy.History` is projected
   from `status=completed` records **only**. Cancelled, failed, and abandoned records are
   review/observability-only and never enter the agent's working memory — exactly matching
   today's behavior (a turn that never reached `_finalize_agent_output` never appends to
   history).

Recorded in `docs/turn_result_design.md`, Amendments A4.

---

## 2. Design tensions and consistency issues

### R9. `channel_id` is the wrong (or insufficient) review namespace

The bundled server has `/new_conversation` and `ConversationStore` with per-channel
conversation ids (`reserve_next_conversation_id`, `active_conversation_id` — see R37), so one
channel spans multiple conversations. `TurnReviewStore.list(channel_id)` interleaves turns
across conversations with no way to scope a review to one conversation. Include the
conversation id in the key (or as required record metadata with a filtered listing) — sourced
from `ConversationStore`, which already owns that counter. Also define the edge case: switching
conversations (`/new_conversation`, `/activate_conversation`) while a turn is suspended.

**RESOLVED 2026-06-10 (with Dhar).** Decisions:

1. **Namespace: resolved structurally by R37/A1.** The conversation id is a key component of
   every turn record (`fw:turn:{channel_id}:{conv_id}:{sortable-ts}-{uuid}`). Per-conversation
   review listing is a prefix scan; channel-wide observability scans remain possible via the
   channel prefix (ordering via the timestamp segment / stored ordinal per R18).
2. **Conversation id reservation is eager.** `active_conversation_id` is guaranteed to exist at
   session creation (restore-last or reserve-new), not reserved lazily on first save — required
   because R16 mints the turn key at logical-turn start. Deployments that never rotate
   conversations (e.g., xray today) get a single implicit conversation with zero management
   burden.
3. **Switch policy: auto-cancel.** Verified that neither `/new_conversation`
   (`__main__.py:1174`) nor `/activate_conversation` (`:1349`) checks `awaiting_user` or the
   pending store today — a suspended turn silently survives the switch and the next message
   resumes a clarification from the previous conversation, which under A1 would diverge agent
   memory from the store. Decision: when a turn is suspended, both endpoints first **cancel**
   it — record the partial turn under its *original* conversation with `status=cancelled` (R39
   semantics), clean up the pending blob and suspend-offloaded payloads (R24) — then proceed
   with the switch. The "carry across" status quo is rejected as incoherent with A1; "block
   with 409" rejected on UX grounds (clients shouldn't need a resolve-or-cancel modal).
4. **Implementation note:** both endpoints currently run without `runtime.lock`; the
   cancel-then-switch sequence must execute under the per-session lock (as `/cancel_pending`
   already does).

Cross-references: depends on R39's `cancelled` status (direction set by the R11 enum) and
R24's orphan cleanup. Recorded in `docs/turn_result_design.md`, Amendments A2.

### R38. `/post_feedback` conflicts with write-once review records

Verified: `run_fastapi_mcp/__main__.py:1276-1316` — users post per-turn feedback
(score and/or NL feedback) which mutates the **last in-memory conversation turn** and persists
via the conversation store. Feedback necessarily arrives *after* turn completion — i.e., after
the write-once review record is sealed. Three options: (a) make review records mutable for the
feedback field (violates write-once and its retry semantics); (b) a separate feedback keyspace
keyed by `turn_key` (recommended — preserves write-once, trivially joined at read time);
(c) feedback stays in the conversation store with a `turn_key` link (fine under R37's
consolidation). User feedback is first-order observability data — the design must give it a
home, and currently does not mention it.

### R43. The CLI / `command_output_queue` contract is undefined after the redesign

`process_message`'s *return value* changes to `TurnResult`, but the queue-based transport has
its own contract the design never updates: `_finalize_agent_output`, `_process_message`, and
`_process_action` all call `_maybe_enqueue_output(command_output)` (e.g.
`workflow_execution_context.py:563`), Topology A's `_ask_user_tool` enqueues mid-turn
clarification `CommandOutput`s directly, and the CLI (`run/__main__.py`) consumes the queue and
iterates `command_responses`. Define: does the queue carry per-event `CommandOutput`s plus a
terminal `TurnResult` (recommended — mirrors the live-then-final shape), or `TurnResult` only?
`ChatSession.keep_alive` and the CLI renderer change either way; neither appears in section 11.

### R44. Raw vs refined user message — capture both

`_refine_user_query` (`workflow_execution_context.py:728+`) rewrites the user's message using
conversation history before the agent sees it. `TurnResult.user_message` is undefined on this
axis. For review fidelity (and for debugging refinement itself — a classic "why did the agent
do that?" cause), store both the raw message and the refined query the agent actually ran on.

### R47. The conversation summary now has three homes

After the redesign as written, the same LLM-generated summary would live in:
`answer.artifacts["conversation_summary"]` (section 5.5), the review record's "searchable
metadata" (section 7.7), and the `conversation_history` turn record persisted by the bundled
server (R37). Pick one canonical location (the review record, under R37's consolidation) and
derive the rest.

### R10. Answer aliasing: a mutation hazard and a duplication bug waiting to happen

In the deterministic path, `answer == command_outputs[-1].command_response` — the *same object*.

1. **Copy-on-serialize is mandatory.** If offload is implemented as in-place replacement of the
   artifact value with a handle, the live `TurnResult` just returned to the caller is corrupted —
   through both aliases. The serializer must operate on a copy. Note: a blind `deepcopy` is the
   wrong tool — artifacts can hold live application objects (xray's `actions`) and
   `command_parameters` holds a Pydantic model (R5.4); the serializer should build a new dict
   tree selectively rather than deep-copying arbitrary object graphs.
2. **Headline/gallery duplication.** The xray mapping builds the headline from `answer` and one
   gallery `ResponseTuple` per payload-bearing output. In deterministic mode the single
   command_response is both — the same payload renders twice. The mapping needs an identity
   dedup rule (skip gallery entries whose `command_response is answer`).

### R11. Replace `awaiting_user` artifact-sniffing with a first-class `TurnResult.status`

Today's protocol is stringly-typed artifact inspection (`_awaiting_user_output` sets
`artifacts["awaiting_user"] = True` at `workflow_execution_context.py:518-520`; the runner and
`utils.py:355-357` sniff it back out). The redesign is the moment to fix this:

```python
class TurnStatus(str, Enum):
    COMPLETED = "completed"
    AWAITING_USER = "awaiting_user"
    FAILED = "failed"
    CANCELLED = "cancelled"   # /cancel_pending (R39)
    ABANDONED = "abandoned"   # written only by the stale-pending sweep, if built (R6.3)

class TurnResult(BaseModel):
    status: TurnStatus
    ...
```

This replaces `_output_is_awaiting_user`, gives R6/R39 their terminal states, gives the runner
an honest branch condition, and gives review records a queryable lifecycle field. The
predicates' magic-string protocol (`artifacts["command_name"] == "abort"`) is a pre-existing
wart, but `status` at the turn level removes the worst consumer of it.

**RESOLVED 2026-06-10 (with Dhar).** Decisions:

1. **Enum: all five values** — `completed`, `awaiting_user`, `failed`, `cancelled`,
   `abandoned`. `cancelled` is already required by R9's auto-cancel-on-switch; `abandoned` is
   reserved now (written only if the R6.3 stale-pending sweep is built) so readers never need
   a schema bump for it.
2. **The `awaiting_user` artifact is removed immediately** — `TurnResult.status` is the only
   turn-lifecycle signal from the release that introduces it. `_awaiting_user_output` stops
   stamping `artifacts["awaiting_user"]`; `_output_is_awaiting_user`-style sniffing
   (`utils.py:354-357`, xray runner) is deleted and the branch becomes
   `turn.status == AWAITING_USER`. This is contained because every consumer of the old signal
   is in-repo or user-owned and is already being rewritten for the `TurnResult` return type in
   the same release. (Note: this is a deliberate exception to dual-publish thinking — it does
   not by itself force a big-bang release in R46; it only requires the lifecycle-signal
   consumers to move in lockstep, which they must anyway.)
3. **No further cleanup (user decision)** — all four `CommandOutput` predicates (`success`,
   `command_handled`, `command_aborted`, `not_what_i_meant`) and the NLU-internal artifact
   protocol (`command_handled`, `command_name`, `cmd_parameters` — the wildcard→executor
   handshake) carry into the redesigned model unchanged, exactly as design section 5.2 has
   them (predicates simplify to single reads under the collapse). For the record: review
   verified that `command_aborted` and `not_what_i_meant` have zero consumers in the framework
   and its tests (definitions only, `__init__.py:78,86`); they are preserved deliberately as
   public API surface. The handshake protocol remains undocumented-internal; formalizing it is
   available as future work but is explicitly out of scope for this redesign.

Recorded in `docs/turn_result_design.md`, Amendments A3. Unblocks R6, R39, R40.

### R12. `nested_turn` issues

1. **It is the same speculative generality the design executes `command_responses` for** — zero
   producers exist today (design section 4.3 confirms). The roadmap justification (nested agents
   imminent) is acceptable, but ship it with a synthetic producer in tests or the recursive
   serializer is dead, untested code until nested agents land.
2. **Singular may be wrong.** A command that loops over a child workflow or fans out to several
   sub-agents produces multiple sub-turns. `nested_turns: list[TurnResult] = []` costs nothing
   now; changing the field shape later is another schema break.
3. **Nested suspension is undefined.** A nested agent calling `ask_user` suspends the whole
   stack; the pending blob must then serialize a partial parent containing a partial child.
   Define this before implementing, since `serialize_state` carries the partial `TurnResult`.
4. **Nested `user_message` semantics are undefined** — presumably the parent's delegation
   instruction; say so.
5. **Embedded-only (decision 20) makes parent records unboundedly large** and forces fetching
   the whole parent to address any nested node. Acceptable now; flag as a revisit-trigger once
   nested agents are real ("if p95 record size exceeds N KB, promote nested turns to references").

### R13. Assistant mode fragments one logical interaction into N unlinked records

In agent mode, `ask_user` makes a multi-exchange interaction one logical turn → one review
record. In assistant (`/`) mode, a parameter-extraction error returns to the user and the
correction arrives as a *new* `process_message` call → separate `TurnResult`s with no linkage.
A reviewer cannot reconstruct that record k+1 was a correction of record k. Consider a
`continuation_of: Optional[turn_key]` field stamped when a turn enters in the extraction-error
state. Related: section 5.5's claim that the deterministic path yields
`command_outputs == [the one CommandOutput]` is inaccurate whenever intent-clarification or
parameter-extraction commands run within the turn (section 3.1 says those *do* funnel through
`invoke_command` and are captured) — specify the actual invariant.

### R14. Decision 9 (text-only streaming) deserves a revisit footnote, not reversal

It was effectively decided before `PayloadStore` existed in the design. Once payloads are
content-addressed and offloaded at suspend boundaries anyway, putting a *handle* (not bytes) on
`CommandTraceEvent` is nearly free and would let a future UI hydrate gallery items mid-turn.
Leave the seam documented even if unused — observability's importance is rising (see §4).

### R15. "Searchable metadata" is an overpromise on Redis

Section 7.7 stores the conversation summary as "searchable metadata inside the value." Redis
cannot search values without an index (RediSearch module or a maintained secondary index);
disk requires scan-and-parse. Either downgrade the language to "stored metadata, filterable
client-side after listing" or specify the index strategy.

---

## 3. Store and durability refinements

### R16. Turn-key idempotency and ownership

Mint the turn key once at **logical-turn start** (first user message arrival), not at write
time. Otherwise a retried `put` after a timeout mints a second uuid → duplicate review records.
A stable turn id from turn start also lets traces and logs reference the turn mid-flight
(feeds §4). Assign ownership explicitly (recommend: WEC mints it when accumulation starts; the
runner reads it off the `TurnResult`). Enforce write-once where cheap: Redis `SET NX`, disk
`O_EXCL` — a failed write-once is a loud signal of a key-minting bug.

### R17. `list()` returning bare keys forces N+1 reads

Any review UI listing a channel does `list()` + N `get()`s. Add a metadata projection to the
listing (timestamp, status, command count, summary-if-present) and pagination/time-range
parameters. Cheap to do now, breaking to retrofit. On disk, shard channel directories by date
(`{channel}/{YYYY-MM-DD}/{turn_key}.json`) so long-lived channels don't accumulate thousands of
files per directory.

### R18. Ordinal: store it, don't derive it

Per-channel review writes are serialized under the session lock, so a per-channel monotonic
sequence number stored *in the record* is nearly free and immune to clock skew across pod
failover (wall-clock-sorted keys are not). Keep the timestamp in the key for range scans;
treat the stored ordinal as authoritative ordering. Also define **which** timestamp the key
carries — recommend logical-turn start (a turn that suspends for two days should sort where it
began; within a channel turns cannot interleave anyway, since the lock serializes them).

### R19. Review records need their own embedded schema version and an extension point

The design bumps `SCHEMA_VERSION` only in `session_state_store.py` (pending blobs, short-lived).
Review records are **long-lived**; every record must embed its own schema version from day one,
and the reader must dispatch on it. Add `metadata: dict[str, Any] = {}` to the serialized
record (and arguably to `TurnResult`) as a forward-compat extension point — token counts, user
ratings, trace ids will want a home (§4).

### R20. The key format is not filesystem-safe — and keys are attacker-influenced path components

ISO-8601 with colons (`2026-06-10T17:04:03.123456Z`) is an invalid NTFS filename; fastWorkflow
is a cross-platform pip package. Use a compact form (`20260610T170403.123456Z`) or encode, as
the pending store already does with `safe_id`. Additionally, `channel_id` and `turn_key` become
filesystem path components in the disk backends — sanitize both (path-traversal hardening), as
`DiskSessionStateStore` already does for channel ids.

### R21. `PayloadStore` API gaps

- `put(data: bytes | str)` — define the hashing rule for `str` (encode UTF-8 first) and the
  hash algorithm (recommend SHA-256, and prefix handles with it: `sha256:...` for agility).
- No content metadata: the handle is opaque bytes; a generic reviewer cannot know CSV from PNG
  from JSON. Either store `content_type`/`size` alongside the blob or require the envelope
  (R5.2) to carry them. (Today `payload_hint` lives in artifacts, which covers xray but not a
  generic review tool.)
- Consider transparent compression (CSV compresses 5–10×) — at minimum leave a
  `content_encoding` slot in the envelope so it can be added without a migration.
- Disk writes must be atomic (write to temp file + rename) to avoid concurrent readers seeing
  torn blobs — concurrent same-hash writes from different channels are expected, not exotic.
- **No `delete`/GC surface on the ABC.** If GC is infra's job, infra must reach behind the
  abstraction and the on-disk/Redis layout becomes a public contract — document the layout as
  stable, or add `delete(handle)` / `delete_prefix(...)` so co-GC tooling can stay in-tree.
  (Moot under R2 option (a), where prefix-delete is natural.)

### R22. Backend deployment caveats are unstated

- **Disk backends are single-node.** For the long-lived review store this bites much harder
  than for the pending store: records invisible across pods, lost on pod restart without a PVC.
  State plainly: disk = dev/single-node only for `TurnReviewStore`/`PayloadStore`. (Note
  `ConversationStore` already has exactly this problem — see R37.)
- **Resume across pods with disk `PayloadStore`** leaves the final review record referencing
  handles whose bytes live on the previous pod's disk (resume itself needs no fetch — by
  design — but the record is born half-dangling).
- **Redis `maxmemory` policy:** under `allkeys-lru`, review records and payloads are silently
  evicted; the reader-tolerates-missing rule covers payloads but not records. Recommend
  documenting `volatile-*` policies + explicit TTLs, or a dedicated logical database.
- **No per-channel quota:** a buggy or hostile session can grow the store without bound; note
  it as an infra expectation alongside retention.

### R23. Completion-sequence atomicity and failure semantics are undefined

The completion branch does: offload payloads → `put` review record → `clear` pending → return.
Decide and document:

- **Does a review-write failure fail the turn?** Recommend no — review is observability, not
  correctness; log loudly and return the `TurnResult`. But then accept (and monitor) silent
  gaps; an unmonitored best-effort write is how audit trails quietly rot.
- **Crash windows:** payloads written but record not → orphan payloads (benign if R16's stable
  key makes the retry idempotent, and R2's resolution defines who sweeps orphans).
- **Lock hold time:** the review write runs inside the per-session lock; a slow/down store with
  naive retries stalls the channel's next turn. Bound retries tightly or move the write to a
  post-lock best-effort step (the per-channel ordinal from R18 must then be assigned inside the
  lock even if the write happens after).

### R24. Suspend-time offload creates orphans the GC contract doesn't cover

Payloads are offloaded at the suspend boundary too (decision 16), referenced — at that point —
only by the **pending** blob. If the turn is abandoned (R6.3) or cancelled (R39), those
payloads are referenced by nothing the co-GC contract knows about. The lifecycle contract must
cover pending-blob references, or the abandoned/cancelled paths must adopt-or-delete them.

### R25. Serialization responsibility is unassigned

`TurnReviewStore.put` takes a `dict` — so *something else* runs the recursive offload-and-
serialize pass of section 8.3, but the design never names it. Define a `TurnSerializer`
component (owns: selective copy, envelope substitution, `PayloadStore` puts, schema version
stamp, recursive `nested_turn` descent) used identically by the suspend path
(`serialize_state`) and the completion path (runner). Two independent implementations of
"light partial" is how the formats drift.

### R26. Accumulator reset semantics need an explicit rule

Today `clear_action_log()` runs at agent-turn start (`workflow_execution_context.py:489`). The
new accumulator must: clear at **logical**-turn start (first message), **not** clear on resume
(`_resume_agent_message` continues accumulation), clear correctly on the assistant path, and
define behavior if `process_action` interleaves (R8). An off-by-one here double-counts a turn's
outputs into the next turn's record — easy bug, worth a stated invariant and a test.

### R27. Multi-pod / non-sticky routing caveat

The per-session lock is in-process. Two pods serving the same channel concurrently produce
distinct review keys (fine) but racing pending-store writes (pre-existing issue) and interleaved
ordinals (R18's counter needs the lock). One sentence in the design acknowledging the
sticky-session assumption would prevent a false sense of safety.

### R45. Payload accumulation changes the per-turn memory profile — consider eager offload

Today the agent path extracts text at the tool boundary and the `CommandOutput` (with its
payload) becomes garbage immediately. Under accumulate-all, **every** payload of the turn is
retained in RAM until turn completion — and chart payloads are *unbounded* (`df.write_csv(...)`
full-frame; only table payloads are capped by `MAXROWS_TABLEPAYLOAD`). Multiply by concurrent
sessions in one server process. Options:

- Accept and document (payloads are usually modest; suspects are charts on large frames).
- **Eager offload (D′):** at capture time, offload above-threshold payloads to `PayloadStore`
  and keep only the envelope in memory. Bounds memory, makes suspend serialization cheaper
  (payloads already offloaded), and the completion write touches only the light record. Cost:
  reintroduces per-command durable I/O for large payloads only — write-only, no fsync
  requirement, and amortizable on a background thread. This is a *better* latency trade than it
  looks because it moves bytes off the critical suspend path too. Worth an explicit decision
  either way; interacts with R41 (an eagerly-offloaded payload must still be servable inline on
  the live path, so keep the bytes until the response is built or fetch once).

---

## 4. Observability and roadmap alignment

### R28. The agent trajectory is already serialized at suspend — persist it at completion and the durable-trajectory roadmap item is half done

`react_blob` (the DSPy ReAct trajectory) is serialized into the pending blob for suspends
(`workflow_execution_context.py:270`) and **discarded at completion** — only the LLM-derived
`conversation_summary` survives. Adding an optional trajectory slot to the review record
(offloaded via `PayloadStore` if large — it is just another blob) is nearly free given
machinery this design already builds, and it directly serves the "externalize agent trajectory
into durable memory" roadmap item. The R1 event-sequence shape gives trajectory entries a
natural home. Recommend at minimum reserving the field now (`trajectory_ref: Optional[str]`).

### R29. No timestamps, durations, or cost anywhere in the model

For observability, capture at the choke point (trivially cheap): per-`CommandOutput`
`started_at`/`duration_ms`; per-turn `started_at`/`completed_at`/suspended-interval count;
and reserve slots (in R19's `metadata`) for token usage and model invocations
(`LLM_PARAM_EXTRACTION`, `LLM_AGENT`, etc. are distinct cost centers worth attributing).
Latency-per-command is the first question every developer asks of a trace.

### R30. Two audiences, one record: define projections

The user's observability need and the developer's differ. Review records will contain
framework-internal executions (the `command_metadata_extraction` clarification commands —
which section 3.1 confirms *are* captured), raw parameters, and error detail. Define a
**user-facing projection** (app commands + exchanges + answer) vs the **developer-facing full
record** — cheapest as a read-time filter on `workflow_name`/an `internal: bool` flag stamped
per `CommandOutput`. Deciding this now shapes whether the flag exists in the schema. (The
agent's consultations of the string-returning helper tools — `what_can_i_do`,
`intent_misunderstood` — are visible only in the trajectory, R28; note that in the projection
docs.)

### R31. Align the serialized schema with OTel GenAI semantic conventions

The tree this design builds (turn = root span, command execution = child span, `nested_turn` =
sub-span, exchanges = events) is isomorphic to a trace. Keeping a documented mapping to
OpenTelemetry GenAI conventions — or simply carrying `trace_id`/`span_id` fields in `metadata` —
means the future observability tool can start as an exporter to existing viewers
(Langfuse/Phoenix/Jaeger) instead of a from-scratch UI. A schema-discipline decision to make
now, cheaply, even with tooling out of scope.

### R32. The read side has no API or access-control model

Nothing serves review records to a UI. The bundled server's JWT binds a caller to a
`channel_id`, which naturally scopes end-user reads to their own channel — but the developer/
admin cross-channel view (the actual observability tool) needs an admin path, and payload
fetches must be authorized **through the record**, never by bare handle (see R2's hash-oracle
risk). Out of scope to build, but the design should state the rule: *payload access is always
mediated by access to the referencing turn record.* (Note `ConversationStore` already has an
admin dump endpoint — `/dump_conversations` — which under R37's consolidation becomes a
template for the review read side.)

### R33. Pydantic properties are absent from `model_dump()`

`success`, `command_aborted`, etc. are properties and therefore never appear on the wire today
(verified against `fastworkflow/__init__.py:73-87` + the `model_dump()` returns in R3). If any
client is expected to read `success` off a serialized `TurnResult`/`CommandOutput`, declare
them as `@computed_field` — and note it as a deliberate wire-schema addition. (Interacts with
R40: turn-level success must exist before it can be serialized.)

---

## 5. Editorial and process

### R34. The design document's section order is scrambled

Sections 10–13 — including "Open questions: **none remaining**" — physically sit between
sections 1.4 and 1.5, before half the design they conclude. Fix before this becomes the
implementation reference. Separately, given this review, section 13 should be rewritten to an
actual open-questions list (R1, R2, R5, R8, R37, R40, R41, R42 at minimum are unresolved design
questions, not implementation details).

### R35. Test-plan implications are unstated

The project's testing philosophy is integration tests against real workflows; this change
touches every example and the build pipeline. Plan for: store round-trip tests (disk + redis),
suspend → resume → complete with offload at both boundaries, cancel and abandon paths (R39,
R6.3), nested-turn serialization with a synthetic producer (R12.1), failed-execution capture
(R6.1), accumulator reset invariants (R26), ask_user interleaving order (R1), conversation
store consolidation (R37), and the compat shim's deprecation path (R3).

### R36. Deployment coupling with xray

The framework change and the xray mapping change (`_response_mapping.py`, runner) must deploy
in lockstep; an old runner against a new framework fails on the `process_message` return type
immediately (good — loud), but a new runner against an old framework fails subtly. Pin the
fastworkflow version in xray and state the lockstep requirement in the rollout notes.

### R46. Release sequencing: the design couples three independently-shippable changes into one big-bang break

The design bundles: (1) the **bug fix** (capture tool outputs, return them — needs `TurnResult`
and the accumulator), (2) the **new feature** (durable review stores), and (3) the **model
cleanup** (`command_responses` collapse). Only (3) is intrinsically breaking; (1) and (2) can
ship additively. For a published framework, a staged train is materially safer:

- **Stage 1 (minor release):** add `TurnResult`, the accumulator, and a new
  `process_turn()`/changed `process_message` behind the compat shim (R3); `CommandOutput` keeps
  `command_responses` with a deprecation warning. This alone fixes the xray payload bug.
- **Stage 2 (minor release):** `TurnReviewStore`/`PayloadStore` + the runner/bundled-server
  persistence wiring + `ConversationStore` consolidation (R37), all behind config (R7).
- **Stage 3 (major release):** remove `command_responses`, retire `action_log`, drop the shim.

Also missing from the deployment story: **in-flight suspended sessions at upgrade time.**
Decision 22 punts old-schema migration to "a separate tool if deemed necessary," but says
nothing about runtime behavior when `apply_serialized_state` meets an old blob. Specify
graceful degradation — on `SCHEMA_VERSION` mismatch, treat pending state as expired, clear it,
and surface "your previous question expired, please re-ask" — rather than an exception on the
user's first post-upgrade message.

### R48. Export the new types

`CommandOutput`/`CommandResponse` are re-exported from `fastworkflow/__init__.py` (the public
surface command authors import). `TurnResult`, `TurnStatus`, and the event types must join
them; trivial, but it is the kind of thing section 11 exists to list.

---

## Suggested resolution order

1. **Resolve the architecture question first: R37.** Whether `TurnReviewStore` subsumes
   `ConversationStore`'s turn persistence (recommended) or runs beside it changes what gets
   built in every later step — including where feedback (R38) and conversation ids (R9) live.
2. Rewrite the contradictory decisions: R1 (event-sequence turn record, exchanges captured at
   `_post_ask_user_response`), R2 (turn-scoped payload keys), R42 (structured-output policy for
   actions/recommendations), R11+R6+R39 (`TurnResult.status` + failure/cancel capture), R40
   (turn-level success). These change the core types — everything else layers on them.
3. Specify the serialization contract: R5 (envelope, thresholds, `command_parameters`), R25
   (`TurnSerializer`), R19 (record schema version + `metadata`), R41 (live path vs handles),
   R45 (eager vs boundary offload).
4. Close the scope holes: R3 (shim + wire-schema statement), R4 (generators), R8
   (`process_action`), R43 (queue/CLI contract), R7 (persistence opt-in + security contract),
   R9 (conversation id), R44 (raw + refined message).
5. Fold in the mechanical refinements (R10, R13–R27, R33, R47, R48) as implementation notes.
6. Reserve the observability seams (R28–R32) — fields and rules only, no tooling.
7. Plan the release train (R46) and the test matrix (R35) before writing code.

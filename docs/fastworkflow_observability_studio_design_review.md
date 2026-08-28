# Adversarial Review — fastWorkflow Observability & Studio Design (v1)

Target: `docs/fastworkflow_observability_studio_design.md` (draft of 2026-08-25)
Process: five parallel adversarial review agents (lenses: concurrency/write-path,
data model/schema, architecture/consolidation, security/privacy, operability/UX),
findings verified against v3.1.2 (main @ 27e8d91) with load-bearing evidence
re-checked by the coordinating reviewer. 31 raw findings deduplicated to 28.
Tracking: one bd child per finding under fix-kw7.1. Resolution protocol: fix-vof
model — each finding gets a written resolution (accept/amend/reject with
rationale) recorded in the Resolution Log below; the design doc is amended per
accepted finding.

Severity index: **5 critical** (R1–R5), **20 major** (R6–R25), **3 minor** (R26–R28).

---

## Critical

### R1 — turns/conversations are keyed on identity values nothing populates [data-model + architecture]
`TurnResult` is constructed without `conversation_id`/`ordinal`
(`workflow_execution_context.py:835-845`; fields default None in `turn.py:247-248`)
and carries no `channel_id` at all; conversation ids are minted per-channel by the
**old** store in the FastAPI layer *after* `work_fn` returns
(`run_fastapi_mcp/utils.py:1391`, `conversation_store.py:141`), and the CLI has
neither channels nor conversations. Yet §3.2 makes `channel_id NOT NULL` a key of
`turns` and §3.1 claims "no new plumbing layers".
**Failure:** every turn row has NULL conversation linkage (or the insert fails on
the CLI path); the `conversations` table is never written; Studio's foundational
conversation→turn drill-down returns nothing — discovered at Phase 4. Under
write-once, records are *permanently* unattributable even after the old store
mints the id.
**Resolution direction:** explicit identity plumbing — embedder passes
channel_id/conversation_id into WEC at bind/begin-turn (reservation moved ahead of
the turn), WEC stamps them plus store-assigned ordinal onto `TurnResult`; the new
DB becomes the sole id-minting authority from Phase A; define the CLI mapping (see
R17).

### R2 — write-once per turn_key contradicts the ask_user suspend/resume lifecycle [data-model + concurrency]
One turn_key legally yields multiple `TurnResult`s: `_build_turn_result` runs on
every `process_turn` call, emitting `AWAITING_USER` (completed_at=None) then later
`COMPLETED`/`FAILED` under the same `_turn_key`
(`workflow_execution_context.py:184-191, 808-845`; key survives rehydration
`:558-579`; `turn.py:240-242` "one logical turn = one key = one record, across any
number of suspensions"). The prior spec kept suspend partials in the *pending*
store and only terminal writes in the turn store
(`turn_result_design_final.md:339`); §3.2 collapses both into one write-once row
whose collision rule only defines identical-content retries.
**Failure:** every ask_user turn is permanently stored as an unanswered
`awaiting_user` row — the terminal record is content-verify-rejected and, per
never-fail-the-turn, silently dropped. Multi-round ask_user turns lose N records.
**Resolution direction:** define the row lifecycle: INSERT at first emission, then
a guarded status-transition upsert (awaiting_user → terminal, incl. ABANDONED
filing); write-once applies only to rows already terminal.

### R3 — Phase B memory rebuild cannot be fed from the proposed turn records [architecture]
History restore consumes exactly three per-turn fields — `"conversation summary"`
(LLM-generated at finalize), `"conversation_traces"`, `"feedback"`
(`conversation_history_io.py:10-32`; appended `workflow_execution_context.py:656-670`;
restored `run_fastapi_mcp/utils.py:523-529`, `__main__.py:1872-1873`) — none of
which `TurnResult` carries or the §3.2 schema stores. Feedback is additionally
*mutable* after the fact (`conversation_store.update_last_conversation_turn:541`,
`utils.py:1421-1451`), which write-once forbids; the prior spec models it as a
separate mutable record (`turn_result_design_final.md:304,371`).
**Failure:** Phase B ships, per-channel DBs deleted, server restarts → restored
history lacks summaries/traces, `_refine_user_query` degrades, and all recorded
user feedback is gone — after the old DBs were already retired.
**Resolution direction:** amend the schema before Phase A: columns (or sink-written
fields) for conversation_summary/conversation_traces, plus a separate mutable
`feedback(turn_key)` table per spec §5; or explicitly define `TurnResult.metadata`
as the carrier and drop "verbatim" reuse of `restore_history_from_turns`.

### R4 — on-by-default capture persists end-user PII to world-readable files [security]
`FW_OBSERVABILITY=1` converts every CLI run and library embedder into a persistent
recorder of user messages, post-extraction parameters, response text, and
artifacts. Today the CLI persists nothing; parameters are PII in the flagship
example (`retail_workflow/_commands/find_user_id_by_email.py:22-24`, name+zip,
address, `payment_method_id`). `state_paths.py` creates every directory with
`os.makedirs(path, exist_ok=True)` — no mode; DB lands 0644 under umask 022
(contrast `jwt_manager.py:94` chmod 0600 for its private key). Spec §13 `[A12]`
makes encryption/least-privilege a *deployment* obligation predicated on choosing
to run a store; this design removes the choosing.
**Failure:** production embedder on a shared box, nobody sets an env var → every
customer email/address/payment id lands in a world-readable
`~/.local/state/.../observability.sqlite3`, retained 30 days, swept into $HOME
backups.
**Resolution direction:** 0700 dirs / 0600 DB explicitly; default ON only when the
operator invoked the framework directly (`fastworkflow run`/`studio`/server),
opt-in for library embedders; document the delta.

### R5 — the Studio read layer is an unauthenticated cross-channel read primitive [security]
The per-workflow DB spans all channels; §3.4's posture is "localhost-only" with
auth punted to §8 Q5. This contradicts the prior spec's record-first authorization
("JWT channel-bound... no bare-handle endpoints, ever",
`turn_result_design_final.md:430-433` `[A39][A41]`) — binding per this doc's own
conflict rule, with no amendment offered. Today every server conversation read is
channel-scoped by JWT (`run_fastapi_mcp/utils.py:174-212`, `jwt_manager.py:156`).
§4's absorbed `GET /turns/{turn_key}` as written *is* the forbidden bare-handle
endpoint.
**Failure:** a deployment uses per-customer JWT channels; a developer runs
`fastworkflow studio` on the host; every channel's messages, PII parameters, and
artifacts are served to anything that can reach the port, no token — the JWT
layer bypassed by construction.
**Resolution direction:** amend explicitly: per-launch random bearer token
(Jupyter pattern) minimum, or the server's admin-scoped JWT path; specify that
`GET /turns/{turn_key}` authorizes the record's channel against the caller's
token.

## Major

### R6 — root spans opened in process A cannot be closed in process B [concurrency]
Suspended turns rehydrate into fresh runtimes, including across server restarts
(`run_fastapi_mcp/utils.py:566`); the serialized accumulator carries no span ids
(`workflow_execution_context.py:371-386`). §3.1 emits `fw.turn` at begin/finalize;
§3.2 has no cross-process close protocol. **Failure:** resume after restart mints
a second root span for the same trace; one stays open forever; human-wait
attribution lands wrong. **Direction:** deterministic span ids
(hash(turn_key+name+attempt)) with idempotent end_ns upsert, or per-attempt
segment spans with the logical envelope computed at read time.

### R7 — CLI exit kills the writer thread with the just-debugged turn still queued [concurrency]
The CLI exit path (`run/__main__.py:218-227` `//exit` → break) calls no
close/flush; house precedent is daemon threads (`chat_session.py:26-30`). §3.2
specifies neither daemonness nor a flush protocol. **Failure:** developer runs one
turn to reproduce a bug, exits, opens Studio — the last turn (the one under
investigation) was enqueued milliseconds before the daemon writer died.
**Direction:** daemon writer + explicit `close()` (sentinel, bounded join, final
commit) wired into CLI exit and `atexit`.

### R8 — cross-process writer contention is unspecified: busy_timeout, transactions, NFS [concurrency]
CLI + server + train + Studio test mode are separate processes, each with "the"
single writer thread, against one DB. No busy_timeout (house precedent sets
`timeout=30.0`, `kvstore.py:24-35`), no transaction discipline, no same-host
constraint (`FASTWORKFLOW_STATE_ROOT` can be NFS, where multi-process WAL is
documented not to work). **Failure:** SQLITE_BUSY on commit + alarm-and-continue =
permanent record loss that write-once forbids reconciling; NFS → corruption.
**Direction:** timeout=30.0, BEGIN IMMEDIATE + short batched transactions,
requeue-with-bounded-retries for turn records (distinct from span drop policy);
document local-filesystem requirement.

### R9 — fix-85g.9 cannot be "served from the store" as designed: key split + eventual consistency [concurrency]
The 202 handle is the *execution* turn_key, minted per TurnExecution
(`run_fastapi_mcp/turns.py:311`), explicitly not the logical
`TurnOutput.turn_key` (`turns.py:786-793`); the DB is keyed on the logical key.
The engine's invariant is persist-before-DONE (`turns.py:22-24, 575-579`); an
async bounded-queue writer makes the DB lag DONE. **Failure:** poll
`GET /turns/{E1}` → 404 forever; poll logical K → 404 for a turn the server just
reported complete (record behind 9k spans in queue; stdlib Queue can't implement
drop-spans-first eviction). **Direction:** record the execution↔logical mapping
(or return K in the 202); registry-first read with DB fallback, or a synchronous
carve-out writing the turn record inside the persist-before-DONE section.

### R10 — no size policy: record_json duplicates everything, span attributes uncapped [data-model + operability]
`record_json` ("full serialized TurnResult") necessarily embeds every artifact
value (`CommandResponse.artifacts` inside `command_outputs`), defeating the
artifacts table's 256 KiB cap; the same response text is stored again in
`fw.command.execute` attributes (uncapped `attributes TEXT`), and in Phase A the
old store writes it a third time. Nothing bounds `CommandResponse.response` (the
framework bounds tracebacks to 4 KB precisely because responses are unbounded;
`distillation.py:290-292` already truncates observations at 500 chars).
Serialization happens on the hot path pre-enqueue. **Failure:** one 25-iteration
turn with a 20 MB query dump serializes it repeatedly on the turn thread
(contradicting "never blocks"), writes ~40–60 MB; ~20 such turns hit the 1 GiB
cap and pruning eats weeks of history; artifacts marked "too big to retain"
remain fully readable in the raw record view. **Direction:** one offload/truncate
mechanism at serialization time (placeholder/ref envelopes *inside* record_json,
artifacts table as sole value holder, per prior spec `[A10]`); emit-time
attribute caps; truncation lossy-and-counted.

### R11 — no schema version or migration mechanism [data-model + operability]
`record_version` versions rows only; no `PRAGMA user_version`/meta table, no
migration slice in §7 — while Phase B already needs conversation-metadata columns
the spec mandates (`next_ordinal`, `status`; `turn_result_design_final.md` §5
`[X2]`), and the state root is shared across fastworkflow versions
(a 3.2 CLI and 3.1 server on one DB). Spec §6 requires strict reader version
dispatch; the Studio reads typed columns directly. **Failure:** ALTER TABLE
against unversioned user DBs; older reader misrenders newer rows with no error.
**Direction:** `PRAGMA user_version` at creation; opener rule (reader refuses
newer; writer migrates forward); fold spec-mandated conversation columns in now.

### R12 — the black box has no lifecycle owner: pruning, VACUUM, WAL growth [operability + concurrency]
§5 names a "retention job" no component runs (§8 Q6 punts, but §7 ships the store
with caps active before Q6 is answered). DELETE never shrinks a SQLite file
without `auto_vacuum` set **at creation** (the kvstore template sets only
WAL+NORMAL); long-lived Studio readers pin the WAL end mark so the `-wal` sidecar
grows uncounted past the cap. Short-lived CLI processes may never fire pruning at
all. **Failure:** file stays at high-water mark, cap check triggers pruning on
every write, history silently eaten while disk usage never drops; or unbounded
growth for CLI-only users. **Direction:** `auto_vacuum=INCREMENTAL` at creation +
periodic `incremental_vacuum`; opportunistic bounded prune at sink startup +
`fastworkflow studio --prune`; per-request read connections in the Studio HTTP
layer; count `-wal` toward the cap.

### R13 — the three queue invariants are mutually unsatisfiable, and the alarm sink doesn't exist [operability]
"Never blocks" + bounded queue + "drop spans first, never turn records" cannot
all hold when the writer wedges: at overflow, a turn-record `put` must block,
drop, or evict queued spans — stdlib `queue.Queue` can do none of these
selectively. The only failure signal is `fw_record_write_failures_total`, a
counter in the unbuilt `metrics.py`. **Failure:** disk-full (the §6 test case!)
forces the implementer to silently break a stated invariant; drops are invisible
to CLI users. **Direction:** two queues (small dedicated turn-record queue,
bounded-timeout put, then drop-with-log; spans droppable), plus a user-visible
surface for writer health (self-row in the DB / Studio banner), replacing the
unbuilt-counter reference.

### R14 — fire-and-forget durability is telemetry semantics applied to the future system of record [architecture]
§3.2's "a write failure never fails a turn" is right for spans but becomes silent
conversation-history corruption at Phase B, when these rows are the only store.
Today's store persists synchronously before DONE (`turns.py:575-579` and failures
propagate); prior spec §1 requires "one **durable** record write per turn".
**Failure:** brief disk-full drops three turn records, turns return 200, alarm
counter unwatched; post-Phase-B restore is missing turns; `GET /turns` 404s for
completed turns. **Direction:** split durability classes at Phase B — turn
records + feedback synchronous-or-acked in the turn path; fire-and-forget
re-scoped to spans/artifacts. State as a Phase-B gating precondition.

### R15 — nothing writes conversations.topic/summary in Phase A: Studio navigation is empty until Phase 7 [architecture]
`TraceSink` has only `emit_span`/`emit_turn_record`; lazy LLM labeling lives in
the FastAPI layer and writes only the old store (`turns.py:451,603`,
`utils.py:1537-1614`, `conversation_store.py:346,495,519`) — and reads the
per-turn summaries that (R3) the new records don't carry. **Failure:** Phase-4
Studio shows an empty/unlabeled conversation list while the real labels sit in
the old per-channel DBs. **Direction:** add a conversation-upsert/label method to
the sink callable from the labeling hook, or have Phase-4 Studio read conversation
metadata from the old store during dual-write — and say which in the doc.

### R16 — the 30-day retention default silently deletes user conversation history after Phase B [architecture]
Today conversations persist indefinitely; post-Phase-B the same data lives in a
DB whose default retention is 30 days / 1 GiB with "oldest conversations pruned".
**Failure:** a user returns on day 31; their history was pruned by a default
nobody changed — a behavioral regression introduced by consolidation.
**Direction:** at Phase B, exempt conversation rows + turn records from
retention (prune spans/artifacts only) or flip conversation retention to
infinite-unless-opted-in; record as a §7 Phase-B gate.

### R17 — the data model has no answer for CLI turns [architecture]
The CLI has no channel or conversation identity (`chat_session.py` — zero channel
references; `ConversationStore` is FastAPI-only, `utils.py:505`; `//new` clears
in-memory history only), yet `channel_id` is NOT NULL and Studio navigation is
conversation-first, while Phase B retires the CLI's only artifact
(`action.jsonl`). **Failure:** CLI turns either violate NOT NULL or hide under a
fabricated undocumented channel; the doc's own motivating debugging surface
becomes *less* inspectable. **Direction:** define the CLI identity mapping (e.g.
synthetic `cli:<session-start>` channel, conversation minted per `//new`) and a
turns-first Studio view for conversation-less rows.

### R18 — "localhost-only" is not an auth boundary [security]
No token, Host/Origin validation, or CSRF story. DNS rebinding reaches a
localhost HTTP server from any web page the developer visits (browser fetches
same-"origin" after TTL-0 rebind; server never checks Host); WSL2
localhostForwarding exposes the port to the Windows side; any local user on a
shared box can curl it — making R4's file permissions moot via HTTP re-export.
**Direction:** per-launch random bearer token embedded in the printed URL +
strict Host/Origin allowlist, stated in §3.4 as invariants.

### R19 — Studio-spawned server inherits 0.0.0.0, CORS *, unsigned-JWT mode, and loaded secrets [security]
`--host` defaults to `0.0.0.0` (`run_fastapi_mcp/__main__.py:606`),
`allow_origins=["*"]` (`:664`), passwords env loaded (`:352-353`), and an
unsigned-JWT trusted-network mode exists (`jwt_manager.py` warning). §3.4's
"studio can start it" specifies none of this. **Failure:** one UI click binds a
command-execution server with live API keys to the coffee-shop LAN. **Direction:**
Studio-spawned servers always `--host 127.0.0.1`, CORS pinned to the Studio
origin, refuse unsigned-JWT mode without an explicit flag; wider bind is a CLI
decision, never a UI one.

### R20 — tracebacks and exception reprs are persisted verbatim with no redaction hook [security]
Failed tool calls capture `traceback.format_exc()[:4000]`, `repr(e)`, `str(e)`
into artifacts/responses/trajectory (`workflow_agent.py:197-209`,
`utils/react.py:406-412`, `command_executor.py:217`) — all landing in durable
`record_json`/`attributes`. LiteLLM auth errors embed provider request context.
**Failure:** a misconfigured key's AuthenticationError body persists 30 days,
served by R5's unauthenticated API. **Direction:** sink-boundary redaction pass
(known key shapes + values of loaded `*_API_KEY`/`*_TOKEN` env vars); traceback
persistence behind a debug flag.

### R21 — consolidation destroys the per-channel erasure primitive [security]
Today erasure is `rm conversations/<channel_id>.sqlite3` (the spec's stated
retention mechanism). The new schema scatters a subject across four tables —
`spans` has no channel_id (join via turns required) — plus WAL and freelist
pages, with only age-based pruning and no delete operation; Phase B removes the
per-channel files. **Failure:** GDPR-style erasure becomes undocumented manual
SQL + VACUUM + checkpoint; miss a step and "erased" data remains recoverable.
**Direction:** add channel_id to spans/artifacts (or spec the join), plus a
first-class forget-channel operation (store API + `studio --forget-channel`)
that deletes across tables and runs `wal_checkpoint(TRUNCATE)` + incremental
vacuum.

### R22 — rendered artifacts and record text are attacker-supplied; no XSS/CSP/SQL contract [security]
"Rendered artifacts" + same-origin read API + arbitrary developer/user-supplied
artifact values (`turn.py:32-56,110-120`) with no output-encoding, sandbox, CSP,
or parameterized-query requirement anywhere in §3.4. **Failure:** an end-user
message or HTML artifact stores XSS; a developer opens Studio; the script walks
the same-origin API across every channel and posts the dump out (the doc bans
CDN *dependencies*, not outbound fetches). **Direction:** record-derived text
rendered as text; HTML-ish artifacts only in sandboxed iframes with
`default-src 'none'`; restrictive CSP (`connect-src 'self'`) on the SPA;
parameterized queries only.

### R23 — the SPA has no packaging or build story, and `studio` breaks the server-extra contract [operability]
`pyproject.toml:18-22` includes exactly three extra files; no JS toolchain in
repo; Poetry silently omits gitignored build outputs from wheels; `cli.py:397-401`
documents the FastAPI server as "the only feature" needing server-only deps —
but test mode needs that server and even debug mode needs an HTTP layer of
unspecified dependency posture. **Failure:** CI wheel ships without SPA assets
(no error); `pip install fastworkflow` + `fastworkflow studio` = 404 shell or
ImportError — the exact rot §5 cites the graveyard for. **Direction:** packaging
subsection: committed (or CI-built with wheel-content assertion test) bundle,
`include` entries, stdlib-served debug mode in the base install, `[server]`
extra gating test mode only.

### R24 — test mode cannot reproduce several things the CLI does [operability]
`/`-prefix is stripped on `/invoke_agent`/`/invoke_agent_stream`
(`__main__.py:1190,1286`; deterministic works only via non-streaming
`/invoke_assistant:1428-1434`); startup command/action/context are process-level
server args (`cli.py:309-311`), not per-`/initialize` inputs; insights
distillation is CLI/Topology-A only (`cli.py:286`,
`workflow_execution_context.py:101-102`). **Failure:** a workflow needing a
startup action can't be tested per-variation without relaunching the server;
developers fall back to the CLI and Phase 5 goes unused. **Direction:** a CLI-
parity table in §3.4; additive per-`/initialize` startup fields; declare
distillation out of Studio scope explicitly.

### R25 — retiring action.jsonl breaks insights distillation and shipped diagnostics tooling [operability]
`distillation.py:33,55-56,566-573` synchronously reads/deletes `action.jsonl` per
agent turn (mirror enabled unconditionally, `chat_session.py:120`); the
diagnostics skill's `trace_turn.py` reads it too. Reading spans instead races the
async writer. **Failure:** post-retirement, distillation's `os.path.exists` guard
makes `actions` silently `[]` — insights extracted from empty trajectories.
**Direction:** port distillation to the in-process `ctx.action_log` (no race, no
file) as a Phase-B precondition; update `trace_turn.py`; name both in the Phase-B
slice.

## Minor

### R26 — "OTel-shaped" ids/kinds/timestamps don't conform; the promised export script needs a translation layer [data-model]
turn_key is `YYYYMMDDTHHMMSS.ffffffZ-<12hex>` (`turn.py:91-107`), not a 16-byte
trace_id; `kind` values (`llm`, `human_wait`) aren't OTel SpanKind; `turns`
timestamps TEXT vs `spans` INTEGER ns. **Direction:** store a derived conformant
trace_id (first 16 bytes of SHA-256(turn_key)) and map kinds at write time, or
amend the claim to "OTel-inspired" with a documented translation table.

### R27 — no secondary indexes; command_name isn't queryable at all [data-model]
Zero `CREATE INDEX` beyond PKs; `spans.trace_id`, `turns(channel,conv,ordinal)`,
`turns.status`, `artifacts.turn_key` unindexed; `command_name` lives only inside
attributes JSON, so filter-by-command can't be indexed as designed.
**Direction:** add indexes for the enumerated query set; promote
command_name/context to a queryable surface.

### R28 — the chosen fw.agent.tool_call emit sites sit inside trace-queue guards [architecture]
The doc's emission sites are inside `if chat_session_obj.command_trace_queue is
not None:` (`workflow_agent.py:170,251`) — queue-less embedders would reproduce
the exact silent no-op §0 complains about. **Direction:** emit spans outside the
queue guards (sink reachable via WEC/config, not the transport queue contract).

---

## Resolution Log

All 28 findings resolved 2026-08-25 with the owner (Dhar Rawal). Rulings and
their design-doc locations are consolidated in
`fastworkflow_observability_studio_design.md` §9 (Decision Log); the design doc
was revised to v2 in the same pass. Decision provenance:

- **Individually ruled via AskUserQuestion** (recommended option accepted in
  each case): R1 (plumb ids into WEC), R2 (status-transition upsert), R3
  (columns + feedback table), R4 (on for fastworkflow entry points; 0700/0600),
  R5 (launch token + JWT-scoped GET), R14 (sync turn records at Phase B), R16
  (exempt conversations from retention), R24 (additive /initialize fields).
- **Bulk-accepted suggested resolution directions** (owner-approved bulk
  process): R6–R13, R15, R17–R23, R25–R28.

bd children fix-kw7.1.1–.28 closed with pointers to §9; parent review task
fix-kw7.1 closed on completion of this log.

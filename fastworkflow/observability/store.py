"""SQLite observability store + background-writer TraceSink (Phase 2).

Implements the "black box" of the observability design
(docs/fastworkflow_observability_studio_design.md §3.2): one
``observability.sqlite3`` per workflow under the state root, holding
conversations, turn records, OTel-shaped spans, offloaded artifacts, train
runs, and a writer-health diagnostics row.

Structure:

- ``ObservabilityStore`` — schema + synchronous operations (id minting,
  upserts, reads, prune, forget-channel). Writes use short-lived
  ``BEGIN IMMEDIATE`` transactions on per-call connections (house precedent:
  ``kvstore.py``; the chatbot's read layer uses per-request connections so
  checkpointing never starves).
- ``SQLiteTraceSink`` — the TraceSink implementation: two queues (a
  small turn-record/label queue with a bounded-timeout put — the only case a
  turn record may drop in v1 — and a droppable span queue bounded by
  ``FW_OBS_QUEUE_MAX``), drained by one daemon writer thread with batched
  transactions; ``close()`` (sentinel + bounded join) is wired to atexit and
  entry-point exit paths. Writer errors/drops land in the
  ``diagnostics`` table and are surfaced by the chatbot UI.
- ``get_observability_sink()`` — process-wide factory, one sink (= one writer
  thread) per DB path. Recording is always on, for fastWorkflow's own entry
  points and library embedders alike: the DB is created owner-only (0600 in a
  0700 directory) and pruned by the retention settings below.

Durability class: everything is best-effort; a write failure
never fails a turn. Multi-process writers are supported on local filesystems
only (WAL constraint — the state root must not be NFS).
"""

from __future__ import annotations

import atexit
import contextlib
import hashlib
import hmac
import json
import os
import queue
import re
import socket
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from pydantic import BaseModel, ConfigDict

import fastworkflow
from fastworkflow.observability import capture_policy as capture_policy_module
from fastworkflow import agent_runtime, runtime_manifest, state_paths, tracing
from fastworkflow.utils.logging import logger

# v2 (fix-42b): experiments.benchmark_id / benchmark_version /
# benchmark_digest_sha256 live in the CREATE TABLE literal
# only. Stores created before them are never migrated.
#
# Fresh schema (fix-49m.3): the `_SCHEMA_STATEMENTS` literal is the ONLY
# creator of every table and column. There is no ALTER/migration path. A
# populated store whose user_version is older than this constant is DELETED
# and recreated empty when the writer opens it (the store has never shipped in
# a release, so such a file can only be a developer's local DB); the read-only
# store refuses it and never deletes anything.
#
# v3 (fix-qe2): experiment_attempts.runtime_snapshot_json -- the binding
# server's credential-free runtime snapshot, stamped at claim time. Create-time
# column only; a v2 store is replaced on open like every older one.
# v4 (fix-aw5): human feedback and its evidence anchors live in this DB.
# v5 (fix-46l.2): feedback provenance distinguishes human, coding-agent, and
# distillation-agent annotations.
# v6 (fix-w6w): experiment archival is a durable annotation.
# Fresh schema only, with no migration of previously recorded evidence.
SCHEMA_VERSION = 6

# Which capture profile this deployment records under (arch §12.0 delta 3).
# Defaults to `debug`, which is byte-for-byte today's behavior: EXP-003 is a
# Phase 0 instrumentation slice, so installing the policy must change nothing
# until a deployment asks it to.
CAPTURE_PROFILE_VAR = "FW_OBS_CAPTURE_PROFILE"
_DEFAULT_CAPTURE_PROFILE = "debug"

# Profiles are immutable and cheap to share, and capture runs on every command of
# every turn, so they are built once per name rather than per turn (FW-NFR-005
# overhead is an EXP-003 stop condition).
_CAPTURE_POLICY_CACHE: dict[str, "capture_policy_module.CapturePolicy"] = {}

TERMINAL_TURN_STATUSES = frozenset({"completed", "failed", "cancelled", "abandoned"})

# Defaults per design §5.
_DEFAULT_DB_MAX_BYTES = 1_073_741_824
_DEFAULT_RETENTION_DAYS = 30
_DEFAULT_INLINE_ARTIFACT_BYTES = 262_144
_DEFAULT_QUEUE_MAX = 10_000

# Turn-record queue: small and separate [R13]. The bounded-timeout put is the
# only case a turn record may drop in v1.
_RECORD_QUEUE_MAX = 256
_RECORD_PUT_TIMEOUT_S = 2.0
_RECORD_BUSY_MAX_RETRIES = 5

# Sync-first turn-record writes (Phase 7 §2.4, rulings I1/I6/C8/C9).
_DEFAULT_SYNC_WRITE_TIMEOUT_S = 5
_DEFAULT_SYNC_BREAKER_COOLDOWN_S = 60
# Terminal records that fell back to the queue and have not been confirmed
# durable ride this ring until a retry lands them. Bounded: it is a memory
# holder on a path that only runs when the DB is already unhealthy, and the
# window the history trim defers by is bounded with it (ruling I1/I2).
_PENDING_RETRY_MAX = 64

_PRUNE_BATCH_ROWS = 5_000
_PRUNE_MAX_BATCHES = 20

# Poll interval for the quiesce handshake (fix-7de). Short, because it is only
# ever spun on for the moment it takes the writer to finish the batch it is in
# and reach the top of its loop, and an archive should not pay a tick for it.
_QUIESCE_POLL_S = 0.005

# Which drop counters carry an affected-turn-key list, and where it lives in the
# health dict. Only drops get one: a write error is about the DB, not about a turn.
_DROP_TURN_KEY_FIELDS: dict[str, str] = {
    "spans_dropped": "spans_dropped_turn_keys",
    "records_dropped": "records_dropped_turn_keys",
}

# Enough to name the affected turns of a run that lost a little evidence, small
# enough that a run losing everything cannot grow the list without bound. Past the
# cap the run is invalid anyway and the exact list has stopped being actionable.
_DROP_TURN_KEY_MAX = 256

# Counters an evidence run compares before and after (§12.4). Turn-record drops
# invalidate a run outright; the rest are reported.
_HEALTH_DELTA_COUNTERS: tuple[str, ...] = (
    "records_dropped",
    "spans_dropped",
    "write_errors",
    "refused_terminal_writes",
    "busy_retries",
    "sync_fallbacks",
)

# ----------------------------------------------------------------------
# The writer-health row is shared property, not one writer's scratchpad (fix-dnb)
# ----------------------------------------------------------------------
#
# `writer_health` is ONE diagnostics row per store, and more than one writer can
# own that store over its lifetime — a restarted server, a second harness, a
# sink recycled because the DB file was replaced. Until fix-dnb every persist
# REPLACED the row with the persisting sink's own counters, which start at zero,
# so a restart silently erased the predecessor's drops. `evidence_run`'s delta is
# `max(0, after - before)`, so a smaller `after` did not read as "impossible", it
# read as "no drops" — the one answer an evidence gate must never invent.
#
# Two changes, and both are needed. The merge below keeps the counters honest
# ACROSS writers; the incarnation stamp keeps them honest ABOUT writers, because
# a merged counter still cannot say whether the interval it spans contained a
# handover during which records were lost before anyone counted them.
_HEALTH_MONOTONE_COUNTERS: tuple[str, ...] = (
    "records_dropped",
    "spans_dropped",
    "write_errors",
    "busy_retries",
    "refused_terminal_writes",
    "sync_writes",
    "sync_fallbacks",
    # A high-water mark rather than a tally, but monotone under `max` all the
    # same: a later writer's smaller peak does not unmake an earlier one.
    "sync_write_ms_max",
    "dropped_turn_keys_elided",
)

# Everything NOT on that list — `pending_retry_depth`, `sync_breaker_open`, the
# incarnation stamp — is a gauge: it describes the writer that is running, not
# the store's history, so the newcomer's value simply wins and the merge below
# needs no rule for it. A predecessor's queue depth is not a floor under
# anything.
#
# Who wrote the counters. `id` is what `health_delta` compares — pid plus start
# time would collide across a fast restart inside one clock second, and a run
# whose writer was replaced must never look like a run whose writer persisted.
WRITER_INCARNATION_FIELD = "writer_incarnation"


def writer_incarnation_id(health: Optional[Mapping[str, Any]]) -> Optional[str]:
    """The id of the writer that last wrote this health snapshot, if it says."""
    if not health:
        return None
    stamp = health.get(WRITER_INCARNATION_FIELD)
    if not isinstance(stamp, Mapping):
        return None
    value = stamp.get("id")
    return str(value) if value else None


def merge_writer_health(
    stored: Optional[Mapping[str, Any]], incoming: Mapping[str, Any]
) -> dict[str, Any]:
    """Fold one writer's counters into the store's row without lowering it.

    Every persist goes through here (baseline, heartbeat, `persist_health`,
    close), so the row is a floor under everything any writer has ever counted
    for this store rather than a snapshot of whoever wrote last. The affected-turn
    lists are unioned for the same reason and under the same cap: a predecessor's
    named turns are the only record that those turns are incomplete.

    A FLOOR, NOT A SUM, and it could not be a sum: every persist carries the
    writer's cumulative total, so adding would count the same drop again on
    every heartbeat. The consequence is real and is covered elsewhere — a
    successor's own drops are invisible to a predecessor's larger count, so the
    row alone cannot be read as "everything this store ever lost". What it can be
    read as is what `evidence_run` needs: a number that never falls, so a
    subtraction across an interval cannot come out lower than the truth. The
    interval that actually SPANS a handover is caught by the incarnation stamp
    instead (`WriterHealthDelta.writer_restarted`), because no arithmetic over
    these counters could catch it.
    """
    incoming = dict(incoming)
    incoming.pop("updated_at", None)
    if not stored:
        return incoming
    merged = dict(stored)
    merged.pop("updated_at", None)
    # The newcomer's word on everything it is authoritative about — the
    # incarnation stamp, the gauges, the profile — then the floor re-imposed.
    merged.update(incoming)
    for name in _HEALTH_MONOTONE_COUNTERS:
        merged[name] = max(
            int(stored.get(name) or 0), int(incoming.get(name) or 0)
        )
    # A writer that has hit no error yet must not erase the error that made the
    # previous writer's run unreportable.
    if incoming.get("last_error") is None and stored.get("last_error") is not None:
        merged["last_error"] = stored.get("last_error")
    elided = 0
    for field in _DROP_TURN_KEY_FIELDS.values():
        union: list[str] = []
        for key in list(stored.get(field) or ()) + list(incoming.get(field) or ()):
            if key in union:
                continue
            if len(union) < _DROP_TURN_KEY_MAX:
                union.append(key)
            else:
                elided += 1
        merged[field] = union
    merged["dropped_turn_keys_elided"] = (
        int(merged.get("dropped_turn_keys_elided") or 0) + elided
    )
    return merged


# Retention pruning must not run while an evaluation is recording (§12.4: "pruning
# shall not run mid-evaluation") — the prune horizon is 30 days by default, but a
# size-capped prune evicts oldest-first regardless of age, so a long or
# high-volume run can delete its own early spans.
#
# Two mechanisms, because a run is not always one process: the chatbot spawns a
# server, so an in-process flag cannot reach the writer that actually prunes. The
# env var propagates to children; the counter serves a same-process harness.
SUPPRESS_PRUNE_VAR = "FW_OBS_SUPPRESS_PRUNE"
_prune_suppression_lock = threading.Lock()
_prune_suppression_depth = 0

# Additive feature markers deliberately do not bump SCHEMA_VERSION. Readers use
# these markers to avoid querying tables/columns that older snapshots lack.
FEATURE_DISTILLATION_V1 = "distillation_v1"
FEATURE_EXPERIMENTS_V1 = "experiments_v1"
FEATURE_EXPERIMENT_LIFECYCLE_V1 = "experiment_lifecycle_v1"
FEATURE_EXPERIMENT_DECLARATIONS_V1 = "experiment_declarations_v1"
FEATURE_EXPERIMENT_CLAIMS_V1 = "experiment_claims_v1"
FEATURE_EXPERIMENT_SEALING_V1 = "experiment_sealing_v1"
# The offload evidence tables (`offload_evidence`, `offload_subjects`): every
# archived execute response lives in this DB, keyed by its turn, and is erased
# and aged with that turn by the same transactions that erase the turn record.
FEATURE_OFFLOAD_EVIDENCE_V1 = "offload_evidence_v1"
# The offload runtime's diagnostic events (`offload_events`): what was
# archived, offloaded, searched and rehydrated, keyed by turn like the evidence
# and erased and aged with it.
FEATURE_OFFLOAD_EVENTS_V1 = "offload_events_v1"
# The evidence used to live in a second SQLite file beside this one, named
# `<db>` plus this suffix. Nothing reads it any more; opening a store deletes it.
LEGACY_OFFLOAD_SIDECAR_SUFFIX = ".offload-handles.sqlite3"
FEEDBACK_PROVENANCES = frozenset({"human", "coding_agent", "distillation_agent"})
# Composer tabs and stored-comment labels. Existing comments already used these
# headings (and "What did not work" as a synonym for went-wrong); reads parse
# them without rewriting the comment column, so older stores stay intact.
HUMAN_FEEDBACK_SECTIONS = (
    ("went_wrong", "What went wrong", ("what went wrong", "what did not work")),
    ("worked", "What worked", ("what worked",)),
    ("should_change", "What should change", ("what should change",)),
)
_HUMAN_FEEDBACK_HEADER_RE = re.compile(
    r"(?im)^[ \t]*(What went wrong|What did not work|What worked|What should change)"
    r"[ \t]*:[ \t]*"
)
_HUMAN_FEEDBACK_HEADER_TO_KEY = {
    alias: key
    for key, _label, aliases in HUMAN_FEEDBACK_SECTIONS
    for alias in aliases
}


def parse_human_feedback_comment(comment: str) -> dict[str, str]:
    """Split a stored comment into the three composer tabs.

    Unlabelled text is left in ``comment`` only: guessing a tab would invent a
    category the author did not choose. Duplicate headings concatenate.
    """
    sections = {key: "" for key, _label, _aliases in HUMAN_FEEDBACK_SECTIONS}
    if not isinstance(comment, str) or not comment:
        return sections
    matches = list(_HUMAN_FEEDBACK_HEADER_RE.finditer(comment))
    if not matches:
        return sections
    for index, match in enumerate(matches):
        key = _HUMAN_FEEDBACK_HEADER_TO_KEY[match.group(1).strip().lower()]
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(comment)
        chunk = comment[start:end].strip()
        if not chunk:
            continue
        sections[key] = f"{sections[key]}\n\n{chunk}".strip() if sections[key] else chunk
    return sections


def compose_human_feedback_comment(
    *,
    went_wrong: str = "",
    worked: str = "",
    should_change: str = "",
    comment: str | None = None,
) -> str:
    """Build the stored comment from tab fields, or keep a legacy free-form comment."""
    values = {
        "went_wrong": went_wrong,
        "worked": worked,
        "should_change": should_change,
    }
    for key, value in values.items():
        if value is None:
            values[key] = ""
        elif not isinstance(value, str):
            raise ValueError(f"{key} must be text")
    parts = []
    for key, label, _aliases in HUMAN_FEEDBACK_SECTIONS:
        text = values[key].strip()
        if text:
            parts.append(f"{label}: {text}")
    if parts:
        composed = "\n\n".join(parts)
    elif isinstance(comment, str) and comment.strip():
        composed = comment.strip()
    else:
        raise ValueError("feedback must contain text (at most 100000 characters)")
    if len(composed) > 100000:
        raise ValueError("feedback must contain text (at most 100000 characters)")
    return composed


def _human_feedback_row(row: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(row)
    value["span_ids"] = json.loads(value.pop("span_ids_json"))
    value.update(parse_human_feedback_comment(value.get("comment") or ""))
    return value

# Single source: the policy engine's own version (fix-49m.3 wiring).
CAPTURE_POLICY_VERSION = capture_policy_module.CAPTURE_POLICY_VERSION
CAPTURE_REGIME_DIAGNOSTIC = "observability_capture_regime"
STORE_IDENTITY_DIAGNOSTIC = "observability_store_identity"


def pruning_suppressed() -> bool:
    """Whether retention pruning is currently withheld."""
    if _env(SUPPRESS_PRUNE_VAR, "0") not in ("0", "false", "False", "no", "off"):
        return True
    with _prune_suppression_lock:
        return _prune_suppression_depth > 0


# Every FW_OBS_* knob that changes what is captured or retained, paired with its
# default. Enumerated rather than discovered by scanning os.environ for the prefix,
# because provenance must record the value **in effect** — including the defaults
# nobody set, which a scan cannot see. A run whose provenance omits
# FW_OBS_RETENTION_DAYS because it was unset is a run nobody can reproduce.
#
# The per-attribute cap is the constant tracing.MAX_ATTR_BYTES, not a setting,
# so it has no entry here.
_OBS_CONFIG_VARS: tuple[tuple[str, str], ...] = (
    (CAPTURE_PROFILE_VAR, _DEFAULT_CAPTURE_PROFILE),
    ("FW_OBS_RETENTION_DAYS", str(_DEFAULT_RETENTION_DAYS)),
    ("FW_OBS_DB_MAX_BYTES", str(_DEFAULT_DB_MAX_BYTES)),
    ("FW_OBS_INLINE_ARTIFACT_BYTES", str(_DEFAULT_INLINE_ARTIFACT_BYTES)),
    ("FW_OBS_CAPTURE_TRACEBACKS", "0"),
    ("FW_OBS_QUEUE_MAX", str(_DEFAULT_QUEUE_MAX)),
    ("FW_OBS_SYNC_WRITE_TIMEOUT_S", str(_DEFAULT_SYNC_WRITE_TIMEOUT_S)),
    ("FW_OBS_SYNC_BREAKER_COOLDOWN_S", str(_DEFAULT_SYNC_BREAKER_COOLDOWN_S)),
    (SUPPRESS_PRUNE_VAR, "0"),
)


def observability_config() -> dict[str, str]:
    """The FW_OBS_* values in effect, defaults included (§12.4)."""
    return {name: _env(name, default) for name, default in _OBS_CONFIG_VARS}


@contextlib.contextmanager
def suppress_pruning():
    """Withhold retention pruning for the duration of the block.

    Re-entrant by counting rather than by a boolean, so two nested evidence runs
    cannot have the inner one's exit re-enable pruning under the outer one.
    """
    global _prune_suppression_depth
    with _prune_suppression_lock:
        _prune_suppression_depth += 1
    try:
        yield
    finally:
        with _prune_suppression_lock:
            _prune_suppression_depth -= 1


def resolve_capture_policy() -> "capture_policy_module.CapturePolicy":
    """The policy this process captures under.

    Raises `CaptureProfileError` on an unrecognized profile name rather than
    falling back — see `capture_policy.policy_for_profile`. The sink resolves this
    in its constructor so a misconfigured deployment fails at startup instead of
    discovering months later that it recorded tenant data verbatim.
    """
    name = _env(CAPTURE_PROFILE_VAR, _DEFAULT_CAPTURE_PROFILE)
    policy = _CAPTURE_POLICY_CACHE.get(name)
    if policy is None:
        policy = capture_policy_module.policy_for_profile(name)
        _CAPTURE_POLICY_CACHE[name] = policy
    return policy


# Turn columns the capture policy deliberately does NOT touch.
#
# These two are not evidence, they are operational state: `get_memory_window` and
# `_USABLE_TURN_FILTER` read exactly `conversation_summary` and
# `conversation_traces` to rebuild the agent's conversation memory, and the filter
# requires the summary to be non-NULL. Withholding them would not reduce what a
# bundle exposes — it would make the agent forget, which is a behavior change and
# therefore outside a Phase 0 slice.
#
# PII in conversation memory is a real gap; it is fix-cj4's. It needs a redaction
# that leaves memory usable, which is a different problem from withholding
# evidence, and solving it by omission here would silently degrade every
# evidence-profile run's agent.
_POLICY_EXEMPT_TURN_COLUMNS = frozenset({"conversation_summary", "conversation_traces"})

# Turn columns that are pure evidence — nothing operational reads them — paired
# with what they actually contain. `failure_reason` is `opaque-payload` rather
# than text because it can embed a provider error body (the [R20] scenario), so
# nobody can say what is in it.
_POLICED_TURN_COLUMNS: tuple[tuple[str, str], ...] = (
    ("user_message", "user-text"),
    ("refined_user_message", "user-text"),
    ("answer", "user-text"),
    ("failure_reason", "opaque-payload"),
)

# ----------------------------------------------------------------------
# The write paths that do NOT ride the TurnResult pipeline (fix-ajv.9)
# ----------------------------------------------------------------------
#
# `serialize_turn_result` is where the capture policy meets a turn, and
# `upsert_turn_row` is where the credential scrub meets one. Five persisted
# surfaces reach SQLite without passing through either: conversation labels,
# feedback, train-run metrics, writer diagnostics, and the SCALAR columns beside
# a span's (already scrubbed) `attributes` JSON. FW-REQ-002 clause 3 requires
# every captured field to have a declared policy, so each of the five is decided
# here rather than by omission — including the three that are deliberately
# scrub-only, whose reasons are recorded at their write sites.
#
# Policy paths are named constants because a deployment re-admitting one of these
# under the evidence profile has to spell the path exactly (see
# `CapturePolicy.policy_for`), and a path that only exists as a literal inside a
# method is a path nobody can find in order to spell it.
POLICY_PATH_SPAN_NAME = "span.name"
POLICY_PATH_SPAN_COMMAND_NAME = "span.command_name"
POLICY_PATH_SPAN_CONTEXT = "span.context"
POLICY_PATH_CONVERSATION_TOPIC = "conversation.topic"
POLICY_PATH_CONVERSATION_SUMMARY = "conversation.summary"
POLICY_PATH_TRAIN_METRICS = "train_run.metrics_json"
# (ido-zlm) The sixth surface: the RAW command response that
# `observation_offloading.archive` persists into `offload_evidence`. It does not
# ride the TurnResult pipeline, so without this path it escaped both
# protections entirely -- a credential in a command response was stored
# verbatim where the same text inside a span attribute was scrubbed.
POLICY_PATH_OFFLOAD_OBSERVATION = "offload.observation.text"


def _protected_text(
    value: Any,
    *,
    redactor: Redactor,
    policy: "capture_policy_module.CapturePolicy",
    field_path: str,
    classification: str,
) -> Any:
    """Credential-scrub a persisted string, then apply the capture policy to it.

    **Scrub first, policy second**, which is the opposite order from
    `_POLICED_TURN_COLUMNS` (there the policy runs in `serialize_turn_result` and
    the scrub runs later, in `upsert_turn_row`). Two reasons it has to be this way
    on these paths:

    * A conversation label can arrive by either of two routes —
      `SQLiteTraceSink._apply_label`, which scrubs before calling
      `apply_label_txn`, or `ObservabilityStore.record_conversation_label`, which
      does not. Scrubbing first makes both produce `policy(scrub(text))`, because
      the scrub is idempotent. Policing first would give the same label two
      different digests depending on which route wrote it, and a digest that
      depends on plumbing is not a digest anyone can compare.
    * The badge left behind carries a digest of what it replaced. Digesting the
      unscrubbed text would make the badge a confirmation oracle for a guessed
      credential, which is a strange thing for a redaction record to be.

    Returns TEXT, always: an envelope is serialized here because every caller
    binds the result to a TEXT column and sqlite3 cannot bind a mapping. Same
    reasoning as `_policed_column`, which does it for the turn row.
    """
    if not value:
        return value
    scrubbed = redactor.redact(value)
    captured = policy.apply(field_path, scrubbed, classification=classification)
    if capture_policy_module.is_capture_envelope(captured):
        return json.dumps(captured, ensure_ascii=False)
    return captured


class WriterHealthDelta(BaseModel):
    """What the store lost between two health snapshots (§12.4).

    The asymmetry is the point. A dropped **turn record** invalidates an evidence
    run: turn records are the evidence, and one missing turn means the run's
    numerator and denominator disagree in a way no analysis can repair. A dropped
    **span** does not invalidate the run — spans are best-effort by design — but it
    must be reported with the turns it affected, so a reader knows which turns have
    incomplete detail rather than assuming all of them are whole.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    records_dropped: int = 0
    spans_dropped: int = 0
    write_errors: int = 0
    refused_terminal_writes: int = 0
    busy_retries: int = 0
    sync_fallbacks: int = 0
    records_dropped_turn_keys: tuple[str, ...] = ()
    spans_dropped_turn_keys: tuple[str, ...] = ()
    # Non-zero means the affected-turn lists were capped and are incomplete.
    dropped_turn_keys_elided: int = 0
    # True when either snapshot was unavailable. Distinct from "no drops": nothing
    # was compared, so nothing may be claimed.
    incomparable: bool = False
    # True when a DIFFERENT writer wrote the two snapshots (fix-dnb). The
    # counters are merged rather than replaced now, so they no longer go
    # backwards across a handover — but a record the dying writer had accepted
    # and not yet written is lost without ever being counted, so the interval
    # cannot claim zero drops however healthy its arithmetic looks.
    writer_restarted: bool = False
    writer_incarnation_before: Optional[str] = None
    writer_incarnation_after: Optional[str] = None

    @property
    def lost_turn_records(self) -> bool:
        return self.records_dropped > 0

    @property
    def evidence_valid(self) -> bool:
        """Whether a run over this interval may be reported as evidence.

        False when a turn record was dropped, and False when the interval could
        not be compared at all — an unknown is not a pass. Dropped spans leave
        this True; they are reported through `problems()`.

        A writer restart inside the interval is fatal for the same reason the
        unknown is: the counters that would have named the loss died with the
        writer that was holding them.
        """
        return (
            not self.incomparable
            and not self.lost_turn_records
            and not self.writer_restarted
        )

    def problems(self) -> tuple[str, ...]:
        """Every reason this interval is imperfect, worst first.

        Returns all of them rather than the first, so an operator sees the whole
        picture in one pass.
        """
        found: list[str] = []
        if self.incomparable:
            found.append(
                "writer health could not be compared (a snapshot was missing); "
                "evidence validity is unknown, which is not the same as valid"
            )
        if self.writer_restarted:
            found.append(
                f"the observability writer was replaced during this run "
                f"(incarnation {self.writer_incarnation_before} -> "
                f"{self.writer_incarnation_after}); records the previous writer "
                f"had accepted may have been lost without ever being counted, so "
                f"this interval is not valid evidence (§12.4)."
            )
        if self.records_dropped:
            affected = ", ".join(self.records_dropped_turn_keys) or "unknown turns"
            found.append(
                f"{self.records_dropped} turn record(s) DROPPED, affecting: "
                f"{affected}. The run is not valid evidence (§12.4)."
            )
        if self.spans_dropped:
            affected = ", ".join(self.spans_dropped_turn_keys) or "unknown turns"
            found.append(
                f"{self.spans_dropped} span(s) dropped, affecting: {affected}. "
                "These turns have incomplete detail; the run remains valid."
            )
        if self.dropped_turn_keys_elided:
            found.append(
                f"{self.dropped_turn_keys_elided} further affected turn key(s) were "
                "not recorded (list capped); the lists above are incomplete"
            )
        if self.refused_terminal_writes:
            found.append(
                f"{self.refused_terminal_writes} write(s) to an already-terminal "
                "turn row were refused"
            )
        if self.write_errors:
            found.append(f"{self.write_errors} write error(s)")
        return tuple(found)


def health_delta(
    before: Optional[dict[str, Any]], after: Optional[dict[str, Any]]
) -> WriterHealthDelta:
    """Compare two `health_snapshot()` results.

    Counters are cumulative and monotonic, so the delta is a subtraction; the
    affected-turn-key lists are set differences, which is what makes the result
    specific to this run rather than to the DB's whole history.

    A missing snapshot yields `incomparable=True` rather than a zero delta,
    because "we could not tell" and "nothing was dropped" are the two answers an
    evidence gate must never confuse.

    A snapshot pair written by two DIFFERENT writers yields `writer_restarted`.
    The subtraction is still performed and still meaningful — the row
    is merged monotonically now, so the counters do not go backwards — but the
    handover itself is unmeasured: whatever the dying writer had taken in and not
    yet committed left no counter behind. Only a stamp on BOTH sides can say
    this; a snapshot with no stamp (the in-process `{}` baseline for a sink that
    appeared mid-run) is not evidence of a restart and is not reported as one.
    """
    if before is None or after is None:
        return WriterHealthDelta(incomparable=True)

    before_writer = writer_incarnation_id(before)
    after_writer = writer_incarnation_id(after)
    restarted = (
        before_writer is not None
        and after_writer is not None
        and before_writer != after_writer
    )
    counters = {
        name: max(0, int(after.get(name) or 0) - int(before.get(name) or 0))
        for name in _HEALTH_DELTA_COUNTERS
    }
    new_keys: dict[str, tuple[str, ...]] = {}
    for counter, field in _DROP_TURN_KEY_FIELDS.items():
        seen_before = set(before.get(field) or ())
        new_keys[field] = tuple(
            key for key in (after.get(field) or ()) if key not in seen_before
        )
    return WriterHealthDelta(
        **counters,
        records_dropped_turn_keys=new_keys["records_dropped_turn_keys"],
        spans_dropped_turn_keys=new_keys["spans_dropped_turn_keys"],
        dropped_turn_keys_elided=max(
            0,
            int(after.get("dropped_turn_keys_elided") or 0)
            - int(before.get("dropped_turn_keys_elided") or 0),
        ),
        writer_restarted=restarted,
        writer_incarnation_before=before_writer,
        writer_incarnation_after=after_writer,
    )


# Turn columns the capture policy deliberately does NOT touch.
#
# These two are not evidence, they are operational state: `get_memory_window` and
# `_USABLE_TURN_FILTER` read exactly `conversation_summary` and
# `conversation_traces` to rebuild the agent's conversation memory, and the filter
# requires the summary to be non-NULL. Withholding them would not reduce what a
# bundle exposes — it would make the agent forget, which is a behavior change and
# therefore outside a Phase 0 slice.
#
# PII in conversation memory is a real gap; it is fix-cj4's. It needs a redaction
# that leaves memory usable, which is a different problem from withholding
# evidence, and solving it by omission here would silently degrade every
# evidence-profile run's agent.
_POLICY_EXEMPT_TURN_COLUMNS = frozenset({"conversation_summary", "conversation_traces"})

# Turn columns that are pure evidence — nothing operational reads them — paired
# with what they actually contain. `failure_reason` is `opaque-payload` rather
# than text because it can embed a provider error body (the [R20] scenario), so
# nobody can say what is in it.
_POLICED_TURN_COLUMNS: tuple[tuple[str, str], ...] = (
    ("user_message", "user-text"),
    ("refined_user_message", "user-text"),
    ("answer", "user-text"),
    ("failure_reason", "opaque-payload"),
)

# ----------------------------------------------------------------------
# The write paths that do NOT ride the TurnResult pipeline (fix-ajv.9)
# ----------------------------------------------------------------------
#
# `serialize_turn_result` is where the capture policy meets a turn, and
# `upsert_turn_row` is where the credential scrub meets one. Five persisted
# surfaces reach SQLite without passing through either: conversation labels,
# feedback, train-run metrics, writer diagnostics, and the SCALAR columns beside
# a span's (already scrubbed) `attributes` JSON. FW-REQ-002 clause 3 requires
# every captured field to have a declared policy, so each of the five is decided
# here rather than by omission — including the three that are deliberately
# scrub-only, whose reasons are recorded at their write sites.
#
# Policy paths are named constants because a deployment re-admitting one of these
# under the evidence profile has to spell the path exactly (see
# `CapturePolicy.policy_for`), and a path that only exists as a literal inside a
# method is a path nobody can find in order to spell it.
POLICY_PATH_SPAN_NAME = "span.name"
POLICY_PATH_SPAN_COMMAND_NAME = "span.command_name"
POLICY_PATH_SPAN_CONTEXT = "span.context"
POLICY_PATH_CONVERSATION_TOPIC = "conversation.topic"
POLICY_PATH_CONVERSATION_SUMMARY = "conversation.summary"
POLICY_PATH_TRAIN_METRICS = "train_run.metrics_json"
# (ido-zlm) The sixth surface: the RAW command response that
# `observation_offloading.archive` persists into `offload_evidence`. It does not
# ride the TurnResult pipeline, so without this path it escaped both
# protections entirely -- a credential in a command response was stored
# verbatim where the same text inside a span attribute was scrubbed.
POLICY_PATH_OFFLOAD_OBSERVATION = "offload.observation.text"


def _protected_text(
    value: Any,
    *,
    redactor: Redactor,
    policy: "capture_policy_module.CapturePolicy",
    field_path: str,
    classification: str,
) -> Any:
    """Credential-scrub a persisted string, then apply the capture policy to it.

    **Scrub first, policy second**, which is the opposite order from
    `_POLICED_TURN_COLUMNS` (there the policy runs in `serialize_turn_result` and
    the scrub runs later, in `upsert_turn_row`). Two reasons it has to be this way
    on these paths:

    * A conversation label can arrive by either of two routes —
      `SQLiteTraceSink._apply_label`, which scrubs before calling
      `apply_label_txn`, or `ObservabilityStore.record_conversation_label`, which
      does not. Scrubbing first makes both produce `policy(scrub(text))`, because
      the scrub is idempotent. Policing first would give the same label two
      different digests depending on which route wrote it, and a digest that
      depends on plumbing is not a digest anyone can compare.
    * The badge left behind carries a digest of what it replaced. Digesting the
      unscrubbed text would make the badge a confirmation oracle for a guessed
      credential, which is a strange thing for a redaction record to be.

    Returns TEXT, always: an envelope is serialized here because every caller
    binds the result to a TEXT column and sqlite3 cannot bind a mapping. Same
    reasoning as `_policed_column`, which does it for the turn row.
    """
    if not value:
        return value
    scrubbed = redactor.redact(value)
    captured = policy.apply(field_path, scrubbed, classification=classification)
    if capture_policy_module.is_capture_envelope(captured):
        return json.dumps(captured, ensure_ascii=False)
    return captured


def protect_offload_observation(text: str) -> str:
    """Scrub-then-police one raw command response bound for `offload_evidence`.

    The evidence row is written by `observation_offloading.archive`, not by the
    TurnResult pipeline, so it cannot ride that pipeline's protections. What it
    can do -- and what this function exists for -- is call the SAME two
    protections in the SAME order as every other policed surface, instead of
    growing a second redactor that drifts from this one.
    `observation_offloading.archive.persist` passes the response text through
    here at write time and stores whatever comes back.

    `opaque-payload`, for the reason `failure_reason` carries that
    classification: a command response is whatever a workflow's command chose to
    return, so nobody can say what is inside it. Under the `debug` profile --
    the default, and what every evaluation run to date was captured under --
    that classification has no effect and this is the credential scrub alone,
    which is exactly the protection a span attribute already had. Under
    `evidence` it withholds the response behind a badge; a deployment that wants
    default-deny spans and full-fidelity observations spells
    `POLICY_PATH_OFFLOAD_OBSERVATION` in a `CaptureFieldPolicy`, which is what
    these path constants exist for.

    Returns TEXT, always, like `_protected_text`: the evidence row stores UTF-8 bytes,
    and a withheld response is stored as its serialized badge -- size, digest and
    class -- never as silence.
    """
    return _protected_text(
        text,
        redactor=Redactor(),
        policy=resolve_capture_policy(),
        field_path=POLICY_PATH_OFFLOAD_OBSERVATION,
        classification="opaque-payload",
    )


class IncompatibleObservabilityDB(RuntimeError):
    """The DB cannot be opened by this build.

    Raised for a DB written by a newer fastWorkflow, which every reader and
    writer refuses; by the read-only store for an older one, which it never
    alters; and by the writer for an older one it could not delete.
    """


class _OlderPopulatedStore(Exception):
    """Internal: `_ensure_schema_once` met a populated DB from an older build."""

    def __init__(self, version: int) -> None:
        super().__init__(version)
        self.version = version


class ExperimentNotFound(KeyError):
    """An experiment write matched no row.

    Raised rather than passed over: `clear_conversations` is an HTTP-triggered
    whole-DB erase that can land while a harness is running, and a silent no-op
    there leaves turns labelled against a container that no longer exists
    (`[XR15]`).
    """

    def __init__(self, experiment_id: str) -> None:
        self.experiment_id = experiment_id
        super().__init__(f"no experiment {experiment_id!r} in this database")


class ExperimentIsClosed(ValueError):
    """An attempt was written to an experiment that is no longer running.

    `complete` and `invalid` are terminal: their attempt rows are the evidence a
    reported score rests on, and a second run under the same id would overwrite
    them in place.
    """

    def __init__(self, experiment_id: str, status: str) -> None:
        self.experiment_id = experiment_id
        self.status = status
        super().__init__(
            f"experiment {experiment_id!r} is {status!r}, not running; its "
            "attempts are closed. Start a new experiment rather than rewriting "
            "the record a score was reported from."
        )


class AttemptValueConflict(ValueError):
    """A terminal attempt value was rewritten to a different value."""


class ExperimentDeclarationConflict(ValueError):
    """An immutable exact-attempt declaration was changed."""


class UndeclaredExperimentAttempt(ValueError):
    """An attempt identity was not part of the immutable declaration."""


class AttemptClaimError(ValueError):
    """An attempt bootstrap or fencing claim was refused."""


class StaleExperimentClaim(AttemptClaimError):
    """A runtime or queued record carries an obsolete attempt epoch."""


class StoreIdentityMismatch(ValueError):
    """A controller opened a different durable observability store."""


class SourceChangedDuringArchive(RuntimeError):
    """The live source changed while its evidence snapshot was being made."""


class WriterStillOpen(RuntimeError):
    """A seal was attempted while this process still owned a live writer."""


class CaptureRegimeChanged(ValueError):
    """An experiment was re-created under a different capture profile/policy."""

    def __init__(self, experiment_id: str, stored: str, incoming: str) -> None:
        self.experiment_id = experiment_id
        super().__init__(
            f"experiment {experiment_id!r} was captured under {stored} and is "
            f"now being written under {incoming}. The two halves would not be "
            "measuring the same columns; record the second half as its own "
            "experiment."
        )


class PartialBenchmarkPin(ValueError):
    """A benchmark pin requires id, version, and digest together."""

    def __init__(self, experiment_id: str) -> None:
        self.experiment_id = experiment_id
        super().__init__(
            f"experiment {experiment_id!r} benchmark pin requires "
            "benchmark_id, benchmark_version, and benchmark_digest_sha256 "
            "together; partial pins are refused"
        )


class BenchmarkPinIsWriteOnce(ValueError):
    """A stored benchmark pin was rewritten to a different value."""

    def __init__(self, experiment_id: str) -> None:
        self.experiment_id = experiment_id
        super().__init__(
            f"experiment {experiment_id!r} already has a benchmark pin; it is "
            "write-once by design"
        )


def _env(name: str, default: str) -> str:
    """FW_* knob: process env first, then the workflow env file, then default."""
    value = os.environ.get(name)
    if value is None or value == "":
        value = fastworkflow._env_vars.get(name)
    if value is None or str(value).strip() == "":
        return default
    return str(value).strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _chunked(values: list[Any], size: int = 400) -> Iterable[list[Any]]:
    for start in range(0, len(values), size):
        yield values[start:start + size]


def _iso_to_ms(value: Optional[str]) -> int:
    """ISO timestamp → ms epoch (legacy conversation-record convention)."""
    if not value:
        return 0
    try:
        return int(datetime.fromisoformat(value).timestamp() * 1000)
    except (ValueError, TypeError):
        return 0


# ----------------------------------------------------------------------
# Redaction [R20]
# ----------------------------------------------------------------------

_SECRET_ENV_SUFFIXES = ("_API_KEY", "_TOKEN", "_SECRET", "_PASSWORD")

# Known credential shapes, scrubbed independently of the environment.
_SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]{16,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{4,}"),
]

_REDACTED = "[REDACTED]"


class Redactor:
    """Sink-boundary scrub of credential shapes and loaded secret env values.

    Collects the VALUES of every ``*_API_KEY``/``*_TOKEN``-style variable from
    the process environment and the loaded fastworkflow env files, and removes
    them (plus well-known credential shapes) from any text persisted.
    """

    def __init__(self) -> None:
        values: set[str] = set()
        sources: list[dict] = [dict(os.environ)]
        env_vars = getattr(fastworkflow, "_env_vars", None)
        if isinstance(env_vars, dict):
            sources.append(env_vars)
        for source in sources:
            for key, value in source.items():
                if not isinstance(value, str) or len(value) < 8:
                    continue
                upper = str(key).upper()
                # Infix match: the house convention is LITELLM_API_KEY_<ROLE>,
                # so the secret marker is not necessarily the suffix.
                if any(marker in upper for marker in _SECRET_ENV_SUFFIXES):
                    values.add(value)
        # Longest first so partial overlaps cannot resurrect a suffix.
        self._values = sorted(values, key=len, reverse=True)

    def redact(self, text: str) -> str:
        if not text:
            return text
        for value in self._values:
            if value in text:
                text = text.replace(value, _REDACTED)
        for pattern in _SECRET_PATTERNS:
            text = pattern.sub(_REDACTED, text)
        return text


# ----------------------------------------------------------------------
# Turn-record serialization (size policy [R10], envelopes, traceback gate)
# ----------------------------------------------------------------------


def _decode_attempt_row(row: Any) -> dict[str, Any]:
    """An `experiment_attempts` row as readers see it.

    `runtime_snapshot_json` is exposed decoded under
    `runtime_snapshot` -- a dict, or None when the binding server recorded no
    snapshot -- so the chatbot UI and the workspace render it without parsing.
    The raw column is dropped from the projection rather than duplicated: one
    key, one shape. An unreadable value is reported as None with the raw text
    kept under `runtime_snapshot_json`, so a corrupt stamp is visible rather
    than silently the same as an absent one.
    """
    record = dict(row)
    raw = record.pop("runtime_snapshot_json", None)
    if raw is None:
        record["runtime_snapshot"] = None
        return record
    try:
        decoded = json.loads(raw)
    except (TypeError, ValueError):
        record["runtime_snapshot"] = None
        record["runtime_snapshot_json"] = raw
        return record
    record["runtime_snapshot"] = decoded if isinstance(decoded, dict) else None
    if record["runtime_snapshot"] is None:
        record["runtime_snapshot_json"] = raw
    return record


def _sanitize_json_value(value: Any) -> Any:
    """Coerce a dumped value into JSON-safe form; non-serializable values
    become placeholder envelopes rather than failing the record."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _sanitize_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_json_value(v) for v in value]
    return {
        "__fw_unserializable__": type(value).__name__,
        "repr": repr(value)[:1024],
    }


def _capture_classify_for_turn(turn_result: Any) -> Optional[Any]:
    """Resolve ``RuntimeMetadata.capture_classification`` for one turn, if any.

    The workflow folderpath is carried on ``TurnResult.metadata`` because the
    sink has no other durable link to the manifest registered at startup.
    Unregistered or absent metadata yields None, which is the evidence
    profile's default-deny input.
    """
    metadata = getattr(turn_result, "metadata", None) or {}
    if not isinstance(metadata, dict):
        return None
    folderpath = metadata.get("workflow_folderpath")
    if not folderpath:
        return None
    runtime = runtime_manifest.get_runtime_metadata(folderpath)
    if runtime is None:
        return None
    return runtime.capture_classification


def _policy_classification(
    classify: Optional[Any], command_name: str, field_name: str
) -> Optional[str]:
    """The workflow's declared classification for one parameter, or None.

    None is the default-deny input, so a resolver that raises must be treated as
    "unclassified" rather than allowed to lose the whole turn record: under the
    evidence profile that omits the field, which is the conservative direction.
    """
    if classify is None:
        return None
    try:
        return classify(command_name, field_name)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug(f"capture classification resolver failed: {exc!r}")
        return None


def _apply_capture_policy(
    record: dict[str, Any],
    policy: "capture_policy_module.CapturePolicy",
    classify: Optional[Any] = None,
) -> None:
    """Apply the field policy to one dumped TurnResult, in place.

    Runs on the `model_dump()` copy, never on the live accumulator objects, so
    what the caller and the user see is untouched — the policy governs what is
    *persisted*, which is the whole reason it is applied here rather than at the
    sink's string boundary.

    Ordering matters: this runs BEFORE the artifact-offload pass. A withheld
    artifact collapses to a small envelope and is therefore never offloaded, so
    its bytes never reach the `artifacts` table. Applying the policy afterwards
    would redact the record while leaving the raw value in `inline_value`.
    """
    for command_output in record.get("turn_output", {}).get("command_outputs", []):
        command_name = command_output.get("command_name") or "unknown"
        # A FAILED command's response and artifacts are diagnostic content —
        # an exception repr, a message, a traceback — not the command's normal
        # output, so they must not inherit the policy written for its happy
        # path. `CapturePolicy.apply` returns a value WHOLE when a declared
        # policy is not gated for this sink, so a perfectly reasonable
        # `command.X.response` rule (X's normal response is benign, keep it)
        # would also release X's failure text once fix-ajv.16 started naming
        # failed commands. A separate segment makes releasing error text
        # something a deployment has to say, rather than something it inherits.
        # fix-ajv.18.
        #
        # ask_user is excluded deliberately [A7]: `success=False` on an
        # ask_user entry means the question is still unanswered, not that
        # anything failed, and its response is the user's ANSWER — ordinary
        # user text that belongs on the ordinary path.
        #
        # Read via the structural marker with the name as fallback, mirroring
        # `CommandOutput.is_ask_user` — this walks the model_dump()ed dict, so
        # it cannot call the property. `ask_user_entry` absent means a record
        # written before that field existed (fix-ajv.17); True/False are
        # authoritative, and False is what a failed command NAMED `ask_user`
        # carries, which is the whole point of not testing the name here.
        response_dict = command_output.get("command_response") or {}
        marker = command_output.get("ask_user_entry")
        is_ask_user = marker if marker is not None else command_name == "ask_user"
        is_failure = response_dict.get("success") is False and not is_ask_user
        # PARAMETERS DELIBERATELY STAY ON THE ORDINARY PATH, and this asymmetry
        # is the point rather than an oversight. A failure's parameters are the
        # SAME values the success path carries, so a rule written to gate them
        # must keep applying; moving them under `.error.` would stop that rule
        # matching and fall through to the profile default — which under
        # `debug` returns the value whole. Separating them would un-gate the
        # one field group the success policy is right about.
        outcome_prefix = (
            f"command.{command_name}.error" if is_failure else f"command.{command_name}"
        )
        parameters = command_output.get("command_parameters")
        if isinstance(parameters, dict):
            for field_name in list(parameters):
                parameters[field_name] = policy.apply(
                    f"command.{command_name}.parameters.{field_name}",
                    parameters[field_name],
                    classification=_policy_classification(
                        classify, command_name, field_name
                    ),
                )
        elif parameters is not None:
            # The ask_user role inversion [A10]: for an `ask_user` entry
            # `command_parameters` is the agent's *question* as a str, not a
            # parameter mapping — and the response below is the user's *answer*.
            command_output["command_parameters"] = policy.apply(
                f"command.{command_name}.parameters",
                parameters,
                classification="user-text",
            )

        response = command_output.get("command_response") or {}
        if response.get("response"):
            response["response"] = policy.apply(
                f"{outcome_prefix}.response",
                response["response"],
                classification="user-text",
            )
        artifacts = response.get("artifacts")
        if not isinstance(artifacts, dict):
            continue
        for key in list(artifacts):
            value = artifacts[key]
            # An artifact ref envelope is a pointer, not content: the value has
            # already been moved out, and digesting a pointer loses the join
            # without protecting anything.
            if isinstance(value, dict) and "__fw_artifact_ref__" in value:
                continue
            artifacts[key] = policy.apply(
                f"{outcome_prefix}.artifacts.{key}",
                value,
                classification=_policy_classification(classify, command_name, key),
            )


def _policed_column(
    policy: "capture_policy_module.CapturePolicy",
    column: str,
    classification: str,
    value: Any,
) -> Any:
    """A turn text column after policy, still bindable as TEXT.

    An envelope is serialized rather than returned as a dict: these are TEXT
    columns and sqlite3 cannot bind a mapping, and `conversation_summary`'s
    non-NULL contract shows how much the read side depends on their shape.
    """
    if not value:
        return value
    captured = policy.apply(f"turn.{column}", value, classification=classification)
    if capture_policy_module.is_capture_envelope(captured):
        return json.dumps(captured, ensure_ascii=False)
    return captured


def serialize_turn_result(
    turn_result: Any,
    *,
    policy: "Optional[capture_policy_module.CapturePolicy]" = None,
    classify: Optional[Any] = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Project a TurnResult into (turn_row, artifact_rows) at emission time.

    - ``record_json`` holds the full internal TurnResult (post-envelope,
      post-capture-policy, pre-credential-redaction — the sink redacts the
      serialized text).
    - Any artifact value over ``FW_OBS_INLINE_ARTIFACT_BYTES`` is replaced in
      place by a ref envelope; the artifacts table is the only value holder.
    - ``traceback`` artifacts persist only under FW_OBS_CAPTURE_TRACEBACKS=1.

    Runs in the caller thread so the row snapshots the turn as emitted (the
    accumulator's CommandOutput objects mutate on resume).

    The capture policy (arch §6.6) runs here rather than at the sink's string
    boundary because it is per-field and `Redactor` operates on already-serialized
    JSON: by the time the text exists, the field structure the policy classifies
    is gone. The two compose — the policy decides what is captured, the redactor
    still scrubs credential shapes out of whatever survives. `policy=None`
    resolves `FW_OBS_CAPTURE_PROFILE`, which defaults to the verbatim `debug`
    profile, so this is a no-op unless a deployment opts in.
    """
    turn_output = turn_result.turn_output
    inline_limit = _env_int("FW_OBS_INLINE_ARTIFACT_BYTES", _DEFAULT_INLINE_ARTIFACT_BYTES)
    capture_tracebacks = _env("FW_OBS_CAPTURE_TRACEBACKS", "0") == "1"
    policy = policy or resolve_capture_policy()

    try:
        record = turn_result.model_dump(mode="python")
    except Exception:
        record = {"turn_output": {"turn_key": turn_output.turn_key}}
    record = _sanitize_json_value(record)
    # computed_field `success` is included by model_dump; make sure it is
    # present even on the fallback path.
    record.setdefault("turn_output", {}).setdefault("success", turn_output.success)

    _apply_capture_policy(record, policy, classify)

    turn_key = turn_output.turn_key
    channel_id = turn_result.channel_id or ""
    artifact_rows: list[dict[str, Any]] = []

    for command_output in record.get("turn_output", {}).get("command_outputs", []):
        response = command_output.get("command_response") or {}
        artifacts = response.get("artifacts")
        if not isinstance(artifacts, dict):
            continue
        for key in list(artifacts.keys()):
            if key == "traceback" and not capture_tracebacks:
                artifacts[key] = "[suppressed; set FW_OBS_CAPTURE_TRACEBACKS=1]"
                continue
            value_json = json.dumps(artifacts[key], ensure_ascii=False)
            size = len(value_json.encode("utf-8"))
            if size <= inline_limit:
                continue
            artifact_id = uuid.uuid4().hex
            sha256 = hashlib.sha256(value_json.encode("utf-8")).hexdigest()
            content_type = (
                "text/plain" if isinstance(artifacts[key], str) else "application/json"
            )
            artifact_rows.append(
                {
                    "artifact_id": artifact_id,
                    "turn_key": turn_key,
                    "channel_id": channel_id,
                    "span_id": None,
                    "key": key,
                    "content_type": content_type,
                    "size_bytes": size,
                    "sha256": sha256,
                    "inline_value": value_json.encode("utf-8"),
                    "error": None,
                    "experiment_id": turn_result.experiment_id,
                    "task_id": turn_result.task_id,
                    "attempt": turn_result.attempt,
                    "claim_epoch": turn_result.claim_epoch,
                    "server_incarnation": turn_result.server_incarnation,
                }
            )
            # Envelope shape per final spec [A10] / this design [R10].
            artifacts[key] = {
                "__fw_artifact_ref__": artifact_id,
                "size": size,
                "content_type": content_type,
                "content_encoding": None,
                "error": None,
            }

    turn_row = {
        "turn_key": turn_key,
        "channel_id": channel_id,
        "conversation_id": turn_result.conversation_id,
        "ordinal": turn_result.ordinal,
        "user_message": turn_result.user_message or "",
        "refined_user_message": turn_result.refined_user_message,
        "entry_workflow_name": turn_result.entry_workflow_name or "",
        "entry_context": turn_result.entry_context or "",
        "status": turn_output.status.value,
        "success": 1 if turn_output.success else 0,
        "failure_reason": turn_output.failure_reason,
        "answer": turn_output.answer or "",
        # Stamped by WEC._build_turn_result only when the turn appended a
        # conversation-history entry, so these are exactly the rows the
        # _USABLE_TURN_FILTER admits as conversation memory.
        "conversation_summary": getattr(turn_result, "conversation_summary", None),
        "conversation_traces": getattr(turn_result, "conversation_traces", None),
        "started_at": (
            turn_result.started_at.isoformat() if turn_result.started_at else None
        ),
        "completed_at": (
            turn_result.completed_at.isoformat() if turn_result.completed_at else None
        ),
        "suspended_ms": int(turn_result.suspended_ms or 0),
        "continuation_of": turn_result.continuation_of,
        # The experiment container's labels (`fix-bn1` `[XR17]`). Bound on the
        # WEC before the turn and copied off the TurnResult here, so they take
        # the same path as channel_id rather than being stitched on by a later
        # query. NULL on every ordinary turn. `upsert_turn_row` derives its
        # column list from this dict, so these three keys are also what writes
        # them -- and a key here with no matching column raises on `_sync_write`
        # and trips the sync breaker, which is why the DDL and this projection
        # must ship together.
        "experiment_id": turn_result.experiment_id,
        "task_id": turn_result.task_id,
        "attempt": (
            None if turn_result.attempt is None else int(turn_result.attempt)
        ),
        "claim_epoch": turn_result.claim_epoch,
        "server_incarnation": turn_result.server_incarnation,
        "record_version": 1,
        "record_json": json.dumps(record, ensure_ascii=False),
    }
    for column, classification in _POLICED_TURN_COLUMNS:
        turn_row[column] = _policed_column(
            policy, column, classification, turn_row[column]
        )
    return turn_row, artifact_rows


# ----------------------------------------------------------------------
# The store
# ----------------------------------------------------------------------

_SCHEMA_STATEMENTS = [
    # This literal is the ONLY creator of every table and column (fresh
    # schema, fix-49m.3): there is no ALTER/migration block anywhere, and a
    # store from an older build is replaced by _ensure_schema rather than
    # upgraded. experiment_id/task_id/attempt are the experiment container's
    # labels (`[XR4]`); NULL means "not part of an experiment".
    """CREATE TABLE IF NOT EXISTS conversations (
        channel_id TEXT NOT NULL, conversation_id INTEGER NOT NULL,
        topic TEXT, summary TEXT, status TEXT, next_ordinal INTEGER,
        started_at TEXT, last_turn_at TEXT, updated_at TEXT,
        experiment_id TEXT, task_id TEXT, attempt INTEGER,
        PRIMARY KEY (channel_id, conversation_id))""",
    """CREATE TABLE IF NOT EXISTS conversation_counters (
        channel_id TEXT PRIMARY KEY, next_id INTEGER NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS turns (
        turn_key TEXT PRIMARY KEY,
        channel_id TEXT NOT NULL, conversation_id INTEGER, ordinal INTEGER,
        user_message TEXT NOT NULL, refined_user_message TEXT,
        entry_workflow_name TEXT, entry_context TEXT,
        status TEXT NOT NULL, success INTEGER NOT NULL,
        failure_reason TEXT, answer TEXT,
        conversation_summary TEXT, conversation_traces TEXT,
        started_at TEXT, completed_at TEXT, suspended_ms INTEGER,
        continuation_of TEXT, record_version INTEGER NOT NULL,
        experiment_id TEXT, task_id TEXT, attempt INTEGER,
        claim_epoch INTEGER, server_incarnation TEXT,
        record_json TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS feedback (
        turn_key TEXT PRIMARY KEY, feedback_json TEXT NOT NULL,
        updated_at TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS human_feedback (
        feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
        turn_key TEXT NOT NULL REFERENCES turns(turn_key),
        target_kind TEXT NOT NULL, span_ids_json TEXT NOT NULL,
        target_label TEXT NOT NULL, comment TEXT NOT NULL,
        provenance TEXT NOT NULL,
        created_at TEXT NOT NULL)""",
    """CREATE INDEX IF NOT EXISTS idx_human_feedback_turn
        ON human_feedback(turn_key, feedback_id)""",
    """CREATE TRIGGER IF NOT EXISTS delete_turn_human_feedback
        AFTER DELETE ON turns BEGIN
        DELETE FROM human_feedback WHERE turn_key=OLD.turn_key;
        END""",
    """CREATE TABLE IF NOT EXISTS spans (
        span_id TEXT PRIMARY KEY, trace_id TEXT NOT NULL,
        parent_span_id TEXT, name TEXT NOT NULL,
        kind TEXT NOT NULL,
        channel_id TEXT,
        command_name TEXT, context TEXT,
        start_ns INTEGER NOT NULL, end_ns INTEGER,
        status TEXT NOT NULL, attributes TEXT NOT NULL,
        experiment_id TEXT, task_id TEXT, attempt INTEGER,
        claim_epoch INTEGER, server_incarnation TEXT)""",
    """CREATE TABLE IF NOT EXISTS artifacts (
        artifact_id TEXT PRIMARY KEY, turn_key TEXT NOT NULL,
        channel_id TEXT,
        span_id TEXT, key TEXT NOT NULL, content_type TEXT,
        size_bytes INTEGER, sha256 TEXT,
        inline_value BLOB, error TEXT,
        experiment_id TEXT, task_id TEXT, attempt INTEGER,
        claim_epoch INTEGER, server_incarnation TEXT)""",
    """CREATE TABLE IF NOT EXISTS train_runs (
        run_id TEXT PRIMARY KEY, workflow_fingerprint TEXT, started_at TEXT,
        completed_at TEXT, metrics_json TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS diagnostics (
        key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS experiments (
        experiment_id TEXT PRIMARY KEY,
        description TEXT NOT NULL,
        notes TEXT,
        arm TEXT,
        baseline_experiment_id TEXT,
        status TEXT NOT NULL,
        invalid_reason TEXT,
        invalid_detail TEXT,
        archived INTEGER NOT NULL DEFAULT 0,
        declared_tasks INTEGER NOT NULL,
        declared_attempts INTEGER NOT NULL,
        required_evidence_segments INTEGER NOT NULL DEFAULT 0,
        workspace_archive_sha256 TEXT,
        workspace_store_identity TEXT,
        evidence_sealed_at TEXT,
        benchmark_id TEXT,
        benchmark_version TEXT,
        benchmark_digest_sha256 TEXT,
        workflow_name TEXT,
        capture_profile TEXT NOT NULL,
        capture_policy_version TEXT NOT NULL,
        created_at TEXT NOT NULL,
        completed_at TEXT)""",
    """CREATE TABLE IF NOT EXISTS experiment_attempts (
        experiment_id TEXT NOT NULL,
        task_id TEXT NOT NULL,
        attempt INTEGER NOT NULL,
        channel_id TEXT NOT NULL,
        conversation_id INTEGER,
        outcome TEXT,
        outcome_source TEXT,
        reward REAL,
        restarts INTEGER NOT NULL DEFAULT 0,
        started_at TEXT NOT NULL,
        execution_status TEXT,
        execution_finished_at TEXT,
        finished_at TEXT,
        detail_json TEXT,
        source_attempt_json TEXT,
        source_key TEXT,
        runtime_snapshot_json TEXT,
        PRIMARY KEY (experiment_id, task_id, attempt))""",
    """CREATE TABLE IF NOT EXISTS experiment_attempt_declarations (
        experiment_id TEXT NOT NULL,
        task_id TEXT NOT NULL,
        native_attempt INTEGER NOT NULL,
        source_key TEXT NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY (experiment_id, task_id, native_attempt))""",
    """CREATE TABLE IF NOT EXISTS experiment_attempt_claims (
        registration_id TEXT PRIMARY KEY,
        experiment_id TEXT NOT NULL,
        task_id TEXT NOT NULL,
        native_attempt INTEGER NOT NULL,
        source_key TEXT NOT NULL,
        channel_id TEXT NOT NULL,
        secret_hash TEXT,
        expires_at REAL NOT NULL,
        state TEXT NOT NULL,
        epoch INTEGER NOT NULL DEFAULT 0,
        server_incarnation TEXT,
        lease_expires_at REAL,
        conversation_id INTEGER,
        recovery_json TEXT,
        created_at TEXT NOT NULL,
        claimed_at TEXT,
        UNIQUE (experiment_id, task_id, native_attempt))""",
    """CREATE TABLE IF NOT EXISTS experiment_evidence_runs (
        experiment_id TEXT NOT NULL,
        seq INTEGER NOT NULL,
        evidence_run_id TEXT NOT NULL,
        valid INTEGER NOT NULL,
        started_at TEXT,
        completed_at TEXT,
        record_json TEXT NOT NULL,
        PRIMARY KEY (experiment_id, seq))""",
    "CREATE INDEX IF NOT EXISTS idx_spans_trace ON spans(trace_id)",
    "CREATE INDEX IF NOT EXISTS idx_spans_command ON spans(command_name) WHERE command_name IS NOT NULL",
    "CREATE INDEX IF NOT EXISTS idx_turns_conv ON turns(channel_id, conversation_id, ordinal)",
    "CREATE INDEX IF NOT EXISTS idx_turns_status ON turns(status)",
    "CREATE INDEX IF NOT EXISTS idx_artifacts_turn ON artifacts(turn_key)",
    "CREATE INDEX IF NOT EXISTS idx_turns_experiment ON turns(experiment_id, task_id, attempt) WHERE experiment_id IS NOT NULL",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_conv_experiment_attempt ON conversations(experiment_id, task_id, attempt) WHERE experiment_id IS NOT NULL",
    "CREATE INDEX IF NOT EXISTS idx_experiments_baseline ON experiments(baseline_experiment_id) WHERE baseline_experiment_id IS NOT NULL",
    "CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_experiment_attempts_channel ON experiment_attempts(channel_id)",
    "CREATE INDEX IF NOT EXISTS idx_experiment_declarations_experiment ON experiment_attempt_declarations(experiment_id)",
    "CREATE INDEX IF NOT EXISTS idx_experiment_claims_channel ON experiment_attempt_claims(channel_id, state)",
    # Offload evidence (FEATURE_OFFLOAD_EVIDENCE_V1). Additive: a store created
    # before these existed gains them on its next open, and an older build
    # ignores tables it does not know. One row per archived observation of one
    # turn, holding the bytes as stored (redacted at write time unless
    # `redaction` says `off`) and the capture record that produced them.
    # `channel_id` is carried beside `turn_key`, as on `spans`/`artifacts`,
    # because a turn with no bound turn key is keyed by its channel id and
    # never appears in `turns`: erasure by channel reaches it through this
    # column. `scope_id` is the in-process scope digest, stored only so an
    # erasure can drop the process caches of that scope.
    """CREATE TABLE IF NOT EXISTS offload_evidence (
        turn_key TEXT NOT NULL,
        channel_id TEXT NOT NULL,
        scope_id TEXT NOT NULL,
        alias TEXT NOT NULL,
        offload_order INTEGER NOT NULL,
        command_name TEXT NOT NULL,
        step_index INTEGER NOT NULL,
        text_utf8 BLOB NOT NULL,
        text_sha256 TEXT NOT NULL,
        capture_policy_version TEXT NOT NULL,
        capture_profile TEXT NOT NULL,
        redaction TEXT NOT NULL,
        redacted INTEGER NOT NULL,
        raw_utf8_bytes INTEGER NOT NULL,
        persisted_at TEXT NOT NULL,
        PRIMARY KEY (turn_key, alias))""",
    # The context an observation is evidence about. Its own table because a
    # subject is recorded at DISPATCH, before the step completes and its
    # evidence row is written, and survives on its own when that write is
    # refused.
    """CREATE TABLE IF NOT EXISTS offload_subjects (
        turn_key TEXT NOT NULL,
        channel_id TEXT NOT NULL,
        scope_id TEXT NOT NULL,
        alias TEXT NOT NULL,
        context_clause TEXT NOT NULL,
        recorded_at TEXT NOT NULL,
        PRIMARY KEY (turn_key, alias))""",
    "CREATE INDEX IF NOT EXISTS idx_offload_evidence_channel ON offload_evidence(channel_id)",
    "CREATE INDEX IF NOT EXISTS idx_offload_evidence_age ON offload_evidence(persisted_at)",
    "CREATE INDEX IF NOT EXISTS idx_offload_subjects_channel ON offload_subjects(channel_id)",
    # Offload events (FEATURE_OFFLOAD_EVENTS_V1), additive like the evidence
    # tables above. One row per diagnostic event the offload runtime records:
    # search questions, model reasoning and answers among them, so
    # `event_json` is written through the same protection as evidence text
    # (`redaction` says which mode produced it). `kind` is the event's own
    # framework-chosen label, kept as a column so a reader can filter on it.
    """CREATE TABLE IF NOT EXISTS offload_events (
        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
        turn_key TEXT NOT NULL,
        channel_id TEXT NOT NULL,
        scope_id TEXT NOT NULL,
        kind TEXT NOT NULL,
        event_json TEXT NOT NULL,
        redaction TEXT NOT NULL,
        redacted INTEGER NOT NULL,
        recorded_at TEXT NOT NULL)""",
    "CREATE INDEX IF NOT EXISTS idx_offload_events_turn ON offload_events(turn_key, event_id)",
    "CREATE INDEX IF NOT EXISTS idx_offload_events_channel ON offload_events(channel_id)",
]

# The offload tables above, named once for the erasure and retention paths,
# each with the column that dates its rows.
_OFFLOAD_EVIDENCE_TABLES = ("offload_evidence", "offload_subjects", "offload_events")
_OFFLOAD_TABLE_TIMESTAMPS = {
    "offload_evidence": "persisted_at",
    "offload_subjects": "recorded_at",
    "offload_events": "recorded_at",
}
# How many turns one retention batch drops from the offload evidence tables.
_OFFLOAD_PRUNE_BATCH_TURNS = 25


def _remove_legacy_offload_sidecar(db_path: str) -> list[str]:
    """Delete the evidence file older builds kept beside this DB, if any.

    The old sidecar's evidence is deliberately NOT imported: it predates
    turn-scoped erasure and write-time redaction, so it is removed together
    with its WAL files and the ``.preserve`` sentinel that used to exempt it.
    Best effort and never fatal -- a file that cannot be removed is logged
    and the store opens anyway. Returns the paths that were removed.
    """
    sidecar = f"{db_path}{LEGACY_OFFLOAD_SIDECAR_SUFFIX}"
    removed: list[str] = []
    for path in (sidecar, f"{sidecar}-wal", f"{sidecar}-shm", f"{sidecar}.preserve"):
        try:
            os.remove(path)
        except FileNotFoundError:
            continue
        except OSError as error:
            logger.warning(
                f"could not remove legacy offload evidence file {path}: {error}"
            )
            continue
        removed.append(path)
    if removed:
        logger.info(
            f"removed legacy offload evidence sidecar beside {db_path}: "
            f"{', '.join(removed)}"
        )
    return removed


def _present_offload_tables(conn: sqlite3.Connection) -> tuple[str, ...]:
    """The offload evidence tables this DB actually has.

    Every store this build opens with its schema ensured has both; a DB opened
    without that step (``open_for_annotation``) may predate them, and erasure
    must not fail on a table that was never there to hold anything.
    """
    marks = ",".join("?" for _ in _OFFLOAD_EVIDENCE_TABLES)
    found = {
        str(row[0])
        for row in conn.execute(
            f"SELECT name FROM sqlite_master WHERE type='table' AND name IN ({marks})",
            _OFFLOAD_EVIDENCE_TABLES,
        ).fetchall()
    }
    return tuple(table for table in _OFFLOAD_EVIDENCE_TABLES if table in found)


class ObservabilityStore:
    """Schema owner + synchronous operations on one observability DB.

    Thread/process-safe by construction: every method opens its own
    short-lived WAL connection (timeout=30, ``BEGIN IMMEDIATE`` for writes).
    """

    def __init__(self, db_path: str, *, migrate: bool = True) -> None:
        self.db_path = db_path
        if migrate:
            self._ensure_schema()
        self._features = self._load_features()

    @staticmethod
    def open_for_annotation(db_path: str) -> "ObservabilityStore":
        """Open an existing DB read-write without creating or migrating it."""
        return ObservabilityStore(db_path, migrate=False)

    def _store_redactor(self) -> Redactor:
        redactor = getattr(self, "_redactor", None)
        if redactor is None:
            redactor = Redactor()
            self._redactor = redactor
        return redactor

    def _store_capture_policy(self) -> "capture_policy_module.CapturePolicy":
        """This store's capture profile, resolved once.

        Resolved here as well as on the sink because the sync label path
        (`record_conversation_label`) reaches SQLite without a sink in sight,
        so it would otherwise write under no profile at all.
        """
        policy = getattr(self, "_capture_policy", None)
        if policy is None:
            policy = resolve_capture_policy()
            self._capture_policy = policy
        return policy

    # -- connections ----------------------------------------------------

    def _connect(self, timeout: float = 30.0) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=timeout, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        """Create or open the schema; replace a populated DB from an older build.

        The store has never shipped in a release, so a populated DB whose
        ``user_version`` is below ``SCHEMA_VERSION`` can only be a developer's
        local DB from an earlier revision. There is no migration, and refusing
        it only degrades everything that records into it, so it is deleted --
        with its ``-wal`` and ``-shm`` -- and a fresh store is created in its
        place. A DB from a NEWER build is refused and never touched. When the
        old files cannot be deleted, the older DB is refused as before.
        """
        try:
            self._ensure_schema_once()
            return
        except _OlderPopulatedStore as older:
            found = older.version
        if not self._delete_older_store(found):
            raise IncompatibleObservabilityDB(
                f"{self.db_path} has schema v{found}; this build requires "
                f"v{SCHEMA_VERSION}, carries no migration, and could not delete "
                "the older store to replace it. Move or delete the file and its "
                "-wal/-shm files to start a new store, or open it read-only with "
                f"a v{found} build."
            )
        logger.warning(
            f"Replaced observability store {self.db_path}: it had schema "
            f"v{found} from an older build and this build requires "
            f"v{SCHEMA_VERSION}, with no migration; its records were deleted."
        )
        try:
            self._ensure_schema_once()
        except _OlderPopulatedStore as again:
            raise IncompatibleObservabilityDB(
                f"{self.db_path} still has schema v{again.version} after it was "
                f"deleted for replacement; this build requires v{SCHEMA_VERSION}."
            ) from None

    def _delete_older_store(self, found: int) -> bool:
        """Delete an older-build DB so it can be recreated; ``False`` if it cannot be.

        The write-ahead log and shared-memory files go first and the main file
        last: a fresh DB must never meet a stale ``-wal`` from the old one, so
        if either companion cannot be removed the main file is left alone and
        the caller refuses the store instead.
        """
        for path in (f"{self.db_path}-wal", f"{self.db_path}-shm", self.db_path):
            try:
                os.remove(path)
            except FileNotFoundError:
                continue
            except OSError as error:
                logger.warning(
                    f"Could not delete {path} to replace an observability store "
                    f"with schema v{found}: {error}"
                )
                return False
        return True

    def _ensure_schema_once(self) -> None:
        parent = os.path.dirname(self.db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
            try:
                os.chmod(parent, 0o700)  # [R4]
            except OSError:
                pass
        # "Fresh" is robust to a file that was only touched: an empty file
        # has no header and no tables, and initialises exactly like a missing one.
        fresh = (
            not os.path.exists(self.db_path)
            or os.path.getsize(self.db_path) == 0
        )
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        try:
            if fresh:
                # auto_vacuum must be set at creation, before any table [R12].
                conn.execute("PRAGMA auto_vacuum=INCREMENTAL")

            # The version is read before the journal mode is touched, so a DB
            # this build will refuse or replace is not switched to WAL first.
            found = conn.execute("PRAGMA user_version").fetchone()[0]
            if found > SCHEMA_VERSION:
                raise IncompatibleObservabilityDB(
                    f"{self.db_path} has schema v{found}; this build reads up to "
                    f"v{SCHEMA_VERSION}. Refusing to open a newer DB [R11]."
                )
            if found < SCHEMA_VERSION:
                # A populated store from an older build is not migrated:
                # every column exists only in the CREATE TABLE literal (fresh
                # schema, fix-49m.3). `_ensure_schema` replaces it, after this
                # connection is closed. A fresh file (no tables yet) proceeds.
                has_tables = (
                    conn.execute(
                        "SELECT 1 FROM sqlite_master WHERE type='table' LIMIT 1"
                    ).fetchone()
                    is not None
                )
                if has_tables:
                    raise _OlderPopulatedStore(found)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            for statement in _SCHEMA_STATEMENTS:
                conn.execute(statement)
            if found < SCHEMA_VERSION:
                # Reached only on a fresh (table-less) file: stamp it.
                conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            # Write probe: every statement above is a no-op on an existing
            # schema, so an unwritable DB would otherwise open "successfully"
            # and fail on every later write. Fail here instead, so the factory
            # degrades to no-sink at open time.
            conn.execute(
                """INSERT INTO diagnostics (key, value, updated_at) VALUES (?, ?, ?)
                   ON CONFLICT(key) DO UPDATE SET
                     value=excluded.value, updated_at=excluded.updated_at""",
                ("schema_opened", json.dumps({"schema_version": SCHEMA_VERSION}), _utcnow_iso()),
            )
            self._merge_schema_features(
                conn,
                [
                    FEATURE_EXPERIMENTS_V1,
                    FEATURE_EXPERIMENT_LIFECYCLE_V1,
                    FEATURE_EXPERIMENT_DECLARATIONS_V1,
                    FEATURE_EXPERIMENT_CLAIMS_V1,
                    FEATURE_EXPERIMENT_SEALING_V1,
                    FEATURE_OFFLOAD_EVIDENCE_V1,
                    FEATURE_OFFLOAD_EVENTS_V1,
                ],
            )
            conn.execute(
                """INSERT INTO diagnostics (key, value, updated_at)
                   VALUES (?, ?, ?)
                   ON CONFLICT(key) DO NOTHING""",
                (
                    STORE_IDENTITY_DIAGNOSTIC,
                    str(uuid.uuid4()),
                    _utcnow_iso(),
                ),
            )
            conn.execute(
                """INSERT INTO diagnostics (key, value, updated_at)
                   VALUES (?, ?, ?)
                   ON CONFLICT(key) DO UPDATE SET
                     value=excluded.value, updated_at=excluded.updated_at""",
                (
                    CAPTURE_REGIME_DIAGNOSTIC,
                    json.dumps(
                        {
                            "capture_profile": _env(
                                CAPTURE_PROFILE_VAR, "debug"
                            ),
                            "capture_policy_version": CAPTURE_POLICY_VERSION,
                        }
                    ),
                    _utcnow_iso(),
                ),
            )
            conn.commit()
        finally:
            conn.close()
        try:
            os.chmod(self.db_path, 0o600)  # [R4]
            for companion in (f"{self.db_path}-wal", f"{self.db_path}-shm"):
                if os.path.exists(companion):
                    os.chmod(companion, 0o600)
        except OSError:
            pass
        _remove_legacy_offload_sidecar(self.db_path)

    @staticmethod
    def _merge_schema_features(
        conn: sqlite3.Connection, features: list[str]
    ) -> None:
        """Merge feature markers without dropping markers from other builds."""
        row = conn.execute(
            "SELECT value FROM diagnostics WHERE key='schema_features'"
        ).fetchone()
        known: list[str] = []
        if row is not None:
            try:
                loaded = json.loads(row[0])
                if isinstance(loaded, list):
                    known = [str(name) for name in loaded]
            except (ValueError, TypeError):
                known = []
        merged = sorted(set(known) | set(features))
        conn.execute(
            """INSERT INTO diagnostics (key, value, updated_at) VALUES (?, ?, ?)
               ON CONFLICT(key) DO UPDATE SET
                 value=excluded.value, updated_at=excluded.updated_at""",
            ("schema_features", json.dumps(merged), _utcnow_iso()),
        )

    def _load_features(self) -> frozenset[str]:
        """Read the feature markers this store's DB declares.

        The `schema_features` row is the only source. There is no
        column-sniffing fallback any more, and re-adding one would be a bug:
        under the fresh-schema rule every DB that reaches this
        method is at `SCHEMA_VERSION` — `ObservabilityStore` replaces an older
        DB and refuses a newer one, `ReadOnlyObservabilityStore` refuses both —
        and such
        a DB was created from the literal `_SCHEMA_STATEMENTS` with
        `_merge_schema_features` writing its markers in the same transaction. So the sniff could only ever re-derive what the row
        already says, and a store whose row is genuinely missing is one whose
        schema this build did not write: guessing its capabilities from column
        names is exactly the dual-shape reader the fresh-schema rule exists to
        forbid. Absent/unreadable therefore means "no features", not "go and
        look".
        """
        conn = None
        try:
            conn = self._connect(timeout=5.0)
            row = conn.execute(
                "SELECT value FROM diagnostics WHERE key='schema_features'"
            ).fetchone()
            if row is not None:
                loaded = json.loads(row[0])
                if isinstance(loaded, list):
                    return frozenset(str(name) for name in loaded)
            return frozenset()
        except Exception:
            return frozenset()
        finally:
            if conn is not None:
                with contextlib.suppress(Exception):
                    conn.close()

    def has_feature(self, name: str) -> bool:
        return name in self._features

    def experiment_declaration_schema_ready(self) -> bool:
        """Whether the advertised exact-plan schema is structurally complete."""
        if not self.has_feature(FEATURE_EXPERIMENT_DECLARATIONS_V1):
            return False
        try:
            with self._connect() as conn:
                declaration_cols = {
                    row[1]
                    for row in conn.execute(
                        "PRAGMA table_info(experiment_attempt_declarations)"
                    ).fetchall()
                }
                experiment_cols = {
                    row[1]
                    for row in conn.execute(
                        "PRAGMA table_info(experiments)"
                    ).fetchall()
                }
                attempt_cols = {
                    row[1]
                    for row in conn.execute(
                        "PRAGMA table_info(experiment_attempts)"
                    ).fetchall()
                }
            return {
                "experiment_id",
                "task_id",
                "native_attempt",
                "source_key",
            } <= declaration_cols and {
                "required_evidence_segments"
            } <= experiment_cols and {"source_key"} <= attempt_cols
        except sqlite3.Error:
            return False

    def experiment_claim_schema_ready(self) -> bool:
        """Whether one-use bootstrap and epoch fencing are installed."""
        if not self.has_feature(FEATURE_EXPERIMENT_CLAIMS_V1):
            return False
        try:
            with self._connect() as conn:
                columns = {
                    row[1]
                    for row in conn.execute(
                        "PRAGMA table_info(experiment_attempt_claims)"
                    ).fetchall()
                }
            return {
                "registration_id",
                "experiment_id",
                "task_id",
                "native_attempt",
                "source_key",
                "channel_id",
                "secret_hash",
                "expires_at",
                "state",
                "epoch",
                "server_incarnation",
                "conversation_id",
            } <= columns
        except sqlite3.Error:
            return False

    def experiment_sealing_schema_ready(self) -> bool:
        """Whether external captures can record their immutable archive handle."""
        if not self.has_feature(FEATURE_EXPERIMENT_SEALING_V1):
            return False
        try:
            with self._connect() as conn:
                columns = {
                    row[1]
                    for row in conn.execute(
                        "PRAGMA table_info(experiments)"
                    ).fetchall()
                }
            return {
                "workspace_archive_sha256",
                "workspace_store_identity",
                "evidence_sealed_at",
            } <= columns
        except sqlite3.Error:
            return False

    def capture_regime(self) -> Optional[tuple[str, str]]:
        """Return the regime installed by the process that owns this store."""
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT value FROM diagnostics WHERE key=?",
                    (CAPTURE_REGIME_DIAGNOSTIC,),
                ).fetchone()
            if row is None:
                return None
            value = json.loads(row["value"])
            return (
                str(value["capture_profile"]),
                str(value["capture_policy_version"]),
            )
        except (KeyError, TypeError, ValueError, sqlite3.Error):
            return None

    def store_identity(self) -> Optional[str]:
        """Return the durable identity minted when this store was installed."""
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT value FROM diagnostics WHERE key=?",
                    (STORE_IDENTITY_DIAGNOSTIC,),
                ).fetchone()
            if row is None or not str(row["value"]).strip():
                return None
            return str(row["value"])
        except sqlite3.Error:
            return None

    # -- identity [R1] ---------------------------------------------------

    def mint_conversation_id(
        self,
        channel_id: str,
        legacy_floor: int = 0,
        *,
        experiment_id: Optional[str] = None,
        task_id: Optional[str] = None,
        attempt: Optional[int] = None,
    ) -> int:
        """Atomically reserve the next conversation id for a channel.

        The observability DB is the sole id-minting authority; dual-write
        consumers (the legacy conversation store) consume the same id so the
        stores cannot diverge on identity.

        Minting is a per-channel monotonic counter (never MAX-derived), so
        forget-channel/prune can never cause id reuse; the counter is seeded
        at first mint from ``max(existing rows, legacy_floor)`` — callers
        crossing the Phase-7 cutover pass the legacy store's
        ``last_conversation_id`` as ``legacy_floor`` so ids never alias
        against pre-cutover conversations (review ruling C2).

        Uses a SHORT busy timeout (ruling C9's principle): minting runs
        synchronously in request paths — FastAPI's event loop included — so a
        contended DB must fail fast (callers degrade to the legacy reserve
        path) rather than stall every channel for the writer timeout.
        """
        with self._connect(
            timeout=float(_env_int("FW_OBS_SYNC_WRITE_TIMEOUT_S", 5))
        ) as conn:
            conn.execute("BEGIN IMMEDIATE")
            new_id = self._mint_conversation_id_in_txn(
                conn,
                channel_id,
                legacy_floor=legacy_floor,
                experiment_id=experiment_id,
                task_id=task_id,
                attempt=attempt,
            )
            conn.commit()
        return new_id

    def _mint_conversation_id_in_txn(
        self,
        conn: sqlite3.Connection,
        channel_id: str,
        legacy_floor: int = 0,
        *,
        experiment_id: Optional[str] = None,
        task_id: Optional[str] = None,
        attempt: Optional[int] = None,
    ) -> int:
        """Reserve a conversation using the caller's existing write transaction."""
        counter = conn.execute(
            "SELECT next_id FROM conversation_counters WHERE channel_id=?",
            (channel_id,),
        ).fetchone()
        max_row = conn.execute(
            "SELECT COALESCE(MAX(conversation_id), 0) FROM conversations WHERE channel_id=?",
            (channel_id,),
        ).fetchone()
        floor = max(int(max_row[0]), int(legacy_floor or 0))
        next_id = int(counter["next_id"]) if counter is not None else 1
        new_id = max(next_id, floor + 1)
        conn.execute(
            """INSERT INTO conversation_counters (channel_id, next_id) VALUES (?, ?)
               ON CONFLICT(channel_id) DO UPDATE SET
                 next_id=MAX(conversation_counters.next_id, excluded.next_id)""",
            (channel_id, new_id + 1),
        )
        now = _utcnow_iso()
        conn.execute(
            """INSERT INTO conversations
               (channel_id, conversation_id, topic, summary, status,
                next_ordinal, started_at, last_turn_at, updated_at,
                experiment_id, task_id, attempt)
               VALUES (?, ?, NULL, NULL, 'open', 1, ?, NULL, ?, ?, ?, ?)""",
            (
                channel_id,
                new_id,
                now,
                now,
                experiment_id,
                self._store_redactor().redact(task_id),
                None if attempt is None else int(attempt),
            ),
        )
        return new_id

    def record_conversation_label(
        self,
        channel_id: str,
        conversation_id: int,
        topic: Optional[str],
        summary: Optional[str],
    ) -> str:
        """Upsert a conversation's topic/summary; labels are mutable.

        A None topic or summary preserves the stored value, so the blank-topic
        policy — a failed generation never clobbers a good title — carries
        over from the legacy store. Topic uniquification runs inside the same
        transaction as the write, so there is no TOCTOU across the async
        label path; casefold is done Python-side, never with SQLite's
        ASCII-only lower().

        Returns the topic actually STORED — collision-suffixed where one was
        written, or the preserved existing title on a blank generation. A
        caller that reports or logs the label must use this rather than its own
        candidate, which is the contract the legacy store's
        ``update_conversation_topic_summary`` established.
        """
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            stored = self.apply_label_txn(
                conn, channel_id, conversation_id, topic, summary
            )
            conn.commit()
        return stored

    def apply_label_txn(
        self,
        conn: sqlite3.Connection,
        channel_id: str,
        conversation_id: int,
        topic: Optional[str],
        summary: Optional[str],
    ) -> str:
        """The single label-write enforcement point (caller owns the txn).

        Returns the stored topic (see ``record_conversation_label``).
        """
        if topic is not None:
            topic = self._unique_topic_in_txn(
                conn, channel_id, topic, exclude_conversation_id=conversation_id
            )
            if not topic:
                # Blank stays the "no title yet" sentinel — never stored as a
                # title (legacy blank-topic policy).
                topic = None
        # fix-ajv.9 item 5, the one live gap: BOTH layers, applied here because
        # this is the single label-write enforcement point and the production
        # route to it — run_fastapi_mcp/utils.ensure_topic_and_summary calling
        # `record_conversation_label` — is the SYNC one, which never touches
        # `SQLiteTraceSink._apply_label` and so never met the credential scrub
        # either. Protecting the enforcement point rather than the two callers is
        # what makes it impossible to add a third route that skips this.
        #
        # `user-text` and not `controlled-vocabulary`: a topic and a summary are
        # LLM output generated FROM a real user's conversation, so their content
        # is whatever the conversation was about — an order number, a name, an
        # address. Under `evidence` they become badges; the UI degrades to
        # "a 34-byte user-text title was here", which is §12.0 delta 3's
        # requirement and is why this is not simply an omission.
        #
        # AFTER uniquification, not before. `_unique_topic_in_txn` compares
        # casefolded titles and appends " 1", " 2" on collision: policing first
        # would append that suffix outside the envelope's closing brace and leave
        # a column holding text that no longer parses as JSON. The cost is that
        # collision suffixing stops distinguishing anything under `evidence`,
        # where two identical titles digest identically — acceptable, because
        # uniquification exists so a human can pick a conversation out of a list
        # by its title, and under `evidence` every title in that list is a badge.
        redactor = self._store_redactor()
        policy = self._store_capture_policy()
        topic = _protected_text(
            topic,
            redactor=redactor,
            policy=policy,
            field_path=POLICY_PATH_CONVERSATION_TOPIC,
            classification="user-text",
        )
        summary = _protected_text(
            summary,
            redactor=redactor,
            policy=policy,
            field_path=POLICY_PATH_CONVERSATION_SUMMARY,
            classification="user-text",
        )
        now = _utcnow_iso()
        conn.execute(
            """INSERT INTO conversations
               (channel_id, conversation_id, topic, summary, status,
                next_ordinal, started_at, last_turn_at, updated_at)
               VALUES (?, ?, ?, ?, 'open', 1, ?, NULL, ?)
               ON CONFLICT(channel_id, conversation_id) DO UPDATE SET
                 topic=COALESCE(excluded.topic, conversations.topic),
                 summary=COALESCE(excluded.summary, conversations.summary),
                 updated_at=excluded.updated_at""",
            (channel_id, conversation_id, topic, summary, now, now),
        )
        if topic is not None:
            return topic
        row = conn.execute(
            "SELECT topic FROM conversations WHERE channel_id=? AND conversation_id=?",
            (channel_id, conversation_id),
        ).fetchone()
        return (row["topic"] or "") if row is not None else ""

    @staticmethod
    def _topic_norm(value: str) -> str:
        # Python casefolding — SQLite lower() is ASCII-only (ruling I9).
        return value.casefold().strip()

    def _unique_topic_in_txn(
        self,
        conn: sqlite3.Connection,
        channel_id: str,
        candidate_topic: str,
        exclude_conversation_id: Optional[int] = None,
    ) -> str:
        """Legacy-faithful uniquification: case/whitespace-insensitive
        collision suffixing, blank exemption decided before the scan,
        self-exclusion, each suffixed candidate renormalized."""
        if not self._topic_norm(candidate_topic):
            return ""
        rows = conn.execute(
            "SELECT conversation_id, topic FROM conversations "
            "WHERE channel_id=? AND topic IS NOT NULL",
            (channel_id,),
        ).fetchall()
        existing = {
            self._topic_norm(row["topic"])
            for row in rows
            if row["conversation_id"] != exclude_conversation_id and row["topic"]
        }
        final_topic = candidate_topic
        collision_count = 0
        while self._topic_norm(final_topic) in existing:
            collision_count += 1
            final_topic = f"{candidate_topic} {collision_count}"
        return final_topic

    # -- writes (used by the writer thread; also callable directly) ------

    def upsert_span_rows(self, conn: sqlite3.Connection, spans: list[tracing.Span], redactor: Redactor) -> None:
        # fix-ajv.9 item 4: the scalar columns beside `attributes`. The
        # attributes JSON has been scrubbed since [R20]; the four scalars written
        # next to it never were, and one of them can carry entity content.
        #
        # `context` is `workflow.current_command_context_displayname`, which calls
        # a workflow-supplied `get_displayname(instance)` hook — the bundled
        # simple_workflow_template returns the work item's absolute path from it.
        # So it is not a type name, it is a label about a specific instance:
        # `user-text`, and withheld under `evidence`.
        #
        # `name` and `command_name` are closed vocabularies — the span taxonomy in
        # tracing.py and the workflow's own command set — so they are declared
        # rather than withheld, which is what FW-REQ-002 clause 3 asks for. The
        # `controlled-vocabulary` default bounds them at 256 bytes and passes
        # anything shorter through untouched, so this is inert for every real
        # command name while still refusing to let an unbounded value in.
        #
        # `channel_id` is SCRUB-ONLY, and deliberately so. It is an identifier, so
        # the evidence default would digest it, and a digest still joins — but
        # `forget_channel` erases a channel with `DELETE FROM spans WHERE
        # channel_id=?`, so digesting this column would silently narrow
        # first-class erasure [R21] to whatever the `trace_id IN (...)` fallback
        # happens to still cover. Reducing exposure by weakening erasure is not a
        # trade this slice gets to make; a joinable pseudonym applied to
        # turns/artifacts/spans at once, with `forget_channel` taught to match it,
        # is the real fix and is follow-up work.
        policy = self._store_capture_policy()
        for span in spans:
            claim = {
                "experiment_id": span.experiment_id,
                "task_id": span.task_id,
                "attempt": span.attempt,
                "epoch": span.claim_epoch,
                "server_incarnation": span.server_incarnation,
            }
            if span.experiment_id and self._reject_stale_claim_in_txn(
                conn, claim, "span"
            ):
                continue
            attributes = redactor.redact(
                json.dumps(_sanitize_json_value(span.attributes), ensure_ascii=False)
            )
            span_name = _protected_text(
                span.name,
                redactor=redactor,
                policy=policy,
                field_path=POLICY_PATH_SPAN_NAME,
                classification="controlled-vocabulary",
            )
            command_name = _protected_text(
                span.command_name,
                redactor=redactor,
                policy=policy,
                field_path=POLICY_PATH_SPAN_COMMAND_NAME,
                classification="controlled-vocabulary",
            )
            context = _protected_text(
                span.context,
                redactor=redactor,
                policy=policy,
                field_path=POLICY_PATH_SPAN_CONTEXT,
                classification="user-text",
            )
            # `Redactor.redact` returns a falsy input unchanged, so a None
            # channel_id stays None rather than becoming "".
            channel_id = redactor.redact(span.channel_id)
            conn.execute(
                """INSERT INTO spans
                   (span_id, trace_id, parent_span_id, name, kind, channel_id,
                    command_name, context, start_ns, end_ns, status, attributes,
                    experiment_id, task_id, attempt, claim_epoch, server_incarnation)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(span_id) DO UPDATE SET
                     claim_epoch=CASE
                       WHEN excluded.claim_epoch > COALESCE(spans.claim_epoch, -1)
                       THEN excluded.claim_epoch ELSE spans.claim_epoch END,
                     server_incarnation=CASE
                       WHEN excluded.claim_epoch >= COALESCE(spans.claim_epoch, -1)
                       THEN excluded.server_incarnation ELSE spans.server_incarnation END,
                     end_ns=COALESCE(excluded.end_ns, spans.end_ns),
                     status=CASE WHEN excluded.end_ns IS NOT NULL OR spans.end_ns IS NULL
                                 THEN excluded.status ELSE spans.status END,
                     attributes=CASE WHEN excluded.end_ns IS NOT NULL OR spans.end_ns IS NULL
                                     THEN excluded.attributes ELSE spans.attributes END,
                     command_name=COALESCE(excluded.command_name, spans.command_name),
                     context=COALESCE(excluded.context, spans.context)""",
                (
                    span.span_id,
                    span.trace_id,
                    span.parent_span_id,
                    span_name,
                    span.kind,
                    channel_id,
                    command_name,
                    context,
                    span.start_ns,
                    span.end_ns,
                    span.status,
                    attributes,
                    span.experiment_id,
                    span.task_id,
                    span.attempt,
                    span.claim_epoch,
                    span.server_incarnation,
                ),
            )

    def upsert_turn_row(
        self,
        conn: sqlite3.Connection,
        turn_row: dict[str, Any],
        artifact_rows: list[dict[str, Any]],
        redactor: Redactor,
    ) -> bool:
        """Apply the turn-row lifecycle: INSERT at first emission; one guarded
        status transition to a terminal status; write-once for rows already
        terminal (identical-content retries claim idempotent success).

        Returns False when a conflicting write against a terminal row was
        refused (counted by the caller).
        """
        turn_row = dict(turn_row)
        claim = {
            "experiment_id": turn_row.get("experiment_id"),
            "task_id": turn_row.get("task_id"),
            "attempt": turn_row.get("attempt"),
            "epoch": turn_row.get("claim_epoch"),
            "server_incarnation": turn_row.get("server_incarnation"),
        }
        if turn_row.get("experiment_id") and self._reject_stale_claim_in_txn(
            conn, claim, "turn"
        ):
            return False
        # failure_reason is included because it can embed exception/provider
        # text (e.g. a LiteLLM AuthenticationError body) — the [R20] scenario.
        # task_id is SCRUB-ONLY and not policed (`[XR6]`/`[XR7]`): policing it
        # would withhold nothing (the plaintext rides into record_json above,
        # which `_apply_capture_policy` never walks) while breaking every
        # equality lookup the experiment read layer is built on. It must be
        # scrubbed on BOTH label routes -- here and in mint_conversation_id --
        # and in the container tables, or the copies stop being joinable.
        #
        # `experiment_id` is deliberately NOT in this list. It is a machine-minted
        # opaque id (`exp-<32 hex>`, `[XR1]`) and the join key of every score, and
        # it is stored raw in `experiments`/`experiment_attempts`/
        # `experiment_evidence_runs`. Scrubbing it here and not there is what
        # makes a join silently return nothing -- the same class of defect the
        # scrub-on-both-routes rule above exists to prevent. Every other
        # machine-minted join key in this file (turn_key, trace_id, run_id,
        # artifact_id) is likewise stored as-is.
        for text_col in (
            "user_message",
            "refined_user_message",
            "answer",
            "failure_reason",
            "conversation_summary",
            "conversation_traces",
            "task_id",
            "record_json",
        ):
            if turn_row.get(text_col):
                turn_row[text_col] = redactor.redact(turn_row[text_col])

        existing = conn.execute(
            "SELECT status, record_json FROM turns WHERE turn_key=?",
            (turn_row["turn_key"],),
        ).fetchone()

        if existing is not None and existing["status"] in TERMINAL_TURN_STATUSES:
            if (
                existing["status"] == turn_row["status"]
                and existing["record_json"] == turn_row["record_json"]
            ):
                return True  # idempotent retry
            if turn_row["status"] not in TERMINAL_TURN_STATUSES:
                # A late-arriving pre-terminal emission (e.g. the queued
                # awaiting_user record draining after the terminal sync write)
                # is expected ordering noise, not a violation — ignore it
                # without counting (ruling C8).
                return True
            logger.warning(
                f"Refusing write to terminal turn row {turn_row['turn_key']} "
                f"(stored {existing['status']}, incoming {turn_row['status']}) [R2]"
            )
            return False

        # Ordinal assignment on first insert of a conversation-bound turn.
        if (
            existing is None
            and turn_row.get("conversation_id") is not None
            and turn_row.get("ordinal") is None
        ):
            turn_row["ordinal"] = self._assign_ordinal(
                conn,
                turn_row["channel_id"],
                turn_row["conversation_id"],
                experiment_id=turn_row.get("experiment_id"),
                task_id=turn_row.get("task_id"),
                attempt=turn_row.get("attempt"),
            )

        columns = list(turn_row.keys())
        placeholders = ", ".join("?" for _ in columns)
        update_cols = [c for c in columns if c != "turn_key"]
        if existing is not None:
            # Keep the ordinal assigned at first insert.
            update_cols = [c for c in update_cols if c != "ordinal"]
        assignments = ", ".join(f"{c}=excluded.{c}" for c in update_cols)
        conn.execute(
            f"INSERT INTO turns ({', '.join(columns)}) VALUES ({placeholders}) "
            f"ON CONFLICT(turn_key) DO UPDATE SET {assignments}",
            [turn_row[c] for c in columns],
        )

        if turn_row.get("conversation_id") is not None:
            now = _utcnow_iso()
            conn.execute(
                """UPDATE conversations SET last_turn_at=?, updated_at=?
                   WHERE channel_id=? AND conversation_id=?""",
                (now, now, turn_row["channel_id"], turn_row["conversation_id"]),
            )

        for artifact in artifact_rows:
            artifact_claim = {
                "experiment_id": artifact.get("experiment_id"),
                "task_id": artifact.get("task_id"),
                "attempt": artifact.get("attempt"),
                "epoch": artifact.get("claim_epoch"),
                "server_incarnation": artifact.get("server_incarnation"),
            }
            if artifact.get("experiment_id") and self._reject_stale_claim_in_txn(
                conn, artifact_claim, "artifact"
            ):
                continue
            inline_value = artifact.get("inline_value")
            if isinstance(inline_value, (bytes, bytearray)):
                redacted = redactor.redact(
                    bytes(inline_value).decode("utf-8", errors="replace")
                )
                inline_value = redacted.encode("utf-8")
            conn.execute(
                """INSERT INTO artifacts
                   (artifact_id, turn_key, channel_id, span_id, key, content_type,
                    size_bytes, sha256, inline_value, error, experiment_id,
                    task_id, attempt, claim_epoch, server_incarnation)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(artifact_id) DO NOTHING""",
                (
                    artifact["artifact_id"],
                    artifact["turn_key"],
                    artifact.get("channel_id"),
                    artifact.get("span_id"),
                    artifact["key"],
                    artifact.get("content_type"),
                    artifact.get("size_bytes"),
                    artifact.get("sha256"),
                    inline_value,
                    artifact.get("error"),
                    artifact.get("experiment_id"),
                    artifact.get("task_id"),
                    artifact.get("attempt"),
                    artifact.get("claim_epoch"),
                    artifact.get("server_incarnation"),
                ),
            )
        return True

    def _assign_ordinal(
        self,
        conn: sqlite3.Connection,
        channel_id: str,
        conversation_id: int,
        *,
        experiment_id: Optional[str] = None,
        task_id: Optional[str] = None,
        attempt: Optional[int] = None,
    ) -> int:
        row = conn.execute(
            "SELECT next_ordinal FROM conversations WHERE channel_id=? AND conversation_id=?",
            (channel_id, conversation_id),
        ).fetchone()
        if row is None:
            # Conversation row not minted here (e.g. restored session) —
            # create it so ordinals stay dense from 1.
            # The labels are copied off the turn row being inserted: this row
            # was not minted here (restored session, or a turn whose conversation
            # predates the experiment binding), so the turn is the only carrier.
            #
            # Scrubbed HERE rather than trusting the caller: three routes reach
            # this insert (`upsert_turn_row`'s text loop, which has scrubbed;
            # `reserve_turn_ordinal` from the sink's degraded queue path, which
            # has not; and a direct call), and a value scrubbed on one route and
            # not another is what makes the turns/conversations join silently
            # return nothing. The scrub is idempotent, so doing it again is free.
            conn.execute(
                """INSERT INTO conversations
                   (channel_id, conversation_id, topic, summary, status,
                    next_ordinal, started_at, last_turn_at,
                    experiment_id, task_id, attempt)
                   VALUES (?, ?, NULL, NULL, 'open', 2, ?, NULL, ?, ?, ?)""",
                (
                    channel_id,
                    conversation_id,
                    _utcnow_iso(),
                    experiment_id,
                    self._store_redactor().redact(task_id),
                    None if attempt is None else int(attempt),
                ),
            )
            return 1
        ordinal = int(row["next_ordinal"] or 1)
        conn.execute(
            "UPDATE conversations SET next_ordinal=? WHERE channel_id=? AND conversation_id=?",
            (ordinal + 1, channel_id, conversation_id),
        )
        return ordinal

    def reserve_turn_ordinal(
        self,
        channel_id: str,
        conversation_id: int,
        *,
        experiment_id: Optional[str] = None,
        task_id: Optional[str] = None,
        attempt: Optional[int] = None,
    ) -> Optional[int]:
        """Reserve a turn ordinal in a tiny standalone transaction.

        Used by the sync-first emit's degraded fallback so ordinals stay
        chronological even when the row itself is queued (ruling I6).
        Returns None when the reservation itself cannot be made.
        """
        try:
            with self._connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                # The labels ride along because `_assign_ordinal` CREATES the
                # conversations row when it is missing: reserving without them
                # would mint an unlabelled attempt conversation on the degraded
                # path, and the UNIQUE index would then refuse the labelled one.
                ordinal = self._assign_ordinal(
                    conn,
                    channel_id,
                    conversation_id,
                    experiment_id=experiment_id,
                    task_id=task_id,
                    attempt=attempt,
                )
                conn.commit()
            return ordinal
        except Exception:
            return None

    # -- consolidation reads (Phase 7; "usable rows" filter per ruling I4) --
    #
    # A turns row exists for every logical turn — cancelled turns, abandoned
    # suspensions, and turns whose history never grew carry a NULL
    # conversation_summary. Conversation-memory consumers must therefore see
    # only rows that correspond to a real conversation-history entry:
    _USABLE_TURN_FILTER = (
        "status IN ('completed','failed') AND conversation_summary IS NOT NULL"
    )

    def count_usable_turns(self, channel_id: str, conversation_id: int) -> int:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT COUNT(*) FROM turns WHERE channel_id=? AND conversation_id=? "
                f"AND {self._USABLE_TURN_FILTER}",
                (channel_id, conversation_id),
            ).fetchone()
            return int(row[0])

    def get_memory_window(
        self, channel_id: str, conversation_id: int, max_turns: int
    ) -> list[dict[str, Any]]:
        """The newest ``max_turns`` usable turns as canonical 3-key memory
        dicts (oldest-first), feedback joined in — the read that
        replaces the legacy ``get_conversation_window``."""
        with self._connect() as conn:
            rows = conn.execute(
                f"""SELECT t.conversation_summary, t.conversation_traces, f.feedback_json
                    FROM turns t LEFT JOIN feedback f ON f.turn_key = t.turn_key
                    WHERE t.channel_id=? AND t.conversation_id=?
                      AND {self._USABLE_TURN_FILTER}
                    ORDER BY t.ordinal DESC, t.turn_key DESC LIMIT ?""",
                (channel_id, conversation_id, max_turns),
            ).fetchall()
        window = []
        for row in reversed(rows):
            feedback = None
            if row["feedback_json"]:
                try:
                    feedback = json.loads(row["feedback_json"])
                except ValueError:
                    feedback = row["feedback_json"]
            window.append(
                {
                    "conversation summary": row["conversation_summary"],
                    "conversation_traces": row["conversation_traces"],
                    "feedback": feedback,
                }
            )
        return window

    def conversation_summaries(
        self, channel_id: str, conversation_id: int
    ) -> list[dict[str, Any]]:
        """Each usable turn's summary, in order (labeling input)."""
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT conversation_summary FROM turns "
                f"WHERE channel_id=? AND conversation_id=? AND {self._USABLE_TURN_FILTER} "
                f"ORDER BY ordinal, turn_key",
                (channel_id, conversation_id),
            ).fetchall()
            return [{"conversation summary": r["conversation_summary"]} for r in rows]

    def conversation_label_state(
        self, channel_id: str, conversation_id: int
    ) -> tuple[str, int]:
        """(stored topic or '', usable turn count) — the lazy-label trigger's
        one read (legacy ``get_conversation_label_state`` parity)."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT topic FROM conversations WHERE channel_id=? AND conversation_id=?",
                (channel_id, conversation_id),
            ).fetchone()
        return (
            (row["topic"] or "") if row is not None else "",
            self.count_usable_turns(channel_id, conversation_id),
        )

    def newest_conversation_ids(self, channel_id: str, limit: int = 2) -> list[int]:
        """Newest conversation ids for a channel (restore + step-back)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT conversation_id FROM conversations WHERE channel_id=? "
                "ORDER BY conversation_id DESC LIMIT ?",
                (channel_id, limit),
            ).fetchall()
            return [int(r[0]) for r in rows]

    def get_last_completed_turn_key(
        self, channel_id: str, conversation_id: int
    ) -> Optional[str]:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT turn_key FROM turns WHERE channel_id=? AND conversation_id=? "
                f"AND {self._USABLE_TURN_FILTER} ORDER BY ordinal DESC, turn_key DESC LIMIT 1",
                (channel_id, conversation_id),
            ).fetchone()
            return row["turn_key"] if row is not None else None

    def list_conversation_summaries(
        self, channel_id: str, limit: int
    ) -> list[dict[str, Any]]:
        """/conversations projection (ruling C7): only conversations with at
        least one usable turn (no reserved-but-empty phantoms), NULLs
        projected to '', timestamps as ms epoch, ordered by updated_at desc."""
        with self._connect() as conn:
            rows = conn.execute(
                f"""SELECT c.conversation_id, c.topic, c.summary, c.started_at,
                           COALESCE(c.updated_at, c.last_turn_at, c.started_at) AS updated_at
                    FROM conversations c
                    WHERE c.channel_id=? AND EXISTS (
                        SELECT 1 FROM turns t
                        WHERE t.channel_id=c.channel_id
                          AND t.conversation_id=c.conversation_id
                          AND {self._USABLE_TURN_FILTER})
                    ORDER BY updated_at DESC LIMIT ?""",
                (channel_id, limit),
            ).fetchall()
        return [
            {
                "conversation_id": int(r["conversation_id"]),
                "topic": r["topic"] or "",
                "summary": r["summary"] or "",
                "created_at": _iso_to_ms(r["started_at"]),
                "updated_at": _iso_to_ms(r["updated_at"]),
            }
            for r in rows
        ]

    def dump_all_conversations(self, channel_id: str) -> list[dict[str, Any]]:
        """Admin-dump reconstruction of the hydrated legacy shape (ruling C7):
        one object per conversation with 3-key turns (+feedback) inlined."""
        dumped = []
        for conv in self.list_conversation_summaries(channel_id, limit=1_000_000):
            conv_id = conv["conversation_id"]
            dumped.append(
                {
                    "channel_id": channel_id,
                    "conversation_id": conv_id,
                    "topic": conv["topic"],
                    "summary": conv["summary"],
                    "created_at": conv["created_at"],
                    "updated_at": conv["updated_at"],
                    "turns": self.get_memory_window(channel_id, conv_id, 1_000_000),
                }
            )
        return dumped

    def upsert_feedback(self, turn_key: str, feedback_json: str) -> None:
        """Upsert a turn's feedback. Credential-scrubbed, NOT policy-withheld.

        The one column where the two protection layers disagree.
        The scrub applies for the same reason it applies everywhere: it is
        unconditional, and `nl_feedback` is free text a user typed, which is a
        place a pasted token lands. Scrubbing serialized JSON cannot corrupt it —
        every credential pattern is confined to characters that cannot appear
        unescaped inside a JSON string, so a replacement can never cross a
        delimiter (pinned by test).
        WHY NO CAPTURE POLICY: this column is read by `get_memory_window`,
        which passes the parsed value straight into `dspy.History` through
        `conversation_history_io.restore_history_from_turns` — it is the agent's
        memory of being corrected, not evidence about the agent. Under `evidence`
        a withheld value would still parse, so the agent would silently receive a
        badge dict where its feedback used to be and behave differently. That is a
        behavior change, not a reduction in exposure, and this layer does not make
        those; it is the same call `_POLICY_EXEMPT_TURN_COLUMNS` records for
        `conversation_summary` and `conversation_traces`, and it belongs with
        conversation-memory redaction generally, which has to leave memory usable.
        """
        feedback_json = self._store_redactor().redact(feedback_json)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """INSERT INTO feedback (turn_key, feedback_json, updated_at)
                   VALUES (?, ?, ?)
                   ON CONFLICT(turn_key) DO UPDATE SET
                     feedback_json=excluded.feedback_json, updated_at=excluded.updated_at""",
                (turn_key, feedback_json, _utcnow_iso()),
            )
            conn.commit()

    def get_feedback(self, turn_key: str) -> Optional[dict[str, Any]]:
        """Return the stored agent-memory feedback for one turn, unchanged.

        This deliberately queries ``feedback`` directly. Feedback remains
        readable when its turn has no conversation summary and is therefore
        excluded from the conversation-memory window.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT turn_key, feedback_json, updated_at "
                "FROM feedback WHERE turn_key=?",
                (turn_key,),
            ).fetchone()
        return dict(row) if row is not None else None

    def list_feedback(
        self, channel_id: Optional[str] = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        """List stored agent-memory feedback without interpreting verdicts."""
        if limit < 0:
            raise ValueError("limit must be non-negative")
        query = (
            "SELECT f.turn_key, f.feedback_json, f.updated_at, t.channel_id "
            "FROM feedback f LEFT JOIN turns t ON t.turn_key=f.turn_key"
        )
        params: list[Any] = []
        if channel_id is not None:
            query += " WHERE t.channel_id=?"
            params.append(channel_id)
        query += " ORDER BY f.updated_at DESC, f.turn_key DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            return [dict(row) for row in conn.execute(query, params).fetchall()]

    def record_train_run(
        self,
        run_id: str,
        workflow_fingerprint: Optional[str],
        started_at: Optional[str],
        completed_at: Optional[str],
        metrics: dict[str, Any],
    ) -> None:
        """Persist one training run's metrics at publication time (Phase 6).

        BOTH protection layers apply, classified `opaque-payload`.

        Not `controlled-vocabulary`, which is what a dict of thresholds and
        F1 scores looks like from the outside. `collect_train_metrics` assembles this
        by reading whatever JSON is sitting in `___command_info`, and one of those
        files carries free text: `heldout_evaluation.EscalationScore.failures`
        records the verbatim `utterance` of every case that failed, and
        `metrics_persistence` copies the whole `escalation` block through. Those
        utterances are synthetic today, but "nobody can enumerate what is in
        here" is the definition of `opaque-payload`, and default-deny exists for
        precisely the field whose contents grow when someone edits a file
        elsewhere.

        An evidence deployment that has reviewed its metrics and wants them in the
        bundle re-admits them by name — a `CaptureFieldPolicy` on
        `POLICY_PATH_TRAIN_METRICS` with `redact_before_trace=False`. The other
        four columns of the row (run_id, fingerprint, timestamps) are unpoliced,
        so a bundle always knows a training run happened and which sources it was
        built from, even when the metrics themselves are a badge.
        """
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """INSERT INTO train_runs
                   (run_id, workflow_fingerprint, started_at, completed_at, metrics_json)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(run_id) DO UPDATE SET
                     workflow_fingerprint=excluded.workflow_fingerprint,
                     started_at=excluded.started_at,
                     completed_at=excluded.completed_at,
                     metrics_json=excluded.metrics_json""",
                (
                    run_id,
                    workflow_fingerprint,
                    started_at,
                    completed_at,
                    _protected_text(
                        json.dumps(_sanitize_json_value(metrics), ensure_ascii=False),
                        redactor=self._store_redactor(),
                        policy=self._store_capture_policy(),
                        field_path=POLICY_PATH_TRAIN_METRICS,
                        classification="opaque-payload",
                    ),
                ),
            )
            conn.commit()

    def set_diagnostic(self, conn: sqlite3.Connection, key: str, value: dict[str, Any]) -> None:
        """Upsert one diagnostics row. Credential-scrubbed, NOT policy-withheld.

        The scrub earns its place here more than anywhere else:
        `writer_health.last_error` is `repr(exc)`, and the scenario that
        motivated the redactor in the first place is a LiteLLM
        `AuthenticationError` whose body echoes the key.

        WHY NO CAPTURE POLICY: this table is not a record of the workload, it is
        the record of whether the record can be trusted. `health_delta` and
        `evidence_run` read `writer_health` to decide whether a run may be
        reported as evidence at all, and `WriterHealthDelta.problems()` names the
        affected turn keys so a partly-damaged run can be salvaged instead of
        discarded. Withholding it under the `evidence` profile would blind the
        evidence gate — under the one profile that exists to make the gate
        meaningful — and would digest the very turn keys an operator needs in
        order to go and look at those turns.
        """
        conn.execute(
            """INSERT INTO diagnostics (key, value, updated_at) VALUES (?, ?, ?)
               ON CONFLICT(key) DO UPDATE SET
                 value=excluded.value, updated_at=excluded.updated_at""",
            (
                key,
                self._store_redactor().redact(
                    json.dumps(value, ensure_ascii=False)
                ),
                _utcnow_iso(),
            ),
        )

    # `set_diagnostic_if_absent` used to live here. It was one way of keeping a
    # baseline from clobbering a predecessor's counters, and
    # `merge_writer_health_row` below now does that job properly — for the
    # heartbeat and `close()` too, which is where the clobbering actually
    # happened. Leaving an insert-if-absent beside a merge would be leaving a
    # second answer to a question that has one.

    def merge_writer_health_row(
        self, conn: sqlite3.Connection, incoming: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Persist one writer's counters through the monotone merge.

        The single write path for `writer_health`. `set_diagnostic` replaces,
        which is right for every other diagnostics key and wrong for this one:
        the row outlives the writer that wrote it, so a second writer's zeros
        would erase a first writer's drops and leave `health_delta` subtracting
        from a history that no longer exists.

        Read-modify-write, so it must run inside the caller's `BEGIN IMMEDIATE`:
        every caller here already holds one, and the write lock is what keeps two
        writers from interleaving a read and a write of the same row.
        """
        row = conn.execute(
            "SELECT value FROM diagnostics WHERE key='writer_health'"
        ).fetchone()
        stored: Optional[dict[str, Any]] = None
        if row is not None:
            try:
                stored = json.loads(row[0])
            except Exception:
                stored = None
        merged = merge_writer_health(stored, incoming)
        self.set_diagnostic(conn, "writer_health", merged)
        return merged

    # -- reads (GET /turns, run_chatbot) ---------------------------------

    def list_human_feedback(self, turn_key: str) -> list[dict[str, Any]]:
        """Human annotations, separate from agent-memory feedback in this DB."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM human_feedback WHERE turn_key=? ORDER BY feedback_id",
                (turn_key,),
            ).fetchall()
        return [_human_feedback_row(row) for row in rows]

    def add_human_feedback(self, turn_key: str, *, target_kind: str,
                           span_ids: list[str], target_label: str,
                           provenance: str, comment: str | None = None,
                           went_wrong: str = "", worked: str = "",
                           should_change: str = "") -> None:
        """Append feedback after validating its provenance and evidence anchor."""
        if not isinstance(turn_key, str) or not turn_key:
            raise ValueError("turn_key is required")
        if target_kind not in ("turn", "phase", "step", "span"):
            raise ValueError("invalid feedback target kind")
        if (not isinstance(span_ids, list) or len(span_ids) > 10000
                or any(not isinstance(v, str) or not v for v in span_ids)):
            raise ValueError("span_ids must be a list of recorded span IDs")
        ids = sorted(set(span_ids))
        if (target_kind == "turn" and ids) or (target_kind != "turn" and not ids):
            raise ValueError("component feedback requires spans; turn feedback has none")
        comment = compose_human_feedback_comment(
            went_wrong=went_wrong, worked=worked, should_change=should_change,
            comment=comment,
        )
        if not isinstance(target_label, str) or not target_label or len(target_label) > 1000:
            raise ValueError("target_label is required (at most 1000 characters)")
        if provenance not in FEEDBACK_PROVENANCES:
            raise ValueError(
                "provenance must be human, coding_agent, or distillation_agent"
            )
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if conn.execute("SELECT 1 FROM turns WHERE turn_key=?", (turn_key,)).fetchone() is None:
                raise ValueError("turn not found")
            recorded = {r[0] for r in conn.execute(
                "SELECT span_id FROM spans WHERE trace_id=?", (turn_key,))}
            if not set(ids).issubset(recorded):
                raise ValueError("feedback spans must belong to the selected turn")
            conn.execute(
                "INSERT INTO human_feedback "
                "(turn_key,target_kind,span_ids_json,target_label,comment,provenance,created_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (turn_key, target_kind, json.dumps(ids), self._scrub(target_label),
                 self._scrub(comment), provenance, _utcnow_iso()),
            )

    def get_turn(self, turn_key: str) -> Optional[dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM turns WHERE turn_key=?", (turn_key,)
            ).fetchone()
            return dict(row) if row is not None else None

    def get_spans(self, trace_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM spans WHERE trace_id=? ORDER BY start_ns", (trace_id,)
            ).fetchall()
            return [dict(r) for r in rows]

    def spans_for_turns(
        self, turn_keys: Iterable[str]
    ) -> dict[str, list[dict[str, Any]]]:
        """`get_spans` for many turns at once: ``{turn_key: [span row, ...]}``.

        The debug UI stamps every listed turn with things derived from its
        spans (cut-at-limit tallies, cost roll-ups, decision signals). Doing
        that through `get_spans` cost one indexed query per listed turn, so a
        rail refresh of 500 turns paid 500 round trips. This answers
        the same rows for a whole page in a bounded number of queries.

        Rows have exactly `get_spans`'s shape and per-turn order (`start_ns`,
        over ``idx_spans_trace``). Every requested key is present in the
        answer, mapping to ``[]`` when the store holds no spans for it, so a
        caller never has to distinguish "absent" from "no spans"; duplicate
        keys are collapsed and blank ones dropped. Keys are bound in chunks
        well under SQLite's variable limit, so an arbitrarily long list is
        safe.
        """
        keys = list(dict.fromkeys(key for key in turn_keys if key))
        spans_by_turn: dict[str, list[dict[str, Any]]] = {key: [] for key in keys}
        if not keys:
            return spans_by_turn
        with self._connect() as conn:
            for chunk in _chunked(keys):
                placeholders = ",".join("?" * len(chunk))
                rows = conn.execute(
                    f"SELECT * FROM spans WHERE trace_id IN ({placeholders}) "
                    "ORDER BY trace_id, start_ns",
                    chunk,
                ).fetchall()
                for row in rows:
                    spans_by_turn[row["trace_id"]].append(dict(row))
        return spans_by_turn

    def list_conversations(
        self, channel_id: Optional[str] = None, limit: int = 100, offset: int = 0
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM conversations"
        params: list[Any] = []
        if channel_id is not None:
            query += " WHERE channel_id=?"
            params.append(channel_id)
        query += " ORDER BY COALESCE(last_turn_at, started_at) DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        with self._connect() as conn:
            return [dict(r) for r in conn.execute(query, params).fetchall()]

    def list_turns(
        self,
        channel_id: Optional[str] = None,
        conversation_id: Optional[int] = None,
        status: Optional[str] = None,
        success: Optional[bool] = None,
        command_name: Optional[str] = None,
        context: Optional[str] = None,
        experiment_id: Optional[str] = None,
        task_id: Optional[str] = None,
        attempt: Optional[int] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Turn rows, newest first, without record_json (fetch one turn for that).

        The experiment filters extend this route rather than getting a parallel
        implementation (`[XR9]`); they ride `idx_turns_experiment`.
        """
        clauses: list[str] = []
        params: list[Any] = []
        if channel_id is not None:
            clauses.append("channel_id=?")
            params.append(channel_id)
        if conversation_id is not None:
            clauses.append("conversation_id=?")
            params.append(conversation_id)
        if status is not None:
            clauses.append("status=?")
            params.append(status)
        if success is not None:
            clauses.append("success=?")
            params.append(1 if success else 0)
        if context is not None:
            # Substring match (the debug UI's semantics), parameterized and
            # LIKE-escaped; SQLite LIKE is ASCII-case-insensitive, matching
            # the previous client-side filter.
            escaped = (
                context.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            )
            clauses.append("entry_context LIKE ? ESCAPE '\\'")
            params.append(f"%{escaped}%")
        if command_name is not None:
            clauses.append(
                "turn_key IN (SELECT trace_id FROM spans WHERE command_name=?)"
            )
            params.append(command_name)
        # [DR29]: a DB written before the experiment columns existed must
        # degrade, not raise. The base turn list is the whole point of the debug
        # UI, and a viewer opened on a post-mortem snapshot never migrates it
        # ([R12]), so an unguarded projection would 500 the main view forever
        # with "internal error: OperationalError" and no actionable reason.
        labelled = self.has_feature(FEATURE_EXPERIMENTS_V1)
        if labelled:
            if experiment_id is not None:
                clauses.append("experiment_id=?")
                params.append(experiment_id)
            if task_id is not None:
                clauses.append("task_id=?")
                params.append(task_id)
            if attempt is not None:
                clauses.append("attempt=?")
                params.append(int(attempt))
        elif experiment_id is not None or task_id is not None or attempt is not None:
            # An experiment filter against a DB that records no experiments
            # matches nothing. Returning [] is the honest answer; silently
            # ignoring the filter and returning every turn would be worse than
            # raising.
            return []
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        query = (
            "SELECT turn_key, channel_id, conversation_id, ordinal, user_message, "
            "entry_workflow_name, entry_context, status, success, failure_reason, "
            "answer, started_at, completed_at, suspended_ms"
            + (", experiment_id, task_id, attempt " if labelled else " ")
            + f"FROM turns{where} ORDER BY turn_key DESC LIMIT ? OFFSET ?"
        )
        params.extend([limit, offset])
        with self._connect() as conn:
            return [dict(r) for r in conn.execute(query, params).fetchall()]

    def list_channels(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT channel_id FROM turns ORDER BY channel_id"
            ).fetchall()
            return [r[0] for r in rows]

    def get_artifact(self, artifact_id: str) -> Optional[dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM artifacts WHERE artifact_id=?", (artifact_id,)
            ).fetchone()
            return dict(row) if row is not None else None

    def list_train_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM train_runs ORDER BY COALESCE(completed_at, started_at) DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def writer_health(self) -> Optional[dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value, updated_at FROM diagnostics WHERE key='writer_health'"
            ).fetchone()
            if row is None:
                return None
            health = json.loads(row["value"])
            health["updated_at"] = row["updated_at"]
            return health

    # -- the experiment container (`fix-bn1`, experiment_container_design.md) --
    #
    # CAPTURE POLICY, decided here rather than by omission (`[XR6]`): every text
    # column of `experiments`, `experiment_attempts` and
    # `experiment_evidence_runs` is SCRUB-ONLY -- `redactor.redact(value)` with
    # no `policy.apply` call, the `spans.channel_id` code shape.
    #
    # The precedent is `set_diagnostic` plus `_POLICY_EXEMPT_TURN_COLUMNS`, not
    # `spans.channel_id`'s erasure argument. These rows are not evidence ABOUT a
    # tenant; they are the record of whether the evidence may be used at all --
    # an `EvidenceRun`'s valid/problems, an attempt's outcome, an experiment's
    # declaration. Withholding them reduces nothing a tenant would care about and
    # makes the bundle uninterpretable under exactly the profile an
    # evidence-grade run uses, since `opaque-payload` and `user-text` both map to
    # `omit` there. The claim that makes this safe is a DATAFLOW claim and is
    # tested: no code path exists by which workflow, model or user content
    # reaches these tables, except `task_id`, which the caller supplies from its
    # own task-set file. The residual risk -- an operator pasting a credential
    # into `notes`, an exception repr inside `record_json.problems` -- is exactly
    # what the scrub catches, which is why this is scrub-only and not untouched.
    #
    # No `POLICY_PATH_EXPERIMENT_*` constants are declared: a constant never
    # passed to `policy.apply` is inert, and the one genuinely scrub-only column
    # in this file, `spans.channel_id`, deliberately has none either.

    _EXPERIMENT_STATUSES = frozenset(
        {
            "running",
            "capture_complete",
            "awaiting_evaluation",
            "complete",
            "invalid",
        }
    )
    _EXECUTION_STATUSES = frozenset(
        {"completed", "failed", "cancelled", "abandoned"}
    )
    _ATTEMPT_OUTCOMES = frozenset({"pass", "fail", "error", "incomplete"})
    _INVALID_REASONS = frozenset(
        {
            "attempt_shortfall",
            "evidence_run_invalid",
            "turns_erased",
            "never_completed",
            "operator",
            "stale_claim_records",
        }
    )

    def _scrub(self, value: Any) -> Any:
        """Credential-scrub one experiment-surface value. Falsy passes through."""
        return self._store_redactor().redact(value)

    @staticmethod
    def _normalize_benchmark_pin(
        benchmark_id: Optional[str],
        benchmark_version: Optional[str],
        benchmark_digest_sha256: Optional[str],
    ) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """Return a scrubbed all-or-none benchmark pin triple."""

        def _clean(value: Optional[str]) -> Optional[str]:
            if value is None:
                return None
            text = str(value).strip()
            return text or None

        normalized = (
            _clean(benchmark_id),
            _clean(benchmark_version),
            _clean(benchmark_digest_sha256),
        )
        provided = [field is not None for field in normalized]
        if any(provided) and not all(provided):
            raise ValueError(
                "benchmark_id, benchmark_version, and benchmark_digest_sha256 "
                "must all be provided together"
            )
        return normalized

    @staticmethod
    def _experiment_benchmark_pin(
        experiment: dict[str, Any],
    ) -> Optional[tuple[str, str, str]]:
        """Return the stored benchmark pin, or None when the row is unpinned."""

        pin = (
            experiment.get("benchmark_id"),
            experiment.get("benchmark_version"),
            experiment.get("benchmark_digest_sha256"),
        )
        if all(field is None for field in pin):
            return None
        return pin  # type: ignore[return-value]

    def create_experiment(
        self,
        experiment_id: str,
        description: str,
        *,
        declared_tasks: int,
        declared_attempts: int,
        required_evidence_segments: int = 0,
        arm: Optional[str] = None,
        baseline_experiment_id: Optional[str] = None,
        workflow_name: Optional[str] = None,
        capture_profile: Optional[str] = None,
        capture_policy_version: Optional[str] = None,
        benchmark_id: Optional[str] = None,
        benchmark_version: Optional[str] = None,
        benchmark_digest_sha256: Optional[str] = None,
    ) -> None:
        """Pre-register an experiment. Written BEFORE any task runs.

        `declared_tasks` and `declared_attempts` are required and positive: they
        are the denominator every score is computed against (`[XR14]`), and a
        score computed over surviving rows instead is the exact failure
        `EvidenceRun` exists to prevent one layer down.

        Re-creating an existing experiment is how a resume re-attaches. The
        `DO UPDATE` set deliberately excludes `status`, `invalid_reason` and
        `invalid_detail`: a resume must not be able to launder an `invalid`
        verdict back to `running`.
        """
        if not experiment_id:
            raise ValueError("experiment_id is required")
        if not isinstance(description, str):
            # Optional free text, not a name: empty is a run whose author wrote
            # nothing, which is not an error.
            raise ValueError("description must be text")
        declared_tasks = int(declared_tasks)
        declared_attempts = int(declared_attempts)
        required_evidence_segments = int(required_evidence_segments)
        if declared_tasks <= 0 or declared_attempts <= 0:
            raise ValueError(
                "declared_tasks and declared_attempts must both be positive: "
                "they are the denominator, and a score over an undeclared "
                "denominator is computed over whatever survived"
            )
        if required_evidence_segments < 0:
            raise ValueError("required_evidence_segments cannot be negative")
        try:
            benchmark_id, benchmark_version, benchmark_digest_sha256 = (
                self._normalize_benchmark_pin(
                    benchmark_id, benchmark_version, benchmark_digest_sha256
                )
            )
        except ValueError as exc:
            raise PartialBenchmarkPin(experiment_id) from exc
        if benchmark_id is not None:
            benchmark_id = self._scrub(benchmark_id)
            benchmark_version = self._scrub(benchmark_version)
            benchmark_digest_sha256 = self._scrub(benchmark_digest_sha256)
        capture_profile = capture_profile or _env("FW_OBS_CAPTURE_PROFILE", "debug")
        capture_policy_version = capture_policy_version or CAPTURE_POLICY_VERSION
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            # A re-create (the resume path) under a DIFFERENT capture regime is
            # refused rather than silently keeping the first one. The stored
            # profile is what `compare_experiments` gates on, so a run whose
            # second half was captured under another policy would compare as if
            # both halves matched -- and the column would say so.
            existing = conn.execute(
                """SELECT capture_profile, capture_policy_version, status,
                          benchmark_id, benchmark_version,
                          benchmark_digest_sha256
                     FROM experiments WHERE experiment_id=?""",
                (experiment_id,),
            ).fetchone()
            if existing is not None and (
                existing["capture_profile"] != capture_profile
                or existing["capture_policy_version"] != capture_policy_version
            ):
                conn.rollback()
                raise CaptureRegimeChanged(
                    experiment_id,
                    f"{existing['capture_profile']}/"
                    f"{existing['capture_policy_version']}",
                    f"{capture_profile}/{capture_policy_version}",
                )
            if existing is not None and benchmark_id is not None:
                stored_pin = (
                    existing["benchmark_id"],
                    existing["benchmark_version"],
                    existing["benchmark_digest_sha256"],
                )
                incoming_pin = (
                    benchmark_id,
                    benchmark_version,
                    benchmark_digest_sha256,
                )
                if all(field is not None for field in stored_pin):
                    if stored_pin != incoming_pin:
                        conn.rollback()
                        raise BenchmarkPinIsWriteOnce(experiment_id)
            conn.execute(
                """INSERT INTO experiments
                   (experiment_id, description, notes, arm,
                    baseline_experiment_id, status, invalid_reason,
                    invalid_detail, declared_tasks, declared_attempts,
                    required_evidence_segments, benchmark_id, benchmark_version,
                    benchmark_digest_sha256, workflow_name, capture_profile,
                    capture_policy_version,
                    created_at, completed_at)
                   VALUES (?, ?, NULL, ?, ?, 'running', NULL, NULL,
                           ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                   ON CONFLICT(experiment_id) DO UPDATE SET
                     description=excluded.description,
                     arm=excluded.arm,
                     baseline_experiment_id=excluded.baseline_experiment_id,
                     -- The denominator is rewritable only while the experiment
                     -- is still running. Once it is complete or invalid, its
                     -- score has been computed against the declaration, and
                     -- changing the declaration afterwards silently restates
                     -- every number already reported from it -- the same
                     -- after-the-fact rewrite the write-once benchmark pin
                     -- below exists to prevent, one field over.
                     declared_tasks=CASE WHEN experiments.status='running'
                       THEN excluded.declared_tasks ELSE experiments.declared_tasks END,
                     declared_attempts=CASE WHEN experiments.status='running'
                       THEN excluded.declared_attempts ELSE experiments.declared_attempts END,
                     required_evidence_segments=CASE
                       WHEN experiments.status='running'
                       THEN excluded.required_evidence_segments
                       ELSE experiments.required_evidence_segments END,
                     -- Write-once pin: a stored pin wins (the guard above has
                     -- already rejected a differing incoming one); an unpinned
                     -- row takes the incoming pin instead of dropping it.
                     benchmark_id=COALESCE(experiments.benchmark_id,
                                           excluded.benchmark_id),
                     benchmark_version=COALESCE(experiments.benchmark_version,
                                                excluded.benchmark_version),
                     benchmark_digest_sha256=COALESCE(
                       experiments.benchmark_digest_sha256,
                       excluded.benchmark_digest_sha256),
                     workflow_name=excluded.workflow_name""",
                (
                    experiment_id,
                    self._scrub(description),
                    self._scrub(arm),
                    baseline_experiment_id,
                    declared_tasks,
                    declared_attempts,
                    required_evidence_segments,
                    benchmark_id,
                    benchmark_version,
                    benchmark_digest_sha256,
                    self._scrub(workflow_name),
                    capture_profile,
                    capture_policy_version,
                    _utcnow_iso(),
                ),
            )
            conn.commit()

    def declare_experiment_attempts(
        self,
        experiment_id: str,
        declarations: Iterable[tuple[str, int, str]],
    ) -> None:
        """Persist one immutable, exact attempt plan before execution starts."""
        normalized = {
            (self._scrub(task_id), int(native_attempt), self._scrub(source_key))
            for task_id, native_attempt, source_key in declarations
        }
        if not normalized:
            raise ValueError("an experiment attempt declaration cannot be empty")
        if any(
            not task_id or native_attempt <= 0 or not source_key
            for task_id, native_attempt, source_key in normalized
        ):
            raise ValueError(
                "each declaration requires task_id, positive native_attempt, "
                "and source_key"
            )
        attempt_identities = {
            (task_id, native_attempt)
            for task_id, native_attempt, _ in normalized
        }
        if len(attempt_identities) != len(normalized):
            raise ValueError(
                "each task_id/native_attempt pair must have exactly one source_key"
            )
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            experiment = conn.execute(
                """SELECT status, declared_tasks, declared_attempts
                     FROM experiments WHERE experiment_id=?""",
                (experiment_id,),
            ).fetchone()
            if experiment is None:
                conn.rollback()
                raise ExperimentNotFound(experiment_id)
            if experiment["status"] != "running":
                conn.rollback()
                raise ExperimentIsClosed(experiment_id, experiment["status"])
            if conn.execute(
                "SELECT 1 FROM experiment_attempts WHERE experiment_id=? LIMIT 1",
                (experiment_id,),
            ).fetchone():
                conn.rollback()
                raise ExperimentDeclarationConflict(
                    f"experiment {experiment_id!r} has already started"
                )
            expected_rows = (
                int(experiment["declared_tasks"])
                * int(experiment["declared_attempts"])
            )
            task_counts: dict[str, int] = {}
            for task_id, _, _ in normalized:
                task_counts[task_id] = task_counts.get(task_id, 0) + 1
            if (
                len(normalized) != expected_rows
                or len(task_counts) != int(experiment["declared_tasks"])
                or set(task_counts.values())
                != {int(experiment["declared_attempts"])}
            ):
                conn.rollback()
                raise ValueError(
                    "exact declarations do not match the declared task/attempt "
                    "display summaries"
                )
            stored = {
                (row["task_id"], int(row["native_attempt"]), row["source_key"])
                for row in conn.execute(
                    """SELECT task_id, native_attempt, source_key
                         FROM experiment_attempt_declarations
                        WHERE experiment_id=?""",
                    (experiment_id,),
                ).fetchall()
            }
            if stored:
                conn.rollback()
                if stored == normalized:
                    return
                raise ExperimentDeclarationConflict(
                    f"experiment {experiment_id!r} already has a different "
                    "immutable attempt declaration"
                )
            now = _utcnow_iso()
            conn.executemany(
                """INSERT INTO experiment_attempt_declarations
                   (experiment_id, task_id, native_attempt, source_key, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                [
                    (experiment_id, task_id, native_attempt, source_key, now)
                    for task_id, native_attempt, source_key in sorted(normalized)
                ],
            )
            conn.commit()

    def register_attempt(
        self,
        experiment_id: str,
        task_id: str,
        native_attempt: int,
        source_key: str,
        channel_id: str,
        *,
        ttl_seconds: float = 300.0,
        recovery: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Issue a one-use bootstrap. Only its SHA-256 digest is persisted.

        Replacing a claimed registration requires both an expired lease and
        affirmative recovery evidence. Expiry alone is only a candidate signal:
        it does not prove the previous process stopped producing side effects.
        """
        if not all((experiment_id, task_id, source_key, channel_id)):
            raise ValueError("attempt identity and channel_id are required")
        native_attempt = int(native_attempt)
        ttl_seconds = float(ttl_seconds)
        if native_attempt <= 0 or ttl_seconds <= 0:
            raise ValueError("native_attempt and ttl_seconds must be positive")
        secret = uuid.uuid4().hex + uuid.uuid4().hex
        registration_id = f"reg-{uuid.uuid4().hex}"
        secret_hash = hashlib.sha256(secret.encode("utf-8")).hexdigest()
        now = time.time()
        task_id = self._scrub(task_id)
        source_key = self._scrub(source_key)
        channel_id = self._scrub(channel_id)
        recovery_json = (
            None
            if recovery is None
            else self._scrub(
                json.dumps(_sanitize_json_value(recovery), sort_keys=True)
            )
        )
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            declared = conn.execute(
                """SELECT 1 FROM experiment_attempt_declarations
                    WHERE experiment_id=? AND task_id=?
                      AND native_attempt=? AND source_key=?""",
                (experiment_id, task_id, native_attempt, source_key),
            ).fetchone()
            if declared is None:
                conn.rollback()
                raise UndeclaredExperimentAttempt(
                    f"attempt {experiment_id}/{task_id}/{native_attempt} with "
                    f"source_key {source_key!r} was not declared"
                )
            existing = conn.execute(
                """SELECT state, expires_at, lease_expires_at
                     FROM experiment_attempt_claims
                    WHERE experiment_id=? AND task_id=? AND native_attempt=?""",
                (experiment_id, task_id, native_attempt),
            ).fetchone()
            if existing is not None and existing["state"] == "claimed":
                lease_expired = (
                    existing["lease_expires_at"] is not None
                    and float(existing["lease_expires_at"]) <= now
                )
                process_dead = bool((recovery or {}).get("owned_process_dead"))
                explicit_fence = bool((recovery or {}).get("explicit_fence"))
                reconciled = bool((recovery or {}).get("reconciled"))
                if not lease_expired or not (
                    process_dead or (explicit_fence and reconciled)
                ):
                    conn.rollback()
                    raise AttemptClaimError(
                        "replacement bootstrap requires an expired lease plus "
                        "proof the owned process is dead, or an explicit fence "
                        "with reconciliation"
                    )
            elif (
                existing is not None
                and existing["state"] == "pending"
                and float(existing["expires_at"]) > now
            ):
                conn.rollback()
                raise AttemptClaimError("an unexpired bootstrap is already pending")
            if existing is None:
                conn.execute(
                    """INSERT INTO experiment_attempt_claims
                       (registration_id, experiment_id, task_id, native_attempt,
                        source_key, channel_id, secret_hash, expires_at, state,
                        epoch, recovery_json, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', 0, ?, ?)""",
                    (
                        registration_id,
                        experiment_id,
                        task_id,
                        native_attempt,
                        source_key,
                        channel_id,
                        secret_hash,
                        now + ttl_seconds,
                        recovery_json,
                        _utcnow_iso(),
                    ),
                )
            else:
                conn.execute(
                    """UPDATE experiment_attempt_claims
                          SET registration_id=?, source_key=?, channel_id=?,
                              secret_hash=?, expires_at=?, state='pending',
                              server_incarnation=NULL, lease_expires_at=NULL,
                              conversation_id=NULL, recovery_json=?,
                              created_at=?, claimed_at=NULL
                        WHERE experiment_id=? AND task_id=? AND native_attempt=?""",
                    (
                        registration_id,
                        source_key,
                        channel_id,
                        secret_hash,
                        now + ttl_seconds,
                        recovery_json,
                        _utcnow_iso(),
                        experiment_id,
                        task_id,
                        native_attempt,
                    ),
                )
            conn.commit()
        return {
            "registration_id": registration_id,
            "secret": secret,
            "channel_id": channel_id,
            "expires_at": now + ttl_seconds,
        }

    def claim_attempt(
        self,
        bootstrap: dict[str, Any],
        *,
        channel_id: str,
        server_incarnation: str,
        lease_seconds: float = 300.0,
        runtime_snapshot: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        """Consume a bootstrap and reserve its labelled conversation atomically.

        ``runtime_snapshot`` is the claiming server's credential-free
        ``runtime_readiness_snapshot``, stored verbatim as JSON on the attempt
        row so the record says which configuration served it. None is stored
        as NULL: an attempt whose server could not be described is a real
        state and must not be dressed up as a described one. A re-claim (a new
        binding) replaces the stamp, because the stamp belongs to the binding,
        not to the attempt's first server.
        """
        registration_id = str(bootstrap.get("registration_id") or "")
        secret = str(bootstrap.get("secret") or "")
        if not registration_id or not secret or not channel_id or not server_incarnation:
            raise AttemptClaimError(
                "registration_id, secret, channel_id and server_incarnation are required"
            )
        runtime_snapshot_json = (
            None
            if runtime_snapshot is None
            else self._scrub(
                json.dumps(
                    _sanitize_json_value(dict(runtime_snapshot)),
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        )
        now = time.time()
        incoming_hash = hashlib.sha256(secret.encode("utf-8")).hexdigest()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """SELECT * FROM experiment_attempt_claims
                    WHERE registration_id=?""",
                (registration_id,),
            ).fetchone()
            if row is None:
                conn.rollback()
                raise AttemptClaimError("unknown attempt registration")
            if row["state"] != "pending" or row["secret_hash"] is None:
                conn.rollback()
                raise AttemptClaimError("attempt bootstrap was already consumed")
            if float(row["expires_at"]) <= now:
                conn.rollback()
                raise AttemptClaimError("attempt bootstrap expired")
            if row["channel_id"] != self._scrub(channel_id):
                conn.rollback()
                raise AttemptClaimError("attempt bootstrap channel mismatch")
            if not hmac.compare_digest(row["secret_hash"], incoming_hash):
                conn.rollback()
                raise AttemptClaimError("attempt bootstrap secret mismatch")
            declared = conn.execute(
                """SELECT 1 FROM experiment_attempt_declarations
                    WHERE experiment_id=? AND task_id=?
                      AND native_attempt=? AND source_key=?""",
                (
                    row["experiment_id"],
                    row["task_id"],
                    row["native_attempt"],
                    row["source_key"],
                ),
            ).fetchone()
            if declared is None:
                conn.rollback()
                raise UndeclaredExperimentAttempt("attempt declaration no longer matches")
            epoch = int(row["epoch"]) + 1
            reserved = conn.execute(
                """SELECT conversation_id FROM conversations
                    WHERE experiment_id=? AND task_id=? AND attempt=?""",
                (
                    row["experiment_id"],
                    row["task_id"],
                    row["native_attempt"],
                ),
            ).fetchone()
            conversation_id = (
                int(reserved["conversation_id"])
                if reserved is not None
                else self._mint_conversation_id_in_txn(
                    conn,
                    row["channel_id"],
                    experiment_id=row["experiment_id"],
                    task_id=row["task_id"],
                    attempt=int(row["native_attempt"]),
                )
            )
            experiment = conn.execute(
                "SELECT status FROM experiments WHERE experiment_id=?",
                (row["experiment_id"],),
            ).fetchone()
            if experiment is None or experiment["status"] != "running":
                conn.rollback()
                if experiment is None:
                    raise ExperimentNotFound(row["experiment_id"])
                raise ExperimentIsClosed(
                    row["experiment_id"], experiment["status"]
                )
            conn.execute(
                """INSERT INTO experiment_attempts
                   (experiment_id, task_id, attempt, channel_id, conversation_id,
                    outcome, outcome_source, reward, restarts, started_at,
                    execution_status, execution_finished_at, finished_at,
                    detail_json, source_attempt_json, source_key,
                    runtime_snapshot_json)
                   VALUES (?, ?, ?, ?, ?, NULL, NULL, NULL, 0, ?,
                           NULL, NULL, NULL, NULL, NULL, ?, ?)
                   ON CONFLICT(experiment_id, task_id, attempt) DO UPDATE SET
                     channel_id=excluded.channel_id,
                     conversation_id=excluded.conversation_id,
                     source_key=excluded.source_key,
                     runtime_snapshot_json=excluded.runtime_snapshot_json""",
                (
                    row["experiment_id"],
                    row["task_id"],
                    row["native_attempt"],
                    row["channel_id"],
                    conversation_id,
                    _utcnow_iso(),
                    row["source_key"],
                    runtime_snapshot_json,
                ),
            )
            updated = conn.execute(
                """UPDATE experiment_attempt_claims
                      SET state='claimed', epoch=?, server_incarnation=?,
                          lease_expires_at=?, conversation_id=?,
                          secret_hash=NULL, claimed_at=?
                    WHERE registration_id=? AND state='pending'
                      AND secret_hash=?""",
                (
                    epoch,
                    server_incarnation,
                    now + float(lease_seconds),
                    conversation_id,
                    _utcnow_iso(),
                    registration_id,
                    row["secret_hash"],
                ),
            )
            if updated.rowcount != 1:
                conn.rollback()
                raise AttemptClaimError("attempt claim lost its compare-and-swap")
            conn.commit()
        return {
            "registration_id": registration_id,
            "experiment_id": row["experiment_id"],
            "task_id": row["task_id"],
            "attempt": int(row["native_attempt"]),
            "source_key": row["source_key"],
            "channel_id": row["channel_id"],
            "conversation_id": conversation_id,
            "epoch": epoch,
            "server_incarnation": server_incarnation,
        }

    def validate_attempt_claim(self, claim: dict[str, Any]) -> None:
        """Fail if a live caller no longer owns the registered attempt."""
        with self._connect() as conn:
            if not self._claim_is_current_in_txn(conn, claim):
                raise StaleExperimentClaim(
                    f"attempt claim epoch {claim.get('epoch')!r} is stale"
                )

    @staticmethod
    def _claim_is_current_in_txn(
        conn: sqlite3.Connection, claim: dict[str, Any]
    ) -> bool:
        if not claim or claim.get("experiment_id") is None:
            return True
        # Internal/in-process experiment producers remain compatible. A
        # registered external record always carries both fields.
        if claim.get("epoch") is None and claim.get("server_incarnation") is None:
            registered = conn.execute(
                """SELECT 1 FROM experiment_attempt_claims
                    WHERE experiment_id=? LIMIT 1""",
                (claim.get("experiment_id"),),
            ).fetchone()
            return registered is None
        if claim.get("epoch") is None or claim.get("server_incarnation") is None:
            return False
        row = conn.execute(
            """SELECT epoch, server_incarnation, state
                 FROM experiment_attempt_claims
                WHERE experiment_id=? AND task_id=? AND native_attempt=?""",
            (
                claim.get("experiment_id"),
                claim.get("task_id"),
                claim.get("attempt"),
            ),
        ).fetchone()
        return bool(
            row is not None
            and row["state"] == "claimed"
            and int(row["epoch"]) == int(claim.get("epoch") or -1)
            and row["server_incarnation"] == claim.get("server_incarnation")
        )

    def _reject_stale_claim_in_txn(
        self, conn: sqlite3.Connection, claim: dict[str, Any], record_kind: str
    ) -> bool:
        if self._claim_is_current_in_txn(conn, claim):
            return False
        row = conn.execute(
            "SELECT value FROM diagnostics WHERE key='stale_claim_records'"
        ).fetchone()
        try:
            previous = json.loads(row["value"]) if row is not None else {}
        except (TypeError, ValueError):
            previous = {}
        count = int(previous.get("count") or 0) + 1
        self.set_diagnostic(
            conn,
            "stale_claim_records",
            {
                "count": count,
                "invalidating": True,
                "last_record_kind": record_kind,
                "experiment_id": claim.get("experiment_id"),
                "epoch": claim.get("epoch"),
            },
        )
        experiment_id = claim.get("experiment_id")
        if experiment_id:
            self.invalidate_experiments_in_txn(
                conn,
                [experiment_id],
                "stale_claim_records",
                f"rejected stale {record_kind} from epoch {claim.get('epoch')}",
            )
        return True

    def update_experiment_notes(self, experiment_id: str, notes: Optional[str]) -> None:
        """Freely editable, like `description` and unlike the write-once pin."""
        self._update_experiment(
            "UPDATE experiments SET notes=? WHERE experiment_id=?",
            (self._scrub(notes), experiment_id),
            experiment_id,
        )

    def update_experiment_archived(
        self, experiment_id: str, archived: bool
    ) -> None:
        """Archive visibility is editable metadata, not recorded evidence."""
        if not isinstance(archived, bool):
            raise ValueError("archived must be true or false")
        self._update_experiment(
            "UPDATE experiments SET archived=? WHERE experiment_id=?",
            (1 if archived else 0, experiment_id),
            experiment_id,
        )

    def _update_experiment(
        self, sql: str, params: tuple, experiment_id: str
    ) -> None:
        """Run an experiment UPDATE, raising when it matches no row.

        A 0-row update means the container is gone -- `clear_conversations` is
        an HTTP-triggered whole-DB erase and can land mid-run. Failing the
        harness loudly beats accumulating turns labelled against a container
        that no longer exists (`[XR15]`).
        """
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.execute(sql, params)
            if cursor.rowcount == 0:
                conn.rollback()
                raise ExperimentNotFound(experiment_id)
            conn.commit()

    def record_evidence_segment(
        self,
        experiment_id: str,
        seq: int,
        evidence_run_id: str,
        record: dict[str, Any],
        *,
        claim: Optional[dict[str, Any]] = None,
    ) -> None:
        """Record one `evidence_run()` segment (`[XR1]`).

        One row per segment rather than an appended JSON array, because
        appending to a column is a read-modify-write and `[XR20]` forbids that
        on any column a capture policy might act on. Here each segment is an
        independent INSERT and `valid` is a queryable column.

        `record` is the WHOLE `EvidenceRun.as_record()`, not its `observability`
        sub-dict: the sub-dict alone carries neither the run id, nor `valid`,
        nor `problems`, nor the archive digest.
        """
        payload = json.dumps(_sanitize_json_value(record), ensure_ascii=False)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if claim and self._reject_stale_claim_in_txn(
                conn, claim, "derived_record"
            ):
                conn.commit()
                raise StaleExperimentClaim(
                    f"attempt claim epoch {claim.get('epoch')!r} is stale"
                )
            experiment = conn.execute(
                "SELECT status FROM experiments WHERE experiment_id=?",
                (experiment_id,),
            ).fetchone()
            if experiment is None:
                conn.rollback()
                raise ExperimentNotFound(experiment_id)
            if experiment["status"] in {"complete", "capture_complete", "invalid"}:
                conn.rollback()
                raise ExperimentIsClosed(experiment_id, experiment["status"])
            conn.execute(
                """INSERT INTO experiment_evidence_runs
                   (experiment_id, seq, evidence_run_id, valid, started_at,
                    completed_at, record_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(experiment_id, seq) DO UPDATE SET
                     evidence_run_id=excluded.evidence_run_id,
                     -- Monotone in invalidity, like `status <> 'invalid'` one
                     -- table over: once a segment has reported that evidence
                     -- was lost, re-writing that seq must not be able to erase
                     -- the report. Every other invalidity in this container is
                     -- terminal or write-once, and a rewritable one is a
                     -- verdict that can be revised after seeing the outcome.
                     valid=CASE WHEN experiment_evidence_runs.valid = 0
                                THEN 0 ELSE excluded.valid END,
                     started_at=excluded.started_at,
                     completed_at=excluded.completed_at,
                     record_json=excluded.record_json""",
                (
                    experiment_id,
                    int(seq),
                    evidence_run_id,
                    1 if record.get("valid") else 0,
                    record.get("started_at"),
                    record.get("completed_at"),
                    self._scrub(payload),
                ),
            )
            conn.commit()

    def begin_workspace_seal(self, experiment_id: str) -> str:
        """Stamp the terminal status a seal is about to freeze.

        Runs BEFORE `archive_to`, and it has to. The archive is a byte-immutable
        snapshot: whatever the source row says at the instant of the snapshot is
        what the archived copy says forever, and the archived copy is the one a
        reader opens. Stamping `complete` afterwards — as the seal used to —
        left every sealed archive reporting `capture_complete` about an
        experiment its own manifest presented as sealed.

        `evidence_sealed_at` is stamped here too, which is what makes the two
        halves of a seal distinguishable afterwards without a new column: a row
        with a seal timestamp and no `workspace_archive_sha256` is a seal whose
        archive never landed, and `experiment_scores` refuses to report on it.

        Idempotent for a retry (`complete` with no digest re-enters), refused
        once a digest exists, and never reached from `invalid` — that verdict
        stays terminal here as everywhere.
        """
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """SELECT status, workspace_archive_sha256
                     FROM experiments WHERE experiment_id=?""",
                (experiment_id,),
            ).fetchone()
            if row is None:
                conn.rollback()
                raise ExperimentNotFound(experiment_id)
            if row["status"] == "invalid":
                conn.rollback()
                raise ExperimentIsClosed(experiment_id, "invalid")
            if row["workspace_archive_sha256"]:
                conn.rollback()
                raise AttemptValueConflict(
                    f"experiment {experiment_id!r} already names a sealed "
                    "workspace archive"
                )
            if row["status"] not in {"capture_complete", "complete"}:
                conn.rollback()
                raise ValueError(
                    f"experiment {experiment_id!r} is {row['status']!r}; "
                    "only capture_complete evidence can be sealed"
                )
            conn.execute(
                """UPDATE experiments
                      SET status='complete',
                          evidence_sealed_at=COALESCE(evidence_sealed_at, ?)
                    WHERE experiment_id=? AND status <> 'invalid'""",
                (_utcnow_iso(), experiment_id),
            )
            conn.commit()
        return "complete"

    def record_workspace_archive(
        self,
        experiment_id: str,
        *,
        sha256: str,
        store_identity: str,
    ) -> str:
        """Attach the sole sealed-evidence handle to an already-promoted run.

        The archive is created first. This write intentionally happens only
        afterwards, so the helper can prove that snapshotting did not modify
        the source DB or its committed WAL — and because a file cannot contain
        its own digest, which is why the digest lives on the source row and in
        the manifest while the STATUS lives in the archive too
        (`begin_workspace_seal`).
        """
        if not re.fullmatch(r"[0-9a-f]{64}", sha256 or ""):
            raise ValueError("sha256 must be a lowercase 64-character digest")
        if not store_identity:
            raise ValueError("store_identity is required")
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """SELECT status, workspace_archive_sha256,
                          workspace_store_identity
                     FROM experiments WHERE experiment_id=?""",
                (experiment_id,),
            ).fetchone()
            if row is None:
                conn.rollback()
                raise ExperimentNotFound(experiment_id)
            if row["status"] == "invalid":
                conn.rollback()
                raise ExperimentIsClosed(experiment_id, "invalid")
            if row["status"] not in {"capture_complete", "complete"}:
                conn.rollback()
                raise ValueError(
                    f"experiment {experiment_id!r} is {row['status']!r}; "
                    "only capture_complete evidence can be sealed"
                )
            stored_handle = (
                row["workspace_archive_sha256"],
                row["workspace_store_identity"],
            )
            incoming_handle = (sha256, store_identity)
            if stored_handle != (None, None) and stored_handle != incoming_handle:
                conn.rollback()
                raise AttemptValueConflict(
                    f"experiment {experiment_id!r} already names a different "
                    "sealed workspace archive"
                )
            if store_identity != self.store_identity():
                conn.rollback()
                raise StoreIdentityMismatch(
                    f"archive store {store_identity!r} does not match source "
                    f"store {self.store_identity()!r}"
                )
            status = "complete" if row["status"] == "capture_complete" else row["status"]
            conn.execute(
                """UPDATE experiments
                      SET workspace_archive_sha256=?,
                          workspace_store_identity=?,
                          evidence_sealed_at=COALESCE(evidence_sealed_at, ?),
                          status=?
                    WHERE experiment_id=?""",
                (sha256, store_identity, _utcnow_iso(), status, experiment_id),
            )
            conn.commit()
        return str(status)

    def start_attempt(
        self,
        experiment_id: str,
        task_id: str,
        attempt: int,
        channel_id: str,
        conversation_id: Optional[int] = None,
        source_attempt_key: Optional[dict[str, Any]] = None,
        source_key: Optional[str] = None,
    ) -> None:
        """Open an attempt row before its first turn.

        The row's existence is not the execution completion marker --
        `execution_finished_at` is (`[XR13]`). An attempt that crashed halfway
        has rows and an open marker, which is what makes it visible to the
        resume selector and fatal to a `complete` verdict.
        """
        if not task_id:
            raise ValueError("task_id is required")
        attempt = int(attempt)
        if attempt <= 0:
            raise ValueError("attempt must be a positive integer")
        scrubbed_task_id = self._scrub(task_id)
        scrubbed_source_key = self._scrub(source_key)
        source_attempt_json = (
            None
            if source_attempt_key is None
            else self._scrub(
                json.dumps(
                    {
                        "version": 1,
                        "value": _sanitize_json_value(source_attempt_key),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        )
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT status FROM experiments WHERE experiment_id=?",
                (experiment_id,),
            ).fetchone()
            if row is None:
                conn.rollback()
                raise ExperimentNotFound(experiment_id)
            if row["status"] != "running":
                # A closed experiment's verdicts are not rewritable. Without
                # this, re-invoking a driver script that pins its experiment_id
                # would silently overwrite all 45 stored outcomes and the
                # evidence segment of a `complete` run whose numbers had already
                # been quoted -- and `run()` has no guard of its own, unlike
                # `resume()`. Enforced here, where the `[XR12]` invariants live.
                conn.rollback()
                raise ExperimentIsClosed(experiment_id, row["status"])
            has_declaration = conn.execute(
                """SELECT 1 FROM experiment_attempt_declarations
                    WHERE experiment_id=? LIMIT 1""",
                (experiment_id,),
            ).fetchone()
            if has_declaration is not None:
                declared = conn.execute(
                    """SELECT 1 FROM experiment_attempt_declarations
                        WHERE experiment_id=? AND task_id=?
                          AND native_attempt=? AND source_key=?""",
                    (
                        experiment_id,
                        scrubbed_task_id,
                        attempt,
                        scrubbed_source_key,
                    ),
                ).fetchone()
                if declared is None:
                    conn.rollback()
                    raise UndeclaredExperimentAttempt(
                        f"attempt {experiment_id}/{task_id}/{attempt} with "
                        f"source_key {source_key!r} was not declared"
                    )
            existing = conn.execute(
                """SELECT channel_id, source_attempt_json, source_key
                     FROM experiment_attempts
                    WHERE experiment_id=? AND task_id=? AND attempt=?""",
                (experiment_id, scrubbed_task_id, attempt),
            ).fetchone()
            if existing is not None and (
                existing["channel_id"] != self._scrub(channel_id)
                or existing["source_attempt_json"] != source_attempt_json
                or existing["source_key"] != scrubbed_source_key
            ):
                conn.rollback()
                raise AttemptValueConflict(
                    f"attempt {experiment_id}/{task_id}/{attempt} was already "
                    "started with different identity metadata"
                )
            conn.execute(
                """INSERT INTO experiment_attempts
                   (experiment_id, task_id, attempt, channel_id, conversation_id,
                    outcome, outcome_source, reward, restarts, started_at,
                    execution_status, execution_finished_at, finished_at,
                    detail_json, source_attempt_json, source_key)
                   VALUES (?, ?, ?, ?, ?, NULL, NULL, NULL, 0, ?,
                           NULL, NULL, NULL, NULL, ?, ?)
                   ON CONFLICT(experiment_id, task_id, attempt) DO UPDATE SET
                     channel_id=excluded.channel_id,
                     conversation_id=COALESCE(excluded.conversation_id,
                                              experiment_attempts.conversation_id)""",
                (
                    experiment_id,
                    scrubbed_task_id,
                    attempt,
                    self._scrub(channel_id),
                    conversation_id,
                    _utcnow_iso(),
                    source_attempt_json,
                    scrubbed_source_key,
                ),
            )
            conn.commit()

    def terminalize_attempt(
        self,
        experiment_id: str,
        task_id: str,
        attempt: int,
        *,
        execution_status: str,
        conversation_id: Optional[int] = None,
    ) -> None:
        """Record execution terminality independently from evaluation."""
        if execution_status not in self._EXECUTION_STATUSES:
            raise ValueError(
                f"execution_status {execution_status!r} is not one of "
                f"{sorted(self._EXECUTION_STATUSES)}"
            )
        task_id = self._scrub(task_id)
        attempt = int(attempt)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """SELECT a.execution_status, a.conversation_id,
                          e.status AS experiment_status
                     FROM experiment_attempts a
                     JOIN experiments e ON e.experiment_id=a.experiment_id
                    WHERE a.experiment_id=? AND a.task_id=? AND a.attempt=?""",
                (experiment_id, task_id, attempt),
            ).fetchone()
            if row is None:
                conn.rollback()
                raise ExperimentNotFound(experiment_id)
            if row["execution_status"] is not None:
                same_conversation = (
                    conversation_id is None
                    or row["conversation_id"] is None
                    or int(row["conversation_id"]) == int(conversation_id)
                )
                if row["execution_status"] == execution_status and same_conversation:
                    conn.rollback()
                    return
                conn.rollback()
                raise AttemptValueConflict(
                    f"attempt {experiment_id}/{task_id}/{attempt} already has "
                    f"execution terminal value {row['execution_status']!r}"
                )
            if row["experiment_status"] != "running":
                conn.rollback()
                raise ExperimentIsClosed(
                    experiment_id, row["experiment_status"]
                )
            conn.execute(
                """UPDATE experiment_attempts
                      SET execution_status=?, execution_finished_at=?,
                          conversation_id=COALESCE(?, conversation_id)
                    WHERE experiment_id=? AND task_id=? AND attempt=?""",
                (
                    execution_status,
                    _utcnow_iso(),
                    conversation_id,
                    experiment_id,
                    task_id,
                    attempt,
                ),
            )
            conn.commit()

    def record_attempt_outcome(
        self,
        experiment_id: str,
        task_id: str,
        attempt: int,
        *,
        outcome: str,
        outcome_source: str,
        reward: Optional[float] = None,
        detail: Optional[dict[str, Any]] = None,
    ) -> None:
        """Record a grade after execution, idempotently and write-once."""
        if outcome not in self._ATTEMPT_OUTCOMES:
            raise ValueError(
                f"outcome {outcome!r} is not one of {sorted(self._ATTEMPT_OUTCOMES)}"
            )
        if not outcome_source:
            raise ValueError(
                "outcome_source is required: an unattributed verdict cannot be "
                "told apart from a fallback"
            )
        task_id = self._scrub(task_id)
        source = self._scrub(outcome_source)
        reward_value = None if reward is None else float(reward)
        detail_json = (
            None
            if detail is None
            else self._scrub(
                json.dumps(
                    _sanitize_json_value(detail),
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        )
        attempt = int(attempt)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """SELECT a.execution_finished_at, a.outcome, a.outcome_source,
                          a.reward, a.detail_json,
                          e.status AS experiment_status
                     FROM experiment_attempts a
                     JOIN experiments e ON e.experiment_id=a.experiment_id
                    WHERE a.experiment_id=? AND a.task_id=? AND a.attempt=?""",
                (experiment_id, task_id, attempt),
            ).fetchone()
            if row is None:
                conn.rollback()
                raise ExperimentNotFound(experiment_id)
            if row["execution_finished_at"] is None:
                conn.rollback()
                raise ValueError(
                    f"attempt {experiment_id}/{task_id}/{attempt} execution "
                    "is still open"
                )
            incoming = (outcome, source, reward_value, detail_json)
            stored = (
                row["outcome"],
                row["outcome_source"],
                row["reward"],
                row["detail_json"],
            )
            if row["outcome"] is not None:
                if stored == incoming:
                    conn.rollback()
                    return
                conn.rollback()
                raise AttemptValueConflict(
                    f"attempt {experiment_id}/{task_id}/{attempt} already has "
                    f"outcome {row['outcome']!r}; conflicting grades are refused"
                )
            if row["experiment_status"] not in {
                "running",
                "awaiting_evaluation",
            }:
                conn.rollback()
                raise ExperimentIsClosed(
                    experiment_id, row["experiment_status"]
                )
            conn.execute(
                """UPDATE experiment_attempts
                      SET outcome=?, outcome_source=?, reward=?, finished_at=?,
                          detail_json=?
                    WHERE experiment_id=? AND task_id=? AND attempt=?""",
                (
                    outcome,
                    source,
                    reward_value,
                    _utcnow_iso(),
                    detail_json,
                    experiment_id,
                    task_id,
                    attempt,
                ),
            )
            conn.commit()

    def finish_attempt(
        self,
        experiment_id: str,
        task_id: str,
        attempt: int,
        *,
        outcome: str,
        outcome_source: str,
        reward: Optional[float] = None,
        detail: Optional[dict[str, Any]] = None,
        conversation_id: Optional[int] = None,
    ) -> None:
        """Record an attempt's verdict (`[XR13]`).

        The verdict is WRITTEN, never derived from turn columns at read time.
        `outcome_source` names who decided -- a benchmark's reward function, a
        contract evaluator, an operator, or the literal `derived` for the
        turn-status fallback. Recording the source is what keeps a fallback from
        masquerading as a measurement.
        """
        self.terminalize_attempt(
            experiment_id,
            task_id,
            attempt,
            execution_status="completed",
            conversation_id=conversation_id,
        )
        self.record_attempt_outcome(
            experiment_id,
            task_id,
            attempt,
            outcome=outcome,
            outcome_source=outcome_source,
            reward=reward,
            detail=detail,
        )

    def restart_attempt(self, experiment_id: str, task_id: str, attempt: int) -> int:
        """Clear a crashed attempt so it can be re-run under the same labels.

        Deletes that attempt's conversations and turns in ONE transaction and
        bumps `restarts`. The deletion is deliberate (`[XR18]`): the abandoned
        partial trajectory is evidence of nothing, `idx_conv_experiment_attempt`
        is UNIQUE so a second conversation under the same three labels is
        refused outright, and leaving the rows would pin the attempt's derived
        diagnostic to 0 forever. `restarts` is what makes a task that keeps
        crashing visible rather than silently retried.

        Returns the number of turn rows deleted.
        """
        task_id = self._scrub(task_id)
        attempt = int(attempt)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """SELECT execution_finished_at FROM experiment_attempts
                    WHERE experiment_id=? AND task_id=? AND attempt=?""",
                (experiment_id, task_id, attempt),
            ).fetchone()
            if row is None:
                conn.rollback()
                raise ExperimentNotFound(experiment_id)
            if row["execution_finished_at"] is not None:
                conn.rollback()
                raise AttemptValueConflict(
                    f"attempt {experiment_id}/{task_id}/{attempt} execution is "
                    "terminal and cannot be restarted"
                )
            turn_keys = [
                r[0]
                for r in conn.execute(
                    """SELECT turn_key FROM turns
                        WHERE experiment_id=? AND task_id=? AND attempt=?""",
                    (experiment_id, task_id, attempt),
                ).fetchall()
            ]
            if turn_keys:
                for chunk in _chunked(turn_keys):
                    marks = ", ".join("?" for _ in chunk)
                    conn.execute(
                        f"DELETE FROM feedback WHERE turn_key IN ({marks})", chunk
                    )
                    conn.execute(
                        f"DELETE FROM artifacts WHERE turn_key IN ({marks})", chunk
                    )
                    conn.execute(
                        f"DELETE FROM spans WHERE trace_id IN ({marks})", chunk
                    )
            deleted = conn.execute(
                """DELETE FROM turns
                    WHERE experiment_id=? AND task_id=? AND attempt=?""",
                (experiment_id, task_id, attempt),
            ).rowcount
            conn.execute(
                """DELETE FROM conversations
                    WHERE experiment_id=? AND task_id=? AND attempt=?""",
                (experiment_id, task_id, attempt),
            )
            conn.execute(
                """UPDATE experiment_attempts
                      SET restarts=restarts+1, outcome=NULL, outcome_source=NULL,
                          reward=NULL, execution_status=NULL,
                          execution_finished_at=NULL, finished_at=NULL,
                          detail_json=NULL,
                          conversation_id=NULL, started_at=?
                    WHERE experiment_id=? AND task_id=? AND attempt=?""",
                (_utcnow_iso(), experiment_id, task_id, attempt),
            )
            conn.commit()
        return deleted

    def complete_experiment(
        self,
        experiment_id: str,
        *,
        force_invalid: Optional[str] = None,
        detail: Optional[str] = None,
    ) -> str:
        """Close an in-process experiment or defer it pending evaluation."""
        return self._complete_experiment(
            experiment_id,
            force_invalid=force_invalid,
            detail=detail,
            external_capture=False,
        )

    def complete_external_capture(self, experiment_id: str) -> str:
        """Close external execution without making its live store reportable."""
        return self._complete_experiment(
            experiment_id,
            force_invalid=None,
            detail=None,
            external_capture=True,
        )

    def _complete_experiment(
        self,
        experiment_id: str,
        *,
        force_invalid: Optional[str],
        detail: Optional[str],
        external_capture: bool,
    ) -> str:
        """Close an experiment. The STORE decides `complete` (`[XR14]`).

        The caller may request completion or force `invalid`; it may not assert
        completeness. A headline score rests on this verdict, so the store owns
        it rather than trusting a caller's word for it.

        `complete` requires all three: every declared (task, attempt) pair
        finished with an outcome, no outcome of `incomplete`, and no evidence
        segment marked invalid. Anything else is `invalid` with a closed reason
        code naming which check failed.

        `invalid` is TERMINAL: the UPDATE carries `AND status <> 'invalid'`, so
        neither a resume nor a later completion can clear a verdict recorded by
        `forget_channel` or by a failed evidence run.

        Returns the status actually stored.
        """
        if force_invalid is not None and force_invalid not in self._INVALID_REASONS:
            raise ValueError(
                f"invalid_reason {force_invalid!r} is not one of "
                f"{sorted(self._INVALID_REASONS)}"
            )
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """SELECT status, declared_tasks, declared_attempts,
                          required_evidence_segments
                     FROM experiments WHERE experiment_id=?""",
                (experiment_id,),
            ).fetchone()
            if row is None:
                conn.rollback()
                raise ExperimentNotFound(experiment_id)
            if row["status"] == "invalid":
                conn.rollback()
                return "invalid"
            if (
                row["status"] in {"complete", "capture_complete"}
                and force_invalid is None
            ):
                conn.rollback()
                return str(row["status"])

            reason: Optional[str] = force_invalid
            status: Optional[str] = None
            if reason is None:
                expected = int(row["declared_tasks"]) * int(row["declared_attempts"])
                # The SHAPE must match the declaration, not merely the count.
                # Counting finished rows against `expected` alone lets a row
                # outside the declared set pay for a declared pair that never
                # ran: a resume whose task list gained two tasks and lost one
                # reaches `finished == expected` with a declared task missing,
                # and `experiment_scores` then divides more scored attempts than
                # the denominator and reports pass@1 = 1.33 as a headline number.
                # So: every row finished, exactly as many rows as declared, and
                # exactly as many distinct tasks as declared.
                counts = conn.execute(
                    """SELECT
                         COUNT(*) AS rows_total,
                         COUNT(DISTINCT task_id) AS tasks_total,
                         SUM(CASE WHEN execution_finished_at IS NOT NULL
                                  THEN 1 ELSE 0 END) AS executed,
                         SUM(CASE WHEN finished_at IS NOT NULL AND outcome IS NOT NULL
                                  THEN 1 ELSE 0 END) AS evaluated,
                         SUM(CASE WHEN outcome='incomplete' THEN 1 ELSE 0 END)
                              AS incomplete
                       FROM experiment_attempts WHERE experiment_id=?""",
                    (experiment_id,),
                ).fetchone()
                rows_total = int(counts["rows_total"] or 0)
                tasks_total = int(counts["tasks_total"] or 0)
                executed = int(counts["executed"] or 0)
                evaluated = int(counts["evaluated"] or 0)
                incomplete = int(counts["incomplete"] or 0)
                declared_tasks = int(row["declared_tasks"])
                declarations = conn.execute(
                    """SELECT task_id, native_attempt, source_key
                         FROM experiment_attempt_declarations
                        WHERE experiment_id=?""",
                    (experiment_id,),
                ).fetchall()
                declaration_set = {
                    (
                        declaration["task_id"],
                        int(declaration["native_attempt"]),
                        declaration["source_key"],
                    )
                    for declaration in declarations
                }
                attempt_set = {
                    (
                        attempt_row["task_id"],
                        int(attempt_row["attempt"]),
                        attempt_row["source_key"],
                    )
                    for attempt_row in conn.execute(
                        """SELECT task_id, attempt, source_key
                             FROM experiment_attempts
                            WHERE experiment_id=?""",
                        (experiment_id,),
                    ).fetchall()
                }
                bad_segments = conn.execute(
                    """SELECT COUNT(*) FROM experiment_evidence_runs
                        WHERE experiment_id=? AND valid=0""",
                    (experiment_id,),
                ).fetchone()[0]
                evidence_segments = conn.execute(
                    """SELECT COUNT(*) FROM experiment_evidence_runs
                        WHERE experiment_id=?""",
                    (experiment_id,),
                ).fetchone()[0]
                if declaration_set and attempt_set != declaration_set:
                    if external_capture and attempt_set < declaration_set:
                        status = "running"
                    else:
                        reason = "attempt_shortfall"
                        detail = (
                            "recorded attempt identities do not exactly match the "
                            f"{len(declaration_set)} immutable declarations"
                        )
                elif (
                    executed != expected
                    or rows_total != expected
                    or tasks_total != declared_tasks
                ):
                    if (
                        external_capture
                        and executed <= expected
                        and rows_total <= expected
                        and tasks_total <= declared_tasks
                    ):
                        status = "running"
                    else:
                        reason = "attempt_shortfall"
                        detail = (
                            f"{executed} finished and {rows_total} recorded of "
                            f"{expected} declared attempts across {tasks_total} of "
                            f"{declared_tasks} declared tasks; {incomplete} incomplete"
                        )
                elif incomplete:
                    reason = "attempt_shortfall"
                    detail = (
                        f"{evaluated} evaluated of {expected} declared attempts; "
                        f"{incomplete} incomplete"
                    )
                elif bad_segments:
                    reason = "evidence_run_invalid"
                    detail = f"{bad_segments} evidence segment(s) reported invalid"
                elif evidence_segments < int(row["required_evidence_segments"]):
                    if external_capture:
                        status = "running"
                    else:
                        reason = "evidence_run_invalid"
                        detail = (
                            f"{evidence_segments} evidence segment(s) recorded; "
                            f"{int(row['required_evidence_segments'])} required"
                        )
                elif evaluated != expected:
                    status = "awaiting_evaluation"
                elif external_capture:
                    status = "capture_complete"
                else:
                    status = "complete"

            if reason is not None:
                status = "invalid"
            elif status is None:
                status = "complete"
            detail_text = self._scrub(detail) if reason is not None else None
            cursor = conn.execute(
                """UPDATE experiments
                      SET status=?, completed_at=?, invalid_reason=?,
                          invalid_detail=CASE
                              WHEN ? IS NULL THEN invalid_detail
                              WHEN invalid_detail IS NULL THEN ?
                              ELSE invalid_detail || char(10) || ? END
                    WHERE experiment_id=? AND status <> 'invalid'""",
                (
                    status,
                    None if status == "awaiting_evaluation" else _utcnow_iso(),
                    reason,
                    # Bound to None on the `complete` branch: `invalid_detail`
                    # is the explanation of an invalid verdict, and a detail
                    # string sitting on a complete experiment reads as one.
                    detail_text,
                    detail_text,
                    detail_text,
                    experiment_id,
                ),
            )
            if cursor.rowcount == 0:
                conn.rollback()
                return "invalid"
            conn.commit()
        return status

    def invalidate_experiments_in_txn(
        self,
        conn: sqlite3.Connection,
        experiment_ids: Iterable[str],
        reason: str,
        detail: Optional[str] = None,
    ) -> int:
        """Mark experiments invalid inside the caller's transaction.

        Used by `forget_channel`, which must not DELETE an experiment (44 of its
        45 attempts may live in other channels) but must never leave one
        scoreable after its turns are gone. `invalid_detail` is append-only so a
        second cause does not erase the first (`[XR15]`).
        """
        ids = [e for e in dict.fromkeys(experiment_ids) if e]
        if not ids:
            return 0
        scrubbed = self._scrub(detail)
        touched = 0
        for chunk in _chunked(ids):
            marks = ", ".join("?" for _ in chunk)
            touched += conn.execute(
                f"""UPDATE experiments
                       SET status='invalid', invalid_reason=?,
                           completed_at=COALESCE(completed_at, ?),
                           invalid_detail=CASE
                               WHEN ? IS NULL THEN invalid_detail
                               WHEN invalid_detail IS NULL THEN ?
                               ELSE invalid_detail || char(10) || ? END
                     WHERE experiment_id IN ({marks})""",
                [reason, _utcnow_iso(), scrubbed, scrubbed, scrubbed, *chunk],
            ).rowcount
        return touched

    # -- experiment reads ------------------------------------------------

    def get_experiment(self, experiment_id: str) -> Optional[dict[str, Any]]:
        """One experiment plus its evidence segments, or None."""
        if not self.has_feature(FEATURE_EXPERIMENTS_V1):
            return None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM experiments WHERE experiment_id=?", (experiment_id,)
            ).fetchone()
            if row is None:
                return None
            experiment = dict(row)
            experiment["archived"] = bool(experiment["archived"])
            segments = []
            for seg in conn.execute(
                """SELECT * FROM experiment_evidence_runs
                    WHERE experiment_id=? ORDER BY seq""",
                (experiment_id,),
            ).fetchall():
                segment = dict(seg)
                try:
                    segment["record"] = json.loads(segment.pop("record_json"))
                except (ValueError, KeyError):
                    segment["record"] = None
                segments.append(segment)
        experiment["evidence_runs"] = segments
        return experiment

    def list_experiments(
        self,
        status: Optional[str] = None,
        arm: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Experiments newest first, each with its observed attempt counts."""
        if not self.has_feature(FEATURE_EXPERIMENTS_V1):
            return []
        clauses: list[str] = []
        params: list[Any] = []
        if status is not None:
            clauses.append("e.status=?")
            params.append(status)
        if arm is not None:
            clauses.append("e.arm=?")
            params.append(arm)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        query = (
            "SELECT e.experiment_id, e.description, e.status, e.arm, "
            "e.baseline_experiment_id, e.declared_tasks, e.declared_attempts, "
            "e.invalid_reason, e.workflow_name, e.capture_profile, "
            "e.benchmark_id, e.benchmark_version, e.benchmark_digest_sha256, "
            "e.archived, e.created_at, e.completed_at, "
            "(SELECT COUNT(*) FROM experiment_attempts a "
            "  WHERE a.experiment_id=e.experiment_id) AS attempts_started, "
            "(SELECT COUNT(*) FROM experiment_attempts a "
            "  WHERE a.experiment_id=e.experiment_id AND a.finished_at IS NOT NULL "
            "    AND a.outcome IS NOT NULL) AS attempts_finished "
            f"FROM experiments e{where} ORDER BY e.created_at DESC LIMIT ? OFFSET ?"
        )
        params.extend([limit, offset])
        with self._connect() as conn:
            rows = [dict(r) for r in conn.execute(query, params).fetchall()]
        for row in rows:
            row["archived"] = bool(row["archived"])
        return rows

    def experiment_attempt_rows(
        self, experiment_id: str, task_id: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """Attempt rows, ordered by task then attempt."""
        if not self.has_feature(FEATURE_EXPERIMENTS_V1):
            return []
        clauses = ["experiment_id=?"]
        params: list[Any] = [experiment_id]
        if task_id is not None:
            clauses.append("task_id=?")
            params.append(task_id)
        with self._connect() as conn:
            rows = conn.execute(
                f"""SELECT * FROM experiment_attempts
                     WHERE {' AND '.join(clauses)}
                     ORDER BY task_id, attempt""",
                params,
            ).fetchall()
            return [_decode_attempt_row(r) for r in rows]

    def experiment_attempt_declarations(
        self, experiment_id: str
    ) -> list[dict[str, Any]]:
        """Return the immutable exact plan in deterministic order."""
        if not self.experiment_declaration_schema_ready():
            return []
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT task_id, native_attempt, source_key, created_at
                     FROM experiment_attempt_declarations
                    WHERE experiment_id=?
                    ORDER BY task_id, native_attempt, source_key""",
                (experiment_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def experiment_tasks(self, experiment_id: str) -> list[dict[str, Any]]:
        """One row per task: its attempts' outcomes, and whether all passed."""
        attempts = self.experiment_attempt_rows(experiment_id)
        by_task: dict[str, dict[str, Any]] = {}
        for row in attempts:
            task = by_task.setdefault(
                row["task_id"],
                {"task_id": row["task_id"], "attempts": [], "outcomes": []},
            )
            task["attempts"].append(row)
            task["outcomes"].append(row["outcome"])
        for task in by_task.values():
            outcomes = task["outcomes"]
            task["passed_all"] = bool(outcomes) and all(o == "pass" for o in outcomes)
            task["passed_any"] = any(o == "pass" for o in outcomes)
        return [by_task[k] for k in sorted(by_task)]

    def experiment_scores(self, experiment_id: str) -> dict[str, Any]:
        """pass@1 / pass^k over an experiment (`[XR13]`, `[XR14]`).

        Both are computed against the DECLARED denominator, never against
        surviving rows: a run that lost 12 of 45 attempts must not score 33/33
        and look perfect.

        A headline number is returned ONLY for `status='complete'`. A running or
        invalid experiment gets its per-task detail and its status in place of a
        score -- a provisional number in a UI becomes a quoted number in a
        document.
        """
        experiment = self.get_experiment(experiment_id)
        if experiment is None:
            raise ExperimentNotFound(experiment_id)
        tasks = self.experiment_tasks(experiment_id)
        declared_tasks = int(experiment["declared_tasks"])
        declared_attempts = int(experiment["declared_attempts"])
        expected = declared_tasks * declared_attempts
        scored = [
            row
            for task in tasks
            for row in task["attempts"]
            if row["finished_at"] is not None and row["outcome"] is not None
        ]
        result: dict[str, Any] = {
            "experiment_id": experiment_id,
            "status": experiment["status"],
            "invalid_reason": experiment["invalid_reason"],
            "declared_tasks": declared_tasks,
            "declared_attempts": declared_attempts,
            "expected_attempts": expected,
            "scored_attempts": len(scored),
            "tasks": tasks,
            "outcome_sources": sorted(
                {row["outcome_source"] for row in scored if row["outcome_source"]}
            ),
            "pass_at_1": None,
            "pass_at_k": None,
            "reportable": False,
            "workspace_archive_sha256": experiment.get(
                "workspace_archive_sha256"
            ),
            "workspace_store_identity": experiment.get(
                "workspace_store_identity"
            ),
        }
        if experiment["status"] != "complete":
            result["reason_not_reportable"] = (
                f"experiment status is {experiment['status']!r}; a score is only "
                "reportable for a complete experiment"
            )
            return result
        if experiment.get("evidence_sealed_at") and not experiment.get(
            "workspace_archive_sha256"
        ):
            # A seal in two halves (fix-tcg): `begin_workspace_seal` stamped
            # `complete` so the ARCHIVE would carry it, and the archive never
            # landed. `complete` alone would otherwise make this reportable, and
            # a headline number resting on sealed evidence that does not exist
            # is the exact failure sealing was added to prevent.
            result["reason_not_reportable"] = (
                "a workspace seal was started for this experiment and recorded "
                "no archive digest; the sealed evidence a score would rest on "
                "does not exist. Re-run the seal."
            )
            return result
        if len(scored) != expected or len(tasks) != declared_tasks:
            # Unreachable while `complete_experiment` is the only way to reach
            # `complete`, and kept anyway: this function divides by the DECLARED
            # denominator, so a set of rows that does not match the declaration
            # produces a ratio above 1.0 rather than an error. A score that can
            # exceed 1.0 is worse than no score.
            result["reportable"] = False
            result["reason_not_reportable"] = (
                f"{len(scored)} scored attempts across {len(tasks)} tasks do not "
                f"match the declared {expected} across {declared_tasks}; the "
                "experiment is marked complete but its rows do not support a score"
            )
            return result
        passed = sum(1 for row in scored if row["outcome"] == "pass")
        # pass^k is over DECLARED tasks: a task with no attempt row at all is a
        # task that did not pass every attempt, and dividing by the tasks that
        # happen to be present is the denominator error this guards against.
        all_passed = sum(1 for task in tasks if task["passed_all"])
        result["pass_at_1"] = passed / expected
        result["pass_at_k"] = all_passed / declared_tasks
        result["reportable"] = True
        return result

    def compare_experiments(
        self, experiment_id: str, baseline_experiment_id: str
    ) -> dict[str, Any]:
        """Treatment vs baseline, per task (`[XR19]`).

        Reports flip counts and sample size; it does NOT claim significance. A
        query layer that emits a p-value is a query layer that will be quoted as
        if it had run the protocol.

        Refuses unless both are complete, both declare the same shape, their
        task-id SETS are equal, and they were captured under the same profile.
        Cardinality is not comparability: two 15x3 runs over disjoint task sets
        would otherwise report "0 regressions" while sharing no task.
        """
        treatment = self.get_experiment(experiment_id)
        baseline = self.get_experiment(baseline_experiment_id)
        if treatment is None:
            raise ExperimentNotFound(experiment_id)
        if baseline is None:
            raise ExperimentNotFound(baseline_experiment_id)
        problems: list[str] = []
        for side, exp in (("treatment", treatment), ("baseline", baseline)):
            if exp["status"] != "complete":
                problems.append(
                    f"{side} {exp['experiment_id']} is {exp['status']!r}, not complete"
                )
        if (treatment["declared_tasks"], treatment["declared_attempts"]) != (
            baseline["declared_tasks"],
            baseline["declared_attempts"],
        ):
            problems.append(
                f"declared shapes differ: treatment "
                f"{treatment['declared_tasks']}x{treatment['declared_attempts']} "
                f"vs baseline {baseline['declared_tasks']}x"
                f"{baseline['declared_attempts']}"
            )
        if treatment["capture_profile"] != baseline["capture_profile"] or (
            treatment["capture_policy_version"] != baseline["capture_policy_version"]
        ):
            problems.append(
                f"capture regimes differ: treatment "
                f"{treatment['capture_profile']}/"
                f"{treatment['capture_policy_version']} vs baseline "
                f"{baseline['capture_profile']}/"
                f"{baseline['capture_policy_version']}; the two arms are not "
                "measuring the same columns"
            )
        treatment_pin = self._experiment_benchmark_pin(treatment)
        baseline_pin = self._experiment_benchmark_pin(baseline)
        if treatment_pin is not None or baseline_pin is not None:
            if treatment_pin != baseline_pin:
                if treatment_pin is None:
                    problems.append(
                        "benchmark pins differ: treatment is unpinned but "
                        f"baseline is pinned to "
                        f"{baseline_pin[0]}/{baseline_pin[1]}"
                    )
                elif baseline_pin is None:
                    problems.append(
                        "benchmark pins differ: baseline is unpinned but "
                        f"treatment is pinned to "
                        f"{treatment_pin[0]}/{treatment_pin[1]}"
                    )
                else:
                    problems.append(
                        "benchmark pins differ: treatment "
                        f"{treatment_pin[0]}/{treatment_pin[1]}/"
                        f"{treatment_pin[2]} vs baseline "
                        f"{baseline_pin[0]}/{baseline_pin[1]}/"
                        f"{baseline_pin[2]}"
                    )
        t_tasks = {t["task_id"]: t for t in self.experiment_tasks(experiment_id)}
        b_tasks = {
            t["task_id"]: t for t in self.experiment_tasks(baseline_experiment_id)
        }
        only_treatment = sorted(set(t_tasks) - set(b_tasks))
        only_baseline = sorted(set(b_tasks) - set(t_tasks))
        if only_treatment or only_baseline:
            problems.append(
                f"task sets differ: {len(only_treatment)} only in treatment, "
                f"{len(only_baseline)} only in baseline"
            )
        if problems:
            return {
                "comparable": False,
                "problems": problems,
                "only_in_treatment": only_treatment,
                "only_in_baseline": only_baseline,
            }
        improved, regressed, unchanged = [], [], []
        expected_flips = 0.0
        k = int(treatment["declared_attempts"])
        for task_id in sorted(t_tasks):
            t_pass = t_tasks[task_id]["passed_all"]
            b_pass = b_tasks[task_id]["passed_all"]
            if t_pass and not b_pass:
                improved.append(task_id)
            elif b_pass and not t_pass:
                regressed.append(task_id)
            else:
                unchanged.append(task_id)
            expected_flips += self._expected_flip_probability(
                t_tasks[task_id], b_tasks[task_id], k
            )
        return {
            "comparable": True,
            "problems": [],
            "treatment": self.experiment_scores(experiment_id),
            "baseline": self.experiment_scores(baseline_experiment_id),
            "improved": improved,
            "regressed": regressed,
            "unchanged": unchanged,
            "tasks_compared": len(t_tasks),
            "attempts_per_task": k,
            "expected_flips_if_nothing_changed": round(expected_flips, 3),
            "observed_flips": len(improved) + len(regressed),
        }

    @staticmethod
    def _expected_flip_probability(
        treatment_task: dict[str, Any], baseline_task: dict[str, Any], k: int
    ) -> float:
        """How often this task's pass^k verdict would flip if NOTHING changed.

        "How many flips are attributable to variance rather than the change"
        has an answer that does not require
        claiming significance, and this is it. Pool both arms' attempts for one
        task to estimate a single per-attempt pass rate p, then a flip in either
        direction has probability 2 * p^k * (1 - p^k) under the hypothesis that
        the arms are identical. Summed over tasks, that is the number of flips a
        pair of arms that differ in nothing would be expected to produce.

        **What this is not.** It is not a p-value and it is not a test. It is an
        expectation under one crude null, offered so that "3 tasks flipped"
        stops reading as "3 tasks improved" when the expected number is 2.6. The
        statistical protocol lives outside this file, deliberately: a query layer
        that emits a significance verdict is a query layer that will be quoted as
        if it had run one.

        A task with no attempts contributes 0: nothing that was never run can
        flip.
        """
        outcomes = [
            o
            for o in (treatment_task["outcomes"] + baseline_task["outcomes"])
            if o is not None
        ]
        if not outcomes or k <= 0:
            return 0.0
        p = sum(1 for o in outcomes if o == "pass") / len(outcomes)
        p_all = p ** k
        return 2.0 * p_all * (1.0 - p_all)

    def experiment_labels_for_turn(
        self, turn_key: str
    ) -> Optional[dict[str, Any]]:
        """Return experiment labels attached to a turn, if any."""
        if not self.has_feature(FEATURE_EXPERIMENTS_V1):
            return None
        with self._connect() as conn:
            row = conn.execute(
                """SELECT t.experiment_id, t.task_id, t.attempt, e.description, e.status
                     FROM turns t LEFT JOIN experiments e
                       ON e.experiment_id = t.experiment_id
                    WHERE t.turn_key=? AND t.experiment_id IS NOT NULL""",
                (turn_key,),
            ).fetchone()
            return dict(row) if row is not None else None

    def list_distillation_runs(
        self, *, experiment_id: Optional[str] = None, **_: Any
    ) -> list[dict[str, Any]]:
        """Compatibility read seam; distillation is not shipped by this port."""
        return []

    # -- maintenance [R12] and erasure [R21] -----------------------------

    def db_size_bytes(self) -> int:
        """DB file size including the -wal sidecar."""
        total = 0
        for path in (self.db_path, f"{self.db_path}-wal"):
            try:
                total += os.path.getsize(path)
            except OSError:
                pass
        return total

    @staticmethod
    def _file_digest(path: str) -> Optional[dict[str, Any]]:
        try:
            digest = hashlib.sha256()
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            stat = os.stat(path)
            return {
                "size_bytes": stat.st_size,
                "sha256": digest.hexdigest(),
            }
        except FileNotFoundError:
            return None

    def archive_to(
        self, destination: str, *, quiesce_live_writer: bool = True
    ) -> dict[str, Any]:
        """Seal a source-read-only snapshot, including committed WAL content.

        The source is opened with ``mode=ro`` and never through ``_connect``,
        whose journal-mode pragma is intentionally write-capable. SQLite's
        backup API reads one consistent transaction including committed WAL.
        Only that copied database is vacuumed into the final destination.

        THE LIVE WRITER. Refusing outright was the wrong half of a
        true idea. The idea is that a snapshot must not be taken of a moving
        file — the digest recorded beside the archive is a claim about bytes
        that were still, and `SourceChangedDuringArchive` is what happens when
        they were not. The wrong half was concluding that the only writer a
        snapshot can survive is a dead one: an in-process run that wants an
        archive of its own evidence then has to kill the writer that is
        recording it, and `ExperimentRunner` — which takes the process sink
        BEFORE opening `evidence_run`, precisely so the verdict rests on live
        counters — could never produce an archive at all.

        So a writer THIS process owns is held still instead: flushed, parked off
        its heartbeat, its counters persisted and its WAL folded back in, for
        the duration of the snapshot (`SQLiteTraceSink.quiesced`). The one-writer
        contract is untouched — no second writer is created, and the one writer
        there is simply stops for a moment. A writer this process cannot reach
        cannot be held still, so that case still refuses: an archive is either
        provably of a stopped store or it is not taken.

        `quiesce_live_writer=False` restores the unconditional refusal for a
        caller whose contract is "the writer must already be gone" — the seal
        path checks that itself, before it promotes the experiment's status.
        """
        target = Path(destination)
        if target.exists():
            raise FileExistsError(
                f"refusing to overwrite an existing evidence archive: {target}"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        source = os.path.abspath(self.db_path)
        live_sink = sink_for_db_path(source)
        if live_sink is not None and not live_sink._closed:
            if not quiesce_live_writer:
                raise WriterStillOpen(
                    f"refusing to seal {source!r} while its writer is open"
                )
            with live_sink.quiesced():
                return self._snapshot_to(target, source)
        self._refuse_if_an_unreachable_writer_holds(source)
        return self._snapshot_to(target, source)

    def _refuse_if_an_unreachable_writer_holds(self, source: str) -> None:
        """Refuse a snapshot of a store some OTHER writer is still holding.

        `sink_for_db_path` only sees the sinks this process's factory minted.
        Without the incarnation stamp the writer-health row now carries, there
        would be nothing to consult about a writer living anywhere else — a
        server in another process, or a sink built directly and never
        registered. Such a run would not refuse; it would race, and the race
        surfaces as `SourceChangedDuringArchive` if it is caught at all.

        The stamp is only trusted where it can be checked. A row left open by a
        writer whose process is gone is a crash marker, not a live writer, and
        must not wedge every future archive of the store — so the refusal needs
        the pid to still exist on this host. A recycled pid can therefore hold a
        seal off for one extra run; refusing an archive that could have been
        taken is recoverable, and taking one of a store being written is not.
        """
        try:
            stamp = (self.writer_health() or {}).get(WRITER_INCARNATION_FIELD)
        except Exception:  # pragma: no cover - defensive; health is diagnostics
            return
        if not isinstance(stamp, Mapping) or not stamp.get("open"):
            return
        if str(stamp.get("host") or "") != socket.gethostname():
            return
        try:
            pid = int(stamp.get("pid") or 0)
        except (TypeError, ValueError):
            return
        if pid <= 0:
            return
        if pid != os.getpid():
            try:
                os.kill(pid, 0)
            except OSError:
                return  # the writer's process is gone; the marker is stale
        raise WriterStillOpen(
            f"refusing to seal {source!r}: writer incarnation "
            f"{stamp.get('id')} (pid {pid}) still holds it and is not reachable "
            f"from this process, so it cannot be held still for the snapshot"
        )

    def _snapshot_to(self, target: Path, source: str) -> dict[str, Any]:
        """Take the snapshot. The caller has already settled the source."""
        source_paths = (source, f"{source}-wal")
        # One connection held open, doing nothing, for the whole snapshot.
        # SQLite checkpoints a WAL database when its LAST connection closes, so
        # an unrelated reader letting go mid-snapshot rewrites `-wal` and the
        # comparison below reports a change to a source nobody wrote to. This
        # pin makes that close never the last one; no statement is ever run on
        # it, and it is what makes "source bytes verified unchanged" a fact
        # about the source rather than about the timing of a garbage collection.
        pin = self._connect()
        before = {path: self._file_digest(path) for path in source_paths}
        confirmed_before = {
            path: self._file_digest(path) for path in source_paths
        }
        if before != confirmed_before:
            with contextlib.suppress(Exception):
                pin.close()
            raise SourceChangedDuringArchive(
                "source DB/WAL bytes changed before the snapshot could start"
            )
        before = confirmed_before
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.snapshot")
        compacted = target.with_name(f".{target.name}.{uuid.uuid4().hex}.compact")
        try:
            source_uri = Path(source).as_uri() + "?mode=ro"
            with sqlite3.connect(source_uri, uri=True) as source_conn:
                with sqlite3.connect(str(temporary)) as snapshot_conn:
                    source_conn.backup(snapshot_conn)
            after_backup = {
                path: self._file_digest(path) for path in source_paths
            }
            if before != after_backup:
                raise SourceChangedDuringArchive(
                    "source DB/WAL bytes changed while taking the snapshot"
                )
            with sqlite3.connect(str(temporary)) as snapshot_conn:
                snapshot_conn.execute("VACUUM INTO ?", (str(compacted),))
            os.replace(compacted, target)
            after_compaction = {
                path: self._file_digest(path) for path in source_paths
            }
            if before != after_compaction:
                raise SourceChangedDuringArchive(
                    "source DB/WAL bytes changed while compacting the destination"
                )
            for sidecar in (f"{target}-wal", f"{target}-shm"):
                if os.path.exists(sidecar):
                    raise RuntimeError(
                        f"sealed archive unexpectedly has sidecar {sidecar!r}"
                    )
            archive_digest = self._file_digest(str(target))
            if archive_digest is None:
                raise RuntimeError("archive disappeared before verification")
            archive_uri = target.resolve().as_uri() + "?mode=ro"
            with sqlite3.connect(archive_uri, uri=True) as archive_conn:
                identity_row = archive_conn.execute(
                    "SELECT value FROM diagnostics WHERE key=?",
                    (STORE_IDENTITY_DIAGNOSTIC,),
                ).fetchone()
                integrity = archive_conn.execute(
                    "PRAGMA integrity_check"
                ).fetchone()[0]
            if integrity != "ok":
                raise RuntimeError(f"archive integrity check failed: {integrity}")
            if identity_row is None or not identity_row[0]:
                raise RuntimeError("archive has no durable store identity")
            target.chmod(0o444)
            return {
                "path": str(target),
                "size_bytes": archive_digest["size_bytes"],
                "sha256": archive_digest["sha256"],
                "store_identity": str(identity_row[0]),
                "schema_version": SCHEMA_VERSION,
                "read_only": True,
                "sealed": True,
                "source_bytes_verified_unchanged": True,
                "sidecar_free": True,
            }
        except Exception:
            with contextlib.suppress(FileNotFoundError):
                target.unlink()
            raise
        finally:
            with contextlib.suppress(Exception):
                pin.close()
            for scratch in (temporary, compacted):
                with contextlib.suppress(FileNotFoundError):
                    scratch.unlink()
                for suffix in ("-wal", "-shm"):
                    with contextlib.suppress(FileNotFoundError):
                        Path(f"{scratch}{suffix}").unlink()

    def prune(
        self,
        retention_days: Optional[int] = None,
        max_bytes: Optional[int] = None,
        include_conversationless_turns: bool = False,
    ) -> dict[str, int]:
        """Bounded prune of spans/artifacts beyond the retention horizon, plus
        oldest-first eviction while over the size cap. Conversations and turn
        records are exempt (config §5). Offload evidence is pruned by the same
        horizon and cap, one whole turn at a time. Runs incremental_vacuum.

        ``include_conversationless_turns`` (operator opt-in) also
        deletes conversation-less turn records (e.g. per-invocation CLI
        channels) older than the horizon, with their feedback — otherwise no
        retention knob ever reaches them.
        """
        if pruning_suppressed():
            return {"suppressed": 1}
        if retention_days is None:
            retention_days = _env_int("FW_OBS_RETENTION_DAYS", _DEFAULT_RETENTION_DAYS)
        if max_bytes is None:
            max_bytes = _env_int("FW_OBS_DB_MAX_BYTES", _DEFAULT_DB_MAX_BYTES)

        horizon_ns = int(
            (time.time() - retention_days * 86_400) * 1_000_000_000
        )
        horizon_moment = datetime.fromtimestamp(
            max(0.0, time.time() - retention_days * 86_400), tz=timezone.utc
        )
        horizon_key = horizon_moment.strftime("%Y%m%dT%H%M%S")
        # Offload evidence is stamped in this format (see
        # `observation_offloading.archive`), so the horizon compares as text.
        horizon_evidence = horizon_moment.strftime("%Y-%m-%dT%H:%M:%SZ")
        deleted = {
            "spans": 0, "artifacts": 0,
            **{table: 0 for table in _OFFLOAD_EVIDENCE_TABLES},
        }
        erased_scopes: set[str] = set()

        with self._connect() as conn:
            # As in `forget_channel`: retention deletes evidence text, and a
            # deleted cell must not survive in a page that still holds others.
            conn.execute("PRAGMA secure_delete=ON")
            for _ in range(_PRUNE_MAX_BATCHES):
                conn.execute("BEGIN IMMEDIATE")
                spans_cur = conn.execute(
                    "DELETE FROM spans WHERE span_id IN "
                    "(SELECT span_id FROM spans WHERE start_ns < ? LIMIT ?)",
                    (horizon_ns, _PRUNE_BATCH_ROWS),
                )
                deleted["spans"] += spans_cur.rowcount
                artifacts_cur = conn.execute(
                    "DELETE FROM artifacts WHERE artifact_id IN "
                    "(SELECT artifact_id FROM artifacts WHERE turn_key < ? LIMIT ?)",
                    (horizon_key, _PRUNE_BATCH_ROWS),
                )
                deleted["artifacts"] += artifacts_cur.rowcount
                # Offload evidence is aged by its TURN, like artifacts, so a
                # turn's observations go whole. A turn's age is its earliest
                # evidence write rather than its key, because a turn with no
                # bound turn key is keyed by its channel id, which does not
                # sort by time.
                aged_turns = self._offload_turns_in_txn(
                    conn, before=horizon_evidence, limit=_OFFLOAD_PRUNE_BATCH_TURNS
                )
                self._delete_offload_turns_in_txn(
                    conn, aged_turns, deleted, erased_scopes
                )
                conn.commit()
                if (
                    spans_cur.rowcount < _PRUNE_BATCH_ROWS
                    and artifacts_cur.rowcount < _PRUNE_BATCH_ROWS
                    and len(aged_turns) < _OFFLOAD_PRUNE_BATCH_TURNS
                ):
                    break

            if include_conversationless_turns:
                deleted["conversationless_turns"] = 0
                for _ in range(_PRUNE_MAX_BATCHES):
                    conn.execute("BEGIN IMMEDIATE")
                    keys = [
                        r[0]
                        for r in conn.execute(
                            "SELECT turn_key FROM turns WHERE conversation_id IS NULL "
                            "AND turn_key < ? LIMIT ?",
                            (horizon_key, _PRUNE_BATCH_ROWS),
                        ).fetchall()
                    ]
                    for key in keys:
                        conn.execute("DELETE FROM feedback WHERE turn_key=?", (key,))
                        conn.execute("DELETE FROM spans WHERE trace_id=?", (key,))
                        conn.execute("DELETE FROM artifacts WHERE turn_key=?", (key,))
                        conn.execute("DELETE FROM turns WHERE turn_key=?", (key,))
                    self._delete_offload_turns_in_txn(
                        conn, keys, deleted, erased_scopes
                    )
                    conn.commit()
                    deleted["conversationless_turns"] += len(keys)
                    if len(keys) < _PRUNE_BATCH_ROWS:
                        break

            # Size-cap eviction, oldest spans first (turn keys sort by time),
            # and the oldest offload evidence turns beside them: the evidence
            # shares this file, so it shares this cap. Each batch vacuums, so
            # the next measurement sees the pages the deletes freed.
            for _ in range(_PRUNE_MAX_BATCHES):
                if self.db_size_bytes() <= max_bytes:
                    break
                conn.execute("BEGIN IMMEDIATE")
                cur = conn.execute(
                    "DELETE FROM spans WHERE span_id IN "
                    "(SELECT span_id FROM spans ORDER BY start_ns LIMIT ?)",
                    (_PRUNE_BATCH_ROWS,),
                )
                oldest_turns = self._offload_turns_in_txn(
                    conn, before=None, limit=_OFFLOAD_PRUNE_BATCH_TURNS
                )
                self._delete_offload_turns_in_txn(
                    conn, oldest_turns, deleted, erased_scopes
                )
                conn.commit()
                if cur.rowcount == 0 and not oldest_turns:
                    break
                # Fetched to completion: each step of this pragma frees one page.
                conn.execute("PRAGMA incremental_vacuum").fetchall()
                conn.commit()
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

            conn.execute("PRAGMA incremental_vacuum")
            conn.commit()
            # Fold the deletes back into the main file now, so the evidence
            # they removed does not wait there for the next checkpoint.
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

        if erased_scopes:
            deleted["offload_scopes_released"] = agent_runtime.reclaim_erased_scopes(
                erased_scopes
            )
        return deleted

    @staticmethod
    def _offload_turns_in_txn(
        conn: sqlite3.Connection, *, before: Optional[str], limit: int
    ) -> list[str]:
        """The oldest offload-evidence turns, optionally only those begun before *before*.

        A turn's age is the earliest timestamp on any of its evidence,
        subject or event rows -- when the turn began -- so a turn is always
        dropped whole and never leaves a subject or an event whose evidence is
        gone.
        """
        tables = _present_offload_tables(conn)
        if not tables:
            return []
        having = "HAVING MIN(at) < ?" if before is not None else ""
        params: list[Any] = [before] if before is not None else []
        params.append(limit)
        dated = " UNION ALL ".join(
            f"SELECT turn_key, {_OFFLOAD_TABLE_TIMESTAMPS[table]} AS at FROM {table}"
            for table in tables
        )
        return [
            str(row[0])
            for row in conn.execute(
                f"SELECT turn_key FROM ({dated}) GROUP BY turn_key {having} "
                "ORDER BY MIN(at), turn_key LIMIT ?",
                params,
            ).fetchall()
        ]

    @staticmethod
    def _delete_offload_turns_in_txn(
        conn: sqlite3.Connection,
        turn_keys: list[str],
        deleted: dict[str, int],
        erased_scopes: set[str],
    ) -> None:
        """Delete these turns' offload evidence, tallying rows and scopes."""
        tables = _present_offload_tables(conn)
        for chunk in _chunked(list(turn_keys)):
            marks = ",".join("?" for _ in chunk)
            for table in tables:
                erased_scopes.update(
                    str(row[0])
                    for row in conn.execute(
                        f"SELECT DISTINCT scope_id FROM {table} "
                        f"WHERE turn_key IN ({marks})",
                        chunk,
                    ).fetchall()
                )
                deleted[table] = deleted.get(table, 0) + conn.execute(
                    f"DELETE FROM {table} WHERE turn_key IN ({marks})", chunk
                ).rowcount

    def offload_events(
        self,
        *,
        turn_key: Optional[str] = None,
        channel_id: Optional[str] = None,
        scope_id: Optional[str] = None,
        kind: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """The offload runtime's recorded events, oldest first.

        Every filter given must match. Each item carries the row's keys and
        capture record beside ``event_text``, the event as it was stored --
        the JSON the runtime recorded, with redaction applied when
        ``redaction`` is ``on`` -- and ``event``, that text parsed, or ``None``
        when it does not parse as a JSON object. Under a capture profile that
        withholds the event, ``event`` is the withholding badge rather than the
        recorded dictionary. A DB that predates the table reads as having no
        events.
        """
        clauses: list[str] = []
        params: list[Any] = []
        for column, value in (
            ("turn_key", turn_key), ("channel_id", channel_id),
            ("scope_id", scope_id), ("kind", kind),
        ):
            if value is not None:
                clauses.append(f"{column}=?")
                params.append(str(value))
        query = (
            "SELECT event_id, turn_key, channel_id, scope_id, kind, event_json, "
            "redaction, redacted, recorded_at FROM offload_events"
        )
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY event_id"
        if limit is not None:
            query += " LIMIT ?"
            params.append(int(limit))
        with contextlib.closing(self._connect()) as conn:
            if "offload_events" not in _present_offload_tables(conn):
                return []
            rows = conn.execute(query, params).fetchall()
        events: list[dict[str, Any]] = []
        for row in rows:
            text = str(row["event_json"])
            try:
                parsed = json.loads(text)
            except ValueError:
                parsed = None
            events.append({
                "event_id": int(row["event_id"]),
                "turn_key": str(row["turn_key"]),
                "channel_id": str(row["channel_id"]),
                "scope_id": str(row["scope_id"]),
                "kind": str(row["kind"]),
                "event": parsed if isinstance(parsed, dict) else None,
                "event_text": text,
                "redaction": str(row["redaction"]),
                "redacted": bool(row["redacted"]),
                "recorded_at": str(row["recorded_at"]),
            })
        return events

    def forget_channel(self, channel_id: str) -> dict[str, int]:
        """First-class erasure: delete a channel across all tables, then
        checkpoint-truncate the WAL and reclaim pages.

        "All tables" includes the offload evidence tables, which hold the
        channel's archived execute responses. They are deleted in the same
        transaction as the turn records, by channel and by the channel's turn
        keys, whether or not a turn belonged to an experiment run. After the
        commit, the process-local caches of the erased turns are dropped too
        (``offload_scopes_released`` counts the scopes released).
        """
        deleted: dict[str, int] = {}
        erased_scopes: set[str] = set()
        with self._connect() as conn:
            # Deleted cells are zeroed rather than left in the free space of
            # pages that still hold other rows, so erased evidence text does
            # not survive in the file beside the rows that were kept.
            conn.execute("PRAGMA secure_delete=ON")
            conn.execute("BEGIN IMMEDIATE")
            touched_experiments = [
                row[0]
                for row in conn.execute(
                    "SELECT DISTINCT experiment_id FROM experiment_attempts "
                    "WHERE channel_id=?",
                    (channel_id,),
                ).fetchall()
            ]
            deleted["feedback"] = conn.execute(
                "DELETE FROM feedback WHERE turn_key IN "
                "(SELECT turn_key FROM turns WHERE channel_id=?)",
                (channel_id,),
            ).rowcount
            deleted["spans"] = conn.execute(
                "DELETE FROM spans WHERE channel_id=? OR trace_id IN "
                "(SELECT turn_key FROM turns WHERE channel_id=?)",
                (channel_id, channel_id),
            ).rowcount
            deleted["artifacts"] = conn.execute(
                "DELETE FROM artifacts WHERE channel_id=? OR turn_key IN "
                "(SELECT turn_key FROM turns WHERE channel_id=?)",
                (channel_id, channel_id),
            ).rowcount
            for table in _present_offload_tables(conn):
                erased_scopes.update(
                    str(row[0])
                    for row in conn.execute(
                        f"SELECT DISTINCT scope_id FROM {table} WHERE channel_id=? "
                        "OR turn_key IN (SELECT turn_key FROM turns WHERE channel_id=?)",
                        (channel_id, channel_id),
                    ).fetchall()
                )
                deleted[table] = conn.execute(
                    f"DELETE FROM {table} WHERE channel_id=? OR turn_key IN "
                    "(SELECT turn_key FROM turns WHERE channel_id=?)",
                    (channel_id, channel_id),
                ).rowcount
            deleted["turns"] = conn.execute(
                "DELETE FROM turns WHERE channel_id=?", (channel_id,)
            ).rowcount
            deleted["conversations"] = conn.execute(
                "DELETE FROM conversations WHERE channel_id=?", (channel_id,)
            ).rowcount
            # The experiment container `[XR15]`. An experiment is NOT the
            # channel's to delete -- 44 of its 45 attempts may live in other
            # channels -- but it must never stay scoreable once its turns are
            # gone, because after this its denominator is unreconstructable.
            # So: delete this channel's attempt rows, and mark every experiment
            # they belonged to terminally invalid. Ids collected BEFORE the
            # deletes, per [DR44].
            deleted["experiment_attempts"] = conn.execute(
                "DELETE FROM experiment_attempts WHERE channel_id=?", (channel_id,)
            ).rowcount
            invalidated = self.invalidate_experiments_in_txn(
                conn,
                touched_experiments,
                "turns_erased",
                f"turns erased by forget_channel for channel {channel_id!r}",
            )
            if invalidated:
                deleted["experiments_invalidated"] = invalidated
            conn.commit()
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            conn.execute("PRAGMA incremental_vacuum")
            conn.commit()

        deleted["offload_scopes_released"] = agent_runtime.reclaim_erased_scopes(
            erased_scopes
        )
        return deleted

    def clear_conversations(self) -> dict[str, int]:
        """Delete every recorded conversation and its turn-level observability.

        Training runs, writer diagnostics, and monotonic conversation counters
        survive. Keeping counters prevents a clear operation from reusing a
        conversation identity that may still be referenced outside this DB.

        This is the action the chatbot UI actually exposes, so it deletes
        every offload evidence row too, in the same transaction -- experiment
        runs' evidence included, as their experiment records are.
        """
        deleted: dict[str, int] = {}
        erased_scopes: set[str] = set()
        with self._connect() as conn:
            # Deleted cells are zeroed rather than left in the free space of
            # pages that still hold other rows, so erased evidence text does
            # not survive in the file beside the rows that were kept.
            conn.execute("PRAGMA secure_delete=ON")
            conn.execute("BEGIN IMMEDIATE")
            for table in (
                "experiment_evidence_runs",
                "experiment_attempts",
                "experiment_attempt_declarations",
                "experiments",
            ):
                deleted[table] = conn.execute(f"DELETE FROM {table}").rowcount
            offload_tables = _present_offload_tables(conn)
            for table in offload_tables:
                erased_scopes.update(
                    str(row[0])
                    for row in conn.execute(
                        f"SELECT DISTINCT scope_id FROM {table}"
                    ).fetchall()
                )
            for table in (
                "feedback", "spans", "artifacts", *offload_tables,
                "turns", "conversations",
            ):
                deleted[table] = conn.execute(f"DELETE FROM {table}").rowcount
            conn.commit()
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            conn.execute("PRAGMA incremental_vacuum")
            conn.commit()

        deleted["offload_scopes_released"] = agent_runtime.reclaim_erased_scopes(
            erased_scopes
        )
        return deleted


class ReadOnlyObservabilityStore(ObservabilityStore):
    """Read-only view of an existing observability DB (the chatbot's debug
    layer). Never creates, migrates, or writes the file — the viewer must be
    able to open a post-mortem snapshot it does not own, and inspecting a DB
    must not mutate it. Construction raises when the file is absent/unopenable
    (``sqlite3.OperationalError``) or written by a different build, newer or
    older (``IncompatibleObservabilityDB``); callers degrade gracefully. Unlike
    the writer, it never replaces an older DB: it only refuses it.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        conn = self._connect()
        try:
            found = conn.execute("PRAGMA user_version").fetchone()[0]
            if found > SCHEMA_VERSION:
                raise IncompatibleObservabilityDB(
                    f"{self.db_path} has schema v{found}; this build reads up to "
                    f"v{SCHEMA_VERSION}. Refusing to open a newer DB [R11]."
                )
            if found < SCHEMA_VERSION:
                # No migration (fresh schema, fix-49m.3): an older store is
                # refused up front with the reason, instead of failing later on
                # a column the reader assumes exists. The writer replaces such
                # a store; a reader must never delete what it inspects.
                raise IncompatibleObservabilityDB(
                    f"{self.db_path} has schema v{found}; this build requires "
                    f"v{SCHEMA_VERSION} and carries no migration (fresh "
                    "observability schema, fix-49m.3). Open it with a "
                    f"v{found} build."
                )
        finally:
            conn.close()
        self._features = self._load_features()

    def _connect(self, timeout: float = 30.0) -> sqlite3.Connection:
        conn = sqlite3.connect(
            f"file:{self.db_path}?mode=ro",
            uri=True,
            timeout=timeout,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        return conn


# ----------------------------------------------------------------------
# The sink: two queues + one daemon writer thread [R7][R8][R13]
# ----------------------------------------------------------------------


class SQLiteTraceSink:
    """TraceSink writing to an ObservabilityStore via a background thread.

    Never raises to callers. Turn records/labels ride a small dedicated queue
    (bounded-timeout put, then drop-with-log — the only case a turn record may
    drop in v1); spans ride a droppable queue bounded by FW_OBS_QUEUE_MAX
    (drop-and-count).
    """

    def __init__(self, db_path: str) -> None:
        self.store = ObservabilityStore(db_path)
        try:
            self._db_ino = os.stat(db_path).st_ino
        except OSError:
            self._db_ino = None
        self._redactor = Redactor()
        # Resolved once, here, so an unrecognized FW_OBS_CAPTURE_PROFILE fails
        # when the sink is built rather than on every turn — and so a deployment
        # that asked for `evidence` cannot end up running verbatim.
        self._capture_policy = resolve_capture_policy()
        self._record_queue: queue.Queue = queue.Queue(maxsize=_RECORD_QUEUE_MAX)
        self._span_queue: queue.Queue = queue.Queue(
            maxsize=_env_int("FW_OBS_QUEUE_MAX", _DEFAULT_QUEUE_MAX)
        )
        self._closed = False
        self._stop = threading.Event()
        # Who this writer is, stamped into every health row it persists
        # (fix-dnb). The row outlives the writer, so without a name on it a
        # reader cannot tell one writer's whole run from two writers' halves —
        # and the handover between two halves is where records go missing
        # without being counted. `open` is also what tells a would-be archiver
        # in another process that this store is still being written to
        # (`_refuse_if_an_unreachable_writer_holds`).
        self._incarnation: dict[str, Any] = {
            "id": uuid.uuid4().hex,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "started_at": _utcnow_iso(),
            "open": True,
        }
        # Quiesce handshake (fix-7de): a snapshot needs the writer to stop
        # touching the file, not to die. The lock serializes archivers, the
        # request/parked pair is the handshake with the writer thread. Both are
        # events rather than a Condition because the writer must be able to
        # notice a request while it is blocked on its own queue poll, and
        # because `close()` has to be able to release a parked writer by setting
        # `_stop` alone.
        self._quiesce_lock = threading.Lock()
        self._quiesce_request = threading.Event()
        self._quiesce_parked = threading.Event()
        self._health = {
            "spans_dropped": 0,
            "records_dropped": 0,
            "write_errors": 0,
            "busy_retries": 0,
            "refused_terminal_writes": 0,
            "sync_writes": 0,
            "sync_fallbacks": 0,
            "sync_write_ms_max": 0,
            "pending_retry_depth": 0,
            "sync_breaker_open": False,
            "last_error": None,
            # Which turns lost evidence, per §12.4. Bounded; see _count.
            "spans_dropped_turn_keys": [],
            "records_dropped_turn_keys": [],
            "dropped_turn_keys_elided": 0,
            WRITER_INCARNATION_FIELD: self._incarnation,
        }
        self._health_dirty = False
        self._health_lock = threading.Lock()
        # Sync-first write state (§2.4). The breaker deadline is a monotonic
        # timestamp; the ring holds terminal rows the sync path could not land.
        self._sync_lock = threading.Lock()
        self._sync_breaker_until = 0.0
        self._pending: "dict[str, tuple]" = {}
        # The zero baseline, published BEFORE the writer thread can count
        # anything (fix-485). Until this existed the first writer-health row
        # appeared only after the first counted event, so an external driver
        # that opened `evidence_run` on a fresh store — the ordinary shape when
        # a harness starts a server and then opens the gate — read
        # `health_before=None`, got `incomparable=True`, and lost the run to a
        # terminal `invalid` verdict it had no way to avoid.
        #
        # This is a measurement, not an assumption: these counters really are
        # zero at this instant, and the writer that publishes them is the writer
        # that will do the counting. It is also what keeps the honesty guard in
        # `evidence_run` armed — that guard fires only when there is a `before`
        # stamp to compare the `after` stamp against, so a run against a silent
        # writer is still reported as unmeasured rather than as clean.
        self._publish_baseline_health()
        self._writer = threading.Thread(
            target=self._writer_loop, name="fw-obs-writer", daemon=True
        )
        self._writer.start()
        # Opportunistic bounded prune at sink startup [R12].
        try:
            self.store.prune()
        except Exception as exc:
            logger.warning(f"Observability startup prune failed: {exc!r}")

    # -- TraceSink protocol ---------------------------------------------

    def emit_span(self, span: tracing.Span) -> None:
        if self._closed:
            return
        try:
            snapshot = tracing.Span(
                span_id=span.span_id,
                trace_id=span.trace_id,
                name=span.name,
                kind=span.kind,
                parent_span_id=span.parent_span_id,
                channel_id=span.channel_id,
                command_name=span.command_name,
                context=span.context,
                start_ns=span.start_ns,
                end_ns=span.end_ns,
                status=span.status,
                attributes=dict(span.attributes),
                experiment_id=span.experiment_id,
                task_id=span.task_id,
                attempt=span.attempt,
                claim_epoch=span.claim_epoch,
                server_incarnation=span.server_incarnation,
            )
            self._span_queue.put_nowait(("span", snapshot))
        except queue.Full:
            # A span's trace_id IS the logical turn key (tracing.py), so the
            # affected turn is known here without extra plumbing.
            self._count("spans_dropped", turn_key=span.trace_id)
        except Exception as exc:
            self._count("write_errors", error=repr(exc))

    def emit_turn_record(self, record: Any) -> bool:
        """Write the turn record, synchronously by default. Returns "stored".

        Sync-first (§2.4 as amended by rulings I6/C8): EVERY turn-record
        emission — awaiting_user and terminal alike — takes the same path, so
        one logical turn can never be split across the sync and queued paths
        and arrive out of order. The queue is only the degraded fallback.

        The return value is the ack ruling I1 requires. The observability DB is
        the conversation record now, so a caller that drops turns out of its
        in-memory history has to know whether they were actually persisted:
        False means "queued, not yet durable" and the caller must defer its
        trim. Never raises; a caller that cannot use the ack can ignore it.
        """
        if self._closed:
            return False
        try:
            turn_row, artifact_rows = serialize_turn_result(
                record,
                policy=self._capture_policy,
                classify=_capture_classify_for_turn(record),
            )
        except Exception as exc:
            self._count("write_errors", error=f"serialize: {exc!r}")
            return False

        if self._sync_available() and self._sync_write(turn_row, artifact_rows):
            self._forget_pending(turn_row["turn_key"])
            return True

        self._count("sync_fallbacks")
        self._queue_turn_row(turn_row, artifact_rows)
        return False

    def _sync_available(self) -> bool:
        with self._sync_lock:
            return time.monotonic() >= self._sync_breaker_until

    def _sync_write(
        self, turn_row: dict[str, Any], artifact_rows: list[dict[str, Any]]
    ) -> bool:
        """One short BEGIN IMMEDIATE on the caller thread. Never raises.

        Its own connection with a SHORT busy timeout (ruling C9): the default
        30 s would put a wedged DB in front of a user's turn for half a minute.
        On failure the breaker opens so a broken disk degrades to Phase-A
        queued behaviour instead of taxing every subsequent turn.
        """
        started = time.monotonic()
        conn = None
        try:
            conn = self.store._connect(
                timeout=float(
                    _env_int("FW_OBS_SYNC_WRITE_TIMEOUT_S", _DEFAULT_SYNC_WRITE_TIMEOUT_S)
                )
            )
            conn.execute("BEGIN IMMEDIATE")
            accepted = self.store.upsert_turn_row(
                conn, turn_row, artifact_rows, self._redactor
            )
            conn.commit()
        except Exception as exc:
            if conn is not None:
                self._rollback(conn)
            self._trip_sync_breaker(exc)
            return False
        finally:
            if conn is not None:
                with contextlib.suppress(Exception):
                    conn.close()
        if not accepted:
            self._count("refused_terminal_writes")
        elapsed_ms = int((time.monotonic() - started) * 1000)
        with self._health_lock:
            self._health["sync_writes"] = int(self._health["sync_writes"]) + 1
            if elapsed_ms > int(self._health["sync_write_ms_max"] or 0):
                self._health["sync_write_ms_max"] = elapsed_ms
            self._health_dirty = True
        # A refusal means a terminal row is already there: the turn IS durable,
        # which is what the ack promises. Only a failed write is not.
        return True

    def _trip_sync_breaker(self, exc: Exception) -> None:
        cooldown = _env_int(
            "FW_OBS_SYNC_BREAKER_COOLDOWN_S", _DEFAULT_SYNC_BREAKER_COOLDOWN_S
        )
        with self._sync_lock:
            self._sync_breaker_until = time.monotonic() + cooldown
        with self._health_lock:
            self._health["sync_breaker_open"] = True
            self._health_dirty = True
        self._count("write_errors", error=f"sync write: {exc!r}")

    def _queue_turn_row(
        self, turn_row: dict[str, Any], artifact_rows: list[dict[str, Any]]
    ) -> None:
        """Degraded path: reserve the ordinal, enqueue, and remember terminals.

        The ordinal is reserved synchronously in its own tiny transaction
        (ruling I6) so a record that rides the queue still sorts where it
        happened — otherwise a turn written while the DB was briefly wedged
        would land after turns that came later.
        """
        if (
            turn_row.get("conversation_id") is not None
            and turn_row.get("ordinal") is None
        ):
            turn_row["ordinal"] = self.store.reserve_turn_ordinal(
                turn_row["channel_id"],
                turn_row["conversation_id"],
                experiment_id=turn_row.get("experiment_id"),
                task_id=turn_row.get("task_id"),
                attempt=turn_row.get("attempt"),
            )
        if turn_row["status"] in TERMINAL_TURN_STATUSES:
            self._remember_pending(turn_row, artifact_rows)
        try:
            self._record_queue.put(
                ("turn", turn_row, artifact_rows, 0), timeout=_RECORD_PUT_TIMEOUT_S
            )
        except queue.Full:
            self._count("records_dropped", turn_key=turn_row.get("turn_key"))
            logger.warning(
                f"Observability turn-record queue full; DROPPED record for "
                f"{turn_row.get('turn_key')} [R13]"
            )
        except Exception as exc:
            self._count("write_errors", error=repr(exc))

    def _remember_pending(
        self, turn_row: dict[str, Any], artifact_rows: list[dict[str, Any]]
    ) -> None:
        """Hold a terminal row for retry until a write of it is confirmed."""
        with self._sync_lock:
            self._pending[turn_row["turn_key"]] = (turn_row, artifact_rows)
            while len(self._pending) > _PENDING_RETRY_MAX:
                # Oldest first: dict preserves insertion order, and the oldest
                # entry is the one whose turn has been unrecorded longest.
                oldest = next(iter(self._pending))
                del self._pending[oldest]
                self._count("records_dropped", turn_key=oldest)
                logger.warning(
                    f"Observability pending-retry ring full; giving up on "
                    f"turn record {oldest} [R13]"
                )
            depth = len(self._pending)
        with self._health_lock:
            self._health["pending_retry_depth"] = depth
            self._health_dirty = True

    def _forget_pending(self, turn_key: str) -> None:
        with self._sync_lock:
            if self._pending.pop(turn_key, None) is None:
                return
            depth = len(self._pending)
        with self._health_lock:
            self._health["pending_retry_depth"] = depth
            self._health_dirty = True

    def pending_retry_depth(self) -> int:
        """Terminal records still awaiting a confirmed write (tests, health)."""
        with self._sync_lock:
            return len(self._pending)

    def health_snapshot(self) -> dict[str, Any]:
        """A copy of the live counters, without waiting for the writer to persist.

        The `diagnostics` row lags by up to a heartbeat, so an evidence run that
        read only the row could snapshot a drop that had already happened as
        though it had not. Lists are copied so a caller holding two snapshots
        cannot find they are the same object.
        """
        with self._health_lock:
            snapshot = dict(self._health)
        for field in _DROP_TURN_KEY_FIELDS.values():
            snapshot[field] = list(snapshot.get(field) or ())
        return snapshot

    def _publish_baseline_health(self) -> None:
        """Publish this writer's opening counters.

        Never lowers what is already there: an existing row belongs to an
        earlier writer over the same DB and carries the drops it recorded, so
        writing zeros over it would erase evidence rather than establish a
        baseline. An insert-if-absent would achieve that much, but it leaves a
        reopened store's row untouched — including the incarnation stamp, so the
        row would go on naming a writer that had already died. The monotone merge
        does the same job without that side effect: the counters keep their
        floor, and the stamp names the writer that is actually running, which is
        what lets `health_delta` see a restart at all.

        Best-effort like every other write on this class: a store that
        cannot take the row degrades to reporting `incomparable` for an
        evidence run over it, which is the honest answer.
        """
        with self._health_lock:
            snapshot = dict(self._health)
        try:
            conn = self.store._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                self.store.merge_writer_health_row(conn, snapshot)
                conn.commit()
            finally:
                conn.close()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Could not publish the writer-health baseline: {exc!r}")

    def persist_health(self) -> None:
        """Force the counters into the `diagnostics` row.

        Called at the end of an evidence run so the archived DB carries the same
        verdict the in-process delta reported; without it the archive can say
        "healthy" about a run that dropped records after the last heartbeat.
        """
        try:
            conn = self.store._connect()
            try:
                self._maybe_write_health(conn, force=True)
            finally:
                conn.close()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Could not persist writer health: {exc!r}")

    def record_conversation_label(
        self,
        channel_id: str,
        conversation_id: int,
        topic: Optional[str],
        summary: Optional[str],
    ) -> None:
        if self._closed:
            return
        try:
            self._record_queue.put(
                ("label", channel_id, conversation_id, topic, summary, 0),
                timeout=_RECORD_PUT_TIMEOUT_S,
            )
        except queue.Full:
            self._count("records_dropped")
        except Exception as exc:
            self._count("write_errors", error=repr(exc))

    # -- lifecycle -------------------------------------------------------

    def flush(self, timeout: float = 10.0) -> bool:
        """Block until everything enqueued so far is written (tests, close)."""
        done = threading.Event()
        try:
            self._record_queue.put(("flush", done), timeout=timeout)
        except queue.Full:
            return False
        return done.wait(timeout)

    def close(self, timeout: float = 10.0) -> None:
        """Stop signal + bounded join + final drain and commit. Idempotent.

        Emissions racing with close are dropped (the sink is closed); the
        writer drains everything already enqueued before exiting, so the last
        turn of a session is never lost.
        """
        if self._closed:
            return
        self._closed = True
        # Stamp the handover BEFORE the writer is told to stop, so the final
        # health write the writer thread makes on its way out already says this
        # writer is gone (fix-dnb). A NEW dict rather than a mutation: snapshots
        # taken earlier hold a reference to the old one and must keep reading
        # `open: True`, which is what they were true about.
        self._incarnation = {**self._incarnation, "open": False}
        with self._health_lock:
            self._health[WRITER_INCARNATION_FIELD] = self._incarnation
            self._health_dirty = True
        self._stop.set()
        # A parked writer releases on `_stop` alone, so an archive in flight
        # cannot wedge a close.
        self._writer.join(timeout)
        if self._writer.is_alive():
            logger.warning("Observability writer did not stop within timeout")
        # Belt and braces: the writer normally persists on its way out, but a
        # writer that died, hung, or timed out above leaves the row claiming an
        # open writer forever — and an open marker is what stops the next
        # archive of this store (`_refuse_if_an_unreachable_writer_holds`).
        #
        # Only into the file this writer actually opened. `get_observability_sink`
        # closes a sink whose DB was deleted or replaced underneath it, and
        # writing there would either resurrect a deleted file as an empty
        # schema-less database or stamp a successor's file with a dead writer's
        # row.
        if self._owns_its_db_file():
            self.persist_health()

    def _owns_its_db_file(self) -> bool:
        """Whether the file at this sink's path is still the file it opened."""
        try:
            return os.stat(self.store.db_path).st_ino == self._db_ino
        except OSError:
            return False

    @contextlib.contextmanager
    def quiesced(self, timeout: float = 10.0):
        """Hold this writer still, without closing it, for a snapshot.

        Everything a snapshot has to survive, in the order it has to happen:

        1. **Flush.** Whatever is already enqueued is written, so the archive
           carries the run it claims to and the queue has nothing left to land
           mid-snapshot.
        2. **Park.** The writer thread stops at the top of its loop and waits.
           This is what stops the heartbeat — the periodic health write that
           lands between two digests and produces the load-sensitive
           `SourceChangedDuringArchive` — along with the pending-retry ring and
           the breaker probe, which are on the same idle tick.
        3. **Settle.** The counters are persisted once, deliberately, and the
           WAL is checkpointed back into the main file, so what the digest
           covers is the whole store rather than a main file plus a sidecar that
           is still moving.

        Then the caller takes its snapshot and the writer is released. The sink
        is never closed and no second writer is created: the one-writer
        contract is about how many threads may write, not about whether the one
        that may is currently mid-stride.

        Raises `WriterStillOpen` if the writer cannot be brought to a stop —
        refusing is right when the alternative is a snapshot of a moving file.
        A sink whose thread is already gone quiesces trivially.
        """
        if not self._quiesce_lock.acquire(timeout=timeout):
            raise WriterStillOpen(
                f"another archive is already holding {self.store.db_path!r} "
                f"still; refusing to take a second snapshot of it"
            )
        try:
            if self._closed or not self._writer.is_alive():
                # Nothing to hold still. A closed sink's writer has already
                # drained, persisted and let go of its connection.
                yield
                return
            if not self.flush(timeout):
                raise WriterStillOpen(
                    f"the writer for {self.store.db_path!r} did not flush within "
                    f"{timeout}s, so a snapshot of it cannot claim to hold the "
                    f"records this run enqueued"
                )
            self._quiesce_request.set()
            try:
                deadline = time.monotonic() + timeout
                while not self._quiesce_parked.is_set():
                    if self._closed or not self._writer.is_alive():
                        break
                    if time.monotonic() >= deadline:
                        raise WriterStillOpen(
                            f"the writer for {self.store.db_path!r} did not park "
                            f"within {timeout}s; refusing to snapshot a store "
                            f"that is still being written"
                        )
                    time.sleep(_QUIESCE_POLL_S)
                self._settle_for_snapshot()
                yield
            finally:
                self._quiesce_request.clear()
        finally:
            self._quiesce_lock.release()

    def _park_if_quiescing(self) -> None:
        """Writer-thread side of `quiesced`. Called at the top of each loop."""
        if not self._quiesce_request.is_set():
            return
        self._quiesce_parked.set()
        try:
            while self._quiesce_request.is_set() and not self._stop.is_set():
                time.sleep(_QUIESCE_POLL_S)
        finally:
            self._quiesce_parked.clear()

    def _settle_for_snapshot(self) -> None:
        """Persist the counters and fold the WAL back in, with the writer parked.

        Both writes happen HERE rather than being left to the heartbeat, which is
        the point: they are the two writes that would otherwise have landed
        between the archive's `before` and `after` digests. Doing them under the
        quiesce makes them part of what is being archived instead of a change to
        it. Best-effort like every other write on this class: a checkpoint that
        cannot take the lock leaves a larger `-wal`, not a wrong archive, because
        the digest comparison is still the thing that decides.
        """
        try:
            conn = self.store._connect()
            try:
                self._maybe_write_health(conn, force=True)
                with contextlib.suppress(Exception):
                    conn.commit()
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            finally:
                conn.close()
        except Exception as exc:
            logger.warning(f"Could not settle the store for a snapshot: {exc!r}")

    # -- internals -------------------------------------------------------

    def _count(
        self, key: str, error: Optional[str] = None, turn_key: Optional[str] = None
    ) -> None:
        """Bump one health counter, and remember which turn a drop belonged to.

        `turn_key` exists because §12.4 requires dropped spans to be "reported
        with the affected turn keys": a count alone tells an evaluation that it
        lost evidence but not which turns are now incomplete, which is the only
        fact that lets a run be salvaged rather than discarded. Bounded, because a
        pathological run must not turn a drop counter into a memory leak — the
        elided count preserves the honesty of the set when the cap is hit.
        """
        with self._health_lock:
            self._health[key] = int(self._health.get(key) or 0) + 1
            if error is not None:
                self._health["last_error"] = error[:500]
            if turn_key and key in _DROP_TURN_KEY_FIELDS:
                affected = self._health[_DROP_TURN_KEY_FIELDS[key]]
                if turn_key not in affected:
                    if len(affected) < _DROP_TURN_KEY_MAX:
                        affected.append(turn_key)
                    else:
                        self._health["dropped_turn_keys_elided"] = (
                            int(self._health.get("dropped_turn_keys_elided") or 0) + 1
                        )
            self._health_dirty = True

    def _writer_loop(self) -> None:
        conn: Optional[sqlite3.Connection] = None
        try:
            conn = self.store._connect()
            while not self._stop.is_set():
                # The one place this thread stops touching the DB on request
                # (fix-7de). At the top of the loop, so a parked writer is
                # between batches and between heartbeats — holding no
                # transaction and owing no write.
                self._park_if_quiescing()
                item = self._next_item()
                if item is None:
                    self._heartbeat(conn)
                    continue
                self._apply_batch(conn, [item] + self._drain_pending())
            # Final drain: everything enqueued before close() is written.
            while items := self._drain_pending():
                self._apply_batch(conn, items)
            # Then the retry ring, which holds terminal rows the queue may have
            # dropped — the last thing standing between a wedged-then-recovered
            # DB and a permanently missing turn.
            self._retry_pending(conn)
        except Exception as exc:  # writer must never crash the process
            self._count("write_errors", error=repr(exc))
            logger.warning(f"Observability writer loop error: {exc!r}")
        finally:
            if conn is not None:
                try:
                    self._maybe_write_health(conn, force=True)
                    conn.commit()
                except Exception:
                    pass
                conn.close()

    def _heartbeat(self, conn: sqlite3.Connection) -> None:
        """Idle-tick work: flush health, retry the pending ring, re-arm the breaker.

        All three are deliberately off the turn path — this runs on the writer
        thread between drains, so a wedged DB costs a background retry rather
        than a user's latency.
        """
        self._retry_pending(conn)
        self._maybe_rearm_sync_breaker()
        self._maybe_write_health(conn)

    def _retry_pending(self, conn: sqlite3.Connection) -> None:
        """Re-write terminal rows the sync path could not land (ruling I1).

        The upsert is idempotent on turn_key, so a row the queue already
        delivered is claimed as an idempotent retry rather than refused.
        """
        with self._sync_lock:
            if not self._pending:
                return
            items = list(self._pending.items())
        landed = []
        for turn_key, (turn_row, artifact_rows) in items:
            try:
                conn.execute("BEGIN IMMEDIATE")
                self.store.upsert_turn_row(
                    conn, turn_row, artifact_rows, self._redactor
                )
                conn.commit()
            except Exception as exc:
                self._rollback(conn)
                self._count("write_errors", error=f"pending retry: {exc!r}")
                break  # still unhealthy; leave the rest for the next tick
            landed.append(turn_key)
        for turn_key in landed:
            self._forget_pending(turn_key)

    def _maybe_rearm_sync_breaker(self) -> None:
        """Close the breaker only after a write probe succeeds (ruling C9).

        The cooldown elapsing proves nothing about the DB, and re-arming blind
        would put the next user turn back in front of the same wedged file.
        The probe is a diagnostics upsert on the sync path's own connection —
        the same write shape, at the same busy timeout, off the turn path.
        """
        with self._sync_lock:
            if self._sync_breaker_until == 0.0:
                return
            if time.monotonic() < self._sync_breaker_until:
                return
        conn = None
        try:
            conn = self.store._connect(
                timeout=float(
                    _env_int("FW_OBS_SYNC_WRITE_TIMEOUT_S", _DEFAULT_SYNC_WRITE_TIMEOUT_S)
                )
            )
            conn.execute("BEGIN IMMEDIATE")
            self.store.set_diagnostic(
                conn, "sync_breaker_probe", {"at": _utcnow_iso()}
            )
            conn.commit()
        except Exception:
            # Still wedged: hold the breaker open for another cooldown rather
            # than probing on every idle tick.
            with self._sync_lock:
                self._sync_breaker_until = time.monotonic() + _env_int(
                    "FW_OBS_SYNC_BREAKER_COOLDOWN_S", _DEFAULT_SYNC_BREAKER_COOLDOWN_S
                )
            return
        finally:
            if conn is not None:
                with contextlib.suppress(Exception):
                    conn.close()
        with self._sync_lock:
            self._sync_breaker_until = 0.0
        with self._health_lock:
            self._health["sync_breaker_open"] = False
            self._health_dirty = True
        logger.info("Observability sync-write breaker re-armed after a successful probe")

    def _next_item(self) -> Any:
        """One item, records first; None on idle timeout (health heartbeat)."""
        try:
            return self._record_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            return self._span_queue.get(timeout=0.25)
        except queue.Empty:
            return None

    def _drain_pending(self, limit: int = 512) -> list:
        items = []
        for _ in range(limit):
            try:
                items.append(self._record_queue.get_nowait())
                continue
            except queue.Empty:
                pass
            try:
                items.append(self._span_queue.get_nowait())
            except queue.Empty:
                break
        return items

    def _apply_batch(self, conn: sqlite3.Connection, items: list) -> None:
        flush_events: list[threading.Event] = []
        spans: list[tracing.Span] = []
        try:
            conn.execute("BEGIN IMMEDIATE")
            for item in items:
                kind = item[0]
                if kind == "span":
                    spans.append(item[1])
                elif kind == "turn":
                    self._apply_turn(conn, item)
                elif kind == "label":
                    self._apply_label(conn, item)
                elif kind == "flush":
                    flush_events.append(item[1])
            if spans:
                self.store.upsert_span_rows(conn, spans, self._redactor)
            self._maybe_write_health(conn, in_txn=True)
            conn.commit()
        except sqlite3.OperationalError as exc:
            # SQLITE_BUSY under multi-process contention [R8].
            self._rollback(conn)
            self._count("busy_retries", error=repr(exc))
            self._requeue_records(items)
        except Exception as exc:
            self._rollback(conn)
            self._count("write_errors", error=repr(exc))
        finally:
            for event in flush_events:
                event.set()

    def _apply_turn(self, conn: sqlite3.Connection, item: tuple) -> None:
        _, turn_row, artifact_rows, _retries = item
        accepted = self.store.upsert_turn_row(
            conn, turn_row, artifact_rows, self._redactor
        )
        if not accepted:
            self._count("refused_terminal_writes")
        # The row landed, so the retry ring no longer owes anyone this turn.
        # Cleared inside the batch txn rather than after the commit: a commit
        # failure rolls the batch back and requeues it, and the ring entry is
        # re-added by that path if it is still needed.
        self._forget_pending(turn_row["turn_key"])

    def _apply_label(self, conn: sqlite3.Connection, item: tuple) -> None:
        _, channel_id, conversation_id, topic, summary, _retries = item
        # Labels are persisted text too — same [R20] sink-boundary scrub as
        # turn rows and span attributes.
        topic = self._redactor.redact(topic) if topic else topic
        summary = self._redactor.redact(summary) if summary else summary
        # Single enforcement point: uniquification inside the writer's own
        # transaction (ruling I9).
        self.store.apply_label_txn(conn, channel_id, conversation_id, topic, summary)

    def _requeue_records(self, items: list) -> None:
        """Bounded retry for turn records/labels on SQLITE_BUSY; spans drop."""
        for item in items:
            kind = item[0]
            if kind == "span":
                self._count("spans_dropped", turn_key=getattr(item[1], "trace_id", None))
                continue
            if kind == "flush":
                item[1].set()
                continue
            # A "turn" item carries its row at [1]; a "label" item has no turn.
            turn_key = item[1].get("turn_key") if kind == "turn" else None
            retries = item[-1]
            if retries >= _RECORD_BUSY_MAX_RETRIES:
                self._count("records_dropped", turn_key=turn_key)
                continue
            retried = item[:-1] + (retries + 1,)
            try:
                self._record_queue.put_nowait(retried)
            except queue.Full:
                self._count("records_dropped", turn_key=turn_key)

    @staticmethod
    def _rollback(conn: sqlite3.Connection) -> None:
        try:
            conn.rollback()
        except Exception:
            pass

    def _maybe_write_health(
        self, conn: sqlite3.Connection, force: bool = False, in_txn: bool = False
    ) -> None:
        with self._health_lock:
            if not (self._health_dirty or force):
                return
            snapshot = dict(self._health)
            self._health_dirty = False
        try:
            if not in_txn:
                conn.execute("BEGIN IMMEDIATE")
            # Merged, never replaced (fix-dnb): the heartbeat is the write that
            # used to reset a predecessor's counters to this writer's own.
            self.store.merge_writer_health_row(conn, snapshot)
            if not in_txn:
                conn.commit()
        except Exception:
            self._rollback(conn)
            with self._health_lock:
                self._health_dirty = True


# ----------------------------------------------------------------------
# Factory [R4]
# ----------------------------------------------------------------------

_sinks_lock = threading.Lock()
_sinks: dict[str, SQLiteTraceSink] = {}


def get_observability_sink(workflow_path: str) -> Optional[SQLiteTraceSink]:
    """The process-wide sink for a workflow's observability DB.

    One sink (one writer thread) per DB path; closed atexit. There is no switch:
    every caller gets a sink. Never raises — a store that cannot open degrades
    to ``None`` plus a warning, and that is the only way to get ``None``.
    """
    try:
        db_path = state_paths.observability_db(workflow_path)
        with _sinks_lock:
            sink = _sinks.get(db_path)
            if sink is not None and not sink._closed and _sink_is_stale(sink, db_path):
                # The DB file was deleted/replaced under the cached sink (its
                # writer would silently write into the old inode). Recycle.
                try:
                    sink.close(timeout=2.0)
                except Exception:
                    pass
                sink = None
            if sink is None or sink._closed:
                sink = SQLiteTraceSink(db_path)
                _sinks[db_path] = sink
            return sink
    except Exception as exc:
        logger.warning(f"Observability sink unavailable for {workflow_path}: {exc!r}")
        return None


def existing_observability_sink(
    workflow_path: str,
) -> Optional[SQLiteTraceSink]:
    """Return this process's live sink without constructing one.

    Never raises: `evidence_run` peeks through this and must not
    lose a run to a path-resolution error. None for "no live sink".
    """
    try:
        db_path = state_paths.observability_db(workflow_path)
        with _sinks_lock:
            sink = _sinks.get(db_path)
            if sink is None or sink._closed or _sink_is_stale(sink, db_path):
                return None
            return sink
    except Exception:
        return None


def sink_for_db_path(db_path: str) -> Optional[SQLiteTraceSink]:
    """Return this process's live writer for an exact DB path, if one exists."""
    resolved = os.path.realpath(db_path)
    with _sinks_lock:
        sink = next(
            (
                candidate
                for path, candidate in _sinks.items()
                if os.path.realpath(path) == resolved
            ),
            None,
        )
        if sink is None or sink._closed or _sink_is_stale(sink, db_path):
            return None
        return sink


def _sink_is_stale(sink: SQLiteTraceSink, db_path: str) -> bool:
    try:
        return os.stat(db_path).st_ino != sink._db_ino
    except OSError:
        return True  # file gone


def close_all_sinks() -> None:
    with _sinks_lock:
        sinks = list(_sinks.values())
        _sinks.clear()
    for sink in sinks:
        try:
            sink.close()
        except Exception:
            pass


def close_sinks_under(directory: str) -> int:
    """Close this process's sinks whose DB lives under *directory*; return how many.

    For a caller about to remove a state root it created -- a test's temporary
    root, say -- so no writer thread is left writing into deleted files while
    sinks for other roots stay open.
    """
    root = os.path.realpath(directory) + os.sep
    with _sinks_lock:
        doomed = [
            path for path in _sinks
            if os.path.realpath(path).startswith(root)
        ]
        sinks = [_sinks.pop(path) for path in doomed]
    for sink in sinks:
        try:
            sink.close()
        except Exception:
            pass
    return len(sinks)


atexit.register(close_all_sinks)

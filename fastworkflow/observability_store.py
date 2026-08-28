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
  checkpointing never starves [R12]).
- ``SQLiteTraceSink`` — the TraceSink implementation: two queues ([R13]: a
  small turn-record/label queue with a bounded-timeout put — the only case a
  turn record may drop in v1 — and a droppable span queue bounded by
  ``FW_OBS_QUEUE_MAX``), drained by one daemon writer thread with batched
  transactions; ``close()`` (sentinel + bounded join) is wired to atexit and
  entry-point exit paths [R7]. Writer errors/drops land in the
  ``diagnostics`` table and are surfaced by the chatbot UI [R13].
- ``get_observability_sink()`` — process-wide factory honoring
  ``FW_OBSERVABILITY`` ([R4]: fastWorkflow's own entry points default it ON;
  library embedders opt in), one sink (= one writer thread) per DB path.

Durability class (Phase A, [R14]): everything is best-effort; a write failure
never fails a turn. Multi-process writers are supported on local filesystems
only (WAL constraint — the state root must not be NFS).
"""

from __future__ import annotations

import atexit
import contextlib
import hashlib
import json
import os
import queue
import re
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import fastworkflow
from fastworkflow import state_paths, tracing
from fastworkflow.utils.logging import logger

SCHEMA_VERSION = 1

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


class IncompatibleObservabilityDB(RuntimeError):
    """The DB was written by a newer fastWorkflow; readers refuse it [R11]."""


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


def serialize_turn_result(turn_result: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Project a TurnResult into (turn_row, artifact_rows) at emission time.

    - ``record_json`` holds the full internal TurnResult (post-envelope,
      pre-redaction — the sink redacts the serialized text) [R10].
    - Any artifact value over ``FW_OBS_INLINE_ARTIFACT_BYTES`` is replaced in
      place by a ref envelope; the artifacts table is the only value holder.
    - ``traceback`` artifacts persist only under FW_OBS_CAPTURE_TRACEBACKS=1
      [R20].

    Runs in the caller thread so the row snapshots the turn as emitted (the
    accumulator's CommandOutput objects mutate on resume).
    """
    turn_output = turn_result.turn_output
    inline_limit = _env_int("FW_OBS_INLINE_ARTIFACT_BYTES", _DEFAULT_INLINE_ARTIFACT_BYTES)
    capture_tracebacks = _env("FW_OBS_CAPTURE_TRACEBACKS", "0") == "1"

    try:
        record = turn_result.model_dump(mode="python")
    except Exception:
        record = {"turn_output": {"turn_key": turn_output.turn_key}}
    record = _sanitize_json_value(record)
    # computed_field `success` is included by model_dump; make sure it is
    # present even on the fallback path.
    record.setdefault("turn_output", {}).setdefault("success", turn_output.success)

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
        "record_version": 1,
        "record_json": json.dumps(record, ensure_ascii=False),
    }
    return turn_row, artifact_rows


# ----------------------------------------------------------------------
# The store
# ----------------------------------------------------------------------

_SCHEMA_STATEMENTS = [
    """CREATE TABLE IF NOT EXISTS conversations (
        channel_id TEXT NOT NULL, conversation_id INTEGER NOT NULL,
        topic TEXT, summary TEXT, status TEXT, next_ordinal INTEGER,
        started_at TEXT, last_turn_at TEXT, updated_at TEXT,
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
        record_json TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS feedback (
        turn_key TEXT PRIMARY KEY, feedback_json TEXT NOT NULL,
        updated_at TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS spans (
        span_id TEXT PRIMARY KEY, trace_id TEXT NOT NULL,
        parent_span_id TEXT, name TEXT NOT NULL,
        kind TEXT NOT NULL,
        channel_id TEXT,
        command_name TEXT, context TEXT,
        start_ns INTEGER NOT NULL, end_ns INTEGER,
        status TEXT NOT NULL, attributes TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS artifacts (
        artifact_id TEXT PRIMARY KEY, turn_key TEXT NOT NULL,
        channel_id TEXT,
        span_id TEXT, key TEXT NOT NULL, content_type TEXT,
        size_bytes INTEGER, sha256 TEXT,
        inline_value BLOB, error TEXT)""",
    """CREATE TABLE IF NOT EXISTS train_runs (
        run_id TEXT PRIMARY KEY, workflow_fingerprint TEXT, started_at TEXT,
        completed_at TEXT, metrics_json TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS diagnostics (
        key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL)""",
    "CREATE INDEX IF NOT EXISTS idx_spans_trace ON spans(trace_id)",
    "CREATE INDEX IF NOT EXISTS idx_spans_command ON spans(command_name) WHERE command_name IS NOT NULL",
    "CREATE INDEX IF NOT EXISTS idx_turns_conv ON turns(channel_id, conversation_id, ordinal)",
    "CREATE INDEX IF NOT EXISTS idx_turns_status ON turns(status)",
    "CREATE INDEX IF NOT EXISTS idx_artifacts_turn ON artifacts(turn_key)",
]


class ObservabilityStore:
    """Schema owner + synchronous operations on one observability DB.

    Thread/process-safe by construction: every method opens its own
    short-lived WAL connection (timeout=30, ``BEGIN IMMEDIATE`` for writes).
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._ensure_schema()

    # -- connections ----------------------------------------------------

    def _connect(self, timeout: float = 30.0) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=timeout, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        parent = os.path.dirname(self.db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
            try:
                os.chmod(parent, 0o700)  # [R4]
            except OSError:
                pass
        fresh = not os.path.exists(self.db_path)
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        try:
            if fresh:
                # auto_vacuum must be set at creation, before any table [R12].
                conn.execute("PRAGMA auto_vacuum=INCREMENTAL")
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")

            found = conn.execute("PRAGMA user_version").fetchone()[0]
            if found > SCHEMA_VERSION:
                raise IncompatibleObservabilityDB(
                    f"{self.db_path} has schema v{found}; this build reads up to "
                    f"v{SCHEMA_VERSION}. Refusing to open a newer DB [R11]."
                )
            for statement in _SCHEMA_STATEMENTS:
                conn.execute(statement)
            # Pre-release column migration (schema v1 was never shipped, but
            # dev DBs created by earlier work-in-progress builds exist):
            # CREATE IF NOT EXISTS cannot add columns to an existing table.
            existing_cols = {
                row[1]
                for row in conn.execute("PRAGMA table_info(conversations)").fetchall()
            }
            if "updated_at" not in existing_cols:
                conn.execute("ALTER TABLE conversations ADD COLUMN updated_at TEXT")
            if found < SCHEMA_VERSION:
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
            conn.commit()
        finally:
            conn.close()
        try:
            os.chmod(self.db_path, 0o600)  # [R4]
            wal = f"{self.db_path}-wal"
            if os.path.exists(wal):
                os.chmod(wal, 0o600)
        except OSError:
            pass

    # -- identity [R1] ---------------------------------------------------

    def mint_conversation_id(self, channel_id: str, legacy_floor: int = 0) -> int:
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
                    next_ordinal, started_at, last_turn_at, updated_at)
                   VALUES (?, ?, NULL, NULL, 'open', 1, ?, NULL, ?)""",
                (channel_id, new_id, now, now),
            )
            conn.commit()
        return new_id

    def record_conversation_label(
        self,
        channel_id: str,
        conversation_id: int,
        topic: Optional[str],
        summary: Optional[str],
    ) -> str:
        """Upsert a conversation's topic/summary ([R15]; labels are mutable).

        A None topic or summary preserves the stored value, so the blank-topic
        policy — a failed generation never clobbers a good title — carries
        over from the legacy store. Topic uniquification runs inside the same
        transaction as the write (review ruling I9: no TOCTOU across the async
        label path; Python-side casefold, never SQLite's ASCII-only lower()).

        Returns the topic actually STORED — collision-suffixed where one was
        written, or the preserved existing title on a blank generation. A
        caller that reports or logs the label must use this rather than its own
        candidate, which is the contract the legacy store's
        ``update_conversation_topic_summary`` established (ruling I9).
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
        for span in spans:
            attributes = redactor.redact(
                json.dumps(_sanitize_json_value(span.attributes), ensure_ascii=False)
            )
            conn.execute(
                """INSERT INTO spans
                   (span_id, trace_id, parent_span_id, name, kind, channel_id,
                    command_name, context, start_ns, end_ns, status, attributes)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(span_id) DO UPDATE SET
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
                    span.name,
                    span.kind,
                    span.channel_id,
                    span.command_name,
                    span.context,
                    span.start_ns,
                    span.end_ns,
                    span.status,
                    attributes,
                ),
            )

    def upsert_turn_row(
        self,
        conn: sqlite3.Connection,
        turn_row: dict[str, Any],
        artifact_rows: list[dict[str, Any]],
        redactor: Redactor,
    ) -> bool:
        """Apply the [R2] lifecycle: INSERT at first emission; one guarded
        status transition to a terminal status; write-once for rows already
        terminal (identical-content retries claim idempotent success).

        Returns False when a conflicting write against a terminal row was
        refused (counted by the caller).
        """
        turn_row = dict(turn_row)
        # failure_reason is included because it can embed exception/provider
        # text (e.g. a LiteLLM AuthenticationError body) — the [R20] scenario.
        for text_col in (
            "user_message",
            "refined_user_message",
            "answer",
            "failure_reason",
            "conversation_summary",
            "conversation_traces",
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
                conn, turn_row["channel_id"], turn_row["conversation_id"]
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
            inline_value = artifact.get("inline_value")
            if isinstance(inline_value, (bytes, bytearray)):
                redacted = redactor.redact(
                    bytes(inline_value).decode("utf-8", errors="replace")
                )
                inline_value = redacted.encode("utf-8")
            conn.execute(
                """INSERT INTO artifacts
                   (artifact_id, turn_key, channel_id, span_id, key, content_type,
                    size_bytes, sha256, inline_value, error)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                ),
            )
        return True

    def _assign_ordinal(
        self, conn: sqlite3.Connection, channel_id: str, conversation_id: int
    ) -> int:
        row = conn.execute(
            "SELECT next_ordinal FROM conversations WHERE channel_id=? AND conversation_id=?",
            (channel_id, conversation_id),
        ).fetchone()
        if row is None:
            # Conversation row not minted here (e.g. restored session) —
            # create it so ordinals stay dense from 1.
            conn.execute(
                """INSERT INTO conversations
                   (channel_id, conversation_id, topic, summary, status,
                    next_ordinal, started_at, last_turn_at)
                   VALUES (?, ?, NULL, NULL, 'open', 2, ?, NULL)""",
                (channel_id, conversation_id, _utcnow_iso()),
            )
            return 1
        ordinal = int(row["next_ordinal"] or 1)
        conn.execute(
            "UPDATE conversations SET next_ordinal=? WHERE channel_id=? AND conversation_id=?",
            (ordinal + 1, channel_id, conversation_id),
        )
        return ordinal

    def reserve_turn_ordinal(self, channel_id: str, conversation_id: int) -> Optional[int]:
        """Reserve a turn ordinal in a tiny standalone transaction.

        Used by the sync-first emit's degraded fallback so ordinals stay
        chronological even when the row itself is queued (ruling I6).
        Returns None when the reservation itself cannot be made.
        """
        try:
            with self._connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                ordinal = self._assign_ordinal(conn, channel_id, conversation_id)
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
        dicts (oldest-first), feedback joined in — the gate-1 [R3] read that
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

    def record_train_run(
        self,
        run_id: str,
        workflow_fingerprint: Optional[str],
        started_at: Optional[str],
        completed_at: Optional[str],
        metrics: dict[str, Any],
    ) -> None:
        """Persist one training run's metrics at publication time (Phase 6)."""
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
                    json.dumps(_sanitize_json_value(metrics), ensure_ascii=False),
                ),
            )
            conn.commit()

    def set_diagnostic(self, conn: sqlite3.Connection, key: str, value: dict[str, Any]) -> None:
        conn.execute(
            """INSERT INTO diagnostics (key, value, updated_at) VALUES (?, ?, ?)
               ON CONFLICT(key) DO UPDATE SET
                 value=excluded.value, updated_at=excluded.updated_at""",
            (key, json.dumps(value, ensure_ascii=False), _utcnow_iso()),
        )

    # -- reads (GET /turns, run_chatbot) ---------------------------------

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
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Turn rows, newest first, without record_json (fetch one turn for that)."""
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
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        query = (
            "SELECT turn_key, channel_id, conversation_id, ordinal, user_message, "
            "entry_workflow_name, entry_context, status, success, failure_reason, "
            "answer, started_at, completed_at, suspended_ms "
            f"FROM turns{where} ORDER BY turn_key DESC LIMIT ? OFFSET ?"
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

    # -- maintenance [R12] and erasure [R21] -----------------------------

    def db_size_bytes(self) -> int:
        """DB file size including the -wal sidecar [R12]."""
        total = 0
        for path in (self.db_path, f"{self.db_path}-wal"):
            try:
                total += os.path.getsize(path)
            except OSError:
                pass
        return total

    def prune(
        self,
        retention_days: Optional[int] = None,
        max_bytes: Optional[int] = None,
        include_conversationless_turns: bool = False,
    ) -> dict[str, int]:
        """Bounded prune of spans/artifacts beyond the retention horizon, plus
        oldest-first eviction while over the size cap. Conversations and turn
        records are exempt (config §5 / [R16]). Runs incremental_vacuum.

        ``include_conversationless_turns`` (operator opt-in, ruling C10) also
        deletes conversation-less turn records (e.g. per-invocation CLI
        channels) older than the horizon, with their feedback — otherwise no
        retention knob ever reaches them.
        """
        if retention_days is None:
            retention_days = _env_int("FW_OBS_RETENTION_DAYS", _DEFAULT_RETENTION_DAYS)
        if max_bytes is None:
            max_bytes = _env_int("FW_OBS_DB_MAX_BYTES", _DEFAULT_DB_MAX_BYTES)

        horizon_ns = int(
            (time.time() - retention_days * 86_400) * 1_000_000_000
        )
        horizon_key = datetime.fromtimestamp(
            max(0.0, time.time() - retention_days * 86_400), tz=timezone.utc
        ).strftime("%Y%m%dT%H%M%S")
        deleted = {"spans": 0, "artifacts": 0}

        with self._connect() as conn:
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
                conn.commit()
                if (
                    spans_cur.rowcount < _PRUNE_BATCH_ROWS
                    and artifacts_cur.rowcount < _PRUNE_BATCH_ROWS
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
                    conn.commit()
                    deleted["conversationless_turns"] += len(keys)
                    if len(keys) < _PRUNE_BATCH_ROWS:
                        break

            # Size-cap eviction, oldest spans first (turn keys sort by time).
            for _ in range(_PRUNE_MAX_BATCHES):
                if self.db_size_bytes() <= max_bytes:
                    break
                conn.execute("BEGIN IMMEDIATE")
                cur = conn.execute(
                    "DELETE FROM spans WHERE span_id IN "
                    "(SELECT span_id FROM spans ORDER BY start_ns LIMIT ?)",
                    (_PRUNE_BATCH_ROWS,),
                )
                conn.commit()
                if cur.rowcount == 0:
                    break
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

            conn.execute("PRAGMA incremental_vacuum")
            conn.commit()
        return deleted

    def forget_channel(self, channel_id: str) -> dict[str, int]:
        """First-class erasure [R21]: delete a channel across all tables, then
        checkpoint-truncate the WAL and reclaim pages."""
        deleted: dict[str, int] = {}
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
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
            deleted["turns"] = conn.execute(
                "DELETE FROM turns WHERE channel_id=?", (channel_id,)
            ).rowcount
            deleted["conversations"] = conn.execute(
                "DELETE FROM conversations WHERE channel_id=?", (channel_id,)
            ).rowcount
            conn.commit()
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            conn.execute("PRAGMA incremental_vacuum")
            conn.commit()
        return deleted

    def clear_conversations(self) -> dict[str, int]:
        """Delete every recorded conversation and its turn-level observability.

        Training runs, writer diagnostics, and monotonic conversation counters
        survive. Keeping counters prevents a clear operation from reusing a
        conversation identity that may still be referenced outside this DB.
        """
        deleted: dict[str, int] = {}
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            for table in ("feedback", "spans", "artifacts", "turns", "conversations"):
                deleted[table] = conn.execute(f"DELETE FROM {table}").rowcount
            conn.commit()
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            conn.execute("PRAGMA incremental_vacuum")
            conn.commit()
        return deleted


class ReadOnlyObservabilityStore(ObservabilityStore):
    """Read-only view of an existing observability DB (the chatbot's debug
    layer). Never creates, migrates, or writes the file — the viewer must be
    able to open a post-mortem snapshot it does not own, and inspecting a DB
    must not mutate it. Construction raises when the file is absent/unopenable
    (``sqlite3.OperationalError``) or written by a newer build
    (``IncompatibleObservabilityDB`` [R11]); callers degrade gracefully.
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
        finally:
            conn.close()

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
    (drop-and-count) [R13].
    """

    def __init__(self, db_path: str) -> None:
        self.store = ObservabilityStore(db_path)
        try:
            self._db_ino = os.stat(db_path).st_ino
        except OSError:
            self._db_ino = None
        self._redactor = Redactor()
        self._record_queue: queue.Queue = queue.Queue(maxsize=_RECORD_QUEUE_MAX)
        self._span_queue: queue.Queue = queue.Queue(
            maxsize=_env_int("FW_OBS_QUEUE_MAX", _DEFAULT_QUEUE_MAX)
        )
        self._closed = False
        self._stop = threading.Event()
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
        }
        self._health_dirty = False
        self._health_lock = threading.Lock()
        # Sync-first write state (§2.4). The breaker deadline is a monotonic
        # timestamp; the ring holds terminal rows the sync path could not land.
        self._sync_lock = threading.Lock()
        self._sync_breaker_until = 0.0
        self._pending: "dict[str, tuple]" = {}
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
            )
            self._span_queue.put_nowait(("span", snapshot))
        except queue.Full:
            self._count("spans_dropped")
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
            turn_row, artifact_rows = serialize_turn_result(record)
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
                turn_row["channel_id"], turn_row["conversation_id"]
            )
        if turn_row["status"] in TERMINAL_TURN_STATUSES:
            self._remember_pending(turn_row, artifact_rows)
        try:
            self._record_queue.put(
                ("turn", turn_row, artifact_rows, 0), timeout=_RECORD_PUT_TIMEOUT_S
            )
        except queue.Full:
            self._count("records_dropped")
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
                self._count("records_dropped")
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
        """Stop signal + bounded join + final drain and commit [R7]. Idempotent.

        Emissions racing with close are dropped (the sink is closed); the
        writer drains everything already enqueued before exiting, so the last
        turn of a session is never lost.
        """
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        self._writer.join(timeout)
        if self._writer.is_alive():
            logger.warning("Observability writer did not stop within timeout")

    # -- internals -------------------------------------------------------

    def _count(self, key: str, error: Optional[str] = None) -> None:
        with self._health_lock:
            self._health[key] = int(self._health.get(key) or 0) + 1
            if error is not None:
                self._health["last_error"] = error[:500]
            self._health_dirty = True

    def _writer_loop(self) -> None:
        conn: Optional[sqlite3.Connection] = None
        try:
            conn = self.store._connect()
            while not self._stop.is_set():
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
        """Bounded retry for turn records/labels on SQLITE_BUSY; spans drop [R8]."""
        for item in items:
            kind = item[0]
            if kind == "span":
                self._count("spans_dropped")
                continue
            if kind == "flush":
                item[1].set()
                continue
            retries = item[-1]
            if retries >= _RECORD_BUSY_MAX_RETRIES:
                self._count("records_dropped")
                continue
            retried = item[:-1] + (retries + 1,)
            try:
                self._record_queue.put_nowait(retried)
            except queue.Full:
                self._count("records_dropped")

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
            self.store.set_diagnostic(conn, "writer_health", snapshot)
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


def observability_enabled(default_on: bool) -> bool:
    """FW_OBSERVABILITY master switch. fastWorkflow's own entry points pass
    default_on=True; library embedders get the sink only with FW_OBSERVABILITY=1."""
    value = _env("FW_OBSERVABILITY", "1" if default_on else "0")
    return value not in ("0", "false", "False", "no", "off")


def get_observability_sink(
    workflow_path: str, *, entry_point: bool = True
) -> Optional[SQLiteTraceSink]:
    """The process-wide sink for a workflow's observability DB, or None when
    disabled. One sink (one writer thread) per DB path; closed atexit [R7].
    Never raises — a store that cannot open degrades to no sink plus a warning.
    """
    if not observability_enabled(default_on=entry_point):
        return None
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


atexit.register(close_all_sinks)

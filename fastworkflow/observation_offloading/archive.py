"""Turn-scoped archive for persist-before-label offloads.

The evidence lives in the workflow's own observability database, in the
``offload_evidence`` and ``offload_subjects`` tables that
``observability.store`` creates with the rest of its schema. Each row is keyed
by the TURN that produced it and carries that turn's channel id, exactly like a
span or an artifact, so the evidence lives and dies with its turn: this module
writes and reads rows and never deletes them, and ``ObservabilityStore``'s
``forget_channel``, ``clear_conversations`` and ``prune`` erase them inside the
same transactions that erase the turn record.

Redaction happens at WRITE time. ``FW_OFFLOAD_EVIDENCE_REDACTION`` is the one
toggle: ``on`` (the default, and what an unconfigured deployment gets) stores
what ``observability.store.protect_offload_observation`` makes of the response
-- the trace sink's own credential scrub and capture policy, called rather than
reimplemented -- and ``off`` stores the response verbatim. Every row records
which of the two produced it, the capture-policy version and profile in force,
whether the stored bytes differ from what the command returned, and how many
bytes it returned.

Redacting at the write would change what an agent reads back mid-turn, so the
RAW text of every row whose stored bytes differ is also kept in this process's
memory for as long as its turn is live. Every in-flight read this process makes
-- the trajectory's own hot cache, ``search_memory``, answer rehydration -- is
therefore exact. That memory is released with the rest of the turn's process
state, by ``state.release_scope``, at the moments the runtime already decides a
turn is over (``StructuredContinuationReAct.bind_scope``,
``WorkflowExecutionContext._reclaim_offloading_scope`` and
``agent_runtime.reclaim_scope``). A turn resumed in a DIFFERENT process has no
such memory and reads the stored, redacted text; that is accepted.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

from fastworkflow.observability import store as observability_store

logger = logging.getLogger(__name__)


class PersistenceError(RuntimeError):
    """Handle text and digest disagree, or an alias collides."""


#: A row under the alias exists for this turn key that this scope may not
#: reuse: its text differs, or it belongs to another channel (which this
#: scope cannot read, so identical text cannot be told from different text).
ALIAS_COLLISION_MESSAGE = (
    "runtime handle alias is already stored for this turn (different text or another channel)"
)


# ---------------------------------------------------------------------------
# Redaction policy (ido-zlm)
# ---------------------------------------------------------------------------

REDACTION_ENV = "FW_OFFLOAD_EVIDENCE_REDACTION"
#: Route stored response bytes through the credential and capture pipeline.
#: The default, and what an unconfigured deployment gets.
REDACTION_ON = "on"
#: Store exactly what the command returned. For development and optimisation,
#: where the archive's job is to reproduce what the agent actually read.
REDACTION_OFF = "off"
_REDACTION_MODES = (REDACTION_ON, REDACTION_OFF)
#: Spellings an operator is likely to reach for, folded onto the two modes.
#: Anything else warns and falls back to the default: a typo must not quietly
#: change policy.
_REDACTION_ALIASES = {
    "on": REDACTION_ON, "1": REDACTION_ON, "true": REDACTION_ON,
    "yes": REDACTION_ON, "enabled": REDACTION_ON,
    "off": REDACTION_OFF, "0": REDACTION_OFF, "false": REDACTION_OFF,
    "no": REDACTION_OFF, "disabled": REDACTION_OFF,
}

#: Recorded per row when no capture policy was consulted, so ``debug`` (a real
#: profile that happens to be inert) and "not asked" never read the same.
PROFILE_NOT_CONSULTED = ""

_warned_redaction: set[str] = set()


def redaction_mode(override: Optional[str] = None) -> str:
    """The active mode: the argument, else the environment, else the default.

    ON by default and on purpose. An unconfigured deployment gets the safe
    direction, because the cost of the wrong default here is an archive that
    reproduces the agent's reading less exactly -- recoverable by re-running --
    while the cost in the other direction is a credential written verbatim to
    disk, which no later configuration change undoes.
    """
    raw = override if override is not None else os.environ.get(REDACTION_ENV, "")
    value = str(raw or "").strip().lower()
    if not value:
        return REDACTION_ON
    resolved = _REDACTION_ALIASES.get(value)
    if resolved is not None:
        return resolved
    if value not in _warned_redaction:
        _warned_redaction.add(value)
        logger.warning(
            "ignoring %s=%s: expected one of %s; using %r",
            REDACTION_ENV, raw, ", ".join(_REDACTION_MODES), REDACTION_ON,
        )
    return REDACTION_ON


def redaction_enabled(override: Optional[str] = None) -> bool:
    return redaction_mode(override) == REDACTION_ON


def capture_record_for(text: str, *, mode: Optional[str] = None) -> tuple[str, dict[str, Any]]:
    """The bytes to store for *text*, and the record that describes their fidelity.

    The record is what makes an archive auditable about itself. It carries the
    capture-policy CONTRACT version in force at the write, the profile that was
    consulted (empty when redaction was off and none was), the toggle state, and
    -- the part a reader actually needs -- whether the stored bytes DIFFER from
    what the command returned. Version plus toggle says which rules applied;
    ``redacted`` says whether they had anything to act on, which is how a reader
    tells a row that was redacted from a row that never contained a secret.

    A failure inside the pipeline stores nothing rather than storing the raw
    text: evidence capture is an optimisation and the write path already knows
    how to keep an observation inline, but silently downgrading to verbatim
    would turn a broken dependency into a credential on disk.
    """
    active = redaction_mode(mode)
    record: dict[str, Any] = {
        "capture_policy_version": str(observability_store.CAPTURE_POLICY_VERSION),
        "capture_profile": PROFILE_NOT_CONSULTED,
        "redaction": active,
        "redacted": False,
        "raw_utf8_bytes": len(text.encode("utf-8")),
    }
    if active == REDACTION_OFF:
        return text, record
    policy = observability_store.resolve_capture_policy()
    record["capture_profile"] = str(policy.profile)
    record["capture_policy_version"] = str(policy.policy_version)
    stored = observability_store.protect_offload_observation(text)
    stored = text if stored is None else str(stored)
    record["redacted"] = stored != text
    return stored, record


# ---------------------------------------------------------------------------
# The live turn's raw copies
# ---------------------------------------------------------------------------

_live_lock = threading.Lock()
#: ``scope_id -> {(db_path, alias): raw copy}`` for every row this process
#: wrote whose stored bytes differ from what the command returned. Each copy is
#: ``{"text", "text_sha256", "stored_sha256"}``; ``stored_sha256`` ties it to
#: the exact row it shadows, so a copy never overlays a row it did not write.
#: Keyed by scope id because that is the unit ``state.release_scope`` releases.
_live_raw: dict[str, dict[tuple[str, str], dict[str, str]]] = {}


def release_live_raw(scope_id: str) -> None:
    """Forget the raw copies of one scope's redacted observations."""
    with _live_lock:
        _live_raw.pop(str(scope_id), None)


def clear_live_raw() -> None:
    """Forget every raw copy this process holds."""
    with _live_lock:
        _live_raw.clear()


#: How long an event write waits for the database lock before it is dropped.
#: Events are diagnostics: a turn must never stall behind one, so this is far
#: shorter than the evidence writes' wait.
EVENT_WRITE_TIMEOUT_SECONDS = 0.5

#: How long ``list_summaries`` waits for the database lock. It only feeds a
#: suggestion printed beside an answer that is already complete, so it gives
#: up on the same terms as an event write rather than the evidence writes' 30 s.
SUMMARY_READ_TIMEOUT_SECONDS = 0.5


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class RuntimeHandleScope:
    """One turn's identity inside this process.

    ``scope_id`` keys every process-local cache of the offloading runtime and
    travels with a suspended turn, so the dataclass keeps all six fields. What
    is PERSISTED is narrower: ``turn_key`` and ``channel_id`` key and erase
    the evidence rows, and ``scope_id`` is stored beside them only so an
    erasure can reach this process's caches.

    Every read and subject write matches the channel as well as the turn key,
    so a scope pairing another channel with this turn's key reads nothing and
    cannot replace this channel's subjects; persisting under it collides.
    """

    store_identity: str
    channel_id: str
    experiment_id: str
    task_id: str
    attempt: int
    turn_key: str

    @property
    def scope_id(self) -> str:
        encoded = json.dumps(
            asdict(self),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def is_broad_scope(scope: RuntimeHandleScope) -> bool:
    """Whether *scope* is keyed by something wider than one turn.

    The process-default scope and the between-turns fallback both use the
    channel (or the process) as the turn key, so their rows span every turn --
    and, for the default scope, every session -- that fell back to it.
    Enumerating the handles of such a scope lists other turns' commands and
    subjects, so ``list_summaries`` refuses to.
    """
    return scope.turn_key == scope.channel_id


class RuntimeHandleArchive:
    #: This one holds a database it can read and write. See
    #: ``UnavailableHandleArchive`` for the one that does not.
    available = True

    def __init__(self, db_path: str) -> None:
        self.db_path = os.path.abspath(os.path.expanduser(db_path))
        # Opening the store is what creates the evidence tables, hardens the
        # file to 0600 in a 0700 directory, and removes the legacy sidecar
        # older builds kept beside it -- whether or not a trace sink ever
        # opened this database. It replaces a database from an older build
        # and refuses one from a newer build, which ``open_handle_archive``
        # degrades into ``UnavailableHandleArchive``.
        observability_store.ObservabilityStore(self.db_path)
        #: The one connection event writes reuse. Events are frequent and
        #: small, and opening and closing a connection per event cost more than
        #: the write itself; ``_event_lock`` serialises its use across threads.
        self._event_conn: Optional[sqlite3.Connection] = None
        self._event_lock = threading.Lock()

    def _connect(
        self, timeout: float = 30.0, *, shared: bool = False
    ) -> sqlite3.Connection:
        # The store's own connection settings, against ``self.db_path``. A
        # ``shared`` connection may be used from any thread, one at a time.
        conn = sqlite3.connect(
            self.db_path, timeout=timeout, check_same_thread=not shared
        )
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        return conn

    def persist(
        self,
        scope: RuntimeHandleScope,
        *,
        alias: str,
        offload_order: int,
        command_name: str,
        step_index: int,
        text: str,
        text_sha256: str,
    ) -> dict[str, Any]:
        """Store one observation and return it as this process reads it.

        ``text``/``text_sha256`` are checked against each other first and
        always: that pair is the caller's integrity claim about what the command
        returned.

        What is stored is ``capture_record_for(text)``: redacted under
        ``on``, verbatim under ``off``. The row's ``text_sha256`` covers the
        STORED bytes, so ``_decode_row`` verifies a read against the bytes
        beside it. When the stored bytes differ, the raw text is kept in memory
        for the live turn and the returned row -- which callers put in their
        hot cache -- is the raw one, so it agrees with the agent's own prompt.

        The insert is insert-or-nothing, so a re-persist of the same alias is a
        readback. A different observation under an alias already stored is a
        collision and raises; it is recognised by the raw digest when this
        process holds the raw copy, and by the stored digest otherwise.
        """
        payload = text.encode("utf-8")
        if hashlib.sha256(payload).hexdigest() != text_sha256:
            raise PersistenceError("runtime handle digest does not match its text")
        stored_text, capture = capture_record_for(text)
        stored_payload = stored_text.encode("utf-8")
        stored_sha256 = hashlib.sha256(stored_payload).hexdigest()
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                INSERT INTO offload_evidence (
                    turn_key, channel_id, scope_id, alias, offload_order,
                    command_name, step_index, text_utf8, text_sha256,
                    capture_policy_version, capture_profile, redaction,
                    redacted, raw_utf8_bytes, persisted_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(turn_key, alias) DO NOTHING
                """,
                (
                    scope.turn_key,
                    scope.channel_id,
                    scope.scope_id,
                    alias,
                    offload_order,
                    command_name,
                    step_index,
                    stored_payload,
                    stored_sha256,
                    capture["capture_policy_version"],
                    capture["capture_profile"],
                    capture["redaction"],
                    1 if capture["redacted"] else 0,
                    int(capture["raw_utf8_bytes"]),
                    _utc_now(),
                ),
            )
            conn.commit()
        row_sha256 = self._stored_sha256(scope, alias)
        if row_sha256 is None:
            raise PersistenceError(ALIAS_COLLISION_MESSAGE)
        live = self._live_copy(scope, alias, row_sha256)
        if live is not None:
            same = live["text_sha256"] == text_sha256
        else:
            same = row_sha256 in (stored_sha256, text_sha256)
        if not same:
            raise PersistenceError(ALIAS_COLLISION_MESSAGE)
        if live is None and row_sha256 == stored_sha256 and stored_text != text:
            with _live_lock:
                _live_raw.setdefault(scope.scope_id, {})[(self.db_path, str(alias))] = {
                    "text": text,
                    "text_sha256": text_sha256,
                    "stored_sha256": stored_sha256,
                }
        stored = self.get(scope, alias)
        if stored is None:
            raise PersistenceError(ALIAS_COLLISION_MESSAGE)
        return stored

    def _stored_sha256(self, scope: RuntimeHandleScope, alias: str) -> Optional[str]:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT text_sha256 FROM offload_evidence "
                "WHERE turn_key = ? AND channel_id = ? AND alias = ?",
                (scope.turn_key, scope.channel_id, str(alias)),
            ).fetchone()
        return None if row is None else str(row["text_sha256"])

    def _live_copy(
        self, scope: RuntimeHandleScope, alias: str, stored_sha256: str
    ) -> Optional[dict[str, str]]:
        """The raw copy shadowing this exact stored row, if this process holds one."""
        with _live_lock:
            copy = _live_raw.get(scope.scope_id, {}).get((self.db_path, str(alias)))
        if copy is None or copy["stored_sha256"] != stored_sha256:
            return None
        return copy

    def _read(self, scope: RuntimeHandleScope, row: sqlite3.Row) -> dict[str, Any]:
        decoded = self._decode_row(row)
        live = self._live_copy(scope, decoded["alias"], decoded["text_sha256"])
        if live is not None:
            decoded["text"] = live["text"]
            decoded["text_sha256"] = live["text_sha256"]
        return decoded

    def get(self, scope: RuntimeHandleScope, alias: str) -> Optional[dict[str, Any]]:
        with closing(self._connect()) as conn:
            row = conn.execute(
                """
                SELECT alias, offload_order, command_name, step_index,
                       text_utf8, text_sha256
                FROM offload_evidence
                WHERE turn_key = ? AND channel_id = ? AND alias = ?
                """,
                (scope.turn_key, scope.channel_id, alias),
            ).fetchone()
        return None if row is None else self._read(scope, row)

    def list(self, scope: RuntimeHandleScope, alias: str = "", *,
             timeout: float = 30.0) -> list[dict[str, Any]]:
        """Every stored row of *scope* (or just *alias*), waiting at most *timeout* for the lock."""
        query = """
            SELECT alias, offload_order, command_name, step_index,
                   text_utf8, text_sha256
            FROM offload_evidence
            WHERE turn_key = ? AND channel_id = ?
        """
        params: list[Any] = [scope.turn_key, scope.channel_id]
        if alias:
            query += " AND alias = ?"
            params.append(alias)
        query += " ORDER BY offload_order, alias"
        with closing(self._connect(timeout=timeout)) as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._read(scope, row) for row in rows]

    def list_summaries(self, scope: RuntimeHandleScope) -> list[dict[str, Any]]:
        """``alias``, ``command``, ``offload_order`` and ``utf8_bytes`` of every row.

        The stored text is measured in SQL and never loaded, for callers that
        choose among a turn's handles without reading them. ``text_utf8`` is
        written as a BLOB, so ``length()`` is its stored UTF-8 byte count and
        SQLite answers it from the record header without reading the content
        (a ``CAST`` would materialise the value first). It is the STORED
        size -- what a search of the row reads -- not ``raw_utf8_bytes``, which
        is the pre-redaction size and differs from it for a redacted row.

        Nothing is listed for a broad scope (``is_broad_scope``), and rows are
        filtered by channel as well as turn key. The wait for the database is
        ``SUMMARY_READ_TIMEOUT_SECONDS``; a failure raises.
        """
        if is_broad_scope(scope):
            return []
        with closing(self._connect(timeout=SUMMARY_READ_TIMEOUT_SECONDS)) as conn:
            rows = conn.execute(
                """
                SELECT alias, command_name, offload_order, length(text_utf8)
                FROM offload_evidence
                WHERE turn_key = ? AND channel_id = ?
                ORDER BY offload_order, alias
                """,
                (scope.turn_key, scope.channel_id),
            ).fetchall()
        return [{"alias": alias, "command": command, "offload_order": order,
                 "utf8_bytes": int(size or 0)}
                for alias, command, order, size in rows]

    # -- diagnostic events ---------------------------------------------------

    def persist_event(self, scope: RuntimeHandleScope, event: Mapping[str, Any]) -> None:
        """Store one offload event as a row of ``offload_events``.

        The event is serialized to JSON and passed through
        ``capture_record_for``, exactly like evidence text: events carry search
        questions, model reasoning and answers. The write waits at most
        ``EVENT_WRITE_TIMEOUT_SECONDS`` for the database and raises on any
        failure; ``state.record_event`` is what makes that failure harmless.
        A connection that failed is discarded, so the next event opens a fresh
        one.
        """
        text = json.dumps(dict(event), ensure_ascii=False, sort_keys=True, default=str)
        stored_text, capture = capture_record_for(text)
        row = (
            scope.turn_key,
            scope.channel_id,
            scope.scope_id,
            str(event.get("kind") or ""),
            stored_text,
            capture["redaction"],
            1 if capture["redacted"] else 0,
            _utc_now(),
        )
        with self._event_lock:
            if self._event_conn is None:
                self._event_conn = self._connect(
                    timeout=EVENT_WRITE_TIMEOUT_SECONDS, shared=True
                )
            conn = self._event_conn
            try:
                conn.execute(
                    """
                    INSERT INTO offload_events (
                        turn_key, channel_id, scope_id, kind, event_json,
                        redaction, redacted, recorded_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    row,
                )
                conn.commit()
            except BaseException:
                self._event_conn = None
                try:
                    conn.close()
                except sqlite3.Error:
                    pass
                raise

    # -- subject metadata (ido-dhw, F3) ------------------------------------

    def put_subject(
        self, scope: RuntimeHandleScope, alias: str, context_clause: str
    ) -> None:
        """Record the subject *alias* is evidence about, durably.

        An UPSERT, not insert-or-nothing: the dispatch-time stamp is a first
        answer and a later, better-informed writer may replace it. The empty
        string is a real value -- "this ran at the workflow root" -- and is
        stored as one; absence of the row is the only thing that means "no
        subject was recorded".
        """
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                INSERT INTO offload_subjects (
                    turn_key, channel_id, scope_id, alias, context_clause,
                    recorded_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(turn_key, alias) DO UPDATE SET
                    context_clause = excluded.context_clause,
                    recorded_at = excluded.recorded_at
                WHERE offload_subjects.channel_id = excluded.channel_id
                """,
                (
                    scope.turn_key,
                    scope.channel_id,
                    scope.scope_id,
                    str(alias),
                    str(context_clause or ""),
                    _utc_now(),
                ),
            )
            conn.commit()

    def get_subject(self, scope: RuntimeHandleScope, alias: str, *,
                    timeout: float = 30.0) -> Optional[str]:
        """The recorded clause, ``""`` at the root, ``None`` when UNRECORDED.

        ``None`` is the answer an alias nobody stamped gives, and it is never
        upgraded to a guess. The wait for the lock is at most *timeout*.
        """
        with closing(self._connect(timeout=timeout)) as conn:
            row = conn.execute(
                "SELECT context_clause FROM offload_subjects "
                "WHERE turn_key = ? AND channel_id = ? AND alias = ?",
                (scope.turn_key, scope.channel_id, str(alias)),
            ).fetchone()
        return None if row is None else str(row["context_clause"])

    def forget_subject(self, scope: RuntimeHandleScope, alias: str) -> None:
        """Drop the recorded subject, so *alias* reads as UNRECORDED again.

        The durable half of ``state.forget_context_clause``: a dispatch-time
        stamp that turns out to be the wrong subject must not survive on disk
        after the process that corrected it is gone.
        """
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "DELETE FROM offload_subjects "
                "WHERE turn_key = ? AND channel_id = ? AND alias = ?",
                (scope.turn_key, scope.channel_id, str(alias)),
            )
            conn.commit()

    # -- capture fidelity (ido-zlm) ----------------------------------------

    def capture_record(
        self, scope: RuntimeHandleScope, alias: str
    ) -> Optional[dict[str, Any]]:
        """How this alias's stored bytes were produced, or ``None`` if nothing is stored.

        ``redacted`` is whether the stored bytes differ from what the command
        returned, which is what tells a redacted row from one that never
        contained a secret; ``redaction`` says whether the pipeline ran at all.
        """
        with closing(self._connect()) as conn:
            row = conn.execute(
                """
                SELECT capture_policy_version, capture_profile, redaction,
                       redacted, raw_utf8_bytes, persisted_at
                FROM offload_evidence
                WHERE turn_key = ? AND channel_id = ? AND alias = ?
                """,
                (scope.turn_key, scope.channel_id, str(alias)),
            ).fetchone()
        if row is None:
            return None
        return {
            "capture_policy_version": str(row["capture_policy_version"]),
            "capture_profile": str(row["capture_profile"]),
            "redaction": str(row["redaction"]),
            "redacted": bool(row["redacted"]),
            "raw_utf8_bytes": int(row["raw_utf8_bytes"]),
            "recorded_at": str(row["persisted_at"]),
        }

    @staticmethod
    def _decode_row(row: sqlite3.Row) -> dict[str, Any]:
        payload = bytes(row["text_utf8"])
        digest = hashlib.sha256(payload).hexdigest()
        if digest != row["text_sha256"]:
            raise PersistenceError("runtime archive text failed digest verification")
        return {
            "alias": str(row["alias"]),
            "offload_order": int(row["offload_order"]),
            "command": str(row["command_name"]),
            "step_index": int(row["step_index"]),
            "text": payload.decode("utf-8"),
            "text_sha256": digest,
        }


class UnavailableHandleArchive:
    """The archive this process could not open, inert and honest about it.

    Opening or creating the observability database can fail for reasons that
    have nothing to do with the turn about to run: a read-only state root, a
    permission bit, a path that holds something which is not a database, or a
    database written by a newer build. Evidence storage is an
    availability optimisation, and the surrounding design already says what a
    storage failure costs -- ``archive_execute_observations`` and
    ``compact_trajectory`` record the refusal and leave the observation inline.
    Without this stand-in an INITIALISATION failure costs the whole turn instead,
    because it raises out of the agent's constructor and the persist-before-label
    recovery never gets to run.

    So the failure is degraded to the policy the writes already have, rather
    than to a second one. Every write refuses with the ``PersistenceError`` the
    write path expects, so the caller records ``archive_refused`` /
    ``offload_refused`` per alias and keeps the original text; every read answers
    "nothing stored here", which is the same answer as an alias that was never
    archived, so ``search_memory`` reports a miss instead of raising. Nothing is
    ever marked archived, so no part of the runtime claims durability this
    object cannot provide.

    ``db_path`` is the path that was WANTED, not a substitute: no evidence is
    silently redirected to another file.
    """

    available = False

    def __init__(self, db_path: str, error: BaseException) -> None:
        self.db_path = os.path.abspath(os.path.expanduser(db_path))
        self.error = error
        self.reason = f"{type(error).__name__}: {error}"

    def persist(self, scope: RuntimeHandleScope, **_: Any) -> dict[str, Any]:
        raise PersistenceError(
            f"runtime handle archive unavailable at {self.db_path}: {self.reason}"
        )

    def capture_record(
        self, scope: RuntimeHandleScope, alias: str
    ) -> Optional[dict[str, Any]]:
        return None

    def get(self, scope: RuntimeHandleScope, alias: str) -> Optional[dict[str, Any]]:
        return None

    def list(self, scope: RuntimeHandleScope, alias: str = "", *,
             timeout: float = 30.0) -> list[dict[str, Any]]:
        return []

    def list_summaries(self, scope: RuntimeHandleScope) -> list[dict[str, Any]]:
        return []

    def persist_event(self, scope: RuntimeHandleScope, event: Mapping[str, Any]) -> None:
        raise PersistenceError(
            f"runtime handle archive unavailable at {self.db_path}: {self.reason}"
        )

    # The subject surface, inert for the same reason the rest of this class
    # is: nothing may claim a durability this object cannot provide, and every
    # read answers "nothing recorded here" -- which is exactly the UNRECORDED
    # answer readers already handle.
    def put_subject(
        self, scope: RuntimeHandleScope, alias: str, context_clause: str
    ) -> None:
        raise PersistenceError(
            f"runtime handle archive unavailable at {self.db_path}: {self.reason}"
        )

    def get_subject(self, scope: RuntimeHandleScope, alias: str, *,
                    timeout: float = 30.0) -> Optional[str]:
        return None

    def forget_subject(self, scope: RuntimeHandleScope, alias: str) -> None:
        return None

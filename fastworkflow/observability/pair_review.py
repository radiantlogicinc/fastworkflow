"""Per-reviewer reviewed/not-reviewed progress for comparison pairs (`fix-9eg.4`).

Reviewing three runs against a pinned reference means walking a list of pairs
and needing to know, on return, which ones have already been looked at. That
is progress, not evidence and not feedback:

- **Not evidence.** It is a fact about a reviewer, recorded about an execution
  that may be sealed. It lives in a control sidecar beside the evidence DB for
  the same reasons `selection.py` gives: inspecting a store must not write to
  it, the evidence schema is fresh-only, and pruning evidence must not erase
  what somebody did.
- **Not feedback.** "Reviewed with nothing to say" is a real and common
  outcome, so reviewed-ness is stored explicitly. Counting comments would
  report every clean pair as unreviewed, which is the one wrong answer this
  module exists to prevent.

Progress is keyed to the EXACT pair — `comparison.review_pair_key`, which is
derived from both `ExecutionRef`s including their pass ids. Pinning a
different reference therefore produces different pair keys, and the rows
recorded against the old reference keep their original meaning instead of
being relabelled. Nothing here is ever rewritten in place: the pointer row
carries the current state and every change appends to history.

A PAIR CAN SPAN TWO EVIDENCE STORES, so this sidecar has the two shapes
`selection.py` already settled on, and records which one it is so they can
never be confused:

- SINGLE (``for_evidence``): one evidence store, bound by ``store_identity()``.
  An archive copies that identity with the rest of the file, so one sidecar
  serves a live store and every snapshot taken from it, and a sidecar carried to
  the wrong store is refused rather than answering about someone else's runs.
- SHARED (``open_shared_pair_review``): one control location for a workflow or a
  workspace, with several evidence stores explicitly authorized as sources under
  the SAME store ids the `ExecutionRef`s use. This is the real product shape:
  winner-versus-candidate spans registered experiments in distinct stores, and a
  single-store sidecar would refuse to record that a reviewer looked at the pair.
  Nothing is discovered — the embedder authorizes each source by
  (``store_id``, ``store_identity()``) — and a pair naming an unauthorized store
  is refused.

This module owns no HTTP surface and no UI. It is the persistence a later
server slice calls.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from fastworkflow.observability.store import FEEDBACK_PROVENANCES, _utcnow_iso

# Fresh-only, like every other schema in this package: a reader that guesses
# at a shape it did not write is the failure mode the rule forbids.
PAIR_REVIEW_SCHEMA_VERSION = 1

# Who marked it. Shares the feedback vocabulary so a human reviewer and a
# coding agent describe their origin the same way.
REVIEWER_KINDS = frozenset(FEEDBACK_PROVENANCES)

STATE_REVIEWED = "reviewed"
STATE_NOT_REVIEWED = "not_reviewed"

# How many evidence stores this control file speaks for, in `selection.py`'s
# vocabulary and for its reason: the two shapes answer a different question
# about the same tables. A single-store sidecar may trust its one attached
# store; a shared one trusts only what it was explicitly told to trust.
CONTROL_MODE_SINGLE = "single"
CONTROL_MODE_SHARED = "shared"
CONTROL_MODES = frozenset({CONTROL_MODE_SINGLE, CONTROL_MODE_SHARED})

# The source id a single-store sidecar uses for its one store, so single and
# shared sidecars have the same row shape.
PRIMARY_SOURCE_ID = "primary"

# The canonical file name for the one shared control location. The DIRECTORY is
# always supplied by the embedder; nothing here goes looking for one.
SHARED_PAIR_REVIEW_FILENAME = "pairreview.control.sqlite3"

_MAX_TEXT = 4000
_EVIDENCE_IDENTITY_KEY = "evidence_store_identity"
_MODE_KEY = "control_mode"


class PairReviewError(RuntimeError):
    """Base class for pair-review control failures."""


class PairReviewUnavailable(PairReviewError):
    """The sidecar cannot be opened or created at this path.

    Environmental, not a programming error: a read-only directory beside a
    sealed archive is the ordinary case. Callers degrade to showing the
    comparison without progress rather than failing the read.
    """


class PairReviewIdentityMismatch(PairReviewError, ValueError):
    """This sidecar records progress about a different evidence store."""


class PairReviewModeMismatch(PairReviewError, ValueError):
    """A single-store sidecar was opened as shared, or the reverse."""


class UnauthorizedEvidenceSource(PairReviewError, ValueError):
    """A pair names a store this shared sidecar was never told to trust.

    Shared mode is the cross-store shape, so the store ids in a pair are the
    source ids the embedder authorized. An unrecognised one is refused rather
    than recorded, because progress filed under an unknown store id is progress
    nobody can find again.
    """

    def __init__(self, store_id: str) -> None:
        self.store_id = store_id
        super().__init__(
            f"evidence store {store_id!r} is not an authorized source of this "
            "shared pair-review control store; authorize_source() it first"
        )


_SCHEMA_STATEMENTS = [
    """CREATE TABLE IF NOT EXISTS control_meta (
        key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL)""",
    # The stores this control file speaks for. `source_id` is the SAME string an
    # `ExecutionRef.store_id` carries, so a stored pair can be resolved back to
    # the stores it compared; `store_identity` is taken from the store itself,
    # never from a path or a caller-supplied string.
    """CREATE TABLE IF NOT EXISTS evidence_sources (
        source_id TEXT PRIMARY KEY,
        store_identity TEXT NOT NULL UNIQUE,
        label TEXT,
        authorized_at TEXT NOT NULL)""",
    # `pair_key` is `comparison.review_pair_key(left, right)`. The two ref ids
    # and their full JSON are stored beside it so a stored row can still say
    # WHAT was compared after the UI that created it has moved on, without
    # re-deriving the key.
    """CREATE TABLE IF NOT EXISTS pair_reviews (
        pair_key TEXT NOT NULL,
        reviewer TEXT NOT NULL,
        reviewer_kind TEXT NOT NULL,
        left_ref_id TEXT NOT NULL,
        right_ref_id TEXT NOT NULL,
        left_ref_json TEXT NOT NULL,
        right_ref_json TEXT NOT NULL,
        state TEXT NOT NULL,
        seq INTEGER NOT NULL,
        note TEXT,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (pair_key, reviewer))""",
    """CREATE TABLE IF NOT EXISTS pair_review_events (
        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
        pair_key TEXT NOT NULL,
        reviewer TEXT NOT NULL,
        reviewer_kind TEXT NOT NULL,
        seq INTEGER NOT NULL,
        previous_state TEXT,
        state TEXT NOT NULL,
        note TEXT,
        created_at TEXT NOT NULL,
        UNIQUE (pair_key, reviewer, seq))""",
    """CREATE INDEX IF NOT EXISTS idx_pair_reviews_left
        ON pair_reviews(left_ref_id, reviewer)""",
]


def shared_pair_review_db_path_for(control_root: str) -> str:
    """The ONE shared control file inside an embedder-supplied directory.

    ``control_root`` is a workflow root or a workspace root — whichever the
    embedder considers the boundary of a review. Nothing here decides that and
    nothing scans for candidates: the wrong root yields a different (empty)
    control file rather than a silent merge of two workspaces' progress.
    """
    return str(Path(control_root) / SHARED_PAIR_REVIEW_FILENAME)


def pair_review_db_path_for(evidence_db_path: str) -> str:
    """``<dir>/<stem>.pairreview.sqlite3`` beside an evidence DB.

    A sibling file rather than a subdirectory, for the reason `selection.py`
    documents: `archive_to` copies the DB and its `-wal` by name and
    enumerates nothing, so a sidecar here is never swept into a sealed
    archive by accident.
    """
    path = Path(evidence_db_path)
    return str(path.with_name(f"{path.stem}.pairreview.sqlite3"))


def _clean(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _require_text(value: Any, field: str) -> str:
    text = _clean(value)
    if not text:
        raise ValueError(f"{field} is required")
    if len(text) > _MAX_TEXT:
        raise ValueError(f"{field} must be at most {_MAX_TEXT} characters")
    return text


class PairReviewStore:
    """Reviewed/not-reviewed state per (pair, reviewer), with history.

    Thread/process-safe the same way `ObservabilityStore` is: every method
    opens its own short-lived connection and every write runs under
    ``BEGIN IMMEDIATE``, so SQLite's file lock serialises two writers.
    """

    def __init__(
        self,
        control_db_path: str,
        *,
        evidence_store_identity: Optional[str] = None,
        mode: str = CONTROL_MODE_SINGLE,
        sources: Optional[Mapping[str, Any]] = None,
        create: bool = True,
    ) -> None:
        if mode not in CONTROL_MODES:
            raise ValueError("mode must be one of " + ", ".join(sorted(CONTROL_MODES)))
        if mode == CONTROL_MODE_SINGLE and sources:
            raise ValueError(
                "a single-store pair-review sidecar records pairs within one "
                "evidence store; use open_shared_pair_review for several"
            )
        self.control_db_path = control_db_path
        self.mode = mode
        self._ensure_schema(create=create)
        self._bind_mode()
        if evidence_store_identity:
            self._bind_identity(evidence_store_identity)
        for source_id, evidence in (sources or {}).items():
            self.authorize_source(source_id, evidence)

    @classmethod
    def for_evidence(
        cls,
        evidence: Any,
        *,
        control_db_path: Optional[str] = None,
        create: bool = True,
    ) -> "PairReviewStore":
        """Attach to the sidecar of ONE evidence store, read-only stores too.

        The single-store shape, unchanged: one store, one sidecar, bound by
        identity. Pairs inside it are not policed by store id — there is one
        store, and the `ExecutionRef`s name it however the caller names it.
        """
        path = control_db_path or pair_review_db_path_for(evidence.db_path)
        identity = None
        try:
            identity = evidence.store_identity()
        except Exception:  # pragma: no cover - identity is optional metadata
            identity = None
        return cls(path, evidence_store_identity=identity, create=create)

    # -- schema ----------------------------------------------------------

    def _connect(self, timeout: float = 30.0) -> sqlite3.Connection:
        conn = sqlite3.connect(self.control_db_path, timeout=timeout)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self, *, create: bool) -> None:
        exists = (
            os.path.exists(self.control_db_path)
            and os.path.getsize(self.control_db_path) > 0
        )
        if not exists and not create:
            raise PairReviewUnavailable(
                f"no pair-review sidecar at {self.control_db_path}"
            )
        parent = os.path.dirname(self.control_db_path)
        try:
            if parent and create:
                os.makedirs(parent, exist_ok=True)
            conn = sqlite3.connect(self.control_db_path, timeout=30.0)
        except (OSError, sqlite3.Error) as exc:
            raise PairReviewUnavailable(
                f"cannot open pair-review sidecar {self.control_db_path}: {exc}"
            ) from exc
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            found = conn.execute("PRAGMA user_version").fetchone()[0]
            if found > PAIR_REVIEW_SCHEMA_VERSION:
                raise PairReviewUnavailable(
                    f"{self.control_db_path} has pair-review schema v{found}; this "
                    f"build reads up to v{PAIR_REVIEW_SCHEMA_VERSION}."
                )
            if found < PAIR_REVIEW_SCHEMA_VERSION:
                has_tables = (
                    conn.execute(
                        "SELECT 1 FROM sqlite_master WHERE type='table' LIMIT 1"
                    ).fetchone()
                    is not None
                )
                if has_tables:
                    raise PairReviewUnavailable(
                        f"{self.control_db_path} has pair-review schema v{found}; "
                        f"this build requires v{PAIR_REVIEW_SCHEMA_VERSION} and "
                        "carries no migration. Move the file aside to start a "
                        "new review history."
                    )
            for statement in _SCHEMA_STATEMENTS:
                conn.execute(statement)
            if found < PAIR_REVIEW_SCHEMA_VERSION:
                conn.execute(f"PRAGMA user_version = {PAIR_REVIEW_SCHEMA_VERSION}")
            conn.commit()
        except sqlite3.Error as exc:
            raise PairReviewUnavailable(
                f"cannot initialise pair-review sidecar {self.control_db_path}: {exc}"
            ) from exc
        finally:
            conn.close()
        try:
            os.chmod(self.control_db_path, 0o600)
        except OSError:
            pass

    def _bind_mode(self) -> None:
        """First writer records the shape; a later disagreement is refused."""
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT value FROM control_meta WHERE key=?", (_MODE_KEY,)
            ).fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO control_meta (key, value, updated_at) "
                    "VALUES (?, ?, ?)",
                    (_MODE_KEY, self.mode, _utcnow_iso()),
                )
                conn.commit()
                return
            found = str(row["value"])
            conn.rollback()
        if found != self.mode:
            raise PairReviewModeMismatch(
                f"{self.control_db_path} is a {found!r} pair-review control "
                f"store, opened as {self.mode!r}. Open it with "
                + (
                    "open_shared_pair_review(...)"
                    if found == CONTROL_MODE_SHARED
                    else "PairReviewStore.for_evidence(...)"
                )
            )

    # -- evidence sources ------------------------------------------------

    def authorize_source(
        self, source_id: str, evidence: Any, *, label: Optional[str] = None
    ) -> dict[str, Any]:
        """Explicitly trust one evidence store under one id. Idempotent.

        ``source_id`` is the string the `ExecutionRef`s for that store carry, so
        authorizing is what makes a cross-store pair recordable. The identity
        comes from the store and is write-once: re-authorizing the same pair is a
        no-op, a different store under the same id is refused, and the same store
        under a second id is refused by the UNIQUE identity.
        """
        source_id = _require_text(source_id, "source_id")
        identity = evidence.store_identity()
        if not identity:
            raise PairReviewError(
                f"evidence store {getattr(evidence, 'db_path', evidence)!r} has "
                "no store identity; a pair-review control store authorizes "
                "stores by identity and will not trust one it cannot name"
            )
        now = _utcnow_iso()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    "SELECT * FROM evidence_sources WHERE source_id=?", (source_id,)
                ).fetchone()
                if row is not None:
                    if str(row["store_identity"]) != identity:
                        raise PairReviewIdentityMismatch(
                            f"source {source_id!r} in {self.control_db_path} is "
                            f"bound to evidence store "
                            f"{str(row['store_identity'])!r}, not {identity!r}. A "
                            "source id names one store for the life of the "
                            "progress that references it."
                        )
                    conn.rollback()
                    return dict(row)
                clash = conn.execute(
                    "SELECT source_id FROM evidence_sources WHERE store_identity=?",
                    (identity,),
                ).fetchone()
                if clash is not None:
                    raise PairReviewIdentityMismatch(
                        f"evidence store {identity!r} is already authorized as "
                        f"source {str(clash['source_id'])!r}; authorizing it "
                        f"again as {source_id!r} would make one store's runs "
                        "comparable with themselves under two names."
                    )
                conn.execute(
                    """INSERT INTO evidence_sources
                       (source_id, store_identity, label, authorized_at)
                       VALUES (?, ?, ?, ?)""",
                    (source_id, identity, _clean(label), now),
                )
                conn.commit()
            except BaseException:
                conn.rollback()
                raise
        return {
            "source_id": source_id,
            "store_identity": identity,
            "label": _clean(label),
            "authorized_at": now,
        }

    def list_sources(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM evidence_sources ORDER BY authorized_at, source_id"
            ).fetchall()
        return [dict(row) for row in rows]

    def _require_authorized(self, store_id: str) -> None:
        """Shared mode only: both sides of a pair must name a trusted store."""
        if self.mode != CONTROL_MODE_SHARED:
            return
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM evidence_sources WHERE source_id=?", (store_id,)
            ).fetchone()
        if row is None:
            raise UnauthorizedEvidenceSource(store_id)

    def _bind_identity(self, identity: str) -> None:
        """First write wins; a later mismatch is refused, not overwritten."""
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT value FROM control_meta WHERE key=?",
                (_EVIDENCE_IDENTITY_KEY,),
            ).fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO control_meta (key, value, updated_at) VALUES (?, ?, ?)",
                    (_EVIDENCE_IDENTITY_KEY, identity, _utcnow_iso()),
                )
                conn.commit()
                return
            stored = str(row["value"])
            conn.rollback()
        if stored != identity:
            raise PairReviewIdentityMismatch(
                f"{self.control_db_path} records review progress about evidence "
                f"store {stored!r}, but was attached to {identity!r}. Review "
                "progress belongs to one evidence store and every archive "
                "taken from it."
            )

    def evidence_store_identity(self) -> Optional[str]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM control_meta WHERE key=?",
                (_EVIDENCE_IDENTITY_KEY,),
            ).fetchone()
        return None if row is None else str(row["value"])

    # -- writes ----------------------------------------------------------

    def set_state(
        self,
        left_ref: Any,
        right_ref: Any,
        *,
        state: str,
        reviewer: str,
        reviewer_kind: str,
        note: Optional[str] = None,
    ) -> dict[str, Any]:
        """Record that this reviewer has, or has not, reviewed this exact pair.

        Idempotent in effect but not in history: re-marking an already
        reviewed pair appends an event, because "looked at it again" is a
        thing that happened. `note` is optional and is not feedback — feedback
        goes through the store's recording API with its own categories.
        """
        from fastworkflow.observability.comparison import ExecutionRef, review_pair_key

        if state not in (STATE_REVIEWED, STATE_NOT_REVIEWED):
            raise ValueError(
                f"state must be {STATE_REVIEWED!r} or {STATE_NOT_REVIEWED!r}"
            )
        if reviewer_kind not in REVIEWER_KINDS:
            raise ValueError(
                "reviewer_kind must be one of " + ", ".join(sorted(REVIEWER_KINDS))
            )
        reviewer = _require_text(reviewer, "reviewer")
        note = _clean(note)
        if note is not None and len(note) > _MAX_TEXT:
            raise ValueError(f"note must be at most {_MAX_TEXT} characters")
        left = left_ref if isinstance(left_ref, ExecutionRef) else ExecutionRef.from_mapping(left_ref)
        right = right_ref if isinstance(right_ref, ExecutionRef) else ExecutionRef.from_mapping(right_ref)
        self._require_authorized(left.store_id)
        self._require_authorized(right.store_id)
        pair_key = review_pair_key(left, right)
        now = _utcnow_iso()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                previous = conn.execute(
                    "SELECT state, seq FROM pair_reviews WHERE pair_key=? AND reviewer=?",
                    (pair_key, reviewer),
                ).fetchone()
                previous_state = None if previous is None else str(previous["state"])
                seq = (0 if previous is None else int(previous["seq"])) + 1
                conn.execute(
                    """INSERT INTO pair_reviews
                       (pair_key, reviewer, reviewer_kind, left_ref_id, right_ref_id,
                        left_ref_json, right_ref_json, state, seq, note, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(pair_key, reviewer) DO UPDATE SET
                         reviewer_kind=excluded.reviewer_kind,
                         state=excluded.state, seq=excluded.seq,
                         note=excluded.note, updated_at=excluded.updated_at""",
                    (
                        pair_key,
                        reviewer,
                        reviewer_kind,
                        left.ref_id(),
                        right.ref_id(),
                        json.dumps(left.as_dict(), sort_keys=True),
                        json.dumps(right.as_dict(), sort_keys=True),
                        state,
                        seq,
                        note,
                        now,
                    ),
                )
                conn.execute(
                    """INSERT INTO pair_review_events
                       (pair_key, reviewer, reviewer_kind, seq, previous_state,
                        state, note, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        pair_key,
                        reviewer,
                        reviewer_kind,
                        seq,
                        previous_state,
                        state,
                        note,
                        now,
                    ),
                )
                conn.commit()
            except BaseException:
                conn.rollback()
                raise
        return {
            "pair_key": pair_key,
            "reviewer": reviewer,
            "reviewer_kind": reviewer_kind,
            "state": state,
            "previous_state": previous_state,
            "seq": seq,
            "note": note,
            "updated_at": now,
        }

    def mark_reviewed(self, left_ref: Any, right_ref: Any, **kwargs: Any) -> dict[str, Any]:
        """Mark a pair reviewed. No comment is required and none is implied."""
        return self.set_state(left_ref, right_ref, state=STATE_REVIEWED, **kwargs)

    def clear_reviewed(self, left_ref: Any, right_ref: Any, **kwargs: Any) -> dict[str, Any]:
        """Return a pair to not-reviewed, keeping the history of both marks."""
        return self.set_state(left_ref, right_ref, state=STATE_NOT_REVIEWED, **kwargs)

    # -- reads -----------------------------------------------------------

    def state(self, pair_key: str, reviewer: str) -> dict[str, Any]:
        """One pair's state for one reviewer.

        An unrecorded pair answers `not_reviewed` with `recorded: False`, so a
        caller can tell "nobody marked this" from "somebody unmarked it".
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM pair_reviews WHERE pair_key=? AND reviewer=?",
                (pair_key, reviewer),
            ).fetchone()
        if row is None:
            return {
                "pair_key": pair_key,
                "reviewer": reviewer,
                "state": STATE_NOT_REVIEWED,
                "recorded": False,
                "seq": 0,
                "updated_at": None,
                "note": None,
            }
        result = dict(row)
        result["recorded"] = True
        return result

    def progress(
        self, reviewer: str, pair_keys: Iterable[str]
    ) -> dict[str, Any]:
        """Reviewed counts over an explicit list of pairs, in the given order.

        The caller passes the pairs it is showing; this never invents a
        universe of pairs of its own, because which runs are "the others" is
        the caller's question, not the sidecar's.
        """
        keys = [key for key in dict.fromkeys(pair_keys) if key]
        states = {key: self.state(key, reviewer) for key in keys}
        reviewed = [key for key in keys if states[key]["state"] == STATE_REVIEWED]
        remaining = [key for key in keys if states[key]["state"] != STATE_REVIEWED]
        return {
            "reviewer": reviewer,
            "pairs": len(keys),
            "reviewed": len(reviewed),
            "remaining": len(remaining),
            # Where "resume" goes: the first pair in the caller's order that is
            # not yet reviewed, or None when the list is done.
            "next_pair_key": remaining[0] if remaining else None,
            "states": [states[key] for key in keys],
        }

    def history(
        self, pair_key: str, *, reviewer: Optional[str] = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Events newest first. Append-only: nothing here is ever rewritten."""
        query = "SELECT * FROM pair_review_events WHERE pair_key=?"
        params: list[Any] = [pair_key]
        if reviewer is not None:
            query += " AND reviewer=?"
            params.append(reviewer)
        query += " ORDER BY event_id DESC LIMIT ?"
        params.append(int(limit))
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def reviews_for_reference(self, left_ref_id: str) -> list[dict[str, Any]]:
        """Every pair recorded against one pinned reference, any reviewer.

        Pinning a new reference does not touch these rows: a different
        reference has a different `left_ref_id`, so the old pairs stay
        readable exactly as they were recorded.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM pair_reviews WHERE left_ref_id=?
                    ORDER BY updated_at, pair_key, reviewer""",
                (left_ref_id,),
            ).fetchall()
        return [dict(row) for row in rows]


def open_shared_pair_review(
    control_root: str,
    *,
    sources: Optional[Mapping[str, Any]] = None,
    control_db_path: Optional[str] = None,
    create: bool = True,
) -> PairReviewStore:
    """The cross-store shape: one control location, several authorized stores.

    ``sources`` maps the `ExecutionRef.store_id` of each store to the opened
    store itself (a `ReadOnlyObservabilityStore` is the right thing to pass —
    recording that somebody reviewed a pair must not write to the evidence). More
    can be authorized later with ``authorize_source``.
    """
    path = control_db_path or shared_pair_review_db_path_for(control_root)
    return PairReviewStore(
        path, mode=CONTROL_MODE_SHARED, sources=sources, create=create
    )


def pair_review_store_for(
    db_path: str, *, control_db_path: Optional[str] = None, create: bool = True
) -> PairReviewStore:
    """Open the pair-review sidecar for an evidence DB path, read-only on evidence.

    The entry point for readers: evidence is opened read-only, so recording
    that somebody reviewed a pair cannot write to the store being reviewed.
    """
    from fastworkflow.observability.store import ReadOnlyObservabilityStore

    evidence = ReadOnlyObservabilityStore(db_path)
    return PairReviewStore.for_evidence(
        evidence, control_db_path=control_db_path, create=create
    )


__all__ = [
    "CONTROL_MODE_SHARED",
    "CONTROL_MODE_SINGLE",
    "PAIR_REVIEW_SCHEMA_VERSION",
    "PRIMARY_SOURCE_ID",
    "REVIEWER_KINDS",
    "SHARED_PAIR_REVIEW_FILENAME",
    "STATE_NOT_REVIEWED",
    "STATE_REVIEWED",
    "PairReviewError",
    "PairReviewIdentityMismatch",
    "PairReviewModeMismatch",
    "PairReviewStore",
    "PairReviewUnavailable",
    "UnauthorizedEvidenceSource",
    "open_shared_pair_review",
    "pair_review_db_path_for",
    "pair_review_store_for",
    "shared_pair_review_db_path_for",
]

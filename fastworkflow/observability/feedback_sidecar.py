"""Recording review notes ABOUT evidence this build must not write to.

Most feedback lands in the evidence database beside the turn it is about. Two
real cases cannot:

- **Sealed workspace evidence.** A workspace store is opened read-only, and a
  sealed one is verified against a digest in the manifest. Appending a row to
  it would break the seal — the archive would no longer be the bytes somebody
  attested to — so the note has to live somewhere else.
- **A store an older build wrote.** A v6 database has no category, subcategory,
  anchor or identity columns. This build does not migrate a database it did not
  create (fresh observability schema, fix-49m.3), and half-writing a note into
  the columns that happen to exist would produce a comment the task view could
  neither file nor deduplicate.

Neither case is a reason to refuse the comment. This is exactly the shape
`selection.py` and `pair_review.py` already settled on for facts recorded
*about* an execution rather than *by* it: a small mutable control file beside
the evidence, bound to the evidence store's identity so it cannot be carried to
a different store and answer about somebody else's runs.

WHAT IS AND IS NOT HERE. The sidecar holds ORDINARY feedback — the same six
owner-confirmed subcategories, the same free-form comment, the same frozen
anchors, written through the same `feedback.record_feedback`. It is not a
second taxonomy, not an insights schema, and it is emphatically not the formal
review sidecar (`fastworkflow/review/sidecar.py`): blinded rater answers stay
there, with their own capabilities and their own blinding, and nothing in this
module reads or writes them.

READING IS THE POINT. `AnnotatedEvidence` presents one evidence store and its
sidecar as a single store, answering `list_human_feedback` and
`list_task_feedback` as the union of the two. The consolidated task read
therefore needs no knowledge of any of this: it sees one store id, one set of
rows, and deduplicates them by the same stable `feedback_uid`. A note is never
copied into the evidence file later, and the evidence file's bytes are the same
before and after somebody comments.
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional

from fastworkflow.observability import feedback as feedback_module
from fastworkflow.observability.store import _human_feedback_row, _utcnow_iso

# Fresh-only, like every other schema in this package.
FEEDBACK_SIDECAR_SCHEMA_VERSION = 1

FEEDBACK_SIDECAR_SUFFIX = ".feedback.sqlite3"

# Which file a row came out of. Not provenance (who wrote it) and not a second
# class of comment: it is how a merged read gives two independent row-id
# sequences one total order.
ORIGIN_EVIDENCE = "evidence"
ORIGIN_ANNOTATION = "annotation"

_SCHEMA = (
    """CREATE TABLE IF NOT EXISTS meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL)""",
    # Column for column the v7 `human_feedback` shape, plus the four fields a
    # join to `turns` would have supplied. The sidecar is a different file from
    # the evidence and cannot join to it, so the anchored turn's recorded scope
    # is copied in at write time -- read off the turn row, never off a label.
    """CREATE TABLE IF NOT EXISTS feedback_notes (
        feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
        feedback_uid TEXT NOT NULL UNIQUE,
        turn_key TEXT NOT NULL,
        target_kind TEXT NOT NULL,
        span_ids_json TEXT NOT NULL,
        target_label TEXT NOT NULL,
        comment TEXT NOT NULL,
        provenance TEXT NOT NULL,
        category TEXT NOT NULL,
        subcategory TEXT NOT NULL,
        anchors_json TEXT NOT NULL,
        pair_experiment_id TEXT,
        pair_task_id TEXT,
        turn_experiment_id TEXT,
        turn_task_id TEXT,
        attempt INTEGER,
        channel_id TEXT,
        created_at TEXT NOT NULL)""",
    "CREATE INDEX IF NOT EXISTS idx_feedback_notes_turn "
    "ON feedback_notes(turn_key)",
    "CREATE INDEX IF NOT EXISTS idx_feedback_notes_task "
    "ON feedback_notes(turn_experiment_id, turn_task_id)",
    "CREATE INDEX IF NOT EXISTS idx_feedback_notes_pair "
    "ON feedback_notes(pair_experiment_id, pair_task_id)",
    # Append-only, enforced by the file rather than by convention: a recorded
    # comment is somebody's statement and editing one in place would leave no
    # trace that it had said something else.
    """CREATE TRIGGER IF NOT EXISTS feedback_notes_no_update
        BEFORE UPDATE ON feedback_notes
        BEGIN SELECT RAISE(ABORT, 'recorded feedback is append-only'); END""",
    """CREATE TRIGGER IF NOT EXISTS feedback_notes_no_delete
        BEFORE DELETE ON feedback_notes
        BEGIN SELECT RAISE(ABORT, 'recorded feedback is append-only'); END""",
)

_SELECT = (
    "SELECT feedback_id, feedback_uid, turn_key, target_kind, span_ids_json, "
    "target_label, comment, provenance, category, subcategory, anchors_json, "
    "turn_experiment_id, turn_task_id, attempt, channel_id, created_at "
    "FROM feedback_notes"
)


class FeedbackSidecarError(RuntimeError):
    """The sidecar does not belong to the evidence it was opened against."""


def feedback_db_path_for(evidence_db_path: str) -> str:
    """``<dir>/<stem>.feedback.sqlite3`` beside an evidence DB.

    A sibling file rather than a subdirectory, for the reason `selection.py`
    documents: `archive_to` copies the DB and its `-wal` by name and enumerates
    nothing, so a sidecar here is never swept into a sealed archive by
    accident. It is also not one of the two names a sealed workspace store
    rejects as an unsealed sidecar (`-wal`, `-shm`), so its presence beside a
    sealed archive does not invalidate the archive.
    """
    path = Path(evidence_db_path)
    return str(path.with_name(f"{path.stem}{FEEDBACK_SIDECAR_SUFFIX}"))


class FeedbackAnnotationStore:
    """Append-only notes about one evidence store, in a file of their own.

    Bound to the evidence store's ``store_identity()``. An archive copies that
    identity along with the rest of the file, so one sidecar serves a live
    store and every snapshot taken from it; a sidecar carried to a different
    store is refused rather than answering about the wrong runs.

    Concurrency-safe the way `ObservabilityStore` is: a short-lived connection
    per call, and ``BEGIN IMMEDIATE`` around the write.
    """

    def __init__(
        self,
        control_db_path: str,
        *,
        evidence_store_identity: Optional[str] = None,
        create: bool = True,
    ) -> None:
        self.control_db_path = control_db_path
        self.read_only = not create
        path = Path(control_db_path)
        existed = path.exists()
        if not create and not existed:
            raise FileNotFoundError(control_db_path)
        # Validated BEFORE any DDL, and on a read-only connection, so an
        # incompatible sidecar is refused with its bytes exactly as they were.
        # Creating the schema first and checking afterwards would "fix" a file
        # this build has already decided it does not understand.
        if existed:
            self._check_existing(evidence_store_identity)
        if create:
            self._create_schema(path)
            if evidence_store_identity:
                self._bind_identity(evidence_store_identity)

    # -- lifecycle -------------------------------------------------------

    @contextmanager
    def _connect(
        self, timeout: float = 30.0, *, write: bool = False
    ) -> Iterator[sqlite3.Connection]:
        """One short-lived connection, closed on the way out.

        A read opens `mode=ro`, so a reader physically cannot create the file,
        add a table or stamp a version into somebody else's control file. A
        store opened for reading refuses a write connection outright.

        Deliberately NOT in WAL mode, unlike the evidence store. This file's
        traffic is a few short appends, and a rollback journal keeps it
        self-contained: one file beside the evidence, with nothing of its own
        left behind next to a sealed archive, and nothing a copy of it could
        leave behind unwritten.
        """
        if write and self.read_only:
            raise FeedbackSidecarError(
                f"{self.control_db_path} was opened for reading; recording a "
                "note opens it for writing through the authorized write path"
            )
        if write:
            conn = sqlite3.connect(
                self.control_db_path, timeout=timeout, check_same_thread=False
            )
        else:
            conn = sqlite3.connect(
                f"{Path(self.control_db_path).resolve().as_uri()}?mode=ro",
                uri=True,
                timeout=timeout,
                check_same_thread=False,
            )
        conn.row_factory = sqlite3.Row
        try:
            if write:
                conn.execute("PRAGMA foreign_keys=ON")
            yield conn
        finally:
            conn.close()

    def _check_existing(self, identity: Optional[str]) -> None:
        """Refuse a file this build does not understand, without touching it."""
        try:
            with self._read_only_connection() as conn:
                rows = conn.execute(
                    "SELECT key, value FROM meta WHERE key IN "
                    "('schema_version','evidence_store_identity')"
                ).fetchall()
        except sqlite3.Error as exc:
            raise FeedbackSidecarError(
                f"{self.control_db_path} is not a readable feedback sidecar: {exc}"
            ) from exc
        meta = {str(row["key"]): str(row["value"]) for row in rows}
        recorded = meta.get("schema_version")
        if recorded != str(FEEDBACK_SIDECAR_SCHEMA_VERSION):
            raise FeedbackSidecarError(
                f"{self.control_db_path} is a v{recorded} feedback sidecar; "
                f"this build reads and writes v{FEEDBACK_SIDECAR_SCHEMA_VERSION} "
                "and carries no migration"
            )
        bound = meta.get("evidence_store_identity")
        if identity and bound is not None and bound != identity:
            raise FeedbackSidecarError(
                f"{self.control_db_path} records notes about store "
                f"{bound!r}, not {identity!r}"
            )

    @contextmanager
    def _read_only_connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(
            f"{Path(self.control_db_path).resolve().as_uri()}?mode=ro",
            uri=True,
            timeout=30.0,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _create_schema(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect(write=True) as conn:
            for statement in _SCHEMA:
                conn.execute(statement)
            conn.execute(
                "INSERT OR IGNORE INTO meta(key,value) VALUES('schema_version',?)",
                (str(FEEDBACK_SIDECAR_SCHEMA_VERSION),),
            )
            conn.commit()

    def _meta(self, key: str) -> Optional[str]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM meta WHERE key=?", (key,)
            ).fetchone()
        return str(row["value"]) if row else None

    def _bind_identity(self, identity: str) -> None:
        recorded = self._meta("evidence_store_identity")
        if recorded is None:
            with self._connect(write=True) as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO meta(key,value) "
                    "VALUES('evidence_store_identity',?)",
                    (identity,),
                )
                conn.commit()
            recorded = self._meta("evidence_store_identity")
        if recorded != identity:
            raise FeedbackSidecarError(
                f"{self.control_db_path} records notes about store "
                f"{recorded!r}, not {identity!r}"
            )

    @property
    def evidence_store_identity(self) -> Optional[str]:
        return self._meta("evidence_store_identity")

    # -- writes ----------------------------------------------------------

    def append(
        self,
        *,
        turn_key: str,
        note: Mapping[str, Any],
        anchors: Any,
        scope: Mapping[str, Any],
        scrub: Optional[Any] = None,
    ) -> dict[str, Any]:
        """Record one already-validated note. Returns the stored row.

        `note` is the output of `feedback.normalize_note` and `anchors` a
        validated `feedback.FeedbackAnchors`: this file holds no evidence and
        so cannot check either, which is why nothing calls it directly —
        `AnnotatedEvidence.add_human_feedback` does the evidence checks first.

        `scrub` is the evidence store's credential redactor, applied to the
        LABELS inside the serialized anchor by
        `feedback.scrubbed_anchor_dict`. The anchor repeats `target_label` and
        `ref.label`, so scrubbing only the columns would leave a credential
        somebody pasted into a component label sitting in the JSON beside it.
        """
        paired = anchors.paired
        feedback_uid = f"fb-{uuid.uuid4().hex}"
        created_at = _utcnow_iso()
        anchors_json = json.dumps(
            feedback_module.scrubbed_anchor_dict(anchors, scrub),
            ensure_ascii=False,
        )
        with self._connect(write=True) as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "INSERT INTO feedback_notes "
                "(feedback_uid,turn_key,target_kind,span_ids_json,target_label,"
                "comment,provenance,category,subcategory,anchors_json,"
                "pair_experiment_id,pair_task_id,turn_experiment_id,"
                "turn_task_id,attempt,channel_id,created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    feedback_uid, turn_key, note["target_kind"],
                    json.dumps(note["span_ids"]), note["target_label"],
                    note["comment"], note["provenance"], note["category"],
                    note["subcategory"],
                    anchors_json,
                    paired.ref.experiment_id if paired else None,
                    paired.ref.task_id if paired else None,
                    scope.get("experiment_id"), scope.get("task_id"),
                    scope.get("attempt"), scope.get("channel_id"),
                    created_at,
                ),
            )
            conn.commit()
        stored = [
            row for row in self.list_for_turn(turn_key)
            if row.get("feedback_uid") == feedback_uid
        ]
        return stored[0] if stored else {}

    # -- reads -----------------------------------------------------------

    def list_for_turn(self, turn_key: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                f"{_SELECT} WHERE turn_key=? ORDER BY created_at, feedback_id",
                (turn_key,),
            ).fetchall()
        return [_annotation_row(row) for row in rows]

    def list_for_task(
        self, *, experiment_id: str, task_id: str
    ) -> list[dict[str, Any]]:
        """Notes anchored in this task, ORed with notes whose frozen pair
        anchor names it — the same two-sided question the evidence store's
        `list_task_feedback` answers, so a comparison comment written against
        sealed evidence is still findable from both of its tasks."""
        with self._connect() as conn:
            rows = conn.execute(
                f"{_SELECT} WHERE (turn_experiment_id=? AND turn_task_id=?) "
                "   OR (pair_experiment_id=? AND pair_task_id=?) "
                "ORDER BY created_at, feedback_id",
                (experiment_id, task_id, experiment_id, task_id),
            ).fetchall()
        return [_annotation_row(row) for row in rows]


def _identity_of(evidence: Any) -> Optional[str]:
    try:
        return evidence.store_identity()
    except Exception:  # pragma: no cover - identity is optional metadata
        return None


def reader_for(evidence: Any, *, control_db_path: Optional[str] = None) -> Any:
    """Whatever answers a READ, having created and written nothing.

    No sidecar file means there are no annotations, which is a fact the
    evidence store already reports correctly — so the evidence store is
    returned unchanged and nothing appears on disk next to it. Listing a
    turn's comments must not be the act that stamps a control file beside
    somebody's sealed archive.

    An existing sidecar is opened read-only. One this build does not
    understand, or one bound to a different store, raises
    `FeedbackSidecarError` with its bytes untouched rather than being
    silently re-stamped into something readable.
    """
    path = control_db_path or feedback_db_path_for(evidence.db_path)
    if not os.path.exists(path):
        return evidence
    sidecar = FeedbackAnnotationStore(
        path, evidence_store_identity=_identity_of(evidence), create=False
    )
    return AnnotatedEvidence(evidence, sidecar)


def _annotation_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Project a sidecar row into the store's own wire shape.

    Reuses `_human_feedback_row` rather than restating it, so a note recorded
    beside sealed evidence and a note recorded inside a writable store read
    back identically — the reader cannot tell, and does not need to.
    """
    value = _human_feedback_row(row)
    value["origin"] = ORIGIN_ANNOTATION
    return value


class AnnotatedEvidence:
    """One read-only evidence store plus its sidecar, presented as one store.

    Implements the small surface `feedback.record_feedback` and
    `feedback.consolidate_task_feedback` use — identity, turn and span reads,
    the two feedback lists, and the append — so neither has to know that the
    comment it is writing or reading lives in a second file.
    """

    def __init__(self, evidence: Any, sidecar: FeedbackAnnotationStore) -> None:
        self.evidence = evidence
        self.sidecar = sidecar
        self.db_path = evidence.db_path

    @classmethod
    def for_writing(
        cls, evidence: Any, *, control_db_path: Optional[str] = None
    ) -> "AnnotatedEvidence":
        """Attach a sidecar that may be created. Only an authorized POST.

        This is the one path that brings a control file into existence beside
        somebody's evidence. Reads use `reader_for`, which creates nothing.
        """
        path = control_db_path or feedback_db_path_for(evidence.db_path)
        sidecar = FeedbackAnnotationStore(
            path, evidence_store_identity=_identity_of(evidence), create=True
        )
        return cls(evidence, sidecar)

    # -- the evidence half, unchanged ------------------------------------

    def store_identity(self) -> Optional[str]:
        return self.evidence.store_identity()

    def get_turn(self, turn_key: str) -> Optional[dict[str, Any]]:
        return self.evidence.get_turn(turn_key)

    def get_spans(self, turn_key: str) -> Iterable[Mapping[str, Any]]:
        return self.evidence.get_spans(turn_key)

    # -- the union ---------------------------------------------------------

    def list_human_feedback(self, turn_key: str) -> list[dict[str, Any]]:
        return _merged(
            self.evidence.list_human_feedback(turn_key),
            self.sidecar.list_for_turn(turn_key),
        )

    def list_task_feedback(
        self, *, experiment_id: str, task_id: str
    ) -> list[dict[str, Any]]:
        return _merged(
            self.evidence.list_task_feedback(
                experiment_id=experiment_id, task_id=task_id
            ),
            self.sidecar.list_for_task(
                experiment_id=experiment_id, task_id=task_id
            ),
        )

    # -- the write ---------------------------------------------------------

    def add_human_feedback(
        self,
        turn_key: str,
        *,
        target_kind: str,
        span_ids: list[str],
        target_label: str,
        provenance: str,
        comment: Optional[str] = None,
        category: Any = None,
        subcategory: Any = None,
        anchors: Any = None,
    ) -> dict[str, Any]:
        """Same call as the evidence store's, landing in the sidecar.

        Every check the evidence store makes is made here against the same
        evidence — the turn is recorded, the spans belong to it, the anchor
        names the turn — because being unable to write to a database is no
        reason to record a comment about a turn that is not in it.
        """
        if not isinstance(turn_key, str) or not turn_key:
            raise ValueError("turn_key is required")
        note = feedback_module.normalize_note(
            target_kind=target_kind,
            span_ids=span_ids,
            target_label=target_label,
            provenance=provenance,
            comment=comment,
            category=category,
            subcategory=subcategory,
        )
        row = self.evidence.get_turn(turn_key)
        if row is None:
            raise ValueError("turn not found")
        recorded = {
            str(span["span_id"])
            for span in self.evidence.get_spans(turn_key)
            if span.get("span_id")
        }
        if not set(note["span_ids"]).issubset(recorded):
            raise ValueError("feedback spans must belong to the selected turn")
        if anchors is None:
            # The evidence store's own default anchor, not a second one: it
            # copies the scope off the turn row, which is the only place this
            # build takes it from.
            anchors = self.evidence._own_anchor(
                turn_key,
                target_kind=note["target_kind"],
                span_ids=note["span_ids"],
                target_label=note["target_label"],
            )
        if anchors.primary.turn_key != turn_key:
            raise ValueError("the primary anchor must name the turn being annotated")
        scrub = getattr(self.evidence, "_scrub", None)
        if not callable(scrub):  # pragma: no cover - every store has one
            scrub = None
        if scrub is not None:
            note = dict(note)
            note["comment"] = scrub(note["comment"])
            note["target_label"] = scrub(note["target_label"])
        return self.sidecar.append(
            turn_key=turn_key,
            note=note,
            anchors=anchors,
            scrub=scrub,
            scope={
                "experiment_id": row.get("experiment_id"),
                "task_id": row.get("task_id"),
                "attempt": row.get("attempt"),
                "channel_id": row.get("channel_id"),
            },
        )


def _merged(
    evidence_rows: Iterable[Mapping[str, Any]],
    annotation_rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Chronological across both files, with a total order either way.

    Two rows written in the same second in two different files still have
    exactly one position, because `origin` breaks the tie before the row id
    does — the two id sequences are independent and would otherwise interleave
    differently depending on which file answered first.
    """
    rows = [
        {**dict(row), "origin": row.get("origin") or ORIGIN_EVIDENCE}
        for row in evidence_rows
    ]
    rows.extend(dict(row) for row in annotation_rows)
    rows.sort(
        key=lambda row: (
            str(row.get("created_at") or ""),
            str(row.get("origin") or ""),
            int(row.get("feedback_id") or 0),
        )
    )
    return rows

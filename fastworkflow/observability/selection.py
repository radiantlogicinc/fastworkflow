"""Current-winner pointer and append-only selection history (`fix-9eg.17.1`).

A winner is a JUDGEMENT ABOUT evidence, not evidence. It is recorded in a
mutable control sidecar next to the observability DB — never in the evidence DB
itself — for three reasons:

- Sealed/archived evidence is immutable and frequently opened read-only
  (``ReadOnlyObservabilityStore``). A decision recorded about an archived run
  must not require writing to the archive, and inspecting a store must not
  change its bytes.
- The evidence schema is fresh-only (`fix-49m.3`): there is no migration, so a
  new evidence table would refuse every store written by an older build,
  including the sealed archives this feature must keep readable.
- Control metadata has a different lifecycle from evidence. Pruning evidence
  does not retract a decision somebody made, and `prune()` must not be able to
  delete history.

Scope of a contest: WORKFLOW + BENCHMARK LINEAGE, never DB identity. A control
sidecar therefore has two shapes, and the shape is recorded in the file so the
two can never be confused:

- SINGLE (the default, and what ``for_evidence``/``selection_control_for``
  build): one evidence store, bound by ``store_identity()``. An archive copies
  that identity with the rest of the file, so the same control DB serves a live
  store and every snapshot taken from it, and a sidecar carried to the wrong
  store is refused rather than silently answering about someone else's runs.
- SHARED (``open_shared_control``): one canonical control location for a
  workflow or a workspace, with SEVERAL evidence stores explicitly authorized
  as sources. This is the real product shape — registered experiments are bound
  to distinct stores, and a winner-versus-candidate comparison must be able to
  span them. Nothing is discovered: the embedder supplies the control location,
  authorizes each source by (``source_id``, ``store_identity()``), and supplies
  a resolver that hands back the store for a ``source_id``. This module never
  searches the filesystem for stores and never accepts a store whose identity
  does not match the one authorized under that id.

Registration can run BEFORE any evidence store exists. ``register_experiment_
reference`` records an experiment from an explicit ``ExperimentReference``
(workflow, benchmark lineage, created_at) so a UI that creates an experiment
can initialize the first winner at creation time rather than at execution time.
``bind_experiment_source`` attaches the evidence store later, explicitly, and
changes no selection. There is no evidence migration anywhere in this path.

Scopes. This module implements the EXPERIMENT-level winner only. The tables
carry ``scope_kind``/``scope_key`` and reserved ``task_id``/``attempt`` columns
so the later per-task "best run" selection (`fix-9eg.17.4`) can reuse this
machinery without a control-schema break. The two scopes live under different
primary keys by construction, so neither can overwrite the other: selecting a
best run for a task cannot promote its experiment, and promoting an experiment
cannot touch a task's best-run history. ``TASK_BEST_SCOPE`` is declared but not
wired: nothing in this module writes it yet.

Selecting a winner records a reference. It does not deploy code, restore data,
rerun anything, or claim the winner succeeded — ``current_winner()`` reports the
experiment's ACTUAL status, which for the automatic first winner is usually
``running``.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional, Union

from fastworkflow.observability.store import (
    FEEDBACK_PROVENANCES,
    ExperimentNotFound,
    ObservabilityStore,
    Redactor,
    StoreIdentityMismatch,
    _utcnow_iso,
)
from fastworkflow.utils.logging import logger

# The control sidecar carries its own version. It is fresh-only for the same
# reason the evidence schema is: a reader that guesses at a shape it did not
# write is the failure mode the rule exists to forbid. A newer sidecar is
# refused; an older one is refused with the path to move aside.
CONTROL_SCHEMA_VERSION = 2

# How many evidence stores this control file speaks for. Recorded in the file
# because the two shapes answer a different question about the same tables: a
# single-store sidecar may trust its one attached store, a shared one may not
# trust anything it was not explicitly told to trust.
CONTROL_MODE_SINGLE = "single"
CONTROL_MODE_SHARED = "shared"
CONTROL_MODES = frozenset({CONTROL_MODE_SINGLE, CONTROL_MODE_SHARED})

# The source id a single-store sidecar uses for its one store, so single and
# shared sidecars have the same row shape and a single-store history stays
# readable by the shared code path.
PRIMARY_SOURCE_ID = "primary"

EXPERIMENT_SCOPE = "experiment"
# Reserved for `fix-9eg.17.4` (best run for one task within an experiment).
# Declared here so the column vocabulary is fixed before two writers exist;
# this module never writes it.
TASK_BEST_SCOPE = "task_best"

DECISION_INITIAL = "initial"
DECISION_PROMOTE = "promote"
DECISION_KEEP = "keep"
DECISION_UNDECIDED = "undecided"
# What `retire_experiment` appends when a registration leaves the contest. Like
# `initial` it is written by the system, never offered to clients: withdrawing
# an experiment is a consequence of deleting it, not a verdict about it.
DECISION_RETIRE = "retire"
# `initial` is not offered to clients: it is what registration writes once, for
# the first experiment in a group.
CLIENT_DECISIONS = frozenset({DECISION_PROMOTE, DECISION_KEEP, DECISION_UNDECIDED})

GROUP_BENCHMARK = "benchmark"
GROUP_ADHOC = "adhoc"

# Who decided. Shares the feedback vocabulary so a human comment and an agent's
# structured decision describe their origin the same way, plus `system` for the
# automatic first-experiment initialization, which no actor asked for.
SYSTEM_ACTOR_KIND = "system"
SELECTION_ACTOR_KINDS = frozenset(FEEDBACK_PROVENANCES | {SYSTEM_ACTOR_KIND})

_AUTOMATIC_PROVENANCE = "automatic_first_experiment"
_RETIREMENT_PROVENANCE = "registration_deleted"
_MAX_TEXT = 4000

# How many older unadopted peers a refusal reports. The list is diagnostic, not
# a work queue: `adopt_existing_experiments` is what resolves it.
_MAX_REPORTED_PEERS = 50

# Bounded retry for the one statement in schema creation that ignores the busy
# timeout (see `_ensure_schema`). Short: this is a cold-start collision between
# processes that all want the same thing, not a queue.
_SCHEMA_OPEN_ATTEMPTS = 5
_SCHEMA_OPEN_BACKOFF = 0.05

# Outcomes of `initialize_winner_for`, which runs AFTER the evidence row is
# committed and therefore cannot fail the creation it follows.
INIT_RECORDED = "recorded"
INIT_UNAVAILABLE = "unavailable"
INIT_REFUSED = "refused"


class SelectionControlError(RuntimeError):
    """Base class for control-metadata failures."""


class SelectionControlUnavailable(SelectionControlError):
    """The control sidecar cannot be opened or created at this path.

    Environmental, not a programming error: a read-only directory beside a
    sealed archive is the ordinary case. Callers degrade (show evidence without
    a winner) instead of failing the read.
    """


class ControlStoreIdentityMismatch(StoreIdentityMismatch):
    """A store was attached under a source id bound to a different store."""


class ControlModeMismatch(SelectionControlError):
    """This sidecar speaks for a different number of evidence stores.

    A single-store sidecar opened as shared (or the reverse) would answer with
    the same tables under different trust rules, so the shape is refused up
    front rather than silently reinterpreted.
    """


class UnauthorizedEvidenceSource(SelectionControlError):
    """An evidence source that was never explicitly authorized here.

    The refusal is the feature: a shared control location must not answer about
    — or accept registrations from — a store the embedder did not name.
    """

    def __init__(self, source_id: str) -> None:
        self.source_id = source_id
        super().__init__(
            f"evidence source {source_id!r} is not authorized for this control "
            "store; call authorize_source(source_id, evidence) first"
        )


class EvidenceSourceUnresolved(SelectionControlError):
    """An authorized source that the embedder's resolver did not hand back.

    Distinct from unauthorized: the control store knows this source and trusts
    it, but nothing supplied an open handle to read it with.
    """

    def __init__(self, source_id: str) -> None:
        self.source_id = source_id
        super().__init__(
            f"no evidence store was supplied for authorized source {source_id!r}"
        )


class ExperimentSourceCollision(SelectionControlError):
    """One experiment id claimed by two different evidence stores.

    Ids are unique per store, not globally, so two stores can legitimately hold
    ``exp-1``. Inside one comparison group that is an ambiguity, not a merge:
    the second claim is refused so a winner pointer can never name a row whose
    store nobody can identify.
    """

    def __init__(self, experiment_id: str, bound_source_id: str, source_id: str) -> None:
        self.experiment_id = experiment_id
        self.bound_source_id = bound_source_id
        self.source_id = source_id
        super().__init__(
            f"experiment {experiment_id!r} is already registered from evidence "
            f"source {bound_source_id!r}; it cannot also be registered from "
            f"{source_id!r}. Experiment ids are unique per store, not across "
            "stores."
        )


class SelectionRetirementRefused(SelectionControlError):
    """This experiment cannot leave the contest. Two reasons, kept apart:

    - ``is_current_winner`` — the group's pointer names it and the group has
      other members (or the caller did not pass ``allow_sole_winner``).
      Withdrawing it would either leave the pointer naming something that is
      gone, or make this module pick a successor nobody asked it to pick.
      Neither is a deletion's business, so the caller is refused and told to
      select a different winner first.
    - ``has_evidence`` — a run was recorded under it. Retirement is for a
      registration that never happened; evidence is history, and history stays
      in the contest it was part of.
    """

    def __init__(self, experiment_id: str, group_id: Optional[str], reason: str, detail: str) -> None:
        self.experiment_id = experiment_id
        self.group_id = group_id
        self.reason = reason
        super().__init__(detail)


class UnknownComparisonGroup(KeyError):
    """No comparison group with that id exists in this sidecar."""

    def __init__(self, group_id: str) -> None:
        self.group_id = group_id
        super().__init__(f"unknown comparison group {group_id!r}")


class ExperimentNotInGroup(ValueError):
    """A candidate must already belong to the group it is judged within."""

    def __init__(self, experiment_id: str, group_id: str) -> None:
        self.experiment_id = experiment_id
        self.group_id = group_id
        super().__init__(
            f"experiment {experiment_id!r} is not a member of comparison group "
            f"{group_id!r}; a winner is only comparable within its group"
        )


class NoCurrentSelection(ValueError):
    """A decision was recorded about a scope that has no current selection.

    For the experiment winner that is almost always a group whose election is
    being withheld because the bootstrap could not read every evidence store
    (see `adopt_existing_experiments`); the message says so, because "no
    current winner" alone reads like a bug rather than a wait.
    """


class StaleSelection(ValueError):
    """The decision was made against a winner that has since been replaced.

    Carries what the caller expected and what is actually current so a UI or an
    agent can say which decision won, rather than reporting a generic conflict.
    """

    def __init__(
        self,
        *,
        group_id: str,
        expected_selection_id: str,
        current_selection_id: str,
        current_experiment_id: str,
        scope_kind: str = "experiment",
        scope_key: str = "",
    ) -> None:
        self.group_id = group_id
        self.expected_selection_id = expected_selection_id
        self.current_selection_id = current_selection_id
        self.current_experiment_id = current_experiment_id
        self.scope_kind = scope_kind
        self.scope_key = scope_key
        where = (
            f"comparison group {group_id!r}"
            if not scope_key
            else f"{scope_kind} scope {scope_key!r}"
        )
        selected = "the winner" if not scope_key else "the selection"
        super().__init__(
            f"selection {expected_selection_id!r} is no longer current for "
            f"{where}: {selected} is now "
            f"{current_experiment_id!r} (selection {current_selection_id!r}). "
            "Re-read the current winner and decide again."
        )


_CONTROL_SCHEMA_STATEMENTS = [
    """CREATE TABLE IF NOT EXISTS control_meta (
        key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL)""",
    # The authorized evidence stores. `store_identity` is UNIQUE so one store
    # cannot be authorized twice under two ids, which would let the same run be
    # registered as two comparable experiments.
    """CREATE TABLE IF NOT EXISTS evidence_sources (
        source_id TEXT PRIMARY KEY,
        store_identity TEXT NOT NULL UNIQUE,
        label TEXT,
        authorized_at TEXT NOT NULL)""",
    # One group per (workflow, benchmark lineage). `benchmark_version` is NOT
    # part of the identity: the winner is scoped to the lineage across versions
    # (`fix-9eg.17`), and each member row records the version it ran so a
    # version change is visible rather than silently starting a new contest.
    # Neither is the evidence store: two runs of one lineage held in two stores
    # are the exact comparison this feature exists for.
    """CREATE TABLE IF NOT EXISTS comparison_groups (
        group_id TEXT PRIMARY KEY,
        group_kind TEXT NOT NULL,
        workflow_name TEXT,
        benchmark_id TEXT,
        created_at TEXT NOT NULL,
        created_from_experiment_id TEXT NOT NULL)""",
    # `source_id` is NULL for an experiment registered from a reference before
    # any store was bound; `bind_experiment_source` fills it in once.
    """CREATE TABLE IF NOT EXISTS comparison_group_members (
        group_id TEXT NOT NULL,
        experiment_id TEXT NOT NULL,
        source_id TEXT,
        benchmark_version TEXT,
        benchmark_digest_sha256 TEXT,
        created_at TEXT,
        joined_at TEXT NOT NULL,
        PRIMARY KEY (group_id, experiment_id))""",
    # An experiment belongs to exactly one group. Membership is first-write-wins
    # (see `register_experiment`): a pin added on a later re-create must not move
    # an experiment out from under a winner pointer that already references it.
    """CREATE UNIQUE INDEX IF NOT EXISTS idx_group_member_experiment
        ON comparison_group_members(experiment_id)""",
    # `task_id`/`attempt` are NULL at experiment scope and reserved for the
    # task-best scope (`fix-9eg.17.4`).
    """CREATE TABLE IF NOT EXISTS selection_pointers (
        scope_kind TEXT NOT NULL,
        group_id TEXT NOT NULL,
        scope_key TEXT NOT NULL,
        experiment_id TEXT NOT NULL,
        task_id TEXT,
        attempt INTEGER,
        selection_id TEXT NOT NULL,
        decision TEXT NOT NULL,
        decision_seq INTEGER NOT NULL,
        decided_at TEXT NOT NULL,
        PRIMARY KEY (scope_kind, group_id, scope_key))""",
    """CREATE TABLE IF NOT EXISTS selection_decisions (
        decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
        scope_kind TEXT NOT NULL,
        group_id TEXT NOT NULL,
        scope_key TEXT NOT NULL,
        seq INTEGER NOT NULL,
        decision TEXT NOT NULL,
        previous_experiment_id TEXT,
        previous_task_id TEXT,
        previous_attempt INTEGER,
        previous_selection_id TEXT,
        candidate_experiment_id TEXT,
        candidate_task_id TEXT,
        candidate_attempt INTEGER,
        new_experiment_id TEXT,
        new_task_id TEXT,
        new_attempt INTEGER,
        new_selection_id TEXT,
        actor TEXT NOT NULL,
        actor_kind TEXT NOT NULL,
        provenance TEXT NOT NULL,
        rationale TEXT,
        created_at TEXT NOT NULL,
        UNIQUE (scope_kind, group_id, scope_key, seq))""",
    """CREATE INDEX IF NOT EXISTS idx_selection_decisions_scope
        ON selection_decisions(scope_kind, group_id, scope_key, seq)""",
]

_MODE_KEY = "control_mode"

# The canonical file name for the one control location of a workflow or a
# workspace. The DIRECTORY is always supplied by the embedder; this module
# derives a name inside it and never goes looking for one.
SHARED_CONTROL_FILENAME = "selection.control.sqlite3"

EvidenceSourceResolver = Callable[[str], Optional[ObservabilityStore]]
SourceSpec = Union[Mapping[str, ObservabilityStore], EvidenceSourceResolver]


def control_db_path_for(evidence_db_path: str) -> str:
    """``<dir>/<stem>.selection.sqlite3`` beside an evidence DB.

    A sibling file rather than a subdirectory, because `archive_to` copies the
    DB and its `-wal` by name and enumerates nothing: a sidecar here is never
    swept into a sealed archive by accident.
    """
    path = Path(evidence_db_path)
    return str(path.with_name(f"{path.stem}.selection.sqlite3"))


def shared_control_db_path_for(control_root: str) -> str:
    """The ONE shared control file inside an embedder-supplied directory.

    ``control_root`` is a workflow root or a workspace root — whichever the
    embedder considers the boundary of a contest. Nothing here decides that,
    and nothing here scans for candidates: passing the wrong root produces a
    different (empty) control file rather than a silent merge of two
    workspaces' histories.
    """
    return str(Path(control_root) / SHARED_CONTROL_FILENAME)


def comparison_group_identity(experiment: Mapping[str, Any]) -> dict[str, Any]:
    """Derive the comparison group an experiment row belongs to.

    Pinned experiments group by (workflow, benchmark lineage) across versions.
    Unpinned ("ad-hoc") experiments group by workflow alone, which is the
    automatically created local comparison group the epic asks for: two ad-hoc
    runs of the same workflow are the things a user would actually compare, and
    a group per experiment would make every experiment its own winner.

    The evidence store is deliberately absent from the key. Two runs of one
    lineage recorded in two different stores are one contest; if DB identity
    were part of this, every store would grow its own uncontested winner.

    Deterministic by construction — the id is derived, never minted — so two
    processes creating the first two experiments at the same moment compute the
    same group id and contend for one pointer row instead of creating two
    groups, each with its own "first" winner.
    """
    workflow_name = _clean(experiment.get("workflow_name")) or ""
    benchmark_id = _clean(experiment.get("benchmark_id")) or ""
    if benchmark_id:
        kind = GROUP_BENCHMARK
        parts = (GROUP_BENCHMARK, workflow_name, benchmark_id)
    else:
        kind = GROUP_ADHOC
        parts = (GROUP_ADHOC, workflow_name)
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]
    return {
        "group_id": f"{kind}-{digest}",
        "group_kind": kind,
        "workflow_name": workflow_name or None,
        "benchmark_id": benchmark_id or None,
    }


@dataclass(frozen=True)
class ExperimentReference:
    """An experiment named before (or without) an evidence store to read it in.

    Carries exactly the fields a contest is scoped by, plus the ordering key.
    ``created_at`` matters: it is what the historical-adoption check compares
    against, so a reference registered with no timestamp is treated as created
    now, which is the truth at UI-creation time.
    """

    experiment_id: str
    workflow_name: Optional[str] = None
    benchmark_id: Optional[str] = None
    benchmark_version: Optional[str] = None
    benchmark_digest_sha256: Optional[str] = None
    created_at: Optional[str] = None
    source_id: Optional[str] = None

    def as_row(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "workflow_name": self.workflow_name,
            "benchmark_id": self.benchmark_id,
            "benchmark_version": self.benchmark_version,
            "benchmark_digest_sha256": self.benchmark_digest_sha256,
            "created_at": self.created_at,
        }


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


def _order_key(created_at: Any, experiment_id: Any) -> tuple[str, str]:
    """The one ordering rule for "older", stated once so two callers agree."""
    return (str(created_at or ""), str(experiment_id or ""))


class SelectionControlStore:
    """Winner pointer + append-only decision history for one contest boundary.

    Thread/process-safe the same way ``ObservabilityStore`` is: every method
    opens its own short-lived connection and every write runs in a
    ``BEGIN IMMEDIATE`` transaction, so SQLite's file lock — not an in-process
    lock — is what serialises two writers. Schema creation is inside that
    transaction too, so two processes racing to create a fresh sidecar cannot
    observe each other's half-built schema.
    """

    def __init__(
        self,
        control_db_path: str,
        *,
        evidence: Optional[ObservabilityStore] = None,
        sources: Optional[SourceSpec] = None,
        mode: Optional[str] = None,
        create: bool = True,
    ) -> None:
        if mode is None:
            mode = CONTROL_MODE_SHARED if evidence is None and sources is not None else CONTROL_MODE_SINGLE
        if mode not in CONTROL_MODES:
            raise ValueError("mode must be one of " + ", ".join(sorted(CONTROL_MODES)))
        if mode == CONTROL_MODE_SINGLE and sources is not None:
            raise ValueError(
                "a single-store control sidecar reads one attached evidence "
                "store; use open_shared_control for several"
            )
        self.control_db_path = control_db_path
        self.mode = mode
        self.evidence = evidence
        self._resolver = self._normalize_sources(sources, evidence)
        self._redactor: Optional[Redactor] = None
        self._identity_cache: dict[str, str] = {}
        self._store_cache: dict[str, ObservabilityStore] = {}
        self._ensure_schema(create=create)
        self._bind_mode()
        if mode == CONTROL_MODE_SINGLE and evidence is not None:
            self._bind_single_source(evidence)

    # -- construction ----------------------------------------------------

    @staticmethod
    def _normalize_sources(
        sources: Optional[SourceSpec],
        evidence: Optional[ObservabilityStore],
    ) -> Optional[EvidenceSourceResolver]:
        if sources is None:
            if evidence is None:
                return None
            return lambda source_id: evidence if source_id == PRIMARY_SOURCE_ID else None
        if isinstance(sources, Mapping):
            mapping = dict(sources)
            return lambda source_id: mapping.get(source_id)
        if callable(sources):
            return sources
        raise TypeError("sources must be a mapping of source_id -> store, or a callable")

    @classmethod
    def for_evidence(
        cls,
        evidence: ObservabilityStore,
        *,
        control_db_path: Optional[str] = None,
        create: bool = True,
    ) -> "SelectionControlStore":
        """Attach to the sidecar of ONE evidence store (read-only stores too).

        ``control_db_path`` overrides the default sibling location, which is
        what an archive opened from a read-only directory needs: the evidence
        lives there, the decisions do not have to.

        This is the simple default and it is unchanged: one store, one sidecar,
        bound by identity. A sidecar written as shared is refused here, with the
        entry point that can read it.
        """
        path = control_db_path or control_db_path_for(evidence.db_path)
        return cls(path, evidence=evidence, mode=CONTROL_MODE_SINGLE, create=create)

    def _connect(self, timeout: float = 30.0) -> sqlite3.Connection:
        conn = sqlite3.connect(self.control_db_path, timeout=timeout)
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def _read(self) -> Iterator[sqlite3.Connection]:
        """A short-lived read connection that is actually closed.

        ``with sqlite3.connect(...)`` commits but does NOT close, so the plain
        form leaks a handle per call in a long-lived server.
        """
        conn = self._connect()
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
                conn.commit()
            except BaseException:
                conn.rollback()
                raise
        finally:
            conn.close()

    def _ensure_schema(self, *, create: bool) -> None:
        exists = (
            os.path.exists(self.control_db_path)
            and os.path.getsize(self.control_db_path) > 0
        )
        if not exists and not create:
            raise SelectionControlUnavailable(
                f"no selection control sidecar at {self.control_db_path}"
            )
        parent = os.path.dirname(self.control_db_path)
        if parent and create:
            try:
                os.makedirs(parent, exist_ok=True)
            except OSError as exc:
                raise SelectionControlUnavailable(
                    f"cannot open selection control sidecar "
                    f"{self.control_db_path}: {exc}"
                ) from exc
        # `PRAGMA journal_mode=WAL` needs a brief exclusive lock and — unlike
        # every other statement here — returns BUSY immediately instead of
        # waiting out the busy timeout. Several processes opening a fresh
        # sidecar at once is the ordinary cold start, so a contended first open
        # is retried rather than reported as an unusable control store.
        last: Optional[sqlite3.Error] = None
        for attempt in range(_SCHEMA_OPEN_ATTEMPTS):
            try:
                self._create_schema_once()
                break
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc) and "busy" not in str(exc).lower():
                    raise SelectionControlUnavailable(
                        f"cannot initialise selection control sidecar "
                        f"{self.control_db_path}: {exc}"
                    ) from exc
                last = exc
                time.sleep(_SCHEMA_OPEN_BACKOFF * (attempt + 1))
        else:
            raise SelectionControlUnavailable(
                f"selection control sidecar {self.control_db_path} stayed locked "
                f"across {_SCHEMA_OPEN_ATTEMPTS} attempts: {last}"
            )
        try:
            os.chmod(self.control_db_path, 0o600)
        except OSError:
            pass

    def _create_schema_once(self) -> None:
        try:
            conn = sqlite3.connect(self.control_db_path, timeout=30.0)
        except sqlite3.Error as exc:
            raise SelectionControlUnavailable(
                f"cannot open selection control sidecar {self.control_db_path}: {exc}"
            ) from exc
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            # The version check, the CREATEs and the version stamp are ONE
            # transaction. Split, they are a cold-start race: a second opener
            # sees user_version=0 with tables already present — which is exactly
            # the shape the fresh-only rule reads as "an older populated store"
            # — and refuses a healthy file it merely caught mid-creation.
            conn.execute("BEGIN IMMEDIATE")
            found = conn.execute("PRAGMA user_version").fetchone()[0]
            if found > CONTROL_SCHEMA_VERSION:
                raise SelectionControlUnavailable(
                    f"{self.control_db_path} has control schema v{found}; this "
                    f"build reads up to v{CONTROL_SCHEMA_VERSION}."
                )
            if found < CONTROL_SCHEMA_VERSION:
                has_tables = (
                    conn.execute(
                        "SELECT 1 FROM sqlite_master WHERE type='table' LIMIT 1"
                    ).fetchone()
                    is not None
                )
                if has_tables:
                    raise SelectionControlUnavailable(
                        f"{self.control_db_path} has control schema v{found}; "
                        f"this build requires v{CONTROL_SCHEMA_VERSION} and "
                        "carries no migration. Move the file aside to start a "
                        "new selection history."
                    )
            for statement in _CONTROL_SCHEMA_STATEMENTS:
                conn.execute(statement)
            if found < CONTROL_SCHEMA_VERSION:
                conn.execute(f"PRAGMA user_version = {CONTROL_SCHEMA_VERSION}")
            conn.commit()
        except SelectionControlUnavailable:
            conn.rollback()
            raise
        except sqlite3.OperationalError:
            # Retried by the caller when it is contention; classified there.
            conn.rollback()
            raise
        except sqlite3.Error as exc:
            conn.rollback()
            raise SelectionControlUnavailable(
                f"cannot initialise selection control sidecar "
                f"{self.control_db_path}: {exc}"
            ) from exc
        finally:
            conn.close()

    def _bind_mode(self) -> None:
        """First writer records the shape; a later disagreement is refused."""
        now = _utcnow_iso()
        try:
            with self._write() as conn:
                row = conn.execute(
                    "SELECT value FROM control_meta WHERE key=?", (_MODE_KEY,)
                ).fetchone()
                if row is None:
                    conn.execute(
                        "INSERT INTO control_meta (key, value, updated_at) "
                        "VALUES (?, ?, ?)",
                        (_MODE_KEY, self.mode, now),
                    )
                    return
                found = str(row["value"])
                if found != self.mode:
                    raise ControlModeMismatch(
                        f"{self.control_db_path} is a {found!r} selection control "
                        f"store, opened as {self.mode!r}. Open it with "
                        + (
                            "open_shared_control(...)"
                            if found == CONTROL_MODE_SHARED
                            else "SelectionControlStore.for_evidence(...)"
                        )
                    )
        except sqlite3.Error as exc:
            raise SelectionControlUnavailable(
                f"cannot read selection control sidecar {self.control_db_path}: {exc}"
            ) from exc

    # -- evidence sources ------------------------------------------------

    def authorize_source(
        self,
        source_id: str,
        evidence: ObservabilityStore,
        *,
        label: Optional[str] = None,
    ) -> dict[str, Any]:
        """Explicitly trust one evidence store under one id. Idempotent.

        The identity is taken from the store itself, never from a path or a
        caller-supplied string, and it is write-once: re-authorizing the same
        pair is a no-op, a different store under the same id is refused, and
        the same store under a second id is refused by the UNIQUE identity.
        """
        source_id = _require_text(source_id, "source_id")
        identity = evidence.store_identity()
        if not identity:
            raise SelectionControlError(
                f"evidence store {evidence.db_path!r} has no store identity; a "
                "shared control store authorizes stores by identity and will "
                "not trust one it cannot name"
            )
        now = _utcnow_iso()
        with self._write() as conn:
            row = conn.execute(
                "SELECT * FROM evidence_sources WHERE source_id=?", (source_id,)
            ).fetchone()
            if row is not None:
                if str(row["store_identity"]) != identity:
                    raise ControlStoreIdentityMismatch(
                        f"source {source_id!r} in {self.control_db_path} is bound "
                        f"to evidence store {str(row['store_identity'])!r}, not "
                        f"{identity!r}. A source id names one store for the life "
                        "of the history that references it."
                    )
                authorized = dict(row)
            else:
                clash = conn.execute(
                    "SELECT source_id FROM evidence_sources WHERE store_identity=?",
                    (identity,),
                ).fetchone()
                if clash is not None:
                    raise ControlStoreIdentityMismatch(
                        f"evidence store {identity!r} is already authorized as "
                        f"source {str(clash['source_id'])!r}; authorizing it "
                        f"again as {source_id!r} would make one store's runs "
                        "comparable with themselves."
                    )
                conn.execute(
                    """INSERT INTO evidence_sources
                       (source_id, store_identity, label, authorized_at)
                       VALUES (?, ?, ?, ?)""",
                    (source_id, identity, self._scrub(_clean(label)), now),
                )
                authorized = {
                    "source_id": source_id,
                    "store_identity": identity,
                    "label": _clean(label),
                    "authorized_at": now,
                }
        self._identity_cache[source_id] = identity
        self._store_cache[source_id] = evidence
        return authorized

    def declare_bound_source(
        self, store_identity: str, *, label: Optional[str] = None
    ) -> dict[str, Any]:
        """Admit a store this workflow ALREADY bound, without opening it today.

        The narrow case, and it is the difference between a right and a wrong
        winner: a registration records the identity of the store a runner bound
        to it, and that file is not readable right now -- unmounted, moved,
        mid-copy. `authorize_source` cannot help, because it takes the identity
        from the open store. So the contest would not know the store exists at
        all, and "the oldest experiment I can see" would quietly become "the
        oldest experiment", electing a newcomer over history nobody opened.

        The identity is not invented here either: it is the one the store
        itself reported when it was bound, written into the registration then.
        Admitting it makes the gap VISIBLE to the existing unresolved-source
        guard -- `_resolve_source` hands back None for it, `_unadopted_older
        _peers` reports it unresolved, and every registration path refuses to
        elect automatically until it can be read. That is one mechanism rather
        than a flag each caller has to remember, which matters because the
        caller that forgot it (the runner) is how the bug reached a user.

        Elections stay refused while the store is unreadable, which is the
        point; an explicit promote decision still works, so a store that is
        gone for good is a decision somebody makes rather than a deadlock.
        """
        identity = _require_text(store_identity, "store_identity")
        now = _utcnow_iso()
        with self._write() as conn:
            row = conn.execute(
                "SELECT * FROM evidence_sources WHERE source_id=?", (identity,)
            ).fetchone()
            if row is not None:
                if str(row["store_identity"]) != identity:
                    raise ControlStoreIdentityMismatch(
                        f"source {identity!r} in {self.control_db_path} is bound "
                        f"to evidence store {str(row['store_identity'])!r}"
                    )
                return dict(row)
            conn.execute(
                """INSERT INTO evidence_sources
                   (source_id, store_identity, label, authorized_at)
                   VALUES (?, ?, ?, ?)""",
                (identity, identity, self._scrub(_clean(label)), now),
            )
        return {
            "source_id": identity,
            "store_identity": identity,
            "label": _clean(label),
            "authorized_at": now,
        }

    def attach_source(
        self, source_id: str, evidence: ObservabilityStore
    ) -> ObservabilityStore:
        """Supply the open store for an ALREADY authorized source.

        The read-side counterpart of ``authorize_source``: it writes nothing
        and it cannot grant trust, so a process that merely holds a store
        handle can serve reads for it without being able to enrol it. An
        unauthorized id is refused, as is a store whose identity is not the one
        that id names.
        """
        source_id = self._require_authorized(source_id)
        authorized = self._authorized_identity(source_id)
        identity = evidence.store_identity()
        if identity != authorized:
            raise ControlStoreIdentityMismatch(
                f"source {source_id!r} is authorized for evidence store "
                f"{authorized!r}, not {identity!r}."
            )
        self._store_cache[source_id] = evidence
        return evidence

    def source_for_identity(self, store_identity: Optional[str]) -> Optional[str]:
        """Which authorized source id names this store, if any.

        How a runner finds its own place in a workspace control file without
        being told: it knows its store's identity, and authorization already
        recorded the mapping. Returns None for a store nobody authorized, which
        is a refusal, not an invitation to add one.
        """
        if not store_identity:
            return None
        with self._read() as conn:
            row = conn.execute(
                "SELECT source_id FROM evidence_sources WHERE store_identity=?",
                (store_identity,),
            ).fetchone()
        return None if row is None else str(row["source_id"])

    def _bind_single_source(self, evidence: ObservabilityStore) -> None:
        """Single-store binding: first write wins, a later mismatch is refused.

        A store with no minted identity stays unbound, as before: it is an
        older store, not a hostile one, and refusing it would make the ordinary
        local case unusable. Shared mode has no such leniency.
        """
        identity = evidence.store_identity()
        if not identity:
            return
        self.authorize_source(PRIMARY_SOURCE_ID, evidence)

    def evidence_store_identity(self) -> Optional[str]:
        """The single-store binding, or None when unbound/shared."""
        return self._authorized_identity(PRIMARY_SOURCE_ID)

    def list_sources(self) -> list[dict[str, Any]]:
        with self._read() as conn:
            rows = conn.execute(
                "SELECT * FROM evidence_sources ORDER BY authorized_at, source_id"
            ).fetchall()
        return [dict(row) for row in rows]

    def _authorized_identity(self, source_id: str) -> Optional[str]:
        cached = self._identity_cache.get(source_id)
        if cached is not None:
            return cached
        with self._read() as conn:
            row = conn.execute(
                "SELECT store_identity FROM evidence_sources WHERE source_id=?",
                (source_id,),
            ).fetchone()
        if row is None:
            return None
        identity = str(row["store_identity"])
        self._identity_cache[source_id] = identity
        return identity

    def _require_authorized(self, source_id: str) -> str:
        source_id = _require_text(source_id, "source_id")
        if self._authorized_identity(source_id) is None:
            raise UnauthorizedEvidenceSource(source_id)
        return source_id

    def _resolve_source(self, source_id: Optional[str]) -> Optional[ObservabilityStore]:
        """The store behind a source id, with its identity re-checked.

        Re-checking on resolve is not paranoia about the embedder: a resolver
        returns whatever handle it was configured with, and a config error that
        pointed one source at another store would otherwise show one store's
        experiment under another's winner.

        Returns None when nothing was supplied for an authorized source; raises
        only when the source is not authorized at all, or when the store handed
        back is not the one that id names.
        """
        if source_id is None:
            # An unbound registration in single mode still reads its one store.
            return self.evidence if self.mode == CONTROL_MODE_SINGLE else None
        cached = self._store_cache.get(source_id)
        if cached is not None:
            return cached
        authorized = self._authorized_identity(source_id)
        if authorized is None:
            raise UnauthorizedEvidenceSource(source_id)
        store = self._resolver(source_id) if self._resolver is not None else None
        if store is None:
            return None
        identity = store.store_identity()
        if identity != authorized:
            raise ControlStoreIdentityMismatch(
                f"the store supplied for source {source_id!r} has identity "
                f"{identity!r}, but {source_id!r} is authorized for "
                f"{authorized!r}."
            )
        self._store_cache[source_id] = store
        return store

    def _source_resolution(
        self,
    ) -> tuple[list[tuple[str, ObservabilityStore]], list[str]]:
        """Split the authorized sources into the ones we can read and the rest.

        The second list is the load-bearing one. An authorized source nobody
        handed back a handle for is a KNOWN store of unknown content: it may
        hold the oldest runs of this very lineage. Treating it as empty is the
        silent wrong answer, so callers that are about to elect a first winner
        ask for it and refuse instead.
        """
        resolved: list[tuple[str, ObservabilityStore]] = []
        unresolved: list[str] = []
        for row in self.list_sources():
            source_id = str(row["source_id"])
            store = self._resolve_source(source_id)
            if store is None:
                unresolved.append(source_id)
            else:
                resolved.append((source_id, store))
        if not resolved and self.mode == CONTROL_MODE_SINGLE and self.evidence is not None:
            # An unbound single store (no minted identity) still reads.
            resolved.append((PRIMARY_SOURCE_ID, self.evidence))
        return resolved, sorted(unresolved)

    def _resolvable_sources(self) -> list[tuple[str, ObservabilityStore]]:
        """Every authorized source the embedder actually handed us a store for."""
        return self._source_resolution()[0]

    def _scrub(self, value: Optional[str]) -> Optional[str]:
        if not value:
            return value
        if self._redactor is None:
            self._redactor = Redactor()
        return self._redactor.redact(value)

    # -- registration ----------------------------------------------------

    def register_experiment(
        self,
        experiment_id: str,
        *,
        experiment: Optional[Mapping[str, Any]] = None,
        source_id: Optional[str] = None,
        allow_initial_winner: bool = True,
        actor: str = "fastworkflow",
        actor_kind: str = SYSTEM_ACTOR_KIND,
        provenance: str = _AUTOMATIC_PROVENANCE,
    ) -> dict[str, Any]:
        """Record an experiment in its group; the first one wins automatically.

        Idempotent, because ``create_experiment`` is re-run on every resume: a
        second call adds no member row and no decision, and cannot re-run the
        automatic initialization against a group whose winner has since moved.

        The initial winner is recorded while the experiment is still RUNNING and
        is never annotated as successful. ``current_winner`` reports the live
        status, so a first experiment that fails stays visibly failed until
        somebody explicitly replaces it.

        The automatic initialization is REFUSED — registration still happens,
        the pointer does not — when the group has no winner yet but older
        experiments of the same lineage already exist unadopted. See
        ``adopt_existing_experiments`` for the bootstrap contract.
        """
        if source_id is None and self.mode == CONTROL_MODE_SINGLE:
            source_id = PRIMARY_SOURCE_ID if self.evidence_store_identity() else None
        if source_id is not None:
            source_id = self._require_authorized(source_id)
        elif self.mode == CONTROL_MODE_SHARED:
            raise ValueError(
                "source_id is required when registering into a shared control "
                "store; use register_experiment_reference for an experiment "
                "that has no evidence store yet"
            )
        row = experiment
        if row is None:
            row = self._experiment_row(experiment_id, source_id)
        if row is None:
            raise ExperimentNotFound(experiment_id)
        return self._register(
            experiment_id,
            row,
            source_id=source_id,
            allow_initial_winner=allow_initial_winner,
            actor=actor,
            actor_kind=actor_kind,
            provenance=provenance,
        )

    def register_experiment_reference(
        self,
        reference: ExperimentReference,
        *,
        allow_initial_winner: bool = True,
        actor: str = "fastworkflow",
        actor_kind: str = SYSTEM_ACTOR_KIND,
        provenance: str = _AUTOMATIC_PROVENANCE,
    ) -> dict[str, Any]:
        """Register an experiment that has no evidence store bound yet.

        This is the UI-creation path: an experiment exists as a decision to run
        something long before a runner opens a store for it, and the first
        winner of a brand-new group should be initialized then, not on first
        execution. The reference carries the lineage fields the contest is
        scoped by; ``bind_experiment_source`` attaches the store afterwards and
        moves no pointer.
        """
        experiment_id = _require_text(reference.experiment_id, "experiment_id")
        source_id = reference.source_id
        if source_id is not None:
            source_id = self._require_authorized(source_id)
        return self._register(
            experiment_id,
            reference.as_row(),
            source_id=source_id,
            allow_initial_winner=allow_initial_winner,
            actor=actor,
            actor_kind=actor_kind,
            provenance=provenance,
        )

    def bind_experiment_source(
        self, experiment_id: str, source_id: str
    ) -> dict[str, Any]:
        """Attach the evidence store of an already-registered experiment.

        Explicit, write-once, and selection-neutral: it records where the
        evidence lives so reads can resolve it, and touches no pointer and no
        history. A second binding to a different store is a collision, not an
        update — the winner pointer may already name this experiment.
        """
        source_id = self._require_authorized(source_id)
        with self._write() as conn:
            row = conn.execute(
                "SELECT * FROM comparison_group_members WHERE experiment_id=?",
                (experiment_id,),
            ).fetchone()
            if row is None:
                raise ExperimentNotFound(experiment_id)
            bound = _clean(row["source_id"])
            if bound is not None and bound != source_id:
                raise ExperimentSourceCollision(experiment_id, bound, source_id)
            if bound is None:
                conn.execute(
                    "UPDATE comparison_group_members SET source_id=? "
                    "WHERE experiment_id=?",
                    (source_id, experiment_id),
                )
            member = dict(row)
        member["source_id"] = source_id
        return member

    def _register(
        self,
        experiment_id: str,
        row: Mapping[str, Any],
        *,
        source_id: Optional[str],
        allow_initial_winner: bool,
        actor: str,
        actor_kind: str,
        provenance: str,
    ) -> dict[str, Any]:
        experiment_id = _require_text(experiment_id, "experiment_id")
        identity = comparison_group_identity(row)
        group_id = identity["group_id"]
        now = _utcnow_iso()
        created_at = _clean(row.get("created_at")) or now
        actor = _require_text(actor, "actor")
        provenance = _require_text(provenance, "provenance")
        if actor_kind not in SELECTION_ACTOR_KINDS:
            raise ValueError(
                "actor_kind must be one of " + ", ".join(sorted(SELECTION_ACTOR_KINDS))
            )

        # Read evidence BEFORE taking the write lock: the scan can touch several
        # stores, and holding the control file's write lock across them would
        # serialise every unrelated registration behind the slowest reader. The
        # transaction below re-checks the pointer, which is what actually
        # decides, so a group initialized while we scanned is seen as initialized.
        # The scan asks about the group the transaction will actually use: an
        # existing member stays in the group it first joined.
        scan_group_id = self._member_group_id(experiment_id) or group_id
        peers: list[str] = []
        scanned: list[str] = []
        unresolved: list[str] = []
        looked = False
        if allow_initial_winner and self._pointer_missing(scan_group_id):
            looked = True
            peers, scanned, unresolved = self._unadopted_older_peers(
                scan_group_id, experiment_id, created_at
            )

        registered = False
        initialized = False
        bootstrap_required = False
        rescan = False
        with self._write() as conn:
            member = conn.execute(
                """SELECT group_id, source_id FROM comparison_group_members
                    WHERE experiment_id=?""",
                (experiment_id,),
            ).fetchone()
            if member is None:
                # The group row is created only when this registration is
                # actually going to join it. Creating it first would leave an
                # empty group behind every time a re-registration derived a
                # different id and then kept its sticky membership — a contest
                # with no members, visible in `list_groups`.
                conn.execute(
                    """INSERT INTO comparison_groups
                       (group_id, group_kind, workflow_name, benchmark_id,
                        created_at, created_from_experiment_id)
                       VALUES (?, ?, ?, ?, ?, ?)
                       ON CONFLICT(group_id) DO NOTHING""",
                    (
                        group_id,
                        identity["group_kind"],
                        identity["workflow_name"],
                        identity["benchmark_id"],
                        now,
                        experiment_id,
                    ),
                )
                conn.execute(
                    """INSERT INTO comparison_group_members
                       (group_id, experiment_id, source_id, benchmark_version,
                        benchmark_digest_sha256, created_at, joined_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        group_id,
                        experiment_id,
                        source_id,
                        _clean(row.get("benchmark_version")),
                        _clean(row.get("benchmark_digest_sha256")),
                        created_at,
                        now,
                    ),
                )
                registered = True
            else:
                bound = _clean(member["source_id"])
                if source_id is not None and bound is not None and bound != source_id:
                    raise ExperimentSourceCollision(experiment_id, bound, source_id)
                if source_id is not None and bound is None:
                    conn.execute(
                        "UPDATE comparison_group_members SET source_id=? "
                        "WHERE experiment_id=?",
                        (source_id, experiment_id),
                    )
                # Membership is sticky. An experiment created unpinned and
                # re-created with a benchmark pin would derive a different
                # group; moving it would orphan a pointer that already
                # names it, so the first group it joined keeps it.
                group_id = str(member["group_id"])
            pointer = conn.execute(
                """SELECT selection_id FROM selection_pointers
                    WHERE scope_kind=? AND group_id=? AND scope_key=?""",
                (EXPERIMENT_SCOPE, group_id, ""),
            ).fetchone()
            if pointer is None and allow_initial_winner:
                if peers or unresolved:
                    bootstrap_required = True
                elif not looked:
                    # The pointer was there when the scan was skipped and is
                    # gone now: a sole winner was deleted in between
                    # (`retire_experiment(allow_sole_winner=True)`). Nothing
                    # was checked for unadopted history, so this transaction
                    # elects nobody and the registration runs once more, as a
                    # member now, with the scan.
                    rescan = True
                else:
                    # The OLDEST member takes the pointer, not the caller. A
                    # group can hold members before it has a winner -- a
                    # bootstrap that could not read every source registers
                    # them and elects nobody -- and the registration that
                    # finally elects is whichever arrives first once the view
                    # is whole, often a newer one than the history already
                    # sitting in the group (`fix-kkod`). Same rule as
                    # `adopt_existing_experiments`: `_order_key`, earliest wins.
                    elected = min(
                        conn.execute(
                            """SELECT experiment_id, created_at
                                 FROM comparison_group_members WHERE group_id=?""",
                            (group_id,),
                        ).fetchall(),
                        key=lambda m: _order_key(m["created_at"], m["experiment_id"]),
                    )
                    elected_id = str(elected["experiment_id"])
                    selection_id = uuid.uuid4().hex
                    # NOT a hardcoded 1. A group can have history BEFORE it has
                    # a winner: a bootstrap that could not see every source
                    # registers members and elects nobody, and a member deleted
                    # in the meantime appends `retire` at seq 1 (`fix-jfy5`).
                    # The election that finally arrives is the next sequence
                    # number; reusing 1 violated the UNIQUE (scope, group,
                    # scope_key, seq) and lost the election to a swallowed
                    # IntegrityError, leaving the group winner-less for good.
                    seq = int(
                        conn.execute(
                            """SELECT COALESCE(MAX(seq), 0) FROM selection_decisions
                                WHERE scope_kind=? AND group_id=? AND scope_key=?""",
                            (EXPERIMENT_SCOPE, group_id, ""),
                        ).fetchone()[0]
                    ) + 1
                    conn.execute(
                        """INSERT INTO selection_pointers
                           (scope_kind, group_id, scope_key, experiment_id,
                            task_id, attempt, selection_id, decision,
                            decision_seq, decided_at)
                           VALUES (?, ?, ?, ?, NULL, NULL, ?, ?, ?, ?)""",
                        (
                            EXPERIMENT_SCOPE,
                            group_id,
                            "",
                            elected_id,
                            selection_id,
                            DECISION_INITIAL,
                            seq,
                            now,
                        ),
                    )
                    self._append_decision_in_txn(
                        conn,
                        group_id=group_id,
                        seq=seq,
                        decision=DECISION_INITIAL,
                        previous_experiment_id=None,
                        previous_selection_id=None,
                        candidate_experiment_id=elected_id,
                        new_experiment_id=elected_id,
                        new_selection_id=selection_id,
                        actor=actor,
                        actor_kind=actor_kind,
                        provenance=provenance,
                        rationale=None,
                        created_at=now,
                    )
                    initialized = True
        if rescan:
            again = self._register(
                experiment_id,
                row,
                source_id=source_id,
                allow_initial_winner=allow_initial_winner,
                actor=actor,
                actor_kind=actor_kind,
                provenance=provenance,
            )
            again["registered"] = again["registered"] or registered
            return again
        if bootstrap_required:
            reason = (
                f"{len(peers)} older experiment(s) of the same lineage are not "
                "adopted yet"
                if peers
                else f"authorized evidence source(s) {', '.join(unresolved)} "
                "could not be read, so whether older runs of this lineage exist "
                "is unknown"
            )
            logger.warning(
                f"observability: experiment {experiment_id!r} was registered in "
                f"comparison group {group_id!r} but did NOT become its initial "
                f"winner: {reason}. Bootstrap this group from an embedder that "
                "can resolve every authorized source."
            )
        return {
            "group_id": group_id,
            "experiment_id": experiment_id,
            "source_id": source_id,
            "registered": registered,
            "initialized": initialized,
            "bootstrap_required": bootstrap_required,
            "unadopted_experiment_ids": peers,
            "unresolved_sources": unresolved,
            "evidence_scanned": bool(scanned),
            "sources_scanned": scanned,
            "winner": self.current_winner(group_id),
        }

    def _pointer_missing(self, group_id: str) -> bool:
        with self._read() as conn:
            row = conn.execute(
                """SELECT 1 FROM selection_pointers
                    WHERE scope_kind=? AND group_id=? AND scope_key=?""",
                (EXPERIMENT_SCOPE, group_id, ""),
            ).fetchone()
        return row is None

    def _unadopted_older_peers(
        self, group_id: str, experiment_id: str, created_at: str
    ) -> tuple[list[str], list[str], list[str]]:
        """Older same-lineage experiments this sidecar has never seen.

        The case this exists for: a store (or a workspace of stores) that was
        already full of runs when selection was switched on. The next run
        created is not "the first experiment" of that lineage, and letting it
        take the initial pointer would file a judgement about history nobody
        made. Returns the ids plus WHICH sources were actually read — an empty
        list from an unread source is not evidence of absence, and a runner
        process typically holds only its own store. A workspace that predates
        selection is therefore bootstrapped once, by an embedder that can
        resolve every source (`adopt_existing_experiments` reports the ones it
        could not), not by whichever runner happens to start first.

        Why this does not strand a group of experiments that all start at once
        (parallel runners, or the resume of a batch): the scan asks only about
        NON-MEMBERS older than the caller, and the OLDEST experiment of a group
        has none by definition. So whichever sibling registers first, the oldest
        one's own registration always elects a winner — no sibling can defer
        forever. A pre-existing history behaves differently for the right
        reason: those experiments finished long ago and nobody re-registers
        them, so the group correctly stays unbootstrapped until an embedder
        calls `adopt_existing_experiments`. Distinguishing the two by a time
        window would need a magic constant; distinguishing them by "does anyone
        still register this?" needs none.

        The third return value is the authorized sources this process could not
        open. In a SHARED control that is a hard stop for automatic election,
        not a footnote: a runner holds its own store and nothing else, so
        "I found no older runs" from such a process means only "I did not
        look". Electing there is precisely how a new candidate silently
        outranks an older experiment sitting in a store nobody opened. Single
        mode has exactly one store — the caller's own — so nothing is withheld
        and the ordinary local case is unchanged.
        """
        sources, unresolved = self._source_resolution()
        if self.mode != CONTROL_MODE_SHARED:
            unresolved = []
        if not sources:
            return [], [], unresolved
        # Members are skipped here because they are already in the contest,
        # not because they are newer: the election in `_register` picks the
        # oldest MEMBER, so an older member is elected rather than outranked.
        members = self._member_ids()
        mine = _order_key(created_at, experiment_id)
        peers: list[str] = []
        for _source_id, store in sources:
            for row in self._iter_experiments(store):
                other_id = str(row.get("experiment_id") or "")
                if not other_id or other_id == experiment_id or other_id in members:
                    continue
                if comparison_group_identity(row)["group_id"] != group_id:
                    continue
                if _order_key(row.get("created_at"), other_id) < mine:
                    peers.append(other_id)
        peers.sort()
        return (
            peers[:_MAX_REPORTED_PEERS],
            sorted(s for s, _ in sources),
            unresolved,
        )

    @staticmethod
    def _iter_experiments(
        store: ObservabilityStore, *, page: int = 200
    ) -> Iterator[Mapping[str, Any]]:
        offset = 0
        while True:
            batch = store.list_experiments(limit=page, offset=offset)
            if not batch:
                return
            yield from batch
            offset += page

    def _member_ids(self) -> set[str]:
        with self._read() as conn:
            rows = conn.execute(
                "SELECT experiment_id FROM comparison_group_members"
            ).fetchall()
        return {str(row["experiment_id"]) for row in rows}

    def adopt_existing_experiments(
        self, *, page: int = 200, allow_initial_winner: bool = True
    ) -> dict[str, Any]:
        """The bootstrap contract: register what already exists, in order.

        The deterministic rule for an imported group, stated once so two
        importers agree: members are considered in ascending
        ``(created_at, experiment_id)`` order, and the EARLIEST is the group's
        initial winner. Ties on `created_at` — two experiments created in the
        same second — fall to the lexicographically smaller id. Already-tracked
        experiments and groups whose winner has moved are left alone.

        In a shared control store this runs across EVERY resolvable authorized
        source and orders them together, so a lineage split over two stores
        elects one winner rather than one per store. Sources the embedder did
        not hand back a handle for are skipped and reported, because adopting a
        partial view would make exactly the silent wrong answer this contract
        exists to prevent.

        ``allow_initial_winner=False`` is how a caller reports a gap this
        method CANNOT see. The skipped-source check above only covers sources
        that were authorized; a store whose file has gone missing was never
        authorized at all, so from here it is indistinguishable from a store
        that never existed, and the oldest experiment still readable would be
        elected over older runs sitting in it. The caller that knows the
        registration naming it (`benchmark.setup._seed_registrations`) says so
        here, and then this registers members and elects nobody.
        """
        sources, skipped = self._source_resolution()
        if not sources:
            raise SelectionControlError(
                "adopt_existing_experiments needs at least one resolvable "
                "evidence source to read"
            )
        resolvable = {source_id for source_id, _ in sources}
        rows: list[tuple[str, Mapping[str, Any]]] = []
        for source_id, store in sources:
            offset = 0
            while True:
                batch = store.list_experiments(limit=page, offset=offset)
                if not batch:
                    break
                rows.extend((source_id, row) for row in batch)
                offset += page
        rows.sort(key=lambda item: _order_key(item[1].get("created_at"), item[1]["experiment_id"]))
        adopted = 0
        for source_id, row in rows:
            result = self._register(
                str(row["experiment_id"]),
                row,
                source_id=source_id if self._authorized_identity(source_id) else None,
                allow_initial_winner=allow_initial_winner,
                actor="fastworkflow",
                actor_kind=SYSTEM_ACTOR_KIND,
                provenance=_AUTOMATIC_PROVENANCE,
            )
            if result["registered"]:
                adopted += 1
        return {
            "experiments_seen": len(rows),
            "experiments_adopted": adopted,
            "sources_read": sorted(resolvable),
            "sources_skipped": sorted(skipped),
            # A partial adoption registers members but elects nobody (see
            # `_unadopted_older_peers`). `complete` is what an embedder retries
            # on, rather than inferring it from an empty skip list.
            "complete": bool(not skipped and allow_initial_winner),
            "groups_without_winner": [
                str(row["group_id"])
                for row in self.list_groups()
                if self._pointer_missing(str(row["group_id"]))
            ],
        }

    def retire_experiment(
        self,
        experiment_id: str,
        *,
        actor: str = "fastworkflow",
        actor_kind: str = SYSTEM_ACTOR_KIND,
        provenance: str = _RETIREMENT_PROVENANCE,
        rationale: Optional[str] = None,
        allow_sole_winner: bool = False,
    ) -> dict[str, Any]:
        """Withdraw a registration that is being deleted (`fix-jfy5`).

        The bug this closes: registration and election happen together at
        creation, deletion only tombstoned the registration, so a group could
        be left naming a winner nobody could open -- unresolvable, listed as a
        member, and impossible to duplicate, which is the one action the winner
        screen offers.

        The policy is deliberately small. A member that is NOT the current
        winner leaves, membership row and all, in ONE transaction that also
        appends the `retire` row. The current winner does not leave while the
        group has anybody else in it:

        - Clearing the pointer and stopping would leave the group with no
          winner while members remain, which is a judgement about the contest
          that nobody made.
        - Clearing it and electing a successor would be this module deciding
          the contest while somebody was deleting something else.

        So it refuses, and the refusal is the whole feature: the caller selects
        a different winner first and then deletes. That is one user action
        more, in exchange for a pointer that can never name a tombstone. The
        refusal covers the AUTOMATIC first winner too -- it is still the
        experiment every read currently reports, and "nobody chose it" is not
        the same as "anybody may remove it".

        The one exception is ``allow_sole_winner=True`` with the winner as the
        group's ONLY member (`fix-65ik`). There is nobody to select instead, so
        the refusal had no way out: promoting a second experiment and deleting
        the first only moved the problem onto the second. Neither objection
        above applies -- a contest with no entrants has no verdict to withhold
        and no successor to pick -- so the member row AND the pointer go, and
        the `retire` row records the winner it removed (`new_experiment_id`
        empty). The next experiment of that lineage is elected automatically,
        like the first one was. Any other member present means refusal as
        before; that is checked inside the same transaction, so a registration
        that joins first turns this into the ordinary refusal.

        The check and the removal are one `BEGIN IMMEDIATE` transaction, which
        is what makes the interesting race safe in both directions: a promotion
        that lands first makes this refuse, and a retirement that lands first
        makes the promotion fail its membership check. Neither order can leave
        the pointer naming a withdrawn experiment.

        An experiment with evidence is refused as well. Retirement is for a
        registration nobody ran; a recorded run stays in the contest it was
        part of.

        History is append-only throughout: a `retire` row is added naming what
        left and the pointer before and after (the same pointer, except in the
        sole-winner case), and nothing is rewritten. An experiment nobody
        registered here is not an error -- retirement is idempotent, so a
        deletion path can call it without first asking whether it applies.

        The GROUP row stays even when its last member leaves, because the
        decisions made in it name it and an append-only history must not end up
        referring to a contest that no longer exists. A group with no members
        and no pointer reads as "nobody has won this yet", which is what it is;
        the next experiment of that lineage rejoins it. Two paths produce one:
        the last NON-winner leaving a group that never elected (a bootstrap
        that could not read every source registers members and elects nobody),
        and the sole winner leaving under ``allow_sole_winner``.
        """
        experiment_id = _require_text(experiment_id, "experiment_id")
        actor = _require_text(actor, "actor")
        provenance = _require_text(provenance, "provenance")
        if actor_kind not in SELECTION_ACTOR_KINDS:
            raise ValueError(
                "actor_kind must be one of " + ", ".join(sorted(SELECTION_ACTOR_KINDS))
            )
        rationale = _clean(rationale)
        if rationale is not None and len(rationale) > _MAX_TEXT:
            raise ValueError(f"rationale must be at most {_MAX_TEXT} characters")

        # Read evidence BEFORE taking the write lock, the way `_register` does:
        # resolving a store can be slow, and the transaction below re-checks the
        # binding, which is the check that actually decides.
        source_id = self._member_source_id(experiment_id)
        if source_id is None and self._experiment_state(experiment_id, None) is not None:
            raise SelectionRetirementRefused(
                experiment_id,
                self._member_group_id(experiment_id),
                "has_evidence",
                f"experiment {experiment_id!r} has recorded evidence and cannot "
                "be withdrawn from its comparison group",
            )

        now = _utcnow_iso()
        with self._write() as conn:
            member = conn.execute(
                "SELECT * FROM comparison_group_members WHERE experiment_id=?",
                (experiment_id,),
            ).fetchone()
            if member is None:
                return {
                    "retired": False,
                    "experiment_id": experiment_id,
                    "group_id": None,
                    "seq": None,
                }
            group_id = str(member["group_id"])
            if _clean(member["source_id"]) is not None:
                raise SelectionRetirementRefused(
                    experiment_id,
                    group_id,
                    "has_evidence",
                    f"experiment {experiment_id!r} is bound to evidence source "
                    f"{str(member['source_id'])!r} and cannot be withdrawn from "
                    "its comparison group",
                )
            pointer = conn.execute(
                """SELECT * FROM selection_pointers
                    WHERE scope_kind=? AND group_id=? AND scope_key=?""",
                (EXPERIMENT_SCOPE, group_id, ""),
            ).fetchone()
            winner_retired = False
            if pointer is not None and str(pointer["experiment_id"]) == experiment_id:
                others = conn.execute(
                    """SELECT 1 FROM comparison_group_members
                        WHERE group_id=? AND experiment_id<>? LIMIT 1""",
                    (group_id, experiment_id),
                ).fetchone()
                if not allow_sole_winner or others is not None:
                    raise SelectionRetirementRefused(
                        experiment_id,
                        group_id,
                        "is_current_winner",
                        f"experiment {experiment_id!r} is the current winner of "
                        f"comparison group {group_id!r}; select a different winner "
                        "before withdrawing it",
                    )
                winner_retired = True
            seq = int(
                conn.execute(
                    """SELECT COALESCE(MAX(seq), 0) FROM selection_decisions
                        WHERE scope_kind=? AND group_id=? AND scope_key=?""",
                    (EXPERIMENT_SCOPE, group_id, ""),
                ).fetchone()[0]
            ) + 1
            conn.execute(
                "DELETE FROM comparison_group_members WHERE group_id=? AND experiment_id=?",
                (group_id, experiment_id),
            )
            if winner_retired:
                conn.execute(
                    """DELETE FROM selection_pointers
                        WHERE scope_kind=? AND group_id=? AND scope_key=?""",
                    (EXPERIMENT_SCOPE, group_id, ""),
                )
            current_experiment_id = (
                None if pointer is None else str(pointer["experiment_id"])
            )
            current_selection_id = (
                None if pointer is None else str(pointer["selection_id"])
            )
            self._append_decision_in_txn(
                conn,
                group_id=group_id,
                seq=seq,
                decision=DECISION_RETIRE,
                previous_experiment_id=current_experiment_id,
                previous_selection_id=current_selection_id,
                candidate_experiment_id=experiment_id,
                # The winner is the same before and after, which is the point --
                # unless the sole winner itself left, and then there is none.
                new_experiment_id=None if winner_retired else current_experiment_id,
                new_selection_id=None if winner_retired else current_selection_id,
                actor=actor,
                actor_kind=actor_kind,
                provenance=provenance,
                rationale=rationale,
                created_at=now,
            )
        return {
            "retired": True,
            "experiment_id": experiment_id,
            "group_id": group_id,
            "seq": seq,
            "winner_retired": winner_retired,
            "winner": self.current_winner(group_id),
        }

    # -- decisions -------------------------------------------------------

    def record_decision(
        self,
        group_id: str,
        decision: str,
        *,
        expected_selection_id: str,
        actor: str,
        actor_kind: str,
        provenance: str,
        candidate_experiment_id: Optional[str] = None,
        rationale: Optional[str] = None,
    ) -> dict[str, Any]:
        """Promote a candidate, keep the current winner, or stay undecided.

        Every decision appends to history; ONLY ``promote`` moves the pointer.
        ``expected_selection_id`` is required for all three: a "keep" recorded
        against a winner that was replaced while the reviewer was reading is a
        judgement about a different experiment than the one it will be filed
        under, which is the same wrong answer a stale promotion gives.

        ``candidate_experiment_id`` is REQUIRED for ``promote`` and OPTIONAL for
        ``keep``/``undecided``, where it names the experiment that was being
        reviewed when the decision was made. Without it the history can say a
        winner was kept but not which challenger was rejected or deferred, which
        is most of what a reviewer wants back. It is validated for membership
        and round-tripped verbatim; on keep/undecided it moves nothing. A keep
        naming the current winner is allowed and means "re-reviewed, still it".
        """
        if decision not in CLIENT_DECISIONS:
            raise ValueError(
                "decision must be one of " + ", ".join(sorted(CLIENT_DECISIONS))
            )
        if actor_kind not in SELECTION_ACTOR_KINDS:
            raise ValueError(
                "actor_kind must be one of " + ", ".join(sorted(SELECTION_ACTOR_KINDS))
            )
        actor = _require_text(actor, "actor")
        provenance = _require_text(provenance, "provenance")
        expected_selection_id = _require_text(
            expected_selection_id, "expected_selection_id"
        )
        rationale = _clean(rationale)
        if rationale is not None and len(rationale) > _MAX_TEXT:
            raise ValueError(f"rationale must be at most {_MAX_TEXT} characters")
        if decision == DECISION_PROMOTE:
            candidate_experiment_id = _require_text(
                candidate_experiment_id, "candidate_experiment_id"
            )
        else:
            candidate_experiment_id = _clean(candidate_experiment_id)
        now = _utcnow_iso()
        with self._write() as conn:
            if conn.execute(
                "SELECT 1 FROM comparison_groups WHERE group_id=?", (group_id,)
            ).fetchone() is None:
                raise UnknownComparisonGroup(group_id)
            pointer = conn.execute(
                """SELECT * FROM selection_pointers
                    WHERE scope_kind=? AND group_id=? AND scope_key=?""",
                (EXPERIMENT_SCOPE, group_id, ""),
            ).fetchone()
            if pointer is None:
                raise NoCurrentSelection(
                    f"comparison group {group_id!r} has no current winner yet, "
                    "so there is nothing to promote over, keep or defer. A "
                    "winner is elected automatically -- the earliest "
                    "experiment of the group -- only once this workflow's "
                    "recorded history can be read in full; until then setup "
                    "registers experiments without electing one, so an older "
                    "run sitting in an unreadable evidence store cannot be "
                    "outranked by a newer one. Make the missing evidence "
                    "store readable again; the next experiment created, "
                    "duplicated or started retries the election."
                )
            current_selection_id = str(pointer["selection_id"])
            current_experiment_id = str(pointer["experiment_id"])
            if current_selection_id != expected_selection_id:
                raise StaleSelection(
                    group_id=group_id,
                    expected_selection_id=expected_selection_id,
                    current_selection_id=current_selection_id,
                    current_experiment_id=current_experiment_id,
                )
            if candidate_experiment_id is not None:
                if conn.execute(
                    """SELECT 1 FROM comparison_group_members
                        WHERE group_id=? AND experiment_id=?""",
                    (group_id, candidate_experiment_id),
                ).fetchone() is None:
                    raise ExperimentNotInGroup(candidate_experiment_id, group_id)
                if (
                    decision == DECISION_PROMOTE
                    and candidate_experiment_id == current_experiment_id
                ):
                    raise ValueError(
                        f"experiment {candidate_experiment_id!r} is already the "
                        f"current winner; record a {DECISION_KEEP!r} decision "
                        "instead of promoting it again"
                    )
            seq = int(
                conn.execute(
                    """SELECT COALESCE(MAX(seq), 0) FROM selection_decisions
                        WHERE scope_kind=? AND group_id=? AND scope_key=?""",
                    (EXPERIMENT_SCOPE, group_id, ""),
                ).fetchone()[0]
            ) + 1
            if decision == DECISION_PROMOTE:
                new_experiment_id = candidate_experiment_id
                new_selection_id = uuid.uuid4().hex
                # Compare-and-set on the identity the caller read. The WHERE
                # clause is the refusal: a promotion that lost the race
                # updates nothing rather than overwriting the winner it
                # never saw.
                updated = conn.execute(
                    """UPDATE selection_pointers
                          SET experiment_id=?, selection_id=?, decision=?,
                              decision_seq=?, decided_at=?
                        WHERE scope_kind=? AND group_id=? AND scope_key=?
                          AND selection_id=?""",
                    (
                        new_experiment_id,
                        new_selection_id,
                        DECISION_PROMOTE,
                        seq,
                        now,
                        EXPERIMENT_SCOPE,
                        group_id,
                        "",
                        expected_selection_id,
                    ),
                ).rowcount
                if updated != 1:
                    raise StaleSelection(
                        group_id=group_id,
                        expected_selection_id=expected_selection_id,
                        current_selection_id=current_selection_id,
                        current_experiment_id=current_experiment_id,
                    )
            else:
                new_experiment_id = current_experiment_id
                new_selection_id = current_selection_id
            self._append_decision_in_txn(
                conn,
                group_id=group_id,
                seq=seq,
                decision=decision,
                previous_experiment_id=current_experiment_id,
                previous_selection_id=current_selection_id,
                candidate_experiment_id=candidate_experiment_id,
                new_experiment_id=new_experiment_id,
                new_selection_id=new_selection_id,
                actor=actor,
                actor_kind=actor_kind,
                provenance=provenance,
                rationale=rationale,
                created_at=now,
            )
        state = self.current_winner(group_id)
        state["decision"] = {
            "decision": decision,
            "seq": seq,
            "previous_experiment_id": current_experiment_id,
            "candidate_experiment_id": candidate_experiment_id,
            "new_experiment_id": new_experiment_id,
            "actor": actor,
            "actor_kind": actor_kind,
            "provenance": provenance,
            "rationale": rationale,
            "created_at": now,
        }
        return state

    def _append_decision_in_txn(
        self,
        conn: sqlite3.Connection,
        *,
        group_id: str,
        seq: int,
        decision: str,
        previous_experiment_id: Optional[str],
        previous_selection_id: Optional[str],
        candidate_experiment_id: Optional[str],
        new_experiment_id: Optional[str],
        new_selection_id: Optional[str],
        actor: str,
        actor_kind: str,
        provenance: str,
        rationale: Optional[str],
        created_at: str,
        scope_kind: str = EXPERIMENT_SCOPE,
        scope_key: str = "",
        previous_task_id: Optional[str] = None,
        previous_attempt: Optional[int] = None,
        candidate_task_id: Optional[str] = None,
        candidate_attempt: Optional[int] = None,
        new_task_id: Optional[str] = None,
        new_attempt: Optional[int] = None,
    ) -> None:
        conn.execute(
            """INSERT INTO selection_decisions
               (scope_kind, group_id, scope_key, seq, decision,
                previous_experiment_id, previous_task_id, previous_attempt,
                previous_selection_id, candidate_experiment_id,
                candidate_task_id, candidate_attempt, new_experiment_id,
                new_task_id, new_attempt, new_selection_id, actor, actor_kind,
                provenance, rationale, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                       ?, ?)""",
            (
                scope_kind,
                group_id,
                scope_key,
                seq,
                decision,
                previous_experiment_id,
                previous_task_id,
                previous_attempt,
                previous_selection_id,
                candidate_experiment_id,
                candidate_task_id,
                candidate_attempt,
                new_experiment_id,
                new_task_id,
                new_attempt,
                new_selection_id,
                self._scrub(actor),
                actor_kind,
                self._scrub(provenance),
                self._scrub(rationale),
                created_at,
            ),
        )

    # -- scoped selections other than the experiment winner ---------------
    #
    # The storage primitive `best_run.py` (`fix-9eg.17.4`) is built on. It is
    # deliberately a primitive: what a valid candidate is, and what the
    # decision words mean, belong to the scope that owns them, not here. What
    # DOES belong here is the guarantee that no other scope can touch the
    # winner pointer, which is why `EXPERIMENT_SCOPE` and an empty `scope_key`
    # are both refused below — the winner moves only through `record_decision`.

    def _require_other_scope(self, scope_kind: str, scope_key: str) -> tuple[str, str]:
        scope_kind = _require_text(scope_kind, "scope_kind")
        scope_key = _require_text(scope_key, "scope_key")
        if scope_kind == EXPERIMENT_SCOPE:
            raise ValueError(
                "the experiment winner is moved only by record_decision; a "
                "scoped decision must name a different scope_kind"
            )
        return scope_kind, scope_key

    def scoped_pointer(
        self, *, scope_kind: str, group_id: str, scope_key: str
    ) -> Optional[dict[str, Any]]:
        """The current selection of one non-winner scope, or None."""
        scope_kind, scope_key = self._require_other_scope(scope_kind, scope_key)
        with self._read() as conn:
            row = conn.execute(
                """SELECT * FROM selection_pointers
                    WHERE scope_kind=? AND group_id=? AND scope_key=?""",
                (scope_kind, group_id, scope_key),
            ).fetchone()
        return None if row is None else dict(row)

    def scoped_history(
        self,
        *,
        scope_kind: str,
        group_id: str,
        scope_key: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Decisions of one non-winner scope, newest first. Append-only."""
        scope_kind, scope_key = self._require_other_scope(scope_kind, scope_key)
        with self._read() as conn:
            rows = conn.execute(
                """SELECT * FROM selection_decisions
                    WHERE scope_kind=? AND group_id=? AND scope_key=?
                    ORDER BY seq DESC LIMIT ? OFFSET ?""",
                (scope_kind, group_id, scope_key, int(limit), int(offset)),
            ).fetchall()
        return [dict(row) for row in rows]

    def apply_scoped_decision(
        self,
        *,
        scope_kind: str,
        group_id: str,
        scope_key: str,
        decision: str,
        expected_selection_id: Optional[str],
        target: Optional[Mapping[str, Any]] = None,
        clear: bool = False,
        candidate: Optional[Mapping[str, Any]] = None,
        actor: str,
        actor_kind: str,
        provenance: str,
        rationale: Optional[str] = None,
    ) -> dict[str, Any]:
        """Move (or clear, or merely record) one non-winner scoped selection.

        ``expected_selection_id`` is the stale-update check and it is NOT
        optional-by-omission: ``None`` states "I read this scope and it had no
        selection". A caller that has not read the scope cannot express that,
        which is the point — two reviewers picking a best run from the same
        stale screen must not silently overwrite each other.

        Three outcomes, kept apart because they are three different statements
        about the same scope:

        - ``target={"experiment_id", "task_id", "attempt"}`` installs a
          selection.
        - ``clear=True`` removes the current one ("this task has no best run
          after all").
        - neither, which records the decision and leaves the pointer alone —
          how "undecided" is filed without either installing or withdrawing
          anything.

        ``candidate`` records what was being looked at when a decision
        installed nothing, so history can say which run was considered and
        passed over.
        """
        if clear and target is not None:
            raise ValueError("a decision either installs a selection or clears it")
        scope_kind, scope_key = self._require_other_scope(scope_kind, scope_key)
        decision = _require_text(decision, "decision")
        actor = _require_text(actor, "actor")
        provenance = _require_text(provenance, "provenance")
        if actor_kind not in SELECTION_ACTOR_KINDS:
            raise ValueError(
                "actor_kind must be one of " + ", ".join(sorted(SELECTION_ACTOR_KINDS))
            )
        if rationale is not None and len(rationale) > _MAX_TEXT:
            raise ValueError(f"rationale must be at most {_MAX_TEXT} characters")
        now = _utcnow_iso()
        new_selection_id: Optional[str] = None
        with self._write() as conn:
            pointer = conn.execute(
                """SELECT * FROM selection_pointers
                    WHERE scope_kind=? AND group_id=? AND scope_key=?""",
                (scope_kind, group_id, scope_key),
            ).fetchone()
            current_id = None if pointer is None else str(pointer["selection_id"])
            if _clean(expected_selection_id) != current_id:
                raise StaleSelection(
                    group_id=group_id,
                    expected_selection_id=_clean(expected_selection_id),
                    current_selection_id=current_id,
                    current_experiment_id=(
                        None if pointer is None else str(pointer["experiment_id"])
                    ),
                    scope_kind=scope_kind,
                    scope_key=scope_key,
                )
            seq_row = conn.execute(
                """SELECT COALESCE(MAX(seq), 0) AS seq FROM selection_decisions
                    WHERE scope_kind=? AND group_id=? AND scope_key=?""",
                (scope_kind, group_id, scope_key),
            ).fetchone()
            seq = int(seq_row["seq"]) + 1
            if clear:
                conn.execute(
                    """DELETE FROM selection_pointers
                        WHERE scope_kind=? AND group_id=? AND scope_key=?""",
                    (scope_kind, group_id, scope_key),
                )
            elif target is not None:
                new_selection_id = uuid.uuid4().hex
                conn.execute(
                    """INSERT INTO selection_pointers
                       (scope_kind, group_id, scope_key, experiment_id, task_id,
                        attempt, selection_id, decision, decision_seq, decided_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(scope_kind, group_id, scope_key) DO UPDATE SET
                         experiment_id=excluded.experiment_id,
                         task_id=excluded.task_id,
                         attempt=excluded.attempt,
                         selection_id=excluded.selection_id,
                         decision=excluded.decision,
                         decision_seq=excluded.decision_seq,
                         decided_at=excluded.decided_at""",
                    (
                        scope_kind,
                        group_id,
                        scope_key,
                        _require_text(target.get("experiment_id"), "experiment_id"),
                        _clean(target.get("task_id")),
                        target.get("attempt"),
                        new_selection_id,
                        decision,
                        seq,
                        now,
                    ),
                )
            self._append_decision_in_txn(
                conn,
                scope_kind=scope_kind,
                scope_key=scope_key,
                group_id=group_id,
                seq=seq,
                decision=decision,
                previous_experiment_id=(
                    None if pointer is None else str(pointer["experiment_id"])
                ),
                previous_task_id=None if pointer is None else _clean(pointer["task_id"]),
                previous_attempt=None if pointer is None else pointer["attempt"],
                previous_selection_id=current_id,
                candidate_experiment_id=(
                    None if candidate is None else _clean(candidate.get("experiment_id"))
                ),
                candidate_task_id=(
                    None if candidate is None else _clean(candidate.get("task_id"))
                ),
                candidate_attempt=(
                    None if candidate is None else candidate.get("attempt")
                ),
                new_experiment_id=(
                    None if target is None else _clean(target.get("experiment_id"))
                ),
                new_task_id=None if target is None else _clean(target.get("task_id")),
                new_attempt=None if target is None else target.get("attempt"),
                new_selection_id=new_selection_id,
                actor=actor,
                actor_kind=actor_kind,
                provenance=provenance,
                rationale=rationale,
                created_at=now,
            )
        if clear:
            resulting = None
        elif target is not None:
            resulting = new_selection_id
        else:
            resulting = current_id
        return {
            "scope_kind": scope_kind,
            "group_id": group_id,
            "scope_key": scope_key,
            "seq": seq,
            "decision": decision,
            # What is current AFTER this decision: the next caller's
            # `expected_selection_id`, whichever of the three shapes ran.
            "selection_id": resulting,
            "new_selection_id": new_selection_id,
            "previous_selection_id": current_id,
            "decided_at": now,
        }

    # -- reads -----------------------------------------------------------

    def current_winner(self, group_id: str) -> Optional[dict[str, Any]]:
        """The current winner of one group, with the experiment's REAL state.

        ``automatic`` marks the winner nobody chose — the first experiment in
        the group — so a UI can label it as such instead of implying somebody
        judged it. ``experiment`` carries the live status/invalid_reason: this
        method never reports an experiment as successful, and a running or
        failed winner reads as running or failed.

        In a shared control store the experiment is read from ITS OWN source.
        ``experiment_resolved`` is False when the store holding it was not
        supplied (or its evidence was pruned): the winner is still a fact, the
        run's current state is simply unknown, and saying so beats implying the
        experiment vanished.
        """
        with self._read() as conn:
            pointer = conn.execute(
                """SELECT * FROM selection_pointers
                    WHERE scope_kind=? AND group_id=? AND scope_key=?""",
                (EXPERIMENT_SCOPE, group_id, ""),
            ).fetchone()
            group = conn.execute(
                "SELECT * FROM comparison_groups WHERE group_id=?", (group_id,)
            ).fetchone()
        if pointer is None or group is None:
            return None
        experiment_id = str(pointer["experiment_id"])
        source_id = self._member_source_id(experiment_id)
        state = self._experiment_state(experiment_id, source_id)
        return {
            "scope_kind": EXPERIMENT_SCOPE,
            "group": dict(group),
            "group_id": group_id,
            "experiment_id": experiment_id,
            "source_id": source_id,
            "selection_id": str(pointer["selection_id"]),
            "decision": str(pointer["decision"]),
            "decision_seq": int(pointer["decision_seq"]),
            "decided_at": str(pointer["decided_at"]),
            "automatic": str(pointer["decision"]) == DECISION_INITIAL,
            "experiment": state,
            "experiment_resolved": state is not None,
        }

    def winner_for_experiment(self, experiment_id: str) -> Optional[dict[str, Any]]:
        """The winner of the group this experiment belongs to, if any."""
        group_id = self._member_group_id(experiment_id)
        return None if group_id is None else self.current_winner(group_id)

    def group_for_experiment(self, experiment_id: str) -> Optional[dict[str, Any]]:
        group_id = self._member_group_id(experiment_id)
        if group_id is None:
            return None
        with self._read() as conn:
            row = conn.execute(
                "SELECT * FROM comparison_groups WHERE group_id=?", (group_id,)
            ).fetchone()
        return None if row is None else dict(row)

    def list_groups(self) -> list[dict[str, Any]]:
        with self._read() as conn:
            rows = conn.execute(
                "SELECT * FROM comparison_groups ORDER BY created_at, group_id"
            ).fetchall()
        return [dict(row) for row in rows]

    def group_members(self, group_id: str) -> list[dict[str, Any]]:
        """Members oldest first, each with its source and benchmark version."""
        with self._read() as conn:
            rows = conn.execute(
                """SELECT * FROM comparison_group_members
                    WHERE group_id=? ORDER BY joined_at, experiment_id""",
                (group_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def decision_history(
        self, group_id: str, *, limit: int = 100, offset: int = 0
    ) -> list[dict[str, Any]]:
        """Decisions newest first. Append-only: nothing here is ever rewritten."""
        with self._read() as conn:
            rows = conn.execute(
                """SELECT * FROM selection_decisions
                    WHERE scope_kind=? AND group_id=? AND scope_key=?
                    ORDER BY seq DESC LIMIT ? OFFSET ?""",
                (EXPERIMENT_SCOPE, group_id, "", int(limit), int(offset)),
            ).fetchall()
        return [dict(row) for row in rows]

    def source_for_experiment(self, experiment_id: str) -> Optional[str]:
        """The authorized source id holding this experiment's evidence, if bound."""
        return self._member_source_id(experiment_id)

    def store_for_source(
        self, source_id: Optional[str]
    ) -> Optional[ObservabilityStore]:
        """The identity-checked evidence store behind an authorized source id.

        The read side other scopes need: a task-best selection has to look at
        the attempts before it can refuse an unfinished one, and it must look
        at them through the same authorization this control already enforces
        rather than opening a path of its own. None means the embedder handed
        back no handle for an authorized source — unknown, not empty.
        """
        return self._resolve_source(source_id)

    def _member_group_id(self, experiment_id: str) -> Optional[str]:
        with self._read() as conn:
            row = conn.execute(
                "SELECT group_id FROM comparison_group_members WHERE experiment_id=?",
                (experiment_id,),
            ).fetchone()
        return None if row is None else str(row["group_id"])

    def _member_source_id(self, experiment_id: str) -> Optional[str]:
        with self._read() as conn:
            row = conn.execute(
                "SELECT source_id FROM comparison_group_members WHERE experiment_id=?",
                (experiment_id,),
            ).fetchone()
        return None if row is None else _clean(row["source_id"])

    def _experiment_row(
        self, experiment_id: str, source_id: Optional[str]
    ) -> Optional[Mapping[str, Any]]:
        store = self._resolve_source(source_id)
        if store is None:
            if source_id is not None:
                raise EvidenceSourceUnresolved(source_id)
            raise SelectionControlError(
                "no evidence store is readable for this registration; pass "
                "experiment=<row>, or use register_experiment_reference"
            )
        return store.get_experiment(experiment_id)

    def _experiment_state(
        self, experiment_id: str, source_id: Optional[str]
    ) -> Optional[dict[str, Any]]:
        """The winner's own words about itself, or None when unreadable."""
        try:
            store = self._resolve_source(source_id)
        except SelectionControlError:
            return None
        if store is None:
            return None
        try:
            row = store.get_experiment(experiment_id)
        except Exception:  # pragma: no cover - a control read must not fail on evidence
            return None
        if row is None:
            return None
        return {
            "experiment_id": experiment_id,
            "source_id": source_id,
            "description": row.get("description"),
            "status": row.get("status"),
            "invalid_reason": row.get("invalid_reason"),
            "archived": bool(row.get("archived")),
            "workflow_name": row.get("workflow_name"),
            "benchmark_id": row.get("benchmark_id"),
            "benchmark_version": row.get("benchmark_version"),
            "benchmark_digest_sha256": row.get("benchmark_digest_sha256"),
            "created_at": row.get("created_at"),
            "completed_at": row.get("completed_at"),
        }


def _read_only_uri(path: str) -> str:
    """``file:`` URI for a path, percent-encoded, with ``mode=ro``.

    Interpolating a bare path into ``file:{path}?mode=ro`` is wrong for any
    path SQLite's URI parser reads as structure: a ``#`` starts a fragment and
    a ``?`` starts the query, so a control root containing either opens (or
    fails to open) some other file. Here that would be silent — `control_mode_of`
    returns None for an unreadable file, a shared control root would be
    mistaken for "no sidecar yet", and the store would quietly open a private
    single-store sidecar instead of joining the contest.
    """
    return Path(os.path.abspath(path)).as_uri() + "?mode=ro"


def control_mode_of(control_db_path: str) -> Optional[str]:
    """The shape an existing sidecar was written as, or None if there is none.

    Read without creating anything, so a caller can pick the right entry point
    for a file it did not write.
    """
    if not os.path.exists(control_db_path) or os.path.getsize(control_db_path) == 0:
        return None
    try:
        conn = sqlite3.connect(_read_only_uri(control_db_path), uri=True, timeout=30.0)
    except (sqlite3.Error, ValueError):
        return None
    try:
        row = conn.execute(
            "SELECT value FROM control_meta WHERE key=?", (_MODE_KEY,)
        ).fetchone()
    except sqlite3.Error:
        return None
    finally:
        conn.close()
    return None if row is None else str(row[0])


def _open_control_for(
    evidence: ObservabilityStore, control_db_path: Optional[str]
) -> tuple[SelectionControlStore, Optional[str]]:
    """Open the sidecar an evidence store should register into.

    A workspace that runs several stores points them all at ONE control file;
    a plain local run has its own sibling sidecar. Which one this is, is a
    property of the file, not of the caller, so it is read from the file. In
    the shared case the store finds its own source id by identity — and is
    refused if nobody authorized it, because a shared control location must not
    grow a source just because something wrote to it.
    """
    path = control_db_path or control_db_path_for(evidence.db_path)
    if control_mode_of(path) != CONTROL_MODE_SHARED:
        return SelectionControlStore.for_evidence(evidence, control_db_path=path), None
    control = open_shared_control(path)
    identity = evidence.store_identity()
    source_id = control.source_for_identity(identity)
    if source_id is None:
        raise UnauthorizedEvidenceSource(str(identity))
    control.attach_source(source_id, evidence)
    return control, source_id


def initialize_winner_for(
    evidence: ObservabilityStore,
    experiment_id: str,
    *,
    control_db_path: Optional[str] = None,
    experiment: Optional[Mapping[str, Any]] = None,
    strict: bool = False,
) -> dict[str, Any]:
    """Best-effort automatic initialization, called from ``create_experiment``.

    This runs AFTER the evidence row is committed, so it cannot fail the
    creation it follows: raising here would report a failure for an experiment
    that exists, and the caller has nothing to retry. Every control-side
    problem is therefore reported in the return value and logged, never raised
    — ``unavailable`` for the environmental cases (read-only directory, locked
    sidecar) at warning level, ``refused`` for the ones that mean somebody's
    configuration is wrong (identity mismatch, unauthorized source, id
    collision, a sidecar of the other shape) at error level, because those do
    not clear up by themselves and silence would mean winners quietly stop
    being recorded.

    Nothing is weakened by not raising: the refusing check still refuses, and
    no pointer or member row is written. ``strict=True`` re-raises instead, for
    an explicit caller (a UI bootstrap) that can act on the failure.

    ``control_db_path`` may name a SHARED workspace control file, in which case
    the store registers under the source id its own identity was authorized as
    — and is refused if it was never authorized. That is how several runners
    with separate evidence stores land in one contest without any of them
    being able to enrol itself.
    """
    try:
        control, source_id = _open_control_for(evidence, control_db_path)
        registration = control.register_experiment(
            experiment_id, experiment=experiment, source_id=source_id
        )
    except SelectionControlUnavailable as exc:
        if strict:
            raise
        logger.warning(
            f"observability: no winner recorded for experiment {experiment_id!r}: {exc}"
        )
        return {"status": INIT_UNAVAILABLE, "registration": None, "error": str(exc)}
    except sqlite3.Error as exc:
        if strict:
            raise
        logger.warning(
            f"observability: no winner recorded for experiment {experiment_id!r}: {exc}"
        )
        return {"status": INIT_UNAVAILABLE, "registration": None, "error": str(exc)}
    except (SelectionControlError, StoreIdentityMismatch) as exc:
        if strict:
            raise
        logger.error(
            f"observability: refusing to record a winner for experiment "
            f"{experiment_id!r}: {exc}"
        )
        return {"status": INIT_REFUSED, "registration": None, "error": str(exc)}
    return {"status": INIT_RECORDED, "registration": registration, "error": None}


def selection_control_for(
    db_path: str, *, control_db_path: Optional[str] = None, create: bool = True
) -> SelectionControlStore:
    """Open the single-store control sidecar for an evidence DB path.

    The entry point for readers (UI, agent API) in the simple case: evidence is
    opened READ-ONLY, so inspecting winners cannot write to the store being
    inspected.
    """
    from fastworkflow.observability.store import ReadOnlyObservabilityStore

    evidence = ReadOnlyObservabilityStore(db_path)
    return SelectionControlStore.for_evidence(
        evidence, control_db_path=control_db_path, create=create
    )


def open_shared_control(
    control_db_path: str,
    *,
    sources: Optional[SourceSpec] = None,
    create: bool = True,
) -> SelectionControlStore:
    """Open the ONE control location that several evidence stores share.

    ``control_db_path`` comes from the embedder (see
    ``shared_control_db_path_for``); ``sources`` is how it hands back an open
    store for a source id — a mapping or a callable. Opening authorizes
    nothing: each store must be passed to ``authorize_source`` once, and a
    store whose identity does not match the one authorized under that id is
    refused on every read.
    """
    return SelectionControlStore(
        control_db_path,
        sources=sources if sources is not None else {},
        mode=CONTROL_MODE_SHARED,
        create=create,
    )


__all__ = [
    "CLIENT_DECISIONS",
    "CONTROL_MODES",
    "CONTROL_MODE_SHARED",
    "CONTROL_MODE_SINGLE",
    "CONTROL_SCHEMA_VERSION",
    "ControlModeMismatch",
    "ControlStoreIdentityMismatch",
    "DECISION_INITIAL",
    "DECISION_KEEP",
    "DECISION_PROMOTE",
    "DECISION_RETIRE",
    "DECISION_UNDECIDED",
    "EXPERIMENT_SCOPE",
    "EvidenceSourceUnresolved",
    "ExperimentNotInGroup",
    "ExperimentReference",
    "ExperimentSourceCollision",
    "GROUP_ADHOC",
    "GROUP_BENCHMARK",
    "INIT_RECORDED",
    "INIT_REFUSED",
    "INIT_UNAVAILABLE",
    "NoCurrentSelection",
    "PRIMARY_SOURCE_ID",
    "SELECTION_ACTOR_KINDS",
    "SHARED_CONTROL_FILENAME",
    "SYSTEM_ACTOR_KIND",
    "SelectionControlError",
    "SelectionControlStore",
    "SelectionControlUnavailable",
    "SelectionRetirementRefused",
    "StaleSelection",
    "TASK_BEST_SCOPE",
    "ExperimentNotFound",
    "UnauthorizedEvidenceSource",
    "UnknownComparisonGroup",
    "comparison_group_identity",
    "control_db_path_for",
    "control_mode_of",
    "initialize_winner_for",
    "open_shared_control",
    "selection_control_for",
    "shared_control_db_path_for",
]

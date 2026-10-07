"""Control tables: judgements kept in the workflow's live evidence DB (`fix-10vj`).

A winner or a pair-review mark is a judgement ABOUT evidence, not evidence.
They live in tables of the live DB rather than in sidecar files beside it,
under one feature marker, `control_v1`, so a DB either has every control table
or none of them. Sealed copies never carry them
(`strip`), which is what keeps a sealed file's digest fixed while judgements
about it keep being made.

When the tables are created: a writable `migrate=True` open
(`ObservabilityStore._ensure_schema`), and a control writer that finds the
marker missing, inside its own `BEGIN IMMEDIATE` (`write`). Reading never
creates anything: a DB without the file or the marker answers every control
read with the empty answer (`rows`).

This module imports nothing from `store.py`, so `store.py` can import it.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterable, Iterator, Optional

FEATURE_CONTROL_V1 = "control_v1"

CONTROL_SCHEMA = [
    """CREATE TABLE IF NOT EXISTS experiment_registrations (
        experiment_id TEXT PRIMARY KEY,
        benchmark_id TEXT NOT NULL,
        benchmark_version TEXT NOT NULL,
        benchmark_digest_sha256 TEXT NOT NULL,
        description TEXT NOT NULL,
        task_ids_json TEXT NOT NULL,
        runs_per_task INTEGER NOT NULL,
        source_experiment_id TEXT,
        changed_fields_json TEXT NOT NULL DEFAULT '[]',
        state TEXT NOT NULL DEFAULT 'registered'
            CHECK (state IN ('registered', 'bound', 'deleted')),
        created_at TEXT NOT NULL,
        bound_at TEXT,
        deleted_at TEXT)""",
    """CREATE INDEX IF NOT EXISTS idx_registrations_benchmark
        ON experiment_registrations(benchmark_id, state)""",
    # One group per (workflow, benchmark lineage). `benchmark_version` is NOT
    # part of the identity: the winner is scoped to the lineage across versions
    # (`fix-9eg.17`), and each member row records the version it ran so a
    # version change is visible rather than silently starting a new contest.
    """CREATE TABLE IF NOT EXISTS comparison_groups (
        group_id TEXT PRIMARY KEY, group_kind TEXT NOT NULL,
        workflow_name TEXT, benchmark_id TEXT,
        created_at TEXT NOT NULL, created_from_experiment_id TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS comparison_group_members (
        group_id TEXT NOT NULL, experiment_id TEXT NOT NULL,
        benchmark_version TEXT, benchmark_digest_sha256 TEXT,
        created_at TEXT, joined_at TEXT NOT NULL,
        PRIMARY KEY (group_id, experiment_id))""",
    # An experiment belongs to exactly one group. Membership is first-write-wins
    # (see `register_experiment`): a pin added on a later re-create must not move
    # an experiment out from under a winner pointer that already references it.
    """CREATE UNIQUE INDEX IF NOT EXISTS idx_group_member_experiment
        ON comparison_group_members(experiment_id)""",
    # `task_id`/`attempt` are NULL at experiment scope and used by the
    # task-best scope (`fix-9eg.17.4`).
    """CREATE TABLE IF NOT EXISTS selection_pointers (
        scope_kind TEXT NOT NULL, group_id TEXT NOT NULL, scope_key TEXT NOT NULL,
        experiment_id TEXT NOT NULL, task_id TEXT, attempt INTEGER,
        selection_id TEXT NOT NULL, decision TEXT NOT NULL,
        decision_seq INTEGER NOT NULL, decided_at TEXT NOT NULL,
        PRIMARY KEY (scope_kind, group_id, scope_key))""",
    """CREATE TABLE IF NOT EXISTS selection_decisions (
        decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
        scope_kind TEXT NOT NULL, group_id TEXT NOT NULL, scope_key TEXT NOT NULL,
        seq INTEGER NOT NULL, decision TEXT NOT NULL,
        previous_experiment_id TEXT, previous_task_id TEXT,
        previous_attempt INTEGER, previous_selection_id TEXT,
        candidate_experiment_id TEXT, candidate_task_id TEXT, candidate_attempt INTEGER,
        new_experiment_id TEXT, new_task_id TEXT, new_attempt INTEGER,
        new_selection_id TEXT,
        actor TEXT NOT NULL, actor_kind TEXT NOT NULL, provenance TEXT NOT NULL,
        rationale TEXT, created_at TEXT NOT NULL,
        UNIQUE (scope_kind, group_id, scope_key, seq))""",
    """CREATE INDEX IF NOT EXISTS idx_selection_decisions_scope
        ON selection_decisions(scope_kind, group_id, scope_key, seq)""",
    """CREATE TRIGGER IF NOT EXISTS selection_decisions_no_update
        BEFORE UPDATE ON selection_decisions
        BEGIN SELECT RAISE(ABORT, 'decision history is append-only'); END""",
    """CREATE TRIGGER IF NOT EXISTS selection_decisions_no_delete
        BEFORE DELETE ON selection_decisions
        BEGIN SELECT RAISE(ABORT, 'decision history is append-only'); END""",
    # `pair_key` is `comparison.review_pair_key(left, right)`. The two ref ids
    # and their full JSON are stored beside it so a stored row can still say
    # WHAT was compared after the UI that created it has moved on, without
    # re-deriving the key.
    """CREATE TABLE IF NOT EXISTS pair_reviews (
        pair_key TEXT NOT NULL, reviewer TEXT NOT NULL, reviewer_kind TEXT NOT NULL,
        left_ref_id TEXT NOT NULL, right_ref_id TEXT NOT NULL,
        left_ref_json TEXT NOT NULL, right_ref_json TEXT NOT NULL,
        state TEXT NOT NULL, seq INTEGER NOT NULL, note TEXT,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (pair_key, reviewer))""",
    """CREATE INDEX IF NOT EXISTS idx_pair_reviews_left
        ON pair_reviews(left_ref_id, reviewer)""",
    """CREATE TABLE IF NOT EXISTS pair_review_events (
        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
        pair_key TEXT NOT NULL, reviewer TEXT NOT NULL, reviewer_kind TEXT NOT NULL,
        seq INTEGER NOT NULL, previous_state TEXT, state TEXT NOT NULL,
        note TEXT, created_at TEXT NOT NULL,
        UNIQUE (pair_key, reviewer, seq))""",
    """CREATE TABLE IF NOT EXISTS sealed_archives (
        experiment_id TEXT PRIMARY KEY,
        archive_sha256 TEXT NOT NULL UNIQUE,
        path TEXT NOT NULL,
        size_bytes INTEGER NOT NULL,
        sealed_at TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS training_process (
        slot INTEGER PRIMARY KEY CHECK (slot = 1),
        pid INTEGER NOT NULL,
        proc_start_ticks INTEGER NOT NULL,
        server_incarnation TEXT,
        log_path TEXT NOT NULL,
        started_at TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS evidence_releases (
        experiment_id TEXT PRIMARY KEY,
        released_at TEXT NOT NULL, actor TEXT NOT NULL, reason TEXT)""",
]

CONTROL_TABLES = (
    "experiment_registrations", "comparison_groups", "comparison_group_members",
    "selection_pointers", "selection_decisions", "pair_reviews",
    "pair_review_events", "sealed_archives",
    "training_process", "evidence_releases",
)

class ControlUnavailable(RuntimeError):
    """There is no live DB to record this judgement in. Nothing was created."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _features(conn: sqlite3.Connection) -> list[str]:
    row = conn.execute(
        "SELECT value FROM diagnostics WHERE key='schema_features'"
    ).fetchone()
    try:
        loaded = json.loads(row[0]) if row is not None else []
    except (ValueError, TypeError):
        loaded = []
    return [str(name) for name in loaded] if isinstance(loaded, list) else []


def _set_features(conn: sqlite3.Connection, features: Iterable[str]) -> None:
    conn.execute(
        """INSERT INTO diagnostics (key, value, updated_at) VALUES (?, ?, ?)
           ON CONFLICT(key) DO UPDATE SET
             value=excluded.value, updated_at=excluded.updated_at""",
        ("schema_features", json.dumps(sorted(set(features))), _now()),
    )


def ensure(conn: sqlite3.Connection) -> None:
    """Create every control table and record the marker, in the caller's txn."""
    features = _features(conn)
    if FEATURE_CONTROL_V1 in features:
        return
    for statement in CONTROL_SCHEMA:
        conn.execute(statement)
    _set_features(conn, [*features, FEATURE_CONTROL_V1])


def strip(conn: sqlite3.Connection) -> None:
    """Remove every control table and the marker from a sealed copy (§3 Rule 1)."""
    for table in CONTROL_TABLES:
        conn.execute(f"DROP TABLE IF EXISTS {table}")
    _set_features(conn, set(_features(conn)) - {FEATURE_CONTROL_V1})
    conn.commit()


def present(store: Any) -> bool:
    """Whether this store's DB holds the control tables. Never writes."""
    if store is None or not os.path.isfile(store.db_path):
        return False
    with store._connect() as conn:
        return FEATURE_CONTROL_V1 in _features(conn)


def rows(store: Any, sql: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
    """A control read; empty when the DB or the marker is absent. Never writes."""
    if store is None or not os.path.isfile(store.db_path):
        return []
    with store._connect() as conn:
        if FEATURE_CONTROL_V1 not in _features(conn):
            return []
        return [dict(row) for row in conn.execute(sql, tuple(params))]


@contextmanager
def write(store: Any) -> Iterator[sqlite3.Connection]:
    """One short `BEGIN IMMEDIATE` control write against an existing live DB."""
    if store is None or not os.path.isfile(store.db_path):
        raise ControlUnavailable(
            "there is no live evidence database to record this in"
            + (f" ({store.db_path})" if store is not None else "")
        )
    with store._connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        ensure(conn)
        yield conn


def write_refusal_status(exc: sqlite3.OperationalError) -> Optional[int]:
    """The HTTP status a failed control write is reported under (§5), if any.

    503 for a DB still busy after the connection's 30 s timeout, which is worth
    retrying; 409 for a read-only file, which is not. None for anything else.
    """
    message = str(exc).lower()
    if "locked" in message or "busy" in message:
        return 503
    if "readonly" in message or "read-only" in message:
        return 409
    return None

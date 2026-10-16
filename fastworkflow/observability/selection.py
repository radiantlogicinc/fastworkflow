"""Current-winner pointer and append-only selection history (`fix-9eg.17.1`).

A winner is a JUDGEMENT ABOUT evidence, not evidence. It is recorded in the
control tables of the workflow's live evidence DB (`control.py`), never in a
sealed copy: sealed copies carry no control table, so a decision recorded
about an archived run never changes the archive's bytes, and pruning evidence
does not retract a decision somebody made.

Scope of a contest: WORKFLOW + BENCHMARK + BENCHMARK VERSION (`fix-873a`).
Experiments run against two versions of a benchmark ran different task sets,
so they are never in one contest and never share a winner. There is one live
DB per workflow, so there is one control location; `store_for` hands back a
member's sealed archive once it has one (`sealed_archives`).

The winner is deliberately plain. A contest has zero or one winner; somebody
sets it (`set_winner`) or removes it (`clear_winner`), and no winner is a
normal state. Nothing is elected automatically, and there is no stale-read
token: the last write wins. Each change appends one row to the decision
history, so who set or removed it, and when, stays readable.

Membership is explicit. An experiment joins its group when it is registered or
recorded (`create_experiment`); one recorded before this build joins when a
decision names it. Nothing scans history and adopts it, and a read never
enrols anything.

Scopes. The same tables carry the per-task "best run" (`best_run.py`,
``TASK_BEST_SCOPE``) under different ``scope_kind``/``scope_key`` values, so the
two can never overwrite each other: selecting a best run for a task cannot
promote its experiment, and setting the winner cannot touch a task's best run.

Selecting a winner records a reference. It does not deploy code, restore data,
rerun anything, or claim the winner succeeded — ``current_winner()`` reports the
experiment's ACTUAL status.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import uuid
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

from fastworkflow.observability import control
from fastworkflow.observability.store import (
    FEEDBACK_PROVENANCES,
    ExperimentNotFound,
    ObservabilityStore,
    ReadOnlyObservabilityStore,
    Redactor,
    _utcnow_iso,
)

EXPERIMENT_SCOPE = "experiment"
# The per-task best run (`best_run.py`). Declared here so the column vocabulary
# is fixed in one place; only `best_run` writes it.
TASK_BEST_SCOPE = "task_best"

# The two winner decisions. History rows written by older builds may carry
# other words (`initial`, `promote`, `keep`, `undecided`, `retire`); they stay
# readable, nothing writes them any more.
DECISION_SET = "set"
DECISION_CLEAR = "clear"

GROUP_BENCHMARK = "benchmark"

# Who decided. Shares the feedback vocabulary so a human comment and an agent's
# structured decision describe their origin the same way, plus `system` for
# changes no actor asked for (a deleted registration leaving its contest).
SYSTEM_ACTOR_KIND = "system"
SELECTION_ACTOR_KINDS = frozenset(FEEDBACK_PROVENANCES | {SYSTEM_ACTOR_KIND})

_MAX_TEXT = 4000


class SelectionControlError(RuntimeError):
    """Base class for control-metadata failures."""


class UnknownComparisonGroup(KeyError):
    """No comparison group with that id exists in this live DB."""

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


class StaleSelection(ValueError):
    """A best-run decision was made against a selection that has since moved.

    Carries what the caller expected and what is actually current so a UI or an
    agent can say which decision won, rather than reporting a generic conflict.
    """

    def __init__(
        self,
        *,
        group_id: str,
        expected_selection_id: Optional[str],
        current_selection_id: Optional[str],
        current_experiment_id: Optional[str],
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


def comparison_group_identity(experiment: Mapping[str, Any]) -> dict[str, Any]:
    """Derive the comparison group an experiment row belongs to.

    Experiments group by (workflow, benchmark, benchmark version): two
    versions of a benchmark are two contests (`fix-873a`). Every experiment
    pins a benchmark version (`fix-ujat`); a row without one has no contest
    and is refused here.

    The evidence store is deliberately absent from the key: two runs of one
    benchmark version recorded in two stores are one contest.

    Deterministic by construction — the id is derived, never minted — so two
    processes registering experiments at the same moment compute the same
    group id instead of creating two groups.
    """
    workflow_name = _clean(experiment.get("workflow_name")) or ""
    benchmark_id = _clean(experiment.get("benchmark_id")) or ""
    benchmark_version = _clean(experiment.get("benchmark_version")) or ""
    if not benchmark_id:
        raise ValueError(
            f"experiment {experiment.get('experiment_id')!r} names no benchmark "
            "version, so it belongs to no contest"
        )
    kind = GROUP_BENCHMARK
    parts = (GROUP_BENCHMARK, workflow_name, benchmark_id, benchmark_version)
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]
    return {
        "group_id": f"{kind}-{digest}",
        "group_kind": kind,
        "workflow_name": workflow_name or None,
        "benchmark_id": benchmark_id or None,
        "benchmark_version": benchmark_version or None,
    }


@dataclass(frozen=True)
class ExperimentReference:
    """An experiment named before it has recorded any evidence.

    Carries exactly the fields a contest is scoped by, plus ``created_at``; a
    reference registered with no timestamp is treated as created now.
    """

    experiment_id: str
    workflow_name: Optional[str] = None
    benchmark_id: Optional[str] = None
    benchmark_version: Optional[str] = None
    benchmark_digest_sha256: Optional[str] = None
    created_at: Optional[str] = None

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


def _require_actor(actor: str, actor_kind: str, provenance: str) -> tuple[str, str]:
    if actor_kind not in SELECTION_ACTOR_KINDS:
        raise ValueError(
            "actor_kind must be one of " + ", ".join(sorted(SELECTION_ACTOR_KINDS))
        )
    return _require_text(actor, "actor"), _require_text(provenance, "provenance")


def _next_seq(conn: sqlite3.Connection, scope_kind: str, group_id: str, scope_key: str) -> int:
    return int(
        conn.execute(
            """SELECT COALESCE(MAX(seq), 0) FROM selection_decisions
                WHERE scope_kind=? AND group_id=? AND scope_key=?""",
            (scope_kind, group_id, scope_key),
        ).fetchone()[0]
    ) + 1


def _pointer(
    conn: sqlite3.Connection, group_id: str, scope_kind: str = EXPERIMENT_SCOPE, scope_key: str = ""
) -> Optional[sqlite3.Row]:
    return conn.execute(
        """SELECT * FROM selection_pointers
            WHERE scope_kind=? AND group_id=? AND scope_key=?""",
        (scope_kind, group_id, scope_key),
    ).fetchone()


class SelectionControlStore:
    """Winner pointer + append-only decision history in one workflow's live DB.

    Thread/process-safe the same way ``ObservabilityStore`` is: every write is
    one short ``BEGIN IMMEDIATE`` transaction (`control.write`), so SQLite's
    file lock — not an in-process lock — is what serialises two writers.

    ``store`` may be a ``ReadOnlyObservabilityStore`` for reads, or None when
    the workflow has no live DB yet: every read then answers empty and every
    write refuses with `control.ControlUnavailable`, creating nothing.
    """

    def __init__(self, store: Optional[ObservabilityStore]) -> None:
        self.store = store
        self._redactor: Optional[Redactor] = None

    def _rows(self, sql: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
        return control.rows(self.store, sql, params)

    def _row(self, sql: str, params: Iterable[Any] = ()) -> Optional[dict[str, Any]]:
        found = self._rows(sql, params)
        return found[0] if found else None

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
    ) -> dict[str, Any]:
        """Record an experiment in its group. Idempotent; elects nobody.

        ``create_experiment`` re-runs on every resume, so a second call adds no
        member row.
        """
        row = experiment
        if row is None and self.store is not None:
            row = self.store.get_experiment(experiment_id)
        if row is None:
            raise ExperimentNotFound(experiment_id)
        return self._register(experiment_id, row)

    def register_experiment_reference(self, reference: ExperimentReference) -> dict[str, Any]:
        """Register an experiment that has recorded no evidence yet.

        The UI-creation path: an experiment exists as a decision to run
        something long before a runner records anything for it. The reference
        carries the fields the contest is scoped by.
        """
        return self._register(
            _require_text(reference.experiment_id, "experiment_id"),
            reference.as_row(),
        )

    def _register(self, experiment_id: str, row: Mapping[str, Any]) -> dict[str, Any]:
        with control.write(self.store) as conn:
            joined = self.join_in_txn(conn, experiment_id, row)
        return {**joined, "winner": self.current_winner(joined["group_id"])}

    def join_in_txn(
        self,
        conn: sqlite3.Connection,
        experiment_id: str,
        row: Mapping[str, Any],
    ) -> dict[str, Any]:
        """The one join path, inside the caller's ``BEGIN IMMEDIATE``.

        Registration, `create_experiment` and the enrolment of a decision
        request all come through here. Joining never sets a winner.
        """
        experiment_id = _require_text(experiment_id, "experiment_id")
        control.ensure(conn)
        identity = comparison_group_identity(row)
        group_id = identity["group_id"]
        now = _utcnow_iso()
        registered = False
        member = conn.execute(
            "SELECT group_id FROM comparison_group_members WHERE experiment_id=?",
            (experiment_id,),
        ).fetchone()
        if member is None:
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
                   (group_id, experiment_id, benchmark_version,
                    benchmark_digest_sha256, created_at, joined_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    group_id,
                    experiment_id,
                    _clean(row.get("benchmark_version")),
                    _clean(row.get("benchmark_digest_sha256")),
                    _clean(row.get("created_at")) or now,
                    now,
                ),
            )
            registered = True
        else:
            # Membership is sticky: an experiment's benchmark pin is
            # write-once, so the group it first joined is the right one.
            group_id = str(member["group_id"])
        return {
            "group_id": group_id,
            "experiment_id": experiment_id,
            "registered": registered,
        }

    def _enrol_in_txn(
        self, conn: sqlite3.Connection, group_id: str, experiment_id: Optional[str]
    ) -> bool:
        """Join a recorded, not-yet-enrolled experiment a decision names.

        One that is not in `experiments`, or whose group is another one, is
        left alone and refused by the caller's own membership check.
        """
        experiment_id = _clean(experiment_id)
        if experiment_id is None or conn.execute(
            "SELECT 1 FROM comparison_group_members WHERE experiment_id=?",
            (experiment_id,),
        ).fetchone() is not None:
            return False
        row = conn.execute(
            "SELECT * FROM experiments WHERE experiment_id=?", (experiment_id,)
        ).fetchone()
        if row is None or comparison_group_identity(dict(row))["group_id"] != group_id:
            return False
        self.join_in_txn(conn, experiment_id, dict(row))
        return True

    def leave_in_txn(self, conn: sqlite3.Connection, experiment_id: str) -> None:
        """A deleted registration leaves its contest, inside the caller's txn.

        Its membership row goes, and so does the winner pointer when it names
        it: the contest then simply has no winner. History is untouched.
        """
        member = conn.execute(
            "SELECT group_id FROM comparison_group_members WHERE experiment_id=?",
            (experiment_id,),
        ).fetchone()
        if member is None:
            return
        conn.execute(
            "DELETE FROM comparison_group_members WHERE experiment_id=?",
            (experiment_id,),
        )
        conn.execute(
            """DELETE FROM selection_pointers
                WHERE scope_kind=? AND group_id=? AND scope_key='' AND experiment_id=?""",
            (EXPERIMENT_SCOPE, str(member["group_id"]), experiment_id),
        )

    # -- the winner --------------------------------------------------------

    def _group_in_txn(self, conn: sqlite3.Connection, experiment_id: str) -> str:
        """The group this experiment is in, enrolling a recorded one first."""
        member = conn.execute(
            "SELECT group_id FROM comparison_group_members WHERE experiment_id=?",
            (experiment_id,),
        ).fetchone()
        if member is not None:
            return str(member["group_id"])
        row = conn.execute(
            "SELECT * FROM experiments WHERE experiment_id=?", (experiment_id,)
        ).fetchone()
        if row is None:
            raise ExperimentNotFound(experiment_id)
        return self.join_in_txn(conn, experiment_id, dict(row))["group_id"]

    def set_winner(
        self,
        experiment_id: str,
        *,
        actor: str,
        actor_kind: str,
        provenance: str,
        rationale: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """Make this experiment the winner of its contest.

        Replaces whatever winner the contest had. Setting the experiment that
        already wins changes nothing and records nothing.
        """
        experiment_id = _require_text(experiment_id, "experiment_id")
        actor, provenance = _require_actor(actor, actor_kind, provenance)
        rationale = _clean(rationale)
        if rationale is not None and len(rationale) > _MAX_TEXT:
            raise ValueError(f"rationale must be at most {_MAX_TEXT} characters")
        now = _utcnow_iso()
        with control.write(self.store) as conn:
            group_id = self._group_in_txn(conn, experiment_id)
            current = _pointer(conn, group_id)
            if current is not None and str(current["experiment_id"]) == experiment_id:
                return self.current_winner(group_id)
            seq = _next_seq(conn, EXPERIMENT_SCOPE, group_id, "")
            selection_id = uuid.uuid4().hex
            conn.execute(
                """INSERT INTO selection_pointers
                   (scope_kind, group_id, scope_key, experiment_id, task_id,
                    attempt, selection_id, decision, decision_seq, decided_at)
                   VALUES (?, ?, '', ?, NULL, NULL, ?, ?, ?, ?)
                   ON CONFLICT(scope_kind, group_id, scope_key) DO UPDATE SET
                     experiment_id=excluded.experiment_id,
                     selection_id=excluded.selection_id,
                     decision=excluded.decision,
                     decision_seq=excluded.decision_seq,
                     decided_at=excluded.decided_at""",
                (EXPERIMENT_SCOPE, group_id, experiment_id, selection_id,
                 DECISION_SET, seq, now),
            )
            self._append_decision_in_txn(
                conn,
                group_id=group_id,
                seq=seq,
                decision=DECISION_SET,
                previous_experiment_id=(
                    None if current is None else str(current["experiment_id"])
                ),
                previous_selection_id=(
                    None if current is None else str(current["selection_id"])
                ),
                candidate_experiment_id=experiment_id,
                new_experiment_id=experiment_id,
                new_selection_id=selection_id,
                actor=actor,
                actor_kind=actor_kind,
                provenance=provenance,
                rationale=rationale,
                created_at=now,
            )
        return self.current_winner(group_id)

    def clear_winner(
        self,
        experiment_id: str,
        *,
        actor: str,
        actor_kind: str,
        provenance: str,
        rationale: Optional[str] = None,
    ) -> None:
        """Remove the winner of this experiment's contest, whichever it is.

        A contest with no winner is left as it is and nothing is recorded.
        """
        experiment_id = _require_text(experiment_id, "experiment_id")
        actor, provenance = _require_actor(actor, actor_kind, provenance)
        rationale = _clean(rationale)
        if rationale is not None and len(rationale) > _MAX_TEXT:
            raise ValueError(f"rationale must be at most {_MAX_TEXT} characters")
        now = _utcnow_iso()
        with control.write(self.store) as conn:
            group_id = self._group_in_txn(conn, experiment_id)
            current = _pointer(conn, group_id)
            if current is None:
                return
            seq = _next_seq(conn, EXPERIMENT_SCOPE, group_id, "")
            conn.execute(
                """DELETE FROM selection_pointers
                    WHERE scope_kind=? AND group_id=? AND scope_key=''""",
                (EXPERIMENT_SCOPE, group_id),
            )
            self._append_decision_in_txn(
                conn,
                group_id=group_id,
                seq=seq,
                decision=DECISION_CLEAR,
                previous_experiment_id=str(current["experiment_id"]),
                previous_selection_id=str(current["selection_id"]),
                candidate_experiment_id=None,
                new_experiment_id=None,
                new_selection_id=None,
                actor=actor,
                actor_kind=actor_kind,
                provenance=provenance,
                rationale=rationale,
                created_at=now,
            )

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
    # are both refused below — the winner moves only through `set_winner` and
    # `clear_winner`.

    def _require_other_scope(self, scope_kind: str, scope_key: str) -> tuple[str, str]:
        scope_kind = _require_text(scope_kind, "scope_kind")
        scope_key = _require_text(scope_key, "scope_key")
        if scope_kind == EXPERIMENT_SCOPE:
            raise ValueError(
                "the experiment winner is moved only by set_winner/clear_winner; a "
                "scoped decision must name a different scope_kind"
            )
        return scope_kind, scope_key

    def scoped_pointer(
        self, *, scope_kind: str, group_id: str, scope_key: str
    ) -> Optional[dict[str, Any]]:
        """The current selection of one non-winner scope, or None."""
        scope_kind, scope_key = self._require_other_scope(scope_kind, scope_key)
        return self._row(
            """SELECT * FROM selection_pointers
                WHERE scope_kind=? AND group_id=? AND scope_key=?""",
            (scope_kind, group_id, scope_key),
        )

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
        return self._rows(
            """SELECT * FROM selection_decisions
                WHERE scope_kind=? AND group_id=? AND scope_key=?
                ORDER BY seq DESC LIMIT ? OFFSET ?""",
            (scope_kind, group_id, scope_key, int(limit), int(offset)),
        )

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

        The experiment the target or candidate names is enrolled first when it
        was recorded here before this build and is not yet a member.
        """
        if clear and target is not None:
            raise ValueError("a decision either installs a selection or clears it")
        scope_kind, scope_key = self._require_other_scope(scope_kind, scope_key)
        decision = _require_text(decision, "decision")
        actor, provenance = _require_actor(actor, actor_kind, provenance)
        if rationale is not None and len(rationale) > _MAX_TEXT:
            raise ValueError(f"rationale must be at most {_MAX_TEXT} characters")
        now = _utcnow_iso()
        new_selection_id: Optional[str] = None
        with control.write(self.store) as conn:
            pointer = _pointer(conn, group_id, scope_kind, scope_key)
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
            named = target if target is not None else candidate
            if named is not None:
                self._enrol_in_txn(conn, group_id, named.get("experiment_id"))
            seq = _next_seq(conn, scope_kind, group_id, scope_key)
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
        """The current winner of one group, or None, with the experiment's REAL state.

        ``experiment`` carries the live status/invalid_reason: this method
        never reports an experiment as successful, and a running or failed
        winner reads as running or failed.

        ``experiment_resolved`` is False when the winner has no recorded row
        (a registration that has not run yet, or evidence that was pruned): the
        winner is still a fact, the run's current state is simply unknown, and
        saying so beats implying the experiment vanished.
        """
        pointer = self._row(
            """SELECT * FROM selection_pointers
                WHERE scope_kind=? AND group_id=? AND scope_key=?""",
            (EXPERIMENT_SCOPE, group_id, ""),
        )
        group = self._row("SELECT * FROM comparison_groups WHERE group_id=?", (group_id,))
        if pointer is None or group is None:
            return None
        experiment_id = str(pointer["experiment_id"])
        state = self._experiment_state(experiment_id)
        return {
            "scope_kind": EXPERIMENT_SCOPE,
            "group": group,
            "group_id": group_id,
            "experiment_id": experiment_id,
            "selection_id": str(pointer["selection_id"]),
            "decision": str(pointer["decision"]),
            "decision_seq": int(pointer["decision_seq"]),
            "decided_at": str(pointer["decided_at"]),
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
        return self._row("SELECT * FROM comparison_groups WHERE group_id=?", (group_id,))

    def list_groups(self) -> list[dict[str, Any]]:
        return self._rows("SELECT * FROM comparison_groups ORDER BY created_at, group_id")

    def group_members(self, group_id: str) -> list[dict[str, Any]]:
        """Members oldest first, each with its benchmark version."""
        return self._rows(
            """SELECT * FROM comparison_group_members
                WHERE group_id=? ORDER BY joined_at, experiment_id""",
            (group_id,),
        )

    def decision_history(
        self, group_id: str, *, limit: int = 100, offset: int = 0
    ) -> list[dict[str, Any]]:
        """Decisions newest first. Append-only: nothing here is ever rewritten."""
        return self._rows(
            """SELECT * FROM selection_decisions
                WHERE scope_kind=? AND group_id=? AND scope_key=?
                ORDER BY seq DESC LIMIT ? OFFSET ?""",
            (EXPERIMENT_SCOPE, group_id, "", int(limit), int(offset)),
        )

    def store_for(self, experiment_id: str) -> Optional[ObservabilityStore]:
        """The store holding one experiment's evidence (§2.6).

        Its sealed archive when one is recorded and its file is there at the
        recorded size, otherwise the live DB. Sealed evidence is the frozen
        truth, and preferring it keeps a sealed member readable after live
        pruning released its spans. The sha is verified at seal time, not here,
        so a read never hashes a file.
        """
        sealed = self._row(
            "SELECT path, size_bytes FROM sealed_archives WHERE experiment_id=?",
            (experiment_id,),
        )
        if sealed is not None and os.path.isfile(sealed["path"]) and (
            os.path.getsize(sealed["path"]) == int(sealed["size_bytes"])
        ):
            return ReadOnlyObservabilityStore(sealed["path"])
        return self.store

    def _member_group_id(self, experiment_id: str) -> Optional[str]:
        row = self._row(
            "SELECT group_id FROM comparison_group_members WHERE experiment_id=?",
            (experiment_id,),
        )
        return None if row is None else str(row["group_id"])

    def _experiment_state(self, experiment_id: str) -> Optional[dict[str, Any]]:
        """The winner's own words about itself, or None when unreadable."""
        store = self.store_for(experiment_id)
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


__all__ = [
    "DECISION_CLEAR",
    "DECISION_SET",
    "EXPERIMENT_SCOPE",
    "ExperimentNotInGroup",
    "ExperimentReference",
    "GROUP_BENCHMARK",
    "SELECTION_ACTOR_KINDS",
    "SYSTEM_ACTOR_KIND",
    "SelectionControlError",
    "SelectionControlStore",
    "StaleSelection",
    "TASK_BEST_SCOPE",
    "ExperimentNotFound",
    "UnknownComparisonGroup",
    "comparison_group_identity",
]

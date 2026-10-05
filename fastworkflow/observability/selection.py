"""Current-winner pointer and append-only selection history (`fix-9eg.17.1`).

A winner is a JUDGEMENT ABOUT evidence, not evidence. It is recorded in the
control tables of the workflow's live evidence DB (`control.py`), never in a
sealed copy: sealed copies carry no control table, so a decision recorded
about an archived run never changes the archive's bytes, and pruning evidence
does not retract a decision somebody made.

Scope of a contest: WORKFLOW + BENCHMARK LINEAGE, never DB identity. There is
one live DB per workflow, so there is one control location and nothing to
authorize: the experiments it judges are the ones its own `experiments` table
records, and `store_for` hands back a member's sealed archive instead once it
has one (`sealed_archives`).

Registration can run BEFORE the experiment has evidence. ``register_experiment_
reference`` records an experiment from an explicit ``ExperimentReference``
(workflow, benchmark lineage, created_at) so a UI that creates an experiment
can initialize the first winner at creation time rather than at execution time.

Membership is explicit. An experiment joins when it is registered or recorded
(`create_experiment`), and an experiment recorded before this build joins only
when a decision request names it (``record_decision`` /
``apply_scoped_decision``). Nothing scans history and adopts it, and a read
never enrols anything.

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


class SelectionControlError(RuntimeError):
    """Base class for control-metadata failures."""


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


class NoCurrentSelection(ValueError):
    """A decision was recorded about a scope that has no current selection.

    For the experiment winner that is a group nobody has joined with an
    election yet; the message says how one is elected, because "no current
    winner" alone reads like a bug rather than a state.
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
    """An experiment named before it has recorded any evidence.

    Carries exactly the fields a contest is scoped by, plus the ordering key.
    ``created_at`` matters: it is what the oldest-member election compares, so
    a reference registered with no timestamp is treated as created now, which
    is the truth at UI-creation time.
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


def _order_key(created_at: Any, experiment_id: Any) -> tuple[str, str]:
    """The one ordering rule for "older", stated once so two callers agree."""
    return (str(created_at or ""), str(experiment_id or ""))


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
        """
        row = experiment
        if row is None and self.store is not None:
            row = self.store.get_experiment(experiment_id)
        if row is None:
            raise ExperimentNotFound(experiment_id)
        return self._register(
            experiment_id,
            row,
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
        """Register an experiment that has recorded no evidence yet.

        This is the UI-creation path: an experiment exists as a decision to run
        something long before a runner records anything for it, and the first
        winner of a brand-new group should be initialized then, not on first
        execution. The reference carries the lineage fields the contest is
        scoped by.
        """
        return self._register(
            _require_text(reference.experiment_id, "experiment_id"),
            reference.as_row(),
            allow_initial_winner=allow_initial_winner,
            actor=actor,
            actor_kind=actor_kind,
            provenance=provenance,
        )

    def _register(
        self,
        experiment_id: str,
        row: Mapping[str, Any],
        *,
        allow_initial_winner: bool,
        actor: str,
        actor_kind: str,
        provenance: str,
    ) -> dict[str, Any]:
        with control.write(self.store) as conn:
            joined = self.join_in_txn(
                conn,
                experiment_id,
                row,
                allow_initial_winner=allow_initial_winner,
                actor=actor,
                actor_kind=actor_kind,
                provenance=provenance,
            )
        return {**joined, "winner": self.current_winner(joined["group_id"])}

    def join_in_txn(
        self,
        conn: sqlite3.Connection,
        experiment_id: str,
        row: Mapping[str, Any],
        *,
        allow_initial_winner: bool = True,
        actor: str = "fastworkflow",
        actor_kind: str = SYSTEM_ACTOR_KIND,
        provenance: str = _AUTOMATIC_PROVENANCE,
    ) -> dict[str, Any]:
        """The one join path, inside the caller's ``BEGIN IMMEDIATE``.

        Registration, `create_experiment` and the explicit enrolment of a
        decision request all come through here, so they share one membership
        and election rule.
        """
        experiment_id = _require_text(experiment_id, "experiment_id")
        actor, provenance = _require_actor(actor, actor_kind, provenance)
        control.ensure(conn)
        identity = comparison_group_identity(row)
        group_id = identity["group_id"]
        now = _utcnow_iso()
        created_at = _clean(row.get("created_at")) or now
        registered = False
        initialized = False
        member = conn.execute(
            "SELECT group_id FROM comparison_group_members WHERE experiment_id=?",
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
                   (group_id, experiment_id, benchmark_version,
                    benchmark_digest_sha256, created_at, joined_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    group_id,
                    experiment_id,
                    _clean(row.get("benchmark_version")),
                    _clean(row.get("benchmark_digest_sha256")),
                    created_at,
                    now,
                ),
            )
            registered = True
        else:
            # Membership is sticky. An experiment created unpinned and
            # re-created with a benchmark pin would derive a different
            # group; moving it would orphan a pointer that already
            # names it, so the first group it joined keeps it.
            group_id = str(member["group_id"])
        if _pointer(conn, group_id) is None and allow_initial_winner:
            # The OLDEST member takes the pointer, not the caller. A group can
            # hold members before it has a winner -- one registered with
            # `allow_initial_winner=False`, or a sole winner retired under
            # `allow_sole_winner` -- and the registration that finally elects
            # may be a newer one than the history already sitting in the group
            # (`fix-kkod`). `_order_key`, earliest wins.
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
            # NOT a hardcoded 1. A group can have history BEFORE it has a
            # winner: a member deleted in the meantime appends `retire` at
            # seq 1 (`fix-jfy5`). The election that finally arrives is the
            # next sequence number; reusing 1 violated the UNIQUE (scope,
            # group, scope_key, seq) and lost the election to a swallowed
            # IntegrityError, leaving the group winner-less for good.
            seq = _next_seq(conn, EXPERIMENT_SCOPE, group_id, "")
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
        return {
            "group_id": group_id,
            "experiment_id": experiment_id,
            "registered": registered,
            "initialized": initialized,
        }

    def _enrol_in_txn(
        self, conn: sqlite3.Connection, group_id: str, experiment_id: Optional[str]
    ) -> bool:
        """Join a recorded, not-yet-enrolled experiment a decision names (§2.3).

        The explicit enrolment of an experiment recorded before this build:
        it joins only because a decide, keep or promote request named it, and
        through the same join and election as any member. One that is not in
        `experiments`, or whose lineage is another group's, is left alone and
        refused by the caller's own membership check.
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
        the last NON-winner leaving a group that never elected, and the sole
        winner leaving under ``allow_sole_winner``.
        """
        experiment_id = _require_text(experiment_id, "experiment_id")
        actor, provenance = _require_actor(actor, actor_kind, provenance)
        rationale = _clean(rationale)
        if rationale is not None and len(rationale) > _MAX_TEXT:
            raise ValueError(f"rationale must be at most {_MAX_TEXT} characters")

        now = _utcnow_iso()
        with control.write(self.store) as conn:
            member = conn.execute(
                "SELECT * FROM comparison_group_members WHERE experiment_id=?",
                (experiment_id,),
            ).fetchone()
            group_id = None if member is None else str(member["group_id"])
            if conn.execute(
                "SELECT 1 FROM experiments WHERE experiment_id=?", (experiment_id,)
            ).fetchone() is not None:
                raise SelectionRetirementRefused(
                    experiment_id,
                    group_id,
                    "has_evidence",
                    f"experiment {experiment_id!r} has recorded evidence and cannot "
                    "be withdrawn from its comparison group",
                )
            if member is None:
                return {
                    "retired": False,
                    "experiment_id": experiment_id,
                    "group_id": None,
                    "seq": None,
                }
            pointer = _pointer(conn, group_id)
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
            seq = _next_seq(conn, EXPERIMENT_SCOPE, group_id, "")
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
        expected_selection_id: Optional[str],
        actor: str,
        actor_kind: str,
        provenance: str,
        candidate_experiment_id: Optional[str] = None,
        rationale: Optional[str] = None,
    ) -> dict[str, Any]:
        """Promote a candidate, keep the current winner, or stay undecided.

        Every decision appends to history; ONLY ``promote`` moves the pointer.
        ``expected_selection_id`` is checked for all three: a "keep" recorded
        against a winner that was replaced while the reviewer was reading is a
        judgement about a different experiment than the one it will be filed
        under, which is the same wrong answer a stale promotion gives. ``None``
        states "I read this contest and it had no winner".

        ``candidate_experiment_id`` is REQUIRED for ``promote`` and OPTIONAL for
        ``keep``/``undecided``, where it names the experiment that was being
        reviewed when the decision was made. Without it the history can say a
        winner was kept but not which challenger was rejected or deferred, which
        is most of what a reviewer wants back. It is validated for membership
        and round-tripped verbatim; on keep/undecided it moves nothing. A keep
        naming the current winner is allowed and means "re-reviewed, still it".

        A candidate recorded in this live DB before this build, and not yet a
        member, is enrolled first, in this same transaction and through the
        ordinary join — which elects it when it is the group's first member.
        A promotion that enrolment has already satisfied records nothing more.
        """
        if decision not in CLIENT_DECISIONS:
            raise ValueError(
                "decision must be one of " + ", ".join(sorted(CLIENT_DECISIONS))
            )
        actor, provenance = _require_actor(actor, actor_kind, provenance)
        expected_selection_id = _clean(expected_selection_id)
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
        with control.write(self.store) as conn:
            read = _pointer(conn, group_id)
            enrolled = self._enrol_in_txn(conn, group_id, candidate_experiment_id)
            if conn.execute(
                "SELECT 1 FROM comparison_groups WHERE group_id=?", (group_id,)
            ).fetchone() is None:
                raise UnknownComparisonGroup(group_id)
            if read is None and expected_selection_id is not None:
                raise NoCurrentSelection(
                    f"comparison group {group_id!r} has no current winner yet, "
                    "so there is nothing to promote over, keep or defer. A "
                    "group elects its earliest member when one joins; a "
                    "decision naming an experiment this workflow recorded "
                    "enrols it."
                )
            if read is not None and str(read["selection_id"]) != expected_selection_id:
                raise StaleSelection(
                    group_id=group_id,
                    expected_selection_id=expected_selection_id,
                    current_selection_id=str(read["selection_id"]),
                    current_experiment_id=str(read["experiment_id"]),
                )
            pointer = _pointer(conn, group_id) if enrolled else read
            if pointer is None:
                raise NoCurrentSelection(
                    f"comparison group {group_id!r} has no current winner yet, "
                    "so there is nothing to promote over, keep or defer."
                )
            current_selection_id = str(pointer["selection_id"])
            current_experiment_id = str(pointer["experiment_id"])
            # The enrolment just elected the experiment this promotes.
            elected = (
                read is None
                and decision == DECISION_PROMOTE
                and candidate_experiment_id == current_experiment_id
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
                    and not elected
                ):
                    raise ValueError(
                        f"experiment {candidate_experiment_id!r} is already the "
                        f"current winner; record a {DECISION_KEEP!r} decision "
                        "instead of promoting it again"
                    )
            new_experiment_id = current_experiment_id
            new_selection_id = current_selection_id
            if elected:
                # Recorded as what it is: the group's election, not a promotion.
                decision, seq = DECISION_INITIAL, int(pointer["decision_seq"])
                current_experiment_id = None
            else:
                seq = _next_seq(conn, EXPERIMENT_SCOPE, group_id, "")
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
                        current_selection_id,
                    ),
                ).rowcount
                if updated != 1:
                    raise StaleSelection(
                        group_id=group_id,
                        expected_selection_id=expected_selection_id,
                        current_selection_id=current_selection_id,
                        current_experiment_id=current_experiment_id,
                    )
            if not elected:
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
        was recorded here before this build and is not yet a member, exactly
        as `record_decision` does.
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
        """The current winner of one group, with the experiment's REAL state.

        ``automatic`` marks the winner nobody chose — the first experiment in
        the group — so a UI can label it as such instead of implying somebody
        judged it. ``experiment`` carries the live status/invalid_reason: this
        method never reports an experiment as successful, and a running or
        failed winner reads as running or failed.

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
    "CLIENT_DECISIONS",
    "DECISION_INITIAL",
    "DECISION_KEEP",
    "DECISION_PROMOTE",
    "DECISION_RETIRE",
    "DECISION_UNDECIDED",
    "EXPERIMENT_SCOPE",
    "ExperimentNotInGroup",
    "ExperimentReference",
    "GROUP_ADHOC",
    "GROUP_BENCHMARK",
    "NoCurrentSelection",
    "SELECTION_ACTOR_KINDS",
    "SYSTEM_ACTOR_KIND",
    "SelectionControlError",
    "SelectionControlStore",
    "SelectionRetirementRefused",
    "StaleSelection",
    "TASK_BEST_SCOPE",
    "ExperimentNotFound",
    "UnknownComparisonGroup",
    "comparison_group_identity",
]

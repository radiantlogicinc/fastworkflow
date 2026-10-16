"""Current winner + append-only selection history (`fix-9eg.17.1`, `fix-873a`).

Integration tests against real SQLite stores — a real `ObservabilityStore`, a
real `ReadOnlyObservabilityStore`, real sealed archives, and real concurrent
writers on one file. The contest lives in the control tables of the live DB
(`observability/control.py`). No Mock fixtures, per `.cursor/rules/testing_rules.mdc`.

The winner is plain: one contest per benchmark VERSION, zero or one winner,
set or cleared by somebody, never elected automatically. Each test names the
property it pins. Several exist to prevent a *silent wrong answer* — a winner
that claims success while its experiment is still running, two versions'
experiments sharing one winner, a judgement recorded into evidence that was
supposed to be immutable — rather than a crash.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import threading
from pathlib import Path

import pytest

from fastworkflow import state_paths
from fastworkflow.observability import control as control_module
from fastworkflow.observability import selection
from fastworkflow.observability import store as obs


# ----------------------------------------------------------------------
# Fixtures and helpers
# ----------------------------------------------------------------------


@pytest.fixture
def workflow_path(tmp_path, monkeypatch) -> str:
    monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "state"))
    wf = tmp_path / "my_workflow"
    wf.mkdir()
    return str(wf)


@pytest.fixture
def db_path(workflow_path) -> str:
    return state_paths.observability_db(workflow_path)


@pytest.fixture
def store(db_path) -> obs.ObservabilityStore:
    return obs.ObservabilityStore(db_path)


@pytest.fixture
def control(store) -> selection.SelectionControlStore:
    return selection.SelectionControlStore(store)


PIN = {
    "benchmark_id": "todo-smoke",
    "benchmark_version": "v1",
    "benchmark_digest_sha256": "a" * 64,
}


def _create(store, experiment_id, *, workflow_name="my_workflow", **pin):
    store.create_experiment(
        experiment_id,
        f"label-{experiment_id}",
        declared_tasks=1,
        declared_attempts=1,
        workflow_name=workflow_name,
        **(pin or PIN),
    )


def _create_unregistered(store, experiment_id, *, workflow_name="my_workflow", **pin):
    """Create evidence WITHOUT joining its contest.

    The shape an experiment recorded before contests lived in the live DB has:
    a row in `experiments` and no member row, until something enrols it.
    """
    store.create_experiment(
        experiment_id,
        f"label-{experiment_id}",
        declared_tasks=1,
        declared_attempts=1,
        workflow_name=workflow_name,
        join_comparison_group=False,
        **(pin or PIN),
    )


def _backdate(store, experiment_id: str, created_at: str) -> None:
    """Move an experiment's creation time, so "older" is unambiguous.

    Experiments created inside one test land in the same second, where the
    documented tie-break is the lexicographic id — correct, but it makes a test
    about chronology depend on how its ids happen to sort.
    """
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            "UPDATE experiments SET created_at=? WHERE experiment_id=?",
            (created_at, experiment_id),
        )
        conn.commit()


def _digest(path: str) -> str:
    """sha256 of a file, or a sentinel when it does not exist."""
    if not os.path.exists(path):
        return "absent"
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _settle(db_path: str) -> None:
    """Fold the WAL into the DB so a fingerprint means what it looks like.

    A WAL checkpoint rewrites the main file and truncates the `-wal` without
    changing a single row, and it happens whenever the last connection to a
    store closes — which is not under a test's control. Taking the baseline
    from a settled file is what makes "no evidence byte changed" a statement
    about writes rather than about when a connection was collected.
    """
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        conn.close()


def _evidence_fingerprint(db_path: str) -> dict[str, str]:
    """The durable evidence: the DB and its write-ahead log.

    `-shm` is deliberately NOT here. It is SQLite's shared-memory index, and a
    READER writes its read-mark into it — that is how WAL readers announce the
    snapshot they are on. Including it would assert that opening a database
    does not open it. The two files below are where rows actually live, so a
    write to evidence still shows up.
    """
    wal = f"{db_path}-wal"
    # An absent `-wal` and an empty one hold the same rows: none. Which one a
    # settled store has depends on whether some connection was still open when
    # `_settle` ran, and a read-only open recreates it empty -- neither is a write.
    no_wal_rows = not os.path.exists(wal) or os.path.getsize(wal) == 0
    return {
        "": _digest(db_path),
        "-wal": "no WAL rows" if no_wal_rows else _digest(wal),
    }


PIN_V2 = {
    "benchmark_id": PIN["benchmark_id"],
    "benchmark_version": "v2",
    "benchmark_digest_sha256": "b" * 64,
}


def _set(control, experiment_id, *, actor="reviewer", actor_kind="human",
         provenance="ui:experiments", rationale=None):
    return control.set_winner(
        experiment_id, actor=actor, actor_kind=actor_kind,
        provenance=provenance, rationale=rationale,
    )


def _clear(control, experiment_id, *, actor="reviewer", actor_kind="human",
           provenance="ui:experiments", rationale=None):
    return control.clear_winner(
        experiment_id, actor=actor, actor_kind=actor_kind,
        provenance=provenance, rationale=rationale,
    )


def _group(control, experiment_id) -> str:
    return control.group_for_experiment(experiment_id)["group_id"]


# ----------------------------------------------------------------------
# Joining a contest never makes anybody the winner
# ----------------------------------------------------------------------


class TestNoAutomaticWinner:
    def test_a_created_experiment_joins_its_group_without_a_winner(self, store, control):
        _create(store, "exp-1", **PIN)

        group_id = _group(control, "exp-1")
        assert control.current_winner(group_id) is None
        assert control.winner_for_experiment("exp-1") is None
        assert control.decision_history(group_id) == []

    def test_later_experiments_join_the_same_group(self, store, control):
        _create(store, "exp-1", **PIN)
        _create(store, "exp-2", **PIN)

        group_id = _group(control, "exp-1")
        assert _group(control, "exp-2") == group_id
        assert [m["experiment_id"] for m in control.group_members(group_id)] == [
            "exp-1",
            "exp-2",
        ]
        assert control.current_winner(group_id) is None

    def test_registration_is_idempotent_across_a_resume(self, store, control):
        """`create_experiment` re-runs on resume; it must not disturb anything."""
        _create(store, "exp-1", **PIN)
        _create(store, "exp-2", **PIN)
        _set(control, "exp-2")
        group_id = _group(control, "exp-1")

        _create(store, "exp-1", **PIN)  # the resume path

        assert control.current_winner(group_id)["experiment_id"] == "exp-2"
        assert len(control.group_members(group_id)) == 2
        assert [d["decision"] for d in control.decision_history(group_id)] == [
            selection.DECISION_SET
        ]

    def test_join_comparison_group_false_records_nothing(self, store, control):
        _create(store, "exp-quiet", join_comparison_group=False, **PIN)

        assert control.group_for_experiment("exp-quiet") is None
        assert control.list_groups() == []

    def test_a_winner_reports_running_not_success(self, store, control):
        """A winner must not read as a run that succeeded."""
        _create(store, "exp-1", **PIN)
        winner = _set(control, "exp-1")

        assert winner["experiment"]["status"] == "running"
        assert winner["experiment"]["completed_at"] is None
        assert "success" not in winner
        assert "score" not in winner
        assert "automatic" not in winner

    def test_a_failed_winner_stays_visibly_failed(self, store, control):
        _create(store, "exp-1", **PIN)
        _set(control, "exp-1")
        with store._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            store.invalidate_experiments_in_txn(
                conn, ["exp-1"], reason="writer_lost_records", detail="2 dropped"
            )
            conn.commit()

        winner = control.winner_for_experiment("exp-1")
        assert winner["experiment_id"] == "exp-1"
        assert winner["experiment"]["status"] == "invalid"
        assert winner["experiment"]["invalid_reason"] == "writer_lost_records"


# ----------------------------------------------------------------------
# Comparison-group scope: one contest per benchmark version
# ----------------------------------------------------------------------


class TestComparisonGroupScope:
    def test_two_versions_of_one_benchmark_are_two_contests(self, store, control):
        _create(store, "exp-v1", **PIN)
        _create(store, "exp-v2", **PIN_V2)

        group_v1 = control.group_for_experiment("exp-v1")
        group_v2 = control.group_for_experiment("exp-v2")
        assert group_v1["group_id"] != group_v2["group_id"]
        assert group_v1["benchmark_id"] == group_v2["benchmark_id"]
        assert [m["benchmark_version"] for m in control.group_members(group_v1["group_id"])] == ["v1"]
        assert [m["benchmark_version"] for m in control.group_members(group_v2["group_id"])] == ["v2"]

    def test_each_version_has_its_own_independent_winner(self, store, control):
        _create(store, "exp-v1", **PIN)
        _create(store, "exp-v2a", **PIN_V2)
        _create(store, "exp-v2b", **PIN_V2)

        _set(control, "exp-v1")
        _set(control, "exp-v2b")
        assert control.winner_for_experiment("exp-v1")["experiment_id"] == "exp-v1"
        assert control.winner_for_experiment("exp-v2a")["experiment_id"] == "exp-v2b"

        _clear(control, "exp-v2a")
        assert control.winner_for_experiment("exp-v2b") is None
        assert control.winner_for_experiment("exp-v1")["experiment_id"] == "exp-v1"

    def test_different_benchmarks_are_different_contests(self, store, control):
        _create(store, "exp-a", **PIN)
        _create(
            store,
            "exp-b",
            benchmark_id="other-bench",
            benchmark_version="v1",
            benchmark_digest_sha256="c" * 64,
        )

        assert _group(control, "exp-a") != _group(control, "exp-b")
        _set(control, "exp-a")
        assert control.winner_for_experiment("exp-b") is None

    def test_an_unpinned_experiment_has_no_contest(self, store):
        """Every experiment pins a benchmark version (`fix-ujat`)."""
        with pytest.raises(ValueError, match="no benchmark"):
            selection.comparison_group_identity({"workflow_name": "wf"})
        with pytest.raises(obs.MissingBenchmarkPin):
            store.create_experiment("adhoc-1", "", declared_tasks=1, declared_attempts=1)

    def test_group_identity_is_derived_not_minted(self):
        """Two processes must compute the same id, or each opens its own contest."""
        row = {"workflow_name": "wf", "benchmark_id": "bench", "benchmark_version": "v1"}
        identity = selection.comparison_group_identity(row)
        assert identity["group_id"] == selection.comparison_group_identity(dict(row))["group_id"]
        assert identity["group_kind"] == selection.GROUP_BENCHMARK
        assert identity["benchmark_version"] == "v1"
        other = selection.comparison_group_identity(dict(row, benchmark_version="v2"))
        assert other["group_id"] != identity["group_id"]


# ----------------------------------------------------------------------
# Setting and clearing the winner
# ----------------------------------------------------------------------


class TestSetAndClear:
    def test_set_names_the_winner_and_records_who(self, store, control):
        _create(store, "exp-1", **PIN)

        winner = _set(control, "exp-1", actor="dhar", rationale="best answers")

        assert winner["experiment_id"] == "exp-1"
        assert winner["decision"] == selection.DECISION_SET
        [row] = control.decision_history(_group(control, "exp-1"))
        assert row["decision"] == selection.DECISION_SET
        assert row["previous_experiment_id"] is None
        assert row["new_experiment_id"] == "exp-1"
        assert row["actor"] == "dhar"
        assert row["actor_kind"] == "human"
        assert row["rationale"] == "best answers"

    def test_set_replaces_the_previous_winner(self, store, control):
        _create(store, "exp-1", **PIN)
        _create(store, "exp-2", **PIN)
        first = _set(control, "exp-1")

        second = _set(control, "exp-2")

        assert second["experiment_id"] == "exp-2"
        assert second["selection_id"] != first["selection_id"]
        history = control.decision_history(_group(control, "exp-1"))
        assert [(d["decision"], d["previous_experiment_id"], d["new_experiment_id"])
                for d in history] == [
            (selection.DECISION_SET, "exp-1", "exp-2"),
            (selection.DECISION_SET, None, "exp-1"),
        ]

    def test_setting_the_current_winner_again_changes_nothing(self, store, control):
        _create(store, "exp-1", **PIN)
        first = _set(control, "exp-1")

        again = _set(control, "exp-1")

        assert again["selection_id"] == first["selection_id"]
        assert len(control.decision_history(_group(control, "exp-1"))) == 1

    def test_clear_leaves_the_contest_with_no_winner(self, store, control):
        _create(store, "exp-1", **PIN)
        _create(store, "exp-2", **PIN)
        _set(control, "exp-1")

        # Cleared from ANY member's side: it removes the contest's winner.
        assert _clear(control, "exp-2", rationale="none is good enough") is None

        group_id = _group(control, "exp-1")
        assert control.current_winner(group_id) is None
        latest = control.decision_history(group_id)[0]
        assert latest["decision"] == selection.DECISION_CLEAR
        assert latest["previous_experiment_id"] == "exp-1"
        assert latest["new_experiment_id"] is None
        assert latest["rationale"] == "none is good enough"
        # Membership is untouched; the experiments can be chosen again.
        assert len(control.group_members(group_id)) == 2
        assert _set(control, "exp-2")["experiment_id"] == "exp-2"

    def test_clearing_a_contest_with_no_winner_records_nothing(self, store, control):
        _create(store, "exp-1", **PIN)

        _clear(control, "exp-1")

        assert control.decision_history(_group(control, "exp-1")) == []

    def test_human_and_agent_decisions_share_one_api(self, store, control):
        _create(store, "exp-1", **PIN)
        _create(store, "exp-2", **PIN)

        _set(control, "exp-1", actor="dhar", actor_kind="human")
        _set(control, "exp-2", actor="claude", actor_kind="coding_agent",
             provenance="mcp:set_winner")

        assert control.winner_for_experiment("exp-1")["experiment_id"] == "exp-2"
        assert [d["actor_kind"] for d in control.decision_history(_group(control, "exp-1"))] == [
            "coding_agent",
            "human",
        ]

    def test_rationale_is_optional(self, store, control):
        _create(store, "exp-1", **PIN)
        _set(control, "exp-1")

        assert control.decision_history(_group(control, "exp-1"))[0]["rationale"] is None

    def test_invalid_actor_kind_is_refused(self, store, control):
        _create(store, "exp-1", **PIN)

        with pytest.raises(ValueError):
            _set(control, "exp-1", actor_kind="robot")
        with pytest.raises(ValueError):
            _clear(control, "exp-1", actor_kind="robot")
        with pytest.raises(ValueError):
            _set(control, "exp-1", actor="")
        assert control.winner_for_experiment("exp-1") is None

    def test_an_unknown_experiment_is_reported_as_such(self, store, control):
        _create(store, "exp-1", **PIN)  # the control tables exist

        with pytest.raises(selection.ExperimentNotFound):
            _set(control, "exp-nobody")
        with pytest.raises(selection.ExperimentNotFound):
            _clear(control, "exp-nobody")

    def test_history_is_append_only(self, store, control, db_path):
        _create(store, "exp-1", **PIN)
        _set(control, "exp-1")

        with sqlite3.connect(db_path) as conn:
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                conn.execute("UPDATE selection_decisions SET decision='forged'")
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                conn.execute("DELETE FROM selection_decisions")


class TestLeavingTheContest:
    def test_a_member_that_leaves_takes_its_winner_pointer_with_it(self, store, control):
        _create(store, "exp-1", **PIN)
        _create(store, "exp-2", **PIN)
        _set(control, "exp-1")
        group_id = _group(control, "exp-1")

        with control_module.write(store) as conn:
            control.leave_in_txn(conn, "exp-1")

        assert control.group_for_experiment("exp-1") is None
        assert control.current_winner(group_id) is None
        assert [m["experiment_id"] for m in control.group_members(group_id)] == ["exp-2"]
        # History is not rewritten.
        assert len(control.decision_history(group_id)) == 1

    def test_a_non_winner_leaving_keeps_the_winner(self, store, control):
        _create(store, "exp-1", **PIN)
        _create(store, "exp-2", **PIN)
        _set(control, "exp-1")

        with control_module.write(store) as conn:
            control.leave_in_txn(conn, "exp-2")
            control.leave_in_txn(conn, "exp-never-joined")  # a no-op

        assert control.winner_for_experiment("exp-1")["experiment_id"] == "exp-1"


# ----------------------------------------------------------------------
# Concurrency
# ----------------------------------------------------------------------


class TestConcurrency:
    def test_simultaneous_creations_join_one_contest_and_elect_nobody(self, store, db_path):
        """Eight writers, one contest, no winner.

        The `store` fixture installs the evidence schema first: concurrent
        *schema creation* on a fresh file is a separate, pre-existing race
        (`fix-upxs`), and this test is about the contest, not about it.
        """
        count = 8
        barrier = threading.Barrier(count)
        errors: list[BaseException] = []

        def create(index: int) -> None:
            writer = obs.ObservabilityStore(db_path)
            try:
                barrier.wait(timeout=30)
                _create(writer, f"exp-{index}", **PIN)
            except BaseException as exc:  # pragma: no cover - surfaced below
                errors.append(exc)

        threads = [threading.Thread(target=create, args=(i,)) for i in range(count)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)

        assert not errors, errors
        control = selection.SelectionControlStore(obs.ObservabilityStore(db_path))
        groups = control.list_groups()
        assert len(groups) == 1
        group_id = groups[0]["group_id"]
        assert len(control.group_members(group_id)) == count
        assert control.current_winner(group_id) is None
        assert control.decision_history(group_id) == []

    def test_simultaneous_sets_leave_exactly_one_winner(self, store, db_path):
        """The last write wins, and the pointer names one of the two."""
        _create(store, "exp-1", **PIN)
        _create(store, "exp-2", **PIN)
        _create(store, "exp-3", **PIN)
        control = selection.SelectionControlStore(store)
        group_id = _group(control, "exp-1")

        barrier = threading.Barrier(2)
        errors: list[BaseException] = []

        def set_winner(candidate: str) -> None:
            writer = selection.SelectionControlStore(obs.ObservabilityStore(db_path))
            barrier.wait(timeout=30)
            try:
                _set(writer, candidate)
            except BaseException as exc:  # pragma: no cover - surfaced below
                errors.append(exc)

        threads = [
            threading.Thread(target=set_winner, args=(candidate,))
            for candidate in ("exp-2", "exp-3")
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)

        assert not errors, errors
        history = control.decision_history(group_id)
        assert [d["decision"] for d in history] == [selection.DECISION_SET] * 2
        assert [d["seq"] for d in history] == [2, 1]
        assert control.current_winner(group_id)["experiment_id"] == (
            history[0]["new_experiment_id"]
        )


# ----------------------------------------------------------------------
# Sealed evidence stays immutable; judgements live in the live DB
# ----------------------------------------------------------------------


class TestEvidenceImmutability:
    def test_a_sealed_archive_keeps_its_own_bytes_while_being_judged(
        self, store, control, tmp_path
    ):
        """The archive is the evidence; the live DB carries the judgement.

        A copy sealed AFTER decisions carries none of them (§3 Rule 1), so a
        later change of winner cannot change the file or its digest.
        """
        _create(store, "exp-1", **PIN)
        _create(store, "exp-2", **PIN)
        _create(store, "exp-3", **PIN)
        _set(control, "exp-2")
        archive = tmp_path / "archive" / "sealed.sqlite3"
        store.archive_to(str(archive))
        sealed_digest = _digest(str(archive))

        with sqlite3.connect(str(archive)) as conn:
            tables = {row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )}
        assert not set(control_module.CONTROL_TABLES) & tables
        assert not control_module.present(obs.ReadOnlyObservabilityStore(str(archive)))

        _set(control, "exp-3")
        _clear(control, "exp-3")

        assert control.winner_for_experiment("exp-1") is None
        assert _digest(str(archive)) == sealed_digest

    def test_a_read_of_a_store_without_control_tables_writes_nothing(
        self, store, db_path
    ):
        _create(store, "exp-1", **PIN)
        with store._connect() as conn:
            control_module.strip(conn)
        del store
        _settle(db_path)
        before = _evidence_fingerprint(db_path)

        reader = selection.SelectionControlStore(obs.ReadOnlyObservabilityStore(db_path))
        assert reader.group_for_experiment("exp-1") is None
        assert reader.list_groups() == []
        assert reader.winner_for_experiment("exp-1") is None

        assert _evidence_fingerprint(db_path) == before
        assert not control_module.present(obs.ReadOnlyObservabilityStore(db_path))


# ----------------------------------------------------------------------
# The winner and the per-task best run never touch each other
# ----------------------------------------------------------------------


class TestScopes:
    def test_task_best_scope_cannot_overwrite_the_experiment_winner(
        self, store, control, db_path
    ):
        """A task-best pointer lives under a different key; neither scope can
        clobber the other. Raw SQL, because the guarantee pinned is the
        schema's, not `best_run`'s."""
        _create(store, "exp-1", **PIN)
        group_id = _group(control, "exp-1")
        before = _set(control, "exp-1")

        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """INSERT INTO selection_pointers
                   (scope_kind, group_id, scope_key, experiment_id, task_id,
                    attempt, selection_id, decision, decision_seq, decided_at)
                   VALUES (?, ?, 't1', 'exp-1', 't1', 3, 'sel-x', 'select', 1,
                           '2026-01-01T00:00:00Z')""",
                (selection.TASK_BEST_SCOPE, group_id),
            )
            conn.commit()

        after = control.current_winner(group_id)
        assert after["selection_id"] == before["selection_id"]
        assert after["scope_kind"] == selection.EXPERIMENT_SCOPE

        _clear(control, "exp-1")
        best = control.scoped_pointer(
            scope_kind=selection.TASK_BEST_SCOPE, group_id=group_id, scope_key="t1"
        )
        assert best["selection_id"] == "sel-x"

    def test_a_scoped_decision_cannot_name_the_winner_scope(self, store, control):
        _create(store, "exp-1", **PIN)
        with pytest.raises(ValueError, match="set_winner"):
            control.apply_scoped_decision(
                scope_kind=selection.EXPERIMENT_SCOPE,
                group_id=_group(control, "exp-1"),
                scope_key="t1",
                decision="select",
                expected_selection_id=None,
                actor="a",
                actor_kind="human",
                provenance="p",
            )

    def test_a_read_only_reader_changes_no_evidence_byte(self, store, db_path):
        _create(store, "exp-1", **PIN)
        _set(selection.SelectionControlStore(store), "exp-1")
        del store
        _settle(db_path)
        before = _evidence_fingerprint(db_path)

        control = selection.SelectionControlStore(obs.ReadOnlyObservabilityStore(db_path))
        group_id = _group(control, "exp-1")

        assert control.current_winner(group_id)["experiment_id"] == "exp-1"
        assert _evidence_fingerprint(db_path) == before


# ----------------------------------------------------------------------
# Experiments recorded before contests lived in the live DB
# ----------------------------------------------------------------------


class TestPreExistingExperiments:
    def test_nothing_adopts_older_runs_automatically(self, store, control):
        """An older experiment joins only when a decision names it."""
        _create_unregistered(store, "old-1", **PIN)
        _backdate(store, "old-1", "2026-01-01T00:00:00Z")

        _create(store, "new-1", **PIN)

        assert control.group_for_experiment("old-1") is None
        assert [m["experiment_id"] for m in control.group_members(_group(control, "new-1"))] == [
            "new-1"
        ]

    def test_setting_an_unenrolled_experiment_enrols_it_into_its_version(
        self, store, control
    ):
        _create_unregistered(store, "old-1", **PIN)
        _create(store, "new-1", **PIN)
        _create(store, "other-version", **PIN_V2)

        winner = _set(control, "old-1")

        assert winner["experiment_id"] == "old-1"
        assert _group(control, "old-1") == _group(control, "new-1")
        assert control.winner_for_experiment("other-version") is None

    def test_clearing_from_an_unenrolled_experiment_enrols_it(self, store, control):
        _create_unregistered(store, "old-1", **PIN)
        _create(store, "new-1", **PIN)
        _set(control, "new-1")

        _clear(control, "old-1")

        assert _group(control, "old-1") == _group(control, "new-1")
        assert control.winner_for_experiment("new-1") is None


# ----------------------------------------------------------------------
# Registration at UI-creation time, recording at execution time
# ----------------------------------------------------------------------


class TestRegistrationBeforeAnyRun:
    def test_a_reference_joins_without_winning_and_can_be_set(self, store, control):
        result = control.register_experiment_reference(
            selection.ExperimentReference(
                experiment_id="exp-planned",
                workflow_name="my_workflow",
                benchmark_id=PIN["benchmark_id"],
                benchmark_version=PIN["benchmark_version"],
            )
        )
        assert result["winner"] is None
        assert result["registered"] is True

        winner = _set(control, "exp-planned")
        # The decision is real; the run's state is simply not knowable yet.
        assert winner["experiment"] is None
        assert winner["experiment_resolved"] is False

    def test_recording_it_later_changes_no_selection(self, store, control):
        control.register_experiment_reference(
            selection.ExperimentReference(
                experiment_id="exp-planned",
                workflow_name="my_workflow",
                benchmark_id=PIN["benchmark_id"],
                benchmark_version=PIN["benchmark_version"],
            )
        )
        before = _set(control, "exp-planned")

        _create(store, "exp-planned", **PIN)

        after = control.winner_for_experiment("exp-planned")
        assert after["selection_id"] == before["selection_id"]
        assert after["experiment_resolved"] is True
        assert after["experiment"]["status"] == "running"
        assert len(control.list_groups()) == 1

    def test_a_write_with_no_live_db_refuses_and_creates_nothing(self, tmp_path):
        missing = tmp_path / "nowhere" / "observability.sqlite3"
        with pytest.raises(control_module.ControlUnavailable):
            obs.open_live_store(str(missing), write=True)
        assert obs.open_live_store(str(missing)) is None
        absent = selection.SelectionControlStore(None)
        assert absent.list_groups() == []
        with pytest.raises(control_module.ControlUnavailable):
            absent.register_experiment_reference(
                selection.ExperimentReference(experiment_id="e", workflow_name="w")
            )
        with pytest.raises(control_module.ControlUnavailable):
            _set(absent, "e")
        assert not missing.parent.exists()


# ----------------------------------------------------------------------
# The control tables' own creation on a store opened without migrating
# ----------------------------------------------------------------------


class TestControlSchemaCreation:
    def test_concurrent_registration_creates_the_tables_once(self, store, db_path):
        """A writer on a `migrate=False` store creates the control tables in its
        own `BEGIN IMMEDIATE`, so six racing first writers neither fail nor
        open two groups."""
        for index in range(6):
            _create_unregistered(store, f"exp-{index}", **PIN)
        with store._connect() as conn:
            control_module.strip(conn)
        barrier = threading.Barrier(6)
        errors: list[BaseException] = []

        def register(index: int) -> None:
            try:
                writer = selection.SelectionControlStore(
                    obs.ObservabilityStore(db_path, migrate=False)
                )
                barrier.wait(timeout=30)
                writer.register_experiment(f"exp-{index}")
            except BaseException as exc:  # pragma: no cover - surfaced below
                errors.append(exc)

        threads = [threading.Thread(target=register, args=(i,)) for i in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)

        assert not errors, errors
        control = selection.SelectionControlStore(store)
        assert control_module.present(store)
        [group] = control.list_groups()
        assert len(control.group_members(group["group_id"])) == 6
        assert control.current_winner(group["group_id"]) is None

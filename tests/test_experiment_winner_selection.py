"""Current winner + append-only selection history (`fix-9eg.17.1`).

Integration tests against real SQLite stores — a real `ObservabilityStore`, a
real `ReadOnlyObservabilityStore`, real sealed archives, and real concurrent
writers on one file. The contest lives in the control tables of the live DB
(`observability/control.py`). No Mock fixtures, per `.cursor/rules/testing_rules.mdc`.

Each test names the property it pins. Several of these exist to prevent a
*silent wrong answer* — a winner that claims success while its experiment is
still running, a promotion that overwrites a decision it never saw, a judgement
recorded into evidence that was supposed to be immutable — rather than a crash,
so a reader who does not know the invariant will read them as redundant.
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
        **pin,
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
        initialize_winner=False,
        **pin,
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


def _promote(control, group_id, candidate, *, actor="reviewer", rationale=None):
    winner = control.current_winner(group_id)
    return control.record_decision(
        group_id,
        selection.DECISION_PROMOTE,
        expected_selection_id=winner["selection_id"],
        candidate_experiment_id=candidate,
        actor=actor,
        actor_kind="human",
        provenance="ui:experiments",
        rationale=rationale,
    )


# ----------------------------------------------------------------------
# The first experiment wins automatically, at creation
# ----------------------------------------------------------------------


class TestAutomaticInitialWinner:
    def test_first_experiment_becomes_winner_at_creation(self, store, control):
        _create(store, "exp-1", **PIN)

        winner = control.winner_for_experiment("exp-1")
        assert winner["experiment_id"] == "exp-1"
        assert winner["decision"] == selection.DECISION_INITIAL
        assert winner["automatic"] is True

    def test_initial_winner_reports_running_not_success(self, store, control):
        """A winner nobody judged must not read as a run that succeeded."""
        _create(store, "exp-1", **PIN)

        winner = control.winner_for_experiment("exp-1")
        assert winner["experiment"]["status"] == "running"
        assert winner["experiment"]["completed_at"] is None
        # There is no "successful"/"passed" claim anywhere in the winner record:
        # the only verdict it carries is the experiment's own status.
        assert "success" not in winner
        assert "score" not in winner

    def test_failed_first_experiment_stays_the_visible_winner(self, store, control):
        _create(store, "exp-1", **PIN)
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

    def test_second_experiment_joins_the_group_without_winning(self, store, control):
        _create(store, "exp-1", **PIN)
        _create(store, "exp-2", **PIN)

        group_id = control.group_for_experiment("exp-1")["group_id"]
        assert control.current_winner(group_id)["experiment_id"] == "exp-1"
        assert [m["experiment_id"] for m in control.group_members(group_id)] == [
            "exp-1",
            "exp-2",
        ]

    def test_registration_is_idempotent_across_a_resume(self, store, control):
        """`create_experiment` re-runs on resume; it must not re-initialize."""
        _create(store, "exp-1", **PIN)
        _create(store, "exp-2", **PIN)
        group_id = control.group_for_experiment("exp-1")["group_id"]
        _promote(control, group_id, "exp-2")

        _create(store, "exp-1", **PIN)  # the resume path

        assert control.current_winner(group_id)["experiment_id"] == "exp-2"
        decisions = control.decision_history(group_id)
        assert [d["decision"] for d in decisions] == [
            selection.DECISION_PROMOTE,
            selection.DECISION_INITIAL,
        ]

    def test_initialize_winner_false_records_nothing(self, store, control):
        _create(store, "exp-quiet", initialize_winner=False, **PIN)

        assert control.group_for_experiment("exp-quiet") is None
        assert control.list_groups() == []


# ----------------------------------------------------------------------
# Comparison-group scope: benchmark lineage, or an automatic ad-hoc group
# ----------------------------------------------------------------------


class TestComparisonGroupScope:
    def test_lineage_spans_benchmark_versions(self, store, control):
        """A new benchmark VERSION continues the contest; it does not restart it."""
        _create(store, "exp-v1", **PIN)
        _create(
            store,
            "exp-v2",
            benchmark_id=PIN["benchmark_id"],
            benchmark_version="v2",
            benchmark_digest_sha256="b" * 64,
        )

        group_id = control.group_for_experiment("exp-v1")["group_id"]
        assert control.group_for_experiment("exp-v2")["group_id"] == group_id
        assert control.current_winner(group_id)["experiment_id"] == "exp-v1"
        # The version change stays visible on the member rows.
        versions = {
            m["experiment_id"]: m["benchmark_version"]
            for m in control.group_members(group_id)
        }
        assert versions == {"exp-v1": "v1", "exp-v2": "v2"}

    def test_different_benchmarks_are_different_contests(self, store, control):
        _create(store, "exp-a", **PIN)
        _create(
            store,
            "exp-b",
            benchmark_id="other-bench",
            benchmark_version="v1",
            benchmark_digest_sha256="c" * 64,
        )

        group_a = control.group_for_experiment("exp-a")["group_id"]
        group_b = control.group_for_experiment("exp-b")["group_id"]
        assert group_a != group_b
        assert control.current_winner(group_a)["experiment_id"] == "exp-a"
        assert control.current_winner(group_b)["experiment_id"] == "exp-b"

    def test_unpinned_runs_share_one_automatic_adhoc_group(self, store, control):
        _create(store, "adhoc-1")
        _create(store, "adhoc-2")

        group = control.group_for_experiment("adhoc-1")
        assert group["group_kind"] == selection.GROUP_ADHOC
        assert control.group_for_experiment("adhoc-2")["group_id"] == group["group_id"]
        assert control.current_winner(group["group_id"])["experiment_id"] == "adhoc-1"

    def test_adhoc_groups_are_per_workflow(self, store, control):
        _create(store, "adhoc-1", workflow_name="alpha")
        _create(store, "adhoc-2", workflow_name="beta")

        assert (
            control.group_for_experiment("adhoc-1")["group_id"]
            != control.group_for_experiment("adhoc-2")["group_id"]
        )

    def test_group_identity_is_derived_not_minted(self):
        """Two processes must compute the same id, or each gets its own winner."""
        row = {"workflow_name": "wf", "benchmark_id": "bench"}
        assert (
            selection.comparison_group_identity(row)["group_id"]
            == selection.comparison_group_identity(dict(row))["group_id"]
        )
        assert selection.comparison_group_identity(row)["group_kind"] == (
            selection.GROUP_BENCHMARK
        )

    def test_membership_is_sticky_when_a_pin_arrives_later(self, store, control):
        """A later pin must not move an experiment out from under a pointer."""
        _create(store, "exp-1")
        adhoc_group = control.group_for_experiment("exp-1")["group_id"]

        _create(store, "exp-1", **PIN)  # re-create adds the write-once pin

        assert control.group_for_experiment("exp-1")["group_id"] == adhoc_group
        assert control.current_winner(adhoc_group)["experiment_id"] == "exp-1"


# ----------------------------------------------------------------------
# Decisions: promote moves the pointer, keep/undecided do not
# ----------------------------------------------------------------------


class TestDecisions:
    def test_promotion_moves_the_pointer_and_records_both_references(
        self, store, control
    ):
        _create(store, "exp-1", **PIN)
        _create(store, "exp-2", **PIN)
        group_id = control.group_for_experiment("exp-1")["group_id"]

        state = _promote(control, group_id, "exp-2", rationale="cleaner plans")

        assert state["experiment_id"] == "exp-2"
        assert state["automatic"] is False
        latest = control.decision_history(group_id)[0]
        assert latest["decision"] == selection.DECISION_PROMOTE
        assert latest["previous_experiment_id"] == "exp-1"
        assert latest["candidate_experiment_id"] == "exp-2"
        assert latest["new_experiment_id"] == "exp-2"
        assert latest["actor"] == "reviewer"
        assert latest["actor_kind"] == "human"
        assert latest["provenance"] == "ui:experiments"
        assert latest["rationale"] == "cleaner plans"
        assert latest["created_at"]

    @pytest.mark.parametrize(
        "decision", [selection.DECISION_KEEP, selection.DECISION_UNDECIDED]
    )
    def test_keep_and_undecided_append_history_without_moving_the_winner(
        self, store, control, decision
    ):
        _create(store, "exp-1", **PIN)
        _create(store, "exp-2", **PIN)
        group_id = control.group_for_experiment("exp-1")["group_id"]
        before = control.current_winner(group_id)

        control.record_decision(
            group_id,
            decision,
            expected_selection_id=before["selection_id"],
            actor="agent-7",
            actor_kind="coding_agent",
            provenance="mcp:record_winner_decision",
            rationale="not enough evidence yet",
        )

        after = control.current_winner(group_id)
        assert after["experiment_id"] == before["experiment_id"]
        assert after["selection_id"] == before["selection_id"]
        assert after["decision"] == selection.DECISION_INITIAL
        history = control.decision_history(group_id)
        assert [d["decision"] for d in history] == [
            decision,
            selection.DECISION_INITIAL,
        ]
        assert history[0]["new_experiment_id"] == "exp-1"

    def test_human_and_agent_decisions_share_one_api(self, store, control):
        _create(store, "exp-1", **PIN)
        _create(store, "exp-2", **PIN)
        group_id = control.group_for_experiment("exp-1")["group_id"]

        control.record_decision(
            group_id,
            selection.DECISION_KEEP,
            expected_selection_id=control.current_winner(group_id)["selection_id"],
            actor="dhar",
            actor_kind="human",
            provenance="ui:experiments",
        )
        control.record_decision(
            group_id,
            selection.DECISION_PROMOTE,
            expected_selection_id=control.current_winner(group_id)["selection_id"],
            candidate_experiment_id="exp-2",
            actor="claude",
            actor_kind="coding_agent",
            provenance="mcp:record_winner_decision",
        )

        assert control.current_winner(group_id)["experiment_id"] == "exp-2"
        assert [d["actor_kind"] for d in control.decision_history(group_id)] == [
            "coding_agent",
            "human",
            selection.SYSTEM_ACTOR_KIND,
        ]

    def test_rationale_is_optional(self, store, control):
        _create(store, "exp-1", **PIN)
        _create(store, "exp-2", **PIN)
        group_id = control.group_for_experiment("exp-1")["group_id"]

        _promote(control, group_id, "exp-2")

        assert control.decision_history(group_id)[0]["rationale"] is None

    def test_a_non_member_cannot_be_promoted(self, store, control):
        _create(store, "exp-1", **PIN)
        _create(store, "other", benchmark_id="x", benchmark_version="v1",
                benchmark_digest_sha256="d" * 64)
        group_id = control.group_for_experiment("exp-1")["group_id"]

        with pytest.raises(selection.ExperimentNotInGroup):
            _promote(control, group_id, "other")

    def test_promoting_the_current_winner_is_refused_as_a_keep(self, store, control):
        _create(store, "exp-1", **PIN)
        group_id = control.group_for_experiment("exp-1")["group_id"]

        with pytest.raises(ValueError, match="already the current winner"):
            _promote(control, group_id, "exp-1")

    def test_invalid_actor_kind_and_decision_are_refused(self, store, control):
        _create(store, "exp-1", **PIN)
        group_id = control.group_for_experiment("exp-1")["group_id"]
        selection_id = control.current_winner(group_id)["selection_id"]

        with pytest.raises(ValueError):
            control.record_decision(
                group_id,
                "deploy",
                expected_selection_id=selection_id,
                actor="a",
                actor_kind="human",
                provenance="p",
            )
        with pytest.raises(ValueError):
            control.record_decision(
                group_id,
                selection.DECISION_KEEP,
                expected_selection_id=selection_id,
                actor="a",
                actor_kind="robot",
                provenance="p",
            )

    def test_unknown_group_is_reported_as_such(self, control):
        with pytest.raises(selection.UnknownComparisonGroup):
            control.record_decision(
                "benchmark-0000000000000000",
                selection.DECISION_KEEP,
                expected_selection_id="whatever",
                actor="a",
                actor_kind="human",
                provenance="p",
            )


# ----------------------------------------------------------------------
# Staleness: a decision about a winner that has since moved
# ----------------------------------------------------------------------


class TestStaleSelection:
    def test_stale_promotion_is_refused_with_what_is_current(self, store, control):
        _create(store, "exp-1", **PIN)
        _create(store, "exp-2", **PIN)
        _create(store, "exp-3", **PIN)
        group_id = control.group_for_experiment("exp-1")["group_id"]
        stale = control.current_winner(group_id)["selection_id"]
        _promote(control, group_id, "exp-2")

        with pytest.raises(selection.StaleSelection) as caught:
            control.record_decision(
                group_id,
                selection.DECISION_PROMOTE,
                expected_selection_id=stale,
                candidate_experiment_id="exp-3",
                actor="reviewer",
                actor_kind="human",
                provenance="ui:experiments",
            )

        error = caught.value
        assert error.expected_selection_id == stale
        assert error.current_experiment_id == "exp-2"
        assert "exp-2" in str(error)
        # The refusal left both the pointer and the history alone.
        assert control.current_winner(group_id)["experiment_id"] == "exp-2"
        assert len(control.decision_history(group_id)) == 2

    def test_stale_keep_is_refused_too(self, store, control):
        """A judgement filed against a replaced winner is about another run."""
        _create(store, "exp-1", **PIN)
        _create(store, "exp-2", **PIN)
        group_id = control.group_for_experiment("exp-1")["group_id"]
        stale = control.current_winner(group_id)["selection_id"]
        _promote(control, group_id, "exp-2")

        with pytest.raises(selection.StaleSelection):
            control.record_decision(
                group_id,
                selection.DECISION_KEEP,
                expected_selection_id=stale,
                actor="reviewer",
                actor_kind="human",
                provenance="ui:experiments",
            )


# ----------------------------------------------------------------------
# Real concurrency on one store file
# ----------------------------------------------------------------------


class TestConcurrency:
    def test_simultaneous_first_creations_produce_one_winner(self, store, db_path):
        """Eight writers, one contest, one initial decision.

        The `store` fixture installs the evidence schema first: concurrent
        *schema creation* on a fresh file is a separate, pre-existing race
        (`fix-upxs`), and this test is about the winner, not about it.
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
        history = control.decision_history(group_id)
        assert [d["decision"] for d in history] == [selection.DECISION_INITIAL]
        assert control.current_winner(group_id)["experiment_id"] == (
            history[0]["new_experiment_id"]
        )

    def test_simultaneous_promotions_leave_exactly_one_winner(self, store, db_path):
        """Both readers saw the same winner; only one promotion may land."""
        _create(store, "exp-1", **PIN)
        _create(store, "exp-2", **PIN)
        _create(store, "exp-3", **PIN)
        control = selection.SelectionControlStore(store)
        group_id = control.group_for_experiment("exp-1")["group_id"]
        expected = control.current_winner(group_id)["selection_id"]

        barrier = threading.Barrier(2)
        outcomes: list[str] = []
        stale: list[selection.StaleSelection] = []

        def promote(candidate: str) -> None:
            writer = selection.SelectionControlStore(obs.ObservabilityStore(db_path))
            barrier.wait(timeout=30)
            try:
                writer.record_decision(
                    group_id,
                    selection.DECISION_PROMOTE,
                    expected_selection_id=expected,
                    candidate_experiment_id=candidate,
                    actor="reviewer",
                    actor_kind="human",
                    provenance="ui:experiments",
                )
                outcomes.append(candidate)
            except selection.StaleSelection as exc:
                stale.append(exc)

        threads = [
            threading.Thread(target=promote, args=(candidate,))
            for candidate in ("exp-2", "exp-3")
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)

        assert len(outcomes) == 1, (outcomes, stale)
        assert len(stale) == 1
        assert control.current_winner(group_id)["experiment_id"] == outcomes[0]
        assert [d["decision"] for d in control.decision_history(group_id)] == [
            selection.DECISION_PROMOTE,
            selection.DECISION_INITIAL,
        ]


# ----------------------------------------------------------------------
# Sealed evidence stays immutable; judgements live in the live DB
# ----------------------------------------------------------------------


class TestEvidenceImmutability:
    def test_a_sealed_archive_keeps_its_own_bytes_while_being_judged(
        self, store, control, tmp_path
    ):
        """The archive is the evidence; the live DB carries the judgement.

        A copy sealed AFTER decisions carries none of them (§3 Rule 1), so a
        later promotion cannot change the file or its digest.
        """
        _create(store, "exp-1", **PIN)
        _create(store, "exp-2", **PIN)
        _create(store, "exp-3", **PIN)
        group_id = control.group_for_experiment("exp-1")["group_id"]
        _promote(control, group_id, "exp-2")
        archive = tmp_path / "archive" / "sealed.sqlite3"
        store.archive_to(str(archive))
        sealed_digest = _digest(str(archive))

        with sqlite3.connect(str(archive)) as conn:
            tables = {row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )}
        assert not set(control_module.CONTROL_TABLES) & tables
        assert not control_module.present(obs.ReadOnlyObservabilityStore(str(archive)))

        _promote(control, group_id, "exp-3")

        assert control.current_winner(group_id)["experiment_id"] == "exp-3"
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
# The scope the next bead will use
# ----------------------------------------------------------------------


class TestAdoptionAndScope:
    def test_task_best_scope_cannot_overwrite_the_experiment_winner(
        self, store, control, db_path
    ):
        """`fix-9eg.17.4` writes a different key; neither scope can clobber the
        other. Written here as raw SQL because this bead does not implement the
        task-best API — the guarantee being pinned is the schema's, and it has
        to hold before the second writer exists."""
        _create(store, "exp-1", **PIN)
        group_id = control.group_for_experiment("exp-1")["group_id"]
        before = control.current_winner(group_id)

        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """INSERT INTO selection_pointers
                   (scope_kind, group_id, scope_key, experiment_id, task_id,
                    attempt, selection_id, decision, decision_seq, decided_at)
                   VALUES (?, ?, 't1', 'exp-1', 't1', 3, 'sel-x', 'promote', 1,
                           '2026-01-01T00:00:00Z')""",
                (selection.TASK_BEST_SCOPE, group_id),
            )
            conn.commit()

        after = control.current_winner(group_id)
        assert after["selection_id"] == before["selection_id"]
        assert after["experiment_id"] == "exp-1"
        assert after["scope_kind"] == selection.EXPERIMENT_SCOPE
        assert [d["decision"] for d in control.decision_history(group_id)] == [
            selection.DECISION_INITIAL
        ]

    def test_a_read_only_reader_changes_no_evidence_byte(self, store, db_path):
        _create(store, "exp-1", **PIN)
        del store
        _settle(db_path)
        before = _evidence_fingerprint(db_path)

        control = selection.SelectionControlStore(obs.ReadOnlyObservabilityStore(db_path))
        group_id = control.group_for_experiment("exp-1")["group_id"]

        assert control.current_winner(group_id)["experiment_id"] == "exp-1"
        assert _evidence_fingerprint(db_path) == before


# ----------------------------------------------------------------------
# A candidate reference on keep/undecided: WHICH experiment was rejected
# ----------------------------------------------------------------------


class TestCandidateOnNonPromotingDecisions:
    """A reviewer who declines a challenger is judging that challenger.

    Without the reference, history records "the winner was kept" and loses the
    only fact a later reader wants: kept over WHAT. These decisions record the
    candidate and move nothing.
    """

    @pytest.mark.parametrize(
        "decision", [selection.DECISION_KEEP, selection.DECISION_UNDECIDED]
    )
    def test_the_considered_candidate_is_retained_without_moving_the_winner(
        self, store, control, decision
    ):
        _create(store, "exp-1", **PIN)
        _create(store, "exp-2", **PIN)
        group_id = control.group_for_experiment("exp-1")["group_id"]
        before = control.current_winner(group_id)

        state = control.record_decision(
            group_id,
            decision,
            expected_selection_id=before["selection_id"],
            candidate_experiment_id="exp-2",
            actor="reviewer",
            actor_kind="human",
            provenance="ui:experiments",
            rationale="slower on the long tasks",
        )

        assert state["experiment_id"] == "exp-1"
        assert state["selection_id"] == before["selection_id"]
        assert state["decision"]["candidate_experiment_id"] == "exp-2"
        latest = control.decision_history(group_id)[0]
        assert latest["decision"] == decision
        assert latest["candidate_experiment_id"] == "exp-2"
        assert latest["previous_experiment_id"] == "exp-1"
        assert latest["new_experiment_id"] == "exp-1"

    def test_a_generic_keep_still_needs_no_candidate(self, store, control):
        _create(store, "exp-1", **PIN)
        group_id = control.group_for_experiment("exp-1")["group_id"]

        state = control.record_decision(
            group_id,
            selection.DECISION_KEEP,
            expected_selection_id=control.current_winner(group_id)["selection_id"],
            actor="reviewer",
            actor_kind="human",
            provenance="ui:experiments",
        )

        assert state["decision"]["candidate_experiment_id"] is None
        assert control.decision_history(group_id)[0]["candidate_experiment_id"] is None

    def test_a_rejected_candidate_must_still_be_a_group_member(self, store, control):
        _create(store, "exp-1", **PIN)
        _create(store, "outsider", benchmark_id="other", benchmark_version="v1",
                benchmark_digest_sha256="e" * 64)
        group_id = control.group_for_experiment("exp-1")["group_id"]

        with pytest.raises(selection.ExperimentNotInGroup):
            control.record_decision(
                group_id,
                selection.DECISION_KEEP,
                expected_selection_id=control.current_winner(group_id)["selection_id"],
                candidate_experiment_id="outsider",
                actor="reviewer",
                actor_kind="human",
                provenance="ui:experiments",
            )

    def test_rejecting_the_same_candidate_twice_reads_as_two_decisions(
        self, store, control
    ):
        """The history is the point: two reviews of one challenger, both kept."""
        _create(store, "exp-1", **PIN)
        _create(store, "exp-2", **PIN)
        group_id = control.group_for_experiment("exp-1")["group_id"]

        for rationale in ("too early", "still too early"):
            control.record_decision(
                group_id,
                selection.DECISION_KEEP,
                expected_selection_id=control.current_winner(group_id)["selection_id"],
                candidate_experiment_id="exp-2",
                actor="reviewer",
                actor_kind="human",
                provenance="ui:experiments",
                rationale=rationale,
            )

        history = control.decision_history(group_id)
        assert [d["rationale"] for d in history[:2]] == ["still too early", "too early"]
        assert {d["candidate_experiment_id"] for d in history[:2]} == {"exp-2"}
        assert control.current_winner(group_id)["experiment_id"] == "exp-1"


# ----------------------------------------------------------------------
# Experiments recorded before contests lived in the live DB (§2.3)
# ----------------------------------------------------------------------


class TestPreExistingExperiments:
    def test_nothing_adopts_older_runs_automatically(self, store, control):
        """Owner resolution 8: an older experiment joins only when a decision
        names it, so a new experiment of its lineage is elected as usual."""
        _create_unregistered(store, "old-1", **PIN)
        _backdate(store, "old-1", "2026-01-01T00:00:00Z")

        _create(store, "new-1", **PIN)

        group_id = control.group_for_experiment("new-1")["group_id"]
        assert control.current_winner(group_id)["experiment_id"] == "new-1"
        assert control.group_for_experiment("old-1") is None

    def test_an_empty_store_still_elects_its_first_experiment(self, store, control):
        _create(store, "exp-1", **PIN)

        group_id = control.group_for_experiment("exp-1")["group_id"]
        assert control.current_winner(group_id)["experiment_id"] == "exp-1"

    def test_older_runs_of_another_lineage_do_not_block_a_new_contest(
        self, store, control
    ):
        _create_unregistered(store, "old-1", **PIN)
        _backdate(store, "old-1", "2026-01-01T00:00:00Z")
        _create(store, "fresh", benchmark_id="different", benchmark_version="v1",
                benchmark_digest_sha256="f" * 64)

        group_id = control.group_for_experiment("fresh")["group_id"]
        assert control.current_winner(group_id)["experiment_id"] == "fresh"

    def test_an_explicit_promotion_enrols_and_elects_an_unenrolled_experiment(
        self, store, control, db_path
    ):
        _create_unregistered(store, "old-1", **PIN)
        group_id = selection.comparison_group_identity(store.get_experiment("old-1"))["group_id"]
        reader = selection.SelectionControlStore(obs.ReadOnlyObservabilityStore(db_path))
        assert reader.winner_for_experiment("old-1") is None
        assert reader.group_for_experiment("old-1") is None

        state = control.record_decision(
            group_id,
            selection.DECISION_PROMOTE,
            expected_selection_id=None,
            candidate_experiment_id="old-1",
            actor="dhar",
            actor_kind="human",
            provenance="ui:experiments",
        )

        assert state["experiment_id"] == "old-1"
        assert control.group_for_experiment("old-1")["group_id"] == group_id
        assert [d["decision"] for d in control.decision_history(group_id)] == [
            selection.DECISION_INITIAL
        ]

    def test_an_explicit_promotion_enrols_into_a_contest_that_has_a_winner(
        self, store, control
    ):
        _create_unregistered(store, "old-1", **PIN)
        _create(store, "new-1", **PIN)
        group_id = control.group_for_experiment("new-1")["group_id"]

        _promote(control, group_id, "old-1")

        assert control.current_winner(group_id)["experiment_id"] == "old-1"
        assert {m["experiment_id"] for m in control.group_members(group_id)} == {
            "old-1", "new-1",
        }

    def test_a_promotion_that_saw_no_winner_is_stale_when_there_is_one(
        self, store, control
    ):
        _create(store, "exp-1", **PIN)
        _create(store, "exp-2", **PIN)
        group_id = control.group_for_experiment("exp-1")["group_id"]

        with pytest.raises(selection.StaleSelection):
            control.record_decision(
                group_id,
                selection.DECISION_PROMOTE,
                expected_selection_id=None,
                candidate_experiment_id="exp-2",
                actor="dhar",
                actor_kind="human",
                provenance="ui:experiments",
            )
        assert control.current_winner(group_id)["experiment_id"] == "exp-1"


# ----------------------------------------------------------------------
# Registration before anything is recorded
# ----------------------------------------------------------------------


class TestRegistrationBeforeAnyRun:
    """Registration at UI-creation time, recording at execution time."""

    def test_a_reference_can_win_before_anything_is_recorded(self, control):
        result = control.register_experiment_reference(
            selection.ExperimentReference(
                experiment_id="exp-planned",
                workflow_name="my_workflow",
                benchmark_id=PIN["benchmark_id"],
                benchmark_version=PIN["benchmark_version"],
            )
        )

        winner = control.current_winner(result["group_id"])
        assert winner["experiment_id"] == "exp-planned"
        # The decision is real; the run's state is simply not knowable yet.
        assert winner["experiment"] is None
        assert winner["experiment_resolved"] is False

    def test_recording_it_later_changes_no_selection(self, store, control):
        control.register_experiment_reference(
            selection.ExperimentReference(
                experiment_id="exp-planned",
                workflow_name="my_workflow",
                benchmark_id=PIN["benchmark_id"],
            )
        )
        group_id = control.list_groups()[0]["group_id"]
        before = control.current_winner(group_id)

        _create(store, "exp-planned", **PIN)

        after = control.current_winner(group_id)
        assert after["selection_id"] == before["selection_id"]
        assert after["decision_seq"] == before["decision_seq"]
        assert after["experiment_resolved"] is True
        assert after["experiment"]["status"] == "running"
        assert [d["decision"] for d in control.decision_history(group_id)] == [
            selection.DECISION_INITIAL
        ]

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
        assert not missing.parent.exists()


# ----------------------------------------------------------------------
# The control tables' own creation on a store opened without migrating
# ----------------------------------------------------------------------


class TestControlSchemaCreation:
    def test_concurrent_registration_creates_the_tables_once_and_elects_one_winner(
        self, store, db_path
    ):
        """A writer on a `migrate=False` store creates the control tables in its
        own `BEGIN IMMEDIATE`, so six racing first writers neither fail nor
        elect twice; the first to join is elected."""
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
        group_id = control.list_groups()[0]["group_id"]
        history = control.decision_history(group_id)
        assert [d["decision"] for d in history] == [selection.DECISION_INITIAL]
        assert len(control.group_members(group_id)) == 6
        assert control.current_winner(group_id)["experiment_id"] == (
            history[0]["new_experiment_id"]
        )

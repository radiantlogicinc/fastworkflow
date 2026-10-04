"""Current winner + append-only selection history (`fix-9eg.17.1`).

Integration tests against real SQLite stores — a real `ObservabilityStore`, a
real `ReadOnlyObservabilityStore`, real sealed archives, and real concurrent
writers on one file. No Mock fixtures, per `.cursor/rules/testing_rules.mdc`.

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
    return selection.SelectionControlStore.for_evidence(store)


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
    """Create evidence WITHOUT registering it in any control store.

    The shape an embedder with a shared workspace control uses — and the shape
    a store that predates selection has — so the tests below can build both.
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

    def test_initialize_winner_false_records_nothing(self, store, db_path):
        _create(store, "exp-quiet", initialize_winner=False, **PIN)

        assert not os.path.exists(selection.control_db_path_for(db_path))

    def test_unwritable_sidecar_does_not_fail_experiment_creation(
        self, store, tmp_path
    ):
        """An environmental problem must not look like a corrupt store."""
        if os.geteuid() == 0:
            pytest.skip("root ignores directory permissions")
        locked = tmp_path / "locked"
        locked.mkdir()
        locked.chmod(0o500)
        try:
            store.create_experiment(
                "exp-1",
                "label",
                declared_tasks=1,
                declared_attempts=1,
                selection_control_db_path=str(locked / "sel.sqlite3"),
            )
        finally:
            locked.chmod(0o700)

        assert store.get_experiment("exp-1")["status"] == "running"


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
        control = selection.SelectionControlStore.for_evidence(
            obs.ObservabilityStore(db_path)
        )
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
        control = selection.SelectionControlStore.for_evidence(store)
        group_id = control.group_for_experiment("exp-1")["group_id"]
        expected = control.current_winner(group_id)["selection_id"]

        barrier = threading.Barrier(2)
        outcomes: list[str] = []
        stale: list[selection.StaleSelection] = []

        def promote(candidate: str) -> None:
            writer = selection.SelectionControlStore.for_evidence(
                obs.ObservabilityStore(db_path)
            )
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
# Evidence stays immutable; judgements live in the control sidecar
# ----------------------------------------------------------------------


class TestEvidenceImmutability:
    def test_no_selection_tables_are_added_to_the_evidence_db(self, store, db_path):
        _create(store, "exp-1", **PIN)

        with sqlite3.connect(db_path) as conn:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        assert not {
            "comparison_groups",
            "comparison_group_members",
            "selection_pointers",
            "selection_decisions",
        } & tables
        assert os.path.exists(selection.control_db_path_for(db_path))

    def test_decisions_about_read_only_evidence_change_no_evidence_byte(
        self, store, db_path, tmp_path
    ):
        control_path = str(tmp_path / "elsewhere" / "sel.sqlite3")
        for experiment_id in ("exp-1", "exp-2"):
            store.create_experiment(
                experiment_id,
                f"label-{experiment_id}",
                declared_tasks=1,
                declared_attempts=1,
                workflow_name="my_workflow",
                selection_control_db_path=control_path,
                **PIN,
            )
        del store
        _settle(db_path)
        before = _evidence_fingerprint(db_path)
        assert not os.path.exists(selection.control_db_path_for(db_path))

        reader = obs.ReadOnlyObservabilityStore(db_path)
        control = selection.SelectionControlStore.for_evidence(
            reader, control_db_path=control_path
        )
        group_id = control.group_for_experiment("exp-1")["group_id"]
        _promote(control, group_id, "exp-2", rationale="read-only review")

        assert control.current_winner(group_id)["experiment_id"] == "exp-2"
        assert _evidence_fingerprint(db_path) == before

    def test_a_sealed_archive_keeps_its_own_bytes_while_being_judged(
        self, store, db_path, tmp_path
    ):
        """The archive is the evidence; the sidecar carries the judgement."""
        _create(store, "exp-1", **PIN)
        _create(store, "exp-2", **PIN)
        archive = tmp_path / "archive" / "sealed.sqlite3"
        store.archive_to(str(archive))
        sealed_digest = _digest(str(archive))

        reader = obs.ReadOnlyObservabilityStore(str(archive))
        # The archive carries the source's store identity, so the live sidecar
        # is the right place for decisions about it.
        control = selection.SelectionControlStore.for_evidence(
            reader, control_db_path=selection.control_db_path_for(db_path)
        )
        group_id = control.group_for_experiment("exp-1")["group_id"]
        _promote(control, group_id, "exp-2")

        assert control.current_winner(group_id)["experiment_id"] == "exp-2"
        assert _digest(str(archive)) == sealed_digest

    def test_a_sidecar_from_another_store_is_refused(self, store, tmp_path):
        _create(store, "exp-1", **PIN)
        sidecar = selection.control_db_path_for(store.db_path)

        other_db = str(tmp_path / "other" / "observability.sqlite3")
        os.makedirs(os.path.dirname(other_db), exist_ok=True)
        other = obs.ObservabilityStore(other_db)

        with pytest.raises(selection.ControlStoreIdentityMismatch):
            selection.SelectionControlStore.for_evidence(
                other, control_db_path=sidecar
            )

    def test_opening_an_absent_sidecar_without_create_is_reported(self, store):
        with pytest.raises(selection.SelectionControlUnavailable):
            selection.SelectionControlStore.for_evidence(store, create=False)


# ----------------------------------------------------------------------
# Imported groups, and the scope the next bead will use
# ----------------------------------------------------------------------


class TestAdoptionAndScope:
    def test_imported_experiments_elect_the_earliest_as_first_winner(
        self, store, db_path
    ):
        """The documented rule: ascending (created_at, experiment_id)."""
        for experiment_id in ("exp-c", "exp-a", "exp-b"):
            _create(store, experiment_id, initialize_winner=False, **PIN)
        # All three were created within the same second in this test, so the
        # tie-break by id is what decides — which is the point of stating it.
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "UPDATE experiments SET created_at=? WHERE experiment_id=?",
                ("2026-01-01T00:00:00Z", "exp-c"),
            )
            conn.execute(
                "UPDATE experiments SET created_at=? WHERE experiment_id=?",
                ("2026-01-02T00:00:00Z", "exp-a"),
            )
            conn.execute(
                "UPDATE experiments SET created_at=? WHERE experiment_id=?",
                ("2026-01-02T00:00:00Z", "exp-b"),
            )
            conn.commit()

        control = selection.SelectionControlStore.for_evidence(store)
        report = control.adopt_existing_experiments()

        assert report["experiments_seen"] == 3
        assert report["experiments_adopted"] == 3
        assert report["sources_read"] == [selection.PRIMARY_SOURCE_ID]
        assert report["sources_skipped"] == []
        group_id = control.group_for_experiment("exp-a")["group_id"]
        assert control.current_winner(group_id)["experiment_id"] == "exp-c"
        assert control.adopt_existing_experiments()["experiments_adopted"] == 0

    def test_adoption_leaves_a_moved_winner_alone(self, store, control):
        _create(store, "exp-1", **PIN)
        _create(store, "exp-2", **PIN)
        group_id = control.group_for_experiment("exp-1")["group_id"]
        _promote(control, group_id, "exp-2")

        control.adopt_existing_experiments()

        assert control.current_winner(group_id)["experiment_id"] == "exp-2"

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

        with sqlite3.connect(selection.control_db_path_for(db_path)) as conn:
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

    def test_reader_entry_point_opens_evidence_read_only(self, store, db_path):
        _create(store, "exp-1", **PIN)
        del store
        _settle(db_path)
        before = _evidence_fingerprint(db_path)

        control = selection.selection_control_for(db_path)
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
# Historical adoption: a new run must not inherit a contest it never entered
# ----------------------------------------------------------------------


class TestHistoricalAdoptionBootstrap:
    def test_a_new_experiment_does_not_win_over_older_unadopted_runs(
        self, store, db_path
    ):
        """The silent wrong answer this prevents: the NEWEST run of a store
        that predates selection quietly becoming 'the winner' of every run
        before it, with an `initial` decision nobody made."""
        _create_unregistered(store, "old-1", **PIN)
        _create_unregistered(store, "old-2", **PIN)
        _backdate(store, "old-1", "2026-01-01T00:00:00Z")
        _backdate(store, "old-2", "2026-01-02T00:00:00Z")

        _create(store, "new-1", **PIN)

        control = selection.SelectionControlStore.for_evidence(store)
        group_id = control.group_for_experiment("new-1")["group_id"]
        assert control.current_winner(group_id) is None
        assert control.decision_history(group_id) == []

    def test_the_earliest_experiment_may_still_initialize(self, store, control):
        """The check is about experiments OLDER than this one, not about any
        unregistered experiment: adoption itself relies on that."""
        _create_unregistered(store, "old-1", **PIN)
        _create_unregistered(store, "new-1", **PIN)
        _backdate(store, "old-1", "2026-01-01T00:00:00Z")

        result = control.register_experiment(
            "old-1", experiment=store.get_experiment("old-1")
        )

        assert result["initialized"] is True
        assert result["bootstrap_required"] is False

    def test_registration_reports_bootstrap_required_with_the_older_ids(
        self, store, control
    ):
        _create_unregistered(store, "old-1", **PIN)
        _create_unregistered(store, "new-1", **PIN)
        _backdate(store, "old-1", "2026-01-01T00:00:00Z")

        result = control.register_experiment(
            "new-1", experiment=store.get_experiment("new-1")
        )

        assert result["registered"] is True
        assert result["initialized"] is False
        assert result["bootstrap_required"] is True
        assert result["unadopted_experiment_ids"] == ["old-1"]
        assert result["evidence_scanned"] is True
        assert result["winner"] is None

    def test_the_bootstrap_resolves_it_deterministically(self, store, control):
        _create_unregistered(store, "old-1", **PIN)
        _backdate(store, "old-1", "2026-01-01T00:00:00Z")
        _create(store, "new-1", **PIN)
        group_id = control.group_for_experiment("new-1")["group_id"]
        assert control.current_winner(group_id) is None

        control.adopt_existing_experiments()

        winner = control.current_winner(group_id)
        assert winner["experiment_id"] == "old-1"
        assert winner["automatic"] is True
        assert {m["experiment_id"] for m in control.group_members(group_id)} == {
            "old-1",
            "new-1",
        }

    def test_an_empty_store_still_elects_its_first_experiment(self, store, control):
        """The ordinary case must not be collateral damage of the check."""
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

    def test_a_reference_registration_says_it_read_no_evidence(self, tmp_path):
        """With no store to read, absence of older runs cannot be verified, and
        the result says so rather than implying it was checked."""
        path = selection.shared_control_db_path_for(str(tmp_path))
        control = selection.open_shared_control(path)

        result = control.register_experiment_reference(
            selection.ExperimentReference(
                experiment_id="exp-ui-1",
                workflow_name="my_workflow",
                benchmark_id="todo-smoke",
            )
        )

        assert result["initialized"] is True
        assert result["evidence_scanned"] is False
        assert result["bootstrap_required"] is False


# ----------------------------------------------------------------------
# One control location, several evidence stores (the real product shape)
# ----------------------------------------------------------------------


@pytest.fixture
def two_stores(tmp_path):
    a = obs.ObservabilityStore(str(tmp_path / "runner-a" / "observability.sqlite3"))
    b = obs.ObservabilityStore(str(tmp_path / "runner-b" / "observability.sqlite3"))
    return a, b


@pytest.fixture
def workspace_control(tmp_path, two_stores):
    a, b = two_stores
    root = tmp_path / "workspace"
    root.mkdir()
    control = selection.open_shared_control(
        selection.shared_control_db_path_for(str(root)),
        sources={"runner-a": a, "runner-b": b},
    )
    control.authorize_source("runner-a", a, label="runner a")
    control.authorize_source("runner-b", b)
    return control


class TestSharedControlAcrossStores:
    def test_one_lineage_in_two_stores_has_exactly_one_winner(
        self, two_stores, workspace_control
    ):
        """The comparison the product actually runs: a winner recorded by one
        runner, a candidate recorded by another, judged against each other."""
        a, b = two_stores
        _create_unregistered(a, "exp-a", **PIN)
        _create_unregistered(b, "exp-b", **PIN)

        workspace_control.register_experiment("exp-a", source_id="runner-a")
        workspace_control.register_experiment("exp-b", source_id="runner-b")

        groups = workspace_control.list_groups()
        assert len(groups) == 1
        group_id = groups[0]["group_id"]
        assert {
            m["experiment_id"]: m["source_id"]
            for m in workspace_control.group_members(group_id)
        } == {"exp-a": "runner-a", "exp-b": "runner-b"}
        winner = workspace_control.current_winner(group_id)
        assert winner["experiment_id"] == "exp-a"
        assert winner["source_id"] == "runner-a"
        assert winner["experiment"]["status"] == "running"

        _promote(workspace_control, group_id, "exp-b")

        winner = workspace_control.current_winner(group_id)
        assert winner["experiment_id"] == "exp-b"
        assert winner["source_id"] == "runner-b"
        # The winner's live state is read from ITS store, not the first one.
        assert winner["experiment"]["description"] == "label-exp-b"
        assert winner["experiment_resolved"] is True
        # And there is still exactly one contest, not one per store.
        assert len(workspace_control.list_groups()) == 1

    def test_deciding_across_two_stores_changes_no_evidence_byte(
        self, two_stores, workspace_control
    ):
        a, b = two_stores
        _create_unregistered(a, "exp-a", **PIN)
        _create_unregistered(b, "exp-b", **PIN)
        workspace_control.register_experiment("exp-a", source_id="runner-a")
        workspace_control.register_experiment("exp-b", source_id="runner-b")
        _settle(a.db_path)
        _settle(b.db_path)
        before = (
            _evidence_fingerprint(a.db_path),
            _evidence_fingerprint(b.db_path),
        )

        group_id = workspace_control.list_groups()[0]["group_id"]
        _promote(workspace_control, group_id, "exp-b", rationale="fewer retries")

        assert (
            _evidence_fingerprint(a.db_path),
            _evidence_fingerprint(b.db_path),
        ) == before
        # No per-store sidecar was created either: the workspace has one.
        assert not os.path.exists(selection.control_db_path_for(a.db_path))
        assert not os.path.exists(selection.control_db_path_for(b.db_path))

    def test_an_unauthorized_source_is_refused(self, two_stores, workspace_control):
        a, _ = two_stores
        _create_unregistered(a, "exp-a", **PIN)

        with pytest.raises(selection.UnauthorizedEvidenceSource):
            workspace_control.register_experiment("exp-a", source_id="runner-z")

    def test_a_source_id_names_one_store_forever(self, two_stores, workspace_control):
        a, b = two_stores

        with pytest.raises(selection.ControlStoreIdentityMismatch):
            workspace_control.authorize_source("runner-a", b)

    def test_one_store_cannot_be_authorized_twice(self, two_stores, workspace_control):
        """Otherwise one run competes with itself under two names."""
        a, _ = two_stores

        with pytest.raises(selection.ControlStoreIdentityMismatch):
            workspace_control.authorize_source("runner-a-again", a)

    def test_a_resolver_handing_back_the_wrong_store_is_refused(
        self, tmp_path, two_stores, workspace_control
    ):
        a, b = two_stores
        _create_unregistered(a, "exp-a", **PIN)
        workspace_control.register_experiment("exp-a", source_id="runner-a")
        group_id = workspace_control.list_groups()[0]["group_id"]

        # Same control file, a misconfigured resolver: 'runner-a' now yields b.
        misconfigured = selection.open_shared_control(
            workspace_control.control_db_path,
            sources={"runner-a": b, "runner-b": b},
        )
        with pytest.raises(selection.ControlStoreIdentityMismatch):
            misconfigured.current_winner(group_id)

    def test_the_same_experiment_id_in_two_stores_is_a_collision_not_a_merge(
        self, two_stores, workspace_control
    ):
        """Ids are unique per store. Two stores can both hold 'exp-1'."""
        a, b = two_stores
        _create_unregistered(a, "exp-1", **PIN)
        _create_unregistered(b, "exp-1", **PIN)
        workspace_control.register_experiment("exp-1", source_id="runner-a")

        with pytest.raises(selection.ExperimentSourceCollision) as caught:
            workspace_control.register_experiment("exp-1", source_id="runner-b")

        assert caught.value.bound_source_id == "runner-a"
        group_id = workspace_control.list_groups()[0]["group_id"]
        assert len(workspace_control.group_members(group_id)) == 1

    def test_adoption_spans_both_stores_and_elects_one_winner(
        self, two_stores, workspace_control
    ):
        a, b = two_stores
        _create_unregistered(a, "exp-a", **PIN)
        _create_unregistered(b, "exp-b", **PIN)
        with sqlite3.connect(a.db_path) as conn:
            conn.execute(
                "UPDATE experiments SET created_at=? WHERE experiment_id=?",
                ("2026-03-02T00:00:00Z", "exp-a"),
            )
            conn.commit()
        with sqlite3.connect(b.db_path) as conn:
            conn.execute(
                "UPDATE experiments SET created_at=? WHERE experiment_id=?",
                ("2026-03-01T00:00:00Z", "exp-b"),
            )
            conn.commit()

        report = workspace_control.adopt_existing_experiments()

        assert report["experiments_seen"] == 2
        assert report["sources_read"] == ["runner-a", "runner-b"]
        assert report["sources_skipped"] == []
        group_id = workspace_control.list_groups()[0]["group_id"]
        # The earliest across BOTH stores wins, not the earliest of each.
        assert workspace_control.current_winner(group_id)["experiment_id"] == "exp-b"

    def test_adoption_reports_sources_it_could_not_read(
        self, two_stores, workspace_control
    ):
        """Adopting a partial view would make exactly the wrong answer this
        contract exists to prevent, so the gap is reported rather than implied.

        A later session is the realistic shape: the authorizations are in the
        file, but this process was only handed one of the two stores.
        """
        a, _ = two_stores
        _create_unregistered(a, "exp-a", **PIN)
        partial = selection.open_shared_control(
            workspace_control.control_db_path, sources={"runner-a": a}
        )

        report = partial.adopt_existing_experiments()

        assert report["sources_read"] == ["runner-a"]
        assert report["sources_skipped"] == ["runner-b"]
        assert report["experiments_adopted"] == 1

    def test_a_partial_view_says_which_sources_it_read(
        self, two_stores, workspace_control
    ):
        """A runner holds its own store, not the workspace's. The registration
        result says which sources the history check could see, so a caller is
        never told 'no older runs exist' about a store nobody opened."""
        a, _ = two_stores
        _create_unregistered(a, "exp-a", **PIN)
        partial = selection.open_shared_control(
            workspace_control.control_db_path, sources={"runner-a": a}
        )

        result = partial.register_experiment("exp-a", source_id="runner-a")

        assert result["sources_scanned"] == ["runner-a"]
        assert {str(r["source_id"]) for r in partial.list_sources()} == {
            "runner-a",
            "runner-b",
        }

    def test_a_shared_sidecar_is_not_a_single_store_sidecar(
        self, two_stores, workspace_control
    ):
        a, _ = two_stores

        with pytest.raises(selection.ControlModeMismatch):
            selection.SelectionControlStore.for_evidence(
                a, control_db_path=workspace_control.control_db_path
            )

    def test_a_single_store_sidecar_is_not_a_shared_one(self, store, db_path):
        _create(store, "exp-1", **PIN)

        with pytest.raises(selection.ControlModeMismatch):
            selection.open_shared_control(selection.control_db_path_for(db_path))


class TestRegistrationBeforeAnyStore:
    """Registration at UI-creation time, binding at execution time."""

    def test_a_reference_can_win_before_a_store_exists(self, tmp_path, two_stores):
        a, _ = two_stores
        control = selection.open_shared_control(
            selection.shared_control_db_path_for(str(tmp_path)),
            sources={"runner-a": a},
        )
        control.authorize_source("runner-a", a)

        result = control.register_experiment_reference(
            selection.ExperimentReference(
                experiment_id="exp-planned",
                workflow_name="my_workflow",
                benchmark_id=PIN["benchmark_id"],
                benchmark_version=PIN["benchmark_version"],
            )
        )

        group_id = result["group_id"]
        winner = control.current_winner(group_id)
        assert winner["experiment_id"] == "exp-planned"
        assert winner["source_id"] is None
        # The decision is real; the run's state is simply not knowable yet.
        assert winner["experiment"] is None
        assert winner["experiment_resolved"] is False

    def test_binding_the_store_later_changes_no_selection(self, tmp_path, two_stores):
        a, _ = two_stores
        control = selection.open_shared_control(
            selection.shared_control_db_path_for(str(tmp_path)),
            sources={"runner-a": a},
        )
        control.authorize_source("runner-a", a)
        control.register_experiment_reference(
            selection.ExperimentReference(
                experiment_id="exp-planned",
                workflow_name="my_workflow",
                benchmark_id=PIN["benchmark_id"],
            )
        )
        group_id = control.list_groups()[0]["group_id"]
        before = control.current_winner(group_id)

        _create_unregistered(a, "exp-planned", **PIN)
        control.bind_experiment_source("exp-planned", "runner-a")

        after = control.current_winner(group_id)
        assert after["selection_id"] == before["selection_id"]
        assert after["decision_seq"] == before["decision_seq"]
        assert after["source_id"] == "runner-a"
        assert after["experiment_resolved"] is True
        assert after["experiment"]["status"] == "running"
        assert [d["decision"] for d in control.decision_history(group_id)] == [
            selection.DECISION_INITIAL
        ]

    def test_binding_to_a_second_store_is_refused(self, tmp_path, two_stores):
        a, b = two_stores
        control = selection.open_shared_control(
            selection.shared_control_db_path_for(str(tmp_path)),
            sources={"runner-a": a, "runner-b": b},
        )
        control.authorize_source("runner-a", a)
        control.authorize_source("runner-b", b)
        control.register_experiment_reference(
            selection.ExperimentReference(
                experiment_id="exp-planned", workflow_name="my_workflow"
            )
        )
        control.bind_experiment_source("exp-planned", "runner-a")

        with pytest.raises(selection.ExperimentSourceCollision):
            control.bind_experiment_source("exp-planned", "runner-b")

    def test_binding_an_unknown_experiment_is_reported(self, tmp_path, two_stores):
        a, _ = two_stores
        control = selection.open_shared_control(
            selection.shared_control_db_path_for(str(tmp_path)),
            sources={"runner-a": a},
        )
        control.authorize_source("runner-a", a)

        with pytest.raises(obs.ExperimentNotFound):
            control.bind_experiment_source("never-registered", "runner-a")


# ----------------------------------------------------------------------
# The control sidecar's own schema creation and post-commit behaviour
# ----------------------------------------------------------------------


class TestControlSchemaAndPostCommit:
    def test_concurrent_first_open_of_a_fresh_sidecar_succeeds(self, store, tmp_path):
        """The sidecar's own cold-start race, which is the new schema's to own.

        Fresh-schema creation outside a transaction is the shape of `fix-upxs`
        in the evidence store: a second opener sees user_version=0 with tables
        already present and refuses a healthy file. The control schema creates
        itself inside BEGIN IMMEDIATE so that cannot happen here.
        """
        path = str(tmp_path / "race" / "sel.sqlite3")
        count = 8
        barrier = threading.Barrier(count)
        errors: list[BaseException] = []
        opened: list[selection.SelectionControlStore] = []

        def open_control() -> None:
            try:
                barrier.wait(timeout=30)
                opened.append(
                    selection.SelectionControlStore.for_evidence(
                        store, control_db_path=path
                    )
                )
            except BaseException as exc:  # pragma: no cover - surfaced below
                errors.append(exc)

        threads = [threading.Thread(target=open_control) for _ in range(count)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)

        assert not errors, errors
        assert len(opened) == count
        with sqlite3.connect(path) as conn:
            assert conn.execute("PRAGMA user_version").fetchone()[0] == (
                selection.CONTROL_SCHEMA_VERSION
            )
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        assert {"evidence_sources", "comparison_groups", "selection_pointers"} <= tables
        assert len(opened[0].list_sources()) == 1

    def test_concurrent_registration_into_a_fresh_sidecar_elects_one_winner(
        self, store, db_path, tmp_path
    ):
        """Schema creation and first registration racing together.

        Also pins the property that keeps the historical-adoption check from
        stranding a batch: every sibling but the oldest defers, and the oldest
        has nothing older to defer to, so a winner is always elected.
        """
        path = str(tmp_path / "race2" / "sel.sqlite3")
        for index in range(6):
            _create_unregistered(store, f"exp-{index}", **PIN)
        barrier = threading.Barrier(6)
        errors: list[BaseException] = []

        def register(index: int) -> None:
            try:
                control = selection.SelectionControlStore.for_evidence(
                    obs.ObservabilityStore(db_path), control_db_path=path
                )
                barrier.wait(timeout=30)
                control.register_experiment(f"exp-{index}")
            except BaseException as exc:  # pragma: no cover - surfaced below
                errors.append(exc)

        threads = [threading.Thread(target=register, args=(i,)) for i in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)

        assert not errors, errors
        control = selection.SelectionControlStore.for_evidence(
            obs.ObservabilityStore(db_path), control_db_path=path
        )
        group_id = control.list_groups()[0]["group_id"]
        history = control.decision_history(group_id)
        assert [d["decision"] for d in history] == [selection.DECISION_INITIAL]
        assert len(control.group_members(group_id)) == 6
        assert control.current_winner(group_id)["experiment_id"] == "exp-0"

    def test_a_post_commit_refusal_does_not_fail_experiment_creation(
        self, store, tmp_path, two_stores
    ):
        """A misconfigured sidecar must not make a created experiment look
        uncreated: the evidence row is already committed and there is nothing
        to roll back."""
        a, b = two_stores
        shared = selection.shared_control_db_path_for(str(tmp_path))
        selection.open_shared_control(shared, sources={})

        store.create_experiment(
            "exp-1",
            "label",
            declared_tasks=1,
            declared_attempts=1,
            workflow_name="my_workflow",
            selection_control_db_path=shared,
            **PIN,
        )

        assert store.get_experiment("exp-1")["status"] == "running"

    def test_the_refusal_is_reported_not_swallowed(self, store, tmp_path):
        """An unauthorized store must not enrol itself in a shared contest by
        the side effect of creating an experiment."""
        shared = selection.shared_control_db_path_for(str(tmp_path))
        control = selection.open_shared_control(shared, sources={})
        _create_unregistered(store, "exp-1", **PIN)

        result = selection.initialize_winner_for(
            store, "exp-1", control_db_path=shared
        )

        assert result["status"] == selection.INIT_REFUSED
        assert "not authorized" in result["error"]
        assert control.list_sources() == []
        assert control.list_groups() == []

    def test_an_unavailable_sidecar_is_reported_as_environmental(
        self, store, tmp_path
    ):
        if os.geteuid() == 0:
            pytest.skip("root ignores directory permissions")
        locked = tmp_path / "locked"
        locked.mkdir()
        locked.chmod(0o500)
        _create_unregistered(store, "exp-1", **PIN)
        try:
            result = selection.initialize_winner_for(
                store, "exp-1", control_db_path=str(locked / "sel.sqlite3")
            )
        finally:
            locked.chmod(0o700)

        assert result["status"] == selection.INIT_UNAVAILABLE

    def test_strict_initialization_raises_for_an_explicit_caller(
        self, store, tmp_path
    ):
        """A UI bootstrap wants the error; `create_experiment` cannot use it."""
        shared = selection.shared_control_db_path_for(str(tmp_path))
        selection.open_shared_control(shared, sources={})
        _create_unregistered(store, "exp-1", **PIN)

        with pytest.raises(selection.UnauthorizedEvidenceSource):
            selection.initialize_winner_for(
                store, "exp-1", control_db_path=shared, strict=True
            )

    def test_two_runners_pointed_at_one_workspace_control_share_a_contest(
        self, two_stores, workspace_control
    ):
        """The whole shared use case, driven the way a runner drives it: each
        store creates its experiment normally and points at the workspace
        control file. Neither is told its source id — it is found by identity,
        which is also why an unauthorized store cannot join.

        Each runner process holds only its OWN store, so neither can see
        whether the other holds older runs of this lineage, and neither elects
        a winner — see the partial/full resolver regression below. Both
        register, into ONE group, under their own sources. An embedder that can
        resolve both (the workflow bootstrap, or `adopt_existing_experiments`)
        is what elects, and it elects the earliest across both stores.
        """
        a, b = two_stores
        shared = workspace_control.control_db_path

        for store, experiment_id in ((a, "exp-a"), (b, "exp-b")):
            store.create_experiment(
                experiment_id,
                f"label-{experiment_id}",
                declared_tasks=1,
                declared_attempts=1,
                workflow_name="my_workflow",
                selection_control_db_path=shared,
                **PIN,
            )

        assert len(workspace_control.list_groups()) == 1
        group_id = workspace_control.list_groups()[0]["group_id"]
        assert {
            m["experiment_id"]: m["source_id"]
            for m in workspace_control.group_members(group_id)
        } == {"exp-a": "runner-a", "exp-b": "runner-b"}
        assert workspace_control.current_winner(group_id) is None

        workspace_control.adopt_existing_experiments()

        assert workspace_control.current_winner(group_id)["experiment_id"] == "exp-a"
        # Still no per-store sidecars: the workspace file is the only control.
        assert not os.path.exists(selection.control_db_path_for(a.db_path))
        assert not os.path.exists(selection.control_db_path_for(b.db_path))

    def test_a_runner_holding_one_store_does_not_elect_over_an_unread_one(
        self, two_stores, workspace_control
    ):
        """The regression for the bootstrap contract.

        `runner-b` holds the OLDER experiment, and the process registering
        `exp-a` cannot open it. "I found no older runs" from that process means
        only "I did not look", and electing on it is exactly how a new
        candidate silently outranks an older experiment. So it registers, says
        which source it could not read, and elects nobody.
        """
        a, b = two_stores
        _create_unregistered(b, "exp-old", **PIN)
        _backdate(b, "exp-old", "2020-01-01T00:00:00.000000+00:00")
        _create_unregistered(a, "exp-new", **PIN)
        partial = selection.open_shared_control(
            workspace_control.control_db_path, sources={"runner-a": a}
        )

        result = partial.register_experiment("exp-new", source_id="runner-a")

        assert result["registered"] is True
        assert result["initialized"] is False
        assert result["bootstrap_required"] is True
        assert result["unresolved_sources"] == ["runner-b"]
        assert result["sources_scanned"] == ["runner-a"]
        assert result["winner"] is None

    def test_the_same_control_elects_once_the_missing_store_is_resolvable(
        self, two_stores, workspace_control
    ):
        """...and recovers by itself when the store comes back.

        Same control file, same authorizations, nothing re-registered by hand:
        the only thing that changed is that this process can open both stores.
        The older experiment wins, which is the answer the partial view
        refused to guess.
        """
        a, b = two_stores
        _create_unregistered(b, "exp-old", **PIN)
        _backdate(b, "exp-old", "2020-01-01T00:00:00.000000+00:00")
        _create_unregistered(a, "exp-new", **PIN)
        partial = selection.open_shared_control(
            workspace_control.control_db_path, sources={"runner-a": a}
        )
        partial.register_experiment("exp-new", source_id="runner-a")
        group_id = partial.list_groups()[0]["group_id"]
        assert partial.current_winner(group_id) is None

        report = workspace_control.adopt_existing_experiments()

        assert report["complete"] is True
        assert report["sources_skipped"] == []
        assert report["groups_without_winner"] == []
        winner = workspace_control.current_winner(group_id)
        assert winner["experiment_id"] == "exp-old"
        assert winner["source_id"] == "runner-b"

    def test_a_partial_adoption_registers_but_elects_nobody(
        self, two_stores, workspace_control
    ):
        """Adoption is held to the same rule as registration.

        Adopting through a partial view would install "the earliest I could
        see" as the group's first winner, which is the silent wrong answer the
        whole contract exists to prevent. It reports the gap instead.
        """
        a, _ = two_stores
        _create_unregistered(a, "exp-a", **PIN)
        partial = selection.open_shared_control(
            workspace_control.control_db_path, sources={"runner-a": a}
        )

        report = partial.adopt_existing_experiments()

        assert report["experiments_adopted"] == 1
        assert report["complete"] is False
        assert report["sources_skipped"] == ["runner-b"]
        assert report["groups_without_winner"] == [partial.list_groups()[0]["group_id"]]

    def test_a_successful_initialization_reports_what_it_did(self, store):
        _create_unregistered(store, "exp-1", **PIN)

        result = selection.initialize_winner_for(store, "exp-1")

        assert result["status"] == selection.INIT_RECORDED
        assert result["registration"]["initialized"] is True
        assert result["registration"]["winner"]["experiment_id"] == "exp-1"

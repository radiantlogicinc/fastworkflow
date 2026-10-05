"""Registration, repeat count and the one workflow control (`fix-9eg.17.2`).

Integration tests against real files, a real `ObservabilityStore` and a real
`ExperimentController`. No Mock fixtures, per `.cursor/rules/testing_rules.mdc`,
and nothing here runs a model or spends anything: every experiment below is an
identity and a declaration.

What these pin, in one sentence each:

- An experiment joins its contest in the workflow's live DB when it is
  CREATED, before anything runs, so the first one is the winner from that
  moment.
- A runner records into that SAME live DB, so one registered experiment never
  has two winners disagreeing about it.
- A repeat count is a whole number somebody meant, not whatever `int()` made
  of what arrived.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from fastworkflow import state_paths
from fastworkflow.benchmark import setup
from fastworkflow.experiment.runner import ExperimentController
from fastworkflow.observability import store as obs


def _benchmark(folder, title="Roster review"):
    return setup.save_benchmark(
        folder, {"title": title, "tasks": [{"prompt": "Review the roster"}]}
    )


@pytest.fixture
def folder(tmp_path):
    wf = tmp_path / "my_workflow"
    wf.mkdir()
    obs.ObservabilityStore(state_paths.observability_db(str(wf)))
    return wf


def _control(folder):
    return setup.workflow_control(folder)


def _winner_id(folder, experiment_id):
    winner = setup.workflow_winner(folder, experiment_id)
    return None if winner is None else winner["experiment_id"]


# ----------------------------------------------------------------------
# The live DB's URI (the pre-integration review item)
# ----------------------------------------------------------------------


class TestLiveDbPathEncoding:
    """A state root is a path, not a URI, and paths contain punctuation."""

    @pytest.mark.parametrize(
        "name", ["plain", "we#ird", "quer?y", "a#b?c d", "100% done"]
    )
    def test_the_contest_is_readable_under_a_punctuated_root(
        self, tmp_path, monkeypatch, name
    ):
        """The read side opens the live DB read-only, by URI.

        Interpolating the path into `file:{path}?mode=ro` makes SQLite read a
        `#` as a fragment and a `?` as the start of the query, so the file it
        opens is not the file it was given. The failure is SILENT: an
        unreadable file reads as "no live DB", and every contest in it reads
        as empty.
        """
        monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / name))
        wf = tmp_path / "my_workflow"
        wf.mkdir()
        obs.ObservabilityStore(state_paths.observability_db(str(wf)))
        record = setup.create_experiment(wf, _benchmark(wf)["benchmark_id"], "v1")

        assert _winner_id(wf, record["experiment_id"]) == record["experiment_id"]

    def test_an_absent_live_db_still_reads_as_absent(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "no#such"))
        wf = tmp_path / "my_workflow"
        wf.mkdir()

        assert setup.workflow_winner(wf, "exp-nobody") is None
        assert not os.path.exists(state_paths.observability_db(str(wf)))


# ----------------------------------------------------------------------
# Registration: the first experiment wins before any evidence exists
# ----------------------------------------------------------------------


class TestRegistrationWinsAtCreation:
    def test_the_first_experiment_is_the_winner_before_anything_runs(self, folder):
        benchmark = _benchmark(folder)

        record = setup.create_experiment(folder, benchmark["benchmark_id"], "v1")

        assert record["runs_per_task"] == 1
        assert _winner_id(folder, record["experiment_id"]) == record["experiment_id"]
        # No evidence was recorded to make that true.
        assert _control(folder).store.get_experiment(record["experiment_id"]) is None

    def test_the_winner_is_not_annotated_as_successful(self, folder):
        benchmark = _benchmark(folder)
        record = setup.create_experiment(folder, benchmark["benchmark_id"], "v1")

        winner = setup.workflow_winner(folder, record["experiment_id"])

        assert winner["automatic"] is True
        # Nothing has run, so there is no state to report and none is invented.
        assert winner["experiment_resolved"] is False

    def test_a_second_registration_joins_without_winning(self, folder):
        benchmark = _benchmark(folder)
        first = setup.create_experiment(folder, benchmark["benchmark_id"], "v1")

        second = setup.create_experiment(folder, benchmark["benchmark_id"], "v1")

        assert second["experiment_id"] != first["experiment_id"]
        assert _winner_id(folder, second["experiment_id"]) == first["experiment_id"]
        group = _control(folder).group_for_experiment(second["experiment_id"])
        assert (
            len(_control(folder).group_members(group["group_id"])) == 2
        )

    def test_reading_a_winner_never_creates_the_live_db(self, tmp_path):
        """A GET that brings a database into existence is a GET that writes."""
        wf = tmp_path / "unrecorded_workflow"
        wf.mkdir()

        assert setup.workflow_winner(wf, "exp-nobody") is None
        assert not os.path.exists(state_paths.observability_db(str(wf)))

    def test_different_benchmarks_are_different_contests(self, folder):
        one = _benchmark(folder, "Roster")
        two = _benchmark(folder, "Billing")
        first = setup.create_experiment(folder, one["benchmark_id"], "v1")
        other = setup.create_experiment(folder, two["benchmark_id"], "v1")

        assert _winner_id(folder, first["experiment_id"]) == first["experiment_id"]
        assert _winner_id(folder, other["experiment_id"]) == other["experiment_id"]

    def test_creation_on_a_workflow_with_no_live_db_creates_it(self, tmp_path):
        """Minting an identity must not fail because nobody has chatted yet.

        The registration lives in the live DB, so creating the first one (a
        POST) creates that DB; reading still never does.
        """
        wf = tmp_path / "unrecorded_workflow"
        wf.mkdir()
        benchmark = _benchmark(wf)
        assert setup.registered_experiments(wf, benchmark["benchmark_id"]) == []
        assert not os.path.exists(state_paths.observability_db(str(wf)))

        record = setup.create_experiment(wf, benchmark["benchmark_id"], "v1")

        assert record["experiment_id"].startswith("exp-")
        assert setup.load_experiment(wf, record["experiment_id"]) == record
        assert _winner_id(wf, record["experiment_id"]) == record["experiment_id"]
        assert os.path.isfile(state_paths.observability_db(str(wf)))


# ----------------------------------------------------------------------
# Repeat count
# ----------------------------------------------------------------------


class TestRunsPerTask:
    @pytest.mark.parametrize("value,expected", [(1, 1), (3, 3), ("3", 3), (100, 100)])
    def test_whole_numbers_are_accepted(self, value, expected):
        assert setup.validate_runs_per_task(value) == expected

    @pytest.mark.parametrize(
        "value",
        [True, False, 2.0, 2.9, "2.0", "three", "", None, 0, -1, 101, [3]],
    )
    def test_everything_else_is_refused(self, value):
        """`bool` IS an `int` in Python and `int(2.9)` is 2.

        Both would be accepted silently by the obvious conversion, and both
        turn into a number of real, paid executions nobody asked for.
        """
        with pytest.raises(ValueError):
            setup.validate_runs_per_task(value)

    def test_the_count_is_kept_on_the_registration(self, folder):
        benchmark = _benchmark(folder)

        record = setup.create_experiment(
            folder, benchmark["benchmark_id"], "v1", runs_per_task=3
        )

        assert record["runs_per_task"] == 3
        assert setup.load_experiment(folder, record["experiment_id"])["runs_per_task"] == 3

    def test_an_invalid_count_creates_nothing(self, folder):
        benchmark = _benchmark(folder)

        with pytest.raises(ValueError):
            setup.create_experiment(
                folder, benchmark["benchmark_id"], "v1", runs_per_task=0
            )

        assert setup.registered_experiments(folder, benchmark["benchmark_id"]) == []


# ----------------------------------------------------------------------
# Duplicating an experiment: the candidate action
# ----------------------------------------------------------------------


class TestDuplicateExperiment:
    def test_everything_is_inherited_and_the_identity_is_new(self, folder):
        benchmark = _benchmark(folder)
        source = setup.create_experiment(
            folder, benchmark["benchmark_id"], "v1", description="first", runs_per_task=3
        )

        candidate = setup.duplicate_experiment(folder, source["experiment_id"])

        assert candidate["experiment_id"] != source["experiment_id"]
        assert candidate["benchmark_id"] == source["benchmark_id"]
        assert candidate["benchmark_version"] == source["benchmark_version"]
        assert candidate["benchmark_digest_sha256"] == source["benchmark_digest_sha256"]
        assert candidate["task_ids"] == source["task_ids"]
        assert candidate["description"] == "first"
        assert candidate["runs_per_task"] == 3
        assert candidate["source_experiment_id"] == source["experiment_id"]
        assert candidate["changed_fields"] == []
        assert candidate["store"] is None

    def test_only_the_explicitly_changed_fields_differ(self, folder):
        benchmark = _benchmark(folder)
        source = setup.create_experiment(
            folder, benchmark["benchmark_id"], "v1", description="first", runs_per_task=1
        )

        candidate = setup.duplicate_experiment(
            folder,
            source["experiment_id"],
            description="with three runs",
            runs_per_task=3,
        )

        assert candidate["changed_fields"] == ["description", "runs_per_task"]
        assert candidate["description"] == "with three runs"
        assert candidate["runs_per_task"] == 3
        # The source is untouched: duplicating is not editing.
        assert setup.load_experiment(folder, source["experiment_id"]) == source

    def test_a_candidate_does_not_outrank_the_experiment_it_copied(self, folder):
        benchmark = _benchmark(folder)
        source = setup.create_experiment(folder, benchmark["benchmark_id"], "v1")

        candidate = setup.duplicate_experiment(folder, source["experiment_id"])

        assert _winner_id(folder, candidate["experiment_id"]) == source["experiment_id"]

    def test_an_invalid_repeat_count_duplicates_nothing(self, folder):
        benchmark = _benchmark(folder)
        source = setup.create_experiment(folder, benchmark["benchmark_id"], "v1")

        with pytest.raises(ValueError):
            setup.duplicate_experiment(
                folder, source["experiment_id"], runs_per_task=True
            )

        assert len(setup.registered_experiments(folder, benchmark["benchmark_id"])) == 1

    def test_duplicating_a_changed_benchmark_is_refused(self, folder):
        """Duplicating a setup whose pinned contents moved is not the same setup."""
        benchmark = _benchmark(folder)
        source = setup.create_experiment(folder, benchmark["benchmark_id"], "v1")
        version = Path(
            folder, "benchmarks", benchmark["benchmark_id"], "v1.json"
        )
        payload = version.read_text().replace("Review the roster", "Something else")
        version.write_text(payload)

        with pytest.raises(setup.BenchmarkSetupConflict):
            setup.duplicate_experiment(folder, source["experiment_id"])

    def test_a_deleted_experiment_cannot_be_duplicated(self, folder):
        benchmark = _benchmark(folder)
        # The first experiment of a group is its winner, and the winner is not
        # deletable (`fix-jfy5`); this one exists so `source` is an ordinary
        # non-selected registration.
        setup.create_experiment(folder, benchmark["benchmark_id"], "v1")
        source = setup.create_experiment(folder, benchmark["benchmark_id"], "v1")
        setup.delete_empty_experiment(folder, source["experiment_id"])

        with pytest.raises(setup.ExperimentDeleted):
            setup.duplicate_experiment(folder, source["experiment_id"])


# ----------------------------------------------------------------------
# The runner records into the SAME live DB
# ----------------------------------------------------------------------


def _controller(folder):
    db_path = state_paths.observability_db(str(folder))
    store = obs.ObservabilityStore(db_path)
    controller = ExperimentController(
        db_path,
        store.store_identity(),
        external=False,
        workflow_folderpath=str(folder),
    )
    return store, controller


def _declare(controller, record, experiment_id=None, attempts=1):
    """Declare a registered experiment's attempts, the way `run` does.

    `workflow_name` is passed because `ExperimentHarness.run` passes it, and it
    is half of the derived comparison group: a driver that omits it records an
    experiment the group identity cannot recognise as this workflow's.
    """
    experiment_id = experiment_id or record["experiment_id"]
    task_ids = record["task_ids"]
    controller.create_experiment(
        experiment_id,
        record.get("description", ""),
        declared_tasks=len(task_ids),
        declared_attempts=attempts,
        declarations=[
            (task_id, n, f"ch-{experiment_id}-{task_id}-{n}")
            for task_id in task_ids
            for n in range(1, attempts + 1)
        ],
        workflow_name=setup.workflow_name_for(controller.workflow_folderpath),
    )


class TestRunnerRecordsIntoTheLiveDb:
    def test_starting_a_runner_keeps_the_registration_winner(self, folder):
        benchmark = _benchmark(folder)
        record = setup.create_experiment(folder, benchmark["benchmark_id"], "v1")
        _store, controller = _controller(folder)

        _declare(controller, record)

        # The registration's winner is unchanged by execution starting.
        winner = setup.workflow_winner(folder, record["experiment_id"])
        assert winner["experiment_id"] == record["experiment_id"]
        assert winner["experiment"]["status"] == "running"

    def test_there_is_no_second_local_winner(self, folder):
        """The conflict this integration exists to prevent.

        A second database beside the live DB would hold its own opinion about
        the same registered experiment, and whichever screen the reader opened
        would decide which winner they saw.
        """
        benchmark = _benchmark(folder)
        record = setup.create_experiment(folder, benchmark["benchmark_id"], "v1")
        store, controller = _controller(folder)

        _declare(controller, record)

        state_dir = os.path.dirname(store.db_path)
        assert [
            name for name in os.listdir(state_dir) if name.endswith(".sqlite3")
        ] == ["observability.sqlite3"]

    def test_running_writes_nothing_into_the_workflow_folder(self, folder):
        """Starting a runner is not an edit to the user's project.

        The contest is a fact about this machine, so it belongs in the state
        dir with the live DB. Writing it under `benchmarks/` instead would put
        runtime bookkeeping into a checked-in source tree, which is how the
        repo's own test workflows picked up an untracked `benchmarks/`
        directory.
        """
        benchmark = _benchmark(folder)
        record = setup.create_experiment(folder, benchmark["benchmark_id"], "v1")
        before = sorted(p.relative_to(folder) for p in folder.rglob("*"))

        store, controller = _controller(folder)
        _declare(controller, record)

        assert sorted(p.relative_to(folder) for p in folder.rglob("*")) == before
        # It was recorded — just somewhere else.
        assert store.get_experiment(record["experiment_id"]) is not None

    def test_a_later_runner_does_not_outrank_the_first_registration(
        self, folder
    ):
        benchmark = _benchmark(folder)
        first = setup.create_experiment(folder, benchmark["benchmark_id"], "v1")
        candidate = setup.duplicate_experiment(folder, first["experiment_id"])
        _store, controller = _controller(folder)

        # Only the CANDIDATE ever runs; the first was registered and never
        # executed. It is still the winner nobody has replaced.
        _declare(controller, candidate)

        assert _winner_id(folder, candidate["experiment_id"]) == first["experiment_id"]

    def test_two_runners_share_one_contest(self, folder):
        benchmark = _benchmark(folder)
        first = setup.create_experiment(folder, benchmark["benchmark_id"], "v1")
        candidate = setup.duplicate_experiment(folder, first["experiment_id"])
        _store_a, controller_a = _controller(folder)
        _store_b, controller_b = _controller(folder)

        _declare(controller_a, first)
        _declare(controller_b, candidate)

        control = _control(folder)
        assert len(control.list_groups()) == 1
        group_id = control.list_groups()[0]["group_id"]
        assert {m["experiment_id"] for m in control.group_members(group_id)} == {
            first["experiment_id"], candidate["experiment_id"],
        }
        winner = control.current_winner(group_id)
        assert winner["experiment_id"] == first["experiment_id"]
        assert winner["experiment_resolved"] is True

    def test_an_unregistered_run_in_a_workflow_still_lands_in_the_contest(
        self, folder
    ):
        """A driver that never used the UI is still part of the workflow.

        It has no registration to enter, so its evidence row is what registers
        it — into the same live DB, in the same transaction.
        """
        _store, controller = _controller(folder)

        controller.create_experiment(
            "exp-adhoc",
            "driver run",
            declared_tasks=1,
            declared_attempts=1,
            declarations=[("task_1", 1, "ch-1")],
            workflow_name=setup.workflow_name_for(folder),
        )

        assert _winner_id(folder, "exp-adhoc") == "exp-adhoc"

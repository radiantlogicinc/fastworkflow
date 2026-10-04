"""Registration, repeat count and the one workflow control (`fix-9eg.17.2`).

Integration tests against real files, a real `ObservabilityStore` and a real
`ExperimentController`. No Mock fixtures, per `.cursor/rules/testing_rules.mdc`,
and nothing here runs a model or spends anything: every experiment below is an
identity and a declaration.

What these pin, in one sentence each:

- An experiment joins its contest when it is CREATED, with no evidence store,
  so the first one is the winner from that moment.
- A runner joins the SAME control file, so one registered experiment never has
  two winners — one local, one shared — disagreeing about it.
- A repeat count is a whole number somebody meant, not whatever `int()` made
  of what arrived.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
from pathlib import Path

import pytest

from fastworkflow import state_paths
from fastworkflow.benchmark import setup
from fastworkflow.experiment.runner import ExperimentController
from fastworkflow.observability import selection
from fastworkflow.observability import store as obs


def _benchmark(folder, title="Roster review"):
    return setup.save_benchmark(
        folder, {"title": title, "tasks": [{"prompt": "Review the roster"}]}
    )


@pytest.fixture
def folder(tmp_path):
    wf = tmp_path / "my_workflow"
    wf.mkdir()
    return wf


def _control(folder):
    return setup.open_workflow_control(folder, create=False)


def _winner_id(folder, experiment_id):
    winner = setup.workflow_winner(folder, experiment_id)
    return None if winner is None else winner["experiment_id"]


# ----------------------------------------------------------------------
# The control helper's URI (the pre-integration review item)
# ----------------------------------------------------------------------


class TestControlPathEncoding:
    """A control root is a path, not a URI, and paths contain punctuation."""

    @pytest.mark.parametrize(
        "name", ["plain", "we#ird", "quer?y", "a#b?c d", "100% done"]
    )
    def test_a_shared_control_is_readable_under_a_punctuated_root(self, tmp_path, name):
        """`control_mode_of` decides which entry point opens a control file.

        Interpolating the path into `file:{path}?mode=ro` makes SQLite read a
        `#` as a fragment and a `?` as the start of the query, so the file it
        opens is not the file it was given. The failure is SILENT: an
        unreadable file reports "no sidecar", the caller concludes this is a
        fresh single-store case, and the shared contest quietly gains a
        private second control nobody is looking at.
        """
        root = tmp_path / name
        root.mkdir()
        path = selection.shared_control_db_path_for(str(root))
        control = selection.open_shared_control(path)

        assert selection.control_mode_of(path) == selection.CONTROL_MODE_SHARED
        # And it is genuinely usable, not merely identifiable.
        control.register_experiment_reference(
            selection.ExperimentReference(
                experiment_id="exp-1",
                workflow_name="my_workflow",
                benchmark_id="todo-smoke",
            )
        )
        group_id = control.list_groups()[0]["group_id"]
        assert control.current_winner(group_id)["experiment_id"] == "exp-1"

    def test_an_absent_control_still_reads_as_absent(self, tmp_path):
        missing = selection.shared_control_db_path_for(str(tmp_path / "no#such"))

        assert selection.control_mode_of(missing) is None


# ----------------------------------------------------------------------
# Registration: the first experiment wins before any evidence exists
# ----------------------------------------------------------------------


class TestRegistrationWinsAtCreation:
    def test_the_first_experiment_is_the_winner_with_no_store_anywhere(self, folder):
        benchmark = _benchmark(folder)

        record = setup.create_experiment(folder, benchmark["benchmark_id"], "v1")

        assert record["runs_per_task"] == 1
        assert _winner_id(folder, record["experiment_id"]) == record["experiment_id"]
        # No evidence database was created to make that true.
        assert not os.path.exists(state_paths.observability_db(str(folder)))

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

    def test_reading_a_winner_never_creates_the_control(self, folder):
        """A GET that brings a database into existence is a GET that writes."""
        assert setup.workflow_winner(folder, "exp-nobody") is None
        assert not os.path.exists(setup.workflow_control_db_path(folder))

    def test_different_benchmarks_are_different_contests(self, folder):
        one = _benchmark(folder, "Roster")
        two = _benchmark(folder, "Billing")
        first = setup.create_experiment(folder, one["benchmark_id"], "v1")
        other = setup.create_experiment(folder, two["benchmark_id"], "v1")

        assert _winner_id(folder, first["experiment_id"]) == first["experiment_id"]
        assert _winner_id(folder, other["experiment_id"]) == other["experiment_id"]

    def test_creation_survives_an_unwritable_control(self, folder, monkeypatch):
        """Minting an identity must not fail because a sidecar is unwritable."""
        benchmark = _benchmark(folder)
        blocked = folder / "blocked"
        blocked.mkdir()
        blocked.chmod(0o500)
        monkeypatch.setattr(
            setup,
            "workflow_control_db_path",
            lambda _path: str(blocked / "selection.control.sqlite3"),
        )
        try:
            record = setup.create_experiment(folder, benchmark["benchmark_id"], "v1")
        finally:
            blocked.chmod(0o700)

        assert record["experiment_id"].startswith("exp-")
        assert setup.load_experiment(folder, record["experiment_id"]) == record


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

    def test_a_setup_field_added_later_is_inherited_without_code_changes(self, folder):
        """The copy is wholesale, then overwritten — not field by field."""
        benchmark = _benchmark(folder)
        source = setup.create_experiment(folder, benchmark["benchmark_id"], "v1")
        record = setup.load_experiment(folder, source["experiment_id"])
        record["future_runner_setting"] = {"temperature": 0}
        setup._atomic_json(
            setup._registration_path(folder, source["experiment_id"]), record
        )

        candidate = setup.duplicate_experiment(folder, source["experiment_id"])

        assert candidate["future_runner_setting"] == {"temperature": 0}

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
# The runner binds its evidence to the SAME control
# ----------------------------------------------------------------------


def _controller(folder, db_path):
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


class TestRunnerBindsToTheWorkflowControl:
    def test_starting_a_runner_binds_the_evidence_source(self, folder, tmp_path):
        benchmark = _benchmark(folder)
        record = setup.create_experiment(folder, benchmark["benchmark_id"], "v1")
        store, controller = _controller(folder, str(tmp_path / "evidence.sqlite3"))

        _declare(controller, record)

        control = _control(folder)
        source_id = control.source_for_experiment(record["experiment_id"])
        assert source_id == store.store_identity()
        assert {r["source_id"] for r in control.list_sources()} == {source_id}
        # The registration's winner is unchanged by execution starting.
        winner = setup.workflow_winner(folder, record["experiment_id"])
        assert winner["experiment_id"] == record["experiment_id"]
        assert winner["experiment"]["status"] == "running"

    def test_there_is_no_second_local_winner(self, folder, tmp_path):
        """The conflict this integration exists to prevent.

        A per-store sidecar beside the evidence DB would hold its own opinion
        about the same registered experiment, and whichever screen the reader
        opened would decide which winner they saw.
        """
        benchmark = _benchmark(folder)
        record = setup.create_experiment(folder, benchmark["benchmark_id"], "v1")
        db_path = str(tmp_path / "evidence.sqlite3")
        _store, controller = _controller(folder, db_path)

        _declare(controller, record)

        assert not os.path.exists(selection.control_db_path_for(db_path))
        assert os.path.exists(setup.workflow_control_db_path(folder))

    def test_binding_writes_nothing_into_the_workflow_folder(self, folder, tmp_path):
        """Starting a runner is not an edit to the user's project.

        Which evidence databases are in the contest is a fact about this
        machine, so it belongs in the state dir with the control file. Writing
        it under `benchmarks/` instead would put runtime bookkeeping into a
        checked-in source tree, which is how the repo's own test workflows
        picked up an untracked `benchmarks/` directory.
        """
        benchmark = _benchmark(folder)
        record = setup.create_experiment(folder, benchmark["benchmark_id"], "v1")
        before = sorted(p.relative_to(folder) for p in folder.rglob("*"))

        store, controller = _controller(folder, str(tmp_path / "evidence.sqlite3"))
        _declare(controller, record)

        assert sorted(p.relative_to(folder) for p in folder.rglob("*")) == before
        # It was recorded — just somewhere else.
        assert setup.known_evidence_sources(folder) == {
            store.store_identity(): os.path.abspath(str(tmp_path / "evidence.sqlite3"))
        }

    def test_a_later_runner_does_not_outrank_the_first_registration(
        self, folder, tmp_path
    ):
        benchmark = _benchmark(folder)
        first = setup.create_experiment(folder, benchmark["benchmark_id"], "v1")
        candidate = setup.duplicate_experiment(folder, first["experiment_id"])
        _store, controller = _controller(folder, str(tmp_path / "evidence.sqlite3"))

        # Only the CANDIDATE ever runs; the first was registered and never
        # executed. It is still the winner nobody has replaced.
        _declare(controller, candidate)

        assert _winner_id(folder, candidate["experiment_id"]) == first["experiment_id"]

    def test_two_runners_with_separate_stores_share_one_contest(
        self, folder, tmp_path
    ):
        benchmark = _benchmark(folder)
        first = setup.create_experiment(folder, benchmark["benchmark_id"], "v1")
        candidate = setup.duplicate_experiment(folder, first["experiment_id"])
        store_a, controller_a = _controller(folder, str(tmp_path / "a.sqlite3"))
        store_b, controller_b = _controller(folder, str(tmp_path / "b.sqlite3"))

        _declare(controller_a, first)
        _declare(controller_b, candidate)

        control = _control(folder)
        assert len(control.list_groups()) == 1
        group_id = control.list_groups()[0]["group_id"]
        assert {
            m["experiment_id"]: m["source_id"]
            for m in control.group_members(group_id)
        } == {
            first["experiment_id"]: store_a.store_identity(),
            candidate["experiment_id"]: store_b.store_identity(),
        }
        winner = control.current_winner(group_id)
        assert winner["experiment_id"] == first["experiment_id"]
        # Read from ITS store, which is store A.
        assert winner["experiment_resolved"] is True

    def test_an_unregistered_run_in_a_workflow_still_lands_in_the_contest(
        self, folder, tmp_path
    ):
        """A driver that never used the UI is still part of the workflow.

        It has no registration to enter, so its evidence row is what registers
        it — into the same control file, not a private one.
        """
        _store, controller = _controller(folder, str(tmp_path / "evidence.sqlite3"))

        controller.create_experiment(
            "exp-adhoc",
            "driver run",
            declared_tasks=1,
            declared_attempts=1,
            declarations=[("task_1", 1, "ch-1")],
            workflow_name=setup.workflow_name_for(folder),
        )

        assert _winner_id(folder, "exp-adhoc") == "exp-adhoc"


# ----------------------------------------------------------------------
# Automatic historical bootstrap
# ----------------------------------------------------------------------


class TestAutomaticBootstrap:
    def test_a_workflow_full_of_runs_elects_its_earliest_by_itself(
        self, folder, tmp_path
    ):
        """No adoption ceremony: creating the next experiment does it.

        The store below predates selection — its experiments were never
        registered anywhere. The new registration must not become the first
        winner of a lineage that has been running for months.
        """
        benchmark = _benchmark(folder)
        db_path = str(tmp_path / "evidence.sqlite3")
        store = obs.ObservabilityStore(db_path)
        for experiment_id in ("exp-old-1", "exp-old-2"):
            store.create_experiment(
                experiment_id,
                "historical",
                declared_tasks=1,
                declared_attempts=1,
                workflow_name=setup.workflow_name_for(folder),
                benchmark_id=benchmark["benchmark_id"],
                benchmark_version="v1",
                benchmark_digest_sha256=benchmark["digest_sha256"],
                initialize_winner=False,
            )
        setup.authorize_evidence_store(folder, store, db_path)

        record = setup.create_experiment(folder, benchmark["benchmark_id"], "v1")

        assert _winner_id(folder, record["experiment_id"]) == "exp-old-1"

    def test_an_unreadable_known_source_elects_nobody_and_says_so(
        self, folder, tmp_path
    ):
        """The reviewer gap: never elect while a known source is unresolved."""
        benchmark = _benchmark(folder)
        db_path = str(tmp_path / "evidence.sqlite3")
        store = obs.ObservabilityStore(db_path)
        store.create_experiment(
            "exp-old",
            "historical",
            declared_tasks=1,
            declared_attempts=1,
            workflow_name=setup.workflow_name_for(folder),
            benchmark_id=benchmark["benchmark_id"],
            benchmark_version="v1",
            benchmark_digest_sha256=benchmark["digest_sha256"],
            initialize_winner=False,
        )
        source_id = setup.authorize_evidence_store(folder, store, db_path)
        os.rename(db_path, db_path + ".moved")

        record = setup.create_experiment(folder, benchmark["benchmark_id"], "v1")

        report = setup.ensure_selection_bootstrap(folder)
        assert report["complete"] is False
        assert report["sources_skipped"] == [source_id]
        assert _winner_id(folder, record["experiment_id"]) is None

    def test_it_recovers_when_the_source_can_be_read_again(self, folder, tmp_path):
        benchmark = _benchmark(folder)
        db_path = str(tmp_path / "evidence.sqlite3")
        store = obs.ObservabilityStore(db_path)
        store.create_experiment(
            "exp-old",
            "historical",
            declared_tasks=1,
            declared_attempts=1,
            workflow_name=setup.workflow_name_for(folder),
            benchmark_id=benchmark["benchmark_id"],
            benchmark_version="v1",
            benchmark_digest_sha256=benchmark["digest_sha256"],
            initialize_winner=False,
        )
        setup.authorize_evidence_store(folder, store, db_path)
        os.rename(db_path, db_path + ".moved")
        record = setup.create_experiment(folder, benchmark["benchmark_id"], "v1")
        assert _winner_id(folder, record["experiment_id"]) is None

        os.rename(db_path + ".moved", db_path)
        report = setup.ensure_selection_bootstrap(folder)

        assert report["complete"] is True
        assert report["groups_without_winner"] == []
        assert _winner_id(folder, record["experiment_id"]) == "exp-old"

    def test_bootstrap_opens_evidence_read_only(self, folder, tmp_path):
        """Inspection that decides who won must not write to what it inspects."""
        benchmark = _benchmark(folder)
        db_path = str(tmp_path / "evidence.sqlite3")
        store = obs.ObservabilityStore(db_path)
        store.create_experiment(
            "exp-old",
            "historical",
            declared_tasks=1,
            declared_attempts=1,
            workflow_name=setup.workflow_name_for(folder),
            benchmark_id=benchmark["benchmark_id"],
            benchmark_version="v1",
            benchmark_digest_sha256=benchmark["digest_sha256"],
            initialize_winner=False,
        )
        setup.authorize_evidence_store(folder, store, db_path)
        resolve = setup._source_resolver(folder)

        resolved = resolve(store.store_identity())

        assert isinstance(resolved, obs.ReadOnlyObservabilityStore)

    def test_nothing_is_discovered_that_was_not_authorized(self, folder, tmp_path):
        """A database sitting next to a known one is not in the contest."""
        benchmark = _benchmark(folder)
        known = str(tmp_path / "known.sqlite3")
        stranger = str(tmp_path / "stranger.sqlite3")
        obs.ObservabilityStore(stranger).create_experiment(
            "exp-stranger",
            "not ours",
            declared_tasks=1,
            declared_attempts=1,
            workflow_name=setup.workflow_name_for(folder),
            benchmark_id=benchmark["benchmark_id"],
            benchmark_version="v1",
            benchmark_digest_sha256=benchmark["digest_sha256"],
            initialize_winner=False,
        )
        setup.authorize_evidence_store(folder, obs.ObservabilityStore(known), known)

        record = setup.create_experiment(folder, benchmark["benchmark_id"], "v1")

        assert _winner_id(folder, record["experiment_id"]) == record["experiment_id"]
        assert "exp-stranger" not in {
            m["experiment_id"]
            for m in _control(folder).group_members(
                _control(folder).group_for_experiment(record["experiment_id"])["group_id"]
            )
        }


def _forget_selection(folder):
    """Leave the workflow exactly as one that predates selection entirely.

    The registrations under `benchmarks/.experiments/` stay; the control file
    and the source map — everything selection ever wrote — go. That is the
    state of every workflow at the moment this feature ships, and the state
    these tests are about.
    """
    control = setup.workflow_control_db_path(folder)
    for path in (control, control + "-wal", control + "-shm",
                 str(setup._sources_path(folder))):
        if os.path.exists(path):
            os.remove(path)


class TestRegistrationsThatPredateSelection:
    """The upgrade case: experiments registered before any control existed.

    A workflow that has been registering and running experiments for weeks has
    all of that history in `.experiments/*.json` and none of it anywhere a
    control file can see. If the next experiment somebody creates is the first
    one the control hears about, it takes the title by default — which is the
    new candidate silently outranking months of older work.
    """

    def test_an_older_unrun_registration_still_outranks_a_new_one(self, folder):
        benchmark = _benchmark(folder)
        old = setup.create_experiment(folder, benchmark["benchmark_id"], "v1")
        _forget_selection(folder)

        fresh = setup.create_experiment(folder, benchmark["benchmark_id"], "v1")

        assert _winner_id(folder, fresh["experiment_id"]) == old["experiment_id"]
        # Seeded, not adopted: the older one never ran, so there is no evidence
        # anywhere that it exists.
        assert old["store"] is None

    def test_older_bound_registrations_across_two_stores_are_re_admitted(
        self, folder, tmp_path
    ):
        benchmark = _benchmark(folder)
        old = setup.create_experiment(folder, benchmark["benchmark_id"], "v1")
        newer = setup.duplicate_experiment(folder, old["experiment_id"])
        store_a, controller_a = _controller(folder, str(tmp_path / "a.sqlite3"))
        store_b, controller_b = _controller(folder, str(tmp_path / "b.sqlite3"))
        _declare(controller_a, old)
        _declare(controller_b, newer)
        _forget_selection(folder)

        fresh = setup.create_experiment(folder, benchmark["benchmark_id"], "v1")

        # Both stores came back through the registrations that name them —
        # explicit authorizations the user already made, not a search.
        assert {r["source_id"] for r in _control(folder).list_sources()} == {
            store_a.store_identity(),
            store_b.store_identity(),
        }
        assert _winner_id(folder, fresh["experiment_id"]) == old["experiment_id"]

    def test_an_older_missing_store_elects_nobody_until_it_returns(
        self, folder, tmp_path
    ):
        """Temporarily missing, not gone: an unmounted disk, a moved file."""
        benchmark = _benchmark(folder)
        old = setup.create_experiment(folder, benchmark["benchmark_id"], "v1")
        db_path = str(tmp_path / "a.sqlite3")
        store, controller = _controller(folder, db_path)
        identity = store.store_identity()
        _declare(controller, old)
        _forget_selection(folder)
        os.rename(db_path, db_path + ".unmounted")

        first = setup.ensure_selection_bootstrap(folder, create=True)

        assert first["complete"] is False
        assert [item["experiment_id"] for item in first["registrations_unavailable"]] == [
            old["experiment_id"]
        ]

        fresh = setup.create_experiment(folder, benchmark["benchmark_id"], "v1")

        assert _winner_id(folder, fresh["experiment_id"]) is None
        report = setup.ensure_selection_bootstrap(folder)
        assert report["complete"] is False
        # After the first bootstrap the gap is a SOURCE the control holds, not
        # a registration only the registration reader can see. That is the
        # whole point: a runner never reads `.experiments/`.
        assert report["sources_skipped"] == [identity]

        os.rename(db_path + ".unmounted", db_path)
        report = setup.ensure_selection_bootstrap(folder)

        assert report["complete"] is True
        assert report["registrations_unavailable"] == []
        assert _winner_id(folder, fresh["experiment_id"]) == old["experiment_id"]

    def test_a_readable_second_store_does_not_elect_while_the_older_is_missing(
        self, folder, tmp_path
    ):
        """The gap adoption cannot see for itself.

        Two bound registrations, two stores, and the OLDER store's file has
        gone. Seeding knows -- the registration names it -- and correctly
        registers without electing. Adoption then ran over the one store it
        could read and elected the newer experiment from it, because a store
        that was never authorized appears in nobody's skipped list: from
        inside the control, a missing file and a file that never existed look
        identical. The caller that knows the difference now says so.
        """
        benchmark = _benchmark(folder)
        old = setup.create_experiment(folder, benchmark["benchmark_id"], "v1")
        newer = setup.duplicate_experiment(folder, old["experiment_id"])
        db_a = str(tmp_path / "a.sqlite3")
        _store_a, controller_a = _controller(folder, db_a)
        _store_b, controller_b = _controller(folder, str(tmp_path / "b.sqlite3"))
        _declare(controller_a, old)
        _declare(controller_b, newer)
        _forget_selection(folder)
        os.rename(db_a, db_a + ".unmounted")

        report = setup.ensure_selection_bootstrap(folder, create=True)

        assert report["complete"] is False
        assert [item["experiment_id"] for item in report["registrations_unavailable"]] == [
            old["experiment_id"]
        ]
        assert _winner_id(folder, newer["experiment_id"]) is None
        # Both are still members: the candidate registration is preserved, it
        # simply holds no title while the view is partial.
        members = {
            row["experiment_id"]
            for row in _control(folder).group_members(
                _control(folder).group_for_experiment(newer["experiment_id"])["group_id"]
            )
        }
        assert members == {old["experiment_id"], newer["experiment_id"]}

        os.rename(db_a + ".unmounted", db_a)
        recovered = setup.ensure_selection_bootstrap(folder)

        assert recovered["complete"] is True
        assert _winner_id(folder, newer["experiment_id"]) == old["experiment_id"]

    def test_the_oldest_registration_wins_on_recovery_even_if_it_never_ran(
        self, folder, tmp_path
    ):
        """Delayed adoption must not settle on the oldest thing still readable.

        Three registrations: the oldest never ran at all, the middle one's
        store is temporarily gone, and the newest one's store reads fine.
        Electing from what is readable would crown the newest. Nobody is
        elected until the middle store returns, and then the title goes to the
        oldest registration -- which has no evidence anywhere and could never
        have been found by reading stores.
        """
        benchmark = _benchmark(folder)
        oldest = setup.create_experiment(folder, benchmark["benchmark_id"], "v1")
        middle = setup.duplicate_experiment(folder, oldest["experiment_id"])
        newest = setup.duplicate_experiment(folder, oldest["experiment_id"])
        db_middle = str(tmp_path / "middle.sqlite3")
        _store_m, controller_m = _controller(folder, db_middle)
        _store_n, controller_n = _controller(folder, str(tmp_path / "newest.sqlite3"))
        _declare(controller_m, middle)
        _declare(controller_n, newest)
        _forget_selection(folder)
        os.rename(db_middle, db_middle + ".unmounted")

        assert setup.ensure_selection_bootstrap(folder, create=True)["complete"] is False
        assert _winner_id(folder, newest["experiment_id"]) is None

        os.rename(db_middle + ".unmounted", db_middle)
        setup.ensure_selection_bootstrap(folder)

        assert _winner_id(folder, newest["experiment_id"]) == oldest["experiment_id"]
        assert oldest["store"] is None

    def test_a_runner_starting_while_a_store_is_missing_does_not_elect_its_own(
        self, folder, tmp_path
    ):
        """The other door into the same wrong winner.

        Seeding refusing to elect only governs seeding. A runner starting an
        hour later binds its own store and registers through the control
        directly, and the contest it meets looks complete -- the older
        registration is a member, and the store that went missing was never
        authorized, so nothing in the control knows it exists. Its brand-new
        experiment took the title over history nobody had opened.

        Declaring the bound-but-unreadable store closes this door and the
        seeding one with the same fact, which is why the runner needs no new
        flag of its own.
        """
        benchmark = _benchmark(folder)
        old = setup.create_experiment(folder, benchmark["benchmark_id"], "v1")
        db_old = str(tmp_path / "old.sqlite3")
        _store_old, controller_old = _controller(folder, db_old)
        _declare(controller_old, old)
        _forget_selection(folder)
        os.rename(db_old, db_old + ".unmounted")
        assert setup.ensure_selection_bootstrap(folder, create=True)["complete"] is False

        newcomer = setup.create_experiment(folder, benchmark["benchmark_id"], "v1")
        store_new, controller_new = _controller(folder, str(tmp_path / "new.sqlite3"))
        _declare(controller_new, newcomer)

        assert _winner_id(folder, newcomer["experiment_id"]) is None
        # The newcomer's own store IS in the contest -- it was refused a title,
        # not refused entry.
        assert store_new.store_identity() in {
            row["source_id"] for row in _control(folder).list_sources()
        }

        os.rename(db_old + ".unmounted", db_old)
        setup.ensure_selection_bootstrap(folder)

        assert _winner_id(folder, newcomer["experiment_id"]) == old["experiment_id"]

    def test_a_store_that_is_not_the_one_named_is_not_admitted(
        self, folder, tmp_path
    ):
        """Same path, different database. The recorded identity is checked."""
        benchmark = _benchmark(folder)
        old = setup.create_experiment(folder, benchmark["benchmark_id"], "v1")
        db_path = str(tmp_path / "a.sqlite3")
        _store, controller = _controller(folder, db_path)
        _declare(controller, old)
        _forget_selection(folder)
        os.remove(db_path)
        impostor = obs.ObservabilityStore(db_path)

        report = setup.ensure_selection_bootstrap(folder, create=True)

        assert report["complete"] is False
        assert impostor.store_identity() not in {
            r["source_id"] for r in _control(folder).list_sources()
        }

    def test_seeding_records_no_decision_it_did_not_make(self, folder, tmp_path):
        """Re-seeding is not a stream of new judgements.

        Bootstrap runs on every registration and every runner start. If it
        re-recorded a decision each time, a task's history would fill with
        promotions nobody performed.
        """
        benchmark = _benchmark(folder)
        record = setup.create_experiment(folder, benchmark["benchmark_id"], "v1")
        control = _control(folder)
        group_id = control.group_for_experiment(record["experiment_id"])["group_id"]
        before = control.decision_history(group_id)

        for _ in range(3):
            setup.ensure_selection_bootstrap(folder)

        assert control.decision_history(group_id) == before


# ----------------------------------------------------------------------
# Two upgrade paths into the same wrong winner
# ----------------------------------------------------------------------


def _historical_experiment(store, folder, benchmark, experiment_id):
    """One experiment recorded the way a pre-selection version recorded them.

    No registration, no control, no winner initialized: the evidence row is the
    whole of what exists, which is exactly what an upgraded workflow has.
    """
    store.create_experiment(
        experiment_id,
        "historical",
        declared_tasks=1,
        declared_attempts=1,
        workflow_name=setup.workflow_name_for(folder),
        benchmark_id=benchmark["benchmark_id"],
        benchmark_version="v1",
        benchmark_digest_sha256=benchmark["digest_sha256"],
        initialize_winner=False,
    )


class TestTheDefaultStoreIsPartOfTheHistory:
    """The upgrade case with NOTHING for the bootstrap to be told about.

    `TestAutomaticBootstrap` above authorizes the historical store by hand
    first, which is the adoption ceremony the design says there is not: a real
    upgraded workflow has its runs in the one database this workflow records
    into by default, and nobody ever called `authorize_evidence_store` for it.
    Nothing here does either.
    """

    def test_history_in_the_default_store_holds_the_contest(self, folder):
        benchmark = _benchmark(folder)
        default_db = state_paths.observability_db(str(folder))
        Path(default_db).parent.mkdir(parents=True, exist_ok=True)
        store = obs.ObservabilityStore(default_db)
        _historical_experiment(store, folder, benchmark, "exp-preexisting")

        record = setup.create_experiment(folder, benchmark["benchmark_id"], "v1")

        # Without the default store in the contest, the brand-new registration
        # is the only thing the control has ever seen, and wins.
        assert _winner_id(folder, record["experiment_id"]) == "exp-preexisting"

    def test_a_runner_starting_later_reaches_the_same_answer(self, folder, tmp_path):
        """The other door in: recovery, not registration.

        A runner opens the control directly. It must meet the same history,
        or the first `fastworkflow run` after an upgrade elects its own run.
        """
        benchmark = _benchmark(folder)
        default_db = state_paths.observability_db(str(folder))
        Path(default_db).parent.mkdir(parents=True, exist_ok=True)
        store = obs.ObservabilityStore(default_db)
        _historical_experiment(store, folder, benchmark, "exp-preexisting")

        runner_db = str(tmp_path / "runner.sqlite3")
        runner_store = obs.ObservabilityStore(runner_db)
        controller = ExperimentController(
            runner_db, runner_store.store_identity(), external=False,
            workflow_folderpath=str(folder),
        )
        controller.create_experiment(
            "exp-newcomer", "started by a runner", declared_tasks=1,
            declared_attempts=1, declarations=[("task_1", 1, "ch-1")],
            workflow_name=setup.workflow_name_for(folder),
            # The same benchmark, which is what puts it in the same contest as
            # the history: a run of something else is not competing with it.
            benchmark_id=benchmark["benchmark_id"], benchmark_version="v1",
            benchmark_digest_sha256=benchmark["digest_sha256"],
        )

        assert _winner_id(folder, "exp-newcomer") == "exp-preexisting"

    def test_a_workflow_that_has_recorded_nothing_has_no_default_store(self, folder):
        """And none is created to look for history in.

        A bootstrap that brought an evidence database into existence would be
        indistinguishable afterwards from one whose evidence was lost.
        """
        benchmark = _benchmark(folder)
        default_db = state_paths.observability_db(str(folder))

        record = setup.create_experiment(folder, benchmark["benchmark_id"], "v1")

        assert not os.path.exists(default_db)
        # Nothing older exists, so the new registration is the winner.
        assert _winner_id(folder, record["experiment_id"]) == record["experiment_id"]

    def test_an_unreadable_default_store_elects_nobody(self, folder):
        """Unreadable is not empty, on this path as on every other."""
        benchmark = _benchmark(folder)
        default_db = state_paths.observability_db(str(folder))
        Path(default_db).parent.mkdir(parents=True, exist_ok=True)
        Path(default_db).write_bytes(b"this is not a database")

        record = setup.create_experiment(folder, benchmark["benchmark_id"], "v1")
        report = setup.ensure_selection_bootstrap(folder)

        assert report["complete"] is False
        assert any(
            os.path.abspath(str(item.get("db_path"))) == os.path.abspath(default_db)
            for item in report["registrations_unavailable"]
        ), report["registrations_unavailable"]
        assert _winner_id(folder, record["experiment_id"]) is None


class TestTheDeferredElectionGoesToTheOldestMember:
    """`fix-kkod`: the withheld election, when it finally happens, is the
    group's OLDEST member's -- not whichever registration arrives first.

    An incomplete bootstrap registers members and elects nobody, so by the time
    the view is whole the older history is already a MEMBER. The older-peer
    scan only looks for non-members, and seeding runs before history adoption,
    so the first registration re-seeded used to take the pointer over history
    that was sitting right there in the group.
    """

    def test_a_rebuilt_control_elects_the_historical_experiment(
        self, folder, tmp_path
    ):
        """The reproduction: control rebuilt while ANOTHER group's store is
        unreadable; history for this group lives only in the default store."""
        roster = _benchmark(folder, "Roster")
        payroll = _benchmark(folder, "Payroll")
        default_db = state_paths.observability_db(str(folder))
        Path(default_db).parent.mkdir(parents=True, exist_ok=True)
        _historical_experiment(
            obs.ObservabilityStore(default_db), folder, roster, "exp-historical"
        )
        other = setup.create_experiment(folder, payroll["benchmark_id"], "v1")
        other_db = str(tmp_path / "payroll.sqlite3")
        _store, controller = _controller(folder, other_db)
        _declare(controller, other)
        newer = setup.create_experiment(folder, roster["benchmark_id"], "v1")
        assert _winner_id(folder, newer["experiment_id"]) == "exp-historical"

        _forget_selection(folder)
        os.rename(other_db, other_db + ".unmounted")
        partial = setup.ensure_selection_bootstrap(folder, create=True)
        assert partial["complete"] is False
        group_id = str(_control(folder).group_for_experiment(
            newer["experiment_id"])["group_id"])
        # Both are members already, and nobody is elected.
        assert {r["experiment_id"] for r in _control(folder).group_members(group_id)} == {
            "exp-historical", newer["experiment_id"]
        }
        assert _winner_id(folder, newer["experiment_id"]) is None

        os.rename(other_db + ".unmounted", other_db)
        report = setup.ensure_selection_bootstrap(folder)

        assert report["complete"] is True
        assert _winner_id(folder, newer["experiment_id"]) == "exp-historical"
        history = _control(folder).decision_history(group_id)
        assert [(r["seq"], r["decision"], r["new_experiment_id"]) for r in history] == [
            (1, "initial", "exp-historical")
        ]

    def test_the_newer_member_registering_first_elects_the_older_one(self, folder):
        """The rule itself, on the control: the caller is not the candidate."""
        control = selection.open_shared_control(
            selection.shared_control_db_path_for(str(folder))
        )
        older = selection.ExperimentReference(
            experiment_id="exp-zzz-older", workflow_name="w", benchmark_id="b",
            created_at="2026-01-01T00:00:00+00:00",
        )
        newer = selection.ExperimentReference(
            experiment_id="exp-aaa-newer", workflow_name="w", benchmark_id="b",
            created_at="2026-01-02T00:00:00+00:00",
        )
        for reference in (newer, older):
            control.register_experiment_reference(reference, allow_initial_winner=False)
        group_id = str(control.group_for_experiment("exp-zzz-older")["group_id"])

        result = control.register_experiment_reference(newer)

        assert result["initialized"] is True
        assert control.current_winner(group_id)["experiment_id"] == "exp-zzz-older"
        (row,) = control.decision_history(group_id)
        assert (row["decision"], row["candidate_experiment_id"], row["new_experiment_id"]) == (
            "initial", "exp-zzz-older", "exp-zzz-older"
        )


class TestARunnerStartingWhileTheHistoryIsUnreadable:
    """The third door into the wrong first winner, and the one with no id.

    `test_an_unreadable_default_store_elects_nobody` above covers the
    REGISTRATION path, which asks the bootstrap whether the view was whole and
    passes the answer on. A runner asks nothing: it calls
    `bind_runner_evidence`, and a control path coming back means "elect". The
    two gaps that already have a known store id are caught a second time
    inside the control, by the durable declaration the resolver checks. This
    one has no id to declare -- the default store cannot be opened, so nobody
    knows what identity it holds -- and from inside the control the contest
    looks complete.
    """

    def _unreadable_default(self, folder):
        default_db = state_paths.observability_db(str(folder))
        Path(default_db).parent.mkdir(parents=True, exist_ok=True)
        Path(default_db).write_bytes(b"this is not a database")
        return default_db

    def _run_one(self, folder, tmp_path, benchmark, experiment_id="exp-runner"):
        runner_db = str(tmp_path / f"{experiment_id}.sqlite3")
        store, controller = _controller(folder, runner_db)
        controller.create_experiment(
            experiment_id, "started by a runner", declared_tasks=1,
            declared_attempts=1, declarations=[("task_1", 1, "ch-1")],
            workflow_name=setup.workflow_name_for(folder),
            # The same pin as the registration, which is what puts this run in
            # the same contest as the history it must not outrank.
            benchmark_id=benchmark["benchmark_id"], benchmark_version="v1",
            benchmark_digest_sha256=benchmark["digest_sha256"],
        )
        return store, runner_db

    def test_a_runner_does_not_elect_itself_over_history_it_cannot_read(
        self, folder, tmp_path
    ):
        benchmark = _benchmark(folder)
        default_db = self._unreadable_default(folder)
        before = Path(default_db).read_bytes()
        record = setup.create_experiment(folder, benchmark["benchmark_id"], "v1")
        assert _winner_id(folder, record["experiment_id"]) is None

        _store, runner_db = self._run_one(folder, tmp_path, benchmark)

        assert setup.ensure_selection_bootstrap(folder)["complete"] is False
        assert _winner_id(folder, "exp-runner") is None
        # And the registration created a moment earlier did not acquire a
        # title through the runner either: they share one group.
        assert _winner_id(folder, record["experiment_id"]) is None
        # Refusing the title is not falling back to a private contest.
        assert not os.path.exists(selection.control_db_path_for(runner_db))
        # Deciding who the earliest experiment is wrote nothing to the
        # evidence file it could not read.
        assert Path(default_db).read_bytes() == before

    def test_the_runner_is_admitted_even_though_it_is_refused_a_title(
        self, folder, tmp_path
    ):
        """Refused entry and refused a title are different refusals.

        If the store stayed out of the contest, recovery would need somebody
        to authorize it by hand -- the adoption ceremony this design says
        there is not.
        """
        benchmark = _benchmark(folder)
        self._unreadable_default(folder)
        setup.create_experiment(folder, benchmark["benchmark_id"], "v1")

        store, runner_db = self._run_one(folder, tmp_path, benchmark)

        assert store.store_identity() in {
            str(row["source_id"]) for row in _control(folder).list_sources()
        }
        assert setup.known_evidence_sources(folder).get(store.store_identity()) == \
            os.path.abspath(runner_db)

    def test_the_restored_history_elects_its_oldest_and_not_the_runner(
        self, folder, tmp_path
    ):
        """Recovery, with the run that started during the outage in the group.

        The history is real here, not a placeholder: an experiment recorded in
        the default store the way a pre-selection version recorded them, moved
        aside so the path is unreadable rather than absent, and put back.
        """
        benchmark = _benchmark(folder)
        default_db = state_paths.observability_db(str(folder))
        Path(default_db).parent.mkdir(parents=True, exist_ok=True)
        history = obs.ObservabilityStore(default_db)
        _historical_experiment(history, folder, benchmark, "exp-preexisting")
        kept = str(tmp_path / "history.sqlite3")
        source = sqlite3.connect(default_db)
        try:
            target = sqlite3.connect(kept)
            try:
                source.backup(target)
            finally:
                target.close()
        finally:
            source.close()
        for suffix in ("", "-wal", "-shm"):
            if os.path.exists(default_db + suffix):
                os.remove(default_db + suffix)
        Path(default_db).write_bytes(b"this is not a database")

        record = setup.create_experiment(folder, benchmark["benchmark_id"], "v1")
        self._run_one(folder, tmp_path, benchmark)
        assert _winner_id(folder, record["experiment_id"]) is None

        shutil.copy(kept, default_db)
        report = setup.ensure_selection_bootstrap(folder)

        assert report["complete"] is True
        # The oldest RECORDED experiment, over both the registration created
        # during the outage and the run that started in it.
        assert _winner_id(folder, "exp-runner") == "exp-preexisting"
        assert _winner_id(folder, record["experiment_id"]) == "exp-preexisting"

    def test_an_older_unrun_registration_still_outranks_them_both(
        self, folder, tmp_path
    ):
        """The oldest thing in the contest may have no evidence at all.

        Recovery must not settle on the oldest experiment it can READ: a
        registration nobody ever ran is older than both, and the history
        coming back is what makes it electable rather than what beats it.
        """
        benchmark = _benchmark(folder)
        oldest = setup.create_experiment(folder, benchmark["benchmark_id"], "v1")
        default_db = state_paths.observability_db(str(folder))
        Path(default_db).parent.mkdir(parents=True, exist_ok=True)
        history = obs.ObservabilityStore(default_db)
        _historical_experiment(history, folder, benchmark, "exp-preexisting")
        kept = str(tmp_path / "history.sqlite3")
        source = sqlite3.connect(default_db)
        try:
            target = sqlite3.connect(kept)
            try:
                source.backup(target)
            finally:
                target.close()
        finally:
            source.close()
        for suffix in ("", "-wal", "-shm"):
            if os.path.exists(default_db + suffix):
                os.remove(default_db + suffix)
        Path(default_db).write_bytes(b"this is not a database")
        _forget_selection(folder)

        self._run_one(folder, tmp_path, benchmark)
        assert _winner_id(folder, "exp-runner") is None

        shutil.copy(kept, default_db)
        setup.ensure_selection_bootstrap(folder, create=True)

        assert _winner_id(folder, "exp-runner") == oldest["experiment_id"]
        assert oldest["store"] is None

    def test_a_readable_default_store_still_lets_a_runner_record_a_winner(
        self, folder, tmp_path
    ):
        """The refusal is the exception, not the new rule.

        A workflow whose history reads fine must keep electing automatically,
        or "do not elect from a partial view" has quietly become "do not
        elect".
        """
        benchmark = _benchmark(folder)
        default_db = state_paths.observability_db(str(folder))
        Path(default_db).parent.mkdir(parents=True, exist_ok=True)
        _historical_experiment(
            obs.ObservabilityStore(default_db), folder, benchmark, "exp-preexisting"
        )

        store, runner_db = self._run_one(folder, tmp_path, benchmark)

        assert setup.bind_runner_evidence(folder, store, runner_db) == \
            setup.workflow_control_db_path(folder)
        assert _winner_id(folder, "exp-runner") == "exp-preexisting"
        assert not os.path.exists(selection.control_db_path_for(runner_db))

    def test_a_workflow_with_no_history_at_all_still_elects_its_runner(
        self, folder, tmp_path
    ):
        """Nothing to read is not the same as something unreadable."""
        benchmark = _benchmark(folder)

        self._run_one(folder, tmp_path, benchmark)

        assert _winner_id(folder, "exp-runner") == "exp-runner"


class TestAStoreReplacedByADifferentOne:
    """A registration's store swapped for a different database entirely.

    Restored from the wrong backup, or a fresh file created where the old one
    stood. The identity does not match what the registration recorded, so the
    old runs are NOT what is at that path -- and the contest must not conclude
    that they never existed.
    """

    def _bound_experiment(self, folder, tmp_path, benchmark):
        db_path = str(tmp_path / "bound.sqlite3")
        store = obs.ObservabilityStore(db_path)
        record = setup.create_experiment(folder, benchmark["benchmark_id"], "v1")
        setup.bind_experiment(
            folder, record["experiment_id"], db_path, store.store_identity()
        )
        _historical_experiment(store, folder, benchmark, record["experiment_id"])
        return record, db_path, store

    def _swap_in_an_impostor(self, tmp_path, store, db_path):
        """A different real store at the same path. The original is kept.

        The writer is closed first and the copy is taken with SQLite's own
        backup API: swapping bytes under a live WAL connection produces a
        corrupt database on both sides, which would test the wrong failure.
        """
        impostor = str(tmp_path / "impostor.sqlite3")
        other = obs.ObservabilityStore(impostor)
        assert other.store_identity()
        assert store.store_identity()
        keep = db_path + ".original"
        source = sqlite3.connect(db_path)
        try:
            target = sqlite3.connect(keep)
            try:
                source.backup(target)
            finally:
                target.close()
        finally:
            source.close()
        for suffix in ("", "-wal", "-shm"):
            if os.path.exists(db_path + suffix):
                os.remove(db_path + suffix)
        shutil.copy(impostor, db_path)
        return keep

    def test_a_runner_starting_later_still_refuses_to_elect(self, folder, tmp_path):
        """The gap this closes: the refusal has to OUTLIVE the bootstrap call.

        Registration-time seeding gets this right on its own, because it reads
        the registration and finds the mismatch. The runner an hour later does
        not: it opens the control, sees a contest whose visible history is
        complete, and elects its own brand-new experiment over runs nobody
        could read. So the mismatch is recorded IN the control, where every
        path into it meets the guard that is already there.
        """
        benchmark = _benchmark(folder)
        first, db_path, store = self._bound_experiment(folder, tmp_path, benchmark)
        self._swap_in_an_impostor(tmp_path, store, db_path)
        # The control is discarded, which is what makes this a bootstrap: a
        # workflow whose decisions were never recorded, or were lost with it.
        control_path = setup.workflow_control_db_path(folder)
        if os.path.exists(control_path):
            os.remove(control_path)

        report = setup.ensure_selection_bootstrap(folder, create=True)
        assert report["complete"] is False

        runner_db = str(tmp_path / "runner.sqlite3")
        runner_store = obs.ObservabilityStore(runner_db)
        controller = ExperimentController(
            runner_db, runner_store.store_identity(), external=False,
            workflow_folderpath=str(folder),
        )
        controller.create_experiment(
            "exp-newcomer", "started by a runner", declared_tasks=1,
            declared_attempts=1, declarations=[("task_1", 1, "ch-1")],
            workflow_name=setup.workflow_name_for(folder),
        )

        assert _winner_id(folder, "exp-newcomer") is None
        # The EXPECTED identity is in the contest; the impostor's is not.
        identities = {
            str(row["source_id"]) for row in _control(folder).list_sources()
        }
        assert first["experiment_id"]
        assert obs.ObservabilityStore(db_path, migrate=False).store_identity() \
            not in identities

    def test_the_impostor_never_answers_for_the_id_it_is_not(self, folder, tmp_path):
        """Resolution is identity-checked, so the wrong file cannot stand in.

        Without this the declaration above would be worse than nothing: the
        control would hand the impostor back for the old source id and raise
        its mismatch error on every read instead of reporting an unreadable
        source.
        """
        benchmark = _benchmark(folder)
        _first, db_path, store = self._bound_experiment(folder, tmp_path, benchmark)
        source_id = setup.authorize_evidence_store(folder, store, db_path)
        self._swap_in_an_impostor(tmp_path, store, db_path)

        resolved = setup.open_workflow_control(folder).store_for_source(source_id)

        assert resolved is None

    def test_the_original_coming_back_recovers_by_itself(self, folder, tmp_path):
        """No decision to undo and no ceremony: the next bootstrap elects."""
        benchmark = _benchmark(folder)
        first, db_path, store = self._bound_experiment(folder, tmp_path, benchmark)
        keep = self._swap_in_an_impostor(tmp_path, store, db_path)
        control_path = setup.workflow_control_db_path(folder)
        if os.path.exists(control_path):
            os.remove(control_path)
        assert setup.ensure_selection_bootstrap(folder, create=True)["complete"] is False
        setup.create_experiment(folder, benchmark["benchmark_id"], "v1")
        assert _winner_id(folder, first["experiment_id"]) is None

        for suffix in ("", "-wal", "-shm"):
            if os.path.exists(db_path + suffix):
                os.remove(db_path + suffix)
        shutil.copy(keep, db_path)
        report = setup.ensure_selection_bootstrap(folder)

        assert report["complete"] is True
        assert _winner_id(folder, first["experiment_id"]) == first["experiment_id"]

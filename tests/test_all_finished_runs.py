"""Every finished run of one task, and whether that population has moved.

`fix-9eg.3.2.2.1` and `fix-9eg.3.2.2.2`. Integration throughout, per
`.cursor/rules/testing_rules.mdc`: real `ObservabilityStore` databases, real
attempts written through the real `ExperimentController`, the real selection
control, the real HTTP server over a real socket. The store is mutated for
real between a summary and a check, because "has the population changed" is
only a question when the population can actually change under the reader.

What these tests defend:

- The RULE and the LIST agree. `scope=all_finished` and an explicit selection
  of the same runs produce the same members, the same figures and the same
  evidence digest, because the rule decides membership and nothing else.
- DETECTION is not capped even though DETAIL is. A run recorded beyond the
  listed unfinished ids, or removed beyond it, still reads as changed.
- A baseline is SCOPE-BOUND. Attempt numbers exist under every task, so a
  baseline echoed under another task or experiment is refused rather than
  answered "unchanged" -- including when neither population has a finished run.
- Population drift and evidence change are DIFFERENT answers. Neither sets
  the other's flag.
"""

from __future__ import annotations

import os
import subprocess
import threading
from contextlib import contextmanager
from pathlib import Path

import pytest

from fastworkflow import state_paths
from fastworkflow.benchmark import setup
from fastworkflow.experiment.runner import ExperimentController
from fastworkflow.observability import selected_runs as sr
from fastworkflow.observability import store as obs
from fastworkflow.run_chatbot import selection_api
from fastworkflow.run_chatbot import server as run_chatbot_server
from tests.test_chatbot_benchmarks import _request
from tests.test_selection_api import _sealed_server, _seed_turn

pytest_plugins = ["tests.test_selection_api", "tests.test_selected_runs"]


# ----------------------------------------------------------------------
# Calling the routes
# ----------------------------------------------------------------------


def _get(folder, experiment_id, task_id, suffix="", **params):
    query: dict[str, list[str]] = {}
    for key, value in params.items():
        query[key] = (
            [str(item) for item in value]
            if isinstance(value, (list, tuple))
            else [str(value)]
        )
    path = (
        f"/api/experiments/{experiment_id}/tasks/{task_id}/selected-runs{suffix}"
    )
    return selection_api.handle_get(folder, path, query)


def _summary(world, task_id=None, **params):
    return _get(world["folder"], world["experiment_id"],
                task_id or world["task_id"], **params)


def _validate(world, task_id=None, **params):
    return _get(world["folder"], world["experiment_id"],
                task_id or world["task_id"], suffix="/validation", **params)


def _baseline_params(payload, **overrides):
    """The baseline a client echoes back, exactly as a summary published it."""
    population = payload["task_population"]
    params = {
        "population_scope": population["population_scope"],
        "expect_population": population["population_digest"],
        "population_rule": population["selection_rule"],
        "baseline_member": population["finished_attempts"],
        "baseline_member_count": population["finished"],
        "baseline_unfinished": population["unfinished_attempts"],
        "baseline_unfinished_complete": (
            "true" if population["unfinished_listing_complete"] else "false"
        ),
    }
    params.update(overrides)
    return params


def _query(population, **overrides):
    """The same baseline as a query string, for the real-socket tests."""
    params = {
        "population_scope": population["population_scope"],
        "expect_population": population["population_digest"],
        "population_rule": population["selection_rule"],
        "baseline_member_count": str(population["finished"]),
        "baseline_unfinished_complete": (
            "true" if population["unfinished_listing_complete"] else "false"
        ),
    }
    params.update({key: str(value) for key, value in overrides.items()})
    parts = [f"{key}={value}" for key, value in params.items()]
    parts += [f"baseline_member={a}" for a in population["finished_attempts"]]
    parts += [f"baseline_unfinished={a}" for a in population["unfinished_attempts"]]
    return "?" + "&".join(parts)


# ----------------------------------------------------------------------
# A task whose population can be moved under the reader
# ----------------------------------------------------------------------


@pytest.fixture
def drift_world(tmp_path, monkeypatch):
    """Three tasks of one experiment, each recording a different population.

    `task` has two finished runs (one passed, one failed) and one still
    running. `empty_task` has one run still going and nothing finished, which
    is the honest zero case. `big_task` has twenty-one finished runs, which is
    one more than a single request will summarize.

    Only `task` records turns, and only so the page has a conversation
    hierarchy to open. Every figure these tests are about is resolved from
    ATTEMPT metadata, and a population check that needed a turn read would be
    the thing under test failing.
    """
    monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "state"))
    folder = tmp_path / "all_finished_workflow"
    folder.mkdir()
    (folder / "_commands").mkdir()
    db = state_paths.observability_db(str(folder))
    store = obs.ObservabilityStore(db)
    benchmark = setup.save_benchmark(
        folder,
        {
            "title": "Populations",
            "tasks": [{"prompt": "one"}, {"prompt": "two"}, {"prompt": "three"}],
        },
    )
    experiment = setup.create_experiment(
        folder, benchmark["benchmark_id"], "v1", runs_per_task=3
    )
    experiment_id = experiment["experiment_id"]
    task_id, empty_task, big_task = experiment["task_ids"][:3]

    controller = ExperimentController(
        str(folder), store.store_identity(), external=False
    )
    # The plan is exact and rectangular: every task declares the same number
    # of attempts. What each task actually RECORDS is what differs here, and
    # a declared attempt nobody started records no row at all.
    declarations = [
        (task, n, f"{prefix}-{n}")
        for task, prefix in ((task_id, "a"), (empty_task, "b"), (big_task, "c"))
        for n in range(1, 22)
    ]
    controller.create_experiment(
        experiment_id,
        experiment["description"],
        declared_tasks=3,
        declared_attempts=21,
        declarations=declarations,
        workflow_name=setup.workflow_name_for(folder),
    )

    def start(task, attempt, channel, *, turn=False):
        conversation = store.mint_conversation_id(
            channel, experiment_id=experiment_id, task_id=task, attempt=attempt
        )
        controller.start_attempt(
            experiment_id, task, attempt, channel, conversation_id=conversation
        )
        if turn:
            _seed_turn(
                store, f"fin-{task[:8]}-a{attempt}", experiment_id=experiment_id,
                task_id=task, attempt=attempt, conversation=conversation,
                ordinal=1, commands=["add_item"],
            )

    def finish(task, attempt, outcome="pass", status="completed"):
        controller.finish_attempt(
            experiment_id, task, attempt, outcome=outcome,
            outcome_source="test", execution_status=status,
        )

    start(task_id, 1, "a-1", turn=True)
    finish(task_id, 1)
    start(task_id, 2, "a-2", turn=True)
    finish(task_id, 2, outcome="fail", status="failed")
    start(task_id, 3, "a-3", turn=True)  # deliberately never finished
    start(empty_task, 1, "b-1")  # nothing finished here at all
    for attempt in range(1, 22):
        start(big_task, attempt, f"c-{attempt}")
        finish(big_task, attempt)

    return {
        "folder": str(folder),
        "db": db,
        "experiment_id": experiment_id,
        "task_id": task_id,
        "empty_task": empty_task,
        "big_task": big_task,
        "store": store,
        "controller": controller,
        "start": start,
        "finish": finish,
    }


@pytest.fixture
def drift_server(drift_world):
    srv = run_chatbot_server.ChatbotServer(
        db_path=drift_world["db"],
        workflow_path=drift_world["folder"],
        port=0,
        spawn_options={"no_server": True},
    )
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    thread.join(timeout=5)


def _remove_attempt(store, experiment_id, task_id, attempt):
    """Delete one recorded attempt row, the way a pruned database would.

    Real removal of real metadata: the population check has to notice a run
    that used to be there and is not, and an id-only comparison against the
    eligible set would not.
    """
    with store._connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "DELETE FROM experiment_attempts WHERE experiment_id=? AND "
            "task_id=? AND attempt=?",
            (experiment_id, task_id, int(attempt)),
        )
        conn.commit()


# ----------------------------------------------------------------------
# The rule resolves the same population an explicit list names
# ----------------------------------------------------------------------


class TestTheRuleAgreesWithTheList:
    def test_all_finished_equals_an_explicit_selection_of_the_same_runs(
        self, runs_world
    ):
        """The rule decides MEMBERSHIP and nothing else, so every figure and
        the evidence digest itself have to be the ones an explicit selection
        of those runs publishes."""
        status, rule = _summary(runs_world, scope="all_finished")
        assert status == 200
        members = [member["attempt"] for member in rule["members"]]
        assert members == [1, 2, 4]

        status, listed = _summary(runs_world, attempt=[1, 2, 4])
        assert status == 200
        assert [m["attempt"] for m in listed["members"]] == members
        assert rule["command_summary"]["totals"] == listed["command_summary"]["totals"]
        assert rule["command_summary"]["groups"] == listed["command_summary"]["groups"]
        assert rule["cost"] == listed["cost"]
        assert rule["run_outcomes"] == listed["run_outcomes"]
        assert rule["population"] == listed["population"]
        # The identity too: how a set was chosen is not part of what it
        # recorded, so the rule must not move the evidence digest.
        assert rule["evidence_digest"] == listed["evidence_digest"]
        assert [m["evidence_digest"] for m in rule["members"]] == [
            m["evidence_digest"] for m in listed["members"]
        ]
        assert rule["scope"]["selection_rule"] == "all_finished"
        assert listed["scope"]["selection_rule"] == "explicit"

    def test_the_population_includes_failures_and_runs_with_no_evidence(
        self, runs_world
    ):
        """Whole finished runs, and all of them. Attempt 2 failed and attempt
        4 finished having recorded nothing; dropping either would shrink a
        denominator the summary claims to be over."""
        _status, payload = _summary(runs_world, scope="all_finished")
        outcomes = payload["run_outcomes"]["by_outcome"]
        assert outcomes["pass"] == 2 and outcomes["fail"] == 1
        assert payload["population"]["missing_evidence"] == [4]
        assert payload["population"]["included"] == 3
        assert payload["task_population"]["unfinished_attempts"] == [3]

    def test_the_task_population_counts_runs_and_the_recorded_task_plan(
        self, runs_world
    ):
        """`runs_per_task` is a plan for THIS task and is reported as one.
        Nothing derives a task plan from an experiment-wide declared total."""
        _status, payload = _summary(runs_world, scope="all_finished")
        population = payload["task_population"]
        assert population["recorded"] == 4
        assert population["finished"] == 3
        assert population["unfinished"] == 1
        assert population["finished_attempts"] == [1, 2, 4]
        assert population["planned"] == 4  # the fixture registered 4 per task
        assert population["planned_source"] == "experiment_registration.runs_per_task"
        assert "planned_note" not in population

    def test_a_task_plan_that_is_not_recorded_is_absent_rather_than_invented(
        self, drift_world
    ):
        _status, payload = _summary(drift_world, scope="all_finished")
        population = payload["task_population"]
        assert population["planned"] == 3
        # 26 attempts were declared across three tasks; none of that is a plan
        # for this one, and the per-task figure is the only one reported.
        assert population["recorded"] == 3


# ----------------------------------------------------------------------
# What the route refuses to be asked
# ----------------------------------------------------------------------


class TestScopeRefusals:
    def test_a_rule_and_a_list_together_are_refused(self, drift_world):
        status, payload = _summary(drift_world, scope="all_finished", attempt=[1])
        assert status == 400
        assert payload["refused"] == "mixed_scope"

    def test_a_scope_this_route_does_not_resolve_is_refused_by_name(
        self, drift_world
    ):
        for scope in ("all_runs", "unfinished", "everything", "ALL_FINISHED"):
            status, payload = _summary(drift_world, scope=scope)
            assert status == 400, scope
            assert payload["refused"] == "unsupported_scope"
            assert payload["accepted"] == ["all_finished"]

    def test_the_whole_request_is_refused_above_the_bound_with_its_counts(
        self, drift_world
    ):
        status, payload = _summary(drift_world, drift_world["big_task"],
                                   scope="all_finished")
        assert status == 400
        assert payload["refused"] == "too_many_runs"
        assert payload["max_runs"] == sr.MAX_SELECTED_RUNS == 20
        assert payload["finished"] == 21 and payload["recorded"] == 21
        assert payload["unfinished"] == 0
        # Nothing was summarized and nothing was sampled.
        assert "command_summary" not in payload and "members" not in payload
        assert "select the runs you mean explicitly" in payload["error"]

    def test_an_explicit_selection_of_the_same_runs_still_works_under_the_bound(
        self, drift_world
    ):
        status, payload = _summary(drift_world, drift_world["big_task"],
                                   attempt=list(range(1, 21)))
        assert status == 200
        assert payload["member_count"] == 20


# ----------------------------------------------------------------------
# Zero finished runs is an answer
# ----------------------------------------------------------------------


class TestZeroFinished:
    def test_a_task_with_nothing_finished_answers_honestly(self, drift_world):
        status, payload = _summary(drift_world, drift_world["empty_task"],
                                   scope="all_finished")
        assert status == 200
        assert payload["members"] == [] and payload["member_count"] == 0
        assert payload["population"]["included"] == 0
        # Never 0: a zero here would be a claim that these runs spent nothing.
        assert payload["cost"]["total"] is None
        population = payload["task_population"]
        assert population["recorded"] == 1 and population["finished"] == 0
        assert population["unfinished_attempts"] == [1]

    def test_a_zero_finished_population_is_still_scope_bound(self, drift_world):
        """The identity comes from authorized metadata, not from a member --
        there is none. Two different tasks with no finished run must not share
        a baseline."""
        _status, empty = _summary(drift_world, drift_world["empty_task"],
                                  scope="all_finished")
        status, refused = _validate(
            drift_world, drift_world["task_id"],
            **_baseline_params(empty),
        )
        assert status == 400
        assert refused["refused"] == "foreign_population_baseline"

    def test_a_zero_finished_population_can_still_be_checked(self, drift_world):
        _status, empty = _summary(drift_world, drift_world["empty_task"],
                                  scope="all_finished")
        status, answer = _validate(drift_world, drift_world["empty_task"],
                                   **_baseline_params(empty))
        assert status == 200
        # No attempt was named, so no run was re-projected.
        assert answer["members"] == [] and answer["stale"] is None
        assert answer["population_check"]["changed"] is False

        drift_world["finish"](drift_world["empty_task"], 1)
        _status, after = _validate(drift_world, drift_world["empty_task"],
                                   **_baseline_params(empty))
        check = after["population_check"]
        assert check["changed"] is True
        assert check["newly_finished"] == [1]
        assert check["added_members"] == [1]


# ----------------------------------------------------------------------
# Population drift, against real mutations of a real store
# ----------------------------------------------------------------------


class TestPopulationDrift:
    def test_a_validation_that_asked_nothing_about_the_population_says_nothing(
        self, drift_world
    ):
        """The legacy drill-down shape is untouched: no baseline, no
        population block, and no unbounded attempt array bolted onto it."""
        _status, answer = _validate(drift_world, attempt=[1])
        assert answer["population_check"] is None
        assert answer["task_population"] is None

    def test_an_explicit_summary_does_not_publish_the_whole_task_population(
        self, drift_world
    ):
        _status, payload = _summary(drift_world, attempt=[1, 2])
        assert payload["task_population"] is None
        assert payload["scope"]["selection_rule"] == "explicit"

    def test_an_unchanged_population_reads_as_unchanged(self, drift_world):
        _status, payload = _summary(drift_world, scope="all_finished")
        _status, answer = _validate(drift_world, attempt=[1],
                                    **_baseline_params(payload))
        check = answer["population_check"]
        assert check["compared"] is True and check["changed"] is False
        assert check["still_members"] == [1, 2]
        assert check["undetailed"] is False

    def test_an_unfinished_run_becoming_eligible_is_detected(self, drift_world):
        """The ids did not change. Attempt 3 was already recorded; what moved
        is its eligibility, which an id-set comparison cannot see."""
        _status, payload = _summary(drift_world, scope="all_finished")
        drift_world["finish"](drift_world["task_id"], 3)
        _status, answer = _validate(drift_world, attempt=[1],
                                    **_baseline_params(payload))
        check = answer["population_check"]
        assert check["changed"] is True
        assert check["newly_finished"] == [3]
        assert check["newly_recorded_finished"] == []
        assert check["added_members"] == [3]
        assert check["still_members"] == [1, 2]
        # The member that was asked about is untouched by any of it.
        assert answer["stale"] is None or answer["stale"] is False

    def test_a_newly_recorded_finished_run_is_detected(self, drift_world):
        _status, payload = _summary(drift_world, scope="all_finished")
        drift_world["start"](drift_world["task_id"], 4, "a-4")
        drift_world["finish"](drift_world["task_id"], 4)
        _status, answer = _validate(drift_world, attempt=[1],
                                    **_baseline_params(payload))
        check = answer["population_check"]
        assert check["changed"] is True
        assert check["newly_recorded_finished"] == [4]
        assert check["newly_finished"] == []

    def test_a_newly_recorded_run_that_is_still_going_is_detected(
        self, drift_world
    ):
        """It will never appear in the eligible set, so a check that only
        compared eligible ids would report this population as unchanged."""
        _status, payload = _summary(drift_world, scope="all_finished")
        drift_world["start"](drift_world["task_id"], 4, "a-4")
        _status, answer = _validate(drift_world, attempt=[1],
                                    **_baseline_params(payload))
        check = answer["population_check"]
        assert check["changed"] is True
        assert check["added_members"] == []
        assert check["newly_recorded_unfinished"] == [4]

    def test_a_removed_member_and_a_removed_unfinished_run_are_detected(
        self, drift_world
    ):
        _status, payload = _summary(drift_world, scope="all_finished")
        _remove_attempt(drift_world["store"], drift_world["experiment_id"],
                        drift_world["task_id"], 2)
        _remove_attempt(drift_world["store"], drift_world["experiment_id"],
                        drift_world["task_id"], 3)
        _status, answer = _validate(drift_world, attempt=[1],
                                    **_baseline_params(payload))
        check = answer["population_check"]
        assert check["changed"] is True
        assert check["removed_finished"] == [2]
        assert check["removed_unfinished"] == [3]
        assert check["still_members"] == [1]

    def test_a_member_that_stopped_reading_as_finished_is_detected(
        self, drift_world
    ):
        _status, payload = _summary(drift_world, scope="all_finished")
        with drift_world["store"]._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "UPDATE experiment_attempts SET execution_finished_at=NULL "
                "WHERE experiment_id=? AND task_id=? AND attempt=?",
                (drift_world["experiment_id"], drift_world["task_id"], 2),
            )
            conn.commit()
        _status, answer = _validate(drift_world, attempt=[1],
                                    **_baseline_params(payload))
        check = answer["population_check"]
        assert check["changed"] is True
        assert check["lost_eligibility"] == [2]

    def test_a_change_beyond_the_listed_ids_is_still_detected(self, drift_world):
        """Capped DETAIL is acceptable; capped DETECTION is not.

        The baseline here lists no unfinished run and says so, so the change
        cannot be attributed to any id it named -- and the answer says THAT
        rather than reporting nothing.
        """
        _status, payload = _summary(drift_world, scope="all_finished")
        drift_world["start"](drift_world["task_id"], 4, "a-4")
        _status, answer = _validate(
            drift_world, attempt=[1],
            **_baseline_params(payload, baseline_unfinished=[],
                               baseline_unfinished_complete="false"),
        )
        check = answer["population_check"]
        assert check["changed"] is True
        assert check["detail_complete"] is False
        assert check["newly_recorded_unfinished"] == []
        assert check["undetailed"] is True

    def test_the_same_attempt_ids_under_another_task_cannot_validate(
        self, drift_world
    ):
        """Every task has an attempt 1. A baseline that was only a set of
        numbers would compare clean against a population it was never about."""
        _status, payload = _summary(drift_world, scope="all_finished")
        status, refused = _validate(drift_world, drift_world["big_task"],
                                    attempt=[1], **_baseline_params(payload))
        assert status == 400
        assert refused["refused"] == "foreign_population_baseline"

    def test_a_baseline_from_another_experiment_cannot_validate(self, world):
        """The two experiments of the shared fixture record the same task in
        two different stores, with the same attempt numbers in both."""
        first, second = world["experiment_id"], world["candidate_id"]
        _status, payload = _get(world["folder"], first, world["task_id"],
                                scope="all_finished")
        status, refused = _get(
            world["folder"], second, world["task_id"], suffix="/validation",
            attempt=[1], **_baseline_params(payload),
        )
        assert status == 400
        assert refused["refused"] == "foreign_population_baseline"

    def test_a_baseline_missing_its_scope_or_digest_is_refused(self, drift_world):
        _status, payload = _summary(drift_world, scope="all_finished")
        for omitted in ("expect_population", "population_scope",
                        "baseline_member_count", "baseline_unfinished_complete"):
            params = _baseline_params(payload)
            params.pop(omitted)
            status, refused = _validate(drift_world, attempt=[1], **params)
            assert status == 400, omitted
            assert refused["refused"] == "incomplete_population_baseline"
            assert refused["missing"] == [omitted]


class TestAMalformedBaselineIsRefusedRatherThanAnswered:
    """An omitted list is not an empty one.

    Without the count, a client that simply forgot to send its members would
    be told that every run of the task is newly eligible -- a wrong answer
    built out of a missing parameter rather than out of anything that changed.
    """

    def test_forgetting_the_member_list_is_not_read_as_an_empty_population(
        self, drift_world
    ):
        _status, payload = _summary(drift_world, scope="all_finished")
        params = _baseline_params(payload)
        params["baseline_member"] = []
        status, refused = _validate(drift_world, attempt=[1], **params)
        assert status == 400
        assert refused["refused"] == "inconsistent_population_baseline"
        assert "baseline_member_count says 2" in refused["error"]

    def test_duplicate_and_overlapping_baseline_ids_are_refused(
        self, drift_world
    ):
        _status, payload = _summary(drift_world, scope="all_finished")
        status, refused = _validate(
            drift_world, attempt=[1],
            **_baseline_params(payload, baseline_member=[1, 1],
                               baseline_member_count=2),
        )
        assert status == 400
        assert refused["refused"] == "inconsistent_population_baseline"

        status, refused = _validate(
            drift_world, attempt=[1],
            **_baseline_params(payload, baseline_unfinished=[2, 3],
                               baseline_unfinished_complete="false"),
        )
        assert status == 400
        # Attempt 2 is a member and cannot also be an unfinished run.
        assert refused["refused"] == "inconsistent_population_baseline"

    def test_a_repeated_singleton_parameter_is_refused(self, drift_world):
        _status, payload = _summary(drift_world, scope="all_finished")
        population = payload["task_population"]
        status, refused = _validate(
            drift_world, attempt=[1],
            **_baseline_params(
                payload,
                expect_population=[population["population_digest"],
                                   "pop-0000000000000000"],
            ),
        )
        assert status == 400
        assert "at most once" in refused["error"]

    def test_a_complete_baseline_that_does_not_hash_to_its_digest_is_refused(
        self, drift_world
    ):
        """The lists and the digest have to be about one population. Answered
        instead, this would report `changed` from the digest beside set diffs
        drawn from ids that digest never saw."""
        _status, payload = _summary(drift_world, scope="all_finished")
        status, refused = _validate(
            drift_world, attempt=[1],
            **_baseline_params(payload, baseline_member=[1, 99],
                               baseline_member_count=2),
        )
        assert status == 400
        assert refused["refused"] == "inconsistent_population_baseline"

    def test_a_truncated_baseline_that_contradicts_its_own_digest_is_refused(
        self, drift_world
    ):
        """Truncation excuses missing DETAIL, not a member list that
        disagrees with a digest claiming nothing moved."""
        _status, payload = _summary(drift_world, scope="all_finished")
        status, refused = _validate(
            drift_world, attempt=[1],
            **_baseline_params(payload, baseline_member=[1],
                               baseline_member_count=1,
                               baseline_unfinished=[],
                               baseline_unfinished_complete="false"),
        )
        assert status == 400
        assert refused["refused"] == "inconsistent_population_baseline"

    def test_more_baseline_members_than_a_summary_can_have_is_refused(
        self, drift_world
    ):
        _status, payload = _summary(drift_world, scope="all_finished")
        status, refused = _validate(
            drift_world, attempt=[1],
            **_baseline_params(payload,
                               baseline_member=list(range(1, 25)),
                               baseline_member_count=24),
        )
        assert status == 400
        assert refused["refused"] == "too_many_parameters"


# ----------------------------------------------------------------------
# Drift and evidence change are different answers
# ----------------------------------------------------------------------


class TestDriftAndEvidenceAreSeparate:
    def test_a_moved_population_does_not_make_a_member_stale(self, runs_world):
        """The run that was still going finishes. That is news about the task
        and about nothing this summary counted, so the member it is asked
        about must still read as unchanged."""
        _status, payload = _summary(runs_world, scope="all_finished")
        member = payload["members"][0]
        with runs_world["store"]._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "UPDATE experiment_attempts SET execution_finished_at=? "
                "WHERE experiment_id=? AND task_id=? AND attempt=3",
                ("2026-09-20T12:00:00+00:00", runs_world["experiment_id"],
                 runs_world["task_id"]),
            )
            conn.commit()
        status, answer = _validate(
            runs_world, attempt=[member["attempt"]],
            expect_member=f"{member['attempt']}:{member['evidence_digest']}",
            **_baseline_params(payload),
        )
        assert status == 200
        assert answer["stale"] is False
        assert answer["members"][0]["state"] == "unchanged"
        # Two different answers on one payload, and neither sets the other.
        check = answer["population_check"]
        assert check["changed"] is True and check["newly_finished"] == [3]

    def test_changed_evidence_does_not_make_the_population_changed(
        self, runs_world
    ):
        from tests.test_selected_runs import _edit_recorded_outcome

        _status, payload = _summary(runs_world, scope="all_finished")
        member = next(m for m in payload["members"] if m["attempt"] == 2)
        _edit_recorded_outcome(runs_world["store"], "sel-a2-t1", 0, False)
        _status, answer = _validate(
            runs_world, attempt=[2],
            expect_member=f"2:{member['evidence_digest']}",
            **_baseline_params(payload),
        )
        assert answer["stale"] is True and answer["changed"] == [2]
        # The runs that EXIST did not move, and saying they did would send a
        # reader to refresh a membership that is already right.
        assert answer["population_check"]["changed"] is False


# ----------------------------------------------------------------------
# Over real HTTP, and in a sealed archive
# ----------------------------------------------------------------------


class TestOverHttp:
    def test_the_rule_and_its_refusals_answer_over_a_real_socket(
        self, drift_world, drift_server
    ):
        base = (
            f"/api/experiments/{drift_world['experiment_id']}"
            f"/tasks/{drift_world['task_id']}/selected-runs"
        )
        status, payload = _request(drift_server, base + "?scope=all_finished")
        assert status == 200
        assert [m["attempt"] for m in payload["members"]] == [1, 2]
        assert payload["task_population"]["unfinished"] == 1

        status, refused = _request(
            drift_server, base + "?scope=all_finished&attempt=1"
        )
        assert status == 400 and refused["refused"] == "mixed_scope"

        status, refused = _request(drift_server, base + "?scope=all_runs")
        assert status == 400 and refused["refused"] == "unsupported_scope"

    def test_a_population_check_answers_over_a_real_socket(
        self, drift_world, drift_server
    ):
        base = (
            f"/api/experiments/{drift_world['experiment_id']}"
            f"/tasks/{drift_world['task_id']}/selected-runs"
        )
        _status, payload = _request(drift_server, base + "?scope=all_finished")
        query = _query(payload["task_population"])
        status, answer = _request(drift_server, base + "/validation" + query)
        assert status == 200
        assert answer["population_check"]["changed"] is False
        drift_world["finish"](drift_world["task_id"], 3)
        _status, after = _request(drift_server, base + "/validation" + query)
        assert after["population_check"]["changed"] is True
        assert after["population_check"]["newly_finished"] == [3]


class TestSealedArchive:
    """A sealed archive carries evidence, so the rule is a read of it and is
    answered rather than refused with the decisions wording."""

    def test_all_finished_reads_an_archive_the_way_a_selection_does(
        self, tmp_path, monkeypatch
    ):
        srv, thread = _sealed_server(tmp_path, monkeypatch)
        base = "/api/experiments/logical/tasks/task/selected-runs"
        try:
            status, payload = _request(srv, base + "?scope=all_finished")
            assert status == 200, payload
            assert payload["sealed"] is True
            assert payload["scope"]["selection_rule"] == "all_finished"
            attempts = [member["attempt"] for member in payload["members"]]
            assert attempts
            for member in payload["members"]:
                # Both of an archive's names survive the rule: the reference's
                # evidence identity and the manifest's own name for it.
                assert member["manifest_store_id"] and member["store_id"]
            status, listed = _request(
                srv, base + "?" + "&".join(f"attempt={a}" for a in attempts)
            )
            assert status == 200
            assert listed["evidence_digest"] == payload["evidence_digest"]
            population = payload["task_population"]
            assert population["sealed"] is True
            # No live registration behind an archive, so no task plan.
            assert population["planned"] is None
            assert "planned_note" in population

            status, refused = _request(
                srv, base + "?scope=all_finished&attempt=1"
            )
            assert status == 400 and refused["refused"] == "mixed_scope"
        finally:
            srv.shutdown()
            thread.join(timeout=5)

    def test_an_archive_population_can_be_checked_and_is_scope_bound(
        self, tmp_path, monkeypatch
    ):
        srv, thread = _sealed_server(tmp_path, monkeypatch)
        base = "/api/experiments/logical/tasks/task/selected-runs"
        try:
            _status, payload = _request(srv, base + "?scope=all_finished")
            query = _query(payload["task_population"])
            status, answer = _request(srv, base + "/validation" + query)
            assert status == 200
            assert answer["population_check"]["changed"] is False
            # An archive's population identity is bound to the archive, so a
            # foreign baseline is refused there too.
            status, refused = _request(
                srv,
                base + "/validation?population_scope=pscope-0000000000000000"
                "&expect_population=pop-0000000000000000"
                "&baseline_member_count=0&baseline_unfinished_complete=true",
            )
            assert status == 400
            assert refused["refused"] == "foreign_population_baseline"
        finally:
            srv.shutdown()
            thread.join(timeout=5)


def _archive(tmp_path, name, *, finished=(), unfinished=()):
    """One sealed archive recording the named attempts of one task."""
    source = str(tmp_path / f"{name}-live.sqlite3")
    store = obs.ObservabilityStore(source)
    store.create_experiment(
        "local", f"archived {name}", declared_tasks=1,
        declared_attempts=len(finished) + len(unfinished),
    )
    for attempt in list(finished) + list(unfinished):
        store.start_attempt("local", "task", attempt, f"{name}-{attempt}")
    for attempt in finished:
        store.finish_attempt(
            "local", "task", attempt, outcome="pass", outcome_source="test"
        )
    return obs.ObservabilityStore(source, migrate=False).archive_to(
        str(tmp_path / f"{name}.sqlite3")
    )


def _stitched_server(tmp_path, monkeypatch, segments):
    from tests.test_observability_workspace import _manifest, _store_decl

    monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "state"))
    stores = [_store_decl(archive, store_id) for store_id, archive in segments]
    manifest = _manifest(
        tmp_path, stores,
        experiments=[{
            "experiment_id": "logical",
            "segments": [
                {"store_id": store_id, "local_experiment_id": "local",
                 "segment_id": store_id}
                for store_id, _archive in segments
            ],
        }],
    )
    srv = run_chatbot_server.ChatbotServer(
        port=0, workspace_manifest_path=str(manifest)
    )
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    return srv, thread


class TestAnArchivePopulationIsBoundToOneArchive:
    """A logical experiment can be stitched from several archives.

    Its runs of a task are then not one population, and hashing them under a
    source of "no single archive" would give two different stitchings the same
    identity -- so a baseline taken under one could validate under the other.
    That is refused. A named segment is how a caller says which archive it
    means, and resolving it from the MANIFEST rather than from whichever rows
    exist is what binds an archive that finished nothing.
    """

    def test_a_population_spanning_two_archives_is_refused_not_hashed(
        self, tmp_path, monkeypatch
    ):
        srv, thread = _stitched_server(tmp_path, monkeypatch, [
            ("first", _archive(tmp_path, "first", finished=(1,))),
            ("second", _archive(tmp_path, "second", unfinished=(2,))),
        ])
        base = "/api/experiments/logical/tasks/task/selected-runs"
        try:
            status, refused = _request(srv, base + "?scope=all_finished")
            assert status == 409
            assert "more than one archive" in refused["error"]
            assert refused["store_ids"] == ["first", "second"]

            # Naming the archive answers it, bound to that archive alone.
            status, first = _request(
                srv, base + "?scope=all_finished&segment_id=first"
            )
            assert status == 200
            assert [m["attempt"] for m in first["members"]] == [1]
            assert first["task_population"]["recorded"] == 1

            # The archive that finished nothing is scope-bound all the same,
            # from the manifest rather than from a member it does not have.
            status, second = _request(
                srv, base + "?scope=all_finished&segment_id=second"
            )
            assert status == 200
            assert second["members"] == []
            population = second["task_population"]
            assert population["recorded"] == 1 and population["finished"] == 0
            assert population["unfinished_attempts"] == [2]
            assert (
                population["population_scope"]
                != first["task_population"]["population_scope"]
            )
        finally:
            srv.shutdown()
            thread.join(timeout=5)

    def test_an_explicit_selection_across_archives_is_unaffected(
        self, tmp_path, monkeypatch
    ):
        """No population was asked about, so nothing new refuses the request
        the archive answered before."""
        srv, thread = _stitched_server(tmp_path, monkeypatch, [
            ("first", _archive(tmp_path, "first", finished=(1,))),
            ("second", _archive(tmp_path, "second", unfinished=(2,))),
        ])
        base = "/api/experiments/logical/tasks/task/selected-runs"
        try:
            status, payload = _request(srv, base + "?attempt=1")
            assert status == 200
            assert [m["attempt"] for m in payload["members"]] == [1]
            assert payload["task_population"] is None
        finally:
            srv.shutdown()
            thread.join(timeout=5)

    def test_a_baseline_cannot_cross_between_two_archives(
        self, tmp_path, monkeypatch
    ):
        """The same attempt number, finished in one archive, in the other."""
        srv, thread = _stitched_server(tmp_path, monkeypatch, [
            ("first", _archive(tmp_path, "first", finished=(1,))),
            ("second", _archive(tmp_path, "second", finished=(1,))),
        ])
        base = "/api/experiments/logical/tasks/task/selected-runs"
        try:
            _status, first = _request(
                srv, base + "?scope=all_finished&segment_id=first"
            )
            query = _query(first["task_population"])
            status, answer = _request(
                srv, base + "/validation" + query + "&segment_id=first"
            )
            assert status == 200
            assert answer["population_check"]["changed"] is False
            status, refused = _request(
                srv, base + "/validation" + query + "&segment_id=second"
            )
            assert status == 400
            assert refused["refused"] == "foreign_population_baseline"
        finally:
            srv.shutdown()
            thread.join(timeout=5)


class TestAnArchiveIsReadForWhatWasAsked:
    """What a population question costs, and what a member reference holds.

    Both were wrong in the same place. Answering "which runs exist" by reading
    every run's turns reads the runs a request is about to refuse or leave
    out, and the read it used stopped at a fixed bound without saying so, so a
    long run's reference described a shorter run than the archive holds.
    """

    def test_a_population_question_reads_no_turn_of_any_run(
        self, tmp_path, monkeypatch
    ):
        srv, thread = _sealed_server(tmp_path, monkeypatch)
        base = "/api/experiments/logical/tasks/task/selected-runs"
        try:
            _status, payload = _request(srv, base + "?scope=all_finished")
            query = _query(payload["task_population"])
            reads = []
            registry = srv.workspace.registry
            opened = registry.open

            @contextmanager
            def counting(store_id):
                with opened(store_id) as store:
                    listed = store.list_turns

                    def watched(*args, **kwargs):
                        reads.append(kwargs.get("attempt"))
                        return listed(*args, **kwargs)

                    store.list_turns = watched
                    yield store

            registry.open = counting
            try:
                # The metadata-only check: no member named, so no run read.
                status, answer = _request(srv, base + "/validation" + query)
                assert status == 200
                assert answer["population_check"]["changed"] is False
                assert reads == [], f"turns were read for attempts {reads}"

                # And a summary reads the turns of its MEMBERS, not of every
                # run the task recorded.
                reads.clear()
                status, one = _request(srv, base + "?attempt=1")
                assert status == 200
                assert reads and set(reads) == {1}, (
                    f"attempts whose turns were read: {sorted(set(reads))}"
                )
            finally:
                registry.open = opened
        finally:
            srv.shutdown()
            thread.join(timeout=5)

    def test_an_over_limit_archive_is_refused_without_reading_its_runs(
        self, tmp_path, monkeypatch
    ):
        from tests.test_observability_workspace import _manifest, _store_decl

        monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "state"))
        archive = _archive(tmp_path, "big", finished=tuple(range(1, 22)))
        manifest = _manifest(
            tmp_path, [_store_decl(archive, "big")],
            experiments=[{"experiment_id": "logical", "segments": [
                {"store_id": "big", "local_experiment_id": "local",
                 "segment_id": "big"}]}],
        )
        srv = run_chatbot_server.ChatbotServer(
            port=0, workspace_manifest_path=str(manifest)
        )
        thread = threading.Thread(target=srv.serve_forever, daemon=True)
        thread.start()
        try:
            reads = []
            registry = srv.workspace.registry
            opened = registry.open

            @contextmanager
            def counting(store_id):
                with opened(store_id) as store:
                    listed = store.list_turns

                    def watched(*args, **kwargs):
                        reads.append(kwargs.get("attempt"))
                        return listed(*args, **kwargs)

                    store.list_turns = watched
                    yield store

            registry.open = counting
            try:
                status, refused = _request(
                    srv,
                    "/api/experiments/logical/tasks/task/selected-runs"
                    "?scope=all_finished",
                )
                assert status == 400
                assert refused["refused"] == "too_many_runs"
                assert refused["finished"] == 21
                assert reads == [], (
                    "a task too large to summarize must not be scanned to say "
                    f"so; turns were read for {reads}"
                )
            finally:
                registry.open = opened
        finally:
            srv.shutdown()
            thread.join(timeout=5)

    def test_a_member_is_read_against_its_own_segment_not_its_number(
        self, tmp_path, monkeypatch
    ):
        """One archive, two segments, an attempt 1 in each.

        An attempt NUMBER is not a run. Resolving a member by looking its
        number up again could read the other segment's turns and hand them to
        this segment's row -- the same figures under the wrong run.
        """
        from tests.test_observability_workspace import _manifest, _store_decl

        monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "state"))
        source = str(tmp_path / "dual-live.sqlite3")
        store = obs.ObservabilityStore(source)
        for local, turns in (("one", 1), ("two", 3)):
            store.create_experiment(
                local, f"segment {local}", declared_tasks=1, declared_attempts=1
            )
            store.start_attempt(local, "task", 1, f"{local}-1")
            conversation = store.mint_conversation_id(
                f"{local}-1", experiment_id=local, task_id="task", attempt=1
            )
            for ordinal in range(turns):
                _seed_turn(
                    store, f"{local}-a1-t{ordinal}", experiment_id=local,
                    task_id="task", attempt=1, conversation=conversation,
                    ordinal=ordinal + 1, commands=["add_item"],
                )
            store.finish_attempt(
                local, "task", 1, outcome="pass", outcome_source="test"
            )
        archive = obs.ObservabilityStore(source, migrate=False).archive_to(
            str(tmp_path / "dual.sqlite3")
        )
        manifest = _manifest(
            tmp_path, [_store_decl(archive, "dual")],
            experiments=[{"experiment_id": "logical", "segments": [
                {"store_id": "dual", "local_experiment_id": "one",
                 "segment_id": "s-one"},
                {"store_id": "dual", "local_experiment_id": "two",
                 "segment_id": "s-two"},
            ]}],
        )
        srv = run_chatbot_server.ChatbotServer(
            port=0, workspace_manifest_path=str(manifest)
        )
        thread = threading.Thread(target=srv.serve_forever, daemon=True)
        thread.start()
        base = "/api/experiments/logical/tasks/task/selected-runs"
        try:
            # Without a segment there is no one population to be about.
            status, refused = _request(srv, base + "?scope=all_finished")
            assert status == 409
            assert "more than one segment" in refused["error"]

            reads = []
            registry = srv.workspace.registry
            opened = registry.open

            @contextmanager
            def counting(store_id):
                with opened(store_id) as opened_store:
                    listed = opened_store.list_turns

                    def watched(*args, **kwargs):
                        reads.append(kwargs.get("experiment_id"))
                        return listed(*args, **kwargs)

                    opened_store.list_turns = watched
                    yield opened_store

            registry.open = counting
            try:
                status, payload = _request(
                    srv, base + "?scope=all_finished&segment_id=s-two"
                )
            finally:
                registry.open = opened
            assert status == 200
            member = payload["members"][0]
            assert member["attempt"] == 1
            assert member["turn_count"] == 3, (
                "the three turns segment s-two recorded, not the one s-one did"
            )
            assert member["execution_ref"]["experiment_id"] == "two"
            assert set(reads) == {"two"}, (
                f"turns were read from segments {sorted(set(reads))}"
            )
        finally:
            srv.shutdown()
            thread.join(timeout=5)

    def test_a_member_reference_holds_every_turn_the_archive_recorded(
        self, tmp_path, monkeypatch
    ):
        """More turns than one page of the enumeration, so the reference is
        only whole if the pages are walked to the end."""
        from fastworkflow.observability import workspace as workspace_module
        from tests.test_observability_workspace import _manifest, _store_decl

        monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "state"))
        page = workspace_module._TURN_PAGE
        total = page + 7
        source = str(tmp_path / "long-live.sqlite3")
        store = obs.ObservabilityStore(source)
        store.create_experiment(
            "local", "a long run", declared_tasks=1, declared_attempts=1
        )
        store.start_attempt("local", "task", 1, "long-1")
        conversation = store.mint_conversation_id(
            "long-1", experiment_id="local", task_id="task", attempt=1
        )
        for ordinal in range(total):
            _seed_turn(
                store, f"long-{ordinal:06d}", experiment_id="local",
                task_id="task", attempt=1, conversation=conversation,
                ordinal=ordinal + 1, commands=["add_item"],
            )
        store.finish_attempt(
            "local", "task", 1, outcome="pass", outcome_source="test"
        )
        archive = obs.ObservabilityStore(source, migrate=False).archive_to(
            str(tmp_path / "long.sqlite3")
        )
        manifest = _manifest(
            tmp_path, [_store_decl(archive, "long")],
            experiments=[{"experiment_id": "logical", "segments": [
                {"store_id": "long", "local_experiment_id": "local"}]}],
        )
        workspace = workspace_module.load_observability_workspace(str(manifest))
        rows = workspace.attempts("logical", task_id="task")
        keys = [ref["logical_turn_key"] for ref in rows[0]["turn_refs"]]
        assert len(keys) == total, (
            f"the archive holds {total} turns and the reference names "
            f"{len(keys)}"
        )
        assert len(set(keys)) == total, "no turn is named twice by paging"
        # Unchanged order: this route returns turn keys descending, and each
        # page continues the last rather than restarting it.
        assert keys == sorted(keys, reverse=True)

        # Metadata only reads no turn and reports no reference at all, rather
        # than reporting that the run recorded nothing.
        bare = workspace.attempts("logical", task_id="task", turn_refs_for=())
        assert "turn_refs" not in bare[0]


@pytest.fixture
def crowded_world(tmp_path, monkeypatch):
    """One task with more finished runs than any list may name.

    The rule refuses a task this size, so explicitly selecting a few of its
    runs is the fallback -- and that fallback has to keep working, including
    the drill-down into one of them.
    """
    monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "state"))
    folder = tmp_path / "crowded_workflow"
    folder.mkdir()
    (folder / "_commands").mkdir()
    db = state_paths.observability_db(str(folder))
    store = obs.ObservabilityStore(db)
    benchmark = setup.save_benchmark(
        folder, {"title": "Crowded", "tasks": [{"prompt": "one"}]}
    )
    experiment = setup.create_experiment(
        folder, benchmark["benchmark_id"], "v1", runs_per_task=100
    )
    experiment_id = experiment["experiment_id"]
    task_id = experiment["task_ids"][0]
    controller = ExperimentController(
        str(folder), store.store_identity(), external=False
    )
    controller.create_experiment(
        experiment_id, experiment["description"], declared_tasks=1,
        declared_attempts=205,
        declarations=[(task_id, n, f"x-{n}") for n in range(1, 206)],
        workflow_name=setup.workflow_name_for(folder),
    )
    for attempt in range(1, 206):
        controller.start_attempt(experiment_id, task_id, attempt, f"x-{attempt}")
        controller.finish_attempt(
            experiment_id, task_id, attempt, outcome="pass", outcome_source="test"
        )
    return {"folder": str(folder), "db": db, "experiment_id": experiment_id,
            "task_id": task_id}


def test_selecting_a_few_runs_of_a_crowded_task_still_works(crowded_world):
    """The fallback the over-limit refusal names. Nothing about the new
    population semantics may burden a request that did not ask for them: a
    task with 205 finished runs must not put 205 attempt numbers on a
    drill-down."""
    status, payload = _summary(crowded_world, scope="all_finished")
    assert status == 400 and payload["refused"] == "too_many_runs"
    assert payload["finished"] == 205

    status, chosen = _summary(crowded_world, attempt=[7])
    assert status == 200
    assert chosen["task_population"] is None
    member = chosen["members"][0]
    status, answer = _validate(
        crowded_world, attempt=[7],
        expect_member=f"7:{member['evidence_digest']}",
    )
    assert status == 200
    assert answer["stale"] is False
    assert answer["population_check"] is None


# ----------------------------------------------------------------------
# The shipped page, in a real DOM
# ----------------------------------------------------------------------


def _dom(server, world, phase, *, on_ready=None, on_steps=None):
    """Drive the shipped page in jsdom, mutating the store where it asks.

    `on_steps` maps a handshake the harness prints to what the store does at
    that point. A phase may need more than one -- editing a run's recorded
    answer and finishing another run are two different mutations at two
    different moments -- and each one is answered on stdin so the page carries
    on only after the real store has really changed.
    """
    jsdom_root = os.environ.get("TEST_JSDOM_ROOT")
    if not jsdom_root:
        pytest.skip("Set TEST_JSDOM_ROOT to run DOM integration with jsdom")
    command = [
        "node", str(Path(__file__).with_name("chatbot_all_finished_dom.cjs")),
        jsdom_root,
        f"http://127.0.0.1:{server.port}/?token={server.token}",
        world["experiment_id"],
        world["task_id"],
        phase,
    ]
    steps = dict(on_steps or {})
    if on_ready is not None:
        steps["READY-FOR-MUTATION"] = on_ready
    if not steps:
        result = subprocess.run(command, capture_output=True, text=True,
                                timeout=180)
        assert result.returncode == 0, result.stdout + result.stderr
        return
    process = subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True,
    )
    try:
        pending = dict(steps)
        for line in process.stdout:
            token = next((name for name in pending if name in line), None)
            if token is None:
                continue
            pending.pop(token)()
            process.stdin.write("go\n")
            process.stdin.flush()
            if not pending:
                break
        stdout, stderr = process.communicate(timeout=180)
    finally:
        if process.poll() is None:
            process.kill()
    assert process.returncode == 0, stdout + stderr
    assert not pending, f"the page never asked for {sorted(pending)}: {stdout}"


def test_the_page_summarizes_every_finished_run_of_a_task(
    drift_server, drift_world
):
    """Driven through the real page against the real server: the rule's own
    button, the population counts, and the runs the server resolved."""
    _dom(drift_server, drift_world, "all")


def test_the_page_checks_for_run_changes_without_replacing_what_is_shown(
    drift_server, drift_world
):
    """A real run finishes under the page. The explicit check says so and the
    membership on screen does not move; an explicit refresh then includes it."""
    _dom(
        drift_server, drift_world, "drift",
        on_ready=lambda: drift_world["finish"](drift_world["task_id"], 3),
    )


def test_the_page_reports_an_over_limit_population_without_summarizing_part(
    drift_server, drift_world
):
    world = dict(drift_world, task_id=drift_world["big_task"])
    _dom(drift_server, world, "overlimit")


# ----------------------------------------------------------------------
# One notice, two requests reporting on the same population (fix-kphi)
# ----------------------------------------------------------------------


def _first_turn_of(world, attempt):
    """The turn key `drift_world` recorded for one attempt of its task."""
    return f"fin-{world['task_id'][:8]}-a{attempt}"


def _make_member_stale(world, attempt):
    """Edit what one member's run recorded, in place, so opening it is refused.

    Real evidence change against the real store -- same turn key, same spans,
    same everything the summary names -- which is what keeps the reader on the
    panel looking at the notice instead of navigating away from it.
    """
    from tests.test_selected_runs import _edit_recorded_outcome

    _edit_recorded_outcome(world["store"], _first_turn_of(world, attempt), 0, False)


def test_an_older_contributor_check_cannot_overwrite_a_newer_population_notice(
    drift_server, drift_world
):
    """`fix-kphi`, in a real DOM against the real server.

    The contributor check and the explicit "Check for run changes" both report
    on the same population into the same notice. Here the contributor check is
    requested FIRST, over a population in which attempt 3 was still running,
    and answers LAST -- after attempt 3 really finished and the explicit check
    said so. Its older reading must not replace the newer one, because a reader
    told "the membership below is unchanged" has no reason to refresh.
    """
    _dom(
        drift_server, drift_world, "notice-order",
        on_steps={
            "READY-FOR-EVIDENCE": lambda: _make_member_stale(drift_world, 1),
            "READY-FOR-MUTATION": lambda: drift_world["finish"](
                drift_world["task_id"], 3
            ),
        },
    )


def test_an_older_run_change_check_cannot_overwrite_a_newer_population_notice(
    drift_server, drift_world
):
    """The same ordering, the other way round.

    The explicit check is requested first and answers last; the drill-down
    validation requested after it carries the fresher reading. One monotonic
    claim has to order both directions, or fixing one would simply move the
    defect to the other.
    """

    def moved():
        drift_world["finish"](drift_world["task_id"], 3)
        _make_member_stale(drift_world, 1)

    _dom(
        drift_server, drift_world, "notice-order-reverse",
        on_steps={"READY-FOR-MUTATION": moved},
    )


def test_checking_for_run_changes_does_not_cancel_a_drill_down_in_flight(
    drift_server, drift_world
):
    """Ownership of the notice is shared; ownership of the NAVIGATION is not.

    A reader who opens a contributor and then asks whether the runs have
    changed has asked two questions, not replaced the first with the second.
    The drill-down still opens when its answer arrives.
    """
    _dom(drift_server, drift_world, "recheck-keeps-navigation")

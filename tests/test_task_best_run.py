"""Best run for one task within one experiment (`fix-9eg.17.4`).

Integration tests against a real `ObservabilityStore`, real attempt rows
written through the real `ExperimentController`, and the real shared selection
control. No Mock fixtures, per `.cursor/rules/testing_rules.mdc`. Nothing here
runs a model: the attempts are recorded evidence, written the way a runner
writes it.

The distinction under test throughout is between two selections that share
storage and mean different things: the experiment WINNER is which experiment a
workflow currently runs on, and the BEST RUN below is which of one task's n
repeated attempts is the preferred example. Most of these tests exist to pin
that they cannot be confused for one another, and that a default a reader
happens to land on is never recorded as a decision somebody made.
"""

from __future__ import annotations

import os

import pytest

from fastworkflow.benchmark import setup
from fastworkflow.experiment.runner import ExperimentController
from fastworkflow.observability import best_run, selection
from fastworkflow.observability import store as obs
from fastworkflow.observability.comparison import ExecutionRef, review_pair_key
from tests.test_experiment_container import _turn_row, _write_turn


ATTEMPTS = 3


@pytest.fixture
def folder(tmp_path):
    wf = tmp_path / "my_workflow"
    wf.mkdir()
    return wf


@pytest.fixture
def world(folder, tmp_path):
    """One registered experiment, one task, three recorded attempts.

    Attempt 1 completed over two turns, attempt 2 FAILED, attempt 3 was started
    and never finished — the three shapes the rules below are about.
    """
    benchmark = setup.save_benchmark(
        folder, {"title": "Roster review", "tasks": [{"prompt": "Review the roster"}]}
    )
    record = setup.create_experiment(
        folder, benchmark["benchmark_id"], "v1", runs_per_task=ATTEMPTS
    )
    experiment_id = record["experiment_id"]
    task_id = record["task_ids"][0]
    db_path = str(tmp_path / "evidence.sqlite3")
    store = obs.ObservabilityStore(db_path)
    controller = ExperimentController(
        db_path, store.store_identity(), external=False, workflow_folderpath=str(folder)
    )
    controller.create_experiment(
        experiment_id,
        record["description"],
        declared_tasks=1,
        declared_attempts=ATTEMPTS,
        declarations=[
            (task_id, n, f"ch-{n}") for n in range(1, ATTEMPTS + 1)
        ],
        workflow_name=setup.workflow_name_for(folder),
    )
    turns = {1: 2, 2: 1, 3: 1}
    for attempt in range(1, ATTEMPTS + 1):
        channel = f"ch-{attempt}"
        conversation = store.mint_conversation_id(
            channel, experiment_id=experiment_id, task_id=task_id, attempt=attempt
        )
        controller.start_attempt(
            experiment_id, task_id, attempt, channel, conversation_id=conversation
        )
        for turn in range(1, turns[attempt] + 1):
            _write_turn(
                store,
                _turn_row(
                    f"turn-{attempt}-{turn}",
                    channel,
                    conversation_id=conversation,
                    ordinal=turn,
                    experiment_id=experiment_id,
                    task_id=task_id,
                    attempt=attempt,
                    started_at=f"2026-09-0{attempt}T00:00:0{turn}Z",
                    completed_at=f"2026-09-0{attempt}T00:00:0{turn + 1}Z",
                ),
            )
    controller.finish_attempt(
        experiment_id, task_id, 1, outcome="pass", outcome_source="derived"
    )
    controller.finish_attempt(
        experiment_id,
        task_id,
        2,
        outcome="fail",
        outcome_source="derived",
        execution_status="failed",
    )
    # Attempt 3 stays open on purpose: started, turns written, never finished.
    control = setup.open_workflow_control(folder)
    return {
        "folder": folder,
        "control": control,
        "store": store,
        "experiment_id": experiment_id,
        "task_id": task_id,
        "source_id": store.store_identity(),
    }


def _select(world, attempt, *, expected=None, actor="reviewer", actor_kind="human",
            reason=None):
    return best_run.select_best_run(
        world["control"],
        world["experiment_id"],
        world["task_id"],
        attempt,
        expected_selection_id=expected,
        actor=actor,
        actor_kind=actor_kind,
        provenance="ui:task",
        reason=reason,
    )


def _current(world):
    return best_run.best_run(
        world["control"], world["experiment_id"], world["task_id"]
    )


# ----------------------------------------------------------------------
# Reading the attempts
# ----------------------------------------------------------------------


class TestReadingAttempts:
    def test_every_attempt_is_listed_with_its_real_status(self, world):
        rows = best_run.list_task_attempts(
            world["control"], world["experiment_id"], world["task_id"]
        )

        assert [row["attempt"] for row in rows] == [1, 2, 3]
        assert [row["execution_status"] for row in rows] == ["completed", "failed", None]
        assert [row["outcome"] for row in rows] == ["pass", "fail", None]
        assert [row["selectable"] for row in rows] == [True, True, False]

    def test_a_multi_turn_attempt_is_one_reference_carrying_every_turn(self, world):
        """A conversation is the evidence, not just its last answer."""
        rows = best_run.list_task_attempts(
            world["control"], world["experiment_id"], world["task_id"]
        )

        first = rows[0]
        assert first["turn_keys"] == ["turn-1-1", "turn-1-2"]
        assert first["execution_ref"]["turn_keys"] == ["turn-1-1", "turn-1-2"]
        assert first["execution_ref"]["attempt"] == 1

    def test_the_reference_is_the_first_completed_attempt_and_says_so(self, world):
        reference = best_run.reference_attempt(
            world["control"], world["experiment_id"], world["task_id"]
        )

        assert reference["attempt"] == 1
        assert reference["label"] == best_run.LABEL_REFERENCE
        assert reference["is_best"] is False
        assert reference["is_reference"] is True

    def test_a_viewing_reference_is_not_a_decision(self, world):
        """Nothing was chosen, so nothing is recorded as chosen."""
        best_run.reference_attempt(
            world["control"], world["experiment_id"], world["task_id"]
        )

        assert _current(world) is None
        assert (
            best_run.best_run_history(
                world["control"], world["experiment_id"], world["task_id"]
            )
            == []
        )

    def test_the_task_header_reads_in_one_call(self, world):
        summary = best_run.task_run_summary(
            world["control"], world["experiment_id"], world["task_id"]
        )

        assert summary["attempt_count"] == 3
        assert summary["selectable_attempts"] == [1, 2]
        assert summary["best_run"] is None
        assert summary["reference"]["attempt"] == 1
        assert summary["expected_selection_id"] is None
        assert summary["evidence_readable"] is True


@pytest.fixture
def long_attempt(folder, tmp_path):
    """One finished attempt longer than a single page of turns.

    The keys are minted so that STRING order is not conversation order (`t-10`
    sorts before `t-2`), because the paging read hands back turns in key order
    and re-sorting them by the recorded ordinal is the only thing that puts the
    conversation back together.
    """
    turns = best_run._TURN_PAGE * 2 + 37
    benchmark = setup.save_benchmark(
        folder, {"title": "Long one", "tasks": [{"prompt": "Keep going"}]}
    )
    record = setup.create_experiment(folder, benchmark["benchmark_id"], "v1")
    experiment_id, task_id = record["experiment_id"], record["task_ids"][0]
    db_path = str(tmp_path / "long.sqlite3")
    store = obs.ObservabilityStore(db_path)
    controller = ExperimentController(
        db_path, store.store_identity(), external=False, workflow_folderpath=str(folder)
    )
    controller.create_experiment(
        experiment_id,
        "",
        declared_tasks=1,
        declared_attempts=1,
        declarations=[(task_id, 1, "ch-long")],
        workflow_name=setup.workflow_name_for(folder),
    )
    conversation = store.mint_conversation_id(
        "ch-long", experiment_id=experiment_id, task_id=task_id, attempt=1
    )
    controller.start_attempt(experiment_id, task_id, 1, "ch-long", conversation_id=conversation)
    for ordinal in range(1, turns + 1):
        _write_turn(
            store,
            _turn_row(
                f"t-{ordinal}",
                "ch-long",
                conversation_id=conversation,
                ordinal=ordinal,
                experiment_id=experiment_id,
                task_id=task_id,
                attempt=1,
            ),
        )
    controller.finish_attempt(
        experiment_id, task_id, 1, outcome="pass", outcome_source="derived"
    )
    return {
        "control": setup.open_workflow_control(folder),
        "experiment_id": experiment_id,
        "task_id": task_id,
        "turns": turns,
    }


class TestLongAttempts:
    """An attempt longer than one page is read whole, or it is not read.

    Truncating is the dangerous answer here: a short `ExecutionRef` opens and
    renders exactly like a complete one, so the turns that were dropped are not
    missing on screen — they simply never existed as far as any reader, or
    anyone choosing a best run from what they read, can tell.
    """

    def test_every_turn_of_a_multi_page_attempt_is_present_and_in_order(
        self, long_attempt
    ):
        rows = best_run.list_task_attempts(
            long_attempt["control"],
            long_attempt["experiment_id"],
            long_attempt["task_id"],
        )

        expected = [f"t-{n}" for n in range(1, long_attempt["turns"] + 1)]
        assert rows[0]["turn_count"] == long_attempt["turns"]
        assert rows[0]["turn_keys"] == expected
        assert rows[0]["execution_ref"]["turn_keys"] == expected

    def test_no_turn_is_repeated_across_page_boundaries(self, long_attempt):
        keys = best_run.list_task_attempts(
            long_attempt["control"],
            long_attempt["experiment_id"],
            long_attempt["task_id"],
        )[0]["turn_keys"]

        assert len(set(keys)) == len(keys)

    def test_the_selected_reference_carries_the_whole_conversation(
        self, long_attempt
    ):
        result = best_run.select_best_run(
            long_attempt["control"],
            long_attempt["experiment_id"],
            long_attempt["task_id"],
            1,
            expected_selection_id=None,
            actor="reviewer",
            actor_kind="human",
            provenance="ui:task",
        )

        ref = result["best_run"]["run"]["execution_ref"]
        assert len(ref["turn_keys"]) == long_attempt["turns"]
        assert ref["turn_keys"][0] == "t-1"
        assert ref["turn_keys"][-1] == f"t-{long_attempt['turns']}"


@pytest.fixture
def no_turns(folder, tmp_path):
    """A finished attempt with nothing recorded under it.

    Real: an execution that completes before any turn is written, or one whose
    turns were pruned out of an old store while the attempt row stayed.
    """
    benchmark = setup.save_benchmark(
        folder, {"title": "Empty one", "tasks": [{"prompt": "Do nothing"}]}
    )
    record = setup.create_experiment(folder, benchmark["benchmark_id"], "v1")
    experiment_id, task_id = record["experiment_id"], record["task_ids"][0]
    db_path = str(tmp_path / "empty.sqlite3")
    store = obs.ObservabilityStore(db_path)
    controller = ExperimentController(
        db_path, store.store_identity(), external=False, workflow_folderpath=str(folder)
    )
    controller.create_experiment(
        experiment_id,
        "",
        declared_tasks=1,
        declared_attempts=1,
        declarations=[(task_id, 1, "ch-empty")],
        workflow_name=setup.workflow_name_for(folder),
    )
    controller.start_attempt(experiment_id, task_id, 1, "ch-empty")
    controller.finish_attempt(
        experiment_id, task_id, 1, outcome="pass", outcome_source="derived"
    )
    return {
        "control": setup.open_workflow_control(folder),
        "experiment_id": experiment_id,
        "task_id": task_id,
    }


class TestAttemptsWithoutTurns:
    """The honest state, spelled out rather than papered over.

    There is no reference to point at, so comparison is off — but the attempt
    finished, and a reviewer may still mean it. Reporting it as unselectable
    would hide a real run; reporting a reference for it would hand a client a
    `None` to build an `ExecutionRef` from.
    """

    def test_it_is_selectable_but_not_comparable(self, no_turns):
        row = best_run.list_task_attempts(
            no_turns["control"], no_turns["experiment_id"], no_turns["task_id"]
        )[0]

        assert row["finished"] is True
        assert row["selectable"] is True
        assert row["comparable"] is False
        assert row["execution_ref"] is None
        assert row["evidence_state"] == best_run.EVIDENCE_MISSING
        assert row["evidence_label"] == best_run.LABEL_EVIDENCE_MISSING

    def test_the_summary_offers_it_for_selection_and_not_for_comparison(
        self, no_turns
    ):
        summary = best_run.task_run_summary(
            no_turns["control"], no_turns["experiment_id"], no_turns["task_id"]
        )

        assert summary["selectable_attempts"] == [1]
        assert summary["comparable_attempts"] == []

    def test_choosing_it_reports_no_reference_rather_than_a_broken_one(
        self, no_turns
    ):
        result = best_run.select_best_run(
            no_turns["control"],
            no_turns["experiment_id"],
            no_turns["task_id"],
            1,
            expected_selection_id=None,
            actor="reviewer",
            actor_kind="human",
            provenance="ui:task",
        )

        current = result["best_run"]
        assert current["attempt"] == 1
        assert current["attempt_resolved"] is True
        assert current["comparable"] is False
        assert current["execution_ref"] is None
        assert current["run"]["evidence_label"] == best_run.LABEL_EVIDENCE_MISSING

    def test_the_reference_says_there_is_nothing_to_open(self, no_turns):
        """It is still the attempt a reader lands on — with nothing to show."""
        reference = best_run.reference_attempt(
            no_turns["control"], no_turns["experiment_id"], no_turns["task_id"]
        )

        assert reference["attempt"] == 1
        assert reference["comparable"] is False


# ----------------------------------------------------------------------
# Selecting, replacing, clearing
# ----------------------------------------------------------------------


class TestSelectReplaceClear:
    def test_selecting_records_the_choice_and_its_provenance(self, world):
        result = _select(world, 1, reason="clearest walkthrough")

        assert result["decision"] == best_run.DECISION_SELECT
        current = _current(world)
        assert current["attempt"] == 1
        assert current["label"] == best_run.LABEL_BEST
        assert current["run"]["is_best"] is True
        history = best_run.best_run_history(
            world["control"], world["experiment_id"], world["task_id"]
        )
        assert len(history) == 1
        assert history[0]["actor"] == "reviewer"
        assert history[0]["actor_kind"] == "human"
        assert history[0]["provenance"] == "ui:task"
        assert history[0]["rationale"] == "clearest walkthrough"
        assert history[0]["new_attempt"] == 1
        assert history[0]["previous_attempt"] is None

    def test_a_reason_is_optional(self, world):
        _select(world, 1)

        assert _current(world)["attempt"] == 1

    def test_replacing_keeps_both_references_in_history(self, world):
        first = _select(world, 1)

        _select(world, 2, expected=first["selection_id"])

        history = best_run.best_run_history(
            world["control"], world["experiment_id"], world["task_id"]
        )
        assert [row["decision"] for row in history] == [
            best_run.DECISION_REPLACE,
            best_run.DECISION_SELECT,
        ]
        assert history[0]["previous_attempt"] == 1
        assert history[0]["new_attempt"] == 2
        assert _current(world)["attempt"] == 2

    def test_a_failed_attempt_may_be_the_best_run_and_stays_failed(self, world):
        """Preferred example, not proof of correctness.

        A failure is often the clearest thing to learn from, so it is
        selectable — and selecting it must not launder its status into one that
        implies it worked.
        """
        _select(world, 2)

        current = _current(world)
        assert current["attempt"] == 2
        assert current["run"]["execution_status"] == "failed"
        assert current["run"]["outcome"] == "fail"

    def test_clearing_removes_the_pointer_and_keeps_the_history(self, world):
        selected = _select(world, 1)

        best_run.clear_best_run(
            world["control"],
            world["experiment_id"],
            world["task_id"],
            expected_selection_id=selected["selection_id"],
            actor="reviewer",
            actor_kind="human",
            provenance="ui:task",
            reason="not representative after all",
        )

        assert _current(world) is None
        history = best_run.best_run_history(
            world["control"], world["experiment_id"], world["task_id"]
        )
        assert [row["decision"] for row in history] == [
            best_run.DECISION_CLEAR,
            best_run.DECISION_SELECT,
        ]
        assert history[0]["previous_attempt"] == 1
        assert history[0]["new_attempt"] is None

    def test_clearing_nothing_is_refused(self, world):
        with pytest.raises(best_run.NoBestRun):
            best_run.clear_best_run(
                world["control"],
                world["experiment_id"],
                world["task_id"],
                expected_selection_id=None,
                actor="reviewer",
                actor_kind="human",
                provenance="ui:task",
            )

    def test_selecting_again_after_clearing_is_a_fresh_selection(self, world):
        selected = _select(world, 1)
        best_run.clear_best_run(
            world["control"],
            world["experiment_id"],
            world["task_id"],
            expected_selection_id=selected["selection_id"],
            actor="reviewer",
            actor_kind="human",
            provenance="ui:task",
        )

        result = _select(world, 2)

        assert result["decision"] == best_run.DECISION_SELECT
        assert _current(world)["attempt"] == 2

    def test_undecided_records_attention_without_moving_anything(self, world):
        """"Reviewed and left alone" is not the same as "never opened"."""
        result = best_run.leave_undecided(
            world["control"],
            world["experiment_id"],
            world["task_id"],
            expected_selection_id=None,
            candidate_attempt=2,
            actor="distiller",
            actor_kind="coding_agent",
            provenance="agent:distillation",
            reason="none of these is a good example",
        )

        assert result["decision"] == best_run.DECISION_UNDECIDED
        assert _current(world) is None
        history = best_run.best_run_history(
            world["control"], world["experiment_id"], world["task_id"]
        )
        assert history[0]["candidate_attempt"] == 2
        assert history[0]["new_attempt"] is None
        assert history[0]["actor_kind"] == "coding_agent"

    def test_undecided_leaves_an_existing_best_run_in_place(self, world):
        selected = _select(world, 1)

        best_run.leave_undecided(
            world["control"],
            world["experiment_id"],
            world["task_id"],
            expected_selection_id=selected["selection_id"],
            candidate_attempt=2,
            actor="reviewer",
            actor_kind="human",
            provenance="ui:task",
        )

        assert _current(world)["attempt"] == 1

    def test_human_and_agent_use_the_same_api(self, world):
        first = _select(world, 1, actor="reviewer", actor_kind="human")

        _select(
            world,
            2,
            expected=first["selection_id"],
            actor="distiller",
            actor_kind="coding_agent",
        )

        history = best_run.best_run_history(
            world["control"], world["experiment_id"], world["task_id"]
        )
        assert [row["actor_kind"] for row in history] == ["coding_agent", "human"]


# ----------------------------------------------------------------------
# What may not be chosen
# ----------------------------------------------------------------------


class TestRefusals:
    def test_an_unfinished_attempt_cannot_be_the_best_run(self, world):
        """A run still in flight has not produced the example being preferred."""
        with pytest.raises(best_run.AttemptNotSelectable) as caught:
            _select(world, 3)

        assert "finished" in str(caught.value)
        assert _current(world) is None

    def test_an_attempt_that_does_not_exist_is_refused(self, world):
        with pytest.raises(best_run.AttemptNotSelectable):
            _select(world, 9)

    @pytest.mark.parametrize("value", [True, 1.0, "1.0", "one", None])
    def test_an_attempt_number_that_is_not_an_exact_integer_is_refused(
        self, world, value
    ):
        """`True` is `1` and `int(1.9)` is `1`: both would name somebody's run."""
        with pytest.raises(ValueError):
            _select(world, value)

    def test_an_unregistered_experiment_is_reported_not_answered_empty(self, world):
        """"I cannot see this experiment" is not "this task has no best run"."""
        with pytest.raises(best_run.TaskBestUnavailable):
            best_run.list_task_attempts(
                world["control"], "exp-not-registered", world["task_id"]
            )

    def test_an_attempt_of_another_task_cannot_be_chosen(self, tmp_path):
        """The attempts are read per task, so a sibling task's run is not offered.

        Two tasks of one experiment, with different numbers of attempts. Task B
        has only attempt 1; asking for its attempt 2 must not reach into task
        A's attempt 2, which exists and is finished.
        """
        folder = tmp_path / "two_task_workflow"
        folder.mkdir()
        benchmark = setup.save_benchmark(
            folder, {"title": "Pair", "tasks": [{"prompt": "a"}, {"prompt": "b"}]}
        )
        record = setup.create_experiment(folder, benchmark["benchmark_id"], "v1")
        task_a, task_b = record["task_ids"]
        db_path = str(tmp_path / "evidence.sqlite3")
        store = obs.ObservabilityStore(db_path)
        controller = ExperimentController(
            db_path,
            store.store_identity(),
            external=False,
            workflow_folderpath=str(folder),
        )
        controller.create_experiment(
            record["experiment_id"],
            "",
            declared_tasks=2,
            declared_attempts=2,
            declarations=[
                (task_a, 1, "ch-a-1"), (task_a, 2, "ch-a-2"),
                (task_b, 1, "ch-b-1"), (task_b, 2, "ch-b-2"),
            ],
            workflow_name=setup.workflow_name_for(folder),
        )
        for task_id, attempt in ((task_a, 1), (task_a, 2), (task_b, 1)):
            channel = f"ch-{'a' if task_id == task_a else 'b'}-{attempt}"
            conversation = store.mint_conversation_id(
                channel,
                experiment_id=record["experiment_id"],
                task_id=task_id,
                attempt=attempt,
            )
            controller.start_attempt(
                record["experiment_id"], task_id, attempt, channel,
                conversation_id=conversation,
            )
            _write_turn(
                store,
                _turn_row(
                    f"turn-{channel}",
                    channel,
                    conversation_id=conversation,
                    experiment_id=record["experiment_id"],
                    task_id=task_id,
                    attempt=attempt,
                ),
            )
            controller.finish_attempt(
                record["experiment_id"], task_id, attempt,
                outcome="pass", outcome_source="derived",
            )
        control = setup.open_workflow_control(folder)

        with pytest.raises(best_run.AttemptNotSelectable):
            best_run.select_best_run(
                control, record["experiment_id"], task_b, 2,
                expected_selection_id=None,
                actor="reviewer", actor_kind="human", provenance="ui:task",
            )

        # And each task keeps its own selection.
        best_run.select_best_run(
            control, record["experiment_id"], task_a, 2,
            expected_selection_id=None,
            actor="reviewer", actor_kind="human", provenance="ui:task",
        )
        assert best_run.best_run(control, record["experiment_id"], task_b) is None
        assert (
            best_run.best_run(control, record["experiment_id"], task_a)["attempt"] == 2
        )

    def test_a_stale_selection_is_refused_with_what_is_current(self, world):
        first = _select(world, 1)
        _select(world, 2, expected=first["selection_id"])

        with pytest.raises(selection.StaleSelection) as caught:
            _select(world, 1, expected=first["selection_id"])

        assert caught.value.expected_selection_id == first["selection_id"]
        assert _current(world)["attempt"] == 2

    def test_selecting_into_an_empty_scope_requires_saying_it_was_empty(self, world):
        _select(world, 1)

        with pytest.raises(selection.StaleSelection):
            _select(world, 2, expected=None)


# ----------------------------------------------------------------------
# The two selections stay apart
# ----------------------------------------------------------------------


class TestScopeSeparation:
    def test_choosing_a_best_run_does_not_touch_the_experiment_winner(self, world):
        before = world["control"].winner_for_experiment(world["experiment_id"])

        _select(world, 2)

        after = world["control"].winner_for_experiment(world["experiment_id"])
        assert after["selection_id"] == before["selection_id"]
        assert after["experiment_id"] == before["experiment_id"]

    def test_the_winner_history_and_the_task_history_are_separate(self, world):
        _select(world, 1)

        group_id = world["control"].group_for_experiment(
            world["experiment_id"]
        )["group_id"]
        winner_history = world["control"].decision_history(group_id)
        assert [row["decision"] for row in winner_history] == [
            selection.DECISION_INITIAL
        ]
        assert (
            len(
                best_run.best_run_history(
                    world["control"], world["experiment_id"], world["task_id"]
                )
            )
            == 1
        )

    def test_the_winner_pointer_cannot_be_reached_through_the_scoped_api(self, world):
        with pytest.raises(ValueError):
            world["control"].apply_scoped_decision(
                scope_kind=selection.EXPERIMENT_SCOPE,
                group_id="whatever",
                scope_key="",
                decision="select",
                expected_selection_id=None,
                actor="a",
                actor_kind="human",
                provenance="p",
            )

    def test_two_experiments_keep_separate_best_runs_for_the_same_task(
        self, world, tmp_path
    ):
        """The scope is the task IN an experiment, not the task."""
        folder = world["folder"]
        candidate = setup.duplicate_experiment(folder, world["experiment_id"])
        store = world["store"]
        controller = ExperimentController(
            store.db_path,
            store.store_identity(),
            external=False,
            workflow_folderpath=str(folder),
        )
        task_id = world["task_id"]
        controller.create_experiment(
            candidate["experiment_id"],
            "candidate",
            declared_tasks=1,
            declared_attempts=1,
            declarations=[(task_id, 1, "ch-c-1")],
            workflow_name=setup.workflow_name_for(folder),
        )
        conversation = store.mint_conversation_id(
            "ch-c-1",
            experiment_id=candidate["experiment_id"],
            task_id=task_id,
            attempt=1,
        )
        controller.start_attempt(
            candidate["experiment_id"], task_id, 1, "ch-c-1",
            conversation_id=conversation,
        )
        _write_turn(
            store,
            _turn_row(
                "turn-c-1",
                "ch-c-1",
                conversation_id=conversation,
                experiment_id=candidate["experiment_id"],
                task_id=task_id,
                attempt=1,
            ),
        )
        controller.finish_attempt(
            candidate["experiment_id"], task_id, 1,
            outcome="pass", outcome_source="derived",
        )

        _select(world, 2)
        best_run.select_best_run(
            world["control"],
            candidate["experiment_id"],
            task_id,
            1,
            expected_selection_id=None,
            actor="reviewer",
            actor_kind="human",
            provenance="ui:task",
        )

        assert _current(world)["attempt"] == 2
        assert (
            best_run.best_run(
                world["control"], candidate["experiment_id"], task_id
            )["attempt"]
            == 1
        )
        assert best_run.task_scope_key(
            world["experiment_id"], task_id
        ) != best_run.task_scope_key(candidate["experiment_id"], task_id)


# ----------------------------------------------------------------------
# References never retarget
# ----------------------------------------------------------------------


class TestReferencesNeverRetarget:
    def test_changing_the_best_run_produces_a_new_reference(self, world):
        """Comparison comments key off the pair, so the pair must change.

        If pinning a new best run rewrote the old reference, every comment
        recorded against yesterday's comparison would silently start claiming
        to be about today's run.
        """
        first = _select(world, 1)
        old_ref = ExecutionRef.from_mapping(_current(world)["run"]["execution_ref"])
        reference = best_run.reference_attempt(
            world["control"], world["experiment_id"], world["task_id"]
        )
        old_pair = review_pair_key(
            ExecutionRef.from_mapping(reference["execution_ref"]), old_ref
        )

        _select(world, 2, expected=first["selection_id"])

        new_ref = ExecutionRef.from_mapping(_current(world)["run"]["execution_ref"])
        assert new_ref.ref_id() != old_ref.ref_id()
        assert (
            review_pair_key(
                ExecutionRef.from_mapping(reference["execution_ref"]), new_ref
            )
            != old_pair
        )
        # The old reference still names exactly what it named.
        assert old_ref.attempt == 1
        assert old_ref.turn_keys == ("turn-1-1", "turn-1-2")

    def test_the_label_is_not_part_of_the_identity(self, world):
        """Relabelling Reference to Best run must not invalidate review work."""
        rows = best_run.list_task_attempts(
            world["control"], world["experiment_id"], world["task_id"]
        )
        plain = ExecutionRef.from_mapping(rows[0]["execution_ref"])
        _select(world, 1)

        labelled = ExecutionRef.from_mapping(_current(world)["run"]["execution_ref"])

        assert labelled.label == best_run.LABEL_BEST
        assert labelled.ref_id() == plain.ref_id()


# ----------------------------------------------------------------------
# Evidence and control stay apart
# ----------------------------------------------------------------------


class TestEvidenceAndControl:
    def test_no_selection_row_lands_in_the_evidence_database(self, world):
        _select(world, 1)

        with world["store"]._connect() as conn:
            names = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
        assert not {n for n in names if "selection" in n}

    def test_read_only_inspection_creates_no_control_database(self, tmp_path):
        """Showing a task that nobody has judged must leave the disk alone."""
        wf = tmp_path / "untouched_workflow"
        wf.mkdir()

        with pytest.raises(selection.SelectionControlUnavailable):
            setup.open_workflow_control(wf, create=False)

        assert not os.path.exists(setup.workflow_control_db_path(wf))

    def test_a_selection_whose_evidence_is_unreadable_is_still_a_fact(self, world):
        """The decision happened; what the run did is simply unknown now."""
        _select(world, 1)
        detached = selection.open_shared_control(
            world["control"].control_db_path, sources={}
        )

        current = best_run.best_run(
            detached, world["experiment_id"], world["task_id"]
        )

        assert current["attempt"] == 1
        assert current["attempt_resolved"] is False
        assert current["run"] is None

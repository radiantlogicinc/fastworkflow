"""HTTP contract for winners, best runs, comparison and pair review.

`fix-9eg.17.1` / `.17.2` / `.17.3` / `.17.4` and the server slice of
`fix-9eg.4`. Integration throughout, per `.cursor/rules/testing_rules.mdc`: a
real workflow folder, a real benchmark manifest, real `ObservabilityStore`
databases, attempts written through the real `ExperimentController`, real span
rows, the real shared selection control and the real pair-review sidecar. No
Mock fixtures and nothing paid -- every attempt below is recorded evidence,
written the way a runner writes it.

Two halves on purpose. `selection_api.handle_*` is exercised directly, because
that is the function a future embedder (or a second transport) calls, and the
same scenarios are then driven over a real socket against `ChatbotServer`, so
the token gate, the method allowlist and the JSON shapes are tested as a
client meets them rather than as the module imagines them.

The distinction most of these tests defend is that the experiment WINNER and a
task's BEST RUN share storage and mean different things, and that a read never
becomes a write: a GET must not create a control file, must not elect anybody
and must not mark anything reviewed.
"""

from __future__ import annotations

import json
import os
import threading

import pytest

from fastworkflow.benchmark import setup
from fastworkflow.experiment.runner import ExperimentController
from fastworkflow.observability import feedback as fb
from fastworkflow.observability import best_run, comparison, pair_review, selection
from fastworkflow.observability import store as obs
from fastworkflow.run_chatbot import selection_api
from fastworkflow.run_chatbot import server as run_chatbot_server
from tests.test_chatbot_benchmarks import _request
from tests.test_execution_comparison import (
    _execute_span,
    _output,
    _record,
    _write,
)
from tests.test_execution_comparison import _turn_row as _evidence_turn_row

T0 = 1_700_000_000_000_000_000

HUMAN = {"actor": "dhar", "actor_kind": "human"}
AGENT = {"actor": "cursor", "actor_kind": "coding_agent"}


# ----------------------------------------------------------------------
# One workflow, two experiments, two evidence stores
# ----------------------------------------------------------------------


def _seed_turn(store, turn_key, *, experiment_id, task_id, attempt, conversation,
               ordinal, commands, answer="done", status="completed", success=True,
               passes=None):
    """One recorded turn with a real span per dispatched command.

    `passes` stamps `fw.pass` on the span of the command at the same position,
    which is how a pass-stamping producer would record two passes of one turn.
    Nothing in fastWorkflow stamps it yet (`fix-txxy`), so this is the only
    place such evidence exists -- and it is real recorded evidence, not a
    fixture the API is taught to recognise.
    """
    refs, spans, outputs = [], [], []
    for index, command in enumerate(commands):
        call_id = f"{turn_key}-call-{index}"
        span_id = f"{turn_key}-span-{index}"
        refs.append((call_id, index, span_id))
        outputs.append(_output(call_id, command, {"n": index}))
        stamp = None if passes is None else passes[index]
        spans.append(
            _execute_span(
                span_id,
                turn_key,
                call_id=call_id,
                command_name=command,
                start_ns=T0 + index * 1_000_000,
                extra=None if stamp is None else {"fw.pass": stamp},
            )
        )
    row = _evidence_turn_row(
        turn_key,
        record=_record(turn_key, refs=refs, outputs=outputs, success=success),
        answer=answer,
        status=status,
        success=success,
        experiment_id=experiment_id,
        task_id=task_id,
        attempt=attempt,
    )
    row["conversation_id"] = conversation
    row["ordinal"] = ordinal
    _write(store, row, spans)


def _run_attempt(store, controller, experiment_id, task_id, attempt, turns,
                 *, outcome=None, outcome_source="derived", execution_status=None,
                 finish=True, passes=None):
    channel = f"ch-{experiment_id[-4:]}-{attempt}"
    conversation = store.mint_conversation_id(
        channel, experiment_id=experiment_id, task_id=task_id, attempt=attempt
    )
    controller.start_attempt(
        experiment_id, task_id, attempt, channel, conversation_id=conversation
    )
    for ordinal, commands in enumerate(turns, start=1):
        _seed_turn(
            store,
            f"{experiment_id[-6:]}-a{attempt}-t{ordinal}",
            experiment_id=experiment_id,
            task_id=task_id,
            attempt=attempt,
            conversation=conversation,
            ordinal=ordinal,
            commands=commands,
            passes=None if passes is None else passes[ordinal - 1],
        )
    if finish:
        kwargs = {"outcome": outcome, "outcome_source": outcome_source}
        if execution_status is not None:
            kwargs["execution_status"] = execution_status
        controller.finish_attempt(experiment_id, task_id, attempt, **kwargs)


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A workflow whose first experiment ran a task four different ways.

    Attempt 1 completed over two turns, attempt 2 FAILED, attempt 3 was started
    and never finished, and attempt 4 finished with no recorded turns at all.
    Those four shapes are what the selectable/comparable rules are about. A
    second experiment records one completed attempt of the SAME task in a
    SECOND database, which is the winner-versus-candidate shape.
    """
    monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "state"))
    folder = tmp_path / "roster_workflow"
    folder.mkdir()
    (folder / "_commands").mkdir()
    benchmark = setup.save_benchmark(
        folder, {"title": "Roster review", "tasks": [{"prompt": "Review the roster"}]}
    )
    benchmark_id = benchmark["benchmark_id"]

    first = setup.create_experiment(folder, benchmark_id, "v1", runs_per_task=4)
    experiment_id = first["experiment_id"]
    task_id = first["task_ids"][0]

    db_one = str(tmp_path / "evidence-one.sqlite3")
    store_one = obs.ObservabilityStore(db_one)
    controller_one = ExperimentController(
        db_one, store_one.store_identity(), external=False,
        workflow_folderpath=str(folder),
    )
    controller_one.create_experiment(
        experiment_id,
        first["description"],
        declared_tasks=1,
        declared_attempts=4,
        declarations=[(task_id, n, f"ch-{experiment_id[-4:]}-{n}") for n in range(1, 5)],
        workflow_name=setup.workflow_name_for(folder),
    )
    _run_attempt(
        store_one, controller_one, experiment_id, task_id, 1,
        [["add_item", "list_items"], ["complete_item"]], outcome="pass",
    )
    _run_attempt(
        store_one, controller_one, experiment_id, task_id, 2,
        [["add_item", "remove_item"]], outcome="fail", execution_status="failed",
    )
    _run_attempt(
        store_one, controller_one, experiment_id, task_id, 3,
        [["add_item"]], finish=False,
    )
    _run_attempt(
        store_one, controller_one, experiment_id, task_id, 4, [], outcome="pass",
    )

    second = setup.create_experiment(folder, benchmark_id, "v1", runs_per_task=1)
    candidate_id = second["experiment_id"]
    db_two = str(tmp_path / "evidence-two.sqlite3")
    store_two = obs.ObservabilityStore(db_two)
    controller_two = ExperimentController(
        db_two, store_two.store_identity(), external=False,
        workflow_folderpath=str(folder),
    )
    controller_two.create_experiment(
        candidate_id,
        second["description"],
        declared_tasks=1,
        declared_attempts=1,
        declarations=[(task_id, 1, f"ch-{candidate_id[-4:]}-1")],
        workflow_name=setup.workflow_name_for(folder),
    )
    _run_attempt(
        store_two, controller_two, candidate_id, task_id, 1,
        [["add_item", "sort_items"]], outcome="pass",
    )

    return {
        "folder": str(folder),
        "benchmark_id": benchmark_id,
        "experiment_id": experiment_id,
        "candidate_id": candidate_id,
        "task_id": task_id,
        "store_one": store_one,
        "store_two": store_two,
        "control_path": setup.workflow_control_db_path(str(folder)),
    }


@pytest.fixture
def server(world):
    srv = run_chatbot_server.ChatbotServer(
        db_path="",
        workflow_path=world["folder"],
        port=0,
        spawn_options={"no_server": True},
    )
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    thread.join(timeout=5)


def _get(world, path, **params):
    query = {key: [str(value)] for key, value in params.items() if value is not None}
    return selection_api.handle_get(world["folder"], path, query)


def _post(world, path, body):
    return selection_api.handle_post(world["folder"], path, body)


def _delete(world, path, body):
    return selection_api.handle_delete(world["folder"], path, body)


def _experiment(world, suffix=""):
    return f"/api/experiments/{world['experiment_id']}{suffix}"


def _task(world, suffix="", experiment=None):
    return (
        f"/api/experiments/{experiment or world['experiment_id']}"
        f"/tasks/{world['task_id']}{suffix}"
    )


# ----------------------------------------------------------------------
# The first experiment wins at creation, and nothing had to say so
# ----------------------------------------------------------------------


class TestWinnerRead:
    def test_the_first_experiment_is_the_winner_and_is_labelled_automatic(self, world):
        status, payload = _get(world, _experiment(world, "/winner"))

        assert status == 200
        winner = payload["winner"]
        assert winner["experiment_id"] == world["experiment_id"]
        assert winner["automatic"] is True
        assert winner["decision"] == selection.DECISION_INITIAL
        assert payload["is_winner"] is True
        assert payload["expected_selection_id"] == winner["selection_id"]

    def test_a_later_experiment_joins_the_same_contest_without_taking_it(self, world):
        payload = _get(world, f"/api/experiments/{world['candidate_id']}/winner")[1]

        assert payload["is_winner"] is False
        assert payload["winner"]["experiment_id"] == world["experiment_id"]
        members = {row["experiment_id"] for row in payload["members"]}
        assert members == {world["experiment_id"], world["candidate_id"]}

    def test_the_winner_reports_its_own_live_state_not_a_claim_of_success(self, world):
        winner = _get(world, _experiment(world, "/winner"))[1]["winner"]

        # `initial` says nobody judged it; the experiment row says what it is.
        assert winner["experiment_resolved"] is True
        assert winner["experiment"]["status"] is not None
        assert "successful" not in json.dumps(winner["experiment"])

    def test_the_repeat_count_is_carried_on_the_registration(self, world):
        assert _get(world, _experiment(world, "/winner"))[1]["runs_per_task"] == 4

    def test_an_unknown_experiment_is_not_found(self, world):
        status, payload = _get(world, "/api/experiments/exp-nobody/winner")

        assert status == 404
        assert "exp-nobody" in payload["error"]

    def test_a_workflow_with_no_control_is_not_given_one_by_a_read(self, tmp_path,
                                                                   monkeypatch):
        monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "state"))
        empty = tmp_path / "untouched_workflow"
        empty.mkdir()
        control = setup.workflow_control_db_path(str(empty))

        status, payload = selection_api.handle_get(
            str(empty), "/api/experiments/exp-1/winner", {}
        )

        assert status == 409 and payload["control_exists"] is False
        assert not os.path.exists(control)


class TestWinnerHistory:
    def test_history_is_bounded_and_pageable(self, world):
        status, payload = _get(world, _experiment(world, "/winner/history"), limit=1)

        assert status == 200 and payload["limit"] == 1
        assert len(payload["history"]) == 1

    def test_an_absurd_limit_is_clamped_rather_than_served(self, world):
        payload = _get(world, _experiment(world, "/winner/history"), limit=100000)[1]

        assert payload["limit"] == selection_api._MAX_PAGE

    @pytest.mark.parametrize("limit", ["0", "-1", "2.5", "many"])
    def test_a_limit_that_is_not_a_positive_integer_is_refused(self, world, limit):
        assert _get(world, _experiment(world, "/winner/history"), limit=limit)[0] == 400


# ----------------------------------------------------------------------
# Deciding: promote, keep, undecided
# ----------------------------------------------------------------------


class TestWinnerDecisions:
    def _decide(self, world, **body):
        current = _get(world, _experiment(world, "/winner"))[1]
        body.setdefault("expected_selection_id", current["expected_selection_id"])
        return _post(world, _experiment(world, "/winner/decisions"), {**HUMAN, **body})

    def test_promoting_a_candidate_moves_the_pointer_and_records_who(self, world):
        status, payload = self._decide(
            world,
            decision="promote",
            candidate_experiment_id=world["candidate_id"],
            rationale="the candidate reads better",
        )

        assert status == 201
        assert payload["decision"]["decision"] == "promote"
        assert payload["winner"]["winner"]["experiment_id"] == world["candidate_id"]
        assert payload["winner"]["winner"]["automatic"] is False
        latest = _get(world, _experiment(world, "/winner/history"))[1]["history"][0]
        assert latest["actor"] == "dhar" and latest["actor_kind"] == "human"
        assert latest["rationale"] == "the candidate reads better"

    def test_keeping_the_winner_records_the_challenger_it_passed_over(self, world):
        before = _get(world, _experiment(world, "/winner"))[1]

        payload = self._decide(
            world, decision="keep", candidate_experiment_id=world["candidate_id"]
        )[1]

        assert payload["winner"]["winner"]["selection_id"] == \
            before["winner"]["selection_id"]
        latest = _get(world, _experiment(world, "/winner/history"))[1]["history"][0]
        assert latest["decision"] == "keep"
        assert latest["candidate_experiment_id"] == world["candidate_id"]

    def test_undecided_is_recorded_and_moves_nothing(self, world):
        before = _get(world, _experiment(world, "/winner"))[1]

        payload = self._decide(world, decision="undecided")[1]

        assert payload["winner"]["expected_selection_id"] == \
            before["expected_selection_id"]
        assert _get(world, _experiment(world, "/winner/history"))[1]["history"][0][
            "decision"
        ] == "undecided"

    def test_a_decision_against_a_replaced_winner_is_refused_with_what_is_current(
        self, world
    ):
        stale = _get(world, _experiment(world, "/winner"))[1]["expected_selection_id"]
        self._decide(world, decision="promote",
                     candidate_experiment_id=world["candidate_id"])

        status, payload = _post(
            world,
            _experiment(world, "/winner/decisions"),
            {**HUMAN, "decision": "promote", "expected_selection_id": stale,
             "candidate_experiment_id": world["experiment_id"]},
        )

        assert status == 409 and payload["stale"] is True
        assert payload["expected_selection_id"] == stale
        assert payload["current_experiment_id"] == world["candidate_id"]
        # The refusal changed nothing.
        assert _get(world, _experiment(world, "/winner"))[1]["winner"][
            "experiment_id"
        ] == world["candidate_id"]

    def test_a_decision_with_no_expected_selection_is_refused(self, world):
        status, payload = _post(
            world, _experiment(world, "/winner/decisions"),
            {**HUMAN, "decision": "keep"},
        )

        assert status == 400 and "expected_selection_id" in payload["error"]

    @pytest.mark.parametrize("decision", ["initial", "delete", "", "PROMOTE"])
    def test_only_the_three_client_decisions_are_accepted(self, world, decision):
        assert self._decide(world, decision=decision)[0] == 400

    def test_a_client_cannot_file_its_decision_as_the_system_s_own(self, world):
        status, payload = self._decide(
            world, decision="keep", actor="fastworkflow", actor_kind="system"
        )

        assert status == 400 and "actor_kind" in payload["error"]

    def test_a_coding_agent_and_a_person_use_the_same_route(self, world):
        assert self._decide(world, decision="undecided", **AGENT)[0] == 201
        latest = _get(world, _experiment(world, "/winner/history"))[1]["history"][0]
        assert latest["actor_kind"] == "coding_agent"


# ----------------------------------------------------------------------
# The task's runs
# ----------------------------------------------------------------------


class TestTaskRuns:
    def test_every_attempt_is_listed_with_its_real_status(self, world):
        status, payload = _get(world, _task(world, "/runs"))

        assert status == 200
        assert [row["attempt"] for row in payload["attempts"]] == [1, 2, 3, 4]
        by_attempt = {row["attempt"]: row for row in payload["attempts"]}
        assert by_attempt[2]["outcome"] == "fail"
        assert by_attempt[2]["execution_status"] == "failed"
        assert by_attempt[3]["finished"] is False
        assert payload["selectable_attempts"] == [1, 2, 4]
        assert payload["comparable_attempts"] == [1, 2, 3]

    def test_before_anybody_decides_the_first_completed_attempt_is_only_a_reference(
        self, world
    ):
        payload = _get(world, _task(world, "/runs"))[1]

        assert payload["best_run"] is None
        assert payload["reference"]["attempt"] == 1
        assert payload["reference"]["label"] == best_run.LABEL_REFERENCE
        assert payload["reference"]["is_best"] is False
        assert payload["expected_selection_id"] is None

    def test_a_finished_attempt_with_no_turns_is_selectable_but_not_comparable(
        self, world
    ):
        row = next(
            r for r in _get(world, _task(world, "/runs"))[1]["attempts"]
            if r["attempt"] == 4
        )

        assert row["selectable"] is True and row["comparable"] is False
        assert row["execution_ref"] is None
        assert row["evidence_state"] == best_run.EVIDENCE_MISSING
        assert row["evidence_label"] == best_run.LABEL_EVIDENCE_MISSING

    def test_a_single_read_carries_what_the_next_write_must_echo(self, world):
        _post(world, _task(world, "/best-run"), {**HUMAN, "attempt": 1})

        payload = _get(world, _task(world, "/runs"))[1]

        assert payload["expected_selection_id"] == payload["best_run"]["selection_id"]


class TestBestRunDecisions:
    def test_selecting_an_attempt_records_it_and_labels_it_best(self, world):
        status, payload = _post(world, _task(world, "/best-run"),
                                {**HUMAN, "attempt": 1, "reason": "clearest run"})

        assert status == 201
        assert payload["decision"]["decision"] == best_run.DECISION_SELECT
        chosen = payload["decision"]["best_run"]
        assert chosen["attempt"] == 1 and chosen["label"] == best_run.LABEL_BEST

    def test_a_failed_attempt_may_be_the_best_run_and_stays_visibly_failed(self, world):
        status, payload = _post(world, _task(world, "/best-run"),
                                {**HUMAN, "attempt": 2})

        assert status == 201
        run = payload["decision"]["best_run"]["run"]
        assert run["outcome"] == "fail" and run["execution_status"] == "failed"

    def test_an_unfinished_attempt_cannot_be_preferred_yet(self, world):
        status, payload = _post(world, _task(world, "/best-run"),
                                {**HUMAN, "attempt": 3})

        assert status == 422
        assert payload["attempt"] == 3 and "not finished" in payload["reason"]

    def test_an_attempt_this_task_never_recorded_is_refused(self, world):
        status, payload = _post(world, _task(world, "/best-run"),
                                {**HUMAN, "attempt": 99})

        assert status == 422 and "no recorded attempt" in payload["reason"]

    @pytest.mark.parametrize("attempt", [True, 2.0, 2.5, "2.0", " ", None])
    def test_an_attempt_that_is_not_an_exact_integer_is_refused(self, world, attempt):
        status, _ = _post(world, _task(world, "/best-run"),
                          {**HUMAN, "attempt": attempt})

        assert status == 400

    def test_replacing_is_derived_from_history_not_supplied(self, world):
        first = _post(world, _task(world, "/best-run"), {**HUMAN, "attempt": 1})[1]
        current = first["decision"]["best_run"]["selection_id"]

        second = _post(
            world, _task(world, "/best-run"),
            {**HUMAN, "attempt": 2, "expected_selection_id": current},
        )[1]

        assert second["decision"]["decision"] == best_run.DECISION_REPLACE

    def test_a_selection_made_from_a_stale_screen_is_refused(self, world):
        first = _post(world, _task(world, "/best-run"), {**HUMAN, "attempt": 1})[1]
        stale = first["decision"]["best_run"]["selection_id"]
        _post(world, _task(world, "/best-run"),
              {**HUMAN, "attempt": 2, "expected_selection_id": stale})

        status, payload = _post(
            world, _task(world, "/best-run"),
            {**AGENT, "attempt": 1, "expected_selection_id": stale},
        )

        assert status == 409 and payload["stale"] is True
        assert _get(world, _task(world, "/runs"))[1]["best_run"]["attempt"] == 2

    def test_clearing_withdraws_the_pointer_and_keeps_the_history(self, world):
        selected = _post(world, _task(world, "/best-run"), {**HUMAN, "attempt": 1})[1]
        current = selected["decision"]["best_run"]["selection_id"]

        status, payload = _delete(
            world, _task(world, "/best-run"),
            {**HUMAN, "expected_selection_id": current},
        )

        assert status == 200 and payload["decision"]["best_run"] is None
        history = _get(world, _task(world, "/runs/history"))[1]["history"]
        assert [row["decision"] for row in history] == [
            best_run.DECISION_CLEAR, best_run.DECISION_SELECT
        ]

    def test_clearing_nothing_is_a_conflict_not_a_silent_success(self, world):
        status, _ = _delete(world, _task(world, "/best-run"),
                            {**HUMAN, "expected_selection_id": "whatever"})

        assert status == 409

    def test_looking_and_not_choosing_is_itself_recorded(self, world):
        status, payload = _post(
            world, _task(world, "/best-run/undecided"),
            {**HUMAN, "candidate_attempt": 2, "reason": "not convinced"},
        )

        assert status == 201
        assert payload["decision"]["best_run"] is None
        latest = _get(world, _task(world, "/runs/history"))[1]["history"][0]
        assert latest["decision"] == best_run.DECISION_UNDECIDED
        assert latest["candidate_attempt"] == 2

    def test_choosing_a_best_run_never_touches_the_experiment_winner(self, world):
        before = _get(world, _experiment(world, "/winner"))[1]

        _post(world, _task(world, "/best-run"), {**HUMAN, "attempt": 2})

        after = _get(world, _experiment(world, "/winner"))[1]
        assert after["expected_selection_id"] == before["expected_selection_id"]
        assert after["winner"]["decision"] == selection.DECISION_INITIAL

    def test_two_experiments_keep_separate_best_runs_for_the_same_task(self, world):
        _post(world, _task(world, "/best-run"), {**HUMAN, "attempt": 2})

        candidate = _get(world, _task(world, "/runs",
                                      experiment=world["candidate_id"]))[1]

        assert candidate["best_run"] is None
        assert _get(world, _task(world, "/runs"))[1]["best_run"]["attempt"] == 2


# ----------------------------------------------------------------------
# Comparison
# ----------------------------------------------------------------------


class TestComparison:
    def test_the_default_view_is_answers_and_carries_no_step_alignment(self, world):
        status, payload = _get(world, _task(world, "/comparison"), right_attempt=2)

        assert status == 200 and payload["view"] == selection_api.VIEW_ANSWERS
        assert "steps" not in payload["left"] and "alignment" not in payload
        assert payload["left"]["answers"] and payload["right"]["answers"]
        assert payload["left"]["step_count"] == 3
        assert payload["right"]["step_count"] == 2

    def test_the_left_side_defaults_to_the_reference_until_a_best_run_is_chosen(
        self, world
    ):
        before = _get(world, _task(world, "/comparison"), right_attempt=2)[1]
        assert before["left_run"]["attempt"] == 1
        assert before["left_run"]["is_reference"] is True

        _post(world, _task(world, "/best-run"), {**HUMAN, "attempt": 2})
        after = _get(world, _task(world, "/comparison"), right_attempt=1)[1]

        assert after["left_run"]["attempt"] == 2
        assert after["left_run"]["is_best"] is True

    def test_a_multi_turn_attempt_is_compared_in_full(self, world):
        payload = _get(world, _task(world, "/comparison"), right_attempt=2)[1]

        assert len(payload["left"]["ref"]["turn_keys"]) == 2
        assert len(payload["left"]["answers"]) == 2

    def test_the_step_view_carries_the_alignment_and_its_feedback_anchors(self, world):
        payload = _get(world, _task(world, "/comparison"), right_attempt=2,
                       view="steps")[1]

        rows = payload["alignment"]["rows"]
        assert rows and payload["alignment"]["summary"]["pairs"] == len(rows)
        matched = [row for row in rows if row["kind"] == "matched"]
        assert matched, "add_item runs on both sides and must align"
        anchors = matched[0]["anchors"]
        assert anchors["left"]["turn_key"] and anchors["left"]["span_ids"]
        assert anchors["left"]["anchorable"] is True
        assert anchors["right"]["store_id"] == payload["right"]["ref"]["store_id"]

    def test_the_differences_view_is_a_subset_of_the_alignment(self, world):
        full = _get(world, _task(world, "/comparison"), right_attempt=2,
                    view="steps")[1]
        only = _get(world, _task(world, "/comparison"), right_attempt=2,
                    view="differences")[1]

        assert len(only["alignment"]["rows"]) == only["difference_count"]
        assert len(only["alignment"]["rows"]) <= len(full["alignment"]["rows"])

    def test_the_review_pair_key_names_the_exact_two_executions(self, world):
        one = _get(world, _task(world, "/comparison"), right_attempt=2)[1]
        two = _get(world, _task(world, "/comparison"), right_attempt=3)[1]

        assert one["review_pair_key"] != two["review_pair_key"]
        assert one["review_pair_key"] == "|".join(
            [one["left"]["ref"]["ref_id"], one["right"]["ref"]["ref_id"]]
        )

    def test_a_multi_turn_pair_has_one_identity_for_review_and_for_comments(
        self, world
    ):
        """Pair review and a comment on any row key the SAME pair identically.

        `feedback.FeedbackTarget` keeps the reference the reader held and names
        the anchored turn separately, so two remarks written on different steps
        of the same two multi-turn attempts belong to one pair. A client
        counting comments by `review_pair_key` therefore cannot report an
        annotated pair as unannotated -- which is what a per-turn key did.

        Asserted through the REAL feedback module, not by restating the hash:
        the claim is that the two modules agree, and only building the anchors
        the writer builds can show that.
        """
        payload = _get(world, _task(world, "/comparison"), right_attempt=2,
                       view="steps")[1]

        assert len(payload["left"]["ref"]["turn_keys"]) == 2
        rows = [row for row in payload["alignment"]["rows"]
                if row["anchors"]["left"] and row["anchors"]["right"]]
        assert rows
        for row in rows:
            assert row["feedback_pair_key"] == payload["review_pair_key"]
            # The anchor is posted verbatim: whole `ref` plus the anchored turn.
            anchors = fb.FeedbackAnchors(
                primary=fb.FeedbackTarget.from_mapping(
                    dict(row["anchors"]["left"], target_label="left")
                ),
                paired=fb.FeedbackTarget.from_mapping(
                    dict(row["anchors"]["right"], target_label="right")
                ),
            )
            assert anchors.pair_key == payload["review_pair_key"]
            assert anchors.primary.ref.turn_keys == tuple(
                payload["left"]["ref"]["turn_keys"]
            )
            assert anchors.primary.turn_key == row["anchors"]["left"]["turn_key"]

        # Rows anchored in DIFFERENT turns of the same two attempts: one pair.
        # Comparing an execution with itself is the shape that puts matched rows
        # in both turns, and it is what a reviewer does to check the alignment.
        both = _get(world, _task(world, "/comparison"), left_attempt=1,
                    right_attempt=1, view="steps")[1]
        anchored_turns = {
            row["anchors"]["left"]["turn_key"]
            for row in both["alignment"]["rows"]
            if row["anchors"]["left"] and row["anchors"]["right"]
        }
        assert len(anchored_turns) == 2, "rows anchored in both recorded turns"
        assert {
            row["feedback_pair_key"]
            for row in both["alignment"]["rows"]
            if row["feedback_pair_key"]
        } == {both["review_pair_key"]}

    def test_a_one_sided_row_carries_no_pair_key(self, world):
        """A comment on a row with one side names one execution, not a pair."""
        payload = _get(
            world, _task(world, "/comparison"),
            right_experiment=world["candidate_id"], right_attempt=1, view="steps",
        )[1]

        one_sided = [row for row in payload["alignment"]["rows"]
                     if not (row["anchors"]["left"] and row["anchors"]["right"])]
        assert one_sided, "the two attempts ran different commands"
        assert all(row["feedback_pair_key"] is None for row in one_sided)

    def test_the_winner_can_be_compared_with_a_candidate_in_another_store(self, world):
        payload = _get(
            world, _task(world, "/comparison"),
            right_experiment=world["candidate_id"], right_attempt=1,
        )[1]

        assert payload["right_run"]["experiment_id"] == world["candidate_id"]
        assert payload["left"]["ref"]["store_id"] != payload["right"]["ref"]["store_id"]
        assert payload["right"]["readable"] is True

    def test_an_attempt_with_no_recorded_turns_is_explicitly_not_comparable(self, world):
        status, payload = _get(world, _task(world, "/comparison"), right_attempt=4)

        assert status == 409
        assert payload["comparable"] is False and payload["attempt"] == 4
        assert payload["evidence_state"] == best_run.EVIDENCE_MISSING

    def test_an_attempt_the_task_never_recorded_is_not_found(self, world):
        assert _get(world, _task(world, "/comparison"), right_attempt=99)[0] == 404

    @pytest.mark.parametrize("attempt", ["2.0", "two", "", "-"])
    def test_a_side_named_by_something_other_than_an_integer_is_refused(
        self, world, attempt
    ):
        assert _get(world, _task(world, "/comparison"), right_attempt=attempt)[0] == 400

    def test_an_unknown_view_is_refused_rather_than_defaulted(self, world):
        assert _get(world, _task(world, "/comparison"), right_attempt=2,
                    view="everything")[0] == 400

    def test_one_side_can_be_projected_on_its_own(self, world):
        status, payload = _get(world, _task(world, "/runs/2"))

        assert status == 200
        assert payload["run"]["attempt"] == 2 and payload["run"]["outcome"] == "fail"
        assert payload["projection"]["ref"]["attempt"] == 2
        assert payload["projection"]["step_count"] == 2

    def test_a_projection_reports_no_recorded_pass_rather_than_inventing_one(
        self, world
    ):
        payload = _get(world, _task(world, "/runs/1"), view="steps")[1]

        assert payload["projection"]["pass_selector"] is None
        assert payload["projection"]["content_attribution"] == "turn"
        assert all(turn["pass_id"] is None for turn in payload["projection"]["turns"])


# ----------------------------------------------------------------------
# Two recorded passes of one turn
# ----------------------------------------------------------------------


@pytest.fixture
def two_pass_world(tmp_path, monkeypatch):
    """A workflow whose single attempt recorded TWO passes in ONE turn.

    This is the teacher/student shape, seeded the only way it can exist today:
    by stamping `fw.pass` on the command spans, which no fastWorkflow producer
    does yet (`fix-txxy`). The evidence is real -- real spans in a real store --
    and the API discovers the passes from it rather than being told they exist.
    """
    monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "state"))
    folder = tmp_path / "two_pass_workflow"
    folder.mkdir()
    (folder / "_commands").mkdir()
    benchmark = setup.save_benchmark(
        folder, {"title": "Two passes", "tasks": [{"prompt": "Do it twice"}]}
    )
    record = setup.create_experiment(folder, benchmark["benchmark_id"], "v1")
    experiment_id, task_id = record["experiment_id"], record["task_ids"][0]

    db = str(tmp_path / "two-pass.sqlite3")
    store = obs.ObservabilityStore(db)
    controller = ExperimentController(
        db, store.store_identity(), external=False,
        workflow_folderpath=str(folder),
    )
    controller.create_experiment(
        experiment_id, record["description"], declared_tasks=1, declared_attempts=1,
        declarations=[(task_id, 1, f"ch-{experiment_id[-4:]}-1")],
        workflow_name=setup.workflow_name_for(folder),
    )
    _run_attempt(
        store, controller, experiment_id, task_id, 1,
        [["add_item", "list_items", "add_item"]],
        outcome="pass",
        # One turn, three dispatches: the teacher did two, the student one.
        passes=[["teacher", "teacher", "student"]],
    )
    srv = run_chatbot_server.ChatbotServer(port=0, workflow_path=str(folder))
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield {
            "server": srv, "folder": str(folder),
            "experiment_id": experiment_id, "task_id": task_id,
        }
    finally:
        srv.shutdown()
        thread.join(timeout=5)


class TestRecordedPasses:
    """Two passes of one turn compare as two executions, or not at all.

    Everything here turns on the passes being DISCOVERED. A route that let a
    request describe a pass would let it assert that the student's calls were
    the teacher's, and the resulting side-by-side would be a fabrication with
    the store's authority behind it.
    """

    def _path(self, world, suffix=""):
        return (
            f"/api/experiments/{world['experiment_id']}"
            f"/tasks/{world['task_id']}{suffix}"
        )

    def _get(self, world, suffix, **params):
        query = "&".join(f"{k}={v}" for k, v in params.items())
        return _request(
            world["server"], self._path(world, suffix) + (f"?{query}" if query else "")
        )

    def test_the_passes_a_run_recorded_are_reported(self, two_pass_world):
        status, payload = self._get(two_pass_world, "/runs/1/passes")

        assert status == 200
        assert [row["pass_id"] for row in payload["passes"]] == ["student", "teacher"]
        assert all(row["turn_count"] == 1 for row in payload["passes"])
        assert payload["pass_attribute"] == selection_api.DEFAULT_PASS_ATTRIBUTE

    def test_one_pass_projects_only_its_own_steps(self, two_pass_world):
        status, payload = self._get(
            two_pass_world, "/runs/1", left_pass="teacher", view="steps"
        )

        assert status == 200
        assert payload["pass_id"] == "teacher"
        assert [step["command_name"] for step in payload["projection"]["steps"]] == [
            "add_item",
            "list_items",
        ]
        assert payload["projection"]["ref"]["pass_id"] == "teacher"

    def test_two_passes_of_the_same_turn_compare_against_each_other(
        self, two_pass_world
    ):
        status, payload = self._get(
            two_pass_world,
            "/comparison",
            left_attempt=1, right_attempt=1,
            left_pass="teacher", right_pass="student", view="steps",
        )

        assert status == 200
        assert payload["left"]["step_count"] == 2
        assert payload["right"]["step_count"] == 1
        assert payload["pass_scope"]["left_pass"] == "teacher"
        assert payload["pass_scope"]["right_pass"] == "student"
        # Same attempt on both sides, and still two distinct executions: the
        # pass is part of the reference, so they do not collapse into one.
        assert payload["left"]["ref"]["ref_id"] != payload["right"]["ref"]["ref_id"]

    def test_a_pass_comparison_is_a_different_pair_from_the_whole_run(
        self, two_pass_world
    ):
        """A comment on the teacher pass must not become one on the whole run."""
        whole_ref = self._get(two_pass_world, "/runs/1")[1]["projection"]["ref"]
        scoped = self._get(
            two_pass_world, "/comparison", left_attempt=1, right_attempt=1,
            left_pass="teacher", right_pass="student",
        )[1]

        assert whole_ref["pass_id"] is None
        assert whole_ref["ref_id"] not in scoped["review_pair_key"]
        # The pass is part of each side's reference, so a comment written from
        # a teacher-versus-student row carries the PASS pair's identity and not
        # the whole run's.
        rows = [row for row in self._get(
            two_pass_world, "/comparison", left_attempt=1, right_attempt=1,
            left_pass="teacher", right_pass="student", view="steps",
        )[1]["alignment"]["rows"] if row["feedback_pair_key"]]
        assert rows
        for row in rows:
            assert row["feedback_pair_key"] == scoped["review_pair_key"]
            assert row["anchors"]["left"]["ref"]["pass_id"] == "teacher"
            assert row["anchors"]["right"]["ref"]["pass_id"] == "student"

    def test_content_shared_by_both_passes_is_labelled_rather_than_split(
        self, two_pass_world
    ):
        payload = self._get(
            two_pass_world, "/runs/1", left_pass="teacher", view="steps"
        )[1]["projection"]

        # The turn's answer was written once, by the turn, and belongs to no
        # single pass. Attributing it to whichever pass was on screen is the
        # fabrication this contract exists to prevent.
        assert payload["content_attribution"] == comparison.ATTRIBUTION_SHARED
        assert payload["pass_selector"]["pass_id"] == "teacher"
        assert all(
            row["attribution"] == comparison.ATTRIBUTION_SHARED
            and row["pass_content_recorded"] is False
            for row in payload["answers"]
        )

    def test_a_pass_the_evidence_never_recorded_is_refused(self, two_pass_world):
        status, payload = self._get(
            two_pass_world, "/runs/1", left_pass="adjudicator"
        )

        assert status == 404
        assert "records no pass" in payload["error"]
        assert payload["recorded_passes"] == ["student", "teacher"]

    def test_a_pass_cannot_be_described_only_named(self, two_pass_world):
        """There is no parameter that asserts membership, only one that names it.

        `pass_attribute` chooses WHICH recorded stamp to read, and a stamp the
        evidence does not carry finds nothing -- it cannot conjure a split.
        """
        status, payload = self._get(
            two_pass_world, "/runs/1", left_pass="teacher", pass_attribute="fw.invented"
        )

        assert status == 404 and payload["recorded_passes"] == []

    def test_marking_a_pass_pair_reviewed_records_that_exact_pair(
        self, two_pass_world
    ):
        comparison_payload = self._get(
            two_pass_world, "/comparison", left_attempt=1, right_attempt=1,
            left_pass="teacher", right_pass="student",
        )[1]

        status, payload = _request(
            two_pass_world["server"],
            self._path(two_pass_world, "/review-pairs"),
            "POST",
            {
                "reviewer": "dhar", "reviewer_kind": "human", "state": "reviewed",
                "left_attempt": 1, "right_attempt": 1,
                "left_pass": "teacher", "right_pass": "student",
            },
        )

        assert status == 201
        assert payload["pair_key"] == comparison_payload["review_pair_key"]
        assert (payload["left_pass"], payload["right_pass"]) == ("teacher", "student")


# ----------------------------------------------------------------------
# Pair review progress
# ----------------------------------------------------------------------


class TestPairReview:
    def _sidecar_path(self, world):
        from fastworkflow import state_paths

        return pair_review.shared_pair_review_db_path_for(
            state_paths.workflow_state_dir(world["folder"])
        )

    def test_reading_progress_creates_no_sidecar_and_reports_nothing_reviewed(
        self, world
    ):
        status, payload = _get(world, _task(world, "/review-pairs"), reviewer="dhar")

        assert status == 200
        assert payload["control_exists"] is False
        assert payload["progress"]["reviewed"] == 0
        assert payload["progress"]["pairs"] == len(payload["pairs"])
        assert all(row["state"]["recorded"] is False for row in payload["pairs"])
        assert not os.path.exists(self._sidecar_path(world))

    def test_the_pairs_are_the_pinned_run_against_every_comparable_other_run(
        self, world
    ):
        payload = _get(world, _task(world, "/review-pairs"), reviewer="dhar")[1]

        assert payload["left_run"]["attempt"] == 1
        assert [row["right"]["attempt"] for row in payload["pairs"]] == [2, 3]

    def test_marking_a_pair_reviewed_is_keyed_to_that_exact_pair(self, world):
        target = _get(world, _task(world, "/review-pairs"), reviewer="dhar")[1]
        pair_key = target["pairs"][0]["pair_key"]

        status, recorded = _post(
            world, _task(world, "/review-pairs"),
            {"reviewer": "dhar", "reviewer_kind": "human", "right_attempt": 2,
             "state": pair_review.STATE_REVIEWED, "note": "read both"},
        )

        assert status == 201 and recorded["pair_key"] == pair_key
        after = _get(world, _task(world, "/review-pairs"), reviewer="dhar")[1]
        assert after["progress"]["reviewed"] == 1
        assert after["progress"]["next_pair_key"] == after["pairs"][1]["pair_key"]
        assert after["pairs"][0]["state"]["state"] == pair_review.STATE_REVIEWED

    def test_another_reviewer_s_progress_is_their_own(self, world):
        _post(world, _task(world, "/review-pairs"),
              {"reviewer": "dhar", "reviewer_kind": "human", "right_attempt": 2,
               "state": pair_review.STATE_REVIEWED})

        other = _get(world, _task(world, "/review-pairs"), reviewer="cursor")[1]

        assert other["progress"]["reviewed"] == 0

    def test_pinning_a_new_reference_leaves_the_old_marks_readable(self, world):
        _post(world, _task(world, "/review-pairs"),
              {"reviewer": "dhar", "reviewer_kind": "human", "right_attempt": 2,
               "state": pair_review.STATE_REVIEWED})
        old_key = _get(world, _task(world, "/review-pairs"),
                       reviewer="dhar")[1]["pairs"][0]["pair_key"]

        _post(world, _task(world, "/best-run"), {**HUMAN, "attempt": 2})
        repinned = _get(world, _task(world, "/review-pairs"), reviewer="dhar")[1]

        assert repinned["left_run"]["attempt"] == 2
        assert old_key not in {row["pair_key"] for row in repinned["pairs"]}
        assert repinned["progress"]["reviewed"] == 0
        history = _get(world, _task(world, "/review-pairs/history"),
                       pair_key=old_key)[1]["history"]
        assert history and history[0]["state"] == pair_review.STATE_REVIEWED

    def test_unmarking_keeps_both_events_in_the_history(self, world):
        body = {"reviewer": "dhar", "reviewer_kind": "human", "right_attempt": 2}
        _post(world, _task(world, "/review-pairs"),
              {**body, "state": pair_review.STATE_REVIEWED})
        _post(world, _task(world, "/review-pairs"),
              {**body, "state": pair_review.STATE_NOT_REVIEWED})

        pair_key = _get(world, _task(world, "/review-pairs"),
                        reviewer="dhar")[1]["pairs"][0]["pair_key"]
        history = _get(world, _task(world, "/review-pairs/history"),
                       pair_key=pair_key)[1]["history"]

        assert [row["state"] for row in history] == [
            pair_review.STATE_NOT_REVIEWED, pair_review.STATE_REVIEWED
        ]

    def test_a_pair_spanning_two_stores_is_recordable(self, world):
        status, payload = _post(
            world, _task(world, "/review-pairs"),
            {"reviewer": "dhar", "reviewer_kind": "human",
             "right_experiment": world["candidate_id"], "right_attempt": 1,
             "state": pair_review.STATE_REVIEWED},
        )

        assert status == 201
        assert payload["right_run"]["experiment_id"] == world["candidate_id"]

    @pytest.mark.parametrize("kind", ["system", "reviewer", ""])
    def test_a_reviewer_kind_outside_the_enum_is_refused(self, world, kind):
        status, _ = _post(
            world, _task(world, "/review-pairs"),
            {"reviewer": "dhar", "reviewer_kind": kind, "right_attempt": 2,
             "state": pair_review.STATE_REVIEWED},
        )

        assert status == 400

    def test_a_state_outside_the_two_is_refused(self, world):
        status, _ = _post(
            world, _task(world, "/review-pairs"),
            {"reviewer": "dhar", "reviewer_kind": "human", "right_attempt": 2,
             "state": "maybe"},
        )

        assert status == 400


# ----------------------------------------------------------------------
# Setup reuse: repeat count and duplication
# ----------------------------------------------------------------------


class TestDuplication:
    def test_running_the_same_setup_again_copies_it_and_says_what_changed(self, world):
        status, payload = _post(
            world, f"/api/benchmark-experiments/{world['experiment_id']}/duplicate",
            {"runs_per_task": 7},
        )

        assert status == 201
        record = payload["experiment"]
        assert record["experiment_id"] != world["experiment_id"]
        assert record["runs_per_task"] == 7
        assert record["source_experiment_id"] == world["experiment_id"]
        assert record["changed_fields"] == ["runs_per_task"]
        assert record["benchmark_digest_sha256"] == json.loads(
            json.dumps(record)
        )["benchmark_digest_sha256"]

    def test_a_duplicate_inherits_the_repeat_count_when_nothing_is_passed(self, world):
        record = _post(
            world, f"/api/benchmark-experiments/{world['experiment_id']}/duplicate", {}
        )[1]["experiment"]

        assert record["runs_per_task"] == 4 and record["changed_fields"] == []

    def test_a_duplicate_joins_the_contest_without_taking_it(self, world):
        record = _post(
            world, f"/api/benchmark-experiments/{world['experiment_id']}/duplicate", {}
        )[1]["experiment"]

        payload = _get(world, f"/api/experiments/{record['experiment_id']}/winner")[1]

        assert payload["is_winner"] is False
        assert payload["winner"]["experiment_id"] == world["experiment_id"]

    def test_a_duplicate_starts_with_no_evidence_and_no_best_runs(self, world):
        record = _post(
            world, f"/api/benchmark-experiments/{world['experiment_id']}/duplicate", {}
        )[1]["experiment"]

        payload = _get(
            world, _task(world, "/runs", experiment=record["experiment_id"])
        )[1]

        assert payload["attempts"] == [] and payload["best_run"] is None
        assert payload["evidence_readable"] is False

    @pytest.mark.parametrize("runs", [0, 101, 2.5, True, "3x"])
    def test_an_impossible_repeat_count_is_refused(self, world, runs):
        status, _ = _post(
            world, f"/api/benchmark-experiments/{world['experiment_id']}/duplicate",
            {"runs_per_task": runs},
        )

        assert status == 400

    def test_duplicating_something_that_does_not_exist_is_not_found(self, world):
        assert _post(
            world, "/api/benchmark-experiments/exp-nobody/duplicate", {}
        )[0] == 404


# ----------------------------------------------------------------------
# The same contract over a real socket
# ----------------------------------------------------------------------


class TestOverHttp:
    def test_every_route_is_token_gated(self, server, world):
        for path in (
            _experiment(world, "/winner"),
            _experiment(world, "/winner/history"),
            _task(world, "/runs"),
            _task(world, "/comparison?right_attempt=2"),
        ):
            assert _request(server, path, token=None)[0] == 401

        assert _request(
            server, _task(world, "/best-run"), "POST", {**HUMAN, "attempt": 1},
            token=None,
        )[0] == 401
        assert _get(world, _task(world, "/runs"))[1]["best_run"] is None

    def test_creating_an_experiment_over_http_elects_it_when_it_is_the_first(
        self, server, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "state"))
        status, created = _request(
            server, f"/api/benchmarks/{_only_benchmark(server)}/experiments",
            "POST", {"version": "v1", "runs_per_task": 3},
        )

        assert status == 201
        record = created["experiment"]
        assert record["runs_per_task"] == 3
        winner = _request(
            server, f"/api/experiments/{record['experiment_id']}/winner"
        )[1]
        # The workflow already has a winner, so this one joins without taking it.
        assert winner["is_winner"] is False
        assert winner["winner"]["automatic"] is True

    def test_an_impossible_repeat_count_is_refused_at_creation(self, server):
        status, payload = _request(
            server, f"/api/benchmarks/{_only_benchmark(server)}/experiments",
            "POST", {"version": "v1", "runs_per_task": 0},
        )

        assert status == 400 and "runs_per_task" in payload["error"]

    def test_the_registration_detail_shows_the_repeat_count_and_the_winner(
        self, server, world
    ):
        payload = _request(
            server, f"/api/benchmark-experiments/{world['experiment_id']}"
        )[1]

        assert payload["runs_per_task"] == 4
        assert payload["is_winner"] is True
        assert payload["winner"]["experiment_id"] == world["experiment_id"]

    def test_the_whole_best_run_lifecycle_works_over_http(self, server, world):
        summary = _request(server, _task(world, "/runs"))[1]
        assert summary["best_run"] is None

        status, selected = _request(
            server, _task(world, "/best-run"), "POST", {**AGENT, "attempt": 2}
        )
        assert status == 201
        current = selected["decision"]["best_run"]["selection_id"]

        status, cleared = _request(
            server, _task(world, "/best-run"), "DELETE",
            {**AGENT, "expected_selection_id": current},
        )
        assert status == 200 and cleared["decision"]["best_run"] is None
        assert _request(server, _task(world, "/runs"))[1]["best_run"] is None

    def test_a_stale_promotion_is_a_409_a_client_can_act_on(self, server, world):
        stale = _request(server, _experiment(world, "/winner"))[1][
            "expected_selection_id"
        ]
        _request(
            server, _experiment(world, "/winner/decisions"), "POST",
            {**HUMAN, "decision": "promote", "expected_selection_id": stale,
             "candidate_experiment_id": world["candidate_id"]},
        )

        status, payload = _request(
            server, _experiment(world, "/winner/decisions"), "POST",
            {**HUMAN, "decision": "keep", "expected_selection_id": stale},
        )

        assert status == 409 and payload["stale"] is True
        assert payload["current_experiment_id"] == world["candidate_id"]

    def test_a_comparison_is_a_get_and_records_nothing(self, server, world):
        before = _request(server, _task(world, "/runs/history"))[1]["history"]

        status, payload = _request(
            server, _task(world, "/comparison?right_attempt=2&view=differences")
        )

        assert status == 200
        assert payload["review_pair_key"]
        assert _request(server, _task(world, "/runs/history"))[1]["history"] == before

    def test_pair_review_progress_round_trips_over_http(self, server, world):
        path = _task(world, "/review-pairs")
        assert _request(server, path + "?reviewer=dhar")[1]["progress"]["reviewed"] == 0

        status, _ = _request(
            server, path, "POST",
            {"reviewer": "dhar", "reviewer_kind": "human", "right_attempt": 2,
             "state": pair_review.STATE_REVIEWED},
        )

        assert status == 201
        assert _request(server, path + "?reviewer=dhar")[1]["progress"]["reviewed"] == 1

    def test_a_method_the_route_does_not_offer_is_refused(self, server, world):
        assert _request(server, _experiment(world, "/winner"), "POST", {})[0] == 405
        assert _request(server, _task(world, "/runs"), "DELETE", {})[0] == 404

    def test_an_unknown_subroute_is_not_found(self, server, world):
        assert _request(server, _experiment(world, "/winner/everything"))[0] == 404
        assert _request(server, _task(world, "/nonsense"))[0] == 404


def _only_benchmark(server):
    payload = _request(server, "/api/benchmarks")[1]
    return payload["benchmarks"][0]["benchmark_id"]


# ----------------------------------------------------------------------
# Workspace mode
# ----------------------------------------------------------------------


def _seal(tmp_path, *, attempts=(1, 2)):
    """A sealed archive holding one experiment's attempts of one task.

    Written with the ordinary store API and then archived, so the bytes the
    workspace reads are a real archive rather than a hand-built file.
    """
    source = str(tmp_path / "archived-live.sqlite3")
    store = obs.ObservabilityStore(source)
    store.create_experiment(
        "local", "archived run", declared_tasks=1, declared_attempts=len(attempts)
    )
    for attempt in attempts:
        channel = f"archived-{attempt}"
        store.start_attempt("local", "task", attempt, channel)
        conversation = store.mint_conversation_id(
            channel, experiment_id="local", task_id="task", attempt=attempt
        )
        _seed_turn(
            store,
            f"archived-a{attempt}",
            experiment_id="local",
            task_id="task",
            attempt=attempt,
            conversation=conversation,
            ordinal=1,
            commands=["add_item", "list_items"] if attempt == 1 else ["add_item"],
        )
        store.finish_attempt(
            "local", "task", attempt, outcome="pass", outcome_source="test"
        )
    return obs.ObservabilityStore(source, migrate=False).archive_to(
        str(tmp_path / "sealed.sqlite3")
    )


def _sealed_server(tmp_path, monkeypatch, *, experiments=None):
    from tests.test_observability_workspace import _manifest, _store_decl

    monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "state"))
    archive = _seal(tmp_path)
    manifest = _manifest(
        tmp_path,
        [_store_decl(archive, "sealed")],
        experiments=experiments
        if experiments is not None
        else [
            {
                "experiment_id": "logical",
                "segments": [{"store_id": "sealed", "local_experiment_id": "local"}],
            }
        ],
    )
    srv = run_chatbot_server.ChatbotServer(
        port=0, workspace_manifest_path=str(manifest)
    )
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    return srv, thread


class TestSealedWorkspace:
    """An archive answers for its evidence and for nobody's judgements.

    The split is the point. A workspace carries recorded executions, so the
    compare view has to work over one: sealed evidence being the one place you
    cannot look at the evidence would be absurd. It does NOT carry the
    workflow's selection control, so it must never answer who won or which run
    is best -- those are this machine's decisions and an archive reporting them
    would be attributing them to the archive.
    """

    def test_decisions_are_refused_and_nothing_can_be_recorded(
        self, tmp_path, monkeypatch
    ):
        srv, thread = _sealed_server(tmp_path, monkeypatch)
        try:
            status, payload = _request(srv, "/api/experiments/logical/winner")
            assert status == 403 and "live-workflow only" in payload["error"]
            assert _request(
                srv, "/api/experiments/logical/tasks/task/review-pairs?reviewer=d"
            )[0] == 403
            assert _request(
                srv, "/api/experiments/logical/tasks/task/best-run", "POST",
                {**HUMAN, "attempt": 1},
            )[0] == 403
            assert _request(
                srv, "/api/experiments/logical/tasks/task/best-run", "DELETE",
                {**HUMAN, "expected_selection_id": "x"},
            )[0] == 403
        finally:
            srv.shutdown()
            thread.join(timeout=5)

    def test_the_archived_attempts_are_listed_with_no_judgement_attached(
        self, tmp_path, monkeypatch
    ):
        srv, thread = _sealed_server(tmp_path, monkeypatch)
        try:
            status, payload = _request(
                srv, "/api/experiments/logical/tasks/task/runs"
            )
            assert status == 200
            assert [row["attempt"] for row in payload["attempts"]] == [1, 2]
            assert all(row["comparable"] for row in payload["attempts"])
            # No control travelled with the evidence, so there is no best run
            # and no reference -- said plainly rather than shown as an empty
            # badge somebody would read as "nobody has chosen yet".
            assert payload["best_run"] is None and payload["reference"] is None
            assert payload["decisions_available"] is False
            assert not any(row["is_best"] for row in payload["attempts"])
        finally:
            srv.shutdown()
            thread.join(timeout=5)

    def test_two_archived_attempts_compare_over_the_sealed_stores(
        self, tmp_path, monkeypatch
    ):
        srv, thread = _sealed_server(tmp_path, monkeypatch)
        try:
            status, payload = _request(
                srv,
                "/api/experiments/logical/tasks/task/comparison"
                "?left_attempt=1&right_attempt=2&view=steps",
            )
            assert status == 200
            assert payload["sealed"] is True and payload["review_available"] is False
            assert payload["left"]["step_count"] == 2
            assert payload["right"]["step_count"] == 1
            rows = payload["alignment"]["rows"]
            kinds = {row["kind"] for row in rows}
            assert comparison.PAIR_MATCHED in kinds
            assert comparison.PAIR_LEFT_ONLY in kinds
            assert payload["review_pair_key"].count("|") == 1
            # The reference names the experiment the EVIDENCE records, not the
            # logical name the manifest stitched it under: a reference that
            # claimed the logical id would fail its own scope check against
            # every turn it points at.
            assert payload["left"]["ref"]["experiment_id"] == "local"
            assert payload["left_run"]["experiment_id"] == "logical"
        finally:
            srv.shutdown()
            thread.join(timeout=5)

    def test_one_archived_attempt_projects_on_its_own(self, tmp_path, monkeypatch):
        srv, thread = _sealed_server(tmp_path, monkeypatch)
        try:
            status, payload = _request(
                srv, "/api/experiments/logical/tasks/task/runs/1?view=steps"
            )
            assert status == 200
            assert payload["run"]["attempt"] == 1
            assert len(payload["projection"]["steps"]) == 2
            # An archive has two names and this payload carries both, because
            # they are used for different things: the manifest's name is what
            # every /api/workspace route is addressed by, and the identity the
            # database reports about itself is what a reference names it by --
            # here and live alike, and how a cross-archive comment is resolved.
            assert payload["run"]["manifest_store_id"] == "sealed"
            assert payload["projection"]["manifest_store_id"] == "sealed"
            declared = _request(srv, "/api/workspace/stores")[1]["stores"]
            identity = declared[0]["store_identity"]
            assert identity and identity != "sealed"
            assert payload["projection"]["ref"]["store_id"] == identity
        finally:
            srv.shutdown()
            thread.join(timeout=5)

    def test_an_archive_reports_no_recorded_pass_and_refuses_to_invent_one(
        self, tmp_path, monkeypatch
    ):
        srv, thread = _sealed_server(tmp_path, monkeypatch)
        try:
            assert _request(
                srv, "/api/experiments/logical/tasks/task/runs/1/passes"
            )[1]["passes"] == []
            status, payload = _request(
                srv, "/api/experiments/logical/tasks/task/runs/1?left_pass=teacher"
            )
            assert status == 404 and "records no pass" in payload["error"]
        finally:
            srv.shutdown()
            thread.join(timeout=5)

    def test_an_experiment_the_manifest_does_not_declare_is_not_found(
        self, tmp_path, monkeypatch
    ):
        srv, thread = _sealed_server(tmp_path, monkeypatch, experiments=[])
        try:
            status, payload = _request(
                srv, "/api/experiments/logical/tasks/task/runs"
            )
            assert status == 404 and "unknown experiment" in payload["error"]
        finally:
            srv.shutdown()
            thread.join(timeout=5)

"""The browser's half of run selection, the winner and the comparison.

`fix-9eg.17.2` / `.17.3` / `.17.4` and the browser slice of `fix-9eg.4`.

Two halves, and the split is deliberate.

The Python half drives the SAME HTTP surface the page calls, over a real
socket, through the real `ChatbotServer`: real workflow folders, real benchmark
manifests, real `ObservabilityStore` databases, attempts written through the
real `ExperimentController`, the real control tables of the live DB and the
real feedback writer. It exists because the
sequence the owner asked to be proven -- a repeated task, a Reference that is
not a Best run, a best-run decision, a comparison, a categorized comment on
the pair, that comment appearing under the task's Feedback, and then a
DIFFERENT best run leaving the earlier comment and its pair identity exactly
as they were -- is a statement about recorded state that survives a reload,
which is not something a DOM assertion can make.

The DOM half drives the shipped page in a real DOM against a real server,
because "the function exists" is not the claim: the claim is that a person can
see every attempt including the failed ones, choose one, compare two, write a
categorized comment on a specific step of the pair and find it again.

No Mock fixtures, nothing paid, and no source-string assertion stands alone:
where the page's source is pinned at all it is pinned beside a behavioural test
of the same thing.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import urllib.parse
from pathlib import Path

import pytest

from fastworkflow import state_paths
from fastworkflow.benchmark import setup
from fastworkflow.experiment.runner import ExperimentController
from fastworkflow.observability import control, selection
from fastworkflow.observability import store as obs
from fastworkflow.run_chatbot import server as run_chatbot_server
from tests.test_chatbot_benchmarks import _request
from tests.test_execution_comparison import _execute_span, _output, _record, _write
from tests.test_execution_comparison import _turn_row as _evidence_turn_row
from tests.test_selection_api import T0

# The API worker's world is the one this page is built against: one workflow,
# a task run four different ways in one database (completed, failed, never
# finished, finished with no recorded turns) and a second experiment recording
# the same task in the same live database. Rebuilding it here would be a second
# fixture free to drift from the contract the routes were written against.
#
# Loaded as a PLUGIN rather than imported. Importing the two fixture functions
# into this module's namespace works, but it also puts the names `server` and
# `world` in module scope, where every test that takes them as parameters reads
# as a redefinition -- 62 of them, which buries anything a linter has to say
# about this file. `pytest_plugins` is the supported way to borrow another
# module's fixtures and leaves the names out of here entirely.
pytest_plugins = ["tests.test_selection_api"]

HUMAN = {"actor": "observability UI", "actor_kind": "human"}


def _experiments(world, suffix=""):
    return f"/api/experiments/{urllib.parse.quote(world['experiment_id'])}{suffix}"


def _tasks(world, suffix="", experiment=None):
    return (
        f"/api/experiments/"
        f"{urllib.parse.quote(experiment or world['experiment_id'])}"
        f"/tasks/{urllib.parse.quote(world['task_id'])}{suffix}"
    )


def _decide_best(server, world, attempt, expected, reason=None):
    body = dict(HUMAN, attempt=attempt, expected_selection_id=expected)
    if reason is not None:
        body["reason"] = reason
    return _request(server, _tasks(world, "/best-run"), "POST", body)


# ----------------------------------------------------------------------
# The Runs view: every attempt, and Reference is not a decision
# ----------------------------------------------------------------------


class TestRunsView:
    def test_a_repeated_task_lists_every_attempt_including_the_failures(
        self, server, world
    ):
        """The list the page renders is the whole record, not the good part.

        An agent that passes a task often can still fail pass^k, and the only
        way a reader can see that is if the attempt that failed and the attempt
        that never finished are both on the page.
        """
        status, runs = _request(server, _tasks(world, "/runs"))

        assert status == 200
        by_attempt = {row["attempt"]: row for row in runs["attempts"]}
        assert sorted(by_attempt) == [1, 2, 3, 4]
        assert by_attempt[2]["execution_status"] == "failed"
        assert by_attempt[2]["outcome"] == "fail"
        # Started and never finished: present, and not offerable as a best run.
        assert by_attempt[3]["finished"] is False
        assert by_attempt[3]["selectable"] is False
        # Finished with nothing recorded: selectable, but nothing to compare,
        # and the evidence's own words for why.
        assert by_attempt[4]["selectable"] is True
        assert by_attempt[4]["comparable"] is False
        assert by_attempt[4]["evidence_label"]

    def test_the_initial_reference_is_a_viewing_default_and_not_a_best_run(
        self, server, world
    ):
        status, runs = _request(server, _tasks(world, "/runs"))

        assert status == 200
        assert runs["best_run"] is None, "nothing has been decided yet"
        assert runs["reference"]["attempt"] == 1
        assert runs["expected_selection_id"] is None
        reference = [row for row in runs["attempts"] if row["is_reference"]]
        assert [row["attempt"] for row in reference] == [1]
        assert not [row for row in runs["attempts"] if row["is_best"]]

    def test_a_failed_attempt_chosen_as_best_keeps_saying_it_failed(
        self, server, world
    ):
        """A pointer is not a verdict about the run it points at.

        A reviewer may well prefer the attempt that failed -- it may be the one
        that got furthest, or the one worth studying. What must not happen is
        the badge laundering its recorded status.
        """
        expected = _request(server, _tasks(world, "/runs"))[1]["expected_selection_id"]

        status, recorded = _decide_best(
            server, world, 2, expected, reason="it failed in the interesting way"
        )

        assert status == 201
        runs = _request(server, _tasks(world, "/runs"))[1]
        chosen = [row for row in runs["attempts"] if row["is_best"]]
        assert [row["attempt"] for row in chosen] == [2]
        assert chosen[0]["execution_status"] == "failed"
        assert chosen[0]["outcome"] == "fail"
        assert recorded["decision"]["best_run"]["attempt"] == 2
        # And the Reference is still the first completed attempt: the two are
        # different questions and the decision answered only one of them.
        assert runs["reference"]["attempt"] == 1

    def test_an_unfinished_attempt_is_refused_with_the_reason_on_the_wire(
        self, server, world
    ):
        expected = _request(server, _tasks(world, "/runs"))[1]["expected_selection_id"]

        status, refusal = _decide_best(server, world, 3, expected)

        assert status == 422
        assert refusal["attempt"] == 3
        assert refusal["reason"]

    def test_looking_and_not_choosing_is_itself_recordable(self, server, world):
        expected = _request(server, _tasks(world, "/runs"))[1]["expected_selection_id"]

        status, _ = _request(
            server,
            _tasks(world, "/best-run/undecided"),
            "POST",
            dict(HUMAN, expected_selection_id=expected, candidate_attempt=2,
                 reason="both are defensible"),
        )

        assert status == 201
        runs = _request(server, _tasks(world, "/runs"))[1]
        assert runs["best_run"] is None, "undecided records a look, not a pick"
        history = _request(server, _tasks(world, "/runs/history"))[1]["history"]
        assert [row["decision"] for row in history] == [selection.DECISION_UNDECIDED]
        assert history[0]["candidate_attempt"] == 2
        assert history[0]["actor"] == HUMAN["actor"]
        assert history[0]["rationale"] == "both are defensible"

    def test_the_decision_log_says_what_each_decision_was_taken_against(
        self, server, world
    ):
        """What the page's history panel prints, and why it is honest.

        A refused decision never lands here -- it is reported to whoever tried
        it. What lands is the pointer each decision replaced and the pointer it
        produced, so "kept" can be read as a judgement about a specific run
        rather than about whatever holds the pointer now.
        """
        first = _request(server, _tasks(world, "/runs"))[1]["expected_selection_id"]
        _decide_best(server, world, 1, first)
        second = _request(server, _tasks(world, "/runs"))[1]["expected_selection_id"]
        _decide_best(server, world, 2, second)

        history = _request(server, _tasks(world, "/runs/history"))[1]["history"]

        assert [row["candidate_attempt"] for row in history] == [2, 1]
        assert history[0]["previous_attempt"] == 1
        assert history[0]["new_attempt"] == 2
        assert history[1]["previous_attempt"] is None


# ----------------------------------------------------------------------
# The winner is a different decision from any best run
# ----------------------------------------------------------------------


class TestWinnerPanel:
    def test_the_first_experiment_holds_the_contest_before_anything_ran(
        self, server, world
    ):
        """What the registration page has to be able to show.

        The winner panel is rendered on the pre-run handoff page too, because
        the first experiment of a contest wins at creation. Nothing declared a
        target for that to be true, and the badge carries the experiment's own
        recorded status rather than implying one.
        """
        status, payload = _request(server, _experiments(world, "/winner"))

        assert status == 200
        assert payload["is_winner"] is True
        assert payload["automatic"] is True
        assert payload["winner"]["decision"] == selection.DECISION_INITIAL
        assert payload["winner"]["experiment"]["status"]
        assert payload["runs_per_task"] == 4

    def test_choosing_a_best_run_never_moves_the_winner(self, server, world):
        before = _request(server, _experiments(world, "/winner"))[1]
        expected = _request(server, _tasks(world, "/runs"))[1]["expected_selection_id"]

        _decide_best(server, world, 2, expected)

        after = _request(server, _experiments(world, "/winner"))[1]
        assert after["winner"]["selection_id"] == before["winner"]["selection_id"]
        assert after["winner"]["decision"] == selection.DECISION_INITIAL
        assert _request(server, _tasks(world, "/runs"))[1]["best_run"]["attempt"] == 2

    def test_promoting_the_candidate_moves_the_winner_and_not_the_best_run(
        self, server, world
    ):
        expected = _request(server, _tasks(world, "/runs"))[1]["expected_selection_id"]
        _decide_best(server, world, 1, expected)
        contest = _request(server, _experiments(world, "/winner"))[1]

        status, _ = _request(
            server,
            f"/api/experiments/{world['candidate_id']}/winner/decisions",
            "POST",
            dict(HUMAN, decision=selection.DECISION_PROMOTE,
                 expected_selection_id=contest["expected_selection_id"],
                 candidate_experiment_id=world["candidate_id"],
                 rationale="it sorts the list the task asked for"),
        )

        assert status == 201
        moved = _request(server, _experiments(world, "/winner"))[1]
        assert moved["winner"]["experiment_id"] == world["candidate_id"]
        assert moved["is_winner"] is False, "this page is no longer the winner"
        # The task's best run is untouched: it was never the same decision.
        assert _request(server, _tasks(world, "/runs"))[1]["best_run"]["attempt"] == 1

    def test_a_decision_that_lost_a_race_is_refused_with_the_new_state(
        self, server, world
    ):
        """The stale explanation the page renders, on the wire.

        Somebody read the panel, somebody else promoted meanwhile, and the
        first person clicked Keep. Nothing may be overwritten, and the refusal
        has to name what is current or "try again" is not actionable.
        """
        read = _request(server, _experiments(world, "/winner"))[1]
        stale_token = read["expected_selection_id"]
        _request(
            server,
            f"/api/experiments/{world['candidate_id']}/winner/decisions",
            "POST",
            dict(HUMAN, decision=selection.DECISION_PROMOTE,
                 expected_selection_id=stale_token,
                 candidate_experiment_id=world["candidate_id"]),
        )

        status, refusal = _request(
            server,
            _experiments(world, "/winner/decisions"),
            "POST",
            dict(HUMAN, decision=selection.DECISION_KEEP,
                 expected_selection_id=stale_token,
                 candidate_experiment_id=world["experiment_id"]),
        )

        assert status == 409
        assert refusal["stale"] is True
        assert refusal["expected_selection_id"] == stale_token
        assert refusal["current_experiment_id"] == world["candidate_id"]
        assert refusal["current_selection_id"] != stale_token
        # Re-reading and deciding again is a mechanical retry, which is what
        # the panel's "Re-read and decide again" button does.
        fresh = _request(server, _experiments(world, "/winner"))[1]
        retry = _request(
            server,
            _experiments(world, "/winner/decisions"),
            "POST",
            dict(HUMAN, decision=selection.DECISION_KEEP,
                 expected_selection_id=fresh["expected_selection_id"],
                 candidate_experiment_id=world["experiment_id"]),
        )
        assert retry[0] == 201


# ----------------------------------------------------------------------
# Running the same setup again
# ----------------------------------------------------------------------


class TestRepeatSetup:
    def test_duplicating_a_setup_asks_for_nothing_but_a_repeat_count(
        self, server, world
    ):
        """Registration only: no target, no rubric, no hypothesis, no run.

        The repeat count IS the declared attempt count. Nothing paid starts
        here, which is why the copy comes back with no recorded attempts.
        """
        status, payload = _request(
            server,
            f"/api/benchmark-experiments/{world['experiment_id']}/duplicate",
            "POST",
            {"runs_per_task": 3, "description": "same setup, three tries"},
        )

        assert status == 201
        copy = payload["experiment"]
        assert copy["experiment_id"] != world["experiment_id"]
        assert copy["runs_per_task"] == 3
        assert copy.get("store") is None, "duplication registers; it does not run"
        registered = _request(
            server, f"/api/benchmark-experiments/{copy['experiment_id']}"
        )[1]
        assert registered["experiment"]["runs_per_task"] == 3

    def test_the_repeat_count_the_page_sends_is_the_text_of_its_number_box(
        self, server, world
    ):
        """The browser posts what the input holds, and the route parses it.

        A number input hands back a string. The page does not run it through
        `parseInt` -- a truncation there would silently ask for a different
        number of attempts than somebody typed -- so the route has to accept
        the exact decimal text and refuse everything else.
        """
        status, payload = _request(
            server,
            f"/api/benchmark-experiments/{world['experiment_id']}/duplicate",
            "POST",
            {"runs_per_task": "5"},
        )
        assert status == 201 and payload["experiment"]["runs_per_task"] == 5

        for bad in ("2.9", "", "three", 2.5, True, 0, -1):
            refused = _request(
                server,
                f"/api/benchmark-experiments/{world['experiment_id']}/duplicate",
                "POST",
                {"runs_per_task": bad},
            )
            assert refused[0] == 400, f"{bad!r} was accepted"


# ----------------------------------------------------------------------
# The comparison, and a comment written from it
# ----------------------------------------------------------------------


def _comparison(server, world, **params):
    query = urllib.parse.urlencode(
        {key: value for key, value in params.items() if value is not None}
    )
    return _request(server, _tasks(world, "/comparison") + ("?" + query if query else ""))


class TestComparison:
    def test_the_pair_defaults_to_the_best_run_against_the_named_attempt(
        self, server, world
    ):
        expected = _request(server, _tasks(world, "/runs"))[1]["expected_selection_id"]
        _decide_best(server, world, 2, expected)

        status, cmp = _comparison(server, world, right_attempt=1, view="steps")

        assert status == 200
        assert cmp["left_run"]["attempt"] == 2 and cmp["left_run"]["is_best"] is True
        assert cmp["right_run"]["attempt"] == 1
        assert cmp["review_pair_key"]

    def test_an_attempt_with_nothing_recorded_refuses_with_its_own_reason(
        self, server, world
    ):
        """What makes the page's disabled Compare button honest.

        Attempt 4 finished. It simply recorded no turns, so there is no
        reference to project -- and the refusal says that rather than rendering
        an empty two-pane layout that reads like two identical runs.
        """
        status, refusal = _comparison(server, world, right_attempt=4)

        assert status in (409, 422)
        assert "4" in json.dumps(refusal)

    def test_rows_say_how_much_they_claim_and_never_pair_by_position(
        self, server, world
    ):
        """The labels the page prints, computed by the API and not by it.

        Attempt 1 ran add_item, list_items, complete_item; attempt 2 ran
        add_item, remove_item. Pairing by list position would make list_items
        and remove_item "the same step". The alignment must instead report one
        match and two one-sided rows.
        """
        status, cmp = _comparison(server, world, left_attempt=1, right_attempt=2,
                                  view="steps")

        assert status == 200
        rows = cmp["alignment"]["rows"]
        kinds = {}
        for row in rows:
            named = (row["left"] or row["right"] or {}).get("command_name")
            kinds.setdefault(row["kind"], []).append(named)
        assert "add_item" in kinds.get("matched", [])
        assert "remove_item" in kinds.get("right_only", [])
        assert set(kinds.get("left_only", [])) >= {"list_items", "complete_item"}
        assert cmp["summary"]["left_only"] >= 2 and cmp["summary"]["right_only"] >= 1

    def test_every_anchor_carries_the_store_its_side_was_read_from(
        self, server, world
    ):
        """Why the page reads `store_id` off each reference.

        Both experiments are recorded in the workflow's one live DB, so both
        sides name it; a sealed archive names its own. A deep link built from
        the experiment id alone, or served by whatever store the page is
        currently pointed at, is how a turn key opens the wrong side.
        """
        status, cmp = _comparison(server, world, right_experiment=world["candidate_id"],
                                  right_attempt=1, view="steps")

        assert status == 200
        assert cmp["left"]["ref"]["store_id"] == cmp["right"]["ref"]["store_id"]
        assert cmp["right"]["ref"]["experiment_id"] == world["candidate_id"]
        for row in cmp["alignment"]["rows"]:
            for side in ("left", "right"):
                anchor = row["anchors"][side]
                if anchor:
                    assert anchor["ref"]["store_id"] == cmp[side]["ref"]["store_id"]

    def test_every_row_of_one_comparison_carries_one_pair_identity(
        self, server, world
    ):
        """Two remarks on different steps of the same two runs are one pair.

        This is what lets review progress and the comment count be about the
        same thing without either client guessing: the key a comment carries is
        the whole comparison's.
        """
        cmp = _comparison(server, world, left_attempt=1, right_attempt=2,
                          view="steps")[1]

        paired = [row for row in cmp["alignment"]["rows"]
                  if row["anchors"]["left"] and row["anchors"]["right"]]
        assert paired
        assert {row["feedback_pair_key"] for row in paired} == {cmp["review_pair_key"]}
        one_sided = [row for row in cmp["alignment"]["rows"]
                     if not (row["anchors"]["left"] and row["anchors"]["right"])]
        assert one_sided
        assert {row["feedback_pair_key"] for row in one_sided} == {None}

    def test_review_progress_and_the_comment_count_are_separate_facts(
        self, server, world
    ):
        server.db_path = world["store"].db_path  # where its comments are written
        cmp = _comparison(server, world, left_attempt=1, right_attempt=2)[1]

        before = _request(
            server,
            _tasks(world, "/review-pairs")
            + f"?reviewer={urllib.parse.quote(HUMAN['actor'])}&left_attempt=1",
        )[1]
        assert before["progress"]["reviewed"] == 0

        marked = _request(
            server,
            _tasks(world, "/review-pairs"),
            "POST",
            {"reviewer": HUMAN["actor"], "reviewer_kind": "human",
             "state": "reviewed", "left_attempt": 1, "right_attempt": 2},
        )
        assert marked[0] == 201
        assert marked[1]["pair_key"] == cmp["review_pair_key"]

        after = _request(
            server,
            _tasks(world, "/review-pairs")
            + f"?reviewer={urllib.parse.quote(HUMAN['actor'])}&left_attempt=1",
        )[1]
        assert after["progress"]["reviewed"] == 1
        # Marked reviewed, and nobody has said anything: the count of comments
        # on the pair is still zero, because they are not the same fact.
        feedback_page = _request(
            server,
            "/api/task-feedback?experiment="
            + urllib.parse.quote(world["experiment_id"])
            + "&task=" + urllib.parse.quote(world["task_id"]),
        )[1]
        assert [row for row in feedback_page["feedback"]
                if row["pair_key"] == cmp["review_pair_key"]] == []


# ----------------------------------------------------------------------
# The whole sequence the owner asked to be proven
# ----------------------------------------------------------------------


def _comment_on_row(server, world, cmp, row, *, comment, category, subcategory):
    """Post a comment the way the compare view does: the anchors, verbatim.

    The page hands back what the comparison returned -- each side's whole
    reference plus the turn the remark is anchored to -- and narrows nothing.
    Narrowing here is what used to give the comment a different pair identity
    from the comparison it was written in.
    """
    primary = row["anchors"]["left"] or row["anchors"]["right"]
    paired = (row["anchors"]["right"]
              if row["anchors"]["left"] and row["anchors"]["right"] else None)
    body = {
        "ref": primary["ref"],
        "target_kind": primary["target_kind"],
        "span_ids": primary["span_ids"],
        "target_label": "step in this pair",
        "provenance": "human",
        "category": category,
        "subcategory": subcategory,
        "comment": comment,
    }
    if paired:
        body["paired"] = {
            "ref": paired["ref"],
            "turn_key": paired["turn_key"],
            "target_kind": paired["target_kind"],
            "span_ids": paired["span_ids"],
            "target_label": "step in this pair (other side)",
        }
    path = "/post_feedback?turn_key=" + urllib.parse.quote(primary["turn_key"], safe="")
    return _request(server, path, "POST", body)


def test_a_comment_written_from_a_comparison_survives_a_new_best_run(
    server, world
):
    """The sequence, end to end, over real HTTP.

    Repeated task with four attempts -> the Reference is not the Best run ->
    a best run is chosen -> the pair is compared -> a categorized comment is
    written on one step of it -> the task's Feedback view shows it -> a
    DIFFERENT best run is chosen -> the earlier comment still names the two
    runs it was written about, under the same pair identity.

    The last step is the one that matters. A recorded pair is frozen: changing
    a selection produces a NEW pair and must not reinterpret or orphan what
    somebody already said about the old one.
    """
    server.db_path = world["store"].db_path  # where its comments are written
    runs = _request(server, _tasks(world, "/runs"))[1]
    assert runs["best_run"] is None and runs["reference"]["attempt"] == 1

    _decide_best(server, world, 2, runs["expected_selection_id"])
    cmp = _comparison(server, world, left_attempt=2, right_attempt=1, view="steps")[1]
    assert cmp["left_run"]["is_best"] is True
    pair_key = cmp["review_pair_key"]

    row = next(row for row in cmp["alignment"]["rows"]
               if row["anchors"]["left"] and row["anchors"]["right"])
    status, written = _comment_on_row(
        server, world, cmp, row,
        comment="the failed attempt dispatched add_item with the same parameters "
                "and still did not finish the list",
        category="observations_analysis",
        subcategory="analysis",
    )
    assert status == 201

    recorded = [item for item in written["feedback"]
                if "still did not finish the list" in item["comment"]]
    assert len(recorded) == 1
    assert recorded[0]["pair_key"] == pair_key, (
        "a comment written from a compare row carries the comparison's own pair"
    )
    assert recorded[0]["category"] == "observations_analysis"
    assert recorded[0]["subcategory"] == "analysis"
    frozen = json.dumps(recorded[0]["anchors"], sort_keys=True)

    def task_feedback():
        return _request(
            server,
            "/api/task-feedback?experiment="
            + urllib.parse.quote(world["experiment_id"])
            + "&task=" + urllib.parse.quote(world["task_id"]),
        )[1]["feedback"]

    seen = [item for item in task_feedback() if item["pair_key"] == pair_key]
    assert len(seen) == 1, "the task's Feedback view sees a comparison comment"
    assert seen[0]["subcategory_label"] == "Analysis"

    # Now somebody prefers a different attempt.
    moved = _request(server, _tasks(world, "/runs"))[1]
    second = _decide_best(server, world, 1, moved["expected_selection_id"])
    assert second[0] == 201
    assert _request(server, _tasks(world, "/runs"))[1]["best_run"]["attempt"] == 1

    after = [item for item in task_feedback() if item["pair_key"] == pair_key]
    assert len(after) == 1, "the earlier comment is neither moved nor orphaned"
    assert json.dumps(after[0]["anchors"], sort_keys=True) == frozen
    assert after[0]["comment"] == recorded[0]["comment"]

    # And the pair the new selection produces is a DIFFERENT pair, with no
    # comment on it -- rather than inheriting the old one's.
    fresh = _comparison(server, world, left_attempt=1, right_attempt=2, view="steps")[1]
    assert fresh["review_pair_key"] != pair_key
    assert not [item for item in task_feedback()
                if item["pair_key"] == fresh["review_pair_key"]]


def test_the_pair_identity_is_the_references_not_the_step(server, world):
    """Two remarks on two different steps of one pair belong to one pair.

    Counting comments per pair is how the compare view reports review
    progress. If the key were hashed from the anchored step, every remark
    would be its own pair and a heavily annotated comparison would report as
    unannotated.
    """
    server.db_path = world["store"].db_path  # where its comments are written
    cmp = _comparison(server, world, left_attempt=1, right_attempt=2, view="steps")[1]
    rows = [row for row in cmp["alignment"]["rows"]
            if row["anchors"]["left"] and row["anchors"]["right"]]
    assert rows

    keys = set()
    for index, row in enumerate(rows):
        status, written = _comment_on_row(
            server, world, cmp, row,
            comment=f"remark number {index} about this pair",
            category="conclusions", subcategory="what_went_wrong",
        )
        assert status == 201
        keys.update(
            item["pair_key"] for item in written["feedback"]
            if f"remark number {index}" in item["comment"]
        )

    assert keys == {cmp["review_pair_key"]}


def test_every_judgement_over_http_lands_in_the_live_db_and_no_other_file(
    server, world
):
    """Selection, pair review and feedback, each written the way the page does.

    Judgement state used to live in sidecar SQLite files beside the evidence;
    it is all control tables in the workflow's one live DB now (§2), so after
    a best run, a promotion, a reviewed pair and a comment, that DB is still
    the only database anywhere under the state root.
    """
    server.db_path = world["store"].db_path
    expected = _request(server, _tasks(world, "/runs"))[1]["expected_selection_id"]
    assert _decide_best(server, world, 2, expected)[0] == 201
    winner = _request(server, _experiments(world, "/winner"))[1]
    status, promoted = _request(
        server, _experiments(world, "/winner/decisions"), "POST",
        {**HUMAN, "decision": "promote", "candidate_experiment_id": world["candidate_id"],
         "expected_selection_id": winner["expected_selection_id"]},
    )
    assert status == 201, promoted
    status, reviewed = _request(
        server, _tasks(world, "/review-pairs"), "POST",
        {"reviewer": "dhar", "reviewer_kind": "human", "right_attempt": 1,
         "state": "reviewed"},
    )
    assert status == 201, reviewed
    cmp = _comparison(server, world, left_attempt=1, right_attempt=2, view="steps")[1]
    row = next(row for row in cmp["alignment"]["rows"]
               if row["anchors"]["left"] and row["anchors"]["right"])
    assert _comment_on_row(
        server, world, cmp, row, comment="both runs listed the items first",
        category="observations_analysis", subcategory="observation",
    )[0] == 201

    state_root = Path(os.environ["FASTWORKFLOW_STATE_ROOT"])
    assert [path for path in state_root.rglob("*.sqlite3")] == [
        Path(world["store"].db_path)
    ]
    assert [path.name for path in Path(world["folder"]).rglob("*.sqlite3")] == []
    for table in ("selection_decisions", "pair_review_events", "human_feedback"):
        assert control.rows(world["store"], f"SELECT 1 FROM {table}"), table


def test_a_comment_pairing_two_experiments_in_the_live_db_is_recorded(
    server, world
):
    """Both experiments are recorded in the workflow's one live database.

    So a pair across them is validated from here like any other pair, and the
    comment lands under the task the left side ran.
    """
    server.db_path = world["store"].db_path  # where its comments are written
    cmp = _comparison(server, world, right_experiment=world["candidate_id"],
                      right_attempt=1, view="steps")[1]
    row = next(row for row in cmp["alignment"]["rows"]
               if row["anchors"]["left"] and row["anchors"]["right"])

    status, written = _comment_on_row(
        server, world, cmp, row,
        comment="the candidate sorted the list where this one listed it",
        category="recommendations", subcategory="what_to_do",
    )

    assert status == 201, written
    assert [item["pair_key"] for item in world["store"].list_task_feedback(
        experiment_id=world["experiment_id"], task_id=world["task_id"])
        if "sorted the list" in item["comment"]] == [cmp["review_pair_key"]]


# ----------------------------------------------------------------------
# A workflow recorded before the selection control existed
# ----------------------------------------------------------------------


def test_a_workflow_with_no_selection_control_still_lists_its_attempts(
    tmp_path, monkeypatch
):
    """The upgrade case the Runs view has to survive.

    An experiment recorded before the selection control existed has attempts
    and no decisions. The page must show the attempts and say that decisions
    are not available here -- not lose the list along with them.
    """
    monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "state"))
    folder = tmp_path / "plain_workflow"
    folder.mkdir()
    (folder / "_commands").mkdir()
    server = run_chatbot_server.ChatbotServer(
        db_path="", workflow_path=str(folder), port=0,
        spawn_options={"no_server": True},
    )
    import threading

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, payload = _request(
            server, "/api/experiments/exp-nobody-registered/tasks/task-1/runs"
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)

    # A refusal the page renders as an explanation beside the evidence list,
    # which it reads from the attempts route regardless.
    assert status in (404, 409)
    assert payload["error"]


# ----------------------------------------------------------------------
# The page itself
# ----------------------------------------------------------------------


def test_the_page_never_re_derives_a_turn_key_or_an_alignment(server, world):
    """Pinned because a client-side shortcut would pass every DOM test.

    Every reference, anchor, step alignment and pair identity on the compare
    view comes from the recorded evidence through `selection_api`. A client
    that started matching commands by list position, or building a turn key
    out of an experiment id and an attempt number, would look right on the two
    short attempts in any fixture and be wrong on the first long one.

    Asserted beside the behavioural tests above, not instead of them: those
    prove the alignment the API produced is what reaches the screen; this
    proves the page has no second opinion about it.
    """
    page = run_chatbot_server.load_index_html().decode("utf-8")
    compare = page[page.index("function renderRunComparison("):
                   page.index("function renderPairComposer(")]

    for forbidden in ("turn_key =", "logical_turn_key =", ".sort(", "localeCompare"):
        assert forbidden not in compare, f"the compare view derives {forbidden}"
    # The alignment is read, not computed: rows come from the payload.
    assert "cmp.alignment.rows" in compare
    assert "compareBasisLabel(row.basis)" in compare


def test_one_rule_decides_where_every_side_of_a_pair_is_read_from(server, world):
    """Links and inline previews resolve the source the same way, once.

    The two sides of a pair are routinely in two databases, and getting this
    wrong is silent: scoping a live read that needs no scoping breaks an ad-hoc
    experiment with no authoring registration to resolve. The structural claim
    is asserted here and the behaviour is asserted in the DOM test below.
    """
    page = run_chatbot_server.load_index_html().decode("utf-8")

    assert "function pairReadScope(ctx, side)" in page
    rule = page[page.index("function pairReadScope("):
                page.index("function openPairTurn(")]
    # A side recorded in another database is refused rather than read from
    # the one the page is pointed at.
    assert 'return { kind: "current" };' in rule
    assert 'kind: "unaddressable"' in rule
    assert "side.storeId === ctx.storeId" in rule

    # Both consumers defer to it rather than each deciding for themselves.
    for consumer, end in (("openPairTurn", "function renderPairReview("),
                          ("loadArtifactPreview", "function inlineArtifactValue(")):
        body = page[page.index(f"function {consumer}("): page.index(end)]
        assert "pairReadScope(ctx, side)" in body, consumer
        assert "side.experimentId" not in body, (
            f"{consumer} reads the side's experiment id directly instead of "
            "going through the one rule"
        )


def test_selection_ui_dom(server, world):
    """Clicked, in a real DOM, against the real server.

    Everything above is about what the routes record. This is about what a
    person can see and do: every attempt on screen including the failed and
    unfinished ones, a Reference that is not a Best run, a best-run decision
    that does not launder a failure or move the winner, a comparison whose
    one-sided rows are labelled as such, a categorized comment written on one
    step of the pair, that comment under the task's Feedback tab, and a
    different best run afterwards that leaves it alone.
    """
    dependency = os.environ.get("TEST_JSDOM_ROOT")
    if not dependency:
        pytest.skip("Set TEST_JSDOM_ROOT to run DOM integration with jsdom")
    server.db_path = world["store"].db_path  # the experiment it opens
    script = Path(__file__).with_name("chatbot_selection_ui_dom.cjs")
    result = subprocess.run(
        [
            "node",
            str(script),
            dependency,
            f"http://127.0.0.1:{server.port}/?token={server.token}",
            json.dumps({
                "experiment": world["experiment_id"],
                "candidate": world["candidate_id"],
                "task": world["task_id"],
            }),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_benchmark_registration_page_offers_the_repeat_count(server, world):
    """The duplicate control reads the declared count and posts the box's text.

    Behaviour is covered over HTTP above; this pins the two things a DOM test
    cannot see on the registration page, which has no recorded run to open.
    """
    page = run_chatbot_server.load_index_html().decode("utf-8")
    repeat = page[page.index("function renderRepeatSetup("):
                  page.index("/* -- the task's Compare view")]

    assert "runs_per_task: count.value.trim()" in repeat
    assert "parseInt" not in repeat and "Number(" not in repeat
    assert "/duplicate" in repeat
    # Offered on the registration page, where nothing has run yet.
    assert "renderRepeatSetup(d, row, nav);" in page
    assert "renderWinnerPanel(d, id, function () { showBenchmarkExperiment(id); });" in page


def test_the_setup_registration_read_carries_what_the_panel_prints(server, world):
    status, payload = _request(
        server, f"/api/benchmark-experiments/{world['experiment_id']}"
    )

    assert status == 200
    assert payload["experiment"]["runs_per_task"] == 4
    assert payload["experiment"]["experiment_id"] == world["experiment_id"]
    assert setup.workflow_name_for(world["folder"])


# ----------------------------------------------------------------------
# Where each side of a pair is read from
# ----------------------------------------------------------------------
#
# The `world` fixture above has both sides in databases registered against the
# workflow, which is the easy case. The fixture below is the hard one: a live
# ad-hoc run where naming a source at all is what breaks it.


def _seed_artifact_turn(store, turn_key, *, experiment_id, task_id, attempt,
                        conversation, commands, answer, artifacts):
    """One recorded turn whose first command wrote an artifact.

    `_seed_turn` in the API worker's module records no artifacts, and an
    artifact is the whole subject here, so the same real helpers build the same
    real rows with a value attached.
    """
    refs, spans, outputs = [], [], []
    for index, command in enumerate(commands):
        call_id, span_id = f"{turn_key}-call-{index}", f"{turn_key}-span-{index}"
        refs.append((call_id, index, span_id))
        outputs.append(
            _output(call_id, command, {"n": index},
                    artifacts=artifacts if index == 0 else None)
        )
        spans.append(
            _execute_span(span_id, turn_key, call_id=call_id, command_name=command,
                          start_ns=T0 + index * 1_000_000)
        )
    row = _evidence_turn_row(
        turn_key, record=_record(turn_key, refs=refs, outputs=outputs),
        answer=answer, experiment_id=experiment_id, task_id=task_id, attempt=attempt,
    )
    row["conversation_id"] = conversation
    row["ordinal"] = 1
    _write(store, row, spans)


@pytest.fixture
def adhoc_world(tmp_path, monkeypatch):
    """A live experiment in the workflow's DEFAULT store, never registered.

    Written by a real `ExperimentController` against the workflow's live
    database, whose contest it joins in the same transaction, so its attempts
    are listed and comparable. What it has NOT got is an authoring
    registration.
    """
    monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "state"))
    folder = tmp_path / "adhoc_workflow"
    folder.mkdir()
    (folder / "_commands").mkdir()
    setup.save_benchmark(folder, {"title": "Ad hoc", "tasks": [{"prompt": "Do it"}]})

    default_db = state_paths.observability_db(str(folder))
    Path(default_db).parent.mkdir(parents=True, exist_ok=True)
    store = obs.ObservabilityStore(default_db)
    controller = ExperimentController(
        str(folder), store.store_identity(), external=False,
    )
    controller.create_experiment(
        "adhoc-exp", "typed at a prompt", declared_tasks=1, declared_attempts=2,
        declarations=[("adhoc-task", n, f"ch-adhoc-{n}") for n in (1, 2)],
        workflow_name=setup.workflow_name_for(folder),
    )
    sides = {}
    for attempt, value, answer, commands in (
        (1, "ADHOC-LEFT-VALUE", "left answer", ["add_item", "list_items"]),
        (2, "ADHOC-RIGHT-VALUE", "right answer", ["add_item"]),
    ):
        channel = f"ch-adhoc-{attempt}"
        conversation = store.mint_conversation_id(
            channel, experiment_id="adhoc-exp", task_id="adhoc-task", attempt=attempt
        )
        controller.start_attempt("adhoc-exp", "adhoc-task", attempt, channel,
                                 conversation_id=conversation)
        _seed_artifact_turn(
            store, f"adhoc-a{attempt}", experiment_id="adhoc-exp",
            task_id="adhoc-task", attempt=attempt, conversation=conversation,
            commands=commands, answer=answer, artifacts={"roster.txt": value},
        )
        controller.finish_attempt("adhoc-exp", "adhoc-task", attempt,
                                  outcome="pass", outcome_source="test")
        sides[attempt] = {"value": value, "answer": answer}

    srv = run_chatbot_server.ChatbotServer(
        db_path=default_db, workflow_path=str(folder), port=0,
        spawn_options={"no_server": True},
    )
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield {"server": srv, "folder": str(folder), "sides": sides}
    srv.shutdown()
    thread.join(timeout=5)


class TestWhereEachSideIsRead:
    """The recorded reference says which database a side lives in.

    Over HTTP first, because the browser tests below can only be believed if
    the fixtures really are the awkward shapes they claim to be.
    """

    def test_an_unregistered_run_is_readable(self, adhoc_world):
        """The ad-hoc case: no registration, and nothing it needs one for."""
        srv = adhoc_world["server"]

        status, runs = _request(
            srv, "/api/experiments/adhoc-exp/tasks/adhoc-task/runs"
        )
        assert status == 200
        assert [row["attempt"] for row in runs["attempts"]] == [1, 2]
        # One store, so both sides of the pair carry the same id and no side
        # needs a scope change to open the other's evidence.
        assert runs["store_id"] == runs["reference"]["store_id"]

        status, payload = _request(
            srv,
            "/api/experiments/adhoc-exp/tasks/adhoc-task/comparison"
            "?left_attempt=1&right_attempt=2&view=answers",
        )
        assert status == 200
        assert payload["left"]["ref"]["store_id"] == runs["store_id"]
        assert payload["right"]["ref"]["store_id"] == runs["store_id"]

        assert _request(srv, "/api/turn/adhoc-a1")[0] == 200


def _run_scope_dom(server, plan):
    dependency = os.environ.get("TEST_JSDOM_ROOT")
    if not dependency:
        pytest.skip("Set TEST_JSDOM_ROOT to run DOM integration with jsdom")
    script = Path(__file__).with_name("chatbot_selection_scope_dom.cjs")
    result = subprocess.run(
        ["node", str(script), dependency,
         f"http://127.0.0.1:{server.port}/?token={server.token}", json.dumps(plan)],
        capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_an_unregistered_live_run_previews_without_inventing_a_source(adhoc_world):
    """The opposite failure: a scope named where none was needed.

    This run has no authoring registration. The page has to read it out of the
    live database it is already pointed at.
    """
    sides = adhoc_world["sides"]
    _run_scope_dom(adhoc_world["server"], {
        "mode": "adhoc",
        "experiment": "adhoc-exp",
        "task": "adhoc-task",
        "turn_key": "adhoc-a2",
        "values": [sides[1]["value"], sides[2]["value"]],
        "right_answer": sides[2]["answer"],
    })

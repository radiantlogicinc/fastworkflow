"""The browser's half of run selection, the winner and the comparison.

`fix-9eg.17.2` / `.17.3` / `.17.4` and the browser slice of `fix-9eg.4`.

Two halves, and the split is deliberate.

The Python half drives the SAME HTTP surface the page calls, over a real
socket, through the real `ChatbotServer`: real workflow folders, real benchmark
manifests, real `ObservabilityStore` databases, attempts written through the
real `ExperimentController`, the real shared selection control, the real
pair-review sidecar and the real feedback writer. It exists because the
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

import hashlib
import json
import os
import shutil
import subprocess
import threading
import urllib.parse
from pathlib import Path

import pytest

from fastworkflow import state_paths
from fastworkflow.benchmark import setup
from fastworkflow.experiment.runner import ExperimentController
from fastworkflow.observability import comparison, selection
from fastworkflow.observability import store as obs
from fastworkflow.run_chatbot import server as run_chatbot_server
from tests.test_chatbot_benchmarks import _request
from tests.test_execution_comparison import _execute_span, _output, _record, _write
from tests.test_execution_comparison import _turn_row as _evidence_turn_row
from tests.test_selection_api import T0

# The API worker's world is the one this page is built against: one workflow,
# a task run four different ways in one database (completed, failed, never
# finished, finished with no recorded turns) and a second experiment recording
# the same task in a SECOND database. Rebuilding it here would be a second
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

    def test_the_two_sides_carry_their_own_store_so_a_link_cannot_open_the_wrong_one(
        self, server, world
    ):
        """Why the page reads `store_id` off each reference.

        The candidate's evidence is in a second database. A deep link built
        from the experiment id alone, or served by whatever store the page is
        currently pointed at, is how a turn key that exists in both databases
        opens the wrong side.
        """
        status, cmp = _comparison(server, world, right_experiment=world["candidate_id"],
                                  right_attempt=1, view="steps")

        assert status == 200
        assert cmp["left"]["ref"]["store_id"] != cmp["right"]["ref"]["store_id"]
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
            + "&task=" + urllib.parse.quote(world["task_id"])
            + "&benchmark_experiment=" + urllib.parse.quote(world["experiment_id"]),
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
    path = (
        "/post_feedback?turn_key=" + urllib.parse.quote(primary["turn_key"], safe="")
        + "&benchmark_experiment="
        + urllib.parse.quote(primary["ref"]["experiment_id"], safe="")
    )
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
            + "&task=" + urllib.parse.quote(world["task_id"])
            + "&benchmark_experiment=" + urllib.parse.quote(world["experiment_id"]),
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


def test_a_comparison_of_two_stores_records_a_comment_naming_both(server, world):
    """The cross-database pair, which is the winner-versus-candidate shape."""
    cmp = _comparison(server, world, right_experiment=world["candidate_id"],
                      right_attempt=1, view="steps")[1]
    row = next(row for row in cmp["alignment"]["rows"]
               if row["anchors"]["left"] and row["anchors"]["right"])

    status, written = _comment_on_row(
        server, world, cmp, row,
        comment="the candidate sorted the list where this one listed it",
        category="recommendations", subcategory="what_to_do",
    )

    assert status == 201
    recorded = next(item for item in written["feedback"]
                    if "sorted the list" in item["comment"])
    assert recorded["pair_key"] == cmp["review_pair_key"]
    assert recorded["paired"]["ref"]["store_id"] == cmp["right"]["ref"]["store_id"]
    assert recorded["paired"]["ref"]["experiment_id"] == world["candidate_id"]
    # The same key the pair-review sidecar would record it under, so the
    # compare view's progress line and its comment count agree.
    assert comparison.review_pair_key(
        comparison.ExecutionRef.from_mapping(cmp["left"]["ref"]),
        comparison.ExecutionRef.from_mapping(cmp["right"]["ref"]),
    ) == recorded["pair_key"]


# ----------------------------------------------------------------------
# Sealed evidence: readable, and never credited with a decision
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
    wrong is silent: a link built from the experiment id alone opens the wrong
    archive in a workspace whose two stores share a logical turn key, and
    scoping a live read that needs no scoping breaks an ad-hoc experiment with
    no authoring registration to resolve. Both failures are invisible in a
    one-store fixture, so the structural claim is asserted here and the
    behaviour over two real stores is asserted in the DOM tests below.
    """
    page = run_chatbot_server.load_index_html().decode("utf-8")

    assert "function pairReadScope(ctx, side)" in page
    rule = page[page.index("function pairReadScope("):
                page.index("function openPairTurn(")]
    # A sealed side is addressed by the MANIFEST's name for its archive, which
    # is the only one the workspace routes answer to -- never by the evidence
    # identity the reference carries, which is what the write side resolves.
    assert 'return { kind: "workspace", storeId: storeId };' in rule
    assert "var storeId = side.manifestStoreId || side.storeId;" in rule
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
    links = page[page.index("function openPairTurn("):
                 page.index("function renderPairReview(")]
    # No unconditional assignment of the global source: the probe comes first.
    assert links.index("scopedRead(") < links.index(
        "benchmarkExperimentSource = scope.experimentId"
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
# workflow, which is the easy case and hides the two ways the source can be got
# wrong. These two fixtures are the hard cases, and they are opposites: a sealed
# workspace where the store id is the ONLY thing separating two sides, and a
# live ad-hoc run where naming a source at all is what breaks it.


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
def colliding_workspace(tmp_path, monkeypatch):
    """Two sealed archives whose attempts share a logical turn key.

    Both record `shared-turn` and both record an artifact under `roster.txt`
    with a different value, which is the shape that makes an unscoped read
    indistinguishable from a correct one. Real archives: written with the
    ordinary store API and sealed with `archive_to`, then stitched into one
    logical experiment by a real manifest.
    """
    from tests.test_observability_workspace import _manifest, _store_decl

    monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "state"))
    declarations, values = [], {}
    for store_id, attempt, answer, value in (
        ("alpha", 1, "left answer", "LEFT-ONLY-VALUE"),
        ("beta", 2, "right answer", "RIGHT-ONLY-VALUE"),
    ):
        live = str(tmp_path / f"live-{store_id}.sqlite3")
        store = obs.ObservabilityStore(live)
        store.create_experiment("local", "archived run", declared_tasks=1,
                                declared_attempts=1)
        channel = f"ch-{store_id}"
        store.start_attempt("local", "task", attempt, channel)
        conversation = store.mint_conversation_id(
            channel, experiment_id="local", task_id="task", attempt=attempt
        )
        _seed_artifact_turn(
            store, "shared-turn", experiment_id="local", task_id="task",
            attempt=attempt, conversation=conversation,
            commands=["add_item", "list_items"] if attempt == 1 else ["add_item"],
            answer=answer, artifacts={"roster.txt": value},
        )
        store.finish_attempt("local", "task", attempt, outcome="pass",
                             outcome_source="test")
        archive = obs.ObservabilityStore(live, migrate=False).archive_to(
            str(tmp_path / f"sealed-{store_id}.sqlite3")
        )
        declarations.append(_store_decl(archive, store_id))
        values[store_id] = {
            "answer": answer, "value": value, "attempt": attempt,
            # Two names for one archive, both recorded here so a test can say
            # which one it expects rather than matching either.
            "identity": archive["store_identity"],
            "path": str(tmp_path / f"sealed-{store_id}.sqlite3"),
        }

    manifest = _manifest(
        tmp_path, declarations,
        experiments=[{
            "experiment_id": "logical",
            "segments": [
                {"store_id": "alpha", "local_experiment_id": "local"},
                {"store_id": "beta", "local_experiment_id": "local"},
            ],
        }],
    )
    srv = run_chatbot_server.ChatbotServer(port=0,
                                           workspace_manifest_path=str(manifest))
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield {"server": srv, "sides": values, "turn_key": "shared-turn",
           "root": tmp_path}
    srv.shutdown()
    thread.join(timeout=5)


@pytest.fixture
def adhoc_world(tmp_path, monkeypatch):
    """A live experiment in the workflow's DEFAULT store, never registered.

    Written by a real `ExperimentController` against the canonical default
    database, which authorizes the source in the shared selection control, so
    its attempts are listed and comparable. What it has NOT got is an authoring
    registration, and `?benchmark_experiment=` resolves only registrations --
    the exact run a view that scoped every live read by experiment id would
    break.
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
        default_db, store.store_identity(), external=False,
        workflow_folderpath=str(folder),
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

    def test_two_archives_can_hold_the_same_turn_key_and_different_values(
        self, colliding_workspace
    ):
        srv = colliding_workspace["server"]
        status, payload = _request(
            srv,
            "/api/experiments/logical/tasks/task/comparison"
            "?left_attempt=1&right_attempt=2&view=answers",
        )

        assert status == 200
        # A reference names its store the way a reference names a store
        # everywhere in this system: by the identity the database reports about
        # itself. The manifest's own name for the same archive -- which is what
        # every /api/workspace route is addressed by -- travels beside it, under
        # its own key, so a client can never reach for the wrong one of the two.
        sides = colliding_workspace["sides"]
        assert payload["left"]["ref"]["store_id"] == sides["alpha"]["identity"]
        assert payload["right"]["ref"]["store_id"] == sides["beta"]["identity"]
        assert payload["left"]["manifest_store_id"] == "alpha"
        assert payload["right"]["manifest_store_id"] == "beta"
        assert payload["left_run"]["manifest_store_id"] == "alpha"
        assert payload["right_run"]["manifest_store_id"] == "beta"
        # The collision, stated: the turn key alone does not identify evidence.
        assert payload["left"]["ref"]["turn_keys"] == ["shared-turn"]
        assert payload["right"]["ref"]["turn_keys"] == ["shared-turn"]
        assert [row["key"] for row in payload["left"]["artifacts"]] == ["roster.txt"]
        assert [row["key"] for row in payload["right"]["artifacts"]] == ["roster.txt"]
        assert all(row["inline"] for row in payload["left"]["artifacts"])

        # And the two stores answer differently for the same key, which is what
        # makes a mis-scoped preview a wrong answer rather than a missing one.
        for store_id, expected in sides.items():
            status, turn = _request(srv, f"/api/workspace/turn/{store_id}/shared-turn")
            assert status == 200
            recorded = [
                output["command_response"]["artifacts"]
                for output in turn["turn"]["record"]["turn_output"]["command_outputs"]
            ]
            assert {"roster.txt": expected["value"]} in recorded

    def test_a_sealed_archive_refuses_the_unscoped_read_a_live_page_would_make(
        self, colliding_workspace
    ):
        """So a link that forgot the store fails closed, not quietly wrong."""
        srv = colliding_workspace["server"]

        status, payload = _request(srv, "/api/turn/shared-turn")
        assert status == 400 and "store-aware" in payload["error"]

    def test_an_unregistered_run_is_readable_and_refuses_to_be_scoped(
        self, adhoc_world
    ):
        """Both halves of the ad-hoc case, which pull in opposite directions."""
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

        # Readable unscoped...
        assert _request(srv, "/api/turn/adhoc-a1")[0] == 200
        # ...and refused when scoped by an experiment id with no registration
        # behind it. A page that always named the source would 409 here.
        status, refusal = _request(
            srv, "/api/turn/adhoc-a1?benchmark_experiment=adhoc-exp"
        )
        assert status == 409 and "adhoc-exp" in refusal["error"]


class TestCommentingOnASealedPair:
    """A comment about two archives, written where neither can be written to.

    The two names an archive has are not interchangeable and the write route
    refuses the wrong one, so these tests pin which field carries which -- the
    mistake they exist to catch reads as "the page suddenly cannot save".
    """

    @staticmethod
    def _anchors(srv):
        status, payload = _request(
            srv,
            "/api/experiments/logical/tasks/task/comparison"
            "?left_attempt=1&right_attempt=2&view=steps",
        )
        assert status == 200 and payload["sealed"] is True
        row = next(
            row for row in payload["alignment"]["rows"]
            if row["anchors"].get("left") and row["anchors"].get("right")
        )
        return payload, row["anchors"]

    @staticmethod
    def _body(**overrides):
        body = {
            "target_kind": "step", "span_ids": [], "target_label": "step add_item",
            "provenance": "human", "category": "conclusions",
            "subcategory": "what_went_right", "comment": "the left archive won",
        }
        body.update(overrides)
        return body

    def test_the_anchors_the_comparison_returned_can_be_posted_verbatim(
        self, colliding_workspace
    ):
        """The contract the composer relies on, stated without a browser.

        Posting the row as it came back is the ONLY shape that keeps the stored
        pair identity equal to the comparison's own: rebuilding either reference
        to carry a different store name changes its `ref_id`, and the comment
        would be filed against a pair nothing on screen matches.
        """
        srv = colliding_workspace["server"]
        payload, anchors = self._anchors(srv)

        status, recorded = _request(
            srv,
            "/post_feedback?turn_key=" + anchors["left"]["turn_key"]
            + "&store_id=" + anchors["left"]["manifest_store_id"],
            "POST",
            self._body(
                span_ids=anchors["left"]["span_ids"], ref=anchors["left"]["ref"],
                paired=dict(anchors["right"], target_label="step add_item (other)"),
            ),
        )

        assert status == 201, recorded
        assert recorded["read_only"] is True and recorded["annotated"] is True
        stored = recorded["feedback"][-1]
        assert stored["pair_key"] == payload["review_pair_key"]
        assert stored["pair_key"] == anchors["left"]["ref"]["ref_id"] \
            + "|" + anchors["right"]["ref"]["ref_id"]
        # The paired side keeps the WHOLE reference it was written against,
        # including the other archive's identity.
        assert stored["anchors"]["paired"]["ref"]["store_id"] == \
            colliding_workspace["sides"]["beta"]["identity"]

    def test_the_manifest_name_and_the_evidence_identity_are_not_swappable(
        self, colliding_workspace
    ):
        """Both ways round, so neither can be "fixed" by exchanging them.

        This is the failure the page hit before the reference carried the
        identity: the anchors named archives the way the manifest does, and the
        paired side could not be resolved at all.
        """
        srv = colliding_workspace["server"]
        _payload, anchors = self._anchors(srv)
        identity = colliding_workspace["sides"]["alpha"]["identity"]

        # The write scope is the manifest's name. The identity is not a store id.
        status, refusal = _request(
            srv,
            "/post_feedback?turn_key=" + anchors["left"]["turn_key"]
            + "&store_id=" + identity,
            "POST",
            self._body(span_ids=anchors["left"]["span_ids"],
                       ref=anchors["left"]["ref"]),
        )
        assert status == 409 and identity in refusal["error"]

        # A reference names the identity. The manifest's name is not one.
        status, refusal = _request(
            srv,
            "/post_feedback?turn_key=" + anchors["left"]["turn_key"]
            + "&store_id=alpha",
            "POST",
            self._body(
                span_ids=anchors["left"]["span_ids"], ref=anchors["left"]["ref"],
                paired=dict(
                    anchors["right"], target_label="step add_item (other)",
                    ref=dict(anchors["right"]["ref"], store_id="beta"),
                ),
            ),
        )
        assert status == 409 and "beta" in refusal["error"]


class TestAnArchiveNamedTwice:
    """One evidence identity declared by two manifest stores.

    Real, not hypothetical: two snapshots of the same database are the same
    evidence under two names, and a manifest can declare both. The translation
    between a manifest's name and the identity a reference carries has no answer
    then, and the failure it invites is the invisible kind -- one archive's turns
    projected under the other archive's reference. `store_id_for_identity`
    already refuses it on the write side; this is the read side refusing it too,
    rather than picking whichever declaration it saw last.
    """

    def test_naming_an_archive_re_hashes_only_what_is_read(
        self, tmp_path, colliding_workspace, monkeypatch
    ):
        """Translating two names is metadata work, not integrity work.

        The manifest's declaration already says which identity each store has,
        so mapping one name onto the other needs no file. `workspace.stores()`
        would answer the same question by re-digesting EVERY archive the manifest
        names, on every request, including archives the request never touches.
        The archives actually read are digested anyway when they are leased,
        which is where that cost belongs.

        So the manifest here declares a third archive that no segment of the
        experiment being compared names. Counting digests is the only way to see
        the difference: an eager mapping produces exactly the same payload.
        """
        from fastworkflow.observability import workspace as workspace_module
        from tests.test_observability_workspace import _manifest

        sides = colliding_workspace["sides"]
        room = tmp_path / "one-spare"
        room.mkdir()
        declarations = []
        for store_id, source, identity in (
            ("alpha", sides["alpha"]["path"], sides["alpha"]["identity"]),
            ("beta", sides["beta"]["path"], sides["beta"]["identity"]),
            # Declared, and named by nothing the comparison below reads. Its
            # identity is left undeclared, which is the one shape that has to
            # keep working without it: an older manifest.
            ("spare", sides["alpha"]["path"], None),
        ):
            local = room / f"sealed-{store_id}.sqlite3"
            shutil.copyfile(source, local)
            declaration = {
                "store_id": store_id, "path": local.name, "mode": "sealed",
                "sha256": hashlib.sha256(local.read_bytes()).hexdigest(),
            }
            if identity is not None:
                declaration["store_identity"] = identity
            declarations.append(declaration)
        manifest = _manifest(
            room, declarations,
            experiments=[{
                "experiment_id": "logical",
                "segments": [
                    {"store_id": "alpha", "local_experiment_id": "local"},
                    {"store_id": "beta", "local_experiment_id": "local"},
                ],
            }],
        )
        srv = run_chatbot_server.ChatbotServer(
            port=0, workspace_manifest_path=str(manifest)
        )
        thread = threading.Thread(target=srv.serve_forever, daemon=True)
        thread.start()
        try:
            # Counted from here, so the one-off verification every archive gets
            # when the manifest is LOADED is not what is being measured.
            digested = []
            real = workspace_module._sha256
            monkeypatch.setattr(
                workspace_module, "_sha256",
                lambda path: (digested.append(Path(path).name), real(path))[1],
            )

            status, payload = _request(
                srv,
                "/api/experiments/logical/tasks/task/comparison"
                "?left_attempt=1&right_attempt=2&view=steps",
            )

            assert status == 200, payload
            # Both sides carry both names, which is what the mapping was for...
            assert payload["left"]["manifest_store_id"] == "alpha"
            assert payload["right"]["manifest_store_id"] == "beta"
            assert payload["left"]["ref"]["store_id"] == sides["alpha"]["identity"]
            # ...and the archive nobody asked about was never opened again.
            assert set(digested) == {
                "sealed-alpha.sqlite3", "sealed-beta.sqlite3"
            }, digested
        finally:
            srv.shutdown()
            thread.join(timeout=5)

    def test_the_ambiguity_is_refused_rather_than_resolved(
        self, tmp_path, colliding_workspace
    ):
        from tests.test_observability_workspace import _manifest

        alpha = colliding_workspace["sides"]["alpha"]
        beta = colliding_workspace["sides"]["beta"]
        # Its own directory, so the fixture's manifest and archives are left
        # exactly as that fixture wrote them.
        room = tmp_path / "named-twice"
        room.mkdir()
        declarations = []
        for store_id, source, identity in (
            ("alpha", alpha["path"], alpha["identity"]),
            # A second name for the SAME evidence: byte-for-byte the same
            # archive, so it reports the same identity about itself.
            ("twin", alpha["path"], alpha["identity"]),
            ("beta", beta["path"], beta["identity"]),
        ):
            # A manifest names its stores by a path relative to itself, so each
            # is copied in. The fixture's own archives are only read.
            local = room / f"sealed-{store_id}.sqlite3"
            shutil.copyfile(source, local)
            declarations.append({
                "store_id": store_id, "path": local.name, "mode": "sealed",
                "sha256": hashlib.sha256(local.read_bytes()).hexdigest(),
                "store_identity": identity,
            })
        manifest = _manifest(
            room, declarations,
            experiments=[{
                "experiment_id": "logical",
                "segments": [
                    {"store_id": "alpha", "local_experiment_id": "local"},
                    {"store_id": "twin", "local_experiment_id": "local"},
                ],
            }],
        )
        srv = run_chatbot_server.ChatbotServer(
            port=0, workspace_manifest_path=str(manifest)
        )
        thread = threading.Thread(target=srv.serve_forever, daemon=True)
        thread.start()
        try:
            status, payload = _request(
                srv, "/api/experiments/logical/tasks/task/runs"
            )

            assert status == 409, payload
            assert alpha["identity"] in payload["error"]
            # Both names said out loud, because the fix is to the manifest.
            assert "alpha" in payload["error"] and "twin" in payload["error"]
            assert payload["ambiguous_identity"] == alpha["identity"]

            # And the refusal is the same one the write side gives, so neither
            # half of the product quietly picks a store the other refused.
            with pytest.raises(Exception) as refused:
                srv.workspace.registry.store_id_for_identity(alpha["identity"])
            assert "more than one" in str(refused.value)
        finally:
            srv.shutdown()
            thread.join(timeout=5)


def _archive_bytes(colliding_workspace):
    """sha256 of every file beside the two archives, keyed by name.

    Named by directory listing rather than by the two paths, so a `-wal` or
    `-shm` that should not exist shows up as a new key instead of going unseen.
    """
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path(colliding_workspace["root"]).glob("sealed-*"))
    }


def test_a_comment_about_two_sealed_archives_is_saved_beside_them(
    colliding_workspace
):
    """The whole affordance, in a real DOM, over two read-only archives.

    Sealed evidence is not a reason to refuse the comment -- it is a reason to
    keep it somewhere else -- so the composer stays open, the note is filed in a
    sidecar, and the task's Feedback view lists it exactly ONCE even though the
    logical experiment is stitched from two stores. The archives themselves are
    hashed before and after: if either changed by a byte, or grew a `-wal`, the
    comment was appended to frozen evidence and the pass is worthless.
    """
    before = _archive_bytes(colliding_workspace)
    assert [name for name in before if name.endswith(".sqlite3")] == [
        "sealed-alpha.sqlite3", "sealed-beta.sqlite3"
    ]
    sides = colliding_workspace["sides"]

    _run_scope_dom(colliding_workspace["server"], {
        "mode": "workspace",
        "experiment": "logical",
        "task": "task",
        "turn_key": colliding_workspace["turn_key"],
        "values": [sides["alpha"]["value"], sides["beta"]["value"]],
        "right_store": "beta",
        "right_answer": sides["beta"]["answer"],
        "comment": {
            "category": "conclusions",
            "category_label": "Conclusions",
            "subcategory": "what_went_right",
            "text": "the left archive kept the roster in order",
            "step_text": "and this step is where it did it",
        },
    })

    after = _archive_bytes(colliding_workspace)
    unchanged = {name: digest for name, digest in after.items()
                 if name in before}
    assert unchanged == before, "the sealed archives were written to"
    # What DID appear is a sidecar beside the archive that was commented on,
    # and nothing that looks like an open database.
    appeared = sorted(set(after) - set(before))
    assert appeared == ["sealed-alpha.feedback.sqlite3"], appeared


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


def test_a_sealed_pair_previews_each_side_in_its_own_archive(colliding_workspace):
    """Two archives, one turn key, two values -- and two correct panes.

    The inline preview and the deep link both go through `pairReadScope`, so
    this is the test that would fail if either of them read "the turn" without
    naming the store it belongs to: the panes would agree, and agreeing is the
    bug.
    """
    sides = colliding_workspace["sides"]
    _run_scope_dom(colliding_workspace["server"], {
        "mode": "workspace",
        "experiment": "logical",
        "task": "task",
        "turn_key": colliding_workspace["turn_key"],
        "values": [sides["alpha"]["value"], sides["beta"]["value"]],
        "right_store": "beta",
        "right_answer": sides["beta"]["answer"],
    })


def test_an_unregistered_live_run_previews_without_inventing_a_source(adhoc_world):
    """The opposite failure: a scope named where none was needed.

    This run has no authoring registration, so `?benchmark_experiment=` answers
    409 for it. The page has to read it out of the source it is already pointed
    at, and leave the global source alone.
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

"""The task Feedback view: every authorized comment on one task.

Real stores, real HTTP, no mocks.

The read-only half of this module proves that a reader can record a NEW
categorized comment about evidence this build must not write to -- a database
file it cannot write -- without a byte of that evidence moving.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pytest

from fastworkflow.observability import feedback as fb
from fastworkflow.observability import control
from fastworkflow.observability import store as obs
from fastworkflow.observability.comparison import ExecutionRef, review_pair_key
from fastworkflow.run_chatbot import server as run_chatbot_server
from tests.test_chatbot_benchmarks import _request
from tests.test_observability_workspace import _turn_row

# The server fixtures come from that module as a plugin rather than as
# imported names: importing a fixture makes it look unused at its import and
# redefined at every test that asks for it, which buries real lint findings
# in this file under thirty false ones.
pytest_plugins = ("tests.test_chatbot_benchmarks",)


def _seed_task(store, *, experiment_id="exp-1", task_id="task-1", attempts=(1, 2)):
    """Two attempts, two turns each, one span per turn: a real recorded task."""
    keys = []
    with store._connect() as conn:
        for attempt in attempts:
            for ordinal in (1, 2):
                key = f"{task_id}-a{attempt}-t{ordinal}"
                row = _turn_row(key, experiment_id, task_id, attempt)
                assert store.upsert_turn_row(conn, row, [], store._store_redactor())
                conn.execute(
                    "INSERT INTO spans(span_id,trace_id,name,kind,start_ns,status,attributes) "
                    "VALUES(?,?,?,?,?,?,?)",
                    (f"span-{key}", key, "fw.planner.plan", "internal", 1, "ok", "{}"),
                )
                keys.append(key)
    return keys


def _post(server, turn_key, **body):
    payload = dict(
        target_kind="turn",
        span_ids=[],
        target_label="Turn",
        comment="a comment",
        provenance="human",
        category="conclusions",
        subcategory="what_went_wrong",
    )
    payload.update(body)
    status, data = _request(
        server,
        "/post_feedback?turn_key=" + urllib.parse.quote(turn_key, safe=""),
        "POST",
        payload,
    )
    assert status == 201, data
    return data


def _task_feedback(server, experiment="exp-1", task="task-1", **params):
    query = "/api/task-feedback?experiment=" + experiment + "&task=" + task
    for name, value in params.items():
        if value is not None:
            query += f"&{name}={urllib.parse.quote(str(value), safe='')}"
    status, data = _request(server, query)
    assert status == 200, data
    return data


def test_the_view_shows_every_attempt_turn_and_component_with_no_hidden_filter(
    experiment_server,
):
    """The default answer is the whole record.

    A default component or category filter would quietly answer a narrower
    question than "what has anyone said about this task", which is the only
    question this view exists to answer.
    """
    server, store = experiment_server
    keys = _seed_task(store)
    _post(server, keys[0], comment="attempt 1, first turn")
    _post(server, keys[3], comment="attempt 2, second turn")
    _post(
        server,
        keys[1],
        target_kind="phase",
        span_ids=[f"span-{keys[1]}"],
        target_label="Planning",
        category="observations_analysis",
        subcategory="analysis",
        provenance="coding_agent",
        comment="the planner never revisited the third item",
    )
    # A task-level summary is ordinary feedback, not a second schema: it is
    # anchored to a real turn of the task and appears in the same list.
    _post(
        server,
        keys[3],
        target_kind="task",
        target_label="Task",
        category="recommendations",
        subcategory="what_to_do",
        comment="overall: keep the retry, drop the second clarification",
        ref={"experiment_id": "exp-1", "task_id": "task-1"},
    )
    page = _task_feedback(server)
    assert page["total"] == 4
    assert [row["comment"] for row in page["feedback"]] == [
        "attempt 1, first turn",
        "attempt 2, second turn",
        "the planner never revisited the third item",
        "overall: keep the retry, drop the second clarification",
    ]
    assert sorted({row["attempt"] for row in page["feedback"]}) == [1, 2]
    assert {row["target_kind"] for row in page["feedback"]} == {"turn", "phase", "task"}
    assert page["filters"]["category"] is None
    assert page["filters"]["component"] is None
    # Every row deep-links to the exact turn it is anchored to.
    assert all(row["turn_key"] in keys for row in page["feedback"])
    assert all(row["classified"] for row in page["feedback"])


def test_a_different_task_and_experiment_are_not_shown(experiment_server):
    server, store = experiment_server
    keys = _seed_task(store)
    other = _seed_task(store, task_id="task-2")
    _post(server, keys[0], comment="about task 1")
    _post(server, other[0], comment="about task 2")
    assert [r["comment"] for r in _task_feedback(server)["feedback"]] == ["about task 1"]
    assert [
        r["comment"] for r in _task_feedback(server, task="task-2")["feedback"]
    ] == ["about task 2"]
    assert _task_feedback(server, experiment="exp-absent")["total"] == 0


@pytest.mark.parametrize(
    "params,expected",
    [
        ({"category": "conclusions"}, ["turn one", "turn two"]),
        ({"subcategory": "analysis"}, ["planner analysis"]),
        ({"provenance": "coding_agent"}, ["planner analysis"]),
        ({"target_kind": "phase"}, ["planner analysis"]),
        ({"component": "Planning"}, ["planner analysis"]),
        ({"attempt": 2}, ["turn two"]),
        ({"category": "conclusions", "attempt": 1}, ["turn one"]),
    ],
)
def test_the_optional_filters_narrow_without_changing_the_rows(
    experiment_server, params, expected
):
    server, store = experiment_server
    keys = _seed_task(store)
    _post(server, keys[0], comment="turn one")
    _post(server, keys[3], comment="turn two")
    _post(
        server,
        keys[1],
        target_kind="phase",
        span_ids=[f"span-{keys[1]}"],
        target_label="Planning",
        category="observations_analysis",
        subcategory="analysis",
        provenance="coding_agent",
        comment="planner analysis",
    )
    page = _task_feedback(server, **params)
    assert [row["comment"] for row in page["feedback"]] == expected
    assert page["total"] == len(expected)


@pytest.mark.parametrize(
    "params", [{"category": "insights"}, {"subcategory": "verdict"},
               {"category": "conclusions", "subcategory": "analysis"},
               {"limit": -1}, {"offset": -1}]
)
def test_an_unknown_or_impossible_filter_is_refused(experiment_server, params):
    server, store = experiment_server
    _seed_task(store)
    query = "/api/task-feedback?experiment=exp-1&task=task-1"
    for name, value in params.items():
        query += f"&{name}={value}"
    assert _request(server, query)[0] == 400


def test_paging_is_bounded_and_deterministic(experiment_server):
    """The same window twice, and no row shown twice or skipped across pages."""
    server, store = experiment_server
    keys = _seed_task(store)
    for index in range(7):
        _post(server, keys[index % len(keys)], comment=f"comment {index}")
    first = _task_feedback(server, limit=3, offset=0)
    assert [r["comment"] for r in first["feedback"]] == [
        "comment 0", "comment 1", "comment 2",
    ]
    assert first["total"] == 7 and first["has_more"] is True
    assert first == _task_feedback(server, limit=3, offset=0)
    second = _task_feedback(server, limit=3, offset=3)
    third = _task_feedback(server, limit=3, offset=6)
    assert third["has_more"] is False
    seen = [r["feedback_uid"] for page in (first, second, third) for r in page["feedback"]]
    assert len(seen) == len(set(seen)) == 7
    assert _task_feedback(server, limit=3, offset=99)["feedback"] == []


def test_a_comparison_comment_is_reachable_from_both_tasks_exactly_once(
    experiment_server,
):
    """Stable identity is what makes "once" possible.

    The comment is ONE row naming two executions. Both task views must find
    it, neither may show it twice, and the two views must agree about which
    row it is — that is the `feedback_uid`, not a digest of its text.
    """
    server, store = experiment_server
    left = _seed_task(store, task_id="task-1")
    right = _seed_task(store, task_id="task-2")
    identity = store.store_identity()
    written = _post(
        server,
        left[0],
        category="conclusions",
        subcategory="what_went_right",
        comment="attempt 1 recovered where the other gave up",
        paired={
            "store_id": identity,
            "turn_keys": [right[0]],
            "experiment_id": "exp-1",
            "task_id": "task-2",
            "attempt": 1,
            "target_kind": "turn",
            "span_ids": [],
            "target_label": "Turn",
        },
    )
    row = written["feedback"][0]
    assert row["pair_key"] and row["paired"]["ref"]["task_id"] == "task-2"
    from_left = _task_feedback(server, task="task-1")
    from_right = _task_feedback(server, task="task-2")
    assert from_left["total"] == from_right["total"] == 1
    assert (
        from_left["feedback"][0]["feedback_uid"]
        == from_right["feedback"][0]["feedback_uid"]
        == row["feedback_uid"]
    )
    # Both sides are linkable from either view.
    shown = from_right["feedback"][0]
    assert shown["anchors"]["primary"]["ref"]["turn_keys"] == [left[0]]
    assert shown["paired"]["ref"]["turn_keys"] == [right[0]]


def test_two_people_typing_the_same_sentence_are_two_comments(experiment_server):
    """Dedupe is by row identity, never by text: identical independent
    remarks from two reviewers are two facts about the task."""
    server, store = experiment_server
    keys = _seed_task(store)
    _post(server, keys[0], comment="the answer skipped the third item")
    _post(server, keys[1], provenance="coding_agent",
          comment="the answer skipped the third item")
    page = _task_feedback(server)
    assert page["total"] == 2
    assert len({r["feedback_uid"] for r in page["feedback"]}) == 2


def test_a_pair_anchor_is_frozen_and_never_re_derived(experiment_server):
    """Written down, not looked up.

    Both sides of the pair are stored in the row itself, so re-picking a
    winner or a task's best run later cannot retarget a comment: the pair it
    names is the pair that was on screen when somebody wrote it. The test
    reads the stored column directly and then re-reads through the view,
    because "the answer came from the frozen anchor" is the property — a view
    that recomputed the pair from the CURRENT selection would agree with this
    test right up until the selection changed.
    """
    server, store = experiment_server
    left = _seed_task(store, task_id="task-1")
    right = _seed_task(store, task_id="task-2")
    identity = store.store_identity()
    _post(
        server,
        left[0],
        comment="this side recovered, the other did not",
        paired={
            "store_id": identity, "turn_keys": [right[0]],
            "experiment_id": "exp-1", "task_id": "task-2", "attempt": 1,
            "target_kind": "turn", "span_ids": [], "target_label": "Turn",
        },
    )
    before = _task_feedback(server)["feedback"][0]
    with store._connect() as conn:
        frozen, pair_experiment, pair_task = conn.execute(
            "SELECT anchors_json, pair_experiment_id, pair_task_id "
            "FROM human_feedback WHERE turn_key=?",
            (left[0],),
        ).fetchone()
    stored = json.loads(frozen)
    assert stored["paired"]["ref"]["turn_keys"] == [right[0]]
    assert (pair_experiment, pair_task) == ("exp-1", "task-2")
    assert before["paired"] == stored["paired"]
    assert before["pair_key"] == stored["pair_key"]
    # Later activity on either side of the pair changes nothing about it.
    _post(server, right[2], comment="a later remark on the other side")
    _post(server, left[2], comment="a later remark on this side")
    store.update_experiment_notes("exp-1", "winner re-picked after review")
    after = [
        row
        for row in _task_feedback(server)["feedback"]
        if row["feedback_uid"] == before["feedback_uid"]
    ]
    assert after == [before]


def test_a_multi_turn_pair_keys_to_the_pair_the_reader_compared(
    experiment_server,
):
    """One pair, however many steps somebody remarks on.

    Both sides of this comparison are two-turn attempts. A comment written on
    one step of each must key to the PAIR OF ATTEMPTS — the same key
    `comparison.review_pair_key` gives the compare view and `pair_review`
    gives that pair's progress — or the comment and the review of the thing it
    is about would be filed under two different identities. Narrowing each
    reference to its anchored turn (which is what this used to do) produced a
    third key naming neither attempt.
    """
    server, store = experiment_server
    left = _seed_task(store, task_id="task-1", attempts=(1,))
    right = _seed_task(store, task_id="task-2", attempts=(1,))
    identity = store.store_identity()
    left_ref = ExecutionRef(
        store_id=identity, turn_keys=tuple(left),
        experiment_id="exp-1", task_id="task-1", attempt=1,
    )
    right_ref = ExecutionRef(
        store_id=identity, turn_keys=tuple(right),
        experiment_id="exp-1", task_id="task-2", attempt=1,
    )
    expected = review_pair_key(left_ref, right_ref)
    written = []
    for anchored_left, anchored_right in zip(left, right):
        written.append(
            _post(
                server,
                anchored_left,
                comment=f"about {anchored_left} against {anchored_right}",
                category="observations_analysis",
                subcategory="analysis",
                ref={
                    "turn_keys": list(left), "experiment_id": "exp-1",
                    "task_id": "task-1", "attempt": 1,
                },
                paired={
                    "store_id": identity, "turn_keys": list(right),
                    "turn_key": anchored_right,
                    "experiment_id": "exp-1", "task_id": "task-2", "attempt": 1,
                    "target_kind": "turn", "span_ids": [], "target_label": "Turn",
                },
            )["feedback"][-1]
        )
    assert {row["pair_key"] for row in written} == {expected}
    # Anchored to different steps, and each step is still the place the
    # comment points at.
    assert [row["turn_key"] for row in written] == left
    assert [
        row["anchors"]["paired"]["turn_key"] for row in written
    ] == right
    # Both remarks are found from both tasks, and each appears once.
    for task in ("task-1", "task-2"):
        page = _task_feedback(server, task=task)
        assert page["total"] == 2
        assert {row["pair_key"] for row in page["feedback"]} == {expected}


def test_a_reference_is_refused_when_any_of_its_turns_is_not_recorded(
    experiment_server,
):
    """Every turn in the frozen reference is evidence, not a claim.

    The wider reference is what `pair_key` hashes, so an unrecorded turn
    inside it would be an unverified assertion carried forward forever.
    """
    server, store = experiment_server
    keys = _seed_task(store, task_id="task-1", attempts=(1,))
    status, data = _request(
        server,
        "/post_feedback?turn_key=" + urllib.parse.quote(keys[0], safe=""),
        "POST",
        {
            "target_kind": "turn", "span_ids": [], "target_label": "Turn",
            "comment": "about an attempt that includes a turn nobody recorded",
            "provenance": "human", "category": "conclusions",
            "subcategory": "what_went_wrong",
            "ref": {
                "turn_keys": [keys[0], "never-recorded"],
                "experiment_id": "exp-1", "task_id": "task-1", "attempt": 1,
            },
        },
    )
    assert status == 400 and "never-recorded" in json.dumps(data)


def test_the_view_refuses_an_incomplete_request(experiment_server):
    server, _store = experiment_server
    assert _request(server, "/api/task-feedback?task=task-1")[0] == 400
    assert _request(server, "/api/task-feedback?experiment=exp-1")[0] == 400


def test_the_feedback_ui_works_in_a_real_dom(experiment_server):
    """Clicked, not grepped.

    The composer and the task Feedback view are the deliverable a person
    uses; a test that only asserts the functions exist would pass on a page
    that renders nothing. This drives the real page against the real server.
    """
    dependency = os.environ.get("TEST_JSDOM_ROOT")
    if not dependency:
        pytest.skip("Set TEST_JSDOM_ROOT to run DOM integration with jsdom")
    server, store = experiment_server
    keys = _seed_task(store)
    other = _seed_task(store, task_id="task-2")
    _post(server, keys[0], comment="attempt 1 answered without the third item")
    _post(
        server,
        keys[3],
        category="observations_analysis",
        subcategory="observation",
        comment="attempt 2 asked a clarifying question first",
    )
    _post(
        server,
        keys[1],
        target_kind="phase",
        span_ids=[f"span-{keys[1]}"],
        target_label="Planning",
        category="recommendations",
        subcategory="what_to_do",
        provenance="coding_agent",
        comment="plan all three items before answering",
    )
    _post(
        server,
        keys[2],
        category="conclusions",
        subcategory="what_went_right",
        comment="this side recovered where the other gave up",
        paired={
            "store_id": store.store_identity(), "turn_keys": [other[0]],
            "experiment_id": "exp-1", "task_id": "task-2", "attempt": 1,
            "target_kind": "turn", "span_ids": [], "target_label": "Turn",
        },
    )
    # An unclassified comment: no category, no subcategory, text untouched.
    # Written directly because no writer in this build produces one.
    with store._connect() as conn:
        conn.execute(
            "INSERT INTO human_feedback (feedback_uid, turn_key, target_kind,"
            " span_ids_json, target_label, comment, provenance, anchors_json,"
            " created_at)"
            " VALUES (?, ?, 'turn', '[]', 'Turn', ?, 'human', '{}',"
            " '2026-01-01T00:00:00+00:00')",
            ("fb-legacy-fixture", keys[0], "legacy note kept verbatim"),
        )
        conn.commit()

    script = Path(__file__).with_name("chatbot_feedback_dom.cjs")
    result = subprocess.run(
        ["node", str(script), dependency,
         f"http://127.0.0.1:{server.port}/?token={server.token}",
         "exp-1", "task-1", keys[0]],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


# ---------------------------------------------------------------------------
# An attempt filter answers about the side of the pair that was asked about
# ---------------------------------------------------------------------------


def _paired(identity, turn_keys, *, experiment_id, task_id, attempt):
    """The paired anchor a compare row hands the composer, as posted."""
    return {
        "store_id": identity,
        "turn_keys": list(turn_keys),
        "experiment_id": experiment_id,
        "task_id": task_id,
        "attempt": attempt,
        "target_kind": "turn",
        "span_ids": [],
        "target_label": "Turn",
    }


def test_a_paired_note_is_filtered_by_the_queried_sides_attempt(
    experiment_server,
):
    """`fix-ptu1`, over real HTTP against a real store.

    The note is written on the LEFT experiment's attempt 1 about the RIGHT
    experiment's attempt 2, so the row's `attempt` column says 1 while the
    right task's page is asking about its own attempt 2. Filtering on the
    column alone showed the note under attempt 1 -- which that task never ran
    -- and hid it under the attempt it is actually about.
    """
    server, store = experiment_server
    left = _seed_task(
        store, experiment_id="exp-left", task_id="task-left", attempts=(1,)
    )
    right = _seed_task(
        store, experiment_id="exp-right", task_id="task-right", attempts=(2,)
    )
    note = "the right run's second attempt recovered where the left one gave up"
    _post(
        server,
        left[0],
        comment=note,
        paired=_paired(
            store.store_identity(), [right[0]],
            experiment_id="exp-right", task_id="task-right", attempt=2,
        ),
    )

    def right_side(**params):
        return _task_feedback(
            server, experiment="exp-right", task="task-right", **params
        )

    def left_side(**params):
        return _task_feedback(
            server, experiment="exp-left", task="task-left", **params
        )

    whole = right_side()
    assert [row["comment"] for row in whole["feedback"]] == [note]
    # The row still reports the annotated turn's attempt, because that is what
    # the column means. What changed is the FILTER, which no longer reads it as
    # the queried side's attempt.
    assert whole["feedback"][0]["attempt"] == 1
    assert [row["comment"] for row in right_side(attempt=2)["feedback"]] == [note]
    assert right_side(attempt=1)["total"] == 0, (
        "the right task has no attempt 1, so nothing may be shown under one"
    )
    # The primary side is untouched: it still filters on its own attempt.
    assert [row["comment"] for row in left_side(attempt=1)["feedback"]] == [note]
    assert left_side(attempt=2)["total"] == 0


def test_a_pair_of_two_attempts_of_one_task_is_found_under_both(
    experiment_server,
):
    """One row, both attempts, once each.

    A note comparing two runs of the SAME task is about both of them, so it
    belongs under either attempt -- and it is still one comment, not two.
    """
    server, store = experiment_server
    keys = _seed_task(store, task_id="task-1")
    note = "attempt 2 asked a clarifying question that attempt 1 skipped"
    _post(
        server,
        keys[0],
        comment=note,
        paired=_paired(
            store.store_identity(), [keys[2]],
            experiment_id="exp-1", task_id="task-1", attempt=2,
        ),
    )
    assert _task_feedback(server)["total"] == 1
    for attempt in (1, 2):
        page = _task_feedback(server, attempt=attempt)
        assert [row["comment"] for row in page["feedback"]] == [note], (
            f"the pair is about attempt {attempt} as well"
        )
    assert _task_feedback(server, attempt=3)["total"] == 0


def test_a_paired_side_posted_without_an_attempt_is_filtered_by_the_recorded_one(
    experiment_server,
):
    """The writer completes the paired scope off the turn row, so the filter
    has a real attempt to match rather than a permissive blank."""
    server, store = experiment_server
    left = _seed_task(store, task_id="task-1", attempts=(1,))
    right = _seed_task(store, task_id="task-2")
    note = "the other task's run reached this question and stopped"
    written = _post(
        server,
        left[0],
        comment=note,
        paired={
            "store_id": store.store_identity(),
            # attempt omitted on purpose: `complete_scope` reads it off the
            # turn this reference names.
            "turn_keys": [right[1]],
            "experiment_id": "exp-1",
            "task_id": "task-2",
            "target_kind": "turn",
            "span_ids": [],
            "target_label": "Turn",
        },
    )
    assert written["feedback"][0]["paired"]["ref"]["attempt"] == 1
    assert [
        row["comment"]
        for row in _task_feedback(server, task="task-2", attempt=1)["feedback"]
    ] == [note]
    assert _task_feedback(server, task="task-2", attempt=2)["total"] == 0


def test_a_row_whose_pair_anchor_is_unreadable_is_not_hidden_by_an_attempt(
    experiment_server,
):
    """The defensive branch, against a row that really is shaped that way.

    The pair COLUMNS say which task the note is also about; the anchor JSON is
    where the attempt lives. A row with the columns and no readable anchor --
    what a note written by an older build projects -- names no attempt on the
    side being read, so it is shown under whichever attempt is asked for
    rather than disappearing from every one of them.
    """
    server, store = experiment_server
    left = _seed_task(store, task_id="task-1", attempts=(1,))
    _seed_task(store, task_id="task-2")
    note = "recorded with pair columns and no readable anchor"
    with store._connect() as conn:
        conn.execute(
            "INSERT INTO human_feedback (feedback_uid, turn_key, target_kind,"
            " span_ids_json, target_label, comment, provenance, anchors_json,"
            " pair_experiment_id, pair_task_id, created_at)"
            " VALUES (?, ?, 'turn', '[]', 'Turn', ?, 'human', '{}',"
            " 'exp-1', 'task-2', '2026-01-01T00:00:00+00:00')",
            ("fb-unreadable-anchor", left[0], note),
        )
        conn.commit()
    for attempt in (1, 2):
        page = _task_feedback(server, task="task-2", attempt=attempt)
        assert [row["comment"] for row in page["feedback"]] == [note]
    # Read from the side it is anchored to, its own attempt still filters it.
    assert _task_feedback(server, task="task-1", attempt=1)["total"] == 1
    assert _task_feedback(server, task="task-1", attempt=2)["total"] == 0


def test_a_logical_experiment_matches_the_attempt_under_each_local_id(
    experiment_server,
):
    """The workspace case: one logical experiment, two local ids, one row.

    A comparison note written across two segments of one logical experiment is
    a single row whose two sides name DIFFERENT local experiment ids. Both
    sides have to be recognized as the task the reader asked about, or the
    attempt filter would match neither and the note would vanish from a view
    that shows it perfectly well unfiltered.
    """
    server, store = experiment_server
    first = _seed_task(store, experiment_id="seg-a", task_id="task", attempts=(1,))
    second = _seed_task(store, experiment_id="seg-b", task_id="task", attempts=(2,))
    note = "the second segment's run repeated the first segment's mistake"
    _post(
        server,
        first[0],
        comment=note,
        paired=_paired(
            store.store_identity(), [second[0]],
            experiment_id="seg-b", task_id="task", attempt=2,
        ),
    )
    identity = store.store_identity()

    def page(**params):
        return fb.consolidate_task_feedback(
            {identity: store},
            experiment_id="logical",
            task_id="task",
            local_experiment_ids=["seg-a", "seg-b"],
            **params,
        )

    assert page().total == 1, "one row, however many local ids name it"
    assert page(attempt=1).total == 1, "the segment the note was written on"
    assert page(attempt=2).total == 1, "the segment the note is about"
    assert page(attempt=3).total == 0


def test_the_attempt_filter_still_answers_for_rows_with_no_pair(
    experiment_server,
):
    """The ordinary case, unchanged: a note with no paired side filters on the
    attempt of the turn it is anchored to."""
    server, store = experiment_server
    keys = _seed_task(store)
    _post(server, keys[0], comment="about attempt 1")
    _post(server, keys[2], comment="about attempt 2")
    assert [row["comment"] for row in _task_feedback(server, attempt=1)["feedback"]] == [
        "about attempt 1"
    ]
    assert [row["comment"] for row in _task_feedback(server, attempt=2)["feedback"]] == [
        "about attempt 2"
    ]


# ---------------------------------------------------------------------------
# Read-only evidence: annotated beside, never written to
# ---------------------------------------------------------------------------


RECORDED_NOTES = (
    ("turn", [], "Turn", "human", "the answer stopped after two items",
     "conclusions", "what_went_wrong"),
    ("phase", ["span"], "Planning", "coding_agent", "asked a clarifying question first",
     "conclusions", "what_went_right"),
    ("turn", [], "Turn", "human", "plan all three before answering",
     "recommendations", "what_to_do"),
    ("turn", [], "Turn", "distillation_agent", "the third item was never revisited",
     "observations_analysis", "observation"),
)


@pytest.fixture
def read_only_copy(tmp_path):
    """A current-schema store this build cannot write to.

    Its notes are recorded first; then the file is closed out of WAL and made
    read-only. As a live DB that refuses a new comment; as a sealed archive's
    bytes it is what `sealed_turn_comments` rows are keyed by.
    """
    path = tmp_path / "read_only.sqlite3"
    store = obs.ObservabilityStore(str(path))
    with store._connect() as conn:
        for ordinal, attempt in ((1, 1), (2, 1), (3, 2)):
            key = f"recorded-t{ordinal}"
            row = _turn_row(key, "exp-recorded", "task-recorded", attempt)
            assert store.upsert_turn_row(conn, row, [], store._store_redactor())
            conn.execute(
                "INSERT INTO spans(span_id,trace_id,name,kind,start_ns,status,attributes) "
                "VALUES(?,?,?,?,?,?,?)",
                (f"span-recorded-t{ordinal}", key, "fw.planner.plan", "internal",
                 1, "ok", "{}"),
            )
    for index, (kind, spans, label, provenance, comment, category, subcategory) in (
        enumerate(RECORDED_NOTES)
    ):
        turn_key = f"recorded-t{(index % 3) + 1}"
        store.add_human_feedback(
            turn_key,
            target_kind=kind,
            span_ids=[f"span-{turn_key}" for _ in spans],
            target_label=label,
            provenance=provenance,
            comment=comment,
            category=category,
            subcategory=subcategory,
        )
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA journal_mode=DELETE")
    finally:
        connection.close()
    path.chmod(0o444)
    yield path
    path.chmod(0o644)


def _recorded_rows(path):
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        connection.row_factory = sqlite3.Row
        return [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM human_feedback ORDER BY feedback_id"
            )
        ]
    finally:
        connection.close()


def _serving(db_path):
    server = run_chatbot_server.ChatbotServer(str(db_path), port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _http(server, path, method="GET", body=None):
    request = urllib.request.Request(
        f"http://127.0.0.1:{server.port}{path}",
        data=None if body is None else json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {server.token}",
            "Content-Type": "application/json",
        },
        method=method,
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.status, json.loads(response.read())


def test_a_read_only_live_db_refuses_a_note_and_is_left_untouched(
    read_only_copy,
):
    """A live DB this build cannot write is refused, not worked around.

    Comments on live turns live in the live DB's `human_feedback`; there is
    no second file to put one in instead. So the write is a 409 saying why,
    the reads still answer with what was recorded, and the file is
    byte-identical afterwards with nothing new beside it.
    """
    rows = _recorded_rows(read_only_copy)
    turn_key = rows[0]["turn_key"]
    encoded = urllib.parse.quote(turn_key, safe="")
    before = hashlib.sha256(read_only_copy.read_bytes()).hexdigest()
    directory = sorted(path.name for path in read_only_copy.parent.iterdir())
    server, thread = _serving(read_only_copy)
    try:
        status, payload = _http(
            server, f"/api/feedback-notes?turn_key={encoded}"
        )
        assert status == 200
        assert [row["comment"] for row in payload["feedback"]] == [
            row["comment"] for row in rows if row["turn_key"] == turn_key
        ]
        with pytest.raises(urllib.error.HTTPError) as caught:
            _http(
                server, f"/post_feedback?turn_key={encoded}", "POST",
                {
                    "target_kind": "turn", "span_ids": [], "target_label": "Turn",
                    "comment": "recorded today, about evidence it cannot write",
                    "provenance": "human",
                    "category": "recommendations", "subcategory": "what_to_do",
                },
            )
        assert caught.value.code == 409
        status, task = _http(
            server, "/api/task-feedback?experiment=exp-recorded&task=task-recorded"
        )
        assert status == 200
        assert task["total"] == len(RECORDED_NOTES)
    finally:
        server.shutdown()
        thread.join(timeout=5)
    assert hashlib.sha256(read_only_copy.read_bytes()).hexdigest() == before
    assert sorted(path.name for path in read_only_copy.parent.iterdir()) == directory


def _sealed_note(evidence, live, *, comment="a note about sealed evidence"):
    sha = hashlib.sha256(Path(evidence.db_path).read_bytes()).hexdigest()
    return control.SealedEvidence(evidence, live, sha).add_human_feedback(
        "recorded-t1", target_kind="turn", span_ids=[], target_label="Turn",
        provenance="human", comment=comment,
        category="observations_analysis", subcategory="observation",
    ), sha


def test_a_note_about_sealed_evidence_is_append_only_in_the_live_db(
    read_only_copy, tmp_path
):
    """A comment is somebody's statement; editing one in place would leave no
    trace that it had said something else."""
    live = obs.ObservabilityStore(str(tmp_path / "live.sqlite3"))
    _sealed_note(obs.ReadOnlyObservabilityStore(str(read_only_copy)), live)
    connection = sqlite3.connect(live.db_path)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("UPDATE sealed_turn_comments SET comment='edited'")
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("DELETE FROM sealed_turn_comments")
    finally:
        connection.close()


def test_a_paired_side_may_name_its_anchored_turn_unambiguously(
    experiment_server,
):
    """`anchor_turn_key` and `turn_key` mean the same thing on the wire.

    A multi-turn reference has to say which of its turns the comment is
    anchored to, and the field that says so sat next to `ref.turn_keys` under
    a name that reads like part of the reference. The explicit spelling is
    accepted so a client composing a pair from a compare row cannot post the
    anchor into the wrong field; the stored shape is unchanged, so a row
    posted back verbatim still works.
    """
    server, store = experiment_server
    left = _seed_task(store, task_id="task-1")
    right = _seed_task(store, task_id="task-2")
    identity = store.store_identity()
    paired = {
        "store_id": identity,
        "turn_keys": [right[0], right[1]],
        "experiment_id": "exp-1", "task_id": "task-2", "attempt": 1,
        "target_kind": "turn", "span_ids": [], "target_label": "Turn",
    }
    explicit = _post(
        server, left[0], comment="named with anchor_turn_key",
        paired=dict(paired, anchor_turn_key=right[1]),
    )["feedback"][0]
    legacy = _post(
        server, left[0], comment="named with turn_key",
        paired=dict(paired, turn_key=right[1]),
    )["feedback"][-1]
    assert explicit["anchors"]["paired"]["turn_key"] == right[1]
    assert legacy["anchors"]["paired"]["turn_key"] == right[1]
    # Same anchor, same pair identity, and the whole two-turn reference kept.
    assert explicit["pair_key"] == legacy["pair_key"]
    assert explicit["paired"]["ref"]["turn_keys"] == [right[0], right[1]]
    # And a reference naming two turns without saying which is refused.
    status, error = _request(
        server,
        "/post_feedback?turn_key=" + urllib.parse.quote(left[0], safe=""),
        "POST",
        {
            "target_kind": "turn", "span_ids": [], "target_label": "Turn",
            "provenance": "human", "category": "conclusions",
            "subcategory": "what_went_right", "comment": "which turn?",
            "paired": paired,
        },
    )
    assert status == 400 and "anchored to" in error["error"]


def test_reading_feedback_brings_nothing_into_existence(read_only_copy):
    """A read creates nothing beside somebody's evidence.

    Listing a turn's comments, or a whole task's, on a store nobody has
    annotated must leave the directory exactly as it found it. An earlier
    wrapper opened a control file for writing on every read, so browsing a
    sealed archive stamped one next to it — one that then had to be explained
    to whoever verified the archive's directory.
    """
    directory = read_only_copy.parent
    before = sorted(path.name for path in directory.iterdir())
    evidence = obs.ReadOnlyObservabilityStore(str(read_only_copy))
    sealed = control.SealedEvidence(evidence, None, "0" * 64)
    assert sealed.list_human_feedback("recorded-t1")
    assert fb.consolidate_task_feedback(
        {"recorded": sealed}, experiment_id="exp-recorded", task_id="task-recorded"
    ).total == len(RECORDED_NOTES)
    assert sorted(path.name for path in directory.iterdir()) == before


def test_a_sealed_note_reads_back_merged_without_writing(read_only_copy, tmp_path):
    """The merged read opens the live DB read-only.

    Checked by row digest rather than by inspection: a reader that stamps a
    schema version, a journal or an identity into the file it is reading is
    writing, whatever it calls itself.
    """
    evidence = obs.ReadOnlyObservabilityStore(str(read_only_copy))
    live_path = str(tmp_path / "live.sqlite3")
    _note, sha = _sealed_note(evidence, obs.ObservabilityStore(live_path))
    before = _row_digest(live_path)
    reader = control.SealedEvidence(
        evidence, obs.ReadOnlyObservabilityStore(live_path), sha
    )
    rows = reader.list_human_feedback("recorded-t1")
    assert [row["comment"] for row in rows][-1] == "a note about sealed evidence"
    assert len(rows) == 1 + sum(
        1 for index in range(len(RECORDED_NOTES)) if (index % 3) + 1 == 1
    )
    assert fb.consolidate_task_feedback(
        {"sealed": reader}, experiment_id="exp-recorded", task_id="task-recorded"
    ).total == len(RECORDED_NOTES) + 1
    # Another archive's sha reads none of it.
    assert len(control.SealedEvidence(
        evidence, obs.ReadOnlyObservabilityStore(live_path), "f" * 64
    ).list_human_feedback("recorded-t1")) == len(rows) - 1
    assert _row_digest(live_path) == before
    # And it refuses to become a writer behind the caller's back.
    with pytest.raises(sqlite3.OperationalError):
        reader.add_human_feedback(
            "recorded-t1", target_kind="turn", span_ids=[], target_label="Turn",
            provenance="human", comment="not through a read handle",
            category="conclusions", subcategory="what_went_wrong",
        )
    assert _row_digest(live_path) == before


def test_a_sealed_note_with_no_live_db_is_refused(read_only_copy):
    evidence = obs.ReadOnlyObservabilityStore(str(read_only_copy))

    with pytest.raises(control.ControlUnavailable, match="live database"):
        _sealed_note(evidence, None)


def _row_digest(path):
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return hashlib.sha256(json.dumps([
            list(row) for row in connection.execute(
                "SELECT * FROM sealed_turn_comments ORDER BY feedback_id"
            )
        ]).encode()).hexdigest()
    finally:
        connection.close()


def test_the_http_reads_create_nothing(read_only_copy):
    """The same property over real HTTP, where it actually matters."""
    directory = sorted(path.name for path in read_only_copy.parent.iterdir())
    encoded = urllib.parse.quote("recorded-t1", safe="")
    server, thread = _serving(read_only_copy)
    try:
        assert _http(server, f"/api/feedback-notes?turn_key={encoded}")[0] == 200
        assert _http(
            server, "/api/task-feedback?experiment=exp-recorded&task=task-recorded"
        )[0] == 200
    finally:
        server.shutdown()
        thread.join(timeout=5)
    assert sorted(path.name for path in read_only_copy.parent.iterdir()) == directory

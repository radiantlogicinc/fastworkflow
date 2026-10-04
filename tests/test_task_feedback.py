"""The task Feedback view: every authorized comment on one task.

Real stores, real HTTP, no mocks.

The legacy half of this module proves the one thing a schema bump can quietly
break: that feedback somebody already wrote is still readable, still says
exactly what its author typed, and is shown as unclassified rather than
guessed into one of the six new subcategories — and that a reader can still
record a NEW categorized comment about that old evidence without a byte of it
moving.

It runs against a compact v6 database built at test time BY THE REAL v6 STORE
CODE, loaded out of this repository's own git history. Nothing about the old
schema is restated and no database is committed, so the guarantee is checked
on every machine rather than only where a private corpus happens to sit. The
private corpus is still read once, by `test_the_private_audit_corpus_reads_
the_same_way`, from a copy of a copy; the source database is never opened.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pytest

from fastworkflow.observability import feedback as fb
from fastworkflow.observability import feedback_sidecar
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

AUDIT_COPY = Path("/tmp/fw-audit-copies/ido_live.sqlite3")


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
    # An unclassified comment, as a pre-taxonomy row projects when a v6 store
    # is consolidated into this view: no category, no subcategory, text
    # untouched. Written directly because no writer in this build produces
    # one — which is the point. `feedback_uid` is NOT NULL here; a real v6
    # file has no such column at all and the reader synthesizes the identity.
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
# Legacy: a real v6 corpus, read-only, never rewritten
# ---------------------------------------------------------------------------


def _v6_store_module():
    """The real v6 store code, loaded out of this repository's own history.

    The v6 schema is not restated here and no v6 database is committed. The
    newest commit whose `store.py` still says `SCHEMA_VERSION = 6` IS the
    build that wrote every v6 corpus in existence, so a fixture it creates has
    the authentic old shape by construction and cannot drift from it the way a
    hand-copied CREATE TABLE would. It is loaded under its own module name, so
    the v6 and v7 stores coexist in one process.
    """
    log = subprocess.run(
        ["git", "log", "--format=%H", "--", "fastworkflow/observability/store.py"],
        capture_output=True, text=True, cwd=str(Path(__file__).parents[1]),
    )
    if log.returncode != 0:
        pytest.skip("not a git checkout; the v6 baseline comes from history")
    for commit in log.stdout.split():
        shown = subprocess.run(
            ["git", "show", f"{commit}:fastworkflow/observability/store.py"],
            capture_output=True, text=True, cwd=str(Path(__file__).parents[1]),
        )
        if shown.returncode == 0 and "\nSCHEMA_VERSION = 6\n" in shown.stdout:
            break
    else:
        pytest.skip("no v6 store.py in history to build a legacy fixture from")
    directory = tempfile.mkdtemp(prefix="fw-v6-baseline-")
    source = Path(directory) / "store_v6.py"
    source.write_text(shown.stdout)
    spec = importlib.util.spec_from_file_location("fastworkflow_store_v6", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules["fastworkflow_store_v6"] = module
    spec.loader.exec_module(module)
    assert module.SCHEMA_VERSION == 6
    return module


@pytest.fixture(scope="session")
def v6_baseline():
    return _v6_store_module()


# Public, synthetic, and deliberately including the OLD three-heading composer
# output: those headings were presentation baked into comment text, and the
# point of the legacy tests is that this build shows them verbatim instead of
# reading `what_went_wrong` out of them.
LEGACY_NOTES = (
    ("turn", [], "Turn", "human", {"comment": "the answer stopped after two items"}),
    ("phase", ["span"], "Planning", "coding_agent",
     {"went_wrong": "never revisited the third item",
      "worked": "asked a clarifying question first"}),
    ("turn", [], "Turn", "human", {"should_change": "plan all three before answering"}),
    ("turn", [], "Turn", "distillation_agent",
     {"comment": "What went wrong: this line is content, not a category"}),
)


@pytest.fixture
def legacy_v6_copy(v6_baseline, tmp_path):
    """A compact, REAL v6 database, written by the real v6 writer.

    Built at test time rather than committed, and seeded with public synthetic
    text, so the legacy guarantees are checked on every machine instead of
    only where a private corpus happens to exist. The private corpus is still
    read, once, by `test_the_private_audit_corpus_reads_the_same_way`.
    """
    path = tmp_path / "legacy_v6.sqlite3"
    store = v6_baseline.ObservabilityStore(str(path))
    with store._connect() as conn:
        for ordinal, attempt in ((1, 1), (2, 1), (3, 2)):
            key = f"legacy-t{ordinal}"
            row = _turn_row(key, "exp-legacy", "task-legacy", attempt)
            assert store.upsert_turn_row(conn, row, [], store._store_redactor())
            conn.execute(
                "INSERT INTO spans(span_id,trace_id,name,kind,start_ns,status,attributes) "
                "VALUES(?,?,?,?,?,?,?)",
                (f"span-legacy-t{ordinal}", key, "fw.planner.plan", "internal",
                 1, "ok", "{}"),
            )
    for index, (kind, spans, label, provenance, text) in enumerate(LEGACY_NOTES):
        turn_key = f"legacy-t{(index % 3) + 1}"
        store.add_human_feedback(
            turn_key,
            target_kind=kind,
            span_ids=[f"span-{turn_key}" for _ in spans],
            target_label=label,
            provenance=provenance,
            **text,
        )
    yield path


@pytest.fixture(scope="session")
def audit_corpus_copy(tmp_path_factory):
    """One copy of the private historical corpus for the whole session.

    Optional by design: it is 600MB of private payload that exists on one
    machine, and copying it per test bought nothing the synthetic fixture does
    not already prove. The source is never opened — this copies a copy.
    """
    if not AUDIT_COPY.exists():
        pytest.skip("no private audit copy available on this machine")
    path = tmp_path_factory.mktemp("audit") / "legacy_v6.sqlite3"
    shutil.copyfile(AUDIT_COPY, path)
    return path


def _legacy_rows(path):
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


def test_the_v6_fixture_is_the_shape_this_test_claims(legacy_v6_copy):
    """Guards the rest of the module: if the fixture stops being v6 feedback
    written before the taxonomy, these tests prove nothing and should say so
    rather than pass vacuously."""
    connection = sqlite3.connect(f"file:{legacy_v6_copy}?mode=ro", uri=True)
    try:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 6
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(human_feedback)")
        }
        assert columns == {
            "feedback_id", "turn_key", "target_kind", "span_ids_json",
            "target_label", "comment", "provenance", "created_at",
        }
        assert connection.execute(
            "SELECT COUNT(*) FROM human_feedback"
        ).fetchone()[0] == len(LEGACY_NOTES)
        # The old composer's headings are in the TEXT of two rows. Nothing may
        # read a category out of them.
        assert connection.execute(
            "SELECT COUNT(*) FROM human_feedback WHERE comment LIKE 'What went wrong:%'"
        ).fetchone()[0] == 2
    finally:
        connection.close()


def test_legacy_feedback_reads_back_verbatim_and_unclassified(legacy_v6_copy):
    """The whole point of keeping v6 readable.

    Every recorded comment still reads, character for character, with its
    original provenance, target and timestamp — and with no category, because
    its author never chose one. Deriving `what_went_wrong` from a heading in
    the text would record a guess as the author's decision.
    """
    expected = _legacy_rows(legacy_v6_copy)
    store = obs.ReadOnlyObservabilityStore(str(legacy_v6_copy))
    assert store.schema_version == 6
    by_turn = {}
    for row in expected:
        by_turn.setdefault(row["turn_key"], []).append(row)
    seen = 0
    for turn_key, original in by_turn.items():
        rows = fb.present(store.list_human_feedback(turn_key))
        assert len(rows) == len(original)
        for read, was in zip(rows, original):
            assert read["comment"] == was["comment"]
            assert read["provenance"] == was["provenance"]
            assert read["target_kind"] == was["target_kind"]
            assert read["target_label"] == was["target_label"]
            assert read["created_at"] == was["created_at"]
            assert read["span_ids"] == json.loads(was["span_ids_json"] or "[]")
            assert read["category"] is None and read["subcategory"] is None
            assert read["classified"] is False
            assert read["category_label"] is None
            assert read["subcategory_label"] is None
            assert read["paired"] is None and read["pair_key"] is None
            seen += 1
    assert seen == len(expected) > 0


def test_legacy_rows_still_deduplicate_and_consolidate(legacy_v6_copy):
    """Unclassified rows are real feedback and appear in the task view."""
    store = obs.ReadOnlyObservabilityStore(str(legacy_v6_copy))
    rows = _legacy_rows(legacy_v6_copy)
    keys = {
        fb.dedupe_key({**row, "store_id": "legacy", "feedback_uid": None})
        for row in rows
    }
    assert len(keys) == len(rows)
    assert all(key.startswith("legacy:") for key in keys)
    # Whatever task the corpus's own turns belong to, consolidation over the
    # read-only store must not raise and must never claim a classification.
    turn = store.get_turn(rows[0]["turn_key"])
    if turn and turn.get("experiment_id") and turn.get("task_id"):
        page = fb.consolidate_task_feedback(
            {"legacy": store},
            experiment_id=turn["experiment_id"],
            task_id=turn["task_id"],
        )
        assert page.total >= 1
        assert all(row["category"] is None for row in page.rows)


def test_a_legacy_row_is_filtered_by_its_own_attempt(legacy_v6_copy):
    """A v6 row has no pair columns at all, so the turn it is anchored to is
    the only side there is -- and the attempt filter still answers from it."""
    store = obs.ReadOnlyObservabilityStore(str(legacy_v6_copy))
    rows = _legacy_rows(legacy_v6_copy)
    by_attempt = {1: 0, 2: 0}
    for index in range(len(rows)):
        by_attempt[2 if (index % 3) + 1 == 3 else 1] += 1
    for attempt, expected in by_attempt.items():
        page = fb.consolidate_task_feedback(
            {"legacy": store},
            experiment_id="exp-legacy",
            task_id="task-legacy",
            attempt=attempt,
        )
        assert page.total == expected
        assert all(row["attempt"] == attempt for row in page.rows)
    assert sum(by_attempt.values()) == len(rows) > 0


def test_reading_the_legacy_copy_does_not_write_to_it(legacy_v6_copy):
    """No source evidence write, and no migration on open either."""
    before = hashlib.sha256(legacy_v6_copy.read_bytes()).hexdigest()
    store = obs.ReadOnlyObservabilityStore(str(legacy_v6_copy))
    for row in _legacy_rows(legacy_v6_copy)[:5]:
        store.list_human_feedback(row["turn_key"])
    assert hashlib.sha256(legacy_v6_copy.read_bytes()).hexdigest() == before
    connection = sqlite3.connect(f"file:{legacy_v6_copy}?mode=ro", uri=True)
    try:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 6
    finally:
        connection.close()


def test_a_v6_store_refuses_a_taxonomy_write_instead_of_half_doing_one(
    legacy_v6_copy,
):
    """The consequence of the bump, stated plainly.

    A v6 database has no category, subcategory, anchor or identity columns.
    The writer refuses it at open rather than inserting a row that the task
    view could never file or deduplicate.
    """
    with pytest.raises(obs.IncompatibleObservabilityDB):
        obs.ObservabilityStore.open_for_annotation(str(legacy_v6_copy))


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


def test_the_chatbot_records_a_note_beside_a_legacy_store_without_touching_it(
    legacy_v6_copy,
):
    """The whole of fix-9eg.19.1 on evidence this build must not write to.

    A v6 database has nowhere to put a category, an anchor or an identity, and
    this build does not migrate a database it did not create. Refusing the
    comment was the wrong answer: it goes to the annotation sidecar beside the
    evidence, the evidence file is byte-identical afterwards, and the reads —
    the turn's notes and the consolidated task view — return the new
    categorized note and the old unclassified ones as one list, deduplicated
    by stable identity.
    """
    rows = _legacy_rows(legacy_v6_copy)
    turn_key = rows[0]["turn_key"]
    encoded = urllib.parse.quote(turn_key, safe="")
    before = hashlib.sha256(legacy_v6_copy.read_bytes()).hexdigest()
    sidecar = Path(feedback_sidecar.feedback_db_path_for(str(legacy_v6_copy)))
    assert not sidecar.exists()
    server, thread = _serving(legacy_v6_copy)
    try:
        status, payload = _http(
            server, f"/api/feedback-notes?turn_key={encoded}"
        )
        assert status == 200
        assert [row["comment"] for row in payload["feedback"]] == [
            row["comment"] for row in rows if row["turn_key"] == turn_key
        ]
        assert all(row["classified"] is False for row in payload["feedback"])
        # Reading does not create the sidecar.
        assert not sidecar.exists()
        status, written = _http(
            server, f"/post_feedback?turn_key={encoded}", "POST",
            {
                "target_kind": "turn", "span_ids": [], "target_label": "Turn",
                "comment": "recorded today, about evidence from before",
                "provenance": "human",
                "category": "recommendations", "subcategory": "what_to_do",
            },
        )
        assert status == 201, written
        new = [
            row for row in written["feedback"]
            if row["comment"] == "recorded today, about evidence from before"
        ]
        assert len(new) == 1
        assert new[0]["category"] == "recommendations" and new[0]["classified"]
        assert new[0]["feedback_uid"].startswith("fb-")
        # The old comments are still there, still unclassified, in one list.
        assert len(written["feedback"]) == len(payload["feedback"]) + 1
        status, task = _http(
            server, "/api/task-feedback?experiment=exp-legacy&task=task-legacy"
        )
        assert status == 200
        assert task["total"] == len(LEGACY_NOTES) + 1
        assert sorted(row["classified"] for row in task["feedback"]) == [
            False, False, False, False, True,
        ]
        assert len({row["feedback_uid"] or row["feedback_id"]
                    for row in task["feedback"]}) == len(task["feedback"])
    finally:
        server.shutdown()
        thread.join(timeout=5)
    assert hashlib.sha256(legacy_v6_copy.read_bytes()).hexdigest() == before
    assert sidecar.exists()
    with sqlite3.connect(f"file:{legacy_v6_copy}?mode=ro", uri=True) as evidence:
        assert evidence.execute(
            "SELECT COUNT(*) FROM human_feedback"
        ).fetchone()[0] == len(LEGACY_NOTES)
        assert evidence.execute("PRAGMA user_version").fetchone()[0] == 6


def test_a_recorded_note_is_append_only_even_in_the_sidecar(legacy_v6_copy):
    """A comment is somebody's statement; editing one in place would leave no
    trace that it had said something else."""
    evidence = obs.ReadOnlyObservabilityStore(str(legacy_v6_copy))
    annotated = feedback_sidecar.AnnotatedEvidence.for_writing(evidence)
    annotated.add_human_feedback(
        "legacy-t1", target_kind="turn", span_ids=[], target_label="Turn",
        provenance="human", comment="a note about old evidence",
        category="observations_analysis", subcategory="observation",
    )
    path = feedback_sidecar.feedback_db_path_for(str(legacy_v6_copy))
    connection = sqlite3.connect(path)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("UPDATE feedback_notes SET comment='edited'")
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("DELETE FROM feedback_notes")
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


def test_reading_feedback_never_brings_a_sidecar_into_existence(legacy_v6_copy):
    """A read creates nothing beside somebody's evidence.

    Listing a turn's comments, or a whole task's, on a store nobody has
    annotated must leave the directory exactly as it found it. The earlier
    wrapper opened the sidecar for writing on every read, so browsing a sealed
    archive stamped a control file next to it — one that then had to be
    explained to whoever verified the archive's directory.
    """
    directory = legacy_v6_copy.parent
    before = sorted(path.name for path in directory.iterdir())
    evidence = obs.ReadOnlyObservabilityStore(str(legacy_v6_copy))
    reader = feedback_sidecar.reader_for(evidence)
    assert reader is evidence, "with no sidecar there is nothing to merge"
    assert reader.list_human_feedback("legacy-t1")
    assert fb.consolidate_task_feedback(
        {"legacy": reader}, experiment_id="exp-legacy", task_id="task-legacy"
    ).total == len(LEGACY_NOTES)
    assert sorted(path.name for path in directory.iterdir()) == before


def test_an_existing_sidecar_is_read_without_being_written_to(legacy_v6_copy):
    """The merged read opens the control file read-only.

    Checked by byte digest rather than by inspection: a reader that stamps a
    schema version, a journal or an identity into the file it is reading is
    writing, whatever it calls itself.
    """
    evidence = obs.ReadOnlyObservabilityStore(str(legacy_v6_copy))
    feedback_sidecar.AnnotatedEvidence.for_writing(evidence).add_human_feedback(
        "legacy-t1", target_kind="turn", span_ids=[], target_label="Turn",
        provenance="human", comment="a note about old evidence",
        category="observations_analysis", subcategory="observation",
    )
    path = Path(feedback_sidecar.feedback_db_path_for(str(legacy_v6_copy)))
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    directory = sorted(item.name for item in path.parent.iterdir())
    reader = feedback_sidecar.reader_for(evidence)
    assert isinstance(reader, feedback_sidecar.AnnotatedEvidence)
    assert reader.sidecar.read_only is True
    rows = reader.list_human_feedback("legacy-t1")
    assert [row["comment"] for row in rows][-1] == "a note about old evidence"
    assert len(rows) == 1 + sum(
        1 for index in range(len(LEGACY_NOTES)) if (index % 3) + 1 == 1
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert sorted(item.name for item in path.parent.iterdir()) == directory
    # And it refuses to become a writer behind the caller's back.
    with pytest.raises(feedback_sidecar.FeedbackSidecarError):
        reader.add_human_feedback(
            "legacy-t1", target_kind="turn", span_ids=[], target_label="Turn",
            provenance="human", comment="not through a read handle",
            category="conclusions", subcategory="what_went_wrong",
        )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


@pytest.mark.parametrize("damage", ["version", "identity", "not_a_database"])
def test_a_sidecar_this_build_does_not_understand_is_refused_untouched(
    legacy_v6_copy, damage
):
    """Refused before any DDL, with its bytes exactly as they were.

    Running CREATE TABLE first and checking the version afterwards would
    "repair" a file this build has already decided it cannot read — and the
    repair is indistinguishable, afterwards, from the file having been fine.
    """
    evidence = obs.ReadOnlyObservabilityStore(str(legacy_v6_copy))
    path = Path(feedback_sidecar.feedback_db_path_for(str(legacy_v6_copy)))
    if damage == "not_a_database":
        path.write_bytes(b"this is not sqlite")
    else:
        feedback_sidecar.AnnotatedEvidence.for_writing(evidence)
        connection = sqlite3.connect(path)
        try:
            if damage == "version":
                connection.execute(
                    "UPDATE meta SET value='99' WHERE key='schema_version'"
                )
            else:
                connection.execute(
                    "UPDATE meta SET value='someone-elses-store' "
                    "WHERE key='evidence_store_identity'"
                )
            connection.commit()
        finally:
            connection.close()
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(feedback_sidecar.FeedbackSidecarError):
        feedback_sidecar.reader_for(evidence)
    with pytest.raises(feedback_sidecar.FeedbackSidecarError):
        feedback_sidecar.AnnotatedEvidence.for_writing(evidence)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_the_http_reads_create_no_sidecar_and_report_an_unreadable_one(
    legacy_v6_copy,
):
    """The same two properties over real HTTP, where it actually matters."""
    sidecar = Path(feedback_sidecar.feedback_db_path_for(str(legacy_v6_copy)))
    encoded = urllib.parse.quote("legacy-t1", safe="")
    server, thread = _serving(legacy_v6_copy)
    try:
        assert _http(server, f"/api/feedback-notes?turn_key={encoded}")[0] == 200
        assert _http(
            server, "/api/task-feedback?experiment=exp-legacy&task=task-legacy"
        )[0] == 200
        assert not sidecar.exists(), "a GET must not create a control file"
        sidecar.write_bytes(b"this is not sqlite")
        before = hashlib.sha256(sidecar.read_bytes()).hexdigest()
        with pytest.raises(urllib.error.HTTPError) as caught:
            _http(server, f"/api/feedback-notes?turn_key={encoded}")
        assert caught.value.code == 409
        assert hashlib.sha256(sidecar.read_bytes()).hexdigest() == before
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_a_sidecar_refuses_to_answer_about_a_different_store(
    legacy_v6_copy, tmp_path
):
    """Bound to the evidence it was opened against, so a control file carried
    to another store is refused rather than reporting somebody else's runs."""
    evidence = obs.ReadOnlyObservabilityStore(str(legacy_v6_copy))
    feedback_sidecar.AnnotatedEvidence.for_writing(evidence)
    carried = tmp_path / "carried.feedback.sqlite3"
    shutil.copyfile(feedback_sidecar.feedback_db_path_for(str(legacy_v6_copy)), carried)
    with pytest.raises(feedback_sidecar.FeedbackSidecarError):
        feedback_sidecar.FeedbackAnnotationStore(
            str(carried), evidence_store_identity="some-other-store"
        )


def test_the_private_audit_corpus_reads_the_same_way(audit_corpus_copy):
    """The optional historical probe.

    The synthetic fixture above proves the contract everywhere; this checks it
    against real recorded feedback from before the taxonomy, on the one
    machine that has a copy. It reads and never writes.
    """
    before = hashlib.sha256(audit_corpus_copy.read_bytes()).hexdigest()
    expected = _legacy_rows(audit_corpus_copy)
    assert len(expected) > 0
    store = obs.ReadOnlyObservabilityStore(str(audit_corpus_copy))
    assert store.schema_version == 6
    seen = 0
    for turn_key in dict.fromkeys(row["turn_key"] for row in expected):
        original = [row for row in expected if row["turn_key"] == turn_key]
        read = fb.present(store.list_human_feedback(turn_key))
        assert [row["comment"] for row in read] == [
            row["comment"] for row in original
        ]
        assert all(row["classified"] is False for row in read)
        assert all(row["category"] is None for row in read)
        seen += len(read)
    assert seen == len(expected)
    assert hashlib.sha256(audit_corpus_copy.read_bytes()).hexdigest() == before

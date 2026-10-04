"""The turn finder scoped to one experiment, task or attempt.

`fix-9eg.3.1.1`. The route has filtered by experiment/task/attempt since the
attempt opener was written against it; what was missing was the finder sending
those filters, saying which scope it is answering about, and keeping the scope
attached to the source it was chosen in.

Integration throughout, per the repo's testing rules: a real
`ObservabilityStore` seeded through its own write methods, attempts written
through the real `ExperimentController`, read back through the real
`ChatbotServer` over a real socket, and -- where jsdom is available -- the
shipped page driven in a real DOM with real clicks.

The world is built so a scoped answer and an unscoped one CANNOT coincide:

- two experiments record the same task id, so an answer that dropped the
  experiment filter would pool two populations under one label;
- one attempt records more turns than the page's own scan bound, so a scoped
  search has to continue and the continuation has to stay scoped;
- turns with no experiment labels at all sit in the same store, so a scope that
  was ignored would show them.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path

import pytest

from fastworkflow import state_paths
from fastworkflow.benchmark import setup
from fastworkflow.experiment.runner import ExperimentController
from fastworkflow.observability import store as obs
from fastworkflow.run_chatbot import server as run_chatbot_server
from tests.test_chatbot_benchmarks import _request
from tests.test_trace_diagnosis import (
    _output,
    _plain_turn,
    _record,
    _turn_row,
    _write,
)

ALPHA = "exp-alpha"
BETA = "exp-beta"
TASK_ONE = "task-one"
TASK_TWO = "task-two"

# Above the page's own TURN_FIND_SCAN (200), so a search scoped to this one
# attempt cannot finish in a single request and the client must continue
# WITHOUT losing the scope -- the failure this number exists to catch.
BULK_TURNS = 205

ALPHA_ONE_A2 = ["turn-alpha-one-a2-0-ok", "turn-alpha-one-a2-1-failed"]
ALPHA_ONE_A2_FAILED = "turn-alpha-one-a2-1-failed"
ALPHA_TWO_A1 = ["turn-alpha-two-a1-0", "turn-alpha-two-a1-1"]
BETA_ONE_A1_FAILED = "turn-beta-one-a1-0-failed"
UNLABELLED = [f"turn-unlabelled-{index}" for index in range(5)]


def _failed_turn(store: obs.ObservabilityStore, turn_key: str, **labels) -> None:
    """A completed turn whose RECORD carries an unsuccessful dispatch.

    Record-only rather than span-only on purpose: `step_unsuccessful` is a
    record-sensitive marker, so this is also the shape that proves a scoped
    marker search still reads the record basis the unscoped one reads.
    """
    _write(
        store,
        _turn_row(
            turn_key,
            record=_record(
                turn_key,
                refs=[("c1", 1, None)],
                outputs=[_output("c1", "delete_todo", success=False)],
            ),
            **labels,
        ),
        [],
    )


@pytest.fixture
def finder_world(tmp_path, monkeypatch):
    """One workflow, two experiments, a shared task id and an unlabelled tail."""
    monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "state"))
    folder = tmp_path / "finder_workflow"
    (folder / "_commands").mkdir(parents=True)
    setup.save_benchmark(folder, {"title": "Finder", "tasks": [{"prompt": "Do it"}]})

    db_path = state_paths.observability_db(str(folder))
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    store = obs.ObservabilityStore(db_path)
    controller = ExperimentController(
        db_path, store.store_identity(), external=False,
        workflow_folderpath=str(folder),
    )
    workflow_name = setup.workflow_name_for(folder)
    controller.create_experiment(
        ALPHA, "the run under the lens", declared_tasks=2, declared_attempts=2,
        # Four declared, three run: `task-two` attempt 2 is never started, which
        # is an ordinary unfinished attempt and keeps the store honest about
        # what a scope can and cannot find.
        declarations=[(TASK_ONE, 1, "ch-alpha-one-1"), (TASK_ONE, 2, "ch-alpha-one-2"),
                      (TASK_TWO, 1, "ch-alpha-two-1"), (TASK_TWO, 2, "ch-alpha-two-2")],
        workflow_name=workflow_name,
    )
    controller.create_experiment(
        BETA, "a second run of the same task", declared_tasks=1,
        declared_attempts=1, declarations=[(TASK_ONE, 1, "ch-beta-one-1")],
        workflow_name=workflow_name,
    )

    # alpha / task-one / attempt 1: the large one.
    controller.start_attempt(ALPHA, TASK_ONE, 1, "ch-alpha-one-1")
    for index in range(BULK_TURNS):
        _plain_turn(store, f"turn-alpha-one-a1-{index:04d}", experiment_id=ALPHA,
                    task_id=TASK_ONE, attempt=1)
    controller.finish_attempt(ALPHA, TASK_ONE, 1, outcome="fail",
                              outcome_source="test")

    # alpha / task-one / attempt 2: two turns, one of which recorded a failure.
    controller.start_attempt(ALPHA, TASK_ONE, 2, "ch-alpha-one-2")
    _plain_turn(store, ALPHA_ONE_A2[0], experiment_id=ALPHA, task_id=TASK_ONE,
                attempt=2)
    _failed_turn(store, ALPHA_ONE_A2_FAILED, experiment_id=ALPHA,
                 task_id=TASK_ONE, attempt=2)
    controller.finish_attempt(ALPHA, TASK_ONE, 2, outcome="pass",
                              outcome_source="test")

    # alpha / task-two / attempt 1: a second task in the same experiment.
    controller.start_attempt(ALPHA, TASK_TWO, 1, "ch-alpha-two-1")
    for turn_key in ALPHA_TWO_A1:
        _plain_turn(store, turn_key, experiment_id=ALPHA, task_id=TASK_TWO,
                    attempt=1)
    controller.finish_attempt(ALPHA, TASK_TWO, 1, outcome="pass",
                              outcome_source="test")

    # beta / task-one / attempt 1: THE SAME TASK ID under another experiment,
    # and it too recorded a failure, so a marker search that lost the
    # experiment filter would return it beside alpha's.
    controller.start_attempt(BETA, TASK_ONE, 1, "ch-beta-one-1")
    _failed_turn(store, BETA_ONE_A1_FAILED, experiment_id=BETA, task_id=TASK_ONE,
                 attempt=1)
    controller.finish_attempt(BETA, TASK_ONE, 1, outcome="fail",
                              outcome_source="test")

    # And turns that belong to no experiment at all.
    for turn_key in UNLABELLED:
        _plain_turn(store, turn_key)

    server = run_chatbot_server.ChatbotServer(
        db_path=db_path, workflow_path=str(folder), port=0,
        spawn_options={"no_server": True},
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield {"server": server, "store": store, "folder": str(folder)}
    server.shutdown()
    thread.join(timeout=5)


# The second world: an experiment REGISTERED against the workflow whose
# evidence lives in its own database, plus decoy turns in the workflow's
# default store carrying the very same experiment/task/attempt labels. The
# decoys are the point: a scoped read routed to the wrong store would answer
# with them, and an answer that merely looked plausible would pass.
EXTERNAL_A1 = ["turn-ext-a1-0", "turn-ext-a1-1"]
EXTERNAL_A2_FAILED = "turn-ext-a2-failed"
DEFAULT_DECOYS = ["turn-default-decoy-0", "turn-default-decoy-1"]


@pytest.fixture
def registered_world(tmp_path, monkeypatch):
    """One registered experiment in an external store, shadowed in the default."""
    monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "state"))
    folder = tmp_path / "registered_workflow"
    (folder / "_commands").mkdir(parents=True)
    benchmark = setup.save_benchmark(
        folder, {"title": "Registered", "tasks": [{"prompt": "Do it"}]}
    )
    registration = setup.create_experiment(
        folder, benchmark["benchmark_id"], "v1", runs_per_task=2
    )
    experiment_id = registration["experiment_id"]
    task_id = registration["task_ids"][0]

    external_db = str(tmp_path / "evidence-external.sqlite3")
    external = obs.ObservabilityStore(external_db)
    controller = ExperimentController(
        external_db, external.store_identity(), external=False,
        workflow_folderpath=str(folder),
    )
    controller.create_experiment(
        experiment_id, registration["description"], declared_tasks=1,
        declared_attempts=2,
        declarations=[(task_id, number, f"ch-ext-{number}") for number in (1, 2)],
        workflow_name=setup.workflow_name_for(folder),
    )
    controller.start_attempt(experiment_id, task_id, 1, "ch-ext-1")
    for turn_key in EXTERNAL_A1:
        _plain_turn(external, turn_key, experiment_id=experiment_id,
                    task_id=task_id, attempt=1)
    controller.finish_attempt(experiment_id, task_id, 1, outcome="pass",
                              outcome_source="test")
    controller.start_attempt(experiment_id, task_id, 2, "ch-ext-2")
    _failed_turn(external, EXTERNAL_A2_FAILED, experiment_id=experiment_id,
                 task_id=task_id, attempt=2)
    controller.finish_attempt(experiment_id, task_id, 2, outcome="fail",
                              outcome_source="test")

    # The workflow's own store, holding turns under the SAME labels. They are
    # written as turn rows rather than through a second controller because that
    # is what the collision is: two databases labelling different runs alike.
    default_db = state_paths.observability_db(str(folder))
    Path(default_db).parent.mkdir(parents=True, exist_ok=True)
    default_store = obs.ObservabilityStore(default_db)
    for turn_key in DEFAULT_DECOYS:
        _plain_turn(default_store, turn_key, experiment_id=experiment_id,
                    task_id=task_id, attempt=1)

    server = run_chatbot_server.ChatbotServer(
        db_path=default_db, workflow_path=str(folder), port=0,
        spawn_options={"no_server": True},
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield {"server": server, "experiment": experiment_id, "task": task_id,
           "folder": str(folder)}
    server.shutdown()
    thread.join(timeout=5)


def _turns(server, query: str) -> dict:
    status, data = _request(server, "/api/turns?" + query)
    assert status == 200, data
    return data


def _keys(page: dict) -> list[str]:
    return [row["turn_key"] for row in page["turns"]]


# ----------------------------------------------------------------------
# What a scope means over HTTP
# ----------------------------------------------------------------------


def test_an_experiment_scope_answers_about_that_experiment_only(finder_world):
    server = finder_world["server"]
    page = _turns(server, f"experiment={ALPHA}&limit=500")

    assert page["total_matched"] == BULK_TURNS + len(ALPHA_ONE_A2) + len(ALPHA_TWO_A1)
    assert page["total_matched_exact"] is True
    assert all(row["experiment_id"] == ALPHA for row in page["turns"])
    assert BETA_ONE_A1_FAILED not in _keys(page)
    assert not set(UNLABELLED) & set(_keys(page))


def test_two_experiments_recording_one_task_id_are_not_pooled(finder_world):
    """The collision the experiment filter exists for.

    Both experiments record `task-one`. A task scope that forgot the
    experiment answers about two populations at once, under one label.
    """
    server = finder_world["server"]
    both = _turns(server, f"task={TASK_ONE}&limit=500")
    scoped = _turns(server, f"experiment={BETA}&task={TASK_ONE}&limit=500")

    assert BETA_ONE_A1_FAILED in _keys(both)
    assert ALPHA_ONE_A2_FAILED in _keys(both)
    assert _keys(scoped) == [BETA_ONE_A1_FAILED]
    assert scoped["total_matched"] == 1


def test_an_attempt_scope_answers_about_one_attempt(finder_world):
    server = finder_world["server"]
    page = _turns(server, f"experiment={ALPHA}&task={TASK_ONE}&attempt=2&limit=500")

    assert sorted(_keys(page)) == sorted(ALPHA_ONE_A2)
    assert all(row["attempt"] == 2 for row in page["turns"])


def test_a_scoped_marker_search_counts_the_scope_and_not_the_store(finder_world):
    """Two turns in the store recorded a failure; one is in this scope.

    The facet is the chip's number, so a facet counted over the store under a
    scoped heading would read as "this attempt failed twice".
    """
    server = finder_world["server"]
    scoped = _turns(
        server,
        f"experiment={ALPHA}&task={TASK_ONE}&attempt=2"
        "&markers_any=step_unsuccessful&limit=500",
    )
    store_wide = _turns(server, "markers_any=step_unsuccessful&limit=500")

    assert _keys(scoped) == [ALPHA_ONE_A2_FAILED]
    assert scoped["facets"]["step_unsuccessful"] == 1
    assert sorted(_keys(store_wide)) == sorted([ALPHA_ONE_A2_FAILED, BETA_ONE_A1_FAILED])
    assert store_wide["facets"]["step_unsuccessful"] == 2
    # Read on the record basis inside the scope too: this failure has no span.
    assert scoped["basis"] == "spans+record"


def test_a_scoped_walk_returns_every_match_once_and_stays_scoped(finder_world):
    """Continuation under a scope, with the page's own limit and scan bound.

    205 turns in the attempt against a 200-turn scan bound and a 25-row page:
    the walk has to continue, and every segment has to carry the scope. A
    segment that dropped it would fetch other runs' turns and append them to a
    list headed by this attempt.
    """
    server = finder_world["server"]
    seen: list[str] = []
    summed = 0
    cursor = None
    for _ in range(50):
        query = f"experiment={ALPHA}&task={TASK_ONE}&attempt=1&limit=25&scan_limit=200"
        if cursor:
            query += "&resume_after=" + cursor
        page = _turns(server, query)
        assert all(row["experiment_id"] == ALPHA for row in page["turns"])
        assert all(row["attempt"] == 1 for row in page["turns"])
        seen.extend(_keys(page))
        summed += page["total_matched"]
        if not page["scan_truncated"]:
            assert page["next_scan_cursor"] is None
            break
        cursor = page["next_scan_cursor"]
        assert cursor
    else:
        pytest.fail("the scoped cursor never finished the attempt")

    assert len(seen) == BULK_TURNS, "every turn of the attempt, once"
    assert len(seen) == len(set(seen))
    assert summed == BULK_TURNS, "summed segment counts are the scope's total"
    assert seen == sorted(seen, reverse=True)


def test_a_scope_nothing_was_recorded_under_answers_empty_and_says_so(finder_world):
    """Empty because the scope is empty -- not because a filter was dropped."""
    server = finder_world["server"]
    page = _turns(server, "experiment=exp-never-run&limit=25")

    assert page["turns"] == []
    assert page["total_matched"] == 0
    assert page["scan_complete"] is True
    assert page["scan_truncated"] is False


def test_a_scope_value_that_is_not_an_attempt_number_is_refused(finder_world):
    """A filter that silently means something else is worse than one that fails."""
    server = finder_world["server"]
    status, data = _request(server, "/api/turns?attempt=two")

    assert status == 400
    assert "attempt" in data["error"]


def test_a_command_scope_means_the_turn_contained_it_not_that_it_failed(
    finder_world,
):
    """The epic's membership rule, at the route the finder's note describes.

    `turn-alpha-one-a2-0-ok` ran `list_todos` and succeeded; its neighbour in
    the same attempt recorded a failed `delete_todo`. A command filter returns
    turns that CONTAIN the command, so `command=list_todos` must return the
    successful turn and must not return the failed one, whose failure belongs
    to a different command entirely.
    """
    server = finder_world["server"]
    page = _turns(
        server,
        f"experiment={ALPHA}&task={TASK_ONE}&attempt=2&command=list_todos&limit=25",
    )

    assert _keys(page) == [ALPHA_ONE_A2[0]]
    assert ALPHA_ONE_A2_FAILED not in _keys(page)


# ----------------------------------------------------------------------
# The selected source answers its own scope
# ----------------------------------------------------------------------


def test_a_scope_on_the_selected_registered_source_reads_that_source(
    registered_world,
):
    """The acceptance the leaf is held to: UI and API select the SAME source.

    Both databases label turns with this experiment, this task and attempt 1.
    The scoped read carrying the selected source returns the registered
    store's turns; the same scope without it returns the workflow's own. A
    route that ignored the selection would answer with the decoys.
    """
    server = registered_world["server"]
    experiment = registered_world["experiment"]
    task = registered_world["task"]

    selected = _turns(
        server,
        f"benchmark_experiment={experiment}&experiment={experiment}"
        f"&task={task}&limit=50",
    )
    unselected = _turns(server, f"experiment={experiment}&task={task}&limit=50")

    assert sorted(_keys(selected)) == sorted(EXTERNAL_A1 + [EXTERNAL_A2_FAILED])
    assert not set(_keys(selected)) & set(DEFAULT_DECOYS)
    assert sorted(_keys(unselected)) == sorted(DEFAULT_DECOYS)


def test_an_attempt_scope_on_the_selected_source_reads_that_attempt(
    registered_world,
):
    server = registered_world["server"]
    experiment = registered_world["experiment"]
    task = registered_world["task"]

    first = _turns(
        server,
        f"benchmark_experiment={experiment}&experiment={experiment}"
        f"&task={task}&attempt=1&limit=50",
    )
    second = _turns(
        server,
        f"benchmark_experiment={experiment}&experiment={experiment}"
        f"&task={task}&attempt=2&markers_any=step_unsuccessful&limit=50",
    )

    assert sorted(_keys(first)) == sorted(EXTERNAL_A1)
    assert _keys(second) == [EXTERNAL_A2_FAILED]
    # The decoys share attempt 1 in the other database and stay out of both.
    assert not set(_keys(first)) & set(DEFAULT_DECOYS)


def test_a_registered_source_the_workflow_never_registered_is_refused(
    registered_world,
):
    """So a selection the page cannot honour fails closed, not quietly wrong."""
    server = registered_world["server"]
    status, data = _request(
        server,
        f"/api/turns?benchmark_experiment=exp-nobody"
        f"&experiment={registered_world['experiment']}&limit=25",
    )

    assert status == 409
    assert "exp-nobody" in data["error"]


# ----------------------------------------------------------------------
# What the page ships
# ----------------------------------------------------------------------


def test_the_page_ships_the_scope_control_and_its_entry_points():
    page = run_chatbot_server.load_index_html()
    for needle in [
        b'id="turnFindScope"',
        b"function turnFindScopeTo(scope)",
        b"function turnFindRenderScope()",
        b"function turnFindEntrySupported()",
        b"function turnFindEntryButton(container, scope, label, help)",
        b'"Find problems in this experiment"',
        b'"Find problems in this task"',
        b'"Find problems in this attempt"',
    ]:
        assert needle in page, needle


def test_the_page_sends_the_scope_it_displays():
    """Pinned beside the DOM test because the names are the route's.

    `experiment` / `task` / `attempt` are what `turn_query_from_params` reads;
    a client that invented its own names would be filtering nothing while
    showing a scope.
    """
    page = run_chatbot_server.load_index_html().decode("utf-8")
    body = page.split("function turnFindPath(cursor) {", 1)[1].split("\n}", 1)[0]
    for needle in [
        'params.push("experiment=" + encodeURIComponent(turnFind.scope.experiment));',
        'params.push("task=" + encodeURIComponent(turnFind.scope.task));',
        'params.push("attempt=" + encodeURIComponent(turnFind.scope.attempt));',
    ]:
        assert needle in body, needle


def test_the_page_never_claims_the_whole_store_under_a_scope():
    """A scoped walk covered the scope; the rest of the store was never read.

    Pinned in the source because a store smaller than one scope would let the
    wrong wording pass every behavioural test in this file.
    """
    page = run_chatbot_server.load_index_html().decode("utf-8")
    assert 'return prefix + (turnFind.scope ? "the whole of this scope" : "the whole store");' in page
    assert '? "No turn in this scope matches. The whole scope was searched."' in page


# ----------------------------------------------------------------------
# The real page, in a real DOM, against the real server
# ----------------------------------------------------------------------


def test_turn_finder_scope_dom(finder_world):
    dependency = os.environ.get("TEST_JSDOM_ROOT")
    if not dependency:
        pytest.skip("Set TEST_JSDOM_ROOT to run DOM integration with jsdom")
    server = finder_world["server"]
    script = Path(__file__).with_name("chatbot_finder_scope_dom.cjs")
    result = subprocess.run(
        [
            "node",
            str(script),
            dependency,
            f"http://127.0.0.1:{server.port}/?token={server.token}",
            json.dumps(
                {
                    "experiment": ALPHA,
                    "otherExperiment": BETA,
                    "task": TASK_ONE,
                    "otherTask": TASK_TWO,
                    "attemptTwoTurns": ALPHA_ONE_A2,
                    "failedTurn": ALPHA_ONE_A2_FAILED,
                    "otherFailedTurn": BETA_ONE_A1_FAILED,
                    "bulkTurns": BULK_TURNS,
                    "page": 25,
                }
            ),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_turn_finder_scope_on_a_selected_registered_source_dom(registered_world):
    """The same controls, driven against an external store the page has open."""
    dependency = os.environ.get("TEST_JSDOM_ROOT")
    if not dependency:
        pytest.skip("Set TEST_JSDOM_ROOT to run DOM integration with jsdom")
    server = registered_world["server"]
    script = Path(__file__).with_name("chatbot_finder_registered_source_dom.cjs")
    result = subprocess.run(
        [
            "node",
            str(script),
            dependency,
            f"http://127.0.0.1:{server.port}/?token={server.token}",
            json.dumps(
                {
                    "experiment": registered_world["experiment"],
                    "task": registered_world["task"],
                    "attemptOneTurns": EXTERNAL_A1,
                    "failedTurn": EXTERNAL_A2_FAILED,
                    "decoys": DEFAULT_DECOYS,
                }
            ),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr

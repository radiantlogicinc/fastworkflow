"""Experiments record into, and seal out of, the one live DB (fix-10vj.6).

Single live DB design §3, §5 and §6, end to end against real processes: a
registered experiment run by a separate process while another process keeps
writing the same live DB, the refusal when a run resolves a different DB from
the one its experiment was registered in, and the experiment-scoped seal.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sqlite3
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from fastworkflow import state_paths
from fastworkflow.benchmark import setup
from fastworkflow.experiment import ExperimentNotRegisteredHere
from fastworkflow.experiment.runner import ExperimentController, ExperimentHarness
from fastworkflow.observability import control
from fastworkflow.observability import store as obs
from fastworkflow.run_chatbot import selection_api
from fastworkflow.run_chatbot import server as run_chatbot_server
from fastworkflow.run_chatbot.turn_annotations import EVIDENCE_VALID
from tests.test_chatbot_benchmarks import _request
from tests.test_selection_api import _seed_turn

REPO = Path(__file__).resolve().parent.parent
TODO_WORKFLOW = REPO / "tests" / "todo_list_workflow"

# Another process on the same live DB: the interactive server, in production.
# It records ad-hoc turns through its own sink until told to stop.
_WRITER = textwrap.dedent(
    """
    import os, sys, time
    import fastworkflow
    from fastworkflow import TurnStatus
    from fastworkflow.observability import store as obs

    folder, stop = sys.argv[1:3]
    fastworkflow.init({})
    sink = obs.get_observability_sink(folder)
    written = 0
    print("ready", flush=True)
    while not os.path.exists(stop):
        sink.emit_turn_record(fastworkflow.TurnResult(
            turn_output=fastworkflow.TurnOutput(
                turn_key=fastworkflow.mint_turn_key(),
                status=TurnStatus.COMPLETED,
                answer=f"ad-hoc {written}",
                command_outputs=[],
            ),
            channel_id="interactive",
            conversation_id=1,
            user_message=f"ad-hoc {written}",
        ))
        written += 1
        time.sleep(0.01)
    obs.close_all_sinks()
    print(written, flush=True)
    """
)

# A separate-process experiment run: what a driver such as a benchmark harness does. Commands
# answer deterministically at `CommandExecutor.invoke_command`, the boundary
# `tests/test_experiment_container.py::deterministic_commands` uses; the
# workflow, the WEC, the sink, the evidence run and the seal are all real.
_RUNNER = textwrap.dedent(
    """
    import json, sys
    import fastworkflow
    from fastworkflow.command_executor import CommandExecutor
    from fastworkflow.experiment.runner import ExperimentHarness, ExperimentTask

    def invoke(cls, session, command):
        return fastworkflow.CommandOutput(
            command_name=command.split()[0] if command else "",
            command_response=fastworkflow.CommandResponse(response="ok:" + command),
        )

    CommandExecutor.invoke_command = classmethod(invoke)
    folder, experiment_id, task_id, bundle = sys.argv[1:5]
    fastworkflow.init({})
    harness = ExperimentHarness.from_benchmark_experiment(
        folder, experiment_id, run_as_agent=False, archive_dir=bundle
    )
    result = harness.run(
        [ExperimentTask(task_id=task_id, messages=["add milk"])],
        grader=lambda run: ("pass", "grader", 1.0, None),
    )
    print(json.dumps({key: result[key] for key in
                      ("status", "evidence_valid", "evidence_problems")}))
    """
)


def _start_writer(folder: str, stop: Path) -> subprocess.Popen:
    writer = subprocess.Popen(
        [sys.executable, "-c", _WRITER, folder, str(stop)],
        cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    assert writer.stdout.readline().strip() == "ready", writer.stderr.read()
    return writer


def _ad_hoc_turns(db_path: str) -> int:
    with sqlite3.connect(db_path) as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM turns WHERE experiment_id IS NULL"
        ).fetchone()[0]


def _await_ad_hoc_turns(db_path: str, more_than: int) -> int:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if os.path.exists(db_path) and (count := _ad_hoc_turns(db_path)) > more_than:
            return count
        time.sleep(0.05)
    raise AssertionError("the other writer process recorded nothing")


def _stop_writer(writer: subprocess.Popen, stop: Path) -> int:
    stop.touch()
    out, err = writer.communicate(timeout=60)
    assert writer.returncode == 0, err
    return int(out.split()[-1])


def _files(root: Path, pattern: str) -> list[Path]:
    return sorted(root.rglob(pattern))


# ----------------------------------------------------------------------
# A separate-process experiment on a live DB another process is writing
# ----------------------------------------------------------------------


def test_a_registered_experiment_run_elsewhere_lands_in_every_view_of_the_live_db(
    tmp_path,
):
    folder = tmp_path / "todo_list_workflow"
    shutil.copytree(TODO_WORKFLOW, folder)
    state_root = Path(os.environ["FASTWORKFLOW_STATE_ROOT"])
    db_path = state_paths.observability_db(str(folder))
    benchmark = setup.save_benchmark(
        folder, {"title": "Shopping", "tasks": [{"prompt": "add milk"}]}
    )
    registered = setup.create_experiment(folder, benchmark["benchmark_id"], "v1")
    experiment_id, task_id = registered["experiment_id"], registered["task_ids"][0]

    stop = tmp_path / "stop"
    writer = _start_writer(str(folder), stop)
    try:
        before = _await_ad_hoc_turns(db_path, 0)
        run = subprocess.run(
            [sys.executable, "-c", _RUNNER, str(folder), experiment_id, task_id,
             str(tmp_path / "bundle")],
            cwd=REPO, capture_output=True, text=True, timeout=600,
        )
        assert run.returncode == 0, run.stdout + run.stderr
        assert _await_ad_hoc_turns(db_path, before) > before
    finally:
        written = _stop_writer(writer, stop)

    result = json.loads(run.stdout.strip().splitlines()[-1])
    assert result == {
        "status": "complete", "evidence_valid": True, "evidence_problems": [],
    }
    assert _ad_hoc_turns(db_path) == written

    # The views read the live DB the run wrote, with no bootstrap in between.
    store = obs.ReadOnlyObservabilityStore(db_path)
    turns = store.list_turns(experiment_id=experiment_id, limit=50)
    assert turns and {turn["task_id"] for turn in turns} == {task_id}
    [segment] = store.get_experiment(experiment_id)["evidence_runs"]
    assert segment["valid"] == 1

    status, winner = selection_api.handle_get(
        str(folder), f"/api/experiments/{experiment_id}/winner", {}
    )
    assert status == 200 and winner["is_winner"] is True
    status, runs = selection_api.handle_get(
        str(folder), f"/api/experiments/{experiment_id}/tasks/{task_id}/runs", {}
    )
    assert status == 200 and runs["evidence_readable"] is True
    assert runs["comparable_attempts"] == [1]

    server = run_chatbot_server.ChatbotServer(
        db_path=db_path, workflow_path=str(folder), port=0,
        spawn_options={"no_server": True},
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, body = _request(server, f"/api/experiment/{experiment_id}")
        assert status == 200
        assert body["experiment"]["status"] == "complete"
        assert body["experiment"]["evidence"]["state"] == EVIDENCE_VALID
        status, body = _request(server, f"/api/experiment/{experiment_id}/attempts")
        assert status == 200
        assert [row["task_id"] for row in body["attempts"]] == [task_id]
    finally:
        server.shutdown()
        thread.join(timeout=5)

    # One live DB, and nothing beside it: no per-run store, no control file.
    assert _files(state_root, "*.sqlite3") == [Path(db_path)]
    assert _files(state_root, "*control*") == []
    assert _files(folder, "*.sqlite3") == []


# ----------------------------------------------------------------------
# A run that resolves another DB than its experiment was registered in
# ----------------------------------------------------------------------


def test_a_registered_experiment_under_another_state_root_is_refused_by_name(
    tmp_path, monkeypatch
):
    folder = tmp_path / "workflow"
    benchmark = setup.save_benchmark(folder, {"title": "x", "tasks": [{}]})
    experiment_id = setup.create_experiment(
        folder, benchmark["benchmark_id"], "v1"
    )["experiment_id"]

    elsewhere = tmp_path / "another-root"
    monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(elsewhere))
    resolved = state_paths.observability_db(str(folder))
    with pytest.raises(ExperimentNotRegisteredHere) as refused:
        ExperimentHarness.from_benchmark_experiment(str(folder), experiment_id)
    message = str(refused.value)
    assert experiment_id in message and resolved in message
    assert f"FASTWORKFLOW_STATE_ROOT={elsewhere}" in message
    assert "FASTWORKFLOW_WORKFLOW_ID=<unset>" in message

    store = obs.ObservabilityStore(resolved)
    controller = ExperimentController(str(folder), store.store_identity(), external=False)
    with pytest.raises(ExperimentNotRegisteredHere, match="FASTWORKFLOW_STATE_ROOT"):
        controller.create_experiment(
            experiment_id, "late", declared_tasks=1, declared_attempts=1,
            declarations=[("task-1", 1, "native-1")], registered=True,
        )
    assert store.get_experiment(experiment_id) is None


def test_an_ad_hoc_run_under_an_overridden_root_warns_once(tmp_path, caplog):
    folder = str(tmp_path / "workflow")
    store = obs.ObservabilityStore(state_paths.observability_db(folder))
    controller = ExperimentController(folder, store.store_identity(), external=False)

    with caplog.at_level(logging.WARNING):
        controller.create_experiment(
            "ad-hoc", "ad hoc", declared_tasks=1, declared_attempts=1,
            declarations=[("task-1", 1, "native-1")],
        )
    [warning] = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert controller.db_path in warning.getMessage()
    assert "FASTWORKFLOW_STATE_ROOT" in warning.getMessage()
    assert store.get_experiment("ad-hoc") is not None


# ----------------------------------------------------------------------
# The experiment-scoped seal
# ----------------------------------------------------------------------


@pytest.fixture
def live(tmp_path, monkeypatch):
    """Two finished experiments and an ad-hoc turn in one live DB."""
    folder = str(tmp_path / "workflow")
    db_path = state_paths.observability_db(folder)
    store = obs.ObservabilityStore(db_path)
    controller = ExperimentController(folder, store.store_identity(), external=True)
    for experiment_id in ("exp-1", "exp-2"):
        controller.create_experiment(
            experiment_id, experiment_id, declared_tasks=1, declared_attempts=1,
            declarations=[("task-1", 1, f"native-{experiment_id}")],
            required_evidence_segments=1,
        )
        channel = f"channel-{experiment_id}"
        conversation = store.mint_conversation_id(
            channel, experiment_id=experiment_id, task_id="task-1", attempt=1
        )
        controller.start_attempt(
            experiment_id, "task-1", 1, channel,
            source_key=f"native-{experiment_id}", conversation_id=conversation,
        )
        _seed_turn(
            store, f"{experiment_id}-turn", experiment_id=experiment_id,
            task_id="task-1", attempt=1, conversation=conversation, ordinal=1,
            commands=["add_item", "list_items"],
        )
        controller.finish_attempt(
            experiment_id, "task-1", 1, outcome="pass", outcome_source="grader"
        )
        controller.record_evidence_segment(experiment_id, 1, "run-1", {
            "valid": True, "problems": [],
            "started_at": "2026-10-04T00:00:00+00:00",
            "completed_at": "2026-10-04T00:01:00+00:00",
        })
        assert controller.complete_experiment(experiment_id) == "capture_complete"
    _seed_turn(
        store, "ad-hoc-turn", experiment_id=None, task_id=None, attempt=None,
        conversation=None, ordinal=None, commands=["add_item"],
    )
    return {"folder": folder, "db_path": db_path, "controller": controller}


def _sealed(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)


def _experiment_rows(conn: sqlite3.Connection, experiment_id: str) -> dict:
    return {
        table: conn.execute(
            f"SELECT * FROM {table} WHERE experiment_id=? ORDER BY 1, 2",
            (experiment_id,),
        ).fetchall()
        for table in ("experiments", "experiment_attempts", "turns", "conversations")
    } | {
        "spans": conn.execute(
            "SELECT * FROM spans WHERE trace_id IN "
            "(SELECT turn_key FROM turns WHERE experiment_id=?) ORDER BY 1",
            (experiment_id,),
        ).fetchall()
    }


def test_an_experiment_seal_holds_that_experiment_only_and_no_control_table(
    live, tmp_path
):
    archive_path = tmp_path / "exp-1.sqlite3"
    archive = live["controller"].seal_workspace_evidence("exp-1", str(archive_path))
    assert archive["consistent_snapshot"] is True
    assert archive["experiment_id"] == "exp-1"

    with _sealed(archive_path) as conn:
        for table in ("experiments", "experiment_attempts", "turns", "conversations"):
            assert {row[0] for row in conn.execute(
                f"SELECT DISTINCT experiment_id FROM {table}"
            )} == {"exp-1"}, table
        assert {row[0] for row in conn.execute(
            "SELECT DISTINCT trace_id FROM spans"
        )} == {"exp-1-turn"}
        tables = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        features = json.loads(conn.execute(
            "SELECT value FROM diagnostics WHERE key='schema_features'"
        ).fetchone()[0])
        assert conn.execute("PRAGMA user_version").fetchone()[0] == obs.SCHEMA_VERSION
        assert conn.execute(
            "SELECT COUNT(*) FROM diagnostics WHERE key GLOB 'writer_health*'"
        ).fetchone()[0] == 0
    assert tables.isdisjoint(control.CONTROL_TABLES)
    assert control.FEATURE_CONTROL_V1 not in features

    # The live DB still holds everything, and the archive opens as a reader.
    with sqlite3.connect(live["db_path"]) as conn:
        assert {row[0] for row in conn.execute("SELECT turn_key FROM turns")} == {
            "exp-1-turn", "exp-2-turn", "ad-hoc-turn",
        }
    reader = obs.ReadOnlyObservabilityStore(str(archive_path))
    assert reader.get_experiment("exp-1")["status"] == "complete"
    assert reader.get_experiment("exp-2") is None
    assert [t["turn_key"] for t in reader.list_turns(limit=50)] == ["exp-1-turn"]


def test_a_seal_taken_while_another_process_writes_is_consistent(live, tmp_path):
    stop = tmp_path / "stop"
    writer = _start_writer(live["folder"], stop)
    try:
        before = _await_ad_hoc_turns(live["db_path"], 1)
        archive_path = tmp_path / "exp-1.sqlite3"
        archive = live["controller"].seal_workspace_evidence(
            "exp-1", str(archive_path)
        )
        assert _await_ad_hoc_turns(live["db_path"], before) > before
    finally:
        _stop_writer(writer, stop)

    assert archive["sealed"] is True and archive["consistent_snapshot"] is True
    with _sealed(archive_path) as sealed, sqlite3.connect(live["db_path"]) as source:
        assert sealed.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        # The experiments row is left out: the digest lands on the live row
        # after the archive is taken, by design (fix-tcg).
        expected = _experiment_rows(source, "exp-1")
        copied = _experiment_rows(sealed, "exp-1")
        assert copied["turns"] == expected["turns"] and copied["turns"]
        assert copied["spans"] == expected["spans"]
        assert copied["experiment_attempts"] == expected["experiment_attempts"]
        assert sealed.execute(
            "SELECT COUNT(*) FROM turns WHERE experiment_id IS NULL"
        ).fetchone()[0] == 0

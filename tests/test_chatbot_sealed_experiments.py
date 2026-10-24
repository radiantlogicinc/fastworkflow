"""Seal, browse and delete benchmark experiments from the chatbot (fix-ajir.24)."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import stat
import subprocess
from pathlib import Path

import pytest

from fastworkflow import state_paths
from fastworkflow.benchmark import setup
from fastworkflow.experiment.runner import ExperimentController
from fastworkflow.observability import control
from fastworkflow.observability import store as obs
from tests.test_chatbot_benchmarks import _request, experiment_server, workflow_dir  # noqa: F401
from tests.test_chatbot_hierarchy import add_turn


def _sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _rows(db_path, sql, params=()):
    with sqlite3.connect(db_path) as conn:
        return conn.execute(sql, params).fetchall()


def _recorded(server, store, *, complete=True, turn="exp-turn"):
    """One registered experiment, one attempt, one turn with a span and an artifact."""
    spec = setup.save_benchmark(
        server.workflow_path,
        {"title": "Sealing benchmark", "description": "", "tasks": [{"prompt": "Do it"}]},
    )
    registration = setup.create_experiment(server.workflow_path, spec["benchmark_id"], "v1")
    eid = registration["experiment_id"]
    task = registration["task_ids"][0]
    controller = ExperimentController(
        server.workflow_path, store.store_identity(), external=False
    )
    controller.create_experiment(
        eid, "Recorded", declared_tasks=1, declared_attempts=1,
        declarations=[(task, 1, "src")], registered=True,
    )
    store.start_attempt(eid, task, 1, "chan-exp", source_key="src")
    add_turn(store, turn, eid=eid, channel="chan-exp")
    with store._connect() as conn:
        conn.execute("UPDATE turns SET task_id=? WHERE turn_key=?", (task, turn))
        conn.execute(
            "INSERT INTO spans(span_id,trace_id,name,kind,start_ns,end_ns,status,"
            "attributes,experiment_id) VALUES(?,?,?,?,?,?,?,?,?)",
            (f"span-{turn}", turn, "fw.planner.plan", "internal", 1, 1000, "ok", "{}", eid),
        )
        conn.execute(
            "INSERT INTO artifacts(artifact_id,turn_key,key,content_type,inline_value,"
            "experiment_id) VALUES(?,?,?,?,?,?)",
            (f"art-{turn}", turn, "answer", "text/plain", b"sealed payload", eid),
        )
    if complete:
        store.finish_attempt(eid, task, 1, outcome="pass", outcome_source="test")
        assert store.complete_experiment(eid) == "complete"
    return eid


@pytest.fixture
def sealed_server(experiment_server):  # noqa: F811
    server, store = experiment_server
    add_turn(store, "plain-turn")
    eid = _recorded(server, store)
    return server, store, eid


def _seal(server, eid):
    return _request(server, "/api/archived-experiments", method="POST", body={"experiment_id": eid})


def test_seal_writes_a_read_only_archive_frees_live_rows_and_reads_from_it(sealed_server):
    server, store, eid = sealed_server
    status, listing = _request(server, "/api/archived-experiments")
    assert status == 200
    assert eid in [row["experiment_id"] for row in listing["ready"]]
    assert listing["archived"] == []

    status, sealed = _seal(server, eid)
    assert status == 201, sealed
    path = sealed["archive"]["path"]
    assert os.path.dirname(path) == state_paths.archives_dir(server.workflow_path)
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o444
    assert sealed["archive"]["sha256"] == _sha256(path)
    assert sealed["freed"]["turns"] == 1 and sealed["freed"]["spans"] == 1
    assert sealed["freed"]["artifacts"] == 1 and sealed["warning"] is None

    # The archive file is the only copy. The live DB keeps the other conversation.
    db = store.db_path
    assert _rows(db, "SELECT turn_key FROM turns") == [("plain-turn",)]
    assert _rows(db, "SELECT COUNT(*) FROM spans WHERE experiment_id=?", (eid,)) == [(0,)]
    assert _rows(db, "SELECT COUNT(*) FROM artifacts") == [(0,)]
    assert store.get_experiment(eid) is None
    assert _rows(db, "SELECT COUNT(*) FROM experiments WHERE experiment_id=?", (eid,)) == [(0,)]
    assert _rows(db, "SELECT COUNT(*) FROM sealed_archives WHERE experiment_id=?", (eid,)) == [(0,)]
    assert _rows(
        db,
        "SELECT COUNT(*) FROM conversation_counters WHERE channel_id LIKE ?",
        (f"exp:{eid}:%",),
    ) == [(0,)]

    status, listing = _request(server, "/api/archived-experiments")
    [row] = listing["archived"]
    assert row["experiment_id"] == eid and row["readable"] is True
    assert row["benchmark_version"] == "v1" and row["size_bytes"] == os.path.getsize(path)
    assert "archive_sha256" not in row
    assert listing["scan_seconds"] >= 0
    assert eid not in [r["experiment_id"] for r in listing["ready"]]

    # Opened the same way as a live experiment: every read finds the archive.
    status, detail = _request(server, f"/api/experiment/{eid}")
    assert status == 200 and detail["experiment"]["archive"]["path"] == path
    status, score = _request(server, f"/api/experiment/{eid}/score")
    assert status == 200 and score["score"]["reportable"] is True
    status, attempts = _request(server, f"/api/experiment/{eid}/attempts")
    assert status == 200 and len(attempts["attempts"]) == 1
    status, turns = _request(server, f"/api/turns?experiment={eid}")
    assert [t["turn_key"] for t in turns["turns"]] == ["exp-turn"]
    status, turn = _request(server, "/api/turn/exp-turn")
    assert status == 200 and turn["turn"]["experiment_id"] == eid
    status, spans = _request(server, "/api/spans/exp-turn")
    assert [s["span_id"] for s in spans["spans"]] == ["span-exp-turn"]
    url = f"http://127.0.0.1:{server.port}/api/artifact/art-exp-turn"
    import urllib.request

    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {server.token}"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        assert resp.read() == b"sealed payload"
    status, _ = _request(server, "/api/turn/plain-turn")
    assert status == 200
    # The run-selection and comparison routes read the archive too.
    task = attempts["attempts"][0]["task_id"]
    status, runs = _request(server, f"/api/experiments/{eid}/tasks/{task}/runs")
    assert status == 200, runs
    status, passes = _request(server, f"/api/experiments/{eid}/tasks/{task}/runs/1/passes")
    assert status == 200, passes
    # Reading never touched the sealed file.
    assert _sha256(path) == sealed["archive"]["sha256"]
    assert not os.path.exists(path + "-wal")


def test_seal_refusals(sealed_server, monkeypatch):
    server, store, eid = sealed_server
    assert _seal(server, "exp-missing")[0] == 404
    assert _request(server, "/api/archived-experiments", method="POST", body={})[0] == 400
    running = _recorded(server, store, complete=False, turn="running-turn")
    status, data = _seal(server, running)
    assert status == 409 and "running" in data["error"]

    # A writer this process holds refuses the seal before the row is touched.
    with monkeypatch.context() as patched:
        patched.setattr(obs, "sink_for_db_path", lambda _path: object())
        status, data = _seal(server, eid)
    assert status == 409 and "writer" in data["error"]
    assert store.get_experiment(eid)["evidence_sealed_at"] is None

    assert _seal(server, eid)[0] == 201
    status, data = _seal(server, eid)
    assert status == 409 and "already archived" in data["error"]


def test_delete_is_refused_while_selected_then_removes_file_and_rows(sealed_server):
    server, store, eid = sealed_server
    status, sealed = _seal(server, eid)
    path = sealed["archive"]["path"]
    group = "group-1"
    with control.write(store) as conn:
        conn.execute(
            "INSERT INTO selection_pointers(scope_kind,group_id,scope_key,experiment_id,"
            "selection_id,decision,decision_seq,decided_at) VALUES(?,?,?,?,?,?,?,?)",
            ("experiment", group, "", eid, "sel-1", "set", 1, "2026-10-22T00:00:00Z"),
        )
    status, listing = _request(server, "/api/archived-experiments")
    assert listing["archived"][0]["selected_as"] == "experiment"
    status, data = _request(server, f"/api/archived-experiments/{eid}", method="DELETE")
    assert status == 409 and "winner" in data["error"]
    assert os.path.exists(path)

    with control.write(store) as conn:
        conn.execute("DELETE FROM selection_pointers")
    status, data = _request(server, f"/api/archived-experiments/{eid}", method="DELETE")
    assert status == 200, data
    assert data["file_removed"] is True and not os.path.exists(path)
    for table in ("experiments", "experiment_attempts", "sealed_archives",
                  "comparison_group_members"):
        assert _rows(
            store.db_path, f"SELECT COUNT(*) FROM {table} WHERE experiment_id=?", (eid,)
        ) == [(0,)], table
    assert _rows(
        store.db_path,
        "SELECT state FROM experiment_registrations WHERE experiment_id=?", (eid,),
    ) == []
    assert _request(server, "/api/turn/exp-turn")[0] == 404
    assert _request(server, "/api/archived-experiments")[1]["archived"] == []
    assert _request(server, f"/api/archived-experiments/{eid}", method="DELETE")[0] == 404


def test_release_closes_a_running_experiment_and_lets_prune_reach_it(sealed_server):
    server, store, eid = sealed_server
    running = _recorded(server, store, complete=False, turn="running-turn")
    status, listing = _request(server, "/api/archived-experiments")
    stuck = {row["experiment_id"]: row for row in listing["unarchivable"]}
    assert stuck[running]["status"] == "running"
    assert running not in [row["experiment_id"] for row in listing["released"]]

    status, data = _request(
        server, "/api/evidence-releases", method="POST", body={"experiment_id": running}
    )
    assert status == 200 and data == {
        "experiment_id": running, "status": "invalid", "released": True}
    assert store.get_experiment(running)["invalid_reason"] == "operator"
    assert _rows(
        store.db_path, "SELECT actor FROM evidence_releases WHERE experiment_id=?", (running,)
    ) == [("observability UI",)]
    listing = _request(server, "/api/archived-experiments")[1]
    assert running not in [r["experiment_id"] for r in listing["unarchivable"]]
    assert [r["experiment_id"] for r in listing["released"]] == [running]

    status, data = _request(
        server, "/api/evidence-releases", method="POST", body={"experiment_id": eid}
    )
    assert status == 409 and "seal it" in data["error"]
    assert _request(
        server, "/api/evidence-releases", method="POST", body={"experiment_id": "nope"}
    )[0] == 404


def test_delete_released_removes_rows_and_refuses_until_released(sealed_server):
    server, store, eid = sealed_server
    running = _recorded(server, store, complete=False, turn="running-turn")
    status, data = _request(
        server, f"/api/released-experiments/{running}", method="DELETE"
    )
    assert status == 409 and "has not been released" in data["error"]
    assert store.get_experiment(running)["status"] == "running"

    status, data = _request(
        server, "/api/evidence-releases", method="POST", body={"experiment_id": running}
    )
    assert status == 200 and data["released"] is True
    group = _rows(
        store.db_path,
        "SELECT group_id FROM comparison_group_members WHERE experiment_id=?",
        (running,),
    )[0][0]
    with control.write(store) as conn:
        conn.execute(
            "INSERT INTO selection_pointers(scope_kind,group_id,scope_key,experiment_id,"
            "selection_id,decision,decision_seq,decided_at) VALUES(?,?,?,?,?,?,?,?)",
            ("experiment", group, "", running, "sel-rel", "set", 1, "2026-10-23T00:00:00Z"),
        )
    status, listing = _request(server, "/api/archived-experiments")
    assert listing["released"][0]["selected_as"] == "experiment"
    status, data = _request(server, f"/api/released-experiments/{running}", method="DELETE")
    assert status == 409 and "winner" in data["error"]
    assert os.path.exists(store.db_path)
    assert store.get_experiment(running) is not None

    with control.write(store) as conn:
        conn.execute("DELETE FROM selection_pointers")
    status, data = _request(server, f"/api/released-experiments/{running}", method="DELETE")
    assert status == 200, data
    assert data["experiment_id"] == running
    for table in ("experiments", "experiment_attempts", "evidence_releases",
                  "comparison_group_members"):
        assert _rows(
            store.db_path, f"SELECT COUNT(*) FROM {table} WHERE experiment_id=?", (running,)
        ) == [(0,)], table
    assert _rows(
        store.db_path,
        "SELECT state FROM experiment_registrations WHERE experiment_id=?", (running,),
    ) == []
    assert _request(server, "/api/turn/running-turn")[0] == 404
    assert _request(server, "/api/turn/plain-turn")[0] == 200
    assert _request(server, "/api/archived-experiments")[1]["released"] == []
    assert _request(server, f"/api/released-experiments/{running}", method="DELETE")[0] == 404

    status, sealed = _seal(server, eid)
    assert status == 201, sealed
    path = sealed["archive"]["path"]
    status, data = _request(server, f"/api/released-experiments/{eid}", method="DELETE")
    assert status == 409 and "is archived" in data["error"]
    assert os.path.exists(path)
    assert _request(server, "/api/archived-experiments")[1]["archived"][0]["experiment_id"] == eid


def test_sealed_experiments_dom(sealed_server):
    dependency = os.environ.get("TEST_JSDOM_ROOT")
    if not dependency:
        pytest.skip("Set TEST_JSDOM_ROOT to run DOM integration with jsdom")
    server, _store, eid = sealed_server
    script = Path(__file__).with_name("chatbot_sealed_experiments_dom.cjs")
    result = subprocess.run(
        ["node", str(script), dependency,
         f"http://127.0.0.1:{server.port}/?token={server.token}", eid],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_sealed_bulk_actions_dom(sealed_server):
    dependency = os.environ.get("TEST_JSDOM_ROOT")
    if not dependency:
        pytest.skip("Set TEST_JSDOM_ROOT to run DOM integration with jsdom")
    server, store, first = sealed_server
    second = _recorded(server, store, turn="exp-turn-2")
    invalid = _recorded(server, store, complete=False, turn="invalid-turn")
    assert store.complete_experiment(
        invalid, force_invalid="operator", detail="closed for the test"
    ) == "invalid"
    script = Path(__file__).with_name("chatbot_sealed_bulk_dom.cjs")
    result = subprocess.run(
        ["node", str(script), dependency,
         f"http://127.0.0.1:{server.port}/?token={server.token}",
         first, second, invalid],
        capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr

"""HTTP tests for chatbot benchmark catalog routes (`fix-42b.3`)."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from fastworkflow.benchmark import setup as benchmark_setup
from fastworkflow.observability import store as obs
from fastworkflow import state_paths
from fastworkflow.benchmark.catalog import (
    SCHEMA,
    benchmarks_root,
    load_version,
    write_version,
)
from fastworkflow.run_chatbot import server as run_chatbot_server


def _turn_row(turn_key: str, experiment_id: str, task_id: str, attempt: int) -> dict:
    """One completed turn row, labelled with an experiment, task and attempt."""
    return {
        "turn_key": turn_key,
        "channel_id": f"channel-{task_id}",
        "conversation_id": None,
        "ordinal": None,
        "user_message": f"run {task_id}",
        "refined_user_message": None,
        "entry_workflow_name": "test-workflow",
        "entry_context": "test",
        "status": "completed",
        "success": 1,
        "failure_reason": None,
        "answer": "done",
        "conversation_summary": None,
        "conversation_traces": None,
        "started_at": "2026-09-04T00:00:00+00:00",
        "completed_at": "2026-09-04T00:00:01+00:00",
        "suspended_ms": 0,
        "continuation_of": None,
        "record_version": 1,
        "experiment_id": experiment_id,
        "task_id": task_id,
        "attempt": attempt,
        "claim_epoch": None,
        "server_incarnation": None,
        "record_json": json.dumps(
            {"turn_output": {"turn_key": turn_key, "success": True}}
        ),
    }


def _sample_spec(
    *,
    benchmark_id: str = "smoke",
    version: str = "v1",
    task_id: str = "case-01",
) -> dict:
    return {
        "benchmark_id": benchmark_id,
        "version": version,
        "description": "what this corpus claims to test",
        "tasks": [
            {
                "task_id": task_id,
                "description": "human/agent one-liner",
                "payload": {"nested": {"keep": "keys"}},
            }
        ],
    }


def _request(server, path, method="GET", body=None, token=...):
    if token is ...:
        token = server.token
    url = f"http://127.0.0.1:{server.port}{path}"
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, method=method, data=data)
    if token is not None:
        req.add_header("Authorization", f"Bearer {token}")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as err:
        raw = err.read()
        try:
            return err.code, json.loads(raw or b"{}")
        except ValueError:
            return err.code, {"raw": raw}


@pytest.fixture
def workflow_dir(tmp_path) -> Path:
    workflow = tmp_path / "workflow"
    workflow.mkdir()
    (workflow / "_commands").mkdir()
    write_version(workflow, _sample_spec())
    return workflow


@pytest.fixture
def live_server(workflow_dir):
    srv = run_chatbot_server.ChatbotServer(
        db_path="",
        workflow_path=str(workflow_dir),
        port=0,
        spawn_options={"no_server": True},
    )
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    thread.join(timeout=5)


@pytest.fixture
def experiment_server(workflow_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "state"))
    db_path = state_paths.observability_db(str(workflow_dir))
    store = obs.ObservabilityStore(db_path)
    store.create_experiment(
        "exp-1",
        "label-exp-1",
        declared_tasks=1,
        declared_attempts=1,
    )
    store.update_experiment_notes("exp-1", "original notes")
    srv = run_chatbot_server.ChatbotServer(
        db_path=db_path,
        workflow_path=str(workflow_dir),
        port=0,
        spawn_options={"no_server": True},
    )
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv, store
    srv.shutdown()
    thread.join(timeout=5)


class TestBenchmarkReadApi:
    def test_list_benchmarks(self, live_server):
        status, data = _request(live_server, "/api/benchmarks")
        assert status == 200
        assert data["benchmarks"] == [
            {"benchmark_id": "smoke", "versions": ["v1"]}
        ]

    def test_benchmark_detail_and_version(self, live_server):
        status, data = _request(live_server, "/api/benchmarks/smoke")
        assert status == 200
        assert data == {"benchmark_id": "smoke", "versions": ["v1"]}

        status, data = _request(live_server, "/api/benchmarks/smoke/versions/v1")
        assert status == 200
        version = data["version"]
        assert version["schema"] == SCHEMA
        assert version["benchmark_id"] == "smoke"
        assert version["version"] == "v1"
        assert version["tasks"][0]["task_id"] == "case-01"
        assert version["tasks"][0]["payload"] == {"nested": {"keep": "keys"}}
        assert len(version["digest_sha256"]) == 64

    def test_missing_version_is_404(self, live_server):
        status, data = _request(live_server, "/api/benchmarks/smoke/versions/v9")
        assert status == 404
        assert "not found" in data["error"]

    def test_experiments_are_listed_newest_first(self, live_server, workflow_dir):
        # The contest lives in the live DB, which registration never creates.
        obs.ObservabilityStore(state_paths.observability_db(str(workflow_dir)))
        older = benchmark_setup.create_experiment(workflow_dir, "smoke", "v1")
        newer = benchmark_setup.create_experiment(workflow_dir, "smoke", "v1")

        status, data = _request(live_server, "/api/benchmarks/smoke/experiments")

        assert status == 200
        assert [row["experiment_id"] for row in data["experiments"]] == [
            newer["experiment_id"],
            older["experiment_id"],
        ]
        # The first experiment holds the contest until somebody promotes
        # another. The list stays newest-first; the pointer is a separate fact.
        by_id = {row["experiment_id"]: row for row in data["experiments"]}
        assert data["winner_experiment_id"] == older["experiment_id"]
        assert data["winner_automatic"] is True
        assert by_id[older["experiment_id"]]["is_winner"] is True
        assert by_id[newer["experiment_id"]]["is_winner"] is False


class TestBenchmarkWriteApi:
    def test_post_version_round_trip(self, live_server, workflow_dir):
        spec = _sample_spec(version="v2", task_id="case-02")
        status, data = _request(
            live_server,
            "/api/benchmarks/smoke/versions",
            method="POST",
            body=spec,
        )
        assert status == 201
        assert data["version"]["version"] == "v2"
        assert data["version"]["digest_sha256"]

        status, listed = _request(live_server, "/api/benchmarks/smoke")
        assert status == 200
        assert listed["versions"] == ["v1", "v2"]

        on_disk = json.loads(
            (benchmarks_root(workflow_dir) / "smoke" / "v2.json").read_text(
                encoding="utf-8"
            )
        )
        assert on_disk["version"] == "v2"
        assert on_disk["tasks"][0]["task_id"] == "case-02"

    def test_post_root_creates_first_version_for_new_benchmark(
        self, live_server, workflow_dir
    ):
        spec = _sample_spec(benchmark_id="alpha", version="v1", task_id="alpha-01")
        status, data = _request(
            live_server,
            "/api/benchmarks",
            method="POST",
            body=spec,
        )
        assert status == 201
        assert data["version"]["benchmark_id"] == "alpha"

        status, listed = _request(live_server, "/api/benchmarks")
        assert status == 200
        assert {row["benchmark_id"] for row in listed["benchmarks"]} == {
            "alpha",
            "smoke",
        }
        assert (benchmarks_root(workflow_dir) / "alpha" / "v1.json").is_file()

    def test_duplicate_version_is_409(self, live_server):
        spec = _sample_spec(version="v1")
        status, data = _request(
            live_server,
            "/api/benchmarks/smoke/versions",
            method="POST",
            body=spec,
        )
        assert status == 409
        assert "immutable" in data["error"]


class TestBenchmarkAnalysisApi:
    def test_put_get_analysis_round_trip(self, live_server, workflow_dir):
        payload = {"findings": ["latency on case-01"], "nested": {"a": 1}}
        status, data = _request(
            live_server,
            "/api/benchmarks/smoke/analysis",
            method="PUT",
            body=payload,
        )
        assert status == 200
        assert data["analysis"] == payload

        status, data = _request(live_server, "/api/benchmarks/smoke/analysis")
        assert status == 200
        assert data["analysis"] == payload
        on_disk = json.loads(
            (benchmarks_root(workflow_dir) / "smoke" / "analysis.json").read_text(
                encoding="utf-8"
            )
        )
        assert on_disk == payload

    def test_version_digest_unchanged_after_analysis_put(self, live_server, workflow_dir):
        before = load_version(workflow_dir, "smoke", "v1")
        version_path = benchmarks_root(workflow_dir) / "smoke" / "v1.json"
        version_bytes_before = version_path.read_bytes()

        status, _ = _request(
            live_server,
            "/api/benchmarks/smoke/analysis",
            method="PUT",
            body={"notes": "post-hoc"},
        )
        assert status == 200

        after = load_version(workflow_dir, "smoke", "v1")
        assert after["digest_sha256"] == before["digest_sha256"]
        assert version_path.read_bytes() == version_bytes_before

    def test_get_analysis_null_when_missing(self, live_server):
        status, data = _request(live_server, "/api/benchmarks/smoke/analysis")
        assert status == 200
        assert data["analysis"] is None

    def test_non_object_analysis_refused(self, live_server):
        status, data = _request(
            live_server,
            "/api/benchmarks/smoke/analysis",
            method="PUT",
            body=["not", "an", "object"],
        )
        assert status == 400
        assert "JSON object" in data["error"]


class TestExperimentNotesApi:
    def test_put_analysis_is_refused(self, experiment_server):
        server, store = experiment_server
        status, _data = _request(
            server,
            "/api/experiment/exp-1/analysis",
            method="PUT",
            body={"findings": ["no regressions"]},
        )
        assert status == 405
        assert "analysis_json" not in store.get_experiment("exp-1")
        assert store.get_experiment("exp-1")["notes"] == "original notes"

    def test_patch_notes_does_not_accept_analysis(self, experiment_server):
        server, store = experiment_server
        status, data = _request(
            server,
            "/api/experiment/exp-1",
            method="PATCH",
            body={"notes": "updated", "analysis": {"ignored": True}},
        )
        assert status == 400
        assert "notes" in data["error"]
        assert store.get_experiment("exp-1")["notes"] == "original notes"

    def test_patch_archives_and_unarchives_an_experiment(self, experiment_server):
        server, store = experiment_server

        status, data = _request(
            server,
            "/api/experiment/exp-1",
            method="PATCH",
            body={"archived": True},
        )
        assert status == 200
        assert data["experiment"]["archived"] is True
        assert store.get_experiment("exp-1")["archived"] is True

        status, data = _request(
            server,
            "/api/experiment/exp-1",
            method="PATCH",
            body={"archived": False},
        )
        assert status == 200
        assert data["experiment"]["archived"] is False


class TestSpaSurface:
    def test_benchmark_browser_ships(self):
        page = run_chatbot_server.load_index_html()
        assert b'id="benchmarksBtn"' not in page
        assert b'id="hierarchyCrumbs"' not in page
        assert b"showBenchmarks" in page
        assert b"/api/benchmarks" in page
        assert b"/analysis" in page
        assert b"Save analysis" in page
        assert b"Save notes" not in page
        assert b"Save postmortem" in page
        assert b"Postmortem saved" in page
        # The detail API retains the digest; the concise experiment Result
        # does not expose raw provenance metadata.
        assert b"benchmark digest" not in page
        assert b"innerHTML" not in page

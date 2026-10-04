"""Real SQLite and HTTP coverage for pre-run review; no runtime/model calls."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import threading

import pytest

from fastworkflow.experiment.setup import (
    ExperimentSetups,
    SetupConflict,
    SCHEMA,
    setup_digest,
)
from fastworkflow.run_chatbot.server import ChatbotServer, load_index_html
from tests.test_chatbot_benchmarks import _request, workspace_server  # noqa: F401


def spec():
    return {
        "schema": SCHEMA,
        "experiment_id": "population-test",
        "description": "A cursor prevents skipped items",
        "model_routes": {"agent": "provider/pinned-model"},
        "configuration": {"workflow_fingerprint": "sha256:fixed", "cursor": True},
        "budgets": {"turn_seconds": 120},
        "repetitions": 2,
        "estimated_cost": {
            "currency": "USD",
            "low": 0.1,
            "high": 0.5,
            "basis": "2 tasks, 2 repeats",
        },
        "tasks": [
            {
                "task_id": "case-1",
                "description": "Check every member",
                "split": "tuning",
                "input": {"messages": ["Check all members"]},
                "expected_outcomes": [
                    {
                        "description": "Every member checked",
                        "evidence": "Compare item ledger to source roster",
                    }
                ],
            }
        ],
    }


def approve(store, row):
    return store.decide(
        row["experiment_id"], row["revision"], row["digest"], "approved", "Human A"
    )


def test_reads_do_not_create_files(tmp_path):
    store = ExperimentSetups(tmp_path)
    assert store.list() == []
    with pytest.raises(KeyError):
        store.get("absent")
    assert list(tmp_path.iterdir()) == []


def test_revision_approval_history_and_export(tmp_path):
    store = ExperimentSetups(tmp_path)
    initial = store.save(spec(), 0)
    with pytest.raises(SetupConflict):
        store.approved(initial["experiment_id"], 1, initial["digest"])
    approved = approve(store, initial)
    assert approved["status"] == "approved"
    assert store.approved(initial["experiment_id"], 1, initial["digest"])["spec"] == spec()
    updated = spec()
    updated["tasks"][0]["input"] = "changed input"
    second = store.save(updated, 1)
    assert second["status"] == "needs_review"
    assert second["digest"] != initial["digest"]
    assert second["history"][1]["spec"] == spec()
    assert second["decisions"][0]["revision"] == 1
    with pytest.raises(SetupConflict):
        approve(store, initial)
    with pytest.raises(SetupConflict):
        store.approved(initial["experiment_id"], 1, initial["digest"])
    approve(store, second)
    store.decide(
        second["experiment_id"],
        2,
        second["digest"],
        "changes_requested",
        "Human B",
        "Evidence is incomplete",
    )
    with pytest.raises(SetupConflict):
        store.approved(second["experiment_id"], 2, second["digest"])
    assert len(store.get(second["experiment_id"])["decisions"]) == 3


def test_parallel_edits_cannot_overwrite(tmp_path):
    store = ExperimentSetups(tmp_path)
    store.save(spec(), 0)

    def edit(description):
        changed = spec()
        changed["description"] = description
        try:
            return store.save(changed, 1)["revision"]
        except SetupConflict:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(edit, ["A", "B"]))
    assert sorted(map(str, results)) == ["2", "conflict"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("tasks", []),
        ("repetitions", True),
        ("repetitions", 0),
        ("model_routes", {}),
        ("configuration", {"x": float("nan")}),
        ("budgets", {"x": (1, 2)}),
        ("experiment_id", "../escape"),
        ("estimated_cost", {"currency": "USD", "low": 10, "high": 1, "basis": "wrong"}),
    ],
)
def test_invalid_setup_rejected_before_write(tmp_path, field, value):
    data = spec()
    data[field] = value
    with pytest.raises(ValueError):
        ExperimentSetups(tmp_path).save(data, 0)
    assert list(tmp_path.iterdir()) == []


def test_description_is_optional_and_the_ui_frame_fields_are_absent(tmp_path):
    data = spec()
    data["description"] = ""
    row = ExperimentSetups(tmp_path).save(data, 0)
    assert row["spec"]["description"] == ""
    assert "control" not in row["spec"]
    data = spec()
    data["description"] = 7
    with pytest.raises(ValueError, match="description"):
        ExperimentSetups(tmp_path).save(data, 0)


def test_task_evidence_and_unique_identity_required(tmp_path):
    data = spec()
    data["tasks"].append(deepcopy(data["tasks"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        ExperimentSetups(tmp_path).save(data, 0)
    data = spec()
    data["tasks"][0]["expected_outcomes"][0]["evidence"] = ""
    with pytest.raises(ValueError, match="evidence"):
        ExperimentSetups(tmp_path).save(data, 0)


def test_digest_includes_all_fields_and_is_key_order_independent():
    data = spec()
    assert setup_digest(data) == setup_digest(dict(reversed(list(data.items()))))
    for key in ("model_routes", "budgets", "configuration"):
        changed = deepcopy(data)
        changed[key]["extra"] = "different"
        assert setup_digest(data) != setup_digest(changed)


@pytest.fixture
def setup_server(tmp_path):
    workflow = tmp_path / "workflow"
    workflow.mkdir()
    (workflow / "_commands").mkdir()
    server = ChatbotServer(
        db_path="",
        workflow_path=str(workflow),
        port=0,
        spawn_options={"no_server": True},
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, workflow
    server.shutdown()
    thread.join(timeout=5)


def test_http_before_first_run_auth_revision_and_export(setup_server):
    server, workflow = setup_server
    path = "/api/experiment-setups"
    assert _request(server, path)[1] == {"setups": []}
    assert not (workflow / "experiment_setups").exists()
    body = {"spec": spec(), "expected_revision": 0}
    assert _request(server, path, "POST", body, token=None)[0] == 401
    assert not (workflow / "experiment_setups").exists()
    status, saved = _request(server, path, "POST", body)
    assert status == 201
    row = saved["setup"]
    detail = path + "/" + row["experiment_id"]
    assert _request(server, detail)[1]["setup"]["spec"] == spec()
    assert _request(server, detail + "/export")[0] == 409
    decision = {
        "revision": 1,
        "digest": row["digest"],
        "decision": "approved",
        "reviewer": "Human A",
    }
    assert (
        _request(server, detail + "/decisions", "POST", decision, token=None)[0] == 401
    )
    assert (
        _request(server, detail + "/decisions", "POST", dict(decision, digest="stale"))[
            0
        ]
        == 409
    )
    assert _request(server, detail + "/decisions", "POST", decision)[0] == 201
    assert _request(server, detail + "/export")[1]["spec"] == spec()
    assert _request(server, path, "POST", body)[0] == 409
    body["expected_revision"] = 1
    body["spec"]["description"] = "Revised description"
    assert _request(server, path, "POST", body)[0] == 201
    assert _request(server, detail + "/decisions", "POST", decision)[0] == 409
    assert _request(server, detail + "/export")[0] == 409
    assert _request(server, path, "POST", [])[0] == 400
    assert _request(server, detail + "/unknown")[0] == 404


def test_workspace_refuses_setup_authoring(workspace_server):
    server = (
        workspace_server[0] if isinstance(workspace_server, tuple) else workspace_server
    )
    assert (
        _request(
            server,
            "/api/experiment-setups",
            "POST",
            {"spec": spec(), "expected_revision": 0},
        )[0]
        == 403
    )


def test_ui_has_pre_run_entry_and_safe_task_rendering():
    html = load_index_html().decode("utf-8")
    assert 'id="benchmarkSetupBtn"' in html
    ui = html[
        html.index("/* Benchmark setup:") : html.index(
            "/* -- capability-gated formal review"
        )
    ]
    assert "Task prompt (optional)" in ui and "Description (optional)" in ui
    assert "Save benchmark" in ui and "New experiment" in ui
    assert "expected_version" in ui and "Experiment ID" in ui
    assert "innerHTML" not in ui


@pytest.mark.parametrize("version", [1, 999])
def test_navigation_and_setup_ignore_incompatible_evidence(
    setup_server, monkeypatch, version
):
    import sqlite3

    server, workflow = setup_server
    monkeypatch.chdir(workflow.parent)
    db = workflow / "old-evidence.sqlite3"
    with sqlite3.connect(db) as conn:
        conn.execute(f"PRAGMA user_version = {version}")
    before = db.read_bytes()
    server.db_path = str(db)
    assert _request(server, "/api/session")[0] == 200
    status, data = _request(server, "/api/workflows")
    assert status == 200
    assert str(workflow) in [row["path"] for row in data["workflows"]]
    assert _request(server, "/api/browse")[0] == 200
    assert _request(server, "/api/benchmarks")[0] == 200
    assert _request(server, "/api/experiment-setups")[0] == 200
    body = {"spec": spec(), "expected_revision": 0}
    assert _request(server, "/api/experiment-setups", "POST", body)[0] == 201
    assert _request(server, "/api/experiment-setups/" + spec()["experiment_id"])[0] == 200
    status, data = _request(server, "/api/turns")
    assert status == 409
    assert "schema" in data["error"].lower()
    assert db.read_bytes() == before


def test_setup_navigation_is_visible_outside_debug_view():
    html = load_index_html().decode("utf-8")
    header = html.split("</header>")[0]
    assert 'id="benchmarkSetupBtn"' in header
    assert html.count('id="benchmarkSetupBtn"') == 1
    handler = html.split("function openBenchmarkSetup()")[1].split(
        "/* -- capability-gated formal review"
    )[0]
    assert 'setTopMode("debug")' in handler
    assert 'setTopMode("picker")' in handler

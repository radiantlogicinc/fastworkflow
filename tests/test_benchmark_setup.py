"""Benchmark authoring through real files, HTTP and the experiment lifecycle."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import sqlite3

import pytest

from fastworkflow import state_paths
from fastworkflow.benchmark import setup as setup
from fastworkflow.observability import store as obs
from fastworkflow.benchmark.catalog import list_versions, load_version, write_version
from fastworkflow.experiment.runner import ExperimentController, ExperimentHarness
from tests.test_experiment_setup import setup_server  # noqa: F401
from tests.test_chatbot_benchmarks import _request
from tests.test_experiment_container import _turn_row, _write_turn


def create(folder):
    return setup.save_benchmark(
        folder,
        {"title": "Roster review", "tasks": [{}, {"prompt": "Review the roster"}]},
    )


def test_generated_versions_ids_optional_text_and_frozen_experiment(tmp_path):
    first = create(tmp_path)
    assert first["version"] == "v1" and first["description"] == ""
    assert first["tasks"][0]["prompt"] == ""
    ids = [t["task_id"] for t in first["tasks"]]
    assert len(set(ids)) == 2
    experiment = setup.create_experiment(tmp_path, first["benchmark_id"], "v1")
    second = setup.save_benchmark(
        tmp_path,
        {
            "benchmark_id": first["benchmark_id"],
            "expected_version": "v1",
            "title": "Updated",
            "description": "New description",
            "tasks": [{"task_id": ids[0], "prompt": "Updated prompt"}, {}],
        },
    )
    assert second["version"] == "v2" and second["tasks"][0]["task_id"] == ids[0]
    assert second["tasks"][1]["task_id"] not in ids
    record, pinned = setup.experiment_manifest(tmp_path, experiment["experiment_id"])
    assert record["benchmark_version"] == "v1" and pinned == first
    with pytest.raises(setup.BenchmarkSetupConflict):
        setup.save_benchmark(
            tmp_path,
            {
                "benchmark_id": first["benchmark_id"],
                "expected_version": "v1",
                "title": "Stale",
                "tasks": [{}],
            },
        )


def test_concurrent_edit_has_one_winner(tmp_path):
    first = create(tmp_path)

    def save(title):
        try:
            return setup.save_benchmark(
                tmp_path,
                {
                    "benchmark_id": first["benchmark_id"],
                    "expected_version": "v1",
                    "title": title,
                    "tasks": [{}],
                },
            )["version"]
        except setup.BenchmarkSetupConflict:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(save, ["A", "B"])) == ["conflict", "v2"]


def test_edit_preserves_existing_opaque_payload(tmp_path):
    old = write_version(
        tmp_path,
        {
            "benchmark_id": "legacy",
            "version": "v1",
            "description": "Old corpus",
            "tasks": [
                {
                    "task_id": "existing",
                    "description": "Old task",
                    "payload": {"domain": [1, 2]},
                }
            ],
        },
    )
    new = setup.save_benchmark(
        tmp_path,
        {
            "benchmark_id": "legacy",
            "expected_version": "v1",
            "title": "Title",
            "tasks": [{"task_id": "existing"}],
        },
    )
    assert new["tasks"][0]["payload"] == old["tasks"][0]["payload"]
    assert load_version(tmp_path, "legacy", "v1") == old


def test_new_task_identity_is_not_caller_supplied(tmp_path):
    with pytest.raises(ValueError, match="automatically"):
        setup.save_benchmark(
            tmp_path, {"title": "Test", "tasks": [{"task_id": "invented"}]}
        )


def test_http_create_register_execute_drilldown_and_plain_conversations(
    setup_server, tmp_path
):
    server, folder = setup_server
    status, result = _request(
        server,
        "/api/benchmark-setup",
        "POST",
        {"title": "Review", "tasks": [{}]},
        token=None,
    )
    assert status == 401
    assert not (folder / "benchmarks").exists()
    status, result = _request(
        server, "/api/benchmark-setup", "POST", {"title": "Review", "tasks": [{}]}
    )
    assert status == 201
    benchmark = result["version"]
    benchmark_id, task_id = benchmark["benchmark_id"], benchmark["tasks"][0]["task_id"]
    base = "/api/benchmarks/" + benchmark_id + "/experiments"
    assert _request(server, base)[1]["experiments"] == []
    assert _request(server, base, "POST", {"version": "v1"}, token=None)[0] == 401
    status, result = _request(server, base, "POST", {"version": "v1"})
    assert status == 201
    experiment_id = result["experiment"]["experiment_id"]
    second = _request(server, base, "POST", {"version": "v1"})[1]["experiment"][
        "experiment_id"
    ]
    assert experiment_id != second
    assert len(_request(server, base)[1]["experiments"]) == 2
    detail_path = "/api/benchmark-experiments/" + experiment_id
    assert _request(server, detail_path)[1]["recorded"] is False

    # The driver records into the workflow's one live database.
    db = state_paths.observability_db(str(folder))
    store = obs.ObservabilityStore(db)
    server.db_path = db
    controller = ExperimentController(
        str(folder), store.store_identity(), external=False
    )
    with pytest.raises(ValueError, match="task IDs"):
        controller.create_experiment(
            experiment_id,
            "test",
            declared_tasks=1,
            declared_attempts=1,
            declarations=[("wrong", 1, "channel")],
        )
    assert store.get_experiment(experiment_id) is None
    controller.create_experiment(
        experiment_id,
        "test",
        declared_tasks=1,
        declared_attempts=1,
        declarations=[(task_id, 1, "channel")],
    )
    recorded = store.get_experiment(experiment_id)
    assert recorded["benchmark_id"] == benchmark_id
    assert recorded["benchmark_version"] == "v1"
    assert recorded["benchmark_digest_sha256"] == benchmark["digest_sha256"]
    conv = store.mint_conversation_id(
        "channel", experiment_id=experiment_id, task_id=task_id, attempt=1
    )
    controller.start_attempt(experiment_id, task_id, 1, "channel", conversation_id=conv)
    _write_turn(
        store,
        _turn_row(
            "registered-turn",
            "channel",
            conversation_id=conv,
            experiment_id=experiment_id,
            task_id=task_id,
            attempt=1,
        ),
    )
    controller.finish_attempt(
        experiment_id, task_id, 1, outcome="pass", outcome_source="fixture"
    )
    assert _request(server, detail_path)[1]["recorded"] is True
    assert (
        _request(server, "/api/experiment/" + experiment_id)[1]["experiment"][
            "benchmark_id"
        ]
        == benchmark_id
    )
    assert (
        _request(server, "/api/experiment/" + experiment_id + "/attempts")[1][
            "attempts"
        ][0]["conversation_id"]
        == conv
    )
    assert _request(server, "/api/turns")[1]["turns"][0]["task_id"] == task_id
    assert (
        _request(server, "/api/turn/registered-turn")[1]["turn"][
            "experiment_id"
        ]
        == experiment_id
    )
    scope = "?turn_key=registered-turn"
    # One write route and a separate read route since fix-9eg.16.
    write_path = "/post_feedback" + scope
    read_path = "/api/feedback-notes" + scope
    comment = {"target_kind": "turn", "span_ids": [], "target_label": "Turn",
               "comment": "Check the final answer against the task prompt.",
               "provenance": "human", "category": "recommendations",
               "subcategory": "what_to_do"}
    assert _request(server, write_path, "POST", comment)[0] == 201
    assert store.list_human_feedback("registered-turn")[0]["comment"] == comment["comment"]
    assert _request(server, read_path)[1]["feedback"][0]["comment"] == comment["comment"]
    assert _request(server, "/api/experiment/" + experiment_id + "/analysis",
                    "PUT", {"analysis": "Free-form review"})[0] == 405
    assert _request(server, "/api/experiment/" + second + "/analysis",
                    "PUT", {"analysis": "wrong target"})[0] == 405


def test_registered_pins_cannot_be_overridden(tmp_path):
    manifest = create(tmp_path)
    record = setup.create_experiment(tmp_path, manifest["benchmark_id"], "v1")
    store = obs.ObservabilityStore(state_paths.observability_db(str(tmp_path)))
    controller = ExperimentController(
        str(tmp_path),
        store.store_identity(),
        external=False,
    )
    declarations = [(tid, 1, tid) for tid in record["task_ids"]]
    with pytest.raises(ValueError, match="benchmark_version"):
        controller.create_experiment(
            record["experiment_id"],
            "x",
            declared_tasks=2,
            declared_attempts=1,
            declarations=declarations,
            benchmark_version="v99",
        )
    assert store.get_experiment(record["experiment_id"]) is None


def test_registration_survives_incompatible_default_evidence(setup_server, tmp_path):
    server, folder = setup_server
    manifest = create(folder)
    db = tmp_path / "old.sqlite3"
    with sqlite3.connect(db) as conn:
        conn.execute("PRAGMA user_version=1")
    server.db_path = str(db)
    path = "/api/benchmarks/" + manifest["benchmark_id"] + "/experiments"
    assert _request(server, path, "POST", {"version": "v1"})[0] == 201
    result = _request(server, path)[1]
    assert len(result["experiments"]) == 1 and result["warning"]


def test_harness_factory_preserves_registered_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "state"))
    folder = tmp_path / "workflow"
    manifest = create(folder)
    record = setup.create_experiment(folder, manifest["benchmark_id"], "v1")
    harness = ExperimentHarness.from_benchmark_experiment(
        str(folder), record["experiment_id"]
    )
    assert harness.experiment_id == record["experiment_id"]
    assert harness.benchmark_version == "v1"
    assert harness.benchmark_digest_sha256 == manifest["digest_sha256"]


def test_delete_empty_registration_preserves_benchmark_and_refuses_stale_runner(setup_server, tmp_path):
    server, folder = setup_server
    benchmark = create(folder)
    # The first experiment of a group is its winner, and the winner is not
    # deletable (`fix-jfy5`). This one holds that title so the registration
    # under test is an ordinary non-selected one.
    selected = setup.create_experiment(folder, benchmark["benchmark_id"], "v1")
    record = setup.create_experiment(folder, benchmark["benchmark_id"], "v1")
    eid = record["experiment_id"]
    path = "/api/benchmark-experiments/" + eid
    before = (folder / "benchmarks" / benchmark["benchmark_id"] / "v1.json").read_bytes()
    assert _request(server, path)[1]["can_delete"] is True
    assert _request(server, path, "DELETE", token=None)[0] == 401
    assert setup.load_experiment(folder, eid)["store"] is None
    status, result = _request(server, path, "DELETE")
    assert status == 200 and result["deleted"] == eid
    assert [row["experiment_id"] for row in
            setup.registered_experiments(folder, benchmark["benchmark_id"])] == [
        selected["experiment_id"]
    ]
    assert _request(server, path)[0] == 404
    assert _request(server, path, "DELETE")[0] == 404
    assert (folder / "benchmarks" / benchmark["benchmark_id"] / "v1.json").read_bytes() == before
    store = obs.ObservabilityStore(state_paths.observability_db(str(folder)))
    controller = ExperimentController(str(folder), store.store_identity(), external=False)
    with pytest.raises(setup.ExperimentDeleted):
        controller.create_experiment(eid, "Late runner", declared_tasks=2, declared_attempts=1,
            declarations=[(task_id, 1, "channel-" + task_id) for task_id in record["task_ids"]])
    assert store.get_experiment(eid) is None
    assert store.experiment_attempt_declarations(eid) == []


def test_delete_refuses_bound_experiment(setup_server):
    server, folder = setup_server
    benchmark = create(folder)
    record = setup.create_experiment(folder, benchmark["benchmark_id"], "v1")
    eid = record["experiment_id"]
    setup.bind_experiment(folder, eid)
    path = "/api/benchmark-experiments/" + eid
    assert _request(server, path)[1]["can_delete"] is False
    assert _request(server, path, "DELETE")[0] == 409
    live = obs.ReadOnlyObservabilityStore(state_paths.observability_db(str(folder)))
    assert setup.load_experiment(folder, eid)["store"] == {"store_id": live.store_identity()}


def test_description_is_optional_free_text_the_author_owns(setup_server, tmp_path):
    """The description is the author's: absent by default, editable until
    a runner claims the registration, and carried into the recorded run."""
    server, folder = setup_server
    benchmark = create(folder)
    assert setup.create_experiment(folder, benchmark["benchmark_id"], "v1")["description"] == ""
    created = setup.create_experiment(
        folder, benchmark["benchmark_id"], "v1", "  Does insight #7 lift pass^3?  "
    )
    eid = created["experiment_id"]
    assert created["description"] == "Does insight #7 lift pass^3?"
    path = "/api/benchmark-experiments/" + eid
    assert _request(server, path, "PATCH", {"description": "Rewritten by its author"})[0] == 200
    assert setup.load_experiment(folder, eid)["description"] == "Rewritten by its author"
    assert _request(server, path, "PATCH", {"notes": "x"})[0] == 400
    assert _request(server, path, "PATCH", {"description": 7})[0] == 400
    assert _request(server, path, "PATCH", {"description": "x"}, token=None)[0] == 401
    assert _request(server, "/api/benchmark-experiments/missing", "PATCH",
                    {"description": "x"})[0] == 404
    # The harness reads the description off the registration, so the run it
    # records carries what the author wrote rather than the benchmark's title.
    store = obs.ObservabilityStore(state_paths.observability_db(str(folder)))
    controller = ExperimentController(str(folder), store.store_identity(), external=False)
    controller.create_experiment(
        eid, setup.load_experiment(folder, eid)["description"], declared_tasks=2, declared_attempts=1,
        declarations=[(task_id, 1, "channel-" + task_id) for task_id in created["task_ids"]])
    assert store.get_experiment(eid)["description"] == "Rewritten by its author"
    # Bound now: the description belongs to the recorded run, not to setup.
    assert _request(server, path, "PATCH", {"description": "too late"})[0] == 409
    assert setup.load_experiment(folder, eid)["description"] == "Rewritten by its author"


def test_refused_redeclaration_of_a_started_experiment_changes_nothing(tmp_path):
    """`fix-spvf`: the refusal used to land after the experiments row was
    rewritten, so the stored description no longer matched the run."""
    benchmark = create(tmp_path)
    record = setup.create_experiment(tmp_path, benchmark["benchmark_id"], "v1", "Attempt 1")
    eid = record["experiment_id"]
    db_path = state_paths.observability_db(str(tmp_path))
    store = obs.ObservabilityStore(db_path)
    controller = ExperimentController(str(tmp_path), store.store_identity(), external=False)
    declarations = [(task_id, 1, "channel-" + task_id) for task_id in record["task_ids"]]
    controller.create_experiment(eid, "Attempt 1", declared_tasks=2, declared_attempts=1,
                                 declarations=declarations, arm="first")
    task_id = record["task_ids"][0]
    controller.start_attempt(eid, task_id, 1, "channel-" + task_id)

    def snapshot():
        with sqlite3.connect(db_path) as conn:
            return list(conn.iterdump())

    before = snapshot()
    assert any("Attempt 1" in line and "INSERT INTO \"experiments\"" in line for line in before)
    with pytest.raises(obs.ExperimentDeclarationConflict, match="already started"):
        controller.create_experiment(
            eid, "Attempt 2", declared_tasks=2, declared_attempts=1, arm="second",
            declarations=[(tid, 1, "other-" + tid) for tid in record["task_ids"]])
    assert snapshot() == before
    assert store.get_experiment(eid)["description"] == "Attempt 1"


def test_refused_pre_start_redeclaration_changes_nothing(tmp_path):
    """`fix-lr1z`: before any attempt starts, a differing plan used to be
    refused only after the experiments row had been rewritten."""
    benchmark = create(tmp_path)
    record = setup.create_experiment(tmp_path, benchmark["benchmark_id"], "v1", "Attempt 1")
    eid = record["experiment_id"]
    db_path = state_paths.observability_db(str(tmp_path))
    store = obs.ObservabilityStore(db_path)
    controller = ExperimentController(str(tmp_path), store.store_identity(), external=False)
    controller.create_experiment(
        eid, "Attempt 1", declared_tasks=2, declared_attempts=1, arm="first",
        declarations=[(tid, 1, "channel-" + tid) for tid in record["task_ids"]])

    def snapshot():
        with sqlite3.connect(db_path) as conn:
            return list(conn.iterdump())

    before = snapshot()
    with pytest.raises(obs.ExperimentDeclarationConflict, match="different immutable"):
        controller.create_experiment(
            eid, "Attempt 2", declared_tasks=2, declared_attempts=1, arm="second",
            declarations=[(tid, 1, "other-" + tid) for tid in record["task_ids"]])
    assert snapshot() == before
    assert store.get_experiment(eid)["description"] == "Attempt 1"


def test_http_creation_accepts_the_authors_description(setup_server):
    server, folder = setup_server
    benchmark = create(folder)
    path = "/api/benchmarks/" + benchmark["benchmark_id"] + "/experiments"
    status, data = _request(server, path, "POST", {"version": "v1", "description": "Trial run"})
    assert status == 201 and data["experiment"]["description"] == "Trial run"
    assert _request(server, path, "POST", {"version": "v1"})[1]["experiment"]["description"] == ""
    assert _request(server, path, "POST", {"version": "v1", "description": 7})[0] == 400


def test_delete_on_an_unrelated_route_is_refused(setup_server):
    server, _folder = setup_server
    assert _request(server, "/api/turn/turn", "DELETE")[0] == 405


def test_delete_and_runner_binding_are_serialized(tmp_path):
    from threading import Barrier
    benchmark = create(tmp_path)
    # Hold the group's winner title with an experiment nobody races, so the
    # raced one is always an ordinary non-selected registration: a winner is
    # not deletable (`fix-jfy5`), and this test is about the transaction that
    # serializes deletion against a runner's binding, not about selection.
    setup.create_experiment(tmp_path, benchmark["benchmark_id"], "v1")
    for _ in range(8):
        record = setup.create_experiment(tmp_path, benchmark["benchmark_id"], "v1")
        eid = record["experiment_id"]
        store = obs.ObservabilityStore(state_paths.observability_db(str(tmp_path)))
        controller = ExperimentController(str(tmp_path), store.store_identity(), external=False)
        barrier = Barrier(2)
        def start():
            barrier.wait()
            try:
                controller.create_experiment(eid, "Race", declared_tasks=2, declared_attempts=1,
                    declarations=[(task_id, 1, "channel-" + task_id) for task_id in record["task_ids"]])
                return "started"
            except setup.ExperimentDeleted:
                return "deleted"
        def delete():
            barrier.wait()
            try:
                setup.delete_empty_experiment(tmp_path, eid)
                return "deleted"
            except setup.BenchmarkSetupConflict:
                return "protected"
        with ThreadPoolExecutor(max_workers=2) as pool:
            launch = pool.submit(start)
            removal = pool.submit(delete)
            outcome = (launch.result(), removal.result())
        assert outcome in (("started", "protected"), ("deleted", "deleted"))
        if outcome[0] == "started":
            assert store.get_experiment(eid) is not None
            assert setup.load_experiment(tmp_path, eid)["store"] is not None
        else:
            assert store.get_experiment(eid) is None
            with pytest.raises(setup.ExperimentDeleted):
                setup.load_experiment(tmp_path, eid)


def test_experiment_lifecycle_over_http_leaves_only_authored_files(setup_server):
    """Create, duplicate, edit and delete through the API: the project keeps
    only authored benchmark files, the state dir no JSON or lock files."""
    server, folder = setup_server
    status, result = _request(server, "/api/benchmark-setup", "POST",
                              {"title": "Review", "tasks": [{}]})
    assert status == 201
    benchmark_id = result["version"]["benchmark_id"]
    base = "/api/benchmarks/" + benchmark_id
    assert _request(server, base + "/analysis", "PUT", {"analysis": "Notes"})[0] == 200
    state_dir = Path(state_paths.workflow_state_dir(str(folder)))
    assert not Path(state_paths.observability_db(str(folder))).exists()

    status, result = _request(server, base + "/experiments", "POST",
                              {"version": "v1", "description": "First"})
    assert status == 201
    first = result["experiment"]
    assert first["store"] is None and "db_path" not in json.dumps(first)
    detail = "/api/benchmark-experiments/" + first["experiment_id"]
    status, result = _request(server, detail + "/duplicate", "POST", {"runs_per_task": 2})
    assert status == 201 and result["changed_fields"] == ["runs_per_task"]
    copy = result["experiment"]
    copy_path = "/api/benchmark-experiments/" + copy["experiment_id"]
    assert _request(server, copy_path, "PATCH", {"description": "Copy"})[0] == 200

    listed = {row["experiment_id"]: row
              for row in _request(server, base + "/experiments")[1]["experiments"]}
    assert set(listed) == {first["experiment_id"], copy["experiment_id"]}
    assert listed[copy["experiment_id"]]["description"] == "Copy"
    assert listed[copy["experiment_id"]]["runs_per_task"] == 2
    assert listed[copy["experiment_id"]]["source_experiment_id"] == first["experiment_id"]
    assert listed[first["experiment_id"]]["is_winner"] is True
    shown = _request(server, copy_path)[1]
    assert shown["experiment"] == dict(copy, description="Copy")
    assert shown["can_delete"] is True

    assert _request(server, copy_path, "DELETE")[0] == 200
    assert _request(server, copy_path)[0] == 404
    assert [row["experiment_id"] for row in
            _request(server, base + "/experiments")[1]["experiments"]] == [first["experiment_id"]]

    benchmarks = folder / "benchmarks"
    assert sorted(p.relative_to(benchmarks).as_posix() for p in benchmarks.rglob("*")) == [
        benchmark_id, benchmark_id + "/analysis.json", benchmark_id + "/v1.json",
    ]
    leftovers = [p.name for p in state_dir.rglob("*") if p.suffix in (".json", ".lock")]
    assert leftovers == []


def test_concurrent_saves_of_the_next_version_one_wins_one_conflicts(setup_server):
    server, folder = setup_server
    first = create(folder)
    for n in range(4):
        body = {"benchmark_id": first["benchmark_id"], "expected_version": f"v{n + 1}",
                "title": "Edit", "tasks": [{}]}
        with ThreadPoolExecutor(max_workers=2) as pool:
            statuses = sorted(pool.map(
                lambda _: _request(server, "/api/benchmark-setup", "POST", body)[0], range(2)
            ))
        assert statuses == [201, 409]
    assert list_versions(folder, first["benchmark_id"]) == ["v1", "v2", "v3", "v4", "v5"]


def test_a_read_only_live_db_refuses_registration_writes_with_409(setup_server):
    server, folder = setup_server
    benchmark = create(folder)
    eid = setup.create_experiment(folder, benchmark["benchmark_id"], "v1")["experiment_id"]
    live = Path(state_paths.observability_db(str(folder)))
    live.chmod(0o444)
    try:
        path = "/api/benchmark-experiments/" + eid
        status, data = _request(server, path, "PATCH", {"description": "x"})
        assert status == 409 and "readonly" in data["error"].lower()
        assert _request(server, path, "DELETE")[0] == 409
    finally:
        live.chmod(0o600)
    assert setup.load_experiment(folder, eid)["description"] == ""

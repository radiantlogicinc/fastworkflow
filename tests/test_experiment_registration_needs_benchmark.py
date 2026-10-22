"""No experiment is registered in a store that holds no benchmark (fix-ajir.24).

The chatbot's seal, browse and delete routes carry no production guard of
their own. What keeps them to benchmark experiments is that every product path
which registers an experiment, an attempt or a claim needs a benchmark version
stored in the workflow's live DB first. Each path is driven here against a
store that holds none.
"""

from __future__ import annotations

import threading

import pytest

from fastworkflow import state_paths
from fastworkflow.benchmark import catalog, setup
from fastworkflow.experiment.runner import ExperimentController, ExperimentNotRegisteredHere
from fastworkflow.observability import store as obs
from fastworkflow.run_chatbot import server as run_chatbot_server
from tests.test_chatbot_benchmarks import _request


@pytest.fixture
def workflow(tmp_path, monkeypatch):
    monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "state"))
    folder = tmp_path / "workflow"
    (folder / "_commands").mkdir(parents=True)
    store = obs.ObservabilityStore(state_paths.observability_db(str(folder)))
    assert catalog.list_benchmarks(str(folder)) == []
    return str(folder), store


def _create(folder, store, **kwargs):
    controller = ExperimentController(folder, store.store_identity(), external=False)
    controller.create_experiment(
        "exp-1", "", declared_tasks=1, declared_attempts=1, **kwargs
    )


def test_the_controller_refuses_every_experiment_without_a_stored_version(workflow):
    folder, store = workflow
    declarations = [("t1", 1, "src")]
    with pytest.raises(obs.MissingBenchmarkPin):
        _create(folder, store, declarations=declarations)
    with pytest.raises(ExperimentNotRegisteredHere):
        _create(folder, store, declarations=declarations, registered=True)
    with pytest.raises(catalog.BenchmarkManifestError, match="not found"):
        _create(
            folder, store, declarations=declarations, benchmark_id="made-up",
            benchmark_version="v1", benchmark_digest_sha256="0" * 64,
        )
    assert store.list_experiments() == []


def test_a_draft_version_without_tasks_registers_nothing(workflow):
    folder, store = workflow
    draft = setup.save_benchmark(folder, {"title": "Draft", "tasks": []})
    version = catalog.load_version(folder, draft["benchmark_id"], "v1")
    assert version["tasks"] == []
    with pytest.raises(ValueError, match="add tasks"):
        setup.create_experiment(folder, draft["benchmark_id"], "v1")
    with pytest.raises(ValueError):
        _create(
            folder, store, declarations=[], benchmark_id=draft["benchmark_id"],
            benchmark_version="v1", benchmark_digest_sha256=version["digest_sha256"],
        )
    assert store.list_experiments() == []


def test_attempts_and_claims_need_a_registered_experiment(workflow):
    """`register_attempt` needs the declaration `create_experiment` writes, and
    FastAPI `/initialize` with `experiment_bootstrap` only consumes a
    registration (`run_fastapi_mcp.utils._claim_registered_attempt`, which
    answers 403 to `AttemptClaimError`)."""
    _folder, store = workflow
    with pytest.raises(obs.UndeclaredExperimentAttempt):
        store.register_attempt("exp-1", "t1", 1, "src", "chan")
    with pytest.raises(obs.AttemptClaimError):
        store.claim_attempt(
            {"registration_id": "reg-made-up", "secret": "s" * 64},
            channel_id="chan", server_incarnation="test",
        )


def test_the_chatbot_registers_nothing_without_a_benchmark(workflow):
    folder, store = workflow
    server = run_chatbot_server.ChatbotServer(
        db_path=store.db_path, workflow_path=folder, port=0,
        spawn_options={"no_server": True},
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, _ = _request(
            server, "/api/benchmarks/made-up/experiments", method="POST",
            body={"version": "v1"},
        )
        assert status == 400
        status, _ = _request(
            server, "/api/benchmark-experiments/exp-made-up/duplicate", method="POST",
            body={},
        )
        assert status == 404
    finally:
        server.shutdown()
        thread.join(timeout=5)
    assert store.list_experiments() == []
    assert setup.registered_experiments(folder) == []

"""Deleting an unused experiment and the winner pointer (`fix-873a`).

A registration nobody has run can always be deleted. It leaves its benchmark
version's contest in the same transaction, so nothing afterwards can name it:
when it was the winner, that version simply has no winner, which is a normal
state. A registration handed to a runner is still refused.

Integration throughout, per `.cursor/rules/testing_rules.mdc`: real
registrations and the control tables of the workflow's real live DB, a real
`ObservabilityStore` written through a real `ExperimentController`, and the real
`ChatbotServer` over a real socket. No Mock fixtures, and nothing here runs a
model or spends anything -- every experiment below is an identity and a
declaration.
"""

from __future__ import annotations

import threading

import pytest

from fastworkflow import state_paths
from fastworkflow.benchmark import setup
from fastworkflow.experiment.runner import ExperimentController
from fastworkflow.observability import store as obs
from fastworkflow.run_chatbot import server as run_chatbot_server
from tests.test_chatbot_benchmarks import _request

HUMAN = {"actor": "dhar", "actor_kind": "human"}


@pytest.fixture
def folder(tmp_path, monkeypatch):
    """A live workflow whose live DB lives under a temp state root."""
    monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "state"))
    workflow = tmp_path / "roster_workflow"
    workflow.mkdir()
    (workflow / "_commands").mkdir()
    obs.ObservabilityStore(state_paths.observability_db(str(workflow)))
    return workflow


@pytest.fixture
def benchmark_id(folder):
    return setup.save_benchmark(
        folder, {"title": "Roster review", "tasks": [{"prompt": "Review the roster"}]}
    )["benchmark_id"]


def _create(folder, benchmark_id):
    return setup.create_experiment(folder, benchmark_id, "v1")["experiment_id"]


def _control(folder):
    return setup.workflow_control(folder, write=True)


def _set_winner(folder, experiment_id):
    _control(folder).set_winner(experiment_id, provenance="human", **HUMAN)


def _winner_id(folder, experiment_id):
    winner = setup.workflow_winner(folder, experiment_id)
    return None if winner is None else winner["experiment_id"]


def _declare(folder, record):
    """Hand a registration to a real runner, the way `run` does."""
    store = obs.ObservabilityStore(state_paths.observability_db(str(folder)))
    controller = ExperimentController(
        str(folder), store.store_identity(), external=False,
    )
    controller.create_experiment(
        record["experiment_id"],
        record.get("description", ""),
        declared_tasks=len(record["task_ids"]),
        declared_attempts=1,
        declarations=[
            (task_id, 1, f"ch-{record['experiment_id'][-4:]}-{task_id}")
            for task_id in record["task_ids"]
        ],
        workflow_name=setup.workflow_name_for(folder),
    )
    return store


class TestDeletion:
    def test_deleting_a_non_winner_leaves_the_winner_alone(self, folder, benchmark_id):
        winner = _create(folder, benchmark_id)
        other = _create(folder, benchmark_id)
        _set_winner(folder, winner)

        setup.delete_empty_experiment(folder, other)

        assert _winner_id(folder, winner) == winner
        group = _control(folder).group_for_experiment(winner)
        members = _control(folder).group_members(str(group["group_id"]))
        assert [row["experiment_id"] for row in members] == [winner]

    def test_deleting_the_winner_leaves_the_version_with_no_winner(
        self, folder, benchmark_id
    ):
        winner = _create(folder, benchmark_id)
        other = _create(folder, benchmark_id)
        _set_winner(folder, winner)

        setup.delete_empty_experiment(folder, winner)

        assert _winner_id(folder, other) is None
        assert setup.benchmark_winner(folder, benchmark_id, "v1") is None
        assert _control(folder).group_for_experiment(winner) is None

    def test_a_bound_registration_is_still_refused(self, folder, benchmark_id):
        record = setup.create_experiment(folder, benchmark_id, "v1")
        store = _declare(folder, record)

        with pytest.raises(setup.BenchmarkSetupConflict):
            setup.delete_empty_experiment(folder, record["experiment_id"])

        assert store.get_experiment(record["experiment_id"]) is not None
        assert _control(folder).group_for_experiment(record["experiment_id"]) is not None


@pytest.fixture
def server(folder):
    srv = run_chatbot_server.ChatbotServer(
        db_path="",
        workflow_path=str(folder),
        port=0,
        spawn_options={"no_server": True},
    )
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    thread.join(timeout=5)


def _create_over_http(server, benchmark_id):
    status, payload = _request(
        server, f"/api/benchmarks/{benchmark_id}/experiments", "POST", {"version": "v1"}
    )
    assert status == 201
    return payload["experiment"]["experiment_id"]


class TestOverHttp:
    def test_the_winner_endpoint_never_names_a_deleted_experiment(
        self, server, benchmark_id
    ):
        first = _create_over_http(server, benchmark_id)
        second = _create_over_http(server, benchmark_id)
        assert _request(
            server, f"/api/experiments/{first}/winner", "POST", HUMAN
        )[0] == 200

        assert _request(
            server, f"/api/benchmark-experiments/{first}", "DELETE"
        )[0] == 200

        status, payload = _request(server, f"/api/experiments/{second}/winner")
        assert status == 200
        assert payload["winner"] is None
        assert payload["is_winner"] is False
        assert [row["experiment_id"] for row in payload["members"]] == [second]
        # The deleted experiment is not a member of anything any more.
        assert _request(server, f"/api/experiments/{first}/winner")[0] == 404

    def test_can_delete_is_true_for_any_unbound_registration(
        self, server, folder, benchmark_id
    ):
        winner = _create_over_http(server, benchmark_id)
        other = _create_over_http(server, benchmark_id)
        assert _request(
            server, f"/api/experiments/{winner}/winner", "POST", HUMAN
        )[0] == 200

        for experiment_id in (winner, other):
            status, detail = _request(server, f"/api/benchmark-experiments/{experiment_id}")
            assert status == 200
            assert detail["can_delete"] is True
        assert _request(
            server, f"/api/benchmark-experiments/{winner}"
        )[1]["is_winner"] is True

    def test_a_bound_registration_is_not_deletable(self, server, folder, benchmark_id):
        record = setup.create_experiment(folder, benchmark_id, "v1")
        _declare(folder, record)
        experiment_id = record["experiment_id"]

        status, detail = _request(server, f"/api/benchmark-experiments/{experiment_id}")
        assert status == 200 and detail["can_delete"] is False
        assert _request(
            server, f"/api/benchmark-experiments/{experiment_id}", "DELETE"
        )[0] == 409

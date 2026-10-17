"""An experiment is checked against its benchmark version (`fix-mgoi`).

Three things, through a real controller, real stores and the real HTTP server:
the task ids an experiment may declare, whether every task got its runs
(complete | incomplete), and whether every run sent its task's saved prompt
(valid | invalid).
"""

import pytest

from fastworkflow import state_paths
from fastworkflow.benchmark import setup
from fastworkflow.benchmark.conformance import experiment_conformance
from fastworkflow.benchmark.catalog import load_version, write_version
from fastworkflow.experiment.runner import ExperimentController, channel_for
from fastworkflow.observability import store as obs
from tests.test_chatbot_benchmarks import _request
from tests.test_experiment_container import _turn_row, _write_turn
from tests.test_experiment_setup import setup_server  # noqa: F401

PROMPT = "Review the roster"


def _benchmark(folder):
    """Two tasks: the first without a prompt (payload-driven), the second with one."""
    return write_version(
        folder,
        {
            "benchmark_id": "roster-review",
            "version": "v1",
            "title": "Roster review",
            "description": "",
            "tasks": [
                {"task_id": "task_driven", "payload": {"script": ["hi"]}},
                {"task_id": "task_prompted", "prompt": PROMPT, "payload": {}},
            ],
        },
    )


def _controller(folder):
    store = obs.ObservabilityStore(state_paths.observability_db(str(folder)))
    return ExperimentController(str(folder), store.store_identity(), external=False), store


def _pin(manifest):
    return {
        "benchmark_id": manifest["benchmark_id"],
        "benchmark_version": manifest["version"],
        "benchmark_digest_sha256": manifest["digest_sha256"],
    }


def _create(controller, experiment_id, task_ids, manifest):
    controller.create_experiment(
        experiment_id,
        "run",
        declared_tasks=len(task_ids),
        declared_attempts=1,
        declarations=[(t, 1, channel_for(experiment_id, t, 1)) for t in task_ids],
        **_pin(manifest),
    )


def _run(controller, store, experiment_id, task_id, message):
    channel = channel_for(experiment_id, task_id, 1)
    conv = store.mint_conversation_id(
        channel, experiment_id=experiment_id, task_id=task_id, attempt=1
    )
    controller.start_attempt(experiment_id, task_id, 1, channel, conversation_id=conv)
    _write_turn(store, _turn_row(
        f"{experiment_id}-{task_id}", channel, conversation_id=conv,
        experiment_id=experiment_id, task_id=task_id, attempt=1,
        user_message=message,
    ))
    controller.finish_attempt(
        experiment_id, task_id, 1, outcome="pass", outcome_source="fixture"
    )


def _conformance(folder, store, manifest, experiment_id):
    return experiment_conformance(
        store, load_version(folder, manifest["benchmark_id"], manifest["version"]),
        store.get_experiment(experiment_id),
    )


class TestDeclaredTaskIds:
    """Every experiment naming a benchmark version runs exactly its tasks."""

    def test_an_unregistered_run_of_a_subset_is_refused(self, tmp_path):
        manifest = _benchmark(tmp_path)
        controller, store = _controller(tmp_path)
        with pytest.raises(ValueError, match="task IDs"):
            _create(controller, "exp-subset", [manifest["tasks"][0]["task_id"]], manifest)
        assert store.get_experiment("exp-subset") is None

    def test_an_unregistered_run_of_invented_tasks_is_refused(self, tmp_path):
        manifest = _benchmark(tmp_path)
        controller, store = _controller(tmp_path)
        with pytest.raises(ValueError, match="task IDs"):
            _create(controller, "exp-made-up", ["task_invented", "task_other"], manifest)
        assert store.get_experiment("exp-made-up") is None

    def test_an_unregistered_run_of_the_whole_version_is_accepted(self, tmp_path):
        manifest = _benchmark(tmp_path)
        controller, store = _controller(tmp_path)
        _create(controller, "exp-all", [t["task_id"] for t in manifest["tasks"]], manifest)
        assert store.get_experiment("exp-all")["benchmark_version"] == "v1"


class TestConformance:
    def test_every_task_run_with_its_prompt_is_complete_and_valid(self, tmp_path):
        manifest = _benchmark(tmp_path)
        controller, store = _controller(tmp_path)
        first, second = (t["task_id"] for t in manifest["tasks"])
        _create(controller, "exp", [first, second], manifest)
        _run(controller, store, "exp", first, "anything: this task has no saved prompt")
        _run(controller, store, "exp", second, "  " + PROMPT + "\n")

        result = _conformance(tmp_path, store, manifest, "exp")
        assert result["completeness"] == "complete"
        assert result["validity"] == "valid"
        assert result["tasks_fully_run"] == result["benchmark_tasks"] == 2
        assert result["tasks_without_prompt"] == 1

    def test_a_task_with_no_run_is_incomplete_not_invalid(self, tmp_path):
        manifest = _benchmark(tmp_path)
        controller, store = _controller(tmp_path)
        first, second = (t["task_id"] for t in manifest["tasks"])
        _create(controller, "exp", [first, second], manifest)
        _run(controller, store, "exp", second, PROMPT)

        result = _conformance(tmp_path, store, manifest, "exp")
        assert result["completeness"] == "incomplete"
        assert result["validity"] == "valid"
        assert result["tasks_fully_run"] == 1
        assert result["short_tasks"] == [{"task_id": first, "runs": 0, "declared": 1}]

    def test_a_run_that_sent_another_prompt_is_invalid(self, tmp_path):
        manifest = _benchmark(tmp_path)
        controller, store = _controller(tmp_path)
        first, second = (t["task_id"] for t in manifest["tasks"])
        _create(controller, "exp", [first, second], manifest)
        _run(controller, store, "exp", first, "hello")
        _run(controller, store, "exp", second, "Do something else entirely")

        result = _conformance(tmp_path, store, manifest, "exp")
        assert result["completeness"] == "complete"
        assert result["validity"] == "invalid"
        assert result["prompt_mismatch_count"] == 1
        assert result["prompt_mismatches"] == [{
            "task_id": second, "attempt": 1,
            "expected": PROMPT, "sent": "Do something else entirely",
        }]

    def test_incomplete_and_invalid_are_reported_together(self, tmp_path):
        manifest = _benchmark(tmp_path)
        controller, store = _controller(tmp_path)
        first, second = (t["task_id"] for t in manifest["tasks"])
        _create(controller, "exp", [first, second], manifest)
        _run(controller, store, "exp", second, "not the prompt")

        result = _conformance(tmp_path, store, manifest, "exp")
        assert (result["completeness"], result["validity"]) == ("incomplete", "invalid")


class TestOverHttp:
    def test_the_experiment_page_and_the_version_list_carry_the_verdict(self, setup_server):
        server, folder = setup_server
        manifest = _benchmark(folder)
        controller, store = _controller(folder)
        server.db_path = store.db_path
        first, second = (t["task_id"] for t in manifest["tasks"])
        _create(controller, "exp-http", [first, second], manifest)
        _run(controller, store, "exp-http", second, "not the prompt")

        detail = _request(server, "/api/experiment/exp-http")[1]["experiment"]
        assert detail["benchmark_conformance"]["completeness"] == "incomplete"
        assert detail["benchmark_conformance"]["validity"] == "invalid"

        listing = _request(
            server, f"/api/benchmarks/{manifest['benchmark_id']}/versions/v1/experiments"
        )[1]["experiments"]
        row = next(r for r in listing if r["experiment_id"] == "exp-http")
        assert row["conformance"]["tasks_fully_run"] == 1
        assert row["conformance"]["benchmark_tasks"] == 2

    def test_an_unpinned_experiment_is_refused(self, tmp_path):
        """Ad-hoc chat needs no experiment; one without a benchmark is refused (`fix-ujat`)."""
        controller, store = _controller(tmp_path)
        with pytest.raises(obs.MissingBenchmarkPin):
            controller.create_experiment(
                "exp-adhoc", "ad hoc", declared_tasks=1, declared_attempts=1,
                declarations=[("task-invented", 1, channel_for("exp-adhoc", "task-invented", 1))],
            )
        with pytest.raises(obs.MissingBenchmarkPin):
            store.create_experiment("exp-adhoc", "ad hoc", declared_tasks=1, declared_attempts=1)
        assert store.get_experiment("exp-adhoc") is None


class TestUserSimulator:
    """A user simulator sends its own utterances, not the prompt (`fix-va9v`)."""

    def _create_simulated(self, controller, experiment_id, manifest):
        task_ids = [t["task_id"] for t in manifest["tasks"]]
        controller.create_experiment(
            experiment_id,
            "simulated",
            declared_tasks=len(task_ids),
            declared_attempts=1,
            declarations=[(t, 1, channel_for(experiment_id, t, 1)) for t in task_ids],
            user_simulator=True,
            **_pin(manifest),
        )

    def test_a_simulated_experiment_reads_simulated_user_not_invalid(self, tmp_path):
        manifest = _benchmark(tmp_path)
        controller, store = _controller(tmp_path)
        first, second = (t["task_id"] for t in manifest["tasks"])
        self._create_simulated(controller, "exp-sim", manifest)
        _run(controller, store, "exp-sim", first, "hi there")
        _run(controller, store, "exp-sim", second, "Can you look over who is on the roster?")

        result = _conformance(tmp_path, store, manifest, "exp-sim")
        assert (result["completeness"], result["validity"]) == ("complete", "simulated_user")
        assert result["user_simulator"] is True
        assert result["prompt_mismatch_count"] == 0
        assert store.get_experiment("exp-sim")["user_simulator"] is True

    def test_completeness_is_still_checked(self, tmp_path):
        manifest = _benchmark(tmp_path)
        controller, store = _controller(tmp_path)
        self._create_simulated(controller, "exp-sim", manifest)
        _run(controller, store, "exp-sim", manifest["tasks"][1]["task_id"], "anything")

        result = _conformance(tmp_path, store, manifest, "exp-sim")
        assert (result["completeness"], result["validity"]) == ("incomplete", "simulated_user")

    def test_the_flag_is_set_once_at_creation(self, tmp_path):
        manifest = _benchmark(tmp_path)
        controller, store = _controller(tmp_path)
        _create(controller, "exp", [t["task_id"] for t in manifest["tasks"]], manifest)
        with pytest.raises(ValueError, match="user_simulator"):
            self._create_simulated(controller, "exp", manifest)
        assert store.get_experiment("exp")["user_simulator"] is False

    def test_a_registered_experiment_takes_the_flag_from_its_registration(self, tmp_path):
        manifest = _benchmark(tmp_path)
        record = setup.create_experiment(
            tmp_path, manifest["benchmark_id"], "v1", user_simulator=True
        )
        assert record["user_simulator"] is True
        controller, store = _controller(tmp_path)
        task_ids = [t["task_id"] for t in manifest["tasks"]]
        declarations = [(t, 1, channel_for(record["experiment_id"], t, 1)) for t in task_ids]
        with pytest.raises(ValueError, match="user_simulator differs"):
            controller.create_experiment(
                record["experiment_id"], "", declared_tasks=2, declared_attempts=1,
                declarations=declarations, registered=True, user_simulator=False,
            )
        controller.create_experiment(
            record["experiment_id"], "", declared_tasks=2, declared_attempts=1,
            declarations=declarations, registered=True,
        )
        assert store.get_experiment(record["experiment_id"])["user_simulator"] is True

    def test_a_duplicate_keeps_the_flag_unless_told_otherwise(self, tmp_path):
        manifest = _benchmark(tmp_path)
        source = setup.create_experiment(
            tmp_path, manifest["benchmark_id"], "v1", user_simulator=True
        )
        kept = setup.duplicate_experiment(tmp_path, source["experiment_id"])
        assert kept["user_simulator"] is True and kept["changed_fields"] == []
        flipped = setup.duplicate_experiment(
            tmp_path, source["experiment_id"], user_simulator=False
        )
        assert flipped["user_simulator"] is False
        assert flipped["changed_fields"] == ["user_simulator"]

    def test_the_http_api_sets_it_at_creation_and_shows_it(self, setup_server):
        server, folder = setup_server
        manifest = _benchmark(folder)
        path = f"/api/benchmarks/{manifest['benchmark_id']}/experiments"
        status, body = _request(server, path, "POST", {"version": "v1", "user_simulator": "yes"})
        assert status == 400
        status, body = _request(server, path, "POST", {"version": "v1", "user_simulator": True})
        assert status == 201 and body["experiment"]["user_simulator"] is True
        experiment_id = body["experiment"]["experiment_id"]
        detail = _request(server, f"/api/benchmark-experiments/{experiment_id}")[1]
        assert detail["user_simulator"] is True

        controller, store = _controller(folder)
        server.db_path = store.db_path
        task_ids = [t["task_id"] for t in manifest["tasks"]]
        controller.create_experiment(
            experiment_id, "", declared_tasks=2, declared_attempts=1, registered=True,
            declarations=[(t, 1, channel_for(experiment_id, t, 1)) for t in task_ids],
        )
        _run(controller, store, experiment_id, task_ids[1], "not the prompt")
        recorded = _request(server, f"/api/experiment/{experiment_id}")[1]["experiment"]
        assert recorded["user_simulator"] is True
        assert recorded["benchmark_conformance"]["validity"] == "simulated_user"

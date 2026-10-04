"""Browsing recorded training runs, and linking them to what ran (`fix-9eg.2`).

Integration throughout, per the repo's testing rules: real
`ObservabilityStore`s written through `record_train_run` (the method
`train.metrics_persistence` calls) and through the real `ExperimentController`
for the attempt stamps, read back through the projection and through the real
stdlib HTTP server. Nothing here trains, downloads or opens `___command_info`.

What the tests are for, in order of how much they matter:

1. The link between a runtime run and a training run is drawn ONLY from an
   identity both sides recorded. A `workflow_fingerprint` match is not it --
   `___command_info/` is under no root the fingerprint hashes, so it stays
   byte-identical across a retrain -- and neither is "the newest run". Both
   near-misses are exercised, because a projection that guesses would pass
   every test that only ever showed it one training run.
2. Held-out metrics keep the names and the dataset they were measured on.
   `in_distribution_f1` is an intent-classification measurement over synthetic
   utterances; nothing may present it as a task-success rate.
3. Absence is reported as absence, with its kind: withheld by the capture
   policy, unreadable, or simply not recorded.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import threading
from pathlib import Path

import pytest

from fastworkflow import state_paths
from fastworkflow.observability import store as obs
from fastworkflow.observability import training_history as th
from fastworkflow.run_chatbot import server as run_chatbot_server
from tests.test_chatbot_benchmarks import _request
from tests.test_observability_workspace import _manifest, _store_decl

# The published version id the trainer writes into the manifest and the binding
# server stamps onto an attempt. The same string on both sides is the ONLY
# thing that licenses a link.
VERSION_A = "20260901T101500"
VERSION_B = "20260902T120000"

# Identical on both training runs on purpose: the tree hash does not move when
# only the trained artifacts do, so this must never be what matches.
FINGERPRINT = "sha256:unchanged-across-both-retrains"

RUN_A = "20260901T101500-aaaaaaaa"
RUN_B = "20260902T120000-bbbbbbbb"
RUN_LEGACY = "20260903T130000-cccccccc"


def _metrics(version_id, **extra):
    """A metrics dict shaped like `collect_train_metrics`' real output."""
    metrics = {
        "contexts": {
            "global": {
                "thresholds": {"threshold": 0.71, "ambiguous_threshold": 0.42},
                "heldout": {
                    "context": "*",
                    "in_distribution_f1": 0.93,
                    "routing": {"top_1": 0.88, "in_list": 0.97},
                    "escalation": {"score": 0.8, "failures": []},
                },
            },
            "TodoListManager": {
                # Carried forward by a selective run: thresholds published, no
                # held-out report of its own.
                "thresholds": {"threshold": 0.66},
            },
        },
        "totals": {"in_distribution_f1": 0.91, "commands": 12},
        "commands": {
            "add_todo": {"seed_count": 8, "generated_count": 120, "row_count": 128},
        },
        "models": {
            "tiny": "google/bert_uncased_L-4_H-128_A-2",
            "large": "distilbert-base-uncased",
        },
        "seed": 1234,
        "train_duration_seconds": 61.5,
        "contexts_retrained": ["global"],
        "contexts_carried_forward": ["TodoListManager"],
    }
    if version_id is not None:
        metrics["version_id"] = version_id
    metrics.update(extra)
    return metrics


def _seed_runs(store):
    """Two versioned runs and one from the pre-versioning artifact layout."""
    store.record_train_run(
        run_id=RUN_A,
        workflow_fingerprint=FINGERPRINT,
        started_at="2026-09-01T10:00:00+00:00",
        completed_at="2026-09-01T10:15:00+00:00",
        metrics=_metrics(VERSION_A),
    )
    store.record_train_run(
        run_id=RUN_B,
        workflow_fingerprint=FINGERPRINT,
        started_at="2026-09-02T11:00:00+00:00",
        completed_at="2026-09-02T12:00:00+00:00",
        metrics=_metrics(VERSION_B, previous_version=VERSION_A),
    )
    store.record_train_run(
        run_id=RUN_LEGACY,
        workflow_fingerprint=FINGERPRINT,
        started_at="2026-09-03T12:00:00+00:00",
        completed_at="2026-09-03T13:00:00+00:00",
        metrics=_metrics(None),
    )


def _controller(db_path, store):
    from fastworkflow.experiment.runner import ExperimentController

    return ExperimentController(
        db_path, store.store_identity(), migrate=False, external=True
    )


def _stamped_attempt(controller, experiment_id, task_id, attempt, snapshot):
    """One attempt claimed by a server that stamped `snapshot` onto it.

    The real claim path, because the stamp is written there and nowhere else.
    """
    registration = controller.register_attempt(
        experiment_id,
        task_id,
        attempt,
        f"job-{task_id}-{attempt}",
        f"registered:{task_id}:{attempt}",
    )
    controller.claim_attempt(
        registration,
        server_incarnation=f"server-{attempt}",
        runtime_snapshot=snapshot,
    )


def _snapshot(version_id, *, legacy=False):
    """The subset of `runtime_readiness_snapshot` this projection reads."""
    return {
        "configuration_valid": True,
        "workflow_fingerprint": FINGERPRINT,
        "workflow_model_version": version_id,
        "workflow_model_legacy_layout": legacy,
        "capture_profile": "debug",
        "effective_features": {"decision_signals_v1": "shadow"},
        "pid": 4242,
    }


def _seed_attempts(db_path, store):
    """Two experiments: one ran on VERSION_A, one on a version nothing trained."""
    controller = _controller(db_path, store)
    controller.create_experiment(
        "exp-on-a",
        "ran on the first trained set",
        declared_tasks=1,
        declared_attempts=2,
        declarations=[("task-1", 1, "job-task-1-1"), ("task-1", 2, "job-task-1-2")],
    )
    _stamped_attempt(controller, "exp-on-a", "task-1", 1, _snapshot(VERSION_A))
    # Second attempt of the same experiment, same version: both must link.
    _stamped_attempt(controller, "exp-on-a", "task-1", 2, _snapshot(VERSION_A))

    controller.create_experiment(
        "exp-unstamped",
        "a server that recorded no snapshot at all",
        declared_tasks=1,
        declared_attempts=1,
        declarations=[("task-2", 1, "job-task-2-1")],
    )
    registration = controller.register_attempt(
        "exp-unstamped", "task-2", 1, "job-task-2-1", "registered:task-2:1"
    )
    controller.claim_attempt(registration, server_incarnation="server-bare")
    return controller


@pytest.fixture
def db_path(tmp_path) -> str:
    return str(tmp_path / "observability.sqlite3")


@pytest.fixture
def seeded_store(db_path):
    store = obs.ObservabilityStore(db_path)
    _seed_runs(store)
    _seed_attempts(db_path, store)
    return store


# ----------------------------------------------------------------------
# The projection
# ----------------------------------------------------------------------


class TestTrainingRunProjection:
    def test_runs_list_newest_first_with_what_the_manifest_recorded(
        self, seeded_store
    ):
        runs = th.list_training_runs(seeded_store)

        assert [run["run_id"] for run in runs] == [RUN_LEGACY, RUN_B, RUN_A]
        first = runs[-1]
        assert first["version_id"] == VERSION_A
        assert first["model_identity_recorded"] is True
        assert first["seed"] == 1234
        assert first["train_duration_seconds"] == 61.5
        assert first["contexts_retrained"] == ["global"]
        assert first["contexts_carried_forward"] == ["TodoListManager"]
        assert first["context_count"] == 2
        assert first["workflow_fingerprint"] == FINGERPRINT
        assert first["started_at"] == "2026-09-01T10:00:00+00:00"
        assert first["completed_at"] == "2026-09-01T10:15:00+00:00"
        assert first["metrics_status"] == th.METRICS_RECORDED

    def test_the_base_checkpoints_are_not_the_trained_sets_identity(
        self, seeded_store
    ):
        """Both runs fine-tuned the same base models and are different sets.

        Reporting `models` as the run's identity would make every training run
        of a workflow look like the same model.
        """
        runs = {run["run_id"]: run for run in th.list_training_runs(seeded_store)}

        assert runs[RUN_A]["base_models"] == runs[RUN_B]["base_models"]
        assert runs[RUN_A]["base_models"]["tiny"].startswith("google/bert_uncased")
        assert runs[RUN_A]["version_id"] != runs[RUN_B]["version_id"]

    def test_heldout_metrics_keep_their_own_names_and_dataset(self, seeded_store):
        """No rebranding: these are the classifier's held-out numbers."""
        detail = th.training_run_detail(seeded_store, RUN_A)
        contexts = {entry["context_folder"]: entry for entry in detail["contexts"]}

        heldout = contexts["global"]["heldout"]
        assert heldout["context"] == "*"
        assert heldout["in_distribution_f1"] == 0.93
        assert heldout["routing"] == {"top_1": 0.88, "in_list": 0.97}
        assert heldout["escalation"] == {"score": 0.8, "failures": []}
        # Nothing invented a task-level verdict out of them.
        assert "pass_rate" not in heldout and "task_success" not in heldout
        assert set(detail["totals"]) == {"in_distribution_f1", "commands"}

    def test_a_context_with_no_heldout_report_says_none_not_zero(
        self, seeded_store
    ):
        detail = th.training_run_detail(seeded_store, RUN_A)
        contexts = {entry["context_folder"]: entry for entry in detail["contexts"]}

        assert contexts["TodoListManager"]["thresholds"] == {"threshold": 0.66}
        assert contexts["TodoListManager"]["heldout"] is None

    def test_thresholds_and_command_counts_are_carried_through(self, seeded_store):
        detail = th.training_run_detail(seeded_store, RUN_A)
        contexts = {entry["context_folder"]: entry for entry in detail["contexts"]}

        assert contexts["global"]["thresholds"] == {
            "threshold": 0.71,
            "ambiguous_threshold": 0.42,
        }
        assert detail["commands"]["add_todo"] == {
            "seed_count": 8,
            "generated_count": 120,
            "row_count": 128,
        }
        assert detail["manifest"]["seed"] == 1234
        assert detail["manifest"]["contexts_retrained"] == ["global"]

    def test_an_unknown_run_id_is_absent_not_an_error(self, seeded_store):
        assert th.training_run_detail(seeded_store, "no-such-run") is None

    def test_a_run_older_than_the_list_window_is_still_readable(self, db_path):
        """A detail read is a primary-key read, not a search of the page.

        Resolving the detail out of the newest-first list made a recorded run
        stop existing as soon as enough newer ones did -- a 404 about the
        caller's page size, reported as a fact about the evidence. The oldest
        of 60 runs is past every default window this feature uses.
        """
        store = obs.ObservabilityStore(db_path)
        for index in range(60):
            store.record_train_run(
                run_id=f"run-{index:03d}",
                workflow_fingerprint=FINGERPRINT,
                started_at=f"2026-07-{index % 28 + 1:02d}T00:00:00+00:00",
                completed_at=f"2026-07-{index % 28 + 1:02d}T01:00:00+00:00",
                metrics=_metrics(f"version-{index:03d}", seed=index),
            )

        oldest = "run-000"
        assert oldest not in {
            run["run_id"] for run in th.list_training_runs(store, limit=50)
        }, "the fixture must actually push it out of the window"

        detail = th.training_run_detail(store, oldest)

        assert detail is not None
        assert detail["run_id"] == oldest
        assert detail["version_id"] == "version-000"
        assert detail["seed"] == 0
        assert detail["contexts"], "the old run's metrics come back whole"


# ----------------------------------------------------------------------
# The runtime link
# ----------------------------------------------------------------------


class TestRuntimeLink:
    def test_every_attempt_that_stamped_the_version_is_linked(self, seeded_store):
        detail = th.training_run_detail(seeded_store, RUN_A)
        link = detail["runtime_link"]

        assert link["status"] == th.LINK_LINKED
        assert link["version_id"] == VERSION_A
        assert [(row["experiment_id"], row["attempt"]) for row in link["attempts"]] == [
            ("exp-on-a", 1),
            ("exp-on-a", 2),
        ]
        assert link["attempts"][0]["task_id"] == "task-1"
        assert link["scan_bounded"] is False

    def test_a_matching_fingerprint_is_not_a_link(self, seeded_store):
        """The near-miss that makes the positive case non-vacuous.

        Both training runs carry the same `workflow_fingerprint` as every
        stamped attempt, and RUN_B is also the newest versioned run. Neither is
        evidence, so RUN_B has to come back as no match while RUN_A links.
        """
        link = th.training_run_detail(seeded_store, RUN_B)["runtime_link"]

        assert link["status"] == th.LINK_NO_MATCH
        assert link["attempts"] == []
        assert link["version_id"] == VERSION_B
        assert "stamped this version id" in link["reason"]
        # The fingerprint it would have matched on is on record for both.
        assert all(
            row["workflow_fingerprint"] == FINGERPRINT
            for row in th.training_run_detail(seeded_store, RUN_A)["runtime_link"][
                "attempts"
            ]
        )

    def test_a_run_with_no_recorded_version_is_unavailable_not_unmatched(
        self, seeded_store
    ):
        """"Unavailable" and "nothing ran on it" are different statements.

        A workflow on the pre-versioning artifact layout is fully trained and
        records no version id, so there is nothing to match in either
        direction -- which is not the same as having looked and found none.
        """
        link = th.training_run_detail(seeded_store, RUN_LEGACY)["runtime_link"]

        assert link["status"] == th.LINK_UNAVAILABLE
        assert link["version_id"] is None
        assert link["attempts"] == []
        assert "no published version id" in link["reason"]

    def test_an_attempt_with_no_snapshot_is_never_linked(self, seeded_store):
        """The unstamped experiment must not be swept in by anything."""
        for run_id in (RUN_A, RUN_B, RUN_LEGACY):
            link = th.training_run_detail(seeded_store, run_id)["runtime_link"]
            assert all(
                row["experiment_id"] != "exp-unstamped" for row in link["attempts"]
            )

    def test_a_truncated_scan_says_so(self, seeded_store):
        """`no_match` over a bounded window is a weaker claim, and says which."""
        link = th.runtime_link(seeded_store, VERSION_A, experiment_scan=1)

        assert link["scan_bounded"] is True
        assert link["experiments_scanned"] == 1
        unbounded = th.runtime_link(seeded_store, VERSION_A)
        assert unbounded["scan_bounded"] is False
        assert unbounded["experiments_scanned"] == 2


# ----------------------------------------------------------------------
# Absent, withheld and unreadable metrics
# ----------------------------------------------------------------------


class TestMissingMetricsAreExplicit:
    def test_an_evidence_profile_withholds_the_numbers_and_keeps_the_run(
        self, db_path, monkeypatch
    ):
        """The default-deny profile stores a badge in `metrics_json`.

        The run still lists, because the other four columns are unpoliced --
        and the badge is reported as `withheld`, never rendered as data.
        """
        monkeypatch.setenv(obs.CAPTURE_PROFILE_VAR, "evidence")
        store = obs.ObservabilityStore(db_path)
        store.record_train_run(
            run_id=RUN_A,
            workflow_fingerprint=FINGERPRINT,
            started_at="2026-09-01T10:00:00+00:00",
            completed_at="2026-09-01T10:15:00+00:00",
            metrics=_metrics(VERSION_A),
        )

        run = th.list_training_runs(store)[0]

        assert run["metrics_status"] == th.METRICS_WITHHELD
        assert run["run_id"] == RUN_A
        assert run["workflow_fingerprint"] == FINGERPRINT
        assert run["completed_at"] == "2026-09-01T10:15:00+00:00"
        # Nothing was invented to fill the hole.
        assert run["version_id"] is None
        assert run["model_identity_recorded"] is False
        assert run["contexts_retrained"] == []
        assert run["seed"] is None
        detail = th.training_run_detail(store, RUN_A)
        assert detail["contexts"] == [] and detail["totals"] == {}
        assert detail["runtime_link"]["status"] == th.LINK_UNAVAILABLE

    def test_an_unreadable_or_absent_blob_is_named_rather_than_guessed(self):
        """Defensive, and the only states `record_train_run` cannot produce.

        A store written by another build, or one whose row was damaged, still
        has to read as something a viewer can show.
        """
        assert th.decode_metrics(None) == (th.METRICS_ABSENT, {})
        assert th.decode_metrics("") == (th.METRICS_ABSENT, {})
        assert th.decode_metrics("{not json") == (th.METRICS_UNREADABLE, {})
        assert th.decode_metrics("[1, 2]") == (th.METRICS_UNREADABLE, {})
        status, metrics = th.decode_metrics('{"version_id": "v"}')
        assert status == th.METRICS_RECORDED and metrics == {"version_id": "v"}


# ----------------------------------------------------------------------
# Over HTTP
# ----------------------------------------------------------------------


@pytest.fixture
def workflow_dir(tmp_path) -> Path:
    workflow = tmp_path / "workflow"
    workflow.mkdir()
    (workflow / "_commands").mkdir()
    return workflow


@pytest.fixture
def training_server(workflow_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "state"))
    db_path = state_paths.observability_db(str(workflow_dir))
    store = obs.ObservabilityStore(db_path)
    _seed_runs(store)
    _seed_attempts(db_path, store)
    server = run_chatbot_server.ChatbotServer(
        db_path=db_path,
        workflow_path=str(workflow_dir),
        port=0,
        spawn_options={"no_server": True},
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, store
    server.shutdown()
    thread.join(timeout=5)


class TestTrainingHistoryOverHttp:
    def test_the_list_route_returns_the_projection(self, training_server):
        server, _store = training_server

        status, data = _request(server, "/api/training-runs")

        assert status == 200
        assert [run["run_id"] for run in data["training_runs"]] == [
            RUN_LEGACY,
            RUN_B,
            RUN_A,
        ]
        assert data["training_runs"][-1]["version_id"] == VERSION_A

    def test_the_detail_route_carries_the_link_and_the_metrics(
        self, training_server
    ):
        server, _store = training_server

        status, data = _request(server, f"/api/training-run/{RUN_A}")

        assert status == 200
        run = data["training_run"]
        assert run["run_id"] == RUN_A
        assert run["runtime_link"]["status"] == th.LINK_LINKED
        assert len(run["runtime_link"]["attempts"]) == 2
        contexts = {entry["context_folder"]: entry for entry in run["contexts"]}
        assert contexts["global"]["heldout"]["in_distribution_f1"] == 0.93

    def test_an_unknown_run_is_a_404(self, training_server):
        server, _store = training_server

        status, data = _request(server, "/api/training-run/no-such-run")

        assert status == 404
        assert "training run not found" in data["error"]

    def test_the_limit_is_bounded_on_both_ends(self, training_server):
        server, _store = training_server

        assert len(_request(server, "/api/training-runs?limit=1")[1]["training_runs"]) == 1
        # Garbage falls back to the default rather than erroring or being
        # passed through to SQLite.
        assert len(
            _request(server, "/api/training-runs?limit=banana")[1]["training_runs"]
        ) == 3
        assert len(
            _request(server, "/api/training-runs?limit=100000")[1]["training_runs"]
        ) == 3

    def test_clicking_an_older_row_of_a_long_list_returns_that_run(
        self, workflow_dir, tmp_path, monkeypatch
    ):
        """The review finding, at the route that would have shipped it.

        The page asks for a page of rows and then asks for one of them by id.
        While the detail was resolved out of a 50-row window, a row returned
        by `?limit=200` past position 50 answered 404 -- the list offering a
        run the detail said did not exist.
        """
        monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "long"))
        db_path = state_paths.observability_db(str(workflow_dir))
        store = obs.ObservabilityStore(db_path)
        for index in range(60):
            store.record_train_run(
                run_id=f"run-{index:03d}",
                workflow_fingerprint=FINGERPRINT,
                started_at=f"2026-07-{index % 28 + 1:02d}T00:00:00+00:00",
                completed_at=f"2026-07-{index % 28 + 1:02d}T01:00:00+00:00",
                metrics=_metrics(f"version-{index:03d}", seed=index),
            )
        server = run_chatbot_server.ChatbotServer(
            db_path=db_path,
            workflow_path=str(workflow_dir),
            port=0,
            spawn_options={"no_server": True},
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            listed = _request(server, "/api/training-runs?limit=200")[1][
                "training_runs"
            ]
            assert len(listed) == 60
            older = listed[55]
            status, data = _request(server, f"/api/training-run/{older['run_id']}")
            # ...and it is still that run when the list was cut short, which is
            # what the page's own default asks for.
            short = _request(server, "/api/training-runs?limit=10")[1][
                "training_runs"
            ]
            repeat = _request(server, f"/api/training-run/{older['run_id']}")
        finally:
            server.shutdown()
            thread.join(timeout=5)

        assert status == 200, data
        assert data["training_run"]["run_id"] == older["run_id"]
        assert data["training_run"]["version_id"] == older["version_id"]
        assert older["run_id"] not in {run["run_id"] for run in short}
        assert repeat[0] == 200 and repeat[1] == data

    def test_the_route_needs_the_token_like_every_other_read(self, training_server):
        server, _store = training_server

        assert _request(server, "/api/training-runs", token=None)[0] == 401

    def test_a_workflow_with_no_store_lists_nothing_rather_than_404(
        self, workflow_dir, tmp_path, monkeypatch
    ):
        """A cold start has no training runs; that is a fact, not an error."""
        monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "cold"))
        server = run_chatbot_server.ChatbotServer(
            db_path="",
            workflow_path=str(workflow_dir),
            port=0,
            spawn_options={"no_server": True},
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            status, data = _request(server, "/api/training-runs")
        finally:
            server.shutdown()
            thread.join(timeout=5)

        assert status == 200
        assert data == {"training_runs": []}


# ----------------------------------------------------------------------
# The real page in a real DOM
# ----------------------------------------------------------------------


class TestTrainingHistoryInThePage:
    def test_the_section_is_reachable_without_touching_the_rail(self):
        """The nav hook is one button in the menu that already holds the tools.

        Pinned in the source because the alternative -- a third rail tab --
        would have had to reach into the hierarchy machinery, and a later
        refactor that moved it there would silently change what a background
        navigation refresh is allowed to repaint.
        """
        page = run_chatbot_server.load_index_html()

        assert b'<button id="trainingHistoryBtn">Training history</button>' in page
        assert b'<dialog id="trainingDialog"' in page
        # [R22] and the page's packaging rules still hold.
        assert b"innerHTML" not in page
        assert b"https://" not in page

    def test_the_page_never_infers_a_model_link(self):
        """A correctness rule, so it is pinned in the source.

        The page builds its link from `runtime_link`, which the server derives
        from stamped version ids only. A client-side fallback that matched on
        `workflow_fingerprint` -- which does not move across a retrain -- would
        pass every behavioural test on a store holding one training run.
        """
        page = run_chatbot_server.load_index_html().decode("utf-8")
        body = page.split("function trainingRuntimeLink(", 1)[1].split(
            "\nfunction ", 1
        )[0]

        assert "workflow_fingerprint" not in body
        assert "link.status" in body and '"unavailable"' in body


# Module level, not a method of the class above: `tests.browser_validation`
# discovers required browser checks by walking a module's TOP-LEVEL functions,
# so a DOM test nested in a class is invisible to the gate -- it would skip
# itself on a machine without jsdom and the gate would report nothing missing.
def test_training_history_dom(training_server):
    dependency = os.environ.get("TEST_JSDOM_ROOT")
    if not dependency:
        pytest.skip("Set TEST_JSDOM_ROOT to run DOM integration with jsdom")
    server, _store = training_server
    script = Path(__file__).with_name("chatbot_training_history_dom.cjs")
    result = subprocess.run(
        [
            "node",
            str(script),
            dependency,
            f"http://127.0.0.1:{server.port}/?token={server.token}",
            json.dumps(
                {
                    "version_a": VERSION_A,
                    "version_b": VERSION_B,
                    "legacy": RUN_LEGACY,
                    "fingerprint": FINGERPRINT,
                }
            ),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


# ----------------------------------------------------------------------
# Workspace mode: an explicit source, and ids that collide across stores
# ----------------------------------------------------------------------


def _training_archive(root: Path, name: str, version_id: str, seed: int) -> dict:
    """A sealed archive holding ONE training run, under a colliding run_id."""
    source = root / f"{name}-live.sqlite3"
    store = obs.ObservabilityStore(str(source))
    store.record_train_run(
        run_id=RUN_A,
        workflow_fingerprint=FINGERPRINT,
        started_at="2026-09-01T10:00:00+00:00",
        completed_at="2026-09-01T10:15:00+00:00",
        metrics=_metrics(version_id, seed=seed),
    )
    return obs.ObservabilityStore(str(source), migrate=False).archive_to(
        str(root / f"{name}.sqlite3")
    )


@pytest.fixture
def workspace_training_server(tmp_path, monkeypatch):
    monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "state"))
    left = _training_archive(tmp_path, "left", VERSION_A, seed=1)
    right = _training_archive(tmp_path, "right", VERSION_B, seed=2)
    manifest = _manifest(
        tmp_path, [_store_decl(left, "left"), _store_decl(right, "right")]
    )
    server = run_chatbot_server.ChatbotServer(
        port=0, workspace_manifest_path=str(manifest)
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, tmp_path
    server.shutdown()
    thread.join(timeout=5)


class TestWorkspaceTrainingHistory:
    def test_an_unscoped_read_is_refused_and_names_the_scoped_route(
        self, workspace_training_server
    ):
        server, _root = workspace_training_server

        status, data = _request(server, "/api/training-runs")

        assert status == 400
        assert "/api/workspace/training-runs?store_id=" in data["error"]
        assert _request(server, f"/api/training-run/{RUN_A}")[0] == 400

    def test_a_store_id_is_required(self, workspace_training_server):
        server, _root = workspace_training_server

        status, data = _request(server, "/api/workspace/training-runs")

        assert status == 400
        assert "never listed across stores" in data["error"]

    def test_colliding_run_ids_resolve_to_the_named_store(
        self, workspace_training_server
    ):
        """One run_id, two stores, two different training runs.

        This is why the workspace routes take the store explicitly instead of
        searching: the ids are each store's own and nothing makes them unique
        across a workspace.
        """
        server, _root = workspace_training_server

        left = _request(server, f"/api/workspace/training-run/left/{RUN_A}")[1]
        right = _request(server, f"/api/workspace/training-run/right/{RUN_A}")[1]

        assert left["training_run"]["version_id"] == VERSION_A
        assert right["training_run"]["version_id"] == VERSION_B
        assert left["training_run"]["seed"] == 1
        assert right["training_run"]["seed"] == 2
        assert left["training_run"]["store_id"] == "left"
        assert right["training_run"]["store_id"] == "right"

    def test_an_old_run_in_a_sealed_archive_is_readable_too(self, tmp_path, monkeypatch):
        """The same primary-key rule on the read-only path.

        A sealed archive is exactly where an old training run lives, so a
        detail read that depended on the newest-first window would fail worst
        on the stores this feature exists to inspect.
        """
        monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "state"))
        source = tmp_path / "long-live.sqlite3"
        store = obs.ObservabilityStore(str(source))
        for index in range(60):
            store.record_train_run(
                run_id=f"run-{index:03d}",
                workflow_fingerprint=FINGERPRINT,
                started_at=f"2026-07-{index % 28 + 1:02d}T00:00:00+00:00",
                completed_at=f"2026-07-{index % 28 + 1:02d}T01:00:00+00:00",
                metrics=_metrics(f"version-{index:03d}", seed=index),
            )
        archive = obs.ObservabilityStore(str(source), migrate=False).archive_to(
            str(tmp_path / "long.sqlite3")
        )
        manifest = _manifest(tmp_path, [_store_decl(archive, "long")])
        server = run_chatbot_server.ChatbotServer(
            port=0, workspace_manifest_path=str(manifest)
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            listed = _request(
                server, "/api/workspace/training-runs?store_id=long&limit=10"
            )[1]["training_runs"]
            status, data = _request(
                server, "/api/workspace/training-run/long/run-000"
            )
        finally:
            server.shutdown()
            thread.join(timeout=5)

        assert len(listed) == 10
        assert "run-000" not in {run["run_id"] for run in listed}
        assert status == 200, data
        assert data["training_run"]["version_id"] == "version-000"
        assert data["training_run"]["store_id"] == "long"

    def test_the_scoped_list_names_its_store(self, workspace_training_server):
        server, _root = workspace_training_server

        status, data = _request(server, "/api/workspace/training-runs?store_id=left")

        assert status == 200
        assert data["store_id"] == "left"
        assert [run["run_id"] for run in data["training_runs"]] == [RUN_A]

    def test_an_unknown_store_is_refused_rather_than_resolved(
        self, workspace_training_server
    ):
        server, _root = workspace_training_server

        status, data = _request(
            server, "/api/workspace/training-runs?store_id=nowhere"
        )

        # The registry refuses a store the manifest never named; it does not
        # fall back to a path, so no read can escape the declared set.
        assert status == 404
        assert "nowhere" in json.dumps(data)
        assert (
            _request(server, f"/api/workspace/training-run/nowhere/{RUN_A}")[0] == 404
        )

    def test_reading_a_sealed_archive_does_not_touch_its_bytes(
        self, workspace_training_server
    ):
        """[R12]: a viewer opened on a snapshot must not migrate or write it."""
        server, root = workspace_training_server
        archive = root / "left.sqlite3"
        before = (archive.stat().st_mtime_ns, archive.read_bytes())

        assert _request(server, f"/api/workspace/training-run/left/{RUN_A}")[0] == 200

        assert (archive.stat().st_mtime_ns, archive.read_bytes()) == before
        assert not (root / "left.sqlite3-wal").exists()


# ----------------------------------------------------------------------
# The source boundary: one source, one store, and reads that outlive both
# ----------------------------------------------------------------------


class TestTheSectionIsScopedToItsSource:
    """The page's guards against a response from the store the reader left.

    A stale response here is not a cosmetic flicker. Two archives in one
    workspace may hold the same `run_id`, so a list that lands after the
    reader has switched leaves rows naming runs that exist in the other
    archive -- and a click then resolves that id against the current store,
    which answers with a different training run under the same name. Both
    halves have to be guarded: the token, which moves when any read starts,
    and the source, which is captured with the request and re-checked when it
    lands.
    """

    @staticmethod
    def _body(name):
        page = run_chatbot_server.load_index_html().decode("utf-8")
        return page.split(f"function {name}(", 1)[1].split("\nfunction ", 1)[0]

    def test_the_store_is_part_of_the_identity_a_read_is_checked_against(self):
        body = self._body("trainingSource")
        assert "sourceIdentity(session)" in body and "trainingHistory.storeId" in body

    def test_starting_a_list_disowns_a_detail_already_in_flight(self):
        """Otherwise a detail from the previous store survives until the next
        list happens to finish, which is a race the reader wins or loses by
        timing."""
        body = self._body("loadTrainingRuns")
        assert "trainingToken()" in body
        assert body.index("trainingToken()") < body.index("api(path)")
        assert "trainingStale(token, source)" in body

    def test_a_detail_url_is_built_when_the_click_happens(self):
        """Building it when the response lands would resolve the id against
        whichever store is selected by then."""
        body = self._body("selectTrainingRun")
        assert body.index("trainingDetailPath(runId)") < body.index("api(path)")
        assert "trainingStale(token, source)" in body

    def test_the_page_source_boundary_stands_the_section_down(self):
        """`resetSourceScopedState` reaches this section through
        `onSourceSwitch`; without the call the dialog would keep showing the
        previous source's evidence."""
        page = run_chatbot_server.load_index_html().decode("utf-8")
        hook = page.split("function onSourceSwitch(", 1)[1].split("\nfunction ", 1)[0]
        assert "trainingHistoryReset()" in hook
        reset = self._body("trainingHistoryReset")
        assert "trainingHistory.seq++" in reset
        assert "trainingHistory.storeId = null" in reset


def test_training_source_switch_dom(workspace_training_server):
    """Real server, real DOM, colliding run ids across two sealed archives."""
    dependency = os.environ.get("TEST_JSDOM_ROOT")
    if not dependency:
        pytest.skip("Set TEST_JSDOM_ROOT to run DOM integration with jsdom")
    server, _root = workspace_training_server
    script = Path(__file__).with_name("chatbot_training_source_switch_dom.cjs")
    result = subprocess.run(
        [
            "node",
            str(script),
            dependency,
            f"http://127.0.0.1:{server.port}/?token={server.token}",
            json.dumps(
                {
                    "version_a": VERSION_A,
                    "version_b": VERSION_B,
                    "run_id": RUN_A,
                }
            ),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr


# ----------------------------------------------------------------------
# Nothing in this feature writes
# ----------------------------------------------------------------------


def test_the_projection_opens_its_store_read_only(db_path):
    """The whole surface is reads of rows that already exist.

    Pinned against the read-only store rather than argued: if any projection
    call ever needed a write, this is where it would fail.
    """
    writable = obs.ObservabilityStore(db_path)
    _seed_runs(writable)
    _seed_attempts(db_path, writable)
    before = Path(db_path).read_bytes()

    readonly = obs.ReadOnlyObservabilityStore(db_path)
    runs = th.list_training_runs(readonly)
    detail = th.training_run_detail(readonly, RUN_A)

    assert len(runs) == 3
    assert detail["runtime_link"]["status"] == th.LINK_LINKED
    assert Path(db_path).read_bytes() == before
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM train_runs").fetchone()[0] == 3

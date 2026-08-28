"""Phase 6 observability: train-time metrics persisted into train_runs (fix-kw7.7).

Tests the metrics-persistence seam directly against a real SQLite store in a
tmp state root — no mocks of fastworkflow components, no actual BERT training,
and (fix-0hb) nothing written into any bundled workflow's ___command_info: the
fixture ``___command_info`` layout is constructed in tmp_path, mimicking the
shape of a real published workflow (examples/hello_world).
"""

from __future__ import annotations

import json
import os
import stat
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import fastworkflow
from fastworkflow import state_paths
from fastworkflow.observability_store import ObservabilityStore
from fastworkflow.train import metrics_persistence


@pytest.fixture
def state_root(tmp_path, monkeypatch):
    """Real tmp state root; observability defaults ON for the train entry point."""
    root = tmp_path / "state"
    monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(root))
    monkeypatch.delenv("FW_OBSERVABILITY", raising=False)
    fastworkflow.init({"FASTWORKFLOW_STATE_ROOT": str(root)})
    return root


@pytest.fixture
def workflow_dir(tmp_path) -> str:
    """A minimal workflow folder with one command source (fingerprintable)."""
    wf = tmp_path / "metrics_wf"
    commands = wf / "_commands"
    commands.mkdir(parents=True)
    (commands / "do_thing.py").write_text("# a command source\n", encoding="utf-8")
    return str(wf)


def _db_path(workflow_dir: str) -> str:
    return state_paths.observability_db(workflow_dir)


# ---------------------------------------------------------------------------
# (a) persist_train_run_metrics writes a readable row
# ---------------------------------------------------------------------------


def test_persist_writes_readable_row(state_root, workflow_dir):
    completed = datetime.now(timezone.utc)
    started = completed - timedelta(seconds=42)
    metrics = {
        "contexts": {"global": {"thresholds": {"threshold": 0.419}}},
        "totals": {"routing_top1": 0.88},
    }

    run_id = metrics_persistence.persist_train_run_metrics(
        workflow_dir, started_at=started, completed_at=completed, metrics=metrics
    )
    assert run_id is not None
    # Sortable id: completion stamp then a uuid suffix.
    assert run_id.startswith(f"{completed:%Y%m%dT%H%M%S}-")

    store = ObservabilityStore(_db_path(workflow_dir))
    rows = store.list_train_runs()
    assert len(rows) == 1
    row = rows[0]
    assert row["run_id"] == run_id
    assert row["started_at"] == started.isoformat()
    assert row["completed_at"] == completed.isoformat()
    # The workflow has real command sources, so the fingerprint is present.
    assert isinstance(row["workflow_fingerprint"], str)
    assert row["workflow_fingerprint"]
    assert json.loads(row["metrics_json"]) == metrics


def test_persist_is_idempotent_per_run_id(state_root, workflow_dir):
    """record_train_run upserts by run_id — replaying a row does not duplicate it."""
    completed = datetime.now(timezone.utc)
    run_id = metrics_persistence.persist_train_run_metrics(
        workflow_dir,
        started_at=completed,
        completed_at=completed,
        metrics={"totals": {}},
    )
    assert run_id is not None

    store = ObservabilityStore(_db_path(workflow_dir))
    store.record_train_run(
        run_id=run_id,
        workflow_fingerprint="fp2",
        started_at=completed.isoformat(),
        completed_at=completed.isoformat(),
        metrics={"totals": {"replayed": True}},
    )
    rows = store.list_train_runs()
    assert len(rows) == 1
    assert json.loads(rows[0]["metrics_json"]) == {"totals": {"replayed": True}}


# ---------------------------------------------------------------------------
# (b) FW_OBSERVABILITY=0 -> no row, no error
# ---------------------------------------------------------------------------


def test_disabled_writes_nothing(state_root, workflow_dir, monkeypatch):
    monkeypatch.setenv("FW_OBSERVABILITY", "0")
    run_id = metrics_persistence.persist_train_run_metrics(
        workflow_dir,
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
        metrics={"totals": {}},
    )
    assert run_id is None
    # The gate short-circuits before the store is even opened.
    assert not os.path.exists(_db_path(workflow_dir))


# ---------------------------------------------------------------------------
# (c) broken DB -> warns and returns None without raising
# ---------------------------------------------------------------------------


def test_broken_db_warns_and_returns_none(state_root, workflow_dir, caplog):
    from fastworkflow.utils.logging import logger as fw_logger

    db_path = _db_path(workflow_dir)
    ObservabilityStore(db_path)  # create a healthy DB first
    os.chmod(db_path, stat.S_IRUSR)  # 0400: schema write-probe must fail
    # The fastWorkflow logger does not propagate; capture it directly.
    fw_logger.addHandler(caplog.handler)
    try:
        run_id = metrics_persistence.persist_train_run_metrics(
            workflow_dir,
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            metrics={"totals": {}},
        )
    finally:
        fw_logger.removeHandler(caplog.handler)
        os.chmod(db_path, stat.S_IRUSR | stat.S_IWUSR)
    assert run_id is None
    assert any(
        "Could not persist train-run metrics" in record.getMessage()
        for record in caplog.records
    )


# ---------------------------------------------------------------------------
# (d) collect_train_metrics against a fixture ___command_info layout
# ---------------------------------------------------------------------------


@pytest.fixture
def published_workflow(tmp_path) -> tuple[str, str]:
    """A tmp workflow with a fixture ___command_info mimicking a real publish.

    Shapes copied from examples/hello_world/___command_info (read, not
    modified): per-context threshold files, the merged heldout_evaluation.json,
    training_report.json rows, and a versions/<id>/manifest.json.
    """
    wf = tmp_path / "published_wf"
    info = wf / "___command_info"
    version_id = "20260825T120000Z-abc123"

    global_dir = info / "global"
    global_dir.mkdir(parents=True)
    (global_dir / "threshold.json").write_text(
        json.dumps({"confidence_threshold": 0.4190716908166283}), encoding="utf-8"
    )
    (global_dir / "tiny_ambiguous_threshold.json").write_text(
        json.dumps({"confidence_threshold": 0.3932633697986603}), encoding="utf-8"
    )
    (global_dir / "large_ambiguous_threshold.json").write_text(
        json.dumps({"confidence_threshold": 0.4614776074886322}), encoding="utf-8"
    )
    todo_dir = info / "TodoItem"
    todo_dir.mkdir()
    (todo_dir / "threshold.json").write_text(
        json.dumps({"confidence_threshold": 0.51}), encoding="utf-8"
    )
    # Reserved layout entries must never be read as context folders.
    versions_dir = info / "versions" / version_id
    versions_dir.mkdir(parents=True)
    (info / "current.json").write_text(
        json.dumps({"version_id": version_id}), encoding="utf-8"
    )
    (versions_dir / "manifest.json").write_text(
        json.dumps(
            {
                "version_id": version_id,
                "seed": 42,
                "train_duration_seconds": 32.09,
                "contexts_retrained": ["*"],
                "contexts_carried_forward": ["TodoItem"],
                "previous_version": "20260825T110000Z-000000",
            }
        ),
        encoding="utf-8",
    )
    (info / "heldout_evaluation.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "totals": {
                    "contexts": 2,
                    "routing_total": 25,
                    "routing_top1": 0.88,
                    "routing_in_list": 0.92,
                    "mean_in_distribution_f1": 0.8959,
                },
                "contexts": [
                    {
                        "context": "*",
                        "in_distribution_f1": 0.8959,
                        "routing": {"total": 25, "top1": 0.88, "in_list": 0.92},
                        "escalation": None,
                        "seed": 42,
                    },
                    {
                        "context": "TodoItem",
                        "in_distribution_f1": 0.91,
                        "routing": {"total": 10, "top1": 0.9, "in_list": 1.0},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (info / "training_report.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "command_name": "add_two_numbers",
                        "status": "thin_seeds",
                        "seed_count": 4,
                        "generated_count": 20,
                        "row_count": 26,
                    },
                    {
                        "command_name": "IntentDetection/reset_context",
                        "status": "ok",
                        "seed_count": 8,
                        "generated_count": 20,
                        "row_count": 28,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    return str(wf), version_id


def test_collect_metrics_from_published_layout(state_root, published_workflow):
    wf, version_id = published_workflow
    metrics = metrics_persistence.collect_train_metrics(wf, version_id=version_id)

    # Version manifest fields.
    assert metrics["version_id"] == version_id
    assert metrics["seed"] == 42
    assert metrics["train_duration_seconds"] == 32.09
    assert metrics["contexts_retrained"] == ["*"]
    assert metrics["contexts_carried_forward"] == ["TodoItem"]

    # Base model ids (env defaults).
    assert metrics["models"]["tiny"] == "google/bert_uncased_L-4_H-128_A-2"
    assert metrics["models"]["large"] == "distilbert-base-uncased"

    # Per-context thresholds; the "*" heldout context maps onto "global".
    assert metrics["contexts"]["global"]["thresholds"] == {
        "threshold": 0.4190716908166283,
        "tiny_ambiguous_threshold": 0.3932633697986603,
        "large_ambiguous_threshold": 0.4614776074886322,
    }
    assert metrics["contexts"]["TodoItem"]["thresholds"] == {"threshold": 0.51}
    assert metrics["contexts"]["global"]["heldout"]["in_distribution_f1"] == 0.8959
    assert metrics["contexts"]["global"]["heldout"]["routing"]["top1"] == 0.88
    assert metrics["contexts"]["TodoItem"]["heldout"]["routing"]["in_list"] == 1.0
    # Reserved layout entries are not contexts.
    assert "versions" not in metrics["contexts"]

    # Heldout totals and per-command utterance counts.
    assert metrics["totals"]["routing_top1"] == 0.88
    assert metrics["commands"]["add_two_numbers"] == {
        "seed_count": 4,
        "generated_count": 20,
        "row_count": 26,
        "status": "thin_seeds",
    }

    # End-to-end: the collected dict persists and reads back intact.
    completed = datetime.now(timezone.utc)
    run_id = metrics_persistence.persist_train_run_metrics(
        wf, started_at=completed, completed_at=completed, metrics=metrics
    )
    assert run_id is not None
    rows = ObservabilityStore(state_paths.observability_db(wf)).list_train_runs()
    assert len(rows) == 1
    assert json.loads(rows[0]["metrics_json"]) == metrics


def test_collect_metrics_missing_artifacts_degrades(state_root, tmp_path):
    """No ___command_info at all -> empty-but-valid metrics, no exception."""
    wf = tmp_path / "untrained_wf"
    wf.mkdir()
    metrics = metrics_persistence.collect_train_metrics(str(wf), version_id=None)
    assert metrics["contexts"] == {}
    assert metrics["totals"] == {}

"""Focused tests for workflow-local benchmark catalogs."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from enum import Enum

import pytest

from fastworkflow import state_paths
from fastworkflow.benchmark.catalog import (
    SCHEMA,
    BenchmarkAlreadyExistsError,
    BenchmarkManifestError,
    list_benchmarks,
    list_versions,
    load_version,
    write_version,
)
from fastworkflow.observability import control
from fastworkflow.observability import store as obs

# The canonical bytes the file-based catalog wrote for `LEGACY_SPEC`, captured
# from the pre-DB code. Their sha256 is the digest a pinned run recorded then,
# so the DB catalog must reproduce it exactly.
LEGACY_SPEC = {
    "benchmark_id": "smoke",
    "version": "v1",
    "title": "Smoke",
    "description": "what this corpus claims to test",
    "tasks": [
        {
            "task_id": "case-01",
            "description": "one-liner",
            "payload": {"nested": {"keep": "keys"}, "order": [3, 2, 1]},
        }
    ],
}
LEGACY_BYTES = (
    '{\n  "benchmark_id": "smoke",\n  "description": "what this corpus claims to test",\n'
    '  "schema": "fastworkflow-benchmark/1",\n  "tasks": [\n    {\n      "description": "one-liner",\n'
    '      "payload": {\n        "nested": {\n          "keep": "keys"\n        },\n'
    '        "order": [\n          3,\n          2,\n          1\n        ]\n      },\n'
    '      "task_id": "case-01"\n    }\n  ],\n  "title": "Smoke",\n  "version": "v1"\n}\n'
).encode("utf-8")
LEGACY_DIGEST = "7442c183cdf68a943754af4bfa5bc21d26baee81241b4b7885c146d22ba442ac"


def _live_db(workflow) -> str:
    return state_paths.observability_db(str(workflow))


def _stored_bytes(workflow, benchmark_id="smoke", version="v1") -> bytes:
    store = obs.ReadOnlyObservabilityStore(_live_db(workflow))
    row = store._open_connection().execute(
        "SELECT manifest_bytes FROM benchmark_versions "
        "WHERE benchmark_id=? AND version=?",
        (benchmark_id, version),
    ).fetchone()
    return bytes(row[0])


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
                "payload": {
                    "nested": {"keep": "keys"},
                    "order": [3, 2, 1],
                },
            }
        ],
    }


def test_write_load_round_trip_preserves_payload_keys(tmp_path):
    workflow = tmp_path / "workflow"
    workflow.mkdir()
    spec = _sample_spec()

    written = write_version(workflow, spec)
    loaded = load_version(workflow, "smoke", "v1")

    assert written["digest_sha256"] == loaded["digest_sha256"]
    assert loaded["schema"] == SCHEMA
    assert loaded["benchmark_id"] == "smoke"
    assert loaded["version"] == "v1"
    assert loaded["tasks"][0]["payload"] == spec["tasks"][0]["payload"]


def test_round_trip_digest_matches_the_pre_db_file_digest(tmp_path):
    """The bytes and digest the file catalog produced for the same manifest."""
    workflow = tmp_path / "workflow"
    workflow.mkdir()

    written = write_version(workflow, LEGACY_SPEC)
    loaded = load_version(workflow, "smoke", "v1")

    assert _stored_bytes(workflow) == LEGACY_BYTES
    assert hashlib.sha256(LEGACY_BYTES).hexdigest() == LEGACY_DIGEST
    assert written["digest_sha256"] == LEGACY_DIGEST
    assert loaded["digest_sha256"] == LEGACY_DIGEST
    assert loaded["title"] == "Smoke"


def test_digest_stability_on_repeated_load(tmp_path):
    workflow = tmp_path / "workflow"
    workflow.mkdir()
    write_version(workflow, _sample_spec())

    first = load_version(workflow, "smoke", "v1")
    second = load_version(workflow, "smoke", "v1")

    assert first["digest_sha256"] == second["digest_sha256"]


def test_stored_digest_is_sha256_of_the_stored_bytes(tmp_path):
    workflow = tmp_path / "workflow"
    workflow.mkdir()
    write_version(workflow, _sample_spec())

    stored = _stored_bytes(workflow)
    loaded = load_version(workflow, "smoke", "v1")

    assert loaded["digest_sha256"] == hashlib.sha256(stored).hexdigest()


def test_missing_version_raises_manifest_error(tmp_path):
    workflow = tmp_path / "workflow"
    workflow.mkdir()

    assert list_benchmarks(workflow) == []
    assert list_versions(workflow, "smoke") == []
    with pytest.raises(BenchmarkManifestError, match="not found"):
        load_version(workflow, "smoke", "v1")
    assert not os.path.exists(_live_db(workflow)), "reads must not create the DB"


def test_stored_row_is_immutable(tmp_path):
    workflow = tmp_path / "workflow"
    workflow.mkdir()
    write_version(workflow, _sample_spec())
    store = obs.ObservabilityStore(_live_db(workflow))

    conn = store._open_connection()
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        conn.execute("UPDATE benchmark_versions SET title='changed'")
    conn.close()
    conn = store._open_connection()
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        conn.execute("DELETE FROM benchmark_versions")
    conn.close()
    assert load_version(workflow, "smoke", "v1")["description"] == (
        "what this corpus claims to test"
    )


def test_store_without_benchmark_marker_gets_table_on_next_write(tmp_path):
    """A store that already has `control_v1` (the owner's) gains the table."""
    workflow = tmp_path / "workflow"
    workflow.mkdir()
    db = _live_db(workflow)
    obs.ObservabilityStore(db)
    conn = sqlite3.connect(db)
    conn.executescript(
        "DROP TRIGGER benchmark_versions_no_update;"
        "DROP TRIGGER benchmark_versions_no_delete;"
        "DROP TABLE benchmark_versions;"
    )
    features = [
        f for f in json.loads(
            conn.execute("SELECT value FROM diagnostics WHERE key='schema_features'").fetchone()[0]
        ) if f != control.FEATURE_BENCHMARK_CATALOG_V1
    ]
    conn.execute(
        "UPDATE diagnostics SET value=? WHERE key='schema_features'", (json.dumps(features),)
    )
    conn.commit()
    conn.close()

    assert control.FEATURE_CONTROL_V1 in features
    assert list_benchmarks(workflow) == []
    with pytest.raises(BenchmarkManifestError, match="not found"):
        load_version(workflow, "smoke", "v1")

    write_version(workflow, _sample_spec())

    assert list_versions(workflow, "smoke") == ["v1"]
    assert load_version(workflow, "smoke", "v1")["version"] == "v1"


def test_prune_erase_and_seal_strip_keep_benchmark_versions(tmp_path):
    workflow = tmp_path / "workflow"
    workflow.mkdir()
    write_version(workflow, _sample_spec())
    store = obs.ObservabilityStore(_live_db(workflow))

    store.prune(retention_days=0, max_bytes=1)
    store.clear_conversations()
    assert list_versions(workflow, "smoke") == ["v1"]
    assert load_version(workflow, "smoke", "v1")["version"] == "v1"

    conn = sqlite3.connect(_live_db(workflow))
    control.strip(conn)
    conn.close()
    assert list_versions(workflow, "smoke") == ["v1"]


def test_duplicate_task_id_refused(tmp_path):
    workflow = tmp_path / "workflow"
    workflow.mkdir()
    spec = _sample_spec()
    spec["tasks"] = [
        {
            "task_id": "case-01",
            "description": "first",
            "payload": {},
        },
        {
            "task_id": "case-01",
            "description": "duplicate",
            "payload": {},
        },
    ]

    with pytest.raises(BenchmarkManifestError, match="duplicate task_id"):
        write_version(workflow, spec)


def test_overwrite_existing_version_refused(tmp_path):
    workflow = tmp_path / "workflow"
    workflow.mkdir()
    write_version(workflow, _sample_spec())

    changed = _sample_spec()
    changed["description"] = "changed description"
    with pytest.raises(BenchmarkAlreadyExistsError, match="immutable"):
        write_version(workflow, changed)


def test_multiple_benchmarks_in_one_workflow(tmp_path):
    workflow = tmp_path / "workflow"
    workflow.mkdir()
    write_version(workflow, _sample_spec(benchmark_id="alpha", version="v1"))
    write_version(
        workflow,
        _sample_spec(
            benchmark_id="beta",
            version="v1",
            task_id="beta-case",
        ),
    )
    write_version(
        workflow,
        _sample_spec(
            benchmark_id="alpha",
            version="v2",
            task_id="alpha-case-2",
        ),
    )

    assert list_benchmarks(workflow) == ["alpha", "beta"]
    assert list_versions(workflow, "alpha") == ["v1", "v2"]
    assert list_versions(workflow, "beta") == ["v1"]


def test_list_versions_sorts_naturally(tmp_path):
    workflow = tmp_path / "workflow"
    workflow.mkdir()
    for version in ("v10", "v2", "v1"):
        write_version(
            workflow,
            _sample_spec(version=version, task_id=f"case-{version}"),
        )

    assert list_versions(workflow, "smoke") == ["v1", "v2", "v10"]


def test_path_traversal_refused(tmp_path):
    workflow = tmp_path / "workflow"
    workflow.mkdir()

    with pytest.raises(BenchmarkManifestError, match="single path segment"):
        write_version(workflow, _sample_spec(benchmark_id="../escape"))

    with pytest.raises(BenchmarkManifestError, match="single path segment"):
        write_version(workflow, _sample_spec(version="../v1"))


def test_slash_in_benchmark_id_refused(tmp_path):
    workflow = tmp_path / "workflow"
    workflow.mkdir()

    with pytest.raises(BenchmarkManifestError, match="single path segment"):
        write_version(workflow, _sample_spec(benchmark_id="foo/bar"))

    assert list_benchmarks(workflow) == []


def test_empty_task_list_is_stored_as_a_draft(tmp_path):
    workflow = tmp_path / "workflow"
    workflow.mkdir()
    spec = _sample_spec()
    spec["tasks"] = []

    written = write_version(workflow, spec)
    assert written["tasks"] == []
    assert load_version(workflow, spec["benchmark_id"], spec["version"]) == written


def test_payload_tuple_refused(tmp_path):
    workflow = tmp_path / "workflow"
    workflow.mkdir()
    spec = _sample_spec()
    spec["tasks"][0]["payload"] = {"bindings": (3, 2, 1)}

    with pytest.raises(
        BenchmarkManifestError,
        match=r"tasks\[0\]\.payload\.bindings must be a JSON-native value",
    ):
        write_version(workflow, spec)


def test_payload_enum_refused(tmp_path):
    class Flavor(Enum):
        VANILLA = "vanilla"

    workflow = tmp_path / "workflow"
    workflow.mkdir()
    spec = _sample_spec()
    spec["tasks"][0]["payload"] = {"flavor": Flavor.VANILLA}

    with pytest.raises(
        BenchmarkManifestError,
        match=r"tasks\[0\]\.payload\.flavor must be a JSON-native value",
    ):
        write_version(workflow, spec)


def test_payload_non_string_key_refused(tmp_path):
    workflow = tmp_path / "workflow"
    workflow.mkdir()
    spec = _sample_spec()
    spec["tasks"][0]["payload"] = {1: "not-a-string-key"}

    with pytest.raises(
        BenchmarkManifestError,
        match=r"tasks\[0\]\.payload keys must be strings",
    ):
        write_version(workflow, spec)


def test_payload_non_finite_float_refused(tmp_path):
    workflow = tmp_path / "workflow"
    workflow.mkdir()
    spec = _sample_spec()
    spec["tasks"][0]["payload"] = {"score": float("nan")}

    with pytest.raises(
        BenchmarkManifestError,
        match=r"tasks\[0\]\.payload\.score must be a finite number",
    ):
        write_version(workflow, spec)


def test_payload_nested_json_native_list_accepted(tmp_path):
    workflow = tmp_path / "workflow"
    workflow.mkdir()
    spec = _sample_spec()
    assert spec["tasks"][0]["payload"]["order"] == [3, 2, 1]

    written = write_version(workflow, spec)

    assert written["tasks"][0]["payload"]["order"] == [3, 2, 1]
    assert written["tasks"][0]["payload"]["nested"] == {"keep": "keys"}

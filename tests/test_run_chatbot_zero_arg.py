"""Zero-argument browsing: `run_chatbot` with no state root and no CLI args.

Two things the owner has to be able to do from a bare `run_chatbot`: find a
sealed collection's manifest by pointing the picker at the folder that holds
the collections (`fix-zns` item 1), and, once a workspace is open, see the
benchmark catalogue the run was pinned to — read-only, with the pin actually
checked against the catalogue file rather than merely displayed (item 2).
"""

from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pytest

from fastworkflow import state_paths
from fastworkflow.observability import store as obs
from fastworkflow.benchmark.catalog import benchmarks_root, load_version, write_version
from fastworkflow.experiment.runner import ExperimentController
from fastworkflow.observability.workspace import (
    WORKSPACE_SCHEMA,
    WorkspaceManifestError,
    load_observability_workspace,
)
from fastworkflow.run_chatbot import server as run_chatbot_server
from fastworkflow.run_chatbot.server import benchmark_pin_check, browse_directories


# ----------------------------------------------------------------------
# item 1: the picker finds a collection's manifest one level down
# ----------------------------------------------------------------------


@pytest.fixture
def collections_tree(tmp_path) -> Path:
    """A folder shaped like `evaluation/collections/`."""
    root = tmp_path / "collections"
    root.mkdir()

    nested = root / "exp029-trial-3.3-lifecycle-2026-09-06" / "workspace"
    nested.mkdir(parents=True)
    (nested / "workspace.json").write_text("{}", encoding="utf-8")
    # Sibling JSON in the same collection: scores, summaries, seal records.
    (nested.parent / "summary.json").write_text("{}", encoding="utf-8")
    (nested / "seal-record.json").write_text("{}", encoding="utf-8")

    flat = root / "exp028-flat"
    flat.mkdir()
    (flat / "workspace.json").write_text("{}", encoding="utf-8")

    plain = root / "no-manifest-here"
    plain.mkdir()
    (plain / "notes.json").write_text("{}", encoding="utf-8")

    hidden = root / ".scratch"
    hidden.mkdir()
    (hidden / "workspace.json").write_text("{}", encoding="utf-8")

    # This directory's own JSON: the well-known name, a renamed manifest that
    # declares the schema, and a stray result file that is not a manifest --
    # only the first two may be offered (fix-o8s).
    (root / "workspace.json").write_text("{}", encoding="utf-8")
    (root / "renamed-manifest.json").write_text(
        json.dumps({"schema": WORKSPACE_SCHEMA, "stores": []}), encoding="utf-8"
    )
    (root / "loose.json").write_text("{}", encoding="utf-8")
    return root


def test_picker_lists_collection_manifests_one_level_down(collections_tree):
    listing = browse_directories(str(collections_tree))
    found = {m["label"]: m["path"] for m in listing["workspace_manifests"]}
    assert found == {
        "renamed-manifest.json": str(collections_tree / "renamed-manifest.json"),
        "workspace.json": str(collections_tree / "workspace.json"),
        "exp028-flat": str(collections_tree / "exp028-flat" / "workspace.json"),
        "exp029-trial-3.3-lifecycle-2026-09-06": str(
            collections_tree
            / "exp029-trial-3.3-lifecycle-2026-09-06"
            / "workspace"
            / "workspace.json"
        ),
    }
    # This directory's own JSON comes first; nested discovery is additive.
    assert [m["label"] for m in listing["workspace_manifests"][:2]] == [
        "renamed-manifest.json",
        "workspace.json",
    ]
    assert all(Path(m["path"]).is_absolute() for m in listing["workspace_manifests"])


def test_picker_never_offers_arbitrary_json_one_level_down(collections_tree):
    names = {
        Path(m["path"]).name for m in browse_directories(str(collections_tree))[
            "workspace_manifests"
        ]
    }
    assert "summary.json" not in names
    assert "seal-record.json" not in names
    assert "notes.json" not in names


# ----------------------------------------------------------------------
# fix-o8s: this directory's own JSON is filtered too
# ----------------------------------------------------------------------
#
# Before this, every `*.json` beside the browsed folder was offered as a
# manifest, so from a project root the picker filled with score dumps and
# trajectory files, each of which opens into an error. The rule now: the
# well-known name `workspace.json`, or a head that declares the workspace
# schema. Everything else is omitted -- not offered, not labelled.


def _manifest_labels(directory) -> set[str]:
    return {
        m["label"] for m in browse_directories(str(directory))["workspace_manifests"]
    }


def test_picker_omits_stray_json_in_the_browsed_directory(collections_tree):
    assert "loose.json" not in _manifest_labels(collections_tree)


def test_picker_offers_workspace_json_on_its_name_alone(tmp_path):
    """Offered without being read: a `workspace.json` that fails validation
    must still be reachable, so the developer sees *why* it is refused rather
    than a picker that silently has nothing."""
    (tmp_path / "workspace.json").write_text("not json at all", encoding="utf-8")
    assert _manifest_labels(tmp_path) == {"workspace.json"}


def test_picker_offers_a_renamed_manifest(tmp_path):
    (tmp_path / "exp029.json").write_text(
        json.dumps(
            {
                "schema": WORKSPACE_SCHEMA,
                "workspace_id": "w1",
                "label": "renamed",
                "stores": [],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "legacy.json").write_text(
        json.dumps({"schema_version": WORKSPACE_SCHEMA, "stores": []}),
        encoding="utf-8",
    )
    assert _manifest_labels(tmp_path) == {"exp029.json", "legacy.json"}


def test_picker_omits_json_declaring_another_schema(tmp_path):
    (tmp_path / "other.json").write_text(
        json.dumps({"schema": "something-else/2", "stores": []}), encoding="utf-8"
    )
    (tmp_path / "nested.json").write_text(
        # The key exists, but nested — not a top-level declaration.
        json.dumps({"payload": {"schema": WORKSPACE_SCHEMA}}), encoding="utf-8"
    )
    assert _manifest_labels(tmp_path) == set()


def test_picker_reads_at_most_the_probe_window(tmp_path):
    """A huge result file is not loaded to decide this. The schema key placed
    past the probe window reads as "not a manifest" — absence of evidence is
    "no" — and the same file with the key first is offered."""
    padding = {"junk": "x" * 200_000}
    (tmp_path / "huge-late.json").write_text(
        json.dumps({**padding, "schema": WORKSPACE_SCHEMA}), encoding="utf-8"
    )
    (tmp_path / "huge-early.json").write_text(
        json.dumps({"schema": WORKSPACE_SCHEMA, **padding}), encoding="utf-8"
    )
    assert _manifest_labels(tmp_path) == {"huge-early.json"}


def test_picker_skips_an_unreadable_json_without_failing(tmp_path):
    (tmp_path / "workspace.json").write_text("{}", encoding="utf-8")
    denied = tmp_path / "denied.json"
    denied.write_text(
        json.dumps({"schema": WORKSPACE_SCHEMA, "stores": []}), encoding="utf-8"
    )
    os.chmod(denied, 0o000)
    try:
        if os.access(denied, os.R_OK):  # running as root: the probe would succeed
            pytest.skip("cannot make a file unreadable for this user")
        listing = browse_directories(str(tmp_path))
        assert "error" not in listing
        assert {m["label"] for m in listing["workspace_manifests"]} == {
            "workspace.json"
        }
    finally:
        os.chmod(denied, 0o644)


def test_picker_skips_undecodable_bytes_without_failing(tmp_path):
    (tmp_path / "binary.json").write_bytes(b"\xff\xfe\x00\x01" * 64)
    listing = browse_directories(str(tmp_path))
    assert "error" not in listing
    assert listing["workspace_manifests"] == []


def test_browse_route_reflects_the_manifest_filter(collections_tree):
    server = run_chatbot_server.ChatbotServer(
        db_path="",
        workflow_path="",
        port=0,
        spawn_options={"no_server": True},
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, data = _request(
            server,
            "/api/browse?dir=" + urllib.parse.quote(str(collections_tree)),
        )
        assert status == 200
        labels = {m["label"] for m in data["workspace_manifests"]}
        assert "loose.json" not in labels
        assert {"workspace.json", "renamed-manifest.json"} <= labels
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_picker_skips_dot_directories(collections_tree):
    paths = [m["path"] for m in browse_directories(str(collections_tree))[
        "workspace_manifests"
    ]]
    assert not any(".scratch" in path for path in paths)
    assert not any(entry["name"] == ".scratch" for entry in
                   browse_directories(str(collections_tree))["entries"])


def test_directory_entries_are_unchanged_by_manifest_discovery(collections_tree):
    listing = browse_directories(str(collections_tree))
    assert [entry["name"] for entry in listing["entries"]] == [
        "exp028-flat",
        "exp029-trial-3.3-lifecycle-2026-09-06",
        "no-manifest-here",
    ]
    assert all(entry["is_workflow"] is False for entry in listing["entries"])


def test_manifest_list_keeps_the_300_entry_cap(tmp_path):
    root = tmp_path / "many"
    root.mkdir()
    for index in range(320):
        folder = root / f"collection-{index:04d}"
        folder.mkdir()
        (folder / "workspace.json").write_text("{}", encoding="utf-8")
    listing = browse_directories(str(root))
    assert len(listing["workspace_manifests"]) == 300
    assert len(listing["entries"]) == 300


def test_browse_over_http_carries_the_labels(tmp_path, collections_tree):
    server = run_chatbot_server.ChatbotServer(
        db_path="",
        workflow_path="",
        port=0,
        spawn_options={"no_server": True},
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, data = _request(
            server,
            "/api/browse?dir=" + urllib.parse.quote(str(collections_tree)),
        )
        assert status == 200
        assert "exp029-trial-3.3-lifecycle-2026-09-06" in {
            m["label"] for m in data["workspace_manifests"]
        }
    finally:
        server.shutdown()
        thread.join(timeout=5)


# ----------------------------------------------------------------------
# item 2: the catalogue, read-only, from workspace mode
# ----------------------------------------------------------------------


def _request(server, path, method="GET", body=None):
    url = f"http://127.0.0.1:{server.port}{path}"
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, method=method, data=data)
    req.add_header("Authorization", f"Bearer {server.token}")
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


def _spec(version: str = "v1") -> dict:
    return {
        "benchmark_id": "g2e-tuning",
        "version": version,
        "description": "five tuning cases",
        "tasks": [
            {"task_id": "case-01", "description": "one", "payload": {"a": 1}},
        ],
    }


@pytest.fixture
def sealed_collection(tmp_path, monkeypatch):
    """A sealed experiment plus the workflow folder whose catalogue it pinned.

    Built the way `tests/test_experiment_sealing.py` builds one: an installed
    store, one declared task, one attempt, one evidence segment, then
    `seal_workspace_evidence` — and the manifest is written from what the seal
    returns, which is where `workflow_folderpath` now comes from.
    """
    monkeypatch.setenv(obs.CAPTURE_PROFILE_VAR, "evidence")
    workflow = tmp_path / "ido_workflow"
    (workflow / "_commands").mkdir(parents=True)
    written = write_version(workflow, _spec())

    store = obs.ObservabilityStore(state_paths.observability_db(str(workflow)))
    identity = store.store_identity()
    controller = ExperimentController(
        str(workflow),
        identity,
        migrate=False,
        external=True,
    )
    controller.create_experiment(
        "exp-1",
        "sealed collection",
        declared_tasks=1,
        declared_attempts=1,
        declarations=[("task-1", 1, "native-1")],
        required_evidence_segments=1,
        benchmark_id="g2e-tuning",
        benchmark_version="v1",
        benchmark_digest_sha256=written["digest_sha256"],
    )
    controller.start_attempt("exp-1", "task-1", 1, "channel-1", source_key="native-1")
    controller.finish_attempt(
        "exp-1", "task-1", 1, outcome="pass", outcome_source="external-grader"
    )
    controller.record_evidence_segment(
        "exp-1",
        1,
        "run-1",
        {
            "valid": True,
            "started_at": "2026-09-06T00:00:00+00:00",
            "completed_at": "2026-09-06T00:01:00+00:00",
            "problems": [],
        },
    )
    assert controller.complete_experiment("exp-1") == "capture_complete"

    workspace_dir = tmp_path / "collection" / "workspace"
    workspace_dir.mkdir(parents=True)
    archive = controller.seal_workspace_evidence(
        "exp-1", str(workspace_dir / "observability-sealed.sqlite3")
    )
    return workflow, workspace_dir, archive, written


def _write_manifest(workspace_dir: Path, archive: dict, **extra) -> Path:
    manifest = {
        "schema": WORKSPACE_SCHEMA,
        "workspace_id": "sealed-collection",
        "label": "sealed collection",
        "stores": [
            {
                "store_id": "sealed",
                "label": "sealed",
                "path": Path(archive["path"]).name,
                "mode": "sealed",
                "sha256": archive["sha256"],
                "store_identity": archive["store_identity"],
            }
        ],
        "experiments": [
            {
                "experiment_id": "exp-1",
                "segments": [{"store_id": "sealed", "local_experiment_id": "exp-1"}],
            }
        ],
        "projected_attempts": [],
    }
    manifest.update(extra)
    path = workspace_dir / "workspace.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


@pytest.fixture
def workspace_with_catalogue(sealed_collection):
    workflow, workspace_dir, archive, written = sealed_collection
    manifest = _write_manifest(
        workspace_dir, archive, workflow_folderpath=archive["workflow_folderpath"]
    )
    server = run_chatbot_server.ChatbotServer(
        port=0, workspace_manifest_path=str(manifest)
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, workflow, written, manifest
    server.shutdown()
    thread.join(timeout=5)


@pytest.fixture
def workspace_without_folder(sealed_collection):
    _workflow, workspace_dir, archive, _written = sealed_collection
    manifest = _write_manifest(workspace_dir, archive)
    server = run_chatbot_server.ChatbotServer(
        port=0, workspace_manifest_path=str(manifest)
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    thread.join(timeout=5)


class TestSealRecordsTheWorkflowFolder:
    def test_seal_returns_the_absolute_workflow_folder(self, sealed_collection):
        workflow, _workspace_dir, archive, _written = sealed_collection
        assert archive["workflow_folderpath"] == os.path.abspath(str(workflow))

    def test_manifest_keeps_a_named_folder_that_no_longer_exists(
        self, sealed_collection, tmp_path
    ):
        _workflow, workspace_dir, archive, _written = sealed_collection
        missing = tmp_path / "moved-away"
        manifest = _write_manifest(
            workspace_dir, archive, workflow_folderpath=str(missing)
        )
        workspace = load_observability_workspace(manifest)
        assert workspace.workflow_folderpath == str(missing)
        assert workspace.benchmark_catalogue_state == "unavailable"
        assert workspace.summary()["benchmark_catalogue"] == "unavailable"

    def test_relative_workflow_folder_is_refused(self, sealed_collection):
        _workflow, workspace_dir, archive, _written = sealed_collection
        manifest = _write_manifest(
            workspace_dir, archive, workflow_folderpath="../ido_workflow"
        )
        with pytest.raises(WorkspaceManifestError) as excinfo:
            load_observability_workspace(manifest)
        assert "absolute" in str(excinfo.value)

    def test_no_field_is_not_an_error(self, workspace_without_folder):
        status, data = _request(workspace_without_folder, "/api/workspace")
        assert status == 200
        assert data["workspace"]["workflow_folderpath"] is None
        assert data["workspace"]["benchmark_catalogue"] == "not_declared"


class TestWorkspaceModeCatalogueReads:
    def test_browse_and_version_are_served(self, workspace_with_catalogue):
        server, _workflow, written, _manifest = workspace_with_catalogue
        status, data = _request(server, "/api/benchmarks")
        assert status == 200
        assert data["benchmarks"] == [
            {"benchmark_id": "g2e-tuning", "versions": ["v1"]}
        ]

        status, data = _request(server, "/api/benchmarks/g2e-tuning/versions/v1")
        assert status == 200
        assert data["version"]["digest_sha256"] == written["digest_sha256"]
        assert data["version"]["tasks"][0]["task_id"] == "case-01"

    def test_analysis_read_is_served(self, workspace_with_catalogue):
        server, _workflow, _written, _manifest = workspace_with_catalogue
        status, data = _request(server, "/api/benchmarks/g2e-tuning/analysis")
        assert status == 200
        assert data["benchmark_id"] == "g2e-tuning"

    def test_409_is_preserved_without_the_field(self, workspace_without_folder):
        status, data = _request(workspace_without_folder, "/api/benchmarks")
        assert status == 409
        assert "live workflow mode" in data["error"]

    def test_409_when_the_named_folder_is_gone(self, sealed_collection, tmp_path):
        _workflow, workspace_dir, archive, _written = sealed_collection
        manifest = _write_manifest(
            workspace_dir, archive, workflow_folderpath=str(tmp_path / "gone")
        )
        server = run_chatbot_server.ChatbotServer(
            port=0, workspace_manifest_path=str(manifest)
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            status, data = _request(server, "/api/benchmarks")
            assert status == 409
            assert str(tmp_path / "gone") in data["error"]
        finally:
            server.shutdown()
            thread.join(timeout=5)


class TestWorkspaceModeCatalogueWritesRefused:
    @pytest.mark.parametrize(
        "path,method,body",
        [
            ("/api/benchmarks/g2e-tuning/versions", "POST", {"version": "v2"}),
            ("/api/benchmarks", "POST", {"benchmark_id": "new", "version": "v1"}),
            ("/api/benchmarks/g2e-tuning/analysis", "PUT", {"note": "x"}),
        ],
    )
    def test_every_write_route_is_refused(
        self, workspace_with_catalogue, path, method, body
    ):
        server, workflow, _written, _manifest = workspace_with_catalogue
        before = sorted(p.name for p in (benchmarks_root(workflow)).rglob("*.json"))
        status, data = _request(server, path, method=method, body=body)
        assert status == 403
        assert "read-only" in data["error"]
        after = sorted(p.name for p in (benchmarks_root(workflow)).rglob("*.json"))
        assert after == before

    def test_the_pinned_version_file_is_untouched(self, workspace_with_catalogue):
        server, workflow, written, _manifest = workspace_with_catalogue
        _request(
            server,
            "/api/benchmarks/g2e-tuning/versions",
            method="POST",
            body=_spec(version="v2"),
        )
        assert load_version(workflow, "g2e-tuning", "v1") == written


class TestPinCheck:
    def test_match_is_reported(self, workspace_with_catalogue):
        server, _workflow, written, _manifest = workspace_with_catalogue
        status, data = _request(
            server, "/api/workspace/experiment/exp-1/segments"
        )
        assert status == 200
        pin = data["segments"][0]["benchmark_pin"]
        assert pin["status"] == "match"
        assert pin["pinned_digest"] == written["digest_sha256"]
        assert pin["catalogue_digest"] == written["digest_sha256"]

    def test_mismatch_is_quoted_verbatim(self, sealed_collection):
        workflow, workspace_dir, archive, written = sealed_collection
        # The catalogue file moved on after the run was pinned. Rewriting the
        # immutable version file is exactly what the pin exists to catch.
        version_file = benchmarks_root(workflow) / "g2e-tuning" / "v1.json"
        payload = json.loads(version_file.read_text(encoding="utf-8"))
        payload["tasks"][0]["payload"] = {"a": 2}
        version_file.write_text(json.dumps(payload), encoding="utf-8")

        manifest = _write_manifest(
            workspace_dir, archive, workflow_folderpath=str(workflow)
        )
        workspace = load_observability_workspace(manifest)
        with workspace.registry.open("sealed") as store:
            detail = store.get_experiment("exp-1")
        check = benchmark_pin_check(workspace.workflow_folderpath, detail)
        assert check["status"] == "mismatch"
        assert check["pinned_digest"] == written["digest_sha256"]
        assert check["catalogue_digest"] != written["digest_sha256"]
        assert len(check["catalogue_digest"]) == 64

    def test_catalogue_unavailable_is_shown_as_such(self, sealed_collection, tmp_path):
        _workflow, workspace_dir, archive, _written = sealed_collection
        manifest = _write_manifest(workspace_dir, archive)
        workspace = load_observability_workspace(manifest)
        with workspace.registry.open("sealed") as store:
            detail = store.get_experiment("exp-1")
        check = benchmark_pin_check(workspace.workflow_folderpath, detail)
        assert check["status"] == "catalogue_unavailable"
        assert check["catalogue_digest"] is None
        assert "no workflow folder" in check["detail"]

        gone = str(tmp_path / "gone")
        check = benchmark_pin_check(gone, detail)
        assert check["status"] == "catalogue_unavailable"
        assert gone in check["detail"]

    def test_an_unpinned_experiment_has_no_check(self):
        assert benchmark_pin_check("/nowhere", {"benchmark_id": None}) is None

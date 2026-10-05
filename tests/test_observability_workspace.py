"""Read-only multi-store observability workspace contracts."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import sqlite3
import threading
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

from fastworkflow.observability import store as obs
from fastworkflow.observability.workspace import (
    WORKSPACE_SCHEMA,
    UnknownWorkspaceStore,
    WorkspaceIntegrityError,
    WorkspaceManifestError,
    load_observability_workspace,
)
from fastworkflow.run_chatbot import server as run_chatbot_server


def _turn_row(turn_key: str, experiment_id: str, task_id: str, attempt: int) -> dict:
    return {
        "turn_key": turn_key,
        "channel_id": f"channel-{task_id}",
        "conversation_id": None,
        "ordinal": None,
        "user_message": f"run {task_id}",
        "refined_user_message": None,
        "entry_workflow_name": "workspace-test",
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


def _seed_archive(
    root: Path,
    name: str,
    *,
    experiment_id: str,
    task_id: str,
    turn_key: str,
    old_features: bool = False,
) -> dict:
    source = root / f"{name}-live.sqlite3"
    store = obs.ObservabilityStore(str(source))
    identity = store.store_identity()
    assert identity
    store.create_experiment(
        experiment_id,
        f"label-{experiment_id}",
        declared_tasks=1,
        declared_attempts=1,
    )
    store.start_attempt(experiment_id, task_id, 1, f"channel-{task_id}")
    store.finish_attempt(
        experiment_id,
        task_id,
        1,
        outcome="pass",
        outcome_source="test",
    )
    with store._connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        assert store.upsert_turn_row(
            conn,
            _turn_row(turn_key, experiment_id, task_id, 1),
            [],
            store._store_redactor(),
        )
        if old_features:
            conn.execute("DELETE FROM diagnostics WHERE key='schema_features'")
        conn.commit()
    archive_path = root / f"{name}.sqlite3"
    archived = obs.ObservabilityStore(str(source), migrate=False).archive_to(
        str(archive_path)
    )
    assert archived["store_identity"] == identity
    return archived


def _manifest(
    root: Path,
    stores: list[dict],
    *,
    experiments: list[dict] | None = None,
    projected_attempts: list[dict] | None = None,
) -> Path:
    path = root / "workspace.json"
    path.write_text(
        json.dumps(
            {
                "schema": WORKSPACE_SCHEMA,
                "workspace_id": "workspace-1",
                "label": "Historical runs",
                "stores": stores,
                "experiments": experiments or [],
                "projected_attempts": projected_attempts or [],
            }
        ),
        encoding="utf-8",
    )
    return path


def _store_decl(archive: dict, store_id: str) -> dict:
    return {
        "store_id": store_id,
        "label": store_id,
        "path": Path(archive["path"]).name,
        "mode": "sealed",
        "sha256": archive["sha256"],
        "store_identity": archive["store_identity"],
    }


def test_valid_workspace_opens_old_and_new_archives_read_only(tmp_path):
    old = _seed_archive(
        tmp_path,
        "old",
        experiment_id="old-local",
        task_id="task-old",
        turn_key="shared-turn",
        old_features=True,
    )
    new = _seed_archive(
        tmp_path,
        "new",
        experiment_id="new-local",
        task_id="task-new",
        turn_key="shared-turn",
    )
    manifest = _manifest(
        tmp_path,
        [_store_decl(old, "old"), _store_decl(new, "new")],
        experiments=[
            {
                "experiment_id": "logical",
                "label": "logical",
                "segments": [
                    {
                        "segment_id": "before",
                        "store_id": "old",
                        "local_experiment_id": "old-local",
                    },
                    {
                        "segment_id": "after",
                        "store_id": "new",
                        "local_experiment_id": "new-local",
                    },
                ],
            }
        ],
    )

    workspace = load_observability_workspace(manifest)
    before = {
        path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for path in (old["path"], new["path"])
    }

    assert workspace.turn("old", "shared-turn")["store_id"] == "old"
    assert workspace.turn("new", "shared-turn")["store_id"] == "new"
    assert {store["integrity"] for store in workspace.stores()} == {"verified"}
    assert before == {
        path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for path in (old["path"], new["path"])
    }
    assert not any(
        Path(f"{path}{suffix}").exists()
        for path in (old["path"], new["path"])
        for suffix in ("-wal", "-shm")
    )


@pytest.mark.parametrize("bad_path", ["../outside.sqlite3", "/tmp/outside.sqlite3"])
def test_path_traversal_and_absolute_paths_are_refused(tmp_path, bad_path):
    manifest = _manifest(
        tmp_path,
        [
            {
                "store_id": "bad",
                "path": bad_path,
                "mode": "sealed",
                "sha256": "0" * 64,
            }
        ],
    )

    with pytest.raises(WorkspaceManifestError, match="relative"):
        load_observability_workspace(manifest)


def test_symlink_resolution_must_stay_under_workspace_root(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside.sqlite3"
    sqlite3.connect(outside).close()
    (tmp_path / "escaped.sqlite3").symlink_to(outside)
    manifest = _manifest(
        tmp_path,
        [
            {
                "store_id": "escaped",
                "path": "escaped.sqlite3",
                "mode": "sealed",
                "sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
            }
        ],
    )

    with pytest.raises(WorkspaceManifestError, match="outside"):
        load_observability_workspace(manifest)


def test_sealed_store_refuses_wal_or_shm_sidecars(tmp_path):
    archive = _seed_archive(
        tmp_path,
        "sealed",
        experiment_id="local",
        task_id="task",
        turn_key="turn",
    )
    manifest = _manifest(tmp_path, [_store_decl(archive, "sealed")])
    Path(f"{archive['path']}-wal").write_bytes(b"active writer evidence")

    with pytest.raises(WorkspaceIntegrityError, match="sidecar"):
        load_observability_workspace(manifest)


def test_sealed_digest_mismatch_is_integrity_failure(tmp_path):
    archive = _seed_archive(
        tmp_path,
        "sealed",
        experiment_id="local",
        task_id="task",
        turn_key="turn",
    )
    declaration = _store_decl(archive, "sealed")
    declaration["sha256"] = "0" * 64

    with pytest.raises(WorkspaceIntegrityError, match="sha256 mismatch"):
        load_observability_workspace(_manifest(tmp_path, [declaration]))


def test_a_live_store_is_refused_and_left_as_it_was(tmp_path):
    """A workspace reads sealed archives only; a live DB is the workflow's own."""
    live_path = tmp_path / "live.sqlite3"
    store = obs.ObservabilityStore(str(live_path))
    declaration = {
        "store_id": "live",
        "path": live_path.name,
        "mode": "live",
        "sha256": "0" * 64,
        "store_identity": store.store_identity(),
    }
    before = live_path.read_bytes()

    with pytest.raises(WorkspaceManifestError, match="sealed archives only"):
        load_observability_workspace(_manifest(tmp_path, [declaration]))

    assert live_path.read_bytes() == before


def test_unknown_store_and_unscoped_turn_are_refused(tmp_path):
    archive = _seed_archive(
        tmp_path,
        "sealed",
        experiment_id="local",
        task_id="task",
        turn_key="turn",
    )
    workspace = load_observability_workspace(
        _manifest(tmp_path, [_store_decl(archive, "known")])
    )

    with pytest.raises(UnknownWorkspaceStore):
        workspace.turn("missing", "turn")
    with pytest.raises(UnknownWorkspaceStore, match="required"):
        workspace.turn("", "turn")


def test_bulk_traces_answer_exactly_what_the_per_turn_reader_answers(tmp_path):
    """`traces` is the archived twin of `spans_for_turns` (fix-tk5).

    Attempt and projected-attempt rows stamp every turn ref they carry; going
    ref by ref through `trace` opened the store and queried once per ref. The
    bulk reader must answer the same scoped, attribute-decoded rows, in the
    same order, and stay just as scoped to one named store.
    """
    from fastworkflow import tracing

    source = tmp_path / "bulk-live.sqlite3"
    store = obs.ObservabilityStore(str(source))
    store.create_experiment("local", "bulk", declared_tasks=1, declared_attempts=1)
    keys = [f"turn-{index:02d}" for index in range(12)]
    spans = [
        tracing.Span(
            span_id=f"{key}-{seq}", trace_id=key, name="fw.llm.call", kind="client",
            channel_id="channel-task", start_ns=1_000 + index * 10 + seq, status="ok",
            attributes={"call_kwargs": json.dumps({"max_tokens": 8})},
        )
        for index, key in enumerate(keys)
        for seq in range(3)
    ]
    redactor = store._store_redactor()
    with store._connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        for key in keys:
            assert store.upsert_turn_row(
                conn, _turn_row(key, "local", "task", 1), [], redactor
            )
        store.upsert_span_rows(conn, spans, redactor)
        conn.commit()
    archived = obs.ObservabilityStore(str(source), migrate=False).archive_to(
        str(tmp_path / "sealed.sqlite3")
    )
    workspace = load_observability_workspace(
        _manifest(tmp_path, [_store_decl(archived, "sealed")])
    )

    asked = keys + ["never-recorded"]
    bulk = workspace.traces("sealed", asked)
    assert bulk == {key: workspace.trace("sealed", key) for key in asked}
    # Not vacuous, and the scope/decoding `trace` applies is applied here too.
    assert all(len(bulk[key]) == 3 for key in keys)
    assert bulk["never-recorded"] == []
    first = bulk[keys[0]][0]
    assert first["store_id"] == "sealed"
    assert first["logical_turn_key"] == keys[0]
    assert first["attributes"]["call_kwargs"] == json.dumps({"max_tokens": 8})

    assert workspace.traces("sealed", []) == {}
    with pytest.raises(UnknownWorkspaceStore, match="required"):
        workspace.traces("", keys)
    with pytest.raises(UnknownWorkspaceStore):
        workspace.traces("missing", keys)


def test_projected_attempt_resolution_across_two_stores(tmp_path):
    first = _seed_archive(
        tmp_path,
        "first",
        experiment_id="first-local",
        task_id="task-a",
        turn_key="turn-a",
    )
    second = _seed_archive(
        tmp_path,
        "second",
        experiment_id="second-local",
        task_id="task-b",
        turn_key="turn-b",
    )
    manifest = _manifest(
        tmp_path,
        [_store_decl(first, "first"), _store_decl(second, "second")],
        projected_attempts=[
            {
                "logical_attempt": {
                    "experiment_id": "historical",
                    "task_id": "joined",
                    "attempt": 1,
                },
                "attempt_refs": [
                    {
                        "store_id": "first",
                        "local_experiment_id": "first-local",
                        "task_id": "task-a",
                        "attempt": 1,
                        "turn_ref": {
                            "store_id": "first",
                            "logical_turn_key": "turn-a",
                        },
                    },
                    {
                        "store_id": "second",
                        "local_experiment_id": "second-local",
                        "task_id": "task-b",
                        "attempt": 1,
                        "turn_ref": {
                            "store_id": "second",
                            "logical_turn_key": "turn-b",
                        },
                    },
                ],
            }
        ],
    )

    projected = load_observability_workspace(manifest).projected_attempts(
        experiment_id="historical", task_id="joined", attempt=1
    )

    assert len(projected) == 1
    assert {
        source["resolved_turn"]["store_id"]
        for source in projected[0]["resolved_sources"]
    } == {"first", "second"}
    assert all(
        source["resolved_attempt"]["outcome"] == "pass"
        for source in projected[0]["resolved_sources"]
    )


def _get(server, path):
    request = urllib.request.Request(
        f"http://127.0.0.1:{server.port}{path}",
        headers={"Authorization": f"Bearer {server.token}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _write(server, method, path, body):
    request = urllib.request.Request(
        f"http://127.0.0.1:{server.port}{path}",
        method=method,
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {server.token}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_server_workspace_routes_are_scoped_and_refuse_legacy_turn_search(
    tmp_path,
):
    archive = _seed_archive(
        tmp_path,
        "sealed",
        experiment_id="local",
        task_id="task",
        turn_key="turn",
    )
    manifest = _manifest(
        tmp_path,
        [_store_decl(archive, "sealed")],
        experiments=[
            {
                "experiment_id": "logical",
                "segments": [
                    {
                        "store_id": "sealed",
                        "local_experiment_id": "local",
                    }
                ],
            }
        ],
    )
    server = run_chatbot_server.ChatbotServer(
        archive["path"],
        port=0,
        workspace_manifest_path=str(manifest),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        assert _get(server, "/api/workspace")[1]["workspace"]["read_only"] is True
        assert len(_get(server, "/api/workspace/stores")[1]["stores"]) == 1
        assert (
            len(
                _get(server, "/api/workspace/experiment/logical/segments")[1][
                    "segments"
                ]
            )
            == 1
        )
        assert (
            len(_get(server, "/api/workspace/experiment/logical/tasks")[1]["tasks"])
            == 1
        )
        assert (
            len(
                _get(server, "/api/workspace/experiment/logical/attempts")[1][
                    "attempts"
                ]
            )
            == 1
        )
        assert (
            _get(server, "/api/workspace/turn/sealed/turn")[1]["turn"]["store_id"]
            == "sealed"
        )
        assert _get(server, "/api/workspace/turn/turn")[0] == 400
        assert _get(server, "/api/workspace/turn/missing/turn")[0] == 404
        # Workspace mode never falls back to an unscoped one-store search.
        assert _get(server, "/api/turn/turn")[0] == 400
        assert _get(server, "/api/experiments")[0] == 400
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_manifest_load_does_not_write_digest_into_archive(tmp_path):
    archive = _seed_archive(
        tmp_path,
        "sealed",
        experiment_id="local",
        task_id="task",
        turn_key="turn",
    )
    before = Path(archive["path"]).read_bytes()

    load_observability_workspace(_manifest(tmp_path, [_store_decl(archive, "sealed")]))

    assert Path(archive["path"]).read_bytes() == before
    assert os.stat(archive["path"]).st_mode & 0o222 == 0


def test_workspace_server_opens_same_turn_key_only_in_named_store(tmp_path):
    first = _seed_archive(
        tmp_path,
        "first",
        experiment_id="first-local",
        task_id="task-first",
        turn_key="shared",
    )
    second = _seed_archive(
        tmp_path,
        "second",
        experiment_id="second-local",
        task_id="task-second",
        turn_key="shared",
    )
    manifest = _manifest(
        tmp_path,
        [_store_decl(first, "first"), _store_decl(second, "second")],
    )
    server = run_chatbot_server.ChatbotServer(
        port=0, workspace_manifest_path=str(manifest)
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        first_turn = _get(server, "/api/workspace/turn/first/shared")[1]["turn"]
        second_turn = _get(server, "/api/workspace/turn/second/shared")[1]["turn"]
        assert first_turn["task_id"] == "task-first"
        assert second_turn["task_id"] == "task-second"
        assert _get(server, "/api/turn/shared")[0] == 400
        assert _get(server, "/api/turns")[0] == 400
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_workspace_browsing_keeps_sealed_archives_byte_identical(tmp_path):
    archive = _seed_archive(
        tmp_path,
        "sealed",
        experiment_id="local",
        task_id="task",
        turn_key="turn",
    )
    manifest = _manifest(
        tmp_path,
        [_store_decl(archive, "sealed")],
        experiments=[
            {
                "experiment_id": "logical",
                "segments": [
                    {
                        "store_id": "sealed",
                        "local_experiment_id": "local",
                    }
                ],
            }
        ],
    )
    before = Path(archive["path"]).read_bytes()
    server = run_chatbot_server.ChatbotServer(
        port=0, workspace_manifest_path=str(manifest)
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        for path in (
            "/api/workspace",
            "/api/workspace/stores",
            "/api/workspace/experiments",
            "/api/workspace/experiment/logical/attempts",
            "/api/workspace/turn/sealed/turn",
            "/api/workspace/trace/sealed/turn",
        ):
            assert _get(server, path)[0] == 200
    finally:
        server.shutdown()
        thread.join(timeout=5)
    assert Path(archive["path"]).read_bytes() == before
    assert not Path(f"{archive['path']}-wal").exists()
    assert not Path(f"{archive['path']}-shm").exists()


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("POST", "/api/select_workflow", {"path": "/tmp"}),
        ("POST", "/api/configure_env", {"create_from_templates": True}),
        ("POST", "/api/train", {"path": "/tmp"}),
        (
            "POST",
            "/api/clear_conversations",
            {"confirm": "clear all conversations"},
        ),
        ("PATCH", "/api/experiment/local", {"notes": "must not write"}),
    ],
)
def test_workspace_mode_rejects_live_and_destructive_actions(
    tmp_path, method, path, body
):
    archive = _seed_archive(
        tmp_path,
        "sealed",
        experiment_id="local",
        task_id="task",
        turn_key="turn",
    )
    manifest = _manifest(tmp_path, [_store_decl(archive, "sealed")])
    before = Path(archive["path"]).read_bytes()
    server = run_chatbot_server.ChatbotServer(
        port=0, workspace_manifest_path=str(manifest)
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, response = _write(server, method, path, body)
        assert status == 403
        assert "read-only" in response["error"]
    finally:
        server.shutdown()
        thread.join(timeout=5)
    assert Path(archive["path"]).read_bytes() == before


def test_token_gated_structured_trace_link_redirects_to_scoped_spa_hash(tmp_path):
    archive = _seed_archive(
        tmp_path,
        "sealed",
        experiment_id="local",
        task_id="task",
        turn_key="turn",
    )
    manifest = _manifest(tmp_path, [_store_decl(archive, "sealed")])
    server = run_chatbot_server.ChatbotServer(
        port=0, workspace_manifest_path=str(manifest)
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    opener = urllib.request.build_opener(NoRedirect)
    try:
        unscoped = urllib.request.Request(
            f"http://127.0.0.1:{server.port}/trace/turn",
            headers={"Authorization": f"Bearer {server.token}"},
        )
        with pytest.raises(urllib.error.HTTPError) as unscoped_error:
            opener.open(unscoped, timeout=10)
        assert unscoped_error.value.code == 400

        scoped = urllib.request.Request(
            f"http://127.0.0.1:{server.port}/trace"
            "?store_id=sealed&logical_turn_key=turn",
            headers={"Authorization": f"Bearer {server.token}"},
        )
        with pytest.raises(urllib.error.HTTPError) as redirect:
            opener.open(scoped, timeout=10)
        assert redirect.value.code == 303
        location = redirect.value.headers["Location"]
        assert location.endswith("#store=sealed&turn=turn")

        missing_token = urllib.request.Request(
            f"http://127.0.0.1:{server.port}/trace"
            "?store_id=sealed&logical_turn_key=turn"
        )
        with pytest.raises(urllib.error.HTTPError) as unauthorized:
            opener.open(missing_token, timeout=10)
        assert unauthorized.value.code == 401
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_spa_pins_store_aware_workspace_navigation():
    page = run_chatbot_server.load_index_html()
    assert b'location.hash = "store="' in page
    assert b'"store=" + encodeURIComponent(storeId)' in page
    assert b"/api/workspace/turn/" in page
    assert b"/api/workspace/trace/" in page
    assert b"Unscoped turn links are refused in workspace mode" in page
    assert b"native capture" in page
    assert b"projected history" in page
    assert b"pending evaluation" in page
    assert b'document.getElementById("modeTest").style.display = "none"' in page
    assert b'document.getElementById("clearConvsBtn").style.display = "none"' in page
    assert b'document.getElementById("advPanel").style.display = "none"' in page


def test_spa_review_pane_reuses_scoped_trace_viewer_and_capability():
    page = run_chatbot_server.load_index_html()

    assert b'id="reviewPane"' in page
    assert b"function loadReviewAssignment()" in page
    assert b"function navigateReviewRow(offset)" in page
    assert b'"X-Review-Capability": review.capability' in page
    assert b"/api/review/assignments/" in page
    assert b"/progress" in page
    assert b"/answers" in page
    assert b'encodeURIComponent(row.id)' in page
    assert b'reviewApi(reviewBase + "/turn", "GET")' in page
    assert b'reviewApi(reviewBase + "/trace", "GET")' in page
    assert b"selectWorkspaceTurn(ref.store_id, ref.logical_turn_key)" in page
    assert b"renderDetail(turn, results[1].spans || [])" in page
    assert b"pane.dataset.blinded" in page


def test_token_gated_browser_selection_loads_workspace(tmp_path):
    archive = _seed_archive(
        tmp_path,
        "sealed",
        experiment_id="local",
        task_id="task",
        turn_key="turn",
    )
    manifest = _manifest(tmp_path, [_store_decl(archive, "sealed")])
    server = run_chatbot_server.ChatbotServer(port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, response = _write(
            server, "POST", "/api/select_workspace", {"path": str(manifest)}
        )
        assert status == 200
        assert response["session"]["workspace_mode"] is True
        assert response["session"]["read_only"] is True
        assert response["session"]["workspace"]["workspace_id"] == "workspace-1"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_run_chatbot_main_forwards_explicit_workspace_manifest(
    tmp_path, monkeypatch, capsys
):
    archive = _seed_archive(
        tmp_path,
        "sealed",
        experiment_id="local",
        task_id="task",
        turn_key="turn",
    )
    manifest = _manifest(tmp_path, [_store_decl(archive, "sealed")])
    monkeypatch.setattr(
        run_chatbot_server.ChatbotServer, "serve_forever", lambda self: None
    )
    monkeypatch.setattr(
        run_chatbot_server.ChatbotServer,
        "shutdown",
        lambda self: self.httpd.server_close(),
    )
    monkeypatch.setattr(signal, "signal", lambda *_args: None)
    result = run_chatbot_server.run_chatbot_main(
        SimpleNamespace(
            workspace_manifest=str(manifest),
            server_port=None,
            expect_encrypted_jwt=False,
        )
    )
    assert result == 0
    output = capsys.readouterr().out
    assert "read-only workspace: Historical runs" in output
    assert "pick a workflow in the browser" not in output

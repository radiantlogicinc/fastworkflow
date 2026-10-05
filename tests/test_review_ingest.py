from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import threading
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from fastworkflow.observability.store import ObservabilityStore
from fastworkflow.observability.workspace import WORKSPACE_SCHEMA
from fastworkflow.review.sidecar import REVIEW_DATABASE_NAME
from fastworkflow.run_chatbot.server import ChatbotServer


def _assignment(assignment_id: str, prompt: str = "Choose an outcome.") -> dict:
    return {
        "id": assignment_id,
        "rater_slots": ["rater-a", "rater-b"],
        "adjudicator_slots": ["adjudicator"],
        "blinded": True,
        "rows": [
            {
                "id": "row-1",
                "turn_ref": {
                    "store_id": "store-a",
                    "logical_turn_key": "turn-1",
                },
            }
        ],
        "questions": [
            {
                "id": "verdict",
                "prompt": prompt,
                "type": "single-select",
                "vocabulary": ["pass", "fail"],
            }
        ],
    }


def _workspace(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "live.sqlite3"
    store = ObservabilityStore(str(source))
    archive = tmp_path / "sealed.sqlite3"
    with sqlite3.connect(source) as conn:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    shutil.copyfile(source, archive)
    archived = {
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "store_identity": store.store_identity(),
    }
    manifest = tmp_path / "workspace.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": WORKSPACE_SCHEMA,
                "workspace_id": "workspace-1",
                "label": "Review workspace",
                "stores": [
                    {
                        "store_id": "store-a",
                        "label": "Sealed evidence",
                        "path": archive.name,
                        "mode": "sealed",
                        "sha256": archived["sha256"],
                        "store_identity": archived["store_identity"],
                    }
                ],
                "experiments": [],
                "projected_attempts": [],
            }
        ),
        encoding="utf-8",
    )
    return manifest, archive


@contextmanager
def _serve(manifest: Path | None) -> Iterator[ChatbotServer]:
    server = ChatbotServer(
        port=0,
        workspace_manifest_path=str(manifest) if manifest is not None else None,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=5)


def _request(
    server: ChatbotServer,
    method: str,
    path: str,
    body: dict | None = None,
    *,
    token: str | None | object = ...,
    capability: str | None = None,
) -> tuple[int, dict]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        f"http://127.0.0.1:{server.port}{path}",
        method=method,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    presented_token = server.token if token is ... else token
    if isinstance(presented_token, str):
        request.add_header("Authorization", f"Bearer {presented_token}")
    if capability is not None:
        request.add_header("X-Review-Capability", capability)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


def test_post_ingests_assignment_and_token_gated_get_returns_it(tmp_path):
    manifest, _archive = _workspace(tmp_path)
    assignment = _assignment("assignment/one")
    encoded_id = urllib.parse.quote(assignment["id"], safe="")

    with _serve(manifest) as server:
        status, response = _request(
            server, "POST", "/api/review/assignments", assignment, token=None
        )
        assert status == 401

        status, created = _request(
            server, "POST", "/api/review/assignments", assignment
        )
        assert status == 201
        assert created["assignment_id"] == assignment["id"]
        assert set(created["rater_capabilities"]) == {"rater-a", "rater-b"}
        assert set(created["adjudicator_capabilities"]) == {"adjudicator"}

        status, response = _request(
            server,
            "GET",
            f"/api/review/assignments/{encoded_id}",
            token=None,
        )
        assert status == 401

        status, response = _request(
            server, "GET", f"/api/review/assignments/{encoded_id}"
        )
        assert status == 200
        assert response["assignment"]["id"] == assignment["id"]
        assert response["assignment"]["questions"][0]["prompt"] == (
            "Choose an outcome."
        )
        assert "rater_capabilities" not in response["assignment"]

        server.open_review_sidecar().capture_answer(
            created["rater_capabilities"]["rater-a"],
            "row-1",
            "verdict",
            "pass",
        )
        status, response = _request(
            server,
            "GET",
            f"/api/review/assignments/{encoded_id}/export",
            token=None,
        )
        assert status == 401

        status, response = _request(
            server, "GET", f"/api/review/assignments/{encoded_id}/export"
        )
        assert status == 403

        status, response = _request(
            server,
            "GET",
            f"/api/review/assignments/{encoded_id}/export",
            capability=created["rater_capabilities"]["rater-a"],
        )
        assert status == 403

        status, response = _request(
            server,
            "GET",
            f"/api/review/assignments/{encoded_id}/export",
            capability=created["adjudicator_capabilities"]["adjudicator"],
        )
        assert status == 200
        assert response["export"]["assignment_id"] == assignment["id"]
        assert response["export"]["rows"][0]["turn_ref"] == {
            "store_id": "store-a",
            "logical_turn_key": "turn-1",
        }
        assert response["export"]["rows"][0]["rater_answers"][0]["answers"] == [
            {"question_id": "verdict", "revision": 1, "answer": "pass"}
        ]


def test_two_assignments_over_same_workspace_coexist(tmp_path):
    manifest, _archive = _workspace(tmp_path)
    first = _assignment("assignment-1", prompt="First rubric")
    second = _assignment("assignment-2", prompt="Second rubric")

    with _serve(manifest) as server:
        first_created = _request(
            server, "POST", "/api/review/assignments", first
        )[1]
        second_created = _request(
            server, "POST", "/api/review/assignments", second
        )[1]

        first_read = _request(
            server, "GET", "/api/review/assignments/assignment-1"
        )[1]["assignment"]
        second_read = _request(
            server, "GET", "/api/review/assignments/assignment-2"
        )[1]["assignment"]

    assert first_read["questions"][0]["prompt"] == "First rubric"
    assert second_read["questions"][0]["prompt"] == "Second rubric"
    assert (
        first_created["rater_capabilities"]["rater-a"]
        != second_created["rater_capabilities"]["rater-a"]
    )


def test_capability_captures_and_progress_resumes_latest_revision(tmp_path):
    manifest, _archive = _workspace(tmp_path)
    assignment = _assignment("assignment/resume")
    encoded_id = urllib.parse.quote(assignment["id"], safe="")

    with _serve(manifest) as server:
        created = _request(
            server, "POST", "/api/review/assignments", assignment
        )[1]
        capability = created["rater_capabilities"]["rater-a"]
        answer_path = f"/api/review/assignments/{encoded_id}/answers"

        status, response = _request(
            server,
            "POST",
            answer_path,
            {"row_id": "row-1", "question_id": "verdict", "answer": "pass"},
        )
        assert status == 403
        assert "capability" in response["error"]

        status, response = _request(
            server,
            "GET",
            f"/api/review/assignments/{encoded_id}/progress",
        )
        assert status == 403
        assert "capability" in response["error"]

        for expected_revision, answer in enumerate(("pass", "fail"), start=1):
            status, response = _request(
                server,
                "POST",
                answer_path,
                {
                    "row_id": "row-1",
                    "question_id": "verdict",
                    "answer": answer,
                },
                capability=capability,
            )
            assert status == 200
            assert response["answer"]["revision"] == expected_revision

        status, response = _request(
            server,
            "GET",
            f"/api/review/assignments/{encoded_id}/progress",
            capability=capability,
        )
        assert status == 200
        progress = response["progress"]
        assert progress["rater_slot_id"] == "rater-a"
        assert progress["current_answers"][0]["answer"] == "fail"
        assert progress["current_answers"][0]["revision"] == 2


def test_invalid_assignment_reports_reason_and_workspace_is_required(tmp_path):
    manifest, _archive = _workspace(tmp_path)
    invalid = _assignment("invalid")
    invalid["questions"][0].pop("vocabulary")

    with _serve(manifest) as server:
        status, response = _request(
            server, "POST", "/api/review/assignments", invalid
        )
        assert status == 400
        assert "vocabulary" in response["error"]

    with _serve(None) as server:
        status, response = _request(
            server,
            "POST",
            "/api/review/assignments",
            _assignment("no-workspace"),
        )
        assert status == 409
        assert "workspace" in response["error"]


def test_ingest_keeps_sealed_archive_and_feedback_byte_identical(tmp_path):
    manifest, archive = _workspace(tmp_path)
    before = archive.read_bytes()
    before_digest = hashlib.sha256(before).hexdigest()
    immutable_uri = f"{archive.resolve().as_uri()}?mode=ro&immutable=1"
    with sqlite3.connect(immutable_uri, uri=True) as conn:
        feedback_before = conn.execute("SELECT COUNT(*) FROM human_feedback").fetchone()[0]

    with _serve(manifest) as server:
        status, _response = _request(
            server,
            "POST",
            "/api/review/assignments",
            _assignment("assignment-1"),
        )
        assert status == 201

    after = archive.read_bytes()
    with sqlite3.connect(immutable_uri, uri=True) as conn:
        feedback_after = conn.execute("SELECT COUNT(*) FROM human_feedback").fetchone()[0]
    assert after == before
    assert hashlib.sha256(after).hexdigest() == before_digest
    assert feedback_after == feedback_before
    assert (tmp_path / REVIEW_DATABASE_NAME).is_file()
    assert not Path(f"{archive}-wal").exists()
    assert not Path(f"{archive}-shm").exists()

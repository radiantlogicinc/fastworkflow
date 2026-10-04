from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from fastworkflow.observability.store import ObservabilityStore
from fastworkflow.observability.workspace import WORKSPACE_SCHEMA
from fastworkflow.run_chatbot.server import ChatbotServer


def _workspace(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "live.sqlite3"
    store = ObservabilityStore(str(source))
    with store._connect() as conn:
        # A recorded developer/agent review note in the evidence store. The
        # point of the fixture is that formal blinded review is a SEPARATE
        # sidecar: it must not read, rewrite or count this row. It used to be
        # a row in the agent-memory `feedback` table, which fix-9eg.16
        # removed; the separation it demonstrates is unchanged.
        conn.execute(
            """INSERT INTO human_feedback
                 (feedback_uid, turn_key, target_kind, span_ids_json,
                  target_label, comment, provenance, category, subcategory,
                  anchors_json, created_at)
               VALUES (?, ?, 'turn', '[]', 'Turn', ?, 'coding_agent',
                       'conclusions', 'what_went_wrong', '{}', 'now')""",
            ("fb-fixture-1", "turn-1", "keep separate"),
        )
        conn.commit()
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    archive = tmp_path / "sealed.sqlite3"
    shutil.copyfile(source, archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    manifest = tmp_path / "workspace.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": WORKSPACE_SCHEMA,
                "workspace_id": "review-integration",
                "label": "Review integration",
                "stores": [
                    {
                        "store_id": "sealed-a",
                        "label": "Sealed evidence",
                        "path": archive.name,
                        "mode": "sealed",
                        "sha256": digest,
                        "store_identity": store.store_identity(),
                    }
                ],
                "experiments": [],
                "projected_attempts": [],
            }
        ),
        encoding="utf-8",
    )
    return manifest, archive


def _assignment() -> dict:
    return {
        "id": "combined-review",
        "rater_slots": ["rater-a", "rater-b"],
        "adjudicator_slots": ["adjudicator-a"],
        "blinded": True,
        "rows": [{"id": "row-1", "turn_ref": {"turn_key": "turn-1"}}],
        "questions": [
            {
                "id": "verdict",
                "prompt": "Independent verdict",
                "type": "single-select",
                "vocabulary": ["alpha", "beta"],
            },
            {
                "id": "note",
                "prompt": "Independent note",
                "type": "bounded-note",
                "max_length": 80,
            }
        ],
    }


@contextmanager
def _serve(manifest: Path) -> Iterator[ChatbotServer]:
    server = ChatbotServer(port=0, workspace_manifest_path=str(manifest))
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
    capability: str | None = None,
) -> tuple[int, dict]:
    request = urllib.request.Request(
        f"http://127.0.0.1:{server.port}{path}",
        method=method,
        data=None if body is None else json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {server.token}",
            "Content-Type": "application/json",
        },
    )
    if capability is not None:
        request.add_header("X-Review-Capability", capability)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _capture(
    server: ChatbotServer,
    path: str,
    capability: str | None,
    answer: str,
) -> tuple[int, dict]:
    return _request(
        server,
        "POST",
        path,
        {"row_id": "row-1", "question_id": "verdict", "answer": answer},
        capability=capability,
    )


def test_assignment_stack_preserves_roles_revisions_and_sealed_evidence(tmp_path):
    manifest, archive = _workspace(tmp_path)
    archive_before = archive.read_bytes()
    immutable_uri = f"{archive.resolve().as_uri()}?mode=ro&immutable=1"
    with sqlite3.connect(immutable_uri, uri=True) as conn:
        feedback_before = conn.execute(
            "SELECT feedback_uid, turn_key, comment, created_at FROM human_feedback"
        ).fetchall()

    with _serve(manifest) as server:
        page = server.index_html.decode("utf-8")
        assert "Formal review · " in page
        assert "separate from developer/agent feedback" in page
        assert 'reviewApi(reviewBase + "/turn", "GET")' in page
        assert 'reviewApi(reviewBase + "/trace", "GET")' in page

        created_status, created = _request(
            server, "POST", "/api/review/assignments", _assignment()
        )
        assert created_status == 201
        answer_path = "/api/review/assignments/combined-review/answers"
        adjudication_path = (
            "/api/review/assignments/combined-review/adjudications"
        )

        status, _ = _capture(server, answer_path, None, "alpha")
        assert status == 403
        status, _ = _capture(
            server,
            answer_path,
            created["adjudicator_capabilities"]["adjudicator-a"],
            "alpha",
        )
        assert status == 403

        rater_a = created["rater_capabilities"]["rater-a"]
        rater_b = created["rater_capabilities"]["rater-b"]
        adjudicator = created["adjudicator_capabilities"]["adjudicator-a"]
        assert _capture(server, adjudication_path, None, "alpha")[0] == 403
        assert _capture(server, adjudication_path, rater_a, "alpha")[0] == 403
        assert _capture(server, answer_path, rater_a, "alpha")[1]["answer"][
            "revision"
        ] == 1
        assert _capture(server, answer_path, rater_a, "beta")[1]["answer"][
            "revision"
        ] == 2
        assert _capture(server, answer_path, rater_b, "alpha")[1]["answer"][
            "revision"
        ] == 1
        assert _request(
            server,
            "POST",
            answer_path,
            {
                "row_id": "row-1",
                "question_id": "note",
                "answer": "other-rater-secret",
            },
            capability=rater_b,
        )[0] == 200
        assert _capture(
            server, adjudication_path, adjudicator, "alpha"
        )[1]["adjudication"]["revision"] == 1
        assert _capture(
            server, adjudication_path, adjudicator, "beta"
        )[1]["adjudication"]["revision"] == 2

        progress_status, progress_payload = _request(
            server,
            "GET",
            "/api/review/assignments/combined-review/progress",
            capability=rater_a,
        )
        assert progress_status == 200
        progress = progress_payload["progress"]
        assert progress["current_answers"] == [
            {
                **progress["current_answers"][0],
                "rater_slot_id": "rater-a",
                "revision": 2,
                "answer": "beta",
            }
        ]
        serialized_progress = json.dumps(progress)
        assert '"status"' not in serialized_progress
        assert '"success"' not in serialized_progress
        assert "other-rater-secret" not in serialized_progress

        export_status, export_payload = _request(
            server,
            "GET",
            "/api/review/assignments/combined-review/export",
            capability=adjudicator,
        )
        assert export_status == 200
        exported = export_payload["export"]
        assert [row["revision"] for row in exported["answer_revisions"]] == [
            1,
            2,
            1,
            1,
        ]
        assert [row["revision"] for row in exported["adjudication_revisions"]] == [
            1,
            2,
        ]
        assert exported["rows"][0]["adjudicator_answers"][0]["answers"][0][
            "answer"
        ] == "beta"

    assert archive.read_bytes() == archive_before
    with sqlite3.connect(immutable_uri, uri=True) as conn:
        feedback_after = conn.execute(
            "SELECT feedback_uid, turn_key, comment, created_at FROM human_feedback"
        ).fetchall()
    assert feedback_after == feedback_before

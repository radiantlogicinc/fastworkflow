"""Status matrix for every chatbot route under the access-control conditions.

Characterization for fix-hzux.10. A real ``ChatbotServer`` on port 0 answers
one representative path of each route branch. The pinned dict is the golden
status for ``(mode, method, path, condition)``. A mismatch prints the cells
that moved.

Modes:
- ``live`` — workflow selected, observability DB present
- ``workspace`` — sealed workspace (workspace-only routes, plus the writes
  whose 403 is ordered differently from the live server)
- ``cold`` — workflow selected, no DB (the ``store is None`` GET branches)

Conditions are independent: ``bad_host`` and ``bad_origin`` still present the
real token, so a 403 is the host/origin gate and not a missing token.
``no_token`` uses the loopback Host and no Origin.

Write bodies are the smallest JSON that stays off destructive work: clear
sends a wrong confirm string, select/train name a path that does not exist,
configure sends a non-string so it 400s before writing, and the analysis PUT
sends an unexpected field so it 400s before ``write_analysis``.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from fastworkflow import state_paths
from fastworkflow.observability import store as obs
from fastworkflow.run_chatbot import server as run_chatbot_server
from tests.test_observability_workspace import _manifest, _seed_archive, _store_decl

CONDITIONS = ("no_token", "bad_host", "bad_origin", "valid")

# Paths that exercise each GET branch. Query strings are part of the path key
# when the branch is the query, not the path template.
_LIVE_GETS = (
    "/",
    "/index.html",
    "/trace",
    "/trace/unscoped",
    "/not-a-page",
    "/api/not-a-route",
    "/api/navigation",
    "/api/feedback-notes",
    "/api/feedback-notes?turn_key=missing",
    "/api/feedback-taxonomy",
    "/api/session",
    "/api/logs/server",
    "/api/logs/train",
    "/api/workflows",
    "/api/browse",
    "/api/experiment-setups",
    "/api/experiment-setups/missing",
    "/api/experiment-setups/missing/export",
    "/api/experiment-setups/missing/nope",
    "/api/benchmarks",
    "/api/benchmarks/missing",
    "/api/benchmarks/missing/experiments",
    "/api/benchmarks/missing/analysis",
    "/api/benchmarks/missing/versions/v1",
    "/api/benchmarks/missing/nope",
    "/api/benchmark-experiments/missing",
    "/api/experiments/e1",
    "/api/experiments/e1/winner",
    "/api/experiments/e1/winner/history",
    "/api/experiments/e1/tasks/t1/runs",
    "/api/experiments/e1/tasks/t1/runs/history",
    "/api/experiments/e1/tasks/t1/runs/1",
    "/api/experiments/e1/tasks/t1/runs/1/passes",
    "/api/experiments/e1/tasks/t1/selected-runs",
    "/api/experiments/e1/tasks/t1/selected-runs/validation",
    "/api/experiments/e1/tasks/t1/comparison",
    "/api/experiments/e1/tasks/t1/consistency",
    "/api/experiments/e1/tasks/t1/review-pairs",
    "/api/experiments/e1/tasks/t1/review-pairs/history",
    "/api/experiments/e1/not-a-route",
    "/api/meta",
    "/api/health",
    "/api/channels",
    "/api/conversations",
    "/api/turns",
    "/api/turn/missing",
    "/api/task-feedback",
    "/api/spans/missing",
    "/api/experiments",
    "/api/experiment/missing",
    "/api/experiment/missing/tasks",
    "/api/experiment/missing/attempts",
    "/api/experiment/missing/score",
    "/api/experiment/missing/compare",
    "/api/experiment/missing/nope",
    "/api/artifact/missing",
    "/api/training-runs",
    "/api/training-run/missing",
    "/api/workspace",
    "/api/review/assignments/missing",
    "/api/review/assignments/missing/export",
    "/api/review/assignments/missing/progress",
    "/api/review/assignments/missing/rows/row1/turn",
    "/api/review/assignments/missing/rows/row1/trace",
    # Removed with the agent-memory feedback table; still a 404 branch.
    "/api/feedback",
)

_WORKSPACE_GETS = (
    "/api/workspace",
    "/api/workspace/summary",
    "/api/workspace/stores",
    "/api/workspace/experiments",
    "/api/workspace/turns",
    "/api/workspace/turns?store_id=sealed",
    "/api/workspace/training-runs",
    "/api/workspace/training-runs?store_id=sealed",
    "/api/workspace/training-run/sealed/missing",
    "/api/workspace/training-run/sealed",
    "/api/workspace/task-feedback",
    "/api/workspace/projected_attempts",
    "/api/workspace/projected_attempts?attempt=nope",
    "/api/workspace/experiment/logical/segments",
    "/api/workspace/experiment/logical/tasks",
    "/api/workspace/experiment/logical/attempts",
    "/api/workspace/experiment/logical/nope",
    "/api/workspace/turn/sealed/turn",
    "/api/workspace/trace/sealed/turn",
    "/api/workspace/spans/sealed/turn",
    "/api/workspace/turn/sealed",
    "/api/workspace/nope",
    "/trace",
    "/trace/unscoped",
    "/trace?store_id=sealed&logical_turn_key=turn",
    "/trace?store_id=sealed&logical_turn_key=missing",
    # Unscoped live routes refused once a workspace is loaded.
    "/api/turns",
    "/api/experiments",
    "/api/training-runs",
    "/api/training-run/missing",
    "/api/turn/missing",
    "/api/spans/missing",
    "/api/experiment/missing",
    "/api/review/assignments/missing",
    "/api/review/assignments/missing/export",
    "/api/review/assignments/missing/progress",
    "/api/review/assignments/missing/rows/row1/turn",
    "/api/review/assignments/missing/rows/row1/trace",
    # Selection reads: judgement routes refuse; evidence routes answer.
    "/api/experiments/logical/winner",
    "/api/experiments/logical/tasks/task/runs",
    "/api/experiments/logical/tasks/task/runs/1",
    "/api/experiments/logical/tasks/task/runs/1/passes",
    "/api/experiments/logical/tasks/task/selected-runs",
    "/api/experiments/logical/tasks/task/selected-runs/validation",
    "/api/experiments/logical/tasks/task/comparison",
    "/api/experiments/logical/tasks/task/consistency",
    "/api/experiments/logical/tasks/task/review-pairs",
)

_COLD_GETS = (
    "/api/meta",
    "/api/health",
    "/api/channels",
    "/api/conversations",
    "/api/turns",
    "/api/experiments",
    "/api/training-runs",
    "/api/turn/missing",
    "/api/spans/missing",
    "/api/task-feedback",
    "/api/artifact/missing",
    "/api/experiment/missing",
)

# Every allowlisted write shape, plus unknown paths under /, /api/, and
# /api/experiments/<id>/.... HEAD/OPTIONS are separate rows below.
_WRITES = (
    ("POST", "/post_feedback"),
    ("POST", "/api/benchmark-setup"),
    ("POST", "/api/benchmarks/missing/experiments"),
    ("POST", "/api/experiments/e1/winner/decisions"),
    ("POST", "/api/experiments/e1/tasks/t1/best-run"),
    ("POST", "/api/experiments/e1/tasks/t1/best-run/undecided"),
    ("POST", "/api/experiments/e1/tasks/t1/review-pairs"),
    ("POST", "/api/benchmark-experiments/missing/duplicate"),
    ("POST", "/api/experiment-setups"),
    ("POST", "/api/experiment-setups/missing/decisions"),
    ("POST", "/api/review/assignments/missing/answers"),
    ("POST", "/api/review/assignments/missing/adjudications"),
    ("POST", "/api/review/assignments"),
    ("POST", "/api/benchmarks"),
    ("POST", "/api/benchmarks/missing/versions"),
    ("POST", "/api/select_workspace"),
    ("POST", "/api/select_workflow"),
    ("POST", "/api/configure_env"),
    ("POST", "/api/train"),
    ("POST", "/api/clear_conversations"),
    ("POST", "/not-a-page"),
    ("POST", "/api/not-a-route"),
    ("POST", "/api/experiments/e1/not-a-route"),
    ("PUT", "/api/benchmarks/missing/analysis"),
    ("PUT", "/api/not-a-route"),
    ("PUT", "/not-a-page"),
    ("DELETE", "/api/experiments/e1/tasks/t1/best-run"),
    ("DELETE", "/api/experiments/e1/not-a-route"),
    ("DELETE", "/api/benchmark-experiments/missing"),
    ("DELETE", "/api/not-a-route"),
    ("DELETE", "/not-a-page"),
    ("PATCH", "/api/experiment/missing"),
    ("PATCH", "/api/benchmark-experiments/missing"),
    ("PATCH", "/api/experiments/e1/not-a-route"),
    ("PATCH", "/api/not-a-route"),
    ("PATCH", "/not-a-page"),
)

_OTHER_VERBS = (
    ("HEAD", "/"),
    ("OPTIONS", "/"),
)


def _body(method: str, path: str):
    if method in ("GET", "HEAD", "OPTIONS"):
        return None
    if path == "/api/clear_conversations":
        return {"confirm": "do not clear"}
    if path in ("/api/select_workflow", "/api/train"):
        return {"path": "/no/such/fastworkflow-dir"}
    if path == "/api/select_workspace":
        return {"path": "/no/such/workspace-manifest.json"}
    if path == "/api/configure_env":
        return {"env_content": 1}
    if method == "PUT":
        return {"analysis": 1, "extra": True}
    return {}


def _cases():
    """(mode, method, path) rows the harness probes."""
    rows = [("live", "GET", path) for path in _LIVE_GETS]
    rows.extend(("live", method, path) for method, path in _WRITES)
    rows.extend(("live", method, path) for method, path in _OTHER_VERBS)
    rows.extend(("workspace", "GET", path) for path in _WORKSPACE_GETS)
    rows.extend(("workspace", method, path) for method, path in _WRITES)
    rows.extend(("cold", "GET", path) for path in _COLD_GETS)
    return rows


def _start(server):
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return thread


def _stop(server, thread):
    server.shutdown()
    thread.join(timeout=5)


@pytest.fixture
def live_server(tmp_path, monkeypatch):
    monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "state"))
    workflow = tmp_path / "my_workflow"
    workflow.mkdir()
    db_path = state_paths.observability_db(str(workflow))
    obs.ObservabilityStore(db_path)
    server = run_chatbot_server.ChatbotServer(
        db_path, workflow_path=str(workflow), port=0
    )
    thread = _start(server)
    yield server
    _stop(server, thread)


@pytest.fixture
def workspace_server(tmp_path):
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
    thread = _start(server)
    yield server
    _stop(server, thread)


@pytest.fixture
def cold_server(tmp_path):
    workflow = tmp_path / "cold_workflow"
    workflow.mkdir()
    server = run_chatbot_server.ChatbotServer(
        workflow_path=str(workflow), port=0
    )
    thread = _start(server)
    yield server
    _stop(server, thread)


def _request(server, method: str, path: str, condition: str) -> int:
    url = f"http://127.0.0.1:{server.port}{path}"
    body = _body(method, path)
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method)
    if condition != "no_token":
        req.add_header("Authorization", f"Bearer {server.token}")
    if condition == "bad_host":
        req.add_header("Host", "evil.example.com")
    if condition == "bad_origin":
        req.add_header("Origin", "http://evil.example.com")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read()
            return resp.status
    except urllib.error.HTTPError as err:
        err.read()
        return err.code


def _collect(servers) -> dict[tuple[str, str, str, str], int]:
    matrix: dict[tuple[str, str, str, str], int] = {}
    for mode, method, path in _cases():
        server = servers[mode]
        for condition in CONDITIONS:
            matrix[(mode, method, path, condition)] = _request(
                server, method, path, condition
            )
    return matrix


def _format_matrix(matrix: dict[tuple[str, str, str, str], int]) -> str:
    lines = ["PINNED_ROUTE_MATRIX = {"]
    for key in sorted(matrix):
        lines.append(f"    {key!r}: {matrix[key]},")
    lines.append("}")
    return "\n".join(lines)


def _cell_diff(expected, actual) -> str:
    lines = []
    for key in sorted(set(expected) | set(actual)):
        old = expected.get(key, "<missing>")
        new = actual.get(key, "<missing>")
        if old != new:
            mode, method, path, condition = key
            lines.append(
                f"  {mode} {method} {path} [{condition}]: {old} -> {new}"
            )
    return "\n".join(lines)


# Option B (owner decision 2026-09-30, fix-hzux.10): every write verb now
# gates Host/Origin, then the bearer token, then route lookup. These cells
# were 405 because the path is not on the write allowlist, which used to be
# checked before the gates. no_token moved 405 -> 401; bad_host and
# bad_origin moved 405 -> 403. The valid-token cell of each path stays 405.
# HEAD / and OPTIONS / stay 501: BaseHTTPRequestHandler answers them before
# any verb method, so `_gate` never sees them.
#
# Cells that changed:
#   live DELETE /api/not-a-route [no_token]: 405 -> 401
#   live DELETE /api/not-a-route [bad_host]: 405 -> 403
#   live DELETE /api/not-a-route [bad_origin]: 405 -> 403
#   live DELETE /not-a-page [no_token]: 405 -> 401
#   live DELETE /not-a-page [bad_host]: 405 -> 403
#   live DELETE /not-a-page [bad_origin]: 405 -> 403
#   live PATCH /api/experiments/e1/not-a-route [no_token]: 405 -> 401
#   live PATCH /api/experiments/e1/not-a-route [bad_host]: 405 -> 403
#   live PATCH /api/experiments/e1/not-a-route [bad_origin]: 405 -> 403
#   live PATCH /api/not-a-route [no_token]: 405 -> 401
#   live PATCH /api/not-a-route [bad_host]: 405 -> 403
#   live PATCH /api/not-a-route [bad_origin]: 405 -> 403
#   live PATCH /not-a-page [no_token]: 405 -> 401
#   live PATCH /not-a-page [bad_host]: 405 -> 403
#   live PATCH /not-a-page [bad_origin]: 405 -> 403
#   live POST /api/experiments/e1/not-a-route [no_token]: 405 -> 401
#   live POST /api/experiments/e1/not-a-route [bad_host]: 405 -> 403
#   live POST /api/experiments/e1/not-a-route [bad_origin]: 405 -> 403
#   live POST /api/not-a-route [no_token]: 405 -> 401
#   live POST /api/not-a-route [bad_host]: 405 -> 403
#   live POST /api/not-a-route [bad_origin]: 405 -> 403
#   live POST /not-a-page [no_token]: 405 -> 401
#   live POST /not-a-page [bad_host]: 405 -> 403
#   live POST /not-a-page [bad_origin]: 405 -> 403
#   live PUT /api/not-a-route [no_token]: 405 -> 401
#   live PUT /api/not-a-route [bad_host]: 405 -> 403
#   live PUT /api/not-a-route [bad_origin]: 405 -> 403
#   live PUT /not-a-page [no_token]: 405 -> 401
#   live PUT /not-a-page [bad_host]: 405 -> 403
#   live PUT /not-a-page [bad_origin]: 405 -> 403
#   workspace DELETE /api/not-a-route [no_token]: 405 -> 401
#   workspace DELETE /api/not-a-route [bad_host]: 405 -> 403
#   workspace DELETE /api/not-a-route [bad_origin]: 405 -> 403
#   workspace DELETE /not-a-page [no_token]: 405 -> 401
#   workspace DELETE /not-a-page [bad_host]: 405 -> 403
#   workspace DELETE /not-a-page [bad_origin]: 405 -> 403
#   workspace PATCH /api/experiments/e1/not-a-route [no_token]: 405 -> 401
#   workspace PATCH /api/experiments/e1/not-a-route [bad_host]: 405 -> 403
#   workspace PATCH /api/experiments/e1/not-a-route [bad_origin]: 405 -> 403
#   workspace PATCH /api/not-a-route [no_token]: 405 -> 401
#   workspace PATCH /api/not-a-route [bad_host]: 405 -> 403
#   workspace PATCH /api/not-a-route [bad_origin]: 405 -> 403
#   workspace PATCH /not-a-page [no_token]: 405 -> 401
#   workspace PATCH /not-a-page [bad_host]: 405 -> 403
#   workspace PATCH /not-a-page [bad_origin]: 405 -> 403
#   workspace POST /api/experiments/e1/not-a-route [no_token]: 405 -> 401
#   workspace POST /api/experiments/e1/not-a-route [bad_host]: 405 -> 403
#   workspace POST /api/experiments/e1/not-a-route [bad_origin]: 405 -> 403
#   workspace POST /api/not-a-route [no_token]: 405 -> 401
#   workspace POST /api/not-a-route [bad_host]: 405 -> 403
#   workspace POST /api/not-a-route [bad_origin]: 405 -> 403
#   workspace POST /not-a-page [no_token]: 405 -> 401
#   workspace POST /not-a-page [bad_host]: 405 -> 403
#   workspace POST /not-a-page [bad_origin]: 405 -> 403
#   workspace PUT /api/not-a-route [no_token]: 405 -> 401
#   workspace PUT /api/not-a-route [bad_host]: 405 -> 403
#   workspace PUT /api/not-a-route [bad_origin]: 405 -> 403
#   workspace PUT /not-a-page [no_token]: 405 -> 401
#   workspace PUT /not-a-page [bad_host]: 405 -> 403
#   workspace PUT /not-a-page [bad_origin]: 405 -> 403
PINNED_ROUTE_MATRIX: dict[tuple[str, str, str, str], int] = {
    ('cold', 'GET', '/api/artifact/missing', 'bad_host'): 403,
    ('cold', 'GET', '/api/artifact/missing', 'bad_origin'): 403,
    ('cold', 'GET', '/api/artifact/missing', 'no_token'): 401,
    ('cold', 'GET', '/api/artifact/missing', 'valid'): 404,
    ('cold', 'GET', '/api/channels', 'bad_host'): 403,
    ('cold', 'GET', '/api/channels', 'bad_origin'): 403,
    ('cold', 'GET', '/api/channels', 'no_token'): 401,
    ('cold', 'GET', '/api/channels', 'valid'): 200,
    ('cold', 'GET', '/api/conversations', 'bad_host'): 403,
    ('cold', 'GET', '/api/conversations', 'bad_origin'): 403,
    ('cold', 'GET', '/api/conversations', 'no_token'): 401,
    ('cold', 'GET', '/api/conversations', 'valid'): 200,
    ('cold', 'GET', '/api/experiment/missing', 'bad_host'): 403,
    ('cold', 'GET', '/api/experiment/missing', 'bad_origin'): 403,
    ('cold', 'GET', '/api/experiment/missing', 'no_token'): 401,
    ('cold', 'GET', '/api/experiment/missing', 'valid'): 404,
    ('cold', 'GET', '/api/experiments', 'bad_host'): 403,
    ('cold', 'GET', '/api/experiments', 'bad_origin'): 403,
    ('cold', 'GET', '/api/experiments', 'no_token'): 401,
    ('cold', 'GET', '/api/experiments', 'valid'): 200,
    ('cold', 'GET', '/api/health', 'bad_host'): 403,
    ('cold', 'GET', '/api/health', 'bad_origin'): 403,
    ('cold', 'GET', '/api/health', 'no_token'): 401,
    ('cold', 'GET', '/api/health', 'valid'): 200,
    ('cold', 'GET', '/api/meta', 'bad_host'): 403,
    ('cold', 'GET', '/api/meta', 'bad_origin'): 403,
    ('cold', 'GET', '/api/meta', 'no_token'): 401,
    ('cold', 'GET', '/api/meta', 'valid'): 200,
    ('cold', 'GET', '/api/spans/missing', 'bad_host'): 403,
    ('cold', 'GET', '/api/spans/missing', 'bad_origin'): 403,
    ('cold', 'GET', '/api/spans/missing', 'no_token'): 401,
    ('cold', 'GET', '/api/spans/missing', 'valid'): 404,
    ('cold', 'GET', '/api/task-feedback', 'bad_host'): 403,
    ('cold', 'GET', '/api/task-feedback', 'bad_origin'): 403,
    ('cold', 'GET', '/api/task-feedback', 'no_token'): 401,
    ('cold', 'GET', '/api/task-feedback', 'valid'): 404,
    ('cold', 'GET', '/api/training-runs', 'bad_host'): 403,
    ('cold', 'GET', '/api/training-runs', 'bad_origin'): 403,
    ('cold', 'GET', '/api/training-runs', 'no_token'): 401,
    ('cold', 'GET', '/api/training-runs', 'valid'): 200,
    ('cold', 'GET', '/api/turn/missing', 'bad_host'): 403,
    ('cold', 'GET', '/api/turn/missing', 'bad_origin'): 403,
    ('cold', 'GET', '/api/turn/missing', 'no_token'): 401,
    ('cold', 'GET', '/api/turn/missing', 'valid'): 404,
    ('cold', 'GET', '/api/turns', 'bad_host'): 403,
    ('cold', 'GET', '/api/turns', 'bad_origin'): 403,
    ('cold', 'GET', '/api/turns', 'no_token'): 401,
    ('cold', 'GET', '/api/turns', 'valid'): 200,
    ('live', 'DELETE', '/api/benchmark-experiments/missing', 'bad_host'): 403,
    ('live', 'DELETE', '/api/benchmark-experiments/missing', 'bad_origin'): 403,
    ('live', 'DELETE', '/api/benchmark-experiments/missing', 'no_token'): 401,
    ('live', 'DELETE', '/api/benchmark-experiments/missing', 'valid'): 404,
    ('live', 'DELETE', '/api/experiments/e1/not-a-route', 'bad_host'): 403,
    ('live', 'DELETE', '/api/experiments/e1/not-a-route', 'bad_origin'): 403,
    ('live', 'DELETE', '/api/experiments/e1/not-a-route', 'no_token'): 401,
    ('live', 'DELETE', '/api/experiments/e1/not-a-route', 'valid'): 404,
    ('live', 'DELETE', '/api/experiments/e1/tasks/t1/best-run', 'bad_host'): 403,
    ('live', 'DELETE', '/api/experiments/e1/tasks/t1/best-run', 'bad_origin'): 403,
    ('live', 'DELETE', '/api/experiments/e1/tasks/t1/best-run', 'no_token'): 401,
    ('live', 'DELETE', '/api/experiments/e1/tasks/t1/best-run', 'valid'): 404,
    ('live', 'DELETE', '/api/not-a-route', 'bad_host'): 403,
    ('live', 'DELETE', '/api/not-a-route', 'bad_origin'): 403,
    ('live', 'DELETE', '/api/not-a-route', 'no_token'): 401,
    ('live', 'DELETE', '/api/not-a-route', 'valid'): 405,
    ('live', 'DELETE', '/not-a-page', 'bad_host'): 403,
    ('live', 'DELETE', '/not-a-page', 'bad_origin'): 403,
    ('live', 'DELETE', '/not-a-page', 'no_token'): 401,
    ('live', 'DELETE', '/not-a-page', 'valid'): 405,
    ('live', 'GET', '/', 'bad_host'): 403,
    ('live', 'GET', '/', 'bad_origin'): 403,
    ('live', 'GET', '/', 'no_token'): 401,
    ('live', 'GET', '/', 'valid'): 200,
    ('live', 'GET', '/api/artifact/missing', 'bad_host'): 403,
    ('live', 'GET', '/api/artifact/missing', 'bad_origin'): 403,
    ('live', 'GET', '/api/artifact/missing', 'no_token'): 401,
    ('live', 'GET', '/api/artifact/missing', 'valid'): 404,
    ('live', 'GET', '/api/benchmark-experiments/missing', 'bad_host'): 403,
    ('live', 'GET', '/api/benchmark-experiments/missing', 'bad_origin'): 403,
    ('live', 'GET', '/api/benchmark-experiments/missing', 'no_token'): 401,
    ('live', 'GET', '/api/benchmark-experiments/missing', 'valid'): 404,
    ('live', 'GET', '/api/benchmarks', 'bad_host'): 403,
    ('live', 'GET', '/api/benchmarks', 'bad_origin'): 403,
    ('live', 'GET', '/api/benchmarks', 'no_token'): 401,
    ('live', 'GET', '/api/benchmarks', 'valid'): 200,
    ('live', 'GET', '/api/benchmarks/missing', 'bad_host'): 403,
    ('live', 'GET', '/api/benchmarks/missing', 'bad_origin'): 403,
    ('live', 'GET', '/api/benchmarks/missing', 'no_token'): 401,
    ('live', 'GET', '/api/benchmarks/missing', 'valid'): 200,
    ('live', 'GET', '/api/benchmarks/missing/analysis', 'bad_host'): 403,
    ('live', 'GET', '/api/benchmarks/missing/analysis', 'bad_origin'): 403,
    ('live', 'GET', '/api/benchmarks/missing/analysis', 'no_token'): 401,
    ('live', 'GET', '/api/benchmarks/missing/analysis', 'valid'): 200,
    ('live', 'GET', '/api/benchmarks/missing/experiments', 'bad_host'): 403,
    ('live', 'GET', '/api/benchmarks/missing/experiments', 'bad_origin'): 403,
    ('live', 'GET', '/api/benchmarks/missing/experiments', 'no_token'): 401,
    ('live', 'GET', '/api/benchmarks/missing/experiments', 'valid'): 200,
    ('live', 'GET', '/api/benchmarks/missing/nope', 'bad_host'): 403,
    ('live', 'GET', '/api/benchmarks/missing/nope', 'bad_origin'): 403,
    ('live', 'GET', '/api/benchmarks/missing/nope', 'no_token'): 401,
    ('live', 'GET', '/api/benchmarks/missing/nope', 'valid'): 404,
    ('live', 'GET', '/api/benchmarks/missing/versions/v1', 'bad_host'): 403,
    ('live', 'GET', '/api/benchmarks/missing/versions/v1', 'bad_origin'): 403,
    ('live', 'GET', '/api/benchmarks/missing/versions/v1', 'no_token'): 401,
    ('live', 'GET', '/api/benchmarks/missing/versions/v1', 'valid'): 404,
    ('live', 'GET', '/api/browse', 'bad_host'): 403,
    ('live', 'GET', '/api/browse', 'bad_origin'): 403,
    ('live', 'GET', '/api/browse', 'no_token'): 401,
    ('live', 'GET', '/api/browse', 'valid'): 200,
    ('live', 'GET', '/api/channels', 'bad_host'): 403,
    ('live', 'GET', '/api/channels', 'bad_origin'): 403,
    ('live', 'GET', '/api/channels', 'no_token'): 401,
    ('live', 'GET', '/api/channels', 'valid'): 200,
    ('live', 'GET', '/api/conversations', 'bad_host'): 403,
    ('live', 'GET', '/api/conversations', 'bad_origin'): 403,
    ('live', 'GET', '/api/conversations', 'no_token'): 401,
    ('live', 'GET', '/api/conversations', 'valid'): 200,
    ('live', 'GET', '/api/experiment-setups', 'bad_host'): 403,
    ('live', 'GET', '/api/experiment-setups', 'bad_origin'): 403,
    ('live', 'GET', '/api/experiment-setups', 'no_token'): 401,
    ('live', 'GET', '/api/experiment-setups', 'valid'): 200,
    ('live', 'GET', '/api/experiment-setups/missing', 'bad_host'): 403,
    ('live', 'GET', '/api/experiment-setups/missing', 'bad_origin'): 403,
    ('live', 'GET', '/api/experiment-setups/missing', 'no_token'): 401,
    ('live', 'GET', '/api/experiment-setups/missing', 'valid'): 404,
    ('live', 'GET', '/api/experiment-setups/missing/export', 'bad_host'): 403,
    ('live', 'GET', '/api/experiment-setups/missing/export', 'bad_origin'): 403,
    ('live', 'GET', '/api/experiment-setups/missing/export', 'no_token'): 401,
    ('live', 'GET', '/api/experiment-setups/missing/export', 'valid'): 404,
    ('live', 'GET', '/api/experiment-setups/missing/nope', 'bad_host'): 403,
    ('live', 'GET', '/api/experiment-setups/missing/nope', 'bad_origin'): 403,
    ('live', 'GET', '/api/experiment-setups/missing/nope', 'no_token'): 401,
    ('live', 'GET', '/api/experiment-setups/missing/nope', 'valid'): 404,
    ('live', 'GET', '/api/experiment/missing', 'bad_host'): 403,
    ('live', 'GET', '/api/experiment/missing', 'bad_origin'): 403,
    ('live', 'GET', '/api/experiment/missing', 'no_token'): 401,
    ('live', 'GET', '/api/experiment/missing', 'valid'): 404,
    ('live', 'GET', '/api/experiment/missing/attempts', 'bad_host'): 403,
    ('live', 'GET', '/api/experiment/missing/attempts', 'bad_origin'): 403,
    ('live', 'GET', '/api/experiment/missing/attempts', 'no_token'): 401,
    ('live', 'GET', '/api/experiment/missing/attempts', 'valid'): 404,
    ('live', 'GET', '/api/experiment/missing/compare', 'bad_host'): 403,
    ('live', 'GET', '/api/experiment/missing/compare', 'bad_origin'): 403,
    ('live', 'GET', '/api/experiment/missing/compare', 'no_token'): 401,
    ('live', 'GET', '/api/experiment/missing/compare', 'valid'): 404,
    ('live', 'GET', '/api/experiment/missing/nope', 'bad_host'): 403,
    ('live', 'GET', '/api/experiment/missing/nope', 'bad_origin'): 403,
    ('live', 'GET', '/api/experiment/missing/nope', 'no_token'): 401,
    ('live', 'GET', '/api/experiment/missing/nope', 'valid'): 404,
    ('live', 'GET', '/api/experiment/missing/score', 'bad_host'): 403,
    ('live', 'GET', '/api/experiment/missing/score', 'bad_origin'): 403,
    ('live', 'GET', '/api/experiment/missing/score', 'no_token'): 401,
    ('live', 'GET', '/api/experiment/missing/score', 'valid'): 404,
    ('live', 'GET', '/api/experiment/missing/tasks', 'bad_host'): 403,
    ('live', 'GET', '/api/experiment/missing/tasks', 'bad_origin'): 403,
    ('live', 'GET', '/api/experiment/missing/tasks', 'no_token'): 401,
    ('live', 'GET', '/api/experiment/missing/tasks', 'valid'): 404,
    ('live', 'GET', '/api/experiments', 'bad_host'): 403,
    ('live', 'GET', '/api/experiments', 'bad_origin'): 403,
    ('live', 'GET', '/api/experiments', 'no_token'): 401,
    ('live', 'GET', '/api/experiments', 'valid'): 200,
    ('live', 'GET', '/api/experiments/e1', 'bad_host'): 403,
    ('live', 'GET', '/api/experiments/e1', 'bad_origin'): 403,
    ('live', 'GET', '/api/experiments/e1', 'no_token'): 401,
    ('live', 'GET', '/api/experiments/e1', 'valid'): 404,
    ('live', 'GET', '/api/experiments/e1/not-a-route', 'bad_host'): 403,
    ('live', 'GET', '/api/experiments/e1/not-a-route', 'bad_origin'): 403,
    ('live', 'GET', '/api/experiments/e1/not-a-route', 'no_token'): 401,
    ('live', 'GET', '/api/experiments/e1/not-a-route', 'valid'): 404,
    ('live', 'GET', '/api/experiments/e1/tasks/t1/comparison', 'bad_host'): 403,
    ('live', 'GET', '/api/experiments/e1/tasks/t1/comparison', 'bad_origin'): 403,
    ('live', 'GET', '/api/experiments/e1/tasks/t1/comparison', 'no_token'): 401,
    ('live', 'GET', '/api/experiments/e1/tasks/t1/comparison', 'valid'): 404,
    ('live', 'GET', '/api/experiments/e1/tasks/t1/consistency', 'bad_host'): 403,
    ('live', 'GET', '/api/experiments/e1/tasks/t1/consistency', 'bad_origin'): 403,
    ('live', 'GET', '/api/experiments/e1/tasks/t1/consistency', 'no_token'): 401,
    ('live', 'GET', '/api/experiments/e1/tasks/t1/consistency', 'valid'): 404,
    ('live', 'GET', '/api/experiments/e1/tasks/t1/review-pairs', 'bad_host'): 403,
    ('live', 'GET', '/api/experiments/e1/tasks/t1/review-pairs', 'bad_origin'): 403,
    ('live', 'GET', '/api/experiments/e1/tasks/t1/review-pairs', 'no_token'): 401,
    ('live', 'GET', '/api/experiments/e1/tasks/t1/review-pairs', 'valid'): 404,
    ('live', 'GET', '/api/experiments/e1/tasks/t1/review-pairs/history', 'bad_host'): 403,
    ('live', 'GET', '/api/experiments/e1/tasks/t1/review-pairs/history', 'bad_origin'): 403,
    ('live', 'GET', '/api/experiments/e1/tasks/t1/review-pairs/history', 'no_token'): 401,
    ('live', 'GET', '/api/experiments/e1/tasks/t1/review-pairs/history', 'valid'): 404,
    ('live', 'GET', '/api/experiments/e1/tasks/t1/runs', 'bad_host'): 403,
    ('live', 'GET', '/api/experiments/e1/tasks/t1/runs', 'bad_origin'): 403,
    ('live', 'GET', '/api/experiments/e1/tasks/t1/runs', 'no_token'): 401,
    ('live', 'GET', '/api/experiments/e1/tasks/t1/runs', 'valid'): 404,
    ('live', 'GET', '/api/experiments/e1/tasks/t1/runs/1', 'bad_host'): 403,
    ('live', 'GET', '/api/experiments/e1/tasks/t1/runs/1', 'bad_origin'): 403,
    ('live', 'GET', '/api/experiments/e1/tasks/t1/runs/1', 'no_token'): 401,
    ('live', 'GET', '/api/experiments/e1/tasks/t1/runs/1', 'valid'): 404,
    ('live', 'GET', '/api/experiments/e1/tasks/t1/runs/1/passes', 'bad_host'): 403,
    ('live', 'GET', '/api/experiments/e1/tasks/t1/runs/1/passes', 'bad_origin'): 403,
    ('live', 'GET', '/api/experiments/e1/tasks/t1/runs/1/passes', 'no_token'): 401,
    ('live', 'GET', '/api/experiments/e1/tasks/t1/runs/1/passes', 'valid'): 404,
    ('live', 'GET', '/api/experiments/e1/tasks/t1/runs/history', 'bad_host'): 403,
    ('live', 'GET', '/api/experiments/e1/tasks/t1/runs/history', 'bad_origin'): 403,
    ('live', 'GET', '/api/experiments/e1/tasks/t1/runs/history', 'no_token'): 401,
    ('live', 'GET', '/api/experiments/e1/tasks/t1/runs/history', 'valid'): 404,
    ('live', 'GET', '/api/experiments/e1/tasks/t1/selected-runs', 'bad_host'): 403,
    ('live', 'GET', '/api/experiments/e1/tasks/t1/selected-runs', 'bad_origin'): 403,
    ('live', 'GET', '/api/experiments/e1/tasks/t1/selected-runs', 'no_token'): 401,
    ('live', 'GET', '/api/experiments/e1/tasks/t1/selected-runs', 'valid'): 404,
    ('live', 'GET', '/api/experiments/e1/tasks/t1/selected-runs/validation', 'bad_host'): 403,
    ('live', 'GET', '/api/experiments/e1/tasks/t1/selected-runs/validation', 'bad_origin'): 403,
    ('live', 'GET', '/api/experiments/e1/tasks/t1/selected-runs/validation', 'no_token'): 401,
    ('live', 'GET', '/api/experiments/e1/tasks/t1/selected-runs/validation', 'valid'): 404,
    ('live', 'GET', '/api/experiments/e1/winner', 'bad_host'): 403,
    ('live', 'GET', '/api/experiments/e1/winner', 'bad_origin'): 403,
    ('live', 'GET', '/api/experiments/e1/winner', 'no_token'): 401,
    ('live', 'GET', '/api/experiments/e1/winner', 'valid'): 404,
    ('live', 'GET', '/api/experiments/e1/winner/history', 'bad_host'): 403,
    ('live', 'GET', '/api/experiments/e1/winner/history', 'bad_origin'): 403,
    ('live', 'GET', '/api/experiments/e1/winner/history', 'no_token'): 401,
    ('live', 'GET', '/api/experiments/e1/winner/history', 'valid'): 404,
    ('live', 'GET', '/api/feedback', 'bad_host'): 403,
    ('live', 'GET', '/api/feedback', 'bad_origin'): 403,
    ('live', 'GET', '/api/feedback', 'no_token'): 401,
    ('live', 'GET', '/api/feedback', 'valid'): 404,
    ('live', 'GET', '/api/feedback-notes', 'bad_host'): 403,
    ('live', 'GET', '/api/feedback-notes', 'bad_origin'): 403,
    ('live', 'GET', '/api/feedback-notes', 'no_token'): 401,
    ('live', 'GET', '/api/feedback-notes', 'valid'): 400,
    ('live', 'GET', '/api/feedback-notes?turn_key=missing', 'bad_host'): 403,
    ('live', 'GET', '/api/feedback-notes?turn_key=missing', 'bad_origin'): 403,
    ('live', 'GET', '/api/feedback-notes?turn_key=missing', 'no_token'): 401,
    ('live', 'GET', '/api/feedback-notes?turn_key=missing', 'valid'): 404,
    ('live', 'GET', '/api/feedback-taxonomy', 'bad_host'): 403,
    ('live', 'GET', '/api/feedback-taxonomy', 'bad_origin'): 403,
    ('live', 'GET', '/api/feedback-taxonomy', 'no_token'): 401,
    ('live', 'GET', '/api/feedback-taxonomy', 'valid'): 200,
    ('live', 'GET', '/api/health', 'bad_host'): 403,
    ('live', 'GET', '/api/health', 'bad_origin'): 403,
    ('live', 'GET', '/api/health', 'no_token'): 401,
    ('live', 'GET', '/api/health', 'valid'): 200,
    ('live', 'GET', '/api/logs/server', 'bad_host'): 403,
    ('live', 'GET', '/api/logs/server', 'bad_origin'): 403,
    ('live', 'GET', '/api/logs/server', 'no_token'): 401,
    ('live', 'GET', '/api/logs/server', 'valid'): 200,
    ('live', 'GET', '/api/logs/train', 'bad_host'): 403,
    ('live', 'GET', '/api/logs/train', 'bad_origin'): 403,
    ('live', 'GET', '/api/logs/train', 'no_token'): 401,
    ('live', 'GET', '/api/logs/train', 'valid'): 200,
    ('live', 'GET', '/api/meta', 'bad_host'): 403,
    ('live', 'GET', '/api/meta', 'bad_origin'): 403,
    ('live', 'GET', '/api/meta', 'no_token'): 401,
    ('live', 'GET', '/api/meta', 'valid'): 200,
    ('live', 'GET', '/api/navigation', 'bad_host'): 403,
    ('live', 'GET', '/api/navigation', 'bad_origin'): 403,
    ('live', 'GET', '/api/navigation', 'no_token'): 401,
    ('live', 'GET', '/api/navigation', 'valid'): 200,
    ('live', 'GET', '/api/not-a-route', 'bad_host'): 403,
    ('live', 'GET', '/api/not-a-route', 'bad_origin'): 403,
    ('live', 'GET', '/api/not-a-route', 'no_token'): 401,
    ('live', 'GET', '/api/not-a-route', 'valid'): 404,
    ('live', 'GET', '/api/review/assignments/missing', 'bad_host'): 403,
    ('live', 'GET', '/api/review/assignments/missing', 'bad_origin'): 403,
    ('live', 'GET', '/api/review/assignments/missing', 'no_token'): 401,
    ('live', 'GET', '/api/review/assignments/missing', 'valid'): 404,
    ('live', 'GET', '/api/review/assignments/missing/export', 'bad_host'): 403,
    ('live', 'GET', '/api/review/assignments/missing/export', 'bad_origin'): 403,
    ('live', 'GET', '/api/review/assignments/missing/export', 'no_token'): 401,
    ('live', 'GET', '/api/review/assignments/missing/export', 'valid'): 404,
    ('live', 'GET', '/api/review/assignments/missing/progress', 'bad_host'): 403,
    ('live', 'GET', '/api/review/assignments/missing/progress', 'bad_origin'): 403,
    ('live', 'GET', '/api/review/assignments/missing/progress', 'no_token'): 401,
    ('live', 'GET', '/api/review/assignments/missing/progress', 'valid'): 404,
    ('live', 'GET', '/api/review/assignments/missing/rows/row1/trace', 'bad_host'): 403,
    ('live', 'GET', '/api/review/assignments/missing/rows/row1/trace', 'bad_origin'): 403,
    ('live', 'GET', '/api/review/assignments/missing/rows/row1/trace', 'no_token'): 401,
    ('live', 'GET', '/api/review/assignments/missing/rows/row1/trace', 'valid'): 404,
    ('live', 'GET', '/api/review/assignments/missing/rows/row1/turn', 'bad_host'): 403,
    ('live', 'GET', '/api/review/assignments/missing/rows/row1/turn', 'bad_origin'): 403,
    ('live', 'GET', '/api/review/assignments/missing/rows/row1/turn', 'no_token'): 401,
    ('live', 'GET', '/api/review/assignments/missing/rows/row1/turn', 'valid'): 404,
    ('live', 'GET', '/api/session', 'bad_host'): 403,
    ('live', 'GET', '/api/session', 'bad_origin'): 403,
    ('live', 'GET', '/api/session', 'no_token'): 401,
    ('live', 'GET', '/api/session', 'valid'): 200,
    ('live', 'GET', '/api/spans/missing', 'bad_host'): 403,
    ('live', 'GET', '/api/spans/missing', 'bad_origin'): 403,
    ('live', 'GET', '/api/spans/missing', 'no_token'): 401,
    ('live', 'GET', '/api/spans/missing', 'valid'): 200,
    ('live', 'GET', '/api/task-feedback', 'bad_host'): 403,
    ('live', 'GET', '/api/task-feedback', 'bad_origin'): 403,
    ('live', 'GET', '/api/task-feedback', 'no_token'): 401,
    ('live', 'GET', '/api/task-feedback', 'valid'): 400,
    ('live', 'GET', '/api/training-run/missing', 'bad_host'): 403,
    ('live', 'GET', '/api/training-run/missing', 'bad_origin'): 403,
    ('live', 'GET', '/api/training-run/missing', 'no_token'): 401,
    ('live', 'GET', '/api/training-run/missing', 'valid'): 404,
    ('live', 'GET', '/api/training-runs', 'bad_host'): 403,
    ('live', 'GET', '/api/training-runs', 'bad_origin'): 403,
    ('live', 'GET', '/api/training-runs', 'no_token'): 401,
    ('live', 'GET', '/api/training-runs', 'valid'): 200,
    ('live', 'GET', '/api/turn/missing', 'bad_host'): 403,
    ('live', 'GET', '/api/turn/missing', 'bad_origin'): 403,
    ('live', 'GET', '/api/turn/missing', 'no_token'): 401,
    ('live', 'GET', '/api/turn/missing', 'valid'): 404,
    ('live', 'GET', '/api/turns', 'bad_host'): 403,
    ('live', 'GET', '/api/turns', 'bad_origin'): 403,
    ('live', 'GET', '/api/turns', 'no_token'): 401,
    ('live', 'GET', '/api/turns', 'valid'): 200,
    ('live', 'GET', '/api/workflows', 'bad_host'): 403,
    ('live', 'GET', '/api/workflows', 'bad_origin'): 403,
    ('live', 'GET', '/api/workflows', 'no_token'): 401,
    ('live', 'GET', '/api/workflows', 'valid'): 200,
    ('live', 'GET', '/api/workspace', 'bad_host'): 403,
    ('live', 'GET', '/api/workspace', 'bad_origin'): 403,
    ('live', 'GET', '/api/workspace', 'no_token'): 401,
    ('live', 'GET', '/api/workspace', 'valid'): 404,
    ('live', 'GET', '/index.html', 'bad_host'): 403,
    ('live', 'GET', '/index.html', 'bad_origin'): 403,
    ('live', 'GET', '/index.html', 'no_token'): 401,
    ('live', 'GET', '/index.html', 'valid'): 200,
    ('live', 'GET', '/not-a-page', 'bad_host'): 403,
    ('live', 'GET', '/not-a-page', 'bad_origin'): 403,
    ('live', 'GET', '/not-a-page', 'no_token'): 401,
    ('live', 'GET', '/not-a-page', 'valid'): 404,
    ('live', 'GET', '/trace', 'bad_host'): 403,
    ('live', 'GET', '/trace', 'bad_origin'): 403,
    ('live', 'GET', '/trace', 'no_token'): 401,
    ('live', 'GET', '/trace', 'valid'): 404,
    ('live', 'GET', '/trace/unscoped', 'bad_host'): 403,
    ('live', 'GET', '/trace/unscoped', 'bad_origin'): 403,
    ('live', 'GET', '/trace/unscoped', 'no_token'): 401,
    ('live', 'GET', '/trace/unscoped', 'valid'): 404,
    ('live', 'HEAD', '/', 'bad_host'): 501,
    ('live', 'HEAD', '/', 'bad_origin'): 501,
    ('live', 'HEAD', '/', 'no_token'): 501,
    ('live', 'HEAD', '/', 'valid'): 501,
    ('live', 'OPTIONS', '/', 'bad_host'): 501,
    ('live', 'OPTIONS', '/', 'bad_origin'): 501,
    ('live', 'OPTIONS', '/', 'no_token'): 501,
    ('live', 'OPTIONS', '/', 'valid'): 501,
    ('live', 'PATCH', '/api/benchmark-experiments/missing', 'bad_host'): 403,
    ('live', 'PATCH', '/api/benchmark-experiments/missing', 'bad_origin'): 403,
    ('live', 'PATCH', '/api/benchmark-experiments/missing', 'no_token'): 401,
    ('live', 'PATCH', '/api/benchmark-experiments/missing', 'valid'): 400,
    ('live', 'PATCH', '/api/experiment/missing', 'bad_host'): 403,
    ('live', 'PATCH', '/api/experiment/missing', 'bad_origin'): 403,
    ('live', 'PATCH', '/api/experiment/missing', 'no_token'): 401,
    ('live', 'PATCH', '/api/experiment/missing', 'valid'): 400,
    ('live', 'PATCH', '/api/experiments/e1/not-a-route', 'bad_host'): 403,
    ('live', 'PATCH', '/api/experiments/e1/not-a-route', 'bad_origin'): 403,
    ('live', 'PATCH', '/api/experiments/e1/not-a-route', 'no_token'): 401,
    ('live', 'PATCH', '/api/experiments/e1/not-a-route', 'valid'): 405,
    ('live', 'PATCH', '/api/not-a-route', 'bad_host'): 403,
    ('live', 'PATCH', '/api/not-a-route', 'bad_origin'): 403,
    ('live', 'PATCH', '/api/not-a-route', 'no_token'): 401,
    ('live', 'PATCH', '/api/not-a-route', 'valid'): 405,
    ('live', 'PATCH', '/not-a-page', 'bad_host'): 403,
    ('live', 'PATCH', '/not-a-page', 'bad_origin'): 403,
    ('live', 'PATCH', '/not-a-page', 'no_token'): 401,
    ('live', 'PATCH', '/not-a-page', 'valid'): 405,
    ('live', 'POST', '/api/benchmark-experiments/missing/duplicate', 'bad_host'): 403,
    ('live', 'POST', '/api/benchmark-experiments/missing/duplicate', 'bad_origin'): 403,
    ('live', 'POST', '/api/benchmark-experiments/missing/duplicate', 'no_token'): 401,
    ('live', 'POST', '/api/benchmark-experiments/missing/duplicate', 'valid'): 404,
    ('live', 'POST', '/api/benchmark-setup', 'bad_host'): 403,
    ('live', 'POST', '/api/benchmark-setup', 'bad_origin'): 403,
    ('live', 'POST', '/api/benchmark-setup', 'no_token'): 401,
    ('live', 'POST', '/api/benchmark-setup', 'valid'): 400,
    ('live', 'POST', '/api/benchmarks', 'bad_host'): 403,
    ('live', 'POST', '/api/benchmarks', 'bad_origin'): 403,
    ('live', 'POST', '/api/benchmarks', 'no_token'): 401,
    ('live', 'POST', '/api/benchmarks', 'valid'): 400,
    ('live', 'POST', '/api/benchmarks/missing/experiments', 'bad_host'): 403,
    ('live', 'POST', '/api/benchmarks/missing/experiments', 'bad_origin'): 403,
    ('live', 'POST', '/api/benchmarks/missing/experiments', 'no_token'): 401,
    ('live', 'POST', '/api/benchmarks/missing/experiments', 'valid'): 400,
    ('live', 'POST', '/api/benchmarks/missing/versions', 'bad_host'): 403,
    ('live', 'POST', '/api/benchmarks/missing/versions', 'bad_origin'): 403,
    ('live', 'POST', '/api/benchmarks/missing/versions', 'no_token'): 401,
    ('live', 'POST', '/api/benchmarks/missing/versions', 'valid'): 400,
    ('live', 'POST', '/api/clear_conversations', 'bad_host'): 403,
    ('live', 'POST', '/api/clear_conversations', 'bad_origin'): 403,
    ('live', 'POST', '/api/clear_conversations', 'no_token'): 401,
    ('live', 'POST', '/api/clear_conversations', 'valid'): 400,
    ('live', 'POST', '/api/configure_env', 'bad_host'): 403,
    ('live', 'POST', '/api/configure_env', 'bad_origin'): 403,
    ('live', 'POST', '/api/configure_env', 'no_token'): 401,
    ('live', 'POST', '/api/configure_env', 'valid'): 400,
    ('live', 'POST', '/api/experiment-setups', 'bad_host'): 403,
    ('live', 'POST', '/api/experiment-setups', 'bad_origin'): 403,
    ('live', 'POST', '/api/experiment-setups', 'no_token'): 401,
    ('live', 'POST', '/api/experiment-setups', 'valid'): 400,
    ('live', 'POST', '/api/experiment-setups/missing/decisions', 'bad_host'): 403,
    ('live', 'POST', '/api/experiment-setups/missing/decisions', 'bad_origin'): 403,
    ('live', 'POST', '/api/experiment-setups/missing/decisions', 'no_token'): 401,
    ('live', 'POST', '/api/experiment-setups/missing/decisions', 'valid'): 400,
    ('live', 'POST', '/api/experiments/e1/not-a-route', 'bad_host'): 403,
    ('live', 'POST', '/api/experiments/e1/not-a-route', 'bad_origin'): 403,
    ('live', 'POST', '/api/experiments/e1/not-a-route', 'no_token'): 401,
    ('live', 'POST', '/api/experiments/e1/not-a-route', 'valid'): 405,
    ('live', 'POST', '/api/experiments/e1/tasks/t1/best-run', 'bad_host'): 403,
    ('live', 'POST', '/api/experiments/e1/tasks/t1/best-run', 'bad_origin'): 403,
    ('live', 'POST', '/api/experiments/e1/tasks/t1/best-run', 'no_token'): 401,
    ('live', 'POST', '/api/experiments/e1/tasks/t1/best-run', 'valid'): 404,
    ('live', 'POST', '/api/experiments/e1/tasks/t1/best-run/undecided', 'bad_host'): 403,
    ('live', 'POST', '/api/experiments/e1/tasks/t1/best-run/undecided', 'bad_origin'): 403,
    ('live', 'POST', '/api/experiments/e1/tasks/t1/best-run/undecided', 'no_token'): 401,
    ('live', 'POST', '/api/experiments/e1/tasks/t1/best-run/undecided', 'valid'): 404,
    ('live', 'POST', '/api/experiments/e1/tasks/t1/review-pairs', 'bad_host'): 403,
    ('live', 'POST', '/api/experiments/e1/tasks/t1/review-pairs', 'bad_origin'): 403,
    ('live', 'POST', '/api/experiments/e1/tasks/t1/review-pairs', 'no_token'): 401,
    ('live', 'POST', '/api/experiments/e1/tasks/t1/review-pairs', 'valid'): 404,
    ('live', 'POST', '/api/experiments/e1/winner/decisions', 'bad_host'): 403,
    ('live', 'POST', '/api/experiments/e1/winner/decisions', 'bad_origin'): 403,
    ('live', 'POST', '/api/experiments/e1/winner/decisions', 'no_token'): 401,
    ('live', 'POST', '/api/experiments/e1/winner/decisions', 'valid'): 404,
    ('live', 'POST', '/api/not-a-route', 'bad_host'): 403,
    ('live', 'POST', '/api/not-a-route', 'bad_origin'): 403,
    ('live', 'POST', '/api/not-a-route', 'no_token'): 401,
    ('live', 'POST', '/api/not-a-route', 'valid'): 405,
    ('live', 'POST', '/api/review/assignments', 'bad_host'): 403,
    ('live', 'POST', '/api/review/assignments', 'bad_origin'): 403,
    ('live', 'POST', '/api/review/assignments', 'no_token'): 401,
    ('live', 'POST', '/api/review/assignments', 'valid'): 409,
    ('live', 'POST', '/api/review/assignments/missing/adjudications', 'bad_host'): 403,
    ('live', 'POST', '/api/review/assignments/missing/adjudications', 'bad_origin'): 403,
    ('live', 'POST', '/api/review/assignments/missing/adjudications', 'no_token'): 401,
    ('live', 'POST', '/api/review/assignments/missing/adjudications', 'valid'): 409,
    ('live', 'POST', '/api/review/assignments/missing/answers', 'bad_host'): 403,
    ('live', 'POST', '/api/review/assignments/missing/answers', 'bad_origin'): 403,
    ('live', 'POST', '/api/review/assignments/missing/answers', 'no_token'): 401,
    ('live', 'POST', '/api/review/assignments/missing/answers', 'valid'): 409,
    ('live', 'POST', '/api/select_workflow', 'bad_host'): 403,
    ('live', 'POST', '/api/select_workflow', 'bad_origin'): 403,
    ('live', 'POST', '/api/select_workflow', 'no_token'): 401,
    ('live', 'POST', '/api/select_workflow', 'valid'): 400,
    ('live', 'POST', '/api/select_workspace', 'bad_host'): 403,
    ('live', 'POST', '/api/select_workspace', 'bad_origin'): 403,
    ('live', 'POST', '/api/select_workspace', 'no_token'): 401,
    ('live', 'POST', '/api/select_workspace', 'valid'): 400,
    ('live', 'POST', '/api/train', 'bad_host'): 403,
    ('live', 'POST', '/api/train', 'bad_origin'): 403,
    ('live', 'POST', '/api/train', 'no_token'): 401,
    ('live', 'POST', '/api/train', 'valid'): 400,
    ('live', 'POST', '/not-a-page', 'bad_host'): 403,
    ('live', 'POST', '/not-a-page', 'bad_origin'): 403,
    ('live', 'POST', '/not-a-page', 'no_token'): 401,
    ('live', 'POST', '/not-a-page', 'valid'): 405,
    ('live', 'POST', '/post_feedback', 'bad_host'): 403,
    ('live', 'POST', '/post_feedback', 'bad_origin'): 403,
    ('live', 'POST', '/post_feedback', 'no_token'): 401,
    ('live', 'POST', '/post_feedback', 'valid'): 400,
    ('live', 'PUT', '/api/benchmarks/missing/analysis', 'bad_host'): 403,
    ('live', 'PUT', '/api/benchmarks/missing/analysis', 'bad_origin'): 403,
    ('live', 'PUT', '/api/benchmarks/missing/analysis', 'no_token'): 401,
    ('live', 'PUT', '/api/benchmarks/missing/analysis', 'valid'): 400,
    ('live', 'PUT', '/api/not-a-route', 'bad_host'): 403,
    ('live', 'PUT', '/api/not-a-route', 'bad_origin'): 403,
    ('live', 'PUT', '/api/not-a-route', 'no_token'): 401,
    ('live', 'PUT', '/api/not-a-route', 'valid'): 405,
    ('live', 'PUT', '/not-a-page', 'bad_host'): 403,
    ('live', 'PUT', '/not-a-page', 'bad_origin'): 403,
    ('live', 'PUT', '/not-a-page', 'no_token'): 401,
    ('live', 'PUT', '/not-a-page', 'valid'): 405,
    ('workspace', 'DELETE', '/api/benchmark-experiments/missing', 'bad_host'): 403,
    ('workspace', 'DELETE', '/api/benchmark-experiments/missing', 'bad_origin'): 403,
    ('workspace', 'DELETE', '/api/benchmark-experiments/missing', 'no_token'): 401,
    ('workspace', 'DELETE', '/api/benchmark-experiments/missing', 'valid'): 403,
    ('workspace', 'DELETE', '/api/experiments/e1/not-a-route', 'bad_host'): 403,
    ('workspace', 'DELETE', '/api/experiments/e1/not-a-route', 'bad_origin'): 403,
    ('workspace', 'DELETE', '/api/experiments/e1/not-a-route', 'no_token'): 401,
    ('workspace', 'DELETE', '/api/experiments/e1/not-a-route', 'valid'): 403,
    ('workspace', 'DELETE', '/api/experiments/e1/tasks/t1/best-run', 'bad_host'): 403,
    ('workspace', 'DELETE', '/api/experiments/e1/tasks/t1/best-run', 'bad_origin'): 403,
    ('workspace', 'DELETE', '/api/experiments/e1/tasks/t1/best-run', 'no_token'): 401,
    ('workspace', 'DELETE', '/api/experiments/e1/tasks/t1/best-run', 'valid'): 403,
    ('workspace', 'DELETE', '/api/not-a-route', 'bad_host'): 403,
    ('workspace', 'DELETE', '/api/not-a-route', 'bad_origin'): 403,
    ('workspace', 'DELETE', '/api/not-a-route', 'no_token'): 401,
    ('workspace', 'DELETE', '/api/not-a-route', 'valid'): 405,
    ('workspace', 'DELETE', '/not-a-page', 'bad_host'): 403,
    ('workspace', 'DELETE', '/not-a-page', 'bad_origin'): 403,
    ('workspace', 'DELETE', '/not-a-page', 'no_token'): 401,
    ('workspace', 'DELETE', '/not-a-page', 'valid'): 405,
    ('workspace', 'GET', '/api/experiment/missing', 'bad_host'): 403,
    ('workspace', 'GET', '/api/experiment/missing', 'bad_origin'): 403,
    ('workspace', 'GET', '/api/experiment/missing', 'no_token'): 401,
    ('workspace', 'GET', '/api/experiment/missing', 'valid'): 400,
    ('workspace', 'GET', '/api/experiments', 'bad_host'): 403,
    ('workspace', 'GET', '/api/experiments', 'bad_origin'): 403,
    ('workspace', 'GET', '/api/experiments', 'no_token'): 401,
    ('workspace', 'GET', '/api/experiments', 'valid'): 400,
    ('workspace', 'GET', '/api/experiments/logical/tasks/task/comparison', 'bad_host'): 403,
    ('workspace', 'GET', '/api/experiments/logical/tasks/task/comparison', 'bad_origin'): 403,
    ('workspace', 'GET', '/api/experiments/logical/tasks/task/comparison', 'no_token'): 401,
    ('workspace', 'GET', '/api/experiments/logical/tasks/task/comparison', 'valid'): 200,
    ('workspace', 'GET', '/api/experiments/logical/tasks/task/consistency', 'bad_host'): 403,
    ('workspace', 'GET', '/api/experiments/logical/tasks/task/consistency', 'bad_origin'): 403,
    ('workspace', 'GET', '/api/experiments/logical/tasks/task/consistency', 'no_token'): 401,
    ('workspace', 'GET', '/api/experiments/logical/tasks/task/consistency', 'valid'): 200,
    ('workspace', 'GET', '/api/experiments/logical/tasks/task/review-pairs', 'bad_host'): 403,
    ('workspace', 'GET', '/api/experiments/logical/tasks/task/review-pairs', 'bad_origin'): 403,
    ('workspace', 'GET', '/api/experiments/logical/tasks/task/review-pairs', 'no_token'): 401,
    ('workspace', 'GET', '/api/experiments/logical/tasks/task/review-pairs', 'valid'): 403,
    ('workspace', 'GET', '/api/experiments/logical/tasks/task/runs', 'bad_host'): 403,
    ('workspace', 'GET', '/api/experiments/logical/tasks/task/runs', 'bad_origin'): 403,
    ('workspace', 'GET', '/api/experiments/logical/tasks/task/runs', 'no_token'): 401,
    ('workspace', 'GET', '/api/experiments/logical/tasks/task/runs', 'valid'): 200,
    ('workspace', 'GET', '/api/experiments/logical/tasks/task/runs/1', 'bad_host'): 403,
    ('workspace', 'GET', '/api/experiments/logical/tasks/task/runs/1', 'bad_origin'): 403,
    ('workspace', 'GET', '/api/experiments/logical/tasks/task/runs/1', 'no_token'): 401,
    ('workspace', 'GET', '/api/experiments/logical/tasks/task/runs/1', 'valid'): 200,
    ('workspace', 'GET', '/api/experiments/logical/tasks/task/runs/1/passes', 'bad_host'): 403,
    ('workspace', 'GET', '/api/experiments/logical/tasks/task/runs/1/passes', 'bad_origin'): 403,
    ('workspace', 'GET', '/api/experiments/logical/tasks/task/runs/1/passes', 'no_token'): 401,
    ('workspace', 'GET', '/api/experiments/logical/tasks/task/runs/1/passes', 'valid'): 200,
    ('workspace', 'GET', '/api/experiments/logical/tasks/task/selected-runs', 'bad_host'): 403,
    ('workspace', 'GET', '/api/experiments/logical/tasks/task/selected-runs', 'bad_origin'): 403,
    ('workspace', 'GET', '/api/experiments/logical/tasks/task/selected-runs', 'no_token'): 401,
    ('workspace', 'GET', '/api/experiments/logical/tasks/task/selected-runs', 'valid'): 400,
    ('workspace', 'GET', '/api/experiments/logical/tasks/task/selected-runs/validation', 'bad_host'): 403,
    ('workspace', 'GET', '/api/experiments/logical/tasks/task/selected-runs/validation', 'bad_origin'): 403,
    ('workspace', 'GET', '/api/experiments/logical/tasks/task/selected-runs/validation', 'no_token'): 401,
    ('workspace', 'GET', '/api/experiments/logical/tasks/task/selected-runs/validation', 'valid'): 400,
    ('workspace', 'GET', '/api/experiments/logical/winner', 'bad_host'): 403,
    ('workspace', 'GET', '/api/experiments/logical/winner', 'bad_origin'): 403,
    ('workspace', 'GET', '/api/experiments/logical/winner', 'no_token'): 401,
    ('workspace', 'GET', '/api/experiments/logical/winner', 'valid'): 403,
    ('workspace', 'GET', '/api/review/assignments/missing', 'bad_host'): 403,
    ('workspace', 'GET', '/api/review/assignments/missing', 'bad_origin'): 403,
    ('workspace', 'GET', '/api/review/assignments/missing', 'no_token'): 401,
    ('workspace', 'GET', '/api/review/assignments/missing', 'valid'): 404,
    ('workspace', 'GET', '/api/review/assignments/missing/export', 'bad_host'): 403,
    ('workspace', 'GET', '/api/review/assignments/missing/export', 'bad_origin'): 403,
    ('workspace', 'GET', '/api/review/assignments/missing/export', 'no_token'): 401,
    ('workspace', 'GET', '/api/review/assignments/missing/export', 'valid'): 403,
    ('workspace', 'GET', '/api/review/assignments/missing/progress', 'bad_host'): 403,
    ('workspace', 'GET', '/api/review/assignments/missing/progress', 'bad_origin'): 403,
    ('workspace', 'GET', '/api/review/assignments/missing/progress', 'no_token'): 401,
    ('workspace', 'GET', '/api/review/assignments/missing/progress', 'valid'): 403,
    ('workspace', 'GET', '/api/review/assignments/missing/rows/row1/trace', 'bad_host'): 403,
    ('workspace', 'GET', '/api/review/assignments/missing/rows/row1/trace', 'bad_origin'): 403,
    ('workspace', 'GET', '/api/review/assignments/missing/rows/row1/trace', 'no_token'): 401,
    ('workspace', 'GET', '/api/review/assignments/missing/rows/row1/trace', 'valid'): 403,
    ('workspace', 'GET', '/api/review/assignments/missing/rows/row1/turn', 'bad_host'): 403,
    ('workspace', 'GET', '/api/review/assignments/missing/rows/row1/turn', 'bad_origin'): 403,
    ('workspace', 'GET', '/api/review/assignments/missing/rows/row1/turn', 'no_token'): 401,
    ('workspace', 'GET', '/api/review/assignments/missing/rows/row1/turn', 'valid'): 403,
    ('workspace', 'GET', '/api/spans/missing', 'bad_host'): 403,
    ('workspace', 'GET', '/api/spans/missing', 'bad_origin'): 403,
    ('workspace', 'GET', '/api/spans/missing', 'no_token'): 401,
    ('workspace', 'GET', '/api/spans/missing', 'valid'): 400,
    ('workspace', 'GET', '/api/training-run/missing', 'bad_host'): 403,
    ('workspace', 'GET', '/api/training-run/missing', 'bad_origin'): 403,
    ('workspace', 'GET', '/api/training-run/missing', 'no_token'): 401,
    ('workspace', 'GET', '/api/training-run/missing', 'valid'): 400,
    ('workspace', 'GET', '/api/training-runs', 'bad_host'): 403,
    ('workspace', 'GET', '/api/training-runs', 'bad_origin'): 403,
    ('workspace', 'GET', '/api/training-runs', 'no_token'): 401,
    ('workspace', 'GET', '/api/training-runs', 'valid'): 400,
    ('workspace', 'GET', '/api/turn/missing', 'bad_host'): 403,
    ('workspace', 'GET', '/api/turn/missing', 'bad_origin'): 403,
    ('workspace', 'GET', '/api/turn/missing', 'no_token'): 401,
    ('workspace', 'GET', '/api/turn/missing', 'valid'): 400,
    ('workspace', 'GET', '/api/turns', 'bad_host'): 403,
    ('workspace', 'GET', '/api/turns', 'bad_origin'): 403,
    ('workspace', 'GET', '/api/turns', 'no_token'): 401,
    ('workspace', 'GET', '/api/turns', 'valid'): 400,
    ('workspace', 'GET', '/api/workspace', 'bad_host'): 403,
    ('workspace', 'GET', '/api/workspace', 'bad_origin'): 403,
    ('workspace', 'GET', '/api/workspace', 'no_token'): 401,
    ('workspace', 'GET', '/api/workspace', 'valid'): 200,
    ('workspace', 'GET', '/api/workspace/experiment/logical/attempts', 'bad_host'): 403,
    ('workspace', 'GET', '/api/workspace/experiment/logical/attempts', 'bad_origin'): 403,
    ('workspace', 'GET', '/api/workspace/experiment/logical/attempts', 'no_token'): 401,
    ('workspace', 'GET', '/api/workspace/experiment/logical/attempts', 'valid'): 200,
    ('workspace', 'GET', '/api/workspace/experiment/logical/nope', 'bad_host'): 403,
    ('workspace', 'GET', '/api/workspace/experiment/logical/nope', 'bad_origin'): 403,
    ('workspace', 'GET', '/api/workspace/experiment/logical/nope', 'no_token'): 401,
    ('workspace', 'GET', '/api/workspace/experiment/logical/nope', 'valid'): 404,
    ('workspace', 'GET', '/api/workspace/experiment/logical/segments', 'bad_host'): 403,
    ('workspace', 'GET', '/api/workspace/experiment/logical/segments', 'bad_origin'): 403,
    ('workspace', 'GET', '/api/workspace/experiment/logical/segments', 'no_token'): 401,
    ('workspace', 'GET', '/api/workspace/experiment/logical/segments', 'valid'): 200,
    ('workspace', 'GET', '/api/workspace/experiment/logical/tasks', 'bad_host'): 403,
    ('workspace', 'GET', '/api/workspace/experiment/logical/tasks', 'bad_origin'): 403,
    ('workspace', 'GET', '/api/workspace/experiment/logical/tasks', 'no_token'): 401,
    ('workspace', 'GET', '/api/workspace/experiment/logical/tasks', 'valid'): 200,
    ('workspace', 'GET', '/api/workspace/experiments', 'bad_host'): 403,
    ('workspace', 'GET', '/api/workspace/experiments', 'bad_origin'): 403,
    ('workspace', 'GET', '/api/workspace/experiments', 'no_token'): 401,
    ('workspace', 'GET', '/api/workspace/experiments', 'valid'): 200,
    ('workspace', 'GET', '/api/workspace/nope', 'bad_host'): 403,
    ('workspace', 'GET', '/api/workspace/nope', 'bad_origin'): 403,
    ('workspace', 'GET', '/api/workspace/nope', 'no_token'): 401,
    ('workspace', 'GET', '/api/workspace/nope', 'valid'): 404,
    ('workspace', 'GET', '/api/workspace/projected_attempts', 'bad_host'): 403,
    ('workspace', 'GET', '/api/workspace/projected_attempts', 'bad_origin'): 403,
    ('workspace', 'GET', '/api/workspace/projected_attempts', 'no_token'): 401,
    ('workspace', 'GET', '/api/workspace/projected_attempts', 'valid'): 200,
    ('workspace', 'GET', '/api/workspace/projected_attempts?attempt=nope', 'bad_host'): 403,
    ('workspace', 'GET', '/api/workspace/projected_attempts?attempt=nope', 'bad_origin'): 403,
    ('workspace', 'GET', '/api/workspace/projected_attempts?attempt=nope', 'no_token'): 401,
    ('workspace', 'GET', '/api/workspace/projected_attempts?attempt=nope', 'valid'): 400,
    ('workspace', 'GET', '/api/workspace/spans/sealed/turn', 'bad_host'): 403,
    ('workspace', 'GET', '/api/workspace/spans/sealed/turn', 'bad_origin'): 403,
    ('workspace', 'GET', '/api/workspace/spans/sealed/turn', 'no_token'): 401,
    ('workspace', 'GET', '/api/workspace/spans/sealed/turn', 'valid'): 200,
    ('workspace', 'GET', '/api/workspace/stores', 'bad_host'): 403,
    ('workspace', 'GET', '/api/workspace/stores', 'bad_origin'): 403,
    ('workspace', 'GET', '/api/workspace/stores', 'no_token'): 401,
    ('workspace', 'GET', '/api/workspace/stores', 'valid'): 200,
    ('workspace', 'GET', '/api/workspace/summary', 'bad_host'): 403,
    ('workspace', 'GET', '/api/workspace/summary', 'bad_origin'): 403,
    ('workspace', 'GET', '/api/workspace/summary', 'no_token'): 401,
    ('workspace', 'GET', '/api/workspace/summary', 'valid'): 200,
    ('workspace', 'GET', '/api/workspace/task-feedback', 'bad_host'): 403,
    ('workspace', 'GET', '/api/workspace/task-feedback', 'bad_origin'): 403,
    ('workspace', 'GET', '/api/workspace/task-feedback', 'no_token'): 401,
    ('workspace', 'GET', '/api/workspace/task-feedback', 'valid'): 400,
    ('workspace', 'GET', '/api/workspace/trace/sealed/turn', 'bad_host'): 403,
    ('workspace', 'GET', '/api/workspace/trace/sealed/turn', 'bad_origin'): 403,
    ('workspace', 'GET', '/api/workspace/trace/sealed/turn', 'no_token'): 401,
    ('workspace', 'GET', '/api/workspace/trace/sealed/turn', 'valid'): 200,
    ('workspace', 'GET', '/api/workspace/training-run/sealed', 'bad_host'): 403,
    ('workspace', 'GET', '/api/workspace/training-run/sealed', 'bad_origin'): 403,
    ('workspace', 'GET', '/api/workspace/training-run/sealed', 'no_token'): 401,
    ('workspace', 'GET', '/api/workspace/training-run/sealed', 'valid'): 400,
    ('workspace', 'GET', '/api/workspace/training-run/sealed/missing', 'bad_host'): 403,
    ('workspace', 'GET', '/api/workspace/training-run/sealed/missing', 'bad_origin'): 403,
    ('workspace', 'GET', '/api/workspace/training-run/sealed/missing', 'no_token'): 401,
    ('workspace', 'GET', '/api/workspace/training-run/sealed/missing', 'valid'): 404,
    ('workspace', 'GET', '/api/workspace/training-runs', 'bad_host'): 403,
    ('workspace', 'GET', '/api/workspace/training-runs', 'bad_origin'): 403,
    ('workspace', 'GET', '/api/workspace/training-runs', 'no_token'): 401,
    ('workspace', 'GET', '/api/workspace/training-runs', 'valid'): 400,
    ('workspace', 'GET', '/api/workspace/training-runs?store_id=sealed', 'bad_host'): 403,
    ('workspace', 'GET', '/api/workspace/training-runs?store_id=sealed', 'bad_origin'): 403,
    ('workspace', 'GET', '/api/workspace/training-runs?store_id=sealed', 'no_token'): 401,
    ('workspace', 'GET', '/api/workspace/training-runs?store_id=sealed', 'valid'): 200,
    ('workspace', 'GET', '/api/workspace/turn/sealed', 'bad_host'): 403,
    ('workspace', 'GET', '/api/workspace/turn/sealed', 'bad_origin'): 403,
    ('workspace', 'GET', '/api/workspace/turn/sealed', 'no_token'): 401,
    ('workspace', 'GET', '/api/workspace/turn/sealed', 'valid'): 400,
    ('workspace', 'GET', '/api/workspace/turn/sealed/turn', 'bad_host'): 403,
    ('workspace', 'GET', '/api/workspace/turn/sealed/turn', 'bad_origin'): 403,
    ('workspace', 'GET', '/api/workspace/turn/sealed/turn', 'no_token'): 401,
    ('workspace', 'GET', '/api/workspace/turn/sealed/turn', 'valid'): 200,
    ('workspace', 'GET', '/api/workspace/turns', 'bad_host'): 403,
    ('workspace', 'GET', '/api/workspace/turns', 'bad_origin'): 403,
    ('workspace', 'GET', '/api/workspace/turns', 'no_token'): 401,
    ('workspace', 'GET', '/api/workspace/turns', 'valid'): 400,
    ('workspace', 'GET', '/api/workspace/turns?store_id=sealed', 'bad_host'): 403,
    ('workspace', 'GET', '/api/workspace/turns?store_id=sealed', 'bad_origin'): 403,
    ('workspace', 'GET', '/api/workspace/turns?store_id=sealed', 'no_token'): 401,
    ('workspace', 'GET', '/api/workspace/turns?store_id=sealed', 'valid'): 200,
    ('workspace', 'GET', '/trace', 'bad_host'): 403,
    ('workspace', 'GET', '/trace', 'bad_origin'): 403,
    ('workspace', 'GET', '/trace', 'no_token'): 401,
    ('workspace', 'GET', '/trace', 'valid'): 400,
    ('workspace', 'GET', '/trace/unscoped', 'bad_host'): 403,
    ('workspace', 'GET', '/trace/unscoped', 'bad_origin'): 403,
    ('workspace', 'GET', '/trace/unscoped', 'no_token'): 401,
    ('workspace', 'GET', '/trace/unscoped', 'valid'): 400,
    ('workspace', 'GET', '/trace?store_id=sealed&logical_turn_key=missing', 'bad_host'): 403,
    ('workspace', 'GET', '/trace?store_id=sealed&logical_turn_key=missing', 'bad_origin'): 403,
    ('workspace', 'GET', '/trace?store_id=sealed&logical_turn_key=missing', 'no_token'): 401,
    ('workspace', 'GET', '/trace?store_id=sealed&logical_turn_key=missing', 'valid'): 404,
    ('workspace', 'GET', '/trace?store_id=sealed&logical_turn_key=turn', 'bad_host'): 403,
    ('workspace', 'GET', '/trace?store_id=sealed&logical_turn_key=turn', 'bad_origin'): 403,
    ('workspace', 'GET', '/trace?store_id=sealed&logical_turn_key=turn', 'no_token'): 401,
    ('workspace', 'GET', '/trace?store_id=sealed&logical_turn_key=turn', 'valid'): 200,
    ('workspace', 'PATCH', '/api/benchmark-experiments/missing', 'bad_host'): 403,
    ('workspace', 'PATCH', '/api/benchmark-experiments/missing', 'bad_origin'): 403,
    ('workspace', 'PATCH', '/api/benchmark-experiments/missing', 'no_token'): 401,
    ('workspace', 'PATCH', '/api/benchmark-experiments/missing', 'valid'): 403,
    ('workspace', 'PATCH', '/api/experiment/missing', 'bad_host'): 403,
    ('workspace', 'PATCH', '/api/experiment/missing', 'bad_origin'): 403,
    ('workspace', 'PATCH', '/api/experiment/missing', 'no_token'): 401,
    ('workspace', 'PATCH', '/api/experiment/missing', 'valid'): 403,
    ('workspace', 'PATCH', '/api/experiments/e1/not-a-route', 'bad_host'): 403,
    ('workspace', 'PATCH', '/api/experiments/e1/not-a-route', 'bad_origin'): 403,
    ('workspace', 'PATCH', '/api/experiments/e1/not-a-route', 'no_token'): 401,
    ('workspace', 'PATCH', '/api/experiments/e1/not-a-route', 'valid'): 405,
    ('workspace', 'PATCH', '/api/not-a-route', 'bad_host'): 403,
    ('workspace', 'PATCH', '/api/not-a-route', 'bad_origin'): 403,
    ('workspace', 'PATCH', '/api/not-a-route', 'no_token'): 401,
    ('workspace', 'PATCH', '/api/not-a-route', 'valid'): 405,
    ('workspace', 'PATCH', '/not-a-page', 'bad_host'): 403,
    ('workspace', 'PATCH', '/not-a-page', 'bad_origin'): 403,
    ('workspace', 'PATCH', '/not-a-page', 'no_token'): 401,
    ('workspace', 'PATCH', '/not-a-page', 'valid'): 405,
    ('workspace', 'POST', '/api/benchmark-experiments/missing/duplicate', 'bad_host'): 403,
    ('workspace', 'POST', '/api/benchmark-experiments/missing/duplicate', 'bad_origin'): 403,
    ('workspace', 'POST', '/api/benchmark-experiments/missing/duplicate', 'no_token'): 401,
    ('workspace', 'POST', '/api/benchmark-experiments/missing/duplicate', 'valid'): 403,
    ('workspace', 'POST', '/api/benchmark-setup', 'bad_host'): 403,
    ('workspace', 'POST', '/api/benchmark-setup', 'bad_origin'): 403,
    ('workspace', 'POST', '/api/benchmark-setup', 'no_token'): 401,
    ('workspace', 'POST', '/api/benchmark-setup', 'valid'): 403,
    ('workspace', 'POST', '/api/benchmarks', 'bad_host'): 403,
    ('workspace', 'POST', '/api/benchmarks', 'bad_origin'): 403,
    ('workspace', 'POST', '/api/benchmarks', 'no_token'): 401,
    ('workspace', 'POST', '/api/benchmarks', 'valid'): 403,
    ('workspace', 'POST', '/api/benchmarks/missing/experiments', 'bad_host'): 403,
    ('workspace', 'POST', '/api/benchmarks/missing/experiments', 'bad_origin'): 403,
    ('workspace', 'POST', '/api/benchmarks/missing/experiments', 'no_token'): 401,
    ('workspace', 'POST', '/api/benchmarks/missing/experiments', 'valid'): 403,
    ('workspace', 'POST', '/api/benchmarks/missing/versions', 'bad_host'): 403,
    ('workspace', 'POST', '/api/benchmarks/missing/versions', 'bad_origin'): 403,
    ('workspace', 'POST', '/api/benchmarks/missing/versions', 'no_token'): 401,
    ('workspace', 'POST', '/api/benchmarks/missing/versions', 'valid'): 403,
    ('workspace', 'POST', '/api/clear_conversations', 'bad_host'): 403,
    ('workspace', 'POST', '/api/clear_conversations', 'bad_origin'): 403,
    ('workspace', 'POST', '/api/clear_conversations', 'no_token'): 401,
    ('workspace', 'POST', '/api/clear_conversations', 'valid'): 403,
    ('workspace', 'POST', '/api/configure_env', 'bad_host'): 403,
    ('workspace', 'POST', '/api/configure_env', 'bad_origin'): 403,
    ('workspace', 'POST', '/api/configure_env', 'no_token'): 401,
    ('workspace', 'POST', '/api/configure_env', 'valid'): 403,
    ('workspace', 'POST', '/api/experiment-setups', 'bad_host'): 403,
    ('workspace', 'POST', '/api/experiment-setups', 'bad_origin'): 403,
    ('workspace', 'POST', '/api/experiment-setups', 'no_token'): 401,
    ('workspace', 'POST', '/api/experiment-setups', 'valid'): 403,
    ('workspace', 'POST', '/api/experiment-setups/missing/decisions', 'bad_host'): 403,
    ('workspace', 'POST', '/api/experiment-setups/missing/decisions', 'bad_origin'): 403,
    ('workspace', 'POST', '/api/experiment-setups/missing/decisions', 'no_token'): 401,
    ('workspace', 'POST', '/api/experiment-setups/missing/decisions', 'valid'): 403,
    ('workspace', 'POST', '/api/experiments/e1/not-a-route', 'bad_host'): 403,
    ('workspace', 'POST', '/api/experiments/e1/not-a-route', 'bad_origin'): 403,
    ('workspace', 'POST', '/api/experiments/e1/not-a-route', 'no_token'): 401,
    ('workspace', 'POST', '/api/experiments/e1/not-a-route', 'valid'): 405,
    ('workspace', 'POST', '/api/experiments/e1/tasks/t1/best-run', 'bad_host'): 403,
    ('workspace', 'POST', '/api/experiments/e1/tasks/t1/best-run', 'bad_origin'): 403,
    ('workspace', 'POST', '/api/experiments/e1/tasks/t1/best-run', 'no_token'): 401,
    ('workspace', 'POST', '/api/experiments/e1/tasks/t1/best-run', 'valid'): 403,
    ('workspace', 'POST', '/api/experiments/e1/tasks/t1/best-run/undecided', 'bad_host'): 403,
    ('workspace', 'POST', '/api/experiments/e1/tasks/t1/best-run/undecided', 'bad_origin'): 403,
    ('workspace', 'POST', '/api/experiments/e1/tasks/t1/best-run/undecided', 'no_token'): 401,
    ('workspace', 'POST', '/api/experiments/e1/tasks/t1/best-run/undecided', 'valid'): 403,
    ('workspace', 'POST', '/api/experiments/e1/tasks/t1/review-pairs', 'bad_host'): 403,
    ('workspace', 'POST', '/api/experiments/e1/tasks/t1/review-pairs', 'bad_origin'): 403,
    ('workspace', 'POST', '/api/experiments/e1/tasks/t1/review-pairs', 'no_token'): 401,
    ('workspace', 'POST', '/api/experiments/e1/tasks/t1/review-pairs', 'valid'): 403,
    ('workspace', 'POST', '/api/experiments/e1/winner/decisions', 'bad_host'): 403,
    ('workspace', 'POST', '/api/experiments/e1/winner/decisions', 'bad_origin'): 403,
    ('workspace', 'POST', '/api/experiments/e1/winner/decisions', 'no_token'): 401,
    ('workspace', 'POST', '/api/experiments/e1/winner/decisions', 'valid'): 403,
    ('workspace', 'POST', '/api/not-a-route', 'bad_host'): 403,
    ('workspace', 'POST', '/api/not-a-route', 'bad_origin'): 403,
    ('workspace', 'POST', '/api/not-a-route', 'no_token'): 401,
    ('workspace', 'POST', '/api/not-a-route', 'valid'): 405,
    ('workspace', 'POST', '/api/review/assignments', 'bad_host'): 403,
    ('workspace', 'POST', '/api/review/assignments', 'bad_origin'): 403,
    ('workspace', 'POST', '/api/review/assignments', 'no_token'): 401,
    ('workspace', 'POST', '/api/review/assignments', 'valid'): 400,
    ('workspace', 'POST', '/api/review/assignments/missing/adjudications', 'bad_host'): 403,
    ('workspace', 'POST', '/api/review/assignments/missing/adjudications', 'bad_origin'): 403,
    ('workspace', 'POST', '/api/review/assignments/missing/adjudications', 'no_token'): 401,
    ('workspace', 'POST', '/api/review/assignments/missing/adjudications', 'valid'): 403,
    ('workspace', 'POST', '/api/review/assignments/missing/answers', 'bad_host'): 403,
    ('workspace', 'POST', '/api/review/assignments/missing/answers', 'bad_origin'): 403,
    ('workspace', 'POST', '/api/review/assignments/missing/answers', 'no_token'): 401,
    ('workspace', 'POST', '/api/review/assignments/missing/answers', 'valid'): 403,
    ('workspace', 'POST', '/api/select_workflow', 'bad_host'): 403,
    ('workspace', 'POST', '/api/select_workflow', 'bad_origin'): 403,
    ('workspace', 'POST', '/api/select_workflow', 'no_token'): 401,
    ('workspace', 'POST', '/api/select_workflow', 'valid'): 403,
    ('workspace', 'POST', '/api/select_workspace', 'bad_host'): 403,
    ('workspace', 'POST', '/api/select_workspace', 'bad_origin'): 403,
    ('workspace', 'POST', '/api/select_workspace', 'no_token'): 401,
    ('workspace', 'POST', '/api/select_workspace', 'valid'): 403,
    ('workspace', 'POST', '/api/train', 'bad_host'): 403,
    ('workspace', 'POST', '/api/train', 'bad_origin'): 403,
    ('workspace', 'POST', '/api/train', 'no_token'): 401,
    ('workspace', 'POST', '/api/train', 'valid'): 403,
    ('workspace', 'POST', '/not-a-page', 'bad_host'): 403,
    ('workspace', 'POST', '/not-a-page', 'bad_origin'): 403,
    ('workspace', 'POST', '/not-a-page', 'no_token'): 401,
    ('workspace', 'POST', '/not-a-page', 'valid'): 405,
    ('workspace', 'POST', '/post_feedback', 'bad_host'): 403,
    ('workspace', 'POST', '/post_feedback', 'bad_origin'): 403,
    ('workspace', 'POST', '/post_feedback', 'no_token'): 401,
    ('workspace', 'POST', '/post_feedback', 'valid'): 400,
    ('workspace', 'PUT', '/api/benchmarks/missing/analysis', 'bad_host'): 403,
    ('workspace', 'PUT', '/api/benchmarks/missing/analysis', 'bad_origin'): 403,
    ('workspace', 'PUT', '/api/benchmarks/missing/analysis', 'no_token'): 401,
    ('workspace', 'PUT', '/api/benchmarks/missing/analysis', 'valid'): 403,
    ('workspace', 'PUT', '/api/not-a-route', 'bad_host'): 403,
    ('workspace', 'PUT', '/api/not-a-route', 'bad_origin'): 403,
    ('workspace', 'PUT', '/api/not-a-route', 'no_token'): 401,
    ('workspace', 'PUT', '/api/not-a-route', 'valid'): 405,
    ('workspace', 'PUT', '/not-a-page', 'bad_host'): 403,
    ('workspace', 'PUT', '/not-a-page', 'bad_origin'): 403,
    ('workspace', 'PUT', '/not-a-page', 'no_token'): 401,
    ('workspace', 'PUT', '/not-a-page', 'valid'): 405,
}


def test_route_status_matrix_matches_pin(live_server, workspace_server, cold_server):
    actual = _collect(
        {
            "live": live_server,
            "workspace": workspace_server,
            "cold": cold_server,
        }
    )
    if actual != PINNED_ROUTE_MATRIX:
        dump = Path("/tmp/fw_route_matrix.py")
        dump.write_text(_format_matrix(actual) + "\n", encoding="utf-8")
    assert actual == PINNED_ROUTE_MATRIX, (
        f"{len(_cell_diff(PINNED_ROUTE_MATRIX, actual).splitlines())} cells differ "
        f"(full pin written to /tmp/fw_route_matrix.py):\n"
        f"{_cell_diff(PINNED_ROUTE_MATRIX, actual)}"
    )

"""Phase 4 observability: fastWorkflow Chatbot debug mode (bead fix-kw7.5).

Server tests run against a real ObservabilityStore in tmp_path (design §6 —
no mocks for stores), seeded through the store's own write methods, with the
stdlib server started on an ephemeral port and exercised via urllib.

Covers the §3.4 invariants: token gate [R5], Host/Origin allowlist [R18],
restrictive CSP [R22], read-only API surface, artifact delivery, the
writer-health endpoint [R13], the prune/forget-channel CLI paths [R12][R21],
and the SPA packaging assertion [R23].
"""

from __future__ import annotations

import argparse
import json
import signal
import sqlite3
import threading
import time
import tomllib
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

from fastworkflow import state_paths, tracing
from fastworkflow import observability_store as obs
from fastworkflow.cli import add_run_chatbot_parser
from fastworkflow.run_chatbot import server as run_chatbot_server


# ----------------------------------------------------------------------
# Seeded store + running server fixtures
# ----------------------------------------------------------------------

TURN1 = "20260825T120000-t1"  # completed, success, conversation 1
TURN2 = "20260825T120500-t2"  # failed, conversation 1
TURN3 = "20260825T121000-t3"  # awaiting_user, conversation-less [R17]
HTML_ARTIFACT_ID = "a" * 32
TEXT_ARTIFACT_ID = "b" * 32
HTML_PAYLOAD = "<html><body><script>alert(1)</script>hi</body></html>"
TEXT_PAYLOAD = "plain text artifact payload"


def _turn_row(
    turn_key: str,
    channel_id: str,
    conversation_id,
    ordinal,
    status: str,
    success: int,
    record: dict,
    failure_reason=None,
    completed_at="2026-08-25T12:01:00+00:00",
) -> dict:
    return {
        "turn_key": turn_key,
        "channel_id": channel_id,
        "conversation_id": conversation_id,
        "ordinal": ordinal,
        "user_message": f"user message for {turn_key}",
        "refined_user_message": None,
        "entry_workflow_name": "todo_list",
        "entry_context": "TodoList",
        "status": status,
        "success": success,
        "failure_reason": failure_reason,
        "answer": f"answer for {turn_key}" if status == "completed" else "",
        "conversation_summary": None,
        "conversation_traces": None,
        "started_at": "2026-08-25T12:00:00+00:00",
        "completed_at": completed_at,
        "suspended_ms": 1500 if status == "awaiting_user" else 0,
        "continuation_of": None,
        "record_version": 1,
        "record_json": json.dumps(record),
    }


@pytest.fixture
def workflow_path(tmp_path, monkeypatch) -> str:
    monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "state"))
    wf = tmp_path / "my_workflow"
    wf.mkdir()
    return str(wf)


@pytest.fixture
def seeded_db(workflow_path) -> str:
    db_path = state_paths.observability_db(workflow_path)
    store = obs.ObservabilityStore(db_path)
    redactor = obs.Redactor()

    conv_id = store.mint_conversation_id("chan1")
    assert conv_id == 1
    store.record_conversation_label("chan1", conv_id, "Groceries", "About milk")

    record1 = {
        "turn_output": {
            "turn_key": TURN1,
            "status": "completed",
            "success": True,
            "command_outputs": [
                {
                    "command_name": "add_todo",
                    "command_response": {
                        "response": "done",
                        "success": True,
                        "artifacts": {
                            "note": "hello inline artifact",
                            "report": {
                                "__fw_artifact_ref__": HTML_ARTIFACT_ID,
                                "size": len(HTML_PAYLOAD),
                                "content_type": "text/html",
                                "content_encoding": None,
                                "error": None,
                            },
                            "log": {
                                "__fw_artifact_ref__": TEXT_ARTIFACT_ID,
                                "size": len(TEXT_PAYLOAD),
                                "content_type": "text/plain",
                                "content_encoding": None,
                                "error": None,
                            },
                        },
                    },
                }
            ],
        }
    }
    artifact_rows = [
        {
            "artifact_id": HTML_ARTIFACT_ID,
            "turn_key": TURN1,
            "channel_id": "chan1",
            "span_id": None,
            "key": "report",
            "content_type": "text/html",
            "size_bytes": len(HTML_PAYLOAD),
            "sha256": "x",
            "inline_value": HTML_PAYLOAD.encode(),
            "error": None,
        },
        {
            "artifact_id": TEXT_ARTIFACT_ID,
            "turn_key": TURN1,
            "channel_id": "chan1",
            "span_id": None,
            "key": "log",
            "content_type": "text/plain",
            "size_bytes": len(TEXT_PAYLOAD),
            "sha256": "y",
            "inline_value": TEXT_PAYLOAD.encode(),
            "error": None,
        },
    ]

    now_ns = time.time_ns()
    spans = [
        tracing.Span(
            span_id="s-root", trace_id=TURN1, name="fw.turn", kind="internal",
            channel_id="chan1", start_ns=now_ns, end_ns=now_ns + 5_000_000_000,
            status="ok", attributes={"user_message": "add milk"},
        ),
        tracing.Span(
            span_id="s-cmd", trace_id=TURN1, name="fw.command.execute",
            kind="tool", channel_id="chan1", command_name="add_todo",
            context="TodoList", parent_span_id="s-root",
            start_ns=now_ns + 1_000_000_000, end_ns=now_ns + 2_000_000_000,
            status="ok", attributes={"parameters": {"description": "milk"}},
        ),
        tracing.Span(
            span_id="s-ask", trace_id=TURN1, name="fw.ask_user",
            kind="human_wait", channel_id="chan1", parent_span_id="s-root",
            start_ns=now_ns + 2_000_000_000, end_ns=now_ns + 4_000_000_000,
            status="ok", attributes={"agent_query": "which list?"},
        ),
        # Open span on the in-progress turn (rendered honestly as open).
        tracing.Span(
            span_id="s-open", trace_id=TURN3, name="fw.ask_user",
            kind="human_wait", channel_id="chan2",
            start_ns=now_ns, end_ns=None, status="awaiting_user",
            attributes={"agent_query": "still waiting"},
        ),
    ]

    conn = store._connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        assert store.upsert_turn_row(
            conn,
            _turn_row(TURN1, "chan1", 1, 1, "completed", 1, record1),
            artifact_rows,
            redactor,
        )
        assert store.upsert_turn_row(
            conn,
            _turn_row(
                TURN2, "chan1", 1, 2, "failed", 0,
                {"turn_output": {"turn_key": TURN2, "success": False}},
                failure_reason="command exploded",
            ),
            [],
            redactor,
        )
        assert store.upsert_turn_row(
            conn,
            _turn_row(
                TURN3, "chan2", None, None, "awaiting_user", 0,
                {"turn_output": {"turn_key": TURN3, "success": False}},
                completed_at=None,
            ),
            [],
            redactor,
        )
        store.upsert_span_rows(conn, spans, redactor)
        store.set_diagnostic(
            conn,
            "writer_health",
            {"spans_dropped": 2, "records_dropped": 1, "write_errors": 3,
             "busy_retries": 0, "refused_terminal_writes": 0,
             "last_error": "disk full"},
        )
        conn.commit()
    finally:
        conn.close()
    return db_path


@pytest.fixture
def server(seeded_db, workflow_path):
    srv = run_chatbot_server.ChatbotServer(seeded_db, workflow_path=workflow_path, port=0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    thread.join(timeout=5)


def _get(server, path, token=..., headers=None):
    """GET helper returning (status, headers, body_bytes); never raises on 4xx."""
    if token is ...:
        token = server.token
    url = f"http://127.0.0.1:{server.port}{path}"
    req = urllib.request.Request(url, method="GET")
    if token is not None:
        req.add_header("Authorization", f"Bearer {token}")
    for name, value in (headers or {}).items():
        req.add_header(name, value)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as err:
        return err.code, dict(err.headers), err.read()


def _get_json(server, path, **kwargs):
    status, headers, body = _get(server, path, **kwargs)
    assert status == 200, f"{path} -> {status}: {body[:300]!r}"
    return json.loads(body)


def _row_counts(db_path: str) -> dict:
    conn = sqlite3.connect(db_path)
    try:
        return {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("conversations", "turns", "spans", "artifacts", "feedback")
        }
    finally:
        conn.close()


# ----------------------------------------------------------------------
# Access control [R5][R18]
# ----------------------------------------------------------------------


class TestAccessControl:
    def test_missing_token_is_401(self, server):
        status, _, body = _get(server, "/api/channels", token=None)
        assert status == 401
        assert b"token" in body

    def test_wrong_token_is_401(self, server):
        status, _, _ = _get(server, "/api/channels", token="not-the-token")
        assert status == 401
        status, _, _ = _get(server, "/", token="not-the-token")
        assert status == 401

    def test_token_accepted_via_query_param(self, server):
        status, _, body = _get(
            server, f"/api/channels?token={server.token}", token=None
        )
        assert status == 200
        assert json.loads(body)["channels"]

    def test_bad_host_is_403(self, server):
        status, _, body = _get(
            server, "/api/channels", headers={"Host": "evil.example.com"}
        )
        assert status == 403
        assert b"forbidden" in body

    def test_bad_origin_is_403(self, server):
        status, _, _ = _get(
            server, "/api/channels", headers={"Origin": "http://evil.example.com"}
        )
        assert status == 403

    def test_localhost_host_allowed(self, server):
        status, _, _ = _get(
            server, "/api/channels", headers={"Host": f"localhost:{server.port}"}
        )
        assert status == 200

    def test_forwarded_loopback_port_allowed(self, server):
        # WSL relays and IDE port forwards re-expose the server on a DIFFERENT
        # local port; the browser's Host names that port. Loopback hosts pass
        # regardless of port — the token stays the authentication [R18].
        for host in (
            f"127.0.0.1:{server.port + 1}",
            "localhost:9999",
            f"[::1]:{server.port}",
            "127.0.0.1",
        ):
            status, _, _ = _get(server, "/api/channels", headers={"Host": host})
            assert status == 200, host

    def test_loopback_origin_any_port_allowed_https_and_null_refused(self, server):
        status, _, _ = _get(
            server, "/api/channels", headers={"Origin": "http://localhost:9999"}
        )
        assert status == 200
        for origin in (f"https://127.0.0.1:{server.port}", "null"):
            status, _, _ = _get(
                server, "/api/channels", headers={"Origin": origin}
            )
            assert status == 403, origin

    def test_403_body_names_the_offending_host(self, server):
        status, _, body = _get(
            server, "/api/channels", headers={"Host": "evil.example.com"}
        )
        assert status == 403
        assert b"evil.example.com" in body

    def test_url_embeds_token(self, server):
        assert server.url.startswith(f"http://127.0.0.1:{server.port}/?token=")
        assert server.token in server.url


# ----------------------------------------------------------------------
# SPA page + CSP [R22]
# ----------------------------------------------------------------------


class TestPage:
    def test_index_served_with_restrictive_csp(self, server):
        status, headers, body = _get(server, "/")
        assert status == 200
        assert headers["Content-Type"].startswith("text/html")
        assert b"fastWorkflow Chatbot" in body
        csp = headers["Content-Security-Policy"]
        assert "default-src 'none'" in csp
        assert "script-src 'self'" in csp
        assert "connect-src 'self'" in csp
        assert "frame-src 'self'" in csp
        # The page's inline script is hash-sourced, not 'unsafe-inline'.
        assert "'sha256-" in csp
        assert "script-src 'self' 'unsafe-inline'" not in csp

    def test_page_never_uses_innerhtml(self, server):
        # [R22]: record-derived text renders via textContent only.
        assert b"innerHTML" not in server.index_html
        assert b'id="tabTurns"' not in server.index_html
        assert b'id="tabConvs"' not in server.index_html
        assert b'id="clearConvsBtn"' in server.index_html
        # The rail nests channel > conversation > turns, with no channel picker.
        assert b'id="channelSel"' not in server.index_html
        assert b"/api/channels" not in server.index_html
        assert b"chanGroup" in server.index_html
        assert b"renderConversationGroups" in server.index_html
        assert b"renderNestedTurns" in server.index_html
        # Empty nodes are hidden at both levels: conversations with no turns,
        # and channels left with no conversation.
        assert b"channelHasConversation" in server.index_html

    def test_unknown_path_404(self, server):
        status, _, _ = _get(server, "/nope")
        assert status == 404
        status, _, _ = _get(server, "/api/nope")
        assert status == 404


# ----------------------------------------------------------------------
# API endpoints
# ----------------------------------------------------------------------


class TestApi:
    def test_meta(self, server, seeded_db, workflow_path):
        meta = _get_json(server, "/api/meta")
        assert meta["db_path"] == seeded_db
        assert meta["workflow_name"] == "my_workflow"
        assert meta["db_size_bytes"] > 0

    def test_channels(self, server):
        assert _get_json(server, "/api/channels")["channels"] == ["chan1", "chan2"]

    def test_conversations_with_labels(self, server):
        convs = _get_json(server, "/api/conversations?channel=chan1")["conversations"]
        assert len(convs) == 1
        assert convs[0]["topic"] == "Groceries"
        assert convs[0]["summary"] == "About milk"
        assert convs[0]["last_turn_at"]  # stamped by the turn upserts

    def test_turns_by_conversation(self, server):
        turns = _get_json(
            server, "/api/turns?channel=chan1&conversation=1"
        )["turns"]
        assert [t["turn_key"] for t in turns] == [TURN2, TURN1]
        assert all("record_json" not in t for t in turns)

    def test_turns_filter_status(self, server):
        turns = _get_json(server, "/api/turns?status=awaiting_user")["turns"]
        assert [t["turn_key"] for t in turns] == [TURN3]
        assert turns[0]["conversation_id"] is None  # turns-first view row [R17]

    def test_turns_filter_success(self, server):
        turns = _get_json(server, "/api/turns?success=1")["turns"]
        assert [t["turn_key"] for t in turns] == [TURN1]
        turns = _get_json(server, "/api/turns?success=0")["turns"]
        assert {t["turn_key"] for t in turns} == {TURN2, TURN3}

    def test_turns_filter_command(self, server):
        turns = _get_json(server, "/api/turns?command=add_todo")["turns"]
        assert [t["turn_key"] for t in turns] == [TURN1]
        assert _get_json(server, "/api/turns?command=nonexistent")["turns"] == []

    def test_turn_detail_parses_record_json(self, server):
        turn = _get_json(server, f"/api/turn/{TURN1}")["turn"]
        assert "record_json" not in turn
        record = turn["record"]
        outputs = record["turn_output"]["command_outputs"]
        assert outputs[0]["command_name"] == "add_todo"
        artifacts = outputs[0]["command_response"]["artifacts"]
        assert artifacts["note"] == "hello inline artifact"
        assert artifacts["report"]["__fw_artifact_ref__"] == HTML_ARTIFACT_ID
        assert turn["failure_reason"] is None
        assert turn["status"] == "completed"

    def test_turn_detail_404(self, server):
        status, _, _ = _get(server, "/api/turn/no-such-turn")
        assert status == 404

    def test_spans_for_turn(self, server):
        spans = _get_json(server, f"/api/spans/{TURN1}")["spans"]
        assert [s["name"] for s in spans] == [
            "fw.turn", "fw.command.execute", "fw.ask_user"
        ]
        by_name = {s["name"]: s for s in spans}
        assert by_name["fw.ask_user"]["kind"] == "human_wait"
        assert by_name["fw.command.execute"]["command_name"] == "add_todo"
        # Attributes come back parsed, not as a JSON string.
        assert by_name["fw.command.execute"]["attributes"]["parameters"] == {
            "description": "milk"
        }

    def test_open_span_kept_open(self, server):
        spans = _get_json(server, f"/api/spans/{TURN3}")["spans"]
        assert spans[0]["end_ns"] is None
        assert spans[0]["status"] == "awaiting_user"

    def test_text_artifact_served(self, server):
        status, headers, body = _get(server, f"/api/artifact/{TEXT_ARTIFACT_ID}")
        assert status == 200
        assert headers["Content-Type"] == "text/plain"
        assert body == TEXT_PAYLOAD.encode()

    def test_html_artifact_sandboxed(self, server):
        status, headers, body = _get(server, f"/api/artifact/{HTML_ARTIFACT_ID}")
        assert status == 200
        assert headers["Content-Type"] == "text/html"
        assert body == HTML_PAYLOAD.encode()
        # Direct navigation to the artifact URL must be inert [R22].
        assert headers["Content-Security-Policy"] == "default-src 'none'; sandbox"
        assert headers["X-Content-Type-Options"] == "nosniff"

    def test_artifact_404(self, server):
        status, _, _ = _get(server, "/api/artifact/" + "f" * 32)
        assert status == 404

    def test_health_reflects_diagnostics(self, server):
        health = _get_json(server, "/api/health")["writer_health"]
        assert health["spans_dropped"] == 2
        assert health["records_dropped"] == 1
        assert health["write_errors"] == 3
        assert health["last_error"] == "disk full"
        assert health["updated_at"]

    def test_clear_all_conversations_requires_explicit_confirmation(
        self, server, seeded_db
    ):
        status, body = _post(server, "/api/clear_conversations", {})
        assert status == 400
        assert "confirmation required" in body["error"]
        assert _row_counts(seeded_db)["turns"] == 3

    def test_clear_all_conversations_removes_turn_observability(
        self, server, seeded_db
    ):
        status, body = _post(
            server,
            "/api/clear_conversations",
            {"confirm": "clear all conversations"},
        )
        assert status == 200
        assert body["deleted"]["turns"] == 3
        assert body["deleted"]["conversations"] == 1
        assert _row_counts(seeded_db) == {
            "conversations": 0,
            "turns": 0,
            "spans": 0,
            "artifacts": 0,
            "feedback": 0,
        }
        # Clearing data never rewinds conversation identity.
        assert obs.ObservabilityStore(seeded_db).mint_conversation_id("chan1") == 2


# ----------------------------------------------------------------------
# Read-only guarantee
# ----------------------------------------------------------------------


class TestReadOnly:
    def test_non_get_methods_rejected(self, server, seeded_db):
        for method in ("POST", "PUT", "DELETE", "PATCH"):
            req = urllib.request.Request(
                f"http://127.0.0.1:{server.port}/api/turns",
                method=method,
                data=b"{}",
                headers={"Authorization": f"Bearer {server.token}"},
            )
            try:
                with urllib.request.urlopen(req, timeout=10) as resp:
                    status = resp.status
            except urllib.error.HTTPError as err:
                status = err.code
            assert status == 405

    def test_gets_do_not_mutate_state(self, server, seeded_db):
        before = _row_counts(seeded_db)
        for path in (
            "/", "/api/meta", "/api/channels", "/api/conversations",
            "/api/turns", f"/api/turn/{TURN1}", f"/api/spans/{TURN1}",
            f"/api/artifact/{TEXT_ARTIFACT_ID}", "/api/health",
        ):
            status, _, _ = _get(server, path)
            assert status == 200
        assert _row_counts(seeded_db) == before


# ----------------------------------------------------------------------
# CLI paths: internal maintenance helpers and the browser-owned launch surface
# ----------------------------------------------------------------------


class TestCliPaths:
    def _parser(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command", required=True)
        add_run_chatbot_parser(subparsers)
        return parser

    def test_run_forget_channel(self, seeded_db):
        deleted = run_chatbot_server.run_forget_channel(seeded_db, "chan1")
        assert deleted["turns"] == 2
        assert deleted["conversations"] == 1
        assert deleted["spans"] == 3
        assert deleted["artifacts"] == 2
        counts = _row_counts(seeded_db)
        assert counts["turns"] == 1  # chan2's turn survives
        assert counts["artifacts"] == 0

    def test_run_prune_returns_counts(self, seeded_db):
        deleted = run_chatbot_server.run_prune(seeded_db)
        assert set(deleted) == {"spans", "artifacts"}
        # Everything seeded is recent; nothing crosses the retention horizon.
        assert deleted["spans"] == 0
        counts = _row_counts(seeded_db)
        assert counts["spans"] == 4

    @pytest.mark.parametrize("flag", ["--prune", "--forget-channel"])
    def test_run_chatbot_cli_rejects_removed_maintenance_flags(self, flag):
        args = ["run_chatbot", flag]
        if flag == "--forget-channel":
            args.append("chan2")
        with pytest.raises(SystemExit):
            self._parser().parse_args(args)

    def test_workflow_path_is_rejected_even_when_the_db_is_missing(self, tmp_path):
        # A missing DB is a normal cold start for the chatbot itself (the
        # auto-spawned server creates it on the first turn). Workflow selection
        # now belongs to the browser, regardless of whether that DB exists.
        wf = tmp_path / "never_ran"
        wf.mkdir()
        with pytest.raises(SystemExit):
            self._parser().parse_args(["run_chatbot", str(wf)])

    def test_run_chatbot_main_opens_the_picker(self, monkeypatch, capsys):
        monkeypatch.setattr(
            run_chatbot_server.ChatbotServer, "serve_forever", lambda self: None
        )
        monkeypatch.setattr(
            run_chatbot_server.ChatbotServer,
            "shutdown",
            lambda self: self.httpd.server_close(),
        )
        monkeypatch.setattr(signal, "signal", lambda *_args: None)
        rc = run_chatbot_server.run_chatbot_main(
            SimpleNamespace(
                server_port=8000,
                expect_encrypted_jwt=False,
            )
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "pick a workflow in the browser" in out

    def test_open_in_browser_is_a_noop_under_pytest(self, monkeypatch):
        # The pytest skip is env-based, not a CLI flag: a test that drives
        # run_chatbot_main must not pop a browser, and must not need --no-browser.
        import os
        import sys

        opened = []

        class _FakeBrowser:
            def open(self, url):
                opened.append(url)

        monkeypatch.setitem(sys.modules, "webbrowser", _FakeBrowser())
        assert "PYTEST_CURRENT_TEST" in os.environ
        run_chatbot_server._open_in_browser("http://127.0.0.1:1/?token=x")
        assert opened == []


# ----------------------------------------------------------------------
# Packaging [R23]
# ----------------------------------------------------------------------


class TestPackaging:
    def test_spa_ships_as_package_data(self):
        import importlib.resources

        resource = (
            importlib.resources.files("fastworkflow.run_chatbot") / "static" / "index.html"
        )
        assert resource.is_file()
        page = resource.read_bytes()
        assert b"fastWorkflow Chatbot" in page
        # Self-contained: no external origins anywhere in the page. Test mode
        # legitimately names loopback origins (the local FastAPI server), so
        # every http(s):// occurrence must be 127.0.0.1 or localhost [R19].
        import re

        for match in re.findall(rb"https?://[^\s\"'`<>)]*", page):
            assert match.startswith(
                (b"http://127.0.0.1", b"http://localhost")
            ), f"non-loopback origin referenced by the SPA: {match!r}"
        assert b"https://" not in page  # loopback is always plain http
        assert b"innerHTML" not in page
        assert b"startTrain" in page
        assert "Training…".encode() in page

    def test_legend_chips_sit_inline_with_labels(self):
        """Waterfall legend color chips must sit next to their span-type labels.

        Chips used to reuse `.wfBar` (`position: absolute`), which pinned them
        to the viewport instead of the legend (fix-kw7.14).
        """
        page = (
            Path(__file__).parent.parent
            / "fastworkflow"
            / "run_chatbot"
            / "static"
            / "index.html"
        ).read_text()
        assert 'el("span", "chip wfBar "' not in page
        assert 'el("span", "chip " + pair[0])' in page
        chip_rule = page.split(".legend .chip {", 1)[1].split("}", 1)[0]
        assert "position: static" in chip_rule
        # Category colors apply without requiring .wfBar, so chips keep their fill.
        assert ".cat-turn { background: var(--bar-turn); }" in page
        assert ".wfBar.cat-turn" not in page

    def test_pyproject_includes_spa(self):
        pyproject = Path(__file__).parent.parent / "pyproject.toml"
        data = tomllib.loads(pyproject.read_text())
        includes = data["tool"]["poetry"]["include"]
        assert "fastworkflow/run_chatbot/static/index.html" in includes

    def test_server_module_is_stdlib_only(self):
        # [R23]: debug mode must work on a base install (no [server] extra).
        import ast

        tree = ast.parse(Path(run_chatbot_server.__file__).read_text())
        imported_roots: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".")[0])
        for forbidden in ("fastapi", "uvicorn", "starlette", "flask", "aiohttp"):
            assert forbidden not in imported_roots, (
                f"chatbot server imports {forbidden}"
            )


# ----------------------------------------------------------------------
# Read-only viewer discipline: the debug layer never creates or writes the
# DB it inspects (epic review follow-up; [R12] + module invariant).
# ----------------------------------------------------------------------


class TestReadOnlyViewer:
    def test_viewer_does_not_create_a_missing_db(self, workflow_path):
        db_path = state_paths.observability_db(workflow_path)
        assert not Path(db_path).exists()
        srv = run_chatbot_server.ChatbotServer(
            db_path, workflow_path=workflow_path, port=0
        )
        thread = threading.Thread(target=srv.serve_forever, daemon=True)
        thread.start()
        try:
            meta = _get_json(srv, "/api/meta")
            assert meta["db_available"] is False
            assert _get_json(srv, "/api/turns")["turns"] == []
            assert _get_json(srv, "/api/conversations")["conversations"] == []
            assert _get_json(srv, "/api/health")["db_available"] is False
        finally:
            srv.shutdown()
            thread.join(timeout=5)
        # The whole point: inspecting must not have created the file.
        assert not Path(db_path).exists()

    def test_viewer_opens_a_read_only_snapshot(self, seeded_db):
        import os as _os

        # Checkpoint so the snapshot has no -wal sidecar, then drop write perms:
        # a post-mortem copy the developer does not own must still open.
        conn = sqlite3.connect(seeded_db)
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.close()
        _os.chmod(seeded_db, 0o400)
        try:
            srv = run_chatbot_server.ChatbotServer(seeded_db, port=0)
            thread = threading.Thread(target=srv.serve_forever, daemon=True)
            thread.start()
            try:
                turns = _get_json(srv, "/api/turns")["turns"]
                assert len(turns) == 3
                assert _get_json(srv, "/api/meta")["db_available"] is True
            finally:
                srv.shutdown()
                thread.join(timeout=5)
        finally:
            _os.chmod(seeded_db, 0o600)


class TestServerSideContextFilter:
    def test_context_filter_is_applied_by_the_server(self, server):
        # Seeded turns carry entry_context "TodoList"; the match is a
        # case-insensitive substring, applied in SQL so it reaches rows older
        # than one page.
        turns = _get_json(server, "/api/turns?context=todolist")["turns"]
        assert len(turns) == 3
        assert _get_json(server, "/api/turns?context=nomatch")["turns"] == []
        # LIKE metacharacters in the filter are literals, not wildcards.
        assert _get_json(server, "/api/turns?context=%25")["turns"] == []


class TestForgetChannelLegacyErasure:
    def test_forget_channel_deletes_legacy_conversation_db(
        self, seeded_db, workflow_path
    ):
        # Ruling C1: during the Phase-A dual-write window, erasure must also
        # remove the legacy per-channel conversations/<channel_id>.sqlite3.
        legacy_dir = Path(state_paths.conversations_dir(workflow_path))
        legacy_dir.mkdir(parents=True, exist_ok=True)
        legacy = legacy_dir / "chan1.sqlite3"
        legacy.write_bytes(b"legacy payload")
        (legacy_dir / "chan1.sqlite3-wal").write_bytes(b"wal")
        deleted = run_chatbot_server.run_forget_channel(
            seeded_db, "chan1", workflow_path
        )
        assert deleted["legacy_conversation_db_files"] == 2
        assert not legacy.exists()
        assert not (legacy_dir / "chan1.sqlite3-wal").exists()

    def test_forget_channel_refuses_path_traversal_channel_ids(
        self, seeded_db, workflow_path, tmp_path
    ):
        outside = tmp_path / "outside.sqlite3"
        outside.write_bytes(b"do not delete")
        deleted = run_chatbot_server.run_forget_channel(
            seeded_db, "../../outside", workflow_path
        )
        assert "legacy_conversation_db_files" not in deleted
        assert outside.exists()


# ----------------------------------------------------------------------
# Control plane: /api/session, workflow picker, POST /api/select_workflow
# ----------------------------------------------------------------------


def _post(server, path, body, token=...):
    if token is ...:
        token = server.token
    req = urllib.request.Request(
        f"http://127.0.0.1:{server.port}{path}",
        method="POST",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    if token is not None:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read() or b"{}")


class TestControlPlane:
    def test_session_reports_managed_identity(self, server):
        s = _get_json(server, "/api/session")["session"]
        # Single-user tool: the channel is chatbot-managed, never typed, and
        # fixed across launches so restarts share one history.
        assert s["channel_id"] == "chatbot"
        assert s["user_id"] == "developer"
        assert s["workflow_name"] == "my_workflow"
        assert s["server_running"] is False  # fixtures never spawn

    def test_workflow_candidates_include_bundled_examples(self, server):
        import fastworkflow

        wfs = _get_json(server, "/api/workflows")["workflows"]
        bundled = str(
            Path(fastworkflow.__file__).resolve().parent / "examples" / "hello_world"
        )
        hello = next(w for w in wfs if w["path"] == bundled)
        assert hello["source"] == "examples"
        assert hello["name"] == "hello_world"
        assert hello["trainable"] is False
        assert "training" in hello
        assert "rel" in hello
        assert Path(hello["path"]).is_dir()

    def test_nested_local_workflows_carry_rel_paths(self, tmp_path, monkeypatch):
        cwd = tmp_path / "proj"
        (cwd / "top_wf" / "_commands").mkdir(parents=True)
        (cwd / "apps" / "team" / "deep_wf" / "_commands").mkdir(parents=True)
        monkeypatch.chdir(cwd)
        wfs = run_chatbot_server.list_workflow_candidates()
        local = {w["name"]: w for w in wfs if w["source"] == "local"}
        assert local["top_wf"]["rel"] == "top_wf"
        assert local["deep_wf"]["rel"] == "apps/team/deep_wf"

    def test_parent_of_nested_workflow_is_omitted(self, tmp_path, monkeypatch):
        cwd = tmp_path / "proj"
        (cwd / "pkg" / "_commands").mkdir(parents=True)
        (cwd / "pkg" / "apps" / "child_wf" / "_commands").mkdir(parents=True)
        monkeypatch.chdir(cwd)
        wfs = run_chatbot_server.list_workflow_candidates()
        local = {w["name"]: w for w in wfs if w["source"] == "local"}
        assert "child_wf" in local
        assert "pkg" not in local
        assert local["child_wf"]["rel"] == "pkg/apps/child_wf"

    def test_rel_under_rejects_paths_outside_root(self, tmp_path):
        root = tmp_path / "root"
        root.mkdir()
        inside = root / "apps" / "wf"
        assert run_chatbot_server._rel_under(str(inside), str(root)) == "apps/wf"
        assert run_chatbot_server._rel_under(str(root), str(root)) == ""
        assert run_chatbot_server._rel_under(str(tmp_path), str(root)) == ""

    def test_browse_lists_directories_flagging_workflows(self, server):
        import urllib.parse

        import fastworkflow

        examples = str(
            Path(fastworkflow.__file__).resolve().parent / "examples"
        )
        data = _get_json(
            server, "/api/browse?dir=" + urllib.parse.quote(examples)
        )
        assert data["dir"] == examples
        by_name = {e["name"]: e for e in data["entries"]}
        assert by_name["hello_world"]["is_workflow"] is True

    def test_select_workflow_requires_token(self, server, tmp_path):
        status, body = _post(
            server, "/api/select_workflow", {"path": str(tmp_path)}, token=None
        )
        assert status == 401

    def test_select_workflow_switches_the_active_db(self, server, tmp_path):
        other = tmp_path / "other_wf"
        (other / "_commands").mkdir(parents=True)
        status, body = _post(
            server, "/api/select_workflow", {"path": str(other)}
        )
        assert status == 200
        s = body["session"]
        assert s["workflow_path"] == str(other)
        assert s["server_running"] is False  # no_server fixtures
        assert server.db_path.endswith("observability.sqlite3")
        assert "other_wf" in server.db_path

    def test_select_workflow_rejects_non_workflow_dirs(self, server, tmp_path):
        plain = tmp_path / "not_a_workflow"
        plain.mkdir()
        status, body = _post(
            server, "/api/select_workflow", {"path": str(plain)}
        )
        assert status == 400
        assert "_commands" in body["error"]

    def test_missing_env_files_are_configured_from_templates(
        self, workflow_path, tmp_path, monkeypatch
    ):
        wf = tmp_path / "needs_env"
        (wf / "_commands").mkdir(parents=True)
        monkeypatch.setattr(
            "fastworkflow.run_chatbot.launcher.missing_server_packages",
            lambda: ["fastapi"],
        )
        srv = run_chatbot_server.ChatbotServer(
            workflow_path=str(wf),
            port=0,
            spawn_options={"no_server": False},
        )
        thread = threading.Thread(target=srv.serve_forever, daemon=True)
        thread.start()
        try:
            session = srv.activate_workflow(str(wf))
            assert session["env_setup_required"] is True
            assert session["server_running"] is False

            status, body = _post(
                srv, "/api/configure_env", {"create_from_templates": True}
            )
            assert status == 200
            assert body["session"]["env_setup_required"] is False
            assert (wf / "fastworkflow.env").is_file()
            assert (wf / "fastworkflow.passwords.env").is_file()
            assert "LLM_AGENT" in (wf / "fastworkflow.env").read_text()
        finally:
            srv.shutdown()
            thread.join(timeout=5)

    def test_selected_env_file_contents_are_copied_locally(self, tmp_path):
        wf = tmp_path / "uploaded_env"
        (wf / "_commands").mkdir(parents=True)
        srv = run_chatbot_server.ChatbotServer(
            workflow_path=str(wf),
            port=0,
            spawn_options={"no_server": True},
        )
        thread = threading.Thread(target=srv.serve_forever, daemon=True)
        thread.start()
        try:
            status, _body = _post(
                srv,
                "/api/configure_env",
                {
                    "env_content": "LLM_AGENT=test/model\n",
                    "passwords_content": "LITELLM_API_KEY_AGENT=test-secret\n",
                },
            )
            assert status == 200
            env_file = wf / "fastworkflow.env"
            passwords_file = wf / "fastworkflow.passwords.env"
            assert env_file.read_text() == "LLM_AGENT=test/model\n"
            assert (
                passwords_file.read_text()
                == "LITELLM_API_KEY_AGENT=test-secret\n"
            )
            assert env_file.stat().st_mode & 0o777 == 0o600
            assert passwords_file.stat().st_mode & 0o777 == 0o600
        finally:
            srv.shutdown()
            thread.join(timeout=5)

    def test_other_posts_are_still_405(self, server):
        status, _ = _post(server, "/api/turns", {})
        assert status == 405
        status, _ = _post(server, "/api/channels", {})
        assert status == 405

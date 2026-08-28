"""Phase 5 observability: fastWorkflow Chatbot test mode (bead fix-kw7.6).

Covers the design's §3.4 test-mode slice:

- [R24] additive ``/initialize`` request fields: ``startup_command`` /
  ``startup_action`` (mutually exclusive, 400) and the NEW per-session
  ``context`` override; absent fields leave existing callers unchanged.
- [R19] spawned-server posture: the spawn decision always pins
  ``--host 127.0.0.1``, pins CORS to exactly the Chatbot origin, refuses
  unsigned-JWT mode without an explicit flag, and requires the [server]
  extra — unit-tested via the pure ``plan_server_spawn`` function (no
  subprocesses are spawned in tests).
- ``--cors_origin`` on run_fastapi_mcp pins the CORS middleware to exactly
  that origin (verified against the real app via preflight).
- The SPA gains a Test tab and still references no non-loopback origins.

No mocks (testing_rules.mdc): the FastAPI tests drive the real app over
tests/hello_world_workflow with real stores under a per-test temp state root
(fastapi_hermetic pattern, as in tests/test_fastapi_turns_serve.py).
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import sys
import threading
import uuid
import urllib.error
import urllib.request

import pytest
from fastapi.testclient import TestClient

from fastworkflow import observability_store as obs
from fastworkflow import state_paths
from fastworkflow.cli import add_run_chatbot_parser
from fastworkflow.run_chatbot import launcher


@pytest.fixture
def workflow_path():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(project_root, "tests", "hello_world_workflow")
    if not os.path.isdir(path):
        pytest.skip(f"hello_world_workflow not found at {path}")
    return path


@pytest.fixture
def env_files():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_file = os.path.join(project_root, "env", ".env")
    passwords_file = os.path.join(project_root, "passwords", ".env")
    if not os.path.isfile(env_file) or not os.path.isfile(passwords_file):
        pytest.skip("env files missing for FastAPI tests")
    return env_file, passwords_file


def _load_app_module(workflow_path, env_files, tmp_path, extra_argv=()):
    env_file, passwords_file = env_files
    sys.argv = [
        "pytest",
        "--workflow_path", workflow_path,
        "--env_file_path", env_file,
        "--passwords_file_path", passwords_file,
        *extra_argv,
    ]
    import fastworkflow.run_fastapi_mcp.__main__ as main

    importlib.reload(main)
    from tests.fastapi_hermetic import init_fastapi_hermetic_env

    previous_env = init_fastapi_hermetic_env(
        env_file, passwords_file, tmp_path / "workflow_contexts"
    )
    return main, previous_env


@pytest.fixture
def app_module(workflow_path, env_files, tmp_path):
    main, previous_env = _load_app_module(workflow_path, env_files, tmp_path)
    from tests.fastapi_hermetic import restore_fastapi_env

    try:
        yield main
    finally:
        restore_fastapi_env(previous_env)


CHATBOT_ORIGIN = "http://127.0.0.1:45871"


@pytest.fixture
def app_module_cors(workflow_path, env_files, tmp_path):
    """The real app built with --cors_origin pinned to the Chatbot origin."""
    main, previous_env = _load_app_module(
        workflow_path, env_files, tmp_path,
        extra_argv=("--cors_origin", CHATBOT_ORIGIN),
    )
    from tests.fastapi_hermetic import restore_fastapi_env

    try:
        yield main
    finally:
        restore_fastapi_env(previous_env)


def _channel(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _count_calls(call_log: str) -> int:
    if not os.path.isfile(call_log):
        return 0
    with open(call_log) as fh:
        return sum(bool(line.strip()) for line in fh)


# ---------------------------------------------------------------------------
# [R24] /initialize additive fields
# ---------------------------------------------------------------------------


class TestInitializeAdditiveFields:
    def test_startup_command_executes_per_session(
        self, app_module, tmp_path, monkeypatch
    ):
        """A request-supplied startup_command runs as the session's first turn."""
        call_log = str(tmp_path / "calls.log")
        monkeypatch.setenv("FW_TEST_ADD_CALL_LOG", call_log)

        channel_id = _channel("chatbot_tm_cmd")
        with TestClient(app_module.app) as client:
            resp = client.post("/initialize", json={
                "channel_id": channel_id,
                "user_id": "chatbot_user",
                "startup_command": "add_two_numbers first_num=5 second_num=3",
                "timeout_seconds": 60,
            })
            assert resp.status_code == 200
            data = resp.json()
            assert data["access_token"]
            out = data["startup_output"]
            assert out is not None
            assert out["status"] == "completed"
            assert out["success"] is True
            assert "8" in out["answer"]
            # [R9] the logical key is reported — this is the key the Chatbot
            # test-mode "view trace" link jumps to in debug mode.
            assert data["startup_logical_turn_key"] == out["turn_key"]
        assert _count_calls(call_log) == 1

    def test_context_field_overrides_per_session(self, app_module):
        """The NEW context field reaches the created workflow's context."""
        channel_id = _channel("chatbot_tm_ctx")
        with TestClient(app_module.app) as client:
            resp = client.post("/initialize", json={
                "channel_id": channel_id,
                "user_id": "chatbot_user",
                "context": {"chatbot_ctx_probe": "present"},
            })
            assert resp.status_code == 200
            runtime = app_module.session_manager._sessions[channel_id]
            wf_context = runtime.execution_context.app_workflow.context
            assert wf_context.get("chatbot_ctx_probe") == "present"

    def test_both_startup_fields_is_400(self, app_module):
        with TestClient(app_module.app) as client:
            resp = client.post("/initialize", json={
                "channel_id": _channel("chatbot_tm_both"),
                "user_id": "chatbot_user",
                "startup_command": "add_two_numbers first_num=1 second_num=2",
                "startup_action": {
                    "command_name": "add_two_numbers",
                    "parameters": {"first_num": 1, "second_num": 2},
                },
            })
            assert resp.status_code == 400
            assert "both startup_command and startup_action" in (
                resp.json()["detail"].lower()
            )

    def test_absent_fields_leave_existing_callers_unchanged(
        self, app_module, tmp_path, monkeypatch
    ):
        """Strictly additive: a plain /initialize behaves exactly as before."""
        call_log = str(tmp_path / "calls.log")
        monkeypatch.setenv("FW_TEST_ADD_CALL_LOG", call_log)

        channel_id = _channel("chatbot_tm_plain")
        with TestClient(app_module.app) as client:
            resp = client.post("/initialize", json={"channel_id": channel_id})
            assert resp.status_code == 200
            data = resp.json()
            assert data["access_token"] and data["refresh_token"]
            assert data["startup_output"] is None
            assert data["startup_turn_key"] is None
            # No per-session context was injected.
            runtime = app_module.session_manager._sessions[channel_id]
            wf_context = runtime.execution_context.app_workflow.context
            assert "chatbot_ctx_probe" not in wf_context
        assert _count_calls(call_log) == 0


# ---------------------------------------------------------------------------
# [R19] --cors_origin pins the middleware to exactly that origin
# ---------------------------------------------------------------------------


def _cors_kwargs(app):
    from fastapi.middleware.cors import CORSMiddleware

    for mw in app.user_middleware:
        if mw.cls is CORSMiddleware:
            return mw.kwargs
    pytest.fail("CORSMiddleware not installed on the app")


class TestCorsOriginFlag:
    def test_middleware_pinned_to_exactly_the_chatbot_origin(self, app_module_cors):
        kwargs = _cors_kwargs(app_module_cors.app)
        assert kwargs["allow_origins"] == [CHATBOT_ORIGIN]
        assert "*" not in kwargs["allow_origins"]

    def test_preflight_allows_only_the_pinned_origin(self, app_module_cors):
        with TestClient(app_module_cors.app) as client:
            ok = client.options("/initialize", headers={
                "Origin": CHATBOT_ORIGIN,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "authorization,content-type",
            })
            assert ok.status_code == 200
            assert ok.headers["access-control-allow-origin"] == CHATBOT_ORIGIN

            denied = client.options("/initialize", headers={
                "Origin": "http://evil.example.com",
                "Access-Control-Request-Method": "POST",
            })
            assert denied.status_code == 400
            assert "access-control-allow-origin" not in denied.headers

    def test_default_remains_wide_open_dev_posture(self, app_module):
        # Unchanged behavior without the flag (documented dev posture).
        kwargs = _cors_kwargs(app_module.app)
        assert kwargs["allow_origins"] == ["*"]


# ---------------------------------------------------------------------------
# Chatbot test-mode static assets
# ---------------------------------------------------------------------------


def _spa_bytes() -> bytes:
    import importlib.resources

    resource = (
        importlib.resources.files("fastworkflow.run_chatbot") / "static" / "index.html"
    )
    return resource.read_bytes()


class TestChatbotTestModeAssets:
    def test_test_tab_exists(self):
        page = _spa_bytes()
        assert b'id="modeTest"' in page
        assert b'id="testMain"' in page
        # The chat pane drives the documented server surface.
        for endpoint in (b"/initialize", b"/invoke_agent", b"/invoke_assistant"):
            assert endpoint in page, f"SPA test mode never calls {endpoint!r}"
        # "/" prefix routes to deterministic execution; still no innerHTML [R22].
        assert b"invoke_assistant" in page
        assert b"innerHTML" not in page
        assert b'id="tabTurns"' not in page
        assert b'id="tabConvs"' not in page
        assert b'id="channelSel"' not in page
        assert b"renderNestedTurns" in page
        assert b"renderConversationGroups" in page
        assert b"workflowTree" in page
        assert b"wfFolder" in page
        assert b"appendMarkdown" in page
        assert b'class="primary newConv"' in page
        assert b"tmLoadLatestConversation" in page
        assert b"tmLatestConversation" in page
        assert b"/api/conversations?channel=" in page
        assert b"Prefer the highest conversation_id that actually has turns" in page
        assert b"/activate_conversation" in page
        assert b"tmClearLog" in page
        assert b'id="clearConvsBtn"' in page
        assert b'id="envFilePick"' in page
        assert b'id="passwordsFilePick"' in page
        assert b'id="createEnvBtn"' in page
        # LLM drill-down: request, response, module result, and reasoning.
        assert b"fw.llm.call" in page
        assert b"LLM input (messages)" in page
        assert b"LLM output (raw)" in page
        assert b"module output (parsed)" in page
        assert b"reasoning" in page

    def test_chat_sends_no_startup_or_context_fields(self):
        """Chat is interactive: the session is driven by what the user types.

        A startup command or a per-session context JSON is a launch-time
        decision — it belongs to `run_fastapi_mcp` flags or a programmatic
        `/initialize` call, not to a chat box. The inputs were removed; this
        keeps them from drifting back in.
        """
        page = _spa_bytes()
        for widget in (b'id="tStartupCmd"', b'id="tContext"'):
            assert widget not in page, f"chat UI re-exposed {widget!r}"
        # It must not populate those request fields either.
        for field in (b"body.startup_command", b"body.startup_action", b"body.context"):
            assert field not in page, f"chat UI sends {field!r} on /initialize"

    def test_no_external_origins_referenced(self):
        # Test mode may name loopback origins (the local FastAPI server) and
        # nothing else [R19].
        page = _spa_bytes()
        for match in re.findall(rb"https?://[^\s\"'`<>)]*", page):
            assert match.startswith((b"http://127.0.0.1", b"http://localhost")), (
                f"non-loopback origin referenced by the SPA: {match!r}"
            )
        assert b"https://" not in page

    def test_page_csp_connects_to_loopback_only(self, tmp_path, monkeypatch):
        from fastworkflow import state_paths
        from fastworkflow.run_chatbot import server as run_chatbot_server

        monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "state"))
        wf = tmp_path / "wf"
        wf.mkdir()
        srv = run_chatbot_server.ChatbotServer(
            state_paths.observability_db(str(wf)), workflow_path=str(wf), port=0
        )
        try:
            assert (
                "connect-src 'self' http://127.0.0.1:* http://localhost:*"
                in srv.page_csp
            )
            # No non-loopback connect targets and no wildcard scheme sources.
            assert "https:" not in srv.page_csp
            assert "http://*" not in srv.page_csp
        finally:
            srv.httpd.server_close()


# ---------------------------------------------------------------------------
# [R19] spawn decision (pure function; no subprocesses in tests)
# ---------------------------------------------------------------------------


class TestSpawnDecision:
    KW = dict(
        workflow_path="/wf",
        env_file_path="/env/.env",
        passwords_file_path="/passwords/.env",
        chatbot_origin="http://127.0.0.1:39131",
        server_port=8000,
    )

    def test_unsigned_jwt_refused_without_explicit_flag(self):
        plan = launcher.plan_server_spawn(**self.KW, missing_packages=[])
        assert plan.ok is False
        assert "--allow-unsigned-jwt" in plan.reason
        assert "unsigned" in plan.reason.lower()
        with pytest.raises(ValueError):
            launcher.spawn_server(plan)

    def test_unsigned_jwt_allowed_with_explicit_flag(self):
        plan = launcher.plan_server_spawn(
            **self.KW, allow_unsigned_jwt=True, missing_packages=[]
        )
        assert plan.ok is True
        assert "--expect_encrypted_jwt" not in plan.cmd

    def test_signed_jwt_needs_no_allow_flag(self):
        plan = launcher.plan_server_spawn(
            **self.KW, expect_encrypted_jwt=True, missing_packages=[]
        )
        assert plan.ok is True
        assert "--expect_encrypted_jwt" in plan.cmd

    def test_spawn_always_pins_loopback_host(self):
        plan = launcher.plan_server_spawn(
            **self.KW, allow_unsigned_jwt=True, missing_packages=[]
        )
        host = plan.cmd[plan.cmd.index("--host") + 1]
        assert host == "127.0.0.1"
        assert "0.0.0.0" not in plan.cmd
        # There is no host parameter to override [R19]: a wider bind is a
        # command-line decision on run_fastapi_mcp itself, never Chatbot's.
        import inspect

        assert "host" not in inspect.signature(launcher.plan_server_spawn).parameters

    def test_cors_pinned_to_loopback_origins(self):
        # [R19 as amended alongside R18]: loopback-only CORS instead of
        # exact-origin pinning — port forwarders (WSL/IDE) serve the chatbot
        # page from a different local port, and its Origin must still pass.
        # Never a wildcard, never a routable origin.
        plan = launcher.plan_server_spawn(
            **self.KW, allow_unsigned_jwt=True, missing_packages=[]
        )
        assert "--cors_loopback_only" in plan.cmd
        assert "--cors_origin" not in plan.cmd
        assert "*" not in plan.cmd

    def test_missing_server_extra_is_a_friendly_refusal(self):
        plan = launcher.plan_server_spawn(
            **self.KW, allow_unsigned_jwt=True,
            missing_packages=["fastapi", "uvicorn"],
        )
        assert plan.ok is False
        assert 'fastworkflow[server]' in plan.reason
        assert "fastapi" in plan.reason
        # Debug mode is explicitly unaffected.
        assert "debug mode" in plan.reason.lower()

    def _chatbot(self, tmp_path, monkeypatch, spawn_options):
        from fastworkflow.run_chatbot import server as run_chatbot_server

        monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "state"))
        wf = tmp_path / "wf"
        (wf / "_commands").mkdir(parents=True)
        env_file = wf / "fastworkflow.env"
        passwords_file = wf / "fastworkflow.passwords.env"
        env_file.write_text("SPEEDDICT_FOLDERNAME=___workflow_contexts\n")
        passwords_file.write_text("LITELLM_API_KEY_AGENT=test-placeholder\n")
        resolved_options = dict(spawn_options)
        resolved_options["env_file_path"] = str(env_file)
        resolved_options["passwords_file_path"] = str(passwords_file)
        srv = run_chatbot_server.ChatbotServer(
            port=0, spawn_options=resolved_options
        )
        return srv, str(wf)

    def test_auto_spawn_defaults_to_unsigned_dev_jwt(self, tmp_path, monkeypatch):
        """Owner decision amending [R19]: the AUTO-spawned loopback server runs
        with unsigned dev JWTs by default (the chatbot mints its own tokens via
        /initialize); --expect-encrypted-jwt restores signed mode. The plan is
        captured through the real planner; a scripted stand-in replaces only
        the subprocess itself."""

        captured = {}
        real_plan = launcher.plan_server_spawn

        def capture_plan(**kwargs):
            captured.update(kwargs)
            return real_plan(**kwargs)

        class _FakeProc:
            pid = 4242

            def poll(self):
                return None

        monkeypatch.setattr(launcher, "plan_server_spawn", capture_plan)
        monkeypatch.setattr(launcher, "spawn_server", lambda plan: _FakeProc())
        monkeypatch.setattr(
            "fastworkflow.run_chatbot.server.time.sleep", lambda s: None
        )
        srv, wf = self._chatbot(
            tmp_path, monkeypatch,
            {"no_server": False, "server_port": 8123,
             "env_file_path": "/env/.env", "passwords_file_path": "/passwords/.env"},
        )
        try:
            session = srv.activate_workflow(wf)
        finally:
            srv.httpd.server_close()
        assert captured["allow_unsigned_jwt"] is True
        assert captured["expect_encrypted_jwt"] is False
        assert session["server_running"] is True
        assert session["server_url"] == "http://127.0.0.1:8123"
        assert session["jwt_mode"] == "unsigned"
        # Fixed, not per-launch: restarts stay in one channel so the debug
        # rail groups a developer's history together instead of by process.
        assert session["channel_id"] == "chatbot"

    def test_expect_encrypted_jwt_disables_unsigned_default(
        self, tmp_path, monkeypatch
    ):
        captured = {}
        real_plan = launcher.plan_server_spawn

        def capture_plan(**kwargs):
            captured.update(kwargs)
            return real_plan(**kwargs)

        class _FakeProc:
            pid = 4242

            def poll(self):
                return None

        monkeypatch.setattr(launcher, "plan_server_spawn", capture_plan)
        monkeypatch.setattr(launcher, "spawn_server", lambda plan: _FakeProc())
        monkeypatch.setattr(
            "fastworkflow.run_chatbot.server.time.sleep", lambda s: None
        )
        srv, wf = self._chatbot(
            tmp_path, monkeypatch,
            {"no_server": False, "expect_encrypted_jwt": True,
             "env_file_path": "/env/.env", "passwords_file_path": "/passwords/.env"},
        )
        try:
            session = srv.activate_workflow(wf)
        finally:
            srv.httpd.server_close()
        assert captured["allow_unsigned_jwt"] is False
        assert captured["expect_encrypted_jwt"] is True
        assert session["jwt_mode"] == "signed"

    def test_missing_server_extra_degrades_to_viewer(self, tmp_path, monkeypatch):
        """A refused spawn must not kill the chatbot: it stays a trace viewer
        and the reason is visible on the session payload."""
        monkeypatch.setattr(
            launcher, "missing_server_packages", lambda: ["fastapi", "uvicorn"]
        )
        srv, wf = self._chatbot(tmp_path, monkeypatch, {"no_server": False})
        try:
            session = srv.activate_workflow(wf)
        finally:
            srv.httpd.server_close()
        assert session["server_running"] is False
        assert session["server_url"] is None
        assert "fastworkflow[server]" in session["spawn_error"]

    def test_named_server_port_skips_spawn_and_exposes_the_url(
        self, tmp_path, monkeypatch
    ):
        spawned = []
        monkeypatch.setattr(
            launcher, "spawn_server", lambda plan: spawned.append(plan)
        )
        srv, wf = self._chatbot(
            tmp_path, monkeypatch, {"no_server": True, "server_port": 9000}
        )
        try:
            session = srv.activate_workflow(wf)
        finally:
            srv.httpd.server_close()
        assert spawned == []
        assert session["server_running"] is False
        assert session["server_url"] == "http://127.0.0.1:9000"

    def test_busy_server_port_moves_to_a_free_one(self, tmp_path, monkeypatch):
        """A squatter on the preferred port (an old server, another chatbot,
        anything) must never be mistaken for our spawn: the chatbot binds a
        free port instead and says so on the session payload."""
        import socket

        captured = {}
        real_plan = launcher.plan_server_spawn

        def capture_plan(**kwargs):
            captured.update(kwargs)
            return real_plan(**kwargs)

        class _FakeProc:
            pid = 4242

            def poll(self):
                return None

        monkeypatch.setattr(launcher, "plan_server_spawn", capture_plan)
        monkeypatch.setattr(launcher, "spawn_server", lambda plan: _FakeProc())
        monkeypatch.setattr(
            "fastworkflow.run_chatbot.server.time.sleep", lambda s: None
        )
        squatter = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        squatter.bind(("127.0.0.1", 0))
        squatter.listen(1)
        busy_port = squatter.getsockname()[1]
        try:
            srv, wf = self._chatbot(
                tmp_path, monkeypatch,
                {"no_server": False, "server_port": busy_port,
                 "env_file_path": "/env/.env",
                 "passwords_file_path": "/passwords/.env"},
            )
            try:
                session = srv.activate_workflow(wf)
            finally:
                srv.httpd.server_close()
        finally:
            squatter.close()
        assert captured["server_port"] != busy_port
        assert session["server_url"] == f"http://127.0.0.1:{captured['server_port']}"
        assert str(busy_port) in session["server_note"]

    def test_session_reports_a_dead_server_honestly(self, tmp_path, monkeypatch):
        class _DeadProc:
            pid = 4242
            returncode = None
            _polls = 0

            def poll(self):
                # Alive through the spawn liveness check and the activation's
                # own session payload (two polls), dead afterwards.
                _DeadProc._polls += 1
                if _DeadProc._polls > 2:
                    self.returncode = 3
                    return 3
                return None

        monkeypatch.setattr(launcher, "spawn_server", lambda plan: _DeadProc())
        monkeypatch.setattr(
            "fastworkflow.run_chatbot.server.time.sleep", lambda s: None
        )
        srv, wf = self._chatbot(
            tmp_path, monkeypatch,
            {"no_server": False, "env_file_path": "/env/.env",
             "passwords_file_path": "/passwords/.env"},
        )
        try:
            first = srv.activate_workflow(wf)
            assert first["server_running"] is True
            second = srv.session_payload()
        finally:
            srv.httpd.server_close()
        assert second["server_running"] is False
        assert second["server_url"] is None
        assert second["server_exit_code"] == 3


class TestRunChatbotCliSurface:
    def _parser(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command", required=True)
        add_run_chatbot_parser(subparsers)
        return parser

    def test_run_chatbot_takes_no_workflow_or_env_paths(self):
        args = self._parser().parse_args(["run_chatbot"])
        assert args.command == "run_chatbot"
        assert not hasattr(args, "workflow_path")
        assert not hasattr(args, "env_file_path")
        assert not hasattr(args, "passwords_file_path")
        assert not hasattr(args, "port")
        assert not hasattr(args, "no_browser")
        assert not hasattr(args, "no_server")
        assert args.server_port is None

    def test_server_port_implies_no_spawn(self):
        from fastworkflow.run_chatbot import server as run_chatbot_server

        args = self._parser().parse_args(["run_chatbot", "--server-port", "9000"])
        opts = run_chatbot_server.spawn_options_from_cli_args(args)
        assert opts["no_server"] is True
        assert opts["server_port"] == 9000

        args = self._parser().parse_args(["run_chatbot"])
        opts = run_chatbot_server.spawn_options_from_cli_args(args)
        assert opts["no_server"] is False
        assert opts["server_port"] == run_chatbot_server.PREFERRED_SPAWN_PORT

    @pytest.mark.parametrize(
        "extra",
        [
            ["/tmp/workflow"],
            ["--env_file_path", "/tmp/fastworkflow.env"],
            ["--passwords_file_path", "/tmp/fastworkflow.passwords.env"],
            ["--port", "8901"],
            ["--no-browser"],
            ["--no-server"],
        ],
    )
    def test_removed_path_parameters_are_rejected(self, extra):
        with pytest.raises(SystemExit):
            self._parser().parse_args(["run_chatbot", *extra])


def _chatbot_turn_row(turn_key, conversation_id, ordinal, user_message, answer):
    return {
        "turn_key": turn_key,
        "channel_id": "chatbot",
        "conversation_id": conversation_id,
        "ordinal": ordinal,
        "user_message": user_message,
        "refined_user_message": None,
        "entry_workflow_name": "todo_list",
        "entry_context": "TodoList",
        "status": "completed",
        "success": 1,
        "failure_reason": None,
        "answer": answer,
        "conversation_summary": None,
        "conversation_traces": None,
        "started_at": "2026-08-26T12:00:00+00:00",
        "completed_at": "2026-08-26T12:00:01+00:00",
        "suspended_ms": 0,
        "continuation_of": None,
        "record_version": 1,
        "record_json": json.dumps({"turn_output": {"turn_key": turn_key, "success": True}}),
    }


def _seed_chatbot_channel_history(db_path: str) -> None:
    """Conv 1 has turns; conv 2 is a reserved empty conversation (new chat).

    Turns on conv 1 are written AFTER minting conv 2 so last_turn_at ranks
    conv 1 first — the trap if the chat pane took conversations[0] instead of
    the highest conversation_id (the id /initialize restores).
    """
    store = obs.ObservabilityStore(db_path)
    redactor = obs.Redactor()
    first = store.mint_conversation_id("chatbot")
    store.record_conversation_label("chatbot", first, "Groceries", "About milk")
    second = store.mint_conversation_id("chatbot")
    assert (first, second) == (1, 2)
    conn = store._connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        assert store.upsert_turn_row(
            conn,
            _chatbot_turn_row("20260826T120000-t1", first, 1, "add milk", "added milk"),
            [],
            redactor,
        )
        assert store.upsert_turn_row(
            conn,
            _chatbot_turn_row("20260826T120100-t2", first, 2, "add eggs", "added eggs"),
            [],
            redactor,
        )
        conn.commit()
    finally:
        conn.close()


def _chatbot_get_json(server, path):
    url = f"http://127.0.0.1:{server.port}{path}"
    req = urllib.request.Request(url, method="GET")
    req.add_header("Authorization", f"Bearer {server.token}")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read()
            status = resp.status
    except urllib.error.HTTPError as err:
        status, body = err.code, err.read()
    assert status == 200, f"{path} -> {status}: {body[:300]!r}"
    return json.loads(body)


class TestChatPaneRestoresLatestConversation:
    """Chat tab paints the conversation /initialize will continue (fix-kw7.6.1)."""

    def test_spa_reloads_latest_turns_on_connect_and_clears_on_new(self):
        page = _spa_bytes()
        assert b"tmLoadLatestConversation" in page
        assert page.find(b"tmLoadLatestConversation") < page.find(b"startup_output")
        assert b"tmClearLog();" in page
        # New conversation must replace the restored thread, not append to it.
        assert b'tm.activeConversationId = null' in page
        assert b"New conversation started (the previous one was saved and titled)." in page
        handler = page.split(b'tm.activeConversationId = null')[1][:200]
        assert b"tmClearLog();" in handler

    def test_highest_conversation_id_is_the_one_to_continue(
        self, tmp_path, monkeypatch
    ):
        from fastworkflow.run_chatbot import server as run_chatbot_server

        monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "state"))
        wf = tmp_path / "wf"
        wf.mkdir()
        db_path = state_paths.observability_db(str(wf))
        _seed_chatbot_channel_history(db_path)
        srv = run_chatbot_server.ChatbotServer(
            db_path, workflow_path=str(wf), port=0
        )
        thread = threading.Thread(target=srv.serve_forever, daemon=True)
        thread.start()
        try:
            convs = _chatbot_get_json(
                srv, "/api/conversations?channel=chatbot&limit=500"
            )["conversations"]
            assert {c["conversation_id"] for c in convs} == {1, 2}
            # last_turn_at ranks the older, populated conversation first —
            # the trap if the pane took conversations[0].
            assert convs[0]["conversation_id"] == 1
            max_id = max(c["conversation_id"] for c in convs)
            assert max_id == 2
            empty = _chatbot_get_json(
                srv, f"/api/turns?channel=chatbot&conversation={max_id}&limit=500"
            )["turns"]
            assert empty == []
            prior = _chatbot_get_json(
                srv, "/api/turns?channel=chatbot&conversation=1&limit=500"
            )["turns"]
            assert [t["user_message"] for t in prior] == ["add eggs", "add milk"]
            assert [t["answer"] for t in prior] == ["added eggs", "added milk"]
            ids_with_turns = {t["conversation_id"] for t in prior}
            latest_with_turns = max(
                c["conversation_id"] for c in convs
                if c["conversation_id"] in ids_with_turns
            )
            assert latest_with_turns == 1
        finally:
            srv.shutdown()
            thread.join(timeout=5)

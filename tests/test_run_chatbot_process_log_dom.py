"""Browser check for the chatbot's View log button (fix-hzux.3).

Discovered by ``tests.browser_validation`` because this module names
``TEST_JSDOM_ROOT``. The page, the chatbot server, and the log files are real.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path

import pytest

from fastworkflow.run_chatbot import launcher
from fastworkflow.run_chatbot import server as run_chatbot_server


@pytest.fixture
def log_server(tmp_path, monkeypatch):
    monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "state"))
    wf = tmp_path / "wf"
    wf.mkdir()
    srv = run_chatbot_server.ChatbotServer(
        workflow_path=str(wf), port=0, spawn_options={"no_server": True}
    )
    server_log = launcher.server_log_path(str(wf), create=True)
    Path(server_log).write_text(
        "server-ready-line\nOPENAI_API_KEY=fw-dom-secret\n",
        encoding="utf-8",
    )
    train_log = launcher.train_log_path(str(wf), create=True)
    Path(train_log).write_text(
        "train-ready-line\nAPP_SECRET=fw-dom-train-secret\n",
        encoding="utf-8",
    )
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv, os.path.abspath(server_log), os.path.abspath(train_log)
    srv.shutdown()
    thread.join(timeout=5)


def test_view_log_button_shows_the_redacted_tail(log_server):
    dependency = os.environ.get("TEST_JSDOM_ROOT")
    if not dependency:
        pytest.skip("Set TEST_JSDOM_ROOT to run DOM integration with jsdom")
    server, server_log, train_log = log_server
    script = Path(__file__).with_name("chatbot_process_log_dom.cjs")
    result = subprocess.run(
        [
            "node",
            str(script),
            dependency,
            f"http://127.0.0.1:{server.port}/?token={server.token}",
            json.dumps({"server_log": server_log, "train_log": train_log}),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr

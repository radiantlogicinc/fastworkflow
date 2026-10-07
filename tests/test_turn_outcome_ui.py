"""The debug UI reports a turn's outcome from its status, not from command success."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from fastworkflow.run_chatbot import server as run_chatbot_server


def test_a_failed_command_does_not_read_as_a_failed_turn(tmp_path):
    jsdom_root = os.environ.get("TEST_JSDOM_ROOT")
    if not jsdom_root:
        pytest.skip("Set TEST_JSDOM_ROOT to run DOM integration with jsdom")
    page = tmp_path / "index.html"
    page.write_bytes(run_chatbot_server.load_index_html())
    script = Path(__file__).with_name("chatbot_turn_outcome_dom.cjs")
    result = subprocess.run(
        ["node", str(script), jsdom_root, str(page)],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr

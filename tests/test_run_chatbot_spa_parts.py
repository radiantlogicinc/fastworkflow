"""The chatbot SPA is assembled from ordered source parts.

``load_index_html`` concatenates ``static/src`` in filename order. The served
page stays one inline classic script.
"""

from __future__ import annotations

import importlib.resources
import shutil
import subprocess
import tempfile

import pytest

from fastworkflow.run_chatbot.server import load_index_html

_SRC = importlib.resources.files("fastworkflow.run_chatbot") / "static" / "src"

# A JS part opens at column 0 on a top-level boundary. The first part is the
# directive that used to sit immediately under <script>; every later part
# opens on an existing section comment.
_JS_STARTS = (b'"use strict";', b"/*", b"function ", b"var ", b"document.")


def _parts():
    return sorted(
        (entry for entry in _SRC.iterdir() if entry.is_file()),
        key=lambda entry: entry.name,
    )


def test_assembled_page_has_exactly_one_script():
    parts = _parts()
    page = b"".join(part.read_bytes() for part in parts)
    assert page == load_index_html()
    assert page.count(b"<script") == 1
    assert page.count(b"</script>") == 1


def test_part_sort_order_is_deterministic():
    names = [part.name for part in _parts()]
    assert names, "SPA source parts are missing"
    assert names == sorted(names)
    assert sorted(list(reversed(names))) == names


def test_every_part_ends_with_a_newline_and_stays_under_2000_lines():
    for part in _parts():
        data = part.read_bytes()
        assert data.endswith(b"\n"), part.name
        # A trailing newline means the newline count is the line count.
        assert data.count(b"\n") <= 2000, part.name


def test_js_parts_start_at_a_top_level_boundary():
    saw_js = False
    for part in _parts():
        if not part.name.endswith(".js"):
            continue
        saw_js = True
        first = part.read_bytes().split(b"\n", 1)[0]
        assert first == first.lstrip(b" \t"), part.name
        assert first.startswith(_JS_STARTS), (part.name, first[:80])
    assert saw_js


def test_concatenated_js_parses():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed; cannot syntax-check the assembled script")
    js = b"".join(
        part.read_bytes() for part in _parts() if part.name.endswith(".js")
    )
    with tempfile.NamedTemporaryFile(suffix=".js") as handle:
        handle.write(js)
        handle.flush()
        result = subprocess.run(
            [node, "--check", handle.name],
            capture_output=True,
            text=True,
            timeout=60,
        )
    assert result.returncode == 0, result.stderr

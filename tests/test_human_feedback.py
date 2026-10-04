"""Feedback uses real stores and HTTP, without model or backend calls."""
import hashlib
import http.server
import json
import os
import re
import shutil
import sqlite3
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from fastworkflow.run_chatbot import server as run_chatbot_server
from fastworkflow.observability import feedback as fb
from fastworkflow.observability import feedback_sidecar
from fastworkflow.observability import store as obs
from tests.test_chatbot_benchmarks import _request
from tests.test_observability_workspace import (
    _manifest, _seed_archive, _store_decl, _turn_row,
)

# Fixtures by plugin, not by import: see tests/test_task_feedback.py.
pytest_plugins = ("tests.test_chatbot_benchmarks",)


def seed(store):
    with store._connect() as conn:
        for key in ('turn-a', 'turn-b'):
            row = _turn_row(key, 'exp-1', 'task-1', 1)
            assert store.upsert_turn_row(conn, row, [], store._store_redactor())
            conn.execute(
                'INSERT INTO spans(span_id,trace_id,name,kind,start_ns,status,attributes) '
                'VALUES(?,?,?,?,?,?,?)',
                ('span-' + key, key, 'fw.planner.plan', 'internal', 1, 'ok', '{}'),
            )


def payload(**kw):
    body = dict(
        target_kind='turn',
        span_ids=[],
        target_label='Turn',
        comment='Needs a clearer answer.',
        provenance='human',
        category='conclusions',
        subcategory='what_went_wrong',
    )
    body.update(kw)
    return body


def test_the_taxonomy_is_exactly_the_six_confirmed_meanings():
    """Pinned as a whole, because the enum values are an owner decision.

    The three-heading composer this replaced ("What went wrong" / "What
    worked" / "What should change") was presentation baked into comment text.
    It is gone: there is no parser, and a heading in a comment is content.
    """
    assert fb.FEEDBACK_SUBCATEGORIES == {
        'observations_analysis': ('observation', 'analysis'),
        'conclusions': ('what_went_right', 'what_went_wrong'),
        'recommendations': ('what_to_do', 'what_not_to_do'),
    }
    assert [c.label for c in fb.FEEDBACK_TAXONOMY] == [
        'Observations / Analysis', 'Conclusions', 'Recommendations',
    ]
    # Every subcategory offers a prompt, because the UI shows one for each.
    assert all(
        sub.watermark.strip() and sub.label.strip()
        for category in fb.FEEDBACK_TAXONOMY
        for sub in category.subcategories
    )
    assert not hasattr(obs, 'parse_human_feedback_comment')
    assert not hasattr(obs, 'compose_human_feedback_comment')
    assert not hasattr(obs, 'HUMAN_FEEDBACK_SECTIONS')


@pytest.mark.parametrize('category,subcategory', [
    ('conclusions', 'observation'),
    ('recommendations', 'what_went_right'),
    ('observations_analysis', 'what_to_do'),
    ('observations', 'observation'),
    ('conclusions', None),
    (None, 'analysis'),
])
def test_a_category_and_subcategory_that_do_not_pair_are_refused(category, subcategory):
    """Membership is not enough: `conclusions` + `observation` is two real
    values that do not go together, and accepting it would make the category
    filter lie about what it is showing."""
    with pytest.raises(fb.FeedbackError):
        fb.validate_category(category, subcategory)


def test_feedback_history_and_evidence_unchanged(experiment_server, tmp_path):
    server, store = experiment_server
    seed(store)
    before = store.get_turn('turn-a'), store.get_spans('turn-a')
    read = '/api/feedback-notes?turn_key=turn-a'
    write = '/post_feedback?turn_key=turn-a'
    assert _request(server, read)[1]['feedback'] == []
    assert _request(server, write, 'POST', payload(), token=None)[0] == 401
    assert _request(server, write, 'POST', payload())[0] == 201
    assert _request(server, write, 'POST', payload(
        category='observations_analysis', subcategory='observation',
        comment='The request names three people, but this plan covers only two.',
    ))[0] == 201
    component = dict(
        target_kind='phase',
        span_ids=['span-turn-a'],
        target_label='Planning',
        comment='Missing a prerequisite.\nUse this order instead.',
        provenance='coding_agent',
        category='recommendations',
        subcategory='what_to_do',
    )
    assert _request(server, write, 'POST', component)[0] == 201
    component['comment'] = 'Follow-up: check the prerequisite first.'
    component['provenance'] = 'distillation_agent'
    component['subcategory'] = 'what_not_to_do'
    assert _request(server, write, 'POST', component)[0] == 201
    rows = _request(server, read)[1]['feedback']
    assert [r['target_kind'] for r in rows] == ['turn', 'turn', 'phase', 'phase']
    assert [r['provenance'] for r in rows] == [
        'human', 'human', 'coding_agent', 'distillation_agent',
    ]
    assert [(r['category'], r['subcategory']) for r in rows] == [
        ('conclusions', 'what_went_wrong'),
        ('observations_analysis', 'observation'),
        ('recommendations', 'what_to_do'),
        ('recommendations', 'what_not_to_do'),
    ]
    assert [r['subcategory_label'] for r in rows] == [
        'What went wrong', 'Observation', 'What to do', 'What not to do',
    ]
    assert all(r['classified'] for r in rows)
    # Comment text is stored and returned verbatim: no headings are added on
    # the way in and none are parsed on the way out.
    assert rows[0]['comment'] == 'Needs a clearer answer.'
    assert rows[2]['comment'] == 'Missing a prerequisite.\nUse this order instead.'
    with store._connect() as conn:
        stored = [r[0] for r in conn.execute(
            'SELECT comment FROM human_feedback WHERE turn_key=? ORDER BY feedback_id',
            ('turn-a',),
        )]
    assert stored[0] == 'Needs a clearer answer.'
    assert 'What went wrong:' not in stored[0]
    # Every row carries a stable identity and its frozen primary anchor.
    assert len({r['feedback_uid'] for r in rows}) == 4
    assert all(r['feedback_uid'].startswith('fb-') for r in rows)
    assert rows[0]['anchors']['primary']['ref']['experiment_id'] == 'exp-1'
    assert rows[0]['anchors']['primary']['ref']['task_id'] == 'task-1'
    assert rows[0]['paired'] is None and rows[0]['pair_key'] is None
    assert all(r['created_at'] for r in rows)
    assert before == (store.get_turn('turn-a'), store.get_spans('turn-a'))
    assert store.list_human_feedback('turn-b') == []
    # A database snapshot carries the comments without another file.
    archive = tmp_path / 'copy.sqlite3'
    store.archive_to(str(archive))
    assert obs.ReadOnlyObservabilityStore(str(archive)).list_human_feedback('turn-a') == rows_without_presentation(rows)
    with store._connect() as conn:
        conn.execute('DELETE FROM turns WHERE turn_key=?', ('turn-a',))
    assert store.list_human_feedback('turn-a') == []


def rows_without_presentation(rows):
    """The store's own shape: `present()` adds labels, it does not store them."""
    drop = ('classified', 'category_label', 'subcategory_label')
    return [{k: v for k, v in row.items() if k not in drop} for row in rows]


@pytest.mark.parametrize('changes', [
    {'target_kind': 'span', 'span_ids': ['span-turn-b']},
    {'target_kind': 'span', 'span_ids': []},
    {'span_ids': ['span-turn-a']},
    {'target_kind': 'invented'},
    {'comment': ''},
    {'comment': 123},
    {'span_ids': 'span-turn-a'},
    {'provenance': 'agent'},
    {'category': 'insights'},
    {'subcategory': 'what_went_wrong', 'category': 'recommendations'},
    {'subcategory': ''},
])
def test_bad_feedback_anchors_refused(experiment_server, changes):
    server, store = experiment_server
    seed(store)
    body = payload()
    body.update(changes)
    assert _request(server, '/post_feedback?turn_key=turn-a', 'POST', body)[0] == 400
    assert store.list_human_feedback('turn-a') == []


@pytest.mark.parametrize('ref,reason', [
    ({'experiment_id': 'exp-forged'}, 'a declared experiment the turn does not record'),
    ({'task_id': 'task-forged'}, 'a declared task the turn does not record'),
    ({'attempt': 7}, 'a declared attempt the turn does not record'),
    ({'store_id': 'not-this-store'}, 'a store the write was not authorized against'),
    ({'pass_id': 'teacher'}, 'a pass with no selector to resolve it'),
])
def test_a_forged_scope_is_refused_against_the_turn_row(experiment_server, ref, reason):
    """A label is not evidence. Every declared field is checked against what
    the turn actually records, so a stale or invented scope cannot ride along
    on real evidence and anchor a comment to something it is not."""
    server, store = experiment_server
    seed(store)
    body = payload(ref=ref)
    status, data = _request(server, '/post_feedback?turn_key=turn-a', 'POST', body)
    assert status == 400, (reason, data)
    assert store.list_human_feedback('turn-a') == []


def test_a_pass_anchor_needs_a_selector_that_resolves(experiment_server):
    """`pass_id` names a recorded pass or it is refused. An unresolvable pass
    would otherwise anchor a comment to the whole turn under a pass's name."""
    server, store = experiment_server
    seed(store)
    path = '/post_feedback?turn_key=turn-a'
    unknown = payload(
        ref={'pass_id': 'teacher'},
        pass_selector={'pass_id': 'teacher', 'span_ids': ['span-not-recorded']},
    )
    assert _request(server, path, 'POST', unknown)[0] == 400
    disagreeing = payload(
        ref={'pass_id': 'teacher'},
        pass_selector={'pass_id': 'student', 'span_ids': ['span-turn-a']},
    )
    assert _request(server, path, 'POST', disagreeing)[0] == 400
    assert store.list_human_feedback('turn-a') == []
    resolving = payload(
        ref={'pass_id': 'teacher'},
        pass_selector={'pass_id': 'teacher', 'span_ids': ['span-turn-a']},
    )
    assert _request(server, path, 'POST', resolving)[0] == 201
    row = store.list_human_feedback('turn-a')[0]
    assert row['anchors']['primary']['ref']['pass_id'] == 'teacher'


def test_unknown_turn_and_malformed_request(experiment_server):
    server, store = experiment_server
    assert _request(server, '/post_feedback?turn_key=missing', 'POST', payload())[0] == 404
    assert _request(server, '/post_feedback', 'POST', payload())[0] == 400
    assert _request(server, '/post_feedback?turn_key=x', 'POST', ['bad'])[0] == 400
    seed(store)
    missing_provenance = payload()
    missing_provenance.pop('provenance')
    assert _request(
        server, '/post_feedback?turn_key=turn-a', 'POST', missing_provenance,
    )[0] == 400
    missing_category = payload()
    missing_category.pop('category')
    assert _request(
        server, '/post_feedback?turn_key=turn-a', 'POST', missing_category,
    )[0] == 400


def test_the_old_write_path_is_gone(experiment_server):
    """`/api/human-feedback` was renamed, not aliased: there is one write
    surface and a stale client learns that rather than being silently served."""
    server, store = experiment_server
    seed(store)
    assert _request(
        server, '/api/human-feedback?turn_key=turn-a', 'POST', payload(),
    )[0] == 405
    assert _request(server, '/api/human-feedback?turn_key=turn-a')[0] == 404
    assert store.list_human_feedback('turn-a') == []


def test_the_read_route_is_a_get_and_the_write_route_is_a_post(experiment_server):
    """The rename kept reads and writes apart, so listing notes can never be
    spelled as an operation that appends one."""
    server, store = experiment_server
    seed(store)
    assert _request(server, '/post_feedback?turn_key=turn-a')[0] == 404
    assert _request(
        server, '/api/feedback-notes?turn_key=turn-a', 'POST', payload(),
    )[0] == 405
    assert store.list_human_feedback('turn-a') == []


def test_the_taxonomy_route_serves_the_same_vocabulary_as_the_module(experiment_server):
    server, _store = experiment_server
    status, data = _request(server, '/api/feedback-taxonomy')
    assert status == 200 and data == fb.taxonomy_payload()


def test_workspace_feedback_is_recorded_beside_the_archive_not_in_it(
    workspace_server,
):
    """Sealed evidence is not appended to, and the comment is still recorded.

    Refusing the write was the old behavior and the wrong half of the
    contract: a reader looking at archived evidence has as much to say about
    it as anyone. The note goes to the annotation sidecar beside the archive
    (fix-9eg.19.1), the archive's bytes do not move, and the read is the union
    so the reader cannot tell which file answered.
    """
    server, _workflow, _before = workspace_server
    stores = server.workspace.stores()
    sid = stores[0]['store_id']
    descriptor = server.workspace.registry.descriptor(sid)
    before = hashlib.sha256(descriptor.path.read_bytes()).hexdigest()
    read = '/api/feedback-notes?turn_key=turn&store_id=' + sid
    status, data = _request(server, read)
    assert status == 200
    assert data == {"feedback": [], "read_only": True, "annotated": True}
    status, written = _request(
        server, '/post_feedback?turn_key=turn&store_id=' + sid, 'POST', payload(),
    )
    assert status == 201, written
    assert [row['comment'] for row in written['feedback']] == ['Needs a clearer answer.']
    assert written['feedback'][0]['category'] == 'conclusions'
    assert written['read_only'] is True
    assert hashlib.sha256(descriptor.path.read_bytes()).hexdigest() == before
    sidecar = Path(feedback_sidecar.feedback_db_path_for(str(descriptor.path)))
    assert sidecar.exists() and sidecar != descriptor.path
    with sqlite3.connect(f'file:{descriptor.path}?mode=ro', uri=True) as evidence:
        assert evidence.execute(
            'SELECT COUNT(*) FROM human_feedback'
        ).fetchone()[0] == 0
    assert _request(server, read)[1]['feedback'][0]['comment'] == 'Needs a clearer answer.'


@pytest.fixture
def two_sealed_stores(tmp_path):
    """Two REAL sealed archives, one logical experiment across both.

    The shape .19.1 names directly: a comparison whose two executions live in
    their own authorized stores. Nothing here is a stand-in — each archive is
    produced by `ObservabilityStore.archive_to` and declared in a manifest
    that names its digest and its evidence identity.
    """
    left = _seed_archive(
        tmp_path, "left", experiment_id="left-local",
        task_id="task-left", turn_key="turn-left",
    )
    right = _seed_archive(
        tmp_path, "right", experiment_id="right-local",
        task_id="task-right", turn_key="turn-right",
    )
    manifest = _manifest(
        tmp_path,
        [_store_decl(left, "left"), _store_decl(right, "right")],
        experiments=[{
            "experiment_id": "logical",
            "label": "logical",
            "segments": [
                {"segment_id": "a", "store_id": "left",
                 "local_experiment_id": "left-local"},
                {"segment_id": "b", "store_id": "right",
                 "local_experiment_id": "right-local"},
            ],
        }],
    )
    server = run_chatbot_server.ChatbotServer(
        port=0, workspace_manifest_path=str(manifest)
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    digests = {
        side["path"]: hashlib.sha256(Path(side["path"]).read_bytes()).hexdigest()
        for side in (left, right)
    }
    yield server, left, right, digests
    server.shutdown()
    thread.join(timeout=5)


def _paired_note(right, *, store_id=None, turn_key="turn-right"):
    return {
        "target_kind": "turn", "span_ids": [], "target_label": "Turn",
        "provenance": "human", "category": "conclusions",
        "subcategory": "what_went_right",
        "comment": "the left archive recovered where the right one gave up",
        "paired": {
            "store_id": store_id or right["store_identity"],
            "turn_keys": [turn_key],
            "experiment_id": "right-local", "task_id": "task-right",
            "attempt": 1, "target_kind": "turn", "span_ids": [],
            "target_label": "Turn",
        },
    }


def test_a_comparison_across_two_sealed_archives_is_recordable(two_sealed_stores):
    """P1 .19.1: paired executions in their OWN authorized stores.

    The comment names a turn in the left archive and a turn in the right one.
    It is recorded in mutable annotation storage beside the left archive,
    both archives' bytes are untouched, and the comment appears ONCE in each
    of the two tasks' Feedback views with its full pair reference intact.
    Before this, a workspace refused the write outright, which left the one
    kind of comment a two-store comparison exists to produce unrecordable.
    """
    server, left, right, digests = two_sealed_stores
    write = "/post_feedback?turn_key=turn-left&store_id=left"
    status, written = _request(server, write, "POST", _paired_note(right))
    assert status == 201, written
    row = written["feedback"][0]
    assert row["pair_key"]
    assert row["paired"]["ref"]["turn_keys"] == ["turn-right"]
    assert row["paired"]["ref"]["store_id"] == right["store_identity"]
    assert row["anchors"]["primary"]["ref"]["turn_keys"] == ["turn-left"]

    # Once in each task's view, and the same row both times.
    def task_view(task_id):
        status, data = _request(
            server,
            "/api/workspace/task-feedback?experiment=logical&task=" + task_id,
        )
        assert status == 200, data
        return data

    from_left = task_view("task-left")
    from_right = task_view("task-right")
    assert from_left["total"] == from_right["total"] == 1
    assert (
        from_left["feedback"][0]["feedback_uid"]
        == from_right["feedback"][0]["feedback_uid"]
        == row["feedback_uid"]
    )
    # The pair identity survives the round trip through the read, so a client
    # can link BOTH sides from either task's view.
    shown = from_right["feedback"][0]
    assert shown["pair_key"] == row["pair_key"]
    assert shown["anchors"]["primary"]["ref"]["turn_keys"] == ["turn-left"]
    assert shown["paired"]["ref"]["turn_keys"] == ["turn-right"]

    # Neither sealed archive moved a byte, and neither grew a journal.
    assert digests == {
        side["path"]: hashlib.sha256(Path(side["path"]).read_bytes()).hexdigest()
        for side in (left, right)
    }
    assert not any(
        Path(f"{side['path']}{suffix}").exists()
        for side in (left, right)
        for suffix in ("-wal", "-shm")
    )
    # The note is in the sidecar beside the left archive, not inside it.
    sidecar = Path(feedback_sidecar.feedback_db_path_for(left["path"]))
    assert sidecar.exists()
    with sqlite3.connect(f"file:{left['path']}?mode=ro", uri=True) as evidence:
        assert evidence.execute(
            "SELECT COUNT(*) FROM human_feedback"
        ).fetchone()[0] == 0


def test_a_paired_reference_to_an_undeclared_store_is_still_refused(
    two_sealed_stores,
):
    """Authorization is the manifest declaration, not the request.

    Widening the workspace to resolve a paired reference must not widen it to
    ANY store id a client cares to name: the identity has to be one the
    manifest declares, or the write is refused and nothing is recorded.
    """
    server, left, right, digests = two_sealed_stores
    write = "/post_feedback?turn_key=turn-left&store_id=left"
    status, error = _request(
        server, write, "POST",
        _paired_note(right, store_id="sha256:" + "0" * 64),
    )
    assert status == 409, error
    assert "no workspace store declares evidence identity" in error["error"]
    assert digests == {
        side["path"]: hashlib.sha256(Path(side["path"]).read_bytes()).hexdigest()
        for side in (left, right)
    }
    status, data = _request(
        server, "/api/workspace/task-feedback?experiment=logical&task=task-left"
    )
    assert status == 200 and data["total"] == 0


def test_a_paired_reference_to_a_turn_the_other_archive_lacks_is_refused(
    two_sealed_stores,
):
    """The declared store is opened and actually checked, not just named."""
    server, left, right, _digests = two_sealed_stores
    status, error = _request(
        server, "/post_feedback?turn_key=turn-left&store_id=left", "POST",
        _paired_note(right, turn_key="no-such-turn"),
    )
    assert status == 400, error


class _DelayingProxy(http.server.BaseHTTPRequestHandler):
    """A real HTTP hop in front of the real server, slow on one path.

    Not a mock and not a stub: every request is forwarded to the actual
    `ChatbotServer` and the actual response is returned. The only thing added
    is latency on `/api/navigation`, which makes the ordering of two
    concurrent browser reads DETERMINISTIC instead of a coin flip — the
    navigation response is guaranteed to land after a turn has been selected
    and while that turn's own trace is still arriving. That window is the
    product race; without the delay a test either reproduces it or does not,
    depending on the machine.
    """

    upstream = ""
    # The trace read is held longer than the navigation read, which is what
    # pins the ordering: the turn is still loading (`state.turn` is null, so
    # `attachTraceHierarchy` cannot align it into the rail) at the moment the
    # navigation response lands. That is the window, and these two numbers are
    # the only reason a test can be inside it every time.
    delays = {"/api/workspace/trace/": 1.5, "/api/navigation": 0.4}
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # noqa: D102 - quiet under pytest
        pass

    def _forward(self, body=None):
        path = self.path.split("?")[0]
        for prefix, seconds in self.delays.items():
            if path.startswith(prefix):
                time.sleep(seconds)
                break
        request = urllib.request.Request(
            self.upstream + self.path,
            data=body,
            method=self.command,
            headers={
                key: value for key, value in self.headers.items()
                if key.lower() not in ("host", "connection", "content-length")
            },
        )
        try:
            with urllib.request.urlopen(request) as response:
                status, headers = response.status, response.headers
                payload = response.read()
        except urllib.error.HTTPError as error:
            status, headers = error.code, error.headers
            payload = error.read()
        self.send_response(status)
        for key, value in headers.items():
            if key.lower() in ("content-length", "transfer-encoding", "connection"):
                continue
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        self._forward()

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        self._forward(self.rfile.read(length) if length else b"")


@pytest.fixture
def slow_navigation_proxy(two_sealed_stores):
    """The two-store workspace, reached through the delaying hop."""
    server, left, right, digests = two_sealed_stores
    handler = type(
        "_BoundDelayingProxy",
        (_DelayingProxy,),
        {"upstream": f"http://127.0.0.1:{server.port}"},
    )
    proxy = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=proxy.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{proxy.server_port}/?token={server.token}", digests
    proxy.shutdown()
    proxy.server_close()
    thread.join(timeout=5)


def test_a_selected_turn_survives_the_navigation_refresh(slow_navigation_proxy):
    """A background navigation read must not repaint over an open trace.

    Found as an intermittent failure of the composer test below and diagnosed
    as a product race, not a test artifact: `refreshConvs` repainted the
    "No conversations yet" placeholder whenever the rail had no path to the
    open record, and `attachTraceHierarchy` can only supply that path once the
    turn's trace has FINISHED loading. Anything selected and still loading was
    fair game — and in workspace mode, where the rail may have no path to a
    scoped turn at all, every periodic refresh could do it, not just the
    first.

    Deterministic because `/api/navigation` is delayed by a real HTTP hop, so
    the refresh reliably lands inside the window rather than sometimes.
    """
    dependency = os.environ.get("TEST_JSDOM_ROOT")
    if not dependency:
        pytest.skip("Set TEST_JSDOM_ROOT to run DOM integration with jsdom")
    if not shutil.which("node"):
        pytest.skip("Node is needed to execute UI integration tests")
    url, digests = slow_navigation_proxy
    script = Path(__file__).with_name("chatbot_nav_refresh_race_dom.cjs")
    result = subprocess.run(
        ["node", str(script), dependency, url, "left", "turn-left"],
        capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert digests == {
        path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for path in digests
    }


def test_the_composer_works_on_sealed_evidence_in_a_real_dom(two_sealed_stores):
    """Clicked, on the branch HTTP tests cannot see.

    `renderFeedback` decides whether to show the composer at all from the
    `read_only`/`annotated` pair the server sends back. Every server-side
    assertion in this file passes against a page that hides the box, which
    would leave a person looking at a sealed archive with nothing to type
    into — the exact human/agent parity .19.1 asks for. So this drives the
    real page: type, classify, save, and find the comment in the task view,
    with the archive's bytes checked here afterwards.
    """
    dependency = os.environ.get("TEST_JSDOM_ROOT")
    if not dependency:
        pytest.skip("Set TEST_JSDOM_ROOT to run DOM integration with jsdom")
    if not shutil.which("node"):
        pytest.skip("Node is needed to execute UI integration tests")
    server, left, right, digests = two_sealed_stores
    script = Path(__file__).with_name("chatbot_annotated_feedback_dom.cjs")
    result = subprocess.run(
        ["node", str(script), dependency,
         f"http://127.0.0.1:{server.port}/?token={server.token}",
         "left", "turn-left", "logical", "task-left"],
        capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert digests == {
        side["path"]: hashlib.sha256(Path(side["path"]).read_bytes()).hexdigest()
        for side in (left, right)
    }
    sidecar = Path(feedback_sidecar.feedback_db_path_for(left["path"]))
    assert sidecar.exists(), "the comment went to the sidecar beside the archive"


def test_experiment_analysis_route_is_gone(experiment_server):
    server, store = experiment_server
    assert _request(server, '/api/experiment/exp-1/analysis', 'PUT', {'analysis': 'x'})[0] == 405
    assert store.get_experiment('exp-1')['notes'] == 'original notes'


@pytest.mark.parametrize('value', ['Free-form observations\nNo JSON required.', '', ['one'], None])
def test_freeform_benchmark_analysis(live_server, value):
    path = '/api/benchmarks/smoke/analysis'
    assert _request(live_server, path, 'PUT', {'analysis': value})[0] == 200
    assert _request(live_server, path)[1]['analysis'] == value


def _ui_script():
    page = run_chatbot_server.load_index_html().decode("utf-8")
    return re.search(r'<script[^>]*>(.*?)</script>', page, re.S).group(1)


def test_ui_analysis_and_stable_component_anchors():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is needed to execute UI helper tests')
    script = _ui_script()
    subprocess.run([node, '--check'], input=script, text=True, check=True, capture_output=True)
    funcs = '\n'.join(
        re.search(r'function ' + name + r'\(.*?\n\}', script, re.S).group(0)
        for name in ['analysisText', 'feedbackAnchor', 'feedbackProvenanceLabel',
                     'feedbackCategory', 'feedbackSubcategory',
                     'feedbackClassification']
    )
    taxonomy = re.search(r'var FEEDBACK_TAXONOMY = \[.*?\n\];', script, re.S).group(0)
    checks = '''
const assert = require('assert');
assert.equal(analysisText(null), ''); assert.equal(analysisText({}), '');
assert.equal(analysisText('free text'), 'free text');
assert.equal(feedbackProvenanceLabel('human'), 'Human');
assert.equal(feedbackProvenanceLabel('coding_agent'), 'Coding Agent');
assert.equal(feedbackProvenanceLabel('distillation_agent'), 'Distillation Agent');
const a = {kind:'span', span:{span_id:'a'}, children:[]};
const b = {kind:'span', span:{span_id:'b'}, children:[]};
assert.deepEqual(feedbackAnchor({kind:'turn',children:[a]}), []);
assert.deepEqual(feedbackAnchor({kind:'phase',children:[b,a,a]}), ['a','b']);
assert.deepEqual(feedbackAnchor({kind:'step',span:{span_id:'step'},children:[a,b]}), ['step']);
assert.equal(feedbackClassification({category:'conclusions', subcategory:'what_went_right'}),
  'Conclusions \\u00b7 What went right');
assert.equal(feedbackClassification({category:'recommendations', subcategory:'what_not_to_do'}),
  'Recommendations \\u00b7 What not to do');
/* A legacy row says it is unclassified rather than borrowing a category. */
assert.equal(feedbackClassification({category:null, subcategory:null}),
  'Unclassified (recorded before categories)');
assert.equal(feedbackClassification({category:'conclusions', subcategory:'observation'}),
  'Unclassified (recorded before categories)');
console.log(JSON.stringify(FEEDBACK_TAXONOMY));
'''
    result = subprocess.run(
        [node, '-e', taxonomy + '\n' + funcs + checks],
        check=True, capture_output=True, text=True,
    )
    # One vocabulary, not two that look alike: the browser constant is the
    # server's payload, field for field.
    assert json.loads(result.stdout)[-1] == fb.taxonomy_payload()['categories'][-1]
    assert json.loads(result.stdout) == fb.taxonomy_payload()['categories']

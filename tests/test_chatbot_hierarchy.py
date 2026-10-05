"""Real SQLite/HTTP navigation contracts and optional DOM click integration."""
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import urllib.error
import urllib.request

import pytest

from fastworkflow.benchmark import setup as setup
from fastworkflow.observability import store as obs
from fastworkflow.experiment.runner import ExperimentController
from fastworkflow.run_chatbot.navigation import build_navigation
from fastworkflow.run_chatbot import server as run_chatbot_server
from tests.test_chatbot_benchmarks import _request, experiment_server, workflow_dir, workspace_server
from tests.test_observability_workspace import _turn_row


def add_turn(store, key, eid=None, day='2026-09-08', channel='chat', cid=1):
    row = _turn_row(key, eid, 'task' if eid else None, 1 if eid else None)
    row.update(channel_id=channel, conversation_id=cid, started_at=day+'T12:00:00+00:00', user_message=key)
    with store._connect() as conn:
        assert store.upsert_turn_row(conn, row, [], store._store_redactor())


def walk(node):
    yield node
    for child in node['children']:
        yield from walk(child)


@pytest.fixture
def hierarchy_server(experiment_server):
    server, default = experiment_server
    for key, day in [('plain today', '2026-09-08'), ('plain yesterday', '2026-09-07')]:
        add_turn(default, key, day=day)
    add_turn(default, 'outside benchmark', eid='exp-1', channel='unassigned')
    spec = setup.save_benchmark(server.workflow_path, {'title': 'Tuning benchmark', 'description': 'Review this benchmark', 'tasks': [{}]})
    registration = setup.create_experiment(server.workflow_path, spec['benchmark_id'], 'v1')
    store = default
    controller = ExperimentController(store.db_path, store.store_identity(), external=False, workflow_folderpath=server.workflow_path)
    eid = registration['experiment_id']
    controller.create_experiment(eid, 'Recorded experiment', declared_tasks=1, declared_attempts=1,
        declarations=[(registration['task_ids'][0], 1, 'registered')])
    store.start_attempt(
        eid, registration['task_ids'][0], 1, 'registered', source_key='registered'
    )
    store.finish_attempt(
        eid, registration['task_ids'][0], 1, outcome='pass', outcome_source='test'
    )
    store.record_evidence_segment(eid, 1, 'evr-recorded',
        {'valid': True, 'problems': [], 'writer_health_delta': {'records_dropped': 0}})
    assert store.complete_experiment(eid) == 'complete'
    add_turn(store, 'experiment-turn', eid=eid, channel='registered')
    with store._connect() as conn:
        conn.execute("INSERT INTO spans(span_id,trace_id,name,kind,start_ns,end_ns,status,attributes) VALUES(?,?,?,?,?,?,?,?)",
                     ('planning', 'experiment-turn', 'fw.planner.plan', 'internal', 1, 1000, 'ok', '{}'))
    setup.create_experiment(server.workflow_path, spec['benchmark_id'], 'v1')
    yield server, spec, eid, default, store


def test_hierarchy_separates_benchmarks_experiments_and_dates(hierarchy_server):
    server, spec, eid, _default, _store = hierarchy_server
    assert _request(server, '/api/navigation', token=None)[0] == 401
    status, data = _request(server, '/api/navigation')
    assert status == 200
    root = data['root']
    benchmark = next(n for n in root['children'] if n.get('benchmark_id') == spec['benchmark_id'])
    assert benchmark['label'] == 'Tuning benchmark'
    assert len(benchmark['children']) == 2
    recorded = next(n for n in benchmark['children'] if n['experiment_id'] == eid)
    assert recorded['recorded']
    turn = next(n for n in walk(recorded) if n['kind'] == 'turn')
    assert turn['turn_key'] == 'experiment-turn' and turn['source'] is None
    adhoc = next(n for n in root['children'] if n['kind'] == 'adhoc')
    assert [n['label'] for n in adhoc['children']] == ['2026-09-08', '2026-09-07']
    assert {n['turn_key'] for n in walk(adhoc) if n['kind'] == 'turn'} == {'plain today', 'plain yesterday'}
    assert any(n['turn_key'] == 'outside benchmark' for n in walk(root) if n['kind'] == 'turn')


def test_more_than_one_page_and_colliding_conversation_ids(tmp_path):
    a = obs.ObservabilityStore(str(tmp_path / 'a.sqlite3'))
    b = obs.ObservabilityStore(str(tmp_path / 'b.sqlite3'))
    for i in range(503):
        add_turn(a, f'a-{i}')
    add_turn(b, 'b')
    root = build_navigation([], [], [{'store': a, 'source': {'store_id': 'a'}}, {'store': b, 'source': {'store_id': 'b'}}])
    assert len([n for n in walk(root) if n['kind'] == 'turn']) == 504
    assert len([n for n in walk(root) if n['kind'] == 'conversation']) == 2


def test_navigation_orders_benchmark_experiments_newest_first():
    rows = [
        {'experiment_id': 'older', 'benchmark_id': 'bench',
         'created_at': '2026-09-08T10:00:00+00:00'},
        {'experiment_id': 'newer', 'benchmark_id': 'bench',
         'created_at': '2026-09-09T10:00:00+00:00'},
    ]
    root = build_navigation(
        [{'benchmark_id': 'bench'}],
        rows,
        [],
    )
    benchmark = root['children'][0]
    assert [node['experiment_id'] for node in benchmark['children']] == [
        'newer',
        'older',
    ]


def _navigation_http(server, *, etag=None):
    url = f'http://127.0.0.1:{server.port}/api/navigation'
    req = urllib.request.Request(url)
    req.add_header('Authorization', f'Bearer {server.token}')
    if etag is not None:
        req.add_header('If-None-Match', etag)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as err:
        return err.code, dict(err.headers), err.read()


def test_navigation_payload_is_slim_and_supports_etag(hierarchy_server):
    """Turn info is whitelisted; other kinds keep full info for showHierarchyInfo."""
    server, _spec, _eid, _default, _store = hierarchy_server
    status, headers, body = _navigation_http(server)
    assert status == 200
    data = json.loads(body)
    turns = [n for n in walk(data['root']) if n['kind'] == 'turn']
    assert turns
    for turn in turns:
        assert set(turn.get('info') or {}).issubset({'status', 'started_at'})
    conversations = [n for n in walk(data['root']) if n['kind'] == 'conversation']
    assert conversations
    assert any(
        {'channel_id', 'conversation_id'} <= set(c.get('info') or {})
        for c in conversations
    )
    etag = headers.get('ETag') or headers.get('etag')
    assert etag
    status304, headers304, body304 = _navigation_http(server, etag=etag)
    assert status304 == 304
    assert body304 == b''
    assert (headers304.get('ETag') or headers304.get('etag')) == etag


def test_navigation_workspace_is_scoped(workspace_server):
    server, _workflow, _before = workspace_server
    status, data = _request(server, '/api/navigation')
    assert status == 200
    turns = [n for n in walk(data['root']) if n['kind'] == 'turn']
    assert turns and all(n['source'].get('store_id') for n in turns)


def test_page_separates_navigation_into_tabs():
    page = run_chatbot_server.load_index_html()
    assert b'id="navConversations"' in page
    assert b'id="navBenchmarks"' in page
    assert b'id="navDistillations"' not in page
    assert b'var navigationSelection = {conversations: null, benchmarks: null};' in page
    assert b'function setNavigationTab(tab, restoreDetail)' in page
    assert b"Distillations are coming soon" not in page
    assert b'setNavigationTab("distillations")' not in page
    assert b"Benchmarks &amp; conversations" not in page
    assert b">WORKSPACE<" not in page


def test_incompatible_default_does_not_hide_registered_experiment(hierarchy_server, tmp_path):
    server, _spec, eid, _default, _store = hierarchy_server
    old = tmp_path / 'old.sqlite3'
    with sqlite3.connect(old) as conn:
        conn.execute('pragma user_version=1')
    server.db_path = str(old)
    status, data = _request(server, '/api/navigation')
    assert status == 200 and data['root']['info']['warnings']
    assert any(n.get('experiment_id') == eid and n['registered'] for n in walk(data['root']) if n['kind'] == 'experiment')


def test_hierarchy_dom_clicks(hierarchy_server):
    dependency = os.environ.get('TEST_JSDOM_ROOT')
    if not dependency:
        pytest.skip('Set TEST_JSDOM_ROOT to run DOM integration with jsdom')
    server, _spec, eid, _default, _store = hierarchy_server
    script = Path(__file__).with_name('chatbot_hierarchy_dom.cjs')
    result = subprocess.run(['node', str(script), dependency,
        f'http://127.0.0.1:{server.port}/?token={server.token}', eid],
        capture_output=True, text=True, timeout=40)
    assert result.returncode == 0, result.stdout + result.stderr


def test_experiment_archive_dom(hierarchy_server):
    dependency = os.environ.get('TEST_JSDOM_ROOT')
    if not dependency:
        pytest.skip('Set TEST_JSDOM_ROOT to run DOM integration with jsdom')
    server, _spec, eid, _default, _store = hierarchy_server
    script = Path(__file__).with_name('chatbot_experiment_archive_dom.cjs')
    result = subprocess.run(['node', str(script), dependency,
        f'http://127.0.0.1:{server.port}/?token={server.token}', eid],
        capture_output=True, text=True, timeout=40)
    assert result.returncode == 0, result.stdout + result.stderr


NAV_TURNS = ['first turn', 'second turn', 'third turn']


@pytest.fixture
def record_nav_server(experiment_server):
    """One conversation of three spanned turns: the arrows need real siblings."""
    server, default = experiment_server
    for index, message in enumerate(NAV_TURNS):
        # A turn_key reaches /api/turn as a path segment, which the server reads
        # verbatim, so the key is a slug and the prose stays in the message.
        key = f'nav-turn-{index + 1}'
        row = _turn_row(key, None, None, None)
        row.update(channel_id='chat', conversation_id=7, ordinal=index + 1,
                   user_message=message,
                   started_at=f'2026-09-08T0{index + 1}:00:00+00:00')
        with default._connect() as conn:
            assert default.upsert_turn_row(conn, row, [], default._store_redactor())
            conn.execute(
                'INSERT INTO spans(span_id,trace_id,name,kind,start_ns,end_ns,status,attributes)'
                ' VALUES(?,?,?,?,?,?,?,?)',
                (f'{key}-plan', key, 'fw.planner.plan', 'internal', 1, 1000, 'ok', '{}'))
    yield server


def test_page_carries_the_record_navigator():
    page = run_chatbot_server.load_index_html()
    assert b'id="recordNav"' in page
    for element in [b'recordNavUp', b'recordNavDown', b'recordNavPrev', b'recordNavNext']:
        assert element in page
    # The tab lives in the gutter #detail reserves for it, so the reserve and
    # the tab have to ship together.
    assert b'--record-nav-gutter' in page
    assert b'function recordNavTarget(move)' in page


def test_record_navigator_dom(record_nav_server):
    dependency = os.environ.get('TEST_JSDOM_ROOT')
    if not dependency:
        pytest.skip('Set TEST_JSDOM_ROOT to run DOM integration with jsdom')
    server = record_nav_server
    script = Path(__file__).with_name('chatbot_record_nav_dom.cjs')
    result = subprocess.run(['node', str(script), dependency,
        f'http://127.0.0.1:{server.port}/?token={server.token}', *NAV_TURNS],
        capture_output=True, text=True, timeout=40)
    assert result.returncode == 0, result.stdout + result.stderr


def _run_dom(script_name, server, *extra, timeout=40):
    dependency = os.environ.get('TEST_JSDOM_ROOT')
    if not dependency:
        pytest.skip('Set TEST_JSDOM_ROOT to run DOM integration with jsdom')
    script = Path(__file__).with_name(script_name)
    result = subprocess.run(
        ['node', str(script), dependency,
         f'http://127.0.0.1:{server.port}/?token={server.token}', *extra],
        capture_output=True, text=True, timeout=timeout)
    assert result.returncode == 0, result.stdout + result.stderr


def test_navigation_etag_dom(hierarchy_server):
    server, _spec, _eid, _default, _store = hierarchy_server
    _run_dom('chatbot_nav_etag_dom.cjs', server)


def test_polling_visibility_dom(hierarchy_server):
    server, _spec, _eid, _default, _store = hierarchy_server
    _run_dom('chatbot_polling_visibility_dom.cjs', server, timeout=60)


def test_keyboard_rail_dom(hierarchy_server):
    server, _spec, _eid, _default, _store = hierarchy_server
    _run_dom('chatbot_keyboard_rail_dom.cjs', server, timeout=90)


def test_unreachable_and_clear_rotation_dom(hierarchy_server):
    server, _spec, _eid, _default, _store = hierarchy_server
    _run_dom('chatbot_unreachable_clear_dom.cjs', server)


def test_perf_render_dom(hierarchy_server):
    server, _spec, _eid, _default, _store = hierarchy_server
    _run_dom('chatbot_perf_render_dom.cjs', server)

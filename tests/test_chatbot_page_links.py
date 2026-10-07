"""The address bar names the page on screen, and every name it shows opens that page.

Real SQLite stores behind a real chatbot server; the DOM checks run in jsdom
and skip unless TEST_JSDOM_ROOT is set (run them through tests.browser_validation).
"""
import json

from fastworkflow.run_chatbot import server as run_chatbot_server
from tests.test_chatbot_benchmarks import _request, _turn_row, experiment_server, workflow_dir
from tests.test_chatbot_hierarchy import _run_dom, hierarchy_server


def _seed_turn(store, key, message, spans, *, channel='chat', cid=9):
    row = _turn_row(key, None, None, None)
    row.update(channel_id=channel, conversation_id=cid, ordinal=1,
               user_message=message, started_at='2026-09-08T01:00:00+00:00')
    with store._connect() as conn:
        assert store.upsert_turn_row(conn, row, [], store._store_redactor())
        for span_id, parent, name, start, end, attrs in spans:
            conn.execute(
                'INSERT INTO spans(span_id,trace_id,parent_span_id,name,kind,start_ns,end_ns,status,attributes)'
                ' VALUES(?,?,?,?,?,?,?,?,?)',
                (span_id, key, parent, name, 'internal', start, end, 'ok', json.dumps(attrs)))
        conn.commit()


def _recorded_spans(prefix):
    """Planning has no span of its own; Execution and its steps do."""
    return [
        (prefix + '-plan', None, 'fw.planner.plan', 1, 1000, {}),
        (prefix + '-exec', None, 'fw.agent.execute', 2000, 9000, {}),
        (prefix + '-step-1', prefix + '-exec', 'fw.agent.step', 2100, 5000,
         {'tool_name': 'first_tool'}),
        (prefix + '-step-2', prefix + '-exec', 'fw.agent.step', 5100, 8900,
         {'tool_name': 'second_tool'}),
    ]


# Recorded before fw.agent.step existed: the steps are inferred from ReAct's
# calls, so neither they nor their execution phase carry a span.
LEGACY_SPANS = [
    ('legacy-r1', None, 'fw.llm.call', 100, 200, {'module_chain': 'fastWorkflowReAct.Predict'}),
    ('legacy-t1', None, 'fw.agent.tool_call', 210, 300, {}),
    ('legacy-r2', None, 'fw.llm.call', 310, 400, {'module_chain': 'fastWorkflowReAct.Predict'}),
]


def test_page_names_pages_without_adding_history_entries():
    page = run_chatbot_server.load_index_html()
    assert b'id="copyLinkBtn"' in page
    assert b'history.replaceState' in page
    # The record navigator is how a reader moves back; a pushed entry would
    # make Back walk the pages instead of leaving the app.
    assert b'pushState' not in page
    assert b'addEventListener("popstate"' not in page


def test_page_links_round_trip_dom(hierarchy_server):
    server, spec, eid, default, _store = hierarchy_server
    _seed_turn(default, 'crumb-turn', 'climb the crumbs', _recorded_spans('crumb'))
    _seed_turn(default, 'legacy-turn', 'an older recording', LEGACY_SPANS,
               channel='cli:local', cid=10)
    status, data = _request(server, f'/api/experiment/{eid}/tasks')
    assert status == 200
    task_id = data['tasks'][0]['task_id']
    _run_dom('chatbot_page_link_dom.cjs', server, eid, spec['benchmark_id'], task_id,
             timeout=120)

"""The diagnostic search and turn diagnosis over HTTP and in the page.

`fix-9eg.18.1/.18.2/.18.3`, browser-facing half. The backend contracts are
pinned in `tests/test_trace_diagnosis.py`; this file pins what a client
actually receives and renders.

Integration throughout, per the repo's testing rules: a real
`ObservabilityStore` seeded through its own write methods, read back through
the real stdlib HTTP server, and -- where jsdom is available -- the shipped SPA
driven in a real DOM. The seeding helpers are imported from the backend test
rather than copied, so a fixture cannot describe one shape here and another
there.

The store is deliberately larger than one page AND larger than the page's scan
bound (`TURN_FIND_SCAN`), because the defect under test is precisely that a
match outside the first bound was reported as no match at all.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path

import pytest

from fastworkflow import state_paths, tracing
from fastworkflow.observability import diagnosis as diag
from fastworkflow.observability import store as obs
from fastworkflow.run_chatbot import server as run_chatbot_server
from tests import test_chatbot_benchmarks as benchmark_fixtures
from tests.test_chatbot_benchmarks import _request
from tests.test_trace_diagnosis import (
    T0,
    _ask_user_span,
    _execute_span,
    _intent_span,
    _output,
    _param_span,
    _plain_turn,
    _record,
    _turn_row,
    _write,
)

# Bound by assignment rather than imported by name: pytest registers a fixture
# under the name it is bound to either way, but a test taking `workflow_dir`
# as a parameter would be flagged as redefining an import (F811).
workflow_dir = benchmark_fixtures.workflow_dir

# Above the page's own scan bound, so the first request cannot reach the end of
# the store and the client must continue to answer at all.
PLAIN_TURNS = 210

# Marker-bearing keys sort BELOW every plain key, so `turn_key DESC` puts each
# of them past the first page and past the first scan segment.
NAV = "turn-00-navigated"
RECORD_ONLY = "turn-01-record-only-failure"
TROUBLE = "turn-02-intent-and-parameter-trouble"
SAME_TYPE = "turn-03-same-type-context"
REPEATED = "turn-04-repeated-not-a-loop"
ASKED = "turn-05-asked-the-user"


def _seed(store: obs.ObservabilityStore) -> None:
    """One store holding every shape the UI has to tell apart."""
    for index in range(PLAIN_TURNS):
        _plain_turn(store, f"turn-9{index:04d}-plain")

    # A dispatch whose recorded context TYPE changes: navigation, provable.
    _write(
        store,
        _turn_row(NAV, record=_record(NAV, refs=[("c1", 1, f"{NAV}-ex")],
                                      outputs=[_output("c1", "open_project")])),
        [_execute_span(f"{NAV}-ex", NAV, call_id="c1", command_name="open_project",
                       start_ns=T0, context_before="Workspace", context_after="Project")],
    )

    # A completed turn whose RECORD carries an unsuccessful command and whose
    # trace carries no span for it: findable only on the record basis.
    _write(
        store,
        _turn_row(
            RECORD_ONLY,
            record=_record(RECORD_ONLY, refs=[("c1", 1, None)],
                           outputs=[_output("c1", "delete_todo", success=False)]),
        ),
        [],
    )

    # Nested decisions that recorded trouble under one executor step.
    _write(
        store,
        _turn_row(TROUBLE, record=_record(TROUBLE, refs=[("c1", 1, f"{TROUBLE}-ex")],
                                          outputs=[_output("c1", "add_todo")])),
        [
            _execute_span(f"{TROUBLE}-ex", TROUBLE, call_id="c1",
                          command_name="add_todo", start_ns=T0),
            _intent_span(f"{TROUBLE}-intent", TROUBLE, start_ns=T0 + 1,
                         parent_span_id=f"{TROUBLE}-ex", ambiguous=True, margin=0.05),
            _param_span(f"{TROUBLE}-param", TROUBLE, start_ns=T0 + 2,
                        parent_span_id=f"{TROUBLE}-ex", parameters_valid=False,
                        missing_fields=["due_date"]),
        ],
    )

    # Two equal context TYPES. A type names a class, not an instance, so this
    # proves nothing either way and must read as unknown.
    _write(
        store,
        _turn_row(SAME_TYPE, record=_record(SAME_TYPE, refs=[("c1", 1, f"{SAME_TYPE}-ex")],
                                            outputs=[_output("c1", "list_todos")])),
        [_execute_span(f"{SAME_TYPE}-ex", SAME_TYPE, call_id="c1",
                       command_name="list_todos", start_ns=T0,
                       context_before="Project", context_after="Project")],
    )

    # The same command three times, all successful. Repetition, not a loop.
    _write(
        store,
        _turn_row(
            REPEATED,
            record=_record(
                REPEATED,
                refs=[(f"c{i}", i, f"{REPEATED}-ex{i}") for i in (1, 2, 3)],
                outputs=[_output(f"c{i}", "list_todos") for i in (1, 2, 3)],
            ),
        ),
        [
            _execute_span(f"{REPEATED}-ex{i}", REPEATED, call_id=f"c{i}",
                          command_name="list_todos", start_ns=T0 + i)
            for i in (1, 2, 3)
        ],
    )

    # A suspension. Not a failure, and must never be chipped as one.
    _write(
        store,
        _turn_row(ASKED, status="awaiting_user", success=False, answer=None,
                  record=_record(ASKED, refs=[("c1", 1, f"{ASKED}-ex")],
                                 outputs=[_output("c1", "add_todo")])),
        [
            _execute_span(f"{ASKED}-ex", ASKED, call_id="c1", command_name="add_todo",
                          start_ns=T0, status=tracing.STATUS_OK, success=None),
            _ask_user_span(f"{ASKED}-ask", ASKED, start_ns=T0 + 1),
        ],
    )


@pytest.fixture
def diagnosis_server(workflow_dir, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "state"))
    db_path = state_paths.observability_db(str(workflow_dir))
    store = obs.ObservabilityStore(db_path)
    _seed(store)
    server = run_chatbot_server.ChatbotServer(
        db_path=db_path,
        workflow_path=str(workflow_dir),
        port=0,
        spawn_options={"no_server": True},
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, store
    server.shutdown()
    thread.join(timeout=5)


def _turns(server, query: str):
    status, data = _request(server, "/api/turns?" + query)
    assert status == 200, data
    return data


# ----------------------------------------------------------------------
# .18.1 -- the search reaches the whole store, over HTTP
# ----------------------------------------------------------------------


def test_a_match_past_the_first_page_is_found_not_reported_as_none(diagnosis_server):
    """The shipped defect, at the route that shipped it.

    The navigating turn sorts last of 216, so every page-sized read reaches it
    only at the end. The old route listed one page and filtered it, and
    answered "no turns" about a store that holds one.
    """
    server, _store = diagnosis_server
    data = _turns(server, "markers_any=context_navigation&limit=5")
    assert [row["turn_key"] for row in data["turns"]] == [NAV]
    assert data["total_matched"] == 1
    assert data["total_matched_exact"] is True
    assert data["counts_scope"] == diag.SCOPE_DATASET
    assert data["total_scanned"] == PLAIN_TURNS + 6
    assert data["has_more"] is False


def test_a_bounded_walk_returns_every_match_exactly_once(diagnosis_server):
    """Continuation over HTTP, with the page smaller than the scan bound.

    This is the pairing that used to lose matches: the page fills before the
    scan bound is reached, and a cursor that trailed the scan stepped over the
    rows it had counted but not returned.
    """
    server, _store = diagnosis_server
    seen: list[str] = []
    summed = 0
    cursor = None
    for _ in range(300):
        query = "limit=2&scan_limit=5"
        if cursor:
            query += "&resume_after=" + cursor
        data = _turns(server, query)
        assert data["counts_scope"] == diag.SCOPE_SEGMENT
        seen.extend(row["turn_key"] for row in data["turns"])
        summed += data["total_matched"]
        if not data["scan_truncated"]:
            assert data["next_scan_cursor"] is None
            break
        cursor = data["next_scan_cursor"]
        assert cursor
    else:
        pytest.fail("the cursor never finished the store")

    assert len(seen) == len(set(seen)), "no turn returned twice"
    assert len(seen) == PLAIN_TURNS + 6, "every turn returned once"
    assert summed == len(seen), "summed segment counts are the dataset's total"
    assert seen == sorted(seen, reverse=True), "one descending order throughout"


def test_the_walk_the_page_performs_reaches_past_its_two_hundredth_match(
    diagnosis_server,
):
    """The client's own parameters, over a store with more matches than the
    page used to be willing to keep.

    216 turns match an unfiltered search and the page walked them 25 at a time
    while discarding everything past the 200th, so the last 16 -- including the
    only turn in the store that navigated context -- were unreachable from the
    browser no matter how long the operator kept clicking. The API was already
    correct here, which is why this failure survived the API tests; the walk
    below uses exactly the limit and scan bound the page uses.
    """
    server, _store = diagnosis_server
    seen: list[str] = []
    cursor = None
    for _ in range(300):
        query = "limit=25&scan_limit=200"
        if cursor:
            query += "&resume_after=" + cursor
        data = _turns(server, query)
        seen.extend(row["turn_key"] for row in data["turns"])
        if not data["scan_truncated"]:
            break
        cursor = data["next_scan_cursor"]
    else:
        pytest.fail("the cursor never finished the store")

    assert len(seen) == PLAIN_TURNS + 6 > 200, "the store has more than the old cap"
    assert len(seen) == len(set(seen))
    assert seen[-1] == NAV, "the last match is the one the cap used to swallow"
    assert seen.index(NAV) >= 200, "and it is genuinely past the two hundredth row"


def test_a_resume_without_a_scan_bound_still_pages_by_cursor(diagnosis_server):
    """The same completeness rule at the public route, with no scan_limit.

    A client that follows `next_scan_cursor` and omits `scan_limit` is still
    walking by cursor. Answering it with a full scan would return one page,
    report the walk finished, and leave every later match unreachable.
    """
    server, _store = diagnosis_server
    first = _turns(server, "limit=2")
    assert first["counts_scope"] == diag.SCOPE_DATASET, (
        "an unresumed, unbounded search is still one complete scan"
    )
    assert first["total_matched"] == PLAIN_TURNS + 6

    seen = [row["turn_key"] for row in first["turns"]]
    cursor = seen[-1]
    for _ in range(300):
        data = _turns(server, "limit=2&resume_after=" + cursor)
        seen.extend(row["turn_key"] for row in data["turns"])
        if not data["scan_truncated"]:
            assert data["next_scan_cursor"] is None
            break
        cursor = data["next_scan_cursor"]
        assert cursor
    else:
        pytest.fail("the cursor never finished the store")

    assert len(seen) == PLAIN_TURNS + 6
    assert len(seen) == len(set(seen))


def test_a_truncated_search_says_so_instead_of_answering_none(diagnosis_server):
    """An unfinished walk with nothing yet is not an answer of "no matches"."""
    server, _store = diagnosis_server
    data = _turns(server, "markers_any=context_navigation&scan_limit=5")
    assert data["turns"] == []
    assert data["scan_truncated"] is True
    assert data["scan_complete"] is False
    assert data["has_more"] is True, "the rest of the store is still unexamined"
    assert data["total_matched_exact"] is False
    assert data["next_scan_cursor"]


def test_facets_count_the_whole_store_and_say_what_they_cannot_count(
    diagnosis_server,
):
    server, _store = diagnosis_server
    data = _turns(server, "limit=1")
    facets = data["facets"]
    assert facets["context_navigation"] == 1
    assert facets["intent_ambiguous"] == 1
    assert facets["parameter_extraction_invalid"] == 1
    assert facets["awaiting_user"] == 1
    assert facets["repeated_command"] == 1
    assert facets["suspected_loop"] == 0, "repetition alone is not a loop"
    # Nobody set a threshold, so the honest answer is "not counted". A zero
    # here would read as "no turn was unconfident".
    assert facets["low_confidence"] is None

    with_threshold = _turns(server, "limit=1&low_confidence_below=0.2")
    assert with_threshold["facets"]["low_confidence"] == 1


def test_text_search_covers_the_whole_store(diagnosis_server):
    server, store = diagnosis_server
    _write(
        store,
        _turn_row("turn-00-needle", user_message="find the plutonium please",
                  record=_record("turn-00-needle", refs=[])),
        [],
    )
    data = _turns(server, "text=PLUTONIUM&limit=5")
    assert [row["turn_key"] for row in data["turns"]] == ["turn-00-needle"]
    assert data["total_matched"] == 1


def test_an_empty_match_is_an_empty_page_and_a_finished_scan(diagnosis_server):
    server, _store = diagnosis_server
    data = _turns(server, "markers_any=parameter_extraction_error")
    assert data["turns"] == []
    assert data["total_matched"] == 0
    assert data["scan_complete"] is True
    assert data["has_more"] is False, "only now may a client say no matches"


def test_a_record_only_failure_is_found_by_default(diagnosis_server):
    """A dispatch with no span still recorded its outcome, and is findable."""
    server, _store = diagnosis_server
    data = _turns(server, "markers_any=step_unsuccessful")
    assert [row["turn_key"] for row in data["turns"]] == [RECORD_ONLY]
    assert data["basis"] == diag.BASIS_SPANS_AND_RECORD
    assert data["query"]["reads_record"] is True

    cheaper = _turns(server, "markers_any=step_unsuccessful&include_record=false")
    assert cheaper["turns"] == []
    assert cheaper["basis"] == diag.BASIS_SPANS, "and the page says which basis"


# ----------------------------------------------------------------------
# Wire values are parsed exactly or refused
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        "limit=abc",
        "limit=5.0",
        "limit=true",
        "limit=1e3",
        "limit=0",
        "limit=100000",
        "offset=-1",
        "attempt=1.5",
        "conversation=seven",
        "scan_limit=0",
        "success=yes",
        "include_record=maybe",
        "low_confidence_below=nan",
        "low_confidence_below=inf",
        "low_confidence_below=-1",
        "low_confidence_below=abc",
        "markers_any=no_such_marker",
        # `markers_any=` is absent from this list on purpose: `parse_qs` drops
        # a valueless parameter entirely, so the wire cannot tell "named it and
        # said nothing" from "did not name it". `_wire_markers` still refuses
        # an empty list for a caller that builds a query in Python.
        "offset=5&scan_limit=10",
        "offset=5&resume_after=turn-01",
    ],
)
def test_a_malformed_filter_is_refused_not_quietly_ignored(diagnosis_server, query):
    """A filter that silently means something else is worse than one that fails.

    The old route fell back to its default when a value did not convert, so
    `?limit=abc` served 100 rows and `?attempt=1.5` served every attempt -- and
    the operator read the result as an answer about the dataset.
    """
    server, _store = diagnosis_server
    status, data = _request(server, "/api/turns?" + query)
    assert status == 400, data
    assert data["error"]


def test_a_refusal_names_the_value_that_was_refused(diagnosis_server):
    server, _store = diagnosis_server
    _status, data = _request(server, "/api/turns?limit=abc")
    assert "'abc'" in data["error"] and "limit" in data["error"]
    _status, data = _request(server, "/api/turns?markers_any=nope")
    assert "nope" in data["error"] and "context_navigation" in data["error"], (
        "an unknown marker is answered with the vocabulary, not just a refusal"
    )


def test_the_existing_filters_still_mean_what_they_meant(diagnosis_server):
    """Query compatibility: the rail's own parameter names keep working."""
    server, store = diagnosis_server
    _write(
        store,
        _turn_row("turn-00-scoped", experiment_id="exp-1", task_id="task-1", attempt=2,
                  record=_record("turn-00-scoped", refs=[])),
        [],
    )
    data = _turns(server, "experiment=exp-1&task=task-1&attempt=2")
    assert [row["turn_key"] for row in data["turns"]] == ["turn-00-scoped"]
    assert _turns(server, "experiment=exp-2")["total_matched"] == 0
    assert _turns(server, "channel=channel-1")["total_matched"] > PLAIN_TURNS
    assert _turns(server, "channel=nobody")["total_matched"] == 0
    assert _turns(server, "success=1")["total_matched"] > 0
    assert _turns(server, "status=awaiting_user")["total_matched"] == 1


def test_a_listed_row_carries_the_chips_the_rail_already_read(diagnosis_server):
    """The existing client keys survive, so the rail needs no rewrite to keep working."""
    server, _store = diagnosis_server
    row = _turns(server, "markers_any=intent_ambiguous")["turns"][0]
    assert row["turn_key"] == TROUBLE
    assert "decision_signals" in row and "intent_margin_min" in row["decision_signals"]
    # The tier-1/2 stamps are not diagnostic markers and are easy to drop when
    # the route changes; the rail renders a chip from each of them.
    assert "llm_calls_cut_at_limit" in row
    assert "llm_cost" in row and "total" in row["llm_cost"]
    assert set(row["markers"]) >= {"intent_ambiguous", "parameter_extraction_invalid"}
    assert row["diagnosis"]["counts"]["intent_ambiguous"] == 1


# ----------------------------------------------------------------------
# .18.2 / .18.3 -- the opened turn's diagnosis
# ----------------------------------------------------------------------


def _detail(server, turn_key: str) -> dict:
    status, data = _request(server, "/api/turn/" + turn_key)
    assert status == 200, data
    return data["turn"]


def test_the_opened_turn_carries_its_diagnosis_over_the_ledger_it_shows(
    diagnosis_server,
):
    """The markers describe the dispatches the ledger renders, not a second list."""
    server, _store = diagnosis_server
    turn = _detail(server, TROUBLE)
    ledger_calls = [row["command_call_id"] for row in turn["execution_ledger"]["rows"]]
    diagnosed = [step["command_call_id"] for step in turn["diagnosis"]["steps"]]
    assert diagnosed == ledger_calls


def test_nested_trouble_is_attributed_to_its_executor_step(diagnosis_server):
    server, _store = diagnosis_server
    turn = _detail(server, TROUBLE)
    step = turn["diagnosis"]["steps"][0]
    assert set(step["markers"]) >= {"intent_ambiguous", "parameter_extraction_invalid"}
    kinds = {event["kind"]: event for event in step["events"]}
    assert kinds[diag.PHASE_INTENT]["span_id"] == f"{TROUBLE}-intent"
    assert kinds[diag.PHASE_PARAM]["span_id"] == f"{TROUBLE}-param"
    assert all(event["owner_call_id"] == "c1" for event in step["events"])
    assert step["anchor"]["span_ids"] == [f"{TROUBLE}-ex"], "a jump target, exactly"


def test_navigation_reports_where_it_went(diagnosis_server):
    server, _store = diagnosis_server
    navigation = _detail(server, NAV)["diagnosis"]["steps"][0]["navigation"]
    assert navigation["state"] == diag.NAV_CHANGED
    assert navigation["basis"] == "context_type_change"
    assert navigation["from"] == "Workspace" and navigation["to"] == "Project"


def test_two_equal_context_types_do_not_prove_the_context_stayed_put(
    diagnosis_server,
):
    """The limitation the UI must not paper over.

    The span records the context's type name, not the instance. Equal types are
    consistent with having navigated between two
    instances of that type, and the honest answer is that nobody recorded it.
    """
    server, _store = diagnosis_server
    navigation = _detail(server, SAME_TYPE)["diagnosis"]["steps"][0]["navigation"]
    assert navigation["state"] == diag.NAV_UNKNOWN
    assert navigation["basis"] == "same_context_type"
    assert "context_navigation" not in _detail(server, SAME_TYPE)["diagnosis"]["markers"]


def test_repetition_is_reported_without_being_called_a_loop(diagnosis_server):
    server, _store = diagnosis_server
    diagnosis = _detail(server, REPEATED)["diagnosis"]
    assert "repeated_command" in diagnosis["markers"]
    assert "suspected_loop" not in diagnosis["markers"]
    group = diagnosis["repeats"][0]
    assert group["count"] == 3 and group["suspected_loop"] is False
    assert group["trouble"] == []
    assert diagnosis["loop_policy"]["min_repeats"] == 3, "the bound travels with the answer"


def test_a_suspension_is_not_a_failure(diagnosis_server):
    server, _store = diagnosis_server
    diagnosis = _detail(server, ASKED)["diagnosis"]
    assert "awaiting_user" in diagnosis["markers"]
    assert "step_unsuccessful" not in diagnosis["markers"]
    assert "step_error" not in diagnosis["markers"]


def test_a_dispatch_with_no_span_stays_visible_and_says_why(diagnosis_server):
    server, _store = diagnosis_server
    turn = _detail(server, RECORD_ONLY)
    rows = turn["execution_ledger"]["rows"]
    assert [row["command_call_id"] for row in rows] == ["c1"]
    assert rows[0]["span_id"] is None
    assert "step_unsuccessful" in turn["diagnosis"]["markers"]
    assert "step_unsuccessful" in turn["diagnosis"]["markers_only_in_record"]
    # No span means nothing finer than the turn can be commented on, and the
    # anchor falls back to turn scope rather than pretending to a step target
    # the store would refuse.
    anchor = turn["diagnosis"]["steps"][0]["anchor"]
    assert anchor["span_ids"] == []
    assert anchor["target_kind"] == "turn"
    assert anchor["command_call_id"] == "c1", "still says which dispatch it is about"


def test_the_search_and_the_opened_turn_agree(diagnosis_server):
    """A row chipped in the list must not lose its markers when opened."""
    server, _store = diagnosis_server
    for turn_key in (NAV, RECORD_ONLY, TROUBLE, REPEATED, ASKED):
        listed = _turns(server, "text=" + turn_key)["turns"]
        assert [row["turn_key"] for row in listed] == [turn_key]
        opened = _detail(server, turn_key)["diagnosis"]["markers"]
        assert set(listed[0]["markers"]) <= set(opened), turn_key


# ----------------------------------------------------------------------
# Scope: a search never widens past the store the caller opened
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# The page itself
# ----------------------------------------------------------------------


def test_the_page_ships_the_search_and_the_diagnosis_renderers():
    page = run_chatbot_server.load_index_html()
    for needle in [
        b'id="turnFind"',
        b'id="turnFindText"',
        b'id="turnFindMarkers"',
        b'id="turnFindStatus"',
        b"function turnFindRun(seq, cursor)",
        b"function renderTurnDiagnosis(container, turn, openSpan)",
        b"function navigationNote(navigation)",
        b"var MARKER_LABEL",
    ]:
        assert needle in page, needle


def test_the_pages_marker_vocabulary_is_the_servers():
    """The labels are a client-side table; this is what stops it drifting.

    A marker the server can return and the page has no word for would render as
    a bare identifier, and one the page knows and the server never emits would
    be a filter that always comes back empty.
    """
    page = run_chatbot_server.load_index_html().decode("utf-8")
    body = page.split("var MARKER_LABEL = {", 1)[1].split("\n};", 1)[0]
    names = [
        line.split(":", 1)[0].strip()
        for line in body.splitlines()
        if line.strip() and not line.strip().startswith(("help", "//"))
        and ":" in line and line.strip()[0].isalpha()
    ]
    assert names == list(diag.MARKER_ORDER)


def test_the_page_never_says_no_matches_before_the_scan_finishes():
    """Pinned in the source because it is a correctness rule, not a style one.

    The empty state is guarded by `turnFind.complete`, which is the negation of
    the server's `scan_truncated`. Without that guard an unfinished walk reads
    exactly like an answered one, which is the bug this slice exists to fix
    reappearing in the client.
    """
    page = run_chatbot_server.load_index_html()
    assert b"if (!turnFind.rows.length && turnFind.complete && !turnFind.running) {" in page
    assert b"No turn in this store matches. The whole store was searched." in page


def test_the_page_bounds_what_one_keystroke_can_cost():
    page = run_chatbot_server.load_index_html()
    assert b"var TURN_FIND_SCAN = 200;" in page
    # A keystroke costs nothing: the search runs only when Find is submitted,
    # so there is no per-keystroke timer to bound.
    assert b"TURN_FIND_DEBOUNCE_MS" not in page
    assert b"function turnFindSchedule" not in page
    assert b'getElementById("turnFindText").addEventListener("input"' not in page
    assert (
        b'document.getElementById("turnFind").addEventListener("submit", function (event) {\n'
        b"  event.preventDefault();"
    ) in page
    # Page smaller than scan is the pairing that used to lose matches; the
    # client picks it deliberately rather than avoiding it.
    assert b"var TURN_FIND_PAGE = 25;" in page


def test_the_page_bounds_cost_without_a_ceiling_on_results():
    """Bounding the REQUESTS is legitimate; bounding the ANSWER is not.

    There used to be a `TURN_FIND_MAX_ROWS = 200` that dropped fetched rows
    while the cursor still advanced past them, which put a 201st match out of
    reach for good -- the same false negative `fix-9eg.18.1` removed from the
    API, reintroduced one layer up. Pinned in the source because a reinstated
    cap would pass every behavioural test on a store smaller than the cap.
    """
    page = run_chatbot_server.load_index_html()
    assert b"TURN_FIND_MAX_ROWS" not in page
    assert b"(page.turns || []).forEach(function (row) { turnFind.rows.push(row); });" in page


# ----------------------------------------------------------------------
# The real page in a real DOM
# ----------------------------------------------------------------------


def test_turn_diagnosis_dom(diagnosis_server):
    dependency = os.environ.get("TEST_JSDOM_ROOT")
    if not dependency:
        pytest.skip("Set TEST_JSDOM_ROOT to run DOM integration with jsdom")
    server, _store = diagnosis_server
    script = Path(__file__).with_name("chatbot_turn_diagnosis_dom.cjs")
    result = subprocess.run(
        [
            "node",
            str(script),
            dependency,
            f"http://127.0.0.1:{server.port}/?token={server.token}",
            json.dumps({"nav": NAV, "same_type": SAME_TYPE, "trouble": TROUBLE,
                        "repeated": REPEATED, "record_only": RECORD_ONLY}),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr

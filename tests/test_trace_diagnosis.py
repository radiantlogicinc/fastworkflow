"""Diagnostic projection and complete-dataset discovery (`fix-9eg.18.1/.2/.3`).

Integration throughout, per the repo's testing rules: a real `ObservabilityStore`
on disk, real span rows through `upsert_span_rows`, real turn rows through
`upsert_turn_row`, the real execution ledger from `run_chatbot/server.py` via
`comparison.project_execution`, and the real `ReadOnlyObservabilityStore` for the
non-destructive read check. No mocks and no stand-in store: the whole claim under
test is that a browser list, an agent read and a detail view describe the same
recorded evidence, and a fake store would not test that.

The seeded traces mirror shapes measured in the pilot corpus rather than
invented ones -- in particular an intent span that records `resolved=false`
without being a failure (1714 of 8558 real spans), an execute span that ended by
exception with no `success` attribute at all (85 real spans), and extraction
spans with no `retry_round` (1701 real spans).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from fastworkflow import tracing
from fastworkflow.observability import capture_policy
from fastworkflow.observability import diagnosis as diag
from fastworkflow.observability import store as obs
from fastworkflow.observability.comparison import (
    ExecutionRef,
    StoreExecutionReader,
    project_execution,
)
from fastworkflow.observability.derived_cache import DerivedTurnCache
from fastworkflow.run_chatbot.server import (
    cost_rollup,
    execution_ledger,
    turn_span_stamps,
)

T0 = 1_700_000_000_000_000_000
STORE_ID = "diagnosis-store"


# ----------------------------------------------------------------------
# Seeding real evidence
# ----------------------------------------------------------------------


def _handle(context_type: str) -> dict:
    """A type-only §6.7 handle, as `tracing.context_handle` projects one.

    Built by the real projector rather than hand-written, so the fixture cannot
    drift from the contract the runtime actually records. `instance_key=None` is
    what `tracing.context_handle` passes, which is why every handle this build
    writes is type-only.
    """
    return capture_policy.project_context_handle(
        context_type=context_type,
        instance_key=None,
        security_scope_ref=tracing.UNSCOPED_SECURITY_SCOPE,
        projector_id=tracing.CONTEXT_PROJECTOR_ID,
        projector_version=tracing.CONTEXT_PROJECTOR_VERSION,
        env={},
    ).model_dump(mode="json")


def _execute_span(
    span_id: str,
    turn_key: str,
    *,
    call_id: str,
    command_name: str,
    start_ns: int,
    context: str = "global",
    status: str = tracing.STATUS_OK,
    success: bool | None = True,
    context_before: str | None = "global",
    context_after: str | None = "global",
    parameters: dict | None = None,
    parent_span_id: str | None = None,
    extra: dict | None = None,
) -> tracing.Span:
    attributes: dict = {tracing.ATTR_COMMAND_CALL_ID: call_id}
    if success is not None:
        attributes["success"] = success
    if context_before is not None:
        attributes[tracing.ATTR_CONTEXT_BEFORE] = _handle(context_before)
    if context_after is not None:
        attributes[tracing.ATTR_CONTEXT_AFTER] = _handle(context_after)
    if parameters is not None:
        attributes["parameters"] = parameters
    if extra:
        attributes.update(extra)
    return tracing.Span(
        span_id=span_id,
        trace_id=turn_key,
        parent_span_id=parent_span_id,
        name=tracing.SPAN_COMMAND_EXECUTE,
        kind=tracing.KIND_INTERNAL,
        channel_id="channel-1",
        command_name=command_name,
        context=context,
        start_ns=start_ns,
        end_ns=start_ns + 1_000_000,
        status=status,
        attributes=attributes,
    )


def _intent_span(
    span_id: str,
    turn_key: str,
    *,
    start_ns: int,
    parent_span_id: str,
    resolved: bool = True,
    ambiguous: bool = False,
    status: str = tracing.STATUS_OK,
    margin: float | None = None,
    stage: str = "INTENT_DETECTION",
) -> tracing.Span:
    attributes: dict = {
        "context": "global",
        "stage": stage,
        "matcher_layer": "classifier",
        "escalation_outcome": "absent",
        "resolved": resolved,
        "ambiguous": ambiguous,
    }
    if margin is not None:
        attributes["decision_uncertainty"] = {
            "signals": [{"kind": "classifier-topk-margin", "value": margin}]
        }
    return tracing.Span(
        span_id=span_id,
        trace_id=turn_key,
        parent_span_id=parent_span_id,
        name=tracing.SPAN_NLU_INTENT,
        kind=tracing.KIND_INTERNAL,
        channel_id="channel-1",
        start_ns=start_ns,
        end_ns=start_ns + 100_000,
        status=status,
        attributes=attributes,
    )


def _param_span(
    span_id: str,
    turn_key: str,
    *,
    start_ns: int,
    parent_span_id: str,
    parameters_valid: bool | None = True,
    retry_round: bool | None = None,
    retry_round_ordinal: int | None = None,
    missing_fields: list | None = None,
    status: str = tracing.STATUS_OK,
) -> tracing.Span:
    attributes: dict = {"command_name": "add_todo"}
    if parameters_valid is not None:
        attributes["parameters_valid"] = parameters_valid
    if retry_round is not None:
        attributes["retry_round"] = retry_round
    if retry_round_ordinal is not None:
        attributes["retry_round_ordinal"] = retry_round_ordinal
    if missing_fields is not None:
        attributes["missing_fields"] = missing_fields
    return tracing.Span(
        span_id=span_id,
        trace_id=turn_key,
        parent_span_id=parent_span_id,
        name=tracing.SPAN_NLU_PARAM_EXTRACTION,
        kind=tracing.KIND_INTERNAL,
        channel_id="channel-1",
        start_ns=start_ns,
        end_ns=start_ns + 100_000,
        status=status,
        attributes=attributes,
    )


def _ask_user_span(span_id: str, turn_key: str, *, start_ns: int) -> tracing.Span:
    return tracing.Span(
        span_id=span_id,
        trace_id=turn_key,
        parent_span_id=None,
        name=tracing.SPAN_ASK_USER,
        kind=tracing.KIND_HUMAN_WAIT,
        channel_id="channel-1",
        start_ns=start_ns,
        end_ns=start_ns + 100_000,
        status=tracing.STATUS_OK,
        attributes={"agent_query": "which one?", "attempt": 1, "user_response": "a"},
    )


def _turn_row(
    turn_key: str,
    *,
    record: dict,
    status: str = "completed",
    success: bool = True,
    user_message: str = "do the thing",
    answer: str = "done",
    experiment_id: str | None = None,
    task_id: str | None = None,
    attempt: int | None = None,
) -> dict:
    return {
        "turn_key": turn_key,
        "channel_id": "channel-1",
        "conversation_id": None,
        "ordinal": None,
        "user_message": user_message,
        "refined_user_message": None,
        "entry_workflow_name": "diagnosis-test",
        "entry_context": "global",
        "status": status,
        "success": 1 if success else 0,
        "failure_reason": None,
        "answer": answer,
        "conversation_summary": None,
        "conversation_traces": None,
        "started_at": "2026-09-19T00:00:00+00:00",
        "completed_at": "2026-09-19T00:00:02+00:00",
        "suspended_ms": 0,
        "continuation_of": None,
        "record_version": 1,
        "experiment_id": experiment_id,
        "task_id": task_id,
        "attempt": attempt,
        "claim_epoch": None,
        "server_incarnation": None,
        "record_json": json.dumps(record),
    }


def _record(
    turn_key: str,
    *,
    refs: list[tuple[str, int, str | None]],
    outputs: list[dict] | None = None,
) -> dict:
    return {
        "turn_output": {
            "turn_key": turn_key,
            "success": True,
            "command_outputs": outputs or [],
        },
        "execution_records": [
            {
                "command_call_id": call_id,
                "parent_call_id": None,
                "command_ordinal": ordinal,
                "span_id": span_id,
            }
            for call_id, ordinal, span_id in refs
        ],
        "routing_events": [],
    }


def _output(call_id: str, command_name: str, *, success: bool = True) -> dict:
    return {
        "command_response": {"response": "ok", "success": success, "artifacts": {}},
        "workflow_name": "diagnosis-test",
        "context": "global",
        "command_name": command_name,
        "command_parameters": {},
        "command_call_id": call_id,
    }


def _write(store: obs.ObservabilityStore, turn_row: dict, spans: list) -> None:
    with store._connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        assert store.upsert_turn_row(conn, turn_row, [], store._store_redactor())
        if spans:
            store.upsert_span_rows(conn, spans, store._store_redactor())
        conn.commit()


@pytest.fixture
def store(tmp_path: Path) -> obs.ObservabilityStore:
    return obs.ObservabilityStore(str(tmp_path / "evidence.sqlite3"))


def _plain_turn(store: obs.ObservabilityStore, turn_key: str, **row_kwargs) -> None:
    """A turn with one ordinary successful dispatch and no marker at all."""
    span = _execute_span(
        f"{turn_key}-ex", turn_key, call_id="c1", command_name="list_todos", start_ns=T0
    )
    record = _record(turn_key, refs=[("c1", 1, f"{turn_key}-ex")], outputs=[_output("c1", "list_todos")])
    _write(store, _turn_row(turn_key, record=record, **row_kwargs), [span])


def _navigating_turn(store: obs.ObservabilityStore, turn_key: str) -> None:
    span = _execute_span(
        f"{turn_key}-ex",
        turn_key,
        call_id="c1",
        command_name="open_project",
        start_ns=T0,
        context_before="Workspace",
        context_after="Project",
    )
    record = _record(turn_key, refs=[("c1", 1, f"{turn_key}-ex")], outputs=[_output("c1", "open_project")])
    _write(store, _turn_row(turn_key, record=record), [span])


# ----------------------------------------------------------------------
# .18.1 -- discovery over the complete dataset
# ----------------------------------------------------------------------


def test_match_beyond_the_first_page_is_found(store: obs.ObservabilityStore) -> None:
    """The defect this slice exists to fix.

    Twelve turns, only the OLDEST carrying the marker. Ordered `turn_key DESC`
    it lands on page 4 of a 3-row page size, which is exactly where the server's
    post-page filter loses it and answers "no matches".
    """
    for index in range(11):
        _plain_turn(store, f"turn-{index + 1:02d}")
    _navigating_turn(store, "turn-00")

    page = diag.search_turns(
        store,
        diag.TurnQuery(markers_any=(diag.MARKER_CONTEXT_NAVIGATION,), limit=3),
        store_id=STORE_ID,
    )
    assert page.total_matched == 1
    assert page.total_matched_exact is True
    assert [row["turn_key"] for row in page.rows] == ["turn-00"]
    assert page.total_scanned == 12
    assert page.has_more is False


def test_paging_is_stable_disjoint_and_totals_agree(store: obs.ObservabilityStore) -> None:
    for index in range(10):
        _navigating_turn(store, f"turn-{index:02d}")

    seen: list[str] = []
    total = None
    for offset in range(0, 12, 4):
        page = diag.search_turns(
            store,
            diag.TurnQuery(
                markers_any=(diag.MARKER_CONTEXT_NAVIGATION,), limit=4, offset=offset
            ),
            store_id=STORE_ID,
        )
        total = page.total_matched
        seen.extend(row["turn_key"] for row in page.rows)

    assert total == 10
    assert len(seen) == 10
    assert len(set(seen)) == 10, "pages must not overlap"
    assert seen == sorted(seen, reverse=True), "order is turn_key DESC"
    assert page.order == "turn_key DESC"


def test_low_confidence_filter_scans_past_the_first_page(
    store: obs.ObservabilityStore,
) -> None:
    """`low_confidence_below` applied to the dataset, not to a loaded page."""
    for index in range(1, 9):
        turn_key = f"turn-{index:02d}"
        execute = _execute_span(
            f"{turn_key}-ex", turn_key, call_id="c1", command_name="list_todos", start_ns=T0
        )
        intent = _intent_span(
            f"{turn_key}-nlu", turn_key, start_ns=T0 + 1, parent_span_id=f"{turn_key}-ex", margin=0.9
        )
        record = _record(turn_key, refs=[("c1", 1, f"{turn_key}-ex")])
        _write(store, _turn_row(turn_key, record=record), [execute, intent])

    unconfident = "turn-00"
    execute = _execute_span(
        f"{unconfident}-ex", unconfident, call_id="c1", command_name="list_todos", start_ns=T0
    )
    intent = _intent_span(
        f"{unconfident}-nlu",
        unconfident,
        start_ns=T0 + 1,
        parent_span_id=f"{unconfident}-ex",
        margin=0.01,
    )
    _write(
        store,
        _turn_row(unconfident, record=_record(unconfident, refs=[("c1", 1, f"{unconfident}-ex")])),
        [execute, intent],
    )

    page = diag.search_turns(
        store, diag.TurnQuery(low_confidence_below=0.2, limit=2), store_id=STORE_ID
    )
    assert [row["turn_key"] for row in page.rows] == [unconfident]
    assert page.total_matched == 1


def test_a_turn_with_no_recorded_margin_is_not_low_confidence(
    store: obs.ObservabilityStore,
) -> None:
    """Missing decision evidence is not low confidence (`.18.1` acceptance)."""
    turn_key = "turn-00"
    execute = _execute_span(
        f"{turn_key}-ex", turn_key, call_id="c1", command_name="list_todos", start_ns=T0
    )
    intent = _intent_span(
        f"{turn_key}-nlu", turn_key, start_ns=T0 + 1, parent_span_id=f"{turn_key}-ex", margin=None
    )
    _write(
        store,
        _turn_row(turn_key, record=_record(turn_key, refs=[("c1", 1, f"{turn_key}-ex")])),
        [execute, intent],
    )

    page = diag.search_turns(
        store, diag.TurnQuery(low_confidence_below=0.5), store_id=STORE_ID
    )
    assert page.total_matched == 0
    assert page.rows == ()


def test_empty_match_is_an_empty_page_not_an_error(store: obs.ObservabilityStore) -> None:
    _plain_turn(store, "turn-00")
    page = diag.search_turns(
        store,
        diag.TurnQuery(markers_any=(diag.MARKER_SUSPECTED_LOOP,)),
        store_id=STORE_ID,
    )
    assert page.rows == ()
    assert page.total_matched == 0
    assert page.total_scanned == 1
    assert page.total_matched_exact is True


def test_facets_count_the_whole_dataset_and_admit_what_they_cannot_answer(
    store: obs.ObservabilityStore,
) -> None:
    for index in range(4):
        _navigating_turn(store, f"turn-1{index}")
    for index in range(3):
        _plain_turn(store, f"turn-0{index}")

    page = diag.search_turns(store, diag.TurnQuery(limit=2), store_id=STORE_ID)
    assert page.facets[diag.MARKER_CONTEXT_NAVIGATION] == 4
    assert page.facets[diag.MARKER_STEP_UNSUCCESSFUL] == 0
    # No threshold was supplied, so the count is unanswerable rather than zero.
    assert page.facets[diag.MARKER_LOW_CONFIDENCE] is None
    assert len(page.rows) == 2
    assert page.total_matched == 7


def test_text_search_covers_the_whole_dataset(store: obs.ObservabilityStore) -> None:
    for index in range(1, 7):
        _plain_turn(store, f"turn-{index:02d}", user_message="routine request")
    _plain_turn(store, "turn-00", user_message="please RECONCILE the ledger")

    page = diag.search_turns(
        store, diag.TurnQuery(text_contains="reconcile", limit=2), store_id=STORE_ID
    )
    assert [row["turn_key"] for row in page.rows] == ["turn-00"]
    assert page.total_matched == 1


def test_scope_is_the_callers_and_is_never_widened(store: obs.ObservabilityStore) -> None:
    _navigating_turn(store, "turn-00")
    turn_key = "turn-01"
    span = _execute_span(
        f"{turn_key}-ex",
        turn_key,
        call_id="c1",
        command_name="open_project",
        start_ns=T0,
        context_before="Workspace",
        context_after="Project",
    )
    _write(
        store,
        _turn_row(
            turn_key,
            record=_record(turn_key, refs=[("c1", 1, f"{turn_key}-ex")]),
            experiment_id="exp-a",
            task_id="task-1",
            attempt=2,
        ),
        [span],
    )

    scoped = diag.search_turns(
        store,
        diag.TurnQuery(
            experiment_id="exp-a",
            task_id="task-1",
            attempt=2,
            markers_any=(diag.MARKER_CONTEXT_NAVIGATION,),
        ),
        store_id=STORE_ID,
    )
    assert [row["turn_key"] for row in scoped.rows] == ["turn-01"]
    assert scoped.total_matched == 1

    other = diag.search_turns(
        store, diag.TurnQuery(experiment_id="exp-b"), store_id=STORE_ID
    )
    assert other.total_matched == 0


def test_scan_truncation_is_reported_rather_than_absorbed(
    store: obs.ObservabilityStore,
) -> None:
    for index in range(6):
        _navigating_turn(store, f"turn-{index:02d}")

    page = diag.search_turns(
        store,
        diag.TurnQuery(markers_any=(diag.MARKER_CONTEXT_NAVIGATION,), scan_limit=3),
        store_id=STORE_ID,
    )
    assert page.total_scanned == 3
    assert page.scan_truncated is True
    assert page.total_matched_exact is False
    assert page.total_matched == 3, "a floor, not the dataset's answer"
    assert page.has_more is True, "unscanned dataset is more, page-full or not"
    assert page.next_scan_cursor == "turn-03"


def _walk_segments(
    store: obs.ObservabilityStore, **query_kwargs
) -> tuple[list[str], int, int]:
    """Follow `next_scan_cursor` to the end; return keys, segments, summed total.

    The shape every segmented caller is expected to use, written once so the
    completeness tests below exercise the contract rather than three private
    interpretations of it.
    """
    keys: list[str] = []
    summed = 0
    segments = 0
    cursor: str | None = None
    while True:
        segments += 1
        assert segments < 200, "the cursor must make progress, not spin"
        page = diag.search_turns(
            store, diag.TurnQuery(resume_after=cursor, **query_kwargs), store_id=STORE_ID
        )
        keys.extend(row["turn_key"] for row in page.rows)
        summed += page.total_matched
        if not page.scan_truncated:
            assert page.next_scan_cursor is None
            return keys, segments, summed
        cursor = page.next_scan_cursor
        assert cursor is not None


def test_a_segmented_walk_returns_every_match_exactly_once(
    store: obs.ObservabilityStore,
) -> None:
    """The reported continuation defect, at its smallest.

    Three matching turns, a page of one and a scan bound of two. A cursor that
    trailed the SCAN rather than the page returned `turn-02`, then handed back a
    cursor already positioned past `turn-01` -- which had been counted, dropped,
    and was then behind the resume point forever. The middle row simply vanished.

    A bounded page is allowed to defer a match to the next segment. It is not
    allowed to count one and then step over it.
    """
    for index in range(3):
        _navigating_turn(store, f"turn-{index:02d}")

    keys, segments, summed = _walk_segments(
        store, markers_any=(diag.MARKER_CONTEXT_NAVIGATION,), limit=1, scan_limit=2
    )
    assert keys == ["turn-02", "turn-01", "turn-00"]
    assert summed == 3, "each match counted once across the walk"
    # Three one-row segments and no terminal empty one: the last full page lands
    # on the dataset's last row, and the probe for anything beyond it comes back
    # empty, so the walk ends without spending a round trip to discover that.
    assert segments == 3


@pytest.mark.parametrize("limit", [1, 2, 3, 5, 7])
@pytest.mark.parametrize("scan_limit", [1, 2, 3, 5, 11])
def test_every_page_and_scan_bound_pairing_stays_complete(
    store: obs.ObservabilityStore, limit: int, scan_limit: int
) -> None:
    """Completeness must not depend on a lucky ratio of page size to scan size.

    The original defect only appeared when the page filled before the scan bound
    was reached, so a UI could have hidden it forever by choosing a page larger
    than its scan. Every pairing is walked here, including those where the page
    is larger, smaller and equal.
    """
    expected = [f"turn-{index:02d}" for index in reversed(range(9))]
    for key in expected:
        _navigating_turn(store, key)

    keys, _, summed = _walk_segments(
        store,
        markers_any=(diag.MARKER_CONTEXT_NAVIGATION,),
        limit=limit,
        scan_limit=scan_limit,
    )
    assert keys == expected
    assert summed == 9


@pytest.mark.parametrize("scan_limit", [1, 2, 3, 4])
def test_a_segmented_walk_over_sparse_matches_stays_complete(
    store: obs.ObservabilityStore, scan_limit: int
) -> None:
    """Matches interleaved with non-matches, so segments straddle both."""
    matching = []
    for index in range(10):
        key = f"turn-{index:02d}"
        if index % 3 == 0:
            _navigating_turn(store, key)
            matching.append(key)
        else:
            _plain_turn(store, key)

    keys, _, summed = _walk_segments(
        store,
        markers_any=(diag.MARKER_CONTEXT_NAVIGATION,),
        limit=1,
        scan_limit=scan_limit,
    )
    assert keys == sorted(matching, reverse=True)
    assert summed == len(matching)


def test_a_segment_never_counts_a_match_it_did_not_return(
    store: obs.ObservabilityStore,
) -> None:
    """The invariant underneath the fix, asserted directly.

    A segment's count is what the caller received plus what it explicitly
    skipped -- never a match left behind the cursor. Checked segment by segment
    rather than only in the total, because a walk can arrive at the right total
    by losing one row and double-counting another.
    """
    for index in range(7):
        _navigating_turn(store, f"turn-{index:02d}")

    cursor: str | None = None
    seen: list[str] = []
    while True:
        page = diag.search_turns(
            store,
            diag.TurnQuery(
                markers_any=(diag.MARKER_CONTEXT_NAVIGATION,),
                limit=2,
                scan_limit=3,
                resume_after=cursor,
            ),
            store_id=STORE_ID,
        )
        assert page.total_matched == len(page.rows), (
            "a segmented scan must stop at a full page, not count past it"
        )
        seen.extend(row["turn_key"] for row in page.rows)
        if not page.scan_truncated:
            break
        cursor = page.next_scan_cursor

    assert seen == [f"turn-{index:02d}" for index in reversed(range(7))]
    assert len(seen) == len(set(seen)), "no row returned twice"


def test_the_cursor_of_a_full_page_resumes_at_the_next_unseen_row(
    store: obs.ObservabilityStore,
) -> None:
    """The cursor points at the last row CONSUMED, which a caller can check."""
    for index in range(4):
        _navigating_turn(store, f"turn-{index:02d}")

    page = diag.search_turns(
        store,
        diag.TurnQuery(
            markers_any=(diag.MARKER_CONTEXT_NAVIGATION,), limit=1, scan_limit=3
        ),
        store_id=STORE_ID,
    )
    assert [row["turn_key"] for row in page.rows] == ["turn-03"]
    assert page.next_scan_cursor == "turn-03", "the row returned, not the row scanned"
    assert page.total_scanned == 1, "a full page stops the scan; it does not run on"
    assert page.counts_scope == diag.SCOPE_SEGMENT


def test_a_bounded_scan_still_reaches_matches_beyond_its_bound(
    store: obs.ObservabilityStore,
) -> None:
    """A per-request bound defers discovery; it must never foreclose it.

    Every match sits beyond the first two segments, so a scan cap that merely
    reported truncation would make them permanently unreachable however the
    caller paged. Following `next_scan_cursor` finds all of them.
    """
    for index in range(5, 9):
        _plain_turn(store, f"turn-{index:02d}")
    for index in range(5):
        _navigating_turn(store, f"turn-{index:02d}")

    found: list[str] = []
    running_total = 0
    segments = 0
    cursor: str | None = None
    while True:
        segments += 1
        assert segments < 20, "the cursor must make progress, not spin"
        page = diag.search_turns(
            store,
            diag.TurnQuery(
                markers_any=(diag.MARKER_CONTEXT_NAVIGATION,),
                scan_limit=2,
                resume_after=cursor,
            ),
            store_id=STORE_ID,
        )
        found.extend(row["turn_key"] for row in page.rows)
        running_total += page.total_matched
        # A resumed call counts its own segment, so its count is never the
        # dataset's -- the dataset's total is the running sum, and it is final
        # when a segment reports the walk complete.
        assert page.counts_scope == diag.SCOPE_SEGMENT
        assert page.total_matched_exact is False
        if not page.scan_truncated:
            assert page.scan_complete is True
            assert page.next_scan_cursor is None
            break
        cursor = page.next_scan_cursor
        assert cursor is not None

    assert sorted(found) == ["turn-00", "turn-01", "turn-02", "turn-03", "turn-04"]
    assert running_total == 5, "summed segment counts are the dataset's total"


def test_a_resume_without_a_scan_bound_still_pages_by_cursor(
    store: obs.ObservabilityStore,
) -> None:
    """A resume is a cursor walk whether or not the caller also bounded it.

    `scan_limit` says how far one call may walk when the page does NOT fill; it
    is not what makes the walk a walk. Treating a bare `resume_after` as an
    unbounded scan ended it at the first full page -- the remaining matches were
    counted, dropped, and left with no cursor to reach them, which is the same
    lost-match defect wearing different parameters.
    """
    for index in range(6):
        _navigating_turn(store, f"turn-{index:02d}")

    first = diag.search_turns(
        store,
        diag.TurnQuery(markers_any=(diag.MARKER_CONTEXT_NAVIGATION,), limit=2),
        store_id=STORE_ID,
    )
    assert [row["turn_key"] for row in first.rows] == ["turn-05", "turn-04"]
    assert first.counts_scope == diag.SCOPE_DATASET, "an unresumed scan is complete"

    # Resuming from the last row received, with no scan bound at all.
    found = list(first.rows)
    cursor = first.rows[-1]["turn_key"]
    for _ in range(10):
        page = diag.search_turns(
            store,
            diag.TurnQuery(
                markers_any=(diag.MARKER_CONTEXT_NAVIGATION,),
                limit=2,
                resume_after=cursor,
            ),
            store_id=STORE_ID,
        )
        assert page.counts_scope == diag.SCOPE_SEGMENT
        assert page.total_matched == len(page.rows), "counted only what it returned"
        found.extend(page.rows)
        if not page.scan_truncated:
            assert page.next_scan_cursor is None
            break
        cursor = page.next_scan_cursor
        assert cursor is not None
    else:
        pytest.fail("a bare resume never finished the dataset")

    assert [row["turn_key"] for row in found] == [
        f"turn-{index:02d}" for index in reversed(range(6))
    ]


def test_a_first_segment_that_finishes_the_dataset_counts_the_dataset(
    store: obs.ObservabilityStore,
) -> None:
    """`scan_limit` alone does not make a count a floor; stopping short does."""
    for index in range(3):
        _navigating_turn(store, f"turn-{index:02d}")

    page = diag.search_turns(
        store,
        diag.TurnQuery(markers_any=(diag.MARKER_CONTEXT_NAVIGATION,), scan_limit=50),
        store_id=STORE_ID,
    )
    assert page.scan_truncated is False
    assert page.counts_scope == diag.SCOPE_DATASET
    assert page.total_matched_exact is True
    assert page.total_matched == 3
    assert page.next_scan_cursor is None
    assert page.has_more is False


def test_offset_may_not_be_combined_with_either_segmenting_control() -> None:
    """Offset pages a complete scan; the cursor walks a segmented one."""
    with pytest.raises(diag.InvalidTurnQuery):
        diag.TurnQuery(resume_after="turn-05", offset=10)
    with pytest.raises(diag.InvalidTurnQuery):
        diag.TurnQuery(scan_limit=5, offset=10)
    # Either control alone is fine, and so is offset alone.
    diag.TurnQuery(scan_limit=5)
    diag.TurnQuery(resume_after="turn-05")
    diag.TurnQuery(offset=10)


def test_reads_never_modify_the_store(tmp_path: Path) -> None:
    """A diagnostic read of evidence is not allowed to change the evidence."""
    db_path = tmp_path / "evidence.sqlite3"
    writable = obs.ObservabilityStore(str(db_path))
    _navigating_turn(writable, "turn-00")
    _plain_turn(writable, "turn-01")

    before = hashlib.sha256(db_path.read_bytes()).hexdigest()
    before_size = db_path.stat().st_size

    reader = obs.ReadOnlyObservabilityStore(str(db_path))
    page = diag.search_turns(reader, diag.TurnQuery(), store_id=STORE_ID)
    assert page.total_matched == 2
    diagnosis = diag.diagnose_execution(
        ExecutionRef(store_id=STORE_ID, turn_keys=("turn-00",)),
        StoreExecutionReader(STORE_ID, reader),
        ledger=execution_ledger,
        cost_rollup=cost_rollup,
    )
    assert diagnosis.turns

    assert hashlib.sha256(db_path.read_bytes()).hexdigest() == before
    assert db_path.stat().st_size == before_size


# ----------------------------------------------------------------------
# .18.2 -- navigation and unsuccessful executor steps
# ----------------------------------------------------------------------


def test_navigation_reports_source_and_destination(store: obs.ObservabilityStore) -> None:
    _navigating_turn(store, "turn-00")
    spans = store.get_spans("turn-00")
    markers = diag.turn_markers("turn-00", spans)

    assert markers.has(diag.MARKER_CONTEXT_NAVIGATION)
    transition = markers.navigation["transitions"][0]
    assert transition["from"] == "Workspace"
    assert transition["to"] == "Project"
    assert transition["span_id"] == "turn-00-ex"
    assert markers.navigation["unknown"] == 0


def _one_dispatch(
    store: obs.ObservabilityStore, turn_key: str, *, extra: dict | None = None, **kwargs
) -> list[dict]:
    span = _execute_span(
        f"{turn_key}-ex",
        turn_key,
        call_id="c1",
        command_name="open_project",
        start_ns=T0,
        extra=extra,
        **kwargs,
    )
    _write(
        store,
        _turn_row(turn_key, record=_record(turn_key, refs=[("c1", 1, f"{turn_key}-ex")])),
        [span],
    )
    return store.get_spans(turn_key)


def _concrete(context_type: str, instance_key: str, key_version: str = "1") -> dict:
    """A CONCRETE handle, from the real projector under a real HMAC key.

    Nothing in the runtime produces one today (`fix-ppmo`), but §6.7 provides
    for it and the diagnosis must be right when it arrives, so the fixture is
    the projector's own output rather than a guess at its shape.
    """
    return capture_policy.project_context_handle(
        context_type=context_type,
        instance_key=instance_key,
        security_scope_ref=tracing.UNSCOPED_SECURITY_SCOPE,
        projector_id=tracing.CONTEXT_PROJECTOR_ID,
        projector_version=tracing.CONTEXT_PROJECTOR_VERSION,
        hmac_key_version=key_version,
        env={capture_policy.HMAC_KEY_VAR: f"test-key-{key_version}"},
    ).model_dump(mode="json")


def test_the_handle_fixtures_are_what_they_claim_to_be() -> None:
    """Guards the navigation tests from becoming vacuous.

    If `_concrete` silently produced a type-only handle -- which is what happens
    when no HMAC key reaches the projector -- every fingerprint test below would
    pass for the wrong reason.
    """
    type_only = _handle("Project")
    assert type_only["instance_fingerprint"] is None

    first = _concrete("Project", "project-a")
    second = _concrete("Project", "project-b")
    assert first["instance_fingerprint"] is not None
    assert first["instance_fingerprint"] != second["instance_fingerprint"]
    assert first["context_type"] == second["context_type"] == "Project"
    for key in diag.FINGERPRINT_COMPATIBILITY_KEYS:
        assert first[key] and first[key] == second[key], key


def test_matching_type_only_handles_do_not_prove_the_context_stayed_put(
    store: obs.ObservabilityStore,
) -> None:
    """Every handle this build writes is type-only (`instance_key=None`).

    `Project` before and `Project` after is equally consistent with one project
    throughout and with a move between two of them, so the honest answer is
    unknown, not unchanged.
    """
    spans = _one_dispatch(store, "turn-00", context_before="Project", context_after="Project")
    markers = diag.turn_markers("turn-00", spans)

    assert not markers.has(diag.MARKER_CONTEXT_NAVIGATION)
    assert markers.navigation["type_only"] == 1
    assert markers.navigation["unknown"] == 0
    assert markers.coverage["dispatches_with_type_only_handles"] == 1
    # A universal property of the projector, not a per-turn capture gap, so it
    # does not spend the `partial_evidence` chip.
    assert not markers.has(diag.MARKER_PARTIAL_EVIDENCE)


def test_concrete_handles_of_one_type_can_prove_a_move_between_instances(
    store: obs.ObservabilityStore,
) -> None:
    turn_key = "turn-00"
    span = _execute_span(
        f"{turn_key}-ex", turn_key, call_id="c1", command_name="open_project", start_ns=T0
    )
    span.attributes[tracing.ATTR_CONTEXT_BEFORE] = _concrete("Project", "project-a")
    span.attributes[tracing.ATTR_CONTEXT_AFTER] = _concrete("Project", "project-b")
    _write(
        store,
        _turn_row(turn_key, record=_record(turn_key, refs=[("c1", 1, f"{turn_key}-ex")])),
        [span],
    )

    markers = diag.turn_markers(turn_key, store.get_spans(turn_key))
    assert markers.has(diag.MARKER_CONTEXT_NAVIGATION)
    assert markers.navigation["transitions"][0]["from"] == "Project"
    assert markers.navigation["transitions"][0]["to"] == "Project"


def test_concrete_handles_with_the_same_instance_are_unchanged(
    store: obs.ObservabilityStore,
) -> None:
    turn_key = "turn-00"
    span = _execute_span(
        f"{turn_key}-ex", turn_key, call_id="c1", command_name="rename", start_ns=T0
    )
    span.attributes[tracing.ATTR_CONTEXT_BEFORE] = _concrete("Project", "project-a")
    span.attributes[tracing.ATTR_CONTEXT_AFTER] = _concrete("Project", "project-a")
    _write(
        store,
        _turn_row(turn_key, record=_record(turn_key, refs=[("c1", 1, f"{turn_key}-ex")])),
        [span],
    )

    markers = diag.turn_markers(turn_key, store.get_spans(turn_key))
    assert not markers.has(diag.MARKER_CONTEXT_NAVIGATION)
    assert markers.navigation["type_only"] == 0
    assert markers.navigation["unknown"] == 0


def _navigation_state(store: obs.ObservabilityStore, before: dict, after: dict) -> dict:
    """Diagnose one dispatch carrying two fully projected context handles."""
    turn_key = "turn-00"
    span = _execute_span(
        f"{turn_key}-ex", turn_key, call_id="c1", command_name="rename", start_ns=T0
    )
    span.attributes[tracing.ATTR_CONTEXT_BEFORE] = before
    span.attributes[tracing.ATTR_CONTEXT_AFTER] = after
    _write(
        store,
        _turn_row(turn_key, record=_record(turn_key, refs=[("c1", 1, f"{turn_key}-ex")])),
        [span],
    )
    markers = diag.turn_markers(turn_key, store.get_spans(turn_key))
    return {
        "navigated": markers.has(diag.MARKER_CONTEXT_NAVIGATION),
        "type_only": markers.navigation["type_only"],
    }


def test_fingerprints_from_different_keys_are_not_compared(
    store: obs.ObservabilityStore,
) -> None:
    """Digests minted under different HMAC keys differ for reasons that have
    nothing to do with the instance; comparing them would invent a move."""
    state = _navigation_state(
        store, _concrete("Project", "project-a", "1"), _concrete("Project", "project-b", "2")
    )
    assert state["navigated"] is False
    assert state["type_only"] == 1


@pytest.mark.parametrize("key", ["projector_id", "projector_version", "security_scope_ref"])
def test_fingerprints_are_not_compared_across_incompatible_metadata(
    store: obs.ObservabilityStore, key: str
) -> None:
    """A digest is an identity only relative to what minted and scoped it.

    A different projector may have hashed a different instance key, and a
    different security scope is not asserting the same thing about the same
    instance -- so "the digests differ" would not mean the context moved.
    """
    before = _concrete("Project", "project-a")
    after = _concrete("Project", "project-b")
    after[key] = "something-else"

    state = _navigation_state(store, before, after)
    assert state["navigated"] is False, f"{key} disagrees; digests are not comparable"
    assert state["type_only"] == 1


@pytest.mark.parametrize("key", diag.FINGERPRINT_COMPATIBILITY_KEYS)
def test_fingerprints_are_not_compared_when_their_metadata_is_missing(
    store: obs.ObservabilityStore, key: str
) -> None:
    before = _concrete("Project", "project-a")
    after = _concrete("Project", "project-b")
    before[key] = None

    state = _navigation_state(store, before, after)
    assert state["navigated"] is False, f"missing {key} makes the digest unreadable"
    assert state["type_only"] == 1


def test_a_producers_explicit_navigation_flag_is_evidence(
    store: obs.ObservabilityStore,
) -> None:
    """History from a producer this build no longer contains is still evidence.

    The pilot corpus carries `auto_navigated` spans that no emitter in this tree
    writes. A producer stating that it navigated settles the question that
    type-only handles cannot.
    """
    spans = _one_dispatch(
        store,
        "turn-00",
        context_before="Project",
        context_after="Project",
        extra={"auto_navigated": True, "auto_navigation_rule": "enter-on-create"},
    )
    markers = diag.turn_markers("turn-00", spans)

    assert markers.has(diag.MARKER_CONTEXT_NAVIGATION)
    assert markers.navigation["type_only"] == 0


def test_a_navigation_flag_recorded_false_does_not_prove_the_negative(
    store: obs.ObservabilityStore,
) -> None:
    spans = _one_dispatch(
        store,
        "turn-00",
        context_before="Project",
        context_after="Project",
        extra={"auto_navigated": False},
    )
    markers = diag.turn_markers("turn-00", spans)

    assert not markers.has(diag.MARKER_CONTEXT_NAVIGATION)
    assert markers.navigation["type_only"] == 1, "still unproven, not proven unchanged"


def test_navigation_is_unknown_when_no_handle_was_recorded(
    store: obs.ObservabilityStore,
) -> None:
    """The pilot store's exception path: status error, no handles, no success."""
    turn_key = "turn-00"
    span = _execute_span(
        f"{turn_key}-ex",
        turn_key,
        call_id="c1",
        command_name="open_project",
        start_ns=T0,
        status=tracing.STATUS_ERROR,
        success=None,
        context_before=None,
        context_after=None,
    )
    _write(
        store,
        _turn_row(turn_key, record=_record(turn_key, refs=[("c1", 1, f"{turn_key}-ex")])),
        [span],
    )
    markers = diag.turn_markers(turn_key, store.get_spans(turn_key))

    assert not markers.has(diag.MARKER_CONTEXT_NAVIGATION)
    assert not markers.has(diag.MARKER_STEP_UNSUCCESSFUL), "absent success is not false"
    assert markers.has(diag.MARKER_STEP_ERROR), "the recorded status is a fact"
    assert markers.has(diag.MARKER_PARTIAL_EVIDENCE)
    assert markers.coverage["dispatches_without_context_handles"] == 1
    assert markers.coverage["dispatches_without_success"] == 1


def test_completed_turn_containing_an_unsuccessful_command_is_flagged(
    store: obs.ObservabilityStore,
) -> None:
    turn_key = "turn-00"
    good = _execute_span(
        f"{turn_key}-ex1", turn_key, call_id="c1", command_name="list_todos", start_ns=T0
    )
    bad = _execute_span(
        f"{turn_key}-ex2",
        turn_key,
        call_id="c2",
        command_name="add_todo",
        start_ns=T0 + 10,
        status=tracing.STATUS_ERROR,
        success=False,
    )
    record = _record(
        turn_key,
        refs=[("c1", 1, f"{turn_key}-ex1"), ("c2", 2, f"{turn_key}-ex2")],
        outputs=[_output("c1", "list_todos"), _output("c2", "add_todo", success=False)],
    )
    # The turn row itself says completed/success: the failure is inside it.
    _write(store, _turn_row(turn_key, record=record, status="completed", success=True), [good, bad])

    markers = diag.turn_markers(turn_key, store.get_spans(turn_key))
    assert markers.has(diag.MARKER_STEP_UNSUCCESSFUL)
    assert markers.evidence[diag.MARKER_STEP_UNSUCCESSFUL]["span_ids"] == [f"{turn_key}-ex2"]

    page = diag.search_turns(
        store,
        diag.TurnQuery(success=True, markers_any=(diag.MARKER_STEP_UNSUCCESSFUL,)),
        store_id=STORE_ID,
    )
    assert [row["turn_key"] for row in page.rows] == [turn_key]


def _record_only_failure_turn(store: obs.ObservabilityStore, turn_key: str) -> None:
    """One dispatch with a span, one whose span was dropped and failed."""
    span = _execute_span(
        f"{turn_key}-ex1", turn_key, call_id="c1", command_name="list_todos", start_ns=T0
    )
    record = _record(
        turn_key,
        refs=[("c1", 1, f"{turn_key}-ex1"), ("c2", 2, None)],
        outputs=[_output("c1", "list_todos"), _output("c2", "add_todo", success=False)],
    )
    _write(store, _turn_row(turn_key, record=record), [span])


def test_a_record_only_failure_is_found_by_default(
    store: obs.ObservabilityStore,
) -> None:
    """A dispatch whose span was dropped still recorded its outcome.

    A failure search that read only traces would answer "no failures" about a
    turn that recorded one -- the same false empty result as filtering after
    paging, so the record is read by default for a record-sensitive marker.
    """
    _record_only_failure_turn(store, "turn-00")

    page = diag.search_turns(
        store,
        diag.TurnQuery(markers_any=(diag.MARKER_STEP_UNSUCCESSFUL,)),
        store_id=STORE_ID,
    )
    assert [row["turn_key"] for row in page.rows] == ["turn-00"]
    assert page.basis == diag.BASIS_SPANS_AND_RECORD
    assert page.query["reads_record"] is True


def test_a_record_only_failure_is_found_beyond_the_first_page(
    store: obs.ObservabilityStore,
) -> None:
    for index in range(1, 9):
        _plain_turn(store, f"turn-{index:02d}")
    _record_only_failure_turn(store, "turn-00")

    page = diag.search_turns(
        store,
        diag.TurnQuery(markers_any=(diag.MARKER_STEP_UNSUCCESSFUL,), limit=2),
        store_id=STORE_ID,
    )
    assert [row["turn_key"] for row in page.rows] == ["turn-00"]
    assert page.total_matched == 1
    assert page.total_scanned == 9


def test_search_and_detail_agree_about_a_record_only_failure(
    store: obs.ObservabilityStore,
) -> None:
    """The same complete evidence must produce the same flags on both routes."""
    _record_only_failure_turn(store, "turn-00")

    page = diag.search_turns(
        store, diag.TurnQuery(include_record=True), store_id=STORE_ID
    )
    listed = set(page.rows[0]["markers"])
    detail = _diagnose(store, "turn-00").turns[0]

    assert diag.MARKER_STEP_UNSUCCESSFUL in listed
    assert listed <= set(detail.markers)


def test_the_cheaper_trace_only_scan_stays_available_and_says_so(
    store: obs.ObservabilityStore,
) -> None:
    """Opting out of the record read is explicit, and the basis reports it."""
    _record_only_failure_turn(store, "turn-00")

    page = diag.search_turns(
        store,
        diag.TurnQuery(
            markers_any=(diag.MARKER_STEP_UNSUCCESSFUL,), include_record=False
        ),
        store_id=STORE_ID,
    )
    assert page.total_matched == 0
    assert page.basis == diag.BASIS_SPANS
    assert page.query["reads_record"] is False


def test_a_query_with_no_record_sensitive_marker_does_not_pay_for_records(
    store: obs.ObservabilityStore,
) -> None:
    _navigating_turn(store, "turn-00")
    query = diag.TurnQuery(markers_any=(diag.MARKER_CONTEXT_NAVIGATION,))
    assert query.reads_record is False

    page = diag.search_turns(store, query, store_id=STORE_ID)
    assert page.basis == diag.BASIS_SPANS
    assert [row["turn_key"] for row in page.rows] == ["turn-00"]


# ----------------------------------------------------------------------
# .18.3 -- intent, parameters, retries and suspected loops
# ----------------------------------------------------------------------


def _reader(store: obs.ObservabilityStore) -> StoreExecutionReader:
    return StoreExecutionReader(STORE_ID, store)


def _diagnose(store: obs.ObservabilityStore, turn_key: str, **kwargs):
    return diag.diagnose_execution(
        ExecutionRef(store_id=STORE_ID, turn_keys=(turn_key,)),
        _reader(store),
        ledger=execution_ledger,
        cost_rollup=cost_rollup,
        **kwargs,
    )


def test_an_unresolved_intent_attempt_is_not_a_failure(
    store: obs.ObservabilityStore,
) -> None:
    """The parent-chain walk records `resolved=false` and is how routing works."""
    turn_key = "turn-00"
    execute = _execute_span(
        f"{turn_key}-ex", turn_key, call_id="c1", command_name="wildcard", start_ns=T0
    )
    walks = [
        _intent_span(
            f"{turn_key}-nlu{i}",
            turn_key,
            start_ns=T0 + i,
            parent_span_id=f"{turn_key}-ex",
            resolved=False,
        )
        for i in range(3)
    ]
    hit = _intent_span(
        f"{turn_key}-nlu-hit", turn_key, start_ns=T0 + 9, parent_span_id=f"{turn_key}-ex"
    )
    _write(
        store,
        _turn_row(turn_key, record=_record(turn_key, refs=[("c1", 1, f"{turn_key}-ex")])),
        [execute, *walks, hit],
    )

    markers = diag.turn_markers(turn_key, store.get_spans(turn_key))
    assert not markers.has(diag.MARKER_INTENT_AMBIGUOUS)
    assert not markers.has(diag.MARKER_INTENT_ERROR)
    assert markers.counts["intent_unresolved_attempts"] == 3, "reported, not flagged"
    assert markers.counts["intent_spans"] == 4


def test_recorded_ambiguity_and_intent_error_are_distinct_states(
    store: obs.ObservabilityStore,
) -> None:
    turn_key = "turn-00"
    execute = _execute_span(
        f"{turn_key}-ex", turn_key, call_id="c1", command_name="wildcard", start_ns=T0
    )
    ambiguous = _intent_span(
        f"{turn_key}-amb",
        turn_key,
        start_ns=T0 + 1,
        parent_span_id=f"{turn_key}-ex",
        resolved=False,
        ambiguous=True,
    )
    raised = _intent_span(
        f"{turn_key}-err",
        turn_key,
        start_ns=T0 + 2,
        parent_span_id=f"{turn_key}-ex",
        status=tracing.STATUS_ERROR,
    )
    _write(
        store,
        _turn_row(turn_key, record=_record(turn_key, refs=[("c1", 1, f"{turn_key}-ex")])),
        [execute, ambiguous, raised],
    )

    markers = diag.turn_markers(turn_key, store.get_spans(turn_key))
    assert markers.has(diag.MARKER_INTENT_AMBIGUOUS)
    assert markers.has(diag.MARKER_INTENT_ERROR)
    assert markers.evidence[diag.MARKER_INTENT_AMBIGUOUS]["span_ids"] == [f"{turn_key}-amb"]
    assert markers.evidence[diag.MARKER_INTENT_ERROR]["span_ids"] == [f"{turn_key}-err"]


def test_parameter_extraction_states_and_missing_retry_coverage(
    store: obs.ObservabilityStore,
) -> None:
    turn_key = "turn-00"
    execute = _execute_span(
        f"{turn_key}-ex", turn_key, call_id="c1", command_name="add_todo", start_ns=T0
    )
    invalid = _param_span(
        f"{turn_key}-p1",
        turn_key,
        start_ns=T0 + 1,
        parent_span_id=f"{turn_key}-ex",
        parameters_valid=False,
        missing_fields=["title"],
        retry_round=False,
    )
    retried = _param_span(
        f"{turn_key}-p2",
        turn_key,
        start_ns=T0 + 2,
        parent_span_id=f"{turn_key}-ex",
        retry_round=True,
    )
    # The pilot store's common shape: no retry_round recorded at all.
    unmeasured = _param_span(
        f"{turn_key}-p3", turn_key, start_ns=T0 + 3, parent_span_id=f"{turn_key}-ex"
    )
    _write(
        store,
        _turn_row(turn_key, record=_record(turn_key, refs=[("c1", 1, f"{turn_key}-ex")])),
        [execute, invalid, retried, unmeasured],
    )

    markers = diag.turn_markers(turn_key, store.get_spans(turn_key))
    assert markers.has(diag.MARKER_PARAM_INVALID)
    assert markers.has(diag.MARKER_PARAM_RETRY)
    assert not markers.has(diag.MARKER_PARAM_ERROR)
    # The unrecorded retry flag is reported in coverage rather than as a
    # `partial_evidence` chip: `retry_round` is absent often enough in real
    # traces (1701 of 5648 spans in the pilot store) that flagging it marked
    # 143 of 152 turns partial, which buries the turns whose navigation or
    # success is genuinely unreadable.
    assert markers.coverage["extractions_without_retry_flag"] == 1
    assert markers.coverage["parameter_extraction_spans"] == 3
    assert not markers.has(diag.MARKER_PARTIAL_EVIDENCE)


def test_a_recorded_retry_ordinal_is_reported_and_an_absent_one_is_not_invented(
    store: obs.ObservabilityStore,
) -> None:
    """`fix-8ko2`: which round, from the producer, or nothing.

    Counting extraction spans would produce a number that looks like an
    ordinal and is wrong exactly when it matters -- a dropped span makes the
    count smaller than the round it claims to name. A v1 span keeps an
    unrecorded round and is counted in coverage instead.
    """
    turn_key = "turn-00"
    execute = _execute_span(
        f"{turn_key}-ex", turn_key, call_id="c1", command_name="add_todo", start_ns=T0
    )
    first = _param_span(
        f"{turn_key}-p1", turn_key, start_ns=T0 + 1, parent_span_id=f"{turn_key}-ex",
        parameters_valid=False, missing_fields=["title"],
        retry_round=False, retry_round_ordinal=0,
    )
    second = _param_span(
        f"{turn_key}-p2", turn_key, start_ns=T0 + 2, parent_span_id=f"{turn_key}-ex",
        retry_round=True, retry_round_ordinal=1,
    )
    legacy = _param_span(
        f"{turn_key}-p3", turn_key, start_ns=T0 + 3, parent_span_id=f"{turn_key}-ex"
    )
    _write(
        store,
        _turn_row(turn_key, record=_record(turn_key, refs=[("c1", 1, f"{turn_key}-ex")])),
        [execute, first, second, legacy],
    )

    diagnosis = diag.diagnose_turn(
        turn_key,
        store.get_turn(turn_key),
        None,
        store.get_spans(turn_key),
        project_execution(
            ExecutionRef(store_id=STORE_ID, turn_keys=(turn_key,)),
            StoreExecutionReader(STORE_ID, store),
            ledger=execution_ledger,
            cost_rollup=cost_rollup,
        ).steps,
    )
    rounds = [
        event.detail["retry_round_ordinal"]
        for event in diagnosis.events
        if event.kind == diag.PHASE_PARAM
    ]
    assert rounds == [0, 1, None], "recorded rounds reported, the absent one left absent"
    assert diagnosis.span_markers.coverage["extractions_without_retry_flag"] == 1
    assert diagnosis.span_markers.coverage["extractions_without_retry_round"] == 1
    assert diagnosis.span_markers.coverage["parameter_extraction_spans"] == 3


def test_a_retry_whose_round_is_unrecorded_is_counted_apart_from_a_missing_flag(
    store: obs.ObservabilityStore,
) -> None:
    """Two different gaps, two different counts.

    A session restored from a continuation records `retry_round=True` -- it
    knows perfectly well this is a retry -- and OMITS the ordinal, because the
    persisted state carries the stored parameters but not their round
    (`fix-7gp9`). Rolling that into `extractions_without_retry_flag` would say
    the flag is missing, which is false, and would leave a consumer asking
    "which retry was this" with no count at all.
    """
    turn_key = "turn-00"
    execute = _execute_span(
        f"{turn_key}-ex", turn_key, call_id="c1", command_name="add_todo", start_ns=T0
    )
    restored_retry = _param_span(
        f"{turn_key}-p1", turn_key, start_ns=T0 + 1, parent_span_id=f"{turn_key}-ex",
        retry_round=True,
    )
    _write(
        store,
        _turn_row(turn_key, record=_record(turn_key, refs=[("c1", 1, f"{turn_key}-ex")])),
        [execute, restored_retry],
    )
    markers = diag.turn_markers(turn_key, store.get_spans(turn_key))

    assert markers.has(diag.MARKER_PARAM_RETRY), "it is still a retry"
    assert markers.coverage["extractions_without_retry_flag"] == 0, "the flag is there"
    assert markers.coverage["extractions_without_retry_round"] == 1, "the round is not"


def test_an_ordinal_beyond_the_first_is_a_retry_even_without_the_flag(
    store: obs.ObservabilityStore,
) -> None:
    """The ordinal stands on its own: round 2 is a retry however the flag reads."""
    turn_key = "turn-00"
    execute = _execute_span(
        f"{turn_key}-ex", turn_key, call_id="c1", command_name="add_todo", start_ns=T0
    )
    retried = _param_span(
        f"{turn_key}-p1", turn_key, start_ns=T0 + 1, parent_span_id=f"{turn_key}-ex",
        retry_round_ordinal=2,
    )
    _write(
        store,
        _turn_row(turn_key, record=_record(turn_key, refs=[("c1", 1, f"{turn_key}-ex")])),
        [execute, retried],
    )
    markers = diag.turn_markers(turn_key, store.get_spans(turn_key))
    assert markers.has(diag.MARKER_PARAM_RETRY)

    # And round 0 is the first attempt, which is not a retry.
    first = _param_span(
        "turn-01-p1", "turn-01", start_ns=T0, parent_span_id="turn-01-ex",
        retry_round_ordinal=0,
    )
    _write(
        store,
        _turn_row("turn-01", record=_record("turn-01", refs=[])),
        [
            _execute_span("turn-01-ex", "turn-01", call_id="c1",
                          command_name="add_todo", start_ns=T0),
            first,
        ],
    )
    assert not diag.turn_markers(
        "turn-01", store.get_spans("turn-01")
    ).has(diag.MARKER_PARAM_RETRY)


@pytest.mark.parametrize("bad", [True, 2.0, "3", -1, None])
def test_a_nonsense_retry_ordinal_is_not_coerced_into_a_round(
    store: obs.ObservabilityStore, bad
) -> None:
    """`True` is not round 1 and `2.0` is not an attempt number.

    A value this module cannot read is an unrecorded round, counted in
    coverage, not a round it picks the nearest integer for.
    """
    turn_key = "turn-00"
    span = _param_span(
        f"{turn_key}-p1", turn_key, start_ns=T0, parent_span_id=f"{turn_key}-ex"
    )
    span.attributes["retry_round_ordinal"] = bad
    _write(
        store,
        _turn_row(turn_key, record=_record(turn_key, refs=[])),
        [
            _execute_span(f"{turn_key}-ex", turn_key, call_id="c1",
                          command_name="add_todo", start_ns=T0),
            span,
        ],
    )
    markers = diag.turn_markers(turn_key, store.get_spans(turn_key))
    assert markers.coverage["extractions_without_retry_flag"] == 1
    assert markers.coverage["extractions_without_retry_round"] == 1
    assert not markers.has(diag.MARKER_PARAM_RETRY)


def test_awaiting_user_is_a_suspension_not_a_failure(store: obs.ObservabilityStore) -> None:
    turn_key = "turn-00"
    execute = _execute_span(
        f"{turn_key}-ex", turn_key, call_id="c1", command_name="add_todo", start_ns=T0
    )
    ask = _ask_user_span(f"{turn_key}-ask", turn_key, start_ns=T0 + 5)
    _write(
        store,
        _turn_row(
            turn_key,
            record=_record(turn_key, refs=[("c1", 1, f"{turn_key}-ex")]),
            status="awaiting_user",
            success=False,
        ),
        [execute, ask],
    )

    markers = diag.turn_markers(turn_key, store.get_spans(turn_key))
    assert markers.has(diag.MARKER_AWAITING_USER)
    assert not markers.has(diag.MARKER_STEP_UNSUCCESSFUL)
    assert not markers.has(diag.MARKER_SUSPECTED_LOOP)


def _repeating_turn(
    store: obs.ObservabilityStore, turn_key: str, *, failing: bool
) -> None:
    """Four identical dispatches back to back, optionally recorded as failing."""
    spans = []
    refs = []
    for index in range(4):
        span_id = f"{turn_key}-ex{index}"
        spans.append(
            _execute_span(
                span_id,
                turn_key,
                call_id=f"c{index}",
                command_name="search",
                start_ns=T0 + index * 1_000_000,
                parameters={"query": "same"},
                status=tracing.STATUS_ERROR if failing else tracing.STATUS_OK,
                success=False if failing else True,
            )
        )
        refs.append((f"c{index}", index + 1, span_id))
    _write(store, _turn_row(turn_key, record=_record(turn_key, refs=refs)), spans)


def test_repetition_alone_is_not_a_loop(store: obs.ObservabilityStore) -> None:
    """Four identical successful dispatches: repeated, and nothing more."""
    _repeating_turn(store, "turn-00", failing=False)
    markers = diag.turn_markers("turn-00", store.get_spans("turn-00"))

    assert markers.has(diag.MARKER_REPEATED_COMMAND)
    assert not markers.has(diag.MARKER_SUSPECTED_LOOP)
    group = markers.repeats[0]
    assert group.count == 4
    assert group.basis == "command+context+parameters"
    assert group.suspected_loop is False
    assert group.trouble == ()


def test_repetition_with_colocated_trouble_is_a_suspected_loop(
    store: obs.ObservabilityStore,
) -> None:
    _repeating_turn(store, "turn-00", failing=True)
    markers = diag.turn_markers("turn-00", store.get_spans("turn-00"))

    assert markers.has(diag.MARKER_SUSPECTED_LOOP)
    group = markers.repeats[0]
    assert group.count == 4
    assert diag.MARKER_STEP_UNSUCCESSFUL in group.trouble
    assert len(group.span_ids) == 4, "every occurrence is reachable as evidence"


def test_unrelated_trouble_elsewhere_in_the_turn_does_not_make_a_loop(
    store: obs.ObservabilityStore,
) -> None:
    """The false positive the heuristic is bounded to avoid.

    A long turn repeats one harmless command AND separately fails a different
    one. Joining them would call almost every agent turn a loop -- measured at
    67 of 152 turns in the pilot store before this rule was tightened.
    """
    turn_key = "turn-00"
    spans = []
    refs = []
    for index in range(4):
        span_id = f"{turn_key}-ok{index}"
        spans.append(
            _execute_span(
                span_id,
                turn_key,
                call_id=f"c{index}",
                command_name="search",
                start_ns=T0 + index * 1_000_000,
                parameters={"query": "same"},
            )
        )
        refs.append((f"c{index}", index + 1, span_id))
    spans.append(
        _execute_span(
            f"{turn_key}-bad",
            turn_key,
            call_id="cbad",
            command_name="delete_everything",
            start_ns=T0 + 9_000_000,
            status=tracing.STATUS_ERROR,
            success=False,
        )
    )
    refs.append(("cbad", 5, f"{turn_key}-bad"))
    _write(store, _turn_row(turn_key, record=_record(turn_key, refs=refs)), spans)

    markers = diag.turn_markers(turn_key, store.get_spans(turn_key))
    assert markers.has(diag.MARKER_REPEATED_COMMAND)
    assert markers.has(diag.MARKER_STEP_UNSUCCESSFUL)
    assert not markers.has(diag.MARKER_SUSPECTED_LOOP)


def test_repeats_outside_the_window_are_not_a_group(store: obs.ObservabilityStore) -> None:
    turn_key = "turn-00"
    spans = []
    refs = []
    names = ["search", "a", "b", "c", "d", "e", "f", "search", "g", "h", "i", "j", "k", "search"]
    for index, name in enumerate(names):
        span_id = f"{turn_key}-ex{index:02d}"
        spans.append(
            _execute_span(
                span_id,
                turn_key,
                call_id=f"c{index}",
                command_name=name,
                start_ns=T0 + index * 1_000_000,
                parameters={"query": "same"} if name == "search" else {"n": index},
            )
        )
        refs.append((f"c{index}", index + 1, span_id))
    _write(store, _turn_row(turn_key, record=_record(turn_key, refs=refs)), spans)

    markers = diag.turn_markers(turn_key, store.get_spans(turn_key))
    assert not markers.has(diag.MARKER_REPEATED_COMMAND)
    assert markers.repeats == ()


def test_the_loop_policy_travels_in_the_output(store: obs.ObservabilityStore) -> None:
    """`.18.3` requires the heuristic's bounds be documented in the output."""
    _repeating_turn(store, "turn-00", failing=True)
    markers = diag.turn_markers("turn-00", store.get_spans("turn-00"))
    policy = markers.loop_policy

    assert policy["name"] == "repeated-dispatch-window/1"
    assert policy["min_repeats"] == 3
    assert policy["window_steps"] == 6
    assert policy["require_recorded_trouble"] is True

    page = diag.search_turns(store, diag.TurnQuery(limit=1), store_id=STORE_ID)
    assert page.query["loop_policy"] == policy


def test_a_looser_policy_is_the_callers_explicit_choice(
    store: obs.ObservabilityStore,
) -> None:
    _repeating_turn(store, "turn-00", failing=False)
    loose = diag.LoopPolicy(require_recorded_trouble=False)
    markers = diag.turn_markers("turn-00", store.get_spans("turn-00"), loop_policy=loose)

    assert markers.has(diag.MARKER_SUSPECTED_LOOP)
    assert markers.loop_policy["require_recorded_trouble"] is False


def test_an_incoherent_policy_is_refused() -> None:
    with pytest.raises(diag.InvalidTurnQuery):
        diag.LoopPolicy(min_repeats=1)
    with pytest.raises(diag.InvalidTurnQuery):
        diag.LoopPolicy(min_repeats=4, window_steps=2)


# ----------------------------------------------------------------------
# Step-level projection: one ledger, exact evidence links
# ----------------------------------------------------------------------


def test_nested_decisions_link_to_their_parent_executor_step(
    store: obs.ObservabilityStore,
) -> None:
    turn_key = "turn-00"
    first = _execute_span(
        f"{turn_key}-ex1", turn_key, call_id="c1", command_name="wildcard", start_ns=T0
    )
    second = _execute_span(
        f"{turn_key}-ex2",
        turn_key,
        call_id="c2",
        command_name="add_todo",
        start_ns=T0 + 5_000_000,
        context_before="Workspace",
        context_after="Project",
    )
    ambiguous = _intent_span(
        f"{turn_key}-amb",
        turn_key,
        start_ns=T0 + 1,
        parent_span_id=f"{turn_key}-ex1",
        ambiguous=True,
        resolved=False,
    )
    invalid = _param_span(
        f"{turn_key}-p1",
        turn_key,
        start_ns=T0 + 5_000_001,
        parent_span_id=f"{turn_key}-ex2",
        parameters_valid=False,
    )
    record = _record(
        turn_key,
        refs=[("c1", 1, f"{turn_key}-ex1"), ("c2", 2, f"{turn_key}-ex2")],
        outputs=[_output("c1", "wildcard"), _output("c2", "add_todo")],
    )
    _write(store, _turn_row(turn_key, record=record), [first, second, ambiguous, invalid])

    diagnosis = _diagnose(store, turn_key)
    turn = diagnosis.turns[0]
    steps = {step.step.command_call_id: step for step in turn.steps}

    assert diag.MARKER_INTENT_AMBIGUOUS in steps["c1"].markers
    assert [event.span_id for event in steps["c1"].events] == [f"{turn_key}-amb"]
    assert steps["c1"].events[0].owner_span_id == f"{turn_key}-ex1"

    assert diag.MARKER_PARAM_INVALID in steps["c2"].markers
    assert diag.MARKER_CONTEXT_NAVIGATION in steps["c2"].markers
    assert steps["c2"].navigation["from"] == "Workspace"
    assert steps["c2"].navigation["to"] == "Project"
    assert steps["c2"].events[0].span_id == f"{turn_key}-p1"

    # The anchor a comment attaches to names the exact span, in the vocabulary
    # `add_human_feedback` already validates.
    anchor = steps["c2"].anchor
    assert anchor["target_kind"] == "step"
    assert anchor["span_ids"] == [f"{turn_key}-ex2"]
    assert anchor["store_id"] == STORE_ID
    assert anchor["anchorable"] is True


def test_the_diagnosis_reuses_the_comparison_ledger_rather_than_a_second_one(
    store: obs.ObservabilityStore,
) -> None:
    _repeating_turn(store, "turn-00", failing=True)
    ref = ExecutionRef(store_id=STORE_ID, turn_keys=("turn-00",))
    projection = project_execution(
        ref, _reader(store), ledger=execution_ledger, cost_rollup=cost_rollup
    )
    diagnosis = _diagnose(store, "turn-00")

    assert [step.step for step in diagnosis.turns[0].steps] == list(projection.steps)
    assert diagnosis.projection.cost == projection.cost


def test_a_projection_supplied_by_the_caller_is_not_re_read(
    store: obs.ObservabilityStore,
) -> None:
    _navigating_turn(store, "turn-00")
    ref = ExecutionRef(store_id=STORE_ID, turn_keys=("turn-00",))
    projection = project_execution(
        ref, _reader(store), ledger=execution_ledger, cost_rollup=cost_rollup
    )
    diagnosis = diag.diagnose_execution(
        ref, _reader(store), projection=projection, ledger=execution_ledger
    )
    assert diagnosis.projection is projection
    assert diagnosis.turns[0].markers


def test_an_unreadable_turn_leaves_the_readable_ones_diagnosed(
    store: obs.ObservabilityStore,
) -> None:
    _navigating_turn(store, "turn-00")
    ref = ExecutionRef(store_id=STORE_ID, turn_keys=("turn-00", "turn-missing"))
    diagnosis = diag.diagnose_execution(
        ref, _reader(store), ledger=execution_ledger, cost_rollup=cost_rollup
    )

    assert [turn.turn_key for turn in diagnosis.turns] == ["turn-00"]
    assert diagnosis.coverage["turns_unavailable"] == 1
    assert diag.MARKER_PARTIAL_EVIDENCE in diagnosis.markers


def test_list_markers_are_a_subset_of_the_opened_turns(
    store: obs.ObservabilityStore,
) -> None:
    """What a list finds and what a detail view shows must not disagree."""
    turn_key = "turn-00"
    span = _execute_span(
        f"{turn_key}-ex1",
        turn_key,
        call_id="c1",
        command_name="open_project",
        start_ns=T0,
        context_before="Workspace",
        context_after="Project",
    )
    record = _record(
        turn_key,
        refs=[("c1", 1, f"{turn_key}-ex1"), ("c2", 2, None)],
        outputs=[_output("c1", "open_project"), _output("c2", "add_todo", success=False)],
    )
    _write(store, _turn_row(turn_key, record=record), [span])

    page = diag.search_turns(store, diag.TurnQuery(), store_id=STORE_ID)
    listed = set(page.rows[0]["markers"])
    detail = _diagnose(store, turn_key).turns[0]

    assert listed <= set(detail.markers)
    assert detail.span_markers.markers == tuple(sorted(listed, key=diag.MARKER_ORDER.index))
    # And the difference is named rather than left for a reader to notice.
    assert diag.MARKER_STEP_UNSUCCESSFUL in detail.markers_only_in_record
    assert detail.coverage["steps_without_span"] == 1


def test_unrecognized_span_kinds_are_reported_not_ignored(
    store: obs.ObservabilityStore,
) -> None:
    """The pilot store holds span names no emitter in this tree writes."""
    turn_key = "turn-00"
    execute = _execute_span(
        f"{turn_key}-ex", turn_key, call_id="c1", command_name="list_todos", start_ns=T0
    )
    foreign = tracing.Span(
        span_id=f"{turn_key}-foreign",
        trace_id=turn_key,
        parent_span_id=f"{turn_key}-ex",
        name="fw.evidence.read",
        kind=tracing.KIND_INTERNAL,
        channel_id="channel-1",
        start_ns=T0 + 1,
        end_ns=T0 + 2,
        status=tracing.STATUS_OK,
        attributes={"tool": "grep"},
    )
    _write(
        store,
        _turn_row(turn_key, record=_record(turn_key, refs=[("c1", 1, f"{turn_key}-ex")])),
        [execute, foreign],
    )

    markers = diag.turn_markers(turn_key, store.get_spans(turn_key))
    assert markers.coverage["unrecognized_span_names"] == ["fw.evidence.read"]
    assert markers.has(diag.MARKER_PARTIAL_EVIDENCE)


def test_a_query_testing_markers_without_them_is_refused_not_answered_false() -> None:
    query = diag.TurnQuery(markers_any=(diag.MARKER_SUSPECTED_LOOP,))
    with pytest.raises(diag.InvalidTurnQuery):
        diag.turn_matches({"turn_key": "turn-00"}, None, query)


def test_query_bounds_are_validated() -> None:
    with pytest.raises(diag.InvalidTurnQuery):
        diag.TurnQuery(limit=0)
    with pytest.raises(diag.InvalidTurnQuery):
        diag.TurnQuery(offset=-1)
    with pytest.raises(diag.InvalidTurnQuery):
        diag.TurnQuery(low_confidence_below=-0.1)
    with pytest.raises(diag.InvalidTurnQuery):
        diag.TurnQuery(markers_any=("not_a_marker",))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"limit": 2.5},
        {"limit": True},
        {"limit": diag.MAX_PAGE_LIMIT + 1},
        {"offset": 1.0},
        {"scan_limit": 0},
        {"scan_limit": "50"},
        {"attempt": 1.5},
        {"conversation_id": True},
        {"low_confidence_below": float("nan")},
        {"low_confidence_below": float("inf")},
        {"low_confidence_below": "0.2"},
        {"low_confidence_below": True},
        {"markers_any": diag.MARKER_SUSPECTED_LOOP},
        {"markers_all": (None,)},
        {"resume_after": ""},
    ],
)
def test_malformed_query_values_are_refused_not_coerced(kwargs: dict) -> None:
    """A bad bound is refused here, naming the value.

    Left alone, `limit=2.5` reaches SQLite as a float and `low_confidence_below
    = nan` silently matches nothing however low a recorded margin was -- both
    surface at a route as an opaque error or a wrong answer.
    """
    with pytest.raises(diag.InvalidTurnQuery):
        diag.TurnQuery(**kwargs)


# ----------------------------------------------------------------------
# Derived-turn cache (markers + page stamps)
# ----------------------------------------------------------------------


def test_search_without_page_stamps_keeps_pre_stamp_row_keys(
    store: obs.ObservabilityStore,
) -> None:
    """Without `page_stamps`, rows stay stamp-free (agent-facing contract)."""
    _navigating_turn(store, "turn-stamp-free")
    page = diag.search_turns(
        store, diag.TurnQuery(limit=10), store_id=STORE_ID
    )
    assert page.rows
    stamp_keys = {"llm_calls_cut_at_limit", "llm_cost"}
    for row in page.rows:
        assert stamp_keys.isdisjoint(row.keys())
        assert "markers" in row
        assert "diagnosis" in row


def test_derived_cache_matches_uncached_search(store: obs.ObservabilityStore) -> None:
    """Cached and uncached complete scans must agree byte-for-byte on the page."""
    from fastworkflow.observability.derived_cache import DerivedTurnCache
    from fastworkflow.run_chatbot.server import turn_span_stamps

    for index in range(6):
        _navigating_turn(store, f"turn-{index:02d}")
    for index in range(6, 10):
        _plain_turn(store, f"turn-{index:02d}")

    query = diag.TurnQuery(
        markers_any=(diag.MARKER_CONTEXT_NAVIGATION,),
        limit=4,
        low_confidence_below=0.25,
    )
    uncached = diag.search_turns(
        store, query, store_id=STORE_ID, page_stamps=turn_span_stamps
    )
    cache = DerivedTurnCache(max_entries=64, max_bytes=1024 * 1024)
    cold = diag.search_turns(
        store,
        query,
        store_id=STORE_ID,
        derived_cache=cache,
        page_stamps=turn_span_stamps,
    )
    warm = diag.search_turns(
        store,
        query,
        store_id=STORE_ID,
        derived_cache=cache,
        page_stamps=turn_span_stamps,
    )
    assert cold.as_dict() == uncached.as_dict()
    assert warm.as_dict() == uncached.as_dict()
    assert len(cache) > 0


def test_derived_cache_invalidates_when_a_span_is_appended(
    store: obs.ObservabilityStore,
) -> None:
    from fastworkflow.observability.derived_cache import DerivedTurnCache

    turn_key = "turn-cache-1"
    _plain_turn(store, turn_key)
    cache = DerivedTurnCache(max_entries=32, max_bytes=1024 * 1024)
    query = diag.TurnQuery(limit=10)
    first = diag.search_turns(
        store, query, store_id=STORE_ID, derived_cache=cache
    )
    assert first.rows[0]["turn_key"] == turn_key
    before_markers = first.rows[0]["markers"]

    # Appending a span changes COUNT(*)/MAX(rowid); the next scan must miss
    # the stale entry and see the new navigation marker.
    span = _execute_span(
        f"{turn_key}-nav",
        turn_key,
        call_id="c-nav",
        command_name="open_project",
        start_ns=T0 + 1,
        context_before="Workspace",
        context_after="Project",
    )
    with store._connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        store.upsert_span_rows(conn, [span], store._store_redactor())
        conn.commit()

    second = diag.search_turns(
        store, query, store_id=STORE_ID, derived_cache=cache
    )
    assert diag.MARKER_CONTEXT_NAVIGATION in second.rows[0]["markers"]
    assert second.rows[0]["markers"] != before_markers


def test_derived_cache_warm_scan_reuses_entries_and_recomputes_only_changed(
    store: obs.ObservabilityStore,
) -> None:
    """A hit hands back the stored entry; a miss puts a freshly built one.

    Equal results alone cannot tell a warm hit from a silent recompute, so
    entry identity is what proves the warm scan skipped the span fetch.
    """
    _plain_turn(store, "turn-warm-a")
    _plain_turn(store, "turn-warm-b")
    cache = DerivedTurnCache(max_entries=32, max_bytes=1024 * 1024)
    query = diag.TurnQuery(limit=10)

    def scan() -> diag.TurnSearchPage:
        return diag.search_turns(
            store,
            query,
            store_id=STORE_ID,
            derived_cache=cache,
            page_stamps=turn_span_stamps,
        )

    cold = scan()
    assert len(cache) == 2
    cold_entries = dict(cache._entries)

    warm = scan()
    assert warm.as_dict() == cold.as_dict()
    assert len(cache) == 2
    assert all(cache._entries[key] is entry for key, entry in cold_entries.items())

    span = _execute_span(
        "turn-warm-a-nav",
        "turn-warm-a",
        call_id="c-nav",
        command_name="open_project",
        start_ns=T0 + 1,
        context_before="Workspace",
        context_after="Project",
    )
    with store._connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        store.upsert_span_rows(conn, [span], store._store_redactor())
        conn.commit()

    changed = scan()
    by_key = {row["turn_key"]: row for row in changed.rows}
    assert diag.MARKER_CONTEXT_NAVIGATION in by_key["turn-warm-a"]["markers"]
    # The stale entry is left to age out; the changed turn adds a new key and
    # the untouched turn is still served from its original entry.
    assert len(cache) == 3
    assert sum(
        cache._entries.get(key) is entry for key, entry in cold_entries.items()
    ) == 2


def test_derived_cache_invalidates_when_turn_row_status_changes(
    store: obs.ObservabilityStore,
) -> None:
    """`turn_markers` reads `status` from the turn row; row-only edits must miss."""
    from fastworkflow.observability.derived_cache import DerivedTurnCache

    turn_key = "turn-status-1"
    _plain_turn(store, turn_key)
    cache = DerivedTurnCache(max_entries=32, max_bytes=1024 * 1024)
    query = diag.TurnQuery(limit=10)
    first = diag.search_turns(
        store, query, store_id=STORE_ID, derived_cache=cache
    )
    assert diag.MARKER_AWAITING_USER not in first.rows[0]["markers"]

    with store._connect() as conn:
        conn.execute(
            "UPDATE turns SET status=? WHERE turn_key=?",
            ("awaiting_user", turn_key),
        )
        conn.commit()

    second = diag.search_turns(
        store, query, store_id=STORE_ID, derived_cache=cache
    )
    assert diag.MARKER_AWAITING_USER in second.rows[0]["markers"]


def test_derived_cache_small_cap_still_matches_uncached(
    store: obs.ObservabilityStore,
) -> None:
    """A complete scan over more turns than the cap must still match uncached."""
    from fastworkflow.observability.derived_cache import DerivedTurnCache

    for index in range(12):
        _plain_turn(store, f"turn-{index:02d}")
    query = diag.TurnQuery(limit=20)
    uncached = diag.search_turns(store, query, store_id=STORE_ID)
    cache = DerivedTurnCache(max_entries=3, max_bytes=1024 * 1024)
    with_cache = diag.search_turns(
        store, query, store_id=STORE_ID, derived_cache=cache
    )
    assert with_cache.as_dict() == uncached.as_dict()
    assert len(cache) <= 3


def test_derived_cache_is_bounded() -> None:
    from fastworkflow.observability.derived_cache import (
        DerivedTurnCache,
        DerivedTurnEntry,
    )

    cache = DerivedTurnCache(max_entries=3, max_bytes=1024 * 1024)
    for index in range(10):
        cache.put(
            ("store", f"turn-{index}", 1, index, (None,), None, (), False),
            DerivedTurnEntry(
                markers={"i": index},
                stamps={"llm_calls_cut_at_limit": 0},
                approx_bytes=64,
            ),
        )
    assert len(cache) == 3
    assert cache.get(("store", "turn-0", 1, 0, (None,), None, (), False)) is None
    assert cache.get(("store", "turn-9", 1, 9, (None,), None, (), False)) is not None

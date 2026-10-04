"""The expensive recorded LLM calls of each compared side (`fix-9eg.3.1.2`).

Integration throughout, per `.cursor/rules/testing_rules.mdc`: a real
`ObservabilityStore` on disk, real span rows through `upsert_span_rows`, a real
benchmark registration and `ExperimentController`, the real selection API, a
real `ChatbotServer` over a real socket and the shipped page in a real DOM. No
Mock fixtures: the claim is that the list a person reads and the roll-up a
coding agent reads are ONE accounting of ONE recorded execution, and a stand-in
for the store or the API would let those drift without the test noticing.

The recorded world is built out of the shapes that make this list easy to get
wrong:

- a dear call, a cheap call, a call that recorded a cost of exactly `0` and a
  call that recorded no cost at all -- four rows whose order and labels must
  keep "spent nothing" apart from "nobody counted";
- one call recorded at TWO levels (a wrapper `fw.llm.call` quoting its own
  descendant's provider response), which is one row and one charge;
- two sibling calls quoting ONE provider response, which are two rows and one
  charge, with the second saying where its money went;
- two recorded distillation passes over one turn, so a pass-scoped side lists
  that pass's calls and says which pass it is;
- twenty-one priced calls with an unpriced and a shared one below them, which is
  where a bounded list would hide the existence of the last two;
- an attempt that dispatched a command and called no LLM at all, which is a
  different finding from an attempt whose calls could not be listed.

What must not happen: a raw per-span sum (the left side's spans carry 0.7520 of
recorded cost and the right's carry 1.5000 while the right really spent 0.7500),
an absence rendered as a zero, or a link that opens the other side's evidence.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path

import pytest

from fastworkflow import tracing
from fastworkflow.benchmark import setup
from fastworkflow.experiment.runner import ExperimentController
from fastworkflow.observability import store as obs
from fastworkflow.observability.comparison import (
    USAGE_SHARED_RESPONSE,
    USAGE_UNRECORDED,
    ExecutionRef,
    StoreExecutionReader,
    project_execution,
    usage_rollup,
)
from fastworkflow.run_chatbot import selection_api
from fastworkflow.run_chatbot import server as run_chatbot_server
from tests.test_execution_comparison import (
    _execute_span,
    _output,
    _record,
    _turn_row,
    _write,
)
from tests.test_usage_and_cache_rollups import COMPLETE_USAGE, T0, _llm_call, _rows

CHEAP_USAGE = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
ZERO_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

LEFT_TURN = "cost-a1-t1"
RIGHT_TURN = "cost-a2-t1"
MANY_TURN = "cost-a3-t1"
SILENT_TURN = "cost-a4-t1"
# Named by a reference and held by no store: the turn a sealed manifest still
# lists after its archive lost it, or one pruned between a reference being
# built and the projection being read.
PRUNED_TURN = "cost-a1-t2"

# More calls than one side lists at once, so the bounded list and its local
# "show all" are exercised against a real payload rather than a contrived one.
MANY_PRICED = 21


def _pass_span(span_id: str, turn_key: str, pass_id: str, *, start_ns: int):
    """A recorded distillation pass, stamped the way `distillation.py` stamps it.

    The stamp is on the pass span only; membership of everything under it is
    resolved by walking ancestry, which is what `discover_pass_selectors` and
    `_pass_scope` do with a real turn.
    """
    return tracing.Span(
        span_id=span_id,
        trace_id=turn_key,
        parent_span_id=None,
        name=tracing.SPAN_DISTILLATION_PASS,
        kind=tracing.KIND_INTERNAL,
        channel_id="channel-1",
        start_ns=start_ns,
        end_ns=start_ns + 2_000_000,
        status=tracing.STATUS_OK,
        attributes={tracing.ATTR_PASS: pass_id},
    )


def _left_spans() -> list:
    """Attempt 1: four calls whose costs are dear, cheap, zero and unrecorded.

    The dear call is recorded inside the teacher pass and the cheap one inside
    the student pass, so the same turn answers a whole-run read and a pass-scoped
    read without either being inferred.
    """
    return [
        _execute_span(
            f"ex-{LEFT_TURN}", LEFT_TURN, call_id="c1", command_name="add_todo",
            start_ns=T0, parameters={"title": "x"},
        ),
        _pass_span("pass-teacher", LEFT_TURN, tracing.PASS_TEACHER, start_ns=T0 + 100),
        _pass_span("pass-student", LEFT_TURN, tracing.PASS_STUDENT, start_ns=T0 + 200),
        _llm_call(
            "a1-dear", LEFT_TURN, start_ns=T0 + 1_000, usage=COMPLETE_USAGE,
            cost=0.75, cache_hit=True, history_uuid="r-dear",
            parent_span_id="pass-teacher",
        ),
        _llm_call(
            "a1-cheap", LEFT_TURN, start_ns=T0 + 2_000, usage=CHEAP_USAGE,
            cost=0.002, cache_hit=False, history_uuid="r-cheap",
            parent_span_id="pass-student",
        ),
        # A recorded zero: this call really spent nothing, and it is not the
        # same fact as the one below it.
        _llm_call(
            "a1-zero", LEFT_TURN, start_ns=T0 + 3_000, usage=ZERO_USAGE,
            cost=0.0, cache_hit=False, history_uuid="r-zero",
        ),
        # Nothing recorded a cost, tokens or a cache flag. Three unknowns.
        _llm_call(
            "a1-silent", LEFT_TURN, start_ns=T0 + 4_000, history_uuid="r-silent",
        ),
    ]


def _right_spans() -> list:
    """Attempt 2: one call recorded twice, and two calls quoting one response.

    `a2-outer` wraps `a2-inner` and quotes its provider response, so the pair is
    ONE call charged once. `a2-first` and `a2-second` are siblings quoting one
    response: two real calls, one charge, the later one accounted under the
    earlier.
    """
    return [
        _execute_span(
            f"ex-{RIGHT_TURN}", RIGHT_TURN, call_id="c1", command_name="add_todo",
            start_ns=T0, parameters={"title": "x"},
        ),
        _llm_call(
            "a2-outer", RIGHT_TURN, start_ns=T0 + 1_000, usage=COMPLETE_USAGE,
            cost=0.5, history_uuid="r-wrap",
        ),
        _llm_call(
            "a2-inner", RIGHT_TURN, start_ns=T0 + 1_100, usage=COMPLETE_USAGE,
            cost=0.5, cache_hit=False, history_uuid="r-wrap",
            parent_span_id="a2-outer",
        ),
        _llm_call(
            "a2-first", RIGHT_TURN, start_ns=T0 + 2_000, usage=COMPLETE_USAGE,
            cost=0.25, cache_hit=True, history_uuid="r-shared",
        ),
        _llm_call(
            "a2-second", RIGHT_TURN, start_ns=T0 + 3_000, usage=COMPLETE_USAGE,
            cost=0.25, cache_hit=True, history_uuid="r-shared",
        ),
    ]


def _many_spans() -> list:
    """Attempt 3: more priced calls than the list shows, plus the two kinds of
    call that sort last.

    The point of the shape: the unknown-cost call and the shared-response call
    are BELOW twenty-one priced ones, so a list that simply truncated would hide
    the existence of both. They stay counted, named by kind, and reachable.
    """
    spans = [
        _execute_span(
            f"ex-{MANY_TURN}", MANY_TURN, call_id="c1", command_name="add_todo",
            start_ns=T0, parameters={"title": "x"},
        )
    ]
    for index in range(MANY_PRICED):
        spans.append(
            _llm_call(
                f"a3-{index:02d}", MANY_TURN, start_ns=T0 + 1_000 + index,
                usage=COMPLETE_USAGE,
                # Descending, so the dearest is `a3-00` and the order on screen
                # is not the order they were recorded in.
                cost=round(0.01 * (MANY_PRICED - index), 4),
                cache_hit=index % 2 == 0,
                history_uuid=f"r3-{index:02d}",
            )
        )
    spans.append(
        _llm_call("a3-silent", MANY_TURN, start_ns=T0 + 9_000,
                  history_uuid="r3-silent")
    )
    spans.append(
        # Quotes the dearest call's provider response: a real call that carries
        # no charge of its own.
        _llm_call("a3-shared", MANY_TURN, start_ns=T0 + 9_100,
                  usage=COMPLETE_USAGE, cost=0.21, cache_hit=True,
                  history_uuid="r3-00")
    )
    return spans


def _silent_spans() -> list:
    """Attempt 4: a dispatch and no LLM call at all.

    The other half of the absence: this side really made no recorded LLM call,
    which is not the same finding as a side whose calls could not be listed.
    """
    return [
        _execute_span(
            f"ex-{SILENT_TURN}", SILENT_TURN, call_id="c1",
            command_name="add_todo", start_ns=T0, parameters={"title": "x"},
        )
    ]


@pytest.fixture
def cost_world(tmp_path, monkeypatch):
    """One workflow, one benchmark, one task run twice, with real recorded spans."""
    monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "state"))
    folder = tmp_path / "cost_workflow"
    folder.mkdir()
    (folder / "_commands").mkdir()
    benchmark = setup.save_benchmark(
        folder, {"title": "Call costs", "tasks": [{"prompt": "Add a todo"}]}
    )
    experiment = setup.create_experiment(
        folder, benchmark["benchmark_id"], "v1", runs_per_task=4
    )
    experiment_id = experiment["experiment_id"]
    task_id = experiment["task_ids"][0]

    db = str(tmp_path / "evidence.sqlite3")
    store = obs.ObservabilityStore(db)
    controller = ExperimentController(
        db, store.store_identity(), external=False, workflow_folderpath=str(folder)
    )
    controller.create_experiment(
        experiment_id,
        experiment["description"],
        declared_tasks=1,
        declared_attempts=4,
        declarations=[(task_id, n, f"ch-{n}") for n in (1, 2, 3, 4)],
        workflow_name=setup.workflow_name_for(folder),
    )

    for attempt, turn_key, spans in (
        (1, LEFT_TURN, _left_spans()),
        (2, RIGHT_TURN, _right_spans()),
        (3, MANY_TURN, _many_spans()),
        (4, SILENT_TURN, _silent_spans()),
    ):
        channel = f"ch-{attempt}"
        conversation = store.mint_conversation_id(
            channel, experiment_id=experiment_id, task_id=task_id, attempt=attempt
        )
        controller.start_attempt(
            experiment_id, task_id, attempt, channel, conversation_id=conversation
        )
        row = _turn_row(
            turn_key,
            record=_record(
                turn_key,
                refs=[("c1", 0, f"ex-{turn_key}")],
                outputs=[_output("c1", "add_todo", {"title": "x"})],
            ),
            experiment_id=experiment_id,
            task_id=task_id,
            attempt=attempt,
        )
        row["conversation_id"] = conversation
        row["ordinal"] = 1
        _write(store, row, spans)
        controller.finish_attempt(
            experiment_id, task_id, attempt, outcome="pass", outcome_source="derived"
        )

    return {
        "folder": str(folder),
        "experiment_id": experiment_id,
        "task_id": task_id,
        "store": store,
        "store_id": store.store_identity(),
        "db": db,
    }


@pytest.fixture
def cost_server(cost_world):
    # Opened on the SAME evidence DB the world wrote, so the live turn and span
    # routes a link lands on read the recorded trace rather than an empty store.
    srv = run_chatbot_server.ChatbotServer(
        db_path=cost_world["db"],
        workflow_path=cost_world["folder"],
        port=0,
        spawn_options={"no_server": True},
    )
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    thread.join(timeout=5)


@pytest.fixture
def unreadable_side_payloads(cost_world):
    """Comparison payloads whose left side names a turn the store does not hold.

    Every part of these is real: the temp evidence store the rest of this
    module wrote, a real `StoreExecutionReader` over it, a real `ExecutionRef`,
    the real `project_execution` that reports the turn it could not find, and
    the same `_projection_payload` the live route publishes each side with.
    The only thing done by hand is naming a turn key the store never held --
    which is the point, because no live route builds such a reference:
    `best_run.project_attempt` takes a reference's turn keys FROM the rows it
    is about to read, so a pruned turn drops out of the reference instead of
    being reported as missing. A sealed manifest that outlived part of its
    archive, and a prune between the reference being built and the projection
    being read, both produce exactly this (`fix-9eg.3.1.2.1`).

    Two payloads, because the absence shows up differently either side of it:
    `absent` is a side with nothing readable at all, sitting next to a side
    that really made no LLM call; `partial` is a side whose readable turn is
    listed in full with the unreadable one counted beside it.
    """
    reader = StoreExecutionReader(cost_world["store_id"], cost_world["store"])

    def projected(turn_keys, attempt):
        return project_execution(
            ExecutionRef(
                store_id=cost_world["store_id"],
                turn_keys=turn_keys,
                experiment_id=cost_world["experiment_id"],
                task_id=cost_world["task_id"],
                attempt=attempt,
            ),
            reader,
        )

    def payload(left, left_attempt, right, right_attempt):
        view = selection_api.VIEW_ANSWERS
        return {
            "experiment_id": cost_world["experiment_id"],
            "task_id": cost_world["task_id"],
            "view": view,
            "left": selection_api._projection_payload(left, view),
            "right": selection_api._projection_payload(right, view),
            # Chrome only: the pane reads the attempt number off these.
            "left_run": {"attempt": left_attempt},
            "right_run": {"attempt": right_attempt},
        }

    gone = projected((PRUNED_TURN,), 1)
    silent = projected((SILENT_TURN,), 4)
    partial = projected((LEFT_TURN, PRUNED_TURN), 1)
    readable = projected((RIGHT_TURN,), 2)
    return {
        "projections": {"gone": gone, "silent": silent, "partial": partial},
        "absent": payload(gone, 1, silent, 4),
        "partial": payload(partial, 1, readable, 2),
    }


def _task_path(world, suffix=""):
    return (
        f"/api/experiments/{world['experiment_id']}"
        f"/tasks/{world['task_id']}{suffix}"
    )


def _comparison(world, **query):
    status, payload = selection_api.handle_get(
        world["folder"],
        _task_path(world, "/comparison"),
        {name: [str(value)] for name, value in query.items()},
    )
    assert status == 200, payload
    return payload


def _calls(payload, side):
    """The per-call anchors of one side, in publication order.

    Flattened over the side's turns exactly as the page's `comparisonSideCalls`
    does, so the two read the same field of the same payload.
    """
    listed = []
    for turn in payload[side]["turns"]:
        listed.extend(turn["usage"].get("calls_detail") or [])
    return listed


def _by_span(payload, side):
    return {call["span_id"]: call for call in _calls(payload, side)}


# ----------------------------------------------------------------------
# What the API publishes for the list to read
# ----------------------------------------------------------------------


class TestTheAnchorsTheListIsBuiltFrom:
    def test_each_side_publishes_its_own_calls_with_turn_and_span_anchors(
        self, cost_world
    ):
        payload = _comparison(cost_world, left_attempt=1, right_attempt=2)

        left = _by_span(payload, "left")
        right = _by_span(payload, "right")
        assert set(left) == {"a1-dear", "a1-cheap", "a1-zero", "a1-silent"}
        # The wrapper is not a second call; the inner record is the one kept.
        assert set(right) == {"a2-inner", "a2-first", "a2-second"}
        assert {call["turn_key"] for call in left.values()} == {LEFT_TURN}
        assert {call["turn_key"] for call in right.values()} == {RIGHT_TURN}
        # No side's anchors leak into the other's list.
        assert not set(left) & set(right)

    def test_zero_is_a_measurement_and_unrecorded_is_not_zero(self, cost_world):
        payload = _comparison(cost_world, left_attempt=1, right_attempt=2)
        left = _by_span(payload, "left")

        assert left["a1-zero"]["cost"] == 0.0
        assert left["a1-zero"]["total_tokens"] == 0
        assert left["a1-silent"]["cost"] is None
        assert left["a1-silent"]["total_tokens"] is None
        assert left["a1-silent"]["usage_state"] == USAGE_UNRECORDED

    def test_a_shared_response_is_a_call_that_carries_no_charge_of_its_own(
        self, cost_world
    ):
        payload = _comparison(cost_world, left_attempt=1, right_attempt=2)
        right = _by_span(payload, "right")

        assert right["a2-first"]["cost"] == 0.25
        assert right["a2-second"]["usage_state"] == USAGE_SHARED_RESPONSE
        assert right["a2-second"]["cost"] is None
        # Still a call, and still a recorded cache observation of its own.
        assert right["a2-second"]["cache_state"] == "hit"

    def test_the_listed_costs_add_up_to_the_totals_already_published(
        self, cost_world
    ):
        """The reason the list may be summed at all: it is the same canonical
        calls the roll-up was summed over, so a reader adding the rows gets the
        chip's number rather than a second, larger one."""
        payload = _comparison(cost_world, left_attempt=1, right_attempt=2)

        for side, expected in (("left", 0.752), ("right", 0.75)):
            listed = [
                call["cost"] for call in _calls(payload, side)
                if call["cost"] is not None
            ]
            assert sum(listed) == pytest.approx(expected)
            assert payload[side]["usage"]["cost"]["total"] == pytest.approx(expected)
            assert payload[side]["usage"]["cost"]["calls"] == len(_calls(payload, side))

        # Non-vacuous: the RAW spans of the right side carry 1.5000 of recorded
        # cost across four `fw.llm.call` records. A list that read the spans
        # instead of the published anchors would double the run's money.
        raw = sum(
            span.attributes["cost"]
            for span in _right_spans()
            if span.name == tracing.SPAN_LLM_CALL
        )
        assert raw == pytest.approx(1.5)

    def test_cache_state_and_token_coverage_survive_into_the_side_rollup(
        self, cost_world
    ):
        payload = _comparison(cost_world, left_attempt=1, right_attempt=2)

        assert payload["left"]["usage"]["cache"] == {
            "hit": 1, "miss": 2, "unknown": 1
        }
        assert payload["left"]["usage"]["tokens"]["coverage"] == "partial"
        assert payload["left"]["usage"]["tokens"]["zero"] == 1
        assert payload["right"]["usage"]["shared_responses"] == 1
        assert payload["right"]["usage"]["wrappers_folded"] == 1

    def test_a_re_emitted_record_is_one_call_and_so_one_row(self):
        """A store upserts a re-emitted span onto the row it re-records, so this
        shape reaches a reader through a live or merged read rather than through
        SQLite. It is folded before the list ever sees it: one anchor, one
        charge, and the fold counted rather than silent."""
        spans = _rows(
            [
                _llm_call("re", "t", start_ns=T0, usage=COMPLETE_USAGE, cost=0.4,
                          history_uuid="r1", ended=False),
                _llm_call("re", "t", start_ns=T0, usage=COMPLETE_USAGE, cost=0.4,
                          history_uuid="r1"),
            ]
        )

        rollup = usage_rollup(spans, turn_key="t")
        assert [call["span_id"] for call in rollup["calls_detail"]] == ["re"]
        assert rollup["records_folded"] == 1
        assert rollup["cost"]["total"] == pytest.approx(0.4)

    def test_a_pass_scoped_side_lists_that_pass_and_names_it(self, cost_world):
        """Pass identity is the side's, not the screen's: the teacher pass of
        attempt 1 lists the call recorded inside it, and the reference carries
        the pass so the list can say which one it is showing."""
        payload = _comparison(
            cost_world,
            left_attempt=1,
            right_attempt=2,
            left_pass=tracing.PASS_TEACHER,
        )

        assert payload["pass_scope"]["left_pass"] == tracing.PASS_TEACHER
        assert payload["left"]["ref"]["pass_id"] == tracing.PASS_TEACHER
        assert [call["span_id"] for call in _calls(payload, "left")] == ["a1-dear"]
        # The other side is untouched by the left side's pass scope.
        assert payload["right"]["ref"]["pass_id"] is None
        assert len(_calls(payload, "right")) == 3

    def test_a_side_with_many_priced_calls_still_publishes_the_unpriced_ones(
        self, cost_world
    ):
        """What the bounded list has to remain honest about: the unknown-cost and
        shared-response calls sort below twenty-one priced ones, so they are the
        two a truncated list would hide."""
        payload = _comparison(cost_world, left_attempt=3, right_attempt=4)
        calls = _by_span(payload, "left")

        assert len(calls) == MANY_PRICED + 2
        assert calls["a3-00"]["cost"] == pytest.approx(0.21)
        assert calls["a3-silent"]["cost"] is None
        assert calls["a3-shared"]["usage_state"] == USAGE_SHARED_RESPONSE
        assert calls["a3-shared"]["cost"] is None
        # The shared twin's money is the dearest call's, counted once.
        assert payload["left"]["usage"]["cost"]["total"] == pytest.approx(
            sum(round(0.01 * (MANY_PRICED - index), 4) for index in range(MANY_PRICED))
        )

    def test_a_side_that_made_no_llm_call_is_not_a_side_with_missing_anchors(
        self, cost_world
    ):
        """Attempt 4 dispatched a command and called no LLM. Its roll-up counts
        zero calls and publishes an empty anchor list -- which is what lets the
        page say "no LLM call is recorded" there and keep that sentence away from
        a side whose evidence merely could not be listed."""
        payload = _comparison(cost_world, left_attempt=3, right_attempt=4)

        assert payload["right"]["usage"]["calls"] == 0
        assert _calls(payload, "right") == []
        assert payload["right"]["turns"][0]["usage"]["calls_detail"] == []
        assert payload["right"]["unavailable"] == []

    def test_a_side_whose_turn_the_store_does_not_hold_is_not_a_side_at_zero(
        self, cost_world, unreadable_side_payloads
    ):
        """The other absence, and the one a live route cannot reach.

        `unavailable` is the only thing that tells these two apart: both
        roll-ups count zero calls and publish no anchors, and a page that read
        only the count would report an execution nobody could read as one that
        spent nothing.
        """
        gone = unreadable_side_payloads["projections"]["gone"]
        silent = unreadable_side_payloads["projections"]["silent"]

        assert len(gone.unavailable) == 1
        assert PRUNED_TURN in gone.unavailable[0]
        assert gone.ref.store_id in gone.unavailable[0]
        assert gone.readable is False
        assert gone.turns == ()
        # Non-vacuous, and read-only: the turn really is absent, and asking for
        # it did not create it.
        assert cost_world["store"].get_turn(PRUNED_TURN) is None

        assert silent.readable is True
        assert silent.unavailable == ()
        assert gone.usage["calls"] == silent.usage["calls"] == 0

    def test_half_an_execution_is_still_listed_with_the_rest_counted_missing(
        self, unreadable_side_payloads
    ):
        """A reference naming one readable turn and one the store lost projects
        the readable one in full -- refusing the whole side would hide the half
        that survived."""
        partial = unreadable_side_payloads["projections"]["partial"]
        side = unreadable_side_payloads["partial"]["left"]

        assert [turn.turn_key for turn in partial.turns] == [LEFT_TURN]
        assert len(partial.unavailable) == 1
        assert partial.usage["calls"] == 4
        listed = side["turns"][0]["usage"]["calls_detail"]
        assert {call["span_id"] for call in listed} == {
            "a1-dear", "a1-cheap", "a1-zero", "a1-silent"
        }

    def test_reading_the_comparison_changed_no_recorded_evidence(self, cost_world):
        """The list is a read. Asserted here because every claim above is about
        recorded evidence that must still say the same thing afterwards."""
        store = cost_world["store"]
        before = {
            turn: [json.dumps(span, sort_keys=True) for span in store.get_spans(turn)]
            for turn in (LEFT_TURN, RIGHT_TURN)
        }

        _comparison(cost_world, left_attempt=1, right_attempt=2)

        after = {
            turn: [json.dumps(span, sort_keys=True) for span in store.get_spans(turn)]
            for turn in (LEFT_TURN, RIGHT_TURN)
        }
        assert after == before


# ----------------------------------------------------------------------
# The shipped page, in a real DOM, over the real API
# ----------------------------------------------------------------------


def test_the_page_lists_each_sides_expensive_calls_in_a_real_dom(
    cost_server, cost_world, unreadable_side_payloads, tmp_path
):
    """The list a person reads, driven through the real page against the real
    server: ordering, the three absences kept apart, the retained cache and
    coverage labels, each side's own source and pass identity, and a link that
    opens the exact recorded call on the side that recorded it.

    The unreadable-side payloads travel on disk because they are the one shape
    no live route builds (see `unreadable_side_payloads`); they are still the
    real projection, rendered through the shipped helper in the same DOM.
    """
    jsdom_root = os.environ.get("TEST_JSDOM_ROOT")
    if not jsdom_root:
        pytest.skip("Set TEST_JSDOM_ROOT to run DOM integration with jsdom")
    unreadable = tmp_path / "unreadable_sides.json"
    unreadable.write_text(
        json.dumps(
            {key: unreadable_side_payloads[key] for key in ("absent", "partial")}
        ),
        encoding="utf-8",
    )
    script = Path(__file__).with_name("chatbot_call_costs_dom.cjs")
    result = subprocess.run(
        [
            "node", str(script), jsdom_root,
            f"http://127.0.0.1:{cost_server.port}/?token={cost_server.token}",
            cost_world["experiment_id"],
            cost_world["task_id"],
            cost_world["store_id"],
            str(unreadable),
        ],
        capture_output=True, text=True, timeout=240,
    )
    assert result.returncode == 0, result.stdout + result.stderr

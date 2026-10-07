"""Token completeness, cost coverage and recorded cache state (`fix-9eg.5`/`.6`).

Integration throughout, per `.cursor/rules/testing_rules.mdc`: real
`ObservabilityStore` databases on disk, real span rows written through
`upsert_span_rows`, the real execution ledger and the real `cost_rollup` from
`run_chatbot/server.py`, a real benchmark registration, a real
`ExperimentController`, a real `ChatbotServer` over a real socket and the
shipped page in a real DOM. No Mock fixtures: the claim under test is that what
a person reads on a chip and what a coding agent reads from the API are ONE
accounting of ONE recorded execution, and a stand-in for the store or the
server would let those two drift without the test noticing.

What the assertions defend, in one list:

- an unrecorded token count is `None`, and a recorded `0` is `0`;
- a call that reported part of its usage is `partial` and visibly so, not
  silently summed as if complete;
- a re-emitted span record (`end_span` reuses the span_id `start_span` opened)
  is ONE call, and its cost is charged once;
- a wrapper `fw.llm.call` quoting its own descendant's provider response is one
  call recorded at two levels, not two calls;
- two calls quoting one provider response are two calls whose tokens and money
  are counted once, and the duplication is reported rather than hidden;
- `cache_hit` true/false/absent are three answers, and the absent one is
  `unknown` rather than a miss.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path

import dspy
import pytest
from dspy.dsp.utils.utils import dotdict

from fastworkflow import state_paths, tracing
from fastworkflow.benchmark import setup
from fastworkflow.experiment.runner import ExperimentController
from fastworkflow.observability import store as obs
from fastworkflow.observability.comparison import (
    CACHE_HIT,
    CACHE_MISS,
    CACHE_UNKNOWN,
    COVERAGE_COMPLETE,
    COVERAGE_NONE,
    COVERAGE_PARTIAL,
    USAGE_COMPLETE,
    USAGE_PARTIAL,
    USAGE_SHARED_RESPONSE,
    USAGE_UNRECORDED,
    ExecutionRef,
    StoreExecutionReader,
    canonical_llm_spans,
    compare_executions,
    comparison_digest,
    merge_usage_rollups,
    project_execution,
    usage_rollup,
)
from fastworkflow.run_chatbot import selection_api
from fastworkflow.run_chatbot import server as run_chatbot_server
from fastworkflow.run_chatbot.server import cost_rollup, execution_ledger
from fastworkflow.utils.dspy_logger import observe_dspy_host
from tests.test_chatbot_benchmarks import _request
from tests.test_dspy_observability import _TraceHost
from tests.test_execution_comparison import (
    _execute_span,
    _output,
    _record,
    _turn_row,
    _write,
)

T0 = 1_700_000_000_000_000_000


# ----------------------------------------------------------------------
# Recorded LLM calls, in the shapes a provider really produces
# ----------------------------------------------------------------------


def _llm_call(
    span_id: str,
    turn_key: str,
    *,
    start_ns: int,
    usage: dict | None = None,
    cost: float | None = None,
    cache_hit: bool | None = None,
    history_uuid: str | None = None,
    parent_span_id: str | None = None,
    model: str | None = "mistral/mistral-small-latest",
    ended: bool = True,
) -> tracing.Span:
    """One `fw.llm.call` span, attributed the way `dspy_logger` attributes it.

    `usage` is JSON text because that is what `dspy_logger._json_text` writes;
    `cost`, `cache_hit` and `history_uuid` are scalars copied straight off the
    DSPy history entry. `ended=False` is a call that started and has not
    finished -- the state a live trace read of an in-flight turn sees, and the
    state in which no usage exists yet.
    """
    attributes: dict = {}
    if model is not None:
        attributes["model"] = model
    if usage is not None:
        attributes["usage"] = json.dumps(usage)
    if cost is not None:
        attributes["cost"] = cost
    if cache_hit is not None:
        attributes["cache_hit"] = cache_hit
    if history_uuid is not None:
        attributes["history_uuid"] = history_uuid
    return tracing.Span(
        span_id=span_id,
        trace_id=turn_key,
        parent_span_id=parent_span_id,
        name=tracing.SPAN_LLM_CALL,
        kind=tracing.KIND_LLM,
        channel_id="channel-1",
        start_ns=start_ns,
        end_ns=start_ns + 500_000 if ended else None,
        status=tracing.STATUS_OK if ended else tracing.STATUS_OPEN,
        attributes=attributes,
    )


def _rows(spans: list[tracing.Span]) -> list[dict]:
    """The spans as a reader sees them: plain rows with JSON text attributes.

    `usage_rollup` is called by the server on rows out of the store and may be
    handed rows a reader decoded itself, so the text form is the stricter of the
    two and is what these unit-scale assertions use.
    """
    return [
        {
            "span_id": span.span_id,
            "parent_span_id": span.parent_span_id,
            "name": span.name,
            "kind": span.kind,
            "start_ns": span.start_ns,
            "end_ns": span.end_ns,
            "status": span.status,
            "attributes": json.dumps(span.attributes),
        }
        for span in spans
    ]


COMPLETE_USAGE = {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150}


def _cache_state_of(attributes: dict) -> str:
    """The cache state the rollup reads out of one recorded call.

    Asks the real consumer rather than the private parser, so a producer change
    is judged by what a reader of the projection would conclude.
    """
    rollup = usage_rollup(
        [
            {
                "span_id": "only",
                "parent_span_id": None,
                "name": tracing.SPAN_LLM_CALL,
                "kind": tracing.KIND_LLM,
                "start_ns": 1,
                "end_ns": 2,
                "status": tracing.STATUS_OK,
                "attributes": json.dumps(attributes),
            }
        ]
    )
    states = [state for state, count in rollup["cache"].items() if count]
    assert len(states) == 1, f"one call cannot be in {len(states)} cache states"
    return states[0]


# ----------------------------------------------------------------------
# Zero, unknown and partial are three different answers
# ----------------------------------------------------------------------


class TestUsageCompleteness:
    def test_a_negative_count_is_malformed_rather_than_a_refund(self):
        """A token count tallies work done, so a negative one is not a smaller
        bill -- and summing it would put a turn's total below one of its own
        calls. It reads as unrecorded, which is what it is."""
        rollup = usage_rollup(
            _rows(
                [
                    _llm_call(
                        "a",
                        "t",
                        start_ns=T0,
                        usage={
                            "prompt_tokens": -120,
                            "completion_tokens": 30,
                            "total_tokens": -90,
                        },
                    )
                ]
            )
        )

        tokens = rollup["tokens"]
        assert tokens["prompt"] is None
        assert tokens["total"] is None
        assert tokens["completion"] == 30
        # One field surviving out of three is the definition of partial, and the
        # call is not silently dropped from the denominator.
        assert rollup["calls"] == 1
        assert tokens[USAGE_PARTIAL] == 1
        assert tokens["coverage"] == COVERAGE_PARTIAL

    def test_every_count_negative_is_indistinguishable_from_unrecorded(self):
        rollup = usage_rollup(
            _rows(
                [
                    _llm_call(
                        "a",
                        "t",
                        start_ns=T0,
                        usage={
                            "prompt_tokens": -1,
                            "completion_tokens": -1,
                            "total_tokens": -1,
                        },
                    )
                ]
            )
        )

        tokens = rollup["tokens"]
        assert (tokens["prompt"], tokens["completion"], tokens["total"]) == (
            None,
            None,
            None,
        )
        assert tokens[USAGE_UNRECORDED] == 1
        assert tokens["zero"] == 0, "nothing was recorded, so nothing recorded zero"
        assert tokens["coverage"] == COVERAGE_NONE

    def test_zero_is_still_a_recorded_zero(self):
        """The narrowing rejects below-zero only; a genuine zero is evidence."""
        rollup = usage_rollup(
            _rows(
                [
                    _llm_call(
                        "a",
                        "t",
                        start_ns=T0,
                        usage={
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                        },
                    )
                ]
            )
        )

        tokens = rollup["tokens"]
        assert (tokens["prompt"], tokens["completion"], tokens["total"]) == (0, 0, 0)
        assert tokens[USAGE_COMPLETE] == 1
        assert tokens["zero"] == 1
        assert tokens["coverage"] == COVERAGE_COMPLETE

    def test_nothing_recorded_is_none_and_never_zero(self):
        rollup = usage_rollup(
            _rows([_llm_call("a", "t", start_ns=T0), _llm_call("b", "t", start_ns=T0 + 1)])
        )

        assert rollup["calls"] == 2
        tokens = rollup["tokens"]
        assert tokens["prompt"] is None
        assert tokens["completion"] is None
        assert tokens["total"] is None
        assert tokens["unrecorded"] == 2
        assert tokens["coverage"] == COVERAGE_NONE
        assert tokens["zero"] == 0

    def test_a_recorded_zero_is_a_measurement(self):
        """The distinction the whole roll-up exists for: a provider that
        reported a zero-token call said something, and the reader must be able
        to tell it from a provider that reported nothing."""
        zero = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        rollup = usage_rollup(_rows([_llm_call("a", "t", start_ns=T0, usage=zero)]))

        assert rollup["tokens"]["total"] == 0
        assert rollup["tokens"]["zero"] == 1
        assert rollup["tokens"]["unrecorded"] == 0
        assert rollup["tokens"]["coverage"] == COVERAGE_COMPLETE
        assert rollup["calls_detail"][0]["usage_state"] == USAGE_COMPLETE

    def test_a_partly_reported_call_is_partial_not_complete(self):
        rollup = usage_rollup(
            _rows(
                [
                    _llm_call("a", "t", start_ns=T0, usage=COMPLETE_USAGE),
                    _llm_call("b", "t", start_ns=T0 + 1, usage={"prompt_tokens": 40}),
                ]
            )
        )

        assert rollup["tokens"]["prompt"] == 160
        # Only the call that reported completion tokens contributes to it.
        assert rollup["tokens"]["completion"] == 30
        assert rollup["tokens"]["total"] == 150
        assert rollup["tokens"]["complete"] == 1
        assert rollup["tokens"]["partial"] == 1
        assert rollup["tokens"]["coverage"] == COVERAGE_PARTIAL
        states = [call["usage_state"] for call in rollup["calls_detail"]]
        assert states == [USAGE_COMPLETE, USAGE_PARTIAL]

    def test_an_empty_usage_object_is_unrecorded_not_partial(self):
        rollup = usage_rollup(_rows([_llm_call("a", "t", start_ns=T0, usage={})]))

        assert rollup["tokens"]["unrecorded"] == 1
        assert rollup["tokens"]["partial"] == 0
        assert rollup["tokens"]["total"] is None

    def test_a_call_still_open_is_counted_as_a_call_with_nothing_known(self):
        rollup = usage_rollup(_rows([_llm_call("a", "t", start_ns=T0, ended=False)]))

        assert rollup["calls"] == 1
        assert rollup["open"] == 1
        assert rollup["completed"] == 0
        assert rollup["tokens"]["total"] is None
        assert rollup["cost"]["total"] is None

    def test_every_call_carries_its_own_span_anchor(self):
        """Human chip and agent read must be able to name the same call."""
        rollup = usage_rollup(
            _rows([_llm_call("a", "t", start_ns=T0, usage=COMPLETE_USAGE)]),
            turn_key="turn-7",
        )

        anchor = rollup["calls_detail"][0]
        assert anchor["span_id"] == "a"
        assert anchor["turn_key"] == "turn-7"
        assert anchor["model"] == "mistral/mistral-small-latest"
        assert anchor["prompt_tokens"] == 120
        assert anchor["completion_tokens"] == 30

    def test_no_llm_call_at_all_is_not_an_incomplete_measurement(self):
        rollup = usage_rollup([])

        assert rollup["calls"] == 0
        assert rollup["tokens"]["coverage"] == COVERAGE_NONE
        assert rollup["cost"] == {
            "calls": 0, "recorded": 0, "unrecorded": 0, "total": None
        }
        assert rollup["cache"] == {CACHE_HIT: 0, CACHE_MISS: 0, CACHE_UNKNOWN: 0}


# ----------------------------------------------------------------------
# One call, counted once
# ----------------------------------------------------------------------


class TestCanonicalCalls:
    def test_a_re_emitted_span_record_is_one_call(self):
        """`tracing.end_span` re-emits the span under the span_id `start_span`
        opened. A reader holding both records -- a live read of a turn that
        finished between two polls, or two reads merged -- has two rows for one
        call, and charging both doubles the turn's money and tokens."""
        opened = _llm_call("a", "t", start_ns=T0, ended=False)
        closed = _llm_call(
            "a", "t", start_ns=T0, usage=COMPLETE_USAGE, cost=0.004, cache_hit=False,
            history_uuid="resp-1",
        )
        rollup = usage_rollup(_rows([opened, closed]))

        assert rollup["calls"] == 1
        assert rollup["records_folded"] == 1
        assert rollup["completed"] == 1
        assert rollup["tokens"]["total"] == 150
        assert rollup["cost"] == {
            "calls": 1, "recorded": 1, "unrecorded": 0, "total": 0.004
        }
        assert rollup["cache"][CACHE_MISS] == 1

    def test_the_ended_record_wins_whichever_order_it_arrives_in(self):
        opened = _llm_call("a", "t", start_ns=T0, ended=False)
        closed = _llm_call(
            "a", "t", start_ns=T0, usage=COMPLETE_USAGE, cost=0.004, history_uuid="r"
        )

        forwards = usage_rollup(_rows([opened, closed]))
        backwards = usage_rollup(_rows([closed, opened]))

        assert forwards == backwards
        assert backwards["tokens"]["total"] == 150
        assert backwards["completed"] == 1

    def test_a_nested_wrapper_quoting_one_response_is_one_call(self):
        """An LM that invokes another LM leaves an `fw.llm.call` inside an
        `fw.llm.call`, and `dspy_logger` copies the same DSPy history entry onto
        both, `history_uuid` included. That is one provider call recorded at two
        levels; the inner record is the one closest to the provider."""
        outer = _llm_call(
            "wrapper", "t", start_ns=T0, usage=COMPLETE_USAGE, cost=0.004,
            history_uuid="resp-1",
        )
        inner = _llm_call(
            "inner", "t", start_ns=T0 + 10, usage=COMPLETE_USAGE, cost=0.004,
            cache_hit=True, history_uuid="resp-1", parent_span_id="wrapper",
        )
        rollup = usage_rollup(_rows([outer, inner]))

        assert rollup["calls"] == 1
        assert rollup["wrappers_folded"] == 1
        assert rollup["tokens"]["total"] == 150
        assert rollup["cost"]["total"] == 0.004
        # The record that survived is the inner one, so the cache flag it
        # carried survives with it.
        assert rollup["calls_detail"][0]["span_id"] == "inner"
        assert rollup["cache"][CACHE_HIT] == 1

    def test_nesting_through_an_intermediate_span_is_still_nesting(self):
        """The wrapper's descendant need not be its direct child: a module span
        commonly sits between two LM calls."""
        outer = _llm_call(
            "wrapper", "t", start_ns=T0, usage=COMPLETE_USAGE, history_uuid="resp-1"
        )
        middle = _execute_span(
            "middle", "t", call_id="c1", command_name="whatever",
            start_ns=T0 + 5, parent_span_id="wrapper",
        )
        inner = _llm_call(
            "inner", "t", start_ns=T0 + 10, usage=COMPLETE_USAGE,
            history_uuid="resp-1", parent_span_id="middle",
        )
        rollup = usage_rollup(_rows([outer, middle, inner]))

        assert rollup["calls"] == 1
        assert rollup["wrappers_folded"] == 1
        assert rollup["tokens"]["total"] == 150

    def test_two_nested_calls_with_different_responses_are_two_calls(self):
        """Non-vacuous guard on the fold: nesting alone must not collapse two
        calls. A wrapper LM that really made its own provider call has its own
        history entry, and both calls were paid for."""
        outer = _llm_call(
            "wrapper", "t", start_ns=T0, usage=COMPLETE_USAGE, cost=0.004,
            history_uuid="resp-1",
        )
        inner = _llm_call(
            "inner", "t", start_ns=T0 + 10, usage=COMPLETE_USAGE, cost=0.004,
            history_uuid="resp-2", parent_span_id="wrapper",
        )
        rollup = usage_rollup(_rows([outer, inner]))

        assert rollup["calls"] == 2
        assert rollup["wrappers_folded"] == 0
        assert rollup["tokens"]["total"] == 300
        assert rollup["cost"]["total"] == pytest.approx(0.008)

    def test_two_sibling_calls_quoting_one_response_report_the_duplication(self):
        """A real duplicate identity that is NOT nesting. `dspy_logger` reads
        `lm.history[-1]` when a call ends, so two calls that finish around each
        other can both copy one entry -- one of them is then quoting usage it
        did not spend. Two calls were made; those tokens were spent once. The
        roll-up says both, and does not pick which call was misattributed."""
        first = _llm_call(
            "s1", "t", start_ns=T0, usage=COMPLETE_USAGE, cost=0.004,
            cache_hit=False, history_uuid="resp-1",
        )
        second = _llm_call(
            "s2", "t", start_ns=T0 + 1_000, usage=COMPLETE_USAGE, cost=0.004,
            cache_hit=False, history_uuid="resp-1",
        )
        rollup = usage_rollup(_rows([first, second]))

        assert rollup["calls"] == 2
        assert rollup["wrappers_folded"] == 0
        assert rollup["shared_responses"] == 1
        assert rollup["tokens"]["total"] == 150
        assert rollup["cost"] == {
            "calls": 2, "recorded": 1, "unrecorded": 1, "total": 0.004
        }
        # Coverage is partial, because one of the two calls has no usage of its
        # own that anybody recorded.
        assert rollup["tokens"]["coverage"] == COVERAGE_PARTIAL
        states = [call["usage_state"] for call in rollup["calls_detail"]]
        assert states == [USAGE_COMPLETE, USAGE_SHARED_RESPONSE]
        # Both calls keep their anchors and their recorded cache state: the
        # second one happened, it just did not spend those tokens twice.
        assert [call["span_id"] for call in rollup["calls_detail"]] == ["s1", "s2"]
        assert rollup["cache"][CACHE_MISS] == 2

    def test_which_of_two_sharing_calls_is_credited_is_decided_the_same_way_twice(self):
        first = _llm_call(
            "zzz", "t", start_ns=T0, usage=COMPLETE_USAGE, history_uuid="resp-1"
        )
        second = _llm_call(
            "aaa", "t", start_ns=T0 + 1_000, usage=COMPLETE_USAGE, history_uuid="resp-1"
        )

        forwards = usage_rollup(_rows([first, second]))
        backwards = usage_rollup(_rows([second, first]))

        assert forwards == backwards
        # Recording order, not span id order: the earlier call is credited.
        assert forwards["calls_detail"][0]["span_id"] == "zzz"

    def test_calls_without_a_recorded_response_id_are_never_folded_together(self):
        """Two calls that recorded no history entry are two calls. Folding on
        absence would merge every uninstrumented call in a turn into one."""
        rollup = usage_rollup(
            _rows(
                [
                    _llm_call("a", "t", start_ns=T0, cost=0.001),
                    _llm_call("b", "t", start_ns=T0 + 1, cost=0.002),
                ]
            )
        )

        assert rollup["calls"] == 2
        assert rollup["shared_responses"] == 0
        assert rollup["cost"]["total"] == pytest.approx(0.003)

    def test_canonical_spans_keep_every_other_span_untouched(self):
        """`canonical_llm_spans` is what feeds the injected `cost_rollup`, so it
        must fold LLM duplicates and change nothing else about the trace."""
        execute = _execute_span(
            "ex", "t", call_id="c1", command_name="add_todo", start_ns=T0
        )
        opened = _llm_call("a", "t", start_ns=T0 + 1, ended=False)
        closed = _llm_call("a", "t", start_ns=T0 + 1, cost=0.004, history_uuid="r")
        rows = _rows([execute, opened, closed])

        canonical = canonical_llm_spans(rows)

        assert [span["span_id"] for span in canonical] == ["ex", "a"]
        assert canonical[0] is rows[0]
        assert cost_rollup(canonical) == {
            "calls": 1, "recorded": 1, "unrecorded": 0, "total": 0.004
        }
        # The defect this guards -- the same rows charged per span -- is now
        # closed at the function itself as well, so `cost_rollup` answers the
        # same whether or not its caller folded first. This assertion used to
        # pin the OLD behaviour (calls == 2) to document the gap; it now pins
        # the parity, because the gap was what shipped inflated per-turn and
        # per-attempt figures (cost-parity-reproduced).
        assert cost_rollup(rows) == cost_rollup(canonical)


# ----------------------------------------------------------------------
# Cache state: three answers, and a hit is not a fault
# ----------------------------------------------------------------------


class TestRecordedCacheState:
    def test_hit_miss_and_unknown_are_counted_apart(self):
        rollup = usage_rollup(
            _rows(
                [
                    _llm_call("a", "t", start_ns=T0, cache_hit=True),
                    _llm_call("b", "t", start_ns=T0 + 1, cache_hit=False),
                    _llm_call("c", "t", start_ns=T0 + 2),
                ]
            )
        )

        assert rollup["cache"] == {CACHE_HIT: 1, CACHE_MISS: 1, CACHE_UNKNOWN: 1}
        states = [call["cache_state"] for call in rollup["calls_detail"]]
        assert states == [CACHE_HIT, CACHE_MISS, CACHE_UNKNOWN]

    def test_a_non_boolean_flag_is_unknown_rather_than_coerced(self):
        """A truthy string would otherwise read as a hit."""
        span = _llm_call("a", "t", start_ns=T0)
        span.attributes["cache_hit"] = "true"

        rollup = usage_rollup(_rows([span]))

        assert rollup["cache"] == {CACHE_HIT: 0, CACHE_MISS: 0, CACHE_UNKNOWN: 1}

    def test_a_cache_hit_does_not_touch_the_usage_or_cost_it_recorded(self):
        """A hit is an observation about how the answer arrived. It is not a
        reason to discard what the call recorded, and nothing here treats it as
        a failure or a stale replay."""
        rollup = usage_rollup(
            _rows(
                [
                    _llm_call(
                        "a", "t", start_ns=T0, usage=COMPLETE_USAGE, cost=0.004,
                        cache_hit=True, history_uuid="r",
                    )
                ]
            )
        )

        assert rollup["cache"][CACHE_HIT] == 1
        assert rollup["tokens"]["total"] == 150
        assert rollup["cost"]["total"] == 0.004
        assert rollup["calls_detail"][0]["usage_state"] == USAGE_COMPLETE


# ----------------------------------------------------------------------
# Merging turns into an execution
# ----------------------------------------------------------------------


class TestMergingTurns:
    def test_a_merge_keeps_none_for_nothing_recorded(self):
        silent = usage_rollup(_rows([_llm_call("a", "t1", start_ns=T0)]))
        other = usage_rollup(_rows([_llm_call("b", "t2", start_ns=T0)]))

        merged = merge_usage_rollups([silent, other])

        assert merged["calls"] == 2
        assert merged["tokens"]["total"] is None
        assert merged["tokens"]["coverage"] == COVERAGE_NONE
        assert merged["cost"]["total"] is None
        assert merged["cost"]["unrecorded"] == 2

    def test_a_merge_sums_what_was_recorded(self):
        one = usage_rollup(
            _rows([_llm_call("a", "t1", start_ns=T0, usage=COMPLETE_USAGE, cost=0.004)])
        )
        two = usage_rollup(
            _rows([_llm_call("b", "t2", start_ns=T0, usage=COMPLETE_USAGE, cost=0.006)])
        )

        merged = merge_usage_rollups([one, two])

        assert merged["tokens"] == {
            "prompt": 240, "completion": 60, "total": 300,
            "complete": 2, "partial": 0, "unrecorded": 0, "shared": 0, "zero": 0,
            "coverage": COVERAGE_COMPLETE,
        }
        assert merged["cost"]["total"] == pytest.approx(0.010)

    def test_the_merge_does_not_repeat_the_per_call_anchors(self):
        """The anchors belong to the turn that recorded them and are published
        there. Repeating them at execution scope would put one span_id on the
        wire twice and invite a reader to tally it twice."""
        part = usage_rollup(
            _rows([_llm_call("a", "t1", start_ns=T0, usage=COMPLETE_USAGE)]),
            turn_key="t1",
        )

        merged = merge_usage_rollups([part])

        assert part["calls_detail"]
        assert "calls_detail" not in merged
        assert merged["calls"] == 1

    def test_merging_nothing_is_an_execution_with_no_calls(self):
        merged = merge_usage_rollups([])

        assert merged["calls"] == 0
        assert merged["tokens"]["coverage"] == COVERAGE_NONE
        assert merged["cache"] == {CACHE_HIT: 0, CACHE_MISS: 0, CACHE_UNKNOWN: 0}


# ----------------------------------------------------------------------
# Through the real projection, over a real store
# ----------------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> obs.ObservabilityStore:
    return obs.ObservabilityStore(str(tmp_path / "evidence.sqlite3"))


@pytest.fixture
def reader(store: obs.ObservabilityStore) -> StoreExecutionReader:
    return StoreExecutionReader("store-1", store)


def _seed(store: obs.ObservabilityStore, turn_key: str, llm_spans: list) -> None:
    """One dispatch and whatever LLM calls the caller wants beneath it."""
    execute = _execute_span(
        "ex-" + turn_key, turn_key, call_id="c1", command_name="add_todo",
        start_ns=T0, parameters={"title": "x"},
    )
    record = _record(
        turn_key,
        refs=[("c1", 0, "ex-" + turn_key)],
        outputs=[_output("c1", "add_todo", {"title": "x"})],
    )
    _write(store, _turn_row(turn_key, record=record), [execute] + llm_spans)


def _project(ref: ExecutionRef, reader: StoreExecutionReader):
    return project_execution(
        ref, reader, ledger=execution_ledger, cost_rollup=cost_rollup
    )


class TestTheProjectionAgreesWithItself:
    def test_the_money_and_the_tokens_are_counted_over_one_set_of_calls(
        self, store, reader
    ):
        """`cost` is still the server's own roll-up, injected unchanged. What
        changed is that it and the token figures beside it see the same calls,
        so the two cannot report different call counts for one turn."""
        _seed(
            store,
            "turn-a",
            [
                _llm_call(
                    "l1", "turn-a", start_ns=T0 + 1, usage=COMPLETE_USAGE,
                    cost=0.004, cache_hit=True, history_uuid="r1",
                ),
                _llm_call("l2", "turn-a", start_ns=T0 + 2, history_uuid="r2"),
            ],
        )

        projection = _project(
            ExecutionRef(store_id="store-1", turn_keys=("turn-a",)), reader
        )

        assert projection.cost == {
            "calls": 2, "recorded": 1, "unrecorded": 1, "total": 0.004
        }
        assert projection.usage["cost"] == projection.cost
        assert projection.usage["calls"] == projection.cost["calls"]
        assert projection.usage["tokens"]["total"] == 150
        assert projection.usage["cache"] == {
            CACHE_HIT: 1, CACHE_MISS: 0, CACHE_UNKNOWN: 1
        }

    def test_a_re_emitted_record_is_not_charged_twice_by_either_roll_up(
        self, store, reader
    ):
        """Written through the real store, so the upsert is the real one. The
        store collapses the two records itself; the projection is asserted to
        agree rather than to depend on which layer did it."""
        _seed(
            store,
            "turn-b",
            [
                _llm_call("l1", "turn-b", start_ns=T0 + 1, ended=False),
                _llm_call(
                    "l1", "turn-b", start_ns=T0 + 1, usage=COMPLETE_USAGE,
                    cost=0.004, history_uuid="r1",
                ),
            ],
        )

        projection = _project(
            ExecutionRef(store_id="store-1", turn_keys=("turn-b",)), reader
        )

        assert projection.cost["calls"] == 1
        assert projection.cost["total"] == 0.004
        assert projection.usage["calls"] == 1
        assert projection.usage["tokens"]["total"] == 150

    def test_a_nested_wrapper_does_not_inflate_the_projected_call_count(
        self, store, reader
    ):
        _seed(
            store,
            "turn-c",
            [
                _llm_call(
                    "outer", "turn-c", start_ns=T0 + 1, usage=COMPLETE_USAGE,
                    cost=0.004, history_uuid="shared",
                ),
                _llm_call(
                    "inner", "turn-c", start_ns=T0 + 2, usage=COMPLETE_USAGE,
                    cost=0.004, cache_hit=True, history_uuid="shared",
                    parent_span_id="outer",
                ),
            ],
        )

        projection = _project(
            ExecutionRef(store_id="store-1", turn_keys=("turn-c",)), reader
        )

        assert projection.usage["calls"] == 1
        assert projection.usage["wrappers_folded"] == 1
        assert projection.cost == {
            "calls": 1, "recorded": 1, "unrecorded": 0, "total": 0.004
        }
        assert projection.usage["cost"] == projection.cost

    def test_each_turn_carries_its_own_anchors_and_the_execution_carries_totals(
        self, store, reader
    ):
        _seed(
            store,
            "turn-d",
            [_llm_call("d1", "turn-d", start_ns=T0 + 1, usage=COMPLETE_USAGE)],
        )
        _seed(
            store,
            "turn-e",
            [_llm_call("e1", "turn-e", start_ns=T0 + 1, usage={"prompt_tokens": 9})],
        )

        projection = _project(
            ExecutionRef(store_id="store-1", turn_keys=("turn-d", "turn-e")), reader
        )

        by_turn = {turn.turn_key: turn.usage for turn in projection.turns}
        assert [call["span_id"] for call in by_turn["turn-d"]["calls_detail"]] == ["d1"]
        assert by_turn["turn-d"]["calls_detail"][0]["turn_key"] == "turn-d"
        assert [call["span_id"] for call in by_turn["turn-e"]["calls_detail"]] == ["e1"]
        assert projection.usage["calls"] == 2
        assert projection.usage["tokens"]["prompt"] == 129
        assert projection.usage["tokens"]["partial"] == 1
        assert projection.usage["tokens"]["coverage"] == COVERAGE_PARTIAL
        assert "calls_detail" not in projection.usage

    def test_an_unreadable_turn_does_not_make_the_usage_unknown(
        self, store, reader
    ):
        """Partial evidence is inspectable: the turn that IS readable still
        reports what it spent, and the missing one is named in `unavailable`."""
        _seed(
            store,
            "turn-f",
            [_llm_call("f1", "turn-f", start_ns=T0 + 1, usage=COMPLETE_USAGE)],
        )

        projection = _project(
            ExecutionRef(store_id="store-1", turn_keys=("turn-f", "turn-gone")), reader
        )

        assert projection.unavailable
        assert projection.usage["calls"] == 1
        assert projection.usage["tokens"]["total"] == 150

    def test_the_wire_payload_carries_usage_beside_cost(self, store, reader):
        _seed(
            store,
            "turn-g",
            [
                _llm_call(
                    "g1", "turn-g", start_ns=T0 + 1, usage=COMPLETE_USAGE,
                    cache_hit=True,
                )
            ],
        )

        payload = _project(
            ExecutionRef(store_id="store-1", turn_keys=("turn-g",)), reader
        ).as_dict()

        assert payload["usage"]["cache"][CACHE_HIT] == 1
        assert payload["turns"][0]["usage"]["calls_detail"][0]["span_id"] == "g1"
        assert payload["cost"]["calls"] == 1

    def test_a_comparison_digest_reports_coverage_for_both_sides(
        self, store, reader
    ):
        """The digest is what a probe over a corpus whose content must not be
        printed may log, so the usage it carries has to be counts only."""
        _seed(
            store,
            "turn-h",
            [_llm_call("h1", "turn-h", start_ns=T0 + 1, usage=COMPLETE_USAGE)],
        )
        _seed(store, "turn-i", [_llm_call("i1", "turn-i", start_ns=T0 + 1)])

        digest = comparison_digest(
            compare_executions(
                ExecutionRef(store_id="store-1", turn_keys=("turn-h",)),
                ExecutionRef(store_id="store-1", turn_keys=("turn-i",)),
                reader,
                ledger=execution_ledger,
                cost_rollup=cost_rollup,
            )
        )

        assert digest["left"]["usage"]["tokens"]["coverage"] == COVERAGE_COMPLETE
        assert digest["right"]["usage"]["tokens"]["coverage"] == COVERAGE_NONE
        assert "calls_detail" not in digest["left"]["usage"]
        assert "calls_detail" not in digest["right"]["usage"]


# ----------------------------------------------------------------------
# Over a real socket, through the real selection API
# ----------------------------------------------------------------------


@pytest.fixture
def usage_world(tmp_path, monkeypatch):
    """One workflow, one benchmark, one task run twice with real LLM spans.

    Attempt 1 recorded a complete usage report, a cache hit and a call that
    reported nothing; attempt 2 recorded a nested wrapper quoting its inner
    call's response. Those are the two shapes the API has to answer honestly.
    """
    monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "state"))
    folder = tmp_path / "usage_workflow"
    folder.mkdir()
    (folder / "_commands").mkdir()
    db = state_paths.observability_db(str(folder))
    store = obs.ObservabilityStore(db)
    benchmark = setup.save_benchmark(
        folder, {"title": "Usage", "tasks": [{"prompt": "Add a todo"}]}
    )
    experiment = setup.create_experiment(
        folder, benchmark["benchmark_id"], "v1", runs_per_task=2
    )
    experiment_id = experiment["experiment_id"]
    task_id = experiment["task_ids"][0]

    controller = ExperimentController(
        str(folder), store.store_identity(), external=False
    )
    controller.create_experiment(
        experiment_id,
        experiment["description"],
        declared_tasks=1,
        declared_attempts=2,
        declarations=[(task_id, n, f"ch-{n}") for n in (1, 2)],
        workflow_name=setup.workflow_name_for(folder),
    )

    llm_by_attempt = {
        1: [
            dict(span_id="a1-l1", usage=COMPLETE_USAGE, cost=0.004,
                 cache_hit=True, history_uuid="a1-r1"),
            dict(span_id="a1-l2", history_uuid="a1-r2"),
        ],
        2: [
            dict(span_id="a2-outer", usage=COMPLETE_USAGE, cost=0.004,
                 history_uuid="a2-shared"),
            dict(span_id="a2-inner", usage=COMPLETE_USAGE, cost=0.004,
                 cache_hit=False, history_uuid="a2-shared",
                 parent_span_id="a2-outer"),
        ],
    }
    for attempt in (1, 2):
        channel = f"ch-{attempt}"
        conversation = store.mint_conversation_id(
            channel, experiment_id=experiment_id, task_id=task_id, attempt=attempt
        )
        controller.start_attempt(
            experiment_id, task_id, attempt, channel, conversation_id=conversation
        )
        turn_key = f"usage-a{attempt}-t1"
        execute = _execute_span(
            f"ex-{turn_key}", turn_key, call_id="c1", command_name="add_todo",
            start_ns=T0, parameters={"title": "x"},
        )
        llm = [
            _llm_call(turn_key=turn_key, start_ns=T0 + 1_000, **spec)
            for spec in llm_by_attempt[attempt]
        ]
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
        _write(store, row, [execute] + llm)
        controller.finish_attempt(
            experiment_id, task_id, attempt, outcome="pass", outcome_source="derived"
        )

    return {
        "folder": str(folder),
        "experiment_id": experiment_id,
        "task_id": task_id,
        "store": store,
        "db": db,
    }


@pytest.fixture
def usage_server(usage_world):
    # Opened on the SAME evidence DB the world wrote, so the live turn and
    # experiment routes read the recorded spans rather than an empty cold-start
    # store. Without it `/api/turn/<key>` answers 404 and the endpoint-parity
    # assertions would pass by never reaching the code they are about.
    srv = run_chatbot_server.ChatbotServer(
        db_path=usage_world["db"],
        workflow_path=usage_world["folder"],
        port=0,
        spawn_options={"no_server": True},
    )
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    thread.join(timeout=5)


def _task_path(world, suffix=""):
    return (
        f"/api/experiments/{world['experiment_id']}"
        f"/tasks/{world['task_id']}{suffix}"
    )


class TestTheApiPublishesUsage:
    def test_one_run_projection_carries_tokens_cache_and_anchors(self, usage_world):
        status, payload = selection_api.handle_get(
            usage_world["folder"], _task_path(usage_world, "/runs/1"), {}
        )

        assert status == 200
        usage = payload["projection"]["usage"]
        assert usage["calls"] == 2
        assert usage["tokens"]["total"] == 150
        assert usage["tokens"]["unrecorded"] == 1
        assert usage["tokens"]["coverage"] == COVERAGE_PARTIAL
        assert usage["cache"] == {CACHE_HIT: 1, CACHE_MISS: 0, CACHE_UNKNOWN: 1}
        anchors = payload["projection"]["turns"][0]["usage"]["calls_detail"]
        assert {call["span_id"] for call in anchors} == {"a1-l1", "a1-l2"}
        assert {call["usage_state"] for call in anchors} == {
            USAGE_COMPLETE, USAGE_UNRECORDED
        }

    def test_the_comparison_answers_both_sides_over_a_real_socket(self, usage_server,
                                                                  usage_world):
        status, payload = _request(
            usage_server,
            _task_path(usage_world, "/comparison") + "?left_attempt=1&right_attempt=2",
        )

        assert status == 200
        left, right = payload["left"]["usage"], payload["right"]["usage"]
        assert left["calls"] == 2 and left["cache"][CACHE_HIT] == 1
        # The wrapper side made ONE call, not the two spans its trace holds.
        assert right["calls"] == 1
        assert right["wrappers_folded"] == 1
        assert right["tokens"]["total"] == 150
        assert right["cost"] == payload["right"]["cost"]
        assert right["cache"][CACHE_MISS] == 1

    def test_the_answers_view_still_carries_usage(self, usage_world):
        """The trimmed view drops steps, not the accounting: an agent asking
        for answers still gets what the run spent."""
        status, payload = selection_api.handle_get(
            usage_world["folder"],
            _task_path(usage_world, "/comparison"),
            {"view": ["answers"], "left_attempt": ["1"], "right_attempt": ["2"]},
        )

        assert status == 200
        assert "steps" not in payload["left"]
        assert payload["left"]["usage"]["calls"] == 2

    def test_nothing_about_the_evidence_changed(self, usage_world):
        """These are reads. The recorded spans are asserted untouched, so a
        roll-up can never be mistaken for a rewrite of what it counted."""
        before = usage_world["store"].get_spans("usage-a2-t1")
        selection_api.handle_get(
            usage_world["folder"],
            _task_path(usage_world, "/comparison"),
            {"left_attempt": ["1"], "right_attempt": ["2"]},
        )

        after = usage_world["store"].get_spans("usage-a2-t1")
        assert [span["span_id"] for span in after] == [
            span["span_id"] for span in before
        ]
        assert after == before


class TestEveryEndpointQuotesTheSameCost:
    """One provider call costs one amount, on every screen that names it.

    Attempt 2's trace holds an outer `fw.llm.call` and the inner call that
    produced its `history_uuid`; both recorded $0.004 for the ONE response. The
    turn list, the opened turn, the attempt rows and the comparison each read
    cost through a different code path, and they used to disagree: everything
    going through `server.cost_rollup` on raw spans reported 2 calls and $0.008,
    while the comparison projection reported 1 call and $0.004. A run that looks
    twice as expensive from the navigation rail as from its own comparison is
    not a display quirk -- whichever number an operator quotes is wrong half the
    time.

    So parity is asserted across ENDPOINTS rather than inside one projection,
    because self-agreement within `project_execution` was exactly what passed
    while the shipped turn and attempt figures stayed inflated.
    """

    TURN_KEY = "usage-a2-t1"
    ONE_CALL = {"calls": 1, "recorded": 1, "unrecorded": 0, "total": 0.004}

    def test_the_wrapper_is_charged_once_by_every_route(self, usage_server,
                                                       usage_world):
        store_id = usage_world["store"].store_identity()

        status, payload = _request(usage_server, f"/api/turn/{self.TURN_KEY}")
        assert status == 200
        assert payload["turn"]["llm_cost"] == self.ONE_CALL, "opened turn"

        status, page = _request(
            usage_server, f"/api/turns?logical_turn_key={self.TURN_KEY}"
        )
        assert status == 200
        rows = [r for r in page["turns"] if r["turn_key"] == self.TURN_KEY]
        assert len(rows) == 1
        assert rows[0]["llm_cost"] == self.ONE_CALL, "turn list row"

        status, attempts = _request(
            usage_server,
            f"/api/experiment/{usage_world['experiment_id']}/attempts"
            f"?task={usage_world['task_id']}",
        )
        assert status == 200
        wrapper_attempt = [a for a in attempts["attempts"] if a["attempt"] == 2]
        assert len(wrapper_attempt) == 1
        assert wrapper_attempt[0]["llm_cost"] == self.ONE_CALL, "attempt row"

        status, cmp = _request(
            usage_server,
            _task_path(usage_world, "/comparison")
            + "?left_attempt=1&right_attempt=2",
        )
        assert status == 200
        assert cmp["right"]["cost"] == self.ONE_CALL, "comparison projection"
        assert cmp["right"]["usage"]["cost"] == self.ONE_CALL, "usage rollup"
        assert store_id  # the world really is one identifiable store

    def test_the_turn_that_needed_no_folding_is_unaffected(self, usage_server):
        """Canonicalizing must not quietly change the honest case: attempt 1
        made two unrelated calls, one of which recorded no cost at all, and both
        facts survive -- including the unrecorded one, which stays counted."""
        status, payload = _request(usage_server, "/api/turn/usage-a1-t1")

        assert status == 200
        assert payload["turn"]["llm_cost"] == {
            "calls": 2,
            "recorded": 1,
            "unrecorded": 1,
            "total": 0.004,
        }

    def test_a_turn_with_no_recorded_cost_still_says_unknown_not_free(
        self, usage_server, usage_world
    ):
        """`total: None` is the schema's unknown. Folding must not turn an
        unpriced call into a zero-cost one."""
        store = usage_world["store"]
        _seed(store, "usage-unpriced", [_llm_call("u1", "usage-unpriced",
                                                  start_ns=T0 + 5)])

        status, payload = _request(usage_server, "/api/turn/usage-unpriced")

        assert status == 200
        assert payload["turn"]["llm_cost"] == {
            "calls": 1,
            "recorded": 0,
            "unrecorded": 1,
            "total": None,
        }

    def test_repeated_records_of_one_span_are_one_call_on_every_route(
        self, usage_server, usage_world
    ):
        """`end_span` re-emits a span under the same `span_id`, so one call can
        be recorded twice -- once open, once closed.

        Two defences, and the test says which one is doing the work. The STORE
        upserts by `span_id`, so a re-emitted record replaces its earlier self
        and the route never sees the pair: that is asserted by reading the rows
        back, not assumed. The fold inside `cost_rollup` therefore cannot change
        this route's answer, and the test does not pretend otherwise -- it
        asserts the fold directly on raw rows instead, which is the shape a
        caller reading an in-flight span buffer rather than the store can still
        hand it.
        """
        store = usage_world["store"]
        turn_key = "usage-reemitted"
        records = [
            _llm_call("r1", turn_key, start_ns=T0 + 6, history_uuid="rr"),
            _llm_call("r1", turn_key, start_ns=T0 + 6, cost=0.004,
                      usage=COMPLETE_USAGE, history_uuid="rr"),
        ]
        _seed(store, turn_key, records)

        stored_llm = [
            s for s in store.get_spans(turn_key) if s["name"] == tracing.SPAN_LLM_CALL
        ]
        assert len(stored_llm) == 1, "the store collapsed the re-emitted record"

        status, payload = _request(usage_server, f"/api/turn/{turn_key}")
        assert status == 200
        assert payload["turn"]["llm_cost"] == self.ONE_CALL
        assert usage_rollup(store.get_spans(turn_key))["cost"] == (
            payload["turn"]["llm_cost"]
        )

        # Both records present, as a caller outside the store can see them.
        raw = _rows(records)
        assert cost_rollup(raw) == self.ONE_CALL
        assert usage_rollup(raw)["records_folded"] == 1
        assert usage_rollup(raw)["cost"] == cost_rollup(raw)

    def test_siblings_quoting_one_response_are_charged_once_by_every_route(
        self, usage_server, usage_world
    ):
        """The gap the nested fold did not close.

        Two `fw.llm.call` spans with NO ancestry between them, each recording
        $0.25 for the same `history_uuid`. `canonical_llm_spans` deliberately
        leaves them alone -- they are two calls, and which one really spent the
        tokens is not decidable -- but only ONE of them may be charged. The
        server used to sum both ($0.50) while the projection charged one
        ($0.25), so the turn row and its own comparison disagreed
        (shared-response-cost).
        """
        store = usage_world["store"]
        turn_key = "usage-siblings"
        _seed(
            store,
            turn_key,
            [
                _llm_call("sib-first", turn_key, start_ns=T0 + 10, cost=0.25,
                          usage=COMPLETE_USAGE, history_uuid="same-response"),
                _llm_call("sib-second", turn_key, start_ns=T0 + 20, cost=0.25,
                          usage=COMPLETE_USAGE, history_uuid="same-response"),
            ],
        )

        status, payload = _request(usage_server, f"/api/turn/{turn_key}")
        assert status == 200
        # Both calls stay counted; one charge between them; the uncharged twin
        # is visible as unrecorded rather than deleted from the tally.
        assert payload["turn"]["llm_cost"] == {
            "calls": 2, "recorded": 1, "unrecorded": 1, "total": 0.25,
        }

        status, page = _request(
            usage_server, f"/api/turns?logical_turn_key={turn_key}"
        )
        assert status == 200
        rows = [r for r in page["turns"] if r["turn_key"] == turn_key]
        assert len(rows) == 1
        assert rows[0]["llm_cost"] == payload["turn"]["llm_cost"]

        rollup = usage_rollup(store.get_spans(turn_key))
        assert rollup["cost"] == payload["turn"]["llm_cost"]
        assert rollup["shared_responses"] == 1
        assert rollup["wrappers_folded"] == 0, "nothing was nested here"
        # Tokens follow the same rule, and the anchors keep both calls.
        assert rollup["tokens"]["total"] == 150
        assert rollup["tokens"]["shared"] == 1
        assert {call["span_id"] for call in rollup["calls_detail"]} == {
            "sib-first", "sib-second"
        }

    def test_the_server_delegates_rather_than_reimplementing_the_fold(self):
        """One accounting, asserted as identity rather than as equal numbers on
        one example: a second implementation is what produced the sibling gap
        after the nested one was closed."""
        shapes = {
            "nested": [
                _llm_call("outer", "t", start_ns=T0, cost=0.25,
                          usage=COMPLETE_USAGE, history_uuid="r"),
                _llm_call("inner", "t", start_ns=T0 + 1, cost=0.25,
                          usage=COMPLETE_USAGE, history_uuid="r",
                          parent_span_id="outer"),
            ],
            "siblings": [
                _llm_call("first", "t", start_ns=T0, cost=0.25,
                          usage=COMPLETE_USAGE, history_uuid="r"),
                _llm_call("second", "t", start_ns=T0 + 1, cost=0.25,
                          usage=COMPLETE_USAGE, history_uuid="r"),
            ],
            "unrelated": [
                _llm_call("a", "t", start_ns=T0, cost=0.25,
                          usage=COMPLETE_USAGE, history_uuid="r1"),
                _llm_call("b", "t", start_ns=T0 + 1, cost=0.25,
                          usage=COMPLETE_USAGE, history_uuid="r2"),
            ],
            "unpriced": [_llm_call("a", "t", start_ns=T0)],
            "empty": [],
        }
        for name, spans in shapes.items():
            rows = _rows(spans)
            assert cost_rollup(rows) == usage_rollup(rows)["cost"], name

        # And the shapes really are different, so the loop is not vacuous.
        assert cost_rollup(_rows(shapes["nested"]))["calls"] == 1
        assert cost_rollup(_rows(shapes["siblings"]))["calls"] == 2
        assert cost_rollup(_rows(shapes["siblings"]))["total"] == 0.25
        assert cost_rollup(_rows(shapes["unrelated"]))["total"] == 0.5

    def test_the_server_and_the_projection_agree_on_roots_counterexample(self):
        """Root's exact reproduction, kept as a regression: outer and inner
        `fw.llm.call` sharing one `history_uuid`, each recording $0.25."""
        rows = _rows(
            [
                _llm_call("outer", "t", start_ns=T0, cost=0.25,
                          history_uuid="response1"),
                _llm_call("inner", "t", start_ns=T0 + 1, cost=0.25,
                          history_uuid="response1", parent_span_id="outer"),
            ]
        )

        assert cost_rollup(rows) == {
            "calls": 1, "recorded": 1, "unrecorded": 0, "total": 0.25,
        }
        assert usage_rollup(rows)["cost"] == cost_rollup(rows)
        # Idempotent, because `project_execution` hands this an already-folded
        # list and folding twice must not drop the call.
        assert cost_rollup(canonical_llm_spans(rows)) == cost_rollup(rows)


# ----------------------------------------------------------------------
# The shipped page, in a real DOM
# ----------------------------------------------------------------------


def test_the_page_shows_usage_and_cache_in_a_real_dom(usage_server, usage_world):
    """The chips a person reads, driven through the real page against the real
    server: the compact usage renderer on the comparison pane, and the
    per-call cache explanation in the trace pane."""
    jsdom_root = os.environ.get("TEST_JSDOM_ROOT")
    if not jsdom_root:
        pytest.skip("Set TEST_JSDOM_ROOT to run DOM integration with jsdom")
    script = Path(__file__).with_name("chatbot_usage_cache_dom.cjs")
    result = subprocess.run(
        [
            "node", str(script), jsdom_root,
            f"http://127.0.0.1:{usage_server.port}/?token={usage_server.token}",
            usage_world["experiment_id"],
            usage_world["task_id"],
        ],
        capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr


class TestThePageSourceKeepsTheDistinctions:
    """Cheap guards on the page text, for the claims the DOM run cannot reach
    without a provider: that no absence is rendered as a zero or a miss."""

    def test_the_helpers_are_present_and_unknown_is_its_own_answer(self, usage_server):
        source = run_chatbot_server.load_index_html()

        assert b"function cacheStateOf(span)" in source
        assert b"function appendCacheChip(parent, cache)" in source
        assert b"function appendUsageChips(parent, usage)" in source
        assert b"function renderComparisonUsage(container, cmp)" in source
        assert b"appendUsageChips(box, side[1].usage)" in source
        assert b'"cache state not recorded"' in source
        assert b'"tokens not recorded"' in source
        # Unknown is rendered as unknown for a single call too, rather than
        # being folded into "miss".
        assert b'"not recorded \xe2\x80\x94 this call recorded no cache flag, so "' in source
        # A cache hit says how the answer arrived and stops there.
        assert b'"cache hit \xe2\x80\x94 this answer was served from the LLM cache "' in source
        assert b'"judgement about it.");' in source

    def test_a_wrapper_is_folded_on_the_page_as_well_as_in_the_api(self,
                                                                  usage_server):
        """The page builds its trace tree from spans it loads itself, so the
        fold has to exist on both sides or one screen contradicts the other."""
        source = run_chatbot_server.load_index_html()

        assert b"function spanResponseId(span)" in source
        assert b"function sumResponses(nodes)" in source
        # The fold is decided from the whole span set, not from tree position:
        # this assertion used to pin `var wrapper = ... charged.indexOf(...)`,
        # which only ever caught a duplicate that happened to NEST. Siblings
        # quoting one response charged twice on screen while the server charged
        # once (shared-response-cost), so the surface it pinned was replaced.
        assert b"function chargeFolds(byId)" in source
        assert b'folds[parent.span_id] = "wrapper"' in source
        assert b'folds[span.span_id] = "shared"' in source
        assert b'var fold = chargeFolds(byId)[span.span_id] || "";' in source
        # Both shapes are charged once, and a shared call is still a call.
        assert b"function sharedTokens()" in source
        assert b"function sharedCost()" in source
        assert b"counted under another call quoting the " in source


class _EchoLM(dspy.BaseLM):
    """A real DSPy LM that answers locally and reports a real cache flag.

    Not a stand-in for an LM: it IS one, on the same `forward_contract="legacy"`
    path `dspy.LM` itself uses, so DSPy normalizes its provider response and
    builds the history entry exactly as it does for a paid provider. The reason
    to own the flag is that a provider cache hit cannot be arranged on demand
    without spending money, and `DummyLM` never sets the field at all.
    """

    def __init__(self, *, cache_hit: bool):
        super().__init__(model="local/echo")
        self._cache_hit = cache_hit

    def forward(self, prompt=None, messages=None, **kwargs):
        return dotdict(
            choices=[
                dotdict(
                    message=dotdict(content="[[ ## answer ## ]]\n4", tool_calls=None),
                    finish_reason="stop",
                )
            ],
            model="local/echo",
            usage=dotdict(**COMPLETE_USAGE),
            cache_hit=self._cache_hit,
            id="resp-echo",
        )


def _capture_llm_call(
    tmp_path: Path, lm, *, disable_history: bool = False
) -> list[dict]:
    """One observed `dspy.Predict` call's spans, read back out of real SQLite."""
    db_path = str(tmp_path / "observability.sqlite3")
    sink = obs.SQLiteTraceSink(db_path)
    host = _TraceHost(sink)
    try:
        with observe_dspy_host(host):
            with dspy.context(lm=lm, disable_history=disable_history):
                dspy.Predict("question -> answer")(question="What is 2+2?")
        assert sink.flush()
    finally:
        sink.close()
    spans = obs.ObservabilityStore(db_path).get_spans(host.current_turn_key)
    llm_spans = [s for s in spans if s["name"] == tracing.SPAN_LLM_CALL]
    assert len(llm_spans) == 1, f"expected one LLM span, got {len(llm_spans)}"
    return llm_spans


class TestTheProducerOnlyClaimsACacheStateItRead:
    """The rollup above can only distinguish hit/miss/unknown if the producer
    stops inventing misses.

    `dspy_logger.on_lm_end` used to write
    ``cache_hit=bool(getattr(response, "cache_hit", False))``, so a history entry
    with no readable response -- a plain dict, a provider object predating the
    field, nothing at all -- came out as a recorded MISS. Every consumer then
    read that as "the provider was called", which is a claim nobody made.

    These run the real callback through a real `dspy.Predict` against DSPy's own
    `DummyLM` and read the span back out of a real SQLite sink, so what is
    asserted is what a reader would find on record.
    """

    @pytest.mark.parametrize(
        "cache_hit, expected", [(True, CACHE_HIT), (False, CACHE_MISS)]
    )
    def test_a_normalized_response_records_the_flag_it_carried(
        self, tmp_path, cache_hit, expected
    ):
        """Both states are evidence, so both get written."""
        spans = _capture_llm_call(tmp_path, _EchoLM(cache_hit=cache_hit))
        attributes = json.loads(spans[0]["attributes"])

        assert attributes["cache_hit"] is cache_hit
        assert _cache_state_of(attributes) == expected

    def test_a_response_carrying_no_flag_records_no_cache_state(self, tmp_path):
        """DSPy's own `DummyLM` goes down the legacy direct path, whose history
        entry keeps the raw provider object -- here a `dotdict` with no
        `cache_hit` at all. This is the case the old producer turned into a
        recorded miss, and it is a real supported LM, not a contrived one."""
        from dspy.utils import DummyLM

        spans = _capture_llm_call(tmp_path, DummyLM([{"answer": "4"}] * 4))
        attributes = json.loads(spans[0]["attributes"])

        assert "cache_hit" not in attributes, (
            "the response carried no flag; writing false would assert a "
            "provider call on no evidence"
        )
        assert _cache_state_of(attributes) == CACHE_UNKNOWN

    def test_a_disabled_history_records_no_cache_state_at_all(self, tmp_path):
        """`disable_history=True` is what run_fastapi_mcp installs process-wide,
        so this is the shape most production spans actually have."""
        spans = _capture_llm_call(
            tmp_path, _EchoLM(cache_hit=True), disable_history=True
        )
        attributes = json.loads(spans[0]["attributes"])

        assert "cache_hit" not in attributes
        assert _cache_state_of(attributes) == CACHE_UNKNOWN

    @pytest.mark.parametrize(
        "response, expected",
        [
            (None, None),
            ({}, None),
            ({"cache_hit": True}, True),
            ({"cache_hit": False}, False),
            ({"cache_hit": "true"}, None),
            ({"cache_hit": 1}, None),
        ],
    )
    def test_only_an_explicit_boolean_counts(self, response, expected):
        """A dict history entry and a truthy non-bool are both absences: the
        first has no field, the second has a value no reader should coerce."""
        from fastworkflow.utils.dspy_logger import _recorded_cache_hit

        assert _recorded_cache_hit(response) is expected

    def test_a_real_normalized_response_reports_both_states(self):
        """`LMResponse.cache_hit` is a plain bool on DSPy's normalized response
        type, so reading it as an attribute is not a guess."""
        from dspy.core.types import LMResponse

        from fastworkflow.utils.dspy_logger import _recorded_cache_hit

        assert _recorded_cache_hit(LMResponse.from_text("x", cache_hit=True)) is True
        assert _recorded_cache_hit(LMResponse.from_text("x", cache_hit=False)) is False

    def test_a_response_from_before_the_field_existed_reads_as_unknown(self):
        """The field is not guaranteed by anything but LMResponse; a raw
        provider object routed straight through has never had it."""
        from fastworkflow.utils.dspy_logger import _recorded_cache_hit

        class ProviderResponseWithoutTheField:
            def __init__(self):
                self.id = "chatcmpl-legacy"

        assert _recorded_cache_hit(ProviderResponseWithoutTheField()) is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

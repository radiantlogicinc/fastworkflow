"""Recorded command outcomes grouped within one run (`fix-9eg.3.1.3`).

Integration throughout, per `.cursor/rules/testing_rules.mdc`: a real
`ObservabilityStore` on disk, real span rows, real turn records, the real
execution ledger from `run_chatbot/server.py`, the real
`comparison.project_execution` and the real HTTP surface in
`run_chatbot/selection_api.py`. No Mock fixtures and nothing paid.

The claims worth defending here are attribution claims, and each has a test
that fails when the attribution slips:

- a turn that contains a failure does not make every command in it a failure;
- a dispatch nothing recorded an outcome for is UNKNOWN, not a success;
- the record and the span disagreeing is a reported fact, not a tie one of them
  silently wins;
- an unreadable turn is missing coverage, not zero failures;
- one ledger naming a dispatch twice is one dispatch;
- a command name nobody recorded is its own bucket, not a guess.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path

import pytest

from fastworkflow import state_paths, tracing
from fastworkflow.benchmark import setup
from fastworkflow.experiment.runner import ExperimentController
from fastworkflow.observability import command_summary as cs
from fastworkflow.observability import store as obs
from fastworkflow.observability.comparison import (
    ExecutionRef,
    StoreExecutionReader,
    discover_pass_selectors,
    project_execution,
)
from fastworkflow.run_chatbot import selection_api
from fastworkflow.run_chatbot import server as run_chatbot_server
from fastworkflow.run_chatbot.server import cost_rollup, execution_ledger
from tests.test_chatbot_benchmarks import _request
from tests.test_execution_comparison import _execute_span, _output, _write
from tests.test_execution_comparison import _turn_row as _evidence_turn_row

T0 = 1_700_000_000_000_000_000


# ----------------------------------------------------------------------
# Seeding real dispatches
# ----------------------------------------------------------------------


def _call(
    name,
    *,
    response_success=True,
    recorded_output=True,
    span=True,
    span_success=True,
    status=tracing.STATUS_OK,
    duration_ns=1_000_000,
    pass_stamp=None,
    child_calls=None,
    named=True,
):
    """One dispatch to record, described the way the evidence will hold it.

    `recorded_output=False` is a dispatch the turn record filed a ref for and
    no `CommandOutput` for -- real, and the shape whose outcome is unknown.
    `named=False` drops the name from both recording sources, which is the only
    way a dispatch genuinely has none.
    """
    return {
        "name": name,
        "response_success": response_success,
        "recorded_output": recorded_output,
        "span": span,
        "span_success": span_success,
        "status": status,
        "duration_ns": duration_ns,
        "pass_stamp": pass_stamp,
        "child_calls": child_calls,
        "named": named,
    }


def _seed_turn(
    store,
    turn_key,
    calls,
    *,
    experiment_id=None,
    task_id=None,
    attempt=None,
    conversation=None,
    ordinal=None,
    status="completed",
    success=True,
    answer="done",
    duplicate_refs=(),
):
    """Write one real turn: a record of refs and outputs, plus execute spans.

    `duplicate_refs` re-lists the named dispatch indexes in
    `execution_records`, which is what a resumed turn's record looks like when
    it files a call the trace already holds.
    """
    refs, spans, outputs = [], [], []
    for index, call in enumerate(calls):
        call_id = f"{turn_key}-call-{index}"
        span_id = f"{turn_key}-span-{index}"
        refs.append(
            {
                "command_call_id": call_id,
                "parent_call_id": None,
                "command_ordinal": index,
                "span_id": span_id if call["span"] else None,
            }
        )
        if call["recorded_output"]:
            outputs.append(
                _output(
                    call_id,
                    call["name"] if call["named"] else "",
                    {"n": index},
                    success=call["response_success"],
                )
            )
        if call["span"]:
            extra = {}
            if call["pass_stamp"]:
                extra["fw.pass"] = call["pass_stamp"]
            spans.append(
                _execute_span(
                    span_id,
                    turn_key,
                    call_id=call_id,
                    command_name=call["name"] if call["named"] else "",
                    start_ns=T0 + index * 10_000_000,
                    duration_ns=call["duration_ns"],
                    status=call["status"],
                    success=call["span_success"],
                    child_calls=call["child_calls"],
                    extra=extra or None,
                )
            )
    for index in duplicate_refs:
        refs.append(dict(refs[index]))
    record = {
        "turn_output": {
            "turn_key": turn_key,
            "success": success,
            "command_outputs": outputs,
        },
        "execution_records": refs,
        "routing_events": [],
    }
    row = _evidence_turn_row(
        turn_key,
        record=record,
        answer=answer,
        status=status,
        success=success,
        experiment_id=experiment_id,
        task_id=task_id,
        attempt=attempt,
    )
    row["conversation_id"] = conversation
    row["ordinal"] = ordinal
    _write(store, row, spans)
    return turn_key


@pytest.fixture
def store(tmp_path):
    return obs.ObservabilityStore(str(tmp_path / "evidence.sqlite3"))


@pytest.fixture
def reader(store):
    return StoreExecutionReader("store-1", store)


def _summarize(store, reader, turn_keys, **ref_kwargs):
    ref = ExecutionRef(store_id="store-1", turn_keys=tuple(turn_keys), **ref_kwargs)
    projection = project_execution(
        ref, reader, ledger=execution_ledger, cost_rollup=cost_rollup
    )
    return cs.summarize_projection(projection), projection


def _group_named(summary, name):
    for group in summary["groups"]:
        if group["command_name"] == name:
            return group
    raise AssertionError(
        f"no group for {name!r}; groups are "
        f"{[group['command_name'] for group in summary['groups']]}"
    )


# ----------------------------------------------------------------------
# Grouping, counting and exact members
# ----------------------------------------------------------------------


class TestGroupsAndCounts:
    def test_dispatches_group_by_the_exact_recorded_name(self, store, reader):
        _seed_turn(
            store,
            "t1",
            [_call("add_item"), _call("add_item"), _call("list_items")],
        )

        summary, _ = _summarize(store, reader, ["t1"])

        assert summary["metric"] == cs.METRIC
        assert summary["metric_version"] == cs.METRIC_VERSION
        assert summary["unit"] == cs.UNIT
        assert [(g["command_name"], g["dispatches"]) for g in summary["groups"]] == [
            ("add_item", 2),
            ("list_items", 1),
        ]
        assert summary["totals"]["dispatches"] == 3

    def test_names_are_never_normalised_into_one_another(self, store, reader):
        """`Add_Item` and `add_item` are two recorded names.

        Folding them would be an inference about a workflow's naming, and the
        one thing a reader of this table must be able to trust is that a row
        names dispatches that really recorded that name.
        """
        _seed_turn(store, "t1", [_call("add_item"), _call("Add_Item")])

        summary, _ = _summarize(store, reader, ["t1"])

        assert {g["command_name"] for g in summary["groups"]} == {
            "add_item",
            "Add_Item",
        }

    def test_a_group_carries_the_exact_dispatches_it_counted(self, store, reader):
        _seed_turn(store, "t1", [_call("add_item"), _call("list_items")])

        summary, projection = _summarize(store, reader, ["t1"])

        group = _group_named(summary, "add_item")
        assert group["dispatches"] == len(group["contributors"]) == 1
        contributor = group["contributors"][0]
        step = [s for s in projection.steps if s.command_name == "add_item"][0]
        # The contributor IS that step, addressed the way the rest of the
        # observability surface addresses one.
        assert contributor["command_call_id"] == step.command_call_id
        assert contributor["turn_key"] == step.turn_key
        assert contributor["span_id"] == step.span_id
        assert contributor["position"] == step.position
        assert contributor["store_id"] == "store-1"

    def test_a_failure_belongs_to_its_dispatch_and_not_to_the_turn(
        self, store, reader
    ):
        """The boundary the epic names: a turn containing command X and a
        failure does not prove X failed."""
        _seed_turn(
            store,
            "t1",
            [
                _call("add_item", response_success=False, span_success=False,
                      status=tracing.STATUS_ERROR),
                _call("list_items"),
                _call("complete_item"),
            ],
            success=False,
            status="failed",
        )

        summary, _ = _summarize(store, reader, ["t1"])

        assert _group_named(summary, "add_item")["response_success"] == {
            "true": 0, "false": 1, "unknown": 0
        }
        for name in ("list_items", "complete_item"):
            assert _group_named(summary, name)["response_success"] == {
                "true": 1, "false": 0, "unknown": 0
            }, name
        assert summary["totals"]["response_success"] == {
            "true": 2, "false": 1, "unknown": 0
        }

    def test_one_failing_dispatch_does_not_condemn_its_own_other_dispatches(
        self, store, reader
    ):
        """Two dispatches of ONE command, one of which failed.

        The group reports one of each rather than a group-level verdict, and
        the two contributors say which was which.
        """
        _seed_turn(
            store,
            "t1",
            [
                _call("add_item"),
                _call("add_item", response_success=False, span_success=False,
                      status=tracing.STATUS_ERROR),
            ],
        )

        group = _group_named(_summarize(store, reader, ["t1"])[0], "add_item")

        assert group["response_success"] == {"true": 1, "false": 1, "unknown": 0}
        assert sorted(c["response_success"] for c in group["contributors"]) == [
            False, True
        ]


# ----------------------------------------------------------------------
# Unknown, conflicting and unnamed
# ----------------------------------------------------------------------


class TestUnknownAndConflict:
    def test_a_dispatch_with_no_recorded_outcome_is_unknown_not_a_success(
        self, store, reader
    ):
        _seed_turn(
            store,
            "t1",
            [_call("add_item"), _call("list_items", recorded_output=False)],
        )

        summary, _ = _summarize(store, reader, ["t1"])

        group = _group_named(summary, "list_items")
        assert group["response_success"] == {"true": 0, "false": 0, "unknown": 1}
        assert group["contributors"][0]["response_success"] is None
        # The span still said something, and that is reported on its own axis
        # rather than promoted into the canonical count.
        assert group["span_outcome"]["ok"] == 1
        assert summary["totals"]["response_success"]["unknown"] == 1

    def test_the_record_and_the_span_disagreeing_is_kept_as_a_disagreement(
        self, store, reader
    ):
        """The record says the dispatch answered successfully; the span it was
        recorded on says it errored. Neither is deleted."""
        _seed_turn(
            store,
            "t1",
            [
                _call("add_item", response_success=True, span_success=False,
                      status=tracing.STATUS_ERROR)
            ],
        )

        summary, _ = _summarize(store, reader, ["t1"])

        group = _group_named(summary, "add_item")
        assert group["response_success"] == {"true": 1, "false": 0, "unknown": 0}
        assert group["span_outcome"] == {"ok": 0, "failed": 1, "unrecorded": 0}
        assert group["conflict_count"] == 1
        conflict = group["conflicts"][0]
        assert conflict["response_success"] is True
        assert conflict["span_success"] is False
        assert conflict["status"] == tracing.STATUS_ERROR
        assert summary["totals"]["conflicts"] == 1

    def test_agreement_is_not_reported_as_a_conflict(self, store, reader):
        _seed_turn(
            store,
            "t1",
            [
                _call("add_item"),
                _call("remove_item", response_success=False, span_success=False,
                      status=tracing.STATUS_ERROR),
            ],
        )

        summary, _ = _summarize(store, reader, ["t1"])

        assert summary["totals"]["conflicts"] == 0

    def test_an_unrecorded_span_outcome_is_not_a_conflict_with_the_record(self):
        """A dispatch with no span outcome at all cannot disagree with one.

        Calling that a conflict would fill the table with the absence of
        evidence, which is exactly the noise that makes a real disagreement
        unreadable.
        """
        observation = cs.CommandObservation(
            store_id="s", turn_key="t", command_call_id="c",
            command_name="add_item", response_success=True,
            span_success=None, status=None,
        )

        assert cs.span_outcome(observation) == cs.SPAN_UNRECORDED
        assert cs.is_conflicting(observation) is False

    def test_a_cancelled_dispatch_is_not_counted_as_a_failed_one(self):
        """`cancelled` is the status an ask-user suspension writes.

        A suspension is a control signal, so reading it as a failure would
        report every question the workflow asked as an error.
        """
        observation = cs.CommandObservation(
            store_id="s", turn_key="t", command_call_id="c",
            command_name="ask", status=tracing.STATUS_CANCELLED,
        )

        assert cs.span_outcome(observation) == cs.SPAN_UNRECORDED

    def test_a_dispatch_nobody_named_goes_to_its_own_bucket(self, store, reader):
        _seed_turn(
            store,
            "t1",
            [_call("add_item"), _call("nameless", named=False,
                                       recorded_output=False)],
        )

        summary, _ = _summarize(store, reader, ["t1"])

        unknown = [g for g in summary["groups"] if g["unknown_command"]]
        assert len(unknown) == 1
        assert unknown[0]["command_name"] is None
        assert unknown[0]["dispatches"] == 1
        assert summary["totals"]["unknown_command_dispatches"] == 1
        # And it sorts last, so it never reads as a command called "".
        assert summary["groups"][-1]["unknown_command"] is True


# ----------------------------------------------------------------------
# Identity: one dispatch counted once
# ----------------------------------------------------------------------


class TestIdentity:
    def test_a_ledger_reference_filed_twice_is_one_dispatch(self, store, reader):
        _seed_turn(store, "t1", [_call("add_item")], duplicate_refs=(0,))

        summary, projection = _summarize(store, reader, ["t1"])

        assert len(projection.steps) == 1, "the ledger itself joins on call id"
        assert _group_named(summary, "add_item")["dispatches"] == 1

    def test_the_reducer_refuses_to_count_one_dispatch_twice(self):
        """The ledger dedupes today; the reducer does not rely on it.

        The reducer is the layer a bounded multi-run list will reach later,
        where the same dispatch can genuinely arrive from two reads of one
        store, and a doubled count there would be invented activity.
        """
        one = cs.CommandObservation(
            store_id="s", turn_key="t", command_call_id="c",
            command_name="add_item", response_success=True,
        )

        summary = cs.summarize_commands([one, one])

        assert summary["totals"]["dispatches"] == 1
        assert summary["coverage"]["duplicate_observations"] == 1

    def test_the_same_call_id_in_two_turns_is_two_dispatches(self):
        """Identity is the turn AND the call, because a call id is only unique
        within the turn that minted it."""
        summary = cs.summarize_commands(
            [
                cs.CommandObservation(store_id="s", turn_key="t1",
                                      command_call_id="c", command_name="add_item"),
                cs.CommandObservation(store_id="s", turn_key="t2",
                                      command_call_id="c", command_name="add_item"),
            ]
        )

        assert _group_named(summary, "add_item")["dispatches"] == 2


# ----------------------------------------------------------------------
# Coverage: what was not read stays visible
# ----------------------------------------------------------------------


class TestCoverage:
    def test_an_unreadable_turn_is_missing_coverage_and_not_zero_failures(
        self, store, reader
    ):
        _seed_turn(store, "t1", [_call("add_item")])

        summary, projection = _summarize(store, reader, ["t1", "t-pruned"])

        assert projection.unavailable, "the reference names a turn the store lacks"
        assert summary["coverage"]["enumeration_complete"] is False
        assert summary["coverage"]["turns_unreadable"] == 1
        assert summary["coverage"]["unreadable_turns"] == list(projection.unavailable)
        assert summary["coverage"]["turns_examined"] == 1
        assert summary["coverage"]["capture"] == cs.COVERAGE_PARTIAL
        assert any("could not be read" in gap
                   for gap in summary["coverage"]["capture_gaps"])
        # The counts describe what was examined and say so; they do not claim
        # the unread turn contained nothing.
        assert summary["totals"]["dispatches"] == 1

    def test_a_fully_read_and_fully_recorded_run_says_so_on_both_axes(
        self, store, reader
    ):
        _seed_turn(store, "t1", [_call("add_item")])

        summary, _ = _summarize(store, reader, ["t1"])

        assert summary["coverage"] == {
            "enumeration_complete": True,
            "turns_examined": 1,
            "turns_with_dispatches": 1,
            "turns_without_dispatches": 0,
            "turns_unreadable": 0,
            "unreadable_turns": [],
            "steps_outside_pass": 0,
            "duplicate_observations": 0,
            "capture": cs.COVERAGE_COMPLETE,
            "capture_gaps": [],
            "dispatches_without_span": 0,
            "dispatches_not_in_record": 0,
            "outcomes_unrecorded": 0,
        }

    def test_reading_every_turn_is_not_the_same_claim_as_recording_every_hop(
        self, store, reader
    ):
        """Enumeration and capture are two questions.

        This run's single turn was read whole, so nothing is missing at the
        turn level -- and its inner hop still recorded no span and no record
        entry. Reporting that as complete is what would let a reader take the
        counts for the whole of what happened.
        """
        _seed_turn(
            store,
            "t1",
            [_call("add_item",
                   child_calls=[{"call_id": "inner-1",
                                 "command_name": "inner_hop"}])],
        )

        coverage = _summarize(store, reader, ["t1"])[0]["coverage"]

        assert coverage["enumeration_complete"] is True
        assert coverage["capture"] == cs.COVERAGE_PARTIAL
        assert coverage["dispatches_without_span"] == 1
        assert coverage["dispatches_not_in_record"] == 1
        assert coverage["outcomes_unrecorded"] == 1
        assert coverage["capture_gaps"]

    def test_a_readable_turn_that_dispatched_nothing_is_still_examined(
        self, store, reader
    ):
        """Counting only dispatch-bearing turns would shrink the examined
        population to the part that had activity, and a turn whose dispatches
        were never captured would vanish from the denominator entirely."""
        _seed_turn(store, "t1", [_call("add_item")])
        _seed_turn(store, "t2", [])

        summary, projection = _summarize(store, reader, ["t1", "t2"])

        assert len(projection.turns) == 2
        coverage = summary["coverage"]
        assert coverage["turns_examined"] == 2
        assert coverage["turns_with_dispatches"] == 1
        assert coverage["turns_without_dispatches"] == 1
        assert summary["scope"]["observed"]["turns"] == 2
        # And it is a capture gap, because a turn that dispatched nothing and a
        # turn whose dispatches were not captured look the same from here.
        assert coverage["capture"] == cs.COVERAGE_PARTIAL
        assert any("recorded no dispatch at all" in gap
                   for gap in coverage["capture_gaps"])

    def test_a_dispatch_missing_from_the_record_or_the_trace_is_counted_as_such(
        self, store, reader
    ):
        """Both tiers are partial on purpose, and the summary says which tier
        held each dispatch rather than presenting one tier as the whole.

        The span-less dispatch also shows why the unknown bucket is not an
        edge case: a command name is recorded on the SPAN, so a dispatch the
        record filed and the trace never held has an outcome and no name.
        """
        _seed_turn(
            store,
            "t1",
            [
                _call(
                    "add_item",
                    child_calls=[{"call_id": "inner-1", "command_name": "inner_hop"}],
                ),
                _call("list_items", span=False),
            ],
        )

        summary, _ = _summarize(store, reader, ["t1"])

        inner = _group_named(summary, "inner_hop")
        assert inner["coverage"]["without_span"] == 1
        assert inner["coverage"]["not_in_record"] == 1
        assert inner["coverage"]["child_calls"] == 1
        assert inner["response_success"] == {"true": 0, "false": 0, "unknown": 1}
        unnamed = _group_named(summary, None)
        assert unnamed["coverage"] == {
            "in_record": 1, "not_in_record": 0, "span_recorded": 0,
            "without_span": 1, "child_calls": 0,
        }
        # Its outcome IS known: the record recorded one. Unknown name and
        # unknown outcome are separate absences.
        assert unnamed["response_success"] == {"true": 1, "false": 0, "unknown": 0}

    def test_an_empty_run_reports_nothing_rather_than_a_zero_verdict(
        self, store, reader
    ):
        """A run with no observed dispatch describes no population.

        `none`, not `complete`: there is nothing here whose capture could be
        complete, and saying otherwise turns an empty read into the claim that
        this run dispatched nothing.
        """
        _seed_turn(store, "t1", [])

        summary, _ = _summarize(store, reader, ["t1"])

        assert summary["groups"] == []
        assert summary["totals"]["dispatches"] == 0
        assert summary["coverage"]["enumeration_complete"] is True
        assert summary["coverage"]["capture"] == cs.COVERAGE_NONE
        assert summary["coverage"]["turns_examined"] == 1

    def test_a_pure_observation_list_does_not_invent_an_examined_population(self):
        """The reducer's own callers may not know how many turns were read, and
        an unknown denominator is reported as unknown rather than as the turns
        that happened to have dispatches."""
        summary = cs.summarize_commands(
            [
                cs.CommandObservation(store_id="s", turn_key="t", command_call_id="c",
                                      command_name="add_item", response_success=True,
                                      in_record=True, span_recorded=True,
                                      duration_ns=1_000)
            ]
        )

        assert summary["coverage"]["turns_examined"] is None
        assert summary["coverage"]["turns_without_dispatches"] is None
        assert summary["coverage"]["turns_with_dispatches"] == 1
        assert summary["coverage"]["capture"] == cs.COVERAGE_COMPLETE


# ----------------------------------------------------------------------
# Scope: one run exposed, no single-run assumption
# ----------------------------------------------------------------------


class TestScope:
    def test_the_summary_states_the_run_it_was_asked_for_and_the_one_it_saw(
        self, store, reader
    ):
        _seed_turn(
            store, "t1", [_call("add_item")],
            experiment_id="exp-1", task_id="task-1", attempt=2,
        )

        summary, _ = _summarize(
            store, reader, ["t1"],
            experiment_id="exp-1", task_id="task-1", attempt=2,
        )

        assert summary["scope"]["requested"] == {
            "store_id": "store-1",
            "experiment_id": "exp-1",
            "task_id": "task-1",
            "attempt": 2,
            "pass_id": None,
            "turns": ["t1"],
        }
        assert summary["scope"]["observed"]["run_count"] == 1
        assert summary["scope"]["observed"]["runs"] == [
            {"store_id": "store-1", "experiment_id": "exp-1", "task_id": "task-1",
             "attempt": 2, "pass_id": None}
        ]

    def test_observations_from_two_runs_keep_their_identities(self, store, reader):
        """The reducer does not assume one run, even though one run is what is
        exposed: the later bounded-list stage reuses this shape, and a reducer
        that flattened run identity would have to be rewritten for it."""
        _seed_turn(store, "t1", [_call("add_item")],
                   experiment_id="exp-1", task_id="task-1", attempt=1)
        _seed_turn(store, "t2", [_call("add_item")],
                   experiment_id="exp-1", task_id="task-1", attempt=2)
        observations = []
        for key, attempt in (("t1", 1), ("t2", 2)):
            ref = ExecutionRef(
                store_id="store-1", turn_keys=(key,),
                experiment_id="exp-1", task_id="task-1", attempt=attempt,
            )
            observations.extend(
                cs.observations_from_projection(
                    project_execution(ref, reader, ledger=execution_ledger,
                                      cost_rollup=cost_rollup)
                )
            )

        summary = cs.summarize_commands(observations)

        assert summary["scope"]["observed"]["run_count"] == 2
        assert _group_named(summary, "add_item")["dispatches"] == 2
        assert {c["attempt"] for c in
                _group_named(summary, "add_item")["contributors"]} == {1, 2}

    def test_a_pass_scoped_summary_counts_that_pass_and_says_what_it_set_aside(
        self, store, reader
    ):
        """Source and pass identity travel with every contributor, so a
        drill-down opens the teacher's dispatch and not the student's."""
        _seed_turn(
            store,
            "t1",
            [
                _call("add_item", pass_stamp="teacher"),
                _call("add_item", pass_stamp="student"),
                _call("list_items", pass_stamp="student"),
            ],
        )
        spans = list(reader.trace("store-1", "t1"))
        selector = [
            s for s in discover_pass_selectors(spans, attribute_key="fw.pass")
            if s.pass_id == "teacher"
        ][0]
        ref = ExecutionRef(store_id="store-1", turn_keys=("t1",), pass_id="teacher")
        projection = project_execution(
            ref, reader, ledger=execution_ledger, cost_rollup=cost_rollup,
            pass_selector=selector,
        )

        summary = cs.summarize_projection(projection)

        assert summary["totals"]["dispatches"] == 1
        assert _group_named(summary, "add_item")["contributors"][0]["pass_id"] == (
            "teacher"
        )
        # The student's two dispatches are excluded AND declared, rather than
        # dropped where a reader would read the teacher's one as the whole turn.
        assert summary["coverage"]["steps_outside_pass"] == 2
        assert summary["scope"]["observed"]["runs"][0]["pass_id"] == "teacher"

    def test_a_dispatch_with_no_recorded_pass_is_set_aside_without_being_blamed_on_one(
        self, store, reader
    ):
        """What `steps_outside_pass` actually contains.

        `project_execution` sets aside every step whose resolved pass id is not
        the selected one, and a step the span tree never stamped resolves to
        None -- so the set mixes another pass's dispatches with unattributed
        ones. Calling all of them the other pass's would invent exactly the
        attribution the pass machinery refuses to guess at.
        """
        _seed_turn(
            store,
            "t1",
            [
                _call("add_item", pass_stamp="teacher"),
                _call("list_items", pass_stamp="student"),
                # Stamped by nobody: a span with no `fw.pass`, and a span-less
                # record-only dispatch. Neither belongs to either pass.
                _call("complete_item"),
                _call("archive_item", span=False),
            ],
        )
        spans = list(reader.trace("store-1", "t1"))
        selector = [
            s for s in discover_pass_selectors(spans, attribute_key="fw.pass")
            if s.pass_id == "teacher"
        ][0]
        ref = ExecutionRef(store_id="store-1", turn_keys=("t1",), pass_id="teacher")
        projection = project_execution(
            ref, reader, ledger=execution_ledger, cost_rollup=cost_rollup,
            pass_selector=selector,
        )

        # The evidence really is mixed: one dispatch the tree stamped for the
        # OTHER pass, and two it stamped for none. And the projection labels
        # all three the same way -- `pass_id` resolves to the selected pass or
        # to None, so "another pass" is not even a distinction the set-aside
        # steps carry, let alone one the summary could report.
        aside = {step.command_name: step.pass_id
                 for step in projection.unassigned_steps}
        assert aside == {"list_items": None, "complete_item": None, None: None}

        summary = cs.summarize_projection(projection)

        assert summary["totals"]["dispatches"] == 1
        assert summary["coverage"]["steps_outside_pass"] == 3
        gap = [g for g in summary["coverage"]["capture_gaps"]
               if "not attributed to this selected pass" in g]
        assert gap, summary["coverage"]["capture_gaps"]
        assert "no recorded pass at all" in gap[0]
        # And no claim about WHICH other pass any of them was.
        assert "student" not in gap[0]


# ----------------------------------------------------------------------
# Recorded dispatch timing (`fix-9eg.3.1.4`)
# ----------------------------------------------------------------------


class TestDispatchDuration:
    def test_a_group_reports_the_order_statistics_of_what_was_timed(
        self, store, reader
    ):
        _seed_turn(
            store,
            "t1",
            [
                _call("add_item", duration_ns=1_000_000),
                _call("add_item", duration_ns=5_000_000),
                _call("add_item", duration_ns=3_000_000),
            ],
        )

        duration = _group_named(_summarize(store, reader, ["t1"])[0], "add_item")[
            "duration"
        ]

        assert duration["metric"] == cs.DURATION_METRIC
        assert duration["metric_version"] == cs.DURATION_VERSION
        assert duration["unit"] == cs.DURATION_UNIT
        assert duration["timed"] == 3
        assert duration["untimed"] == 0
        assert (duration["min_ns"], duration["median_ns"], duration["max_ns"]) == (
            1_000_000, 3_000_000, 5_000_000
        )

    def test_the_summary_publishes_no_total_and_says_it_is_not_additive(
        self, store, reader
    ):
        """The boundary the epic names: inclusive observations are not elapsed
        time, so there is no sum to read and no mean to mistake for one."""
        _seed_turn(store, "t1", [_call("add_item"), _call("add_item")])

        summary, _ = _summarize(store, reader, ["t1"])
        duration = _group_named(summary, "add_item")["duration"]

        assert duration["basis"] == cs.DURATION_BASIS == "inclusive_dispatch"
        assert duration["additive"] is False
        assert "total_ns" not in duration and "mean_ns" not in duration
        assert "total_ns" not in summary["totals"]["duration"]
        assert summary["totals"]["duration"]["additive"] is False

    def test_a_parent_and_its_inner_dispatch_are_not_added_together(
        self, store, reader
    ):
        """A parent's recorded duration already contains its child's.

        Both observations are kept and the overlap is counted, which is the
        fact that makes the numbers unaddable -- and the reason no total is
        offered for somebody to add them with.
        """
        _seed_turn(
            store,
            "t1",
            [_call("outer", duration_ns=9_000_000,
                   child_calls=[{"call_id": "t1-inner",
                                 "command_name": "outer"}])],
        )

        nested = _group_named(_summarize(store, reader, ["t1"])[0], "outer")

        assert nested["dispatches"] == 2
        assert nested["duration"]["nested_within_group"] == 1
        assert nested["duration"]["child_dispatches"] == 1
        # The child recorded no span of its own, so it is untimed rather than
        # timed at zero, and the parent's 9 ms is the only observation.
        assert nested["duration"]["timed"] == 1
        assert nested["duration"]["untimed"] == 1
        assert nested["duration"]["max_ns"] == 9_000_000

    def test_overlap_is_counted_by_identity_not_by_a_bare_call_id(self):
        """A call id is unique only inside the turn that minted it.

        One group already spans the turns of a run, so matching a parent by id
        alone would call an identically named parent in ANOTHER turn an overlap
        inside this one -- and the overlap count is the thing that tells a
        reader these durations contain each other.
        """
        def observation(turn, call, parent, store="s"):
            return cs.CommandObservation(
                store_id=store, turn_key=turn, command_call_id=call,
                command_name="add_item", parent_call_id=parent,
                duration_ns=1_000, span_recorded=True,
            )

        duration = _group_named(
            cs.summarize_commands([
                # A real nesting: parent and child in the same turn.
                observation("t1", "call-0", None),
                observation("t1", "call-1", "call-0"),
                # The same parent id in a different turn, and again in a
                # different store. Neither is inside this turn's parent.
                observation("t2", "call-1", "call-0"),
                observation("t1", "call-1", "call-0", store="other-store"),
            ]),
            "add_item",
        )["duration"]

        assert duration["nested_within_group"] == 1

    def test_a_group_nothing_timed_reports_unknown_rather_than_zero(
        self, store, reader
    ):
        _seed_turn(
            store,
            "t1",
            [_call("add_item",
                   child_calls=[{"call_id": "inner-1",
                                 "command_name": "inner_hop"}])],
        )

        duration = _group_named(_summarize(store, reader, ["t1"])[0], "inner_hop")[
            "duration"
        ]

        assert duration["timed"] == 0
        assert duration["untimed"] == 1
        assert duration["min_ns"] is None
        assert duration["median_ns"] is None
        assert duration["max_ns"] is None

    def test_one_observation_is_a_whole_group(self, store, reader):
        _seed_turn(store, "t1", [_call("add_item", duration_ns=2_500_000)])

        duration = _group_named(_summarize(store, reader, ["t1"])[0], "add_item")[
            "duration"
        ]

        assert (duration["min_ns"], duration["median_ns"], duration["max_ns"]) == (
            2_500_000, 2_500_000, 2_500_000
        )
        assert duration["timed"] == 1

    def test_an_even_count_takes_the_median_of_the_two_middle_observations(self):
        summary = cs.summarize_commands(
            [
                cs.CommandObservation(store_id="s", turn_key="t",
                                      command_call_id=f"c{index}",
                                      command_name="add_item", duration_ns=value)
                for index, value in enumerate((10, 20, 30, 41))
            ]
        )

        assert _group_named(summary, "add_item")["duration"]["median_ns"] == 25

    def test_a_negative_or_nonfinite_duration_is_untimed_and_not_zero(self):
        """A clock that went backwards is a broken measurement, and counting it
        as zero would pull a minimum to a number nothing ran in."""
        summary = cs.summarize_commands(
            [
                cs.CommandObservation(store_id="s", turn_key="t",
                                      command_call_id="good",
                                      command_name="add_item",
                                      duration_ns=6_000_000,
                                      span_recorded=True),
                cs.CommandObservation(store_id="s", turn_key="t",
                                      command_call_id="backwards",
                                      command_name="add_item",
                                      duration_ns=-5, span_recorded=True),
                cs.CommandObservation(store_id="s", turn_key="t",
                                      command_call_id="infinite",
                                      command_name="add_item",
                                      duration_ns=float("inf"),
                                      span_recorded=True),
            ]
        )

        duration = _group_named(summary, "add_item")["duration"]
        assert duration["timed"] == 1
        assert duration["untimed"] == 2
        assert duration["invalid"] == 2
        assert duration["min_ns"] == duration["max_ns"] == 6_000_000
        assert summary["coverage"]["capture"] == cs.COVERAGE_PARTIAL
        assert any("not usable" in gap for gap in summary["coverage"]["capture_gaps"])

    def test_a_fractional_duration_is_refused_rather_than_truncated(self):
        """A group's minimum must be a duration some dispatch actually
        recorded.

        Truncating 1.5 ns into the statistics while the contributor still says
        1.5 would put two different numbers for one dispatch on one screen, so
        a value the recorder cannot have produced is invalid instead.
        """
        summary = cs.summarize_commands(
            [
                cs.CommandObservation(store_id="s", turn_key="t",
                                      command_call_id="fraction",
                                      command_name="add_item", duration_ns=1.5,
                                      span_recorded=True),
                cs.CommandObservation(store_id="s", turn_key="t",
                                      command_call_id="whole",
                                      command_name="add_item", duration_ns=4.0,
                                      span_recorded=True),
            ]
        )

        group = _group_named(summary, "add_item")
        assert group["duration"]["timed"] == 1
        assert group["duration"]["invalid"] == 1
        assert group["duration"]["min_ns"] == group["duration"]["max_ns"] == 4
        by_call = {c["command_call_id"]: c for c in group["contributors"]}
        # Contributor and statistic are the same number and the same type.
        assert by_call["whole"]["duration_ns"] == 4
        assert isinstance(by_call["whole"]["duration_ns"], int)
        assert by_call["fraction"]["duration_ns"] is None
        assert by_call["fraction"]["duration_invalid"] is True

    def test_every_published_statistic_is_a_value_some_contributor_recorded(
        self, store, reader
    ):
        """The general form of the same claim, over real recorded evidence:
        min and max are observations, not derived numbers."""
        _seed_turn(
            store, "t1",
            [_call("add_item", duration_ns=2_000_000),
             _call("add_item", duration_ns=9_000_000),
             _call("add_item", duration_ns=5_000_000)],
        )

        group = _group_named(_summarize(store, reader, ["t1"])[0], "add_item")

        recorded = {c["duration_ns"] for c in group["contributors"]}
        assert group["duration"]["min_ns"] in recorded
        assert group["duration"]["max_ns"] in recorded
        assert group["duration"]["median_ns"] in recorded

    def test_a_span_recorded_without_a_usable_duration_is_named_as_a_gap(
        self, store, reader
    ):
        """Separately from the span-less dispatches, which have no duration by
        definition: this is evidence the trace held and did not time."""
        summary = cs.summarize_commands(
            [
                cs.CommandObservation(store_id="s", turn_key="t",
                                      command_call_id="c", command_name="add_item",
                                      response_success=True, in_record=True,
                                      span_recorded=True, duration_ns=None)
            ]
        )

        assert any("carry no usable duration" in gap
                   for gap in summary["coverage"]["capture_gaps"])

    def test_the_underlying_durations_stay_on_the_contributors_for_a_later_merge(
        self, store, reader
    ):
        """Medians of medians are the failure this keeps open the door against:
        the observations themselves survive on the contributors, so a later
        bounded multi-run summary can take the median of the DURATIONS."""
        _seed_turn(
            store, "t1",
            [_call("add_item", duration_ns=1_000_000),
             _call("add_item", duration_ns=7_000_000)],
        )

        group = _group_named(_summarize(store, reader, ["t1"])[0], "add_item")

        assert sorted(c["duration_ns"] for c in group["contributors"]) == [
            1_000_000, 7_000_000
        ]
        assert all(c["duration_recorded"] for c in group["contributors"])
        # And recomputing the group's statistics from the contributors gives
        # the published ones back, which is what makes the merge possible.
        merged = cs.summarize_commands(
            [
                cs.CommandObservation(
                    store_id=c["store_id"], turn_key=c["turn_key"],
                    command_call_id=c["command_call_id"],
                    command_name=c["command_name"],
                    duration_ns=c["duration_ns"],
                )
                for c in group["contributors"]
            ]
        )
        assert _group_named(merged, "add_item")["duration"]["median_ns"] == (
            group["duration"]["median_ns"]
        )


# ----------------------------------------------------------------------
# The HTTP surface, over real routes
# ----------------------------------------------------------------------


def _run_attempt(store, controller, experiment_id, task_id, attempt, turns,
                 *, outcome="pass"):
    channel = f"ch-{attempt}"
    conversation = store.mint_conversation_id(
        channel, experiment_id=experiment_id, task_id=task_id, attempt=attempt
    )
    controller.start_attempt(
        experiment_id, task_id, attempt, channel, conversation_id=conversation
    )
    for ordinal, calls in enumerate(turns, start=1):
        _seed_turn(
            store,
            f"cmd-a{attempt}-t{ordinal}",
            calls,
            experiment_id=experiment_id,
            task_id=task_id,
            attempt=attempt,
            conversation=conversation,
            ordinal=ordinal,
        )
    controller.finish_attempt(
        experiment_id, task_id, attempt, outcome=outcome, outcome_source="derived"
    )


@pytest.fixture
def command_world(tmp_path, monkeypatch):
    """One workflow, one task, two recorded attempts of it in one database.

    Attempt 1 dispatched `add_item` twice (one of them recorded as failed), a
    `list_items` nothing recorded an outcome for, and an inner hop with no span
    of its own. Attempt 2 is the quiet one. That is enough for the page to show
    a difference and for the counts to be checkable by hand.
    """
    monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "state"))
    folder = tmp_path / "roster_workflow"
    folder.mkdir()
    (folder / "_commands").mkdir()
    benchmark = setup.save_benchmark(
        folder, {"title": "Roster review", "tasks": [{"prompt": "Review the roster"}]}
    )
    experiment = setup.create_experiment(
        folder, benchmark["benchmark_id"], "v1", runs_per_task=2
    )
    experiment_id = experiment["experiment_id"]
    task_id = experiment["task_ids"][0]
    # The workflow's DEFAULT database, which is the source the page is pointed
    # at: a drill-down from the summary reads the evidence the same way the
    # rest of the page does, so the link is tested against a real source rather
    # than against a store only the API knows how to reach.
    db = state_paths.observability_db(str(folder))
    Path(db).parent.mkdir(parents=True, exist_ok=True)
    store = obs.ObservabilityStore(db)
    controller = ExperimentController(
        db, store.store_identity(), external=False, workflow_folderpath=str(folder)
    )
    controller.create_experiment(
        experiment_id,
        experiment["description"],
        declared_tasks=1,
        declared_attempts=2,
        declarations=[(task_id, n, f"ch-{n}") for n in (1, 2)],
        workflow_name=setup.workflow_name_for(folder),
    )
    _run_attempt(
        store, controller, experiment_id, task_id, 1,
        [
            [
                _call("add_item"),
                _call("add_item", response_success=False, span_success=False,
                      status=tracing.STATUS_ERROR, duration_ns=7_000_000),
                _call("list_items", recorded_output=False,
                      child_calls=[{"call_id": "inner-1",
                                    "command_name": "inner_hop"}]),
            ],
            [_call("complete_item", duration_ns=3_000_000)],
        ],
    )
    _run_attempt(
        store, controller, experiment_id, task_id, 2,
        [[_call("add_item"), _call("complete_item")]],
    )
    return {
        "folder": str(folder),
        "experiment_id": experiment_id,
        "task_id": task_id,
        "store": store,
        "db": db,
    }


@pytest.fixture
def command_server(command_world):
    srv = run_chatbot_server.ChatbotServer(
        db_path=command_world["db"],
        workflow_path=command_world["folder"],
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


def _get(world, path, **params):
    query = {key: [str(value)] for key, value in params.items() if value is not None}
    return selection_api.handle_get(world["folder"], path, query)


class TestTheApiExposesTheSameSummary:
    def test_one_run_carries_its_command_summary(self, command_world):
        status, payload = _get(command_world, _task_path(command_world, "/runs/1"))

        assert status == 200
        summary = payload["projection"]["command_summary"]
        assert summary["metric_version"] == cs.METRIC_VERSION
        assert _group_named(summary, "add_item")["response_success"] == {
            "true": 1, "false": 1, "unknown": 0
        }
        assert _group_named(summary, "list_items")["response_success"]["unknown"] == 1
        assert summary["scope"]["observed"]["run_count"] == 1
        assert summary["scope"]["requested"]["attempt"] == 1

    def test_the_default_answers_view_carries_it_even_though_steps_are_trimmed(
        self, command_world
    ):
        """The summary is derived before the view trims, so the view a person
        lands on is not the one view without it."""
        answers = _get(command_world, _task_path(command_world, "/runs/1"))[1]
        steps = _get(
            command_world, _task_path(command_world, "/runs/1"), view="steps"
        )[1]

        assert "steps" not in answers["projection"]
        assert answers["projection"]["command_summary"] == (
            steps["projection"]["command_summary"]
        )

    def test_both_compared_sides_are_summarised_separately(self, command_world):
        status, payload = _get(
            command_world, _task_path(command_world, "/comparison"),
            left_attempt=1, right_attempt=2,
        )

        assert status == 200
        left = payload["left"]["command_summary"]
        right = payload["right"]["command_summary"]
        assert _group_named(left, "add_item")["dispatches"] == 2
        assert _group_named(right, "add_item")["dispatches"] == 1
        # Separately per side: no pooled figure anywhere in the payload.
        assert left["scope"]["requested"]["attempt"] == 1
        assert right["scope"]["requested"]["attempt"] == 2
        assert "command_summary" not in payload

    def test_the_counts_match_the_steps_the_same_payload_returns(
        self, command_world
    ):
        """One accounting. A second tally of the same evidence is how a chip
        and an agent's structured read come to disagree."""
        payload = _get(
            command_world, _task_path(command_world, "/runs/1"), view="steps"
        )[1]
        projection = payload["projection"]

        summary = projection["command_summary"]
        assert summary["totals"]["dispatches"] == len(projection["steps"])
        for group in summary["groups"]:
            recorded = [
                step for step in projection["steps"]
                if step["command_name"] == group["command_name"]
            ]
            assert len(recorded) == group["dispatches"], group["command_name"]
            assert {c["command_call_id"] for c in group["contributors"]} == {
                step["command_call_id"] for step in recorded
            }

    def test_the_summary_survives_a_real_socket_and_stays_additive(
        self, command_server, command_world
    ):
        status, payload = _request(
            command_server, _task_path(command_world, "/runs/1")
        )

        assert status == 200
        # Everything an existing client reads is still exactly where it was.
        for key in ("ref", "turns", "answers", "artifacts", "timing", "cost",
                    "usage", "unavailable", "step_count"):
            assert key in payload["projection"], key
        assert payload["projection"]["command_summary"]["totals"]["dispatches"] == 5

    def test_the_api_carries_the_same_timing_the_page_shows(self, command_world):
        """One definition of the timing metric for both clients, and the same
        inclusive-basis label travelling with it."""
        payload = _get(command_world, _task_path(command_world, "/runs/1"))[1]
        summary = payload["projection"]["command_summary"]

        duration = _group_named(summary, "add_item")["duration"]
        assert duration["metric_version"] == cs.DURATION_VERSION
        assert duration["basis"] == "inclusive_dispatch"
        assert duration["timed"] == 2
        # Seeded 1 ms and 7 ms, so the exact observations are readable back.
        assert duration["min_ns"] == 1_000_000
        assert duration["max_ns"] == 7_000_000
        assert summary["totals"]["duration"]["untimed"] == 1, (
            "the span-less inner hop is untimed, not zero"
        )

    def test_the_summary_is_json_serialisable_as_the_wire_needs(
        self, command_world
    ):
        payload = _get(command_world, _task_path(command_world, "/runs/1"))[1]

        assert json.loads(json.dumps(payload["projection"]["command_summary"]))


# ----------------------------------------------------------------------
# The shipped page, in a real DOM
# ----------------------------------------------------------------------


def test_the_page_shows_command_outcomes_in_a_real_dom(command_server,
                                                       command_world):
    """The table a person reads, driven through the real page against the real
    server: the per-side counts, the drill-down into exact dispatches, and the
    coverage a partly recorded run must not hide."""
    jsdom_root = os.environ.get("TEST_JSDOM_ROOT")
    if not jsdom_root:
        pytest.skip("Set TEST_JSDOM_ROOT to run DOM integration with jsdom")
    script = Path(__file__).with_name("chatbot_command_summary_dom.cjs")
    result = subprocess.run(
        [
            "node", str(script), jsdom_root,
            f"http://127.0.0.1:{command_server.port}/?token={command_server.token}",
            command_world["experiment_id"],
            command_world["task_id"],
            "cmd-a1-t1",
        ],
        capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr


class TestThePageKeepsTheDistinctionsTheDomRunCannotReach:
    """Cheap guards beside the behavioural run, for the two claims a green DOM
    result would not by itself pin: that the exact-dispatch link goes through
    the shared span helper, and that the page reads the server's tally rather
    than deriving a second one."""

    def test_the_drilldown_asks_for_the_exact_span_where_one_was_recorded(self):
        """The shared helper, called directly.

        Pinned beside the DOM run rather than instead of it: the browser check
        proves the click lands on the dispatch, and this proves it got there
        through the one navigation helper every contributor list shares, rather
        than through a second copy of the source-scoping rules.
        """
        source = run_chatbot_server.load_index_html()

        assert b"openPairSpan(ctx, side, contributor.turn_key, contributor.span_id, note)" in source
        # And the turn-scoped fallback stays only for a dispatch with no span.
        assert b"var focusable = !!contributor.span_id;" in source
        # No silent downgrade: a missing shared helper must fail, not quietly
        # open the whole turn as though it were the dispatch.
        assert b'typeof openPairSpan === "function"' not in source

    def test_the_page_renders_the_servers_summary_and_does_not_recount(self):
        source = run_chatbot_server.load_index_html()

        assert b"function renderCommandSummary(container, cmp, ctx)" in source
        assert b"projection.command_summary" in source
        # The three outcomes are read off the reducer's own keys, so a page
        # that disagreed with an agent's read would have to disagree with the
        # payload both of them were handed.
        assert b"outcomes.unknown || 0" in source

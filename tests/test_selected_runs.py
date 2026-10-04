"""Aggregating explicitly selected whole runs of one task (`fix-9eg.3.2.1`).

Integration throughout, per `.cursor/rules/testing_rules.mdc`: real
`ObservabilityStore` databases on disk, attempts written through the real
`ExperimentController`, real span rows and real turn records, the real shared
selection control, the real HTTP server over a real socket and a real sealed
archive. No mocks: the whole question here is whether pooled figures agree
with the evidence they claim to be about, and a fake store would not test it.

What these tests defend, in order of how badly each would mislead:

- The POPULATION is explicit. A run that was asked for and is not in the
  numbers has to be named -- unfinished, never recorded, or finished with
  evidence that cannot be read -- and the last of those stays a member, so an
  unreadable run cannot quietly shrink a denominator.
- The ARITHMETIC pools observations, not summaries. Four dispatches spread
  three-and-one across two runs have one median, and it is the median of the
  four durations rather than the average of the two runs' medians.
- The EVIDENCE is exactly what was selected. Contributors name their own run,
  and no unselected attempt's evidence appears anywhere in the answer.
- The SCOPE cannot be widened by a parameter. A request that names a pass, a
  second experiment or a store is refused, not quietly answered narrower.
- A CHANGE inside an already-named turn is disclosed. The validation route
  re-projects the evidence, so an outcome, a duration or a cost edited in
  place -- with every identifier unchanged -- reads as stale.
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
from fastworkflow.observability import best_run, comparison
from fastworkflow.observability import selected_runs as sr
from fastworkflow.observability import store as obs
from fastworkflow.run_chatbot import selection_api
from fastworkflow.run_chatbot import server as run_chatbot_server
from tests.test_chatbot_benchmarks import _request
from tests.test_execution_comparison import (
    _execute_span,
    _output,
    _record,
    _write,
)
from tests.test_execution_comparison import _turn_row as _evidence_turn_row
from tests.test_selection_api import _sealed_server
from tests.test_usage_and_cache_rollups import _llm_call

# `world` and `server` are the two-experiment, two-database workflow the
# selection API tests already build: attempt 1 completed, attempt 2 failed,
# attempt 3 never finished, attempt 4 finished with no turns, and the SAME
# task recorded under a second experiment in a second store.
pytest_plugins = ["tests.test_selection_api"]

T0 = 1_800_000_000_000_000_000


# ----------------------------------------------------------------------
# A task whose runs recorded unequal numbers of dispatches
# ----------------------------------------------------------------------


def _seed(store, turn_key, *, experiment_id, task_id, attempt, conversation,
          ordinal, calls, llm=(), answer="done", success=True):
    """One turn, with the exact durations and outcomes each dispatch recorded.

    `calls` are `(command, duration_ns, response_success)`. The duration is
    the span's, the outcome is the turn record's `CommandOutput`, and they are
    written separately here for the same reason the reducer keeps them apart.
    """
    refs, spans, outputs = [], [], []
    for index, (command, duration_ns, response_success) in enumerate(calls):
        call_id = f"{turn_key}-call-{index}"
        span_id = f"{turn_key}-span-{index}"
        refs.append((call_id, index, span_id))
        outputs.append(
            _output(call_id, command, {"n": index}, success=response_success)
        )
        spans.append(
            _execute_span(
                span_id,
                turn_key,
                call_id=call_id,
                command_name=command,
                start_ns=T0 + index * 10_000_000,
                duration_ns=duration_ns,
                success=response_success,
                status=tracing.STATUS_OK if response_success else tracing.STATUS_ERROR,
            )
        )
    spans.extend(llm)
    row = _evidence_turn_row(
        turn_key,
        record=_record(turn_key, refs=refs, outputs=outputs, success=success),
        answer=answer,
        success=success,
        experiment_id=experiment_id,
        task_id=task_id,
        attempt=attempt,
    )
    row["conversation_id"] = conversation
    row["ordinal"] = ordinal
    _write(store, row, spans)
    return row, spans


@pytest.fixture
def runs_world(tmp_path, monkeypatch):
    """One experiment, one task, four runs of deliberately unequal shape.

    Attempt 1 dispatched `cmd_a` three times (10, 20 and 30 microseconds) and
    recorded one priced LLM call beside one that recorded no cost. Attempt 2
    dispatched `cmd_a` once and took ten times as long (100 microseconds),
    failed a `cmd_b`, and recorded a zero-cost call whose response a second
    call re-used. Attempt 3 is still running. Attempt 4 finished having
    recorded nothing.

    The durations are chosen so that the pooled median (25 microseconds over
    the four observations) and the median of the two runs' medians (60) are
    different numbers, which is the whole point of pooling observations.
    """
    monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "state"))
    folder = tmp_path / "selected_runs_workflow"
    folder.mkdir()
    (folder / "_commands").mkdir()
    benchmark = setup.save_benchmark(
        folder, {"title": "Pooling", "tasks": [{"prompt": "Do the thing"}]}
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
        declarations=[(task_id, n, f"ch-{n}") for n in range(1, 5)],
        workflow_name=setup.workflow_name_for(folder),
    )

    conversations: dict[int, str] = {}

    def start(attempt):
        channel = f"ch-{attempt}"
        conversation = store.mint_conversation_id(
            channel, experiment_id=experiment_id, task_id=task_id, attempt=attempt
        )
        controller.start_attempt(
            experiment_id, task_id, attempt, channel, conversation_id=conversation
        )
        conversations[attempt] = conversation
        return conversation

    conversation = start(1)
    _seed(
        store, "sel-a1-t1",
        experiment_id=experiment_id, task_id=task_id, attempt=1,
        conversation=conversation, ordinal=1,
        calls=[("cmd_a", 10_000, True), ("cmd_a", 20_000, True),
               ("cmd_a", 30_000, True)],
        llm=[
            _llm_call("sel-a1-llm-1", "sel-a1-t1", start_ns=T0,
                      usage={"prompt_tokens": 10, "completion_tokens": 5},
                      cost=0.01, history_uuid="h-a1-1"),
            # Recorded no cost at all: unknown, and never counted as zero.
            _llm_call("sel-a1-llm-2", "sel-a1-t1", start_ns=T0 + 1_000_000,
                      usage={"prompt_tokens": 3, "completion_tokens": 1},
                      history_uuid="h-a1-2"),
        ],
    )
    controller.finish_attempt(
        experiment_id, task_id, 1, outcome="pass", outcome_source="test"
    )

    conversation = start(2)
    _seed(
        store, "sel-a2-t1",
        experiment_id=experiment_id, task_id=task_id, attempt=2,
        conversation=conversation, ordinal=1,
        calls=[("cmd_a", 100_000, True), ("cmd_b", 5_000, False)],
        llm=[
            # A recorded zero is a measurement, not a silence.
            _llm_call("sel-a2-llm-1", "sel-a2-t1", start_ns=T0,
                      usage={"prompt_tokens": 0, "completion_tokens": 0},
                      cost=0.0, history_uuid="h-a2"),
            # The same provider response, re-used: folded, never charged twice.
            _llm_call("sel-a2-llm-2", "sel-a2-t1", start_ns=T0 + 1_000_000,
                      usage={"prompt_tokens": 0, "completion_tokens": 0},
                      cost=0.0, history_uuid="h-a2"),
        ],
        success=False, answer="it failed",
    )
    controller.finish_attempt(
        experiment_id, task_id, 2, outcome="fail", outcome_source="test",
        execution_status="failed",
    )

    conversation = start(3)
    _seed(
        store, "sel-a3-t1",
        experiment_id=experiment_id, task_id=task_id, attempt=3,
        conversation=conversation, ordinal=1,
        calls=[("cmd_a", 999_000, True)],
    )
    # Deliberately never finished.

    start(4)
    controller.finish_attempt(
        experiment_id, task_id, 4, outcome="pass", outcome_source="test"
    )

    return {
        "folder": str(folder),
        "db": db,
        "experiment_id": experiment_id,
        "task_id": task_id,
        "store": store,
        "conversations": conversations,
    }


@pytest.fixture
def runs_server(runs_world):
    srv = run_chatbot_server.ChatbotServer(
        db_path=runs_world["db"],
        workflow_path=runs_world["folder"],
        port=0,
        spawn_options={"no_server": True},
    )
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    thread.join(timeout=5)


def _write_spans(store, spans):
    """Re-record spans only, through the store's own writer.

    Same span ids, same turn: what the trace SAYS changes and nothing that
    names it does. A completed turn row is immutable (`[R2]`), which is why
    the record-level edit below goes in as the repair of a database would.
    """
    with store._connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        store.upsert_span_rows(conn, spans, store._store_redactor())
        conn.commit()


def _edit_recorded_outcome(store, turn_key, index, success):
    """Change what one dispatch's recorded `CommandOutput` says, in place.

    The store refuses to rewrite a completed turn through `upsert_turn_row`,
    so this is done the way the evidence would actually change under a
    reader: the row is edited in the database. Nothing else moves -- same
    turn key, same call id, same span id, same everything the selection
    names -- which is precisely the change an identity digest cannot see.
    """
    with store._connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        raw = conn.execute(
            "SELECT record_json FROM turns WHERE turn_key=?", (turn_key,)
        ).fetchone()
        record = json.loads(raw["record_json"])
        record["turn_output"]["command_outputs"][index]["command_response"][
            "success"
        ] = success
        conn.execute(
            "UPDATE turns SET record_json=? WHERE turn_key=?",
            (json.dumps(record), turn_key),
        )
        conn.commit()


def _path(world, suffix=""):
    return (
        f"/api/experiments/{world['experiment_id']}"
        f"/tasks/{world['task_id']}/selected-runs{suffix}"
    )


def _selected(world, *attempts, suffix="", **extra):
    query = {"attempt": [str(attempt) for attempt in attempts]}
    for key, value in extra.items():
        query[key] = value if isinstance(value, list) else [str(value)]
    return selection_api.handle_get(world["folder"], _path(world, suffix), query)


# ----------------------------------------------------------------------
# Scope: what the route refuses to be asked
# ----------------------------------------------------------------------


class TestScope:
    def test_a_parameter_this_route_does_not_implement_is_refused(self, runs_world):
        """Ignoring `left_pass` would answer the whole-run question and label
        it with the caller's words for the pass question."""
        for parameter in ("left_pass", "right_experiment", "store_id", "view",
                          "pass_attribute", "turn_key", "limit"):
            status, payload = _selected(runs_world, 1, **{parameter: "x"})
            assert status == 400, parameter
            assert payload["refused"] == "unsupported_parameter"
            assert payload["unsupported"] == [parameter]
            # `scope` joined the accepted list with the all-finished rule
            # (`fix-9eg.3.2.2.1`); everything else is still refused by name.
            assert payload["accepted"] == ["attempt", "scope"]

    def test_naming_no_run_is_refused_rather_than_answered_over_nothing(
        self, runs_world
    ):
        status, payload = _selected(runs_world)
        assert status == 400
        assert payload["refused"] == "no_runs_selected"

    def test_more_runs_than_the_bound_is_refused_not_sampled(self, runs_world):
        status, payload = _selected(runs_world, *range(1, 22))
        assert status == 400
        assert payload["refused"] == "too_many_runs"
        assert payload["max_runs"] == sr.MAX_SELECTED_RUNS == 20
        assert payload["requested"] == 21
        # Nothing was summarized: a bound that silently returned the first
        # twenty would publish a figure over a population nobody chose.
        assert "command_summary" not in payload and "members" not in payload

    def test_an_attempt_that_is_not_an_exact_integer_is_refused(self, runs_world):
        status, payload = _selected(runs_world, "2.9")
        assert status == 400 and "exact integer" in payload["error"]


# ----------------------------------------------------------------------
# Population: requested, included, excluded, and what stays in a denominator
# ----------------------------------------------------------------------


class TestPopulation:
    def test_duplicates_unfinished_and_unrecorded_runs_are_each_named(
        self, runs_world
    ):
        status, payload = _selected(runs_world, 1, 1, 2, 3, 4, 99)
        assert status == 200
        scope = payload["scope"]
        assert scope["requested_attempts"] == [1, 2, 3, 4, 99]
        assert scope["duplicate_requests"] == 1
        population = payload["population"]
        assert population["requested"] == 5
        assert population["unfinished"] == [3]
        assert population["not_recorded"] == [99]
        # Finished with unreadable evidence: a MEMBER, so it cannot vanish
        # from a denominator, contributing nothing and saying so.
        assert population["missing_evidence"] == [4]
        assert population["included"] == 3
        assert population["with_readable_evidence"] == 2
        assert [row["attempt"] for row in payload["members"]] == [1, 2, 4]
        reasons = {row["attempt"]: row["reason"]
                   for row in population["excluded_runs"]}
        assert reasons == {3: "unfinished", 99: "not_recorded"}

    def test_a_finished_run_with_no_evidence_is_a_member_with_nothing_in_it(
        self, runs_world
    ):
        status, payload = _selected(runs_world, 4)
        assert status == 200
        member = payload["members"][0]
        assert member["attempt"] == 4 and member["finished"] is True
        assert member["execution_ref"] is None and member["dispatches"] == 0
        assert payload["population"]["included"] == 1
        assert payload["run_outcomes"]["runs_without_readable_evidence"] == 1
        # Nothing recorded a cost, so the total is unknown and never zero.
        assert payload["cost"]["total"] is None
        assert payload["command_summary"]["coverage"]["capture"] == "none"

    def test_a_run_with_no_evidence_lowers_the_coverage_beside_the_counts(
        self, runs_world
    ):
        """Attempt 4 names no turn, so the reducer's turn-level coverage
        cannot see it. Left alone, a five-run selection missing two runs
        would print `enumeration_complete: true` next to a population that
        says two are missing."""
        payload = _selected(runs_world, 1, 4)[1]
        coverage = payload["command_summary"]["coverage"]
        assert coverage["runs_requested"] == 2
        assert coverage["runs_with_readable_evidence"] == 1
        assert coverage["runs_missing_evidence"] == [4]
        assert coverage["population_complete"] is False
        assert coverage["enumeration_complete"] is False
        assert coverage["capture"] == "partial"
        assert any("attempt 4" in gap for gap in coverage["capture_gaps"])
        # No invented turn count: nobody knows how many turns a run that
        # recorded none would have had.
        assert coverage["turns_examined"] == 1

    def test_a_selection_whose_every_run_recorded_nothing_says_so(self, runs_world):
        coverage = _selected(runs_world, 4)[1]["command_summary"]["coverage"]
        assert coverage["capture"] == "none"
        assert coverage["population_complete"] is False
        assert coverage["runs_with_readable_evidence"] == 0

    def test_failed_runs_are_retained_and_kept_apart_from_failed_dispatches(
        self, runs_world
    ):
        status, payload = _selected(runs_world, 1, 2)
        assert status == 200
        outcomes = payload["run_outcomes"]
        assert outcomes["by_outcome"] == {"pass": 1, "fail": 1}
        assert outcomes["by_execution_status"]["failed"] == 1
        # One run contains a failed dispatch; that is not the same count as
        # the number of failed runs, and both are reported.
        assert outcomes["runs_with_failed_dispatches"] == 1
        assert payload["command_summary"]["totals"]["response_success"] == {
            "true": 4, "false": 1, "unknown": 0
        }


# ----------------------------------------------------------------------
# Arithmetic over pooled observations
# ----------------------------------------------------------------------


class TestPooling:
    def test_the_median_is_over_the_durations_not_over_the_run_medians(
        self, runs_world
    ):
        status, payload = _selected(runs_world, 1, 2)
        assert status == 200
        groups = {group["command_name"]: group
                  for group in payload["command_summary"]["groups"]}
        duration = groups["cmd_a"]["duration"]
        assert groups["cmd_a"]["dispatches"] == 4
        assert duration["min_ns"] == 10_000 and duration["max_ns"] == 100_000
        # 10, 20, 30, 100 pooled -> 25. The median of the runs' own medians
        # (20 and 100) would be 60, which is the figure this refuses to be.
        assert duration["median_ns"] == 25_000
        assert duration["timed"] == 4 and duration["untimed"] == 0

    def test_the_summary_is_the_same_whatever_order_the_runs_were_named_in(
        self, runs_world
    ):
        first = _selected(runs_world, 1, 2)[1]
        second = _selected(runs_world, 2, 1)[1]
        assert first["evidence_digest"] == second["evidence_digest"]
        for payload in (first, second):
            payload.pop("generated_at")
        assert first == second

    def test_every_contributor_names_its_own_run_and_no_other(self, runs_world):
        status, payload = _selected(runs_world, 1, 2)
        assert status == 200
        contributors = [
            contributor
            for group in payload["command_summary"]["groups"]
            for contributor in group["contributors"]
        ]
        assert {contributor["attempt"] for contributor in contributors} == {1, 2}
        assert {contributor["task_id"] for contributor in contributors} == {
            runs_world["task_id"]
        }
        # The unselected runs' evidence is nowhere in the answer, neither as a
        # contributor nor as a turn key: attempt 3 is still running and
        # attempt 4 recorded nothing.
        assert "sel-a3" not in json.dumps(payload)
        assert [member["ref_id"] for member in payload["members"][:2]] == [
            member["execution_ref"]["ref_id"] for member in payload["members"][:2]
        ]

    def test_the_selection_scope_travels_with_the_summary(self, runs_world):
        payload = _selected(runs_world, 1, 2)[1]
        observed = payload["command_summary"]["scope"]["observed"]
        assert observed["run_count"] == 2
        assert {row["attempt"] for row in observed["runs"]} == {1, 2}
        requested = payload["command_summary"]["scope"]["requested"]
        assert requested["kind"] == "selected_runs"
        assert requested["attempts"] == [1, 2]


# ----------------------------------------------------------------------
# Recorded LLM cost: canonical, never allocated, never inferred
# ----------------------------------------------------------------------


class TestCost:
    def test_zero_unknown_and_shared_costs_each_keep_their_meaning(
        self, runs_world
    ):
        status, payload = _selected(runs_world, 1, 2)
        assert status == 200
        cost = payload["cost"]
        # Four recorded calls: one priced, one that recorded no cost, one
        # recorded zero, and one re-using the zero call's response.
        assert cost["calls"] == 4
        assert cost["recorded"] == 2 and cost["unrecorded"] == 2
        assert cost["total"] == pytest.approx(0.01)
        assert cost["allocated_to_commands"] is False
        assert payload["usage"]["shared_responses"] == 1
        # No per-command or per-dispatch cost anywhere: the evidence charges a
        # provider response, not a dispatch.
        for group in payload["command_summary"]["groups"]:
            assert "cost" not in group
            for contributor in group["contributors"]:
                assert "cost" not in contributor

    def test_a_complete_looking_cost_does_not_claim_a_complete_population(
        self, runs_world
    ):
        """Every call that WAS observed recorded a price. That says nothing
        about the run whose evidence could not be read."""
        payload = _selected(runs_world, 1, 4)[1]
        cost = payload["cost"]
        assert cost["members_without_readable_evidence"] == 1
        assert cost["population_complete"] is False
        assert cost["covers_observed_calls_only"] is True

    def test_each_member_carries_the_cost_its_own_run_recorded(self, runs_world):
        payload = _selected(runs_world, 1, 2)[1]
        by_attempt = {member["attempt"]: member for member in payload["members"]}
        assert by_attempt[1]["cost"]["recorded"] == 1
        assert by_attempt[1]["cost"]["total"] == pytest.approx(0.01)
        assert by_attempt[2]["cost"]["total"] == pytest.approx(0.0)
        # Anchors stay on the runs that recorded them rather than being
        # re-listed at selection scope.
        assert "calls_detail" not in by_attempt[1]["usage"]
        assert "calls_detail" not in payload["usage"]


# ----------------------------------------------------------------------
# Source boundaries
# ----------------------------------------------------------------------


class TestSources:
    def test_the_same_attempt_number_in_another_source_is_another_run(self, world):
        """Attempt 1 exists under both experiments, in two different
        databases. Each path reads the store ITS experiment is authorized
        against; there is no default store to fall back to."""
        first = selection_api.handle_get(
            world["folder"],
            f"/api/experiments/{world['experiment_id']}"
            f"/tasks/{world['task_id']}/selected-runs",
            {"attempt": ["1"]},
        )[1]
        second = selection_api.handle_get(
            world["folder"],
            f"/api/experiments/{world['candidate_id']}"
            f"/tasks/{world['task_id']}/selected-runs",
            {"attempt": ["1"]},
        )[1]
        assert first["store_id"] != second["store_id"]
        assert first["members"][0]["ref_id"] != second["members"][0]["ref_id"]
        assert first["evidence_digest"] != second["evidence_digest"]
        first_commands = {
            group["command_name"] for group in first["command_summary"]["groups"]
        }
        second_commands = {
            group["command_name"] for group in second["command_summary"]["groups"]
        }
        assert "sort_items" in second_commands
        assert "sort_items" not in first_commands

    def test_an_experiment_in_no_contest_is_refused_not_answered_from_a_default(
        self, world
    ):
        status, payload = selection_api.handle_get(
            world["folder"],
            f"/api/experiments/not-an-experiment"
            f"/tasks/{world['task_id']}/selected-runs",
            {"attempt": ["1"]},
        )
        assert status == 404 and "unknown experiment" in payload["error"]

    def test_two_stores_in_one_selection_are_refused_rather_than_pooled(
        self, runs_world
    ):
        """The live route cannot produce this -- one experiment has one
        authorized source -- so the coordinator itself is asked, because it is
        what a second caller would reach."""
        rows = [
            {"attempt": 1, "finished": True,
             "execution_ref": {"store_id": "one", "turn_keys": ["t1"],
                               "attempt": 1}},
            {"attempt": 2, "finished": True,
             "execution_ref": {"store_id": "two", "turn_keys": ["t2"],
                               "attempt": 2}},
        ]
        with pytest.raises(sr.SelectionIncoherent) as caught:
            sr.aggregate_selected_runs(
                experiment_id="e", task_id="t", source_id=None, store_id=None,
                requested=[1, 2], duplicate_requests=0, recorded_attempts=[1, 2],
                candidate_rows=rows, reader=_RefusingReader(),
            )
        assert caught.value.reason == "multiple_sources"

    def test_two_runs_naming_one_turn_are_refused_rather_than_counted_twice(
        self, runs_world
    ):
        rows = [
            {"attempt": 1, "finished": True,
             "execution_ref": {"store_id": "one", "turn_keys": ["shared"],
                               "attempt": 1}},
            {"attempt": 2, "finished": True,
             "execution_ref": {"store_id": "one", "turn_keys": ["shared"],
                               "attempt": 2}},
        ]
        with pytest.raises(sr.SelectionIncoherent) as caught:
            sr.aggregate_selected_runs(
                experiment_id="e", task_id="t", source_id=None, store_id=None,
                requested=[1, 2], duplicate_requests=0, recorded_attempts=[1, 2],
                candidate_rows=rows, reader=_RefusingReader(),
            )
        assert caught.value.reason == "overlapping_runs"


class _RefusingReader:
    """A reader for refusal tests: reaching it at all would mean the refusal
    came too late, so it answers nothing."""

    def turn(self, store_id, turn_key):
        return None

    def trace(self, store_id, turn_key):
        return []


# ----------------------------------------------------------------------
# Validation: a change inside an already-named turn
# ----------------------------------------------------------------------


class TestDigest:
    """What the digest has to notice, and what it must ignore."""

    def _rows(self, runs_world, attempts):
        control = setup.open_workflow_control(runs_world["folder"], create=False)
        resolved = best_run.select_task_attempts(
            control, runs_world["experiment_id"], runs_world["task_id"], attempts
        )
        reader = comparison.StoreExecutionReader(
            resolved["store_id"], runs_world["store"]
        )
        return resolved, reader

    def _aggregate(self, runs_world, resolved, reader, rows, **overrides):
        kwargs = {
            "experiment_id": runs_world["experiment_id"],
            "task_id": runs_world["task_id"],
            "source_id": resolved["source_id"],
            "store_id": resolved["store_id"],
            "requested": [int(row["attempt"]) for row in rows],
            "duplicate_requests": 0,
            "recorded_attempts": resolved["recorded_attempts"],
            "candidate_rows": rows,
            "reader": reader,
        }
        kwargs.update(overrides)
        return sr.aggregate_selected_runs(**kwargs)

    def test_a_run_outcome_corrected_without_touching_a_turn_is_disclosed(
        self, runs_world
    ):
        """Same turns, same dispatches, same costs, different published
        outcome -- and the run tally on the screen moves with it."""
        resolved, reader = self._rows(runs_world, [1])
        row = dict(resolved["selected"][0])
        before = self._aggregate(runs_world, resolved, reader, [row])
        after = self._aggregate(
            runs_world, resolved, reader, [dict(row, outcome="fail")]
        )
        assert before["run_outcomes"]["by_outcome"] != (
            after["run_outcomes"]["by_outcome"]
        )
        assert (
            before["members"][0]["evidence_digest"]
            != after["members"][0]["evidence_digest"]
        )
        assert before["evidence_digest"] != after["evidence_digest"]

    def test_pinning_a_different_best_run_does_not_make_the_evidence_stale(
        self, runs_world
    ):
        """`is_best` is a pointer's label about a decision, not a fact about
        what the run recorded, so it is deliberately outside the digest."""
        resolved, reader = self._rows(runs_world, [1])
        row = dict(resolved["selected"][0])
        before = self._aggregate(runs_world, resolved, reader, [row])
        after = self._aggregate(
            runs_world, resolved, reader, [dict(row, is_best=not row["is_best"])]
        )
        assert before["evidence_digest"] == after["evidence_digest"]

    def test_the_digest_is_bound_to_the_scope_even_with_no_evidence_at_all(
        self, runs_world
    ):
        """Attempt 4 recorded nothing, so every value in the summary is empty
        or zero. Two such selections under different tasks are still two
        different answers."""
        resolved, reader = self._rows(runs_world, [4])
        rows = [dict(row) for row in resolved["selected"]]
        here = self._aggregate(runs_world, resolved, reader, rows)
        elsewhere = self._aggregate(
            runs_world, resolved, reader, rows, task_id="another-task"
        )
        other_source = self._aggregate(
            runs_world, resolved, reader, rows, source_id="another-source"
        )
        assert here["members"][0]["dispatches"] == 0
        assert len({here["evidence_digest"], elsewhere["evidence_digest"],
                    other_source["evidence_digest"]}) == 3


class TestPartialEvidence:
    """Read in part is a third state, and the one that hides most easily.

    A run that named two turns and yielded one is not "no evidence" and is
    not complete: both simpler labels are false statements about the counts
    it is inside.
    """

    def _partial(self, runs_world):
        """Attempt 1's real reference, plus a turn the store no longer holds.

        A pruned turn a reference outlives is the ordinary way this happens,
        and it is the same shape the projection already reports: the readable
        turns project, the rest is named in `unavailable`.
        """
        control = setup.open_workflow_control(runs_world["folder"], create=False)
        resolved = best_run.select_task_attempts(
            control, runs_world["experiment_id"], runs_world["task_id"], [1]
        )
        row = dict(resolved["selected"][0])
        ref = dict(row["execution_ref"])
        ref["turn_keys"] = list(ref["turn_keys"]) + ["sel-a1-t2-pruned"]
        row["execution_ref"] = ref
        row["turn_count"] = len(ref["turn_keys"])
        reader = comparison.StoreExecutionReader(
            resolved["store_id"], runs_world["store"]
        )
        return sr.aggregate_selected_runs(
            experiment_id=runs_world["experiment_id"],
            task_id=runs_world["task_id"],
            source_id=resolved["source_id"],
            store_id=resolved["store_id"],
            requested=[1],
            duplicate_requests=0,
            recorded_attempts=resolved["recorded_attempts"],
            candidate_rows=[row],
            reader=reader,
        )

    def test_a_run_read_only_in_part_is_neither_complete_nor_empty(
        self, runs_world
    ):
        payload = self._partial(runs_world)
        coverage = payload["command_summary"]["coverage"]
        assert coverage["runs_partially_read"] == [1]
        assert coverage["runs_missing_evidence"] == []
        assert coverage["runs_unreadable"] == []
        assert coverage["population_complete"] is False
        assert coverage["enumeration_complete"] is False
        # The turn counts are kept, because here they ARE known.
        assert any("1 of 2 turn(s) read" in gap
                   for gap in coverage["capture_gaps"])
        # And what WAS read is still counted rather than discarded.
        assert payload["command_summary"]["totals"]["dispatches"] == 3

    def test_the_cost_of_a_partly_read_run_does_not_claim_the_whole_run(
        self, runs_world
    ):
        payload = self._partial(runs_world)
        cost = payload["cost"]
        # Every observed call recorded a price; the unread turn's calls are
        # not among the observed ones.
        assert cost["members_with_unreadable_turns"] == 1
        assert cost["members_without_readable_evidence"] == 0
        assert cost["population_complete"] is False
        assert cost["total"] == pytest.approx(0.01)

    def test_an_excluded_run_keeps_the_cost_from_claiming_the_selection(
        self, runs_world
    ):
        """Two runs asked for, one of them still running: the total is over
        the one that was summed, and says so."""
        cost = _selected(runs_world, 1, 3)[1]["cost"]
        assert cost["runs_excluded"] == 1
        assert cost["population_complete"] is False


class TestValidation:
    def test_unchanged_evidence_validates_as_unchanged(self, runs_world):
        summary = _selected(runs_world, 1, 2)[1]
        status, payload = _selected(
            runs_world, 1, 2, suffix="/validation",
            expect=summary["evidence_digest"],
        )
        assert status == 200
        assert payload["stale"] is False
        assert payload["evidence_digest"] == summary["evidence_digest"]
        assert [row["state"] for row in payload["members"]] == [
            "unchanged", "unchanged"
        ]
        assert payload["validation"] == "optimistic"

    def test_an_outcome_edited_in_place_is_disclosed(self, runs_world):
        """No identifier changes: the same turn, the same dispatch, the same
        span. Only what it recorded changes, which is exactly the case an
        identity digest would call unchanged."""
        summary = _selected(runs_world, 1, 2)[1]
        baselines = {
            member["attempt"]: member["evidence_digest"]
            for member in summary["members"]
        }
        _edit_recorded_outcome(runs_world["store"], "sel-a2-t1", 0, False)

        after = _selected(runs_world, 1, 2)[1]
        assert after["command_summary"]["totals"]["response_success"] == {
            "true": 3, "false": 2, "unknown": 0
        }
        status, payload = _selected(
            runs_world, 1, 2, suffix="/validation",
            expect=summary["evidence_digest"],
            expect_member=[
                f"{attempt}:{digest}" for attempt, digest in baselines.items()
            ],
        )
        assert status == 200
        assert payload["stale"] is True
        assert payload["changed"] == [2]
        states = {row["attempt"]: row["state"] for row in payload["members"]}
        # The run nobody touched is still unchanged: the edit is attributed
        # to the member it happened in, not to the whole selection.
        assert states == {1: "unchanged", 2: "changed"}

    def test_a_cost_edited_in_place_is_disclosed(self, runs_world):
        summary = _selected(runs_world, 1)[1]
        member = summary["members"][0]
        # The same call span, re-recorded with a different price. Every
        # dispatch count above it is unchanged, so only a digest over the
        # projected VALUES can tell that this is no longer the run that was
        # summarized.
        _write_spans(
            runs_world["store"],
            [
                _llm_call("sel-a1-llm-1", "sel-a1-t1", start_ns=T0,
                          usage={"prompt_tokens": 10, "completion_tokens": 5},
                          cost=0.25, history_uuid="h-a1-1"),
            ],
        )

        after = _selected(runs_world, 1)[1]
        assert after["command_summary"]["totals"]["response_success"] == (
            summary["command_summary"]["totals"]["response_success"]
        )
        assert after["cost"]["total"] == pytest.approx(0.25)
        payload = _selected(
            runs_world, 1, suffix="/validation",
            expect_member=[f"1:{member['evidence_digest']}"],
        )[1]
        assert payload["stale"] is True and payload["changed"] == [1]

    def test_one_member_can_be_revalidated_on_its_own(self, runs_world):
        """A drill-down opens one run, so it re-projects one run."""
        summary = _selected(runs_world, 1, 2)[1]
        member = next(m for m in summary["members"] if m["attempt"] == 1)
        payload = _selected(
            runs_world, 1, suffix="/validation",
            expect_member=[f"1:{member['evidence_digest']}"],
        )[1]
        assert payload["stale"] is False
        assert [row["attempt"] for row in payload["members"]] == [1]

    def test_a_run_that_is_no_longer_a_member_reads_as_missing(self, runs_world):
        payload = _selected(
            runs_world, 3, 99, suffix="/validation",
        )[1]
        assert payload["missing"] == [3, 99]
        assert payload["stale"] is True
        assert payload["unfinished"] == [3] and payload["not_recorded"] == [99]

    def test_a_check_with_no_baseline_never_reports_unchanged(self, runs_world):
        """"Nothing has changed" is a claim, and a request that supplied
        nothing to compare against has not made it."""
        payload = _selected(runs_world, 1, 2, suffix="/validation")[1]
        assert payload["compared"] is False
        assert payload["stale"] is None
        assert payload["not_compared"] == [1, 2]
        assert {row["state"] for row in payload["members"]} == {"not_compared"}

    def test_a_result_digest_that_no_longer_matches_names_nobody_unchanged(
        self, runs_world
    ):
        summary = _selected(runs_world, 1, 2)[1]
        # A duration corrected on one span: the dispatch, its outcome and its
        # ids are untouched, and the pooled median moves.
        _write_spans(
            runs_world["store"],
            [
                _execute_span(
                    "sel-a1-t1-span-0", "sel-a1-t1",
                    call_id="sel-a1-t1-call-0", command_name="cmd_a",
                    start_ns=T0, duration_ns=11_000,
                ),
            ],
        )
        assert (
            _selected(runs_world, 1, 2)[1]["evidence_digest"]
            != summary["evidence_digest"]
        )
        payload = _selected(
            runs_world, 1, 2, suffix="/validation",
            expect=summary["evidence_digest"],
        )[1]
        assert payload["stale"] is True
        # The result digest says something moved, not which run did, so a
        # member with no baseline of its own is not called unchanged.
        assert {row["state"] for row in payload["members"]} == {"not_compared"}

    def test_an_expectation_about_a_run_outside_the_selection_is_refused(
        self, runs_world
    ):
        status, payload = _selected(
            runs_world, 1, suffix="/validation", expect_member=["2:mev-whatever"]
        )
        assert status == 400
        assert payload["refused"] == "unrelated_expectation"
        assert payload["attempts"] == [2]

    def test_a_malformed_expectation_is_refused(self, runs_world):
        for value in ("nonsense", "1:", ":mev-x"):
            status, payload = _selected(
                runs_world, 1, suffix="/validation", expect_member=[value]
            )
            assert status == 400, value
            assert "expect_member" in payload["error"]

    def test_the_answer_says_it_does_not_freeze_the_evidence(self, runs_world):
        payload = _selected(runs_world, 1, suffix="/validation")[1]
        assert payload["validation"] == "optimistic"
        assert "does not freeze" in payload["guarantee"]

    def test_validation_refuses_the_same_widening_parameters(self, runs_world):
        status, payload = _selected(
            runs_world, 1, suffix="/validation", left_pass="teacher"
        )
        assert status == 400 and payload["unsupported"] == ["left_pass"]


# ----------------------------------------------------------------------
# Over a real socket, and over a sealed archive
# ----------------------------------------------------------------------


class TestOverHttp:
    def test_a_coding_agent_reads_the_same_summary_the_page_does(self, world, server):
        path = (
            f"/api/experiments/{world['experiment_id']}"
            f"/tasks/{world['task_id']}/selected-runs?attempt=1&attempt=2"
        )
        status, payload = _request(server, path)
        assert status == 200
        assert [member["attempt"] for member in payload["members"]] == [1, 2]
        assert payload["command_summary"]["totals"]["dispatches"] == 5
        status, refused = _request(server, path + "&right_experiment=other")
        assert status == 400 and refused["refused"] == "unsupported_parameter"

    def test_validation_answers_over_http_too(self, world, server):
        base = (
            f"/api/experiments/{world['experiment_id']}"
            f"/tasks/{world['task_id']}/selected-runs"
        )
        summary = _request(server, base + "?attempt=1&attempt=2")[1]
        status, payload = _request(
            server,
            base + "/validation?attempt=1&attempt=2&expect="
            + summary["evidence_digest"],
        )
        assert status == 200 and payload["stale"] is False


# ----------------------------------------------------------------------
# The shipped page, in a real DOM
# ----------------------------------------------------------------------


def _dom(runs_server, runs_world, phase, *, on_ready=None):
    """One phase of the page harness, against the real server.

    `on_ready` is for the one phase that needs the evidence to change WHILE
    the page is on it: the harness says when it is ready, the caller edits
    the store, and the page carries on.
    """
    jsdom_root = os.environ.get("TEST_JSDOM_ROOT")
    if not jsdom_root:
        pytest.skip("Set TEST_JSDOM_ROOT to run DOM integration with jsdom")
    command = [
        "node", str(Path(__file__).with_name("chatbot_selected_runs_dom.cjs")),
        jsdom_root,
        f"http://127.0.0.1:{runs_server.port}/?token={runs_server.token}",
        runs_world["experiment_id"],
        runs_world["task_id"],
        phase,
    ]
    if on_ready is None:
        result = subprocess.run(command, capture_output=True, text=True,
                                timeout=180)
        assert result.returncode == 0, result.stdout + result.stderr
        return
    process = subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True,
    )
    try:
        for line in process.stdout:
            if "READY-FOR-MUTATION" in line:
                on_ready()
                process.stdin.write("go\n")
                process.stdin.flush()
                break
        stdout, stderr = process.communicate(timeout=180)
    finally:
        if process.poll() is None:
            process.kill()
    assert process.returncode == 0, stdout + stderr


def test_the_page_summarizes_the_runs_a_reader_ticks(runs_server, runs_world):
    """Driven through the real page against the real server: the ticks, the
    unfinished run that cannot be ticked, the pooled median, the run that
    recorded nothing, and a drill-down landing on the run that recorded the
    dispatch."""
    _dom(runs_server, runs_world, "basic")


def test_a_summary_answering_late_is_not_shown_under_another_selection(
    runs_server, runs_world
):
    """A real request, held back until after the reader ticked something
    else. The answer describes a population that is no longer on screen, so
    it is dropped rather than re-labelled with the new ticks."""
    _dom(runs_server, runs_world, "discard")


def test_only_the_last_drilldown_click_navigates_and_it_compares_whole_runs(
    runs_server, runs_world
):
    """Two real validation requests in flight, the EARLIER one answering
    first: the run the reader clicked last is the one that opens, and a whole
    run opens whole rather than under a pass scope left from an earlier
    comparison."""
    _dom(runs_server, runs_world, "race")


def test_a_drilldown_refuses_evidence_that_no_longer_supports_the_totals(
    runs_server, runs_world
):
    """The summary is computed, one dispatch's recorded outcome is then
    edited in place -- same turn, same call, same span, every identifier
    unchanged -- and the drill-down says so instead of opening it. The run
    nobody touched still opens."""
    _dom(
        runs_server, runs_world, "stale",
        on_ready=lambda: _edit_recorded_outcome(
            runs_world["store"], "sel-a2-t1", 0, False
        ),
    )


class TestSealedArchive:
    """A sealed archive carries EVIDENCE, and summarizing the evidence of runs
    a reader names is a read of it. It is answered, not refused with the
    decisions wording, which is about winners and best runs."""

    def test_an_archive_summarizes_the_runs_a_reader_names(
        self, tmp_path, monkeypatch
    ):
        srv, thread = _sealed_server(tmp_path, monkeypatch)
        try:
            status, payload = _request(
                srv,
                "/api/experiments/logical/tasks/task/selected-runs"
                "?attempt=1&attempt=2",
            )
            assert status == 200, payload
            assert payload["sealed"] is True
            assert [member["attempt"] for member in payload["members"]] == [1, 2]
            assert payload["command_summary"]["totals"]["dispatches"] == 3
            status, refused = _request(
                srv,
                "/api/experiments/logical/tasks/task/selected-runs"
                "?attempt=1&left_pass=teacher",
            )
            assert status == 400
            assert refused["refused"] == "unsupported_parameter"
        finally:
            srv.shutdown()
            thread.join(timeout=5)

    def test_members_keep_both_of_an_archives_names(self, tmp_path, monkeypatch):
        """The reference names the archive by evidence identity; every
        workspace turn and span route is addressed by the manifest's own
        name. A drill-down that used the wrong one would not open."""
        srv, thread = _sealed_server(tmp_path, monkeypatch)
        try:
            payload = _request(
                srv,
                "/api/experiments/logical/tasks/task/selected-runs?attempt=1",
            )[1]
            member = payload["members"][0]
            assert member["manifest_store_id"] == "sealed"
            assert member["store_id"] != "sealed"
            assert member["execution_ref"]["store_id"] == member["store_id"]
            contributors = [
                contributor
                for group in payload["command_summary"]["groups"]
                for contributor in group["contributors"]
            ]
            assert {c["store_id"] for c in contributors} == {member["store_id"]}
            # Validation answers under the same scope, so its digest is the
            # one the summary published.
            status, validation = _request(
                srv,
                "/api/experiments/logical/tasks/task/selected-runs/validation"
                "?attempt=1&expect=" + payload["evidence_digest"],
            )
            assert status == 200
            assert validation["stale"] is False
        finally:
            srv.shutdown()
            thread.join(timeout=5)

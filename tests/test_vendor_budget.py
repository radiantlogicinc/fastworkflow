"""One per-turn vendor budget with a hard wall-clock cutoff (fix-i94q, fix-sotm).

Real HTTP through the real typesafe_sdk to the loopback stand-in (``jev_stub``),
in its stall, trickle and delay modes. Budgets are scaled down through
constructor arguments so the file stays fast; the shipped numbers are asserted
separately.
"""
from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import dspy
import pytest

import fastworkflow
from fastworkflow import tracing
from fastworkflow.observation_offloading import finish_check, jev_client, search_router
from fastworkflow.observation_offloading.agent import build_tool_agent
from fastworkflow.observation_offloading.continuation import StructuredContinuationReAct
from fastworkflow.observation_offloading.finish_check import FinishChecker
from fastworkflow.observation_offloading.jev_client import Noul, TurnBudget
from fastworkflow.observation_offloading.search_router import ROUTER_BUDGET, SearchRouter
from fastworkflow.observation_offloading.state import reset_runtime_state, snapshot_events
from fastworkflow.turn_plan import PlanStep, PlanSubject, TurnPlan
from tests.jev_stub import choice_answer

#: What a bounded call may take past its limit: thread hand-off and HTTP set-up.
SLACK = 0.2
LISTING = "3 holder(s).\nuid  label\nu1  Alan\nu2  Bea\nu3  Cy\n"
QUESTIONS = {"q": Noul(instructions="Is this a test?", criteria={"true": "yes", "false": "no"})}


def _client(stub, timeout=5.0):
    """A real SDK client whose own (per-phase) timeout is far past every cutoff under test."""
    return jev_client.make_client("stub-key", "jev-test", timeout, stub.base_url)


def _plan():
    return TurnPlan(steps=[PlanStep(text="Find Alan Cooper", commands=["find_identity"]),
                           PlanStep(text="List Alan Cooper's accounts", commands=["list_accounts"])],
                    subjects=[PlanSubject(name="Alan Cooper", kind="person")])


def _read_only(_command):
    """Every command declared read-only, so every step of the plan is checked."""
    return "read_only"


def _agent(budget=None):
    return SimpleNamespace(current_trajectory={}, plan_source=_plan, vendor_budget=budget,
                           command_effect=_read_only)


def _router(stub, call_seconds=0.4):
    return SearchRouter(_client(stub), search_router._questions({}), model="jev-test",
                        call_seconds=call_seconds)


def _finish_event():
    return [e for e in snapshot_events() if e["kind"] == "finish_check"][-1]


@pytest.fixture(autouse=True)
def _fresh_state():
    reset_runtime_state()
    jev_client._WARNED.clear()
    yield
    jev_client._WARNED.clear()


def _timed(fn):
    started = time.monotonic()
    value = fn()
    return value, time.monotonic() - started


def test_the_shipped_numbers():
    assert (jev_client.VENDOR_WORKERS, jev_client.TURN_VENDOR_SECONDS, jev_client.ROUTER_CALLS_PER_TURN) == (
        4, 10.0, 3)
    assert (finish_check.CHECK_BUDGET_SECONDS, finish_check.CALL_TIMEOUT_SECONDS) == (8.0, 4.0)
    assert search_router.ROUTER_TIMEOUT_SECONDS == 2.0
    budget = TurnBudget()
    assert (budget.seconds, budget.router_calls_max, budget.vendor_ms) == (10.0, 3, 0)


# ---------------------------------------------------------------------------
# The hard cutoff: stall, trickle, delay
# ---------------------------------------------------------------------------

def test_a_stalled_call_is_cut_off_at_its_cap(jev_stub):
    jev_stub.stall = True
    budget = TurnBudget()
    started = time.monotonic()
    with pytest.raises(jev_client.CallTimedOut):
        jev_client.bounded_call(_client(jev_stub), cap=0.4, budget=budget, state="s", questions=QUESTIONS)
    assert time.monotonic() - started < 0.4 + SLACK
    assert 400 <= budget.vendor_ms <= (0.4 + SLACK) * 1000


def test_a_trickled_body_is_cut_off_although_every_slice_beats_the_read_timeout(jev_stub):
    """The SDK's timeout is per read: 12 slices 0.25 s apart would take 3 s and succeed."""
    jev_stub.trickle = (12, 0.25)
    started = time.monotonic()
    with pytest.raises(jev_client.CallTimedOut):
        jev_client.bounded_call(_client(jev_stub), cap=0.3, state="s", questions=QUESTIONS)
    assert time.monotonic() - started < 0.3 + SLACK
    assert len(jev_stub.requests) == 1, "an abandoned call is not retried"


def test_the_turn_budget_binds_before_the_cap_and_counts_time_in_calls(jev_stub):
    jev_stub.delay = 0.15
    budget = TurnBudget(seconds=0.3)
    client = _client(jev_stub)
    jev_client.bounded_call(client, cap=0.25, budget=budget, state="s", questions=QUESTIONS)
    time.sleep(0.1)  # time between calls is not vendor time
    assert 0.05 < budget.remaining < 0.15
    started = time.monotonic()
    with pytest.raises(jev_client.OutOfTime):
        jev_client.bounded_call(client, cap=0.25, budget=budget, state="s", questions=QUESTIONS)
    assert time.monotonic() - started < 0.15 + SLACK
    assert budget.remaining < jev_client.MIN_CALL_SECONDS and 290 <= budget.vendor_ms <= 300 + SLACK * 1000
    sent = len(jev_stub.requests)
    with pytest.raises(jev_client.OutOfTime):
        jev_client.bounded_call(client, cap=0.25, budget=budget, state="s", questions=QUESTIONS)
    assert len(jev_stub.requests) == sent, "a spent budget sends nothing"


@pytest.mark.parametrize("mode", ["stall", "trickle"])
def test_the_finish_check_fails_open_at_its_call_cap(jev_stub, mode):
    if mode == "stall":
        jev_stub.stall = True
    else:
        jev_stub.trickle = (12, 0.25)
    checker = FinishChecker(_client(jev_stub), model="jev-test", budget_seconds=0.45, call_seconds=0.3)
    budget = TurnBudget()
    note, elapsed = _timed(lambda: checker.note(_agent(budget), {"user_query": "Audit Alan Cooper"},
                                                iterations_left=10))
    assert note == "" and elapsed < 0.3 + SLACK
    event = _finish_event()
    assert (event["reason"], event["error_type"], event["error_stage"]) == ("error", "CallTimedOut", "request")
    assert event["vendor_ms"] == budget.vendor_ms and 290 <= event["vendor_ms"] <= (0.3 + SLACK) * 1000


def test_the_finish_check_fails_open_when_its_budget_runs_out_across_calls(jev_stub):
    jev_stub.delay = 0.2
    checker = FinishChecker(_client(jev_stub), model="jev-test", budget_seconds=0.35, call_seconds=0.25)
    note, elapsed = _timed(lambda: checker.note(_agent(TurnBudget()), {"user_query": "Audit Alan Cooper"},
                                                iterations_left=10))
    assert note == "" and elapsed < 0.35 + SLACK
    event = _finish_event()
    assert (event["reason"], event["error_type"], event["error_stage"]) == ("error", "OutOfTime", "budget")
    assert event["requests"] == 1


def test_the_check_clock_starts_before_the_ledger_is_built(jev_stub):
    """A ledger that took the whole budget to build leaves nothing to send."""
    checker = FinishChecker(_client(jev_stub), model="jev-test", budget_seconds=0.25, call_seconds=0.4)

    class SlowTrajectory(dict):
        def get(self, key, default=None):
            time.sleep(0.1)
            return super().get(key, default)

    trajectory = SlowTrajectory({"tool_name_0": "execute_workflow_query",
                                 "tool_args_0": {"command": "find_identity"}, "observation_0": "x"})
    agent = SimpleNamespace(current_trajectory=trajectory, plan_source=_plan, vendor_budget=TurnBudget(),
                            command_effect=_read_only)
    note, elapsed = _timed(lambda: checker.note(agent, {"user_query": "q"}, iterations_left=10))
    assert note == "" and elapsed < 0.3 + 0.3 + SLACK
    event = _finish_event()
    assert (event["error_type"], event["error_stage"], event["requests"]) == ("OutOfTime", "budget", 0)
    assert jev_stub.requests == []


@pytest.mark.parametrize("mode", ["stall", "trickle", "delay"])
def test_the_router_fails_open_within_its_bound(jev_stub, mode):
    budget = TurnBudget(seconds=0.3)
    if mode == "stall":
        jev_stub.stall = True
    elif mode == "trickle":
        jev_stub.trickle = (12, 0.25)
    else:
        jev_stub.delay = 0.5
    router = _router(jev_stub, call_seconds=0.25 if mode != "delay" else 0.4)
    route, elapsed = _timed(lambda: router.route("Who are the holders?", "", LISTING, budget=budget))
    assert route["choice"] is None and not router.wants_all_rows(route)
    if mode == "delay":
        # 0.3 s of turn budget left against a 0.4 s cap: the budget binds.
        assert elapsed < 0.3 + SLACK
        assert (route["error"], route["error_stage"]) == ("OutOfTime", "budget")
    else:
        assert elapsed < 0.25 + SLACK
        assert (route["error"], route["error_stage"]) == ("CallTimedOut", "request")
    assert route["vendor_ms"] == budget.vendor_ms


# ---------------------------------------------------------------------------
# The per-turn caps
# ---------------------------------------------------------------------------

def test_a_turn_routes_at_most_three_searches(jev_stub):
    jev_stub.answer = lambda name, q: choice_answer("all_rows", q["criteria"]) if q["type"] == "choice" else 0.8
    budget = TurnBudget()
    router = _router(jev_stub).within_budget(lambda: budget)
    sink = SimpleNamespace(spans=[], emit_span=lambda span: sink.spans.append(span))
    host = SimpleNamespace(trace_sink=sink, current_turn_key="turn-1")
    routes = [router.route("Who are the holders?", "", LISTING, host=host) for _ in range(5)]
    assert len(jev_stub.requests) == 3
    assert [r["choice"] for r in routes[:3]] == ["all_rows"] * 3
    assert [(r["error"], r["error_stage"]) for r in routes[3:]] == [(ROUTER_BUDGET, "budget")] * 2
    assert all(not router.wants_all_rows(r) for r in routes[3:])
    assert len([s for s in sink.spans if s.name == tracing.SPAN_SEARCH_ROUTE]) == 3
    assert routes[-1]["vendor_ms"] == budget.vendor_ms == routes[2]["vendor_ms"]
    assert budget.router_calls == 3

    # A new turn's budget routes again; the shared router is not changed by binding.
    budget = TurnBudget()
    assert router.route("Who?", "", LISTING)["choice"] == "all_rows"
    assert _router(jev_stub).route("Who?", "", LISTING)["vendor_ms"] is None


def test_a_slow_router_and_a_check_never_exceed_the_turn_total(jev_stub):
    jev_stub.delay = 0.15
    budget = TurnBudget(seconds=0.5)
    router = _router(jev_stub, call_seconds=0.25).within_budget(lambda: budget)
    checker = FinishChecker(_client(jev_stub), model="jev-test", budget_seconds=8.0, call_seconds=0.25)
    started = time.monotonic()
    for _ in range(3):
        assert router.route("Who?", "", LISTING).get("error") is None
    assert checker.note(_agent(budget), {"user_query": "Audit Alan Cooper"}, iterations_left=10) == ""
    elapsed = time.monotonic() - started
    assert elapsed < 0.5 + 3 * SLACK
    assert budget.remaining < jev_client.MIN_CALL_SECONDS and budget.vendor_ms <= 500 + SLACK * 1000
    event = _finish_event()
    assert (event["error_type"], event["error_stage"]) == ("OutOfTime", "budget")
    assert event["vendor_ms"] == budget.vendor_ms
    # Nothing is left for a fourth route either, even under the call cap.
    assert router.route("Who?", "", LISTING)["error"] == ROUTER_BUDGET


def test_a_saturated_pool_fails_at_once_with_vendor_busy(jev_stub):
    jev_stub.stall = True
    client = _client(jev_stub)
    outcomes = []

    def occupy():
        try:
            jev_client.call_within(client, 0.4, state="s", questions=QUESTIONS)
        except Exception as error:  # noqa: BLE001
            outcomes.append(type(error).__name__)

    threads = [threading.Thread(target=occupy) for _ in range(jev_client.VENDOR_WORKERS)]
    for thread in threads:
        thread.start()
    deadline = time.monotonic() + 2
    while jev_client.calls_in_flight() < jev_client.VENDOR_WORKERS and time.monotonic() < deadline:
        time.sleep(0.01)
    assert jev_client.calls_in_flight() == jev_client.VENDOR_WORKERS

    started = time.monotonic()
    with pytest.raises(jev_client.VendorBusy, match="vendor busy"):
        jev_client.bounded_call(client, cap=0.4, state="s", questions=QUESTIONS)
    assert time.monotonic() - started < 0.05
    route, elapsed = _timed(lambda: _router(jev_stub).route("Who?", "", LISTING, budget=TurnBudget()))
    assert (route["error"], route["error_stage"]) == ("VendorBusy", "request") and elapsed < 0.1
    checker = FinishChecker(client, model="jev-test", budget_seconds=0.6, call_seconds=0.4)
    note, elapsed = _timed(lambda: checker.note(_agent(TurnBudget()), {"user_query": "q"}, iterations_left=10))
    assert note == "" and elapsed < 0.1
    event = _finish_event()
    assert (event["error_type"], event["error_stage"]) == ("VendorBusy", "request")
    assert len(jev_stub.requests) == jev_client.VENDOR_WORKERS, "a busy call sends nothing"

    for thread in threads:
        thread.join(timeout=2)
    assert outcomes == ["CallTimedOut"] * jev_client.VENDOR_WORKERS


def test_abandoned_trickled_calls_give_their_slots_back_within_cap_plus_margin(jev_stub):
    """A trickled body never trips the SDK's per-read timeout; the slot is freed anyway."""
    jev_stub.trickle = (40, 0.1)
    client = _client(jev_stub)
    cap = 0.3
    outcomes = []

    def abandon():
        try:
            jev_client.bounded_call(client, cap=cap, state="s", questions=QUESTIONS)
        except Exception as error:  # noqa: BLE001
            outcomes.append(type(error).__name__)

    threads = [threading.Thread(target=abandon) for _ in range(jev_client.VENDOR_WORKERS)]
    started = time.monotonic()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)
    assert outcomes == ["CallTimedOut"] * jev_client.VENDOR_WORKERS
    assert jev_client.calls_in_flight() == jev_client.VENDOR_WORKERS

    # While the slots are held, the check's event and the router's record say so.
    checker = FinishChecker(client, model="jev-test", budget_seconds=0.6, call_seconds=0.4)
    assert checker.note(_agent(TurnBudget()), {"user_query": "q"}, iterations_left=10) == ""
    event = _finish_event()
    assert (event["error_type"], event["vendor_calls_in_flight"]) == ("VendorBusy", jev_client.VENDOR_WORKERS)
    route = _router(jev_stub).route("Who?", "", LISTING, budget=TurnBudget())
    assert (route["error"], route["vendor_calls_in_flight"]) == ("VendorBusy", jev_client.VENDOR_WORKERS)

    deadline = started + cap + jev_client.CUTOFF_MARGIN_SECONDS + SLACK
    while jev_client.calls_in_flight() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert jev_client.calls_in_flight() == 0
    assert jev_client.calls_orphaned() == jev_client.VENDOR_WORKERS, "the requests themselves still run"
    jev_stub.trickle = None
    jev_client.bounded_call(client, cap=1.0, state="s", questions=QUESTIONS)
    route = _router(jev_stub).route("Who?", "", LISTING, budget=TurnBudget())
    assert "vendor_calls_in_flight" not in route


def _wait_until(condition, seconds):
    deadline = time.monotonic() + seconds
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.01)
    return condition()


def test_enough_orphaned_calls_refuse_new_ones_until_the_trickles_end(jev_stub):
    """Freed slots do not free threads: past ``ORPHANED_CALLS_MAX`` live orphans nothing more is sent."""
    assert jev_client.ORPHANED_CALLS_MAX == 8
    jev_stub.trickle = (200, 0.1)
    client = _client(jev_stub)
    cap = 0.1

    def abandon():
        with pytest.raises(jev_client.CallTimedOut):
            jev_client.bounded_call(client, cap=cap, state="s", questions=QUESTIONS)

    for _batch in range(jev_client.ORPHANED_CALLS_MAX // jev_client.VENDOR_WORKERS):
        threads = [threading.Thread(target=abandon) for _ in range(jev_client.VENDOR_WORKERS)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=2)
        assert _wait_until(lambda: jev_client.calls_in_flight() == 0,
                           cap + jev_client.CUTOFF_MARGIN_SECONDS + SLACK)
    assert jev_client.calls_orphaned() == jev_client.ORPHANED_CALLS_MAX
    sent = len(jev_stub.requests)
    assert sent == jev_client.ORPHANED_CALLS_MAX

    # Every slot is free, yet a call is refused at once and nothing reaches the stub.
    started = time.monotonic()
    with pytest.raises(jev_client.VendorBusy):
        jev_client.bounded_call(client, cap=1.0, state="s", questions=QUESTIONS)
    assert time.monotonic() - started < 0.05
    checker = FinishChecker(client, model="jev-test", budget_seconds=0.6, call_seconds=0.4)
    assert checker.note(_agent(TurnBudget()), {"user_query": "q"}, iterations_left=10) == ""
    event = _finish_event()
    assert (event["error_type"], event["error_stage"]) == ("VendorBusy", "request")
    assert event["vendor_calls_orphaned"] == jev_client.ORPHANED_CALLS_MAX
    assert "vendor_calls_in_flight" not in event
    route = _router(jev_stub).route("Who?", "", LISTING, budget=TurnBudget())
    assert (route["error"], route["vendor_calls_orphaned"]) == ("VendorBusy", jev_client.ORPHANED_CALLS_MAX)
    assert len(jev_stub.requests) == sent, "a refused call sends nothing"

    # The trickles end, the orphans' threads with them, and calls go through again.
    jev_stub.end_trickles()
    assert _wait_until(lambda: jev_client.calls_orphaned() == 0, 2.0)
    jev_client.bounded_call(client, cap=1.0, state="s", questions=QUESTIONS)
    assert len(jev_stub.requests) == sent + 1
    route = _router(jev_stub).route("Who?", "", LISTING, budget=TurnBudget())
    assert route.get("error") is None and "vendor_calls_orphaned" not in route


_ABANDON_AND_EXIT = """
import sys, time
from fastworkflow.observation_offloading import jev_client
from fastworkflow.observation_offloading.jev_client import Noul
client = jev_client.make_client("stub-key", "jev-test", 30.0, sys.argv[1])
questions = {"q": Noul(instructions="?", criteria={"true": "yes", "false": "no"})}
try:
    jev_client.bounded_call(client, cap=0.3, state="s", questions=questions)
except jev_client.CallTimedOut:
    print("abandoned", jev_client.calls_in_flight(), flush=True)
"""


def test_an_abandoned_call_does_not_delay_interpreter_exit(jev_stub):
    """The stalled request's SDK timeout is 30 s; the worker must not be joined at exit."""
    jev_stub.stall = True
    process = subprocess.Popen([sys.executable, "-c", _ABANDON_AND_EXIT, jev_stub.base_url],
                               cwd=str(Path(__file__).resolve().parent.parent),
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        line = process.stdout.readline()
        abandoned = time.monotonic()
        process.wait(timeout=10)
        exited = time.monotonic()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    assert line.split() == ["abandoned", "1"], process.stderr.read()
    assert process.returncode == 0
    assert exited - abandoned < 2.0


# ---------------------------------------------------------------------------
# Where the budget lives: one per turn, kept across an ask_user resume
# ---------------------------------------------------------------------------

class _BudgetRecorder(StructuredContinuationReAct):
    """Records the budget each segment run sees; everything around it is real."""

    def _run_segments(self, trajectory, idx, input_args, max_iters):
        self.seen.append(self.vendor_budget)
        return SimpleNamespace(trajectory=trajectory)


def _recording_agent(factory):
    agent = _BudgetRecorder.__new__(_BudgetRecorder)
    agent.seen = []
    agent.max_iters = 5
    agent._suspended = None
    agent.vendor_budget_factory = factory
    return agent


def test_every_turn_gets_a_fresh_budget_and_a_resume_keeps_it():
    agent = _recording_agent(TurnBudget)
    agent.forward(user_query="first")
    agent.forward(user_query="second")
    first, second = agent.seen
    assert isinstance(first, TurnBudget) and isinstance(second, TurnBudget) and first is not second

    agent._suspended = {"trajectory": {}, "idx": 0, "input_args": {"user_query": "second"}, "max_iters": 5}
    agent.resume("the user's reply")
    assert agent.seen[-1] is second


def test_no_budget_when_routing_and_the_check_are_both_off():
    agent = _recording_agent(None)
    agent.forward(user_query="q")
    assert agent.seen == [None]


class _Signature(dspy.Signature):
    user_query: str = dspy.InputField()
    answer: str = dspy.OutputField()


def _noop_tool(command: str) -> str:
    """Return the command unchanged."""
    return command


def _build(tmp_path, monkeypatch, **flags):
    monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path))
    for name in (finish_check.CHECK_ENV, search_router.ROUTER_ENV, jev_client.KEY_ENV):
        monkeypatch.delitem(fastworkflow._env_vars, name, raising=False)
        monkeypatch.delenv(name, raising=False)
    for name, value in flags.items():
        monkeypatch.setenv(name, value)
    finish_check._CHECKERS.clear()
    search_router._ROUTERS.clear()
    return build_tool_agent(SimpleNamespace(), _Signature, [_noop_tool], max_iters=3)


def test_the_agent_holds_one_budget_for_routing_and_the_check(jev_stub, tmp_path, monkeypatch):
    agent = _build(tmp_path, monkeypatch, **{finish_check.CHECK_ENV: "jev", search_router.ROUTER_ENV: "jev",
                                             jev_client.KEY_ENV: "stub-key"})
    assert agent.finish_checker is not None and isinstance(agent.vendor_budget, TurnBudget)
    # A session with no workflow has no manifest: every command is unknown to the check.
    assert callable(agent.command_effect) and agent.command_effect("find_identity") == "unknown"
    agent.vendor_budget = TurnBudget(router_calls=1)
    router = agent.search_router
    assert router.route("Who?", "", LISTING).get("error") is None
    assert router.route("Who?", "", LISTING)["error"] == ROUTER_BUDGET
    assert search_router._ROUTERS.get(("", jev_client.DEFAULT_MODEL, jev_stub.base_url), "stub-key",
                                      lambda: None)._budget_source is None, "the shared router is not bound"
    finish_check._CHECKERS.clear()
    search_router._ROUTERS.clear()


def test_the_default_agent_has_no_budget(tmp_path, monkeypatch):
    agent = _build(tmp_path, monkeypatch)
    assert (agent.finish_checker, agent.search_router, agent.vendor_budget, agent.vendor_budget_factory,
            agent.command_effect) == (None, None, None, None, None)

"""The decision-provider seam: what the Jev adapter puts on the wire, and registered providers.

The golden tests run the finish check and the search router over real HTTP,
through the real typesafe_sdk, against the loopback stand-in (``jev_stub``),
and compare every request body the stand-in received with the bodies recorded
in ``tests/fixtures/decision_provider_golden_bodies.json`` -- key order
included. The measured precision and recall hold only for exactly those
requests, so a difference fails here rather than silently changing what was
measured. The fixture was captured before the questions moved behind the
provider interface; regenerate it only for a deliberate change of wording.

The registered-provider tests use a real in-process class implementing
``decision.DecisionProvider`` with deterministic answers; the stand-in runs
alongside so that "no HTTP" is observed, not assumed.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

import fastworkflow
from fastworkflow.observability import store as observability_store
from fastworkflow.observation_offloading import decision, finish_check, jev_client, search_router
from fastworkflow.observation_offloading.decision import (
    Answers,
    OneOf,
    ProviderFailure,
    StateTooLarge,
    YesNo,
    register_decision_provider,
    unregister_decision_provider,
)
from fastworkflow.observation_offloading.finish_check import FinishChecker, checker_from_env
from fastworkflow.observation_offloading.search_router import SearchRouter, router_for_workflow
from fastworkflow.observation_offloading.state import reset_runtime_state, snapshot_events
from fastworkflow.turn_plan import PlanPart, PlanStep, PlanSubject, TurnPlan
from fastworkflow.utils.logging import logger
from tests.jev_stub import choice_answer

GOLDEN_PATH = Path(__file__).parent / "fixtures" / "decision_provider_golden_bodies.json"
GOLDEN_MODEL = "jev-golden"
SECRET = "sk-abcdefghijklmnopqrstuvwxyz123456"
TOO_LONG = {"error": {"type": "max_tokens_exceeded", "message": "The request is longer than the model accepts."}}

UID = "28c5aeb5b64e4ac6c40c57b0235980e2"
GOLDEN_PLAN = TurnPlan(steps=[
    PlanStep(text="Find Alan Cooper and Grace Hopper", commands=["find_identity"]),
    PlanStep(text="For each person: open the identity and list the accounts",
             parts=[PlanPart(text="open_identity_by_uid", commands=["open_identity_by_uid"]),
                    PlanPart(text="list_accounts", commands=["list_accounts"])]),
    PlanStep(text="Show the portraits", commands=["open_portrait"], optional=True),
    PlanStep(text="Remove the accounts the user picks", commands=["remove_account"], needs_user=True),
    PlanStep(text="Summarise what was found", commands=[]),
], subjects=[PlanSubject(name="Alan Cooper", kind="person"), PlanSubject(name="Grace Hopper", kind="person")])
GOLDEN_REQUEST = f"Audit Alan Cooper and Grace Hopper (token Bearer {SECRET})"
GOLDEN_LEDGER = [
    {"n": 1, "command": "find_identity <name>Alan Cooper</name>", "context": "DirectoryExplorer",
     "acted_on": "DirectoryExplorer", "refers_to": {}, "names_in_output": ["Alan Cooper"],
     "outcome": "result", "head": f"1 identity(s).\n{UID}  Alan Cooper"},
    {"n": 2, "command": f"open_identity_by_uid <uid>{UID}</uid>", "context": "DirectoryExplorer",
     "acted_on": f"Identity {UID} Alan Cooper",
     "refers_to": {UID: "Alan Cooper (listed by find_identity <name>Alan Cooper</name>)"},
     "names_in_output": [], "outcome": "result", "head": "Entered Identity context."},
    {"n": 3, "command": "list_accounts", "context": f"Identity {UID} Alan Cooper", "acted_on": "",
     "refers_to": {}, "names_in_output": [], "outcome": "empty", "head": "No accounts found."},
    {"n": 4, "command": "find_identity <name>Grace Hopper</name>", "context": f"Identity {UID} Alan Cooper",
     "acted_on": "", "refers_to": {}, "names_in_output": ["Grace Hopper"], "outcome": "error",
     "head": "Execution error: identity lookup failed"},
]
ROUTER_EXAMPLES = {"all_rows": ["show every holder"], "rows_about_named_item": ["is Alan among them"]}
ROUTER_LISTING = "3 holder(s).\nuid  label\nu1  Alan\nu2  Bea\nu3  Cy\nu4  Dee\n"


def _read_only(_command):
    """Every command declared read-only, so the golden plan's steps are all checked as recorded."""
    return "read_only"


def _golden_answer(name, question):
    if question.get("type") == "choice":
        return choice_answer("all_rows", question["criteria"])
    # Part 2 of step 2 does not concern Grace Hopper: her question is about part 1 only.
    return 0.1 if name == "p2_2_1" else 0.9


def _refuse_long_ledgers(body):
    state = body["state"]
    if isinstance(state, dict) and len(state.get("ledger", [])) > 2:
        return 400, TOO_LONG
    return None


def _finish_check_bodies(stub, checker):
    stub.respond = _refuse_long_ledgers
    stub.answer = _golden_answer
    result = checker.check(GOLDEN_PLAN, GOLDEN_REQUEST, GOLDEN_LEDGER, command_effect=_read_only)
    assert (result.error, result.splits) == (None, 1)
    bodies = stub.bodies
    del stub.requests[:]
    return bodies


def _router_bodies(stub, router):
    stub.respond = None
    stub.answer = _golden_answer
    route = router.route("Who are the holders?", f"I saw Bearer {SECRET}", ROUTER_LISTING)
    assert route["choice"] == "all_rows" and route.get("error") is None
    bodies = stub.bodies
    del stub.requests[:]
    return bodies


def _client(stub):
    return jev_client.make_client("golden-key", GOLDEN_MODEL, 2.0, stub.base_url)


def _golden():
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def _same_bytes(actual, expected):
    """Equal as serialised, key order included: what was measured is exactly these requests."""
    return json.dumps(actual, ensure_ascii=False) == json.dumps(expected, ensure_ascii=False)


FLAGS = (finish_check.CHECK_ENV, search_router.ROUTER_ENV, jev_client.KEY_ENV,
         finish_check.MODEL_ENV, search_router.ROUTER_MODEL_ENV, observability_store.CAPTURE_PROFILE_VAR)
IN_PROCESS = "in-house"


@pytest.fixture(autouse=True)
def _fresh_state(monkeypatch):
    for name in FLAGS:
        monkeypatch.delitem(fastworkflow._env_vars, name, raising=False)
        monkeypatch.delenv(name, raising=False)
    reset_runtime_state()
    jev_client._WARNED.clear()
    finish_check._CHECKERS.clear()
    search_router._ROUTERS.clear()
    yield
    unregister_decision_provider(IN_PROCESS)
    jev_client._WARNED.clear()
    finish_check._CHECKERS.clear()
    search_router._ROUTERS.clear()


def test_the_finish_check_sends_the_golden_bodies(jev_stub):
    bodies = _finish_check_bodies(jev_stub, FinishChecker(_client(jev_stub), model=GOLDEN_MODEL))
    # Scope questions, then the whole ledger (refused), then its two halves.
    assert [len(b["state"].get("ledger", [])) for b in bodies] == [0, 4, 2, 2]
    assert SECRET not in json.dumps(bodies)
    assert _same_bytes(bodies, _golden()["finish_check"])


def test_the_router_sends_the_golden_body(jev_stub):
    router = SearchRouter(_client(jev_stub), search_router._questions(ROUTER_EXAMPLES), model=GOLDEN_MODEL)
    bodies = _router_bodies(jev_stub, router)
    assert len(bodies) == 1 and SECRET not in json.dumps(bodies)
    assert _same_bytes(bodies, _golden()["router"])


def test_an_explicit_jev_provider_sends_the_same_golden_bodies(jev_stub):
    provider = jev_client.JevProvider(_client(jev_stub))
    assert (provider.name, provider.third_party) == ("jev", True)
    assert isinstance(provider, decision.DecisionProvider)
    checker = FinishChecker(provider, model=GOLDEN_MODEL)
    assert _same_bytes(_finish_check_bodies(jev_stub, checker), _golden()["finish_check"])
    router = SearchRouter(provider, search_router._questions(ROUTER_EXAMPLES), model=GOLDEN_MODEL)
    assert _same_bytes(_router_bodies(jev_stub, router), _golden()["router"])


# ---------------------------------------------------------------------------
# The Jev adapter's error mapping
# ---------------------------------------------------------------------------

def test_the_adapter_maps_a_refused_state_and_other_failures_to_neutral_errors(jev_stub):
    provider = jev_client.JevProvider(_client(jev_stub))
    questions = {"q": YesNo(instructions="Is this a test?", true="yes", false="no")}
    jev_stub.respond = lambda _body: (400, TOO_LONG)
    with pytest.raises(StateTooLarge) as refused:
        provider.ask({"k": "v"}, questions, cap=2.0)
    assert refused.value.failure == {"error_type": "TypeSafeBadRequestError", "status": 400,
                                     "request_id": "req-1", "code": "max_tokens_exceeded"}
    jev_stub.respond = lambda _body: (429, {"error": {"type": "rate_limit_exceeded", "message": "Slow down."}})
    with pytest.raises(ProviderFailure) as failed:
        provider.ask({"k": "v"}, questions, cap=2.0)
    assert not isinstance(failed.value, StateTooLarge)
    assert failed.value.failure == {"error_type": "TypeSafeRateLimitError", "status": 429,
                                    "request_id": "req-2", "code": "rate_limit_exceeded"}
    assert decision.describe(failed.value) == failed.value.failure
    jev_stub.respond = None
    jev_stub.stall = True
    with pytest.raises(ProviderFailure) as cut_off:
        provider.ask({"k": "v"}, questions, cap=0.2)
    assert cut_off.value.failure["error_type"] == "CallTimedOut"
    with pytest.raises(decision.OutOfTime):
        provider.ask({"k": "v"}, questions, cap=2.0, budget=jev_client.TurnBudget(seconds=0.0))
    assert jev_client.OutOfTime is decision.OutOfTime


def test_the_adapter_reads_both_answer_kinds_and_the_usage(jev_stub):
    jev_stub.answer = lambda name, q: choice_answer("b", q["criteria"], 0.7) if q["type"] == "choice" else 0.25
    answers = jev_client.JevProvider(_client(jev_stub)).ask(
        "s", {"n": YesNo(instructions="?", true="yes", false="no"),
              "c": OneOf(instructions="?", choices={"a": "first", "b": "second"})}, cap=2.0)
    assert answers == {"n": 0.25, "c": ("b", {"a": pytest.approx(0.3), "b": 0.7})}
    assert answers.usage == {"input_tokens": 7, "output_tokens": 1}


# ---------------------------------------------------------------------------
# Registered providers
# ---------------------------------------------------------------------------

class InProcessProvider:
    """A deterministic provider in this process: every ``OneOf`` picks its first choice."""

    third_party = False
    model = "rules-v1"

    def __init__(self, answer=lambda key: 0.9):
        self.answer = answer
        self.calls = []

    def ask(self, state, questions, *, cap, deadline=None, budget=None):
        self.calls.append({"state": state, "questions": dict(questions), "cap": cap,
                           "deadline": deadline, "budget": budget})
        answers = {}
        for key, question in questions.items():
            if isinstance(question, OneOf):
                first = next(iter(question.choices))
                answers[key] = (first, {label: 0.9 if label == first else 0.02 for label in question.choices})
            else:
                answers[key] = self.answer(key)
        return Answers(answers, usage={"input_tokens": 5, "output_tokens": 1})


class _Warnings(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@pytest.fixture
def warnings_logged():
    handler = _Warnings()
    logger.addHandler(handler)
    yield handler.messages
    logger.removeHandler(handler)


def _two_step_plan():
    return TurnPlan(steps=[PlanStep(text="Find Alan Cooper", commands=["find_identity"]),
                           PlanStep(text="List Alan Cooper's accounts", commands=["list_accounts"])],
                    subjects=[PlanSubject(name="Alan Cooper", kind="person")])


def _step_two_unexecuted(key):
    return 0.1 if key.startswith(("e2_", "g2")) else 0.9


def _finish_events():
    return [e for e in snapshot_events() if e["kind"] == "finish_check"]


def test_a_registered_provider_runs_the_finish_check_end_to_end_without_http(jev_stub, monkeypatch,
                                                                             warnings_logged):
    provider = InProcessProvider(_step_two_unexecuted)
    built = []
    register_decision_provider("In-House", lambda: built.append(1) or provider)
    monkeypatch.setenv(finish_check.CHECK_ENV, "IN-HOUSE")
    checker = checker_from_env()
    assert checker is not None and checker_from_env() is checker and built == [1]
    trajectory = {"tool_name_0": "execute_workflow_query",
                  "tool_args_0": {"command": "find_identity <name>Alan Cooper</name>"},
                  "observation_0": "1 identity(s).",
                  "tool_name_1": "finish", "tool_args_1": {}, "observation_1": "Completed."}
    agent = SimpleNamespace(current_trajectory=trajectory, plan_source=_two_step_plan,
                            vendor_budget=jev_client.TurnBudget(), command_effect=_read_only)
    note = checker.note(agent, {"user_query": "Audit Alan Cooper"}, iterations_left=10)

    assert "step 2" in note and "Alan Cooper" in note
    event = _finish_events()[-1]
    assert (event["reason"], event["fired"], event["provider"], event["model"]) == (
        "unexecuted steps", True, IN_PROCESS, "rules-v1")
    assert (event["requests"], event["input_tokens"], event["calibration"]) == (2, 10, finish_check.CALIBRATION)
    assert jev_stub.requests == [], "no HTTP"
    # Neutral questions, the ledger as the Jev path sends it, and the turn's budget.
    scope_call, exec_call = provider.calls
    assert all(isinstance(q, YesNo) for call in provider.calls for q in call["questions"].values())
    assert set(scope_call["state"]) == {"request", "plan_steps"} and len(exec_call["state"]["ledger"]) == 1
    assert scope_call["cap"] == finish_check.CALL_TIMEOUT_SECONDS
    assert scope_call["budget"] is agent.vendor_budget and scope_call["deadline"] is not None
    # Selecting it says, once, that the check was calibrated with Jev.
    calibration = [m for m in warnings_logged if "calibrated with Jev" in m]
    assert len(calibration) == 1 and finish_check.CHECK_ENV in calibration[0]


def test_every_jev_event_names_the_jev_provider(jev_stub, monkeypatch):
    monkeypatch.setenv(finish_check.CHECK_ENV, "jev")
    monkeypatch.setenv(jev_client.KEY_ENV, "stub-key")
    checker = checker_from_env()
    checker.note(SimpleNamespace(current_trajectory={}, plan_source=_two_step_plan, command_effect=_read_only),
                 {"user_query": "q"}, iterations_left=10)
    assert _finish_events()[-1]["provider"] == "jev"
    checker.record_skip(SimpleNamespace(current_trajectory={}), reason="cap reached", iterations_left=10)
    assert _finish_events()[-1]["provider"] == "jev"


def test_a_registered_provider_routes_searches(jev_stub, monkeypatch):
    provider = InProcessProvider()
    register_decision_provider(IN_PROCESS, lambda: provider)
    monkeypatch.setenv(search_router.ROUTER_ENV, IN_PROCESS)
    router = router_for_workflow("wf")
    assert router is not None and router_for_workflow("wf") is router
    route = router.route("Who are the holders?", "", ROUTER_LISTING, budget=jev_client.TurnBudget())
    assert (route["choice"], route["p_all_rows"], route["for_report"], route["provider"]) == (
        "all_rows", 0.9, 0.9, IN_PROCESS)
    assert route["usage"] == {"input_tokens": 5, "output_tokens": 1} and router.wants_all_rows(route)
    wants = provider.calls[0]["questions"]["wants"]
    assert isinstance(wants, OneOf) and list(wants.choices)[0] == search_router.ALL_ROWS
    assert jev_stub.requests == []


def test_one_provider_serves_both_features(monkeypatch):
    built = []
    register_decision_provider(IN_PROCESS, lambda: built.append(InProcessProvider()) or built[-1])
    monkeypatch.setenv(finish_check.CHECK_ENV, IN_PROCESS)
    monkeypatch.setenv(search_router.ROUTER_ENV, IN_PROCESS)
    assert checker_from_env() is not None and router_for_workflow("wf") is not None
    assert len(built) == 1


def test_a_factory_may_use_the_registry_itself():
    """The factory runs outside the registry lock: one that reads or writes it does not deadlock."""
    seen = []

    def factory():
        seen.append(decision.registration(IN_PROCESS))
        register_decision_provider("in-house-helper", InProcessProvider)
        return InProcessProvider()

    register_decision_provider(IN_PROCESS, factory)
    built = []
    worker = threading.Thread(target=lambda: built.append(decision.build_provider(
        decision.registration(IN_PROCESS))), daemon=True)
    try:
        worker.start()
        worker.join(timeout=5)
        assert not worker.is_alive(), "build_provider deadlocked on the registry lock"
        assert isinstance(built[0], InProcessProvider) and seen[0].name == IN_PROCESS
        assert decision.registration("in-house-helper") is not None
        # Built once per registration: the second ask reuses it.
        assert decision.build_provider(decision.registration(IN_PROCESS)) is built[0]
    finally:
        unregister_decision_provider("in-house-helper")


def test_re_registering_a_name_builds_the_new_provider(monkeypatch):
    first, second = InProcessProvider(), InProcessProvider()
    register_decision_provider(IN_PROCESS, lambda: first)
    monkeypatch.setenv(finish_check.CHECK_ENV, IN_PROCESS)
    before = checker_from_env()
    register_decision_provider(IN_PROCESS, lambda: second)
    after = checker_from_env()
    assert after is not before and after._provider is second and before._provider is first


def test_an_unknown_or_unregistered_name_stays_off_with_one_warning(monkeypatch, warnings_logged):
    monkeypatch.setenv(finish_check.CHECK_ENV, IN_PROCESS)
    assert checker_from_env() is None and checker_from_env() is None
    assert len(warnings_logged) == 1 and "is not recognised" in warnings_logged[0]
    assert "registered decision provider" in warnings_logged[0]
    register_decision_provider(IN_PROCESS, InProcessProvider)
    assert checker_from_env() is not None
    unregister_decision_provider(IN_PROCESS)
    finish_check._CHECKERS.clear()
    assert checker_from_env() is None


def test_the_default_path_builds_no_provider(monkeypatch):
    built = []
    register_decision_provider(IN_PROCESS, lambda: built.append(1) or InProcessProvider())
    assert checker_from_env() is None and router_for_workflow("wf") is None
    monkeypatch.setenv(finish_check.CHECK_ENV, "off")
    assert checker_from_env() is None
    assert built == []


@pytest.mark.parametrize("name", ["jev", "JEV", "off", "none", "", "  ", "has space", "a/b", "mod:attr"])
def test_a_reserved_or_malformed_name_cannot_be_registered(name):
    with pytest.raises(ValueError):
        register_decision_provider(name, InProcessProvider)


def test_a_factory_must_be_callable():
    with pytest.raises(TypeError):
        register_decision_provider(IN_PROCESS, InProcessProvider())


def test_a_broken_factory_leaves_the_feature_off_with_one_warning(monkeypatch, warnings_logged):
    def broken():
        raise RuntimeError("model file missing")

    register_decision_provider(IN_PROCESS, broken)
    monkeypatch.setenv(finish_check.CHECK_ENV, IN_PROCESS)
    assert checker_from_env() is None and checker_from_env() is None
    assert len(warnings_logged) == 1 and "RuntimeError" in warnings_logged[0]
    assert "model file missing" not in warnings_logged[0]
    register_decision_provider(IN_PROCESS, lambda: object())
    assert checker_from_env() is None
    assert "TypeError" in warnings_logged[-1]


def test_the_capture_policy_gate_applies_to_a_registered_provider(monkeypatch, warnings_logged):
    built = []
    register_decision_provider(IN_PROCESS, lambda: built.append(1) or InProcessProvider())
    monkeypatch.setenv(finish_check.CHECK_ENV, IN_PROCESS)
    monkeypatch.setenv(observability_store.CAPTURE_PROFILE_VAR, "evidence")
    assert checker_from_env() is None and built == []
    assert len(warnings_logged) == 1 and "withholds command output" in warnings_logged[0]


def test_a_registered_provider_refusing_the_state_gets_the_halves():
    class Refusing(InProcessProvider):
        def ask(self, state, questions, **kwargs):
            if isinstance(state, dict) and len(state.get("ledger", [])) > 2:
                self.calls.append({"state": state})
                raise StateTooLarge({"error_type": "ContextTooLong"})
            return super().ask(state, questions, **kwargs)

    provider = Refusing()
    result = FinishChecker(provider, provider_name=IN_PROCESS).check(GOLDEN_PLAN, "Audit", GOLDEN_LEDGER,
                                                                      command_effect=_read_only)
    assert (result.error, result.splits) == (None, 1)
    assert [len(c["state"].get("ledger", [])) for c in provider.calls] == [0, 4, 2, 2]


def test_a_registered_provider_failure_lands_on_the_event():
    class Failing(InProcessProvider):
        def ask(self, state, questions, **kwargs):
            raise ProviderFailure({"error_type": "ModelUnavailable", "status": 503, "request_id": "r-9",
                                   "code": "overloaded"})

    checker = FinishChecker(Failing(), provider_name=IN_PROCESS)
    agent = SimpleNamespace(current_trajectory={}, plan_source=_two_step_plan, command_effect=_read_only)
    assert checker.note(agent, {"user_query": "q"}, iterations_left=10) == ""
    event = _finish_events()[-1]
    assert (event["reason"], event["error_type"], event["error_status"], event["error_request_id"],
            event["error_code"], event["error_stage"], event["provider"]) == (
        "error", "ModelUnavailable", 503, "r-9", "overloaded", "request", IN_PROCESS)

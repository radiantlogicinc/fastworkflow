"""Jev's pick among the candidates of an undecided fuzzy match (DatabaseValidator.fuzzy_match_with_request).

The Predict and Choice stand-ins are hand-written (no unittest.mock); the real-Jev tests skip without
JEV_API_KEY. Only the Predict call the helper makes is replaced, so the option texts are checked as sent.
"""
from __future__ import annotations

import os
from difflib import SequenceMatcher
from types import SimpleNamespace

import dspy
import pytest
from dotenv import dotenv_values
from dspy.utils import DummyLM

import fastworkflow
from fastworkflow import workflow_agent
from fastworkflow.utils import signatures
from fastworkflow.utils.signatures import DatabaseValidator, VALUE_PICK_THRESHOLD

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

KEYS = ["billing-api", "billing-api-staging", "billing-worker"]
DESCRIPTIONS = {
    "billing-api": "production",
    "billing-api-staging": "staging environment",
    "billing-worker": "background jobs",
}
REQUEST = "restart the billing api on staging"
UNDECIDED = (False, None, ["billing-api", "billing-api-staging"])


class _Recorder:
    """A stand-in for dspy.Predict: answers one fixed pick and records every call."""

    def __init__(self, pick: str = "billing-api-staging", probability: float = 0.95,
                 error: Exception | None = None):
        self.pick = pick
        self.probability = probability
        self.error = error
        self.calls: list[dict] = []

    def __call__(self, signature):
        def predict(request, argument):
            self.calls.append({"request": request, "argument": argument, "signature": signature})
            if self.error is not None:
                raise self.error
            choice = SimpleNamespace(
                value=self.pick,
                probabilities={self.pick: self.probability},
                confidence=self.probability,
            )
            return SimpleNamespace(value=choice)
        return predict


_REAL_CHOICE = signatures.Choice   # captured before any test replaces signatures.Choice with the spy


class _ChoiceSpy:
    """Wraps dspy.experimental.Choice so the option tuple the helper declares can be inspected."""

    options: tuple | None = None

    def __class_getitem__(cls, options):
        _ChoiceSpy.options = options
        return _REAL_CHOICE[options]


def _use(monkeypatch, recorder: _Recorder | None = None, lm=None):
    monkeypatch.setattr(workflow_agent, "_jev_lm", lambda: DummyLM([]) if lm is None else lm)
    monkeypatch.setattr(dspy, "Predict", recorder if recorder is not None else _Recorder())
    monkeypatch.setattr(signatures, "Choice", _ChoiceSpy)
    _ChoiceSpy.options = None


@pytest.fixture(autouse=True)
def _switch_on(monkeypatch):
    monkeypatch.setitem(fastworkflow._env_vars, "FW_JEV_VALUE_PICK", "on")


def _pick(key_values=KEYS, request=REQUEST, descriptions=DESCRIPTIONS):
    return DatabaseValidator.fuzzy_match_with_request("billing api", key_values, request, descriptions)


def test_a_unique_fuzzy_match_returns_without_a_jev_call(monkeypatch):
    recorder = _Recorder()
    _use(monkeypatch, recorder)
    got = DatabaseValidator.fuzzy_match_with_request("BILLING-WORKER", KEYS, REQUEST, DESCRIPTIONS)
    assert got == (True, "billing-worker", [])
    assert recorder.calls == []


def test_no_descriptions_makes_no_call_and_returns_the_fuzzy_result(monkeypatch):
    recorder = _Recorder()
    _use(monkeypatch, recorder)
    assert _pick(descriptions=None) == UNDECIDED
    assert _pick(descriptions={}) == UNDECIDED
    assert recorder.calls == []


def test_an_empty_request_makes_no_call(monkeypatch):
    recorder = _Recorder()
    _use(monkeypatch, recorder)
    assert _pick(request="") == UNDECIDED
    assert recorder.calls == []


def test_the_off_switch_makes_no_call(monkeypatch):
    recorder = _Recorder()
    _use(monkeypatch, recorder)
    monkeypatch.setitem(fastworkflow._env_vars, "FW_JEV_VALUE_PICK", "OFF")
    assert _pick() == UNDECIDED
    assert recorder.calls == []


def test_no_jev_key_makes_no_call(monkeypatch):
    recorder = _Recorder()
    _use(monkeypatch, recorder)
    monkeypatch.setattr(workflow_agent, "_jev_lm", lambda: None)
    assert _pick() == UNDECIDED
    assert recorder.calls == []


def test_an_empty_value_or_no_keys_makes_no_call(monkeypatch):
    recorder = _Recorder()
    _use(monkeypatch, recorder)
    assert DatabaseValidator.fuzzy_match_with_request("", KEYS, REQUEST, DESCRIPTIONS) == (False, None, [])
    assert DatabaseValidator.fuzzy_match_with_request("NOT_FOUND", KEYS, REQUEST, DESCRIPTIONS) == (False, None, [])
    assert DatabaseValidator.fuzzy_match_with_request("billing api", [], REQUEST, DESCRIPTIONS) == (False, None, [])
    assert recorder.calls == []


def test_a_confident_pick_is_returned_as_a_match(monkeypatch):
    recorder = _Recorder(pick="billing-api-staging", probability=0.95)
    _use(monkeypatch, recorder)
    assert _pick() == (True, "billing-api-staging", [])
    assert len(recorder.calls) == 1
    call = recorder.calls[0]
    assert call["request"] == REQUEST
    assert call["argument"] == "value written for it: 'billing api'"


def test_a_pick_below_the_threshold_returns_the_fuzzy_result(monkeypatch):
    _use(monkeypatch, _Recorder(pick="billing-api-staging", probability=0.85))
    assert _pick() == UNDECIDED


def test_ask_returns_the_fuzzy_result(monkeypatch):
    _use(monkeypatch, _Recorder(pick="ask", probability=0.99))
    assert _pick() == UNDECIDED


def test_a_failed_call_returns_the_fuzzy_result(monkeypatch):
    _use(monkeypatch, _Recorder(error=RuntimeError("provider down")))
    assert _pick() == UNDECIDED


def test_the_options_are_the_suggestions_with_their_descriptions_and_ask(monkeypatch):
    _use(monkeypatch, _Recorder())
    _pick(descriptions={"billing-api": "production", "billing-api-staging": "staging environment"})
    assert _ChoiceSpy.options == (
        ("billing-api", "billing-api (production)"),
        ("billing-api-staging", "billing-api-staging (staging environment)"),
        ("ask", "The request does not say which one; the user must choose."),
    )


def test_a_candidate_without_a_description_gets_the_default_text(monkeypatch):
    _use(monkeypatch, _Recorder())
    _pick(descriptions={"billing-api": "production"})
    assert _ChoiceSpy.options[1] == ("billing-api-staging", "billing-api-staging (no other details)")


def test_a_candidate_named_ask_makes_no_call(monkeypatch):
    recorder = _Recorder()
    _use(monkeypatch, recorder)
    keys = ["ask", "ask-me-later"]
    descriptions = {"ask": "a value", "ask-me-later": "another"}
    got = DatabaseValidator.fuzzy_match_with_request("ask me", keys, REQUEST, descriptions)
    assert got == (False, None, ["ask", "ask-me-later"])
    assert recorder.calls == []


def test_with_no_suggestions_the_five_closest_keys_are_the_candidates(monkeypatch):
    names = ["Alice Smith", "Bob Jones", "Carol White", "Dan Brown", "Eve Black", "Frank Green", "Grace Hall"]
    value = "Qxzvw Pltrm"
    assert DatabaseValidator.fuzzy_match(value, names) == (False, None, [])
    expected = sorted(names, key=lambda k: SequenceMatcher(None, value.lower(), k.lower()).ratio(),
                      reverse=True)[:5]
    recorder = _Recorder(pick=expected[0], probability=0.95)
    _use(monkeypatch, recorder)
    got = DatabaseValidator.fuzzy_match_with_request(value, names, "ask about the names", {n: "x" for n in names})
    assert got == (True, expected[0], [])
    offered = [v for v, _ in _ChoiceSpy.options if v != "ask"]
    assert offered == expected


def test_the_argument_name_prefixes_the_argument_text(monkeypatch):
    recorder = _Recorder()
    _use(monkeypatch, recorder)
    DatabaseValidator.fuzzy_match_with_request("billing api", KEYS, REQUEST, DESCRIPTIONS,
                                               argument_name="service")
    assert recorder.calls[0]["argument"] == "service; value written for it: 'billing api'"


@pytest.fixture
def jev_env():
    """Load the env files into fastworkflow and skip when JEV_API_KEY is absent; restore afterwards."""
    previous = dict(fastworkflow._env_vars)
    env_vars = {
        **dotenv_values(os.path.join(ROOT, "fastworkflow", "examples", "fastworkflow.env")),
        **dotenv_values(os.path.join(ROOT, "fastworkflow", "examples", "fastworkflow.passwords.env")),
    }
    override = os.path.join(ROOT, "passwords", ".env")
    if os.path.exists(override):
        env_vars.update(dotenv_values(override))
    if not env_vars.get("JEV_API_KEY"):
        pytest.skip("JEV_API_KEY not set in passwords/.env: the value pick needs a real Jev call")
    fastworkflow.init(env_vars=env_vars)
    workflow_agent._jev_lm.cache_clear()
    try:
        yield
    finally:
        fastworkflow.init(env_vars=previous)
        workflow_agent._jev_lm.cache_clear()


def test_real_jev_picks_the_staging_service_when_the_request_says_staging(jev_env):
    got = DatabaseValidator.fuzzy_match_with_request(
        "billing api", KEYS, "restart the billing api on staging", DESCRIPTIONS)
    assert got == (True, "billing-api-staging", [])


def test_real_jev_asks_when_the_request_does_not_say_which_service(jev_env):
    got = DatabaseValidator.fuzzy_match_with_request(
        "billing api", KEYS, "restart the billing api", DESCRIPTIONS)
    assert got[0] is False, "Jev picked a value the request does not settle; report this, do not weaken the test"

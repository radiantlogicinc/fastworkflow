"""Jev's check of the fields an LLM extraction filled: a filled field the statement does not give
is reset to its not-given value (signatures.fields_not_given). The Predict stand-ins are hand-written
(no unittest.mock); the real-Jev test skips without JEV_API_KEY."""
from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Optional

import dspy
import pytest
from dspy.utils import DummyLM
from dotenv import dotenv_values
from pydantic import BaseModel, Field

import fastworkflow
from fastworkflow import workflow_agent
from fastworkflow.utils import signatures
from fastworkflow.utils.signatures import (
    FIELD_GIVEN_THRESHOLD, NOT_FOUND, InputForParamExtraction, fields_not_given,
    reset_to_not_given)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Access(BaseModel):
    resource_id: str = Field(description="the resource to check")
    filter: Optional[str] = Field(default=None, description="Name fragment to narrow the holders")
    limit: int = Field(default=10, description="how many holders to list")
    tags: list[str] = Field(default_factory=list, description="tags to match")


def _verdict(value: str, probability: float) -> SimpleNamespace:
    other = "given" if value == "not_given" else "not_given"
    return SimpleNamespace(verdict=SimpleNamespace(
        value=value, probabilities={value: probability, other: 1.0 - probability}, confidence=probability))


class _Recorder:
    """A stand-in for dspy.Predict: answers per argument name and records every call."""

    def __init__(self, answers: dict, error: Exception | None = None):
        self.answers = answers
        self.error = error
        self.calls: list[str] = []

    def __call__(self, signature):
        def predict(statement, argument):
            if self.error is not None:
                raise self.error
            name = argument.split(":", 1)[0]
            self.calls.append(name)
            value, probability = self.answers.get(name, ("given", 0.99))
            return _verdict(value, probability)
        return predict


def _use(monkeypatch, recorder: _Recorder, lm=None):
    monkeypatch.setattr(workflow_agent, "_jev_lm", lambda: DummyLM([]) if lm is None else lm)
    monkeypatch.setattr(dspy, "Predict", recorder)


@pytest.fixture(autouse=True)
def _switch_on(monkeypatch):
    monkeypatch.setitem(fastworkflow._env_vars, "FW_JEV_FIELD_GIVEN", "on")


def test_a_filled_field_the_statement_does_not_give_is_returned_at_threshold(monkeypatch):
    assert FIELD_GIVEN_THRESHOLD == 0.9
    _use(monkeypatch, _Recorder({"filter": ("not_given", 0.95)}))
    got = fields_not_given("show_group_members", Access, {"resource_id": "R1", "filter": "Jane Doe"})
    assert got == {"filter": 0.95}


def test_a_not_given_verdict_below_threshold_is_not_returned(monkeypatch):
    _use(monkeypatch, _Recorder({"filter": ("not_given", 0.85)}))
    assert fields_not_given("show_group_members", Access, {"resource_id": "R1", "filter": "Jane Doe"}) == {}


def test_a_given_verdict_is_not_returned(monkeypatch):
    _use(monkeypatch, _Recorder({"filter": ("given", 0.99)}))
    assert fields_not_given("show_group_members filter=Jane", Access,
                            {"resource_id": "R1", "filter": "Jane"}) == {}


def test_unfilled_fields_are_never_sent_to_jev(monkeypatch):
    recorder = _Recorder({})
    _use(monkeypatch, recorder)
    values = {"resource_id": NOT_FOUND, "filter": None, "limit": 10, "tags": []}
    assert fields_not_given("show_group_members", Access, values) == {}
    assert recorder.calls == []


def test_a_failed_check_leaves_the_value_standing(monkeypatch):
    _use(monkeypatch, _Recorder({}, error=RuntimeError("provider down")))
    assert fields_not_given("show_group_members", Access, {"resource_id": "R1", "filter": "Jane Doe"}) == {}


def test_no_jev_key_makes_no_call(monkeypatch):
    recorder = _Recorder({"filter": ("not_given", 0.99)})
    _use(monkeypatch, recorder, lm=None)
    monkeypatch.setattr(workflow_agent, "_jev_lm", lambda: None)
    assert fields_not_given("show_group_members", Access, {"resource_id": "R1", "filter": "Jane Doe"}) == {}
    assert recorder.calls == []


def test_the_off_switch_makes_no_call(monkeypatch):
    recorder = _Recorder({"filter": ("not_given", 0.99)})
    _use(monkeypatch, recorder)
    monkeypatch.setitem(fastworkflow._env_vars, "FW_JEV_FIELD_GIVEN", "OFF")
    assert fields_not_given("show_group_members", Access, {"resource_id": "R1", "filter": "Jane Doe"}) == {}
    assert recorder.calls == []


def test_a_reset_required_str_is_not_given_and_an_optional_is_none(monkeypatch):
    _use(monkeypatch, _Recorder({"resource_id": ("not_given", 0.97), "filter": ("not_given", 0.95)}))
    values = {"resource_id": "R1", "filter": "Jane Doe", "limit": 3}
    reset = fields_not_given("show_group_members", Access, values)
    assert set(reset) == {"resource_id", "filter"}
    # The reset step extract_parameters runs on the result of fields_not_given (tested here directly,
    # since extract_parameters itself needs a real extraction LM).
    reset_to_not_given(Access, values, reset)
    assert values["resource_id"] == NOT_FOUND
    assert values["filter"] is None
    assert values["limit"] == 3


def test_an_instance_starts_with_no_reset_fields():
    assert InputForParamExtraction(command="show_group_members").fields_reset_not_given == {}


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
        pytest.skip("JEV_API_KEY not set in passwords/.env: the field check needs a real Jev call")
    fastworkflow.init(env_vars=env_vars)
    workflow_agent._jev_lm.cache_clear()
    try:
        yield
    finally:
        fastworkflow.init(env_vars=previous)
        workflow_agent._jev_lm.cache_clear()


class _Filter(BaseModel):
    filter: Optional[str] = Field(default=None, description="Name fragment to narrow the holders")


def test_real_jev_resets_a_filter_the_statement_does_not_give(jev_env):
    reset = fields_not_given("show_group_members", _Filter, {"filter": "Jane Doe"})
    assert reset.get("filter", 0.0) >= FIELD_GIVEN_THRESHOLD


def test_real_jev_keeps_a_filter_the_statement_gives(jev_env):
    assert fields_not_given("show_group_members filter=Jane", _Filter, {"filter": "Jane"}) == {}

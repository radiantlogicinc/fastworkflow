"""Jev's tie-break among an ambiguous intent's candidates: when it routes, and what it is offered."""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import dspy
import pytest
from dotenv import dotenv_values
from dspy.utils import DummyLM

import fastworkflow
from fastworkflow import workflow_agent
from fastworkflow.workflow_agent import INTENT_TIEBREAK_THRESHOLD, choose_ambiguous_command

ROOT = Path(__file__).resolve().parents[1]

CANDIDATES = ["ChatRoom/show_messages", "User/show_profile"]
DESCRIPTIONS = {
    "ChatRoom/show_messages": "Show the messages posted in a room.",
    "User/show_profile": "Show a user's profile.",
}


def _fake_command_predictor(value, probability, probabilities=None):
    def predict(**_inputs):
        probs = probabilities if probabilities is not None else {value: probability}
        choice = SimpleNamespace(value=value, probabilities=probs, confidence=probability)
        return SimpleNamespace(command=choice)

    return predict


def _use_predictor(monkeypatch, predict) -> None:
    monkeypatch.setattr(workflow_agent, "_jev_lm", lambda: DummyLM([]))
    monkeypatch.setattr(workflow_agent.dspy, "Predict", lambda signature: predict)


@pytest.fixture(autouse=True)
def _tiebreak_on(monkeypatch):
    monkeypatch.delenv("FW_JEV_INTENT_TIEBREAK", raising=False)


def test_a_confident_choice_is_returned(monkeypatch):
    _use_predictor(monkeypatch, _fake_command_predictor("ChatRoom/show_messages", 0.95))
    label, probability = choose_ambiguous_command(
        "show me the messages in the general room", CANDIDATES, DESCRIPTIONS)
    assert label == "ChatRoom/show_messages"
    assert probability == pytest.approx(0.95)


def test_a_choice_below_the_threshold_is_not_routed(monkeypatch):
    _use_predictor(monkeypatch, _fake_command_predictor("ChatRoom/show_messages", 0.85))
    assert choose_ambiguous_command("messages", CANDIDATES, DESCRIPTIONS) == (None, 0.85)


def test_the_threshold_is_point_nine():
    assert INTENT_TIEBREAK_THRESHOLD == 0.9


def test_none_is_not_routed(monkeypatch):
    _use_predictor(monkeypatch, _fake_command_predictor("none", 0.97))
    label, probability = choose_ambiguous_command("hello there", CANDIDATES, DESCRIPTIONS)
    assert label is None
    assert probability == pytest.approx(0.97)


def test_a_failed_call_is_no_answer_and_does_not_raise(monkeypatch):
    def broken(**_inputs):
        raise RuntimeError("Jev is down")

    _use_predictor(monkeypatch, broken)
    assert choose_ambiguous_command("messages", CANDIDATES, DESCRIPTIONS) == (None, None)


def test_without_a_jev_key_nothing_is_called(monkeypatch):
    monkeypatch.setattr(workflow_agent, "_jev_lm", lambda: None)

    def must_not_be_called(signature):
        raise AssertionError("Predict must not run without a Jev key")

    monkeypatch.setattr(workflow_agent.dspy, "Predict", must_not_be_called)
    assert choose_ambiguous_command("messages", CANDIDATES, DESCRIPTIONS) == (None, None)


@pytest.mark.parametrize("value", ["off", "0", "false", "OFF", "False"])
def test_the_switch_turns_it_off(monkeypatch, value):
    monkeypatch.setenv("FW_JEV_INTENT_TIEBREAK", value)

    def must_not_be_called(signature):
        raise AssertionError("Predict must not run while the tie-break is switched off")

    monkeypatch.setattr(workflow_agent, "_jev_lm", lambda: DummyLM([]))
    monkeypatch.setattr(workflow_agent.dspy, "Predict", must_not_be_called)
    assert choose_ambiguous_command("messages", CANDIDATES, DESCRIPTIONS) == (None, None)


def test_fewer_than_two_distinct_candidates_is_no_question(monkeypatch):
    def must_not_be_called(signature):
        raise AssertionError("Predict must not run for a single candidate")

    monkeypatch.setattr(workflow_agent, "_jev_lm", lambda: DummyLM([]))
    monkeypatch.setattr(workflow_agent.dspy, "Predict", must_not_be_called)
    assert choose_ambiguous_command("messages", ["ChatRoom/show_messages"] * 2, DESCRIPTIONS) == (None, None)


def test_the_options_offered_are_the_candidates_reserved_labels_and_none(monkeypatch):
    seen = {}

    def capture(signature):
        seen["signature"] = signature
        return _fake_command_predictor("none", 0.99)

    monkeypatch.setattr(workflow_agent, "_jev_lm", lambda: DummyLM([]))
    monkeypatch.setattr(workflow_agent.dspy, "Predict", capture)
    candidates = ["ChatRoom/show_messages", "User/show_profile", "wildcard", "parameter_value"]
    # The candidate without a description falls back to its leaf name, underscores as spaces.
    choose_ambiguous_command("messages", candidates, {
        "ChatRoom/show_messages": DESCRIPTIONS["ChatRoom/show_messages"],
    })

    signature = seen["signature"]
    assert signature.__doc__.strip() == "Decide which command a message to a workflow asks for."
    assert list(signature.input_fields) == ["message"]
    output = signature.output_fields["command"]
    # The Choice annotation carries its options as a JSON map of value -> description.
    offered = json.loads(output.annotation._criteria_json)
    assert offered["ChatRoom/show_messages"] == "Show the messages posted in a room."
    assert offered["User/show_profile"] == "show profile"
    assert offered["wildcard"] == (
        "None of these commands: the message is not a command of the current context.")
    assert offered["parameter_value"] == (
        "The message is a bare value answering an earlier question, not a command.")
    assert offered["none"] == "None of these commands fits the message."
    assert set(offered) == set(candidates) | {"none"}


def _jev_env():
    """``env/.env`` and ``passwords/.env`` loaded into fastworkflow, with Jev's cache cleared; skip
    without a JEV_API_KEY. Returns the previous env to restore."""
    previous = dict(fastworkflow._env_vars)
    env_vars = {
        **dotenv_values(str(ROOT / "fastworkflow" / "examples" / "fastworkflow.env")),
        **dotenv_values(str(ROOT / "fastworkflow" / "examples" / "fastworkflow.passwords.env")),
    }
    for override in (ROOT / "env" / ".env", ROOT / "passwords" / ".env"):
        if override.exists():
            env_vars.update(dotenv_values(str(override)))
    if not env_vars.get("JEV_API_KEY"):
        pytest.skip("JEV_API_KEY not set in passwords/.env: the tie-break needs a real Jev call")
    fastworkflow.init(env_vars=env_vars)
    workflow_agent._jev_lm.cache_clear()
    return previous


def test_real_jev_chooses_the_right_candidate():
    previous = _jev_env()
    try:
        label, probability = choose_ambiguous_command(
            "show me the messages in the general room", CANDIDATES, DESCRIPTIONS)
        assert label == "ChatRoom/show_messages"
        assert probability >= 0.9
    finally:
        fastworkflow.init(env_vars=previous)
        workflow_agent._jev_lm.cache_clear()


def test_a_command_without_a_docstring_is_described_by_its_example_utterances(setup_test_environment):
    """fix-jd9o.1: messaging_app_4's User/send_message has no docstring, so its description is its
    example utterances (measured: descriptions 148/148 routes right, bare names 148/151)."""
    workflow_agent._command_first_sentences.cache_clear()
    folder = str(ROOT / "fastworkflow" / "examples" / "messaging_app_4")
    descriptions = workflow_agent.intent_tiebreak_descriptions(
        folder, "User", ["User/send_message", "IntentDetection/go_up"])
    assert descriptions["User/send_message"].startswith("e.g. ")
    assert "john@fastworkflow.ai" in descriptions["User/send_message"]
    # a command with a docstring keeps its first sentence
    assert descriptions["IntentDetection/go_up"] == "Change context to the parent of the current context."
    workflow_agent._command_first_sentences.cache_clear()

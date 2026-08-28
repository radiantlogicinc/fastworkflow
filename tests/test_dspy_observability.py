"""Integration coverage for DSPy module/API LLM-call observability.

The load-bearing case is ``disable_history=True``: run_fastapi_mcp installs that
process-wide as a memory bound (run_fastapi_mcp/server_memory.py), so any capture
that reads ``lm.history`` records nothing in the one process the chatbot traces.
"""

from __future__ import annotations

import json

import dspy
import pytest
from dspy.utils import DummyLM

from fastworkflow import tracing
from fastworkflow.observability_store import ObservabilityStore, SQLiteTraceSink
from fastworkflow.utils.dspy_logger import DSPyForward, observe_dspy_host

QUESTION = "What is the capital of France?"
REASONING = "France's capital city is Paris."


class _TraceHost:
    def __init__(self, sink):
        self.trace_sink = sink
        self.current_turn_key = "20260826T120000.000000Z-dspytrace001"
        self.observability_channel_id = "dspy-test-channel"
        self.trace_span_stack = []


class _AuthoredModule(dspy.Module):
    """A fastWorkflow-authored module using the repo's own logging decorator."""

    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought("question -> answer")

    @DSPyForward.intercept
    def forward(self, question=None):
        return self.predictor(question=question)


def _dummy_lm() -> DummyLM:
    return DummyLM([{"reasoning": REASONING, "answer": "Paris"}] * 4)


def _capture(tmp_path, program, *, disable_history: bool) -> list[dict]:
    """Run one program under an observed turn; return its recorded spans."""
    db_path = str(tmp_path / "observability.sqlite3")
    sink = SQLiteTraceSink(db_path)
    host = _TraceHost(sink)
    try:
        with observe_dspy_host(host):
            with dspy.context(lm=_dummy_lm(), disable_history=disable_history):
                program()
        assert sink.flush()
    finally:
        sink.close()
    return ObservabilityStore(db_path).get_spans(host.current_turn_key)


def _attributes(spans: list[dict]) -> dict:
    assert len(spans) == 1, f"expected one LLM span, got {len(spans)}"
    span = spans[0]
    assert span["name"] == tracing.SPAN_LLM_CALL
    assert span["kind"] == tracing.KIND_LLM
    assert span["status"] == tracing.STATUS_OK
    return json.loads(span["attributes"])


@pytest.mark.parametrize("disable_history", [False, True])
def test_decorated_module_capture_survives_disabled_history(tmp_path, disable_history):
    module = _AuthoredModule()
    attributes = _attributes(
        _capture(
            tmp_path,
            lambda: module(question=QUESTION),
            disable_history=disable_history,
        )
    )
    assert attributes["capture_source"] == "module_decorator"
    assert attributes["module"] == "_AuthoredModule"
    assert "_AuthoredModule" in attributes["module_chain"]
    # The prompt actually sent, the completion actually returned, and the
    # module's parsed reasoning — none of which may depend on lm.history.
    assert QUESTION in attributes["messages"]
    assert QUESTION in attributes["module_input"]
    assert REASONING in attributes["output"]
    assert REASONING in attributes["module_output"]
    assert json.loads(attributes["reasoning"]) == REASONING


@pytest.mark.parametrize("disable_history", [False, True])
def test_undecorated_module_falls_back_to_the_dspy_api(tmp_path, disable_history):
    attributes = _attributes(
        _capture(
            tmp_path,
            lambda: dspy.ChainOfThought("question -> answer")(question=QUESTION),
            disable_history=disable_history,
        )
    )
    assert attributes["capture_source"] == "dspy_api"
    assert attributes["module_chain"] == "ChainOfThought > Predict"
    assert QUESTION in attributes["messages"]
    assert REASONING in attributes["output"]
    assert json.loads(attributes["reasoning"]) == REASONING


def test_history_only_fields_are_reported_as_unavailable(tmp_path):
    """Usage and cost genuinely need history; say so instead of dropping them."""
    with_history = _attributes(
        _capture(
            tmp_path / "on",
            lambda: dspy.ChainOfThought("question -> answer")(question=QUESTION),
            disable_history=False,
        )
    )
    assert "usage" in with_history
    assert "usage_capture" not in with_history
    # Token counts are usage, not credentials, and must not be redacted away.
    assert json.loads(with_history["usage"])["total_tokens"] is not None

    without_history = _attributes(
        _capture(
            tmp_path / "off",
            lambda: dspy.ChainOfThought("question -> answer")(question=QUESTION),
            disable_history=True,
        )
    )
    assert "usage" not in without_history
    assert "history is disabled" in without_history["usage_capture"]


def test_usage_is_captured_on_every_call_under_a_bounded_history(tmp_path):
    """A capped history never grows, so "did a new entry land" cannot be a length test.

    The chatbot's server keeps history at one entry per LM
    (server_memory.INSPECT_POLICY_SETTINGS) — enough to read usage back,
    small enough not to retain request-sized payloads. A list that trims as it
    appends has a constant length, so detecting the new entry by growth
    reported "history is disabled" for every call after the first.
    """
    db_path = str(tmp_path / "observability.sqlite3")
    sink = SQLiteTraceSink(db_path)
    host = _TraceHost(sink)
    calls = 4
    try:
        with observe_dspy_host(host):
            with dspy.context(
                lm=_dummy_lm(), disable_history=False, max_history_size=1
            ):
                for _ in range(calls):
                    dspy.ChainOfThought("question -> answer")(question=QUESTION)
        assert sink.flush()
    finally:
        sink.close()

    llm_spans = [
        span
        for span in ObservabilityStore(db_path).get_spans(host.current_turn_key)
        if span["name"] == tracing.SPAN_LLM_CALL
    ]
    assert len(llm_spans) == calls
    for index, span in enumerate(llm_spans):
        attributes = json.loads(span["attributes"])
        assert "usage_capture" not in attributes, (
            f"call {index} was reported as history-less under a bounded history"
        )
        assert json.loads(attributes["usage"])["total_tokens"] is not None


def test_no_spans_are_recorded_outside_an_observed_turn(tmp_path):
    db_path = str(tmp_path / "observability.sqlite3")
    sink = SQLiteTraceSink(db_path)
    host = _TraceHost(sink)
    try:
        with dspy.context(lm=_dummy_lm()):
            dspy.ChainOfThought("question -> answer")(question=QUESTION)
        sink.flush()
    finally:
        sink.close()
    assert ObservabilityStore(db_path).get_spans(host.current_turn_key) == []

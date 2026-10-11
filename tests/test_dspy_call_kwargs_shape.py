"""Contract on the shape of the `fw.llm.call` span's `call_kwargs` attribute.

The attribute exists so a reader can ask what the provider was actually told —
`call_kwargs.max_tokens`, `call_kwargs.timeout`. DSPy delivers those per-call
arguments to `on_lm_start` under a nested `kwargs` key, and recording the
payload verbatim produced `{"kwargs": {"max_tokens": ...}}`: every lookup by
argument name silently found nothing, with no error to say so. These tests pin
the flattened shape, because a shape that is only implied by a comment is a
shape that regresses.
"""

from __future__ import annotations

import json

import dspy
import pytest
from dspy.utils import DummyLM

from fastworkflow import tracing
from fastworkflow.observability.store import ObservabilityStore, SQLiteTraceSink
from fastworkflow.utils.dspy_logger import (
    DSPyObservabilityCallback,
    observe_dspy_host,
)


class _TraceHost:
    def __init__(self, sink):
        self.trace_sink = sink
        self.current_turn_key = "20260904T120000.000000Z-callkwargs01"
        self.observability_channel_id = "call-kwargs-test-channel"
        self.trace_span_stack = []


def _llm_attributes(tmp_path, program) -> dict:
    db_path = str(tmp_path / "observability.sqlite3")
    sink = SQLiteTraceSink(db_path)
    host = _TraceHost(sink)
    try:
        with observe_dspy_host(host):
            with dspy.context(
                lm=DummyLM([{"reasoning": "r", "answer": "Paris"}] * 4),
                disable_history=True,
            ):
                program()
        assert sink.flush()
    finally:
        sink.close()
    spans = [
        span
        for span in ObservabilityStore(db_path).get_spans(host.current_turn_key)
        if span["name"] == tracing.SPAN_LLM_CALL
    ]
    assert len(spans) == 1, f"expected one LLM span, got {len(spans)}"
    return json.loads(spans[0]["attributes"])


def test_per_call_provider_arguments_are_queryable_by_their_own_name(tmp_path):
    """`config=` is how `ExtractionBound.as_call_config` reaches the provider.

    That is the live producer of this attribute's only non-empty content, so it
    is the case the contract is written against.
    """
    attributes = _llm_attributes(
        tmp_path,
        lambda: dspy.Predict("question -> answer")(
            question="q?",
            config={"max_tokens": 1234, "temperature": 0.25},  # DSPy 3.4 refuses a per-call timeout on a test engine
        ),
    )
    call_kwargs = json.loads(attributes["call_kwargs"])
    assert call_kwargs["max_tokens"] == 1234
    assert call_kwargs["temperature"] == 0.25
    # The failure this contract exists to prevent: the arguments present, but
    # one level down, where no reader looks.
    assert "kwargs" not in call_kwargs


def test_an_empty_per_call_payload_records_no_attribute_at_all(tmp_path):
    """`{"kwargs": {}}` is not evidence of anything; absence is honest."""
    attributes = _llm_attributes(
        tmp_path,
        lambda: dspy.Predict("question -> answer")(question="q?"),
    )
    assert "call_kwargs" not in attributes


def _start_span_attributes(tmp_path, inputs: dict) -> dict:
    """Drive `on_lm_start` directly with a payload of our choosing.

    Going through the callback rather than a live DSPy call is deliberate: the
    shapes below are the ones a future DSPy release could hand us, and there is
    no way to ask the current release to produce them.
    """
    sink = SQLiteTraceSink(str(tmp_path / "observability.sqlite3"))
    host = _TraceHost(sink)
    callback = DSPyObservabilityCallback()
    try:
        with observe_dspy_host(host):
            callback.on_lm_start(
                "call-1", DummyLM([{"answer": "x"}]), dict(inputs)
            )
        span = callback._calls["call-1"][1]
    finally:
        sink.close()
    assert span is not None, "on_lm_start opened no span"
    return dict(span.attributes)


def test_a_non_mapping_kwargs_value_is_kept_rather_than_dropped(tmp_path):
    """Flattening must not become a way to lose evidence about a payload the
    contract did not anticipate."""
    attributes = _start_span_attributes(
        tmp_path, {"messages": [], "kwargs": ["not", "a", "mapping"]}
    )
    assert json.loads(attributes["call_kwargs"]) == {
        "kwargs": ["not", "a", "mapping"]
    }


def test_a_top_level_argument_wins_over_the_nested_copy(tmp_path):
    """A key the callback delivered at the top level came from DSPy directly;
    the nested copy is the one being unwrapped, so it must not overwrite it."""
    attributes = _start_span_attributes(
        tmp_path,
        {
            "messages": [],
            "max_tokens": 4096,
            "kwargs": {"max_tokens": 20480, "timeout": 908.0},
        }
    )
    call_kwargs = json.loads(attributes["call_kwargs"])
    assert call_kwargs["max_tokens"] == 4096
    assert call_kwargs["timeout"] == 908.0


def test_call_kwargs_is_a_declared_attribute_of_the_llm_span_contract():
    contract = tracing.SPAN_CONTRACTS[tracing.SPAN_LLM_CALL]
    assert "call_kwargs" in contract.attributes


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

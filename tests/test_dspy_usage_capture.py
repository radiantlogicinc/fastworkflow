"""Per-call usage, cost and cache status, captured from the call's own result.

Every case runs with ``disable_history=True`` -- the setting run_fastapi_mcp and
the experiment harness install (run_fastapi_mcp/server_memory.py) -- because a
capture that needs DSPy history records nothing exactly where it is wanted.
Offline only: fake LMs return real litellm ``ModelResponse`` objects, and the
Jev client is driven through a fake ``typesafe_sdk`` client.
"""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import dspy
import pytest
from dspy.clients.cache import Cache
from litellm import ModelResponse
from litellm.types.utils import Choices, Message, Usage

from fastworkflow import tracing
from fastworkflow.observability.store import ObservabilityStore, SQLiteTraceSink
from fastworkflow.utils.dspy_logger import observe_dspy_host


class _TraceHost:
    def __init__(self, sink, turn_key="20261022T120000.000000Z-usagecapture1"):
        self.trace_sink = sink
        self.current_turn_key = turn_key
        self.observability_channel_id = "usage-capture-channel"
        self.trace_span_stack = []


def _response(prompt_tokens: int, cost: float) -> ModelResponse:
    response = ModelResponse(
        choices=[Choices(message=Message(
            content="[[ ## answer ## ]]\nok\n\n[[ ## completed ## ]]", role="assistant"))],
        model="fake/provider-model",
        usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=7,
                    total_tokens=prompt_tokens + 7),
    )
    response._hidden_params["response_cost"] = cost
    return response


class _FakeLM(dspy.LM):
    """Only the provider call is replaced; prompt tokens encode the question."""

    def __init__(self, cache=False, barrier=None):
        super().__init__(model="fake/provider-model", cache=cache)
        self._barrier = barrier

    def forward(self, prompt=None, messages=None, **kwargs):
        text = json.dumps(messages)
        tokens = 1000 if "first" in text else 2000
        if self._barrier is not None:
            self._barrier.wait(timeout=10)  # both workers are mid-call together
        return _response(tokens, cost=tokens / 1_000_000)


@pytest.fixture
def private_cache(monkeypatch):
    monkeypatch.setattr(dspy, "cache", Cache(enable_disk_cache=False, enable_memory_cache=True,
                                             disk_cache_dir=None))


def _run(tmp_path, body, turn_key="20261022T120000.000000Z-usagecapture1"):
    db_path = str(tmp_path / "observability.sqlite3")
    sink = SQLiteTraceSink(db_path)
    host = _TraceHost(sink, turn_key)
    try:
        with observe_dspy_host(host):
            with dspy.context(disable_history=True):
                body()
        assert sink.flush()
    finally:
        sink.close()
    return [json.loads(span["attributes"])
            for span in ObservabilityStore(db_path).get_spans(host.current_turn_key)
            if span["name"] == tracing.SPAN_LLM_CALL]


def _predict(lm, question):
    with dspy.context(lm=lm):
        return dspy.Predict("question -> answer")(question=question)


def test_hooks_are_installed_on_the_dspy_result_points():
    """Pins the DSPy internals the capture rides on; an upgrade that renames
    them fails here rather than silently recording no usage."""
    from dspy.clients import execution
    from dspy.clients.typesafe import TypeSafe

    assert getattr(execution.finalize, "_fastworkflow_capture", False)
    assert getattr(TypeSafe._finish, "_fastworkflow_capture", False)


def test_usage_and_cost_are_recorded_with_history_disabled(tmp_path):
    lm = _FakeLM()
    [attributes] = _run(tmp_path, lambda: _predict(lm, "first"))
    assert "usage_capture" not in attributes
    usage = json.loads(attributes["usage"])
    assert (usage["prompt_tokens"], usage["completion_tokens"], usage["total_tokens"]) == (1000, 7, 1007)
    assert attributes["cost"] == pytest.approx(0.001)
    assert attributes["cache_hit"] is False
    assert attributes["response_model"] == "fake/provider-model"
    assert attributes["history_uuid"]
    assert lm.history == []  # nothing was read from (or kept in) history


def test_two_threads_sharing_one_lm_each_get_their_own_usage(tmp_path):
    lm = _FakeLM(barrier=threading.Barrier(2))
    errors = []

    def worker(question):
        try:
            _predict(lm, question)
        except Exception as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    def body():
        import contextvars

        threads = [threading.Thread(target=contextvars.copy_context().run, args=(worker, q))
                   for q in ("first", "second")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    spans = _run(tmp_path, body)
    assert not errors
    assert len(spans) == 2
    by_question = {("first" if "first" in span["messages"] else "second"): span for span in spans}
    assert json.loads(by_question["first"]["usage"])["prompt_tokens"] == 1000
    assert json.loads(by_question["second"]["usage"])["prompt_tokens"] == 2000
    assert by_question["first"]["cost"] == pytest.approx(0.001)
    assert by_question["second"]["cost"] == pytest.approx(0.002)
    assert by_question["first"]["history_uuid"] != by_question["second"]["history_uuid"]


def test_a_cache_hit_is_marked_and_bills_no_tokens_or_cost(tmp_path, private_cache):
    # The real LiteLLM engine, with litellm's own offline mock_response: a real
    # ModelResponse whose usage and response_cost litellm computes itself.
    lm = dspy.LM("openai/gpt-4o-mini", engine="litellm", cache=True,
                 mock_response="[[ ## answer ## ]]\nok\n\n[[ ## completed ## ]]")
    miss, hit = _run(tmp_path, lambda: [_predict(lm, "first") for _ in range(2)])
    assert miss["cache_hit"] is False
    assert json.loads(miss["usage"])["total_tokens"] > 0
    assert miss["cost"] > 0
    assert hit["cache_hit"] is True
    assert json.loads(hit["usage"]) == {}
    assert hit["cost"] == 0  # nothing was billed, though DSPy keeps the original cost
    assert miss["history_uuid"] != hit["history_uuid"]


class _FakeTypeSafeClient:
    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def system_one(self, state, questions):
        return SimpleNamespace(
            model="jev-latest",
            usage=SimpleNamespace(input_tokens=321, output_tokens=4),
            answers={name: SimpleNamespace(choice="a", confidence=0.9, probabilities={"a": 0.9, "b": 0.1})
                     for name in questions},
        )


def test_jev_usage_is_recorded_and_its_missing_cost_stays_unrecorded(tmp_path, monkeypatch, private_cache):
    import typesafe_sdk
    from dspy.experimental import TypeSafe

    monkeypatch.setattr(typesafe_sdk, "TypeSafeClient", _FakeTypeSafeClient)
    jev = TypeSafe(api_key="test-key", cache=True)
    questions = {"job": {"type": "choice", "instructions": "Which?", "criteria": {"a": "A", "b": "B"}}}
    miss, hit = _run(tmp_path, lambda: [jev(state="do a", questions=questions) for _ in range(2)])

    assert "usage_capture" not in miss
    assert json.loads(miss["usage"]) == {"prompt_tokens": 321, "completion_tokens": 4}
    assert miss["cost"] is None  # Jev reports no cost; none is invented
    assert miss["cache_hit"] is False
    assert miss["response_model"] == "jev-latest"
    assert hit["cache_hit"] is True
    assert json.loads(hit["usage"]) == {}
    assert hit["cost"] == 0

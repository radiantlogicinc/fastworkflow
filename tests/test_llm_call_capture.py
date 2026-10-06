"""The capture policy on an `fw.llm.call` span's content attributes (fix-sblf).

Under the `evidence` profile the prompt, the completion and everything that
quotes them is withheld behind a capture envelope -- whether it arrived whole,
cut into a `tracing.cap_attr_value` envelope, or as a structured value -- and
the cut envelope's raw prefix and raw sha256 do not survive beside it. The
bookkeeping the cost and usage readers depend on is untouched, and under the
`debug` profile nothing changes at all.

Real tracing, real enrichers, real SQLite sink (no mocks).
"""

from __future__ import annotations

import hashlib
import json
import uuid

import pytest

from fastworkflow import tracing
from fastworkflow.observability import capture_policy, enrichment, prompt_slots
from fastworkflow.observability import store as obs

SECRET = "Angelica Schneider lives at 12 Quarry Lane"
CONTENT_KEYS = (
    "messages",
    "prompt",
    "output",
    "reasoning",
    "module_input",
    "module_output",
    "module_exception",
    "exception",
    "provider_response",
)
BOOKKEEPING = {
    "usage": json.dumps({"prompt_tokens": 1200, "completion_tokens": 30, "total_tokens": 1230}),
    "cost": 0.0042,
    "call_kwargs": json.dumps({"max_tokens": 512, "temperature": 0.0}),
    "cache_hit": False,
    "history_uuid": "history-1",
    "response_model": "test-model-001",
}


class _Host:
    def __init__(self, sink, turn_key: str) -> None:
        self.trace_sink = sink
        self.current_turn_key = turn_key
        self.observability_channel_id = "llm-call-capture"
        self.trace_span_stack: list = []


@pytest.fixture(autouse=True)
def _enrichers(monkeypatch):
    monkeypatch.setenv("FW_OBS_RETENTION_DAYS", "100000")
    manifest_was_installed = enrichment._manifest_registration is not None
    prompt_slots.install_prompt_slot_enrichment()
    enrichment.install_trajectory_manifest_enrichment()
    yield
    prompt_slots.uninstall_prompt_slot_enrichment()
    if not manifest_was_installed:
        enrichment.uninstall_trajectory_manifest_enrichment()


@pytest.fixture
def db_path(tmp_path) -> str:
    return str(tmp_path / "observability.sqlite3")


def _messages(padding: int) -> list[dict]:
    user = (
        f"[[ ## user_query ## ]]\nwhere does {SECRET}?\n\n"
        "[[ ## trajectory ## ]]\n"
        "[[ ## thought_0 ## ]]\nlook them up\n\n"
        f"[[ ## observation_0 ## ]]\nFound: {SECRET}\n\n"
        "Respond with the corresponding output fields."
    )
    return [
        {"role": "system", "content": "You are an Agent.\n" + ("Commands.\n" * padding)},
        {"role": "user", "content": user},
    ]


def _content(messages_json: str, padding: int) -> tuple[dict, dict]:
    start = {
        "model": "test-model",
        "module": "fastWorkflowReAct",
        "capture_source": "dspy_api",
        "module_chain": "fastWorkflowReAct > Predict",
        "messages": messages_json,
        "prompt": f"prompt for {SECRET}" + ("." * padding),
        "module_input": json.dumps({"user_query": SECRET}),
        "call_kwargs": BOOKKEEPING["call_kwargs"],
    }
    end = {
        "output": json.dumps([f"answer: {SECRET}" + ("!" * padding)]),
        "reasoning": json.dumps(f"because {SECRET}"),
        "module_output": json.dumps({"answer": SECRET}),
        "module_exception": f"ValueError('{SECRET}')",
        "exception": f"RuntimeError('{SECRET}')",
        "provider_response": json.dumps({"choices": [{"text": SECRET}]}),
        **{key: value for key, value in BOOKKEEPING.items() if key != "call_kwargs"},
    }
    return start, end


def _record(db_path: str, start: dict, end: dict) -> dict:
    turn_key = f"20261006T000000.000000Z-{uuid.uuid4().hex[:12]}"
    sink = obs.SQLiteTraceSink(db_path)
    host = _Host(sink, turn_key)
    try:
        span = tracing.start_span(
            host, tracing.SPAN_LLM_CALL, kind=tracing.KIND_LLM,
            attributes=start, use_stack=False,
        )
        assert span is not None
        tracing.end_span(host, span, attributes=end)
        assert sink.flush()
    finally:
        sink.close()
    [row] = obs.ReadOnlyObservabilityStore(db_path).get_spans(turn_key)
    return row


def _raw_digests(*texts: str) -> list[str]:
    return [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts]


def test_the_store_polices_the_llm_call_span_the_producer_names():
    """Restated, not imported: a drift silently stops policing the prompt."""
    assert obs._SPAN_LLM_CALL == tracing.SPAN_LLM_CALL
    policed = obs._POLICED_SPAN_ATTRIBUTES[tracing.SPAN_LLM_CALL]
    contract = tracing.SPAN_CONTRACTS[tracing.SPAN_LLM_CALL]
    assert set(CONTENT_KEYS) <= set(policed)
    assert set(policed) - {"trajectory_manifest"} <= set(contract.attributes)
    assert not set(policed) & set(BOOKKEEPING)
    assert not set(policed) & {"model", "module", "capture_source", "module_chain", "usage_capture"}


@pytest.mark.parametrize("padding", [0, 3000], ids=["under-cap", "over-cap"])
def test_under_the_evidence_profile_every_content_attribute_is_withheld(
    db_path, monkeypatch, padding
):
    monkeypatch.setenv(obs.CAPTURE_PROFILE_VAR, "evidence")
    messages_json = json.dumps(_messages(padding * 3), ensure_ascii=False)
    start, end = _content(messages_json, padding * 10)
    over_cap = padding > 0
    assert (len(messages_json.encode()) > tracing.MAX_ATTR_BYTES) is over_cap
    assert (len(end["output"].encode()) > tracing.MAX_ATTR_BYTES) is over_cap

    row = _record(db_path, start, end)
    attributes = json.loads(row["attributes"])

    for key in CONTENT_KEYS:
        assert capture_policy.is_capture_envelope(attributes[key]), key
        assert "value" not in attributes[key] and "sha256" not in attributes[key], key
    for key in ("messages", "prompt", "output"):
        assert attributes[key].get("truncated_before_capture", False) is over_cap, key
    assert SECRET not in row["attributes"]
    for digest in _raw_digests(messages_json, start["prompt"], end["output"]):
        assert digest not in row["attributes"]
    assert {key: attributes[key] for key in BOOKKEEPING} == BOOKKEEPING
    assert attributes["model"] == "test-model"
    assert attributes["module_chain"] == "fastWorkflowReAct > Predict"


def test_a_messages_the_manifest_replaced_is_withheld_and_so_is_the_manifest(
    db_path, monkeypatch
):
    """The manifest empties the cut prefix but leaves the raw sha256; its rows
    digest each raw observation. Neither digest may reach an evidence row."""
    monkeypatch.setenv(obs.CAPTURE_PROFILE_VAR, "evidence")
    messages_json = json.dumps(_messages(9000), ensure_ascii=False)
    start, end = _content(messages_json, 0)
    manifest = enrichment.manifest_from_messages_json(messages_json)
    assert manifest is not None

    row = _record(db_path, start, end)
    attributes = json.loads(row["attributes"])

    assert attributes["messages"]["replaced_by"] == "trajectory_manifest"
    assert attributes["messages"]["truncated_before_capture"] is True
    assert "sha256" not in attributes["messages"]
    assert capture_policy.is_capture_envelope(attributes["trajectory_manifest"])
    for observation in manifest["observations"]:
        assert observation["sha256"] not in row["attributes"]
    assert _raw_digests(messages_json)[0] not in row["attributes"]


def test_a_structured_content_value_is_withheld_too(db_path, monkeypatch):
    monkeypatch.setenv(obs.CAPTURE_PROFILE_VAR, "evidence")
    start, end = _content(json.dumps(_messages(0)), 0)
    end["provider_response"] = {"choices": [{"text": SECRET}]}
    end["module_output"] = [SECRET]

    attributes = json.loads(_record(db_path, start, end)["attributes"])

    assert capture_policy.is_capture_envelope(attributes["provider_response"])
    assert capture_policy.is_capture_envelope(attributes["module_output"])
    assert SECRET not in json.dumps(attributes)


@pytest.mark.parametrize("padding", [0, 3000], ids=["under-cap", "over-cap"])
def test_under_the_debug_profile_nothing_changes(db_path, monkeypatch, padding):
    monkeypatch.setenv(obs.CAPTURE_PROFILE_VAR, "debug")
    messages_json = json.dumps(_messages(padding * 3), ensure_ascii=False)
    start, end = _content(messages_json, padding * 10)
    end["provider_response"] = {"choices": [{"text": SECRET}]}

    attributes = json.loads(_record(db_path, start, end)["attributes"])

    expected = tracing._capped({**start, **end})
    expected.pop(tracing.ATTR_PROMPT_SLOT_TEXTS, None)
    for key in (*CONTENT_KEYS, *BOOKKEEPING, "trajectory_manifest"):
        assert attributes.get(key) == expected.get(key), key
    assert not any(capture_policy.is_capture_envelope(value) for value in attributes.values())
    if padding:
        assert attributes["messages"]["sha256"] == _raw_digests(messages_json)[0]

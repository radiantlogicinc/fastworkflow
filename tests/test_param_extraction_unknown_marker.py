"""Parameter extraction maps the ``<UNKNOWN>`` marker to not-given and keeps the JSON schema (fix-1pfu).

Extraction runs under plain ``dspy.JSONAdapter``, which sends the signature's JSON schema as
``response_format``. The schema stays because its type checks stop the model packing two IDs into
one string field (the airline failure). On Bedrock, litellm turns the schema into a forced tool
call, and Claude then sometimes writes ``<UNKNOWN>`` for an argument the request does not give
instead of the prompt's ``NOT_FOUND``. That marker is therefore mapped to the field's not-given
value at the extraction boundary, and bracketed unknown variants are treated as placeholders.

These tests run offline. The LM is a recording DummyLM that reports the capabilities of a Bedrock
Claude model (it accepts response_format and native tools), so the real request kwargs are checked
at the LLM boundary. The live Bedrock check is in ``tests/test_param_extraction_live_bedrock.py``
and is not run by default.
"""
from typing import List

import dspy
import pytest
from dspy.utils import DummyLM
from pydantic import BaseModel

from fastworkflow.utils import signatures
from fastworkflow.utils.signatures import (
    NOT_FOUND,
    InputForParamExtraction,
    classify_bracketed_value,
    drop_unknown_markers,
)


class _Params(BaseModel):
    order_id: str
    address: str
    item_ids: List[str]


class _BedrockLikeLM(DummyLM):
    """A scripted LM that accepts response_format and native tools, as Bedrock Claude does, and
    records each request's kwargs at the LLM boundary."""

    def __init__(self, answers):
        super().__init__(answers, adapter=dspy.JSONAdapter())
        self.requests = []

    @property
    def supported_params(self):
        return {"response_format", "tools", "tool_choice"}

    @property
    def supports_response_schema(self):
        return True

    def forward(self, prompt=None, messages=None, **kwargs):
        self.requests.append(dict(kwargs))
        return super().forward(prompt=prompt, messages=messages, **kwargs)


@pytest.fixture
def extraction(monkeypatch):
    """Route extract_parameters to a scripted LM; no provider, no env, no Jev call."""
    monkeypatch.setattr(signatures, "LLM_PARAM_EXTRACTION", "bedrock/test-model")
    monkeypatch.setattr(signatures, "_field_given_enabled", lambda: False)

    def run(answer):
        lm = _BedrockLikeLM([answer] * 3)
        monkeypatch.setattr(signatures.dspy_utils, "get_lm", lambda *a, **k: lm)
        result = InputForParamExtraction(command="cancel order #W1").extract_parameters(
            CommandParameters=_Params, subject_command_name=None, workflow_folderpath=None)
        return lm, result

    return run


def test_extraction_sends_response_format(extraction):
    """The JSON schema is kept: extraction sends response_format, so the type checks apply."""
    lm, _ = extraction({
        "reasoning": "order id given; address and items are not",
        "order_id": "#W1",
        "address": NOT_FOUND,
        "item_ids": [],
    })
    assert lm.requests, "extraction made no LM call"
    for request in lm.requests:
        assert "response_format" in request, "extraction did not send the JSON schema"


def test_json_reply_is_parsed_into_the_fields(extraction):
    _, result = extraction({
        "reasoning": "order id and address given",
        "order_id": "#W1",
        "address": "12 Oak Street",
        "item_ids": ["sku-1", "sku-2"],
    })
    assert result.order_id == "#W1"
    assert result.address == "12 Oak Street"
    assert result.item_ids == ["sku-1", "sku-2"]


def test_unknown_marker_from_the_model_is_not_given(extraction):
    """``<UNKNOWN>`` for an unstated field becomes its not-given value: NOT_FOUND for a str, [] for a
    list. It does not reach the command as the value UNKNOWN."""
    _, result = extraction({
        "reasoning": "address not stated",
        "order_id": "#W1",
        "address": "<UNKNOWN>",
        "item_ids": ["<UNKNOWN>"],
    })
    assert result.order_id == "#W1"
    assert result.address == NOT_FOUND
    assert result.item_ids == []


@pytest.mark.parametrize("marker", ["<UNKNOWN>", "<unknown>", "<<UNKNOWN>>", " < UNKNOWN > "])
def test_bracketed_unknown_is_a_placeholder_not_a_value(marker):
    kind, value = classify_bracketed_value(marker, "address")
    assert kind == "placeholder"
    assert value == marker


def test_drop_unknown_markers_keeps_real_list_items():
    assert drop_unknown_markers(["sku-1", "<UNKNOWN>"], []) == ["sku-1"]
    assert drop_unknown_markers("<UNKNOWN>", NOT_FOUND) == NOT_FOUND
    assert drop_unknown_markers("Oak Street", NOT_FOUND) == "Oak Street"

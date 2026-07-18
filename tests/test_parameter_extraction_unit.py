"""Unit tests for the parameter-extraction robustness helpers.

Covers _unwrap_optional (Optional/Union unwrapping) and the XML list-field
extraction (repeated tags collected via findall, robust list parsing). Pure
logic — no LLM/network.
"""

from __future__ import annotations

from typing import List, Optional, Union

from pydantic import BaseModel, Field

from fastworkflow._workflows.command_metadata_extraction.parameter_extraction import (
    _unwrap_optional,
    ParameterExtraction,
)


# ---------------------------------------------------------------------------
# _unwrap_optional
# ---------------------------------------------------------------------------

def test_unwrap_optional_plain_type_unchanged():
    inner, origin = _unwrap_optional(str)
    assert inner is str
    assert origin is None


def test_unwrap_optional_unwraps_optional():
    inner, origin = _unwrap_optional(Optional[str])
    assert inner is str


def test_unwrap_optional_unwraps_optional_list_to_list_origin():
    inner, origin = _unwrap_optional(Optional[List[str]])
    # Origin should be list so callers can detect list fields.
    assert origin is list


def test_unwrap_optional_union_with_none():
    inner, _ = _unwrap_optional(Union[int, None])
    assert inner is int


# ---------------------------------------------------------------------------
# _extract_parameters_from_xml — list fields via repeated tags
# ---------------------------------------------------------------------------

class _ListParams(BaseModel):
    order_id: str = Field(default="NOT_FOUND")
    item_ids: List[str] = Field(default_factory=list)


def test_xml_extraction_collects_repeated_list_tags():
    # An agent may repeat a list tag once per item; all must be collected
    # (regression: re.search dropped all but the first).
    command = (
        "<order_id>#W1</order_id> "
        "<item_ids>a</item_ids> <item_ids>b</item_ids> <item_ids>c</item_ids>"
    )
    result = ParameterExtraction._extract_parameters_from_xml(command, _ListParams)
    assert result is not None
    assert result.order_id == "#W1"
    assert result.item_ids == ["a", "b", "c"]


def test_xml_extraction_single_list_item():
    command = "<order_id>#W1</order_id> <item_ids>only</item_ids>"
    result = ParameterExtraction._extract_parameters_from_xml(command, _ListParams)
    assert result is not None
    assert result.item_ids == ["only"]


class _MultiParams(BaseModel):
    order_id: str = Field(default="NOT_FOUND")
    reason: str = Field(default="NOT_FOUND")


def test_xml_extraction_multiple_scalar_fields():
    command = "<order_id>#W123</order_id> <reason>no longer needed</reason>"
    result = ParameterExtraction._extract_parameters_from_xml(command, _MultiParams)
    assert result is not None
    assert result.order_id == "#W123"
    assert result.reason == "no longer needed"


def test_xml_extraction_no_params_returns_empty_model():
    class _NoParams(BaseModel):
        pass

    result = ParameterExtraction._extract_parameters_from_xml("anything", _NoParams)
    assert result is not None
    assert isinstance(result, _NoParams)

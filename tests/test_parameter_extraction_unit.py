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


# ---------------------------------------------------------------------------
# _extract_parameters_from_key_values — ``name=value`` commands (fix-hia1.10)
# ---------------------------------------------------------------------------

class _FindParams(BaseModel):
    query: Optional[str] = Field(default=None)
    start: Optional[int] = Field(default=None)


def _kv(command, cls=_FindParams):
    return ParameterExtraction._extract_parameters_from_key_values(command, cls)


def test_key_value_keeps_spaces_and_trailing_digits():
    for value in ("Shelf_Reading Room Clerk 1", "Archive_Senior Records Analyst 2"):
        result = _kv(f"query={value}")
        assert result is not None
        assert result.query == value
        assert result.start is None


def test_key_value_after_a_leading_command_token():
    text = "find_book query=Archive_Senior Records Analyst 2"
    for name in ("find_book", "Shelf/find_book"):
        result = ParameterExtraction._extract_parameters_from_key_values(
            text, _FindParams, name)
        assert (result.query, result.start) == ("Archive_Senior Records Analyst 2", None)
    assert ParameterExtraction._extract_parameters_from_key_values(
        "Shelf/find_book query=Ann", _FindParams, "Shelf/find_book").query == "Ann"


def test_key_value_lead_token_must_be_this_command():
    # fix-hia1.12: the arguments of a call routed to another command are not bound.
    assert ParameterExtraction._extract_parameters_from_key_values(
        "open_book query=Ann", _FindParams, "Shelf/find_book") is None
    assert _kv("find_book query=Ann") is None  # no command name given: no lead allowed


def test_key_value_two_keys_converts_types():
    result = _kv("query=Ann Lee 2 start=20")
    assert (result.query, result.start) == ("Ann Lee 2", 20)


def test_key_value_quoted_value_and_trailing_note():
    assert _kv('query="Ann Lee"').query == "Ann Lee"
    assert _kv("query='Ann Lee' start=5").query == "Ann Lee"
    assert _kv("query=Ann Lee (this opens it)").query == "Ann Lee"


def test_key_value_falls_back_on_doubt():
    assert _kv("query=Ann sort=name") is None  # unknown key
    assert _kv("title=Dune") is None  # unknown key alone
    assert _kv("query=Ann start=soon") is None  # not an int
    assert _kv("query=<the name>") is None  # placeholder
    assert _kv("query=Ann query=Lee") is None  # repeated key
    assert _kv("please find query=Ann") is None  # prose before the pairs
    assert _kv("query=") is None  # empty value


def test_key_value_bare_command_is_untouched():
    assert _kv("") is None
    assert _kv("find_book") is None
    assert _kv("Ann Lee") is None


class _RequiredParams(BaseModel):
    order_id: str
    reason: str = "NOT_FOUND"
    item_ids: List[str] = Field(default_factory=list)


def test_key_value_missing_required_field_left_for_validation():
    result = _kv("reason=no longer needed", _RequiredParams)
    assert result is not None
    assert result.reason == "no longer needed"
    assert result.order_id is None
    assert _kv("order_id=W1 item_ids=a, b", _RequiredParams).item_ids == ["a", "b"]

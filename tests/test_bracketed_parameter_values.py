"""Angle brackets copied from usage text around a parameter value (fix-7t0q).

A real value inside brackets is unwrapped before the command runs; bracketed
text with no value in it is refused as a parameter error the caller sees.
"""
from types import SimpleNamespace
from typing import List, Optional

import pytest
from pydantic import BaseModel, Field

from fastworkflow.utils import signatures
from fastworkflow.utils.signatures import (
    InputForParamExtraction,
    classify_bracketed_value,
)


@pytest.fixture(autouse=True)
def _error_messages(monkeypatch):
    monkeypatch.setattr(signatures, "MISSING_INFORMATION_ERRMSG", "Missing parameter values: ")
    monkeypatch.setattr(signatures, "INVALID_INFORMATION_ERRMSG", "Invalid parameter values: ")
    monkeypatch.setattr(signatures, "NOT_FOUND", "NOT_FOUND")


class ItemParams(BaseModel):
    item_uid: str = Field(description="uid of the item")


class SearchParams(BaseModel):
    query: str = Field(description="search text")
    filter: Optional[str] = Field(default=None, description="narrowing text")


class ListParams(BaseModel):
    item_uids: List[str] = Field(description="uids")


AGENT_WORKFLOW = SimpleNamespace(context={"run_as_agent": True})


def _validate(params, hook_class=None, diagnostics=None):
    extraction = InputForParamExtraction(
        command="cmd", input_for_param_extraction_class=hook_class)
    return extraction.validate_parameters(
        AGENT_WORKFLOW, "Ctx/cmd", params, diagnostics=diagnostics)


# --- classification ----------------------------------------------------------

@pytest.mark.parametrize("value, expected", [
    ("<9a1b2c3d4e5f60718293a4b5c6d7e8f9>", "9a1b2c3d4e5f60718293a4b5c6d7e8f9"),
    ("  < 9a1b2c3d4e5f60718293a4b5c6d7e8f9 > ", "9a1b2c3d4e5f60718293a4b5c6d7e8f9"),
    ("<<abc123>>", "abc123"),
    ("<Ann Lee>", "Ann Lee"),
    ("<PayHub>", "PayHub"),
    ("<payhub>", "payhub"),
    ("<case_Ug6A17H6389PRTCiJ1MeSH>", "case_Ug6A17H6389PRTCiJ1MeSH"),
])
def test_a_real_value_in_brackets_is_unwrapped(value, expected):
    assert classify_bracketed_value(value, "item_uid") == ("unwrapped", expected)


@pytest.mark.parametrize("value, field", [
    ("<item_uid>", "repository_uid"),
    ("<uid>", "identity_uid"),
    ("<UID>", "identity_uid"),
    ("<first_order_uid>", "order_uid"),
    ("<account_uid_from_find_account>", "account_uid"),
    ("<&_catalog_item_uid_>", "item_uid"),
    ("<filter>", "filter"),
    ("<>", "query"),
    ("<...>", "query"),
    ("<value>", "query"),
])
def test_bracketed_text_with_no_value_is_a_placeholder(value, field):
    assert classify_bracketed_value(value, field) == ("placeholder", value)


@pytest.mark.parametrize("value", [
    "9a1b2c3d4e5f60718293a4b5c6d7e8f9",
    "a <b> c",
    "<a> and <b>",
    "x<y>",
    "desired-tag",
    None,
    42,
])
def test_values_not_wholly_bracketed_are_left_alone(value):
    assert classify_bracketed_value(value, "item_uid") == ("unchanged", value)


def test_list_items_are_checked_one_by_one():
    assert classify_bracketed_value(["<abc123>", "def456"], "item_uids") == (
        "unwrapped", ["abc123", "def456"])
    assert classify_bracketed_value(["abc123", "<item_uid>"], "item_uids")[0] == "placeholder"


# --- validate_parameters -------------------------------------------------------

def test_validation_unwraps_before_the_hook_sees_the_value():
    seen = {}

    class Hook:
        @staticmethod
        def validate_extracted_parameters(workflow, command, cmd_parameters):
            seen["item_uid"] = cmd_parameters.item_uid
            return True, ""

    params = ItemParams.model_construct(item_uid="<9a1b2c3d4e5f60718293a4b5c6d7e8f9>")
    diagnostics = {}
    is_valid, _, _, fields = _validate(params, Hook, diagnostics)

    assert is_valid and fields == []
    assert params.item_uid == "9a1b2c3d4e5f60718293a4b5c6d7e8f9"
    assert seen == {"item_uid": "9a1b2c3d4e5f60718293a4b5c6d7e8f9"}
    assert diagnostics["unwrapped_bracketed_values"] == {
        "item_uid": "<9a1b2c3d4e5f60718293a4b5c6d7e8f9>"}


def test_a_placeholder_is_refused_with_a_message_naming_the_field():
    hook_calls = []

    class Hook:
        @staticmethod
        def validate_extracted_parameters(workflow, command, cmd_parameters):
            hook_calls.append(command)
            return True, ""

    params = ItemParams.model_construct(item_uid="<item_uid>")
    diagnostics = {}
    is_valid, message, _, fields = _validate(params, Hook, diagnostics)

    assert not is_valid
    assert fields == ["item_uid"]
    assert "item_uid '<item_uid>' is a placeholder, not a value" in message
    assert "earlier command's output" in message
    assert hook_calls == []
    assert diagnostics["placeholder_values"] == {"item_uid": "<item_uid>"}
    assert diagnostics["invalid_fields"] == ["item_uid"]


def test_a_placeholder_is_not_sent_to_db_lookup():
    lookups = []

    class LookupParams(BaseModel):
        item_uid: str = Field(description="uid", json_schema_extra={"db_lookup": True})

    class Hook:
        @staticmethod
        def db_lookup(workflow, field_name, field_value):
            lookups.append(field_value)
            return False, None, ["abc"]

    is_valid, message, _, fields = _validate(
        LookupParams.model_construct(item_uid="<uid>"), Hook)

    assert not is_valid and fields == ["item_uid"]
    assert lookups == []
    assert "Did you mean" not in message


def test_only_the_placeholder_field_is_refused():
    params = SearchParams.model_construct(query="<Ann Lee>", filter="<filter>")
    is_valid, message, _, fields = _validate(params)

    assert not is_valid
    assert fields == ["filter"]
    assert params.query == "Ann Lee"
    assert "filter '<filter>' is a placeholder" in message


def test_plain_values_validate_as_before():
    is_valid, message, _, fields = _validate(
        SearchParams.model_construct(query="desired-tag", filter=None))
    assert is_valid and fields == []
    assert message == "All required parameters are valid."


def test_a_list_field_with_a_placeholder_item_is_refused():
    is_valid, message, _, fields = _validate(
        ListParams.model_construct(item_uids=["abc123", "<item_uid>"]))
    assert not is_valid and fields == ["item_uids"]
    assert "is a placeholder" in message

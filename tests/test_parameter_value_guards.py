"""Kept quotes, unbracketed placeholders and made-up numbers (fix-7t0q).

The bracket guard in test_bracketed_parameter_values.py covers ``<...>``. These
cover the rest of the same class of extraction mistakes: a value that kept the
quotes the request set it off with, a placeholder-shaped value the request never
writes, and an int the model filled with a sentinel or an impossible number.
"""
from types import SimpleNamespace
from typing import List, Optional

import pytest
from pydantic import BaseModel, Field

from fastworkflow.utils import signatures
from fastworkflow.utils.signatures import (
    INVALID_INT_VALUE,
    InputForParamExtraction,
    is_sentinel_number,
    is_ungrounded_placeholder,
    unwrap_quoted_value,
)


@pytest.fixture(autouse=True)
def _error_messages(monkeypatch):
    monkeypatch.setattr(signatures, "MISSING_INFORMATION_ERRMSG", "Missing parameter values: ")
    monkeypatch.setattr(signatures, "INVALID_INFORMATION_ERRMSG", "Invalid parameter values: ")
    monkeypatch.setattr(signatures, "NOT_FOUND", "NOT_FOUND")


class ShowParams(BaseModel):
    group_name: str = Field(description="name of the group")
    filter: Optional[str] = Field(default=None, description="narrowing text")
    start: Optional[int] = Field(default=None, description="first row to show")
    limit: int = Field(default=20, description="rows to show")


class TagParams(BaseModel):
    item_uid: str = Field(description="uid of the item")
    tag: str = Field(description="tag to add")


class BoundedParams(BaseModel):
    rank: int = Field(description="rank", ge=1, le=10)
    utc_offset: Optional[int] = Field(default=None, description="hours", ge=-12, le=14)


class CountParams(BaseModel):
    count: int = Field(description="how many")


class ListParams(BaseModel):
    names: List[str] = Field(description="names")


def _validate(params, command="cmd", context=None, diagnostics=None, hook_class=None):
    workflow = SimpleNamespace(context=context if context is not None else {"run_as_agent": True})
    extraction = InputForParamExtraction(
        command=command, input_for_param_extraction_class=hook_class)
    return extraction.validate_parameters(
        workflow, "Ctx/cmd", params, diagnostics=diagnostics)


# --- quotes ---------------------------------------------------------------------

@pytest.mark.parametrize("value, expected", [
    ('"Ann Lee"', "Ann Lee"),
    ("'Ann Lee'", "Ann Lee"),
    ("“Ann Lee”", "Ann Lee"),
    ("‘Payroll’", "Payroll"),
    ("`Billing`", "Billing"),
    ("«Billing»", "Billing"),
    ('  " Ann Lee "  ', "Ann Lee"),
    ("'\"Ann Lee\"'", "Ann Lee"),
])
def test_a_wholly_quoted_value_is_unwrapped(value, expected):
    assert unwrap_quoted_value(value) == (True, expected)


@pytest.mark.parametrize("value", [
    "Ann Lee",
    '"a" or "b"',
    "O'Brien",
    "'O'Brien'",
    '""',
    '"Ann Lee',
    "“Ann Lee\"",
    None,
    42,
])
def test_values_not_wholly_in_one_quote_pair_are_left_alone(value):
    assert unwrap_quoted_value(value) == (False, value)


def test_quotes_the_request_writes_inside_quotes_are_kept():
    request = "show rows whose note is '\"urgent\"' exactly"
    assert unwrap_quoted_value('"urgent"', request) == (False, '"urgent"')
    # Quotes that only set the value off in the request are not part of it.
    assert unwrap_quoted_value('"Ann Lee"', 'filter by "Ann Lee"') == (True, "Ann Lee")


def test_list_items_are_unwrapped_one_by_one():
    assert unwrap_quoted_value(['"Payroll"', "Billing"]) == (True, ["Payroll", "Billing"])


def test_validation_unwraps_quotes_and_records_them():
    seen = {}

    class Hook:
        @staticmethod
        def validate_extracted_parameters(workflow, command, cmd_parameters):
            seen["filter"] = cmd_parameters.filter
            return True, ""

    params = ShowParams.model_construct(
        group_name="Payroll", filter='"Ann Lee"', start=None, limit=20)
    diagnostics = {}
    is_valid, _, _, fields = _validate(
        params, command='show Payroll members filter "Ann Lee"',
        diagnostics=diagnostics, hook_class=Hook)

    assert is_valid and fields == []
    assert params.filter == "Ann Lee"
    assert seen == {"filter": "Ann Lee"}
    assert diagnostics["unwrapped_quoted_values"] == {"filter": '"Ann Lee"'}


def test_quotes_inside_brackets_are_unwrapped_too():
    params = TagParams.model_construct(item_uid='<"9a1b2c3d">', tag="Billing")
    diagnostics = {}
    is_valid, _, _, _ = _validate(params, command="tag 9a1b2c3d Billing", diagnostics=diagnostics)
    assert is_valid
    assert params.item_uid == "9a1b2c3d"
    assert diagnostics["unwrapped_bracketed_values"] == {"item_uid": '<"9a1b2c3d">'}
    assert diagnostics["unwrapped_quoted_values"] == {"item_uid": '<"9a1b2c3d">'}


def test_list_field_items_lose_their_quotes():
    params = ListParams.model_construct(names=['"Payroll"', "'Billing'"])
    is_valid, _, _, _ = _validate(params)
    assert is_valid
    assert params.names == ["Payroll", "Billing"]


# --- placeholders without brackets ------------------------------------------------

@pytest.mark.parametrize("value, field", [
    ("desired-tag", "tag"),
    ("your_item_uid", "item_uid"),
    ("new-name", "group_name"),
    ("the-tag", "tag"),
    ("some-value", "tag"),
    ("uid", "item_uid"),
    ("tag", "tag"),
])
def test_placeholder_shaped_values_the_request_never_writes(value, field):
    assert is_ungrounded_placeholder(value, field, "label the item 9a1b2c3d")


@pytest.mark.parametrize("value, field, request_text", [
    ("desired-tag", "tag", "add the tag desired-tag to it"),
    ("New-Tag", "tag", "add a tag"),           # mixed case: a name, not a variable
    ("new-hires", "tag", "add a tag"),         # qualifier, but no field word
    ("on_leave", "tag", "add a tag"),
    ("billing", "tag", "add a tag"),
    ("desired-tag", "tag", None),              # nothing to ground against
    ("tag2", "tag", "add a tag"),
])
def test_values_that_are_not_ungrounded_placeholders(value, field, request_text):
    assert not is_ungrounded_placeholder(value, field, request_text)


def test_agent_mode_grounds_against_the_users_request():
    # The agent's own command always contains the value it wrote, so the
    # user's request is what a placeholder is checked against.
    context = {"run_as_agent": True, "raw_user_message": "tag item 9a1b2c3d as Billing"}
    params = TagParams.model_construct(item_uid="9a1b2c3d", tag="desired-tag")
    diagnostics = {}
    is_valid, message, _, fields = _validate(
        params, command="add_tag item_uid=9a1b2c3d tag=desired-tag",
        context=context, diagnostics=diagnostics)

    assert not is_valid and fields == ["tag"]
    assert "tag 'desired-tag' is a placeholder, not a value" in message
    assert diagnostics["placeholder_values"] == {"tag": "desired-tag"}


def test_a_placeholder_shaped_value_the_user_wrote_is_kept():
    context = {"run_as_agent": True, "raw_user_message": "tag item 9a1b2c3d with new-tag"}
    params = TagParams.model_construct(item_uid="9a1b2c3d", tag="new-tag")
    is_valid, _, _, _ = _validate(
        params, command="add_tag item_uid=9a1b2c3d tag=new-tag", context=context)
    assert is_valid and params.tag == "new-tag"


def test_assistant_mode_grounds_against_the_command_text():
    params = TagParams.model_construct(item_uid="9a1b2c3d", tag="desired-tag")
    is_valid, _, _, fields = _validate(
        params, command="tag 9a1b2c3d as Billing", context={})
    assert not is_valid and fields == ["tag"]


# --- numbers ----------------------------------------------------------------------

@pytest.mark.parametrize("value, expected", [
    (INVALID_INT_VALUE, True),
    (-(2**63), True),
    (2**31 - 1, True),
    (-(2**31), True),
    (2**53 + 1, True),
    (float("inf"), True),
    (float("nan"), True),
    (0, False),
    (-5, False),
    (1_700_000_000_000, False),   # a millisecond timestamp is a real number
    (2.5, False),
])
def test_sentinel_numbers(value, expected):
    assert is_sentinel_number(value) is expected


def test_the_not_given_marker_on_an_optional_int_becomes_none():
    seen = {}

    class Hook:
        @staticmethod
        def validate_extracted_parameters(workflow, command, cmd_parameters):
            seen["start"] = cmd_parameters.start
            return True, ""

    params = ShowParams.model_construct(
        group_name="Payroll", filter=None, start=INVALID_INT_VALUE, limit=20)
    diagnostics = {}
    is_valid, _, _, fields = _validate(params, diagnostics=diagnostics, hook_class=Hook)

    assert is_valid and fields == []
    assert params.start is None
    assert seen == {"start": None}
    # The framework's own marker is routine and not recorded.
    assert "defaulted_sentinel_values" not in diagnostics


def test_the_not_given_marker_on_an_int_with_a_default_becomes_the_default():
    params = ShowParams.model_construct(
        group_name="Payroll", filter=None, start=None, limit=str(INVALID_INT_VALUE))
    is_valid, _, _, _ = _validate(params)
    assert is_valid and params.limit == 20


def test_a_made_up_extreme_is_treated_as_not_given_and_recorded():
    params = ShowParams.model_construct(
        group_name="Payroll", filter=None, start=2**63 - 1, limit=20)
    diagnostics = {}
    is_valid, _, _, _ = _validate(params, diagnostics=diagnostics)
    assert is_valid and params.start is None
    assert diagnostics["defaulted_sentinel_values"] == {"start": str(2**63 - 1)}


def test_an_extreme_for_a_field_without_a_default_is_a_parameter_error():
    params = CountParams.model_construct(count=2**31 - 1)
    diagnostics = {}
    is_valid, message, _, fields = _validate(params, diagnostics=diagnostics)
    assert not is_valid and fields == ["count"]
    assert "count 2147483647 is out of range (not a real value)" in message
    assert diagnostics["out_of_range_values"] == {"count": "2147483647"}


def test_the_not_given_marker_on_a_required_int_is_still_missing():
    is_valid, message, _, fields = _validate(CountParams.model_construct(count=INVALID_INT_VALUE))
    assert not is_valid and fields == ["count"]
    assert "Missing parameter values: count" in message


def test_a_negative_position_or_count_is_refused():
    params = ShowParams.model_construct(group_name="Payroll", filter=None, start=-3, limit=20)
    is_valid, message, _, fields = _validate(params)
    assert not is_valid and fields == ["start"]
    assert "start -3 is out of range (must not be negative)" in message


def test_declared_bounds_are_enforced():
    is_valid, message, _, fields = _validate(BoundedParams.model_construct(rank="11", utc_offset=None))
    assert not is_valid and fields == ["rank"]
    assert "rank 11 is out of range (must be at most 10)" in message

    params = BoundedParams.model_construct(rank=3, utc_offset=-5)
    is_valid, _, _, _ = _validate(params)
    assert is_valid and params.utc_offset == -5   # a declared ge= allows negatives


def test_in_range_numbers_validate_as_before():
    params = ShowParams.model_construct(group_name="Payroll", filter=None, start="10", limit=5)
    is_valid, message, _, _ = _validate(params)
    assert is_valid and message == "All required parameters are valid."
    assert params.start == 10 and params.limit == 5

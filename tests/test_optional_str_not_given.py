"""An Optional str the request does not give reaches the command as its default (fix-3t6p).

The extraction signature tells the LLM to answer ``NOT_FOUND`` (or ``None``) for a
value the request does not mention. For a required field ``NOT_FOUND`` means "ask
the user", and that must not change. For an ``Optional[str]`` field the text used
to reach the command as the value, while the key=value and XML paths give the
field's default. The str analogue of fix-7t0q (INVALID_INT on an Optional int).

No LLM: the extractor is a fake, the validator is the real one.
"""

from __future__ import annotations

from enum import Enum
from types import SimpleNamespace
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from fastworkflow import ModuleType
from fastworkflow._workflows.command_metadata_extraction.parameter_extraction import (
    NOT_FOUND,
    ParameterExtraction,
)
from fastworkflow.utils import signatures
from fastworkflow.utils.signatures import InputForParamExtraction

COMMAND = "People/find_person"


@pytest.fixture(autouse=True)
def _error_messages(monkeypatch):
    monkeypatch.setattr(signatures, "MISSING_INFORMATION_ERRMSG", "Missing parameter values: ")
    monkeypatch.setattr(signatures, "INVALID_INFORMATION_ERRMSG", "Invalid parameter values: ")
    monkeypatch.setattr(signatures, "NOT_FOUND", NOT_FOUND)


class _Color(str, Enum):
    NONE = "None"
    RED = "red"


class _Params(BaseModel):
    name: str = Field(description="person's name")
    filter: Optional[str] = Field(default=None, description="narrowing text")
    sort: Optional[str] = Field(default="label", description="sort column")
    tags: Optional[List[str]] = Field(default=None)
    color: Optional[_Color] = Field(default=None)


def _validate(**values):
    params = _Params.model_construct(**{"name": "Bob", **values})
    ok, message, _, fields = InputForParamExtraction(command="find Bob").validate_parameters(
        SimpleNamespace(context={}), COMMAND, params)
    return ok, message, fields, params


@pytest.mark.parametrize("marker", [NOT_FOUND, "None", " None "])
def test_an_unfilled_optional_str_takes_none(marker):
    ok, message, _, params = _validate(filter=marker)
    assert ok, message
    assert params.filter is None


@pytest.mark.parametrize("marker", [NOT_FOUND, "None"])
def test_an_unfilled_optional_str_takes_its_declared_default(marker):
    ok, message, _, params = _validate(sort=marker)
    assert ok, message
    assert params.sort == "label"


def test_a_real_optional_value_is_kept():
    ok, _, _, params = _validate(filter="Nonesuch", sort="NOT_FOUNDATION")
    assert ok
    assert (params.filter, params.sort) == ("Nonesuch", "NOT_FOUNDATION")


def test_a_required_str_holding_not_found_is_still_missing():
    ok, message, fields, params = _validate(name=NOT_FOUND, filter=NOT_FOUND)
    assert not ok
    assert fields == ["name"]
    assert "name" in message
    assert params.name == NOT_FOUND
    assert params.filter is None


def test_a_required_str_holding_none_text_is_left_alone():
    ok, _, _, params = _validate(name="None")
    assert ok
    assert params.name == "None"


def test_non_str_optional_fields_are_left_to_their_own_rules():
    # An enum member whose value is "None" is a real value, not the marker.
    ok, _, _, params = _validate(color=_Color.NONE)
    assert ok
    assert params.color is _Color.NONE


# ---------------------------------------------------------------------------
# End to end through the (fake) LLM path
# ---------------------------------------------------------------------------

@pytest.fixture
def run_round():
    state = {"cls": None, "llm": None}
    definition = MagicMock()
    definition.get_command_class.side_effect = (
        lambda name, kind: state["cls"] if kind == ModuleType.COMMAND_PARAMETERS_CLASS else None)

    def fake_llm(self, cls, command_name, folder):
        assert state["llm"] is not None, "the LLM extractor ran"
        return cls.model_construct(**state["llm"])

    def run(cls, text, cme, llm_values=None):
        state["cls"], state["llm"] = cls, llm_values
        app = MagicMock()
        app.context = {}
        extractor = ParameterExtraction(cme, app, COMMAND, text)
        diagnostics: dict = {}
        return extractor._extract_impl(diagnostics), diagnostics

    with patch("fastworkflow.RoutingRegistry") as registry, \
            patch.object(InputForParamExtraction, "extract_parameters", fake_llm):
        registry.get_definition.return_value = definition
        yield run


class _FilterThenName(BaseModel):
    filter: Optional[str] = Field(default=None, description="narrowing text")
    name: str = Field(description="person's name")


@pytest.mark.parametrize("marker", [NOT_FOUND, "None"])
def test_the_llm_path_gives_an_unmentioned_optional_str_its_default(run_round, marker):
    cme = MagicMock()
    cme.context = {}
    output, diagnostics = run_round(_FilterThenName, "find Bob", cme,
                                    llm_values={"filter": marker, "name": "Bob"})
    assert diagnostics["extraction_method"] == "llm"
    assert output.parameters_are_valid, output.error_msg
    assert (output.cmd_parameters.filter, output.cmd_parameters.name) == (None, "Bob")


def test_the_reply_to_the_clarification_still_goes_to_the_required_field(run_round):
    # Both markers from the LLM: the Optional one is not asked for, the
    # required one is, and the reply fills it (with fix-mbkw).
    cme = MagicMock()
    cme.context = {}
    first, _ = run_round(_FilterThenName, "find a person", cme,
                         llm_values={"filter": NOT_FOUND, "name": NOT_FOUND})
    assert not first.parameters_are_valid
    assert cme.context["stored_parameters"].filter is None

    second, diagnostics = run_round(_FilterThenName, "Ann", cme)
    assert diagnostics["extraction_method"] == "stored_merge"
    assert second.parameters_are_valid, second.error_msg
    assert (second.cmd_parameters.filter, second.cmd_parameters.name) == (None, "Ann")

"""A reply to a parameter clarification fills the field it was asked for (fix-mbkw).

When extraction comes up short, the missing or invalid fields are stored as
``NOT_FOUND`` and the user is asked for them. The reply is merged into the
stored parameters by ``_extract_and_merge_missing_parameters``. An Optional
field left at ``None`` was not asked for, so the reply must not land in it ahead
of the required field that was: "Ann" for a missing ``name`` used to fill an
Optional ``filter`` declared before it, and ``name`` was asked for again.

No LLM: the extractor is a fake, the validator is the real one.
"""

from __future__ import annotations

from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from fastworkflow import ModuleType
from fastworkflow._workflows.command_metadata_extraction import parameter_extraction as pe_module
from fastworkflow._workflows.command_metadata_extraction.parameter_extraction import (
    NOT_FOUND,
    ParameterExtraction,
)
from fastworkflow.utils import signatures
from fastworkflow.utils.signatures import INVALID_INT_VALUE, InputForParamExtraction

COMMAND = "People/find_person"


@pytest.fixture(autouse=True)
def _error_messages(monkeypatch):
    monkeypatch.setattr(signatures, "MISSING_INFORMATION_ERRMSG", "Missing parameter values: ")
    monkeypatch.setattr(signatures, "INVALID_INFORMATION_ERRMSG", "Invalid parameter values: ")
    monkeypatch.setattr(signatures, "NOT_FOUND", NOT_FOUND)


class _FilterThenName(BaseModel):
    filter: Optional[str] = Field(default=None, description="narrowing text")
    name: str = Field(description="person's name")


class _FilterThenTwo(BaseModel):
    filter: Optional[str] = Field(default=None, description="narrowing text")
    name: str = Field(description="person's name")
    dept: str = Field(description="department")


class _MixedKinds(BaseModel):
    tags: Optional[List[str]] = Field(default=None)
    start: Optional[int] = Field(default=None)
    count: int = Field(description="how many")


def _merge(stored, reply):
    return ParameterExtraction._extract_and_merge_missing_parameters(stored, reply)


# ---------------------------------------------------------------------------
# The merge itself
# ---------------------------------------------------------------------------

def test_reply_fills_the_required_field_not_an_earlier_optional_one():
    stored = _FilterThenName.model_construct(filter=None, name=NOT_FOUND)
    merged = _merge(stored, "Ann")
    assert (merged.filter, merged.name) == (None, "Ann")


def test_reply_fills_a_required_int_not_earlier_optional_fields():
    stored = _MixedKinds.model_construct(tags=None, start=None, count=INVALID_INT_VALUE)
    merged = _merge(stored, "3")
    assert (merged.tags, merged.start, merged.count) == (None, None, "3")


def test_comma_reply_with_one_part_per_asked_field_fills_them():
    stored = _FilterThenTwo.model_construct(filter=None, name=NOT_FOUND, dept=NOT_FOUND)
    merged = _merge(stored, "Ann, Sales")
    assert (merged.filter, merged.name, merged.dept) == (None, "Ann", "Sales")


def test_an_invalid_optional_field_stored_as_not_found_is_still_asked():
    # _extract_impl stores an invalid Optional field as NOT_FOUND: it was asked.
    stored = _FilterThenName.model_construct(filter=NOT_FOUND, name="Bob")
    merged = _merge(stored, "Ann")
    assert (merged.filter, merged.name) == ("Ann", "Bob")


def test_with_nothing_asked_an_unset_optional_field_still_takes_the_reply():
    # Unchanged behaviour: no asked field, so the first unset field is filled.
    stored = _FilterThenName.model_construct(filter=None, name="Bob")
    merged = _merge(stored, "Ann")
    assert (merged.filter, merged.name) == ("Ann", "Bob")


def test_only_required_fields_unset_keeps_declaration_order():
    # Unchanged behaviour when no Optional field is unset.
    stored = _FilterThenTwo.model_construct(filter="x", name=NOT_FOUND, dept=NOT_FOUND)
    assert _merge(stored, "Ann").model_dump() == {"filter": "x", "name": "Ann", "dept": NOT_FOUND}
    assert _merge(stored, "Ann, Sales").model_dump() == {"filter": "x", "name": "Ann", "dept": "Sales"}


# ---------------------------------------------------------------------------
# End to end: first round through the (fake) LLM, second round is the reply
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
        assert pe_module.InputForParamExtraction is InputForParamExtraction
        yield run


def test_clarification_round_binds_the_reply_to_the_missing_required_field(run_round):
    cme = MagicMock()
    cme.context = {}
    first, diagnostics = run_round(_FilterThenName, "find a person", cme,
                                   llm_values={"filter": None, "name": NOT_FOUND})
    assert not first.parameters_are_valid
    assert diagnostics["extraction_method"] == "llm"
    assert "name" in first.error_msg

    second, diagnostics = run_round(_FilterThenName, "Ann", cme)
    assert diagnostics["extraction_method"] == "stored_merge"
    assert second.parameters_are_valid, second.error_msg
    assert (second.cmd_parameters.filter, second.cmd_parameters.name) == (None, "Ann")

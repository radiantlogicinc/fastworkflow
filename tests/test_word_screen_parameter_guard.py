"""A free-text value that is only a pronoun, or only a kind word or field name of the workflow, is
treated as not given.

The extractor writes a word that names the kind of record (``book``), a field name
(``title``, ``author_name``) or a pronoun (``Her``) into a free-text input. The screen refuses
such a value so the normal missing message is produced, with wording that says why. A value that
merely contains such a word passes, and a declared choice (Literal or Enum) is never screened.

Integration: the real ``planning_tutorial_workflow`` (its vocabulary is derived in memory, since no
``workflow_vocabulary.json`` is stored there) and the real ``validate_parameters``. No mocks.
"""

from __future__ import annotations

import uuid
from enum import Enum
from pathlib import Path
from typing import Literal, Optional

import pytest
from pydantic import BaseModel, Field

import fastworkflow
from fastworkflow.utils import signatures
from fastworkflow.utils.signatures import InputForParamExtraction

WORKFLOW_PATH = str(Path(__file__).parent.joinpath("planning_tutorial_workflow").resolve())


@pytest.fixture(autouse=True)
def _error_messages(monkeypatch):
    monkeypatch.setattr(signatures, "MISSING_INFORMATION_ERRMSG", "Missing parameter values: ")
    monkeypatch.setattr(signatures, "INVALID_INFORMATION_ERRMSG", "Invalid parameter values: ")
    monkeypatch.setattr(signatures, "NOT_FOUND", "NOT_FOUND")


@pytest.fixture
def workflow(tmp_path):
    fastworkflow.init({"FASTWORKFLOW_STATE_ROOT": str(tmp_path / "workflow_contexts")})
    fastworkflow.RoutingRegistry.clear_registry()
    fastworkflow.RoutingRegistry.get_definition(WORKFLOW_PATH, load_cached=False)
    app_workflow = fastworkflow.Workflow.create(
        WORKFLOW_PATH, workflow_id_str=f"wsp_{uuid.uuid4().hex[:8]}")
    yield app_workflow
    fastworkflow.RoutingRegistry.clear_registry()


class TitleParams(BaseModel):
    title: str = Field(description="the title of the book")


class OptionalTitleParams(BaseModel):
    title: Optional[str] = Field(default=None, description="the title of the book")


class AuthorNameParams(BaseModel):
    author_name: str = Field(description="the author")


class QueryParams(BaseModel):
    query: str = Field(description="free text to search for")


class BookChoice(BaseModel):
    kind: Literal["book", "member"] = Field(description="the kind of record")


class BookKind(str, Enum):
    BOOK = "book"
    MEMBER = "member"


class EnumChoice(BaseModel):
    kind: BookKind = Field(description="the kind of record")


def _validate(app_workflow, params, subject="show_book", diagnostics=None):
    extraction = InputForParamExtraction(
        command="show the book", input_for_param_extraction_class=None)
    return extraction.validate_parameters(app_workflow, subject, params, diagnostics=diagnostics)


# --- a kind word of the workflow -------------------------------------------------------

@pytest.mark.parametrize("value", ["book", "Book", "BOOK", "member", "branch"])
def test_a_kind_word_of_the_workflow_is_refused_as_a_required_value(workflow, value):
    diagnostics = {}
    is_valid, message, _, fields = _validate(
        workflow, TitleParams(title=value), diagnostics=diagnostics)

    assert not is_valid and fields == ["title"], value
    assert "Missing parameter values: title" in message
    assert f"title {value!r} is a kind word of this workflow, not a value" in message
    assert diagnostics["word_screen_values"] == {"title": value}


def test_a_field_name_of_the_workflow_is_refused(workflow):
    is_valid, message, _, fields = _validate(workflow, AuthorNameParams(author_name="title"))

    assert not is_valid and fields == ["author_name"]
    assert "author_name 'title' is a kind word of this workflow, not a value" in message


def test_a_field_name_is_refused_in_any_case(workflow):
    is_valid, _, _, fields = _validate(workflow, AuthorNameParams(author_name="Title"))
    assert not is_valid and fields == ["author_name"]

    is_valid, _, _, fields = _validate(workflow, TitleParams(title="TITLE"))
    assert not is_valid and fields == ["title"]


def test_an_optional_field_holding_a_kind_word_takes_its_default(workflow):
    params = OptionalTitleParams(title="book")
    diagnostics = {}
    is_valid, _, _, _ = _validate(workflow, params, diagnostics=diagnostics)

    assert is_valid
    assert params.title is None
    assert diagnostics["word_screen_values"] == {"title": "book"}


def test_a_value_that_merely_contains_a_kind_word_passes(workflow):
    for value in ("Book Club", "The Book of Tea", "book-keeping ledger"):
        params = QueryParams(query=value)
        is_valid, _, _, _ = _validate(workflow, params)
        assert is_valid and params.query == value, value


# --- a pronoun ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", ["Her", "that", "them", "that one", "it"])
def test_a_pronoun_is_refused_as_a_value(workflow, value):
    diagnostics = {}
    is_valid, message, _, fields = _validate(
        workflow, QueryParams(query=value), subject="find_member", diagnostics=diagnostics)

    assert not is_valid and fields == ["query"], value
    assert f"query {value!r} is a pronoun, not a value" in message
    assert diagnostics["word_screen_values"] == {"query": value}


def test_a_pronoun_inside_a_name_passes(workflow):
    params = QueryParams(query="Her Majesty")
    is_valid, _, _, _ = _validate(workflow, params)
    assert is_valid and params.query == "Her Majesty"


# --- declared choices are never screened ----------------------------------------------

def test_a_declared_literal_choice_that_is_a_kind_word_is_never_refused(workflow):
    params = BookChoice(kind="book")
    is_valid, _, _, fields = _validate(workflow, params)
    assert is_valid and fields == [] and params.kind == "book"


def test_a_declared_enum_choice_that_is_a_kind_word_is_never_refused(workflow):
    is_valid, _, _, fields = _validate(workflow, EnumChoice(kind="book"))
    assert is_valid and fields == []


# --- an ordinary value ------------------------------------------------------------------

def test_an_ordinary_value_is_kept(workflow):
    params = TitleParams(title="Moby Dick")
    is_valid, _, _, _ = _validate(workflow, params)
    assert is_valid and params.title == "Moby Dick"

"""A name=value binding checks each field's own validators, but never an Input's model validators.

``_extract_parameters_from_key_values`` binds ``name=value`` text in code. A value that fails its field
(type, ``Field`` constraint, or a ``field_validator``) is no binding, and the LLM extractor takes over.
A model validator belongs to the validation step, which runs with the command context and refuses with
the field it names. So a model validator's refusal must not make the binding None.

Real components only: the parameter classes below are plain pydantic models, and case (c) uses the real
``tests/one_hook_workflow`` command ``pick_seat``.
"""

from __future__ import annotations

import datetime
import uuid
from pathlib import Path

import pytest
from pydantic import BaseModel, Field, field_validator, model_validator

import fastworkflow
from fastworkflow import ModuleType
from fastworkflow._workflows.command_metadata_extraction.parameter_extraction import ParameterExtraction
from fastworkflow.utils.signatures import InputForParamExtraction

ONE_HOOK = Path(__file__).parent / "one_hook_workflow"


class _IsoDateParams(BaseModel):
    date: str = Field(default="", description="the date, ISO yyyy-mm-dd")

    @field_validator("date")
    @classmethod
    def _must_be_iso(cls, value: str) -> str:
        if value and not (len(value) == 10 and value[4] == "-" and value[7] == "-"):
            raise ValueError("date must be yyyy-mm-dd")
        return value


class _CalendarDateParams(BaseModel):
    date: datetime.date = Field(description="the calendar date")


def test_a_str_field_validator_refusal_makes_the_binding_none_so_the_llm_extracts():
    """(a) name=value non-ISO date on a str field whose field_validator raises: no binding."""
    assert ParameterExtraction._extract_parameters_from_key_values(
        "book_trip date=3rd jan 2012", _IsoDateParams, "book_trip") is None


def test_a_str_field_validator_accepts_a_valid_value_in_the_binding():
    params = ParameterExtraction._extract_parameters_from_key_values(
        "book_trip date=2012-01-03", _IsoDateParams, "book_trip")
    assert params is not None
    assert params.date == "2012-01-03"


def test_a_value_that_fails_its_type_makes_the_binding_none():
    """(b) name=value non-date text on a datetime.date field: no binding."""
    assert ParameterExtraction._extract_parameters_from_key_values(
        "book_trip date=3rd jan 2012", _CalendarDateParams, "book_trip") is None



class _DefaultsReadingContext(BaseModel):
    limit: int = Field(default=10, description="rows to show")

    @model_validator(mode="after")
    def _reads_the_command_text(self, info):
        # Reads the validation context, as an Input's cross-field rule may. Built with no context
        # (as a bare binding builds its defaults) this raises TypeError.
        info.context["command_text"]
        return self


@pytest.fixture
def one_hook_path() -> str:
    return str(ONE_HOOK.resolve())


@pytest.fixture
def initialized_one_hook(tmp_path, one_hook_path: str):
    fastworkflow.init({"FASTWORKFLOW_STATE_ROOT": str(tmp_path / "workflow_contexts")})
    fastworkflow.RoutingRegistry.clear_registry()
    fastworkflow.RoutingRegistry.get_definition(one_hook_path, load_cached=False)
    yield
    fastworkflow.RoutingRegistry.clear_registry()


def test_a_model_validator_refusal_keeps_the_binding_and_the_refusal_names_its_field(
        initialized_one_hook, one_hook_path: str):
    """(c) pick_seat's model validator refuses seat B9 with field='seat'. The binding is kept, and
    validate_parameters (the validation step, with the command context) produces the refusal."""
    command_class = fastworkflow.RoutingRegistry.get_definition(one_hook_path).get_command_class(
        "pick_seat", ModuleType.COMMAND_PARAMETERS_CLASS)

    params = ParameterExtraction._extract_parameters_from_key_values(
        "pick_seat seat=B9 note=window", command_class, "pick_seat")
    assert params is not None, "a model validator must not turn a binding into None"
    assert params.seat == "B9"

    workflow = fastworkflow.Workflow.create(one_hook_path, workflow_id_str=f"bind_{uuid.uuid4().hex[:8]}")
    extractor = InputForParamExtraction.create(workflow, "pick_seat", "pick_seat seat=B9 note=window")
    diagnostics: dict = {}
    is_valid, message, _suggestions, _missing = extractor.validate_parameters(
        workflow, "pick_seat", params, diagnostics=diagnostics)

    assert not is_valid
    assert "seat B9 is taken" in message
    assert "seat" in diagnostics["validation_hook"]["refused_fields"]


def test_a_bare_command_binds_its_defaults_when_a_model_validator_reads_the_context():
    """A command sent with nothing after its name binds its defaults, even when an Input's model
    validator reads the validation context: a bare binding runs no model validator."""
    params = ParameterExtraction._extract_parameters_from_bare(
        "book_trip", _DefaultsReadingContext, "book_trip")
    assert params is not None
    assert params.limit == 10

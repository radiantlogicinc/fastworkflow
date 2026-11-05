"""Argument checks on an execution command's LLM parameter extraction (fix-qjf4).

The checks a planning command's typed call gets (``planning_commands.check_arguments``) also run on
the values the LLM extractor gives an execution command (``fastworkflow.utils.argument_checks``): a
``str`` value must be written in the request and takes the request's spelling, a placeholder is
refused, a value outside the field's declared allowed values is refused, and an ``int`` must be a
number the request writes. A refused value makes the extractor run once more with the refusal
appended to the command text; a value still refused is treated as not given (asked for when
required, the default when optional).

Integration: real workflows (``tests/planning_tutorial_workflow``, and the bundled retail example
for a field with ``json_schema_extra`` allowed values and a ``pattern`` field), the real routing,
``ParameterExtraction``, the real DSPy extractor and the real validator. The model boundary is a
scripted ``DummyLM`` that records each request, so the retry's note can be read where a live model
would read it. No test calls a model.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import dspy
import pytest
from dspy.utils import DummyLM
from pydantic import BaseModel, Field

import fastworkflow
from fastworkflow._workflows.command_metadata_extraction.parameter_extraction import ParameterExtraction
from fastworkflow.utils import argument_checks, signatures
from fastworkflow.utils.argument_checks import check_extracted_values
from fastworkflow.utils.signatures import NOT_FOUND

TUTORIAL = str(Path(__file__).parent.joinpath("planning_tutorial_workflow").resolve())
RETAIL = str(Path(__file__).parent.parent.joinpath("fastworkflow", "examples", "retail_workflow").resolve())
NOTE_START = "(A first extraction from this command was refused:"


class _ScriptedLM(DummyLM):
    """Answers the extractor's calls in order and records the text of each request."""

    def __init__(self, answers):
        super().__init__(answers, adapter=dspy.JSONAdapter())
        self.prompts: list[str] = []

    def forward(self, prompt=None, messages=None, **kwargs):
        self.prompts.append("\n".join(str(m.get("content", "")) for m in (messages or [])) or str(prompt))
        return super().forward(prompt=prompt, messages=messages, **kwargs)


@pytest.fixture(autouse=True)
def _framework(tmp_path, monkeypatch):
    fastworkflow.init({"FASTWORKFLOW_STATE_ROOT": str(tmp_path / "workflow_contexts")})
    fastworkflow.RoutingRegistry.clear_registry()
    # Jev's "is this field given" check is a model call: off, so the scripted LM answers alone.
    monkeypatch.setitem(fastworkflow._env_vars, "FW_JEV_FIELD_GIVEN", "off")
    monkeypatch.setattr(signatures, "LLM_PARAM_EXTRACTION", "scripted/extractor")
    yield
    fastworkflow.RoutingRegistry.clear_registry()


@pytest.fixture
def extract(monkeypatch):
    """Run one extraction round of *command_name* in *folder* (assistant mode: the LLM path) with the
    extractor answering *answers* in order. Returns (output, diagnostics, lm)."""
    def run(folder: str, command_name: str, text: str, *answers: dict):
        fastworkflow.RoutingRegistry.get_definition(folder, load_cached=False)
        workflow = fastworkflow.Workflow.create(folder, workflow_id_str=f"checks_{uuid.uuid4().hex[:8]}")
        # One answer per extraction: BestOfN stops at an answer that copies no description example.
        lm = _ScriptedLM(list(answers))
        monkeypatch.setattr(signatures.dspy_utils, "get_lm", lambda *a, **k: lm)
        diagnostics: dict = {}
        cme = SimpleNamespace(context={})
        output = ParameterExtraction(cme, workflow, command_name, text)._extract_impl(diagnostics)
        return output, diagnostics, lm
    return run


def _calls(lm: _ScriptedLM) -> int:
    return len(lm.prompts)


# ---------------------------------------------------------------- grounding (planning_tutorial_workflow)

def test_an_invented_str_value_is_refused_retried_once_and_then_asked_for(extract):
    output, diagnostics, lm = extract(
        TUTORIAL, "find_member", "find the member Ann Lee",
        {"reasoning": "name", "query": "Bob Jones"},
        {"reasoning": "name again", "query": "Bob Jones"})

    assert diagnostics["argument_check_refusals"] == ["query='Bob Jones' is not written in the request"]
    assert diagnostics["argument_check_retry"] is True
    assert _calls(lm) == 2
    # the second request carries the refusal, as typed_planning_call's REFUSED_CALL_NOTE does
    assert NOTE_START not in lm.prompts[0]
    assert NOTE_START in lm.prompts[1]
    assert "query='Bob Jones' is not written in the request" in lm.prompts[1]
    # still refused: treated as not given, so the required field is asked for (the NOT_FOUND path)
    assert output.parameters_are_valid is False
    assert output.cmd_parameters.query == NOT_FOUND
    assert "Missing parameter values: query" in output.error_msg
    assert "Values not used: query='Bob Jones' is not written in the request" in output.error_msg


def test_the_retry_fixes_a_refused_value(extract):
    output, diagnostics, lm = extract(
        TUTORIAL, "find_member", "find the member Ann Lee",
        {"reasoning": "name", "query": "Ann Leigh"},
        {"reasoning": "copied", "query": "Ann Lee"})

    assert diagnostics["argument_check_retry"] is True
    assert "argument_check_refusals_final" not in diagnostics
    assert _calls(lm) == 2
    assert output.parameters_are_valid is True
    assert output.cmd_parameters.query == "Ann Lee"


def test_a_grounded_value_takes_the_request_spelling_without_a_retry(extract):
    output, diagnostics, lm = extract(
        TUTORIAL, "find_member", "find the member Ann Lee",
        {"reasoning": "name", "query": "ann lee."})

    assert _calls(lm) == 1
    assert "argument_check_retry" not in diagnostics
    assert diagnostics["argument_check_respelled"] == {"query": "Ann Lee"}
    assert output.parameters_are_valid is True
    assert output.cmd_parameters.query == "Ann Lee"


def test_a_dash_variant_is_grounded(extract):
    output, _, lm = extract(
        TUTORIAL, "find_member", "find the member Ann Lee–Smith",
        {"reasoning": "name", "query": "Ann Lee-Smith"})

    assert _calls(lm) == 1
    assert output.cmd_parameters.query == "Ann Lee–Smith"


# ---------------------------------------------------------------- numbers (planning_tutorial_workflow)

@pytest.mark.parametrize("start", [None, signatures.INVALID_INT_VALUE])
def test_an_absent_optional_int_is_not_refused_when_the_request_writes_numbers(extract, start):
    """The planning rule refused an absent ``Optional[int] = None`` whenever the request wrote two or
    more numbers (22 of 974 Haiku calls in the replay); an absent value is not checked."""
    output, diagnostics, lm = extract(
        TUTORIAL, "list_loans", "list the loans of book b1 for the 2 or 3 members who asked",
        {"reasoning": "book only", "book_uid": "b1", "start": start})

    assert _calls(lm) == 1
    assert "argument_check_refusals" not in diagnostics
    assert output.parameters_are_valid is True
    assert output.cmd_parameters.book_uid == "b1"
    assert output.cmd_parameters.start is None


def test_an_int_the_request_does_not_write_is_refused_not_replaced(extract):
    """One number written (10) and another extracted (20): refused, never silently replaced by 10.
    Optional, so after the retry it takes its default."""
    output, diagnostics, lm = extract(
        TUTORIAL, "list_loans", "list the loans of book b1 from row 10",
        {"reasoning": "page", "book_uid": "b1", "start": 20},
        {"reasoning": "page", "book_uid": "b1", "start": 20})

    assert diagnostics["argument_check_refusals"] == ["start=20 is not a number the request writes (10)"]
    assert _calls(lm) == 2
    assert output.parameters_are_valid is True
    assert output.cmd_parameters.start is None


def test_an_int_the_request_writes_passes(extract):
    output, diagnostics, lm = extract(
        TUTORIAL, "list_loans", "list the loans of book b1 from row 10",
        {"reasoning": "page", "book_uid": "b1", "start": 10})

    assert _calls(lm) == 1
    assert "argument_check_refusals" not in diagnostics
    assert output.cmd_parameters.start == 10


# ---------------------------------------------------------------- allowed values, exempt field (retail)

def test_a_value_outside_json_schema_extra_allowed_values_is_refused(extract):
    """retail ``cancel_pending_order.reason`` declares its values only in ``json_schema_extra``; an
    invented reason copied from the request is refused (fix-1pfu.enum-in-extra) and, the field being
    optional, takes its default. ``order_id`` declares a pattern: a converted value (``#W123`` of
    "order W123") is exempt from grounding and left as the extractor wrote it."""
    output, diagnostics, lm = extract(
        RETAIL, "cancel_pending_order", "cancel order W123, the price dropped elsewhere",
        {"reasoning": "id and reason", "order_id": "#W123", "reason": "price dropped elsewhere"},
        {"reasoning": "id and reason", "order_id": "#W123", "reason": "price dropped elsewhere"})

    assert diagnostics["argument_check_refusals"] == [
        "reason='price dropped elsewhere' is not one of no longer needed, ordered by mistake"]
    assert _calls(lm) == 2
    assert "Valid values: 'no longer needed', 'ordered by mistake'" in lm.prompts[0]
    assert output.cmd_parameters.order_id == "#W123"
    assert output.cmd_parameters.reason == "ordered by mistake"


def test_an_allowed_value_takes_the_declared_spelling(extract):
    output, diagnostics, lm = extract(
        RETAIL, "cancel_pending_order", "cancel order #W123, no longer needed",
        {"reasoning": "id and reason", "order_id": "#W123", "reason": "No Longer Needed"})

    assert _calls(lm) == 1
    assert diagnostics["argument_check_respelled"] == {"reason": "no longer needed"}
    assert output.cmd_parameters.reason == "no longer needed"


def test_the_switch_turns_the_checks_off(extract, monkeypatch):
    monkeypatch.setitem(fastworkflow._env_vars, "FW_EXTRACTION_ARGUMENT_CHECKS", "off")
    output, diagnostics, lm = extract(
        TUTORIAL, "find_member", "find the member Ann Lee",
        {"reasoning": "name", "query": "Bob Jones"})

    assert _calls(lm) == 1
    assert "argument_check_refusals" not in diagnostics
    assert output.cmd_parameters.query == "Bob Jones"


# ---------------------------------------------------------------- the check itself, field by field

class _Fields(BaseModel):
    name: str = Field(description="a person's name")
    due: str = Field(description="the due date, YYYY-MM-DD", json_schema_extra={"in_request": False})
    kind: Optional[str] = Field(default=None, description="the kind of record. One of: book, member, branch")
    code: str = Field(default="", description="a code", examples=["XY-12"])
    count: int = Field(default=5, description="how many")
    start: Optional[int] = Field(default=None, description="first row")


def _check(request: str, **values):
    defaults = {"name": NOT_FOUND, "due": NOT_FOUND, "kind": None, "code": "", "count": 5, "start": None}
    return check_extracted_values(_Fields.model_construct(**{**defaults, **values}), request)


def test_the_in_request_false_marker_exempts_a_converted_value():
    check = _check("add Ann Lee, due next Monday", name="Ann Lee", due="2026-11-09")
    assert check.refused == {}


def test_an_exempt_field_still_refuses_a_placeholder():
    check = _check("add Ann Lee, due next Monday", name="Ann Lee", due="{due_date}")
    assert check.refused == {"due": ("{due_date}", "is a placeholder, not a value")}


def test_one_of_in_the_description_is_the_allowed_list():
    assert argument_checks.allowed_values(_Fields.model_fields["kind"]) == ("book", "member", "branch")
    assert _check("find the shelf Ann", name="Ann", kind="shelf").refused == {
        "kind": ("shelf", "is not one of book, member, branch")}
    assert _check("find the member Ann", name="Ann", kind="Member").respelled == {"kind": ("Member", "member")}


def test_a_copied_description_example_is_refused():
    check = _check("add Ann Lee", name="Ann Lee", code="XY-12")
    assert check.refused == {"code": ("XY-12", argument_checks.EXAMPLE_WHY)}
    assert _check("add Ann Lee with code XY-12", name="Ann Lee", code="XY-12").refused == {}


class _Converted(BaseModel):
    origin: str = Field(description="IATA code", examples=["JFK"], json_schema_extra={"in_request": False})
    date: str = Field(description="YYYY-MM-DD", examples=["2024-05-20"], pattern=r"^\d{4}-\d{2}-\d{2}$")


def test_a_converted_value_equal_to_a_description_example_is_not_refused():
    """tau-bench airline: "New York ... May 20" gives origin JFK and date 2024-05-20, both the fields' own
    examples. An exempt field's value is converted, so matching an example is no evidence it was copied."""
    params = _Converted.model_construct(origin="JFK", date="2024-05-20")
    assert check_extracted_values(params, "fly from New York to Seattle on May 20").refused == {}


def test_punctuation_alone_is_refused():
    assert _check("find the member Ann", name="...").refused == {"name": ("...", argument_checks.NO_VALUE_WHY)}


def test_values_not_given_are_not_checked():
    check = _check("find twenty two books", name="<UNKNOWN>", due="&lt;UNKNOWN&gt;", count=5, start=None)
    assert check.refused == {}


def test_number_words_count_as_written_numbers():
    assert _check("show eight rows", name=NOT_FOUND, count=8).refused == {}
    assert "count" in _check("show eight rows", name=NOT_FOUND, count=9).refused


def test_digits_inside_a_written_str_value_are_not_numbers_the_request_writes():
    """``Doyle and Sons 1``: the 1 belongs to the name, so the request writes no number for count."""
    assert _check("find Doyle and Sons 1", name="Doyle and Sons 1", count=7).refused == {}

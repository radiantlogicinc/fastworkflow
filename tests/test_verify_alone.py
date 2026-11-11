"""``verify: "alone"`` (fix-rmpn): a bool that only the verify classifier decides.

The typed planning call and the parameter extractor are not asked for the field; any value given for
it is ignored; a sure verdict decides it, and it keeps its default otherwise (also without
JEV_API_KEY). Each case runs the real loader (``option_params``), the real typed signature, the real
guards (``check_arguments``) and the real extractor signature; a verdict is the
``{field: (value, probability)}`` that ``check_verdicts`` returns. The live test asks the real
classifier and skips without a key.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import dotenv_values
from pydantic import BaseModel, Field

from fastworkflow import planning_commands as pc
from fastworkflow.utils.signatures import InputForParamExtraction

ROOT = Path(__file__).resolve().parent.parent


class ReportInput(BaseModel):
    title: str = Field(description="the book's title")
    loans: bool = Field(default=False, description="the request asks who has the book on loan",
                        json_schema_extra={"classifier_confirms": True})
    leave_out_author: bool = Field(
        default=False,
        description="the request says to leave out the book's author, or asks only for other parts",
        json_schema_extra={"classifier_decides": True})


def _command() -> pc.PlanningCommandCommand:
    return pc.option_command(ReportInput, "book_report")


def _param(name: str) -> pc.Param:
    return next(p for p in _command().params if p.name == name)


# ---------------------------------------------------------------- loading

def test_verify_alone_loads_as_a_verify_bool_the_classifier_alone_decides():
    alone, plain = _param("leave_out_author"), _param("loans")
    assert (alone.verify, alone.alone) == (True, True)
    assert (plain.verify, plain.alone) == (True, False)
    assert pc.classifier_alone_fields(ReportInput) == frozenset({"leave_out_author"})


def test_verify_alone_on_a_field_that_is_not_a_bool_is_a_load_error():
    class Bad(BaseModel):
        kind: str = Field(description="a kind", json_schema_extra={"classifier_decides": True})

    with pytest.raises(pc.PlanningCommandFormatError, match='option classifier_decides does not apply'):
        pc.option_params(Bad, "bad")


def test_another_verify_word_is_still_a_load_error():
    class Bad(BaseModel):
        flag: bool = Field(default=False, description="a flag", json_schema_extra={"verify": "only"})

    with pytest.raises(pc.PlanningCommandFormatError, match='verify: "only" is removed: write'):
        pc.option_params(Bad, "bad")


# ---------------------------------------------------------------- who is asked

def test_the_typed_planning_call_is_not_asked_for_a_verify_alone_field():
    fields = pc.typed_signature(_command()).output_fields   # the option fields (option_command)
    assert "loans" in fields
    assert "leave_out_author" not in fields


def test_the_parameter_extractor_is_not_asked_for_a_verify_alone_field():
    signature = InputForParamExtraction.create_signature_from_pydantic_model(
        ReportInput, pc.option_sentences(ReportInput), pc.classifier_alone_fields(ReportInput))
    assert "title" in signature.output_fields and "loans" in signature.output_fields
    assert "leave_out_author" not in signature.output_fields


# ---------------------------------------------------------------- the guards

REQUEST = "Tell me about Dune, but skip the author."


def test_a_value_given_for_a_verify_alone_field_is_ignored_without_a_verdict():
    # no classifier (no key), or one that is not sure: the default stands, whatever a model wrote
    for verdicts in ({}, {"leave_out_author": (True, 0.5)}):
        args, _notes = pc.check_arguments(_command(), {"loans": False, "leave_out_author": True},
                                          REQUEST, verdicts)
        assert args["leave_out_author"] is False, verdicts


def test_a_sure_verdict_decides_a_verify_alone_field():
    args, notes = pc.check_arguments(_command(), {"loans": False}, REQUEST, {"leave_out_author": (True, 0.97)})
    assert args["leave_out_author"] is True
    assert any("leave_out_author" in n and "check 0.97" in n for n in notes)


def test_a_plain_verify_field_keeps_the_planner_filled_value_until_a_sure_verdict():
    args, _ = pc.check_arguments(_command(), {"loans": True}, "who has Dune on loan", {})
    assert args["loans"] is True
    args, _ = pc.check_arguments(_command(), {"loans": True}, "who has Dune on loan", {"loans": (False, 0.9)})
    assert args["loans"] is False


def test_the_preview_marks_a_verify_alone_field():
    from fastworkflow import command_preview
    rows = command_preview._inputs_section(_command())
    assert any(r.startswith("| `leave_out_author` |") and "| decides |" in r for r in rows), rows
    assert any(r.startswith("| `loans` |") and "| yes |" in r for r in rows), rows


# ---------------------------------------------------------------- the real classifier

def _jev_key() -> str:
    settings = {}
    for name in ("fastworkflow/examples/fastworkflow.passwords.env", "passwords/.env"):
        settings.update({k: v for k, v in dotenv_values(ROOT / name).items() if v})
    return os.environ.get("JEV_API_KEY") or settings.get("JEV_API_KEY", "")


@pytest.mark.slow
@pytest.mark.requires_llm_key
@pytest.mark.skipif(not _jev_key(), reason="No JEV_API_KEY; the verify classifier cannot be asked.")
def test_the_classifier_decides_a_leave_out_part_from_the_request_alone():
    from dspy.experimental import TypeSafe
    lm = TypeSafe(api_key=_jev_key(), cache=False)
    command = _command()

    left_out = pc.check_verdicts(lm, command, REQUEST)
    whole = pc.check_verdicts(lm, command, "Tell me everything about Dune, its author included.")

    assert left_out["leave_out_author"][0] is True
    assert whole["leave_out_author"][0] is False

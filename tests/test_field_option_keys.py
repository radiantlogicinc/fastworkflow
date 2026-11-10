"""The author-facing field option keys of planning command and execution command Inputs (spec sections 2.0 to 2.7).

Each key is read by the real loader (``planning_commands.option_params`` and the resolver) and must give the
internal option its old spelling gave. Each removed spelling fails the load naming its replacement. No mocks:
the Inputs are real pydantic models, and the producer edge is resolved by the real resolver.
"""

from __future__ import annotations

import re
from typing import Literal, Optional

import pytest
from pydantic import BaseModel, Field

import fastworkflow
from fastworkflow import FieldOptions
from fastworkflow import planning_commands as pc
from fastworkflow.field_options import OPTION_KEYS, ConfirmsOptions, CopiedOptions, RecordNamedOptions
from fastworkflow.utils import argument_checks


def _params(model, planning=False):
    return {p.name: p for p in pc.option_params(model, "app/show", planning=planning)}


_DEFAULTS = {str: "", int: 0, bool: False, "list[str]": []}


def _field(annotation, extra, **kwargs):
    """A one-field Input model ``value`` of *annotation* with the Field options *extra* (a default that
    fits the type unless *kwargs* gives one; a Literal has none, so it is required)."""
    key = "list[str]" if annotation == list[str] else annotation
    if "default" not in kwargs and key in _DEFAULTS:
        kwargs["default"] = _DEFAULTS[key]
    return type("Input", (BaseModel,), {"__annotations__": {"value": annotation},
                                         "value": Field(json_schema_extra=extra, **kwargs)})


# ---------------------------------------------------------------- each key is the internal option its old spelling gave

@pytest.mark.parametrize("annotation, extra, expected", [
    # copied_from_command_text (old: in_request True / {"ignore": ...})
    (str, {"copied_from_command_text": True}, dict(in_request=True, subject=False, verify=False, ignore=())),
    (str, {"copied_from_command_text": {"ignore": ["the"]}},
     dict(in_request=True, subject=False, verify=False, ignore=("the",))),
    (int, {"copied_from_command_text": True}, dict(in_request=True, subject=False, verify=False)),
    (list[str], {"copied_from_command_text": True}, dict(in_request=True, subject=False, verify=False)),
    # record_named_in_command_text (old: in_request "subject" / subject + ignore)
    (str, {"record_named_in_command_text": True}, dict(in_request=True, subject=True, verify=False, ignore=())),
    (str, {"record_named_in_command_text": {"ignore": ["group", "unit"]}},
     dict(in_request=True, subject=True, verify=False, ignore=("group", "unit"))),
    # classifier_confirms (old: verify True / a meanings dict on an enum)
    (bool, {"classifier_confirms": True}, dict(verify=True, alone=False, in_request=False)),
    (Literal["a", "b"], {"classifier_confirms": {"meanings": {"a": "the first"}}},
     dict(verify=True, alone=False, means=(("a", "the first"),))),
    (str, {"record_named_in_command_text": True, "classifier_confirms": True},
     dict(in_request=True, subject=True, verify=True, alone=False)),
    # classifier_decides (old: verify "alone" on a bool)
    (bool, {"classifier_decides": True}, dict(verify=True, alone=True)),
])
def test_each_key_gives_the_internal_option_its_old_spelling_gave(annotation, extra, expected):
    model = _field(annotation, extra)
    param = _params(model, planning=True)["value"]
    for attr, value in expected.items():
        assert getattr(param, attr) == value, (attr, param)


def test_a_planning_command_field_with_no_check_option_is_not_a_param():
    model = _field(str, {"from_command_output": "find_book.uid"})
    assert pc.option_params(model, "app/show", planning=True) == ()


# ---------------------------------------------------------------- the same keys through a FieldOptions dict

def test_fieldoptions_is_a_plain_dict_the_loader_reads_the_same_as_a_dict_literal():
    literal = _field(str, {"copied_from_command_text": {"ignore": ["the"]}})
    typed = _field(str, FieldOptions(copied_from_command_text={"ignore": ["the"]}))
    assert type(FieldOptions(copied_from_command_text=True)) is dict
    assert _params(literal)["value"] == _params(typed)["value"]


def test_fieldoptions_names_the_six_keys_and_the_option_typeddicts_are_exported():
    assert set(FieldOptions.__annotations__) == set(OPTION_KEYS)
    assert {"ignore"} == set(CopiedOptions.__annotations__) == set(RecordNamedOptions.__annotations__)
    assert set(ConfirmsOptions.__annotations__) == {"meanings"}
    for name in ("FieldOptions", "CopiedOptions", "RecordNamedOptions", "ConfirmsOptions"):
        assert getattr(fastworkflow, name) is getattr(fastworkflow.planning, name)


# ---------------------------------------------------------------- the producer and positional keys

def test_command_text_after_command_name_is_read_as_the_positional_field_of_an_execution_command():
    from fastworkflow._workflows.command_metadata_extraction.parameter_extraction import ParameterExtraction

    class Input(BaseModel):
        note: Optional[str] = Field(default=None, json_schema_extra={"command_text_after_command_name": True})
        other: str = Field(default="", description="a plain field")

    assert ParameterExtraction._is_positional_field(Input.model_fields["note"])
    assert not ParameterExtraction._is_positional_field(Input.model_fields["other"])
    # the old spelling is no longer read as the marker
    old = Input.model_fields["note"]
    assert not ParameterExtraction._is_positional_field(
        type("F", (), {"json_schema_extra": {"positional": True}})())
    assert old.json_schema_extra == {"command_text_after_command_name": True}


def test_two_command_text_after_command_name_fields_in_one_input_fail_the_load():
    class Input(BaseModel):
        first: str = Field(default="", json_schema_extra={"command_text_after_command_name": True})
        second: str = Field(default="", json_schema_extra={"command_text_after_command_name": True})

    with pytest.raises(pc.PlanningCommandFormatError, match="one field per Input takes the command text"):
        pc.option_params(Input, "app/show")


def test_converted_from_command_text_takes_a_field_out_of_the_copy_check_and_is_read_by_argument_checks():
    class Input(BaseModel):
        due: str = Field(default="", description="the due date", json_schema_extra={"converted_from_command_text": True})
        name: str = Field(default="", description="a name", json_schema_extra={"copied_from_command_text": True})

    assert argument_checks.is_grounding_exempt(Input.model_fields["due"])
    assert not argument_checks.is_grounding_exempt(Input.model_fields["name"])
    # an execution command's Input: a converted field is not a planning option, so no param
    assert "due" not in _params(Input)


def _resolve(consumer_input_cls, producer_output_cls, *, consumer="Library/open_book"):
    """Resolve the producer edges of *consumer_input_cls* against a producer command ``Library/find_book``."""
    class Producer:
        """Find books."""

        class Input(BaseModel):
            query: str = Field(default="", description="the query")

        Output = producer_output_cls

    class Consumer:
        """Open a book."""

        Input = consumer_input_cls

    resolution = pc.FieldSourceResolution(
        workflow_folderpath="", problems=(), skipped_commands=frozenset(), model_to_command={},
        signatures={"Library/find_book": Producer, consumer: Consumer}, command_files={}, import_errors={},
        base_contexts={})
    problems = pc.resolve_signature_field_sources(consumer, Consumer, "open_book.py", resolution)
    return Consumer, problems


class _Output(BaseModel):
    book_uids: list[str] = Field(description="the uids of the matching books")


def test_from_command_output_writes_the_producer_edge_as_from_the_old_from_did():
    class Input(BaseModel):
        book_uid: str = Field(default="", description="the book", json_schema_extra={"from_command_output": "find_book.book_uids"})

    consumer, problems = _resolve(Input, _Output)
    assert problems == []
    assert list(consumer.Input.model_fields["book_uid"].json_schema_extra["available_from"]) == [
        "Library/find_book.book_uids"]


def test_from_command_output_may_be_a_list_of_sources():
    class Input(BaseModel):
        book_uid: str = Field(default="", json_schema_extra={"from_command_output": ["find_book.book_uids"]})

    consumer, problems = _resolve(Input, _Output)
    assert problems == []
    assert "Library/find_book.book_uids" in list(consumer.Input.model_fields["book_uid"].json_schema_extra["available_from"])


def test_from_command_output_on_an_output_field_is_a_load_problem_naming_the_key():
    class Output(BaseModel):
        author: str = Field(default="", json_schema_extra={"from_command_output": "find_book.book_uids"})

    class Consumer(BaseModel):
        book_uid: str = Field(default="")

    class Input(BaseModel):
        book_uid: str = Field(default="")

    class Command:
        """Open a book."""

    Command.Input = Input
    Command.Output = Output
    resolution = pc.FieldSourceResolution(
        workflow_folderpath="", problems=(), skipped_commands=frozenset(), model_to_command={},
        signatures={"Library/open_book": Command}, command_files={}, import_errors={}, base_contexts={})
    problems = pc.resolve_signature_field_sources("Library/open_book", Command, "open_book.py", resolution)
    assert any("cannot declare from_command_output" in p.message for p in problems), problems


def test_a_hand_written_available_from_is_still_a_load_problem():
    class Input(BaseModel):
        book_uid: str = Field(default="", json_schema_extra={"available_from": ["Library/find_book.book_uids"]})

    _, problems = _resolve(Input, _Output)
    assert any("available_from is written by from_command_output" in p.message for p in problems), problems


def test_from_as_a_key_on_an_input_is_a_load_error_naming_the_replacement():
    class Input(BaseModel):
        book_uid: str = Field(default="", json_schema_extra={"from": "find_book.book_uids"})

    with pytest.raises(pc.PlanningCommandFormatError, match=re.escape(
            'from: "find_book.book_uids" is removed: write "from_command_output": "find_book.book_uids"')):
        pc.option_params(Input, "app/show")


# ---------------------------------------------------------------- each removed spelling fails, naming its replacement

@pytest.mark.parametrize("annotation, extra, message", [
    (str, {"in_request": True}, '"copied_from_command_text": True'),
    (str, {"in_request": {"ignore": ["the"]}}, '"copied_from_command_text": {"ignore": ["the"]}'),
    (str, {"in_request": "subject"}, '"record_named_in_command_text": True'),
    (str, {"in_request": {"subject": True, "ignore": ["group"]}}, '"record_named_in_command_text": {"ignore": ["group"]}'),
    (bool, {"verify": True}, '"classifier_confirms": True'),
    (bool, {"verify": "alone"}, '"classifier_decides": True'),
    (Literal["a", "b"], {"verify": {"a": "x"}}, '"classifier_confirms": {"meanings": {"a": "x"}}'),
    (str, {"positional": True}, '"command_text_after_command_name": True'),
    (str, {"subject": True}, '"record_named_in_command_text": True'),
    (Literal["a", "b"], {"means": {"a": "x"}}, '"classifier_confirms": {"meanings": {"a": "x"}}'),
    (str, {"strip": ["group"]}, '"copied_from_command_text": {"ignore": ["group"]}'),
])
def test_a_removed_spelling_fails_the_load_naming_its_replacement(annotation, extra, message):
    model = _field(annotation, extra)
    with pytest.raises(pc.PlanningCommandFormatError) as caught:
        pc.option_params(model, "app/show", planning=True)
    assert message in str(caught.value)
    assert "is removed: write" in str(caught.value)


def test_the_load_error_for_in_request_subject_reads_as_the_spec_example():
    class Input(BaseModel):
        who: str = Field(default="", json_schema_extra={"in_request": "subject"})

    with pytest.raises(pc.PlanningCommandFormatError) as caught:
        pc.option_params(Input, "app/who", planning=True)
    assert str(caught.value) == ('app/who: field \'who\': in_request: "subject" is removed: write '
                                 '"record_named_in_command_text": True')


def test_a_removed_spelling_on_an_execution_command_fails_the_load_too():
    class Input(BaseModel):
        who: str = Field(default="", json_schema_extra={"in_request": True})

    with pytest.raises(pc.PlanningCommandFormatError, match='"copied_from_command_text": True'):
        pc.option_params(Input, "app/show")


# ---------------------------------------------------------------- unknown keys, bad values, conflicts, types

@pytest.mark.parametrize("extra, suggestion", [
    ({"copied_from_command_txt": True}, "did you mean 'copied_from_command_text'?"),
    ({"record_named_in_command_txt": True}, "did you mean 'record_named_in_command_text'?"),
    ({"classifier_confirm": True}, "did you mean 'classifier_confirms'?"),
])
def test_an_unknown_key_suggests_the_closest_option(extra, suggestion):
    class Input(BaseModel):
        name: str = Field(default="", json_schema_extra=extra)

    with pytest.raises(pc.PlanningCommandFormatError, match=re.escape(suggestion)):
        pc.option_params(Input, "app/show", planning=True)


def test_an_unknown_key_on_a_planning_command_fails_even_without_a_close_match():
    class Input(BaseModel):
        name: str = Field(default="", json_schema_extra={"colour": "red"})

    with pytest.raises(pc.PlanningCommandFormatError, match="unknown option 'colour'"):
        pc.option_params(Input, "app/show", planning=True)


def test_an_execution_command_may_carry_its_own_keys_but_not_a_misspelled_option():
    class Input(BaseModel):
        kind: str = Field(default="", json_schema_extra={"enum": ["book", "member"], "examples": ["book"]})

    assert pc.option_params(Input, "app/show") == ()
    class Bad(BaseModel):
        kind: str = Field(default="", json_schema_extra={"copied_from_command_txt": True})

    with pytest.raises(pc.PlanningCommandFormatError, match="did you mean 'copied_from_command_text'"):
        pc.option_params(Bad, "app/show")


@pytest.mark.parametrize("extra, message", [
    ({"copied_from_command_text": False}, "copied_from_command_text: False is not a value; leave the key out"),
    ({"record_named_in_command_text": False}, "record_named_in_command_text: False is not a value"),
    ({"classifier_decides": False}, "classifier_decides: False is not a value"),
    ({"copied_from_command_text": "yes"}, "copied_from_command_text must be True or a dict of ignore"),
    ({"copied_from_command_text": {}}, "copied_from_command_text is empty"),
    ({"copied_from_command_text": {"ignor": ["the"]}}, "did you mean 'ignore'?"),
    ({"copied_from_command_text": {"ignore": "the"}}, "copied_from_command_text ignore must be a list of words"),
    ({"classifier_confirms": {"meanings": {}}}, "classifier_confirms meanings must be a dict"),
    ({"classifier_decides": {"x": 1}}, "classifier_decides must be True"),
    ({"from_command_output": 3}, 'from_command_output must be "[Context/]command[.field]" text or a list of it'),
])
def test_a_false_or_bad_value_is_rejected(extra, message):
    model = _field(bool if "classifier_decides" in extra else str, extra)
    with pytest.raises(pc.PlanningCommandFormatError, match=re.escape(message)):
        pc.option_params(model, "app/show", planning=True)


@pytest.mark.parametrize("annotation, extra, message", [
    (str, {"copied_from_command_text": True, "record_named_in_command_text": True},
     "copied_from_command_text and record_named_in_command_text cannot both be on a field"),
    (bool, {"classifier_confirms": True, "classifier_decides": True},
     "classifier_confirms and classifier_decides cannot both be on a field"),
    (str, {"converted_from_command_text": True, "copied_from_command_text": True},
     "converted_from_command_text takes the field out of the copy check"),
    (str, {"classifier_confirms": True}, "option classifier_confirms does not apply to field 'value' of type str"),
])
def test_conflicting_or_incomplete_pairs_fail_the_load(annotation, extra, message):
    model = _field(annotation, extra)
    with pytest.raises(pc.PlanningCommandFormatError, match=re.escape(message)):
        pc.option_params(model, "app/show", planning=True)


@pytest.mark.parametrize("annotation, extra, message, planning", [
    (bool, {"copied_from_command_text": True},
     "option copied_from_command_text does not apply to field 'value' of type bool", True),
    (int, {"record_named_in_command_text": True},
     "option record_named_in_command_text does not apply to field 'value' of type int (it takes a str)", True),
    (str, {"classifier_decides": True},
     "option classifier_decides does not apply to field 'value' of type str (it takes a bool)", True),
    (int, {"command_text_after_command_name": True},
     "option command_text_after_command_name does not apply to field 'value' of type int (it takes a str)", False),
    (Literal["a", "b"], {"copied_from_command_text": True},
     "option copied_from_command_text does not apply to field 'value' of type enum (it takes a str, an int or a list[str])", True),
])
def test_a_key_on_a_type_it_does_not_apply_to_fails_naming_the_types(annotation, extra, message, planning):
    model = _field(annotation, extra)
    with pytest.raises(pc.PlanningCommandFormatError, match=re.escape(message)):
        pc.option_params(model, "app/show", planning=planning)


def test_a_planning_command_field_rejects_the_execution_only_keys():
    class Input(BaseModel):
        note: str = Field(default="", json_schema_extra={"command_text_after_command_name": True})

    with pytest.raises(pc.PlanningCommandFormatError, match="command_text_after_command_name is for an execution command"):
        pc.option_params(Input, "app/show", planning=True)


def test_meanings_keys_must_be_values_of_the_literal():
    model = _field(Literal["a", "b"], {"classifier_confirms": {"meanings": {"c": "x"}}})
    with pytest.raises(pc.PlanningCommandFormatError, match="classifier_confirms meanings: 'c' is not a value of it"):
        pc.option_params(model, "app/show", planning=True)


def test_the_new_keys_read_together_on_one_planning_input():
    class Input(BaseModel):
        who: str = Field(description="the record", json_schema_extra={"record_named_in_command_text": {"ignore": ["group"]}})
        kind: Literal["group", "unit"] = Field(description="what it is",
                                               json_schema_extra={"classifier_confirms": {"meanings": {"group": "a group"}}})
        verified: bool = Field(default=False, description="asked", json_schema_extra={"classifier_decides": True})

    params = _params(Input, planning=True)
    assert (params["who"].subject, params["who"].ignore) == (True, ("group",))
    assert (params["kind"].verify, params["kind"].means) == (True, (("group", "a group"),))
    assert (params["verified"].alone, params["verified"].verify) == (True, True)

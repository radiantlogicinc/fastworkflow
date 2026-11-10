"""The planning step helpers after the rename: ``run`` (7.1-7.4), paging (7.5), ``count_and_names_read``
(7.7), the removed spellings, and the Input validators of planning commands (section 3).

Integration style: real ``fastworkflow.planning`` helpers and the real planning command loader; no mocks.
"""
from __future__ import annotations

import types
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, Field, model_validator

from fastworkflow import planning, planning_commands
from fastworkflow.planning import (StepFormError, command_inputs_for_steps, count,
                                   count_and_names_read, row_id_field, run)
from fastworkflow.planning_phases import CODE_MAX_PAGES, Placeholder


class OpenBookInput(BaseModel):
    book_uid: str = Field(description="the book's uid")


class DeclaredIdInput(BaseModel):
    """The id is the field named by from_command_output, not the one required str field."""
    label: str = Field(description="a label")
    item_ref: str = Field(description="the item", json_schema_extra={"from_command_output": "find.refs"})


class TwoRequiredStrInput(BaseModel):
    first: str = Field(description="first")
    second: str = Field(description="second")


def _sigs(**inputs):
    return {name: SimpleNamespace(Input=model) for name, model in inputs.items()}


# 7.1 run(command, *, named=None, all_pages=False, **inputs)

def test_inputs_are_the_command_fields_and_none_is_left_out():
    assert run("open_book", book_uid="b1") == "open_book book_uid=b1"
    assert run("open_book", book_uid=None) == "open_book"
    assert run("list_loans", book_uid="b1", start=None) == "list_loans book_uid=b1"


def test_an_unknown_keyword_fails_the_load_as_before():
    with command_inputs_for_steps(_sigs(open_book=OpenBookInput)):
        with pytest.raises(StepFormError, match="passes title to 'open_book'"):
            run("open_book", title="Dune")


# 7.2 run(..., named=...) replaces pick

def test_named_with_a_concrete_id_writes_the_field_and_value():
    assert run("open_book", named="Dune", book_uid="b1") == "open_book book_uid=b1"


def test_named_without_an_id_is_the_match_wording_as_pick_wrote_it():
    # Byte-identical to the text pick() wrote (the step words are read by the model).
    assert run("open_book", named="Dune") == "open_book with the uid of the Dune match"
    assert run("open_book", named="Dune", book_uid=None) == "open_book with the uid of the Dune match"
    assert run("open_book", named="Dune", book_uid=planning.Placeholder("find.uids[0]")) == \
        "open_book with the uid of the Dune match"


def test_the_row_id_is_the_declared_field_then_the_one_required_str():
    assert row_id_field(DeclaredIdInput) == "item_ref"
    assert row_id_field(OpenBookInput) == "book_uid"
    assert row_id_field(TwoRequiredStrInput) is None


def test_a_non_uid_id_field_is_opened_by_name_too():
    with command_inputs_for_steps(_sigs(open_item=DeclaredIdInput), collect=True) as (_, steps):
        assert run("open_item", named="Pen") == "open_item with the uid of the Pen match"
    assert steps == [("open_item", ())]


def test_a_command_with_no_row_id_is_a_load_problem_that_asks_for_the_input():
    with command_inputs_for_steps(_sigs(pair=TwoRequiredStrInput)):
        with pytest.raises(StepFormError, match="declares no row id input"):
            run("pair", named="x")
    with command_inputs_for_steps(_sigs(pair=TwoRequiredStrInput), collect=True) as (_, steps):
        run("pair", named="x")
    assert ("pair", ("named",)) in steps   # the marker the load check reports


def test_named_with_other_concrete_input_and_no_id_is_refused():
    class Loan(BaseModel):
        book_uid: str = Field(description="uid")
        days: int = Field(default=7, description="days")

    with command_inputs_for_steps(_sigs(loan=Loan)):
        with pytest.raises(StepFormError, match="takes no other input"):
            run("loan", named="Dune", days=3)


def test_named_must_not_be_blank():
    with pytest.raises(StepFormError, match="named= is the name the row is picked by"):
        run("open_book", named="  ")


# 7.3 run(..., all_pages=True) replaces full_list

def test_all_pages_is_the_full_list_step():
    assert run("list_loans", book_uid="b1", all_pages=True) == "list_loans book_uid=b1 (full list)"
    assert run("list_loans", all_pages=True) == "list_loans (full list)"


def test_all_pages_and_named_together_are_refused():
    with pytest.raises(StepFormError, match="all_pages=True"):
        run("open_book", named="Dune", all_pages=True)


# 7.4 reserved Input names

def _planning_module(input_class, name="reserved"):
    class Signature:
        """A planning command for the reserved-name check."""
        plain_utterances = ["x"]
        Input = input_class

    class PlanGenerator:
        def find(self, input):
            return ["say that done"]

    module = types.ModuleType(f"_fw_test_{name}")
    module.Signature, module.PlanGenerator = Signature, PlanGenerator
    return module


@pytest.mark.parametrize("field_name", ["named", "all_pages"])
def test_a_planning_command_input_field_named_like_a_helper_keyword_fails_the_load(field_name):
    model = type("Input", (BaseModel,), {"__annotations__": {field_name: str}})
    with pytest.raises(planning_commands.PlanningCommandFormatError, match=f"{field_name!r} is reserved"):
        planning_commands.command_from_module("reserved", _planning_module(model), "_commands/reserved.py")


def test_the_reserved_message_names_the_command_file():
    model = type("Input", (BaseModel,), {"__annotations__": {"named": str}})
    assert planning.reserved_input_name_message(model, "_commands/x.py: x") == (
        "_commands/x.py: x: Input field 'named' is reserved: run() takes named= as its own keyword, so "
        "no command call can pass a field of that name; rename the field")
    assert planning.reserved_input_name_message(OpenBookInput, "x") is None


# Section 3: the Input validators of planning commands are guarded too

def test_a_planning_command_input_validator_that_needs_the_context_says_so_without_one():
    class GuardedInput(BaseModel):
        title: str = Field(description="title")

        @model_validator(mode="after")
        def needs_command_text(self, info):
            return info.context["command_text"] and self

    module = _planning_module(GuardedInput, name="guarded")
    # The template is built with no validation context, so this planning command does not load; the
    # guard is applied before that, at the load point (planning_command_from_parts).
    with pytest.raises(planning_commands.PlanningCommandFormatError):
        planning_commands.command_from_module("guarded", module, "_commands/guarded.py")
    from fastworkflow.utils.input_validation import _CONTEXT_GUARD

    validators = GuardedInput.__pydantic_decorators__.model_validators
    assert validators and all(getattr(dec.func, _CONTEXT_GUARD, False) for dec in validators.values())
    with pytest.raises(ValueError, match="needs the validation context"):
        GuardedInput(title="Dune")


# 7.5 paging: start only; the cap is the constant the catalog prints

def test_only_start_pages_and_the_cap_is_twenty_pages():
    from fastworkflow.observation_offloading.facts import PAGING_PARAMETERS

    assert PAGING_PARAMETERS == ("start",)
    assert CODE_MAX_PAGES == 20


# 7.7 count_and_names_read (listed) and the removed names

def test_count_and_names_read_wording():
    assert count_and_names_read([], 0, "group") == "none (0 groups)"
    assert count_and_names_read([], 4, "book") == "4 books, none of them read"
    assert count_and_names_read(["a"], 3, "group") == "3 groups (a; 1 of 3 shown)"
    assert count_and_names_read(["a", "b"], 2, "right", ("names as above", ["b", "a"])) == \
        "2 rights (the same names as above)"
    assert count_and_names_read(["a"], 1, "person", plural="people") == "1 person (a)"
    assert count(1, "book") == "1 book" and count(3, "copy", "copies") == "3 copies"
    assert count(Placeholder("x.n"), "book") == "<x.n> books"
    assert count_and_names_read(Placeholder("x.names"), Placeholder("x.total"), "group") == \
        "<x.total> groups (<x.names>)"


@pytest.mark.parametrize("name, replacement", [
    ("pick", "run(command, named=<name>"),
    ("full_list", "all_pages=True"),
    ("listed", "count_and_names_read"),
    ("num", "number"),
])
def test_a_removed_name_fails_the_import_naming_its_replacement(name, replacement):
    with pytest.raises(ImportError, match=f"{name} is removed") as info:
        exec(f"from fastworkflow.planning import {name}", {})
    assert replacement in str(info.value)


def test_a_removed_name_is_not_an_attribute_of_the_module():
    with pytest.raises(ImportError, match="pick is removed"):
        planning.pick
    with pytest.raises(AttributeError):
        planning.not_a_helper  # noqa: B018


def test_the_removed_names_are_not_exported():
    assert not {"pick", "full_list", "listed", "num"} & set(planning.__all__)
    assert {"count", "count_and_names_read"} <= set(planning.__all__)

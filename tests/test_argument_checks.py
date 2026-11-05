"""Phase 11 steps 1 and 5: a planning command's ``Input`` takes any type the parameter extractor
supports (``list[str]``, ``float``), a ``bool`` may default to true, and an open-ended type gets one
sample value in the preview (design 2.22, 11.5). Each case runs the real loader, planner, phase
runner or preview; the planner's model is a ``DummyLM`` (the accepted LLM boundary)."""
from __future__ import annotations

import os
import re
import sys
from typing import Literal

import pytest
from dspy.utils import DummyLM
from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

from fastworkflow import command_preview, tracing, workflow_agent
from fastworkflow import planning_commands as pc
from fastworkflow.plan_frame import TURN_MAX_STEPS_ENV
from fastworkflow.planning_commands import PlanningCommandInputError
from fastworkflow.turn import TurnStatus
from fastworkflow.utils.signatures import InputForParamExtraction
from tests.test_turn_result_capture import no_review_gate  # noqa: F401 - the review gate is off for these turns
from tests.test_planning_commands import _clear_library_wf_sys_modules  # noqa: F401 - the library_wf purge
from tests.test_text_planner_and_split import (  # noqa: F401 - workflow_env is a fixture
    RecordingTraceSink, _prompt, _session, workflow_env)
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401 - shared fixture

#: A planning command whose Input takes a list of text and a float, and a bool that defaults to true.
#: Its steps check the first share of the titles (``ratio``), one nested ``book_by_title`` call each.
STOCK_CHECK = '''"""stock_check: check the copies of several titles, for the test."""
from pydantic import BaseModel, Field

from fastworkflow.planning import run


class Signature:
    """Check the copies of several titles."""

    class Input(BaseModel):
        titles: list[str] = Field(description="the titles to check")
        ratio: float = Field(default=0.5, description="the share of the titles to check")
        checked: bool = Field(default=True, description="also report the titles checked")
        tags: list[str] = Field(default_factory=list, description="labels of the titles")


class PlanGenerator:
    def steps(self, input):
        count = max(1, round(input.ratio * len(input.titles)))
        return [run("book_by_title", title=title) for title in input.titles[:count]]
'''

#: The same Input with steps that name the step stubs of the test folder (find_group), for the loader,
#: the planner and the preview, which run no library.
STOCK_LIST = '''"""stock_check: check the copies of several titles, for the test."""
from pydantic import BaseModel, Field

from fastworkflow.planning import run


class Signature:
    """Check the copies of several titles."""

    class Input(BaseModel):
        titles: list[str] = Field(description="the titles to check")
        ratio: float = Field(default=0.5, description="the share of the titles to check")
        checked: bool = Field(default=True, description="also report the titles checked")
        tags: list[str] = Field(default_factory=list, description="labels of the titles")


class PlanGenerator:
    def steps(self, input):
        count = max(1, round(input.ratio * len(input.titles)))
        return [run("find_group", query=title) for title in input.titles[:count]]
'''

#: Sample-value cases: examples on the list and float, and the placeholders of the ones without.
SAMPLES = '''"""samples: the sample values of open-ended types, for the test."""
from pydantic import BaseModel, Field

from fastworkflow.planning import run


class Signature:
    """Check some titles, for the sample-value test."""

    class Input(BaseModel):
        titles: list[str] = Field(description="the titles", examples=[["Dune", "Emma"], ["Persuasion"]])
        ratio: float = Field(description="a share", examples=[0.25, 0.75])
        weight: float = Field(description="a weight, required and without examples")
        tags: list[str] = Field(default=["fiction"], description="labels, optional with a default")
        notes: list[str] = Field(default_factory=list, description="notes, optional and empty")


class PlanGenerator:
    def steps(self, input):
        return [run("find_group", query=input.titles[0])]
'''

ROUTE = "stock_check"


def _module(field: str) -> str:
    """A planning command file whose Input has one field, as *field* writes it."""
    return ('from pydantic import BaseModel, Field\n\n\nclass Signature:\n    """Weigh one title."""\n\n'
            f'    class Input(BaseModel):\n        {field}\n\n\n'
            'class PlanGenerator:\n    def steps(self, input):\n        return ["say the weight"]\n')


def _reset_loader_caches() -> None:
    """The step-command stubs this file writes share command names with other tests' workflows, and the
    routing cache is keyed by command name alone, so each test starts and ends with the caches cleared."""
    from fastworkflow.command_routing import RoutingDefinition

    RoutingDefinition._GLOBAL_COMMAND_CLASS_CACHE.clear()
    pc.resolve_field_sources.cache_clear()
    workflow_agent._build_parsed_planning_commands.cache_clear()


@pytest.fixture(autouse=True)
def _loader_caches_per_test():
    _reset_loader_caches()
    yield
    _reset_loader_caches()


def _write(tmp_path, files: dict) -> str:
    from tests.test_planning_command_files import _write as write_files
    return write_files(tmp_path, files)


def _load_error(folder: str, planning_command: str = "bad") -> str:
    from tests.test_planning_command_files import _load_error as load_error
    return load_error(folder, planning_command)


def _plan_planning_command(*args, **kwargs):
    from tests.test_planning_command_files import _plan_planning_command as plan
    return plan(*args, **kwargs)


def _command(tmp_path, text: str = STOCK_LIST, name: str = ROUTE):
    folder = _write(tmp_path, {name: text})
    return workflow_agent.load_planning_command_map(folder)[name]


# ---------------------------------------------------------------- loading (step 1)

def test_a_list_of_text_a_float_and_a_bool_that_defaults_true_load_with_their_types_and_defaults(tmp_path):
    command = _command(tmp_path)

    assert [(p.name, p.type, p.required, p.default) for p in command.params] == [
        ("titles", "list[str]", True, None),
        ("ratio", "float", False, 0.5),
        ("checked", "bool", False, True),
        ("tags", "list[str]", False, ()),
    ]
    assert command.signature_text().splitlines()[0] == (
        "stock_check(titles: list[str], ratio: float = 0.5, checked: bool = true, tags: list[str] = [])")


@pytest.mark.parametrize("field, message", [
    ("counts: list[int] = []", "has type"),
    ("ratio: float = 'x'", "is not a number"),
    ("ratio: float = float('nan')", "is not a number"),
    ("tags: list[str] = ['a', 2]", "default must be a list of text"),
    ("flag: bool = 'yes'", "default must be true or false"),
])
def test_a_field_of_an_unsupported_type_or_a_default_of_the_wrong_kind_fails_the_load(tmp_path, field, message):
    assert re.search(message, _load_error(_write(tmp_path, {"bad": _module(field)})))


def test_the_bool_default_true_is_the_value_an_absent_argument_takes(tmp_path):
    command = _command(tmp_path)
    args, _ = pc.check_arguments(command, {"titles": ["Dune"], "checked": ""}, "check Dune")
    assert args["checked"] is True
    args, _ = pc.check_arguments(command, {"titles": ["Dune"], "checked": False}, "check Dune")
    assert args["checked"] is False


def test_the_typed_call_declares_the_list_float_and_bool_fields_with_their_types(tmp_path):
    command = _command(tmp_path)
    fields = pc.typed_signature(command).output_fields
    assert fields["titles"].annotation == list[str]
    assert fields["ratio"].annotation is float
    assert fields["checked"].annotation is bool
    assert pc._field_desc(command.param("checked")) == (
        "also report the titles checked. false only when the request explicitly asks against this; otherwise true")


def test_the_cme_signature_of_a_list_and_a_float_field_carries_their_types(tmp_path):
    command = _command(tmp_path)
    signature = InputForParamExtraction.create_signature_from_pydantic_model(command.input_model)
    assert signature.model_fields["titles"].annotation == list[str]
    assert signature.model_fields["ratio"].annotation is float


# ---------------------------------------------------------------- arguments (step 1)

def test_list_float_and_bool_arguments_are_read_from_their_several_spellings(tmp_path):
    command = _command(tmp_path)
    request = "check the copies of Dune and Emma"

    args, _ = pc.check_arguments(command, {"titles": ["Dune", "Emma"], "ratio": 2.5, "checked": ""}, request)
    assert args["titles"] == ["Dune", "Emma"] and args["ratio"] == 2.5 and args["checked"] is True

    args, _ = pc.check_arguments(command, {"titles": '["Dune"]', "ratio": "0.25", "checked": False}, request)
    assert args["titles"] == ["Dune"] and args["ratio"] == 0.25 and args["checked"] is False

    args, _ = pc.check_arguments(command, {"titles": "Dune, Emma", "ratio": ""}, request)
    assert args["titles"] == ["Dune", "Emma"] and args["ratio"] == 0.5


def test_a_list_of_one_text_is_a_list_not_its_element(tmp_path):
    command = _command(tmp_path)
    args, _ = pc.check_arguments(command, {"titles": ["Dune"]}, "check Dune")
    assert args["titles"] == ["Dune"]


def test_a_required_list_with_no_element_and_a_float_that_is_not_a_number_are_refused(tmp_path):
    command = _command(tmp_path)
    with pytest.raises(PlanningCommandInputError, match="missing required argument titles"):
        pc.check_arguments(command, {"titles": []}, "check the titles")
    with pytest.raises(PlanningCommandInputError, match="ratio='lots' is not a number"):
        pc.check_arguments(command, {"titles": ["Dune"], "ratio": "lots"}, "check Dune")


def test_the_extractors_not_given_float_marker_is_missing_or_the_default(tmp_path):
    command = _command(tmp_path)
    args, _ = pc.check_arguments(command, {"titles": ["Dune"], "ratio": -sys.float_info.max}, "check Dune")
    assert args["ratio"] == 0.5

    weigh = _command(tmp_path / "weigh", _module('weight: float = Field(description="the weight")'),
                     name="weigh")
    with pytest.raises(PlanningCommandInputError, match="missing required argument weight"):
        pc.check_arguments(weigh, {"weight": -sys.float_info.max}, "weigh it")


# ---------------------------------------------------------------- the planner's typed call (step 1)

def test_the_typed_call_fills_a_list_and_a_float_and_the_turn_frame_expands_them(tmp_path, workflow_env):
    workflow_env()
    command = _command(tmp_path, STOCK_LIST.replace("stock_list", ROUTE))
    request = "Check the copies of Dune and Emma, all of them"

    sink = RecordingTraceSink()
    ctx, wf = _session(sink)
    ctx._begin_turn(request)
    lm = DummyLM([{"reasoning": "r", "titles": ["Dune", "Emma"], "ratio": 1.0, "checked": True, "tags": [],
                   "steps_before": [], "steps_after": []}])
    result = _plan_planning_command(ctx, wf, lm, request, command)

    span = sink.named(tracing.SPAN_PLANNER_PLAN)[-1]
    assert span.attributes["plan_source"] == "skill_call" and span.attributes["skill"] == ROUTE
    assert span.attributes["plan"].splitlines()[:2] == ["1. find_group query=Dune", "2. find_group query=Emma"]
    assert "Execute these next steps:\n1. find_group query=Dune" in result
    assert ("stock_check(titles: list[str], ratio: float = 0.5, checked: bool = true, tags: list[str] = [])"
            in _prompt(lm, 0))


# ---------------------------------------------------------------- the turn frame and a nested call (step 1)

NESTED_TITLES = ["Dune", "Emma"]   # the library's Dune (b1) and Emma (b3)


@pytest.fixture
def stock_env(tmp_path, initialized_fastworkflow, monkeypatch):
    # imported here: the nested-call commands' module has side effects on the library workflow's modules
    from tests.test_nested_planning_commands import NESTED_COMMANDS
    from tests.test_step_helpers import _make_library_runner_env
    monkeypatch.delenv(TURN_MAX_STEPS_ENV, raising=False)
    state = _make_library_runner_env(
        tmp_path, initialized_fastworkflow, monkeypatch,
        extra_planning_commands={"Library/stock_check": STOCK_CHECK},
        extra_commands=NESTED_COMMANDS, real_dispatch=True)
    yield state
    state.ctx.pop_active_workflow()


def _run_stock_check(state, args: dict):
    state.planning_command = pc.planning_command(state.wf.folderpath, "Library/stock_check")
    assert state.planning_command is not None
    state.args = args
    return state.ctx.process_turn("check the copies of Dune and Emma")


def _nested_book_uids(state) -> list:
    from tests.test_nested_planning_commands import BOOK_BY_TITLE
    return [o.command_response.output.book_uid for o in state.ctx._turn_outputs if o.command_name == BOOK_BY_TITLE]


def test_a_list_and_float_planning_command_is_called_at_the_turn_frame_and_its_list_item_as_a_nested_call(
        stock_env):
    result = _run_stock_check(stock_env, {"titles": ["Emma"], "ratio": 1.0, "checked": True, "tags": []})

    assert result.status == TurnStatus.COMPLETED
    # the list's one element is the nested book_by_title call's title (Emma, b3)
    assert _nested_book_uids(stock_env) == ["b3"]
    [span] = [s for s in stock_env.sink.named(tracing.SPAN_PLANNING_COMMAND_PHASE)
              if s.attributes.get("skill") == "Library/stock_check"]
    assert span.attributes["steps"] == "1. book_by_title title=Emma"
    assert span.attributes["steps_by_code"] == 1
    assert stock_env.model.asked == [] and stock_env.nested.asked == []


def test_the_float_decides_how_many_list_items_the_nested_calls_check(stock_env):
    # ratio 0.5 of two titles checks the first one only: one nested call, for Dune (b1)
    result = _run_stock_check(stock_env, {"titles": NESTED_TITLES, "ratio": 0.5, "checked": True, "tags": []})

    assert result.status == TurnStatus.COMPLETED
    assert _nested_book_uids(stock_env) == ["b1"]


# ---------------------------------------------------------------- the preview (step 1)

def test_the_preview_of_a_list_and_float_planning_command_renders(tmp_path):
    folder = _write(tmp_path, {ROUTE: STOCK_LIST.replace("stock_list", ROUTE)})

    preview = command_preview.render_previews(folder)[f"{ROUTE}.md"]

    assert "| `titles` | list[str] | required | `[\"<titles>\"]` |" in preview
    assert "| `ratio` | float | `0.5` | `0.5` |" in preview
    assert "| `checked` | bool | true | false |" in preview


# ---------------------------------------------------------------- the sample values (step 5)

def test_the_sample_of_a_list_is_its_first_example_else_its_default_else_a_placeholder(tmp_path):
    command = _command(tmp_path, SAMPLES, name="samples")
    values = {p.name: command_preview.sample_value(command, p) for p in command.params}

    assert values["titles"] == ["Dune", "Emma"]          # the first example
    assert values["tags"] == ["fiction"]                 # optional with a default, no example
    assert values["notes"] == ["<notes>"]                # optional and empty, no example: the placeholder


def test_the_sample_of_a_float_is_its_first_example_else_its_default_else_one(tmp_path):
    command = _command(tmp_path, SAMPLES, name="samples")
    values = {p.name: command_preview.sample_value(command, p) for p in command.params}

    assert values["ratio"] == 0.25 and isinstance(values["ratio"], float)  # the first example
    assert values["weight"] == 1.0                                       # required, no example


def test_the_sample_of_a_list_and_a_float_without_examples_is_the_placeholder_or_default(tmp_path):
    command = _command(tmp_path)  # STOCK_LIST: no examples
    values = {p.name: command_preview.sample_value(command, p) for p in command.params}

    assert values["titles"] == ["<titles>"]
    assert values["ratio"] == 0.5    # optional with a default
    assert values["tags"] == ["<tags>"]  # optional and empty


# ---------------------------------------------------------------- check options on any command (step 2)

class GroupInput(BaseModel):
    """An execution command's Input whose fields carry the check options (2.46)."""

    name: str = Field(description="the group's name", json_schema_extra={"in_request": True})
    plain: str = Field(default="", description="a free note, with no option")
    count: int = Field(default=2, description="how many members", json_schema_extra={"in_request": True})
    titles: list[str] = Field(default_factory=list, description="titles",
                              json_schema_extra={"in_request": True})
    ratio: float = Field(default=0.5, description="a share, which takes no option")
    checked: bool = Field(default=False, description="also check the titles",
                          json_schema_extra={"verify": True})


class GroupSubjectInput(BaseModel):
    name: str = Field(description="the record's name", json_schema_extra={"in_request": "subject"})


class GroupStripInput(BaseModel):
    name: str = Field(description="the group", json_schema_extra={"in_request": {"ignore": ["group"]}})


class ValidatedInput(BaseModel):
    """An Input whose own validator reads the request from its validation context."""

    name: str = Field(description="a name, which the request must write")

    @field_validator("name")
    @classmethod
    def written_in_the_request(cls, value, info: ValidationInfo):
        if value not in str((info.context or {}).get("request", "")):
            raise ValueError("not written in the request")
        return value


def _exec(request: str):
    """The validation step of an execution command, for a request (the app workflow is not needed)."""
    return InputForParamExtraction(command=request, input_for_param_extraction_class=None)


def test_an_execution_command_str_with_in_request_refuses_a_value_the_request_does_not_write():
    class Input(GroupInput):
        pass

    is_valid, message, *_ = _exec("Show the Payroll group").validate_parameters(
        None, "Library/show_group", Input(name="Billing", titles=["Dune"]))
    assert not is_valid
    assert message == "name='Billing' is not written in the request"


def test_the_same_execution_command_str_without_the_option_accepts_the_value():
    class Input(BaseModel):
        name: str = Field(description="the group's name")

    is_valid, *_ = _exec("Show the Payroll group").validate_parameters(
        None, "Library/show_group", Input(name="Billing"))
    assert is_valid


def test_a_planning_command_str_with_in_request_refuses_and_the_same_str_without_it_accepts(tmp_path):
    with_option = _command(tmp_path / "a", _module('name: str = Field(description="n", json_schema_extra={"in_request": True})'),
                           name="weigh")
    without = _command(tmp_path / "b", _module('name: str = Field(description="n")'), name="weigh")
    with pytest.raises(PlanningCommandInputError, match="name='Billing' is not written in the request"):
        pc.check_arguments(with_option, {"name": "Billing"}, "Show the Payroll group")
    args, _ = pc.check_arguments(without, {"name": "Billing"}, "Show the Payroll group")
    assert args["name"] == "Billing"


def test_subject_strip_and_in_request_on_an_int_act_on_an_execution_command_as_on_a_planning_command(tmp_path):
    subject = pc.option_command(GroupSubjectInput, "Library/show_group")
    with pytest.raises(PlanningCommandInputError, match="is not written in the request"):
        pc.check_arguments(subject, {"name": "Billing"}, "Show the Payroll group")

    strip = pc.option_command(GroupStripInput, "Library/show_group")
    args, _ = pc.check_arguments(strip, {"name": "Payroll group"}, "Show the Payroll group")
    assert args["name"] == "Payroll"

    count = pc.option_command(GroupInput, "Library/show_group")
    args, notes = pc.check_arguments(count, {"count": 5, "name": "Payroll"}, "Open 3 members of Payroll")
    assert args["count"] == 3 and notes

    # the same through the execution command's validation step
    instance = GroupStripInput(name="Payroll group")
    is_valid, *_ = _exec("Show the Payroll group").validate_parameters(
        None, "Library/show_group", instance)
    assert is_valid and instance.name == "Payroll"


def test_verify_checks_nothing_without_a_classifier_and_takes_the_verdict_with_one_on_either_kind(
        tmp_path, monkeypatch):
    request = "Check the titles Dune of Payroll"
    # no classifier (no JEV_API_KEY): the field's value stands
    monkeypatch.setattr("fastworkflow.workflow_agent._jev_lm", lambda: None)
    instance = GroupInput(name="Payroll", checked=True, titles=["Dune"])
    assert _exec(request).validate_parameters(None, "Library/show_group", instance)[0]
    assert instance.checked is True

    # the classifier's verdict is the seam: check_verdicts is asked for each verify field
    seen = []

    def verdicts(lm, command, text):
        seen.append((command.name, [p.name for p in command.params if p.verify]))
        return {"checked": (False, 0.99)}

    monkeypatch.setattr("fastworkflow.workflow_agent._jev_lm", lambda: object())
    monkeypatch.setattr(pc, "check_verdicts", verdicts)
    instance = GroupInput(name="Payroll", checked=True, titles=["Dune"])
    assert _exec(request).validate_parameters(None, "Library/show_group", instance)[0]
    assert instance.checked is False
    assert seen == [("Library/show_group", ["checked"])]

    command = _command(tmp_path, _module('checked: bool = Field(default=True, description="check it", '
                                         'json_schema_extra={"verify": True})'), name="weigh")
    args, _ = pc.check_arguments(command, {"checked": True}, request, {"checked": (False, 0.99)})
    assert args["checked"] is False


def test_validate_extracted_parameters_runs_once_after_the_option_checks_for_an_execution_command():
    calls = []

    class Hook:
        @staticmethod
        def validate_extracted_parameters(workflow, command, cmd_parameters):
            calls.append(cmd_parameters.name)
            return None

    instance = GroupStripInput(name="Payroll group")
    is_valid, *_ = InputForParamExtraction(command="Show the Payroll group",
                                           input_for_param_extraction_class=Hook).validate_parameters(
        None, "Library/show_group", instance)
    assert is_valid and calls == ["Payroll"]          # after the strip and the request's spelling, once

    calls.clear()
    refused = GroupStripInput(name="Billing")
    is_valid, message, *_ = InputForParamExtraction(command="Show the Payroll group",
                                                    input_for_param_extraction_class=Hook).validate_parameters(
        None, "Library/show_group", refused)
    assert not is_valid and "not written in the request" in message
    assert calls == []                                 # a refused value never reaches the hook


HOOKED_WEIGH = '''"""weigh: weigh one group, for the test."""
import fastworkflow
from pydantic import BaseModel, Field


class Signature:
    """Weigh one group."""

    template = ["say the weight"]

    class Input(BaseModel):
        name: str = Field(description="the group", json_schema_extra={"in_request": {"ignore": ["group"]}})

    @staticmethod
    def validate_extracted_parameters(workflow, command, cmd_parameters, command_text=None):
        # the hook sees the value after the option checks (the ignore words are already gone)
        if cmd_parameters.name != "Payroll":
            raise fastworkflow.Refused(f"the hook saw {cmd_parameters.name}", field="name")
        return None


class PlanGenerator:
    def steps(self, input):
        return ["say the weight"]
'''


def test_validate_extracted_parameters_runs_once_after_the_option_checks_for_a_planning_command(tmp_path):
    command = _command(tmp_path, HOOKED_WEIGH, name="weigh")
    assert command.validate is not None
    # the option's ignore words and the request's spelling come first; the hook then sees "Payroll"
    args, _ = pc.check_arguments(command, {"name": "Payroll group"}, "Show the Payroll group")
    assert pc._run_validate(command, args, "Show the Payroll group", None)["name"] == "Payroll"
    # a value the option refuses never reaches the hook
    with pytest.raises(PlanningCommandInputError, match="name='Billing' is not written in the request"):
        pc.check_arguments(command, {"name": "Billing"}, "Show the Payroll group")


def test_an_input_validator_that_reads_the_request_refuses_on_either_kind():
    refusal = _exec("Show the Payroll group").validate_parameters(
        None, "Library/show_group", ValidatedInput.model_construct(name="Billing"))
    assert not refusal[0] and "not written in the request" in refusal[1]

    option = pc.option_command(ValidatedInput, "Library/show_group")
    with pytest.raises(PlanningCommandInputError, match="not written in the request"):
        pc.check_input(option, {"name": "Billing"}, "Show the Payroll group", None)
    pc.check_input(option, {"name": "Payroll"}, "Show the Payroll group", None)


def test_an_option_on_a_type_it_does_not_apply_to_is_a_load_error_naming_the_option_and_the_type(tmp_path):
    class BadFloat(BaseModel):
        ratio: float = Field(description="a share", json_schema_extra={"in_request": True})

    with pytest.raises(pc.PlanningCommandFormatError,
                       match="option in_request does not apply to field 'ratio' of type float"):
        pc.option_params(BadFloat, "app/show")

    class BadVerify(BaseModel):
        name: str = Field(description="a name", json_schema_extra={"verify": True})

    with pytest.raises(pc.PlanningCommandFormatError,
                       match="option verify does not apply to field 'name' of type str"):
        pc.option_params(BadVerify, "app/show")

    class BadMeans(BaseModel):
        kind: str = Field(description="a kind", json_schema_extra={"verify": {"a": "x"}})

    with pytest.raises(pc.PlanningCommandFormatError,
                       match="option verify \\(a dict of meanings\\) does not apply to field 'kind' of type str"):
        pc.option_params(BadMeans, "app/show")

    class VerifyDictOnEnum(BaseModel):
        kind: Literal["a", "b"] = Field(description="a kind", json_schema_extra={"verify": {"a": "x"}})
    (check,) = pc.option_params(VerifyDictOnEnum, "app/show")
    assert (check.verify, check.means) == (True, (("a", "x"),))   # the dict is the check: no separate verify

    pick = _module('kind: Literal["a", "b"] = Field(description="a kind", json_schema_extra={"means": {"a": "x"}})')
    pick = pick.replace("from pydantic import", "from typing import Literal\nfrom pydantic import", 1)
    error = _load_error(_write(tmp_path, {"pick": pick}), "pick")
    assert "option means is retired: write verify: {value: meaning, ...}" in error

    error = _load_error(_write(tmp_path, {"weigh": _module(
        'ratio: float = Field(description="a share", json_schema_extra={"in_request": True})')}), "weigh")
    assert "option in_request does not apply to field 'ratio' of type float" in error


def test_the_extractor_reads_the_options_sentences_for_both_kinds_and_none_for_a_field_without_options(tmp_path):
    assert pc.option_sentences(GroupInput) == {
        "name": (pc.IN_REQUEST_STR_SENTENCE,),
        "count": (pc.IN_REQUEST_INT_SENTENCE,),
        "titles": (pc.IN_REQUEST_LIST_SENTENCE,),
        "checked": (pc.VERIFY_BOOL_SENTENCE,),
    }
    assert "plain" not in pc.option_sentences(GroupInput) and "ratio" not in pc.option_sentences(GroupInput)

    signature = InputForParamExtraction.create_signature_from_pydantic_model(
        GroupInput, pc.option_sentences(GroupInput))
    desc = {name: signature.fields[name].json_schema_extra["desc"] for name in ("name", "titles", "ratio")}
    assert pc.IN_REQUEST_STR_SENTENCE in desc["name"]
    assert pc.IN_REQUEST_LIST_SENTENCE in desc["titles"]
    assert desc["ratio"].startswith("a share, which takes no option")
    assert "Copy" not in desc["ratio"] and "request writes" not in desc["ratio"]

    command = _command(tmp_path, _module('name: str = Field(description="the name", '
                                         'json_schema_extra={"in_request": True})'), name="weigh")
    assert pc.option_sentences(command.input_model) == {"name": (pc.IN_REQUEST_STR_SENTENCE,)}
    plain = _command(tmp_path / "plain", _module('name: str = Field(description="the name")'), name="weigh")
    assert pc.option_sentences(plain.input_model) == {}


def test_an_in_request_list_keeps_the_written_items_and_drops_the_rest_per_element(tmp_path):
    command = _command(tmp_path, _module('titles: list[str] = Field(description="titles", '
                                         'json_schema_extra={"in_request": True})'), name="weigh")
    args, notes = pc.check_arguments(command, {"titles": ["Dune", "Hamlet"]}, "check Dune and Emma")
    assert args["titles"] == ["Dune"]
    assert any("'Hamlet' is not written in the request: dropped" in note for note in notes)

    with pytest.raises(PlanningCommandInputError, match="titles=.* is not written in the request"):
        pc.check_arguments(command, {"titles": ["Hamlet"]}, "check Dune and Emma")

    option = pc.option_command(type("ListInput", (BaseModel,), {
        "__annotations__": {"titles": list[str]},
        "titles": Field(description="titles", json_schema_extra={"in_request": True})}), "x")
    args, _ = pc.check_arguments(option, {"titles": ["Dune", "Hamlet"]}, "check Dune and Emma")
    assert args["titles"] == ["Dune"]


# ---------------------------------------------------------------- the tutorial workflow's fields (step 6)

TUTORIAL_WORKFLOW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "tests", "planning_tutorial_workflow")


def test_the_tutorial_workflows_checked_field_refuses_an_unwritten_value_and_an_unchecked_one_accepts_it():
    # book_by_title's title carries in_request (tutorial pages 4 and 12): a title the request does not write is refused.
    # member_loans' member carries no option (the tutorial's member_loans has none), so the same kind of value passes.
    commands = workflow_agent.load_planning_command_map(TUTORIAL_WORKFLOW)

    with pytest.raises(PlanningCommandInputError, match="title='Emma' is not written in the request"):
        pc.check_arguments(commands["book_by_title"], {"title": "Emma"}, "Who wrote Dune?")
    args, _ = pc.check_arguments(commands["book_by_title"], {"title": "Dune"}, "Who wrote Dune?")
    assert args["title"] == "Dune"

    args, _ = pc.check_arguments(commands["member_loans"], {"member": "Ann Lee"}, "What has Raj Patel borrowed?")
    assert args["member"] == "Ann Lee"


# ---------------------------------------------------------------- the two-option form (fix-3rsq.two-options)

def _literal_module(field: str) -> str:
    return _module(field).replace("from pydantic import", "from typing import Literal\nfrom pydantic import", 1)


@pytest.mark.parametrize("field, message", [
    ('name: str = Field(description="n", json_schema_extra={"subject": True})',
     'option subject is retired: write in_request: "subject"'),
    ('kind: Literal["a", "b"] = Field(description="k", json_schema_extra={"means": {"a": "x"}})',
     "option means is retired: write verify: {value: meaning, ...}"),
    ('name: str = Field(description="n", json_schema_extra={"strip": ["group"]})',
     'option strip is retired: write in_request: {\"ignore\": [...]}'),
])
def test_a_retired_option_is_a_load_error_naming_the_new_spelling_on_a_planning_command(tmp_path, field, message):
    error = _load_error(_write(tmp_path, {"bad": _literal_module(field)}))
    assert message in error


@pytest.mark.parametrize("field, message", [
    ('name: str = Field(description="n", json_schema_extra={"subject": True})',
     'option subject is retired: write in_request: "subject"'),
    ('kind: Literal["a", "b"] = Field(description="k", json_schema_extra={"means": {"a": "x"}})',
     "option means is retired: write verify: {value: meaning, ...}"),
    ('name: str = Field(description="n", json_schema_extra={"strip": ["group"]})',
     'option strip is retired: write in_request: {\"ignore\": [...]}'),
])
def test_a_retired_option_is_a_load_error_naming_the_new_spelling_on_an_execution_command(field, message):
    class Input(BaseModel):
        pass

    module = ("from typing import Literal\nfrom pydantic import BaseModel, Field\n\n\n"
              "class Input(BaseModel):\n        " + field + "\n")
    namespace: dict = {}
    exec(compile(module, "<retired>", "exec"), namespace)
    with pytest.raises(pc.PlanningCommandFormatError, match=re.escape(message)):
        pc.option_params(namespace["Input"], "app/show")


def test_a_verify_dict_with_a_value_the_enum_lacks_is_a_load_error(tmp_path):
    error = _load_error(_write(tmp_path, {"bad": _literal_module(
        'kind: Literal["a", "b"] = Field(description="k", json_schema_extra={"verify": {"c": "x"}})')}))
    assert "verify of 'kind': 'c' is not a value of it, or not text" in error


def test_a_verify_dict_on_a_non_enum_is_a_load_error_naming_the_type(tmp_path):
    error = _load_error(_write(tmp_path, {"bad": _module(
        'flag: bool = Field(default=False, description="f", json_schema_extra={"verify": {"true": "asked"}})')}))
    assert "option verify (a dict of meanings) does not apply to field 'flag' of type bool (it takes an enum)" in error


def test_in_request_subject_on_a_non_str_is_a_load_error_and_on_a_str_is_the_subject_option(tmp_path):
    error = _load_error(_write(tmp_path, {"bad": _module(
        'count: int = Field(default=1, description="c", json_schema_extra={"in_request": "subject"})')}))
    assert 'option in_request: "subject" does not apply to field \'count\' of type int (it takes a str)' in error

    command = _command(tmp_path / "ok", _module(
        'name: str = Field(description="n", json_schema_extra={"in_request": "subject"})'), name="weigh")
    param = command.param("name")
    assert (param.subject, param.in_request) == (True, True)
    assert pc.extraction_sentences(param) == (pc.SUBJECT_SENTENCE,)


def test_in_request_on_a_planning_command_read_as_true_or_false_or_subject_only(tmp_path):
    error = _load_error(_write(tmp_path, {"bad": _module(
        'name: str = Field(description="n", json_schema_extra={"in_request": "yes"})')}))
    assert 'in_request must be true, false, "subject" or a dict' in error


def test_a_verify_dict_is_the_check_and_its_meanings_are_the_sentences_the_check_reads(tmp_path):
    command = _command(tmp_path, _literal_module(
        'kind: Literal["a", "b"] = Field(description="a kind", json_schema_extra={"verify": {"b": "the b one"}})'
    ), name="weigh")
    param = command.param("kind")
    assert (param.verify, param.means) == (True, (("b", "the b one"),))
    assert pc._check_options(param) == (("a", "a"), ("b", "the b one"))


def test_strip_words_cleans_a_value_from_a_validator_and_from_the_validation_hook(tmp_path):
    class Cleaned(BaseModel):
        name: str = Field(description="the group")

        @field_validator("name")
        @classmethod
        def without_group_words(cls, value):
            return pc.strip_words(value, ("group",))

    # a validator returns the cleaned value to its caller (model_validate)
    assert Cleaned.model_validate({"name": "Payroll group"}).name == "Payroll"
    assert pc.strip_words("Payroll group", ("group",)) == "Payroll"
    assert pc.strip_words("Ann's group", ("group",)) == "Ann"
    # the hook is the step that keeps the cleaned value: _run_validate returns the hook's fields
    command = _command(tmp_path, HOOKED_WEIGH, name="weigh")
    args, _ = pc.check_arguments(command, {"name": "Payroll group"}, "Show the Payroll group")
    assert pc._run_validate(command, args, "Show the Payroll group", None)["name"] == "Payroll"


# ---------------------------------------------------------------- in_request as a dict: ignore (fix-3rsq.two-options)

def test_the_in_request_dict_forms_read_as_subject_and_ignore(tmp_path):
    class Forms(BaseModel):
        plain: str = Field(description="p", json_schema_extra={"in_request": {"ignore": ["group"]}})
        named: str = Field(description="n", json_schema_extra={"in_request": {"subject": True}})
        both: str = Field(description="b", json_schema_extra={"in_request": {"subject": True, "ignore": ["unit"]}})

    params = {p.name: p for p in pc.option_params(Forms, "app/show")}
    assert (params["plain"].in_request, params["plain"].subject, params["plain"].ignore) == (True, False, ("group",))
    assert (params["named"].in_request, params["named"].subject, params["named"].ignore) == (True, True, ())
    assert (params["both"].in_request, params["both"].subject, params["both"].ignore) == (True, True, ("unit",))
    # the dict {"subject": True} is the same option as "subject"
    assert pc.extraction_sentences(params["named"]) == (pc.SUBJECT_SENTENCE,)


@pytest.mark.parametrize("value, message", [
    ('{"sbject": True}', "in_request has unknown key(s) sbject"),
    ('{"ignore": "group"}', "in_request ignore must be a list of words"),
    ('{"ignore": [1]}', "in_request ignore must be a list of words"),
    ('{}', "in_request {} is empty"),
    ('{"subject": "yes"}', "in_request subject must be true or false"),
])
def test_a_malformed_in_request_dict_is_a_load_error(tmp_path, value, message):
    error = _load_error(_write(tmp_path, {"bad": _module(
        f'name: str = Field(description="n", json_schema_extra={{"in_request": {value}}})')}))
    assert message in error


def test_ignore_on_a_non_str_is_a_load_error_naming_the_option_and_the_type(tmp_path):
    error = _load_error(_write(tmp_path, {"bad": _module(
        'count: int = Field(default=1, description="c", json_schema_extra={"in_request": {"ignore": ["x"]}})')}))
    assert "option in_request ignore does not apply to field 'count' of type int (it takes a str)" in error

    class BadInt(BaseModel):
        count: int = Field(default=1, description="c", json_schema_extra={"in_request": {"ignore": ["x"]}})

    with pytest.raises(pc.PlanningCommandFormatError, match="in_request ignore does not apply to field 'count'"):
        pc.option_params(BadInt, "app/show")


class IgnoreGroupInput(BaseModel):
    name: str = Field(description="the group", json_schema_extra={"in_request": {"ignore": ["group"]}})


def test_ignore_keeps_the_strip_outcomes_of_head_on_the_payroll_group_cases():
    """HEAD (the strip option) against the new form, the value the model writes being "Payroll group":
    a required in_request str, an optional one, and a subject, with the request writing "Show Payroll".
    HEAD gave "Payroll" in all three."""
    request = "Show Payroll"
    required = pc.option_command(IgnoreGroupInput, "Library/show_group")
    args, _ = pc.check_arguments(required, {"name": "Payroll group"}, request)
    assert args["name"] == "Payroll"

    class OptionalIgnore(BaseModel):
        name: str = Field(default="", description="the group",
                          json_schema_extra={"in_request": {"ignore": ["group"]}})

    args, _ = pc.check_arguments(pc.option_command(OptionalIgnore, "Library/show_group"),
                                 {"name": "Payroll group"}, request)
    assert args["name"] == "Payroll"

    class SubjectIgnore(BaseModel):
        name: str = Field(description="the group", json_schema_extra={"in_request": {"subject": True,
                                                                                      "ignore": ["group"]}})

    args, _ = pc.check_arguments(pc.option_command(SubjectIgnore, "Library/show_group"),
                                 {"name": "Payroll group"}, request)
    assert args["name"] == "Payroll"


# ---------------------------------------------------------------- Input validators write back (design 3.4)

def test_a_field_validator_that_cleans_a_value_changes_the_argument_of_a_planning_command(tmp_path):
    text = ('from pydantic import BaseModel, Field, field_validator\n'
            'from fastworkflow.planning_commands import strip_words\n\n\n'
            'class Signature:\n    """Weigh one group."""\n\n'
            '    class Input(BaseModel):\n'
            '        name: str = Field(description="the group", json_schema_extra={"in_request": True})\n\n'
            '        @field_validator("name")\n'
            '        @classmethod\n'
            '        def without_group(cls, value):\n'
            '            return strip_words(value, ("group",))\n\n\n'
            'class PlanGenerator:\n    def steps(self, input):\n        return ["say the weight"]\n')
    command = _command(tmp_path, text, name="weigh")
    args, notes = pc.check_arguments(command, {"name": "Payroll"}, "Show the Payroll group")
    assert args["name"] == "Payroll"
    assert pc._validate_arguments(command, {"name": "Payroll group"}, "Show the Payroll group",
                                  notes=notes)["name"] == "Payroll"
    assert pc.check_input(command, {"name": "Payroll group"}, "Show the Payroll group") == {"name": "Payroll"}
    # a value the validator leaves as it is writes nothing back
    assert pc.check_input(command, {"name": "Payroll"}, "Show the Payroll group") == {}


def test_a_field_validator_that_cleans_a_value_changes_the_argument_of_an_execution_command():
    class Cleaned(BaseModel):
        name: str = Field(description="the group", json_schema_extra={"in_request": True})

        @field_validator("name")
        @classmethod
        def without_group(cls, value):
            return pc.strip_words(value, ("group",))

    instance = Cleaned(name="Payroll")
    is_valid, *_ = _exec("Show the Payroll group").validate_parameters(None, "Library/show_group", instance)
    assert is_valid and instance.name == "Payroll"

    instance = Cleaned.model_construct(name="Payroll group")
    is_valid, *_ = _exec("Show the Payroll group").validate_parameters(None, "Library/show_group", instance)
    assert is_valid and instance.name == "Payroll"


def test_a_refusing_validator_still_refuses_on_either_kind_and_a_model_validator_writes_back(tmp_path):
    class Rewrites(BaseModel):
        name: str = Field(description="the group", json_schema_extra={"in_request": True})

        @model_validator(mode="after")
        def upper_name(self):
            if self.name == "refuse":
                raise ValueError("refused by the validator")
            self.name = self.name.upper()
            return self

    # an execution command: the rewrite is written back, and a refusal refuses
    instance = Rewrites.model_construct(name="Payroll")
    is_valid, *_ = InputForParamExtraction(command="Show the Payroll group",
                                           input_for_param_extraction_class=None).validate_parameters(
        None, "Library/show_group", instance)
    assert is_valid and instance.name == "PAYROLL"
    refused = Rewrites.model_construct(name="refuse")
    is_valid, message, *_ = InputForParamExtraction(command="Show refuse", input_for_param_extraction_class=None
                                                    ).validate_parameters(None, "Library/show_group", refused)
    assert not is_valid and "refused by the validator" in message

    # a planning command: the same validator writes back into the arguments, and a refusal refuses
    option = pc.option_command(Rewrites, "Library/show_group")
    assert pc.check_input(option, {"name": "Payroll"}, "Show the Payroll group") == {"name": "PAYROLL"}
    with pytest.raises(PlanningCommandInputError, match="refused by the validator"):
        pc.check_input(option, {"name": "refuse"}, "Show the Payroll group")

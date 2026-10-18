"""Skills as Python skill files (``_skills/<name>/skill.py``): loading, expansion, argument
screen, validation hook, the planner path, and the SKILL.md migration path."""
from __future__ import annotations

import logging
import sys
import textwrap

import pytest
from dspy.utils import DummyLM

from fastworkflow import skill_commands as sc
from fastworkflow import workflow_agent
from fastworkflow.skill_commands import SkillCallError, SkillFormatError
from tests.test_text_planner_and_split import (  # noqa: F401 - workflow_env is a fixture
    RecordingTraceSink, _prompt, _session, _text_answer, workflow_env)

#: The example of the skill_commands module docstring, verbatim.
EXAMPLE = sc.__doc__.split("A skill file::\n\n", 1)[1].split("\n- ``Signature``", 1)[0]
EXAMPLE = textwrap.dedent(EXAMPLE)

CLOSE_OUT = '''
class Signature:
    """Close out a record: note what was done."""


def steps():
    return ["write a short summary", "say what is left open"]
'''


def _write(tmp_path, skills: dict[str, str], name: str = "skill.py") -> str:
    for skill, text in skills.items():
        folder = tmp_path / "_skills" / skill
        folder.mkdir(parents=True, exist_ok=True)
        (folder / name).write_text(text)
    return str(tmp_path)


def _example(tmp_path, text: str = EXAMPLE) -> sc.SkillCommand:
    folder = _write(tmp_path, {"show-record": text, "close-out": CLOSE_OUT})
    return workflow_agent.load_skill_commands(folder)["show-record"]


def _args(**overrides):
    return {"name": "Payroll", "kind": "group", "person": "", "include_risk": False, "count": 2, **overrides}


# ---------------------------------------------------------------- loading

def test_the_documented_example_loads_as_a_skill_with_params(tmp_path):
    command = _example(tmp_path)
    assert command.description == "Show one named group or unit and its members. One job per record."
    assert [(p.name, p.type, p.required, p.default) for p in command.params] == [
        ("name", "str", True, None), ("kind", "enum", True, None), ("person", "str", False, ""),
        ("include_risk", "bool", False, False), ("count", "int", False, 2)]
    name, kind, person, risk, count = command.params
    assert (name.strip, name.subject, name.desc) == (("group", "unit"), True,
                                                      "the record's name exactly as the request writes it")
    assert (kind.values, kind.verify, kind.means) == (
        ("group", "unit"), True, (("group", "a group of people"), ("unit", "an organisation unit")))
    assert person.subject and risk.verify and count.in_request
    assert command.plain_utterances == ("who is in the Payroll group", "is Ann Lee in Billing")
    assert command.maps == () and command.body == ()
    # the declared template is what the planner sees; it is not checked against steps()
    assert sc.template_steps(command)[0] == "<find_group or find_unit, by kind> query=<name>"
    assert "expands to:" in command.signature_text()


def test_load_skills_gives_the_description_and_template_and_a_plain_skill_its_steps(tmp_path):
    folder = _write(tmp_path, {"show-record": EXAMPLE, "close-out": CLOSE_OUT})
    skills = {name: (desc, steps) for name, desc, steps in workflow_agent.load_skills(folder)}
    assert skills["close-out"] == ("Close out a record: note what was done.",
                                   ("write a short summary", "say what is left open"))
    assert skills["show-record"][1][-3:] == ("repeat count times (n = 1 to count): open member <n>",
                                             "write a short summary", "say what is left open")
    assert workflow_agent.load_skill_commands(folder).keys() == {"show-record"}
    assert "- close-out: Close out a record" in workflow_agent._job_types_for(folder)


def test_steps_expand_the_arguments_and_includes(tmp_path):
    command = _example(tmp_path)
    assert sc.expand(command, _args(count=3)) == [
        "find_group query=Payroll", "list_members", "open member 1", "open member 2", "open member 3",
        "write a short summary", "say what is left open"]
    assert sc.expand(command, _args(kind="unit", person="Ann Lee", include_risk=True, count=0)) == [
        "find_unit query=Payroll", "list_members filter=Ann Lee", "assess_risk",
        "write a short summary", "say what is left open"]


def test_a_repeat_beyond_the_limit_and_a_failing_steps_are_skill_call_errors(tmp_path):
    command = _example(tmp_path)
    with pytest.raises(SkillCallError, match="more than"):
        sc.expand(command, _args(count=sc.REPEAT_LIMIT + 1))
    assert list(sc.repeat(3, 1)) == [2, 3] and list(sc.repeat(0)) == []
    broken = EXAMPLE.replace('out = [f"{FIND[input.kind]} query={input.name}"]', 'out = [FIND[input.name]]')
    with pytest.raises(SkillCallError, match="steps\\(\\) of show-record failed: KeyError"):
        sc.expand(_example(tmp_path / "broken", broken), _args())


def test_an_include_of_an_unknown_skill_fails_the_call(tmp_path):
    text = EXAMPLE.replace('out.append("@close-out")', 'out.append("@nowhere")')
    folder = _write(tmp_path, {"show-record": text, "close-out": CLOSE_OUT})
    with pytest.raises(SkillCallError, match="includes 'nowhere'"):
        sc.expand(workflow_agent.load_skill_commands(folder)["show-record"], _args())


def test_without_a_declared_template_steps_are_rendered_with_placeholder_inputs(tmp_path):
    text = EXAMPLE.replace("    template = [", "    template_ = [")
    folder = _write(tmp_path, {"show-record": text, "close-out": CLOSE_OUT})
    (_n, _d, steps), = [s for s in workflow_agent.load_skills(folder) if s[0] == "show-record"]
    # every enum value's steps, merged; a str reads <name>; a bool is true; an int its default
    assert steps == ("find_group query=<name>", "find_unit query=<name>", "list_members filter=<person>",
                     "assess_risk", "open member 1", "open member 2",
                     "write a short summary", "say what is left open")


def test_a_skill_file_without_input_may_take_the_empty_input(tmp_path):
    text = '"""x"""\nclass Signature:\n    """Plain."""\n\n\ndef steps(input):\n    return ["one", "@close-out"]\n'
    folder = _write(tmp_path, {"plain": text, "close-out": CLOSE_OUT})
    assert ("plain", "Plain.", ("one", "write a short summary", "say what is left open")) in \
        workflow_agent.load_skills(folder)


def test_a_skill_file_relative_import_and_a_folder_name_that_is_not_an_identifier(tmp_path):
    package = tmp_path / "acme_flow"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "texts.py").write_text('FIND = {"group": "find_group", "unit": "find_unit"}\n')
    text = EXAMPLE.replace('FIND = {"group": "find_group", "unit": "find_unit"}', "from ...texts import FIND")
    folder = _write(package, {"show-record": text, "close-out": CLOSE_OUT})
    assert sc.expand(workflow_agent.load_skill_commands(folder)["show-record"], _args(count=0))[0] == \
        "find_group query=Payroll"


def test_a_skill_file_that_cannot_be_imported_is_logged_and_skipped(tmp_path, caplog):
    folder = _write(tmp_path, {"broken": "import no_such_module_anywhere\n", "close-out": CLOSE_OUT})
    with caplog.at_level(logging.ERROR):
        skills = workflow_agent.load_skills(folder)
    assert [s[0] for s in skills] == ["close-out"]
    assert "skipping skill 'broken'" in caplog.text


@pytest.mark.parametrize("text, message", [
    ("def steps():\n    return ['a']\n", "needs a class Signature"),
    ("class Signature:\n    pass\n\ndef steps():\n    return ['a']\n", "needs a docstring"),
    ('class Signature:\n    """D."""\n', "needs a function steps"),
    ('class Signature:\n    """D."""\n\ndef steps(a, b):\n    return []\n', "takes one argument"),
    ('class Signature:\n    """D."""\n\ndef steps():\n    return "a step"\n', "must return a list of str"),
    ('class Signature:\n    """D."""\n\ndef steps():\n    return []\n', "returns no steps"),
])
def test_a_malformed_skill_file_fails_the_load(tmp_path, text, message):
    folder = _write(tmp_path, {"bad": text})
    with pytest.raises(SkillFormatError, match=message):
        workflow_agent.load_skills(folder)


FIELD = '''from typing import Literal, Optional
from pydantic import BaseModel, Field


class Signature:
    """D."""

    class Input(BaseModel):
        {field}


def steps(input):
    return ["step"]
'''


@pytest.mark.parametrize("field, message", [
    ("ratio: float = 0.5", "has type"),
    ("name: Optional[str] = None", "has type"),
    ("name: str = None", "default must be a value of its type"),
    ("flag: bool = True", "must default to false"),
    ("kind: Literal['a', 'b'] = 'c'", "not one of its values"),
    ("kind: Literal[1, 2]", "a Literal of text values only"),
    ("name: str = Field(json_schema_extra={'colour': 'red'})", "unknown key"),
    ("name: str = Field(json_schema_extra={'verify': True})", "only a bool, enum or subject parameter takes verify"),
    ("name: str = Field(json_schema_extra={'strip': 'group'})", "strip must be a list of words"),
    ("name: str = Field(json_schema_extra={'subject': 'yes'})", "subject must be true or false"),
    ("count: int = Field(default=1, json_schema_extra={'requires': 'nmae'})", "unknown parameter 'nmae'"),
    ("kind: Literal['a', 'b'] = Field(json_schema_extra={'means': {'c': 'x'}})", "'c' is not a value of it"),
    ("n: int = 1", "bad parameter name 'n'"),
])
def test_a_malformed_input_field_fails_the_load(tmp_path, field, message):
    folder = _write(tmp_path, {"bad": FIELD.format(field=field)})
    with pytest.raises(SkillFormatError, match=message):
        workflow_agent.load_skills(folder)


def test_steps_that_fail_on_placeholder_inputs_need_a_declared_template(tmp_path):
    text = FIELD.format(field="count: int = 2").replace('return ["step"]', 'return ["x"] * (10 // (input.count - 2))')
    with pytest.raises(SkillFormatError, match="declare Signature.template"):
        workflow_agent.load_skills(_write(tmp_path, {"bad": text}))


# ---------------------------------------------------------------- SKILL.md migration path

MD = "---\nparams:\n  name:\n    type: str\n---\nShow one record.\nfind_record query={name}\n"


def test_skill_md_is_read_with_a_deprecation_warning(tmp_path, caplog):
    folder = _write(tmp_path, {"show": MD}, name="SKILL.md")
    with caplog.at_level(logging.WARNING):
        command = workflow_agent.load_skill_commands(folder)["show"]
    assert "SKILL.md is deprecated" in caplog.text
    # a SKILL.md skill gets an Input model too, for the argument screen
    assert list(command.input_model.model_fields) == ["name"]


def test_a_skill_file_wins_over_a_skill_md_in_the_same_folder(tmp_path, caplog):
    _write(tmp_path, {"show-record": MD}, name="SKILL.md")
    with caplog.at_level(logging.WARNING):
        command = _example(tmp_path)
    assert command.path.endswith("skill.py")
    assert "is ignored" in caplog.text


def test_with_the_switch_off_a_skill_md_is_not_read(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "READ_SKILL_MD", False)
    folder = _write(tmp_path, {"show": MD}, name="SKILL.md")
    _write(tmp_path, {"close-out": CLOSE_OUT})
    assert [s[0] for s in workflow_agent.load_skills(folder)] == ["close-out"]


def test_a_skill_md_may_include_a_plain_skill_file_but_not_one_with_params(tmp_path):
    _write(tmp_path, {"close-out": CLOSE_OUT, "show-record": EXAMPLE})
    folder = _write(tmp_path, {"audit": MD + "@close-out\n"}, name="SKILL.md")
    command = workflow_agent.load_skill_commands(folder)["audit"]
    assert sc.expand(command, {"name": "Ann Lee"})[1:] == ["write a short summary", "say what is left open"]
    _write(tmp_path, {"outer": "Outer.\n@show-record\n"}, name="SKILL.md")
    workflow_agent._parsed_skills.cache_clear()
    with pytest.raises(SkillFormatError, match="which has params"):
        workflow_agent.load_skills(folder)


# ---------------------------------------------------------------- argument screen (shared with commands)

def test_the_command_parameter_screen_applies_to_skill_arguments(tmp_path):
    command = _example(tmp_path)
    request = "Who is in Payroll? Open the members."
    # a real value in angle brackets is unwrapped
    args, notes = sc.check_arguments(command, {"name": "<Payroll>", "kind": "group"}, request)
    assert args["name"] == "Payroll" and notes == ["name='<Payroll>' -> 'Payroll'"]
    # a placeholder: an optional argument takes its default, a required one fails the call
    args, notes = sc.check_arguments(command, {"name": "Payroll", "kind": "group", "person": "<person>"}, request)
    assert args["person"] == "" and notes == ["person='<person>' is a placeholder, not a value: default"]
    with pytest.raises(SkillCallError, match="is a placeholder"):
        sc.check_arguments(command, {"name": "desired-name", "kind": "group"}, request)
    # the extractor's "not given" number and a made-up extreme count as not given
    for made_up in (-sys.maxsize, 2**31 - 1):
        assert sc.check_arguments(command, {"name": "Payroll", "kind": "group", "count": made_up},
                                  request)[0]["count"] == 2
    # a declared bound (ge=0) refuses a negative count
    args, notes = sc.check_arguments(command, {"name": "Payroll", "kind": "group", "count": -1}, request)
    assert args["count"] == 2 and "is out of range (must be at least 0)" in notes[0]


def test_a_required_int_holding_the_not_given_marker_is_missing(tmp_path):
    text = FIELD.format(field="count: int")
    command = workflow_agent.load_skill_commands(_write(tmp_path, {"take": text}))["take"]
    with pytest.raises(SkillCallError, match="missing required argument count"):
        sc.check_arguments(command, {"count": -sys.maxsize}, "take some")


def test_the_screen_applies_to_skill_md_arguments_too(tmp_path):
    folder = _write(tmp_path, {"show": MD}, name="SKILL.md")
    command = workflow_agent.load_skill_commands(folder)["show"]
    with pytest.raises(SkillCallError, match="is a placeholder"):
        sc.check_arguments(command, {"name": "<name>"}, "Show the record <name>")


def test_a_unicode_dash_in_the_request_matches_a_plain_one(tmp_path):
    command = _example(tmp_path)
    request = "Who is in Pay‑roll?"
    args, notes = sc.check_arguments(command, {"name": "Pay-roll", "kind": "group"}, request)
    assert args["name"] == "Pay‑roll" and notes == ["name='Pay-roll' -> 'Pay‑roll'"]


# ---------------------------------------------------------------- validate_extracted_parameters

def test_the_validation_hook_can_refuse_or_correct_a_call(tmp_path):
    command = _example(tmp_path)
    with pytest.raises(SkillCallError, match="a unit is listed whole"):
        sc.validate_call(command, _args(kind="unit", person="Ann Lee"), "x")
    assert sc.validate_call(command, _args(), "x") == _args()

    seen = []

    def correcting(workflow, name, params, command_text=None):
        seen.append((workflow, name, command_text))
        params.name = params.name.upper()
        return True, ""

    from dataclasses import replace
    notes: list[str] = []
    args = sc.validate_call(replace(command, validate=correcting), _args(), "who is in Payroll", "WF", notes)
    assert args["name"] == "PAYROLL" and seen == [("WF", "show-record", "who is in Payroll")]
    assert notes == ["name='Payroll' -> 'PAYROLL' (validate_extracted_parameters)"]

    def raising(workflow, name, params):
        raise RuntimeError("backend down")
    with pytest.raises(SkillCallError, match="raised RuntimeError: backend down"):
        sc.validate_call(replace(command, validate=raising), _args(), "x")


def _typed(**fields):
    answer = {"reasoning": "r", "name": "Payroll", "kind": "group", "person": "", "include_risk": False,
              "count": 1, "steps_before": [], "steps_after": []}
    answer.update(fields)
    return answer


def _plan_skill(ctx, wf, lm, request, command):
    ctx.push_active_workflow(wf)
    try:
        return workflow_agent.build_query_with_next_steps(request, ctx, planner_lm=lm, plan_template="TEMPLATE TEXT",
                                                          skill_command=command)
    finally:
        ctx.pop_active_workflow()


def test_a_skill_file_is_planned_as_a_call_and_its_hook_refusal_falls_back(tmp_path, workflow_env):
    workflow_env()
    command = _example(tmp_path)
    request = "Who is in Payroll? Is Ann Lee one of them?"

    sink = RecordingTraceSink()
    ctx, wf = _session(sink)
    ctx._begin_turn(request)
    lm = DummyLM([_typed(person="Ann Lee")])
    result = _plan_skill(ctx, wf, lm, request, command)
    span = sink.named("fw.planner.plan")[-1]
    assert span.attributes["plan_source"] == "skill_call" and span.attributes["skill"] == "show-record"
    assert span.attributes["plan"].splitlines()[:3] == [
        "1. find_group query=Payroll", "2. list_members filter=Ann Lee", "3. open member 1"]
    assert "show-record(name: str, kind: group|unit" in _prompt(lm, 0)
    assert "Execute these next steps:\n1. find_group query=Payroll" in result

    sink = RecordingTraceSink()
    ctx, wf = _session(sink)
    ctx._begin_turn(request)
    lm = DummyLM([_typed(kind="unit", person="Ann Lee"), _text_answer()])
    _plan_skill(ctx, wf, lm, request, command)
    span = sink.named("fw.planner.plan")[-1]
    assert span.attributes["plan_source"] == "text"
    assert "a unit is listed whole" in span.attributes["skill_fallback"]
    assert "TEMPLATE TEXT" in _prompt(lm, 1)

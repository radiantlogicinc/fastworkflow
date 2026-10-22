"""Skills as Python skill files (``_skills/<name>.py``): loading, expansion, the
template text, argument screen, validation hook and the planner path."""
from __future__ import annotations

import logging
import re
import sys
import textwrap

import pytest
from dspy.utils import DummyLM

from fastworkflow import skill_commands as sc
from fastworkflow import skill_preview, workflow_agent
from fastworkflow.skill_commands import SkillCallError
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


def _write(tmp_path, skills: dict[str, str]) -> str:
    folder = tmp_path / "_skills"
    folder.mkdir(parents=True, exist_ok=True)
    for skill, text in skills.items():
        (folder / f"{skill}.py").write_text(text)
    return str(tmp_path)


def _example(tmp_path, text: str = EXAMPLE) -> sc.SkillCommand:
    folder = _write(tmp_path, {"show_record": text, "close_out": CLOSE_OUT})
    return workflow_agent.load_skill_commands(folder)["show_record"]


def _load_error(folder: str, skill: str = "bad") -> str:
    """Why *skill* does not load: the loader skips it (the workflow's other skills still load)
    and ``fastworkflow skill preview`` fails, naming it and its error."""
    assert skill not in {name for name, _, _ in workflow_agent.load_skills(folder)}
    with pytest.raises(skill_preview.PreviewError) as info:
        skill_preview.render_previews(folder)
    assert f"  {skill}: " in str(info.value)
    return str(info.value)


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
    # the template is steps() rendered on placeholder inputs; the planner sees it under "expands to:"
    assert command.template[0] == "<find_group or find_unit, by kind> query=<name>"
    assert "expands to:\n    1. <find_group or find_unit, by kind> query=<name>" in command.signature_text()


def test_load_skills_gives_the_description_and_template_and_a_plain_skill_its_steps(tmp_path):
    folder = _write(tmp_path, {"show_record": EXAMPLE, "close_out": CLOSE_OUT})
    skills = {name: (desc, steps) for name, desc, steps in workflow_agent.load_skills(folder)}
    assert skills["close_out"] == ("Close out a record: note what was done.",
                                   ("write a short summary", "say what is left open"))
    # steps() rendered on placeholder inputs
    assert skills["show_record"][1] == (
        "<find_group or find_unit, by kind> query=<name>",
        "list_members filter=<person> (only when person is given)",
        "list_members (only when person is empty)",
        "assess_risk (only when include_risk)",
        "open member 1", "open member 2")
    assert workflow_agent.load_skill_commands(folder).keys() == {"show_record"}
    assert "- close_out: Close out a record" in workflow_agent._job_types_for(folder)


def test_steps_expand_the_arguments(tmp_path):
    command = _example(tmp_path)
    assert sc.expand(command, _args(count=3)) == [
        "find_group query=Payroll", "list_members", "open member 1", "open member 2", "open member 3"]
    assert sc.expand(command, _args(kind="unit", person="Ann Lee", include_risk=True, count=0)) == [
        "find_unit query=Payroll", "list_members filter=Ann Lee", "assess_risk"]


def test_a_failing_steps_is_a_skill_call_error(tmp_path):
    # fails for one argument only (on the placeholder inputs it renders): the call fails, not the load
    broken = EXAMPLE.replace('out = [f"{FIND[input.kind]} query={input.name}"]',
                             'out = [FIND[input.name] if input.name == "Payroll" else input.name]')
    with pytest.raises(SkillCallError, match="steps\\(\\) of show_record failed: KeyError"):
        sc.expand(_example(tmp_path / "broken", broken), _args())


def test_a_declared_template_is_the_planner_template_as_written(tmp_path):
    declared = (
        '    template = [\n'
        '        "<find_group or find_unit, by kind> query=<name>",\n'
        '        "repeat count times (n = 1 to count): open member <n>",\n'
        '    ]\n\n'
        '    @staticmethod')
    text = EXAMPLE.replace("    @staticmethod", declared, 1)
    assert text != EXAMPLE
    command = _example(tmp_path, text)
    assert command.template == ("<find_group or find_unit, by kind> query=<name>",
                                "repeat count times (n = 1 to count): open member <n>")
    assert sc.expand(command, _args(count=1))[-1:] == ["open member 1"]


LABEL = '''from typing import Literal

from pydantic import BaseModel, Field


class Signature:
    """Put a label on one record or take it off."""

    class Input(BaseModel):
        name: str
        action: Literal["add", "remove"]
        list_all: bool = False
        detail: bool = False

        @staticmethod
        def validate_extracted_parameters(workflow, command, cmd_parameters, command_text=None):
            if not cmd_parameters.list_all:
                cmd_parameters.detail = False
            return True, ""


def steps(input):
    out = [f"open {input.name}"]
    if input.action == "add":
        out.append("add_label (change nothing else; the label must exist already)")
    else:
        out.append("remove_label (change nothing else)")
    if input.detail or not input.list_all:
        out.append("show_labels")
    if input.list_all:
        out.append("list_labels")
    return out
'''


def test_the_rendered_template_says_when_each_step_is_written(tmp_path):
    (_n, _d, steps), = workflow_agent.load_skills(_write(tmp_path, {"label": LABEL}))
    # different steps per enum value stay apart; a condition may be an "or"; a combination the
    # validation hook resets (detail without list_all) is not rendered
    assert steps == ("open <name>",
                     "add_label (change nothing else; the label must exist already) (only when action is add)",
                     "remove_label (change nothing else) (only when action is remove)",
                     "show_labels (only when not list_all or detail)",
                     "list_labels (only when list_all)")


def test_the_template_leaves_out_what_the_validation_hook_changes_or_refuses(tmp_path):
    # a hook refusing a unit with a person: the template renders a unit without one only
    refusing = EXAMPLE.replace(
        "if cmd_parameters.include_risk and cmd_parameters.person:",
        'if cmd_parameters.kind == "unit" and cmd_parameters.person:')
    assert refusing != EXAMPLE
    assert _example(tmp_path / "refusing", refusing).template[:4] == (
        "find_group query=<name> (only when kind is group)",
        "list_members filter=<person> (only when person is given)",
        "find_unit query=<name> (only when kind is unit)",
        "list_members (only when person is empty)")
    always = FIELD.format(field="name: str").replace(
        '    class Input', '    @staticmethod\n    def validate_extracted_parameters(workflow, command, p):\n'
        '        p.name = "x"\n        return True, ""\n\n    class Input')
    assert "changes or refuses every placeholder combination" in \
        _load_error(_write(tmp_path / "always", {"bad": always}))


def test_a_skill_file_without_input_may_take_the_empty_input(tmp_path):
    text = '"""x"""\nclass Signature:\n    """Plain."""\n\n\ndef steps(input):\n    return ["one", "two"]\n'
    folder = _write(tmp_path, {"plain": text})
    assert ("plain", "Plain.", ("one", "two")) in \
        workflow_agent.load_skills(folder)


def test_a_skill_file_imports_a_helper_of_its_folder_and_of_its_package(tmp_path):
    package = tmp_path / "acme_flow"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "texts.py").write_text('UNIT = "find_unit"\n')
    text = EXAMPLE.replace('FIND = {"group": "find_group", "unit": "find_unit"}',
                           "from ._kinds import FIND\nfrom ..texts import UNIT\nassert FIND['unit'] == UNIT")
    folder = _write(package, {"show_record": text, "close_out": CLOSE_OUT})
    (package / "_skills" / "_kinds.py").write_text('FIND = {"group": "find_group", "unit": "find_unit"}\n')
    # the helper _kinds.py is not a skill
    assert [s[0] for s in workflow_agent.load_skills(folder)] == ["close_out", "show_record"]
    assert sc.expand(workflow_agent.load_skill_commands(folder)["show_record"], _args(count=0))[0] == \
        "find_group query=Payroll"


def test_a_skill_file_that_cannot_be_imported_is_logged_and_skipped(tmp_path, caplog):
    folder = _write(tmp_path, {"broken": "import no_such_module_anywhere\n", "close_out": CLOSE_OUT})
    with caplog.at_level(logging.ERROR):
        skills = workflow_agent.load_skills(folder)
    assert [s[0] for s in skills] == ["close_out"]
    assert "skipping skill 'broken'" in caplog.text


def test_a_malformed_skill_file_is_logged_skipped_and_not_read_again(tmp_path, caplog, monkeypatch):
    folder = _write(tmp_path, {"bad": "def steps():\n    return ['a']\n", "close_out": CLOSE_OUT})
    with caplog.at_level(logging.ERROR):
        assert [s[0] for s in workflow_agent.load_skills(folder)] == ["close_out"]
    assert "skipping skill 'bad'" in caplog.text and "needs a class Signature" in caplog.text
    # the result is kept: a later turn does not import the skill files again
    monkeypatch.setattr(workflow_agent, "_load_skill_file", lambda *a: pytest.fail("skills read again"))
    assert [s[0] for s in workflow_agent.load_skills(folder)] == ["close_out"]
    assert workflow_agent.skill_utterances(folder) == {"close_out": ()}


@pytest.mark.parametrize("text, message", [
    ("def steps():\n    return ['a']\n", "needs a class Signature"),
    ("class Signature:\n    pass\n\ndef steps():\n    return ['a']\n", "needs a docstring"),
    ('class Signature:\n    """D."""\n', "needs a function steps"),
    ('class Signature:\n    """D."""\n\ndef steps(a, b):\n    return []\n', "takes one argument"),
    ('class Signature:\n    """D."""\n\ndef steps():\n    return "a step"\n', "must return a list of str"),
    ('class Signature:\n    """D."""\n\ndef steps():\n    return []\n', "returns no steps"),
])
def test_a_malformed_skill_file_fails_the_load(tmp_path, text, message):
    assert re.search(message, _load_error(_write(tmp_path, {"bad": text})))


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
    ("count: int = Field(default=1, json_schema_extra={'requires': 'name'})",
     "requires is not a skill option; .*Signature.validate_extracted_parameters"),
    ("kind: Literal['a', 'b'] = Field(json_schema_extra={'means': {'c': 'x'}})", "'c' is not a value of it"),
    ("reasoning: int = 1", "bad parameter name 'reasoning'"),
])
def test_a_malformed_input_field_fails_the_load(tmp_path, field, message):
    assert re.search(message, _load_error(_write(tmp_path, {"bad": FIELD.format(field=field)})))


def test_steps_that_fail_on_placeholder_inputs_need_a_declared_template(tmp_path):
    text = FIELD.format(field="count: int = 2").replace('return ["step"]', 'return ["x"] * (10 // (input.count - 2))')
    assert "declare Signature.template" in _load_error(_write(tmp_path, {"bad": text}))


# ---------------------------------------------------------------- files that are not skills

def test_only_the_python_files_of_the_skills_folder_not_starting_with_an_underscore_are_skills(tmp_path, caplog):
    folder = _write(tmp_path, {"close_out": CLOSE_OUT, "_helpers": "import no_such_module_anywhere\n"})
    skills = tmp_path / "_skills"
    (skills / "_split.py").write_text('class Signature:\n    """Split it my way."""\n')
    (skills / "notes.md").write_text("Not read.")
    (skills / "__pycache__").mkdir()
    # a subfolder is reserved for a skill context: logged and skipped, its files not loaded
    (skills / "Record").mkdir()
    (skills / "Record" / "show_record.py").write_text(EXAMPLE)
    with caplog.at_level(logging.WARNING):
        assert [s[0] for s in workflow_agent.load_skills(folder)] == ["close_out"]
    assert "Record: skill contexts (subfolders of _skills) are not loaded yet" in caplog.text
    assert "__pycache__" not in caplog.text and "_helpers" not in caplog.text


def test_a_skill_file_name_that_is_not_an_identifier_fails_the_load(tmp_path):
    folder = _write(tmp_path, {"close-out": CLOSE_OUT})
    assert "a Python identifier (snake_case): rename it to close_out.py" in _load_error(folder, "close-out")


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


def test_a_unicode_dash_in_the_request_matches_a_plain_one(tmp_path):
    command = _example(tmp_path)
    request = "Who is in Pay‑roll?"
    args, notes = sc.check_arguments(command, {"name": "Pay-roll", "kind": "group"}, request)
    assert args["name"] == "Pay‑roll" and notes == ["name='Pay-roll' -> 'Pay‑roll'"]


# ---------------------------------------------------------------- validate_extracted_parameters

def test_the_validation_hook_can_refuse_or_correct_a_call(tmp_path):
    command = _example(tmp_path)
    with pytest.raises(SkillCallError, match="the risk is assessed for the whole record"):
        sc.validate_call(command, _args(person="Ann Lee", include_risk=True), "x")
    assert sc.validate_call(command, _args(), "x") == _args()

    seen = []

    def correcting(workflow, name, params, command_text=None):
        seen.append((workflow, name, command_text))
        params.name = params.name.upper()
        return True, ""

    from dataclasses import replace
    notes: list[str] = []
    args = sc.validate_call(replace(command, validate=correcting), _args(), "who is in Payroll", "WF", notes)
    assert args["name"] == "PAYROLL" and seen == [("WF", "show_record", "who is in Payroll")]
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
    assert span.attributes["plan_source"] == "skill_call" and span.attributes["skill"] == "show_record"
    assert span.attributes["plan"].splitlines()[:3] == [
        "1. find_group query=Payroll", "2. list_members filter=Ann Lee", "3. open member 1"]
    assert "show_record(name: str, kind: group|unit" in _prompt(lm, 0)
    assert "Execute these next steps:\n1. find_group query=Payroll" in result

    sink = RecordingTraceSink()
    ctx, wf = _session(sink)
    ctx._begin_turn(request)
    # the hook's refusal is shown to the planner once; refused again, the text planner plans
    lm = DummyLM([_typed(person="Ann Lee", include_risk=True), _typed(person="Ann Lee", include_risk=True),
                  _text_answer()])
    _plan_skill(ctx, wf, lm, request, command)
    span = sink.named("fw.planner.plan")[-1]
    assert span.attributes["plan_source"] == "text"
    assert "the risk is assessed for the whole record" in span.attributes["skill_fallback"]
    assert "was refused: validate_extracted_parameters: the risk is assessed" in _prompt(lm, 1)
    assert "TEMPLATE TEXT" in _prompt(lm, 2)

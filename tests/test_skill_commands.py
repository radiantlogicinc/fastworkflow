"""Skills as parameterized commands: skill files with params, expansion, guards, planner path."""
from __future__ import annotations

import re

import pytest
from dspy.utils import DummyLM

from fastworkflow import skill_commands as sc
from fastworkflow import workflow_agent
from fastworkflow.skill_commands import SkillCallError
from tests.test_skill_files import _load_error
from tests.test_text_planner_and_split import (  # noqa: F401 - workflow_env is a fixture
    RecordingTraceSink, _fake_matcher, _plan, _prompt, _session, _text_answer, workflow_env)

LOOKUP = '''
from typing import Literal

from pydantic import BaseModel, Field

FIND = {"group": "find_group", "feed": "find_feed"}
COUNT_WORD = {1: "one", 2: "two"}


def listing(input):
    return "list_members" if input.kind == "group" else f"list_logins of {input.name}"


class Signature:
    """Who is in one named group or feed, or whether one person is. One job per record."""

    class Input(BaseModel):
        name: str = Field(description="the record's name exactly as the request writes it",
                          json_schema_extra={"strip": ["group", "feed"]})
        kind: Literal["group", "feed"] = Field(description="group, or feed (a source of logins)")
        person: str = Field(default="", description="one named person to check for; empty lists everyone")
        include_risk: bool = Field(default=False, description="the request asks how risky it is")
        count: int = Field(default=2, description="how many members to open")
        owner: str = Field(default="", description="the owner to compare the risk with")

    @staticmethod
    def validate_extracted_parameters(workflow, command, cmd_parameters, command_text=None):
        if not cmd_parameters.include_risk:   # an owner only for the risk comparison
            cmd_parameters.owner = ""
        return True, ""


def steps(input):
    out = ["open_directory", f"{FIND[input.kind]} query={input.name}"]
    out.append(f"{listing(input)} filter={input.person}" if input.person else listing(input))
    if input.include_risk:
        out.append(f"assess_risk{{owner={input.owner}}}")
    if input.kind == "group" and not (input.person or input.include_risk):
        out.append(f"open_group_by_uid with the uid of the {input.name} match")
    for n in range(2, input.count + 1):
        out.append(f"open member {n}")
    out.append(f"stop after the first {COUNT_WORD.get(input.count, input.count)} members")
    return out
'''


def _plain(description: str, steps: list[str]) -> str:
    """The skill file of a skill without params."""
    return f"class Signature:\n    __doc__ = {description!r}\n\n\ndef steps():\n    return {steps!r}\n"


def _write(tmp_path, skills: dict[str, str]) -> str:
    (tmp_path / "_skills").mkdir(parents=True, exist_ok=True)
    for name, text in skills.items():
        (tmp_path / "_skills" / f"{name}.py").write_text(text)
    return str(tmp_path)


def _command(tmp_path, text: str = LOOKUP, name: str = "lookup") -> sc.SkillCommand:
    return workflow_agent.load_skill_commands(_write(tmp_path, {name: text}))[name]


def _args(**overrides):
    return {"name": "Payroll", "kind": "group", "person": "", "include_risk": False, "count": 2,
            "owner": "", **overrides}


# ---------------------------------------------------------------- loading

def test_a_skill_file_gives_typed_params_and_the_description(tmp_path):
    folder = _write(tmp_path, {"lookup": LOOKUP})
    command = workflow_agent.load_skill_commands(folder)["lookup"]

    assert command.description == "Who is in one named group or feed, or whether one person is. One job per record."
    assert [(p.name, p.type, p.required, p.default) for p in command.params] == [
        ("name", "str", True, None), ("kind", "enum", True, None), ("person", "str", False, ""),
        ("include_risk", "bool", False, False), ("count", "int", False, 2), ("owner", "str", False, "")]
    assert command.param("kind").values == ("group", "feed")
    assert command.param("name").strip == ("group", "feed")
    assert callable(command.validate)
    # what the matcher sees is the description, as for a plain skill
    assert workflow_agent._job_types_for(folder) == f"- lookup: {command.description}"


def _with_utterances(text: str, utterances: list[str]) -> str:
    """``text`` (a skill file) with ``Signature.plain_utterances`` declared after the docstring."""
    head, sep, rest = text.partition("class Signature:\n")
    doc_end = rest.index('"""', rest.index('"""') + 3) + 3 if '"""' in rest else rest.index("\n") + 1
    return f"{head}{sep}{rest[:doc_end]}\n    plain_utterances = {utterances!r}\n{rest[doc_end:]}"


def test_the_matcher_lists_a_skills_plain_utterances_as_capped_examples(tmp_path):
    many = ["who belongs to this group", "is  someone\n in that feed", "list a group's members",
            "how many people are in it", "is it empty", "who joined last", "list its owners",
            "who manages it", "is a person in it", "a tenth one never shown"]
    folder = _write(tmp_path, {
        "lookup": _with_utterances(LOOKUP, many),
        "plain": _with_utterances(_plain("Close an account.", ["close it"]), ["shut this account down"]),
        "quiet": _plain("Say hello.", ["greet"]),
    })

    assert workflow_agent.skill_utterances(folder) == {
        "lookup": tuple(many), "plain": ("shut this account down",), "quiet": ()}
    lines = workflow_agent._matcher_job_types(folder).splitlines()
    description = workflow_agent.load_skill_commands(folder)["lookup"].description
    assert lines == [
        f'- lookup: {description} e.g. "who belongs to this group", "is someone in that feed", '
        '"list a group\'s members", "how many people are in it", "is it empty", "who joined last", '
        '"list its owners", "who manages it", "is a person in it"',
        '- plain: Close an account. e.g. "shut this account down"',
        "- quiet: Say hello.",
    ]
    assert workflow_agent.MATCHER_EXAMPLES == 9
    assert workflow_agent._matcher_job_types(folder, examples=1).splitlines()[0].endswith(
        'e.g. "who belongs to this group"')
    # the splitter's job types stay description-only
    assert workflow_agent._job_types_for(folder).splitlines()[1] == "- plain: Close an account."


def test_without_utterances_the_matcher_listing_is_the_description_only(tmp_path):
    folder = _write(tmp_path, {"lookup": LOOKUP, "quiet": _plain("Say hello.", ["greet"])})
    assert workflow_agent._matcher_job_types(folder) == workflow_agent._job_types_for(folder)


def test_match_skill_sends_the_listing_with_examples_to_the_matcher(tmp_path, monkeypatch):
    from types import SimpleNamespace

    folder = _write(tmp_path, {"plain": _with_utterances(_plain("Close an account.", ["close it"]),
                                                         ["shut this account down"])})
    seen = {}

    def predict(**inputs):
        seen.update(inputs)
        return SimpleNamespace(job=SimpleNamespace(value="plain", probabilities={"plain": 0.9}, confidence=0.9))

    monkeypatch.setattr(workflow_agent, "_skill_matcher", lambda f: (predict, DummyLM([])))
    assert workflow_agent.match_skill("please shut it", folder) == ("plain", ("close it",))
    assert seen["job_types"] == '- plain: Close an account. e.g. "shut this account down"'


def test_the_matcher_signature_mentions_examples_only_when_a_skill_has_them(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow_agent, "_jev_lm", lambda: object())
    with_examples = _write(tmp_path / "a", {"plain": _with_utterances(_plain("Close it.", ["x"]), ["shut it"])})
    without = _write(tmp_path / "b", {"plain": _plain("Close it.", ["x"])})

    def desc(folder):
        return workflow_agent._skill_matcher(folder)[0].signature.input_fields["job_types"].json_schema_extra["desc"]
    try:
        assert "example requests" in desc(with_examples)
        assert desc(without) == "the available job types, one per line as '- name: description'"
    finally:
        workflow_agent._skill_matcher.cache_clear()


def test_a_skill_with_params_offers_its_steps_as_template_text(tmp_path):
    folder = _write(tmp_path, {"lookup": LOOKUP})
    (_name, _desc, steps), = workflow_agent.load_skills(folder)
    # steps() on placeholder inputs: each step says when it is written, an enum's steps are
    # joined, a step whose str may be empty is one step; a repeat shows the int's default
    assert steps == (
        "open_directory",
        "<find_group or find_feed, by kind> query=<name>",
        "<list_members or list_logins of <name>, by kind> filter=<person> (only when person is given)",
        "<list_members or list_logins of <name>, by kind> (only when person is empty)",
        "assess_risk{owner=<owner>} (only when include_risk)",
        "open_group_by_uid with the uid of the <name> match (only when kind is group and person is empty "
        "and not include_risk)",
        "open member 2",
        "stop after the first two members",
    )


def test_a_skill_without_params_is_read_as_its_fixed_steps(tmp_path):
    # an "@<skill>" line is plain text: skills do not include other skills
    folder = _write(tmp_path, {"leaver": _plain("Someone is leaving. One job per person.",
                                                 ["find {them} [if asked]", "[repeat x]", "@close"]),
                               "close": _plain("Close an account.", ["close it"])})
    assert workflow_agent.load_skills(folder) == (
        ("close", "Close an account.", ("close it",)),
        ("leaver", "Someone is leaving. One job per person.", ("find {them} [if asked]", "[repeat x]", "@close")),
    )
    assert workflow_agent.load_skill_commands(folder) == {}


# ---------------------------------------------------------------- expansion

def test_expansion_substitutes_values_drops_false_conditions_and_unrolls_repeats(tmp_path):
    command = _command(tmp_path)
    assert sc.expand(command, _args(count=3)) == [
        "open_directory", "find_group query=Payroll", "list_members",
        "open_group_by_uid with the uid of the Payroll match",
        "open member 2", "open member 3", "stop after the first 3 members"]
    assert sc.expand(command, _args(kind="feed", name="NORTHFEED", person="Ann Lee", include_risk=True,
                                    owner="the customer", count=1)) == [
        "open_directory", "find_feed query=NORTHFEED", "list_logins of NORTHFEED filter=Ann Lee",
        "assess_risk{owner=the customer}", "stop after the first one members"]
    assert sc.number(["a", "b"]) == "1. a\n2. b"


# ---------------------------------------------------------------- guards

def test_a_string_argument_must_be_written_in_the_request_and_takes_its_spelling(tmp_path):
    command = _command(tmp_path)
    request = "Who is in the payroll group? Is Ann Lee one of them?"
    args, notes = sc.check_arguments(
        command, {"name": '"Payroll group"', "kind": "Group", "person": "ann lee", "count": "3"}, request)
    assert args == _args(name="payroll", person="Ann Lee", count=3)
    assert notes == ["name='\"Payroll group\"' -> 'payroll'", "person='ann lee' -> 'Ann Lee'"]


def test_strip_words_remove_a_kind_word_and_its_possessive(tmp_path):
    command = _command(tmp_path)
    args, _ = sc.check_arguments(command, {"name": "Ann Lee's feed", "kind": "feed"}, "Show Ann Lee's logins")
    assert args["name"] == "Ann Lee"
    # never down to nothing
    assert sc._strip_words("group", ("group",)) == "group"


def test_a_one_element_list_is_its_element_and_a_longer_list_is_not(tmp_path):
    command = _command(tmp_path)
    args, notes = sc.check_arguments(command, {"name": ["Payroll"], "kind": '["group"]'}, "Who is in Payroll?")
    assert args["name"] == "Payroll" and args["kind"] == "group"
    assert notes == ["name=['Payroll'] -> 'Payroll'"]
    with pytest.raises(sc.SkillInputError, match="not written in the request"):
        sc.check_arguments(command, {"name": '["Payroll", "Billing"]', "kind": "group"}, "Who is in Payroll?")


def test_an_argument_not_in_the_request_fails_a_required_one_and_resets_an_optional_one(tmp_path):
    command = _command(tmp_path)
    with pytest.raises(sc.SkillInputError, match="not written in the request"):
        sc.check_arguments(command, {"name": "Billing", "kind": "group"}, "Who is in Payroll?")
    args, notes = sc.check_arguments(command, {"name": "Payroll", "kind": "group", "person": "Bo Diaz"},
                                     "Who is in Payroll?")
    assert args["person"] == "" and "default" in notes[0]


def test_an_optional_argument_repeating_another_one_takes_its_default(tmp_path):
    args, _ = sc.check_arguments(_command(tmp_path), {"name": "Payroll", "kind": "group", "person": "payroll"},
                                 "Who is in Payroll?")
    assert args["person"] == ""


def test_a_dependent_argument_is_reset_by_the_validation_hook(tmp_path):
    command = _command(tmp_path)
    request = "Who is in Payroll, compared with the customer?"
    args, notes = sc.check_arguments(command, {"name": "Payroll", "kind": "group", "owner": "the customer"}, request)
    assert args["owner"] == "the customer"   # the argument check leaves it
    args = sc.validate_call(command, args, request, None, notes)
    assert args["owner"] == "" and notes[-1] == "owner='the customer' -> '' (validate_extracted_parameters)"
    args, _ = sc.check_arguments(command, {"name": "Payroll", "kind": "group", "owner": "the customer",
                                           "include_risk": True}, request)
    assert sc.validate_call(command, args, request)["owner"] == "the customer"


def test_booleans_default_false_and_bad_values_fail_the_call(tmp_path):
    command = _command(tmp_path)
    request = "Who is in Payroll?"
    assert sc.check_arguments(command, {"name": "Payroll", "kind": "group"}, request)[0]["include_risk"] is False
    assert sc.check_arguments(command, {"name": "Payroll", "kind": "group", "include_risk": "True"},
                              request)[0]["include_risk"] is True
    for bad in ({"kind": "group"}, {"name": "Payroll", "kind": "team"},
                {"name": "Payroll", "kind": "group", "count": "a few"},
                {"name": "Payroll", "kind": "group", "include_risk": "maybe"}):
        with pytest.raises(SkillCallError):
            sc.check_arguments(command, bad, request)


def test_extra_steps_the_expansion_already_runs_are_dropped():
    names = frozenset({"open_directory", "find_group", "list_members", "assess_risk", "show_tags"})
    expanded = ["open_directory", "find_group query=Payroll", "list_members"]
    kept, dropped = sc.filter_extra_steps(
        ["open_directory", "find_group Payroll; list_members", "lookup(name='Payroll')", "count them",
         "show_tags for Payroll", "find_group, then assess_risk", " "], expanded, names, "lookup")
    assert kept == ["show_tags for Payroll", "find_group, then assess_risk"]
    assert dropped == ["open_directory", "find_group Payroll; list_members", "lookup(name='Payroll')", "count them"]
    assert sc.command_names("Commands:\n- open_directory\n  Enter it.\n- find_group\n") == {"open_directory", "find_group"}


def test_the_typed_signature_has_one_field_per_param_plus_extra_steps(tmp_path):
    signature = sc.typed_signature(_command(tmp_path))
    assert list(signature.output_fields) == ["name", "kind", "person", "include_risk", "count", "owner",
                                             "steps_before", "steps_after"]
    assert "true only when the request explicitly asks" in signature.output_fields["include_risk"].json_schema_extra["desc"]
    assert "lookup(name: str, kind: group|feed, person: str = \"\"" in signature.instructions
    assert "expands to:" in signature.instructions


# ---------------------------------------------------------------- planner path

def _typed_answer(**fields):
    answer = {"reasoning": "r", "name": "Payroll", "kind": "group", "person": "", "include_risk": False,
              "count": 1, "owner": "", "steps_before": [], "steps_after": []}
    answer.update(fields)
    return answer


REQUEST = "Who is in Payroll, and how risky is it?"


def _plan_request(ctx, wf, lm, **kwargs):
    ctx.push_active_workflow(wf)
    try:
        return workflow_agent.build_query_with_next_steps(REQUEST, ctx, planner_lm=lm, **kwargs)
    finally:
        ctx.pop_active_workflow()


def _plan_span(sink):
    return sink.named("fw.planner.plan")[-1]


def test_a_matched_skill_with_params_is_planned_as_a_call_and_expanded(tmp_path, workflow_env):
    workflow_env()
    sink = RecordingTraceSink()
    ctx, wf = _session(sink)
    ctx._begin_turn(REQUEST)
    lm = DummyLM([_typed_answer(include_risk=True, steps_after=["open_directory", "say which is riskier"])])
    command = _command(tmp_path)

    result = _plan_request(ctx, wf, lm, plan_template="TEMPLATE", skill_command=command)

    assert len(lm.history) == 1
    assert "Skill command:" in _prompt(lm, 0) and "lookup(name: str, kind: group|feed" in _prompt(lm, 0)
    span = _plan_span(sink)
    assert span.attributes["plan_source"] == "skill_call"
    assert span.attributes["skill"] == "lookup"
    assert span.attributes["skill_args"] == _args(include_risk=True, count=1)
    assert span.attributes["skill_fallback"] is None
    assert span.attributes["plan"].splitlines()[:3] == ["1. open_directory", "2. find_group query=Payroll",
                                                        "3. list_members"]
    # the extra step names no available command of this workflow: dropped
    assert "riskier" not in span.attributes["plan"]
    assert result.startswith(REQUEST + "\n\nExecute these next steps:\n1. open_directory 2. find_group")


def test_a_failed_skill_call_falls_back_to_the_template_path(tmp_path, workflow_env):
    workflow_env()
    sink = RecordingTraceSink()
    ctx, wf = _session(sink)
    ctx._begin_turn("show my todo items")
    # "Billing" is not in the request: the guard refuses the call twice, the text planner plans
    lm = DummyLM([_typed_answer(name="Billing"), _typed_answer(name="Billing"), _text_answer()])

    _plan(ctx, wf, lm, plan_template="TEMPLATE TEXT", skill_command=_command(tmp_path))

    assert len(lm.history) == 3
    assert "TEMPLATE TEXT" in _prompt(lm, 2)
    span = _plan_span(sink)
    assert span.attributes["plan_source"] == "text"
    assert span.attributes["plan"] == "1. Show all todo items"
    assert span.attributes["skill"] == "lookup"
    assert span.attributes["skill_args"] is None
    assert "not written in the request" in span.attributes["skill_fallback"]


def test_an_exception_in_the_skill_call_falls_back_too(tmp_path, workflow_env, monkeypatch):
    workflow_env()
    sink = RecordingTraceSink()
    ctx, wf = _session(sink)
    ctx._begin_turn("show my todo items")

    def broken(*_args, **_kwargs):
        raise RuntimeError("provider down")

    monkeypatch.setattr(workflow_agent, "plan_skill_call", broken)
    lm = DummyLM([_text_answer()])
    _plan(ctx, wf, lm, plan_template="TEMPLATE TEXT", skill_command=_command(tmp_path))
    span = _plan_span(sink)
    assert span.attributes["plan_source"] == "text"
    assert span.attributes["skill_fallback"] == "RuntimeError: provider down"


def test_a_replan_ignores_the_skill_command(tmp_path, workflow_env):
    workflow_env()
    sink = RecordingTraceSink()
    ctx, wf = _session(sink)
    ctx._begin_turn("show my todo items")
    lm = DummyLM([_text_answer()])
    _plan(ctx, wf, lm, trace_trigger="parameter_extraction_error", skill_command=_command(tmp_path))
    replan = sink.named("fw.planner.replan")[-1]
    assert replan.attributes["plan_source"] == "text"
    assert replan.attributes["skill"] is None


def test_without_a_skill_command_the_planner_is_unchanged(workflow_env):
    workflow_env()
    sink = RecordingTraceSink()
    ctx, wf = _session(sink)
    ctx._begin_turn("show my todo items")
    lm = DummyLM([_text_answer()])
    result = _plan(ctx, wf, lm, plan_template="TEMPLATE TEXT")
    span = _plan_span(sink)
    assert (span.attributes["plan_source"], span.attributes["skill"], span.attributes["skill_args"],
            span.attributes["skill_fallback"]) == ("text", None, None, None)
    assert result == "show my todo items\n\nExecute these next steps:\n1. Show all todo items"


def test_one_match_gives_the_template_and_the_command_of_a_skill_with_params(tmp_path, monkeypatch):
    folder = _write(tmp_path, {"lookup": LOOKUP, "leaver": "Someone is leaving.\nfind them\n"})
    monkeypatch.setattr(workflow_agent, "_skill_matcher", lambda f: _fake_matcher("lookup", 0.9))
    template, command = workflow_agent.planner_skill_for("Who is in Payroll?", folder)
    assert command is workflow_agent.load_skill_commands(folder)["lookup"]
    assert template.startswith("A job like this is usually done as:\n- open_directory\n- <find_group or find_feed")
    # a skill without params: the template only, as before
    monkeypatch.setattr(workflow_agent, "_skill_matcher", lambda f: _fake_matcher("leaver", 0.9))
    assert workflow_agent.planner_skill_for("Ann Lee is leaving", folder) == (
        workflow_agent.plan_template_for("Ann Lee is leaving", folder), None)


def test_the_execution_context_passes_the_matched_command_except_to_a_continuation(workflow_env, monkeypatch):
    workflow_env()
    ctx, _wf = _session(RecordingTraceSink())
    sentinel = object()
    seen = []

    class Stop(Exception):
        pass

    def fake_build(query, _ctx, **kwargs):
        seen.append((query, kwargs["plan_template"], kwargs["skill_command"]))
        raise Stop

    monkeypatch.setattr(workflow_agent, "planner_skill_for", lambda text, folder: ("TEMPLATE", sentinel))
    monkeypatch.setattr(workflow_agent, "build_query_with_next_steps", fake_build)
    with pytest.raises(Stop):
        ctx._call_agent_for_query("Who is in Payroll?", "Who is in Payroll?", template_text="Who is in Payroll?")
    with pytest.raises(Stop):
        ctx._call_agent_for_query("Who is in Payroll?\n\nAlready done in this sub-task: ...", "x",
                                  template_text="Who is in Payroll?")
    assert seen == [("Who is in Payroll?", "TEMPLATE", sentinel),
                    ("Who is in Payroll?\n\nAlready done in this sub-task: ...", "TEMPLATE", None)]


# ---------------------------------------------------------------- checks against the request (verify, in_request)

CHECKED = LOOKUP.replace(
    '''kind: Literal["group", "feed"] = Field(description="group, or feed (a source of logins)")''',
    '''kind: Literal["group", "feed"] = Field(
            description="group, or feed (a source of logins)",
            json_schema_extra={"verify": True, "means": {"group": "a group of people",
                                                         "feed": "a feed that logins arrive on"}})''',
).replace(
    '''include_risk: bool = Field(default=False, description="the request asks how risky it is")''',
    '''include_risk: bool = Field(default=False, description="the request asks how risky it is",
                                    json_schema_extra={"verify": True})''',
).replace(
    '''count: int = Field(default=2, description="how many members to open")''',
    '''count: int = Field(default=2, description="how many members to open",
                             json_schema_extra={"in_request": True})''',
)


def test_verify_means_and_in_request_are_read_from_the_input_fields(tmp_path):
    assert CHECKED.count("json_schema_extra") == LOOKUP.count("json_schema_extra") + 3
    command = _command(tmp_path, CHECKED)
    kind, risk, count = command.param("kind"), command.param("include_risk"), command.param("count")
    assert (kind.verify, kind.means) == (True, (("group", "a group of people"), ("feed", "a feed that logins arrive on")))
    assert (risk.verify, risk.means, risk.in_request) == (True, (), False)
    assert (count.verify, count.in_request) == (False, True)
    # without the keys, nothing changes
    assert not any(p.verify or p.means or p.in_request for p in _command(tmp_path / "plain").params)
    assert sc._check_options(kind) == (("group", "a group of people"), ("feed", "a feed that logins arrive on"))
    assert [v for v, _ in sc._check_options(risk)] == ["true", "false"]


CHECK_FIELD = '''from typing import Literal
from pydantic import BaseModel, Field


class Signature:
    """One."""

    class Input(BaseModel):
        {field}


def steps(input):
    return ["step"]
'''


@pytest.mark.parametrize("field, message", [
    ("name: str = Field(json_schema_extra={'verify': True})", "only a bool, enum or subject parameter takes verify"),
    ("flag: bool = Field(default=False, json_schema_extra={'subject': True})", "only a str parameter takes subject"),
    ("flag: bool = Field(default=False, json_schema_extra={'verify': 'yes'})", "verify must be true or false"),
    ("flag: bool = Field(default=False, json_schema_extra={'means': {'true': 'asked'}})",
     "means of 'flag' must be a block of its enum values"),
    ("kind: Literal['a', 'b'] = Field(json_schema_extra={'means': {'c': 'x'}})", "'c' is not a value of it"),
    ("kind: Literal['a', 'b'] = Field(json_schema_extra={'in_request': True})", "only an int parameter takes in_request"),
])
def test_a_malformed_check_fails_the_load(tmp_path, field, message):
    assert re.search(message, _load_error(_write(tmp_path, {"bad": CHECK_FIELD.replace("{field}", field)})))


def test_a_sure_check_replaces_the_argument_and_an_unsure_one_does_not(tmp_path):
    command = _command(tmp_path, CHECKED)
    request = "Who is in Payroll?"
    raw = {"name": "Payroll", "kind": "feed", "include_risk": True}
    args, notes = sc.check_arguments(command, raw, request,
                                     {"kind": ("group", 0.9), "include_risk": (False, 0.95)})
    assert (args["kind"], args["include_risk"]) == ("group", False)
    assert notes == ["kind='feed' -> 'group' (check 0.90)", "include_risk=True -> False (check 0.95)"]
    args, notes = sc.check_arguments(command, raw, request, {"kind": ("group", sc.VERIFY_THRESHOLD - 0.01)})
    assert (args["kind"], notes) == ("feed", [])
    # a sure check stands in for a missing or invalid enum; a param without verify is never checked
    assert sc.check_arguments(command, {"name": "Payroll", "kind": "team"}, request,
                              {"kind": ("group", 0.8)})[0]["kind"] == "group"
    assert sc.check_arguments(_command(tmp_path / "plain"), raw, request,
                              {"kind": ("group", 1.0)})[0]["kind"] == "feed"


def test_a_dependent_argument_follows_the_checked_value(tmp_path):
    command = _command(tmp_path, CHECKED)
    raw = {"name": "Payroll", "kind": "group", "include_risk": True, "owner": "the customer"}
    request = "Who is in Payroll, next to the customer?"
    args, notes = sc.check_arguments(command, raw, request, {"include_risk": (False, 0.9)})
    args = sc.validate_call(command, args, request, None, notes)
    assert (args["include_risk"], args["owner"]) == (False, "")
    assert notes[-1] == "owner='the customer' -> '' (validate_extracted_parameters)"


def test_an_in_request_count_must_be_a_number_the_request_writes(tmp_path):
    command = _command(tmp_path, CHECKED)
    raw = {"name": "Payroll 2", "kind": "group", "count": 3}
    # "the first two": 3 is not written; the number inside the name does not count
    args, notes = sc.check_arguments(command, raw, "Open the first two members of Payroll 2.")
    assert args["count"] == 2 and notes == ["count=3 -> 2 (the number the request writes)"]
    assert sc.check_arguments(command, raw, "Open 3 members of Payroll 2.")[0]["count"] == 3
    # no number written: the argument (or its default) stands
    assert sc.check_arguments(command, {"name": "Payroll 2", "kind": "group"}, "Open members of Payroll 2.")[0][
        "count"] == 2
    # several numbers, none of them the argument: the default when it is written, else the call fails
    assert sc.check_arguments(command, raw, "Open two of the five members of Payroll 2.")[0]["count"] == 2
    with pytest.raises(SkillCallError, match="not a number the request writes"):
        sc.check_arguments(command, raw, "Open four of the five members of Payroll 2.")


def test_extra_steps_lose_placeholders_and_invented_ids():
    names = frozenset({"open_directory", "find_group", "list_members", "list_logins", "show_tags", "open_group_by_uid"})
    expanded = ["open_directory", "find_group query=Payroll", "list_members", "open_group_by_uid with its uid"]
    notes: list[str] = []
    kept, dropped = sc.filter_extra_steps(
        ["list_logins login_uid=<Ann Lee's identity uid>", "show_tags <tag>", "open_group_by_uid(<uid>)",
         "list_logins login_uid=3f2a9c0d4b5e6f708192a3b4c5d6e7f8", "show_tags for 0a1b2c3d4e5f60718293a4b5c6d7e8f9",
         "<the next step>"],
        expanded, names, "lookup", request="Show the logins and tags of 0a1b2c3d4e5f60718293a4b5c6d7e8f9", notes=notes)
    # a placeholder goes, the command stays; a uid the request writes stays
    assert kept == ["list_logins", "show_tags", "show_tags for 0a1b2c3d4e5f60718293a4b5c6d7e8f9"]
    # what the expansion runs once its placeholder is gone, an invented uid, nothing left
    assert dropped == ["open_group_by_uid(<uid>)", "list_logins login_uid=3f2a9c0d4b5e6f708192a3b4c5d6e7f8",
                       "<the next step>"]
    assert notes[0] == "extra step \"list_logins login_uid=<Ann Lee's identity uid>\" -> 'list_logins' (placeholder removed)"


def test_check_verdicts_runs_each_verified_param_and_skips_a_failed_one(tmp_path, monkeypatch):
    command = _command(tmp_path, CHECKED)
    seen = []

    def fake(lm, cmd, p, request):
        seen.append(p.name)
        if p.name == "kind":
            raise RuntimeError("classifier down")
        return True, 0.9

    monkeypatch.setattr(sc, "_check_one", fake)
    assert sc.check_verdicts(object(), command, "Who is in Payroll?") == {"include_risk": (True, 0.9)}
    assert sorted(seen) == ["include_risk", "kind"]
    assert sc.check_verdicts(None, command, "Who is in Payroll?") == {}


def test_the_planner_path_checks_verified_params_with_jev(tmp_path, workflow_env, monkeypatch):
    workflow_env()
    sink = RecordingTraceSink()
    ctx, wf = _session(sink)
    ctx._begin_turn(REQUEST)
    jev = object()
    calls = []

    def fake_verdicts(lm, command, request):
        calls.append((lm, command.name, request))
        return {"include_risk": (True, 0.92)}

    monkeypatch.setattr(workflow_agent, "_jev_lm", lambda: jev)
    monkeypatch.setattr(sc, "check_verdicts", fake_verdicts)
    lm = DummyLM([_typed_answer(include_risk=False)])
    _plan_request(ctx, wf, lm, plan_template="TEMPLATE", skill_command=_command(tmp_path, CHECKED))
    assert calls == [(jev, "lookup", REQUEST)]
    span = _plan_span(sink)
    assert span.attributes["plan_source"] == "skill_call"
    assert span.attributes["skill_args"]["include_risk"] is True
    assert "assess_risk" in span.attributes["plan"]


# ---------------------------------------------------------------- record names (subject), identifiers, extra steps

NAMED = '''
from pydantic import BaseModel, Field


class Signature:
    """A person asks for a right."""

    class Input(BaseModel):
        right: str = Field(description="the right's name", json_schema_extra={"strip": ["right"], "subject": True})
        person: str = Field(default="", description="the person asking for it",
                            json_schema_extra={"subject": True, "verify": True})
        note: str = Field(default="", description="free text")


def steps(input):
    out = [f"find_right query={input.right}"]
    if input.person:
        out.append(f"find_person query={input.person}")
    return out
'''


def _named(tmp_path, identifiers=()):
    from dataclasses import replace
    return replace(_command(tmp_path, NAMED, "ask"), identifiers=identifiers)


def test_subject_is_read_from_the_input_fields(tmp_path):
    command = _named(tmp_path)
    assert [p.subject for p in command.params] == [True, True, False]
    assert command.param("person").verify


def test_a_subject_argument_must_name_a_record_of_the_request(tmp_path):
    command = _named(tmp_path)
    request = "Before I approve anything on the right Payroll_Approver 2: how risky is it?"
    # "I" is written in the request but names no record: the optional argument takes its default
    args, notes = sc.check_arguments(command, {"right": "Payroll_Approver 2", "person": "I"}, request)
    assert args["person"] == "" and "names no record" in notes[0]
    # a required one fails the call (the planner falls back to the template)
    with pytest.raises(SkillCallError, match="names no record"):
        sc.check_arguments(command, {"right": "approve", "person": ""}, request)
    # a name the request writes whole stands; so does a capitalised name the subject finder misses
    assert sc.check_arguments(command, {"right": "Payroll_Approver 2", "person": "Ann Lee"},
                              "Ann Lee asks for the right Payroll_Approver 2.")[0]["person"] == "Ann Lee"
    assert sc.check_arguments(command, {"right": "Billing"}, "Billing: who may approve it?")[0]["right"] == "Billing"
    # a plain str parameter is not checked
    assert sc.check_arguments(command, {"right": "Billing", "note": "approve"},
                              "Billing: may I approve it?")[0]["note"] == "approve"


def test_a_subject_cut_short_becomes_the_whole_name_and_a_part_of_a_compound_is_refused(tmp_path):
    command = _named(tmp_path)
    request = "Audit the right Payroll_Administrator wherever it appears, for Ann Lee."
    args, notes = sc.check_arguments(command, {"right": "Payroll_Admin", "person": "Ann Lee"}, request)
    assert args["right"] == "Payroll_Administrator" and notes == ["right='Payroll_Admin' -> 'Payroll_Administrator'"]
    # a first name only is "Ann Lee" cut short at a space: completed
    args, notes = sc.check_arguments(command, {"right": "Payroll_Administrator", "person": "Ann"}, request)
    assert args["person"] == "Ann Lee"
    # a part cut at "_" may be a record of its own ("Payroll" of "Payroll_Approver"): refused
    with pytest.raises(SkillCallError, match="names no record"):
        sc.check_arguments(command, {"right": "Payroll"}, "Who holds Payroll_Approver?")
    # the tail of a name opening a sentence stands for it ("Treat Ledger" -> "Ledger")
    assert sc.check_arguments(command, {"right": "Ledger"},
                              "Treat Ledger as a right: who holds it?")[0]["right"] == "Ledger"


def test_a_subject_written_in_lower_case_stands_when_it_is_a_whole_name(tmp_path):
    command = _named(tmp_path)
    # a full name in lower case, between common words, stands
    args, notes = sc.check_arguments(command, {"right": "Payroll", "person": "ann lee"}, "is ann lee in Payroll?")
    assert args["person"] == "ann lee" and notes == []
    assert sc.check_arguments(command, {"right": "Billing", "person": "bo diaz"},
                              "may bo diaz have Billing, and who else has it?")[0]["person"] == "bo diaz"
    # a part of a longer lower-case name is refused ("ann lee" of "ann lee smith", "lee smith")
    request = "is ann lee smith in Payroll?"
    for part in ("ann lee", "lee smith"):
        args, notes = sc.check_arguments(command, {"right": "Payroll", "person": part}, request)
        assert args["person"] == "" and "names no record" in notes[0]
    assert sc.check_arguments(command, {"right": "Billing", "person": "ann lee"},
                              "what does ann lee's login hold on Billing?")[0]["person"] == "ann lee"
    # a finite verb or a placing preposition next to it is no part of the name; a part of a longer one still is
    for request in ("does ann lee hold Payroll?", "is ann lee among the holders of Payroll?",
                    "ann lee leaves on friday: who else has Payroll?"):
        assert sc.check_arguments(command, {"right": "Payroll", "person": "ann lee"}, request)[0]["person"] == "ann lee"
    for request, part in (("does ann lee smith hold Payroll?", "ann lee"), ("does ann lee smith hold Payroll?", "lee smith"),
                          ("is ann lee smith among the holders of Payroll?", "ann lee"),
                          ("is ann lee smith among the holders of Payroll?", "lee smith"),
                          ("which rights does mary ann lee hold on Payroll?", "ann lee")):
        args, notes = sc.check_arguments(command, {"right": "Payroll", "person": part}, request)
        assert args["person"] == "" and "names no record" in notes[0], (request, part)
    # a word other than a verb or preposition after it may still continue it ("ann lee violates" is refused)
    args, notes = sc.check_arguments(command, {"right": "Payroll", "person": "ann lee"}, "does ann lee violate Payroll?")
    assert args["person"] == "" and "names no record" in notes[0]
    # one of the command's own enum values right before it, with a clause word after it, is no part of it either
    from dataclasses import replace
    kinded = replace(command, params=command.params + (sc.Param("kind", "enum", ("right", "group"), "right"),))
    assert sc.check_arguments(kinded, {"right": "Payroll", "person": "ann lee"},
                              "list every right ann lee holds besides Payroll")[0]["person"] == "ann lee"
    assert sc.check_arguments(command, {"right": "Payroll", "person": "ann lee"},
                              "list every right ann lee holds besides Payroll")[0]["person"] == ""
    for request, part in (("list every right ann lee smith holds besides Payroll", "lee smith"),
                          ("list every right ann lee smith holds besides Payroll", "ann lee"),
                          ("show the right ann lee smith next to Payroll", "ann lee"),
                          ("show the right ann lee next to Payroll", "ann lee")):
        args, notes = sc.check_arguments(kinded, {"right": "Payroll", "person": part}, request)
        assert args["person"] == "" and "names no record" in notes[0], (request, part)
    # one name of a chain joined by "and" that ends on a company word is a part of that name
    request = "which controls does henderson ramirez and lewis and sons trip on Billing?"
    for part in ("henderson ramirez", "lewis and sons"):
        args, notes = sc.check_arguments(command, {"right": "Billing", "person": part}, request)
        assert args["person"] == "" and "names no record" in notes[0], part
    assert sc.check_arguments(command, {"right": "Billing", "person": "bo diaz"},
                              "may ann lee and bo diaz have Billing?")[0]["person"] == "bo diaz"
    # a single lower-case word reads as a common noun: refused, as are common words
    for person, request in (("roster", "who is on the roster for Payroll?"),
                            ("user", "which user am I acting as on Billing?"),
                            ("ann", "is ann lee in Payroll?"),
                            ("full list", "give me the full list for Billing")):
        args, notes = sc.check_arguments(command, {"right": request.split()[-1].strip("?"), "person": person}, request)
        assert args["person"] == "" and "names no record" in notes[0], person
    with pytest.raises(SkillCallError, match="names no record"):
        sc.check_arguments(command, {"right": "approve"},
                           "Before I approve anything on the right Payroll_Approver 2: how risky is it?")
    # a capitalised name cut short still becomes the whole subject; its sentence-opening tail stands
    assert sc.check_arguments(command, {"right": "Billing", "person": "Ann"},
                              "May Ann Lee have Billing?")[0]["person"] == "Ann Lee"
    assert sc.check_arguments(command, {"right": "Ledger"},
                              "Treat Ledger as a right: who holds it?")[0]["right"] == "Ledger"


def test_an_argument_inside_an_identifier_the_request_writes_becomes_the_whole_identifier(tmp_path):
    import re
    command = _named(tmp_path, (re.compile(r"\bacct_[0-9]+"),))
    request = "Who may hold acct_4421 or acct_9972?"
    args, notes = sc.check_arguments(command, {"right": "4421"}, request)
    assert args["right"] == "acct_4421" and notes == ["right='4421' -> 'acct_4421'"]
    assert sc.check_arguments(command, {"right": "acct_9972"}, request)[0]["right"] == "acct_9972"
    # a part of two identifiers is not one of them: refused
    with pytest.raises(SkillCallError, match="more than one identifier"):
        sc.check_arguments(command, {"right": "acct_"}, request)


def test_a_sure_check_of_a_subject_chooses_among_the_request_subjects_or_none(tmp_path):
    command = _named(tmp_path)
    request = "Ann Lee and Bo Diaz: may Bo Diaz have the right Billing?"
    raw = {"right": "Billing", "person": "Ann Lee"}
    args, notes = sc.check_arguments(command, raw, request, {"person": ("Bo Diaz", 0.9)})
    assert args["person"] == "Bo Diaz" and notes == ["person='Ann Lee' -> 'Bo Diaz' (check 0.90)"]
    assert sc.check_arguments(command, raw, request, {"person": (sc.NO_SUBJECT, 0.9)})[0]["person"] == ""
    assert sc.check_arguments(command, raw, request, {"person": ("Bo Diaz", 0.5)})[0]["person"] == "Ann Lee"
    options = sc._check_options(command.param("person"), ["Ann Lee", "Bo Diaz"])
    assert [v for v, _ in options] == ["Ann Lee", "Bo Diaz", sc.NO_SUBJECT]


def test_a_subject_check_offers_the_tail_of_a_name_opening_a_sentence(tmp_path, monkeypatch):
    import dspy
    command = _named(tmp_path)
    request = "Treat Ledger as a right: who holds it?"
    # the name opening the sentence holds its first word; offered alone, the check could only
    # choose "Treat Ledger" or no record, and a sure "none" refused the planner's "Ledger"
    assert sc._request_subjects(command, request) == ["Treat Ledger"]
    assert sc._check_subjects(command, request) == ["Treat Ledger", "Ledger"]
    assert sc._check_subjects(command, "Before Ann Lee asks. Treat Ledger as a right") == [
        "Treat Ledger", "Ledger", "Ann Lee"]
    # a name inside a sentence is offered as it is
    assert sc._check_subjects(command, "May Bo Diaz hold Ledger?") == ["Bo Diaz", "Ledger"]

    from types import SimpleNamespace
    offered = []
    options = sc._check_options

    def recorded(p, subjects=()):
        offered.extend(v for v, _ in options(p, subjects))
        return options(p, subjects)

    class Predict:
        def __init__(self, signature):
            pass

        def __call__(self, **inputs):
            choice = SimpleNamespace(value="Ledger", probabilities={"Ledger": 0.9}, confidence=0.9)
            return SimpleNamespace(value=choice)
    monkeypatch.setattr(sc, "_check_options", recorded)
    monkeypatch.setattr(dspy, "Predict", Predict)
    assert sc._check_one(object(), command, command.param("right"), request) == ("Ledger", 0.9)
    assert offered == ["Treat Ledger", "Ledger", sc.NO_SUBJECT]
    args, notes = sc.check_arguments(command, {"right": "Ledger"}, request, {"right": ("Ledger", 0.9)})
    assert args["right"] == "Ledger" and notes == []


def test_check_verdicts_skips_a_subject_check_when_the_request_names_nothing(tmp_path, monkeypatch):
    command = _named(tmp_path)
    monkeypatch.setattr(sc, "_check_one", lambda lm, cmd, p, request: ("Ann Lee", 0.9))
    assert sc.check_verdicts(object(), command, "may ann have it?") == {}
    assert sc.check_verdicts(object(), command, "May Ann Lee have Billing?") == {"person": ("Ann Lee", 0.9)}


def test_load_skill_commands_gives_each_command_the_workflow_identifiers(tmp_path):
    folder = _write(tmp_path, {"ask": NAMED})
    (tmp_path / "_skills" / "_split.py").write_text(
        "import re\n\n# codes\nidentifier_patterns = [re.compile(r'\\bacct_[0-9]+')]\n")
    command = workflow_agent.load_skill_commands(folder)["ask"]
    assert [p.pattern for p in command.identifiers] == [r"\bacct_[0-9]+"]


def test_an_extra_step_is_kept_only_for_a_part_of_the_request_the_expansion_leaves_open():
    names = frozenset({"open_directory", "find_group", "list_members", "show_tags", "show_nesting", "list_controls",
                       "open_account_by_uid", "browse_catalog"})
    expanded = ["open_directory", "find_group query=Payroll", "list_members (its members and the controls it trips)"]
    request = "Who is in Payroll, what is it nested under, and which tags does it carry? It trips controls."
    kept, dropped = sc.filter_extra_steps(
        ["show_tags", "show_nesting", "list_controls", "open_account_by_uid", "browse_catalog"],
        expanded, names, "lookup", request=request)
    # tags and nesting are asked and the expansion says nothing of them; controls it covers already
    assert kept == ["show_tags", "show_nesting"]
    assert dropped == ["list_controls", "open_account_by_uid", "browse_catalog"]
    assert sc._same_stem("holders", "hold") and sc._same_stem("sensitivity", "sensitive")
    assert not sc._same_stem("control", "contractor")


def test_an_extra_step_is_kept_for_a_synonym_its_command_says_of_itself(monkeypatch):
    catalog = ("Commands available in the current context (global):\n"
               "- list_bills\n  List the customer's bills. Not for payments.\n"
               "  Before executing list_bills, navigate to Customer as follows: open_customer\n"
               "- list_orders\n  List the customer's orders.\n"
               "- show_weather\n  Show the weather.\n")
    texts = sc.command_texts(catalog)
    # a single text (no utterances): its first sentence only, cautions and navigation lines aside
    assert texts["list_bills"] == "bills customer"
    utterances = {"list_bills": ["show every invoice this customer has", "which invoices are still open",
                                 "check whether the bills are paid"],
                  "show_weather": ["is it raining", "show the forecast"]}
    assert sc.command_descriptions(catalog)["list_bills"] == "List the customer's bills."
    names = sc.command_names(catalog) | {"open_customer", "find_customer"}
    expanded = ["find_customer query=Ann Lee", "open_customer with its uid", "list_orders"]
    request = "For the customer Ann Lee, list every order and every invoice, and check the forecast."

    monkeypatch.setattr(sc, "command_utterances", lambda folder: utterances)
    texts = sc.command_texts(catalog, "/some/workflow")
    # "invoices" recurs in two of list_bills's texts; "check"/"whether" in one only
    assert "invoices" in texts["list_bills"].split() and "check" not in texts["list_bills"].split()
    extra = ["list_bills", "show_weather"]
    # the name alone: no word of list_bills or show_weather shares a stem with the request
    assert sc.filter_extra_steps(extra, expanded, names, "lookup", request=request) == ([], extra)
    # with the texts: invoices keeps list_bills; "forecast" is said once only, show_weather still goes
    assert sc.filter_extra_steps(extra, expanded, names, "lookup", request=request, texts=texts) == (
        ["list_bills"], ["show_weather"])
    # an unrelated request keeps nothing
    assert sc.filter_extra_steps(extra, expanded, names, "lookup", request="List Ann Lee's orders.",
                                 texts=texts) == ([], extra)


def test_command_utterances_never_raises_on_an_unreadable_workflow(tmp_path):
    assert sc.command_utterances(str(tmp_path / "missing")) == {}


# ---------------------------------------------------------------- phased skills (skill_phases)

PHASED_LOOKUP = LOOKUP + '''

from fastworkflow.skill_phases import Phase


class Found(BaseModel):
    uids: list[str] = Field(json_schema_extra={"from": "find_group.uids"})


def plan(input):
    flat = steps(input)
    return [Phase("find", lambda: flat[:2], output=Found),
            Phase("rest", lambda find: flat[2:])]
'''


def test_a_skill_file_with_plan_loads_and_keeps_its_flat_steps(tmp_path, caplog):
    command = _command(tmp_path, PHASED_LOOKUP)
    assert command.plan_fn is not None
    assert [m.__name__ for m in command.phase_outputs] == ["Found"]
    # The template (what matching and the fallback read) is the flat steps, as before.
    assert command.template == _command(tmp_path / "flat", LOOKUP).template


def test_a_phased_skill_call_hands_its_phases_to_the_runner(tmp_path, workflow_env):
    workflow_env()
    sink = RecordingTraceSink()
    ctx, wf = _session(sink)
    ctx._begin_turn(REQUEST)
    lm = DummyLM([_typed_answer(include_risk=True)])
    command = _command(tmp_path, PHASED_LOOKUP)

    _plan_request(ctx, wf, lm, plan_template="TEMPLATE", skill_command=command)

    assert _plan_span(sink).attributes["plan_source"] == "skill_call"
    called, call = ctx._phased_skill_call
    assert called is command
    assert [phase.name for phase in call.phases] == ["find", "rest"]
    assert call.args == _args(include_risk=True, count=1)
    assert call.steps_before == [] and call.steps_after == []


def test_an_unphased_skill_call_runs_as_one_phase_named_after_the_skill(tmp_path, workflow_env):
    workflow_env()
    sink = RecordingTraceSink()
    ctx, wf = _session(sink)
    ctx._begin_turn(REQUEST)
    lm = DummyLM([_typed_answer(include_risk=True)])
    command = _command(tmp_path)
    _plan_request(ctx, wf, lm, plan_template="TEMPLATE", skill_command=command)
    called, call = ctx._phased_skill_call
    assert called is command
    assert [phase.name for phase in call.phases] == ["lookup"]
    assert call.phases[0].steps_fn() == sc.expand(command, call.args)


def test_call_phases_gives_plan_phases_or_one_phase_of_the_flat_steps(tmp_path):
    phased = _command(tmp_path / "phased", PHASED_LOOKUP)
    args = _args(include_risk=True, count=1)
    assert [p.name for p in sc.call_phases(phased, args)] == \
        [p.name for p in sc.phases_for(phased, args)] == ["find", "rest"]
    flat = _command(tmp_path / "flat", LOOKUP)
    (one,) = sc.call_phases(flat, args)
    assert one.name == "lookup" and one.output is None and one.steps_fn() == sc.expand(flat, args)
    # plan() returning None: phases_for still says "flat", call_phases gives the one phase.
    none_plan = _command(tmp_path / "none", LOOKUP + "\n\ndef plan(input):\n    return None\n")
    assert sc.phases_for(none_plan, args) is None
    assert [p.name for p in sc.call_phases(none_plan, args)] == ["lookup"]
    assert sc.call_phases(none_plan, args)[0].steps_fn() == sc.expand(none_plan, args)


PAIR = '''
from typing import Literal

from pydantic import BaseModel, Field, ValidationInfo, model_validator


class Signature:
    """Weigh two named groups or feeds against each other. One job for the pair."""

    class Input(BaseModel):
        first: str = Field(description="the first record's name exactly as the request writes it")
        second: str = Field(description="the second record's name exactly as the request writes it")
        kind: Literal["group", "feed"] = Field(description="group, or feed (a source of logins)")

        @model_validator(mode="after")
        def _two_records(self, info: ValidationInfo):
            if self.first.casefold() == self.second.casefold():
                raise ValueError(f"first and second are both {self.first!r}; second is the other record "
                                 "the request names")
            request = (info.context or {}).get("request", "")
            if self.kind != "feed" and " feeds" in request:
                raise ValueError(f"kind={self.kind!r}, but the request says feeds; kind is feed")
            return self


def steps(input):
    return [f"open {input.kind} {input.first}", f"open {input.kind} {input.second}", "say which is riskier"]
'''

PAIR_REQUEST = "Which of Payroll and Billing is riskier?"


def _pair_answer(**fields):
    answer = {"reasoning": "r", "first": "Payroll", "second": "Billing", "kind": "group",
              "steps_before": [], "steps_after": []}
    answer.update(fields)
    return answer


def test_input_validators_refuse_a_call_with_a_message_that_says_what_to_fix(tmp_path):
    command = _command(tmp_path, PAIR, "pair")
    args, notes = sc.check_arguments(command, {"first": "Payroll", "second": "payroll", "kind": "group"},
                                     PAIR_REQUEST)
    with pytest.raises(sc.SkillInputError, match="rule of pair: first and second are both 'Payroll'; second is the other record") as refused:
        sc.validate_call(command, args, PAIR_REQUEST, None, notes)
    assert "Value error" not in str(refused.value)
    assert isinstance(refused.value, SkillCallError)
    # the request reaches the validators through the validation context
    request = "Which of the Payroll and Billing feeds is riskier?"
    args, _ = sc.check_arguments(command, {"first": "Payroll", "second": "Billing", "kind": "group"}, request)
    with pytest.raises(sc.SkillInputError, match="kind is feed"):
        sc.validate_call(command, args, request)
    assert sc.validate_call(command, {**args, "kind": "feed"}, request)["kind"] == "feed"
    # the template is rendered as before: the placeholders differ and the request is empty
    assert command.template[:2] == ("open <group or feed, by kind> <first>",
                                    "open <group or feed, by kind> <second>")


def test_a_hook_refusal_is_an_input_error_and_a_hook_exception_is_not(tmp_path):
    hooked = LOOKUP.replace('        return True, ""', '        if cmd_parameters.count > 5:\n'
                            '            return False, "count is at most 5"\n'
                            '        if cmd_parameters.count < 0:\n'
                            '            raise RuntimeError("bug")\n'
                            '        return True, ""')
    command = _command(tmp_path, hooked)
    with pytest.raises(sc.SkillInputError, match="count is at most 5"):
        sc.validate_call(command, _args(count=9), "Who is in Payroll?")
    with pytest.raises(SkillCallError) as raised:
        sc.validate_call(command, _args(count=-1), "Who is in Payroll?")
    assert not isinstance(raised.value, sc.SkillInputError)


def _plan_pair(ctx, wf, lm, command):
    ctx.push_active_workflow(wf)
    try:
        return workflow_agent.build_query_with_next_steps(PAIR_REQUEST, ctx, planner_lm=lm,
                                                          plan_template="TEMPLATE", skill_command=command)
    finally:
        ctx.pop_active_workflow()


def test_a_refused_skill_call_is_planned_again_with_the_refusal(tmp_path, workflow_env):
    workflow_env()
    sink = RecordingTraceSink()
    ctx, wf = _session(sink)
    ctx._begin_turn(PAIR_REQUEST)
    lm = DummyLM([_pair_answer(second="Payroll"), _pair_answer()])

    _plan_pair(ctx, wf, lm, _command(tmp_path, PAIR, "pair"))

    assert len(lm.history) == 2
    assert "was refused" not in _prompt(lm, 0)
    assert "A first call of this skill was refused" in _prompt(lm, 1) and "second is the other record" in _prompt(lm, 1)
    span = _plan_span(sink)
    assert span.attributes["plan_source"] == "skill_call"
    assert span.attributes["skill_args"] == {"first": "Payroll", "second": "Billing", "kind": "group"}


def test_a_second_refusal_falls_back_to_the_template_path(tmp_path, workflow_env):
    workflow_env()
    sink = RecordingTraceSink()
    ctx, wf = _session(sink)
    ctx._begin_turn(PAIR_REQUEST)
    lm = DummyLM([_pair_answer(second="Payroll"), _pair_answer(first="Billing"), _text_answer()])

    _plan_pair(ctx, wf, lm, _command(tmp_path, PAIR, "pair"))

    assert len(lm.history) == 3
    span = _plan_span(sink)
    assert span.attributes["plan_source"] == "text"
    assert "break a rule of pair" in span.attributes["skill_fallback"]


def test_a_guard_refusal_is_planned_again_with_the_refusal(tmp_path, workflow_env):
    workflow_env()
    sink = RecordingTraceSink()
    ctx, wf = _session(sink)
    ctx._begin_turn(PAIR_REQUEST)
    # a name the request does not write: planned again as a skill call, not the text plan
    lm = DummyLM([_pair_answer(second="Finance"), _pair_answer()])

    _plan_pair(ctx, wf, lm, _command(tmp_path, PAIR, "pair"))

    assert len(lm.history) == 2   # the typed call, then the typed call again
    assert "A first call of this skill was refused" in _prompt(lm, 1) and "'Finance' is not written" in _prompt(lm, 1)
    span = _plan_span(sink)
    assert span.attributes["plan_source"] == "skill_call"
    assert span.attributes["skill_fallback"] is None
    assert span.attributes["skill_args"] == {"first": "Payroll", "second": "Billing", "kind": "group"}


def test_a_second_guard_refusal_falls_back_to_the_template_path(tmp_path, workflow_env):
    workflow_env()
    sink = RecordingTraceSink()
    ctx, wf = _session(sink)
    ctx._begin_turn(PAIR_REQUEST)
    lm = DummyLM([_pair_answer(second="Finance"), _pair_answer(second="Finance"), _text_answer()])

    _plan_pair(ctx, wf, lm, _command(tmp_path, PAIR, "pair"))

    assert len(lm.history) == 3   # two typed calls, then the text planner
    assert "not written in the request" in _plan_span(sink).attributes["skill_fallback"]


def test_a_one_element_list_argument_stands_for_its_element(tmp_path, workflow_env):
    workflow_env()
    sink = RecordingTraceSink()
    ctx, wf = _session(sink)
    ctx._begin_turn(PAIR_REQUEST)
    lm = DummyLM([_pair_answer(second='["Billing"]')])

    _plan_pair(ctx, wf, lm, _command(tmp_path, PAIR, "pair"))

    assert len(lm.history) == 1
    span = _plan_span(sink)
    assert span.attributes["plan_source"] == "skill_call"
    assert span.attributes["skill_args"] == {"first": "Payroll", "second": "Billing", "kind": "group"}


# ---------------------------------------------------------------- the template's grid (fix-6llp.16)

#: Thirteen bools: 8192 combinations, past MAX_GRID. Each bool adds a phase; "pair" needs two.
WIDE = '''
from pydantic import BaseModel, Field

from fastworkflow.skill_phases import Phase

FLAGS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa", "lambda_",
         "mu", "nu"]


class Signature:
    """Survey one named shelf, with each optional section the request asks for. One job per shelf."""

    class Input(BaseModel):
        name: str = Field(description="the shelf's name exactly as the request writes it")
        alpha: bool = Field(default=False, description="asks for section alpha")
        beta: bool = Field(default=False, description="asks for section beta")
        gamma: bool = Field(default=False, description="asks for section gamma")
        delta: bool = Field(default=False, description="asks for section delta")
        epsilon: bool = Field(default=False, description="asks for section epsilon")
        zeta: bool = Field(default=False, description="asks for section zeta")
        eta: bool = Field(default=False, description="asks for section eta")
        theta: bool = Field(default=False, description="asks for section theta")
        iota: bool = Field(default=False, description="asks for section iota")
        kappa: bool = Field(default=False, description="asks for section kappa")
        lambda_: bool = Field(default=False, description="asks for section lambda")
        mu: bool = Field(default=False, description="asks for section mu")
        nu: bool = Field(default=False, description="asks for section nu")


def plan(input):
    phases = [Phase("find", lambda: [f"find_shelf query={input.name}"])]
    for flag in FLAGS:
        phases.append(Phase(flag, lambda flag=flag: [f"show_{flag}"], when=getattr(input, flag)))
    phases.append(Phase("pair", lambda: ["compare_alpha_beta"], when=input.alpha and input.beta))
    phases.append(Phase("report", lambda: ["say what the shelf holds"]))
    return phases
'''


def test_the_grid_is_every_combination_up_to_max_grid_in_product_order():
    import itertools

    assert sc.MAX_GRID == 4096
    assert sc.input_grid([2, 3, 2]) == list(itertools.product(range(2), range(3), range(2)))
    assert sc.input_grid([2, 3, 2], (1, 0, 0))[0] == (1, 0, 0)
    assert sc.input_grid([2] * 12) == list(itertools.product(*[range(2)] * 12))   # 4096: all
    grid = sc.input_grid([2] * 13, (1,) + (0,) * 12)   # 8192: the first, one and two changes
    assert grid[0] == (1,) + (0,) * 12 and len(grid) == 1 + 13 + 78 == len(set(grid))
    assert all(sum(a != b for a, b in zip(c, grid[0])) <= 2 for c in grid)
    for i, j in itertools.combinations(range(13), 2):
        assert {(c[i], c[j]) for c in grid} == {(0, 0), (0, 1), (1, 0), (1, 1)}


def test_a_skill_past_max_grid_loads_quickly_warns_once_and_its_template_uses_the_cover(
        tmp_path, monkeypatch):
    import time

    from fastworkflow.utils.logging import logger

    warnings = []
    monkeypatch.setattr(logger, "warning", lambda message, *a, **k: warnings.append(message))
    start = time.perf_counter()
    command = _command(tmp_path, WIDE, "survey_shelf")
    assert time.perf_counter() - start < 10
    assert len([w for w in warnings if "survey_shelf" in w]) == 1
    assert "8192 combinations" in warnings[0] and f"past {sc.MAX_GRID}" in warnings[0]
    template = list(command.template)
    assert template[0] == "find_shelf query=<name>" and template[-1] == "say what the shelf holds"
    for flag in ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa",
                 "lambda_", "mu", "nu"]:
        assert f"show_{flag} (only when {flag})" in template
    assert "compare_alpha_beta (only when alpha and beta)" in template   # two inputs: reached
    sc.placeholder_template(command)   # a second render in this process: no second warning
    assert len([w for w in warnings if "survey_shelf" in w]) == 1


#: An enum of forty values: its conditions are made from the values each line sees (fix-ajir.2).
SHELVES = """
from typing import Literal

from pydantic import BaseModel, Field

KINDS = [f"k{i}" for i in range(40)]


class Signature:
    \"\"\"Survey one named shelf of one kind. One job per shelf.\"\"\"

    class Input(BaseModel):
        name: str = Field(description="the shelf's name exactly as the request writes it")
        kind: Literal[KINDS_LITERAL] = Field(description="the shelf's kind")
        person: str = Field(default="", description="a person to compare with")
        deep: bool = Field(default=False, description="the request asks for a deep survey")


def steps(input):
    out = [f"find_shelf query={input.name}"]
    if input.kind in KINDS[:3]:
        out.append("check_a")
    if input.kind in KINDS[:20] and input.deep:
        out.append("deep_scan")
    if input.kind == KINDS[0] or input.deep:
        out.append("extra")
    if input.kind not in KINDS[:2] and input.person:
        out.append(f"compare with {input.person}")
    if input.kind != KINDS[-1]:
        out.append("not_last")
    out.append("say what the shelf holds")
    return out
""".replace("KINDS_LITERAL", ", ".join(repr(f"k{i}") for i in range(40)))


def test_a_skill_with_a_forty_value_enum_loads_quickly_with_its_conditions(tmp_path):
    import time

    start = time.perf_counter()
    template = list(_command(tmp_path, SHELVES, "survey_shelf").template)
    assert time.perf_counter() - start < 1
    first20 = " or ".join(f"k{i}" for i in range(20))
    assert template == [
        "find_shelf query=<name>",
        "check_a (only when kind is k0 or k1 or k2)",
        f"deep_scan (only when kind is {first20} and deep)",
        "extra (only when kind is k0 or deep)",
        "compare with <person> (only when kind is " + " or ".join(f"k{i}" for i in range(2, 40))
        + " and person is given)",
        "not_last (only when kind is not k39)",
        "say what the shelf holds",
    ]


def test_threads_that_load_skills_together_build_them_once(tmp_path, monkeypatch):
    import threading
    import time

    folder = _write(tmp_path, {"lookup": LOOKUP})
    built = []
    load = workflow_agent._load_skill_file

    def slow_load(name, path, workflow_folderpath):
        built.append(name)
        time.sleep(0.2)
        return load(name, path, workflow_folderpath)

    monkeypatch.setattr(workflow_agent, "_load_skill_file", slow_load)
    barrier = threading.Barrier(6)
    results = []

    def worker():
        barrier.wait()
        results.append(workflow_agent.load_skills(folder))

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert built == ["lookup"]
    assert len(results) == 6 and all(r == results[0] for r in results)


SHARED_CHECK = '''
from typing import Literal
from pydantic import BaseModel, Field

class Signature:
    """Look a record up by kind."""
    class Input(BaseModel):
        name: str = Field(description="the record's name")
        kind: Literal["k0", "k1", "k2", "k3", "k4"] = Field(description="the kind")


def steps(input):
    out = ["open_directory", f"find_{input.kind} query={input.name}"]
    if input.kind in ("k0", "k1", "k2"):
        out.append("check_a")
    out.append("say done")
    return out
'''


def test_an_enum_joins_only_lines_at_one_place_in_their_branches(tmp_path):
    """fix-ajir.38: "check_a", the second step of k0..k2, is not joined with "find_k3", the
    first step of k3 (it was ``<check_a or find_k3 query=<name> or find_k4 query=<name>, by kind>``)."""
    (_name, _desc, steps), = workflow_agent.load_skills(_write(tmp_path, {"shared_check": SHARED_CHECK}))
    assert steps == (
        "open_directory",
        "<find_k0 or find_k1 or find_k2, by kind> query=<name> (only when kind is k0 or k1 or k2)",
        "check_a (only when kind is k0 or k1 or k2)",
        "<find_k3 or find_k4, by kind> query=<name> (only when kind is k3 or k4)",
        "say done",
    )

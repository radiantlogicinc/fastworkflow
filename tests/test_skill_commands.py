"""Skills as parameterized commands: skill files with params, expansion, guards, planner path."""
from __future__ import annotations

import pytest
from dspy.utils import DummyLM

from fastworkflow import skill_commands as sc
from fastworkflow import workflow_agent
from fastworkflow.skill_commands import SkillCallError, SkillFormatError
from tests.test_text_planner_and_split import (  # noqa: F401 - workflow_env is a fixture
    RecordingTraceSink, _fake_matcher, _plan, _prompt, _session, _text_answer, workflow_env)

LOOKUP = '''
from typing import Literal

from pydantic import BaseModel, Field

from fastworkflow.skill_commands import repeat

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
        owner: str = Field(default="", description="the owner to compare the risk with",
                           json_schema_extra={"requires": "include_risk"})


def steps(input):
    out = ["open_directory", f"{FIND[input.kind]} query={input.name}"]
    out.append(f"{listing(input)} filter={input.person}" if input.person else listing(input))
    if input.include_risk:
        out.append(f"assess_risk{{owner={input.owner}}}")
    if input.kind == "group" and not (input.person or input.include_risk):
        out.append(f"open_group_by_uid with the uid of the {input.name} match")
    for n in repeat(input.count, 1):
        out.append(f"open member {n}")
    out.append(f"stop after the first {COUNT_WORD.get(input.count, input.count)} members")
    return out
'''


def _plain(description: str, steps: list[str]) -> str:
    """The skill file of a skill without params."""
    return f"class Signature:\n    __doc__ = {description!r}\n\n\ndef steps():\n    return {steps!r}\n"


def _write(tmp_path, skills: dict[str, str]) -> str:
    for name, text in skills.items():
        (tmp_path / "_skills" / name).mkdir(parents=True)
        (tmp_path / "_skills" / name / "skill.py").write_text(text)
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
    assert command.param("owner").requires.source == "include_risk"
    # what the matcher sees is the description, as for a plain skill
    assert workflow_agent._job_types_for(folder) == f"- lookup: {command.description}"


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
    folder = _write(tmp_path, {"leaver": _plain("Someone is leaving. One job per person.",
                                                 ["find {them} [if asked]", "[repeat x]", "@close"]),
                               "close": _plain("Close an account.", ["close it"])})
    assert workflow_agent.load_skills(folder) == (
        ("close", "Close an account.", ("close it",)),
        ("leaver", "Someone is leaving. One job per person.", ("find {them} [if asked]", "[repeat x]", "close it")),
    )
    assert workflow_agent.load_skill_commands(folder) == {}


AUDIT = '''
from pydantic import BaseModel


class Signature:
    """Audit one person."""

    class Input(BaseModel):
        name: str


def steps(input):
    return [f"find_person query={input.name}", "@notes"]
'''


def test_an_included_plain_skill_is_literal_text_inside_a_skill_with_params(tmp_path):
    folder = _write(tmp_path, {"audit": AUDIT,
                               "notes": _plain("Notes.", ["write {braces} and [if kept] as they are"])})
    command = workflow_agent.load_skill_commands(folder)["audit"]
    assert sc.expand(command, {"name": "Ann Lee"}) == [
        "find_person query=Ann Lee", "write {braces} and [if kept] as they are"]


def test_a_skill_with_params_cannot_be_included(tmp_path):
    folder = _write(tmp_path, {"outer": _plain("Outer.", ["@lookup"]), "lookup": LOOKUP})
    with pytest.raises(SkillFormatError, match=r"includes 'lookup', which has params"):
        workflow_agent.load_skills(folder)


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


def test_a_repeat_beyond_the_limit_is_refused(tmp_path):
    with pytest.raises(SkillCallError, match="more than"):
        sc.expand(_command(tmp_path), _args(count=sc.REPEAT_LIMIT + 2))


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


def test_an_argument_not_in_the_request_fails_a_required_one_and_resets_an_optional_one(tmp_path):
    command = _command(tmp_path)
    with pytest.raises(SkillCallError, match="not written in the request"):
        sc.check_arguments(command, {"name": "Billing", "kind": "group"}, "Who is in Payroll?")
    args, notes = sc.check_arguments(command, {"name": "Payroll", "kind": "group", "person": "Bo Diaz"},
                                     "Who is in Payroll?")
    assert args["person"] == "" and "default" in notes[0]


def test_an_optional_argument_repeating_another_one_takes_its_default(tmp_path):
    args, _ = sc.check_arguments(_command(tmp_path), {"name": "Payroll", "kind": "group", "person": "payroll"},
                                 "Who is in Payroll?")
    assert args["person"] == ""


def test_a_dependent_argument_is_reset_when_its_condition_is_false(tmp_path):
    command = _command(tmp_path)
    request = "Who is in Payroll, compared with the customer?"
    args, notes = sc.check_arguments(command, {"name": "Payroll", "kind": "group", "owner": "the customer"}, request)
    assert args["owner"] == "" and "needs include_risk" in notes[-1]
    args, _ = sc.check_arguments(command, {"name": "Payroll", "kind": "group", "owner": "the customer",
                                           "include_risk": True}, request)
    assert args["owner"] == "the customer"


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
    # "Billing" is not in the request: the guard refuses the call, the text planner plans
    lm = DummyLM([_typed_answer(name="Billing"), _text_answer()])

    _plan(ctx, wf, lm, plan_template="TEMPLATE TEXT", skill_command=_command(tmp_path))

    assert len(lm.history) == 2
    assert "TEMPLATE TEXT" in _prompt(lm, 1)
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
    folder = _write(tmp_path, {"bad": CHECK_FIELD.replace("{field}", field)})
    with pytest.raises(SkillFormatError, match=message):
        workflow_agent.load_skills(folder)


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
    args, notes = sc.check_arguments(command, raw, "Who is in Payroll, next to the customer?",
                                     {"include_risk": (False, 0.9)})
    assert (args["include_risk"], args["owner"]) == (False, "")
    assert "needs include_risk" in notes[-1]


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


def test_check_verdicts_skips_a_subject_check_when_the_request_names_nothing(tmp_path, monkeypatch):
    command = _named(tmp_path)
    monkeypatch.setattr(sc, "_check_one", lambda lm, cmd, p, request: ("Ann Lee", 0.9))
    assert sc.check_verdicts(object(), command, "may ann have it?") == {}
    assert sc.check_verdicts(object(), command, "May Ann Lee have Billing?") == {"person": ("Ann Lee", 0.9)}


def test_load_skill_commands_gives_each_command_the_workflow_identifiers(tmp_path):
    folder = _write(tmp_path, {"ask": NAMED})
    (tmp_path / "_skills" / "identifier_patterns.txt").write_text("# codes\n\\bacct_[0-9]+\n")
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

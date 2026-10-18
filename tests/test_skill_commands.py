"""Skills as parameterized commands: SKILL.md front matter, expansion, guards, planner path."""
from __future__ import annotations

import pytest
from dspy.utils import DummyLM

from fastworkflow import skill_commands as sc
from fastworkflow import workflow_agent
from fastworkflow.skill_commands import SkillCallError, SkillFormatError
from tests.test_text_planner_and_split import (  # noqa: F401 - workflow_env is a fixture
    RecordingTraceSink, _fake_matcher, _plan, _prompt, _session, _text_answer, workflow_env)

LOOKUP = """---
params:
  name:
    type: str
    strip: group, feed
    desc: the record's name exactly as the request writes it
  kind:
    type: enum[group, feed]
    desc: group, or feed (a source of logins)
  person:
    type: str
    default: ""
    desc: one named person to check for; empty lists everyone
  include_risk:
    type: bool
    desc: the request asks how risky it is
  count:
    type: int
    default: 2
    desc: how many members to open
  owner:
    type: str
    default: ""
    requires: include_risk
    desc: the owner to compare the risk with
maps:
  find:
    by: kind
    values:
      group: find_group
      feed: find_feed
  listing:
    by: kind
    values:
      group: list_members
    default: list_logins of {name}
  count_word:
    by: count
    values:
      1: one
      2: two
---
Who is in one named group or feed, or whether one person is. One job per record.
open_directory
{find} query={name}
[if person] {listing} filter={person}
[if not person] {listing}
[if include_risk] assess_risk{{owner={owner}}}
[if kind == "group" and not (person or include_risk)] open_group_by_uid with the uid of the {name} match
[repeat count - 1]
open member {n}
[end]
stop after the first {count_word} members
"""


def _write(tmp_path, skills: dict[str, str]) -> str:
    for name, text in skills.items():
        (tmp_path / "_skills" / name).mkdir(parents=True)
        (tmp_path / "_skills" / name / "SKILL.md").write_text(text)
    return str(tmp_path)


def _command(tmp_path, text: str = LOOKUP, name: str = "lookup") -> sc.SkillCommand:
    return workflow_agent.load_skill_commands(_write(tmp_path, {name: text}))[name]


def _args(**overrides):
    return {"name": "Payroll", "kind": "group", "person": "", "include_risk": False, "count": 2,
            "owner": "", **overrides}


# ---------------------------------------------------------------- parsing

def test_front_matter_gives_typed_params_maps_and_the_description_line(tmp_path):
    folder = _write(tmp_path, {"lookup": LOOKUP})
    command = workflow_agent.load_skill_commands(folder)["lookup"]

    assert command.description == "Who is in one named group or feed, or whether one person is. One job per record."
    assert [(p.name, p.type, p.required, p.default) for p in command.params] == [
        ("name", "str", True, None), ("kind", "enum", True, None), ("person", "str", False, ""),
        ("include_risk", "bool", False, False), ("count", "int", False, 2), ("owner", "str", False, "")]
    assert command.param("kind").values == ("group", "feed")
    assert command.param("name").strip == ("group", "feed")
    assert command.param("owner").requires.source == "include_risk"
    assert command.value_map("count_word").values == ((1, "one"), (2, "two"))
    assert command.value_map("listing").default == "list_logins of {name}"
    # what the matcher sees is the description line, as for a plain skill
    assert workflow_agent._job_types_for(folder) == f"- lookup: {command.description}"


def test_a_skill_with_params_offers_its_steps_as_template_text(tmp_path):
    folder = _write(tmp_path, {"lookup": LOOKUP})
    (_name, _desc, steps), = workflow_agent.load_skills(folder)
    assert steps == (
        "open_directory",
        "<find_group or find_feed, by kind> query=<name>",
        "<list_members or list_logins of <name>, by kind> filter=<person> (only when person is given)",
        "<list_members or list_logins of <name>, by kind> (only when person is empty)",
        "assess_risk{owner=<owner>} (only when include_risk)",
        "open_group_by_uid with the uid of the <name> match (only when kind is group and not (person is given or include_risk))",
        "repeat count - 1 times (n = 2 to count): open member <n>",
        "stop after the first <one or two, by count> members",
    )


def test_a_skill_without_params_is_read_exactly_as_before(tmp_path):
    text = "Someone is leaving. One job per person.\nfind {them} [if asked]\n[repeat x]\n@close\n"
    folder = _write(tmp_path, {"leaver": text, "close": "Close an account.\nclose it\n"})
    assert workflow_agent.load_skills(folder) == (
        ("close", "Close an account.", ("close it",)),
        ("leaver", "Someone is leaving. One job per person.", ("find {them} [if asked]", "[repeat x]", "close it")),
    )
    assert workflow_agent.load_skill_commands(folder) == {}


def test_an_included_plain_skill_is_literal_text_inside_a_skill_with_params(tmp_path):
    folder = _write(tmp_path, {
        "audit": "---\nparams:\n  name:\n    type: str\n---\nAudit one person.\nfind_person query={name}\n@notes\n",
        "notes": "Notes.\nwrite {braces} and [if kept] as they are\n",
    })
    command = workflow_agent.load_skill_commands(folder)["audit"]
    assert sc.expand(command, {"name": "Ann Lee"}) == [
        "find_person query=Ann Lee", "write {braces} and [if kept] as they are"]


@pytest.mark.parametrize("text, message", [
    ("---\nparams:\n  name:\n    type: str\nOne.\nstep\n", "no closing '---'"),
    ("---\n---\nOne.\nstep\n", "params must be a non-empty block"),
    ("---\nparams:\n  name:\n    type: text\n---\nOne.\nstep\n", "unknown type 'text'"),
    ("---\nparams:\n  flag:\n    type: bool\n    default: true\n---\nOne.\nstep\n", "must default to false"),
    ("---\nparams:\n  n:\n    type: int\n---\nOne.\nstep\n", "bad parameter name 'n'"),
    ("---\nparams:\n  name:\n    type: str\n    colour: red\n---\nOne.\nstep\n", "unknown key(s) colour"),
    ("---\nparams:\n  name:\n    type: str\n   desc: x\n---\nOne.\nstep\n", "inconsistent indentation"),
    ("---\nparams:\n  kind:\n    type: enum[a, b]\n    default: c\n---\nOne.\nstep\n", "not one of its values"),
    ("---\nparams:\n  name:\n    type: str\n---\nOne.\nfind {nmae}\n", "unknown placeholder {nmae}"),
    ("---\nparams:\n  name:\n    type: str\n---\nOne.\nfind {name\n", "stray brace"),
    ("---\nparams:\n  name:\n    type: str\n---\nOne.\n[if nmae] find\n", "unknown parameter 'nmae'"),
    ("---\nparams:\n  kind:\n    type: enum[a, b]\n---\nOne.\n[if kind] find\n", "compare enum 'kind'"),
    ("---\nparams:\n  kind:\n    type: enum[a, b]\n---\nOne.\n[if kind == \"c\"] find\n", "'c' is not a value"),
    ("---\nparams:\n  flag:\n    type: bool\n---\nOne.\n[if flag and] find\n", "bad condition"),
    ("---\nparams:\n  flag:\n    type: bool\n---\nOne.\n[if flag]\n", "malformed marker"),
    ("---\nparams:\n  flag:\n    type: bool\n---\nOne.\n[repeat flag]\nstep\n[end]\n", "must repeat an int"),
    ("---\nparams:\n  k:\n    type: int\n---\nOne.\n[repeat k]\n[repeat k]\nstep\n[end]\n[end]\n", "do not nest"),
    ("---\nparams:\n  k:\n    type: int\n---\nOne.\n[repeat k]\nstep\n", "without [end]"),
    ("---\nparams:\n  k:\n    type: int\n---\nOne.\nstep\n[end]\n", "[end] without [repeat]"),
    ("---\nparams:\n  name:\n    type: str\nmaps:\n  m:\n    by: name\n    values:\n      a: b\n---\nOne.\n{m}\n",
     "must be by an enum, bool or int"),
    ("---\nparams:\n  kind:\n    type: enum[a, b]\nmaps:\n  m:\n    by: kind\n    values:\n      c: x\n---\nOne.\n{m}\n",
     "'c' is not a value of 'kind'"),
    ("---\nparams:\n  kind:\n    type: enum[a, b]\nmaps:\n  m:\n    by: kind\n    values:\n      a: \"{w}\"\n"
     "  w:\n    by: kind\n    values:\n      a: \"{m}\"\n---\nOne.\n{m}\n", "cycle"),
])
def test_a_malformed_skill_fails_the_load_naming_its_file(tmp_path, text, message):
    folder = _write(tmp_path, {"bad": text})
    with pytest.raises(SkillFormatError) as err:
        workflow_agent.load_skills(folder)
    assert message in str(err.value)
    assert "bad/SKILL.md" in str(err.value) or "_skills" in str(err.value)


def test_a_skill_with_params_cannot_be_included(tmp_path):
    folder = _write(tmp_path, {"outer": "Outer.\n@lookup\n", "lookup": LOOKUP})
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

"""The SKILL.md -> skill.py converter (``tools/skill_md_to_py.py``, a dev tool outside the
package): every construct the SKILL.md parser accepts converts to a skill file that parses
to the same skill (description, parameters, template text) and expands to the same steps
over a grid of arguments."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

import fastworkflow
from fastworkflow import skill_commands as sc
from fastworkflow import workflow_agent
from tests.test_skill_commands import CHECKED, LOOKUP, NAMED

REPO = Path(__file__).resolve().parents[1]
TOOL = REPO / "tools" / "skill_md_to_py.py"


def _tool():
    spec = importlib.util.spec_from_file_location("skill_md_to_py", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tool = _tool()

ROSTER = """---
params:
  team:
    type: str
    strip: team, crew
    desc: the team's name exactly as the request writes it
  scope:
    type: enum[every, first, named]
    desc: every person; the first few; only the people the request names
    verify: true
    means:
      every: the request walks every person on the list
      first: the request limits the walk to a number of people
      named: the request asks only about people it names
  people:
    type: int
    default: 0
    requires: scope == "first"
    in_request: true
    desc: how many people to take; 0 otherwise
  person:
    type: str
    default: ""
    requires: scope == "named"
    desc: the first person the request names
  person_2:
    type: str
    default: ""
    requires: person
    desc: a second person the request names
---
Go through one team member by member. One job per team.
find_team query={team}
show_members (the list; next page with start=<n> until complete)
[if scope == "every"] open_member <the first person on the list>
[if scope == "every"] back to the team
[repeat people]
open_member <person {n} on the list>
[if person] note {person} on round {n}
back to the team
[end]
[if person] show_members filter={person}
[if person_2] show_members filter={person_2}
answer with the list total
"""

MAPS = """---
params:
  name:
    type: str
    desc: the record
  kind:
    type: enum[group, unit, feed]
    default: unit
    desc: the kind
  flag:
    type: bool
    desc: a flag
  count:
    type: int
    default: 1
    desc: how many
maps:
  find:
    by: kind
    values:
      group: find_group
      unit: find_unit
  open:
    by: kind
    values:
      group: open_group {name}
    default: open_other {find} for {name}
  flagged:
    by: flag
    values:
      true: "with flag {name}; "
      false: ""
  count_word:
    by: count
    values:
      1: one
      2: "two {open}"
    default: many
  round:
    by: count
    values:
      2: round {n} of two
  total:
    by: flag
    values:
      true: all
      false: some
---
Maps of every kind.
{find} {name}
{open}
{flagged}then stop
[if not flag] {count_word} items
[repeat count]
{round}: {total}
[end]
plain {kind} {flag} {count} {total}
"""

CONDITIONS = """---
params:
  a:
    type: bool
  b:
    type: bool
  kind:
    type: enum[x, y]
    desc: "a \\"quoted\\" kind"
  k:
    type: int
    default: 3
  who:
    type: str
    default: someone
---
Conditions: and, or, not, parentheses, != and an int "compared" with a number.
[if a and b] both
[if a or b] either
[if not (a and b)] not both
[if not a or b] not a, or b
[if (a or b) and not kind == "x"] y and a flag
[if kind != "x" and k == 2] two of y
[if not who] nobody
[if who] by {who}
[repeat k - 2]
round {n}
[if a] round {n} with a
[end]
after
"""

ESCAPES = r"""---
params:
  name:
    type: str
    desc: "a \"name\" with a back\\slash"
---
Escapes: "quotes", back\slash, {braces}, unicode — and ü.
say "{name}" with a back\slash
write {{literal}} braces and {{{name}}}
it's 'single' and "double"
@notes
"""

NOTES = "Notes.\nwrite {braces} and [if kept] as they are\n@more\n"
MORE = "More.\n[repeat x]\nthe end\n"

FIXTURES = {"lookup": LOOKUP + "@notes\n", "checked": CHECKED, "ask": NAMED, "team-walk": ROSTER,
            "maps": MAPS, "conditions": CONDITIONS, "escapes": ESCAPES, "notes": NOTES, "more": MORE}


def _skills(tmp_path, fixtures=FIXTURES) -> Path:
    root = tmp_path / "wf" / "_skills"
    for name, text in fixtures.items():
        (root / name).mkdir(parents=True)
        (root / name / "SKILL.md").write_text(text, encoding="utf-8")
    (root / "identifier_patterns.txt").write_text("\\bacct_[0-9]+\n")
    return root


def test_the_converter_is_a_dev_tool_outside_the_package():
    assert TOOL.is_file()
    package = Path(fastworkflow.__file__).resolve().parent
    assert package not in TOOL.parents
    assert "tools" not in (REPO / "pyproject.toml").read_text().split("[tool.poetry.scripts]")[0]


def test_every_fixture_converts_to_a_skill_file_that_parses_and_expands_the_same(tmp_path):
    report = tool.compare_folder(_skills(tmp_path), workdir=str(tmp_path))
    assert set(report) == set(FIXTURES)
    assert report == {name: [] for name in FIXTURES}


@pytest.mark.parametrize("name", ["lookup", "checked", "ask", "team-walk", "maps", "conditions", "escapes"])
def test_the_converted_source_compiles_and_covers_the_argument_grid(name, tmp_path):
    source = tool.convert(FIXTURES[name], name)
    compile(source, f"{name}/skill.py", "exec")
    _d, _s, command = tool.read_skill_md(FIXTURES[name], name)
    assert tool.argument_grid(command)


def test_a_difference_is_reported(tmp_path):
    root = _skills(tmp_path, {"maps": MAPS})
    folder = str(root.parent)
    old = workflow_agent.load_skill_commands(folder)["maps"]
    py = tmp_path / "py" / "_skills" / "maps"
    py.mkdir(parents=True)
    (py / "skill.py").write_text(tool.convert(MAPS, "maps").replace('f"plain ', 'f"PLAIN '))
    new = workflow_agent.load_skill_commands(str(tmp_path / "py"))["maps"]
    problems = tool.compare(old, new)
    assert problems and all("steps for" in p for p in problems)


def test_the_argument_screen_and_guards_agree_on_both_forms(tmp_path):
    root = _skills(tmp_path, {"lookup": LOOKUP})
    old = workflow_agent.load_skill_commands(str(root.parent))["lookup"]
    py = tmp_path / "py" / "_skills" / "lookup"
    py.mkdir(parents=True)
    (py / "skill.py").write_text(tool.convert(LOOKUP, "lookup"))
    new = workflow_agent.load_skill_commands(str(tmp_path / "py"))["lookup"]
    request = "Who is in the payroll group? Is Ann Lee one of them? Open 3."
    for raw in ({"name": '"Payroll group"', "kind": "Group", "person": "ann lee", "count": "3"},
                {"name": "<payroll>", "kind": "group", "person": "<person>", "count": -1},
                {"name": "payroll", "kind": "group", "owner": "Ann Lee", "include_risk": True}):
        assert sc.check_arguments(old, raw, request) == sc.check_arguments(new, raw, request)


@pytest.mark.filterwarnings("ignore:Field name")
def test_a_parameter_name_pydantic_reserves_is_refused():
    text = "---\nparams:\n  schema:\n    type: str\n---\nOne.\nfind {schema}\n"
    with pytest.raises(sc.SkillFormatError, match="cannot be a pydantic field name"):
        tool.convert(text, "bad")


def test_the_command_line_writes_checks_and_removes(tmp_path, capsys):
    root = _skills(tmp_path, {"lookup": LOOKUP, "notes": NOTES, "more": MORE})
    assert tool.main([str(root), "--check"]) == 0
    assert not list(root.glob("*/skill.py"))
    assert tool.main([str(root), "--write", "--check", "--remove-md"]) == 0
    assert sorted(p.parent.name for p in root.glob("*/skill.py")) == ["lookup", "more", "notes"]
    assert not list(root.glob("*/SKILL.md"))
    out = capsys.readouterr().out
    assert "lookup: ok" in out and "wrote" in out and "removed" in out
    workflow_agent._parsed_skills.cache_clear()
    command = workflow_agent.load_skill_commands(str(root.parent))["lookup"]
    assert command.path.endswith("skill.py")

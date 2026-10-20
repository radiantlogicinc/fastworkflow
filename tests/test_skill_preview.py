"""Skill previews (fix-553q): ``fastworkflow skill preview`` and ``fastworkflow.skill_preview``.

Offline, no model: a neutral catalog workflow with a phased skill (with a conditional
phase), a flat skill and a skill without inputs, rendered to markdown.
"""
from __future__ import annotations

import itertools
import os
import subprocess
import sys
import textwrap

import pytest

from fastworkflow import cli, skill_preview

SHOW_ITEM = '''
"""show_item: one named item, its parts and, when asked, its open checks."""
from typing import Optional

from pydantic import BaseModel, Field

from fastworkflow.skill_phases import Phase, concatenated_steps


class Signature:
    """Show one named item and its parts, and its open checks when asked. One job per item."""

    plain_utterances = ["what parts does the Alpha item have", "show item Beta and its checks"]

    class Input(BaseModel):
        name: str = Field(description="the item's name exactly as the request writes it",
                          json_schema_extra={"subject": True})
        checks: bool = Field(default=False, description="the request asks about its open checks",
                             json_schema_extra={"verify": True})


class ItemMatch(BaseModel):
    """The items the search listed."""
    item_uids: list[str] = Field(json_schema_extra={"from": "find_item.item_uids"}, min_length=1)


class ItemFacts(BaseModel):
    uid: str = Field(json_schema_extra={"from": "open_item_by_uid.item_uid"})
    part_labels: list[str] = Field(json_schema_extra={"from": "list_parts.labels"})
    weight: float = Field(json_schema_extra={"from": "list_parts.weight"})


class CheckFacts(BaseModel):
    total: int = Field(json_schema_extra={"from": "count_checks.total"})


def plan(input: Signature.Input) -> list[Phase]:
    phases = [
        Phase("find", lambda: find_steps(input.name), output=ItemMatch),
        Phase("item", lambda find: item_steps(input.name, find), output=ItemFacts),
        Phase("checks", lambda item: checks_steps(item), output=CheckFacts, when=input.checks),
    ]
    if input.checks:
        phases.append(Phase("report", lambda item, checks: report_steps(item, checks)))
    else:
        phases.append(Phase("report", lambda item: report_steps(item)))
    return phases


def steps(input: Signature.Input) -> list[str]:
    return concatenated_steps(plan(input))


def find_steps(name):
    return ["open_catalog", f"find_item query={name}"]


def item_steps(name, find: Optional[ItemMatch] = None):
    uid = find.item_uids[0] if find else f"<the uid of the {name} match>"
    return [f"open_item_by_uid item_uid={uid}", "list_parts (full list)"]


def checks_steps(item: Optional[ItemFacts] = None):
    return [f"count_checks on {item.uid if item else 'the item'}"]


def report_steps(item: Optional[ItemFacts] = None, checks: Optional[CheckFacts] = None):
    if item is None:
        return ["say which parts the item has"]
    counted = f"; open checks: {checks.total}" if checks else ""
    return [f"say that {item.uid} has the parts {', '.join(item.part_labels)}{counted}, "
            f"weight {item.weight:.1f}"]
'''

LIST_PARTS = '''
"""list_parts_of: the parts of one named item or kit, optionally only one supplier's."""
from typing import Literal

from pydantic import BaseModel, Field

FIND = {"item": "find_item", "kit": "find_kit"}


class Signature:
    """List the parts of one named item or kit. One job per record."""

    class Input(BaseModel):
        name: str = Field(description="the record's name | as written",
                          json_schema_extra={"strip": ["item", "kit"]})
        kind: Literal["item", "kit"] = Field(default="item", description="what the request is about",
                                             json_schema_extra={"means": {"item": "one item",
                                                                          "kit": "a set of items"}})
        supplier: str = Field(default="", description="one supplier to keep; empty for all")
        detail: bool = Field(default=False, description="never changes the steps")
        count: int = Field(default=2, description="how many parts to open")

    @staticmethod
    def validate_extracted_parameters(workflow, command, cmd_parameters, command_text=None):
        if cmd_parameters.kind == "kit" and cmd_parameters.supplier:
            return False, "a kit is listed for every supplier"
        return True, ""


def steps(input: Signature.Input) -> list[str]:
    out = lookup_steps(input.kind, input.name)
    out.append(f"list_parts supplier={input.supplier}" if input.supplier else "list_parts")
    return out + ["@close_out"]


def lookup_steps(kind, name):
    return [f"{FIND[kind]} query={name}"]
'''

CLOSE_OUT = '''
class Signature:
    """Go back to the catalog. Fixed steps."""


def steps():
    return ["go_up", "say what was shown"]
'''


def _workflow(root) -> str:
    folder = root / "catalog_wf"
    (folder / "_skills").mkdir(parents=True)
    for name, text in {"show_item": SHOW_ITEM, "list_parts_of": LIST_PARTS,
                       "close_out": CLOSE_OUT}.items():
        (folder / "_skills" / f"{name}.py").write_text(textwrap.dedent(text))
    return str(folder)


@pytest.fixture
def workflow(tmp_path):
    return _workflow(tmp_path)


def _run_cli(monkeypatch, *argv) -> int:
    monkeypatch.setattr(sys, "argv", ["fastworkflow", *argv])
    try:
        cli.main()
    except SystemExit as exit_:
        return int(exit_.code or 0)
    return 0


def test_every_skill_renders_one_preview(workflow):
    previews = skill_preview.render_previews(workflow)
    assert sorted(previews) == ["close_out.md", "list_parts_of.md", "show_item.md"]
    for name, text in previews.items():
        assert text.startswith("<!-- Generated by `fastworkflow skill preview` from _skills/")
        assert text.endswith("\n") and not text.endswith("\n\n")
        assert workflow not in text   # no absolute path


def test_phased_preview_shows_each_phase_its_facts_and_the_condition_both_ways(workflow):
    text = skill_preview.render_previews(workflow)["show_item.md"]
    assert "Kind: phased" in text
    assert "| `name` | str | required | `<name>` |  | yes | the item's name exactly as the request writes it |" in text
    assert "| `checks` | bool | false | false | yes |  | the request asks about its open checks |" in text
    assert "- what parts does the Alpha item have" in text
    # the phases, and when each runs
    assert "| `find` | `ItemMatch` | always |" in text
    assert "| `checks` | `CheckFacts` | only when checks = true |" in text
    assert "| `report` | none | always |" in text
    # facts and their sources
    assert "| `item_uids` | `list[str]` | `find_item.item_uids` |" in text
    assert "| `weight` | `float` | `list_parts.weight` |" in text
    assert "The items the search listed." in text
    # the condition both ways, earlier facts as placeholders
    off = text.index("### Plan when checks = false")
    on = text.index("### Plan when checks = true")
    assert off < on
    assert "1. `open_item_by_uid item_uid=<find.item_uids[0]>`" in text
    assert "Reads: `find`. Output: `ItemFacts`." in text
    assert "count_checks" not in text[off:on]
    # the item's uid is the uid the find listed (one placeholder for one value)
    assert "1. `count_checks on <find.item_uids[0]>`" in text[on:]
    assert ("`say that <find.item_uids[0]> has the parts <item.part_labels[0]>; open checks: "
            "<checks.total>, weight <item.weight>`") in text[on:]
    assert ("`say that <find.item_uids[0]> has the parts <item.part_labels[0]>, weight "
            "<item.weight>`") in text[off:on]
    # a phase that reads the same as in the first plan is one line
    assert "As in the plan when checks = false." in text[on:]


def test_flat_preview_renders_the_inputs_that_change_the_steps(workflow):
    text = skill_preview.render_previews(workflow)["list_parts_of.md"]
    assert "Kind: flat" in text
    assert "| `name` | str | required | `<name>` |  |  | the record's name \\| as written |" in text
    assert "| `kind` | one of item, kit | `item` | `item` |" in text
    assert "| `count` | int | `2` | `2` |" in text
    assert "- `name` strips the words `item`, `kit` from its ends." in text
    assert "- `kind` means: `item`: one item; `kit`: a set of items." in text
    assert "inputs that change them give: `kind`, `supplier`." in text   # not `detail`
    # each different list of steps once, titled by the inputs that decide it
    assert "### Steps when supplier = given" in text
    assert "### Steps when kind = item, supplier = empty" in text
    assert "### Steps when kind = kit\n" in text
    assert "1. `find_item query=<name>`\n2. `list_parts supplier=<supplier>`\n3. `go_up`" in text
    # a combination the hook refuses is listed, not rendered
    assert "supplier = given, kind" not in text and "kind = kit, supplier" not in text
    assert "- every other combination: validate_extracted_parameters: a kit is listed" in text


def test_fixed_skill_preview(workflow):
    text = skill_preview.render_previews(workflow)["close_out.md"]
    assert "None: the steps are fixed." in text
    assert "## Steps\n\n1. `go_up`\n2. `say what was shown`" in text


def test_render_is_byte_stable_across_processes(workflow, tmp_path):
    first = skill_preview.render_previews(workflow)
    assert skill_preview.render_previews(workflow) == first
    out = tmp_path / "other_process"
    env = {**os.environ, "PYTHONHASHSEED": "12345"}
    subprocess.run([sys.executable, "-m", "fastworkflow.cli", "skill", "preview", workflow,
                    "--out", str(out)], check=True, env=env, capture_output=True)
    assert {f: (out / f).read_text(encoding="utf-8") for f in os.listdir(out)} == first


def test_default_out_dir_is_docs_next_to_the_workflow(workflow):
    assert skill_preview.default_out_dir(workflow) == os.path.join(
        os.path.dirname(workflow), "docs", "skill_previews")


def test_check_passes_on_fresh_output_and_fails_on_an_edited_missing_or_extra_one(
        workflow, tmp_path, monkeypatch, capsys):
    out = tmp_path / "previews"
    assert _run_cli(monkeypatch, "skill", "preview", workflow, "--check", "--out", str(out)) == 1
    assert _run_cli(monkeypatch, "skill", "preview", workflow, "--out", str(out)) == 0
    assert _run_cli(monkeypatch, "skill", "preview", workflow, "--check", "--out", str(out)) == 0
    assert skill_preview.check_previews(workflow, str(out)) == []

    edited = out / "show_item.md"
    edited.write_text(edited.read_text(encoding="utf-8").replace("open_catalog", "open_shop"),
                      encoding="utf-8")
    capsys.readouterr()
    assert _run_cli(monkeypatch, "skill", "preview", workflow, "--check", "--out", str(out)) == 1
    assert "stale: show_item.md" in capsys.readouterr().out

    (out / "close_out.md").unlink()
    (out / "notes.md").write_text("hand-written\n", encoding="utf-8")
    assert skill_preview.check_previews(workflow, str(out)) == [
        "missing: close_out.md", "stale: show_item.md", "extra: notes.md"]


def test_write_removes_stale_generated_previews_and_keeps_other_files(workflow, tmp_path):
    out = tmp_path / "previews"
    skill_preview.write_previews(workflow, str(out))
    gone = out / "old_skill.md"
    gone.write_text(skill_preview.MARKER.format(source="_skills/old_skill.py") + "\n",
                    encoding="utf-8")
    (out / "notes.md").write_text("hand-written\n", encoding="utf-8")
    result = skill_preview.write_previews(workflow, str(out))
    assert result["removed"] == ["old_skill.md"] and result["kept"] == ["notes.md"]
    assert result["unchanged"] == ["close_out.md", "list_parts_of.md", "show_item.md"]
    assert not gone.exists()


def test_a_workflow_without_skills_is_an_error(tmp_path):
    with pytest.raises(skill_preview.PreviewError):
        skill_preview.render_previews(str(tmp_path))


# ---------------------------------------------------------------- fix-hia1.9: conditions, until, flat

CHECK_ITEMS = '''
"""check_items: the named items checked one by one, phased only for the deep check."""
from pydantic import BaseModel, Field

from fastworkflow.skill_phases import Phase


class Signature:
    """Check the named items one by one until two pass. One job for the whole list."""

    class Input(BaseModel):
        names: str = Field(description="the items' names, comma separated")
        deep: bool = Field(default=False, description="the request asks for the deep check")


class Match(BaseModel):
    item_uids: list[str] = Field(json_schema_extra={"from": "find_item.item_uids"})


class Check(BaseModel):
    uid: str = Field(json_schema_extra={"from": "open_item_by_uid.item_uid"})
    passed: bool = Field(json_schema_extra={"from": "run_check.passed"})


def plan(input: Signature.Input):
    if not input.deep:
        return None
    return [
        Phase("find", lambda: [f"find_item query={input.names}"], output=Match),
        Phase("checks", lambda uid=None: check_steps(uid), output=Check,
              over="find.item_uids", item="uid", max_items=5, noun="matches",
              until=lambda checks: sum(c.passed for c in checks) >= 2),
        Phase("extra", lambda find: [f"count_checks for {len(find.item_uids)} items"],
              when=lambda find: len(find.item_uids) > 1),
        Phase("report", lambda checks: report_steps(checks), when=has_checks),
    ]


def steps(input: Signature.Input):
    return [f"find_item query={input.names}", "run_check on each match", "say which passed"]


def has_checks(checks):
    return bool(checks)


def check_steps(uid=None):
    return [f"open_item_by_uid item_uid={uid or '<the uid>'}", "run_check"]


def report_steps(checks=None):
    passed = [c.uid for c in checks or [] if c.passed]
    return [f"say that {len(passed)} items passed: {', '.join(passed) or 'none'}"]
'''


@pytest.fixture
def checks_workflow(tmp_path):
    folder = tmp_path / "checks_wf"
    (folder / "_skills").mkdir(parents=True)
    (folder / "_skills" / "check_items.py").write_text(textwrap.dedent(CHECK_ITEMS))
    return str(folder)


def test_conditions_fan_out_settings_and_a_flat_plan_are_rendered(checks_workflow):
    text = skill_preview.render_previews(checks_workflow)["check_items.md"]
    assert "| phase | output | runs | fans out |" in text
    # When plan() is phased is said once; the table is about the phased calls.
    assert ("`plan()` returns phases only when deep = true; for other inputs the call runs "
            "flat, from `steps()`.") in text
    assert "| `find` | `Match` | always | no |" in text
    assert ("| `checks` | `Check` | always | per item of `find.item_uids` as "
            "`uid` (matches); at most 5; until `lambda checks: sum(c.passed for c in checks) >= 2` |") in text
    assert "| `extra` | none | only when `lambda find: len(find.item_uids) > 1` | no |" in text
    # A named function is shown by its name.
    assert "| `report` | none | only when `has_checks` | no |" in text
    assert "Settings: `when=lambda find: len(find.item_uids) > 1`." in text
    assert ("Settings: `over='find.item_uids'`, `item='uid'`, `max_items=5`, "
            "`until=lambda checks: sum(c.passed for c in checks) >= 2`, `noun='matches'`.") in text
    # For deep = false plan() returns None: the flat steps() are rendered, and said so.
    flat = text.index("### Plan when deep = false")
    phased = text.index("### Plan when deep = true")
    assert ("`plan()` returns no phases for these inputs: the call runs flat, from `steps()`."
            "\n\n1. `find_item query=<names>`\n2. `run_check on each match`") in text[flat:phased]


def test_a_phase_computing_on_facts_is_marked_as_rendered_with_placeholders(checks_workflow,
                                                                           workflow):
    text = skill_preview.render_previews(checks_workflow)["check_items.md"]
    note = f"_{skill_preview.PLACEHOLDER_NOTE}_"
    phases = text.split("#### Phase ")[1:]
    marked = {part.split("`")[1]: note in part for part in phases}
    # `find` and `checks` read no earlier facts (an item is not a fact); the rest do.
    assert marked == {"find": False, "checks": False, "extra": True, "report": True}
    # The report computes "0 items passed" from placeholder facts: marked, not trusted.
    assert "1. `say that 0 items passed: none`" in text
    shown = skill_preview.render_previews(workflow)["show_item.md"]
    assert shown.count(note) >= 2   # item and report read earlier facts


def test_condition_text_reads_the_lambda_source_or_falls_back_to_the_name():
    def named(find):
        return True

    one = [lambda find: ["x"], lambda find: bool(find)]   # two lambdas on one line
    assert skill_preview.condition_text(named) == "named"
    assert skill_preview.condition_text(one[1]) == "<lambda>"   # cannot be told apart
    pair = dict(steps=lambda find: ["x"], when=lambda find: bool(find))
    assert skill_preview.condition_text(pair["when"], "when") == "lambda find: bool(find)"
    assert skill_preview.condition_text(eval("lambda find: True")) == "<lambda>"   # no source


def test_the_new_settings_render_byte_stable(checks_workflow, tmp_path):
    first = skill_preview.render_previews(checks_workflow)
    assert skill_preview.render_previews(checks_workflow) == first
    out = tmp_path / "other_process"
    env = {**os.environ, "PYTHONHASHSEED": "777"}
    subprocess.run([sys.executable, "-m", "fastworkflow.cli", "skill", "preview", checks_workflow,
                    "--out", str(out)], check=True, env=env, capture_output=True)
    assert {f: (out / f).read_text(encoding="utf-8") for f in os.listdir(out)} == first


# ---------------------------------------------------------------- the readability fixes

WALK_BOXES = '''
"""walk_boxes: one shelf's boxes, every box or only the named ones, each opened and weighed."""
from typing import Literal

from pydantic import BaseModel, Field

from fastworkflow.skill_phases import Phase


class Signature:
    """Walk the boxes of one shelf, or only the boxes the request names. One job per shelf."""

    class Input(BaseModel):
        shelf: str = Field(description="the shelf's name")
        scope: Literal["every", "named"] = Field(default="every", description="every box, or the named ones")
        first: str = Field(default="", description="the first box named")
        second: str = Field(default="", description="a second box named")
        boxes: int = Field(default=0, ge=0, description="how many boxes to take; 0 for every box")

    @staticmethod
    def validate_extracted_parameters(workflow, command, cmd_parameters, command_text=None):
        if cmd_parameters.scope != "named":
            cmd_parameters.first = ""
        if not cmd_parameters.first:
            cmd_parameters.second = ""
        return True, ""


class Shelf(BaseModel):
    box_uids: list[str] = Field(json_schema_extra={"from": "list_boxes.box_uids"})
    box_labels: list[str] = Field(json_schema_extra={"from": "list_boxes.labels"})


class Box(BaseModel):
    uid: str = Field(json_schema_extra={"from": "open_box.box_uid"})
    label: str = Field(json_schema_extra={"from": "open_box.label"})
    weight: float = Field(json_schema_extra={"from": "weigh_box.weight"})


def plan(input: Signature.Input):
    phases = [Phase("shelf", lambda: [f"find_shelf query={input.shelf}", "list_boxes"], output=Shelf)]
    if input.scope == "named":
        names = [name for name in (input.first, input.second) if name]
        phases.append(Phase("box", lambda name: [f"open_box label={name}", "weigh_box"],
                            output=Box, over=names, item="name", bind=False))
    else:
        phases.append(Phase("box", lambda uid, shelf: box_steps(uid, shelf), output=Box,
                            over="shelf.box_uids", item="uid", max_items=input.boxes or None))
    phases.append(Phase("report", lambda shelf=None, box=None: report_steps(shelf, box)))
    return phases


def box_steps(uid=None, shelf=None):
    if shelf is None or uid not in shelf.box_uids:
        return ["open_box with the uid of the next box", "weigh_box"]
    index = shelf.box_uids.index(uid)
    return (["go_up"] if index else []) + [
        f"open_box box_uid={uid} ({shelf.box_labels[index]})", "weigh_box"]


def report_steps(shelf=None, box=None):
    if shelf is None or box is None:
        return ["say what each box opened weighs"]
    lines = []
    for one in box:
        kind = one.label.split("_", 1)[0]
        where = "on the shelf" if one.uid in shelf.box_uids else "not on the shelf"
        lines.append(f"say that {one.label} (kind {kind}) is {where} and weighs {int(one.weight)}")
    return lines or ["say that no box was opened"]
'''

PICK_KIT = '''
"""pick_kit: one named record of four kinds, found in one of three modes."""
from typing import Literal

from pydantic import BaseModel, Field

from fastworkflow.skill_phases import Phase


class Signature:
    """Find one named record and say what it holds. One job per record."""

    class Input(BaseModel):
        name: str = Field(description="the record's name")
        kind: Literal["box", "crate", "bin", "tray"] = Field(default="box", description="its kind")
        mode: Literal["quick", "deep", "full"] = Field(default="quick", description="how to look")
        audit: bool = Field(default=False, description="the request asks for an audit")
        note: str = Field(default="", description="a note to add")
        count: int = Field(default=0, ge=1, description="never changes the steps")


class BoxMatch(BaseModel):
    uid: str = Field(json_schema_extra={"from": "find_box.box_uid"})

    def matches(self, name):
        return bool(name)


class CrateMatch(BoxMatch):
    pass


class Inner(BaseModel):
    labels: list[str] = Field(json_schema_extra={"from": "open_inner.labels"})


MODELS = {"box": BoxMatch, "crate": CrateMatch, "bin": BoxMatch, "tray": BoxMatch}


def plan(input: Signature.Input):
    wanted = input.name
    phases = [Phase("find", lambda: [f"find_{input.kind} query={input.name} mode={input.mode}"
                                     + (f" note={input.note}" if input.note else "")],
                    output=MODELS[input.kind])]
    if input.kind == "crate" and input.mode == "deep":
        phases.append(Phase("inner", lambda find: [f"open_inner of {uid_of(find)}"], output=Inner,
                            optional=True))
    phases.append(Phase("audit", lambda: ["run_audit"], when=input.audit))
    phases.append(Phase("answer", lambda find: [f"say what {uid_of(find)} holds"],
                        when=lambda find: find.matches(
                            wanted)))
    return phases


def uid_of(find=None):
    return find.uid if find else "the record found"
'''


@pytest.fixture
def shelf_workflow(tmp_path):
    folder = tmp_path / "shelf_wf"
    (folder / "_skills").mkdir(parents=True)
    for name, text in {"walk_boxes": WALK_BOXES, "pick_kit": PICK_KIT}.items():
        (folder / "_skills" / f"{name}.py").write_text(textwrap.dedent(text))
    return str(folder)


def _plan(text: str, title: str) -> str:
    """The section of one plan, from its title to the next."""
    start = text.index(f"### Plan {title}\n")
    end = text.find("\n### Plan ", start + 1)
    return text[start:end if end != -1 else len(text)]


def test_a_fan_out_over_a_literal_list_renders_each_item(shelf_workflow):
    """Fix 1: each item of the list is its own sub-plan; none is rendered as None."""
    text = skill_preview.render_previews(shelf_workflow)["walk_boxes.md"]
    assert "=None" not in text
    both = _plan(text, "when second = given")
    assert "For item 1 of 2, `name` = `<first>`:\n\n1. `open_box label=<first>`" in both
    assert "For item 2 of 2, `name` = `<second>`:\n\n1. `open_box label=<second>`" in both
    # a fan-out over an empty list runs nothing, and says so
    none = _plan(text, "when scope = named, first = empty")
    assert skill_preview.NO_ITEMS_NOTE in none
    assert "1. `say that no box was opened`" in none


def test_the_runs_column_names_the_deciding_input_only(shelf_workflow):
    """Fix 2: a phase that runs for one input value says so, not every combination."""
    text = skill_preview.render_previews(shelf_workflow)["pick_kit.md"]
    assert "| `audit` | none | only when audit = true | no |" in text
    # An optional phase says so in its "runs" cell, not in its settings.
    assert ("| `inner` | `Inner` | only when kind = crate, mode = deep; optional (its failure "
            "does not stop the job) | no |") in text
    assert "optional=" not in text
    assert "(the rest as in the first)" not in text
    assert skill_preview.describe_inputs([(1, 0)], [(0, 0), (0, 1), (1, 0), (1, 1)], [
        skill_preview._Axis("a", (False, True), ("false", "true")),
        skill_preview._Axis("b", (False, True), ("false", "true"))]) == "when a = true, b = false"
    axes = [skill_preview._Axis("a", (0, 1, 2), ("x", "y", "z"))]
    assert skill_preview.describe_inputs([(0,), (1,)], [(0,), (1,), (2,)], axes) == "unless a = z"


def test_fan_out_caps_and_outputs_are_shown_with_the_inputs_that_give_them(shelf_workflow):
    """Fixes 3 and 8: per-variant fan-outs and output models each carry their condition."""
    previews = skill_preview.render_previews(shelf_workflow)
    cell = next(line for line in previews["walk_boxes.md"].splitlines() if line.startswith("| `box` |"))
    alternatives = cell.split(" | ")[3].split("<br>")
    assert "when scope = every, boxes = 3: per item of `shelf.box_uids` as `uid`; at most 3" in alternatives
    assert ("when second = given: per item of `['<first>', '<second>']` as `name`; not bound "
            "to a record") in alternatives
    assert all(a.startswith(("when ", "unless ")) for a in alternatives)
    find = next(line for line in previews["pick_kit.md"].splitlines() if line.startswith("| `find` |"))
    assert "when kind = crate: `CrateMatch`" in find and "unless kind = crate: `BoxMatch`" in find


def test_placeholders_survive_splitting_and_number_conversion(shelf_workflow):
    """Fixes 4 and 5: a split on "_" keeps the whole placeholder; int() gives 999999, said so."""
    text = skill_preview.render_previews(shelf_workflow)["walk_boxes.md"]
    every = _plan(text, "when scope = every, boxes = 0")
    assert ("1. `say that <shelf.box_labels[0]> (kind <shelf.box_labels[0]>) is on the shelf "
            "and weighs 999999`") in every
    assert "(kind <shelf)" not in text and "weighs 0" not in text
    assert "999999" in skill_preview.PLACEHOLDER_NOTE


def test_placeholder_facts_are_consistent_so_the_matched_path_renders(shelf_workflow):
    """Fix 6: a fan-out item's uid and label are the iterated list's entries, so code that
    looks the uid up in the list finds it, and later items show their own steps."""
    text = skill_preview.render_previews(shelf_workflow)["walk_boxes.md"]
    every = _plan(text, "when scope = every, boxes = 0")
    assert ("For the first item, `uid` = `<shelf.box_uids[0]>`:\n\n"
            "1. `open_box box_uid=<shelf.box_uids[0]> (<shelf.box_labels[0]>)`") in every
    assert ("For each later item (shown for the second), `uid` = `<shelf.box_uids[1]>`:\n\n"
            "1. `go_up`\n2. `open_box box_uid=<shelf.box_uids[1]> (<shelf.box_labels[1]>)`") in every
    assert "not on the shelf" not in every
    assert "open_box with the uid of the next box" not in text


def test_int_inputs_are_varied_and_samples_respect_their_bounds(shelf_workflow):
    """Fix 7: an int that changes the steps is rendered at its sample and a typical value;
    a sample below the field's ``ge`` is raised to it."""
    previews = skill_preview.render_previews(shelf_workflow)
    walk = previews["walk_boxes.md"]
    assert "the inputs that change them give: `scope`, `first`, `second`, `boxes`." in walk
    assert "### Plan when scope = every, boxes = 3\n" in walk
    assert "Settings: `over='shelf.box_uids'`, `item='uid'`, `max_items=3`." in walk
    kit = previews["pick_kit.md"]
    assert "| `count` | int | `0` | `1` |" in kit
    assert "`count`" not in kit.split("## Plan")[1].split("\n\n")[1]   # never changes the steps


def test_many_renders_show_plans_that_need_two_changed_inputs(shelf_workflow):
    """Fix 8: past MAX_VARIANTS renders, a pairwise cover is shown, so a plan that needs
    two inputs changed from the sample is rendered."""
    text = skill_preview.render_previews(shelf_workflow)["pick_kit.md"]
    assert "They give 48 different plans, too many to show each" in text
    titles = [line for line in text.splitlines() if line.startswith("### Plan ")]
    assert len(titles) < 48
    assert any("kind = crate, mode = deep" in t for t in titles)
    crate = _plan(text, next(t[len("### Plan "):] for t in titles if "kind = crate, mode = deep" in t))
    assert "1. `open_inner of <find.uid>`" in crate
    sizes = [4, 3, 2, 2]
    cover = skill_preview._pairwise(sizes, (0, 0, 0, 0))
    assert cover[0] == (0, 0, 0, 0) and len(cover) < 48
    for i, j in itertools.combinations(range(4), 2):
        assert {(c[i], c[j]) for c in cover} == set(itertools.product(range(sizes[i]), range(sizes[j])))


def test_a_condition_shows_its_bound_values_on_one_line(shelf_workflow):
    """Fix 9: a closure name reads as its value, and a lambda over two lines joins without
    a space after the bracket."""
    text = skill_preview.render_previews(shelf_workflow)["pick_kit.md"]
    assert "| `answer` | none | only when `lambda find: find.matches('<name>')` | no |" in text
    assert "( " not in text

    def outer(kind):
        return dict(when=lambda found: found.kind == kind and (
            found.size > LIMIT))

    assert skill_preview.condition_text(outer("crate")["when"], "when") == (
        "lambda found: found.kind == 'crate' and (found.size > 12)")


LIMIT = 12

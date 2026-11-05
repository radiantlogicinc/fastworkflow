"""Skill previews (fix-553q): ``fastworkflow command preview`` and ``fastworkflow.command_preview``.

Offline, no model: a neutral catalog workflow with a phased skill (with a conditional
phase), a flat skill and a skill without inputs, rendered to markdown.
"""
from __future__ import annotations

import itertools
import re
import os
import subprocess
import sys
import textwrap
import uuid
from typing import Optional

import pytest
from pydantic import BaseModel

from fastworkflow import cli, planning_commands, command_preview

SHOW_ITEM = '''
"""show_item: one named item, its parts and, when asked, its open checks."""
from pydantic import BaseModel, Field

from .count_checks import Signature as CountChecks
from .find_item import Signature as FindItem
from .list_parts import Signature as ListParts
from .open_item_by_uid import Signature as OpenItem


class Signature:
    """Show one named item and its parts, and its open checks when asked. One job per item."""

    plain_utterances = ["what parts does the Alpha item have", "show item Beta and its checks"]

    class Input(BaseModel):
        name: str = Field(description="the item's name",
                          json_schema_extra={"in_request": "subject"})
        checks: bool = Field(default=False, description="the request asks about its open checks",
                             json_schema_extra={"verify": True})


class ItemMatch(BaseModel):
    """The items the search listed."""
    found: FindItem.Output

    @property
    def item_uids(self) -> list[str]:
        return self.found.item_uids


class ItemFacts(BaseModel):
    opened: OpenItem.Output
    parts: ListParts.Output

    @property
    def uid(self) -> str:
        return self.opened.item_uid

    @property
    def part_labels(self) -> list[str]:
        return self.parts.labels

    @property
    def weight(self) -> float:
        return self.parts.weight


class CheckFacts(BaseModel):
    counted: CountChecks.Output

    @property
    def total(self) -> int:
        return self.counted.total


class PlanGenerator:
    """find -> item -> checks (when asked) -> report."""

    def __phases__(self, input):
        checks = [self.checks] if input.checks else []
        return [self.find, self.item, *checks]

    def find(self, input) -> ItemMatch:
        return ["open_catalog", f"find_item query={input.name}"]

    def item(self, find) -> ItemFacts:
        return [f"open_item_by_uid item_uid={find.item_uids[0]}", "list_parts (full list)"]

    def checks(self, item) -> CheckFacts:
        return [f"count_checks on {item.uid}"]

    def __result__(self, input, item, checks=None):
        counted = f"; open checks: {checks.total}" if checks is not None else ""
        return (f"{item.uid} has the parts {self._parts(item)}{counted}, "
                f"weight {item.weight:.1f}.")

    def _parts(self, item):
        return ", ".join(item.part_labels)
'''

LIST_PARTS = '''
"""list_parts_of: the parts of one named item or kit, optionally only one supplier's."""
from typing import Literal

import fastworkflow

from pydantic import BaseModel, Field

FIND = {"item": "find_item", "kit": "find_kit"}


class Signature:
    """List the parts of one named item or kit. One job per record."""

    class Input(BaseModel):
        name: str = Field(description="the record's name | as written")
        kind: Literal["item", "kit"] = Field(default="item", description="what the request is about",
                                             json_schema_extra={"verify": {"item": "one item",
                                                                           "kit": "a set of items"}})
        supplier: str = Field(default="", description="one supplier to keep; empty for all")
        detail: bool = Field(default=False, description="never changes the steps")
        count: int = Field(default=2, description="how many parts to open")

    @staticmethod
    def validate_extracted_parameters(workflow, command, cmd_parameters, command_text=None):
        if cmd_parameters.kind == "kit" and cmd_parameters.supplier:
            raise fastworkflow.Refused("a kit is listed for every supplier")
        return None


class PlanGenerator:
    def steps(self, input):
        out = lookup_steps(input.kind, input.name)
        out.append(f"list_parts supplier={input.supplier}" if input.supplier else "list_parts")
        return out + ["go_up", "say what was shown"]


def lookup_steps(kind, name):
    return [f"{FIND[kind]} query={name}"]
'''

CLOSE_OUT = '''
class Signature:
    """Go back to the catalog. Fixed steps."""


class PlanGenerator:
    def steps(self):
        return ["go_up", "say what was shown"]
'''


#: The execution commands the fixtures' steps name (a step must name a command of the workflow; checked
#: on load). Each stub takes only the Input fields its fixtures pass and writes only the Output fields a fact
#: source reads (``(field, type, default)``), so nothing else in a preview is changed by the stub.
STEP_STUBS = {
    "compare_alpha_beta": ((), ()),
    "count_checks": ((), (("total", "int", 0),)),
    "find_bin": (("query", "mode", "note"), (("box_uid", "str", ""),)),
    "find_box": (("query", "mode", "note"), (("box_uid", "str", ""),)),
    "find_crate": (("query", "mode", "note"), (("box_uid", "str", ""),)),
    "find_item": (('query', 'supplier'), (("item_uids", "list[str]", "list"),)),
    "find_kit": (("query",), ()),
    "find_shelf": (('query',), (("shelf_uid", "str", ""),)),
    "find_tray": (("query", "mode", "note"), (("box_uid", "str", ""),)),
    "list_boxes": ((), (("box_uids", "list[str]", "list"), ("labels", "list[str]", "list"))),
    "list_items": ((), ()),
    "list_parts": (('supplier',), (("labels", "list[str]", "list"), ("weight", "float", 0.0))),
    "open_box": (('box_uid', 'label'), (("box_uid", "str", ""), ("label", "str", ""))),
    "open_box_by_uid": ((), (("box_uid", "str", ""),)),
    "open_catalog": ((), ()),
    "open_inner": ((), (("labels", "list[str]", "list"),)),
    "open_item_by_uid": (('item_uid',), (("item_uid", "str", ""),)),
    "run_audit": ((), ()),
    "run_check": ((), (("passed", "bool", False),)),
    "show_alpha": ((), ()),
    "show_beta": ((), ()),
    "show_delta": ((), ()),
    "show_epsilon": ((), ()),
    "show_eta": ((), ()),
    "show_gamma": ((), ()),
    "show_iota": ((), ()),
    "show_kappa": ((), ()),
    "show_label": ((), ()),
    "show_mu": ((), ()),
    "show_nu": ((), ()),
    "show_theta": ((), ()),
    "show_zeta": ((), ()),
    "weigh_box": ((), (("weight", "float", 0.0),)),
}


def _write_step_stubs(folder: str) -> None:
    """Write a stub execution command for each name in STEP_STUBS into the workflow's _commands/."""
    for name, (params, outputs) in STEP_STUBS.items():
        inputs = "\n".join(f'        {p}: str = Field(default="", description="a stub parameter")' for p in params)
        fields = "\n".join(f'        {f}: {t} = Field({"default_factory=list" if d == "list" else f"default={d!r}"}, description="a stub output")'
                            for f, t, d in outputs)
        text = ("import fastworkflow\nfrom pydantic import BaseModel, Field\n\n\nclass Signature:\n"
                '    """Stub: a command the preview fixtures\' steps name."""\n\n    class Input(BaseModel):\n'
                + (inputs or "        pass") + "\n\n    class Output(BaseModel):\n"
                + (fields or "        pass") + "\n\n\n"
                "class ResponseGenerator:\n    def __call__(self, workflow: fastworkflow.Workflow, command: str, command_parameters):\n"
                "        return Signature.Output()\n")
        with open(os.path.join(folder, "_commands", f"{name}.py"), "w", encoding="utf-8") as handle:
            handle.write(text)


def _workflow(root) -> str:
    folder = root / "catalog_wf"
    (folder / "_commands").mkdir(parents=True)
    _write_step_stubs(str(folder))
    for name, text in {"show_item": SHOW_ITEM, "list_parts_of": LIST_PARTS,
                       "close_out": CLOSE_OUT}.items():
        (folder / "_commands" / f"{name}.py").write_text(textwrap.dedent(text))
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


def test_every_planning_command_renders_one_preview(workflow):
    previews = command_preview.render_previews(workflow)
    assert sorted(previews) == ["close_out.md", "list_parts_of.md", "show_item.md"]
    for name, text in previews.items():
        assert text.startswith("<!-- Generated by `fastworkflow command preview` from _commands/")
        assert text.endswith("\n") and not text.endswith("\n\n")
        assert workflow not in text   # no absolute path


def test_phased_preview_shows_each_phase_its_facts_and_the_condition_both_ways(workflow):
    text = command_preview.render_previews(workflow)["show_item.md"]
    assert "Source: `_commands/show_item.py`.\n" in text and "Kind:" not in text
    assert "| `name` | str | required | `<name>` |  | subject | the item's name |" in text
    assert "| `checks` | bool | false | false | yes |  | the request asks about its open checks |" in text
    assert "- what parts does the Alpha item have" in text
    # the phases, and when each runs
    assert "| `find` | `ItemMatch` | always |" in text
    assert "| `checks` | `CheckFacts` | only when checks = true |" in text
    assert "| `find` | `ItemMatch` | always |" in text and "`__result__`" in text
    # facts and their sources: each typed field says which command's Output it is
    assert "| `found` | `Output` | `find_item`'s `Output` |" in text
    assert "| `parts` | `Output` | `list_parts`'s `Output` |" in text
    assert "| `counted` | `Output` | `count_checks`'s `Output` |" in text
    assert "The items the search listed." in text
    # the condition both ways, earlier facts as placeholders
    off = text.index("### Plan when checks = false")
    on = text.index("### Plan when checks = true")
    assert off < on
    assert "1. `open_item_by_uid item_uid=<find.found.item_uids[0]>`" in text
    assert "Reads: `find`. Output: `ItemFacts`." in text
    assert "count_checks" not in text[off:on]
    # the check reads the uid the item phase opened (one placeholder for one value)
    assert "1. `count_checks on <item.opened.item_uid>`" in text[on:]
    # the answer __result__ writes, per plan, with the placeholders escaped for Markdown
    answer_off = text[text.index("### Answer when checks = false"):text.index("### Answer when checks = true")]
    answer_on = text[text.index("### Answer when checks = true"):]
    assert ("Written by `__result__`: \\<item.opened.item_uid> has the parts \\<item.parts.labels[0]>; "
            "open checks: \\<checks.counted.total>, weight \\<item.parts.weight>.") in answer_on
    assert ("Written by `__result__`: \\<item.opened.item_uid> has the parts \\<item.parts.labels[0]>, "
            "weight \\<item.parts.weight>.") in answer_off
    # a phase that reads the same as in the first plan is one line
    assert "As in the plan when checks = false." in text[on:]


def test_flat_preview_renders_the_inputs_that_change_the_steps(workflow):
    text = command_preview.render_previews(workflow)["list_parts_of.md"]
    assert "Source: `_commands/list_parts_of.py`.\n" in text and "Kind:" not in text
    assert "| `name` | str | required | `<name>` |  |  | the record's name \\| as written |" in text
    assert "| `kind` | one of item, kit | `item` | `item` |" in text
    assert "| `count` | int | `2` | `2` |" in text
    assert "strips the words" not in text   # the strip option is retired: the cleanup is validator code
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


def test_fixed_planning_command_preview(workflow):
    text = command_preview.render_previews(workflow)["close_out.md"]
    assert "None: the steps are fixed." in text
    assert "## Steps\n\n1. `go_up`\n2. `say what was shown`" in text


def test_render_is_byte_stable_across_processes(workflow, tmp_path):
    first = command_preview.render_previews(workflow)
    assert command_preview.render_previews(workflow) == first
    out = tmp_path / "other_process"
    env = {**os.environ, "PYTHONHASHSEED": "12345"}
    subprocess.run([sys.executable, "-m", "fastworkflow.cli", "command", "preview", workflow,
                    "--out", str(out)], check=True, env=env, capture_output=True)
    assert {f: (out / f).read_text(encoding="utf-8") for f in os.listdir(out)} == first


def test_default_out_dir_is_docs_next_to_the_workflow(workflow):
    assert command_preview.default_out_dir(workflow) == os.path.join(
        os.path.dirname(workflow), "docs", "planning_command_previews")


def test_check_passes_on_fresh_output_and_fails_on_an_edited_missing_or_extra_one(
        workflow, tmp_path, monkeypatch, capsys):
    out = tmp_path / "previews"
    assert _run_cli(monkeypatch, "command", "preview", workflow, "--check", "--out", str(out)) == 1
    assert _run_cli(monkeypatch, "command", "preview", workflow, "--out", str(out)) == 0
    assert _run_cli(monkeypatch, "command", "preview", workflow, "--check", "--out", str(out)) == 0
    assert command_preview.check_previews(workflow, str(out)) == []

    edited = out / "show_item.md"
    edited.write_text(edited.read_text(encoding="utf-8").replace("open_catalog", "open_shop"),
                      encoding="utf-8")
    capsys.readouterr()
    assert _run_cli(monkeypatch, "command", "preview", workflow, "--check", "--out", str(out)) == 1
    assert "stale: show_item.md" in capsys.readouterr().out

    (out / "close_out.md").unlink()
    (out / "notes.md").write_text("hand-written\n", encoding="utf-8")
    assert command_preview.check_previews(workflow, str(out)) == [
        "missing: close_out.md", "stale: show_item.md", "extra: notes.md"]


def test_write_removes_stale_generated_previews_and_keeps_other_files(workflow, tmp_path):
    out = tmp_path / "previews"
    command_preview.write_previews(workflow, str(out))
    gone = out / "old_planning_command.md"
    gone.write_text(command_preview.MARKER.format(source="_commands/old_planning_command.py") + "\n",
                    encoding="utf-8")
    (out / "notes.md").write_text("hand-written\n", encoding="utf-8")
    result = command_preview.write_previews(workflow, str(out))
    assert result["removed"] == ["old_planning_command.md"] and result["kept"] == ["notes.md"]
    assert result["unchanged"] == ["close_out.md", "list_parts_of.md", "show_item.md"]
    assert not gone.exists()


def test_check_fails_on_a_planning_command_file_the_loader_skips(workflow, monkeypatch, capsys):
    command_preview.write_previews(workflow)
    with open(os.path.join(workflow, "_commands", "broken.py"), "w", encoding="utf-8") as handle:
        # a planning command (a top-level PlanGenerator) whose steps() returns no list: a PlanningCommandFormatError
        handle.write('class Signature:\n    """Broken."""\n\n\nclass PlanGenerator:\n'
                     '    def steps(self):\n        return "a step"\n')
    assert _run_cli(monkeypatch, "command", "preview", workflow, "--check") == 1
    assert "broken: PlanningCommandFormatError:" in capsys.readouterr().out
    # preview_main itself reports it as a problem line, not an exception
    import argparse
    assert command_preview.preview_main(argparse.Namespace(workflow_folder=workflow, out=None, check=True)) == 1
    out = capsys.readouterr().out
    assert "cannot be checked: planning command file(s) not loaded:" in out and "  broken: PlanningCommandFormatError:" in out


COPY_FACTS = '''
"""copy_item: a facts field that copies a value from a command, which is no longer read."""
from pydantic import BaseModel, Field


class Signature:
    """Find one named item by copying its uids."""

    class Input(BaseModel):
        name: str = Field(description="the item's name")


class PlanGenerator:
    class ItemCopy(BaseModel):
        item_uids: list[str] = Field(json_schema_extra={"from": "find_item.item_uids"})

    def find(self, input) -> ItemCopy:
        return [f"find_item query={input.name}"]
'''


def test_check_fails_on_a_fact_source_the_workflow_does_not_have(workflow, monkeypatch, capsys):
    """A facts field that copies (``"from"``) is a load error: the file is skipped, and
    ``command preview --check`` fails and names the command, which a typed field never needs."""
    command_preview.write_previews(workflow)
    assert _run_cli(monkeypatch, "command", "preview", workflow, "--check") == 0
    with open(os.path.join(workflow, "_commands", "copy_item.py"), "w", encoding="utf-8") as handle:
        handle.write(textwrap.dedent(COPY_FACTS))
    capsys.readouterr()
    assert _run_cli(monkeypatch, "command", "preview", workflow, "--check") == 1
    out = capsys.readouterr().out
    assert "planning command file(s) not loaded" in out and "copy_item" in out
    errors = [p.message for p in planning_commands.resolve_field_sources(workflow).errors
              if p.command == "copy_item"]
    assert errors and all(
        "ItemCopy.item_uids is list[str]; a facts field is typed with a command's Output or Input. "
        "Compute other values in a property or method of ItemCopy; "
        "copies are no longer read (fact_from and \"from\" are removed)" in m for m in errors)


def test_a_workflow_without_planning_commands_is_an_error(tmp_path):
    with pytest.raises(command_preview.PreviewError):
        command_preview.render_previews(str(tmp_path))


# ---------------------------------------------------------------- fix-hia1.9: conditions, until, flat

CHECK_ITEMS = '''
"""check_items: the named items checked one by one, phased only for the deep check."""
from pydantic import BaseModel, Field

from fastworkflow.planning import phase

from .find_item import Signature as FindItem
from .open_item_by_uid import Signature as OpenItem
from .run_check import Signature as RunCheck


class Signature:
    """Check the named items one by one until two pass. One job for the whole list."""

    class Input(BaseModel):
        names: str = Field(description="the items' names, comma separated")
        deep: bool = Field(default=False, description="the request asks for the deep check")


class Check(BaseModel):
    opened: OpenItem.Output
    ran: RunCheck.Output

    @property
    def uid(self) -> str:
        return self.opened.item_uid

    @property
    def passed(self) -> bool:
        return self.ran.passed


class PlanGenerator:
    """Without the deep check: one plan (flat). With it: find, checks (once per match, until
    two passed), extra (more than one match); __result__ writes the answer (when an item was
    checked)."""

    def __phases__(self, input):
        if not input.deep:
            return [self.quick]
        return [self.find, self.checks, self.extra]

    def quick(self, input):
        return [f"find_item query={input.names}", "run_check on each match", "say which passed"]

    def find(self, input) -> FindItem.Output:
        return [f"find_item query={input.names}"]

    @phase(over="find.item_uids", max_items=5, noun="matches",
           until=lambda checks: sum(c.passed for c in checks) >= 2)
    def checks(self, uid) -> Check:
        return [f"open_item_by_uid item_uid={uid}", "run_check"]

    def extra(self, find):
        if not len(find.item_uids) > 1:
            return []
        return [f"count_checks for {len(find.item_uids)} items"]

    def __result__(self, checks):
        return f"{self._passed(checks) if checks else '0 items passed: none'}."

    def _passed(self, checks):
        passed = [c.uid for c in checks if c.passed]
        return f"{len(passed)} items passed: {', '.join(passed) or 'none'}"
'''


@pytest.fixture
def checks_workflow(tmp_path):
    folder = tmp_path / "checks_wf"
    (folder / "_commands").mkdir(parents=True)
    _write_step_stubs(str(folder))
    (folder / "_commands" / "check_items.py").write_text(textwrap.dedent(CHECK_ITEMS))
    return str(folder)


def test_conditions_fan_out_settings_and_a_flat_plan_are_rendered(checks_workflow):
    text = command_preview.render_previews(checks_workflow)["check_items.md"]
    assert "| phase | output | runs | fans out |" in text
    # When the call runs as phases is said once; the table is about the phased calls.
    assert ("The call runs as phases only when deep = true; for other inputs it runs flat, "
            "as one plan.") in text
    assert "| `find` | `find_item`'s `Output` | always | no |" in text
    assert ("| `checks` | `Check` | always | per item of `find.item_uids` as "
            "`uid` (matches); at most 5; until `lambda checks: sum(c.passed for c in checks) >= 2` |") in text
    # A phase's condition is the tests its method made on placeholder facts (len() reads
    # one item: a test that the list is not empty).
    assert "| `extra` | none | only when `find.item_uids` | no |" in text
    assert "| `extra` | none | only when `find.item_uids` | no |" in text
    assert "Settings: `when=find.item_uids`." in text
    assert ("Settings: `over='find.item_uids'`, `item='uid'`, `max_items=5`, "
            "`until=lambda checks: sum(c.passed for c in checks) >= 2`, `noun='matches'`.") in text
    # For deep = false __call__ gives one phase that reads no facts: it runs flat, said so.
    flat = text.index("### Plan when deep = false")
    phased = text.index("### Plan when deep = true")
    assert (f"{command_preview.FLAT_NOTE}"
            "\n\n1. `find_item query=<names>`\n2. `run_check on each match`") in text[flat:phased]


def test_a_phase_computing_on_facts_is_marked_as_rendered_with_placeholders(checks_workflow,
                                                                           workflow):
    text = command_preview.render_previews(checks_workflow)["check_items.md"]
    note = f"_{command_preview.PLACEHOLDER_NOTE}_"
    phases = text.split("#### Phase ")[1:]
    marked = {part.split("`")[1]: note in part for part in phases}
    # `find` and `checks` read no earlier facts (an item is not a fact); the rest do.
    assert marked == {"find": False, "checks": False, "extra": True}
    # The answer computes "0 items passed" from placeholder facts: not trusted, but shown.
    assert "Written by `__result__`: 0 items passed: none." in text
    shown = command_preview.render_previews(workflow)["show_item.md"]
    assert shown.count(note) >= 2   # item and report read earlier facts


def test_condition_text_reads_the_lambda_source_or_falls_back_to_the_name():
    def named(find):
        return True

    one = [lambda find: ["x"], lambda find: bool(find)]   # two lambdas on one line
    assert command_preview.condition_text(named) == "named"
    assert command_preview.condition_text(one[1]) == "<lambda>"   # cannot be told apart
    pair = dict(steps=lambda find: ["x"], when=lambda find: bool(find))
    assert command_preview.condition_text(pair["when"], "when") == "lambda find: bool(find)"
    assert command_preview.condition_text(eval("lambda find: True")) == "<lambda>"   # no source


def test_the_new_settings_render_byte_stable(checks_workflow, tmp_path):
    first = command_preview.render_previews(checks_workflow)
    assert command_preview.render_previews(checks_workflow) == first
    out = tmp_path / "other_process"
    env = {**os.environ, "PYTHONHASHSEED": "777"}
    subprocess.run([sys.executable, "-m", "fastworkflow.cli", "command", "preview", checks_workflow,
                    "--out", str(out)], check=True, env=env, capture_output=True)
    assert {f: (out / f).read_text(encoding="utf-8") for f in os.listdir(out)} == first


# ---------------------------------------------------------------- the readability fixes

WALK_BOXES = '''
"""walk_boxes: one shelf's boxes, every box or only the named ones, each opened and weighed."""
from typing import Literal

from pydantic import BaseModel, Field

from fastworkflow.planning import phase

from .list_boxes import Signature as ListBoxes
from .open_box import Signature as OpenBox
from .weigh_box import Signature as WeighBox


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
        return None


class Box(BaseModel):
    opened: OpenBox.Output
    weighed: WeighBox.Output

    @property
    def uid(self) -> str:
        return self.opened.box_uid

    @property
    def label(self) -> str:
        return self.opened.label

    @property
    def weight(self) -> float:
        return self.weighed.weight


class PlanGenerator:
    def shelf(self, input) -> ListBoxes.Output:
        return [f"find_shelf query={input.shelf}", "list_boxes"]

    @phase(over=lambda input: [n for n in (input.first, input.second) if n]
           if input.scope == "named" else "shelf.box_uids",
           bind=lambda input: input.scope != "named",
           max_items=lambda input: None if input.scope == "named" else input.boxes or None)
    def box(self, input, uid, shelf=None) -> Box:
        if input.scope == "named":
            return [f"open_box label={uid}", "weigh_box"]
        return box_steps(uid, shelf)

    def __result__(self, shelf, box=None):
        if not box:
            return "No box was opened."
        lines = []
        for one in box:
            kind = one.label.split("_", 1)[0]
            lines.append(f"{one.label} (kind {kind}) is on the shelf and weighs {int(one.weight)}.")
        return "\\n".join(lines)


def box_steps(uid, shelf):
    if uid not in shelf.box_uids:
        return ["open_box with the uid of the next box", "weigh_box"]
    index = next(i for i, one in enumerate(shelf.box_uids) if one == uid)
    return (["go_up"] if index else []) + [
        f"open_box box_uid={uid} ({shelf.labels[index]})", "weigh_box"]
'''

PICK_KIT = '''
"""pick_kit: one named record of four kinds, found in one of three modes."""
from typing import Literal

from pydantic import BaseModel, Field

from fastworkflow.planning import phase

from .find_bin import Signature as FindBin
from .find_box import Signature as FindBox
from .find_crate import Signature as FindCrate
from .find_tray import Signature as FindTray
from .open_inner import Signature as OpenInner


class Signature:
    """Find one named record and say what it holds. One job per record."""

    class Input(BaseModel):
        name: str = Field(description="the record's name")
        kind: Literal["box", "crate", "bin", "tray"] = Field(default="box", description="its kind")
        mode: Literal["quick", "deep", "full"] = Field(default="quick", description="how to look")
        audit: bool = Field(default=False, description="the request asks for an audit")
        note: str = Field(default="", description="a note to add")
        count: int = Field(default=0, ge=1, description="never changes the steps")


class _Match:
    @property
    def uid(self) -> str:
        return self.found.box_uid

    def matches(self, name):
        return bool(name)


class BoxMatch(_Match, BaseModel):
    found: FindBox.Output


class CrateMatch(_Match, BaseModel):
    found: FindCrate.Output


class BinMatch(_Match, BaseModel):
    found: FindBin.Output


class TrayMatch(_Match, BaseModel):
    found: FindTray.Output


class Inner(BaseModel):
    opened: OpenInner.Output

    @property
    def labels(self) -> list[str]:
        return self.opened.labels


MODELS = {"box": BoxMatch, "crate": CrateMatch, "bin": BinMatch, "tray": TrayMatch}


class PlanGenerator:
    def __phases__(self, input):
        inner = [self.inner] if input.kind == "crate" and input.mode == "deep" else []
        audit = [self.audit] if input.audit else []
        return [self.find, *inner, *audit, self.answer]

    @phase(output=lambda input: MODELS[input.kind])
    def find(self, input):
        return [f"find_{input.kind} query={input.name} mode={input.mode}"
                + (f" note={input.note}" if input.note else "")]

    @phase(optional=True)
    def inner(self, find) -> Inner:
        return [f"open_inner of {find.uid}"]

    def audit(self):
        return ["run_audit"]

    def answer(self, input, find):
        if not find.matches(input.name):
            return []
        return [f"say what {find.uid} holds"]
'''


@pytest.fixture
def shelf_workflow(tmp_path):
    folder = tmp_path / "shelf_wf"
    (folder / "_commands").mkdir(parents=True)
    _write_step_stubs(str(folder))
    for name, text in {"walk_boxes": WALK_BOXES, "pick_kit": PICK_KIT}.items():
        (folder / "_commands" / f"{name}.py").write_text(textwrap.dedent(text))
    return str(folder)


def _plan(text: str, title: str) -> str:
    """The section of one plan, from its title to the next."""
    start = text.index(f"### Plan {title}\n")
    end = text.find("\n### Plan ", start + 1)
    return text[start:end if end != -1 else len(text)]


def test_a_fan_out_over_a_literal_list_renders_each_item(shelf_workflow):
    """Fix 1: each item of the list is its own sub-plan; none is rendered as None."""
    text = command_preview.render_previews(shelf_workflow)["walk_boxes.md"]
    assert "=None" not in text
    both = _plan(text, "when second = given")
    assert "For item 1 of 2, `uid` = `<first>`:\n\n1. `open_box label=<first>`" in both
    assert "For item 2 of 2, `uid` = `<second>`:\n\n1. `open_box label=<second>`" in both
    # a fan-out over an empty list runs nothing, and says so
    none = _plan(text, "when scope = named, first = empty")
    assert command_preview.NO_ITEMS_NOTE in none
    assert "Written by `__result__`: No box was opened." in text[text.index("### Answer when scope = named, first = empty"):]


def test_the_runs_column_names_the_deciding_input_only(shelf_workflow):
    """Fix 2: a phase that runs for one input value says so, not every combination."""
    text = command_preview.render_previews(shelf_workflow)["pick_kit.md"]
    assert "| `audit` | none | only when audit = true | no |" in text
    # An optional phase says so in its "runs" cell, not in its settings.
    assert ("| `inner` | `Inner` | only when kind = crate, mode = deep; optional (its failure "
            "does not stop the job) | no |") in text
    assert "optional=" not in text
    assert "(the rest as in the first)" not in text
    assert command_preview.describe_inputs([(1, 0)], [(0, 0), (0, 1), (1, 0), (1, 1)], [
        command_preview._Axis("a", (False, True), ("false", "true")),
        command_preview._Axis("b", (False, True), ("false", "true"))]) == "when a = true, b = false"
    axes = [command_preview._Axis("a", (0, 1, 2), ("x", "y", "z"))]
    assert command_preview.describe_inputs([(0,), (1,)], [(0,), (1,), (2,)], axes) == "unless a = z"


def test_fan_out_caps_and_outputs_are_shown_with_the_inputs_that_give_them(shelf_workflow):
    """Fixes 3 and 8: per-variant fan-outs and output models each carry their condition."""
    previews = command_preview.render_previews(shelf_workflow)
    cell = next(line for line in previews["walk_boxes.md"].splitlines() if line.startswith("| `box` |"))
    alternatives = cell.split(" | ")[3].split("<br>")
    assert "when scope = every, boxes = 3: per item of `shelf.box_uids` as `uid`; at most 3" in alternatives
    assert ("when second = given: per item of `['<first>', '<second>']` as `uid`; not bound "
            "to a record") in alternatives
    assert all(a.startswith(("when ", "unless ")) for a in alternatives)
    find = next(line for line in previews["pick_kit.md"].splitlines() if line.startswith("| `find` |"))
    assert "when kind = crate: `CrateMatch`" in find and "when kind = box: `BoxMatch`" in find
    assert "when kind = bin: `BinMatch`" in find and "when kind = tray: `TrayMatch`" in find


def test_placeholders_survive_splitting_and_number_conversion(shelf_workflow):
    """Fixes 4 and 5: a split on "_" keeps the whole placeholder; int() gives 999999, said so."""
    text = command_preview.render_previews(shelf_workflow)["walk_boxes.md"]
    every = _plan(text, "when scope = every, boxes = 0")
    assert ("Written by `__result__`: \\<box[0].opened.label> (kind \\<box[0].opened.label>) is on "
            "the shelf and weighs 999999.") in text[text.index("### Answer when scope = every, boxes = 0"):]
    assert "(kind <shelf)" not in text and "weighs 0" not in text
    assert "999999" in command_preview.PLACEHOLDER_NOTE


def test_placeholder_facts_are_consistent_so_the_matched_path_renders(shelf_workflow):
    """Fix 6: a fan-out item's uid and label are the iterated list's entries, so code that
    looks the uid up in the list finds it, and later items show their own steps."""
    text = command_preview.render_previews(shelf_workflow)["walk_boxes.md"]
    every = _plan(text, "when scope = every, boxes = 0")
    assert ("For the first item, `uid` = `<shelf.box_uids[0]>`:\n\n"
            "1. `open_box box_uid=<shelf.box_uids[0]> (<shelf.labels[0]>)`") in every
    assert ("For each later item (shown for the second), `uid` = `<shelf.box_uids[1]>`:\n\n"
            "1. `go_up`\n2. `open_box box_uid=<shelf.box_uids[1]> (<shelf.labels[1]>)`") in every
    assert "not on the shelf" not in every
    assert "open_box with the uid of the next box" not in text


def test_int_inputs_are_varied_and_samples_respect_their_bounds(shelf_workflow):
    """Fix 7: an int that changes the steps is rendered at its sample and a typical value;
    a sample below the field's ``ge`` is raised to it."""
    previews = command_preview.render_previews(shelf_workflow)
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
    text = command_preview.render_previews(shelf_workflow)["pick_kit.md"]
    assert "They give 48 different plans, too many to show each" in text
    titles = [line for line in text.splitlines() if line.startswith("### Plan ")]
    assert len(titles) < 48
    assert any("kind = crate, mode = deep" in t for t in titles)
    crate = _plan(text, next(t[len("### Plan "):] for t in titles if "kind = crate, mode = deep" in t))
    assert "1. `open_inner of <find.found.box_uid>`" in crate
    sizes = [4, 3, 2, 2]
    cover = command_preview._pairwise(sizes, (0, 0, 0, 0))
    assert cover[0] == (0, 0, 0, 0) and len(cover) < 48
    for i, j in itertools.combinations(range(4), 2):
        assert {(c[i], c[j]) for c in cover} == set(itertools.product(range(sizes[i]), range(sizes[j])))


def test_a_condition_shows_its_bound_values_on_one_line(shelf_workflow):
    """Fix 9: a closure name reads as its value, and a lambda over two lines joins without
    a space after the bracket. A PlanGenerator phase's condition is the tests its method
    made on the placeholder facts (a call reads with no arguments)."""
    text = command_preview.render_previews(shelf_workflow)["pick_kit.md"]
    assert "| `answer` | none | only when `find.matches()` | no |" in text
    assert "( " not in text

    def outer(kind):
        return dict(when=lambda found: found.kind == kind and (
            found.size > LIMIT))

    assert command_preview.condition_text(outer("crate")["when"], "when") == (
        "lambda found: found.kind == 'crate' and (found.size > 12)")


LIMIT = 12


def test_a_planning_command_past_max_grid_lists_every_phase(tmp_path):
    """fix-6llp.16 (folds in .15): past MAX_GRID combinations the preview tries the grid
    the template uses, so a phase only some inputs reach is still listed."""
    from tests.test_planning_command_commands import WIDE

    folder = tmp_path / "shelves_wf"
    (folder / "_commands").mkdir(parents=True)
    _write_step_stubs(str(folder))
    (folder / "_commands" / "survey_shelf.py").write_text(WIDE)
    text = command_preview.render_previews(str(folder))["survey_shelf.md"]
    for phase in ["find", "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
                  "iota", "kappa", "lambda_", "mu", "nu", "pair", "report"]:
        assert f"| `{phase}` |" in text
    assert "| `pair` | none | only when alpha = true, beta = true |" in text


# ---------------------------------------------------------------- answer modes (fix-ajir.10)

@pytest.mark.parametrize("steps, line", [
    (["say that b1 has loans l1, l2"],
     "Answer: written by code, word for word (step 1 verbatim)."),
    (["say that b1 has loans l1, l2", "say that what b2 holds: l3", "say that l1 > l3",
      "apply no change"],
     "Answer: written by code, word for word (steps 1-3 verbatim; step 4 dropped as a guard)."),
    (["say that b1 has loans l1", "do not open them", "say that b2 has none", "never guess"],
     "Answer: written by code, word for word (steps 1 and 3 verbatim; steps 2 and 4 dropped "
     "as guards)."),
    (["say which is riskier"],
     "Answer: written by the model (step 1 needs judgement)."),
    (["say that b1 has loans l1", "say which loan is the oldest", "apply no change",
      "summarise the rest"],
     "Answer: written by the model (steps 2 and 4 need judgement)."),
    (["apply no change", "do not open anything"],
     "Answer: written by the model (no step is verbatim)."),
])
def test_answer_mode_reads_the_steps_with_the_runtime_rules(steps, line):
    from fastworkflow.planning_commands import number
    from fastworkflow.planning_phases import answer_by_code

    assert command_preview.answer_mode(steps) == line
    assert line.startswith("Answer: written by code") == answer_by_code(number(steps))


def test_answer_mode_is_by_code_only_for_say_that_and_guards_need_a_negation():
    # "say <question>: <text>" needs a judgement: only "say that" is copied (fix-ajir.10).
    assert command_preview.answer_mode(["say which is riskier: A has score 3, B has score 5"]) == (
        "Answer: written by the model (step 1 needs judgement).")
    # A step starting "apply" with no negation word is no guard: the model applies it.
    assert command_preview.answer_mode(["say that the total is 100",
                                      "apply a 10% discount to the total"]) == (
        "Answer: written by the model (step 2 needs judgement).")
    assert command_preview.answer_mode(["say that the total is 100", "apply no discount"]) == (
        "Answer: written by code, word for word (step 1 verbatim; step 2 dropped as a guard).")


# A call with no __result__: its verbatim answer phase is written by code (answer_by_code), and
# its last phase is a judgement step the model writes. The answer modes exist for these calls
# only: a call with __result__ writes its answer there, and its answer-only phases do not run.
ANSWER_MODE = '''
"""answer_mode: a verbatim answer phase, then a judgement last phase, and no __result__."""
from pydantic import BaseModel, Field

from .find_item import Signature as FindItem


class Signature:
    """Say how one named item is listed."""

    class Input(BaseModel):
        name: str = Field(description="the item's name")


class PlanGenerator:
    def find(self, input) -> FindItem.Output:
        return ["open_catalog", f"find_item query={input.name}"]

    def summary(self, find):
        return [f"say that {find.item_uids[0]} is listed"]

    def report(self, find):
        return ["say which item is heavier"]
'''


def _add_answer_mode(folder) -> None:
    (folder / "_commands" / "answer_mode.py").write_text(textwrap.dedent(ANSWER_MODE))


def test_an_answer_only_phase_shows_its_answer_mode_and_a_command_phase_none(workflow):
    from pathlib import Path

    _add_answer_mode(Path(workflow))
    previews = command_preview.render_previews(workflow)
    plan = previews["answer_mode.md"][previews["answer_mode.md"].index("## Plan"):]
    blocks = plan.split("### Phase ")[1:]
    # the find phase runs commands, so it has no answer mode; the two answer-only phases do
    assert "Answer: " not in blocks[0]
    assert [block.split("`")[1] for block in blocks if "Answer: " in block] == ["summary", "report"]
    # summary is before the last phase: its say-that line writes no answer (3.16 (a))
    assert ("Answer: none (a phase before the last writes no answer from 'say that' lines; "
            "its text reaches the answer only through __result__).") in blocks[1]
    assert "Answer: written by the model (step 1 needs judgement)." in blocks[2]
    # A flat skill and a fixed one have no phases, so no answer mode.
    assert "Answer: " not in previews["list_parts_of.md"]
    assert "Answer: " not in previews["close_out.md"]


def test_without_command_names_no_answer_mode_is_shown(workflow):
    from pathlib import Path

    from fastworkflow import workflow_agent

    _add_answer_mode(Path(workflow))
    commands = workflow_agent.load_planning_command_map(workflow)
    (name, description, steps), = [s for s in workflow_agent.load_planning_commands(workflow)
                                   if s[0] == "answer_mode"]
    bare = command_preview.render_preview(name, description, steps, commands[name], (),
                                        "_commands/answer_mode.py")
    named = command_preview.render_preview(name, description, steps, commands[name], (),
                                         "_commands/answer_mode.py",
                                         command_preview.phase_command_names(workflow))
    assert "Answer: " not in bare
    assert [line for line in named.splitlines() if line not in bare.splitlines()] == [
        "Answer: none (a phase before the last writes no answer from 'say that' lines; "
        "its text reaches the answer only through __result__).",
        "Answer: written by the model (step 1 needs judgement)."]


# ------------------------------------------------- which steps read every page (fix-ajir.14)

class _Paged(BaseModel):
    query: Optional[str] = None
    start: Optional[int] = None


def _paging_names(**extra):
    paged = {"find_item", "list_parts"}
    return {"commands": ["Catalog/open_catalog", "Catalog/find_item", "Item/open_item_by_uid",
                         "Item/list_parts", "Item/count_checks"],
            "core_commands": ["IntentDetection/go_up"],
            "parameters": lambda name: [_Paged] if name in paged else [None], **extra}


@pytest.mark.parametrize("steps, line", [
    (["find_item query=Alpha", "list_parts (full list)"],
     "Pages: step 1 reads page 1 only; step 2 reads up to 20 pages."),
    (["open_catalog", "list_parts (full list: next page with start=<n> until complete)",
      "find_item query=a", "find_item query=b (all of them)"],
     "Pages: step 2 reads up to 20 pages; steps 3 and 4 read page 1 only."),
    (["find_item query=Alpha start=10", "list_parts"], "Pages: step 2 reads page 1 only."),
])
def test_pages_line_names_the_steps_that_read_every_page_and_page_1(steps, line):
    assert command_preview.pages_line(steps, _paging_names()) == [line, ""]


def test_pages_line_is_empty_without_a_paged_step_or_parameters():
    assert command_preview.pages_line(["open_catalog", "find_item query=a start=0"],
                                    _paging_names()) == []
    assert command_preview.pages_line(["say that the item has no parts"], _paging_names()) == []
    names = _paging_names()
    del names["parameters"]
    assert command_preview.pages_line(["list_parts (full list)"], names) == []
    assert command_preview.pages_line(["list_parts (full list)"], None) == []


def test_a_phase_on_a_paged_command_shows_its_pages_line(workflow):
    from fastworkflow import workflow_agent

    commands = workflow_agent.load_planning_command_map(workflow)
    (name, description, steps), = [s for s in workflow_agent.load_planning_commands(workflow)
                                   if s[0] == "show_item"]
    bare = command_preview.render_preview(name, description, steps, commands[name], (),
                                        "_commands/show_item.py", _paging_names(parameters=None))
    named = command_preview.render_preview(name, description, steps, commands[name], (),
                                         "_commands/show_item.py", _paging_names())
    # Only the Pages lines are new: find reads page 1 of its search, item up to 20 pages of
    # its parts (the second plan shows them "as in the plan"); the others page nothing.
    added = [line for line in named.splitlines() if line not in bare.splitlines()]
    assert added == ["Pages: step 2 reads page 1 only.", "Pages: step 2 reads up to 20 pages."]
    kept = [line for line in named.splitlines() if line and not line.startswith("Pages: ")]
    assert kept == [line for line in bare.splitlines() if line]


def test_a_flat_planning_command_shows_its_pages_line(workflow):
    from fastworkflow import workflow_agent

    commands = workflow_agent.load_planning_command_map(workflow)
    (name, description, steps), = [s for s in workflow_agent.load_planning_commands(workflow)
                                   if s[0] == "list_parts_of"]
    text = command_preview.render_preview(name, description, steps, commands.get(name), (),
                                        "_commands/list_parts_of.py", _paging_names())
    # it runs as one phase: list_parts (no paging value, no note) reads page 1
    assert "Pages: step 2 reads page 1 only." in text.splitlines()


# ------------------------------------------------- a phase that may be skipped, in the template

SHELF_WALK = '''
"""shelf_walk: one shelf, its named box when the request names one, and its label."""
from pydantic import BaseModel, Field

from .find_shelf import Signature as FindShelf
from .open_box_by_uid import Signature as OpenBox


class Signature:
    """Walk one shelf, its named box when there is one, and its label. One job per shelf."""

    class Input(BaseModel):
        shelf: str = Field(description="the shelf's name")
        box: str = Field(default="", description="a box to open; empty for none")


class PlanGenerator:
    def __phases__(self, input):
        box = [self.box] if input.box else []
        return [self.find, *box, self.label, self.only_after_box]

    def find(self, input) -> FindShelf.Output:
        return [f"find_shelf query={input.shelf}"]

    class BoxFacts(BaseModel):
        opened: OpenBox.Output

        @property
        def uid(self) -> str:
            return self.opened.box_uid

    def box(self, input, find) -> BoxFacts:
        return [f"open_box_by_uid with the uid of the {input.box} match"]

    def label(self, find, box=None):
        return (["go_up (back to the shelf)"] if box is not None else []) + ["show_label"]

    def only_after_box(self, box=None):
        if box is None:
            return []
        return [f"list_items on {box.uid}"]
'''


def test_a_phase_reading_a_skipped_phase_reads_as_if_it_ran_in_the_template(tmp_path):
    """The template is built with placeholder facts, never None (the former "template reads
    as not run" check, fix-ajir.13, is gone): a phase reading a phase the call has shows its
    wording for when that phase ran; for a call without that phase its default holds."""
    from fastworkflow import workflow_agent

    folder = tmp_path / "shelf_wf"
    (folder / "_commands").mkdir(parents=True)
    _write_step_stubs(str(folder))
    (folder / "_commands" / "shelf_walk.py").write_text(textwrap.dedent(SHELF_WALK))
    command = workflow_agent.load_planning_command_map(str(folder))["shelf_walk"]
    assert list(command.template) == [
        "find_shelf query=<shelf>",
        "open_box_by_uid with the uid of the <box> match (only when box is given)",
        "go_up (back to the shelf) (only when box is given)",
        "show_label",
        "list_items on <box.uid> (only when box is given)",
    ]


# ---------------------------------------------------------------- fix-3rsq.5.7: calls, depth, cycles

CALLS_OUTER = '''
"""outer: close out, then say what the named item is."""
from pydantic import BaseModel, Field


class Signature:
    """Close out a visit, then say what one named item is."""

    class Input(BaseModel):
        name: str = Field(description="the item's name", json_schema_extra={"in_request": True})


class PlanGenerator:
    def steps(self, input):
        return ["close_out", f"say what {input.name} is"]
'''

PING = '''
class Signature:
    """Ping, which pongs back."""


class PlanGenerator:
    def steps(self):
        return ["pong", "say ping done"]
'''

PONG = '''
class Signature:
    """Pong, which pings back."""


class PlanGenerator:
    def steps(self):
        return ["ping", "say pong done"]
'''

AGAIN = '''
class Signature:
    """Again, which calls itself."""


class PlanGenerator:
    def steps(self):
        return ["again", "say again done"]
'''


def _planning_commands_workflow(root, name: str, planning_commands: dict[str, str]) -> str:
    folder = root / name
    (folder / "_commands").mkdir(parents=True)
    _write_step_stubs(str(folder))
    for planning_command, text in planning_commands.items():
        (folder / "_commands" / f"{planning_command}.py").write_text(textwrap.dedent(text))
    return str(folder)


def test_calls_and_depth_lines_name_each_callee_and_the_chain_depth(tmp_path):
    """A planning command's ``Calls:`` names the planning commands its steps call (``none``
    for a fixed-steps command that calls none); ``Depth:`` is the longest chain from it,
    itself counted, a lower bound since open steps name no command. No ``Kind:`` line."""
    folder = _planning_commands_workflow(tmp_path, "calls_wf", {"outer": CALLS_OUTER, "close_out": CLOSE_OUT})
    previews = command_preview.render_previews(folder)
    assert ("Calls: `close_out`.\nDepth: 2 (lower bound; open steps name no command)\n"
            in previews["outer.md"])
    assert ("Calls: none.\nDepth: 1 (lower bound; open steps name no command)\n"
            in previews["close_out.md"])
    assert not any("Kind:" in text for text in previews.values())


def test_a_call_cycle_is_a_load_error_for_each_command_on_it_and_preview_check_fails(
        tmp_path, monkeypatch, capsys):
    """ping and pong call each other, again calls itself: each is an error naming its path, a
    command off the cycle is not, and ``command preview --check`` fails on the cycle."""
    folder = _planning_commands_workflow(tmp_path, "loop_wf", {
        "ping": PING, "pong": PONG, "again": AGAIN, "close_out": CLOSE_OUT,
    })
    res = planning_commands.resolve_field_sources(folder)
    assert [p.message for p in res.errors if p.command == "ping"] == [
        "a planning command cannot call itself, even through others: ping -> pong -> ping"]
    assert [p.message for p in res.errors if p.command == "pong"] == [
        "a planning command cannot call itself, even through others: pong -> ping -> pong"]
    assert [p.message for p in res.errors if p.command == "again"] == [
        "a planning command cannot call itself: again -> again"]
    assert not [p for p in res.errors if p.command == "close_out"]

    out = str(tmp_path / "loop_previews")
    command_preview.write_previews(folder, out)
    assert _run_cli(monkeypatch, "command", "preview", folder, "--out", out, "--check") == 1
    stdout = capsys.readouterr().out
    assert "a planning command cannot call itself, even through others: ping -> pong -> ping" in stdout
    assert "a planning command cannot call itself: again -> again" in stdout


def test_the_command_group_replaces_planning_command_with_no_alias(tmp_path, monkeypatch, capsys):
    """``command preview`` writes the previews; ``skill preview`` is no longer a command."""
    folder = _planning_commands_workflow(tmp_path, "rename_wf", {"close_out": CLOSE_OUT})
    out = str(tmp_path / "rename_previews")
    assert _run_cli(monkeypatch, "command", "preview", folder, "--out", out) == 0
    assert os.listdir(out) == ["close_out.md"]
    capsys.readouterr()
    assert _run_cli(monkeypatch, "skill", "preview", folder, "--out", out) == 2
    assert "invalid choice: 'skill'" in capsys.readouterr().err


#: A union-output fixture (design 3.12): two commands of one context, ``find_book`` and
#: ``find_member``, and a planning command whose find phase returns either one's Output.
#: ``OUTPUT`` is that phase's return annotation, ``OVER`` the field its fan-out reads.
UNION_CONTEXT = '''
class Context:
    @classmethod
    def get_parent(cls, command_context_object) -> None:
        return None

    @classmethod
    def get_displayname(cls, command_context_object) -> str:
        return "Lib"
'''

UNION_STUB = '''
import fastworkflow
from pydantic import BaseModel, Field


class Signature:
    """Find one kind of record. A stub the union fixture's steps name."""

    class Input(BaseModel):
        query: str = Field(default="", description="a stub parameter")

    class Output(BaseModel):
        OUTPUT_FIELDS


class ResponseGenerator:
    def __call__(self, workflow: fastworkflow.Workflow, command: str, command_parameters):
        return Signature.Output()
'''

UNION_FIND = '''
"""find_either: whether a book or a member goes by a name, and which."""
from typing import Literal, Optional, Union

from pydantic import BaseModel, Field

from fastworkflow.planning import phase

from .find_book import Signature as FindBook
from .find_member import Signature as FindMember


class Signature:
    """Say whether a book or a member goes by a name, and name each match."""

    plain_utterances = ["do you have a book called Emma", "is there a member named Lee"]

    class Input(BaseModel):
        name: str = Field(description="the name to look for")
        kind: Literal["book", "member"] = Field(default="book", description="a book or a member")


class PlanGenerator:
    def __phases__(self, input):
        return [self.find, self.each]

    def find(self, input) -> OUTPUT:
        return [f"find_{input.kind} query={input.name}"]

    @phase(over="OVER")
    def each(self, title):
        return [f"find_book query={title}"]

    def __result__(self, input, find):
        return f"{input.name} matches {find.total}."
'''

BOOK_FIELDS = ("titles: list[str] = Field(default_factory=list, description=\"a stub output\")\n"
               "        total: int = Field(default=0, description=\"a stub output\")")
MEMBER_FIELDS = ("names: list[str] = Field(default_factory=list, description=\"a stub output\")\n"
                 "        total: int = Field(default=0, description=\"a stub output\")")


def _union_workflow(root, over: str = "find.titles", output: str = "FindBook.Output | FindMember.Output") -> str:
    # a folder name of its own: the loader imports a workflow's modules by their names, so two
    # fixtures sharing one name in one process would share (and mix up) the classes of the first
    folder = root / f"union_{uuid.uuid4().hex[:8]}"
    commands = folder / "_commands" / "Lib"
    commands.mkdir(parents=True)
    (commands / "_Lib.py").write_text(UNION_CONTEXT)
    (commands / "find_book.py").write_text(UNION_STUB.replace("OUTPUT_FIELDS", BOOK_FIELDS))
    (commands / "find_member.py").write_text(UNION_STUB.replace("OUTPUT_FIELDS", MEMBER_FIELDS))
    text = UNION_FIND.replace("OVER", over).replace("OUTPUT", output)
    (commands / "find_either.py").write_text(text)
    return str(folder)


@pytest.fixture
def union_workflow(tmp_path):
    return _union_workflow(tmp_path)


def test_a_union_output_renders_each_member_as_a_labelled_variant(union_workflow):
    text = command_preview.render_previews(union_workflow)["Lib/find_either.md"]
    # the phase's output names both commands' Outputs, and the table cell escapes the separator
    assert "| `find` | `Lib/find_book`'s `Output` \\| `Lib/find_member`'s `Output` | always | no |" in text
    assert "### `find` -> `Lib/find_book`'s `Output` | `Lib/find_member`'s `Output`" in text
    # each member, labelled by its command, with its own fields
    assert ("`Lib/find_book`'s `Output`:\n\n| fact | type | from |\n|---|---|---|\n"
            "| `titles` | `list[str]` | `Lib/find_book`'s `Output` |\n"
            "| `total` | `int` | `Lib/find_book`'s `Output` |") in text
    assert ("`Lib/find_member`'s `Output`:\n\n| fact | type | from |\n|---|---|---|\n"
            "| `names` | `list[str]` | `Lib/find_member`'s `Output` |\n"
            "| `total` | `int` | `Lib/find_member`'s `Output` |") in text


def test_a_from_edge_to_a_field_of_one_member_resolves_and_says_which_member_has_it(union_workflow):
    text = command_preview.render_previews(union_workflow)["Lib/find_either.md"]
    # kind = book runs find_book, whose Output has titles: the fan-out renders its items
    assert "| `each` | none | always | per item of `find.titles` as `title` |" in text
    book = _plan(text, "when kind = book")
    assert "For the first item, `title` = `<find.titles[0]>`:\n\n1. `find_book query=<find.titles[0]>`" in book
    # kind = member runs find_member, which has no titles: the preview says so, and says who has it
    member = _plan(text, "when kind = member")
    assert "_Items not rendered: phase 'each': over='find.titles': Output has no field 'titles'. " \
           "`titles` is a field of `Lib/find_book`'s `Output` only._" in member
    assert "find_book query=<find.titles[0]>" not in member


def test_a_field_in_neither_member_is_reported_as_unresolved(tmp_path):
    folder = _union_workflow(tmp_path, over="find.authors")
    with pytest.raises(command_preview.PreviewError) as error:
        command_preview.render_previews(folder)
    assert "find_book.Output | find_member.Output has no field 'authors'" in str(error.value)
    assert "find_either" in str(error.value)


@pytest.mark.parametrize("output", [
    "FindBook.Output | None",
    "Optional[FindBook.Output]",
    "Union[FindBook.Output, FindMember.Output]",
    "Union[FindBook.Output, FindMember.Output, None]",
    "Optional[Union[FindBook.Output, FindMember.Output]]",
])
def test_optional_and_typing_unions_render_their_members(tmp_path, output):
    text = command_preview.render_previews(_union_workflow(tmp_path, output=output))["Lib/find_either.md"]
    assert "### `find` -> `Lib/find_book`'s `Output`" in text
    assert "`Lib/find_book`'s `Output`:\n\n| fact | type | from |" in text
    assert "per item of `find.titles` as `title`" in text
    # None is not a member: no variant for it
    assert "NoneType" not in text and "None`'s" not in text


def test_output_members_and_placeholder_member_read_unions_and_optionals():
    from typing import Optional, Union

    from fastworkflow import planning_phases

    class Book(BaseModel):
        titles: list[str] = []

    class Member(BaseModel):
        names: list[str] = []

    assert planning_phases.output_members(Book) == (Book,)
    assert planning_phases.output_members(Book | Member) == (Book, Member)
    assert planning_phases.output_members(Union[Book, Member]) == (Book, Member)
    assert planning_phases.output_members(Optional[Book]) == (Book,)
    assert planning_phases.output_members(Book | None) == (Book,)
    assert planning_phases.output_members(Union[Book, Member, None]) == (Book, Member)
    for not_a_model in (None, int, Book | int, list[str]):
        assert planning_phases.output_members(not_a_model) == ()
    assert planning_phases.placeholder_member(Book | Member) is Book
    assert planning_phases.placeholder_member(Book | Member, "names") is Member
    assert planning_phases.placeholder_member(Book | Member, "nowhere") is Book
    assert planning_phases.placeholder_member(Book) is Book


LIBRARY = os.path.join(os.path.dirname(os.path.abspath(command_preview.__file__)), "examples", "library")


def test_the_library_find_record_preview_renders_both_variants():
    text = command_preview.render_previews(LIBRARY)["Library/find_record.md"]
    assert "`Library/find_book`'s `Output`:" in text
    assert "`Library/find_member`'s `Output`:" in text
    assert "| `names` | `list[str]` |" in text and "| `titles` | `list[str]` |" in text
    assert "1. `find_member query=<name>`" in text


def test_the_library_previews_write_and_check_clean_in_a_tmp_dir(tmp_path, monkeypatch):
    out = tmp_path / "library_previews"
    assert _run_cli(monkeypatch, "command", "preview", LIBRARY, "--out", str(out)) == 0
    assert (out / "Library" / "find_record.md").is_file()
    assert _run_cli(monkeypatch, "command", "preview", LIBRARY, "--check", "--out", str(out)) == 0


# ---------------------------------------------------------------------------
# over= may name a property (of the facts model, or of a typed field's model); the preview shows
# the items on the placeholder facts, or says they are decided at run time (Phase 15 follow-up).

PROP_FIND = '''
"""find_props: find a book, then go through the titles the facts compute."""
from typing import Optional

from pydantic import BaseModel, Field

from fastworkflow.planning import phase

from .find_book import Signature as FindBook


class Listing(BaseModel):
    """The phase's facts: a typed command Output, and properties computed from it."""

    found: Optional[FindBook.Output] = None

    @property
    def shown(self) -> list[str]:
        return sorted(self.found.titles) if self.found else []

    @property
    def ranked(self) -> str:
        return {"first": "1"}[self.found.titles[0]]


class Signature:
    """Find a book and go through its titles."""

    plain_utterances = ["go through the titles of Dune"]

    class Input(BaseModel):
        name: str = Field(description="the name to look for")


class PlanGenerator:
    def __phases__(self, input):
        return [self.find, self.each]

    def find(self, input) -> Listing:
        return [f"find_book query={input.name}"]

    @phase(over="OVER")
    def each(self, title):
        return [f"find_book query={title}"]
'''


def _props_workflow(root, over: str = "find.shown", output: str = "Listing") -> str:
    # The folder's basename is fixed ("props_workflow") and its parent is the caller's: two copies at
    # two paths share a basename, so only the path tells their namespaces apart.
    folder = root / "props_workflow"
    commands = folder / "_commands" / "Lib"
    commands.mkdir(parents=True)
    (commands / "_Lib.py").write_text(UNION_CONTEXT)
    (commands / "find_book.py").write_text(UNION_STUB.replace("OUTPUT_FIELDS", BOOK_FIELDS))
    text = PROP_FIND.replace("OVER", over).replace("-> Listing", f"-> {output}")
    (commands / "find_props.py").write_text(text)
    return str(folder)


def test_a_property_of_the_facts_model_fans_out_on_the_placeholder_facts(tmp_path):
    text = command_preview.render_previews(_props_workflow(tmp_path / "one"))["Lib/find_props.md"]
    # the property reads the placeholder titles: one item, its placeholder, and its steps read with it
    assert "| `each` | none | always | per item of `find.shown` as `title` |" in text
    assert "For the first item, `title` = `<find.found.titles[0]>`:\n\n1. `find_book query=<find.found.titles[0]>`" \
        in text
    assert "Items decided at run time" not in text


def test_a_fan_out_over_a_property_shows_the_later_item_block_as_a_field_does(tmp_path):
    # the property reads a list one typed command model down (found.titles); the placeholder list
    # gives the fan-out its second item, so the later-items block renders as it does over a field
    via_property = command_preview.render_previews(_props_workflow(tmp_path / "prop"))["Lib/find_props.md"]
    via_field = command_preview.render_previews(
        _props_workflow(tmp_path / "field", over="find.found.titles"))["Lib/find_props.md"]
    for text, path in ((via_property, "find.found.titles"), (via_field, "find.found.titles")):
        assert ("For each later item (shown for the second), `title` = `<find.found.titles[1]>`:\n\n"
                "1. `find_book query=<find.found.titles[1]>`") in text, path
    assert ("For the first item, `title` = `<find.found.titles[0]>`:\n\n"
            "1. `find_book query=<find.found.titles[0]>`") in via_property


def test_a_property_that_raises_on_the_placeholder_facts_is_decided_at_run_time(tmp_path):
    text = command_preview.render_previews(
        _props_workflow(tmp_path / "one", over="find.ranked"))["Lib/find_props.md"]
    assert ("_Items decided at run time (KeyError while the template was built, with every fact unknown)._"
            in text)
    assert "find_book query=<find.found.titles[0]>" not in text


def test_an_over_naming_neither_a_field_nor_a_property_is_still_unresolved(tmp_path):
    folder = _props_workflow(tmp_path / "one", over="find.nosuch")
    with pytest.raises(command_preview.PreviewError) as error:
        command_preview.render_previews(folder)
    assert "Listing has no field 'nosuch'" in str(error.value)


def test_two_copies_of_one_workflow_at_different_paths_preview_identically(tmp_path):
    # the Optional facts field (found) renders its type by the class's plain name, so no path or
    # namespace digest reaches the preview: the two copies' previews are the same text
    first = command_preview.render_previews(_props_workflow(tmp_path / "first"))
    second = command_preview.render_previews(_props_workflow(tmp_path / "second"))
    assert first == second
    for text in first.values():
        assert not re.search(r"__[0-9a-f]{8}", text)
        assert "talk_to" not in text and "props_workflow." not in text


def test_an_optional_facts_field_renders_its_type_by_plain_class_name(tmp_path):
    text = command_preview.render_previews(_props_workflow(tmp_path / "one"))["Lib/find_props.md"]
    # the Optional field: its command Output by plain name, as a non-Optional typed field renders
    assert "Optional" not in text and "typing." not in text
    assert "| `found` | `Output | None` |" in text.replace("\\|", "|")

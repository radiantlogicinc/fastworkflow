"""The skill tutorials (docs/tutorials/skills) are checked against the code they teach.

Every tutorial page marks what it shows with an HTML comment right above a fenced block:

- ``<!-- file: tests/skill_tutorial_workflow/_skills/x.py -->``: the block is that file;
- ``<!-- template: x -->``: the block is skill x's template, the steps the planner sees;
- ``<!-- preview: x -->``: the block is part of x's ``fastworkflow skill preview``;
- ``<!-- dry-run: x {"arg": "value"} -->``: the block is ``dry_run("x", {...})``;
- ``<!-- refusal: x {...} -->``: the block is the refusal a call of x with those arguments gets;
- ``<!-- doctest -->``: the block runs as a doctest;
- ``<!-- broken: x -->`` then ``<!-- load-error -->``, ``<!-- check-error -->`` or
  ``<!-- template-line -->``: the first block, saved as ``_skills/x.py`` in a copy of the
  tutorial workflow, gives that load error, ``skill preview --check`` problem, or template line.

``dry_run`` runs a skill call offline, with no model: each phase's steps as the runtime
writes them, the leading steps code runs (``skill_phases.plan_code_steps`` and ``pick_step``,
the rules ``CodeSteps`` uses) run against the tutorial workflow's commands, each phase's
facts filled from those runs (``skill_phases.fill_output``), and the answer code writes.
Where code stops (an open step) the agent would take over; the dry run says so and stops
that phase. Try it: ``python -m tests.test_skill_tutorials book_by_title '{"title": "Dune"}'``.
"""
from __future__ import annotations

import doctest
import json
import logging
import os
import re
import shutil
import sys

import pytest

from fastworkflow import ModuleType, skill_commands, skill_preview, workflow_agent
from fastworkflow import skill_phases as sp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKFLOW = os.path.join(ROOT, "tests", "skill_tutorial_workflow")
PAGES = os.path.join(ROOT, "docs", "tutorials", "skills")
PREVIEWS = os.path.join(PAGES, "previews")
CORE = ["go_up", "reset_context"]

_BLOCK = re.compile(r"<!-- ([\w-]+)(?:: ([^\n]*?))? -->\n```[\w-]*\n(.*?)\n```", re.S)


def _routing(folder: str = WORKFLOW):
    from fastworkflow.command_routing import RoutingRegistry

    return RoutingRegistry.get_definition(folder)


def _run(routing, command: str):
    """Run *command* (``name k=v ...``) with the tutorial workflow's own ResponseGenerator."""
    from fastworkflow._workflows.command_metadata_extraction.parameter_extraction import (
        ParameterExtraction)

    name = command.split()[0]
    cls = routing.get_command_class(name, ModuleType.COMMAND_PARAMETERS_CLASS)
    generator = routing.get_command_class(name, ModuleType.RESPONSE_GENERATION_INFERENCE)()
    if cls is None:
        return {}, generator(None, name).command_response.output
    bound = (cls() if command == name else
             ParameterExtraction._extract_parameters_from_key_values(command, cls, name))
    return bound.model_dump(), generator(None, name, bound).command_response.output


def dry_run(skill: str, args: dict, folder: str = WORKFLOW) -> str:
    """What a call of *skill* with *args* runs, phase by phase, offline (see the module doc)."""
    routing = _routing(folder)
    parameters = sp.parameter_classes(routing)
    commands = sorted(n for n in routing.routing_definition_map if n != "wildcard" and "/" not in n)
    command = workflow_agent.load_skill_commands(folder)[skill]
    args = {p.name: p.default for p in command.params if not p.required} | args
    phases = skill_commands.call_phases(command, args)
    lines: list[str] = []
    if len(phases) == 1 and phases[0].name == skill:
        lines.append("Runs flat: one plan, one executor run.")
    outputs: dict = {}

    def run_steps(steps: list[str], last: bool):
        text = skill_commands.number(steps)
        lines.extend(f"  {line}" for line in text.splitlines())
        runs, lookup, open_step = [], None, False
        for step, run, _ in sp.plan_code_steps(text, commands, CORE, parameters):
            if run == "":
                continue
            if run is None and lookup is not None:
                run, _ = sp.pick_step(step, lookup[0], lookup[1], parameters(step.command))
            if run is None:
                lines.append(f"  step {step.number} is open: the agent runs it and the rest "
                             "(the dry run stops here)")
                open_step = True
                break
            given, output = _run(routing, run)
            lines.append(f"  code runs: {run}")
            runs.append({"command": run.split()[0], "input": given, "output": output.model_dump(),
                         "success": True, "context_clause": ""})
            lookup = (step, output.model_dump())
        if sp.is_answer_only(text, commands, CORE):
            if sp.answer_by_code(text):
                lines.append("  answer written by code:")
                lines.extend(f"    {line}" for line in sp.answer_from_steps(text).splitlines())
            else:
                lines.append("  answer written by the model (one call, no tools)")
        elif last and not open_step:
            lines.append("  the agent writes the answer from these runs")
        return text, runs, open_step

    for index, phase in enumerate(phases):
        last = index == len(phases) - 1
        if phase.fans_out:
            items, _ = sp.fan_out_items(phase, outputs)
            filled = []
            if not items:
                lines.append(f"Phase {phase.name}: no items, so it runs nothing")
            for item in items:
                lines.append(f"Phase {phase.name} (item {item})")
                text, runs, stop = run_steps(sp.phase_steps(phase, outputs, item), last)
                if stop:
                    return "\n".join(lines)
                if phase.output is not None:
                    filled.append(sp.fill_output(phase.output, runs, record=None,
                                                 plan_text=text, commands=commands))
                    lines.append(f"  facts: {filled[-1]!r}")
            outputs[phase.name] = filled
            continue
        if not sp.runs_now(phase, outputs):
            lines.append(f"Phase {phase.name}: skipped (its method returned [])")
            continue
        lines.append(f"Phase {phase.name}")
        text, runs, stop = run_steps(sp.phase_steps(phase, outputs), last)
        if stop:
            break
        if phase.output is not None:
            outputs[phase.name] = sp.fill_output(phase.output, runs, record=None,
                                                 plan_text=text, commands=commands)
            lines.append(f"  facts: {outputs[phase.name]!r}")
    return "\n".join(lines)


def refusal(skill: str, args: dict) -> str:
    """The message a call of *skill* with *args* is refused with (empty: not refused)."""
    command = workflow_agent.load_skill_commands(WORKFLOW)[skill]
    try:
        skill_commands.validate_call(command, args, "")
    except skill_commands.SkillInputError as error:
        return str(error)
    return ""


def _template(skill: str, folder: str = WORKFLOW) -> str:
    return "\n".join(dict((n, s) for n, _, s in workflow_agent.load_skills(folder))[skill])


def _copy_workflow(tmp_path, name: str) -> str:
    """A copy of the tutorial workflow without its skills; a fresh package name each time."""
    folder = tmp_path / f"tutorial_{name}"
    shutil.copytree(WORKFLOW, folder, ignore=shutil.ignore_patterns(
        "_skills", "__pycache__", "___*"))
    (folder / "_skills").mkdir()
    return str(folder)


def load_problem(folder: str, name: str) -> str:
    """Why ``_skills/<name>.py`` of *folder* does not load (or fails ``--check``), with the
    folder's path dropped from the message."""
    path = os.path.join(folder, "_skills", f"{name}.py")
    try:
        loaded = workflow_agent._read_skill_file(name, path, folder)
    except Exception as error:  # noqa: BLE001 - the message is what the page shows
        text = f"{type(error).__name__}: {error}"
    else:
        problems = []
        if loaded.command is not None:
            problems = (workflow_agent.step_command_problems(loaded.command, folder)
                        or workflow_agent.phase_source_problems(loaded.command, folder))
        text = "\n".join(problems)
    return text.replace(folder + os.sep, "")


def _blocks(page: str) -> list[tuple[str, str, str]]:
    with open(os.path.join(PAGES, page), encoding="utf-8") as fh:
        return [(kind, arg or "", block) for kind, arg, block in _BLOCK.findall(fh.read())]


def _pages() -> list[str]:
    return sorted(f for f in os.listdir(PAGES) if f.endswith(".md"))


@pytest.fixture(scope="module", autouse=True)
def _quiet():
    logging.disable(logging.ERROR)
    yield
    logging.disable(logging.NOTSET)


@pytest.fixture(scope="module")
def previews():
    return skill_preview.render_previews(WORKFLOW)


def test_every_tutorial_skill_loads_and_its_committed_preview_is_current():
    names = {n for n, _, _ in workflow_agent.load_skills(WORKFLOW)}
    files = {f[:-3] for f in os.listdir(os.path.join(WORKFLOW, "_skills"))
             if f.endswith(".py") and not f.startswith("_")}
    assert names == files
    assert skill_preview.check_previews(WORKFLOW, PREVIEWS) == []


def test_every_page_shows_real_blocks(previews):
    """Each marked block of each page is what the code gives (see the module doc)."""
    shown = 0
    for page in _pages():
        for kind, arg, block in _blocks(page):
            where = f"{page}: <!-- {kind}: {arg} -->"
            if kind == "file":
                with open(os.path.join(ROOT, arg), encoding="utf-8") as fh:
                    assert block == fh.read().rstrip("\n"), where
            elif kind == "template":
                assert block == _template(arg), where
            elif kind == "preview":
                assert block in previews[f"{arg}.md"], where
            elif kind == "dry-run":
                skill, _, given = arg.partition(" ")
                assert block == dry_run(skill, json.loads(given or "{}")), where
            elif kind == "refusal":
                skill, _, given = arg.partition(" ")
                assert block == refusal(skill, json.loads(given)), where
            elif kind == "doctest":
                from fastworkflow import skill as skill_api

                test = doctest.DocTestParser().get_doctest(
                    block, dict(vars(skill_api)), where, page, 0)
                result = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS).run(test)
                assert result.failed == 0, where
            elif kind in ("broken", "load-error", "check-error", "template-line"):
                continue   # test_troubleshooting_errors_are_real
            else:
                raise AssertionError(f"{where}: unknown marker")
            shown += 1
    assert shown >= 30


def test_troubleshooting_errors_are_real(tmp_path):
    checked = 0
    for page in _pages():
        blocks = _blocks(page)
        for (kind, name, text), (expect, _, message) in zip(blocks, blocks[1:]):
            if kind != "broken":
                continue
            folder = _copy_workflow(tmp_path, f"{checked}_{name}")
            with open(os.path.join(folder, "_skills", f"{name}.py"), "w") as fh:
                fh.write(text + "\n")
            where = f"{page}: broken {name}"
            if expect == "load-error":
                assert load_problem(folder, name) == message, where
            elif expect == "check-error":
                assert load_problem(folder, name) == message, where
                problems = skill_preview.check_previews(folder, str(tmp_path / f"out{checked}"))
                assert any(message in p for p in problems), where
            elif expect == "template-line":
                assert any(line.startswith(message) for line in
                           _template(name, folder).splitlines()), where
            else:
                raise AssertionError(f"{where}: no expected result after it")
            checked += 1
    assert checked >= 5


# ----------------------------------------------------------- run-time scenarios, by skill

def test_a_skipped_phase_reads_as_none_and_the_answer_says_so():
    text = dry_run("book_details", {"title": "Ulysses"})
    assert "Phase book: skipped (its method returned [])" in text
    assert text.endswith("answer written by code:\n    No book matches Ulysses.")


def test_pick_opens_the_exact_title_and_else_leaves_the_row_to_code_or_the_agent():
    assert "code runs: show_book book_uid=b1" in dry_run("book_by_title", {"title": "Dune"})
    assert "code runs: show_book book_uid=b2" in dry_run("book_by_title",
                                                       {"title": "dune messiah"})
    # no title is exactly "Dune Messia": the step is written in pick words, for the agent
    near = dry_run("book_by_title", {"title": "Dune Messia"})
    assert "1. show_book with the uid of the Dune Messia match" in near
    assert near.endswith("step 1 is open: the agent runs it and the rest (the dry run stops here)")


def test_call_shapes_the_plan_and_a_brief_call_runs_flat():
    assert dry_run("book_report", {"title": "Dune", "brief": True}).startswith(
        "Runs flat: one plan, one executor run.")
    without = dry_run("book_report", {"title": "Dune"})
    assert "Phase loans" not in without and "on loan" not in without
    assert "Phase loans" in dry_run("book_report", {"title": "Dune", "loans": True})


def test_a_fan_out_runs_once_per_found_book():
    text = dry_run("author_books", {"author": "Jane Austen"})
    assert "Phase book (item b3)" in text and "Phase book (item b4)" in text


def test_refusal_and_a_plain_call():
    assert "names no one member" in refusal("member_loans", {"member": "everyone"})
    assert refusal("member_loans", {"member": "Ann Lee"}) == ""


def test_the_composed_skill_places_the_base_phases_and_extends_the_base_answer():
    text = dry_run("book_loans", {"title": "Emma"})
    assert [line for line in text.splitlines() if line.startswith("Phase")] == [
        "Phase find", "Phase book", "Phase loans", "Phase answer"]
    assert text.endswith("answer written by code:\n"
                         "    Emma is by Jane Austen; the library owns 2 copies.\n"
                         "    It is on loan to 1 borrower (Raj Patel).")


def test_a_skill_with_no_inputs_runs_its_two_phases():
    command = workflow_agent.load_skill_commands(WORKFLOW)["overdue_summary"]
    assert command.params == ()
    text = dry_run("overdue_summary", {})
    assert [line for line in text.splitlines() if line.startswith("Phase")] == [
        "Phase branches", "Phase overdue"]
    assert "code runs: list_branches" in text and "code runs: count_overdue" in text


if __name__ == "__main__":
    logging.disable(logging.ERROR)
    print(dry_run(sys.argv[1], json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}))

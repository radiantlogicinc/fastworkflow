"""The skill tutorials (docs/tutorials/planning_commands) are checked against the code they teach.

Every tutorial page marks what it shows with an HTML comment right above a fenced block:

- ``<!-- file: fastworkflow/examples/library/_commands/x.py -->``: the block is that file
  (a path from the repository root; the pages name the library's planning commands);
- ``<!-- template: x -->``: the block is skill x's template, the steps the planner sees;
- ``<!-- preview: x -->``: the block is part of x's ``fastworkflow command preview``;
- ``<!-- dry-run: x {"arg": "value"} -->``: the block is ``dry_run("x", {...})``;
- ``<!-- refusal: x {...} -->``: the block is the refusal a call of x with those arguments gets;
- ``<!-- doctest -->``: the block runs as a doctest;
- ``<!-- broken: x -->`` then ``<!-- load-error -->``, ``<!-- check-error -->`` or
  ``<!-- template-line -->``: the first block, saved as ``_commands/x.py`` (a planning command,
  with a top-level ``PlanGenerator``) in a copy of the tutorial workflow, gives that load error,
  ``command preview --check`` problem, or template line.

``dry_run`` is ``fastworkflow.planning.run_code`` (``fastworkflow command run-code``) on the tutorial
workflow: each phase's steps as the runtime writes them, the steps code runs against the tutorial workflow's
commands on a real session (``planning_phases.plan_code_steps``, ``pick_listing`` and ``pick_step``), each
phase's facts filled from those runs (``planning_phases.fill_output``), and the answer code writes. Where code
stops (an open step) the agent would take over; the run says so and stops. Try it:
``python -m tests.test_planning_command_tutorials book_by_title '{"title": "Dune"}'``.
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

from fastworkflow import ModuleType, planning_commands, command_preview, workflow_agent
from fastworkflow import planning_phases as sp
from fastworkflow.observation_offloading.finish_check import plain_command_name

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKFLOW = os.path.join(ROOT, "tests", "planning_tutorial_workflow")
LIBRARY = os.path.join(ROOT, "fastworkflow", "examples", "library")
PAGES = os.path.join(ROOT, "docs", "tutorials", "planning_commands")
PREVIEWS = os.path.join(PAGES, "previews")
CORE = ["go_up", "reset_context"]

_BLOCK = re.compile(r"<!-- ([\w-]+)(?:: ([^\n]*?))? -->\n```[\w-]*\n(.*?)\n```", re.S)


def _routing(folder: str = WORKFLOW):
    from fastworkflow.command_routing import RoutingRegistry

    return RoutingRegistry.get_definition(folder)


def dry_run(planning_command: str, args: dict, folder: str = WORKFLOW) -> str:
    """What a call of *planning_command* with *args* runs, phase by phase, with no model: the text of
    ``fastworkflow.planning.run_code`` (its steps are the tutorial workflow's real commands, run on a real session)."""
    from fastworkflow.planning import run_code

    return run_code(folder, planning_command, args).text


def refusal(planning_command: str, args: dict) -> str:
    """The message a call of *skill* with *args* is refused with (empty: not refused)."""
    command = workflow_agent.load_planning_command_map(WORKFLOW)[planning_command]
    try:
        planning_commands._validate_arguments(command, args, "")
    except planning_commands.PlanningCommandInputError as error:
        return str(error)
    return ""


def _template(planning_command: str, folder: str = WORKFLOW) -> str:
    return "\n".join(dict((n, s) for n, _, s in workflow_agent.load_planning_commands(folder))[planning_command])


def _copy_workflow(tmp_path, name: str) -> str:
    """A copy of the tutorial workflow without its planning commands; a fresh package name each
    time. The execution commands stay, as they do in the workflow itself."""
    folder = tmp_path / f"tutorial_{name}"
    shutil.copytree(WORKFLOW, folder, ignore=shutil.ignore_patterns("__pycache__", "___*"))
    for _, path in workflow_agent._planning_command_files(os.path.join(str(folder), "_commands")):
        os.remove(path)
    return str(folder)


def load_problem(folder: str, name: str) -> str:
    """Why ``_commands/<name>.py`` of *folder* does not load (or fails ``--check``), with the
    folder's path dropped from the message."""
    path = os.path.join(folder, "_commands", f"{name}.py")
    try:
        loaded = workflow_agent._read_planning_command_file(name, path, folder)
    except Exception as error:  # noqa: BLE001 - the message is what the page shows
        text = f"{type(error).__name__}: {error}"
    else:
        problems = []
        if loaded.command is not None:
            problems = list(dict.fromkeys(
                workflow_agent.step_command_problems(loaded.command, folder)
                or planning_commands.resolve_field_sources(folder).errors_for(name)))
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
    return command_preview.render_previews(WORKFLOW)


def test_every_tutorial_planning_command_loads_and_its_committed_preview_is_current():
    names = {n for n, _, _ in workflow_agent.load_planning_commands(WORKFLOW)}
    files = {n for n, _ in workflow_agent._planning_command_files(os.path.join(WORKFLOW, "_commands"))}
    assert names == files
    assert command_preview.check_previews(WORKFLOW, PREVIEWS) == []


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
                planning_command, _, given = arg.partition(" ")
                assert block == dry_run(planning_command, json.loads(given or "{}")), where
            elif kind == "refusal":
                planning_command, _, given = arg.partition(" ")
                assert block == refusal(planning_command, json.loads(given)), where
            elif kind == "doctest":
                from fastworkflow import planning as planning_api

                test = doctest.DocTestParser().get_doctest(
                    block, dict(vars(planning_api)), where, page, 0)
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
            with open(os.path.join(folder, "_commands", f"{name}.py"), "w") as fh:
                fh.write(text + "\n")
            where = f"{page}: broken {name}"
            if expect == "load-error":
                assert load_problem(folder, name) == message, where
            elif expect == "check-error":
                assert load_problem(folder, name) == message, where
                problems = command_preview.check_previews(folder, str(tmp_path / f"out{checked}"))
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
    assert text.endswith("answer written by __result__:\n  No book matches Ulysses.")


def test_pick_opens_the_exact_title_and_else_leaves_the_row_to_code_or_the_agent():
    assert "code runs: show_book book_uid=b1" in dry_run("book_by_title", {"title": "Dune"})
    assert "code runs: show_book book_uid=b2" in dry_run("book_by_title",
                                                       {"title": "dune messiah"})
    # no title is exactly "Dune Messia": the step is written in pick words, for the agent
    near = dry_run("book_by_title", {"title": "Dune Messia"})
    assert "1. show_book with the uid of the Dune Messia match" in near
    assert "code runs: show_book book_uid=b2" in near
    two = dry_run("book_by_title", {"title": "Dun"})
    assert "1. show_book with the uid of the Dun match" in two
    assert two.endswith("step 1 is open: the agent would run it and the rest")


def test_call_shapes_the_plan_and_a_brief_call_runs_flat():
    brief = dry_run("book_report", {"title": "Dune", "brief": True})
    assert [line for line in brief.splitlines() if line.startswith("Phase")] == ["Phase find"]
    assert brief.endswith("answer written by __result__:\n  Yes, the library has Dune.")
    without = dry_run("book_report", {"title": "Dune"})
    assert "Phase loans" not in without and "on loan" not in without
    assert "Phase loans" in dry_run("book_report", {"title": "Dune", "loans": True})


def test_a_fan_out_runs_once_per_found_book():
    text = dry_run("author_books", {"author": "Jane Austen"})
    assert "Phase book (item b3)" in text and "Phase book (item b4)" in text


def test_refusal_and_a_plain_call():
    assert "names no one member" in refusal("member_loans", {"member": "everyone"})
    assert refusal("member_loans", {"member": "Ann Lee"}) == ""


def test_the_composed_planning_command_places_the_base_phases_and_extends_the_base_answer():
    text = dry_run("book_loans", {"title": "Emma"})
    assert [line for line in text.splitlines() if line.startswith("Phase")] == [
        "Phase find", "Phase book", "Phase loans"]
    assert text.endswith("answer written by __result__:\n"
                         "  Emma is by Jane Austen; the library owns 2 copies.\n"
                         "  It is on loan to 1 borrower (Raj Patel).")


def test_a_planning_command_with_one_optional_input_runs_its_two_phases():
    # fix-1744: overdue_summary takes the branch a question names (optional), and its answer step
    # names the branches the first phase listed.
    command = workflow_agent.load_planning_command_map(WORKFLOW)["overdue_summary"]
    assert [(p.name, p.required) for p in command.params] == [("branch", False)]
    text = dry_run("overdue_summary", {})
    assert [line for line in text.splitlines() if line.startswith("Phase")] == [
        "Phase branches", "Phase overdue"]
    assert "code runs: list_branches" in text and "code runs: count_overdue" in text
    assert "name the branches (Central, Riverside)" in text
    text = dry_run("overdue_summary", {"branch": "Central"})
    assert "code runs: count_overdue branch=Central" in text
    assert "the number of overdue loans at Central" in text


def test_a_planning_step_is_read_against_its_input_model_on_the_library():
    """The reading dry_run uses for planning commands (Phase 5): the library's planning
    commands from resolve_field_sources, each read against its Input model."""
    planning = planning_commands.resolve_field_sources(LIBRARY).planning_commands
    assert {"Library/book_loans", "Library/book_by_title"} <= set(planning)
    parameters = sp.parameter_classes(_routing(LIBRARY), planning)
    [(step, run, _)] = sp.plan_code_steps("1. book_by_title title=Dune", ["book_by_title"], CORE,
                                          parameters)
    assert run == "book_by_title title=Dune"


if __name__ == "__main__":
    logging.disable(logging.ERROR)
    print(dry_run(sys.argv[1], json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}))

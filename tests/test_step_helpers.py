"""Phase 13 step helpers (``run``, ``full_list``, ``pick``, ``when``, ``do_not_run``):
their ``Step`` strings and ``.note()``, value quoting, unknown-keyword load checks, and
``pick`` reading the listing the call read across phases and pages."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import types
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, Field

import fastworkflow
from fastworkflow import skill_commands, skill_phases, skill_preview, tracing, workflow_agent
from fastworkflow._workflows.command_metadata_extraction.parameter_extraction import (
    ParameterExtraction,
)
from fastworkflow.command_executor import CommandExecutor, call_response_generator
from fastworkflow.examples.library._commands.Record.add_note import Signature as AddNote
from fastworkflow.observation_offloading import finish_check
from fastworkflow.observation_offloading.facts import workflow_signatures
from fastworkflow.planning import (
    StepFormError,
    do_not_run,
    fact_from,
    from_,
    full_list,
    pick,
    run,
    when,
)
from fastworkflow.skill import Step, command_inputs_for_steps, picks_noted, planned
from fastworkflow.skill_commands import SkillCallPlan, SkillFormatError, call_phases
from fastworkflow.skill_phases import Placeholder, step_paging
from fastworkflow.turn import TurnStatus
from fastworkflow.utils import python_utils
from fastworkflow.workflow_agent import initialize_workflow_tool_agent
from tests.test_skill_code_steps import Model
from tests.test_skill_phases import RecordingSink
from tests.test_turn_result_capture import (  # noqa: F401 - shared fixtures
    _make_agent_ctx,
    _set_agents,
    initialized_fastworkflow,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIBRARY = os.path.join(ROOT, "fastworkflow", "examples", "library")

_COPIED_PLANNING_COMMANDS = (
    "_commands/Library/overdue_summary.py",
    "_commands/Book/loan_report.py",
)

_COPIED_COMMANDS = (
    "_commands/library_hours.py",
    "_commands/startup.py",
    "_commands/Library/count_overdue.py",
    "_commands/Library/find_book.py",
    "_commands/Library/find_member.py",
    "_commands/Library/list_branches.py",
    "_commands/Library/open_book.py",
    "_commands/Library/open_member.py",
    "_commands/Book/list_loans.py",
    "_commands/Book/show_book.py",
    "_commands/Member/list_member_loans.py",
    "_commands/Member/show_member.py",
    "_commands/Record/add_note.py",
    "_commands/Record/show_notes.py",
)

_COPIED_CONTEXT_CLASSES = (
    "_commands/Library/_Library.py",
    "_commands/Book/_Book.py",
    "_commands/Member/_Member.py",
    "_commands/Record/_Record.py",
)


def _library_workflow(
    tmp_path,
    extra_skills: dict[str, str] | None = None,
    extra_commands: tuple[str, ...] = (),
) -> str:
    """Build a runnable copy of the library workflow in *tmp_path*."""
    folder = tmp_path / "library_wf"
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy2(os.path.join(LIBRARY, "__init__.py"), folder / "__init__.py")
    shutil.copy2(os.path.join(LIBRARY, "startup_action.json"), folder / "startup_action.json")
    shutil.copytree(os.path.join(LIBRARY, "application"), folder / "application")

    commands_dir = folder / "_commands"
    commands_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        os.path.join(LIBRARY, "_commands", "context_inheritance_model.json"),
        commands_dir / "context_inheritance_model.json",
    )
    for rel in (*_COPIED_CONTEXT_CLASSES, *_COPIED_COMMANDS, *extra_commands):
        src = os.path.join(LIBRARY, rel)
        dst = folder / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    skills_dir = folder / "_skills"
    skills_dir.mkdir(parents=True, exist_ok=True)
    for name, text in (extra_skills or {}).items():
        skill_path = skills_dir / f"{name}.py"
        skill_path.parent.mkdir(parents=True, exist_ok=True)
        skill_path.write_text(text, encoding="utf-8")

    return str(folder)


# ----------------------------------------------------------------------
# Step 1: the helpers, their Step strings, .note(), classification, quoting
# ----------------------------------------------------------------------

def test_each_helper_returns_the_step_text_and_marks_its_helper():
    s_run = run("find_book", query="Dune")
    assert s_run == "find_book query=Dune"
    assert isinstance(s_run, Step) and isinstance(s_run, str) and s_run.helper == "run"

    s_bare = run("list_branches")
    assert s_bare == "list_branches" and isinstance(s_bare, Step) and s_bare.helper == "run"

    s_full = full_list("list_loans", member_uid=Placeholder("member.uid"))
    assert s_full == "list_loans member_uid=<member.uid> (full list)"
    assert isinstance(s_full, Step) and s_full.helper == "full_list"

    s_pick_open = pick("open_book", named="Dune")
    assert s_pick_open == "open_book with the uid of the Dune match"
    assert isinstance(s_pick_open, Step) and s_pick_open.helper == "pick"

    s_pick_uid = pick("open_book", named="Dune", book_uid="b1")
    assert s_pick_uid == "open_book book_uid=b1"
    assert isinstance(s_pick_uid, Step) and s_pick_uid.helper == "pick"

    s_when = when("the member has loans", run("list_member_loans"))
    assert s_when == "only when the member has loans, list_member_loans"
    assert isinstance(s_when, Step) and s_when.helper == "when"

    s_dnr = do_not_run("add_note")
    assert s_dnr == "do not run add_note"
    assert isinstance(s_dnr, Step) and s_dnr.helper == "do_not_run"

    noted = s_run.note("the first page")
    assert noted == "find_book query=Dune (the first page)"
    assert isinstance(noted, Step) and noted.helper == "run"


def test_the_runtime_classifiers_read_full_list_when_and_do_not_run():
    commands = ["find_book", "list_loans", "list_member_loans", "add_note"]
    plan = "\n".join([
        f"1. {full_list('list_loans')}",
        f"2. {when('the member has loans', run('list_member_loans'))}",
        f"3. {do_not_run('add_note')}",
    ])
    steps = finish_check.parse_plan(plan, commands)
    assert [(s.number, s.kind, s.command) for s in steps] == [
        (1, finish_check.KIND_COMMAND, "list_loans"),
        (2, finish_check.KIND_CONDITIONAL, "list_member_loans"),
        (3, finish_check.KIND_SKIPPED, "add_note"),
    ]

    class LoansInput(BaseModel):
        start: int = 0

    paging, every_page = step_paging(
        steps[0], lambda name: [LoansInput] if name == "list_loans" else []
    )
    assert paging == ["start"] and every_page is True


def test_run_and_full_list_quote_parser_characters_and_keep_plain_names_and_placeholders():
    # Spaces alone are not quoted; None is left out; placeholders are written unquoted.
    assert run("find_member", query="Raj Patel") == "find_member query=Raj Patel"
    assert run("find_book", query="Dune", start=None) == "find_book query=Dune"
    assert run("find_book", query=Placeholder("title")) == "find_book query=<title>"
    assert run("find_book", query="<title>") == "find_book query=<title>"
    assert run("add_note", text="   ") == "add_note text=   "

    # Values holding "(", ";", "=", ", ", brackets, quotes or "..." are quoted.
    assert run("add_note", text="prefers email (not phone)") == (
        'add_note text="prefers email (not phone)"'
    )
    assert run("add_note", text="x; y") == 'add_note text="x; y"'
    assert run("add_note", text="a=b") == 'add_note text="a=b"'
    assert run("add_note", text="Lee, Ann") == 'add_note text="Lee, Ann"'
    assert run("add_note", text="see [1]") == 'add_note text="see [1]"'
    assert run("add_note", text="see {a}") == 'add_note text="see {a}"'
    assert run("add_note", text="see <link>") == 'add_note text="see <link>"'
    assert run("add_note", text="wait...") == 'add_note text="wait..."'
    assert run("add_note", text='say "hi"') == """add_note text='say "hi"'"""
    assert full_list("find_book", query="Dune (1965)") == (
        'find_book query="Dune (1965)" (full list)'
    )


def test_an_unknown_keyword_raises_step_form_error_and_reports_in_step_command_problems(tmp_path):
    bad_command = '''from pydantic import BaseModel, Field
from fastworkflow.planning import run


class Signature:
    """Search for a book."""

    class Input(BaseModel):
        title: str = Field(description="the book's title")


class PlanGenerator:
    def search(self, input):
        return [run("find_books", query=input.title),
                run("find_book", title=input.title)]
'''
    quoted_note = '''from pydantic import BaseModel, Field
from fastworkflow.planning import run


class Signature:
    """Add a note with an equals sign."""

    class Input(BaseModel):
        member: str = Field(description="the member's name")


class PlanGenerator:
    def note(self, input):
        return [run("add_note", text="a=b"),
                run("add_note", text="x, y=z"),
                run("add_note", text="see (a=b)")]
'''
    folder = _library_workflow(
        tmp_path,
        {"Library/bad_command": bad_command, "Record/quoted_note": quoted_note},
    )
    sigs = workflow_signatures(folder)

    with command_inputs_for_steps(sigs):
        assert run("find_book", query="Dune") == "find_book query=Dune"
        with pytest.raises(
            StepFormError,
            match="a step passes title to 'find_book', whose Signature.Input has no such field",
        ):
            run("find_book", title="Dune")
        with pytest.raises(
            StepFormError,
            match="a step passes branch to 'list_branches', whose Signature.Input has no such field",
        ):
            run("list_branches", branch="Central")

    bad_path = os.path.join(folder, "_skills", "Library", "bad_command.py")
    loaded_bad = workflow_agent._read_skill_file("Library/bad_command", bad_path, folder)
    problems = [
        p.replace(folder + os.sep, "")
        for p in workflow_agent.step_command_problems(loaded_bad.command, folder)
    ]
    assert problems == [
        "_skills/Library/bad_command.py: a step runs 'find_books', which is not a command of the workflow",
        "_skills/Library/bad_command.py: a step passes title to 'find_book', whose Signature.Input has no such field",
    ]

    # Quoted values holding "a=b", "x, y=z", or "see (a=b)" do not report spurious keys.
    note_path = os.path.join(folder, "_skills", "Record", "quoted_note.py")
    loaded_note = workflow_agent._read_skill_file("Record/quoted_note", note_path, folder)
    assert workflow_agent.step_command_problems(loaded_note.command, folder) == []


# ----------------------------------------------------------------------
# Step 2: a note that holds a condition is a preview --check warning
# ----------------------------------------------------------------------

def test_a_note_holding_a_condition_warns_on_preview_check_and_when_or_bare_only_do_not(tmp_path, capsys):
    bad_note = '''from pydantic import BaseModel, Field
from fastworkflow.planning import run


class Signature:
    """Add a note."""

    class Input(BaseModel):
        text: str = Field(description="the note")


class PlanGenerator:
    def note(self, input):
        return [run("add_note", text=input.text).note("only when the member asked")]
'''
    bad_hand_note = '''from pydantic import BaseModel, Field


class Signature:
    """Renew a loan."""

    class Input(BaseModel):
        loan: str = Field(description="the loan")


class PlanGenerator:
    def check(self, input):
        return ["show_notes (if the member has any)"]
'''
    good_steps = '''from pydantic import BaseModel, Field
from fastworkflow.planning import fact_from, full_list, run, when


class Found(BaseModel):
    book_uids: list[str] = fact_from("find_book.book_uids")


class Signature:
    """Check loans and notes."""

    class Input(BaseModel):
        title: str = Field(description="the book's title")
        text: str = Field(description="the note")


class PlanGenerator:
    def find(self, input) -> Found:
        return [
            run("find_book", query=input.title).note("the first page"),
            run("find_book", query=input.title).note("first page only"),
        ]

    def details(self, input, find):
        if not find.book_uids:
            return []
        return [
            run("add_note", text=input.text).note("the number only"),
            run("add_note", text=input.text).note("only if the member asked"),
            full_list("list_loans"),
            when("the member asked", run("add_note", text=input.text)),
        ]
'''
    folder = _library_workflow(
        tmp_path,
        {
            "Record/bad_note": bad_note,
            "Record/bad_hand_note": bad_hand_note,
            "Library/good_steps": good_steps,
        },
    )
    sigs = workflow_signatures(folder)

    bad_path = os.path.join(folder, "_skills", "Record", "bad_note.py")
    loaded_bad = workflow_agent._read_skill_file("Record/bad_note", bad_path, folder)
    assert workflow_agent.step_command_problems(loaded_bad.command, folder) == []
    bad_note_warnings = [
        w.replace(folder + os.sep, "")
        for w in skill_commands.check_note_condition_steps(loaded_bad.command, sigs)
    ]
    assert bad_note_warnings == [
        "_skills/Record/bad_note.py: phase 'note', step 'add_note text=<text> (only when the member asked)': "
        "the note holds a condition, but code never reads a note, so add_note would run every time. "
        'Decide it in code with if, or write when("the member asked", "add_note text=<text>") for the agent to decide'
    ]

    hand_path = os.path.join(folder, "_skills", "Record", "bad_hand_note.py")
    loaded_hand = workflow_agent._read_skill_file("Record/bad_hand_note", hand_path, folder)
    assert workflow_agent.step_command_problems(loaded_hand.command, folder) == []
    hand_warnings = [
        w.replace(folder + os.sep, "")
        for w in skill_commands.check_note_condition_steps(loaded_hand.command, sigs)
    ]
    assert hand_warnings == [
        "_skills/Record/bad_hand_note.py: phase 'check', step 'show_notes (if the member has any)': "
        "the note holds a condition, but code never reads a note, so show_notes would run every time. "
        'Decide it in code with if, or write when("the member has any", "show_notes") for the agent to decide'
    ]

    loaded_good = workflow_agent._read_skill_file(
        "Library/good_steps",
        os.path.join(folder, "_skills", "Library", "good_steps.py"),
        folder,
    )
    assert workflow_agent.step_command_problems(loaded_good.command, folder) == []
    assert any("only when find.book_uids" in s for s in loaded_good.command.template)
    good_warnings = skill_commands.check_note_condition_steps(loaded_good.command, sigs)
    assert len(good_warnings) == 1
    assert "only if the member asked" in good_warnings[0]
    assert not any("first page only" in w for w in good_warnings)
    assert not any("the number only" in w for w in good_warnings)
    assert {name for name, _, _ in workflow_agent.load_skills(folder)} == {
        "Record/bad_note",
        "Record/bad_hand_note",
        "Library/good_steps",
    }

    out_dir = str(tmp_path / "previews")
    skill_preview.write_previews(folder, out_dir)
    capsys.readouterr()
    rc = skill_preview.preview_main(
        argparse.Namespace(workflow_folder=folder, out=out_dir, check=True)
    )
    assert rc == 0
    note_lines = [
        line.replace(folder + os.sep, "")
        for line in capsys.readouterr().out.splitlines()
        if "the note holds a condition" in line
    ]
    assert len(note_lines) == 3
    assert {line.split(":")[0] for line in note_lines} == {
        "_skills/Record/bad_note.py",
        "_skills/Record/bad_hand_note.py",
        "_skills/Library/good_steps.py",
    }


def test_pick_without_named_is_a_skill_load_error(tmp_path):
    bad_pick = '''from pydantic import BaseModel, Field
from fastworkflow.planning import fact_from, pick, run


class Found(BaseModel):
    book_uids: list[str] = fact_from("find_book.book_uids")


class Signature:
    """Open a book without naming the row."""

    class Input(BaseModel):
        title: str = Field(description="the book's title")


class PlanGenerator:
    def find(self, input) -> Found:
        return [run("find_book", query=input.title)]

    def book(self, input, find):
        return [pick("open_book")]
'''
    folder = _library_workflow(tmp_path, {"Library/bad_pick": bad_pick})
    path = os.path.join(folder, "_skills", "Library", "bad_pick.py")
    with pytest.raises(
        SkillFormatError,
        match=r"bad_pick\.py: phase 'book' fails while its template is built .*"
        r"pick\('open_book'\): named= is the name the row is picked by",
    ):
        workflow_agent._read_skill_file("Library/bad_pick", path, folder)
    assert {name for name, _, _ in workflow_agent.load_skills(folder)} == set()


# ----------------------------------------------------------------------
# Step 3: the examples guard, and quoted values in code-run steps
# ----------------------------------------------------------------------

def test_code_runs_quoted_values_and_example_values_while_hand_typed_and_cme_keep_their_guards(
    library_env,
):
    # 1. Values holding "(", "a=b", "<link>", "...", ';', and '"' bind whole when written with run(),
    # and so does a value equal to one of the field's examples ("the cover is torn").
    special_values = [
        "prefers email (not phone)",
        "a=b",
        "see <link>",
        "wait...",
        'say "hi"',
        "the cover is torn",  # equal to AddNote.Input.text's examples[0]
    ]

    class WriteNotes:
        def find(self, input) -> FoundBooks:
            return [run("find_book", query=input.title)]

        def open(self, input, find) -> OpenedBook:
            return [pick("open_book", named=input.title)]

        def notes(self, open):
            return [run("add_note", text=v).note("for the record") for v in special_values]

        def answer(self, open):
            return [f"say that notes were added to {open.book_uid}"]

    # LibraryRunner reads wf.context["by_code"], which run_execute_workflow_query sets for
    # code-run steps and clears in its finally block.
    orig_extract = ParameterExtraction._extract_parameters_from_key_values
    library_env.skill = _build_skill(WriteNotes)

    # First verify the parser without code_written declines the example value and placeholders,
    # and drops the trailing parenthetical on the unquoted f-string form.
    assert (
        orig_extract('add_note text="the cover is torn"', AddNote.Input, "add_note") is None
    )
    assert orig_extract('add_note text="see <link>"', AddNote.Input, "add_note") is None
    assert orig_extract('add_note text="wait..."', AddNote.Input, "add_note") is None
    assert orig_extract('add_note text="a=b"', AddNote.Input, "add_note") is not None
    fstring_step = f"add_note text={'prefers email (not phone)'}"
    fstring_bound = orig_extract(fstring_step, AddNote.Input, "add_note")
    assert fstring_bound is not None and fstring_bound.text == "prefers email"
    fstring_plan_step = finish_check.parse_plan(f"1. {fstring_step}", ["add_note"])[0]
    assert skill_phases.code_step(fstring_plan_step, AddNote.Input) == "add_note text=prefers email"

    # Now run the skill end-to-end through the phase runner and verify every note bound whole.
    lib = library_env.wf.root_command_context
    before_notes = list(lib.books["b1"].notes)
    result = library_env.ctx.process_turn("look up Dune")

    assert result.status == TurnStatus.COMPLETED
    assert library_env.model.asked == []
    added = lib.books["b1"].notes[len(before_notes):]
    assert added == special_values
    # None of the ".note('for the record')" text was sent to the command.
    assert all("for the record" not in c for c in library_env.runner.sent)
    # After the turn, by_code is cleared from the workflow context, and a user-typed
    # command path declines an example value.
    assert "by_code" not in library_env.wf.context
    user_code_written = bool(library_env.wf.context.get("by_code"))
    assert (
        orig_extract(
            'add_note text="the cover is torn"',
            AddNote.Input,
            "add_note",
            code_written=user_code_written,
        )
        is None
    )


def test_the_cme_declines_an_example_value_in_a_step_the_agent_wrote(library_env):
    """Agent ``execute_workflow_query`` keeps the examples guard (``by_code`` is false).

    A hand-typed plan step is still run by ``code_step`` with ``code_written=True``, so the
    example value would bind in code; the agent must write the command outside that path.
    """
    lib = library_env.wf.root_command_context
    book_cls = type(lib.books["b1"])
    lib.books["b5"] = book_cls(lib, "b5", "Dune", "Frank Herbert", 1, "Riverside")

    class Plan:
        def find(self, input) -> FoundBooks:
            return [run("find_book", query=input.title)]

        def book(self, input, find):
            return [pick("open_book", named=input.title)]

        def answer(self):
            return ["say that the book was opened"]

    library_env.skill = _build_skill(Plan)
    before_notes = list(lib.books["b1"].notes)
    library_env.model.run("open_book book_uid=b1")
    library_env.model.run("add_note text=the cover is torn")
    library_env.model.run("add_note text=spine is cracked")
    library_env.model.finish()
    result = library_env.ctx.process_turn("look up Dune")

    assert result.status == TurnStatus.COMPLETED
    assert lib.books["b1"].notes[len(before_notes):] == ["spine is cracked"]
    assert "add_note text=the cover is torn" in library_env.runner.sent
    assert "add_note text=spine is cracked" in library_env.runner.sent
    assert "by_code" not in library_env.wf.context


# ----------------------------------------------------------------------
# Step 1a: pick from the listing the call read
# ----------------------------------------------------------------------

def test_pick_without_uid_and_error_forms():
    assert pick("open_book", named="Dune") == "open_book with the uid of the Dune match"
    assert pick("open_book", named="Dune", book_uid=None) == (
        "open_book with the uid of the Dune match"
    )
    assert pick("open_book", named="Dune", book_uid="b2") == "open_book book_uid=b2"
    with pytest.raises(StepFormError, match="named= is the name the row is picked by"):
        pick("open_book")
    with pytest.raises(StepFormError, match="named= is the name the row is picked by"):
        pick("open_book", named="  ")
    with pytest.raises(StepFormError):
        pick("open_book", named="Dune", book_uid="b1", member_uid="m1")
    with pytest.raises(StepFormError):
        pick("open_book", named="Dune", title="Dune")

    class OpenBookInput(BaseModel):
        book_uid: str

    with command_inputs_for_steps({"Library/open_book": SimpleNamespace(Input=OpenBookInput)}):
        with picks_noted() as picks:
            pick("open_book", named="Dune")
    assert picks == [("open_book", "book_uid")]


def _qualified(workflow, name: str):
    routing = fastworkflow.RoutingRegistry.get_definition(workflow.folderpath)
    for ctx in (workflow.current_command_context_name, "*"):
        for candidate in routing.get_command_names(ctx):
            if candidate.split("/")[-1] == name:
                return candidate
    for candidate in routing.routing_definition_map:
        if candidate != "wildcard" and candidate.split("/")[-1] == name:
            return candidate
    return None


class LibraryRunner:
    """Runs the library copy's commands in code, as the NLU would route them."""

    def __init__(self, wf):
        self.wf = wf
        self.sent: list[str] = []
        #: When set, append ``Next page: start=…`` to ``find_book`` observations that still
        #: have rows (the library command returns JSON only; code reads paging from the text).
        self.name_find_book_next_pages = False

    def __call__(self, session, command: str):
        self.sent.append(command)
        wf = self.wf
        name = command.split()[0]
        if name in ("go_up", "reset_context"):
            if name == "reset_context" or wf.get_parent(wf.current_command_context) is None:
                wf.current_command_context = wf.root_command_context
            else:
                wf.current_command_context = wf.get_parent(wf.current_command_context)
            return fastworkflow.CommandOutput(
                command_name=f"IntentDetection/{name}",
                command_response=fastworkflow.CommandResponse(
                    response=f"Context is now '{wf.current_command_context_displayname}'"
                ),
            )
        qualified = _qualified(wf, name)
        routing = fastworkflow.RoutingRegistry.get_definition(wf.folderpath)
        cls = routing.get_command_class(qualified, fastworkflow.ModuleType.COMMAND_PARAMETERS_CLASS)
        gen_cls = routing.get_command_class(
            qualified, fastworkflow.ModuleType.RESPONSE_GENERATION_INFERENCE
        )
        params = None
        if cls is not None:
            by_code = bool(wf.context.get("by_code"))
            params = (
                ParameterExtraction._extract_parameters_from_key_values(
                    command, cls, name, code_written=by_code
                )
                or ParameterExtraction._extract_parameters_from_bare(command, cls, name)
            )
        wf.command_context_for_response_generation = wf.current_command_context
        raw = call_response_generator(gen_cls(), wf, command, params)
        if isinstance(raw, fastworkflow.CommandOutput):
            out = raw
        elif isinstance(raw, BaseModel):
            out = fastworkflow.CommandOutput(
                command_name=qualified,
                command_parameters=params,
                command_response=fastworkflow.CommandResponse(
                    response=raw.model_dump_json(), output=raw
                ),
            )
        else:
            out = fastworkflow.CommandOutput(
                command_name=qualified,
                command_parameters=params,
                command_response=fastworkflow.CommandResponse(response=str(raw)),
            )
        out.command_name = qualified
        out.command_parameters = params
        if self.name_find_book_next_pages and name == "find_book":
            data = out.command_response.output
            if data is not None and hasattr(data, "book_uids") and hasattr(data, "total"):
                shown = (out.command_parameters.start or 0) + len(data.book_uids)
                if shown < data.total:
                    out.command_response.response = (
                        f"{out.command_response.response}\nNext page: start={shown}"
                    )
        return out


@pytest.fixture
def library_env(tmp_path, initialized_fastworkflow, monkeypatch):
    """A runnable library workflow in *tmp_path* driven through the real phase runner."""
    monkeypatch.delenv(skill_phases.PHASE_MAX_STEPS_ENV, raising=False)
    folder = _library_workflow(tmp_path)
    ctx, wf = _make_agent_ctx(folder, monkeypatch)
    sink = RecordingSink()
    ctx.set_trace_sink(sink)
    ctx.push_active_workflow(wf)

    runner = LibraryRunner(wf)
    runner(ctx, "startup")

    agent = initialize_workflow_tool_agent(ctx)
    _set_agents(ctx, agent)
    model = Model(agent)
    monkeypatch.setattr(agent, "_call_with_potential_trajectory_truncation", model)
    monkeypatch.setattr(
        CommandExecutor,
        "invoke_command",
        classmethod(lambda cls, session, command: runner(session, command)),
    )

    state = SimpleNamespace(
        ctx=ctx,
        wf=wf,
        sink=sink,
        agent=agent,
        model=model,
        runner=runner,
        skill=None,
        args={"title": "Dune"},
    )

    def fake_plan(user_query, session, with_agent_inputs_and_trajectory=False,
                  planner_lm=None, skill_command=None, **kwargs):
        if skill_command is None:
            return user_query
        call = SkillCallPlan(
            plan="",
            args=dict(state.args),
            notes=[],
            prediction=None,
            phases=call_phases(skill_command, dict(state.args)),
            steps_after=[],
            steps_before=[],
        )
        session._phased_skill_call = (skill_command, call)
        session._initial_plan_text = ""
        return user_query

    monkeypatch.setattr("fastworkflow.workflow_agent.build_query_with_next_steps", fake_plan)
    monkeypatch.setattr(
        "fastworkflow.workflow_agent.planner_skill_for",
        lambda text, folder_path, **_: ("", state.skill),
    )
    monkeypatch.setattr(
        "fastworkflow.workflow_agent.split_into_subtasks",
        lambda *args, **kwargs: ["look up the book"],
    )
    monkeypatch.setattr("fastworkflow.skill_phases.write_answer", lambda query: "written")

    runner.sent.clear()
    yield state
    ctx.pop_active_workflow()


def _real_planning_command(qualified_name: str, rel_under_library: str):
    fpath = os.path.join(LIBRARY, rel_under_library)
    mod = python_utils.get_module(fpath, LIBRARY)
    return skill_commands.command_from_module(qualified_name, mod, fpath)


def _make_library_runner_env(
    tmp_path,
    initialized_fastworkflow,
    monkeypatch,
    *,
    extra_skills: dict[str, str] | None = None,
    extra_commands: tuple[str, ...] = (),
    args: dict | None = None,
):
    """Like ``library_env``, on a copy that may include extra commands or skills."""
    monkeypatch.delenv(skill_phases.PHASE_MAX_STEPS_ENV, raising=False)
    folder = _library_workflow(tmp_path, extra_skills=extra_skills, extra_commands=extra_commands)
    ctx, wf = _make_agent_ctx(folder, monkeypatch)
    sink = RecordingSink()
    ctx.set_trace_sink(sink)
    ctx.push_active_workflow(wf)

    runner = LibraryRunner(wf)
    runner(ctx, "startup")

    agent = initialize_workflow_tool_agent(ctx)
    _set_agents(ctx, agent)
    model = Model(agent)
    monkeypatch.setattr(agent, "_call_with_potential_trajectory_truncation", model)
    monkeypatch.setattr(
        CommandExecutor,
        "invoke_command",
        classmethod(lambda cls, session, command: runner(session, command)),
    )

    state = SimpleNamespace(
        ctx=ctx,
        wf=wf,
        sink=sink,
        agent=agent,
        model=model,
        runner=runner,
        skill=None,
        args=args or {},
    )

    def fake_plan(user_query, session, with_agent_inputs_and_trajectory=False,
                  planner_lm=None, skill_command=None, **kwargs):
        if skill_command is None:
            return user_query
        call = SkillCallPlan(
            plan="",
            args=dict(state.args),
            notes=[],
            prediction=None,
            phases=call_phases(skill_command, dict(state.args)),
            steps_after=[],
            steps_before=[],
        )
        session._phased_skill_call = (skill_command, call)
        session._initial_plan_text = ""
        return user_query

    monkeypatch.setattr("fastworkflow.workflow_agent.build_query_with_next_steps", fake_plan)
    monkeypatch.setattr(
        "fastworkflow.workflow_agent.planner_skill_for",
        lambda text, folder_path, **_: ("", state.skill),
    )
    monkeypatch.setattr(
        "fastworkflow.workflow_agent.split_into_subtasks",
        lambda *args, **kwargs: ["look up the book"],
    )
    monkeypatch.setattr("fastworkflow.skill_phases.write_answer", lambda query: "written")

    runner.sent.clear()
    return state


@pytest.fixture
def library_proof_env(tmp_path, initialized_fastworkflow, monkeypatch):
    state = _make_library_runner_env(
        tmp_path,
        initialized_fastworkflow,
        monkeypatch,
        extra_commands=_COPIED_PLANNING_COMMANDS,
    )
    yield state
    state.ctx.pop_active_workflow()


def _note_member_without_in_request() -> str:
    path = os.path.join(LIBRARY, "_commands", "Library", "note_member.py")
    text = open(path, encoding="utf-8").read()
    return text.replace(
        'member: str = Field(description="the member\'s name",\n'
        '                            json_schema_extra={"in_request": True})',
        'member: str = Field(description="the member\'s name")',
    )


class FoundBooks(BaseModel):
    book_uids: list[str] = fact_from("find_book.book_uids")


class OpenedBook(BaseModel):
    book_uid: str = fact_from("open_book.book_uid")
    title: str = fact_from("open_book.title")


def _build_skill(plan_cls, args_fields: dict[str, tuple[type, object]] | None = None):
    fields = args_fields or {"title": (str, Field(description="the book's title"))}
    input_cls = type(
        "Input",
        (BaseModel,),
        {
            "__annotations__": {k: t for k, (t, _) in fields.items()},
            **{k: v for k, (_, v) in fields.items()},
        },
    )
    sig_cls = type("Signature", (), {"__doc__": "Look up a book.", "Input": input_cls})
    module = types.ModuleType("book_lookup")
    module.Signature, module.PlanGenerator = sig_cls, plan_cls
    return skill_commands.command_from_module(
        "Library/book_lookup", module, "_skills/Library/book_lookup.py"
    ).command


def test_pick_in_a_later_phase_runs_by_code_on_the_one_exact_match(library_env):
    class Plan:
        def find(self, input) -> FoundBooks:
            return [run("find_book", query=input.title)]

        def book(self, input, find) -> OpenedBook:
            return [pick("open_book", named=input.title)]

        def answer(self, book):
            return [f"say that {book.title} ({book.book_uid}) was opened"]

    library_env.skill = _build_skill(Plan)
    result = library_env.ctx.process_turn("look up Dune")

    assert result.status == TurnStatus.COMPLETED
    assert library_env.model.asked == []
    # find_book query=Dune returned b1 ("Dune") and b2 ("Dune Messiah"); pick opened b1 in code.
    assert library_env.runner.sent == ["find_book query=Dune", "open_book book_uid=b1"]
    spans = library_env.sink.named(tracing.SPAN_SKILL_PHASE)
    assert [s.attributes["phase"] for s in spans] == ["find", "book", "answer"]
    assert spans[1].attributes["code_stop"] == "done"
    assert spans[1].attributes["facts"] == {"book_uid": "b1", "title": "Dune"}


def test_pick_goes_to_the_agent_when_two_rows_or_none_match(library_env):
    class Plan:
        def find(self, input) -> FoundBooks:
            return [run("find_book", query=input.title)]

        def book(self, input, find):
            return [pick("open_book", named=input.title)]

    library_env.skill = _build_skill(Plan)

    # 1. Two rows titled "Dune": add a second edition in the tmp copy's in-memory library.
    lib = library_env.wf.root_command_context
    book_cls = type(lib.books["b1"])
    lib.books["b5"] = book_cls(lib, "b5", "Dune", "Frank Herbert", 1, "Riverside")

    library_env.model.run("open_book book_uid=b1")
    library_env.model.finish()
    result = library_env.ctx.process_turn("look up Dune")

    assert result.status == TurnStatus.COMPLETED
    assert len(library_env.model.asked) == 2
    assert "1. open_book with the uid of the Dune match" in library_env.model.asked[0]["query"]
    spans = library_env.sink.named(tracing.SPAN_SKILL_PHASE)
    assert spans[1].attributes["code_stop"] == "open step 1"

    # 2. No row matches the named title ("Dune" when only "Dune 1" and "Dune Messiah" exist).
    del lib.books["b5"]
    lib.books["b1"].title = "Dune 1"
    library_env.wf.current_command_context = lib
    library_env.runner.sent.clear()
    library_env.model.asked.clear()
    library_env.model.run("open_book book_uid=b1")
    library_env.model.finish()

    result2 = library_env.ctx.process_turn("look up Dune")
    assert result2.status == TurnStatus.COMPLETED
    assert len(library_env.model.asked) == 2
    assert "1. open_book with the uid of the Dune match" in library_env.model.asked[0]["query"]


def test_pick_chooses_the_listing_whose_uid_list_fits_the_command_parameter(library_env):
    # find_member query=Dune runs after find_book query=Dune and finds a member named Dune,
    # but open_book's book_uid parameter still picks from find_book's book_uids.
    lib = library_env.wf.root_command_context
    member_cls = type(lib.members["m1"])
    lib.members["m9"] = member_cls(lib, "m9", "Dune")

    class Plan:
        def find(self, input) -> FoundBooks:
            return [
                run("find_book", query=input.title),
                run("find_member", query=input.title),
            ]

        def book(self, input, find) -> OpenedBook:
            return [pick("open_book", named=input.title)]

        def answer(self, book):
            return [f"say that {book.book_uid} was opened"]

    library_env.skill = _build_skill(Plan)
    result = library_env.ctx.process_turn("look up Dune")

    assert result.status == TurnStatus.COMPLETED
    assert library_env.model.asked == []
    assert library_env.runner.sent == [
        "find_book query=Dune",
        "find_member query=Dune",
        "open_book book_uid=b1",
    ]


def test_pick_uses_an_explicit_uid_as_given(library_env):
    class Plan:
        def find(self, input) -> FoundBooks:
            return [run("find_book", query=input.title)]

        def book(self, input, find) -> OpenedBook:
            return [pick("open_book", named=input.title, book_uid=find.book_uids[-1])]

        def answer(self, book):
            return [f"say that {book.book_uid} was opened"]

    library_env.skill = _build_skill(Plan)
    result = library_env.ctx.process_turn("look up Dune")

    assert result.status == TurnStatus.COMPLETED
    assert library_env.model.asked == []
    assert library_env.runner.sent == ["find_book query=Dune", "open_book book_uid=b2"]


def test_pick_joins_pages_of_a_full_list_listing(library_env, monkeypatch):
    # Shrink PAGE_SIZE to 1 on the tmp copy's library module so find_book query="Frank Herbert"
    # has b1 ("Dune") on page 1 and b2 (renamed "Frank Herbert") on page 2, and pick("open_book",
    # named="Frank Herbert") in the next phase picks b2 from the joined pages.
    lib = library_env.wf.root_command_context
    lib_mod = sys.modules[type(lib).__module__]
    monkeypatch.setattr(lib_mod, "PAGE_SIZE", 1)
    lib.books["b2"].title = "Frank Herbert"
    # Real find_book only returns structured Output (book_uids, total); it does not write
    # "Next page: start=…" in the response. Code reads that phrase from the observation text,
    # so the harness names the next page the way paged commands do elsewhere in the suite.
    library_env.runner.name_find_book_next_pages = True

    class AuthorPlan:
        def find(self, input) -> FoundBooks:
            return [full_list("find_book", query="Frank Herbert")]

        def book(self, input, find) -> OpenedBook:
            return [pick("open_book", named="Frank Herbert")]

        def answer(self, book):
            return [f"say that {book.book_uid} was opened"]

    library_env.skill = _build_skill(AuthorPlan)

    result = library_env.ctx.process_turn("look up Frank Herbert")
    assert result.status == TurnStatus.COMPLETED
    assert library_env.model.asked == []
    assert library_env.runner.sent == [
        "find_book query=Frank Herbert",
        "find_book query=Frank Herbert start=1",
        "open_book book_uid=b2",
    ]


# ----------------------------------------------------------------------
# Step 4: preview --check warns on hand-typed forms without failing
# ----------------------------------------------------------------------

def test_hand_typed_forms_get_their_warnings_and_preview_check_exits_zero(tmp_path, capsys):
    hand_typed = '''from pydantic import BaseModel, Field


class Signature:
    """Exercise every hand-typed step form."""

    class Input(BaseModel):
        title: str = Field(description="the book's title")
        text: str = Field(default="prefers email (not phone)", description="the note")


class PlanGenerator:
    def branches(self):
        return ["list_branches"]

    def search(self, input):
        return [
            f"find_book query={input.title}",
            f"add_note text={'prefers email (not phone)'}",
            "open_book with the uid of the <title> match",
            "list_loans (full list)",
            "only when the book has notes, show_notes",
            "do not run add_note",
            "say what was found",
        ]
'''
    helper_written = '''from pydantic import BaseModel, Field
from fastworkflow.planning import do_not_run, fact_from, full_list, pick, run, when


class Found(BaseModel):
    book_uids: list[str] = fact_from("find_book.book_uids")


class Signature:
    """Exercise every helper-written step form."""

    class Input(BaseModel):
        title: str = Field(description="the book's title")
        text: str = Field(description="the note")


class PlanGenerator:
    def branches(self):
        return [run("list_branches")]

    def find(self, input) -> Found:
        return [run("find_book", query=input.title).note("the first page")]

    def book(self, input, find):
        if not find.book_uids:
            return []
        return [
            pick("open_book", named=input.title),
            full_list("list_loans"),
            when("the member asked", run("add_note", text=input.text)),
            do_not_run("add_note"),
            "say what was found",
        ]
'''
    folder = _library_workflow(
        tmp_path,
        {
            "Library/hand_typed": hand_typed,
            "Library/helper_written": helper_written,
        },
    )
    out_dir = str(tmp_path / "previews")
    skill_preview.write_previews(folder, out_dir)
    capsys.readouterr()

    # preview --check exits 0 and prints the warnings for hand_typed, none for helper_written.
    rc = skill_preview.preview_main(
        argparse.Namespace(workflow_folder=folder, out=out_dir, check=True)
    )
    assert rc == 0
    stdout = capsys.readouterr().out
    rel_lines = [
        line.replace(folder + os.sep, "")
        for line in stdout.splitlines()
        if "written by hand" in line
    ]
    assert rel_lines == [
        "_skills/Library/hand_typed.py: phase 'branches', step 'list_branches': written by hand; write run(...) so that code reads it as you mean",
        "_skills/Library/hand_typed.py: phase 'search', step 'find_book query=<title>': written by hand; write run(...) so that code reads it as you mean",
        "_skills/Library/hand_typed.py: phase 'search', step 'add_note text=prefers email (not phone)': written by hand; write run(...) so that code reads it as you mean",
        "_skills/Library/hand_typed.py: phase 'search', step 'open_book with the uid of the <title> match': written by hand; write pick(...) so that code reads it as you mean",
        "_skills/Library/hand_typed.py: phase 'search', step 'list_loans (full list)': written by hand; write full_list(...) so that code reads it as you mean",
        "_skills/Library/hand_typed.py: phase 'search', step 'only when the book has notes, show_notes': written by hand; write when(...) so that code reads it as you mean",
        "_skills/Library/hand_typed.py: phase 'search', step 'do not run add_note': written by hand; write do_not_run(...) so that code reads it as you mean",
    ]
    assert not any("helper_written" in line for line in rel_lines)

    # Also verify the library's loadable planning commands (overdue_summary, Book/loan_report) get no warnings.
    sigs = workflow_signatures(LIBRARY)
    for qname, rel_path in (
        ("Book/loan_report", "fastworkflow/examples/library/_commands/Book/loan_report.py"),
        ("Library/overdue_summary", "fastworkflow/examples/library/_commands/Library/overdue_summary.py"),
    ):
        fpath = os.path.join(ROOT, rel_path)
        mod = python_utils.get_module(fpath, LIBRARY)
        sf = skill_commands.command_from_module(qname, mod, fpath)
        assert skill_commands.check_hand_typed_steps(sf, sigs) == []


# ----------------------------------------------------------------------
# Library proof (13.10)
# ----------------------------------------------------------------------

# Pins today's load state for every library ``PlanGenerator``; update rows when Phase 3
# (``find_record`` union return) and Phase 11 (``in_request`` on str fields) land — do not
# delete ``test_library_planning_commands_load_inventory``.
_LIBRARY_PLANNING_INVENTORY = (
    ("Book/loan_report", "_commands/Book/loan_report.py", True, "", ""),
    ("Book/shelf_report", "_commands/Book/shelf_report.py", True, "", ""),
    ("Library/overdue_summary", "_commands/Library/overdue_summary.py", True, "", ""),
    ("Library/author_books", "_commands/Library/author_books.py", False, "in_request", "11"),
    ("Library/book_by_title", "_commands/Library/book_by_title.py", False, "in_request", "11"),
    ("Library/book_loans", "_commands/Library/book_loans.py", False, "in_request", "11"),
    ("Library/count_books", "_commands/Library/count_books.py", False, "in_request", "11"),
    ("Library/find_record", "_commands/Library/find_record.py", False, "facts model it fills", "3"),
    ("Library/member_report", "_commands/Library/member_report.py", False, "in_request", "11"),
    ("Library/note_member", "_commands/Library/note_member.py", False, "in_request", "11"),
)


def test_library_planning_commands_load_inventory():
    """Every ``PlanGenerator`` under the real library folder: load status for 13.10."""
    sigs = workflow_signatures(LIBRARY)
    for qname, rel, expect_load, err_key, phase in _LIBRARY_PLANNING_INVENTORY:
        fpath = os.path.join(LIBRARY, rel)
        text = open(fpath, encoding="utf-8").read()
        imports_planning = "fastworkflow.planning" in text
        assert imports_planning, f"{qname} should import from fastworkflow.planning"
        hand = []
        try:
            sf = _real_planning_command(qname, rel)
            problems = workflow_agent.step_command_problems(sf.command, LIBRARY)
            loads = not problems
            hand = skill_commands.check_hand_typed_steps(sf, sigs)
        except SkillFormatError as error:
            loads = False
            assert err_key in str(error), (qname, error)
        assert loads == expect_load, qname
        if loads:
            assert hand == [], qname
        if not expect_load:
            assert phase


def test_library_overdue_summary_from_real_module_runs_code_steps(library_proof_env):
    sf = _real_planning_command("Library/overdue_summary", "_commands/Library/overdue_summary.py")
    assert workflow_agent.step_command_problems(sf.command, LIBRARY) == []
    assert skill_commands.check_hand_typed_steps(sf, workflow_signatures(LIBRARY)) == []

    library_proof_env.skill = sf.command
    library_proof_env.args = {}
    result = library_proof_env.ctx.process_turn("library overview")

    assert result.status == TurnStatus.COMPLETED
    assert library_proof_env.runner.sent == ["list_branches", "count_overdue"]
    assert library_proof_env.model.asked == []


def test_library_loan_report_from_real_module_runs_full_list_on_open_book(library_proof_env):
    # The ``__result__`` answer on tutorial page 14 (lines 103-108) needs Phase 3: a phase
    # annotated ``-> ListLoans.Output`` must hand on ``list_loans``'s Output (``full_list``
    # pages joined). That is asserted in Phase 3 tests; here we prove only the full_list
    # code step. The turn may still complete with phase failure text until Phase 3 lands.
    sf = _real_planning_command("Book/loan_report", "_commands/Book/loan_report.py")
    assert workflow_agent.step_command_problems(sf.command, LIBRARY) == []
    assert skill_commands.check_hand_typed_steps(sf, workflow_signatures(LIBRARY)) == []

    library_proof_env.runner(library_proof_env.ctx, "open_book book_uid=b1")
    library_proof_env.skill = sf.command
    library_proof_env.args = {}
    library_proof_env.runner.sent.clear()
    result = library_proof_env.ctx.process_turn("who has this book")

    assert result.status == TurnStatus.COMPLETED
    assert library_proof_env.model.asked == []
    assert library_proof_env.runner.sent == ["list_loans"]
    spans = library_proof_env.sink.named(tracing.SPAN_SKILL_PHASE)
    assert len(spans) == 1
    assert spans[0].attributes["code_stop"] == "done"
    assert spans[0].attributes["steps_by_code"] == 1
    assert spans[0].attributes["steps"] == "1. list_loans (full list)"


def test_library_note_member_pick_without_uid_on_test_copy(tmp_path, initialized_fastworkflow, monkeypatch):
    state = _make_library_runner_env(
        tmp_path,
        initialized_fastworkflow,
        monkeypatch,
        extra_skills={"Library/note_member": _note_member_without_in_request()},
    )
    try:
        skill_path = os.path.join(
            str(tmp_path / "library_wf"),
            "_skills",
            "Library",
            "note_member.py",
        )
        loaded = workflow_agent._read_skill_file("Library/note_member", skill_path, state.wf.folderpath)
        assert workflow_agent.step_command_problems(loaded.command, state.wf.folderpath) == []

        lib = state.wf.root_command_context
        before = list(lib.members["m2"].notes)
        state.skill = loaded.command
        state.args = {"member": "Raj Patel", "text": "card renewed"}
        result = state.ctx.process_turn("note on Raj Patel")

        assert result.status == TurnStatus.COMPLETED
        assert state.runner.sent == [
            "find_member query=Raj Patel",
            "open_member member_uid=m2",
            "add_note text=card renewed",
        ]
        assert state.model.asked == []
        assert lib.members["m2"].notes == before + ["card renewed"]
    finally:
        state.ctx.pop_active_workflow()



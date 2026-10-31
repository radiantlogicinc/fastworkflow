"""Field sources (``from_`` and ``resolve_field_sources``): exports, producer edge
resolution, error rows of 3.3, type checks, unqualified resolution and ambiguity,
hand-written ``available_from`` errors, ``Signature.Output`` / ``Signature.Input``
model map, ``check_previews`` integration, and unchanged ``fact_from`` copy-edge
behaviour."""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import uuid
from typing import Optional

from dotenv import dotenv_values
import pytest
from pydantic import BaseModel, Field

import fastworkflow
from fastworkflow import ModuleType, from_, skill_commands, skill_preview, workflow_agent
from fastworkflow.command_directory import CommandDirectory, get_cached_command_directory
from fastworkflow.command_metadata_api import CommandMetadataAPI
from fastworkflow.command_routing import RoutingRegistry
from fastworkflow.observation_offloading.facts import facts_from_runs
from fastworkflow.skill import fact_from
from fastworkflow.skill import from_ as skill_from_
from fastworkflow.skill_phases import fill_output
from fastworkflow.skill_phases import from_ as phases_from_
from fastworkflow.utils.signatures import InputForParamExtraction

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIBRARY = os.path.join(ROOT, "fastworkflow", "examples", "library")
RETAIL = os.path.join(ROOT, "fastworkflow", "examples", "retail_workflow")

_COPIED_COMMANDS = (
    "_commands/library_hours.py",
    "_commands/startup.py",
    "_commands/Library/count_overdue.py",
    "_commands/Library/find_book.py",
    "_commands/Library/find_member.py",
    "_commands/Book/show_book.py",
    "_commands/Book/list_loans.py",
    "_commands/Member/show_member.py",
    "_commands/Record/show_notes.py",
)

_COPIED_CONTEXT_CLASSES = (
    "_commands/Library/_Library.py",
    "_commands/Book/_Book.py",
    "_commands/Member/_Member.py",
    "_commands/Record/_Record.py",
)

ROOT_SKILL = '''
class Signature:
    """Say when the library's branches are open."""


class PlanGenerator:
    def steps(self):
        return ["library_hours", "say when the branches are open"]
'''


def _build_workflow(
    tmp_path,
    name: str = "wf",
    extra_commands: dict[str, str] | None = None,
    extra_skills: dict[str, str] | None = None,
    extra_inheritance: dict[str, dict] | None = None,
) -> str:
    """Build an isolated workflow folder in *tmp_path* with real ``_commands/`` and ``_commands/``."""
    folder = tmp_path / name
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy2(os.path.join(LIBRARY, "__init__.py"), folder / "__init__.py")
    shutil.copy2(os.path.join(LIBRARY, "startup_action.json"), folder / "startup_action.json")
    shutil.copytree(os.path.join(LIBRARY, "application"), folder / "application")

    with open(os.path.join(LIBRARY, "_commands", "context_inheritance_model.json"),
              encoding="utf-8") as handle:
        inheritance = json.load(handle)
    if extra_inheritance:
        inheritance.update(extra_inheritance)

    commands_dir = folder / "_commands"
    commands_dir.mkdir(parents=True, exist_ok=True)
    (commands_dir / "context_inheritance_model.json").write_text(
        json.dumps(inheritance, indent=2) + "\n", encoding="utf-8"
    )

    for rel in (*_COPIED_CONTEXT_CLASSES, *_COPIED_COMMANDS):
        src = os.path.join(LIBRARY, rel)
        dst = folder / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    if extra_commands:
        for cmd_name, code in extra_commands.items():
            cmd_path = commands_dir / f"{cmd_name}.py"
            cmd_path.parent.mkdir(parents=True, exist_ok=True)
            cmd_path.write_text(code, encoding="utf-8")

    skills_dir = folder / "_commands"
    skills_dir.mkdir(parents=True, exist_ok=True)
    skills = {"hours_overview": ROOT_SKILL, **(extra_skills or {})}
    for skill_name, code in skills.items():
        skill_path = skills_dir / f"{skill_name}.py"
        skill_path.parent.mkdir(parents=True, exist_ok=True)
        skill_path.write_text(code, encoding="utf-8")

    return str(folder)


def _preview_check(folder: str, out_dir: str, capsys) -> tuple[int, str]:
    code = skill_preview.preview_main(
        argparse.Namespace(workflow_folder=folder, out=out_dir, check=True)
    )
    return code, capsys.readouterr().out


# ------------------------------------------------------------------
# Step 1: from_ and its exports, SKILL_FIELD_KEYS, fact_from
# ------------------------------------------------------------------


def test_from_exports_and_field_construction():
    assert from_ is skill_from_
    assert from_ is phases_from_
    assert fastworkflow.from_ is from_
    assert "from" in skill_commands.SKILL_FIELD_KEYS
    assert "available_from" in skill_commands.SKILL_FIELD_KEYS

    class SampleInput(BaseModel):
        uid: str = from_(
            "find_book.book_uids",
            description="the book uid",
            json_schema_extra={"subject": True},
        )
        multi: str = from_(
            ["find_book.book_uids", "Library/find_member.member_uids"],
            default="b1",
        )

    uid_info = SampleInput.model_fields["uid"]
    assert uid_info.description == "the book uid"
    assert uid_info.is_required()
    assert uid_info.json_schema_extra == {
        "subject": True,
        "from": "find_book.book_uids",
    }

    multi_info = SampleInput.model_fields["multi"]
    assert multi_info.default == "b1"
    assert multi_info.json_schema_extra == {
        "from": ["find_book.book_uids", "Library/find_member.member_uids"],
    }


def test_from_has_no_kind_keyword():
    import inspect

    sig = inspect.signature(from_)
    assert "kind" not in sig.parameters
    assert list(sig.parameters) == ["source", "field_kwargs"]


def test_fact_from_and_copy_edges_keep_existing_behaviour(tmp_path):
    class FoundBooks(BaseModel):
        uids: list[str] = fact_from("find_book.book_uids", min_length=1)
        titles: list[str] = fact_from("find_book.titles")
        total: int = fact_from("find_book.total")

    runs = [
        {
            "command": "Library/find_book",
            "input": {"query": "Dune"},
            "output": {"book_uids": ["b1", "b2"], "titles": ["Dune", "Dune Messiah"], "total": 2},
            "success": True,
            "context_clause": "Library",
            "context_name": "Library",
        }
    ]
    filled = facts_from_runs(FoundBooks, runs)
    assert filled.uids == ["b1", "b2"]
    assert filled.titles == ["Dune", "Dune Messiah"]
    assert filled.total == 2

    # Phased skill using fact_from loads cleanly without triggering from_ error on facts model
    phased_skill = '''from pydantic import BaseModel
from fastworkflow.skill import fact_from


class Signature:
    """Find books and report their titles."""

    class Input(BaseModel):
        query: str


class PlanGenerator:
    class Found(BaseModel):
        book_uids: list[str] = fact_from("find_book.book_uids")
        titles: list[str] = fact_from("find_book.titles")

    def find(self, input) -> Found:
        return [f"find_book query={input.query}"]

    def answer(self, input, find):
        return [f"say that found {find.titles}"]
'''
    folder = _build_workflow(tmp_path, "wf_fact_from", extra_skills={"Library/find_titles": phased_skill})
    res = skill_commands.resolve_field_sources(folder)
    assert res.errors == ()
    loaded = workflow_agent.load_skill_commands(folder)
    assert "Library/find_titles" in loaded
    cmd = loaded["Library/find_titles"]
    assert len(cmd.phase_outputs) == 1
    found_model = cmd.phase_outputs[0]
    runtime_filled = fill_output(
        found_model,
        runs,
        record="Library",
        record_context="Library",
    )
    assert runtime_filled.book_uids == ["b1", "b2"]
    assert runtime_filled.titles == ["Dune", "Dune Messiah"]


# ------------------------------------------------------------------
# Step 2: Producer edge resolution (field and command-only), model map
# ------------------------------------------------------------------


def test_producer_edge_field_and_command_only_resolution_and_model_map(tmp_path, capsys):
    open_book_cmd = '''from typing import Optional
import fastworkflow
from fastworkflow import from_
from pydantic import BaseModel, Field


class Signature:
    """Open one book by its uid."""

    class Input(BaseModel):
        book_uid: str = from_(
            ["find_book.book_uids", "Book/show_book.book_uid"],
            description="the uid of the book to open",
        )
        branch_hint: Optional[str] = from_("library_hours", default=None, description="hours info")

    class Output(BaseModel):
        book_uid: str = Field(description="the uid")
        title: str = Field(description="the title")


class ResponseGenerator:
    def __call__(self, workflow: fastworkflow.Workflow, command: str,
                 command_parameters: Signature.Input) -> Signature.Output:
        return Signature.Output(book_uid=command_parameters.book_uid, title="Dune")
'''
    skill_with_from = '''from typing import Optional
from pydantic import BaseModel
from fastworkflow import from_


class Signature:
    """Inspect a book by uid."""

    class Input(BaseModel):
        book_uid: str = from_(
            ["Library/find_book.book_uids", "Book/show_book.book_uid"],
            description="the uid",
        )
        hours_note: Optional[str] = from_("library_hours", default=None, description="hours")


class PlanGenerator:
    def steps(self, input):
        return [f"open_book book_uid={input.book_uid}", "say the book"]
'''
    folder = _build_workflow(
        tmp_path,
        "wf_valid",
        extra_commands={"Library/open_book": open_book_cmd},
        extra_skills={"Library/inspect_book": skill_with_from},
    )
    res = skill_commands.resolve_field_sources(folder)
    assert res.errors == ()
    assert len(res.infos) == 2
    info_msgs = [i.message for i in res.infos]
    assert any("Input.branch_hint (from 'library_hours'): command-only producer reference" in m for m in info_msgs)
    assert any("Input.hours_note (from 'library_hours'): command-only producer reference" in m for m in info_msgs)

    # Check available_from written onto execution command and planning command Input fields
    open_sig = res.signatures["Library/open_book"]
    assert open_sig.Input.model_fields["book_uid"].json_schema_extra["available_from"] == [
        "Library/find_book.book_uids",
        "Book/show_book.book_uid",
    ]
    assert open_sig.Input.model_fields["branch_hint"].json_schema_extra["available_from"] == [
        "library_hours"
    ]

    inspect_sig = res.signatures["Library/inspect_book"]
    assert inspect_sig.Input.model_fields["book_uid"].json_schema_extra["available_from"] == [
        "Library/find_book.book_uids",
        "Book/show_book.book_uid",
    ]
    assert inspect_sig.Input.model_fields["hours_note"].json_schema_extra["available_from"] == [
        "library_hours"
    ]

    # Check 3.12 model_to_command map
    assert res.model_to_command[open_sig.Input] == "Library/open_book"
    assert res.model_to_command[open_sig.Output] == "Library/open_book"
    assert res.model_to_command[inspect_sig.Input] == "Library/inspect_book"

    # preview --check succeeds (exit 0) and prints the info line
    out_dir = str(tmp_path / "previews")
    skill_preview.write_previews(folder, out_dir)
    code, stdout = _preview_check(folder, out_dir, capsys)
    assert code == 0
    assert "info:" in stdout
    assert "Input.branch_hint (from 'library_hours')" in stdout
    assert "Input.hours_note (from 'library_hours')" in stdout


def test_planning_command_under_commands_gets_producer_missing_output_field_check(tmp_path):
    """Step 2: ``from_`` producer field checks apply to planning commands in ``_commands/``."""
    planning_cmd = '''import fastworkflow
from fastworkflow import from_
from pydantic import BaseModel, Field


class Signature:
    """Planning command referencing a missing Output field."""

    class Input(BaseModel):
        book_uid: str = from_("find_book.uids", description="the uid")


class PlanGenerator:
    def steps(self, input):
        return ["say hello"]
'''
    folder = _build_workflow(
        tmp_path,
        "wf_planning_commands_from",
        extra_commands={"Library/plan_bad_from": planning_cmd},
    )
    res = skill_commands.resolve_field_sources(folder)
    assert any(
        "Input.book_uid (from 'find_book.uids'): Library/find_book has no Output field 'uids'"
        in e.message
        for e in res.errors
    )
    # Rule changed by owner decision 2026-10-28 (fix-3rsq.3.8): from_ errors load; skipped_commands
    # lists commands with a field-source error, not commands dropped at load.
    assert "Library/plan_bad_from" in res.skipped_commands
    assert "Library/plan_bad_from" in CommandDirectory.scan(folder).map_command_2_metadata
    plan_sig = res.signatures["Library/plan_bad_from"]
    assert "available_from" not in (plan_sig.Input.model_fields["book_uid"].json_schema_extra or {})


def test_unqualified_resolution_across_own_context_inherited_base_and_global_and_qualified_other_context(tmp_path):
    # Book inherits from Record; library_hours is global; show_book is in Book;
    # Library/find_member is in another context (checked for existence and type at load)
    book_cmd = '''import fastworkflow
from fastworkflow import from_
from pydantic import BaseModel, Field


class Signature:
    """Check book details."""

    class Input(BaseModel):
        title: str = from_("show_book.title", description="from own context Book")
        note: str = from_("show_notes.notes", description="from base context Record")
        branch: str = from_("library_hours.branches", description="from global")
        member_uid: str = from_("Library/find_member.member_uids", description="qualified in another context")


class ResponseGenerator:
    def __call__(self, workflow, command, command_parameters):
        return fastworkflow.CommandOutput(
            command_response=fastworkflow.CommandResponse(response="ok")
        )
'''
    bad_qualified_cmd = '''import fastworkflow
from fastworkflow import from_
from pydantic import BaseModel


class Signature:
    """Bad qualified references in another context."""

    class Input(BaseModel):
        missing_cmd: str = from_("Library/no_such_cmd.out")
        wrong_type: int = from_("Library/find_member.member_uids")


class ResponseGenerator:
    def __call__(self, workflow, command, command_parameters):
        return fastworkflow.CommandOutput(
            command_response=fastworkflow.CommandResponse(response="ok")
        )
'''
    folder = _build_workflow(
        tmp_path,
        "wf_lookup",
        extra_commands={
            "Book/check_book": book_cmd,
            "Book/bad_qualified": bad_qualified_cmd,
        },
    )
    res = skill_commands.resolve_field_sources(folder)
    assert "Book/check_book" not in res.skipped_commands
    assert "Book/bad_qualified" in res.skipped_commands
    sig = res.signatures["Book/check_book"]
    fields = sig.Input.model_fields
    assert fields["title"].json_schema_extra["available_from"] == ["Book/show_book.title"]
    assert fields["note"].json_schema_extra["available_from"] == ["Record/show_notes.notes"]
    assert fields["branch"].json_schema_extra["available_from"] == [
        "library_hours.branches"
    ]
    assert fields["member_uid"].json_schema_extra["available_from"] == [
        "Library/find_member.member_uids"
    ]
    messages = [e.message for e in res.errors]
    assert any(
        "Input.missing_cmd (from 'Library/no_such_cmd.out'): no command 'Library/no_such_cmd'" in m
        for m in messages
    )
    assert any(
        "Input.wrong_type (from 'Library/find_member.member_uids'): int does not match "
        "Library/find_member's Output.member_uids: list[str]" in m
        for m in messages
    )


def test_unqualified_name_in_unrelated_context_and_ambiguity_errors(tmp_path):
    # 1. Book referencing Library/find_member unqualified -> not on Book's static path (Book -> Record -> global)
    # 2. Both Book and Record (base of Book) or Book and global defining the same command name -> ambiguity
    global_show_notes = '''import fastworkflow
from pydantic import BaseModel, Field


class Signature:
    """Global show_notes colliding with Record/show_notes on Book's static path."""

    class Output(BaseModel):
        notes: list[str] = Field(description="notes")


class ResponseGenerator:
    def __call__(self, workflow, command, command_parameters=None):
        return Signature.Output(notes=[])
'''
    bad_book_cmd = '''import fastworkflow
from fastworkflow import from_
from pydantic import BaseModel


class Signature:
    """Reserve book for member."""

    class Input(BaseModel):
        member_uid: str = from_("find_member.member_uids")
        note: str = from_("show_notes.notes")


class ResponseGenerator:
    def __call__(self, workflow, command, command_parameters):
        return fastworkflow.CommandOutput(
            command_response=fastworkflow.CommandResponse(response="ok")
        )
'''
    folder = _build_workflow(
        tmp_path,
        "wf_ambig",
        extra_commands={
            "show_notes": global_show_notes,
            "Book/reserve_for": bad_book_cmd,
        },
    )
    res = skill_commands.resolve_field_sources(folder)
    messages = [e.message for e in res.errors]
    assert any(
        "Input.member_uid (from 'find_member.member_uids'): no command find_member in Book, "
        "its bases or the global commands; qualify it (Library/find_member)" in m
        for m in messages
    )
    assert any(
        "Input.note (from 'show_notes.notes'): ambiguous command 'show_notes' matches "
        "Record/show_notes, show_notes; qualify it as <Context>/<command>" in m
        for m in messages
    )
    assert "Book/reserve_for" in res.skipped_commands


def test_target_field_in_both_input_and_output_resolves_to_output(tmp_path):
    # Library/open_book has book_uid in both Input (str) and Output (str)
    open_book_cmd = '''import fastworkflow
from pydantic import BaseModel, Field


class Signature:
    """Open book."""

    class Input(BaseModel):
        book_uid: int = Field(description="wrong type in Input to prove Output wins")

    class Output(BaseModel):
        book_uid: str = Field(description="str type in Output")


class ResponseGenerator:
    def __call__(self, workflow, command, command_parameters):
        return Signature.Output(book_uid="b1")
'''
    consumer_cmd = '''import fastworkflow
from fastworkflow import from_
from pydantic import BaseModel


class Signature:
    """Consumer."""

    class Input(BaseModel):
        book_uid: str = from_("open_book.book_uid")


class ResponseGenerator:
    def __call__(self, workflow, command, command_parameters):
        return fastworkflow.CommandOutput(
            command_response=fastworkflow.CommandResponse(response="ok")
        )
'''
    folder = _build_workflow(
        tmp_path,
        "wf_both_io",
        extra_commands={
            "Library/open_book": open_book_cmd,
            "Library/use_book": consumer_cmd,
        },
    )
    res = skill_commands.resolve_field_sources(folder)
    assert res.errors == ()
    sig = res.signatures["Library/use_book"]
    assert sig.Input.model_fields["book_uid"].json_schema_extra["available_from"] == [
        "Library/open_book.book_uid"
    ]


def test_target_field_only_in_input_produces_exact_error_text(tmp_path):
    bad_cmd = '''import fastworkflow
from fastworkflow import from_
from pydantic import BaseModel


class Signature:
    """Count books."""

    class Input(BaseModel):
        query: str = from_("find_book.query")


class ResponseGenerator:
    def __call__(self, workflow, command, command_parameters):
        return fastworkflow.CommandOutput(
            command_response=fastworkflow.CommandResponse(response="ok")
        )
'''
    folder = _build_workflow(
        tmp_path,
        "wf_input_only",
        extra_commands={"Library/count_books": bad_cmd},
    )
    res = skill_commands.resolve_field_sources(folder)
    assert len(res.errors) == 1
    assert res.errors[0].message == (
        "Input.query (from 'find_book.query'): find_book's query is an Input field; "
        "from_ names the Output field a value comes from. To describe the value, write "
        "Field(description=...); the preview shows find_book.query's description beside it"
    )
    cmd_file = os.path.join(folder, "_commands", "Library", "count_books.py")
    assert res.errors[0].format() == (
        f"{cmd_file}: Input.query (from 'find_book.query'): find_book's query is an Input field; "
        "from_ names the Output field a value comes from. To describe the value, write "
        "Field(description=...); the preview shows find_book.query's description beside it"
    )


def test_error_rows_output_field_and_facts_model_field_and_missing_target_field(tmp_path, caplog, capsys):
    bad_output_cmd = '''import fastworkflow
from fastworkflow import from_
from pydantic import BaseModel


class Signature:
    """Command with from_ on Output."""

    class Output(BaseModel):
        author: str = from_("show_book.author")


class ResponseGenerator:
    def __call__(self, workflow, command, command_parameters=None):
        return Signature.Output(author="Herbert")
'''
    bad_output_skill = '''from pydantic import BaseModel
from fastworkflow import from_


class Signature:
    """Skill with from_ on Output."""

    class Output(BaseModel):
        author: str = from_("Book/show_book.author")


class PlanGenerator:
    def steps(self):
        return ["library_hours"]

    def __result__(self, input):
        return Signature.Output(author="Herbert")
'''
    bad_missing_field_cmd = '''import fastworkflow
from fastworkflow import from_
from pydantic import BaseModel


class Signature:
    """Command naming nonexistent Output field."""

    class Input(BaseModel):
        book_uid: str = from_("find_book.uids")


class ResponseGenerator:
    def __call__(self, workflow, command, command_parameters):
        return fastworkflow.CommandOutput(
            command_response=fastworkflow.CommandResponse(response="ok")
        )
'''
    bad_facts_skill = '''from pydantic import BaseModel
from fastworkflow import from_


class Signature:
    """Skill with from_ on a facts model field."""

    class Input(BaseModel):
        title: str


class PlanGenerator:
    class Shelf(BaseModel):
        branch: str = from_("show_book.branch")

    def shelf(self, input) -> Shelf:
        return ["show_book"]

    def answer(self, input, shelf):
        return [f"say {shelf.branch}"]
'''
    folder = _build_workflow(
        tmp_path,
        "wf_error_rows",
        extra_commands={
            "Library/bad_output": bad_output_cmd,
            "Library/bad_missing": bad_missing_field_cmd,
        },
        extra_skills={
            "Book/bad_output_skill": bad_output_skill,
            "Book/shelf_report": bad_facts_skill,
        },
    )
    with caplog.at_level(logging.ERROR):
        loaded_skills = workflow_agent.load_skill_commands(folder)
    # Rule changed by owner decision 2026-10-28 (fix-3rsq.3.8): from_ errors load; --check fails.
    assert "Book/bad_output_skill" in loaded_skills
    assert "Book/shelf_report" in loaded_skills
    assert "skipping skill 'Book/bad_output_skill'" not in caplog.text
    assert "skipping skill 'Book/shelf_report'" not in caplog.text

    res = skill_commands.resolve_field_sources(folder)
    messages = [e.message for e in res.errors]
    output_errors = [
        e for e in res.errors
        if "Signature.Output.author: an Output field cannot declare from_; build the value in __result__"
        in e.message
    ]
    assert {e.command for e in output_errors} == {"Library/bad_output", "Book/bad_output_skill"}
    assert any(
        "Input.book_uid (from 'find_book.uids'): Library/find_book has no Output field 'uids'" in m
        for m in messages
    )
    assert any(
        "Shelf.branch (from 'show_book.branch'): from_ is only for a command's Input field; "
        "type the field with the command's Output instead" in m
        for m in messages
    )
    assert {
        "Library/bad_output",
        "Book/bad_output_skill",
        "Library/bad_missing",
        "Book/shelf_report",
    } <= res.skipped_commands

    # preview --check fails on the skill errors (unloaded skills) and on the execution command errors
    out_dir = str(tmp_path / "previews_err_rows")
    code, stdout = _preview_check(folder, out_dir, capsys)
    assert code == 1
    assert "Book/bad_output_skill" in stdout
    assert "Book/shelf_report" in stdout

    # When only execution commands have errors (skills are valid), preview --check still fails on each error
    folder_exec_errs = _build_workflow(
        tmp_path,
        "wf_exec_error_rows",
        extra_commands={
            "Library/bad_output": bad_output_cmd,
            "Library/bad_missing": bad_missing_field_cmd,
            "Library/bad_input_only": '''import fastworkflow
from fastworkflow import from_
from pydantic import BaseModel


class Signature:
    """Target field only in Input."""

    class Input(BaseModel):
        query: str = from_("find_book.query")


class ResponseGenerator:
    def __call__(self, workflow, command, command_parameters):
        return fastworkflow.CommandOutput(
            command_response=fastworkflow.CommandResponse(response="ok")
        )
''',
        },
    )
    out_exec = str(tmp_path / "previews_exec_errs")
    skill_preview.write_previews(folder_exec_errs, out_exec)
    code_exec, stdout_exec = _preview_check(folder_exec_errs, out_exec, capsys)
    assert code_exec == 1
    assert "Signature.Output.author: an Output field cannot declare from_" in stdout_exec
    assert "Library/find_book has no Output field 'uids'" in stdout_exec
    assert "find_book's query is an Input field; from_ names the Output field a value comes from" in stdout_exec


def test_type_check_passes_and_failures(tmp_path):
    producer_cmd = '''import fastworkflow
from typing import Optional
from pydantic import BaseModel, Field


class Signature:
    """Producer with various Output types."""

    class Output(BaseModel):
        exact_str: str = Field(description="str")
        opt_str: Optional[str] = Field(default=None, description="opt str")
        str_list: list[str] = Field(description="list of str")
        opt_str_list: Optional[list[str]] = Field(default=None, description="opt list of str")
        count: int = Field(description="int")


class ResponseGenerator:
    def __call__(self, workflow, command, command_parameters=None):
        return Signature.Output(exact_str="a", str_list=["a"], count=1)
'''
    valid_consumer = '''import fastworkflow
from typing import Optional
from fastworkflow import from_
from pydantic import BaseModel


class Signature:
    """Consumer with matching types."""

    class Input(BaseModel):
        a: str = from_("producer.exact_str")
        b: Optional[str] = from_("producer.exact_str")
        c: str = from_("producer.opt_str")
        d: str = from_("producer.str_list")
        e: Optional[str] = from_("producer.str_list")
        f: str = from_("producer.opt_str_list")
        g: list[str] = from_("producer.str_list")


class ResponseGenerator:
    def __call__(self, workflow, command, command_parameters):
        return fastworkflow.CommandOutput(
            command_response=fastworkflow.CommandResponse(response="ok")
        )
'''
    invalid_consumer = '''import fastworkflow
from fastworkflow import from_
from pydantic import BaseModel


class Signature:
    """Consumer with mismatched types."""

    class Input(BaseModel):
        book_uid: int = from_("find_book.book_uids")


class ResponseGenerator:
    def __call__(self, workflow, command, command_parameters):
        return fastworkflow.CommandOutput(
            command_response=fastworkflow.CommandResponse(response="ok")
        )
'''
    folder = _build_workflow(
        tmp_path,
        "wf_types",
        extra_commands={
            "Library/producer": producer_cmd,
            "Library/valid_consumer": valid_consumer,
            "Library/open_book": invalid_consumer,
        },
    )
    res = skill_commands.resolve_field_sources(folder)
    assert len(res.errors) == 1
    assert res.errors[0].command == "Library/open_book"
    assert res.errors[0].message == (
        "Input.book_uid (from 'find_book.book_uids'): int does not match "
        "find_book's Output.book_uids: list[str]"
    )
    assert "Library/valid_consumer" not in res.skipped_commands
    assert "Library/open_book" in res.skipped_commands


def test_hand_written_available_from_loads_planning_command_and_keeps_execution_command_routable(tmp_path, caplog, capsys):
    hand_cmd = '''import fastworkflow
from pydantic import BaseModel, Field


class Signature:
    """Command with hand-written available_from."""

    class Input(BaseModel):
        order_id: str = Field(json_schema_extra={"available_from": ["find_book"]})

    plain_utterances = ["get order {order_id}"]


class ResponseGenerator:
    def __call__(self, workflow, command, command_parameters):
        return fastworkflow.CommandOutput(
            command_response=fastworkflow.CommandResponse(response="ok")
        )
'''
    hand_skill = '''from pydantic import BaseModel, Field


class Signature:
    """Skill with hand-written available_from."""

    class Input(BaseModel):
        book_uid: str = Field(json_schema_extra={"available_from": ["find_book"]})


class PlanGenerator:
    def steps(self, input):
        return [f"find_book query={input.book_uid}"]
'''
    folder = _build_workflow(
        tmp_path,
        "wf_hand_af",
        extra_commands={"Library/get_order": hand_cmd},
        extra_skills={"Library/hand_skill": hand_skill},
    )
    with caplog.at_level(logging.ERROR):
        loaded_skills = workflow_agent.load_skill_commands(folder)
    # Rule changed by owner decision 2026-10-28 (fix-3rsq.3.8): planning commands with from_ /
    # available_from errors stay matchable; the error is logged once, not via "skipping skill".
    assert "Library/hand_skill" in loaded_skills
    assert "skipping skill 'Library/hand_skill'" not in caplog.text
    hand_skill_logs = [
        rec for rec in caplog.records
        if "command 'Library/hand_skill':" in rec.getMessage()
    ]
    assert len(hand_skill_logs) == 1
    assert "available_from is written by from_" in hand_skill_logs[0].getMessage()

    res = skill_commands.resolve_field_sources(folder)
    assert {"Library/get_order", "Library/hand_skill"} <= res.skipped_commands
    hand_sig = res.signatures["Library/hand_skill"]
    assert "available_from" not in (hand_sig.Input.model_fields["book_uid"].json_schema_extra or {})
    messages = [e.message for e in res.errors]
    assert any(
        'Input.order_id: available_from is written by from_; write from_("find_book") '
        'or from_("find_book.<field>")' in m
        for m in messages
    )

    # Execution command policy: when only an execution command has hand-written available_from,
    # hydration logs the error once, keeps the command routable, and check_previews fails.
    folder_exec_only = _build_workflow(
        tmp_path,
        "wf_hand_exec_only",
        extra_commands={"Library/get_order": hand_cmd},
    )
    env_vars = dict(dotenv_values(os.path.join(ROOT, "fastworkflow", "examples", "fastworkflow.env")))
    fastworkflow.init(env_vars=env_vars)
    RoutingRegistry.clear_registry()
    caplog.clear()
    with caplog.at_level(logging.ERROR):
        routing_def = RoutingRegistry.get_definition(folder_exec_only, load_cached=False)
        for _ in range(2):
            routing_def.command_directory.ensure_command_hydrated("Library/get_order")
    get_order_logs = [
        rec for rec in caplog.records
        if "command 'Library/get_order':" in rec.getMessage()
    ]
    assert len(get_order_logs) == 1
    assert "available_from is written by from_" in get_order_logs[0].getMessage()
    assert "Library/get_order" in routing_def.get_command_names("Library")
    param_cls = routing_def.get_command_class("Library/get_order", ModuleType.COMMAND_PARAMETERS_CLASS)
    assert param_cls is not None

    out_dir = str(tmp_path / "previews_exec")
    skill_preview.write_previews(folder_exec_only, out_dir)
    problems = skill_preview.check_previews(folder_exec_only, out_dir)
    assert any("available_from is written by from_" in p for p in problems)
    code, stdout = _preview_check(folder_exec_only, out_dir, capsys)
    assert code == 1
    assert "available_from is written by from_" in stdout

    out_dir_both = str(tmp_path / "previews_both")
    skill_preview.write_previews(folder, out_dir_both)
    problems_both = skill_preview.check_previews(folder, out_dir_both)
    assert any("available_from is written by from_" in p for p in problems_both)
    code_both, stdout_both = _preview_check(folder, out_dir_both, capsys)
    assert code_both == 1
    assert "available_from is written by from_" in stdout_both


def test_planning_command_with_from_error_loads_matchable_logs_once_and_preview_check_fails(
    tmp_path, caplog, capsys,
):
    """Planning skill with a bad ``from_`` stays in the registry and ``load_skill_commands``."""
    bad_planning = '''import fastworkflow
from fastworkflow import from_
from pydantic import BaseModel


class Signature:
    """Planning command with invalid from_ reference."""

    class Input(BaseModel):
        book_uid: int = from_("find_book.book_uids")


class PlanGenerator:
    def steps(self, input):
        return [f"say {input.book_uid}"]
'''
    folder = _build_workflow(
        tmp_path,
        "wf_plan_from_err",
        extra_skills={"Library/bad_plan": bad_planning},
    )
    caplog.clear()
    with caplog.at_level(logging.ERROR):
        loaded = workflow_agent.load_skill_commands(folder)
    assert "Library/bad_plan" in loaded
    assert "skipping skill 'Library/bad_plan'" not in caplog.text
    plan_logs = [
        rec for rec in caplog.records if "command 'Library/bad_plan':" in rec.getMessage()
    ]
    assert len(plan_logs) == 1
    assert "int does not match" in plan_logs[0].getMessage()

    res = skill_commands.resolve_field_sources(folder)
    bad_sig = res.signatures["Library/bad_plan"]
    assert "available_from" not in (bad_sig.Input.model_fields["book_uid"].json_schema_extra or {})

    out_dir = str(tmp_path / "previews_plan")
    skill_preview.write_previews(folder, out_dir)
    problems = skill_preview.check_previews(folder, out_dir)
    assert any("int does not match" in p for p in problems)
    code, stdout = _preview_check(folder, out_dir, capsys)
    assert code == 1
    assert "int does not match" in stdout


def test_a_from_naming_a_callee_of_its_own_steps_is_an_error_that_asks_for_a_description(tmp_path):
    """fix-3rsq.5.7: a producer the command's steps run is run by the steps, so the agent
    cannot be told to run it first; the Input must carry a description. A producer the
    steps do not call stays a valid ``from_``."""
    calls_producer = '''import fastworkflow
from fastworkflow import from_
from pydantic import BaseModel, Field


class Signature:
    """Say how many books a title matches, from the count its find gives."""

    class Input(BaseModel):
        title: str = Field(description="the book's title")
        total: int = from_("find_book.total", description="how many books match")


class PlanGenerator:
    def steps(self, input):
        return [f"find_book query={input.title}", "say the count"]
'''
    described = calls_producer.replace(
        'from_("find_book.total", description="how many books match")',
        'Field(description="how many books match")')
    runs_other_producer = calls_producer.replace(
        'from_("find_book.total", description="how many books match")',
        'from_("find_member.total", description="how many members match")')
    folder = _build_workflow(
        tmp_path,
        "wf_callee_from",
        extra_skills={"Library/plan_calls_find": calls_producer,
                      "Library/plan_described": described,
                      "Library/plan_other_producer": runs_other_producer},
    )
    res = skill_commands.resolve_field_sources(folder)
    assert [p.message for p in res.errors if p.command == "Library/plan_described"] == []
    assert [p.message for p in res.errors if p.command == "Library/plan_other_producer"] == []
    assert [p.message for p in res.errors if p.command == "Library/plan_calls_find"] == [
        "Input.total (from 'find_book.total') tells the agent to run Library/find_book "
        "first, but this command's own steps call Library/find_book. To describe the value, "
        "write Field(description=...) instead"]


def test_library_planning_commands_resolve_without_source_errors_and_with_their_callees():
    """The real library: no field-source error on any planning command (the phase checker
    reads an earlier phase's steps on placeholder facts, not as empty), and the callee graph
    of book_loans, which calls a planning command and an execution command."""
    res = skill_commands.resolve_field_sources(LIBRARY)
    assert [p.format() for p in res.errors] == []
    assert res.callees["Library/book_loans"] == frozenset({"Book/list_loans", "Library/book_by_title"})
    assert skill_commands.call_depth("Library/book_loans", res) == 2


def test_target_command_whose_file_fails_to_import_is_reported(tmp_path):
    broken_target = '''import non_existent_module_xyz
'''
    consumer = '''import fastworkflow
from fastworkflow import from_
from pydantic import BaseModel


class Signature:
    """Consumer pointing to a broken command file."""

    class Input(BaseModel):
        val: str = from_("broken_cmd.out")


class ResponseGenerator:
    def __call__(self, workflow, command, command_parameters):
        return fastworkflow.CommandOutput(
            command_response=fastworkflow.CommandResponse(response="ok")
        )
'''
    folder = _build_workflow(
        tmp_path,
        "wf_broken_import",
        extra_commands={
            "Library/broken_cmd": broken_target,
            "Library/consumer": consumer,
        },
    )
    res = skill_commands.resolve_field_sources(folder)
    assert "Library/broken_cmd" in res.import_errors
    assert len(res.errors) == 1
    assert "target command 'Library/broken_cmd' could not be imported" in res.errors[0].message
    assert "No module named 'non_existent_module_xyz'" in res.errors[0].message


def test_hand_written_available_from_alongside_from_is_reported_and_rerun_is_idempotent(tmp_path):
    both_cmd = '''import fastworkflow
from fastworkflow import from_
from pydantic import BaseModel


class Signature:
    """Command with both from_ and a hand-written available_from."""

    class Input(BaseModel):
        book_uid: str = from_(
            "find_book.book_uids",
            json_schema_extra={"available_from": ["find_book"]},
        )


class ResponseGenerator:
    def __call__(self, workflow, command, command_parameters):
        return fastworkflow.CommandOutput(
            command_response=fastworkflow.CommandResponse(response="ok")
        )
'''
    valid_cmd = '''import fastworkflow
from fastworkflow import from_
from pydantic import BaseModel


class Signature:
    """Command with valid from_."""

    class Input(BaseModel):
        book_uid: str = from_("find_book.book_uids")


class ResponseGenerator:
    def __call__(self, workflow, command, command_parameters):
        return fastworkflow.CommandOutput(
            command_response=fastworkflow.CommandResponse(response="ok")
        )
'''
    folder = _build_workflow(
        tmp_path,
        "wf_both_af_and_from",
        extra_commands={
            "Library/both_cmd": both_cmd,
            "Library/valid_cmd": valid_cmd,
        },
    )
    res = skill_commands.resolve_field_sources(folder)
    assert "Library/both_cmd" in res.skipped_commands
    assert "Library/valid_cmd" not in res.skipped_commands
    assert len(res.errors) == 1
    assert res.errors[0].command == "Library/both_cmd"
    assert (
        'Input.book_uid: available_from is written by from_; write from_("find_book") '
        'or from_("find_book.<field>")'
    ) in res.errors[0].message

    # Re-running resolve_signature_field_sources on an already-resolved Signature is idempotent
    valid_sig = res.signatures["Library/valid_cmd"]
    valid_path = res.command_files["Library/valid_cmd"]
    rerun_problems = skill_commands.resolve_signature_field_sources(
        "Library/valid_cmd", valid_sig, valid_path, res
    )
    assert rerun_problems == []
    assert valid_sig.Input.model_fields["book_uid"].json_schema_extra["available_from"] == [
        "Library/find_book.book_uids"
    ]


def test_broken_context_or_workflow_inheritance_models_are_reported_as_problems(tmp_path):
    folder = _build_workflow(tmp_path, "wf_broken_models")
    ctx_model_path = os.path.join(folder, "_commands", "context_inheritance_model.json")
    inh_model_path = os.path.join(folder, "workflow_inheritance_model.json")
    with open(ctx_model_path, "w", encoding="utf-8") as fh:
        fh.write("{not valid json")
    with open(inh_model_path, "w", encoding="utf-8") as fh:
        fh.write('{"base": ["nonexistent_base_workflow_pkg_xyz"]}')

    res = skill_commands.resolve_field_sources(folder)
    files_with_errors = {e.file for e in res.errors}
    assert ctx_model_path in files_with_errors
    assert inh_model_path in files_with_errors

    # When only workflow_inheritance_model.json is broken (context_inheritance_model.json is valid),
    # CommandContextModel.load_base_contexts raises WorkflowInheritanceModelValidationError and
    # resolve_field_sources reports the error on workflow_inheritance_model.json (not context_inheritance_model.json).
    folder_inh_only = _build_workflow(tmp_path, "wf_broken_inh_only")
    inh_only_path = os.path.join(folder_inh_only, "workflow_inheritance_model.json")
    with open(inh_only_path, "w", encoding="utf-8") as fh:
        fh.write('{"base": ["nonexistent_base_workflow_pkg_xyz"]}')
    with pytest.raises(fastworkflow.workflow_inheritance_model.WorkflowInheritanceModelValidationError):
        fastworkflow.command_context_model.CommandContextModel.load_base_contexts(folder_inh_only)
    res_inh_only = skill_commands.resolve_field_sources(folder_inh_only)
    assert {e.file for e in res_inh_only.errors} == {inh_only_path}

    # Inheritance cycle and unknown base in context_inheritance_model.json are also reported
    folder_cycle = _build_workflow(
        tmp_path,
        "wf_cycle_model",
        extra_inheritance={
            "Book": {"base": ["Member"]},
            "Member": {"base": ["Book"]},
        },
    )
    res_cycle = skill_commands.resolve_field_sources(folder_cycle)
    assert any(
        e.file == os.path.join(folder_cycle, "_commands", "context_inheritance_model.json")
        and "Inheritance cycle detected" in e.message
        for e in res_cycle.errors
    )

    folder_unknown_base = _build_workflow(
        tmp_path,
        "wf_unknown_base_model",
        extra_inheritance={"Book": {"base": ["NoSuchBaseContext"]}},
    )
    res_unknown = skill_commands.resolve_field_sources(folder_unknown_base)
    assert any(
        e.file == os.path.join(folder_unknown_base, "_commands", "context_inheritance_model.json")
        and "Context 'NoSuchBaseContext' not found in model" in e.message
        for e in res_unknown.errors
    )


def test_base_contexts_preserved_when_another_command_fails_to_import(tmp_path):
    broken_cmd = '''import non_existent_planning_module_xyz
'''
    book_cmd = '''import fastworkflow
from fastworkflow import from_
from pydantic import BaseModel


class Signature:
    """Book command referencing inherited Record/show_notes unqualified."""

    class Input(BaseModel):
        note: str = from_("show_notes.notes", description="from base context Record")


class ResponseGenerator:
    def __call__(self, workflow, command, command_parameters):
        return fastworkflow.CommandOutput(
            command_response=fastworkflow.CommandResponse(response="ok")
        )
'''
    folder = _build_workflow(
        tmp_path,
        "wf_base_with_import_error",
        extra_commands={
            "Library/broken_planning_cmd": broken_cmd,
            "Book/inspect_notes": book_cmd,
        },
    )
    res = skill_commands.resolve_field_sources(folder)
    assert "Library/broken_planning_cmd" in res.import_errors
    assert "No module named 'non_existent_planning_module_xyz'" in res.import_errors["Library/broken_planning_cmd"]
    assert res.base_contexts["Book"] == ("Record",)
    assert res.base_contexts["Member"] == ("Record",)
    assert res.errors == ()
    sig = res.signatures["Book/inspect_notes"]
    assert sig.Input.model_fields["note"].json_schema_extra["available_from"] == [
        "Record/show_notes.notes"
    ]


def test_resolve_field_sources_does_not_create_command_info_or_write_to_workflow_folder(tmp_path):
    open_book_cmd = '''import fastworkflow
from fastworkflow import from_
from pydantic import BaseModel


class Signature:
    """Open book."""

    class Input(BaseModel):
        book_uid: str = from_("find_book.book_uids")
        note: str = from_("Record/show_notes.notes")


class ResponseGenerator:
    def __call__(self, workflow, command, command_parameters):
        return fastworkflow.CommandOutput(
            command_response=fastworkflow.CommandResponse(response="ok")
        )
'''
    folder = _build_workflow(
        tmp_path,
        "wf_no_writes",
        extra_commands={"Library/open_book": open_book_cmd},
    )
    assert not os.path.exists(os.path.join(folder, "___command_info"))

    def snapshot_non_pycache(root: str) -> set[str]:
        found: set[str] = set()
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d != "__pycache__"]
            rel_dir = os.path.relpath(dirpath, root)
            if rel_dir != ".":
                found.add(rel_dir)
            for fn in filenames:
                if not fn.endswith(".pyc"):
                    found.add(os.path.normpath(os.path.join(rel_dir, fn)))
        return found

    before = snapshot_non_pycache(folder)
    res = skill_commands.resolve_field_sources(folder)
    after = snapshot_non_pycache(folder)

    assert res.errors == ()
    assert not os.path.exists(os.path.join(folder, "___command_info"))
    assert after == before


# ------------------------------------------------------------------
# Step 3: Hydration of execution commands, caching, and snapshots
# ------------------------------------------------------------------


def test_hydrating_execution_command_resolves_from_before_command_metadata_api(tmp_path):
    open_book_cmd = '''from typing import Optional
import fastworkflow
from fastworkflow import from_
from pydantic import BaseModel, Field


class Signature:
    """Open one book by its uid."""

    class Input(BaseModel):
        book_uid: str = from_(
            ["find_book.book_uids", "Book/show_book.book_uid"],
            description="the uid of the book to open",
        )
        branch_hint: Optional[str] = from_("library_hours", default=None, description="hours info")

    class Output(BaseModel):
        book_uid: str = Field(description="the uid")
        title: str = Field(description="the title")

    plain_utterances = ["open book {book_uid}"]


class ResponseGenerator:
    def __call__(self, workflow: fastworkflow.Workflow, command: str,
                 command_parameters: Signature.Input) -> Signature.Output:
        return Signature.Output(book_uid=command_parameters.book_uid, title="Dune")
'''
    folder = _build_workflow(
        tmp_path,
        "wf_hydrate",
        extra_commands={"Library/open_book": open_book_cmd},
    )
    env_vars = dict(dotenv_values(os.path.join(ROOT, "fastworkflow", "examples", "fastworkflow.env")))
    fastworkflow.init(env_vars=env_vars)
    RoutingRegistry.clear_registry()
    cme_path = fastworkflow.get_internal_workflow_path("command_metadata_extraction")

    # Without calling resolve_field_sources directly first, query CommandMetadataAPI
    info = CommandMetadataAPI.get_enhanced_command_info(
        subject_workflow_path=folder,
        cme_workflow_path=cme_path,
        active_context_name="Library",
    )
    open_book_meta = next(
        (c for c in info["commands"] if c["qualified_name"] == "Library/open_book"),
        None,
    )
    assert open_book_meta is not None
    inputs_by_name = {inp["name"]: inp for inp in open_book_meta["inputs"]}
    assert inputs_by_name["book_uid"]["available_from"] == (
        "['Library/find_book.book_uids', 'Book/show_book.book_uid']"
    )
    assert inputs_by_name["branch_hint"]["available_from"] == "['library_hours']"

    # Also check display text used by the agent and planner
    display_text = CommandMetadataAPI.get_command_display_text(
        subject_workflow_path=folder,
        cme_workflow_path=cme_path,
        active_context_name="Library",
        for_agents=True,
    )
    assert "available_from: ['Library/find_book.book_uids', 'Book/show_book.book_uid']" in display_text
    assert "available_from: ['library_hours']" in display_text

    # Single-command hydration without building full RoutingDefinition first (no import-order dependence)
    folder_single = _build_workflow(
        tmp_path,
        "wf_hydrate_single",
        extra_commands={"Library/open_book": open_book_cmd},
    )
    RoutingRegistry.clear_registry()
    cmd_dir = get_cached_command_directory(folder_single)
    cmd_dir.ensure_command_hydrated("Library/open_book")
    cmd_meta = cmd_dir.get_command_metadata("Library/open_book")
    assert cmd_meta.parameter_extraction_signature_module_path is not None
    mod = fastworkflow.utils.python_utils.get_module(
        cmd_meta.parameter_extraction_signature_module_path,
        cmd_meta.workflow_folderpath,
    )
    assert mod.Signature.Input.model_fields["book_uid"].json_schema_extra["available_from"] == [
        "Library/find_book.book_uids",
        "Book/show_book.book_uid",
    ]


def test_field_source_resolution_is_cached_per_workflow(tmp_path):
    open_book_cmd = '''import fastworkflow
from fastworkflow import from_
from pydantic import BaseModel, Field


class Signature:
    """Open one book by its uid."""

    class Input(BaseModel):
        book_uid: str = from_("find_book.book_uids", description="the uid")

    plain_utterances = ["open book"]


class ResponseGenerator:
    def __call__(self, workflow, command, command_parameters):
        return fastworkflow.CommandOutput(
            command_response=fastworkflow.CommandResponse(response="ok")
        )
'''
    folder = _build_workflow(
        tmp_path,
        "wf_cache",
        extra_commands={"Library/open_book": open_book_cmd},
    )
    RoutingRegistry.clear_registry()
    info_before = skill_commands.resolve_field_sources.cache_info()

    cmd_dir = CommandDirectory.load(folder)
    # Hydrate multiple commands; resolve_field_sources must run only once for the workflow
    for cmd_name in cmd_dir.get_commands():
        cmd_dir.ensure_command_hydrated(cmd_name)

    info_after = skill_commands.resolve_field_sources.cache_info()
    assert info_after.misses - info_before.misses == 1
    assert info_after.hits > info_before.hits

    res1 = skill_commands.resolve_field_sources(os.path.abspath(folder))
    res2 = skill_commands.resolve_field_sources(os.path.abspath(folder))
    assert res1 is res2


def test_execution_command_with_from_error_logs_error_once_leaves_field_unwritten_and_stays_routable(tmp_path, caplog):
    bad_exec_cmd = '''import fastworkflow
from fastworkflow import from_
from pydantic import BaseModel


class Signature:
    """Command with invalid from_ reference."""

    class Input(BaseModel):
        book_uid: int = from_("find_book.book_uids")

    plain_utterances = ["bad open book"]


class ResponseGenerator:
    def __call__(self, workflow, command, command_parameters):
        return fastworkflow.CommandOutput(
            command_response=fastworkflow.CommandResponse(response="ok")
        )
'''
    folder = _build_workflow(
        tmp_path,
        "wf_bad_exec_hydrate",
        extra_commands={"Library/bad_open": bad_exec_cmd},
    )
    env_vars = dict(dotenv_values(os.path.join(ROOT, "fastworkflow", "examples", "fastworkflow.env")))
    fastworkflow.init(env_vars=env_vars)
    RoutingRegistry.clear_registry()
    cme_path = fastworkflow.get_internal_workflow_path("command_metadata_extraction")

    with caplog.at_level(logging.ERROR):
        routing_def = RoutingRegistry.get_definition(folder, load_cached=False)
        info = CommandMetadataAPI.get_enhanced_command_info(
            subject_workflow_path=folder,
            cme_workflow_path=cme_path,
            active_context_name="Library",
        )

    assert "command 'Library/bad_open':" in caplog.text
    assert "skipping command 'Library/bad_open'" not in caplog.text
    assert "Input.book_uid (from 'find_book.book_uids'): int does not match" in caplog.text
    # Error for Library/bad_open is logged exactly once even when ensure_command_hydrated
    # is called repeatedly for the same Signature class
    bad_open_logs = [
        rec for rec in caplog.records
        if "command 'Library/bad_open':" in rec.getMessage()
    ]
    assert len(bad_open_logs) == 1
    for _ in range(3):
        routing_def.command_directory.ensure_command_hydrated("Library/bad_open")
    assert len([
        rec for rec in caplog.records
        if "command 'Library/bad_open':" in rec.getMessage()
    ]) == 1

    # Clearing resolve_field_sources cache also clears the per-command Signature cache
    skill_commands.resolve_field_sources.cache_clear()
    with caplog.at_level(logging.ERROR):
        routing_def.command_directory.ensure_command_hydrated("Library/bad_open")
    assert len([
        rec for rec in caplog.records
        if "command 'Library/bad_open':" in rec.getMessage()
    ]) == 2

    # Command stays in routing (preserving trained intent model label sets),
    # but the broken from_ edge is not written to available_from
    param_cls = routing_def.get_command_class(
        "Library/bad_open", ModuleType.COMMAND_PARAMETERS_CLASS
    )
    assert param_cls is not None
    assert "available_from" not in (param_cls.model_fields["book_uid"].json_schema_extra or {})
    bad_meta = next(c for c in info["commands"] if c["qualified_name"] == "Library/bad_open")
    assert bad_meta["inputs"][0]["available_from"] is None


def test_snapshot_reloaded_in_fresh_subprocess_preserves_resolved_available_from(tmp_path):
    open_book_cmd = '''import fastworkflow
from fastworkflow import from_
from pydantic import BaseModel, Field


class Signature:
    """Open one book by its uid."""

    class Input(BaseModel):
        book_uid: str = from_(
            ["find_book.book_uids", "Book/show_book.book_uid"],
            description="the uid of the book to open",
        )

    plain_utterances = ["open book {book_uid}"]


class ResponseGenerator:
    def __call__(self, workflow, command, command_parameters):
        return fastworkflow.CommandOutput(
            command_response=fastworkflow.CommandResponse(response="ok")
        )
'''
    folder = _build_workflow(
        tmp_path,
        "wf_subproc",
        extra_commands={"Library/open_book": open_book_cmd},
    )
    env_vars = dict(dotenv_values(os.path.join(ROOT, "fastworkflow", "examples", "fastworkflow.env")))
    fastworkflow.init(env_vars=env_vars)
    RoutingRegistry.clear_registry()
    # Build and persist command_directory.json and routing_definition.json in this process
    routing_def = RoutingRegistry.get_definition(folder, load_cached=False)
    assert routing_def is not None

    cmd_dir_json = os.path.join(folder, "___command_info", "command_directory.json")
    routing_json = os.path.join(folder, "___command_info", "routing_definition.json")
    assert os.path.isfile(cmd_dir_json)
    assert os.path.isfile(routing_json)

    # Verify available_from is not serialized into the JSON snapshots (avoiding stale edge leaks)
    with open(cmd_dir_json, encoding="utf-8") as fh:
        assert "available_from" not in fh.read()
    with open(routing_json, encoding="utf-8") as fh:
        assert "available_from" not in fh.read()

    # Reload from the persisted snapshots in a brand-new Python process and inspect via CommandMetadataAPI
    script = f"""
import json
import fastworkflow
from fastworkflow.command_metadata_api import CommandMetadataAPI
from fastworkflow.command_routing import RoutingRegistry

fastworkflow.init({{}})
folder = {folder!r}
assert RoutingRegistry._persisted_definition_is_fresh(folder)
cme_path = fastworkflow.get_internal_workflow_path("command_metadata_extraction")
info = CommandMetadataAPI.get_enhanced_command_info(
    subject_workflow_path=folder,
    cme_workflow_path=cme_path,
    active_context_name="Library",
)
cmd = next(c for c in info["commands"] if c["qualified_name"] == "Library/open_book")
print("RESULT_JSON:" + json.dumps(cmd["inputs"]))
"""
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    marker_line = next(
        line for line in proc.stdout.splitlines() if line.startswith("RESULT_JSON:")
    )
    inputs = json.loads(marker_line[len("RESULT_JSON:"):])
    assert len(inputs) == 1
    assert inputs[0]["name"] == "book_uid"
    assert inputs[0]["available_from"] == (
        "['Library/find_book.book_uids', 'Book/show_book.book_uid']"
    )


# ------------------------------------------------------------------
# Step 4: Missing-parameter message and metadata rendering
# ------------------------------------------------------------------


def test_missing_parameter_message_strips_context_prefix_and_field_suffix(tmp_path):
    open_book_cmd = '''from typing import Optional
import fastworkflow
from fastworkflow import from_
from pydantic import BaseModel, Field

NOT_FOUND = fastworkflow.get_env_var("NOT_FOUND")


class Signature:
    """Open one book by its uid."""

    class Input(BaseModel):
        book_uid: str = from_(
            ["find_book.book_uids", "Book/show_book.book_uid", "library_hours"],
            default=NOT_FOUND,
            description="the uid of the book to open",
        )

    plain_utterances = ["open book {book_uid}"]


class ResponseGenerator:
    def __call__(self, workflow, command, command_parameters):
        return fastworkflow.CommandOutput(
            command_response=fastworkflow.CommandResponse(response="ok")
        )
'''
    folder = _build_workflow(
        tmp_path,
        "wf_missing_msg",
        extra_commands={"Library/open_book": open_book_cmd},
    )
    env_vars = dict(dotenv_values(os.path.join(ROOT, "fastworkflow", "examples", "fastworkflow.env")))
    fastworkflow.init(env_vars=env_vars)
    fastworkflow.RoutingRegistry.clear_registry()

    routing_def = fastworkflow.RoutingRegistry.get_definition(folder, load_cached=False)
    workflow = fastworkflow.Workflow.create(
        folder, workflow_id_str=f"test-missing-msg-{uuid.uuid4()}"
    )

    param_cls = routing_def.get_command_class(
        "Library/open_book", ModuleType.COMMAND_PARAMETERS_CLASS
    )
    assert param_cls is not None
    # Metadata carries the resolved qualified names and .field suffixes
    assert param_cls.model_fields["book_uid"].json_schema_extra["available_from"] == [
        "Library/find_book.book_uids",
        "Book/show_book.book_uid",
        "library_hours",
    ]

    params = param_cls()
    extractor = InputForParamExtraction.create(
        workflow, "Library/open_book", "Library/open_book"
    )
    is_valid, message, _, _ = extractor.validate_parameters(
        workflow, "Library/open_book", params
    )
    assert is_valid is False
    # Missing-parameter message lists bare command names only (no Context/ prefix, no .field suffix)
    assert (
        "abort and use the find_book or show_book or library_hours command(s) "
        "to get book_uid information. OR..."
    ) in message


def test_retail_missing_parameter_messages_are_byte_identical_to_baseline():
    fixture_path = os.path.join(ROOT, "tests", "fixtures", "retail_missing_parameter_messages.json")
    with open(fixture_path, encoding="utf-8") as fh:
        expected_messages: dict[str, str] = json.load(fh)

    env_vars = dict(dotenv_values(os.path.join(ROOT, "fastworkflow", "examples", "fastworkflow.env")))
    fastworkflow.init(env_vars=env_vars)
    fastworkflow.RoutingRegistry.clear_registry()

    routing_def = fastworkflow.RoutingRegistry.get_definition(RETAIL)
    workflow = fastworkflow.Workflow.create(
        RETAIL, workflow_id_str=f"retail-baseline-{uuid.uuid4()}"
    )

    assert len(expected_messages) == 10
    for cmd, expected in expected_messages.items():
        param_cls = routing_def.get_command_class(cmd, ModuleType.COMMAND_PARAMETERS_CLASS)
        assert param_cls is not None, f"Missing parameter class for {cmd}"
        params = param_cls()
        extractor = InputForParamExtraction.create(workflow, cmd, cmd)
        is_valid, message, _, _ = extractor.validate_parameters(workflow, cmd, params)
        assert is_valid is False
        assert message == expected, f"Mismatch for {cmd}:\nActual:\n{message!r}\nExpected:\n{expected!r}"



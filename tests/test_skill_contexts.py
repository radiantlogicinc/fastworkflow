"""Skill contexts (``_skills/<Context>/<name>.py``): loading, unknown-context and
command-name collision load errors, preview ``--check`` reporting, ``skill_scopes``
lookup paths, and scope-by-scope Jev matching."""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import pytest
from dotenv import dotenv_values

import fastworkflow
from fastworkflow import skill_commands, skill_preview, workflow_agent
from fastworkflow.observation_offloading.facts import workflow_signatures
from fastworkflow.utils import python_utils

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIBRARY = os.path.join(ROOT, "fastworkflow", "examples", "library")

_COPIED_COMMANDS = (
    "_commands/library_hours.py",
    "_commands/startup.py",
    "_commands/Library/count_overdue.py",
    "_commands/Library/find_book.py",
    "_commands/Library/find_member.py",
    "_commands/Library/open_book.py",
    "_commands/Library/open_member.py",
    "_commands/Book/show_book.py",
    "_commands/Member/show_member.py",
    "_commands/Record/show_notes.py",
)

_COPIED_CONTEXT_CLASSES = (
    "_commands/Library/_Library.py",
    "_commands/Book/_Book.py",
    "_commands/Member/_Member.py",
    "_commands/Record/_Record.py",
)

BRANCH_APP_ADDITION = """

class Branch(Library):
    \"\"\"One branch of a library: inherits ``Library`` and holds the parent ``Library``.\"\"\"

    def __init__(self, library: Library, name: str = "Central") -> None:
        super().__init__()
        self.library = library
        self.name = name
"""

BRANCH_CONTEXT = """from ...application.library import Branch, Library


class Context:
    @classmethod
    def get_parent(cls, command_context_object: Branch) -> Library:
        return command_context_object.library

    @classmethod
    def get_displayname(cls, command_context_object: Branch) -> str:
        return f"Branch: {command_context_object.name}"
"""

BRANCH_COMMAND = """import fastworkflow
from pydantic import BaseModel, Field


class Signature:
    \"\"\"Show the open branch's name.\"\"\"

    class Output(BaseModel):
        branch: str = Field(description="the branch name")


class ResponseGenerator:
    def __call__(self, workflow: fastworkflow.Workflow, command: str,
                 command_parameters: None) -> Signature.Output:
        return Signature.Output(branch="Central")
"""

ROOT_SKILL = '''
class Signature:
    """Say when the library's branches are open."""


class PlanGenerator:
    def steps(self):
        return ["library_hours", "say when the branches are open"]
'''

LIBRARY_SKILL = '''
class Signature:
    """Look up Dune in the library and report how many books match."""


class PlanGenerator:
    def steps(self):
        return ["find_book query=Dune", "say how many books match"]
'''


def _library_workflow(tmp_path, extra_skills: dict[str, str] | None = None,
                      default_skills: bool = True) -> str:
    """Build a library test workflow in *tmp_path* with root and ``Library`` skills."""
    folder = tmp_path / "library_wf"
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy2(os.path.join(LIBRARY, "__init__.py"), folder / "__init__.py")
    shutil.copy2(os.path.join(LIBRARY, "startup_action.json"), folder / "startup_action.json")
    shutil.copytree(os.path.join(LIBRARY, "application"), folder / "application")
    app_lib = folder / "application" / "library.py"
    app_lib.write_text(app_lib.read_text(encoding="utf-8") + BRANCH_APP_ADDITION, encoding="utf-8")

    with open(os.path.join(LIBRARY, "_commands", "context_inheritance_model.json"),
              encoding="utf-8") as handle:
        inheritance = json.load(handle)
    inheritance["Branch"] = {"base": ["Library"]}
    commands_dir = folder / "_commands"
    commands_dir.mkdir(parents=True, exist_ok=True)
    (commands_dir / "context_inheritance_model.json").write_text(
        json.dumps(inheritance, indent=2) + "\n", encoding="utf-8")

    for rel in (*_COPIED_CONTEXT_CLASSES, *_COPIED_COMMANDS):
        src = os.path.join(LIBRARY, rel)
        dst = folder / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    branch_dir = commands_dir / "Branch"
    branch_dir.mkdir(parents=True, exist_ok=True)
    (branch_dir / "_Branch.py").write_text(BRANCH_CONTEXT, encoding="utf-8")
    (branch_dir / "branch_name.py").write_text(BRANCH_COMMAND, encoding="utf-8")

    skills: dict[str, str] = {}
    if default_skills:
        skills["hours_overview"] = ROOT_SKILL
        skills["Library/dune_search"] = LIBRARY_SKILL
    if extra_skills:
        skills.update(extra_skills)

    skills_dir = folder / "_skills"
    skills_dir.mkdir(parents=True, exist_ok=True)
    for name, text in skills.items():
        skill_path = skills_dir / f"{name}.py"
        skill_path.parent.mkdir(parents=True, exist_ok=True)
        skill_path.write_text(text, encoding="utf-8")

    return str(folder)


def _preview_check(folder: str, out_dir: str, capsys) -> tuple[int, str]:
    code = skill_preview.preview_main(
        argparse.Namespace(workflow_folder=folder, out=out_dir, check=True))
    return code, capsys.readouterr().out


def test_copied_execution_commands_all_load_their_signatures(tmp_path):
    folder = _library_workflow(tmp_path)
    expected = {
        rel.removeprefix("_commands/").removesuffix(".py")
        for rel in _COPIED_COMMANDS
        if not rel.endswith("/startup.py")
    } | {"Branch/branch_name"}
    signatures = workflow_signatures(folder)
    assert expected <= set(signatures)
    for name in expected:
        assert signatures[name].__name__ == "Signature"

    resolution = skill_commands.resolve_field_sources(folder)
    assert resolution.errors == ()
    assert resolution.infos == ()
    assert signatures["Library/open_book"].Input.model_fields["book_uid"].json_schema_extra[
        "available_from"
    ] == ["Library/find_book.book_uids"]
    assert signatures["Library/open_member"].Input.model_fields["member_uid"].json_schema_extra[
        "available_from"
    ] == ["Library/find_member.member_uids"]


def test_library_workflow_loads_context_and_root_skills(tmp_path):
    parametric = '''from pydantic import BaseModel, Field


class Signature:
    """Find books matching a query in the library."""

    class Input(BaseModel):
        query: str = Field(description="words of the title or author")


class PlanGenerator:
    def steps(self, input):
        return [f"find_book query={input.query}", "say how many books match"]
'''
    folder = _library_workflow(tmp_path, {"Library/search_books": parametric})
    skills = dict((name, (desc, steps)) for name, desc, steps in workflow_agent.load_skills(folder))
    assert set(skills) == {"hours_overview", "Library/dune_search", "Library/search_books"}
    assert skills["hours_overview"] == (
        "Say when the library's branches are open.",
        ("library_hours", "say when the branches are open"),
    )
    assert skills["Library/dune_search"] == (
        "Look up Dune in the library and report how many books match.",
        ("find_book query=Dune", "say how many books match"),
    )
    # Plain (fixed-steps) skills are commands too: a call of one runs through the phase runner.
    commands = workflow_agent.load_skill_commands(folder)
    assert set(commands) == {"hours_overview", "Library/dune_search", "Library/search_books"}
    assert commands["Library/search_books"].name == "Library/search_books"


def test_unknown_context_folder_is_a_load_error_for_each_file(tmp_path, caplog, capsys):
    folder = _library_workflow(tmp_path, {
        "Nowhere/first_skill": ROOT_SKILL,
        "Nowhere/second_skill": LIBRARY_SKILL,
    })
    with caplog.at_level(logging.ERROR):
        loaded = [name for name, _, _ in workflow_agent.load_skills(folder)]
    assert loaded == ["Library/dune_search", "hours_overview"]
    assert "skipping skill 'Nowhere/first_skill'" in caplog.text
    assert "skipping skill 'Nowhere/second_skill'" in caplog.text
    assert "Nowhere: context 'Nowhere' does not exist in this workflow" in caplog.text

    with pytest.raises(skill_preview.PreviewError) as info:
        skill_preview.render_previews(folder)
    assert "Nowhere/first_skill" in str(info.value)
    assert "Nowhere/second_skill" in str(info.value)
    assert "context 'Nowhere' does not exist in this workflow" in str(info.value)

    code, out = _preview_check(folder, str(tmp_path / "previews"), capsys)
    assert code == 1
    assert "Nowhere/first_skill" in out and "Nowhere/second_skill" in out
    assert "context 'Nowhere' does not exist in this workflow" in out


def test_root_skill_colliding_with_root_command_is_a_load_error(tmp_path, caplog, capsys):
    folder = _library_workflow(tmp_path, {"library_hours": ROOT_SKILL})
    with caplog.at_level(logging.ERROR):
        loaded = [name for name, _, _ in workflow_agent.load_skills(folder)]
    assert loaded == ["Library/dune_search", "hours_overview"]
    assert "skipping skill 'library_hours'" in caplog.text
    assert "_commands/library_hours.py" in caplog.text

    code, out = _preview_check(folder, str(tmp_path / "previews"), capsys)
    assert code == 1
    assert "library_hours: SkillFormatError:" in out
    assert "_skills/library_hours.py" in out and "_commands/library_hours.py" in out


def test_context_skill_colliding_with_context_command_is_a_load_error(tmp_path, caplog, capsys):
    folder = _library_workflow(tmp_path, {"Library/count_overdue": LIBRARY_SKILL})
    with caplog.at_level(logging.ERROR):
        loaded = [name for name, _, _ in workflow_agent.load_skills(folder)]
    assert loaded == ["Library/dune_search", "hours_overview"]
    assert "skipping skill 'Library/count_overdue'" in caplog.text
    assert "_commands/Library/count_overdue.py" in caplog.text

    code, out = _preview_check(folder, str(tmp_path / "previews"), capsys)
    assert code == 1
    assert "Library/count_overdue: SkillFormatError:" in out
    assert "_skills/Library/count_overdue.py" in out and "_commands/Library/count_overdue.py" in out


def test_underscore_and_dot_prefixed_entries_in_skill_context_are_skipped_silently(tmp_path, caplog):
    folder = _library_workflow(tmp_path)
    lib_skills = tmp_path / "library_wf" / "_skills" / "Library"
    (lib_skills / "_helpers.py").write_text("import no_such_module_anywhere\n", encoding="utf-8")
    (lib_skills / ".hidden.py").write_text("import no_such_module_anywhere\n", encoding="utf-8")
    (lib_skills / "__pycache__").mkdir()
    (lib_skills / "__pycache__" / "cached.py").write_text(LIBRARY_SKILL, encoding="utf-8")
    (lib_skills / ".cache").mkdir()
    (lib_skills / ".cache" / "cached.py").write_text(LIBRARY_SKILL, encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        loaded = [name for name, _, _ in workflow_agent.load_skills(folder)]
    assert loaded == ["Library/dune_search", "hours_overview"]
    assert caplog.text == ""


def test_folder_nested_deeper_than_skill_context_is_logged_and_skipped(tmp_path, caplog):
    folder = _library_workflow(tmp_path)
    nested = tmp_path / "library_wf" / "_skills" / "Library" / "Deep"
    nested.mkdir()
    (nested / "ignored.py").write_text(LIBRARY_SKILL, encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        loaded = [name for name, _, _ in workflow_agent.load_skills(folder)]
    assert loaded == ["Library/dune_search", "hours_overview"]
    assert f"{nested}: folders nested inside a skill context are not loaded; skipped" in caplog.text


# ------------------------------------------------------------------
# Lookup path (skill_scopes) and Jev scope-by-scope matching
# ------------------------------------------------------------------


def _live_library(folder: str, workflow_id_str: str):
    """Create a live ``Workflow`` at *folder* with a ``Library`` root and return ``(wf, lib_mod, library)``."""
    if fastworkflow.CommandContextModel is None:
        fastworkflow.init({})
    lib_mod = python_utils.get_module(os.path.join(folder, "application", "library.py"), folder)
    wf = fastworkflow.Workflow.create(folder, workflow_id_str=workflow_id_str)
    library = lib_mod.Library()
    wf.root_command_context = library
    return wf, lib_mod, library


@pytest.fixture
def jev_env():
    """Load ``env/.env`` and ``passwords/.env`` into fastworkflow, clear Jev caches,
    and skip when ``JEV_API_KEY`` is absent; restore the environment and caches after."""
    previous = dict(fastworkflow._env_vars)
    env_vars = {
        **dotenv_values(os.path.join(ROOT, "fastworkflow", "examples", "fastworkflow.env")),
        **dotenv_values(os.path.join(ROOT, "fastworkflow", "examples", "fastworkflow.passwords.env")),
    }
    for override in (os.path.join(ROOT, "env", ".env"), os.path.join(ROOT, "passwords", ".env")):
        if os.path.exists(override):
            env_vars.update(dotenv_values(override))
    if not env_vars.get("JEV_API_KEY"):
        pytest.skip("JEV_API_KEY not set in passwords/.env: matching needs a real Jev call")
    fastworkflow.init(env_vars=env_vars)
    workflow_agent._jev_lm.cache_clear()
    workflow_agent._skill_matcher.cache_clear()
    try:
        yield
    finally:
        fastworkflow.init(env_vars=previous)
        workflow_agent._jev_lm.cache_clear()
        workflow_agent._skill_matcher.cache_clear()


def test_skill_scopes_along_library_contexts_and_without_library(tmp_path):
    folder = _library_workflow(tmp_path)
    wf, lib_mod, library = _live_library(folder, f"scopes-{tmp_path.name}")

    # 1. Library active (root = Library): ("Library", "")
    assert workflow_agent.skill_scopes(wf) == ("Library", "")

    # 2. A Book inside the Library: ("Book", "Record", "Library", "")
    wf.current_command_context = library.books["b1"]
    assert workflow_agent.skill_scopes(wf) == ("Book", "Record", "Library", "")

    # 3. A Member inside the Library: ("Member", "Record", "Library", "")
    wf.current_command_context = library.members["m1"]
    assert workflow_agent.skill_scopes(wf) == ("Member", "Record", "Library", "")

    # 4. A Branch whose base is Library and whose parent is also the Library:
    # Library is added via Branch's base and skipped when visited again as Branch's parent.
    wf.current_command_context = lib_mod.Branch(library, "Central")
    assert workflow_agent.skill_scopes(wf) == ("Branch", "Library", "")

    # 5. Cases where Library is NOT on the path:
    # (a) Bare root before startup (root_command_context and current_command_context are None -> "*").
    bare_wf = fastworkflow.Workflow.create(folder, workflow_id_str=f"bare-{tmp_path.name}")
    assert bare_wf.current_command_context is None
    assert workflow_agent.skill_scopes(bare_wf) == ("",)

    # (b) A Book detached from any Library (book.library = None) when root is not set to a Library:
    # its context is Book, its base is Record, and its get_parent returns None, so Library is never reached.
    detached_wf = fastworkflow.Workflow.create(folder, workflow_id_str=f"detached-{tmp_path.name}")
    detached_wf.current_command_context = lib_mod.Book(None, "b9", "Orphan", "Anon", 1, "Central")
    assert workflow_agent.skill_scopes(detached_wf) == ("Book", "Record", "")


def test_jev_library_and_root_skills_match_only_where_visible(tmp_path, jev_env):
    folder = _library_workflow(tmp_path)
    wf, lib_mod, library = _live_library(folder, f"jev-vis-{tmp_path.name}")

    dune_query = "Look up Dune in the library and tell me how many books match."
    hours_query = "When are the library's branches open?"

    # 1. The Library context skill matches when Library is active, when a Branch is active,
    # and when a Book inside the Library is active; the root skill matches in each of them too.
    for active in (library, lib_mod.Branch(library, "Central"), library.books["b1"]):
        wf.current_command_context = active
        scopes = workflow_agent.skill_scopes(wf)
        matched_lib = workflow_agent.match_skill(dune_query, folder, scopes=scopes)
        assert matched_lib is not None and matched_lib[0] == "Library/dune_search"
        template, _ = workflow_agent.planner_skill_for(dune_query, folder, scopes=scopes)
        assert "find_book query=Dune" in template

        matched_root = workflow_agent.match_skill(hours_query, folder, scopes=scopes)
        assert matched_root is not None and matched_root[0] == "hours_overview"

    # 2. Where Library is NOT on the path (bare root before startup, or detached Book),
    # the Library skill does not match, while the root skill still matches everywhere.
    bare_wf = fastworkflow.Workflow.create(folder, workflow_id_str=f"jev-bare-{tmp_path.name}")
    detached_wf = fastworkflow.Workflow.create(folder, workflow_id_str=f"jev-detached-{tmp_path.name}")
    detached_wf.current_command_context = lib_mod.Book(None, "b9", "Orphan", "Anon", 1, "Central")
    for non_lib_wf in (bare_wf, detached_wf):
        scopes = workflow_agent.skill_scopes(non_lib_wf)
        assert "Library" not in scopes
        assert workflow_agent.match_skill(dune_query, folder, scopes=scopes) is None
        assert workflow_agent.planner_skill_for(dune_query, folder, scopes=scopes) == ("", None)
        matched_root = workflow_agent.match_skill(hours_query, folder, scopes=scopes)
        assert matched_root is not None and matched_root[0] == "hours_overview"


def test_jev_library_scope_wins_before_root_scope_when_both_fit(tmp_path, caplog, jev_env):
    competing_root_dune = '''
class Signature:
    """Look up Dune in the library and report how many books match."""


class PlanGenerator:
    def steps(self):
        return ["library_hours", "say how many books match"]
'''
    folder = _library_workflow(tmp_path, {"root_dune_search": competing_root_dune})
    wf, _lib_mod, _library = _live_library(folder, f"jev-win-{tmp_path.name}")
    scopes = workflow_agent.skill_scopes(wf)
    assert scopes == ("Library", "")

    dune_query = "Look up Dune in the library and tell me how many books match."
    caplog.clear()
    with caplog.at_level(logging.INFO):
        matched = workflow_agent.match_skill(dune_query, folder, scopes=scopes)
    assert matched is not None
    assert matched[0] == "Library/dune_search"
    assert "skill match in scope 'Library'" in caplog.text
    assert "skill match in scope 'root'" not in caplog.text


# ------------------------------------------------------------------
# Previews for context skills (<out>/<Context>/<name>.md)
# ------------------------------------------------------------------


def test_write_and_check_previews_include_context_subfolders(tmp_path, capsys):
    folder = _library_workflow(tmp_path / "wf")
    out = tmp_path / "previews"
    result = skill_preview.write_previews(folder, str(out))
    assert result == {
        "written": ["Library/dune_search.md", "hours_overview.md"],
        "unchanged": [],
        "removed": [],
        "kept": [],
    }
    assert (out / "hours_overview.md").is_file()
    context_md = out / "Library" / "dune_search.md"
    assert context_md.is_file()
    text = context_md.read_text(encoding="utf-8")
    assert "from _skills/Library/dune_search.py." in text
    assert "Source: `_skills/Library/dune_search.py`." in text
    assert "# Library/dune_search" in text

    assert skill_preview.check_previews(folder, str(out)) == []
    code, stdout = _preview_check(folder, str(out), capsys)
    assert code == 0
    assert "are current" in stdout

    # Second write reports both as unchanged.
    second = skill_preview.write_previews(folder, str(out))
    assert second["unchanged"] == ["Library/dune_search.md", "hours_overview.md"]
    assert second["written"] == []


def test_check_previews_reports_stale_missing_and_extra_in_context_subfolders(tmp_path, capsys):
    folder = _library_workflow(tmp_path / "wf")
    out = tmp_path / "previews"
    skill_preview.write_previews(folder, str(out))

    context_md = out / "Library" / "dune_search.md"
    context_md.write_text(
        context_md.read_text(encoding="utf-8").replace("find_book", "search_book"),
        encoding="utf-8",
    )
    assert skill_preview.check_previews(folder, str(out)) == ["stale: Library/dune_search.md"]
    code, stdout = _preview_check(folder, str(out), capsys)
    assert code == 1
    assert "stale: Library/dune_search.md" in stdout

    context_md.unlink()
    assert skill_preview.check_previews(folder, str(out)) == ["missing: Library/dune_search.md"]

    skill_preview.write_previews(folder, str(out))
    (out / "Library" / "other.md").write_text("hand-written\n", encoding="utf-8")
    # Deeper folders and _/. folders are ignored by _present_previews.
    (out / "Library" / "Deep").mkdir()
    (out / "Library" / "Deep" / "nested.md").write_text("ignored\n", encoding="utf-8")
    (out / "_private").mkdir()
    (out / "_private" / "ignored.md").write_text("ignored\n", encoding="utf-8")
    (out / ".hidden").mkdir()
    (out / ".hidden" / "ignored.md").write_text("ignored\n", encoding="utf-8")
    assert skill_preview.check_previews(folder, str(out)) == ["extra: Library/other.md"]


def test_write_previews_removes_orphan_context_preview_and_empty_folder_but_keeps_handwritten(tmp_path):
    with_context = _library_workflow(tmp_path / "wf_with_context")
    without_context = _library_workflow(
        tmp_path / "wf_without_context",
        extra_skills={"hours_overview": ROOT_SKILL},
        default_skills=False,
    )
    out = tmp_path / "previews"
    skill_preview.write_previews(with_context, str(out))
    assert (out / "Library" / "dune_search.md").is_file()

    # A second subfolder holds both a generated orphan and a hand-written note.
    member_dir = out / "Member"
    member_dir.mkdir()
    (member_dir / "old_member_skill.md").write_text(
        skill_preview.MARKER.format(source="_skills/Member/old_member_skill.py") + "\n",
        encoding="utf-8",
    )
    (member_dir / "notes.md").write_text("hand-written member notes\n", encoding="utf-8")

    result = skill_preview.write_previews(without_context, str(out))
    assert result["unchanged"] == ["hours_overview.md"]
    assert result["written"] == []
    assert result["removed"] == ["Library/dune_search.md", "Member/old_member_skill.md"]
    assert result["kept"] == ["Member/notes.md"]
    assert not (out / "Library").exists()
    assert not (member_dir / "old_member_skill.md").exists()
    assert (member_dir / "notes.md").is_file()


# ------------------------------------------------------------------
# JEV_API_KEY startup warning and passwords template
# ------------------------------------------------------------------


@pytest.fixture
def no_jev_env(monkeypatch):
    """Ensure ``JEV_API_KEY`` is absent from both ``os.environ`` and ``fastworkflow._env_vars``."""
    previous = dict(fastworkflow._env_vars)
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    env_vars = {k: v for k, v in previous.items() if k != "JEV_API_KEY"}
    fastworkflow.init(env_vars=env_vars)
    workflow_agent._jev_lm.cache_clear()
    workflow_agent._skill_matcher.cache_clear()
    try:
        yield
    finally:
        fastworkflow.init(env_vars=previous)
        workflow_agent._jev_lm.cache_clear()
        workflow_agent._skill_matcher.cache_clear()


def test_missing_jev_key_logs_an_info_line_once_per_workflow_folder(tmp_path, caplog, no_jev_env):
    folder = _library_workflow(tmp_path)
    workflow_agent._jev_lm.cache_clear()
    workflow_agent._skill_matcher.cache_clear()

    caplog.clear()
    with caplog.at_level(logging.INFO):
        workflow_agent.warm_skills(folder)
        workflow_agent.warm_skills(folder)
        assert workflow_agent.match_skill("anything", folder) is None

    # 3.9's INFO line; this library's planning commands have no verify field, so none is named.
    expected = (
        f"{folder}: planning commands are chosen and adapted by the agent as it plans; "
        "set JEV_API_KEY to add Jev's choice."
    )
    infos = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert infos.count(expected) == 1
    assert not any(r.levelno >= logging.WARNING and "JEV_API_KEY" in r.getMessage()
                   for r in caplog.records)


def test_missing_jev_key_does_not_warn_without_planning_commands(tmp_path, caplog, no_jev_env):
    empty_skills_folder = _library_workflow(tmp_path / "empty_skills", default_skills=False)
    no_skills_folder = _library_workflow(tmp_path / "no_skills", default_skills=False)
    shutil.rmtree(os.path.join(no_skills_folder, "_skills"))

    for folder in (empty_skills_folder, no_skills_folder):
        workflow_agent._jev_lm.cache_clear()
        workflow_agent._skill_matcher.cache_clear()
        caplog.clear()
        with caplog.at_level(logging.INFO):
            workflow_agent.warm_skills(folder)
            workflow_agent.warm_skills(folder)
            assert workflow_agent.match_skill("anything", folder) is None
        assert not any("JEV_API_KEY" in r.getMessage() for r in caplog.records)


def test_skill_preview_does_not_warn_when_jev_key_is_unset(tmp_path, caplog, no_jev_env):
    folder = _library_workflow(tmp_path / "wf")
    out = str(tmp_path / "previews")
    workflow_agent._jev_lm.cache_clear()
    workflow_agent._skill_matcher.cache_clear()

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        rendered = skill_preview.render_previews(folder)
        skill_preview.write_previews(folder, out)
        assert skill_preview.check_previews(folder, out) == []
    assert set(rendered) == {"Library/dune_search.md", "hours_overview.md"}
    assert not any("JEV_API_KEY" in r.getMessage() for r in caplog.records)


def test_passwords_template_has_commented_jev_api_key():
    template_path = os.path.join(ROOT, "fastworkflow", "examples", "fastworkflow.passwords.env")
    with open(template_path, encoding="utf-8") as handle:
        lines = [line.strip() for line in handle]
    assert any(line.startswith("#") and "JEV_API_KEY=" in line for line in lines)
    parsed = dotenv_values(template_path)
    assert "JEV_API_KEY" not in parsed


# ---------------------------------------------------------------------------
# The rule sentences (design 3.4, Descriptions) and a run without JEV_API_KEY (3.9)
# ---------------------------------------------------------------------------

RULE_CHECK = "Library/rule_check"
RULE_CHECK_FILE = "_commands/Library/rule_check.py"
#: A planning command with a field for each option that has a rule sentence: a subject str, an int
#: with in_request, a bool with verify. It loads without JEV_API_KEY.
RULE_CHECK_SOURCE = '''"""rule_check: how many copies of a book the request asks about, and whether it is on loan.

A fixture for the rule sentences (design 3.4): one field for each option that has a sentence.
"""
from pydantic import BaseModel, Field

from fastworkflow.planning import run

from .find_book import Signature as FindBook


class Signature:
    """Say how many copies of a book the request asks about, and whether it is on loan."""

    class Input(BaseModel):
        title: str = Field(description="the book's title",
                           json_schema_extra={"subject": True})
        copies: int = Field(default=1, description="how many copies the request asks about",
                            json_schema_extra={"in_request": True})
        loaned: bool = Field(default=False, description="whether the request asks about loans",
                             json_schema_extra={"verify": True})


class PlanGenerator:
    def find(self, input) -> FindBook.Output:
        return [run("find_book", query=input.title)]

    def __result__(self, input, find):
        return f"{input.title}: {input.copies} asked about."
'''


def _with_rule_check(tmp_path) -> str:
    """A library copy with the rule_check planning command, loaded as a workflow's commands are."""
    # Every library test copy is a library_wf package; a stale one left in sys.modules by an
    # earlier test poisons the relative imports of this copy (see test_planning_commands).
    for key in list(sys.modules):
        if key == "library_wf" or key.startswith("library_wf."):
            del sys.modules[key]
    folder = _library_workflow(tmp_path)
    # The Library planning command the rule test reads (its extraction sentences), and its helper.
    for rel in ("_commands/Library/_wording.py", "_commands/Library/book_by_title.py"):
        shutil.copy2(os.path.join(LIBRARY, rel), Path(folder, rel))
    Path(folder, RULE_CHECK_FILE).write_text(RULE_CHECK_SOURCE, encoding="utf-8")
    fastworkflow.RoutingRegistry.clear_registry()
    fastworkflow.RoutingRegistry.get_definition(folder, load_cached=False)
    return folder


def test_the_jev_info_line_names_the_verify_fields_and_is_logged_once_per_folder(
        tmp_path, no_jev_env, caplog):
    folder = _with_rule_check(tmp_path)

    caplog.clear()
    with caplog.at_level(logging.INFO):
        workflow_agent._warn_missing_jev_key(folder)
        workflow_agent._warn_missing_jev_key(folder)

    expected = (f"{folder}: planning commands are chosen and adapted by the agent as it plans; "
                f"set JEV_API_KEY to add Jev's choice and the verify checks of {RULE_CHECK}.loaned.")
    infos = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert infos.count(expected) == 1
    assert not any(r.levelno >= logging.WARNING and "JEV_API_KEY" in r.getMessage()
                   for r in caplog.records)

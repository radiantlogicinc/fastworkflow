"""CommandDirectory planning-command registration (Phase 3 step 4)."""
from __future__ import annotations

import json
import os
import shutil

import pytest

from fastworkflow.command_directory import (
    CommandDirectory,
    CommandMetadata,
    compute_commands_source_fingerprint,
    get_cached_command_directory,
)
from fastworkflow.skill_commands import SkillFormatError


def _minimal_workflow(tmp_path, name: str = "wf") -> str:
    folder = tmp_path / name
    (folder / "_commands").mkdir(parents=True)
    (folder / "_skills").mkdir(parents=True)
    (folder / "__init__.py").write_text("", encoding="utf-8")
    inheritance = {"Library": {"base": []}}
    (folder / "_commands" / "context_inheritance_model.json").write_text(
        json.dumps(inheritance), encoding="utf-8"
    )
    (folder / "_commands" / "Library").mkdir()
    (folder / "_commands" / "Library" / "_Library.py").write_text(
        "class Context:\n    pass\n", encoding="utf-8"
    )
    (folder / "_commands" / "Library" / "noop.py").write_text(_EXEC_CMD, encoding="utf-8")
    return str(folder)


_EXEC_CMD = '''
class Signature:
    """An execution command."""


class ResponseGenerator:
    def __call__(self, workflow, command, command_parameters):
        return None
'''

_PLAN_CMD = '''
class Signature:
    """A planning command."""


class PlanGenerator:
    def steps(self):
        return ["noop"]
'''


def test_skills_planning_command_registered_with_qualified_name_and_is_planning(tmp_path):
    folder = _minimal_workflow(tmp_path)
    lib_skills = os.path.join(folder, "_skills", "Library")
    os.makedirs(lib_skills, exist_ok=True)
    with open(os.path.join(lib_skills, "my_plan.py"), "w", encoding="utf-8") as handle:
        handle.write(_PLAN_CMD)

    directory = CommandDirectory.load(folder)
    assert "Library/my_plan" in directory.get_commands()
    meta = directory.get_command_metadata("Library/my_plan")
    assert meta.is_planning is True
    assert meta.response_generation_class_name == "PlanGenerator"
    assert "Library/my_plan" not in directory.get_execution_commands()


def test_commands_tree_planning_detected_by_ast(tmp_path):
    folder = _minimal_workflow(tmp_path)
    with open(os.path.join(folder, "_commands", "Library", "only_plan.py"), "w", encoding="utf-8") as handle:
        handle.write(_PLAN_CMD)

    directory = CommandDirectory.load(folder)
    meta = directory.get_command_metadata("Library/only_plan")
    assert meta.is_planning is True
    assert meta.response_generation_class_name == "PlanGenerator"


def test_context_with_only_planning_commands_loads(tmp_path):
    folder = _minimal_workflow(tmp_path)
    with open(os.path.join(folder, "_commands", "Library", "only_plan.py"), "w", encoding="utf-8") as handle:
        handle.write(_PLAN_CMD)

    directory = CommandDirectory.load(folder)
    assert "Library/only_plan" in directory.get_commands()


def test_adding_skill_file_changes_commands_source_fingerprint(tmp_path):
    folder = _minimal_workflow(tmp_path)
    fp_before = compute_commands_source_fingerprint(folder)
    lib_skills = os.path.join(folder, "_skills", "Library")
    os.makedirs(lib_skills, exist_ok=True)
    with open(os.path.join(lib_skills, "new_skill.py"), "w", encoding="utf-8") as handle:
        handle.write(_PLAN_CMD)
    fp_after = compute_commands_source_fingerprint(folder)
    assert fp_after != fp_before


def test_skill_collision_with_command_raises_same_error(tmp_path):
    folder = _minimal_workflow(tmp_path)
    with open(os.path.join(folder, "_commands", "Library", "dup_name.py"), "w", encoding="utf-8") as handle:
        handle.write(_EXEC_CMD)
    lib_skills = os.path.join(folder, "_skills", "Library")
    os.makedirs(lib_skills, exist_ok=True)
    skill_path = os.path.join(lib_skills, "dup_name.py")
    with open(skill_path, "w", encoding="utf-8") as handle:
        handle.write(_PLAN_CMD)

    with pytest.raises(SkillFormatError) as info:
        CommandDirectory.check_skill_path("Library/dup_name", skill_path, folder)
    assert "same name as command" in str(info.value)
    assert "_commands/Library/dup_name.py" in str(info.value)


def test_unknown_skill_context_raises_same_error(tmp_path):
    folder = _minimal_workflow(tmp_path)
    nowhere = os.path.join(folder, "_skills", "Nowhere")
    os.makedirs(nowhere, exist_ok=True)
    skill_path = os.path.join(nowhere, "orphan.py")
    with open(skill_path, "w", encoding="utf-8") as handle:
        handle.write(_PLAN_CMD)

    with pytest.raises(SkillFormatError) as info:
        CommandDirectory.check_skill_path("Nowhere/orphan", skill_path, folder)
    assert "does not exist in this workflow" in str(info.value)

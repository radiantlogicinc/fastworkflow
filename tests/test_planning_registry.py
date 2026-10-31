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


def _minimal_workflow(tmp_path, name: str = "wf") -> str:
    folder = tmp_path / name
    (folder / "_commands").mkdir(parents=True)
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


def test_planning_commands_planning_command_registered_with_qualified_name_and_is_planning(tmp_path):
    folder = _minimal_workflow(tmp_path)
    lib_planning_commands = os.path.join(folder, "_commands", "Library")
    os.makedirs(lib_planning_commands, exist_ok=True)
    with open(os.path.join(lib_planning_commands, "my_plan.py"), "w", encoding="utf-8") as handle:
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


def test_adding_planning_command_file_changes_commands_source_fingerprint(tmp_path):
    folder = _minimal_workflow(tmp_path)
    fp_before = compute_commands_source_fingerprint(folder)
    lib_planning_commands = os.path.join(folder, "_commands", "Library")
    os.makedirs(lib_planning_commands, exist_ok=True)
    with open(os.path.join(lib_planning_commands, "new_planning_command.py"), "w", encoding="utf-8") as handle:
        handle.write(_PLAN_CMD)
    fp_after = compute_commands_source_fingerprint(folder)
    assert fp_after != fp_before


def test_planning_command_collision_with_command_raises_same_error(tmp_path):
    # Phase 7 (one tree): a planning command and an execution command cannot share a file, so the
    # cross-tree collision error is retired. The one file is the planning command, and it is registered
    # as planning (not as a second command of that name).
    folder = _minimal_workflow(tmp_path)
    lib_planning_commands = os.path.join(folder, "_commands", "Library")
    os.makedirs(lib_planning_commands, exist_ok=True)
    planning_command_path = os.path.join(lib_planning_commands, "dup_name.py")
    with open(planning_command_path, "w", encoding="utf-8") as handle:
        handle.write(_PLAN_CMD)

    assert CommandDirectory.check_planning_command_path("Library/dup_name", planning_command_path) is None
    directory = CommandDirectory.load(folder, hydrate_utterances=False)
    assert directory.get_command_metadata("Library/dup_name").is_planning


def test_unknown_planning_command_context_raises_same_error(tmp_path):
    # Phase 7 (one tree): a _commands/<Context>/ folder is itself the context, so there is no unknown-context
    # error; the planning command in it is a command of that context.
    folder = _minimal_workflow(tmp_path)
    nowhere = os.path.join(folder, "_commands", "Nowhere")
    os.makedirs(nowhere, exist_ok=True)
    planning_command_path = os.path.join(nowhere, "orphan.py")
    with open(planning_command_path, "w", encoding="utf-8") as handle:
        handle.write(_PLAN_CMD)

    assert CommandDirectory.check_planning_command_path("Nowhere/orphan", planning_command_path) is None
    directory = CommandDirectory.load(folder, hydrate_utterances=False)
    assert directory.get_command_metadata("Nowhere/orphan").is_planning

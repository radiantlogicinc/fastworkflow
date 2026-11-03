"""The ``approval`` annotation: declared on a command class, validated at load, exposed (Phase 10 step 10.13).

``approval = "mandatory" | "delegatable"`` on a command's ``ResponseGenerator`` (or a planning command's
``PlanGenerator``) is read the way ``descends_to`` is. Absent means agent discretion. This file covers the
declaration only: nothing is gated here (enforcement is fix-3rsq.10.14).

Each case runs on a COPY of the library example, annotated in the copy. Nothing is mocked: the routing
definition, the command classes, the planning commands and the listing builders are the copy's own.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

import fastworkflow
from fastworkflow import approval_annotations as aa
from fastworkflow.approval_annotations import (
    ApprovalAnnotationError, command_approval, planning_steps_needing_approval)
from fastworkflow.command_metadata_api import CommandMetadataAPI
from fastworkflow.command_routing import RoutingRegistry
from fastworkflow.examples.library.application.library import Library
from fastworkflow.run_listing import command_shortlist_text
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401 - shared fixture

pytestmark = pytest.mark.usefixtures("initialized_fastworkflow")

LIBRARY = Path(fastworkflow.__file__).resolve().parent / "examples" / "library"

_COPY_IGNORE = shutil.ignore_patterns(
    "___command_info", "___workflow_contexts", "___convo_info", "__pycache__",
)

MARKER_MANDATORY = "(needs your approval before it runs)"
MARKER_DELEGATABLE = "(may need your approval)"


def _copy(tmp_path, name: str) -> Path:
    folder = tmp_path / name
    shutil.copytree(LIBRARY, folder, ignore=_COPY_IGNORE, symlinks=True)
    return folder


def _annotate(folder: Path, command_file: str, value_text: str, *, on: str = "ResponseGenerator") -> Path:
    """Write ``approval = <value_text>`` on the command's ``ResponseGenerator`` (or ``PlanGenerator``) in the copy."""
    path = folder / "_commands" / command_file
    text = path.read_text()
    marker = f"class {on}:\n"
    assert text.count(marker) == 1, path
    path.write_text(text.replace(marker, f"class {on}:\n    approval = {value_text}\n", 1))
    return path


def _cme() -> str:
    return fastworkflow.get_internal_workflow_path("command_metadata_extraction")


def _listing_app(folder: Path):
    """The copy as an app workflow, entered at the Library root (the shortlist's own context)."""
    app = fastworkflow.Workflow.create(workflow_folderpath=str(folder), workflow_id_str=f"approval-{folder.name}")
    library = Library()
    app.root_command_context = library
    app.current_command_context = library
    return app


# ---------------------------------------------------------------------------
# Accepted values
# ---------------------------------------------------------------------------

def test_both_allowed_values_load_and_are_read_by_the_accessor(tmp_path):
    folder = _copy(tmp_path, "accepted")
    _annotate(folder, "Library/open_book.py", '"mandatory"')
    _annotate(folder, "Library/open_member.py", '"delegatable"')

    assert command_approval(str(folder), "Library/open_book") == "mandatory"
    assert command_approval(str(folder), "Library/open_member") == "delegatable"


def test_an_unannotated_command_has_no_approval(tmp_path):
    folder = _copy(tmp_path, "unannotated")
    assert command_approval(str(folder), "Library/find_book") is None


def test_an_unknown_command_has_no_approval(tmp_path):
    folder = _copy(tmp_path, "unknown_name")
    assert command_approval(str(folder), "Library/no_such_command") is None


def test_the_value_is_the_snapshot_in_command_directory_json(tmp_path):
    folder = _copy(tmp_path, "snapshot")
    _annotate(folder, "Library/open_book.py", '"mandatory"')
    RoutingRegistry.get_definition(str(folder))

    snapshot = json.loads((folder / "___command_info" / "command_directory.json").read_text())
    assert snapshot["map_command_2_metadata"]["Library/open_book"]["approval"] == "mandatory"


# ---------------------------------------------------------------------------
# Invalid values are load errors
# ---------------------------------------------------------------------------

def test_an_unknown_value_is_a_load_error_naming_command_file_and_allowed_values(tmp_path):
    folder = _copy(tmp_path, "unknown_value")
    _annotate(folder, "Library/open_book.py", '"sometimes"')

    with pytest.raises(ApprovalAnnotationError) as caught:
        RoutingRegistry.get_definition(str(folder))
    message = str(caught.value)
    assert "Library/open_book" in message
    assert "Library/open_book.py" in message
    assert "'sometimes'" in message
    assert "'mandatory'" in message and "'delegatable'" in message


def test_a_non_string_value_is_a_load_error(tmp_path):
    folder = _copy(tmp_path, "wrong_type")
    _annotate(folder, "Library/open_book.py", "True")

    with pytest.raises(ApprovalAnnotationError) as caught:
        RoutingRegistry.get_definition(str(folder))
    message = str(caught.value)
    assert "Library/open_book" in message
    assert "must be the string" in message
    assert "not bool" in message


def test_the_accessor_surfaces_a_load_error_rather_than_hiding_it(tmp_path):
    folder = _copy(tmp_path, "accessor_error")
    _annotate(folder, "Library/open_book.py", '"maybe"')

    with pytest.raises(ApprovalAnnotationError):
        command_approval(str(folder), "Library/open_book")


# ---------------------------------------------------------------------------
# Planning commands
# ---------------------------------------------------------------------------

def test_planning_steps_needing_approval_lists_the_annotated_commands_its_template_calls(tmp_path):
    folder = _copy(tmp_path, "planning_steps")
    _annotate(folder, "Library/open_member.py", '"delegatable"')

    # member_report's template calls find_member, open_member, list_member_loans and show_notes;
    # only open_member is annotated.
    assert planning_steps_needing_approval(str(folder), "Library/member_report") == [
        ("Library/open_member", "delegatable")]


def test_planning_steps_needing_approval_is_empty_without_annotated_steps(tmp_path):
    folder = _copy(tmp_path, "planning_none")
    assert planning_steps_needing_approval(str(folder), "Library/member_report") == []


def test_a_planning_command_may_carry_the_annotation_itself(tmp_path):
    folder = _copy(tmp_path, "planning_self")
    _annotate(folder, "Library/member_report.py", '"mandatory"', on="PlanGenerator")

    assert command_approval(str(folder), "Library/member_report") == "mandatory"


# ---------------------------------------------------------------------------
# Listing markers
# ---------------------------------------------------------------------------

def test_the_planner_and_executor_listing_mark_an_annotated_command(tmp_path):
    folder = _copy(tmp_path, "listing_markers")
    _annotate(folder, "Library/open_book.py", '"mandatory"')
    _annotate(folder, "Library/open_member.py", '"delegatable"')
    folder_str = str(folder)

    full = CommandMetadataAPI.get_command_display_text(
        subject_workflow_path=folder_str, cme_workflow_path=_cme(), active_context_name="Library")
    assert "- open_book " + MARKER_MANDATORY in full
    assert "- find_book " + MARKER_MANDATORY not in full
    assert "- find_book " + MARKER_DELEGATABLE not in full

    callable_here = CommandMetadataAPI.format_callable_here_command(
        subject_workflow_path=folder_str, cme_workflow_path=_cme(), owning_context_name="Library",
        qualified_command_name="Library/open_member", active_context_name="Library")
    assert callable_here.splitlines()[0] == "- open_member " + MARKER_DELEGATABLE

    deferred = CommandMetadataAPI.format_deferred_planner_command(
        subject_workflow_path=folder_str, cme_workflow_path=_cme(), owning_context_name="Library",
        qualified_command_name="Library/open_member", active_context_name="*")
    assert deferred.splitlines()[0] == "- open_member " + MARKER_DELEGATABLE


def test_the_single_command_listing_marks_the_name_line(tmp_path):
    folder = _copy(tmp_path, "single_marker")
    _annotate(folder, "Library/open_book.py", '"mandatory"')

    text = CommandMetadataAPI.get_command_display_text_for_command(
        subject_workflow_path=str(folder), cme_workflow_path=_cme(), active_context_name="Library",
        qualified_command_name="Library/open_book")
    assert text.splitlines()[0] == "- open_book " + MARKER_MANDATORY


def test_the_run_listing_shortlist_marks_an_annotated_command(tmp_path):
    folder = _copy(tmp_path, "shortlist_marker")
    _annotate(folder, "Library/open_member.py", '"delegatable"')
    app = _listing_app(folder)

    text = command_shortlist_text(app, ["Library/open_member", "Library/find_member"])
    assert "- open_member " + MARKER_DELEGATABLE in text
    assert "- find_member " + MARKER_DELEGATABLE not in text
    assert "- find_member (" not in text


def test_marker_text_is_the_only_difference_an_annotation_makes(tmp_path):
    """The annotated listing equals the plain one once the marker is taken out."""
    plain_folder = _copy(tmp_path, "plain_side")
    annotated_folder = _copy(tmp_path, "annotated_side")
    _annotate(annotated_folder, "Library/open_book.py", '"mandatory"')

    def listing(folder):
        # Command classes are cached by name for the process, so each copy is read on a fresh registry.
        RoutingRegistry.clear_registry()
        return CommandMetadataAPI.get_command_display_text(
            subject_workflow_path=str(folder), cme_workflow_path=_cme(), active_context_name="Library")

    annotated = listing(annotated_folder)
    plain = listing(plain_folder)
    assert MARKER_MANDATORY in annotated and MARKER_MANDATORY not in plain
    assert annotated.replace(" " + MARKER_MANDATORY, "") == plain


# ---------------------------------------------------------------------------
# Unannotated workflows are unchanged
# ---------------------------------------------------------------------------

def test_an_unannotated_workflow_writes_no_approval_key_and_no_marker(tmp_path):
    folder = _copy(tmp_path, "byte_identical")
    RoutingRegistry.get_definition(str(folder))

    snapshot = (folder / "___command_info" / "command_directory.json").read_text()
    assert '"approval"' not in snapshot

    listing = CommandMetadataAPI.get_command_display_text(
        subject_workflow_path=str(folder), cme_workflow_path=_cme(), active_context_name="Library")
    assert "approval" not in listing
    assert MARKER_MANDATORY not in listing and MARKER_DELEGATABLE not in listing


def test_the_marker_is_empty_for_an_unannotated_command(tmp_path):
    folder = _copy(tmp_path, "marker_empty")
    assert aa.approval_marker(str(folder), "Library/find_book") == ""
    assert aa.approval_marker(str(folder), "Library/no_such_command") == ""

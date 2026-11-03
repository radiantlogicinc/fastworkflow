"""Framework planning commands are system machinery (fix-3rsq.plan-review.1, coordinator correction):

(a) not loaded into the internal command_metadata_extraction (CME) workflow;
(b) not head labels, and not counted by the artifact construction check or the training signature, so no
    workflow needs a retrain because of them;
(c) not listed to the planner, the agent or the user (routing listings, the agent's name resolution, labels);
(d) still resolvable by the framework through the loader and the executor when called explicitly, and marked by
    an explicit ``framework`` flag on the command metadata.

Integration tests on the real routing, loader, snapshots and artifact check. No model is called.
"""
from __future__ import annotations

import json
from pathlib import Path

import fastworkflow
from fastworkflow import command_executor, planning_commands
from fastworkflow.command_directory import CommandDirectory, is_visible_planning
from fastworkflow.train.artifact_versioning import construction_record_problem, workflow_has_planning_commands
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401 - the routing registry

pytestmark = __import__('pytest').mark.usefixtures('initialized_fastworkflow')

FRAMEWORK = ("review_subtasks", "review_plan")
LIBRARY = str(Path(fastworkflow.__file__).resolve().parent / "examples" / "library")
HELLO_WORLD = str(Path(fastworkflow.__file__).resolve().parent / "examples" / "hello_world")
CME = fastworkflow.get_internal_workflow_path("command_metadata_extraction")


def test_a_the_internal_cme_workflow_never_loads_the_framework_planning_commands():
    directory = CommandDirectory.load(CME)
    assert not [name for name in FRAMEWORK if name in directory.map_command_2_metadata]
    assert not [name for name, meta in directory.map_command_2_metadata.items() if meta.framework]


def test_b_the_construction_check_passes_for_the_cme_and_hello_world_with_the_framework_folder_present():
    assert construction_record_problem(CME) is None
    assert construction_record_problem(HELLO_WORLD) is None
    assert workflow_has_planning_commands(HELLO_WORLD) is False
    assert construction_record_problem(LIBRARY) is None


def test_b_the_framework_commands_are_not_counted_as_planning_commands_of_a_workflow():
    snapshot = json.loads(Path(LIBRARY, "___command_info", "command_directory.json").read_text())
    entries = snapshot["map_command_2_metadata"]
    for name in FRAMEWORK:
        assert entries[name]["is_planning"] is True and entries[name]["framework"] is True
    assert workflow_has_planning_commands(LIBRARY) is True          # the app's own planning commands still count
    directory = CommandDirectory.load(LIBRARY)
    assert not any(is_visible_planning(directory.map_command_2_metadata[name]) for name in FRAMEWORK)


def test_c_the_framework_commands_are_not_in_the_planner_or_the_agent_listing():
    routing = fastworkflow.RoutingRegistry.get_definition(LIBRARY)
    global_listing = routing.get_planning_command_names("*")
    assert not [name for name in FRAMEWORK if name in global_listing]
    for context in ("Library", "Book"):
        assert not [name for name in FRAMEWORK if name in routing.get_planning_command_names(context)]


def test_c_the_agent_cannot_name_a_framework_command_as_a_tool():
    routing = fastworkflow.RoutingRegistry.get_definition(LIBRARY)
    assert "review_plan" not in routing.get_planning_command_names("*")
    assert "review_plan" in CommandDirectory.load(LIBRARY).map_command_2_metadata      # the loader still has it


def test_c_a_framework_command_is_not_a_label_of_the_intent_model():
    from fastworkflow.model_pipeline_training import _is_planning_name

    directory = CommandDirectory.load(LIBRARY)
    assert _is_planning_name(directory, "review_plan") is False
    assert _is_planning_name(directory, "Library/note_member") is True   # an app planning command is still one


def test_d_the_framework_commands_resolve_through_the_loader_and_the_executor_when_called_explicitly():
    assert planning_commands.planning_command(LIBRARY, "review_subtasks") is not None
    assert planning_commands.planning_command(LIBRARY, "review_plan") is not None
    assert command_executor._resolve_planning_command_name(LIBRARY, "review_plan") == "review_plan"
    assert command_executor._resolve_planning_command_name(LIBRARY, "review_subtasks") == "review_subtasks"


def test_d_the_flag_is_on_the_metadata_only_for_framework_commands():
    directory = CommandDirectory.load(LIBRARY)
    assert directory.map_command_2_metadata["review_plan"].framework is True
    assert directory.map_command_2_metadata["Library/note_member"].framework is False
    assert is_visible_planning(directory.map_command_2_metadata["Library/note_member"]) is True
    assert is_visible_planning(directory.map_command_2_metadata["review_plan"]) is False


FRAMEWORK_NAMES = {"review_subtasks", "review_plan", "review_proposal"}


def test_e_the_training_report_has_no_row_for_a_framework_command_and_the_gate_does_not_refuse_it():
    from fastworkflow.train import __main__ as trainer
    from fastworkflow.train.training_report import build_report

    report = build_report(LIBRARY)
    assert not [row.command_name for row in report.rows if row.command_name in FRAMEWORK_NAMES]
    blocking = {row.command_name for row in report.blocking_rows}
    assert not blocking & FRAMEWORK_NAMES
    # The completeness gate itself, called directly: it refuses only for the library's own rows, never for these.
    if report.has_blocking_problems:
        try:
            trainer._require_publishable_training_report(report)
        except trainer.TrainingDataError as error:
            assert not any(name in str(error) for name in FRAMEWORK_NAMES)
    else:
        trainer._require_publishable_training_report(report)


def test_e_the_parameter_example_commands_exclude_framework_commands():
    from fastworkflow.train.__main__ import _get_commands_with_parameters

    names = set(_get_commands_with_parameters(
        str(Path(LIBRARY, "___command_info", "command_directory.json"))))
    assert not names & FRAMEWORK_NAMES
    assert "add_note" in names                                             # an app execution command still trains


def test_f_the_test_set_inventory_excludes_framework_commands_and_counts_only_app_planning_commands():
    from fastworkflow.train.planning_test_set import inventory
    from fastworkflow.planning_commands import resolve_field_sources, visible_planning_commands

    planning = set(inventory(LIBRARY).planning)
    assert not planning & FRAMEWORK_NAMES
    loaded = set(resolve_field_sources(LIBRARY).planning_commands)
    assert FRAMEWORK_NAMES - {"review_proposal"} <= loaded                 # the loader still holds them
    assert planning == set(visible_planning_commands(LIBRARY))
    assert len(planning) == len(loaded) - len(FRAMEWORK_NAMES - {"review_proposal"})


def test_f_the_scoring_listing_excludes_framework_commands():
    from fastworkflow import planning_commands

    assert not set(planning_commands.visible_planning_commands(LIBRARY)) & FRAMEWORK_NAMES

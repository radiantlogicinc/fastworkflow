"""Phase 9 step 5 (fix-3rsq.9.7): ``context_hierarchy_model.json`` is retired (design 3.9).

Parents come from the live chain (``Workflow.get_parent``), so a workflow that still has the
file loads, warns ONCE per workflow load that the file is ignored and can be deleted, and routes
exactly as it does without the file. The planner's dirtiness rules, the internal CME's honest
construction record, and the refusal of the retired ancestor arms are asserted here too.

Integration tests on private copies of ``tests/todo_list_workflow`` and
``fastworkflow/examples/messaging_app_4``. Nothing here trains: the planner is asked what a change
would retrain against placeholder artifact files, the same way ``test_not_here_selective.py`` does.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import pytest

import fastworkflow
from fastworkflow.command_context_model import CommandContextModel, get_workflow_info
from fastworkflow.command_metadata_api import CommandMetadataAPI
from fastworkflow.command_routing import RoutingDefinition, RoutingRegistry
from fastworkflow.context_navigation import collect_descend_edges
from fastworkflow.model_pipeline_training import (
    INTERNAL_CME_ARM,
    RESERVED_CLASS_ARMS,
    RETIRED_ARMS,
    arm_for_workflow,
    construction_records,
    experiment_reserved_class_arm,
    not_here_construction,
    resolve_reserved_class_arm,
)
from fastworkflow.train import artifact_versioning as av
from fastworkflow.train import not_here_pilot
from fastworkflow.train import score_not_here as scorer
from fastworkflow.train import selective_training as st
from fastworkflow.train.determinism import PROVENANCE_FILENAME
from fastworkflow.utils.logging import logger as fastworkflow_logger

REPO = Path(__file__).resolve().parent.parent
TODO = REPO / "tests" / "todo_list_workflow"
MESSAGING = REPO / "fastworkflow" / "examples" / "messaging_app_4"
WARNING_MARK = "context_hierarchy_model.json, which is ignored"
HIERARCHY = "context_hierarchy_model.json"
SEED = 42


@pytest.fixture(scope="module", autouse=True)
def initialized_fastworkflow():
    fastworkflow.init({})
    yield
    RoutingRegistry.clear_registry()


@pytest.fixture
def fastworkflow_logs(caplog):
    """The real ``fastWorkflow`` logger does not propagate to root, so attach caplog's handler."""
    previous_level = fastworkflow_logger.level
    fastworkflow_logger.setLevel(logging.DEBUG)
    fastworkflow_logger.addHandler(caplog.handler)
    try:
        yield caplog
    finally:
        fastworkflow_logger.removeHandler(caplog.handler)
        fastworkflow_logger.setLevel(previous_level)


def _copy(tmp_path: Path, source: Path, name: str) -> str:
    destination = tmp_path / name
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("___*", "__pycache__"))
    RoutingRegistry.clear_registry()
    return str(destination)


def _warnings(caplog) -> list[str]:
    return [record.getMessage() for record in caplog.records if WARNING_MARK in record.getMessage()]


def _rebuild(workflow: str) -> None:
    """Re-derive the routing artifacts after an edit, dropping every cached layer first."""
    RoutingRegistry.clear_registry()
    CommandContextModel.load(workflow)
    RoutingDefinition.build(workflow)
    RoutingRegistry.get_definition(workflow, load_cached=False)


# ---------------------------------------------------------------- the warning, and routing

def test_a_workflow_with_the_file_warns_once_per_load_and_says_it_can_be_deleted(
    tmp_path, fastworkflow_logs
):
    with_file = _copy(tmp_path, TODO, "with_file")
    assert (Path(with_file) / HIERARCHY).is_file()

    CommandContextModel.load(with_file)
    CommandContextModel.load(with_file)  # every get_parent call loads again
    get_workflow_info(with_file)

    messages = _warnings(fastworkflow_logs)
    assert len(messages) == 1, messages
    assert "ignored" in messages[0] and "can be deleted" in messages[0]
    assert "with_file" in messages[0]


def test_a_workflow_without_the_file_is_silent(tmp_path, fastworkflow_logs):
    without_file = _copy(tmp_path, TODO, "without_file")
    Path(without_file, HIERARCHY).unlink()

    CommandContextModel.load(without_file)
    get_workflow_info(without_file)

    assert _warnings(fastworkflow_logs) == []


def test_routing_is_the_same_with_and_without_the_file(tmp_path, fastworkflow_logs):
    with_file = _copy(tmp_path, TODO, "routing_with")
    without_file = _copy(tmp_path, TODO, "routing_without")
    Path(without_file, HIERARCHY).unlink()

    def routing(workflow: str):
        RoutingRegistry.clear_registry()
        model = CommandContextModel.load(workflow)
        crd = RoutingRegistry.get_definition(workflow, load_cached=False)
        contexts = sorted(model._command_contexts)
        commands = {context: model.commands(context) for context in contexts}
        display = CommandMetadataAPI.get_all_contexts_command_display_text(
            workflow, fastworkflow.get_internal_workflow_path("command_metadata_extraction"),
            "TodoItem",
        )
        return contexts, commands, dict(crd.contexts), display, collect_descend_edges(workflow)

    assert routing(with_file) == routing(without_file)
    assert len(_warnings(fastworkflow_logs)) == 1  # the file-bearing copy, once


def test_no_code_path_reads_the_file_so_a_malformed_one_raises_nothing(tmp_path, fastworkflow_logs):
    """A deliberately broken file: any reader would raise, so nothing may raise but the warning."""
    workflow = _copy(tmp_path, TODO, "malformed")
    Path(workflow, HIERARCHY).write_text("{ this is not json", encoding="utf-8")

    model = CommandContextModel.load(workflow)
    assert model.commands("TodoItem")
    info = get_workflow_info(workflow)
    assert "context_hierarchy_model" not in info
    assert CommandMetadataAPI.get_workflow_definition_display_text(info)
    RoutingRegistry.clear_registry()
    assert RoutingRegistry.get_definition(workflow, load_cached=False).contexts
    assert collect_descend_edges(workflow) is not None

    assert len(_warnings(fastworkflow_logs)) == 1


# ---------------------------------------------------------------- the planner's dirtiness rules

def _seed_utterance_cache(workflow: str) -> None:
    """A placeholder cache entry per execution command, which the planner requires for carry-forward.

    A carried head shares commands with a retrained head; without a cache entry for each one the
    planner (rightly) retrains everything, because a shared command would be redrawn from the LLM.
    """
    root = Path(workflow, "___command_info", "utterance_cache")
    root.mkdir(parents=True, exist_ok=True)
    context_commands, _ = st.build_context_maps(workflow, st.contexts_for_training(workflow))
    for commands in context_commands.values():
        for command in commands:
            Path(root, f"{fastworkflow.train.utterance_cache.slugify(command)}.baseline.json").write_text(
                json.dumps({"entries": {}}), encoding="utf-8")


def _publish_baseline(workflow: str) -> str:
    """Publish a complete-looking sparse_1 version. No model is trained."""
    RoutingRegistry.clear_registry()
    _seed_utterance_cache(workflow)
    contexts = sorted(st.contexts_for_training(workflow))
    version = av.new_version_id()
    for context_name in contexts:
        folder = av.version_dir(workflow, version) / av.context_folder_name(context_name)
        folder.mkdir(parents=True, exist_ok=True)
        for artifact in st.REQUIRED_CONTEXT_ARTIFACTS:
            (folder / artifact).touch()
    signature, _unresolved = st.compute_training_signature(
        workflow, contexts, seed=SEED, reserved_class_arm="sparse_1")
    fields = st.context_contribution_manifest_fields(signature, st._full_retrain(contexts, "baseline"))
    av.write_manifest(workflow, version, seed=SEED, **fields)
    st.save_training_signature(workflow, version, signature)
    av.publish_version(workflow, version)
    return version


def _plan(workflow: str, version: str):
    RoutingRegistry.clear_registry()
    plan, _signature = st.compute_training_plan(
        workflow, sorted(st.contexts_for_training(workflow)),
        changed_only=True, seed=SEED, carry_forward_from=version, reserved_class_arm="sparse_1")
    return plan


def _assert_every_head_dirty(plan, workflow: str) -> None:
    everything = set(st.contexts_for_training(workflow))
    assert plan.is_full_retrain is False, plan.global_reasons  # the per-head path, not a global one
    assert set(plan.contexts_to_train) == everything
    assert plan.contexts_carried_forward == []
    assert any("every head is retrained" in reason
               for reasons in plan.reasons.values() for reason in reasons)


def test_adding_a_command_dirties_every_head(tmp_path):
    workflow = _copy(tmp_path, MESSAGING, "add")
    version = _publish_baseline(workflow)
    source = Path(workflow) / "_commands" / "ChatRoom" / "list_users.py"
    (Path(workflow) / "_commands" / "ChatRoom" / "list_recent_users.py").write_text(
        source.read_text(encoding="utf-8"), encoding="utf-8")
    _rebuild(workflow)

    _assert_every_head_dirty(_plan(workflow, version), workflow)


def test_removing_a_command_dirties_every_head(tmp_path):
    workflow = _copy(tmp_path, MESSAGING, "remove")
    version = _publish_baseline(workflow)
    Path(workflow, "_commands", "ChatRoom", "list_users.py").unlink()
    _rebuild(workflow)

    _assert_every_head_dirty(_plan(workflow, version), workflow)


def test_changing_a_commands_seeds_dirties_only_the_heads_that_label_it(tmp_path):
    """Design 2.50: a seed edit retrains the edited command's own heads, and no others."""
    workflow = _copy(tmp_path, MESSAGING, "seeds")
    version = _publish_baseline(workflow)
    path = Path(workflow, "_commands", "ChatRoom", "list_users.py")
    text = path.read_text(encoding="utf-8")
    assert "plain_utterances = [" in text
    path.write_text(text.replace("plain_utterances = [",
                                 'plain_utterances = [\n        "a seed the planner test adds",', 1),
                    encoding="utf-8")
    _rebuild(workflow)

    plan = _plan(workflow, version)
    context_commands, _u = st.build_context_maps(workflow, st.contexts_for_training(workflow))
    own_heads = {ctx for ctx, commands in context_commands.items() if "ChatRoom/list_users" in commands}
    assert plan.is_full_retrain is False, plan.global_reasons
    assert own_heads and set(plan.contexts_to_train) == own_heads
    assert set(plan.contexts_carried_forward) == set(st.contexts_for_training(workflow)) - own_heads


def test_an_unchanged_workflow_still_carries_every_head_forward(tmp_path):
    """The control for the three rules above: without an edit, nothing is dirty."""
    workflow = _copy(tmp_path, MESSAGING, "unchanged")
    version = _publish_baseline(workflow)

    plan = _plan(workflow, version)

    assert plan.contexts_to_train == []
    assert set(plan.contexts_carried_forward) == set(st.contexts_for_training(workflow))


# ---------------------------------------------------------------- the CME record and the retired arms

def test_the_internal_cme_record_is_none_and_its_artifacts_still_load(tmp_path):
    cme = fastworkflow.get_internal_workflow_path("command_metadata_extraction")
    record = {"construction": "none", "reason": "internal workflow: no wildcard class"}

    assert arm_for_workflow(cme) == INTERNAL_CME_ARM
    assert not_here_construction(folderpath=cme) == record
    manifest_fields, run_metadata = construction_records(None, None, cme)
    assert manifest_fields == {"not_here": record}
    assert run_metadata["reserved_class_arm"] == INTERNAL_CME_ARM

    workflow = str(tmp_path / "cme_recorded")
    version = av.new_version_id()
    av.write_manifest(workflow, version, seed=SEED, **manifest_fields)
    assert av.construction_record_problem(workflow) is None

    # An artifact trained under the old record is still loadable: the check reads only that a
    # construction is recorded, so the CME does not need a retrain to load.
    old = str(tmp_path / "cme_old_record")
    old_version = av.new_version_id()
    av.write_manifest(old, old_version, seed=SEED, not_here={
        "construction": "ancestor", "ratio": 1.0,
        "reason": "internal workflow: keeps today's construction, no wildcard class"})
    assert av.construction_record_problem(old) is None


@pytest.mark.parametrize("arm", sorted(RETIRED_ARMS))
def test_a_retired_arm_is_refused_with_the_reason_everywhere_it_could_be_named(arm, tmp_path):
    assert arm not in RESERVED_CLASS_ARMS
    with pytest.raises(ValueError, match="retired"):
        resolve_reserved_class_arm(arm)
    with pytest.raises(ValueError, match="retired"):
        with experiment_reserved_class_arm(arm):
            pass  # pragma: no cover - refused on entry
    with pytest.raises(ValueError, match="retired"):
        arm_for_workflow(None, arm)
    assert arm not in not_here_pilot.ARM_CHOICES

    provenance = tmp_path / "armed" / "___command_info"
    provenance.mkdir(parents=True)
    (provenance / PROVENANCE_FILENAME).write_text(json.dumps(
        {"schema_version": 2, "commands": {}, "context_training": {}, "reserved_class_arm": arm}),
        encoding="utf-8")
    with pytest.raises(ValueError, match="retired"):
        scorer.check_arm(str(tmp_path / "armed"), None)


def test_the_sparse_arms_are_the_only_arms_left():
    assert sorted(RESERVED_CLASS_ARMS) == ["sparse_1", "sparse_8"]
    assert sorted(not_here_pilot.ARM_CHOICES) == ["sparse_1", "sparse_8"]

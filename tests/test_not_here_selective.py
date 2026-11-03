"""Phase 9 step 1: each arm's selective retrain, and the publish-time check on sparse records.

Nothing here trains. A baseline version is published with placeholder artifact files, as
tests/test_selective_training_integration.py does, and the planner is asked what a seed edit
would retrain. Phase 9 step 5a (design 2.50): a head records one sample of each foreign command, so a seed
edit dirties only the heads that label the command, and a planning-command edit dirties only the
heads that recorded one of its seeds.

Phase 9 step 5: the ``control`` arm is retired, so the planner cannot be asked about it; the
control-arm cases are replaced by the refusal, and the parent-axis helper is gone.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

import fastworkflow
from fastworkflow.command_routing import RoutingRegistry
from fastworkflow.train import artifact_versioning
from fastworkflow.train import selective_training as st
from fastworkflow.train import utterance_cache

REPO = Path(__file__).resolve().parent.parent
MESSAGING = REPO / "fastworkflow" / "examples" / "messaging_app_4"
LIBRARY = REPO / "fastworkflow" / "examples" / "library"
SEED = 42


@pytest.fixture(scope="module", autouse=True)
def initialized_fastworkflow():
    fastworkflow.init({})
    yield
    RoutingRegistry.clear_registry()


def _copy(tmp_path: Path, source: Path, name: str) -> str:
    destination = tmp_path / name
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("___*", "__pycache__"))
    RoutingRegistry.clear_registry()
    return str(destination)


def _seed_utterance_cache(workflow: str) -> None:
    """A placeholder cache entry per execution command, which the planner requires for carry-forward."""
    root = os.path.join(workflow, "___command_info", utterance_cache.CACHE_DIRNAME)
    os.makedirs(root, exist_ok=True)
    context_commands, _ = st.build_context_maps(workflow, st.contexts_for_training(workflow))
    for commands in context_commands.values():
        for command in commands:
            with open(os.path.join(root, f"{utterance_cache.slugify(command)}.baseline.json"), "w",
                      encoding="utf-8") as handle:
                json.dump({"entries": {}}, handle)


def _publish_baseline(workflow: str, arm, *, with_contributions: bool = False) -> str:
    """Publish a complete-looking version under *arm*. No model is trained."""
    RoutingRegistry.clear_registry()
    _seed_utterance_cache(workflow)
    contexts = sorted(st.contexts_for_training(workflow))
    version = artifact_versioning.new_version_id()
    for context_name in contexts:
        folder = artifact_versioning.version_dir(workflow, version) / artifact_versioning.context_folder_name(
            context_name)
        folder.mkdir(parents=True, exist_ok=True)
        for artifact in st.REQUIRED_CONTEXT_ARTIFACTS:
            (folder / artifact).touch()
    signature, _unresolved = st.compute_training_signature(
        workflow, contexts, seed=SEED, reserved_class_arm=arm)
    fields = {}
    if with_contributions:
        fields = st.context_contribution_manifest_fields(signature, st._full_retrain(contexts, "baseline"))
    artifact_versioning.write_manifest(workflow, version, seed=SEED, **fields)
    st.save_training_signature(workflow, version, signature)
    artifact_versioning.publish_version(workflow, version)
    return version


def _plan(workflow: str, version: str, arm):
    RoutingRegistry.clear_registry()  # the registry caches metadata per process; an edit must be re-read
    plan, _signature = st.compute_training_plan(
        workflow, sorted(st.contexts_for_training(workflow)),
        changed_only=True, seed=SEED, carry_forward_from=version, reserved_class_arm=arm)
    return plan


def _execution_command_files(workflow: str):
    commands_dir = Path(workflow) / "_commands"
    for path in sorted(commands_dir.rglob("*.py")):
        rel = path.relative_to(commands_dir)
        if any(part.startswith(("_", ".")) for part in rel.parts):
            continue
        text = path.read_text(encoding="utf-8")
        if "PlanGenerator" in text or "plain_utterances = [" not in text:
            continue
        yield path, str(rel.with_suffix("")).replace(os.sep, "/")


def _edit_seed(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace("plain_utterances = [",
                                 'plain_utterances = [\n        "seed added by the planner test",', 1),
                    encoding="utf-8")


def _edited_execution_command(workflow: str):
    """Edit one seed of a real execution command; return (its qualified name, file path)."""
    context_commands, _u = st.build_context_maps(workflow, st.contexts_for_training(workflow))
    known = set().union(*context_commands.values())
    for path, name in _execution_command_files(workflow):
        if name in known:
            _edit_seed(path)
            return name, path
    raise AssertionError("no execution command with a seed list was found to edit")


# ---------------------------------------------------------------- planner, messaging_app_4

def test_the_retired_control_arm_cannot_be_planned_or_published(tmp_path):
    """Phase 9 step 5 (was: a seed edit under control dirties only the heads that use it).

    The ancestor construction is gone. Planning under it is refused with the reason, before any
    artifact is read.
    """
    workflow = _copy(tmp_path, MESSAGING, "control")
    with pytest.raises(ValueError, match="retired"):
        _plan(workflow, "20990101T000000Z", arm="control")


def test_under_a_sparse_arm_a_seed_edit_dirties_only_its_own_heads(tmp_path):
    """Design 2.50: the edited command's own heads, and no others, are retrained."""
    workflow = _copy(tmp_path, MESSAGING, "sparse_edit")
    version = _publish_baseline(workflow, arm="sparse_8")
    command, _path = _edited_execution_command(workflow)

    plan = _plan(workflow, version, arm="sparse_8")

    context_commands, _u = st.build_context_maps(workflow, st.contexts_for_training(workflow))
    own_heads = {ctx for ctx, commands in context_commands.items() if command in commands}
    assert own_heads and own_heads != set(st.contexts_for_training(workflow))
    assert set(plan.contexts_to_train) == own_heads


def test_an_unchanged_sparse_baseline_carries_everything_forward(tmp_path):
    workflow = _copy(tmp_path, MESSAGING, "sparse_clean")
    version = _publish_baseline(workflow, arm="sparse_1")

    plan = _plan(workflow, version, arm="sparse_1")

    assert plan.contexts_to_train == []


def test_switching_arm_marks_every_head_dirty(tmp_path):
    workflow = _copy(tmp_path, MESSAGING, "switch")
    shipped_version = _publish_baseline(workflow, arm="sparse_1")

    plan = _plan(workflow, shipped_version, arm="sparse_8")

    assert set(plan.contexts_to_train) == set(st.contexts_for_training(workflow))
    # Phase 9 step 4: the arm is part of the header's not-here record, so a switch is a global
    # difference (a full retrain) rather than a per-context "arm changed" reason.
    assert plan.is_full_retrain
    assert any("not_here" in reason for reason in plan.global_reasons)


def test_the_sparse_sources_are_every_foreign_command_and_nothing_else(tmp_path):
    workflow = _copy(tmp_path, MESSAGING, "sources")
    contexts = sorted(st.contexts_for_training(workflow))
    signature, _u = st.compute_training_signature(workflow, contexts, seed=SEED, reserved_class_arm="sparse_8")
    context_commands, _x = st.build_context_maps(workflow, contexts)
    everything = set().union(*context_commands.values())
    for name, context in signature.contexts.items():
        own = set(context.label_space)
        assert set(context.wildcard_sources) == {
            c for c in everything if c not in own and c.split("/")[-1] != "wildcard"}
        assert context.reserved_class_arm == "sparse_8"


# ---------------------------------------------------------------- planner, library (planning commands)

def test_a_planning_seed_edit_dirties_only_the_head_that_labels_it(tmp_path):
    """Design 2.50, Phase 10 step 1: a planning command is a label of the heads on its lookup path.

    ``Book/loan_report`` is a label of the Book head, so its seed edit changes that head's training
    rows and only Book retrains. Every other head holds it as a not-here sample, and the edit adds a
    seed that no head recorded, so those heads' samples still hold and they carry forward. The
    shipped arm behaves the same way.
    """
    workflow = _copy(tmp_path, LIBRARY, "library_planning")
    sparse_version = _publish_baseline(workflow, arm="sparse_8")
    planning_file = Path(workflow) / "_commands" / "Book" / "loan_report.py"
    _edit_seed(planning_file)

    plan = _plan(workflow, sparse_version, arm="sparse_8")
    assert plan.contexts_to_train == ["Book"]
    assert sorted(plan.contexts_carried_forward) == ["*", "Library", "Member", "Record"]

    shipped_workflow = _copy(tmp_path, LIBRARY, "library_planning_shipped")
    shipped_version = _publish_baseline(shipped_workflow, arm="sparse_1")
    _edit_seed(Path(shipped_workflow) / "_commands" / "Book" / "loan_report.py")
    shipped = _plan(shipped_workflow, shipped_version, arm="sparse_1")
    assert shipped.contexts_to_train == ["Book"]


# ---------------------------------------------------------------- publish-time check

def test_the_publish_check_accepts_a_sparse_record_and_refuses_a_tampered_one(tmp_path):
    workflow = _copy(tmp_path, LIBRARY, "publish_check")
    version = _publish_baseline(workflow, arm="sparse_8", with_contributions=True)
    assert artifact_versioning.verify_version_consistency(workflow, version).problems == []

    manifest_path = artifact_versioning.version_dir(workflow, version) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    block = manifest[artifact_versioning.CONTEXT_CONTRIBUTIONS_KEY]
    some_context = sorted(block)[0]
    block[some_context]["wildcard_sources"] = block[some_context]["wildcard_sources"][:-1]
    manifest_path.write_text(json.dumps(manifest))
    problems = artifact_versioning.verify_version_consistency(workflow, version).problems
    assert any("wildcard-class sources" in problem for problem in problems)

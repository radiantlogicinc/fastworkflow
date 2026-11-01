"""Phase 9 step 4: sparse_1 is the shipped "not here" construction, and artifacts record it.

Integration tests on private copies of ``fastworkflow/examples`` workflows and on on-disk version
folders in ``tmp_path``. Nothing here trains: the trainer's record writes are reached through
``model_pipeline_training.construction_records`` (the same call ``train_workflow`` makes), the
planner through ``compute_training_plan`` (which decides before any model is fitted), and the load
refusal through ``ChatSession.start_workflow``, which refuses before it creates a workflow.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
from pathlib import Path

import pytest

import fastworkflow
from fastworkflow.command_routing import RoutingRegistry
from fastworkflow.model_pipeline_training import (
    INTERNAL_CME_ARM,
    NOT_HERE_SEEDS,
    RETIRED_ARMS,
    SHIPPED_ARM,
    construction_records,
    experiment_reserved_class_arm,
    not_here_construction,
    not_here_seed_rows,
    resolve_reserved_class_arm,
    select_escalation_rows,
    trained_command_labels,
)
from fastworkflow.train import artifact_versioning as av
from fastworkflow.train import selective_training
from tests.test_not_here_class import (
    MESSAGING,
    _head_inputs,
    _private_copy,
    _seed_cache,
    _wildcard_utterances,
)

SHIPPED_RECORD = {"construction": "sparse", "seeds_per_command": 1}


@pytest.fixture(scope="module", autouse=True)
def initialized_fastworkflow():
    fastworkflow.init({})
    yield
    RoutingRegistry.clear_registry()


@pytest.fixture
def messaging_copy(tmp_path) -> str:
    path = _private_copy(tmp_path, MESSAGING, "messaging_copy")
    yield path
    RoutingRegistry.clear_registry()


def _is_planning(crd, name: str) -> bool:
    try:
        return bool(crd.command_directory.get_command_metadata(name).is_planning)
    except KeyError:
        return False


def _write_version(workflow: str, version_id: str, *, manifest: dict | None,
                   signature: dict | None = None) -> None:
    """An on-disk version folder with an optional manifest and signature, and a current pointer."""
    version = av.version_dir(workflow, version_id)
    version.mkdir(parents=True, exist_ok=True)
    if manifest is not None:
        (version / av.MANIFEST_FILENAME).write_text(
            json.dumps({"version_id": version_id, **manifest}), encoding="utf-8")
    if signature is not None:
        (version / selective_training.SIGNATURE_FILENAME).write_text(
            json.dumps(signature), encoding="utf-8")
    pointer = av.pointer_path(workflow)
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(json.dumps({"version_id": version_id}), encoding="utf-8")


# ---------------------------------------------------------------- the shipped default

def test_shipped_construction_is_sparse_one_and_the_experiment_arms_keep_theirs():
    """Phase 9 step 5: the ancestor arms are retired. The sparse experiment arm keeps its record;
    the retired ones are refused by name, both through the override and by resolution."""
    assert NOT_HERE_SEEDS == 1
    assert SHIPPED_ARM == "sparse_1"
    assert resolve_reserved_class_arm() == "sparse_1"
    assert not_here_construction() == SHIPPED_RECORD
    with experiment_reserved_class_arm("sparse_8"):
        assert not_here_construction() == {"construction": "sparse", "seeds_per_command": 8}
    assert resolve_reserved_class_arm() == "sparse_1"
    for retired in ("control", "ratio_0_5"):
        assert retired in RETIRED_ARMS
        with pytest.raises(ValueError, match="retired"):
            with experiment_reserved_class_arm(retired):
                pass  # pragma: no cover - refused before the block runs


def test_no_override_keeps_label_sets_and_builds_exactly_the_sparse_one_rows(messaging_copy):
    RoutingRegistry.clear_registry()
    crd = RoutingRegistry.get_definition(messaging_copy)
    cache = _seed_cache(crd)
    command_cache = dict(cache)
    workflow = fastworkflow.Workflow.create(messaging_copy, workflow_id_str="not-here-production")
    checked = 0
    try:
        wildcard = _wildcard_utterances(workflow, crd)
        for ctx in sorted(crd.contexts):
            labels = trained_command_labels(ctx, crd)
            # heads stay flattened: own plus inherited labels, planning commands excluded
            flattened = {
                name for name in set(crd.contexts[ctx]) | set(crd.command_directory.core_command_names)
                if not _is_planning(crd, name)
            }
            assert labels == flattened, ctx
            with contextlib.redirect_stdout(io.StringIO()):
                context_utterances, own = _head_inputs(
                    workflow, crd, ctx, cache, command_cache, wildcard)
                shipped = select_escalation_rows(
                    ctx, crd, workflow, context_utterances=context_utterances)
                explicit = select_escalation_rows(
                    ctx, crd, workflow, context_utterances=context_utterances,
                    reserved_class_arm="sparse_1")
                picks, _available = not_here_seed_rows(
                    messaging_copy, crd, labels, NOT_HERE_SEEDS, context_name=ctx)
            assert shipped.rows == explicit.rows, ctx
            assert shipped.included == explicit.included, ctx
            # sparse: no ancestor sampling and no ratio fill
            assert shipped.budget is None and shipped.always_include_rows == [], ctx
            chosen = {row for rows in picks.values() for row in rows}
            assert set(shipped.rows) == chosen - context_utterances, ctx
            assert all(len(rows) <= 1 for rows in picks.values()), ctx
            assert not (set(shipped.rows) & context_utterances), ctx
            checked += 1
    finally:
        workflow.close()
    assert checked > 0


# ---------------------------------------------------------------- the records

def test_construction_records_are_written_to_manifest_and_provenance_form():
    manifest_fields, run_metadata = construction_records()
    assert manifest_fields == {"not_here": SHIPPED_RECORD}
    assert run_metadata["not_here"] == SHIPPED_RECORD
    assert run_metadata["reserved_class_arm"] == "sparse_1"
    assert "head_scope" not in run_metadata

    manifest_fields, run_metadata = construction_records("sparse_8", "scoped")
    assert manifest_fields == {"not_here": {"construction": "sparse", "seeds_per_command": 8}}
    assert run_metadata["reserved_class_arm"] == "sparse_8"
    assert run_metadata["head_scope"] == "scoped"


def test_signature_and_manifest_carry_the_record_and_a_round_trip_keeps_it(messaging_copy, tmp_path):
    RoutingRegistry.clear_registry()
    contexts = selective_training.contexts_for_training(messaging_copy)
    signature, unresolved = selective_training.compute_training_signature(
        messaging_copy, contexts, seed=7)
    assert signature.global_inputs()["not_here"] == SHIPPED_RECORD
    assert all(sig.reserved_class_arm == "sparse_1" for sig in signature.contexts.values())
    assert not any(unresolved.values()), "every context of the copy must be describable"

    workflow = str(tmp_path / "signed_copy")
    shutil.copytree(messaging_copy, workflow, ignore=shutil.ignore_patterns("___*"))
    version_id = "20990101T000000Z-aaaaaa"
    selective_training.save_training_signature(workflow, version_id, signature)
    loaded = selective_training.load_training_signature(workflow, version_id)
    assert loaded is not None and loaded.not_here == SHIPPED_RECORD

    fields = selective_training.context_contribution_manifest_fields(
        signature, selective_training.TrainingPlan())
    for name, record in fields[av.CONTEXT_CONTRIBUTIONS_KEY].items():
        assert record["reserved_class_arm"] == "sparse_1", name
        assert record["reserved_class_construction"] == "not_here", name

    manifest_fields, _ = construction_records()
    av.write_manifest(workflow, version_id, seed=7, **manifest_fields)
    assert av.read_manifest(workflow, version_id)["not_here"] == SHIPPED_RECORD
    assert av.construction_record_problem(workflow) is None


# ---------------------------------------------------------------- the load refusal

def test_load_refuses_a_version_without_the_record_and_names_the_folder(tmp_path):
    workflow = str(tmp_path / "old_app")
    _write_version(workflow, "20260101T000000Z-oldold", manifest={"seed": 1})
    problem = av.construction_record_problem(workflow)
    assert problem is not None
    assert "retrain with `fastworkflow train`" in problem
    assert workflow in problem
    with pytest.raises(av.StaleArtifactError, match=r"retrain with `fastworkflow train`"):
        av.require_construction_record(workflow)


def test_load_refuses_unversioned_legacy_artifacts_and_accepts_a_recorded_version(tmp_path):
    legacy = str(tmp_path / "legacy_app")
    (Path(legacy) / "___command_info" / "global").mkdir(parents=True)
    (Path(legacy) / "___command_info" / "global" / "threshold.json").write_text("{}")
    assert av.legacy_layout_in_use(legacy)
    assert "retrain with `fastworkflow train`" in av.construction_record_problem(legacy)

    recorded = str(tmp_path / "new_app")
    _write_version(recorded, "20990101T000000Z-newnew", manifest={"not_here": SHIPPED_RECORD})
    assert av.construction_record_problem(recorded) is None

    # a record that is not an object with a construction is no record
    malformed = str(tmp_path / "bad_app")
    _write_version(malformed, "20990101T000000Z-badbad", manifest={"not_here": "sparse"})
    assert "retrain with `fastworkflow train`" in av.construction_record_problem(malformed)


def test_a_workflow_with_no_artifacts_is_not_refused_here(tmp_path):
    empty = str(tmp_path / "untrained_app")
    os.makedirs(empty)
    assert av.construction_record_problem(empty) is None


def test_chat_session_refuses_an_old_artifact_before_it_creates_a_workflow(tmp_path):
    from fastworkflow.chat_session import ChatSession

    workflow = str(tmp_path / "old_chat_app")
    _write_version(workflow, "20260101T000000Z-oldchat", manifest={"seed": 1})
    with pytest.raises(av.StaleArtifactError, match=r"retrain with `fastworkflow train`"):
        ChatSession().start_workflow(workflow)


# ---------------------------------------------------------------- training is not blocked

def test_planner_full_retrains_over_an_old_artifact_instead_of_refusing(messaging_copy, tmp_path):
    """The planner reads the old baseline and retrains everything; it never raises the refusal."""
    RoutingRegistry.clear_registry()
    contexts = selective_training.contexts_for_training(messaging_copy)
    current, _unresolved = selective_training.compute_training_signature(
        messaging_copy, contexts, seed=7)
    old = current.model_dump()
    old.pop("not_here")  # an artifact trained before the record existed
    workflow = str(tmp_path / "old_planner_copy")
    shutil.copytree(messaging_copy, workflow, ignore=shutil.ignore_patterns("___*"))
    old_version = "20260101T000000Z-oldpln"
    _write_version(workflow, old_version, manifest={"seed": 7}, signature=old)

    assert av.construction_record_problem(workflow) is not None  # the runtime would refuse it
    plan, signature = selective_training.compute_training_plan(
        workflow, contexts, changed_only=True, seed=7,
        carry_forward_from=old_version, cache_mode="reuse")
    assert plan.is_full_retrain is True
    assert sorted(plan.contexts_to_train) == sorted(contexts)
    assert plan.contexts_carried_forward == []
    assert any("not_here" in reason for reason in plan.global_reasons)
    assert signature.not_here == SHIPPED_RECORD


def test_a_matching_recorded_baseline_carries_nothing_forward_under_the_record_either(messaging_copy, tmp_path):
    """Sanity for the rule above: with the record present and identical, the header does not force a retrain."""
    RoutingRegistry.clear_registry()
    contexts = selective_training.contexts_for_training(messaging_copy)
    current, _unresolved = selective_training.compute_training_signature(
        messaging_copy, contexts, seed=7)
    workflow = str(tmp_path / "recorded_planner_copy")
    shutil.copytree(messaging_copy, workflow, ignore=shutil.ignore_patterns("___*"))
    version = "20260101T000000Z-recrec"
    _write_version(workflow, version, manifest={"not_here": SHIPPED_RECORD},
                   signature=current.model_dump())
    plan, _signature = selective_training.compute_training_plan(
        workflow, contexts, changed_only=True, seed=7,
        carry_forward_from=version, cache_mode="reuse")
    assert not any("not_here" in reason for reason in plan.global_reasons)


# ---------------------------------------------------------------- the CME-path predicate

def test_the_automatic_cme_check_skips_only_the_package_and_its_cme_workflow(tmp_path):
    from fastworkflow.train.__main__ import _trains_the_fastworkflow_package

    package = fastworkflow.get_fastworkflow_package_path()
    cme = os.path.join(package, "_workflows", "command_metadata_extraction")
    assert _trains_the_fastworkflow_package(package)
    assert _trains_the_fastworkflow_package(cme)
    # an app that lives inside a checkout named "fastworkflow" is not the package: the CME is checked
    assert not _trains_the_fastworkflow_package(os.path.join(package, "examples", "library"))
    assert not _trains_the_fastworkflow_package(str(tmp_path / "fastworkflow" / "my_app"))


def test_every_execution_context_refuses_an_old_cme_before_it_builds_anything(tmp_path, monkeypatch):
    """The CME serves every session, so the context's constructor is the gate for every runtime path."""
    from fastworkflow.workflow_execution_context import WorkflowExecutionContext

    old_cme = str(tmp_path / "old_cme")
    _write_version(old_cme, "20260101T000000Z-oldcme", manifest={"seed": 1})
    monkeypatch.setattr(fastworkflow, "get_internal_workflow_path", lambda name: old_cme)
    with pytest.raises(av.StaleArtifactError, match=r"retrain with `fastworkflow train`"):
        WorkflowExecutionContext(run_as_agent=False)


def test_the_internal_cme_keeps_today_s_construction_with_no_wildcard_class(tmp_path, monkeypatch):
    """The CME heads get no sparse class: labels and rows are exactly today's, with no override.

    The real CME folder is never opened: the internal-CME path is pointed at a private copy, which
    is what the detection compares against.
    """
    from fastworkflow.model_pipeline_training import (
        arm_for_workflow,
        construction_records,
        is_internal_cme_folder,
        not_here_construction,
    )

    source = fastworkflow.get_internal_workflow_path("command_metadata_extraction")
    cme_copy = _private_copy(tmp_path, source, "cme_copy")
    RoutingRegistry.clear_registry()
    monkeypatch.setattr(fastworkflow, "get_internal_workflow_path", lambda name: cme_copy)
    assert is_internal_cme_folder(cme_copy)
    assert not is_internal_cme_folder(str(tmp_path / "some_app"))

    assert arm_for_workflow(cme_copy) == INTERNAL_CME_ARM
    assert arm_for_workflow(cme_copy, "sparse_8") == INTERNAL_CME_ARM  # no override reaches the CME
    # Phase 9 step 5: the honest record. The CME has no wildcard class, so it says so.
    record = not_here_construction(folderpath=cme_copy)
    assert record == {"construction": "none", "reason": "internal workflow: no wildcard class"}

    crd = RoutingRegistry.get_definition(cme_copy)
    cache = _seed_cache(crd)
    command_cache = dict(cache)
    workflow = fastworkflow.Workflow.create(cme_copy, workflow_id_str="cme-production")
    checked = 0
    try:
        wildcard = _wildcard_utterances(workflow, crd)
        for ctx in sorted(crd.contexts):
            flattened = {
                name for name in set(crd.contexts[ctx]) | set(crd.command_directory.core_command_names)
                if not _is_planning(crd, name)
            }
            assert trained_command_labels(ctx, crd) == flattened, ctx
            with contextlib.redirect_stdout(io.StringIO()):
                context_utterances, _own = _head_inputs(workflow, crd, ctx, cache, command_cache, wildcard)
                shipped = select_escalation_rows(
                    ctx, crd, workflow, context_utterances=context_utterances)
            assert shipped.rows == [], ctx
            assert shipped.included is False, ctx
            checked += 1
    finally:
        workflow.close()
    assert checked > 0

    manifest_fields, run_metadata = construction_records(None, None, cme_copy)
    assert manifest_fields["not_here"]["construction"] == "none"
    assert run_metadata["reserved_class_arm"] == INTERNAL_CME_ARM
    # the CME artifacts still carry a record, so they load
    recorded = str(tmp_path / "cme_recorded")
    _write_version(recorded, "20990101T000000Z-cmerec", manifest=manifest_fields)
    assert av.construction_record_problem(recorded) is None

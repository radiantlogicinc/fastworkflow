"""Focused tests for the training CLI's publication and call orchestration.

The no-op tests use real temporary workflow directories, artifact versions, pointers,
compatibility links, and retention. AST checks are limited to call wiring whose runtime
form would otherwise train the internal CME workflow or invoke model training.
"""

import ast
import inspect
import json
import os
import shutil
import threading
import textwrap
from pathlib import Path

import pytest

import fastworkflow
from fastworkflow.command_directory import CommandDirectory
from fastworkflow.command_routing import RoutingDefinition, RoutingRegistry
from fastworkflow import model_pipeline_training
from fastworkflow.model_pipeline_training import TrainingDataError
from fastworkflow.train import __main__ as train_orchestration
from fastworkflow.train import artifact_versioning as av
from fastworkflow.train import selective_training
from fastworkflow.train import training_report


_FASTWORKFLOW_GLOBAL_STATE_LOCK = threading.RLock()


def _make_version(
    workflow: Path,
    contexts: dict[str, float],
    *,
    previous_version: str | None = None,
) -> str:
    """Create a small but real artifact version with readable context markers."""
    version_id = av.new_version_id()
    for context_name, marker in contexts.items():
        context_dir = av.context_artifact_dir(str(workflow), version_id, context_name)
        (context_dir / "threshold.json").write_text(
            json.dumps({"confidence_threshold": marker}), encoding="utf-8"
        )
    av.write_manifest(
        str(workflow),
        version_id,
        seed=42,
        previous_version=previous_version,
    )
    return version_id


def _resolved_version(path: Path) -> str:
    return Path(os.path.realpath(path)).parent.name


def _function_tree(function) -> ast.FunctionDef:
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    node = tree.body[0]
    assert isinstance(node, ast.FunctionDef)
    return node


def _call_name(call: ast.Call) -> str:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        parts = [call.func.attr]
        value = call.func.value
        while isinstance(value, ast.Attribute):
            parts.append(value.attr)
            value = value.value
        if isinstance(value, ast.Name):
            parts.append(value.id)
        return ".".join(reversed(parts))
    return ""


@pytest.fixture
def cme_copy(tmp_path: Path, setup_test_environment):
    """A real CME copy guarded while it uses process-global routing caches.

    pytest-xdist workers are separate processes, so each has independent fastworkflow
    globals. The lock prevents in-process threaded callers from interleaving with the
    registry reset, while the session fixture initializes fastworkflow once per worker.
    """
    with _FASTWORKFLOW_GLOBAL_STATE_LOCK:
        package_root = Path(fastworkflow.__file__).parent
        source = package_root / "_workflows" / "command_metadata_extraction"
        consultation_source = package_root / "_workflows" / "consultation"
        copied_package_root = tmp_path / "fastworkflow"
        destination = (
            copied_package_root / "_workflows" / "command_metadata_extraction"
        )
        consultation_destination = copied_package_root / "_workflows" / "consultation"
        destination.parent.mkdir(parents=True)
        shutil.copytree(
            source,
            destination,
            ignore=shutil.ignore_patterns(
                "___command_info",
                "___workflow_contexts",
                "___convo_info",
                "__pycache__",
            ),
        )
        if consultation_source.is_dir():
            shutil.copytree(
                consultation_source,
                consultation_destination,
                ignore=shutil.ignore_patterns(
                    "___command_info",
                    "___workflow_contexts",
                    "___convo_info",
                    "__pycache__",
                ),
            )
            RoutingDefinition.build(str(consultation_destination))
            CommandDirectory.load(str(consultation_destination)).save()
        RoutingRegistry.clear_registry()
        RoutingDefinition.build(str(destination))
        CommandDirectory.load(str(destination)).save()
        try:
            yield copied_package_root, destination
        finally:
            RoutingRegistry.clear_registry()


def test_noop_plan_republishes_current_version_before_retention(tmp_path: Path):
    """A no-op train repairs missing and mixed reader paths without losing recovery."""
    workflow = tmp_path / "workflow"
    (workflow / "_commands").mkdir(parents=True)
    previous = _make_version(workflow, {"*": 0.1, "TodoItem": 0.2})
    current = _make_version(
        workflow,
        {"*": 0.9, "TodoItem": 0.8},
        previous_version=previous,
    )
    doomed = _make_version(workflow, {"*": 0.5})
    av.publish_version(str(workflow), current)

    info = av.command_info_root(str(workflow))
    os.unlink(info / "TodoItem")
    os.unlink(info / "global")
    os.symlink(
        os.path.relpath(av.version_dir(str(workflow), doomed) / "global", info),
        info / "global",
        target_is_directory=True,
    )
    assert not (info / "TodoItem").exists()
    assert _resolved_version(info / "global") == doomed

    train_orchestration._repair_noop_publication(str(workflow), current)

    assert av.resolve_current_version(str(workflow)) == current
    assert _resolved_version(info / "global") == current
    assert _resolved_version(info / "TodoItem") == current
    assert (
        json.loads((info / "global" / "threshold.json").read_text())[
            "confidence_threshold"
        ]
        == 0.9
    )
    assert {item.version_id for item in av.list_versions(str(workflow))} == {
        previous,
        current,
    }
    assert not av.version_dir(str(workflow), doomed).exists()


def test_noop_plan_fails_clearly_without_a_current_version(tmp_path: Path):
    workflow = tmp_path / "workflow"
    workflow.mkdir()

    with pytest.raises(
        TrainingDataError,
        match="no current artifact version is available",
    ):
        train_orchestration._repair_noop_publication(str(workflow), None)

    assert not av.versions_root(str(workflow)).exists()


def test_missing_training_report_refuses_publication_with_training_data_error():
    with pytest.raises(
        TrainingDataError,
        match="safety report could not be produced; refusing to publish models",
    ):
        train_orchestration._require_publishable_training_report(None)


def _stamp_internal_workflow_trained(workflow: Path) -> None:
    trainable_contexts = selective_training.contexts_for_training(str(workflow))
    command_info = workflow / "___command_info"
    for context_name in trainable_contexts:
        context_folder = (
            train_orchestration.GLOBAL_CONTEXT_FOLDER
            if context_name == "*"
            else context_name
        )
        context_dir = command_info / context_folder
        context_dir.mkdir(parents=True, exist_ok=True)
        for artifact_name in selective_training.REQUIRED_CONTEXT_ARTIFACTS:
            artifact_path = context_dir / artifact_name
            if artifact_name.endswith(".pth"):
                artifact_path.mkdir()
            else:
                artifact_path.write_text("{}", encoding="utf-8")
    version_id = av.new_version_id()
    signature, _ = selective_training.compute_training_signature(
        str(workflow), trainable_contexts
    )
    selective_training.save_training_signature(str(workflow), version_id, signature)
    av._write_pointer(str(workflow), version_id, "test")
    record_fields, _ = model_pipeline_training.construction_records(
        folderpath=str(workflow)
    )
    av.write_manifest(str(workflow), version_id, **record_fields)


def test_cme_readiness_checks_only_contexts_the_trainer_produces(cme_copy):
    package_root, cme_workflow = cme_copy
    trainable_contexts = selective_training.contexts_for_training(str(cme_workflow))
    assert trainable_contexts == {"*", "IntentDetection"}

    command_info = cme_workflow / "___command_info"
    for context_name in trainable_contexts:
        context_folder = (
            train_orchestration.GLOBAL_CONTEXT_FOLDER
            if context_name == "*"
            else context_name
        )
        context_dir = command_info / context_folder
        context_dir.mkdir(parents=True, exist_ok=True)
        for artifact_name in selective_training.REQUIRED_CONTEXT_ARTIFACTS:
            artifact_path = context_dir / artifact_name
            if artifact_name.endswith(".pth"):
                artifact_path.mkdir()
            else:
                artifact_path.write_text("{}", encoding="utf-8")

    # Artifacts alone are not enough: readiness is judged against the recorded
    # training signature (fix-a7r7).
    assert not train_orchestration.is_fast_workflow_trained(str(package_root))
    version_id = av.new_version_id()
    signature, _ = selective_training.compute_training_signature(
        str(cme_workflow), trainable_contexts
    )
    selective_training.save_training_signature(str(cme_workflow), version_id, signature)
    av._write_pointer(str(cme_workflow), version_id, "test")

    # The not-here construction record is part of readiness (Phase 9 step 4, design 2.23):
    # a version without it is refused at load, so the CME is not "trained" either.
    av.write_manifest(str(cme_workflow), version_id)
    assert not train_orchestration.is_fast_workflow_trained(str(package_root))
    record_fields, _ = model_pipeline_training.construction_records(
        folderpath=str(cme_workflow)
    )
    av.write_manifest(str(cme_workflow), version_id, **record_fields)

    consultation_workflow = package_root / "_workflows" / "consultation"
    if consultation_workflow.is_dir():
        _stamp_internal_workflow_trained(consultation_workflow)

    assert train_orchestration.is_fast_workflow_trained(str(package_root))
    assert not (command_info / "ErrorCorrection").exists()

    missing_artifact = command_info / "IntentDetection" / "largemodel.pth"
    shutil.rmtree(missing_artifact)
    assert not train_orchestration.is_fast_workflow_trained(str(package_root))

    missing_artifact.mkdir()
    newest_artifact_mtime = max(
        path.stat().st_mtime
        for context_name in trainable_contexts
        for path in (
            command_info
            / (
                train_orchestration.GLOBAL_CONTEXT_FOLDER
                if context_name == "*"
                else context_name
            )
        ).iterdir()
    )
    command_source = next((cme_workflow / "_commands").rglob("*.py"))
    os.utime(
        command_source,
        (newest_artifact_mtime + 10, newest_artifact_mtime + 10),
    )
    # A newer mtime with the same content is what a checkout produces; it must
    # not buy a paid CME retrain (fix-a7r7).
    assert train_orchestration.is_fast_workflow_trained(str(package_root))

    command_source.write_text(
        command_source.read_text(encoding="utf-8") + "\n# changed\n",
        encoding="utf-8",
    )
    assert not train_orchestration.is_fast_workflow_trained(str(package_root))


def test_cme_report_exempts_commands_in_an_untrained_declared_context(cme_copy):
    _package_root, cme_workflow = cme_copy

    report = training_report.build_report(str(cme_workflow))
    error_correction_rows = [
        row for row in report.rows if row.command_name.startswith("ErrorCorrection/")
    ]

    assert {row.command_name for row in error_correction_rows} == {
        "ErrorCorrection/abort",
        "ErrorCorrection/you_misunderstood",
    }
    assert all(
        row.status is training_report.RowStatus.EXCLUDED
        for row in error_correction_rows
    )
    assert not {
        row.command_name for row in error_correction_rows
    } & {row.command_name for row in report.blocking_rows}


def test_training_report_gate_runs_before_publication():
    """The tested None guard must remain wired into the production publish path."""
    source = inspect.getsource(train_orchestration.train_workflow)
    report = source.index("training_report.report_training_data(")
    gate = source.index("_require_publishable_training_report(report)")
    publish = source.index(
        "artifact_versioning.publish_version(workflow_path, version_id)"
    )
    assert report < gate < publish


def test_nonempty_full_retrain_plan_is_always_formatted():
    """The format call is unconditional after the empty-plan early return."""
    function = _function_tree(train_orchestration.train_workflow)
    parent: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(function):
        for child in ast.iter_child_nodes(node):
            parent[child] = node

    calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and _call_name(node) == "selective_training.format_plan"
    ]
    assert len(calls) == 1
    ancestor = parent.get(calls[0])
    while ancestor is not None and ancestor is not function:
        assert not isinstance(ancestor, ast.If)
        ancestor = parent.get(ancestor)

    full_plan = selective_training.TrainingPlan(
        contexts_to_train=["*", "TodoItem"],
        is_full_retrain=True,
        global_reasons=["all contexts are dirty"],
    )
    assert selective_training.format_plan(full_plan).startswith(
        "Training plan: full retrain of 2 context(s)."
    )


def test_trains_the_fastworkflow_package_is_a_path_test_not_a_substring_test(tmp_path):
    """A workflow inside a checkout named 'fastworkflow' must still get the CME check.

    The old substring test skipped the CME check for every such workflow, which made the
    CME unretrainable through `fastworkflow train <app>` there (fix-3rsq.9.6).
    """
    decoy = tmp_path / "fastworkflow" / "my_app"
    decoy.mkdir(parents=True)

    assert not train_orchestration._trains_the_fastworkflow_package(str(decoy))
    assert train_orchestration._trains_the_fastworkflow_package(
        os.path.realpath(fastworkflow.get_fastworkflow_package_path())
    )
    assert train_orchestration._trains_the_fastworkflow_package(
        fastworkflow.get_internal_workflow_path("command_metadata_extraction")
    )
    assert train_orchestration._trains_the_fastworkflow_package(
        fastworkflow.get_internal_workflow_path("consultation")
    )


def test_regeneration_flag_is_forwarded_only_inside_existing_cme_condition():
    function = _function_tree(train_orchestration.train_main)
    cme_if = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.If)
        and "is_fast_workflow_trained(fastworkflow_folderpath)"
        in ast.unparse(node.test)
    )
    condition = ast.unparse(cme_if.test)
    # Phase 9 step 4 (fix-3rsq.9.6): the old substring test ("'fastworkflow' not in
    # workflow_path") is replaced by _trains_the_fastworkflow_package, which compares
    # resolved paths and the internal CME folder. See test below for the behaviour.
    assert "not _trains_the_fastworkflow_package(workflow_path)" in condition
    assert "not is_fast_workflow_trained(fastworkflow_folderpath)" in condition

    calls = [
        node
        for node in ast.walk(cme_if)
        if isinstance(node, ast.Call) and _call_name(node) == "train_workflow"
    ]
    assert len(calls) == 1
    call = calls[0]
    assert ast.unparse(call.args[0]) == "fastworkflow_folderpath"
    keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in call.keywords}
    assert keywords["regenerate_utterances"] == "regenerate_utterances"
    assert "does not force" in inspect.getsource(train_orchestration.train_main)

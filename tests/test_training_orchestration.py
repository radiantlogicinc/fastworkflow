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
import textwrap
from pathlib import Path

import pytest

import fastworkflow
from fastworkflow.command_directory import CommandDirectory
from fastworkflow.command_routing import RoutingDefinition, RoutingRegistry
from fastworkflow.model_pipeline_training import TrainingDataError
from fastworkflow.train import __main__ as train_orchestration
from fastworkflow.train import artifact_versioning as av
from fastworkflow.train import selective_training
from fastworkflow.train import training_report


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
def cme_copy(tmp_path: Path):
    """A real CME workflow copy whose generated artifacts are safe to modify."""
    previous_env = dict(fastworkflow._env_vars)
    fastworkflow.init(env_vars={})
    package_root = Path(fastworkflow.__file__).parent
    source = package_root / "_workflows" / "command_metadata_extraction"
    copied_package_root = tmp_path / "fastworkflow"
    destination = (
        copied_package_root / "_workflows" / "command_metadata_extraction"
    )
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
    RoutingRegistry.clear_registry()
    RoutingDefinition.build(str(destination))
    CommandDirectory.load(str(destination)).save()
    yield copied_package_root, destination
    RoutingRegistry.clear_registry()
    fastworkflow.init(env_vars=previous_env)


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
        (context_dir / "threshold.json").write_text("{}", encoding="utf-8")

    assert train_orchestration.is_fast_workflow_trained(str(package_root))
    assert not (command_info / "ErrorCorrection").exists()

    (command_info / "IntentDetection" / "threshold.json").unlink()
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
    assert "'fastworkflow' not in workflow_path" in condition
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

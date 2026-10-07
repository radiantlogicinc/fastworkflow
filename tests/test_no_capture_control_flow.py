"""EXP-003's exit criterion: nothing branches on a captured value.

Architecture §17.3 makes this a stop condition rather than a style preference. A
confidence or a consequence class that is captured and then quietly consulted converts an instrumentation slice into an unmeasured behavior
change — and FW-REQ-021 clause 4 forbids any threshold at all until calibration
has been measured against realized correctness, which does not exist until G2A.
The failure mode is invisible in a diff review: one `if` in one emitter, in a file
whose whole purpose is recording, and the slice is no longer Phase 0.

Two halves, because either alone can pass while the property is false:

**Structural.** An AST pass over the three runtime files this slice touched,
asserting no captured value reaches a condition — including via a boolean
computed first and branched on later, which a naive `if` scan would miss. Modelled
on `test_module_defines_no_decision_function` in tests/test_observability/decision_signals.py.

**Behavioral.** The same command run under different capture configurations must
produce byte-identical outcomes. The structural test can only see the files it
knows about; this one would catch a read anywhere at all, because a read that
changes nothing observable is not the read the stop condition is about.
"""

from __future__ import annotations

import ast
import uuid
from contextlib import suppress
from pathlib import Path

import pytest

import fastworkflow
from fastworkflow import tracing
from fastworkflow.runtime_manifest import (
    CommandDeclaration,
    EffectContract,
    RuntimeManifest,
    clear_runtime_metadata,
    merge_and_gate,
    register_runtime_metadata,
)
from fastworkflow.workflow_execution_context import WorkflowExecutionContext

from tests.todo_list_workflow.application.todo_manager import TodoListManager

REPO_ROOT = Path(__file__).resolve().parents[1]

# The runtime files this slice added capture to. Deliberately not
# `observability/decision_signals.py`: it legitimately validates its own fields
# (a `ConsequenceAssessment` that could grade below its floor is the defect its
# validators exist to refuse), and tests/test_observability/decision_signals.py
# already pins that it exports nothing to branch on.
SCANNED_FILES = (
    "fastworkflow/command_executor.py",
    "fastworkflow/tracing.py",
    "fastworkflow/workflow_execution_context.py",
)

# Names that hold, produce, or address a captured value. A condition mentioning
# any of these is either a read of a capture or close enough to one that it
# should be argued for explicitly rather than slipped in.
CAPTURED_NAMES = frozenset(
    {
        # locals holding a projection
        "consequence",
        "child_calls",
        # fields of those projections
        "consequence_class",
        "effect_kind",
        "reversibility",
        "blast_radius",
        "decision_critical",
        "write_capable",
        # uncertainty, which fix-ajv.4 will emit into these same files
        "decision_uncertainty",
        "uncertainty",
        "calibrated",
        # the producers
        "consequence_assessment",
        "assess_consequence",
        "ConsequenceAssessment",
        "DecisionUncertainty",
        "UncertaintySignal",
        # the span attribute keys, so `attributes[ATTR_CONSEQUENCE]` counts too
        "ATTR_CONSEQUENCE",
        "ATTR_CHILD_CALLS",
    }
)


def _referenced_names(node: ast.AST) -> set[str]:
    """Every identifier a subtree mentions, as a name, attribute, or string key.

    String constants count because these values are dicts once projected, so
    `assessment["consequence_class"]` addresses the same thing `x.consequence_class`
    does and a check that saw only attributes would miss it.
    """
    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            names.add(child.id)
        elif isinstance(child, ast.Attribute):
            names.add(child.attr)
        elif isinstance(child, ast.Constant) and isinstance(child.value, str):
            names.add(child.value)
    return names


def _condition_subtrees(tree: ast.AST):
    """Every expression a branch is decided by, with a label for the message."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.While, ast.IfExp, ast.Assert)):
            yield type(node).__name__, node.lineno, node.test
        elif isinstance(node, ast.comprehension):
            for condition in node.ifs:
                yield "comprehension", condition.lineno, condition
        elif isinstance(node, ast.Match):
            yield "Match", node.lineno, node.subject
        elif isinstance(node, ast.Compare):
            # Anywhere, not only in a condition: computing
            # `risky = consequence_class == "high"` and branching on `risky`
            # later is the same read wearing a different name.
            yield "Compare", node.lineno, node


@pytest.mark.parametrize("relative_path", SCANNED_FILES)
def test_no_condition_reads_a_captured_value(relative_path):
    path = REPO_ROOT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"))

    offenders = [
        f"{relative_path}:{lineno} ({kind}) reads {sorted(hits)}"
        for kind, lineno, subtree in _condition_subtrees(tree)
        if (hits := _referenced_names(subtree) & CAPTURED_NAMES)
    ]

    assert offenders == [], (
        "a captured uncertainty/consequence/context value reaches control flow, "
        "which is EXP-003's exit criterion and architecture §17.3's stop "
        "condition:\n  " + "\n  ".join(offenders)
    )


def test_the_scan_would_actually_catch_a_read():
    """The detector, checked against a known-bad sample.

    A structural test that cannot fail is worse than no test, because it reports
    a property it never checked. Both shapes below are things a well-meaning
    change could introduce.
    """
    direct = ast.parse("if consequence['consequence_class'] == 'high':\n    pass\n")
    laundered = ast.parse(
        "risky = consequence_class == 'high'\nif risky:\n    pass\n"
    )

    for tree in (direct, laundered):
        hits = [
            _referenced_names(subtree) & CAPTURED_NAMES
            for _kind, _lineno, subtree in _condition_subtrees(tree)
        ]
        assert any(hits), "the scan missed a read it is supposed to catch"


# ----------------------------------------------------------------------
# Behavioral half
# ----------------------------------------------------------------------


LIST_COMMAND = "TodoListManager/list_todo_lists"


@pytest.fixture
def todo_workflow_path() -> str:
    return str(Path(__file__).parent.joinpath("todo_list_workflow").resolve())


@pytest.fixture
def initialized_fastworkflow():
    fastworkflow.init({})
    from fastworkflow.command_routing import RoutingRegistry

    RoutingRegistry.clear_registry()
    yield
    RoutingRegistry.clear_registry()


def _run_one_command(todo_workflow_path: str, tmp_path) -> fastworkflow.TurnOutput:
    workflow = fastworkflow.Workflow.create(
        todo_workflow_path,
        workflow_id_str=f"nocontrolflow-{uuid.uuid4().hex}",
    )
    ctx = WorkflowExecutionContext(run_as_agent=False)
    ctx.bind_app_workflow(workflow)
    workflow.root_command_context = TodoListManager(str(tmp_path / "todo_list.json"))
    try:
        return ctx.process_action_turn(
            fastworkflow.Action(command_name=LIST_COMMAND, command="list them")
        )
    finally:
        with suppress(Exception):
            ctx.close()


def test_a_declared_effect_contract_changes_no_outcome(
    initialized_fastworkflow, todo_workflow_path, tmp_path
):
    """`write` and undeclared grade differently and must execute identically.

    This is the strongest available statement of the stop condition: the two runs
    produce the most different consequence assessments the assessor can produce
    from a declaration, so if anything downstream consulted one, the answers would
    diverge.
    """
    undeclared = _run_one_command(todo_workflow_path, tmp_path)

    manifest = RuntimeManifest(
        schema_version=1,
        manifest_version="1.0.0",
        commands={
            LIST_COMMAND: CommandDeclaration(effect=EffectContract(kind="write"))
        },
    )
    register_runtime_metadata(
        todo_workflow_path, merge_and_gate(manifest, deployment_features={}, env={})
    )
    try:
        declared = _run_one_command(todo_workflow_path, tmp_path)
    finally:
        clear_runtime_metadata()

    assert declared.answer == undeclared.answer
    assert declared.success == undeclared.success
    assert declared.status == undeclared.status


def test_the_projection_helpers_return_data_not_decisions():
    """The helper never answers "should we proceed", at any argument.

    A helper that returned a bool would be the stop condition arriving disguised
    as a convenience, so the shape is pinned rather than left to review.
    """
    for value in (
        tracing.consequence_assessment(None, None),
        tracing.consequence_assessment("/no/such/workflow", "anything"),
    ):
        assert isinstance(value, dict)
        assert not isinstance(value, bool)

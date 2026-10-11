"""EXP-003's exit criterion: nothing branches on a captured value.

Architecture §17.3 makes this a stop condition rather than a style preference. A
confidence or a consequence class that is captured and then quietly consulted converts an instrumentation slice into an unmeasured behavior
change — and FW-REQ-021 clause 4 forbids any threshold at all until calibration
has been measured against realized correctness, which does not exist until G2A.
The failure mode is invisible in a diff review: one `if` in one emitter, in a file
whose whole purpose is recording, and the slice is no longer Phase 0.

**Structural.** An AST pass over the runtime files this slice touched, asserting
no captured value reaches a condition, including via a boolean computed first and
branched on later, which a naive `if` scan would miss. Modelled on
`test_module_defines_no_decision_function` in tests/test_observability/decision_signals.py.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

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
        "DecisionUncertainty",
        "UncertaintySignal",
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

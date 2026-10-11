"""Coverage for the uncertainty contract (arch §6.6.1).

These are capture-only records, so there is no behavior to assert — what is worth
testing is the set of ways a plausible-looking record can be wrong. Most of this
file is therefore about rejection:

* `True` entering a calibration curve as a confidence of 1.0, which Pydantic's
  lax bool-to-int coercion allowed until a `before` validator stopped it;
* a decision that carries neither a signal nor a reason it has none, which is
  indistinguishable from an uninstrumented one;
* entity content reaching a signal value, which would make the record unsafe to
  retain under the evidence capture profile.

Two tests assert conformance against *other files* rather than against this
module's own beliefs: the slot-binding vocabulary is checked against the literals
`parameter_extraction.py` actually emits, and the leaf-import constraint of arch
§22 is checked by reading the import statements. Both catch drift that a
self-consistent unit test cannot.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import get_args

import pytest
from pydantic import ValidationError

import fastworkflow.observability.decision_signals as decision_signals
from fastworkflow.observability.decision_signals import (
    SIGNAL_DOMAINS,
    SLOT_BINDING_SOURCES,
    DecisionKind,
    DecisionUncertainty,
    SignalKind,
    UncertaintySignal,
    ambiguity_set_size,
    classifier_confidence,
    classifier_topk_margin,
    fuzzy_distance,
    slot_binding_source,
)


# ----------------------------------------------------------------------
# Enum exhaustiveness: a new member must not land without its metadata
# ----------------------------------------------------------------------


def test_every_signal_kind_declares_a_domain():
    """A kind with no domain has no declared unit and no declared polarity.

    `fuzzy-score` is the reason this matters: its value is a distance, so a
    consumer that assumes "higher is better" from the name inverts the curve.
    Adding a kind without answering that question must fail here.
    """
    assert set(SIGNAL_DOMAINS) == set(get_args(SignalKind))


def test_message_intent_is_a_recorded_decision_kind():
    """Requirements §4.14: misclassifying a cancellation as an answer is a
    high-consequence decision and is recorded like any other."""
    kinds = set(get_args(DecisionKind))
    assert "message-intent" in kinds
    assert kinds == {
        "command-identity",
        "target-binding",
        "slot-binding",
        "branch-selection",
        "predicate-evaluation",
        "message-intent",
    }


def test_fuzzy_score_polarity_is_recorded_as_distance():
    """The one contract name that reads backwards, pinned so it stays documented."""
    domain = SIGNAL_DOMAINS["fuzzy-score"]
    assert domain.higher_is_more_confident is False
    assert "levenshtein" in domain.unit
    assert fuzzy_distance(0.28, signal_version="lev/1").kind == "fuzzy-score"


def test_enumerated_kinds_declare_no_polarity():
    """`llm` is not more confident than `stored_merge`; the order is undefined."""
    for kind in ("slot-binding-source", "predicate-evidence"):
        assert SIGNAL_DOMAINS[kind].higher_is_more_confident is None


# ----------------------------------------------------------------------
# Signal values: numeric or enumerated, never free text, never bool
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(
            lambda: UncertaintySignal(
                signal_id="s", signal_version="v", kind="classifier-confidence", value=True
            ),
            id="bool-true-as-confidence",
        ),
        pytest.param(
            lambda: UncertaintySignal(
                signal_id="s", signal_version="v", kind="classifier-confidence", value=False
            ),
            id="bool-false-as-confidence",
        ),
        pytest.param(
            lambda: ambiguity_set_size(True, signal_version="v"), id="bool-as-count"
        ),
    ],
)
def test_bool_is_never_a_signal_value(build):
    """`bool` subclasses `int`, and Pydantic's lax union coerces `True` to `1`.

    Measured: before the `before` validator existed, `value=True` was accepted
    and stored as a confidence of 1.0, so the obvious `after`-validator
    `isinstance(..., bool)` check was dead code that looked correct.
    """
    with pytest.raises(ValidationError):
        build()


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda: classifier_confidence(1.5, signal_version="v"), id="above-1"),
        pytest.param(lambda: classifier_confidence(-0.1, signal_version="v"), id="below-0"),
        pytest.param(
            lambda: classifier_topk_margin(2.0, signal_version="v"), id="margin-above-1"
        ),
        pytest.param(lambda: fuzzy_distance(1.7, signal_version="v"), id="distance-above-1"),
        pytest.param(lambda: ambiguity_set_size(-1, signal_version="v"), id="negative-count"),
    ],
)
def test_numeric_signals_stay_in_their_declared_range(build):
    with pytest.raises(ValidationError):
        build()


@pytest.mark.parametrize(
    "value",
    ["telepathy", "", "sara_doe_496", "user@example.com"],
)
def test_enumerated_signals_reject_anything_outside_the_vocabulary(value):
    """Also the entity-content guard: a uid is not a binding source."""
    with pytest.raises(ValidationError):
        slot_binding_source(value, signal_version="v")


def test_numeric_kinds_reject_free_text():
    with pytest.raises(ValidationError):
        UncertaintySignal(
            signal_id="s",
            signal_version="v",
            kind="classifier-confidence",
            value="sara_doe_496",
        )


def test_enumerated_kinds_reject_numbers():
    with pytest.raises(ValidationError):
        UncertaintySignal(
            signal_id="s", signal_version="v", kind="slot-binding-source", value=0.5
        )


def test_valid_signals_round_trip():
    assert classifier_confidence(0.87, signal_version="tiny/3").value == pytest.approx(0.87)
    assert ambiguity_set_size(4, signal_version="tiny/3").value == 4
    assert slot_binding_source("db_lookup", signal_version="pe/1").value == "db_lookup"


def test_signals_are_frozen_and_reject_unknown_fields():
    signal = classifier_confidence(0.5, signal_version="v")
    with pytest.raises(ValidationError):
        signal.value = 0.9
    with pytest.raises(ValidationError):
        UncertaintySignal(
            signal_id="s",
            signal_version="v",
            kind="classifier-confidence",
            value=0.5,
            threshold=0.7,
        )


def test_calibration_ref_is_reported_not_assumed():
    """FW-REQ-021 clause 5: an uncalibrated signal is recordable, not proof."""
    assert classifier_confidence(0.9, signal_version="v").calibrated is False
    assert (
        classifier_confidence(0.9, signal_version="v", calibration_ref="cal/1").calibrated
        is True
    )
    # A count states no probability, so there is nothing about it to calibrate.
    assert ambiguity_set_size(3, signal_version="v").calibration_ref is None


# ----------------------------------------------------------------------
# Decision records: signals, or a stated reason there are none
# ----------------------------------------------------------------------


def test_a_decision_without_signals_must_say_why():
    """Exit criterion 1. Silence and 'not instrumented' must not look alike."""
    with pytest.raises(ValidationError):
        DecisionUncertainty(decision_kind="command-identity", candidate_count=1)


def test_a_deterministic_resolution_records_no_confidence():
    """An exact match has no uncertainty; reporting 1.0 would be a measurement."""
    decision = DecisionUncertainty(
        decision_kind="command-identity",
        candidate_count=1,
        signals_absent_reason="deterministic-resolution",
    )
    assert decision.signals == ()
    assert decision.reducible is None


def test_signals_and_an_absence_reason_are_mutually_exclusive():
    with pytest.raises(ValidationError):
        DecisionUncertainty(
            decision_kind="command-identity",
            candidate_count=2,
            signals=(classifier_confidence(0.4, signal_version="v"),),
            signals_absent_reason="not-applicable",
        )


def test_decision_calibration_requires_every_signal_to_be_backed():
    partly = DecisionUncertainty(
        decision_kind="message-intent",
        candidate_count=4,
        signals=(
            classifier_confidence(0.42, signal_version="v", calibration_ref="cal/1"),
            classifier_topk_margin(0.03, signal_version="v"),
        ),
    )
    assert partly.calibrated is False


def test_candidate_count_cannot_be_negative():
    with pytest.raises(ValidationError):
        DecisionUncertainty(
            decision_kind="command-identity",
            candidate_count=-1,
            signals_absent_reason="not-applicable",
        )


# ----------------------------------------------------------------------
# Conformance against other files
# ----------------------------------------------------------------------


def test_slot_binding_vocabulary_matches_what_the_runtime_emits():
    """The enum is checked against `parameter_extraction.py`, not against itself.

    A new `extraction_method` added there without a member here would be dropped
    at capture time, which is invisible in a self-consistent unit test.
    """
    source = (
            Path(inspect.getfile(decision_signals)).parent.parent
        / "_workflows"
        / "command_metadata_extraction"
        / "parameter_extraction.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))
    emitted = {
        node.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
        for target in node.targets
        if isinstance(target, ast.Subscript)
        and isinstance(target.slice, ast.Constant)
        and target.slice.value == "extraction_method"
    }
    assert emitted, "found no extraction_method assignments to compare against"
    assert emitted <= SLOT_BINDING_SOURCES, (
        f"parameter_extraction.py emits {sorted(emitted - SLOT_BINDING_SOURCES)}, "
        "which SLOT_BINDING_SOURCES does not allow"
    )


def test_module_stays_a_leaf():
    """Arch §22: standard library and Pydantic only."""
    tree = ast.parse(Path(inspect.getfile(decision_signals)).read_text(encoding="utf-8"))
    fastworkflow_imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module
        and node.module.startswith("fastworkflow")
    }
    assert fastworkflow_imports == set()


def test_module_defines_no_decision_function():
    """EXP-003 exit criterion 2, at module scope.

    No threshold lives here and nothing returns a permission, so there is nothing
    for control flow to import and branch on. A function answering "should we
    proceed" would be the §17.3 stop condition arriving disguised as a helper.
    """
    tree = ast.parse(Path(inspect.getfile(decision_signals)).read_text(encoding="utf-8"))
    names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    forbidden = {
        name
        for name in names
        if any(
            token in name.lower()
            for token in ("should", "allow", "permit", "gate", "proceed", "threshold")
        )
    }
    assert not forbidden, f"decision-shaped helpers must not live here: {sorted(forbidden)}"

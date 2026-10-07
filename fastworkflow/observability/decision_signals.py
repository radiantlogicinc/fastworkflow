"""How unsure the runtime was, and what it would have cost to be wrong.

Architecture §6.6.1, satisfying the P0 (representation) clauses of FW-REQ-021.
Two records: `DecisionUncertainty` says how confidently the runtime committed to
one decision, and `ConsequenceAssessment` says what that decision risked.

**This module is capture-only, and that is a load-bearing property rather than a
phase-ordering accident.** Nothing here compares a value to a threshold, and
nothing here returns a decision, a branch, or a permission. The reason is
FW-REQ-021 clause 4: a threshold may not be set until calibration has been
measured against realized correctness, and that measurement does not exist until
G2A. A confidence captured and then quietly branched on would convert an
instrumentation slice into an unmeasured behavior change, which is why the
absence of any such branch is both an EXP-003 exit criterion and an architecture
§17.3 stop condition. There is deliberately no `should_proceed()` here to import.

Four things this module exists to get right, each measured against this codebase
rather than assumed:

**`fuzzy-score` is a distance, and the name lies about the direction.** The enum
value is fixed by architecture §6.6.1, so it is spelled `fuzzy-score` here, but
`find_best_matches` returns a *normalized Levenshtein distance* where lower means
a better match, and `intent_detection.py` passes `threshold=0.3` as a maximum.
Recording that under a name a reader will take for a similarity inverts every
calibration curve drawn from it. The polarity is therefore machine-readable in
`SIGNAL_DOMAINS`, not left to the field name.

**Unknown is not zero.** An absent effect contract yields `effect_kind="unknown"`,
and §6.6.1 requires that this reads as write-capable and high consequence. The
enum's ordering in `runtime_manifest._EFFECT_SEVERITY` already places `unknown`
above `read_only` for the same reason. Here it is enforced by a validator rather
than left to each caller's care, because "unknown quietly became `none`" is a
failure that produces a clean-looking record and no error.

**A read is not automatically cheap.** §4.15: reading a stale attribute that will
authorize a later revocation is not low consequence. Consequence is a property of
the candidate action *in its binding*, so `decision_critical` — whether anything
downstream depends on the result — escalates a read.

**A decision with no signals must say so.** Exit criterion 1 accepts an explicit
reason why no signal applies, but not silence: an exact-prefix match genuinely has
no uncertainty to report, while an uninstrumented emitter has uncertainty nobody
captured, and a record that omits both looks identical. `signals_absent_reason`
separates them, and is a closed vocabulary rather than free text so that the
§6.6.1 "no free text, no entity content" rule holds for the whole record.

Leaf module (arch §22): standard library, Pydantic, and `runtime_manifest` only —
`EffectKind` is imported rather than restated so that the two cannot drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from fastworkflow.runtime_manifest import EffectKind

# Version of the *shape* of these records. Bumped when a field or an enum member
# is added, because a calibration report joins signals across runs and needs to
# know that an absent kind means "this engine could not emit it" rather than "it
# did not occur". Distinct from the per-signal `signal_version`, which versions
# the *producer* of one value: retraining the intent classifier changes what a
# confidence of 0.8 means without changing this contract at all.
SIGNAL_CONTRACT_VERSION = 1

# Version of the default consequence assessor below. §6.6.1 requires this be
# recorded so that a reassessment is distinguishable from a behavior change.
ASSESSOR_VERSION = "default/1"

SignalKind = Literal[
    "classifier-confidence",
    "classifier-topk-margin",
    "fuzzy-score",
    "ambiguity-set-size",
    "slot-binding-source",
    "predicate-evidence",
]

# §6.6.1's five, plus message-intent from requirements §4.14: misclassifying a
# mid-task cancellation as an answer to the previous question is itself a
# high-consequence decision, and is recorded like any other.
DecisionKind = Literal[
    "command-identity",
    "target-binding",
    "slot-binding",
    "branch-selection",
    "predicate-evaluation",
    "message-intent",
]

ConsequenceClass = Literal["none", "low", "medium", "high", "critical"]
Reversibility = Literal["reversible", "compensable", "irreversible", "unknown"]
BlastRadius = Literal["single-entity", "multi-entity", "tenant-wide", "unknown"]

# Ordered worst-last so that escalation is an index comparison. Not a public
# contract: consumers read the class name.
_CONSEQUENCE_ORDER: tuple[ConsequenceClass, ...] = (
    "none",
    "low",
    "medium",
    "high",
    "critical",
)

# Exactly what `parameter_extraction.py` records as `extraction_method` today,
# plus `db_lookup`, which substitutes a value during validation and is therefore
# a binding source even though it is reported as a separate diagnostic. Adding a
# member is a `SIGNAL_CONTRACT_VERSION` bump: a calibration report that silently
# gains a category cannot be compared with the one before it.
SLOT_BINDING_SOURCES: frozenset[str] = frozenset(
    {"stored_merge", "xml_regex", "llm", "db_lookup"}
)

# Declared, not yet emitted: goal predicates arrive with the P1 planning ontology,
# so nothing in the runtime produces this kind today. It is enumerated now so the
# contract does not change shape when the first emitter lands.
PREDICATE_EVIDENCE_KINDS: frozenset[str] = frozenset(
    {"authoritative", "derived", "stale", "absent"}
)

# Why a decision legitimately carries no signal. Closed vocabulary, because a
# free-text reason is a place for entity content to end up.
SignalsAbsentReason = Literal[
    # Deterministic resolution: an exact command-name match has nothing to be
    # unsure about. The honest report is "no uncertainty", not "confidence 1.0",
    # which would enter a calibration curve as a real measurement.
    "deterministic-resolution",
    # The decision happened and no emitter captured it. This is the GAP-18 case
    # and reads as missing evidence, never as confidence.
    "not-instrumented",
    # The kind does not apply to this decision at all.
    "not-applicable",
]


@dataclass(frozen=True)
class SignalDomain:
    """What one signal kind's value may be, and which direction is confident."""

    value_type: type | frozenset[str]
    # None for the enumerated kinds, where "more confident" is not ordered.
    higher_is_more_confident: Optional[bool]
    unit: str


# The polarity table. A calibration report needs this to bin a signal against
# realized correctness; getting it from the field name is exactly the mistake the
# `fuzzy-score` entry documents.
SIGNAL_DOMAINS: dict[str, SignalDomain] = {
    "classifier-confidence": SignalDomain(float, True, "probability-0-1"),
    "classifier-topk-margin": SignalDomain(float, True, "probability-delta-0-1"),
    # Lower is a better match. See the module docstring.
    "fuzzy-score": SignalDomain(float, False, "normalized-levenshtein-distance-0-1"),
    # More candidates is more ambiguity.
    "ambiguity-set-size": SignalDomain(int, False, "count"),
    "slot-binding-source": SignalDomain(SLOT_BINDING_SOURCES, None, "enum"),
    "predicate-evidence": SignalDomain(PREDICATE_EVIDENCE_KINDS, None, "enum"),
}


class _Strict(BaseModel):
    """Reject unknown keys and mutation.

    Same reasoning as `runtime_manifest._Strict`: a typo'd field that parses is a
    field nobody notices is missing. Frozen additionally means a captured signal
    cannot be edited after the fact by the code being measured.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


class UncertaintySignal(_Strict):
    """One measured quantity bearing on one decision.

    Values are numeric or drawn from a closed vocabulary, per §6.6.1: signals
    carry no free text and no entity content, so that the whole record is safe to
    retain.
    """

    signal_id: str
    # Versions the *producer*. A retrained classifier emits the same `kind` with
    # a new `signal_version`, and FW-REQ-021 clause 13 requires thresholds be
    # re-validated when it changes.
    signal_version: str
    kind: SignalKind
    value: float | int | str
    # Points at the calibration record for this signal version. None means the
    # signal is uncalibrated, which FW-REQ-021 clause 5 permits to be recorded
    # and to support abstention, but never to justify proceeding.
    calibration_ref: Optional[str] = None

    @property
    def calibrated(self) -> bool:
        """Whether a calibration record backs this value.

        Deliberately not a threshold and not a permission — see the module
        docstring. It answers "may this number be reported as a probability",
        which is a property of the record, not a decision about the action.
        """
        return self.calibration_ref is not None

    @field_validator("value", mode="before")
    @classmethod
    def _reject_bool(cls, value: object) -> object:
        """`True` is not a confidence of 1.

        This has to run `before`: `bool` is a subclass of `int`, and Pydantic's
        lax coercion into the `float | int | str` union turns `True` into `1`
        before any `after` validator sees it — so the obvious `isinstance(...,
        bool)` check downstream is dead code that appears to work. Measured, not
        assumed: it accepted `value=True` as confidence 1.0 until this existed.
        """
        if isinstance(value, bool):
            raise ValueError(
                "an uncertainty signal value may not be a bool; True would be "
                "coerced to a confidence of 1"
            )
        return value

    @model_validator(mode="after")
    def _value_matches_kind(self) -> "UncertaintySignal":
        domain = SIGNAL_DOMAINS[self.kind]
        if isinstance(domain.value_type, frozenset):
            if not isinstance(self.value, str):
                raise ValueError(
                    f"signal kind '{self.kind}' takes one of "
                    f"{sorted(domain.value_type)}; got {type(self.value).__name__}"
                )
            if self.value not in domain.value_type:
                raise ValueError(
                    f"signal kind '{self.kind}' takes one of "
                    f"{sorted(domain.value_type)}; got '{self.value}'"
                )
            return self

        if not isinstance(self.value, (int, float)):
            raise ValueError(
                f"signal kind '{self.kind}' takes a number; got "
                f"{type(self.value).__name__}"
            )
        if domain.unit.endswith("0-1") and not 0.0 <= float(self.value) <= 1.0:
            raise ValueError(
                f"signal kind '{self.kind}' is normalized to [0,1]; got {self.value}"
            )
        if domain.unit == "count" and self.value < 0:
            raise ValueError(
                f"signal kind '{self.kind}' is a count; got {self.value}"
            )
        return self


class DecisionUncertainty(_Strict):
    """How confidently the runtime committed to one decision.

    `resolution_tier` on the execution record remains authoritative for *which*
    path ran (§6.6.1); this records how sure that path was.
    """

    decision_kind: DecisionKind
    signals: tuple[UncertaintySignal, ...] = ()
    candidate_count: int = Field(ge=0)
    # Whether more evidence gathered inside the task could lower this
    # (requirements §4.14). None means nobody assessed it — not "no".
    reducible: Optional[bool] = None
    signals_absent_reason: Optional[SignalsAbsentReason] = None

    @property
    def calibrated(self) -> bool:
        """Whether every signal present is backed by a calibration record.

        Vacuously true for a decision with no signals, which is why callers
        reporting calibration coverage read `signals` too.
        """
        return all(signal.calibrated for signal in self.signals)

    @model_validator(mode="after")
    def _absence_is_explained(self) -> "DecisionUncertainty":
        """Exit criterion 1: signals, or a stated reason there are none."""
        if self.signals and self.signals_absent_reason is not None:
            raise ValueError(
                f"decision '{self.decision_kind}' carries {len(self.signals)} "
                "signal(s) and a signals_absent_reason; the reason is for the "
                "empty case only"
            )
        if not self.signals and self.signals_absent_reason is None:
            raise ValueError(
                f"decision '{self.decision_kind}' records no uncertainty signal "
                "and no reason why none applies; pass "
                "signals_absent_reason='not-instrumented' if the emitter does "
                "not exist yet"
            )
        return self


class ConsequenceAssessment(_Strict):
    """What this candidate action risked if the decision was wrong.

    Per §4.15 this describes the action *in its binding*, not the command
    definition: the same command against a different entity can carry a
    different blast radius.
    """

    consequence_class: ConsequenceClass
    effect_kind: EffectKind
    reversibility: Reversibility
    blast_radius: BlastRadius
    # Whether anything downstream depends on this result. §4.15's stale-attribute
    # case: a read that authorizes a later revocation is not low consequence.
    decision_critical: bool
    # The task-level declaration from a supported-task contract (§4.8), carried
    # for correlation. The action-level evaluation is `consequence_class`.
    risk_class: Optional[str] = None
    assessor_version: str = ASSESSOR_VERSION

    @property
    def write_capable(self) -> bool:
        """Whether this action may write. `unknown` counts (§6.6.1)."""
        return self.effect_kind != "read_only"

    @model_validator(mode="after")
    def _unknown_is_never_free(self) -> "ConsequenceAssessment":
        """§6.6.1: unknown never resolves to `none`, and reads as high.

        Enforced on the model rather than trusted to `assess_consequence`,
        because a hand-built record is the case that would otherwise slip
        through — and it produces a plausible-looking row, not an error.
        """
        floor: Optional[ConsequenceClass] = None
        if self.effect_kind == "unknown":
            floor = "high"
        elif self.reversibility == "unknown" or self.blast_radius == "unknown":
            floor = "high"
        elif self.effect_kind == "write" and self.consequence_class == "none":
            floor = "low"

        if floor is not None and _rank(self.consequence_class) < _rank(floor):
            raise ValueError(
                f"consequence_class '{self.consequence_class}' is below the "
                f"'{floor}' floor implied by effect_kind='{self.effect_kind}', "
                f"reversibility='{self.reversibility}', "
                f"blast_radius='{self.blast_radius}'; an unknown contract is a "
                "reason for more caution, not less (arch §6.6.1)"
            )
        return self


# Where an action starts before its reversibility and blast radius are taken into
# account. `unknown` outranks `read_only` for the same reason it does in
# `runtime_manifest._EFFECT_SEVERITY`: an absent contract is a reason for more
# caution, not less (§6.6.1).
_EFFECT_BASE: dict[EffectKind, ConsequenceClass] = {
    "read_only": "low",
    "write": "medium",
    "unknown": "high",
}

# The floor each reversibility imposes; None leaves the base alone. `unknown`
# shares `irreversible`'s floor deliberately — not knowing whether an effect can
# be undone is planned for as though it cannot be.
_REVERSIBILITY_FLOOR: dict[Reversibility, Optional[ConsequenceClass]] = {
    "reversible": None,
    "compensable": "medium",
    "irreversible": "high",
    "unknown": "high",
}


def _rank(consequence_class: ConsequenceClass) -> int:
    return _CONSEQUENCE_ORDER.index(consequence_class)


def _raise_to(current: ConsequenceClass, floor: ConsequenceClass) -> ConsequenceClass:
    return current if _rank(current) >= _rank(floor) else floor


def _step_up(current: ConsequenceClass) -> ConsequenceClass:
    return _CONSEQUENCE_ORDER[min(_rank(current) + 1, len(_CONSEQUENCE_ORDER) - 1)]


def assess_consequence(
    *,
    effect_kind: EffectKind,
    reversibility: Reversibility = "unknown",
    blast_radius: BlastRadius = "unknown",
    decision_critical: bool = False,
    risk_class: Optional[str] = None,
) -> ConsequenceAssessment:
    """Grade one candidate action, conservatively and deterministically.

    Deterministic given its inputs and free of model calls, so a replay
    reproduces it exactly. It reads no threshold and returns no decision — the
    P1 decision table of §19.6 is the thing that eventually consumes this, and it
    does not exist yet.

    **`none` is unreachable from this assessor, by design.** Claiming an action
    has zero consequence is a claim about the world that a declared effect
    contract cannot support — a read still has privacy impact and may still feed
    a later write. The member stays in the enum for a workflow-supplied assessor
    that knows more than the manifest does; the default one never asserts it.

    Callers get the effect contract from `RuntimeMetadata.effect_kind()`, which
    already answers `unknown` for an undeclared command rather than guessing
    `read_only`.
    """
    consequence = _EFFECT_BASE[effect_kind]

    if floor := _REVERSIBILITY_FLOOR[reversibility]:
        consequence = _raise_to(consequence, floor)

    if blast_radius == "tenant-wide":
        consequence = _raise_to(consequence, "critical")
    elif blast_radius == "multi-entity":
        consequence = _step_up(consequence)
    elif blast_radius == "unknown":
        consequence = _raise_to(consequence, "high")

    # §4.15's stale-attribute case, and the reason a read is not automatically
    # cheap: the cost is carried by whatever acts on the result.
    if decision_critical:
        consequence = _step_up(consequence)

    return ConsequenceAssessment(
        consequence_class=consequence,
        effect_kind=effect_kind,
        reversibility=reversibility,
        blast_radius=blast_radius,
        decision_critical=decision_critical,
        risk_class=risk_class,
        assessor_version=ASSESSOR_VERSION,
    )


# ----------------------------------------------------------------------
# Constructors for the signals this runtime actually produces
# ----------------------------------------------------------------------
#
# One per emission site that exists today, so that the polarity and the unit are
# decided here once instead of at each call site. `signal_version` is required
# rather than defaulted: it identifies the trained artifact or matcher that
# produced the number, and a default would silently claim that every run's
# classifier was the same one.


def classifier_confidence(
    value: float, *, signal_version: str, calibration_ref: Optional[str] = None
) -> UncertaintySignal:
    """The winning label's probability from `CommandRouter.predict_with_details`."""
    return UncertaintySignal(
        signal_id="nlu.classifier.confidence",
        signal_version=signal_version,
        kind="classifier-confidence",
        value=value,
        calibration_ref=calibration_ref,
    )


def classifier_topk_margin(
    value: float, *, signal_version: str, calibration_ref: Optional[str] = None
) -> UncertaintySignal:
    """Gap between the top two labels. Small means the classifier nearly tied."""
    return UncertaintySignal(
        signal_id="nlu.classifier.topk_margin",
        signal_version=signal_version,
        kind="classifier-topk-margin",
        value=value,
        calibration_ref=calibration_ref,
    )


def fuzzy_distance(
    value: float, *, signal_version: str, calibration_ref: Optional[str] = None
) -> UncertaintySignal:
    """Normalized Levenshtein distance from `find_best_matches`.

    Named for what it is. The `kind` it carries is architecture §6.6.1's
    `fuzzy-score`, and `SIGNAL_DOMAINS` records that lower is the confident
    direction — this is the one signal whose contract name reads backwards.
    """
    return UncertaintySignal(
        signal_id="nlu.fuzzy.distance",
        signal_version=signal_version,
        kind="fuzzy-score",
        value=value,
        calibration_ref=calibration_ref,
    )


def ambiguity_set_size(value: int, *, signal_version: str) -> UncertaintySignal:
    """How many candidates the runtime could not choose between.

    No `calibration_ref`: a count is an observation, not a stated probability,
    so there is nothing about it to calibrate.
    """
    return UncertaintySignal(
        signal_id="nlu.ambiguity.set_size",
        signal_version=signal_version,
        kind="ambiguity-set-size",
        value=value,
        calibration_ref=None,
    )


def slot_binding_source(value: str, *, signal_version: str) -> UncertaintySignal:
    """Which mechanism bound a parameter — `extraction_method`, or `db_lookup`.

    An enumerated provenance rather than a confidence: `llm` and `stored_merge`
    are not more or less confident than each other in any ordered sense, which is
    why `SIGNAL_DOMAINS` gives this kind no polarity.
    """
    return UncertaintySignal(
        signal_id="nlu.slot.binding_source",
        signal_version=signal_version,
        kind="slot-binding-source",
        value=value,
        calibration_ref=None,
    )

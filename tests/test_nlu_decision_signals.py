"""Uncertainty signals at the NLU resolution boundary (bead fix-ajv.4).

EXP-003 Phase 0 is instrumentation: the runtime records how unsure it was and
routes exactly as it did before. Both halves need proof, and they need different
kinds of proof.

*What is recorded* is checked against real components (testing_rules.mdc): the
trained ``fastworkflow/examples/hello_world`` workflow drives the real matching
layers, the real ``find_best_matches`` scores real candidates, and the real
embedding cache is exercised with the real DistilBERT pipeline. The assembling
functions are pure functions of the capture bag, so most tiers can be covered
without a model at all — which is the point of building them that way.

*That nothing reads what is recorded* is checked statically, in the same spirit as
``test_observability/decision_signals.py::test_module_defines_no_decision_function``. The rule
enforced here is narrower than "no conditional anywhere" and deliberately so: a
recorder has to read facts to shape a record. What may never happen is a captured
*measurement* — a confidence, a distance, a similarity, an assembled
``DecisionUncertainty`` — being compared against anything, or used as a truth
value. A ``x is None`` presence check cannot express a threshold and is allowed;
``<``, ``>``, ``==`` and bare truthiness can, and are not.

The polarity trap has its own test. ``find_best_matches`` returns a normalized
Levenshtein DISTANCE where lower is a better match, while architecture §6.6.1
names the kind ``fuzzy-score``. A consumer that reads the direction off the name
inverts every curve drawn from it, so the direction is asserted here against real
matcher output rather than trusted to the field name.
"""

from __future__ import annotations

import ast
import inspect
import uuid
from pathlib import Path

import pytest
from dotenv import dotenv_values

import fastworkflow
from fastworkflow import tracing
from fastworkflow._workflows.command_metadata_extraction import intent_detection
from fastworkflow._workflows.command_metadata_extraction import parameter_extraction
from fastworkflow._workflows.command_metadata_extraction.intent_detection import (
    ESCALATION_OUTCOMES,
    ESCALATION_OUTCOME_ABSENT,
    ESCALATION_OUTCOME_HONORED,
    ESCALATION_OUTCOME_NOT_EVALUATED,
    ESCALATION_OUTCOME_SUPPRESSED_BY_AMBIGUITY,
    _FUZZY_MATCHER_VERSION,
    _FUZZY_PREMATCH_MAX_DISTANCE,
    CommandNamePrediction,
    command_identity_uncertainty,
    escalation_outcome_of,
)
from fastworkflow._workflows.command_metadata_extraction.parameter_extraction import (
    slot_binding_uncertainty,
)
from fastworkflow.command_executor import CommandExecutor
from fastworkflow.observability.decision_signals import (
    SIGNAL_DOMAINS,
    SLOT_BINDING_SOURCES,
    DecisionUncertainty,
    classifier_confidence,
)
from fastworkflow.nlu_labels import PARAMETER_VALUE_LABEL, WILDCARD_LABEL
from fastworkflow.utils.fuzzy_match import find_best_matches
from fastworkflow.workflow_execution_context import WorkflowExecutionContext

HELLO_WORLD = str(
    Path(__file__).parent.parent / "fastworkflow" / "examples" / "hello_world"
)

# The five layers `_predict_impl` can attribute a resolution to, plus the "nothing
# claimed it" case that leaves the attribute unset.
MATCHER_LAYERS = (
    "exact_prefix",
    "fuzzy_prematch",
    "embedding_cache",
    "classifier",
    "clarification_default",
    None,
)


@pytest.fixture(scope="module", autouse=True)
def initialized_fastworkflow():
    env = dotenv_values("fastworkflow/examples/fastworkflow.env")
    fastworkflow.init(dict(env))
    from fastworkflow.command_routing import RoutingRegistry

    RoutingRegistry.clear_registry()
    yield
    RoutingRegistry.clear_registry()


class RecordingTraceSink:
    """A real TraceSink implementation — the pluggable seam the design defines."""

    def __init__(self):
        self.spans: list[tracing.Span] = []

    def emit_span(self, span: tracing.Span) -> None:
        self.spans.append(span)

    def emit_turn_record(self, record) -> None:
        pass

    def record_conversation_label(self, *args) -> None:
        pass

    def named(self, name: str) -> list[tracing.Span]:
        return [s for s in self.spans if s.name == name]


@pytest.fixture
def hello_ctx():
    # `is_workflow_trained`, not "___command_info exists": an untrained
    # scaffold already has that directory (3.3 port adaptation).
    from fastworkflow.model_pipeline_training import is_workflow_trained

    if not is_workflow_trained(HELLO_WORLD)[0]:
        pytest.skip("hello_world is not trained on this machine")
    sink = RecordingTraceSink()
    wf = fastworkflow.Workflow.create(
        HELLO_WORLD,
        workflow_id_str=f"nlusig-{uuid.uuid4().hex}",
        workflow_context={"run_as_agent": True},
    )
    ctx = WorkflowExecutionContext(run_as_agent=True, trace_sink=sink)
    ctx.bind_app_workflow(wf)
    ctx._begin_turn("nlu decision signal test")
    ctx.push_active_workflow(wf)
    yield ctx, sink
    ctx.pop_active_workflow()


# The recorder wrappers skip their work when no span was opened; a real one stands
# in wherever the test is about what gets recorded rather than about that skip.
_ANY_SPAN = tracing.Span(span_id="test", trace_id="test", name=tracing.SPAN_NLU_INTENT)


def _kinds(uncertainty: DecisionUncertainty) -> list[str]:
    return [signal.kind for signal in uncertainty.signals]


def _signal(uncertainty: DecisionUncertainty, kind: str):
    matching = [s for s in uncertainty.signals if s.kind == kind]
    assert matching, f"no '{kind}' signal in {uncertainty!r}"
    return matching[0]


# ----------------------------------------------------------------------
# The fuzzy distance: recorded at all, and recorded with the right direction
# ----------------------------------------------------------------------


def test_fuzzy_distance_polarity_is_measured_not_assumed():
    """A better match produces a LOWER value, through the real matcher.

    The bead's polarity trap: the contract kind is `fuzzy-score` but the number is
    a distance. Asserting this against real `find_best_matches` output rather than
    against a hand-written constant is what makes it catch a future inversion at
    the emitter.
    """
    candidates = ["add_two_numbers", "what_can_i_do", "you_misunderstood"]

    exact, exact_distance = find_best_matches(
        "what_can_i_do", candidates, threshold=_FUZZY_PREMATCH_MAX_DISTANCE
    )
    typo, typo_distance = find_best_matches(
        "what_can_i_du", candidates, threshold=_FUZZY_PREMATCH_MAX_DISTANCE
    )
    assert exact == ["what_can_i_do"]
    assert typo == ["what_can_i_do"]
    assert exact_distance < typo_distance, "the better match must score lower"

    better = command_identity_uncertainty(
        {"matcher_layer": "fuzzy_prematch", "fuzzy_distance": exact_distance,
         "candidate_count": 1}
    )
    worse = command_identity_uncertainty(
        {"matcher_layer": "fuzzy_prematch", "fuzzy_distance": typo_distance,
         "candidate_count": 1}
    )
    assert _signal(better, "fuzzy-score").value < _signal(worse, "fuzzy-score").value

    # And the direction is machine-readable, so a calibration report does not have
    # to infer it from a field name that reads backwards.
    assert SIGNAL_DOMAINS["fuzzy-score"].higher_is_more_confident is False


def test_fuzzy_threshold_is_a_maximum_and_is_recorded_beside_the_distance(hello_ctx):
    """A distance with no threshold beside it cannot be binned.

    Also pins what the threshold means: `find_best_matches` admits a match at or
    below it and returns ([], None) above, so it is a maximum on a distance, not a
    minimum on a score.
    """
    candidates = ["add_two_numbers"]
    _matched, close = find_best_matches(
        "add_two_number", candidates, threshold=_FUZZY_PREMATCH_MAX_DISTANCE
    )
    assert close <= _FUZZY_PREMATCH_MAX_DISTANCE

    rejected, distance = find_best_matches(
        "completely_different_text",
        candidates,
        threshold=_FUZZY_PREMATCH_MAX_DISTANCE,
    )
    assert rejected == [] and distance is None

    ctx, sink = hello_ctx
    CommandExecutor.invoke_command(ctx, "what can i do")
    span = sink.named(tracing.SPAN_NLU_INTENT)[-1]
    assert span.attributes["matcher_layer"] == "fuzzy_prematch"
    assert span.attributes["fuzzy_threshold"] == _FUZZY_PREMATCH_MAX_DISTANCE
    assert span.attributes["fuzzy_distance"] <= _FUZZY_PREMATCH_MAX_DISTANCE


# ----------------------------------------------------------------------
# Every tier produces a well-formed DecisionUncertainty
# ----------------------------------------------------------------------


def _trace_for(matcher_layer):
    """The capture bag `_predict_impl` leaves behind for each tier."""
    if matcher_layer == "exact_prefix":
        return {"matcher_layer": "exact_prefix"}
    if matcher_layer == "fuzzy_prematch":
        return {
            "matcher_layer": "fuzzy_prematch",
            "fuzzy_distance": 0.125,
            "fuzzy_threshold": _FUZZY_PREMATCH_MAX_DISTANCE,
            "candidate_count": 2,
        }
    if matcher_layer == "embedding_cache":
        return {
            "matcher_layer": "embedding_cache",
            "cache_similarity": 0.91,
            "cache_similarity_threshold": 0.85,
            "candidate_count": 1,
        }
    if matcher_layer == "classifier":
        return {
            "matcher_layer": "classifier",
            "classifier": {
                "model_tier": "tiny",
                "confidence": 0.73,
                "ambiguous_threshold": 0.4,
                "confident": True,
                "top_label": "add_two_numbers",
                "topk_labels": ["add_two_numbers", "what_can_i_do"],
            },
            "classifier_signal_version": "intent-classifier/unversioned/global/tiny",
            "candidate_count": 1,
        }
    if matcher_layer == "clarification_default":
        return {"matcher_layer": "clarification_default", "candidate_count": 1}
    return {"fuzzy_distance": None, "candidate_count": 0}


@pytest.mark.parametrize("matcher_layer", MATCHER_LAYERS)
def test_every_tier_satisfies_the_signal_or_reason_rule(matcher_layer):
    """Exit criterion 1: signals, or a stated reason there are none — never both,
    never neither. The model validator enforces it; this proves each tier reaches a
    combination it accepts, which construction alone does not."""
    uncertainty = command_identity_uncertainty(_trace_for(matcher_layer))
    assert uncertainty.decision_kind == "command-identity"
    assert bool(uncertainty.signals) != bool(uncertainty.signals_absent_reason)
    assert uncertainty.candidate_count >= 0
    # Serializable, because it goes onto a span attribute bag as JSON.
    assert uncertainty.model_dump(mode="json")["decision_kind"] == "command-identity"


def test_exact_prefix_is_deterministic_and_states_no_confidence():
    """An exact command-name match has nothing to be unsure about.

    Confidence 1.0 here would enter a calibration curve as a real measurement of a
    classifier that never ran. The record must therefore carry no signal at all,
    and must stay distinguishable from the record that says 1.0 — those are the
    two rows §6.6.1 needs a report to be able to tell apart.
    """
    uncertainty = command_identity_uncertainty(_trace_for("exact_prefix"))
    assert uncertainty.signals == ()
    assert uncertainty.signals_absent_reason == "deterministic-resolution"
    assert uncertainty.candidate_count == 1
    assert uncertainty.model_dump(mode="json")["signals"] == []

    stated_certainty = DecisionUncertainty(
        decision_kind="command-identity",
        signals=(classifier_confidence(1.0, signal_version="hypothetical/1"),),
        candidate_count=1,
    )
    assert stated_certainty != uncertainty
    assert stated_certainty.signals_absent_reason is None


def test_fuzzy_tier_carries_the_distance_and_the_ambiguity_it_resolved():
    uncertainty = command_identity_uncertainty(_trace_for("fuzzy_prematch"))
    assert _kinds(uncertainty) == ["fuzzy-score", "ambiguity-set-size"]
    assert _signal(uncertainty, "fuzzy-score").value == 0.125
    # Two candidates tied and the clarification stages take [0]; the record says so
    # rather than presenting the pick as unambiguous.
    assert _signal(uncertainty, "ambiguity-set-size").value == 2
    assert uncertainty.candidate_count == 2
    assert _signal(uncertainty, "fuzzy-score").signal_version == _FUZZY_MATCHER_VERSION


def test_classifier_tier_carries_confidence_and_a_producer_version():
    uncertainty = command_identity_uncertainty(_trace_for("classifier"))
    confidence = _signal(uncertainty, "classifier-confidence")
    assert confidence.value == 0.73
    # signal_version identifies the artifact, so a retrained classifier's 0.73 is
    # distinguishable from this one's.
    assert confidence.signal_version.startswith("intent-classifier/")
    assert confidence.calibrated is False, "no calibration record exists yet (G2A)"
    # No top-k margin: predict_with_details returns label NAMES, not their scores.
    assert "classifier-topk-margin" not in _kinds(uncertainty)


def test_embedding_cache_reports_an_uncapturable_measurement_honestly():
    """The cosine similarity decided this, and §6.6.1's SignalKind enum has no
    member for it — so the structured record says "not-instrumented" rather than
    silently looking like a decision nobody measured."""
    uncertainty = command_identity_uncertainty(_trace_for("embedding_cache"))
    assert uncertainty.signals == ()
    assert uncertainty.signals_absent_reason == "not-instrumented"


def test_unmatched_utterance_records_zero_candidates_and_missing_evidence():
    """No matcher claimed it. find_best_matches discards the distance above its
    threshold, so the one measurement taken is not recoverable here."""
    uncertainty = command_identity_uncertainty(_trace_for(None))
    assert uncertainty.candidate_count == 0
    assert uncertainty.signals_absent_reason == "not-instrumented"


def test_ambiguity_is_recorded_as_reducible():
    """Requirements §4.14: the runtime is about to ask, and the answer resolves it."""
    trace = _trace_for("classifier")
    trace["candidate_count"] = 3
    trace["candidates"] = ["a", "b", "c"]
    trace["reducible"] = True
    uncertainty = command_identity_uncertainty(trace)
    assert uncertainty.reducible is True
    assert _signal(uncertainty, "ambiguity-set-size").value == 3

    # Absent means nobody assessed it, which is not the same as "no".
    assert command_identity_uncertainty(_trace_for("exact_prefix")).reducible is None


# ----------------------------------------------------------------------
# The escalation outcome, as a first-class value
# ----------------------------------------------------------------------


def test_escalation_outcome_vocabulary_is_closed():
    assert ESCALATION_OUTCOMES == {
        ESCALATION_OUTCOME_NOT_EVALUATED,
        ESCALATION_OUTCOME_ABSENT,
        ESCALATION_OUTCOME_HONORED,
        ESCALATION_OUTCOME_SUPPRESSED_BY_AMBIGUITY,
    }


def test_escalation_outcome_uses_the_real_label_vocabulary():
    """`wildcard` escalates; `parameter_value` does not, because a bare value says
    nothing about whether an ancestor can serve the utterance (nlu_labels.py)."""
    assert escalation_outcome_of([WILDCARD_LABEL]) == ESCALATION_OUTCOME_HONORED
    assert escalation_outcome_of([PARAMETER_VALUE_LABEL]) == ESCALATION_OUTCOME_ABSENT
    assert escalation_outcome_of(["add_two_numbers"]) == ESCALATION_OUTCOME_ABSENT
    assert (
        escalation_outcome_of([WILDCARD_LABEL, "add_two_numbers", "what_can_i_do"])
        == ESCALATION_OUTCOME_SUPPRESSED_BY_AMBIGUITY
    )
    # Fully qualified labels compare on the bare name, as `label_of` does.
    assert (
        escalation_outcome_of([f"ChatRoom/{WILDCARD_LABEL}"])
        == ESCALATION_OUTCOME_HONORED
    )


def test_suppression_is_exactly_what_the_ambiguity_prompt_does():
    """The outcome name is only worth recording if it describes what happened.

    `suppressed_by_ambiguity` claims the user is offered the local candidates and
    the escalation signal goes nowhere; the real message formatter is asked.
    """
    predictions = [WILDCARD_LABEL, "add_two_numbers", "what_can_i_do"]
    assert (
        escalation_outcome_of(predictions)
        == ESCALATION_OUTCOME_SUPPRESSED_BY_AMBIGUITY
    )
    assert CommandNamePrediction.escalation_signals_in(predictions) == [WILDCARD_LABEL]
    message = CommandNamePrediction._formulate_ambiguous_command_error_message(
        predictions, run_as_agent=True
    )
    assert WILDCARD_LABEL not in message
    assert "add_two_numbers" in message


def test_honored_escalation_resolves_to_no_local_command():
    """`honored` claims the parent-chain walk is reached; the walk is driven by
    `command_name=None`, so that is what the resolver must return."""
    assert escalation_outcome_of([WILDCARD_LABEL]) == ESCALATION_OUTCOME_HONORED
    assert (
        CommandNamePrediction.resolve_fully_qualified_command_name(
            WILDCARD_LABEL, {"add_two_numbers": "add_two_numbers"}
        )
        is None
    )


# ----------------------------------------------------------------------
# Slot binding
# ----------------------------------------------------------------------


def test_slot_binding_source_comes_from_the_allowed_vocabulary():
    for method in ("stored_merge", "xml_regex", "llm"):
        uncertainty = slot_binding_uncertainty({"extraction_method": method})
        assert uncertainty.decision_kind == "slot-binding"
        source = _signal(uncertainty, "slot-binding-source")
        assert source.value == method
        assert source.value in SLOT_BINDING_SOURCES
        assert source.signal_version.startswith("param-extraction/")


def test_a_hook_is_a_binding_source_only_when_it_rewrote_the_value():
    """`is_valid` alone means the hook agreed with what was extracted; only a
    rewrite (`changed_fields`) makes validation_hook the thing that bound the slot
    (fix-3rsq.one-hook: the hook replaces db_lookup)."""
    def hook(**overrides):
        verdict = {"ran": True, "is_valid": True, "message": None,
                   "changed_fields": [], "refused_fields": []}
        verdict.update(overrides)
        return verdict

    agreed = slot_binding_uncertainty({
        "extraction_method": "llm",
        "validation_hook": hook(),
    })
    assert [s.value for s in agreed.signals] == ["llm"]

    rewrote = slot_binding_uncertainty({
        "extraction_method": "llm",
        "validation_hook": hook(changed_fields=["user_name"]),
    })
    assert [s.value for s in rewrote.signals] == ["llm", "validation_hook"]


def test_a_refusal_binds_nothing_and_its_suggestions_are_words_not_candidates():
    """A refusal's suggestions live in its message; the extractor binds one value per slot and
    enumerates no structured alternatives, so the candidate set stays one and the slot is reducible."""
    uncertainty = slot_binding_uncertainty({
        "extraction_method": "llm",
        "validation_hook": {"ran": True, "is_valid": False,
                            "message": "Did you mean Alice or Bob?",
                            "changed_fields": [], "refused_fields": ["user_name"]},
        "invalid_fields": ["user_name"],
    })
    assert [s.value for s in uncertainty.signals] == ["llm"]
    assert uncertainty.candidate_count == 1
    assert uncertainty.reducible is True


def test_capture_failure_degrades_instead_of_failing_the_turn():
    """`tracing` never lets a broken recorder cost a turn; this assembly runs just
    outside that guard, so it carries the same promise.

    The pure functions still raise — that is how a contract violation surfaces in a
    test — and only the span-attribute wrappers absorb it, into a value that reads
    as neither "no decision" nor "recorded before this existed".
    """
    # matcher_layer says the fuzzy tier decided, but the distance it decided on is
    # missing: a recorder bug, and exactly the shape of one.
    broken = {"matcher_layer": "fuzzy_prematch", "candidate_count": 1}
    with pytest.raises(KeyError):
        command_identity_uncertainty(broken)
    assert intent_detection._recorded_command_identity_uncertainty(
        _ANY_SPAN, broken
    ) == {"capture_error": "KeyError"}

    # A well-formed bag still produces a well-formed record through the wrapper.
    recorded = intent_detection._recorded_command_identity_uncertainty(
        _ANY_SPAN, _trace_for("exact_prefix")
    )
    assert DecisionUncertainty(**recorded).signals_absent_reason == (
        "deterministic-resolution"
    )

    # And on the slot-binding side, a genuine "no decision" stays distinguishable
    # from a capture failure.
    assert (
        parameter_extraction._recorded_slot_binding_uncertainty(_ANY_SPAN, {}) is None
    )
    assert parameter_extraction._recorded_slot_binding_uncertainty(
        _ANY_SPAN,
        {"extraction_method": "xml_regex", "validation_hook": "not-a-dict-of-outcomes"},
    ) == {"capture_error": "AttributeError"}


def test_no_span_means_no_assembly():
    """`start_span` declines without a sink or an open turn, and `end_span` then
    throws its attributes away. Building a record for it is overhead against the
    epic's capture-budget stop condition, and it is what let a stubbed classifier's
    empty details dict reach a recorder at all."""
    assert (
        intent_detection._recorded_command_identity_uncertainty(
            None, _trace_for("classifier")
        )
        is None
    )
    assert (
        parameter_extraction._recorded_slot_binding_uncertainty(
            None, {"extraction_method": "llm"}
        )
        is None
    )


def test_a_router_that_reports_no_tier_cannot_fail_a_turn():
    """`predict_with_details`'s dict belongs to `CommandRouter`, and the seam the
    fuzzy-tie and escalation tests inject through returns it empty on purpose. The
    signal version has to survive that from inside the resolution path, where an
    exception is a lost turn rather than a missing attribute."""
    version = intent_detection._classifier_signal_version(
        f"{HELLO_WORLD}/___command_info/*",
        {}.get("model_tier", intent_detection._UNKNOWN_MODEL_TIER),
    )
    assert version.endswith(f"/{intent_detection._UNKNOWN_MODEL_TIER}")
    assert version.startswith("intent-classifier/")


def test_no_binding_decision_produces_no_record():
    """A command with no parameters class binds no slots, and an extraction that
    raised before choosing a mechanism made no decision. Neither is an
    uninstrumented decision, so neither gets a record claiming to be one."""
    assert slot_binding_uncertainty({}) is None
    assert slot_binding_uncertainty({"retry_round": False}) is None


def test_every_extraction_method_the_emitter_writes_is_a_known_source():
    """Guards the same drift `test_observability/decision_signals.py` guards, from this side: a
    new `extraction_method` literal must be added to SLOT_BINDING_SOURCES, and the
    correct fix if this fails is to widen the vocabulary deliberately, not here."""
    tree = ast.parse(
        Path(inspect.getfile(parameter_extraction)).read_text(encoding="utf-8")
    )
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
    assert emitted == {"stored_merge", "xml_regex", "llm"}
    assert emitted <= SLOT_BINDING_SOURCES


# ----------------------------------------------------------------------
# Phase 0: routing is unchanged
# ----------------------------------------------------------------------


class TestRoutingIsUnchanged:
    """The most important tests here. Capture is worthless if it moved a route.

    Each utterance pins the layer that claims it AND the command it resolves to,
    against the real trained hello_world models. The first two duplicate assertions
    in `test_nlu_span_emission.py` on purpose: that file was written before this
    change, so its passing is the before/after comparison, and repeating the
    outcomes here keeps them from being weakened in one place only.
    """

    def test_exact_prefix_still_routes_to_add_two_numbers(self, hello_ctx):
        ctx, sink = hello_ctx
        out = CommandExecutor.invoke_command(
            ctx,
            "add_two_numbers <first_num>5</first_num><second_num>3</second_num>",
        )
        assert out.success
        span = sink.named(tracing.SPAN_NLU_INTENT)[-1]
        assert span.attributes["matcher_layer"] == "exact_prefix"
        assert span.attributes["command_name"] == "add_two_numbers"
        assert span.attributes["resolved"] is True
        assert span.attributes["ambiguous"] is False

    def test_fuzzy_prematch_still_routes_to_what_can_i_do(self, hello_ctx):
        ctx, sink = hello_ctx
        CommandExecutor.invoke_command(ctx, "what can i do")
        span = sink.named(tracing.SPAN_NLU_INTENT)[-1]
        assert span.attributes["matcher_layer"] == "fuzzy_prematch"
        assert span.attributes["command_name"] == "IntentDetection/what_can_i_do"
        assert span.attributes["is_cme_command"] is True

    def test_embedding_cache_lookup_is_unchanged_by_asking_for_details(
        self, hello_ctx
    ):
        """The one call this change altered in another module's API.

        `cache_match(..., return_details=True)` must pick the same label as the
        plain call and still answer None on a miss, or the embedding-cache tier
        routes differently. Real store, real DistilBERT embeddings, real matcher.
        """
        from fastworkflow.cache_matching import cache_match, store_utterance_cache
        from fastworkflow.model_pipeline_training import CommandRouter

        pipeline = CommandRouter(
            f"{HELLO_WORLD}/___command_info/*"
        ).modelpipeline
        cache_path = str(
            Path(HELLO_WORLD) / "___convo_info" / f"sigtest-{uuid.uuid4().hex}.sqlite3"
        )
        try:
            store_utterance_cache(
                cache_path, "add five and three", "add_two_numbers", pipeline
            )
            plain = cache_match(cache_path, "add five and three", pipeline, 0.85)
            detailed = cache_match(
                cache_path, "add five and three", pipeline, 0.85, return_details=True
            )
            assert plain == "add_two_numbers"
            assert detailed[0] == plain
            assert 0.85 <= detailed[1] <= 1.0
            # A miss stays falsy, which is what the walrus in `_predict_impl` tests.
            assert (
                cache_match(
                    cache_path,
                    "totally unrelated wording about umbrellas",
                    pipeline,
                    0.999,
                    return_details=True,
                )
                is None
            )
        finally:
            Path(cache_path).unlink(missing_ok=True)


class TestSpansCarryTheRecords:
    def test_intent_span_carries_a_decision_uncertainty(self, hello_ctx):
        ctx, sink = hello_ctx
        CommandExecutor.invoke_command(
            ctx,
            "add_two_numbers <first_num>5</first_num><second_num>3</second_num>",
        )
        span = sink.named(tracing.SPAN_NLU_INTENT)[-1]
        recorded = span.attributes["decision_uncertainty"]
        assert recorded["decision_kind"] == "command-identity"
        assert recorded["signals_absent_reason"] == "deterministic-resolution"
        assert recorded["signals"] == []
        # Re-parses into the contract, so what lands on the span is not a
        # look-alike dict that would fail validation on the way back in.
        assert DecisionUncertainty(**recorded).candidate_count == 1

    def test_intent_span_always_carries_an_escalation_outcome(self, hello_ctx):
        ctx, sink = hello_ctx
        CommandExecutor.invoke_command(ctx, "what can i do")
        CommandExecutor.invoke_command(
            ctx, "please compute the total of five plus three"
        )
        spans = sink.named(tracing.SPAN_NLU_INTENT)
        assert spans
        for span in spans:
            assert span.attributes["escalation_outcome"] in ESCALATION_OUTCOMES

        by_layer = {s.attributes.get("matcher_layer"): s for s in spans}
        # Only the classifier can produce an escalation label; every other layer
        # says so rather than reporting a misleading "absent".
        assert (
            by_layer["fuzzy_prematch"].attributes["escalation_outcome"]
            == ESCALATION_OUTCOME_NOT_EVALUATED
        )

    def test_classifier_span_carries_a_confidence_signal(self, hello_ctx):
        ctx, sink = hello_ctx
        CommandExecutor.invoke_command(
            ctx, "please compute the total of five plus three"
        )
        spans = [
            s
            for s in sink.named(tracing.SPAN_NLU_INTENT)
            if s.attributes.get("matcher_layer") == "classifier"
        ]
        assert spans, "classifier layer never ran"
        span = spans[0]
        recorded = DecisionUncertainty(**span.attributes["decision_uncertainty"])
        confidence = _signal(recorded, "classifier-confidence")
        # The same number the existing classifier attribute reports — one
        # measurement, recorded twice, never two.
        assert confidence.value == span.attributes["classifier"]["confidence"]
        assert confidence.signal_version == span.attributes["classifier_signal_version"]
        assert recorded.candidate_count == len(
            span.attributes.get("candidates")
            or [span.attributes["classifier"]["top_label"]]
        )

    def test_param_extraction_span_carries_a_slot_binding_record(self, hello_ctx):
        ctx, sink = hello_ctx
        CommandExecutor.invoke_command(
            ctx,
            "add_two_numbers <first_num>5</first_num><second_num>3</second_num>",
        )
        span = sink.named(tracing.SPAN_NLU_PARAM_EXTRACTION)[-1]
        recorded = DecisionUncertainty(**span.attributes["decision_uncertainty"])
        assert recorded.decision_kind == "slot-binding"
        source = _signal(recorded, "slot-binding-source")
        assert source.value == span.attributes["extraction_method"] == "xml_regex"
        assert source.value in SLOT_BINDING_SOURCES


# ----------------------------------------------------------------------
# EXP-003 exit criterion 2: nothing reads what is captured
# ----------------------------------------------------------------------

# Imported from `fastworkflow.observability.decision_signals`; anything one of these produces is
# a captured record.
_CONTRACT_NAMES = frozenset({
    "DecisionUncertainty",
    "UncertaintySignal",
    "ambiguity_set_size",
    "classifier_confidence",
    "fuzzy_distance",
    "slot_binding_source",
})

# The assembling functions and the locals that hold their output, plus the two raw
# measurements this change stopped discarding. Named explicitly rather than
# inferred, so that renaming one and leaving this list alone fails the presence
# check below instead of silently emptying the rule.
_CAPTURED_VALUE_NAMES = frozenset({
    "_recorded_command_identity_uncertainty",
    "_recorded_slot_binding_uncertainty",
    "best_distance",
    "cache_similarity",
    "command_identity_uncertainty",
    "escalation_outcome_of",
    "signals",
    "slot_binding_uncertainty",
    "uncertainty",
})

_GUARDED_MODULES = (intent_detection, parameter_extraction)

# The functions that decide where a message goes. Nothing they contain may come
# from the signals contract at all.
_ROUTING_FUNCTIONS = ("_predict_impl", "_extract_impl")

# `is` / `is not` against None asks whether a record exists; it cannot express a
# threshold. Every other comparison can.
_THRESHOLDABLE_OPS = (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq, ast.In,
                      ast.NotIn)


def _tree(module) -> ast.Module:
    return ast.parse(Path(inspect.getfile(module)).read_text(encoding="utf-8"))


def _names_in(node) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} | {
        n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)
    }


def _presence_checked_names(test) -> set[str]:
    """Names *test* only asks about the existence of.

    `x is None` answers "was a record produced", which no amount of ingenuity
    turns into "was the runtime confident enough". Exempting it here is the same
    exemption `_THRESHOLDABLE_OPS` already makes, kept consistent so a presence
    check does not have to be written around.
    """
    checked: set[str] = set()
    for node in ast.walk(test):
        if isinstance(node, ast.Compare) and all(
            isinstance(op, (ast.Is, ast.IsNot)) for op in node.ops
        ):
            checked |= _names_in(node)
    return checked


def test_the_guarded_names_actually_exist_in_the_guarded_modules():
    """A rule over names nobody uses passes for the wrong reason."""
    seen: set[str] = set()
    for module in _GUARDED_MODULES:
        seen |= _names_in(_tree(module))
    missing = (_CONTRACT_NAMES | _CAPTURED_VALUE_NAMES) - seen
    assert not missing, f"guarded names no longer present: {sorted(missing)}"


def _comparison_offenders(tree: ast.Module, guarded: frozenset[str]) -> set[str]:
    """Guarded names that *tree* compares against something."""
    offenders: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        if not any(isinstance(op, _THRESHOLDABLE_OPS) for op in node.ops):
            continue
        offenders |= _names_in(node) & guarded
    return offenders


def _truthiness_offenders(tree: ast.Module, guarded: frozenset[str]) -> set[str]:
    """Guarded names *tree* uses as a bare truth value."""
    tests = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.While, ast.IfExp, ast.Assert)):
            tests.append(node.test)
        elif isinstance(node, ast.comprehension):
            tests.extend(node.ifs)

    offenders: set[str] = set()
    for test in tests:
        bare = {
            n.id
            for n in ast.walk(test)
            if isinstance(n, ast.Name) and n.id in guarded
        }
        # A guarded name that is the function being called (`fuzzy_distance(x)`)
        # is a constructor, not a value being consulted.
        called = {
            n.func.id
            for n in ast.walk(test)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        }
        offenders |= bare - called - _presence_checked_names(test)
    return offenders


def test_the_no_read_rules_actually_fire():
    """A static guard nobody has seen reject anything is decoration.

    Each snippet is a plausible way the stop condition would arrive: a threshold on
    a distance, a threshold on a confidence, and a record standing in for a verdict.
    """
    guarded = frozenset({"best_distance", "uncertainty"})

    assert _comparison_offenders(
        ast.parse("if best_distance < 0.15:\n    route_directly()\n"), guarded
    ) == {"best_distance"}
    assert _comparison_offenders(
        ast.parse("proceed = signal.value >= 0.9\n"), frozenset({"value"})
    ) == {"value"}
    assert _truthiness_offenders(
        ast.parse("if uncertainty:\n    ask_the_user()\n"), guarded
    ) == {"uncertainty"}

    # And the exemptions do not fire: a presence check is not a threshold.
    assert not _comparison_offenders(
        ast.parse("recorded = None if uncertainty is None else uncertainty.dump()\n"),
        guarded,
    )
    assert not _truthiness_offenders(
        ast.parse("recorded = None if uncertainty is None else uncertainty.dump()\n"),
        guarded,
    )


@pytest.mark.parametrize("module", _GUARDED_MODULES, ids=lambda m: m.__name__)
def test_no_captured_value_is_compared_against_anything(module):
    """Arch §17.3 stop condition / EXP-003 exit criterion 2.

    A captured confidence, distance, similarity, or assembled record may be
    written to a span and nothing else. The moment one appears in a comparison,
    an instrumentation slice has become an unmeasured behavior change — and it
    would be a threshold set before calibration was ever measured (FW-REQ-021
    clause 4).
    """
    offenders = _comparison_offenders(
        _tree(module), _CONTRACT_NAMES | _CAPTURED_VALUE_NAMES
    )
    assert not offenders, (
        f"{module.__name__} compares captured value(s) {sorted(offenders)}; "
        "captured signals are recorded, never read"
    )


@pytest.mark.parametrize("module", _GUARDED_MODULES, ids=lambda m: m.__name__)
def test_no_captured_value_is_used_as_a_truth_value(module):
    """Truthiness is a comparison with the syntax removed.

    Covers `if confidence:` and `if signals and ...:`, which the comparison test
    above cannot see.
    """
    offenders = _truthiness_offenders(
        _tree(module), _CONTRACT_NAMES | _CAPTURED_VALUE_NAMES
    )
    assert not offenders, (
        f"{module.__name__} branches on captured value(s) {sorted(offenders)}"
    )


@pytest.mark.parametrize("module", _GUARDED_MODULES, ids=lambda m: m.__name__)
def test_routing_functions_do_not_touch_the_signals_contract(module):
    """The strongest form of "no behavior change" available statically.

    `_predict_impl` and `_extract_impl` decide where a message goes. They record
    raw facts into the capture bag, exactly as they already did; the structured
    records are assembled afterwards, by functions the routing code never calls.
    If a contract name appears inside one of these bodies, the two have become
    entangled and this test is the place to argue about it.
    """
    for node in ast.walk(_tree(module)):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name not in _ROUTING_FUNCTIONS:
            continue
        offenders = _names_in(node) & _CONTRACT_NAMES
        assert not offenders, (
            f"{module.__name__}.{node.name} references the signals contract "
            f"({sorted(offenders)}); assembly belongs outside the routing path"
        )

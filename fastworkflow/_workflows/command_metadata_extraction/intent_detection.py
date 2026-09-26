from typing import Optional
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from pydantic import BaseModel

import fastworkflow
from fastworkflow.utils.logging import logger
from fastworkflow import NLUPipelineStage, tracing
from fastworkflow.cache_matching import cache_match, store_utterance_cache
from fastworkflow.observability.decision_signals import (
    DecisionUncertainty,
    UncertaintySignal,
    ambiguity_set_size,
    classifier_confidence,
    classifier_topk_margin,
    fuzzy_distance,
)
from fastworkflow.kvstore import KVStore
from fastworkflow.model_pipeline_training import (
    CommandRouter,
    GLOBAL_CONTEXT_FOLDER,
)
# `entry_declarations` owns the entry-command declaration contract: one
# canonical source for "which command enters this context", read here by the
# foreign-context hint. Nothing in the routing definition or the context model
# records the fact, and inferring it from a command's NAME would bake one
# workflow's spelling conventions into the framework.
# `CONTEXT_ENTER_COMMAND_ATTRS` is re-exported because this module is where the
# foreign-context callers already look for it.
from fastworkflow.entry_declarations import (
    CONTEXT_ENTER_COMMAND_ATTRS,
    declared_entry_commands,
)
from fastworkflow.nlu_labels import is_escalation, is_non_routable
from fastworkflow.train.artifact_versioning import VERSIONS_DIRNAME

from fastworkflow.utils.fuzzy_match import find_best_matches


# A low-confidence top-k prediction containing an escalation label remains an
# ambiguity: prompt with the local command candidates and log that the parent signal
# was discarded. This is product behaviour, not workflow configuration.


# What became of an escalation signal on this prediction. FW-REQ-021 clause 1 asks
# for a first-class escalation outcome rather than only the list of labels that were
# thrown away: "no escalation label was predicted" and "we never ran the classifier
# that could predict one" are different facts, and `escalation_labels_discarded`
# being absent cannot tell them apart.
#
# Only the classifier can produce an escalation label, so every other matching layer
# reports NOT_EVALUATED rather than ABSENT.
ESCALATION_OUTCOME_NOT_EVALUATED = "not_evaluated"
ESCALATION_OUTCOME_ABSENT = "absent"
# The escalation label won outright: `resolve_fully_qualified_command_name` maps it
# to None and the CME wildcard command walks the parent chain, which is the signal
# being acted on.
ESCALATION_OUTCOME_HONORED = "honored"
# The classifier ranked an escalation label alongside local candidates and was not
# confident. The user is prompted with the local candidates only (the ambiguity
# message filters non-routable labels) and the "try my parent" signal is dropped —
# GAP-18, recorded here as an outcome instead of inferred from a log line.
ESCALATION_OUTCOME_SUPPRESSED_BY_AMBIGUITY = "suppressed_by_ambiguity"

ESCALATION_OUTCOMES: frozenset[str] = frozenset({
    ESCALATION_OUTCOME_NOT_EVALUATED,
    ESCALATION_OUTCOME_ABSENT,
    ESCALATION_OUTCOME_HONORED,
    ESCALATION_OUTCOME_SUPPRESSED_BY_AMBIGUITY,
})

# Maximum normalized Levenshtein distance the fuzzy pre-match will accept. Named so
# the span can record what a distance was compared against — a distance with no
# threshold beside it cannot be binned by a calibration report. The value is
# unchanged.
_FUZZY_PREMATCH_MAX_DISTANCE = 0.3  # Adjust threshold as needed

# Identifies the matcher that produced a fuzzy distance, so a curve drawn from one
# matcher is not silently continued by another. The name carries the two properties
# that fix the units: a normalized Levenshtein distance, and `best_window=False`,
# which scores only the candidate's LEADING len(input) characters. Turning
# `best_window` on can only lower distances, so it would shift every bin without
# changing the field name.
_FUZZY_MATCHER_VERSION = "levenshtein-leading-window/1"

# The matching layer between the context-scoped exact-name match and the fuzzy
# pre-match: the first token names a real command of this workflow, and this
# context is not one that owns it. Recorded as its own layer
# rather than folded into "no matcher claimed it", because the two are opposite
# facts -- this one is a deterministic refusal to answer, and a span that says
# so is how a misroute is counted after the fact.
MATCHER_LAYER_KNOWN_NAME_FOREIGN_CONTEXT = "known_name_foreign_context"

# Reported when the classifier artifacts are not under the versioned layout. A
# tree that has never been trained under versioning has no version to report, and
# saying so is better than inventing one that would look comparable across runs.
_UNVERSIONED_ARTIFACT = "unversioned"

# Reported when the router did not say which model answered. Named rather than
# defaulted to "tiny", because a signal_version that claims the wrong tier is worse
# than one that admits it does not know.
_UNKNOWN_MODEL_TIER = "unknown-tier"


def foreign_context_hint(
    command_name: str,
    owner_contexts: list[str],
    enter_commands: dict[str, list[str]],
) -> str:
    """What to say when a real command name reaches a context that cannot run it.

    The foreign-context matcher returns None for such a name and the parent walk
    carries it up; when no context on that chain owns it either, the walk used to
    end at `you_misunderstood`, which says only that nothing matched. That is true
    and useless: the name IS a command, the runtime knows exactly which contexts
    have it, and the caller is one navigation away from being able to run it. The
    CME wildcard command now returns this hint as the whole response instead.

    A hint, never an action: this composes text and nothing here navigates. The
    difference matters because auto-navigating on a misrouted call would change
    the workflow's state on the strength of a guess about what was meant.

    Generic by construction. The owning contexts come from the routing
    definition, and the entering command only from a context's own
    `enter_command` declaration -- never from the shape of a command's name.

    *enter_commands* is keyed by owning context. With one owner the entering
    commands are simply listed; with several, each is named beside the context
    it enters, because a merged list cannot say which command opens which.
    Owners that declare nothing are left out of the entry text, and still named
    among the owners.
    """
    owners = ", ".join(owner_contexts)
    plural = "contexts" if len(owner_contexts) > 1 else "context"
    declared = [
        (context_name, enter_commands[context_name])
        for context_name in owner_contexts
        if enter_commands.get(context_name)
    ]
    entry = ""
    if len(owner_contexts) == 1 and declared:
        entry = " Enter it with: " + ", ".join(
            f"'{c}'" for c in declared[0][1]) + "."
    elif declared:
        entry = "".join(
            f" Enter {context_name} with: "
            + ", ".join(f"'{c}'" for c in commands) + "."
            for context_name, commands in declared
        )
    if entry:
        step = f"{entry} Then run '{command_name}' there."
    else:
        where = "that context" if len(owner_contexts) == 1 else "one of those contexts"
        step = f" Move into {where} first, then run '{command_name}' there."
    return (
        f"'{command_name}' is a command of the {owners} {plural}, which this "
        f"context and its parents do not provide.{step} Use 'what_can_i_do' to "
        f"list the commands available where you are now."
    )


def escalation_outcome_of(predictions: list[str]) -> str:
    """What became of an escalation signal in *predictions*.

    The classifier returns one label when it was confident and its top-k when it was
    not (``CommandRouter.predict_with_details``), so a lone escalation label is one
    the runtime acted on — it resolves to ``command_name=None`` and the CME wildcard
    command walks the parent chain. An escalation label ranked among several is the
    GAP-18 case: the ambiguity message filters non-routable labels out, so the user
    sees the local candidates and the "try my parent" signal goes nowhere.

    Reports an outcome, never a decision: the caller stores the result on the span
    and the resolution path never reads it back.
    """
    if not any(is_escalation(prediction) for prediction in predictions):
        return ESCALATION_OUTCOME_ABSENT
    if len(predictions) == 1:
        return ESCALATION_OUTCOME_HONORED
    return ESCALATION_OUTCOME_SUPPRESSED_BY_AMBIGUITY


def _classifier_signal_version(model_artifact_path: str, model_tier: str) -> str:
    """Identity of the trained artifact behind a classifier signal.

    `signal_version` exists so that a retrained classifier is distinguishable: a
    confidence of 0.8 from one artifact is not the same measurement as a
    confidence of 0.8 from the next, and FW-REQ-021 clause 13 requires thresholds
    be re-validated when the producing artifact changes.

    Under the versioned artifact layout the per-context entry in
    ``___command_info`` is a compatibility link into
    ``___command_info/versions/<version>/<context>``, so one
    ``realpath`` recovers the published version without importing the trainer's
    resolver or re-reading its pointer file on every prediction. The ``*`` to
    ``global`` mapping mirrors ``CommandRouter.__init__``, which does the same
    substitution before opening the artifacts.
    """
    resolved = os.path.realpath(
        model_artifact_path.replace('*', GLOBAL_CONTEXT_FOLDER)
    )
    versions_parent = os.path.dirname(os.path.dirname(resolved))
    version = (
        os.path.basename(os.path.dirname(resolved))
        if os.path.basename(versions_parent) == VERSIONS_DIRNAME
        else _UNVERSIONED_ARTIFACT
    )
    return (
        f"intent-classifier/{version}"
        f"/{os.path.basename(resolved)}/{model_tier}"
    )


def _topk_margin_signals(
    classifier_details: dict, signal_version: str
) -> list[UncertaintySignal]:
    """The gap between the top two label probabilities, as a 0- or 1-item list.

    A list rather than an Optional so the caller has nothing to test. That is not
    style: `nlu_trace["classifier"]` belongs to `CommandRouter`, and the test
    doubles that inject labels return it EMPTY on purpose, so the emitter has to
    tolerate a router that reports no top-k exactly as it tolerates one that
    reports no `model_tier`. Pairing the kept scores with their own tail yields one
    pair for two scores and no pair at all for fewer, which gets that tolerance
    without comparing a captured measurement against anything (arch §17.3).

    The scores are positionally aligned with `topk_labels` and come from the single
    forward pass `predict_batch` already made, so this is a projection of a number
    that was computed and thrown away, not a new measurement.
    """
    top_two = list(classifier_details.get("topk_scores") or ())[:2]
    return [
        classifier_topk_margin(
            float(first) - float(second), signal_version=signal_version
        )
        for first, second in zip(top_two, top_two[1:])
    ]


def command_identity_uncertainty(nlu_trace: dict) -> DecisionUncertainty:
    """The §6.6.1 record for the command-identity decision *nlu_trace* describes.

    A pure function of the facts ``_predict_impl`` recorded, and deliberately
    one-way: it reads the capture bag and writes nothing back, so the resolution
    path cannot come to depend on what is being measured about it. That is the
    architecture §17.3 stop condition — capture only, no threshold, no branch.

    It reads ``matcher_layer``, which is the name of the branch that has already
    run, not a measurement of it. No confidence, distance, count, or assembled
    record is read by anything other than ``tracing.end_span``.
    """
    matcher_layer = nlu_trace.get("matcher_layer")
    signals: list[UncertaintySignal] = []
    signals_absent_reason = None
    # Every tier that enumerates candidates records how many; the tiers below that
    # do not enumerate resolved to exactly one command, or to none at all.
    candidate_count = nlu_trace.get("candidate_count", 1)

    if matcher_layer == MATCHER_LAYER_KNOWN_NAME_FOREIGN_CONTEXT:
        # The first token is a command name this workflow owns elsewhere. That
        # is a lookup against an inventory, not a measurement: there is no
        # confidence, distance or candidate set behind it, and the decision it
        # produced was "decline, let the parent chain answer". Reporting it as
        # deterministic keeps it out of every calibration curve, where a
        # confidence of 1.0 (or an uncaptured-measurement marker) would both be
        # false.
        signals_absent_reason = "deterministic-resolution"
        candidate_count = 0
    elif matcher_layer == "exact_prefix":
        # An exact command-name match has nothing to be unsure about. Emitting
        # confidence 1.0 here would enter a calibration curve as a real
        # measurement of a classifier that never ran.
        signals_absent_reason = "deterministic-resolution"
    elif matcher_layer == "clarification_default":
        # 'what can i do?' is a constant the code substitutes when no matcher
        # claimed the utterance in a clarification stage. Nothing was measured and
        # nothing could have been, so this is deterministic in the same sense as an
        # exact match rather than an uncaptured measurement.
        signals_absent_reason = "deterministic-resolution"
    elif matcher_layer == "fuzzy_prematch":
        signals.extend([
            fuzzy_distance(
                nlu_trace["fuzzy_distance"], signal_version=_FUZZY_MATCHER_VERSION
            ),
            ambiguity_set_size(candidate_count, signal_version=_FUZZY_MATCHER_VERSION),
        ])
    elif matcher_layer == "embedding_cache":
        # The cosine similarity that decided this is on the span as
        # `cache_similarity`, but §6.6.1's SignalKind enum has no member for an
        # embedding similarity, so the structured record genuinely cannot carry it.
        # "not-instrumented" is the honest report of a decision whose measurement
        # the contract cannot represent yet.
        signals_absent_reason = "not-instrumented"
    elif matcher_layer == "classifier":
        signal_version = nlu_trace["classifier_signal_version"]
        signals.append(
            classifier_confidence(
                nlu_trace["classifier"]["confidence"], signal_version=signal_version
            )
        )
        # classifier-topk-margin is missing on purpose: `predict_with_details`
        # returns the winning label's probability and the top-k label NAMES, and
        # `predict_single_sentence` drops `top_k_scores` from what `predict_batch`
        # computed. There is no second-best probability to subtract, and inventing
        # one is worse than its absence.
        #
        # That is no longer true: `predict_single_sentence` carries `top_k_scores`
        # through to `predict_with_details`, so the second-best probability is a
        # fact the same forward pass already produced. It stays absent when the
        # details dict does not carry it — a stubbed router, or an older record —
        # which is why the helper returns a list instead of raising on a missing
        # key.
        signals.extend(
            _topk_margin_signals(nlu_trace["classifier"], signal_version)
        )
        signals.append(
            ambiguity_set_size(candidate_count, signal_version=signal_version)
        )
    else:
        # No matcher claimed the utterance. The fuzzy pre-match did run and compared
        # a distance against the threshold, but `find_best_matches` returns
        # ``([], None)`` above the threshold and discards the value, so the one
        # measurement taken here is not recoverable at this call site.
        signals_absent_reason = "not-instrumented"
        candidate_count = 0

    return DecisionUncertainty(
        decision_kind="command-identity",
        signals=tuple(signals),
        candidate_count=candidate_count,
        reducible=nlu_trace.get("reducible"),
        signals_absent_reason=signals_absent_reason,
    )


def _recorded_command_identity_uncertainty(
    span, nlu_trace: dict
) -> Optional[dict]:
    """``command_identity_uncertainty`` in span-attribute form, never raising.

    Skipped entirely when *span* is None. ``start_span`` declines without a sink
    or an open turn and ``end_span`` then discards its attributes, so assembling a
    record nobody will store is pure overhead against this slice's declared
    budget — and it keeps the promise the test doubles in
    ``test_intent_detection_fuzzy_tie.py`` were written against, that a stubbed
    classifier's details reach nothing but a span that no-ops.

    ``tracing`` wraps every one of its own calls because a broken recorder must
    degrade to a log line rather than a failed turn, and this assembly runs just
    outside that guard, in the caller's frame. Capture that can lose a user's turn
    is precisely the behavior change Phase 0 exists to avoid.

    A record that violates its own contract is a bug in the recorder, so it is
    logged loudly and the attribute says what happened rather than going quietly
    absent — which would read as "this span predates the capture" — or null, which
    on the parameter-extraction side means "no decision". The pure function above
    still raises, which is what the tests exercise.
    """
    if span is None:
        return None
    try:
        return command_identity_uncertainty(nlu_trace).model_dump(mode="json")
    except Exception as exc:
        logger.warning(f"command-identity uncertainty capture failed: {exc!r}")
        return {"capture_error": type(exc).__name__}


class CommandNamePrediction:
    class Output(BaseModel):
        command_name: Optional[str] = None
        error_msg: Optional[str] = None
        is_cme_command: bool = False
        # Set only when the first token names a command this context does not
        # own. The walk carries the hint so the failure message at the end of
        # the chain can say where the command actually lives; a resolved
        # prediction leaves them None and nothing reads them.
        known_name_owner_contexts: Optional[list[str]] = None
        routing_hint: Optional[str] = None

    def __init__(self, cme_workflow: fastworkflow.Workflow):
        self.cme_workflow = cme_workflow
        self.app_workflow = cme_workflow.context["app_workflow"]
        self.app_workflow_folderpath = self.app_workflow.folderpath
        self.app_workflow_id = self.app_workflow.id

        self.convo_path = os.path.join(self.app_workflow_folderpath, "___convo_info")
        self.cache_path = self._get_cache_path(self.app_workflow_id, self.convo_path)
        self.path = self._get_cache_path_cache(self.convo_path, self.app_workflow_id)
        # Built once per predictor and reused across the parent-chain walk,
        # which asks the same question in every context it visits.
        self._command_inventory: Optional[dict[str, tuple[str, ...]]] = None
        self._enter_commands: dict[str, list[str]] = {}

    def predict(self, command_context_name: str, command: str, nlu_pipeline_stage: NLUPipelineStage) -> "CommandNamePrediction.Output":
        """Predict, wrapped in a ``fw.nlu.intent`` span (D3 as amended).

        One span per prediction attempt — the wildcard command's parent-chain
        walk calls this once per context, and each attempt is recorded with
        the context it ran against. The span carries which matching layer
        decided (exact prefix / fuzzy pre-match / embedding cache /
        classifier), the classifier's confidence and threshold when it ran,
        and the candidate set on an ambiguity. Emission never affects the
        prediction: the helpers no-op without a bound host/sink.

        It also carries the §6.6.1 ``DecisionUncertainty`` for the
        command-identity decision, assembled by ``command_identity_uncertainty``
        from the facts below. That record is written to the span and read by
        nothing else (FW-REQ-021 P0: representation only).
        """
        host = tracing.current_host()
        span = tracing.start_span(
            host,
            tracing.SPAN_NLU_INTENT,
            attributes={
                "context": command_context_name,
                "stage": nlu_pipeline_stage.name,
                "utterance": command,
            },
        )
        nlu_trace: dict = {}
        try:
            output = self._predict_impl(
                command_context_name, command, nlu_pipeline_stage, nlu_trace
            )
        except BaseException:
            # No DecisionUncertainty here: the prediction did not complete, so
            # there is no decision to characterise. The raw facts gathered so far
            # still go out.
            tracing.end_span(
                host, span, status=tracing.STATUS_ERROR, attributes=nlu_trace
            )
            raise
        tracing.end_span(
            host,
            span,
            status=tracing.STATUS_OK,
            attributes={
                **nlu_trace,
                "decision_uncertainty": _recorded_command_identity_uncertainty(
                    span, nlu_trace
                ),
                "command_name": output.command_name,
                "is_cme_command": output.is_cme_command,
                "ambiguous": output.error_msg is not None,
                # None = no local prediction; the caller walks up the context
                # chain (or files a misunderstanding) — exactly the routing
                # signal a debugging agent needs.
                "resolved": output.command_name is not None,
            },
        )
        return output

    def command_inventory(self) -> dict[str, tuple[str, ...]]:
        """Every command name this app workflow owns, and the contexts owning it.

        Keyed on the lowercased simple name, which is what the exact-name
        matcher compares against, and valued with the context names so a span
        can say where the call should have gone. Reserved labels (`wildcard`,
        `parameter_value`) name no command and are left out.

        Read from the same `RoutingDefinition` the per-context candidate set is
        built from (`RoutingRegistry` caches it per workflow folder), so there
        is no second source of truth to drift: this is that definition read
        across every context instead of one.
        """
        if self._command_inventory is None:
            app_crd = fastworkflow.RoutingRegistry.get_definition(
                self.app_workflow_folderpath)
            owners: dict[str, set[str]] = {}
            for context_name, qualified_names in app_crd.contexts.items():
                for qualified_name in qualified_names:
                    simple_name = qualified_name.split('/')[-1]
                    if is_non_routable(simple_name):
                        continue
                    owners.setdefault(simple_name.lower(), set()).add(context_name)
            self._command_inventory = {
                name: tuple(sorted(contexts)) for name, contexts in owners.items()
            }
        return self._command_inventory

    def enter_commands_for(self, context_name: str) -> list[str]:
        """The command(s) a workflow declares as entering *context_name*.

        Read through `entry_declarations.declared_entry_commands`, which owns the
        declaration contract; the context's own callback class is the only
        place the fact is recorded. Empty when the workflow declares nothing,
        when the context has no callback class, or when loading it fails -- a
        hint that names the context alone is worth more than a failed turn, so
        nothing here is allowed to raise.
        """
        if context_name in self._enter_commands:
            return self._enter_commands[context_name]
        declared = declared_entry_commands(self.app_workflow_folderpath, context_name)
        self._enter_commands[context_name] = declared
        return declared

    def routing_hint_for(self, command_name: str, owner_contexts: list[str]) -> str:
        """The hint for one foreign known name, entering commands resolved."""
        enter_commands = {
            context_name: self.enter_commands_for(context_name)
            for context_name in owner_contexts
        }
        return foreign_context_hint(command_name, owner_contexts, enter_commands)

    def foreign_owner_contexts(
        self,
        normalized_command_name: str,
        command_name_dict: dict[str, str],
        command_context_name: str,
    ) -> list[str]:
        """Contexts owning *normalized_command_name*, when this one does not.

        Empty when the token names no command of this workflow (ordinary free
        text, which is what the classifier is for) and empty when this
        context's own candidate set contains it (already matched above). A
        non-empty result is the foreign-context condition: a real command name,
        reached in a context that cannot execute it.

        Compared in lowercase on both sides: the inventory is keyed on the
        lowercased name while `command_name_dict` keeps each command's own
        spelling, so a case-sensitive test would call a context's own
        mixed-case command foreign. The current context is dropped from the
        owners for the same reason -- a context may never refuse a name as
        belonging somewhere else when the somewhere else is itself.
        """
        if normalized_command_name in {name.lower() for name in command_name_dict}:
            return []
        return [
            context_name
            for context_name in self.command_inventory().get(normalized_command_name, ())
            if context_name != command_context_name
        ]

    def _predict_impl(
        self,
        command_context_name: str,
        command: str,
        nlu_pipeline_stage: NLUPipelineStage,
        nlu_trace: dict,
    ) -> "CommandNamePrediction.Output":
        # sourcery skip: extract-duplicate-method

        # Set before any matching so the attribute is always present: an absent
        # escalation outcome would be indistinguishable from "no escalation", and
        # only the classifier below can produce an escalation label at all.
        nlu_trace["escalation_outcome"] = ESCALATION_OUTCOME_NOT_EVALUATED

        model_artifact_path = f"{self.app_workflow_folderpath}/___command_info/{command_context_name}"
        command_router = CommandRouter(model_artifact_path)

        # Re-use the already-built ModelPipeline attached to the router
        # instead of instantiating a fresh one.  This avoids reloading HF
        # checkpoints and transferring tensors each time we see a new
        # message for the same context.
        modelpipeline = command_router.modelpipeline

        crd = fastworkflow.RoutingRegistry.get_definition(
            self.cme_workflow.folderpath)
        cme_command_names = crd.get_command_names('IntentDetection')

        valid_command_names = set()
        if nlu_pipeline_stage == NLUPipelineStage.INTENT_AMBIGUITY_CLARIFICATION:
            valid_command_names = self._get_suggested_commands(self.path)
        elif nlu_pipeline_stage in (
                NLUPipelineStage.INTENT_DETECTION, NLUPipelineStage.INTENT_MISUNDERSTANDING_CLARIFICATION):
            app_crd = fastworkflow.RoutingRegistry.get_definition(
                self.app_workflow_folderpath)
            valid_command_names = (
                set(cme_command_names) | 
                set(app_crd.get_command_names(command_context_name))
            )

        command_name_dict = {
            fully_qualified_command_name.split('/')[-1]: fully_qualified_command_name 
            for fully_qualified_command_name in valid_command_names
        }

        if nlu_pipeline_stage in (
                NLUPipelineStage.INTENT_AMBIGUITY_CLARIFICATION,
                NLUPipelineStage.INTENT_MISUNDERSTANDING_CLARIFICATION):
            # what_can_i_do is special in both clarification stages
            # We will not predict, just match plain utterances with exact or fuzzy match
            # Both stages also fall back to the plain utterance 'what can i do?'
            # when nothing matches (below), so it must be a key in both.
            command_name_dict |= {
                plain_utterance: 'IntentDetection/what_can_i_do'
                for plain_utterance in crd.command_directory.map_command_2_utterance_metadata[
                    'IntentDetection/what_can_i_do'
                ].plain_utterances
            }

        if nlu_pipeline_stage != NLUPipelineStage.INTENT_DETECTION:
            # abort is special. 
            # We will not predict, just match plain utterances with exact or fuzzy match
            command_name_dict |= {
                plain_utterance: 'ErrorCorrection/abort'
                for plain_utterance in crd.command_directory.map_command_2_utterance_metadata[
                    'ErrorCorrection/abort'
                ].plain_utterances
            }

        if nlu_pipeline_stage != NLUPipelineStage.INTENT_MISUNDERSTANDING_CLARIFICATION:
            # you_misunderstood is special. 
            # We will not predict, just match plain utterances with exact or fuzzy match
            command_name_dict |= {
                plain_utterance: 'ErrorCorrection/you_misunderstood'
                for plain_utterance in crd.command_directory.map_command_2_utterance_metadata[
                    'ErrorCorrection/you_misunderstood'
                ].plain_utterances
            }

        # A command name ends at whitespace or a '(' -- ANY whitespace, not a
        # space alone: `list_permissions\n<scope>all</scope>` otherwise makes the
        # whole line the tentative name, so neither the matcher nor the foreign-
        # name guard sees one (F31/ido-nx6). "" splits to [], hence the guard.
        tentative_command_name = (
            command.split(None, 1)[0].split("(", 1)[0] if command.strip() else "")
        normalized_command_name = tentative_command_name.lower()
        # The typed token is lowercased, but `command_name_dict` keeps each
        # command's own spelling, so the exact match looks the token up by its
        # lowercase form and resolves to the key as spelled.
        exact_key = {
            name.lower(): name for name in command_name_dict
        }.get(normalized_command_name)
        command_name = None
        if exact_key is not None:
            command_name = exact_key
            command = command.replace(f"{tentative_command_name}", "").strip().replace("  ", " ")
            nlu_trace["matcher_layer"] = "exact_prefix"
        elif (
            nlu_pipeline_stage in (
                NLUPipelineStage.INTENT_DETECTION,
                NLUPipelineStage.INTENT_MISUNDERSTANDING_CLARIFICATION)
            and (owner_contexts := self.foreign_owner_contexts(
                normalized_command_name, command_name_dict, command_context_name))
        ):
            # R1 (ido-8ps.8): a known command name may not be answered by a
            # context that does not own it. The exact-name matcher above is
            # scoped to THIS context's command set, so a root ('*') or sibling
            # command typed verbatim -- `fetch_result_page <handle>O9</handle>`,
            # `show_holders <filter>...`, `list_permissions` -- is invisible to
            # layers 1 and 2 here and would be adjudicated by this context's
            # classifier, which answered four of them with a confident
            # "No permissions found." (ido-8ps.6.1 section 4.2: 69 silent
            # misroutes across 31 stored runs).
            #
            # None is already the signal that drives the parent-chain walk
            # (`_commands/wildcard.py`), so returning it here carries the
            # call up to the context that does own the name, where the
            # deterministic matcher resolves it. Nothing below this line runs:
            # no fuzzy candidates, no embedding cache, no classifier. A name
            # whose owner is not on this chain now reaches the routing hint
            # below instead of a lucky classifier guess -- loud instead of silent.
            #
            # INTENT_DETECTION and INTENT_MISUNDERSTANDING_CLARIFICATION only.
            # The ambiguity stage matches against a constrained suggestion set,
            # where "not in this context's set" is the normal case rather than
            # a misroute. The misunderstanding stage matches this context's
            # full command set, the same set intent detection uses, so a real
            # command name missing from it is the same misroute there; without
            # the guard it fell to the 'what can i do?' default and the owner
            # was never named.
            hint = self.routing_hint_for(normalized_command_name, owner_contexts)
            nlu_trace["matcher_layer"] = MATCHER_LAYER_KNOWN_NAME_FOREIGN_CONTEXT
            nlu_trace["known_name_foreign_context"] = True
            nlu_trace["known_name_owner_contexts"] = owner_contexts
            # On the span beside the event it belongs to, so a summary can count
            # how often the hint was given and how often the next call followed
            # it, without re-deriving the text from the inventory.
            nlu_trace["known_name_foreign_context_hint"] = hint
            return CommandNamePrediction.Output(
                command_name=None,
                known_name_owner_contexts=owner_contexts,
                routing_hint=hint,
            )
        else:
            # Use Levenshtein distance for fuzzy matching with the full command part after @
            # No match is ([], None), never (None, None) — len() here is safe.
            best_matched_commands, best_distance = find_best_matches(
                command.replace(" ", "_"),
                command_name_dict.keys(),
                threshold=_FUZZY_PREMATCH_MAX_DISTANCE
            )
            # The distance used to be thrown away with `_` (FW-REQ-021 clause 1:
            # a signal must not be discarded after its threshold comparison). It
            # is a DISTANCE, so lower is a better match; the threshold beside it
            # is a maximum, and `SIGNAL_DOMAINS["fuzzy-score"]` records that
            # polarity for anyone binning these values. It is None above the
            # threshold, because `find_best_matches` returns ([], None) there.
            nlu_trace["fuzzy_distance"] = best_distance
            nlu_trace["fuzzy_threshold"] = _FUZZY_PREMATCH_MAX_DISTANCE
            nlu_trace["candidate_count"] = len(best_matched_commands)
            if (
                len(best_matched_commands) > 1
                and nlu_pipeline_stage == NLUPipelineStage.INTENT_DETECTION
            ):
                # Commands sharing a prefix tie at distance 0, because scoring
                # compares only the leading len(input) characters, and
                # command_name_dict iterates a set — so picking [0] would choose
                # nondeterministically between them across processes. Leave the
                # name unset so the classifier and its ambiguity prompt decide.
                # The clarification stages are deliberately excluded: they have
                # no classifier to fall back to.
                logger.warning(
                    f"Fuzzy pre-match tied across {best_matched_commands} for "
                    f"utterance '{command}' in context '{command_context_name}'. "
                    "Deferring to the classifier instead of picking one."
                )
                nlu_trace["fuzzy_prematch_tie"] = [str(c) for c in best_matched_commands]
            elif best_matched_commands:
                command_name = best_matched_commands[0]
                nlu_trace["matcher_layer"] = "fuzzy_prematch"

        if nlu_pipeline_stage == NLUPipelineStage.INTENT_DETECTION:
            if not command_name:
                # return_details asks for the similarity alongside the label. The
                # match itself is unchanged — the same threshold, the same winner —
                # but the number it was compared against is no longer discarded
                # (FW-REQ-021 clause 1). A hit is a 2-tuple, a miss is still None,
                # so the walrus test behaves exactly as it did.
                if cache_result := cache_match(
                    self.path, command, modelpipeline, 0.85, return_details=True
                ):
                    command_name, cache_similarity = cache_result
                    nlu_trace["matcher_layer"] = "embedding_cache"
                    nlu_trace["cache_similarity_threshold"] = 0.85
                    # A cosine similarity: HIGHER is a better match, the opposite
                    # of the fuzzy distance above, and the threshold beside it is a
                    # minimum. Recorded as a plain attribute because §6.6.1's
                    # SignalKind enum has no member for it.
                    nlu_trace["cache_similarity"] = float(cache_similarity)
                    nlu_trace["candidate_count"] = 1
                else:
                    predictions, classifier_details = (
                        command_router.predict_with_details(command)
                    )
                    # predictions = majority_vote_predictions(command_router, command)
                    nlu_trace["matcher_layer"] = "classifier"
                    nlu_trace["classifier"] = classifier_details
                    # `.get`, not `[...]`: the details dict belongs to
                    # `CommandRouter`, and a router that reports no tier must not be
                    # able to fail a turn from inside the resolution path. The test
                    # doubles that inject labels return an empty dict on purpose.
                    nlu_trace["classifier_signal_version"] = _classifier_signal_version(
                        model_artifact_path,
                        classifier_details.get("model_tier", _UNKNOWN_MODEL_TIER),
                    )
                    nlu_trace["candidate_count"] = len(predictions)
                    nlu_trace["escalation_outcome"] = escalation_outcome_of(predictions)

                    if len(predictions)==1:
                        command_name = predictions[0].split('/')[-1]
                    else:
                        # If confidence is low, treat as ambiguous command (type 1)
                        if escalation_signals := self.escalation_signals_in(predictions):
                            logger.warning(
                                f"Top-k escalation signal discarded in "
                                f"context '{command_context_name}' for utterance "
                                f"'{command}'. predictions={predictions}, "
                                f"suppressed={escalation_signals}. The classifier ranked "
                                "an escalation label alongside local candidates; the user "
                                "will be prompted with the local candidates only and the "
                                "'this belongs to an ancestor context' signal is dropped."
                            )
                            nlu_trace["escalation_labels_discarded"] = [
                                str(label) for label in escalation_signals
                            ]

                        error_msg = self._formulate_ambiguous_command_error_message(
                            predictions, "run_as_agent" in self.app_workflow.context)

                        # Store suggested commands
                        nlu_trace["candidates"] = [str(p) for p in predictions]
                        # The runtime is about to ask, and the answer resolves this:
                        # requirements §4.14's reducible uncertainty, recorded where
                        # the asking happens rather than inferred afterwards from
                        # the shape of the record.
                        nlu_trace["reducible"] = True
                        self._store_suggested_commands(self.path, predictions, 1)
                        return CommandNamePrediction.Output(error_msg=error_msg)

        elif nlu_pipeline_stage in (
            NLUPipelineStage.INTENT_AMBIGUITY_CLARIFICATION,
            NLUPipelineStage.INTENT_MISUNDERSTANDING_CLARIFICATION
        ) and not command_name:
            command_name = "what can i do?"
            nlu_trace["matcher_layer"] = "clarification_default"
            # The fuzzy pre-match above found nothing and left a count of 0; the
            # default that replaces it is one command, chosen by the code.
            nlu_trace["candidate_count"] = 1

        fully_qualified_command_name = self.resolve_fully_qualified_command_name(
            command_name, command_name_dict)
        if fully_qualified_command_name is None:
            is_cme_command=False
        else:
            is_cme_command=(
                fully_qualified_command_name in cme_command_names or 
                fully_qualified_command_name in crd.get_command_names('ErrorCorrection')
            )

        if (
            nlu_pipeline_stage
            in (
                NLUPipelineStage.INTENT_AMBIGUITY_CLARIFICATION,
                NLUPipelineStage.INTENT_MISUNDERSTANDING_CLARIFICATION,
            )
            # A reserved label resolves to None; the clarification cache keys on a
            # real command name, so there is nothing to store for it.
            and fully_qualified_command_name is not None
            and not fully_qualified_command_name.endswith('abort')
            and not fully_qualified_command_name.endswith('what_can_i_do')
            and not fully_qualified_command_name.endswith('you_misunderstood')
        ):
            command = self.cme_workflow.context["command"]
            store_utterance_cache(self.path, command, command_name, modelpipeline)

        return CommandNamePrediction.Output(
            command_name=fully_qualified_command_name,
            is_cme_command=is_cme_command
        )

    @staticmethod
    def _get_cache_path(workflow_id, convo_path):
        """
        Generate cache file path based on workflow ID
        """
        base_dir = convo_path
        # Create directory if it doesn't exist
        os.makedirs(base_dir, exist_ok=True)
        return os.path.join(base_dir, f"{workflow_id}.sqlite3")

    @staticmethod
    def _get_cache_path_cache(convo_path, workflow_id=None):
        """
        Path to the utterance/clarification cache.

        Shared across sessions by default, which is the point of the cache: a
        disambiguation learned in one session helps the next.

        `FW_UTTERANCE_CACHE_SCOPE=workflow` shards it by workflow id instead.
        Repeated independent runs must not have correlated attempts, and this
        file is read on the runtime turn path (`cache_match`) and written on it
        (`store_utterance_cache`) while being keyed on nothing -- so a second
        run would inherit the first run's disambiguation decisions whenever
        both run against the same workflow folder. The sibling
        `_get_cache_path` is already sharded this way; this is the same
        treatment, opt-in so ordinary runs keep their shared cache.
        """
        base_dir = convo_path
        # Create directory if it doesn't exist
        os.makedirs(base_dir, exist_ok=True)
        scope = fastworkflow.get_env_var("FW_UTTERANCE_CACHE_SCOPE", default="shared")
        if scope == "workflow" and workflow_id is not None:
            return os.path.join(base_dir, f"cache-{workflow_id}.sqlite3")
        return os.path.join(base_dir, "cache.sqlite3")

    # Store the suggested commands with the flag type
    @staticmethod
    def _store_suggested_commands(cache_path, command_list, flag_type):
        """
        Store the list of suggested commands for the constrained selection

        Args:
            cache_path: Path to the cache database
            command_list: List of suggested commands
            flag_type: Type of constraint (1=ambiguous, 2=misclassified)
        """
        with KVStore(cache_path) as db:
            # predict() returns a numpy ndarray of labels; JSON needs plain strs.
            db["suggested_commands"] = [str(c) for c in list(command_list)]
            db["flag_type"] = int(flag_type)

    # Get the suggested commands
    @staticmethod
    def _get_suggested_commands(cache_path):
        """
        Get the list of suggested commands for the constrained selection
        """
        with KVStore(cache_path) as db:
            return db.get("suggested_commands", [])

    @staticmethod
    def _get_count(cache_path):
        with KVStore(cache_path) as db:
            return db.get("utterance_count", 0)  # Default to 0 if key doesn't exist

    @staticmethod
    def _print_db_contents(cache_path):
        with KVStore(cache_path) as db:
            print("All keys in database:", list(db.keys()))
            for key in db.keys():
                print(f"Key: {key}, Value: {db[key]}")

    @staticmethod
    def _store_utterance(cache_path, utterance, label):
        """
        Store utterance in existing or new database
        Returns: The utterance count used
        """
        with KVStore(cache_path) as db:
            # Get existing counter or initialize to 0
            utterance_count = db.get("utterance_count", 0)

            # Create and store the utterance entry
            utterance_data = {
                "utterance": utterance,
                "label": label
            }

            db[utterance_count] = utterance_data

            # Increment and store the counter
            utterance_count += 1
            db["utterance_count"] = utterance_count

            return utterance_count - 1  # Return the count used for this utterance

    # Function to read from database
    @staticmethod
    def _read_utterance(cache_path, utterance_id):
        """
        Read a specific utterance from the database
        """
        with KVStore(cache_path) as db:
            return db.get(utterance_id)['utterance']
    @staticmethod
    def resolve_fully_qualified_command_name(
        command_name: Optional[str], command_name_dict: dict[str, str]) -> Optional[str]:
        """Map a predicted label to a fully qualified command name, or None.

        Reserved labels (`wildcard`, `parameter_value`) name no command, so they
        must resolve to None rather than be looked up in `command_name_dict` —
        which would raise KeyError. None is what drives the parent-chain walk in
        the CME wildcard command.
        """
        if not command_name or is_non_routable(command_name):
            return None
        return command_name_dict[command_name]

    @staticmethod
    def escalation_signals_in(route_choice_list: list[str]) -> list[str]:
        """Return the escalation labels present in a prediction list."""
        return [
            route_choice for route_choice in route_choice_list
            if is_escalation(route_choice)
        ]

    @staticmethod
    def _formulate_ambiguous_command_error_message(
        route_choice_list: list[str], run_as_agent: bool) -> str:
        command_list = (
            "\n".join([
                f"{route_choice.split('/')[-1].lower()}"
                for route_choice in route_choice_list if not is_non_routable(route_choice)
            ])
        )

        return (
            "The command is ambiguous. "
            + (
                "Choose the correct command name from these possible options and update your command:\n"
                if run_as_agent
                else "Please choose a command name from these possible options:\n"
            )
            + f"{command_list}\n\nor type 'what can i do' to see all commands\n"
            + ("or type 'abort' to cancel" if run_as_agent else '')
        )


# TODO - generation is deterministic. They all return the same answer
# TODO - Need 'temperature' for intent detection pipeline
def majority_vote_predictions(command_router, command: str, n_predictions: int = 5) -> list[str]:
    """
    Generate N prediction sets in parallel and return the set that wins the majority vote.
    
    This function improves prediction reliability by running multiple parallel predictions
    and selecting the most common result through majority voting. This helps reduce
    the impact of random variations in model predictions.
    
    Args:
        command_router: The CommandRouter instance to use for predictions
        command: The input command string
        n_predictions: Number of parallel predictions to generate (default: 5)
                      Can be configured via N_PARALLEL_PREDICTIONS environment variable
        
    Returns:
        The prediction set that received the majority vote. Falls back to a single
        prediction if all parallel predictions fail.
        
    Note:
        Uses ThreadPoolExecutor with max_workers limited to min(n_predictions, 10)
        to avoid overwhelming the system with too many concurrent threads.
    """
    def get_single_prediction():
        """Helper function to get a single prediction"""
        return command_router.predict(command)
    
    # Generate N predictions in parallel
    prediction_sets = []
    with ThreadPoolExecutor(max_workers=min(n_predictions, 10)) as executor:
        # Submit all prediction tasks
        futures = [executor.submit(get_single_prediction) for _ in range(n_predictions)]
        
        # Collect results as they complete
        for future in as_completed(futures):
            try:
                prediction_set = future.result()
                prediction_sets.append(prediction_set)
            except Exception as e:
                logger.warning(f"Prediction failed: {e}")
                # Continue with other predictions even if one fails
    
    if not prediction_sets:
        # Fallback to single prediction if all parallel predictions failed
        logger.warning("All parallel predictions failed, falling back to single prediction")
        return command_router.predict(command)
    
    # Convert lists to tuples so they can be hashed and counted
    prediction_tuples = [tuple(sorted(pred_set)) for pred_set in prediction_sets]
    
    # Count occurrences of each unique prediction set
    vote_counts = Counter(prediction_tuples)
    
    # Get the prediction set with the most votes
    winning_tuple = vote_counts.most_common(1)[0][0]
    
    # Convert back to list and return
    return list(winning_tuple)

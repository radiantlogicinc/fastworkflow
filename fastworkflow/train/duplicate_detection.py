"""Detect and report near-duplicate command capabilities (spec R9b / finding F14).

The problem
-----------
Some workflows expose the same capability twice. On a large multi-context workflow,
``ControlsMonitor/list_findings`` and ``Directory/search_control_findings`` answer the same
question. Other pairs are legitimate neighbours or opposites whose current seed lists do not
express the distinction strongly enough. Both shapes present as benchmark failures. The value
of this module is diagnostic honesty — showing what the classifier can and cannot separate so
the developer can merge, alias, knowingly accept, or improve the seeds for the pair.

This module only reports. It never changes what is trained, which labels exist, or what any
model predicts.

The operational definition
--------------------------
"Near-duplicate" is defined as a property of the *training data*, not of the two commands'
meanings, because the training data is the only thing the classifier ever sees:

    Two commands are near-duplicates when a classifier restricted to that pair, trained on
    their own utterances, cannot tell them apart.

Made concrete as **pairwise separability**: leave-one-out, balanced, nearest-centroid
accuracy over the two commands' utterances in TF-IDF space (`pairwise_separability`).

* Leave-one-out, because an utterance sitting inside its own centroid classifies itself.
* Balanced (the mean of the two per-command recalls), so the chance line is 0.5 regardless
  of how many utterances each command has.
* TF-IDF with document frequency computed across **all** commands in the workflow, so the
  vocabulary every command shares — "order", "my", "please", the workflow's own subject
  matter — is discounted automatically, and only genuinely distinguishing terms carry
  weight. This is what stops a workflow full of ``modify_pending_order_address`` /
  ``modify_pending_order_items`` / ``modify_pending_order_payment`` from producing a wall
  of false positives: those commands share their boilerplate and differ in exactly the
  terms IDF promotes.

A pair is reported as a **duplicate** when separability is at or below
`DEFAULT_DUPLICATE_SEPARABILITY` (0.5, the coin-flip line) and as **overlapping** when it
is at or below `DEFAULT_OVERLAP_SEPARABILITY` *and* centroid similarity is high enough to
make the overlap worth a developer's attention. Overlapping pairs are a lower-severity
observation, not a warning.

Two honest limitations, stated here because the alternative is a developer over-trusting
the output:

1. **This is a lexical instrument.** What it sees is vocabulary overlap weighted by how
   much each term narrows down the command. Synonymous *verbs* do not defeat it — the
   ``tests/duplicate_capability_workflow`` control pair says "list/pull/give" against
   "search/find/look/get" and still scores 0.00, because the terms that carry the weight
   are the shared domain nouns. What defeats it is a duplicate pair whose vocabulary is
   genuinely disjoint end to end: two commands that mean the same thing and share no
   distinctive term. Nothing lexical can see that, and for it you need
   `find_confusable_commands`, which asks the trained model.
2. **Leave-one-out is biased downward on small utterance sets**: removing an utterance
   from its own centroid shrinks self-similarity while the comparison centroid keeps all
   its mass. The 0.5 cut is therefore more permissive than a true chance line would be.
   The empirical headroom on the shipped retail workflow is reported in the R9b notes:
   its worst pair scores 0.600 against a 0.5 threshold.

The second instrument
---------------------
`find_confusable_commands` takes a ``predict_fn(utterance) -> ranked labels`` — the same
contract `heldout_evaluation` uses and exactly what ``CommandRouter.predict`` returns — and
measures how often each command's own utterances are routed to another command. That is the
"cluster label centroids" half of R9b, done through the model's decisions rather than its
weights: it needs no torch import here, it works with any model, and it reports the thing a
developer actually cares about (what the router does) rather than a distance in a space
nobody can inspect.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

from pydantic import BaseModel, Field

import fastworkflow

COMMAND_INFO_DIRNAME = "___command_info"
REPORT_FILENAME = "duplicate_capabilities.json"
REPORT_SCHEMA_VERSION = 1

#: Separability at or below which a pair is reported as a duplicate. 0.5 is the chance
#: line for a balanced two-class decision, not a fitted constant.
DEFAULT_DUPLICATE_SEPARABILITY = 0.50

#: Separability at or below which a pair may be reported as merely overlapping, provided
#: it also clears `DEFAULT_OVERLAP_SIMILARITY`.
DEFAULT_OVERLAP_SEPARABILITY = 0.65

#: Centroid cosine a pair must reach before an overlapping (not duplicate) verdict is
#: worth reporting. Two commands that share little vocabulary but score poorly on
#: separability are usually just short on utterances, which the notes already say.
DEFAULT_OVERLAP_SIMILARITY = 0.50

#: Pre-filter only, on cost grounds: a workflow with 160 commands has 12,720 pairs, and
#: the separability computation is quadratic in utterances. Pairs below this centroid
#: cosine share almost no vocabulary, so their separability is ~1.0 and computing it is
#: wasted work. Raising it makes the scan faster and can in principle suppress a finding;
#: it is not part of the definition of a duplicate.
DEFAULT_SIMILARITY_PREFILTER = 0.10

#: A separability estimate over fewer utterances than this is dominated by which
#: particular sentences the author happened to write. Such pairs are skipped and counted,
#: never silently scored.
DEFAULT_MIN_UTTERANCES = 3

#: Symmetric misroute rate at or above which `find_confusable_commands` reports a pair.
DEFAULT_CONFUSION_THRESHOLD = 0.40

#: Reserved, non-routable labels. They are not capabilities and must never be reported as
#: duplicating one; ``parameter_value``'s contentless literals in particular are near
#: everything and nothing.
NON_CAPABILITY_LABELS: frozenset[str] = frozenset({"wildcard", "parameter_value"})

_TOKEN_RE = re.compile(r"[a-z0-9']+")

#: Terms of one character are punctuation debris after tokenisation; they carry no signal
#: and inflate every vector's norm.
_MIN_TOKEN_LENGTH = 2

PredictFn = Callable[[str], Sequence[str]]


# ---------------------------------------------------------------------------
# Vector space
# ---------------------------------------------------------------------------


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens of *text*, discarding single characters."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) >= _MIN_TOKEN_LENGTH]


def _sublinear_tf(count: int) -> float:
    """1 + log(count). Saturates a term repeated in every utterance of one command.

    Without it, a command whose ten seeds all say "order" would have "order" dominate its
    vector purely by repetition, and two such commands would look alike for a reason that
    says nothing about what they do.
    """
    return 1.0 + math.log(count) if count > 0 else 0.0


def document_frequencies(utterances_by_command: Mapping[str, Sequence[str]]) -> dict[str, int]:
    """Count, per term, how many *commands* use it at least once.

    Document frequency is over commands, not utterances: the question IDF is being asked
    is "how much does this term narrow down which command the user means", and that is a
    per-command quantity.
    """
    df: Counter = Counter()
    for utterances in utterances_by_command.values():
        terms: set[str] = set()
        for utterance in utterances:
            terms.update(tokenize(utterance))
        df.update(terms)
    return dict(df)


def inverse_document_frequencies(
    document_frequencies_map: Mapping[str, int], num_documents: int
) -> dict[str, float]:
    """Smoothed IDF: ``log((1 + N) / (1 + df)) + 1``.

    The +1s keep a term used by every command at a small positive weight rather than
    exactly zero, so a pair whose vocabulary is *entirely* shared still produces a
    well-defined vector instead of a zero one.
    """
    return {
        term: math.log((1.0 + num_documents) / (1.0 + df)) + 1.0
        for term, df in document_frequencies_map.items()
    }


def tfidf_vector(text_or_texts, idf: Mapping[str, float]) -> dict[str, float]:
    """L2-normalised TF-IDF vector of a string or a collection of strings.

    Terms absent from *idf* (which can only happen when scoring text that was not part of
    the corpus the IDF table was built from) get weight 1.0 — the weight of a term seen in
    every command, i.e. the most conservative assumption available.
    """
    counts: Counter = Counter()
    if isinstance(text_or_texts, str):
        counts.update(tokenize(text_or_texts))
    else:
        for text in text_or_texts:
            counts.update(tokenize(text))

    vector = {
        term: _sublinear_tf(count) * idf.get(term, 1.0)
        for term, count in counts.items()
    }
    norm = math.sqrt(sum(weight * weight for weight in vector.values()))
    if not norm:
        return {}
    return {term: weight / norm for term, weight in vector.items()}


def cosine(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    """Cosine similarity of two L2-normalised sparse vectors."""
    if len(a) > len(b):
        a, b = b, a
    return sum(weight * b.get(term, 0.0) for term, weight in a.items())


def _centroid(vectors: Sequence[Mapping[str, float]]) -> dict[str, float]:
    accumulator: dict[str, float] = defaultdict(float)
    for vector in vectors:
        for term, weight in vector.items():
            accumulator[term] += weight
    norm = math.sqrt(sum(weight * weight for weight in accumulator.values()))
    if not norm:
        return {}
    return {term: weight / norm for term, weight in accumulator.items()}


# ---------------------------------------------------------------------------
# Separability
# ---------------------------------------------------------------------------


class PairSeparability(BaseModel):
    """How well a two-class classifier restricted to one pair can do."""

    #: Mean of the two per-command recalls. 0.5 is chance; 1.0 is perfectly separable.
    balanced_accuracy: float
    recall_a: float
    recall_b: float
    utterances_a: int
    utterances_b: int


def pairwise_separability(
    utterances_a: Sequence[str],
    utterances_b: Sequence[str],
    idf: Mapping[str, float],
) -> Optional[PairSeparability]:
    """Leave-one-out balanced nearest-centroid accuracy for one pair.

    Each utterance is classified against its own command's centroid computed **without**
    it and the other command's full centroid. Returns None when either side has fewer
    than two utterances, at which point a leave-one-out centroid does not exist.

    Ties go to the *other* command. A tie means the utterance is exactly as close to the
    two centroids, which is the definition of the classifier having no information; scoring
    it as a success would inflate separability precisely on the pairs this module exists
    to find.
    """
    vectors_a = [tfidf_vector(u, idf) for u in utterances_a]
    vectors_b = [tfidf_vector(u, idf) for u in utterances_b]
    if len(vectors_a) < 2 or len(vectors_b) < 2:
        return None

    recalls: list[float] = []
    for own, other in ((vectors_a, vectors_b), (vectors_b, vectors_a)):
        other_centroid = _centroid(other)
        correct = 0
        for index, vector in enumerate(own):
            own_centroid = _centroid(own[:index] + own[index + 1:])
            if cosine(vector, own_centroid) > cosine(vector, other_centroid):
                correct += 1
        recalls.append(correct / len(own))

    return PairSeparability(
        balanced_accuracy=(recalls[0] + recalls[1]) / 2.0,
        recall_a=recalls[0],
        recall_b=recalls[1],
        utterances_a=len(vectors_a),
        utterances_b=len(vectors_b),
    )


def distinguishing_terms(
    utterances_a: Sequence[str],
    utterances_b: Sequence[str],
    limit: int = 5,
) -> tuple[list[str], list[str], list[str]]:
    """Return ``(only_in_a, only_in_b, shared)`` terms, ranked by coverage.

    Purely explanatory: it answers the developer's immediate next question, which is
    always "why does the tool think these are the same?" or "what should I change?".
    Coverage is the fraction of a command's utterances containing the term, so a term
    appearing in one stray seed does not outrank one appearing in all of them.
    """

    def coverage(utterances: Sequence[str]) -> dict[str, float]:
        if not utterances:
            return {}
        counts: Counter = Counter()
        for utterance in utterances:
            counts.update(set(tokenize(utterance)))
        return {term: count / len(utterances) for term, count in counts.items()}

    cov_a = coverage(utterances_a)
    cov_b = coverage(utterances_b)

    only_a = sorted(
        (t for t in cov_a if t not in cov_b), key=lambda t: (-cov_a[t], t)
    )[:limit]
    only_b = sorted(
        (t for t in cov_b if t not in cov_a), key=lambda t: (-cov_b[t], t)
    )[:limit]
    shared = sorted(
        (t for t in cov_a if t in cov_b),
        key=lambda t: (-min(cov_a[t], cov_b[t]), t),
    )[:limit]
    return only_a, only_b, shared


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------


class DuplicateFinding(BaseModel):
    """One reported pair, with everything needed to judge it without rerunning anything."""

    command_a: str
    command_b: str
    #: "duplicate" or "overlapping".
    severity: str
    separability: float
    recall_a: float
    recall_b: float
    centroid_similarity: float
    utterances_a: int
    utterances_b: int
    #: Contexts whose label space contains BOTH commands. A pair that co-occurs is a live
    #: classifier conflict; a pair in disjoint contexts is a design ambiguity that surfaces
    #: through the escalation class and the parent walk instead. The developer's response
    #: differs, so the two are not merged into one verdict.
    shared_contexts: list[str] = Field(default_factory=list)
    terms_only_in_a: list[str] = Field(default_factory=list)
    terms_only_in_b: list[str] = Field(default_factory=list)
    shared_terms: list[str] = Field(default_factory=list)

    @property
    def pair(self) -> tuple[str, str]:
        return (self.command_a, self.command_b)


class ConfusionFinding(BaseModel):
    """One pair the trained model itself routes interchangeably."""

    command_a: str
    command_b: str
    #: Mean of the two directional misroute rates, so the number does not depend on which
    #: command has more utterances.
    symmetric_confusion: float
    a_routed_to_b: float
    b_routed_to_a: float
    cases_a: int
    cases_b: int


class DuplicateReport(BaseModel):
    """The result of one scan, in the shape written to disk and printed."""

    workflow_folderpath: Optional[str] = None
    commands_examined: int = 0
    pairs_examined: int = 0
    pairs_skipped_too_few_utterances: int = 0
    duplicates: list[DuplicateFinding] = Field(default_factory=list)
    overlapping: list[DuplicateFinding] = Field(default_factory=list)
    confusable: list[ConfusionFinding] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    thresholds: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Lexical detection
# ---------------------------------------------------------------------------


def find_duplicate_capabilities(
    utterances_by_command: Mapping[str, Sequence[str]],
    contexts: Optional[Mapping[str, Sequence[str]]] = None,
    duplicate_separability: float = DEFAULT_DUPLICATE_SEPARABILITY,
    overlap_separability: float = DEFAULT_OVERLAP_SEPARABILITY,
    overlap_similarity: float = DEFAULT_OVERLAP_SIMILARITY,
    similarity_prefilter: float = DEFAULT_SIMILARITY_PREFILTER,
    min_utterances: int = DEFAULT_MIN_UTTERANCES,
    workflow_folderpath: Optional[str] = None,
) -> DuplicateReport:
    """Scan every command pair and report the ones the training data does not separate.

    *utterances_by_command* maps a fully-qualified command name to its utterances. Seed
    utterances are enough and are what `utterances_from_workflow` returns; generated
    utterances work too and give a sharper estimate, since they are what is actually
    trained on.

    *contexts* optionally maps a context name to the commands in its label space, used to
    populate `DuplicateFinding.shared_contexts`.
    """
    corpus = {
        command: list(utterances)
        for command, utterances in utterances_by_command.items()
        if command.split("/")[-1] not in NON_CAPABILITY_LABELS and utterances
    }

    report = DuplicateReport(
        workflow_folderpath=workflow_folderpath,
        commands_examined=len(corpus),
        thresholds={
            "duplicate_separability": duplicate_separability,
            "overlap_separability": overlap_separability,
            "overlap_similarity": overlap_similarity,
            "similarity_prefilter": similarity_prefilter,
            "min_utterances": min_utterances,
        },
    )
    if len(corpus) < 2:
        report.notes.append(
            "Fewer than two commands with utterances; no pair could be examined."
        )
        return report

    idf = inverse_document_frequencies(document_frequencies(corpus), len(corpus))
    centroids = {
        command: tfidf_vector(utterances, idf) for command, utterances in corpus.items()
    }
    contexts_by_command = _contexts_by_command(contexts)

    names = sorted(corpus)
    for index, command_a in enumerate(names):
        for command_b in names[index + 1:]:
            similarity = cosine(centroids[command_a], centroids[command_b])
            if similarity < similarity_prefilter:
                continue

            utterances_a = corpus[command_a]
            utterances_b = corpus[command_b]
            if min(len(utterances_a), len(utterances_b)) < min_utterances:
                report.pairs_skipped_too_few_utterances += 1
                continue

            report.pairs_examined += 1
            separability = pairwise_separability(utterances_a, utterances_b, idf)
            if separability is None:
                report.pairs_skipped_too_few_utterances += 1
                continue

            score = separability.balanced_accuracy
            if score <= duplicate_separability:
                severity = "duplicate"
            elif score <= overlap_separability and similarity >= overlap_similarity:
                severity = "overlapping"
            else:
                continue

            only_a, only_b, shared = distinguishing_terms(utterances_a, utterances_b)
            finding = DuplicateFinding(
                command_a=command_a,
                command_b=command_b,
                severity=severity,
                separability=score,
                recall_a=separability.recall_a,
                recall_b=separability.recall_b,
                centroid_similarity=similarity,
                utterances_a=separability.utterances_a,
                utterances_b=separability.utterances_b,
                shared_contexts=sorted(
                    contexts_by_command.get(command_a, set())
                    & contexts_by_command.get(command_b, set())
                ),
                terms_only_in_a=only_a,
                terms_only_in_b=only_b,
                shared_terms=shared,
            )
            if severity == "duplicate":
                report.duplicates.append(finding)
            else:
                report.overlapping.append(finding)

    report.duplicates.sort(key=lambda f: (f.separability, f.command_a, f.command_b))
    report.overlapping.sort(key=lambda f: (f.separability, f.command_a, f.command_b))

    if report.pairs_skipped_too_few_utterances:
        report.notes.append(
            f"{report.pairs_skipped_too_few_utterances} pair(s) were skipped because one "
            f"side had fewer than {min_utterances} utterances. A separability estimate "
            f"over so few sentences reflects which ones the author happened to write."
        )
    return report


def _contexts_by_command(
    contexts: Optional[Mapping[str, Sequence[str]]],
) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    for context_name, commands in (contexts or {}).items():
        for command in commands:
            result[command].add(context_name)
    return result


# ---------------------------------------------------------------------------
# Model-based detection
# ---------------------------------------------------------------------------


def find_confusable_commands(
    utterances_by_command: Mapping[str, Sequence[str]],
    predict_fn: PredictFn,
    confusion_threshold: float = DEFAULT_CONFUSION_THRESHOLD,
    use_top1_only: bool = True,
) -> list[ConfusionFinding]:
    """Report pairs the trained model routes interchangeably.

    ``predict_fn(utterance)`` returns ranked candidate labels, top-1 first — the contract
    ``CommandRouter.predict`` already satisfies and the one `heldout_evaluation` uses.

    This is the semantic complement to `find_duplicate_capabilities`. Two commands with no
    distinctive term in common are lexically separable but can still land in the same
    region of the fine-tuned encoder's space, and only the model can say so.

    Run on utterances the model **trained on**, a non-zero confusion rate is a strong
    signal: the model failed to separate them even with the answer in its training set.
    Run on held-out utterances it is a weaker but more realistic one. Either is valid; the
    caller chooses what to feed in, and the report does not guess which was used.
    """
    scanned = {
        command
        for command, utterances in utterances_by_command.items()
        if utterances and command.split("/")[-1] not in NON_CAPABILITY_LABELS
    }

    misroutes: dict[tuple[str, str], int] = defaultdict(int)
    totals: dict[str, int] = defaultdict(int)

    for command in sorted(scanned):
        for utterance in utterances_by_command[command]:
            # Not `predict_fn(...) or []`: CommandRouter.predict returns the top-k labels
            # as a numpy array in the unconfident branch, and truth-testing an array of
            # more than one element raises.
            raw_predictions = predict_fn(utterance)
            predictions = (
                [] if raw_predictions is None else [str(p) for p in raw_predictions]
            )
            totals[command] += 1
            candidates = predictions[:1] if use_top1_only else predictions
            for predicted in candidates:
                # A misroute to a command outside the scan cannot be scored symmetrically
                # -- there is no denominator for the other direction -- so it is left to
                # the routing metrics in heldout_evaluation, which do measure it.
                if predicted == command or predicted not in scanned:
                    continue
                misroutes[(command, predicted)] += 1

    findings: list[ConfusionFinding] = []
    seen: set[tuple[str, str]] = set()
    for source, target in misroutes:
        command_a, command_b = sorted((source, target))
        pair = (command_a, command_b)
        if pair in seen:
            continue
        seen.add(pair)
        rate_a = misroutes.get((command_a, command_b), 0) / (totals.get(command_a) or 1)
        rate_b = misroutes.get((command_b, command_a), 0) / (totals.get(command_b) or 1)
        symmetric = (rate_a + rate_b) / 2.0
        if symmetric < confusion_threshold:
            continue
        findings.append(
            ConfusionFinding(
                command_a=command_a,
                command_b=command_b,
                symmetric_confusion=symmetric,
                a_routed_to_b=rate_a,
                b_routed_to_a=rate_b,
                cases_a=totals.get(command_a, 0),
                cases_b=totals.get(command_b, 0),
            )
        )

    findings.sort(
        key=lambda f: (-f.symmetric_confusion, f.command_a, f.command_b)
    )
    return findings


# ---------------------------------------------------------------------------
# Workflow input
# ---------------------------------------------------------------------------


def utterances_from_workflow(workflow_folderpath: str) -> dict[str, list[str]]:
    """Collect each command's hand-written seed utterances from a workflow folder.

    Seeds, not generated utterances, so this runs before training, without an LLM key and
    without a network call. Two commands whose *seeds* do not separate cannot be rescued
    by generation: every generated utterance is conditioned on the seed list and on a
    keyword bag built from it (`generate_synthetic.py:249-257`), so generation amplifies
    the seeds' vocabulary rather than adding to it.
    """
    crd = fastworkflow.RoutingRegistry.get_definition(workflow_folderpath)
    command_directory = crd.command_directory

    utterances_by_command: dict[str, list[str]] = {}
    for command in sorted(command_directory.get_commands()):
        if command.split("/")[-1] in NON_CAPABILITY_LABELS:
            continue
        utterance_metadata = command_directory.get_utterance_metadata(command)
        if not utterance_metadata:
            continue
        seeds = list(utterance_metadata.plain_utterances) + list(
            utterance_metadata.template_utterances
        )
        if seeds:
            utterances_by_command[command] = seeds
    return utterances_by_command


def contexts_from_workflow(workflow_folderpath: str) -> dict[str, list[str]]:
    """Return ``context name -> commands in its label space``."""
    crd = fastworkflow.RoutingRegistry.get_definition(workflow_folderpath)
    return {
        context_name: sorted(commands)
        for context_name, commands in crd.contexts.items()
    }


def scan_workflow(
    workflow_folderpath: str,
    **kwargs,
) -> DuplicateReport:
    """Convenience: collect a workflow's seeds and contexts, then scan them."""
    return find_duplicate_capabilities(
        utterances_from_workflow(workflow_folderpath),
        contexts=contexts_from_workflow(workflow_folderpath),
        workflow_folderpath=workflow_folderpath,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _format_finding(finding: DuplicateFinding) -> list[str]:
    where = (
        f"both in context(s) {', '.join(finding.shared_contexts)}"
        if finding.shared_contexts
        else "no shared context (surfaces via escalation and the parent walk)"
    )
    lines = [
        f"  {finding.command_a}",
        f"  {finding.command_b}",
        f"    separability {finding.separability:.2f} "
        f"(per-command {finding.recall_a:.2f} / {finding.recall_b:.2f}), "
        f"centroid similarity {finding.centroid_similarity:.2f}, "
        f"{finding.utterances_a} vs {finding.utterances_b} utterances",
        f"    {where}",
    ]
    if finding.shared_terms:
        lines.append(f"    shared terms: {', '.join(finding.shared_terms)}")
    lines.append(
        f"    only in first: {', '.join(finding.terms_only_in_a) or '(none)'}"
    )
    lines.append(
        f"    only in second: {', '.join(finding.terms_only_in_b) or '(none)'}"
    )
    return lines


def format_report(report: DuplicateReport) -> str:
    """Render the scan for the end of a training run."""
    lines = [
        "Near-duplicate capability scan (R9b)",
        "-" * 78,
        f"{report.commands_examined} command(s), {report.pairs_examined} pair(s) examined.",
    ]

    if not report.duplicates and not report.overlapping and not report.confusable:
        lines.append("No near-duplicate capabilities detected.")
    else:
        if report.duplicates:
            lines.append("")
            lines.append(
                f"DUPLICATE CAPABILITIES ({len(report.duplicates)}): the training data does "
                f"not separate these pairs. Merge, alias, or knowingly accept true "
                f"duplicates. For legitimate neighbours or opposites, write more distinctive "
                f"seed utterances and re-run the scan."
            )
            for finding in report.duplicates:
                lines.extend(_format_finding(finding))
        if report.overlapping:
            lines.append("")
            lines.append(
                f"OVERLAPPING ({len(report.overlapping)}): separable, but only just. Usually "
                f"a seed list that describes its neighbour in passing. Not a defect."
            )
            for finding in report.overlapping:
                lines.extend(_format_finding(finding))
        if report.confusable:
            lines.append("")
            lines.append(
                f"MODEL CONFUSION ({len(report.confusable)}): the trained router sends these "
                f"commands' own utterances to each other."
            )
            for confusion in report.confusable:
                lines.append(
                    f"  {confusion.command_a} <-> {confusion.command_b}: "
                    f"{confusion.symmetric_confusion:.0%} symmetric "
                    f"({confusion.a_routed_to_b:.0%} / {confusion.b_routed_to_a:.0%} over "
                    f"{confusion.cases_a} / {confusion.cases_b} utterances)"
                )

    if report.notes:
        lines.append("")
        lines.append("Notes:")
        lines.extend(f"  {note}" for note in report.notes)

    lines.append("")
    lines.append(
        "This scan reports only. It does not change what is trained or what any model "
        "predicts. It is a lexical instrument: a duplicate pair sharing no distinctive "
        "term is separable here and will not appear above."
    )
    return "\n".join(lines)


def write_report(workflow_folderpath: str, report: DuplicateReport) -> str:
    """Write the JSON report to ``<workflow>/___command_info/duplicate_capabilities.json``."""
    output_dir = Path(workflow_folderpath) / COMMAND_INFO_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / REPORT_FILENAME

    payload = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "definition": (
            "A pair is a near-duplicate when a nearest-centroid classifier restricted to "
            "that pair, scored leave-one-out and balanced over the two commands, cannot "
            "beat chance on their own utterances. Lexical only: a duplicate pair sharing "
            "no distinctive vocabulary at all scores as separable here and is missed."
        ),
        "report": report.model_dump(),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    return str(output_path)

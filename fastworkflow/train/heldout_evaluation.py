"""Held-out evaluation for fastWorkflow intent models (R1a / R1b).

Why this module exists
----------------------
`model_pipeline_training.train()` reports a weighted F1 computed on
``train_test_split(dataset, test_size=0.25, random_state=42)`` — a random split of the
*same* synthetic utterance set the model trains on. Every utterance for one command comes
from a handful of personas expanding one seed list, so the 25% "test" rows are near
duplicates of the 75% "train" rows. That number measures memorisation. Measured gap on a
160-command workflow: ~0.94 reported F1 against 46.2% held-out top-1.

This module provides the missing held-out concept, on two axes that are deliberately
reported separately (decision D2 — they trade against each other and one blended score
hides the trade):

* **routing** — top-1 and in-list accuracy on phrasings the model did not train on.
* **escalation** — recall of the "this command lives upstairs" signal. A phrasing aimed at
  a command that is provably absent from the tested context's label space but present in
  one of its ancestors must produce a *confident, lone* escalation label, because only a
  lone escalation label makes the runtime walk the parent chain
  (``_commands/wildcard.py:100-104``). A escalation label buried in a top-k list takes the
  ambiguity branch instead and the signal is discarded (finding F7).

Two sources of held-out data are supported:

1. **Persona holdout** (`split_by_persona`) — whole personas are reserved, never a random
   sample of rows (decision D1). Holding out 25% of the rows of a persona whose other rows
   are in training is the defect, not the fix.
2. **A developer-supplied benchmark file** (`load_benchmark_file`) — hand-written
   phrasings, enforced disjoint from the seed table (`assert_benchmark_disjoint_from_seeds`).
   Without that enforcement a benchmark silently decays into a memorisation test the first
   time someone pastes a failing case into their seeds to "fix" it.

The module never loads a model. Scoring takes a caller-supplied
``predict_fn(utterance) -> list[str]`` returning the ranked candidate labels (top-1 first),
which is exactly what ``CommandRouter.predict`` returns. That keeps this module importable
and testable without torch.

The legacy in-distribution F1 is preserved on `HeldoutReport`, explicitly named
`in_distribution_f1`, so the number that has been quietly misleading people is still
available and no longer misread.
"""

from __future__ import annotations

import difflib
import json
import math
import random
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Literal, Mapping, Sequence

from pydantic import BaseModel, Field

# The reserved-label vocabulary (R7.1). Standard-library only by design, so importing
# it here costs nothing and keeps the escalation set from drifting between the trainer,
# the runtime and this module.
from fastworkflow.nlu_labels import ESCALATION_LABELS as _ESCALATION_LABELS
from fastworkflow.nlu_labels import NON_ROUTABLE_LABELS

#: Rows a label needs to survive training. Owned here rather than in the trainer only
#: because the import runs this way; ``split_training_data`` is what enforces it, and
#: the persona split must not hand it a label it will reject. One number, two callers.
MIN_TRAINING_ROWS_PER_LABEL = 2
# Provided by the determinism work (R2). It owns persona attribution; this module only
# consumes it.
# The former fallback kept held-out evaluation independently importable; direct in-package
# imports now intentionally enforce the package contract instead.
from fastworkflow.train.determinism import (
    PERSONA_ID_SEPARATOR,
    SEED_PERSONA_ID,
    UNRESOLVED_PERSONA_PREFIX,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COMMAND_INFO_DIRNAME = "___command_info"
REPORT_FILENAME = "heldout_evaluation.json"
DEFAULT_BENCHMARK_FILENAME = "intent_benchmark.json"
BENCHMARK_SCHEMA_VERSION = 1
# v2 makes routing top-1 mean a lone, confident route rather than merely the
# first-ranked label in an ambiguity list.
REPORT_SCHEMA_VERSION = 2

DEFAULT_HOLDOUT_FRACTION = 0.25
DEFAULT_SEED = 42

#: Labels whose *lone, confident* prediction counts as a correct escalation.
#:
#: Sourced from `nlu_labels`, which is the single owner of the reserved-label vocabulary
#: and is deliberately dependency-light so both the trainer and this module can import it.
#: Note this is the ESCALATION set, not the non-routable set: R7.1 split the bare-value
#: catcher out under `parameter_value`, and that label must NOT count as an escalation —
#: a bare value says nothing about whether an ancestor context can serve the request.
#: Callers can still pass their own set to `score_escalation`.
DEFAULT_ESCALATION_LABELS: frozenset[str] = _ESCALATION_LABELS

#: Similarity at or above which a benchmark utterance is *reported* as a near-duplicate of
#: a seed utterance. Near-duplicates warn; exact (normalised) matches fail the run.
DEFAULT_NEAR_DUPLICATE_THRESHOLD = 0.90

PredictFn = Callable[[str], Sequence[str]]


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

_WHITESPACE_RE = re.compile(r"\s+")

# Stripped from both ends of an utterance before comparison. An exact string match is too
# weak to catch the realistic version of the seed/benchmark leak: someone pastes a failing
# benchmark case into a seed list and drops the trailing period while doing it.
_EDGE_PUNCTUATION = "\"'`*.,;:!?…()[]{}"

_SMART_QUOTES = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
}


def normalize_utterance(text: str) -> str:
    """Return the comparison form of *text*.

    NFKC-normalise, fold smart quotes to ASCII, collapse internal whitespace, strip
    surrounding whitespace and punctuation, casefold. ``"Close the account."`` and
    ``"close the account"`` both normalise to ``"close the account"``.
    """
    normalized = unicodedata.normalize("NFKC", text)
    for smart, plain in _SMART_QUOTES.items():
        normalized = normalized.replace(smart, plain)
    normalized = _WHITESPACE_RE.sub(" ", normalized).strip()
    normalized = normalized.strip(_EDGE_PUNCTUATION + " ")
    return normalized.casefold()


def normalize_label(label: str) -> str:
    """Return the comparison form of a label.

    Labels are fully-qualified (``Context/command``) and must match the trained
    ``LabelEncoder`` classes exactly, so this only trims whitespace. It exists so callers
    have one place to look rather than guessing whether comparison is fuzzy.
    """
    return label.strip()


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class LabeledUtterance(BaseModel):
    """One training row plus the persona that produced it."""

    utterance: str
    label: str
    #: ``SEED_PERSONA_ID`` for hand-written seed utterances and the command-name token.
    persona: str = SEED_PERSONA_ID


class PersonaSplit(BaseModel):
    """The result of reserving whole personas for evaluation."""

    train: list[LabeledUtterance] = Field(default_factory=list)
    heldout: list[LabeledUtterance] = Field(default_factory=list)
    train_personas: list[str] = Field(default_factory=list)
    heldout_personas: list[str] = Field(default_factory=list)
    #: Anything the caller should know before trusting the numbers: labels with no
    #: held-out coverage, labels rescued from having zero training rows, leaked
    #: utterances dropped, single-persona labels, an empty holdout.
    notes: list[str] = Field(default_factory=list)


class RoutingScore(BaseModel):
    """Top-1 and in-list accuracy over a set of routing cases."""

    total: int = 0
    top1_correct: int = 0
    in_list_correct: int = 0
    top1: float = 0.0
    in_list: float = 0.0
    #: command -> {"total": int, "top1_correct": int, "in_list_correct": int}
    per_command: dict[str, dict] = Field(default_factory=dict)


class EscalationScore(BaseModel):
    """Recall of the confident-escalation signal over a set of escalation cases."""

    total: int = 0
    correct: int = 0
    recall: float = 0.0
    failures: list[dict] = Field(default_factory=list)


class HeldoutReport(BaseModel):
    """Per-context evaluation result written at the end of a training run."""

    context: str
    #: The legacy same-distribution split score. Kept because it is what previous runs
    #: reported; named so it can no longer be mistaken for a generalisation measure.
    in_distribution_f1: float | None = None
    routing: RoutingScore | None = None
    #: Persona-held-out rows whose expected label is an escalation class rather than a
    #: command. Scored on lone-escalation semantics and reported on its own axis: these
    #: rows used to be counted as routing successes, which in a context with ancestors
    #: meant the headline routing number was mostly escalation recognition (D2).
    holdout_escalation: EscalationScore | None = None
    #: Routing scored on the DEVELOPER-SUPPLIED benchmark, kept separate from `routing`
    #: because the two are different populations and must not be averaged. It is also the
    #: only one of the two that can be compared across runs: the persona holdout is
    #: re-drawn every run, so two runs score different cases, whereas the benchmark file
    #: is fixed and its cases are paired by construction.
    benchmark_routing: RoutingScore | None = None
    escalation: EscalationScore | None = None
    seed: int = DEFAULT_SEED
    heldout_personas: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class BenchmarkCase(BaseModel):
    """One hand-written benchmark case."""

    #: Which context's classifier this case tests.
    context: str
    utterance: str
    #: Fully-qualified expected label. Required for ``kind == "routing"``.
    expected_label: str | None = None
    kind: Literal["routing", "escalation"] = "routing"
    #: Escalation only: the command that IS present in an ancestor of ``context`` and
    #: absent from ``context``'s own label space.
    expected_ancestor_command: str | None = None


class BenchmarkLeakError(Exception):
    """Raised when benchmark utterances overlap the seed table.

    A benchmark that shares utterances with the seeds the model trained on measures
    memorisation, which is the defect this whole module exists to remove. The run fails
    rather than reporting a number that cannot mean what it appears to mean.
    """


# ---------------------------------------------------------------------------
# Persona splitting (R1a, decision D1)
# ---------------------------------------------------------------------------


def _holdout_persona_count(num_personas: int, holdout_fraction: float) -> int:
    """How many personas to reserve, leaving at least one for training."""
    if num_personas < 2 or holdout_fraction <= 0.0:
        return 0
    # floor(x + 0.5) rather than round(), which is banker's rounding: round(0.5) == 0.
    count = int(math.floor(num_personas * holdout_fraction + 0.5))
    return max(1, min(count, num_personas - 1))


def expand_persona_id(persona_id: str) -> frozenset[str]:
    """Return the atomic personas a possibly-composite persona id refers to.

    The determinism module attributes one utterance *text* to every persona that produced
    it, joining the ids with ``PERSONA_ID_SEPARATOR`` (``"1041+37822"``) and prefixing
    ``UNRESOLVED_PERSONA_PREFIX`` when the producing persona could not be resolved from the
    LLM's echoed header (in which case the suffix lists every persona in the batch).
    The canonical prefix covers the whole union, but each separator-delimited atom is
    checked defensively so legacy incrementally merged forms cannot create a phantom
    ``"__unresolved__:<id>"`` persona.

    Treating those strings as opaque would break the whole-persona holdout: ``"1041+37822"``
    would become a persona in its own right and could be reserved for evaluation while
    persona ``37822``'s other utterances sat in training — a leak of exactly the kind D1
    exists to prevent.
    """
    if persona_id == SEED_PERSONA_ID:
        return frozenset({SEED_PERSONA_ID})
    parts: set[str] = set()
    for part in persona_id.split(PERSONA_ID_SEPARATOR):
        while part.startswith(UNRESOLVED_PERSONA_PREFIX):
            part = part[len(UNRESOLVED_PERSONA_PREFIX):]
        if part:
            parts.add(part)
    return frozenset(parts)


def _summarize_labels(labels: Iterable[str], limit: int = 10) -> str:
    ordered = sorted(labels)
    shown = ", ".join(ordered[:limit])
    if len(ordered) > limit:
        shown += f", ... (+{len(ordered) - limit} more)"
    return shown


def split_by_persona(
    records: Sequence[LabeledUtterance],
    holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION,
    seed: int = DEFAULT_SEED,
) -> PersonaSplit:
    """Reserve whole personas for evaluation.

    Personas are ordered deterministically (sorted by persona id) and selected with a
    seeded RNG, so the same ``seed`` always yields the same held-out persona set.

    Rules, all of which exist because the alternative silently produces a meaningless
    number:

    * ``SEED_PERSONA_ID`` records always go to ``train``. Hand-written seeds are the
      developer's declared input, not generalisation data.
    * A label that would end up with **zero training rows** has all of its held-out rows
      returned to training, and the fact is noted. A label missing from training makes the
      command unroutable — the same class of bug as F3, and never an acceptable price for
      an evaluation metric.
    * An utterance produced by both a held-out and a training persona is a leak; the
      held-out copy is dropped (the conservative choice) and the count is noted.
    * Labels with only one persona, labels with no held-out coverage, and an empty holdout
      are all reported in ``notes`` rather than passing silently.
    """
    records = list(records)
    notes: list[str] = []

    if not records:
        notes.append("No labeled utterances supplied; nothing to hold out.")
        return PersonaSplit(notes=notes)

    # Composite ids ("1041+37822") are expanded to their contributors before anything is
    # counted or sampled, so a persona is reserved as a whole even when some of its
    # utterances were also produced by another persona. See `expand_persona_id`.
    contributors_by_record = [expand_persona_id(record.persona) for record in records]

    all_personas = sorted(
        {
            persona
            for contributors in contributors_by_record
            for persona in contributors
        }
        - {SEED_PERSONA_ID}
    )

    personas_by_label: dict[str, set[str]] = defaultdict(set)
    for record, contributors in zip(records, contributors_by_record):
        if named := contributors - {SEED_PERSONA_ID}:
            personas_by_label[record.label].update(named)

    single_persona_labels = {
        label for label, personas in personas_by_label.items() if len(personas) < 2
    }
    labels_without_personas = {
        record.label for record in records
    } - set(personas_by_label)
    single_persona_labels |= labels_without_personas
    if single_persona_labels:
        notes.append(
            f"{len(single_persona_labels)} label(s) are covered by fewer than 2 personas, "
            f"so a persona holdout cannot be an independent generalisation test for them: "
            f"{_summarize_labels(single_persona_labels)}"
        )

    holdout_count = _holdout_persona_count(len(all_personas), holdout_fraction)
    if holdout_count == 0:
        notes.append(
            f"Only {len(all_personas)} persona(s) available at holdout_fraction="
            f"{holdout_fraction}; no personas could be reserved. Held-out routing is not "
            f"measurable for this context."
        )
        return PersonaSplit(
            train=records,
            heldout=[],
            train_personas=all_personas,
            heldout_personas=[],
            notes=notes,
        )

    rng = random.Random(seed)
    heldout_personas = sorted(rng.sample(all_personas, holdout_count))
    heldout_persona_set = set(heldout_personas)
    train_personas = [p for p in all_personas if p not in heldout_persona_set]

    train: list[LabeledUtterance] = []
    heldout: list[LabeledUtterance] = []
    for record, contributors in zip(records, contributors_by_record):
        named = contributors - {SEED_PERSONA_ID}
        # An utterance is held out only when EVERY persona that produced it is held out.
        # If any contributor is still training, the model has seen this phrasing's author
        # and scoring it would measure memorisation.
        if named and named <= heldout_persona_set:
            heldout.append(record)
        else:
            train.append(record)

    # Guard: a label must never drop below the trainer's floor. Rescue it wholesale
    # rather than leaving a partially-trained label, which would corrupt the model to
    # buy a metric.
    #
    # The threshold is MIN_TRAINING_ROWS_PER_LABEL, not zero. Rescuing only at zero
    # left a label with exactly one training row -- enough to pass here, one short of
    # what split_training_data demands -- so the run died mid-loop, after earlier
    # contexts had already spent their LLM and GPU budget. Two modules enforcing the
    # same requirement with different numbers is the defect; sharing the constant is
    # the fix.
    trained_counts = Counter(record.label for record in train)
    starved_labels = {
        record.label
        for record in heldout
        if trained_counts[record.label] < MIN_TRAINING_ROWS_PER_LABEL
    }
    trained_labels = set(trained_counts)
    if starved_labels:
        rescued = [r for r in heldout if r.label in starved_labels]
        heldout = [r for r in heldout if r.label not in starved_labels]
        train.extend(rescued)
        notes.append(
            f"{len(starved_labels)} label(s) would have been left below the trainer's "
            f"floor of {MIN_TRAINING_ROWS_PER_LABEL} training row(s) after the persona "
            f"split; their {len(rescued)} held-out row(s) were returned to train, so "
            f"those labels have no held-out coverage: {_summarize_labels(starved_labels)}"
        )

    # Leak: the same utterance text produced by both a held-out and a training persona.
    train_texts = {normalize_utterance(record.utterance) for record in train}
    leaked = [r for r in heldout if normalize_utterance(r.utterance) in train_texts]
    if leaked:
        heldout = [
            r for r in heldout if normalize_utterance(r.utterance) not in train_texts
        ]
        notes.append(
            f"Dropped {len(leaked)} held-out utterance(s) that were also produced by a "
            f"training persona; they would have measured memorisation."
        )

    covered_labels = {record.label for record in heldout}
    uncovered = trained_labels - covered_labels
    if uncovered:
        notes.append(
            f"{len(uncovered)} label(s) have no held-out rows and are therefore unmeasured "
            f"on the routing axis: {_summarize_labels(uncovered)}"
        )

    if not heldout:
        notes.append(
            "Held-out set is empty after guards; held-out routing is not measurable for "
            "this context."
        )

    return PersonaSplit(
        train=train,
        heldout=heldout,
        train_personas=train_personas,
        heldout_personas=heldout_personas,
        notes=notes,
    )


def labeled_utterances_from_provenance(
    entries: Iterable[Mapping],
    default_persona: str = SEED_PERSONA_ID,
) -> list[LabeledUtterance]:
    """Build `LabeledUtterance` records from provenance dictionaries.

    Tolerates the obvious key spellings (``utterance``/``text``,
    ``label``/``command``/``command_name``, ``persona``/``persona_id``) so the caller does
    not have to reshape the determinism module's records by hand. Entries missing an
    utterance or a label are skipped.
    """
    result: list[LabeledUtterance] = []
    for entry in entries:
        utterance = entry.get("utterance") or entry.get("text")
        label = entry.get("label") or entry.get("command") or entry.get("command_name")
        if not utterance or not label:
            continue
        persona = entry.get("persona") or entry.get("persona_id") or default_persona
        result.append(
            LabeledUtterance(utterance=utterance, label=label, persona=str(persona))
        )
    return result


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _expected_routing_label(case) -> str | None:
    label = getattr(case, "label", None)
    if label is None:
        label = getattr(case, "expected_label", None)
    return label


def partition_by_routability(
    cases: Sequence[LabeledUtterance | BenchmarkCase],
) -> tuple[list, list]:
    """Split *cases* into (routable, escalation-class) by their expected label.

    The two populations answer different questions -- "does this reach the right
    command" versus "does this correctly give up and escalate" -- and averaging
    them produces a number that answers neither.
    """
    routable: list = []
    escalation_class: list = []
    for case in cases:
        expected = _expected_routing_label(case)
        target = (
            escalation_class
            if expected is not None and normalize_label(expected) in NON_ROUTABLE_LABELS
            else routable
        )
        target.append(case)
    return routable, escalation_class


def score_routing(
    cases: Sequence[LabeledUtterance | BenchmarkCase],
    predict_fn: PredictFn,
) -> RoutingScore:
    """Score routing cases with *predict_fn*.

    ``predict_fn(utterance)`` returns the ranked candidate labels, top-1 first — exactly
    ``CommandRouter.predict``'s contract (one label when confident, top-k when not).

    * **top-1** — the lone, confident returned label is the expected label.
    * **in-list** — the expected label appears anywhere among the returned candidates.

    The two are reported separately because any multi-candidate result is a clarification
    prompt, not a route, even when the expected label is ranked first.
    """
    score = RoutingScore()
    per_command: dict[str, dict] = {}

    for case in cases:
        expected = _expected_routing_label(case)
        if expected is None:
            raise ValueError(
                f"Routing case has no expected label: {getattr(case, 'utterance', case)!r}"
            )
        expected = normalize_label(expected)
        if expected in NON_ROUTABLE_LABELS:
            # Decision D2: routing and escalation are never blended. A lone
            # wildcard escalates to an ancestor -- it is not a route -- so
            # counting predictions[0] == "wildcard" as a correct top-1 would
            # score a semantics the runtime does not have. Escalation-class rows
            # are 48-91% of rows in contexts with ancestors, so admitting them
            # here does not skew the number, it replaces it. Raised rather than
            # filtered because a caller that reaches this has mixed two
            # populations and needs to say which one it meant.
            raise ValueError(
                f"Non-routable label {expected!r} was passed to score_routing "
                f"(utterance: {getattr(case, 'utterance', case)!r}). Score "
                f"escalation-class rows with score_escalation instead."
            )
        predictions = [normalize_label(p) for p in (predict_fn(case.utterance) or [])]

        bucket = per_command.setdefault(
            expected, {"total": 0, "top1_correct": 0, "in_list_correct": 0}
        )
        bucket["total"] += 1
        score.total += 1

        if len(predictions) == 1 and predictions[0] == expected:
            bucket["top1_correct"] += 1
            score.top1_correct += 1
        if expected in predictions:
            bucket["in_list_correct"] += 1
            score.in_list_correct += 1

    score.per_command = per_command
    if score.total:
        score.top1 = score.top1_correct / score.total
        score.in_list = score.in_list_correct / score.total
    return score


def score_escalation(
    cases: Sequence[BenchmarkCase],
    predict_fn: PredictFn,
    escalation_labels: Iterable[str] = DEFAULT_ESCALATION_LABELS,
) -> EscalationScore:
    """Score escalation cases with *predict_fn*.

    A case is correct only when the prediction is a **lone** escalation label. That is not
    a stylistic choice: only a lone escalation label drives the parent-chain walk in
    ``_commands/wildcard.py:100-104``. An escalation label returned *alongside* local
    candidates takes the ambiguity branch and is filtered out of the message shown to the
    user (finding F7), so counting it as a success would score a behaviour the runtime
    does not have.

    ``escalation_labels`` is a parameter because a second non-routable label may be
    introduced alongside ``wildcard``.
    """
    accepted = {normalize_label(label) for label in escalation_labels}
    score = EscalationScore()

    for case in cases:
        score.total += 1
        predictions = [normalize_label(p) for p in (predict_fn(case.utterance) or [])]
        escalated = len(predictions) == 1 and predictions[0] in accepted
        if escalated:
            score.correct += 1
            continue
        score.failures.append(
            {
                "context": getattr(case, "context", None),
                "utterance": case.utterance,
                "predicted": predictions,
                "expected_ancestor_command": getattr(
                    case, "expected_ancestor_command", None
                ),
                "reason": (
                    "escalation label present but not alone (ambiguity branch would win)"
                    if any(p in accepted for p in predictions)
                    else "no escalation label predicted"
                ),
            }
        )

    if score.total:
        score.recall = score.correct / score.total
    return score


# ---------------------------------------------------------------------------
# Benchmark file (R1b)
# ---------------------------------------------------------------------------


def default_benchmark_path(workflow_folderpath: str) -> str:
    """Return the path the orchestrator should look for a benchmark file at."""
    return str(Path(workflow_folderpath) / DEFAULT_BENCHMARK_FILENAME)


def load_benchmark_file(path: str) -> list[BenchmarkCase]:
    """Load a developer-supplied benchmark file.

    Accepts either the documented object form
    ``{"schema_version": 1, "cases": [...]}`` or a bare list of case objects.
    See ``docs/intent_benchmark_format.md``.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        version = raw.get("schema_version", BENCHMARK_SCHEMA_VERSION)
        if version != BENCHMARK_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported benchmark schema_version {version!r} in {path}; "
                f"this build understands {BENCHMARK_SCHEMA_VERSION}."
            )
        raw_cases = raw.get("cases")
        if raw_cases is None:
            raise ValueError(f"Benchmark file {path} has no 'cases' key.")
    elif isinstance(raw, list):
        raw_cases = raw
    else:
        raise ValueError(
            f"Benchmark file {path} must contain an object with a 'cases' key or a list "
            f"of cases; found {type(raw).__name__}."
        )

    # Reject the removed out-of-scope axis before Pydantic validation so developers get
    # an actionable design reference instead of a generic unsupported-Literal error.
    # Keep this explicit guard until fix-d28 settles the runtime contract and evidence bar.
    unsupported = [
        index
        for index, case in enumerate(raw_cases)
        if isinstance(case, dict) and case.get("kind") == "out_of_scope"
    ]
    if unsupported:
        indices = ", ".join(str(index) for index in unsupported)
        raise ValueError(
            f"Benchmark file {path} uses unsupported kind='out_of_scope' in case(s) "
            f"{indices}. Out-of-scope scoring is not supported; see fix-d28 and its "
            f"design work before adding these cases."
        )

    cases = [BenchmarkCase(**case) for case in raw_cases]

    problems: list[str] = []
    for index, case in enumerate(cases):
        if case.kind == "routing" and not case.expected_label:
            problems.append(f"case {index}: routing case has no 'expected_label'")
        if case.kind == "escalation" and not case.expected_ancestor_command:
            problems.append(
                f"case {index}: escalation case has no 'expected_ancestor_command'"
            )
    if problems:
        raise ValueError(
            f"Invalid benchmark file {path}:\n  " + "\n  ".join(problems)
        )

    return cases


def assert_benchmark_disjoint_from_seeds(
    cases: Sequence[BenchmarkCase],
    seed_utterances_by_command: Mapping[str, Sequence[str]],
) -> None:
    """Raise `BenchmarkLeakError` listing every benchmark utterance that is also a seed.

    Comparison is on the normalised form (`normalize_utterance`), so
    ``"Close the account."`` collides with ``"close the account"``. An exact string match
    would miss the realistic version of this mistake: a developer pastes a failing
    benchmark case into a seed list to "fix" it, adjusting the capitalisation or dropping
    the full stop on the way.
    """
    seed_index: dict[str, list[str]] = defaultdict(list)
    for command, utterances in seed_utterances_by_command.items():
        for utterance in utterances:
            seed_index[normalize_utterance(utterance)].append(command)

    overlaps: list[str] = []
    for case in cases:
        normalized = normalize_utterance(case.utterance)
        commands = seed_index.get(normalized)
        if commands:
            overlaps.append(
                f"  {case.context}: {case.utterance!r} is a seed utterance for "
                f"{', '.join(sorted(set(commands)))}"
            )

    if overlaps:
        raise BenchmarkLeakError(
            f"{len(overlaps)} benchmark utterance(s) also appear in the seed table. A "
            f"benchmark that shares phrasings with the training seeds measures "
            f"memorisation, not generalisation. Rephrase the benchmark case or remove the "
            f"seed:\n" + "\n".join(overlaps)
        )


def find_near_duplicate_benchmark_cases(
    cases: Sequence[BenchmarkCase],
    seed_utterances_by_command: Mapping[str, Sequence[str]],
    threshold: float = DEFAULT_NEAR_DUPLICATE_THRESHOLD,
) -> list[str]:
    """Return warnings for benchmark utterances that closely resemble a seed utterance.

    Exact (normalised) overlap fails the run; a close-but-not-equal match is a judgement
    call for the developer, so it is reported and not enforced.
    """
    seed_pairs = [
        (normalize_utterance(utterance), command)
        for command, utterances in seed_utterances_by_command.items()
        for utterance in utterances
    ]

    warnings: list[str] = []
    matcher = difflib.SequenceMatcher(autojunk=False)
    for case in cases:
        normalized = normalize_utterance(case.utterance)
        matcher.set_seq2(normalized)
        for seed_text, command in seed_pairs:
            if seed_text == normalized:
                continue  # exact overlap is the caller's hard failure, not a warning
            matcher.set_seq1(seed_text)
            if matcher.real_quick_ratio() < threshold or matcher.quick_ratio() < threshold:
                continue
            ratio = matcher.ratio()
            if ratio >= threshold:
                warnings.append(
                    f"{case.context}: {case.utterance!r} is {ratio:.0%} similar to seed "
                    f"utterance {seed_text!r} of {command}"
                )
    return warnings


def validate_escalation_cases(
    cases: Sequence[BenchmarkCase],
    context_label_space: Mapping[str, set[str]],
    ancestor_map: Mapping[str, Sequence[str]],
) -> list[str]:
    """Return human-readable problems with the escalation cases.

    Escalation is defined structurally, not by vibes: the expected command must be
    provably ABSENT from the tested context's label space and PRESENT in one of that
    context's ancestors. A case that fails either half is not an escalation case — it is
    a routing case, or a typo — and scoring it as escalation would report a number about
    a behaviour that was never tested.
    """
    problems: list[str] = []

    for index, case in enumerate(cases):
        if case.kind != "escalation":
            continue

        prefix = f"case {index} ({case.context}: {case.utterance!r})"

        if case.context not in context_label_space:
            problems.append(
                f"{prefix}: tested context {case.context!r} has no known label space"
            )
            continue

        expected = case.expected_ancestor_command
        if not expected:
            problems.append(f"{prefix}: no 'expected_ancestor_command'")
            continue
        expected = normalize_label(expected)

        local_labels = {
            normalize_label(label) for label in context_label_space[case.context]
        }
        if expected in local_labels:
            problems.append(
                f"{prefix}: expected command {expected!r} IS in the tested context's label "
                f"space, so this is a routing case, not an escalation case"
            )
            continue

        ancestors = list(ancestor_map.get(case.context, []))
        if not ancestors:
            problems.append(
                f"{prefix}: context {case.context!r} has no ancestors, so nothing can be "
                f"escalated to"
            )
            continue

        found_in = [
            ancestor
            for ancestor in ancestors
            if expected
            in {
                normalize_label(label)
                for label in context_label_space.get(ancestor, set())
            }
        ]
        if not found_in:
            problems.append(
                f"{prefix}: expected command {expected!r} is not present in any ancestor of "
                f"{case.context!r} ({', '.join(ancestors)}), so there is nothing to escalate to"
            )

    return problems


def validate_routing_cases(
    cases: Sequence[BenchmarkCase],
    context_label_space: Mapping[str, set[str]],
) -> list[str]:
    """Return human-readable problems with the routing cases.

    A routing case whose expected label is not in the tested context's label space can
    never pass; it is a benchmark defect, not a model failure, and it drags the reported
    score down for the wrong reason.
    """
    problems: list[str] = []
    for index, case in enumerate(cases):
        if case.kind != "routing":
            continue
        prefix = f"case {index} ({case.context}: {case.utterance!r})"
        if case.context not in context_label_space:
            problems.append(
                f"{prefix}: tested context {case.context!r} has no known label space"
            )
            continue
        if not case.expected_label:
            problems.append(f"{prefix}: no 'expected_label'")
            continue
        local_labels = {
            normalize_label(label) for label in context_label_space[case.context]
        }
        if normalize_label(case.expected_label) not in local_labels:
            problems.append(
                f"{prefix}: expected label {case.expected_label!r} is not in the label "
                f"space of context {case.context!r}, so this case can never pass"
            )
    return problems


def benchmark_cases_for_context(
    cases: Sequence[BenchmarkCase],
    context: str,
    kind: Literal["routing", "escalation"],
) -> list[BenchmarkCase]:
    """Return the *kind* cases targeting *context*."""
    return [case for case in cases if case.context == context and case.kind == kind]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def aggregate_totals(reports: Sequence[HeldoutReport]) -> dict:
    """Aggregate routing and escalation counts across contexts.

    Counts are summed, not averaged over contexts: a context with 4 held-out cases should
    not weigh the same as one with 200.
    """
    routing_total = sum(r.routing.total for r in reports if r.routing)
    top1 = sum(r.routing.top1_correct for r in reports if r.routing)
    in_list = sum(r.routing.in_list_correct for r in reports if r.routing)
    escalation_total = sum(r.escalation.total for r in reports if r.escalation)
    escalation_correct = sum(r.escalation.correct for r in reports if r.escalation)
    holdout_escalation_total = sum(
        r.holdout_escalation.total for r in reports if r.holdout_escalation
    )
    holdout_escalation_correct = sum(
        r.holdout_escalation.correct for r in reports if r.holdout_escalation
    )
    benchmark_total = sum(
        r.benchmark_routing.total for r in reports if r.benchmark_routing
    )
    benchmark_top1 = sum(
        r.benchmark_routing.top1_correct for r in reports if r.benchmark_routing
    )
    benchmark_in_list = sum(
        r.benchmark_routing.in_list_correct for r in reports if r.benchmark_routing
    )
    f1_values = [
        r.in_distribution_f1 for r in reports if r.in_distribution_f1 is not None
    ]

    return {
        "contexts": len(reports),
        "routing_total": routing_total,
        "routing_top1_correct": top1,
        "routing_in_list_correct": in_list,
        # None, not 0.0, when the denominator is zero. In JSON "0.0" is
        # indistinguishable from a measured total failure, and a reader diffing
        # two runs cannot tell "we did not measure this" from "this scored zero".
        "routing_top1": (top1 / routing_total) if routing_total else None,
        "routing_in_list": (in_list / routing_total) if routing_total else None,
        # The fixed benchmark is the only cross-run-comparable population: the
        # persona holdout is re-drawn every run, so two runs score different
        # cases. Excluding it from the totals left the one comparable number
        # visible nowhere but a transient per-context print.
        "benchmark_routing_total": benchmark_total,
        "benchmark_routing_top1_correct": benchmark_top1,
        "benchmark_routing_in_list_correct": benchmark_in_list,
        "benchmark_routing_top1": (
            benchmark_top1 / benchmark_total if benchmark_total else None
        ),
        "benchmark_routing_in_list": (
            benchmark_in_list / benchmark_total if benchmark_total else None
        ),
        "escalation_total": escalation_total,
        "escalation_correct": escalation_correct,
        "escalation_recall": (
            escalation_correct / escalation_total if escalation_total else None
        ),
        "holdout_escalation_total": holdout_escalation_total,
        "holdout_escalation_correct": holdout_escalation_correct,
        "holdout_escalation_recall": (
            holdout_escalation_correct / holdout_escalation_total
            if holdout_escalation_total
            else None
        ),
        "mean_in_distribution_f1": (
            sum(f1_values) / len(f1_values) if f1_values else None
        ),
    }


def _fmt_pct(numerator: int, denominator: int) -> str:
    return f"{numerator / denominator:.1%}" if denominator else "-"


def format_report(reports: Sequence[HeldoutReport]) -> str:
    """Render a human-readable table for the end of a training run."""
    if not reports:
        return "Held-out evaluation: no contexts evaluated."

    header = (
        f"{'Context':<28} {'in-dist F1':>10} {'heldout N':>9} {'top-1':>8} "
        f"{'in-list':>8} {'bench N':>8} {'bench top-1':>12} "
        f"{'escal N':>8} {'escal recall':>13}"
    )
    rule = "-" * len(header)

    lines = [
        "Held-out evaluation (whole-persona holdout; see docs/intent_benchmark_format.md)",
        rule,
        header,
        rule,
    ]

    for report in sorted(reports, key=lambda r: r.context):
        routing = report.routing
        escalation = report.escalation
        bench = report.benchmark_routing
        f1 = (
            f"{report.in_distribution_f1:.3f}"
            if report.in_distribution_f1 is not None
            else "-"
        )
        lines.append(
            f"{report.context:<28} {f1:>10} "
            f"{(routing.total if routing else 0):>9} "
            f"{(_fmt_pct(routing.top1_correct, routing.total) if routing else '-'):>8} "
            f"{(_fmt_pct(routing.in_list_correct, routing.total) if routing else '-'):>8} "
            f"{(bench.total if bench else 0):>8} "
            f"{(_fmt_pct(bench.top1_correct, bench.total) if bench else '-'):>12} "
            f"{(escalation.total if escalation else 0):>8} "
            f"{(_fmt_pct(escalation.correct, escalation.total) if escalation else '-'):>13}"
        )

    totals = aggregate_totals(reports)
    mean_f1 = totals["mean_in_distribution_f1"]
    lines.extend(
        [
            rule,
            f"{'TOTAL':<28} "
            f"{(f'{mean_f1:.3f}' if mean_f1 is not None else '-'):>10} "
            f"{totals['routing_total']:>9} "
            f"{_fmt_pct(totals['routing_top1_correct'], totals['routing_total']):>8} "
            f"{_fmt_pct(totals['routing_in_list_correct'], totals['routing_total']):>8} "
            f"{totals['benchmark_routing_total']:>8} "
            f"{_fmt_pct(totals['benchmark_routing_top1_correct'], totals['benchmark_routing_total']):>12} "
            f"{totals['escalation_total']:>8} "
            f"{_fmt_pct(totals['escalation_correct'], totals['escalation_total']):>13}",
            rule,
            "'in-dist F1' is the legacy same-distribution split score. It is reported for",
            "continuity only: it is computed on utterances drawn from the same personas and",
            "seed list the model trained on, so it measures memorisation. Judge the models on",
            "top-1 / in-list / escalation recall.",
            "Routing accuracy and escalation recall trade against each other and are never",
            "blended into one number.",
            "'heldout N' is re-drawn every run, so its top-1 is NOT comparable across runs.",
            "'bench N' is the fixed developer-supplied benchmark: that is the number to",
            "compare between run N and run N-1.",
            "The evaluated model IS the shipped model: held-out personas are removed from",
            "training, so these are LOWER BOUNDS. The holdout is in-generator (held-out",
            "personas came from the same generator), so it measures generalisation across",
            "personas, not across real users.",
        ]
    )

    holdout_escalation_total = totals["holdout_escalation_total"]
    if holdout_escalation_total:
        lines.append(
            f"Persona-holdout escalation-class rows: "
            f"{totals['holdout_escalation_correct']}/{holdout_escalation_total} "
            f"({_fmt_pct(totals['holdout_escalation_correct'], holdout_escalation_total)}) "
            f"scored separately -- these are not routes and are excluded from top-1."
        )

    notes = [
        (report.context, note)
        for report in sorted(reports, key=lambda r: r.context)
        for note in report.notes
    ]
    if notes:
        lines.append("")
        lines.append("Notes:")
        lines.extend(f"  [{context}] {note}" for context, note in notes)

    return "\n".join(lines)


def write_report(workflow_folderpath: str, reports: Sequence[HeldoutReport]) -> str:
    """Write the JSON report to ``<workflow>/___command_info/heldout_evaluation.json``.

    Returns the path written.
    """
    output_dir = Path(workflow_folderpath) / COMMAND_INFO_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / REPORT_FILENAME

    payload = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metric_notes": {
            "in_distribution_f1": (
                "Legacy same-distribution split score; measures memorisation. Reported for "
                "continuity, not for judging generalisation."
            ),
            "routing": (
                "Held-out accuracy: top-1 requires one lone, confident correct label; "
                "in-list credits the expected label anywhere in an ambiguity list."
            ),
            "escalation": (
                "Recall of a lone, confident escalation label; reported separately from "
                "routing accuracy because the two trade against each other."
            ),
            "holdout_escalation": (
                "Persona-held-out rows whose expected label is an escalation class. "
                "Scored on lone-escalation semantics and kept off the routing axis: a "
                "lone wildcard escalates to an ancestor, it is not a route."
            ),
            "benchmark_routing": (
                "Routing on the fixed developer-supplied benchmark. This is the only "
                "number comparable BETWEEN runs; the persona holdout is re-drawn every "
                "run, so its top-1 scores different cases each time."
            ),
            "which_model_was_evaluated": (
                "The evaluated model IS the shipped model: held-out personas are removed "
                "from training entirely, so the published model is trained without them. "
                "Treat every held-out number as a LOWER BOUND on the shipped model's "
                "quality. This is also an in-generator holdout -- held-out personas came "
                "from the same generator as the training rows -- so it measures "
                "generalisation across personas, not across real users."
            ),
            "zero_denominator": (
                "A metric with no cases is null, never 0.0. A 0.0 here would be "
                "indistinguishable from a measured total failure."
            ),
        },
        "totals": aggregate_totals(reports),
        "contexts": [report.model_dump() for report in reports],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
        f.write("\n")

    return str(output_path)

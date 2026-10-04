"""One scoped-execution reference and one comparison projection (`fix-9eg.4`).

Three things the product wants to compare turn out to be the same thing:

- the current experiment winner against a candidate,
- the selected best run of a task against its other attempts,
- a recorded teacher pass against the student pass that followed it.

They differ only in HOW the two sides are named, so this module defines ONE
reference — `ExecutionRef` — and one projection over it, rather than three
views that would drift. A comparison is a pure read: nothing here writes to an
evidence store, and nothing here is a judgement. "Reference", "best" and
"teacher" are labels a caller chose; a difference from the left side is a
difference, not an error.

Reference vocabulary is borrowed, not invented. `store_id` + logical turn key
is exactly what `workspace.py` already uses for a portable evidence address,
and experiment/task/attempt are the `experiments` / `experiment_attempts`
columns. A reference that DECLARES an experiment, task or attempt is checked
against the turn rows it names: a scope the evidence does not record is
refused, because a forged or stale scope would otherwise ride along on real
evidence and label it — and anchor feedback to it — as something it is not.

The only genuinely new part is `pass_id`, for a turn that holds more than one
recorded pass. Membership is resolved ONLY from recorded spans, through a
`PassSelector` the caller supplies and this module verifies: a rule naming a
span or a stamp the turn does not contain is an unknown pass and is refused
rather than answered with an empty view. Pass identity and pass scope travel
together — a `pass_id` with no selector, or a selector with no `pass_id`, is
refused, because either alone yields a projection whose contents and whose
`ref_id` disagree about what was shown.

WHAT A PASS-SCOPED PROJECTION SHOWS. Membership always comes from recorded
spans, and so does content -- when a producer recorded any. `distillation.py`
opens one `fw.distillation.pass` span per pass (`fix-txxy`): it stamps
`fw.pass`, which everything the pass did inherits through ancestry, and it
carries that pass's own answer, plan and outcome, which the shared turn row has
nowhere to put. A projection scoped to such a pass reports those as the PASS's
and keeps the turn row's answer and status beside them.

A trace with no pass span -- every trace recorded before that producer change,
and every ordinary single-pass turn -- is unchanged: `discover_pass_selectors`
returns `[]` and the caller compares whole turns. Where a caller scopes such a
trace by an explicit span list or subtree, the projection carries only the
STEPS and the LLM cost the spans attribute to the pass, and the turn's answer,
status and wall time stay labelled `shared_across_passes` rather than claimed
by a pass. Nothing is back-filled and nothing is inferred:
`ObservabilityStore.list_distillation_runs` is still a stub that returns `[]`,
and a pass that recorded no answer is reported as having recorded none.

Reads go through an injected reader: no filesystem lookup, no cross-store
search, and a reference naming an unknown store fails rather than being
resolved somewhere else. Absence is preserved rather than repaired -- an
unreadable turn is reported in `unavailable` and the rest still renders, so an
execution with no scores, a half-pruned trace, or one missing side can still be
inspected.

The per-turn step list is the execution ledger from `observability/turn_derivations.py`
(re-exported by `run_chatbot/turn_annotations.py`), injected rather than re-derived, because a second implementation of "what
dispatches happened in this turn" is exactly the semantic mismatch this work
must not introduce. Nested wrappers therefore appear exactly once, as the
ledger files them, and every roll-up here that could double-count a parent and
its span-less inner hop counts ROOT rows only.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Mapping, Optional, Protocol, Sequence

from fastworkflow.observability.turn_derivations import execution_ledger

# The span name the ledger builds its rows from, restated here (as
# `observability/turn_derivations.py` restates it) so that reading a stored trace does not
# import the HTTP layer.
SPAN_COMMAND_EXECUTE = "fw.command.execute"

# The span a pass-recording producer opens for a pass, restated for the same
# reason. Pass MEMBERSHIP is resolved from any span in a step's ancestry, so it
# needs no span name; pass CONTENT is recorded once, on this span, because a
# pass's own answer is a fact about the pass rather than about anything it did.
SPAN_DISTILLATION_PASS = "fw.distillation.pass"

# What that span carries. A restatement can rot, so
# tests/test_distillation_pass_capture.py asserts these spell the producer's
# contract (`tracing.SPAN_CONTRACTS`) rather than something adjacent to it.
PASS_CONTENT_KEYS = ("answer", "plan", "status", "failure_reason", "model")

# How a matched pair was decided. `recorded` means a stored structured
# alignment said so; the rest are this module's deterministic fallback, named
# after the recorded fields they agreed on. Cross-run `command_call_id`
# equality is deliberately NOT a basis: call ids are minted per run, so equal
# ids across two runs are either meaningless or a copied trace.
BASIS_RECORDED = "recorded"
BASIS_COMMAND_CONTEXT_PARAMETERS = "command+context+parameters"
BASIS_COMMAND_CONTEXT = "command+context"
BASIS_COMMAND = "command"
# Unmatched steps carry a basis too, so a reader can tell "we looked and found
# no counterpart" from "we could not form a key for this step at all".
BASIS_UNMATCHED = "unmatched"
BASIS_UNKNOWN = "unknown"

PAIR_MATCHED = "matched"
PAIR_LEFT_ONLY = "left_only"
PAIR_RIGHT_ONLY = "right_only"

# Whose content a projected value describes. A whole-turn projection owns its
# turn's answer and wall time; a pass-scoped one does not -- the passes share
# one turn row, so its text is `shared_across_passes` and saying otherwise
# would attribute the student's answer to the teacher.
ATTRIBUTION_TURN = "turn"
ATTRIBUTION_SHARED = "shared_across_passes"
# An artifact whose `CommandOutput` recorded no `command_call_id` cannot be
# joined to a dispatch, so a pass-scoped projection can say only that the turn
# produced it. It is listed apart rather than claimed by both passes.
ATTRIBUTION_PASS = "pass"
ATTRIBUTION_UNATTRIBUTED = "unattributed"

# Beyond this many step pairs the quadratic alignment is skipped in favour of
# an order-preserving greedy pass, and the result says so. A multi-turn attempt
# with thousands of dispatches is rare; silently spending minutes on one in a
# request handler is worse than an explicitly degraded answer.
_MAX_ALIGNMENT_CELLS = 250_000


class ComparisonError(RuntimeError):
    """Base class for comparison failures."""


class InvalidExecutionRef(ComparisonError, ValueError):
    """A reference does not name a readable scope."""


class ExecutionScopeMismatch(InvalidExecutionRef):
    """A reference declares an experiment/task/attempt the evidence does not record.

    Raised rather than dropped to a label, because the failure it prevents is
    silent: a reference that claims `attempt=2` over a turn recorded under
    attempt 1 would render real evidence under a false heading, and every
    feedback anchor and review-pair key derived from it would carry that claim.
    """


class UnknownRecordedPass(InvalidExecutionRef):
    """A pass selector resolves against no recorded evidence in a turn.

    A selector is external input. When its rules name a span id or a stamp the
    turn does not contain, the honest answer is "this turn records no such
    pass" -- not an empty pass view, which reads as "the pass did nothing".
    """


class InvalidRecordedAlignment(ComparisonError, ValueError):
    """A supplied alignment names steps that are not in the recorded evidence."""


def _clean(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _text_or_none(value: Any) -> Optional[str]:
    return value if isinstance(value, str) and value else None


def _exact_int(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _required_exact_int(value: Any, field: str) -> int:
    """An attempt number, or a refusal. Never a truncation and never a bool.

    `int(value)` was wrong here: it turns 1.9 into attempt 1 and `True` into
    attempt 1, so a malformed reference would silently name a DIFFERENT
    attempt's evidence. Exact integers only; the string form is accepted at the
    wire boundary alone (`from_mapping`), where query parameters arrive as text.
    """
    exact = _exact_int(value)
    if exact is None:
        raise InvalidExecutionRef(
            f"{field} must be an exact integer, not {value!r}"
        )
    return exact


def _canonical_json(value: Any) -> str:
    """Stable text for a recorded value, for digesting only.

    `default=repr` rather than raising: a record that round-tripped through
    JSON is already plain data, but a hand-built one may hold anything, and a
    step whose parameters cannot be serialized still deserves a key.
    """
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, default=repr)
    except (TypeError, ValueError):  # pragma: no cover - default=repr covers it
        return repr(value)


def _digest(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]


# ----------------------------------------------------------------------
# The reference
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ExecutionRef:
    """One scoped recorded execution: the unit both sides of a comparison name.

    `store_id` + `turn_keys` is the mandatory part and is the same address
    `workspace.py` uses (`logical_turn_key` within a named store). `turn_keys`
    is a SEQUENCE, so a multi-turn attempt is one reference and stays navigable
    in full rather than collapsing to its final answer. Everything else narrows
    it:

    - `experiment_id` / `task_id` / `attempt` when the turns belong to an
      experiment attempt. Optional, because an ad-hoc chat turn and a
      teacher/student pass recorded outside any experiment are both legitimate
      things to compare; declared, they are CHECKED against the turn rows by
      `project_execution` rather than taken as labels.
    - `pass_id` when one turn holds more than one recorded pass. Two passes
      sharing a trace id are two references differing only in this field.

    `label` is display-only and is excluded from `ref_id`, so renaming
    "Reference" to "Best run" does not invalidate review progress or feedback
    recorded against the pair.
    """

    store_id: str
    turn_keys: tuple[str, ...]
    experiment_id: Optional[str] = None
    task_id: Optional[str] = None
    attempt: Optional[int] = None
    pass_id: Optional[str] = None
    label: Optional[str] = None

    def __post_init__(self) -> None:
        store_id = _clean(self.store_id)
        if not store_id:
            raise InvalidExecutionRef(
                "store_id is required; executions are never searched across stores"
            )
        keys = tuple(key for key in (_clean(k) for k in self.turn_keys) if key)
        if not keys:
            raise InvalidExecutionRef("at least one turn key is required")
        if len(set(keys)) != len(keys):
            raise InvalidExecutionRef("turn keys must be distinct")
        attempt = self.attempt
        if attempt is not None:
            attempt = _required_exact_int(attempt, "attempt")
            if attempt < 0:
                raise InvalidExecutionRef("attempt must not be negative")
        object.__setattr__(self, "store_id", store_id)
        object.__setattr__(self, "turn_keys", keys)
        object.__setattr__(self, "experiment_id", _clean(self.experiment_id))
        object.__setattr__(self, "task_id", _clean(self.task_id))
        object.__setattr__(self, "attempt", attempt)
        object.__setattr__(self, "pass_id", _clean(self.pass_id))
        object.__setattr__(self, "label", _clean(self.label))

    def ref_id(self) -> str:
        """A stable id for this scope, for anchors and review-pair keys.

        Derived, never minted, so two processes that build the same reference
        agree without coordinating, and a reference reconstructed from a URL
        keys the same review row it did yesterday. `label` is excluded on
        purpose (see the class docstring).
        """
        return "xr-" + _digest(
            self.store_id,
            "\x1e".join(self.turn_keys),
            self.experiment_id or "",
            self.task_id or "",
            "" if self.attempt is None else str(self.attempt),
            self.pass_id or "",
        )

    def as_dict(self) -> dict[str, Any]:
        """The wire shape. Human UI and agent reads use this same object."""
        return {
            "ref_id": self.ref_id(),
            "store_id": self.store_id,
            "turn_keys": list(self.turn_keys),
            "experiment_id": self.experiment_id,
            "task_id": self.task_id,
            "attempt": self.attempt,
            "pass_id": self.pass_id,
            "label": self.label,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ExecutionRef":
        """Parse the wire shape, tolerating the workspace's field names.

        `logical_turn_keys` / `logical_turn_key` are accepted because that is
        what a workspace attempt row's `turn_refs` are called; they name the
        same strings.

        `attempt` may arrive as the text an HTTP query carries, and only as the
        exact decimal form of an integer: `"2"` is attempt 2, while `"2.0"`,
        `2.5` and `True` are refused here rather than truncated into a
        reference that names somebody else's attempt.
        """
        if not isinstance(value, Mapping):
            raise InvalidExecutionRef("an execution reference must be an object")
        raw_keys: Any = None
        for key in ("turn_keys", "logical_turn_keys", "turns"):
            if value.get(key) is not None:
                raw_keys = value[key]
                break
        if raw_keys is None:
            single = value.get("turn_key", value.get("logical_turn_key"))
            raw_keys = [single] if single is not None else []
        if isinstance(raw_keys, str):
            raw_keys = [raw_keys]
        if not isinstance(raw_keys, (list, tuple)):
            raise InvalidExecutionRef("turn_keys must be an array of turn keys")
        attempt = value.get("attempt")
        if isinstance(attempt, str):
            text = attempt.strip()
            if not (text.lstrip("-").isdigit()):
                raise InvalidExecutionRef(
                    f"attempt must be an exact integer, not {attempt!r}"
                )
            attempt = int(text)
        return cls(
            store_id=value.get("store_id"),
            turn_keys=tuple(str(k) for k in raw_keys if k is not None),
            experiment_id=value.get("experiment_id"),
            task_id=value.get("task_id"),
            attempt=attempt,
            pass_id=value.get("pass_id"),
            label=value.get("label"),
        )


def review_pair_key(left: ExecutionRef, right: ExecutionRef) -> str:
    """The identity of one review pair: the EXACT two executions compared.

    Ordered, not a set: "this candidate reviewed against that reference" is
    not the same statement as the reverse, and pinning a different reference
    must produce new pairs rather than relabel old ones.
    """
    return f"{left.ref_id()}|{right.ref_id()}"


# ----------------------------------------------------------------------
# Recorded passes within one turn
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class PassSelector:
    """How to tell one recorded pass's activity from another's, within a turn.

    No rule infers a pass from ordering, timing or model name: the failure this
    module must avoid is a confident wrong attribution, and showing the
    student's calls under the teacher's heading is worse than showing them as
    unattributed.

    Every rule names RECORDED SPANS and is checked against the turn's spans
    before use (`resolve_against`), so a selector is evidence a caller can point
    at rather than an assertion this module takes on trust. Rules are ORed:

    - `attribute_key`/`attribute_value`: a span attribute recorded on the
      step's own `fw.command.execute` span or on any of its ancestors. This is
      the rule a producer that stamps its passes should use, and the one
      `discover_pass_selectors` builds.
    - `root_span_ids`: the step's span lies in the subtree of one of these. A
      producer that opens one span per pass needs nothing else.
    - `span_ids`: an explicit membership list of recorded span ids.

    `exclude_span_ids` removes activity that belongs to neither pass -- insight
    extraction runs after both and its LLM calls would otherwise land in
    whichever pass a subtree rule happened to cover. Exclusion is checked
    against a span's whole ancestry, so excluding a root excludes its subtree,
    and exclusion wins over every inclusion rule.

    There is deliberately no rule keyed on `command_call_id` and no separate
    subtree-exclusion set: the first would let a caller assert pass membership
    for dispatches the span tree does not place in the pass (an in-process
    action log is not recorded evidence), and the second is what
    `exclude_span_ids` already does.
    """

    pass_id: str
    attribute_key: Optional[str] = None
    attribute_value: Optional[str] = None
    root_span_ids: frozenset[str] = frozenset()
    span_ids: frozenset[str] = frozenset()
    exclude_span_ids: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        pass_id = _clean(self.pass_id)
        if not pass_id:
            raise ValueError("pass_id is required")
        object.__setattr__(self, "pass_id", pass_id)
        for name in ("root_span_ids", "span_ids", "exclude_span_ids"):
            object.__setattr__(self, name, frozenset(getattr(self, name)))
        if (self.attribute_key is None) != (self.attribute_value is None):
            raise ValueError(
                "attribute_key and attribute_value are set together or not at all"
            )
        if not (self.attribute_key or self.root_span_ids or self.span_ids):
            raise ValueError(
                f"pass selector {pass_id!r} has no membership rule; a selector "
                "that matches everything cannot separate two passes"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "pass_id": self.pass_id,
            "attribute_key": self.attribute_key,
            "attribute_value": self.attribute_value,
            "root_span_ids": sorted(self.root_span_ids),
            "span_ids": sorted(self.span_ids),
            "exclude_span_ids": sorted(self.exclude_span_ids),
        }

    def resolve_against(self, tree: "_SpanTree", turn_key: str) -> None:
        """Refuse unless every rule names evidence this turn actually recorded.

        Checked per turn, before any step is attributed. An id or a stamp the
        turn does not contain means the caller is describing a pass that is not
        in this evidence, and the answer it would otherwise get -- a projection
        with no steps -- is indistinguishable from a pass that ran and did
        nothing.
        """
        missing = sorted(
            (self.root_span_ids | self.span_ids | self.exclude_span_ids)
            - set(tree.by_id)
        )
        if missing:
            raise UnknownRecordedPass(
                f"pass {self.pass_id!r} names span(s) {missing} that turn "
                f"{turn_key!r} does not record"
            )
        if self.attribute_key is not None and not any(
            tree.attributes(span).get(self.attribute_key) == self.attribute_value
            for span in tree.by_id.values()
        ):
            raise UnknownRecordedPass(
                f"pass {self.pass_id!r} is selected by "
                f"{self.attribute_key}={self.attribute_value!r}, which no span of "
                f"turn {turn_key!r} records"
            )


class _SpanTree:
    """Parent links and decoded attributes for one turn's spans."""

    def __init__(self, spans: Sequence[Mapping[str, Any]]) -> None:
        self.by_id: dict[str, Mapping[str, Any]] = {}
        for span in spans:
            span_id = _text_or_none(span.get("span_id"))
            if span_id:
                self.by_id[span_id] = span

    def attributes(self, span: Mapping[str, Any]) -> dict[str, Any]:
        """Span attributes as a mapping, whether the reader decoded them.

        `workspace.trace` decodes the JSON column; `ObservabilityStore.get_spans`
        hands back the raw text. Both are legitimate readers, so accept both
        rather than making the caller normalize.
        """
        raw = span.get("attributes")
        if isinstance(raw, Mapping):
            return dict(raw)
        if isinstance(raw, str):
            try:
                decoded = json.loads(raw)
            except (ValueError, TypeError):
                return {}
            return decoded if isinstance(decoded, dict) else {}
        return {}

    def ancestry(self, span_id: Optional[str]) -> list[Mapping[str, Any]]:
        """The span and its ancestors, nearest first. Cycle-safe."""
        chain: list[Mapping[str, Any]] = []
        seen: set[str] = set()
        cursor = span_id
        while cursor and cursor in self.by_id and cursor not in seen:
            seen.add(cursor)
            span = self.by_id[cursor]
            chain.append(span)
            cursor = _text_or_none(span.get("parent_span_id"))
        return chain


def discover_pass_selectors(
    spans: Iterable[Mapping[str, Any]],
    *,
    attribute_key: str,
    exclude_values: Iterable[str] = (),
) -> list[PassSelector]:
    """Selectors for every distinct recorded value of one span attribute.

    Discovery, not inference: it reports the pass labels a producer actually
    stamped, in sorted order so two callers agree, and returns `[]` when
    nothing was stamped. A caller that gets `[]` has a turn with no recorded
    pass identity and must compare it whole.
    """
    tree = _SpanTree(list(spans))
    excluded = {str(value) for value in exclude_values}
    found: set[str] = set()
    for span in tree.by_id.values():
        value = tree.attributes(span).get(attribute_key)
        if isinstance(value, str) and value and value not in excluded:
            found.add(value)
    return [
        PassSelector(
            pass_id=value, attribute_key=attribute_key, attribute_value=value
        )
        for value in sorted(found)
    ]


# ----------------------------------------------------------------------
# Readers
# ----------------------------------------------------------------------


class ExecutionReader(Protocol):
    """What a projection needs, and nothing more.

    Both methods take an explicit `store_id`. A reader is free to refuse one
    it was not constructed for; none of them may go looking for the turn
    elsewhere.
    """

    def turn(self, store_id: str, turn_key: str) -> Optional[Mapping[str, Any]]:
        """The turn row with its decoded `record`, or None if absent."""

    def trace(self, store_id: str, turn_key: str) -> list[Mapping[str, Any]]:
        """The turn's span rows, or `[]`."""


class StoreExecutionReader:
    """Reader over ONE opened `ObservabilityStore`, bound to one store id.

    The store is opened and named by the caller -- typically a
    `ReadOnlyObservabilityStore`, so inspecting evidence cannot write to it.
    A reference naming a different store raises instead of being answered from
    this one.
    """

    def __init__(self, store_id: str, store: Any) -> None:
        store_id = _clean(store_id)
        if not store_id:
            raise ValueError("store_id is required")
        self.store_id = store_id
        self._store = store

    def _check(self, store_id: str) -> None:
        if store_id != self.store_id:
            raise InvalidExecutionRef(
                f"this reader serves store {self.store_id!r}, not {store_id!r}; "
                "executions are never searched across stores"
            )

    def turn(self, store_id: str, turn_key: str) -> Optional[dict[str, Any]]:
        self._check(store_id)
        row = self._store.get_turn(turn_key)
        if row is None:
            return None
        result = dict(row)
        raw = result.pop("record_json", None)
        try:
            result["record"] = json.loads(raw) if isinstance(raw, str) else None
        except (ValueError, TypeError):
            result["record"] = None
        result["store_id"] = store_id
        result["logical_turn_key"] = turn_key
        return result

    def trace(self, store_id: str, turn_key: str) -> list[dict[str, Any]]:
        self._check(store_id)
        return [dict(span) for span in self._store.get_spans(turn_key)]


class WorkspaceExecutionReader:
    """Reader over a loaded `ObservabilityWorkspace`.

    The workspace already enforces manifest-bound, per-store, read-only access
    and already decodes the turn record and span attributes, so this is a
    two-line adapter rather than a second access path.
    """

    def __init__(self, workspace: Any) -> None:
        self._workspace = workspace

    def turn(self, store_id: str, turn_key: str) -> Optional[dict[str, Any]]:
        return self._workspace.turn(store_id, turn_key)

    def trace(self, store_id: str, turn_key: str) -> list[dict[str, Any]]:
        return list(self._workspace.trace(store_id, turn_key))


# ----------------------------------------------------------------------
# Injected projections that already exist elsewhere
# ----------------------------------------------------------------------

LedgerProjection = Callable[[Any, Iterable[Mapping[str, Any]]], Mapping[str, Any]]
CostRollup = Callable[[Iterable[Mapping[str, Any]]], Mapping[str, Any]]


def default_ledger_projection() -> LedgerProjection:
    """`observability/turn_derivations.py`'s `execution_ledger` -- the one implementation.

    Imported from a stdlib-only module below this one, so this module stays
    usable without the HTTP layer and the dependency runs one way at import
    time. Callers that already hold the function (the server does) should pass
    it instead.
    """
    # No cycle: turn_derivations imports nothing from fastworkflow, and
    # turn_annotations (which imports project_execution from here) re-exports it.
    return execution_ledger


def default_cost_rollup() -> CostRollup:
    """This module's own `cost_rollup`, beside the `usage_rollup` it delegates to.

    Recorded cost only: it answers `total: None`, never 0, when no LLM call
    recorded a cost, and this module passes that through unchanged.
    """
    # No cycle: `cost_rollup` is defined here; turn_annotations re-exports it.
    return cost_rollup


# ----------------------------------------------------------------------
# Projection
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ArtifactRef:
    """One artifact a command produced, as the record refers to it.

    `artifact_id` is set when the value was offloaded to the `artifacts` table
    (the `__fw_artifact_ref__` envelope `serialize_turn_result` writes); it is
    None for a value small enough to have stayed inline, which is a fact about
    size, not about whether the artifact exists. Either way the artifact is
    reachable: by id from the store, or from the record itself.

    `attribution` says WHOSE artifact this is: the turn's (`turn`), the selected
    pass's, because its dispatch is in that pass (`pass`), or nobody's in
    particular (`unattributed`) -- a `CommandOutput` that recorded no
    `command_call_id` cannot be joined to a dispatch, so in a pass-scoped
    projection it is reported apart instead of being shown under both passes as
    if each had produced it.
    """

    turn_key: str
    command_call_id: Optional[str]
    command_name: Optional[str]
    key: str
    artifact_id: Optional[str] = None
    size_bytes: Optional[int] = None
    content_type: Optional[str] = None
    inline: bool = True
    error: Optional[str] = None
    attribution: str = ATTRIBUTION_TURN

    def as_dict(self) -> dict[str, Any]:
        return {
            "turn_key": self.turn_key,
            "command_call_id": self.command_call_id,
            "command_name": self.command_name,
            "key": self.key,
            "artifact_id": self.artifact_id,
            "size_bytes": self.size_bytes,
            "content_type": self.content_type,
            "inline": self.inline,
            "error": self.error,
            "attribution": self.attribution,
        }


@dataclass(frozen=True)
class ExecutionStep:
    """One dispatch, as the ledger filed it plus what the record recorded.

    Everything up to `asked_user` is the ledger row verbatim. `parameters` and
    `response_success` are added here from the two places a dispatch's inputs
    and outcome are recorded -- the turn record's `CommandOutput` (durable,
    joined on `command_call_id`) and the execute span's attributes
    (best-effort) -- with the record preferred. `parameters_digest` is what
    alignment keys on; it is None when neither source recorded parameters, and
    a step with no digest is never matched on a parameter basis.
    """

    turn_index: int
    turn_key: str
    position: int
    command_call_id: str
    parent_call_id: Optional[str]
    command_ordinal: Optional[int]
    span_id: Optional[str]
    command_name: Optional[str]
    context: Optional[str]
    status: Optional[str]
    success: Optional[bool]
    start_ns: Optional[int]
    duration_ns: Optional[int]
    in_record: bool
    span_recorded: bool
    child_call: bool
    asked_user: int
    parameters: Optional[Any] = None
    parameters_source: Optional[str] = None
    parameters_digest: Optional[str] = None
    response_success: Optional[bool] = None
    pass_id: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "turn_key": self.turn_key,
            "position": self.position,
            "command_call_id": self.command_call_id,
            "parent_call_id": self.parent_call_id,
            "command_ordinal": self.command_ordinal,
            "span_id": self.span_id,
            "command_name": self.command_name,
            "context": self.context,
            "status": self.status,
            "success": self.success,
            "start_ns": self.start_ns,
            "duration_ns": self.duration_ns,
            "in_record": self.in_record,
            "span_recorded": self.span_recorded,
            "child_call": self.child_call,
            "asked_user": self.asked_user,
            "parameters": self.parameters,
            "parameters_source": self.parameters_source,
            "parameters_digest": self.parameters_digest,
            "response_success": self.response_success,
            "pass_id": self.pass_id,
        }


@dataclass(frozen=True)
class TurnProjection:
    """One logical turn of the execution: its answer and what it cost.

    `answer`, `status`, `success`, `failure_reason`, `user_message` and the
    timestamps are the TURN ROW's, quoted as recorded. When this projection is
    scoped to a pass they are still the turn row's -- the passes share it --
    UNLESS the producer recorded that pass's own content on its pass span, which
    is the one place a per-pass answer exists at all. So there are two shapes:

    - nothing recorded for the pass: `content_attribution` says
      `shared_across_passes` and `pass_content_recorded` is False. Presenting
      the turn's answer as the pass's own would invent the one thing a
      teacher/student comparison is read for, so it is shown labelled instead.
    - the pass recorded its own: `answer`, `status`, `failure_reason` and `plan`
      are THAT PASS's, `content_attribution` says `pass`, and
      `pass_content_recorded` is True. `turn_answer` and `turn_status` keep the
      shared turn row's values beside them, the way `timing` keeps
      `turn_wall_ms` beside a pass's own wall time.

    `success` is None under a recorded pass. It is a command-success code the
    turn row carries for the whole turn and no producer records a per-pass
    equivalent, so absent is what "not recorded" looks like here; the turn's own
    value is one whole-turn projection away.

    `plan` is the producer's recorded text, quoted rather than re-parsed: a
    projection that parsed it would have to decide what to show when it no
    longer parses, and the honest answer -- what was recorded -- is the same
    either way. An over-limit answer or plan is recorded as a truncation
    envelope ([R10]); the recorded prefix is quoted here and
    `pass_content` carries the envelope that says so, with the original length
    and digest.

    `cost` is genuinely pass-scoped when a selector is in play: it rolls up only
    the LLM spans the span tree attributes to that pass.

    `usage` is the same spans' tokens, cache state and per-call anchors
    (`usage_rollup`), so the money and the tokens on one turn are counted over
    one set of canonical calls rather than two independent tallies.
    """

    turn_index: int
    turn_key: str
    status: Optional[str]
    success: Optional[bool]
    failure_reason: Optional[str]
    answer: Optional[str]
    user_message: Optional[str]
    started_at: Optional[str]
    completed_at: Optional[str]
    suspended_ms: Optional[int]
    experiment_id: Optional[str] = None
    task_id: Optional[str] = None
    attempt: Optional[int] = None
    ledger_summary: Mapping[str, Any] = field(default_factory=dict)
    cost: Mapping[str, Any] = field(default_factory=dict)
    usage: Mapping[str, Any] = field(default_factory=dict)
    pass_id: Optional[str] = None
    content_attribution: str = ATTRIBUTION_TURN
    pass_content_recorded: bool = False
    # The pass's own recorded plan, and the turn row's answer/status kept beside
    # a pass's own. All three are None/absent on a whole-turn projection, where
    # `answer` and `status` already ARE the turn's.
    plan: Optional[str] = None
    turn_answer: Optional[str] = None
    turn_status: Optional[str] = None
    # The pass span's content attributes exactly as recorded, truncation
    # envelopes included, so a reader can see what the quoted fields were
    # derived from.
    pass_content: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "turn_key": self.turn_key,
            "status": self.status,
            "success": self.success,
            "failure_reason": self.failure_reason,
            "answer": self.answer,
            "user_message": self.user_message,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "suspended_ms": self.suspended_ms,
            "experiment_id": self.experiment_id,
            "task_id": self.task_id,
            "attempt": self.attempt,
            "ledger_summary": dict(self.ledger_summary),
            "cost": dict(self.cost),
            "usage": dict(self.usage),
            "pass_id": self.pass_id,
            "content_attribution": self.content_attribution,
            "pass_content_recorded": self.pass_content_recorded,
            "plan": self.plan,
            "turn_answer": self.turn_answer,
            "turn_status": self.turn_status,
            "pass_content": dict(self.pass_content),
        }


# How a projection ordered the turns it read. The reference's turn vector is
# an IDENTITY, not a chronology: `ref_id` -- and so every feedback anchor and
# review-pair key ever derived from it -- is a digest over that vector, and
# reordering it would re-key comments already written. So the vector is left
# alone and the CONTENT is ordered by what the evidence recorded, which is what
# "the last turn's answer" and "the steps in order" are questions about.
TURN_ORDER_CHRONOLOGICAL = "recorded_chronology"
# The evidence did not say. Nothing is guessed: the reference's order is used
# and the projection says it is doing that, so a caller can decline to claim
# which turn ended the run.
TURN_ORDER_REFERENCE = "reference_vector"


@dataclass(frozen=True)
class ExecutionProjection:
    """Everything one side of a comparison shows, from read-only evidence.

    Partial by design. `unavailable` lists what could not be read (a pruned
    turn, a turn the store never held) and the projection still carries the
    turns that were readable, because half an execution is inspectable and
    refusing to render it would hide the half that survived.
    """

    ref: ExecutionRef
    turns: tuple[TurnProjection, ...]
    steps: tuple[ExecutionStep, ...]
    artifacts: tuple[ArtifactRef, ...]
    timing: Mapping[str, Any]
    cost: Mapping[str, Any]
    # Tokens, cache state and coverage over the same calls `cost` was summed
    # over. Per-call anchors stay on the turns that recorded them; see
    # `merge_usage_rollups`.
    usage: Mapping[str, Any] = field(default_factory=dict)
    unavailable: tuple[str, ...] = ()
    unassigned_steps: tuple[ExecutionStep, ...] = ()
    unattributed_artifacts: tuple[ArtifactRef, ...] = ()
    pass_selector: Optional[PassSelector] = None
    # Which of the two rules above put `turns` and `steps` in the order they
    # are in. Published rather than assumed, because a caller that reads the
    # last turn as the run's ending needs to know whether the evidence said
    # so.
    turn_order: str = TURN_ORDER_CHRONOLOGICAL
    turn_order_reason: Optional[str] = None

    @property
    def readable(self) -> bool:
        """At least one named turn was found. False is inspectable, not fatal."""
        return bool(self.turns)

    @property
    def content_attribution(self) -> str:
        """Whose the turn-level text is: the turn's, the pass's, or shared.

        `pass` only when EVERY projected turn recorded that pass's own content.
        One turn that did not is a view whose text is partly the shared turn's,
        and one summary word cannot say `pass` for it without overstating what
        the mixed set shows -- the per-turn labels still say which is which.
        """
        if not self.pass_selector:
            return ATTRIBUTION_TURN
        if self.turns and all(turn.pass_content_recorded for turn in self.turns):
            return ATTRIBUTION_PASS
        return ATTRIBUTION_SHARED

    def answers(self) -> list[dict[str, Any]]:
        """The answer of each turn, in order. The default view of a run.

        Each row carries its `attribution`, so a pass-scoped view cannot be read
        as "this is what the teacher answered" when what is recorded is what the
        turn answered.
        """
        return [
            {
                "turn_index": turn.turn_index,
                "turn_key": turn.turn_key,
                "answer": turn.answer,
                "status": turn.status,
                "success": turn.success,
                "failure_reason": turn.failure_reason,
                "pass_id": turn.pass_id,
                "attribution": turn.content_attribution,
                "pass_content_recorded": turn.pass_content_recorded,
            }
            for turn in self.turns
        ]

    def as_dict(self) -> dict[str, Any]:
        return {
            "ref": self.ref.as_dict(),
            "readable": self.readable,
            "content_attribution": self.content_attribution,
            "turns": [turn.as_dict() for turn in self.turns],
            "answers": self.answers(),
            "steps": [step.as_dict() for step in self.steps],
            "artifacts": [artifact.as_dict() for artifact in self.artifacts],
            "timing": dict(self.timing),
            "cost": dict(self.cost),
            "usage": dict(self.usage),
            "unavailable": list(self.unavailable),
            "turn_order": self.turn_order,
            "turn_order_reason": self.turn_order_reason,
            "unassigned_steps": [step.as_dict() for step in self.unassigned_steps],
            "unattributed_artifacts": [
                artifact.as_dict() for artifact in self.unattributed_artifacts
            ],
            "pass_selector": (
                self.pass_selector.as_dict() if self.pass_selector else None
            ),
        }


def _command_outputs(record: Any) -> list[Mapping[str, Any]]:
    if not isinstance(record, Mapping):
        return []
    turn_output = record.get("turn_output")
    if not isinstance(turn_output, Mapping):
        return []
    outputs = turn_output.get("command_outputs")
    if not isinstance(outputs, list):
        return []
    return [value for value in outputs if isinstance(value, Mapping)]


def _check_scope(ref: ExecutionRef, turn_key: str, row: Mapping[str, Any]) -> None:
    """Refuse a reference whose declared scope the turn row does not record.

    Only DECLARED fields are checked: a reference that names no experiment is
    not claiming one, and the recorded scope is reported on the projected turn
    either way. A declared field that the row leaves NULL is a mismatch too --
    "attempt 2 of task-1" over a turn recorded outside any experiment is not a
    partial truth, it is a different statement from the evidence's.
    """
    declared = (
        ("experiment_id", ref.experiment_id, _text_or_none(row.get("experiment_id"))),
        ("task_id", ref.task_id, _text_or_none(row.get("task_id"))),
        ("attempt", ref.attempt, _exact_int(row.get("attempt"))),
    )
    for field_name, claimed, recorded in declared:
        if claimed is None or claimed == recorded:
            continue
        raise ExecutionScopeMismatch(
            f"reference declares {field_name}={claimed!r} but turn {turn_key!r} "
            f"in store {ref.store_id!r} records "
            + (
                f"{field_name}={recorded!r}"
                if recorded is not None
                else f"no {field_name}"
            )
        )


def _artifacts_from_record(turn_key: str, record: Any) -> list[ArtifactRef]:
    refs: list[ArtifactRef] = []
    for output in _command_outputs(record):
        response = output.get("command_response")
        artifacts = response.get("artifacts") if isinstance(response, Mapping) else None
        if not isinstance(artifacts, Mapping):
            continue
        call_id = _text_or_none(output.get("command_call_id"))
        command_name = _text_or_none(output.get("command_name"))
        for key in sorted(artifacts):
            value = artifacts[key]
            envelope = value if isinstance(value, Mapping) else {}
            artifact_id = _text_or_none(envelope.get("__fw_artifact_ref__"))
            refs.append(
                ArtifactRef(
                    turn_key=turn_key,
                    command_call_id=call_id,
                    command_name=command_name,
                    key=str(key),
                    artifact_id=artifact_id,
                    size_bytes=_exact_int(envelope.get("size")),
                    content_type=_text_or_none(envelope.get("content_type")),
                    inline=artifact_id is None,
                    error=_text_or_none(envelope.get("error")),
                )
            )
    return refs


def _recorded_parameters(record: Any) -> dict[str, tuple[Any, Optional[bool]]]:
    """`{command_call_id: (parameters, response success)}` from the turn record.

    Keyed on `command_call_id` because that is the join the substrate already
    maintains between a `CommandOutput` and the dispatch that produced it. An
    output with no call id (a hand-built one, or a path that never stamped it)
    contributes nothing rather than being matched by position.
    """
    found: dict[str, tuple[Any, Optional[bool]]] = {}
    for output in _command_outputs(record):
        call_id = _text_or_none(output.get("command_call_id"))
        if call_id is None:
            continue
        response = output.get("command_response")
        success = (
            response.get("success") if isinstance(response, Mapping) else None
        )
        found[call_id] = (
            output.get("command_parameters"),
            success if isinstance(success, bool) else None,
        )
    return found


def _pass_id_for(
    selector: Optional[PassSelector],
    tree: _SpanTree,
    span_id: Optional[str],
) -> Optional[str]:
    """The pass a span belongs to, or None when the evidence does not say.

    A step with no recorded span can never be attributed: the span tree is the
    only thing that says which pass ran it, so such a step is withheld from
    every pass view (and counted in `unassigned_steps`) rather than assigned to
    the pass whose neighbours it sat between.
    """
    if selector is None:
        return None
    chain = tree.ancestry(span_id)
    chain_ids = {
        _text_or_none(span.get("span_id"))
        for span in chain
        if _text_or_none(span.get("span_id"))
    }
    if selector.exclude_span_ids & chain_ids:
        return None
    if span_id is not None and span_id in selector.span_ids:
        return selector.pass_id
    if selector.root_span_ids & chain_ids:
        return selector.pass_id
    if selector.attribute_key is not None:
        for span in chain:
            value = tree.attributes(span).get(selector.attribute_key)
            if isinstance(value, str) and value == selector.attribute_value:
                return selector.pass_id
    return None


def _spans_in_pass(
    selector: Optional[PassSelector],
    tree: _SpanTree,
    spans: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """The spans attributable to the selected pass, for cost roll-up.

    A span with no attributable pass is left out rather than shared between
    passes: a cost that cannot be attributed is better reported as the
    `unrecorded` it already is than split by guesswork.
    """
    if selector is None:
        return list(spans)
    selected = []
    for span in spans:
        span_id = _text_or_none(span.get("span_id"))
        if _pass_id_for(selector, tree, span_id) == selector.pass_id:
            selected.append(span)
    return selected


# The marker `capture_policy.CapturedValue.to_envelope` writes. Restated for the
# same reason as `SPAN_DISTILLATION_PASS` above -- this module reads stored
# evidence and stays off the runtime's import path -- and checked against the
# producer by `tests/test_distillation_pass_capture`.
CAPTURE_ENVELOPE_MARKER = "__fw_capture__"


def _recorded_text(value: Any) -> Optional[str]:
    """A recorded string attribute, including one an envelope stands in for.

    Three shapes arrive here and they mean three different things.

    A plain string is the value, whole.

    A tracing cap envelope (`{truncated, original_length, sha256, value}`,
    [R10]) means the emitter cut an over-limit attribute. The prefix in there is
    still recorded evidence, so it is quoted.

    A capture-policy envelope (`{__fw_capture__: True, ...}`) means the SINK
    acted on the field: `bounded-text` leaves a `prefix`, which is likewise
    recorded evidence and is quoted; every other disposition leaves no text at
    all, and this answers `None` for those.

    `None` here is therefore ambiguous on its own -- withheld and never-recorded
    look alike -- which is exactly why the envelope stays in
    `TurnProjection.pass_content`. A reader that needs to tell "this pass said
    nothing" from "this pass said something nobody may see" reads the envelope;
    the chatbot page does (`captureEnvelope` / `policedText` in
    run_chatbot/static/src/110-state-format.js), and badges the second.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping) and value.get("truncated"):
        text = value.get("value")
        return text if isinstance(text, str) else None
    if isinstance(value, Mapping) and value.get(CAPTURE_ENVELOPE_MARKER) is True:
        prefix = value.get("prefix")
        return prefix if isinstance(prefix, str) else None
    return None


def _pass_content(
    selector: Optional[PassSelector],
    tree: _SpanTree,
    spans: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], Optional[int]]:
    """What a producer recorded AS this pass's own, and the pass's wall time.

    Read off the pass span -- the one span of the pass whose name says it
    describes the pass rather than something the pass did. `({}, None)` when
    there is none, which is every trace recorded before producers stamped
    passes: the turn's content is then reported as shared rather than filled in
    from the turn and relabelled.

    Two such spans for one pass is an ambiguous question, and the answer to an
    ambiguous question is not one of the candidates: nothing is claimed.

    The wall time is the pass span's own recorded duration, so it is the pass's
    in the same sense the turn's is the turn's -- measured, not apportioned.
    """
    if selector is None:
        return {}, None
    found = [
        span
        for span in spans
        if span.get("name") == SPAN_DISTILLATION_PASS
        and _pass_id_for(selector, tree, _text_or_none(span.get("span_id")))
        == selector.pass_id
    ]
    if len(found) != 1:
        return {}, None
    attributes = tree.attributes(found[0])
    content = {key: attributes[key] for key in PASS_CONTENT_KEYS if key in attributes}
    start_ns = _exact_int(found[0].get("start_ns"))
    end_ns = _exact_int(found[0].get("end_ns"))
    wall_ms = (
        (end_ns - start_ns) // 1_000_000
        if start_ns is not None and end_ns is not None and end_ns >= start_ns
        else None
    )
    return content, wall_ms


def _root_duration_ns(steps: Sequence[ExecutionStep]) -> tuple[Optional[int], int]:
    """Summed duration of ROOT dispatches only, and how many had none.

    Root means "no parent among the steps in this list". A nested dispatch's
    time is already inside its parent's span, and the ledger deliberately
    lists both -- so summing every row would double-count exactly the nested
    wrappers this projection must not double-count. Returns None, never 0,
    when nothing recorded a duration.
    """
    known = {step.command_call_id for step in steps}
    total = 0
    counted = 0
    missing = 0
    for step in steps:
        if step.parent_call_id in known:
            continue
        if step.duration_ns is None:
            missing += 1
            continue
        total += step.duration_ns
        counted += 1
    return (total if counted else None), missing


def project_execution(
    ref: ExecutionRef,
    reader: ExecutionReader,
    *,
    ledger: Optional[LedgerProjection] = None,
    cost_rollup: Optional[CostRollup] = None,
    pass_selector: Optional[PassSelector] = None,
) -> ExecutionProjection:
    """Read one scoped execution into the shared projection.

    Read-only and side-effect free. Every turn named by the reference is
    attempted; one that cannot be read is recorded in `unavailable` and the
    rest still project. A turn that IS readable but whose recorded
    experiment/task/attempt contradicts the reference is a refusal, not a
    partial answer (`_check_scope`).

    Pass scope and pass identity must agree, in all three directions:

    - `ref.pass_id` with no `pass_selector` is refused. Nothing here can resolve
      pass membership without one, so the projection would be the whole turn
      under a pass's name -- the wrong answer this module exists to avoid.
    - a `pass_selector` with no `ref.pass_id` is refused: the steps would be one
      pass's while `ref_id()` -- and so every feedback anchor and review-pair
      key derived from it -- would be the whole turn's.
    - both set and disagreeing is refused.

    The selector is then resolved against each turn's recorded spans, so a pass
    the evidence does not contain fails loudly (`UnknownRecordedPass`).
    """
    if ref.pass_id is not None and pass_selector is None:
        raise InvalidExecutionRef(
            f"reference names pass {ref.pass_id!r} but no pass selector was "
            "supplied; pass membership is resolved from recorded spans and is "
            "never assumed"
        )
    if pass_selector is not None and ref.pass_id is None:
        raise InvalidExecutionRef(
            f"a pass selector for {pass_selector.pass_id!r} was supplied for a "
            "reference that names no pass; the projection would be pass-scoped "
            "while its ref_id, anchors and review pairs would be the whole "
            "turn's"
        )
    if pass_selector is not None and pass_selector.pass_id != ref.pass_id:
        raise InvalidExecutionRef(
            f"reference names pass {ref.pass_id!r} but the selector resolves "
            f"pass {pass_selector.pass_id!r}"
        )
    ledger_fn = ledger or default_ledger_projection()
    cost_fn = cost_rollup or default_cost_rollup()

    turns: list[TurnProjection] = []
    steps: list[ExecutionStep] = []
    unassigned: list[ExecutionStep] = []
    artifacts: list[ArtifactRef] = []
    unattributed_artifacts: list[ArtifactRef] = []
    unavailable: list[str] = []
    cost_parts: list[Mapping[str, Any]] = []
    usage_parts: list[Mapping[str, Any]] = []
    wall_ms_total = 0
    wall_ms_known = 0
    pass_wall_ms_total = 0
    pass_wall_ms_known = 0

    # Read first, in reference order, so `unavailable` still reads in the
    # order the reference names its turns and a scope contradiction still
    # raises on the first turn that has one.
    readable: list[tuple[str, Mapping[str, Any]]] = []
    for turn_key in ref.turn_keys:
        row = reader.turn(ref.store_id, turn_key)
        if row is None:
            unavailable.append(f"turn {turn_key!r} is not in store {ref.store_id!r}")
            continue
        _check_scope(ref, turn_key, row)
        readable.append((turn_key, row))

    # Then order the CONTENT by what was recorded. The reference vector is
    # untouched: `ref` goes into the projection exactly as it arrived.
    ordered, turn_order, turn_order_reason = _chronological_turns(readable)

    for turn_index, (turn_key, row) in enumerate(ordered):
        spans = list(reader.trace(ref.store_id, turn_key))
        tree = _SpanTree(spans)
        if pass_selector is not None:
            pass_selector.resolve_against(tree, turn_key)
        record = row.get("record")
        ledger_rows = ledger_fn(record, spans)
        recorded_params = _recorded_parameters(record)

        by_span: dict[str, Mapping[str, Any]] = {
            _text_or_none(span.get("span_id")): span
            for span in spans
            if _text_or_none(span.get("span_id"))
        }

        turn_steps: list[ExecutionStep] = []
        for raw in ledger_rows.get("rows") or []:
            call_id = _text_or_none(raw.get("command_call_id"))
            if call_id is None:
                continue
            span_id = _text_or_none(raw.get("span_id"))
            parameters: Any = None
            source: Optional[str] = None
            response_success: Optional[bool] = None
            if call_id in recorded_params:
                parameters, response_success = recorded_params[call_id]
                if parameters is not None:
                    source = "record"
            if parameters is None and span_id in by_span:
                span = by_span[span_id]
                if span.get("name") == SPAN_COMMAND_EXECUTE:
                    candidate = tree.attributes(span).get("parameters")
                    if candidate is not None:
                        parameters = candidate
                        source = "span"
            step = ExecutionStep(
                turn_index=turn_index,
                turn_key=turn_key,
                position=int(raw.get("position") or 0),
                command_call_id=call_id,
                parent_call_id=_text_or_none(raw.get("parent_call_id")),
                command_ordinal=_exact_int(raw.get("command_ordinal")),
                span_id=span_id,
                command_name=_text_or_none(raw.get("command_name")),
                context=_text_or_none(raw.get("context")),
                status=_text_or_none(raw.get("status")),
                success=raw.get("success") if isinstance(raw.get("success"), bool) else None,
                start_ns=_exact_int(raw.get("start_ns")),
                duration_ns=_exact_int(raw.get("duration_ns")),
                in_record=bool(raw.get("in_record")),
                span_recorded=bool(raw.get("span_recorded")),
                child_call=bool(raw.get("child_call")),
                asked_user=int(raw.get("asked_user") or 0),
                parameters=parameters,
                parameters_source=source,
                parameters_digest=(
                    _digest(_canonical_json(parameters)) if parameters is not None else None
                ),
                response_success=response_success,
                pass_id=_pass_id_for(pass_selector, tree, span_id),
            )
            if pass_selector is not None and step.pass_id != pass_selector.pass_id:
                unassigned.append(step)
                continue
            turn_steps.append(step)

        pass_spans = _spans_in_pass(pass_selector, tree, spans)
        # `cost_fn` is INJECTED, and the default one folds duplicates itself
        # (`server.cost_rollup` delegates to `usage_rollup`). Folding the input
        # here too is for the callers that pass their own: handing a raw list to
        # a naive roll-up charged a re-emitted span (`end_span` reuses the
        # span_id `start_span` opened) and a nested wrapper twice, while the
        # token figures beside it counted once -- two numbers on one screen
        # disagreeing about how many calls a turn made. The fold is idempotent
        # and leaves every non-LLM span untouched, so the default roll-up is
        # unaffected by seeing an already-folded list.
        #
        # It does NOT fold non-nested calls that merely share a response: those
        # are two calls and `usage_rollup` is the only layer that decides which
        # one is credited. An injected roll-up that sums both will disagree with
        # the projection on that shape; the shipped one does not, because it is
        # the same accounting.
        turn_cost = dict(cost_fn(canonical_llm_spans(pass_spans)))
        turn_usage = usage_rollup(pass_spans, turn_key=turn_key)
        cost_parts.append(turn_cost)
        usage_parts.append(turn_usage)

        started = _text_or_none(row.get("started_at"))
        completed = _text_or_none(row.get("completed_at"))
        wall = _wall_ms(started, completed)
        if wall is not None:
            wall_ms_total += wall
            wall_ms_known += 1

        # What the producer recorded as this pass's own, if anything did.
        pass_content, pass_wall = _pass_content(pass_selector, tree, spans)
        pass_recorded = bool(pass_content)
        if pass_wall is not None:
            pass_wall_ms_total += pass_wall
            pass_wall_ms_known += 1
        turn_answer = row.get("answer") if isinstance(row.get("answer"), str) else None
        turn_status = _text_or_none(row.get("status"))

        turns.append(
            TurnProjection(
                turn_index=turn_index,
                turn_key=turn_key,
                status=(
                    _text_or_none(pass_content.get("status"))
                    if pass_recorded
                    else turn_status
                ),
                # The turn row's success code describes the whole turn, so under
                # a recorded pass it is withheld rather than re-labelled: no
                # producer records a per-pass one, and absent is what that is.
                success=None if pass_recorded else _bool_column(row.get("success")),
                failure_reason=(
                    _text_or_none(pass_content.get("failure_reason"))
                    if pass_recorded
                    else _text_or_none(row.get("failure_reason"))
                ),
                answer=(
                    _recorded_text(pass_content.get("answer"))
                    if pass_recorded
                    else turn_answer
                ),
                user_message=(
                    row.get("user_message")
                    if isinstance(row.get("user_message"), str)
                    else None
                ),
                started_at=started,
                completed_at=completed,
                suspended_ms=_exact_int(row.get("suspended_ms")),
                experiment_id=_text_or_none(row.get("experiment_id")),
                task_id=_text_or_none(row.get("task_id")),
                attempt=_exact_int(row.get("attempt")),
                ledger_summary={
                    key: ledger_rows.get(key)
                    for key in (
                        "record_rows",
                        "span_rows",
                        "rows_not_in_record",
                        "rows_without_span",
                        "asked_user_outside_dispatch",
                    )
                },
                cost=turn_cost,
                usage=turn_usage,
                pass_id=ref.pass_id,
                content_attribution=(
                    ATTRIBUTION_PASS
                    if pass_recorded
                    else (ATTRIBUTION_SHARED if pass_selector else ATTRIBUTION_TURN)
                ),
                # True only where the producer actually recorded this pass's own
                # content. A trace with no pass span reads False and its turn
                # text stays shared -- the pre-`fix-txxy` shape, preserved
                # rather than repaired.
                pass_content_recorded=pass_recorded,
                plan=(
                    _recorded_text(pass_content.get("plan"))
                    if pass_recorded
                    else None
                ),
                turn_answer=turn_answer,
                turn_status=turn_status,
                pass_content=pass_content,
            )
        )
        steps.extend(turn_steps)
        # Artifacts follow their dispatch. In a pass-scoped projection an
        # artifact whose CommandOutput recorded no command_call_id cannot be
        # joined to one, so it is reported as the turn's unattributed output
        # rather than shown under both passes as if each had produced it.
        in_pass = {step.command_call_id for step in turn_steps}
        for artifact in _artifacts_from_record(turn_key, record):
            if pass_selector is None:
                artifacts.append(artifact)
            elif artifact.command_call_id is None:
                unattributed_artifacts.append(
                    replace(artifact, attribution=ATTRIBUTION_UNATTRIBUTED)
                )
            elif artifact.command_call_id in in_pass:
                artifacts.append(replace(artifact, attribution=ATTRIBUTION_PASS))

    duration_ns, steps_without_duration = _root_duration_ns(steps)
    turn_wall_ms = wall_ms_total if wall_ms_known else None
    pass_wall_ms = pass_wall_ms_total if pass_wall_ms_known else None
    timing = {
        "turns": len(turns),
        # Two passes share one turn row's started_at/completed_at, so the turn
        # figure is never a pass's own. A pass span, where one was recorded, has
        # its own measured duration and that IS the pass's; where none was, this
        # stays None and the shared turn figure is reported beside it, labelled,
        # rather than presented as the pass's. `root_step_duration_ns` and
        # `cost` below are pass-scoped either way: they are summed over the
        # steps and spans attributed to the pass.
        "wall_ms": pass_wall_ms if pass_selector else turn_wall_ms,
        "wall_ms_attribution": (
            (ATTRIBUTION_PASS if pass_wall_ms_known else ATTRIBUTION_SHARED)
            if pass_selector
            else ATTRIBUTION_TURN
        ),
        "turn_wall_ms": turn_wall_ms,
        # How many of the projected turns recorded a pass span to measure, so a
        # partial figure cannot read as a complete one.
        "pass_wall_ms_turns_recorded": pass_wall_ms_known,
        "wall_ms_turns_recorded": wall_ms_known,
        "wall_ms_turns_unrecorded": len(turns) - wall_ms_known,
        "root_step_duration_ns": duration_ns,
        "root_steps_without_duration": steps_without_duration,
        "steps": len(steps),
    }
    return ExecutionProjection(
        ref=ref,
        turns=tuple(turns),
        steps=tuple(steps),
        artifacts=tuple(artifacts),
        timing=timing,
        cost=_merge_cost(cost_parts),
        usage=merge_usage_rollups(usage_parts),
        unavailable=tuple(unavailable),
        unassigned_steps=tuple(unassigned),
        unattributed_artifacts=tuple(unattributed_artifacts),
        pass_selector=pass_selector,
        turn_order=turn_order,
        turn_order_reason=turn_order_reason,
    )


def _chronological_turns(
    readable: list[tuple[str, Mapping[str, Any]]],
) -> tuple[list[tuple[str, Mapping[str, Any]]], str, Optional[str]]:
    """Readable turns in the order the evidence says they happened.

    Two rules, in this order, and an explicit unknown when neither holds.

    **`ordinal`, within ONE conversation.** The store assigns it densely from 1
    at first insert and never rewrites it, and the store's own turn listings
    order by it: inside a conversation it IS the recorded sequence number.
    Its scope is exactly that, though -- `conversation_counters` is keyed by
    channel and the ids are minted per channel (`mint_conversation_id`), so
    ordinal 2 of one conversation and ordinal 2 of another say nothing about
    each other, and two conversations can interleave or resume. So the ordinal
    rule is applied ONLY when every turn records the same channel and the same
    conversation and their ordinals are usable and distinct.

    **Parsed `started_at`, across conversations.** Timestamps are compared as
    instants, not as strings: two recorded with different UTC offsets sort
    backwards lexicographically. Used only when every turn has one that parses
    and no two are the same instant.

    Otherwise the reference's order is kept and the projection says the order
    is unknown. Nothing here tie-breaks on `turn_key`: a key sorts, it does not
    record when anything happened, and a projection that reordered turns on
    that would move a run's ending -- which is the one thing this function
    exists to get right.
    """
    if len(readable) < 2:
        return readable, TURN_ORDER_CHRONOLOGICAL, None

    def _int_or_none(value: Any) -> Optional[int]:
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    channels = {_text_or_none(row.get("channel_id")) for _key, row in readable}
    conversations = {_int_or_none(row.get("conversation_id")) for _key, row in readable}
    ordinals = [_int_or_none(row.get("ordinal")) for _key, row in readable]
    one_conversation = (
        len(channels) == 1
        and None not in channels
        and len(conversations) == 1
        and None not in conversations
    )
    if one_conversation and None not in ordinals and len(set(ordinals)) == len(ordinals):
        return (
            sorted(readable, key=lambda item: _int_or_none(item[1].get("ordinal")) or 0),
            TURN_ORDER_CHRONOLOGICAL,
            None,
        )

    instants = [_instant(row.get("started_at")) for _key, row in readable]
    if None not in instants and len(set(instants)) == len(instants):
        order = {id(row): instant for (_key, row), instant in zip(readable, instants)}
        return (
            sorted(readable, key=lambda item: order[id(item[1])]),
            TURN_ORDER_CHRONOLOGICAL,
            None,
        )

    if not one_conversation:
        why = (
            "these turns were recorded across more than one conversation or "
            "channel, where ordinals restart and say nothing about each other, "
            "and their start times are missing or tied"
        )
    else:
        why = (
            "these turns record no usable distinct ordinal and no distinct "
            "start time"
        )
    return (
        readable,
        TURN_ORDER_REFERENCE,
        (
            f"{why}, so the order they happened in is not recoverable from the "
            "evidence; the reference's order is shown and nothing here claims "
            "it is the recorded one"
        ),
    )


def _instant(value: Any) -> Optional[float]:
    """A recorded timestamp as an instant, or None if it is not one.

    Compared as time rather than as text: `2026-09-19T02:00:00+02:00` is BEFORE
    `2026-09-19T01:00:00+00:00`, and a string sort puts them the other way
    round.
    """
    text = _text_or_none(value)
    if text is None:
        return None
    from datetime import datetime, timezone

    try:
        parsed = datetime.fromisoformat(text)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        # A naive stamp is a recorded local reading with no offset beside it.
        # Treated as UTC only for ORDERING against other naive stamps; mixing
        # it with an offset-bearing one is exactly the ambiguity the caller
        # falls back to "unknown" for, and equal instants trip that anyway.
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _bool_column(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    return None


def _wall_ms(started_at: Optional[str], completed_at: Optional[str]) -> Optional[int]:
    """Recorded wall time of one turn, or None. Never a negative duration."""
    if not started_at or not completed_at:
        return None
    from datetime import datetime

    try:
        start = datetime.fromisoformat(started_at)
        end = datetime.fromisoformat(completed_at)
    except (TypeError, ValueError):
        return None
    delta = int((end - start).total_seconds() * 1000)
    return delta if delta >= 0 else None


def _merge_cost(parts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Sum per-turn roll-ups, keeping `total: None` when nothing recorded one."""
    calls = recorded = unrecorded = 0
    total = 0.0
    for part in parts:
        calls += int(part.get("calls") or 0)
        recorded += int(part.get("recorded") or 0)
        unrecorded += int(part.get("unrecorded") or 0)
        value = part.get("total")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            total += float(value)
    return {
        "calls": calls,
        "recorded": recorded,
        "unrecorded": unrecorded,
        "total": total if recorded else None,
    }


# ----------------------------------------------------------------------
# Token, cost and cache completeness over recorded LLM calls
# (`fix-9eg.5`, `fix-9eg.6`)
# ----------------------------------------------------------------------
#
# `cost_rollup` above already answers "what did the money add up to, and how
# many calls did not say". Two things a reader also asks it cannot answer, and
# one thing it can get wrong:
#
#   - HOW MANY TOKENS, split into prompt and completion, with zero kept apart
#     from unknown. `usage` is an attribute on `fw.llm.call` that a provider
#     may omit entirely, may fill in completely, or may fill in partly; a
#     total of 0 is a measurement and "no usage attribute" is not.
#   - WHETHER THE CALL WAS SERVED FROM THE LLM CACHE. `cache_hit` is recorded
#     when the DSPy history entry carried a response to read it from, and is
#     simply absent otherwise. A hit is an observation about how the answer
#     arrived, not a verdict about it, and an absent flag is unknown -- never
#     "miss" (`fix-9eg.6`).
#   - ONE LLM CALL CAN BE RECORDED MORE THAN ONCE. `tracing.end_span` re-emits
#     a span under the span_id `start_span` opened, so a reader that sees both
#     records (a live trace read of an in-flight turn, or two reads merged)
#     has two rows for one call; and a wrapper LM that itself invokes an LM
#     leaves a nested `fw.llm.call` quoting the SAME provider response as the
#     inner one. Summing per row double-counts both.
#
# So the accounting below resolves CANONICAL CALLS first and rolls up from
# those. The two folds are named separately in the result, because they are
# different facts about the recording and a reader who sees a difference
# between this and a naive per-span count deserves to know which one caused it.
#
# The same numbers serve the browser and a coding agent: they travel on the
# projection, beside `cost`, with the span_id of every call that contributed
# (`calls_detail`), so a human chip and an agent's structured read anchor to
# the same recorded call rather than to two independent tallies.

SPAN_LLM_CALL = "fw.llm.call"

# What one call's `usage` attribute amounted to.
USAGE_COMPLETE = "complete"        # prompt, completion and total all recorded
USAGE_PARTIAL = "partial"          # some of the three, not all
USAGE_UNRECORDED = "unrecorded"    # no usage attribute at all -- unknown, not 0
# A call whose provider response another call already accounted for. Its tokens
# and cost are not added again; it is still a call that was made.
USAGE_SHARED_RESPONSE = "shared_response"

# What the recorded `cache_hit` flag said. `unknown` is its own answer.
CACHE_HIT = "hit"
CACHE_MISS = "miss"
CACHE_UNKNOWN = "unknown"

# Coverage of the token figures: nothing recorded, some calls recorded, or
# every counted call recorded the full split.
COVERAGE_NONE = "none"
COVERAGE_PARTIAL = "partial"
COVERAGE_COMPLETE = "complete"


@dataclass(frozen=True)
class LlmCallUsage:
    """One canonical `fw.llm.call`: what it spent, and how it was answered.

    The span_id is the anchor: it is what the trace view renders a level for
    and what an agent quotes to point at this exact call, so a chip and a
    structured read never disagree about which call they mean.

    `response_id` is the recorded `history_uuid` -- DSPy's identifier for the
    provider response the usage and cost were copied from. It is what makes a
    duplicate recognisable: two calls quoting one response cannot both have
    spent those tokens.
    """

    span_id: str
    turn_key: Optional[str]
    parent_span_id: Optional[str]
    response_id: Optional[str]
    model: Optional[str]
    completed: bool
    status: Optional[str]
    usage_state: str
    cache_state: str
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]
    cost: Optional[float]
    records_folded: int = 0
    wrappers_folded: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "turn_key": self.turn_key,
            "parent_span_id": self.parent_span_id,
            "response_id": self.response_id,
            "model": self.model,
            "completed": self.completed,
            "status": self.status,
            "usage_state": self.usage_state,
            "cache_state": self.cache_state,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost": self.cost,
            "records_folded": self.records_folded,
            "wrappers_folded": self.wrappers_folded,
        }


def _usage_attribute(tree: _SpanTree, span: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    """The `usage` attribute as a mapping, or None when the call recorded none.

    Persisted as JSON text by `dspy_logger._json_text` and handed back either
    decoded or raw depending on the reader, exactly as the enclosing attributes
    are -- so both forms are accepted here for the same reason `_SpanTree`
    accepts both for the attributes themselves.
    """
    raw = tree.attributes(span).get("usage")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            return None
    return dict(raw) if isinstance(raw, Mapping) else None


def _exact_count(value: Any) -> Optional[int]:
    """An exact, non-negative integer, or None.

    A token count is a tally of things that happened, so a negative one is not a
    smaller expenditure -- it is a malformed record, and summing it would make a
    turn's total smaller than one of its own calls. Kept local to the token
    parser on purpose: `_exact_int` is what reads attempt numbers, ordinals and
    durations elsewhere in this module, and narrowing it would change how those
    are read for a reason that has nothing to do with them.
    """
    exact = _exact_int(value)
    return exact if exact is not None and exact >= 0 else None


def _cache_state(attributes: Mapping[str, Any]) -> str:
    """`hit`, `miss` or `unknown`, from the recorded flag only.

    `dspy_logger` writes a real boolean; anything else on record is a shape
    this reader does not recognise and is reported as unknown rather than
    coerced, because a truthy string would otherwise read as a hit.
    """
    value = attributes.get("cache_hit")
    if value is True:
        return CACHE_HIT
    if value is False:
        return CACHE_MISS
    return CACHE_UNKNOWN


def _llm_cost_attribute(attributes: Mapping[str, Any]) -> Optional[float]:
    """The recorded `cost`, or None. Same rule as `server.llm_call_cost`:
    finite, non-negative, and never a bool -- `True` is not a cost of 1."""
    value = attributes.get("cost")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number if number >= 0 else None


def _canonical_llm_records(
    spans: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], int]:
    """One record per `fw.llm.call` span_id, plus how many were folded away.

    A re-emission of a span carries the same span_id (`[R2][R6]`: a store treats
    it as an idempotent upsert), and the ENDED record is the complete one, so it
    wins -- the same precedence `ObservabilityStore.upsert_span_rows` applies in
    SQL. Among two records that are both ended, or both still open, the later
    one in the list wins, because a reader that saw both read them in order.

    A span with no span_id cannot be folded or anchored to, so it is kept as
    its own record and counted as a call; dropping it would lose a call that
    was made.
    """
    canonical: dict[str, Mapping[str, Any]] = {}
    anonymous: list[Mapping[str, Any]] = []
    folded = 0
    for span in spans:
        if span.get("name") != SPAN_LLM_CALL:
            continue
        span_id = _text_or_none(span.get("span_id"))
        if span_id is None:
            anonymous.append(span)
            continue
        previous = canonical.get(span_id)
        if previous is None:
            canonical[span_id] = span
            continue
        folded += 1
        if previous.get("end_ns") is not None and span.get("end_ns") is None:
            continue  # an open re-read must not overwrite the ended record
        canonical[span_id] = span
    return list(canonical.values()) + anonymous, folded


def _wrapper_span_ids(
    tree: _SpanTree, records: Sequence[Mapping[str, Any]]
) -> set[str]:
    """The span_ids of `fw.llm.call` records that are a WRAPPER of another.

    A wrapper is an `fw.llm.call` with an `fw.llm.call` descendant quoting the
    same `history_uuid`: one provider response recorded at two levels, which is
    one call. The inner record is kept because it is the one closest to the
    provider, and the outer is folded away -- so a nested LM adapter cannot
    make a turn look like it made twice as many calls as it did.

    Calls that merely SHARE a response without being nested are left alone
    here: they are two calls, and which of them really spent the tokens is not
    something this layer can decide. `_usage_rollup_from` counts that response
    once and says how many calls quoted it.
    """
    by_id = {
        _text_or_none(record.get("span_id")): record
        for record in records
        if _text_or_none(record.get("span_id"))
    }
    wrappers: set[str] = set()
    for span_id, record in by_id.items():
        response = _text_or_none(tree.attributes(record).get("history_uuid"))
        if response is None:
            continue
        for ancestor in tree.ancestry(_text_or_none(record.get("parent_span_id"))):
            ancestor_id = _text_or_none(ancestor.get("span_id"))
            if ancestor_id is None or ancestor_id not in by_id:
                continue
            if _text_or_none(tree.attributes(ancestor).get("history_uuid")) == response:
                wrappers.add(ancestor_id)
    return wrappers


def canonical_llm_spans(
    spans: Iterable[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """`spans` with every `fw.llm.call` present exactly once.

    A re-emitted record is merged into the one call it re-records, and a
    wrapper `fw.llm.call` quoting its own descendant's provider response is
    dropped. Spans of every other name pass through untouched and in order, so
    this is safe to hand to any roll-up that filters on the LLM span name --
    which is how the injected `cost_rollup` gets the same denominator the
    token figures beside it use, without being reimplemented here.
    """
    span_list = list(spans)
    records, _ = _canonical_llm_records(span_list)
    tree = _SpanTree(
        [span for span in span_list if span.get("name") != SPAN_LLM_CALL] + records
    )
    wrappers = _wrapper_span_ids(tree, records)
    kept = {
        _text_or_none(record.get("span_id")) or id(record): record
        for record in records
        if (_text_or_none(record.get("span_id")) or "") not in wrappers
    }
    emitted: set[Any] = set()
    result: list[Mapping[str, Any]] = []
    for span in span_list:
        if span.get("name") != SPAN_LLM_CALL:
            result.append(span)
            continue
        key = _text_or_none(span.get("span_id")) or id(span)
        if key in emitted or key not in kept:
            continue
        emitted.add(key)
        result.append(kept[key])
    return result


def _order_key(record: Mapping[str, Any]) -> tuple[int, int, str]:
    """Recording order: by start time, then by span_id so ties are stable.

    Which of two calls quoting one response is credited with it has to be
    decided the same way twice, or the same evidence would project two
    different answers.
    """
    start = _exact_int(record.get("start_ns"))
    return (0 if start is not None else 1, start or 0, str(record.get("span_id") or ""))


def usage_rollup(
    spans: Iterable[Mapping[str, Any]], *, turn_key: Optional[str] = None
) -> dict[str, Any]:
    """Tokens, cost and cache state over the canonical LLM calls in `spans`.

    Read-only and total: every `fw.llm.call` in the input is either counted as
    a call or named as a fold, and every figure that nothing recorded is None
    or `unknown` rather than 0.

    `turn_key` is stamped on each anchor so a call can be addressed from a
    projection that spans several turns.
    """
    span_list = list(spans)
    records, records_folded = _canonical_llm_records(span_list)
    # The tree is built with the CANONICAL records last, so that they win
    # `_SpanTree`'s last-one-per-span_id rule. Reading a wrapper's response id
    # off a superseded open record would find nothing -- `history_uuid` is
    # written when the span ends -- and the nesting fold would silently stop
    # working on exactly the traces it exists for.
    tree = _SpanTree(
        [span for span in span_list if span.get("name") != SPAN_LLM_CALL] + records
    )
    wrappers = _wrapper_span_ids(tree, records)
    calls: list[LlmCallUsage] = []
    counted_responses: set[str] = set()

    for record in sorted(records, key=_order_key):
        span_id = _text_or_none(record.get("span_id"))
        if span_id is not None and span_id in wrappers:
            continue
        attributes = tree.attributes(record)
        response = _text_or_none(attributes.get("history_uuid"))
        usage = _usage_attribute(tree, record)
        shared = response is not None and response in counted_responses
        if response is not None:
            counted_responses.add(response)

        prompt = completion = total = None
        cost = None
        if shared:
            state = USAGE_SHARED_RESPONSE
        elif usage is None:
            state = USAGE_UNRECORDED
        else:
            prompt = _exact_count(usage.get("prompt_tokens"))
            completion = _exact_count(usage.get("completion_tokens"))
            total = _exact_count(usage.get("total_tokens"))
            recorded = [value for value in (prompt, completion, total) if value is not None]
            if not recorded:
                state = USAGE_UNRECORDED
            elif len(recorded) == 3:
                state = USAGE_COMPLETE
            else:
                state = USAGE_PARTIAL
        if not shared:
            cost = _llm_cost_attribute(attributes)

        calls.append(
            LlmCallUsage(
                span_id=span_id or "",
                turn_key=turn_key,
                parent_span_id=_text_or_none(record.get("parent_span_id")),
                response_id=response,
                model=_text_or_none(attributes.get("model")),
                completed=record.get("end_ns") is not None,
                status=_text_or_none(record.get("status")),
                usage_state=state,
                cache_state=_cache_state(attributes),
                prompt_tokens=prompt,
                completion_tokens=completion,
                total_tokens=total,
                cost=cost,
            )
        )

    return _usage_rollup_from(
        calls,
        records_folded=records_folded,
        wrappers_folded=len(wrappers),
    )


def cost_rollup(spans: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Sum of recorded costs over the CANONICAL LLM calls, with the unrecorded
    count beside it. `total` is None -- never 0 -- when no call recorded a cost.

    Canonical because one provider call can appear in a span list more than
    once, and charging each appearance bills money that was never spent
    (fix-9eg.5). Two things duplicate: `end_span` re-emits a span under the same
    `span_id`, and a wrapper `fw.llm.call` nests around the inner call that
    produced the same `history_uuid`. An outer+inner pair that each recorded
    $0.25 for one response was summed to $0.50 here while the comparison
    projection said $0.25 -- so a turn row and the same turn's comparison
    disagreed about what it cost, and every per-turn, per-attempt and
    navigation figure that reads this function was inflated.

    DELEGATED rather than reimplemented. Folding duplicates needs three rules --
    one record per `span_id`, the innermost record of a nested pair, and the
    first of several siblings quoting one `history_uuid` -- and a second copy of
    those rules is a second chance to get them apart. It already happened: a
    partial fold here agreed with `usage_rollup` on nested calls and still
    reported $0.50 against its $0.25 for two NON-nested spans naming the same
    response. `usage_rollup` is the one accounting, and it never calls back into
    `cost_rollup`, so there is no cycle.

    The returned shape is unchanged: `calls` still counts every call the trace
    holds, a duplicate that is not charged still shows up in `unrecorded` rather
    than vanishing, and `total` is still None -- never 0 -- when no call
    recorded a cost.
    """
    return usage_rollup(spans)["cost"]


def _sum_recorded(values: Iterable[Optional[int]]) -> Optional[int]:
    """Sum of the values that were recorded, or None when none was.

    The distinction this whole section exists for: an execution whose calls
    recorded no completion count has completion `None`, and one whose single
    call recorded `completion_tokens: 0` has completion `0`.
    """
    recorded = [value for value in values if value is not None]
    return sum(recorded) if recorded else None


def _coverage(complete: int, partial: int, unrecorded: int) -> str:
    if complete == 0 and partial == 0:
        return COVERAGE_NONE
    if partial == 0 and unrecorded == 0:
        return COVERAGE_COMPLETE
    return COVERAGE_PARTIAL


def _usage_rollup_from(
    calls: Sequence[LlmCallUsage], *, records_folded: int, wrappers_folded: int
) -> dict[str, Any]:
    """The wire shape, from resolved calls. One place, so the merge below and
    `usage_rollup` above cannot drift on what `coverage` or `total` mean."""
    complete = sum(1 for call in calls if call.usage_state == USAGE_COMPLETE)
    partial = sum(1 for call in calls if call.usage_state == USAGE_PARTIAL)
    unrecorded = sum(1 for call in calls if call.usage_state == USAGE_UNRECORDED)
    shared = sum(1 for call in calls if call.usage_state == USAGE_SHARED_RESPONSE)
    with_cost = [call.cost for call in calls if call.cost is not None]
    totals = _sum_recorded(call.total_tokens for call in calls)
    return {
        "calls": len(calls),
        "completed": sum(1 for call in calls if call.completed),
        "open": sum(1 for call in calls if not call.completed),
        "records_folded": records_folded,
        "wrappers_folded": wrappers_folded,
        "shared_responses": shared,
        "tokens": {
            "prompt": _sum_recorded(call.prompt_tokens for call in calls),
            "completion": _sum_recorded(call.completion_tokens for call in calls),
            "total": totals,
            "complete": complete,
            "partial": partial,
            "unrecorded": unrecorded,
            "shared": shared,
            # A recorded zero is a measurement; it is reported as one so that a
            # reader can tell "this call spent nothing" from "nobody counted".
            "zero": sum(1 for call in calls if call.total_tokens == 0),
            "coverage": _coverage(complete, partial, unrecorded + shared),
        },
        "cost": {
            "calls": len(calls),
            "recorded": len(with_cost),
            "unrecorded": len(calls) - len(with_cost),
            "total": sum(with_cost) if with_cost else None,
        },
        "cache": {
            CACHE_HIT: sum(1 for call in calls if call.cache_state == CACHE_HIT),
            CACHE_MISS: sum(1 for call in calls if call.cache_state == CACHE_MISS),
            CACHE_UNKNOWN: sum(
                1 for call in calls if call.cache_state == CACHE_UNKNOWN
            ),
        },
        "calls_detail": [call.as_dict() for call in calls],
    }


def merge_usage_rollups(parts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Sum per-turn roll-ups into one, keeping None for "nothing recorded".

    Per-turn parts are summed rather than re-derived from a concatenated span
    list because a trace is read per turn; a provider response cannot appear in
    two turns' traces without one of them being a copy, so the per-turn
    deduplication is the whole of it.

    `calls_detail` is NOT carried up. The anchors belong to the turn that
    recorded them and are published there, once; repeating them at execution
    scope would put the same span_id on the wire twice and invite a reader to
    tally it twice.
    """
    merged: dict[str, Any] = {
        "calls": 0,
        "completed": 0,
        "open": 0,
        "records_folded": 0,
        "wrappers_folded": 0,
        "shared_responses": 0,
    }
    tokens = {
        "complete": 0, "partial": 0, "unrecorded": 0, "shared": 0, "zero": 0,
    }
    sums: dict[str, Optional[int]] = {"prompt": None, "completion": None, "total": None}
    cost_calls = cost_recorded = cost_unrecorded = 0
    cost_total: Optional[float] = None
    cache = {CACHE_HIT: 0, CACHE_MISS: 0, CACHE_UNKNOWN: 0}

    for part in parts:
        for key in merged:
            merged[key] += int(part.get(key) or 0)
        part_tokens = part.get("tokens") or {}
        for key in tokens:
            tokens[key] += int(part_tokens.get(key) or 0)
        for key in sums:
            value = _exact_count(part_tokens.get(key))
            if value is not None:
                sums[key] = value if sums[key] is None else sums[key] + value
        part_cost = part.get("cost") or {}
        cost_calls += int(part_cost.get("calls") or 0)
        cost_recorded += int(part_cost.get("recorded") or 0)
        cost_unrecorded += int(part_cost.get("unrecorded") or 0)
        amount = part_cost.get("total")
        if isinstance(amount, (int, float)) and not isinstance(amount, bool):
            cost_total = float(amount) if cost_total is None else cost_total + float(amount)
        part_cache = part.get("cache") or {}
        for key in cache:
            cache[key] += int(part_cache.get(key) or 0)

    merged["tokens"] = {
        "prompt": sums["prompt"],
        "completion": sums["completion"],
        "total": sums["total"],
        "coverage": _coverage(
            tokens["complete"],
            tokens["partial"],
            tokens["unrecorded"] + tokens["shared"],
        ),
        **tokens,
    }
    merged["cost"] = {
        "calls": cost_calls,
        "recorded": cost_recorded,
        "unrecorded": cost_unrecorded,
        "total": cost_total if cost_recorded else None,
    }
    merged["cache"] = cache
    return merged


# ----------------------------------------------------------------------
# Alignment
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class AlignedPair:
    """One row of the comparison: two steps, one step, or an ambiguous match.

    `ambiguous` is not a soft form of `matched`. It means the recorded
    evidence admits more than one correspondence -- a weaker key, or a key
    that repeats -- and the UI is expected to show both sides without
    asserting they are the same call.
    """

    kind: str
    basis: str
    left: Optional[ExecutionStep] = None
    right: Optional[ExecutionStep] = None
    ambiguous: bool = False
    ambiguity_reason: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "basis": self.basis,
            "ambiguous": self.ambiguous,
            "ambiguity_reason": self.ambiguity_reason,
            "left": self.left.as_dict() if self.left else None,
            "right": self.right.as_dict() if self.right else None,
        }


@dataclass(frozen=True)
class Alignment:
    """The pairs plus the counts a reader needs before reading them."""

    pairs: tuple[AlignedPair, ...]
    degraded: bool = False
    degraded_reason: Optional[str] = None

    def summary(self) -> dict[str, Any]:
        matched = sum(1 for pair in self.pairs if pair.kind == PAIR_MATCHED)
        return {
            "pairs": len(self.pairs),
            "matched": matched,
            "ambiguous": sum(1 for pair in self.pairs if pair.ambiguous),
            "left_only": sum(1 for pair in self.pairs if pair.kind == PAIR_LEFT_ONLY),
            "right_only": sum(1 for pair in self.pairs if pair.kind == PAIR_RIGHT_ONLY),
            "unknown": sum(1 for pair in self.pairs if pair.basis == BASIS_UNKNOWN),
            "recorded_matches": sum(
                1 for pair in self.pairs if pair.basis == BASIS_RECORDED
            ),
            "degraded": self.degraded,
            "degraded_reason": self.degraded_reason,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary(),
            "pairs": [pair.as_dict() for pair in self.pairs],
        }


def _keys_for(step: ExecutionStep) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """The three key tiers for one step, strongest first; None where unusable.

    `child_call` is part of every tier: a span-less inner hop and a top-level
    dispatch of the same command are different rows in the ledger and must not
    be matched to each other, or a wrapper on one side would silently absorb a
    real call on the other.
    """
    name = step.command_name
    if name is None:
        return (None, None, None)
    marker = "child" if step.child_call else "root"
    context = step.context or ""
    weak = _digest(marker, name)
    medium = _digest(marker, name, context)
    strong = (
        _digest(marker, name, context, step.parameters_digest)
        if step.parameters_digest is not None
        else None
    )
    return (strong, medium, weak)


def _lcs_pairs(
    left: Sequence[int],
    right: Sequence[int],
    left_keys: Sequence[Optional[str]],
    right_keys: Sequence[Optional[str]],
) -> list[tuple[int, int]]:
    """Longest common subsequence over key equality, ties to the earlier index.

    Order-preserving and content-based: nothing here matches by list position,
    and inserted, removed and repeated calls fall out as gaps rather than
    being paired up to make the lists the same length.
    """
    n, m = len(left), len(right)
    if n == 0 or m == 0:
        return []
    table = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        ki = left_keys[left[i]]
        row = table[i]
        nxt = table[i + 1]
        for j in range(m - 1, -1, -1):
            if ki is not None and ki == right_keys[right[j]]:
                row[j] = nxt[j + 1] + 1
            else:
                row[j] = nxt[j] if nxt[j] >= row[j + 1] else row[j + 1]
    pairs: list[tuple[int, int]] = []
    i = j = 0
    while i < n and j < m:
        ki = left_keys[left[i]]
        if ki is not None and ki == right_keys[right[j]]:
            pairs.append((left[i], right[j]))
            i += 1
            j += 1
        elif table[i + 1][j] >= table[i][j + 1]:
            i += 1
        else:
            j += 1
    return pairs


def _greedy_pairs(
    left: Sequence[int],
    right: Sequence[int],
    left_keys: Sequence[Optional[str]],
    right_keys: Sequence[Optional[str]],
) -> list[tuple[int, int]]:
    """Order-preserving first-fit, used only when the LCS would be too large."""
    pairs: list[tuple[int, int]] = []
    cursor = 0
    for li in left:
        key = left_keys[li]
        if key is None:
            continue
        for offset in range(cursor, len(right)):
            if right_keys[right[offset]] == key:
                pairs.append((li, right[offset]))
                cursor = offset + 1
                break
    return pairs


def _align_segment(
    left_idx: list[int],
    right_idx: list[int],
    left_steps: Sequence[ExecutionStep],
    right_steps: Sequence[ExecutionStep],
) -> tuple[list[tuple[int, int, str, bool, Optional[str]]], bool]:
    """Align one gap by descending key strength; returns pairs and a degraded flag.

    Three passes, strongest key first, each one only over what the previous
    left unmatched. A pass that matches on a weaker key says so in its basis,
    so "these two are the same call" and "these two are both `add_todo`" are
    never reported as the same claim.
    """
    matched: list[tuple[int, int, str, bool, Optional[str]]] = []
    remaining_left = list(left_idx)
    remaining_right = list(right_idx)
    degraded = False
    tiers = (
        (0, BASIS_COMMAND_CONTEXT_PARAMETERS),
        (1, BASIS_COMMAND_CONTEXT),
        (2, BASIS_COMMAND),
    )
    left_keys_all = [_keys_for(step) for step in left_steps]
    right_keys_all = [_keys_for(step) for step in right_steps]
    for tier, basis in tiers:
        if not remaining_left or not remaining_right:
            break
        left_keys = [keys[tier] for keys in left_keys_all]
        right_keys = [keys[tier] for keys in right_keys_all]
        if len(remaining_left) * len(remaining_right) > _MAX_ALIGNMENT_CELLS:
            degraded = True
            found = _greedy_pairs(remaining_left, remaining_right, left_keys, right_keys)
        else:
            found = _lcs_pairs(remaining_left, remaining_right, left_keys, right_keys)
        if not found:
            continue
        # A key that occurs more than once on either side inside this segment
        # cannot distinguish which repetition is which. The pair still shows,
        # labelled, rather than being withheld or silently asserted.
        left_counts: dict[str, int] = {}
        for index in remaining_left:
            key = left_keys[index]
            if key is not None:
                left_counts[key] = left_counts.get(key, 0) + 1
        right_counts: dict[str, int] = {}
        for index in remaining_right:
            key = right_keys[index]
            if key is not None:
                right_counts[key] = right_counts.get(key, 0) + 1
        for li, ri in found:
            key = left_keys[li]
            repeated = (
                key is not None
                and (left_counts.get(key, 0) > 1 or right_counts.get(key, 0) > 1)
            )
            weak = basis == BASIS_COMMAND
            reason = None
            if repeated:
                reason = "repeated-key"
            elif weak:
                reason = "command-name-only"
            matched.append((li, ri, basis, repeated or weak, reason))
        paired_left = {li for li, _, _, _, _ in matched}
        paired_right = {ri for _, ri, _, _, _ in matched}
        remaining_left = [i for i in remaining_left if i not in paired_left]
        remaining_right = [i for i in remaining_right if i not in paired_right]
    return matched, degraded


def align_steps(
    left_steps: Sequence[ExecutionStep],
    right_steps: Sequence[ExecutionStep],
    *,
    recorded_alignment: Optional[Iterable[Mapping[str, Any]]] = None,
) -> Alignment:
    """The shared alignment every consumer reads -- UI, API and extractor alike.

    A supplied structured alignment is authoritative: its pairs are emitted with
    `basis="recorded"` and are never second-guessed. Steps it does not mention
    are then aligned deterministically between its anchors, so a partial
    alignment adds information instead of hiding the rest of the run.

    It is also VERIFIED against the projected steps: every entry must name a
    `command_call_id` present on its side, and one that does not is refused.
    Nothing in this repo writes such an alignment today -- there is no alignment
    table and `list_distillation_runs` is a stub -- so the input is external by
    definition, and dropping the entries that do not resolve would report a
    rejected alignment as an accepted one.

    With no supplied alignment the whole thing is one deterministic segment,
    which is what every caller gets today. Either way the result is a function
    of the recorded steps alone, so a browser and an agent reading the same
    evidence get the same pairs and neither recomputes a diff of its own.
    """
    recorded_pairs: list[tuple[int, int]] = []
    out_of_order: list[tuple[int, int]] = []
    if recorded_alignment is not None:
        left_by_call = {step.command_call_id: i for i, step in enumerate(left_steps)}
        right_by_call = {step.command_call_id: i for i, step in enumerate(right_steps)}
        raw: list[tuple[int, int]] = []
        for entry in recorded_alignment:
            if not isinstance(entry, Mapping):
                raise InvalidRecordedAlignment(
                    f"an alignment entry must be an object, not {entry!r}"
                )
            left_call = _text_or_none(entry.get("left_command_call_id"))
            right_call = _text_or_none(entry.get("right_command_call_id"))
            li = left_by_call.get(left_call)
            ri = right_by_call.get(right_call)
            if li is None or ri is None:
                unknown = [
                    f"left_command_call_id={left_call!r}" if li is None else None,
                    f"right_command_call_id={right_call!r}" if ri is None else None,
                ]
                raise InvalidRecordedAlignment(
                    "alignment entry names "
                    + " and ".join(part for part in unknown if part)
                    + ", which the projected steps do not contain"
                )
            raw.append((li, ri))
        raw.sort()
        seen_left: set[int] = set()
        seen_right: set[int] = set()
        last_right = -1
        for li, ri in raw:
            if li in seen_left or ri in seen_right:
                continue
            seen_left.add(li)
            seen_right.add(ri)
            if ri > last_right:
                recorded_pairs.append((li, ri))
                last_right = ri
            else:
                # A stored alignment that crosses itself cannot also order the
                # gaps around it. The pair is kept -- it is recorded evidence --
                # but it stops being an anchor.
                out_of_order.append((li, ri))

    anchored_left = {li for li, _ in recorded_pairs} | {li for li, _ in out_of_order}
    anchored_right = {ri for _, ri in recorded_pairs} | {ri for _, ri in out_of_order}

    pairs: list[AlignedPair] = []
    degraded = False

    def emit_segment(l_start: int, l_end: int, r_start: int, r_end: int) -> None:
        nonlocal degraded
        left_idx = [
            i for i in range(l_start, l_end) if i not in anchored_left
        ]
        right_idx = [
            j for j in range(r_start, r_end) if j not in anchored_right
        ]
        matched, seg_degraded = _align_segment(
            left_idx, right_idx, left_steps, right_steps
        )
        degraded = degraded or seg_degraded
        by_left = {li: (ri, basis, amb, reason) for li, ri, basis, amb, reason in matched}
        by_right = {ri: li for li, ri, _, _, _ in matched}
        li_cursor = 0
        ri_cursor = 0
        left_list = left_idx
        right_list = right_idx
        while li_cursor < len(left_list) or ri_cursor < len(right_list):
            if li_cursor < len(left_list):
                li = left_list[li_cursor]
                if li in by_left:
                    ri, basis, amb, reason = by_left[li]
                    # Everything on the right before this partner is an
                    # insertion, and is emitted before the pair so the reader
                    # sees it where it happened.
                    while ri_cursor < len(right_list) and right_list[ri_cursor] != ri:
                        rj = right_list[ri_cursor]
                        if rj not in by_right:
                            pairs.append(_unmatched(right_steps[rj], PAIR_RIGHT_ONLY))
                        ri_cursor += 1
                    pairs.append(
                        AlignedPair(
                            kind=PAIR_MATCHED,
                            basis=basis,
                            left=left_steps[li],
                            right=right_steps[ri],
                            ambiguous=amb,
                            ambiguity_reason=reason,
                        )
                    )
                    li_cursor += 1
                    ri_cursor += 1
                    continue
                pairs.append(_unmatched(left_steps[li], PAIR_LEFT_ONLY))
                li_cursor += 1
                continue
            rj = right_list[ri_cursor]
            if rj not in by_right:
                pairs.append(_unmatched(right_steps[rj], PAIR_RIGHT_ONLY))
            ri_cursor += 1

    previous_left = 0
    previous_right = 0
    for li, ri in recorded_pairs:
        emit_segment(previous_left, li, previous_right, ri)
        pairs.append(
            AlignedPair(
                kind=PAIR_MATCHED,
                basis=BASIS_RECORDED,
                left=left_steps[li],
                right=right_steps[ri],
            )
        )
        previous_left = li + 1
        previous_right = ri + 1
    emit_segment(previous_left, len(left_steps), previous_right, len(right_steps))

    for li, ri in out_of_order:
        pairs.append(
            AlignedPair(
                kind=PAIR_MATCHED,
                basis=BASIS_RECORDED,
                left=left_steps[li],
                right=right_steps[ri],
                ambiguous=True,
                ambiguity_reason="recorded-alignment-out-of-order",
            )
        )

    return Alignment(
        pairs=tuple(pairs),
        degraded=degraded,
        degraded_reason=(
            "step count exceeded the quadratic alignment budget; matches are "
            "order-preserving first-fit"
            if degraded
            else None
        ),
    )


def _unmatched(step: ExecutionStep, kind: str) -> AlignedPair:
    """An unmatched step, saying whether it had a usable key at all."""
    strong, _, weak = _keys_for(step)
    unknown = weak is None
    return AlignedPair(
        kind=kind,
        basis=BASIS_UNKNOWN if unknown else BASIS_UNMATCHED,
        left=step if kind == PAIR_LEFT_ONLY else None,
        right=step if kind == PAIR_RIGHT_ONLY else None,
        ambiguous=unknown,
        ambiguity_reason="no-recorded-command-name" if unknown else None,
    )


# ----------------------------------------------------------------------
# Evidence anchors
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class EvidenceAnchor:
    """Where a comment attaches, in the vocabulary the store already accepts.

    `target_kind` and `span_ids` are exactly what
    `ObservabilityStore.add_human_feedback` validates, so a feedback worker
    can pass these straight through without translating. This module defines
    the anchor and does NOT record feedback: storage, the three categories and
    their subcategories belong to the feedback slice.

    `anchorable` is False for a dispatch the trace has no span for (a span-less
    inner hop). The store requires at least one recorded span for anything
    finer than turn scope, so such a step can only be commented on at
    `fallback_target_kind` -- which is a real limitation of the evidence, worth
    showing rather than working around.
    """

    ref_id: str
    store_id: str
    turn_key: str
    target_kind: str
    span_ids: tuple[str, ...]
    command_call_id: Optional[str] = None
    step_position: Optional[int] = None
    pass_id: Optional[str] = None

    @property
    def anchorable(self) -> bool:
        return self.target_kind == "turn" or bool(self.span_ids)

    @property
    def fallback_target_kind(self) -> str:
        return "turn"

    def as_dict(self) -> dict[str, Any]:
        return {
            "ref_id": self.ref_id,
            "store_id": self.store_id,
            "turn_key": self.turn_key,
            "target_kind": self.target_kind,
            "span_ids": list(self.span_ids),
            "command_call_id": self.command_call_id,
            "step_position": self.step_position,
            "pass_id": self.pass_id,
            "anchorable": self.anchorable,
        }


def anchor_for_step(ref: ExecutionRef, step: ExecutionStep) -> EvidenceAnchor:
    return EvidenceAnchor(
        ref_id=ref.ref_id(),
        store_id=ref.store_id,
        turn_key=step.turn_key,
        target_kind="step" if step.span_id else "turn",
        span_ids=(step.span_id,) if step.span_id else (),
        command_call_id=step.command_call_id,
        step_position=step.position,
        pass_id=step.pass_id,
    )


def anchor_for_turn(ref: ExecutionRef, turn: TurnProjection) -> EvidenceAnchor:
    return EvidenceAnchor(
        ref_id=ref.ref_id(),
        store_id=ref.store_id,
        turn_key=turn.turn_key,
        target_kind="turn",
        span_ids=(),
        pass_id=ref.pass_id,
    )


def anchors_for_pair(
    comparison: "ExecutionComparison", pair: AlignedPair
) -> dict[str, Any]:
    """Anchors for one comparison row: one side, the other, or both.

    A comment on a matched pair is about both executions, which is why both
    anchors are returned rather than a single merged one -- the two sides live
    in different turns and possibly different stores, and the store anchors
    feedback per turn.
    """
    return {
        "left": (
            anchor_for_step(comparison.left.ref, pair.left).as_dict()
            if pair.left
            else None
        ),
        "right": (
            anchor_for_step(comparison.right.ref, pair.right).as_dict()
            if pair.right
            else None
        ),
    }


# ----------------------------------------------------------------------
# The comparison
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ExecutionComparison:
    """Two projections and the alignment between them.

    `left` is the pinned side -- the winner, the selected best run, the
    teacher. That is a caller's choice of viewpoint and carries no claim that
    the left side is correct.
    """

    left: ExecutionProjection
    right: ExecutionProjection
    alignment: Alignment

    def summary(self) -> dict[str, Any]:
        summary = dict(self.alignment.summary())
        summary.update(
            {
                "left_readable": self.left.readable,
                "right_readable": self.right.readable,
                "left_steps": len(self.left.steps),
                "right_steps": len(self.right.steps),
                "left_unavailable": len(self.left.unavailable),
                "right_unavailable": len(self.right.unavailable),
                "left_turns": len(self.left.turns),
                "right_turns": len(self.right.turns),
            }
        )
        return summary

    def differences(self) -> list[AlignedPair]:
        """The rows a "Differences only" view shows.

        A matched pair counts as a difference when the recorded outcome or the
        recorded parameters differ; an unmatched or ambiguous row always does.
        """
        rows = []
        for pair in self.alignment.pairs:
            if pair.kind != PAIR_MATCHED or pair.ambiguous:
                rows.append(pair)
                continue
            left, right = pair.left, pair.right
            if left is None or right is None:  # pragma: no cover - matched has both
                rows.append(pair)
                continue
            if (
                left.status != right.status
                or left.success != right.success
                or left.parameters_digest != right.parameters_digest
            ):
                rows.append(pair)
        return rows

    def as_dict(self) -> dict[str, Any]:
        return {
            "left": self.left.as_dict(),
            "right": self.right.as_dict(),
            "alignment": self.alignment.as_dict(),
            "summary": self.summary(),
            "review_pair_key": review_pair_key(self.left.ref, self.right.ref),
        }


def compare_executions(
    left: ExecutionRef,
    right: ExecutionRef,
    reader: ExecutionReader,
    *,
    ledger: Optional[LedgerProjection] = None,
    cost_rollup: Optional[CostRollup] = None,
    left_pass: Optional[PassSelector] = None,
    right_pass: Optional[PassSelector] = None,
    recorded_alignment: Optional[Iterable[Mapping[str, Any]]] = None,
) -> ExecutionComparison:
    """Project both sides and align them. Read-only; records nothing.

    Both sides may be unreadable, one side may be, or both may be scoreless --
    none of that is refused. Comparing an execution with itself is legal and
    useful (it is how a reviewer confirms the alignment is behaving).
    """
    left_projection = project_execution(
        left,
        reader,
        ledger=ledger,
        cost_rollup=cost_rollup,
        pass_selector=left_pass,
    )
    right_projection = project_execution(
        right,
        reader,
        ledger=ledger,
        cost_rollup=cost_rollup,
        pass_selector=right_pass,
    )
    alignment = align_steps(
        left_projection.steps,
        right_projection.steps,
        recorded_alignment=recorded_alignment,
    )
    return ExecutionComparison(
        left=left_projection, right=right_projection, alignment=alignment
    )


def comparison_digest(comparison: ExecutionComparison) -> dict[str, Any]:
    """Metadata, ids and counts only -- no user text, answers or parameters.

    For probes and logs against corpora whose content must not be printed.
    Everything here is either a count, a boolean, an opaque digest or an
    identifier the store itself already treats as non-content.
    """

    def side(projection: ExecutionProjection) -> dict[str, Any]:
        return {
            "ref_id": projection.ref.ref_id(),
            "store_id": projection.ref.store_id,
            "turn_count": len(projection.ref.turn_keys),
            "experiment_id": projection.ref.experiment_id,
            "task_id": projection.ref.task_id,
            "attempt": projection.ref.attempt,
            "pass_id": projection.ref.pass_id,
            "turns_read": len(projection.turns),
            "unavailable": len(projection.unavailable),
            "steps": len(projection.steps),
            "unassigned_steps": len(projection.unassigned_steps),
            "steps_with_parameters": sum(
                1 for step in projection.steps if step.parameters_digest is not None
            ),
            "steps_without_span": sum(
                1 for step in projection.steps if not step.span_recorded
            ),
            "child_call_steps": sum(1 for step in projection.steps if step.child_call),
            "artifacts": len(projection.artifacts),
            "artifacts_offloaded": sum(
                1 for artifact in projection.artifacts if artifact.artifact_id
            ),
            "unattributed_artifacts": len(projection.unattributed_artifacts),
            "content_attribution": projection.content_attribution,
            "answers_recorded": sum(
                1 for turn in projection.turns if turn.answer is not None
            ),
            "timing": dict(projection.timing),
            "cost": dict(projection.cost),
            # Counts and coverage only: the per-call anchors are span ids,
            # which this digest may carry, but they live on the turns and
            # repeating them here would make a log of a corpus much larger
            # without saying anything the counts do not.
            "usage": dict(projection.usage),
        }

    return {
        "left": side(comparison.left),
        "right": side(comparison.right),
        "summary": comparison.summary(),
        "differences": len(comparison.differences()),
        "review_pair_key": review_pair_key(
            comparison.left.ref, comparison.right.ref
        ),
    }


__all__ = [
    "ATTRIBUTION_PASS",
    "ATTRIBUTION_SHARED",
    "ATTRIBUTION_TURN",
    "ATTRIBUTION_UNATTRIBUTED",
    "CACHE_HIT",
    "CACHE_MISS",
    "CACHE_UNKNOWN",
    "COVERAGE_COMPLETE",
    "COVERAGE_NONE",
    "COVERAGE_PARTIAL",
    "BASIS_COMMAND",
    "BASIS_COMMAND_CONTEXT",
    "BASIS_COMMAND_CONTEXT_PARAMETERS",
    "BASIS_RECORDED",
    "BASIS_UNKNOWN",
    "BASIS_UNMATCHED",
    "PAIR_LEFT_ONLY",
    "PAIR_MATCHED",
    "PAIR_RIGHT_ONLY",
    "SPAN_DISTILLATION_PASS",
    "SPAN_LLM_CALL",
    "USAGE_COMPLETE",
    "USAGE_PARTIAL",
    "USAGE_SHARED_RESPONSE",
    "USAGE_UNRECORDED",
    "AlignedPair",
    "Alignment",
    "ArtifactRef",
    "ComparisonError",
    "EvidenceAnchor",
    "ExecutionComparison",
    "ExecutionProjection",
    "ExecutionReader",
    "ExecutionRef",
    "ExecutionScopeMismatch",
    "ExecutionStep",
    "InvalidExecutionRef",
    "InvalidRecordedAlignment",
    "LlmCallUsage",
    "PassSelector",
    "StoreExecutionReader",
    "TurnProjection",
    "UnknownRecordedPass",
    "WorkspaceExecutionReader",
    "align_steps",
    "anchor_for_step",
    "anchor_for_turn",
    "anchors_for_pair",
    "canonical_llm_spans",
    "compare_executions",
    "comparison_digest",
    "cost_rollup",
    "default_cost_rollup",
    "default_ledger_projection",
    "discover_pass_selectors",
    "merge_usage_rollups",
    "project_execution",
    "review_pair_key",
    "usage_rollup",
]

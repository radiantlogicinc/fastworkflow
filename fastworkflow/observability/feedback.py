"""The recorded-feedback taxonomy, its anchors, and the consolidated task read.

One write surface and one vocabulary for everybody. A human typing in the
Observability UI and a coding agent posting through the HTTP API record the
SAME row, with the same category, the same subcategory, the same provenance and
the same evidence anchors. The UI's selectors and watermark prompts exist to
help a person choose; they are not a second schema, and nothing here reads a
comment's text to decide what the comment is. Category and subcategory arrive
as enums or the write is refused.

THE OWNER-CONFIRMED TAXONOMY (three categories, two subcategories each):

    observations_analysis  "Observations / Analysis"
        observation        what was seen, quoted rather than judged
        analysis           what the observation is evidence of
    conclusions            "Conclusions"
        what_went_right    behavior worth keeping
        what_went_wrong    behavior that failed
    recommendations        "Recommendations"
        what_to_do         the change to make
        what_not_to_do     the change to avoid

Comment TEXT is free-form for both authors. An agent may structure its comment
however it likes; that structure is content, never metadata, and no heading in
it is parsed. The store's older three-heading composer ("What went wrong:" /
"What worked:" / "What should change:") was exactly such a parser and is gone:
it is not this taxonomy, and re-deriving one of these six subcategories from a
legacy heading would be a guess recorded as if it were the author's choice.
Comments written before the taxonomy therefore read back with
``category``/``subcategory`` of None and ``classified`` False -- visibly
unclassified, text untouched.

ANCHORS. A comment names one target execution and optionally a second one, for
a comparison. Both sides are `comparison.ExecutionRef` -- the same scoped
reference the compare view and pair review use, so a comment written from a
comparison row anchors to the exact pair that was on screen. Every reference is
validated against the evidence of ITS OWN store: the turn must be there, a
declared experiment/task/attempt must match what the turn row records
(`comparison._check_scope`), the named spans must belong to that turn, and a
named pass must resolve against recorded spans. The store a reference names
must be handed in explicitly by the caller, so a reference cannot reach
evidence the caller was not authorized to read, and a label is never taken as
proof of anything.

Anchors are frozen into the row at write time. Re-picking an experiment winner
or a task's best run afterwards does not retarget a comment: the pair it names
is the pair that existed when somebody wrote it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence

from fastworkflow.observability.comparison import (
    EvidenceAnchor,
    ExecutionRef,
    InvalidExecutionRef,
    PassSelector,
    # Imported rather than re-derived on purpose. A second implementation of
    # "does this reference's declared scope match the evidence" is precisely
    # the drift this shared-reference design exists to prevent, and the span
    # tree is what `PassSelector.resolve_against` expects.
    _check_scope,
    _SpanTree,
    review_pair_key,
)

# ----------------------------------------------------------------------
# The taxonomy
# ----------------------------------------------------------------------

CATEGORY_OBSERVATIONS_ANALYSIS = "observations_analysis"
CATEGORY_CONCLUSIONS = "conclusions"
CATEGORY_RECOMMENDATIONS = "recommendations"

SUBCATEGORY_OBSERVATION = "observation"
SUBCATEGORY_ANALYSIS = "analysis"
SUBCATEGORY_WHAT_WENT_RIGHT = "what_went_right"
SUBCATEGORY_WHAT_WENT_WRONG = "what_went_wrong"
SUBCATEGORY_WHAT_TO_DO = "what_to_do"
SUBCATEGORY_WHAT_NOT_TO_DO = "what_not_to_do"


@dataclass(frozen=True)
class Subcategory:
    """One subcategory: its enum value, its UI label, and its prompt.

    `watermark` is placeholder text in the composer. It is guidance for a
    person, not a template: nothing validates a comment against it and nothing
    reads it back out.
    """

    value: str
    label: str
    watermark: str

    def as_dict(self) -> dict[str, str]:
        return {"value": self.value, "label": self.label, "watermark": self.watermark}


@dataclass(frozen=True)
class Category:
    value: str
    label: str
    subcategories: tuple[Subcategory, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "label": self.label,
            "subcategories": [sub.as_dict() for sub in self.subcategories],
        }


FEEDBACK_TAXONOMY: tuple[Category, ...] = (
    Category(
        value=CATEGORY_OBSERVATIONS_ANALYSIS,
        label="Observations / Analysis",
        subcategories=(
            Subcategory(
                value=SUBCATEGORY_OBSERVATION,
                label="Observation",
                watermark=(
                    "What did you see? Quote or point at the evidence — the "
                    "request, the answer, the step — without judging it yet."
                ),
            ),
            Subcategory(
                value=SUBCATEGORY_ANALYSIS,
                label="Analysis",
                watermark=(
                    "What do you think the observation means? You do not need "
                    "to diagnose the implementation to say what it points at."
                ),
            ),
        ),
    ),
    Category(
        value=CATEGORY_CONCLUSIONS,
        label="Conclusions",
        subcategories=(
            Subcategory(
                value=SUBCATEGORY_WHAT_WENT_RIGHT,
                label="What went right",
                watermark=(
                    "What is worth preserving? Justified clarifying questions "
                    "and honest reporting of incomplete work count."
                ),
            ),
            Subcategory(
                value=SUBCATEGORY_WHAT_WENT_WRONG,
                label="What went wrong",
                watermark=(
                    "What was missing or incorrect? Anchor it to the first "
                    "point where behavior went wrong, not only to the answer."
                ),
            ),
        ),
    ),
    Category(
        value=CATEGORY_RECOMMENDATIONS,
        label="Recommendations",
        subcategories=(
            Subcategory(
                value=SUBCATEGORY_WHAT_TO_DO,
                label="What to do",
                watermark=(
                    "What should happen instead, next time? State the expected "
                    "behavior rather than the code change."
                ),
            ),
            Subcategory(
                value=SUBCATEGORY_WHAT_NOT_TO_DO,
                label="What not to do",
                watermark=(
                    "What should be avoided? Name the tempting fix that would "
                    "make things worse, and why."
                ),
            ),
        ),
    ),
)

FEEDBACK_CATEGORIES: dict[str, Category] = {
    category.value: category for category in FEEDBACK_TAXONOMY
}
FEEDBACK_SUBCATEGORIES: dict[str, tuple[str, ...]] = {
    category.value: tuple(sub.value for sub in category.subcategories)
    for category in FEEDBACK_TAXONOMY
}

# What the store accepts as a target. `task` is a whole-task summary comment:
# ordinary feedback with no span, anchored to a real recorded turn of the task
# so that it still has evidence behind it, and surfaced in the task view under
# its own heading rather than in a separate summary schema.
FEEDBACK_TARGET_KINDS = ("turn", "phase", "step", "span", "task")


class FeedbackError(ValueError):
    """A feedback write does not describe recorded evidence."""


def taxonomy_payload() -> dict[str, Any]:
    """The wire shape of the taxonomy: one source for the UI and for agents."""
    return {
        "categories": [category.as_dict() for category in FEEDBACK_TAXONOMY],
        "target_kinds": list(FEEDBACK_TARGET_KINDS),
    }


def validate_category(category: Any, subcategory: Any) -> tuple[str, str]:
    """Return the validated pair, or refuse.

    Pairing is checked, not just membership: `conclusions` + `observation` is
    two real enum values that do not go together, and accepting it would make
    the category filter lie about what it is showing.
    """
    if not isinstance(category, str) or category not in FEEDBACK_CATEGORIES:
        raise FeedbackError(
            "category must be one of " + ", ".join(sorted(FEEDBACK_CATEGORIES))
        )
    allowed = FEEDBACK_SUBCATEGORIES[category]
    if not isinstance(subcategory, str) or subcategory not in allowed:
        raise FeedbackError(
            f"subcategory for {category} must be one of " + ", ".join(allowed)
        )
    return category, subcategory


def normalize_note(
    *,
    target_kind: Any,
    span_ids: Any,
    target_label: Any,
    provenance: Any,
    comment: Any,
    category: Any,
    subcategory: Any,
) -> dict[str, Any]:
    """Check and normalize one note's own fields before it is stored.

    Evidence questions — does the turn exist, do the spans belong to it — are
    NOT asked here: only the caller holding the evidence can answer those.
    """
    from fastworkflow.observability.store import (
        FEEDBACK_COMMENT_MAX_CHARS,
        FEEDBACK_PROVENANCES,
    )

    if target_kind not in FEEDBACK_TARGET_KINDS:
        raise ValueError("invalid feedback target kind")
    if (not isinstance(span_ids, list) or len(span_ids) > 10000
            or any(not isinstance(v, str) or not v for v in span_ids)):
        raise ValueError("span_ids must be a list of recorded span IDs")
    ids = sorted(set(span_ids))
    whole = target_kind in ("turn", "task")
    if (whole and ids) or (not whole and not ids):
        raise ValueError("component feedback requires spans; turn feedback has none")
    if not isinstance(comment, str) or not comment.strip():
        raise ValueError(
            f"feedback must contain text (at most {FEEDBACK_COMMENT_MAX_CHARS} characters)"
        )
    comment = comment.strip()
    if len(comment) > FEEDBACK_COMMENT_MAX_CHARS:
        raise ValueError(
            f"feedback must contain text (at most {FEEDBACK_COMMENT_MAX_CHARS} characters)"
        )
    if not isinstance(target_label, str) or not target_label or len(target_label) > 1000:
        raise ValueError("target_label is required (at most 1000 characters)")
    if provenance not in FEEDBACK_PROVENANCES:
        raise ValueError(
            "provenance must be human, coding_agent, or distillation_agent"
        )
    category, subcategory = validate_category(category, subcategory)
    return {
        "target_kind": target_kind,
        "span_ids": ids,
        "target_label": target_label,
        "provenance": provenance,
        "comment": comment,
        "category": category,
        "subcategory": subcategory,
    }


def category_label(category: Optional[str]) -> Optional[str]:
    found = FEEDBACK_CATEGORIES.get(category or "")
    return found.label if found else None


def subcategory_label(category: Optional[str], subcategory: Optional[str]) -> Optional[str]:
    found = FEEDBACK_CATEGORIES.get(category or "")
    if found is None:
        return None
    for sub in found.subcategories:
        if sub.value == subcategory:
            return sub.label
    return None


# ----------------------------------------------------------------------
# Anchors
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class FeedbackTarget:
    """One side of a comment's anchor: a scoped execution and where in it.

    `ref` carries store, turns, experiment/task/attempt and pass; the remaining
    fields are exactly what the store's row already holds, so the primary
    target needs no translation on the way in or out.

    `ref` IS NOT NARROWED to the anchored turn, and `turn_key` is a field of
    its own. A comment attaches to one place, but the execution it is about
    may be a whole multi-turn attempt, and the two facts are different:

    - The store's `turn_key` column, the deep link and the span check all use
      `turn_key` — one place, as before.
    - `ref` stays the reference the reader was actually looking at, so
      `ref_id()` and therefore `FeedbackAnchors.pair_key` agree with
      `comparison.review_pair_key` and with the pair's review progress in
      `pair_review`. Narrowing here used to make a two-turn pair hash to a
      third thing that named neither side's real scope.

    Every turn in `ref` is checked against the evidence by `validate_target`,
    so the wider scope is recorded evidence rather than an assertion.
    """

    ref: ExecutionRef
    target_kind: str
    span_ids: tuple[str, ...]
    target_label: str
    turn_key: str = ""

    def __post_init__(self) -> None:
        turn_key = str(self.turn_key or "").strip()
        if not turn_key:
            if len(self.ref.turn_keys) != 1:
                raise FeedbackError(
                    "a reference spanning "
                    f"{len(self.ref.turn_keys)} turns must say which one the "
                    "comment is anchored to"
                )
            turn_key = self.ref.turn_keys[0]
        if turn_key not in self.ref.turn_keys:
            raise FeedbackError(
                f"anchored turn {turn_key!r} is not one of the turns the "
                "reference names"
            )
        object.__setattr__(self, "turn_key", turn_key)
        if self.target_kind not in FEEDBACK_TARGET_KINDS:
            raise FeedbackError(
                "target_kind must be one of " + ", ".join(FEEDBACK_TARGET_KINDS)
            )
        ids = tuple(sorted({str(v) for v in self.span_ids}))
        if any(not v for v in ids) or len(ids) > 10000:
            raise FeedbackError("span_ids must be a list of recorded span IDs")
        # `turn` and `task` are whole-execution scopes and carry no span;
        # anything finer must point at recorded evidence or it is a label.
        if self.target_kind in ("turn", "task"):
            if ids:
                raise FeedbackError(
                    f"{self.target_kind} feedback has no spans; it is anchored "
                    "to the whole recorded execution"
                )
        elif not ids:
            raise FeedbackError("component feedback requires spans")
        label = self.target_label
        if not isinstance(label, str) or not label.strip() or len(label) > 1000:
            raise FeedbackError("target_label is required (at most 1000 characters)")
        object.__setattr__(self, "span_ids", ids)
        object.__setattr__(self, "target_label", label.strip())

    @property
    def store_id(self) -> str:
        return self.ref.store_id

    def as_dict(self) -> dict[str, Any]:
        return {
            "ref": self.ref.as_dict(),
            "turn_key": self.turn_key,
            "target_kind": self.target_kind,
            "span_ids": list(self.span_ids),
            "target_label": self.target_label,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FeedbackTarget":
        """Parse the wire shape the UI and coding agents both post.

        The reference fields may be inline (`store_id`, `turn_key`,
        `experiment_id`, ...) or nested under `ref`, because a caller coming
        from a comparison row already holds a whole `ExecutionRef` dict.

        When `ref` is nested and names more than one turn, the anchored turn
        is a separate field. `anchor_turn_key` is the unambiguous spelling and
        is preferred in new callers; plain `turn_key` still means the same
        thing, because inline references have exactly one turn and the two
        readings cannot disagree there. Stored anchors keep emitting
        `turn_key`, so a row posted back verbatim round-trips.
        """
        if not isinstance(value, Mapping):
            raise FeedbackError("a feedback target must be an object")
        raw_ref = value.get("ref") if isinstance(value.get("ref"), Mapping) else value
        try:
            ref = ExecutionRef.from_mapping(raw_ref)
        except InvalidExecutionRef as exc:
            raise FeedbackError(str(exc)) from exc
        span_ids = value.get("span_ids") or []
        if not isinstance(span_ids, (list, tuple)):
            raise FeedbackError("span_ids must be a list of recorded span IDs")
        if any(not isinstance(v, str) for v in span_ids):
            raise FeedbackError("span_ids must be a list of recorded span IDs")
        return cls(
            ref=ref,
            turn_key=str(
                value.get("anchor_turn_key")
                or value.get("turn_key")
                or value.get("logical_turn_key")
                or ""
            ),
            target_kind=str(value.get("target_kind") or ""),
            span_ids=tuple(span_ids),
            target_label=str(value.get("target_label") or ""),
        )

    @classmethod
    def from_evidence_anchor(
        cls, ref: ExecutionRef, anchor: EvidenceAnchor, target_label: str
    ) -> "FeedbackTarget":
        """Adapt a comparison row's anchor without restating it.

        `comparison.anchor_for_step` / `anchor_for_turn` already produce the
        store's vocabulary, so the compare view can hand a row straight to the
        writer. The reference is passed through UNCHANGED and the anchor's
        turn is recorded beside it: a remark written on one step of a two-turn
        attempt is about that step, within that attempt, and a pair of such
        remarks keys to the same pair the compare view and pair review named.
        """
        return cls(
            ref=ref,
            turn_key=anchor.turn_key,
            target_kind=anchor.target_kind,
            span_ids=anchor.span_ids,
            target_label=target_label,
        )


@dataclass(frozen=True)
class FeedbackAnchors:
    """A validated primary target and, for a comparison, its counterpart.

    `pair_key` is `comparison.review_pair_key` over the two references exactly
    as the reader held them, so a comment on a pair and the pair's review
    progress in `pair_review` agree on what "this pair" means without either
    owning the other. That is why `FeedbackTarget` keeps the whole reference
    and names the anchored turn separately: two remarks written on different
    steps of the same two multi-turn attempts belong to ONE pair, and they do
    only if the key is hashed from the attempts rather than from the steps.
    """

    primary: FeedbackTarget
    paired: Optional[FeedbackTarget] = None

    @property
    def pair_key(self) -> Optional[str]:
        if self.paired is None:
            return None
        return review_pair_key(self.primary.ref, self.paired.ref)

    def as_dict(self) -> dict[str, Any]:
        return {
            "primary": self.primary.as_dict(),
            "paired": self.paired.as_dict() if self.paired else None,
            "pair_key": self.pair_key,
        }


def human_feedback_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """One stored comment in wire shape."""
    value = dict(row)
    value["span_ids"] = json.loads(value.pop("span_ids_json"))
    raw_anchors = value.pop("anchors_json", None)
    anchors: Any = None
    if isinstance(raw_anchors, str) and raw_anchors:
        try:
            anchors = json.loads(raw_anchors)
        except ValueError:
            anchors = None
    value["anchors"] = anchors
    paired = anchors.get("paired") if isinstance(anchors, Mapping) else None
    value["paired"] = paired
    value["pair_key"] = (
        anchors.get("pair_key") if isinstance(anchors, Mapping) else None
    )
    return value


def scrubbed_anchor_dict(anchors: FeedbackAnchors, scrub: Optional[Any] = None) -> dict[str, Any]:
    """The frozen anchor as it is STORED: identity exact, labels scrubbed.

    The anchor repeats presentation text the row already has a column for —
    `target_label` on each side, and `ExecutionRef.label` — and those columns
    are credential-scrubbed on the way in. Serializing the anchor unscrubbed
    put a credential somebody had pasted into a component label back into the
    same row as JSON, defeating the scrub next to the value it had just
    cleaned. Every label in the anchor goes through the same redactor.

    Nothing that establishes IDENTITY is touched: `store_id`, `turn_keys`,
    `turn_key`, `experiment_id`, `task_id`, `attempt`, `pass_id`, `span_ids`,
    `ref_id` and `pair_key` are what the task view, the deep links and the
    pair lookup match on, and a redactor that rewrote one of them would
    quietly orphan the comment. `label` is excluded from `ref_id` by design
    (`comparison.ExecutionRef`), so scrubbing it cannot move the identity.
    """
    value = anchors.as_dict()
    if scrub is None:
        return value
    for side in ("primary", "paired"):
        target = value.get(side)
        if not isinstance(target, dict):
            continue
        if target.get("target_label") is not None:
            target["target_label"] = scrub(target["target_label"])
        ref = target.get("ref")
        if isinstance(ref, dict) and ref.get("label") is not None:
            ref["label"] = scrub(ref["label"])
    return value


def validate_target(
    target: FeedbackTarget,
    store: Any,
    *,
    pass_selector: Optional[PassSelector] = None,
) -> None:
    """Refuse unless `store` actually records what the target claims.

    Checked in the order a forged reference fails soonest: the store's own
    identity, then the turns, then the declared experiment/task/attempt, then
    the spans, then the pass. Nothing here trusts `target_label` or the
    caller's current selection.

    EVERY turn the reference names is checked, not only the anchored one. A
    reference spanning a whole attempt is part of the comment's frozen
    identity — it is what `pair_key` hashes — so an unrecorded turn inside it
    would be an unverified claim carried forward forever.
    """
    identity = None
    try:
        identity = store.store_identity()
    except Exception:  # pragma: no cover - a store without identity is refused below
        identity = None
    if identity is None or identity != target.store_id:
        raise FeedbackError(
            f"reference names store {target.store_id!r}, which is not the "
            "store it was authorized against"
        )
    row = None
    for turn_key in target.ref.turn_keys:
        recorded_turn = store.get_turn(turn_key)
        if recorded_turn is None:
            raise FeedbackError(
                f"turn {turn_key!r} is not recorded in store {target.store_id!r}"
            )
        try:
            _check_scope(target.ref, turn_key, recorded_turn)
        except InvalidExecutionRef as exc:
            raise FeedbackError(str(exc)) from exc
        if turn_key == target.turn_key:
            row = recorded_turn
    if row is None:  # pragma: no cover - __post_init__ keeps the anchor inside
        raise FeedbackError(
            f"turn {target.turn_key!r} is not recorded in store {target.store_id!r}"
        )
    if target.target_kind == "task" and not (
        target.ref.experiment_id and target.ref.task_id
    ):
        raise FeedbackError(
            "a task summary must declare the experiment_id and task_id it "
            "summarizes, so it can be found from the task it is about"
        )
    spans = list(store.get_spans(target.turn_key))
    if target.span_ids:
        recorded = {
            str(span["span_id"]) for span in spans if span.get("span_id")
        }
        missing = sorted(set(target.span_ids) - recorded)
        if missing:
            raise FeedbackError(
                f"span(s) {missing} are not recorded on turn {target.turn_key!r}"
            )
    if target.ref.pass_id is not None:
        if pass_selector is None:
            raise FeedbackError(
                f"reference names pass {target.ref.pass_id!r} but no pass "
                "selector was supplied; pass membership is resolved from "
                "recorded spans and is never assumed"
            )
        if pass_selector.pass_id != target.ref.pass_id:
            raise FeedbackError(
                f"reference names pass {target.ref.pass_id!r} but the selector "
                f"resolves pass {pass_selector.pass_id!r}"
            )
        try:
            pass_selector.resolve_against(_SpanTree(spans), target.turn_key)
        except InvalidExecutionRef as exc:
            raise FeedbackError(str(exc)) from exc
    elif pass_selector is not None:
        raise FeedbackError(
            f"a pass selector for {pass_selector.pass_id!r} was supplied for a "
            "reference that names no pass"
        )


def complete_scope(target: FeedbackTarget, store: Any) -> FeedbackTarget:
    """Fill in experiment/task/attempt the caller did not declare.

    Read off the anchored turn's own row, never off a label or the reader's
    current selection, and only where the caller said nothing: a DECLARED
    scope is still checked against the row by `validate_target` and a wrong
    one is refused rather than corrected. The point is that the frozen anchor
    of an ordinary turn comment says which task it belongs to, so the other
    side of a comparison can be named without reopening its database.
    """
    row = store.get_turn(target.turn_key)
    if row is None:
        return target
    from dataclasses import replace

    fields = {
        name: row.get(name)
        for name in ("experiment_id", "task_id", "attempt")
        if getattr(target.ref, name) is None and row.get(name) is not None
    }
    if not fields:
        return target
    return replace(target, ref=replace(target.ref, **fields))


def build_anchors(
    primary: FeedbackTarget,
    *,
    sources: Mapping[str, Any],
    paired: Optional[FeedbackTarget] = None,
    primary_pass_selector: Optional[PassSelector] = None,
    paired_pass_selector: Optional[PassSelector] = None,
) -> FeedbackAnchors:
    """Validate both sides against their OWN authorized stores.

    `sources` maps store id to an opened store the caller is allowed to read.
    A reference to a store that is not in the table is refused rather than
    resolved somewhere else — the two sides of a comparison routinely live in
    different evidence databases, and "the store I happen to have open" is not
    an authorization.
    """
    completed: list[Optional[FeedbackTarget]] = []
    for target, selector in ((primary, primary_pass_selector), (paired, paired_pass_selector)):
        if target is None:
            completed.append(None)
            continue
        store = sources.get(target.store_id)
        if store is None:
            raise FeedbackError(
                f"store {target.store_id!r} was not authorized for this write"
            )
        # Filled in BEFORE validation, not after: a scope copied off one turn
        # row is then checked against every turn the reference names, so a
        # multi-turn reference cannot acquire a task id that only its first
        # turn actually records.
        target = complete_scope(target, store)
        validate_target(target, store, pass_selector=selector)
        completed.append(target)
    primary, paired = completed[0], completed[1]
    if paired is not None and paired.ref.ref_id() == primary.ref.ref_id():
        raise FeedbackError(
            "a comparison comment names two different executions; both sides "
            "resolve to the same one"
        )
    return FeedbackAnchors(primary=primary, paired=paired)


def record_feedback(
    store: Any,
    *,
    primary: FeedbackTarget,
    comment: str,
    category: str,
    subcategory: str,
    provenance: str,
    sources: Optional[Mapping[str, Any]] = None,
    paired: Optional[FeedbackTarget] = None,
    primary_pass_selector: Optional[PassSelector] = None,
    paired_pass_selector: Optional[PassSelector] = None,
) -> dict[str, Any]:
    """The single writer. Humans and coding agents both arrive here.

    `store` is the writable store the row lands in, and it must be the store
    the primary reference names: a comment lives with the evidence it is
    primarily about. The paired side may live anywhere `sources` authorizes,
    including another database, and is recorded as a frozen reference rather
    than as a second row.
    """
    table = dict(sources or {})
    table.setdefault(primary.store_id, store)
    if table.get(primary.store_id) is not store:
        raise FeedbackError(
            "the primary reference must name the store being written to"
        )
    anchors = build_anchors(
        primary,
        sources=table,
        paired=paired,
        primary_pass_selector=primary_pass_selector,
        paired_pass_selector=paired_pass_selector,
    )
    category, subcategory = validate_category(category, subcategory)
    return store.add_human_feedback(
        primary.turn_key,
        target_kind=primary.target_kind,
        span_ids=list(primary.span_ids),
        target_label=primary.target_label,
        provenance=provenance,
        comment=comment,
        category=category,
        subcategory=subcategory,
        anchors=anchors,
    )


# ----------------------------------------------------------------------
# The consolidated task read
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class TaskFeedbackPage:
    """One bounded page of a task's feedback, plus what was filtered out.

    `total` counts the authorized comments for the task BEFORE paging and
    after filtering, so a reader can tell "there is more" from "that is all"
    without fetching everything.
    """

    rows: tuple[dict[str, Any], ...]
    total: int
    limit: int
    offset: int
    filters: Mapping[str, Any]
    stores: tuple[str, ...]

    @property
    def has_more(self) -> bool:
        return self.offset + len(self.rows) < self.total

    def as_dict(self) -> dict[str, Any]:
        return {
            "feedback": list(self.rows),
            "total": self.total,
            "limit": self.limit,
            "offset": self.offset,
            "has_more": self.has_more,
            "filters": dict(self.filters),
            "stores": list(self.stores),
        }


def _sort_key(row: Mapping[str, Any]) -> tuple[str, str, int]:
    """Deterministic chronology that never depends on which store answered.

    Timestamp first, then the store id, then the row id: two comments written
    in the same second in two databases still have exactly one order, and
    paging over it cannot show or skip a row because the merge happened to run
    differently.
    """
    return (
        str(row.get("created_at") or ""),
        str(row.get("store_id") or ""),
        int(row.get("feedback_id") or 0),
    )


def dedupe_key(row: Mapping[str, Any]) -> str:
    """One identity per comment, stable across the views that can find it.

    A comparison comment is reachable from both of the tasks it names, and an
    archived copy of a store holds the same row as the live one. Both must show
    the comment once. Two people who independently type the same sentence have
    written two comments, and this key keeps them apart -- it is the row's
    identity, never a digest of its text.
    """
    uid = row.get("feedback_uid")
    if isinstance(uid, str) and uid:
        return uid
    return f"legacy:{row.get('store_id') or ''}:{row.get('feedback_id')}"


def _side_scope(side: Any) -> Optional[tuple[Any, Any, Any]]:
    """The experiment/task/attempt one stored anchor side names, or None."""
    if not isinstance(side, Mapping):
        return None
    ref = side.get("ref")
    if not isinstance(ref, Mapping):
        return None
    return (ref.get("experiment_id"), ref.get("task_id"), ref.get("attempt"))


def task_attempts(
    row: Mapping[str, Any],
    *,
    experiment_id: str,
    task_id: str,
) -> Optional[frozenset[int]]:
    """Which attempts OF THE QUERIED TASK one stored comment is about.

    `list_task_feedback` returns a row when EITHER the annotated turn or the
    frozen paired anchor names the task, while the `attempt` column always
    belongs to the annotated turn. A comparison note written on
    left-experiment/left-task attempt 1 against right-experiment/right-task
    attempt 2 therefore carries `attempt=1` when the RIGHT task's page asks
    about its own attempt 2: filtering on the column alone shows that note
    under an attempt the right task never ran and hides it under the one it
    did.

    So each side is matched in its own right, and the attempts of the sides
    that actually name the queried task are what comes back:

    - BOTH sides may match. A note pairing two attempts of one task is about
      both of them, and it stays one row either way (`dedupe_key`).
    - A side that names the task WITHOUT an attempt matches every attempt of
      it: a note anchored to the task as a whole is not evidence about one run,
      and hiding it from every attempt view would lose it entirely.
    - `None` means "no attempt is named", the same permissive answer.
    """
    wanted = str(experiment_id)
    task = str(task_id)
    sides: list[tuple[Any, Any, Any]] = [
        (row.get("turn_experiment_id"), row.get("turn_task_id"), row.get("attempt"))
    ]
    paired = _side_scope(row.get("paired"))
    if paired is None:
        anchors = row.get("anchors")
        if isinstance(anchors, Mapping):
            paired = _side_scope(anchors.get("paired"))
    if paired is None and row.get("pair_task_id"):
        # The pair columns without a readable anchor: the row is known to be
        # about the other side's task, and the attempt it names is not
        # recoverable, so it is treated as naming none.
        paired = (row.get("pair_experiment_id"), row.get("pair_task_id"), None)
    if paired is not None:
        sides.append(paired)

    matched = False
    attempts: set[int] = set()
    for side_experiment, side_task, attempt in sides:
        if side_experiment is None or side_task is None:
            continue
        if str(side_task) != task or str(side_experiment) != wanted:
            continue
        matched = True
        if attempt is None:
            return None
        try:
            attempts.add(int(attempt))
        except (TypeError, ValueError):  # pragma: no cover - a non-numeric attempt
            return None
    if not matched:
        # Unreachable for rows this module fetched, because the read that
        # produced them matched the task on one side or the other. A caller
        # filtering rows from somewhere else keeps the column's own meaning
        # rather than having the filter quietly pass everything.
        attempt = row.get("attempt")
        return frozenset({int(attempt)}) if isinstance(attempt, int) else None
    return frozenset(attempts)


def _attempt_matches(
    row: Mapping[str, Any],
    *,
    experiment_id: str,
    task_id: str,
    attempt: Optional[int],
) -> bool:
    if attempt is None:
        return True
    named = task_attempts(row, experiment_id=experiment_id, task_id=task_id)
    return named is None or int(attempt) in named


def consolidate_task_feedback(
    stores: Mapping[str, Any],
    *,
    experiment_id: str,
    task_id: str,
    category: Optional[str] = None,
    subcategory: Optional[str] = None,
    provenance: Optional[str] = None,
    target_kind: Optional[str] = None,
    component: Optional[str] = None,
    attempt: Optional[int] = None,
    limit: int = 100,
    offset: int = 0,
) -> TaskFeedbackPage:
    """Every authorized comment on one task, across attempts, turns and stores.

    There is no default component filter and no default category filter: the
    task view's job is to show the whole record, and a hidden default would
    silently answer a different question than the one the reader asked.

    `stores` is the authorized set, keyed by store id — an experiment's
    evidence may be split across the live store and one or more registered or
    archived ones, and this follows the stores it was given rather than
    assuming a default database.
    """
    if not isinstance(experiment_id, str) or not experiment_id:
        raise FeedbackError("experiment_id is required")
    if not isinstance(task_id, str) or not task_id:
        raise FeedbackError("task_id is required")
    if category is not None and category not in FEEDBACK_CATEGORIES:
        raise FeedbackError("unknown category filter")
    if subcategory is not None:
        allowed: Iterable[str] = (
            FEEDBACK_SUBCATEGORIES[category]
            if category is not None
            else [v for values in FEEDBACK_SUBCATEGORIES.values() for v in values]
        )
        if subcategory not in allowed:
            raise FeedbackError("unknown subcategory filter")
    if limit < 0 or offset < 0:
        raise FeedbackError("limit and offset must not be negative")

    merged: dict[str, dict[str, Any]] = {}
    for store_id, store in sorted(stores.items()):
        for row in store.list_task_feedback(
            experiment_id=experiment_id, task_id=task_id
        ):
            row = dict(row)
            row["store_id"] = store_id
            merged.setdefault(dedupe_key(row), row)

    rows = sorted(merged.values(), key=_sort_key)
    filtered = [
        row
        for row in rows
        if (category is None or row.get("category") == category)
        and (subcategory is None or row.get("subcategory") == subcategory)
        and (provenance is None or row.get("provenance") == provenance)
        and (target_kind is None or row.get("target_kind") == target_kind)
        and (component is None or row.get("target_label") == component)
        # The attempt of whichever side names the task being read, not the
        # annotated turn's: see `task_attempts`.
        and _attempt_matches(
            row,
            experiment_id=experiment_id,
            task_id=task_id,
            attempt=attempt,
        )
    ]
    window = filtered[offset : offset + limit] if limit else []
    return TaskFeedbackPage(
        rows=tuple(window),
        total=len(filtered),
        limit=limit,
        offset=offset,
        filters={
            "experiment_id": experiment_id,
            "task_id": task_id,
            "category": category,
            "subcategory": subcategory,
            "provenance": provenance,
            "target_kind": target_kind,
            "component": component,
            "attempt": attempt,
        },
        stores=tuple(sorted(stores)),
    )


def presentation(row: Mapping[str, Any]) -> dict[str, Any]:
    """Add display labels without inventing source identity.

    A row that was written before the taxonomy gets `classified: False` and
    labels of None. It is shown as unclassified legacy feedback, with its text
    exactly as it was stored; guessing which of the six subcategories its
    author would have picked is the one thing this must not do.
    """
    value = dict(row)
    category = value.get("category")
    subcategory = value.get("subcategory")
    value["classified"] = bool(category and subcategory)
    value["category_label"] = category_label(category)
    value["subcategory_label"] = subcategory_label(category, subcategory)
    return value


def present(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [presentation(row) for row in rows]

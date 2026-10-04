"""Best run for one task within one experiment (`fix-9eg.17.4`).

A DIFFERENT selection from the experiment winner, and the distinction is the
whole reason this module exists rather than a flag on `selection.py`:

- The **winner** (`selection.record_decision`) is a judgement about an
  experiment: which configuration a workflow currently runs on.
- The **best run** here is a judgement about one task inside one experiment:
  which of its n repeated attempts is the preferred example. It is a teaching
  example for distillation, not proof of correctness, and choosing it deploys
  nothing, reruns nothing, and moves no winner.

The two share storage — one control sidecar, one append-only decision table —
and are kept apart by the pointer's primary key: this scope writes
`scope_kind='task_best'` with a non-empty `scope_key`, and
`SelectionControlStore.apply_scoped_decision` refuses the experiment scope
outright, so no best-run write can reach the winner pointer even by mistake.

Three properties this module is built around, each of which is a silent wrong
answer if it breaks:

1. **Only a recorded, finished attempt of THIS task may be chosen.** Finished
   means `execution_finished_at` is set — the execution completion marker
   (`[XR13]`), not "has turns". A `failed` attempt is selectable on purpose
   (its status stays visible; a failure can be the clearest example of a
   behaviour), but an attempt still running has not produced the example
   anybody is claiming to prefer.

2. **The viewing reference is never Best.** Before anybody decides, the first
   completed attempt is a reasonable thing to show, and it is labelled
   `Reference` here in the data, not only in a UI string, so an agent reading
   this API cannot mistake a default for a decision.

3. **Old references never retarget.** Everything returned points at a
   `comparison.ExecutionRef`, whose `ref_id` is derived from the scope it
   names and excludes `label`. Pinning a different best run produces a
   different reference and therefore different `review_pair_key`s: comparison
   comments recorded against yesterday's pair still name yesterday's runs.

An attempt is one reference even when it took ten turns: `turn_keys` is the
full conversation in order, so a multi-turn attempt stays navigable instead of
collapsing to its final answer.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from fastworkflow.observability.comparison import ExecutionRef
from fastworkflow.observability.selection import (
    SelectionControlError,
    SelectionControlStore,
    TASK_BEST_SCOPE,
)

# The decision vocabulary of THIS scope. Deliberately not the winner's words:
# "promote" moves a workflow onto a configuration, "select" marks an example.
DECISION_SELECT = "select"
DECISION_REPLACE = "replace"
DECISION_CLEAR = "clear"
DECISION_UNDECIDED = "undecided"

BEST_RUN_DECISIONS = frozenset(
    {DECISION_SELECT, DECISION_REPLACE, DECISION_CLEAR, DECISION_UNDECIDED}
)

# Labels travel in the data, not only in the UI, so a coding agent reading this
# API sees the same distinction a human sees on the screen.
LABEL_BEST = "Best run"
LABEL_REFERENCE = "Reference"
# A finished attempt whose turns are not there to read. Named, because the
# alternative to a label is each client inventing its own way of saying it.
LABEL_EVIDENCE_MISSING = "No recorded turns"

EVIDENCE_READABLE = "readable"
EVIDENCE_MISSING = "missing"

# One keyset page of turns. A page SIZE is a read-efficiency choice; a page
# LIMIT would be a correctness one, so there isn't one — `_turn_rows` walks
# until the store stops answering.
_TURN_PAGE = 500
_MAX_REASON = 4000


def _conversation_order(row: Mapping[str, Any]) -> tuple:
    """Sort key putting one attempt's turns in the order they were recorded.

    `turn_key` order is NOT conversation order — it is a string, so turn 10
    sorts before turn 2 under any of the key shapes this repo mints. The
    recorded ordinal within a conversation is the real sequence; `started_at`
    and finally the key itself only break ties for rows that predate it. Each
    component is `(0, value)` when present and `(1, default)` when not, so a
    missing ordinal sorts last instead of raising against an int.
    """
    conversation = row.get("conversation_id")
    ordinal = row.get("ordinal")
    started = row.get("started_at")
    return (
        (0, int(conversation)) if isinstance(conversation, int) else (1, 0),
        (0, int(ordinal)) if isinstance(ordinal, int) else (1, 0),
        (0, str(started)) if started else (1, ""),
        str(row["turn_key"]),
    )


class BestRunError(RuntimeError):
    """Base class for task-best selection problems."""


class TaskBestUnavailable(BestRunError):
    """The experiment or its evidence is not readable through this control.

    Raised rather than answered with None because "this task has no best run"
    and "I cannot see this experiment" are different facts, and a UI that
    renders the second as the first invites somebody to pick a best run that
    lands in a contest they are not looking at.
    """


class AttemptNotSelectable(BestRunError, ValueError):
    """The named attempt is not a recorded, finished attempt of this task."""

    def __init__(self, experiment_id: str, task_id: str, attempt: Any, reason: str):
        self.experiment_id = experiment_id
        self.task_id = task_id
        self.attempt = attempt
        self.reason = reason
        super().__init__(
            f"attempt {attempt!r} of task {task_id!r} in experiment "
            f"{experiment_id!r} cannot be the best run: {reason}"
        )


class NoBestRun(BestRunError, ValueError):
    """Clearing a selection that is not there."""


def _exact_int(value: Any, field: str) -> int:
    """An attempt number, refusing the things that silently become one.

    `bool` is an `int` in Python and `int(2.7)` is 2, so the ordinary
    conversion would accept `True` as attempt 1 and `2.7` as attempt 2 — a
    reference to somebody else's run, recorded without complaint.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        if isinstance(value, str) and value.strip().lstrip("-").isdigit():
            return int(value.strip())
        raise ValueError(f"{field} must be an exact integer, not {value!r}")
    return value


def task_scope_key(experiment_id: str, task_id: str) -> str:
    """The scope one best-run selection lives under.

    Both ids, because the scope is "this task in THIS experiment": the same
    task run under a candidate experiment is a different set of attempts and
    gets its own decision. `\\x1f` is the unit separator, which cannot occur in
    an id, so no pair of ids can be made to collide by concatenation.
    """
    experiment_id = str(experiment_id or "").strip()
    task_id = str(task_id or "").strip()
    if not experiment_id or not task_id:
        raise ValueError("experiment_id and task_id are both required")
    return f"{experiment_id}\x1ftask\x1f{task_id}"


class _TaskScope:
    """Everything one task's selection needs, resolved once per call."""

    __slots__ = ("control", "experiment_id", "task_id", "group_id", "source_id",
                 "store", "store_id", "scope_key")

    def __init__(
        self,
        control: SelectionControlStore,
        experiment_id: str,
        task_id: str,
        *,
        store_id: Optional[str] = None,
        require_store: bool = True,
    ) -> None:
        self.control = control
        self.experiment_id = str(experiment_id or "").strip()
        self.task_id = str(task_id or "").strip()
        self.scope_key = task_scope_key(self.experiment_id, self.task_id)
        group = control.group_for_experiment(self.experiment_id)
        if group is None:
            raise TaskBestUnavailable(
                f"experiment {self.experiment_id!r} is not registered in this "
                "control store; register it before selecting a best run so the "
                "selection lands in the contest the experiment belongs to"
            )
        self.group_id = str(group["group_id"])
        self.source_id = control.source_for_experiment(self.experiment_id)
        try:
            self.store = control.store_for_source(self.source_id)
        except SelectionControlError as exc:
            raise TaskBestUnavailable(str(exc)) from exc
        if self.store is None and require_store:
            raise TaskBestUnavailable(
                f"no evidence store was supplied for source {self.source_id!r}; "
                "the attempts of this task cannot be read, so none of them can "
                "be verified as a finished attempt of it"
            )
        # `ExecutionRef.store_id` names the store a reader will open this turn
        # through. The control's source id IS that name in a shared workspace;
        # an embedder whose reader keys stores differently overrides it, and
        # then its own refs and its own reader agree.
        self.store_id = store_id or self.source_id or "primary"

    def attempt_rows(self) -> list[dict[str, Any]]:
        if self.store is None:
            return []
        return [
            dict(row)
            for row in self.store.experiment_attempt_rows(
                self.experiment_id, task_id=self.task_id
            )
        ]

    def turn_keys(self, attempt: int) -> list[str]:
        """This attempt's WHOLE conversation, oldest turn first.

        Whole matters more than it looks. An `ExecutionRef` naming a subset of
        an attempt's turns is not distinguishable, downstream, from one naming
        all of them: it opens, it renders, and the turns that were dropped are
        simply not on the screen. A long attempt would then be silently
        reviewed, compared and chosen as a best run on part of its evidence —
        so this pages until the store is exhausted rather than reading one
        capped page.

        Paging is by KEYSET (`before_turn_key`), not offset: offsets shift
        under a store still being written, which can repeat or skip a turn.
        """
        return [str(row["turn_key"]) for row in self._turn_rows(attempt)]

    def _turn_rows(self, attempt: int) -> list[dict[str, Any]]:
        if self.store is None:
            return []
        collected: dict[str, dict[str, Any]] = {}
        before: Optional[str] = None
        while True:
            page = self.store.list_turns(
                experiment_id=self.experiment_id,
                task_id=self.task_id,
                attempt=int(attempt),
                limit=_TURN_PAGE,
                before_turn_key=before,
            )
            if not page:
                break
            for row in page:
                collected.setdefault(str(row["turn_key"]), dict(row))
            before = str(page[-1]["turn_key"])
            if len(page) < _TURN_PAGE:
                break
        return sorted(collected.values(), key=_conversation_order)


def attempt_is_finished(row: Mapping[str, Any]) -> bool:
    """Whether an attempt row is a RECORDED, finished attempt.

    `execution_finished_at` is the execution completion marker (`[XR13]`). An
    attempt that crashed halfway has turns and an open marker; it is evidence
    of an interruption, not of a run somebody could prefer.
    """
    return row.get("execution_finished_at") is not None


def project_attempt(
    scope: "_TaskScope", row: Mapping[str, Any], *, label: Optional[str] = None
) -> dict[str, Any]:
    """One attempt as both clients read it: status first, never laundered.

    `outcome` and `execution_status` are passed through exactly as recorded, so
    an attempt chosen as the best run while it `failed` still reads as failed.

    A finished attempt with no recorded turns is a real state — an execution
    that completed without any turn being written, or whose turns were pruned —
    and it is reported honestly rather than hidden or rejected:

    - `execution_ref` is None, because there is nothing to point a reader at.
      A client must branch on this rather than construct `ExecutionRef` from
      it; `comparable` is the flag to branch on.
    - `comparable` is False: there is no reference, so there is no pair, so
      comparison is off for this attempt.
    - `selectable` is still whatever `finished` says. Somebody may legitimately
      pin the run that finished, and refusing it would hide from the task's
      history that it is the one they meant.
    - `evidence_state` is `"missing"` and `evidence_label` carries the words
      for it, so the honest state has one spelling across clients.
    """
    attempt = int(row["attempt"])
    turn_keys = scope.turn_keys(attempt)
    finished = attempt_is_finished(row)
    ref = ExecutionRef(
        store_id=scope.store_id,
        turn_keys=tuple(turn_keys),
        experiment_id=scope.experiment_id,
        task_id=scope.task_id,
        attempt=attempt,
        label=label,
    ) if turn_keys else None
    return {
        "experiment_id": scope.experiment_id,
        "task_id": scope.task_id,
        "attempt": attempt,
        "source_id": scope.source_id,
        "store_id": scope.store_id,
        "channel_id": row.get("channel_id"),
        "conversation_id": row.get("conversation_id"),
        "execution_status": row.get("execution_status"),
        "execution_finished_at": row.get("execution_finished_at"),
        "outcome": row.get("outcome"),
        "outcome_source": row.get("outcome_source"),
        "reward": row.get("reward"),
        "restarts": row.get("restarts"),
        "started_at": row.get("started_at"),
        "finished_at": row.get("finished_at"),
        "turn_keys": turn_keys,
        "turn_count": len(turn_keys),
        "finished": finished,
        # Selectable and finished are the same test today. They are two keys
        # because a client should ask "may I offer this?" rather than re-derive
        # the rule, which is the thing that drifts.
        "selectable": finished,
        "comparable": ref is not None,
        "evidence_state": EVIDENCE_READABLE if ref is not None else EVIDENCE_MISSING,
        "evidence_label": None if ref is not None else LABEL_EVIDENCE_MISSING,
        "label": label,
        "execution_ref": None if ref is None else ref.as_dict(),
    }


def list_task_attempts(
    control: SelectionControlStore,
    experiment_id: str,
    task_id: str,
    *,
    store_id: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Every recorded attempt of this task, ascending, with its real status.

    Every attempt, including the failed and the unfinished ones: an n-run task
    whose screen hides its failures is a screen that reports a success rate it
    did not measure.
    """
    scope = _TaskScope(control, experiment_id, task_id, store_id=store_id)
    rows = sorted(scope.attempt_rows(), key=lambda row: int(row["attempt"]))
    return [project_attempt(scope, row) for row in rows]


def select_task_attempts(
    control: SelectionControlStore,
    experiment_id: str,
    task_id: str,
    attempts: Optional[Sequence[int]],
    *,
    store_id: Optional[str] = None,
    max_selected: Optional[int] = None,
) -> dict[str, Any]:
    """The attempts a caller NAMED, projected; the rest enumerated only.

    `list_task_attempts` pages every attempt's whole conversation, which is
    what the task header needs and exactly what a caller asking about three of
    forty runs must not pay for -- and, more importantly, must not PUBLISH:
    projecting the other thirty-seven would put evidence nobody asked about
    into the answer.

    So the attempt rows are enumerated (metadata the store already holds: the
    attempt number, its status, its completion marker) and `project_attempt`
    -- the part that pages turn keys and builds an `ExecutionRef` -- runs for
    the named attempts only. `recorded_attempts` says which numbers exist, so a
    caller can report the ones it named that do not, without reading them.

    `best_attempt` and `reference_attempt` are resolved as NUMBERS here, from
    the control pointer and from the attempt rows' own status, for the same
    reason: labelling the named runs does not require projecting the others.

    `attempts=None` is the SERVER-RESOLVED rule (`fix-9eg.3.2.2.1`): every
    finished attempt of this task, decided from the same enumerated metadata
    rather than from a list somebody sent or a row somebody could see. It is
    one read either way -- the finished/unfinished partition and the caller's
    named set come off the same rows -- which is what keeps a population check
    from re-reading anything.

    `max_selected` bounds that rule, and is checked BEFORE any projection
    happens: a population over the bound answers `over_limit` with its counts
    and costs no turn read at all, so the caller can refuse the whole request
    rather than summarize part of it.
    """
    scope = _TaskScope(control, experiment_id, task_id, store_id=store_id,
                       require_store=False)
    rows = sorted(scope.attempt_rows(), key=lambda row: int(row["attempt"]))
    by_attempt = {int(row["attempt"]): row for row in rows}
    finished_attempts = [
        int(row["attempt"]) for row in rows if attempt_is_finished(row)
    ]
    unfinished_attempts = [
        int(row["attempt"]) for row in rows if not attempt_is_finished(row)
    ]
    pointer = control.scoped_pointer(
        scope_kind=TASK_BEST_SCOPE, group_id=scope.group_id, scope_key=scope.scope_key
    )
    best_attempt = (
        None
        if pointer is None or pointer.get("attempt") is None
        else int(pointer["attempt"])
    )
    reference = next(
        (
            int(row["attempt"])
            for row in rows
            if row.get("execution_status") == "completed" and attempt_is_finished(row)
        ),
        None,
    )
    all_finished = attempts is None
    wanted = (
        list(finished_attempts)
        if all_finished
        else sorted({int(attempt) for attempt in attempts})
    )
    over_limit = (
        all_finished and max_selected is not None and len(wanted) > int(max_selected)
    )
    selected: list[dict[str, Any]] = []
    if not over_limit:
        for attempt in wanted:
            row = by_attempt.get(attempt)
            if row is None:
                continue
            projected = project_attempt(scope, row)
            projected["is_best"] = best_attempt is not None and attempt == best_attempt
            projected["is_reference"] = reference is not None and attempt == reference
            selected.append(projected)
    return {
        "experiment_id": scope.experiment_id,
        "task_id": scope.task_id,
        "group_id": scope.group_id,
        "source_id": scope.source_id,
        "store_id": scope.store_id,
        "scope_key": scope.scope_key,
        "evidence_readable": scope.store is not None,
        "recorded_attempts": [int(row["attempt"]) for row in rows],
        "finished_attempts": finished_attempts,
        "unfinished_attempts": unfinished_attempts,
        "best_attempt": best_attempt,
        "reference_attempt": reference,
        "requested_attempts": wanted,
        "over_limit": over_limit,
        "selected": selected,
        "not_recorded": [
            attempt for attempt in wanted if attempt not in by_attempt
        ],
    }


def reference_attempt(
    control: SelectionControlStore,
    experiment_id: str,
    task_id: str,
    *,
    store_id: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """The first COMPLETED attempt, for viewing only. Never "Best".

    Returned labelled `Reference` whether or not a best run exists, because
    what it is does not change when somebody decides: it is simply the one a
    reader lands on by default. `is_best` is always False here — the only place
    `LABEL_BEST` is attached is a recorded decision.
    """
    scope = _TaskScope(control, experiment_id, task_id, store_id=store_id)
    for row in sorted(scope.attempt_rows(), key=lambda row: int(row["attempt"])):
        if row.get("execution_status") == "completed" and attempt_is_finished(row):
            projected = project_attempt(scope, row, label=LABEL_REFERENCE)
            projected["is_best"] = False
            projected["is_reference"] = True
            return projected
    return None


def best_run(
    control: SelectionControlStore,
    experiment_id: str,
    task_id: str,
    *,
    store_id: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """The selected best run of this task, or None if nobody has chosen one.

    `attempt_resolved` is False when the selection is recorded but its attempt
    row cannot be read right now (evidence pruned, store not supplied). The
    decision is still a fact; what the run did is simply unknown, and saying so
    beats implying the selection was never made.
    """
    scope = _TaskScope(control, experiment_id, task_id, store_id=store_id,
                       require_store=False)
    pointer = control.scoped_pointer(
        scope_kind=TASK_BEST_SCOPE, group_id=scope.group_id, scope_key=scope.scope_key
    )
    if pointer is None:
        return None
    attempt = None if pointer["attempt"] is None else int(pointer["attempt"])
    projected = None
    if scope.store is not None and attempt is not None:
        for row in scope.attempt_rows():
            if int(row["attempt"]) == attempt:
                projected = project_attempt(scope, row, label=LABEL_BEST)
                projected["is_best"] = True
                projected["is_reference"] = False
                break
    return {
        "scope_kind": TASK_BEST_SCOPE,
        "group_id": scope.group_id,
        "scope_key": scope.scope_key,
        "experiment_id": str(pointer["experiment_id"]),
        "task_id": None if pointer["task_id"] is None else str(pointer["task_id"]),
        "attempt": attempt,
        "source_id": scope.source_id,
        "selection_id": str(pointer["selection_id"]),
        "decision": str(pointer["decision"]),
        "decision_seq": int(pointer["decision_seq"]),
        "decided_at": str(pointer["decided_at"]),
        "label": LABEL_BEST,
        "attempt_resolved": projected is not None,
        # Hoisted out of `run` so a client can disable comparison without
        # having to handle `run=None` and `run["execution_ref"]=None` as two
        # separate cases: both mean there is no reference to compare.
        "comparable": bool(projected is not None and projected["comparable"]),
        "execution_ref": None if projected is None else projected["execution_ref"],
        "run": projected,
    }


def best_run_history(
    control: SelectionControlStore,
    experiment_id: str,
    task_id: str,
    *,
    limit: int = 100,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """This task's decisions, newest first. Append-only; nothing is rewritten."""
    scope = _TaskScope(control, experiment_id, task_id, require_store=False)
    return control.scoped_history(
        scope_kind=TASK_BEST_SCOPE,
        group_id=scope.group_id,
        scope_key=scope.scope_key,
        limit=limit,
        offset=offset,
    )


def task_run_summary(
    control: SelectionControlStore,
    experiment_id: str,
    task_id: str,
    *,
    store_id: Optional[str] = None,
) -> dict[str, Any]:
    """One read for the task header: best, reference and every attempt.

    `expected_selection_id` is included so the client that rendered this screen
    can echo it back into the next decision without a second round trip — which
    is what makes stale-update detection usable rather than merely available.
    """
    scope = _TaskScope(control, experiment_id, task_id, store_id=store_id,
                       require_store=False)
    current = best_run(control, experiment_id, task_id, store_id=store_id)
    reference = (
        None
        if scope.store is None
        else reference_attempt(control, experiment_id, task_id, store_id=store_id)
    )
    attempts = (
        []
        if scope.store is None
        else list_task_attempts(control, experiment_id, task_id, store_id=store_id)
    )
    best_attempt = None if current is None else current["attempt"]
    for row in attempts:
        row["is_best"] = best_attempt is not None and row["attempt"] == best_attempt
        row["is_reference"] = (
            reference is not None and row["attempt"] == reference["attempt"]
        )
    return {
        "experiment_id": scope.experiment_id,
        "task_id": scope.task_id,
        "group_id": scope.group_id,
        "source_id": scope.source_id,
        "store_id": scope.store_id,
        "scope_key": scope.scope_key,
        "evidence_readable": scope.store is not None,
        "best_run": current,
        "reference": reference,
        "attempts": attempts,
        "attempt_count": len(attempts),
        "selectable_attempts": [row["attempt"] for row in attempts if row["selectable"]],
        # A subset of the selectable ones: an attempt with no recorded turns
        # may be pinned, but there is no reference to open or pair.
        "comparable_attempts": [
            row["attempt"] for row in attempts if row["comparable"]
        ],
        "expected_selection_id": None if current is None else current["selection_id"],
    }


def _decide(
    scope: "_TaskScope",
    *,
    decision: str,
    expected_selection_id: Optional[str],
    target: Optional[Mapping[str, Any]],
    clear: bool,
    candidate: Optional[Mapping[str, Any]],
    actor: str,
    actor_kind: str,
    provenance: str,
    reason: Optional[str],
) -> dict[str, Any]:
    if reason is not None and len(reason) > _MAX_REASON:
        raise ValueError(f"reason must be at most {_MAX_REASON} characters")
    result = scope.control.apply_scoped_decision(
        scope_kind=TASK_BEST_SCOPE,
        group_id=scope.group_id,
        scope_key=scope.scope_key,
        decision=decision,
        expected_selection_id=expected_selection_id,
        target=target,
        clear=clear,
        candidate=candidate,
        actor=actor,
        actor_kind=actor_kind,
        provenance=provenance,
        rationale=reason,
    )
    result["experiment_id"] = scope.experiment_id
    result["task_id"] = scope.task_id
    result["best_run"] = best_run(
        scope.control, scope.experiment_id, scope.task_id, store_id=scope.store_id
    )
    return result


def select_best_run(
    control: SelectionControlStore,
    experiment_id: str,
    task_id: str,
    attempt: Any,
    *,
    expected_selection_id: Optional[str],
    actor: str,
    actor_kind: str,
    provenance: str,
    reason: Optional[str] = None,
    store_id: Optional[str] = None,
) -> dict[str, Any]:
    """Use this attempt as the best run for this task.

    The decision WORD is derived, not supplied: the first selection is
    `select` and every later one is `replace`. A caller cannot mislabel its own
    history, and "was there a previous choice here?" stays answerable from the
    history alone.

    `expected_selection_id` must be what the client last read — `None` meaning
    "there was no selection". A decision made against a selection somebody has
    already replaced is refused with `selection.StaleSelection`, which carries
    what is current, rather than silently overwriting it.
    """
    scope = _TaskScope(control, experiment_id, task_id, store_id=store_id)
    attempt = _exact_int(attempt, "attempt")
    rows = {int(row["attempt"]): row for row in scope.attempt_rows()}
    row = rows.get(attempt)
    if row is None:
        raise AttemptNotSelectable(
            scope.experiment_id, scope.task_id, attempt,
            "there is no recorded attempt with that number for this task",
        )
    if not attempt_is_finished(row):
        raise AttemptNotSelectable(
            scope.experiment_id, scope.task_id, attempt,
            f"it has not finished (execution_status="
            f"{row.get('execution_status')!r}); a run still in flight is not "
            "yet the example anybody is preferring",
        )
    current = control.scoped_pointer(
        scope_kind=TASK_BEST_SCOPE, group_id=scope.group_id, scope_key=scope.scope_key
    )
    decision = DECISION_SELECT if current is None else DECISION_REPLACE
    target = {
        "experiment_id": scope.experiment_id,
        "task_id": scope.task_id,
        "attempt": attempt,
    }
    return _decide(
        scope,
        decision=decision,
        expected_selection_id=expected_selection_id,
        target=target,
        clear=False,
        candidate=target,
        actor=actor,
        actor_kind=actor_kind,
        provenance=provenance,
        reason=reason,
    )


def clear_best_run(
    control: SelectionControlStore,
    experiment_id: str,
    task_id: str,
    *,
    expected_selection_id: str,
    actor: str,
    actor_kind: str,
    provenance: str,
    reason: Optional[str] = None,
    store_id: Optional[str] = None,
) -> dict[str, Any]:
    """Withdraw this task's best run, leaving its history intact.

    The attempts, the decisions that chose and unchose them, and any comparison
    comments recorded against them all stay exactly where they are. What is
    removed is one pointer.
    """
    scope = _TaskScope(control, experiment_id, task_id, store_id=store_id,
                       require_store=False)
    current = control.scoped_pointer(
        scope_kind=TASK_BEST_SCOPE, group_id=scope.group_id, scope_key=scope.scope_key
    )
    if current is None:
        raise NoBestRun(
            f"task {scope.task_id!r} in experiment {scope.experiment_id!r} has "
            "no best run to clear"
        )
    return _decide(
        scope,
        decision=DECISION_CLEAR,
        expected_selection_id=expected_selection_id,
        target=None,
        clear=True,
        candidate={
            "experiment_id": str(current["experiment_id"]),
            "task_id": None if current["task_id"] is None else str(current["task_id"]),
            "attempt": current["attempt"],
        },
        actor=actor,
        actor_kind=actor_kind,
        provenance=provenance,
        reason=reason,
    )


def leave_undecided(
    control: SelectionControlStore,
    experiment_id: str,
    task_id: str,
    *,
    expected_selection_id: Optional[str],
    candidate_attempt: Any = None,
    actor: str,
    actor_kind: str,
    provenance: str,
    reason: Optional[str] = None,
    store_id: Optional[str] = None,
) -> dict[str, Any]:
    """File "I looked and I am not choosing", optionally naming what was looked at.

    Records a decision and moves nothing. Without this, a task somebody
    reviewed and deliberately left alone is indistinguishable from one nobody
    opened, which is the difference between "no preference" and "no attention".
    """
    scope = _TaskScope(control, experiment_id, task_id, store_id=store_id,
                       require_store=candidate_attempt is not None)
    candidate = None
    if candidate_attempt is not None:
        attempt = _exact_int(candidate_attempt, "candidate_attempt")
        if attempt not in {int(row["attempt"]) for row in scope.attempt_rows()}:
            raise AttemptNotSelectable(
                scope.experiment_id, scope.task_id, attempt,
                "there is no recorded attempt with that number for this task",
            )
        candidate = {
            "experiment_id": scope.experiment_id,
            "task_id": scope.task_id,
            "attempt": attempt,
        }
    return _decide(
        scope,
        decision=DECISION_UNDECIDED,
        expected_selection_id=expected_selection_id,
        target=None,
        clear=False,
        candidate=candidate,
        actor=actor,
        actor_kind=actor_kind,
        provenance=provenance,
        reason=reason,
    )


__all__ = [
    "AttemptNotSelectable",
    "BEST_RUN_DECISIONS",
    "BestRunError",
    "DECISION_CLEAR",
    "DECISION_REPLACE",
    "DECISION_SELECT",
    "DECISION_UNDECIDED",
    "EVIDENCE_MISSING",
    "EVIDENCE_READABLE",
    "LABEL_BEST",
    "LABEL_EVIDENCE_MISSING",
    "LABEL_REFERENCE",
    "NoBestRun",
    "TASK_BEST_SCOPE",
    "TaskBestUnavailable",
    "attempt_is_finished",
    "best_run",
    "best_run_history",
    "clear_best_run",
    "leave_undecided",
    "list_task_attempts",
    "project_attempt",
    "reference_attempt",
    "select_best_run",
    "select_task_attempts",
    "task_run_summary",
    "task_scope_key",
]

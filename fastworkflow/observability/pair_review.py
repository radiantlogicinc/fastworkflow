"""Per-reviewer reviewed/not-reviewed progress for comparison pairs (`fix-9eg.4`).

Reviewing three runs against a pinned reference means walking a list of pairs
and needing to know, on return, which ones have already been looked at. That
is progress, not evidence and not feedback:

- **Not evidence.** It is a fact about a reviewer, recorded about an execution
  that may be sealed. It lives in the control tables of the workflow's live DB
  (`control.py`) for the same reasons `selection.py` gives: sealed copies carry
  no control table, so marking a pair never changes an archive, and pruning
  evidence must not erase what somebody did.
- **Not feedback.** "Reviewed with nothing to say" is a real and common
  outcome, so reviewed-ness is stored explicitly. Counting comments would
  report every clean pair as unreviewed, which is the one wrong answer this
  module exists to prevent.

Progress is keyed to the EXACT pair — `comparison.review_pair_key`, which is
derived from both `ExecutionRef`s including their pass ids. Pinning a
different reference therefore produces different pair keys, and the rows
recorded against the old reference keep their original meaning instead of
being relabelled. Nothing here is ever rewritten in place: the pointer row
carries the current state and every change appends to history.

Pair keys stay stable across sealing: `ExecutionRef.store_id` is the live
store's identity, and sealed copies carry that same identity.

This module owns no HTTP surface and no UI. It is the persistence a later
server slice calls.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Optional

from fastworkflow.observability import control
from fastworkflow.observability.comparison import ExecutionRef, review_pair_key
from fastworkflow.observability.store import FEEDBACK_PROVENANCES, _utcnow_iso

# Who marked it. Shares the feedback vocabulary so a human reviewer and a
# coding agent describe their origin the same way.
REVIEWER_KINDS = frozenset(FEEDBACK_PROVENANCES)

STATE_REVIEWED = "reviewed"
STATE_NOT_REVIEWED = "not_reviewed"

_MAX_TEXT = 4000


def _clean(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _require_text(value: Any, field: str) -> str:
    text = _clean(value)
    if not text:
        raise ValueError(f"{field} is required")
    if len(text) > _MAX_TEXT:
        raise ValueError(f"{field} must be at most {_MAX_TEXT} characters")
    return text


class PairReviewStore:
    """Reviewed/not-reviewed state per (pair, reviewer), with history.

    Thread/process-safe the same way `ObservabilityStore` is: every write is
    one short ``BEGIN IMMEDIATE`` transaction (`control.write`), so SQLite's
    file lock serialises two writers. ``store`` may be read-only for reads, or
    None when the workflow has no live DB: reads then answer empty and writes
    refuse with `control.ControlUnavailable`.
    """

    def __init__(self, store: Any) -> None:
        self.store = store

    # -- writes ----------------------------------------------------------

    def set_state(
        self,
        left_ref: Any,
        right_ref: Any,
        *,
        state: str,
        reviewer: str,
        reviewer_kind: str,
        note: Optional[str] = None,
    ) -> dict[str, Any]:
        """Record that this reviewer has, or has not, reviewed this exact pair.

        Idempotent in effect but not in history: re-marking an already
        reviewed pair appends an event, because "looked at it again" is a
        thing that happened. `note` is optional and is not feedback — feedback
        goes through the store's recording API with its own categories.
        """
        if state not in (STATE_REVIEWED, STATE_NOT_REVIEWED):
            raise ValueError(
                f"state must be {STATE_REVIEWED!r} or {STATE_NOT_REVIEWED!r}"
            )
        if reviewer_kind not in REVIEWER_KINDS:
            raise ValueError(
                "reviewer_kind must be one of " + ", ".join(sorted(REVIEWER_KINDS))
            )
        reviewer = _require_text(reviewer, "reviewer")
        note = _clean(note)
        if note is not None and len(note) > _MAX_TEXT:
            raise ValueError(f"note must be at most {_MAX_TEXT} characters")
        left = left_ref if isinstance(left_ref, ExecutionRef) else ExecutionRef.from_mapping(left_ref)
        right = right_ref if isinstance(right_ref, ExecutionRef) else ExecutionRef.from_mapping(right_ref)
        pair_key = review_pair_key(left, right)
        now = _utcnow_iso()
        with control.write(self.store) as conn:
            previous = conn.execute(
                "SELECT state, seq FROM pair_reviews WHERE pair_key=? AND reviewer=?",
                (pair_key, reviewer),
            ).fetchone()
            previous_state = None if previous is None else str(previous["state"])
            seq = (0 if previous is None else int(previous["seq"])) + 1
            conn.execute(
                """INSERT INTO pair_reviews
                   (pair_key, reviewer, reviewer_kind, left_ref_id, right_ref_id,
                    left_ref_json, right_ref_json, state, seq, note, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(pair_key, reviewer) DO UPDATE SET
                     reviewer_kind=excluded.reviewer_kind,
                     state=excluded.state, seq=excluded.seq,
                     note=excluded.note, updated_at=excluded.updated_at""",
                (
                    pair_key,
                    reviewer,
                    reviewer_kind,
                    left.ref_id(),
                    right.ref_id(),
                    json.dumps(left.as_dict(), sort_keys=True),
                    json.dumps(right.as_dict(), sort_keys=True),
                    state,
                    seq,
                    note,
                    now,
                ),
            )
            conn.execute(
                """INSERT INTO pair_review_events
                   (pair_key, reviewer, reviewer_kind, seq, previous_state,
                    state, note, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    pair_key,
                    reviewer,
                    reviewer_kind,
                    seq,
                    previous_state,
                    state,
                    note,
                    now,
                ),
            )
        return {
            "pair_key": pair_key,
            "reviewer": reviewer,
            "reviewer_kind": reviewer_kind,
            "state": state,
            "previous_state": previous_state,
            "seq": seq,
            "note": note,
            "updated_at": now,
        }

    def mark_reviewed(self, left_ref: Any, right_ref: Any, **kwargs: Any) -> dict[str, Any]:
        """Mark a pair reviewed. No comment is required and none is implied."""
        return self.set_state(left_ref, right_ref, state=STATE_REVIEWED, **kwargs)

    def clear_reviewed(self, left_ref: Any, right_ref: Any, **kwargs: Any) -> dict[str, Any]:
        """Return a pair to not-reviewed, keeping the history of both marks."""
        return self.set_state(left_ref, right_ref, state=STATE_NOT_REVIEWED, **kwargs)

    # -- reads -----------------------------------------------------------

    def state(self, pair_key: str, reviewer: str) -> dict[str, Any]:
        """One pair's state for one reviewer.

        An unrecorded pair answers `not_reviewed` with `recorded: False`, so a
        caller can tell "nobody marked this" from "somebody unmarked it".
        """
        found = control.rows(
            self.store,
            "SELECT * FROM pair_reviews WHERE pair_key=? AND reviewer=?",
            (pair_key, reviewer),
        )
        if not found:
            return {
                "pair_key": pair_key,
                "reviewer": reviewer,
                "state": STATE_NOT_REVIEWED,
                "recorded": False,
                "seq": 0,
                "updated_at": None,
                "note": None,
            }
        return {**found[0], "recorded": True}

    def progress(
        self, reviewer: str, pair_keys: Iterable[str]
    ) -> dict[str, Any]:
        """Reviewed counts over an explicit list of pairs, in the given order.

        The caller passes the pairs it is showing; this never invents a
        universe of pairs of its own, because which runs are "the others" is
        the caller's question, not the control tables'.
        """
        keys = [key for key in dict.fromkeys(pair_keys) if key]
        states = {key: self.state(key, reviewer) for key in keys}
        reviewed = [key for key in keys if states[key]["state"] == STATE_REVIEWED]
        remaining = [key for key in keys if states[key]["state"] != STATE_REVIEWED]
        return {
            "reviewer": reviewer,
            "pairs": len(keys),
            "reviewed": len(reviewed),
            "remaining": len(remaining),
            # Where "resume" goes: the first pair in the caller's order that is
            # not yet reviewed, or None when the list is done.
            "next_pair_key": remaining[0] if remaining else None,
            "states": [states[key] for key in keys],
        }

    def history(
        self, pair_key: str, *, reviewer: Optional[str] = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Events newest first. Append-only: nothing here is ever rewritten."""
        query = "SELECT * FROM pair_review_events WHERE pair_key=?"
        params: list[Any] = [pair_key]
        if reviewer is not None:
            query += " AND reviewer=?"
            params.append(reviewer)
        query += " ORDER BY event_id DESC LIMIT ?"
        params.append(int(limit))
        return control.rows(self.store, query, params)

    def reviews_for_reference(self, left_ref_id: str) -> list[dict[str, Any]]:
        """Every pair recorded against one pinned reference, any reviewer.

        Pinning a new reference does not touch these rows: a different
        reference has a different `left_ref_id`, so the old pairs stay
        readable exactly as they were recorded.
        """
        return control.rows(
            self.store,
            """SELECT * FROM pair_reviews WHERE left_ref_id=?
                ORDER BY updated_at, pair_key, reviewer""",
            (left_ref_id,),
        )


__all__ = [
    "REVIEWER_KINDS",
    "STATE_NOT_REVIEWED",
    "STATE_REVIEWED",
    "PairReviewStore",
]

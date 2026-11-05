"""A per-turn memo for the candidate lookups a turn repeats: the trained heads' answers and the Jev planning-command match.

The sub-task review gate asks for a sub-task's candidates (the fast-path check, then the draft), and the sub-task's run
asks again (its planner's template, its listing). Each lookup is a model or head call, so within one turn each runs once
per key. ``begin_turn`` starts a fresh memo, called where a turn begins; outside a turn (no memo begun) ``memo`` just
computes. A resumed turn (after ask_user) keeps the memo of its first call in the process, which is the same turn.
"""
from __future__ import annotations

import contextvars
from typing import Any, Callable, Hashable, Optional

_memo_var: contextvars.ContextVar[Optional[dict]] = contextvars.ContextVar("fw_turn_candidate_memo", default=None)


def begin_turn() -> None:
    """Start a fresh memo for the turn that is beginning."""
    _memo_var.set({})


def memo(key: Hashable, compute: Callable[[], Any]) -> Any:
    """The value for *key* in this turn's memo, computed once; computed every time when no turn has begun."""
    cache = _memo_var.get()
    if cache is None:
        return compute()
    if key not in cache:
        cache[key] = compute()
    return cache[key]

"""Reserved-class budget (R7.2) for intent training.

The budget changes how many escalation rows `model_pipeline_training.train()`
assembles for a context.

R7.2 — reserved-class budget
----------------------------

A context's label space is its real commands plus the reserved labels from
`fastworkflow.nlu_labels`: ``wildcard`` (every ancestor-context utterance not also
valid locally — the escalation class) and ``parameter_value`` (the seven bare-value
literals). Nothing currently bounds the escalation class, so on a deep hierarchy it
arrives already expanded by the synthetic generator at up to 91% of the rows (F8).

**The budget is for the escalation class. ``parameter_value`` is deliberately
exempt.** It is a fixed, hand-authored seven-row literal list; it has no growth
problem to budget against, and the cost invariant below would only ever bind on it in
a context whose real commands contribute fewer than seven rows in total. Shrinking it
was measured directly anyway — see the R7.3 note below — and it trades the bare-value
catch away faster than it buys routing accuracy, so the budget must not be pointed at
it.

**The budget is derived from a training-time target, not a class-balance target**
(spec §4 R7.2 as amended by AR4). The reasoning has exactly three steps and no fitted
constants:

1. Capping was tested directly on one workflow and produced no significant change on
   either axis (routing p = 0.25, escalation p = 0.38, decision D3). The one effect it
   was measured to have is cost: 44% less training time. Cost is therefore the only
   quantity the budget may be principled about.
2. Per-context cost is linear in row count — fixed epoch counts (12 tiny + 5 large),
   fixed batch size, two models per context. A row cap *is* a training-time target.
3. So state the target as a cost invariant and let the row count follow:

       reserved rows <= 1.0 * real-command rows

   At the default ratio of 1.0 the invariant reads "the reserved classes never
   outweigh the real data, so they can at most double a context's training cost".
   That is a nameable property, which is what AR4 asked for and what "3x the mean"
   was not. **1.0 is not claimed to be accuracy-optimal, and no accuracy claim
   attaches to any value of this ratio.** Expressing it as a ratio buys scale-freedom
   across workflows; the principle is in the invariant, not the number.

**Coverage floor.** The budget is raised — never lowered — to one utterance per
ancestor command. This floor *is* derived rather than fitted: it is the minimum that
satisfies the escalation requirement the wildcard class exists to serve. A command
with zero rows in a descendant's escalation class cannot be escalated to from there,
which is F3's failure mode reached by a different route. **When cost and coverage
conflict, coverage wins**, because cost is a preference and coverage is a requirement.

**Selection is round-robin** (decision D4), in two passes:

* a coverage pass taking one row from every ancestor command, which is what makes the
  floor true by construction rather than by arithmetic;
* a fill pass cycling over ancestor contexts and, within each, over that context's
  commands, one row per command per cycle. Surplus is therefore split evenly *per
  command*, not per context: an ancestor with 40 intents does get more surplus than
  one with 2, because it has 40 intents to represent. What round-robin prevents is a
  *verbose* command or context — one whose generator happened to emit 200 rows —
  taking the budget from the others, which is the failure mode actually observed.

Within a command, rows are taken earliest-first. `_get_utterances` returns
``[command_name] + seeds + synthetic`` (`generate_synthetic.py`), so earliest-first
keeps the hand-written seeds and discards the synthetic tail. Rejected: uniform random
sampling and head truncation over the flattened set — both drop whole intents.

R7.3 class weighting was measured against an unweighted control and produced no
corrected benefit on routing or escalation. It is intentionally not shipped: the
trainer keeps HuggingFace's native unweighted cross-entropy path.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

# R7.3's optional torch-backed weighting path was measured null and intentionally
# not shipped. This module stays pure data manipulation and importable without torch.
# Production always applies the fixed cap; there is deliberately no uncapped branch.


def reserved_class_budget(
    own_row_count: int,
    coverage_floor: int,
    ratio: float,
) -> int:
    """Rows the reserved classes of one context may contribute.

    `own_row_count` is the number of rows the context's real commands contributed;
    `coverage_floor` is the number of distinct sources that must each keep at least
    one row (for the escalation class, the number of ancestor commands).

    The cost invariant gives `ratio * own_row_count`; the coverage floor raises it
    when the two conflict, because coverage is a requirement of the escalation
    design and cost is a preference (see module docstring).
    """
    floor = max(0, int(coverage_floor))
    cost_budget = int(max(0.0, ratio) * max(0, int(own_row_count)))
    return max(cost_budget, floor)


def select_reserved_rows(
    rows_by_group: Mapping[str, Mapping[str, Sequence[str]]],
    budget: int,
    always_include: Iterable[str] = (),
    exclude: Iterable[str] = (),
) -> list[str]:
    """Round-robin selection of at most `budget` rows, preserving coverage.

    `rows_by_group` is ``{group: {source: [row, ...]}}`` — for the escalation class,
    ``{ancestor_context: {command: utterances}}``. Two passes:

    1. **Coverage.** One row from every source, in sorted (group, source) order. This
       pass ignores `budget`: dropping a source's last row is what round-robin exists
       to prevent, so the floor is enforced by construction rather than by trusting
       the caller to have computed `reserved_class_budget` correctly.
    2. **Fill.** Cycle over groups, and within each group over its sources, taking one
       unused row per source per cycle. Every source therefore gets an equal share of
       the surplus regardless of how many rows it has available, which is what stops a
       verbose source from spending the budget the others need.

    Rows are taken earliest-first within a source, which preserves hand-written seeds
    over synthetic expansions. `always_include` rows are emitted first and are never
    dropped *by the budget*, though they are still subject to `exclude`. `exclude`
    removes rows that are valid in the local context — an utterance that means
    something *here* must not also train the "ask my parent" class.

    Returns rows in selection order, deduplicated. The caller sorts if it wants a
    stable training-row order.
    """
    excluded = set(exclude)
    selected: list[str] = []
    seen: set[str] = set()

    def _take(row: str) -> bool:
        if row in seen or row in excluded:
            return False
        seen.add(row)
        selected.append(row)
        return True

    for row in always_include:
        _take(row)

    # Materialise as ordered lists once; both passes walk the same structure and the
    # fill pass needs a stable cursor per source.
    groups: list[tuple[str, list[tuple[str, list[str]]]]] = [
        (
            group,
            [(source, list(rows)) for source, rows in sorted(rows_by_group[group].items())],
        )
        for group in sorted(rows_by_group)
    ]

    cursors: dict[tuple[str, str], int] = {}

    for group, sources in groups:
        for source, rows in sources:
            index = 0
            while index < len(rows) and not _take(rows[index]):
                index += 1
            cursors[(group, source)] = index + 1 if index < len(rows) else index

    if budget <= len(selected):
        return selected

    progressed = True
    while progressed and len(selected) < budget:
        progressed = False
        for group, sources in groups:
            for source, rows in sources:
                if len(selected) >= budget:
                    break
                index = cursors[(group, source)]
                while index < len(rows) and not _take(rows[index]):
                    index += 1
                if index < len(rows):
                    progressed = True
                    index += 1
                cursors[(group, source)] = index
            if len(selected) >= budget:
                break

    return selected


def reserved_candidate_counts(
    rows_by_group: Mapping[str, Mapping[str, Sequence[str]]],
    exclude: Iterable[str] = (),
) -> tuple[int, int]:
    """Return raw and de-duplicated candidate counts for one reserved class.

    The raw count is the number of ancestor-list entries before local rows are removed
    or duplicate utterance text is collapsed. The de-duplicated count applies both
    operations, matching the candidate population from which `select_reserved_rows`
    selects. Always-included rows are reported separately by the trainer because they
    are not ancestor candidates and historically used a different denominator.
    """
    excluded = set(exclude)
    rows = [
        row
        for sources in rows_by_group.values()
        for source_rows in sources.values()
        for row in source_rows
    ]
    return len(rows), len(set(rows) - excluded)


def group_ancestor_utterances(
    ancestor_contexts: Iterable[str],
    cache: Mapping[str, Mapping[str, Sequence[str]]],
    skip_labels: Iterable[str] = (),
) -> dict[str, dict[str, list[str]]]:
    """Regroup the trainer's flat utterance cache by ancestor context and command.

    `cache_ancestor_utterances` already populates ``cache[ancestor_ctx][cmd]`` but
    returns a flattened set, which throws away exactly the grouping round-robin needs.
    This reads the grouping back out of the cache the trainer already holds, so the
    cap needs no second pass over the generator and no change to how utterances are
    produced.

    Ancestor contexts absent from the cache are skipped rather than raising: a context
    can legitimately contribute nothing, and a missing key here must not be the thing
    that fails a training run.
    """
    skipped = set(skip_labels)
    grouped: dict[str, dict[str, list[str]]] = {}
    for ancestor_ctx in ancestor_contexts:
        commands = cache.get(ancestor_ctx)
        if not commands:
            continue
        by_command = {
            cmd: list(rows)
            for cmd, rows in commands.items()
            if rows and cmd.split("/")[-1] not in skipped
        }
        if by_command:
            grouped[ancestor_ctx] = by_command
    return grouped


def coverage_floor_of(rows_by_group: Mapping[str, Mapping[str, Sequence[str]]]) -> int:
    """Number of distinct sources across all groups — the derived coverage floor."""
    return sum(len(sources) for sources in rows_by_group.values())

"""Score the two head layouts of the scoped lookup (Phase 9 step 3, fix-3rsq.9.5, design 3.9).

EXPERIMENT TOOL. ``probe`` runs already-trained heads; it trains nothing. The two layouts are
``flattened`` (today: a head learns its own and inherited commands, and the runtime climbs the
live parent chain while a head answers "not here") and ``scoped`` (each head learns only its own
commands, and the walk visits the scoped path, the records of the chain with their bases).

A **position** is a live chain: a context O plus one parent path from O up to the top of
``context_hierarchy_model.json``. A parent ``*`` ends the chain. Every held-out utterance of a
command C is scored at every position. C is *in scope* at a position when its owning context is
on that position's scoped path; the expected binding is the chain record at which the owner was
first reached. Otherwise the case is out of scope and the expected outcome is "no command".

Subcommands::

    python -m fastworkflow.train.score_lookup probe  --workflow W --out P [--train-timings T]
    python -m fastworkflow.train.score_lookup score  --workflow W --probe P --layout {flattened,scoped,flattened_global_first} --out R [--root-context X]
    python -m fastworkflow.train.score_lookup rescore --results R0 --probe P --layout L --out R [--root-context X]
    python -m fastworkflow.train.score_lookup compare --base R1[,R2,R3] --arm R1[,R2,R3] [--arm-layout L] --out J --md M

``probe`` reads ``___command_info/lookup_holdout.json`` and the version's heads; ``score`` and
``compare`` read JSON only and load no model. ``score`` reads ``command_directory.json`` (the
trainer's snapshot) and the two JSON models read-only; it never calls ``CommandContextModel.load``
because that can rewrite the snapshot.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Any, Iterable, Optional, Sequence

from fastworkflow.nlu_labels import PARAMETER_VALUE_LABEL, WILDCARD_LABEL, is_non_routable, label_of
from fastworkflow.train.score_not_here import Z_95, mcnemar_exact, wilson_interval

SCHEMA_VERSION = 1
GLOBAL = "*"
ROOT = "root"  # binding of the global scope: the root object (design 3.9, "Binding")
CORE = "core"  # owner of the core commands: every context carries them
LAYOUTS = ("flattened", "scoped")
BASE_LAYOUT = "flattened"
ARM_LAYOUT = "scoped"
#: The flattened heads, walked with the global head asked first (``walk_flattened_global_first``).
#: It is scored, never trained: its heads are the flattened heads. ``LAYOUTS`` keeps the two step 3
#: layouts, because the step 3 runner plans and compares exactly those.
GLOBAL_FIRST_LAYOUT = "flattened_global_first"
SCORABLE_LAYOUTS = (*LAYOUTS, GLOBAL_FIRST_LAYOUT)
TRAINED_AS = {"flattened": "flattened", "scoped": "scoped", GLOBAL_FIRST_LAYOUT: "flattened"}
REWEIGHT_GLOBAL_SHARE = 0.05  # the sensitivity check: global-command cases at 5% of in-scope weight
HOLDOUT_FILENAME = "lookup_holdout.json"
SNAPSHOT_FILENAME = "command_directory.json"

KIND_OWN = "own"
KIND_BASE = "base"
KIND_ANCESTOR = "ancestor"
KIND_GLOBAL = "global"
KIND_OFF_PATH = "off_path"
KINDS = (KIND_OWN, KIND_BASE, KIND_ANCESTOR, KIND_GLOBAL, KIND_OFF_PATH)

# The owner's bars (step 3 spec). Differences are arm (scoped) minus base (flattened).
E2E_MIN_DIFFERENCE = -0.02        # end-to-end accuracy: lower bound of the difference >= -2 points
WRONG_ACCEPT_MAX_RISE = 0.02      # wrong accepts: upper bound of the difference <= +2 points
TOP1_MIN_DIFFERENCE = -0.01       # in-scope top-1: lower bound of the difference >= -1 point


# ---------------------------------------------------------------- scopes and paths

def ordered_bases(context: str, inheritance: dict[str, Sequence[str]]) -> list[str]:
    """Every base the context inherits, transitively, depth-first in declared order, each once.

    Mirrors ``CommandContextModel._ordered_bases_from_map``. Unknown contexts give ``[]``.
    """
    ordered: list[str] = []
    seen = {context}

    def walk(name: str) -> None:
        for base in inheritance.get(name) or []:
            if base in seen:
                continue
            seen.add(base)
            ordered.append(base)
            walk(base)

    walk(context)
    return ordered


def positions(hierarchy: dict) -> list[dict]:
    """Every live chain (context, one parent path to the top), deduplicated, in declared order.

    Each record is ``{"id", "context", "chain", "depth"}``; ``chain`` lists the records from the
    position's context upward and never contains ``*``. A simple path only: a parent already on
    the chain is skipped. A parent ``*`` ends the chain. A context with no declared parent is a
    top on its own. A parent that is neither ``*`` nor a key of the hierarchy is an error.
    """
    found: list[tuple[str, ...]] = []

    def walk(chain: tuple[str, ...]) -> None:
        parents = list(hierarchy.get(chain[-1], {}).get("parent") or [])
        if not parents:
            found.append(chain)
            return
        for parent in parents:
            if parent == GLOBAL:
                found.append(chain)
            elif parent in chain:
                continue
            elif parent not in hierarchy:
                raise ValueError(f"context '{parent}' is a parent of '{chain[-1]}' but has no entry in the hierarchy")
            else:
                walk(chain + (parent,))

    for context in hierarchy:
        walk((context,))
    unique: list[tuple[str, ...]] = []
    for chain in found:
        if chain not in unique:
            unique.append(chain)
    return [{"id": ">".join(chain), "context": chain[0], "chain": list(chain), "depth": len(chain)}
            for chain in unique]


def scoped_path(chain: Sequence[str], inheritance: dict[str, Sequence[str]]) -> list[tuple[str, str]]:
    """The scope sequence for a chain, as ``(scope, binding)`` pairs (design 3.9, "The path").

    For each record on the chain from the position upward: the record's context, then its bases
    depth-first in declared order, skipping any scope already visited. The binding of a scope is
    the chain record it was first reached at. The global scope ``*`` comes last, bound to ROOT.
    """
    bound: dict[str, str] = {}
    order: list[tuple[str, str]] = []
    for record in chain:
        for scope in [record, *ordered_bases(record, inheritance)]:
            if scope not in bound and scope != GLOBAL:
                bound[scope] = record
                order.append((scope, record))
    order.append((GLOBAL, ROOT))
    return order


def climb_path(chain: Sequence[str], root_context: Optional[str] = None) -> list[tuple[str, str]]:
    """The heads the flattened runtime asks, in order, as ``(head, binding)`` pairs.

    Mirrors ``_commands/wildcard.py``: the position's own head is asked first. The climb then asks
    the parent record's head while the current record is not the root (the loop's condition). At
    the top of the chain the parent is ``None`` (every top in the measured workflow returns ``None`` from
    ``get_parent``), so the global head ``*`` is asked after a non-root top. The climb stops after
    asking the root record itself, and never asks ``*`` when the root is on the chain.
    """
    asks: list[tuple[str, str]] = []
    for record in chain:
        asks.append((record, record))
        if record == root_context:
            return asks
    asks.append((GLOBAL, ROOT))
    return asks


def scope_kind(owner: str, binding: str, context: str) -> str:
    """Own / base / ancestor / global, from the owner and where its scope was bound."""
    if owner == GLOBAL:
        return KIND_GLOBAL
    if owner == context:
        return KIND_OWN
    if binding == context:
        return KIND_BASE
    return KIND_ANCESTOR


# ---------------------------------------------------------------- ownership and expectation

def command_owners(names: Iterable[str], core: Iterable[str]) -> dict[str, str]:
    """Qualified command name -> owning context (``*`` for a global command), core names -> CORE.

    Planning commands are not in *names* (the caller filters them, as ``get_execution_commands``
    does). A reserved label (``wildcard``) is never a command and is left out. A command's owner
    is the context whose ``"/"`` list holds it: the prefix before the last slash, or ``*``.
    """
    core_set = set(core)
    owners: dict[str, str] = {}
    for name in names:
        if is_non_routable(name):
            continue
        if name in core_set:
            owners[name] = CORE
            continue
        owner, sep, _ = name.rpartition("/")
        owners[name] = owner if sep else GLOBAL
    for name in core_set:
        if not is_non_routable(name):
            owners.setdefault(name, CORE)
    return owners


def owner_of(label: str, owners: dict[str, str]) -> Optional[str]:
    """The owner of a held-out label, or None when the label names no command of the workflow."""
    if is_non_routable(label):
        return None
    return owners.get(label)


def expected(case: dict, position: dict, inheritance: dict[str, Sequence[str]]) -> dict[str, Any]:
    """Whether a case is in scope at a position, and the binding it should get.

    Core commands are owned by every context: in scope, bound to the position's context.
    Otherwise in scope iff the owner is on the scoped path; the binding is the scope's binding.
    """
    owner = case.get("owner")
    if owner is None:
        raise ValueError(f"case {case.get('label')!r} has no owner; skip it before scoring")
    context = position["context"]
    if owner == CORE:
        return {"in_scope": True, "binding": context, "kind": KIND_OWN}
    bindings = dict(scoped_path(position["chain"], inheritance))
    if owner not in bindings:
        return {"in_scope": False, "binding": None, "kind": KIND_OFF_PATH}
    binding = bindings[owner]
    return {"in_scope": True, "binding": binding, "kind": scope_kind(owner, binding, context)}


# ---------------------------------------------------------------- walks over the prediction cache

def _head(cache: dict, head: str, utterance: str) -> dict:
    try:
        return cache[head][utterance]
    except KeyError:
        raise KeyError(f"the probe has no prediction for head {head!r} on {utterance!r}; re-run probe") from None


def _named(label: str, binding: str, asked: list[str]) -> dict[str, Any]:
    return {"kind": "named", "named": label, "bound": binding, "candidates": [], "asked": asked}


def _asked(candidates: list[str], asked: list[str]) -> dict[str, Any]:
    return {"kind": "asked", "named": None, "bound": None, "candidates": candidates, "asked": asked}


def _none(asked: list[str]) -> dict[str, Any]:
    return {"kind": "none", "named": None, "bound": None, "candidates": [], "asked": asked}


def _primary(scopes: Sequence[tuple[str, str]], utterance: str, cache: dict) -> dict[str, Any]:
    """Primary walk over the top label of each head. Each head is asked at most once."""
    asked: list[str] = []
    for head, binding in scopes:
        if head in asked:
            continue
        asked.append(head)
        top = str(_head(cache, head, utterance)["top_label"])
        if not is_non_routable(top):
            return _named(top, binding, asked)
        if label_of(top) == PARAMETER_VALUE_LABEL:
            return _none(asked)  # a bare value never starts a climb (design 2.18)
        # wildcard: "not here", on to the next head
    return _none(asked)


def walk_flattened(position: dict, utterance: str, cache: dict, root_context: Optional[str] = None) -> dict[str, Any]:
    """Primary walk of the flattened layout (the runtime climb, top labels only)."""
    return _primary(climb_path(position["chain"], root_context), utterance, cache)


def walk_scoped(position: dict, utterance: str, cache: dict, inheritance: dict[str, Sequence[str]]) -> dict[str, Any]:
    """Primary walk of the scoped layout (the scoped path, top labels only)."""
    return _primary(scoped_path(position["chain"], inheritance), utterance, cache)


def _secondary(scopes: Sequence[tuple[str, str]], utterance: str, cache: dict, *, keep_candidates: bool) -> dict[str, Any]:
    """Confidence-aware walk.

    An unsure head (not confident) ends the walk among its own labels, unless
    ``keep_candidates`` is set and "not here" (wildcard) is among its top-3: then its commands are
    kept as candidates and the walk goes on. The walk ends at the next head that names a command,
    confidently or not; if candidates were kept, the user is asked once among them and that head's
    command. A confident "not here" moves on; a confident bare value ends the walk.
    """
    kept: list[str] = []
    asked: list[str] = []
    for head, binding in scopes:
        if head in asked:
            continue
        asked.append(head)
        entry = _head(cache, head, utterance)
        top = str(entry["top_label"])
        topk = [str(label) for label in entry["topk"]]
        commands = [label for label in topk if not is_non_routable(label)]
        if not entry["confident"]:
            wildcard_listed = any(label_of(label) == WILDCARD_LABEL for label in topk)
            if keep_candidates and wildcard_listed:
                kept.extend(label for label in commands if label not in kept)
                continue
            return _asked(kept + [label for label in commands if label not in kept], asked)
        if label_of(top) == PARAMETER_VALUE_LABEL:
            return _asked(kept, asked) if kept else _none(asked)
        if is_non_routable(top):
            continue
        if kept:
            return _asked(kept + ([top] if top not in kept else []), asked)
        return _named(top, binding, asked)
    return _asked(kept, asked) if kept else _none(asked)


def walk_flattened_confident(position: dict, utterance: str, cache: dict, root_context: Optional[str] = None) -> dict[str, Any]:
    """Secondary walk, flattened: an unsure head always ends the walk (today's suppression)."""
    return _secondary(climb_path(position["chain"], root_context), utterance, cache, keep_candidates=False)


def walk_scoped_confident(position: dict, utterance: str, cache: dict, inheritance: dict[str, Sequence[str]]) -> dict[str, Any]:
    """Secondary walk, scoped: an unsure head with "not here" in its top-3 keeps its candidates."""
    return _secondary(scoped_path(position["chain"], inheritance), utterance, cache, keep_candidates=True)


# ---------------------------------------------------------------- outcomes

def primary_outcome(in_scope: bool, label: str, expected_binding: Optional[str], result: dict) -> str:
    """correct / wrong_binding / wrong_command / declined (in scope); wrong_accept / declined (out)."""
    if result["kind"] != "named":
        return "declined"
    if not in_scope:
        return "wrong_accept"
    if result["named"] != label:
        return "wrong_command"
    return "correct" if result["bound"] == expected_binding else "wrong_binding"


def secondary_outcome(in_scope: bool, label: str, expected_binding: Optional[str], result: dict) -> str:
    """Confidence-aware outcome: correct_confident / correct_in_candidates / wrong / declined."""
    if in_scope:
        if result["kind"] == "named":
            ok = result["named"] == label and result["bound"] == expected_binding
            return "correct_confident" if ok else "wrong"
        if result["kind"] == "asked":
            return "correct_in_candidates" if label in result["candidates"] else "declined"
        return "declined"
    return "wrong" if result["kind"] == "named" else "declined"


def _global_first_hit(utterance: str, cache: dict) -> Optional[str]:
    """The command the global head names, when it is confident and names a command; else None."""
    entry = _head(cache, GLOBAL, utterance)
    top = str(entry["top_label"])
    if entry["confident"] and not is_non_routable(top):
        return top
    return None


def _unique(labels: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(labels))


def walk_flattened_global_first(position: dict, utterance: str, cache: dict,
                                root_context: Optional[str] = None) -> dict[str, Any]:
    """Primary walk of the flattened_global_first layout.

    The global head ``*`` is asked first. Its top label is a command and it is confident: that
    command is named, bound to ROOT, and the walk stops. Otherwise the walk is exactly
    ``walk_flattened``, whose climb still ends at ``*``. ``asked`` lists each head once, in the
    order the walk first asks it, so ``*`` is not listed twice.
    """
    top = _global_first_hit(utterance, cache)
    if top is not None:
        return _named(top, ROOT, [GLOBAL])
    result = walk_flattened(position, utterance, cache, root_context)
    return {**result, "asked": _unique([GLOBAL, *result["asked"]])}


def walk_flattened_global_first_confident(position: dict, utterance: str, cache: dict,
                                          root_context: Optional[str] = None) -> dict[str, Any]:
    """Secondary walk of the same layout: the same first ask of ``*``, then ``walk_flattened_confident``."""
    top = _global_first_hit(utterance, cache)
    if top is not None:
        return _named(top, ROOT, [GLOBAL])
    result = walk_flattened_confident(position, utterance, cache, root_context)
    return {**result, "asked": _unique([GLOBAL, *result["asked"]])}


def _walks(layout: str, position: dict, utterance: str, cache: dict,
           inheritance: dict[str, Sequence[str]], root_context: Optional[str]) -> tuple[dict, dict]:
    """The layout's primary walk and its secondary (confidence-aware) walk."""
    if layout == "flattened":
        return (walk_flattened(position, utterance, cache, root_context),
                walk_flattened_confident(position, utterance, cache, root_context))
    if layout == "scoped":
        return (walk_scoped(position, utterance, cache, inheritance),
                walk_scoped_confident(position, utterance, cache, inheritance))
    return (walk_flattened_global_first(position, utterance, cache, root_context),
            walk_flattened_global_first_confident(position, utterance, cache, root_context))


def evaluate_case(case: dict, position: dict, layout: str, cache: dict,
                  inheritance: dict[str, Sequence[str]], root_context: Optional[str] = None) -> dict[str, Any]:
    """One (held-out utterance, position) result, in the layout's primary walk plus the secondary."""
    if layout not in SCORABLE_LAYOUTS:
        raise ValueError(f"unknown layout {layout!r}")
    exp = expected(case, position, inheritance)
    return score_expected(exp, case, position, layout, cache, inheritance, root_context)


def score_expected(exp: dict[str, Any], case: dict, position: dict, layout: str, cache: dict,
                   inheritance: dict[str, Sequence[str]], root_context: Optional[str] = None) -> dict[str, Any]:
    """One result from an expectation already known (computed, or read from a score file)."""
    in_scope, binding, label, utterance = exp["in_scope"], exp["binding"], case["label"], case["utterance"]
    result, secondary = _walks(layout, position, utterance, cache, inheritance, root_context)
    outcome = primary_outcome(in_scope, label, binding, result)
    return {
        "utterance": utterance,
        "label": label,
        "owner": case["owner"],
        "position": position["id"],
        "depth": position["depth"],
        "in_scope": in_scope,
        "expected_binding": binding,
        "owner_kind": exp["kind"],
        "layout": layout,
        "named": result["named"],
        "bound": result["bound"],
        "asked": result["asked"],
        "outcome": outcome,
        "correct": bool(in_scope and outcome == "correct"),
        "top1": bool(in_scope and result["named"] == label),
        "wrong_accept": bool((not in_scope) and result["kind"] == "named"),
        "wrong_decline": bool(in_scope and result["kind"] == "none"),
        "secondary": {
            "outcome": secondary_outcome(in_scope, label, binding, secondary),
            "kind": secondary["kind"],
            "named": secondary["named"],
            "bound": secondary["bound"],
            "candidates": secondary["candidates"],
            "asked": secondary["asked"],
        },
    }


# ---------------------------------------------------------------- statistics

def _rate(count: int, trials: int) -> dict[str, Any]:
    interval = wilson_interval(count, trials)
    return {"count": count, "trials": trials, "rate": (count / trials) if trials else None,
            "wilson95": list(interval) if interval else None}


def paired_wald(pairs: Sequence[tuple[int, int]]) -> dict[str, Any]:
    """Paired difference of two binary indicators (arm x, base y): arm minus base.

    Same formula as ``not_here_step1.paired_difference`` (paired Wald 95% on n10/n01), with the
    exact McNemar p-value on the same discordant pairs.
    """
    n = len(pairs)
    if n == 0:
        return {"n": 0, "arm_rate": None, "base_rate": None, "difference": None,
                "lower": None, "upper": None, "arm_only": 0, "base_only": 0, "p_two_sided": None}
    n10 = sum(1 for x, y in pairs if x == 1 and y == 0)
    n01 = sum(1 for x, y in pairs if x == 0 and y == 1)
    difference = (n10 - n01) / n
    variance = max(0.0, (n10 + n01 - (n10 - n01) ** 2 / n) / (n * n))
    half = Z_95 * variance ** 0.5
    return {
        "n": n,
        "arm_rate": sum(x for x, _ in pairs) / n,
        "base_rate": sum(y for _, y in pairs) / n,
        "difference": difference,
        "lower": difference - half,
        "upper": difference + half,
        "arm_only": n10,
        "base_only": n01,
        "p_two_sided": mcnemar_exact(n10, n01),
    }


def _pairs(base_cases: Sequence[dict], arm_cases: Sequence[dict], field: str, *, in_scope: bool,
           only: Optional[tuple[str, Any]] = None) -> list[tuple[int, int]]:
    """Per-case indicator pairs (arm, base) for the cases of one scope kind, optionally one group."""
    pairs = []
    for base, arm in zip(base_cases, arm_cases):
        if bool(base["in_scope"]) != in_scope:
            continue
        if only is not None and base[only[0]] != only[1]:
            continue
        pairs.append((int(bool(arm[field])), int(bool(base[field]))))
    return pairs


def case_key(case: dict) -> tuple:
    return (case["utterance"], case["label"], case["position"], bool(case["in_scope"]))


def pool_seeds(results_by_seed: Sequence[dict]) -> list[dict]:
    """Concatenate the seeds' case lists in seed order, after checking every seed has the same cases in order."""
    if not results_by_seed:
        raise ValueError("no runs to pool")
    reference = [case_key(case) for case in results_by_seed[0]["cases"]]
    pooled: list[dict] = []
    for index, run in enumerate(results_by_seed):
        keys = [case_key(case) for case in run["cases"]]
        if keys != reference:
            raise ValueError(f"run {index + 1} scores different cases (or in another order) from run 1")
        pooled.extend(run["cases"])
    return pooled


def compare_cases(base_cases: Sequence[dict], arm_cases: Sequence[dict]) -> None:
    """Raise unless both lists score the same cases in the same order (the pairing precondition)."""
    if len(base_cases) != len(arm_cases):
        raise ValueError(f"the layouts score different numbers of cases ({len(base_cases)} vs {len(arm_cases)})")
    for index, (base, arm) in enumerate(zip(base_cases, arm_cases)):
        if case_key(base) != case_key(arm):
            raise ValueError(f"case {index} differs between the layouts: {case_key(base)} vs {case_key(arm)}")


def paired_metrics(base_cases: Sequence[dict], arm_cases: Sequence[dict]) -> dict[str, Any]:
    """The four paired blocks: end-to-end and in-scope top-1 (in scope), wrong accepts (out),
    wrong declines (in, reported with no bar)."""
    compare_cases(base_cases, arm_cases)
    return {
        "e2e": paired_wald(_pairs(base_cases, arm_cases, "correct", in_scope=True)),
        "top1": paired_wald(_pairs(base_cases, arm_cases, "top1", in_scope=True)),
        "wrong_accepts": paired_wald(_pairs(base_cases, arm_cases, "wrong_accept", in_scope=False)),
        "wrong_declines": paired_wald(_pairs(base_cases, arm_cases, "wrong_decline", in_scope=True)),
    }


def reweighted_in_scope(base_cases: Sequence[dict], arm_cases: Sequence[dict],
                        group_share: float = REWEIGHT_GLOBAL_SHARE, group: str = KIND_GLOBAL) -> dict[str, Any]:
    """Point estimates only (no interval, no bar): in-scope paired metrics with the global-command
    cases at ``group_share`` of the in-scope weight.

    Each in-scope case of the group gets weight ``group_share / n_group``; each other in-scope case
    gets ``(1 - group_share) / n_rest``. The pooled case lists are the same as ``paired_metrics``'
    (the caller checks the pairing). Out-of-scope cases carry no in-scope weight and are not used.
    """
    compare_cases(base_cases, arm_cases)
    rows = [(base, arm) for base, arm in zip(base_cases, arm_cases) if base["in_scope"]]
    in_group = [(b, a) for b, a in rows if b["owner_kind"] == group]
    rest = [(b, a) for b, a in rows if b["owner_kind"] != group]
    share_before = len(in_group) / len(rows) if rows else None
    if not in_group or not rest:
        raise ValueError(f"cannot reweight: {len(in_group)} {group} cases and {len(rest)} other in-scope cases")
    weighted = [(b, a, group_share / len(in_group)) for b, a in in_group]
    weighted += [(b, a, (1.0 - group_share) / len(rest)) for b, a in rest]
    total = sum(w for _, _, w in weighted)

    def block(field: str) -> dict[str, Any]:
        arm_rate = sum(w * int(bool(a[field])) for _, a, w in weighted) / total
        base_rate = sum(w * int(bool(b[field])) for b, _, w in weighted) / total
        return {"arm_rate": arm_rate, "base_rate": base_rate, "difference": arm_rate - base_rate}

    return {
        "group": group,
        "group_share": group_share,
        "group_share_before": share_before,
        "group_cases": len(in_group),
        "other_in_scope_cases": len(rest),
        "e2e": block("correct"),
        "top1": block("top1"),
        "unchanged": "wrong_accepts is out of scope: reweighting in-scope cases does not move it",
    }


def evaluate_layout(paired: dict[str, Any]) -> dict[str, Any]:
    """The owner's bars. Passes when all three hold (wrong declines are reported only)."""
    e2e, accepts, top1 = paired["e2e"], paired["wrong_accepts"], paired["top1"]
    bars = {
        "e2e": {"rule": f"lower bound of the difference >= {E2E_MIN_DIFFERENCE}",
                "passes": e2e["lower"] is not None and e2e["lower"] >= E2E_MIN_DIFFERENCE},
        "wrong_accepts": {"rule": f"upper bound of the difference <= {WRONG_ACCEPT_MAX_RISE}",
                          "passes": accepts["upper"] is not None and accepts["upper"] <= WRONG_ACCEPT_MAX_RISE},
        "top1": {"rule": f"lower bound of the difference >= {TOP1_MIN_DIFFERENCE}",
                 "passes": top1["lower"] is not None and top1["lower"] >= TOP1_MIN_DIFFERENCE},
    }
    return {"passes": all(block["passes"] for block in bars.values()), "bars": bars}


def breakdowns(base_cases: Sequence[dict], arm_cases: Sequence[dict]) -> dict[str, Any]:
    """Paired blocks per chain depth and per owner-scope kind (own/base/ancestor/global/off_path)."""
    out: dict[str, Any] = {"by_depth": {}, "by_owner_kind": {}}
    for depth in sorted({case["depth"] for case in base_cases}):
        out["by_depth"][str(depth)] = _group_block(base_cases, arm_cases, "depth", depth)
    for kind in KINDS:
        if any(case["owner_kind"] == kind for case in base_cases):
            out["by_owner_kind"][kind] = _group_block(base_cases, arm_cases, "owner_kind", kind)
    return out


def _group_block(base_cases, arm_cases, field, value) -> dict[str, Any]:
    group = (field, value)
    return {
        "e2e": paired_wald(_pairs(base_cases, arm_cases, "correct", in_scope=True, only=group)),
        "top1": paired_wald(_pairs(base_cases, arm_cases, "top1", in_scope=True, only=group)),
        "wrong_accepts": paired_wald(_pairs(base_cases, arm_cases, "wrong_accept", in_scope=False, only=group)),
        "wrong_declines": paired_wald(_pairs(base_cases, arm_cases, "wrong_decline", in_scope=True, only=group)),
        "cases": sum(1 for case in base_cases if case[field] == value),
    }


# ---------------------------------------------------------------- summaries

def summarize_cases(cases: Sequence[dict]) -> dict[str, Any]:
    """Pooled counts and rates for one layout's cases (Wilson 95% on each rate)."""
    n_in = sum(1 for case in cases if case["in_scope"])
    n_out = len(cases) - n_in
    secondary = Counter()
    for case in cases:
        secondary[(("in" if case["in_scope"] else "out"), case["secondary"]["outcome"])] += 1
    return {
        "cases": len(cases),
        "in_scope_n": n_in,
        "out_of_scope_n": n_out,
        "e2e": _rate(sum(case["correct"] for case in cases), n_in),
        "in_scope_top1": _rate(sum(case["top1"] for case in cases), n_in),
        "wrong_accepts": _rate(sum(case["wrong_accept"] for case in cases), n_out),
        "wrong_declines": _rate(sum(case["wrong_decline"] for case in cases), n_in),
        "secondary": {
            "in_scope": {name: secondary[("in", name)] for name in
                         ("correct_confident", "correct_in_candidates", "wrong", "declined")},
            "out_of_scope": {name: secondary[("out", name)] for name in ("wrong", "declined")},
        },
        "by_depth": {str(d): _summary_group(cases, "depth", d)
                     for d in sorted({case["depth"] for case in cases})},
        "by_owner_kind": {k: _summary_group(cases, "owner_kind", k)
                          for k in KINDS if any(case["owner_kind"] == k for case in cases)},
    }


def _summary_group(cases: Sequence[dict], field: str, value: Any) -> dict[str, Any]:
    group = [case for case in cases if case[field] == value]
    n_in = sum(1 for case in group if case["in_scope"])
    n_out = len(group) - n_in
    return {"cases": len(group), "in_scope_n": n_in, "out_of_scope_n": n_out,
            "e2e": _rate(sum(case["correct"] for case in group), n_in),
            "wrong_accepts": _rate(sum(case["wrong_accept"] for case in group), n_out)}


def _fmt_rate(block: dict[str, Any]) -> str:
    if not block["trials"]:
        return "n/a"
    lo, hi = block["wilson95"]
    return f"{block['count']}/{block['trials']} = {100 * block['rate']:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]"


def format_score_summary(summary: dict[str, Any], layout: str) -> str:
    lines = [f"Lookup scoring, layout {layout} ({summary['cases']} cases: "
             f"{summary['in_scope_n']} in scope, {summary['out_of_scope_n']} out of scope)", "",
             f"end-to-end (in scope):  {_fmt_rate(summary['e2e'])}",
             f"in-scope top-1:         {_fmt_rate(summary['in_scope_top1'])}",
             f"wrong accepts:          {_fmt_rate(summary['wrong_accepts'])}",
             f"wrong declines:         {_fmt_rate(summary['wrong_declines'])}",
             f"secondary (in scope):   {summary['secondary']['in_scope']}",
             f"secondary (out):        {summary['secondary']['out_of_scope']}"]
    return "\n".join(lines)


def _signed(block: dict[str, Any]) -> str:
    if block["difference"] is None:
        return "n/a"
    return (f"{100 * block['difference']:+.1f} [{100 * block['lower']:+.1f}, "
            f"{100 * block['upper']:+.1f}] pts, p={block['p_two_sided']:.3g}")


def format_markdown(report: dict[str, Any]) -> str:
    paired = report["paired"]
    arm = report.get("arm_layout", ARM_LAYOUT)
    verdict = report["verdict"]
    lines = [f"# Phase 9 step 3: {BASE_LAYOUT} vs {arm} lookup", "",
             f"Verdict: **{'PASSES' if verdict['passes'] else 'FAILS'}** "
             f"({arm} against {BASE_LAYOUT}; {report['pooled_cases']} pooled cases, "
             f"{len(report['runs'][BASE_LAYOUT])} + {len(report['runs'][arm])} trainings)", "",
             f"## Bars (paired, arm = {arm} minus base = {BASE_LAYOUT})", "",
             "| metric | difference [95% paired Wald] | bar | passes |", "|---|---|---|---|"]
    for metric, label in (("e2e", "end-to-end accuracy"), ("top1", "in-scope top-1"),
                          ("wrong_accepts", "wrong accepts")):
        bar = verdict["bars"][metric]
        lines.append(f"| {label} | {_signed(paired[metric])} | {bar['rule']} | "
                     f"{'yes' if bar['passes'] else 'no'} |")
    lines += ["", f"Wrong declines (no bar): {_signed(paired['wrong_declines'])}", "",
              "## Pooled rates", "", "| layout | end-to-end | in-scope top-1 | wrong accepts | wrong declines |",
              "|---|---|---|---|---|"]
    for layout in (BASE_LAYOUT, arm):
        s = report["summary"][layout]
        lines.append(f"| {layout} | {_fmt_rate(s['e2e'])} | {_fmt_rate(s['in_scope_top1'])} | "
                     f"{_fmt_rate(s['wrong_accepts'])} | {_fmt_rate(s['wrong_declines'])} |")
    for axis, title in (("by_depth", "chain depth"), ("by_owner_kind", "owner-scope kind")):
        lines += ["", f"## Breakdown by {title}", "",
                  "| group | cases | e2e flattened | e2e scoped | e2e diff | wrong accepts diff |",
                  "|---|---|---|---|---|---|"]
        for group, block in report["breakdowns"][axis].items():
            base_e2e = _fmt_rate(report["summary"][BASE_LAYOUT][axis][group]["e2e"])
            arm_e2e = _fmt_rate(report["summary"][arm][axis][group]["e2e"])
            lines.append(f"| {group} | {block['cases']} | {base_e2e} | {arm_e2e} | "
                         f"{_signed(block['e2e'])} | {_signed(block['wrong_accepts'])} |")
    lines += ["", "## Secondary (confidence-aware) walk", "",
              f"- flattened: {report['summary'][BASE_LAYOUT]['secondary']}",
              f"- scoped: {report['summary'][arm]['secondary']}", "",
              "## Trainings: rows and time per head (no bar)", ""]
    for layout, runs in report["runs"].items():
        for run in runs:
            rows = run.get("training_rows_per_head") or {}
            seconds = run.get("training_seconds_per_head") or {}
            lines.append(f"- {layout} seed {run['seed']}: rows " +
                         ", ".join(f"{h}={rows[h]}" for h in sorted(rows)) +
                         ("; seconds " + ", ".join(f"{h}={seconds[h]}" for h in sorted(seconds))
                          if seconds else "; seconds not recorded"))
    lines += ["", "## Method", "",
              "Pooled over the three seeds of each layout by concatenation in seed order; the layouts' case "
              "lists are asserted identical and paired index by index. The seeds repeat the same cases, so "
              "the pooled intervals are narrower than a fresh draw of cases would give.", ""]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- workflow models (read only)

def _load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_models(workflow_folderpath: str) -> dict[str, Any]:
    """Hierarchy, inheritance, owners and the holdout's model inputs, read without writing anything."""
    # EXPERIMENT INPUT, read directly. The runtime and the trainer no longer read this file
    # (Phase 9 step 5: parents come from the live chain). It is read here only to measure the
    # recorded positions of a workflow's artifacts, which is a measurement of recorded inputs.
    hierarchy_path = os.path.join(workflow_folderpath, "context_hierarchy_model.json")
    inheritance_path = os.path.join(workflow_folderpath, "_commands", "context_inheritance_model.json")
    snapshot_path = os.path.join(workflow_folderpath, "___command_info", SNAPSHOT_FILENAME)
    hierarchy = _load_json(hierarchy_path) if os.path.isfile(hierarchy_path) else {}
    raw_inheritance = _load_json(inheritance_path) if os.path.isfile(inheritance_path) else {}
    inheritance = {name: list((definition or {}).get("base") or []) for name, definition in raw_inheritance.items()}
    if not os.path.isfile(snapshot_path):
        raise SystemExit(f"{snapshot_path} does not exist; the trainer writes it, train the copy first")
    snapshot = _load_json(snapshot_path)
    metadata = snapshot["map_command_2_metadata"]
    execution = [name for name, meta in metadata.items() if not meta.get("is_planning")]
    owners = command_owners(execution, snapshot.get("core_command_names") or [])
    return {"hierarchy": hierarchy, "inheritance": inheritance, "owners": owners}


def build_cases(holdout: dict, owners: dict[str, str]) -> tuple[list[dict], list[str]]:
    """Held-out records as case templates (utterance, label, owner), in holdout order.

    A record whose label names no command of the workflow (a planning command or the startup
    command, if one ever leaked in) is skipped and its label reported.
    """
    cases: list[dict] = []
    skipped: list[str] = []
    for record in holdout["heldout"]:
        owner = owner_of(record["label"], owners)
        if owner is None:
            skipped.append(record["label"])
            continue
        cases.append({"utterance": record["utterance"], "label": record["label"], "owner": owner})
    return cases, sorted(set(skipped))


# ---------------------------------------------------------------- heads on disk

#: What makes a version folder a trained head. ``tinymodel.pth`` is a DIRECTORY (HF save_pretrained),
#: so it is tested with ``exists``, not ``is_file``.
HEAD_MARKERS = ("tinymodel.pth", "label_encoder.pkl", "threshold.json")


class MissingHeadsError(ValueError):
    """The probe lacks a head that the walks will ask."""


def discover_heads(version_folder: str) -> dict[str, str]:
    """Head name -> folder for every trained head in one version folder. ``global`` is ``*``."""
    from fastworkflow.model_pipeline_training import GLOBAL_CONTEXT_FOLDER

    heads: dict[str, str] = {}
    for folder in sorted(os.scandir(version_folder), key=lambda entry: entry.name):
        if not folder.is_dir():
            continue
        if not all(os.path.exists(os.path.join(folder.path, marker)) for marker in HEAD_MARKERS):
            continue
        head = GLOBAL if folder.name == GLOBAL_CONTEXT_FOLDER else folder.name
        heads[head] = folder.path
    return heads


def required_heads(models: dict[str, Any], layout: str, root_context: Optional[str] = None) -> list[str]:
    """Every head the layout's walks can ask, over all positions (secondary walks ask the same heads)."""
    if layout not in SCORABLE_LAYOUTS:
        raise ValueError(f"unknown layout {layout!r}")
    needed: set[str] = set()
    for position in positions(models["hierarchy"]):
        if layout == "scoped":
            needed.update(head for head, _ in scoped_path(position["chain"], models["inheritance"]))
        else:
            needed.update(head for head, _ in climb_path(position["chain"], root_context))
            if layout == GLOBAL_FIRST_LAYOUT:
                needed.add(GLOBAL)  # asked first at every position
    return sorted(needed)


def check_heads_present(needed: Sequence[str], present: Iterable[str], where: str) -> None:
    """Raise MissingHeadsError naming every needed head that is absent."""
    missing = sorted(set(needed) - set(present))
    if missing:
        raise MissingHeadsError(
            f"{where} lacks {len(missing)} head(s) the walks will ask: {', '.join(missing)}. "
            f"Train the copy, or check that each head folder has {', '.join(HEAD_MARKERS)}.")


# ---------------------------------------------------------------- probe (loads heads)

def confident_flag(confidence: float, used_distil: bool, tiny_threshold: float,
                   large_threshold: float) -> tuple[bool, float]:
    """``CommandRouter.predict_with_details``'s rule: the large tier's threshold when DistilBERT answered."""
    threshold = large_threshold if used_distil else tiny_threshold
    return confidence > threshold, threshold


def probe_rows(router: Any, utterances: Sequence[str]) -> list[dict]:
    """Top label, top-3, confident flag for every utterance, from one batched pass of one head.

    Uses the same batch call and the same tier threshold as ``predict_with_details``. The labels
    are the top-3 (or top-2 with two classes), as ``predict_single_sentence`` ranks them.
    """
    from fastworkflow.model_pipeline_training import get_label_encoder

    encoder = get_label_encoder(router.label_encoder_path)
    n_classes = len(encoder.classes_)
    k_val = 3 if n_classes > 2 else 2
    batch = router.modelpipeline.predict_batch(list(utterances), k_val=k_val)
    rows = []
    for row in range(len(utterances)):
        top = str(encoder.inverse_transform([batch["predictions"][row]])[0])
        topk = [str(label) for label in encoder.inverse_transform(batch["top_k_predictions"][row])][:3]
        confidence = float(batch["confidences"][row])
        used_distil = bool(batch["used_distil"][row])
        confident, threshold = confident_flag(confidence, used_distil,
                                              router.tiny_ambiguous_confidence_threshold,
                                              router.large_ambiguous_confidence_threshold)
        rows.append({"top_label": top, "topk": topk, "confident": bool(confident),
                     "confidence": confidence, "threshold": float(threshold),
                     "model_tier": "large" if used_distil else "tiny"})
    return rows


def probe_workflow(workflow_folderpath: str, utterances: Sequence[str], verify_sample: int = 10,
                   train_timings: Optional[dict[str, float]] = None,
                   layout: Optional[str] = None) -> dict[str, Any]:
    """Every trained head on every utterance, once. Returns the probe document (no file I/O).

    Fails loudly (SystemExit) when no head is found, or when a head the layout's walks need is
    missing. With *layout* None the needed heads are the union of both layouts.
    """
    from fastworkflow.model_pipeline_training import CommandRouter
    from fastworkflow.train import artifact_versioning
    from fastworkflow.train.score_not_here import training_rows_per_head

    version = artifact_versioning.resolve_current_version(workflow_folderpath)
    if version is None:
        raise SystemExit(f"{workflow_folderpath} has no trained artifact version; train it first")
    base = artifact_versioning.version_dir(workflow_folderpath, version)
    found = discover_heads(str(base))
    if not found:
        raise SystemExit(f"found 0 trained heads under {base} (a head needs {', '.join(HEAD_MARKERS)})")
    models = load_models(workflow_folderpath)
    layouts = [layout] if layout else list(LAYOUTS)
    needed = sorted({head for name in layouts for head in required_heads(models, name)})
    try:
        check_heads_present(needed, found, str(base))
    except MissingHeadsError as error:
        raise SystemExit(str(error)) from None
    utterances = list(dict.fromkeys(utterances))
    heads: dict[str, dict[str, dict]] = {}
    verify = {"sampled": 0, "mismatches": 0}
    for head in sorted(found):
        router = CommandRouter(found[head])
        rows = probe_rows(router, utterances)
        heads[head] = dict(zip(utterances, rows))
        for utterance in utterances[:verify_sample]:
            labels, details = router.predict_with_details(utterance)
            verify["sampled"] += 1
            mine = heads[head][utterance]
            if details["top_label"] != mine["top_label"] or details["confident"] != mine["confident"]:
                verify["mismatches"] += 1
    return {
        "schema": SCHEMA_VERSION,
        "workflow_version": version,
        "utterances": len(utterances),
        "heads": heads,
        "training_rows_per_head": training_rows_per_head(workflow_folderpath),
        "training_seconds_per_head": dict(train_timings or {}) or None,
        "verify_predict_with_details": verify,
    }


# ---------------------------------------------------------------- score and compare

def score_layout(workflow_folderpath: str, probe: dict, layout: str,
                 root_context: Optional[str] = None) -> dict[str, Any]:
    """Walk every held-out case at every position for one layout. Reads JSON only."""
    holdout_path = os.path.join(workflow_folderpath, "___command_info", HOLDOUT_FILENAME)
    if not os.path.isfile(holdout_path):
        raise SystemExit(f"{holdout_path} does not exist; train the copy under the step 3 flag first")
    holdout = _load_json(holdout_path)
    head_scope = holdout.get("head_scope")
    if layout not in SCORABLE_LAYOUTS:
        raise ValueError(f"unknown layout {layout!r}")
    if head_scope is not None and head_scope != TRAINED_AS[layout]:
        raise ValueError(f"{workflow_folderpath} was trained as layout {head_scope!r}; "
                         f"score it with a layout trained as {head_scope!r}")
    models = load_models(workflow_folderpath)
    check_heads_present(required_heads(models, layout, root_context), probe["heads"],
                        f"the probe of {workflow_folderpath}")
    all_positions = positions(models["hierarchy"])
    cases, skipped = build_cases(holdout, models["owners"])
    cache = probe["heads"]
    results: list[dict] = []
    for template in cases:
        for position in all_positions:
            results.append(evaluate_case(template, position, layout, cache, models["inheritance"], root_context))
    for index, case in enumerate(results):
        case["index"] = index
    return {
        "schema": SCHEMA_VERSION,
        "layout": layout,
        "root_context": root_context,
        "workflow_version": probe.get("workflow_version"),
        "head_scope": head_scope,
        "arm": holdout.get("arm"),
        "seed": holdout.get("seed"),
        "heldout_records": len(holdout["heldout"]),
        "skipped_labels": skipped,
        "positions": len(all_positions),
        "training_rows_per_head": probe.get("training_rows_per_head") or {},
        "training_seconds_per_head": probe.get("training_seconds_per_head"),
        "summary": summarize_cases(results),
        "cases": results,
    }


def _reconstruct_templates(prior_cases: Sequence[dict]) -> tuple[list[dict], list[dict]]:
    """Case templates and positions, from a score file's case list (written in template-major order).

    Returns ``(templates, positions)``: each template is ``{"utterance", "label", "owner", "exp"}``
    where ``exp`` lists the stored expectation (in_scope, binding, kind) of each position, in the
    position order; each position is ``{"id", "context", "chain", "depth"}``. Refuses a list that is
    not whole blocks of one position order, because the pairing depends on that order.
    """
    if not prior_cases:
        raise ValueError("the score file has no cases")
    ids: list[str] = []
    for case in prior_cases:
        if case["position"] in ids:
            break
        ids.append(case["position"])
    width = len(ids)
    if len(prior_cases) % width:
        raise ValueError(f"{len(prior_cases)} cases do not divide into blocks of {width} positions")
    positions_out: list[dict] = []
    for case in prior_cases[:width]:
        chain = case["position"].split(">")
        if len(chain) != case["depth"]:
            raise ValueError(f"position {case['position']!r} does not match its depth {case['depth']}")
        positions_out.append({"id": case["position"], "context": chain[0], "chain": chain, "depth": len(chain)})
    templates: list[dict] = []
    for start in range(0, len(prior_cases), width):
        block = prior_cases[start:start + width]
        head = block[0]
        for offset, case in enumerate(block):
            if case["position"] != ids[offset] or (case["utterance"], case["label"], case["owner"]) != (
                    head["utterance"], head["label"], head["owner"]):
                raise ValueError(f"case {start + offset} breaks the block order of the score file")
        templates.append({"utterance": head["utterance"], "label": head["label"], "owner": head["owner"],
                          "exp": [{"in_scope": case["in_scope"], "binding": case["expected_binding"],
                                   "kind": case["owner_kind"]} for case in block]})
    return templates, positions_out


def rescore_from_results(prior: dict, probe: dict, layout: str, root_context: Optional[str] = None) -> dict[str, Any]:
    """Walk a layout over the cases of an earlier score file, with the same case order.

    For when the holdout and the models json are gone: the earlier file carries each case's
    utterance, label, owner, position and stored expectation (in scope, binding, owner kind), which
    is all the walks need. Reads no workflow folder. Output has the shape of ``score_layout``.
    """
    if layout not in SCORABLE_LAYOUTS:
        raise ValueError(f"unknown layout {layout!r}")
    if prior.get("root_context") != root_context:
        raise ValueError(f"the earlier file has root_context {prior.get('root_context')!r}, not {root_context!r}")
    if layout == "scoped":
        # The scoped walk reads the scoped path, which needs the inheritance model this path does not load.
        raise ValueError("rescore_from_results walks the flattened heads only; score the scoped layout from its workflow")
    templates, positions_out = _reconstruct_templates(prior["cases"])
    check_heads_present(_needed_for_positions(positions_out, layout, root_context), probe["heads"], "the probe")
    cache = probe["heads"]
    inheritance: dict[str, Sequence[str]] = {}  # unused by the flattened walks
    results: list[dict] = []
    for template in templates:
        case = {"utterance": template["utterance"], "label": template["label"], "owner": template["owner"]}
        for position, exp in zip(positions_out, template["exp"]):
            results.append(score_expected(exp, case, position, layout, cache, inheritance, root_context))
    for index, case in enumerate(results):
        case["index"] = index
    return {
        "schema": SCHEMA_VERSION,
        "layout": layout,
        "root_context": root_context,
        "workflow_version": probe.get("workflow_version"),
        "head_scope": TRAINED_AS[layout],
        "arm": prior.get("arm"),
        "seed": prior.get("seed"),
        "heldout_records": len(templates),
        "skipped_labels": prior.get("skipped_labels") or [],
        "positions": len(positions_out),
        "training_rows_per_head": prior.get("training_rows_per_head") or {},
        "training_seconds_per_head": prior.get("training_seconds_per_head"),
        "rescored_from": prior.get("layout"),
        "summary": summarize_cases(results),
        "cases": results,
    }


def _needed_for_positions(positions_out: Sequence[dict], layout: str, root_context: Optional[str]) -> list[str]:
    needed: set[str] = set()
    for position in positions_out:
        needed.update(head for head, _ in climb_path(position["chain"], root_context))
    if layout == GLOBAL_FIRST_LAYOUT:
        needed.add(GLOBAL)
    return sorted(needed)


def _layout_of(runs: Sequence[dict], expected_layout: str) -> None:
    for run in runs:
        if run.get("layout") != expected_layout:
            raise ValueError(f"expected layout {expected_layout!r}, found {run.get('layout')!r}")


def compare_runs(base_runs: Sequence[dict], arm_runs: Sequence[dict], arm_layout: str = ARM_LAYOUT) -> dict[str, Any]:
    """Pool each layout's seeds, pair them, and apply the bars and breakdowns."""
    if arm_layout not in SCORABLE_LAYOUTS:
        raise ValueError(f"unknown arm layout {arm_layout!r}")
    _layout_of(base_runs, BASE_LAYOUT)
    _layout_of(arm_runs, arm_layout)
    base_cases = pool_seeds(base_runs)
    arm_cases = pool_seeds(arm_runs)
    paired = paired_metrics(base_cases, arm_cases)
    return {
        "schema": SCHEMA_VERSION,
        "arm_layout": arm_layout,
        "pooled_cases": len(base_cases),
        "runs": {
            BASE_LAYOUT: [_run_brief(run) for run in base_runs],
            arm_layout: [_run_brief(run) for run in arm_runs],
        },
        "summary": {BASE_LAYOUT: summarize_cases(base_cases), arm_layout: summarize_cases(arm_cases)},
        "paired": paired,
        "verdict": evaluate_layout(paired),
        "breakdowns": breakdowns(base_cases, arm_cases),
    }


def _run_brief(run: dict) -> dict[str, Any]:
    return {"seed": run.get("seed"), "arm": run.get("arm"), "workflow_version": run.get("workflow_version"),
            "training_rows_per_head": run.get("training_rows_per_head"),
            "training_seconds_per_head": run.get("training_seconds_per_head"),
            "summary": {"e2e": run["summary"]["e2e"]["rate"], "wrong_accepts": run["summary"]["wrong_accepts"]["rate"]}}


# ---------------------------------------------------------------- CLI

def _write_json(path: Optional[str], payload: Any) -> None:
    if not path:
        return
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _split(value: str) -> list[str]:
    return [item for item in value.split(",") if item]


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="score_lookup", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p_probe = sub.add_parser("probe", help="run every head on every held-out utterance once")
    p_probe.add_argument("--workflow", required=True)
    p_probe.add_argument("--out", required=True)
    p_probe.add_argument("--train-timings", default=None, help="JSON {head: seconds} from the training runner")

    p_score = sub.add_parser("score", help="walk every case at every position for one layout")
    p_score.add_argument("--workflow", required=True)
    p_score.add_argument("--probe", required=True)
    p_score.add_argument("--layout", required=True, choices=SCORABLE_LAYOUTS)
    p_score.add_argument("--root-context", default=None,
                         help="context name of the root record; default: none, so the climb asks '*' after a top")
    p_score.add_argument("--out", required=True)

    p_compare = sub.add_parser("compare", help="pooled, paired comparison of two layouts")
    p_compare.add_argument("--base", required=True, help="flattened score files, comma separated, in seed order")
    p_compare.add_argument("--arm", required=True, help="scoped score files, comma separated, in seed order")
    p_compare.add_argument("--arm-layout", default=ARM_LAYOUT, choices=SCORABLE_LAYOUTS,
                           help="the layout of the --arm files (default scoped)")
    p_compare.add_argument("--out", required=True)
    p_compare.add_argument("--md", required=True)

    p_rescore = sub.add_parser("rescore", help="walk one layout over an earlier score file's cases (no workflow read)")
    p_rescore.add_argument("--results", required=True, help="an earlier score file (its cases are reused)")
    p_rescore.add_argument("--probe", required=True)
    p_rescore.add_argument("--layout", required=True, choices=SCORABLE_LAYOUTS)
    p_rescore.add_argument("--root-context", default=None)
    p_rescore.add_argument("--out", required=True)

    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "probe":
        holdout = _load_json(os.path.join(args.workflow, "___command_info", HOLDOUT_FILENAME))
        utterances = [record["utterance"] for record in holdout["heldout"]]
        timings = _load_json(args.train_timings) if args.train_timings else None
        probe = probe_workflow(args.workflow, utterances, train_timings=timings,
                               layout=holdout.get("head_scope"))
        _write_json(args.out, probe)
        print(f"probed {len(probe['heads'])} heads on {probe['utterances']} utterances -> {args.out}")
        return 0
    if args.command == "score":
        probe = _load_json(args.probe)
        try:
            report = score_layout(args.workflow, probe, args.layout, args.root_context)
        except MissingHeadsError as error:
            raise SystemExit(str(error)) from None
        _write_json(args.out, report)
        print(format_score_summary(report["summary"], args.layout))
        return 0
    if args.command == "rescore":
        prior = _load_json(args.results)
        probe = _load_json(args.probe)
        try:
            report = rescore_from_results(prior, probe, args.layout, args.root_context)
        except MissingHeadsError as error:
            raise SystemExit(str(error)) from None
        _write_json(args.out, report)
        print(format_score_summary(report["summary"], args.layout))
        return 0
    base_runs = [_load_json(path) for path in _split(args.base)]
    arm_runs = [_load_json(path) for path in _split(args.arm)]
    report = compare_runs(base_runs, arm_runs, args.arm_layout)
    _write_json(args.out, report)
    with open(args.md, "w", encoding="utf-8") as handle:
        handle.write(format_markdown(report))
    print(format_markdown(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())

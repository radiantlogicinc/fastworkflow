"""Threshold the "not here" probability and score the thresholds against a control arm.

EXPERIMENT TOOL (Phase 9, follow-up to ``score_not_here``). It loads trained heads only to
``probe``; ``sweep`` reads JSON and loads no model. Nothing here changes a production path.

Decision rule for a threshold ``t``, per case::

    declined = is_non_routable(top_label) or p_not_here >= t

``p_not_here`` is the summed softmax probability of the non-routable labels (``wildcard``,
``parameter_value``) in the tier that actually produced the case's prediction: TinyBERT, or
DistilBERT when TinyBERT's max probability was under its confidence threshold. ``t = inf`` is
the current behaviour (no threshold).

Under a threshold, a declined case is scored as a decline (top label replaced by ``wildcard``,
top-k emptied) and an accepted case keeps its prediction. Metrics and paired intervals are the
ones ``score_not_here`` and ``not_here_step1`` already define, so they are not re-derived here.

Subcommands::

    python -m fastworkflow.train.not_here_threshold probe --workflow W --cases C --out P
    python -m fastworkflow.train.not_here_threshold sweep --probe P --control R [--out J] [--md M]

``probe`` runs each head's cases once through ``ModelPipeline.predict_batch`` with k equal to
the number of classes, so the full distribution of the tier used is kept. ``sweep`` pairs the
probe output index by index with an existing ``score_not_here predict`` results file for the
control arm (same cases, same order) and fails loudly when they disagree.

Honest selection is cross-fit: cases are split into two folds by a sha256 hash of either the
``expected`` label or the ``utterance``. On each fold the threshold that minimises wrong accepts
among thresholds passing the bar is chosen (ties: higher t; none passing: ``inf``), and it is
then scored on the other fold. The two out-of-fold halves are pooled and judged again.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from typing import Any, Iterable, Optional, Sequence

from fastworkflow.nlu_labels import WILDCARD_LABEL, is_non_routable, label_of
from fastworkflow.train import not_here_step1 as step1
from fastworkflow.train import score_not_here as scorer

SCHEMA_VERSION = 1
INF = math.inf
GRID: tuple[float, ...] = tuple(round(i / 100, 2) for i in range(101)) + (INF,)
REPORTED: tuple[float, ...] = tuple(round(i * 0.05, 2) for i in range(21)) + (INF,)
BARS: tuple[str, ...] = ("full", "without_decline")
FOLD_KEYS: tuple[str, ...] = ("expected", "utterance")
QUANTILES: tuple[float, ...] = (0.50, 0.75, 0.90, 0.95, 0.99)
_ALIGNED_FIELDS = ("head", "utterance", "in_scope", "expected")


class CaseAlignmentError(ValueError):
    """The probe and the control results do not score the same cases in the same order."""


# ---------------------------------------------------------------- labels and the decision rule

def threshold_label(threshold: float) -> str:
    return "inf" if math.isinf(threshold) else f"{threshold:.2f}"


def not_here_mass(labels: Sequence[str], scores: Sequence[float]) -> float:
    """Summed probability of the non-routable labels in one full distribution."""
    return float(sum(score for label, score in zip(labels, scores) if is_non_routable(str(label))))


def declines_at(case: dict, threshold: float) -> bool:
    """The decline rule: a non-routable top label, or P(not here) at or above the threshold."""
    top = str(case.get("top_label") or "")
    if (is_non_routable(top) if top else True):
        return True
    return float(case["p_not_here"]) >= threshold


def scored_case(probe_case: dict, threshold: float) -> dict:
    """The case as ``score_not_here`` scores it under *threshold* (control-results shape)."""
    if declines_at(probe_case, threshold):
        top_label, topk = WILDCARD_LABEL, []
    else:
        top_label, topk = str(probe_case["top_label"]), [str(label) for label in probe_case["topk"]]
    return {"head": probe_case["head"], "utterance": probe_case["utterance"],
            "expected": probe_case["expected"], "in_scope": bool(probe_case["in_scope"]),
            "top_label": top_label, "topk": topk}


def arm_cases(probe_cases: Sequence[dict], threshold: float) -> list[dict]:
    return [scored_case(case, threshold) for case in probe_cases]


# ---------------------------------------------------------------- alignment

def align(probe_cases: Sequence[dict], control_cases: Sequence[dict]) -> None:
    """Raise CaseAlignmentError unless the two lists hold the same cases in the same order."""
    if len(probe_cases) != len(control_cases):
        raise CaseAlignmentError(
            f"probe has {len(probe_cases)} cases but the control results have {len(control_cases)}")
    for index, (probe, control) in enumerate(zip(probe_cases, control_cases)):
        for field in _ALIGNED_FIELDS:
            if probe.get(field) != control.get(field):
                raise CaseAlignmentError(
                    f"case {index}: {field} differs (probe {probe.get(field)!r}, "
                    f"control {control.get(field)!r}); the files are not from the same cases")


# ---------------------------------------------------------------- metrics and the bar

def arm_metrics(arm: list[dict], control: list[dict]) -> dict[str, Any]:
    """Rates for one arm, its paired differences against control, and the owner's bar.

    The paired intervals and the bar come from ``not_here_step1.evaluate_arm``.
    ``passes_bar_without_decline`` is the same bar minus the wrong-declines criterion.
    """
    verdict = step1.evaluate_arm(arm, list(control))
    return {
        "overall": scorer.summarize({"cases": arm})["overall"],
        "vs_control": verdict,
        "passes_bar": bool(verdict["passes"]),
        "passes_bar_without_decline": bool(verdict["wrong_accepts"]["passes"] and verdict["top1"]["passes"]),
    }


def metrics_at(probe_cases: Sequence[dict], control_cases: Sequence[dict], threshold: float) -> dict[str, Any]:
    return {"threshold": threshold_label(threshold),
            **arm_metrics(arm_cases(probe_cases, threshold), control_cases)}


def evaluate_grid(probe_cases: Sequence[dict], control_cases: Sequence[dict],
                  thresholds: Sequence[float] = GRID) -> dict[float, dict[str, Any]]:
    return {t: metrics_at(probe_cases, control_cases, t) for t in thresholds}


def choose_threshold(grid: dict[float, dict[str, Any]], bar: str = "full") -> float:
    """Among thresholds passing *bar*, the one with the fewest wrong accepts; ties go to the
    higher threshold. When none passes, ``inf`` (no threshold)."""
    key = "passes_bar" if bar == "full" else "passes_bar_without_decline"
    passing = [t for t, metrics in grid.items() if metrics[key]]
    if not passing:
        return INF
    return min(passing, key=lambda t: (grid[t]["overall"]["wrong_accepts"]["count"], -t))


# ---------------------------------------------------------------- folds and cross-fit

def fold_of(key: str, n_folds: int = 2) -> int:
    """Stable fold id: sha256 of the key, never Python's salted ``hash``."""
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % n_folds


def assign_folds(cases: Sequence[dict], by: str) -> list[int]:
    if by not in FOLD_KEYS:
        raise ValueError(f"fold key must be one of {FOLD_KEYS}, not {by!r}")
    return [fold_of(str(case[by])) for case in cases]


def _take(items: Sequence[Any], indices: Sequence[int]) -> list[Any]:
    return [items[i] for i in indices]


def crossfit(probe_cases: Sequence[dict], control_cases: Sequence[dict], folds: Sequence[int],
             bar: str = "full") -> dict[str, Any]:
    """Pick a threshold on each fold, score it on the other fold, and pool the out-of-fold halves."""
    n = len(probe_cases)
    picks: dict[int, float] = {}
    per_fold: dict[str, Any] = {}
    for fold in (0, 1):
        selection = [i for i in range(n) if folds[i] == fold]
        other = [i for i in range(n) if folds[i] != fold]
        grid = evaluate_grid(_take(probe_cases, selection), _take(control_cases, selection))
        picks[fold] = choose_threshold(grid, bar)
        per_fold[f"fold_{fold}"] = {
            "pick": threshold_label(picks[fold]),
            "selection_n": len(selection),
            "selection_pick_passes_bar": bool(grid[picks[fold]][
                "passes_bar" if bar == "full" else "passes_bar_without_decline"]),
            "scored_on_other_fold": {
                "n": len(other),
                **metrics_at(_take(probe_cases, other), _take(control_cases, other), picks[fold]),
            },
        }

    pooled_arm = [scored_case(probe_cases[i], picks[1 - folds[i]]) for i in range(n)]
    pooled = arm_metrics(pooled_arm, list(control_cases))
    return {"bar": bar, "picks": {f"fold_{f}": threshold_label(picks[f]) for f in (0, 1)},
            "folds": per_fold, "pooled": {"threshold": "cross-fit", **pooled}}


# ---------------------------------------------------------------- separation

def quantile(values: Sequence[float], q: float) -> Optional[float]:
    """Linear-interpolated quantile; None for no values."""
    if not values:
        return None
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def auc_by_rank(positives: Sequence[float], negatives: Sequence[float]) -> Optional[float]:
    """ROC AUC as the Mann-Whitney statistic: P(positive > negative) + half the ties."""
    if not positives or not negatives:
        return None
    pairs = sorted([(value, 1) for value in positives] + [(value, 0) for value in negatives])
    rank_sum_positive = 0.0
    index = 0
    while index < len(pairs):
        end = index
        while end + 1 < len(pairs) and pairs[end + 1][0] == pairs[index][0]:
            end += 1
        average_rank = (index + end) / 2 + 1  # 1-based ranks index+1..end+1
        rank_sum_positive += average_rank * sum(1 for k in range(index, end + 1) if pairs[k][1] == 1)
        index = end + 1
    n_pos, n_neg = len(positives), len(negatives)
    return (rank_sum_positive - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def separation(probe_cases: Sequence[dict]) -> dict[str, Any]:
    """p_not_here among accepted cases (top label not non-routable): quantiles and AUC."""
    accepted = [case for case in probe_cases if not declines_at(case, INF)]

    def correct(case: dict) -> bool:
        return bool(case["in_scope"]) and label_of(str(case["top_label"])) == label_of(str(case["expected"]))

    groups = {
        "in-scope, correct top-1": [c["p_not_here"] for c in accepted if correct(c)],
        "in-scope, wrong command": [c["p_not_here"] for c in accepted if c["in_scope"] and not correct(c)],
        "out-of-scope": [c["p_not_here"] for c in accepted if not c["in_scope"]],
    }
    in_scope_accepted = [c["p_not_here"] for c in accepted if c["in_scope"]]
    out_accepted = groups["out-of-scope"]
    return {
        "groups": {name: {"n": len(values),
                          "quantiles": {f"{int(q * 100)}": quantile(values, q) for q in QUANTILES}}
                   for name, values in groups.items()},
        "auc_out_of_scope_vs_in_scope": auc_by_rank(out_accepted, in_scope_accepted),
        "n_out_of_scope_accepted": len(out_accepted),
        "n_in_scope_accepted": len(in_scope_accepted),
    }


def agreement(probe_cases: Sequence[dict], control_cases: Sequence[dict]) -> dict[str, int]:
    """How often the probe reproduces the control's top-1 and top-3 (sanity check on the probe)."""
    return {
        "n": len(probe_cases),
        "top1_same": sum(1 for p, c in zip(probe_cases, control_cases)
                         if str(p["top_label"]) == str(c["top_label"])),
        "topk_same": sum(1 for p, c in zip(probe_cases, control_cases)
                         if [str(x) for x in p["topk"]] == [str(x) for x in c["topk"]]),
    }


# ---------------------------------------------------------------- sweep

def sweep(probe_cases: Sequence[dict], control_cases: Sequence[dict]) -> dict[str, Any]:
    align(probe_cases, control_cases)
    grid = evaluate_grid(probe_cases, control_cases)
    decline_diffs = [m["vs_control"]["wrong_declines"]["difference"] for m in grid.values()
                     if m["vs_control"]["wrong_declines"]["difference"] is not None]
    crossfits: dict[str, Any] = {}
    for key in FOLD_KEYS:
        folds = assign_folds(probe_cases, key)
        for bar in BARS:
            crossfits[f"folds_by_{key}/{bar}"] = {"fold_key": key, **crossfit(probe_cases, control_cases, folds, bar)}
    n_in = sum(1 for case in probe_cases if case["in_scope"])
    return {
        "schema": SCHEMA_VERSION,
        "n_cases": len(probe_cases), "n_in_scope": n_in, "n_out_of_scope": len(probe_cases) - n_in,
        "wrong_decline_diff_min": min(decline_diffs) if decline_diffs else None,
        "agreement": agreement(probe_cases, control_cases),
        "grid": [grid[t] for t in GRID],
        "crossfit": crossfits,
        "separation": separation(probe_cases),
    }


# ---------------------------------------------------------------- markdown

def _pct(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def _count(block: dict[str, Any]) -> str:
    if not block["trials"]:
        return "n/a"
    return f"{block['count']}/{block['trials']} ({_pct(block['rate'])})"


def _diff(block: dict[str, Any]) -> str:
    if block["difference"] is None:
        return "n/a"
    return (f"{100 * block['difference']:+.1f} [{100 * block['lower']:+.1f}, "
            f"{100 * block['upper']:+.1f}] pts")


def _row(label: str, metrics: dict[str, Any]) -> str:
    overall, vs = metrics["overall"], metrics["vs_control"]
    bars = "/".join("ok" if vs[m]["passes"] else "no" for m in ("wrong_declines", "wrong_accepts", "top1"))
    return (f"| {label} | {_count(overall['wrong_declines'])} | {_count(overall['wrong_accepts'])} | "
            f"{_count(overall['top1'])} | {_count(overall['in_list'])} | {_diff(vs['wrong_declines'])} | "
            f"{_diff(vs['wrong_accepts'])} | {_diff(vs['top1'])} | {bars} | "
            f"{'PASS' if metrics['passes_bar'] else 'fail'} |")


_ROW_HEADER = ("| threshold | wrong declines | wrong accepts | top-1 | in-list | "
               "d wrong declines | d wrong accepts | d top-1 | bars wd/wa/t1 | bar |\n"
               "|---|---|---|---|---|---|---|---|---|---|")


def format_markdown(result: dict[str, Any], control_arm: str, workflow_version: Optional[str]) -> str:
    by_label = {m["threshold"]: m for m in result["grid"]}
    lines = [
        "# Not-here probability threshold, scored against control", "",
        f"Control arm: `{control_arm}`. Workflow version: `{workflow_version}`. Cases: {result['n_cases']} "
        f"({result['n_in_scope']} in scope, {result['n_out_of_scope']} out of scope).",
        "",
        "Decision rule: declined = top label non-routable OR P(not here) >= t, using the tier that "
        "produced the case. `inf` is the current behaviour. Differences are threshold minus control, "
        "paired 95% Wald intervals (`not_here_step1.paired_difference`); McNemar p-values are in the JSON.",
        "",
    ]
    diff_min = result["wrong_decline_diff_min"]
    if diff_min is not None and diff_min >= 0:
        lines += [
            f"**The wrong-declines criterion cannot pass under this rule.** A threshold only adds declines, "
            f"so the wrong-declines difference is never negative (smallest over the grid: {100 * diff_min:+.1f} pts). "
            "The bar needs its upper bound below 0, and the control itself gives exactly 0. "
            "Every threshold therefore fails the full bar, and every full-bar cross-fit pick is `inf`. "
            "The bar without that criterion is shown alongside.",
            "",
        ]
    agree = result["agreement"]
    lines += [f"Probe vs control: top-1 identical on {agree['top1_same']}/{agree['n']} cases, top-3 identical on "
              f"{agree['topk_same']}/{agree['n']}.", ""]

    crossfit_picks: list[str] = []
    for name, block in result["crossfit"].items():
        for label in block["picks"].values():
            if label not in crossfit_picks:
                crossfit_picks.append(label)
    rows = [threshold_label(t) for t in REPORTED]
    rows += [label for label in crossfit_picks if label not in rows]

    lines += ["## Selected thresholds (every 0.05, inf, and cross-fit picks)", "", _ROW_HEADER]
    for label in rows:
        lines.append(_row(label, by_label[label]))
    lines += ["", "Cross-fit picks appearing only in the table above come from the cross-fit section.", ""]

    lines += ["## Cross-fit", ""]
    for name, block in result["crossfit"].items():
        lines += [f"### {name}", "",
                  f"Picks: fold 0 -> `{block['picks']['fold_0']}`, fold 1 -> `{block['picks']['fold_1']}`. "
                  "Each pick is scored on the fold it was not chosen on.", "",
                  "| fold | pick | scored on other fold | wrong declines | wrong accepts | top-1 | in-list | "
                  "d wrong declines | d wrong accepts | d top-1 | bars wd/wa/t1 | bar |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for fold in ("fold_0", "fold_1"):
            info = block["folds"][fold]
            other = info["scored_on_other_fold"]
            lines.append(_crossfit_row(fold, info["pick"], other))
        lines.append(_crossfit_row("pooled out-of-fold", "cross-fit", block["pooled"], pooled=True))
        lines.append("")

    sep = result["separation"]
    lines += ["## Separation among accepted cases (top label not non-routable)", "",
              "| group | n | p50 | p75 | p90 | p95 | p99 |", "|---|---|---|---|---|---|---|"]
    for name, group in sep["groups"].items():
        q = group["quantiles"]
        cells = " | ".join("n/a" if q[k] is None else f"{q[k]:.4f}" for k in ("50", "75", "90", "95", "99"))
        lines.append(f"| {name} | {group['n']} | {cells} |")
    auc = sep["auc_out_of_scope_vs_in_scope"]
    lines += ["", f"ROC AUC of p_not_here, out-of-scope vs in-scope (accepted cases): "
              f"{'n/a' if auc is None else f'{auc:.4f}'} "
              f"({sep['n_out_of_scope_accepted']} out-of-scope, {sep['n_in_scope_accepted']} in-scope).", ""]
    return "\n".join(lines) + "\n"


def _crossfit_row(label: str, pick: str, metrics: dict[str, Any], pooled: bool = False) -> str:
    overall, vs = metrics["overall"], metrics["vs_control"]
    bars = "/".join("ok" if vs[m]["passes"] else "no" for m in ("wrong_declines", "wrong_accepts", "top1"))
    scored = "all cases" if pooled else f"n={metrics.get('n', '-')}"
    return (f"| {label} | {pick} | {scored} | {_count(overall['wrong_declines'])} | "
            f"{_count(overall['wrong_accepts'])} | {_count(overall['top1'])} | {_count(overall['in_list'])} | "
            f"{_diff(vs['wrong_declines'])} | {_diff(vs['wrong_accepts'])} | {_diff(vs['top1'])} | {bars} | "
            f"{'PASS' if metrics['passes_bar'] else 'fail'} |")


# ---------------------------------------------------------------- probe (loads models)

def probe_cases(workflow_folderpath: str, cases: Sequence[dict]) -> dict[str, Any]:
    """Run every case through its head's router, keeping the full distribution of the tier used."""
    from fastworkflow.model_pipeline_training import CommandRouter, get_label_encoder
    from fastworkflow.train import artifact_versioning

    version = artifact_versioning.resolve_current_version(workflow_folderpath)
    if version is None:
        raise SystemExit(f"{workflow_folderpath} has no trained artifact version; train it first")
    base = artifact_versioning.version_dir(workflow_folderpath, version)

    by_head: dict[str, list[int]] = {}
    for index, case in enumerate(cases):
        by_head.setdefault(case["head"], []).append(index)

    out: list[Optional[dict]] = [None] * len(cases)
    for head, indices in by_head.items():
        router = CommandRouter(str(base / artifact_versioning.context_folder_name(head)))
        encoder = get_label_encoder(router.label_encoder_path)
        n_classes = len(encoder.classes_)
        batch = router.modelpipeline.predict_batch([cases[i]["utterance"] for i in indices], k_val=n_classes)
        for row, index in enumerate(indices):
            labels = [str(label) for label in encoder.inverse_transform(batch["top_k_predictions"][row])]
            scores = [float(score) for score in batch["top_k_scores"][row]]
            top_label = str(encoder.inverse_transform([batch["predictions"][row]])[0])
            out[index] = {
                **cases[index],
                "top_label": top_label,
                "top_prob": float(batch["confidences"][row]),
                "p_not_here": not_here_mass(labels, scores),
                "used_distil": bool(batch["used_distil"][row]),
                "topk": labels[:3],
            }
    return {"schema": SCHEMA_VERSION, "workflow_version": version, "cases": out}


# ---------------------------------------------------------------- CLI

def _load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Optional[str], payload: dict) -> None:
    if not path:
        return
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="not_here_threshold", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p_probe = sub.add_parser("probe", help="full softmax per case from the trained heads")
    p_probe.add_argument("--workflow", required=True)
    p_probe.add_argument("--cases", required=True)
    p_probe.add_argument("--out", required=True)

    p_sweep = sub.add_parser("sweep", help="thresholds scored against a control arm's predict results")
    p_sweep.add_argument("--probe", required=True)
    p_sweep.add_argument("--control", required=True)
    p_sweep.add_argument("--out", default=None)
    p_sweep.add_argument("--md", default=None)

    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "probe":
        cases = _load_json(args.cases)["cases"]
        result = probe_cases(args.workflow, cases)
        _write_json(args.out, result)
        print(f"wrote {len(result['cases'])} probe results to {args.out}")
        return 0

    probe = _load_json(args.probe)
    control = _load_json(args.control)
    result = sweep(probe["cases"], control["cases"])
    result["control_arm"] = control.get("arm")
    result["workflow_version"] = probe.get("workflow_version")
    _write_json(args.out, result)
    markdown = format_markdown(result, str(control.get("arm")), probe.get("workflow_version"))
    if args.md:
        with open(args.md, "w", encoding="utf-8") as handle:
            handle.write(markdown)
    print(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())

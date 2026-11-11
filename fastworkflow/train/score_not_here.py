"""Score the sparse "not here" class in both directions (Phase 9 step 1, design 2.15).

EXPERIMENT TOOL. It scores already-trained heads; it does not train them. Arms are
selected at training time by ``model_pipeline_training.experiment_reserved_class_arm``.

Each case names a head (a context), an utterance, and whether the utterance is in scope:

* in scope: the utterance is one of the head's own commands; expected = that command;
* out of scope: the utterance belongs to any other command of the workflow; the head
  should decline it ("not here").

Per case the head's top-1 label decides the verdict. A top-1 label that is non-routable
(``wildcard`` or ``parameter_value``, see ``nlu_labels``) is a decline; any other label
is an accept.

Per head and overall it reports, each with a 95% Wilson interval:

* wrong declines: in-scope cases declined (out of n in-scope);
* wrong accepts: out-of-scope cases accepted (out of n out-of-scope);
* top-1 accuracy and in-list accuracy on in-scope cases;
* training rows per head, from ``___command_info/training_provenance.json``.

``compare`` pairs two arms' per-case results (same cases, same order) and reports an
exact two-sided McNemar test per metric.

Subcommands::

    python -m fastworkflow.train.score_not_here predict --workflow W --cases C --out R [--arm NAME]
    python -m fastworkflow.train.score_not_here score   --results R [--out S]
    python -m fastworkflow.train.score_not_here compare --a RA --b RB [--out M]

``predict`` loads each head's CommandRouter from the workflow's current artifact version
(the real runtime path). ``score`` and ``compare`` read JSON only and load no model.

Cases file: ``{"cases": [{"head": "Book", "utterance": "...", "expected": "Book/loan_report",
"in_scope": true}, ...]}``. For out-of-scope cases ``expected`` names the owning command
and is for audit only; the scorer reads ``in_scope``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from typing import Any, Iterable, Optional, Sequence

from fastworkflow.nlu_labels import is_non_routable, label_of

SCHEMA_VERSION = 1
Z_95 = 1.959963984540054
_ACCEPTED_STATUSES = ("included", "included_fallback")

_METRICS = (
    "wrong_decline",
    "wrong_accept",
    "top1",
    "in_list",
    "verdict",
)


# ---------------------------------------------------------------- statistics

def wilson_interval(successes: int, trials: int, z: float = Z_95) -> Optional[tuple[float, float]]:
    """Wilson score interval for a binomial proportion; None when there are no trials."""
    if trials <= 0:
        return None
    p = successes / trials
    z2 = z * z
    denom = 1.0 + z2 / trials
    centre = (p + z2 / (2 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials + z2 / (4 * trials * trials)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(only_a_wrong: int, only_b_wrong: int) -> float:
    """Two-sided exact McNemar p-value from the two discordant counts.

    Under H0 the discordant pairs split 50/50, so p = min(1, 2 * P(X <= min(b, c)))
    with X ~ Binomial(b + c, 1/2). Returns 1.0 when there are no discordant pairs.
    """
    b, c = int(only_a_wrong), int(only_b_wrong)
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


# ---------------------------------------------------------------- per-case verdicts

def case_outcome(case: dict) -> dict[str, Optional[bool]]:
    """Error indicators for one scored case. None means the metric does not apply.

    Keys are the error indicators used by ``summarize`` and ``compare``:
    ``wrong_decline``/``top1_miss``/``in_list_miss`` apply to in-scope cases;
    ``wrong_accept`` applies to out-of-scope cases; ``verdict_error`` applies to all.
    """
    top = str(case.get("top_label") or "")
    topk = [str(label) for label in case.get("topk") or []]
    declined = is_non_routable(top) if top else True
    if case["in_scope"]:
        expected = label_of(str(case["expected"]))
        top1 = bool(top) and label_of(top) == expected
        in_list = expected in {label_of(label) for label in topk} or top1
        wrong_decline = declined
        return {
            "wrong_decline": bool(wrong_decline),
            "wrong_accept": None,
            "top1_miss": not top1,
            "in_list_miss": not in_list,
            "verdict_error": not top1,
        }
    accepted = not declined
    return {
        "wrong_decline": None,
        "wrong_accept": bool(accepted),
        "top1_miss": None,
        "in_list_miss": None,
        "verdict_error": bool(accepted),
    }


def _rate_block(successes: int, trials: int) -> dict[str, Any]:
    interval = wilson_interval(successes, trials)
    return {
        "count": successes,
        "trials": trials,
        "rate": (successes / trials) if trials else None,
        "wilson95": list(interval) if interval else None,
    }


def summarize(results: dict) -> dict[str, Any]:
    """Per-head and overall metrics from a ``predict`` results document."""
    cases = results["cases"]
    rows_per_head: dict[str, int] = dict(results.get("training_rows") or {})
    heads = sorted({case["head"] for case in cases} | set(rows_per_head))

    def empty() -> dict[str, int]:
        return Counter({"n_in": 0, "n_out": 0, "wrong_decline": 0, "wrong_accept": 0,
                        "top1_correct": 0, "in_list_correct": 0})

    totals = empty()
    per_head: dict[str, dict[str, Any]] = {}
    for head in heads:
        counts = empty()
        for case in (c for c in cases if c["head"] == head):
            outcome = case_outcome(case)
            if case["in_scope"]:
                counts["n_in"] += 1
                counts["wrong_decline"] += int(bool(outcome["wrong_decline"]))
                counts["top1_correct"] += int(not outcome["top1_miss"])
                counts["in_list_correct"] += int(not outcome["in_list_miss"])
            else:
                counts["n_out"] += 1
                counts["wrong_accept"] += int(bool(outcome["wrong_accept"]))
        totals.update(counts)
        per_head[head] = _head_block(counts, rows_per_head.get(head))

    overall = _head_block(totals, sum(rows_per_head.values()) if rows_per_head else None)
    return {"schema": SCHEMA_VERSION, "arm": results.get("arm"), "overall": overall, "per_head": per_head}


def _head_block(counts: Any, training_rows: Optional[int]) -> dict[str, Any]:
    return {
        "in_scope_n": counts["n_in"],
        "out_of_scope_n": counts["n_out"],
        "wrong_declines": _rate_block(counts["wrong_decline"], counts["n_in"]),
        "wrong_accepts": _rate_block(counts["wrong_accept"], counts["n_out"]),
        "top1": _rate_block(counts["top1_correct"], counts["n_in"]),
        "in_list": _rate_block(counts["in_list_correct"], counts["n_in"]),
        "training_rows": training_rows,
    }


def _case_key(case: dict) -> tuple[str, str, bool]:
    return (case["head"], case["utterance"], bool(case["in_scope"]))


def compare(results_a: dict, results_b: dict) -> dict[str, Any]:
    """Paired McNemar per metric between two arms scored on the same cases."""
    cases_a = results_a["cases"]
    cases_b = results_b["cases"]
    keys_a = [_case_key(c) for c in cases_a]
    keys_b = [_case_key(c) for c in cases_b]
    if keys_a != keys_b:
        raise ValueError("the two results do not score the same cases in the same order")

    paired: dict[str, dict[str, Any]] = {}
    for metric in _METRICS:
        field = {"wrong_decline": "wrong_decline", "wrong_accept": "wrong_accept",
                 "top1": "top1_miss", "in_list": "in_list_miss", "verdict": "verdict_error"}[metric]
        a_only = b_only = both = neither = 0
        for case_a, case_b in zip(cases_a, cases_b):
            err_a = case_outcome(case_a)[field]
            err_b = case_outcome(case_b)[field]
            if err_a is None or err_b is None:
                continue
            if err_a and err_b:
                both += 1
            elif err_a:
                a_only += 1
            elif err_b:
                b_only += 1
            else:
                neither += 1
        paired[metric] = {
            "pairs": a_only + b_only + both + neither,
            "errors_a": a_only + both,
            "errors_b": b_only + both,
            "only_a_wrong": a_only,
            "only_b_wrong": b_only,
            "p_two_sided": mcnemar_exact(a_only, b_only),
        }
    return {"schema": SCHEMA_VERSION, "arm_a": results_a.get("arm"), "arm_b": results_b.get("arm"),
            "paired": paired}


# ---------------------------------------------------------------- text summary

def _fmt_rate(block: dict[str, Any]) -> str:
    if block["trials"] == 0:
        return "n/a"
    lo, hi = block["wilson95"]
    return f"{block['count']}/{block['trials']} = {100 * block['rate']:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]"


def format_summary(summary: dict[str, Any]) -> str:
    lines = [f"Not-here scoring (arm: {summary.get('arm') or 'unlabelled'})", ""]
    header = f"{'head':<24}{'rows':>7}  {'wrong declines':<34}{'wrong accepts':<34}{'top-1':<34}in-list"
    lines.append(header)
    for head, block in list(summary["per_head"].items()) + [("OVERALL", summary["overall"])]:
        rows = "-" if block["training_rows"] is None else str(block["training_rows"])
        lines.append(
            f"{head:<24}{rows:>7}  {_fmt_rate(block['wrong_declines']):<34}"
            f"{_fmt_rate(block['wrong_accepts']):<34}{_fmt_rate(block['top1']):<34}"
            f"{_fmt_rate(block['in_list'])}"
        )
    lines.append("")
    lines.append("Intervals are 95% Wilson. Wrong declines are over in-scope cases; wrong accepts over out-of-scope.")
    return "\n".join(lines)


def format_comparison(comparison: dict[str, Any]) -> str:
    lines = [f"Paired comparison: {comparison.get('arm_a')} vs {comparison.get('arm_b')} (exact McNemar)", ""]
    for metric, block in comparison["paired"].items():
        lines.append(
            f"{metric:<14} pairs={block['pairs']:<5} errors {block['errors_a']} vs {block['errors_b']}  "
            f"only-a-wrong={block['only_a_wrong']} only-b-wrong={block['only_b_wrong']}  p={block['p_two_sided']:.4g}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------- arms and cases

class ArmMismatchError(ValueError):
    """``predict --arm`` names a reserved-class arm other than the one the artifact was trained with."""


def refuse_retired(arm: str) -> None:
    """Raise ValueError when *arm* is a retired arm (Phase 9 step 5): it no longer exists to score."""
    from fastworkflow.model_pipeline_training import RETIRED_ARMS

    if arm in RETIRED_ARMS:
        raise ValueError(f"arm {arm!r} is retired: {RETIRED_ARMS[arm]}")


def recorded_arm(workflow_folderpath: str) -> str:
    """The reserved-class arm the current artifacts were trained with (control when unrecorded).

    ``control`` is the label of artifacts trained before the arm's retirement. It is a history
    label only: ``check_arm`` refuses to score it.
    """
    from fastworkflow.train.determinism import ProvenanceRecorder

    return ProvenanceRecorder.load_run_metadata(workflow_folderpath).get("reserved_class_arm") or "control"


def check_arm(workflow_folderpath: str, requested: Optional[str]) -> str:
    """The arm to score under. Refuses when *requested* disagrees with the artifact's arm.

    ``requested=None`` means "not stated": the artifact's own arm is used.
    """
    recorded = recorded_arm(workflow_folderpath)
    if requested is not None:
        refuse_retired(requested)
    refuse_retired(recorded)
    if requested is None:
        return recorded
    if requested != recorded:
        raise ArmMismatchError(
            f"refusing to score {workflow_folderpath}: --arm {requested!r} was given, but its "
            f"artifacts were trained with arm {recorded!r} (training_provenance.json). "
            f"Score it with --arm {recorded!r}, or retrain under {requested!r} first."
        )
    return recorded


def build_cases(holdout: dict) -> list[dict]:
    """Scoring cases from an exported holdout document (``not_here_holdout.json``).

    For every head H:

    * in scope: each held-out utterance of H's own commands, gold = that command;
    * not here: each held-out utterance of any other command of the workflow, gold = not
      here. The pool is the union of every head's own held-out utterances, so each command's
      held-out utterances appear once per (utterance, command).

    The cases do NOT depend on the arm. An earlier version dropped the utterances a head
    trained as not-here rows; those rows differ by arm, so that dropped a different set under
    each arm and the paired comparison could not pair the cases. The overlap is a leak only
    for the ancestor arms, and it is small (a held-out text rarely recurs as an ancestor row).
    """
    heads: dict[str, dict] = holdout["heads"]
    pool: dict[tuple[str, str], str] = {}
    for head in heads.values():
        for record in head["own_heldout"]:
            pool.setdefault((record["utterance"], record["label"]), record["label"])

    cases: list[dict] = []
    for head_name in sorted(heads):
        head = heads[head_name]
        own = set(head["own_labels"])
        for record in head["own_heldout"]:
            cases.append({"head": head_name, "utterance": record["utterance"],
                          "expected": record["label"], "in_scope": True})
        for (utterance, label) in sorted(pool):
            if label in own:
                continue
            cases.append({"head": head_name, "utterance": utterance,
                          "expected": label, "in_scope": False})
    return cases


def load_holdout(workflow_folderpath: str) -> dict:
    path = os.path.join(workflow_folderpath, "___command_info", "not_here_holdout.json")
    if not os.path.isfile(path):
        raise SystemExit(
            f"{path} does not exist. It is written only by a training run under an experiment "
            f"arm (experiment_reserved_class_arm); train the workflow under an arm first."
        )
    return _load_json(path)


# ---------------------------------------------------------------- predict (real routers)

def training_rows_per_head(workflow_folderpath: str) -> dict[str, int]:
    """Rows each context trained on, from the trainer's provenance (included rows only)."""
    from fastworkflow.train.determinism import ProvenanceRecorder

    rows: Counter = Counter()
    for (context, _command), record in ProvenanceRecorder.load_context_records(workflow_folderpath).items():
        if record.status.value in _ACCEPTED_STATUSES:
            rows[context] += int(record.row_count or 0)
    return dict(rows)


def predict_cases(workflow_folderpath: str, cases: Sequence[dict], arm: Optional[str]) -> dict[str, Any]:
    """Run every case through its head's CommandRouter in the current artifact version.

    Refuses (ArmMismatchError) before loading any model when *arm* disagrees with the arm
    the artifacts were trained with.
    """
    from fastworkflow.train import artifact_versioning
    from fastworkflow.model_pipeline_training import CommandRouter

    effective_arm = check_arm(workflow_folderpath, arm)
    version = artifact_versioning.resolve_current_version(workflow_folderpath)
    if version is None:
        raise SystemExit(f"{workflow_folderpath} has no trained artifact version; train it first")
    base = artifact_versioning.version_dir(workflow_folderpath, version)

    routers: dict[str, CommandRouter] = {}
    out_cases = []
    for case in cases:
        head = case["head"]
        if head not in routers:
            routers[head] = CommandRouter(str(base / artifact_versioning.context_folder_name(head)))
        _labels, details = routers[head].predict_with_details(case["utterance"])
        out_cases.append({
            **case,
            "top_label": details["top_label"],
            "topk": details["topk_labels"],
            "confident": details["confident"],
        })
    return {"schema": SCHEMA_VERSION, "arm": effective_arm, "workflow_version": version,
            "training_rows": training_rows_per_head(workflow_folderpath), "cases": out_cases}


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
    parser = argparse.ArgumentParser(prog="score_not_here", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p_predict = sub.add_parser("predict", help="run cases through the trained heads")
    p_predict.add_argument("--workflow", required=True)
    p_predict.add_argument("--cases", required=True)
    p_predict.add_argument("--out", required=True)
    p_predict.add_argument("--arm", default=None, help="label recorded in the results")

    p_cases = sub.add_parser("cases", help="cases file from a training run's exported persona holdout")
    p_cases.add_argument("--workflow", required=True)
    p_cases.add_argument("--out", required=True)

    p_score = sub.add_parser("score", help="metrics from a predict results file")
    p_score.add_argument("--results", required=True)
    p_score.add_argument("--out", default=None)

    p_compare = sub.add_parser("compare", help="paired McNemar between two results files")
    p_compare.add_argument("--a", required=True)
    p_compare.add_argument("--b", required=True)
    p_compare.add_argument("--out", default=None)

    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "cases":
        cases = build_cases(load_holdout(args.workflow))
        _write_json(args.out, {"schema": SCHEMA_VERSION, "cases": cases})
        in_scope = sum(1 for case in cases if case["in_scope"])
        print(f"wrote {len(cases)} cases ({in_scope} in scope, {len(cases) - in_scope} not here) to {args.out}")
        return 0
    if args.command == "predict":
        cases = _load_json(args.cases)["cases"]
        results = predict_cases(args.workflow, cases, args.arm)
        _write_json(args.out, results)
        print(f"wrote {len(results['cases'])} case results to {args.out}")
        return 0
    if args.command == "score":
        summary = summarize(_load_json(args.results))
        _write_json(args.out, summary)
        print(format_summary(summary))
        return 0
    comparison = compare(_load_json(args.a), _load_json(args.b))
    _write_json(args.out, comparison)
    print(format_comparison(comparison))
    return 0


if __name__ == "__main__":
    sys.exit(main())

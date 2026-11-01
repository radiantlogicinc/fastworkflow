"""Phase 9 step 1: the 12 trainings (4 arms x 3 train seeds), their pooled scoring, and the bar.

Runs sequentially, one subprocess per training (so a run's memory is released before the next
starts). Each run is `not_here_pilot --arm A --train-seed S` in its own fresh scratch subdir.
A failed run is recorded and the runner goes on. Then the runs are pooled per arm and judged
against the bars recorded in fix-3rsq.9 (BARS). Nothing here trains unless the CLI is run
without --dry-run; --dry-run prints the plan and touches nothing.

Bars (owner, fix-3rsq.9 BARS), per arm against control, pooled over its three trainings:
  * wrong declines fall: 95% CI of (arm - control) on the in-scope wrong-decline rate
    excludes zero on the falling side (upper bound < 0);
  * wrong accepts rise by at most 2 points at the top of their 95% CI (upper <= 0.02);
  * in-scope top-1 does not fall: lower bound of (arm - control) >= -0.01.
An arm PASSES when all three hold. Between passing arms: sparse 1 seed wins unless sparse 8
seeds is measurably better (McNemar p < 0.05 on wrong accepts or in-scope top-1, with fewer
errors). No arm passes: stop and report.

Pooling. The three seeds of an arm are concatenated, case by case, in seed order. Every seed
scores the same cases, so the pooled list pairs with the control's pooled list index by index.
Each pooled case therefore appears three times. The McNemar tests and the Wilson intervals are
computed on the pooled counts, and the paired-difference interval uses the same pooled pairs.
The three seeds are repeats on the same cases, not independent samples of cases, so the pooled
intervals are narrower than a fresh draw of cases would give. The report says so, and shows each
seed's rates so the spread between seeds is visible.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from fastworkflow.train import score_not_here as scorer
from fastworkflow.train.not_here_pilot import CACHE_FROM_ENV, SOURCE_ENV, WORKFLOW_DIRNAME, required_path

#: Every arm the step 1 measurement reported on, in report order. ``control`` and ``ratio_0_5``
#: are RETIRED (Phase 9 step 5): their results stay as history, and ``plan_runs`` refuses them.
ARMS: tuple[str, ...] = ("control", "sparse_1", "sparse_8", "ratio_0_5")
#: The arms a new run may train (the retired ones are absent).
TRAINABLE_ARMS: tuple[str, ...] = ("sparse_1", "sparse_8")
TRAIN_SEEDS: tuple[int, ...] = (1, 2, 3)
BASELINE = "control"  # history only: the recorded control results are the comparison base
Z_95 = 1.959963984540054
MAX_WRONG_ACCEPT_RISE = 0.02
MIN_TOP1_DIFFERENCE = -0.01
ALPHA = 0.05
SUMMARY_FILENAME = "summary.json"
RESULTS_FILENAME = "results.json"
TIMINGS_FILENAME = "timings.json"


@dataclass(frozen=True)
class RunSpec:
    arm: str
    seed: int
    directory: str

    @property
    def name(self) -> str:
        return f"{self.arm}_seed{self.seed}"


# ---------------------------------------------------------------- plan

def refuse_retired_arms(arms: Sequence[str]) -> None:
    """Raise ValueError naming every retired arm in *arms*, with the trainer's reason."""
    retired = [arm for arm in arms if arm in ("control", "ratio_0_5")]
    if not retired:
        return
    from fastworkflow.model_pipeline_training import RETIRED_ARMS

    raise ValueError("; ".join(f"arm {arm!r} is retired: {RETIRED_ARMS[arm]}" for arm in retired))


def plan_runs(root: str, arms: Sequence[str] = TRAINABLE_ARMS, seeds: Sequence[int] = TRAIN_SEEDS) -> list[RunSpec]:
    """The trainings in run order: arm by arm, seeds in order within each arm."""
    refuse_retired_arms(arms)
    return [RunSpec(arm=arm, seed=int(seed), directory=os.path.join(root, f"{arm}_seed{seed}"))
            for arm in arms for seed in seeds]


def run_is_done(directory: str) -> bool:
    return os.path.isfile(os.path.join(directory, SUMMARY_FILENAME))


def pilot_command(spec: RunSpec, *, source: str, cache_from: str, env_file: str,
                  passwords_file: Optional[str]) -> list[str]:
    command = [sys.executable, "-m", "fastworkflow.train.not_here_pilot",
               "--arm", spec.arm, "--train-seed", str(spec.seed),
               "--scratch", spec.directory, "--source", source,
               "--cache-from", cache_from, "--env-file", env_file]
    if passwords_file:
        command += ["--passwords-file", passwords_file]
    return command


# ---------------------------------------------------------------- pooling

def _case_key(case: dict) -> tuple[str, str, bool]:
    return (case["head"], case["utterance"], bool(case["in_scope"]))


def pool_arm(results_by_seed: Sequence[dict], arm: str) -> dict:
    """Concatenate one arm's per-seed results, case by case, in seed order.

    Every seed must score the same cases in the same order; otherwise the pooled pairs would not
    line up with the control's, and the function raises rather than pool a mismatch.
    """
    if not results_by_seed:
        raise ValueError(f"no finished runs to pool for arm {arm!r}")
    reference = [_case_key(case) for case in results_by_seed[0]["cases"]]
    pooled: list[dict] = []
    for index, results in enumerate(results_by_seed):
        keys = [_case_key(case) for case in results["cases"]]
        if keys != reference:
            raise ValueError(f"arm {arm!r}: run {index + 1} scores different cases from run 1")
        pooled.extend(results["cases"])
    return {"schema": scorer.SCHEMA_VERSION, "arm": arm, "runs_pooled": len(results_by_seed), "cases": pooled}


# ---------------------------------------------------------------- paired differences

def _indicator(case: dict, metric: str) -> Optional[int]:
    """1 when the case is an error for the metric (or a correct top-1 for 'top1'); None when not counted."""
    outcome = scorer.case_outcome(case)
    if metric == "wrong_declines":
        return None if not case["in_scope"] else int(bool(outcome["wrong_decline"]))
    if metric == "wrong_accepts":
        return None if case["in_scope"] else int(bool(outcome["wrong_accept"]))
    if metric == "top1":
        return None if not case["in_scope"] else int(not outcome["top1_miss"])
    raise ValueError(f"unknown metric {metric!r}")


def paired_difference(arm_cases: Sequence[dict], control_cases: Sequence[dict], metric: str) -> dict:
    """Difference (arm - control) of the metric's indicator rate, with a paired Wald 95% interval.

    For the error metrics a negative difference is an improvement. For 'top1' the indicator is
    'correct', so a positive difference is an improvement. Pairs are the same case under both arms.
    """
    if len(arm_cases) != len(control_cases):
        raise ValueError("the arms score different numbers of cases")
    pairs: list[tuple[int, int]] = []
    for arm_case, control_case in zip(arm_cases, control_cases):
        if _key_mismatch(arm_case, control_case):
            raise ValueError("the arms score different cases")
        x = _indicator(arm_case, metric)
        y = _indicator(control_case, metric)
        if x is None or y is None:
            continue
        pairs.append((x, y))
    n = len(pairs)
    if n == 0:
        return {"metric": metric, "n": 0, "difference": None, "lower": None, "upper": None,
                "arm_rate": None, "control_rate": None}
    n10 = sum(1 for x, y in pairs if x == 1 and y == 0)
    n01 = sum(1 for x, y in pairs if x == 0 and y == 1)
    difference = (n10 - n01) / n
    variance = max(0.0, (n10 + n01 - (n10 - n01) ** 2 / n) / (n * n))
    half = Z_95 * math.sqrt(variance)
    return {
        "metric": metric,
        "n": n,
        "arm_rate": sum(x for x, _ in pairs) / n,
        "control_rate": sum(y for _, y in pairs) / n,
        "difference": difference,
        "lower": difference - half,
        "upper": difference + half,
        "only_arm_differs": n10,
        "only_control_differs": n01,
    }


def _key_mismatch(a: dict, b: dict) -> bool:
    return _case_key(a) != _case_key(b)


# ---------------------------------------------------------------- bar

def evaluate_arm(arm_cases: Sequence[dict], control_cases: Sequence[dict]) -> dict:
    """The owner's three bars for one arm against control, plus the McNemar tests behind them."""
    decline = paired_difference(arm_cases, control_cases, "wrong_declines")
    accept = paired_difference(arm_cases, control_cases, "wrong_accepts")
    top1 = paired_difference(arm_cases, control_cases, "top1")
    decline["passes"] = decline["upper"] is not None and decline["upper"] < 0
    accept["passes"] = accept["upper"] is not None and accept["upper"] <= MAX_WRONG_ACCEPT_RISE
    top1["passes"] = top1["lower"] is not None and top1["lower"] >= MIN_TOP1_DIFFERENCE
    mcnemar = scorer.compare(
        {"arm": BASELINE, "cases": list(control_cases)},
        {"arm": "arm", "cases": list(arm_cases)},
    )["paired"]
    return {
        "wrong_declines": decline,
        "wrong_accepts": accept,
        "top1": top1,
        "mcnemar_vs_control": mcnemar,
        "passes": bool(decline["passes"] and accept["passes"] and top1["passes"]),
    }


def measurably_better(comparison: dict) -> bool:
    """Whether the second arm (8 seeds) beats the first (1 seed) measurably: McNemar p < 0.05 on
    wrong accepts or in-scope top-1, with fewer errors on that metric."""
    for metric in ("wrong_accept", "top1"):
        block = comparison["paired"][metric]
        if block["p_two_sided"] < ALPHA and block["errors_b"] < block["errors_a"]:
            return True
    return False


def choose(evaluations: dict[str, dict], one_vs_eight: Optional[dict]) -> dict:
    """The verdict from the arms that pass. Sparse 1 over sparse 8 unless 8 is measurably better."""
    passing = [arm for arm in ("sparse_1", "sparse_8", "ratio_0_5")
               if evaluations.get(arm, {}).get("passes")]
    if not passing:
        return {"decision": "no arm passes: stop and report; the owner decides",
                "chosen": None, "passing": []}
    sparse = [arm for arm in ("sparse_1", "sparse_8") if arm in passing]
    if len(sparse) == 2 and one_vs_eight is not None:
        chosen = "sparse_8" if measurably_better(one_vs_eight) else "sparse_1"
        reason = ("sparse 8 is measurably better on wrong accepts or top-1" if chosen == "sparse_8"
                  else "sparse 8 is not measurably better, so the one-seed arm wins")
    elif sparse:
        chosen = sparse[0]
        reason = "the only passing sparse arm"
    else:
        chosen = "ratio_0_5"
        reason = "only today's class at ratio 0.5 passes"
    return {"decision": f"choose {chosen}: {reason}", "chosen": chosen, "passing": passing}


# ---------------------------------------------------------------- report

def build_report(runs: Sequence[dict], results_by_arm_seed: dict[tuple[str, int], dict]) -> dict:
    """The pooled report for finished runs. *runs* carries each run's status and timing."""
    arms_present = sorted({run["arm"] for run in runs}, key=ARMS.index)
    pooled: dict[str, dict] = {}
    incomplete: dict[str, str] = {}
    for arm in arms_present:
        seeds = sorted(seed for (a, seed) in results_by_arm_seed if a == arm)
        if not seeds:
            incomplete[arm] = "no finished run"
            continue
        try:
            pooled[arm] = pool_arm([results_by_arm_seed[(arm, seed)] for seed in seeds], arm)
        except ValueError as error:
            incomplete[arm] = str(error)

    per_arm = {arm: {**scorer.summarize(data), "runs_pooled": data["runs_pooled"],
                     "per_seed": _per_seed(arm, results_by_arm_seed)}
               for arm, data in pooled.items()}

    evaluations: dict[str, dict] = {}
    if BASELINE in pooled:
        for arm, data in pooled.items():
            if arm != BASELINE:
                evaluations[arm] = evaluate_arm(data["cases"], pooled[BASELINE]["cases"])
    one_vs_eight = None
    if "sparse_1" in pooled and "sparse_8" in pooled:
        one_vs_eight = scorer.compare(pooled["sparse_1"], pooled["sparse_8"])

    all_three = all(
        sum(1 for run in runs if run["arm"] == arm and run["status"] in ("ok", "resumed")) == len(TRAIN_SEEDS)
        for arm in ARMS
    )
    if not all_three or incomplete or BASELINE not in pooled:
        verdict = {"decision": "INCOMPLETE: every arm needs its three finished trainings before the bar applies",
                   "chosen": None, "passing": []}
    else:
        verdict = choose(evaluations, one_vs_eight)

    return {
        "schema": 1,
        "method": {
            "pooling": "the three seeds of an arm are concatenated case by case in seed order; "
                       "pairs line up with control index by index",
            "caveat": "the seeds repeat the same cases, so pooled intervals are narrower than a "
                      "fresh draw of cases would give; per-seed rates are reported beside them",
            "intervals": "Wilson 95% for each rate; paired Wald 95% for each difference (arm - control)",
            "tests": "exact two-sided McNemar from scorer.compare on paired per-case errors",
            "bars": {
                "wrong_declines": "upper bound of the difference < 0",
                "wrong_accepts": f"upper bound of the difference <= {MAX_WRONG_ACCEPT_RISE}",
                "top1": f"lower bound of the difference >= {MIN_TOP1_DIFFERENCE}",
                "one_seed_rule": "between passing sparse arms, 1 seed unless 8 seeds has McNemar p < 0.05 "
                                 "on wrong accepts or top-1 with fewer errors",
            },
        },
        "runs": list(runs),
        "incomplete": incomplete,
        "per_arm": per_arm,
        "evaluations": evaluations,
        "one_vs_eight": one_vs_eight["paired"] if one_vs_eight else None,
        "verdict": verdict,
    }


def _per_seed(arm: str, results_by_arm_seed: dict[tuple[str, int], dict]) -> dict[str, dict]:
    out = {}
    for (a, seed), results in sorted(results_by_arm_seed.items()):
        if a == arm:
            overall = scorer.summarize(results)["overall"]
            out[str(seed)] = {"wrong_declines": overall["wrong_declines"]["rate"],
                              "wrong_accepts": overall["wrong_accepts"]["rate"],
                              "top1": overall["top1"]["rate"], "in_list": overall["in_list"]["rate"]}
    return out


def format_markdown(report: dict) -> str:
    def pct(value):
        return "n/a" if value is None else f"{100 * value:.1f}%"

    def interval(block):
        if not block or block.get("wilson95") is None:
            return "n/a"
        low, high = block["wilson95"]
        return f"{pct(block['rate'])} [{pct(low)}, {pct(high)}]"

    lines = ["# Phase 9 step 1: sparse 'not here' class", "",
             f"Verdict: **{report['verdict']['decision']}**", "",
             "## Per arm (pooled over the arm's trainings)", "",
             "| arm | runs | in-scope n | out-of-scope n | wrong declines | wrong accepts | top-1 | in-list |",
             "|---|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        if arm not in report["per_arm"]:
            lines.append(f"| {arm} | - | - | - | incomplete | - | - | - |")
            continue
        block = report["per_arm"][arm]
        overall = block["overall"]
        lines.append(
            f"| {arm} | {block['runs_pooled']} | {overall['in_scope_n']} | {overall['out_of_scope_n']} | "
            f"{interval(overall['wrong_declines'])} | {interval(overall['wrong_accepts'])} | "
            f"{interval(overall['top1'])} | {interval(overall['in_list'])} |")
    lines += ["", "## Against control (arm - control; paired 95% intervals)", "",
              "| arm | wrong declines diff | bar | wrong accepts diff | bar | top-1 diff | bar | passes |",
              "|---|---|---|---|---|---|---|---|"]
    for arm, ev in report["evaluations"].items():
        d, a, t = ev["wrong_declines"], ev["wrong_accepts"], ev["top1"]
        lines.append(
            f"| {arm} | {_signed(d)} | {'falls' if d['passes'] else 'no'} | {_signed(a)} | "
            f"{'ok' if a['passes'] else 'no'} | {_signed(t)} | {'ok' if t['passes'] else 'no'} | "
            f"{'yes' if ev['passes'] else 'no'} |")
    lines += ["", "## McNemar p-values against control (exact, two-sided)", ""]
    for arm, ev in report["evaluations"].items():
        mc = ev["mcnemar_vs_control"]
        lines.append(f"- {arm}: " + ", ".join(
            f"{metric} p={mc[metric]['p_two_sided']:.3g}" for metric in
            ("wrong_decline", "wrong_accept", "top1", "in_list")))
    if report["one_vs_eight"]:
        lines += ["", "## Sparse 1 vs sparse 8 (McNemar)", ""]
        for metric, block in report["one_vs_eight"].items():
            lines.append(f"- {metric}: errors sparse_1 {block['errors_a']}, sparse_8 {block['errors_b']}, "
                         f"p={block['p_two_sided']:.3g}")
    lines += ["", "## Runs", "", "| run | status | train seconds | wall seconds |", "|---|---|---|---|"]
    for run in report["runs"]:
        lines.append(f"| {run['name']} | {run['status']} | {run.get('train_seconds', '-')} | "
                     f"{run.get('wall_seconds', '-')} |")
    if report["incomplete"]:
        lines += ["", "## Incomplete arms", ""] + [f"- {arm}: {why}" for arm, why in report["incomplete"].items()]
    lines += ["", "## Method", "", report["method"]["pooling"] + ".", "", report["method"]["caveat"] + ".",
              "Intervals: " + report["method"]["intervals"] + ".", ""]
    return "\n".join(lines) + "\n"


def _signed(block: dict) -> str:
    if block["difference"] is None:
        return "n/a"
    return f"{100 * block['difference']:+.1f} [{100 * block['lower']:+.1f}, {100 * block['upper']:+.1f}] pts"


# ---------------------------------------------------------------- run

def prune_models(directory: str) -> None:
    """Remove a finished copy's trained models. The caches, the provenance and the outputs stay."""
    info = os.path.join(directory, WORKFLOW_DIRNAME, "___command_info")
    for name in ("versions", "current"):
        path = os.path.join(info, name)
        if os.path.islink(path) or os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)


def _read_json(path: str) -> Optional[dict]:
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _run_one(spec: RunSpec, args: argparse.Namespace, repo_root: str, log_dir: str) -> dict:
    if args.resume and run_is_done(spec.directory):
        timings = _read_json(os.path.join(spec.directory, TIMINGS_FILENAME)) or {}
        return {"name": spec.name, "arm": spec.arm, "seed": spec.seed, "status": "resumed",
                "train_seconds": timings.get("train_seconds"), "directory": spec.directory}
    if os.path.exists(spec.directory):
        shutil.rmtree(spec.directory)  # a partial run: the pilot copies into a fresh directory
    command = pilot_command(spec, source=args.source, cache_from=args.cache_from,
                            env_file=args.env_file, passwords_file=args.passwords_file)
    log_path = os.path.join(log_dir, f"{spec.name}.log")
    started = time.monotonic()
    with open(log_path, "w", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=repo_root, stdout=log, stderr=subprocess.STDOUT)
    wall = round(time.monotonic() - started, 2)
    timings = _read_json(os.path.join(spec.directory, TIMINGS_FILENAME)) or {}
    ok = completed.returncode == 0 and run_is_done(spec.directory)
    record = {"name": spec.name, "arm": spec.arm, "seed": spec.seed, "directory": spec.directory,
              "status": "ok" if ok else "failed", "returncode": completed.returncode,
              "wall_seconds": wall, "train_seconds": timings.get("train_seconds"), "log": log_path}
    if not ok:
        with open(log_path, encoding="utf-8", errors="replace") as log:
            record["failure_tail"] = "".join(log.readlines()[-25:])
    elif args.prune:
        prune_models(spec.directory)
        record["pruned"] = True
    return record


def _collect(runs: Sequence[dict]) -> dict[tuple[str, int], dict]:
    collected = {}
    for run in runs:
        if run["status"] in ("ok", "resumed"):
            results = _read_json(os.path.join(run["directory"], RESULTS_FILENAME))
            if results is not None:
                collected[(run["arm"], run["seed"])] = results
    return collected


def _dry_run(specs: Sequence[RunSpec], args: argparse.Namespace) -> None:
    print("Planned trainings (dry run; nothing is created, copied, trained or run):")
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for index, spec in enumerate(specs, 1):
        skip = " [would be SKIPPED by --resume: summary.json exists]" if args.resume and run_is_done(spec.directory) else ""
        print(f"  {index:2d}. {spec.name}: {spec.directory}{skip}")
        print("      " + " ".join(pilot_command(spec, source=args.source, cache_from=args.cache_from,
                                              env_file=args.env_file, passwords_file=args.passwords_file)))
    print(f"  working directory for each run: {repo}")
    print(f"  prune trained models after each finished run: {args.prune}")
    print("Bars:")
    for line in build_report([], {})["method"]["bars"].items():
        print(f"  {line[0]}: {line[1]}")
    print(f"Report files: {os.path.join(args.root, 'step1_report.json')} and {os.path.join(args.root, 'step1_report.md')}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="not_here_step1", description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", required=True, help="directory holding the 12 run subdirectories and the report")
    parser.add_argument("--cache-from", default=None,
                        help="a workflow checkout whose ___command_info holds the filled caches (read only). "
                             f"Required unless ${CACHE_FROM_ENV} is set")
    parser.add_argument("--env-file", required=True, help="the env file the caches were generated with")
    parser.add_argument("--passwords-file", default=None)
    parser.add_argument("--source", default=None,
                        help=f"the worktree workflow (read only). Required unless ${SOURCE_ENV} is set")
    parser.add_argument("--arms", default=",".join(TRAINABLE_ARMS))
    parser.add_argument("--seeds", default=",".join(str(s) for s in TRAIN_SEEDS))
    parser.add_argument("--prune", action="store_true", help="remove each finished copy's trained models after scoring")
    parser.add_argument("--resume", action="store_true", help="skip runs whose summary.json already exists")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and the bars, then exit")
    args = parser.parse_args(argv)

    try:
        args.source = required_path(args.source, SOURCE_ENV, "--source")
        args.cache_from = required_path(args.cache_from, CACHE_FROM_ENV, "--cache-from")
    except ValueError as error:
        parser.error(str(error))
    arms = [a for a in args.arms.split(",") if a]
    seeds = [int(s) for s in args.seeds.split(",") if s]
    specs = plan_runs(os.path.abspath(args.root), arms, seeds)
    if args.dry_run:
        _dry_run(specs, args)
        return 0

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    log_dir = os.path.join(args.root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    runs = []
    for spec in specs:  # sequential: never two trainings at once
        record = _run_one(spec, args, repo_root, log_dir)
        runs.append(record)
        _write(os.path.join(args.root, "runs.json"), {"runs": runs})
        print(f"{record['name']}: {record['status']}", flush=True)

    report = build_report(runs, _collect(runs))
    _write(os.path.join(args.root, "step1_report.json"), report)
    with open(os.path.join(args.root, "step1_report.md"), "w", encoding="utf-8") as handle:
        handle.write(format_markdown(report))
    print(report["verdict"]["decision"])
    return 0


def _write(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


if __name__ == "__main__":
    sys.exit(main())

"""Phase 10 step 1c (offline): the two planning-command head arms, three trainings each, pooled report.

EXPERIMENT DRIVER. Two offline arms on one named test set (``tests/planning_test_sets/library.json``):

* ``heads_not_here`` (the BASELINE): the sparse "not here" class with planning commands only in it
  (``model_pipeline_training.experiment_planning_heads("not_here")``). Not production any more.
* ``heads_labelled``: the same sparse arm with planning commands as head labels. This is PRODUCTION
  (the default layout); the run passes ``experiment_planning_heads("labelled")`` explicitly.

Each training runs in its own subprocess (``planning_step1 worker ...``), one at a time, so a run's
memory is released before the next starts. A worker copies ``--source`` (read only) into a fresh
scratch directory, copies only the two cache directories from ``--cache-from`` (read only), blocks
LLM access the way ``not_here_pilot`` does (invalid keys, closed proxy base), trains the copy, fails
the run if any command fell back to seeds (a fallback means a call was attempted), scores the copy on
the test set with ``score_planning_match.predict_arm``, and prunes the copy's trained models when
``--prune`` is given. The caches, provenance and outputs stay.

Then the runs are pooled per arm (the three seeds' case lists concatenated in seed order, which
pairs with the baseline index by index) and compared with ``score_planning_match.compare_arms``.
The recommended bars are printed as GUIDANCE only. The owner decides; this driver never does.

Pooling caveat: the three seeds score the same cases, so the pooled counts repeat each case three
times. Pooled rates and point differences are right; pooled intervals and McNemar p-values are too
narrow. The report therefore shows per-seed rates and per-seed paired differences beside them.

Usage::

    python -m fastworkflow.train.planning_step1 run --root R --source W --cache-from C --env-file E
        [--passwords-file P] [--test-set T] [--arms ...] [--seeds 1,2,3] [--prune] [--resume] [--dry-run]

``--dry-run`` prints the plan and touches nothing. The LLM is never called by this driver.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from typing import Any, Optional, Sequence

from fastworkflow.nlu_labels import label_of
from fastworkflow.train import planning_test_set as pts
from fastworkflow.train import score_planning_match as spm
from fastworkflow.train.not_here_pilot import PilotFailure
from fastworkflow.train.not_here_step1 import prune_models

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_TEST_SET = os.path.join(REPO_ROOT, "tests", "planning_test_sets", "library.json")
BASELINE = spm.BASELINE_ARM  # heads_not_here
ARMS: tuple[str, ...] = (spm.ARM_HEADS_NOT_HERE, spm.ARM_HEADS_LABELLED)
TRAIN_SEEDS: tuple[int, ...] = (1, 2, 3)
#: Both offline arms train inside the sparse "not here" class (the production arm).
SPARSE_ARM = "sparse_1"
WORKFLOW_DIRNAME = "workflow"
RESULTS_FILENAME = "results.json"
SUMMARY_FILENAME = "summary.json"  # written last: its presence marks a finished run
TIMINGS_FILENAME = "timings.json"
RUNS_FILENAME = "runs.json"
REPORT_JSON = "planning_step1_report.json"
REPORT_MD = "planning_step1_report.md"
CACHE_DIRS = ("utterance_cache", "param_example_cache")


class StepFailure(RuntimeError):
    """A worker step failed in a way that invalidates the run."""


# ---------------------------------------------------------------- plan

def plan_runs(root: str, arms: Sequence[str] = ARMS, seeds: Sequence[int] = TRAIN_SEEDS) -> list[dict]:
    """The trainings in run order: arm by arm (baseline first), seeds in order within each arm."""
    for arm in arms:
        if arm not in spm.OFFLINE_ARMS:
            raise ValueError(f"{arm!r} is not an offline arm; choose from {list(spm.OFFLINE_ARMS)}")
    return [{"name": f"{arm}_seed{seed}", "arm": arm, "seed": int(seed),
             "directory": os.path.join(root, f"{arm}_seed{seed}")}
            for arm in arms for seed in seeds]


def is_done(directory: str) -> bool:
    return os.path.isfile(os.path.join(directory, SUMMARY_FILENAME))


def worker_command(spec: dict, args: argparse.Namespace) -> list[str]:
    command = [sys.executable, "-m", "fastworkflow.train.planning_step1", "worker",
               "--arm", spec["arm"], "--train-seed", str(spec["seed"]),
               "--scratch", spec["directory"], "--source", args.source,
               "--cache-from", args.cache_from, "--env-file", args.env_file,
               "--test-set", args.test_set]
    if args.passwords_file:
        command += ["--passwords-file", args.passwords_file]
    if args.prune:
        command.append("--prune")
    return command


# ---------------------------------------------------------------- pooling and comparison

def pool_results(results_by_seed: Sequence[dict], arm: str) -> dict:
    """Concatenate one arm's per-seed case lists in seed order. Every seed must score the same cases."""
    if not results_by_seed:
        raise ValueError(f"no finished runs to pool for arm {arm!r}")
    reference = [spm._key(case) for case in results_by_seed[0]["cases"]]
    cases: list[dict] = []
    for index, results in enumerate(results_by_seed):
        if [spm._key(case) for case in results["cases"]] != reference:
            raise ValueError(f"arm {arm!r}: run {index + 1} scores different cases from run 1")
        cases.extend(results["cases"])
    return {"schema": spm.SCHEMA_VERSION, "arm": arm, "runs_pooled": len(results_by_seed), "cases": cases}


def routing_diagnostics(results: dict, planning_names: Sequence[str]) -> dict:
    """Where the wrong routings go, on the heldout cases: a callee request routed to a planning command
    is the 'labelled heads steal callee requests' failure the report must show."""
    planning_labels = {label_of(name) for name in planning_names}
    heldout = [case for case in results["cases"] if spm._eligible(case)]
    callee = [c for c in heldout if c["kind"] == pts.KIND_CALLEE]
    planning = [c for c in heldout if c["kind"] == pts.KIND_PLANNING]
    to_planning = sum(1 for c in callee
                      if c.get("top_label") and label_of(c["top_label"]) in planning_labels)
    return {
        "callee_n": len(callee),
        "callee_routed_to_a_planning_command": to_planning,
        "callee_wrong_accepts": sum(spm.case_verdict(c.get("top_label"), c["expected"])["wrong_accept"]
                                    for c in callee),
        "planning_n": len(planning),
        "planning_wrong_accepts": sum(spm.case_verdict(c.get("top_label"), c["expected"])["wrong_accept"]
                                      for c in planning),
    }


def _seed_block(results: dict) -> dict:
    summary = spm.summarize(results)
    return {
        "planning_routed": summary["planning_routed"],
        "callee_routed": summary["callee_routed"],
        "wrong_accepts": summary["wrong_accepts"],
        "heldout_n": summary["heldout_n"],
        "training_rows_total": sum((results.get("training_rows") or {}).values()),
        "training_rows": results.get("training_rows") or {},
        "train_seconds": results.get("train_seconds"),
    }


def build_report(runs: Sequence[dict], results_by_arm_seed: dict[tuple[str, int], dict],
                 planning_names: Sequence[str]) -> dict:
    """Pooled and per-seed numbers for every arm against the baseline. Guidance only; no verdict."""
    arms_present = [arm for arm in ARMS if any(a == arm for a, _ in results_by_arm_seed)]
    incomplete = {arm: f"{sum(1 for r in runs if r['arm'] == arm and r['status'] in ('ok', 'resumed'))}"
                       f" of {len(TRAIN_SEEDS)} trainings finished" for arm in ARMS
                  if sum(1 for r in runs if r["arm"] == arm and r["status"] in ("ok", "resumed")) < len(TRAIN_SEEDS)}
    pooled: dict[str, dict] = {}
    per_seed: dict[str, dict] = {}
    diagnostics: dict[str, dict] = {}
    for arm in arms_present:
        seeds = sorted(seed for (a, seed) in results_by_arm_seed if a == arm)
        seed_results = [results_by_arm_seed[(arm, seed)] for seed in seeds]
        try:
            pooled[arm] = pool_results(seed_results, arm)
        except ValueError as error:
            incomplete[arm] = str(error)
            continue
        per_seed[arm] = {str(seed): _seed_block(results_by_arm_seed[(arm, seed)]) for seed in seeds}
        diagnostics[arm] = {
            "pooled": routing_diagnostics(pooled[arm], planning_names),
            "per_seed": {str(seed): routing_diagnostics(results_by_arm_seed[(arm, seed)], planning_names)
                         for seed in seeds},
        }
    report: dict[str, Any] = {
        "schema": 1,
        "baseline": BASELINE,
        "method": {
            "pooling": "the three seeds of an arm are concatenated case by case in seed order; pairs line up "
                       "with the baseline index by index",
            "caveat": "the seeds repeat the same heldout cases, so pooled intervals and McNemar p-values "
                      "count each case three times and are too narrow; per-seed paired comparisons are shown beside them",
            "intervals": "Wilson 95% for each rate; paired Wald 95% for each difference (arm - baseline)",
            "tests": "exact two-sided McNemar (score_planning_match.paired_indicator_difference)",
            "guidance": "the epic's recommended bars, shown for reference only; the owner decides",
        },
        "runs": list(runs),
        "incomplete": incomplete,
        "pooled": {},
        "per_seed": per_seed,
        "diagnostics": diagnostics,
        "comparison_pooled": {},
        "comparison_per_seed": {},
        "guidance_pooled": {},
    }
    for arm, data in pooled.items():
        report["pooled"][arm] = {**spm.summarize(data), "runs_pooled": data["runs_pooled"]}
        if arm == BASELINE or BASELINE not in pooled:
            continue
        if data["runs_pooled"] != pooled[BASELINE]["runs_pooled"]:
            continue  # pairs line up only when both arms pool the same seeds; the arm is incomplete
        comparison = spm.compare_arms(data, pooled[BASELINE])
        report["comparison_pooled"][arm] = comparison
        report["guidance_pooled"][arm] = spm.guidance(comparison)
        report["comparison_per_seed"][arm] = {}
        for seed in sorted(seed for (a, seed) in results_by_arm_seed if a == arm):
            base_key = (BASELINE, seed)
            if base_key in results_by_arm_seed:
                report["comparison_per_seed"][arm][str(seed)] = spm.compare_arms(
                    results_by_arm_seed[(arm, seed)], results_by_arm_seed[base_key])
    return report


# ---------------------------------------------------------------- rescore (stored predictions, no training)

def rescore(root: str, test_set_path: str, live_path: Optional[str] = None) -> dict:
    """Rescore the finished runs under *root* on the deduped held-out set, with no training and no prediction.

    The stored per-case predictions are filtered to the held-out cases of *test_set_path*. A case
    that any run's head trained on (its ``trained_on`` flag) is excluded from every run, so all arms
    and seeds score exactly the same analysis set. The stored flags are kept in ``rescore`` so the
    exclusions are visible. With *live_path* (a live arm's results), that arm is also scored on the
    analysis cases that fall in its sample, and against each baseline seed on the same cases.
    """
    test_set = pts.load_test_set(test_set_path)
    keep = {spm._key(case) for case in pts.heldout(test_set)}
    runs = (_read_json(os.path.join(root, RUNS_FILENAME)) or {}).get("runs", [])
    filtered: dict[tuple[str, int], dict] = {}
    for run in runs:
        if run["status"] not in ("ok", "resumed"):
            continue
        results = _read_json(os.path.join(run["directory"], RESULTS_FILENAME))
        if results is None:
            continue
        cases = [case for case in results["cases"] if spm._key(case) in keep]
        filtered[(run["arm"], run["seed"])] = {**results, "cases": cases}
    trained_on_by_run = {f"{arm}_seed{seed}": sorted({spm._key(c) for c in r["cases"] if c.get("trained_on")})
                         for (arm, seed), r in filtered.items()}
    excluded = set().union(*(set(keys) for keys in trained_on_by_run.values())) if trained_on_by_run else set()
    analysis = keep - excluded
    scored: dict[tuple[str, int], dict] = {}
    for arm_seed, results in filtered.items():
        cases = [{**case, "trained_on": False} for case in results["cases"] if spm._key(case) in analysis]
        scored[arm_seed] = {**results, "cases": cases}
    planning_names = test_set.get("planning_commands", [])
    report = build_report(runs, scored, planning_names)
    report["rescore"] = {
        "test_set": test_set_path,
        "deduped_heldout_cases": len(keep),
        "trained_on_excluded": len(excluded),
        "analysis_cases": len(analysis),
        "trained_on_by_run": {name: len(keys) for name, keys in trained_on_by_run.items()},
        "planning_cases": sum(1 for key in analysis if key[3] == pts.KIND_PLANNING),
        "callee_cases": sum(1 for key in analysis if key[3] == pts.KIND_CALLEE),
    }
    if live_path:
        report["live"] = _rescore_live(live_path, scored, analysis, keep, planning_names)
    return report


def _rescore_live(live_path: str, scored: dict, analysis: set, keep: set, planning_names: Sequence[str]) -> dict:
    """A live arm's stored results on the deduped cases in its sample, beside each baseline seed on the same cases."""
    live = _read_json(live_path) or {}
    sample_keys = {spm._key(case) for case in live.get("cases", [])}
    in_sample = sample_keys & keep
    same = sample_keys & analysis
    live_cases = [{**c, "trained_on": False} for c in live.get("cases", []) if spm._key(c) in same]
    out = {"arm": live.get("arm"), "sample_cases": len(sample_keys), "sample_in_deduped": len(in_sample),
           "same_as_offline_analysis": len(same), "summary": spm.summarize({"arm": live.get("arm"),
                                                                          "cases": live_cases}),
           "routing": routing_diagnostics({"cases": live_cases}, planning_names), "vs_baseline": {}}
    for (arm, seed), results in sorted(scored.items()):
        if arm != BASELINE:
            continue
        base_cases = [c for c in results["cases"] if spm._key(c) in same]
        if len(base_cases) != len(live_cases):
            continue
        out["vs_baseline"][str(seed)] = spm.compare_arms({"arm": live.get("arm"), "cases": live_cases},
                                                         {"arm": BASELINE, "cases": base_cases})
    return out


# ---------------------------------------------------------------- markdown

def _pct(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def _rate(block: dict) -> str:
    if not block or not block.get("trials"):
        return "n/a"
    low, high = block["wilson95"]
    return f"{block['count']}/{block['trials']} = {_pct(block['rate'])} [{_pct(low)}, {_pct(high)}]"


def _diff(block: dict) -> str:
    if not block or block.get("n") in (0, None):
        return "n/a"
    return (f"{100 * block['difference']:+.1f} [{100 * block['lower']:+.1f}, {100 * block['upper']:+.1f}] pts; "
            f"only-arm {block['only_arm']}, only-baseline {block['only_base']}, McNemar p={block['p_two_sided']:.3g}")


def format_markdown(report: dict) -> str:
    lines = ["# Phase 10 step 1c: planning commands in the heads (offline)", "",
             "GUIDANCE ONLY. The owner decides; nothing below gates anything.", ""]
    if report["incomplete"]:
        lines += ["## Incomplete", ""] + [f"- {arm}: {why}" for arm, why in report["incomplete"].items()] + [""]
    lines += ["## Per arm, pooled over the arm's trainings (heldout cases)", "",
              "| arm | runs | planning routed | callee routed | wrong accepts | callee routed to a planning command |",
              "|---|---|---|---|---|---|"]
    for arm in ARMS:
        if arm not in report["pooled"]:
            lines.append(f"| {arm} | - | incomplete | - | - | - |")
            continue
        block = report["pooled"][arm]
        diag = report["diagnostics"][arm]["pooled"]
        lines.append(f"| {arm} | {block['runs_pooled']} | {_rate(block['planning_routed'])} | "
                     f"{_rate(block['callee_routed'])} | {_rate(block['wrong_accepts'])} | "
                     f"{diag['callee_routed_to_a_planning_command']}/{diag['callee_n']} (pooled counts) |")
    lines += ["", "## Per seed (heldout cases)", "",
              "| arm | seed | planning routed | callee routed | wrong accepts | training rows | train seconds |",
              "|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        for seed, block in report["per_seed"].get(arm, {}).items():
            lines.append(f"| {arm} | {seed} | {_rate(block['planning_routed'])} | {_rate(block['callee_routed'])} | "
                         f"{_rate(block['wrong_accepts'])} | {block['training_rows_total']} | "
                         f"{block['train_seconds'] if block['train_seconds'] is not None else '-'} |")
    lines += ["", "## Paired difference against the baseline, pooled (arm - baseline)", "",
              "| arm | planning routed | callee routed | wrong accepts |", "|---|---|---|---|"]
    for arm, comparison in report["comparison_pooled"].items():
        m = comparison["metrics"]
        lines.append(f"| {arm} | {_diff(m['planning_routed'])} | {_diff(m['callee_routed'])} | "
                     f"{_diff(m['wrong_accepts'])} |")
    lines += ["", "Pooled intervals and McNemar p-values count each case three times (see the method note); "
              "read the per-seed block below for the honest spread.", ""]
    lines += ["## Paired difference per seed (arm - baseline, same seed)", "",
              "| arm | seed | planning routed | callee routed | wrong accepts |", "|---|---|---|---|---|"]
    for arm, by_seed in report["comparison_per_seed"].items():
        for seed, comparison in by_seed.items():
            m = comparison["metrics"]
            lines.append(f"| {arm} | {seed} | {_diff(m['planning_routed'])} | {_diff(m['callee_routed'])} | "
                         f"{_diff(m['wrong_accepts'])} |")
    lines += ["", "## Guidance (the epic's recommended bars; NOT gates; pooled)", ""]
    for arm, guidance in report["guidance_pooled"].items():
        lines.append(f"- {arm}: " + ", ".join(f"{name}: {'met' if ok else 'not met'}"
                                              for name, ok in guidance["checks"].items()))
    lines += ["", "## Routing diagnostics (heldout)", ""]
    for arm, diag in report["diagnostics"].items():
        d = diag["pooled"]
        lines.append(f"- {arm} (pooled counts): planning wrong accepts {d['planning_wrong_accepts']}/{d['planning_n']}, "
                     f"callee wrong accepts {d['callee_wrong_accepts']}/{d['callee_n']}, "
                     f"callee routed to a planning command {d['callee_routed_to_a_planning_command']}/{d['callee_n']}")
    lines += ["", "## Runs", "", "| run | status | train seconds | wall seconds | training rows |",
              "|---|---|---|---|---|"]
    for run in report["runs"]:
        lines.append(f"| {run['name']} | {run['status']} | {run.get('train_seconds', '-')} | "
                     f"{run.get('wall_seconds', '-')} | {run.get('training_rows_total', '-')} |")
    if "rescore" in report:
        r = report["rescore"]
        lines += ["", "## Rescore on the deduped set (stored predictions; no training)", "",
                  f"- test set: {r['test_set']}",
                  f"- deduped held-out cases: {r['deduped_heldout_cases']}; trained-on exclusions (any run): "
                  f"{r['trained_on_excluded']}; analysis cases (same for every arm and seed): {r['analysis_cases']} "
                  f"({r['planning_cases']} planning, {r['callee_cases']} callee)",
                  "- trained-on flags per run (cases the head trained on): "
                  + ", ".join(f"{k}: {v}" for k, v in r["trained_on_by_run"].items()), ""]
    if "live" in report:
        live = report["live"]
        summary = live["summary"]
        lines += ["## Live arm on the same cases (stored predictions)", "",
                  f"- arm {live['arm']}: sample {live['sample_cases']} cases, {live['sample_in_deduped']} in the "
                  f"deduped set, {live['same_as_offline_analysis']} on the offline analysis set",
                  f"- planning routed {_rate(summary['planning_routed'])}; callee routed {_rate(summary['callee_routed'])}; "
                  f"wrong accepts {_rate(summary['wrong_accepts'])}",
                  f"- callee requests routed to a planning command: {live['routing']['callee_routed_to_a_planning_command']}"
                  f"/{live['routing']['callee_n']}", ""]
        for seed, comparison in live["vs_baseline"].items():
            m = comparison["metrics"]
            lines.append(f"- vs baseline seed {seed} (same cases): planning {_diff(m['planning_routed'])}; "
                         f"callee {_diff(m['callee_routed'])}; wrong accepts {_diff(m['wrong_accepts'])}")
        lines.append("")
    lines += ["", "## Method", "", report["method"]["pooling"] + ".", "", report["method"]["caveat"] + ".",
              "Intervals: " + report["method"]["intervals"] + ".", ""]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- worker (one training, one subprocess)

def worker(args: argparse.Namespace) -> int:
    """Copy, block, train, check for fallbacks, score on the test set, record, optionally prune."""
    import fastworkflow
    from fastworkflow.model_pipeline_training import (
        experiment_head_scope,
        experiment_planning_heads,
        experiment_reserved_class_arm,
    )
    from fastworkflow.train import determinism
    from fastworkflow.train import not_here_pilot as pilot
    from fastworkflow.train.__main__ import train_workflow

    scratch = os.path.abspath(args.scratch)
    workflow_copy = os.path.join(scratch, WORKFLOW_DIRNAME)
    os.makedirs(scratch, exist_ok=True)
    pilot.copy_workflow(args.source, workflow_copy)
    pilot.copy_caches(args.cache_from, workflow_copy)
    passwords = args.passwords_file or pilot.default_passwords(args.cache_from)
    env = pilot.read_env(workflow_copy, args.env_file, passwords)
    fastworkflow.init(env_vars=pilot.blocked_env(env))

    planning_heads = spm.REQUIRED_PLANNING_HEADS[args.arm]
    determinism.set_train_seed(args.train_seed)
    started = time.monotonic()
    with experiment_reserved_class_arm(SPARSE_ARM), experiment_head_scope(None), \
            experiment_planning_heads(planning_heads):
        train_workflow(workflow_copy)
    train_seconds = round(time.monotonic() - started, 2)

    fallen = pilot.fallen_back_commands(workflow_copy)
    if fallen:
        raise StepFailure(f"LLM calls were attempted (fell back to seeds): {', '.join(fallen)}")

    test_set = pts.load_test_set(args.test_set)
    results = spm.predict_arm(workflow_copy, test_set, args.arm, train_seconds=train_seconds)
    _dump(os.path.join(scratch, RESULTS_FILENAME), results)
    timings = {"arm": args.arm, "train_seed": args.train_seed, "planning_heads": planning_heads,
               "head_scope": "unset (today's layout, flattened)", "train_seconds": train_seconds,
               "training_rows_total": sum(results["training_rows"].values())}
    _dump(os.path.join(scratch, TIMINGS_FILENAME), timings)
    if args.prune:
        prune_models(scratch)
    summary = spm.summarize(results)
    _dump(os.path.join(scratch, SUMMARY_FILENAME), summary)
    print(json.dumps(timings, indent=2))
    return 0


# ---------------------------------------------------------------- driver

def _read_json(path: str) -> Optional[dict]:
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _dump(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False, default=str)
        handle.write("\n")


def run_one(spec: dict, args: argparse.Namespace, log_dir: str) -> dict:
    """One training: skipped (resumed) when finished and --resume, else run in a fresh subprocess."""
    base = {"name": spec["name"], "arm": spec["arm"], "seed": spec["seed"], "directory": spec["directory"]}
    if args.resume and is_done(spec["directory"]):
        timings = _read_json(os.path.join(spec["directory"], TIMINGS_FILENAME)) or {}
        return {**base, "status": "resumed", "train_seconds": timings.get("train_seconds"),
                "training_rows_total": timings.get("training_rows_total")}
    if os.path.exists(spec["directory"]):
        shutil.rmtree(spec["directory"])  # a partial run: the worker copies into a fresh directory
    log_path = os.path.join(log_dir, f"{spec['name']}.log")
    started = time.monotonic()
    with open(log_path, "w", encoding="utf-8") as log:
        completed = subprocess.run(worker_command(spec, args), cwd=REPO_ROOT,
                                   stdout=log, stderr=subprocess.STDOUT)
    wall = round(time.monotonic() - started, 2)
    timings = _read_json(os.path.join(spec["directory"], TIMINGS_FILENAME)) or {}
    ok = completed.returncode == 0 and is_done(spec["directory"])
    record = {**base, "status": "ok" if ok else "failed", "returncode": completed.returncode,
              "wall_seconds": wall, "train_seconds": timings.get("train_seconds"),
              "training_rows_total": timings.get("training_rows_total"), "log": log_path}
    if not ok:
        with open(log_path, encoding="utf-8", errors="replace") as log:
            record["failure_tail"] = "".join(log.readlines()[-25:])
    return record


def _collect(runs: Sequence[dict]) -> dict[tuple[str, int], dict]:
    collected = {}
    for run in runs:
        if run["status"] in ("ok", "resumed"):
            results = _read_json(os.path.join(run["directory"], RESULTS_FILENAME))
            if results is not None:
                collected[(run["arm"], run["seed"])] = results
    return collected


def _dry_run(specs: Sequence[dict], args: argparse.Namespace) -> None:
    print("Planned trainings (dry run; nothing is created, copied, trained or run):")
    for index, spec in enumerate(specs, 1):
        skip = " [would be SKIPPED by --resume: summary.json exists]" \
            if args.resume and is_done(spec["directory"]) else ""
        print(f"  {index:2d}. {spec['name']}: {spec['directory']}{skip}")
        print("      " + " ".join(worker_command(spec, args)))
    print(f"  working directory for each worker: {REPO_ROOT}")
    print(f"  prune trained models after each finished run: {args.prune}")
    print(f"  test set: {args.test_set}")
    print(f"  baseline: {BASELINE}; compared arm: {spm.ARM_HEADS_LABELLED}; seeds {list(TRAIN_SEEDS)}")
    print(f"Report files: {os.path.join(args.root, REPORT_JSON)} and {os.path.join(args.root, REPORT_MD)}")


def run_driver(args: argparse.Namespace) -> int:
    root = os.path.abspath(args.root)
    arms = [a for a in args.arms.split(",") if a]
    seeds = [int(s) for s in args.seeds.split(",") if s]
    specs = plan_runs(root, arms, seeds)
    if args.dry_run:
        _dry_run(specs, args)
        return 0

    log_dir = os.path.join(root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    runs: list[dict] = []
    for spec in specs:  # sequential: never two trainings at once
        record = run_one(spec, args, log_dir)
        runs.append(record)
        _dump(os.path.join(root, RUNS_FILENAME), {"runs": runs})
        print(f"{record['name']}: {record['status']}", flush=True)

    planning_names = (pts.load_test_set(args.test_set) or {}).get("planning_commands", [])
    report = build_report(runs, _collect(runs), planning_names)
    _dump(os.path.join(root, REPORT_JSON), report)
    with open(os.path.join(root, REPORT_MD), "w", encoding="utf-8") as handle:
        handle.write(format_markdown(report))
    print(f"report: {os.path.join(root, REPORT_MD)}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="planning_step1", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="the trainings (sequential), then the pooled report")
    run.add_argument("--root", required=True, help="directory for the run subdirectories, logs and the report")
    run.add_argument("--source", required=True, help="the workflow to copy (read only); no default")
    run.add_argument("--cache-from", required=True,
                     help="a workflow checkout whose ___command_info holds the filled caches (read only)")
    run.add_argument("--env-file", required=True, help="the env file the caches were generated with")
    run.add_argument("--passwords-file", default=None)
    run.add_argument("--test-set", default=DEFAULT_TEST_SET, help="the named planning test set (JSON)")
    run.add_argument("--arms", default=",".join(ARMS))
    run.add_argument("--seeds", default=",".join(str(s) for s in TRAIN_SEEDS))
    run.add_argument("--prune", action="store_true", help="remove each finished copy's trained models")
    run.add_argument("--resume", action="store_true", help="skip runs whose summary.json already exists")
    run.add_argument("--dry-run", action="store_true", help="print the plan, then exit")

    resc = sub.add_parser("rescore", help="rescore stored predictions on a test set (no training, no prediction)")
    resc.add_argument("--root", required=True)
    resc.add_argument("--test-set", required=True)
    resc.add_argument("--live", default=None, help="a live arm's results JSON to score on the same cases")
    resc.add_argument("--name", default="planning_step1_rescore", help="file stem for the report")

    work = sub.add_parser("worker", help="one training (internal: started by run)")
    work.add_argument("--arm", required=True, choices=spm.OFFLINE_ARMS)
    work.add_argument("--train-seed", type=int, required=True)
    work.add_argument("--scratch", required=True)
    work.add_argument("--source", required=True)
    work.add_argument("--cache-from", required=True)
    work.add_argument("--env-file", required=True)
    work.add_argument("--passwords-file", default=None)
    work.add_argument("--test-set", required=True)
    work.add_argument("--prune", action="store_true")

    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "run":
        return run_driver(args)
    if args.command == "rescore":
        report = rescore(os.path.abspath(args.root), os.path.abspath(args.test_set),
                         os.path.abspath(args.live) if args.live else None)
        _dump(os.path.join(args.root, f"{args.name}.json"), report)
        with open(os.path.join(args.root, f"{args.name}.md"), "w", encoding="utf-8") as handle:
            handle.write(format_markdown(report))
        print(json.dumps(report["rescore"], indent=2))
        return 0
    try:
        return worker(args)
    except (StepFailure, PilotFailure) as failure:
        print(f"planning_step1 worker failed: {failure}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

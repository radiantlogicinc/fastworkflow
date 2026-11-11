"""Phase 9 step 3 (fix-3rsq.9.5): the flattened-vs-scoped lookup, six trainings and their scoring.

Both head layouts run under the step 1 winner (the reserved-class arm ``sparse_1``), on a copy of
the measured workflow, with train seeds 1, 2, 3: six trainings, sequential, one subprocess at a
time. Each run ``ROOT/<layout>_seed<N>`` does four things:

    1  train  not_here_pilot --arm sparse_1 --head-scope <layout> --train-seed N (log: ROOT/logs/<name>.log)
    2  probe  score_lookup probe --workflow <dir>/workflow --out <dir>/lookup_probe.json
              [--train-timings <dir>/timings.json]
    3  score  score_lookup score --workflow <dir>/workflow --probe <dir>/lookup_probe.json
              --layout <layout> --out <dir>/lookup_results.json
    4  prune  (--prune) remove the trained models with not_here_step1.prune_models

A failed step is recorded (``failed_step``) and the runner stops without starting the next run.
``--resume`` skips a run whose ``lookup_results.json`` exists. ``runs.json`` is rewritten after
every run. When every layout has all its seeds, score_lookup ``compare`` pools each layout's seeds
(flattened = base, scoped = arm, in seed order) into ``step3_report.json`` and ``step3_report.md``
and the verdict line is printed; otherwise the runner prints INCOMPLETE and writes no report.

``--dry-run`` prints the planned commands, the bar constants and the report paths. It creates nothing.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Optional, Sequence

from fastworkflow.train import score_lookup as lookup
from fastworkflow.train.not_here_pilot import CACHE_FROM_ENV, SOURCE_ENV, WORKFLOW_DIRNAME, required_path
from fastworkflow.train.not_here_step1 import _read_json, _write, prune_models

ARM = "sparse_1"
LAYOUTS: tuple[str, ...] = lookup.LAYOUTS  # ("flattened", "scoped"): flattened first
TRAIN_SEEDS: tuple[int, ...] = (1, 2, 3)
TIMINGS_FILENAME = "timings.json"  # written by the pilot
PILOT_DONE_FILENAME = "summary.json"  # written by the pilot after scoring
PROBE_FILENAME = "lookup_probe.json"
RESULTS_FILENAME = "lookup_results.json"  # the run's done marker for --resume
RUNS_FILENAME = "runs.json"
REPORT_JSON_FILENAME = "step3_report.json"
REPORT_MD_FILENAME = "step3_report.md"
MODEL_ENTRIES = ("versions", "current")  # what prune_models removes under ___command_info


@dataclass(frozen=True)
class LookupRun:
    layout: str
    seed: int
    directory: str

    @property
    def name(self) -> str:
        return f"{self.layout}_seed{self.seed}"

    @property
    def workflow(self) -> str:
        return os.path.join(self.directory, WORKFLOW_DIRNAME)

    @property
    def timings_path(self) -> str:
        return os.path.join(self.directory, TIMINGS_FILENAME)

    @property
    def probe_path(self) -> str:
        return os.path.join(self.directory, PROBE_FILENAME)

    @property
    def results_path(self) -> str:
        return os.path.join(self.directory, RESULTS_FILENAME)


# ---------------------------------------------------------------- plan

def plan_runs(root: str, layouts: Sequence[str] = LAYOUTS, seeds: Sequence[int] = TRAIN_SEEDS) -> list[LookupRun]:
    """The runs in order: layout by layout (as given), seeds in order within each layout."""
    return [LookupRun(layout=layout, seed=int(seed), directory=os.path.join(root, f"{layout}_seed{seed}"))
            for layout in layouts for seed in seeds]


def is_done(run: LookupRun) -> bool:
    return os.path.isfile(run.results_path)


def models_present(directory: str) -> bool:
    info = os.path.join(directory, WORKFLOW_DIRNAME, "___command_info")
    return any(os.path.lexists(os.path.join(info, name)) for name in MODEL_ENTRIES)


def pilot_command(run: LookupRun, *, source: str, cache_from: str, env_file: str,
                  passwords_file: Optional[str]) -> list[str]:
    command = [sys.executable, "-m", "fastworkflow.train.not_here_pilot",
               "--arm", ARM, "--head-scope", run.layout, "--train-seed", str(run.seed),
               "--scratch", run.directory, "--source", source,
               "--cache-from", cache_from, "--env-file", env_file]
    if passwords_file:
        command += ["--passwords-file", passwords_file]
    return command


def probe_command(run: LookupRun, *, with_timings: bool = True) -> list[str]:
    command = [sys.executable, "-m", "fastworkflow.train.score_lookup", "probe",
               "--workflow", run.workflow, "--out", run.probe_path]
    if with_timings:
        command += ["--train-timings", run.timings_path]
    return command


def score_command(run: LookupRun, *, root_context: Optional[str] = None) -> list[str]:
    command = [sys.executable, "-m", "fastworkflow.train.score_lookup", "score",
               "--workflow", run.workflow, "--probe", run.probe_path, "--layout", run.layout,
               "--out", run.results_path]
    if root_context:
        command += ["--root-context", root_context]
    return command


def compare_command(root: str, runs: Sequence[LookupRun]) -> list[str]:
    """score_lookup compare: base = flattened runs, arm = scoped runs, both in seed order."""
    by_layout = runs_by_layout(runs)
    base = [run.results_path for run in by_layout[lookup.BASE_LAYOUT]]
    arm = [run.results_path for run in by_layout[lookup.ARM_LAYOUT]]
    return [sys.executable, "-m", "fastworkflow.train.score_lookup", "compare",
            "--base", ",".join(base), "--arm", ",".join(arm),
            "--out", os.path.join(root, REPORT_JSON_FILENAME), "--md", os.path.join(root, REPORT_MD_FILENAME)]


def runs_by_layout(runs: Sequence[LookupRun]) -> dict[str, list[LookupRun]]:
    grouped: dict[str, list[LookupRun]] = {}
    for run in sorted(runs, key=lambda r: r.seed):
        grouped.setdefault(run.layout, []).append(run)
    return grouped


# ---------------------------------------------------------------- run

def _run_step(command: list[str], *, cwd: str, log_path: str, label: str) -> tuple[int, float]:
    started = time.monotonic()
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n=== {label}: {' '.join(command)}\n")
        log.flush()
        completed = subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
    return completed.returncode, round(time.monotonic() - started, 2)


def _run_one(run: LookupRun, args: argparse.Namespace, repo_root: str, log_dir: str) -> dict:
    base = {"name": run.name, "arm": ARM, "layout": run.layout, "seed": run.seed, "directory": run.directory}
    if args.resume and is_done(run):
        timings = _read_json(run.timings_path) or {}
        return {**base, "status": "resumed", "returncode": None, "failed_step": None,
                "train_seconds": timings.get("train_seconds"), "wall_seconds": None,
                "pruned": not models_present(run.directory)}
    if os.path.exists(run.directory):
        shutil.rmtree(run.directory)  # a partial run: the pilot copies into a fresh directory
    log_path = os.path.join(log_dir, f"{run.name}.log")
    with open(log_path, "w", encoding="utf-8"):
        pass
    started = time.monotonic()
    command = pilot_command(run, source=args.source, cache_from=args.cache_from,
                            env_file=args.env_file, passwords_file=args.passwords_file)
    code, _ = _run_step(command, cwd=repo_root, log_path=log_path, label="train")
    timings = _read_json(run.timings_path) or {}
    failed_step = None
    if code != 0 or not os.path.isfile(os.path.join(run.directory, PILOT_DONE_FILENAME)):
        failed_step, code = "train", code or 1
    if failed_step is None:
        code, _ = _run_step(probe_command(run, with_timings=os.path.isfile(run.timings_path)),
                            cwd=repo_root, log_path=log_path, label="probe")
        failed_step = "probe" if code != 0 else None
    if failed_step is None:
        code, _ = _run_step(score_command(run, root_context=args.root_context),
                            cwd=repo_root, log_path=log_path, label="score")
        failed_step = "score" if code != 0 else None
    record = {**base, "status": "failed" if failed_step else "ok", "returncode": code,
              "failed_step": failed_step, "train_seconds": timings.get("train_seconds"),
              "wall_seconds": round(time.monotonic() - started, 2), "pruned": False, "log": log_path}
    if failed_step is None and args.prune:
        prune_models(run.directory)
        record["pruned"] = True
    return record


def _dry_run(runs: Sequence[LookupRun], args: argparse.Namespace, root: str) -> None:
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print("Planned lookup runs (dry run; nothing is created, copied, trained or run):")
    for index, run in enumerate(runs, 1):
        skip = " [would be SKIPPED by --resume: lookup_results.json exists]" if args.resume and is_done(run) else ""
        print(f"  {index:2d}. {run.name} (arm {ARM}, layout {run.layout}, train seed {run.seed}): {run.directory}{skip}")
        print("      train: " + " ".join(pilot_command(
            run, source=args.source, cache_from=args.cache_from, env_file=args.env_file,
            passwords_file=args.passwords_file)))
        print("      probe: " + " ".join(probe_command(run)))
        print("      score: " + " ".join(score_command(run, root_context=args.root_context)))
        print(f"      prune trained models after scoring: {args.prune}")
    if all(layout in {run.layout for run in runs} for layout in LAYOUTS):
        print("  compare (only when every layout has all its seeds):")
        print("      " + " ".join(compare_command(root, runs)))
    print(f"  working directory for each subprocess: {repo}")
    print("Bars (score_lookup; scoped minus flattened, paired 95%):")
    print(f"  E2E_MIN_DIFFERENCE = {lookup.E2E_MIN_DIFFERENCE}  (end-to-end: lower bound of the difference >= this)")
    print(f"  WRONG_ACCEPT_MAX_RISE = {lookup.WRONG_ACCEPT_MAX_RISE}  (wrong accepts: upper bound <= this)")
    print(f"  TOP1_MIN_DIFFERENCE = {lookup.TOP1_MIN_DIFFERENCE}  (in-scope top-1: lower bound >= this)")
    print(f"Report files: {os.path.join(root, REPORT_JSON_FILENAME)} and {os.path.join(root, REPORT_MD_FILENAME)}")
    print(f"Run ledger: {os.path.join(root, RUNS_FILENAME)}")


def _verdict_line(report: dict) -> str:
    verdict = report["verdict"]
    if verdict["passes"]:
        return "Verdict: scoped PASSES the lookup bars against flattened (step 3)"
    failed = [metric for metric, bar in verdict["bars"].items() if not bar["passes"]]
    return f"Verdict: scoped FAILS the lookup bars against flattened (step 3): {', '.join(failed)}"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="lookup_step3", description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", required=True, help="directory holding the run subdirectories, logs and the report")
    parser.add_argument("--cache-from", default=None,
                        help="a workflow checkout whose ___command_info holds the filled caches (read only). "
                             f"Required unless ${CACHE_FROM_ENV} is set")
    parser.add_argument("--env-file", required=True, help="the env file the caches were generated with")
    parser.add_argument("--passwords-file", default=None)
    parser.add_argument("--source", default=None,
                        help=f"the worktree workflow (read only). Required unless ${SOURCE_ENV} is set")
    parser.add_argument("--layouts", default=",".join(LAYOUTS), help="layouts in run order (default flattened,scoped)")
    parser.add_argument("--seeds", default=",".join(str(s) for s in TRAIN_SEEDS))
    parser.add_argument("--root-context", default=None,
                        help="name of the root context, passed to score; default: none (the climb asks '*' after a top)")
    parser.add_argument("--prune", action="store_true", help="remove each scored run's trained models")
    parser.add_argument("--resume", action="store_true", help="skip runs whose lookup_results.json already exists")
    parser.add_argument("--dry-run", action="store_true", help="print the plan, the bars and the report paths, then exit")
    args = parser.parse_args(argv)

    try:
        args.source = required_path(args.source, SOURCE_ENV, "--source")
        args.cache_from = required_path(args.cache_from, CACHE_FROM_ENV, "--cache-from")
    except ValueError as error:
        parser.error(str(error))
    layouts = [item for item in args.layouts.split(",") if item]
    bad = [layout for layout in layouts if layout not in LAYOUTS]
    if bad or not layouts:
        parser.error(f"--layouts takes {','.join(LAYOUTS)}, not {bad or 'nothing'}")
    seeds = [int(item) for item in args.seeds.split(",") if item]
    if not seeds:
        parser.error("--seeds needs at least one seed")
    root = os.path.abspath(args.root)
    runs = plan_runs(root, layouts, seeds)
    if args.dry_run:
        _dry_run(runs, args, root)
        return 0

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    log_dir = os.path.join(root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    ledger: list[dict] = []
    failure: Optional[dict] = None
    for run in runs:  # sequential: never two trainings at once
        record = _run_one(run, args, repo_root, log_dir)
        ledger.append(record)
        _write(os.path.join(root, RUNS_FILENAME), {"runs": ledger})
        print(f"{record['name']}: {record['status']}"
              + (f" (failed at {record['failed_step']}; see {record['log']})" if record["status"] == "failed" else ""),
              flush=True)
        if record["status"] == "failed":
            failure = record
            break

    missing = [run.name for run in runs if not is_done(run)]
    complete = (not missing and set(layouts) == set(LAYOUTS))
    if not complete:
        reason = (f"run {failure['name']} failed at {failure['failed_step']}; the runner stopped"
                  if failure else f"missing {', '.join(missing) or 'a layout'}")
        print(f"INCOMPLETE: {reason}")
        return 1 if failure else 0

    compare_log = os.path.join(log_dir, "compare.log")
    code, _ = _run_step(compare_command(root, runs), cwd=repo_root, log_path=compare_log, label="compare")
    if code != 0:
        print(f"INCOMPLETE: score_lookup compare failed (see {compare_log})")
        return 1
    report = _read_json(os.path.join(root, REPORT_JSON_FILENAME))
    print(_verdict_line(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())

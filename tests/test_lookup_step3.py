"""Phase 9 step 3 runner (fastworkflow.train.lookup_step3): the plan, the commands, resume, the
ledger, the compare inputs, INCOMPLETE and the dry run. No training and no mocks.

Score files are hand-built with the scorer's own summary function; the compare step is the real
score_lookup CLI in a subprocess. The one failing run uses a source path that does not exist, so
the pilot fails at its first copy, before any model is loaded.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from fastworkflow.train import lookup_step3 as step3
from fastworkflow.train import score_lookup as lookup

SEEDS = (1, 2, 3)


def _case(index: int, layout: str, in_scope: bool, correct: bool, wrong_accept: bool,
          wrong_decline: bool) -> dict:
    return {"index": index, "utterance": f"utterance {index}", "label": "Account/show_owner",
            "owner": "Account", "position": "Account", "depth": 1, "in_scope": in_scope,
            "expected_binding": "Account" if in_scope else None, "owner_kind": "own",
            "layout": layout, "named": None, "bound": None, "asked": [], "outcome": "x",
            "correct": correct, "top1": correct, "wrong_accept": wrong_accept,
            "wrong_decline": wrong_decline,
            "secondary": {"outcome": "correct_confident" if correct else "declined", "kind": "none",
                          "named": None, "bound": None, "candidates": [], "asked": []}}


def _write_score(path: Path, layout: str, seed: int, *, shifted: bool) -> None:
    """A score file with six cases, the scoped one fixing a decline and the other keeping it."""
    cases = [
        _case(0, layout, True, correct=not (shifted and layout == "flattened"), wrong_accept=False,
              wrong_decline=layout == "flattened" and not shifted),
        _case(1, layout, True, correct=True, wrong_accept=False, wrong_decline=False),
        _case(2, layout, True, correct=True, wrong_accept=False, wrong_decline=False),
        _case(3, layout, False, correct=False, wrong_accept=False, wrong_decline=False),
        _case(4, layout, False, correct=False, wrong_accept=layout == "flattened" and shifted,
              wrong_decline=False),
        _case(5, layout, False, correct=False, wrong_accept=False, wrong_decline=False),
    ]
    document = {"schema": 1, "layout": layout, "arm": step3.ARM, "seed": seed, "workflow_version": "v",
                "heldout_records": 6, "skipped_labels": [], "positions": 1,
                "training_rows_per_head": {"root": 10}, "training_seconds_per_head": None,
                "summary": lookup.summarize_cases(cases), "cases": cases}
    path.write_text(json.dumps(document), encoding="utf-8")


def _args(root: Path, **overrides) -> object:
    values = {"source": "/nonexistent/source", "cache_from": "/nonexistent/cache",
              "env_file": "/nonexistent/env", "passwords_file": None, "root_context": None,
              "prune": False, "resume": True, "dry_run": False}
    values.update(overrides)
    return type("Args", (), values)()


def _tree(path: Path) -> list[str]:
    return sorted(str(p.relative_to(path)) for p in path.rglob("*")) if path.exists() else []


# ---------------------------------------------------------------- plan and commands

def test_the_plan_runs_flattened_first_then_scoped_with_seeds_in_order(tmp_path):
    runs = step3.plan_runs(str(tmp_path))
    assert [run.name for run in runs] == [
        "flattened_seed1", "flattened_seed2", "flattened_seed3",
        "scoped_seed1", "scoped_seed2", "scoped_seed3"]
    assert [run.directory for run in runs][:2] == [str(tmp_path / "flattened_seed1"), str(tmp_path / "flattened_seed2")]
    assert all(run.workflow == os.path.join(run.directory, "workflow") for run in runs)


def test_the_pilot_command_carries_the_sparse_arm_the_head_scope_and_the_train_seed(tmp_path):
    run = step3.plan_runs(str(tmp_path))[4]  # scoped_seed2
    command = step3.pilot_command(run, source="/src", cache_from="/cache", env_file="/env.env",
                                  passwords_file=None)
    assert command[command.index("--arm") + 1] == "sparse_1"
    assert command[command.index("--head-scope") + 1] == "scoped"
    assert command[command.index("--train-seed") + 1] == "2"
    assert command[command.index("--scratch") + 1] == run.directory
    assert "--passwords-file" not in command
    with_passwords = step3.pilot_command(run, source="/src", cache_from="/cache", env_file="/env.env",
                                         passwords_file="/pw.env")
    assert with_passwords[with_passwords.index("--passwords-file") + 1] == "/pw.env"


def test_probe_and_score_commands_point_at_the_copy_and_the_layout(tmp_path):
    run = step3.plan_runs(str(tmp_path))[0]
    probe = step3.probe_command(run)
    assert probe[probe.index("--workflow") + 1] == run.workflow
    assert probe[probe.index("--train-timings") + 1] == run.timings_path
    assert "--train-timings" not in step3.probe_command(run, with_timings=False)
    score = step3.score_command(run)
    assert score[score.index("--layout") + 1] == "flattened"
    assert score[score.index("--out") + 1] == run.results_path
    assert "--root-context" not in score
    assert step3.score_command(run, root_context="Root")[-1] == "Root"


def test_compare_takes_flattened_as_base_and_scoped_as_arm_in_seed_order(tmp_path):
    runs = list(reversed(step3.plan_runs(str(tmp_path))))  # any input order
    command = step3.compare_command(str(tmp_path), runs)
    base = command[command.index("--base") + 1].split(",")
    arm = command[command.index("--arm") + 1].split(",")
    assert base == [str(tmp_path / f"flattened_seed{s}" / "lookup_results.json") for s in SEEDS]
    assert arm == [str(tmp_path / f"scoped_seed{s}" / "lookup_results.json") for s in SEEDS]
    assert command[command.index("--out") + 1] == str(tmp_path / "step3_report.json")
    assert command[command.index("--md") + 1] == str(tmp_path / "step3_report.md")


# ---------------------------------------------------------------- resume and the ledger

def test_resume_skips_a_run_whose_lookup_results_exist_without_training_it(tmp_path):
    root = tmp_path / "step3"
    run = step3.plan_runs(str(root))[0]
    Path(run.directory, "workflow", "___command_info", "versions").mkdir(parents=True)
    Path(run.directory, "timings.json").write_text(json.dumps({"train_seconds": 12.5}))
    Path(run.results_path).write_text("{}")
    log_dir = root / "logs"
    log_dir.mkdir(parents=True)

    record = step3._run_one(run, _args(root), str(tmp_path), str(log_dir))

    assert record["status"] == "resumed"
    assert record["train_seconds"] == 12.5
    assert record["pruned"] is False                       # the models are still on disk
    assert Path(run.directory, "workflow", "___command_info", "versions").is_dir()  # not removed
    assert not (log_dir / f"{run.name}.log").exists()     # no subprocess was started


def test_a_failed_run_is_recorded_with_its_step_and_the_runner_stops(tmp_path, capsys):
    root = tmp_path / "step3"
    code = step3.main(["--root", str(root), "--cache-from", str(tmp_path / "cache"),
                       "--env-file", str(tmp_path / "env"), "--source", str(tmp_path / "missing"),
                       "--layouts", "flattened", "--seeds", "1,2"])

    out = capsys.readouterr().out
    ledger = json.loads((root / "runs.json").read_text())["runs"]
    assert code == 1
    assert [r["name"] for r in ledger] == ["flattened_seed1"]     # seed 2 was never started
    assert ledger[0]["status"] == "failed" and ledger[0]["failed_step"] == "train"
    assert ledger[0]["returncode"] != 0 and ledger[0]["pruned"] is False
    assert "INCOMPLETE" in out and "flattened_seed1" in out
    assert not (root / "step3_report.json").exists()


def test_the_ledger_records_each_run_with_its_fields(tmp_path):
    """A resumed run with models still on disk, written as runs.json; the prune path itself is
    exercised by not_here_step1's tests (prune_models) and by a real run."""
    root = tmp_path / "step3"
    for run in step3.plan_runs(str(root)):
        run_dir = Path(run.directory)
        (run_dir / "workflow" / "___command_info").mkdir(parents=True)
        (run_dir / "workflow" / "___command_info" / "current").write_text("v")  # a model, to be pruned
        (run_dir / "timings.json").write_text(json.dumps({"train_seconds": float(run.seed)}))
        Path(run.results_path).write_text("{}")
    log_dir = root / "logs"
    log_dir.mkdir(parents=True)
    run = step3.plan_runs(str(root))[0]
    record = step3._run_one(run, _args(root, prune=True), str(tmp_path), str(log_dir))
    assert record["status"] == "resumed"
    assert record["pruned"] is False                       # resumed runs report what is on disk now

    # the same ledger shape written by the runner
    step3._write(str(root / "runs.json"), {"runs": [record]})
    written = json.loads((root / "runs.json").read_text())["runs"][0]
    for key in ("name", "arm", "layout", "seed", "status", "returncode", "failed_step",
                "train_seconds", "wall_seconds", "pruned", "directory"):
        assert key in written
    assert written["arm"] == "sparse_1" and written["layout"] == "flattened" and written["seed"] == 1


# ---------------------------------------------------------------- compare, complete and incomplete

def test_a_complete_set_of_six_runs_is_compared_into_the_report_without_training(tmp_path, capsys):
    root = tmp_path / "step3"
    for run in step3.plan_runs(str(root)):
        Path(run.directory).mkdir(parents=True)
        _write_score(Path(run.results_path), run.layout, run.seed, shifted=run.layout == "flattened")

    code = step3.main(["--root", str(root), "--source", "/nonexistent", "--cache-from", "/nonexistent",
                       "--env-file", "/nonexistent", "--resume"])

    out = capsys.readouterr().out
    assert code == 0
    ledger = json.loads((root / "runs.json").read_text())["runs"]
    assert len(ledger) == 6 and all(record["status"] == "resumed" for record in ledger)
    report = json.loads((root / "step3_report.json").read_text())
    assert report["pooled_cases"] == 6 * len(SEEDS)        # three seeds of six cases each
    assert [r["seed"] for r in report["runs"]["flattened"]] == list(SEEDS)
    assert [r["seed"] for r in report["runs"]["scoped"]] == list(SEEDS)
    assert (root / "step3_report.md").read_text().startswith("# Phase 9 step 3")
    assert out.count("Verdict: scoped") == 1
    assert "INCOMPLETE" not in out


def test_the_runner_prints_incomplete_and_writes_no_report_while_runs_are_missing(tmp_path, capsys):
    root = tmp_path / "step3"
    done = step3.plan_runs(str(root), layouts=("flattened",), seeds=(1,))[0]
    Path(done.directory).mkdir(parents=True)
    _write_score(Path(done.results_path), "flattened", 1, shifted=False)

    code = step3.main(["--root", str(root), "--cache-from", str(tmp_path / "c"), "--env-file", str(tmp_path / "e"),
                       "--layouts", "flattened", "--seeds", "1,2", "--source", str(tmp_path / "missing"),
                       "--resume"])

    out = capsys.readouterr().out
    assert code == 1
    assert "INCOMPLETE" in out
    assert not (root / "step3_report.json").exists()
    ledger = json.loads((root / "runs.json").read_text())["runs"]
    assert [(r["name"], r["status"]) for r in ledger] == [("flattened_seed1", "resumed"),
                                                          ("flattened_seed2", "failed")]


# ---------------------------------------------------------------- dry run

def test_a_dry_run_prints_the_plan_the_bars_and_the_report_paths_and_creates_nothing(tmp_path, capsys):
    root = tmp_path / "step3"
    finished = step3.plan_runs(str(root))[0]
    Path(finished.directory).mkdir(parents=True)
    Path(finished.results_path).write_text("{}")
    before = _tree(tmp_path)

    code = step3.main(["--root", str(root), "--source", str(tmp_path / "source"),
                       "--cache-from", str(tmp_path / "cache"),
                       "--env-file", str(tmp_path / "env"), "--prune", "--resume", "--dry-run"])

    out = capsys.readouterr().out
    assert code == 0
    assert _tree(tmp_path) == before                       # nothing created, copied or trained
    assert out.count("--train-seed") == 6                 # one pilot line per run
    assert out.count("--arm sparse_1") == 6
    assert out.count("--head-scope flattened") == 3 and out.count("--head-scope scoped") == 3
    assert "SKIPPED" in out and "flattened_seed1" in out
    assert "score_lookup compare" in out
    assert f"E2E_MIN_DIFFERENCE = {lookup.E2E_MIN_DIFFERENCE}" in out
    assert f"WRONG_ACCEPT_MAX_RISE = {lookup.WRONG_ACCEPT_MAX_RISE}" in out
    assert f"TOP1_MIN_DIFFERENCE = {lookup.TOP1_MIN_DIFFERENCE}" in out
    assert str(root / "step3_report.json") in out and str(root / "step3_report.md") in out

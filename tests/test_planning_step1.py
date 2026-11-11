"""Phase 10 step 1c driver (``fastworkflow.train.planning_step1``): the plan, resume, prune, pooling and report.

Integration tests with no mocks and no training: the dry run is printed and nothing is created, a finished
run is skipped by ``--resume`` without a subprocess, prune removes trained models and keeps the caches, and
the pooled report is computed from real result files written into a temporary directory and from hand-built
case lists (the same style as ``test_planning_match_tooling``).
"""

from __future__ import annotations

import argparse
import json
import os

import pytest

from fastworkflow.train import planning_step1 as step1

LIBRARY = os.path.join("fastworkflow", "examples", "library")
LIBRARY_ENV = os.path.join(LIBRARY, "fastworkflow.env")


def _args(**overrides) -> argparse.Namespace:
    values = {"source": LIBRARY, "cache_from": LIBRARY, "env_file": LIBRARY_ENV, "passwords_file": None,
              "test_set": step1.DEFAULT_TEST_SET, "prune": False, "resume": False, "dry_run": False}
    values.update(overrides)
    return argparse.Namespace(**values)


# ---------------------------------------------------------------- plan and dry run

def test_plan_is_two_offline_arms_three_seeds_each_baseline_first(tmp_path):
    specs = step1.plan_runs(str(tmp_path), step1.ARMS, step1.TRAIN_SEEDS)
    assert [s["name"] for s in specs] == [
        "heads_not_here_seed1", "heads_not_here_seed2", "heads_not_here_seed3",
        "heads_labelled_seed1", "heads_labelled_seed2", "heads_labelled_seed3",
    ]
    assert step1.BASELINE == "heads_not_here"


def test_each_offline_arm_enters_the_layout_its_worker_asks_for_and_records_it(tmp_path):
    """The worker enters ``experiment_planning_heads(REQUIRED_PLANNING_HEADS[arm])`` and the copy records it."""
    import shutil

    import fastworkflow
    from fastworkflow.command_routing import RoutingRegistry
    from fastworkflow.model_pipeline_training import (
        construction_records,
        experiment_planning_heads,
        experiment_reserved_class_arm,
    )
    from fastworkflow.train import score_planning_match as spm

    fastworkflow.init({})
    copy = tmp_path / "library_arms"
    shutil.copytree(LIBRARY, copy, ignore=shutil.ignore_patterns("___*", "__pycache__"))
    RoutingRegistry.clear_registry()
    for arm in spm.OFFLINE_ARMS:
        layout = spm.REQUIRED_PLANNING_HEADS[arm]
        with experiment_reserved_class_arm("sparse_1"), experiment_planning_heads(layout):
            _manifest, run_metadata = construction_records(None, None, str(copy))
        assert run_metadata["planning_heads"] == layout, arm
    assert spm.REQUIRED_PLANNING_HEADS == {spm.ARM_HEADS_LABELLED: "labelled", spm.ARM_HEADS_NOT_HERE: "not_here"}
    assert step1.BASELINE == spm.ARM_HEADS_NOT_HERE
    RoutingRegistry.clear_registry()


def test_plan_refuses_a_live_arm_and_an_unknown_arm(tmp_path):
    with pytest.raises(ValueError, match="not an offline arm"):
        step1.plan_runs(str(tmp_path), ("jev",), (1,))
    with pytest.raises(ValueError, match="not an offline arm"):
        step1.plan_runs(str(tmp_path), ("sparse_1",), (1,))


def test_dry_run_prints_six_worker_commands_and_creates_nothing(tmp_path, capsys):
    root = str(tmp_path / "run")
    code = step1.main(["run", "--root", root, "--source", LIBRARY, "--cache-from", LIBRARY,
                       "--env-file", LIBRARY_ENV, "--prune", "--dry-run"])
    out = capsys.readouterr().out
    assert code == 0
    assert not os.path.exists(root), "a dry run must not create the root"
    assert out.count(" worker --arm ") == 6
    assert "--arm heads_labelled --train-seed 3" in out
    assert "--arm heads_not_here --train-seed 1" in out
    assert "dry run" in out


def test_source_cache_from_and_env_file_are_required(tmp_path):
    with pytest.raises(SystemExit):
        step1.main(["run", "--root", str(tmp_path), "--cache-from", LIBRARY, "--env-file", LIBRARY_ENV, "--dry-run"])
    with pytest.raises(SystemExit):
        step1.main(["run", "--root", str(tmp_path), "--source", LIBRARY, "--env-file", LIBRARY_ENV, "--dry-run"])
    with pytest.raises(SystemExit):
        step1.main(["run", "--root", str(tmp_path), "--source", LIBRARY, "--cache-from", LIBRARY, "--dry-run"])


def test_dry_run_marks_a_finished_run_as_skipped_under_resume(tmp_path, capsys):
    root = tmp_path / "run"
    done = root / "heads_not_here_seed2"
    done.mkdir(parents=True)
    (done / step1.SUMMARY_FILENAME).write_text("{}")
    step1.main(["run", "--root", str(root), "--source", LIBRARY, "--cache-from", LIBRARY,
                "--env-file", LIBRARY_ENV, "--resume", "--dry-run"])
    out = capsys.readouterr().out
    assert "heads_not_here_seed2" in out and "SKIPPED by --resume" in out
    assert out.count("SKIPPED by --resume") == 1


# ---------------------------------------------------------------- resume and prune

def test_resume_returns_a_finished_run_without_starting_a_worker(tmp_path):
    spec = step1.plan_runs(str(tmp_path / "run"), ("heads_labelled",), (2,))[0]
    os.makedirs(spec["directory"])
    with open(os.path.join(spec["directory"], step1.SUMMARY_FILENAME), "w") as handle:
        handle.write("{}")
    with open(os.path.join(spec["directory"], step1.TIMINGS_FILENAME), "w") as handle:
        json.dump({"train_seconds": 12.5, "training_rows_total": 40}, handle)
    log_dir = str(tmp_path / "logs")
    os.makedirs(log_dir)
    record = step1.run_one(spec, _args(resume=True), log_dir)
    assert record["status"] == "resumed"
    assert record["train_seconds"] == 12.5 and record["training_rows_total"] == 40
    assert not os.path.exists(os.path.join(log_dir, f"{spec['name']}.log")), "no worker log: nothing ran"


def test_prune_removes_trained_models_and_keeps_the_caches(tmp_path):
    scratch = tmp_path / "scratch"
    info = scratch / "workflow" / "___command_info"
    (info / "versions" / "v1" / "Book").mkdir(parents=True)
    (info / "versions" / "v1" / "Book" / "model.bin").write_text("weights")
    os.symlink(str(info / "versions" / "v1"), str(info / "current"))
    (info / "current.json").write_text("{}")
    (info / "utterance_cache").mkdir()
    (info / "utterance_cache" / "a.json").write_text("{}")
    (info / "param_example_cache").mkdir()
    (info / "training_provenance.json").write_text("{}")

    step1.prune_models(str(scratch))

    assert not (info / "versions").exists()
    assert not os.path.lexists(info / "current")
    assert (info / "current.json").is_file(), "the pointer file stays; only the models and the link go"
    assert (info / "utterance_cache" / "a.json").is_file()
    assert (info / "param_example_cache").is_dir()
    assert (info / "training_provenance.json").is_file()


# ---------------------------------------------------------------- pooling and the report

def _case(utterance, expected, kind, top, split="heldout", head="Book"):
    return {"head": head, "utterance": utterance, "expected": expected, "kind": kind,
            "split": split, "top_label": top, "topk": [], "source": "hand-written", "related": expected}


PLANNING = "Book/loan_report"
CALLEE = "Book/find_book"


def _results(arm, tops, rows=None):
    """Four heldout cases: two planning requests and two callee requests, routed to *tops* (in that order)."""
    cases = [
        _case("who has this book", PLANNING, "planning", tops[0]),
        _case("loan report for books", PLANNING, "planning", tops[1]),
        _case("find the book called dune", CALLEE, "callee", tops[2]),
        _case("look up the dune title", CALLEE, "callee", tops[3]),
    ]
    return {"schema": 1, "arm": arm, "cases": cases, "training_rows": rows or {"Book": 10},
            "train_seconds": 100.0}


def _runs(arm, seeds, status="ok"):
    return [{"name": f"{arm}_seed{s}", "arm": arm, "seed": s, "status": status, "directory": "unused"}
            for s in seeds]


def test_pool_concatenates_in_seed_order_and_refuses_different_cases():
    a = _results("heads_labelled", [PLANNING, PLANNING, CALLEE, CALLEE])
    b = _results("heads_labelled", [PLANNING, "", CALLEE, PLANNING])
    pooled = step1.pool_results([a, b], "heads_labelled")
    assert pooled["runs_pooled"] == 2
    assert len(pooled["cases"]) == 8
    assert pooled["cases"][:4] == a["cases"]
    other = _results("heads_labelled", [PLANNING, PLANNING, CALLEE, CALLEE])
    other["cases"][0]["utterance"] = "a different request"
    with pytest.raises(ValueError, match="scores different cases"):
        step1.pool_results([a, other], "heads_labelled")


def test_report_pools_pairs_per_seed_and_flags_callee_requests_routed_to_planning():
    baseline = {s: _results("heads_not_here", ["", "", CALLEE, PLANNING]) for s in (1, 2, 3)}
    labelled = {s: _results("heads_labelled", [PLANNING, PLANNING, CALLEE, CALLEE]) for s in (1, 2, 3)}
    results = {}
    for s in (1, 2, 3):
        results[("heads_not_here", s)] = baseline[s]
        results[("heads_labelled", s)] = labelled[s]
    runs = _runs("heads_not_here", (1, 2, 3)) + _runs("heads_labelled", (1, 2, 3))
    report = step1.build_report(runs, results, [PLANNING])

    assert report["incomplete"] == {}
    assert report["pooled"]["heads_labelled"]["planning_routed"]["rate"] == 1.0
    assert report["pooled"]["heads_not_here"]["planning_routed"]["rate"] == 0.0
    diff = report["comparison_pooled"]["heads_labelled"]["metrics"]["planning_routed"]
    assert diff["difference"] == pytest.approx(1.0)
    assert set(report["comparison_per_seed"]["heads_labelled"]) == {"1", "2", "3"}
    assert report["comparison_per_seed"]["heads_labelled"]["2"]["metrics"]["planning_routed"]["difference"] \
        == pytest.approx(1.0)
    # The baseline routes its one callee request to the planning command in each seed: the diagnostic counts it.
    baseline_diag = report["diagnostics"]["heads_not_here"]
    assert baseline_diag["per_seed"]["1"]["callee_routed_to_a_planning_command"] == 1
    assert baseline_diag["pooled"]["callee_routed_to_a_planning_command"] == 3
    assert report["diagnostics"]["heads_labelled"]["pooled"]["callee_routed_to_a_planning_command"] == 0
    markdown = step1.format_markdown(report)
    assert "GUIDANCE ONLY" in markdown and "heads_labelled" in markdown


def test_report_marks_an_arm_incomplete_until_its_three_trainings_finish():
    results = {("heads_not_here", s): _results("heads_not_here", ["", "", CALLEE, CALLEE]) for s in (1, 2, 3)}
    results.update({("heads_labelled", s): _results("heads_labelled", [PLANNING, PLANNING, CALLEE, CALLEE])
                    for s in (1, 2)})
    runs = _runs("heads_not_here", (1, 2, 3)) + _runs("heads_labelled", (1, 2)) \
        + _runs("heads_labelled", (3,), status="failed")
    report = step1.build_report(runs, results, [PLANNING])
    assert "heads_labelled" in report["incomplete"]
    assert report["pooled"]["heads_labelled"]["runs_pooled"] == 2
    assert report["incomplete"]["heads_labelled"].endswith("2 of 3 trainings finished")



def test_rescore_scores_every_arm_and_seed_on_the_same_deduped_cases(tmp_path):
    """Stored predictions only. The set holds three heldout requests; a fourth stored request is not in
    it (dropped), and the labelled runs trained on one callee request, which leaves every arm."""
    root = tmp_path / "run"
    kept = [_case("who has this book", PLANNING, "planning", PLANNING),
            _case("find the book called dune", CALLEE, "callee", CALLEE),
            _case("look up the dune title", CALLEE, "callee", CALLEE)]
    set_path = tmp_path / "set.json"
    set_path.write_text(json.dumps({"schema": 1, "workflow": "library", "planning_commands": [PLANNING],
                                    "cases": [{k: v for k, v in c.items() if k != "topk"} for c in kept]}))
    runs = []
    for arm in ("heads_not_here", "heads_labelled"):
        for seed in (1, 2):
            directory = root / f"{arm}_seed{seed}"
            directory.mkdir(parents=True)
            callee_top = PLANNING if arm == "heads_labelled" else CALLEE
            cases = [
                _case("who has this book", PLANNING, "planning", PLANNING),
                _case("find the book called dune", CALLEE, "callee", callee_top),
                _case("look up the dune title", CALLEE, "callee", callee_top),
                _case("loan report for books", PLANNING, "planning", ""),  # not in the deduped set
            ]
            if arm == "heads_labelled":
                cases[2]["trained_on"] = True
            (directory / step1.RESULTS_FILENAME).write_text(json.dumps({"schema": 1, "arm": arm,
                                                                       "cases": cases, "training_rows": {}}))
            runs.append({"name": f"{arm}_seed{seed}", "arm": arm, "seed": seed, "status": "ok",
                         "directory": str(directory)})
    (root / step1.RUNS_FILENAME).write_text(json.dumps({"runs": runs}))

    report = step1.rescore(str(root), str(set_path))
    assert report["rescore"]["deduped_heldout_cases"] == 3
    assert report["rescore"]["trained_on_excluded"] == 1
    assert report["rescore"]["analysis_cases"] == 2
    for arm in ("heads_not_here", "heads_labelled"):
        assert report["pooled"][arm]["planning_routed"]["trials"] == 2  # one planning case, two seeds
        assert report["pooled"][arm]["callee_routed"]["trials"] == 2    # one callee case, two seeds
    assert report["pooled"]["heads_labelled"]["callee_routed"]["count"] == 0
    assert report["pooled"]["heads_not_here"]["callee_routed"]["count"] == 2
    assert report["diagnostics"]["heads_labelled"]["pooled"]["callee_routed_to_a_planning_command"] == 2
    assert report["comparison_pooled"]["heads_labelled"]["paired_n"] == 4

"""Phase 9 step 1: arm provenance, the refusal on a mismatched arm, the holdout cases, and the
pilot driver's guards. No training runs here: the provenance writer and the planner are called
directly, and the refusal reads a hand-written provenance file.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import fastworkflow
from fastworkflow.command_routing import RoutingRegistry
from fastworkflow.model_pipeline_training import (
    HOLDOUT_EXPORT_FILENAME,
    RESERVED_CLASS_ARMS,
    _export_holdout,
    write_holdout_export,
)
from fastworkflow.train import determinism, not_here_pilot
from fastworkflow.train import score_not_here as scorer
from fastworkflow.train.heldout_evaluation import LabeledUtterance, PersonaSplit

REPO = Path(__file__).resolve().parent.parent
MESSAGING = REPO / "fastworkflow" / "examples" / "messaging_app_4"


@pytest.fixture(scope="module", autouse=True)
def initialized_fastworkflow():
    fastworkflow.init({})
    yield
    RoutingRegistry.clear_registry()


def _write_provenance(folder: Path, payload: dict) -> None:
    (folder / "___command_info").mkdir(parents=True, exist_ok=True)
    (folder / "___command_info" / determinism.PROVENANCE_FILENAME).write_text(json.dumps(payload), encoding="utf-8")


# ---------------------------------------------------------------- provenance writer

def test_provenance_records_the_arm_and_a_shipped_run_writes_no_new_key(tmp_path):
    shipped = tmp_path / "shipped"
    shipped.mkdir()
    determinism.ProvenanceRecorder(str(shipped)).save()
    payload = json.loads((shipped / "___command_info" / determinism.PROVENANCE_FILENAME).read_text())
    assert "reserved_class_arm" not in payload
    assert determinism.ProvenanceRecorder.load_run_metadata(str(shipped)) == {}

    armed = tmp_path / "armed"
    armed.mkdir()
    recorder = determinism.ProvenanceRecorder(str(armed))
    recorder.run_metadata = {"reserved_class_arm": "sparse_8"}
    recorder.save()
    assert determinism.ProvenanceRecorder.load_run_metadata(str(armed)) == {"reserved_class_arm": "sparse_8"}


def test_arm_recorded_in_provenance_is_what_the_scorer_reads(tmp_path):
    _write_provenance(tmp_path, {"schema_version": 2, "commands": {}, "context_training": {},
                                 "reserved_class_arm": "ratio_0_5"})
    assert scorer.recorded_arm(str(tmp_path)) == "ratio_0_5"


# ---------------------------------------------------------------- refusal

def test_predict_refuses_a_mismatched_arm_before_loading_any_model(tmp_path):
    """Phase 9 step 5: ``control`` is retired, so the mismatch case uses a live arm instead."""
    _write_provenance(tmp_path, {"schema_version": 2, "commands": {}, "context_training": {},
                                 "reserved_class_arm": "sparse_8"})
    with pytest.raises(scorer.ArmMismatchError, match="sparse_8"):
        scorer.check_arm(str(tmp_path), "sparse_1")
    # predict fails on the arm check; no artifact version exists, so reaching a router would fail differently
    with pytest.raises(scorer.ArmMismatchError):
        scorer.predict_cases(str(tmp_path), [], "sparse_1")
    assert scorer.check_arm(str(tmp_path), "sparse_8") == "sparse_8"
    assert scorer.check_arm(str(tmp_path), None) == "sparse_8"


def test_a_retired_arm_is_refused_by_name_whether_asked_for_or_recorded(tmp_path):
    """Phase 9 step 5 (was: an unarmed artifact is control and refuses an armed request).

    The unrecorded arm is still the label ``control`` for history, but ``control`` no longer
    exists to score, so both the request and the recorded label are refused with the reason.
    """
    _write_provenance(tmp_path, {"schema_version": 2, "commands": {}, "context_training": {}})
    assert scorer.recorded_arm(str(tmp_path)) == "control"
    with pytest.raises(ValueError, match="retired"):
        scorer.check_arm(str(tmp_path), None)
    with pytest.raises(ValueError, match="retired"):
        scorer.check_arm(str(tmp_path), "control")


def test_fallen_back_commands_are_detected_from_provenance(tmp_path):
    _write_provenance(tmp_path, {
        "commands": {"A/one": {"fell_back": True}, "A/two": {"fell_back": False}},
        "context_training": {"A": {"A/one": {"status": "included_fallback"}}},
    })
    assert not_here_pilot.fallen_back_commands(str(tmp_path)) == ["A/one"]


# ---------------------------------------------------------------- holdout export and cases

def _record(utterance, label, persona):
    return LabeledUtterance(utterance=utterance, label=label, persona=persona)


def test_holdout_export_is_off_unless_an_arm_is_exporting(tmp_path):
    from fastworkflow import model_pipeline_training as mpt

    split = PersonaSplit(heldout=[_record("a1", "A/one", "p1")], train=[])
    assert mpt._HOLDOUT_EXPORT is None
    _export_holdout("A", {"A/one"}, split)  # no-op: nothing to write into
    assert mpt._HOLDOUT_EXPORT is None

    mpt._HOLDOUT_EXPORT = {}
    try:
        split = PersonaSplit(
            heldout=[_record("a1", "A/one", "p1"), _record("x9", "B/x", "p2"),
                     _record("bare", "parameter_value", "p3")],
            train=[_record("x1", "wildcard", "p4"), _record("a0", "A/one", "p5")],
        )
        _export_holdout("A", {"A/one"}, split)
        export = mpt._HOLDOUT_EXPORT["A"]
    finally:
        mpt._HOLDOUT_EXPORT = None
    assert export["own_labels"] == ["A/one"]
    assert export["own_heldout"] == [{"utterance": "a1", "label": "A/one", "persona": "p1"}]
    assert export["not_here_training"] == ["x1"]

    path = write_holdout_export(str(tmp_path), "sparse_8", 42)
    assert os.path.basename(path) == HOLDOUT_EXPORT_FILENAME
    assert json.loads(Path(path).read_text())["reserved_class_arm"] == "sparse_8"


def test_build_cases_scores_both_directions_and_does_not_depend_on_the_arm():
    holdout = {"heads": {
        "A": {"own_labels": ["A/one", "A/two"],
              "own_heldout": [{"utterance": "a1", "label": "A/one", "persona": "p"},
                              {"utterance": "a2", "label": "A/two", "persona": "p"}],
              "not_here_training": ["x1"]},
        "B": {"own_labels": ["B/x"],
              "own_heldout": [{"utterance": "b1", "label": "B/x", "persona": "q"},
                              {"utterance": "x1", "label": "B/x", "persona": "q"}],
              "not_here_training": []},
    }}
    cases = scorer.build_cases(holdout)
    got = {(c["head"], c["utterance"], c["expected"], c["in_scope"]) for c in cases}
    assert got == {
        # head A: own held-out cases, then the foreign command B/x (both of its held-out texts)
        ("A", "a1", "A/one", True),
        ("A", "a2", "A/two", True),
        ("A", "b1", "B/x", False),
        ("A", "x1", "B/x", False),
        # head B: its own held-out cases, then A's commands; x1 is B's own label, so not a foreign case
        ("B", "b1", "B/x", True),
        ("B", "x1", "B/x", True),
        ("B", "a1", "A/one", False),
        ("B", "a2", "A/two", False),
    }
    assert len(cases) == len(got)  # no duplicates
    # the arm's own trained not-here rows must not change the cases
    holdout["heads"]["A"]["not_here_training"] = ["x1", "b1"]
    assert scorer.build_cases(holdout) == cases


# ---------------------------------------------------------------- planner under an arm

def test_under_an_arm_the_planner_retrains_every_head_from_scratch():
    from fastworkflow.train.selective_training import compute_training_plan, contexts_for_training

    folder = str(MESSAGING)
    contexts = contexts_for_training(folder)
    plan, _signature = compute_training_plan(
        folder, contexts, changed_only=False, seed=42, carry_forward_from=None)
    assert plan.is_full_retrain is True
    assert set(plan.contexts_to_train) == set(contexts)
    assert plan.contexts_carried_forward == []


# ---------------------------------------------------------------- pilot driver guards

def test_arm_choices_of_the_driver_match_the_trainer():
    assert set(not_here_pilot.ARM_CHOICES) == set(RESERVED_CLASS_ARMS)


def test_blocked_env_invalidates_keys_but_keeps_model_strings():
    env = {"LITELLM_API_KEY_SYNDATA_GEN": "sk-real", "LLM_SYNDATA_GEN": "mistral/mistral-small-latest",
           "LITELLM_PROXY_API_BASE": "http://proxy.internal:4000", "LITELLM_API_KEY_AGENT": "sk-agent"}
    blocked = not_here_pilot.blocked_env(env)
    assert blocked["LITELLM_API_KEY_SYNDATA_GEN"] == not_here_pilot.BLOCKED_KEY
    assert blocked["LITELLM_API_KEY_AGENT"] == not_here_pilot.BLOCKED_KEY
    assert blocked["LLM_SYNDATA_GEN"] == "mistral/mistral-small-latest"  # the cache key depends on it
    assert blocked["LITELLM_PROXY_API_BASE"] == not_here_pilot.CLOSED_PROXY_BASE
    assert blocked["LLM_CALLS_BLOCKED_BY_PILOT"] == "1"
    assert env["LITELLM_API_KEY_SYNDATA_GEN"] == "sk-real"  # the input is not mutated


def test_one_seed_edit_changes_only_the_copy(tmp_path):
    workflow_copy = tmp_path / "copy"
    shutil.copytree(MESSAGING, workflow_copy, ignore=shutil.ignore_patterns("___*", "__pycache__"))
    before = {p: p.read_text() for p in (workflow_copy / "_commands").rglob("*.py")}
    source_before = {p: p.read_text() for p in (MESSAGING / "_commands").rglob("*.py")}

    path, command, seed = not_here_pilot.add_one_seed(str(workflow_copy))

    assert seed in Path(path).read_text()
    changed = [p for p, text in before.items() if p.read_text() != text]
    assert [str(p) for p in changed] == [path]
    assert {p: p.read_text() for p in (MESSAGING / "_commands").rglob("*.py")} == source_before


def test_driver_dry_run_prints_the_plan_and_touches_nothing(tmp_path):
    scratch = tmp_path / "scratch"
    cache = tmp_path / "cache"
    (cache / "___command_info" / "utterance_cache").mkdir(parents=True)
    result = subprocess.run(
        [sys.executable, "-m", "fastworkflow.train.not_here_pilot", "--arm", "sparse_8",
         "--scratch", str(scratch), "--env-file", "fastworkflow.env.nemotron",
         "--source", str(MESSAGING), "--cache-from", str(cache), "--dry-run"],
        cwd=REPO, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "dry run" in result.stdout and "train" in result.stdout
    assert not scratch.exists()


def test_driver_refuses_a_seed_edit_without_llm_permission(tmp_path):
    scratch = tmp_path / "scratch"
    result = subprocess.run(
        [sys.executable, "-m", "fastworkflow.train.not_here_pilot", "--arm", "sparse_8",
         "--scratch", str(scratch), "--env-file", "x.env", "--one-seed-edit",
         "--source", str(tmp_path / "source"), "--cache-from", str(tmp_path / "cache")],
        cwd=REPO, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 2
    assert "allow-llm-for-edit" in result.stderr
    assert not scratch.exists()

"""Phase 9 step 1 runner: the seed handling, the pooling, the bar and the verdict, on hand-built
results. No training: the seed tests call the seed functions and the sampler directly, on a
private copy of the library example.
"""

from __future__ import annotations

import io
import shutil
from contextlib import redirect_stdout
from pathlib import Path

import pytest

import fastworkflow
from fastworkflow.command_routing import RoutingRegistry
from fastworkflow.model_pipeline_training import choose_not_here_sample, not_here_seed_rows, trained_command_labels
from fastworkflow.train import determinism
from fastworkflow.train import not_here_step1 as step1
from fastworkflow.train import score_not_here as scorer
from fastworkflow.train.heldout_evaluation import LabeledUtterance, split_by_persona

LIBRARY = Path(__file__).resolve().parent.parent / "fastworkflow" / "examples" / "library"


@pytest.fixture(scope="module", autouse=True)
def initialized_fastworkflow():
    fastworkflow.init({})
    yield
    determinism.set_train_seed(None)
    RoutingRegistry.clear_registry()


@pytest.fixture(autouse=True)
def no_leftover_train_seed():
    determinism.set_train_seed(None)
    yield
    determinism.set_train_seed(None)


# ---------------------------------------------------------------- seeds

def test_the_split_seed_and_the_persona_split_do_not_depend_on_the_train_seed():
    records = [LabeledUtterance(utterance=f"u{p}-{i}", label="A/one", persona=f"p{p}")
               for p in range(8) for i in range(3)]
    determinism.set_train_seed(None)
    default_split = determinism.get_split_seed()
    default_model = determinism.get_model_seed()
    determinism.set_train_seed(1)
    seed_one = (determinism.get_split_seed(), determinism.get_model_seed())
    determinism.set_train_seed(2)
    seed_two = (determinism.get_split_seed(), determinism.get_model_seed())

    assert default_model == default_split == determinism.get_training_seed() == 42
    assert seed_one[0] == seed_two[0] == default_split          # the split never moves
    assert seed_one[1] != seed_two[1]                           # model randomness does
    held_one = {r.utterance for r in split_by_persona(records, seed=seed_one[0]).heldout}
    held_two = {r.utterance for r in split_by_persona(records, seed=seed_two[0]).heldout}
    assert held_one == held_two and held_one


def test_the_chosen_reserved_rows_repeat_and_do_not_vary_with_the_train_seed():
    """Design 2.50: a sample is chosen once and recorded, so the train seed does not redraw it.

    The choice is a stable hash of (head, command, seed text), so it is the same in every call
    and every process, and nothing about the train seed reaches it.
    """
    seeds = [f"utterance number {i}" for i in range(10)]
    same_a = choose_not_here_sample("A", "A/one", seeds, 3)
    same_b = choose_not_here_sample("A", "A/one", seeds, 3)
    assert same_a == same_b
    assert len(same_a) == 3 and set(same_a) <= set(seeds)
    assert len(set(choose_not_here_sample("A", "A/one", seeds, 10))) == 10  # ranks every seed
    determinism.set_train_seed(1)
    with_one = choose_not_here_sample("A", "A/one", seeds, 3)
    determinism.set_train_seed(2)
    with_two = choose_not_here_sample("A", "A/one", seeds, 3)
    assert with_one == with_two == same_a


def planning_names_of(folder: str) -> set[str]:
    from fastworkflow.workflow_agent import planning_command_utterances

    return set(planning_command_utterances(folder))


def planning_utterances_of(folder: str) -> dict:
    from fastworkflow.workflow_agent import planning_command_utterances

    return planning_command_utterances(folder)


def test_the_default_reserved_rows_are_the_stable_hash_choice(tmp_path):
    destination = tmp_path / "library_copy"
    shutil.copytree(LIBRARY, destination, ignore=shutil.ignore_patterns("___*", "__pycache__"))
    RoutingRegistry.clear_registry()
    crd = RoutingRegistry.get_definition(str(destination))
    labels = trained_command_labels("Book", crd)

    determinism.set_train_seed(None)
    default_picks, _ = not_here_seed_rows(str(destination), crd, labels, 1, context_name="Book")
    determinism.set_train_seed(1)
    first, _ = not_here_seed_rows(str(destination), crd, labels, 1, context_name="Book")
    again, _ = not_here_seed_rows(str(destination), crd, labels, 1, context_name="Book")
    determinism.set_train_seed(2)
    second, _ = not_here_seed_rows(str(destination), crd, labels, 1, context_name="Book")
    determinism.set_train_seed(None)

    assert first == again == second == default_picks             # the train seed never redraws
    cmd_dir = crd.command_directory
    assert default_picks, "the library example has foreign commands"
    for name, rows in default_picks.items():                     # each is the stable-hash choice
        if name in planning_names_of(str(destination)):
            seeds = list(planning_utterances_of(str(destination))[name])
        else:
            seeds = list(cmd_dir.get_utterance_metadata(name).plain_utterances) if cmd_dir.get_utterance_metadata(name) else []
        assert rows == choose_not_here_sample("Book", name, seeds, 1)


def test_the_persona_split_is_taken_over_the_split_seed_only():
    """The trainer's split call passes get_split_seed(); this pins that the function output is the
    same for two train seeds when given that seed."""
    records = [LabeledUtterance(utterance=f"x{p}-{i}", label="B/y", persona=f"q{p}")
               for p in range(10) for i in range(2)]
    determinism.set_train_seed(5)
    first = [r.utterance for r in split_by_persona(records, seed=determinism.get_split_seed()).heldout]
    determinism.set_train_seed(9)
    second = [r.utterance for r in split_by_persona(records, seed=determinism.get_split_seed()).heldout]
    assert first == second


def test_the_pilot_command_carries_the_train_seed_and_the_arm():
    spec = step1.RunSpec(arm="sparse_8", seed=3, directory="/x/sparse_8_seed3")
    command = step1.pilot_command(spec, source="/src", cache_from="/cache", env_file="/env.env",
                                  passwords_file=None)
    assert command[command.index("--arm") + 1] == "sparse_8"
    assert command[command.index("--train-seed") + 1] == "3"
    assert "--passwords-file" not in command


def test_the_provenance_records_the_train_seed_next_to_the_arm(tmp_path):
    from fastworkflow.train.determinism import ProvenanceRecorder

    recorder = ProvenanceRecorder(str(tmp_path))
    recorder.run_metadata = {"reserved_class_arm": "sparse_8", "train_seed": 2}
    recorder.save()
    assert ProvenanceRecorder.load_run_metadata(str(tmp_path)) == {"reserved_class_arm": "sparse_8", "train_seed": 2}


# ---------------------------------------------------------------- hand-built results

def _case(index: int, in_scope: bool, top: str) -> dict:
    expected = "A/one" if in_scope else "B/x"
    return {"head": "A", "utterance": f"u{index}", "expected": expected, "in_scope": in_scope,
            "top_label": top, "topk": [top]}


def _cases(*, decline_fixed: bool, accept_wrong: int = 0) -> list[dict]:
    """400 in-scope and 100 out-of-scope cases. Control declines the first 100 in-scope cases and
    the first 100 out-of-scope cases are declined too; the arm answers the in-scope ones
    correctly when *decline_fixed*, and accepts *accept_wrong* of the out-of-scope ones."""
    cases = []
    for i in range(400):
        control_top = "wildcard" if i < 100 else "A/one"
        arm_top = "A/one" if (decline_fixed or i >= 100) else "wildcard"
        cases.append((_case(i, True, control_top), _case(i, True, arm_top)))
    for j in range(100):
        index = 400 + j
        control_top = "wildcard"
        arm_top = "A/one" if j < accept_wrong else "wildcard"
        cases.append((_case(index, False, control_top), _case(index, False, arm_top)))
    return cases


def _arm_results(arm: str, pairs, which: int) -> dict:
    return {"schema": 1, "arm": arm, "cases": [pair[which] for pair in pairs]}


def test_pooling_concatenates_the_seeds_case_by_case_and_refuses_different_cases():
    pairs = _cases(decline_fixed=True)
    results = [_arm_results("sparse_1", pairs, 1) for _ in range(3)]
    pooled = step1.pool_arm(results, "sparse_1")
    assert pooled["runs_pooled"] == 3
    assert len(pooled["cases"]) == 3 * len(pairs)
    assert pooled["cases"][0] == pairs[0][1]
    different = {"cases": results[0]["cases"][:-1] + [{**results[0]["cases"][-1], "utterance": "other"}]}
    with pytest.raises(ValueError, match="different cases"):
        step1.pool_arm([results[0], different], "sparse_1")


def test_a_paired_difference_matches_the_hand_computation():
    # 4 in-scope cases; control declines the first, the arm declines none: difference -0.25,
    # n10 = 0, n01 = 1, variance = (1 - 1/4) / 16 = 0.046875
    control = [_case(i, True, "wildcard" if i == 0 else "A/one") for i in range(4)]
    arm = [_case(i, True, "A/one") for i in range(4)]
    block = step1.paired_difference(arm, control, "wrong_declines")
    assert block["difference"] == pytest.approx(-0.25)
    half = 1.959963984540054 * (0.046875 ** 0.5)
    assert block["lower"] == pytest.approx(-0.25 - half)
    assert block["upper"] == pytest.approx(-0.25 + half)


def test_the_arm_that_fixes_the_declines_and_keeps_accepts_passes_the_bar():
    control = [c for c, _ in _cases(decline_fixed=True)]
    arm = [a for _, a in _cases(decline_fixed=True)]
    evaluation = step1.evaluate_arm(arm * 3, control * 3)
    assert evaluation["wrong_declines"]["passes"] is True
    assert evaluation["wrong_accepts"]["passes"] is True   # no accepts added
    assert evaluation["top1"]["passes"] is True
    assert evaluation["passes"] is True


def test_an_arm_that_accepts_many_wrong_requests_fails_the_accept_bar():
    pairs = _cases(decline_fixed=True, accept_wrong=30)
    control = [c for c, _ in pairs]
    arm = [a for _, a in pairs]
    evaluation = step1.evaluate_arm(arm * 3, control * 3)
    assert evaluation["wrong_accepts"]["upper"] > step1.MAX_WRONG_ACCEPT_RISE
    assert evaluation["wrong_accepts"]["passes"] is False
    assert evaluation["passes"] is False


def test_an_arm_that_does_not_fix_declines_fails_the_decline_bar():
    pairs = _cases(decline_fixed=False)
    control = [c for c, _ in pairs]
    arm = [a for _, a in pairs]
    evaluation = step1.evaluate_arm(arm * 3, control * 3)
    assert evaluation["wrong_declines"]["passes"] is False


def test_the_verdict_prefers_one_seed_unless_eight_is_measurably_better():
    evaluations = {"sparse_1": {"passes": True}, "sparse_8": {"passes": True}, "ratio_0_5": {"passes": False}}
    tie = {"paired": {"wrong_accept": {"p_two_sided": 1.0, "errors_a": 4, "errors_b": 4},
                      "top1": {"p_two_sided": 0.4, "errors_a": 9, "errors_b": 7}}}
    better = {"paired": {"wrong_accept": {"p_two_sided": 0.01, "errors_a": 9, "errors_b": 3},
                         "top1": {"p_two_sided": 1.0, "errors_a": 5, "errors_b": 5}}}
    assert step1.choose(evaluations, tie)["chosen"] == "sparse_1"
    assert step1.choose(evaluations, better)["chosen"] == "sparse_8"
    assert step1.choose({"sparse_1": {"passes": False}, "sparse_8": {"passes": False}}, tie)["chosen"] is None
    only_eight = {"sparse_1": {"passes": False}, "sparse_8": {"passes": True}}
    assert step1.choose(only_eight, None)["chosen"] == "sparse_8"


def _complete_runs(results_for_arm) -> tuple[list[dict], dict]:
    runs, collected = [], {}
    for arm in step1.ARMS:
        for seed in step1.TRAIN_SEEDS:
            runs.append({"name": f"{arm}_seed{seed}", "arm": arm, "seed": seed, "status": "ok"})
            collected[(arm, seed)] = results_for_arm(arm)
    return runs, collected


def test_the_report_passes_the_good_arms_and_chooses_one_seed_on_a_complete_set(tmp_path):
    good = _cases(decline_fixed=True)
    bad = _cases(decline_fixed=True, accept_wrong=30)

    def results_for(arm):
        pairs = bad if arm == "ratio_0_5" else good
        if arm == "control":
            return {"schema": 1, "arm": arm, "cases": [c for c, _ in pairs]}
        return {"schema": 1, "arm": arm, "cases": [a for _, a in pairs]}

    runs, collected = _complete_runs(results_for)
    report = step1.build_report(runs, collected)
    assert report["evaluations"]["sparse_1"]["passes"] is True
    assert report["evaluations"]["sparse_8"]["passes"] is True
    assert report["evaluations"]["ratio_0_5"]["passes"] is False
    assert report["verdict"]["chosen"] == "sparse_1"
    text = step1.format_markdown(report)
    assert "Verdict" in text and "sparse_1" in text and "Method" in text


def test_the_report_is_incomplete_while_an_arm_has_no_finished_run():
    good = _cases(decline_fixed=True)
    runs = [{"name": "control_seed1", "arm": "control", "seed": 1, "status": "ok"}]
    collected = {("control", 1): {"schema": 1, "arm": "control", "cases": [c for c, _ in good]}}
    report = step1.build_report(runs, collected)
    assert report["verdict"]["chosen"] is None
    assert report["verdict"]["decision"].startswith("INCOMPLETE")


# ---------------------------------------------------------------- planning and resume

def test_a_dry_run_lists_six_trainings_and_skips_the_finished_ones(tmp_path, capsys):
    """Phase 9 step 5: the two retired arms are no longer trained, so a new run is 2 arms x 3 seeds.""" 
    root = tmp_path / "step1"
    finished = root / "sparse_1_seed1"
    finished.mkdir(parents=True)
    (finished / step1.SUMMARY_FILENAME).write_text("{}")

    code = step1.main(["--root", str(root), "--source", str(tmp_path / "source"),
                       "--cache-from", str(tmp_path / "cache"),
                       "--env-file", str(tmp_path / "env"), "--dry-run", "--resume"])

    out = capsys.readouterr().out
    assert code == 0
    assert out.count("--train-seed") == 6
    assert "SKIPPED" in out
    assert not (root / "control_seed1").exists()          # nothing was created
    assert len(step1.plan_runs(str(root))) == 6
    with pytest.raises(ValueError, match="retired"):      # a retired arm is refused, not trained
        step1.plan_runs(str(root), arms=("control",))

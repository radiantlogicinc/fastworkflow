"""Threshold sweep on the "not here" probability: decision rule, metrics, bar, folds, cross-fit,
AUC, alignment and formatting, on hand-built case lists. No model is loaded.
"""

from __future__ import annotations

import hashlib
import math

import pytest

from fastworkflow.train import not_here_threshold as nht

INF = math.inf
BOOK = "Book/loan_report"
CHECK = "Book/check_status"
CARD = "Card/block"


def probe_case(head, utterance, expected, in_scope, top, p_not_here, topk=None):
    return {"head": head, "utterance": utterance, "expected": expected, "in_scope": in_scope,
            "top_label": top, "p_not_here": p_not_here, "topk": topk if topk is not None else [top],
            "top_prob": 0.9, "used_distil": False}


def control_case(probe):
    """The control results row for a probe row, as score_not_here predict writes it."""
    return {k: probe[k] for k in ("head", "utterance", "expected", "in_scope", "top_label", "topk")}


def toy_probe():
    """Six cases: two wrong declines/accepts become visible only at some thresholds."""
    return [
        probe_case("Book", "renew it", BOOK, True, BOOK, 0.10, [BOOK, CHECK, CARD]),          # accepted, right
        probe_case("Book", "where is it", BOOK, True, "wildcard", 0.20, ["wildcard", BOOK, CHECK]),  # declined
        probe_case("Book", "pay my fine", BOOK, True, CHECK, 0.60, [CHECK, BOOK, CARD]),       # wrong label
        probe_case("Book", "card balance", CARD, False, CARD, 0.70, [CARD, BOOK, CHECK]),      # wrong accept
        probe_case("Book", "france", CARD, False, "parameter_value", 0.05, ["parameter_value", CARD, BOOK]),
        probe_case("Book", "block card", CARD, False, BOOK, 0.90, [BOOK, CARD, CHECK]),        # wrong accept
    ]


# ---------------------------------------------------------------- decision rule

def test_decline_rule_at_threshold():
    accepted_low = probe_case("Book", "u", BOOK, True, BOOK, 0.30)
    assert nht.declines_at(accepted_low, 0.5) is False
    assert nht.declines_at(accepted_low, 0.30) is True  # boundary: >= declines
    assert nht.declines_at(accepted_low, INF) is False  # inf is the current behaviour


def test_non_routable_top_declines_at_any_threshold():
    declined = probe_case("Book", "u", BOOK, True, "wildcard", 0.0)
    assert nht.declines_at(declined, INF) is True
    assert nht.declines_at(declined, 1.0) is True


def test_empty_top_label_declines():
    assert nht.declines_at(probe_case("Book", "u", BOOK, True, "", 0.0), INF) is True


def test_scored_case_replaces_threshold_declines_with_wildcard_and_empties_topk():
    case = toy_probe()[2]  # wrong label CHECK, p 0.60
    kept = nht.scored_case(case, 0.8)
    assert kept["top_label"] == CHECK and kept["topk"] == [CHECK, BOOK, CARD]
    declined = nht.scored_case(case, 0.5)
    assert declined["top_label"] == "wildcard" and declined["topk"] == []


def test_not_here_mass_sums_only_non_routable_labels():
    mass = nht.not_here_mass(["Book/x", "wildcard", "parameter_value"], [0.5, 0.3, 0.2])
    assert mass == pytest.approx(0.5)


# ---------------------------------------------------------------- metrics

def test_metrics_at_inf_match_the_current_behaviour_counts():
    probe = toy_probe()
    control = [control_case(c) for c in probe]
    metrics = nht.metrics_at(probe, control, INF)["overall"]
    assert metrics["wrong_declines"]["count"] == 1 and metrics["wrong_declines"]["trials"] == 3
    assert metrics["wrong_accepts"]["count"] == 2 and metrics["wrong_accepts"]["trials"] == 3
    assert metrics["top1"]["count"] == 1 and metrics["top1"]["trials"] == 3


def test_metrics_at_half_declines_in_scope_and_out_of_scope_cases():
    probe = toy_probe()
    control = [control_case(c) for c in probe]
    at_half = nht.metrics_at(probe, control, 0.5)
    overall = at_half["overall"]
    # in scope: CHECK (p 0.60) is now declined as well as the wildcard case
    assert overall["wrong_declines"]["count"] == 2
    # out of scope: 0.70 and 0.90 are declined; the parameter_value case stays declined
    assert overall["wrong_accepts"]["count"] == 0
    assert overall["top1"]["count"] == 1
    diff = at_half["vs_control"]["wrong_declines"]
    assert diff["difference"] == pytest.approx(1 / 3)
    assert diff["n"] == 3


def test_threshold_only_adds_declines_so_the_wrong_decline_difference_is_never_negative():
    probe = toy_probe()
    control = [control_case(c) for c in probe]
    for t in nht.GRID:
        diff = nht.metrics_at(probe, control, t)["vs_control"]["wrong_declines"]["difference"]
        assert diff is None or diff >= 0


def test_control_against_itself_fails_the_wrong_decline_bar():
    probe = toy_probe()
    control = [control_case(c) for c in probe]
    metrics = nht.metrics_at(probe, control, INF)
    assert metrics["vs_control"]["wrong_declines"]["upper"] == pytest.approx(0.0)
    assert metrics["passes_bar"] is False


# ---------------------------------------------------------------- bar and choice

def _grid_entry(count, passes, passes_without_decline=None):
    return {"overall": {"wrong_accepts": {"count": count}},
            "passes_bar": passes,
            "passes_bar_without_decline": passes if passes_without_decline is None else passes_without_decline}


def test_choose_threshold_minimises_wrong_accepts_among_passing_then_prefers_higher_t():
    grid = {0.3: _grid_entry(2, True), 0.5: _grid_entry(1, True), 0.7: _grid_entry(1, True),
            0.9: _grid_entry(0, False), INF: _grid_entry(0, False)}
    assert nht.choose_threshold(grid) == 0.7


def test_choose_threshold_picks_inf_when_nothing_passes():
    grid = {0.3: _grid_entry(0, False), INF: _grid_entry(2, False)}
    assert nht.choose_threshold(grid) == INF


def test_choose_threshold_uses_the_without_decline_bar_when_asked():
    grid = {0.3: _grid_entry(2, False, passes_without_decline=True), INF: _grid_entry(2, False)}
    assert nht.choose_threshold(grid, bar="full") == INF
    assert nht.choose_threshold(grid, bar="without_decline") == 0.3


# ---------------------------------------------------------------- folds

def test_fold_assignment_is_the_sha256_parity_and_stable():
    expected = int(hashlib.sha256(BOOK.encode("utf-8")).hexdigest(), 16) % 2
    assert nht.fold_of(BOOK) == expected
    assert nht.fold_of(BOOK) == nht.fold_of(BOOK)
    cases = toy_probe()
    assert nht.assign_folds(cases, "expected") == nht.assign_folds(cases, "expected")


def test_folds_by_expected_keep_every_case_of_a_command_together():
    cases = toy_probe()
    folds = nht.assign_folds(cases, "expected")
    by_expected = {}
    for case, fold in zip(cases, folds):
        by_expected.setdefault(case["expected"], set()).add(fold)
    assert all(len(v) == 1 for v in by_expected.values())


def test_fold_key_must_be_known():
    with pytest.raises(ValueError):
        nht.assign_folds(toy_probe(), "head")


# ---------------------------------------------------------------- cross-fit

def test_crossfit_picks_inf_when_nothing_passes_and_pooled_equals_control():
    probe = toy_probe()
    control = [control_case(c) for c in probe]
    folds = [0, 1, 0, 1, 0, 1]
    result = nht.crossfit(probe, control, folds, bar="full")
    assert result["picks"] == {"fold_0": "inf", "fold_1": "inf"}
    pooled = result["pooled"]["overall"]
    control_overall = nht.metrics_at(probe, control, INF)["overall"]
    assert pooled["wrong_accepts"]["count"] == control_overall["wrong_accepts"]["count"]
    assert pooled["wrong_declines"]["count"] == control_overall["wrong_declines"]["count"]
    assert pooled["top1"]["count"] == control_overall["top1"]["count"]


def test_crossfit_scores_each_fold_on_the_other_fold():
    probe = toy_probe()
    control = [control_case(c) for c in probe]
    folds = [0, 0, 0, 1, 1, 1]
    result = nht.crossfit(probe, control, folds, bar="full")
    assert result["folds"]["fold_0"]["scored_on_other_fold"]["n"] == 3
    assert result["folds"]["fold_1"]["scored_on_other_fold"]["n"] == 3


# ---------------------------------------------------------------- AUC and quantiles

def test_auc_on_a_known_toy_example():
    # pairs: 0.9>0.1, 0.9>0.85, 0.8>0.1, 0.8<0.85 -> 3 of 4
    assert nht.auc_by_rank([0.9, 0.8], [0.1, 0.85]) == pytest.approx(0.75)


def test_auc_perfect_separation_and_full_ties():
    assert nht.auc_by_rank([0.6, 0.7], [0.1, 0.2]) == pytest.approx(1.0)
    assert nht.auc_by_rank([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.5)


def test_auc_is_none_without_both_classes():
    assert nht.auc_by_rank([], [0.1]) is None


def test_quantile_interpolates_linearly():
    assert nht.quantile([1.0, 2.0, 3.0, 4.0], 0.5) == pytest.approx(2.5)
    assert nht.quantile([], 0.5) is None


def test_separation_uses_only_accepted_cases():
    probe = toy_probe()
    summary = nht.separation(probe)
    # accepted at inf: the correct in-scope case, the wrong-label case, and the two wrong accepts
    assert summary["n_in_scope_accepted"] == 2
    assert summary["n_out_of_scope_accepted"] == 2
    assert summary["auc_out_of_scope_vs_in_scope"] == pytest.approx(1.0)  # 0.70, 0.90 vs 0.10, 0.60


# ---------------------------------------------------------------- alignment

def test_alignment_passes_for_identical_cases():
    probe = toy_probe()
    nht.align(probe, [control_case(c) for c in probe])


def test_alignment_fails_loudly_on_a_swapped_utterance():
    probe = toy_probe()
    control = [control_case(c) for c in probe]
    control[1], control[2] = control[2], control[1]
    with pytest.raises(nht.CaseAlignmentError, match="utterance differs"):
        nht.align(probe, control)


def test_alignment_fails_on_a_length_mismatch():
    probe = toy_probe()
    with pytest.raises(nht.CaseAlignmentError, match="cases"):
        nht.align(probe, [control_case(c) for c in probe[:-1]])


def test_sweep_refuses_mismatched_inputs():
    probe = toy_probe()
    control = [control_case(c) for c in probe]
    control[0] = dict(control[0], in_scope=False)
    with pytest.raises(nht.CaseAlignmentError):
        nht.sweep(probe, control)


# ---------------------------------------------------------------- sweep and formatting

def test_sweep_grid_covers_every_threshold_and_formats_to_markdown():
    probe = toy_probe()
    control = [control_case(c) for c in probe]
    result = nht.sweep(probe, control)
    labels = [m["threshold"] for m in result["grid"]]
    assert labels[0] == "0.00" and labels[-2] == "1.00" and labels[-1] == "inf"
    assert len(labels) == 102
    assert set(result["crossfit"]) == {f"folds_by_{k}/{b}" for k in ("expected", "utterance")
                                       for b in ("full", "without_decline")}
    markdown = nht.format_markdown(result, "control", "v1")
    assert "Selected thresholds" in markdown and "Cross-fit" in markdown and "ROC AUC" in markdown
    assert "| inf |" in markdown


def test_threshold_labels_are_json_friendly():
    assert nht.threshold_label(INF) == "inf"
    assert nht.threshold_label(0.5) == "0.50"

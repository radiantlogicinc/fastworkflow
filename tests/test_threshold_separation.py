"""The tier threshold and the tiny ambiguity threshold must not collapse.

No mocks (repo rule `.cursor/rules/testing_rules.mdc`): every test calls the real
writer `model_pipeline_training.write_ambiguity_thresholds` and reads the real JSON
files back off disk. What is *not* rebuilt here is TinyBERT and DistilBERT -- the
writer takes the confidence statistics the trainer measures, so the statistics are
supplied directly and the arithmetic, the invariant and the files are the real ones.

The context table below is not invented. It is the confidence statistics taken from
a real published router, all 18 of whose contexts had `tiny_ambiguous <= tier` --
7 of them byte-identical -- which is the state that made the tiny tier
structurally incapable of reporting an ambiguity.
"""

import json
import os

import numpy as np
import pytest

from fastworkflow.model_pipeline_training import (
    MAX_AMBIGUITY_THRESHOLD,
    SINGLE_LABEL_RESOLUTION_FLOOR,
    TIER_AMBIGUITY_MIN_SEPARATION,
    floored_large_ambiguous_threshold,
    keeps_only_certain_predictions,
    resolvable_ambiguity_ceiling,
    separated_tiny_ambiguous_threshold,
    write_ambiguity_thresholds,
)


def _stats(failed_mean, successful_mean):
    """The shape `evaluate_confidence_stats` returns; only the means are read."""
    return {
        'failed': {'min': None, 'max': None, 'mean': failed_mean, 'median': None},
        'successful': {'min': None, 'max': None, 'mean': successful_mean, 'median': None},
    }


# context -> (tier threshold, tiny failed mean, tiny successful mean, large failed mean)
# Tier thresholds and the large failed means are the exact floats published in
# 20260905T132341Z-a0605e; the tiny failed mean is that version's published
# tiny_ambiguous_threshold, which the old writer set to precisely that statistic.
PUBLISHED_CONTEXTS = {
    "Account": (0.5476, 0.4452, 0.8123, 0.6387),
    "Application": (0.66119, 0.54636, 0.8641, 0.6063),
    "ControlCatalog": (0.34679, 0.30996, 0.7412, 0.42157),
    "ControlFinding": (0.63501, 0.63501, 0.8802, 0.72594),
    "ControlsMonitor": (0.43638, 0.43638, 0.7955, 0.59442),
    "Directory": (0.32766, 0.25068, 0.7003, 0.41043),
    "DirectoryExplorer": (0.27662, 0.27662, 0.6891, 0.54595),
    "EntityLookup": (0.23795, 0.23795, 0.6502, 0.43015),
    "Group": (0.36315, 0.36315, 0.7314, 0.51176),
    "Identity": (0.505, 0.39996, 0.8210, 0.62953),
    "Organization": (0.47206, 0.39985, 0.8004, 0.6626),
    "Permission": (0.50255, 0.50255, 0.8455, 0.71277),
    "ReconciliationWorkspace": (0.47805, 0.46631, 0.7788, 0.60528),
    "Repository": (0.54802, 0.41738, 0.8339, 0.68464),
    "Resource": (0.31921, 0.30061, 0.7120, 0.73566),
    "Subscription": (0.56268, 0.47072, 0.8520, 0.62518),
    "SubscriptionManager": (0.47951, 0.47306, 0.8091, 0.55976),
    "global": (0.33307, 0.33307, 0.7266, 0.53095),
}


def test_every_written_context_separates_its_thresholds(tmp_path):
    """The headline invariant, over every context of a real published workflow.

    `tiny_ambiguous > tier` is what gives the tiny tier an ambiguity band at all:
    `ModelPipeline` keeps a prediction on the tiny tier when `confidence >= tier`
    and `CommandRouter` calls it confident when `confidence > tiny_ambiguous`, so an
    ambiguity is reachable only in the open interval between the two.
    """
    written = {}
    for ctx, (tier, tiny_failed, tiny_ok, large_failed) in PUBLISHED_CONTEXTS.items():
        ctx_dir = tmp_path / ctx
        written[ctx] = write_ambiguity_thresholds(
            str(ctx_dir), tier, _stats(tiny_failed, tiny_ok), _stats(large_failed, 0.9)
        )

    assert len(written) == len(PUBLISHED_CONTEXTS)
    for ctx, (tier, *_rest) in PUBLISHED_CONTEXTS.items():
        tiny_amb, large_amb = written[ctx]
        on_disk = {
            name: json.load(open(tmp_path / ctx / name))['confidence_threshold']
            for name in ("tiny_ambiguous_threshold.json", "large_ambiguous_threshold.json")
        }
        assert on_disk["tiny_ambiguous_threshold.json"] == tiny_amb, ctx
        assert on_disk["large_ambiguous_threshold.json"] == large_amb, ctx
        assert tiny_amb > tier, f"{ctx}: tiny ambiguity band is empty"
        assert tiny_amb >= tier + TIER_AMBIGUITY_MIN_SEPARATION, ctx
        assert tiny_amb >= SINGLE_LABEL_RESOLUTION_FLOOR, ctx
        assert large_amb >= SINGLE_LABEL_RESOLUTION_FLOOR, ctx


def test_the_seven_collapsed_contexts_are_the_ones_that_move_most():
    """Regression pin on the defect's own signature.

    These seven published contexts had `threshold.json` and
    `tiny_ambiguous_threshold.json` byte-identical. Equality is the unmistakable
    form of the bug, so the fix has to move exactly these off equality -- and it has
    to move the other eleven too, since `tiny_ambiguous <= tier` collapses the band
    just as completely as equality does.
    """
    collapsed = [
        ctx for ctx, (tier, tiny_failed, *_r) in PUBLISHED_CONTEXTS.items()
        if tier == tiny_failed
    ]
    assert len(collapsed) == 7

    for ctx, (tier, tiny_failed, tiny_ok, _large) in PUBLISHED_CONTEXTS.items():
        assert tiny_failed <= tier, f"{ctx} was not actually collapsed before R3"
        assert separated_tiny_ambiguous_threshold(tier, _stats(tiny_failed, tiny_ok)) > tier


def test_the_margin_is_the_sweep_midpoint_when_that_is_the_binding_term():
    """The preferred margin is the midpoint of the interval the tier sweep ran over."""
    tier, failed_mean, successful_mean = 0.40, 0.40, 0.90
    assert separated_tiny_ambiguous_threshold(
        tier, _stats(failed_mean, successful_mean)
    ) == pytest.approx(0.65)


def test_a_high_tier_threshold_still_gets_a_band():
    """When the sweep lands above the midpoint, the minimum separation carries it."""
    assert separated_tiny_ambiguous_threshold(0.80, _stats(0.40, 0.90)) == pytest.approx(
        0.80 + TIER_AMBIGUITY_MIN_SEPARATION
    )


def test_no_tier_resolves_a_single_label_below_the_floor():
    """The floor is absolute: it binds when both measured terms are under it.

    A top-1 probability at or below 0.5 puts at least as much posterior mass outside
    the winning label as in it, so a single-label resolution is not supported by the
    model's own distribution whichever tier produced it.
    """
    assert separated_tiny_ambiguous_threshold(0.10, _stats(0.10, 0.30)) == pytest.approx(
        SINGLE_LABEL_RESOLUTION_FLOOR
    )
    assert floored_large_ambiguous_threshold(_stats(0.41043, 0.9)) == pytest.approx(
        SINGLE_LABEL_RESOLUTION_FLOOR
    )
    assert floored_large_ambiguous_threshold(_stats(0.71277, 0.9)) == pytest.approx(0.71277)


def test_missing_statistics_fall_back_to_the_floor_not_to_zero():
    """A context with no failures (or no successes) used to be written 0.0, which is
    the most permissive value there is. The floor is the safe direction."""
    assert separated_tiny_ambiguous_threshold(-1, _stats(None, None)) == pytest.approx(
        SINGLE_LABEL_RESOLUTION_FLOOR
    )
    assert floored_large_ambiguous_threshold(_stats(None, None)) == pytest.approx(
        SINGLE_LABEL_RESOLUTION_FLOOR
    )


def test_the_writer_refuses_to_publish_a_collapsed_pair(tmp_path, monkeypatch):
    """The guard is in the writer, so a future change to the margin arithmetic cannot
    republish the defect quietly."""
    monkeypatch.setattr(
        "fastworkflow.model_pipeline_training.separated_tiny_ambiguous_threshold",
        lambda tier, stats: tier,
    )
    with pytest.raises(ValueError, match="never report an ambiguity"):
        write_ambiguity_thresholds(
            str(tmp_path / "Broken"), 0.6, _stats(0.6, 0.9), _stats(0.7, 0.95)
        )
    assert not os.path.exists(tmp_path / "Broken" / "tiny_ambiguous_threshold.json")


# ---------------------------------------------------------------------------
# ido-ik6 (F13): a tier threshold at or above the flat cap must still publish.
#
# `find_optimal_threshold` sweeps `linspace(failed_mean, successful_mean, 20)` and
# picks the best point, so a context whose tiny tier is confidently separated
# legitimately selects a tier threshold at or above `MAX_AMBIGUITY_THRESHOLD`. The
# statistics below are of that shape: a failed mean of 0.6 and a successful mean of
# 0.997 put the top of the sweep at 0.997, so 0.99 and 0.995 are points the sweep
# can actually return. Clamping the ambiguity threshold to a flat 0.99 there made
# the writer raise inside `train()`'s per-context loop and abort training for the
# whole workflow.
# ---------------------------------------------------------------------------
CONFIDENT_TINY = _stats(0.6, 0.997)
CONFIDENT_LARGE = _stats(0.7, 0.99)


@pytest.mark.parametrize("tier", [0.99, 0.995, 0.997, 0.9999])
def test_a_tier_at_or_above_the_flat_cap_publishes_a_separated_pair(tmp_path, tier):
    """The headline regression: no ValueError for a tier the sweep can pick."""
    ctx_dir = tmp_path / f"Confident{tier}"
    tiny_amb, large_amb = write_ambiguity_thresholds(
        str(ctx_dir), tier, CONFIDENT_TINY, CONFIDENT_LARGE
    )

    on_disk = json.load(open(ctx_dir / "tiny_ambiguous_threshold.json"))
    assert on_disk['confidence_threshold'] == tiny_amb
    assert json.load(open(ctx_dir / "large_ambiguous_threshold.json"))[
        'confidence_threshold'
    ] == large_amb

    # Usable, not merely written: strictly above the tier so an ambiguity is
    # reachable, and strictly below certainty so a resolution is reachable too.
    assert tiny_amb > tier
    assert tiny_amb < 1.0
    assert tiny_amb >= SINGLE_LABEL_RESOLUTION_FLOOR
    assert tiny_amb == pytest.approx((tier + 1.0) / 2.0)


def test_the_band_above_the_flat_cap_is_the_remaining_headroom_halved():
    """The ceiling above the cap is the midpoint between the tier and certainty."""
    assert resolvable_ambiguity_ceiling(0.99) == pytest.approx(0.995)
    assert resolvable_ambiguity_ceiling(0.995) == pytest.approx(0.9975)
    assert separated_tiny_ambiguous_threshold(0.99, CONFIDENT_TINY) == pytest.approx(0.995)
    assert separated_tiny_ambiguous_threshold(0.995, CONFIDENT_TINY) == pytest.approx(0.9975)


@pytest.mark.parametrize(
    "tier, expected",
    [
        (0.90, 0.95),                      # tier + minimum separation, cap not binding
        (0.94, MAX_AMBIGUITY_THRESHOLD),   # cap binds, exactly as before
        (0.985, MAX_AMBIGUITY_THRESHOLD),  # cap binds with only 0.005 of separation
        (0.9899, MAX_AMBIGUITY_THRESHOLD),  # the last tier below the boundary
        (-1, 0.7985),  # find_optimal_threshold's sentinel: the sweep midpoint binds
    ],
)
def test_a_tier_below_the_flat_cap_is_unchanged(tmp_path, tier, expected):
    """Every tier under `MAX_AMBIGUITY_THRESHOLD` keeps its unclamped value, so
    the clamp is confined to the range that used to abort training."""
    assert resolvable_ambiguity_ceiling(tier) == MAX_AMBIGUITY_THRESHOLD
    tiny_amb, _large = write_ambiguity_thresholds(
        str(tmp_path / f"Tier{tier}"), tier, CONFIDENT_TINY, CONFIDENT_LARGE
    )
    assert tiny_amb == pytest.approx(expected)


@pytest.mark.parametrize("tier", [1.5, 2.0])
def test_a_genuinely_collapsed_pair_still_refuses_to_publish(tmp_path, tier):
    """A tier threshold above certainty is not something a sweep over softmax
    probabilities can return: nothing can sit above it and still be reachable, so the
    writer must keep refusing rather than publish an unsatisfiable pair."""
    ctx_dir = tmp_path / f"Collapsed{tier}"
    with pytest.raises(ValueError, match="never report an ambiguity"):
        write_ambiguity_thresholds(str(ctx_dir), tier, CONFIDENT_TINY, CONFIDENT_LARGE)
    assert not os.path.exists(ctx_dir / "tiny_ambiguous_threshold.json")


# ---------------------------------------------------------------------------
# A float32-saturated sweep picks a tier of exactly 1.0.
#
# float32 softmax rounds the top probability to 1.0 once the logit gap passes about
# 17, so a context whose correct held-out rows all saturate has a successful mean of
# np.float32(1.0), and that is the last point of the sweep. Refusing that pair raised
# inside `train()`'s per-context loop and discarded the whole workflow's run.
# ---------------------------------------------------------------------------
SATURATED_TINY = _stats(np.float32(0.6), np.mean([np.float32(1.0)] * 50))


@pytest.mark.parametrize("tier", [1.0, np.float32(1.0)])
def test_a_saturated_tier_of_exactly_one_publishes_the_flat_cap(tmp_path, tier):
    """At tier 1.0 the tiny tier keeps only certain predictions, so no band is needed:
    the writer publishes the flat cap instead of an unsatisfiable 1.0 or a refusal."""
    assert SATURATED_TINY['successful']['mean'] == 1.0
    ctx_dir = tmp_path / f"Saturated{type(tier).__name__}"
    tiny_amb, large_amb = write_ambiguity_thresholds(
        str(ctx_dir), tier, SATURATED_TINY, CONFIDENT_LARGE
    )

    assert tiny_amb == MAX_AMBIGUITY_THRESHOLD
    assert json.load(open(ctx_dir / "tiny_ambiguous_threshold.json"))[
        'confidence_threshold'
    ] == MAX_AMBIGUITY_THRESHOLD
    assert large_amb >= SINGLE_LABEL_RESOLUTION_FLOOR
    # A kept prediction (confidence exactly 1.0) still resolves to a single label.
    assert 1.0 > tiny_amb


def test_the_ceiling_at_exactly_one_is_the_flat_cap_not_certainty():
    """`(1 + 1) / 2` would be 1.0, which `confidence > threshold` can never satisfy."""
    assert resolvable_ambiguity_ceiling(1.0) == MAX_AMBIGUITY_THRESHOLD
    assert resolvable_ambiguity_ceiling(np.float32(1.0)) == MAX_AMBIGUITY_THRESHOLD
    assert keeps_only_certain_predictions(1.0)
    assert not keeps_only_certain_predictions(0.9999)
    assert not keeps_only_certain_predictions(None)

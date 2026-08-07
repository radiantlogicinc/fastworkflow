"""Carry-forward safety must depend on cache CONTENT, not just cache MODE (fix-k0i.8).

`compute_training_plan` refused to carry a context forward when the utterance cache
MODE forbade reuse. It never asked whether the cache had anything in it. A deleted
cache directory, a fresh clone, or a cache that was never committed all leave
`mode == reuse` with no entries -- and then every command in a retrained context is
redrawn from the LLM while the carried-forward contexts keep the superseded text in
their wildcard class. That is the internally inconsistent version AR3 says versioning
cannot detect and the implementation must prevent.

These tests drive the shipped helpers against a real cache directory on disk rather
than simulating the lookup, because the defect was precisely that nothing looked.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from fastworkflow.train import utterance_cache
from fastworkflow.train.selective_training import (
    _commands_contributing_to,
    _shared_commands_absent_from_cache,
)

CONTEXT_COMMANDS = {
    "TodoListManager": {"TodoListManager/create", "TodoListManager/delete"},
    "TodoList": {"TodoList/add_item"},
    "TodoItem": {"TodoItem/complete"},
}
# TodoList inherits TodoListManager's commands as its wildcard class; TodoItem
# inherits both. This is the shape that makes one command feed several contexts.
CONTEXT_ANCESTORS = {
    "TodoListManager": set(),
    "TodoList": {"TodoListManager"},
    "TodoItem": {"TodoList", "TodoListManager"},
}


@pytest.fixture
def workflow(tmp_path) -> Path:
    root = tmp_path / "wf"
    (root / "___command_info").mkdir(parents=True)
    return root


def _cache_entry(workflow: Path, command_name: str) -> Path:
    """Write a cache file for *command_name* the way the cache names them."""
    cache_root = workflow / "___command_info" / utterance_cache.CACHE_DIRNAME
    cache_root.mkdir(parents=True, exist_ok=True)
    path = cache_root / f"{utterance_cache.slugify(command_name)}.v1abc.json"
    path.write_text(json.dumps({"entries": {}}), encoding="utf-8")
    return path


def test_ancestor_commands_count_as_contributing():
    """A context's wildcard class is built from its ancestors' utterances.

    If contribution meant only "own commands", the shared set would be empty for
    exactly the inheritance shape the hazard needs, and the check would never fire.
    """
    contributing = _commands_contributing_to(
        "TodoItem", CONTEXT_COMMANDS, CONTEXT_ANCESTORS
    )

    assert "TodoItem/complete" in contributing
    assert "TodoList/add_item" in contributing, "parent's commands are the wildcard class"
    assert "TodoListManager/create" in contributing, "grandparent's too"


def test_a_missing_cache_directory_blocks_carry_forward(workflow: Path):
    """The named scenarios -- deleted cache, fresh clone, never committed."""
    absent = _shared_commands_absent_from_cache(
        str(workflow),
        to_train=["TodoListManager"],
        carried=["TodoList"],
        context_commands=CONTEXT_COMMANDS,
        context_ancestors=CONTEXT_ANCESTORS,
    )

    # TodoListManager's commands feed TodoList's wildcard class, and none is cached.
    assert absent == {"TodoListManager/create", "TodoListManager/delete"}


def test_a_populated_cache_allows_carry_forward(workflow: Path):
    for command in ("TodoListManager/create", "TodoListManager/delete"):
        _cache_entry(workflow, command)

    absent = _shared_commands_absent_from_cache(
        str(workflow),
        to_train=["TodoListManager"],
        carried=["TodoList"],
        context_commands=CONTEXT_COMMANDS,
        context_ancestors=CONTEXT_ANCESTORS,
    )

    assert absent == set(), "a fully cached shared set must not force a full retrain"


def test_a_partially_populated_cache_still_blocks(workflow: Path):
    """The coarse question 'is the cache empty' is not enough.

    One uncached shared command is one command redrawn from the LLM into a
    retrained context while a carried context keeps the old text for it.
    """
    _cache_entry(workflow, "TodoListManager/create")

    absent = _shared_commands_absent_from_cache(
        str(workflow),
        to_train=["TodoListManager"],
        carried=["TodoList"],
        context_commands=CONTEXT_COMMANDS,
        context_ancestors=CONTEXT_ANCESTORS,
    )

    assert absent == {"TodoListManager/delete"}


def test_commands_not_shared_with_a_carried_context_are_ignored(workflow: Path):
    """Only the OVERLAP matters.

    A command that feeds only retrained contexts can be redrawn freely -- nothing
    carried forward embodies its old text -- so forcing a full retrain for it would
    make selective training useless without buying any consistency.
    """
    absent = _shared_commands_absent_from_cache(
        str(workflow),
        to_train=["TodoItem"],
        carried=[],
        context_commands=CONTEXT_COMMANDS,
        context_ancestors=CONTEXT_ANCESTORS,
    )

    assert absent == set()


def test_the_check_matches_on_the_cache_naming_scheme(workflow: Path):
    """Entries are `<slug>.<variant>.json`, and the variant is not knowable here.

    Matching on the stem is what lets this run before generation, when the variant
    key for this run has not been computed yet.
    """
    path = _cache_entry(workflow, "TodoListManager/create")
    assert path.name.startswith(utterance_cache.slugify("TodoListManager/create") + ".")
    assert path.name.endswith(".json")

    absent = _shared_commands_absent_from_cache(
        str(workflow),
        to_train=["TodoListManager"],
        carried=["TodoList"],
        context_commands=CONTEXT_COMMANDS,
        context_ancestors={"TodoList": {"TodoListManager"}},
    )

    assert "TodoListManager/create" not in absent, "a variant file must count as cached"


def test_a_non_json_file_in_the_cache_does_not_count_as_an_entry(workflow: Path):
    cache_root = workflow / "___command_info" / utterance_cache.CACHE_DIRNAME
    cache_root.mkdir(parents=True, exist_ok=True)
    (cache_root / f"{utterance_cache.slugify('TodoListManager/create')}.tmp").write_text("x")

    absent = _shared_commands_absent_from_cache(
        str(workflow),
        to_train=["TodoListManager"],
        carried=["TodoList"],
        context_commands=CONTEXT_COMMANDS,
        context_ancestors={"TodoList": {"TodoListManager"}},
    )

    assert "TodoListManager/create" in absent, "a temp file is not a cached entry"


# ---------------------------------------------------------------------------
# fix-k0i.11: the holdout must be aligned to generation-batch boundaries
# ---------------------------------------------------------------------------


def _batch_records(spec):
    from fastworkflow.train.heldout_evaluation import LabeledUtterance

    return [
        LabeledUtterance(utterance=u, label=label, persona=persona)
        for persona, label, utterances in spec
        for u in utterances
    ]


def test_batch_groups_are_transitive_across_commands():
    """A persona can sit in one batch for command A and another for command B.

    That transitively binds all three groups, which is why this is union-find and
    not a per-batch grouping.
    """
    from fastworkflow.train.heldout_evaluation import persona_batch_groups

    assert persona_batch_groups([["p1", "p2"], ["p2", "p3"], ["p9"]]) == [
        ["p1", "p2", "p3"], ["p9"]
    ]


def test_seed_persona_never_joins_a_batch_group():
    from fastworkflow.train.heldout_evaluation import (
        SEED_PERSONA_ID, persona_batch_groups,
    )

    # p1 survives as a group of one; only the seed marker is dropped.
    assert persona_batch_groups([[SEED_PERSONA_ID, "p1"]]) == [["p1"]]
    assert persona_batch_groups([[SEED_PERSONA_ID]]) == []


def test_a_batch_is_never_split_across_the_holdout_boundary():
    """The AR1 requirement: personas in one batch come from ONE completion.

    Holding out half a batch measures a persona whose utterances were drawn while
    the model could see its batch-mates' training rows.
    """
    from fastworkflow.train.heldout_evaluation import split_by_persona

    records = _batch_records([
        (f"p{i}", "Ctx/cmd", [f"utterance {i} a", f"utterance {i} b"])
        for i in range(8)
    ])
    batches = [["p0", "p1"], ["p2", "p3"], ["p4", "p5"], ["p6", "p7"]]

    split = split_by_persona(records, seed=3, persona_batches=batches)

    heldout = set(split.heldout_personas)
    for batch in batches:
        shared = heldout & set(batch)
        assert shared in (set(), set(batch)), (
            f"batch {batch} was split across the holdout boundary: {shared}"
        )


def test_an_unaligned_split_says_so():
    """Without batch composition the split cannot honour AR1, and the number is
    optimistic. Previously it was reported as a clean holdout with no disclosure."""
    from fastworkflow.train.heldout_evaluation import split_by_persona

    records = _batch_records([
        (f"p{i}", "Ctx/cmd", [f"utterance {i}"]) for i in range(8)
    ])

    split = split_by_persona(records, seed=3)

    assert any("NOT batch-aligned" in note for note in split.notes)
    assert any("optimistic" in note for note in split.notes)


def test_an_aligned_split_does_not_carry_the_disclosure():
    from fastworkflow.train.heldout_evaluation import split_by_persona

    records = _batch_records([
        (f"p{i}", "Ctx/cmd", [f"utterance {i}"]) for i in range(8)
    ])

    split = split_by_persona(
        records, seed=3, persona_batches=[["p0", "p1"], ["p2", "p3"],
                                          ["p4", "p5"], ["p6", "p7"]]
    )

    assert not any("NOT batch-aligned" in note for note in split.notes)


def test_batches_larger_than_the_target_reserve_nothing_and_say_so():
    """Undershooting is the safe direction, but it must not be silent."""
    from fastworkflow.train.heldout_evaluation import split_by_persona

    records = _batch_records([
        (f"p{i}", "Ctx/cmd", [f"utterance {i}"]) for i in range(4)
    ])

    # One batch of 4 against a 25% target of 1 persona: nothing can be reserved.
    split = split_by_persona(
        records, seed=3, persona_batches=[["p0", "p1", "p2", "p3"]]
    )

    assert split.heldout_personas == []
    assert any("smallest generation batch" in note for note in split.notes)


# ---------------------------------------------------------------------------
# fix-k0i.44: the holdout number needs a calibration, not just a caveat
# ---------------------------------------------------------------------------


def _split(train_spec, heldout_spec):
    from fastworkflow.train.heldout_evaluation import LabeledUtterance, PersonaSplit

    def rows(spec, persona):
        return [
            LabeledUtterance(utterance=u, label=label, persona=persona)
            for label, utterances in spec
            for u in utterances
        ]

    return PersonaSplit(
        train=rows(train_spec, "p1"),
        heldout=rows(heldout_spec, "p2"),
        train_personas=["p1"],
        heldout_personas=["p2"],
        notes=[],
    )


def test_a_duplicated_corpus_calibrates_as_leaking():
    """The failure this instrument exists to catch: a holdout that is a copy.

    Every held-out row repeats a training row, so the routing score would be near
    perfect while measuring nothing but recall.
    """
    from fastworkflow.train.heldout_evaluation import calibrate_leak

    shared = [("Ctx/cmd", ["add milk to the list", "remove eggs from the list"])]
    calibration = calibrate_leak(_split(shared, shared))

    assert calibration.median_max_overlap == 1.0
    assert calibration.fraction_above_0_8 == 1.0
    assert calibration.exact_duplicates == 2


def test_a_genuinely_distinct_holdout_calibrates_as_clean():
    from fastworkflow.train.heldout_evaluation import calibrate_leak

    calibration = calibrate_leak(
        _split(
            [("Ctx/cmd", ["add milk to the list"])],
            [("Ctx/cmd", ["please jot down dairy purchases"])],
        )
    )

    assert calibration.median_max_overlap == 0.0
    assert calibration.fraction_above_0_8 == 0.0
    assert calibration.exact_duplicates == 0


def test_overlap_is_measured_within_a_label_not_across_the_corpus():
    """Overlap with a DIFFERENT command's utterances is vocabulary, not leakage.

    Corpus-wide comparison would report a workflow with a consistent domain
    vocabulary as leaking, which would make the instrument useless exactly where
    it is needed.
    """
    from fastworkflow.train.heldout_evaluation import calibrate_leak

    calibration = calibrate_leak(
        _split(
            [("Ctx/other", ["add milk to the list"])],
            [("Ctx/cmd", ["add milk to the list"])],
        )
    )

    assert calibration.median_max_overlap == 0.0, "a different label is not a leak"
    assert calibration.exact_duplicates == 0


def test_an_empty_holdout_reports_nothing_rather_than_zero():
    """0.0 overlap and "not measured" are different facts."""
    from fastworkflow.train.heldout_evaluation import calibrate_leak

    calibration = calibrate_leak(_split([("Ctx/cmd", ["add milk"])], []))

    assert calibration.total == 0
    assert calibration.median_max_overlap is None
    assert calibration.fraction_above_0_8 is None


def test_the_calibration_reaches_the_human_report():
    from fastworkflow.train.heldout_evaluation import (
        HeldoutReport, RoutingScore, calibrate_leak, format_report,
    )

    shared = [("Ctx/cmd", ["add milk to the list"])]
    report = HeldoutReport(
        context="Ctx",
        routing=RoutingScore(total=1, top1_correct=1, top1=1.0),
        leak_calibration=calibrate_leak(_split(shared, shared)),
    )

    rendered = format_report([report])
    assert "Leak calibration" in rendered
    assert "closer to recall" in rendered
    assert "verbatim training rows" in rendered, "exact duplicates are a bug signal"

"""Phase 9 step 1a: the sparse "not here" class, its experiment arms, and the scorer.

Integration tests on a copy of ``fastworkflow/examples/library`` (no mocks, no model
fitting). The sparse construction and the arm selection are tested through the shipped
``select_escalation_rows``; the scorer is tested on hand-built results and on the McNemar
and Wilson arithmetic. Nothing here trains an intent head.

Phase 9 step 5: the ancestor arms (``control``, ``ratio_0_5``) are retired. Their tests are
kept and now assert the refusal; the ancestor arithmetic they compared against is gone.
"""

from __future__ import annotations

import contextlib
import io
import os
import shutil
from pathlib import Path

import pytest

import fastworkflow
from fastworkflow.command_routing import RoutingRegistry
from fastworkflow.model_pipeline_training import (
    NOT_HERE_SEEDS,
    RESERVED_CLASS_ARMS,
    RETIRED_ARMS,
    _get_cached_command_utterances,  # noqa: F401  (documents the cache contract used below)
    cache_context_command_utterances,
    experiment_reserved_class_arm,
    not_here_seed_rows,
    resolve_reserved_class_arm,
    select_escalation_rows,
    trained_command_labels,
)
from fastworkflow.nlu_labels import WILDCARD_LABEL, label_of
from fastworkflow.planning_commands import command_utterances  # noqa: F401
from fastworkflow.train import score_not_here as scorer
from fastworkflow.workflow_agent import planning_command_utterances

LIBRARY = os.path.join("fastworkflow", "examples", "library")
#: A multi-context workflow with inheritance, so every head has a real label space to sample from.
MESSAGING = os.path.join("fastworkflow", "examples", "messaging_app_4")
HEAD = "Book"


@pytest.fixture(scope="module", autouse=True)
def initialized_fastworkflow():
    fastworkflow.init({})
    yield
    RoutingRegistry.clear_registry()


def _private_copy(tmp_path, source: str, name: str) -> str:
    """A private copy without trained artifacts, caches or bytecode: nothing here writes into examples/."""
    destination = tmp_path / name
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("___*", "__pycache__"))
    RoutingRegistry.clear_registry()
    return str(destination)


@pytest.fixture
def library_copy(tmp_path) -> str:
    path = _private_copy(tmp_path, LIBRARY, "library_copy")
    yield path
    RoutingRegistry.clear_registry()


@pytest.fixture
def messaging_copy(tmp_path) -> str:
    path = _private_copy(tmp_path, MESSAGING, "messaging_copy")
    yield path
    RoutingRegistry.clear_registry()


def _seeds_of(cmd_dir, name: str) -> list[str]:
    metadata = cmd_dir.get_utterance_metadata(name)
    return list(metadata.plain_utterances) if metadata else []


def _seed_cache(crd) -> dict[str, list[str]]:
    """Every execution command's declared seeds: the warm cache the trainer reads first."""
    cmd_dir = crd.command_directory
    cache: dict[str, list[str]] = {}
    for name in sorted(set(cmd_dir.get_commands()) | set(cmd_dir.core_command_names)):
        if label_of(name) == WILDCARD_LABEL:
            continue
        cmd_dir.ensure_command_hydrated(name)
        cache[name] = _seeds_of(cmd_dir, name)
    return cache


def _head_inputs(workflow, crd, ctx, cache, command_cache, wildcard_utterances):
    rows_by_command = cache_context_command_utterances(ctx, crd, workflow, cache, command_cache)
    context_utterances: set[str] = set()
    own = 0
    for rows in rows_by_command.values():
        own += len(rows)
        context_utterances |= set(rows)
    return context_utterances, own


def _wildcard_utterances(workflow, crd) -> set[str]:
    from fastworkflow.model_pipeline_training import _get_utterances
    return set(_get_utterances(workflow, workflow.folderpath, crd.command_directory, WILDCARD_LABEL))


# ---------------------------------------------------------------- sparse construction

def test_sparse_construction_takes_at_most_n_seeds_and_includes_planning_commands(library_copy):
    RoutingRegistry.clear_registry()
    crd = RoutingRegistry.get_definition(library_copy)
    labels = trained_command_labels(HEAD, crd)
    planning = {name: tuple(seeds) for name, seeds in planning_command_utterances(library_copy).items()}
    planning_with_seeds = {name for name, seeds in planning.items() if seeds}
    assert planning_with_seeds, "the library example must have planning commands with seeds"

    one, avail_one = not_here_seed_rows(library_copy, crd, labels, 1)
    eight, avail_eight = not_here_seed_rows(library_copy, crd, labels, 8)  # sparse_8; NOT_HERE_SEEDS is now 1

    assert avail_one == avail_eight > 0
    assert all(len(rows) <= 1 for rows in one.values())
    assert all(len(rows) <= 8 for rows in eight.values())
    assert sum(len(r) for r in eight.values()) >= sum(len(r) for r in one.values())

    # planning commands are foreign to every head and are picked like execution commands
    assert planning_with_seeds - labels <= set(eight), "a planning command was left out of the not-here class"
    assert planning_with_seeds - labels <= set(one)

    # every pick is one of that command's own declared seeds, and no head label is picked
    cmd_dir = crd.command_directory
    for name, rows in eight.items():
        assert name not in labels
        pool = set(planning.get(name, ())) or set(_seeds_of(cmd_dir, name))
        assert set(rows) <= pool, name
    # a command with more seeds than N contributes exactly N
    for name, rows in eight.items():
        pool = set(planning.get(name, ())) or set(_seeds_of(cmd_dir, name))
        if len(pool) >= 8:
            assert len(rows) == 8, name


def test_sparse_selection_is_foreign_only_and_never_duplicates_a_local_row(library_copy):
    RoutingRegistry.clear_registry()
    crd = RoutingRegistry.get_definition(library_copy)
    cache = _seed_cache(crd)
    command_cache = dict(cache)
    workflow = fastworkflow.Workflow.create(library_copy, workflow_id_str="not-here-sparse")
    try:
        wildcard = _wildcard_utterances(workflow, crd)
        with contextlib.redirect_stdout(io.StringIO()):
            context_utterances, _own = _head_inputs(workflow, crd, HEAD, cache, command_cache, wildcard)
            selection = select_escalation_rows(
                HEAD, crd, workflow, context_utterances=context_utterances,
                reserved_class_arm="sparse_1",
            )
        assert selection.included and selection.rows
        assert not (set(selection.rows) & context_utterances)
        assert selection.budget is None
        assert selection.always_include_rows == []
    finally:
        workflow.close()


# ---------------------------------------------------------------- arms

def test_default_arm_is_the_shipped_sparse_one_and_the_retired_arms_are_refused(messaging_copy):
    """Phase 9 step 5 (was: the default arm builds today's ancestor rows).

    With no override the head gets the shipped sparse_1 rows, whatever the caller passes
    through the default path. The ancestor arms no longer exist: naming one is refused with
    the reason, both explicitly and through the experiment override.
    """
    RoutingRegistry.clear_registry()
    crd = RoutingRegistry.get_definition(messaging_copy)
    cache = _seed_cache(crd)
    command_cache = dict(cache)
    workflow = fastworkflow.Workflow.create(messaging_copy, workflow_id_str="not-here-default")
    checked = 0
    included = 0
    try:
        wildcard = _wildcard_utterances(workflow, crd)
        for ctx in sorted(crd.contexts):
            with contextlib.redirect_stdout(io.StringIO()):
                context_utterances, _own = _head_inputs(workflow, crd, ctx, cache, command_cache, wildcard)
            if not context_utterances:
                continue
            with contextlib.redirect_stdout(io.StringIO()):
                default = select_escalation_rows(
                    ctx, crd, workflow, context_utterances=context_utterances)
                shipped = select_escalation_rows(
                    ctx, crd, workflow, context_utterances=context_utterances,
                    reserved_class_arm="sparse_1")
            assert default.rows == shipped.rows and default.included == shipped.included, ctx
            included += int(default.included)
            checked += 1
            for retired in sorted(RETIRED_ARMS):
                with pytest.raises(ValueError, match="retired"):
                    select_escalation_rows(
                        ctx, crd, workflow, context_utterances=context_utterances,
                        reserved_class_arm=retired)
    finally:
        workflow.close()
    assert checked > 0
    assert included > 0, "no escalation class was built, so nothing was compared"


def test_retired_ratio_arm_is_refused_with_its_reason_not_as_an_unknown_name(messaging_copy):
    """Phase 9 step 5 (was: the 0.5 ratio arm caps the ancestor class at half the head)."""
    assert set(RETIRED_ARMS) == {"control", "ratio_0_5"}
    assert "ratio_0_5" not in RESERVED_CLASS_ARMS and "control" not in RESERVED_CLASS_ARMS
    with pytest.raises(ValueError, match="retired"):
        resolve_reserved_class_arm("ratio_0_5")
    with pytest.raises(ValueError, match="retired"):
        with experiment_reserved_class_arm("ratio_0_5"):
            pass  # pragma: no cover - refused before the block runs


def test_arm_resolution_is_explicit_then_override_then_shipped_sparse_1():
    # Phase 9 step 4: with no override the arm is the shipped sparse_1, not control.
    # Phase 9 step 5: the only arms left are the two sparse ones.
    assert resolve_reserved_class_arm() == "sparse_1"
    assert resolve_reserved_class_arm("sparse_8") == "sparse_8"
    with experiment_reserved_class_arm("sparse_8"):
        assert resolve_reserved_class_arm() == "sparse_8"
        assert resolve_reserved_class_arm("sparse_1") == "sparse_1"
    assert resolve_reserved_class_arm() == "sparse_1"
    with pytest.raises(ValueError, match="unknown"):
        resolve_reserved_class_arm("sparse_99")
    assert sorted(RESERVED_CLASS_ARMS) == ["sparse_1", "sparse_8"]
    assert RESERVED_CLASS_ARMS["sparse_8"][2] == 8  # pinned: sparse_8 does not follow NOT_HERE_SEEDS
    assert RESERVED_CLASS_ARMS["sparse_1"][2] is None  # the shipped arm follows NOT_HERE_SEEDS


# ---------------------------------------------------------------- scorer

def _case(head, utterance, expected, in_scope, top, topk=None):
    return {"head": head, "utterance": utterance, "expected": expected, "in_scope": in_scope,
            "top_label": top, "topk": topk if topk is not None else [top]}


def _hand_built_results(arm="x"):
    cases = [
        # head A, in scope (4): 3 correct, one wrong decline (wildcard top-1, command in list)
        _case("A", "a1", "A/one", True, "A/one"),
        _case("A", "a2", "A/two", True, "A/two"),
        _case("A", "a3", "A/three", True, "A/four", ["A/four", "A/three"]),
        _case("A", "a4", "A/one", True, "wildcard", ["wildcard", "A/one"]),
        # head A, out of scope (4): 1 wrong accept, one parameter_value (a decline), two wildcard
        _case("A", "o1", "B/x", False, "A/one"),
        _case("A", "o2", "B/y", False, "parameter_value", ["parameter_value"]),
        _case("A", "o3", "B/z", False, "wildcard", ["wildcard"]),
        _case("A", "o4", "C/q", False, "wildcard", ["wildcard"]),
        # head B, in scope (2): both declined
        _case("B", "b1", "B/x", True, "wildcard"),
        _case("B", "b2", "B/y", True, "parameter_value"),
    ]
    return {"schema": 1, "arm": arm, "training_rows": {"A": 30, "B": 12}, "cases": cases}


def test_scorer_counts_both_directions_on_a_hand_built_result_set():
    summary = scorer.summarize(_hand_built_results())
    a = summary["per_head"]["A"]
    b = summary["per_head"]["B"]

    assert (a["in_scope_n"], a["out_of_scope_n"]) == (4, 4)
    assert a["wrong_declines"]["count"] == 1
    assert a["wrong_accepts"]["count"] == 1
    assert a["top1"]["count"] == 2            # a1, a2 (a3's top-1 is A/four)
    assert a["in_list"]["count"] == 4         # a3 holds A/three in its list; a4 holds A/one in its list
    assert a["training_rows"] == 30

    assert (b["in_scope_n"], b["out_of_scope_n"]) == (2, 0)
    assert b["wrong_declines"]["count"] == 2
    assert b["wrong_accepts"]["trials"] == 0 and b["wrong_accepts"]["rate"] is None

    overall = summary["overall"]
    assert overall["wrong_declines"]["count"] == 3 and overall["wrong_declines"]["trials"] == 6
    assert overall["wrong_accepts"]["count"] == 1 and overall["wrong_accepts"]["trials"] == 4
    assert overall["training_rows"] == 42
    low, high = overall["wrong_accepts"]["wilson95"]
    assert low < 0.25 < high


def test_mcnemar_matches_the_known_exact_value():
    # 5 discordant pairs one way, 15 the other: 2 * sum_{k<=5} C(20,k) / 2^20
    assert scorer.mcnemar_exact(5, 15) == pytest.approx(0.0413894653, abs=1e-9)
    assert scorer.mcnemar_exact(0, 0) == 1.0
    assert scorer.mcnemar_exact(3, 3) == 1.0


def test_wilson_interval_matches_the_textbook_value():
    low, high = scorer.wilson_interval(5, 10)
    assert low == pytest.approx(0.2366, abs=1e-4)
    assert high == pytest.approx(0.7634, abs=1e-4)
    assert scorer.wilson_interval(0, 0) is None


def test_compare_pairs_two_arms_on_the_same_cases():
    base = _hand_built_results("control")  # a recorded history label; compare() does not look it up
    other = _hand_built_results("sparse_8")
    # flip one in-scope case from declined to correct in the other arm
    other["cases"][3] = _case("A", "a4", "A/one", True, "A/one")
    comparison = scorer.compare(base, other)
    declines = comparison["paired"]["wrong_decline"]
    assert declines["only_a_wrong"] == 1 and declines["only_b_wrong"] == 0
    # head B's two in-scope declines are in both arms: 3 errors for control (a4, b1, b2), 2 for the other
    assert declines["errors_a"] == 3 and declines["errors_b"] == 2
    assert declines["p_two_sided"] == 1.0  # one discordant pair is never significant
    with pytest.raises(ValueError):
        scorer.compare(base, {"cases": other["cases"][:-1]})

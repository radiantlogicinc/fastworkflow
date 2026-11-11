"""Phase 9 step 3 scorer: positions, scoped paths, expectation, walks, metrics and the bar.

Integration-style: when FW_LOOKUP_TEST_WORKFLOW names a trained workflow folder, its models are read
(read only) by the tests marked needs_real_workflow; the walks run on small prediction caches built by
hand, as plain data. No mocks.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Sequence

import pytest

from fastworkflow.train import score_lookup as s
from fastworkflow.train import not_here_step1 as step1
from fastworkflow.train import score_not_here as scorer

REAL_WORKFLOW = os.path.expanduser(os.environ.get("FW_LOOKUP_TEST_WORKFLOW", ""))
needs_real_workflow = pytest.mark.skipif(
    not (REAL_WORKFLOW and os.path.isfile(os.path.join(REAL_WORKFLOW, "context_hierarchy_model.json"))),
    reason="set FW_LOOKUP_TEST_WORKFLOW to a trained workflow folder to run this test",
)


# ---------------------------------------------------------------- helpers

def label(top, topk=None, confident=None):
    """A head's cache entry. Confident defaults to True when the top-3 is just the top."""
    topk = list(topk) if topk is not None else [top]
    if confident is None:
        confident = True
    return {"top_label": top, "topk": topk, "confident": bool(confident), "confidence": 0.9, "threshold": 0.5,
            "model_tier": "tiny"}


class CountingCache(dict):
    """A cache that counts head reads, so a test can see how often a head was asked."""

    def __init__(self, heads):
        super().__init__(heads)
        self.reads: dict[str, int] = {}

    def __getitem__(self, head):
        self.reads[head] = self.reads.get(head, 0) + 1
        return super().__getitem__(head)


TINY_HIERARCHY = {
    "D": {"parent": ["*"]},
    "B": {"parent": ["D"]},
    "C": {"parent": ["D"]},
    "A": {"parent": ["B", "C"]},           # diamond: A>B>D and A>C>D
    "E": {"parent": ["A", "B", "A"]},      # duplicate parent: its A-paths must appear once
    "F": {"parent": ["*"]},
    "X": {"parent": ["Y"]},                # a cycle X<->Y: the simple paths stop at the repeat
    "Y": {"parent": ["X", "*"]},
}

TINY_INHERITANCE = {
    "Zeta": [],
    "Alpha": ["Zeta"],
    "Beta": ["Zeta"],
    "Gamma": ["Zeta"],
    "Explorer": ["Delta"],
    "Delta": ["Lookup", "Catalog"],
    "Lookup": [],
    "Catalog": [],
    "Deep": ["Left", "Right"],             # depth-first: Left, Left's base, then Right
    "Left": ["LeftBase"],
    "LeftBase": [],
    "Right": [],
}


# ---------------------------------------------------------------- positions

def test_positions_on_tiny_hierarchy_are_deduplicated_and_in_declared_order():
    ids = [p["id"] for p in s.positions(TINY_HIERARCHY)]
    assert ids == [
        "D",
        "B>D",
        "C>D",
        "A>B>D",
        "A>C>D",
        "E>A>B>D",
        "E>A>C>D",
        "E>B>D",
        "F",
        "X>Y",
        "Y",
    ]
    # the cycle: X>Y is the only chain for X; Y>X is skipped because X is already on the chain
    assert "Y>X" not in ids


def test_each_position_records_its_context_chain_and_depth():
    by_id = {p["id"]: p for p in s.positions(TINY_HIERARCHY)}
    assert by_id["E>A>B>D"] == {"id": "E>A>B>D", "context": "E", "chain": ["E", "A", "B", "D"], "depth": 4}
    assert by_id["F"]["chain"] == ["F"]


def test_a_parent_that_has_no_entry_is_an_error():
    with pytest.raises(ValueError, match="no entry"):
        s.positions({"Z": {"parent": ["Missing"]}})


@needs_real_workflow
def test_positions_on_a_real_workflow_follow_the_multi_parent_model():
    hierarchy = s.load_models(REAL_WORKFLOW)["hierarchy"]
    positions = s.positions(hierarchy)
    ids = {p["id"] for p in positions}
    assert len(positions) == 23
    assert len(ids) == len(positions)
    # Account under Identity under Organization, up to DirectoryExplorer (the chain's top)
    assert "Account>Identity>Organization>DirectoryExplorer" in ids
    # Account reaches DirectoryExplorer through each of its four parents
    assert {"Account>DirectoryExplorer", "Account>Identity>DirectoryExplorer",
            "Account>Application>DirectoryExplorer", "Account>Repository>DirectoryExplorer"} <= ids
    # the tops whose parent is '*' end their chains there
    assert "ControlFinding>ControlsMonitor" in ids and "ControlsMonitor" in ids


# ---------------------------------------------------------------- bases and scoped paths

def test_ordered_bases_is_depth_first_in_declared_order_each_once():
    assert s.ordered_bases("Deep", TINY_INHERITANCE) == ["Left", "LeftBase", "Right"]
    assert s.ordered_bases("Unknown", TINY_INHERITANCE) == []


@needs_real_workflow
def test_ordered_bases_matches_command_context_model_on_a_real_workflow():
    from fastworkflow.command_context_model import CommandContextModel

    raw = json.loads(Path(REAL_WORKFLOW, "_commands", "context_inheritance_model.json").read_text())
    inheritance = {name: definition["base"] for name, definition in raw.items()}
    as_model = {name: {"base": bases} for name, bases in inheritance.items()}
    for context in list(inheritance) + ["Directory", "Resource"]:
        assert s.ordered_bases(context, inheritance) == CommandContextModel._ordered_bases_from_map(context, as_model)


def test_scoped_path_binds_each_scope_at_the_first_chain_record_that_reaches_it():
    path = s.scoped_path(["Alpha", "Beta", "Explorer"], TINY_INHERITANCE)
    assert path == [
        ("Alpha", "Alpha"),
        ("Zeta", "Alpha"),          # shared base: bound once, at the first record
        ("Beta", "Beta"),         # Zeta is already visited, so it is skipped here
        ("Explorer", "Explorer"),
        ("Delta", "Explorer"),
        ("Lookup", "Explorer"),
        ("Catalog", "Explorer"),
        ("*", "root"),
    ]


def test_scoped_path_skips_a_base_already_visited_and_ends_with_the_global_scope():
    path = s.scoped_path(["Gamma", "Alpha"], TINY_INHERITANCE)
    scopes = [scope for scope, _ in path]
    assert scopes == ["Gamma", "Zeta", "Alpha", "*"]
    assert len(scopes) == len(set(scopes))
    assert path[-1] == ("*", "root")


def test_scoped_path_for_a_context_with_no_bases_is_itself_then_global():
    assert s.scoped_path(["Right"], TINY_INHERITANCE) == [("Right", "Right"), ("*", "root")]


@needs_real_workflow
def test_real_workflow_scoped_path_binds_each_scope_at_its_first_record():
    models = s.load_models(REAL_WORKFLOW)
    path = s.scoped_path(["Account", "Identity", "Organization", "DirectoryExplorer"], models["inheritance"])
    bindings = dict(path)
    assert bindings["Account"] == "Account"
    assert bindings["Resource"] == "Account"          # Account's base, reached first
    assert bindings["Identity"] == "Identity"
    assert bindings["Organization"] == "Organization"
    assert bindings["DirectoryExplorer"] == "DirectoryExplorer"
    assert bindings["Directory"] == "DirectoryExplorer"
    assert bindings["*"] == "root"
    assert [scope for scope, _ in path].count("Resource") == 1


# ---------------------------------------------------------------- the flattened climb

def test_climb_asks_the_position_then_each_parent_and_global_when_no_root_is_on_the_chain():
    assert s.climb_path(["Alpha", "Beta", "Explorer"]) == [
        ("Alpha", "Alpha"), ("Beta", "Beta"), ("Explorer", "Explorer"), ("*", "root")]


def test_climb_stops_after_asking_the_root_record_and_never_asks_global():
    assert s.climb_path(["Beta", "Gamma", "Explorer"], root_context="Gamma") == [
        ("Beta", "Beta"), ("Gamma", "Gamma")]


def test_climb_of_a_root_position_asks_only_itself():
    assert s.climb_path(["Explorer"], root_context="Explorer") == [("Explorer", "Explorer")]


# ---------------------------------------------------------------- ownership and expectation

def test_owners_map_qualified_names_global_names_and_core():
    owners = s.command_owners(
        ["Alpha/show_owner", "open_directory", "wildcard", "IntentDetection/go_up"],
        core=["IntentDetection/go_up", "wildcard"],
    )
    assert owners == {"Alpha/show_owner": "Alpha", "open_directory": "*", "IntentDetection/go_up": "core"}
    assert s.owner_of("Alpha/show_owner", owners) == "Alpha"
    assert s.owner_of("wildcard", owners) is None
    assert s.owner_of("parameter_value", owners) is None
    assert s.owner_of("Nowhere/x", owners) is None


def test_expected_is_own_base_ancestor_global_and_core_and_off_path():
    chain = ["Alpha", "Beta", "Explorer"]
    position = {"id": "Alpha>Beta>Explorer", "context": "Alpha", "chain": chain, "depth": 3}

    own = s.expected({"owner": "Alpha"}, position, TINY_INHERITANCE)
    assert own == {"in_scope": True, "binding": "Alpha", "kind": "own"}

    base = s.expected({"owner": "Zeta"}, position, TINY_INHERITANCE)
    assert base == {"in_scope": True, "binding": "Alpha", "kind": "base"}

    ancestor = s.expected({"owner": "Beta"}, position, TINY_INHERITANCE)
    assert ancestor == {"in_scope": True, "binding": "Beta", "kind": "ancestor"}

    deep_ancestor = s.expected({"owner": "Delta"}, position, TINY_INHERITANCE)
    assert deep_ancestor == {"in_scope": True, "binding": "Explorer", "kind": "ancestor"}

    global_ = s.expected({"owner": "*"}, position, TINY_INHERITANCE)
    assert global_ == {"in_scope": True, "binding": "root", "kind": "global"}

    core = s.expected({"owner": s.CORE}, position, TINY_INHERITANCE)
    assert core == {"in_scope": True, "binding": "Alpha", "kind": "own"}

    off = s.expected({"owner": "Gamma"}, position, TINY_INHERITANCE)
    assert off == {"in_scope": False, "binding": None, "kind": "off_path"}


def test_expected_refuses_a_case_without_an_owner():
    with pytest.raises(ValueError, match="no owner"):
        s.expected({"label": "x/y"}, {"context": "A", "chain": ["A"]}, {})


# ---------------------------------------------------------------- walks on hand-built caches

CHAIN_POSITION = {"id": "Alpha>Beta>Explorer", "context": "Alpha",
                  "chain": ["Alpha", "Beta", "Explorer"], "depth": 3}


def test_flattened_wildcard_climbs_to_the_parent_that_names_the_command():
    cache = {
        "Alpha": {"u": label("wildcard")},
        "Beta": {"u": label("wildcard")},
        "Explorer": {"u": label("Explorer/open_directory")},
    }
    result = s.walk_flattened(CHAIN_POSITION, "u", cache)
    assert result["named"] == "Explorer/open_directory"
    assert result["bound"] == "Explorer"
    assert result["asked"] == ["Alpha", "Beta", "Explorer"]


def test_flattened_parameter_value_stops_the_climb_with_no_command():
    cache = {
        "Alpha": {"u": label("parameter_value")},
        "Beta": {"u": label("Beta/show_identity")},
        "Explorer": {"u": label("wildcard")},
    }
    result = s.walk_flattened(CHAIN_POSITION, "u", cache)
    assert result["named"] is None
    assert result["asked"] == ["Alpha"]


def test_flattened_global_head_is_asked_after_a_non_root_top_and_binds_root():
    cache = {
        "Alpha": {"u": label("wildcard")},
        "Beta": {"u": label("wildcard")},
        "Explorer": {"u": label("wildcard")},
        "*": {"u": label("open_directory")},
    }
    result = s.walk_flattened(CHAIN_POSITION, "u", cache)
    assert result == {"kind": "named", "named": "open_directory", "bound": "root",
                      "candidates": [], "asked": ["Alpha", "Beta", "Explorer", "*"]}


def test_scoped_walk_binds_a_base_command_at_the_record_that_owns_the_scope():
    cache = {
        "Alpha": {"u": label("wildcard")},
        "Zeta": {"u": label("Zeta/list_items")},
    }
    result = s.walk_scoped(CHAIN_POSITION, "u", cache, TINY_INHERITANCE)
    assert result["named"] == "Zeta/list_items"
    assert result["bound"] == "Alpha"


def test_scoped_walk_parameter_value_stops_and_wildcard_moves_to_the_next_scope():
    cache = {
        "Alpha": {"u": label("wildcard")},
        "Zeta": {"u": label("wildcard")},
        "Beta": {"u": label("parameter_value")},
    }
    result = s.walk_scoped(CHAIN_POSITION, "u", cache, TINY_INHERITANCE)
    assert result["named"] is None
    assert result["asked"] == ["Alpha", "Zeta", "Beta"]


def test_each_head_is_read_at_most_once_per_walk():
    cache = CountingCache({
        head: {"u": label("wildcard")} for head in ["Alpha", "Zeta", "Beta", "Gamma",
                                                      "Explorer", "Delta", "Lookup",
                                                      "Catalog", "*"]
    })
    position = {"id": "x", "context": "Alpha", "chain": ["Alpha", "Gamma", "Explorer"], "depth": 3}
    result = s.walk_scoped(position, "u", cache, TINY_INHERITANCE)
    assert result["named"] is None
    assert result["asked"] == [scope for scope, _ in s.scoped_path(position["chain"], TINY_INHERITANCE)]
    assert set(cache.reads.values()) == {1}


def test_the_flattened_walk_reads_only_the_heads_it_asks():
    cache = CountingCache({"Alpha": {"u": label("Alpha/show_owner")}})
    result = s.walk_flattened(CHAIN_POSITION, "u", cache)
    assert result["named"] == "Alpha/show_owner"
    assert cache.reads == {"Alpha": 1}


def test_a_missing_head_prediction_is_reported_clearly():
    with pytest.raises(KeyError, match="re-run probe"):
        s.walk_flattened(CHAIN_POSITION, "u", {})


# ---------------------------------------------------------------- secondary (confidence-aware) walks

def test_scoped_unsure_head_with_not_here_keeps_candidates_and_the_next_naming_head_ends_the_walk():
    cache = {
        "Alpha": {"u": label("wildcard", topk=["wildcard", "Alpha/show_owner", "Alpha/list_permissions"],
                               confident=False)},
        "Zeta": {"u": label("Zeta/list_items")},
    }
    result = s.walk_scoped_confident(CHAIN_POSITION, "u", cache, TINY_INHERITANCE)
    assert result["kind"] == "asked"
    assert result["candidates"] == ["Alpha/show_owner", "Alpha/list_permissions", "Zeta/list_items"]
    assert s.secondary_outcome(True, "Zeta/list_items", "Alpha", result) == "correct_in_candidates"


def test_scoped_unsure_head_without_not_here_ends_the_walk_among_its_labels():
    cache = {
        "Alpha": {"u": label("Alpha/show_owner", topk=["Alpha/show_owner", "Alpha/list_permissions", "Alpha/x"],
                               confident=False)},
        "Zeta": {"u": label("Zeta/list_items")},
    }
    result = s.walk_scoped_confident(CHAIN_POSITION, "u", cache, TINY_INHERITANCE)
    assert result["kind"] == "asked"
    assert result["asked"] == ["Alpha"]
    assert result["candidates"] == ["Alpha/show_owner", "Alpha/list_permissions", "Alpha/x"]


def test_flattened_suppresses_the_not_here_signal_so_an_unsure_head_always_ends_the_walk():
    cache = {
        "Alpha": {"u": label("wildcard", topk=["wildcard", "Alpha/show_owner"], confident=False)},
        "Beta": {"u": label("Beta/show_identity")},
    }
    result = s.walk_flattened_confident(CHAIN_POSITION, "u", cache)
    assert result["kind"] == "asked"
    assert result["asked"] == ["Alpha"]
    assert result["candidates"] == ["Alpha/show_owner"]


def test_confident_not_here_moves_on_and_a_confident_command_is_named_with_its_binding():
    cache = {
        "Alpha": {"u": label("wildcard")},
        "Beta": {"u": label("Beta/show_identity")},
    }
    result = s.walk_flattened_confident(CHAIN_POSITION, "u", cache)
    assert result["kind"] == "named"
    assert result["bound"] == "Beta"


def test_candidates_end_at_the_path_when_nothing_names_a_command():
    cache = {
        "Alpha": {"u": label("wildcard", topk=["wildcard", "Alpha/show_owner"], confident=False)},
        "Zeta": {"u": label("wildcard", topk=["wildcard", "Zeta/list_items"], confident=False)},
        "*": {"u": label("wildcard")},
    }
    position = {"id": "a", "context": "Alpha", "chain": ["Alpha"], "depth": 1}
    result = s.walk_scoped_confident(position, "u", cache, TINY_INHERITANCE)
    assert result["kind"] == "asked"
    assert result["candidates"] == ["Alpha/show_owner", "Zeta/list_items"]


def test_secondary_outcome_categories():
    named_ok = {"kind": "named", "named": "A/x", "bound": "A", "candidates": [], "asked": []}
    named_wrong_binding = {**named_ok, "bound": "B"}
    none = {"kind": "none", "named": None, "bound": None, "candidates": [], "asked": []}
    asked = {"kind": "asked", "named": None, "bound": None, "candidates": ["A/x"], "asked": []}
    assert s.secondary_outcome(True, "A/x", "A", named_ok) == "correct_confident"
    assert s.secondary_outcome(True, "A/x", "A", named_wrong_binding) == "wrong"
    assert s.secondary_outcome(True, "A/x", "A", none) == "declined"
    assert s.secondary_outcome(True, "A/x", "A", asked) == "correct_in_candidates"
    assert s.secondary_outcome(False, "A/x", None, named_ok) == "wrong"
    assert s.secondary_outcome(False, "A/x", None, asked) == "declined"


def test_primary_outcome_categories():
    named = {"kind": "named", "named": "A/x", "bound": "A", "candidates": [], "asked": []}
    assert s.primary_outcome(True, "A/x", "A", named) == "correct"
    assert s.primary_outcome(True, "A/x", "B", named) == "wrong_binding"
    assert s.primary_outcome(True, "A/y", "A", named) == "wrong_command"
    assert s.primary_outcome(False, "A/x", None, named) == "wrong_accept"
    assert s.primary_outcome(True, "A/x", "A", {"kind": "none"}) == "declined"


# ---------------------------------------------------------------- evaluate_case

def test_evaluate_case_records_the_expected_binding_and_both_walks():
    cache = {
        "Alpha": {"u": label("wildcard", topk=["wildcard", "Alpha/show_owner"], confident=False)},
        "Zeta": {"u": label("Zeta/list_items")},
    }
    case = {"utterance": "u", "label": "Zeta/list_items", "owner": "Zeta"}
    row = s.evaluate_case(case, CHAIN_POSITION, "scoped", cache, TINY_INHERITANCE)
    assert row["in_scope"] is True
    assert row["expected_binding"] == "Alpha"
    assert row["owner_kind"] == "base"
    assert row["outcome"] == "correct"
    assert row["correct"] and row["top1"] and not row["wrong_accept"] and not row["wrong_decline"]
    assert row["secondary"]["outcome"] == "correct_in_candidates"


def test_evaluate_case_rejects_an_unknown_layout():
    with pytest.raises(ValueError, match="unknown layout"):
        s.evaluate_case({"utterance": "u", "label": "x", "owner": "*"}, CHAIN_POSITION, "nope", {}, {})


# ---------------------------------------------------------------- statistics and pairing

def _step1_case(in_scope, declined):
    top = "wildcard" if declined else "Ctx/cmd"
    return {"head": "H", "utterance": "u", "in_scope": in_scope, "expected": "Ctx/cmd", "top_label": top,
            "topk": [top]}


def test_paired_wald_matches_the_step1_helper_on_the_same_discordant_pairs():
    arm = [_step1_case(True, True), _step1_case(True, False), _step1_case(True, False), _step1_case(True, True),
           _step1_case(True, False)]
    control = [_step1_case(True, False), _step1_case(True, False), _step1_case(True, True),
               _step1_case(True, False), _step1_case(True, False)]
    expected_block = step1.paired_difference(arm, control, "wrong_declines")
    mine = s.paired_wald([(int(bool(scorer.case_outcome(a)["wrong_decline"])),
                           int(bool(scorer.case_outcome(c)["wrong_decline"])))
                          for a, c in zip(arm, control)])
    assert mine["difference"] == pytest.approx(expected_block["difference"])
    assert mine["lower"] == pytest.approx(expected_block["lower"])
    assert mine["upper"] == pytest.approx(expected_block["upper"])
    assert mine["n"] == expected_block["n"]


def test_paired_wald_counts_and_mcnemar():
    block = s.paired_wald([(1, 0), (1, 0), (0, 1), (1, 1), (0, 0)])
    assert block["n"] == 5
    assert block["arm_only"] == 2 and block["base_only"] == 1
    assert block["difference"] == pytest.approx(1 / 5)
    assert block["p_two_sided"] == pytest.approx(scorer.mcnemar_exact(2, 1))


def test_paired_wald_with_no_pairs_gives_no_numbers():
    block = s.paired_wald([])
    assert block["difference"] is None and block["p_two_sided"] is None


def _case(in_scope, correct=False, top1=None, wrong_accept=False, wrong_decline=None, depth=1, kind="own",
          utterance="u", label="A/x", position="A", binding="A"):
    return {"utterance": utterance, "label": label, "position": position, "in_scope": in_scope,
            "correct": bool(correct), "top1": bool(correct if top1 is None else top1),
            "wrong_accept": wrong_accept, "wrong_decline": bool(not correct) if wrong_decline is None and in_scope
            else bool(wrong_decline), "depth": depth, "owner_kind": kind, "expected_binding": binding,
            "secondary": {"outcome": "declined"}}


def test_compare_refuses_layouts_that_score_different_cases():
    base = [_case(True, correct=True, utterance="a"), _case(True, utterance="b")]
    arm = [_case(True, correct=True, utterance="a"), _case(True, utterance="c")]
    with pytest.raises(ValueError, match="differs between the layouts"):
        s.compare_cases(base, arm)
    with pytest.raises(ValueError, match="different numbers"):
        s.compare_cases(base, arm[:1])


def test_pool_seeds_concatenates_in_seed_order_and_refuses_a_reordered_seed():
    seed1 = {"cases": [_case(True, utterance="a"), _case(False, utterance="b")]}
    seed2 = {"cases": [_case(True, utterance="a"), _case(False, utterance="b")]}
    pooled = s.pool_seeds([seed1, seed2])
    assert len(pooled) == 4
    assert [c["utterance"] for c in pooled] == ["a", "b", "a", "b"]
    reordered = {"cases": [_case(False, utterance="b"), _case(True, utterance="a")]}
    with pytest.raises(ValueError, match="different cases"):
        s.pool_seeds([seed1, reordered])


def test_evaluate_layout_applies_the_three_bars_with_the_owner_thresholds():
    def block(lower=None, upper=None, difference=0.0):
        return {"lower": lower, "upper": upper, "difference": difference}

    passing = s.evaluate_layout({
        "e2e": block(lower=-0.019, upper=0.05),
        "top1": block(lower=-0.009, upper=0.05),
        "wrong_accepts": block(lower=-0.05, upper=0.02),
        "wrong_declines": block(lower=-0.1, upper=0.1),
    })
    assert passing["passes"] is True
    assert all(bar["passes"] for bar in passing["bars"].values())

    e2e_fails = s.evaluate_layout({
        "e2e": block(lower=-0.021, upper=0.05),
        "top1": block(lower=0.0, upper=0.05),
        "wrong_accepts": block(lower=-0.05, upper=0.0),
        "wrong_declines": block(lower=0.0, upper=0.0),
    })
    assert e2e_fails["passes"] is False
    assert e2e_fails["bars"]["e2e"]["passes"] is False

    accept_fails = s.evaluate_layout({
        "e2e": block(lower=0.0, upper=0.05),
        "top1": block(lower=0.0, upper=0.05),
        "wrong_accepts": block(lower=0.0, upper=0.021),
        "wrong_declines": block(lower=0.0, upper=0.0),
    })
    assert accept_fails["bars"]["wrong_accepts"]["passes"] is False


def test_evaluate_layout_fails_on_an_empty_difference():
    empty = {"lower": None, "upper": None, "difference": None}
    verdict = s.evaluate_layout({"e2e": empty, "top1": empty, "wrong_accepts": empty, "wrong_declines": empty})
    assert verdict["passes"] is False


def test_breakdowns_group_by_depth_and_owner_kind():
    base = [_case(True, correct=False, depth=1, kind="own"), _case(True, correct=False, depth=2, kind="ancestor",
                                                                  utterance="v")]
    arm = [_case(True, correct=True, depth=1, kind="own"), _case(True, correct=True, depth=2, kind="ancestor",
                                                               utterance="v")]
    out = s.breakdowns(base, arm)
    assert set(out["by_depth"]) == {"1", "2"}
    assert out["by_depth"]["1"]["e2e"]["difference"] == pytest.approx(1.0)
    assert set(out["by_owner_kind"]) == {"own", "ancestor"}


def test_confident_flag_uses_the_tier_threshold_that_answered():
    assert s.confident_flag(0.8, False, 0.7, 0.5) == (True, 0.7)
    assert s.confident_flag(0.6, True, 0.7, 0.5) == (True, 0.5)
    assert s.confident_flag(0.6, False, 0.7, 0.5) == (False, 0.7)


# ---------------------------------------------------------------- score and compare, end to end on files

def _write_workflow(root: Path, head_scope: str = "scoped") -> None:
    (root / "___command_info").mkdir(parents=True)
    (root / "_commands").mkdir()
    (root / "context_hierarchy_model.json").write_text(json.dumps({
        "Explorer": {"parent": ["*"]},
        "Alpha": {"parent": ["Explorer"]},
    }))
    (root / "_commands" / "context_inheritance_model.json").write_text(json.dumps({
        "Alpha": {"base": ["Zeta"]},
        "Zeta": {"base": []},
    }))
    snapshot = {"map_command_2_metadata": {
        "Alpha/show_owner": {"is_planning": False},
        "Zeta/list_items": {"is_planning": False},
        "Explorer/open_directory": {"is_planning": False},
        "plan_a": {"is_planning": True},
        "wildcard": {"is_planning": False},
        "IntentDetection/go_up": {"is_planning": False},
    }, "core_command_names": ["IntentDetection/go_up", "wildcard"]}
    (root / "___command_info" / "command_directory.json").write_text(json.dumps(snapshot))
    holdout = {"schema": 1, "head_scope": head_scope, "arm": "sparse_1", "seed": 1,
               "heldout": [
                   {"utterance": "owner of acme", "label": "Alpha/show_owner", "persona": "p"},
                   {"utterance": "list items", "label": "Zeta/list_items", "persona": "p"},
                   {"utterance": "open dir", "label": "Explorer/open_directory", "persona": "p"},
                   {"utterance": "a plan", "label": "plan_a", "persona": "p"},
               ],
               "heldout_personas": ["p"], "notes": []}
    (root / "___command_info" / "lookup_holdout.json").write_text(json.dumps(holdout))


def _probe_for(heads):
    return {"schema": 1, "workflow_version": "v1", "heads": heads, "training_rows_per_head": {"Alpha": 10},
            "training_seconds_per_head": None}


def test_score_layout_walks_every_case_at_every_position_and_scores_it(tmp_path):
    _write_workflow(tmp_path)
    # Each head answers the same way for every utterance: Alpha and Zeta know their own commands,
    # Explorer knows its own, the global head answers "not here" for everything.
    heads = {
        "Alpha": {u: label("wildcard") for u in ["owner of acme", "list items", "open dir", "a plan"]},
        "Zeta": {u: label("wildcard") for u in ["owner of acme", "list items", "open dir", "a plan"]},
        "Explorer": {u: label("wildcard") for u in ["owner of acme", "list items", "open dir", "a plan"]},
        "*": {u: label("wildcard") for u in ["owner of acme", "list items", "open dir", "a plan"]},
    }
    heads["Alpha"]["owner of acme"] = label("Alpha/show_owner")
    heads["Zeta"]["list items"] = label("Zeta/list_items")
    heads["Explorer"]["open dir"] = label("Explorer/open_directory")
    heads["*"]["open dir"] = label("wildcard")
    report = s.score_layout(str(tmp_path), _probe_for(heads), "scoped")
    positions_count = report["positions"]
    assert positions_count == 2  # Alpha>Explorer and Explorer
    assert report["skipped_labels"] == ["plan_a"]
    assert len(report["cases"]) == 3 * positions_count
    by_key = {(c["label"], c["position"]): c for c in report["cases"]}
    owner_at_account = by_key[("Alpha/show_owner", "Alpha>Explorer")]
    assert owner_at_account["outcome"] == "correct"
    assert owner_at_account["owner_kind"] == "own"
    list_items = by_key[("Zeta/list_items", "Alpha>Explorer")]
    assert list_items["expected_binding"] == "Alpha"
    assert list_items["bound"] == "Alpha" and list_items["correct"] is True
    global_pair = by_key[("Explorer/open_directory", "Alpha>Explorer")]
    assert global_pair["in_scope"] is True  # Explorer is on the scoped path
    assert global_pair["correct"] is True and global_pair["bound"] == "Explorer"
    # At Alpha>Explorer all three commands are on the path. At Explorer alone only
    # open_directory is: Alpha's and Zeta's commands are off that path.
    assert report["summary"]["in_scope_n"] == 3 + 1
    assert report["summary"]["out_of_scope_n"] == 2
    assert by_key[("Alpha/show_owner", "Explorer")]["in_scope"] is False


def test_score_refuses_a_copy_trained_under_another_layout(tmp_path):
    _write_workflow(tmp_path)
    with pytest.raises(ValueError, match="was trained as layout 'scoped'"):
        s.score_layout(str(tmp_path), _probe_for({}), "flattened")


def _all_wildcard_heads(scoped_heads_for_resource: bool = True) -> dict:
    utterances = ["owner of acme", "list items", "open dir", "a plan"]
    heads = {head: {u: label("wildcard") for u in utterances} for head in ["Alpha", "Zeta", "Explorer", "*"]}
    heads["Alpha"]["owner of acme"] = label("Alpha/show_owner")
    heads["Zeta"]["list items"] = label("Zeta/list_items")
    heads["Explorer"]["open dir"] = label("Explorer/open_directory")
    return heads


def test_compare_pools_the_seeds_pairs_the_layouts_and_reports_the_bar(tmp_path):
    flattened_dir, scoped_dir = tmp_path / "flat", tmp_path / "scoped"
    _write_workflow(flattened_dir, head_scope="flattened")
    _write_workflow(scoped_dir, head_scope="scoped")
    # Each layout has its own trained heads. In the flattened copy the Alpha head also learns its
    # base's commands, so it names Zeta/list_items itself; in the scoped copy it says "not here".
    flattened_heads = _all_wildcard_heads()
    flattened_heads["Alpha"]["list items"] = label("Zeta/list_items")
    scoped_heads = _all_wildcard_heads()
    flattened = s.score_layout(str(flattened_dir), _probe_for(flattened_heads), "flattened")
    scoped = s.score_layout(str(scoped_dir), _probe_for(scoped_heads), "scoped")
    report = s.compare_runs([flattened, dict(flattened)], [scoped, dict(scoped)])
    assert report["pooled_cases"] == 2 * len(scoped["cases"])
    assert set(report["paired"]) == {"e2e", "top1", "wrong_accepts", "wrong_declines"}
    assert report["paired"]["e2e"]["difference"] == pytest.approx(0.0)
    assert report["verdict"]["passes"] is True
    text = s.format_markdown(report)
    assert "Verdict: **PASSES**" in text
    assert "## Breakdown by chain depth" in text


def test_compare_runs_refuses_a_run_of_the_wrong_layout(tmp_path):
    _write_workflow(tmp_path, head_scope="scoped")
    scoped = s.score_layout(str(tmp_path), _probe_for(_all_wildcard_heads()), "scoped")
    with pytest.raises(ValueError, match="expected layout"):
        s.compare_runs([scoped], [scoped])


@needs_real_workflow
def test_real_workflow_owners_match_the_filesystem_command_folders():
    if not os.path.isdir(os.path.join(REAL_WORKFLOW, "_commands")):
        pytest.skip("FW_LOOKUP_TEST_WORKFLOW does not name a workflow")
    models = s.load_models(REAL_WORKFLOW)
    fs = set()
    for folder in os.listdir(os.path.join(REAL_WORKFLOW, "_commands")):
        path = os.path.join(REAL_WORKFLOW, "_commands", folder)
        if os.path.isdir(path) and not folder.startswith(("_", ".")):
            for name in os.listdir(path):
                if name.endswith(".py") and not name.startswith("_"):
                    fs.add(f"{folder}/{name[:-3]}")
    owned = {name for name, owner in models["owners"].items() if owner not in (s.CORE,) and "/" in name}
    assert fs <= owned
    assert models["owners"]["open_directory"] == "*"
    assert models["owners"]["Account/show_account_owner"] == "Account"


# ---------------------------------------------------------------- heads on disk (regression)

def _make_head(folder: Path) -> None:
    """A head folder as the trainer leaves it: the models are DIRECTORIES (save_pretrained)."""
    folder.mkdir(parents=True)
    (folder / "tinymodel.pth").mkdir()
    (folder / "largemodel.pth").mkdir()
    (folder / "label_encoder.pkl").write_bytes(b"not loaded by these tests")
    (folder / "threshold.json").write_text('{"confidence_threshold": 0.6}')


def _current_version(root: Path, heads: Sequence[str]) -> Path:
    """Write a version folder with the given heads and point ___command_info at it, as the trainer does."""
    from fastworkflow.train.artifact_versioning import CURRENT_POINTER_FILENAME, VERSIONS_DIRNAME

    info = root / "___command_info"
    info.mkdir(parents=True, exist_ok=True)
    version = info / VERSIONS_DIRNAME / "20261101T000000Z-abc123"
    version.mkdir(parents=True)
    for head in heads:
        _make_head(version / head)
    (info / CURRENT_POINTER_FILENAME).write_text(json.dumps({"version_id": version.name}))
    return version


def test_discover_heads_treats_directory_models_as_heads_and_maps_global(tmp_path):
    version = tmp_path / "version"
    _make_head(version / "Alpha")
    _make_head(version / "global")
    broken = version / "Broken"          # no threshold.json: not a head
    broken.mkdir()
    (broken / "tinymodel.pth").mkdir()
    (version / "notes.txt").write_text("not a folder")
    found = s.discover_heads(str(version))
    assert found == {"Alpha": str(version / "Alpha"), "*": str(version / "global")}


def test_probe_with_zero_heads_fails_loudly(tmp_path):
    _write_workflow(tmp_path)
    _current_version(tmp_path, heads=[])
    with pytest.raises(SystemExit, match="found 0 trained heads"):
        s.probe_workflow(str(tmp_path), ["owner of acme"])


def test_probe_names_the_missing_heads_the_layout_needs(tmp_path):
    _write_workflow(tmp_path, head_scope="scoped")
    _current_version(tmp_path, heads=["Alpha"])
    with pytest.raises(SystemExit) as error:
        s.probe_workflow(str(tmp_path), ["owner of acme"], layout="scoped")
    message = str(error.value)
    assert "lacks 3 head(s)" in message
    assert "Explorer" in message and "Zeta" in message and "*" in message


def test_score_refuses_before_walking_when_a_walk_head_is_missing(tmp_path):
    _write_workflow(tmp_path, head_scope="scoped")
    probe = _probe_for({"Alpha": {u: label("wildcard") for u in ["owner of acme", "list items", "open dir", "a plan"]}})
    with pytest.raises(s.MissingHeadsError) as error:
        s.score_layout(str(tmp_path), probe, "scoped")
    assert "Zeta" in str(error.value) and "Explorer" in str(error.value)
    assert "*" in str(error.value)


def test_required_heads_for_each_layout_on_the_tiny_model(tmp_path):
    _write_workflow(tmp_path, head_scope="scoped")
    models = s.load_models(str(tmp_path))
    assert s.required_heads(models, "scoped") == ["*", "Alpha", "Explorer", "Zeta"]
    assert s.required_heads(models, "flattened") == ["*", "Alpha", "Explorer"]


# ---------------------------------------------------------------- flattened_global_first

GLOBAL_FIRST_CHAIN = {"id": "Alpha>Beta>Explorer", "context": "Alpha",
                      "chain": ["Alpha", "Beta", "Explorer"], "depth": 3}


def test_global_first_names_a_confident_global_command_bound_to_root():
    cache = {
        "Alpha": {"u": label("Alpha/show_owner")},
        "*": {"u": label("open_directory", confident=True)},
    }
    result = s.walk_flattened_global_first(GLOBAL_FIRST_CHAIN, "u", cache)
    assert result == {"kind": "named", "named": "open_directory", "bound": "root",
                      "candidates": [], "asked": ["*"]}


def test_global_first_unconfident_global_command_falls_through_to_the_climb():
    cache = {
        "Alpha": {"u": label("wildcard")},
        "Beta": {"u": label("Beta/show_identity")},
        "*": {"u": label("open_directory", confident=False)},
    }
    result = s.walk_flattened_global_first(GLOBAL_FIRST_CHAIN, "u", cache)
    assert result == s.walk_flattened(GLOBAL_FIRST_CHAIN, "u", cache) | {"asked": ["*", "Alpha", "Beta"]}
    assert result["named"] == "Beta/show_identity"
    assert result["bound"] == "Beta"


def test_global_first_non_routable_global_top_falls_through(tmp_path):
    # A confident "not here" or a confident bare value on "*" never names anything first.
    for top in ("wildcard", "parameter_value"):
        cache = {
            "Alpha": {"u": label("Alpha/show_owner")},
            "*": {"u": label(top, confident=True)},
        }
        result = s.walk_flattened_global_first(GLOBAL_FIRST_CHAIN, "u", cache)
        assert result["named"] == "Alpha/show_owner", top
        assert result["bound"] == "Alpha", top
        assert result["asked"] == ["*", "Alpha"], top


def test_global_first_asks_the_global_head_once_and_first():
    # Every climb head says "not here" and "*" is unconfident: the climb's own last ask of "*" names it,
    # bound to root, and "*" is listed once, at the front.
    cache = CountingCache({
        "Alpha": {"u": label("wildcard")},
        "Beta": {"u": label("wildcard")},
        "Explorer": {"u": label("wildcard")},
        "*": {"u": label("open_directory", confident=False)},
    })
    result = s.walk_flattened_global_first(GLOBAL_FIRST_CHAIN, "u", cache)
    assert result["named"] == "open_directory"
    assert result["bound"] == "root"
    assert result["asked"] == ["*", "Alpha", "Beta", "Explorer"]
    assert cache.reads["*"] == 2  # the first ask, then the climb's own last ask, as the flattened walk does


def test_global_first_secondary_uses_the_same_first_ask():
    cache = {
        "Alpha": {"u": label("Alpha/show_owner", confident=False)},
        "*": {"u": label("open_directory", confident=True)},
    }
    secondary = s.walk_flattened_global_first_confident(GLOBAL_FIRST_CHAIN, "u", cache)
    assert secondary["named"] == "open_directory"
    assert secondary["bound"] == "root"
    unsure = {"Alpha": {"u": label("Alpha/show_owner", confident=False)},
              "*": {"u": label("open_directory", confident=False)}}
    falls = s.walk_flattened_global_first_confident(GLOBAL_FIRST_CHAIN, "u", unsure)
    assert falls["kind"] == "asked"
    assert falls["asked"] == ["*", "Alpha"]


def test_rescore_rebuilds_the_cases_from_a_score_file_and_walks_the_new_layout():
    cache = {
        "Alpha": {"u": label("wildcard")},
        "Beta": {"u": label("wildcard")},
        "Explorer": {"u": label("wildcard")},
        "*": {"u": label("open_directory", confident=True)},
    }
    positions = [GLOBAL_FIRST_CHAIN, {"id": "Alpha", "context": "Alpha", "chain": ["Alpha"], "depth": 1}]
    templates = [{"utterance": "u", "label": "open_directory", "owner": "*"},
                 {"utterance": "u", "label": "Alpha/show_owner", "owner": "Alpha"}]
    inheritance = {}
    prior_cases = [s.evaluate_case(template, position, "flattened", cache, inheritance)
                   for template in templates for position in positions]
    prior = {"layout": "flattened", "root_context": None, "arm": "sparse_1", "seed": 7,
             "workflow_version": "v1", "cases": prior_cases, "skipped_labels": [],
             "training_rows_per_head": {}, "training_seconds_per_head": None}
    probe = {"heads": cache, "workflow_version": "v1"}

    flattened = s.rescore_from_results(prior, probe, "flattened")
    assert [c["named"] for c in flattened["cases"]] == [c["named"] for c in prior_cases]
    for rebuilt, stored in zip(flattened["cases"], prior_cases):
        assert {k: rebuilt[k] for k in stored} == stored

    first_global = s.rescore_from_results(prior, probe, s.GLOBAL_FIRST_LAYOUT)
    assert first_global["layout"] == s.GLOBAL_FIRST_LAYOUT
    assert first_global["head_scope"] == "flattened"
    assert len(first_global["cases"]) == len(prior_cases)
    global_case = first_global["cases"][0]  # owner "*", at the first position: named by the first ask
    assert (global_case["named"], global_case["bound"], global_case["asked"]) == ("open_directory", "root", ["*"])
    assert global_case["outcome"] == "correct"

    report = s.compare_runs([flattened], [first_global], s.GLOBAL_FIRST_LAYOUT)
    assert report["arm_layout"] == s.GLOBAL_FIRST_LAYOUT
    assert "flattened_global_first" in report["summary"]
    assert "Verdict:" in s.format_markdown(report)
    assert "flattened_global_first" in s.format_markdown(report)


def test_rescore_refuses_a_file_whose_cases_are_not_whole_position_blocks():
    case = {"utterance": "u", "label": "x/y", "owner": "x", "position": "Alpha", "depth": 1,
            "in_scope": True, "expected_binding": "Alpha", "owner_kind": "own"}
    other = {**case, "position": "Beta>Alpha", "depth": 2}
    prior = {"root_context": None, "cases": [case, other, case]}  # two positions, then a third case: not whole blocks
    with pytest.raises(ValueError, match="do not divide"):
        s.rescore_from_results(prior, {"heads": {}}, "flattened")


def test_reweighting_gives_the_global_cases_their_share_and_leaves_the_rest_alone():
    def row(utterance, owner_kind, base_correct, arm_correct):
        common = {"utterance": utterance, "label": "L", "position": "P", "depth": 1, "in_scope": True,
                  "owner_kind": owner_kind, "owner": "*" if owner_kind == "global" else "A"}
        return ({**common, "correct": base_correct, "top1": base_correct},
                {**common, "correct": arm_correct, "top1": arm_correct})

    pairs = [row("g", "global", 0, 1), row("a", "own", 1, 1), row("b", "own", 1, 1), row("c", "base", 1, 1)]
    base = [b for b, _ in pairs]
    arm = [a for _, a in pairs]
    result = s.reweighted_in_scope(base, arm)
    assert result["group_share_before"] == pytest.approx(0.25)
    assert result["group_share"] == pytest.approx(0.05)
    # arm = 0.05 * 1 + 0.95 * 1 = 1.0; base = 0.05 * 0 + 0.95 * 1 = 0.95
    assert result["e2e"]["difference"] == pytest.approx(0.05)
    assert result["e2e"]["arm_rate"] == pytest.approx(1.0)

"""Phase 10 step 1 tooling: the named planning test set, the labelled-heads switch, the live arms'
refusal and estimate, stratified sampling, and the paired report math.

Integration tests on a private copy of ``fastworkflow/examples/library`` (no mocks). Nothing here
trains a model or calls a provider: the live arms are exercised only up to their refusal and their
dry-run estimate, and the label-set and report functions are called directly on real routing data
or on hand-built results.
"""

from __future__ import annotations

import difflib
import json
import os
import shutil

import pytest

import fastworkflow
from fastworkflow.command_routing import RoutingRegistry
from fastworkflow.model_pipeline_training import (
    PLANNING_HEADS_LABELLED,
    PLANNING_HEADS_NOT_HERE,
    active_experiment_planning_heads,
    active_planning_heads,
    construction_records,
    experiment_planning_heads,
    experiment_reserved_class_arm,
    planning_command_names_of,
    planning_head_labels,
    trained_command_labels,
)
from fastworkflow.planning_commands import resolve_field_sources, visible_planning_commands
from fastworkflow.train import planning_test_set as pts
from fastworkflow.train import score_planning_match as spm

LIBRARY = os.path.join("fastworkflow", "examples", "library")
COMMITTED_SET = os.path.join("tests", "planning_test_sets", "library.json")
HANDWRITTEN = os.path.join("tests", "planning_test_sets", "library_handwritten.json")


@pytest.fixture(scope="module", autouse=True)
def initialized_fastworkflow():
    fastworkflow.init({})
    yield
    RoutingRegistry.clear_registry()


@pytest.fixture(scope="module")
def library_copy(tmp_path_factory) -> str:
    """A private copy without trained artifacts, caches or bytecode: nothing here writes into examples/."""
    destination = tmp_path_factory.mktemp("planning_match") / "library_copy"
    shutil.copytree(LIBRARY, destination, ignore=shutil.ignore_patterns("___*", "__pycache__"))
    yield str(destination)
    RoutingRegistry.clear_registry()


@pytest.fixture(scope="module")
def committed_set() -> dict:
    return pts.load_test_set(COMMITTED_SET)


# ---------------------------------------------------------------- the named test set

def test_builder_names_the_real_planning_commands_and_their_callees(library_copy):
    inv = pts.inventory(library_copy)
    resolution = resolve_field_sources(library_copy)
    assert inv.planning == sorted(visible_planning_commands(library_copy))     # framework commands are not inventoried
    assert len(inv.planning) == 10
    # callees are the execution commands the real phase steps call; a planning callee is not one
    assert inv.callees["Book/loan_report"] == ["Book/list_loans"]
    assert inv.callees["Library/member_report"] == [
        "Library/find_member", "Library/open_member", "Member/list_member_loans", "Record/show_notes"]
    assert inv.planning_callees["Library/book_loans"] == ["Library/book_by_title"]
    assert "Library/book_by_title" not in inv.all_callees


def test_builder_gives_every_planning_command_its_requests_and_every_callee_its_own(library_copy):
    test_set = pts.build_test_set(library_copy, per_command=8, handwritten_path=os.path.abspath(HANDWRITTEN),
                                  cache_root=os.path.join(library_copy, "___no_cache"))
    targets = test_set["targets"]
    planning = [name for name, block in targets.items() if block["kind"] == pts.KIND_PLANNING]
    callees = [name for name, block in targets.items() if block["kind"] == pts.KIND_CALLEE]
    assert sorted(planning) == sorted(visible_planning_commands(library_copy))
    assert set(callees) == set(pts.inventory(library_copy).all_callees)
    # hand-written requests cover every planning command to the per-command count, with no cache
    for name in planning:
        assert targets[name]["heldout"] == 8, name
    # every case's expected command is its target, and a callee case is about a planning command that calls it
    for case in test_set["cases"]:
        assert case["expected"] in targets
        if case["kind"] == pts.KIND_CALLEE:
            assert case["expected"] in targets[case["related"]]["callees"]
        else:
            assert case["expected"] == case["related"]


def test_seed_cases_are_in_sample_and_hand_written_cases_are_heldout(library_copy):
    test_set = pts.build_test_set(library_copy, per_command=8, handwritten_path=os.path.abspath(HANDWRITTEN),
                                  cache_root=os.path.join(library_copy, "___no_cache"))
    by_source = {}
    for case in test_set["cases"]:
        by_source.setdefault(case["source"], set()).add(case["split"])
    assert by_source[pts.SOURCE_SEED] == {pts.SPLIT_TRAIN}
    assert by_source[pts.SOURCE_HANDWRITTEN] == {pts.SPLIT_HELDOUT}
    assert pts.SOURCE_PERSONA not in by_source  # no cache, so no persona-holdout source
    seeds = [c for c in test_set["cases"] if c["source"] == pts.SOURCE_SEED]
    assert any(c["origin"] == "docstring" for c in seeds)
    assert any(c["origin"] == "plain_utterances" for c in seeds)


def test_committed_library_set_matches_the_loader_and_its_counts(committed_set):
    doc = committed_set
    assert doc["workflow"] == "library"
    assert doc["per_command"] == 8
    planning = [n for n, b in doc["targets"].items() if b["kind"] == pts.KIND_PLANNING]
    assert len(planning) == 10
    # every target is held-out cases plus its shortfall to 8; trained rows are excluded, so some are short
    assert all(doc["targets"][n]["heldout"] + doc["targets"][n]["shortfall"] == 8 for n in planning)
    assert doc["counts"]["heldout_by_kind"][pts.KIND_PLANNING] == 79
    assert doc["counts"]["total"] == len(doc["cases"])
    # the set holds requests only: no folder path of the workflow it was built from
    assert os.path.abspath(LIBRARY) not in json.dumps(doc)


def test_hand_written_cases_must_name_a_planning_command_or_its_callee(tmp_path, library_copy):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"cases": [{"utterance": "hello there", "expected": "Library/no_such_command"}]}))
    with pytest.raises(ValueError, match="neither a planning command nor a callee"):
        pts.build_test_set(library_copy, handwritten_path=str(bad), cache_root=str(tmp_path / "none"))
    wrong = tmp_path / "wrong.json"
    wrong.write_text(json.dumps({"cases": [{"utterance": "find a book", "expected": "Library/find_book",
                                            "related": "Book/loan_report"}]}))
    with pytest.raises(ValueError, match="does not call"):
        pts.build_test_set(library_copy, handwritten_path=str(wrong), cache_root=str(tmp_path / "none"))


# ---------------------------------------------------------------- production labels and the switch

def test_production_label_sets_carry_each_planning_command_on_its_lookup_path(library_copy):
    """Phase 10 step 1 (production): a planning command labels every head whose lookup path holds it."""
    RoutingRegistry.clear_registry()
    crd = RoutingRegistry.get_definition(library_copy)
    planning = set(planning_command_names_of(crd))
    assert planning, "the library example must have planning commands"
    assert active_experiment_planning_heads() is None  # production: no override
    assert active_planning_heads() == PLANNING_HEADS_LABELLED
    labels = {ctx: trained_command_labels(ctx, crd) for ctx in sorted(crd.contexts)}
    for ctx, names in labels.items():
        # the head holds exactly the planning commands whose scope is on its lookup path
        assert (names & planning) == planning_head_labels(ctx, crd), ctx
    assert {"Library/member_report", "Library/note_member"} <= labels["Library"]
    assert not (labels["Member"] & planning)  # no planning command is scoped to Member or Record
    assert not (labels["Record"] & planning)


def test_baseline_switch_removes_the_planning_labels_and_the_switch_is_gone_after(library_copy):
    RoutingRegistry.clear_registry()
    crd = RoutingRegistry.get_definition(library_copy)
    production = {ctx: trained_command_labels(ctx, crd) for ctx in sorted(crd.contexts)}
    with experiment_reserved_class_arm("sparse_1"), experiment_planning_heads(PLANNING_HEADS_NOT_HERE):
        assert active_planning_heads() == PLANNING_HEADS_NOT_HERE
        baseline = {ctx: trained_command_labels(ctx, crd) for ctx in sorted(crd.contexts)}
    planning = set(planning_command_names_of(crd))
    core = set(crd.command_directory.core_command_names)
    for ctx in production:
        assert not (baseline[ctx] & planning), ctx
        # the baseline is the pre-step-1 label set, verbatim: execution and core commands, no planning
        assert baseline[ctx] == {name for name in set(crd.contexts[ctx]) | core if name not in planning}, ctx
        assert production[ctx] - baseline[ctx] == planning_head_labels(ctx, crd), ctx
        assert baseline[ctx] <= production[ctx], ctx
    # and the switch is gone after the block: production again
    assert active_experiment_planning_heads() is None
    assert trained_command_labels("Library", crd) == production["Library"]


def test_labelled_switch_is_still_selectable_and_matches_production(library_copy):
    RoutingRegistry.clear_registry()
    crd = RoutingRegistry.get_definition(library_copy)
    production = {ctx: trained_command_labels(ctx, crd) for ctx in sorted(crd.contexts)}
    with experiment_reserved_class_arm("sparse_1"), experiment_planning_heads(PLANNING_HEADS_LABELLED):
        labelled = {ctx: trained_command_labels(ctx, crd) for ctx in sorted(crd.contexts)}
    assert labelled == production


def test_planning_labels_follow_the_scope_of_each_head(library_copy):
    RoutingRegistry.clear_registry()
    crd = RoutingRegistry.get_definition(library_copy)
    with experiment_reserved_class_arm("sparse_1"), experiment_planning_heads(PLANNING_HEADS_NOT_HERE):
        baseline = {ctx: trained_command_labels(ctx, crd) for ctx in sorted(crd.contexts)}
    labels = {ctx: trained_command_labels(ctx, crd) for ctx in sorted(crd.contexts)}
    # the Library head takes the Library planning commands that take utterances, and nothing else
    added = labels["Library"] - baseline["Library"]
    assert {"Library/member_report", "Library/note_member", "Library/find_record"} <= added
    assert "Library/overdue_summary" in added  # a planning command without parameters is a label too
    assert not any(name.startswith("Member/") for name in added)
    # the Book head takes its own planning commands, not the Library's
    assert {"Book/loan_report", "Book/shelf_report"} <= (labels["Book"] - baseline["Book"])
    assert not (labels["Book"] - baseline["Book"]) & {"Library/member_report"}
    # every head's labels are its baseline labels plus planning commands only
    for ctx in baseline:
        assert baseline[ctx] <= labels[ctx], ctx
        assert (labels[ctx] - baseline[ctx]) <= set(planning_command_names_of(crd)), ctx


def test_planning_switch_refuses_an_unknown_mode_and_needs_the_sparse_arm():
    with pytest.raises(ValueError, match="unknown planning-head mode"):
        with experiment_planning_heads("everything"):
            pass
    with pytest.raises(ValueError, match="needs a reserved-class arm"):
        with experiment_planning_heads(PLANNING_HEADS_NOT_HERE):
            pass
    with pytest.raises(ValueError, match="needs a reserved-class arm"):
        with experiment_planning_heads(PLANNING_HEADS_LABELLED):
            pass
    with experiment_planning_heads(None):  # None is production, always allowed
        pass


def test_run_metadata_records_the_layout_in_force(library_copy):
    _manifest, production = construction_records(None, None, library_copy)
    assert production["planning_heads"] == PLANNING_HEADS_LABELLED
    with experiment_reserved_class_arm("sparse_1"), experiment_planning_heads(PLANNING_HEADS_NOT_HERE):
        _manifest, baseline = construction_records(None, None, library_copy)
    assert baseline["planning_heads"] == PLANNING_HEADS_NOT_HERE


def test_offline_arm_refuses_a_trained_copy_built_under_the_other_layout(library_copy):
    provenance = os.path.join(library_copy, "___command_info", "training_provenance.json")
    os.makedirs(os.path.dirname(provenance), exist_ok=True)

    with open(provenance, "w", encoding="utf-8") as handle:
        json.dump({"schema_version": 2, "planning_heads": "not_here"}, handle)
    spm.check_planning_arm(library_copy, spm.ARM_HEADS_NOT_HERE)
    with pytest.raises(spm.ArmMismatchError, match="needs planning_heads='labelled'"):
        spm.check_planning_arm(library_copy, spm.ARM_HEADS_LABELLED)

    with open(provenance, "w", encoding="utf-8") as handle:
        json.dump({"schema_version": 2, "planning_heads": "labelled"}, handle)
    spm.check_planning_arm(library_copy, spm.ARM_HEADS_LABELLED)
    with pytest.raises(spm.ArmMismatchError, match="needs planning_heads='not_here'"):
        spm.check_planning_arm(library_copy, spm.ARM_HEADS_NOT_HERE)
    os.remove(provenance)


# ---------------------------------------------------------------- the live arms

def test_live_arm_refuses_without_allow_live(library_copy, capsys):
    code = spm.main(["live", "--arm", "jev", "--workflow", library_copy, "--test-set", COMMITTED_SET,
                     "--per-stratum", "5"])
    assert code == 2
    assert "refused" in capsys.readouterr().err


@pytest.mark.parametrize("arm", ["jev", "agent_plan"])
def test_dry_run_prints_the_estimate_and_calls_nothing(library_copy, capsys, arm):
    code = spm.main(["live", "--arm", arm, "--workflow", library_copy, "--test-set", COMMITTED_SET,
                     "--per-stratum", "5", "--dry-run"])
    out = capsys.readouterr().out
    assert code == 0
    assert "provider calls" in out and "input tokens" in out
    assert "dry run: no call made" in out


def test_estimate_counts_one_planner_call_per_request(committed_set, library_copy):
    sample = spm.stratified_sample(pts.heldout(committed_set), 5, seed=0)
    rows = spm.routing_rows(sample)
    est = spm.estimate(spm.ARM_AGENT_PLAN, rows, library_copy)
    assert est["calls"] == len(rows)
    assert est["input_tokens"] > 0


# ---------------------------------------------------------------- stratified sampling

def _cases(per: int) -> list[dict]:
    out = []
    for stratum in ("P/a", "P/b", "C/x"):
        for i in range(per):
            out.append({"expected": stratum, "utterance": f"{stratum} request {i}", "kind": "planning",
                        "head": "P", "split": "heldout", "related": "P/a", "source": "hand-written"})
    return out


def test_stratified_sample_is_deterministic_and_caps_each_stratum():
    cases = _cases(10)
    first = spm.stratified_sample(cases, 3, seed=0)
    again = spm.stratified_sample(list(reversed(cases)), 3, seed=0)
    assert first == spm.stratified_sample(cases, 3, seed=0)  # same input, same output
    assert sorted(c["utterance"] for c in first) == sorted(c["utterance"] for c in again)
    per = {}
    for case in first:
        per[case["expected"]] = per.get(case["expected"], 0) + 1
    assert per == {"P/a": 3, "P/b": 3, "C/x": 3}
    positions = [cases.index(c) for c in first]
    assert positions == sorted(positions)  # the input order is kept


def test_stratified_sample_changes_with_the_seed_and_keeps_a_small_stratum_whole():
    cases = _cases(10)
    assert spm.stratified_sample(cases, 3, seed=0) != spm.stratified_sample(cases, 3, seed=1)
    small = _cases(2)
    assert len(spm.stratified_sample(small, 5, seed=0)) == len(small)
    with pytest.raises(ValueError):
        spm.stratified_sample(cases, 0)


# ---------------------------------------------------------------- the report math

def _case(index, split, kind, expected, top, *, trained_on=False):
    """A hand-built result row. The utterance depends on the index only, so two arms score the same key."""
    return {"head": "H", "utterance": f"{kind} {expected} request {index}",
            "kind": kind, "expected": expected, "related": expected, "split": split,
            "source": "hand-written", "top_label": top, "topk": [], "trained_on": trained_on}


def _base_cases():
    rows = [
        ("planning", "P/a", ""),          # declined
        ("planning", "P/b", "C/y"),       # wrong accept
        ("planning", "P/a", ""),          # declined
        ("planning", "P/b", ""),          # declined
        ("callee", "C/x", "C/x"),         # routed
        ("callee", "C/x", "P/a"),         # wrong accept
        ("callee", "C/y", "C/y"),         # routed
        ("callee", "C/y", "wildcard"),    # declined
    ]
    return [_case(i, "heldout", kind, expected, top) for i, (kind, expected, top) in enumerate(rows)]


def _arm_cases():
    rows = [
        ("planning", "P/a", "P/a"),       # routed
        ("planning", "P/b", "P/b"),       # routed
        ("planning", "P/a", "P/a"),       # routed
        ("planning", "P/b", ""),          # declined
        ("callee", "C/x", "C/x"),
        ("callee", "C/x", "C/x"),
        ("callee", "C/y", "C/y"),
        ("callee", "C/y", "C/y"),
    ]
    return [_case(i, "heldout", kind, expected, top) for i, (kind, expected, top) in enumerate(rows)]


def test_rates_and_wilson_interval_on_hand_built_results():
    base = {"arm": "base", "cases": _base_cases()}
    summary = spm.summarize(base)
    assert summary["heldout_n"] == 8
    assert summary["planning_routed"]["count"] == 0 and summary["planning_routed"]["trials"] == 4
    assert summary["callee_routed"]["count"] == 2 and summary["callee_routed"]["trials"] == 4
    assert summary["wrong_accepts"]["count"] == 2 and summary["wrong_accepts"]["trials"] == 8
    # 0 of 4 has the textbook Wilson upper bound z^2 / (n + z^2)
    assert summary["planning_routed"]["wilson95"][1] == pytest.approx(3.8415 / (4 + 3.8415), abs=2e-3)
    assert summary["per_target"]["P/b"] == {"n": 2, "routed": 0, "declined": 1, "wrong_accept": 1}


def test_paired_difference_and_mcnemar_on_hand_built_results():
    comparison = spm.compare_arms({"arm": "arm", "cases": _arm_cases()}, {"arm": "base", "cases": _base_cases()})
    planning = comparison["metrics"]["planning_routed"]
    assert planning["n"] == 4 and planning["only_arm"] == 3 and planning["only_base"] == 0
    assert planning["difference"] == pytest.approx(0.75)
    # paired Wald: var = (3 + 0 - 3^2/4) / 16 = 0.046875
    assert planning["lower"] == pytest.approx(0.75 - 1.959964 * 0.046875 ** 0.5, abs=1e-4)
    # exact McNemar on 3 vs 0 discordant pairs: 2 * (1/8)
    assert planning["p_two_sided"] == pytest.approx(0.25)
    accept = comparison["metrics"]["wrong_accepts"]
    assert accept["n"] == 8 and accept["only_arm"] == 0 and accept["only_base"] == 2
    assert accept["difference"] == pytest.approx(-0.25)
    assert accept["p_two_sided"] == pytest.approx(0.5)


def test_guidance_bars_are_read_off_the_intervals_and_do_not_gate():
    base = {"arm": "heads_not_here", "cases": _base_cases()}
    report = spm.report(base, {"heads_labelled": {"arm": "heads_labelled", "cases": _arm_cases()}})
    row = report["arms"]["heads_labelled"]
    checks = row["guidance"]["checks"]
    assert checks["planning_routed_lower_at_least_0"] is True       # lower bound 0.33 >= 0
    assert checks["callee_routed_lower_at_least_minus_2pts"] is True
    assert checks["wrong_accepts_upper_at_most_2pts"] is False      # arm has 0 wrong accepts; base's
    # difference upper bound is well above 0.02 on eight cases: the bar is reported as not met
    assert row["guidance"]["all_met"] is False
    assert "guidance only" in row["guidance"]["note"]
    assert "heads_labelled" in spm.format_report(report)


def test_trained_on_cases_are_excluded_and_mismatched_cases_are_refused():
    cases = _base_cases() + [_case(99, "heldout", "planning", "P/a", "P/a", trained_on=True)]
    summary = spm.summarize({"arm": "x", "cases": cases})
    assert summary["heldout_n"] == 8 and summary["trained_on_excluded"] == 1
    arm = {"arm": "arm", "cases": _arm_cases()}
    shifted = {"arm": "base", "cases": list(reversed(_base_cases()))}
    with pytest.raises(ValueError, match="same cases in the same order"):
        spm.compare_arms(arm, shifted)


# ---------------------------------------------------------------- agent_plan: plans read as production reads them

PLAN_NAMES = ["Library/find_book", "Library/author_books", "Book/loan_report", "Library/book_loans"]
PLAN_CORE = ["IntentDetection/go_up"]


def _plan(raw: str) -> dict:
    return spm.read_plan(raw, PLAN_NAMES, PLAN_CORE)


def test_prose_plan_names_no_command_and_is_not_a_call():
    plan = _plan("1. Look up who wrote Dune.\n2. Show the loans.")
    assert plan["plan_length"] == 2
    assert plan["command_steps"] == [] and plan["chosen_command"] == ""
    assert spm.classify_plan("Library/find_book", plan) == spm.PLAN_NO_CALL


def test_empty_plan_is_no_plan():
    plan = _plan("")
    assert plan["plan_length"] == 0 and plan["chosen_command"] == ""
    assert spm.classify_plan("Library/find_book", plan) == spm.PLAN_NO_PLAN


def test_single_call_plan_is_the_fast_path_and_names_the_qualified_command():
    plan = _plan("1. Use Library/find_book with title Dune")
    assert plan["chosen_command"] == "Library/find_book"
    assert plan["plan_length"] == 1
    assert spm.classify_plan("Library/find_book", plan) == spm.PLAN_SINGLE_CALL


def test_call_plus_plain_steps_is_call_plus_steps_and_first_command_step_is_chosen():
    plan = _plan("1. Library/author_books title=Dune\n2. Tell the user the branch holding it.")
    assert plan["command_steps"] == ["Library/author_books"]
    assert plan["chosen_command"] == "Library/author_books"
    assert plan["plan_length"] == 2
    assert spm.classify_plan("Library/author_books", plan) == spm.PLAN_CALL_PLUS_STEPS


def test_multi_step_plan_chooses_the_first_command_step_not_a_later_one():
    plan = _plan("1. Find the author of Dune.\n2. Library/author_books title=Dune\n3. Library/find_book title=Dune")
    assert plan["plan_length"] == 3
    assert plan["command_steps"] == ["Library/author_books", "Library/find_book"]
    assert plan["chosen_command"] == "Library/author_books"
    assert spm.classify_plan("Library/find_book", plan) == spm.PLAN_CALL_PLUS_STEPS


def test_negated_step_is_not_the_chosen_command():
    plan = _plan("1. Do not use Library/find_book.\n2. Use Library/author_books title=Dune")
    assert plan["chosen_command"] == "Library/author_books"


def test_plan_naming_a_planning_command_of_another_head_is_no_call_for_this_case():
    plan = _plan("1. Book/loan_report title=Dune")
    assert plan["chosen_command"] == "Book/loan_report"
    assert spm.classify_plan("Library/find_book", plan) == spm.PLAN_NO_CALL


def test_plain_name_shared_by_two_commands_resolves_by_the_step_text():
    names = ["Book/show_book", "Library/show_book"]
    plan = spm.read_plan("1. Library/show_book title=Dune", names, [])
    assert plan["chosen_command"] == "Library/show_book"


def test_partly_covered_label_checks_the_right_extra_steps():
    right = {"plan": _plan("1. Library/author_books title=Dune\n2. Library/find_book title=Dune"),
             "expected": "Library/author_books", "extra_commands": ["Library/find_book"]}
    wrong = {"plan": _plan("1. Library/author_books title=Dune\n2. Book/loan_report"),
             "expected": "Library/author_books", "extra_commands": ["Library/find_book"]}
    unlabelled = {"plan": right["plan"], "expected": "Library/author_books"}
    assert spm.right_extra_steps(right) is True
    assert spm.right_extra_steps(wrong) is False
    assert spm.right_extra_steps(unlabelled) is None


def _planning_case(expected: str, raw: str, **extra) -> dict:
    return {"expected": expected, "utterance": f"request for {expected} {len(raw)}", "kind": "planning",
            "head": expected.split("/")[0], "split": "heldout", "related": expected, "source": "hand-written",
            "plan": _plan(raw), **extra}


def test_plan_2_44_summary_counts_covered_single_calls_and_leaves_partly_unmeasured_when_unlabelled():
    cases = [
        _planning_case("Library/find_book", "1. Use Library/find_book title=Dune"),
        _planning_case("Library/author_books", "1. Library/author_books title=Dune\n2. Look at the branch."),
        _planning_case("Book/loan_report", "1. Look up the answer."),
    ]
    summary = spm.plan_2_44_summary(cases)
    covered = summary["covered"]
    assert covered["n"] == 3
    assert covered["classes"] == {"single_call": 1, "call_plus_steps": 1, "no_call": 1, "no_plan": 0}
    assert covered["single_call"]["count"] == 1 and covered["single_call"]["trials"] == 3
    partly = summary["partly_covered"]
    assert partly["n"] == 0 and partly["labelled"] == 0
    assert "not measured" in partly["note"]


def test_plan_2_44_summary_scores_labelled_partly_covered_cases():
    labelled = _planning_case("Library/author_books", "1. Library/author_books title=Dune\n2. Library/find_book",
                              partly_covered=True, extra_commands=["Library/find_book"])
    summary = spm.plan_2_44_summary([labelled])
    assert summary["covered"]["n"] == 0
    assert summary["partly_covered"]["labelled"] == 1
    assert summary["partly_covered"]["call_with_right_extra_steps"]["count"] == 1


def test_summarize_carries_the_2_44_block_only_for_runs_with_plans():
    with_plans = {"arm": "agent_plan", "cases": [_planning_case("Library/find_book", "1. Use Library/find_book")]}
    assert "plan_2_44" in spm.summarize(with_plans)
    jev_like = {"arm": "jev", "cases": [{k: v for k, v in with_plans["cases"][0].items() if k != "plan"}]}
    assert "plan_2_44" not in spm.summarize(jev_like)


def test_run_live_agent_plan_stores_raw_steps_and_top_label(library_copy, monkeypatch):
    """The planner is stubbed with hand-built plans: nothing calls a provider."""
    canned = [_plan("1. Use Library/find_book title=Dune\n2. Say the answer."),
              _plan("1. Look up the author.")]
    monkeypatch.setattr(spm, "_agent_plans", lambda rows, folder: canned)
    rows = [{"expected": "Library/find_book", "utterance": "a", "kind": "planning", "head": "Library",
             "related": "Library/find_book", "split": "heldout", "source": "seed"},
            {"expected": "Library/author_books", "utterance": "b", "kind": "callee", "head": "Library",
             "related": "Library/book_loans", "split": "heldout", "source": "seed"}]
    results = spm.run_live(spm.ARM_AGENT_PLAN, rows, library_copy)
    first, second = results["cases"]
    assert first["top_label"] == "Library/find_book" and first["plan"]["plan_length"] == 2
    assert first["plan"]["raw"].startswith("1. Use Library/find_book")
    assert second["top_label"] == "" and second["plan"]["plan_length"] == 1


# ---------------------------------------------------------------- live runs: one run per arm and run id

def _live_args(library_copy: str, out: str, run_id: str, *extra: str) -> list[str]:
    return ["live", "--arm", "jev", "--workflow", library_copy, "--test-set", COMMITTED_SET,
            "--per-stratum", "5", "--allow-live", "--out", out, "--run-id", run_id, *extra]


def _refuse_any_provider_call(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("a provider call was attempted")
    monkeypatch.setattr(spm, "run_live", boom)


def test_live_run_refused_while_its_lock_is_held(library_copy, tmp_path, capsys, monkeypatch):
    out = str(tmp_path / "jev_r1.json")
    lock = spm.live_lock_path("jev", out, "r1")
    open(lock, "w").close()
    _refuse_any_provider_call(monkeypatch)
    assert spm.main(_live_args(library_copy, out, "r1")) == 2
    assert "lock" in capsys.readouterr().err
    assert not os.path.exists(out)


def test_live_run_refused_when_its_results_file_already_exists(library_copy, tmp_path, capsys, monkeypatch):
    out = str(tmp_path / "jev_r2.json")
    with open(out, "w") as handle:
        handle.write("{}")
    _refuse_any_provider_call(monkeypatch)
    assert spm.main(_live_args(library_copy, out, "r2")) == 2
    assert "already exists" in capsys.readouterr().err


def test_second_claim_of_the_same_run_is_refused_and_the_first_lock_survives(tmp_path):
    out = str(tmp_path / "agent_plan_r3.json")
    lock = spm.claim_live_run("agent_plan", out, "r3")
    with pytest.raises(spm.LiveRunRefused):
        spm.claim_live_run("agent_plan", out, "r3")
    assert os.path.exists(lock)
    spm.release_live_run(lock)
    assert not os.path.exists(lock)


def test_different_run_ids_do_not_block_each_other(tmp_path):
    out = str(tmp_path / "agent_plan_r4.json")
    lock_a = spm.claim_live_run("agent_plan", out, "a")
    lock_b = spm.claim_live_run("agent_plan", out, "b")
    spm.release_live_run(lock_a)
    spm.release_live_run(lock_b)


def test_live_run_releases_its_lock_after_a_clean_run(library_copy, tmp_path, monkeypatch):
    out = str(tmp_path / "jev_r5.json")
    monkeypatch.setattr(spm, "run_live", lambda arm, cases, folder: {
        "schema": spm.SCHEMA_VERSION, "arm": arm, "cases": [], "training_rows": {}, "train_seconds": None})
    assert spm.main(_live_args(library_copy, out, "r5")) == 0
    assert os.path.exists(out)
    assert not os.path.exists(spm.live_lock_path("jev", out, "r5"))


def test_live_run_releases_its_lock_when_the_run_fails(library_copy, tmp_path, monkeypatch):
    out = str(tmp_path / "jev_r6.json")

    def fails(arm, cases, folder):
        raise RuntimeError("provider error")
    monkeypatch.setattr(spm, "run_live", fails)
    with pytest.raises(RuntimeError):
        spm.main(_live_args(library_copy, out, "r6"))
    assert not os.path.exists(spm.live_lock_path("jev", out, "r6"))
    assert not os.path.exists(out)


def test_live_run_needs_a_run_id(library_copy, tmp_path, capsys, monkeypatch):
    _refuse_any_provider_call(monkeypatch)
    code = spm.main(["live", "--arm", "jev", "--workflow", library_copy, "--test-set", COMMITTED_SET,
                     "--allow-live", "--out", str(tmp_path / "x.json")])
    assert code == 2 and "--run-id" in capsys.readouterr().err


def test_dry_run_reports_that_a_repeat_would_be_refused(library_copy, tmp_path, capsys):
    out = str(tmp_path / "agent_plan_r7.json")
    with open(out, "w") as handle:
        handle.write("{}")
    code = spm.main(["live", "--arm", "agent_plan", "--workflow", library_copy, "--test-set", COMMITTED_SET,
                     "--per-stratum", "5", "--dry-run", "--out", out, "--run-id", "r7"])
    out_text = capsys.readouterr().out
    assert code == 0 and "would be refused" in out_text and "dry run: no call made" in out_text


# ---------------------------------------------------------------- held-out dedupe against seeds

def test_deduper_drops_equal_and_near_seeds_and_near_duplicate_heldout_requests():
    seeds = [("Find a book called Dune", "Library/find_book"), ("open the loan report", "Book/loan_report")]
    deduper = pts.HeldoutDeduper(seeds)
    # equal after casefold and whitespace collapse
    assert not deduper.check("  find A  book called   dune ", "Library/open_book", "persona-holdout")
    # near a seed (ratio >= 0.9)
    assert not deduper.check("open the loan report!", "Book/loan_report", "persona-holdout")
    # kept
    assert deduper.check("which branch holds this title", "Library/find_book", "persona-holdout")
    # near-duplicate of a kept held-out request, across targets
    assert not deduper.check("which branch holds this title?", "Library/open_book", "hand-written")
    # exact repeat of a kept held-out request
    assert not deduper.check("which branch holds this title", "Library/find_book", "hand-written")
    reasons = [record["reason"] for record in deduper.dropped]
    assert reasons[0].startswith("equals a seed or trained row of Library/find_book")
    assert reasons[1].startswith("near a seed or trained row of Book/loan_report")
    assert reasons[2].startswith("near a kept held-out request of Library/find_book")
    assert reasons[3].startswith("equals a kept held-out request of Library/find_book")
    assert all(record["source"] for record in deduper.dropped)


def test_committed_library_heldout_requests_are_not_near_any_seed(library_copy):
    inv = pts.inventory(LIBRARY)
    seeds = [pts.normalise(text) for text, _owner in pts.all_seed_texts(inv)]
    checked = [pts.normalise(case["utterance"]) for case in pts.heldout(load_committed_library())]
    assert checked
    for held in checked:
        for seed in seeds:
            assert held != seed
            assert difflib.SequenceMatcher(None, held, seed).ratio() < pts.DEDUPE_RATIO, (held, seed)


def load_committed_library() -> dict:
    return pts.load_test_set(COMMITTED_SET)


def test_rebuilding_the_library_set_is_deterministic_and_matches_the_committed_file(committed_set):
    rebuilt = pts.build_test_set(LIBRARY, handwritten_path=HANDWRITTEN)
    again = pts.build_test_set(LIBRARY, handwritten_path=HANDWRITTEN)
    assert json.dumps(rebuilt, sort_keys=True) == json.dumps(again, sort_keys=True)
    assert rebuilt["cases"] == committed_set["cases"]
    assert rebuilt["dropped"] == committed_set["dropped"]
    assert rebuilt["counts"] == committed_set["counts"]


def test_call_with_one_retry_retries_a_transient_failure_once():
    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) == 1:
            raise TimeoutError("transient")
        return "prediction"
    prediction, count, error = spm.call_with_one_retry(flaky)
    assert prediction == "prediction" and count == 2 and error == ""


def test_call_with_one_retry_gives_up_after_two_attempts_and_records_the_error():
    attempts = []

    def broken():
        attempts.append(1)
        raise RuntimeError("provider down")
    prediction, count, error = spm.call_with_one_retry(broken)
    assert prediction is None and count == 2 and len(attempts) == 2
    assert "provider down" in error


# ---------------------------------------------------------------- reach measures (all steps read)

REACH_NAMES = ["Library/find_book", "Library/open_book", "Book/list_loans", "Book/loan_report",
               "Library/author_books", "Library/find_member", "Library/open_member"]
REACH_COVERS = {"Book/list_loans": {"Book/loan_report"}, "Library/find_book": {"Library/author_books"}}


def _reach_case(kind: str, expected: str, raw: str) -> dict:
    return {"kind": kind, "expected": expected, "utterance": "u", "head": "X", "split": "heldout",
            "plan": spm.read_plan(raw, REACH_NAMES, PLAN_CORE)}


def test_planning_request_is_reached_in_any_step_and_is_not_a_single_call_when_it_has_other_steps():
    case = _reach_case("planning", "Library/author_books",
                       "1. Library/find_book title=Dune\n2. Library/author_books title=Dune")
    measured = spm.plan_reach(case, REACH_COVERS)
    assert measured["uses_planning"] is True and measured["single_call"] is False
    assert measured["reach"] == spm.REACH_REACHES


def test_single_call_planning_request_is_the_fast_path():
    case = _reach_case("planning", "Library/author_books", "1. Library/author_books title=Dune")
    measured = spm.plan_reach(case, REACH_COVERS)
    assert measured["uses_planning"] is True and measured["single_call"] is True


def test_callee_reached_at_the_end_of_a_multi_step_plan():
    case = _reach_case("callee", "Book/list_loans",
                       "1. Library/find_book title=Dune\n2. Library/open_book\n3. Book/list_loans")
    measured = spm.plan_reach(case, REACH_COVERS)
    assert measured["reach"] == spm.REACH_REACHES and measured["uses_planning"] is None


def test_callee_not_named_but_its_covering_planning_command_is_named_is_delegation():
    case = _reach_case("callee", "Book/list_loans", "1. Book/loan_report title=Dune")
    assert spm.plan_reach(case, REACH_COVERS)["reach"] == spm.REACH_DELEGATES


def test_plan_naming_only_other_commands_is_misroute():
    planning = _reach_case("planning", "Library/author_books", "1. Library/find_member query=Lee")
    callee = _reach_case("callee", "Book/list_loans", "1. Library/find_member query=Lee")
    assert spm.plan_reach(planning, REACH_COVERS)["reach"] == spm.REACH_MISROUTE
    assert spm.plan_reach(callee, REACH_COVERS)["reach"] == spm.REACH_MISROUTE


def test_no_command_plans_split_into_clarifying_questions_and_descriptions():
    question = _reach_case("planning", "Library/author_books",
                           "1. Clarify with the user which author they mean")
    description = _reach_case("callee", "Book/list_loans", "Count the overdue loans across the library.")
    assert spm.plan_reach(question, REACH_COVERS)["clarify"] is True
    described = spm.plan_reach(description, REACH_COVERS)
    assert described["reach"] == spm.REACH_NO_COMMAND and described["clarify"] is False


def test_coverage_map_lists_the_planning_commands_whose_callees_include_a_callee():
    test_set = {"targets": {
        "Book/loan_report": {"kind": "planning", "callees": ["Book/list_loans"]},
        "Library/author_books": {"kind": "planning", "callees": ["Library/find_book", "Library/open_book"]},
        "Book/list_loans": {"kind": "callee", "callers": ["Book/loan_report"]},
    }}
    covers = spm.coverage_map(test_set)
    assert covers == {"Book/list_loans": {"Book/loan_report"},
                      "Library/find_book": {"Library/author_books"},
                      "Library/open_book": {"Library/author_books"}}


def test_jev_choice_is_read_as_a_one_step_plan():
    chosen = {"kind": "planning", "expected": "Library/author_books", "top_label": "Library/author_books"}
    assert spm.plan_reach(dict(chosen, plan=spm.jev_as_plan(chosen)), REACH_COVERS)["reach"] == spm.REACH_REACHES
    none = {"kind": "callee", "expected": "Book/list_loans", "top_label": ""}
    assert spm.plan_reach(dict(none, plan=spm.jev_as_plan(none)), REACH_COVERS)["reach"] == spm.REACH_NO_COMMAND


def test_reach_summary_counts_each_outcome_and_the_clarifying_split():
    cases = [
        _reach_case("planning", "Library/author_books", "1. Library/author_books title=Dune"),
        _reach_case("planning", "Library/author_books", "1. Library/find_member query=Lee"),
        _reach_case("callee", "Book/list_loans", "1. Which loan do you mean?"),
    ]
    summary = spm.reach_summary(cases, REACH_COVERS)
    assert summary["planning"]["uses_planning"]["count"] == 1
    assert summary["planning"]["single_call"]["count"] == 1
    assert summary["planning"]["outcomes"][spm.REACH_MISROUTE] == 1
    assert summary["no_command"] == {"n": 1, "clarifying_questions": 1, "descriptions": 0}


def test_a_conditional_step_still_names_its_command_for_reach():
    case = _reach_case("callee", "Book/list_loans",
                       "1. Library/find_book title=Dune\n2. If a book is found, execute Book/list_loans")
    assert spm.plan_reach(case, REACH_COVERS)["reach"] == spm.REACH_REACHES


def test_a_negated_step_does_not_count_as_naming_a_command():
    case = _reach_case("callee", "Book/list_loans", "1. Do not use Book/list_loans; say the answer.")
    assert spm.plan_reach(case, REACH_COVERS)["reach"] == spm.REACH_NO_COMMAND


# ---------------------------------------------------------------- lever 2: the agent gets the head's candidates

LEVER2_TEMPLATES = {"Library/overdue_summary": ("list the branches", "count the overdue loans")}


def test_lever2_hint_format_numbers_candidates_and_shows_planning_templates():
    hint = spm.lever2_hint(["Library/overdue_summary", "Library/count_overdue"], LEVER2_TEMPLATES)
    assert hint.splitlines() == [
        spm.LEVER2_HEADER,
        "1. Library/overdue_summary [planning command; template: list the branches; count the overdue loans]",
        "2. Library/count_overdue [command]",
    ]


def test_lever2_hint_is_empty_without_candidates_and_flags_a_planning_command_with_no_template():
    assert spm.lever2_hint([], LEVER2_TEMPLATES) == ""
    assert "[planning command; template: (no template)]" in spm.lever2_hint(["Book/x"], {"Book/x": ()}).replace(
        "1. Book/x", "1. Book/x")


def test_lever2_candidates_are_the_real_topk_top_first_and_skip_not_here_labels():
    head = {"top_label": "wildcard", "topk": ["wildcard", "Book/shelf_report", "Book/loan_report", "parameter_value"]}
    assert spm.lever2_candidates(head) == ["Book/shelf_report", "Book/loan_report"]
    head = {"top_label": "Book/loan_report", "topk": ["Book/shelf_report", "Book/loan_report"]}
    assert spm.lever2_candidates(head) == ["Book/loan_report", "Book/shelf_report"]


def test_lever2_cases_keep_only_unsure_or_declined_heads_on_the_shared_analysis_cases():
    def head(key, top, confident):
        return {"head": key[0], "utterance": key[1], "expected": key[2], "kind": key[3], "split": key[4],
                "source": "seed", "related": key[2], "top_label": top, "confident": confident,
                "topk": [top] if top else []}
    keys = {("H", "u1", "H/a", "planning", "heldout"), ("H", "u2", "H/a", "planning", "heldout"),
            ("H", "u3", "H/a", "planning", "heldout"), ("H", "u4", "H/a", "planning", "heldout")}
    heads = [head(("H", "u1", "H/a", "planning", "heldout"), "H/a", False),
             head(("H", "u2", "H/a", "planning", "heldout"), "wildcard", True),
             head(("H", "u3", "H/a", "planning", "heldout"), "H/a", True),
             head(("H", "u4", "H/a", "planning", "heldout"), "H/b", False)]
    jev = {("H", "u1", "H/a", "planning", "heldout"), ("H", "u2", "H/a", "planning", "heldout"),
           ("H", "u3", "H/a", "planning", "heldout")}
    selected = spm.lever2_cases(heads, keys, jev)
    assert [c["utterance"] for c in selected] == ["u1", "u2"]  # u3 confident; u4 not in the jev sample


def test_run_lever2_passes_each_hint_to_the_planner_and_stores_it(monkeypatch):
    seen = {}

    def fake_agent_plans(rows, folder, hints=None):
        seen["hints"] = list(hints)
        return [spm.read_plan("1. Library/count_overdue", REACH_NAMES + ["Library/count_overdue"], PLAN_CORE) for _ in rows]
    monkeypatch.setattr(spm, "_agent_plans", fake_agent_plans)
    rows = [{"expected": "Library/count_overdue", "utterance": "how many overdue", "kind": "planning",
             "head": "Library", "related": "Library/count_overdue", "split": "heldout", "source": "seed",
             "hint": "HINT"}]
    results = spm.run_lever2(rows, "unused")
    assert seen["hints"] == ["HINT"]
    assert results["arm"] == "lever2" and results["cases"][0]["top_label"] == "Library/count_overdue"
    assert results["cases"][0]["hint"] == "HINT" and results["cases"][0]["plan"]["plan_length"] == 1


def test_lever2_live_refuses_on_an_existing_results_file_and_never_calls(library_copy, tmp_path, capsys, monkeypatch):
    out = tmp_path / "lever2_s1.json"
    out.write_text("{}")
    monkeypatch.setattr(spm, "run_lever2", lambda rows, folder: (_ for _ in ()).throw(AssertionError("called")))
    heads = tmp_path / "heads.json"
    heads.write_text(json.dumps({"cases": []}))
    keys = tmp_path / "keys.json"
    keys.write_text(json.dumps({"analysis_keys": []}))
    jev = tmp_path / "jev.json"
    jev.write_text(json.dumps({"cases": []}))
    code = spm.main(["lever2", "--workflow", library_copy, "--test-set", COMMITTED_SET,
                     "--heads-results", str(heads), "--analysis-keys", str(keys), "--jev-results", str(jev),
                     "--allow-live", "--out", str(out), "--run-id", "lever2_s1"])
    assert code == 2 and "already exists" in capsys.readouterr().err


# ---------------------------------------------------------------- listing arms (planner-only live run, 10.7)

def _listing_wf(label: str, n: int, chars: int) -> dict:
    return {"label": label, "arm": "full", "rows": [{"utterance": "u" * 10, "listing_chars": chars} for _ in range(n)]}


def test_listing_gates_refuse_over_the_call_or_prompt_limit():
    ok = spm.listing_dry_run_totals([_listing_wf("library", 63, 6000), _listing_wf("app", 40, 5000)])
    assert ok["calls"] == 103 and ok["gate_ok"] is True
    too_many = spm.listing_dry_run_totals([_listing_wf("x", spm.GATE_MAX_CALLS + 1, 10)])
    assert too_many["gate_ok"] is False
    too_big = spm.listing_dry_run_totals([_listing_wf("x", 1, spm.GATE_MAX_PROMPT_TOKENS * 4)])
    assert too_big["gate_ok"] is False


def test_usage_is_summed_from_the_new_lm_history_entries_and_cache_hits_are_flagged():
    entries = [{"usage": {"prompt_tokens": 1200, "completion_tokens": 300}},
               {"usage": {"prompt_tokens": 800, "completion_tokens": 100}, "cache_hit": True}]
    assert spm._usage_of(entries) == (2000, 400, True)
    assert spm._usage_of([{}]) == (0, 0, False)


def test_listing_score_counts_reach_of_all_gold_and_splits_the_misses_by_cause():
    plan_hit = spm.read_plan("1. Library/author_books title=Dune\n2. Library/find_book title=Dune", REACH_NAMES, PLAN_CORE)
    plan_miss = spm.read_plan("1. Library/find_member query=Lee", REACH_NAMES, PLAN_CORE)
    base = {"listing": "", "listing_chars": 0, "latency_s": 1.0, "prompt_tokens": 100, "output_tokens": 20,
            "cache_hit": False, "lm_calls": 1, "kind": "planning", "expected": "Library/author_books",
            "gold": ["Library/author_books", "Library/find_book"], "utterance": "u", "source": "seed"}
    doc = {"workflows": [{"label": "app", "arm": "command_shortlist", "active": "Library", "bases": {},
                          "rows": [dict(base, plan=plan_hit),
                                   dict(base, plan=plan_miss, utterance="v")]}]}
    report = spm.score_listing_run(doc)["app"]
    assert report["n"] == 2 and report["reach_all_gold"] == 1
    assert report["names_gold_planning"] == 1 and report["single_call_of_planning"] == 0
    assert report["prompt_tokens_total"] == 200 and report["failed_requests"] == 0
    # the missed gold is not on the (empty) listing: a shortlist miss
    assert report["misses"] == {"shortlist (gold not on the listing)": 1}


def test_listing_plan_retries_cap_at_one_attempt_after_the_first():
    seen = []

    def always_fails():
        seen.append(1)
        raise RuntimeError("provider down")
    prediction, attempts, error = spm.call_with_one_retry(always_fails)
    assert prediction is None and attempts == 2 and len(seen) == 2 and "provider down" in error

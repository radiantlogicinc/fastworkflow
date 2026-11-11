"""The offline measurement of the run-listing arms (Phase 10 step 4; fastworkflow/train/score_listing.py).

Integration-style, no mocks, no model call. The workflow cases run on COPIES of the committed tutorial and todo
workflows (and the library's shortlist check runs on a copy of the library), built the way tests/test_run_listing.py
builds them. The metric arithmetic is checked on hand-built listings, gold sets and rows.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from fastworkflow import run_listing
from fastworkflow.train import score_listing as sl

ROOT = Path(__file__).resolve().parent
TEST_SETS = ROOT / "planning_test_sets"
TUTORIAL = ROOT / "planning_tutorial_workflow"
TODO = ROOT / "todo_list_workflow"
LIBRARY_SET = TEST_SETS / "library.json"
TUTORIAL_SET = TEST_SETS / "tutorial_no_hints.json"
TODO_SET = TEST_SETS / "todo_no_hints.json"


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def test_wilson_interval_known_values():
    lo, hi = sl.wilson(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-3) and hi == pytest.approx(0.7634, abs=1e-3)
    assert sl.wilson(0, 0) == (0.0, 0.0)
    lo, hi = sl.wilson(10, 10)
    assert hi == pytest.approx(1.0) and lo == pytest.approx(0.7225, abs=1e-3)
    lo, hi = sl.wilson(0, 40)
    assert lo == pytest.approx(0.0) and hi == pytest.approx(0.0883, abs=1e-3)


def test_mcnemar_exact_known_values():
    assert sl.mcnemar_exact(0, 0) == 1.0
    assert sl.mcnemar_exact(5, 0) == pytest.approx(0.0625)      # 2 * 0.5**5
    assert sl.mcnemar_exact(3, 3) == pytest.approx(1.0)
    assert sl.mcnemar_exact(0, 1) == pytest.approx(1.0)         # 2 * 0.5 capped at 1
    assert sl.mcnemar_exact(10, 0) == pytest.approx(2 * 0.5 ** 10)


def test_percentile95_is_nearest_rank():
    assert sl.percentile95(list(range(1, 21))) == 19
    assert sl.percentile95([7.0]) == 7.0
    assert sl.percentile95([]) == 0.0


# ---------------------------------------------------------------------------
# Reading a listing: the commands it gives an entry to
# ---------------------------------------------------------------------------

HAND_LISTING = """Commands available in the current context (Library):
- author_books
  Name each book by an author.
  inputs:
  - the author's name, type: str
  template:
  - find_book query=<author>

- go_up
  Change context to the parent of the current context.

Commands available here from the global scope (on your lookup path; run by name, no navigation):
- find_record
  Say whether a book or a member goes by a name.

Commands available after entering the Member context (not callable until then):
- list_member_loans
  List a member's loans.
  This sentence wraps
onto an unindented line and says - nothing about a command.

Global commands (callable from anywhere by name, no navigation): library_hours.

Other contexts (enter one to see and run its commands; a context whose commands are callable here needs no entering):
- Book: commands: show_book, loan_report. To enter: open_book <book_uid> (needs book_uid)
"""


def test_listed_entries_reads_each_section_owner_and_ignores_summaries_and_wrapped_lines():
    entries = sl.listed_entries(HAND_LISTING, "Library")
    assert entries == frozenset({
        "Library/author_books", "Library/go_up", "find_record", "Member/list_member_loans", "library_hours"})
    # a context summary line names commands but is not an entry
    assert "Book/show_book" not in entries and "show_book" not in entries


def test_listed_entries_raises_on_an_unknown_header_rather_than_reading_fewer_commands():
    with pytest.raises(ValueError, match="unrecognised listing header"):
        sl.listed_entries("Commands available somewhere new (Library):\n- go_up\n", "Library")


def test_named_in_is_the_looser_reading():
    text = "Commands available in the current context (Book):\n- list_loans\n  see also show_book\n"
    assert sl.named_in(text, ["Book/list_loans", "Book/show_book", "Book/loan_report"]) == frozenset(
        {"Book/list_loans", "Book/show_book"})


# ---------------------------------------------------------------------------
# Gold sets and the planning distribution, from the test sets
# ---------------------------------------------------------------------------

def test_library_gold_sets_are_the_planning_command_plus_its_callees_or_the_callee_alone():
    requests = sl.load_test_set(LIBRARY_SET)
    assert len(requests) == 143                       # the held-out split only, after the leakage fix (trained rows removed)
    planning = [r for r in requests if r.kind == "planning" and r.expected == "Book/loan_report"]
    assert planning and all(r.gold == frozenset({"Book/loan_report", "Book/list_loans"}) for r in planning)
    callee = [r for r in requests if r.kind == "callee" and r.expected == "Book/list_loans"]
    assert callee and all(r.gold == frozenset({"Book/list_loans"}) for r in callee)


def test_handwritten_gold_sets_come_from_the_workflows_own_inventory():
    requests = sl.load_handwritten(TUTORIAL_SET, str(TUTORIAL))
    book_loans = [r for r in requests if r.expected == "book_loans"]
    assert book_loans and book_loans[0].kind == "planning"
    assert book_loans[0].gold == frozenset({"book_loans", "find_book", "list_loans", "show_book"})
    callee = [r for r in requests if r.expected == "show_book"]
    assert callee and all(r.kind == "callee" and r.gold == frozenset({"show_book"}) for r in callee)


def test_todo_requests_have_no_planning_commands_and_a_single_gold_command():
    requests = sl.load_handwritten(TODO_SET, str(TODO))
    assert requests and all(r.kind == "callee" and len(r.gold) == 1 for r in requests)


def test_planning_distribution_brackets_jev_the_way_the_module_docstring_says():
    planning = sl.Request("who has Dune on loan", "planning", "Library/book_loans", frozenset({"Library/book_loans"}))
    callee = sl.Request("list the loans", "callee", "Book/list_loans", frozenset({"Book/list_loans"}))
    assert sl.planning_distribution(sl.MODE_NONE, planning) is None
    assert sl.planning_distribution(sl.MODE_ORACLE, planning) is None
    assert sl.planning_distribution(sl.MODE_GOLD, planning) == {"Library/book_loans": 0.9}
    assert sl.planning_distribution(sl.MODE_GOLD, callee) == {"none": 0.9}


# ---------------------------------------------------------------------------
# Recall arithmetic, on hand-built rows
# ---------------------------------------------------------------------------

def _row(index, kind, arm, hit, mode=sl.MODE_NONE, chars=100, full_applies=False):
    return sl.RowResult(index=index, kind=kind, mode=mode, arm=arm, hit_entry=hit, hit_strict=hit,
                        hit_named=hit, chars=chars, full_applies=full_applies)


def test_inherited_commands_count_as_reachable_but_not_as_strict_entries():
    """Book inherits Record: Record/show_notes is listed as Book/show_notes, which the agent can call once Book is
    entered. The strict reading (own section only) does not count it; the reachable reading does."""
    bases = {"Book": frozenset({"Record"}), "Member": frozenset({"Record"}), "Record": frozenset()}
    entries = frozenset({"Book/show_notes", "Library/go_up"})
    assert sl.covered(frozenset({"Record/show_notes"}), entries, bases)
    assert not (frozenset({"Record/show_notes"}) <= entries)
    # a command of an unrelated context with the same leaf is not the inherited one
    assert not sl.covered(frozenset({"Record/show_notes"}), frozenset({"Member/show_notes"}), {"Member": frozenset()})
    # a base's command is not covered by an inheriting context's own command of another name
    assert not sl.covered(frozenset({"Book/show_book"}), frozenset({"Record/show_notes"}), bases)
    # a global command and a root-level entry with the same leaf are the same command
    assert sl.covered(frozenset({"library_hours"}), frozenset({"library_hours"}), {})


def test_the_inheritance_declaration_is_read_transitively_from_the_workflow(tmp_path):
    (tmp_path / "_commands").mkdir()
    (tmp_path / "_commands" / "context_inheritance_model.json").write_text(json.dumps({
        "Book": {"base": ["Media"]}, "Media": {"base": ["Record"]}, "Record": {}}))
    bases = sl.inheritance_bases(tmp_path)
    assert bases["Book"] == frozenset({"Media", "Record"})
    assert bases["Record"] == frozenset()
    assert sl.inheritance_bases(tmp_path / "missing") == {}


def test_recall_arithmetic_per_arm_with_intervals_and_paired_mcnemar():
    requests = [
        sl.Request("a", "planning", "P", frozenset({"P"})),
        sl.Request("b", "planning", "P", frozenset({"P"})),
        sl.Request("c", "callee", "C", frozenset({"C"})),
        sl.Request("d", "callee", "C", frozenset({"C"})),
    ]
    full = [True, True, False, True]          # 3/4
    command = [True, False, False, False]     # 1/4
    rows = []
    for i, req in enumerate(requests):
        rows.append(_row(i, req.kind, "full", full[i], mode=sl.MODE_NONE, chars=6000))
        rows.append(_row(i, req.kind, "command_shortlist", command[i], mode=sl.MODE_NONE, chars=1000))
        for arm in ("full", "command_shortlist"):   # the other modes are not under test here
            rows.append(_row(i, req.kind, arm, True, mode=sl.MODE_GOLD))
            rows.append(_row(i, req.kind, arm, True, mode=sl.MODE_ORACLE))
    summary = sl.summarise(rows, requests)["none"]

    assert summary["full"]["hits"] == 3 and summary["full"]["n"] == 4
    assert summary["full"]["recall"] == pytest.approx(0.75)
    assert summary["full"]["ci95"] == [pytest.approx(v) for v in sl.wilson(3, 4)]
    assert summary["command_shortlist"]["recall"] == pytest.approx(0.25)
    assert summary["command_shortlist"]["by_kind"]["planning"] == {
        "n": 2, "hits": 1, "recall": 0.5, "ci95": list(sl.wilson(1, 2))}
    assert summary["command_shortlist"]["by_kind"]["callee"]["recall"] == 0.0
    assert summary["command_shortlist"]["chars_mean"] == 1000
    assert summary["full"]["tokens_mean"] == 1500          # chars / 4
    # paired discordance: full hit where command missed on requests 0 and 3 (2), command hit where full missed (0)
    assert summary["mcnemar"]["full_vs_command_shortlist"] == {
        "a_hit_b_missed": 2, "b_hit_a_missed": 0, "p_two_sided": pytest.approx(0.5)}
    # the BY_CONTEXT arm is removed (10.15): no by_context summary or pairing is produced
    assert "by_context" not in summary
    assert "full_vs_by_context" not in summary["mcnemar"]


def test_the_full_listing_applies_share_and_the_size_stats_come_from_the_rows():
    requests = [sl.Request("x", "callee", "C", frozenset({"C"})) for _ in range(3)]
    rows = [_row(0, "callee", "full", True, chars=8000, full_applies=False),
            _row(1, "callee", "full", True, chars=8000, full_applies=False),
            _row(2, "callee", "full", True, chars=8000, full_applies=False),
            _row(0, "callee", "command_shortlist", True, chars=1200, full_applies=False),
            _row(1, "callee", "command_shortlist", True, chars=4000, full_applies=True),
            _row(2, "callee", "command_shortlist", True, chars=1600, full_applies=False)]
    for mode in (sl.MODE_GOLD, sl.MODE_ORACLE):
        for i in range(3):
            for arm in ("full", "command_shortlist"):
                rows.append(_row(i, "callee", arm, True, mode=mode))
    summary = sl.summarise(rows, requests)["none"]
    assert summary["command_shortlist"]["full_applies_share"] == pytest.approx(1 / 3)
    assert summary["command_shortlist"]["chars_p95"] == 4000
    assert summary["command_shortlist"]["chars_mean"] == pytest.approx((1200 + 4000 + 1600) / 3)


# ---------------------------------------------------------------------------
# Integration: the scorer on the committed tutorial and todo copies
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tutorial_run(tmp_path_factory):
    spec = sl.Spec("tutorial_no_hints", TUTORIAL, TUTORIAL_SET, True)
    scratch = tmp_path_factory.mktemp("score_listing_tutorial")
    return sl.run_spec(spec, scratch, (3000, 5000, 9000))


def test_tutorial_has_no_producers_or_navigation_and_its_full_listing_is_over_the_threshold(tutorial_run):
    assert tutorial_run["requests"] == 47
    assert tutorial_run["by_kind"] == {"planning": 33, "callee": 14}
    # the size moves with the tree (other agents add internal commands); the threshold is the production constant
    assert tutorial_run["threshold"] == run_listing.FULL_LISTING_MAX_CHARS
    assert tutorial_run["full_listing_chars"] > 0


def test_full_listing_recall_is_one_and_the_full_listing_applies_only_within_the_threshold(tutorial_run):
    summary = tutorial_run["summary"]
    assert summary[sl.MODE_NONE]["full"]["recall"] == 1.0
    # the full listing applies to a request exactly when its characters are at most the production threshold
    fits = tutorial_run["full_listing_chars"] <= run_listing.FULL_LISTING_MAX_CHARS
    assert summary[sl.MODE_NONE]["full"]["full_applies_share"] == (1.0 if fits else 0.0)
    sweep = tutorial_run["sweep"]["thresholds"]
    assert sweep["3000"]["full_applies_by_context"]["*"] is False
    assert sweep["9000"]["full_applies_by_context"]["*"] is True
    # at 9000 the full listing applies to every request, so each arm's recall is the full listing's
    assert sweep["9000"]["recall"][sl.MODE_GOLD]["command_shortlist"] == sweep["9000"]["recall"][sl.MODE_GOLD]["full"]
    assert sweep["3000"]["recall"][sl.MODE_NONE]["command_shortlist"] == 0.0


def test_the_oracle_seeds_bound_the_shortlist_from_above_on_this_workflow(tutorial_run):
    oracle = tutorial_run["summary"][sl.MODE_ORACLE]
    assert oracle["command_shortlist"]["recall"] == 1.0


def test_without_a_planning_match_the_exact_name_gather_misses_natural_requests(tutorial_run):
    """Offline gathering reads exact names only (the per-scope head answers need live heads): recall is the floor."""
    # the production threshold now gives the tutorial's full listing whole (6,353 < 7,000), so the shortlist's own
    # floor is read from the sweep row where the threshold forces the shortlist
    assert tutorial_run["sweep"]["thresholds"]["3000"]["recall"][sl.MODE_NONE]["command_shortlist"] == 0.0


def test_the_scorer_reuses_cached_texts_and_matches_production_at_the_production_threshold():
    """The threshold is applied to the cached texts; at the production constant that must be run_listing_text
    under the production policy (no experiment arm)."""
    scratch = Path(tempfile.mkdtemp(prefix="score_listing_check_"))
    spec = sl.Spec("tutorial_check", TUTORIAL, TUTORIAL_SET, True)
    opened = sl.open_workflow(spec.folder, spec.label, scratch)
    try:
        requests = sl.load_handwritten(TUTORIAL_SET, str(spec.folder))[:6]
        for request in requests:
            for mode in (sl.MODE_NONE, sl.MODE_GOLD):
                built = sl.build_listings(opened, request, mode)
                expected = run_listing.run_listing_text(
                    opened.app, request.utterance, arm=None,
                    planning_probabilities=sl.planning_distribution(mode, request))
                got = sl.arm_text(built, "command_shortlist", len(built.full), run_listing.FULL_LISTING_MAX_CHARS)
                assert got == expected, (request.utterance, mode)
    finally:
        opened.close()


def test_head_seeds_take_each_heads_answer_and_planning_scores_from_the_top_k(tmp_path):
    """Hand-built head answers (data, not a model): the labels a head answers on the gather contexts, the planning
    labels among its top-k scores, and the non-command labels left out."""
    spec = sl.Spec("tutorial_heads_check", TUTORIAL, TUTORIAL_SET, True)
    opened = sl.open_workflow(spec.folder, spec.label, tmp_path)
    try:
        opened.heads = {"*": {"who has Dune": {
            "labels": ["book_loans"], "topk_labels": ["book_loans", "wildcard", "find_book"],
            "topk_scores": [0.8, 0.1, 0.05], "confident": True}}}
        labels, planning = sl.head_seeds(opened, "who has Dune")
        assert labels == {"book_loans"}
        assert planning == {"book_loans": 0.8}           # find_book is not a planning command; wildcard is not one
        built = sl.build_listings(opened, sl.Request("who has Dune", "planning", "book_loans",
                                                     frozenset({"book_loans"})), sl.MODE_HEADS)
        assert "- book_loans" in built.command_shortlist or "- book_loans" in built.full
        assert "book_loans" in built.shortlist
    finally:
        opened.close()


def test_a_workflow_without_trained_heads_has_no_head_modes_and_says_so(tmp_path):
    """The committed tutorial and todo copies are not trained, so the realistic-gathering modes are not measured."""
    assert sl.source_head_answers(TUTORIAL, ["who has Dune"]) is None
    result = sl.run_spec(sl.Spec("tutorial_no_heads", TUTORIAL, TUTORIAL_SET, True), tmp_path, None)
    assert result["trained_heads"] is None
    assert set(result["summary"]) == set(sl.MODES)


def test_the_run_is_deterministic_in_process(tmp_path):
    spec = sl.Spec("tutorial_no_hints", TUTORIAL, TUTORIAL_SET, True)
    first = sl.run_spec(spec, tmp_path / "a", (5000,))
    second = sl.run_spec(spec, tmp_path / "b", (5000,))
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_the_run_is_deterministic_across_hash_seeds(tmp_path):
    """The same offline output with two PYTHONHASHSEED values: no result depends on set iteration order."""
    outputs = []
    for seed in ("1", "2"):
        out = tmp_path / f"seed{seed}"
        env = dict(os.environ, PYTHONHASHSEED=seed)
        subprocess.run(
            [sys.executable, "-m", "fastworkflow.train.score_listing", "offline", "--out", str(out),
             "--only", "tutorial_no_hints,todo_no_hints"],
            check=True, env=env, cwd=str(ROOT.parent), capture_output=True, timeout=600)
        outputs.append(json.loads((out / "offline.json").read_text()))
    assert outputs[0] == outputs[1]


def test_todo_is_under_the_threshold_so_every_arm_gives_the_full_listing(tmp_path):
    spec = sl.Spec("todo_no_hints", TODO, TODO_SET, True)
    result = sl.run_spec(spec, tmp_path, None)
    assert result["full_listing_chars"] <= run_listing.FULL_LISTING_MAX_CHARS
    for mode in sl.MODES:
        for arm in sl.ARM_NAMES:
            assert result["summary"][mode][arm]["full_applies_share"] == 1.0


# ---------------------------------------------------------------------------
# The live-run estimate: arithmetic only
# ---------------------------------------------------------------------------

def test_stratified_sample_takes_per_command_planning_requests_and_each_callee_once():
    callees = {"P1": ["C1", "C2"], "P2": ["C2"]}
    requests = []
    for p in ("P1", "P2"):
        for i in range(6):
            requests.append(sl.Request(f"{p} req {i}", "planning", p, frozenset({p})))
    for c in ("C1", "C2"):
        for i in range(6):
            requests.append(sl.Request(f"{c} req {i}", "callee", c, frozenset({c})))
    sample = sl.stratified_sample(requests, 2, callees, seed=7)
    again = sl.stratified_sample(requests, 2, callees, seed=7)
    assert [r.utterance for r in sample] == [r.utterance for r in again]
    planning = [r for r in sample if r.kind == "planning"]
    callee = [r for r in sample if r.kind == "callee"]
    assert len(planning) == 4 and len(callee) == 4          # C2 is shared by P1 and P2 but sampled once
    assert {r.expected for r in callee} == {"C1", "C2"}


def test_estimate_is_calls_times_tokens_and_is_labelled_as_a_dry_run():
    sizes = {arm: {"tokens_mean": t} for arm, t in (("full", 1600.0), ("command_shortlist", 300.0))}
    est = sl.estimate_live("x", sizes, 10, calls_per_request=3, overhead_tokens=1500, output_tokens=300)
    assert est["arms"]["full"]["calls"] == 30
    assert est["arms"]["full"]["prompt_tokens_total"] == round(30 * (1500 + 1600))
    assert est["arms"]["command_shortlist"]["output_tokens_total"] == 30 * 300
    assert est["assumptions"]["calls_per_request"] == 3


def test_estimate_command_writes_json_marked_not_live_and_makes_no_model_call(tmp_path, capsys):
    rc = sl.main(["estimate", "--out", str(tmp_path), "--only", "tutorial_no_hints"])
    assert rc == 0
    data = json.loads((tmp_path / "estimate.json").read_text())
    assert data["live"] is False
    (est,) = data["estimates"]
    assert est["arms"]["full"]["requests"] == est["arms"]["command_shortlist"]["requests"]
    assert "by_context" not in est["arms"]
    assert est["arms"]["full"]["calls"] == est["arms"]["full"]["requests"] * 3
    assert "DRY RUN" in capsys.readouterr().out

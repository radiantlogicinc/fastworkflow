"""Integration tests for held-out intent evaluation.

Per the repo testing rules these are integration tests against the real module: no Mock
fixtures and no patching of fastWorkflow components. The labeled utterances, benchmark
files and `predict_fn` callables below are plain test *inputs* — the system under test is
the real `fastworkflow.train.heldout_evaluation` module, exercised end to end. Keeping
`predict_fn` a plain callable is the point of the design: it makes the evaluation logic
testable without torch, a trained model, network access, or API keys.
"""

import json
import logging

import pytest

from collections import Counter
from pathlib import Path

from fastworkflow.model_pipeline_training import (
    TrainingDataError,
    _score_heldout_context,
    split_training_data,
)
from fastworkflow.utils.logging import logger as fastworkflow_logger
from fastworkflow.nlu_labels import WILDCARD_LABEL
from fastworkflow.train.selective_training import _recompute_heldout_totals
from fastworkflow.train.heldout_evaluation import (
    REPORT_SCHEMA_VERSION,
    SEED_PERSONA_ID,
    UNRESOLVED_PERSONA_PREFIX,
    BenchmarkCase,
    BenchmarkLeakError,
    EscalationScore,
    HeldoutReport,
    LabeledUtterance,
    MIN_TRAINING_ROWS_PER_LABEL,
    RoutingScore,
    aggregate_totals,
    assert_benchmark_disjoint_from_seeds,
    benchmark_cases_for_context,
    find_near_duplicate_benchmark_cases,
    expand_persona_id,
    format_report,
    labeled_utterances_from_provenance,
    load_benchmark_file,
    normalize_utterance,
    partition_by_routability,
    score_escalation,
    score_routing,
    split_by_persona,
    validate_escalation_cases,
    validate_routing_cases,
    write_report,
)


# ---------------------------------------------------------------------------
# Helpers (plain test data builders, not fixtures of the system under test)
# ---------------------------------------------------------------------------


def _records(spec):
    """Build LabeledUtterance records from (persona, label, [utterances]) triples."""
    return [
        LabeledUtterance(utterance=utterance, label=label, persona=persona)
        for persona, label, utterances in spec
        for utterance in utterances
    ]


def _multi_persona_records(num_personas=8, labels=("Ctx/a", "Ctx/b", "Ctx/c")):
    spec = []
    for index in range(num_personas):
        persona = f"persona_{index:02d}"
        for label in labels:
            spec.append(
                (
                    persona,
                    label,
                    [f"{label} phrasing {index}-{n}" for n in range(3)],
                )
            )
    return _records(spec)


def _lookup_predictor(table, default=None):
    """A predict_fn backed by a plain dict: utterance -> ranked candidate labels."""

    def predict(utterance):
        return list(table.get(utterance, default if default is not None else []))

    return predict


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def test_normalize_utterance_folds_case_whitespace_and_trailing_punctuation():
    assert normalize_utterance("  Close   the account.  ") == "close the account"
    assert normalize_utterance("close the account") == "close the account"
    assert normalize_utterance("Close the account!!") == "close the account"
    assert normalize_utterance('"Close the account"') == "close the account"
    # Curly apostrophes are folded so a copy-paste from a doc still collides with a seed.
    assert normalize_utterance("Don\u2019t close it.") == normalize_utterance("don't close it")


# ---------------------------------------------------------------------------
# Persona splitting (R1a / decision D1)
# ---------------------------------------------------------------------------


def test_split_by_persona_is_deterministic_for_a_fixed_seed():
    records = _multi_persona_records()

    first = split_by_persona(records, holdout_fraction=0.25, seed=42)
    second = split_by_persona(records, holdout_fraction=0.25, seed=42)

    assert first.heldout_personas == second.heldout_personas
    assert [r.utterance for r in first.heldout] == [r.utterance for r in second.heldout]
    assert [r.utterance for r in first.train] == [r.utterance for r in second.train]
    # 8 personas at 25% reserves 2 of them, and never all of them.
    assert len(first.heldout_personas) == 2
    assert set(first.train_personas).isdisjoint(first.heldout_personas)


def test_split_by_persona_varies_with_the_seed():
    records = _multi_persona_records()
    selections = {
        tuple(split_by_persona(records, holdout_fraction=0.25, seed=s).heldout_personas)
        for s in range(20)
    }
    assert len(selections) > 1


def test_heldout_rows_come_only_from_heldout_personas():
    records = _multi_persona_records()
    split = split_by_persona(records, holdout_fraction=0.25, seed=7)

    heldout_set = set(split.heldout_personas)
    assert heldout_set
    assert {r.persona for r in split.heldout} <= heldout_set
    assert heldout_set.isdisjoint({r.persona for r in split.train})
    assert len(split.train) + len(split.heldout) == len(records)


def test_composite_persona_id_is_held_out_only_when_every_contributor_is():
    # The determinism module attributes one utterance text to every persona that produced
    # it, joining ids with "+". Treating "p1+p2" as an opaque persona of its own would let
    # it be reserved for evaluation while p2's other rows trained the model — the leak D1
    # exists to prevent. It must land in train unless BOTH contributors are held out.
    records = _records(
        [
            ("p1", "Ctx/a", ["a one", "a two"]),
            ("p2", "Ctx/a", ["a three", "a four"]),
            ("p3", "Ctx/a", ["a five", "a six"]),
            ("p4", "Ctx/a", ["a seven", "a eight"]),
            ("p1+p2", "Ctx/a", ["shared phrasing from p1 and p2"]),
        ]
    )

    split = split_by_persona(records, holdout_fraction=0.25, seed=7)

    # The composite id never becomes a persona in its own right.
    assert "p1+p2" not in split.heldout_personas
    assert "p1+p2" not in split.train_personas
    assert set(split.heldout_personas) <= {"p1", "p2", "p3", "p4"}

    heldout_set = set(split.heldout_personas)
    composite_rows = [r for r in split.heldout if r.persona == "p1+p2"]
    if {"p1", "p2"} <= heldout_set:
        assert composite_rows, "both contributors reserved; the row is legitimately held out"
    else:
        assert not composite_rows, (
            f"composite row leaked into holdout with contributors {heldout_set} reserved"
        )


def test_unresolved_persona_id_expands_to_every_contributor():
    resolved = expand_persona_id("1041+37822")
    assert resolved == frozenset({"1041", "37822"})

    unresolved = expand_persona_id(f"{UNRESOLVED_PERSONA_PREFIX}1041+37822")
    assert unresolved == frozenset({"1041", "37822"})

    unresolved_after_separator = expand_persona_id(
        f"1041+{UNRESOLVED_PERSONA_PREFIX}37822"
    )
    assert unresolved_after_separator == frozenset({"1041", "37822"})

    unresolved_on_each_atom = expand_persona_id(
        f"{UNRESOLVED_PERSONA_PREFIX}1041+{UNRESOLVED_PERSONA_PREFIX}37822"
    )
    assert unresolved_on_each_atom == frozenset({"1041", "37822"})

    assert expand_persona_id("1041") == frozenset({"1041"})
    assert expand_persona_id(SEED_PERSONA_ID) == frozenset({SEED_PERSONA_ID})


def test_mid_composite_unresolved_prefix_never_becomes_a_holdout_persona():
    """Legacy merge shapes must reserve real contributors, never prefix-bearing atoms."""
    mixed_composite = f"p1+{UNRESOLVED_PERSONA_PREFIX}p2"
    records = _records(
        [
            ("p1", "Ctx/a", ["p1 one", "p1 two"]),
            ("p2", "Ctx/a", ["p2 one", "p2 two"]),
            ("p3", "Ctx/a", ["p3 one", "p3 two"]),
            ("p4", "Ctx/a", ["p4 one", "p4 two"]),
            (mixed_composite, "Ctx/a", ["shared unresolved wording"]),
        ]
    )

    split = split_by_persona(records, holdout_fraction=0.25, seed=42)

    assert len(split.heldout_personas) == 1
    assert set(split.train_personas + split.heldout_personas) == {
        "p1", "p2", "p3", "p4"
    }
    assert all(
        not persona.startswith(UNRESOLVED_PERSONA_PREFIX)
        for persona in split.train_personas + split.heldout_personas
    )
    assert "shared unresolved wording" in {
        record.utterance for record in split.train
    }


def test_seed_persona_records_never_land_in_holdout():
    records = _multi_persona_records(num_personas=4)
    seeds = _records(
        [
            (SEED_PERSONA_ID, "Ctx/a", ["a", "hand written seed for a"]),
            (SEED_PERSONA_ID, "Ctx/b", ["b", "hand written seed for b"]),
        ]
    )

    split = split_by_persona(records + seeds, holdout_fraction=0.25, seed=42)

    assert SEED_PERSONA_ID not in {r.persona for r in split.heldout}
    assert SEED_PERSONA_ID not in split.heldout_personas
    assert SEED_PERSONA_ID not in split.train_personas
    train_texts = {r.utterance for r in split.train}
    for seed_record in seeds:
        assert seed_record.utterance in train_texts


def test_label_that_would_have_zero_training_rows_is_returned_to_train():
    # Two personas, one reserved. "Ctx/only_p1" and "Ctx/only_p2" are each produced by a
    # single persona, so whichever persona is reserved starves one of them. A label with
    # zero training rows is unroutable - the same class of bug as F3 - so the guard must
    # return its rows to train rather than buy a metric with a broken model.
    records = _records(
        [
            ("p1", "Ctx/shared", ["shared one", "shared two"]),
            ("p2", "Ctx/shared", ["shared three", "shared four"]),
            ("p1", "Ctx/only_p1", ["only p1 says this", "and this"]),
            ("p2", "Ctx/only_p2", ["only p2 says this", "and that"]),
        ]
    )

    split = split_by_persona(records, holdout_fraction=0.5, seed=42)

    assert len(split.heldout_personas) == 1
    train_labels = {r.label for r in split.train}
    assert {"Ctx/shared", "Ctx/only_p1", "Ctx/only_p2"} == train_labels

    starved = "Ctx/only_p1" if split.heldout_personas == ["p1"] else "Ctx/only_p2"
    assert starved not in {r.label for r in split.heldout}
    # The note now says "below the trainer's floor" rather than "zero training rows":
    # the rescue threshold was raised from 0 to the trainer's floor of 2 so the two
    # modules stop disagreeing. Zero is still rescued -- it is below the floor.
    assert any("floor" in note for note in split.notes)
    assert any(starved in note for note in split.notes)


def test_leaked_utterance_is_dropped_from_holdout():
    # The same phrasing produced by a training persona and a held-out persona is not
    # held-out data. Normalisation catches the punctuation-only variant too.
    records = _records(
        [
            ("p1", "Ctx/a", ["a one", "a two", "a three", "shared phrasing"]),
            ("p2", "Ctx/a", ["a four", "a five", "a six", "Shared phrasing."]),
            ("p3", "Ctx/a", ["a seven", "a eight", "a nine", "shared phrasing"]),
            ("p1", "Ctx/b", ["b one", "b two"]),
            ("p2", "Ctx/b", ["b three", "b four"]),
            ("p3", "Ctx/b", ["b five", "b six"]),
        ]
    )

    split = split_by_persona(records, holdout_fraction=0.34, seed=42)

    assert len(split.heldout_personas) == 1
    heldout_texts = {normalize_utterance(r.utterance) for r in split.heldout}
    assert "shared phrasing" not in heldout_texts
    assert any("also produced by a training persona" in note for note in split.notes)
    # The non-leaked rows of the held-out persona survive.
    assert len(split.heldout) == 5


def test_single_persona_context_holds_nothing_out_and_says_so():
    records = _records([("p1", "Ctx/a", ["one", "two", "three"])])

    split = split_by_persona(records, holdout_fraction=0.25, seed=42)

    assert split.heldout == []
    assert split.heldout_personas == []
    assert len(split.train) == 3
    assert any("fewer than 2 personas" in note for note in split.notes)
    assert any("no personas could be reserved" in note for note in split.notes)


def test_labels_without_heldout_coverage_are_reported():
    records = _records(
        [
            ("p1", "Ctx/a", ["a one", "a two"]),
            ("p2", "Ctx/a", ["a three", "a four"]),
            ("p3", "Ctx/a", ["a five", "a six"]),
            ("p4", "Ctx/a", ["a seven", "a eight"]),
            (SEED_PERSONA_ID, "Ctx/seed_only", ["seed only phrasing"]),
        ]
    )

    split = split_by_persona(records, holdout_fraction=0.25, seed=42)

    assert any("no held-out rows" in note for note in split.notes)
    assert any("Ctx/seed_only" in note for note in split.notes)


def test_empty_record_set_is_reported_not_crashed():
    split = split_by_persona([], holdout_fraction=0.25, seed=42)
    assert split.train == []
    assert split.heldout == []
    assert any("nothing to hold out" in note for note in split.notes)


def test_labeled_utterances_from_provenance_accepts_key_aliases():
    entries = [
        {"utterance": "close it", "label": "Ctx/close", "persona": "p1"},
        {"text": "shut it down", "command": "Ctx/close", "persona_id": "p2"},
        {"utterance": "no label here"},
    ]

    records = labeled_utterances_from_provenance(entries)

    assert len(records) == 2
    assert records[1].utterance == "shut it down"
    assert records[1].label == "Ctx/close"
    assert records[1].persona == "p2"


# ---------------------------------------------------------------------------
# Routing scoring
# ---------------------------------------------------------------------------


def test_score_routing_separates_top1_from_in_list():
    cases = [
        LabeledUtterance(utterance="u_top1", label="Ctx/a", persona="p1"),
        LabeledUtterance(utterance="u_first_ranked", label="Ctx/a", persona="p1"),
        LabeledUtterance(utterance="u_second", label="Ctx/a", persona="p1"),
        LabeledUtterance(utterance="u_third", label="Ctx/b", persona="p1"),
        LabeledUtterance(utterance="u_miss", label="Ctx/b", persona="p1"),
    ]
    predict = _lookup_predictor(
        {
            "u_top1": ["Ctx/a"],
            # A correct first-ranked label in a multi-candidate result is still an
            # ambiguity prompt at runtime, so it is in-list but not a routed top-1.
            "u_first_ranked": ["Ctx/a", "Ctx/b", "wildcard"],
            # Correct label in 2nd place: in-list but NOT top-1. This is a clarification
            # prompt at runtime, not a correct route, and the two must not be conflated.
            "u_second": ["Ctx/b", "Ctx/a", "wildcard"],
            "u_third": ["wildcard", "Ctx/a", "Ctx/b"],
            "u_miss": ["Ctx/a", "wildcard"],
        }
    )

    score = score_routing(cases, predict)

    assert score.total == 5
    assert score.top1_correct == 1
    assert score.in_list_correct == 4
    assert score.top1 == pytest.approx(0.2)
    assert score.in_list == pytest.approx(0.8)
    assert score.per_command["Ctx/a"] == {
        "total": 3,
        "top1_correct": 1,
        "in_list_correct": 3,
    }
    assert score.per_command["Ctx/b"] == {
        "total": 2,
        "top1_correct": 0,
        "in_list_correct": 1,
    }


def test_score_routing_handles_empty_prediction_and_empty_case_list():
    score = score_routing(
        [LabeledUtterance(utterance="u", label="Ctx/a", persona="p1")],
        _lookup_predictor({}),
    )
    assert score.total == 1
    assert score.top1_correct == 0
    assert score.in_list_correct == 0

    empty = score_routing([], _lookup_predictor({}))
    assert empty.total == 0
    assert empty.top1 == 0.0
    assert empty.in_list == 0.0


def test_score_routing_accepts_benchmark_cases():
    cases = [
        BenchmarkCase(context="Ctx", utterance="u1", expected_label="Ctx/a"),
        BenchmarkCase(context="Ctx", utterance="u2", expected_label="Ctx/b"),
    ]
    score = score_routing(cases, _lookup_predictor({"u1": ["Ctx/a"], "u2": ["Ctx/a"]}))
    assert score.top1_correct == 1
    assert score.total == 2


def test_score_routing_rejects_a_case_with_no_expected_label():
    case = BenchmarkCase(context="Ctx", utterance="u1", kind="escalation",
                         expected_ancestor_command="Parent/cmd")
    with pytest.raises(ValueError):
        score_routing([case], _lookup_predictor({}))


# ---------------------------------------------------------------------------
# Escalation scoring (F10 / decision D2)
# ---------------------------------------------------------------------------


def test_score_escalation_requires_a_lone_confident_escalation_label():
    cases = [
        BenchmarkCase(
            context="ReviewTicket",
            utterance="escalates cleanly",
            kind="escalation",
            expected_ancestor_command="AccessReviewWorkspace/bulk_decide",
        ),
        BenchmarkCase(
            context="ReviewTicket",
            utterance="wildcard buried in topk",
            kind="escalation",
            expected_ancestor_command="AccessReviewWorkspace/bulk_decide",
        ),
        BenchmarkCase(
            context="ReviewTicket",
            utterance="routed locally instead",
            kind="escalation",
            expected_ancestor_command="AccessReviewWorkspace/bulk_decide",
        ),
    ]
    predict = _lookup_predictor(
        {
            "escalates cleanly": ["wildcard"],
            # A wildcard alongside local candidates takes the ambiguity branch at runtime
            # and the escalation signal is discarded (F7), so this must not score as a pass.
            "wildcard buried in topk": [
                "wildcard",
                "ReviewTicket/certify_approve",
                "ReviewTicket/show_review_item",
            ],
            "routed locally instead": ["ReviewTicket/certify_approve"],
        }
    )

    score = score_escalation(cases, predict)

    assert score.total == 3
    assert score.correct == 1
    assert score.recall == pytest.approx(1 / 3)
    assert len(score.failures) == 2
    reasons = [failure["reason"] for failure in score.failures]
    assert any("not alone" in reason for reason in reasons)
    assert any("no escalation label" in reason for reason in reasons)
    assert score.failures[0]["expected_ancestor_command"] == (
        "AccessReviewWorkspace/bulk_decide"
    )
    assert score.failures[0]["context"] == "ReviewTicket"


def test_score_escalation_label_set_is_a_parameter():
    cases = [
        BenchmarkCase(
            context="Ctx",
            utterance="u1",
            kind="escalation",
            expected_ancestor_command="Parent/cmd",
        ),
        BenchmarkCase(
            context="Ctx",
            utterance="u2",
            kind="escalation",
            expected_ancestor_command="Parent/cmd",
        ),
    ]
    predict = _lookup_predictor({"u1": ["wildcard"], "u2": ["other_escalation"]})

    default_only = score_escalation(cases, predict)
    assert default_only.correct == 1

    widened = score_escalation(
        cases, predict, escalation_labels={"wildcard", "other_escalation"}
    )
    assert widened.correct == 2
    assert widened.recall == pytest.approx(1.0)
    assert widened.failures == []


def test_score_escalation_with_no_cases():
    score = score_escalation([], _lookup_predictor({}))
    assert score.total == 0
    assert score.recall == 0.0


# ---------------------------------------------------------------------------
# Benchmark file (R1b)
# ---------------------------------------------------------------------------


def _write_benchmark(tmp_path, payload):
    path = tmp_path / "intent_benchmark.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def test_load_benchmark_file_round_trip(tmp_path):
    payload = {
        "schema_version": 1,
        "cases": [
            {
                "context": "Account",
                "utterance": "wind this account down for me",
                "expected_label": "Account/close_account",
                "kind": "routing",
            },
            {
                "context": "ReviewTicket",
                "utterance": "approve everything from this app at once",
                "kind": "escalation",
                "expected_ancestor_command": "AccessReviewWorkspace/bulk_decide",
            },
        ],
    }
    path = _write_benchmark(tmp_path, payload)

    cases = load_benchmark_file(path)

    assert len(cases) == 2
    assert cases[0].kind == "routing"
    assert cases[0].expected_label == "Account/close_account"
    assert cases[0].expected_ancestor_command is None
    assert cases[1].kind == "escalation"
    assert cases[1].expected_ancestor_command == "AccessReviewWorkspace/bulk_decide"

    routing = benchmark_cases_for_context(cases, "Account", "routing")
    assert [c.utterance for c in routing] == ["wind this account down for me"]
    assert benchmark_cases_for_context(cases, "Account", "escalation") == []


def test_load_benchmark_file_accepts_a_bare_list(tmp_path):
    path = _write_benchmark(
        tmp_path,
        [{"context": "Account", "utterance": "u", "expected_label": "Account/x"}],
    )
    cases = load_benchmark_file(path)
    assert len(cases) == 1
    assert cases[0].kind == "routing"


def test_load_benchmark_file_rejects_incomplete_cases(tmp_path):
    path = _write_benchmark(
        tmp_path,
        {"schema_version": 1, "cases": [{"context": "Account", "utterance": "u"}]},
    )
    with pytest.raises(ValueError) as excinfo:
        load_benchmark_file(path)
    assert "expected_label" in str(excinfo.value)


def test_load_benchmark_file_rejects_an_unknown_schema_version(tmp_path):
    path = _write_benchmark(tmp_path, {"schema_version": 99, "cases": []})
    with pytest.raises(ValueError) as excinfo:
        load_benchmark_file(path)
    assert "schema_version" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Unsupported out-of-scope benchmark guard (fix-d28)
# ---------------------------------------------------------------------------


def test_load_benchmark_file_rejects_out_of_scope_as_unsupported(tmp_path):
    path = _write_benchmark(
        tmp_path,
        {
            "schema_version": 1,
            "cases": [
                {
                    "context": "global",
                    "utterance": "tell me a joke",
                    "kind": "out_of_scope",
                }
            ],
        },
    )
    with pytest.raises(
        ValueError,
        match="Out-of-scope scoring is not supported; see fix-d28 and its design work",
    ):
        load_benchmark_file(path)


def test_assert_benchmark_disjoint_raises_on_a_normalised_near_match():
    # The realistic leak: a failing benchmark case gets pasted into the seed list with the
    # capitalisation and trailing period changed. An exact string match would miss it.
    cases = [
        BenchmarkCase(
            context="Account",
            utterance="Close the account.",
            expected_label="Account/close_account",
        )
    ]
    seeds = {"Account/close_account": ["close the account", "shut it down"]}

    with pytest.raises(BenchmarkLeakError) as excinfo:
        assert_benchmark_disjoint_from_seeds(cases, seeds)

    message = str(excinfo.value)
    assert "Close the account." in message
    assert "Account/close_account" in message
    assert "1 benchmark utterance" in message


def test_assert_benchmark_disjoint_lists_every_overlap():
    cases = [
        BenchmarkCase(context="A", utterance="One.", expected_label="A/one"),
        BenchmarkCase(context="A", utterance="  TWO  ", expected_label="A/two"),
        BenchmarkCase(context="A", utterance="genuinely new phrasing", expected_label="A/one"),
    ]
    seeds = {"A/one": ["one"], "A/two": ["two"]}

    with pytest.raises(BenchmarkLeakError) as excinfo:
        assert_benchmark_disjoint_from_seeds(cases, seeds)

    assert "2 benchmark utterance" in str(excinfo.value)


def test_assert_benchmark_disjoint_passes_when_disjoint():
    cases = [
        BenchmarkCase(
            context="Account",
            utterance="wind this account down for me",
            expected_label="Account/close_account",
        )
    ]
    seeds = {"Account/close_account": ["close the account", "shut it down"]}

    # `assert`, not a bare expression: without it the `is None` comparison is dead code
    # and the line asserts nothing at all. The raise-or-not is still what this test is
    # really about, but a silently discarded comparison is how a reader is misled about
    # which of the two is being checked. bd fix-k0i.48.
    assert assert_benchmark_disjoint_from_seeds(cases, seeds) is None


def test_near_duplicate_benchmark_cases_are_warned_about_not_raised():
    cases = [
        BenchmarkCase(
            context="Account",
            utterance="close the accounts",
            expected_label="Account/close_account",
        )
    ]
    seeds = {"Account/close_account": ["close the account"]}

    assert assert_benchmark_disjoint_from_seeds(cases, seeds) is None

    warnings = find_near_duplicate_benchmark_cases(cases, seeds)
    assert len(warnings) == 1
    assert "close the accounts" in warnings[0]
    assert "Account/close_account" in warnings[0]


# ---------------------------------------------------------------------------
# Structural validation of benchmark cases
# ---------------------------------------------------------------------------


_LABEL_SPACE = {
    "ReviewTicket": {
        "ReviewTicket/certify_approve",
        "ReviewTicket/show_review_item",
        "wildcard",
    },
    "AccessReviewWorkspace": {
        "AccessReviewWorkspace/bulk_decide",
        "AccessReviewWorkspace/list_items",
    },
}
_ANCESTOR_MAP = {"ReviewTicket": ["AccessReviewWorkspace"], "AccessReviewWorkspace": []}


def test_validate_escalation_cases_accepts_a_structurally_valid_case():
    cases = [
        BenchmarkCase(
            context="ReviewTicket",
            utterance="approve everything from this app at once",
            kind="escalation",
            expected_ancestor_command="AccessReviewWorkspace/bulk_decide",
        )
    ]
    assert validate_escalation_cases(cases, _LABEL_SPACE, _ANCESTOR_MAP) == []


def test_validate_escalation_cases_rejects_a_command_present_in_the_tested_context():
    cases = [
        BenchmarkCase(
            context="ReviewTicket",
            utterance="approve this one",
            kind="escalation",
            expected_ancestor_command="ReviewTicket/certify_approve",
        )
    ]
    problems = validate_escalation_cases(cases, _LABEL_SPACE, _ANCESTOR_MAP)
    assert len(problems) == 1
    assert "IS in the tested context's label space" in problems[0]


def test_validate_escalation_cases_rejects_a_command_absent_from_every_ancestor():
    cases = [
        BenchmarkCase(
            context="ReviewTicket",
            utterance="do something else entirely",
            kind="escalation",
            expected_ancestor_command="SomewhereElse/do_thing",
        )
    ]
    problems = validate_escalation_cases(cases, _LABEL_SPACE, _ANCESTOR_MAP)
    assert len(problems) == 1
    assert "not present in any ancestor" in problems[0]


def test_validate_escalation_cases_rejects_a_context_with_no_ancestors():
    cases = [
        BenchmarkCase(
            context="AccessReviewWorkspace",
            utterance="nothing above this",
            kind="escalation",
            expected_ancestor_command="ReviewTicket/certify_approve",
        )
    ]
    problems = validate_escalation_cases(cases, _LABEL_SPACE, _ANCESTOR_MAP)
    assert len(problems) == 1
    assert "no ancestors" in problems[0]


def test_validate_escalation_cases_ignores_routing_cases():
    cases = [
        BenchmarkCase(
            context="ReviewTicket",
            utterance="approve this one",
            expected_label="ReviewTicket/certify_approve",
        )
    ]
    assert validate_escalation_cases(cases, _LABEL_SPACE, _ANCESTOR_MAP) == []


def test_validate_routing_cases_rejects_a_label_outside_the_context():
    cases = [
        BenchmarkCase(
            context="ReviewTicket",
            utterance="bulk decide please",
            expected_label="AccessReviewWorkspace/bulk_decide",
        )
    ]
    problems = validate_routing_cases(cases, _LABEL_SPACE)
    assert len(problems) == 1
    assert "can never pass" in problems[0]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _sample_reports():
    return [
        HeldoutReport(
            context="Account",
            in_distribution_f1=0.94,
            routing=RoutingScore(
                total=4,
                top1_correct=2,
                in_list_correct=3,
                top1=0.5,
                in_list=0.75,
                per_command={
                    "Account/close_account": {
                        "total": 4,
                        "top1_correct": 2,
                        "in_list_correct": 3,
                    }
                },
            ),
            escalation=EscalationScore(total=2, correct=1, recall=0.5, failures=[]),
            seed=42,
            heldout_personas=["persona_03", "persona_06"],
            notes=["1 label(s) have no held-out rows and are therefore unmeasured"],
        ),
        HeldoutReport(
            context="ReviewTicket",
            in_distribution_f1=0.90,
            routing=RoutingScore(
                total=6, top1_correct=3, in_list_correct=5, top1=0.5, in_list=5 / 6
            ),
            escalation=None,
            seed=42,
            heldout_personas=["persona_03"],
        ),
    ]


def test_aggregate_totals_sums_counts_rather_than_averaging_contexts():
    totals = aggregate_totals(_sample_reports())

    assert totals["contexts"] == 2
    assert totals["routing_total"] == 10
    assert totals["routing_top1_correct"] == 5
    assert totals["routing_top1"] == pytest.approx(0.5)
    assert totals["routing_in_list"] == pytest.approx(0.8)
    assert totals["escalation_total"] == 2
    assert totals["escalation_recall"] == pytest.approx(0.5)
    assert totals["mean_in_distribution_f1"] == pytest.approx(0.92)


def test_format_report_labels_the_legacy_metric_as_in_distribution():
    text = format_report(_sample_reports())

    assert "in-dist F1" in text
    assert "memorisation" in text
    assert "Account" in text
    assert "ReviewTicket" in text
    assert "TOTAL" in text
    assert "escal recall" in text
    assert "no held-out rows" in text


def test_format_report_with_no_contexts():
    assert "no contexts evaluated" in format_report([])


def test_write_report_writes_json_under_command_info(tmp_path):
    workflow = tmp_path / "my_workflow"
    workflow.mkdir()

    path = write_report(str(workflow), _sample_reports())

    assert path == str(workflow / "___command_info" / "heldout_evaluation.json")
    payload = json.loads((workflow / "___command_info" / "heldout_evaluation.json").read_text())

    assert payload["schema_version"] == REPORT_SCHEMA_VERSION
    assert payload["totals"]["routing_total"] == 10
    assert [c["context"] for c in payload["contexts"]] == ["Account", "ReviewTicket"]
    assert payload["contexts"][0]["in_distribution_f1"] == pytest.approx(0.94)
    assert payload["contexts"][0]["heldout_personas"] == ["persona_03", "persona_06"]
    assert payload["contexts"][1]["escalation"] is None
    assert "memorisation" in payload["metric_notes"]["in_distribution_f1"]
    assert "lone, confident" in payload["metric_notes"]["routing"]


def test_write_report_creates_the_command_info_directory(tmp_path):
    workflow = tmp_path / "fresh_workflow"
    workflow.mkdir()
    assert not (workflow / "___command_info").exists()

    write_report(str(workflow), [])

    assert (workflow / "___command_info" / "heldout_evaluation.json").is_file()


# ---------------------------------------------------------------------------
# End-to-end: split -> score -> report, with no model involved
# ---------------------------------------------------------------------------


def test_training_scoring_keeps_benchmark_when_persona_holdout_is_empty():
    report = HeldoutReport(context="Ctx", in_distribution_f1=0.94)
    benchmark_cases = [
        BenchmarkCase(
            context="Ctx",
            utterance="route this",
            expected_label="Ctx/route",
        ),
        BenchmarkCase(
            context="Ctx",
            utterance="serve this upstairs",
            kind="escalation",
            expected_ancestor_command="Parent/serve",
        ),
    ]
    predict = _lookup_predictor(
        {
            "route this": ["Ctx/route"],
            "serve this upstairs": ["wildcard"],
        }
    )

    _score_heldout_context(report, [], benchmark_cases, predict)

    assert report.routing is None
    assert report.benchmark_routing is not None
    assert report.benchmark_routing.total == 1
    assert report.benchmark_routing.top1_correct == 1
    assert report.escalation is not None
    assert report.escalation.total == 1
    assert report.escalation.correct == 1
    assert any("Persona holdout unavailable" in note for note in report.notes)


def test_persona_split_feeds_scoring_and_reporting_end_to_end(tmp_path):
    records = _multi_persona_records(num_personas=8, labels=("Ctx/a", "Ctx/b"))
    split = split_by_persona(records, holdout_fraction=0.25, seed=42)
    assert split.heldout

    # A predictor that gets "Ctx/a" right and puts "Ctx/b" second: exactly the shape the
    # in-distribution F1 hides, since top-1 and in-list diverge.
    def predict(utterance):
        return ["Ctx/a"] if "Ctx/a" in utterance else ["Ctx/a", "Ctx/b"]

    routing = score_routing(split.heldout, predict)
    assert routing.total == len(split.heldout)
    assert routing.top1 == pytest.approx(0.5)
    assert routing.in_list == pytest.approx(1.0)

    report = HeldoutReport(
        context="Ctx",
        in_distribution_f1=0.94,
        routing=routing,
        escalation=score_escalation([], predict),
        seed=42,
        heldout_personas=split.heldout_personas,
        notes=split.notes,
    )
    path = write_report(str(tmp_path), [report])
    payload = json.loads(open(path, encoding="utf-8").read())

    assert payload["totals"]["routing_top1"] == pytest.approx(0.5)
    assert payload["contexts"][0]["in_distribution_f1"] == pytest.approx(0.94)
    assert payload["contexts"][0]["heldout_personas"] == split.heldout_personas


# ---------------------------------------------------------------------------
# D2: routing and escalation are never blended (fix-k0i.10, fix-k0i.15, fix-k0i.35)
# ---------------------------------------------------------------------------


def test_score_routing_refuses_a_non_routable_expected_label():
    """A lone wildcard escalates to an ancestor; it is not a route.

    Escalation-class rows carry real persona attribution, so they reach the
    holdout split like any other row. Counting predictions[0] == "wildcard" as a
    correct top-1 scores a semantics the runtime does not have -- and in a
    context with ancestors those rows are most of the population, so it does not
    skew the headline number, it replaces it.
    """
    cases = [LabeledUtterance(utterance="do the thing", label=WILDCARD_LABEL,
                              persona="p1")]

    with pytest.raises(ValueError) as excinfo:
        score_routing(cases, lambda _u: [WILDCARD_LABEL])

    assert WILDCARD_LABEL in str(excinfo.value)
    assert "score_escalation" in str(excinfo.value)


def test_partition_by_routability_splits_the_two_populations():
    records = _records([
        ("p1", "TodoList/add_item", ["add milk"]),
        ("p1", WILDCARD_LABEL, ["something an ancestor owns"]),
        ("p2", "TodoList/add_item", ["add eggs"]),
    ])

    routable, escalation_class = partition_by_routability(records)

    assert [r.utterance for r in routable] == ["add milk", "add eggs"]
    assert [r.utterance for r in escalation_class] == ["something an ancestor owns"]


def test_heldout_escalation_rows_are_scored_on_their_own_axis():
    """Split, not dropped: in a context with ancestors these are most of the rows."""
    report = HeldoutReport(context="TodoList")
    records = _records([
        ("p1", "TodoList/add_item", ["add milk"]),
        ("p1", WILDCARD_LABEL, ["what can this app do"]),
    ])

    # Routes the command correctly; escalates correctly on the wildcard row.
    def predict(utterance):
        return ["TodoList/add_item"] if utterance == "add milk" else [WILDCARD_LABEL]

    _score_heldout_context(report, records, [], predict)

    assert report.routing is not None
    assert report.routing.total == 1, "escalation rows must not inflate routing N"
    assert report.routing.top1_correct == 1

    assert report.holdout_escalation is not None
    assert report.holdout_escalation.total == 1
    assert report.holdout_escalation.correct == 1


def test_a_context_of_only_escalation_rows_scores_no_routing_and_says_so():
    report = HeldoutReport(context="TodoList")
    records = _records([("p1", WILDCARD_LABEL, ["ancestor thing", "another"])])

    _score_heldout_context(report, records, [], lambda _u: [WILDCARD_LABEL])

    assert report.routing is None, "0-case routing is absent, not a fake 0%"
    assert report.holdout_escalation.total == 2
    assert any("escalation-class" in note for note in report.notes)


def test_benchmark_routing_reaches_the_totals_and_the_table():
    """The fixed benchmark is the only number comparable between two runs.

    The persona holdout is re-drawn every run, so its top-1 cannot be compared.
    Benchmark routing used to be visible only in a transient per-context print.
    """
    report = HeldoutReport(
        context="TodoList",
        routing=RoutingScore(total=10, top1_correct=7, in_list_correct=9),
        benchmark_routing=RoutingScore(total=4, top1_correct=3, in_list_correct=4),
    )

    totals = aggregate_totals([report])
    assert totals["benchmark_routing_total"] == 4
    assert totals["benchmark_routing_top1_correct"] == 3
    assert totals["benchmark_routing_top1"] == pytest.approx(0.75)
    # Kept separate from the persona-holdout population, never summed into it.
    assert totals["routing_total"] == 10

    rendered = format_report([report])
    assert "bench N" in rendered
    assert "bench top-1" in rendered


def test_zero_denominator_metrics_are_null_not_zero():
    """0.0 and "not measured" are different facts and JSON cannot tell them apart.

    format_report renders "-" for both, but a JSON consumer diffing two runs
    reads a missing measurement as a total failure.
    """
    totals = aggregate_totals([HeldoutReport(context="TodoList")])

    assert totals["escalation_recall"] is None
    assert totals["routing_top1"] is None
    assert totals["routing_in_list"] is None
    assert totals["benchmark_routing_top1"] is None
    assert totals["holdout_escalation_recall"] is None
    assert totals["mean_in_distribution_f1"] is None


def test_merged_and_fresh_totals_have_the_same_schema():
    """A merged report and a full-retrain report must be diffable.

    The selective path used to re-sum the totals in a parallel implementation
    that dropped keys and turned a missing mean into a literal 0.0, so diffing a
    selective run against a full one read as a collapse in quality.
    """
    entries = [
        {
            "context": "TodoList",
            "routing": {"total": 10, "top1_correct": 7, "in_list_correct": 9},
            "escalation": {"total": 2, "correct": 1},
            "in_distribution_f1": 0.9,
        },
        {"context": "TodoItem", "carried_forward": True},
    ]

    merged = _recompute_heldout_totals(entries)
    fresh = aggregate_totals([HeldoutReport.model_validate(e) for e in entries])

    assert set(merged) == set(fresh)
    assert merged == fresh


def test_merged_totals_preserve_a_missing_mean_as_null():
    merged = _recompute_heldout_totals([{"context": "TodoList"}])

    assert merged["mean_in_distribution_f1"] is None, "a missing mean is not 0.0"
    assert merged["escalation_recall"] is None


def test_rescue_threshold_matches_the_trainers_floor():
    """Two modules used to enforce the same requirement with different numbers.

    The rescue fired only at ZERO training rows while split_training_data demands
    two, so a label could pass the split with exactly one row and then abort the
    whole run mid-loop -- after earlier contexts had already spent their LLM and
    GPU budget.
    """
    # p2 is held out. 'rare' then has exactly one training row (from p1), which
    # the old zero-threshold rescue accepted and the trainer then rejected.
    records = _records([
        ("p1", "Ctx/rare", ["only training row"]),
        ("p2", "Ctx/rare", ["held out row"]),
        ("p1", "Ctx/common", ["a", "b", "c"]),
        ("p2", "Ctx/common", ["d", "e", "f"]),
    ])

    split = split_by_persona(records, seed=7, holdout_fraction=0.5)

    trained = Counter(r.label for r in split.train)
    assert trained["Ctx/rare"] >= MIN_TRAINING_ROWS_PER_LABEL, (
        "the split handed the trainer a label it will refuse"
    )
    # The rescued label is no longer measured, and the report says so.
    assert not any(r.label == "Ctx/rare" for r in split.heldout)
    assert any("floor" in note for note in split.notes)


class _WarningSink(logging.Handler):
    """Collects real `LogRecord`s off the real logger.

    The `fastWorkflow` logger sets ``propagate = False``, so pytest's `caplog`, whose
    handler lives on the root logger, never sees these records.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno == logging.WARNING:
            self.messages.append(record.getMessage())


def test_split_training_data_names_commands_not_encoded_ids():
    """A developer cannot map label id 3 to a command without the LabelEncoder.

    The thin label is now a warning rather than an abort - it trains unmeasured, the
    way this module's own persona split already rescues a starved label - so the name
    has to travel on the warning.
    """
    dataset = [("only row", 3), ("a", 11), ("b", 11)]

    sink = _WarningSink()
    fastworkflow_logger.addHandler(sink)
    try:
        train_rows, evaluation_rows = split_training_data(
            dataset, lambda encoded: f"TodoList/cmd_{encoded}"
        )
    finally:
        fastworkflow_logger.removeHandler(sink)

    assert ("only row", 3) in train_rows
    assert 3 not in {label for _, label in evaluation_rows}
    message = "\n".join(sink.messages)
    assert "TodoList/cmd_3" in message
    assert "[3]" not in message


def test_report_discloses_which_model_was_evaluated(tmp_path):
    """AR1: state whether the number is a lower bound. It was stated nowhere a
    JSON or table consumer could see."""
    reports = [HeldoutReport(context="TodoList",
                             routing=RoutingScore(total=4, top1_correct=3))]
    path = write_report(str(tmp_path), reports)
    payload = json.loads(Path(path).read_text())

    disclosure = payload["metric_notes"]["which_model_was_evaluated"]
    assert "LOWER BOUND" in disclosure
    assert "in-generator" in disclosure

    rendered = format_report(reports)
    assert "LOWER BOUND" in rendered

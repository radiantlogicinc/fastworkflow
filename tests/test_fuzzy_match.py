"""Tests for Levenshtein fuzzy matching and the db_lookup validator built on it.

Covers fix-86c: `find_best_matches` scored candidates on their leading window
only, so a candidate containing the input anywhere else could lose to an
unrelated one, and `DatabaseValidator.fuzzy_match` then reported that wrong
candidate as a confident match.
"""
import json
import os

import pytest

import fastworkflow
from fastworkflow.utils.fuzzy_match import (
    best_window_distance,
    find_best_matches,
    normalize_text,
    normalized_levenshtein_distance,
)
from fastworkflow.utils.signatures import DatabaseValidator

PEOPLE = ["Aaron Garrison", "Barry Jones"]

_RETAIL_DATA = os.path.join(
    os.path.dirname(os.path.abspath(fastworkflow.__file__)),
    "examples", "retail_workflow", "retail_data",
)


@pytest.fixture(scope="module")
def retail_names() -> list[str]:
    """379 real customer names from the in-tree retail workflow."""
    with open(os.path.join(_RETAIL_DATA, "users.json")) as f:
        users = json.load(f)
    return sorted({
        f"{u['name']['first_name']} {u['name']['last_name']}"
        for u in users.values()
    })


@pytest.fixture(scope="module")
def retail_products() -> list[str]:
    with open(os.path.join(_RETAIL_DATA, "products.json")) as f:
        products = json.load(f)
    return sorted({p["name"] for p in products.values()})


class TestBestWindowDistance:
    """The scoring primitive behind the opt-in matching mode."""

    def test_contained_input_scores_zero(self):
        assert best_window_distance("garrison", "aarongarrison") == 0.0

    def test_leading_input_scores_zero(self):
        assert best_window_distance("fred", "fredflintstone") == 0.0

    def test_unrelated_candidate_scores_high(self):
        assert best_window_distance("garrison", "barryjones") > 0.3

    def test_never_exceeds_leading_window_distance(self):
        """Best-window can only lower a distance, never raise it."""
        for query in ("garrison", "fred", "ltd", "invoice"):
            for candidate in ("aarongarrison", "fredflintstone", "itdivision",
                              "invoiceprocessing", "barryjones"):
                leading = normalized_levenshtein_distance(
                    query, candidate[:len(query)])
                assert best_window_distance(query, candidate) <= leading

    def test_candidate_shorter_than_input_is_compared_whole(self):
        assert (best_window_distance("garrison", "jones")
                == normalized_levenshtein_distance("garrison", "jones"))

    def test_empty_candidate(self):
        assert best_window_distance("garrison", "") == 1.0


class TestFindBestMatchesLeadingWindow:
    """Default behaviour, which intent detection depends on. Do not widen it
    here without measuring the routing effect first."""

    def test_reproduces_the_reported_wrong_answer(self):
        """The defect itself, pinned: this is why best_window had to be opt-in
        rather than a change of default."""
        matches, distance = find_best_matches("Garrison", PEOPLE, threshold=0.7)
        assert matches == ["Barry Jones"]
        assert distance == pytest.approx(0.375)

    def test_bare_fragment_does_not_match_a_longer_utterance(self):
        """Pins the intent_detection.py pre-match. Under best-window scoring
        'product' would match at distance 0.0 and route ahead of the
        classifier, bypassing it entirely."""
        candidates = ["list all the product types you carry",
                      "cancel the order that has not shipped yet"]
        for fragment in ("product", "types", "shipped"):
            matches, _ = find_best_matches(
                fragment.replace(" ", "_"), candidates, threshold=0.3)
            assert matches == [], f"{fragment!r} newly matches {matches}"

    def test_partial_input_matches_longer_candidate_by_prefix(self):
        matches, distance = find_best_matches(
            "Fred", ["Fred Flintstone", "Barney Rubble"], threshold=0.7)
        assert matches == ["Fred Flintstone"]
        assert distance == 0.0


class TestFindBestMatchesBestWindow:
    def test_contained_candidate_wins(self):
        matches, distance = find_best_matches(
            "Garrison", PEOPLE, threshold=0.7, best_window=True)
        assert matches == ["Aaron Garrison"]
        assert distance == 0.0

    def test_partial_prefix_behaviour_is_preserved(self):
        matches, distance = find_best_matches(
            "Fred", ["Fred Flintstone", "Barney Rubble"],
            threshold=0.7, best_window=True)
        assert matches == ["Fred Flintstone"]
        assert distance == 0.0

    def test_several_containing_candidates_tie(self):
        """Ties are what let the caller report honest ambiguity instead of
        picking one."""
        matches, distance = find_best_matches(
            "invoice", ["invoice_processing", "invoice_validation",
                        "payment_processing"],
            threshold=0.7, best_window=True)
        assert set(matches) == {"invoice_processing", "invoice_validation"}
        assert distance == 0.0

    def test_one_edit_typo_still_recovered(self):
        matches, _ = find_best_matches(
            "Garrisen", PEOPLE, threshold=0.7, best_window=True)
        assert matches == ["Aaron Garrison"]

    def test_threshold_still_gates(self):
        matches, distance = find_best_matches(
            "zzzzzzzzzzzz", PEOPLE, threshold=0.1, best_window=True)
        assert matches == []
        assert distance is None


class TestFindBestMatchesContract:
    def test_returns_empty_list_not_none_when_nothing_is_close(self):
        """The docstring promised (None, None); callers testing `is None` took
        the wrong branch."""
        assert find_best_matches("zzzzzzzzzzzz", PEOPLE, 0.01) == ([], None)

    def test_returns_empty_list_for_empty_candidates(self):
        assert find_best_matches("Garrison", [], 0.7) == ([], None)

    def test_accepts_a_one_shot_iterable_of_candidates(self):
        """Candidates are consumed twice internally, so a bare iterator must be
        materialised first."""
        matches, _ = find_best_matches("Garrison", iter(PEOPLE), 0.7)
        assert matches == ["Barry Jones"]

    def test_normalization_ignores_case_spaces_and_underscores(self):
        assert normalize_text("Aaron_Garrison ") == "aarongarrison"


class TestDatabaseValidatorFuzzyMatch:
    def test_reported_case_now_resolves_correctly(self):
        assert (DatabaseValidator.fuzzy_match("Garrison", PEOPLE)
                == (True, "Aaron Garrison", []))

    def test_exact_match_short_circuits(self):
        assert (DatabaseValidator.fuzzy_match("Aaron Garrison", PEOPLE)
                == (True, "Aaron Garrison", []))

    def test_exact_match_is_case_insensitive_and_returns_canonical_case(self):
        assert (DatabaseValidator.fuzzy_match("aaron garrison", PEOPLE)
                == (True, "Aaron Garrison", []))

    def test_contained_fragment_no_longer_picks_an_unrelated_candidate(self):
        """'Ltd' turned into 'IT Division' on the reported dataset."""
        matched, corrected, _ = DatabaseValidator.fuzzy_match(
            "Ltd", ["IT Division", "Acme Ltd"])
        assert (matched, corrected) == (True, "Acme Ltd")

    def test_ambiguous_fragment_returns_suggestions_not_a_choice(self):
        matched, corrected, suggestions = DatabaseValidator.fuzzy_match(
            "invoice", ["invoice_processing", "invoice_validation"])
        assert matched is False
        assert corrected is None
        assert set(suggestions) == {"invoice_processing", "invoice_validation"}

    def test_empty_value_declines(self):
        assert DatabaseValidator.fuzzy_match("", PEOPLE) == (False, None, [])

    def test_empty_candidate_list_declines(self):
        assert DatabaseValidator.fuzzy_match("Garrison", []) == (False, None, [])

    def test_auto_apply_threshold_downgrades_a_typo_to_a_suggestion(self):
        """A caller for whom a wrong substitution is expensive can require an
        exact or containing match without giving up suggestions."""
        default = DatabaseValidator.fuzzy_match("Garrisen", PEOPLE)
        assert default == (True, "Aaron Garrison", [])

        strict = DatabaseValidator.fuzzy_match(
            "Garrisen", PEOPLE, auto_apply_threshold=0.0)
        assert strict == (False, None, ["Aaron Garrison"])

    def test_auto_apply_threshold_of_zero_still_applies_a_containing_match(self):
        assert (DatabaseValidator.fuzzy_match(
            "Garrison", PEOPLE, auto_apply_threshold=0.0)
            == (True, "Aaron Garrison", []))

    def test_suggest_threshold_gates_the_levenshtein_stage(self):
        """The defect: this stage used a hardcoded 0.7 and ignored the caller."""
        matched, corrected, suggestions = DatabaseValidator.fuzzy_match(
            "Garrisen", PEOPLE, suggest_threshold=0.0)
        assert (matched, corrected) == (False, None)
        assert "Aaron Garrison" in suggestions

    def test_difflib_stage_suggests_when_levenshtein_finds_nothing(self):
        """The final stage never applies anything, only suggests. Note how loose
        the 0.2 default cutoff is: an unrelated name is suggested too, which is
        why any non-empty suggestion list invalidating the field is worth
        remembering when choosing a candidate list."""
        matched, corrected, suggestions = DatabaseValidator.fuzzy_match(
            "Garisn", PEOPLE, suggest_threshold=0.0, threshold=0.2)
        assert (matched, corrected) == (False, None)
        assert "Aaron Garrison" in suggestions

        tight = DatabaseValidator.fuzzy_match(
            "Garisn", PEOPLE, suggest_threshold=0.0, threshold=0.6)
        assert tight == (False, None, ["Aaron Garrison"])


class TestDefaultThresholds:
    """Covers fix-11r and fix-63k, measured against the in-tree retail
    catalogues rather than invented data."""

    def test_non_member_value_is_not_silently_applied(self, retail_names):
        """fix-11r: at the previous 0.7 default, 'Batman' was rewritten to a
        real customer (d=0.333) and the turn proceeded against their account."""
        matched, corrected, suggestions = DatabaseValidator.fuzzy_match(
            "Batman", retail_names)
        assert matched is False, f"'Batman' was silently applied as {corrected!r}"
        assert corrected is None

    @pytest.mark.parametrize("non_member", [
        "Batman", "Sherlock Holmes", "Ada Lovelace", "Zaphod Beeblebrox",
        "Aaaaa Bbbbb",
    ])
    def test_no_non_member_name_is_applied(self, non_member, retail_names):
        matched, corrected, _ = DatabaseValidator.fuzzy_match(
            non_member, retail_names)
        assert matched is False, f"{non_member!r} -> {corrected!r}"

    def test_real_typos_of_real_members_still_apply(self, retail_names):
        """The other half of fix-11r: tightening must not cost typo recall."""
        for typo, expected in [("Yusuf Rossy", "Yusuf Rossi"),
                               ("Aarav Andersen", "Aarav Anderson"),
                               ("Aarav Andrsn", "Aarav Anderson")]:
            assert (DatabaseValidator.fuzzy_match(typo, retail_names)
                    == (True, expected, []))

    def test_exact_member_still_applies(self, retail_names):
        member = retail_names[0]
        assert (DatabaseValidator.fuzzy_match(member, retail_names)
                == (True, member, []))

    def test_cross_domain_value_is_not_applied(self, retail_names, retail_products):
        """A legal value from the wrong domain must not be rewritten. At 0.7,
        'aarav_santos_2259' became 'Garden Hose'."""
        matched, corrected, _ = DatabaseValidator.fuzzy_match(
            "aarav_santos_2259", retail_products)
        assert matched is False, f"user id was applied as {corrected!r}"

    def test_short_value_typo_is_offered_rather_than_applied(self):
        """The known, accepted cost of the strict default: distance is
        edits/length, so one edit on a 3-character value scores 0.333."""
        assert (DatabaseValidator.fuzzy_match("Ltx", ["Ltd", "Acme"])
                == (False, None, ["Ltd"]))

    def test_realistic_enum_typos_are_unaffected(self):
        """Values of five or more characters, which covers ordinary enums."""
        statuses = ["pending", "processed", "delivered", "cancelled",
                    "exchange requested"]
        for typo, expected in [("pendinx", "pending"),
                               ("processex", "processed"),
                               ("deliverex", "delivered")]:
            assert (DatabaseValidator.fuzzy_match(typo, statuses)
                    == (True, expected, []))

    def test_absent_value_is_declined_rather_than_rejected(self, retail_names):
        """fix-63k: a value that is simply not in this candidate list must come
        back as 'no opinion' so the framework leaves it alone, instead of being
        failed on unrelated suggestions. A 32-char uid against a name list was
        the reported case."""
        uid = "a3f8c1d24b6e9f0071e2c5a8b4d3f6e9"
        assert DatabaseValidator.fuzzy_match(uid, retail_names) == (False, None, [])

    def test_loose_cutoff_would_reject_it_on_unrelated_suggestions(self, retail_names):
        """Why the cutoff default moved: at 0.2 the same uid comes back with
        suggestions, and any non-empty suggestion list fails validation."""
        uid = "a3f8c1d24b6e9f0071e2c5a8b4d3f6e9"
        _, _, suggestions = DatabaseValidator.fuzzy_match(
            uid, retail_names, threshold=0.2)
        assert suggestions, "expected the loose cutoff to produce suggestions"

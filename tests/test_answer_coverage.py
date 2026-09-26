"""The roster nudge: the agent is told which named items it never looked at.

Offline only. Nothing here starts a server, calls a model or touches a backend:
the archive is a real SQLite file in a temp dir, the
extract module is a recorder, and every assertion is about bytes already on disk
before the test begins.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
import unicodedata
import unittest
from unittest import mock

import dspy

from fastworkflow import answer_coverage
from fastworkflow.answer_coverage import (
    INSTRUCTED_KINDS,
    NUDGE_MAX_BYTES,
    NUDGE_MIN_ITERS_LEFT,
    build_nudge,
    named_entities,
    normalise,
    nudge_block,
    request_text,
    split_by_presence,
    strip_query_echoes,
    subject_corpus,
)
from fastworkflow.observation_offloading.archive import (
    RuntimeHandleArchive,
    RuntimeHandleScope,
)
from fastworkflow.observation_offloading.state import (
    record_context_clause,
    reset_runtime_state,
    snapshot_events,
)
from tests.test_answer_rehydration import Recorder

CARD = (
    "Two Active Directory rights keep coming back on this quarter's "
    "privileged-access exceptions: Active Directory_Cloud Administrator and "
    "Active Directory_Compliance Officer. Audit both wherever they appear — "
    "the right, the system that publishes it. Five people on this quarter's "
    "exception list need the same treatment — Alan Cooper, Alisha Ochoa, "
    "Anna Garcia, Barbara Sanchez and Brandon Miller. Separately, the control "
    "'Active contractor identities whom manager left' has an open finding. "
    "Christopher Hubbard is one of the people it names and leaves on Friday."
)


def scope_for(name: str) -> RuntimeHandleScope:
    return RuntimeHandleScope(
        store_identity=name, channel_id="c", experiment_id="exp-ido-8ps-22",
        task_id="task", attempt=1, turn_key="turn-1",
    )


class RequestSplit(unittest.TestCase):
    """The plan the planner appended is not part of the request."""

    def test_the_todo_list_is_cut_off(self) -> None:
        self.assertEqual(
            request_text("Find Alan Cooper.\n\nExecute these next steps:\n1. Zeno Ppp"),
            "Find Alan Cooper.",
        )

    def test_the_user_query_prefix_is_removed(self) -> None:
        self.assertEqual(request_text("User Query:\nFind Alan Cooper."),
                         "Find Alan Cooper.")

    def test_a_query_without_a_plan_is_its_own_request(self) -> None:
        self.assertEqual(request_text("Find Alan Cooper."), "Find Alan Cooper.")

    def test_a_planner_invention_never_becomes_a_named_item(self) -> None:
        names = [e.text for e in named_entities(
            request_text("Find Alan Cooper.\n\nExecute these next steps:\n"
                         "1. open Zeno Ppp then Quentin Rrr")
        )]
        self.assertEqual(names, ["Alan Cooper"])


class EntityExtraction(unittest.TestCase):
    """Deterministic spans: no model, no dictionary, no workflow lookup."""

    def test_the_card_yields_exactly_its_named_items(self) -> None:
        found = [(e.kind, e.text) for e in named_entities(CARD)]
        self.assertEqual(found, [
            ("name", "Active Directory"),
            ("name", "Active Directory_Cloud Administrator"),
            ("name", "Active Directory_Compliance Officer"),
            ("name", "Alan Cooper"),
            ("name", "Alisha Ochoa"),
            ("name", "Anna Garcia"),
            ("name", "Barbara Sanchez"),
            ("name", "Brandon Miller"),
            ("quoted", "Active contractor identities whom manager left"),
            ("name", "Christopher Hubbard"),
        ])

    def test_a_sentence_capital_is_dropped_only_when_two_tokens_remain(self) -> None:
        # "Two Active Directory rights" is about Active Directory.
        self.assertIn("Active Directory",
                      [e.text for e in named_entities("Two Active Directory rights.")])
        # "Christopher Hubbard is one of..." keeps both: a bare surname is worse.
        self.assertEqual([e.text for e in named_entities("Christopher Hubbard is here.")],
                         ["Christopher Hubbard"])

    def test_a_comma_separates_two_names(self) -> None:
        self.assertEqual(
            [e.text for e in named_entities("we saw Alan Cooper, Anna Garcia today")],
            ["Alan Cooper", "Anna Garcia"],
        )

    def test_a_single_capitalised_token_is_not_a_name(self) -> None:
        self.assertEqual(named_entities("Separately, he leaves on Friday."), [])

    def test_an_apostrophe_does_not_open_a_quotation(self) -> None:
        kinds = {e.kind for e in named_entities("this quarter's list and that one's")}
        self.assertNotIn("quoted", kinds)

    def test_a_possessive_ends_the_name_it_marks(self) -> None:
        # ido-jf6/F29. "Cooper's" ends in a letter, so the run used to carry on
        # into the next capital and name a thing no observation can contain.
        self.assertEqual(
            [e.text for e in named_entities(
                "List Alan Cooper's Active Directory rights and "
                "Brandon Miller's accounts."
            )],
            ["Alan Cooper", "Active Directory", "Brandon Miller"],
        )

    def test_a_curly_possessive_reads_the_same_as_a_straight_one(self) -> None:
        self.assertEqual(
            [e.text for e in named_entities("Audit Barbara Sanchez’s permissions.")],
            ["Barbara Sanchez"],
        )
        self.assertEqual(
            [e.text for e in named_entities("open Alan Cooper’s Active Directory row")],
            ["Alan Cooper", "Active Directory"],
        )

    def test_a_plural_possessive_is_a_bare_apostrophe(self) -> None:
        self.assertEqual(
            [e.text for e in named_entities(
                "Audit the Cooper Brothers' Active Directory rights.")],
            ["Cooper Brothers", "Active Directory"],
        )

    def test_an_apostrophe_inside_a_name_survives(self) -> None:
        # Only the possessive marker comes off; O'Brien is the name itself.
        self.assertEqual(
            [e.text for e in named_entities("Find Sean O'Brien and Anna Garcia.")],
            ["Sean O'Brien", "Anna Garcia"],
        )

    def test_a_sentence_initial_imperative_is_not_part_of_a_name(self) -> None:
        # ido-jf6/F29. "List Identities", "Show Accounts" and "Compare Alan"
        # were named items; the capital is grammar, not a handle.
        self.assertEqual(
            named_entities("List Identities whose manager left. Show Accounts for each."),
            [],
        )
        self.assertEqual(named_entities("Compare Alan and Brandon."), [])
        self.assertEqual(
            [e.text for e in named_entities("Show Alan Cooper's manager.")],
            ["Alan Cooper"],
        )

    def test_an_imperative_word_away_from_the_sentence_start_is_kept(self) -> None:
        # Only the FIRST token of a sentence is tested against the verb list.
        self.assertEqual(
            [e.text for e in named_entities("we saw Alan Cooper, Barbara List today")],
            ["Alan Cooper", "Barbara List"],
        )

    def test_a_sentence_capital_that_is_not_a_verb_is_still_kept(self) -> None:
        self.assertEqual(
            [e.text for e in named_entities("Christopher Hubbard leaves on Friday.")],
            ["Christopher Hubbard"],
        )

    def test_a_slash_or_a_dash_separates_two_named_items(self) -> None:
        self.assertEqual(
            [e.text for e in named_entities(
                "Audit Alan Cooper/Brandon Miller and Anna Garcia—contractor.")],
            ["Alan Cooper", "Brandon Miller", "Anna Garcia"],
        )
        self.assertEqual(
            [e.text for e in named_entities("Anna Garcia – contractor and Alan Cooper.")],
            ["Anna Garcia", "Alan Cooper"],
        )

    def test_an_ordinary_multi_word_name_is_unchanged(self) -> None:
        self.assertEqual(
            [e.text for e in named_entities(
                "we saw Alan Cooper and Active Directory_Cloud Administrator today")],
            ["Alan Cooper", "Active Directory_Cloud Administrator"],
        )

    def test_a_particle_or_an_initial_is_not_part_of_a_name(self) -> None:
        # Documented limit, unchanged by ido-jf6: a lowercase particle and the
        # full stop of an initial both end a run, so these name nobody. The
        # module never invents a handle it cannot spell from the request.
        self.assertEqual(
            named_entities("Check Maria de la Cruz and J. R. Smith and "
                           "Ludwig van der Berg."),
            [],
        )

    def test_uids_and_addresses_are_named_items(self) -> None:
        found = {e.kind: e.text for e in named_entities(
            "open 28c5aeb5b64e4ac6c40c57b0235980e2 and mail a.cooper@example.com"
        )}
        self.assertEqual(found["uid"], "28c5aeb5b64e4ac6c40c57b0235980e2")
        self.assertEqual(found["email"], "a.cooper@example.com")

    def test_duplicates_collapse_on_the_normalised_form(self) -> None:
        self.assertEqual(
            [e.text for e in named_entities("Alan Cooper met ALAN COOPER again.")],
            ["Alan Cooper"],
        )

    def test_only_names_uids_and_addresses_are_ever_instructed(self) -> None:
        self.assertEqual(INSTRUCTED_KINDS, frozenset({"name", "uid", "email"}))


class Presence(unittest.TestCase):
    """Normalisation and the context clause."""

    def test_case_and_whitespace_are_normalised(self) -> None:
        entities = named_entities("Find Alan Cooper.")
        observed, unobserved = split_by_presence(
            entities, normalise("ALAN\n  COOPER  holds the right")
        )
        self.assertEqual([e.text for e in observed], ["Alan Cooper"])
        self.assertEqual(unobserved, [])

    def test_nfkc_folds_a_full_width_spelling(self) -> None:
        entities = named_entities("Find Alan Cooper.")
        observed, _ = split_by_presence(
            entities, normalise("Ａｌａｎ Cooper"))
        self.assertEqual([e.text for e in observed], ["Alan Cooper"])

    # -- ido-c7m / F36: an invisible character cannot hide a name ---------

    def test_a_zero_width_space_in_a_row_leaves_the_name_observed(self) -> None:
        """Standing where the name's space is, and beside it."""
        entities = named_entities("Find Alan Cooper.")
        for row in ("Identity 1  Alan\u200bCooper",
                    "Identity 1  Alan \u200bCooper",
                    "Identity 1  Alan\u200b Cooper"):
            observed, unobserved = split_by_presence(entities, normalise(row))
            self.assertEqual([e.text for e in observed], ["Alan Cooper"], row)
            self.assertEqual(unobserved, [], row)

    def test_a_zero_width_space_inside_a_token_splits_that_token(self) -> None:
        """The deliberate half of the trade-off, pinned so it is not silent.

        U+200B is a break opportunity, so it folds to a space and one inside a
        token separates it. The other invisible characters -- the soft hyphen
        and the rest -- are removed, which is what a within-token break needs;
        only the one character that is named a space behaves like one.
        """
        self.assertEqual(normalise("Al\u200ban Cooper"), "al an cooper")
        self.assertEqual(normalise("Al\u00adan Cooper"), "alan cooper")

    def test_a_soft_hyphen_in_a_row_leaves_the_name_observed(self) -> None:
        entities = named_entities("Find Brandon Miller.")
        observed, unobserved = split_by_presence(
            entities, normalise("Identity 2  Brandon Mil\u00adler"))
        self.assertEqual([e.text for e in observed], ["Brandon Miller"])
        self.assertEqual(unobserved, [])

    def test_a_byte_order_mark_in_a_row_leaves_the_name_observed(self) -> None:
        entities = named_entities("Find Anna Garcia.")
        observed, _ = split_by_presence(
            entities, normalise("\ufeffIdentity 3  Anna\ufeff Garcia"))
        self.assertEqual([e.text for e in observed], ["Anna Garcia"])

    def test_two_different_names_do_not_collide_once_stripped(self) -> None:
        """Only what renders as nothing is removed: visible spellings stay apart."""
        rows = normalise(
            "Identity 4  Alan\u200bCooperman\n"
            "Identity 5  Alan Coop\u00ader\n"      # Alan Cooper, hidden
            "Identity 6  Brandon Mill\u00ader"     # Brandon Miller, hidden
        )
        for absent in ("Alisha Ochoa", "Alan Cooperman Jr", "Brandon Millerman",
                       "Alan Coopér", "Brendon Miller", "Cooperman Alan"):
            _, unobserved = split_by_presence(
                named_entities("Find %s." % absent), rows)
            self.assertEqual([e.text for e in unobserved], [absent], absent)
        # And what WAS hidden is found, so those rows really were stripped.
        for present in ("Alan Cooper", "Brandon Miller"):
            observed, _ = split_by_presence(
                named_entities("Find %s." % present), rows)
            self.assertEqual([e.text for e in observed], [present], present)

    def test_stripping_does_not_join_two_names_into_a_third(self) -> None:
        """A zero-width SPACE folds to a space, so it can only split a run."""
        self.assertEqual(normalise("Alan\u200bCooper"), "alan cooper")
        self.assertEqual(normalise("Anna\u200bGarcia Brandon\u200bMiller"),
                         "anna garcia brandon miller")
        _, unobserved = split_by_presence(
            named_entities("Find Garcia Brandon."),
            normalise("Identity 7  Anna\u200bGarcia\nIdentity 8  Brandon Miller"))
        self.assertEqual([e.text for e in unobserved], ["Garcia Brandon"])

    def test_text_with_nothing_invisible_in_it_is_untouched(self) -> None:
        for text in ("Identity 1  Alan Cooper", "ALAN\n  COOPER", "Ａｌａｎ Cooper",
                     "", "Alan Coopér"):
            self.assertEqual(
                normalise(text),
                " ".join(unicodedata.normalize("NFKC", text).casefold().split()),
                text,
            )

class QueryEchoes(unittest.TestCase):
    """A miss that quotes the query back is not a retrieval.

    The contract is "a name the agent merely typed into a query can never make
    that name look retrieved". A filtered listing that matched nothing still
    prints its filter in the header, and a backend may word its miss with the
    literal in it, so the echo has to go before presence is decided.
    """

    ZERO_PAGE = (
        'result_handle=O2 filter="Christopher Hubbard" '
        "filter_columns=name page 1 rows 0 of 12 matched=0 materialized=12 "
        "total=12 source_complete=true matched_complete=true "
        "continuation=none outcome=complete has_more=false\n"
        'No rows matched the literal "Christopher Hubbard" in these fields: '
        "name. That is a complete zero for this literal in this listing; it is "
        "not evidence that the person or object does not exist."
    )

    def test_a_filtered_miss_does_not_retrieve_its_own_literal(self) -> None:
        entities = named_entities("Find Christopher Hubbard and list his rights.")
        observed, unobserved = split_by_presence(entities, normalise(self.ZERO_PAGE))
        self.assertEqual([e.text for e in observed], [])
        self.assertEqual([e.text for e in unobserved], ["Christopher Hubbard"])

    def test_a_backend_wording_of_the_miss_is_no_different(self) -> None:
        echo = ("Observation O4 (execute_workflow_query, in DirectoryExplorer)\n"
                "No identity matching 'Christopher Hubbard' was found.")
        entities = named_entities("Find Christopher Hubbard and list his rights.")
        observed, unobserved = split_by_presence(entities, normalise(echo))
        self.assertEqual([e.text for e in observed], [])
        self.assertEqual([e.text for e in unobserved], ["Christopher Hubbard"])

    def test_a_name_in_a_retrieved_row_stays_observed(self) -> None:
        page = (
            'result_handle=O2 filter="Alan Cooper" page 1 rows 1-1 of 1 '
            "matched=1\nuid-1  Alan Cooper  active"
        )
        entities = named_entities("Find Alan Cooper and list his rights.")
        observed, unobserved = split_by_presence(entities, normalise(page))
        self.assertEqual([e.text for e in observed], ["Alan Cooper"])
        self.assertEqual(unobserved, [])

    def test_only_the_echo_is_removed_never_the_line_around_it(self) -> None:
        """A row that merely contains a miss keeps every name it retrieved."""
        row = "uid-1  Alan Cooper  no manager found  active"
        entities = named_entities("Find Alan Cooper and list his rights.")
        observed, _ = split_by_presence(entities, normalise(row))
        self.assertEqual([e.text for e in observed], ["Alan Cooper"])
        self.assertEqual(strip_query_echoes(row), row)

class SubjectOfACommand(unittest.TestCase):
    """The context clause, not the corpus, answers "did you go there"."""

    def setUp(self) -> None:
        reset_runtime_state()
        self.addCleanup(reset_runtime_state)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.scope = scope_for(self.directory.name)
        self.archive = RuntimeHandleArchive(
            os.path.join(self.directory.name, "obs.sqlite3"))

    def _archive(self, alias: str, text: str, clause: str = "") -> None:
        self.archive.persist(
            self.scope, alias=alias, offload_order=int(alias[1:]),
            command_name="find_identity", step_index=int(alias[1:]), text=text,
            text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        )
        record_context_clause(self.scope, alias, clause)

    def test_a_listing_row_is_not_a_subject(self) -> None:
        self._archive("O1", "5 identities.\nuid-2  Anna Garcia", "DirectoryExplorer")
        haystack = subject_corpus(scope=self.scope, archive=self.archive)
        self.assertNotIn(normalise("Anna Garcia"), haystack)

    def test_a_context_instance_is_a_subject(self) -> None:
        self._archive("O1", "3 accounts.", "Identity 3f22  Anna Garcia")
        haystack = subject_corpus(scope=self.scope, archive=self.archive)
        self.assertIn(normalise("Anna Garcia"), haystack)

    def _rights(self) -> None:
        """The two rights and their application, opened as every attempt does."""
        self._archive("O8", "477 holders.",
                      "Permission 85cd  Active Directory_Cloud Administrator")
        self._archive("O9", "513 holders.",
                      "Permission 3e3d  Active Directory_Compliance Officer")

    def test_the_nudge_names_the_people_the_run_never_opened(self) -> None:
        self._archive("O1", "1 identity.\nuid-1  Alan Cooper", "DirectoryExplorer")
        self._archive("O2", "29 permissions.", "Identity 28c5  Alan Cooper")
        self._rights()
        text, report = build_nudge(
            user_query=CARD, iterations_left=10,
            scope=self.scope, archive=self.archive,
        )
        self.assertTrue(report.fired)
        self.assertNotIn("Alan Cooper", text)
        for name in ("Alisha Ochoa", "Anna Garcia", "Barbara Sanchez",
                     "Brandon Miller", "Christopher Hubbard"):
            self.assertIn(name, text)
        self.assertIn("10 more actions", text)
        self.assertIn("do not ask the user", text)
        self.assertEqual(report.subjects_named, 5)

    def test_a_run_that_reached_everyone_is_not_nudged(self) -> None:
        for index, name in enumerate(
            ["Alan Cooper", "Alisha Ochoa", "Anna Garcia", "Barbara Sanchez",
             "Brandon Miller", "Christopher Hubbard"], start=1
        ):
            self._archive(f"O{index}", "rows", f"Identity uid-{index}  {name}")
        self._rights()
        text, report = build_nudge(
            user_query=CARD, iterations_left=10,
            scope=self.scope, archive=self.archive,
        )
        self.assertEqual(text, "")
        self.assertFalse(report.fired)
        self.assertEqual(report.reason, "every named item was already a subject")

    def test_a_possessive_spelling_never_nudges_an_opened_subject(self) -> None:
        # ido-jf6/F29. Both people were opened; the nudge used to fire on
        # "Alan Cooper's" and "Brandon Miller's", which no clause can contain.
        text, report = build_nudge(
            user_query="List Alan Cooper's rights and Brandon Miller's accounts.",
            iterations_left=10,
            clauses=normalise("Identity 28c5  Alan Cooper\nIdentity 9a1  Brandon Miller"),
        )
        self.assertEqual(text, "")
        self.assertFalse(report.fired)
        self.assertEqual(report.subjects_missing, [])

    def test_a_workflow_that_declares_no_instance_identity_is_never_nudged(self) -> None:
        # Every person was opened, but the contexts declare no instance label,
        # so each clause is a bare context name that no name can ever match.
        self._archive("O1", "1 identity.\nuid-1  Alan Cooper", "DirectoryExplorer")
        self._archive("O2", "29 permissions.", "Identity")
        self._archive("O3", "4 accounts.", "Account")
        text, report = build_nudge(
            user_query=CARD, iterations_left=10,
            scope=self.scope, archive=self.archive,
        )
        self.assertEqual(text, "")
        self.assertFalse(report.fired)
        self.assertEqual(report.reason, "no recorded clause names an instance")

    def test_one_labelled_clause_is_enough_to_judge_the_others(self) -> None:
        text, report = build_nudge(
            user_query=CARD, iterations_left=10,
            clauses=["DirectoryExplorer", "Identity 28c5  Alan Cooper"],
        )
        self.assertTrue(report.fired)
        self.assertNotIn("Alan Cooper", text)
        self.assertIn("Brandon Miller", text)

    def test_the_note_claims_only_that_an_item_was_never_a_subject(self) -> None:
        text, report = build_nudge(
            user_query=CARD, iterations_left=10,
            clauses=["Identity 28c5  Alan Cooper"],
        )
        self.assertTrue(report.fired)
        self.assertIn("the subject of any command", text)
        self.assertNotIn("nothing about them has been retrieved", text)
        self.assertNotIn("has been retrieved", text)

    def test_no_recorded_clause_says_nothing(self) -> None:
        text, report = build_nudge(
            user_query=CARD, iterations_left=10,
            scope=self.scope, archive=self.archive,
        )
        self.assertEqual(text, "")
        self.assertEqual(report.reason, "no context clauses recorded")

    def test_a_turn_with_no_room_is_never_nudged(self) -> None:
        text, report = build_nudge(
            user_query=CARD, iterations_left=NUDGE_MIN_ITERS_LEFT - 1,
            clauses=normalise("DirectoryExplorer"),
        )
        self.assertEqual(text, "")
        self.assertEqual(report.reason, "no room to act")

    def test_the_note_is_bounded_and_counts_what_it_cut(self) -> None:
        names = [f"Personname Number{index:03d}" for index in range(120)]
        text, named = nudge_block(names, 5)
        self.assertLessEqual(len(text.encode("utf-8")), NUDGE_MAX_BYTES)
        self.assertLess(named, len(names))
        self.assertIn(f"and {len(names) - named} more", text)

    def test_the_same_state_always_gives_the_same_note(self) -> None:
        clauses = normalise("Identity 28c5  Alan Cooper")
        first, _ = build_nudge(user_query=CARD, iterations_left=7, clauses=clauses)
        second, _ = build_nudge(user_query=CARD, iterations_left=7, clauses=clauses)
        self.assertEqual(first, second)
        self.assertNotEqual("", first)


class Stub:
    """Stands in for ``self.react``: replays a fixed list of actions."""

    def __init__(self, actions) -> None:
        self.actions = list(actions)
        self.calls: list[str] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs["trajectory"])
        name, args = self.actions[min(len(self.calls) - 1, len(self.actions) - 1)]
        return dspy.Prediction(next_thought="t", next_tool_name=name,
                               next_tool_args=args)


class LoopHook(unittest.TestCase):
    """The finish action, and one nudge per turn.

    There is no ``FW_ROSTER_NUDGE`` setting: the loop check runs on every
    finish action, so the interesting case is not a disabled loop but a run
    that has nothing to be nudged about.
    """

    def setUp(self) -> None:
        reset_runtime_state()
        self.addCleanup(reset_runtime_state)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        from fastworkflow.utils.react import fastWorkflowReAct

        def a_tool(value: str = "") -> str:
            """A tool."""
            return f"observed {value}"

        self.scope = scope_for(self.directory.name)
        self.archive = RuntimeHandleArchive(
            os.path.join(self.directory.name, "obs.sqlite3"))
        text = "1 identity.\nuid-1  Alan Cooper"
        self.archive.persist(
            self.scope, alias="O1", offload_order=1, command_name="find_identity",
            step_index=0, text=text,
            text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        )
        record_context_clause(self.scope, "O1", "Identity 28c5  Alan Cooper")
        for alias, clause in (
            ("O8", "Permission 85cd  Active Directory_Cloud Administrator"),
            ("O9", "Permission 3e3d  Active Directory_Compliance Officer"),
        ):
            self.archive.persist(
                self.scope, alias=alias, offload_order=int(alias[1:]),
                command_name="show_holders", step_index=int(alias[1:]),
                text="holders", text_sha256=hashlib.sha256(b"holders").hexdigest(),
            )
            record_context_clause(self.scope, alias, clause)
        self.agent = fastWorkflowReAct("user_query -> final_answer",
                                       tools=[a_tool], max_iters=12)
        self.agent.continuation_scope = self.scope
        self.agent.observation_archive = self.archive

    def _run(self, actions):
        self.agent.react = Stub(actions)
        trajectory: dict = {}
        self.agent._roster_nudges_fired = 0
        self.agent.iteration_counter = 0
        self.agent._run_loop(trajectory, 0, {"user_query": CARD}, 12, 0)
        return trajectory

    def test_a_request_with_no_named_items_ends_at_the_finish_action(self) -> None:
        """The check runs on every finish action; this is what it
        does when there is nothing to say: the trajectory is the one a finish
        action always produced, the nudge is recorded as not fired, and the
        reason names why rather than naming a setting."""
        self.agent.react = Stub([("finish", {}), ("a_tool", {"value": "x"})])
        trajectory: dict = {}
        self.agent._roster_nudges_fired = 0
        self.agent.iteration_counter = 0
        self.agent._run_loop(trajectory, 0, {"user_query": "list everything"}, 12, 0)
        self.assertEqual(
            trajectory,
            {"thought_0": "t", "tool_name_0": "finish", "tool_args_0": {},
             "observation_0": "Completed."},
        )
        event = [e for e in snapshot_events() if e["kind"] == "roster_nudge"][0]
        self.assertFalse(event["fired"])

    def test_the_nudge_returns_control_to_the_loop(self) -> None:
        trajectory = self._run([("finish", {}), ("a_tool", {"value": "more"}),
                                ("finish", {})])
        self.assertIn("Harness check before this turn ends",
                      trajectory["observation_0"])
        self.assertIn("Brandon Miller", trajectory["observation_0"])
        self.assertNotIn("Alan Cooper", trajectory["observation_0"])
        self.assertEqual(trajectory["observation_1"], "observed more")
        self.assertEqual(trajectory["tool_name_2"], "finish")
        self.assertEqual(trajectory["observation_2"], "Completed.")

    def test_at_most_one_nudge_per_turn(self) -> None:
        trajectory = self._run([("finish", {})])
        self.assertIn("Harness check", trajectory["observation_0"])
        self.assertEqual(trajectory["observation_1"], "Completed.")
        self.assertNotIn("observation_2", trajectory)
        fired = [event for event in snapshot_events()
                 if event["kind"] == "roster_nudge" and event["fired"]]
        self.assertEqual(len(fired), 1)

    def test_the_event_carries_what_the_summarizer_counts(self) -> None:
        self._run([("finish", {})])
        event = [e for e in snapshot_events() if e["kind"] == "roster_nudge"][0]
        self.assertTrue(event["fired"])
        self.assertEqual(event["subjects_total"], 9)
        self.assertEqual(event["subjects_named"], 5)
        self.assertIn("Brandon Miller", event["subjects_missing"])
        self.assertGreater(event["text_bytes"], 0)
        self.assertLessEqual(event["text_bytes"], NUDGE_MAX_BYTES)
        self.assertGreater(event["iterations_left"], 0)

    def test_never_on_a_turn_with_no_room(self) -> None:
        self.agent.react = Stub([("finish", {})])
        trajectory: dict = {}
        self.agent._roster_nudges_fired = 0
        self.agent.iteration_counter = 11  # max_iters is 12: nothing left to do
        self.agent._run_loop(trajectory, 0, {"user_query": CARD}, 12, 0)
        self.assertEqual(trajectory["observation_0"], "Completed.")
        self.assertFalse(self.agent._exhausted_last_run)
        event = [e for e in snapshot_events() if e["kind"] == "roster_nudge"][0]
        self.assertFalse(event["fired"])
        self.assertEqual(event["reason"], "no room to act")

    def test_a_failure_leaves_the_loop_alone(self) -> None:
        with mock.patch.object(answer_coverage, "build_nudge",
                               side_effect=RuntimeError("boom")):
            trajectory = self._run([("finish", {})])
        self.assertEqual(trajectory["observation_0"], "Completed.")
        self.assertEqual([e["kind"] for e in snapshot_events()],
                         ["roster_nudge_failed"])

    def test_both_controls_can_be_disabled(self) -> None:
        self.agent.finish_reminders_enabled = False
        self.agent.coverage_instructions_enabled = False
        trajectory = self._run([("finish", {}), ("a_tool", {"value": "more"})])
        self.assertEqual(trajectory["observation_0"], "Completed.")
        self.assertNotIn("tool_name_1", trajectory)
        recorder = Recorder()
        self.agent.extract = recorder
        self.agent._extract_prediction(trajectory, user_query=CARD)
        self.assertEqual(len(recorder.calls), 1)
        self.assertNotIn("Coverage of this run:", recorder.calls[0])


if __name__ == "__main__":
    unittest.main()


#: ido-8ps.28. The statement this fixture produced at 90a1565, byte for byte --
#: which is also the statement it produces now whenever no subject has anything
#: to state. Frozen here, not recomputed, because it is a claim about bytes a
#: model received on a day and a recomputed expectation would move with the code
#: it is meant to pin.
STATEMENT_AT_90A1565 = (
    "Coverage of this run: the loop ended normally. These named items from the "
    "request appear in no retrieved observation: Active Directory; Active "
    "Directory_Cloud Administrator; Active Directory_Compliance Officer; "
    "Alisha Ochoa; Anna Garcia; Barbara Sanchez; Brandon Miller; Christopher "
    'Hubbard. For each of them report "not retrieved" and nothing else - no '
    "value, no unavailability, no absence. These named items of the request DO "
    "appear in this run's observations: Alan Cooper. Every other named item of "
    "the request WAS retrieved: it appears in this run's observations and must "
    'be reported from them. Do not write "not retrieved", "not available", "no '
    'data", or any other statement of absence about an item that is not named '
    "in the unobserved list above. For items that appear, report only what the "
    "observations show."
)



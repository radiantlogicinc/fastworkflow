"""The extract step reads the evidence behind offload labels.

Offline only. Nothing here starts a server, calls a model or touches a backend:
the archive and the result-handle store are real SQLite files in a temp dir, the
extract module is a recorder, and every assertion is about bytes that are already
on disk before the test begins.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
import unittest
from unittest import mock

import dspy
import pytest
from litellm import ContextWindowExceededError

from fastworkflow.answer_rehydration import (
    ANSWER_REHYDRATION_MAX_BYTES_ENV,
    DEFAULT_MAX_BYTES,
    KIND_LABEL,
    MIN_MAX_BYTES,
    NOT_REHYDRATED_KEY,
    NOT_REHYDRATED_PREFIX,
    max_bytes_from_env,
    rehydrate,
    rehydrated_label,
    trajectory_bytes,
)
from fastworkflow.observation_offloading.archive import (
    RuntimeHandleArchive,
    RuntimeHandleScope,
)
from fastworkflow.observation_offloading.labels import (
    alias_line,
    offload_label,
    printed_context,
)
from fastworkflow.observation_offloading.state import (
    reset_observation_state,
    snapshot_events,
)
from fastworkflow.utils.react import fastWorkflowReAct

# The budgets derive from the agent model's window; pin it to the reference.
pytestmark = pytest.mark.usefixtures("reference_agent_window")


def scope_for(name: str) -> RuntimeHandleScope:
    return RuntimeHandleScope(
        channel_id="c", turn_key="turn-1",
    )


def rows(prefix: str, count: int, width: int = 40) -> list[str]:
    return ["%s-%03d | %s" % (prefix, index, "x" * width) for index in range(count)]


class Fixture:
    """A real archive, with one offloaded observation behind a label."""

    def __init__(self, directory: str, *, listing_rows: int = 60,
                 record_subject: bool = True) -> None:
        self.scope = scope_for(directory)
        path = os.path.join(directory, "obs.sqlite3")
        self.archive = RuntimeHandleArchive(path)
        # (a) an offloaded observation, archived without its presentation line
        self.o1_text = "identity_uid | rights\n" + "\n".join(rows("ident", 30))
        self.archive.persist(
            self.scope, alias="O0", command_name="list_permissions", step_index=0, text=self.o1_text,
            text_sha256=hashlib.sha256(self.o1_text.encode("utf-8")).hexdigest(),
            context_clause="Identity 28c5aeb5 Jane Roe" if record_subject else None,
        )
        self.listing_rows = rows("holder", listing_rows)

    def trajectory(self) -> dict:
        return {
            "thought_0": "look up the holders",
            "tool_name_0": "execute_workflow_query",
            "tool_args_0": {"command": "list_permissions"},
            "observation_0": offload_label(
                alias="O0", command_name="list_permissions",
                response=self.o1_text, description="the entitlement rows",
            ),
            "thought_1": "list the holders",
            "tool_name_1": "execute_workflow_query",
            "tool_args_1": {"command": "show_owners"},
            "observation_1": (
                alias_line("O1", "Permission 6fadcafc Cloud Administrator")
                + "result_handle=O1 page 1 rows 1-25 of 60\n"
                + "\n".join(self.listing_rows[:25])
            ),
            "thought_2": "page it",
            "tool_name_2": "execute_workflow_query",
            "tool_args_2": {"command": "fetch_result_page"},
            "observation_2": (
                alias_line("O2")
                + "result_handle=O1 page 2 rows 26-50 of 60\n"
                + "\n".join(self.listing_rows[25:50])
            ),
        }


class Budget(unittest.TestCase):
    """The budget: derived from the window, overridable, defended.

    Rehydration is what the extract step always does. What is left here is
    the byte budget, which is a fraction of the model's context window with
    ``FW_ANSWER_REHYDRATION_MAX_BYTES`` as a tuning override.
    ``tests/test_context_budget.py`` owns the derivation; this owns
    the module's view of it.
    """

    def setUp(self) -> None:
        os.environ.pop(ANSWER_REHYDRATION_MAX_BYTES_ENV, None)

    tearDown = setUp

    def test_the_derived_budget_is_the_accepted_value(self) -> None:
        self.assertEqual(max_bytes_from_env(), DEFAULT_MAX_BYTES)
        self.assertEqual(DEFAULT_MAX_BYTES, 250_000)

    def test_the_override_is_read_and_defended(self) -> None:
        os.environ[ANSWER_REHYDRATION_MAX_BYTES_ENV] = "60000"
        self.assertEqual(max_bytes_from_env(), 60_000)
        os.environ[ANSWER_REHYDRATION_MAX_BYTES_ENV] = "not-a-number"
        self.assertEqual(max_bytes_from_env(), DEFAULT_MAX_BYTES)
        os.environ[ANSWER_REHYDRATION_MAX_BYTES_ENV] = str(MIN_MAX_BYTES - 1)
        self.assertEqual(max_bytes_from_env(), DEFAULT_MAX_BYTES)

    def test_the_env_file_is_read_first(self) -> None:
        import fastworkflow

        with mock.patch.dict(fastworkflow._env_vars,
                             {ANSWER_REHYDRATION_MAX_BYTES_ENV: "60000"},
                             clear=False):
            self.assertEqual(max_bytes_from_env(), 60_000)


class RehydratesEachKind(unittest.TestCase):

    def setUp(self) -> None:
        reset_observation_state()
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.addCleanup(reset_observation_state)
        self.fixture = Fixture(self.directory.name)

    def rehydrate(self, trajectory=None, budget=DEFAULT_MAX_BYTES):
        return rehydrate(
            trajectory if trajectory is not None else self.fixture.trajectory(),
            scope=self.fixture.scope, archive=self.fixture.archive,
            budget=budget,
        )

    def test_a_label_becomes_the_archived_observation(self) -> None:
        copy, report = self.rehydrate()
        self.assertIn(self.fixture.o1_text, copy["observation_0"])
        self.assertEqual(report.counts[KIND_LABEL], 1)

    def test_the_alias_and_context_lines_are_preserved(self) -> None:
        copy, _ = self.rehydrate()
        self.assertTrue(copy["observation_0"].startswith(
            "Observation O0 (execute_workflow_query ran in context 'Identity' and 28c5aeb5 Jane Roe)\n"))
        self.assertEqual(printed_context(copy["observation_0"]),
                         "Identity 28c5aeb5 Jane Roe")

    def test_the_change_suffix_is_restored_with_the_context(self) -> None:
        fixture = Fixture(tempfile.mkdtemp())
        fixture.archive.persist(
            fixture.scope, alias="O2", command_name="switch_account", step_index=2,
            text="switched\n", text_sha256=hashlib.sha256(b"switched\n").hexdigest(),
            context_clause="Account 1", context_changed=True,
        )
        restored = rehydrated_label("O2", scope=fixture.scope, archive=fixture.archive)
        self.assertEqual(
            restored,
            "Observation O2 (execute_workflow_query ran in prior context 'Account' and 1; "
            "and resulted in a context change)\nswitched\n",
        )
        self.assertTrue(fixture.archive.get(fixture.scope, "O2")["context_changed"])

    def test_an_alias_with_no_recorded_context_prints_the_plain_a1_line(self) -> None:
        fixture = Fixture(tempfile.mkdtemp(), record_subject=False)
        copy, _ = rehydrate(
            fixture.trajectory(), scope=fixture.scope, archive=fixture.archive,
            budget=DEFAULT_MAX_BYTES,
        )
        self.assertTrue(copy["observation_0"].startswith(
            "Observation O0 (execute_workflow_query)\n"))


    def test_nothing_is_invented_for_an_unarchived_alias(self) -> None:
        empty_dir = tempfile.mkdtemp()
        empty_scope = scope_for(empty_dir)
        empty_archive = RuntimeHandleArchive(os.path.join(empty_dir, "empty.sqlite3"))
        trajectory = {
            "tool_name_0": "execute_workflow_query",
            "tool_args_0": {"command": "list_permissions"},
            "observation_0": offload_label(
                alias="O0", command_name="list_permissions", response="x" * 4000),
        }
        copy, report = rehydrate(
            trajectory, scope=empty_scope, archive=empty_archive, budget=DEFAULT_MAX_BYTES,
        )
        self.assertEqual(copy["observation_0"], trajectory["observation_0"])
        self.assertIn("O0", report.unresolved_aliases)

    def test_a_plain_observation_is_untouched(self) -> None:
        trajectory = self.fixture.trajectory()
        trajectory["observation_3"] = "Observation O4 (execute_workflow_query)\nfine"
        trajectory["tool_name_3"] = "execute_workflow_query"
        copy, _ = self.rehydrate(trajectory)
        self.assertEqual(copy["observation_3"], trajectory["observation_3"])

    def test_the_input_trajectory_is_never_mutated(self) -> None:
        trajectory = self.fixture.trajectory()
        before = dict(trajectory)
        copy, _ = self.rehydrate(trajectory)
        self.assertEqual(trajectory, before)
        self.assertIsNot(copy, trajectory)

    def test_the_archive_is_only_read(self) -> None:
        before_archive = self.fixture.archive.list(self.fixture.scope)
        self.rehydrate()
        self.assertEqual(self.fixture.archive.list(self.fixture.scope),
                         before_archive)


class BudgetAndOrder(unittest.TestCase):

    def setUp(self) -> None:
        reset_observation_state()
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.addCleanup(reset_observation_state)
        self.fixture = Fixture(self.directory.name)

    def test_the_budget_is_respected(self) -> None:
        trajectory = self.fixture.trajectory()
        budget = trajectory_bytes(trajectory) + 200
        copy, report = rehydrate(
            trajectory, scope=self.fixture.scope, archive=self.fixture.archive,
            budget=budget,
        )
        self.assertLessEqual(trajectory_bytes(copy) - len(
            copy.get(NOT_REHYDRATED_KEY, "").encode("utf-8")), budget)
        self.assertTrue(report.dropped_aliases)

    def test_the_dropped_alias_line(self) -> None:
        trajectory = self.fixture.trajectory()
        copy, report = rehydrate(
            trajectory, scope=self.fixture.scope, archive=self.fixture.archive,
            budget=trajectory_bytes(trajectory),
        )
        self.assertEqual(copy[NOT_REHYDRATED_KEY], report.note_line)
        self.assertTrue(report.note_line.startswith(NOT_REHYDRATED_PREFIX))
        self.assertEqual(report.note_line,
                         NOT_REHYDRATED_PREFIX + ", ".join(report.dropped_aliases))
        # The agent was told observations are normally restored, so the
        # extractor is told to state the omission in the answer itself.
        self.assertIn("evidence limit was reached", report.note_line)
        self.assertIn("say in the final answer that the rows of these observations "
                      "are not included in it", report.note_line)
        # Deterministic: ascending by step index, every time.
        self.assertEqual(report.dropped_aliases,
                         sorted(report.dropped_aliases,
                                key=lambda alias: int(alias[1:])))

    def test_no_drop_line_when_everything_fits(self) -> None:
        copy, report = rehydrate(
            self.fixture.trajectory(), scope=self.fixture.scope,
            archive=self.fixture.archive,             budget=DEFAULT_MAX_BYTES,
        )
        self.assertNotIn(NOT_REHYDRATED_KEY, copy)
        self.assertEqual(report.dropped_aliases, [])

    def test_the_report_counts_the_bytes(self) -> None:
        trajectory = self.fixture.trajectory()
        copy, report = rehydrate(
            trajectory, scope=self.fixture.scope, archive=self.fixture.archive,
            budget=DEFAULT_MAX_BYTES,
        )
        self.assertEqual(report.bytes_before, trajectory_bytes(trajectory))
        self.assertEqual(report.bytes_after, trajectory_bytes(copy))
        self.assertGreater(report.bytes_after, report.bytes_before)


class Recorder:
    """Stands in for ``self.extract``: records what it was called with."""

    def __init__(self, *, fail_times: int = 0) -> None:
        self.calls: list[str] = []
        self.fail_times = fail_times

    def __call__(self, **kwargs):
        self.calls.append(kwargs["trajectory"])
        if len(self.calls) <= self.fail_times:
            raise ContextWindowExceededError(
                message="too long", model="m", llm_provider="p")
        return dspy.Prediction(answer="done")



def build_agent() -> fastWorkflowReAct:
    def a_tool(value: str) -> str:
        """A tool."""
        return value

    return fastWorkflowReAct("user_query -> answer", tools=[a_tool], max_iters=2)


class ExtractHook(unittest.TestCase):

    def setUp(self) -> None:
        reset_observation_state()
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.addCleanup(reset_observation_state)
        os.environ.pop(ANSWER_REHYDRATION_MAX_BYTES_ENV, None)
        self.addCleanup(
            lambda: os.environ.pop(ANSWER_REHYDRATION_MAX_BYTES_ENV, None))
        self.fixture = Fixture(self.directory.name)
        self.agent = build_agent()
        for target, value in (("current_scope", self.fixture.scope),
                              ("durable_archive", self.fixture.archive)):
            patcher = mock.patch(
                f"fastworkflow.observation_offloading.state.{target}",
                return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_the_extractor_is_handed_the_evidence(self) -> None:
        """Nothing set, and the extract call still gets the evidence."""
        trajectory = self.fixture.trajectory()
        plain = self.agent._format_trajectory(trajectory)
        recorder = Recorder()
        self.agent.extract = recorder
        self.agent._extract_prediction(trajectory, user_query="q")
        self.assertNotEqual(recorder.calls[0], plain)
        self.assertIn(self.fixture.o1_text, recorder.calls[0])

    def test_the_react_trajectory_is_unchanged_after_extraction(self) -> None:
        trajectory = self.fixture.trajectory()
        before = dict(trajectory)
        self.agent.extract = Recorder()
        self.agent._extract_prediction(trajectory, user_query="q")
        self.assertEqual(trajectory, before)

    def test_the_events_carry_the_measures(self) -> None:
        os.environ[ANSWER_REHYDRATION_MAX_BYTES_ENV] = "80000"
        self.agent.extract = Recorder()
        self.agent._extract_prediction(self.fixture.trajectory(), user_query="q")
        events = snapshot_events()
        started = [e for e in events if e["kind"] == "rehydration_started"]
        finished = [e for e in events if e["kind"] == "rehydration_finished"]
        self.assertEqual(len(started), 1)
        self.assertEqual(len(finished), 1)
        self.assertEqual(started[0]["budget_bytes"], 80_000)
        self.assertGreater(finished[0]["bytes_after"], finished[0]["bytes_before"])
        self.assertEqual(finished[0]["rehydrated_labels"], 1)
        self.assertIn("dropped_aliases", finished[0])
        self.assertFalse(finished[0]["rehydration_overflow"])
        self.assertIn("extract_prompt_tokens", finished[0])
        self.assertIsInstance(finished[0]["extract_duration_ms"], float)

    def test_the_overflow_fallback_still_truncates_and_is_recorded(self) -> None:
        trajectory = self.fixture.trajectory()
        recorder = Recorder(fail_times=1)
        self.agent.extract = recorder
        self.agent._extract_prediction(trajectory, user_query="q")
        self.assertEqual(len(recorder.calls), 2)
        overflow = [e for e in snapshot_events()
                    if e["kind"] == "rehydration_overflow"]
        self.assertEqual(len(overflow), 1)
        finished = [e for e in snapshot_events()
                    if e["kind"] == "rehydration_finished"][0]
        self.assertTrue(finished["rehydration_overflow"])
        # The fallback truncated the COPY; the loop's trajectory still has step 0.
        self.assertIn("observation_0", trajectory)

    def test_a_broken_archive_costs_the_evidence_and_not_the_answer(self) -> None:
        """The extract call still happens, over the pointers it already had.

        What the fallback owes is the answer, and the record of why: no
        archived text reaches the extractor and ``rehydration_failed`` is
        filed.
        """
        trajectory = self.fixture.trajectory()
        recorder = Recorder()
        self.agent.extract = recorder
        with mock.patch("fastworkflow.answer_rehydration.rehydrate",
                        side_effect=RuntimeError("no archive")):
            self.agent._extract_prediction(trajectory, user_query="q")
        self.assertEqual(len(recorder.calls), 1)
        self.assertNotIn(self.fixture.o1_text, recorder.calls[0])
        self.assertTrue([e for e in snapshot_events()
                         if e["kind"] == "rehydration_failed"])


class ContinuationSite(unittest.TestCase):
    """Answer extraction goes through the rehydration hook."""

    def test_finish_prediction_goes_through_the_hook(self) -> None:
        agent = fastWorkflowReAct(
            "user_query -> answer", tools=[lambda value: value], max_iters=2)
        with mock.patch.object(
            fastWorkflowReAct, "_extract_prediction",
            return_value={"answer": "a"},
        ) as hook:
            agent._extract_prediction({"thought_0": "t"}, user_query="q")
        hook.assert_called_once()




class WhatTheStopActuallyCost(unittest.TestCase):
    """The note names lost evidence, and each alias costs once.

    The dropped list is read by the extractor as "evidence exists under these
    observations and you have not got it", so an observation that lost nothing
    to the stop must not be in it, and one archived observation must not be
    paid for twice because two steps printed its label.
    """

    def setUp(self) -> None:
        reset_observation_state()
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.addCleanup(reset_observation_state)
        self.fixture = Fixture(self.directory.name)

    def rehydrate(self, trajectory, budget):
        return rehydrate(
            trajectory, scope=self.fixture.scope, archive=self.fixture.archive,
            budget=budget,
        )

    def with_a_plain_step(self) -> dict:
        """The fixture, with a plain inline observation O9 older than the page.

        O1 (label) stays at step 0, O9 goes in at step 1, and the listing and
        the page move up to 2 and 3 -- so O9 is older than the stop and has
        nothing behind it but its own text.
        """
        base = self.fixture.trajectory()
        out = {key: value for key, value in base.items() if key.endswith("_0")}
        out.update({
            "thought_1": "count them",
            "tool_name_1": "execute_workflow_query",
            "tool_args_1": {"command": "count_identities"},
            "observation_1": (alias_line("O1", "ItemExplorer")
                              + "There are 5 identities."),
        })
        for key, value in base.items():
            if key.endswith("_1"):
                out[key[:-1] + "2"] = value
            elif key.endswith("_2"):
                out[key[:-1] + "3"] = value
        return out

    def test_a_plain_observation_is_not_listed_as_not_rehydrated(self) -> None:
        trajectory = self.with_a_plain_step()
        copy, report = self.rehydrate(
            trajectory, budget=trajectory_bytes(trajectory) + 10)
        # The walk stops on the newest candidate, so everything else is older.
        self.assertEqual(report.stopped_on, "O0")
        self.assertEqual(report.rehydrated, [])
        # O9 is whole in the copy, so nothing about it is unresolved.
        self.assertEqual(copy["observation_1"], trajectory["observation_1"])
        self.assertIn("There are 5 identities.", copy["observation_1"])
        self.assertNotIn("O1", report.dropped_aliases)
        self.assertNotIn("O1", copy.get(NOT_REHYDRATED_KEY, ""))
        # And the aliases that really did lose evidence are still all named.
        self.assertEqual(report.dropped_aliases, ["O0"])

    def test_one_label_alias_on_two_steps_spends_the_budget_once(self) -> None:
        base = self.fixture.trajectory()
        _, plain = self.rehydrate(base, budget=DEFAULT_MAX_BYTES)
        single = [item for item in plain.rehydrated if item["alias"] == "O0"][0]

        twice = dict(base)
        twice.update({
            "thought_5": "read it again",
            "tool_name_5": "execute_workflow_query",
            "tool_args_5": {"command": "list_permissions"},
            "observation_5": offload_label(
                alias="O5", command_name="list_permissions",
                response=self.fixture.o1_text, description="the entitlement rows",
            ),
        })
        digest = hashlib.sha256(self.fixture.o1_text.encode("utf-8")).hexdigest()
        self.fixture.archive.persist(
            self.fixture.scope, alias="O5", command_name="list_permissions", step_index=5, text=self.fixture.o1_text,
            text_sha256=digest, context_clause="Identity 28c5aeb5 Jane Roe",
        )
        copy, report = self.rehydrate(twice, budget=DEFAULT_MAX_BYTES)
        entries = [item for item in report.rehydrated if item["alias"] == "O5"]
        self.assertEqual(len(entries), 1)
        # The most recent step is the one that carries the archived text...
        self.assertEqual(entries[0]["step_index"], 5)
        self.assertEqual(entries[0]["added_bytes"], single["added_bytes"])
        self.assertEqual(report.counts[KIND_LABEL], 1)
        # ...and the older step keeps its label, unpaid for and undropped.
        self.assertEqual(copy["observation_0"], base["observation_0"])
        self.assertNotIn("O5", report.dropped_aliases)
        self.assertEqual(
            report.bytes_after - report.bytes_before,
            sum(item["added_bytes"] for item in report.rehydrated),
        )

    def test_a_duplicate_label_older_than_the_stop_is_not_dropped(self) -> None:
        """Its text is in the copy under the newer step: nothing was lost."""
        base = self.fixture.trajectory()
        twice = dict(base)
        twice.update({
            "thought_5": "read it again",
            "tool_name_5": "execute_workflow_query",
            "tool_args_5": {"command": "list_permissions"},
            "observation_5": offload_label(
                alias="O5", command_name="list_permissions",
                response=self.fixture.o1_text, description="the entitlement rows",
            ),
        })
        digest = hashlib.sha256(self.fixture.o1_text.encode("utf-8")).hexdigest()
        self.fixture.archive.persist(
            self.fixture.scope, alias="O5", command_name="list_permissions", step_index=5, text=self.fixture.o1_text,
            text_sha256=digest,
        )
        label_added = len(
            rehydrated_label("O5", scope=self.fixture.scope,
                             archive=self.fixture.archive).encode("utf-8")
        ) - len(twice["observation_5"].encode("utf-8"))
        budget = trajectory_bytes(twice) + label_added
        copy, report = self.rehydrate(twice, budget=budget)
        self.assertEqual([item["alias"] for item in report.rehydrated], ["O5"])
        self.assertEqual(report.stopped_on, "")
        self.assertNotIn("O5", report.dropped_aliases)
        self.assertEqual(copy["observation_0"], base["observation_0"])


if __name__ == "__main__":
    unittest.main()

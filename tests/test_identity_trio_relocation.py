"""The identity trio, after its move out of ``result_handles.paging``.

Both trio consumers sit inside a bare ``except Exception: pass``
(``fastworkflow/command_executor.py`` ``_remember_execute_context``, and
``observation_offloading/state.py`` ``durable_archive``), so a botched move
does not raise -- it silently records nothing and strips every context clause,
contextual alias line, finish-reminder haystack and rehydration label from the
turn. These tests assert the RESULT is non-empty rather than that the import
resolves, which is the only way that failure mode is visible.
"""
import unittest

from fastworkflow import tracing
from fastworkflow.command_executor import CommandExecutor
from fastworkflow.observation_offloading import state
from fastworkflow.observation_offloading.state import (
    context_clause_of,
    current_execute_alias,
    current_scope,
    default_scope,
)


class FakeAgent:
    """An agent with one execute step in flight, as ReAct leaves it."""

    def __init__(self):
        self.current_trajectory = {
            "tool_name_0": "execute_workflow_query",
            "observation_0": "done",
            "tool_name_1": "execute_workflow_query",
        }
        self.execute_ordinal_by_step = {0: 1, 1: 2}


class FakeHost:
    def __init__(self, agent):
        self.workflow_tool_agent = agent


class FakeContext:
    def __str__(self):
        return "Alan Cooper"


class FakeWorkflow:
    """Enough of a workflow for ``context_clause_for`` to name a context.

    ``is_current_command_context_root`` must be False or ``context_identity``
    returns ``("", "")`` by design: the root context is deliberately unnamed.
    """

    is_current_command_context_root = False
    current_command_context_name = "Account"
    folderpath = "/nonexistent-workflow"

    @property
    def current_command_context(self):
        return FakeContext()


class FakeChatSession:
    pass


class TrioLivesOnState(unittest.TestCase):
    def test_the_trio_is_importable_from_state(self):
        self.assertEqual(current_scope.__module__,
                         "fastworkflow.observation_offloading.state")
        self.assertEqual(current_execute_alias.__module__,
                         "fastworkflow.observation_offloading.state")
        self.assertEqual(state._current_agent.__module__,
                         "fastworkflow.observation_offloading.state")

    def test_alias_reads_the_in_flight_step_from_the_ledger(self):
        self.assertEqual(current_execute_alias(FakeAgent()), "O2")

    def test_scope_falls_back_to_the_process_default(self):
        self.assertEqual(current_scope(), default_scope())


class DispatchRecordsANonEmptyClause(unittest.TestCase):
    """The verification Step 2 demands: the clause the dispatch files is real."""

    def setUp(self):
        self.agent = FakeAgent()
        self.host = FakeHost(self.agent)
        state.reset_observation_state()
        self.addCleanup(state.reset_observation_state)

    def test_remember_execute_context_files_a_non_empty_clause(self):
        session = FakeChatSession()
        original = CommandExecutor._active_workflow
        CommandExecutor._active_workflow = staticmethod(lambda cs: FakeWorkflow())
        self.addCleanup(setattr, CommandExecutor, "_active_workflow", original)

        with tracing.host_scope(self.host):
            alias = current_execute_alias()
            self.assertEqual(alias, "O2")
            CommandExecutor._remember_execute_context(session)
            clause = context_clause_of(current_scope(), alias)

        self.assertTrue(clause, "the dispatch recorded no context clause at all")
        self.assertIn("Account", clause)

    def test_the_archive_lookup_resolves_the_agent_without_paging(self):
        self.agent.observation_archive = object()
        with tracing.host_scope(self.host):
            self.assertIs(state.durable_archive(None), self.agent.observation_archive)


if __name__ == "__main__":
    unittest.main()

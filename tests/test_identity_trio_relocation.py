"""The identity trio, after its move out of ``result_handles.paging``.

``durable_archive`` sits inside a bare ``except Exception`` (``observation_offloading/
state.py``), so a botched move does not raise. These tests assert the RESULT is
non-empty rather than that the import resolves, which is the only way that failure
mode is visible.
"""
import os
import unittest

from fastworkflow import tracing
from fastworkflow.observation_offloading import state
from fastworkflow.observation_offloading.state import (
    current_execute_alias,
    current_scope,
    default_scope,
)


class FakeAgent:
    """An agent with one execute step in flight, as ReAct leaves it."""

    def __init__(self):
        self.trajectory = {
            "tool_name_0": "execute_workflow_query",
            "observation_0": "done",
            "tool_name_1": "execute_workflow_query",
        }


class FakeHost:
    def __init__(self, agent):
        self.workflow_tool_agent = agent


class TrioLivesOnState(unittest.TestCase):
    def test_the_trio_is_importable_from_state(self):
        self.assertEqual(current_scope.__module__,
                         "fastworkflow.observation_offloading.state")
        self.assertEqual(current_execute_alias.__module__,
                         "fastworkflow.observation_offloading.state")
        self.assertEqual(state._current_agent.__module__,
                         "fastworkflow.observation_offloading.state")

    def test_alias_reads_the_in_flight_step_index(self):
        self.assertEqual(current_execute_alias(FakeAgent()), "O1")

    def test_scope_falls_back_to_the_process_default(self):
        self.assertEqual(current_scope(), default_scope())


class DurableArchiveResolvesTheSession(unittest.TestCase):
    def setUp(self):
        self.host = FakeHost(FakeAgent())
        state.reset_observation_state()
        self.addCleanup(state.reset_observation_state)

    def test_the_archive_lookup_resolves_the_session_database(self):
        with tracing.host_scope(self.host):
            archive = state.durable_archive(None)
        self.assertEqual(archive.db_path,
                         os.path.abspath(state.observability_db_path(self.host)))


if __name__ == "__main__":
    unittest.main()

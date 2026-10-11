"""The in-memory offload event log is a bounded ring.

Events are kept in process for a live turn to read back; the durable copy is the
``offload_events`` table, covered by the event tests.
"""
from __future__ import annotations

import tempfile
import unittest

from fastworkflow.observation_offloading import state as offload_state
from fastworkflow.observation_offloading.archive import RuntimeHandleScope
from fastworkflow.observation_offloading.state import (
    record_event,
    reset_observation_state,
    snapshot_events,
)


class EventBufferBoundTests(unittest.TestCase):
    """The in-memory event log is a ring, not a ledger, with a fixed bound."""

    def setUp(self) -> None:
        reset_observation_state()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.scope = RuntimeHandleScope(
            channel_id="ring", turn_key="ring-turn")

    def tearDown(self) -> None:
        reset_observation_state()

    def fill(self, count: int) -> None:
        for index in range(count):
            record_event({"kind": "fixture",
                          "n": index})

    def test_the_buffer_stops_at_the_cap_and_keeps_the_newest(self) -> None:
        cap = offload_state.EVENT_BUFFER_MAX
        self.fill(cap + 25)
        events = snapshot_events()
        self.assertEqual(len(events), cap)
        self.assertEqual([item["n"] for item in events], list(range(25, cap + 25)))

    def test_the_cap_is_a_module_constant(self) -> None:
        self.assertEqual(offload_state.EVENT_BUFFER_MAX, 2000)
        for retired in ("event_buffer_max_from_env", "EVENT_BUFFER_MAX_ENV",
                        "DEFAULT_EVENT_BUFFER_MAX"):
            with self.subTest(name=retired):
                self.assertFalse(hasattr(offload_state, retired))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

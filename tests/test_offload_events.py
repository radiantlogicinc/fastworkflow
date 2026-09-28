"""Offload diagnostic events are stored in the turn's observability database.

The offloading runtime records an event for what it archives, offloads,
searches and rehydrates. The in-process ring (``snapshot_events``) keeps the
newest of them; the durable copy is a row of ``offload_events`` in the same
database as the turn's evidence, keyed the same way and read back through
``ObservabilityStore.offload_events``. Events carry search questions, model
reasoning and answers, so their text is written through the same protection as
evidence: redacted when ``FW_OFFLOAD_EVIDENCE_REDACTION`` is on, raw when it
is off. The plaintext events file and its settings are gone.

Everything runs against databases and workflows created in this test's own
temporary directory. No model, no backend, no network.
"""
from __future__ import annotations

import os
import tempfile
import unittest
import uuid
from pathlib import Path

import dspy

import fastworkflow
from fastworkflow import state_paths, tracing
from fastworkflow.observability import store as obs
from fastworkflow.observation_offloading import archive as archive_module
from fastworkflow.observation_offloading import state as offload_state
from fastworkflow.observation_offloading.agent import build_tool_agent
from fastworkflow.observation_offloading.archive import (
    RuntimeHandleArchive,
    RuntimeHandleScope,
)
from fastworkflow.observation_offloading.state import (
    record_event,
    register_scope,
    reset_runtime_state,
    snapshot_events,
)
from fastworkflow.workflow_execution_context import WorkflowExecutionContext

REDACTION_ENV = archive_module.REDACTION_ENV

#: A credential shape the store's ``Redactor`` recognises with no help from
#: the environment.
SK_TOKEN = "sk-livekey1234567890abcdef"
REDACTED = "[REDACTED]"


def chatbot_scope(channel: str = "chat", turn: str = "turn-1") -> RuntimeHandleScope:
    return RuntimeHandleScope(
        store_identity="store", channel_id=channel, experiment_id="unbound",
        task_id="unbound", attempt=0, turn_key=turn,
    )


def file_bytes(db_path: str) -> bytes:
    blob = b""
    for suffix in ("", "-wal"):
        if os.path.exists(db_path + suffix):
            with open(db_path + suffix, "rb") as handle:
                blob += handle.read()
    return blob


class EventFixture(unittest.TestCase):
    """An observability database in a temporary directory, clean configuration."""

    def setUp(self) -> None:
        reset_runtime_state()
        self.addCleanup(reset_runtime_state)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self._restore_env: dict[str, str | None] = {}
        for name in (REDACTION_ENV, "FW_OBS_CAPTURE_PROFILE"):
            self._restore_env[name] = os.environ.pop(name, None)
        self.addCleanup(self._restore_environment)
        self.db_path = os.path.join(self.temp.name, "observability.sqlite3")
        self.archive = RuntimeHandleArchive(self.db_path)
        self.scope = chatbot_scope()
        register_scope(self.scope, self.archive)

    def _restore_environment(self) -> None:
        for name, value in self._restore_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def stored(self, **filters) -> list[dict]:
        return obs.ObservabilityStore(self.db_path).offload_events(**filters)

    def search_event(self, answer: str) -> dict:
        return {"kind": "search_memory", "scope_id": self.scope.scope_id,
                "alias": "O1", "question": "what is the key?",
                "reasoning": "the connector listing names it", "answer": answer}


class PersistenceTests(EventFixture):
    """Every recorded event lands in the turn's database, keyed by its turn."""

    def test_a_recorded_event_is_stored_with_its_turn(self) -> None:
        record_event(self.search_event("the key is on row 3"))

        stored = self.stored()
        self.assertEqual(len(stored), 1)
        row = stored[0]
        self.assertEqual((row["turn_key"], row["channel_id"], row["scope_id"]),
                         ("turn-1", "chat", self.scope.scope_id))
        self.assertEqual(row["kind"], "search_memory")
        self.assertEqual(row["event"]["answer"], "the key is on row 3")
        self.assertEqual(row["redaction"], archive_module.REDACTION_ON)
        self.assertFalse(row["redacted"])
        # And the ring still holds it for the live turn.
        self.assertEqual(snapshot_events()[-1]["answer"], "the key is on row 3")

    def test_the_reader_filters_by_turn_channel_kind_and_limit(self) -> None:
        other = chatbot_scope("other", "turn-2")
        register_scope(other, self.archive)
        for index in range(3):
            record_event({"kind": "observation_archived",
                          "scope_id": self.scope.scope_id, "n": index})
        record_event({"kind": "hot_evict", "scope_id": other.scope_id})

        self.assertEqual(len(self.stored(turn_key="turn-1")), 3)
        self.assertEqual([row["kind"] for row in self.stored(channel_id="other")],
                         ["hot_evict"])
        self.assertEqual(len(self.stored(kind="observation_archived")), 3)
        self.assertEqual(
            [row["event"]["n"] for row in self.stored(turn_key="turn-1", limit=2)],
            [0, 1],
        )

    def test_an_event_for_a_scope_this_process_never_saw_stays_in_memory(self) -> None:
        """Without its scope an event cannot be placed in a turn, so it is not stored."""
        record_event({"kind": "search_memory", "scope_id": "unknown-scope"})
        self.assertEqual(self.stored(), [])
        self.assertEqual(snapshot_events()[-1]["scope_id"], "unknown-scope")

    def test_a_released_scope_stops_being_routed(self) -> None:
        offload_state.reclaim_scope(self.scope)
        record_event({"kind": "late", "scope_id": self.scope.scope_id})
        self.assertEqual(self.stored(), [])

    def test_an_unwritable_database_never_fails_the_event(self) -> None:
        broken = RuntimeHandleArchive(os.path.join(self.temp.name, "other.sqlite3"))
        broken.db_path = self.temp.name  # a directory: every connect fails
        scope = chatbot_scope("broken", "turn-9")
        register_scope(scope, broken)
        with self.assertLogs(offload_state.logger, level="WARNING") as logs:
            record_event({"kind": "probe", "scope_id": scope.scope_id})
            record_event({"kind": "probe-again", "scope_id": scope.scope_id})
        # Logged once for the database, not once per event.
        self.assertEqual(len(logs.output), 1)
        self.assertEqual([item["kind"] for item in snapshot_events()],
                         ["probe", "probe-again"])

    def test_the_events_table_is_an_additive_feature(self) -> None:
        store = obs.ObservabilityStore(self.db_path)
        self.assertTrue(store.has_feature(obs.FEATURE_OFFLOAD_EVENTS_V1))


class RedactionTests(EventFixture):
    """Event text is protected at the write exactly like evidence text."""

    def test_event_text_is_redacted_on_write_when_the_toggle_is_on(self) -> None:
        record_event(self.search_event(f"the key is {SK_TOKEN}"))

        row = self.stored()[0]
        self.assertEqual(row["redaction"], archive_module.REDACTION_ON)
        self.assertTrue(row["redacted"])
        self.assertIn(REDACTED, row["event"]["answer"])
        self.assertNotIn(SK_TOKEN, row["event_text"])
        self.assertNotIn(SK_TOKEN.encode("ascii"), file_bytes(self.db_path))
        # Everything that was not a secret is still readable.
        self.assertEqual(row["event"]["question"], "what is the key?")

    def test_event_text_is_stored_raw_when_the_toggle_is_off(self) -> None:
        os.environ[REDACTION_ENV] = archive_module.REDACTION_OFF
        record_event(self.search_event(f"the key is {SK_TOKEN}"))

        row = self.stored()[0]
        self.assertEqual(row["redaction"], archive_module.REDACTION_OFF)
        self.assertFalse(row["redacted"])
        self.assertEqual(row["event"]["answer"], f"the key is {SK_TOKEN}")
        self.assertIn(SK_TOKEN.encode("ascii"), file_bytes(self.db_path))


class RetiredSettingsTests(EventFixture):
    """The events file and its settings are gone."""

    def test_the_file_api_is_gone(self) -> None:
        for name in ("EVENTS_ENV", "forget_events_file", "EVENT_BUFFER_MAX_ENV",
                     "event_buffer_max_from_env"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(offload_state, name))


class Signature(dspy.Signature):
    user_query: str = dspy.InputField()
    final_answer: str = dspy.OutputField()


class LiveTurnEventTests(unittest.TestCase):
    """A real turn stores its events in the workflow's own observability database."""

    workflow_path = str(Path(__file__).parent.joinpath("todo_list_workflow").resolve())

    def setUp(self) -> None:
        reset_runtime_state()
        self.temp = tempfile.TemporaryDirectory()
        self._restore = {name: os.environ.pop(name, None)
                         for name in (REDACTION_ENV,)}
        self.state_root = os.path.join(self.temp.name, "state")
        os.environ["FASTWORKFLOW_STATE_ROOT"] = self.state_root
        fastworkflow.init({"FASTWORKFLOW_STATE_ROOT": self.state_root})
        self.sessions: list[tuple] = []

    def tearDown(self) -> None:
        for ctx, workflow in self.sessions:
            try:
                ctx.close()
                workflow.close()
            except Exception:  # noqa: BLE001
                pass
        reset_runtime_state()
        os.environ.pop("FASTWORKFLOW_STATE_ROOT", None)
        for name, value in self._restore.items():
            if value is not None:
                os.environ[name] = value
        self.temp.cleanup()

    def test_a_finished_turn_leaves_its_events_in_the_workflow_database(self) -> None:
        rows = ["row-%03d  fixture row %s" % (i, "x" * 48) for i in range(40)]
        rows.append(f"api_key: {SK_TOKEN}")

        def execute_workflow_query(command: str) -> str:
            """A locally generated listing."""
            return "\n".join(rows)

        workflow = fastworkflow.Workflow.create(
            self.workflow_path, workflow_id_str="events-%s" % uuid.uuid4().hex
        )
        ctx = WorkflowExecutionContext(run_as_agent=False, session_key="events")
        ctx.bind_app_workflow(workflow)
        ctx.bind_observability_identity(channel_id="events-channel")
        ctx._turn_key = "events-turn"
        ctx.push_active_workflow(workflow)
        agent = build_tool_agent(ctx, Signature, [execute_workflow_query], max_iters=8)
        ctx._workflow_tool_agent = agent
        self.sessions.append((ctx, workflow))
        agent.extract = lambda **kwargs: dspy.Prediction(final_answer="done")
        queue = iter([("execute_workflow_query", {"command": "show_rows"}),
                      ("finish", {})])

        def decide(**kwargs):
            name, arguments = next(queue)
            return dspy.Prediction(next_thought="scripted", next_tool_name=name,
                                   next_tool_args=arguments)

        agent.react = decide
        with tracing.host_scope(ctx):
            agent.forward(user_query="fixture")

        db_path = state_paths.observability_db(self.workflow_path)
        self.assertEqual(agent.observation_archive.db_path, os.path.abspath(db_path))
        stored = obs.ObservabilityStore(db_path).offload_events(turn_key="events-turn")
        kinds = {row["kind"] for row in stored}
        self.assertIn("observation_archived", kinds)
        self.assertTrue(all(row["channel_id"] == "events-channel" for row in stored))
        self.assertNotIn(SK_TOKEN.encode("ascii"), file_bytes(db_path))

        deleted = obs.ObservabilityStore(db_path).forget_channel("events-channel")
        self.assertGreaterEqual(deleted["offload_events"], len(stored))
        self.assertEqual(
            obs.ObservabilityStore(db_path).offload_events(channel_id="events-channel"),
            [],
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

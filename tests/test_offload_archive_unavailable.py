"""An archive that cannot be OPENED must not fail the turn.

Agent construction opens or creates the workflow's observability database --
where the offload evidence lives -- before the fail-open compaction hook is
installed, so every failure mode of that one file -- a read-only state root, a
permission bit, a path holding something that is not a database, a database
from a newer build -- used to raise out of ``build_tool_agent`` and
abort the turn before a single step ran. The persist-before-label recovery, which exists
precisely so that a storage failure keeps the evidence inline, never got the
chance to run.

Everything here breaks ONLY the archive file and leaves the rest of the state
root usable, which is the case at issue. No model, no backend: a
temporary directory, a permission bit and scripted ReAct decisions.
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace

import dspy

import fastworkflow
from fastworkflow import state_paths, tracing
from fastworkflow.observation_offloading.agent import (
    build_tool_agent,
    open_handle_archive,
)
from fastworkflow.observation_offloading.archive import (
    PersistenceError,
    RuntimeHandleArchive,
    RuntimeHandleScope,
    UnavailableHandleArchive,
)
from fastworkflow.observation_offloading.search import search_memory
from fastworkflow.observation_offloading.state import (
    reset_runtime_state,
    snapshot_events,
)
from fastworkflow.workflow_execution_context import WorkflowExecutionContext

NOT_A_DATABASE = b"this file is not a sqlite database\n" * 8


def break_with_garbage(path: str) -> None:
    """The archive path holds something that is not a database."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(NOT_A_DATABASE)


def plant_an_older_schema(path: str) -> None:
    """The archive path holds a populated store from an older build."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.execute("CREATE TABLE turns (turn_key TEXT PRIMARY KEY)")
        conn.execute("INSERT INTO turns VALUES ('older-build-turn')")
        conn.execute("PRAGMA user_version = 1")
        conn.commit()
    finally:
        conn.close()


def break_with_a_newer_schema(path: str) -> None:
    """The archive path holds a store from a newer build, which is refused."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.execute("CREATE TABLE turns (turn_key TEXT PRIMARY KEY)")
        conn.execute("PRAGMA user_version = 999")
        conn.commit()
    finally:
        conn.close()


def break_with_permissions(path: str) -> None:
    """The archive path exists and this process may not open it."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(b"")
    os.chmod(path, 0)


class Signature(dspy.Signature):
    user_query: str = dspy.InputField()
    final_answer: str = dspy.OutputField()


def noop_tool(command: str) -> str:
    """Return the command unchanged."""
    return command


class ArchiveInitialisationFailure(unittest.TestCase):
    """Construction degrades; it does not raise."""

    def setUp(self) -> None:
        reset_runtime_state()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.cleanup)
        self.previous_root = os.environ.get("FASTWORKFLOW_STATE_ROOT")
        os.environ["FASTWORKFLOW_STATE_ROOT"] = self.temp.name
        self.archive_path = state_paths.observability_db("")
        os.makedirs(os.path.dirname(self.archive_path), exist_ok=True)

    def cleanup(self) -> None:
        for root, _dirs, files in os.walk(self.temp.name):
            for name in files:
                try:
                    os.chmod(os.path.join(root, name), 0o600)
                except OSError:
                    pass
        if self.previous_root is None:
            os.environ.pop("FASTWORKFLOW_STATE_ROOT", None)
        else:
            os.environ["FASTWORKFLOW_STATE_ROOT"] = self.previous_root
        self.temp.cleanup()
        reset_runtime_state()

    def state_root_still_works(self) -> bool:
        """The rest of the state root is usable: only the archive is broken."""
        beside = os.path.join(os.path.dirname(self.archive_path), "probe.sqlite3")
        with open(beside, "wb") as handle:
            handle.write(b"ok")
        os.remove(beside)
        return True

    # -- the precondition ----------------------------------------------------

    def test_the_real_archive_still_raises_on_a_file_that_is_not_a_database(self) -> None:
        """The failure being degraded is real, not hypothetical."""
        break_with_garbage(self.archive_path)
        with self.assertRaises(Exception):
            RuntimeHandleArchive(self.archive_path)

    def test_a_store_from_a_newer_build_degrades_too(self) -> None:
        """The store refuses a newer DB and never touches it; the archive degrades."""
        break_with_a_newer_schema(self.archive_path)
        with open(self.archive_path, "rb") as handle:
            before = handle.read()
        store = open_handle_archive(self.archive_path)
        self.assertIsInstance(store, UnavailableHandleArchive)
        self.assertIn("IncompatibleObservabilityDB", store.reason)
        with open(self.archive_path, "rb") as handle:
            self.assertEqual(handle.read(), before)

    def test_a_store_from_an_older_build_is_replaced_and_offloading_works(self) -> None:
        """An older populated DB is recreated in place, so the archive is usable."""
        plant_an_older_schema(self.archive_path)
        store = open_handle_archive(self.archive_path)
        self.assertIsInstance(store, RuntimeHandleArchive)
        scope = RuntimeHandleScope(
            store_identity="s", channel_id="c", experiment_id="e",
            task_id="t", attempt=0, turn_key="k",
        )
        text = "holder row\n" * 50
        store.persist(scope, alias="O1", offload_order=1, command_name="c",
                      step_index=0, text=text,
                      text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest())
        self.assertEqual(store.get(scope, "O1")["text"], text)
        with sqlite3.connect(self.archive_path) as conn:
            self.assertEqual(
                conn.execute("SELECT count(*) FROM turns").fetchone()[0], 0)
        self.assertEqual(
            [e for e in snapshot_events() if e["kind"] == "archive_unavailable"], [])

    # -- the degradation -----------------------------------------------------

    def test_open_handle_archive_degrades_and_reports_once(self) -> None:
        break_with_garbage(self.archive_path)
        store = open_handle_archive(self.archive_path)
        self.assertIsInstance(store, UnavailableHandleArchive)
        self.assertFalse(store.available)
        # The path that was WANTED, so nothing is silently redirected.
        self.assertEqual(store.db_path, os.path.abspath(self.archive_path))
        reported = [e for e in snapshot_events() if e["kind"] == "archive_unavailable"]
        self.assertEqual(len(reported), 1)
        self.assertEqual(
            reported[0]["reason"], "initialization_failed_observations_inline"
        )
        self.assertEqual(reported[0]["db_path"], os.path.abspath(self.archive_path))

    def test_a_writable_archive_is_untouched_by_the_degradation(self) -> None:
        store = open_handle_archive(self.archive_path)
        self.assertIsInstance(store, RuntimeHandleArchive)
        self.assertTrue(store.available)
        self.assertEqual(
            [e for e in snapshot_events() if e["kind"] == "archive_unavailable"], []
        )

    def test_the_unavailable_archive_refuses_writes_and_reads_empty(self) -> None:
        """The write path's own degradation policy is what catches this."""
        break_with_garbage(self.archive_path)
        store = open_handle_archive(self.archive_path)
        scope = RuntimeHandleScope(
            store_identity="s", channel_id="c", experiment_id="e",
            task_id="t", attempt=0, turn_key="k",
        )
        with self.assertRaises(PersistenceError):
            store.persist(scope, alias="O1", offload_order=1, command_name="c",
                          step_index=0, text="x", text_sha256="0" * 64)
        self.assertIsNone(store.get(scope, "O1"))
        self.assertEqual(store.list(scope), [])
        # A read for an alias is a miss, exactly as for an alias never archived.
        answer = search_memory("Who?", "O1", scope=scope, selected_archive=store)
        self.assertIn("O1", answer)

    def test_the_agent_is_built_when_only_the_archive_is_broken(self) -> None:
        break_with_garbage(self.archive_path)
        self.assertTrue(self.state_root_still_works())
        agent = build_tool_agent(
            SimpleNamespace(), Signature, [noop_tool], max_iters=3
        )
        self.assertIsInstance(agent.observation_archive, UnavailableHandleArchive)
        self.assertEqual(set(agent.tools), {"noop_tool", "search_memory", "finish"})

    @unittest.skipIf(os.geteuid() == 0, "root ignores the permission bit")
    def test_the_agent_is_built_when_the_archive_cannot_be_opened(self) -> None:
        break_with_permissions(self.archive_path)
        self.assertTrue(self.state_root_still_works())
        agent = build_tool_agent(
            SimpleNamespace(), Signature, [noop_tool], max_iters=3
        )
        self.assertIsInstance(agent.observation_archive, UnavailableHandleArchive)

    def test_a_step_keeps_its_observation_inline_instead_of_aborting(self) -> None:
        """The compacting hook runs, refuses the write and changes nothing."""
        break_with_garbage(self.archive_path)
        agent = build_tool_agent(
            SimpleNamespace(), Signature, [noop_tool], max_iters=3
        )
        observation = "holder row " * 2_000
        trajectory = {
            "thought_0": "look",
            "tool_name_0": "execute_workflow_query",
            "tool_args_0": {"command": "show_holders"},
            "observation_0": observation,
        }
        self.assertTrue(agent._on_step_complete(0, trajectory))
        self.assertIn(observation, str(trajectory["observation_0"]))
        refused = [e for e in snapshot_events() if e["kind"] == "archive_refused"]
        self.assertTrue(refused)
        self.assertEqual(
            refused[0]["reason"], "persistence_failed_original_retained"
        )
        # Nothing claimed durability it does not have.
        self.assertEqual(
            [e for e in snapshot_events() if e["kind"] == "observation_archived"], []
        )

    # -- downstream tolerance ------------------------------------------------

    def test_the_result_handle_store_is_not_redirected_to_another_file(self) -> None:
        """An unavailable archive must not silently move evidence elsewhere."""
        break_with_garbage(self.archive_path)
        agent = build_tool_agent(
            SimpleNamespace(), Signature, [noop_tool], max_iters=3
        )
        self.assertEqual(
            os.path.abspath(agent.observation_archive.db_path),
            os.path.abspath(self.archive_path),
        )

    def test_a_degraded_run_writes_no_second_evidence_file(
        self,
    ) -> None:
        """Degrading neither repairs the broken file nor writes evidence beside it."""
        break_with_garbage(self.archive_path)
        agent = build_tool_agent(
            SimpleNamespace(), Signature, [noop_tool], max_iters=3
        )
        self.assertIsInstance(agent.observation_archive, UnavailableHandleArchive)
        with open(self.archive_path, "rb") as handle:
            self.assertEqual(handle.read(), NOT_A_DATABASE)
        self.assertEqual(
            sorted(os.listdir(os.path.dirname(self.archive_path))),
            [os.path.basename(self.archive_path)],
        )


class InlineOnlyTurn(unittest.TestCase):
    """A real turn, with only the archive file broken, still reaches an answer."""

    workflow_path = str(Path(__file__).parent.joinpath("todo_list_workflow").resolve())

    def setUp(self) -> None:
        reset_runtime_state()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.cleanup)
        self.state_root = os.path.join(self.temp.name, "state")
        os.environ["FASTWORKFLOW_STATE_ROOT"] = self.state_root
        fastworkflow.init({"FASTWORKFLOW_STATE_ROOT": self.state_root})
        self.archive_path = state_paths.observability_db(self.workflow_path)
        break_with_garbage(self.archive_path)
        self.open_sessions: list[tuple] = []

    def cleanup(self) -> None:
        for ctx, workflow in self.open_sessions:
            try:
                ctx.close()
                workflow.close()
            except Exception:  # noqa: BLE001
                pass
        reset_runtime_state()
        os.environ.pop("FASTWORKFLOW_STATE_ROOT", None)
        self.temp.cleanup()

    def test_the_turn_succeeds_with_its_observation_left_inline(self) -> None:
        rows = ["row-%03d  fixture row %d %s" % (i, i, "x" * 48) for i in range(40)]

        def execute_workflow_query(command: str) -> str:
            """A locally generated listing, big enough to want offloading."""
            return "\n".join(rows)

        workflow = fastworkflow.Workflow.create(
            self.workflow_path, workflow_id_str="t5x-%s" % uuid.uuid4().hex
        )
        ctx = WorkflowExecutionContext(run_as_agent=False, session_key="t5x")
        ctx.bind_app_workflow(workflow)
        ctx.bind_observability_identity(channel_id="t5x")
        ctx._turn_key = "turn-1"
        ctx.push_active_workflow(workflow)
        # The whole point: the agent is constructed although its archive is not
        # a database, and the rest of the state root is perfectly usable.
        agent = build_tool_agent(ctx, Signature, [execute_workflow_query], max_iters=8)
        ctx._workflow_tool_agent = agent
        self.open_sessions.append((ctx, workflow))
        self.assertIsInstance(agent.observation_archive, UnavailableHandleArchive)

        agent.extract = lambda **kwargs: dspy.Prediction(final_answer="done")
        queue = iter([("execute_workflow_query", {"command": "show_rows"}),
                      ("finish", {})])

        def decide(**kwargs):
            name, arguments = next(queue)
            return dspy.Prediction(next_thought="scripted", next_tool_name=name,
                                   next_tool_args=arguments)

        agent.react = decide
        with tracing.host_scope(ctx):
            prediction = agent.forward(user_query="fixture")

        self.assertEqual(prediction.final_answer, "done")
        trajectory = agent.current_trajectory
        self.assertIn(rows[0], str(trajectory.get("observation_0")))
        kinds = {e["kind"] for e in snapshot_events()}
        self.assertIn("archive_unavailable", kinds)
        self.assertIn("archive_refused", kinds)
        # The file was never replaced by a database this run could write.
        with open(self.archive_path, "rb") as handle:
            self.assertEqual(handle.read(), NOT_A_DATABASE)


if __name__ == "__main__":
    unittest.main()

"""Offload evidence is bounded even when no trace sink ever opens its database.

Retention pruning used to run only when a trace sink started on the store. The
offloading archive writes evidence whatever the sink, so a context built with
``tracing.NoOpTraceSink()`` wrote evidence, subjects and events that nothing
ever pruned. ``open_handle_archive`` now prunes the database the first time it
opens it in a process -- once per PATH, because every agent build opens a new
archive object and a per-object prune would re-prune a large database on every
first turn. A failed prune is logged and the agent is still built.

Everything here runs against databases in this test's own temporary state
root. No model and no backend: real stores, a real workflow, scripted ReAct
decisions, and an SQLite trigger where a prune has to fail.
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
import tempfile
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import dspy

import fastworkflow
from fastworkflow import state_paths, tracing
from fastworkflow.observation_offloading.agent import (
    build_tool_agent,
    open_handle_archive,
)
from fastworkflow.observation_offloading.archive import (
    RuntimeHandleArchive,
    RuntimeHandleScope,
)
from fastworkflow.observation_offloading.state import reset_runtime_state
from fastworkflow.workflow_execution_context import WorkflowExecutionContext


class Signature(dspy.Signature):
    user_query: str = dspy.InputField()
    final_answer: str = dspy.OutputField()


def noop_tool(command: str) -> str:
    """Return the command unchanged."""
    return command


def scope_for(turn: str) -> RuntimeHandleScope:
    return RuntimeHandleScope(
        store_identity="store",
        channel_id="chat",
        experiment_id="unbound",
        task_id="unbound",
        attempt=0,
        turn_key=turn,
    )


def plant_aged_turn(db_path: str, turn: str, *, days: int = 400) -> None:
    """One turn of evidence written by the production writer, then backdated."""
    scope = scope_for(turn)
    text = "aged row for %s\n" % turn * 20
    archive = RuntimeHandleArchive(db_path)
    archive.put_subject(scope, "O1", "Fixture " + turn)
    archive.persist(
        scope,
        alias="O1",
        offload_order=1,
        command_name="execute_workflow_query",
        step_index=1,
        text=text,
        text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )
    stamp = (datetime.now(timezone.utc) - timedelta(days=days)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE offload_evidence SET persisted_at=? WHERE turn_key=?", (stamp, turn)
        )
        conn.execute(
            "UPDATE offload_subjects SET recorded_at=? WHERE turn_key=?", (stamp, turn)
        )
        conn.commit()


def evidence_rows(db_path: str, turn: str) -> int:
    with sqlite3.connect(db_path) as conn:
        return conn.execute(
            "SELECT count(*) FROM offload_evidence WHERE turn_key=?", (turn,)
        ).fetchone()[0]


class StateRoot(unittest.TestCase):
    """A private state root with a short retention horizon."""

    def setUp(self) -> None:
        reset_runtime_state()
        self.temp = tempfile.TemporaryDirectory()
        self._restore: dict[str, str | None] = {}
        for name, value in (
            ("FASTWORKFLOW_STATE_ROOT", os.path.join(self.temp.name, "state")),
            ("FW_OBS_RETENTION_DAYS", "30"),
        ):
            self._restore[name] = os.environ.get(name)
            os.environ[name] = value
        self.addCleanup(self.cleanup)

    def cleanup(self) -> None:
        reset_runtime_state()
        for name, value in self._restore.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        self.temp.cleanup()


class ArchiveOpenPrunesOncePerPath(StateRoot):
    def setUp(self) -> None:
        super().setUp()
        self.db_path = state_paths.observability_db("")
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

    def test_the_first_open_prunes_evidence_past_the_horizon(self) -> None:
        plant_aged_turn(self.db_path, "turn-old")
        self.assertEqual(evidence_rows(self.db_path, "turn-old"), 1)

        open_handle_archive(self.db_path)

        self.assertEqual(evidence_rows(self.db_path, "turn-old"), 0)

    def test_a_second_open_of_the_same_path_does_not_prune_again(self) -> None:
        open_handle_archive(self.db_path)
        plant_aged_turn(self.db_path, "turn-after-first-open")

        open_handle_archive(self.db_path)

        self.assertEqual(evidence_rows(self.db_path, "turn-after-first-open"), 1)

    def test_a_second_spelling_of_the_same_path_is_the_same_path(self) -> None:
        open_handle_archive(self.db_path)
        plant_aged_turn(self.db_path, "turn-respelled")
        respelled = os.path.join(
            os.path.dirname(self.db_path), ".", os.path.basename(self.db_path)
        )

        open_handle_archive(respelled)

        self.assertEqual(evidence_rows(self.db_path, "turn-respelled"), 1)

    def test_a_runtime_reset_forgets_which_paths_were_pruned(self) -> None:
        open_handle_archive(self.db_path)
        plant_aged_turn(self.db_path, "turn-before-reset")
        reset_runtime_state()

        open_handle_archive(self.db_path)

        self.assertEqual(evidence_rows(self.db_path, "turn-before-reset"), 0)

    def test_a_failed_prune_still_returns_a_usable_archive(self) -> None:
        plant_aged_turn(self.db_path, "turn-guarded")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "CREATE TRIGGER refuse_prune BEFORE DELETE ON offload_evidence "
                "BEGIN SELECT RAISE(ABORT, 'prune refused by fixture'); END"
            )
            conn.commit()

        with self.assertLogs(
            "fastworkflow.observation_offloading.state", level="WARNING"
        ) as logged:
            opened = open_handle_archive(self.db_path)

        self.assertIsInstance(opened, RuntimeHandleArchive)
        self.assertTrue(any("could not prune" in line for line in logged.output))
        self.assertEqual(evidence_rows(self.db_path, "turn-guarded"), 1)
        self.assertIsNotNone(opened.get(scope_for("turn-guarded"), "O1"))


class NoOpSinkAgentTurn(StateRoot):
    """A NoOpTraceSink context still writes offload evidence, and it is bounded."""

    workflow_path = str(Path(__file__).parent.joinpath("todo_list_workflow").resolve())

    def setUp(self) -> None:
        super().setUp()
        fastworkflow.init({"FASTWORKFLOW_STATE_ROOT": os.environ["FASTWORKFLOW_STATE_ROOT"]})
        self.db_path = state_paths.observability_db(self.workflow_path)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.open_sessions: list[tuple] = []
        self.addCleanup(self.close_sessions)

    def close_sessions(self) -> None:
        for ctx, workflow in self.open_sessions:
            try:
                ctx.close()
                workflow.close()
            except Exception:  # noqa: BLE001
                pass

    def context(self) -> WorkflowExecutionContext:
        workflow = fastworkflow.Workflow.create(
            self.workflow_path, workflow_id_str="noop-%s" % uuid.uuid4().hex
        )
        ctx = WorkflowExecutionContext(
            run_as_agent=False, session_key="noop", trace_sink=tracing.NoOpTraceSink()
        )
        ctx.bind_app_workflow(workflow)
        ctx.bind_observability_identity(channel_id="noop")
        ctx._turn_key = "turn-noop"
        ctx.push_active_workflow(workflow)
        self.open_sessions.append((ctx, workflow))
        return ctx

    def test_evidence_is_written_and_old_evidence_is_pruned_without_a_sink(self) -> None:
        plant_aged_turn(self.db_path, "turn-old")
        rows = ["row-%03d  fixture row %d %s" % (i, i, "x" * 48) for i in range(40)]

        def execute_workflow_query(command: str) -> str:
            """A locally generated listing."""
            return "\n".join(rows)

        ctx = self.context()
        agent = build_tool_agent(ctx, Signature, [execute_workflow_query], max_iters=8)
        ctx._workflow_tool_agent = agent

        self.assertIsInstance(ctx.trace_sink, tracing.NoOpTraceSink)
        self.assertEqual(evidence_rows(self.db_path, "turn-old"), 0)

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
        with sqlite3.connect(self.db_path) as conn:
            stored = conn.execute(
                "SELECT count(*) FROM offload_evidence WHERE channel_id=?", ("noop",)
            ).fetchone()[0]
        self.assertGreaterEqual(stored, 1)

    def test_a_second_agent_build_does_not_prune_again(self) -> None:
        ctx = self.context()
        build_tool_agent(ctx, Signature, [noop_tool], max_iters=3)
        plant_aged_turn(self.db_path, "turn-between-builds")

        build_tool_agent(ctx, Signature, [noop_tool], max_iters=3)

        self.assertEqual(evidence_rows(self.db_path, "turn-between-builds"), 1)


if __name__ == "__main__":
    unittest.main()

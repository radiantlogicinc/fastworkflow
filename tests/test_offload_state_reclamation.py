"""The offloading runtime's process-global state is bounded and reclaimed.

The registries here are all keyed by turn scope. If nothing released them, a
long-lived server would hold every search question, every piece of reasoning,
every full answer and every cached row of every turn it had ever run, for as
long as the process lived -- outside whatever retention its sessions were
configured for.

Nothing in this module calls a reset helper before it measures. That is the
point: the reclamation under test has to be reached by the production lifecycle
(the next turn binding its scope, and the execution context being closed or
evicted), because a bound that only a test can trigger is the defect, not the
fix. ``tearDown`` resets afterwards, so a leak here cannot be hidden by the
helper and cannot leak into the next test.

Everything is local: scripted ReAct decisions, a fixture tool, temporary SQLite.
No model, no backend.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
import uuid
from pathlib import Path
import dspy

import fastworkflow
from fastworkflow import tracing
from fastworkflow.observation_offloading import archive as archive_module
from fastworkflow.observation_offloading import state as offload_state
from fastworkflow.observation_offloading.agent import build_tool_agent
from fastworkflow.observation_offloading.archive import RuntimeHandleScope
from fastworkflow.observation_offloading.labels import printed_alias
from fastworkflow.observation_offloading.state import (
    context_clause_of,
    reclaim_scope,
    record_context_clause,
    record_event,
    reset_runtime_state,
    snapshot_events,
)
from fastworkflow.observation_offloading.state import (
    current_execute_alias,
    current_scope,
)
from fastworkflow.utils.react import AskUserSuspend
from fastworkflow.workflow_execution_context import WorkflowExecutionContext


#: A credential shape the store's ``Redactor`` recognises, so every listing
#: below is redacted when it is archived and its raw text is held in memory for
#: the live turn -- one more per-scope registry that must be released.
SK_TOKEN = "sk-livekey1234567890abcdef"


def rows_for(command: str, count: int = 30) -> list[str]:
    return ["%s-%03d  fixture row %d %s" % (command, index, index, "x" * 32)
            for index in range(count)] + ["api_key: " + SK_TOKEN]


class OffloadStateFixture(unittest.TestCase):
    """A real session: a workflow, an execution context and a built tool agent."""

    workflow_path = str(Path(__file__).parent.joinpath("todo_list_workflow").resolve())

    class Signature(dspy.Signature):
        user_query: str = dspy.InputField()
        final_answer: str = dspy.OutputField()

    def setUp(self) -> None:
        reset_runtime_state()
        self.temp = tempfile.TemporaryDirectory()
        os.environ["FASTWORKFLOW_STATE_ROOT"] = os.path.join(self.temp.name, "state")
        fastworkflow.init({"FASTWORKFLOW_STATE_ROOT":
                           os.path.join(self.temp.name, "state")})
        self.dispatches: list[dict] = []
        self.open_sessions: list[tuple] = []

    def tearDown(self) -> None:
        for ctx, workflow in self.open_sessions:
            try:
                ctx.close()
                workflow.close()
            except Exception:  # noqa: BLE001
                pass
        reset_runtime_state()
        os.environ.pop("FASTWORKFLOW_STATE_ROOT", None)
        self.temp.cleanup()

    # -- session construction ------------------------------------------------

    def make_session(self, *, channel: str, turn: str):
        """A session shaped exactly like a served one, with scripted decisions."""
        workflow = fastworkflow.Workflow.create(
            self.workflow_path, workflow_id_str=f"{channel}-{turn}-{uuid.uuid4().hex}"
        )
        ctx = WorkflowExecutionContext(run_as_agent=False, session_key=channel)
        ctx.bind_app_workflow(workflow)
        ctx.bind_observability_identity(channel_id=channel)
        ctx._turn_key = turn
        ctx.push_active_workflow(workflow)

        def execute_workflow_query(command: str) -> str:
            """Declare a locally generated listing under this step's own alias."""
            alias = current_execute_alias()
            scope = current_scope()
            record_context_clause(scope, alias, "Fixture " + command)
            self.dispatches.append({"command": command, "dispatch_alias": alias})
            rows = rows_for(command)
            return "\n".join(rows)

        def ask_user(question: str) -> str:
            """Suspend the turn on a local fixture question."""
            raise AskUserSuspend(question)

        agent = build_tool_agent(ctx, self.Signature,
                                 [execute_workflow_query, ask_user], max_iters=8)
        ctx._workflow_tool_agent = agent
        agent.extract = lambda **kwargs: dspy.Prediction(final_answer="done")
        self.open_sessions.append((ctx, workflow))
        return ctx, workflow, agent

    @staticmethod
    def script(agent, steps) -> None:
        queue = iter(steps)

        def decide(**kwargs):
            name, arguments = next(queue)
            return dspy.Prediction(next_thought="scripted", next_tool_name=name,
                                   next_tool_args=arguments)

        agent.react = decide

    def close_session(self, ctx, workflow) -> None:
        """Exactly what the fleet's session cache does when it retires a channel."""
        ctx.pop_active_workflow()
        self.assertTrue(ctx.close())
        workflow.close()
        if (ctx, workflow) in self.open_sessions:
            self.open_sessions.remove((ctx, workflow))

    # -- measurement ---------------------------------------------------------

    def counts(self) -> dict[str, int]:
        return {
            "events": len(snapshot_events()),
            "archived": len(offload_state._archived),
            "context_clauses": len(offload_state._context_clauses),
            "hot_observations": len(offload_state._handles),
            "search_answers": len(offload_state._search_answers),
            "raw_copies": len(archive_module._live_raw),
        }

    def scope_counts(self, scope: RuntimeHandleScope) -> dict[str, int]:
        """What one scope holds, across every registry keyed by it."""
        prefix = "%s:" % scope.scope_id
        return {
            "archived": sum(1 for key in offload_state._archived
                            if key.startswith(prefix)),
            "context_clauses": sum(1 for key in offload_state._context_clauses
                                   if key.startswith(prefix)),
            "hot_observations": sum(1 for key in offload_state._handles
                                    if key.startswith(prefix)),
            "events": sum(1 for item in snapshot_events()
                          if str(item.get("scope_id") or "") == scope.scope_id),
            "raw_copies": len(archive_module._live_raw.get(scope.scope_id, {})),
        }

    def run_finished_turn(self, ctx, agent, command: str):
        """One turn that reaches finish."""
        self.script(agent, [("execute_workflow_query", {"command": command}),
                            ("finish", {})])
        with tracing.host_scope(ctx):
            prediction = agent.forward(user_query="fixture")
        return prediction


class CompletedSessionReclamationTests(OffloadStateFixture):
    """Finished sessions must not leave the process holding their turns."""

    SESSIONS = 12

    def test_many_finished_and_closed_sessions_leave_nothing_behind(self) -> None:
        """The acceptance case: complete and evict many sessions, reset nothing."""
        growth = []
        for index in range(1, self.SESSIONS + 1):
            ctx, workflow, agent = self.make_session(
                channel="session-%d" % index, turn="turn-%d" % index)
            self.run_finished_turn(ctx, agent, "record-%d" % index)
            scope = agent.continuation_scope
            # The turn really did fill the registries before it was closed.
            self.assertGreater(self.scope_counts(scope)["archived"], 0)
            self.assertGreater(self.scope_counts(scope)["events"], 0)
            self.assertGreater(self.scope_counts(scope)["raw_copies"], 0)
            self.close_session(ctx, workflow)
            growth.append(self.counts())

        first, last = growth[0], growth[-1]
        for name, value in last.items():
            with self.subTest(registry=name):
                # One store per archive FILE is the design (ido-pg2); every
                # session here shares one workflow, so one store is the floor.
                ceiling = 1 if name == "stores" else 0
                self.assertLessEqual(
                    value, ceiling,
                    "%s grew to %d over %d finished sessions (was %d after one)"
                    % (name, value, self.SESSIONS, first[name]),
                )

    def test_a_finished_turn_is_released_when_the_next_turn_binds_its_scope(self) -> None:
        """A session that keeps running does not accumulate its own past turns."""
        ctx, workflow, agent = self.make_session(channel="chan", turn="turn-1")
        self.run_finished_turn(ctx, agent, "first")
        first_scope = agent.continuation_scope
        self.assertGreater(self.scope_counts(first_scope)["archived"], 0)

        ctx._turn_key = "turn-2"
        self.run_finished_turn(ctx, agent, "second")
        second_scope = agent.continuation_scope
        self.assertNotEqual(first_scope, second_scope)
        self.assertEqual(
            self.scope_counts(first_scope),
            dict.fromkeys(self.scope_counts(first_scope), 0),
        )
        # And the turn that is actually running kept everything.
        self.assertGreater(self.scope_counts(second_scope)["archived"], 0)
        self.assertEqual(context_clause_of(second_scope, "O1"), "Fixture second")
        self.close_session(ctx, workflow)

    def test_closing_one_session_leaves_another_live_session_usable(self) -> None:
        """Per-scope, not global: a close must not invalidate the turns beside it."""
        live_ctx, live_workflow, live_agent = self.make_session(
            channel="live", turn="turn-live")
        self.run_finished_turn(live_ctx, live_agent, "live")
        live_scope = live_agent.continuation_scope
        live_before = self.scope_counts(live_scope)

        done_ctx, done_workflow, done_agent = self.make_session(
            channel="done", turn="turn-done")
        self.run_finished_turn(done_ctx, done_agent, "done")
        done_scope = done_agent.continuation_scope
        self.close_session(done_ctx, done_workflow)

        self.assertEqual(self.scope_counts(done_scope),
                         dict.fromkeys(self.scope_counts(done_scope), 0))
        self.assertEqual(self.scope_counts(live_scope), live_before)
        # Still usable, not merely still counted: the clause still reads back.
        self.assertEqual(context_clause_of(live_scope, "O1"), "Fixture live")
        self.close_session(live_ctx, live_workflow)


    def test_a_suspended_turn_survives_its_sessions_close_and_still_resumes(self) -> None:
        """A suspension is state to keep, not state to reclaim."""
        ctx, workflow, agent = self.make_session(channel="susp", turn="turn-susp")
        self.script(agent, [("execute_workflow_query", {"command": "first"}),
                            ("execute_workflow_query", {"command": "second"}),
                            ("ask_user", {"question": "continue?"})])
        with tracing.host_scope(ctx):
            prediction = agent.forward(user_query="fixture")
        self.assertTrue(prediction.suspended)
        scope = agent.continuation_scope
        before = self.scope_counts(scope)
        self.assertEqual([entry["dispatch_alias"] for entry in self.dispatches],
                         ["O1", "O2"])
        # Exactly what a session state file carries across an eviction.
        blob = json.loads(json.dumps(agent.export_suspended()))

        ctx._awaiting_user = True
        ctx.pop_active_workflow()
        self.assertTrue(ctx.close())
        self.assertEqual(self.scope_counts(scope), before)

        ctx.push_active_workflow(workflow)
        agent.import_suspended(blob)
        self.script(agent, [("execute_workflow_query", {"command": "third"}),
                            ("finish", {})])
        with tracing.host_scope(ctx):
            resumed = agent.resume("continue")
        third = self.dispatches[-1]
        self.assertEqual(third["dispatch_alias"], "O3")
        self.assertEqual(printed_alias(resumed.trajectory["observation_3"]), "O3")
        self.assertEqual(context_clause_of(scope, "O1"), "Fixture first")
        self.assertEqual(context_clause_of(scope, "O3"), "Fixture third")
        self.assertEqual(context_clause_of(scope, "O2"), "Fixture second")

        ctx._awaiting_user = False
        self.close_session(ctx, workflow)
        self.assertEqual(self.scope_counts(scope),
                         dict.fromkeys(self.scope_counts(scope), 0))



class EventBufferBoundTests(unittest.TestCase):
    """The in-memory event log is a ring, not a ledger, with a fixed bound."""

    def setUp(self) -> None:
        reset_runtime_state()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.scope = RuntimeHandleScope(
            store_identity="store", channel_id="ring", experiment_id="unbound",
            task_id="unbound", attempt=0, turn_key="ring-turn")
        self.route()

    def route(self) -> None:
        """Events without a scope are stored in the process-default database;
        an explicit scope routed to an archive in this test's own directory
        keeps every durable copy here."""
        offload_state.register_scope(
            self.scope, offload_state.archive_for_path(
                os.path.join(self.temp.name, "observability.sqlite3")))

    def tearDown(self) -> None:
        reset_runtime_state()

    def fill(self, count: int) -> None:
        for index in range(count):
            record_event({"kind": "fixture", "scope_id": self.scope.scope_id,
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




class ReclaimScopeIsNotAGlobalResetTests(unittest.TestCase):
    """The review is explicit that a global reset is the wrong answer."""

    def setUp(self) -> None:
        reset_runtime_state()

    def tearDown(self) -> None:
        reset_runtime_state()

    def scope(self, turn: str) -> RuntimeHandleScope:
        return RuntimeHandleScope(store_identity="store", channel_id="channel",
                                  experiment_id="exp", task_id="task", attempt=1,
                                  turn_key=turn)

    def test_only_the_named_scope_loses_its_clauses_entries_and_events(self) -> None:
        gone, kept = self.scope("gone"), self.scope("kept")
        for scope in (gone, kept):
            record_context_clause(scope, "O1", "Fixture " + scope.turn_key)
            offload_state.mark_archived(scope, "O1", text_sha256="d" * 64)
            offload_state.remember_handle(
                scope, {"alias": "O1", "text": "payload", "text_sha256": "d" * 64})
            offload_state.next_search_answer_sequence(scope)
            record_event({"kind": "fixture", "scope_id": scope.scope_id})

        reclaim_scope(gone)

        # RESIDENCY, never evidence. Since ido-dhw the clause has a durable
        # tier, so what reclamation releases is the process-local copy -- read
        # through the private maps, because the public readers deliberately
        # rebuild from the archive and would answer from disk, which is the
        # cold-resume path working as designed. The survival of that row is
        # asserted straight after.
        self.assertNotIn(
            offload_state.handle_key(gone, "O1"), offload_state._context_clauses)
        self.assertIsNone(offload_state.archived_digest(gone, "O1"))
        self.assertEqual(offload_state.stored_handles(gone), {})
        self.assertEqual(offload_state.next_search_answer_sequence(gone), 1)

        self.assertEqual(context_clause_of(gone, "O1"), "Fixture gone")

        self.assertEqual(context_clause_of(kept, "O1"), "Fixture kept")
        self.assertEqual(offload_state.archived_digest(kept, "O1"), "d" * 64)
        self.assertEqual(sorted(offload_state.stored_handles(kept)), ["O1"])
        self.assertEqual(
            [item["scope_id"] for item in snapshot_events()
             if item["kind"] == "fixture"],
            [kept.scope_id],
        )

    def test_reclaiming_an_unknown_scope_is_a_no_op(self) -> None:
        kept = self.scope("kept")
        record_context_clause(kept, "O1", "Fixture kept")
        reclaim_scope(self.scope("never-seen"))
        reclaim_scope("not-a-scope-id")
        self.assertEqual(context_clause_of(kept, "O1"), "Fixture kept")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

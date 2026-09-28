"""Evidence is redacted when it is written; only memory holds the raw text.

With ``FW_OFFLOAD_EVIDENCE_REDACTION`` on (the default), what reaches the
observability database is what the store's scrub-and-capture pipeline makes of
a command response, from the very first write. Redacting at the write would
change what the agent reads back mid-turn, so the process that wrote a redacted
row also keeps its RAW text in memory for as long as the turn is live, and
every read that process makes during the turn -- the hot cache,
``search_memory``, answer rehydration -- is exact. That memory is released at
the moments the runtime already decides a turn is over: the agent binding the
next turn, and the execution context being closed. A suspended turn keeps it.
A turn resumed in a DIFFERENT process has no such memory and reads the stored,
redacted text; that is the accepted cost.

There is no seal, no ledger of raw rows, and no crash sweep any more: a
process that dies mid-turn leaves nothing raw on disk, because nothing raw was
ever written there.

The claims are made in BYTES wherever a leak is the thing being denied: a row
read back through the archive's API proves what the API returns, and the file
is what a stolen disk is about. Every scan opens the database and its
write-ahead log and searches the raw bytes.

Everything runs against databases and workflows created in this test's own
temporary directory. No model, no backend, no network.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
import unittest
import uuid
from pathlib import Path

import dspy

import fastworkflow
from fastworkflow import tracing
from fastworkflow.answer_rehydration import archived_observation
from fastworkflow.observability import store as obs
from fastworkflow.observation_offloading import archive as archive_module
from fastworkflow.observation_offloading import state as offload_state
from fastworkflow.observation_offloading.agent import build_tool_agent
from fastworkflow.observation_offloading.archive import (
    PersistenceError,
    RuntimeHandleArchive,
    RuntimeHandleScope,
)
from fastworkflow.observation_offloading.compact import archive_execute_observations
from fastworkflow.observation_offloading.state import (
    record_context_clause,
    reset_runtime_state,
    stored_handles,
)
from fastworkflow.observation_offloading.state import (
    current_execute_alias,
    current_scope,
)
from fastworkflow.utils.react import AskUserSuspend
from fastworkflow.workflow_execution_context import WorkflowExecutionContext

REDACTION_ENV = archive_module.REDACTION_ENV
REDACTION_ON = archive_module.REDACTION_ON
REDACTION_OFF = archive_module.REDACTION_OFF
LEGACY_SIDECAR_SUFFIX = ".offload-handles.sqlite3"

#: A credential shape ``Redactor._SECRET_PATTERNS`` recognises with no help
#: from the environment.
SK_TOKEN = "sk-livekey1234567890abcdef"
REDACTED = "[REDACTED]"


def response_with_credential(command: str = "sync") -> str:
    """One command response of the shape a workflow really returns."""
    return (
        "connector: okta-prod\n"
        f"api_key: {SK_TOKEN}\n"
        f"subject: {command}\n"
        "rows: 3 users synchronised\n"
    )


def chatbot_scope(channel: str = "chat", turn: str = "turn-1") -> RuntimeHandleScope:
    return RuntimeHandleScope(
        store_identity="store", channel_id=channel, experiment_id="unbound",
        task_id="unbound", attempt=0, turn_key=turn,
    )


def experiment_scope(channel: str = "exp", turn: str = "turn-1") -> RuntimeHandleScope:
    return RuntimeHandleScope(
        store_identity="store", channel_id=channel, experiment_id="exp-7",
        task_id="task-3", attempt=1, turn_key=turn,
    )


def file_bytes(db_path: str) -> bytes:
    """Every byte the database occupies, write-ahead log included."""
    blob = b""
    for suffix in ("", "-wal", "-journal"):
        if os.path.exists(db_path + suffix):
            with open(db_path + suffix, "rb") as handle:
                blob += handle.read()
    return blob


class EvidenceFixture(unittest.TestCase):
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
        archive_module._warned_redaction.clear()
        self.addCleanup(archive_module._warned_redaction.clear)
        self.db_path = os.path.join(self.temp.name, "observability.sqlite3")

    def _restore_environment(self) -> None:
        for name, value in self._restore_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    # -- helpers ---------------------------------------------------------

    def file_bytes(self) -> bytes:
        return file_bytes(self.db_path)

    def persist(self, text: str, *, scope=None, alias: str = "O1", order: int = 1,
                archive=None):
        scope = scope or chatbot_scope()
        target = archive or RuntimeHandleArchive(self.db_path)
        stored = target.persist(
            scope, alias=alias, offload_order=order,
            command_name="execute_workflow_query", step_index=order,
            text=text, text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        )
        return target, stored


class WriteTimeRedactionTests(EvidenceFixture):
    """The file is redacted from the first write; the live turn still reads raw."""

    def test_a_mid_turn_row_is_already_redacted_on_disk(self) -> None:
        text = response_with_credential()
        archive, stored = self.persist(text)
        scope = chatbot_scope()

        # The live turn reads exactly what the command returned...
        self.assertEqual(stored["text"], text)
        self.assertEqual(archive.get(scope, "O1")["text"], text)
        self.assertEqual(archive.list(scope)[0]["text"], text)
        # ...while the file never held the credential, not even mid-turn.
        blob = self.file_bytes()
        self.assertNotIn(SK_TOKEN.encode("ascii"), blob)
        self.assertIn(b"rows: 3 users synchronised", blob)

    def test_the_raw_copy_is_released_with_the_turn(self) -> None:
        text = response_with_credential()
        archive, _ = self.persist(text)
        scope = chatbot_scope()

        offload_state.reclaim_scope(scope)

        stored = archive.get(scope, "O1")
        self.assertIn(REDACTED, stored["text"])
        self.assertNotIn(SK_TOKEN, stored["text"])
        # Everything that was not a secret survives, which is what keeps a
        # redacted archive worth reading.
        self.assertIn("rows: 3 users synchronised", stored["text"])
        self.assertEqual(
            stored["text_sha256"],
            hashlib.sha256(stored["text"].encode("utf-8")).hexdigest(),
        )
        # Releasing twice is harmless.
        offload_state.reclaim_scope(scope)
        self.assertEqual(archive.get(scope, "O1")["text"], stored["text"])

    def test_releasing_one_turn_keeps_the_turn_beside_it_exact(self) -> None:
        """One turn finishing must not degrade the turn running beside it."""
        mine = chatbot_scope("mine", "turn-1")
        theirs = chatbot_scope("theirs", "turn-2")
        archive, _ = self.persist(response_with_credential(), scope=mine)
        self.persist(response_with_credential(), scope=theirs, archive=archive)

        offload_state.reclaim_scope(mine)

        self.assertIn(REDACTED, archive.get(mine, "O1")["text"])
        self.assertEqual(archive.get(theirs, "O1")["text"],
                         response_with_credential())

    def test_redaction_off_keeps_no_raw_copy_and_stores_verbatim(self) -> None:
        """Off means never redact, so there is nothing for memory to shadow."""
        os.environ[REDACTION_ENV] = REDACTION_OFF
        text = response_with_credential()
        archive, stored = self.persist(text)
        scope = chatbot_scope()

        self.assertEqual(stored["text"], text)
        self.assertNotIn(scope.scope_id, archive_module._live_raw)
        self.assertIn(SK_TOKEN.encode("ascii"), self.file_bytes())
        offload_state.reclaim_scope(scope)
        self.assertEqual(archive.get(scope, "O1")["text"], text)

    def test_a_fresh_archive_with_no_raw_copy_reads_the_redacted_text(self) -> None:
        """What another process reads: the stored bytes, nothing from memory."""
        text = response_with_credential()
        self.persist(text)
        scope = chatbot_scope()
        # A different process never saw the write, so it holds no raw copy.
        archive_module.clear_live_raw()

        other = RuntimeHandleArchive(self.db_path)
        stored = other.get(scope, "O1")
        self.assertIn(REDACTED, stored["text"])
        self.assertNotIn(SK_TOKEN, stored["text"])
        self.assertEqual(other.list(scope)[0]["text"], stored["text"])


class FidelityRecordTests(EvidenceFixture):
    """The capture record lives on the evidence row and describes its bytes."""

    def test_redacted_verbatim_and_clean_rows_are_distinguishable(self) -> None:
        scope = chatbot_scope()
        archive = RuntimeHandleArchive(self.db_path)
        self.persist(response_with_credential(), scope=scope, alias="O1",
                     archive=archive)
        os.environ[REDACTION_ENV] = REDACTION_OFF
        self.persist(response_with_credential(), scope=scope, alias="O2", order=2,
                     archive=archive)
        os.environ.pop(REDACTION_ENV)
        self.persist("no secret at all\n", scope=scope, alias="O3", order=3,
                     archive=archive)

        redacted = archive.capture_record(scope, "O1")
        verbatim = archive.capture_record(scope, "O2")
        clean = archive.capture_record(scope, "O3")
        self.assertEqual((redacted["redaction"], redacted["redacted"]),
                         (REDACTION_ON, True))
        self.assertEqual((verbatim["redaction"], verbatim["redacted"]),
                         (REDACTION_OFF, False))
        self.assertEqual((clean["redaction"], clean["redacted"]),
                         (REDACTION_ON, False))
        self.assertEqual(verbatim["capture_profile"], "")
        self.assertEqual(redacted["capture_profile"], "debug")
        self.assertIsNone(archive.capture_record(scope, "O404"))

    def test_there_are_no_side_tables_for_seal_or_capture_state(self) -> None:
        """One evidence row per observation; the old ledgers are not created."""
        self.persist(response_with_credential())
        with sqlite3.connect(self.db_path) as conn:
            tables = {row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
            columns = {row[1] for row in conn.execute(
                "PRAGMA table_info(offload_evidence)")}
        for retired in ("observation_seal_state", "observation_capture_policy",
                        "observation_context_entries", "observation_offload_handles"):
            with self.subTest(table=retired):
                self.assertNotIn(retired, tables)
        self.assertTrue({"capture_policy_version", "capture_profile", "redaction",
                         "redacted", "raw_utf8_bytes"} <= columns)
        self.assertFalse({"seal_state", "owner_id", "sealed_at"} & columns)


class ErasureAndRetentionTests(EvidenceFixture):
    """A live turn's evidence is erased like any other, raw copy included."""

    def test_a_channel_is_erased_with_or_without_a_raw_copy_in_memory(self) -> None:
        for moment, release in (("live", False), ("finished", True)):
            with self.subTest(moment=moment):
                reset_runtime_state()
                db_path = os.path.join(self.temp.name, f"{moment}.sqlite3")
                scope = chatbot_scope("erase")
                archive = RuntimeHandleArchive(db_path)
                self.persist(response_with_credential(), scope=scope,
                             archive=archive)
                if release:
                    offload_state.reclaim_scope(scope)

                deleted = obs.ObservabilityStore(db_path).forget_channel("erase")

                self.assertEqual(deleted["offload_evidence"], 1)
                self.assertIsNone(archive.get(scope, "O1"))
                self.assertNotIn(scope.scope_id, archive_module._live_raw)
                blob = file_bytes(db_path)
                self.assertNotIn(b"okta-prod", blob)
                self.assertNotIn(SK_TOKEN.encode("ascii"), blob)

    def test_retention_prunes_a_turn_with_or_without_a_raw_copy(self) -> None:
        for moment, release in (("live", False), ("finished", True)):
            with self.subTest(moment=moment):
                reset_runtime_state()
                db_path = os.path.join(self.temp.name, f"prune-{moment}.sqlite3")
                scope = chatbot_scope("old")
                archive = RuntimeHandleArchive(db_path)
                self.persist(response_with_credential(), scope=scope,
                             archive=archive)
                if release:
                    offload_state.reclaim_scope(scope)
                with sqlite3.connect(db_path) as conn:
                    conn.execute("UPDATE offload_evidence SET persisted_at=?",
                                 ("2000-01-01T00:00:00Z",))
                    conn.commit()

                deleted = obs.ObservabilityStore(db_path).prune(
                    retention_days=30, max_bytes=1_000_000_000)

                self.assertEqual(deleted["offload_evidence"], 1)
                self.assertIsNone(archive.get(scope, "O1"))
                self.assertNotIn(scope.scope_id, archive_module._live_raw)

    def test_an_experiment_turn_is_erased_too(self) -> None:
        erased = experiment_scope("exp-channel")
        also = chatbot_scope("chat-channel", turn="turn-2")
        archive, _ = self.persist(response_with_credential(), scope=erased)
        self.persist(response_with_credential(), scope=also, archive=archive)

        deleted = obs.ObservabilityStore(self.db_path).clear_conversations()

        self.assertEqual(deleted["offload_evidence"], 2)
        self.assertIsNone(archive.get(erased, "O1"))
        self.assertIsNone(archive.get(also, "O1"))


class DigestTests(EvidenceFixture):
    """The caller's digest, the stored digest, and idempotence after release."""

    def test_the_archiver_is_idempotent_after_the_turn_is_released(self) -> None:
        """A re-persist once the raw copy is gone is a readback, not a collision.

        The in-process memo that makes the archiver skip a step it already
        wrote is dropped with the turn, so a process that revisits the scope
        reaches ``persist`` with the RAW digest of text whose row holds the
        redacted bytes. The archive recognises it by redacting the candidate
        the same way and comparing stored digests; no raw digest is persisted.
        """
        scope = chatbot_scope()
        archive = RuntimeHandleArchive(self.db_path)
        text = response_with_credential()
        trajectory = {
            "thought_0": "look",
            "tool_name_0": "execute_workflow_query",
            "tool_args_0": {"command": "show_connector"},
            "observation_0": text,
        }
        archive_execute_observations(trajectory, scope=scope,
                                     selected_archive=archive)
        offload_state.reclaim_scope(scope)
        stored = archive.get(scope, "O1")
        archive_module.clear_live_raw()

        again = archive_execute_observations(trajectory, scope=scope,
                                             selected_archive=archive)

        self.assertEqual([row["alias"] for row in again], ["O1"])
        refused = [event for event in offload_state.snapshot_events()
                   if event["kind"] == "archive_refused"]
        self.assertEqual(refused, [])
        # The row was not rewritten...
        with sqlite3.connect(self.db_path) as conn:
            (digest,) = conn.execute(
                "SELECT text_sha256 FROM offload_evidence").fetchone()
        self.assertEqual(digest, stored["text_sha256"])
        # ...and this process, which holds the raw text again, reads it again.
        self.assertEqual(stored_handles(scope)["O1"]["text"], text)
        self.assertNotIn(SK_TOKEN.encode("ascii"), self.file_bytes())

    def test_the_raw_digest_is_never_written_to_the_file(self) -> None:
        """A digest of the unredacted text beside the redaction is an oracle."""
        scope = chatbot_scope()
        text = response_with_credential()
        raw_digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        archive, stored = self.persist(text, scope=scope)

        # The live turn is handed the raw digest with the raw text...
        self.assertEqual(stored["text_sha256"], raw_digest)
        # ...and the file holds neither.
        blob = self.file_bytes()
        self.assertNotIn(raw_digest.encode("ascii"), blob)
        self.assertNotIn(SK_TOKEN.encode("ascii"), blob)

    def test_a_different_text_is_still_refused_after_release(self) -> None:
        scope = chatbot_scope()
        archive, _ = self.persist(response_with_credential(), scope=scope)
        offload_state.reclaim_scope(scope)
        other = "a completely different observation with no secret\n"
        with self.assertRaises(PersistenceError):
            archive.persist(
                scope, alias="O1", offload_order=1,
                command_name="execute_workflow_query", step_index=1,
                text=other,
                text_sha256=hashlib.sha256(other.encode("utf-8")).hexdigest(),
            )

    def test_a_redacted_row_reads_back_through_its_own_digest(self) -> None:
        """``_decode_row`` verifies every read against the stored bytes."""
        scope = chatbot_scope()
        self.persist(response_with_credential(), scope=scope)
        archive_module.clear_live_raw()
        reopened = RuntimeHandleArchive(self.db_path)
        self.assertIsNotNone(reopened.get(scope, "O1"))
        self.assertEqual(len(reopened.list(scope)), 1)


class NoSealLifecycleTests(EvidenceFixture):
    """What the crash sweep existed for cannot happen any more."""

    def test_a_process_that_died_mid_turn_left_nothing_raw_on_disk(self) -> None:
        self.persist(response_with_credential())
        # The process dies: its memory is gone and nobody releases anything.
        archive_module.clear_live_raw()
        self.assertNotIn(SK_TOKEN.encode("ascii"), self.file_bytes())
        stored = RuntimeHandleArchive(self.db_path).get(chatbot_scope(), "O1")
        self.assertIn(REDACTED, stored["text"])

    def test_opening_the_database_again_rewrites_no_row(self) -> None:
        self.persist(response_with_credential())
        with sqlite3.connect(self.db_path) as conn:
            before = conn.execute(
                "SELECT text_utf8, text_sha256, persisted_at FROM offload_evidence"
            ).fetchall()

        RuntimeHandleArchive(self.db_path)

        with sqlite3.connect(self.db_path) as conn:
            after = conn.execute(
                "SELECT text_utf8, text_sha256, persisted_at FROM offload_evidence"
            ).fetchall()
        self.assertEqual(before, after)


    def test_the_seal_and_sweep_surface_is_gone(self) -> None:
        for owner, names in (
            (archive_module, ("seal_grace_seconds", "owner_id", "SEAL_GRACE_ENV",
                              "SEAL_PENDING", "UNKNOWN_CAPTURE_RECORD")),
            (archive_module.RuntimeHandleArchive,
             ("seal_scope", "sweep_unsealed", "seal_state", "pending_aliases",
              "compact", "list_subjects", "put_context_entry",
              "list_context_entries")),
            (archive_module.UnavailableHandleArchive,
             ("seal_scope", "sweep_unsealed", "seal_state", "list_subjects",
              "put_context_entry", "list_context_entries")),
            (offload_state, ("seal_scope",)),
        ):
            for name in names:
                with self.subTest(owner=getattr(owner, "__name__", owner), name=name):
                    self.assertFalse(hasattr(owner, name))

    def test_retention_deletes_but_never_rewrites_a_recent_turn(self) -> None:
        self.persist(response_with_credential())
        with sqlite3.connect(self.db_path) as conn:
            before = conn.execute("SELECT text_utf8 FROM offload_evidence").fetchall()

        deleted = obs.ObservabilityStore(self.db_path).prune(
            retention_days=30, max_bytes=1_000_000_000)

        self.assertEqual(deleted["offload_evidence"], 0)
        with sqlite3.connect(self.db_path) as conn:
            after = conn.execute("SELECT text_utf8 FROM offload_evidence").fetchall()
        self.assertEqual(before, after)

    def test_a_legacy_preserve_sentinel_is_removed_with_its_sidecar(self) -> None:
        sidecar = self.db_path + LEGACY_SIDECAR_SUFFIX
        for path in (sidecar, sidecar + ".preserve"):
            with open(path, "wb") as handle:
                handle.write(b"legacy")

        RuntimeHandleArchive(self.db_path)

        self.assertFalse(os.path.exists(sidecar))
        self.assertFalse(os.path.exists(sidecar + ".preserve"))

    def test_no_raw_row_ledger_is_created(self) -> None:
        self.persist(response_with_credential())
        with sqlite3.connect(self.db_path) as conn:
            tables = {row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertNotIn("observation_seal_state", tables)


# ---------------------------------------------------------------------------
# The production lifecycle, with a real session
# ---------------------------------------------------------------------------


class Signature(dspy.Signature):
    user_query: str = dspy.InputField()
    final_answer: str = dspy.OutputField()


class LiveTurnFixture(unittest.TestCase):
    """A real session: a workflow, an execution context and a built tool agent.

    Modelled on ``test_offload_state_reclamation``'s fixture, and for the same
    reason: the release under test has to be reached by the production
    lifecycle, because a release only a test can trigger is not the fix.
    """

    workflow_path = str(Path(__file__).parent.joinpath("todo_list_workflow").resolve())

    def setUp(self) -> None:
        reset_runtime_state()
        self.temp = tempfile.TemporaryDirectory()
        self._restore_env: dict[str, str | None] = {}
        for name in (REDACTION_ENV, "FW_OBS_CAPTURE_PROFILE"):
            self._restore_env[name] = os.environ.pop(name, None)
        os.environ["FASTWORKFLOW_STATE_ROOT"] = os.path.join(self.temp.name, "state")
        fastworkflow.init({"FASTWORKFLOW_STATE_ROOT":
                           os.path.join(self.temp.name, "state")})
        self.open_sessions: list[tuple] = []

    def tearDown(self) -> None:
        for ctx, workflow in list(self.open_sessions):
            try:
                ctx.close()
                workflow.close()
            except Exception:  # noqa: BLE001
                pass
        reset_runtime_state()
        os.environ.pop("FASTWORKFLOW_STATE_ROOT", None)
        for name, value in self._restore_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        self.temp.cleanup()

    def make_session(self, *, channel: str, turn: str):
        workflow = fastworkflow.Workflow.create(
            self.workflow_path, workflow_id_str=f"{channel}-{turn}-{uuid.uuid4().hex}"
        )
        ctx = WorkflowExecutionContext(run_as_agent=False, session_key=channel)
        ctx.bind_app_workflow(workflow)
        ctx.bind_observability_identity(channel_id=channel)
        ctx._turn_key = turn
        ctx.push_active_workflow(workflow)

        def execute_workflow_query(command: str) -> str:
            alias = current_execute_alias()
            record_context_clause(current_scope(), alias, "Fixture " + command)
            return response_with_credential(command)

        def ask_user(question: str) -> str:
            raise AskUserSuspend(question)

        agent = build_tool_agent(ctx, Signature,
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
        ctx.pop_active_workflow()
        ctx.close()
        workflow.close()
        if (ctx, workflow) in self.open_sessions:
            self.open_sessions.remove((ctx, workflow))

    # -- the reads an agent can make -------------------------------------

    def reads(self, agent, scope, alias: str = "O1") -> dict[str, str]:
        """Every read path an agent has, at one moment."""
        archive = agent.observation_archive
        # search_memory's own two-tier resolution: hot cache, then SQLite.
        served = stored_handles(scope).get(alias) or archive.get(scope, alias)
        return {
            "search_memory": None if served is None else served["text"],
            "rehydration": archived_observation(alias, scope=scope,
                                                archive=archive),
            "archive_row": (archive.get(scope, alias) or {}).get("text"),
        }

    def assert_all_raw(self, reads: dict[str, str], where: str) -> None:
        for path, text in reads.items():
            with self.subTest(path=path, moment=where):
                self.assertIsNotNone(text, f"{path} read nothing {where}")
                self.assertIn(SK_TOKEN, text, f"{path} was degraded {where}")
                self.assertNotIn(REDACTED, text)

    def assert_stored_text(self, agent, scope, alias: str = "O1") -> None:
        """What a reader with no raw copy gets: the redacted row."""
        row = agent.observation_archive.get(scope, alias)
        self.assertIsNotNone(row)
        self.assertIn(REDACTED, row["text"])
        self.assertNotIn(SK_TOKEN, row["text"])

    def assert_disk_is_redacted(self, agent) -> None:
        self.assertNotIn(SK_TOKEN.encode("ascii"),
                         file_bytes(agent.observation_archive.db_path))


class TurnCompletionReleasesTests(LiveTurnFixture):
    """The raw copies follow the runtime's existing notion of "over"."""

    def test_binding_the_next_scope_releases_the_previous_turns_raw_copy(self) -> None:
        ctx, workflow, agent = self.make_session(channel="chan", turn="turn-1")
        self.script(agent, [("execute_workflow_query", {"command": "first"}),
                            ("finish", {})])
        with tracing.host_scope(ctx):
            agent.forward(user_query="fixture")
        first = agent.continuation_scope
        # Mid-turn, every read is raw and the file is not.
        self.assert_all_raw(self.reads(agent, first), "during the turn")
        self.assert_disk_is_redacted(agent)

        # The agent binding the NEXT turn is it saying the previous one is over.
        ctx._turn_key = "turn-2"
        self.script(agent, [("execute_workflow_query", {"command": "second"}),
                            ("finish", {})])
        with tracing.host_scope(ctx):
            agent.forward(user_query="fixture")
        second = agent.continuation_scope

        self.assertNotEqual(first, second)
        self.assert_stored_text(agent, first)
        # And the turn that is actually running kept its raw evidence.
        self.assert_all_raw(self.reads(agent, second), "in the second turn")
        self.assert_disk_is_redacted(agent)

    def test_closing_the_session_releases_the_turns_raw_copy(self) -> None:
        ctx, workflow, agent = self.make_session(channel="closed", turn="turn-1")
        self.script(agent, [("execute_workflow_query", {"command": "only"}),
                            ("finish", {})])
        with tracing.host_scope(ctx):
            agent.forward(user_query="fixture")
        scope = agent.continuation_scope
        self.assert_all_raw(self.reads(agent, scope), "during the turn")

        self.close_session(ctx, workflow)

        self.assert_stored_text(agent, scope)
        self.assertNotIn(scope.scope_id, archive_module._live_raw)
        # The hot copy went too, so memory cannot serve raw text for the turn.
        self.assertEqual(stored_handles(scope), {})

    def test_the_awaiting_user_guard_keeps_the_raw_copy(self) -> None:
        """A turn waiting on the user is not over, so its reads stay exact."""
        ctx, workflow, agent = self.make_session(channel="susp", turn="turn-1")
        self.script(agent, [("execute_workflow_query", {"command": "one"}),
                            ("ask_user", {"question": "continue?"})])
        with tracing.host_scope(ctx):
            prediction = agent.forward(user_query="fixture")
        self.assertTrue(prediction.suspended)
        scope = agent.continuation_scope

        ctx._awaiting_user = True
        ctx.pop_active_workflow()
        ctx.close()

        self.assert_all_raw(self.reads(agent, scope), "after a suspended close")
        self.assert_disk_is_redacted(agent)

    def test_an_exported_suspension_keeps_the_raw_copy(self) -> None:
        """The second half of the guard: ``export_suspended() is not None``."""
        ctx, workflow, agent = self.make_session(channel="susp2", turn="turn-1")
        self.script(agent, [("execute_workflow_query", {"command": "one"}),
                            ("ask_user", {"question": "continue?"})])
        with tracing.host_scope(ctx):
            agent.forward(user_query="fixture")
        scope = agent.continuation_scope
        self.assertIsNotNone(agent.export_suspended())

        # _awaiting_user was never set -- only the agent knows it is suspended.
        self.assertFalse(ctx._awaiting_user)
        ctx.pop_active_workflow()
        ctx.close()

        self.assertIn(scope.scope_id, archive_module._live_raw)
        self.assertIn(SK_TOKEN, agent.observation_archive.get(scope, "O1")["text"])

    def test_the_bind_scope_guard_keeps_a_still_suspended_agents_raw_copy(self) -> None:
        """``bind_scope`` releases nothing while ``self._suspended is not None``."""
        ctx, workflow, agent = self.make_session(channel="susp3", turn="turn-1")
        self.script(agent, [("execute_workflow_query", {"command": "one"}),
                            ("ask_user", {"question": "continue?"})])
        with tracing.host_scope(ctx):
            agent.forward(user_query="fixture")
        scope = agent.continuation_scope
        self.assertIsNotNone(agent._suspended)

        # A caller binding a scope by hand over a still-suspended agent.
        ctx._turn_key = "turn-2"
        with tracing.host_scope(ctx):
            agent.bind_scope()

        self.assertNotEqual(agent.continuation_scope, scope)
        self.assertIn(SK_TOKEN, agent.observation_archive.get(scope, "O1")["text"])


class SuspensionRoundTripTests(LiveTurnFixture):
    """A suspension resumed in the same process, and in a fresh one."""

    def test_a_suspension_resumed_in_a_fresh_process_reads_the_stored_text(self) -> None:
        """The accepted cost of redacting at the write, pinned rather than implied.

        The turn suspends, its payload goes through ``json.dumps``/``loads``
        exactly as a session state file carries it, the session is CLOSED (the
        eviction), and every process-local registry is emptied -- as close to
        a fresh process as one interpreter allows. That process has no raw
        copy, so the archive answers with the stored, redacted text. Once the
        resumed turn's own compaction re-archives the observations its
        trajectory still holds, this process holds their raw text again.
        """
        ctx, workflow, agent = self.make_session(channel="rt", turn="turn-1")
        self.script(agent, [("execute_workflow_query", {"command": "first"}),
                            ("execute_workflow_query", {"command": "second"}),
                            ("ask_user", {"question": "continue?"})])
        with tracing.host_scope(ctx):
            prediction = agent.forward(user_query="fixture")
        self.assertTrue(prediction.suspended)
        scope = agent.continuation_scope
        path = agent.observation_archive.db_path
        self.assert_all_raw(self.reads(agent, scope, "O1"), "during the turn")

        blob = json.loads(json.dumps(agent.export_suspended()))
        ctx._awaiting_user = True
        ctx.pop_active_workflow()
        ctx.close()
        reset_runtime_state()
        self.assertEqual(stored_handles(scope), {})

        ctx2, workflow2, agent2 = self.make_session(channel="rt", turn="turn-1")
        agent2.import_suspended(blob)
        self.assertEqual(agent2.continuation_scope, scope)
        self.assertEqual(agent2.observation_archive.db_path, path)
        for alias in ("O1", "O2"):
            self.assert_stored_text(agent2, scope, alias)

        self.script(agent2, [("execute_workflow_query", {"command": "third"}),
                             ("finish", {})])
        with tracing.host_scope(ctx2):
            resumed = agent2.resume("go on")

        self.assertIn(SK_TOKEN, resumed.trajectory["observation_3"])
        self.assert_all_raw(self.reads(agent2, scope, "O3"), "after the resume")
        self.assert_disk_is_redacted(agent2)

        self.close_session(ctx2, workflow2)
        for alias in ("O1", "O2", "O3"):
            self.assert_stored_text(agent2, scope, alias)


# ---------------------------------------------------------------------------
# The conversation summary that feeds the NEXT turn
# ---------------------------------------------------------------------------


class SummaryTests(LiveTurnFixture):
    """The summary fed to the next turn is built from RAW text.

    ``_finalize_agent_output`` summarises ``self._action_log``, whose
    ``response`` is the command's ``response_text`` captured at execution time
    and never read back from the evidence tables, and the summary it produces
    is what ``_refine_user_query`` feeds the LLM that refines the next turn's
    query. If anyone later routes the action log through the capture pipeline,
    these fail loudly.
    """

    def finalize(self, ctx, agent, scope):
        """Run the real ``_finalize_agent_output``, capturing what it summarises."""
        seen: dict[str, object] = {}
        archive = agent.observation_archive

        def spy(user_query, workflow_actions, final_agent_response):
            seen["user_query"] = user_query
            seen["workflow_actions"] = json.loads(json.dumps(workflow_actions))
            seen["final_agent_response"] = final_agent_response
            # What the archive served, and what the file held, AT THE MOMENT
            # the summary was produced.
            row = archive.get(scope, "O1")
            seen["archive_text_at_summary_time"] = None if row is None else row["text"]
            seen["file_at_summary_time"] = file_bytes(archive.db_path)
            return "a summary mentioning the api key", json.dumps({"seen": True})

        ctx._extract_conversation_summary = spy
        return seen, ctx._finalize_agent_output(
            "what is the connector's api key?",
            dspy.Prediction(final_answer=f"the key is {SK_TOKEN}"),
        )

    def test_the_summary_is_built_from_raw_text(self) -> None:
        ctx, workflow, agent = self.make_session(channel="sum", turn="turn-1")
        self.script(agent, [("execute_workflow_query", {"command": "show"}),
                            ("finish", {})])
        with tracing.host_scope(ctx):
            agent.forward(user_query="fixture")
        scope = agent.continuation_scope
        # The action log the real dispatch path appends, with the real response.
        ctx.append_action_log({
            "command": "show_connector",
            "command_name": "show_connector",
            "parameters": {},
            "response": response_with_credential("show"),
        })

        seen, output = self.finalize(ctx, agent, scope)

        actions = seen["workflow_actions"]
        self.assertEqual(len(actions), 1)
        self.assertIn(SK_TOKEN, actions[0]["response"])
        self.assertNotIn(REDACTED, actions[0]["response"])
        self.assertIn(SK_TOKEN, seen["final_agent_response"])
        # The turn was recorded, and the entry the next turn's refinement reads
        # is the one the summary produced.
        self.assertEqual(
            output.command_response.artifacts["conversation_summary"],
            "a summary mentioning the api key",
        )
        newest = ctx.conversation_history.messages[-1]
        self.assertIn("a summary mentioning the api key", json.dumps(newest))
        # And the action log itself is never rewritten by the offloading
        # runtime, which is the other way this could regress.
        self.assertIn(SK_TOKEN, ctx.action_log[0]["response"])

    def test_at_summary_time_the_file_is_redacted_and_the_turn_reads_raw(self) -> None:
        ctx, workflow, agent = self.make_session(channel="order", turn="turn-1")
        self.script(agent, [("execute_workflow_query", {"command": "show"}),
                            ("finish", {})])
        with tracing.host_scope(ctx):
            agent.forward(user_query="fixture")
        scope = agent.continuation_scope
        ctx.append_action_log({
            "command": "show_connector",
            "command_name": "show_connector",
            "parameters": {},
            "response": response_with_credential("show"),
        })

        seen, _ = self.finalize(ctx, agent, scope)

        self.assertIn(SK_TOKEN, seen["archive_text_at_summary_time"])
        self.assertNotIn(SK_TOKEN.encode("ascii"), seen["file_at_summary_time"])
        # Only the session close, which is strictly later, ends the raw reads.
        self.close_session(ctx, workflow)
        self.assert_stored_text(agent, scope)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

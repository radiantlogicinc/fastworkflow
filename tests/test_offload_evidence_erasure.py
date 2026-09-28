"""Offload evidence lives in the observability database and dies with its turn.

Every archived execute response is a row of ``offload_evidence`` (and its
subject a row of ``offload_subjects``) in the workflow's own observability
database, keyed by the turn that produced it and carrying that turn's channel
id. ``ObservabilityStore.forget_channel``, ``clear_conversations`` and
``prune`` erase those rows in the same transactions that erase the turn
record, experiment runs included. The evidence used to live in a second file
beside the database; that file is deleted, not imported, the next time a store
opens.

Everything here runs against databases created in this test's own temporary
directory. Nothing in this module reads or writes a store it did not create.
"""
from __future__ import annotations

import hashlib
import importlib
import os
import sqlite3
import stat
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from fastworkflow.observability import store as obs
from fastworkflow.observation_offloading import archive as archive_module
from fastworkflow.observation_offloading.archive import (
    RuntimeHandleArchive,
    RuntimeHandleScope,
)
from fastworkflow.observation_offloading.state import (
    record_event,
    register_scope,
    reset_runtime_state,
)
from fastworkflow.run_chatbot.server import run_clear_conversations, run_forget_channel

#: The file older builds kept the evidence in, spelled out rather than imported
#: so this module states the name it proves is gone.
LEGACY_SIDECAR_SUFFIX = ".offload-handles.sqlite3"

#: The evidence tables and the column that dates each row.
EVIDENCE_TABLES = {
    "offload_evidence": "persisted_at",
    "offload_subjects": "recorded_at",
    "offload_events": "recorded_at",
}

#: A credential shape the store's ``Redactor`` recognises with no help from
#: the environment.
SK_TOKEN = "sk-livekey1234567890abcdef"


def chatbot_scope(channel: str, turn: str = "turn-1") -> RuntimeHandleScope:
    """A scope exactly as ``scope_for_host`` builds it with no experiment claim."""
    return RuntimeHandleScope(
        store_identity="store",
        channel_id=channel,
        experiment_id="unbound",
        task_id="unbound",
        attempt=0,
        turn_key=turn,
    )


def experiment_scope(
    channel: str, turn: str = "turn-1", experiment: str = "exp-7"
) -> RuntimeHandleScope:
    """A scope as it is built when the session carries an experiment claim."""
    return RuntimeHandleScope(
        store_identity="store",
        channel_id=channel,
        experiment_id=experiment,
        task_id="task-3",
        attempt=1,
        turn_key=turn,
    )


def rows(marker: str, count: int = 30) -> list[str]:
    return ["%s-%03d  confidential row %d" % (marker, i, i) for i in range(count)]


def counts(db_path: str, turn_key: str | None = None) -> dict[str, int]:
    """Row counts per evidence table, optionally for one turn."""
    with sqlite3.connect(db_path) as conn:
        clause = " WHERE turn_key=?" if turn_key else ""
        args = (turn_key,) if turn_key else ()
        return {
            name: conn.execute(
                f"SELECT count(*) FROM {name}{clause}", args
            ).fetchone()[0]
            for name in EVIDENCE_TABLES
        }


def table_names(db_path: str) -> set[str]:
    with sqlite3.connect(db_path) as conn:
        return {
            name
            for (name,) in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }


class EvidenceFixture(unittest.TestCase):
    """An observability database populated through the real writers."""

    def setUp(self) -> None:
        reset_runtime_state()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(reset_runtime_state)
        self._restore_env: dict[str, str | None] = {}
        for name in (archive_module.REDACTION_ENV,):
            self._restore_env[name] = os.environ.pop(name, None)
        self.addCleanup(self._restore_environment)
        self.db_path = os.path.join(self.temp.name, "state", "observability.sqlite3")

    def _restore_environment(self) -> None:
        for name, value in self._restore_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    # -- population ------------------------------------------------------

    def populate(
        self, scope: RuntimeHandleScope, marker: str, *, text: str | None = None
    ) -> RuntimeHandleArchive:
        """One turn's worth of evidence, written by the production writers.

        An archived observation, its recorded subject and one diagnostic event
        naming the same marker -- one row in each of the tables a turn writes.
        """
        text = text or (
            "Observation O1 (execute_workflow_query)\n" + "\n".join(rows(marker))
        )
        archive = RuntimeHandleArchive(self.db_path)
        register_scope(scope, archive)
        record_event({"kind": "search_memory", "scope_id": scope.scope_id,
                      "question": "whose rows?", "answer": marker})
        archive.put_subject(scope, "O1", "Fixture " + marker)
        archive.persist(
            scope,
            alias="O1",
            offload_order=1,
            command_name="execute_workflow_query",
            step_index=1,
            text=text,
            text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        )
        return archive

    def seed_turn(self, scope: RuntimeHandleScope) -> None:
        """The turn record that names the same channel."""
        store = obs.ObservabilityStore(self.db_path)
        row = dict(
            turn_key=scope.turn_key, channel_id=scope.channel_id,
            conversation_id=None, ordinal=None, user_message="fixture",
            refined_user_message=None, entry_workflow_name="fixture",
            entry_context="", status="completed", success=1,
            failure_reason=None, answer="fixture answer",
            conversation_summary=None, conversation_traces=None,
            started_at=None, completed_at=None, suspended_ms=0,
            continuation_of=None, record_version=1, record_json="{}",
        )
        with store._connect() as conn:
            self.assertTrue(store.upsert_turn_row(conn, row, [], obs.Redactor()))

    def age_rows(self, days: int, *, turn_key: str | None = None) -> None:
        """Backdate every evidence timestamp, for one turn or for all of them."""
        stamp = (
            datetime.now(timezone.utc) - timedelta(days=days)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
        with sqlite3.connect(self.db_path) as conn:
            for table, column in EVIDENCE_TABLES.items():
                clause = " WHERE turn_key=?" if turn_key else ""
                conn.execute(
                    f'UPDATE "{table}" SET "{column}"=?{clause}',
                    (stamp, turn_key) if turn_key else (stamp,),
                )
            conn.commit()

    def file_bytes(self) -> bytes:
        blob = b""
        for suffix in ("", "-wal"):
            if os.path.exists(self.db_path + suffix):
                with open(self.db_path + suffix, "rb") as handle:
                    blob += handle.read()
        return blob

    def assert_readable(self, scope: RuntimeHandleScope, marker: str):
        """This turn's evidence is still there and still usable."""
        archive = RuntimeHandleArchive(self.db_path)
        recovered = archive.get(scope, "O1")
        self.assertIsNotNone(recovered)
        self.assertIn(marker, recovered["text"])
        self.assertEqual(archive.get_subject(scope, "O1"), "Fixture " + marker)

    def assert_erased(self, scope: RuntimeHandleScope, marker: str):
        """Nothing of this turn is left anywhere in the file."""
        self.assertEqual(
            set(counts(self.db_path, scope.turn_key).values()), {0},
            counts(self.db_path, scope.turn_key),
        )
        archive = RuntimeHandleArchive(self.db_path)
        self.assertIsNone(archive.get(scope, "O1"))
        self.assertIsNone(archive.get_subject(scope, "O1"))
        blob = self.file_bytes()
        self.assertNotIn(marker.encode("utf-8"), blob)
        self.assertNotIn(scope.scope_id.encode("ascii"), blob)


class ChannelErasureTests(EvidenceFixture):
    """Forgetting a channel takes its evidence with it, and only its own."""

    def test_public_forget_channel_erases_the_channels_evidence(self):
        erased = chatbot_scope("erase")
        kept = chatbot_scope("keep", turn="turn-2")
        self.populate(erased, "confidential-erase")
        self.populate(kept, "confidential-keep")
        self.seed_turn(erased)
        self.seed_turn(kept)
        self.assertEqual(counts(self.db_path, erased.turn_key),
                         dict.fromkeys(EVIDENCE_TABLES, 1))

        deleted = run_forget_channel(self.db_path, "erase")

        # The claim first, in the terms a deletion request is made in: none of
        # this channel's evidence is left anywhere in the file.
        self.assert_erased(erased, "confidential-erase")
        self.assert_readable(kept, "confidential-keep")
        self.assertEqual(deleted["turns"], 1)
        self.assertEqual(deleted["offload_evidence"], 1)
        self.assertEqual(deleted["offload_subjects"], 1)
        self.assertEqual(deleted["offload_events"], 1)
        self.assertEqual(deleted["offload_scopes_released"], 1)
        with sqlite3.connect(self.db_path) as conn:
            self.assertEqual(
                [row[0] for row in conn.execute("SELECT channel_id FROM turns")],
                ["keep"],
            )

    def test_both_evidence_tables_are_erased_with_the_channel(self):
        """A subject recorded without an observation goes with the channel too.

        A subject is written at dispatch, before its step completes, and
        stays on its own when the archive refuses the observation. It names
        whose listing a step was, so it is exactly the text a deletion request
        is about.
        """
        scope = chatbot_scope("erase")
        archive = RuntimeHandleArchive(self.db_path)
        archive.put_subject(scope, "O7", "Fixture subject-only-erase")
        self.assertEqual(counts(self.db_path), {"offload_evidence": 0,
                                                "offload_subjects": 1,
                                                "offload_events": 0})

        deleted = obs.ObservabilityStore(self.db_path).forget_channel("erase")

        self.assertEqual(deleted["offload_subjects"], 1)
        self.assertEqual(set(counts(self.db_path).values()), {0})
        self.assertNotIn(b"subject-only-erase", self.file_bytes())

    def test_clear_conversations_removes_all_evidence(self):
        first = chatbot_scope("one")
        second = experiment_scope("two", turn="turn-2")
        self.populate(first, "confidential-one")
        self.populate(second, "confidential-two")

        deleted = run_clear_conversations(self.db_path)

        self.assertEqual(deleted["offload_evidence"], 2)
        self.assertEqual(deleted["offload_subjects"], 2)
        self.assert_erased(first, "confidential-one")
        self.assert_erased(second, "confidential-two")

    def test_forget_channel_does_not_create_a_legacy_sidecar(self):
        deleted = obs.ObservabilityStore(self.db_path).forget_channel("nobody")
        self.assertEqual(deleted["offload_evidence"], 0)
        self.assertEqual(deleted["offload_subjects"], 0)
        self.assertFalse(os.path.exists(self.db_path + LEGACY_SIDECAR_SUFFIX))

    def test_erasure_drops_the_raw_copy_a_live_turn_was_reading(self):
        """In-process memory is part of the channel too.

        With redaction on, this process keeps the raw text of a redacted
        observation for as long as its turn is live. Forgetting the channel
        must not leave that copy answering reads after the row is gone.
        """
        scope = chatbot_scope("erase")
        text = f"connector: okta-prod\napi_key: {SK_TOKEN}\n"
        archive = self.populate(scope, "unused", text=text)
        self.assertEqual(archive.get(scope, "O1")["text"], text)
        self.assertIn(scope.scope_id, archive_module._live_raw)

        obs.ObservabilityStore(self.db_path).forget_channel("erase")

        self.assertNotIn(scope.scope_id, archive_module._live_raw)
        self.assertIsNone(archive.get(scope, "O1"))


class ExperimentEvidenceTests(EvidenceFixture):
    """Experiment evidence is not preserved; erasure is by channel and turn."""

    def test_no_scope_classification_is_persisted(self):
        """The experiment signal the old preservation rule read is not stored.

        Rows carry the turn, the channel and an opaque scope digest; nothing
        that says whether a row belongs to an experiment, and no store
        identity.
        """
        self.populate(experiment_scope("exp-channel"), "confidential-exp")
        with sqlite3.connect(self.db_path) as conn:
            for table in EVIDENCE_TABLES:
                columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
                with self.subTest(table=table):
                    self.assertTrue({"turn_key", "channel_id"} <= columns)
                    self.assertFalse(
                        {"scope_json", "experiment_id", "store_identity"} & columns
                    )
        self.assertNotIn(b"exp-7", self.file_bytes())

    def test_a_forget_aimed_at_another_channel_leaves_an_experiment_run(self):
        kept = experiment_scope("exp-channel")
        erased = chatbot_scope("chat-channel", turn="turn-2")
        self.populate(kept, "confidential-exp")
        self.populate(erased, "confidential-chat")

        obs.ObservabilityStore(self.db_path).forget_channel("chat-channel")

        self.assert_readable(kept, "confidential-exp")
        self.assert_erased(erased, "confidential-chat")

    def test_an_experiment_run_is_erased_by_a_forget_of_its_own_channel(self):
        scope = experiment_scope("exp-channel")
        self.populate(scope, "confidential-exp")
        self.seed_turn(scope)

        deleted = obs.ObservabilityStore(self.db_path).forget_channel("exp-channel")

        self.assertEqual(deleted["offload_evidence"], 1)
        self.assert_erased(scope, "confidential-exp")

    def test_an_experiment_run_is_aged_by_retention_like_any_turn(self):
        scope = experiment_scope("exp-channel")
        self.populate(scope, "confidential-exp")
        self.age_rows(400)

        deleted = obs.ObservabilityStore(self.db_path).prune(
            retention_days=30, max_bytes=1_000_000_000
        )

        self.assertEqual(deleted["offload_evidence"], 1)
        self.assert_erased(scope, "confidential-exp")

    def test_a_turn_with_no_turn_record_is_erased_by_its_channel(self):
        """With no bound turn key, a scope's turn key IS its channel id.

        Such a turn never appears in ``turns``, so a delete by the channel's
        turn keys alone would miss it; the ``channel_id`` column is what makes
        it erasable.
        """
        scope = chatbot_scope("bare-channel", turn="bare-channel")
        self.populate(scope, "confidential-bare")

        deleted = obs.ObservabilityStore(self.db_path).forget_channel("bare-channel")

        self.assertEqual(deleted["offload_evidence"], 1)
        self.assert_erased(scope, "confidential-bare")

    def test_a_turn_recorded_under_the_channel_is_erased_by_its_turn_key(self):
        """A row whose own channel column differs still goes with its turn."""
        scope = chatbot_scope("scope-channel", turn="turn-9")
        self.populate(scope, "confidential-turn")
        self.seed_turn(chatbot_scope("turn-channel", turn="turn-9"))

        obs.ObservabilityStore(self.db_path).forget_channel("turn-channel")

        self.assert_erased(scope, "confidential-turn")


    def test_clear_conversations_erases_experiment_evidence(self):
        scope = experiment_scope("exp-channel")
        self.populate(scope, "confidential-exp")

        obs.ObservabilityStore(self.db_path).clear_conversations()

        self.assert_erased(scope, "confidential-exp")

    def test_the_erasure_module_is_gone(self):
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("fastworkflow.observation_offloading.erasure")


class LegacySidecarTests(EvidenceFixture):
    """The old evidence file is deleted, never imported."""

    def plant_legacy_files(self) -> list[str]:
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        sidecar = self.db_path + LEGACY_SIDECAR_SUFFIX
        with sqlite3.connect(sidecar) as conn:
            conn.execute(
                "CREATE TABLE observation_offload_handles "
                "(scope_id TEXT, scope_json TEXT, alias TEXT, text_utf8 BLOB)"
            )
            conn.execute(
                "INSERT INTO observation_offload_handles VALUES (?,?,?,?)",
                ("s", "{}", "O1", b"legacy-confidential-row"),
            )
            conn.commit()
        planted = [sidecar]
        for suffix in ("-wal", "-shm", ".preserve"):
            with open(sidecar + suffix, "wb") as handle:
                handle.write(b"legacy")
            planted.append(sidecar + suffix)
        return planted

    def test_opening_the_archive_deletes_the_legacy_sidecar(self):
        planted = self.plant_legacy_files()

        RuntimeHandleArchive(self.db_path)

        for path in planted:
            with self.subTest(path=os.path.basename(path)):
                self.assertFalse(os.path.exists(path))
        self.assertNotIn(b"legacy-confidential-row", self.file_bytes())
        self.assertNotIn("observation_offload_handles", table_names(self.db_path))

    def test_opening_the_store_deletes_the_legacy_sidecar_and_its_sentinel(self):
        planted = self.plant_legacy_files()

        obs.ObservabilityStore(self.db_path)

        for path in planted:
            with self.subTest(path=os.path.basename(path)):
                self.assertFalse(os.path.exists(path))

    def test_an_undeletable_legacy_sidecar_does_not_stop_the_open(self):
        """Best effort: a sidecar that cannot be removed is logged, not raised."""
        os.makedirs(self.db_path + LEGACY_SIDECAR_SUFFIX)
        archive = RuntimeHandleArchive(self.db_path)
        self.assertTrue(archive.available)
        self.assertTrue(os.path.isdir(self.db_path + LEGACY_SIDECAR_SUFFIX))


class FileModeTests(EvidenceFixture):
    """The evidence is as private as every other row of the store."""

    def test_the_archive_alone_creates_a_private_database(self):
        """No trace sink, no store opened by anything else: only the archive."""
        parent = os.path.dirname(self.db_path)
        self.assertFalse(os.path.exists(parent))
        archive = self.populate(chatbot_scope("modes"), "confidential-modes")
        # Hold a connection open so the write-ahead log and shared-memory
        # files exist while their modes are read.
        conn = archive._connect()
        try:
            conn.execute("SELECT count(*) FROM offload_evidence").fetchone()
            self.assertEqual(stat.S_IMODE(os.stat(parent).st_mode), 0o700)
            for suffix in ("", "-wal", "-shm"):
                path = self.db_path + suffix
                if suffix and not os.path.exists(path):
                    continue
                with self.subTest(file=os.path.basename(path)):
                    self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)
            self.assertTrue(os.path.exists(self.db_path + "-wal"))
        finally:
            conn.close()

    def test_the_evidence_tables_are_an_additive_feature(self):
        """Declared by marker, created in place, and the schema version is unchanged."""
        RuntimeHandleArchive(self.db_path)
        store = obs.ObservabilityStore(self.db_path)
        self.assertTrue(store.has_feature(obs.FEATURE_OFFLOAD_EVIDENCE_V1))
        self.assertTrue(set(EVIDENCE_TABLES) <= table_names(self.db_path))
        with sqlite3.connect(self.db_path) as conn:
            self.assertEqual(
                conn.execute("PRAGMA user_version").fetchone()[0],
                obs.SCHEMA_VERSION,
            )


class RetentionTests(EvidenceFixture):
    """Age and size retention reach the evidence, one whole turn at a time."""

    def test_the_horizon_drops_an_old_turn_whole_and_keeps_a_recent_one(self):
        old = chatbot_scope("chat", turn="turn-old")
        recent = chatbot_scope("chat", turn="turn-recent")
        self.populate(old, "confidential-old")
        self.populate(recent, "confidential-recent")
        self.age_rows(400, turn_key=old.turn_key)

        deleted = obs.ObservabilityStore(self.db_path).prune(
            retention_days=30, max_bytes=1_000_000_000
        )

        self.assert_erased(old, "confidential-old")
        self.assert_readable(recent, "confidential-recent")
        self.assertEqual(deleted["offload_evidence"], 1)
        self.assertEqual(deleted["offload_subjects"], 1)

    def test_a_turn_is_aged_by_its_earliest_row_so_it_goes_whole(self):
        scope = chatbot_scope("chat")
        self.populate(scope, "confidential")
        # Only the subject is old: it is written at dispatch, before the
        # observation, so the turn began before the horizon and goes entirely.
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE offload_subjects SET recorded_at=?",
                ("2000-01-01T00:00:00Z",),
            )
            conn.commit()

        deleted = obs.ObservabilityStore(self.db_path).prune(
            retention_days=1, max_bytes=1_000_000_000
        )

        self.assertEqual(deleted["offload_evidence"], 1)
        self.assertEqual(set(counts(self.db_path).values()), {0})

    def test_the_size_cap_evicts_the_oldest_turns_first(self):
        """One retention batch frees enough, so the newest turn survives it."""
        turns = [chatbot_scope("chat", turn=f"turn-{index:02d}") for index in range(26)]
        for index, scope in enumerate(turns):
            self.populate(
                scope, f"sized-{index:02d}",
                text=f"sized-{index:02d}\n" + os.urandom(6_000).hex(),
            )
            self.age_rows(100 - index, turn_key=scope.turn_key)
        store = obs.ObservabilityStore(self.db_path)
        with store._connect() as conn:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        cap = store.db_size_bytes() - 60_000

        deleted = store.prune(retention_days=3650, max_bytes=cap)

        self.assertEqual(deleted["offload_evidence"], 25)
        self.assertLessEqual(store.db_size_bytes(), cap)
        remaining = RuntimeHandleArchive(self.db_path)
        for scope in turns[:25]:
            self.assertIsNone(remaining.get(scope, "O1"))
        self.assertIn("sized-25", remaining.get(turns[25], "O1")["text"])

    def test_a_pruned_conversationless_turn_takes_its_evidence(self):
        """The operator opt-in that deletes turn records deletes their evidence too."""
        scope = chatbot_scope("cli", turn="20000101T000000-cli")
        self.populate(scope, "confidential-cli")
        self.seed_turn(scope)

        deleted = obs.ObservabilityStore(self.db_path).prune(
            retention_days=30, max_bytes=1_000_000_000,
            include_conversationless_turns=True,
        )

        self.assertEqual(deleted["conversationless_turns"], 1)
        self.assert_erased(scope, "confidential-cli")

    def test_retention_does_not_create_a_legacy_sidecar(self):
        deleted = obs.ObservabilityStore(self.db_path).prune(retention_days=1)
        self.assertEqual(deleted["offload_evidence"], 0)
        self.assertFalse(os.path.exists(self.db_path + LEGACY_SIDECAR_SUFFIX))


class EventErasureTests(EvidenceFixture):
    """A turn's diagnostic events are erased and aged with its evidence."""

    def events(self, **filters) -> list[dict]:
        return obs.ObservabilityStore(self.db_path).offload_events(**filters)

    def test_forgetting_a_channel_removes_its_events_and_keeps_the_others(self):
        erased = chatbot_scope("erase")
        kept = chatbot_scope("keep", turn="turn-2")
        self.populate(erased, "confidential-erase")
        self.populate(kept, "confidential-keep")
        self.assertEqual(len(self.events(channel_id="erase")), 1)

        deleted = obs.ObservabilityStore(self.db_path).forget_channel("erase")

        self.assertEqual(deleted["offload_events"], 1)
        self.assertEqual(self.events(channel_id="erase"), [])
        remaining = self.events()
        self.assertEqual([item["channel_id"] for item in remaining], ["keep"])
        self.assertEqual(remaining[0]["event"]["answer"], "confidential-keep")
        self.assertNotIn(b"confidential-erase", self.file_bytes())

    def test_an_erasure_reports_the_scopes_it_released(self):
        """What replaced the event-file line count: the process scopes dropped."""
        self.populate(chatbot_scope("erase"), "confidential-erase")
        store = obs.ObservabilityStore(self.db_path)
        self.assertEqual(store.forget_channel("erase")["offload_scopes_released"], 1)
        self.assertEqual(store.forget_channel("erase")["offload_scopes_released"], 0)
        self.assertNotIn("offload_events_removed", store.forget_channel("erase"))

    def test_clear_conversations_removes_every_event(self):
        self.populate(chatbot_scope("one"), "confidential-one")
        self.populate(experiment_scope("two", turn="turn-2"), "confidential-two")

        deleted = obs.ObservabilityStore(self.db_path).clear_conversations()

        self.assertEqual(deleted["offload_events"], 2)
        self.assertEqual(self.events(), [])

    def test_retention_ages_events_with_their_turn(self):
        old = chatbot_scope("chat", turn="turn-old")
        recent = chatbot_scope("chat", turn="turn-recent")
        self.populate(old, "confidential-old")
        self.populate(recent, "confidential-recent")
        self.age_rows(400, turn_key=old.turn_key)

        deleted = obs.ObservabilityStore(self.db_path).prune(
            retention_days=30, max_bytes=1_000_000_000
        )

        self.assertEqual(deleted["offload_events"], 1)
        self.assertEqual([item["turn_key"] for item in self.events()], ["turn-recent"])

    def test_an_event_alone_ages_its_turn(self):
        """A turn whose earliest row is an event is dated by that event."""
        scope = chatbot_scope("chat", turn="turn-events-only")
        register_scope(scope, RuntimeHandleArchive(self.db_path))
        record_event({"kind": "rehydration_started", "scope_id": scope.scope_id})
        self.age_rows(400)

        deleted = obs.ObservabilityStore(self.db_path).prune(
            retention_days=30, max_bytes=1_000_000_000
        )

        self.assertEqual(deleted["offload_events"], 1)
        self.assertEqual(self.events(), [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

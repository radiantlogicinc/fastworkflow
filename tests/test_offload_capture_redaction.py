"""Stored offload evidence rides the trace sink's redaction pipeline.

An archive that wrote exactly the bytes a command returned would land a
credential in a command response verbatim in the observability database,
while the same text inside a span attribute was scrubbed by
``observability.store.Redactor``. The archive must not bypass that pipeline.

Redaction is a toggle, ``FW_OFFLOAD_EVIDENCE_REDACTION``, ON by default, and it
happens at WRITE time. Both states are first-class: a developer turns it off
because the archive's whole job is to reproduce what the agent read, and devops
leave it on so a secret is not written to disk. Each archived observation
records which mode produced it, so an archive stays auditable about its own
fidelity.

The process that wrote a redacted row keeps its raw text in memory while the
turn is live, so its own reads are exact; that is the subject of
``test_offload_seal_timing``. The claims here about STORED text are made after
that memory is released, which is what any other process sees.

The claims are made in BYTES wherever a leak is the thing being denied: every
scan opens the database (and its write-ahead log) and searches the raw bytes.

Everything runs against databases created in this test's own temporary
directory. Nothing here reads or writes a store it did not create.
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
import tempfile
import unittest

from fastworkflow.observability import store as obs
from fastworkflow.observation_offloading import archive as archive_module
from fastworkflow.observation_offloading.archive import (
    PersistenceError,
    RuntimeHandleArchive,
    RuntimeHandleScope,
    redaction_mode,
)
from fastworkflow.observation_offloading.compact import archive_execute_observations
from fastworkflow.observation_offloading.state import (
    archived_digest,
    reclaim_scope,
    reset_runtime_state,
    stored_handles,
)

REDACTION_ENV = archive_module.REDACTION_ENV
REDACTION_ON = archive_module.REDACTION_ON
REDACTION_OFF = archive_module.REDACTION_OFF

#: A credential shape ``Redactor._SECRET_PATTERNS`` recognises with no help
#: from the environment, so this case does not depend on how the process was
#: started.
SK_TOKEN = "sk-livekey1234567890abcdef"

#: ...and one the scrub only knows about because the variable's NAME marks it
#: as secret. Included because it is the half of the pipeline a hand-rolled
#: regex in this package would not have had, and therefore the evidence that
#: the real one is being reused rather than imitated.
API_KEY_VAR = "IDOZLM_PLANTED_SERVICE_API_KEY"
ENV_SECRET = "planted-env-secret-value-zlm"

REDACTED = "[REDACTED]"


def response_with_credential() -> str:
    """One command response of the shape a workflow really returns."""
    return (
        "connector: okta-prod\n"
        f"api_key: {SK_TOKEN}\n"
        f"fallback_key: {ENV_SECRET}\n"
        "rows: 3 users synchronised\n"
    )


def innocent_response() -> str:
    return (
        "connector: okta-prod\n"
        "rows: 3 users synchronised\n"
    )


def chatbot_scope(channel: str = "chat", turn: str = "turn-1") -> RuntimeHandleScope:
    """A scope exactly as ``scope_for_host`` builds it with no experiment claim."""
    return RuntimeHandleScope(
        store_identity="store", channel_id=channel, experiment_id="unbound",
        task_id="unbound", attempt=0, turn_key=turn,
    )


def experiment_scope(channel: str = "chat", turn: str = "turn-1") -> RuntimeHandleScope:
    return RuntimeHandleScope(
        store_identity="store", channel_id=channel, experiment_id="exp-7",
        task_id="task-3", attempt=1, turn_key=turn,
    )


class RedactionFixture(unittest.TestCase):
    """An observability database in a temporary directory, clean configuration."""

    def setUp(self) -> None:
        reset_runtime_state()
        self.addCleanup(reset_runtime_state)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self._restore_env: dict[str, str | None] = {}
        for name in (
            REDACTION_ENV,
            "FW_OBS_CAPTURE_PROFILE",
            API_KEY_VAR,
        ):
            self._restore_env[name] = os.environ.pop(name, None)
        self.addCleanup(self._restore_environment)
        # Warn-once state is module-level and would otherwise leak between
        # cases in either direction.
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

    def plant_env_secret(self) -> None:
        """A loaded secret, set BEFORE any Redactor is built (it snapshots)."""
        os.environ[API_KEY_VAR] = ENV_SECRET

    def file_bytes(self) -> bytes:
        """Every byte the database occupies, write-ahead log included."""
        blob = b""
        for suffix in ("", "-wal", "-journal"):
            path = self.db_path + suffix
            if os.path.exists(path):
                with open(path, "rb") as handle:
                    blob += handle.read()
        return blob

    def persist(self, text: str, *, scope=None, alias: str = "O1", release: bool = True):
        """Archive *text*; by default end the turn, so reads return what is stored."""
        scope = scope or chatbot_scope()
        archive = RuntimeHandleArchive(self.db_path)
        archive.persist(
            scope, alias=alias, offload_order=1,
            command_name="execute_workflow_query", step_index=1,
            text=text, text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        )
        if release:
            reclaim_scope(scope)
        return archive, archive.get(scope, alias)


class DefaultIsOnTests(RedactionFixture):
    """An unconfigured deployment is the safe one."""

    def test_nothing_configured_means_redaction_on(self) -> None:
        self.assertNotIn(REDACTION_ENV, os.environ)
        self.assertEqual(redaction_mode(), REDACTION_ON)

    def test_an_empty_value_is_not_a_configuration(self) -> None:
        os.environ[REDACTION_ENV] = "   "
        self.assertEqual(redaction_mode(), REDACTION_ON)

    def test_an_unrecognised_value_warns_and_uses_the_default(self) -> None:
        os.environ[REDACTION_ENV] = "of"  # the typo that matters most
        with self.assertLogs(archive_module.logger, level="WARNING") as logs:
            self.assertEqual(redaction_mode(), REDACTION_ON)
        self.assertIn("FW_OFFLOAD_EVIDENCE_REDACTION=of", "\n".join(logs.output))
        # Warned once, not once per archived observation.
        self.assertEqual(redaction_mode(), REDACTION_ON)

    def test_a_typo_does_not_leave_a_credential_in_the_file(self) -> None:
        """The warning is not the point; what it falls back to is."""
        os.environ[REDACTION_ENV] = "yes-please"
        with self.assertLogs(archive_module.logger, level="WARNING"):
            self.persist(response_with_credential(), release=False)
        self.assertNotIn(SK_TOKEN.encode("ascii"), self.file_bytes())
        record = RuntimeHandleArchive(self.db_path).capture_record(chatbot_scope(), "O1")
        self.assertEqual(record["redaction"], REDACTION_ON)
        self.assertTrue(record["redacted"])

    def test_the_operator_spellings_resolve(self) -> None:
        for value, expected in (
            ("on", REDACTION_ON), ("ON", REDACTION_ON), ("1", REDACTION_ON),
            ("true", REDACTION_ON), ("off", REDACTION_OFF), ("0", REDACTION_OFF),
            ("false", REDACTION_OFF), ("No", REDACTION_OFF),
        ):
            with self.subTest(value=value):
                os.environ[REDACTION_ENV] = value
                self.assertEqual(redaction_mode(), expected)


class StoredBytesTests(RedactionFixture):
    """What is on disk, in both states, scanned as bytes."""

    def test_a_credential_is_not_left_in_the_file_by_default(self) -> None:
        self.plant_env_secret()
        archive, stored = self.persist(response_with_credential())

        blob = self.file_bytes()
        self.assertNotIn(SK_TOKEN.encode("ascii"), blob)
        self.assertNotIn(ENV_SECRET.encode("ascii"), blob)
        # Not silence, and not a stub: everything that was not a secret is
        # still there, which is what keeps a redacted archive worth reading.
        self.assertIn(b"rows: 3 users synchronised", blob)
        self.assertIn(REDACTED, stored["text"])
        self.assertIn("connector: okta-prod", stored["text"])
        self.assertEqual(archive.get(chatbot_scope(), "O1")["text"], stored["text"])

    def test_redaction_off_stores_the_response_verbatim(self) -> None:
        self.plant_env_secret()
        os.environ[REDACTION_ENV] = REDACTION_OFF
        text = response_with_credential()
        archive, stored = self.persist(text)

        blob = self.file_bytes()
        self.assertIn(SK_TOKEN.encode("ascii"), blob)
        self.assertIn(ENV_SECRET.encode("ascii"), blob)
        # And the archive reproduces the observation exactly, even with no raw
        # copy in memory, which is the reason a developer turns the toggle off.
        self.assertEqual(stored["text"], text)
        self.assertEqual(archive.get(chatbot_scope(), "O1")["text"], text)

    def test_a_response_with_no_secret_is_stored_byte_identical_under_both(self) -> None:
        text = innocent_response()
        for mode in (REDACTION_ON, REDACTION_OFF):
            with self.subTest(mode=mode):
                reset_runtime_state()
                os.environ[REDACTION_ENV] = mode
                db_path = os.path.join(self.temp.name, f"{mode}.sqlite3")
                archive = RuntimeHandleArchive(db_path)
                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                archive.persist(
                    chatbot_scope(), alias="O1", offload_order=1,
                    command_name="execute_workflow_query", step_index=1,
                    text=text, text_sha256=digest,
                )
                reclaim_scope(chatbot_scope())
                stored = archive.get(chatbot_scope(), "O1")
                self.assertEqual(stored["text"], text)
                self.assertEqual(stored["text_sha256"], digest)


class CapturePolicyRecordTests(RedactionFixture):
    """An archive that can be asked about its own fidelity."""

    def test_the_policy_version_is_recorded_with_redaction_on(self) -> None:
        self.plant_env_secret()
        archive, _ = self.persist(response_with_credential())
        record = archive.capture_record(chatbot_scope(), "O1")
        self.assertEqual(
            record["capture_policy_version"], obs.CAPTURE_POLICY_VERSION
        )
        self.assertEqual(record["capture_profile"], "debug")
        self.assertEqual(record["redaction"], REDACTION_ON)
        self.assertTrue(record["redacted"])
        self.assertEqual(
            record["raw_utf8_bytes"],
            len(response_with_credential().encode("utf-8")),
        )

    def test_the_policy_version_is_recorded_with_redaction_off(self) -> None:
        os.environ[REDACTION_ENV] = REDACTION_OFF
        archive, _ = self.persist(response_with_credential())
        record = archive.capture_record(chatbot_scope(), "O1")
        self.assertEqual(
            record["capture_policy_version"], obs.CAPTURE_POLICY_VERSION
        )
        self.assertEqual(record["redaction"], REDACTION_OFF)
        self.assertFalse(record["redacted"])
        # No policy was consulted, and that is not the same as the `debug`
        # profile having been consulted and done nothing.
        self.assertEqual(record["capture_profile"], "")

    def test_a_redacted_row_is_distinguishable_from_one_with_no_secret(self) -> None:
        """The requirement the version alone does not meet."""
        archive, _ = self.persist(response_with_credential(), alias="O1",
                                  release=False)
        archive.persist(
            chatbot_scope(), alias="O2", offload_order=2,
            command_name="execute_workflow_query", step_index=2,
            text=innocent_response(),
            text_sha256=hashlib.sha256(
                innocent_response().encode("utf-8")
            ).hexdigest(),
        )
        leaky = archive.capture_record(chatbot_scope(), "O1")
        clean = archive.capture_record(chatbot_scope(), "O2")
        self.assertEqual(leaky["redaction"], clean["redaction"], REDACTION_ON)
        self.assertEqual(
            leaky["capture_policy_version"], clean["capture_policy_version"]
        )
        self.assertTrue(leaky["redacted"])
        self.assertFalse(clean["redacted"])

    def test_the_record_is_part_of_the_evidence_row(self) -> None:
        """No separate fidelity table, so no row can lack its record.

        A row with no record used to read as UNKNOWN; that state is gone with
        the table, because the record is written in the same INSERT as the
        bytes it describes.
        """
        self.persist(innocent_response())
        with sqlite3.connect(self.db_path) as conn:
            tables = {row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
            record = conn.execute(
                "SELECT capture_policy_version, capture_profile, redaction, "
                "redacted, raw_utf8_bytes FROM offload_evidence"
            ).fetchone()
        self.assertNotIn("observation_capture_policy", tables)
        self.assertEqual(record[2], REDACTION_ON)
        self.assertFalse(hasattr(archive_module, "UNKNOWN_CAPTURE_RECORD"))
        # An alias nothing was stored for has no record, and says so.
        self.assertIsNone(
            RuntimeHandleArchive(self.db_path).capture_record(chatbot_scope(), "O404")
        )

    def test_the_record_describes_the_bytes_that_were_actually_kept(self) -> None:
        """A second persist of the same alias does not rewrite either half.

        The row was written under redaction ``off``. Turning the toggle on
        afterwards and re-persisting the SAME text must not change either the
        bytes or the record that describes them, because ``persist`` is
        insert-or-nothing. It is also not a collision -- it is the same
        observation -- so it is allowed rather than refused.
        """
        os.environ[REDACTION_ENV] = REDACTION_OFF
        archive, _ = self.persist(response_with_credential())
        os.environ[REDACTION_ENV] = REDACTION_ON
        again = archive.persist(
            chatbot_scope(), alias="O1", offload_order=1,
            command_name="execute_workflow_query", step_index=1,
            text=response_with_credential(),
            text_sha256=hashlib.sha256(
                response_with_credential().encode("utf-8")
            ).hexdigest(),
        )
        self.assertEqual(again["text"], response_with_credential())
        reclaim_scope(chatbot_scope())
        self.assertEqual(
            archive.get(chatbot_scope(), "O1")["text"], response_with_credential()
        )
        self.assertEqual(
            archive.capture_record(chatbot_scope(), "O1")["redaction"], REDACTION_OFF
        )

    def test_a_different_text_under_one_alias_is_still_a_collision(self) -> None:
        """Refused while the raw copy is held, and after it is released."""
        other = "an entirely different observation\n"
        for release in (False, True):
            with self.subTest(released=release):
                reset_runtime_state()
                scope = chatbot_scope(turn=f"turn-{release}")
                archive, _ = self.persist(response_with_credential(), scope=scope,
                                          release=release)
                with self.assertRaises(PersistenceError):
                    archive.persist(
                        scope, alias="O1", offload_order=1,
                        command_name="execute_workflow_query", step_index=1,
                        text=other,
                        text_sha256=hashlib.sha256(other.encode("utf-8")).hexdigest(),
                    )


class DigestMeaningTests(RedactionFixture):
    """Two digests, and neither quietly becomes the other."""

    def test_the_callers_digest_is_still_checked_against_the_raw_text(self) -> None:
        archive = RuntimeHandleArchive(self.db_path)
        with self.assertRaises(PersistenceError):
            archive.persist(
                chatbot_scope(), alias="O1", offload_order=1,
                command_name="execute_workflow_query", step_index=1,
                text=response_with_credential(),
                text_sha256=hashlib.sha256(b"something else").hexdigest(),
            )

    def test_the_stored_digest_covers_the_stored_bytes(self) -> None:
        self.plant_env_secret()
        archive, stored = self.persist(response_with_credential())
        self.assertEqual(
            stored["text_sha256"],
            hashlib.sha256(stored["text"].encode("utf-8")).hexdigest(),
        )
        # A read verifies the row against that column, so a redacted row is
        # readable rather than a permanent digest failure.
        self.assertIsNotNone(archive.get(chatbot_scope(), "O1"))
        self.assertEqual(len(archive.list(chatbot_scope())), 1)

    def test_the_raw_digest_still_means_the_raw_response(self) -> None:
        """Through the real archiver, which is where both digests are made."""
        self.plant_env_secret()
        scope = chatbot_scope()
        archive = RuntimeHandleArchive(self.db_path)
        text = response_with_credential()
        trajectory = {
            "thought_0": "look",
            "tool_name_0": "execute_workflow_query",
            "tool_args_0": {"command": "show_connector"},
            "observation_0": text,
        }
        archived = archive_execute_observations(
            trajectory, scope=scope, selected_archive=archive
        )
        raw_digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        self.assertEqual([row["alias"] for row in archived], ["O1"])
        # The archiver's own record of "this text is already written" is the
        # digest of what the COMMAND returned, unchanged by redaction. It is an
        # idempotence key; computing it from the stored bytes would make the
        # archiver rewrite every step at every step.
        self.assertEqual(archived[0]["text_sha256"], raw_digest)
        self.assertEqual(archived_digest(scope, "O1"), raw_digest)
        # While the turn is live this process reads the raw text with its raw
        # digest, so the hot copy, the archive read and the agent's prompt agree.
        self.assertEqual(archive.get(scope, "O1")["text_sha256"], raw_digest)
        self.assertEqual(stored_handles(scope)["O1"]["text"], text)
        # The file never held the raw text.
        self.assertNotIn(SK_TOKEN.encode("ascii"), self.file_bytes())
        # And once the turn is over the archive's digest covers what it kept.
        reclaim_scope(scope)
        stored = archive.get(scope, "O1")
        self.assertNotEqual(stored["text_sha256"], raw_digest)
        self.assertEqual(
            stored["text_sha256"],
            hashlib.sha256(stored["text"].encode("utf-8")).hexdigest(),
        )

    def test_with_redaction_off_every_digest_is_the_same_digest(self) -> None:
        os.environ[REDACTION_ENV] = REDACTION_OFF
        scope = chatbot_scope()
        archive = RuntimeHandleArchive(self.db_path)
        text = response_with_credential()
        trajectory = {
            "thought_0": "look",
            "tool_name_0": "execute_workflow_query",
            "tool_args_0": {"command": "show_connector"},
            "observation_0": text,
        }
        archive_execute_observations(
            trajectory, scope=scope, selected_archive=archive
        )
        raw_digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        self.assertEqual(archived_digest(scope, "O1"), raw_digest)
        self.assertEqual(archive.get(scope, "O1")["text_sha256"], raw_digest)
        self.assertEqual(stored_handles(scope)["O1"]["text_sha256"], raw_digest)


class ErasureStillWorksTests(RedactionFixture):
    """The toggle changes what is stored, not whether it can be erased."""

    def test_a_redacted_row_and_its_fidelity_record_are_erased_together(self) -> None:
        scope = chatbot_scope("erase")
        archive, _ = self.persist(response_with_credential(), scope=scope)
        self.assertIsNotNone(archive.capture_record(scope, "O1"))

        deleted = obs.ObservabilityStore(self.db_path).forget_channel("erase")

        self.assertEqual(deleted["offload_evidence"], 1)
        self.assertIsNone(archive.get(scope, "O1"))
        self.assertIsNone(archive.capture_record(scope, "O1"))
        blob = self.file_bytes()
        self.assertNotIn(b"okta-prod", blob)
        self.assertNotIn(scope.scope_id.encode("ascii"), blob)

    def test_an_experiment_scope_is_erased_like_any_other(self) -> None:
        experiment = experiment_scope("exp-channel")
        chatbot = chatbot_scope("chat-channel", turn="turn-2")
        self.persist(response_with_credential(), scope=experiment)
        self.persist(response_with_credential(), scope=chatbot)

        deleted = obs.ObservabilityStore(self.db_path).clear_conversations()

        self.assertEqual(deleted["offload_evidence"], 2)
        archive = RuntimeHandleArchive(self.db_path)
        self.assertIsNone(archive.get(experiment, "O1"))
        self.assertIsNone(archive.get(chatbot, "O1"))

    def test_retention_ages_a_redacted_turn_with_its_record(self) -> None:
        scope = chatbot_scope("old")
        self.persist(response_with_credential(), scope=scope)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE offload_evidence SET persisted_at=?",
                         ("2000-01-01T00:00:00Z",))
            conn.commit()

        deleted = obs.ObservabilityStore(self.db_path).prune(
            retention_days=30, max_bytes=1_000_000_000
        )

        self.assertEqual(deleted["offload_evidence"], 1)
        archive = RuntimeHandleArchive(self.db_path)
        self.assertIsNone(archive.get(scope, "O1"))
        self.assertIsNone(archive.capture_record(scope, "O1"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

"""The experiment container (epic `fix-bn1`), end to end.

Integration tests against a real SQLite store, the real `todo_list_workflow`,
and the real stdlib chatbot server on an ephemeral port. No Mock fixtures, per
`.cursor/rules/testing_rules.mdc` — the fakes stop at the NLU/command boundary,
which is the same line `tests/test_observability/store.py` draws.

Design: `docs/experiment_container_design.md`. Each test names the ruling it
pins, because several of these exist to prevent a *silent* wrong answer rather
than a crash, and a reader who does not know which invariant is at stake will
delete them as redundant.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import stat
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import pytest

import fastworkflow
from fastworkflow.observability import store as obs
from fastworkflow import state_paths
from fastworkflow.command_executor import CommandExecutor
from fastworkflow.experiment.runner import (
    ExperimentHarness,
    ExperimentTask,
    channel_for,
    derived_outcome,
)
from fastworkflow.run_chatbot import server as run_chatbot_server
from fastworkflow.workflow_execution_context import WorkflowExecutionContext


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def workflow_path(tmp_path, monkeypatch) -> str:
    monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "state"))
    wf = tmp_path / "my_workflow"
    wf.mkdir()
    return str(wf)


@pytest.fixture
def db_path(workflow_path) -> str:
    return state_paths.observability_db(workflow_path)


@pytest.fixture
def store(db_path) -> obs.ObservabilityStore:
    return obs.ObservabilityStore(db_path)


@pytest.fixture
def todo_workflow_path() -> str:
    return str(Path(__file__).parent.joinpath("todo_list_workflow").resolve())


@pytest.fixture
def initialized_fastworkflow(tmp_path, monkeypatch):
    monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "state"))
    fastworkflow.init({})
    from fastworkflow.command_routing import RoutingRegistry

    RoutingRegistry.clear_registry()
    yield
    RoutingRegistry.clear_registry()
    obs.close_all_sinks()


@pytest.fixture
def deterministic_commands(monkeypatch):
    """Every command succeeds with a fixed response.

    The fake stops at `CommandExecutor.invoke_command`, the same boundary
    `tests/test_observability/store.py` fakes: the workflow, the WEC, the turn
    accumulator, the sink and the store are all real.
    """

    def fake_invoke(cls, session, command: str):
        return fastworkflow.CommandOutput(
            command_name=command.split()[0] if command else "",
            command_response=fastworkflow.CommandResponse(response=f"ok:{command}"),
        )

    monkeypatch.setattr(CommandExecutor, "invoke_command", classmethod(fake_invoke))


def _sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _a_probe_turn():
    """One real turn record, for asserting a quiesced writer resumed work."""
    output = fastworkflow.CommandOutput(
        command_name="add_todo",
        command_response=fastworkflow.CommandResponse(response="ok"),
    )
    turn_output = fastworkflow.TurnOutput(
        turn_key=fastworkflow.mint_turn_key(),
        status=fastworkflow.TurnStatus.COMPLETED,
        answer="answer",
        command_outputs=[output],
    )
    return fastworkflow.TurnResult(
        turn_output=turn_output,
        channel_id="probe",
        conversation_id=1,
        user_message="probe",
        conversation_summary="s",
        conversation_traces="t",
        entry_workflow_name="w",
        entry_context="C",
    )


def _turn_row(turn_key: str, channel_id: str, **overrides) -> dict:
    row = {
        "turn_key": turn_key,
        "channel_id": channel_id,
        "conversation_id": None,
        "ordinal": None,
        "user_message": "hello",
        "refined_user_message": None,
        "entry_workflow_name": "wf",
        "entry_context": "ctx",
        "status": "completed",
        "success": 1,
        "failure_reason": None,
        "answer": "hi",
        "conversation_summary": None,
        "conversation_traces": None,
        "started_at": "2026-08-29T00:00:00Z",
        "completed_at": "2026-08-29T00:00:01Z",
        "suspended_ms": 0,
        "continuation_of": None,
        "record_version": 1,
        "experiment_id": None,
        "task_id": None,
        "attempt": None,
        "record_json": "{}",
    }
    row.update(overrides)
    return row


def _write_turn(store: obs.ObservabilityStore, row: dict) -> None:
    with store._connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        store.upsert_turn_row(conn, row, [], store._store_redactor())
        conn.commit()


def _seed(store, experiment_id, tasks, attempts, outcomes, declared=None):
    """Create an experiment and write its attempt rows with given outcomes."""
    store.create_experiment(
        experiment_id,
        f"label-{experiment_id}",
        declared_tasks=declared[0] if declared else tasks,
        declared_attempts=declared[1] if declared else attempts,
    )
    for task_index in range(tasks):
        task_id = f"t{task_index}"
        for n in range(1, attempts + 1):
            channel = channel_for(experiment_id, task_id, n)
            store.start_attempt(experiment_id, task_id, n, channel)
            outcome = outcomes(task_id, n)
            if outcome is not None:
                store.finish_attempt(
                    experiment_id,
                    task_id,
                    n,
                    outcome=outcome,
                    outcome_source="test_grader",
                )


# ----------------------------------------------------------------------
# `[XR5]`: the schema change is additive in both directions
# ----------------------------------------------------------------------


class TestAdditiveSchema:
    def test_fresh_db_has_the_tables_columns_and_indexes(self, db_path):
        obs.ObservabilityStore(db_path)
        conn = sqlite3.connect(db_path)
        try:
            tables = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            indexes = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='index'"
                )
            }
            turn_cols = {r[1] for r in conn.execute("PRAGMA table_info(turns)")}
            conv_cols = {
                r[1] for r in conn.execute("PRAGMA table_info(conversations)")
            }
        finally:
            conn.close()
        assert {
            "experiments",
            "experiment_attempts",
            "experiment_evidence_runs",
        } <= tables
        # The columns must come from the CREATE TABLE literals on a fresh DB:
        # the guarded ALTER is skipped there (PRAGMA table_info returns nothing
        # before the table exists), and idx_turns_experiment names them, so a
        # DDL that relied on the ALTER alone would raise "no such column" here.
        assert {"experiment_id", "task_id", "attempt"} <= turn_cols
        assert {"experiment_id", "task_id", "attempt"} <= conv_cols
        assert {"idx_turns_experiment", "idx_conv_experiment_attempt"} <= indexes

    def test_schema_version_is_six_for_create_time_only_columns(self, db_path):
        """fix-42b added create-time-only experiment columns and bumped v1->v2;
        fix-qe2 added experiment_attempts.runtime_snapshot_json and bumped
        v2->v3; fix-aw5 added feedback in v4; fix-46l.2 added feedback
        provenance in v5; fix-w6w added experiment archival in v6. All are
        create-time columns with no migration path."""
        obs.ObservabilityStore(db_path)
        conn = sqlite3.connect(db_path)
        try:
            assert conn.execute("PRAGMA user_version").fetchone()[0] == 6
            attempt_cols = {
                r[1] for r in conn.execute("PRAGMA table_info(experiment_attempts)")
            }
            experiment_cols = {
                r[1] for r in conn.execute("PRAGMA table_info(experiments)")
            }
        finally:
            conn.close()
        assert obs.SCHEMA_VERSION == 6
        assert "runtime_snapshot_json" in attempt_cols
        assert "archived" in experiment_cols

    def test_a_pre_v6_db_is_replaced_instead_of_migrated(self, db_path):
        """No legacy support: a populated v1 store is never migrated. The writer
        deletes it and creates a fresh store in its place; the read-only view
        refuses it and leaves it untouched.

        Built with the pre-`fix-bn1` CREATE TABLE statements at user_version 1.
        There is no ALTER/migration path at all any more (fresh schema,
        fix-49m.3): the CREATE literal is the only creator of every column.
        """
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """CREATE TABLE conversations (
                channel_id TEXT NOT NULL, conversation_id INTEGER NOT NULL,
                topic TEXT, summary TEXT, status TEXT, next_ordinal INTEGER,
                started_at TEXT, last_turn_at TEXT, updated_at TEXT,
                PRIMARY KEY (channel_id, conversation_id))"""
        )
        conn.execute(
            """CREATE TABLE turns (
                turn_key TEXT PRIMARY KEY, channel_id TEXT NOT NULL,
                conversation_id INTEGER, ordinal INTEGER,
                user_message TEXT NOT NULL, refined_user_message TEXT,
                entry_workflow_name TEXT, entry_context TEXT,
                status TEXT NOT NULL, success INTEGER NOT NULL,
                failure_reason TEXT, answer TEXT, conversation_summary TEXT,
                conversation_traces TEXT, started_at TEXT, completed_at TEXT,
                suspended_ms INTEGER, continuation_of TEXT,
                record_version INTEGER NOT NULL, record_json TEXT NOT NULL)"""
        )
        conn.execute(
            """CREATE TABLE diagnostics (
                key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL)"""
        )
        conn.execute(
            "INSERT INTO diagnostics VALUES ('schema_features', ?, 'x')",
            (json.dumps(["distillation_v1"]),),
        )
        conn.execute(
            "INSERT INTO turns (turn_key, channel_id, user_message, status, "
            "success, record_version, record_json) "
            "VALUES ('legacy', 'c1', 'hi', 'completed', 1, 1, '{}')"
        )
        conn.execute("PRAGMA user_version = 1")
        conn.commit()
        conn.close()

        # A reader never alters what it inspects.
        with pytest.raises(obs.IncompatibleObservabilityDB) as excinfo:
            obs.ReadOnlyObservabilityStore(db_path)
        message = str(excinfo.value)
        assert "schema v1" in message
        assert "requires v6" in message
        assert "carries no migration" in message

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
            turn_cols = {r[1] for r in conn.execute("PRAGMA table_info(turns)")}
            tables = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            row = dict(
                conn.execute("SELECT * FROM turns WHERE turn_key='legacy'").fetchone()
            )
            features = set(
                json.loads(
                    conn.execute(
                        "SELECT value FROM diagnostics WHERE key='schema_features'"
                    ).fetchone()[0]
                )
            )
        finally:
            conn.close()
        # Untouched: no columns added, no tables created, no markers merged.
        assert "experiment_id" not in turn_cols
        assert "experiments" not in tables
        assert row["user_message"] == "hi"
        assert features == {"distillation_v1"}

        # The writer replaces it: the legacy row is gone and the file is a
        # current store.
        store = obs.ObservabilityStore(db_path)
        assert store.has_feature(obs.FEATURE_EXPERIMENTS_V1)
        conn = sqlite3.connect(db_path)
        try:
            assert conn.execute("PRAGMA user_version").fetchone()[0] == 6
            assert conn.execute(
                "SELECT count(*) FROM turns WHERE turn_key='legacy'"
            ).fetchone()[0] == 0
            assert "experiment_id" in {
                r[1] for r in conn.execute("PRAGMA table_info(turns)")
            }
        finally:
            conn.close()

    def test_the_marker_row_is_the_only_source_of_features(self, db_path):
        """fix-9zb: the column-sniffing fallback is gone, and must stay gone.

        It existed for "a DB migrated by a build that added the columns before
        the marker existed" — a legacy shape the fresh-schema rule (fix-49m.3)
        no longer admits: every DB that opens is at SCHEMA_VERSION and was
        created from the literal schema with `_merge_schema_features` writing
        its markers in the same transaction. Sniffing could then only re-derive
        what the row already says, and on a DB whose row is genuinely missing it
        would be guessing the capabilities of a schema this build did not write.
        So the answer is now the honest empty set, even though every experiments
        column the old fallback keyed on is still present.
        """
        obs.ObservabilityStore(db_path)
        conn = sqlite3.connect(db_path)
        try:
            conn.execute("DELETE FROM diagnostics WHERE key='schema_features'")
            conn.commit()
            turn_cols = {
                row[1] for row in conn.execute("PRAGMA table_info(turns)").fetchall()
            }
        finally:
            conn.close()
        assert "experiment_id" in turn_cols  # what the old fallback keyed on
        store = obs.ObservabilityStore(db_path, migrate=False)
        assert store.has_feature(obs.FEATURE_EXPERIMENTS_V1) is False
        # This upstream-shaped port intentionally excludes the distillation
        # container. Its marker must therefore not be inferred from experiment
        # columns alone.
        assert store.has_feature(obs.FEATURE_DISTILLATION_V1) is False
        # And a store opened normally, with its markers intact, still reads them.
        assert obs.ObservabilityStore(db_path).has_feature(obs.FEATURE_EXPERIMENTS_V1)

    def test_migration_is_idempotent(self, db_path):
        for _ in range(3):
            obs.ObservabilityStore(db_path)
        conn = sqlite3.connect(db_path)
        try:
            cols = [r[1] for r in conn.execute("PRAGMA table_info(turns)")]
        finally:
            conn.close()
        assert cols.count("experiment_id") == 1

    def test_a_pre_experiments_db_degrades_instead_of_raising(self, db_path):
        """`[DR29]`: readers feature-detect rather than raise `no such table`."""
        conn = sqlite3.connect(db_path)
        conn.execute(
            "CREATE TABLE diagnostics (key TEXT PRIMARY KEY, value TEXT NOT NULL, "
            "updated_at TEXT NOT NULL)"
        )
        conn.commit()
        conn.close()
        store = obs.ObservabilityStore(db_path, migrate=False)
        assert store.has_feature(obs.FEATURE_EXPERIMENTS_V1) is False
        assert store.list_experiments() == []
        assert store.get_experiment("exp-x") is None
        assert store.experiment_attempt_rows("exp-x") == []
        assert store.experiment_labels_for_turn("tk") is None
        assert store.list_distillation_runs(experiment_id="exp-x") == []

    def test_list_turns_still_works_on_an_unmigrated_db(self, db_path):
        """The base turn list is the whole point of the debug UI.

        A viewer opened on a post-mortem snapshot never migrates it ([R12]), so
        an unguarded projection of the three new columns would 500 the main
        view forever with "internal error: OperationalError" and no reason a
        human could act on.
        """
        conn = sqlite3.connect(db_path)
        conn.execute(
            """CREATE TABLE turns (
                turn_key TEXT PRIMARY KEY, channel_id TEXT NOT NULL,
                conversation_id INTEGER, ordinal INTEGER,
                user_message TEXT NOT NULL, refined_user_message TEXT,
                entry_workflow_name TEXT, entry_context TEXT,
                status TEXT NOT NULL, success INTEGER NOT NULL,
                failure_reason TEXT, answer TEXT, conversation_summary TEXT,
                conversation_traces TEXT, started_at TEXT, completed_at TEXT,
                suspended_ms INTEGER, continuation_of TEXT,
                record_version INTEGER NOT NULL, record_json TEXT NOT NULL)"""
        )
        conn.execute(
            "CREATE TABLE diagnostics (key TEXT PRIMARY KEY, value TEXT NOT NULL, "
            "updated_at TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT INTO turns (turn_key, channel_id, user_message, status, "
            "success, record_version, record_json) "
            "VALUES ('old', 'c1', 'hi', 'completed', 1, 1, '{}')"
        )
        conn.commit()
        conn.close()
        store = obs.ObservabilityStore(db_path, migrate=False)
        assert store.has_feature(obs.FEATURE_EXPERIMENTS_V1) is False
        rows = store.list_turns()  # must not raise
        assert [r["turn_key"] for r in rows] == ["old"]
        assert "experiment_id" not in rows[0]
        # An experiment filter against a DB that records none matches nothing —
        # honest, where silently ignoring the filter would be worse than raising.
        assert store.list_turns(experiment_id="exp-x") == []


# ----------------------------------------------------------------------
# `[XR12]`: terminal invalid — at the STORE
# ----------------------------------------------------------------------


class TestExperimentDescription:
    def test_the_description_is_free_text_a_recreate_may_revise(self, store):
        store.create_experiment(
            "exp-1", "first", declared_tasks=1, declared_attempts=1
        )
        store.create_experiment(
            "exp-1", "second", declared_tasks=1, declared_attempts=1
        )
        assert store.get_experiment("exp-1")["description"] == "second"

    def test_an_unwritten_description_is_empty_not_an_error(self, store):
        store.create_experiment("exp-1", "", declared_tasks=1, declared_attempts=1)
        assert store.get_experiment("exp-1")["description"] == ""

    def test_a_non_string_description_is_refused(self, store):
        with pytest.raises(ValueError):
            store.create_experiment(
                "exp-1", None, declared_tasks=1, declared_attempts=1
            )


class TestExperimentArchive:
    def test_archive_is_durable_and_included_in_reads(self, store, db_path):
        store.create_experiment(
            "exp-1", "first", declared_tasks=1, declared_attempts=1
        )
        store.update_experiment_archived("exp-1", True)

        reopened = obs.ObservabilityStore(db_path)
        assert reopened.get_experiment("exp-1")["archived"] is True
        assert reopened.list_experiments()[0]["archived"] is True

        reopened.update_experiment_archived("exp-1", False)
        assert obs.ObservabilityStore(db_path).get_experiment("exp-1")["archived"] is False

    def test_archive_requires_a_boolean_and_an_existing_experiment(self, store):
        store.create_experiment(
            "exp-1", "first", declared_tasks=1, declared_attempts=1
        )
        with pytest.raises(ValueError):
            store.update_experiment_archived("exp-1", 1)
        with pytest.raises(obs.ExperimentNotFound):
            store.update_experiment_archived("missing", True)


class TestInvalidIsTerminal:
    def test_a_resume_cannot_clear_an_invalid_verdict(self, store):
        _seed(store, "exp-1", tasks=1, attempts=1, outcomes=lambda t, n: "pass")
        assert store.complete_experiment("exp-1") == "complete"
        store.complete_experiment("exp-1", force_invalid="operator", detail="bad run")
        # re-create (the resume path) then try to complete again
        store.create_experiment(
            "exp-1", "L", declared_tasks=1, declared_attempts=1
        )
        assert store.get_experiment("exp-1")["status"] == "invalid"
        assert store.complete_experiment("exp-1") == "invalid"

    def test_invalid_detail_is_append_only(self, store):
        store.create_experiment("exp-1", "L", declared_tasks=1, declared_attempts=1)
        with store._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            store.invalidate_experiments_in_txn(conn, ["exp-1"], "turns_erased", "first cause")
            store.invalidate_experiments_in_txn(conn, ["exp-1"], "operator", "second cause")
            conn.commit()
        detail = store.get_experiment("exp-1")["invalid_detail"]
        assert "first cause" in detail and "second cause" in detail

    def test_a_closed_experiment_refuses_new_attempts(self, store):
        """Re-running a driver script that pins its experiment_id must not
        silently overwrite the 45 verdicts a reported score rested on. `run()`
        has no guard of its own — unlike `resume()` — so it lives here."""
        _seed(store, "exp-1", tasks=1, attempts=1, outcomes=lambda t, n: "pass")
        assert store.complete_experiment("exp-1") == "complete"
        with pytest.raises(obs.ExperimentIsClosed):
            store.start_attempt("exp-1", "t0", 1, "c0")
        store.complete_experiment("exp-1", force_invalid="operator")
        with pytest.raises(obs.ExperimentIsClosed):
            store.start_attempt("exp-1", "t9", 1, "c9")

    def test_a_recorded_evidence_failure_cannot_be_overwritten_clean(self, store):
        """Segment invalidity is monotone, like `status <> 'invalid'`.

        A rewritable verdict is a verdict that can be revised after seeing the
        outcome, which is the thing this container exists to prevent.
        """
        store.create_experiment("exp-1", "L", declared_tasks=1, declared_attempts=1)
        store.record_evidence_segment(
            "exp-1", 1, "evr-x", {"valid": False, "problems": ["dropped a record"]}
        )
        store.record_evidence_segment("exp-1", 1, "evr-x", {"valid": True})
        assert store.get_experiment("exp-1")["evidence_runs"][0]["valid"] == 0
        store.start_attempt("exp-1", "t0", 1, "c0")
        store.finish_attempt("exp-1", "t0", 1, outcome="pass", outcome_source="g")
        assert store.complete_experiment("exp-1") == "invalid"

    def test_recreating_under_a_different_capture_regime_is_refused(self, store):
        """The stored profile is what `compare_experiments` gates on, so a run
        whose second half was captured under another policy must not compare as
        if both halves matched."""
        store.create_experiment(
            "exp-1", "L", declared_tasks=1, declared_attempts=1,
            capture_profile="debug", capture_policy_version="1",
        )
        with pytest.raises(obs.CaptureRegimeChanged):
            store.create_experiment(
                "exp-1", "L", declared_tasks=1, declared_attempts=1,
                capture_profile="evidence", capture_policy_version="1",
            )
        # the same regime is a normal resume and is allowed
        store.create_experiment(
            "exp-1", "L2", declared_tasks=1, declared_attempts=1,
            capture_profile="debug", capture_policy_version="1",
        )
        assert store.get_experiment("exp-1")["description"] == "L2"

    def test_a_write_to_a_missing_experiment_raises(self, store):
        with pytest.raises(obs.ExperimentNotFound):
            store.update_experiment_notes("exp-nope", "x")
        with pytest.raises(obs.ExperimentNotFound):
            store.start_attempt("exp-nope", "t", 1, "chan")
        with pytest.raises(obs.ExperimentNotFound):
            store.record_evidence_segment("exp-nope", 1, "evr-x", {"valid": True})


# ----------------------------------------------------------------------
# `[XR13]` / `[XR14]`: the verdict is written, the denominator is declared
# ----------------------------------------------------------------------


class TestScoring:
    def test_declared_denominator_is_required_and_positive(self, store):
        with pytest.raises(ValueError):
            store.create_experiment("e", "L", declared_tasks=0, declared_attempts=1)
        with pytest.raises(ValueError):
            store.create_experiment("e", "L", declared_tasks=1, declared_attempts=-2)
        with pytest.raises(TypeError):
            store.create_experiment("e", "L")  # both are required kwargs

    def test_pass_k_is_computed_from_retained_per_attempt_outcomes(self, store):
        """`[XR3]`/`[XR13]`: a per-task RATE cannot produce this answer.

        Two tasks, three attempts each, five of six passing. pass@1 is 5/6 for
        either arrangement of the single failure — but pass^3 is 1/2 when both
        failures land on one task and would be 0 if they were spread. Storing a
        rate at write time destroys exactly this distinction.
        """
        outcomes = {("t0", 1): "pass", ("t0", 2): "pass", ("t0", 3): "pass",
                    ("t1", 1): "pass", ("t1", 2): "pass", ("t1", 3): "fail"}
        _seed(store, "exp-1", tasks=2, attempts=3,
              outcomes=lambda t, n: outcomes[(t, n)])
        assert store.complete_experiment("exp-1") == "complete"
        score = store.experiment_scores("exp-1")
        assert score["pass_at_1"] == pytest.approx(5 / 6)
        assert score["pass_at_k"] == pytest.approx(1 / 2)
        # and the per-attempt detail is still there to be re-derived from
        rows = store.experiment_attempt_rows("exp-1")
        assert [r["outcome"] for r in rows] == [
            "pass", "pass", "pass", "pass", "pass", "fail"
        ]

    def test_an_unfinished_attempt_blocks_completion_even_though_it_has_rows(
        self, store
    ):
        """`[XR13]`: existence is not completion.

        The attempt that crashed halfway has a row and turns, so an
        existence-based denominator check would count it as observed and mark
        the experiment complete — the failure this ruling exists to prevent.
        """
        _seed(store, "exp-1", tasks=2, attempts=1,
              outcomes=lambda t, n: "pass" if t == "t0" else None)
        # t1 crashed HALFWAY: it has turn rows, which is what makes an
        # existence-based check count it as observed. Without these the test
        # only exercises the row half of its own docstring.
        conv = store.mint_conversation_id(
            channel_for("exp-1", "t1", 1), experiment_id="exp-1",
            task_id="t1", attempt=1,
        )
        _write_turn(
            store,
            _turn_row("tk-crash", channel_for("exp-1", "t1", 1),
                      conversation_id=conv, experiment_id="exp-1",
                      task_id="t1", attempt=1),
        )
        assert len(store.list_turns(experiment_id="exp-1", task_id="t1")) == 1
        assert store.complete_experiment("exp-1") == "invalid"
        experiment = store.get_experiment("exp-1")
        assert experiment["invalid_reason"] == "attempt_shortfall"
        assert "1 finished and 2 recorded of 2 declared" in experiment["invalid_detail"]

    def test_extra_attempts_cannot_pay_for_a_declared_task_that_never_ran(
        self, store
    ):
        """`[XR14]`: the SHAPE must match the declaration, not just the count.

        A resume whose task list gained two tasks and lost one reaches
        `finished == expected` with a declared task missing. Counting alone let
        that through, and `experiment_scores` then divided 4 scored attempts by
        a declared denominator of 3 and reported **pass@1 = 1.33** with
        `reportable: True`. A score above 1.0 is worse than no score.
        """
        store.create_experiment("exp-1", "L", declared_tasks=3, declared_attempts=1)
        for task_id in ("t0", "t1", "t2"):
            store.start_attempt("exp-1", task_id, 1, f"c:{task_id}")
        for task_id in ("t0", "t1"):
            store.finish_attempt(
                "exp-1", task_id, 1, outcome="pass", outcome_source="g"
            )
        # t2 never finished; two tasks outside the declared set did.
        for task_id in ("t3", "t4"):
            store.start_attempt("exp-1", task_id, 1, f"c:{task_id}")
            store.finish_attempt(
                "exp-1", task_id, 1, outcome="pass", outcome_source="g"
            )
        assert store.complete_experiment("exp-1") == "invalid"
        detail = store.get_experiment("exp-1")["invalid_detail"]
        assert "5 of 3 declared tasks" in detail
        score = store.experiment_scores("exp-1")
        assert score["reportable"] is False
        assert score["pass_at_1"] is None

    def test_a_score_can_never_exceed_one(self, store):
        """Defence in depth: `experiment_scores` divides by the DECLARED
        denominator, so rows that outnumber it produce a ratio above 1.0 rather
        than an error. Force the inconsistent state past the completion gate and
        assert the score layer still refuses."""
        store.create_experiment("exp-1", "L", declared_tasks=1, declared_attempts=1)
        store.start_attempt("exp-1", "t0", 1, "c0")
        store.finish_attempt("exp-1", "t0", 1, outcome="pass", outcome_source="g")
        assert store.complete_experiment("exp-1") == "complete"
        # The API refuses to write an attempt to a closed experiment, so force
        # the inconsistent state the way only a bug or a foreign writer could:
        # straight into the table. That is exactly what defence in depth is for.
        with store._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """INSERT INTO experiment_attempts
                   (experiment_id, task_id, attempt, channel_id, outcome,
                    outcome_source, restarts, started_at, finished_at)
                   VALUES ('exp-1', 't1', 1, 'c1', 'pass', 'g', 0, 's', 'f')"""
            )
            conn.commit()
        score = store.experiment_scores("exp-1")
        assert score["reportable"] is False
        assert score["pass_at_1"] is None
        assert "do not support a score" in score["reason_not_reportable"]

    def test_an_incomplete_outcome_blocks_completion(self, store):
        _seed(store, "exp-1", tasks=1, attempts=2,
              outcomes=lambda t, n: "pass" if n == 1 else "incomplete")
        assert store.complete_experiment("exp-1") == "invalid"

    def test_an_invalid_evidence_segment_blocks_completion(self, store):
        _seed(store, "exp-1", tasks=1, attempts=1, outcomes=lambda t, n: "pass")
        store.record_evidence_segment(
            "exp-1", 1, "evr-x", {"valid": False, "problems": ["a turn record was dropped"]}
        )
        assert store.complete_experiment("exp-1") == "invalid"
        assert store.get_experiment("exp-1")["invalid_reason"] == "evidence_run_invalid"

    def test_a_running_or_invalid_experiment_refuses_a_headline_number(self, store):
        _seed(store, "exp-1", tasks=1, attempts=1, outcomes=lambda t, n: "pass")
        running = store.experiment_scores("exp-1")
        assert running["reportable"] is False
        assert running["pass_at_1"] is None and running["pass_at_k"] is None
        assert "running" in running["reason_not_reportable"]
        # the per-task detail is still served — refusing a score is not refusing data
        assert running["tasks"]

    def test_the_closed_vocabularies_are_enforced(self, store):
        """`[XR15]`/§6.3: `outcome` and `invalid_reason` are closed sets.

        `invalid_reason` in particular: a free-text one would interpolate a
        `channel_id`, which under `[XR18]` embeds a caller-supplied `task_id`.
        """
        _seed(store, "exp-1", tasks=1, attempts=1, outcomes=lambda t, n: None)
        with pytest.raises(ValueError):
            store.finish_attempt(
                "exp-1", "t0", 1, outcome="passed", outcome_source="g"
            )
        with pytest.raises(ValueError):
            store.finish_attempt("exp-1", "t0", 1, outcome="pass", outcome_source="")
        with pytest.raises(ValueError):
            store.complete_experiment(
                "exp-1", force_invalid="turns erased by channel exp:x:t0:1"
            )

    def test_derived_verdicts_are_reported_as_their_own_source(self, store):
        """§8.1: `experiment_scores` surfaces the distinct sources it saw, which
        is what keeps a fallback from reading as a measurement."""
        _seed(store, "exp-1", tasks=1, attempts=1, outcomes=lambda t, n: None)
        store.finish_attempt(
            "exp-1", "t0", 1, outcome="pass", outcome_source="derived"
        )
        assert store.complete_experiment("exp-1") == "complete"
        assert store.experiment_scores("exp-1")["outcome_sources"] == ["derived"]

    def test_the_store_computes_the_verdict_rather_than_the_caller(self, store):
        """`[XR14]`: `complete_experiment` takes no `status` argument at all."""
        import inspect

        params = inspect.signature(store.complete_experiment).parameters
        assert "status" not in params
        assert set(params) == {"experiment_id", "force_invalid", "detail"}


class TestComparison:
    def _complete(self, store, experiment_id, failures=()):
        _seed(
            store, experiment_id, tasks=2, attempts=2,
            outcomes=lambda t, n: "fail" if (t, n) in failures else "pass",
        )
        assert store.complete_experiment(experiment_id) == "complete"

    def test_equal_cardinality_over_disjoint_task_sets_is_refused(self, store):
        """`[XR19]`: cardinality is not comparability.

        Two 2x2 runs sharing no task would otherwise join on `task_id`, find no
        overlap, and report "0 regressions" — a clean bill of health for a
        comparison that compared nothing.
        """
        self._complete(store, "exp-a")
        store.create_experiment("exp-b", "B", declared_tasks=2, declared_attempts=2)
        for task_id in ("z0", "z1"):
            for n in (1, 2):
                store.start_attempt("exp-b", task_id, n, f"c:{task_id}:{n}")
                store.finish_attempt(
                    "exp-b", task_id, n, outcome="pass", outcome_source="g"
                )
        assert store.complete_experiment("exp-b") == "complete"
        result = store.compare_experiments("exp-b", "exp-a")
        assert result["comparable"] is False
        assert any("task sets differ" in p for p in result["problems"])
        assert set(result["only_in_treatment"]) == {"z0", "z1"}

    def test_a_running_arm_is_refused(self, store):
        self._complete(store, "exp-a")
        store.create_experiment("exp-b", "B", declared_tasks=2, declared_attempts=2)
        result = store.compare_experiments("exp-b", "exp-a")
        assert result["comparable"] is False
        assert any("not complete" in p for p in result["problems"])

    def test_a_differing_declared_shape_is_refused(self, store):
        """`[XR19]`: comparing 2x2 against 2x1 is arithmetic that means nothing."""
        self._complete(store, "exp-base")
        _seed(store, "exp-treat", tasks=2, attempts=1, outcomes=lambda t, n: "pass")
        assert store.complete_experiment("exp-treat") == "complete"
        result = store.compare_experiments("exp-treat", "exp-base")
        assert result["comparable"] is False
        assert any("declared shapes differ" in p for p in result["problems"])

    def test_a_differing_capture_regime_is_refused(self, store):
        """`[XR19]`: two arms captured under different profiles are not
        measuring the same columns."""
        self._complete(store, "exp-base")
        self._complete(store, "exp-treat")
        with store._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "UPDATE experiments SET capture_profile='evidence' "
                "WHERE experiment_id='exp-treat'"
            )
            conn.commit()
        result = store.compare_experiments("exp-treat", "exp-base")
        assert result["comparable"] is False
        assert any("capture regimes differ" in p for p in result["problems"])

    def test_a_real_comparison_reports_both_flip_directions(self, store):
        self._complete(store, "exp-base", failures={("t0", 1)})
        self._complete(store, "exp-treat", failures={("t1", 2)})
        result = store.compare_experiments("exp-treat", "exp-base")
        assert result["comparable"] is True
        assert result["improved"] == ["t0"]
        assert result["regressed"] == ["t1"]
        # It reports flips and sample size and claims no significance.
        assert "p_value" not in result and "significant" not in result

    def test_it_says_how_many_flips_chance_alone_would_produce(self, store):
        """`fix-bn1.7`: "how many flips are attributable to variance".

        Two arms that differ in nothing still flip tasks, because pass^k is a
        threshold on a noisy quantity. Reporting "2 tasks flipped" without
        saying that ~1 was expected is how noise gets read as a result.
        """
        self._complete(store, "exp-base", failures={("t0", 1)})
        self._complete(store, "exp-treat", failures={("t1", 2)})
        result = store.compare_experiments("exp-treat", "exp-base")
        assert result["observed_flips"] == 2
        # Pinned, not bounded. Three tasks; t0 and t1 each have 3 passes and 1
        # fail pooled across the arms (p=0.75, p^2=0.5625 -> 2p^2(1-p^2)=0.4922
        # each) and t2 passes 4/4 (p=1 -> 0). 2 x 0.4922 = 0.984. A bound of
        # "greater than zero" would survive almost any change to the formula.
        assert result["expected_flips_if_nothing_changed"] == pytest.approx(
            0.984, abs=1e-3
        )

    def test_identical_arms_flip_nothing_but_still_carry_the_expectation(
        self, store
    ):
        self._complete(store, "exp-base", failures={("t0", 1)})
        self._complete(store, "exp-treat", failures={("t0", 1)})
        result = store.compare_experiments("exp-treat", "exp-base")
        assert result["observed_flips"] == 0
        # A task passed by both arms every time cannot flip (p=1 -> 0), so the
        # whole expectation comes from t0, which fails once in each arm:
        # pooled p = 2/4 = 0.5, p^2 = 0.25, 2 x 0.25 x 0.75 = 0.375.
        assert result["expected_flips_if_nothing_changed"] == pytest.approx(
            0.375, abs=1e-3
        )


# ----------------------------------------------------------------------
# `[XR17]`: binding validation, and the ordinary turn is untouched
# ----------------------------------------------------------------------


class TestBinding:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"experiment_id": "e"},
            {"experiment_id": "e", "task_id": ""},
            {"experiment_id": "e", "task_id": "t"},
            {"experiment_id": "e", "task_id": "t", "attempt": 0},
            {"experiment_id": "e", "task_id": "t", "attempt": -1},
            {"experiment_id": "e", "task_id": "t", "attempt": "1"},
            {"experiment_id": "e", "task_id": "t", "attempt": True},
            {"task_id": "t", "attempt": 1},
        ],
    )
    def test_a_partial_or_ill_typed_triple_is_refused(self, kwargs):
        ctx = WorkflowExecutionContext()
        with pytest.raises(ValueError):
            ctx.bind_observability_identity(channel_id="c", **kwargs)

    def test_the_bool_rejection_is_not_incidental(self):
        """`bool` is a subclass of `int`; `attempt=True` must not become 1."""
        ctx = WorkflowExecutionContext()
        with pytest.raises(ValueError):
            ctx.bind_observability_identity(
                channel_id="c", experiment_id="e", task_id="t", attempt=True
            )

    def test_a_complete_triple_binds(self):
        ctx = WorkflowExecutionContext()
        ctx.bind_observability_identity(
            channel_id="c", experiment_id="exp-1", task_id="t1", attempt=3
        )
        assert (ctx._experiment_id, ctx._task_id, ctx._attempt) == ("exp-1", "t1", 3)

    def test_an_ordinary_binding_leaves_the_labels_null(self):
        ctx = WorkflowExecutionContext()
        ctx.bind_observability_identity(channel_id="c", conversation_id=4)
        assert ctx._experiment_id is None
        assert ctx._task_id is None
        assert ctx._attempt is None

    def test_the_labels_survive_a_cross_process_suspension(self, tmp_path):
        """They ride `serialize_state` for the same reason `channel_id` does."""
        ctx = WorkflowExecutionContext(session_key="chan")
        ctx.bind_observability_identity(
            channel_id="chan", experiment_id="exp-1", task_id="t1", attempt=2
        )
        blob = ctx.serialize_state(channel_id="chan")
        assert (blob["experiment_id"], blob["task_id"], blob["attempt"]) == (
            "exp-1", "t1", 2,
        )
        restored = WorkflowExecutionContext(session_key="chan")
        restored.apply_serialized_state(blob)
        assert (
            restored._experiment_id,
            restored._task_id,
            restored._attempt,
        ) == ("exp-1", "t1", 2)

    def test_a_pre_fix_bn1_blob_restores_as_not_an_experiment(self):
        ctx = WorkflowExecutionContext(session_key="chan")
        ctx.bind_observability_identity(channel_id="chan")
        blob = ctx.serialize_state(channel_id="chan")
        blob.pop("experiment_id")
        blob.pop("task_id")
        blob.pop("attempt")
        restored = WorkflowExecutionContext(session_key="chan")
        restored.apply_serialized_state(blob)  # must not raise
        assert restored._experiment_id is None


class TestOrdinaryTurnIsUnaffected:
    def test_an_unlabelled_turn_round_trips_with_null_labels(
        self, initialized_fastworkflow, todo_workflow_path, db_path,
        deterministic_commands, tmp_path,
    ):
        sink = obs.SQLiteTraceSink(db_path)
        try:
            workflow = fastworkflow.Workflow.create(
                todo_workflow_path, workflow_id_str=f"plain-{uuid.uuid4().hex}"
            )
            ctx = WorkflowExecutionContext(run_as_agent=False, trace_sink=sink)
            ctx.bind_app_workflow(workflow)
            ctx.bind_observability_identity(channel_id="cli:plain")
            ctx.process_turn("add milk")
            assert sink.flush()
        finally:
            sink.close()
        store = obs.ObservabilityStore(db_path)
        turns = store.list_turns(channel_id="cli:plain")
        assert len(turns) == 1
        assert turns[0]["experiment_id"] is None
        assert turns[0]["task_id"] is None
        assert turns[0]["attempt"] is None
        # and it is invisible to every experiment-scoped read
        assert store.list_turns(experiment_id="exp-anything") == []


# ----------------------------------------------------------------------
# `[XR6]` / `[XR7]`: the capture-policy decision, and the dataflow claim
# ----------------------------------------------------------------------


class TestCapturePolicy:
    _SECRET = "sk-live-0123456789abcdefghijklmnopqrstuv"

    def test_experiment_prose_is_credential_scrubbed(self, store):
        store.create_experiment(
            "exp-1",
            f"label {self._SECRET}",
            declared_tasks=1,
            declared_attempts=1,
        )
        store.update_experiment_notes("exp-1", f"notes {self._SECRET}")
        experiment = store.get_experiment("exp-1")
        blob = json.dumps(experiment)
        assert self._SECRET not in blob
        # ...but the prose is otherwise intact and readable, which is the whole
        # point of scrub-only: a digest badge here would make an evidence-grade
        # bundle's own declaration unreadable.
        assert experiment["description"].startswith("label ")
        assert experiment["notes"].startswith("notes ")

    def test_the_evidence_segment_record_is_scrubbed(self, store):
        store.create_experiment("exp-1", "L", declared_tasks=1, declared_attempts=1)
        store.record_evidence_segment(
            "exp-1", 1, "evr-x",
            {"valid": True, "problems": [f"provider said {self._SECRET}"]},
        )
        segment = store.get_experiment("exp-1")["evidence_runs"][0]
        assert self._SECRET not in json.dumps(segment)
        assert segment["valid"] == 1

    def test_task_id_is_scrubbed_identically_on_both_write_routes(self, store):
        """`[XR7]`/§6.5: two routes must not produce two values for one label.

        `turns.task_id` is scrubbed by `upsert_turn_row`'s text loop and
        `conversations.task_id` by `mint_conversation_id`. If only one scrubbed,
        the two copies would stop being joinable — which is exactly the failure
        `_protected_text`'s docstring exists to prevent one layer down.
        """
        dirty = f"task-{self._SECRET}"
        store.create_experiment("exp-1", "L", declared_tasks=1, declared_attempts=1)
        conv = store.mint_conversation_id(
            "chan", experiment_id="exp-1", task_id=dirty, attempt=1
        )
        _write_turn(
            store,
            _turn_row(
                "tk1", "chan", conversation_id=conv,
                experiment_id="exp-1", task_id=dirty, attempt=1,
            ),
        )
        conn = sqlite3.connect(store.db_path)
        conn.row_factory = sqlite3.Row
        try:
            turn_task = conn.execute(
                "SELECT task_id FROM turns WHERE turn_key='tk1'"
            ).fetchone()["task_id"]
            conv_task = conn.execute(
                "SELECT task_id FROM conversations WHERE channel_id='chan'"
            ).fetchone()["task_id"]
        finally:
            conn.close()
        assert self._SECRET not in turn_task
        assert turn_task == conv_task  # joinable
        assert store.list_turns(task_id=conv_task)  # and selectable

    def test_a_secret_reaching_an_experiment_table_by_any_route_is_scrubbed(
        self, initialized_fastworkflow, todo_workflow_path, monkeypatch
    ):
        """`[XR6]`'s guarantee, asserted behaviourally.

        The signature guard below is a lint; this is the actual contract. Drive
        a real attempt whose command RAISES with a credential in the message and
        whose grader hands back a credential-bearing detail, then assert nothing
        in the three experiment tables carries it. These are the two routes by
        which content that is not caller-typed prose can reach the container:
        an exception repr through `error`/`detail_json`, and a grader payload.
        """

        def exploding(cls, session, command: str):
            raise RuntimeError(f"provider rejected {self._SECRET}")

        monkeypatch.setattr(
            CommandExecutor, "invoke_command", classmethod(exploding)
        )
        harness = ExperimentHarness(
            todo_workflow_path, description=f"L {self._SECRET}", run_as_agent=False,
        )
        harness.run(
            [ExperimentTask(task_id="t0", messages=["add milk"])],
            attempts=1,
            grader=lambda run: (
                "error", "test", None, {"note": f"grader saw {self._SECRET}"}
            ),
        )
        store = obs.ObservabilityStore(
            state_paths.observability_db(todo_workflow_path)
        )
        blob = json.dumps(
            [
                store.get_experiment(harness.experiment_id),
                store.experiment_attempt_rows(harness.experiment_id),
            ]
        )
        assert self._SECRET not in blob

    def test_the_experiment_write_methods_take_only_caller_supplied_scalars(self):
        """A lint, not the contract — see the behavioural test above.

        It guards one specific regression: a future writer that starts accepting
        a TurnResult/Workflow/CommandOutput would widen what these tables can
        hold, and `[XR6]`'s scrub-only ruling assumes they cannot.
        """
        import inspect

        forbidden = {"turn_result", "turn_output", "command_output", "workflow", "record_json"}
        for name in (
            "create_experiment",
            "update_experiment_notes",
            "start_attempt",
            "finish_attempt",
            "complete_experiment",
        ):
            params = set(
                inspect.signature(getattr(obs.ObservabilityStore, name)).parameters
            )
            assert not (params & forbidden), f"{name} accepts {params & forbidden}"

    def test_no_policy_path_constants_were_declared_for_this_surface(self):
        """`[XR6]`: a constant never passed to `policy.apply` is inert.

        `spans.channel_id`, the one genuinely scrub-only column already in the
        file, deliberately has no constant either. Declaring one here would
        promise a deployment an override that does not exist.
        """
        names = [n for n in dir(obs) if n.startswith("POLICY_PATH_")]
        assert not [n for n in names if "EXPERIMENT" in n]


# ----------------------------------------------------------------------
# `[XR18]` / `[XR16]`: the harness — independence and concurrency
# ----------------------------------------------------------------------


class TestHarness:
    def test_attempts_run_concurrently_on_separate_channels(
        self, initialized_fastworkflow, todo_workflow_path, monkeypatch
    ):
        """`[XR18]`: concurrency is the whole reason for one channel per attempt.

        A shared channel would serialise the run by construction (one
        `SessionStateStore` pending slot, one topic namespace), so this asserts
        overlap in wall-clock rather than merely asserting distinct channel ids.
        """
        live = []
        peak = {"n": 0}
        lock = threading.Lock()

        def slow_invoke(cls, session, command: str):
            with lock:
                live.append(1)
                peak["n"] = max(peak["n"], len(live))
            time.sleep(0.2)
            with lock:
                live.pop()
            return fastworkflow.CommandOutput(
                command_name="add_todo",
                command_response=fastworkflow.CommandResponse(response="ok"),
            )

        monkeypatch.setattr(
            CommandExecutor, "invoke_command", classmethod(slow_invoke)
        )
        tasks = [ExperimentTask(task_id=f"t{i}", messages=["add milk"]) for i in range(4)]
        harness = ExperimentHarness(
            todo_workflow_path, description="concurrency",
            run_as_agent=False, max_workers=4,
        )
        result = harness.run(tasks, attempts=1, grader=lambda run: ("pass", "g", 1.0, None))
        assert result["status"] == "complete"
        assert peak["n"] > 1, "attempts did not overlap; the run was serialised"

        store = obs.ObservabilityStore(
            state_paths.observability_db(todo_workflow_path)
        )
        rows = store.experiment_attempt_rows(harness.experiment_id)
        assert len({row["channel_id"] for row in rows}) == 4

    def test_two_attempts_of_one_task_do_not_share_a_conversation(
        self, initialized_fastworkflow, todo_workflow_path, deterministic_commands
    ):
        """Task independence: two attempts must not see each other's context."""
        harness = ExperimentHarness(
            todo_workflow_path, description="independence",
            run_as_agent=False, max_workers=2,
        )
        result = harness.run(
            [ExperimentTask(task_id="t0", messages=["add milk"])],
            attempts=2,
            grader=lambda run: ("pass", "g", 1.0, None),
        )
        assert result["status"] == "complete"
        store = obs.ObservabilityStore(
            state_paths.observability_db(todo_workflow_path)
        )
        turns = store.list_turns(experiment_id=harness.experiment_id, limit=100)
        assert len({t["channel_id"] for t in turns}) == 2
        # Distinct channels mean distinct conversation namespaces, so both
        # attempts are conversation 1 of their own channel and neither can see
        # the other's topic or history.
        assert {t["conversation_id"] for t in turns} == {1}

    def test_the_unique_index_refuses_a_second_conversation_for_one_attempt(
        self, store
    ):
        """`[XR3]`: uniqueness is enforced, not merely asserted."""
        store.create_experiment("exp-1", "L", declared_tasks=1, declared_attempts=1)
        store.mint_conversation_id("c1", experiment_id="exp-1", task_id="t0", attempt=1)
        with pytest.raises(sqlite3.IntegrityError):
            store.mint_conversation_id(
                "c2", experiment_id="exp-1", task_id="t0", attempt=1
            )

    def test_a_crash_leaves_the_experiment_running_and_unscoreable(
        self, initialized_fastworkflow, todo_workflow_path, deterministic_commands
    ):
        """A harness crash must never look like a finished run."""
        store = obs.ObservabilityStore(
            state_paths.observability_db(todo_workflow_path)
        )
        harness = ExperimentHarness(
            todo_workflow_path, description="crash", run_as_agent=False
        )
        tasks = [ExperimentTask(task_id=f"t{i}", messages=["add milk"]) for i in range(2)]
        store.create_experiment(
            harness.experiment_id, "crash", declared_tasks=2, declared_attempts=1,
            )
        harness._prepare_process()
        harness._run_attempt(tasks[0], 1, lambda run: ("pass", "g", 1.0, None))
        # ...and the process dies here, before task 1 and before completion.
        assert store.get_experiment(harness.experiment_id)["status"] == "running"
        score = store.experiment_scores(harness.experiment_id)
        assert score["reportable"] is False
        assert score["pass_at_1"] is None

    def test_resume_selects_on_the_completion_marker_not_on_turn_presence(
        self, initialized_fastworkflow, todo_workflow_path, deterministic_commands
    ):
        """`[XR13]`: an attempt with turns but no `finished_at` is unfinished.

        A resume that looked for "no terminal turn" would skip the attempt that
        crashed after its first turn — the exact case resume exists for.
        """
        store = obs.ObservabilityStore(
            state_paths.observability_db(todo_workflow_path)
        )
        tasks = [ExperimentTask(task_id=f"t{i}", messages=["add milk"]) for i in range(2)]
        harness = ExperimentHarness(
            todo_workflow_path, description="resume", run_as_agent=False
        )
        store.create_experiment(
            harness.experiment_id, "resume", declared_tasks=2, declared_attempts=1,
            )
        harness._prepare_process()
        harness._run_attempt(tasks[0], 1, lambda run: ("pass", "g", 1.0, None))
        harness._run_attempt(tasks[1], 1, lambda run: ("pass", "g", 1.0, None))
        # t1 crashed after writing a turn but before execution terminality or
        # its verdict landed.
        store._update_experiment(
            "UPDATE experiment_attempts SET execution_status=NULL, "
            "execution_finished_at=NULL, finished_at=NULL, outcome=NULL "
            "WHERE experiment_id=? AND task_id='t1'",
            (harness.experiment_id,),
            harness.experiment_id,
        )
        assert store.list_turns(experiment_id=harness.experiment_id, task_id="t1")

        resumed = ExperimentHarness(
            todo_workflow_path, description="resume", experiment_id=harness.experiment_id,
            run_as_agent=False,
        )
        result = resumed.resume(tasks, grader=lambda run: ("pass", "g", 1.0, None))
        assert result["status"] == "complete"
        rows = {r["task_id"]: r for r in store.experiment_attempt_rows(harness.experiment_id)}
        assert rows["t1"]["restarts"] == 1
        assert rows["t0"]["restarts"] == 0
        # restart deleted the abandoned turns rather than leaving two attempts'
        # worth of rows under one set of labels
        assert len(store.list_turns(experiment_id=harness.experiment_id, task_id="t1")) == 1

    def test_a_grader_that_raises_does_not_abort_the_run_or_score_the_agent(
        self, initialized_fastworkflow, todo_workflow_path, deterministic_commands
    ):
        """A judge's bug must not be attributed to the thing being judged."""

        def flaky(run):
            if run.task.task_id == "t1":
                raise RuntimeError("grader blew up")
            return ("pass", "g", 1.0, None)

        harness = ExperimentHarness(
            todo_workflow_path, description="grader", run_as_agent=False
        )
        tasks = [ExperimentTask(task_id=f"t{i}", messages=["add milk"]) for i in range(3)]
        result = harness.run(tasks, attempts=1, grader=flaky)
        store = obs.ObservabilityStore(
            state_paths.observability_db(todo_workflow_path)
        )
        rows = {r["task_id"]: r for r in store.experiment_attempt_rows(harness.experiment_id)}
        assert rows["t0"]["outcome"] == "pass"
        assert rows["t2"]["outcome"] == "pass"
        # `incomplete`, not `fail`: nothing was measured about the agent.
        assert rows["t1"]["outcome"] == "incomplete"
        assert rows["t1"]["outcome_source"] == "grader_error"
        assert result["status"] == "invalid"
        assert store.experiment_scores(harness.experiment_id)["reportable"] is False

    def test_the_evidence_segment_is_recorded_and_names_the_cache_posture(
        self, initialized_fastworkflow, todo_workflow_path, deterministic_commands
    ):
        """`[XR16]`: a run whose cache posture is unrecorded is uninterpretable."""
        harness = ExperimentHarness(
            todo_workflow_path, description="evidence", run_as_agent=False
        )
        harness.run(
            [ExperimentTask(task_id="t0", messages=["add milk"])],
            attempts=1,
            grader=lambda run: ("pass", "g", 1.0, None),
        )
        store = obs.ObservabilityStore(
            state_paths.observability_db(todo_workflow_path)
        )
        segments = store.get_experiment(harness.experiment_id)["evidence_runs"]
        assert len(segments) == 1
        assert segments[0]["evidence_run_id"].startswith("evr-")  # `[XR10]`
        posture = segments[0]["record"]["experiment"]["cache_posture"]
        assert posture["FW_LM_CACHE"] == "0"
        assert posture["FW_UTTERANCE_CACHE_SCOPE"] == "workflow"
        # ...and the posture is not merely echoed into its own record: the
        # process env dict really carries the values the seams read.
        assert fastworkflow.get_env_var("FW_LM_CACHE") == "0"
        assert fastworkflow.get_env_var("FW_UTTERANCE_CACHE_SCOPE") == "workflow"
        # Recorded either way, so a sweep that ran without the DSPy memory
        # policy is visible in its own evidence rather than silently different.
        assert "dspy_memory_policy" in posture

    def test_the_memory_policy_is_opt_in_because_it_cannot_be_uninstalled(
        self, todo_workflow_path
    ):
        """`install_policy` claims DSPy's config-owner thread process-wide and
        has no uninstall, so a library that called it on construction would
        poison every process that built a harness — including a test process,
        where the next FastAPI app's lifespan then fails `claim_async_owner()`.
        Observed exactly that way; hence opt-in.
        """
        import inspect

        params = inspect.signature(ExperimentHarness.__init__).parameters
        assert params["install_memory_policy"].default is False

    def test_defeating_the_caches_does_not_wipe_the_process_configuration(
        self, todo_workflow_path, restore_env_vars, tmp_path, monkeypatch
    ):
        """`fastworkflow.init` REPLACES the env dict; the harness must merge.

        Calling `init({FW_LM_CACHE: "0", ...})` here would delete `LLM_AGENT`
        and every `LITELLM_API_KEY_*` the run needs, and the failure would
        surface as "DSPy Language Model not provided" from somewhere unrelated.
        """
        monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "state"))
        fastworkflow.init({"LLM_AGENT": "m/x", "LITELLM_API_KEY_AGENT": "secret"})
        harness = ExperimentHarness(
            todo_workflow_path, description="env", run_as_agent=False
        )
        harness._prepare_process()
        assert fastworkflow.get_env_var("LLM_AGENT") == "m/x"
        assert fastworkflow.get_env_var("LITELLM_API_KEY_AGENT") == "secret"
        assert fastworkflow.get_env_var("FW_LM_CACHE") == "0"

    def test_duplicate_task_ids_in_a_task_set_are_refused(
        self, todo_workflow_path, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "state"))
        harness = ExperimentHarness(
            todo_workflow_path, description="dupes", run_as_agent=False
        )
        with pytest.raises(ValueError):
            harness.run(
                [ExperimentTask(task_id="t0"), ExperimentTask(task_id="t0")],
                attempts=1,
            )

    def test_an_in_process_run_with_an_archive_dir_produces_one(
        self, initialized_fastworkflow, todo_workflow_path, deterministic_commands,
        tmp_path,
    ):
        """fix-7de: the archive an in-process runner asked for must exist.

        The bug this pins was total and silent. `_prepare_process` takes the
        process sink BEFORE `evidence_run` opens — deliberately, so the verdict
        rests on live counters — and `archive_to` then refused every store whose
        writer was open. So a harness with `archive_dir` set produced NO archive,
        ever, and recorded "evidence archival failed: WriterStillOpen" as a
        problem on its own evidence segment: a run that did everything right and
        was marked invalid for it.

        Asserted end to end rather than on `archive_to` alone, because the two
        halves that were wrong are the runner's lifecycle and the store's
        refusal, and either one passing on its own proves nothing about the
        shape a real run takes.
        """
        bundle = tmp_path / "bundle"
        harness = ExperimentHarness(
            todo_workflow_path, description="archived",
            run_as_agent=False, archive_dir=str(bundle),
        )
        result = harness.run(
            [ExperimentTask(task_id="t0", messages=["add milk"])],
            attempts=1,
            grader=lambda run: ("pass", "g", 1.0, None),
        )

        assert result["status"] == "complete"
        assert result["evidence_problems"] == []
        assert result["evidence_valid"] is True
        # The writer was never closed to buy the archive: this is an archive
        # taken of a store that is still being recorded into.
        sink = obs.existing_observability_sink(todo_workflow_path)
        assert sink is not None and sink._closed is False

        store = obs.ObservabilityStore(
            state_paths.observability_db(todo_workflow_path)
        )
        segments = store.get_experiment(harness.experiment_id)["evidence_runs"]
        assert len(segments) == 1
        record = segments[0]["record"]
        assert segments[0]["valid"] == 1
        assert record["problems"] == []
        archive = record["archive"]
        assert archive is not None

        path = Path(archive["path"])
        assert path.parent == bundle
        assert _sha256_file(path) == archive["sha256"]
        assert archive["sealed"] is True
        assert archive["source_bytes_verified_unchanged"] is True
        assert stat.S_IMODE(path.stat().st_mode) == 0o444
        assert not Path(f"{path}-wal").exists()

        archived = obs.ReadOnlyObservabilityStore(str(path))
        assert (
            archived.get_experiment(harness.experiment_id)["experiment_id"]
            == harness.experiment_id
        )
        assert archived.list_turns(experiment_id=harness.experiment_id, limit=50)

    def test_archiving_a_live_store_does_not_change_its_bytes(
        self, initialized_fastworkflow, todo_workflow_path, deterministic_commands,
        tmp_path,
    ):
        """The quiesce is what makes the digest claim true (fix-7de).

        `archive_to` records `source_bytes_verified_unchanged` by digesting the
        source before and after the snapshot, and the writer's periodic health
        persist is a write that lands between those two digests — which is why
        `test_the_archive_is_immutable_and_its_digest_verifies` was flaky under
        load rather than wrong. Holding the writer still is not an optimisation
        of that check; it is the only thing that makes it pass for a reason.
        """
        harness = ExperimentHarness(
            todo_workflow_path, description="stable", run_as_agent=False
        )
        harness.run(
            [ExperimentTask(task_id="t0", messages=["add milk"])],
            attempts=1,
            grader=lambda run: ("pass", "g", 1.0, None),
        )
        db_path = state_paths.observability_db(todo_workflow_path)
        sink = obs.existing_observability_sink(todo_workflow_path)
        assert sink is not None

        store = obs.ObservabilityStore(db_path)
        # Ten back-to-back snapshots of a live store: one that succeeds by luck
        # succeeds once, not ten times.
        for index in range(10):
            archive = store.archive_to(str(tmp_path / f"live-{index}.sqlite3"))
            assert archive["source_bytes_verified_unchanged"] is True
            assert _sha256_file(Path(archive["path"])) == archive["sha256"]
        assert sink._closed is False
        assert sink._writer.is_alive()
        # And the writer picked its work straight back up.
        assert sink.emit_turn_record(_a_probe_turn()) is True

    def test_a_writer_this_process_cannot_reach_still_refuses(
        self, initialized_fastworkflow, todo_workflow_path, tmp_path
    ):
        """Quiescing replaces the refusal only where it can actually be done.

        A sink built directly never enters the factory registry, so
        `sink_for_db_path` cannot see it and `quiesced()` cannot be called on
        it — the same blindness this process has toward a writer in another
        process, which is the topology `_external_writer` exists to simulate in
        `tests/test_observability/evidence_run.py`. There is nothing to hold still, so the
        only honest answer is the one fix-7de kept: refuse.
        """
        db_path = state_paths.observability_db(todo_workflow_path)
        unreachable = obs.SQLiteTraceSink(db_path)
        try:
            assert obs.sink_for_db_path(db_path) is None  # invisible, as in prod
            with pytest.raises(obs.WriterStillOpen):
                obs.ObservabilityStore(db_path).archive_to(
                    str(tmp_path / "refused.sqlite3")
                )
            assert not (tmp_path / "refused.sqlite3").exists()
        finally:
            unreachable.close()

        # Once that writer is gone its marker is stale, and the store is
        # archivable again — an open marker must not wedge a store forever.
        archive = obs.ObservabilityStore(db_path).archive_to(
            str(tmp_path / "allowed.sqlite3")
        )
        assert archive["sealed"] is True


@pytest.fixture
def restore_env_vars():
    """`fastworkflow.init` REPLACES the process env dict (`__init__.py:253`).

    A test that calls it and does not put the old dict back leaves every later
    test in the process running under whatever two keys it happened to set.
    """
    saved = dict(fastworkflow._env_vars)
    yield
    fastworkflow._env_vars.clear()
    fastworkflow._env_vars.update(saved)


class TestDeterminismSeams:
    def test_the_lm_cache_can_be_defeated_process_wide(
        self, monkeypatch, restore_env_vars
    ):
        """`[XR16]`: without this, k attempts return byte-identical trajectories."""
        import dspy

        from fastworkflow.utils import dspy_utils

        captured = {}

        class FakeLM:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr(dspy, "LM", FakeLM)
        fastworkflow.init({"LLM_AGENT": "test/model"})
        dspy_utils.get_lm("LLM_AGENT")
        assert "cache" not in captured  # unset leaves today's behaviour alone

        captured.clear()
        fastworkflow.init({"LLM_AGENT": "test/model", "FW_LM_CACHE": "0"})
        dspy_utils.get_lm("LLM_AGENT")
        assert captured["cache"] is False

    def test_the_utterance_cache_can_be_sharded_per_attempt(
        self, tmp_path, restore_env_vars
    ):
        """`[XR16]`: `___convo_info/cache.sqlite3` is keyed on nothing today.

        Attempt 2 would otherwise inherit attempt 1's disambiguation decisions —
        correlated attempts, which is exactly what pass^k must not have — and a
        treatment arm would inherit the baseline arm's, both running against the
        same workflow folder.
        """
        from fastworkflow._workflows.command_metadata_extraction.intent_detection import (
            CommandNamePrediction,
        )

        convo = str(tmp_path / "convo")
        fastworkflow.init({})
        shared = CommandNamePrediction._get_cache_path_cache(convo, "wid-a")
        assert Path(shared).name == "cache.sqlite3"

        fastworkflow.init({"FW_UTTERANCE_CACHE_SCOPE": "workflow"})
        a = CommandNamePrediction._get_cache_path_cache(convo, "wid-a")
        b = CommandNamePrediction._get_cache_path_cache(convo, "wid-b")
        assert a != b
        assert Path(a).name == "cache-wid-a.sqlite3"


class TestDerivedOutcomeIsNotAScore:
    def test_the_fallback_is_named_for_what_it_measures(self):
        """`[XR13]`: it reports a command-failure signal, not "the task passed"."""
        from fastworkflow.experiment.runner import AttemptRun

        task = ExperimentTask(task_id="t0")
        completed = type("S", (), {"value": "completed"})()
        ok = type("O", (), {"status": completed, "success": True})()
        run = AttemptRun(task=task, attempt=1, channel_id="c", conversation_id=1,
                         turn_outputs=[ok])
        outcome, source, _, detail = derived_outcome(run)
        assert outcome == "pass"
        assert source == "derived"
        assert detail["predicate"] == "no_command_reported_failure"

    def test_an_unanswered_suspension_is_incomplete_not_a_fail(self):
        from fastworkflow.experiment.runner import AttemptRun

        awaiting = type("S", (), {"value": "awaiting_user"})()
        suspended = type("O", (), {"status": awaiting, "success": False})()
        run = AttemptRun(
            task=ExperimentTask(task_id="t0"), attempt=1, channel_id="c",
            conversation_id=1, turn_outputs=[suspended],
        )
        outcome, _, _, detail = derived_outcome(run)
        assert outcome == "incomplete"
        assert detail["reason"] == "awaiting_user"


# ----------------------------------------------------------------------
# `[XR15]`: erasure and retention
# ----------------------------------------------------------------------


class TestErasure:
    def _seeded(self, store):
        """Two attempts on TWO channels.

        One channel would make `DELETE ... WHERE channel_id=?` and
        `DELETE ... WHERE experiment_id IN (touched)` indistinguishable — the
        erasure test would pass against an implementation that wiped the whole
        experiment's attempt rows when one of its channels was forgotten.
        """
        store.create_experiment("exp-1", "L", declared_tasks=1, declared_attempts=2)
        channels = []
        for n in (1, 2):
            channel = channel_for("exp-1", "t0", n)
            channels.append(channel)
            store.start_attempt("exp-1", "t0", n, channel)
            conv = store.mint_conversation_id(
                channel, experiment_id="exp-1", task_id="t0", attempt=n
            )
            _write_turn(
                store,
                _turn_row(f"tk{n}", channel, conversation_id=conv,
                          experiment_id="exp-1", task_id="t0", attempt=n),
            )
            store.finish_attempt(
                "exp-1", "t0", n, outcome="pass", outcome_source="g"
            )
        assert store.complete_experiment("exp-1") == "complete"
        return channels

    def test_forget_channel_invalidates_rather_than_deleting_the_container(
        self, store
    ):
        """`[XR15]`: an erased experiment must never still render a score."""
        channels = self._seeded(store)
        assert store.experiment_scores("exp-1")["reportable"] is True

        deleted = store.forget_channel(channels[0])
        assert deleted["experiments_invalidated"] == 1
        # ONE attempt row, not both: erasure is channel-scoped. The surviving
        # attempt lives in another channel and is not the erased channel's to
        # delete — only the container's verdict is.
        assert deleted["experiment_attempts"] == 1
        survivors = store.experiment_attempt_rows("exp-1")
        assert [r["attempt"] for r in survivors] == [2]
        assert len(store.list_turns(experiment_id="exp-1")) == 1

        experiment = store.get_experiment("exp-1")
        assert experiment is not None  # not deleted: it is not the channel's to delete
        assert experiment["status"] == "invalid"
        assert experiment["invalid_reason"] == "turns_erased"
        assert store.experiment_scores("exp-1")["reportable"] is False

    def test_clear_conversations_removes_the_container_entirely(self, store):
        self._seeded(store)
        deleted = store.clear_conversations()
        assert deleted["experiments"] == 1
        assert deleted["experiment_attempts"] == 2
        assert store.get_experiment("exp-1") is None

    def test_prune_leaves_the_experiment_tables_alone(self, store):
        """Turns and conversations are already retention-exempt (`[R16]`); a
        container pruned out from under attempts that still exist would be the
        orphan shape `[DR44]` prevents."""
        self._seeded(store)
        store.prune(retention_days=0, max_bytes=1)
        assert store.get_experiment("exp-1") is not None
        assert store.experiment_attempt_rows("exp-1")


# ----------------------------------------------------------------------
# `[XR9]`: the read API, over real HTTP
# ----------------------------------------------------------------------


def _request(server, path, method="GET", body=None, token=...):
    if token is ...:
        token = server.token
    url = f"http://127.0.0.1:{server.port}{path}"
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, method=method, data=data)
    if token is not None:
        req.add_header("Authorization", f"Bearer {token}")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as err:
        raw = err.read()
        try:
            return err.code, json.loads(raw or b"{}")
        except ValueError:
            return err.code, {"raw": raw}


@pytest.fixture
def experiment_server(workflow_path, db_path):
    store = obs.ObservabilityStore(db_path)
    _seed(store, "exp-a", tasks=2, attempts=2,
          outcomes=lambda t, n: "fail" if (t, n) == ("t1", 2) else "pass")
    channel = channel_for("exp-a", "t0", 1)
    conv = store.mint_conversation_id(
        channel, experiment_id="exp-a", task_id="t0", attempt=1
    )
    _write_turn(
        store,
        _turn_row("tk-exp", channel, conversation_id=conv,
                  experiment_id="exp-a", task_id="t0", attempt=1),
    )
    _write_turn(store, _turn_row("tk-plain", "chatbot"))
    store.record_evidence_segment(
        "exp-a", 1, "evr-x", {"valid": True, "problems": []}
    )
    assert store.complete_experiment("exp-a") == "complete"

    srv = run_chatbot_server.ChatbotServer(db_path, workflow_path=workflow_path, port=0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    thread.join(timeout=5)


class TestReadApi:
    def test_list_and_detail(self, experiment_server):
        status, data = _request(experiment_server, "/api/experiments")
        assert status == 200
        assert [e["experiment_id"] for e in data["experiments"]] == ["exp-a"]
        assert data["experiments"][0]["attempts_finished"] == 4

        status, data = _request(experiment_server, "/api/experiment/exp-a")
        assert status == 200
        assert data["experiment"]["status"] == "complete"
        assert data["experiment"]["evidence_runs"][0]["evidence_run_id"] == "evr-x"

    def test_tasks_attempts_and_score(self, experiment_server):
        _, data = _request(experiment_server, "/api/experiment/exp-a/tasks")
        assert {t["task_id"] for t in data["tasks"]} == {"t0", "t1"}

        _, data = _request(
            experiment_server, "/api/experiment/exp-a/attempts?task=t1"
        )
        assert [a["outcome"] for a in data["attempts"]] == ["pass", "fail"]

        _, data = _request(experiment_server, "/api/experiment/exp-a/score")
        assert data["score"]["reportable"] is True
        assert data["score"]["pass_at_1"] == pytest.approx(3 / 4)
        assert data["score"]["pass_at_k"] == pytest.approx(1 / 2)

    def test_turns_filter_by_experiment_task_and_attempt(self, experiment_server):
        _, data = _request(experiment_server, "/api/turns?experiment=exp-a")
        assert [t["turn_key"] for t in data["turns"]] == ["tk-exp"]
        _, data = _request(
            experiment_server, "/api/turns?experiment=exp-a&task=t0&attempt=1"
        )
        assert len(data["turns"]) == 1
        _, data = _request(experiment_server, "/api/turns?experiment=exp-a&attempt=99")
        assert data["turns"] == []
        # the ordinary turn is still served when nothing is filtered
        _, data = _request(experiment_server, "/api/turns")
        assert "tk-plain" in {t["turn_key"] for t in data["turns"]}

    def test_an_unknown_experiment_is_404(self, experiment_server):
        status, _ = _request(experiment_server, "/api/experiment/exp-nope")
        assert status == 404

    def test_compare_refuses_an_incomparable_pair_with_409_and_a_reason(
        self, experiment_server, db_path
    ):
        """`[XR19]`: the refusal carries the reason, and the SPA renders it.

        409-with-a-body rather than 200-with-a-flag, because a client that
        renders whatever it got would render a comparison of two runs that
        share no task.
        """
        store = obs.ObservabilityStore(db_path)
        # A genuinely incomparable arm: same 2x2 shape, disjoint task set.
        store.create_experiment("exp-b", "B", declared_tasks=2, declared_attempts=2)
        for task_id in ("z0", "z1"):
            for n in (1, 2):
                store.start_attempt("exp-b", task_id, n, f"c:{task_id}:{n}")
                store.finish_attempt(
                    "exp-b", task_id, n, outcome="pass", outcome_source="g"
                )
        assert store.complete_experiment("exp-b") == "complete"

        status, data = _request(
            experiment_server, "/api/experiment/exp-b/compare?baseline=exp-a"
        )
        assert status == 409
        assert data["comparable"] is False
        assert any("task sets differ" in p for p in data["problems"])
        assert set(data["only_in_treatment"]) == {"z0", "z1"}

    def test_compare_of_a_comparable_pair_is_200(self, experiment_server):
        status, data = _request(
            experiment_server, "/api/experiment/exp-a/compare?baseline=exp-a"
        )
        assert status == 200 and data["comparable"] is True

    def test_compare_error_cases(self, experiment_server):
        status, _ = _request(
            experiment_server, "/api/experiment/exp-a/compare?baseline=exp-missing"
        )
        assert status == 404
        status, _ = _request(experiment_server, "/api/experiment/exp-a/compare")
        assert status == 400  # no baseline set and none supplied

    def test_patch_updates_notes(self, experiment_server):
        status, data = _request(
            experiment_server, "/api/experiment/exp-a", method="PATCH",
            body={"notes": "reviewed 2026-08-29"},
        )
        assert status == 200
        assert data["experiment"]["notes"] == "reviewed 2026-08-29"

    def test_patch_refuses_a_body_with_nothing_to_patch(self, experiment_server):
        status, data = _request(
            experiment_server, "/api/experiment/exp-a", method="PATCH",
            body={"description": "rewritten"},
        )
        assert status == 400
        assert "notes" in data["error"]

    def test_patch_is_gated_like_every_other_verb(self, experiment_server):
        """The gates are applied per verb method, with no shared chokepoint."""
        status, _ = _request(
            experiment_server, "/api/experiment/exp-a", method="PATCH",
            body={"notes": "x"}, token="wrong-token",
        )
        assert status == 401
        # ...and the Host allowlist, the other gate do_PATCH has to repeat
        # because the server applies them per verb method with no chokepoint.
        url = f"http://127.0.0.1:{experiment_server.port}/api/experiment/exp-a"
        req = urllib.request.Request(
            url, method="PATCH", data=json.dumps({"notes": "x"}).encode()
        )
        req.add_header("Authorization", f"Bearer {experiment_server.token}")
        req.add_header("Content-Type", "application/json")
        req.add_header("Host", "evil.example.com")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                status = resp.status
        except urllib.error.HTTPError as err:
            status = err.code
        assert status == 403
        status, _ = _request(
            experiment_server, "/api/turn/tk-exp", method="PATCH", body={"notes": "x"}
        )
        assert status == 405  # PATCH is admitted only under /api/experiment/
        # ...and only at the collection root: a sub-path must not be a silent
        # alias for the notes route the way split-and-discard would make it.
        status, _ = _request(
            experiment_server, "/api/experiment/exp-a/score", method="PATCH",
            body={"notes": "x"},
        )
        assert status == 404

    def test_experiment_scoped_aggregate_returns_only_that_experiment(
        self, experiment_server, db_path
    ):
        """The boundary `fix-9eg.3` currently lacks.

        Its aggregates run over whatever turns are in the DB; scoped by
        experiment_id they must exclude the ordinary chatbot turn entirely.
        """
        # Through the route, not through raw SQL: the boundary fix-9eg.3 needs
        # is the one an aggregate view will actually call.
        _, scoped = _request(experiment_server, "/api/turns?experiment=exp-a")
        _, everything = _request(experiment_server, "/api/turns")
        assert [t["turn_key"] for t in scoped["turns"]] == ["tk-exp"]
        assert {t["turn_key"] for t in everything["turns"]} == {"tk-exp", "tk-plain"}
        # The ordinary chatbot turn is excluded because it carries no label at
        # all, not because it happens to sort out of the page.
        plain = [t for t in everything["turns"] if t["turn_key"] == "tk-plain"][0]
        assert plain["experiment_id"] is None


class TestCrossLink:
    """`[XR9]`: reachable from both directions, with one "run" concept."""

    @pytest.mark.skip(
        reason="distillation is intentionally excluded from fix-rj2.11"
    )
    def test_distillation_runs_can_be_filtered_by_experiment(self, store):
        store.create_experiment("exp-1", "L", declared_tasks=1, declared_attempts=1)
        channel = channel_for("exp-1", "t0", 1)
        conv = store.mint_conversation_id(
            channel, experiment_id="exp-1", task_id="t0", attempt=1
        )
        _write_turn(
            store,
            _turn_row("tk-in", channel, conversation_id=conv,
                      experiment_id="exp-1", task_id="t0", attempt=1),
        )
        _write_turn(store, _turn_row("tk-out", "other"))
        for run_id, turn_key in (("run-in", "tk-in"), ("run-out", "tk-out")):
            with store._connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                store.upsert_distillation_row(
                    conn, "run",
                    {"run_id": run_id, "turn_key": turn_key, "user_message": "m",
                     "comparable": 1, "run_json": "{}"},
                    store._store_redactor(),
                )
                conn.commit()
        scoped = store.list_distillation_runs(experiment_id="exp-1")
        assert [r["run_id"] for r in scoped] == ["run-in"]
        assert len(store.list_distillation_runs()) == 2

    def test_a_turn_resolves_back_to_its_experiment(self, store):
        store.create_experiment("exp-1", "L", declared_tasks=1, declared_attempts=1)
        channel = channel_for("exp-1", "t0", 1)
        conv = store.mint_conversation_id(
            channel, experiment_id="exp-1", task_id="t0", attempt=1
        )
        _write_turn(
            store,
            _turn_row("tk-in", channel, conversation_id=conv,
                      experiment_id="exp-1", task_id="t0", attempt=1),
        )
        _write_turn(store, _turn_row("tk-out", "other"))
        labels = store.experiment_labels_for_turn("tk-in")
        assert labels["experiment_id"] == "exp-1"
        assert labels["task_id"] == "t0" and labels["attempt"] == 1
        assert labels["description"] == "L"
        assert store.experiment_labels_for_turn("tk-out") is None


class TestColdStart:
    def test_experiments_serves_an_empty_state_before_the_db_exists(
        self, workflow_path, tmp_path
    ):
        missing = str(tmp_path / "nope" / "observability.sqlite3")
        srv = run_chatbot_server.ChatbotServer(
            missing, workflow_path=workflow_path, port=0
        )
        thread = threading.Thread(target=srv.serve_forever, daemon=True)
        thread.start()
        try:
            status, data = _request(srv, "/api/experiments")
            assert status == 200
            assert data == {"experiments": []}
        finally:
            srv.shutdown()
            thread.join(timeout=5)


class TestSpaSurface:
    def test_the_experiment_browser_ships_and_obeys_the_page_rules(self):
        page = run_chatbot_server.load_index_html()
        assert b'id="navBenchmarks"' in page
        assert b"showExperiments" in page
        assert b"showExperimentTask" in page
        assert b"openExperimentAttempt" in page
        # The routes it calls, not just the functions it defines: a page that
        # defined every function and called the wrong path would pass otherwise.
        assert b"/api/experiments?limit=" in page
        assert b'"/api/experiment/"' in page
        assert b'"/attempts?task="' in page
        assert b'"/compare"' in page
        assert b"/api/turns?experiment=" in page
        # An invalid experiment must be visually unmistakable.
        assert b"expInvalid" in page
        # ...and a derived verdict must be labelled as not-a-judgement (§8.1).
        assert b"derived, not graded" in page
        assert b"not a judgement that the task was accomplished" in page
        # An experiment is named by its id: its label is the author's optional
        # description, which may be a paragraph or may be absent entirely.
        assert b'"Experiment \xc2\xb7 " + experimentId.slice(-8)' in page
        # [R22] and the packaging rules still hold.
        assert b"innerHTML" not in page
        assert b"https://" not in page


# ----------------------------------------------------------------------
# fix-qe2: the binding server's runtime snapshot on the attempt row
# ----------------------------------------------------------------------


class TestRuntimeSnapshotStamp:
    """Stamped at claim, read back decoded, null when the server had none."""

    @staticmethod
    def _controller(db_path):
        from fastworkflow.experiment.runner import ExperimentController

        store = obs.ObservabilityStore(db_path)
        controller = ExperimentController(
            db_path, store.store_identity(), migrate=False, external=True
        )
        controller.create_experiment(
            "exp-stamp",
            "runtime snapshot stamp",
            declared_tasks=1,
            declared_attempts=2,
            declarations=[("task-1", 1, "job-1"), ("task-1", 2, "job-2")],
        )
        return controller

    @staticmethod
    def _bootstrap(controller, attempt):
        return controller.register_attempt(
            "exp-stamp",
            "task-1",
            attempt,
            f"job-{attempt}",
            f"registered:task-1:{attempt}",
        )

    def test_a_claimed_attempt_carries_its_servers_snapshot_readable_back(
        self, db_path
    ):
        controller = self._controller(db_path)
        snapshot = {
            "configuration_valid": True,
            "effective_features": {"decision_signals_v1": "shadow"},
            "workflow_fingerprint": "sha256:abc",
            "capture_profile": "debug",
            "pid": 4242,
        }

        controller.claim_attempt(
            self._bootstrap(controller, 1),
            server_incarnation="server-a",
            runtime_snapshot=snapshot,
        )

        row = controller.store.experiment_attempt_rows("exp-stamp")[0]
        assert row["runtime_snapshot"] == snapshot
        assert "runtime_snapshot_json" not in row
        with sqlite3.connect(db_path) as conn:
            raw = conn.execute(
                "SELECT runtime_snapshot_json FROM experiment_attempts "
                "WHERE experiment_id='exp-stamp' AND attempt=1"
            ).fetchone()[0]
        assert json.loads(raw) == snapshot
        # start_attempt on the same row (the WEC's first turn) does not
        # clobber the stamp: it belongs to the binding, not to the turn.
        controller.store.start_attempt(
            "exp-stamp", "task-1", 1, "registered:task-1:1", source_key="job-1"
        )
        assert controller.store.experiment_attempt_rows("exp-stamp")[0][
            "runtime_snapshot"
        ] == snapshot

    def test_an_attempt_bound_without_a_snapshot_reads_back_null(self, db_path):
        controller = self._controller(db_path)

        controller.claim_attempt(
            self._bootstrap(controller, 2), server_incarnation="server-b"
        )

        rows = controller.store.experiment_attempt_rows("exp-stamp", task_id="task-1")
        row = next(r for r in rows if r["attempt"] == 2)
        assert row["runtime_snapshot"] is None
        assert "runtime_snapshot_json" not in row

    def test_the_chatbot_attempts_api_exposes_the_decoded_stamp(
        self, experiment_server, db_path
    ):
        """The UI (fix-49m.6) reads `runtime_snapshot` off the attempt rows the
        chatbot server already returns; rows that never bound read null."""
        controller = self._controller(db_path)
        snapshot = {"configuration_valid": True, "pid": 7, "effective_features": {}}
        controller.claim_attempt(
            self._bootstrap(controller, 1),
            server_incarnation="server-api",
            runtime_snapshot=snapshot,
        )

        status, data = _request(experiment_server, "/api/experiment/exp-stamp/attempts")
        assert status == 200
        assert data["attempts"][0]["runtime_snapshot"] == snapshot
        assert "runtime_snapshot_json" not in data["attempts"][0]

        _, data = _request(experiment_server, "/api/experiment/exp-a/attempts?task=t1")
        assert [a["runtime_snapshot"] for a in data["attempts"]] == [None, None]

    def test_a_v2_store_is_replaced_on_open_not_migrated(self, db_path):
        """The column is create-time only; a populated v2 store is recreated
        rather than altered, so the column exists afterwards."""
        obs.ObservabilityStore(db_path)
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA user_version = 2")
        conn.commit()
        conn.close()

        obs.ObservabilityStore(db_path)
        conn = sqlite3.connect(db_path)
        try:
            assert conn.execute("PRAGMA user_version").fetchone()[0] == 6
            assert "runtime_snapshot_json" in {
                r[1] for r in conn.execute("PRAGMA table_info(experiment_attempts)")
            }
        finally:
            conn.close()

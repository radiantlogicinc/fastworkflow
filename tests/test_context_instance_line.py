"""The alias line names the context instance a command ran in.

Offline only. Nothing here starts a server, calls a model or touches a backend:
the whole change is a presentation line and a turn-scoped lookup, and both are
decidable from a trajectory and a fixture context class.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from fastworkflow.context_identity import (
    context_clause_for,
    context_identity,
    declared_instance_label,
)
from fastworkflow.observation_offloading.archive import (
    RuntimeHandleArchive,
    RuntimeHandleScope,
)
from fastworkflow.observation_offloading.compact import (
    annotate_execute_observations,
    compact_trajectory,
)
from fastworkflow.observation_offloading.continuation import replan_trajectory_skeleton
from fastworkflow.observation_offloading.labels import (
    MAX_INSTANCE_LABEL_CHARS,
    alias_line,
    context_clause,
    is_offload_label,
    label_alias,
    offload_label,
    printed_alias,
    printed_context,
    strip_alias_line,
)
from fastworkflow.observation_offloading.state import (
    context_clause_of,
    record_context_clause,
    reset_runtime_state,
    snapshot_events,
)


# --------------------------------------------------------------------------
# Fixture context callback classes: exactly what a workflow may declare.
# --------------------------------------------------------------------------

class AccountContext:
    """Declares the hook as a classmethod (the IDO form)."""

    @classmethod
    def instance_label(cls, command_context_object):
        obj = command_context_object
        uid = getattr(obj, "uid", None) or ""
        label = getattr(obj, "label", None) or ""
        if uid and label:
            return f"{uid} ({label})"
        return str(uid or label or "")


class GroupContext:
    """Declares the attribute form instead of a method."""

    instance_label_attr = "uid"


class DirectoryExplorerContext:
    """A workspace: navigable, but not an instance of anything."""


class RaisingContext:
    @classmethod
    def instance_label(cls, command_context_object):
        raise RuntimeError("a workflow's hook may be broken")


class FakeWorkflow:
    """The two attributes ``context_identity`` reads, and nothing else."""

    def __init__(self, name, obj, *, root=False):
        self._name = name
        self._obj = obj
        self._root = root

    @property
    def is_current_command_context_root(self):
        return self._root

    @property
    def current_command_context_name(self):
        return self._name

    @property
    def current_command_context(self):
        return self._obj


def _entity(uid=None, label=None):
    return SimpleNamespace(uid=uid, label=label)


class ContextClauseFormat(unittest.TestCase):
    """The printed line, with and without an instance."""

    def test_root_prints_exactly_the_a1_line(self) -> None:
        self.assertEqual(alias_line("O42"),
                         "Observation O42 (execute_workflow_query)\n")
        self.assertEqual(alias_line("O42", ""), alias_line("O42"))
        self.assertIsNone(printed_context(alias_line("O42")))
        self.assertEqual(printed_alias(alias_line("O42")), "O42")

    def test_context_with_an_instance(self) -> None:
        line = alias_line("O42", context_clause("Account", "28c5aeb5 (Alan Cooper)"))
        self.assertEqual(
            line,
            "Observation O42 (execute_workflow_query, in Account 28c5aeb5 Alan Cooper)\n")
        self.assertEqual(printed_alias(line), "O42")
        self.assertEqual(printed_context(line), "Account 28c5aeb5 Alan Cooper")

    def test_context_without_an_instance_prints_the_name_alone(self) -> None:
        line = alias_line("O7", context_clause("DirectoryExplorer", ""))
        self.assertEqual(
            line, "Observation O7 (execute_workflow_query, in DirectoryExplorer)\n")
        self.assertEqual(printed_context(line), "DirectoryExplorer")

    def test_no_context_name_means_no_clause(self) -> None:
        self.assertEqual(context_clause("", "28c5aeb5"), "")
        self.assertEqual(alias_line("O1", context_clause("", "28c5aeb5")),
                         alias_line("O1"))

    def test_parentheses_and_newlines_can_never_reach_the_line(self) -> None:
        clause = context_clause("Account", "28c5 (Alan\nCooper)\n)")
        self.assertNotIn("(", clause)
        self.assertNotIn(")", clause)
        self.assertNotIn("\n", clause)
        line = alias_line("O3", clause)
        self.assertEqual(printed_alias(line), "O3")
        self.assertEqual(printed_context(line), clause)
        self.assertEqual(strip_alias_line(line + "body"), "body")

    def test_the_label_is_capped(self) -> None:
        clause = context_clause("Account", "u" * 500)
        label = clause.split(" ", 1)[1]
        self.assertLessEqual(len(label), MAX_INSTANCE_LABEL_CHARS)
        self.assertTrue(label.endswith("..."))

    def test_the_line_is_stripped_and_digests_are_unchanged(self) -> None:
        body = "permission_uid  label\n85cde168  Active Directory_Cloud Administrator\n"
        for clause in ("", "Account 28c5aeb5 Alan Cooper", "DirectoryExplorer"):
            shown = alias_line("O9", clause) + body
            self.assertEqual(strip_alias_line(shown), body)
            self.assertEqual(
                hashlib.sha256(strip_alias_line(shown).encode()).hexdigest(),
                hashlib.sha256(body.encode()).hexdigest())

    def test_an_a1_line_recorded_before_this_change_still_reads(self) -> None:
        legacy = "Observation O12 (execute_workflow_query)\n477 holder(s).\n"
        self.assertEqual(printed_alias(legacy), "O12")
        self.assertIsNone(printed_context(legacy))
        self.assertEqual(strip_alias_line(legacy), "477 holder(s).\n")

    def test_an_offload_label_is_not_an_alias_line(self) -> None:
        label = offload_label(alias="O5", command_name="Account/list_permissions",
                              response="rows")
        self.assertTrue(is_offload_label(label))
        self.assertIsNone(printed_alias(label))


class DeclaredInstanceIdentity(unittest.TestCase):
    """The generic hook on the context callback class."""

    def test_classmethod_hook(self) -> None:
        self.assertEqual(
            declared_instance_label(AccountContext, _entity("28c5aeb5", "Alan Cooper")),
            "28c5aeb5 (Alan Cooper)")

    def test_attribute_hook(self) -> None:
        self.assertEqual(
            declared_instance_label(GroupContext, _entity("g-1")), "g-1")

    def test_no_declaration_yields_nothing(self) -> None:
        self.assertEqual(
            declared_instance_label(DirectoryExplorerContext, _entity("x")), "")

    def test_a_broken_hook_yields_nothing_rather_than_raising(self) -> None:
        self.assertEqual(declared_instance_label(RaisingContext, _entity("x")), "")

    def test_an_absent_identity_is_never_invented(self) -> None:
        # The object carries no uid and no label: the context name stands alone.
        self.assertEqual(declared_instance_label(AccountContext, _entity()), "")
        workflow = FakeWorkflow("Account", _entity())
        self.assertEqual(context_identity(workflow), ("Account", ""))
        self.assertEqual(context_clause_for(workflow), "Account")

    def test_root_context_has_no_identity(self) -> None:
        workflow = FakeWorkflow("IDO", _entity("root"), root=True)
        self.assertEqual(context_identity(workflow), ("", ""))
        self.assertEqual(context_clause_for(workflow), "")

    def test_no_workflow_is_not_an_error(self) -> None:
        self.assertEqual(context_identity(None), ("", ""))
        self.assertEqual(context_clause_for(None), "")


class AnnotationUsesTheRecordedContext(unittest.TestCase):
    """What `annotate_execute_observations` prints, and what it leaves alone."""

    def setUp(self) -> None:
        reset_runtime_state()
        self.addCleanup(reset_runtime_state)
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.archive = RuntimeHandleArchive(
            str(Path(self.tempdir.name) / "handles.sqlite3"))
        self.scope = RuntimeHandleScope(
            store_identity="fixture-store", channel_id="fixture-channel",
            experiment_id="fixture-experiment", task_id="fixture-task",
            attempt=1, turn_key="fixture-turn")

    def compact(self, trajectory, **kwargs):
        return compact_trajectory(trajectory, scope=self.scope,
                                  selected_archive=self.archive, **kwargs)

    @staticmethod
    def _step(trajectory, index, tool, observation, command=None):
        trajectory[f"thought_{index}"] = f"think-{index}"
        trajectory[f"tool_name_{index}"] = tool
        if command is not None:
            trajectory[f"tool_args_{index}"] = {"command": command}
        trajectory[f"observation_{index}"] = observation

    def test_a_command_that_moves_the_context_prints_where_it_RAN(self) -> None:
        """The rule: context BEFORE the command, not the one it entered.

        `open_account_by_uid` runs in DirectoryExplorer and ends in Account. It
        must read as the DirectoryExplorer command it is; the `list_permissions`
        that follows it is the one that belongs to the account.
        """
        trajectory: dict = {}
        self._step(trajectory, 0, "execute_workflow_query", "Entered Account context.",
                   command="open_account_by_uid <account_uid>28c5aeb5</account_uid>")
        self._step(trajectory, 1, "execute_workflow_query", "permission rows",
                   command="list_permissions")
        # Recorded at dispatch: O1 ran in DirectoryExplorer, O2 in the account.
        record_context_clause(self.scope, "O1", context_clause("DirectoryExplorer", ""))
        record_context_clause(self.scope, "O2",
                              context_clause("Account", "28c5aeb5 (Alan Cooper)"))
        self.compact(trajectory)
        self.assertEqual(printed_context(trajectory["observation_0"]),
                         "DirectoryExplorer")
        self.assertEqual(printed_context(trajectory["observation_1"]),
                         "Account 28c5aeb5 Alan Cooper")

    def test_root_and_unrecorded_steps_print_the_plain_line(self) -> None:
        trajectory: dict = {}
        self._step(trajectory, 0, "execute_workflow_query", "root rows", command="find_identity")
        self._step(trajectory, 1, "execute_workflow_query", "other rows", command="whatever")
        record_context_clause(self.scope, "O1", "")   # ran at the root
        # O2 was never recorded at all (an older recording, or a capture that failed).
        self.compact(trajectory)
        self.assertEqual(trajectory["observation_0"], alias_line("O1") + "root rows")
        self.assertEqual(trajectory["observation_1"], alias_line("O2") + "other rows")
        events = {e["alias"]: e for e in snapshot_events() if e["kind"] == "context_line"}
        self.assertEqual(events["O1"]["context_recorded"], True)
        self.assertEqual(events["O2"]["context_recorded"], False)
        self.assertEqual(events["O1"]["clause_utf8_bytes"], 0)

    def test_the_archive_stores_the_response_without_the_line(self) -> None:
        body = "permission_uid  label\n85cde168  Active Directory_Cloud Administrator\n"
        trajectory: dict = {}
        self._step(trajectory, 0, "execute_workflow_query", body, command="list_permissions")
        record_context_clause(self.scope, "O1",
                              context_clause("Account", "28c5aeb5 (Alan Cooper)"))
        self.compact(trajectory)
        self.assertIn("in Account 28c5aeb5 Alan Cooper", trajectory["observation_0"])
        stored = self.archive.get(self.scope, "O1")
        self.assertEqual(stored["text"], body)
        self.assertEqual(stored["text_sha256"],
                         hashlib.sha256(body.encode()).hexdigest())

    def test_an_offload_label_is_unchanged_by_the_clause(self) -> None:
        """The label describes the RESPONSE, so the clause cannot reach it.

        Six executes, because the five most recent are protected from
        offloading; the oldest is the one that gets a label.
        """
        body = "holder rows\n" + "x" * 9_000
        for clause in ("", context_clause("Account", "28c5aeb5 (Alan Cooper)")):
            reset_runtime_state()
            trajectory: dict = {}
            self._step(trajectory, 0, "execute_workflow_query", body,
                       command="list_permissions")
            for index in range(1, 6):
                self._step(trajectory, index, "execute_workflow_query",
                           f"small-{index}", command="show_rights")
            for ordinal in range(1, 7):
                record_context_clause(self.scope, f"O{ordinal}", clause)
            self.compact(trajectory, packed_target_bytes=500)
            self.assertTrue(is_offload_label(trajectory["observation_0"]))
            self.assertEqual(label_alias(trajectory["observation_0"]), "O1")
            self.assertEqual(
                trajectory["observation_0"],
                offload_label(alias="O1", command_name="list_permissions",
                              response=body))

    def test_the_replan_skeleton_is_unchanged_by_the_clause(self) -> None:
        skeletons = []
        for clause in ("", context_clause("Account", "28c5aeb5 (Alan Cooper)")):
            reset_runtime_state()
            trajectory: dict = {}
            self._step(trajectory, 0, "execute_workflow_query",
                       "holder rows\n" + "x" * 9_000, command="show_holders")
            self._step(trajectory, 1, "execute_workflow_query", "small rows",
                       command="show_rights")
            record_context_clause(self.scope, "O1", clause)
            record_context_clause(self.scope, "O2", clause)
            self.compact(trajectory)
            skeleton, metadata = replan_trajectory_skeleton(
                trajectory, greedy_max_bytes=1_000,
                scope=self.scope, selected_archive=self.archive)
            self.assertEqual(metadata["labeled_aliases"], ["O1"])
            skeletons.append(skeleton)
        # The labelled observation is identical; the inlined one differs only by
        # the clause, and strips back to the same bytes.
        self.assertEqual(skeletons[0]["observation_0"], skeletons[1]["observation_0"])
        self.assertEqual(strip_alias_line(skeletons[0]["observation_1"]),
                         strip_alias_line(skeletons[1]["observation_1"]))

    def test_the_clause_is_never_rewritten_once_printed(self) -> None:
        trajectory: dict = {}
        self._step(trajectory, 0, "execute_workflow_query", "rows", command="list_permissions")
        record_context_clause(self.scope, "O1", context_clause("Account", "28c5aeb5"))
        self.compact(trajectory)
        frozen = trajectory["observation_0"]
        record_context_clause(self.scope, "O1", context_clause("Group", "g-9"))
        self.compact(trajectory)
        self.assertEqual(trajectory["observation_0"], frozen)

    def test_the_line_grows_the_packed_trajectory_by_exactly_the_clause(self) -> None:
        body = "rows\n"
        plain: dict = {}
        self._step(plain, 0, "execute_workflow_query", body, command="list_permissions")
        record_context_clause(self.scope, "O1", "")
        self.compact(plain)
        reset_runtime_state()
        clause = context_clause("Account", "28c5aeb5 (Alan Cooper)")
        with_clause: dict = {}
        self._step(with_clause, 0, "execute_workflow_query", body, command="list_permissions")
        record_context_clause(self.scope, "O1", clause)
        self.compact(with_clause)
        grown = (len(json.dumps(with_clause, ensure_ascii=False).encode())
                 - len(json.dumps(plain, ensure_ascii=False).encode()))
        self.assertEqual(grown, len(f", in {clause}".encode()))

    def test_the_context_line_event_carries_the_measures(self) -> None:
        trajectory: dict = {}
        self._step(trajectory, 0, "execute_workflow_query", "rows", command="list_permissions")
        clause = context_clause("Account", "28c5aeb5 (Alan Cooper)")
        record_context_clause(self.scope, "O1", clause)
        self.compact(trajectory)
        [event] = [e for e in snapshot_events() if e["kind"] == "context_line"]
        self.assertEqual(event["alias"], "O1")
        self.assertEqual(event["context"], clause)
        self.assertTrue(event["has_instance"])
        self.assertEqual(event["clause_utf8_bytes"], len(f", in {clause}".encode()))
        self.assertEqual(event["line_utf8_bytes"],
                         len(alias_line("O1", clause).encode()))

    def test_a_workspace_context_records_no_instance(self) -> None:
        trajectory: dict = {}
        self._step(trajectory, 0, "execute_workflow_query", "rows", command="list_accounts")
        record_context_clause(self.scope, "O1", context_clause("DirectoryExplorer", ""))
        self.compact(trajectory)
        [event] = [e for e in snapshot_events() if e["kind"] == "context_line"]
        self.assertFalse(event["has_instance"])
        self.assertEqual(event["context"], "DirectoryExplorer")

    def test_the_registry_does_not_outlive_the_turn(self) -> None:
        record_context_clause(self.scope, "O1", "Account 28c5aeb5")
        self.assertEqual(context_clause_of(self.scope, "O1"), "Account 28c5aeb5")
        reset_runtime_state()
        self.assertIsNone(context_clause_of(self.scope, "O1"))


if __name__ == "__main__":
    unittest.main()

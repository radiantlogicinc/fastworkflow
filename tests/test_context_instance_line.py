"""The alias line names the context instance a command ran in.

Offline only. Nothing here starts a server, calls a model or touches a backend:
the whole change is a presentation line and a turn-scoped lookup, and both are
decidable from a trajectory and a fixture context class.
"""
from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastworkflow.context_identity import (
    context_clause_for,
    context_identity,
    declared_instance_label,
)
from fastworkflow.observation_offloading.archive import (
    RuntimeHandleArchive,
    RuntimeHandleScope,
)
from fastworkflow.observation_offloading.compact import compact_trajectory
from fastworkflow.observation_offloading.labels import (
    MAX_INSTANCE_LABEL_CHARS,
    alias_line,
    context_clause,
    is_offload_label,
    offload_label,
    printed_alias,
    printed_context,
    strip_alias_line,
)
from fastworkflow.observation_offloading.state import reset_observation_state
from fastworkflow.workflow_agent import initialize_workflow_tool_agent


# --------------------------------------------------------------------------
# Fixture context callback classes: exactly what a workflow may declare.
# --------------------------------------------------------------------------

class AccountContext:
    """Declares the hook as a classmethod (the form real workflows use)."""

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


class ItemExplorerContext:
    """A workspace: navigable, but not an instance of anything."""


class RaisingContext:
    @classmethod
    def instance_label(cls, command_context_object):
        raise RuntimeError("a workflow's hook may be broken")


class FakeWorkflow:
    """The attributes ``context_identity`` and the execute closure read."""

    def __init__(self, name, obj, *, root=False):
        self._name = name
        self._obj = obj
        self._root = root
        self.context: dict = {}

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

    def test_unrecorded_prints_exactly_the_a1_line_and_root_prints_global(self) -> None:
        self.assertEqual(alias_line("O42"),
                         "Observation O42 (execute_workflow_query)\n")
        self.assertEqual(alias_line("O42", ""),
                         "Observation O42 (execute_workflow_query ran in context 'global')\n")
        self.assertIsNone(printed_context(alias_line("O42")))
        self.assertIsNone(printed_context(alias_line("O42", "")))
        self.assertEqual(printed_alias(alias_line("O42")), "O42")

    def test_context_with_an_instance(self) -> None:
        line = alias_line("O42", context_clause("Account", "28c5aeb5 (Jane Roe)"))
        self.assertEqual(
            line,
            "Observation O42 (execute_workflow_query ran in context 'Account' and 28c5aeb5 Jane Roe)\n")
        self.assertEqual(printed_alias(line), "O42")
        self.assertEqual(printed_context(line), "Account 28c5aeb5 Jane Roe")

    def test_context_without_an_instance_prints_the_name_alone(self) -> None:
        line = alias_line("O0", context_clause("ItemExplorer", ""))
        self.assertEqual(
            line, "Observation O0 (execute_workflow_query ran in context 'ItemExplorer')\n")
        self.assertEqual(printed_context(line), "ItemExplorer")

    def test_a_context_change_is_named_after_the_clause(self) -> None:
        line = alias_line("O0", context_clause("Identity", "4a0d (Sam Poe)"),
                          context_changed=True)
        self.assertEqual(
            line,
            "Observation O0 (execute_workflow_query ran in prior context 'Identity' and 4a0d Sam Poe"
            "; and resulted in a context change)\n")
        self.assertEqual(printed_alias(line), "O0")
        self.assertEqual(printed_context(line), "Identity 4a0d Sam Poe")
        self.assertEqual(strip_alias_line(line + "body"), "body")

    def test_no_context_name_means_no_clause(self) -> None:
        self.assertEqual(context_clause("", "28c5aeb5"), "")
        self.assertEqual(alias_line("O0", context_clause("", "28c5aeb5")),
                         alias_line("O0", ""))

    def test_parentheses_and_newlines_can_never_reach_the_line(self) -> None:
        clause = context_clause("Account", "28c5 (Alan\nCooper)\n)")
        self.assertNotIn("(", clause)
        self.assertNotIn(")", clause)
        self.assertNotIn("\n", clause)

    def test_a_label_ending_like_the_change_suffix_is_read_back_whole(self) -> None:
        clause = context_clause("Account", "28c5; and resulted in a context change")
        self.assertNotIn(";", clause)
        self.assertEqual(printed_context(alias_line("O0", clause)), clause)
        line = alias_line("O0", clause)
        self.assertEqual(printed_alias(line), "O0")
        self.assertEqual(printed_context(line), clause)
        self.assertEqual(strip_alias_line(line + "body"), "body")

    def test_the_label_is_capped(self) -> None:
        clause = context_clause("Account", "u" * 500)
        label = clause.split(" ", 1)[1]
        self.assertLessEqual(len(label), MAX_INSTANCE_LABEL_CHARS)
        self.assertTrue(label.endswith("..."))

    def test_the_line_is_stripped_and_digests_are_unchanged(self) -> None:
        body = "permission_uid  label\n85cde168  Item Catalog_Cloud Administrator\n"
        for clause in ("", "Account 28c5aeb5 Jane Roe", "ItemExplorer"):
            shown = alias_line("O0", clause) + body
            self.assertEqual(strip_alias_line(shown), body)
            self.assertEqual(
                hashlib.sha256(strip_alias_line(shown).encode()).hexdigest(),
                hashlib.sha256(body.encode()).hexdigest())

    def test_an_a1_line_recorded_before_this_change_still_reads(self) -> None:
        legacy = "Observation O12 (execute_workflow_query)\n477 holder(s).\n"
        self.assertEqual(printed_alias(legacy), "O12")
        self.assertIsNone(printed_context(legacy))
        self.assertEqual(strip_alias_line(legacy), "477 holder(s).\n")

    def test_a_comma_in_line_recorded_before_this_change_still_reads(self) -> None:
        legacy = "Observation O3 (execute_workflow_query, in ItemExplorer)\nrows"
        self.assertEqual(printed_alias(legacy), "O3")
        self.assertEqual(printed_context(legacy), "ItemExplorer")
        self.assertEqual(strip_alias_line(legacy), "rows")

    def test_an_offload_label_is_not_an_alias_line(self) -> None:
        label = offload_label(alias="O0", command_name="Account/list_permissions",
                              response="rows")
        self.assertTrue(is_offload_label(label))
        self.assertIsNone(printed_alias(label))


class DeclaredInstanceIdentity(unittest.TestCase):
    """The generic hook on the context callback class."""

    def test_classmethod_hook(self) -> None:
        self.assertEqual(
            declared_instance_label(AccountContext, _entity("28c5aeb5", "Jane Roe")),
            "28c5aeb5 (Jane Roe)")

    def test_attribute_hook(self) -> None:
        self.assertEqual(
            declared_instance_label(GroupContext, _entity("g-1")), "g-1")

    def test_no_declaration_yields_nothing(self) -> None:
        self.assertEqual(
            declared_instance_label(ItemExplorerContext, _entity("x")), "")

    def test_a_broken_hook_yields_nothing_rather_than_raising(self) -> None:
        self.assertEqual(declared_instance_label(RaisingContext, _entity("x")), "")

    def test_an_absent_identity_is_never_invented(self) -> None:
        # The object carries no uid and no label: the context name stands alone.
        self.assertEqual(declared_instance_label(AccountContext, _entity()), "")
        workflow = FakeWorkflow("Account", _entity())
        self.assertEqual(context_identity(workflow), ("Account", ""))
        self.assertEqual(context_clause_for(workflow), "Account")

    def test_root_context_has_no_identity(self) -> None:
        workflow = FakeWorkflow("Demo", _entity("root"), root=True)
        self.assertEqual(context_identity(workflow), ("", ""))
        self.assertEqual(context_clause_for(workflow), "")

    def test_no_workflow_is_not_an_error(self) -> None:
        self.assertEqual(context_identity(None), ("", ""))
        self.assertEqual(context_clause_for(None), "")


class ExecuteStepPrintsTheContextItRanIn(unittest.TestCase):
    """The execute closure prints the context its command RAN IN, before dispatch."""

    def setUp(self) -> None:
        reset_observation_state()
        self.addCleanup(reset_observation_state)
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.archive = RuntimeHandleArchive(
            str(Path(self.tempdir.name) / "handles.sqlite3"))
        self.scope = RuntimeHandleScope(
            channel_id="fixture-channel",
            turn_key="fixture-turn")
        self.workflow = FakeWorkflow("ItemExplorer", None)
        self.trajectory: dict = {}
        self.session = SimpleNamespace(
            workflow_tool_agent=SimpleNamespace(trajectory=self.trajectory, iteration_counter=0),
            get_active_workflow=lambda: self.workflow)
        self.moves = {"open_account_by_uid": ("Account", _entity("28c5aeb5", "Jane Roe"))}
        self.dispatched: list[str] = []

        def dispatch(command, chat_session_obj):
            self.dispatched.append(command)
            name = command.split()[0]
            if name in self.moves:
                self.workflow._name, self.workflow._obj = self.moves[name]
            return "Entered the context." if name in self.moves else "rows"

        def context_class(_workflow, name):
            return AccountContext if name == "Account" else ItemExplorerContext

        patches = [
            patch("fastworkflow.workflow_agent._execute_workflow_query", dispatch),
            patch("fastworkflow.context_identity.context_class_for", context_class),
        ]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)

        captured: dict = {}

        def build(_session, _signature, tools, **_kwargs):
            captured.update({getattr(tool, "name", None) or tool.__name__: tool for tool in tools})
            return SimpleNamespace()

        with patch("fastworkflow.workflow_agent.build_tool_agent", build):
            initialize_workflow_tool_agent(self.session)
        self.execute = captured["execute_workflow_query"]

    def archive_steps(self, *indexes: int) -> None:
        for index in indexes:
            compact_trajectory(self.trajectory, step_index=index, scope=self.scope,
                               selected_archive=self.archive)

    def step(self, index: int, command: str) -> str:
        self.trajectory[f"tool_name_{index}"] = "execute_workflow_query"
        response = self.execute(command)
        self.trajectory[f"observation_{index}"] = response
        return response

    def test_a_command_that_moves_the_context_prints_where_it_ran(self) -> None:
        """`open_account_by_uid` runs in ItemExplorer and ends in Account.

        It must read as the ItemExplorer command it is, with the change
        named; the `list_permissions` that follows belongs to the account.
        """
        first = self.step(0, "open_account_by_uid <account_uid>28c5aeb5</account_uid>")
        self.assertEqual(
            first,
            alias_line("O0", "ItemExplorer", context_changed=True,
                       now_in="Account 28c5aeb5 Jane Roe")
            + "Entered the context.")
        second = self.step(1, "list_permissions")
        self.assertEqual(second, alias_line("O1", "Account 28c5aeb5 Jane Roe") + "rows")
        self.archive_steps(0, 1)
        moved = self.archive.get(self.scope, "O0")
        self.assertEqual((moved["context_clause"], moved["context_changed"]),
                         ("ItemExplorer", True))
        stayed = self.archive.get(self.scope, "O1")
        self.assertEqual((stayed["context_clause"], stayed["context_changed"]),
                         ("Account 28c5aeb5 Jane Roe", False))

    def test_the_archive_stores_the_response_without_the_line(self) -> None:
        self.step(0, "open_account_by_uid <account_uid>28c5aeb5</account_uid>")
        compact_trajectory(self.trajectory, step_index=0, scope=self.scope,
                           selected_archive=self.archive)
        stored = self.archive.get(self.scope, "O0")
        self.assertEqual(stored["text"], "Entered the context.")
        self.assertEqual(stored["text_sha256"],
                         hashlib.sha256(b"Entered the context.").hexdigest())

    def test_a_root_command_prints_global(self) -> None:
        self.workflow = FakeWorkflow("Demo", None, root=True)
        self.assertEqual(self.step(0, "find_person"),
                         alias_line("O0", "") + "rows")
        self.archive_steps(0)
        self.assertEqual(self.archive.get(self.scope, "O0")["context_clause"], "")

    def test_no_agent_step_in_flight_leaves_the_response_alone(self) -> None:
        self.trajectory["tool_name_0"] = "what_can_i_do"
        self.assertEqual(self.execute("what_can_i_do"), "rows")
        self.assertIsNone(self.archive.get(self.scope, "O0"))

    def test_the_context_is_read_back_after_the_process_state_is_reset(self) -> None:
        self.step(0, "list_permissions")
        self.archive_steps(0)
        reset_observation_state()
        self.assertEqual(self.archive.get(self.scope, "O0")["context_clause"],
                         "ItemExplorer")


if __name__ == "__main__":
    unittest.main()


def test_a_line_printed_in_the_earlier_wording_still_parses():
    from fastworkflow.observation_offloading.labels import printed_subject

    old = ("Observation O3 (execute_workflow_query ran in Identity 28c5 Jane Roe"
           "; and resulted in a context change)\nbody")

    assert printed_subject(old) == ("Identity 28c5 Jane Roe", True)
    assert printed_subject("Observation O3 (execute_workflow_query ran in global)\nbody") == ("", False)


def test_a_named_destination_parses_back_to_the_context_the_command_ran_in():
    from fastworkflow.observation_offloading.labels import alias_line, printed_subject

    line = alias_line("O4", "ItemExplorer", context_changed=True,
                      now_in="Identity c062 Alisha Ochoa")

    assert line == ("Observation O4 (execute_workflow_query ran in prior context "
                    "'ItemExplorer'; now in context 'Identity' and c062 Alisha Ochoa)\n")
    assert printed_subject(line + "body") == ("ItemExplorer", True)


def test_printed_instances_reads_both_header_forms_in_order():
    from fastworkflow.observation_offloading.labels import printed_instances

    entered = ("Observation O5 (execute_workflow_query ran in prior context 'Identity' and "
               "4a0d Sam Poe; now in context 'Account' and 738e9c85 John Doe)\nbody")
    ran_in = "Observation O6 (execute_workflow_query ran in context 'Account' and 738e9c85)\nbody"

    assert printed_instances(entered) == [
        ("Identity", "4a0d", "Sam Poe"),
        ("Account", "738e9c85", "John Doe"),
    ]
    assert printed_instances(ran_in) == [("Account", "738e9c85", "")]


def test_printed_instances_skips_contexts_without_an_instance_and_non_headers():
    from fastworkflow.observation_offloading.labels import printed_instances

    assert printed_instances("Observation O7 (execute_workflow_query ran in context 'global')\nx") == []
    assert printed_instances("Observation O8 (execute_workflow_query)\nran in context 'Account' and 1 A") == []
    assert printed_instances("Offloaded observation O9 returned by execute_workflow_query.") == []

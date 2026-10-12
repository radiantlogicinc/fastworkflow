"""The unavailable-command refusal names the path, and the parameters it still needs.

``go_up`` follows the live parent, not every parent the hierarchy file allows.
``reset_context`` is used when it is shorter than climbing. A descend step says
``needs <field>`` when the command cannot run without it, and ``<field> required
to enter <context>`` when the command runs either way and only navigates when
the field is present.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

import fastworkflow
from fastworkflow import context_navigation
from fastworkflow.command_executor import CommandNotFoundError
from fastworkflow.context_navigation import (
    DescendEdge,
    live_context_chain,
    plan_navigation,
    render_unavailable_command,
)
from fastworkflow.workflow_agent import _explicit_agent_command

@pytest.fixture(autouse=True)
def _restore_sys_path(monkeypatch):
    """get_module() prepends the workflow's parent dir to sys.path; undo it per test."""
    monkeypatch.setattr(sys, "path", list(sys.path))


def _steps(path):
    return [step.command for step in path]


def test_global_reaches_a_workspace_by_its_descend_command():
    edges = (DescendEdge("*", "open_item_explorer", "ItemExplorer"),)
    path = plan_navigation("*", ["ItemExplorer"], ("*",), edges)["ItemExplorer"]
    assert _steps(path) == ["open_item_explorer"]
    assert path[0].needs == ()
    assert path[0].required_to_enter == ()


def test_go_up_follows_the_live_parent_not_every_declared_one():
    """Account may sit under ItemExplorer or Identity. The object in hand
    is under Identity, so the climb is two steps, not the static shortcut."""
    chain = ("Account", "Identity", "ItemExplorer", "*")
    path = plan_navigation("Account", ["ItemExplorer"], chain, ())["ItemExplorer"]
    assert _steps(path) == ["go_up", "go_up"]


def test_one_go_up_beats_reset_when_the_parent_is_the_target():
    chain = ("Identity", "ItemExplorer", "*")
    edges = (DescendEdge("*", "open_item_explorer", "ItemExplorer"),)
    path = plan_navigation("Identity", ["ItemExplorer"], chain, edges)["ItemExplorer"]
    assert _steps(path) == ["go_up"]


def test_reset_is_used_when_it_is_shorter_than_climbing():
    chain = ("ControlFinding", "ControlsMonitor", "*")
    edges = (DescendEdge("*", "open_item_explorer", "ItemExplorer"),)
    path = plan_navigation(
        "ControlFinding", ["ItemExplorer"], chain, edges)["ItemExplorer"]
    assert _steps(path) == ["reset_context", "open_item_explorer"]


def test_equal_length_prefers_the_step_that_needs_nothing():
    edges = (
        DescendEdge("ItemExplorer", "list_accounts", "Account", required_to_enter=("account_uid",)),
        DescendEdge("ItemExplorer", "open_account_free", "Account"),
    )
    path = plan_navigation(
        "ItemExplorer", ["Account"], ("ItemExplorer", "*"), edges)["Account"]
    assert _steps(path) == ["open_account_free"]


def test_a_shorter_conditional_step_is_kept_and_names_its_parameter():
    edges = (
        DescendEdge("Identity", "list_accounts", "Account", required_to_enter=("account_uid",)),
        DescendEdge("ItemExplorer", "open_account_by_uid", "Account", needs=("account_uid",)),
    )
    chain = ("Identity", "ItemExplorer", "*")
    path = plan_navigation("Identity", ["Account"], chain, edges)["Account"]
    assert _steps(path) == ["list_accounts"]
    assert path[0].required_to_enter == ("account_uid",)


def test_no_declared_descend_has_no_path():
    path = plan_navigation("*", ["Subscription"], ("*",), ())["Subscription"]
    assert path is None


def test_the_message_for_find_person_says_open_item_explorer():
    edges = (DescendEdge("*", "open_item_explorer", "ItemExplorer"),)
    paths = plan_navigation("*", ["ItemExplorer"], ("*",), edges)
    text = render_unavailable_command("find_person", "*", ["ItemExplorer"], paths)
    assert text == (
        "Command 'find_person' is not available in the current context 'global'. "
        "It is available in: 'ItemExplorer'. "
        "From here to 'ItemExplorer': open_item_explorer. Then retry 'find_person'."
    )
    assert "go_up or reset_context" not in text


def test_the_message_lists_both_kinds_of_caveat():
    edges = (
        DescendEdge("*", "open_item_explorer", "ItemExplorer"),
        DescendEdge(
            "ItemExplorer", "open_account_by_uid", "Account", needs=("account_uid",)),
        DescendEdge(
            "Identity", "list_accounts", "Account", required_to_enter=("account_uid",)),
    )
    opened = plan_navigation("*", ["Account"], ("*",), edges)
    from_global = render_unavailable_command("list_permissions", "*", ["Account"], opened)
    assert (
        "open_item_explorer, then open_account_by_uid <account_uid> (needs account_uid)"
        in from_global
    )

    chain = ("Identity", "ItemExplorer", "*")
    listed = plan_navigation("Identity", ["Account"], chain, edges)
    from_identity = render_unavailable_command("list_permissions", "Identity", ["Account"], listed)
    assert (
        "list_accounts <account_uid> (account_uid required to enter Account)"
        in from_identity
    )


def test_each_home_context_gets_its_own_path():
    edges = (
        DescendEdge("*", "open_item_explorer", "ItemExplorer"),
        DescendEdge("*", "open_controls_monitor", "ControlsMonitor"),
    )
    paths = plan_navigation("*", ["ControlsMonitor", "ItemExplorer"], ("*",), edges)
    text = render_unavailable_command("list_findings", "*", ["ControlsMonitor", "ItemExplorer"], paths)
    assert "From here to 'ControlsMonitor': open_controls_monitor." in text
    assert "From here to 'ItemExplorer': open_item_explorer." in text


def test_live_chain_stops_at_global_when_the_parent_is_none():
    class Explorer:
        pass

    class Workflow:
        current_command_context_name = "ItemExplorer"
        current_command_context = Explorer()
        root_command_context = None

        def get_parent(self, obj):
            return None

    assert live_context_chain(Workflow()) == ("ItemExplorer", "*")


class _Model:
    def inherited_base_contexts(self, name):
        if name == "ItemExplorer":
            return {"Directory"}
        return set()


class _App:
    contexts = {
        "*": ["open_item_explorer"],
        "Directory": ["Directory/find_person"],
        "ItemExplorer": ["Directory/find_person"],
    }
    context_model = _Model()

    def get_command_names(self, context):
        return self.contexts[context]


class _Cme:
    def get_command_names(self, context):
        return []


class _Workflow:
    folderpath = "app"
    current_command_context_name = "*"
    current_command_context = None
    root_command_context = None

    def get_parent(self, obj):
        return None


def test_the_agent_guard_reports_the_path(monkeypatch):
    internal = fastworkflow.get_internal_workflow_path("command_metadata_extraction")

    class Registry:
        @staticmethod
        def get_definition(path, load_cached=True):
            if path == internal:
                return _Cme()
            return _App()

    monkeypatch.setattr(fastworkflow, "RoutingRegistry", Registry)
    monkeypatch.setattr(
        context_navigation,
        "collect_descend_edges",
        lambda path: (DescendEdge("*", "open_item_explorer", "ItemExplorer"),),
    )
    with pytest.raises(CommandNotFoundError) as caught:
        _explicit_agent_command("find_person", _Workflow())
    assert "open_item_explorer" in str(caught.value)
    assert "go_up or reset_context" not in str(caught.value)


def test_go_up_at_the_top_level_names_the_commands_to_use_instead():
    from fastworkflow._workflows.command_metadata_extraction._commands.IntentDetection.go_up import (
        MAX_HINT_COMMANDS, ResponseGenerator)

    fastworkflow.init({})
    todo = str(Path(__file__).resolve().parent / "todo_list_workflow")
    app = fastworkflow.Workflow.create(workflow_folderpath=todo, workflow_id_str="go-up-top-level")
    wrapper = fastworkflow.Workflow.create(workflow_folderpath="/tmp", workflow_id_str="go-up-wrapper")
    wrapper.context = {"app_workflow": app}
    assert app.is_current_command_context_root

    text = ResponseGenerator()(wrapper, "IntentDetection/go_up").command_response.response

    assert text.startswith("Already at the top-level 'global' context. From here, use one of: ")
    names = text.removeprefix("Already at the top-level 'global' context. From here, use one of: ")
    names = names.removesuffix(".").split(", ")
    assert 0 < len(names) <= MAX_HINT_COMMANDS
    assert not {"go_up", "reset_context", "what_can_i_do", "what_is_current_context", "abort"} & set(names)
    assert all("/" not in name for name in names)

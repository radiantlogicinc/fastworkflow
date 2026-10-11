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
    collect_descend_edges,
    live_context_chain,
    plan_navigation,
    render_unavailable_command,
    unavailable_command_message,
)
from fastworkflow.workflow_agent import _explicit_agent_command

IDO_WORKFLOW = Path(__file__).resolve().parents[2] / "ido" / "ido_workflow"


@pytest.fixture(autouse=True)
def _restore_sys_path(monkeypatch):
    """get_module() prepends the workflow's parent dir (~/rl/ido) to sys.path; undo it per test."""
    monkeypatch.setattr(sys, "path", list(sys.path))


@pytest.mark.skipif(
    not IDO_WORKFLOW.is_dir(),
    reason="ido workflow is not checked out beside fastworkflow",
)
def test_deferred_planner_skips_mixin_contexts_and_navigates_to_explorer():
    import fastworkflow
    from fastworkflow.command_metadata_api import CommandMetadataAPI

    class Workflow:
        folderpath = str(IDO_WORKFLOW)
        current_command_context_name = "*"
        current_command_context = None
        root_command_context = None

        def get_parent(self, obj):
            return None

    fastworkflow.init({"NOT_FOUND": "NOT_FOUND"})
    text = CommandMetadataAPI.get_all_contexts_command_display_text(
        subject_workflow_path=str(IDO_WORKFLOW),
        cme_workflow_path=fastworkflow.get_internal_workflow_path(
            "command_metadata_extraction"
        ),
        active_context_name="*",
        navigation_workflow=Workflow(),
    )
    assert "after entering the Directory context" not in text
    assert "after entering the DirectoryExplorer context" in text
    assert (
        "Before executing find_account, navigate to DirectoryExplorer as follows: "
        "open_directory"
    ) in text
    assert "navigate to Directory as follows:" not in text


def test_deferred_planner_command_matches_find_identity_shape():
    import fastworkflow
    from fastworkflow.command_metadata_api import CommandMetadataAPI

    class Workflow:
        folderpath = str(IDO_WORKFLOW)
        current_command_context_name = "*"
        current_command_context = None
        root_command_context = None

        def get_parent(self, obj):
            return None

    block = CommandMetadataAPI.format_deferred_planner_command(
        subject_workflow_path=str(IDO_WORKFLOW),
        cme_workflow_path=fastworkflow.get_internal_workflow_path(
            "command_metadata_extraction"
        ),
        owning_context_name="DirectoryExplorer",
        qualified_command_name="Directory/find_identity",
        active_context_name="*",
        navigation_workflow=Workflow(),
    )
    assert block.startswith("- find_identity\n")
    assert "Find identities by name, login or email" in block
    assert (
        "Before executing find_identity, navigate to DirectoryExplorer as follows: "
        "open_directory"
    ) in block


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


@pytest.mark.skipif(
    not IDO_WORKFLOW.is_dir(),
    reason="ido workflow is not checked out beside fastworkflow",
)
def test_ido_global_find_identity_path_is_open_directory():
    class Workflow:
        folderpath = str(IDO_WORKFLOW)
        current_command_context_name = "*"
        current_command_context = None
        root_command_context = None

        def get_parent(self, obj):
            return None

    text = unavailable_command_message(
        Workflow(), "find_identity", "*", ["DirectoryExplorer"])
    assert "From here to 'DirectoryExplorer': open_directory." in text
    assert "go_up or reset_context" not in text

    class DirectoryExplorer:
        pass

    class Identity:
        pass

    class InIdentity:
        folderpath = str(IDO_WORKFLOW)
        current_command_context_name = "Identity"
        current_command_context = Identity()
        root_command_context = None

        def get_parent(self, obj):
            if isinstance(obj, Identity):
                return DirectoryExplorer()
            return None

    from_identity = unavailable_command_message(
        InIdentity(), "list_permissions", "Identity", ["Account"])
    assert "list_accounts <account_uid>" in from_identity
    assert "account_uid required to enter Account" in from_identity

    edges = collect_descend_edges(str(IDO_WORKFLOW))
    opened = plan_navigation("*", ["Account"], ("*",), edges)["Account"]
    assert [step.command for step in opened] == ["open_directory", "open_account_by_uid"]
    assert "account_uid" in opened[1].needs or "account_uid" in opened[1].required_to_enter


@pytest.mark.skipif(
    not IDO_WORKFLOW.is_dir(),
    reason="ido workflow is not checked out beside fastworkflow",
)
def test_the_command_that_enters_the_current_context_says_you_are_already_there():
    class Permission:
        uid = "85cde168"
        label = "Active Directory_Cloud Administrator"

    class InPermission:
        folderpath = str(IDO_WORKFLOW)
        current_command_context_name = "Permission"
        current_command_context = Permission()
        root_command_context = None
        is_current_command_context_root = False

        def get_parent(self, obj):
            return None

    from fastworkflow.context_navigation import (
        enters_current_context, render_already_in_context)

    assert enters_current_context(InPermission(), "open_permission_by_uid", "Permission")
    assert not enters_current_context(InPermission(), "find_permission", "Permission")
    text = render_already_in_context("open_permission_by_uid", "Permission", InPermission())

    assert text.startswith("You are already in context 'Permission'")
    assert "show_properties" in text
    assert "go_up first, then retry 'open_permission_by_uid'" in text


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

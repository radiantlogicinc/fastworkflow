"""Calls by exact name along the lookup path (design 3.9, Phase 9 step 4).

A command typed by its exact name runs when it is on the path from the current object: the
current context and its bases, each enclosing record on the live chain with its bases, then the
global scope. It runs bound to the record of its scope (the root object for the global scope) and
is filed under that record. The nearest scope wins; a qualified name must name a context on the
path; the qualified-token collapse rule holds within each scope.

Each test runs the real pipeline on a copy of ``tests/todo_list_workflow`` with real application
objects (a manager, a list, an item, linked by ``parent``). The copy gets one global command the
workflow lacks (``global_probe``, which reports the record it was bound to). The one seam is the
intent classifier, replaced by a double that is never consulted: every name here is an exact
command name, which is decided without a classifier.
"""

import os
import shutil

import pytest

import fastworkflow
from fastworkflow._workflows.command_metadata_extraction import intent_detection
from fastworkflow.command_executor import CommandExecutor, CommandNotFoundError, call_response_generator
from fastworkflow.command_metadata_api import CommandMetadataAPI
from fastworkflow.workflow_agent import _explicit_agent_command
from tests.todo_list_workflow.application.todo_item import TodoItem
from tests.todo_list_workflow.application.todo_list import TodoList
from tests.todo_list_workflow.application.todo_manager import TodoListManager

TODO_SOURCE = os.path.join(os.path.dirname(__file__), "todo_list_workflow")

GLOBAL_PROBE = '''"""global_probe: report the record a global command was bound to (test copy only)."""
import fastworkflow
from fastworkflow import CommandOutput, CommandResponse


class ResponseGenerator:
    def __call__(self, workflow: fastworkflow.Workflow, command: str, input) -> CommandResponse:
        bound = workflow.record
        return CommandResponse(response=f"global_probe bound to {type(bound).__name__}")
'''


@pytest.fixture
def chain(tmp_path):
    """A workflow copy with a live chain TodoItem -> TodoList -> TodoListManager (root) and a global command."""
    workflow_path = tmp_path / "todo_list_workflow"
    shutil.copytree(
        TODO_SOURCE, workflow_path,
        ignore=shutil.ignore_patterns(
            "___command_info", "___workflow_contexts", "___convo_info", "__pycache__",
        ),
    )
    (workflow_path / "_commands" / "global_probe.py").write_text(GLOBAL_PROBE)
    app = fastworkflow.Workflow.create(
        workflow_folderpath=str(workflow_path),
        workflow_id_str=f"lookup-exact-{tmp_path.name}",
    )
    cme = fastworkflow.Workflow.create(
        workflow_folderpath=fastworkflow.get_internal_workflow_path("command_metadata_extraction"),
        parent_workflow_id=app.id,
        workflow_context={"app_workflow": app},
    )
    manager = TodoListManager(file_path=str(tmp_path / "lists.json"), load_from_file=False)
    todo_list = TodoList(1, "groceries")
    todo_list.parent = manager
    item = TodoItem(1, "milk")
    item.parent = todo_list
    app.root = manager
    app.record = item
    return app, cme, item, todo_list, manager


class _RefusingRouter:
    """The classifier must not be consulted for an exact command name."""

    def __init__(self, model_artifact_path):
        self.modelpipeline = None

    def predict(self, command):
        raise AssertionError(f"the classifier was consulted for {command!r}")

    def predict_with_details(self, command):
        return self.predict(command)


class Run:
    """What one exact-name call did: the CME walk's artifacts, and the target's own response."""

    def __init__(self, artifacts: dict, response: str, output=None):
        self.artifacts = artifacts
        self.command_name = artifacts["command_name"]
        self.ran_on_context_name = artifacts.get("ran_on_context_name")
        self.ran_on_context_clause = artifacts.get("ran_on_context_clause")
        self.response = response
        self.output = output


def _run(app, cme, monkeypatch, command: str) -> Run:
    """Run *command* as the agent tool does: resolve it, let the CME walk bind and parse it, then
    run the target's response generator on the record the walk bound (the executor's own step).
    The classifier is never consulted."""
    monkeypatch.setattr(intent_detection, "CommandRouter", _RefusingRouter)
    resolved = _explicit_agent_command(command, app)
    walked = CommandExecutor.perform_action(
        cme, fastworkflow.Action(command_name="wildcard", command=resolved))
    assert walked.success, walked.command_response.response
    artifacts = walked.command_response.artifacts
    routing = fastworkflow.RoutingRegistry.get_definition(app.folderpath)
    gen_cls = routing.get_command_class(
        artifacts["command_name"], fastworkflow.ModuleType.RESPONSE_GENERATION_INFERENCE)
    params = artifacts["cmd_parameters"]
    raw = call_response_generator(gen_cls(), app, artifacts["command"], params)
    response = raw.command_response.response if isinstance(raw, fastworkflow.CommandOutput) else str(raw)
    return Run(artifacts, response, output=raw)


def test_an_enclosing_context_command_by_exact_name_runs_bound_and_filed_under_its_record(
        chain, monkeypatch, setup_test_environment):
    app, cme, item, todo_list, _ = chain
    assert _explicit_agent_command("get_all_children", app) == "get_all_children"

    output = _run(app, cme, monkeypatch, "get_all_children")

    assert output.command_name == "TodoList/get_all_children"
    # Bound to the list, not the item: the command ran on the record it is filed under.
    assert app.record is todo_list
    assert output.ran_on_context_name == "TodoList"
    assert output.ran_on_context_clause == "TodoList"
    # The current object does not move.
    assert app._current_command_context is item


def test_a_global_command_by_exact_name_from_inside_a_context_runs_bound_to_the_root(
        chain, monkeypatch, setup_test_environment):
    app, cme, item, _, manager = chain
    assert _explicit_agent_command("global_probe", app) == "global_probe"

    output = _run(app, cme, monkeypatch, "global_probe")

    assert output.response == "global_probe bound to TodoListManager"
    # The global scope binds the root object, and the run is filed under it.
    assert app.record is manager
    assert output.ran_on_context_name == "TodoListManager"
    assert app._current_command_context is item


def test_the_nearest_scope_wins_on_a_short_name_clash(chain, monkeypatch, setup_test_environment):
    """get_properties is a command of TodoItem and of TodoList: from the item, the item's runs."""
    app, cme, item, _, _ = chain

    output = _run(app, cme, monkeypatch, "get_properties")

    assert output.command_name == "TodoItem/get_properties"
    assert output.ran_on_context_name is None
    assert app.record is item


def test_a_qualified_name_on_the_path_runs_that_scope_even_when_a_nearer_scope_shares_the_short_name(
        chain, monkeypatch, setup_test_environment):
    """From the item, TodoList/get_properties names the list's command, not the item's own."""
    app, cme, _, todo_list, _ = chain

    output = _run(app, cme, monkeypatch, "TodoList/get_properties")

    assert output.command_name == "TodoList/get_properties"
    assert app.record is todo_list
    assert output.ran_on_context_name == "TodoList"


def test_a_qualified_name_off_the_path_gets_the_navigation_message(
        chain, monkeypatch, setup_test_environment):
    """From the root, TodoItem is neither enclosing nor a base: its command is not called here."""
    app, _, _, _, manager = chain
    app.record = manager
    monkeypatch.setattr(intent_detection, "CommandRouter", _RefusingRouter)

    with pytest.raises(CommandNotFoundError) as refused:
        _explicit_agent_command("TodoItem/status", app)

    message = str(refused.value)
    assert "'TodoItem/status' is not available in the current context 'TodoListManager'" in message
    assert "It is available in: 'TodoList'." in message


def test_the_other_contexts_text_does_not_send_the_agent_into_a_context_for_a_command_on_the_path(
        chain, setup_test_environment):
    """From the list, the root is on the lookup path: its commands run here, so the other-contexts
    text says so instead of telling the agent to enter the root to run them."""
    app, _, _, todo_list, _ = chain
    app.record = todo_list

    text = CommandMetadataAPI.get_other_contexts_text(
        subject_workflow_path=app.folderpath,
        active_context_name="TodoList",
        navigation_workflow=app,
    )

    root_line = next(line for line in text.splitlines() if line.startswith("- TodoListManager:"))
    assert "callable here" in root_line
    assert "To enter" not in root_line
    assert "enter one to see and run its commands" in text.splitlines()[0]


def test_the_other_contexts_text_still_says_how_to_enter_a_context_off_the_path(
        chain, setup_test_environment):
    """From the root, the list is off the path: it is still entered, as before."""
    app, _, _, _, manager = chain
    app.record = manager

    text = CommandMetadataAPI.get_other_contexts_text(
        subject_workflow_path=app.folderpath,
        active_context_name="TodoListManager",
        navigation_workflow=app,
    )

    list_line = next(line for line in text.splitlines() if line.startswith("- TodoList:"))
    assert "To enter" in list_line
    assert "callable here" not in list_line


def test_a_base_command_named_by_its_qualified_name_runs_on_the_derived_record(
        chain, monkeypatch, setup_test_environment):
    """TodoList inherits from TodoItem, so TodoItem's scope is on the path from the list, and its
    get_properties runs on the list (the current object, for its bases). The qualified token must not
    collapse onto the list's own get_properties: the collapse rule is checked within each scope, and
    TodoItem's scope has one get_properties."""
    app, cme, _, todo_list, _ = chain
    app.record = todo_list

    output = _run(app, cme, monkeypatch, "TodoItem/get_properties")

    assert output.command_name == "TodoItem/get_properties"
    assert app.record is todo_list
    assert output.ran_on_context_name is None

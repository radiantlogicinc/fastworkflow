"""Three defects that do not depend on the scoped lookup (design 2.18, Phase 9 step 2).

1. ``parameter_value`` must not start the parent-context climb. A head that answers
   "this is a bare value" ends the walk where it stands (3.9, step 4).
2. The planner's listing must not call an enclosing context's command "not callable
   until then". It is callable here, from the live chain (3.9, Catalogs).
3. A run the climb executes on an enclosing record is filed under that record's
   context, not under the current one (3.9, Filing the run).

Each test runs the real pipeline on a copy of ``tests/todo_list_workflow`` with real
application objects (a manager, a list, an item, linked by ``parent``). The one seam
is the intent classifier: this copy has no trained artifacts, so ``CommandRouter`` is
replaced, as the other intent-detection tests do, by a double that either answers
with the labels a test names or refuses to answer at all. Everything else, including
the parent walk in ``_commands/wildcard.py``, is the real code.
"""

import os
import shutil

import pytest

import fastworkflow
from fastworkflow._workflows.command_metadata_extraction import intent_detection
from fastworkflow._workflows.command_metadata_extraction._commands import wildcard
from fastworkflow.command_executor import CommandExecutor
from fastworkflow.command_metadata_api import CommandMetadataAPI
from fastworkflow.workflow_execution_context import WorkflowExecutionContext
from tests.todo_list_workflow.application.todo_item import TodoItem
from tests.todo_list_workflow.application.todo_list import TodoList
from tests.todo_list_workflow.application.todo_manager import TodoListManager

TODO_SOURCE = os.path.join(os.path.dirname(__file__), "todo_list_workflow")


@pytest.fixture
def todo_chain(tmp_path):
    """A real workflow copy with a live chain: TodoItem -> TodoList -> TodoListManager (root)."""
    workflow_path = tmp_path / "todo_list_workflow"
    shutil.copytree(
        TODO_SOURCE, workflow_path,
        ignore=shutil.ignore_patterns(
            "___command_info", "___workflow_contexts", "___convo_info", "__pycache__",
        ),
    )
    app = fastworkflow.Workflow.create(
        workflow_folderpath=str(workflow_path),
        workflow_id_str=f"lookup-defects-{tmp_path.name}",
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
    app.root_command_context = manager
    app.current_command_context = item
    return app, cme, item, todo_list, manager


def _labelling_router(labels_by_context: dict, consulted: list):
    """A classifier double that answers with the labels named per context folder."""

    class LabellingRouter:
        def __init__(self, model_artifact_path):
            self.context_name = model_artifact_path.rstrip("/").split("/")[-1]
            self.modelpipeline = None  # keeps the embedding cache off, as in the other tests

        def predict(self, command):
            return self.predict_with_details(command)[0]

        def predict_with_details(self, command):
            consulted.append(self.context_name)
            return labels_by_context[self.context_name], {}

    return LabellingRouter


def _refusing_router():
    class RefusingRouter:
        def __init__(self, model_artifact_path):
            self.modelpipeline = None

        def predict(self, command):
            raise AssertionError(f"the classifier was consulted for {command!r}")

        def predict_with_details(self, command):
            return self.predict(command)

    return RefusingRouter


# ---------------------------------------------------------------------------
# Defect 1: parameter_value does not start the climb
# ---------------------------------------------------------------------------

def test_a_bare_value_answer_stops_the_climb_where_it_is_given(
        todo_chain, monkeypatch, setup_test_environment):
    """TodoItem answers parameter_value. TodoList would answer a real command: it must not be asked."""
    app, cme, *_ = todo_chain
    consulted: list[str] = []
    monkeypatch.setattr(intent_detection, "CommandRouter", _labelling_router(
        {"TodoItem": ["parameter_value"],
         "TodoList": ["get_all_children"],
         "TodoListManager": ["list_todo_lists"]},
        consulted))

    output = wildcard.ResponseGenerator()(cme, "3", None)

    assert consulted == ["TodoItem"]
    # Handled by the you_misunderstood path: the routing Output is handled and names no command to run.
    assert output.handled
    assert output.command_name is None


def test_a_bare_value_at_an_ancestor_does_not_climb_further(
        todo_chain, monkeypatch, setup_test_environment):
    """TodoItem declines (wildcard), TodoList answers parameter_value: the walk stops at TodoList."""
    app, cme, *_ = todo_chain
    consulted: list[str] = []
    monkeypatch.setattr(intent_detection, "CommandRouter", _labelling_router(
        {"TodoItem": ["wildcard"],
         "TodoList": ["parameter_value"],
         "TodoListManager": ["list_todo_lists"]},
        consulted))

    output = wildcard.ResponseGenerator()(cme, "3", None)

    assert consulted == ["TodoItem", "TodoList"]
    assert output.handled
    assert output.command_name is None


def test_a_decline_still_climbs_to_the_ancestor_that_answers(
        todo_chain, monkeypatch, setup_test_environment):
    """The fix must not disable the climb itself: a decline followed by a command still resolves."""
    app, cme, *_ = todo_chain
    consulted: list[str] = []
    monkeypatch.setattr(intent_detection, "CommandRouter", _labelling_router(
        {"TodoItem": ["wildcard"],
         "TodoList": ["get_all_children"],
         "TodoListManager": ["list_todo_lists"]},
        consulted))

    wildcard.ResponseGenerator()(cme, "show the children", None)

    assert consulted == ["TodoItem", "TodoList"]
    assert cme.context["command_name"] == "TodoList/get_all_children"


# ---------------------------------------------------------------------------
# Defect 2: the planner's listing calls enclosing commands callable here
# ---------------------------------------------------------------------------

def test_enclosing_commands_are_listed_as_callable_here(todo_chain, setup_test_environment):
    app, *_ = todo_chain
    text = CommandMetadataAPI.get_all_contexts_command_display_text(
        subject_workflow_path=app.folderpath,
        cme_workflow_path=fastworkflow.get_internal_workflow_path("command_metadata_extraction"),
        active_context_name="TodoItem",
        navigation_workflow=app,
    )

    assert "not callable until then" not in text
    assert "Commands available here from the TodoList context" in text
    block = text.split("Commands available here from the TodoList context", 1)[1]
    block = block.split("\nCommands available", 1)[0]
    assert "- get_all_children" in block
    # Callable here: no instruction to navigate first.
    assert "navigate to" not in block


def test_a_context_off_the_live_chain_keeps_the_navigation_wording(todo_chain, setup_test_environment):
    """Without a live object there is no chain to read, so nothing is called callable here."""
    app, *_ = todo_chain
    text = CommandMetadataAPI.get_all_contexts_command_display_text(
        subject_workflow_path=app.folderpath,
        cme_workflow_path=fastworkflow.get_internal_workflow_path("command_metadata_extraction"),
        active_context_name="TodoItem",
    )
    assert "Commands available after entering the TodoList context (not callable until then):" in text


# ---------------------------------------------------------------------------
# Defect 3: a run the climb executes on an enclosing record is filed under it
# ---------------------------------------------------------------------------

def test_a_run_on_an_enclosing_record_is_filed_under_that_record(
        todo_chain, monkeypatch, setup_test_environment):
    app, *_ = todo_chain
    # The name is an exact command of TodoList, not of TodoItem: the foreign-name
    # guard declines it in TodoItem and the climb resolves it in TodoList, so no
    # classifier is consulted anywhere on this path.
    monkeypatch.setattr(intent_detection, "CommandRouter", _refusing_router())
    ctx = WorkflowExecutionContext(run_as_agent=True)
    ctx.bind_app_workflow(app)
    ctx.push_active_workflow(app)
    try:
        output = CommandExecutor.invoke_command(ctx, "get_all_children")
    finally:
        ctx.pop_active_workflow()

    assert output.success
    assert output.command_name == "TodoList/get_all_children"
    # The command ran on the TodoList record, so that is the context it is filed under.
    assert output.context == "TodoList"
    assert output.ran_in_context_name == "TodoList"
    assert output.ran_in_context_clause == "TodoList"
    # The turn's held run for this step names the same record.
    from fastworkflow.workflow_agent import _held_context
    assert _held_context(ctx, output, "TodoItem", "TodoItem") == ("TodoList", "TodoList")


def test_a_run_on_the_current_record_is_still_filed_under_it(
        todo_chain, monkeypatch, setup_test_environment):
    """No climb, no change: a command of the current context keeps the current context."""
    app, *_ = todo_chain
    monkeypatch.setattr(intent_detection, "CommandRouter", _refusing_router())
    ctx = WorkflowExecutionContext(run_as_agent=True)
    ctx.bind_app_workflow(app)
    ctx.push_active_workflow(app)
    try:
        output = CommandExecutor.invoke_command(ctx, "get_properties")
    finally:
        ctx.pop_active_workflow()

    assert output.context == "TodoItem"
    assert getattr(output, "ran_in_context_name", None) in (None, "")


def test_the_step_that_ran_on_an_enclosing_record_holds_that_record(
        todo_chain, monkeypatch, setup_test_environment):
    """The agent step's held run and its run state both name the enclosing record."""
    from types import SimpleNamespace

    from fastworkflow.workflow_agent import _execute_workflow_query

    app, *_ = todo_chain
    # Free text, not a command name: the agent tool refuses a name this context does not
    # offer before the walk, so a climb in agent mode is reached through the classifier.
    consulted: list[str] = []
    monkeypatch.setattr(intent_detection, "CommandRouter", _labelling_router(
        {"TodoItem": ["wildcard"], "TodoList": ["get_all_children"]}, consulted))
    ctx = WorkflowExecutionContext(run_as_agent=True)
    ctx.bind_app_workflow(app)
    ctx.push_active_workflow(app)
    run_state: dict = {}
    ctx._workflow_tool_agent = SimpleNamespace(run_state=run_state)
    try:
        response = _execute_workflow_query("show the children", chat_session_obj=ctx)
    finally:
        ctx.pop_active_workflow()

    assert consulted == ["TodoItem", "TodoList"]
    assert "todoitem_ids" in response
    assert run_state.get("ran_on_clause") == "TodoList"

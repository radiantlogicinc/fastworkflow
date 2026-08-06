"""Round-trip the bundled workflows through their serialization hooks.

Each workflow is driven by its own commands until it holds a live command
context, projected with ``project_command_contexts``, and restored into a
*different* ``Workflow``. Nothing here calls a ``Context`` hook directly: what
is under test is that the framework's projection and its restore agree, and in
particular that the current context comes back as a node *inside* the restored
anchor rather than as an equal copy of one.
"""

from __future__ import annotations

import logging
import uuid
from contextlib import contextmanager
from pathlib import Path

import pytest

import fastworkflow
from fastworkflow import serialization_hooks
from fastworkflow.command_routing import RoutingRegistry
from fastworkflow.serialization_hooks import ProjectionOutcome
from fastworkflow.state_serialization import validate_state
from fastworkflow.utils.logging import logger

EXAMPLES = Path(fastworkflow.__file__).parent / "examples"
TESTS = Path(__file__).parent


@pytest.fixture
def initialized_fastworkflow(tmp_path):
    fastworkflow.init({"SPEEDDICT_FOLDERNAME": str(tmp_path / "speedict")})
    RoutingRegistry.clear_registry()
    serialization_hooks.reset_warnings()
    yield
    RoutingRegistry.clear_registry()
    serialization_hooks.reset_warnings()


@pytest.fixture
def todo_list_json_preserved():
    """Undo the writes create_todo_list makes to the workflow's own JSON file."""
    filepath = TESTS / "todo_list_workflow" / "application" / "todo_list.json"
    original = filepath.read_text()
    yield
    filepath.write_text(original)


def make_workflow(folderpath: Path) -> fastworkflow.Workflow:
    """A workflow with an id of its own, so that "fresh" really is a new object.

    Given no id, Workflow.create derives one from the folder name and hands back
    the live instance already registered under it — which would be the very
    workflow a restore is supposed to be independent of.
    """
    return fastworkflow.Workflow.create(
        str(folderpath), workflow_id_str=f"hooks-{uuid.uuid4().hex}"
    )


def run_command(workflow: fastworkflow.Workflow, command_name: str, command: str, **parameters):
    """Run one of the workflow's own commands, dispatching as perform_action does.

    Not perform_action itself: its direct-action path constructs an
    InputForParamExtraction without the command's Signature class, so a command
    carrying a db_lookup field — messaging_app_4's set_current_user — raises
    before its response generator ever runs. Parameters are supplied here
    already valid, so parameter extraction is not what these tests exercise.
    """
    routing = RoutingRegistry.get_definition(workflow.folderpath)
    generator = routing.get_command_class(
        command_name, fastworkflow.ModuleType.RESPONSE_GENERATION_INFERENCE
    )()
    parameters_class = routing.get_command_class(
        command_name, fastworkflow.ModuleType.COMMAND_PARAMETERS_CLASS
    )

    workflow.command_context_for_response_generation = workflow.current_command_context
    if parameters_class is None:
        return generator(workflow, command)
    return generator(workflow, command, parameters_class(**parameters))


def project(workflow: fastworkflow.Workflow):
    """Project the workflow's contexts and check the record is durable-safe."""
    projection = serialization_hooks.project_command_contexts(workflow)
    assert projection.outcome is ProjectionOutcome.SERIALIZABLE, projection.reason
    assert projection.is_checkpointable

    record = projection.as_record()
    validate_state({"context": record})
    return projection, record


@contextmanager
def captured_warnings():
    """Collect WARNING records from the fastWorkflow logger.

    caplog cannot see them: that logger sets propagate=False, so its records
    never reach the root handler pytest installs.
    """
    records: list[logging.LogRecord] = []

    class Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = Collector(level=logging.WARNING)
    logger.addHandler(handler)
    try:
        yield records
    finally:
        logger.removeHandler(handler)


# ---------------------------------------------------------------------------
# simple_workflow_template — root is a cyclic, identity-keyed WorkItem tree
# ---------------------------------------------------------------------------

def test_simple_workflow_template_round_trip(initialized_fastworkflow):
    folderpath = EXAMPLES / "simple_workflow_template"
    workflow = make_workflow(folderpath)

    run_command(workflow, "startup", "startup")
    run_command(workflow, "WorkItem/add_child_workitem", "add bug b1",
                workitem_type="Bug", id="b1")
    run_command(workflow, "WorkItem/go_to_workitem", "go to the task",
                path="/Story[index=0]/Task[index=0]")
    run_command(workflow, "WorkItem/mark_as_complete", "done", is_complete=True)
    run_command(workflow, "WorkItem/go_to_workitem", "go to bug b1", path="/Bug[id=b1]")

    live_root = workflow.root_command_context
    assert workflow.current_command_context is not live_root

    projection, record = project(workflow)
    assert projection.anchor_is_root
    assert projection.current_locator == "/Bug[id=b1]"
    assert projection.response_locator == "/Story[index=0]/Task[index=0]"

    restored = make_workflow(folderpath)
    assert serialization_hooks.restore_command_contexts(restored, record)

    root = restored.root_command_context
    assert root is not live_root
    assert root.get_absolute_path() == "/"
    assert [child.type for child in root._children] == ["Story", "Bug"]

    story = root.get_workitem("/Story[index=0]")
    task = root.get_workitem("/Story[index=0]/Task[index=0]")
    bug = root.get_workitem("/Bug[id=b1]")

    # Identity, not equality: the slots must be nodes of the restored tree.
    assert restored.current_command_context is bug
    assert restored.command_context_for_response_generation is task

    # The links a JSON snapshot cannot carry.
    assert bug.parent is root
    assert task.parent is story
    assert story.parent is root
    assert root.index_of(story) == 0
    assert root.index_of(bug) == 1
    assert story.index_of(task) == 0

    # The schema is shared by the whole tree, not copied per node.
    assert story._workflow_schema is root._workflow_schema
    assert task._workflow_schema is root._workflow_schema

    # min_cardinality children are restored, not re-created on top of the snapshot.
    assert root.get_child_count("Story") == 1
    assert story.get_child_count("Task") == 1

    assert bug.id == "b1"
    assert task.is_complete is True
    assert story.is_complete is True
    assert root.is_complete is False
    assert bug.is_complete is False


def test_simple_workflow_template_survives_data_and_new_children(initialized_fastworkflow):
    """Application state added after startup has to come back too."""
    folderpath = EXAMPLES / "simple_workflow_template"
    workflow = make_workflow(folderpath)

    run_command(workflow, "startup", "startup")
    run_command(workflow, "WorkItem/go_to_workitem", "go to the story",
                path="/Story[index=0]")
    run_command(workflow, "WorkItem/add_child_workitem", "add task t9",
                workitem_type="Task", id="t9")
    workflow.current_command_context["notes"] = "carried across the round trip"

    _, record = project(workflow)

    restored = make_workflow(folderpath)
    assert serialization_hooks.restore_command_contexts(restored, record)

    story = restored.root_command_context.get_workitem("/Story[index=0]")
    assert [child.id for child in story._children] == [None, "t9"]
    assert story["notes"] == "carried across the round trip"


# ---------------------------------------------------------------------------
# todo_list_workflow — root is a TodoListManager, current descends into it
# ---------------------------------------------------------------------------

def test_todo_list_workflow_round_trip(initialized_fastworkflow, todo_list_json_preserved):
    folderpath = TESTS / "todo_list_workflow"
    workflow = make_workflow(folderpath)

    run_command(workflow, "startup", "startup")
    run_command(workflow, "TodoListManager/create_todo_list", "new list chores",
                description="chores")
    run_command(workflow, "TodoList/add_child_todolist", "add kitchen",
                description="kitchen", assign_to="Sam", is_complete=False)
    run_command(workflow, "TodoList/add_child_todoitem", "add laundry",
                description="laundry", assign_to="Sam", is_complete=False)

    live_root = workflow.root_command_context
    assert workflow.current_command_context is not live_root

    projection, record = project(workflow)
    assert projection.anchor_is_root
    assert projection.current_locator == "2/1/1"
    assert projection.response_locator == "2/1"

    restored = make_workflow(folderpath)
    assert serialization_hooks.restore_command_contexts(restored, record)

    manager = restored.root_command_context
    assert manager is not live_root
    assert {id_: lst.description for id_, lst in manager.lists.items()} == {
        1: "groceries",
        2: "chores",
    }

    chores = manager.get_todo_list(2)
    kitchen = chores.get_child_by_id(1)
    laundry = kitchen.get_child_by_id(1)

    # Identity, not equality: the slots must be nodes of the restored hierarchy.
    assert restored.current_command_context is laundry
    assert restored.command_context_for_response_generation is kitchen

    assert laundry.parent is kitchen
    assert kitchen.parent is chores
    assert chores.parent is manager
    assert laundry.description == "laundry"
    assert laundry.assign_to == "Sam"


def test_todo_list_workflow_restore_ignores_the_json_file(
    initialized_fastworkflow, todo_list_json_preserved
):
    """The snapshot wins over the file, which lags every unsaved mutation."""
    folderpath = TESTS / "todo_list_workflow"
    workflow = make_workflow(folderpath)

    run_command(workflow, "startup", "startup")
    run_command(workflow, "TodoListManager/create_todo_list", "new list chores",
                description="chores")
    # add_child_todoitem does not save, so this item exists only in memory.
    run_command(workflow, "TodoList/add_child_todoitem", "add laundry",
                description="laundry", assign_to="Sam", is_complete=False)

    _, record = project(workflow)

    restored = make_workflow(folderpath)
    assert serialization_hooks.restore_command_contexts(restored, record)

    chores = restored.root_command_context.get_todo_list(2)
    assert [child.description for child in chores.children] == ["laundry"]


# ---------------------------------------------------------------------------
# messaging_app_4 — root is a ChatRoom, current moves to one of its users
# ---------------------------------------------------------------------------

def test_messaging_app_4_round_trip(initialized_fastworkflow):
    folderpath = EXAMPLES / "messaging_app_4"
    workflow = make_workflow(folderpath)

    run_command(workflow, "set_root_context", "start the chat room")
    run_command(workflow, "ChatRoom/add_user", "add Alice",
                user_name="Alice", is_premium_user=False)
    run_command(workflow, "ChatRoom/add_user", "add Bob Smith",
                user_name="Bob Smith", is_premium_user=True)
    run_command(workflow, "ChatRoom/set_current_user", "Bob Smith is the current user",
                user_name="Bob Smith")

    live_root = workflow.root_command_context
    assert workflow.current_command_context is not live_root

    projection, record = project(workflow)
    assert projection.anchor_is_root
    assert projection.current_locator == "Bob Smith"
    assert projection.response_locator == serialization_hooks.ANCHOR_LOCATOR

    restored = make_workflow(folderpath)
    assert serialization_hooks.restore_command_contexts(restored, record)

    chatroom = restored.root_command_context
    assert chatroom is not live_root
    assert chatroom.list_users() == ["Alice", "Bob Smith"]

    bob = restored.current_command_context
    # Identity, not equality: current, the room's members and its current_user
    # all have to be the same object.
    assert bob is chatroom.users[1]
    assert chatroom.current_user is bob
    assert bob.chatroom is chatroom
    assert restored.command_context_for_response_generation is chatroom

    # The subclass decides which commands the context exposes, so it must survive.
    assert type(bob).__name__ == "PremiumUser"
    assert type(chatroom.users[0]).__name__ == "User"


# ---------------------------------------------------------------------------
# messaging_app_2 — root is a User and current never leaves it
# ---------------------------------------------------------------------------

def test_messaging_app_2_round_trip(initialized_fastworkflow):
    folderpath = EXAMPLES / "messaging_app_2"
    workflow = make_workflow(folderpath)

    run_command(workflow, "startup", "login as Carol", name="Carol")
    run_command(workflow, "User/send_message", "message jane",
                to="jane.doe@xyz.edu", message="hello there")

    live_root = workflow.root_command_context
    assert workflow.current_command_context is live_root

    projection, record = project(workflow)
    assert projection.anchor_is_root
    assert projection.current_locator == serialization_hooks.ANCHOR_LOCATOR
    assert projection.response_locator == serialization_hooks.ANCHOR_LOCATOR

    restored = make_workflow(folderpath)
    assert serialization_hooks.restore_command_contexts(restored, record)

    user = restored.root_command_context
    assert user is not live_root
    assert user.name == "Carol"
    assert restored.current_command_context is user
    assert restored.command_context_for_response_generation is user


# ---------------------------------------------------------------------------
# messaging_app_3 — no root at all; the current context is the anchor
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "is_premium_user, expected_context",
    [(False, "User"), (True, "PremiumUser")],
)
def test_messaging_app_3_round_trip(initialized_fastworkflow, is_premium_user, expected_context):
    folderpath = EXAMPLES / "messaging_app_3"
    workflow = make_workflow(folderpath)

    run_command(workflow, "initialize_user", "login as Dave",
                name="Dave", is_premium_user=is_premium_user)
    # PremiumUser inherits this command, so it stays qualified under User.
    run_command(workflow, "User/send_message", "message jane",
                to="jane.doe@xyz.edu", message="hello there")

    live_current = workflow.current_command_context
    assert workflow.root_command_context is None

    projection, record = project(workflow)
    assert projection.anchor_is_root is False
    assert projection.context_name == expected_context
    assert projection.current_locator == serialization_hooks.ANCHOR_LOCATOR

    restored = make_workflow(folderpath)
    assert serialization_hooks.restore_command_contexts(restored, record)

    user = restored.current_command_context
    assert user is not live_current
    assert user.name == "Dave"
    assert type(user).__name__ == expected_context
    # A workflow that only ever set a current context must not acquire a root.
    assert restored.root_command_context is None


# ---------------------------------------------------------------------------
# The trichotomy: an absent hook is not the same as a declared no-op
# ---------------------------------------------------------------------------

class Widget:
    """Anchor whose context class never implemented get_state."""


class Gadget:
    """Anchor whose context class declared its state ephemeral."""


NO_HOOK_CONTEXT = '''
class Context:
    @classmethod
    def get_parent(cls, command_context_object) -> None:
        return None
'''

EPHEMERAL_CONTEXT = '''
import fastworkflow


class Context:
    state_version = 1

    @classmethod
    def get_parent(cls, command_context_object) -> None:
        return None

    @classmethod
    def get_state(cls, command_context_object):
        return fastworkflow.EPHEMERAL
'''

POKE_COMMAND = '''
import fastworkflow


class ResponseGenerator:
    def __call__(self, workflow, command) -> fastworkflow.CommandOutput:
        return fastworkflow.CommandOutput(
            command_responses=[fastworkflow.CommandResponse(response="poked")]
        )
'''


def write_scratch_workflow(tmp_path: Path) -> Path:
    """A workflow whose two contexts sit on opposite sides of the trichotomy.

    Written here rather than copied from a bundled workflow so that no trained
    ___command_info comes with it; nothing in it is ever trained or routed
    through intent detection.
    """
    folderpath = tmp_path / "scratch_workflow"
    commands = folderpath / "_commands"
    for context_name, context_body in (("Widget", NO_HOOK_CONTEXT),
                                       ("Gadget", EPHEMERAL_CONTEXT)):
        context_folder = commands / context_name
        context_folder.mkdir(parents=True)
        (context_folder / f"_{context_name}.py").write_text(context_body)
        (context_folder / "poke.py").write_text(POKE_COMMAND)
    (commands / "context_inheritance_model.json").write_text("{}")
    return folderpath


def test_context_without_get_state_pins_and_warns_once(initialized_fastworkflow, tmp_path):
    workflow = make_workflow(write_scratch_workflow(tmp_path))
    workflow.root_command_context = Widget()

    projection = serialization_hooks.project_command_contexts(workflow)
    assert projection.outcome is ProjectionOutcome.HOOK_ABSENT
    assert not projection.is_checkpointable
    assert projection.context_name == "Widget"

    with captured_warnings() as records:
        serialization_hooks.warn_if_pinned(workflow, projection)
        serialization_hooks.warn_if_pinned(workflow, projection)

    assert len(records) == 1
    message = records[0].getMessage()
    assert "Widget" in message
    assert "fastworkflow.EPHEMERAL" in message


def test_context_declaring_ephemeral_pins_without_warning(initialized_fastworkflow, tmp_path):
    workflow = make_workflow(write_scratch_workflow(tmp_path))
    workflow.root_command_context = Gadget()

    projection = serialization_hooks.project_command_contexts(workflow)
    assert projection.outcome is ProjectionOutcome.DECLARED_EPHEMERAL
    assert not projection.is_checkpointable
    assert projection.context_name == "Gadget"

    with captured_warnings() as records:
        serialization_hooks.warn_if_pinned(workflow, projection)

    assert records == []

    # Nothing was projected, so nothing is restored — silently, by design.
    restored = make_workflow(workflow.folderpath)
    assert serialization_hooks.restore_command_contexts(restored, projection.as_record()) is False
    assert restored.root_command_context is None

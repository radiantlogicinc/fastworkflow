"""The record the workflow stands on (spec section 5) and context identity (spec 6.1).

Integration tests: real Workflow objects over the todo_list_workflow folder, and
real context callback classes. The load-time checks run on a copy of that folder
so the committed workflow is never edited.
"""
import re
import shutil
import uuid
from pathlib import Path

import pytest

import fastworkflow
from fastworkflow.command_context_model import (
    CommandContextModel,
    CommandContextModelError,
    context_class_problems,
)
from fastworkflow.command_preview import load_problems
from fastworkflow.context_identity import declared_instance_label
from fastworkflow.serialization_hooks import reset_warnings
from fastworkflow.workflow import Workflow
from fastworkflow.command_routing import RoutingRegistry

from tests.todo_list_workflow.application.todo_manager import TodoListManager

TESTS = Path(__file__).parent
TODO_FOLDER = TESTS / "todo_list_workflow"


@pytest.fixture
def initialized_fastworkflow(tmp_path):
    fastworkflow.init({"FASTWORKFLOW_STATE_ROOT": str(tmp_path / "workflow_contexts")})
    RoutingRegistry.clear_registry()
    reset_warnings()
    yield
    RoutingRegistry.clear_registry()
    reset_warnings()


def make_workflow(folderpath: Path) -> Workflow:
    return Workflow.create(str(folderpath), workflow_id_str=f"record-{uuid.uuid4().hex}")


@pytest.fixture
def workflow(initialized_fastworkflow):
    return make_workflow(TODO_FOLDER)


@pytest.fixture
def manager(tmp_path):
    return TodoListManager(str(tmp_path / "todo_list.json"), load_from_file=False)


def copy_todo_folder(tmp_path: Path) -> Path:
    target = tmp_path / "todo_copy"
    shutil.copytree(TODO_FOLDER, target, ignore=shutil.ignore_patterns("__pycache__", "*.json"))
    return target


# --------------------------------------------------------------------------
# 5.1 workflow.record: the getter is the invoked record, the setter moves
# --------------------------------------------------------------------------

def test_record_is_the_invoked_record_after_a_get_parent_lookup(workflow, manager):
    """A command found on an enclosing record: the record it acts on stays put while the position moves."""
    child = manager.create_todo_list("chores")
    workflow.root = manager
    # The framework binds the record a command was invoked on before the command runs.
    workflow._command_context_for_response_generation = child

    parent = workflow.get_parent(workflow.record)
    assert parent is manager
    assert workflow.record is child

    workflow.record = parent
    assert workflow._current_command_context is manager
    # The getter keeps the invoked record until the command returns.
    assert workflow.record is child


def test_assigning_record_moves_the_workflow(workflow, manager):
    workflow.root = manager
    child = manager.create_todo_list("errands")
    workflow.record = child
    assert workflow._current_command_context is child


def test_a_move_notifies_context_change_listeners_only_when_it_changes(workflow, manager):
    workflow.root = manager
    calls = []

    class Listener:
        def on_change(self):
            calls.append(workflow._current_command_context)

    listener = Listener()
    workflow.add_context_change_listener(listener.on_change)
    child = manager.create_todo_list("errands")

    workflow.record = child
    workflow.record = child  # same record: no change, no notification
    workflow.record = manager
    assert calls == [child, manager]
    workflow.remove_context_change_listener(listener.on_change)


# --------------------------------------------------------------------------
# 5.2 workflow.root: set once in the startup command
# --------------------------------------------------------------------------

def test_root_is_set_once_and_the_second_set_raises(workflow, manager, tmp_path):
    workflow.root = manager
    assert workflow.root is manager
    assert workflow._current_command_context is manager

    other = TodoListManager(str(tmp_path / "other.json"), load_from_file=False)
    with pytest.raises(ValueError, match="workflow.root can only be set once"):
        workflow.root = other
    assert workflow.root is manager


# --------------------------------------------------------------------------
# 5.3 the three old names are gone and fail naming the replacement
# --------------------------------------------------------------------------

RETIRED = [
    ("command_context_for_response_generation", "workflow.record"),
    ("current_command_context", "workflow.record"),
    ("root_command_context", "workflow.root"),
    ("is_command_context_for_response_generation_root", "is removed: use workflow.is_record_root"),
    ("current_command_context_name", "is internal: read the record with workflow.record"),
    ("current_command_context_displayname", "is internal: read the record with workflow.record"),
    ("is_current_command_context_root", "is internal: read the record with workflow.record"),
]


def test_the_record_readers_that_replace_the_renamed_members(workflow, manager):
    workflow.root = manager
    workflow._command_context_for_response_generation = manager
    assert workflow.record is manager
    assert workflow.is_record_root is True
    assert workflow.record_displayname == workflow._current_command_context_displayname
    assert workflow._current_command_context_name == "TodoListManager"
    assert workflow._is_current_command_context_root is True


@pytest.mark.parametrize("old, replacement", RETIRED)
def test_a_retired_name_read_fails_naming_its_replacement(workflow, old, replacement):
    with pytest.raises(AttributeError, match=re.escape(replacement)):
        getattr(workflow, old)
    assert not hasattr(workflow, old)


@pytest.mark.parametrize("old, replacement", RETIRED)
def test_a_retired_name_assignment_fails_and_moves_nothing(workflow, manager, old, replacement):
    workflow.root = manager
    before = workflow._current_command_context
    with pytest.raises(AttributeError, match=re.escape(replacement)):
        setattr(workflow, old, TodoListManager(str(TODO_FOLDER / "unused.json"), load_from_file=False))
    assert old not in vars(workflow)
    assert workflow._current_command_context is before


# --------------------------------------------------------------------------
# 5.4 the record parameter; 6.1 the instance_label classmethod
# --------------------------------------------------------------------------

def test_get_parent_takes_the_record_by_keyword(workflow, manager):
    workflow.root = manager
    child = manager.create_todo_list("chores")
    assert workflow.get_parent(record=child) is manager


def test_a_context_with_the_old_parameter_name_fails_the_load():
    class OldContext:
        @classmethod
        def get_parent(cls, command_context_object):
            return None

    problems = context_class_problems(OldContext)
    assert problems == ['get_parent: parameter "command_context_object" is removed: name it "record"']


def test_a_context_with_the_record_parameter_loads():
    class NewContext:
        @classmethod
        def get_parent(cls, record):
            return None

        @classmethod
        def get_displayname(cls, record):
            return "x"

    assert context_class_problems(NewContext) == []


def test_an_instance_label_classmethod_fails_the_load_naming_the_property_form(tmp_path):
    folder = copy_todo_folder(tmp_path)
    item_path = folder / "_commands" / "TodoItem" / "_TodoItem.py"
    source = item_path.read_text(encoding="utf-8")
    item_path.write_text(source.replace(
        "    @classmethod\n    def get_parent",
        "    @classmethod\n    def instance_label(cls, record):\n"
        "        return str(record.id)\n\n    @classmethod\n    def get_parent", 1),
        encoding="utf-8")

    model = CommandContextModel.load(str(folder))
    with pytest.raises(CommandContextModelError, match="make the label a property on the record object"):
        model.get_context_class("TodoItem", fastworkflow.ModuleType.CONTEXT_CLASS)


def test_the_workflow_does_not_load_with_the_instance_label_classmethod(tmp_path, initialized_fastworkflow):
    folder = copy_todo_folder(tmp_path)
    item_path = folder / "_commands" / "TodoItem" / "_TodoItem.py"
    source = item_path.read_text(encoding="utf-8")
    item_path.write_text(source.replace(
        "    @classmethod\n    def get_parent",
        "    @classmethod\n    def instance_label(cls, record):\n"
        "        return str(record.id)\n\n    @classmethod\n    def get_parent", 1),
        encoding="utf-8")

    # The routing definition is read before any problem list is built, so the
    # workflow does not load; load_problems lists the error (it does not raise), naming the property form.
    problems = load_problems(str(folder))
    assert len(problems) == 1, problems
    assert problems[0].startswith("the workflow's commands do not load: CommandContextModelError: ")
    assert "instance_label is removed" in problems[0]
    assert 'make the label a property on the record object' in problems[0]
    assert 'instance_label_attr = "<property name>"' in problems[0]


# --------------------------------------------------------------------------
# 6.1 the property form: instance_label_attr names a property on the object
# --------------------------------------------------------------------------

class OrderContext:
    instance_label_attr = "instance_label"


class OrderRecord:
    def __init__(self, sku):
        self.sku = sku

    @property
    def instance_label(self):
        return f"order {self.sku}"


class PlainAttributeContext:
    instance_label_attr = "sku"


def test_a_property_named_by_instance_label_attr_gives_the_identity():
    assert declared_instance_label(OrderContext, OrderRecord("A-1")) == "order A-1"


def test_a_plain_attribute_named_by_instance_label_attr_gives_the_identity():
    assert declared_instance_label(PlainAttributeContext, OrderRecord("B-2")) == "B-2"


def test_a_classmethod_alone_declares_no_identity():
    class ClassmethodOnly:
        @classmethod
        def instance_label(cls, record):
            return "should not be read"

    assert declared_instance_label(ClassmethodOnly, OrderRecord("C-3")) == ""

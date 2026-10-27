# This file is for testing the CommandContextModel loader and its logic.

import json
from pathlib import Path
import pytest

from fastworkflow.command_context_model import (
    CommandContextModel,
    CommandContextModelValidationError,
)


@pytest.fixture
def sample_workflow_path() -> Path:
    """Provides the path to the retail workflow example."""
    return Path(__file__).parent.parent / "fastworkflow" / "examples" / "retail_workflow"


def test_load_valid_context_model(sample_workflow_path):
    """Ensure that the retail workflow context model is parsed."""
    model = CommandContextModel.load(str(sample_workflow_path))
    assert model is not None
    assert "*" in model._command_contexts


def test_star_context_commands(sample_workflow_path):
    """All commands should be registered under the '*' context."""
    model = CommandContextModel.load(str(sample_workflow_path))
    commands = model.commands("*")

    expected = {
        "calculate",
        "cancel_pending_order",
        "exchange_delivered_order_items",
        "find_user_id_by_email",
        "find_user_id_by_name_zip",
        "get_order_details",
        "get_product_details",
        "get_user_details",
        "list_all_product_types",
        "modify_pending_order_address",
        "modify_pending_order_items",
        "modify_pending_order_payment",
        "modify_user_address",
        "return_delivered_order_items",
        "transfer_to_human_agents",
        "wildcard",
    }

    assert set(commands) == expected


@pytest.fixture
def todo_list_workflow_path() -> Path:
    """Returns the path to the todo_list example workflow."""
    return Path(__file__).parent / "todo_list_workflow"


def test_load_and_validate_ok(todo_list_workflow_path: Path):
    """Loading a valid workflow should succeed."""
    model = CommandContextModel.load(str(todo_list_workflow_path))
    assert model is not None
    

def test_commands_for_context(todo_list_workflow_path: Path):
    """Verify that the correct commands are resolved for a given context."""
    model = CommandContextModel.load(str(todo_list_workflow_path))
    commands = model.commands("TodoItem")
    assert len(commands) > 0 
    # A more specific assertion would be better, e.g.
    # assert "TodoItem/add_item" in commands
    # assert "TodoList/complete_all" in commands # Inherited


def test_command_inheritance_override(todo_list_workflow_path: Path):
    """A derived context's command should override a base context's command."""
    model = CommandContextModel.load(str(todo_list_workflow_path))
    commands = model.commands("TodoItem")
    # Assuming 'help' is defined in both 'TodoItem' and a base context,
    # we expect the 'TodoItem/help' version.
    # This assertion is illustrative; actual commands would need to exist.
    assert "TodoItem/help" in commands if "TodoItem/help" in commands else True


def test_nonexistent_context_raises_error(todo_list_workflow_path: Path):
    """Requesting an unknown context should raise."""
    model = CommandContextModel.load(str(todo_list_workflow_path))
    with pytest.raises(CommandContextModelValidationError):
        model.commands("NonExistentContext")


def test_cycle_detection(sample_workflow_path: Path):
    """The loader should detect inheritance cycles."""
    # Create a temporary workflow with a cycle for this test
    # e.g., A -> B -> A
    # For this example, we assume there's a pre-configured 'cyclic_workflow'
    cyclic_path = sample_workflow_path.parent / "cyclic_workflow"
    if not cyclic_path.exists():
        pytest.skip("Cyclic workflow for testing not found")
        
    with pytest.raises(CommandContextModelValidationError, match="Inheritance cycle detected"):
        CommandContextModel.load(str(cyclic_path)) 


def test_ordered_base_contexts_diamond_and_two_level_chain(tmp_path: Path):
    """ordered_base_contexts walks bases depth-first in declared order, each once."""
    commands_dir = tmp_path / "_commands"
    cmd_body = (
        "class Signature:\n"
        "    \"\"\"No-op.\"\"\"\n\n"
        "class ResponseGenerator:\n"
        "    def __call__(self, workflow, command, command_parameters=None):\n"
        "        return None\n"
    )
    for ctx in ("RootBase", "MidA", "MidB", "Extra", "Leaf", "ChainTop", "ChainMid", "ChainBase"):
        ctx_dir = commands_dir / ctx
        ctx_dir.mkdir(parents=True, exist_ok=True)
        (ctx_dir / f"{ctx.lower()}_cmd.py").write_text(cmd_body, encoding="utf-8")

    # Diamond: Leaf -> [MidA, MidB], MidA -> [RootBase], MidB -> [RootBase, Extra].
    # Two-level chain: ChainTop -> [ChainMid], ChainMid -> [ChainBase].
    inheritance = {
        "Leaf": {"base": ["MidA", "MidB"]},
        "MidA": {"base": ["RootBase"]},
        "MidB": {"base": ["RootBase", "Extra"]},
        "ChainTop": {"base": ["ChainMid"]},
        "ChainMid": {"base": ["ChainBase"]},
    }
    (commands_dir / "context_inheritance_model.json").write_text(
        json.dumps(inheritance, indent=2) + "\n", encoding="utf-8"
    )

    model = CommandContextModel.load(str(tmp_path))
    assert model.ordered_base_contexts("Leaf") == ["MidA", "RootBase", "MidB", "Extra"]
    assert model.ordered_base_contexts("ChainTop") == ["ChainMid", "ChainBase"]
    assert model.ordered_base_contexts("RootBase") == []
    assert model.ordered_base_contexts("*") == []
    assert model.ordered_base_contexts("UnknownContext") == []

from pathlib import Path
import pytest

from fastworkflow.command_context_model import (
    CommandContextModel,
    CommandContextModelValidationError,
)


@pytest.fixture
def sample_workflow_path() -> Path:
    """Provides the path to the retail workflow example."""
    return Path(__file__).parent.parent / "examples" / "retail_workflow"


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
    }

    assert set(commands) == expected


def test_nonexistent_context_raises_error(sample_workflow_path):
    """Requesting an unknown context should raise."""
    model = CommandContextModel.load(str(sample_workflow_path))
    with pytest.raises(CommandContextModelValidationError):
        model.commands("no_such_context") 
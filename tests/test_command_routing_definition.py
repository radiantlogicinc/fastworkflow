import pytest
from pathlib import Path
from pydantic import BaseModel

from fastworkflow.command_routing_definition import CommandRoutingRegistry, CommandRoutingDefinition, ModuleType


@pytest.fixture(scope="module")
def sample_workflow_path() -> Path:
    """Provides the path to the sample workflow example."""
    return Path(__file__).parent.parent / "examples" / "retail_workflow"


@pytest.fixture(scope="function")
def command_routing_definition(sample_workflow_path: Path) -> CommandRoutingDefinition:
    """
    Builds and returns a command routing definition for the sample workflow.
    Uses function scope and clears the registry to ensure test isolation.
    """
    CommandRoutingRegistry.clear_registry()
    return CommandRoutingRegistry.get_definition(str(sample_workflow_path))


class TestCommandRoutingDefinition:
    """Test suite for the refactored CommandRoutingDefinition and CommandRoutingRegistry."""

    def test_definition_creation_and_caching(self, sample_workflow_path: Path):
        """Tests that the CommandRoutingDefinition is created and properly cached."""
        CommandRoutingRegistry.clear_registry()
        
        definition1 = CommandRoutingRegistry.get_definition(str(sample_workflow_path))
        assert definition1 is not None
        assert isinstance(definition1, CommandRoutingDefinition)
        
        # Calling get_definition again for the same path should return the cached instance
        definition2 = CommandRoutingRegistry.get_definition(str(sample_workflow_path))
        assert definition1 is definition2

    def test_get_command_names_for_valid_context(self, command_routing_definition: CommandRoutingDefinition):
        """Tests that command names are correctly retrieved for a valid, inherited context."""
        context = "*"
        command_names = command_routing_definition.get_command_names(context)
        
        expected_commands = {
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
            "Core/misunderstood_intent",
        }
        assert set(command_names) == expected_commands

    def test_get_command_names_for_invalid_context(self, command_routing_definition: CommandRoutingDefinition):
        """Tests that requesting command names for an invalid context raises a ValueError."""
        with pytest.raises(ValueError, match="Context 'invalid_context' not found"):
            command_routing_definition.get_command_names("invalid_context")

    def test_get_command_class_for_parameters(self, command_routing_definition: CommandRoutingDefinition):
        """Tests getting the command parameters class (e.g., Signature.Input)."""
        command_name = "find_user_id_by_email"
        
        param_class = command_routing_definition.get_command_class(
            command_name, ModuleType.COMMAND_PARAMETERS_CLASS
        )
        
        assert param_class is not None
        assert issubclass(param_class, BaseModel)
        assert "email" in param_class.model_fields

    def test_get_command_class_for_response_generator(self, command_routing_definition: CommandRoutingDefinition):
        """Tests getting the response generator class for a command."""
        command_name = "find_user_id_by_email"

        rg_class = command_routing_definition.get_command_class(
            command_name, ModuleType.RESPONSE_GENERATION_INFERENCE
        )

        assert rg_class is not None
        assert hasattr(rg_class, "__call__")
        assert rg_class.__name__ == "ResponseGenerator"

    def test_get_command_class_for_nonexistent_command_file(self, command_routing_definition: CommandRoutingDefinition):
        """
        Tests that requesting a class for a command that is NOT in the command directory
        (e.g., has no corresponding .py file) returns None.
        """
        command_name = "this_command_truly_does_not_exist"
        
        param_class = command_routing_definition.get_command_class(
            command_name, ModuleType.COMMAND_PARAMETERS_CLASS
        )
        
        assert param_class is None 
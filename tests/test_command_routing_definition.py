import os
import pytest
import shutil
import fastworkflow
from fastworkflow.command_routing_definition import CommandRoutingDefinition, CommandRoutingRegistry, ModuleType
from pydantic import BaseModel

@pytest.fixture(scope="module")
def retail_workflow_path():
    """Get the path to the retail workflow example."""
    return os.path.join(os.path.dirname(__file__), "..", "examples", "retail_workflow")

@pytest.fixture(scope="module")
def command_routing_definition(retail_workflow_path):
    """Create and return a command routing definition for the retail workflow."""
    # Ensure command info is cleared before running tests for this module
    command_info_path = os.path.join(retail_workflow_path, "___command_info")
    if os.path.exists(command_info_path):
        shutil.rmtree(command_info_path)
        
    return CommandRoutingRegistry.create_definition(retail_workflow_path)


class TestCommandRoutingDefinition:
    def test_definition_creation(self, command_routing_definition):
        """Test that the CommandRoutingDefinition is created correctly."""
        assert command_routing_definition is not None
        assert isinstance(command_routing_definition, CommandRoutingDefinition)

    def test_get_command_names(self, command_routing_definition):
        """Test that command names are correctly discovered."""
        # The workitem path is derived from the folder structure.
        workitem_path = "/retail_workflow"
        command_names = command_routing_definition.get_command_names(workitem_path)

        assert "cancel_pending_order" in command_names
        assert "get_user_details" in command_names
        assert "list_all_product_types" in command_names

    def test_get_command_class_for_parameters(self, command_routing_definition):
        """Test getting the command parameters class (Signature.Input)."""
        workitem_path = "/retail_workflow"
        command_name = "cancel_pending_order"
        
        param_class = command_routing_definition.get_command_class(
            workitem_path, command_name, ModuleType.COMMAND_PARAMETERS_CLASS
        )
        
        assert param_class is not None
        assert issubclass(param_class, BaseModel)
        # Check a field to be sure
        assert "order_id" in param_class.model_fields

    def test_get_command_class_for_response_generator(self, command_routing_definition):
        """Test getting the response generator class."""
        workitem_path = "/retail_workflow"
        command_name = "cancel_pending_order"

        rg_class = command_routing_definition.get_command_class(
            workitem_path, command_name, ModuleType.RESPONSE_GENERATION_INFERENCE
        )

        assert rg_class is not None
        assert hasattr(rg_class, "__call__")
        assert rg_class.__name__ == "ResponseGenerator"
        
    def test_get_command_class_for_input_signature(self, command_routing_definition):
        """Test getting the input signature class (Signature)."""
        workitem_path = "/retail_workflow"
        command_name = "cancel_pending_order"

        sig_class = command_routing_definition.get_command_class(
            workitem_path, command_name, ModuleType.INPUT_FOR_PARAM_EXTRACTION_CLASS
        )

        assert sig_class is not None
        assert hasattr(sig_class, "Input")
        assert hasattr(sig_class, "Output")
        assert sig_class.__name__ == "Signature" 
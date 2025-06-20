import os
import pytest
from fastworkflow.command_directory import CommandDirectory, CommandMetadata

@pytest.fixture(scope="module")
def sample_workflow_path():
    """Get the path to the sample workflow example."""
    # Construct an absolute path to the sample_workflow directory
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "examples", "retail_workflow"))

class TestCommandDirectory:
    def test_load_command_directory(self, sample_workflow_path):
        """Tests the direct loading of the CommandDirectory."""
        command_directory = CommandDirectory.load(sample_workflow_path)

        assert command_directory is not None
        assert isinstance(command_directory, CommandDirectory)
        assert command_directory.workflow_folderpath == sample_workflow_path

    def test_get_commands(self, sample_workflow_path):
        """Tests that the command directory finds all commands."""
        command_directory = CommandDirectory.load(sample_workflow_path)
        command_keys = set(command_directory.get_commands())
        
        # Based on the file structure of sample_workflow/_commands
        # and the Core/misunderstood_intent command from the internal workflow
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
        # The commands are now stored with their context prefix for core commands
        assert command_keys == expected_commands

    def test_get_command_metadata(self, sample_workflow_path):
        """Tests that the metadata for a specific command is correct."""
        command_directory = CommandDirectory.load(sample_workflow_path)
        metadata = command_directory.get_command_metadata("list_all_product_types")

        assert isinstance(metadata, CommandMetadata)

        expected_module_path = os.path.join(sample_workflow_path, "_commands", "list_all_product_types.py")
        assert os.path.abspath(metadata.parameter_extraction_signature_module_path) == expected_module_path
        assert os.path.abspath(metadata.response_generation_module_path) == expected_module_path

        assert metadata.command_parameters_class == "Signature.Input"
        assert metadata.input_for_param_extraction_class == "Signature"
        assert metadata.response_generation_class_name == "ResponseGenerator" 
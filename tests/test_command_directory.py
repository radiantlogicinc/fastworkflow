import os
import pytest
import fastworkflow
from fastworkflow.command_directory import CommandDirectory, CommandSource, CommandMetadata

@pytest.fixture(scope="module")
def retail_workflow_path():
    """Get the path to the retail workflow example."""
    return os.path.join(os.path.dirname(__file__), "..", "examples", "retail_workflow")

class TestCommandDirectory:
    def test_create_and_load_command_directory(self, retail_workflow_path):
        """Tests the creation and loading of the CommandDirectory for a workflow."""
        # The command directory is created as part of the command routing definition.
        # This will create and save the command_directory.json
        fastworkflow.CommandRoutingRegistry.create_definition(retail_workflow_path)

        # Now, load it directly
        command_directory = CommandDirectory.load(retail_workflow_path)

        assert command_directory is not None
        assert isinstance(command_directory, CommandDirectory)
        assert os.path.abspath(command_directory.workflow_folderpath) == os.path.abspath(retail_workflow_path)

        # Check for some expected commands
        command_keys = command_directory.get_command_keys()
        assert "_base_commands/cancel_pending_order" in command_keys
        assert "_base_commands/get_user_details" in command_keys

        # Check metadata for a specific command
        metadata = command_directory.get_command_metadata("_base_commands/cancel_pending_order")
        assert isinstance(metadata, CommandMetadata)
        assert metadata.command_source == CommandSource.BASE_COMMANDS

        expected_module_path = os.path.join(retail_workflow_path, "_base_commands", "cancel_pending_order.py")
        assert metadata.parameter_extraction_signature_module_path == expected_module_path
        assert metadata.response_generation_module_path == expected_module_path

        assert metadata.command_parameters_class == "Signature.Input"
        assert metadata.input_for_param_extraction_class == "Signature"
        assert metadata.response_generation_class_name == "ResponseGenerator" 
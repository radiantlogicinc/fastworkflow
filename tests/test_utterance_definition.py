import pytest
from pathlib import Path

from fastworkflow.utterance_definition import UtteranceDefinition
from fastworkflow.command_directory import CommandDirectory
from fastworkflow.command_context_model import CommandContextModel
from fastworkflow.command_directory import UtteranceMetadata


@pytest.fixture(scope="module")
def sample_workflow_path() -> Path:
    """Get the path to the sample workflow example."""
    return Path(__file__).parent.parent / "examples" / "retail_workflow"


@pytest.fixture(scope="module")
def utterance_definition(sample_workflow_path: Path) -> UtteranceDefinition:
    """
    Builds and returns an utterance definition for the sample workflow.
    This now directly instantiates the required components.
    """
    context_model = CommandContextModel.load(str(sample_workflow_path))
    command_directory = CommandDirectory.load(str(sample_workflow_path))
    return UtteranceDefinition(
        workflow_folderpath=str(sample_workflow_path),
        context_model=context_model,
        command_directory=command_directory,
    )


class TestUtteranceDefinition:
    """Test suite for the refactored UtteranceDefinition."""

    def test_definition_creation(self, utterance_definition: UtteranceDefinition):
        """Test that the UtteranceDefinition is created correctly."""
        assert utterance_definition is not None
        assert isinstance(utterance_definition, UtteranceDefinition)
        assert isinstance(utterance_definition.context_model, CommandContextModel)
        assert isinstance(utterance_definition.command_directory, CommandDirectory)

    def test_get_command_names(self, utterance_definition: UtteranceDefinition):
        """Test that command names are correctly retrieved for a given context."""
        context = "*"
        command_names = utterance_definition.get_command_names(context)
        
        # This context in `context_model.json` contains these commands.
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
        }
        assert set(command_names) == expected_commands

    def test_get_command_utterances(self, utterance_definition: UtteranceDefinition):
        """Test that utterances are correctly extracted for a specific command."""
        command_name = "find_user_id_by_email"
        utterance_metadata = utterance_definition.get_command_utterances(command_name)

        assert isinstance(utterance_metadata, UtteranceMetadata)
        
        assert isinstance(utterance_metadata.plain_utterances, list)
        assert "Can you tell me the ID linked to" in utterance_metadata.plain_utterances[0]
        
        assert utterance_metadata.generated_utterances_module_filepath.endswith("find_user_id_by_email.py")
        assert utterance_metadata.generated_utterances_func_name == "Signature.generate_utterances"

    def test_get_sample_utterances_from_context(self, utterance_definition: UtteranceDefinition):
        """Test that sample utterances can be retrieved for a whole context."""
        context = "*"
        sample_utterances = utterance_definition.get_sample_utterances(context)

        assert isinstance(sample_utterances, list)
        assert len(sample_utterances) > 0
        # The 'list_all_product_types' command should provide a sample about products.
        assert any("products" in utt.lower() or "product" in utt.lower() for utt in sample_utterances)

    def test_get_sample_utterances_handles_generated_utterances(self, utterance_definition: UtteranceDefinition):
        """
        Tests that get_sample_utterances correctly calls the generation function
        for commands that have one, like 'find_user_id_by_email'.
        """
        context = "*"
        sample_utterances = utterance_definition.get_sample_utterances(context)

        assert isinstance(sample_utterances, list)
        # Check for a dynamically generated utterance from find_user_id_by_email.py
        assert any("id linked" in utt.lower() or "email" in utt.lower() for utt in sample_utterances)

    def test_get_utterances_for_nonexistent_command(self, utterance_definition: UtteranceDefinition):
        """Tests that requesting utterances for a command not in the directory raises a ValueError."""
        with pytest.raises(ValueError, match="Could not find utterance metadata for command 'nonexistent_command'"):
            utterance_definition.get_command_utterances("nonexistent_command")

    def test_inheritance_in_sample_utterances(self, utterance_definition: UtteranceDefinition):
        """
        Tests that get_sample_utterances correctly includes utterances from inherited contexts.
        The 'cmdset1_specialized' context inherits from 'cmd_set_1'.
        """
        context = "*"
        sample_utterances = utterance_definition.get_sample_utterances(context)

        assert isinstance(sample_utterances, list)
        # Should include a sample about product types
        assert any("product" in utt.lower() for utt in sample_utterances) 
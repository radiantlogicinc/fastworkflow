import os
import pytest
import shutil
import fastworkflow
from fastworkflow.utterance_definition import UtteranceDefinition, UtteranceRegistry
from fastworkflow.command_directory import UtteranceMetadata

@pytest.fixture(scope="module")
def retail_workflow_path():
    """Get the path to the retail workflow example."""
    return os.path.join(os.path.dirname(__file__), "..", "examples", "retail_workflow")

@pytest.fixture(scope="module")
def utterance_definition(retail_workflow_path):
    """
    Creates the command routing and utterance definitions for the retail workflow.
    Returns the utterance definition.
    """
    # Ensure command info is cleared before running tests for this module
    command_info_path = os.path.join(retail_workflow_path, "___command_info")
    if os.path.exists(command_info_path):
        shutil.rmtree(command_info_path)

    # Creating utterance definition depends on command routing being created first.
    UtteranceRegistry.create_definition(retail_workflow_path)
    return UtteranceRegistry.get_definition(retail_workflow_path)


class TestUtteranceDefinition:

    def test_definition_creation(self, utterance_definition):
        """Test that the UtteranceDefinition is created correctly."""
        assert utterance_definition is not None
        assert isinstance(utterance_definition, UtteranceDefinition)

    def test_get_command_utterances(self, utterance_definition):
        # sourcery skip: class-extract-method
        """Test that utterances are correctly extracted from single-file commands."""
        workitem_path = "/retail_workflow"
        command_name = "cancel_pending_order"

        utterance_metadata = utterance_definition.get_command_utterances(workitem_path, command_name)

        assert isinstance(utterance_metadata, UtteranceMetadata)
        
        # Check plain utterances
        assert isinstance(utterance_metadata.plain_utterances, list)
        assert len(utterance_metadata.plain_utterances) > 0
        assert "Can you cancel my pending order?" in utterance_metadata.plain_utterances
        
        # Check template utterances (should be empty for this command)
        assert isinstance(utterance_metadata.template_utterances, list)
        
        # Check generated utterances function metadata
        assert utterance_metadata.generated_utterances_module_filepath.endswith("cancel_pending_order.py")
        assert utterance_metadata.generated_utterances_func_name == "Signature.generate_utterances"

    def test_get_generated_utterances_func(self, utterance_definition):
        """Test that the nested generate_utterances function can be retrieved."""
        workitem_path = "/retail_workflow"
        command_name = "cancel_pending_order"

        utterance_metadata = utterance_definition.get_command_utterances(workitem_path, command_name)
        
        gen_func = utterance_metadata.get_generated_utterances_func(utterance_definition.workflow_folderpath)
        
        assert callable(gen_func)

    def test_get_sample_utterances(self, utterance_definition):
        """Test that sample utterances can be retrieved."""
        workitem_path = "/retail_workflow"
        sample_utterances = utterance_definition.get_sample_utterances(workitem_path)

        assert isinstance(sample_utterances, list)
        assert len(sample_utterances) > 0
        # Check if one of the first utterances is present
        assert "I want to cancel my order because I no longer need it." in sample_utterances 
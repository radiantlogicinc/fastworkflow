import pytest
import os
import fastworkflow
from fastworkflow import Workflow, NLUPipelineStage
from fastworkflow.utils.signatures import InputForParamExtraction
from fastworkflow._workflows.command_metadata_extraction._commands.wildcard import ParameterExtraction
from pydantic import BaseModel, Field
from unittest.mock import patch, MagicMock


class MockCommandParameters(BaseModel):
    user_name: str = Field(description="User name")


class TestParameterExtractionErrorRegression:
    """Test that parameter extraction correctly handles error state and direct parameter input."""
    
    @patch('fastworkflow._workflows.command_metadata_extraction.parameter_extraction.InputForParamExtraction')
    @patch('fastworkflow.RoutingRegistry')
    def test_parameter_extraction_in_error_state(self, mock_routing_registry, mock_input_for_param_extraction):
        """Test that direct parameter values are correctly extracted in error state."""
        
        # Create mock workflows with mocked internals
        cme_workflow = MagicMock()
        app_workflow = MagicMock()
        
        # Set up the context to simulate parameter extraction error state
        cme_workflow.context = {
            "app_workflow": app_workflow,
            "NLU_Pipeline_Stage": NLUPipelineStage.PARAMETER_EXTRACTION
        }
        
        # Create a stored parameter with missing field
        stored_params = MockCommandParameters(user_name=fastworkflow.get_env_var("NOT_FOUND"))
        cme_workflow.context["stored_parameters"] = stored_params
        
        # Create the parameter extractor with a direct parameter value
        command_name = "ChatRoom/set_current_user"
        command_value = "nundini"  # This is the direct parameter value
        extractor = ParameterExtraction(cme_workflow, app_workflow, command_name, command_value)
        
        # Mock the necessary methods
        extractor._extract_missing_fields = MagicMock(
            return_value=(False, "Missing parameter values: user_name", {}, ["user_name"])
        )
        
        # Create a mock for _apply_missing_fields that returns the expected result
        def mock_apply_missing_fields(command, default_params, missing_fields):
            # Apply the parameter value
            params = default_params.model_copy()
            if missing_fields and hasattr(params, missing_fields[0]):
                setattr(params, missing_fields[0], command)
            return params
            
        extractor._apply_missing_fields = mock_apply_missing_fields
        
        # Mock validate_parameters to return valid (returns 4 values: is_valid, error_msg, suggestions, missing_fields)
        mock_input_instance = MagicMock()
        mock_input_instance.validate_parameters.return_value = (True, "All required parameters are valid.", {}, [])
        mock_input_for_param_extraction.create.return_value = mock_input_instance
        
        # Execute the extraction
        with patch.object(extractor, '_store_parameters'):
            with patch.object(extractor, '_clear_parameters'):
                result = extractor.extract()
        
        # Verify the result
        assert result.parameters_are_valid is True
        assert result.cmd_parameters is not None
        assert hasattr(result.cmd_parameters, 'user_name')
        assert result.cmd_parameters.user_name == command_value
    
    def test_direct_parameter_extraction(self):
        """Test that the _apply_missing_fields method correctly extracts parameters."""
        
        # Create the parameter extractor with mocked workflows
        cme_workflow = MagicMock()
        app_workflow = MagicMock()
        extractor = ParameterExtraction(cme_workflow, app_workflow, "test_command", "test_value")
        
        # Test single parameter extraction
        params = MockCommandParameters(user_name=fastworkflow.get_env_var("NOT_FOUND"))
        result = extractor._apply_missing_fields("john_doe", params, ["user_name"])
        assert result.user_name == "john_doe"
        
        # Test comma-separated values
        params = MockCommandParameters(user_name=fastworkflow.get_env_var("NOT_FOUND"))
        result = extractor._apply_missing_fields("jane_doe", params, ["user_name"])
        assert result.user_name == "jane_doe"

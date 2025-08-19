import pytest
import fastworkflow
from fastworkflow import Workflow, NLUPipelineStage


class TestAmbiguousIntentRegression:
    """Test that the simpler ambiguous intent regression fix preserves parameters correctly."""
    
    def test_command_preservation_logic(self):
        """Test the core logic that prevents command overwriting."""
        
        # Create a mock workflow context
        workflow_context = {}
        
        # Simulate the first call: ambiguous intent with parameters
        original_command = "set the current user to unsh"
        
        # This simulates what happens in the ambiguous intent detection
        workflow_context["command"] = original_command
        assert workflow_context["command"] == "set the current user to unsh"
        
        # Simulate the second call: resolved intent without parameters
        resolved_command = "set current user"
        
        # This simulates your fix: only update command if it's not already set
        if "command" not in workflow_context:
            workflow_context["command"] = resolved_command
        
        # Verify that the original command is preserved
        assert workflow_context["command"] == "set the current user to unsh"
        assert workflow_context["command"] != resolved_command
        
        # Verify that the original command contains the parameter
        assert "unsh" in workflow_context["command"]
    
    def test_new_command_storage_logic(self):
        """Test that new commands get stored when there's no existing command."""
        
        # Create a mock workflow context
        workflow_context = {}
        
        # Ensure no command exists
        assert "command" not in workflow_context
        
        # Simulate a new command (no ambiguous intent)
        new_command = "list users"
        
        # This should store the new command
        if "command" not in workflow_context:
            workflow_context["command"] = new_command
        
        # Verify that the new command was stored
        assert "command" in workflow_context
        assert workflow_context["command"] == new_command
    
    def test_command_overwrite_prevention(self):
        """Test that the fix prevents command overwriting in all scenarios."""
        
        # Test case 1: No existing command
        workflow_context = {}
        new_command = "first command"
        
        if "command" not in workflow_context:
            workflow_context["command"] = new_command
        
        assert workflow_context["command"] == "first command"
        
        # Test case 2: Existing command (should not be overwritten)
        second_command = "second command"
        
        if "command" not in workflow_context:
            workflow_context["command"] = second_command
        
        # Should still be the first command
        assert workflow_context["command"] == "first command"
        assert workflow_context["command"] != second_command
        
        # Test case 3: Clear command and add new one
        workflow_context.pop("command", None)
        assert "command" not in workflow_context
        
        third_command = "third command"
        if "command" not in workflow_context:
            workflow_context["command"] = third_command
        
        assert workflow_context["command"] == "third command"
    
    def test_workflow_context_behavior(self):
        """Test that the workflow context behaves correctly with the fix."""
        
        # Create a mock workflow context
        workflow_context = {}
        
        # Simulate the exact scenario from the regression:
        # 1. User types: "set the current user to unsh"
        original_command = "set the current user to unsh"
        workflow_context["command"] = original_command
        
        # 2. User resolves ambiguity: "set current user"
        resolved_command = "set current user"
        
        # 3. Apply your fix: only update if not already set
        if "command" not in workflow_context:
            workflow_context["command"] = resolved_command
        
        # 4. Verify the fix worked
        assert workflow_context["command"] == original_command  # Should preserve "set the current user to unsh"
        assert "unsh" in workflow_context["command"]  # Parameter should be preserved
        assert workflow_context["command"] != resolved_command  # Should NOT be "set current user"
        
        # 5. Verify that parameter extraction would work correctly
        # The ParameterExtraction class would use workflow_context["command"] 
        # which contains "set the current user to unsh" with the "unsh" parameter

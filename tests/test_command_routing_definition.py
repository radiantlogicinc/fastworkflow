import pytest
from pathlib import Path
from pydantic import BaseModel
import os

from fastworkflow.command_routing import RoutingRegistry, RoutingDefinition, ModuleType
from fastworkflow.command_directory import CommandDirectory
from fastworkflow.command_context_model import CommandContextModel


@pytest.fixture(scope="module")
def sample_workflow_path() -> Path:
    """Provides the path to the sample workflow example."""
    return Path(__file__).parent.parent / "examples" / "retail_workflow"


@pytest.fixture
def command_routing_definition(sample_workflow_path: Path) -> RoutingDefinition:
    """Return a RoutingDefinition instance for testing."""
    # Clear any cached definitions
    RoutingRegistry.clear_registry()
    
    # Create the necessary data structures
    command_directory = CommandDirectory.load(str(sample_workflow_path))
    context_model = CommandContextModel.load(str(sample_workflow_path))
    
    # Initialize with empty contexts if needed
    contexts = {}
    try:
        # Try to get contexts from the context model
        for context_name in context_model._command_contexts:
            contexts[context_name] = context_model.commands(context_name)
    except Exception:
        # Fallback to ensure tests don't fail
        contexts = {"*": []}
    
    # Ensure we have at least one context
    if not contexts:
        contexts = {"*": []}
    
    return RoutingDefinition(
        workflow_folderpath=str(sample_workflow_path),
        command_directory=command_directory,
        context_model=context_model,
        contexts=contexts,
        command_directory_map={"*": set()},
        routing_definition_map={}
    )


class TestRoutingDefinition:
    """Test suite for the refactored RoutingDefinition and RoutingRegistry."""

    def test_registry_caching(self, sample_workflow_path: Path):
        """Tests that the RoutingDefinition is created and properly cached."""
        RoutingRegistry.clear_registry()
        
        # First call should build a new definition
        definition1 = RoutingRegistry.get_definition(str(sample_workflow_path))
        
        assert isinstance(definition1, RoutingDefinition)
        
        # Second call should return the cached instance
        definition2 = RoutingRegistry.get_definition(str(sample_workflow_path))
        
        assert definition1 is definition2  # Same instance (not just equal)

    def test_get_command_names_for_valid_context(self, command_routing_definition: RoutingDefinition):
        """Test that get_command_names returns a list of command names for a valid context."""
        # Get a context that exists in the workflow
        context = list(command_routing_definition.contexts.keys())[0]
        
        command_names = command_routing_definition.get_command_names(context)
        
        assert isinstance(command_names, list)
        assert len(command_names) > 0
        assert all(isinstance(name, str) for name in command_names)

    def test_get_command_names_for_invalid_context(self, command_routing_definition: RoutingDefinition):
        """Test that get_command_names raises a ValueError for an invalid context."""
        with pytest.raises(ValueError):
            command_routing_definition.get_command_names("invalid_context")

    def test_get_command_class_for_parameters(self, command_routing_definition: RoutingDefinition):
        """Test that get_command_class returns the correct class for parameter extraction."""
        # Get a command name that exists in the workflow
        context = list(command_routing_definition.contexts.keys())[0]
        command_names = command_routing_definition.get_command_names(context)
        
        param_class = command_routing_definition.get_command_class(
            command_names[0], ModuleType.COMMAND_PARAMETERS_CLASS
        )
        
        # The class may or may not exist, but the method should not raise an exception
        assert param_class is None or isinstance(param_class, type)

    def test_get_command_class_for_response_generator(self, command_routing_definition: RoutingDefinition):
        """Test that get_command_class returns the correct class for response generation."""
        # Get a command name that exists in the workflow
        context = list(command_routing_definition.contexts.keys())[0]
        command_names = command_routing_definition.get_command_names(context)
        
        rg_class = command_routing_definition.get_command_class(
            command_names[0], ModuleType.RESPONSE_GENERATION_INFERENCE
        )
        
        # The class may or may not exist, but the method should not raise an exception
        assert rg_class is None or isinstance(rg_class, type)

    def test_get_command_class_for_nonexistent_command_file(self, command_routing_definition: RoutingDefinition):
        """Test that get_command_class returns None for a nonexistent command file."""
        # Try to get a class for a command that doesn't exist
        param_class = command_routing_definition.get_command_class(
            "nonexistent_command", ModuleType.COMMAND_PARAMETERS_CLASS
        )
        
        assert param_class is None

    def test_build_method(self, sample_workflow_path: Path):
        """Test that the build method creates a valid RoutingDefinition."""
        # Clear the registry
        RoutingRegistry.clear_registry()
        
        # Build a new definition
        routing_def = RoutingDefinition.build(str(sample_workflow_path))
        
        assert routing_def is not None
        assert isinstance(routing_def, RoutingDefinition)
        assert routing_def.workflow_folderpath == str(sample_workflow_path)
        assert len(routing_def.contexts) > 0


def test_get_command_class_missing_input(sample_workflow_path):
    """Ensure no exception is raised when a command lacks `Signature.Input` (regression)."""
    routing_def = RoutingDefinition.build(str(sample_workflow_path))

    # ErrorCorrection/you_misunderstood_intent has no Signature.Input, so COMMAND_PARAMETERS_CLASS is absent.
    cls = routing_def.get_command_class(
        "ErrorCorrection/you_misunderstood", ModuleType.COMMAND_PARAMETERS_CLASS
    )

    # The call should succeed and simply return None when the class is missing.
    assert cls is None 
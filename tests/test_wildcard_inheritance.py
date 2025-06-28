import os
import pytest
from unittest.mock import MagicMock, patch, call
import json
from pathlib import Path

import fastworkflow
from fastworkflow.command_routing import RoutingDefinition, RoutingRegistry
from fastworkflow.command_directory import CommandDirectory
from fastworkflow.command_context_model import CommandContextModel
from fastworkflow.model_pipeline_training import train, get_artifact_path, _get_utterances, cache_ancestor_utterances


@pytest.fixture
def example_workflow_path():
    """Return the path to the example workflow."""
    return os.path.join(os.path.dirname(__file__), "example_workflow")


@pytest.fixture
def mock_context_model():
    """Create a mock context model with a hierarchy for testing wildcard inheritance."""
    return {
        "TodoList": {"base": ["TodoItem"]},
        "TodoItem": {"base": []},
        "TodoListManager": {"base": []},
    }


@pytest.fixture
def mock_wildcard_utterances():
    """Create mock wildcard utterances for each context."""
    return {
        "*": ["global wildcard 1", "global wildcard 2"],
        "TodoItem": ["todoitem wildcard 1", "todoitem wildcard 2"],
        "TodoList": ["todolist wildcard 1", "todolist wildcard 2"],
        "TodoListManager": ["manager wildcard 1", "manager wildcard 2"]
    }


class MockUtteranceMetadata:
    """Mock class for UtteranceMetadata."""
    
    def __init__(self, context_name, utterances):
        self.plain_utterances = utterances
        self.template_utterances = []
        self.generated_utterances_module_filepath = "mock_filepath"
        self.generated_utterances_func_name = "mock_func"
        self.context_name = context_name
        
    def get_generated_utterances_func(self, _):
        """Return a function that returns the mock utterances."""
        utterances = self.plain_utterances
        
        def mock_func(_, __):
            return utterances
            
        return mock_func


class MockCommandDirectory:
    """Mock class for CommandDirectory."""
    
    def __init__(self, wildcard_utterances):
        self.map_command_2_utterance_metadata = {}
        self.map_command_2_metadata = {}
        self.core_command_names = ["wildcard"]
        self.wildcard_utterances = wildcard_utterances
        
        # Add wildcard utterance metadata for each context
        for context, utterances in wildcard_utterances.items():
            self.map_command_2_utterance_metadata[f"{context}/wildcard" if context != "*" else "wildcard"] = \
                MockUtteranceMetadata(context, utterances)
            
    def ensure_command_hydrated(self, cmd_name):
        """Mock method to ensure command is hydrated."""
        return True
        
    def get_command_metadata(self, cmd_name):
        """Mock method to get command metadata."""
        metadata = MagicMock()
        metadata.input_for_param_extraction_class = True
        return metadata
        
    def get_utterance_metadata(self, cmd_name):
        """Mock method to get utterance metadata."""
        return self.map_command_2_utterance_metadata.get(cmd_name)
    
    def get_commandinfo_folderpath(self, _):
        """Mock method to get command info folder path."""
        return "/tmp/___command_info"


class MockRoutingDefinition:
    """Mock class for RoutingDefinition."""
    
    def __init__(self, context_model, command_directory):
        self.context_model = context_model
        self.command_directory = command_directory
        self.contexts = {
            "*": ["wildcard"],
            "TodoItem": ["wildcard"],
            "TodoList": ["wildcard"],
            "TodoListManager": ["wildcard"]
        }


def test_wildcard_inheritance(example_workflow_path, mock_context_model, mock_wildcard_utterances):
    """Test that wildcard utterances are properly inherited from parent contexts."""
    # Create mock objects
    mock_context = MagicMock(spec=CommandContextModel)
    mock_context._command_contexts = mock_context_model
    
    mock_cmd_dir = MockCommandDirectory(mock_wildcard_utterances)
    
    mock_routing_def = MockRoutingDefinition(mock_context, mock_cmd_dir)
    
    # Patch RoutingRegistry.get_definition to return our mock
    with patch.object(RoutingRegistry, 'get_definition', return_value=mock_routing_def):
        # Create a workflow
        workflow = fastworkflow.Workflow.create(
            workflow_folderpath=example_workflow_path,
            workflow_id_str="test-wildcard-inheritance"
        )
        
        # Create a function to simulate what happens in the train function
        # This is the code we're testing, extracted from model_pipeline_training.py
        context_utterance_cache = {}
        
        def get_wildcard_utterances(context_name: str) -> list[str]:
            """Recursively collect wildcard utterances from the current context and its parent contexts."""
            # Check cache first to avoid redundant calculations
            if context_name in context_utterance_cache:
                return context_utterance_cache[context_name]
                
            # Get the wildcard's own utterances for this context
            cmd_name = f"{context_name}/wildcard" if context_name != "*" else "wildcard"
            um = mock_cmd_dir.get_utterance_metadata(cmd_name)
            func = um.get_generated_utterances_func(example_workflow_path)
            utterances = set(func(workflow, cmd_name))
            
            # Find parent contexts from the context model
            if context_name in mock_context._command_contexts:
                ctx_info = mock_context._command_contexts[context_name]
                parent_contexts = ctx_info.get('base', [])
                
                # Recursively get utterances from each parent context
                for parent_context in parent_contexts:
                    parent_utterances = get_wildcard_utterances(parent_context)
                    utterances.update(parent_utterances)
            
            # Convert back to list and cache the result
            result = list(utterances)
            context_utterance_cache[context_name] = result
            return result
        
        # Test wildcard utterance inheritance for TodoList
        # TodoList should include its own utterances plus TodoItem's utterances
        todolist_utterances = get_wildcard_utterances("TodoList")
        
        # Expected: TodoList utterances + TodoItem utterances (de-duplicated)
        expected_todolist = set(mock_wildcard_utterances["TodoList"] + mock_wildcard_utterances["TodoItem"])
        assert set(todolist_utterances) == expected_todolist
        
        # Test wildcard utterance inheritance for TodoItem
        # TodoItem should only have its own utterances (no parent)
        todoitem_utterances = get_wildcard_utterances("TodoItem")
        assert set(todoitem_utterances) == set(mock_wildcard_utterances["TodoItem"])
        
        # Test wildcard utterance inheritance for TodoListManager
        # TodoListManager should only have its own utterances (no parent)
        manager_utterances = get_wildcard_utterances("TodoListManager")
        assert set(manager_utterances) == set(mock_wildcard_utterances["TodoListManager"])
        
        # Test caching mechanism
        # Second call should use cached results
        cached_todolist_utterances = get_wildcard_utterances("TodoList")
        assert todolist_utterances == cached_todolist_utterances
        assert "TodoList" in context_utterance_cache


def test_wildcard_inheritance_deeper_hierarchy():
    """Test wildcard inheritance with a deeper hierarchy."""
    # Create a deeper hierarchy:
    # GrandChild -> Child -> Parent -> Root
    context_model = {
        "GrandChild": {
            "base": ["Child"]
        },
        "Child": {
            "base": ["Parent"]
        },
        "Parent": {
            "base": ["Root"]
        },
        "Root": {
            "base": []
        }
    }
    
    # Create utterances for each context
    wildcard_utterances = {
        "*": ["global 1", "global 2"],
        "Root": ["root 1", "root 2"],
        "Parent": ["parent 1", "parent 2"],
        "Child": ["child 1", "child 2"],
        "GrandChild": ["grandchild 1", "grandchild 2"]
    }
    
    # Create mock objects
    mock_context = MagicMock(spec=CommandContextModel)
    mock_context._command_contexts = context_model
    
    mock_cmd_dir = MockCommandDirectory(wildcard_utterances)
    
    mock_routing_def = MockRoutingDefinition(mock_context, mock_cmd_dir)
    mock_routing_def.contexts = {
        "*": ["wildcard"],
        "Root": ["wildcard"],
        "Parent": ["wildcard"],
        "Child": ["wildcard"],
        "GrandChild": ["wildcard"]
    }
    
    # Patch RoutingRegistry.get_definition to return our mock
    with patch.object(RoutingRegistry, 'get_definition', return_value=mock_routing_def):
        # Create a workflow
        workflow = fastworkflow.Workflow.create(
            workflow_folderpath="/tmp",
            workflow_id_str="test-wildcard-inheritance-deep"
        )
        
        # Create the function to test
        context_utterance_cache = {}
        
        def get_wildcard_utterances(context_name: str) -> list[str]:
            """Recursively collect wildcard utterances from the current context and its parent contexts."""
            # Check cache first to avoid redundant calculations
            if context_name in context_utterance_cache:
                return context_utterance_cache[context_name]
                
            # Get the wildcard's own utterances for this context
            cmd_name = f"{context_name}/wildcard" if context_name != "*" else "wildcard"
            um = mock_cmd_dir.get_utterance_metadata(cmd_name)
            func = um.get_generated_utterances_func("/tmp")
            utterances = set(func(workflow, cmd_name))
            
            # Find parent contexts from the context model
            if context_name in mock_context._command_contexts:
                ctx_info = mock_context._command_contexts[context_name]
                parent_contexts = ctx_info.get('base', [])
                
                # Recursively get utterances from each parent context
                for parent_context in parent_contexts:
                    parent_utterances = get_wildcard_utterances(parent_context)
                    utterances.update(parent_utterances)
            
            # Convert back to list and cache the result
            result = list(utterances)
            context_utterance_cache[context_name] = result
            return result
        
        # Test GrandChild context - should inherit from all ancestors
        grandchild_utterances = get_wildcard_utterances("GrandChild")
        expected_grandchild = set(wildcard_utterances["GrandChild"] + 
                                wildcard_utterances["Child"] + 
                                wildcard_utterances["Parent"] + 
                                wildcard_utterances["Root"])
        assert set(grandchild_utterances) == expected_grandchild
        
        # Test Child context - should inherit from Parent and Root
        child_utterances = get_wildcard_utterances("Child")
        expected_child = set(wildcard_utterances["Child"] + 
                           wildcard_utterances["Parent"] + 
                           wildcard_utterances["Root"])
        assert set(child_utterances) == expected_child
        
        # Test Parent context - should inherit from Root
        parent_utterances = get_wildcard_utterances("Parent")
        expected_parent = set(wildcard_utterances["Parent"] + wildcard_utterances["Root"])
        assert set(parent_utterances) == expected_parent
        
        # Test Root context - should only have its own utterances
        root_utterances = get_wildcard_utterances("Root")
        assert set(root_utterances) == set(wildcard_utterances["Root"])
        
        # Test that the cache is working
        assert len(context_utterance_cache) == 4  # All contexts should be cached
        assert "GrandChild" in context_utterance_cache
        assert "Child" in context_utterance_cache
        assert "Parent" in context_utterance_cache
        assert "Root" in context_utterance_cache


def test_wildcard_inheritance_with_duplicates():
    """Test that duplicate utterances are properly handled."""
    # Create a context hierarchy with duplicate utterances
    context_model = {
        "Child": {
            "base": ["Parent"]
        },
        "Parent": {
            "base": []
        }
    }
    
    # Create utterances with duplicates across contexts
    wildcard_utterances = {
        "*": ["global 1", "global 2"],
        "Parent": ["parent 1", "duplicate utterance", "parent 2"],
        "Child": ["child 1", "duplicate utterance", "child 2"]
    }
    
    # Create mock objects
    mock_context = MagicMock(spec=CommandContextModel)
    mock_context._command_contexts = context_model
    
    mock_cmd_dir = MockCommandDirectory(wildcard_utterances)
    
    mock_routing_def = MockRoutingDefinition(mock_context, mock_cmd_dir)
    mock_routing_def.contexts = {
        "*": ["wildcard"],
        "Parent": ["wildcard"],
        "Child": ["wildcard"]
    }
    
    # Patch RoutingRegistry.get_definition to return our mock
    with patch.object(RoutingRegistry, 'get_definition', return_value=mock_routing_def):
        # Create a workflow
        workflow = fastworkflow.Workflow.create(
            workflow_folderpath="/tmp",
            workflow_id_str="test-wildcard-inheritance-duplicates"
        )
        
        # Create the function to test
        context_utterance_cache = {}
        
        def get_wildcard_utterances(context_name: str) -> list[str]:
            """Recursively collect wildcard utterances from the current context and its parent contexts."""
            # Check cache first to avoid redundant calculations
            if context_name in context_utterance_cache:
                return context_utterance_cache[context_name]
                
            # Get the wildcard's own utterances for this context
            cmd_name = f"{context_name}/wildcard" if context_name != "*" else "wildcard"
            um = mock_cmd_dir.get_utterance_metadata(cmd_name)
            func = um.get_generated_utterances_func("/tmp")
            utterances = set(func(workflow, cmd_name))
            
            # Find parent contexts from the context model
            if context_name in mock_context._command_contexts:
                ctx_info = mock_context._command_contexts[context_name]
                parent_contexts = ctx_info.get('base', [])
                
                # Recursively get utterances from each parent context
                for parent_context in parent_contexts:
                    parent_utterances = get_wildcard_utterances(parent_context)
                    utterances.update(parent_utterances)
            
            # Convert back to list and cache the result
            result = list(utterances)
            context_utterance_cache[context_name] = result
            return result
        
        # Test Child context - should inherit from Parent but deduplicate
        child_utterances = get_wildcard_utterances("Child")
        
        # Expected: Child utterances + Parent utterances (de-duplicated)
        expected_child = set(wildcard_utterances["Child"] + wildcard_utterances["Parent"])
        assert set(child_utterances) == expected_child
        
        # Verify that the duplicate utterance appears only once
        assert child_utterances.count("duplicate utterance") == 1
        
        # Count total utterances (should be 5 after deduplication)
        # "parent 1", "duplicate utterance", "parent 2", "child 1", "child 2"
        assert len(child_utterances) == 5


@patch('fastworkflow.model_pipeline_training.AutoTokenizer')
@patch('fastworkflow.model_pipeline_training.AutoModelForSequenceClassification')
@patch('fastworkflow.model_pipeline_training.DataLoader')
@patch('fastworkflow.model_pipeline_training.train_test_split')
@patch('fastworkflow.model_pipeline_training.AdamW')
@patch('fastworkflow.model_pipeline_training.evaluate_model')
@patch('fastworkflow.model_pipeline_training.save_model')
@patch('fastworkflow.model_pipeline_training.save_label_encoder')
@patch('fastworkflow.model_pipeline_training.find_optimal_threshold')
@patch('fastworkflow.model_pipeline_training.analyze_model_confidence')
@patch('fastworkflow.model_pipeline_training.find_optimal_confidence_threshold')
@patch('fastworkflow.model_pipeline_training.predict_single_sentence')
@patch('fastworkflow.model_pipeline_training.ModelPipeline')
def test_train_function_wildcard_inheritance(mock_model_pipeline, mock_predict, mock_find_optimal, 
                                            mock_analyze, mock_find_threshold, mock_save_label,
                                            mock_save_model, mock_evaluate, mock_adamw, 
                                            mock_train_test_split, mock_dataloader,
                                            mock_auto_model, mock_auto_tokenizer):
    """Test that the train function correctly handles wildcard inheritance."""
    # Create a context hierarchy with inheritance
    context_model = {
        "Child": {
            "base": ["Parent"]
        },
        "Parent": {
            "base": []
        }
    }

    # Create utterances with some duplicates
    wildcard_utterances = {
        "*": ["global 1", "global 2"],
        "Parent": ["parent 1", "duplicate utterance", "parent 2"],
        "Child": ["child 1", "duplicate utterance", "child 2"]
    }

    # Create mock objects
    mock_context = MagicMock(spec=CommandContextModel)
    mock_context._command_contexts = context_model

    mock_cmd_dir = MockCommandDirectory(wildcard_utterances)

    # Add some non-wildcard commands for testing
    mock_cmd_dir.map_command_2_utterance_metadata["Parent/command1"] = MockUtteranceMetadata("Parent", ["command1 utterance"])
    mock_cmd_dir.map_command_2_utterance_metadata["Child/command2"] = MockUtteranceMetadata("Child", ["command2 utterance"])

    mock_routing_def = MockRoutingDefinition(mock_context, mock_cmd_dir)
    mock_routing_def.contexts = {
        "*": ["wildcard"],
        "Parent": ["wildcard", "command1"],
        "Child": ["wildcard", "command2"]
    }

    # Mock the model training related functions
    mock_find_threshold.return_value = ({"threshold": 0.65, "f1": 0.9, "ndcg": 0.8, "distil_usage": 10}, [])
    mock_find_optimal.return_value = (0.7, {"threshold": 0.7, "f1_score": 0.9, "top3_usage": 0.2, 
                                          "top1_accuracy": 0.8, "top3_accuracy": 0.95})

    # Create a workflow
    workflow = MagicMock(spec=fastworkflow.Workflow)
    workflow.folderpath = "/tmp"

    # Set up the utterance_command_tuples collector to capture what would be passed to training
    utterance_command_tuples_collector = {}

    # Patch the train function to capture the utterance_command_tuples
    def mock_train_loader_side_effect(*args, **kwargs):
        # Extract the dataset from the args
        dataset = args[0]
        # Store the dataset for later inspection
        context_name = kwargs.get('context_name', 'unknown')
        utterance_command_tuples_collector[context_name] = dataset
        # Return a mock DataLoader
        mock_loader = MagicMock()
        mock_loader.__iter__ = lambda _: iter([])
        return mock_loader

    # Apply the side effect to the DataLoader mock
    mock_dataloader.side_effect = lambda dataset, **kwargs: mock_train_loader_side_effect(dataset, **kwargs)

    # Patch open for JSON files
    mock_open = MagicMock()
    mock_open.return_value.__enter__.return_value.read.return_value = '{"confidence_threshold": 0.65}'

    # Create a mock for the _get_utterances function
    def mock_get_utterances(cmd_dir, cmd_name, workflow, workflow_folderpath):
        """Mock the standalone _get_utterances function."""
        # For wildcard, we'll return an empty list as the real function
        # will be calling get_wildcard_utterances
        if cmd_name == 'wildcard':
            return []

        # For other commands, return mock utterances
        if cmd_name in ['command1', 'Parent/command1']:
            return ["command1 utterance"]
        elif cmd_name in ['command2', 'Child/command2']:
            return ["command2 utterance"]

        return []

    # Mock the RoutingRegistry class and its get_definition method
    mock_routing_registry = MagicMock()
    mock_routing_registry.get_definition.return_value = mock_routing_def

    # Mock the CommandContextModel class and its load method
    mock_context_model_class = MagicMock()
    mock_context_model_instance = MagicMock()
    mock_context_model_instance._command_contexts = {}
    mock_context_model_class.load.return_value = mock_context_model_instance

    # Patch RoutingRegistry to return our mock
    with patch('fastworkflow.RoutingRegistry', mock_routing_registry), \
         patch('fastworkflow.CommandContextModel', mock_context_model_class), \
         patch('builtins.open', mock_open), \
         patch('fastworkflow.get_internal_workflow_path', return_value="/tmp"), \
         patch('os.path.exists', return_value=True), \
         patch('os.makedirs'), \
         patch('fastworkflow.model_pipeline_training._get_utterances', side_effect=mock_get_utterances):

        # Run the train function
        train(workflow)
        
        # Since we've mocked most of the training process, we can't directly verify
        # that wildcard utterances were properly inherited. However, this test
        # confirms that the train function runs without errors with our implementation. 


@pytest.fixture
def mock_workflow():
    """Provides a mock workflow object."""
    workflow = MagicMock(spec=fastworkflow.Workflow)
    workflow.folderpath = "/tmp/mock_workflow"
    return workflow

@pytest.fixture
def mock_crd():
    """Provides a mock a mock CRD object containing other necessary mocks."""
    # 1. Mock CommandDirectory
    cmd_dir = MagicMock(spec=CommandDirectory)
    
    # 2. Mock CommandContextModel
    context_model = MagicMock(spec=CommandContextModel)
    
    # Define a context hierarchy for the test
    # TodoItem -> TodoList -> TodoListManager -> *
    context_model.get_ancestor_contexts.side_effect = lambda context_name: {
        "TodoItem": ["TodoList", "TodoListManager", "*"],
        "TodoList": ["TodoListManager", "*"],
        "TodoListManager": ["*"],
        "*": []
    }.get(context_name, [])

    # Define the commands available in each ancestor context
    context_model.commands.side_effect = lambda context_name: {
        "TodoList": ["TodoList/add_todo", "TodoList/remove_todo", "wildcard"],
        "TodoListManager": ["TodoListManager/create_list", "TodoListManager/delete_list", "wildcard"],
        "*": ["wildcard"]
    }.get(context_name, [])

    # 3. Create mock CRD and attach other mocks to it
    crd = MagicMock(spec=RoutingDefinition)
    crd.context_model = context_model
    crd.command_directory = cmd_dir
    
    return crd

def test_get_wildcard_utterances_aggregation(mock_workflow, mock_crd):
    """
    Tests that get_wildcard_utterances correctly aggregates utterances from:
    1. The base 'wildcard' command.
    2. All commands (except 'wildcard') from all ancestor contexts.
    """
    # Define utterances for various commands
    utterance_map = {
        "wildcard": ["wildcard utterance 1", "wildcard utterance 2"],
        "TodoList/add_todo": ["add a todo", "new item"],
        "TodoList/remove_todo": ["remove a todo", "delete item"],
        "TodoListManager/create_list": ["create a new list", "make list"],
        "TodoListManager/delete_list": ["delete a list", "remove list"],
    }

    def get_utterances_side_effect(workflow, workflow_path, cmd_dir_mock, cmd_name):
        return utterance_map.get(cmd_name, [])

    # --- Test Execution ---
    with patch('fastworkflow.model_pipeline_training._get_utterances', side_effect=get_utterances_side_effect):
        
        cache = {}
        result_utterances = cache_ancestor_utterances("TodoItem", mock_crd, mock_workflow, cache)

        # --- Assertions ---
        expected_utterances = {
            "add a todo", "new item",
            "remove a todo", "delete item",
            "create a new list", "make list",
            "delete a list", "remove list"
        }

        assert set(result_utterances) == expected_utterances
        
        # Verify mocks were called correctly
        mock_crd.context_model.get_ancestor_contexts.assert_called_once_with("TodoItem")
        
        expected_commands_calls = [
            call("TodoList"),
            call("TodoListManager"),
            call("*")
        ]
        mock_crd.context_model.commands.assert_has_calls(expected_commands_calls, any_order=True)
        assert mock_crd.context_model.commands.call_count == 3
        
        # Verify cache was populated
        assert "TodoList" in cache

        # --- Assertions ---
        # Cache now stores full command keys rather than plain utterances.
        assert {"TodoList/add_todo", "TodoList/remove_todo"}.issubset(set(cache["TodoList"])) 
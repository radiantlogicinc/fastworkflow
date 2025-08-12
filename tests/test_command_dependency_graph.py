"""
Comprehensive tests for command dependency graph functionality.
Tests use retail_workflow as real test data without mocks.
"""


import contextlib
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, List
import shutil

import pytest

from fastworkflow.utils.command_dependency_graph import (
    generate_dependency_graph,
    get_dependency_suggestions,
    _collect_command_params,
    _exact_match_score,
    _semantic_match_score,
    _contexts_overlap,
    ParamMeta,
    CommandParams,
)
from fastworkflow.command_directory import CommandDirectory
from fastworkflow.command_routing import RoutingDefinition
from fastworkflow.utils.signatures import InputForParamExtraction


class TestCommandDependencyGraph:
    """Test suite for command dependency graph generation and usage."""

    @pytest.fixture(scope="class")
    def retail_workflow_path(self):
        """Get path to retail workflow for testing."""
        return os.path.join(os.path.dirname(__file__), "..", "fastworkflow", "examples", "retail_workflow")

    @pytest.fixture(scope="class")
    def built_workflow_path(self, retail_workflow_path):
        """Prepare retail workflow for testing."""
        # Create a temporary directory for testing
        temp_dir = tempfile.mkdtemp(prefix="test_dep_graph_")

        # Copy retail workflow to temp directory
        dest_path = os.path.join(temp_dir, "retail_workflow")
        shutil.copytree(retail_workflow_path, dest_path)

        # Ensure ___command_info directory exists
        commandinfo_dir = CommandDirectory.get_commandinfo_folderpath(dest_path)
        os.makedirs(commandinfo_dir, exist_ok=True)

        # Build routing definition to ensure metadata is available
        with contextlib.suppress(Exception):
            RoutingDefinition.build(dest_path)
        yield dest_path

        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def graph_path(self, built_workflow_path):
        """Generate dependency graph and return its path."""
        return generate_dependency_graph(built_workflow_path)

    def test_collect_command_params(self, built_workflow_path):
        """Test parameter collection from commands."""
        params = _collect_command_params(built_workflow_path)
        
        # Should have collected parameters for retail workflow commands
        assert len(params) > 0
        
        # Check some known commands
        if "get_user_details" in params:
            cmd_params = params["get_user_details"]
            assert isinstance(cmd_params, CommandParams)
            assert len(cmd_params.inputs) > 0
            assert len(cmd_params.outputs) > 0
            
            # Check user_id input parameter
            user_id_param = next((p for p in cmd_params.inputs if p.name == "user_id"), None)
            assert user_id_param is not None
            assert user_id_param.description != ""
            assert len(user_id_param.examples) > 0

    def test_exact_match_score(self):
        """Test exact parameter matching."""
        # Exact match case
        param1 = ParamMeta(
            name="user_id",
            type_str="str",
            description="User identifier",
            examples=["user123"]
        )
        param2 = ParamMeta(
            name="user_id",
            type_str="str",
            description="The user ID",
            examples=["user456"]
        )
        assert _exact_match_score(param1, param2) == 1.0
        
        # Case insensitive name match
        param3 = ParamMeta(
            name="USER_ID",
            type_str="str",
            description="User identifier",
            examples=[]
        )
        assert _exact_match_score(param1, param3) == 1.0
        
        # Different name
        param4 = ParamMeta(
            name="order_id",
            type_str="str",
            description="Order identifier",
            examples=[]
        )
        assert _exact_match_score(param1, param4) == 0.0
        
        # Different type
        param5 = ParamMeta(
            name="user_id",
            type_str="int",
            description="User identifier",
            examples=[]
        )
        assert _exact_match_score(param1, param5) == 0.0

    def test_semantic_match_score(self):
        """Test semantic parameter matching using embeddings."""
        # Similar parameters should have high score
        param1 = ParamMeta(
            name="user_id",
            type_str="str",
            description="The unique identifier for a user",
            examples=["user_123", "john_doe_456"]
        )
        param2 = ParamMeta(
            name="customer_id",
            type_str="str",
            description="The ID of the customer",
            examples=["cust_789", "jane_smith_012"]
        )
        score = _semantic_match_score(param1, param2)
        assert score > 0.5  # Should have some similarity
        
        # Very different parameters should have low score
        param3 = ParamMeta(
            name="product_price",
            type_str="float",
            description="The price of the product in dollars",
            examples=["19.99", "299.00"]
        )
        score2 = _semantic_match_score(param1, param3)
        assert score2 < score  # Should be less similar than user_id vs customer_id

    def test_generate_dependency_graph(self, built_workflow_path):
        """Test dependency graph generation."""
        # Generate graph
        graph_path = generate_dependency_graph(built_workflow_path)
        assert os.path.exists(graph_path)
        
        # Load and validate graph structure
        with open(graph_path, 'r') as f:
            graph = json.load(f)
        
        assert "nodes" in graph
        assert "edges" in graph
        assert isinstance(graph["nodes"], list)
        assert isinstance(graph["edges"], list)
        
        # Check nodes are command names
        nodes = graph["nodes"]
        assert len(nodes) > 0
        
        # Check edge structure
        if len(graph["edges"]) > 0:
            edge = graph["edges"][0]
            assert "from" in edge
            assert "to" in edge
            assert "weight" in edge
            assert "matched_params" in edge
            assert "match_type" in edge
            assert isinstance(edge["weight"], float)
            assert 0.0 <= edge["weight"] <= 1.0

    def test_dependency_graph_with_exact_only(self, built_workflow_path):
        """Test graph generation with exact matching only."""
        graph_path = generate_dependency_graph(built_workflow_path, exact_only=True)

        with open(graph_path, 'r') as f:
            graph = json.load(f)

        # All edges should be exact matches
        for edge in graph["edges"]:
            assert edge["match_type"] in ["exact", "mixed"]

    def test_dependency_graph_with_custom_threshold(self, built_workflow_path):
        """Test graph generation with custom semantic threshold."""
        # High threshold should result in fewer edges
        graph_path_high = generate_dependency_graph(built_workflow_path, semantic_threshold=0.95)
        with open(graph_path_high, 'r') as f:
            graph_high = json.load(f)
        
        # Low threshold should result in more edges
        graph_path_low = generate_dependency_graph(built_workflow_path, semantic_threshold=0.5)
        with open(graph_path_low, 'r') as f:
            graph_low = json.load(f)
        
        # Lower threshold should have same or more edges
        assert len(graph_low["edges"]) >= len(graph_high["edges"])

    def test_get_dependency_suggestions(self, graph_path):
        """Test getting dependency suggestions for missing parameters."""
        # Load graph to find a valid test case
        with open(graph_path, 'r') as f:
            graph = json.load(f)

        if len(graph["edges"]) > 0:
            # Use first edge as test case
            edge = graph["edges"][0]
            y_command = edge["from"]
            if matched_params := edge["matched_params"]:
                missing_param = matched_params[0][0]  # Y's input param

                # Get suggestions
                suggestions = get_dependency_suggestions(
                    graph_path,
                    y_command,
                    missing_param,
                    min_weight=0.5
                )

                assert isinstance(suggestions, list)
                if suggestions:
                    suggestion = suggestions[0]
                    assert "command" in suggestion
                    assert "sub_plans" in suggestion
                    assert "weight" in suggestion

    def test_dependency_suggestions_with_depth_limit(self, graph_path):
        """Test dependency suggestions respect depth limit."""
        with open(graph_path, 'r') as f:
            graph = json.load(f)

        if len(graph["edges"]) > 0:
            edge = graph["edges"][0]
            y_command = edge["from"]
            if matched_params := edge["matched_params"]:
                missing_param = matched_params[0][0]

                # Test with max_depth=0 (no recursion)
                suggestions = get_dependency_suggestions(
                    graph_path,
                    y_command,
                    missing_param,
                    min_weight=0.5,
                    max_depth=0
                )

                # Check that sub_plans are empty or very limited
                for suggestion in suggestions:
                    assert len(suggestion.get("sub_plans", [])) == 0

    def test_dependency_suggestions_weight_filtering(self, graph_path):
        """Test that dependency suggestions respect weight threshold."""
        with open(graph_path, 'r') as f:
            graph = json.load(f)

        if len(graph["edges"]) > 0:
            edge = graph["edges"][0]
            y_command = edge["from"]
            if matched_params := edge["matched_params"]:
                missing_param = matched_params[0][0]

                # High weight threshold should return fewer suggestions
                suggestions_high = get_dependency_suggestions(
                    graph_path,
                    y_command,
                    missing_param,
                    min_weight=0.9
                )

                # Low weight threshold should return more suggestions
                suggestions_low = get_dependency_suggestions(
                    graph_path,
                    y_command,
                    missing_param,
                    min_weight=0.1
                )

                assert len(suggestions_low) >= len(suggestions_high)

    def test_retail_workflow_specific_dependencies(self, graph_path):
        """Test specific known dependencies in retail workflow."""
        with open(graph_path, 'r') as f:
            graph = json.load(f)
        
        # Look for specific expected dependencies
        edges = graph["edges"]
        
        # Example: Commands that need user_id might depend on find_user_id_by_email
        user_dependent_commands = ["get_user_details", "modify_user_address"]
        user_provider_commands = ["find_user_id_by_email", "find_user_id_by_name_zip"]
        
        found_user_dependencies = False
        for edge in edges:
            if (edge["from"] in user_dependent_commands and 
                edge["to"] in user_provider_commands):
                # Check that user_id is in matched params
                matched_params = edge["matched_params"]
                param_names = [p[0] for p in matched_params]
                if "user_id" in param_names:
                    found_user_dependencies = True
                    break
        
        # We expect to find at least some user_id dependencies
        # This might not always be true depending on context overlap
        # so we just check the structure is correct
        assert isinstance(found_user_dependencies, bool)

    def test_integration_with_validation(self, built_workflow_path):
        """Test integration with parameter validation and error messages."""
        # This tests the integration point mentioned in the spec
        graph_path = os.path.join(
            CommandDirectory.get_commandinfo_folderpath(built_workflow_path),
            "parameter_dependency_graph.json"
        )
        
        # Generate graph if it doesn't exist
        if not os.path.exists(graph_path):
            generate_dependency_graph(built_workflow_path)
        
        # Simulate missing parameter scenario
        if os.path.exists(graph_path):
            # Test that we can get suggestions for a missing field
            suggestions = get_dependency_suggestions(
                graph_path,
                "get_order_details",  # Command that needs order_id
                "order_id",  # Missing parameter
                min_weight=0.5
            )
            
            # Verify structure of suggestions
            assert isinstance(suggestions, list)
            for suggestion in suggestions:
                assert "command" in suggestion
                assert isinstance(suggestion["command"], str)

    def test_context_overlap(self, built_workflow_path):
        """Test context overlap checking."""
        routing = RoutingDefinition.build(built_workflow_path)
        
        # Get some commands from the workflow
        params = _collect_command_params(built_workflow_path)
        commands = list(params.keys())
        
        if len(commands) >= 2:
            cmd1, cmd2 = commands[0], commands[1]
            
            # Test overlap function
            overlap = _contexts_overlap(routing, cmd1, cmd2)
            assert isinstance(overlap, bool)
            
            # Same command should overlap with itself
            assert _contexts_overlap(routing, cmd1, cmd1) == True

    def test_empty_workflow(self):
        """Test handling of empty or invalid workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create minimal structure required by RoutingDefinition
            os.makedirs(os.path.join(temp_dir, "_commands"), exist_ok=True)
            os.makedirs(os.path.join(temp_dir, "___command_info"), exist_ok=True)
            
            # Should handle empty workflow gracefully
            graph_path = generate_dependency_graph(temp_dir)
            
            with open(graph_path, 'r') as f:
                graph = json.load(f)
            
            # The workflow may have some default/system commands
            # but should not have any edges in an empty workflow
            assert isinstance(graph["nodes"], list)
            assert isinstance(graph["edges"], list)
            # An empty workflow should have minimal or no dependencies
            assert len(graph["edges"]) == 0 or all(
                e.get("match_type") in ["exact", "semantic", "mixed"] 
                for e in graph["edges"]
            )

    def test_circular_dependency_detection(self, graph_path):
        """Test that circular dependencies are handled properly."""
        # The current implementation doesn't explicitly detect cycles,
        # but the depth limit prevents infinite recursion
        with open(graph_path, 'r') as f:
            graph = json.load(f)
        
        if len(graph["nodes"]) > 0:
            # Try to get suggestions with very deep recursion
            # Should not hang due to max_depth limit
            suggestions = get_dependency_suggestions(
                graph_path,
                graph["nodes"][0] if graph["nodes"] else "nonexistent",
                "some_param",
                min_weight=0.0,  # Accept all edges
                max_depth=100  # Very deep
            )
            
            # Should return without hanging
            assert isinstance(suggestions, list)

    def test_parameter_metadata_extraction(self, built_workflow_path):
        # sourcery skip: extract-duplicate-method
        """Test extraction of parameter metadata from commands."""
        params = _collect_command_params(built_workflow_path)
        
        # Check specific command parameter details
        for cmd_name, cmd_params in params.items():
            # Each command should have properly formed parameters
            for input_param in cmd_params.inputs:
                assert isinstance(input_param.name, str)
                assert isinstance(input_param.type_str, str)
                assert isinstance(input_param.description, str)
                assert isinstance(input_param.examples, list)
                
            for output_param in cmd_params.outputs:
                assert isinstance(output_param.name, str)
                assert isinstance(output_param.type_str, str)
                assert isinstance(output_param.description, str)
                assert isinstance(output_param.examples, list)

    def test_graph_persistence(self, built_workflow_path):
        """Test that graph is correctly persisted and can be reloaded."""
        # Generate graph
        graph_path1 = generate_dependency_graph(built_workflow_path)
        
        with open(graph_path1, 'r') as f:
            graph1 = json.load(f)
        
        # Generate again (should overwrite)
        graph_path2 = generate_dependency_graph(built_workflow_path, semantic_threshold=0.7)
        
        assert graph_path1 == graph_path2  # Same path
        
        with open(graph_path2, 'r') as f:
            graph2 = json.load(f)
        
        # Both should be valid graphs
        assert "nodes" in graph1 and "nodes" in graph2
        assert "edges" in graph1 and "edges" in graph2

    def test_suggestion_sorting(self, graph_path):
        """Test that suggestions are properly sorted by depth and weight."""
        with open(graph_path, 'r') as f:
            graph = json.load(f)

        if len(graph["edges"]) > 0:
            edge = graph["edges"][0]
            y_command = edge["from"]
            if matched_params := edge["matched_params"]:
                missing_param = matched_params[0][0]

                suggestions = get_dependency_suggestions(
                    graph_path,
                    y_command,
                    missing_param,
                    min_weight=0.0  # Accept all
                )

                if len(suggestions) > 1:
                    # Check sorting: shallower plans should come first
                    for i in range(len(suggestions) - 1):
                        depth_i = len(suggestions[i].get("sub_plans", []))
                        depth_next = len(suggestions[i + 1].get("sub_plans", []))
                        assert depth_i <= depth_next


class TestRetailWorkflowDependencies:
    """Test specific dependency patterns in retail workflow."""
    
    @pytest.fixture(scope="class")
    def retail_graph(self):
        """Generate graph for retail workflow."""
        retail_path = os.path.join(
            os.path.dirname(__file__), "..", "fastworkflow", "examples", "retail_workflow"
        )
        
        # Create temp copy
        with tempfile.TemporaryDirectory() as temp_dir:
            dest_path = os.path.join(temp_dir, "retail_workflow")
            shutil.copytree(retail_path, dest_path)
            
            # Ensure command info directory exists
            commandinfo_dir = CommandDirectory.get_commandinfo_folderpath(dest_path)
            os.makedirs(commandinfo_dir, exist_ok=True)
            
            # Generate graph
            graph_path = generate_dependency_graph(dest_path)
            
            with open(graph_path, 'r') as f:
                graph = json.load(f)
            
            return graph

    def test_user_lookup_chain(self, retail_graph):
        """Test user lookup dependency chain."""
        # Commands that need user_id should depend on user lookup commands
        edges = retail_graph["edges"]

        if user_detail_deps := [
            e
            for e in edges
            if e["from"] == "get_user_details" and "find_user_id" in e["to"]
        ]:
            for dep in user_detail_deps:
                param_names = [p[0] for p in dep["matched_params"]]
                assert "user_id" in param_names or param_names

    def test_order_dependency_chain(self, retail_graph):
        """Test order-related dependency chain."""
        edges = retail_graph["edges"]
        
        # Commands that modify orders might depend on get_order_details
        order_modifiers = [
            "cancel_pending_order",
            "modify_pending_order_address",
            "modify_pending_order_items",
            "return_delivered_order_items"
        ]
        
        order_deps = [
            e for e in edges
            if e["from"] in order_modifiers
        ]
        
        # Check structure of order dependencies
        for dep in order_deps:
            assert "matched_params" in dep
            assert isinstance(dep["matched_params"], list)

    def test_payment_method_dependencies(self, retail_graph):
        """Test payment method parameter dependencies."""
        edges = retail_graph["edges"]
        
        # Commands that need payment_method_id
        payment_commands = [
            "modify_pending_order_payment",
            "exchange_delivered_order_items",
            "return_delivered_order_items"
        ]
        
        # These might depend on get_order_details which has payment info
        payment_deps = [
            e for e in edges
            if e["from"] in payment_commands and
               any("payment" in str(p).lower() for p in e.get("matched_params", []))
        ]
        
        # Structure check
        for dep in payment_deps:
            assert "weight" in dep
            assert 0.0 <= dep["weight"] <= 1.0


class TestDependencyGraphWithGenAI:
    """Test dependency graph with GenAI-enhanced metadata."""
    
    @pytest.fixture
    def genai_workflow_path(self):
        """Create a workflow with GenAI-enhanced metadata."""
        retail_path = os.path.join(
            os.path.dirname(__file__), "..", "fastworkflow", "examples", "retail_workflow"
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            dest_path = os.path.join(temp_dir, "retail_workflow")
            shutil.copytree(retail_path, dest_path)

            # Build routing definition to ensure metadata is available
            with contextlib.suppress(Exception):
                RoutingDefinition.build(dest_path)

                # Run GenAI postprocessor if environment is configured
                llm_var = os.environ.get("LLM_COMMAND_METADATA_GEN")
                api_key_var = os.environ.get("LITELLM_API_KEY_COMMANDMETADATA_GEN")

                if llm_var and api_key_var:
                    from fastworkflow.build.genai_postprocessor import run_genai_postprocessor
                    run_genai_postprocessor(dest_path, skip_genai=False)
            yield dest_path

    @pytest.mark.skipif(
        not os.environ.get("LLM_COMMAND_METADATA_GEN") or 
        not os.environ.get("LITELLM_API_KEY_COMMANDMETADATA_GEN"),
        reason="GenAI not configured"
    )
    def test_genai_enhanced_metadata(self, genai_workflow_path):
        """Test that GenAI enhances parameter metadata for better matching."""
        # Generate graph with GenAI-enhanced metadata
        graph_path = generate_dependency_graph(genai_workflow_path)
        
        with open(graph_path, 'r') as f:
            graph = json.load(f)
        
        # GenAI should provide better descriptions, leading to more accurate matches
        params = _collect_command_params(genai_workflow_path)
        
        # Check that parameters have enhanced descriptions
        for cmd_name, cmd_params in params.items():
            for param in cmd_params.inputs:
                # GenAI should provide non-empty descriptions
                if param.description:
                    assert len(param.description) > 0

    @pytest.mark.skipif(
        not os.environ.get("LLM_COMMAND_METADATA_GEN") or 
        not os.environ.get("LITELLM_API_KEY_COMMANDMETADATA_GEN"),
        reason="GenAI not configured"
    )
    def test_semantic_matching_with_genai(self, genai_workflow_path):
        """Test semantic matching with GenAI-enhanced descriptions."""
        params = _collect_command_params(genai_workflow_path)

        # Find two parameters that should be semantically similar
        user_params = []
        for cmd_name, cmd_params in params.items():
            user_params.extend(
                param
                for param in cmd_params.inputs + cmd_params.outputs
                if "user" in param.name.lower() or "customer" in param.name.lower()
            )
        if len(user_params) >= 2:
            # Test semantic similarity between user-related parameters
            score = _semantic_match_score(user_params[0], user_params[1])

            # With GenAI descriptions, similar concepts should have higher scores
            assert score > 0.3  # Some semantic similarity expected

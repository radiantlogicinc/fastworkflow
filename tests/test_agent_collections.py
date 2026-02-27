"""
Tests for the agent collection categorization system.
"""

import json
import os
import pytest
from pathlib import Path


@pytest.fixture
def workspace_root():
    """Get the workspace root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def inventory_path(workspace_root):
    """Get the inventory file path."""
    return workspace_root / "workspaces" / "agent-directory" / "inventory.json"


@pytest.fixture
def collections_dir(workspace_root):
    """Get the collections directory."""
    return workspace_root / "workspaces" / "collections"


@pytest.fixture
def inventory_data(inventory_path):
    """Load the inventory data."""
    with open(inventory_path, 'r') as f:
        return json.load(f)


class TestAgentCollections:
    """Tests for agent collection system."""
    
    def test_inventory_file_exists(self, inventory_path):
        """Test that the inventory file exists."""
        assert inventory_path.exists(), "inventory.json should exist"
    
    def test_inventory_structure(self, inventory_data):
        """Test that inventory has required structure."""
        assert 'metadata' in inventory_data
        assert 'workflows' in inventory_data
        assert 'total_workflows' in inventory_data['metadata']
        assert 'total_commands' in inventory_data['metadata']
        assert 'collections' in inventory_data['metadata']
    
    def test_four_collections_exist(self, collections_dir):
        """Test that all four collections exist."""
        expected_collections = ['core', 'business', 'development', 'advanced']
        for collection in expected_collections:
            coll_dir = collections_dir / collection
            assert coll_dir.exists(), f"Collection {collection} should exist"
            assert coll_dir.is_dir(), f"Collection {collection} should be a directory"
    
    def test_collection_metadata_files(self, collections_dir):
        """Test that collection metadata files exist."""
        expected_collections = ['core', 'business', 'development', 'advanced']
        for collection in expected_collections:
            metadata_file = collections_dir / collection / ".collection_metadata.json"
            assert metadata_file.exists(), f"Metadata for {collection} should exist"
            
            # Validate metadata structure
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            assert 'collection' in metadata
            assert 'name' in metadata
            assert 'description' in metadata
            assert metadata['collection'] == collection
    
    def test_collection_readmes(self, collections_dir):
        """Test that collection READMEs exist."""
        expected_collections = ['core', 'business', 'development', 'advanced']
        for collection in expected_collections:
            readme_file = collections_dir / collection / "README.md"
            assert readme_file.exists(), f"README for {collection} should exist"
    
    def test_all_workflows_categorized(self, inventory_data):
        """Test that all workflows have a collection assigned."""
        for workflow in inventory_data['workflows']:
            assert 'collection' in workflow, f"Workflow {workflow['id']} missing collection"
            assert workflow['collection'] in ['core', 'business', 'development', 'advanced'], \
                f"Workflow {workflow['id']} has invalid collection: {workflow['collection']}"
    
    def test_test_workflows_in_development(self, inventory_data):
        """Test that test workflows are in development collection."""
        for workflow in inventory_data['workflows']:
            if workflow.get('type') == 'test':
                assert workflow['collection'] == 'development', \
                    f"Test workflow {workflow['id']} should be in development collection"
    
    def test_workflow_counts(self, inventory_data):
        """Test that workflow counts are consistent."""
        total_workflows = len(inventory_data['workflows'])
        metadata_total = inventory_data['metadata']['total_workflows']
        assert total_workflows == metadata_total, \
            f"Workflow count mismatch: {total_workflows} vs {metadata_total}"
    
    def test_collection_distribution(self, inventory_data):
        """Test that workflows are reasonably distributed across collections."""
        collection_counts = {}
        for workflow in inventory_data['workflows']:
            coll = workflow['collection']
            collection_counts[coll] = collection_counts.get(coll, 0) + 1
        
        # Ensure at least 3 collections have workflows
        non_empty_collections = sum(1 for count in collection_counts.values() if count > 0)
        assert non_empty_collections >= 3, \
            "At least 3 collections should have workflows"
    
    def test_command_counts(self, inventory_data):
        """Test that command counts are tracked."""
        for workflow in inventory_data['workflows']:
            assert 'commands' in workflow, f"Workflow {workflow['id']} missing command count"
            assert isinstance(workflow['commands'], int), \
                f"Workflow {workflow['id']} command count should be integer"
            assert workflow['commands'] >= 0, \
                f"Workflow {workflow['id']} command count should be non-negative"
    
    def test_workflow_metadata(self, inventory_data):
        """Test that workflows have required metadata."""
        required_fields = ['id', 'name', 'path', 'type', 'commands', 'collection', 'description', 'status']
        for workflow in inventory_data['workflows']:
            for field in required_fields:
                assert field in workflow, \
                    f"Workflow {workflow.get('id', 'unknown')} missing field: {field}"
    
    def test_categorization_rules_exist(self, workspace_root):
        """Test that categorization rules document exists."""
        rules_file = workspace_root / "workspaces" / "agent-directory" / "categorization_rules.md"
        assert rules_file.exists(), "Categorization rules document should exist"
    
    def test_workspace_collections_doc_exists(self, workspace_root):
        """Test that workspace collections documentation exists."""
        doc_file = workspace_root / "docs" / "workspace_collections.md"
        assert doc_file.exists(), "Workspace collections documentation should exist"
    
    def test_management_script_exists(self, workspace_root):
        """Test that collection management script exists."""
        script_file = workspace_root / "workspaces" / "agent-directory" / "manage_collections.py"
        assert script_file.exists(), "Management script should exist"
        assert os.access(script_file, os.X_OK), "Management script should be executable"


class TestCollectionCategorization:
    """Tests for specific categorization logic."""
    
    def test_core_collection_has_templates(self, inventory_data):
        """Test that core collection has template workflows."""
        core_workflows = [w for w in inventory_data['workflows'] if w['collection'] == 'core']
        has_template = any('template' in w['name'].lower() or 'template' in w['id'].lower() 
                          for w in core_workflows)
        assert has_template, "Core collection should have template workflows"
    
    def test_business_collection_has_domain_workflows(self, inventory_data):
        """Test that business collection has domain-specific workflows."""
        business_workflows = [w for w in inventory_data['workflows'] if w['collection'] == 'business']
        # Business workflows typically have more commands
        if business_workflows:
            avg_commands = sum(w['commands'] for w in business_workflows) / len(business_workflows)
            assert avg_commands > 2, "Business workflows should have multiple commands on average"
    
    def test_development_collection_has_tests(self, inventory_data):
        """Test that development collection contains test workflows."""
        dev_workflows = [w for w in inventory_data['workflows'] if w['collection'] == 'development']
        test_workflows = [w for w in dev_workflows if w.get('type') == 'test']
        assert len(test_workflows) > 0, "Development collection should have test workflows"
    
    def test_hello_world_in_core(self, inventory_data):
        """Test that hello_world is in core collection."""
        hello_world = next((w for w in inventory_data['workflows'] if 'hello_world' in w['id']), None)
        if hello_world and hello_world.get('type') != 'test':
            assert hello_world['collection'] == 'core', \
                "hello_world example should be in core collection"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

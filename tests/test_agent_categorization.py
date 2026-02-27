"""
Tests for Agent Categorization System

This module tests:
- Agent definition format and schema compliance
- Collection structure and integrity
- Categorization logic and rules
- Manifest completeness
- Cross-references and dependencies
"""

import os
import yaml
import pytest
from pathlib import Path
from typing import Dict, List, Set


WORKSPACE_ROOT = Path(__file__).parent.parent
AGENT_DIR = WORKSPACE_ROOT / "agent-directory"
COLLECTIONS_DIR = WORKSPACE_ROOT / "workspaces" / "collections"


class TestAgentDefinitions:
    """Test agent definition files"""
    
    def test_agent_directory_exists(self):
        """Verify agent-directory exists"""
        assert AGENT_DIR.exists(), "agent-directory not found"
        assert AGENT_DIR.is_dir(), "agent-directory is not a directory"
    
    def test_schema_file_exists(self):
        """Verify schema.yaml exists"""
        schema_file = AGENT_DIR / "schema.yaml"
        assert schema_file.exists(), "schema.yaml not found"
        
        with open(schema_file) as f:
            schema = yaml.safe_load(f)
        
        assert "schema_version" in schema
        assert "required_fields" in schema
        assert "field_definitions" in schema
    
    def test_template_file_exists(self):
        """Verify template.yaml exists"""
        template_file = AGENT_DIR / "template.yaml"
        assert template_file.exists(), "template.yaml not found"
    
    def test_exactly_43_agents(self):
        """Verify exactly 43 agent definition files exist"""
        agent_files = list(AGENT_DIR.glob("[0-9][0-9]_*.yaml"))
        assert len(agent_files) == 43, f"Expected 43 agents, found {len(agent_files)}"
    
    def test_agent_naming_convention(self):
        """Verify agent files follow naming convention"""
        agent_files = list(AGENT_DIR.glob("[0-9][0-9]_*.yaml"))
        
        for agent_file in agent_files:
            name = agent_file.stem
            # Should be like "01_hello_world_agent"
            parts = name.split("_", 1)
            assert len(parts) == 2, f"Invalid agent filename: {name}"
            assert parts[0].isdigit(), f"Agent number not numeric: {parts[0]}"
            assert len(parts[0]) == 2, f"Agent number should be 2 digits: {parts[0]}"
    
    def test_agent_sequential_numbering(self):
        """Verify agents are numbered 01-43"""
        agent_files = sorted(AGENT_DIR.glob("[0-9][0-9]_*.yaml"))
        numbers = [int(f.stem.split("_")[0]) for f in agent_files]
        
        assert numbers == list(range(1, 44)), "Agent numbering is not sequential 1-43"
    
    def test_agent_schema_compliance(self):
        """Verify each agent follows the schema"""
        # Load schema
        with open(AGENT_DIR / "schema.yaml") as f:
            schema = yaml.safe_load(f)
        
        required_fields = schema["required_fields"]
        agent_files = list(AGENT_DIR.glob("[0-9][0-9]_*.yaml"))
        
        for agent_file in agent_files:
            with open(agent_file) as f:
                agent = yaml.safe_load(f)
            
            # Check required fields
            for field in required_fields:
                assert field in agent, f"{agent_file.name} missing required field: {field}"
            
            # Validate field types
            assert isinstance(agent["name"], str), f"{agent_file.name}: name must be string"
            assert isinstance(agent["description"], str), f"{agent_file.name}: description must be string"
            assert isinstance(agent["capabilities"], list), f"{agent_file.name}: capabilities must be list"
            assert isinstance(agent["use_cases"], list), f"{agent_file.name}: use_cases must be list"
            
            # Validate metadata if present
            if "metadata" in agent:
                metadata = agent["metadata"]
                if "complexity" in metadata:
                    assert metadata["complexity"] in ["low", "medium", "high"], \
                        f"{agent_file.name}: invalid complexity value"
                if "integration_level" in metadata:
                    assert metadata["integration_level"] in ["simple", "moderate", "complex"], \
                        f"{agent_file.name}: invalid integration_level value"
    
    def test_agent_categories_valid(self):
        """Verify agent categories are valid"""
        valid_categories = ["basic_workflow", "integration", "complex_workflow", "specialized_domain"]
        agent_files = list(AGENT_DIR.glob("[0-9][0-9]_*.yaml"))
        
        for agent_file in agent_files:
            with open(agent_file) as f:
                agent = yaml.safe_load(f)
            
            assert agent["category"] in valid_categories, \
                f"{agent_file.name}: invalid category '{agent['category']}'"
    
    def test_agent_types_valid(self):
        """Verify agent types are valid"""
        valid_types = ["workflow_agent", "integration_agent", "data_agent", "domain_agent", "utility_agent"]
        agent_files = list(AGENT_DIR.glob("[0-9][0-9]_*.yaml"))
        
        for agent_file in agent_files:
            with open(agent_file) as f:
                agent = yaml.safe_load(f)
            
            assert agent["type"] in valid_types, \
                f"{agent_file.name}: invalid type '{agent['type']}'"


class TestCollectionStructure:
    """Test collection directory structure"""
    
    def test_collections_directory_exists(self):
        """Verify collections directory exists"""
        assert COLLECTIONS_DIR.exists(), "collections directory not found"
    
    def test_four_collections_exist(self):
        """Verify exactly 4 collections exist"""
        collections = [d for d in COLLECTIONS_DIR.iterdir() if d.is_dir()]
        collection_names = [c.name for c in collections]
        
        expected = ["basic_workflow", "integration", "complex_workflow", "specialized_domain"]
        assert sorted(collection_names) == sorted(expected), \
            f"Expected collections {expected}, found {collection_names}"
    
    def test_collection_manifests_exist(self):
        """Verify each collection has a manifest"""
        collections = ["basic_workflow", "integration", "complex_workflow", "specialized_domain"]
        
        for collection in collections:
            manifest = COLLECTIONS_DIR / collection / "manifest.yaml"
            assert manifest.exists(), f"Manifest not found for {collection}"
    
    def test_collection_readmes_exist(self):
        """Verify each collection has a README"""
        collections = ["basic_workflow", "integration", "complex_workflow", "specialized_domain"]
        
        for collection in collections:
            readme = COLLECTIONS_DIR / collection / "README.md"
            assert readme.exists(), f"README not found for {collection}"
    
    def test_index_file_exists(self):
        """Verify collections index.yaml exists"""
        index = COLLECTIONS_DIR / "index.yaml"
        assert index.exists(), "index.yaml not found"
        
        with open(index) as f:
            data = yaml.safe_load(f)
        
        assert data["total_agents"] == 43
        assert data["total_collections"] == 4
        assert len(data["collections"]) == 4


class TestManifests:
    """Test collection manifest files"""
    
    def test_manifest_structure(self):
        """Verify manifest structure is valid"""
        collections = ["basic_workflow", "integration", "complex_workflow", "specialized_domain"]
        
        for collection in collections:
            manifest_path = COLLECTIONS_DIR / collection / "manifest.yaml"
            with open(manifest_path) as f:
                manifest = yaml.safe_load(f)
            
            # Check required fields
            assert "collection_name" in manifest
            assert "total_agents" in manifest
            assert "agents" in manifest
            assert isinstance(manifest["agents"], list)
    
    def test_manifest_agent_counts(self):
        """Verify agent counts in manifests"""
        expected_counts = {
            "basic_workflow": 10,
            "integration": 10,
            "complex_workflow": 10,
            "specialized_domain": 13
        }
        
        for collection, expected_count in expected_counts.items():
            manifest_path = COLLECTIONS_DIR / collection / "manifest.yaml"
            with open(manifest_path) as f:
                manifest = yaml.safe_load(f)
            
            assert manifest["total_agents"] == expected_count, \
                f"{collection}: expected {expected_count} agents, manifest says {manifest['total_agents']}"
            assert len(manifest["agents"]) == expected_count, \
                f"{collection}: expected {expected_count} agents, found {len(manifest['agents'])}"
    
    def test_total_agents_equals_43(self):
        """Verify all manifests together reference 43 agents"""
        collections = ["basic_workflow", "integration", "complex_workflow", "specialized_domain"]
        total = 0
        
        for collection in collections:
            manifest_path = COLLECTIONS_DIR / collection / "manifest.yaml"
            with open(manifest_path) as f:
                manifest = yaml.safe_load(f)
            total += len(manifest["agents"])
        
        assert total == 43, f"Expected 43 total agents across all collections, found {total}"
    
    def test_manifest_agent_ids_match_files(self):
        """Verify manifest agent IDs match actual files"""
        collections = ["basic_workflow", "integration", "complex_workflow", "specialized_domain"]
        
        for collection in collections:
            manifest_path = COLLECTIONS_DIR / collection / "manifest.yaml"
            with open(manifest_path) as f:
                manifest = yaml.safe_load(f)
            
            for agent_entry in manifest["agents"]:
                agent_id = agent_entry["agent_id"]
                expected_file = AGENT_DIR / f"{agent_id}_{agent_entry['name']}.yaml"
                assert expected_file.exists(), \
                    f"Agent file not found: {expected_file}"
    
    def test_no_duplicate_agents_across_collections(self):
        """Verify no agent appears in multiple collections"""
        collections = ["basic_workflow", "integration", "complex_workflow", "specialized_domain"]
        all_agents: Set[str] = set()
        
        for collection in collections:
            manifest_path = COLLECTIONS_DIR / collection / "manifest.yaml"
            with open(manifest_path) as f:
                manifest = yaml.safe_load(f)
            
            for agent_entry in manifest["agents"]:
                agent_id = agent_entry["agent_id"]
                assert agent_id not in all_agents, \
                    f"Agent {agent_id} appears in multiple collections"
                all_agents.add(agent_id)


class TestCategorization:
    """Test categorization logic and rules"""
    
    def test_categorization_rules_exist(self):
        """Verify categorization rules document exists"""
        rules_file = COLLECTIONS_DIR / "categorization_rules.md"
        assert rules_file.exists(), "categorization_rules.md not found"
    
    def test_basic_workflow_agents_match_criteria(self):
        """Verify basic_workflow agents meet the criteria"""
        manifest_path = COLLECTIONS_DIR / "basic_workflow" / "manifest.yaml"
        with open(manifest_path) as f:
            manifest = yaml.safe_load(f)
        
        for agent_entry in manifest["agents"]:
            # Basic workflow should have low-medium complexity and simple integration
            assert agent_entry["complexity"] in ["low", "medium"], \
                f"Agent {agent_entry['name']} in basic_workflow has complexity {agent_entry['complexity']}"
            assert agent_entry["integration_level"] == "simple", \
                f"Agent {agent_entry['name']} in basic_workflow has integration_level {agent_entry['integration_level']}"
    
    def test_integration_agents_match_criteria(self):
        """Verify integration agents meet the criteria"""
        manifest_path = COLLECTIONS_DIR / "integration" / "manifest.yaml"
        with open(manifest_path) as f:
            manifest = yaml.safe_load(f)
        
        for agent_entry in manifest["agents"]:
            # Integration agents should have moderate-complex integration level
            assert agent_entry["integration_level"] in ["moderate", "complex"], \
                f"Agent {agent_entry['name']} in integration has integration_level {agent_entry['integration_level']}"
    
    def test_complex_workflow_agents_match_criteria(self):
        """Verify complex_workflow agents meet the criteria"""
        manifest_path = COLLECTIONS_DIR / "complex_workflow" / "manifest.yaml"
        with open(manifest_path) as f:
            manifest = yaml.safe_load(f)
        
        for agent_entry in manifest["agents"]:
            # Complex workflow should have high complexity
            assert agent_entry["complexity"] == "high", \
                f"Agent {agent_entry['name']} in complex_workflow has complexity {agent_entry['complexity']}"
    
    def test_specialized_domain_agents_have_domain(self):
        """Verify specialized_domain agents have domain field"""
        manifest_path = COLLECTIONS_DIR / "specialized_domain" / "manifest.yaml"
        with open(manifest_path) as f:
            manifest = yaml.safe_load(f)
        
        for agent_entry in manifest["agents"]:
            # Specialized domain agents should have a domain field
            assert "domain" in agent_entry, \
                f"Agent {agent_entry['name']} in specialized_domain missing domain field"
            assert agent_entry["domain"], \
                f"Agent {agent_entry['name']} has empty domain field"


class TestDocumentation:
    """Test documentation completeness"""
    
    def test_main_guide_exists(self):
        """Verify main agent categorization guide exists"""
        guide = WORKSPACE_ROOT / "docs" / "agent_categorization_guide.md"
        assert guide.exists(), "agent_categorization_guide.md not found"
    
    def test_agent_directory_readme_exists(self):
        """Verify agent-directory README exists"""
        readme = AGENT_DIR / "README.md"
        assert readme.exists(), "agent-directory/README.md not found"
    
    def test_workspaces_readme_exists(self):
        """Verify workspaces README exists"""
        readme = WORKSPACE_ROOT / "workspaces" / "README.md"
        assert readme.exists(), "workspaces/README.md not found"
    
    def test_collections_readme_exists(self):
        """Verify collections README exists"""
        readme = COLLECTIONS_DIR / "README.md"
        assert readme.exists(), "workspaces/collections/README.md not found"


class TestIntegrity:
    """Test overall system integrity"""
    
    def test_all_agent_numbers_accounted_for(self):
        """Verify agents numbered 01-43 all exist and are in manifests"""
        # Get all agent files
        agent_files = {int(f.stem.split("_")[0]): f for f in AGENT_DIR.glob("[0-9][0-9]_*.yaml")}
        assert len(agent_files) == 43
        
        # Get all agents from manifests
        collections = ["basic_workflow", "integration", "complex_workflow", "specialized_domain"]
        manifest_agents: Set[int] = set()
        
        for collection in collections:
            manifest_path = COLLECTIONS_DIR / collection / "manifest.yaml"
            with open(manifest_path) as f:
                manifest = yaml.safe_load(f)
            
            for agent_entry in manifest["agents"]:
                manifest_agents.add(int(agent_entry["agent_id"]))
        
        # Verify same set
        assert agent_files.keys() == manifest_agents, \
            "Mismatch between agent files and manifest entries"
    
    def test_agent_names_match_between_definition_and_manifest(self):
        """Verify agent names are consistent"""
        collections = ["basic_workflow", "integration", "complex_workflow", "specialized_domain"]
        
        for collection in collections:
            manifest_path = COLLECTIONS_DIR / collection / "manifest.yaml"
            with open(manifest_path) as f:
                manifest = yaml.safe_load(f)
            
            for agent_entry in manifest["agents"]:
                agent_id = agent_entry["agent_id"]
                manifest_name = agent_entry["name"]
                
                # Load agent definition
                agent_file = AGENT_DIR / f"{agent_id}_{manifest_name}.yaml"
                with open(agent_file) as f:
                    agent_def = yaml.safe_load(f)
                
                # Verify names match
                assert agent_def["name"] == manifest_name, \
                    f"Name mismatch for agent {agent_id}: definition={agent_def['name']}, manifest={manifest_name}"
    
    def test_agent_categories_match_collection(self):
        """Verify agent category field matches its collection"""
        collections = ["basic_workflow", "integration", "complex_workflow", "specialized_domain"]
        
        for collection in collections:
            manifest_path = COLLECTIONS_DIR / collection / "manifest.yaml"
            with open(manifest_path) as f:
                manifest = yaml.safe_load(f)
            
            for agent_entry in manifest["agents"]:
                agent_id = agent_entry["agent_id"]
                manifest_name = agent_entry["name"]
                
                # Load agent definition
                agent_file = AGENT_DIR / f"{agent_id}_{manifest_name}.yaml"
                with open(agent_file) as f:
                    agent_def = yaml.safe_load(f)
                
                # Verify category matches collection
                assert agent_def["category"] == collection, \
                    f"Category mismatch for {manifest_name}: in {collection} collection but category is {agent_def['category']}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

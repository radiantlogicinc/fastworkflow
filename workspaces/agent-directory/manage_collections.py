#!/usr/bin/env python3
"""
Agent Collection Management Tool

This script helps manage the fastWorkflow agent collections,
including listing, categorizing, and validating workflows.
"""

import json
import os
from pathlib import Path
from typing import Dict, List


class CollectionManager:
    """Manages fastWorkflow workspace collections."""
    
    def __init__(self, workspace_root: str = None):
        if workspace_root is None:
            workspace_root = Path(__file__).parent.parent
        self.workspace_root = Path(workspace_root)
        self.collections_dir = self.workspace_root / "collections"
        self.inventory_path = self.workspace_root / "agent-directory" / "inventory.json"
        
    def load_inventory(self) -> Dict:
        """Load the agent inventory."""
        with open(self.inventory_path, 'r') as f:
            return json.load(f)
    
    def save_inventory(self, inventory: Dict):
        """Save the agent inventory."""
        with open(self.inventory_path, 'w') as f:
            json.dump(inventory, f, indent=2)
    
    def list_collections(self) -> List[str]:
        """List all available collections."""
        collections = []
        for item in self.collections_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                collections.append(item.name)
        return sorted(collections)
    
    def get_collection_workflows(self, collection: str) -> List[Dict]:
        """Get all workflows in a collection."""
        inventory = self.load_inventory()
        return [w for w in inventory['workflows'] if w.get('collection') == collection]
    
    def assign_workflow(self, workflow_id: str, collection: str):
        """Assign a workflow to a collection."""
        inventory = self.load_inventory()
        
        # Find and update workflow
        for workflow in inventory['workflows']:
            if workflow['id'] == workflow_id:
                old_collection = workflow.get('collection')
                workflow['collection'] = collection
                print(f"Moved '{workflow_id}' from '{old_collection}' to '{collection}'")
                break
        else:
            print(f"Workflow '{workflow_id}' not found in inventory")
            return
        
        # Update metadata
        self._update_collection_counts(inventory)
        self.save_inventory(inventory)
    
    def _update_collection_counts(self, inventory: Dict):
        """Update collection counts in metadata."""
        for collection in ['core', 'business', 'development', 'advanced']:
            workflows = [w for w in inventory['workflows'] if w.get('collection') == collection]
            inventory['metadata']['collections'][collection] = [w['id'] for w in workflows]
    
    def validate_categorization(self) -> List[str]:
        """Validate that all workflows are properly categorized."""
        issues = []
        inventory = self.load_inventory()
        
        for workflow in inventory['workflows']:
            # Check if collection is set
            if not workflow.get('collection'):
                issues.append(f"Workflow '{workflow['id']}' has no collection assigned")
            
            # Check if collection exists
            elif workflow['collection'] not in ['core', 'business', 'development', 'advanced']:
                issues.append(f"Workflow '{workflow['id']}' has invalid collection: {workflow['collection']}")
            
            # Check consistency with rules
            if workflow['type'] == 'test' and workflow.get('collection') != 'development':
                issues.append(f"Warning: Test workflow '{workflow['id']}' is not in development collection")
        
        return issues
    
    def generate_report(self) -> str:
        """Generate a categorization report."""
        inventory = self.load_inventory()
        
        report = []
        report.append("=" * 60)
        report.append("FASTWORKFLOW AGENT COLLECTION REPORT")
        report.append("=" * 60)
        report.append(f"Total Workflows: {inventory['metadata']['total_workflows']}")
        report.append(f"Total Commands: {inventory['metadata']['total_commands']}")
        report.append("")
        
        for collection in ['core', 'business', 'development', 'advanced']:
            workflows = self.get_collection_workflows(collection)
            commands = sum(w.get('commands', 0) for w in workflows)
            
            report.append(f"\n{collection.upper()} COLLECTION")
            report.append("-" * 40)
            report.append(f"Workflows: {len(workflows)}")
            report.append(f"Commands: {commands}")
            
            if workflows:
                report.append("\nWorkflows:")
                for w in workflows:
                    report.append(f"  - {w['id']} ({w.get('commands', 0)} commands)")
        
        return "\n".join(report)


def main():
    """Main entry point."""
    import sys
    
    manager = CollectionManager()
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python manage_collections.py list              # List collections")
        print("  python manage_collections.py report            # Generate report")
        print("  python manage_collections.py validate          # Validate categorization")
        print("  python manage_collections.py assign <id> <coll> # Assign workflow")
        return
    
    command = sys.argv[1]
    
    if command == 'list':
        print("Available collections:")
        for coll in manager.list_collections():
            print(f"  - {coll}")
    
    elif command == 'report':
        print(manager.generate_report())
    
    elif command == 'validate':
        issues = manager.validate_categorization()
        if issues:
            print("Validation issues found:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("✓ All workflows properly categorized")
    
    elif command == 'assign' and len(sys.argv) == 4:
        workflow_id = sys.argv[2]
        collection = sys.argv[3]
        manager.assign_workflow(workflow_id, collection)
    
    else:
        print(f"Unknown command: {command}")


if __name__ == '__main__':
    main()

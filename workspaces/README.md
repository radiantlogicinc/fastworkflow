# FastWorkflow Workspace Collections

## Overview
This workspace contains the organizational structure for categorizing and managing fastWorkflow agents and workflows into four logical collections.

## Quick Links
- **[Collections Overview](../docs/workspace_collections.md)** - Complete documentation of the collection system
- **[Categorization Rules](agent-directory/categorization_rules.md)** - Rules for assigning workflows to collections
- **[Agent Inventory](agent-directory/inventory.json)** - Complete inventory of all workflows

## The Four Collections

### 1. [Core & Foundation](collections/core/)
Essential workflows and fundamental system agents for learning and templates.
- **Workflows:** 4
- **Commands:** 18
- **Audience:** Beginners, learners, template-seekers

### 2. [Business & Domain](collections/business/)
Domain-specific workflows for real-world business applications.
- **Workflows:** 4
- **Commands:** 31
- **Audience:** Production developers, domain experts

### 3. [Development & Tools](collections/development/)
Testing, development utilities, and tooling workflows.
- **Workflows:** 5
- **Commands:** 26
- **Audience:** Contributors, testers, framework developers

### 4. [Advanced & Specialized](collections/advanced/)
Complex, experimental, and research-focused workflows.
- **Workflows:** 0
- **Commands:** 0
- **Audience:** Researchers, advanced users

## Directory Structure

```
workspaces/
├── README.md                           # This file
├── collections/                        # The four collections
│   ├── core/
│   │   ├── README.md
│   │   └── .collection_metadata.json
│   ├── business/
│   │   ├── README.md
│   │   └── .collection_metadata.json
│   ├── development/
│   │   ├── README.md
│   │   └── .collection_metadata.json
│   └── advanced/
│       ├── README.md
│       └── .collection_metadata.json
└── agent-directory/                    # Management and inventory
    ├── inventory.json                  # Complete workflow inventory
    ├── categorization_rules.md         # Categorization guidelines
    └── manage_collections.py           # Management tool
```

## Using the Collections

### For Users
1. **Start with Core** if you're new to fastWorkflow or need templates
2. **Browse Business** for production-ready, domain-specific solutions
3. **Reference Development** for testing patterns and development tools
4. **Explore Advanced** for cutting-edge features and experimental work

### For Contributors
When adding or modifying workflows:
1. Review the [categorization rules](agent-directory/categorization_rules.md)
2. Use the management tool to validate categorization
3. Update the inventory and collection READMEs
4. Run tests to ensure consistency

## Management Tools

### Collection Manager CLI
```bash
# Generate a report
python workspaces/agent-directory/manage_collections.py report

# Validate categorization
python workspaces/agent-directory/manage_collections.py validate

# List collections
python workspaces/agent-directory/manage_collections.py list

# Assign a workflow to a collection
python workspaces/agent-directory/manage_collections.py assign <workflow-id> <collection>
```

### Running Tests
```bash
# Run collection system tests
pytest tests/test_agent_collections.py -v
```

## Statistics

- **Total Workflows:** 13
- **Total Commands:** 75
- **Collections:** 4
- **Test Coverage:** 18 test cases

## Categorization Philosophy

The categorization system is designed to:
- **Help users find relevant workflows quickly**
- **Organize complexity progressively** (simple → advanced)
- **Separate concerns** (learning vs. production vs. development)
- **Evolve with the framework** (workflows can migrate between collections)

## Maintenance

### Regular Reviews
Collections are reviewed quarterly to:
- Ensure proper categorization
- Migrate evolved workflows
- Archive deprecated workflows
- Update documentation

### Contributing
When contributing:
1. Follow the categorization rules
2. Update inventory.json
3. Run validation tests
4. Update collection READMEs
5. Document changes

## Support

For questions about:
- **Categorization:** See [categorization_rules.md](agent-directory/categorization_rules.md)
- **Collection details:** See individual collection READMEs
- **Overall system:** See [docs/workspace_collections.md](../docs/workspace_collections.md)

---

*This categorization system helps organize fastWorkflow's growing ecosystem of agents and workflows for better discoverability and usability.*

# Agent Directory

This directory contains standardized definitions for 43 different agent types that can be used with fastWorkflow.

## Structure

Each agent is defined in its own YAML file with the following structure:

```yaml
name: Agent Name
type: agent_type
category: category_name
description: Brief description of what this agent does
capabilities:
  - Capability 1
  - Capability 2
use_cases:
  - Use case 1
  - Use case 2
required_tools:
  - tool_1
  - tool_2
dependencies:
  - dependency_1
example_workflow: path/to/example
metadata:
  complexity: low|medium|high
  domain: domain_name
  integration_level: simple|moderate|complex
```

## Agent Categories

Agents are organized into 4 main collections based on their purpose and complexity:

1. **Basic Workflow Agents** - Simple, single-purpose agents
2. **Integration Agents** - Agents that integrate with external systems
3. **Complex Workflow Agents** - Multi-step, stateful agents
4. **Specialized Domain Agents** - Domain-specific agents (retail, messaging, etc.)

## Adding New Agents

To add a new agent:

1. Create a new YAML file in this directory
2. Follow the standardized structure above
3. Run categorization to assign it to the appropriate collection
4. Update tests to include the new agent

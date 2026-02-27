# Workspaces

This directory contains collections of agents organized by their purpose, complexity, and domain.

## Structure

The workspaces are organized into collections that group similar agents together for easier discovery and management.

### Collections

The `/collections` directory contains 4 main buckets:

1. **basic_workflow** - Simple, single-purpose agents that are easy to understand and implement
2. **integration** - Agents that integrate with external systems and services
3. **complex_workflow** - Multi-step, stateful agents with advanced workflow capabilities
4. **specialized_domain** - Domain-specific agents for particular industries or use cases

## Usage

Each collection contains:
- A manifest file listing all agents in that collection
- Reference links to agent definitions in the agent-directory
- Collection-specific documentation

## Adding Agents to Collections

Agents are categorized based on:
- **Complexity**: How complex the agent's logic and workflows are
- **Integration Level**: How many external systems it integrates with
- **Domain Specificity**: Whether it's general-purpose or domain-specific
- **Use Case Scope**: How broad or narrow its applicability is

See `/workspaces/collections/categorization_rules.md` for detailed criteria.

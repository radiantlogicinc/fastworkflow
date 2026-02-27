# Agent Categorization System Guide

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Understanding Collections](#understanding-collections)
4. [Using the Agent Directory](#using-the-agent-directory)
5. [Finding the Right Agent](#finding-the-right-agent)
6. [Adding New Agents](#adding-new-agents)
7. [Categorization Reference](#categorization-reference)
8. [Examples](#examples)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

---

## Overview

The fastWorkflow agent categorization system organizes 43 pre-built agents into 4 logical collections, making it easy to discover and use the right agent for your needs.

### What is an Agent?

In fastWorkflow, an agent is a specialized workflow component that handles specific tasks or processes. Each agent is defined with:
- Clear capabilities
- Standardized interfaces
- Documented use cases
- Integration requirements

### The 4 Collections

| Collection | Count | Purpose |
|------------|-------|---------|
| **Basic Workflow** | 10 | Learning and simple automation |
| **Integration** | 10 | External system connectivity |
| **Complex Workflow** | 10 | Advanced multi-step processes |
| **Specialized Domain** | 13 | Industry-specific solutions |

---

## Quick Start

### 1. Browse Available Agents

```bash
# View all collections
ls workspaces/collections/

# View agents in a collection
cat workspaces/collections/basic_workflow/manifest.yaml
```

### 2. Find an Agent

```bash
# Search by name
find agent-directory -name "*email*"

# Search by capability
grep -r "payment processing" agent-directory/
```

### 3. Review Agent Details

```bash
# Read agent definition
cat agent-directory/01_hello_world_agent.yaml
```

### 4. Use the Agent

Follow the agent's `example_workflow` path for implementation guidance.

---

## Understanding Collections

### Basic Workflow Collection

**Purpose**: Foundational agents for learning and simple tasks

**When to Use**:
- Learning fastWorkflow
- Building proof-of-concepts
- Simple automation needs
- Testing and prototyping

**Agents Include**:
- Hello World Agent
- Calculator Agent
- File Manager Agent
- Todo Manager Agent
- And 6 more...

**Characteristics**:
- ✅ Minimal dependencies
- ✅ Quick setup (< 30 min)
- ✅ Beginner-friendly
- ✅ Well-documented examples

---

### Integration Collection

**Purpose**: Connect fastWorkflow to external systems

**When to Use**:
- Integrating with third-party services
- Building data pipelines
- Automating cross-system workflows
- Cloud service integration

**Agents Include**:
- Database Connector Agent
- REST API Agent
- AWS Integration Agent
- Slack Integration Agent
- And 6 more...

**Characteristics**:
- ⚙️ Requires API credentials
- ⚙️ External dependencies
- ⚙️ Moderate complexity
- ⚙️ Production-ready

**Prerequisites**:
- API keys or credentials
- Network connectivity
- Service account setup

---

### Complex Workflow Collection

**Purpose**: Sophisticated multi-step business processes

**When to Use**:
- Enterprise workflow automation
- State machine implementations
- Multi-context processes
- Orchestration needs

**Agents Include**:
- Retail Workflow Agent
- Approval Workflow Agent
- ETL Pipeline Agent
- CI/CD Pipeline Agent
- And 6 more...

**Characteristics**:
- 🚀 Advanced state management
- 🚀 Multiple workflow stages
- 🚀 High sophistication
- 🚀 Production-grade

**Prerequisites**:
- Advanced fastWorkflow knowledge
- State machine concepts
- Multiple integrated systems

---

### Specialized Domain Collection

**Purpose**: Industry-specific workflow solutions

**When to Use**:
- Building vertical solutions
- Industry-specific automation
- Compliance-driven workflows
- Domain expertise required

**Agents Include**:
- Healthcare Patient Agent
- Financial Trading Agent
- Legal Document Agent
- Insurance Claims Agent
- And 9 more...

**Industries Covered**:
- Healthcare
- Finance
- Legal
- Real Estate
- Education
- Insurance
- Manufacturing
- Hospitality
- Logistics
- Events
- Marketing

**Characteristics**:
- 🏥 Domain-specific
- 🏥 Compliance aware
- 🏥 Industry workflows
- 🏥 Specialized integrations

**Prerequisites**:
- Domain knowledge
- Industry compliance understanding
- Specialized tools/services

---

## Using the Agent Directory

### Directory Structure

```
agent-directory/
├── README.md                    # Overview and getting started
├── schema.yaml                  # Agent definition schema
├── template.yaml               # Template for new agents
├── 01_hello_world_agent.yaml   # Agent definitions (43 total)
├── 02_messaging_agent.yaml
└── ...
```

### Agent Definition Format

Each agent is defined in YAML with:

```yaml
name: agent_name
type: agent_type
category: collection_category
description: What this agent does
capabilities: [list of capabilities]
use_cases: [when to use this agent]
required_tools: [dependencies]
dependencies: [other agents]
example_workflow: path/to/example
metadata:
  complexity: low|medium|high
  domain: domain_name
  integration_level: simple|moderate|complex
  tags: [searchable tags]
```

---

## Finding the Right Agent

### Decision Matrix

```
START
  |
  ├─ Need industry-specific solution?
  │   └─ YES → Specialized Domain Collection
  │
  ├─ Integrating with external service?
  │   └─ YES → Integration Collection
  │
  ├─ Building complex multi-step workflow?
  │   └─ YES → Complex Workflow Collection
  │
  └─ Learning or simple automation?
      └─ YES → Basic Workflow Collection
```

### Search by Use Case

| Need | Collection | Recommended Agents |
|------|------------|-------------------|
| Send emails | Integration | `email_integration_agent`, `sendgrid_email_agent` |
| Process payments | Integration | `stripe_payment_agent` |
| Manage tasks | Basic Workflow | `todo_manager_agent`, `work_item_manager_agent` |
| E-commerce | Complex Workflow | `retail_workflow_agent` |
| Healthcare | Specialized Domain | `healthcare_patient_agent` |
| CI/CD | Complex Workflow | `cicd_pipeline_agent` |

### Search by Tags

```bash
# Find all agents with specific tag
grep -r "tag: ecommerce" workspaces/collections/
grep -r "tag: compliance" agent-directory/
```

---

## Adding New Agents

### Step 1: Create Agent Definition

```bash
# Copy template
cp agent-directory/template.yaml agent-directory/44_my_new_agent.yaml

# Edit with your agent details
vim agent-directory/44_my_new_agent.yaml
```

### Step 2: Score Your Agent

Use the categorization worksheet:

| Dimension | Score (1-10) |
|-----------|--------------|
| Complexity | ___ |
| Integration Level | ___ |
| Domain Specificity | ___ |
| Workflow Scope | ___ |

See `workspaces/collections/categorization_rules.md` for scoring guidance.

### Step 3: Assign to Collection

Based on your scores, add to the appropriate collection manifest:

```yaml
# Example: adding to basic_workflow/manifest.yaml
agents:
  - agent_id: "44"
    name: my_new_agent
    path: ../../agent-directory/44_my_new_agent.yaml
    complexity: low
    integration_level: simple
    tags: [my, tags]
```

### Step 4: Update Documentation

- Add to collection README
- Update collection count
- Add examples if available
- Update index.yaml

### Step 5: Test

- Validate YAML syntax
- Test agent functionality
- Verify categorization
- Add to test suite

---

## Categorization Reference

### Complexity Levels

**Low (1-3)**
- Single-purpose
- Simple logic
- Minimal state management
- Direct execution

**Medium (4-6)**
- Multi-purpose
- Moderate logic
- Some state management
- Conditional flows

**High (7-10)**
- Complex multi-step
- State machines
- Advanced orchestration
- Error recovery

### Integration Levels

**Simple (1-3)**
- 0-2 external systems
- Basic auth
- Direct API calls

**Moderate (4-6)**
- 2-4 external systems
- OAuth/token auth
- Data transformation
- Retry logic

**Complex (7-10)**
- 5+ external systems
- Multiple auth methods
- Circuit breakers
- Advanced error handling

### Domain Specificity

**General (1-3)**
- Cross-industry
- No domain knowledge needed

**Moderate (4-6)**
- Some industry focus
- Basic domain knowledge helpful

**Specialized (7-10)**
- Industry-specific
- Deep domain expertise required
- Compliance requirements

---

## Examples

### Example 1: Building a Simple Notification System

**Requirement**: Send email notifications when tasks are completed

**Solution**:
1. Start with `notification_agent` (Basic Workflow)
2. Add `email_integration_agent` (Integration)
3. Integrate with `todo_manager_agent` (Basic Workflow)

**Implementation**:
```python
from fastworkflow import Workflow
# Use notification_agent and email_integration_agent
# See examples/notification_workflow
```

---

### Example 2: E-commerce Platform

**Requirement**: Full e-commerce workflow with payments

**Solution**:
1. Use `retail_workflow_agent` (Complex Workflow)
2. Add `stripe_payment_agent` (Integration)
3. Add `database_connector_agent` (Integration)
4. Add `email_integration_agent` (Integration)

**Implementation**:
See `examples/retail_workflow`

---

### Example 3: Healthcare Patient Management

**Requirement**: HIPAA-compliant patient workflows

**Solution**:
1. Use `healthcare_patient_agent` (Specialized Domain)
2. Add secure `database_connector_agent` (Integration)
3. Add `notification_agent` for appointments (Basic Workflow)

**Implementation**:
See `examples/healthcare_patient_workflow`

---

## Best Practices

### Choosing Agents

✅ **DO**:
- Start with the simplest agent that meets your needs
- Combine multiple agents when appropriate
- Review agent dependencies before committing
- Test with example workflows first

❌ **DON'T**:
- Use complex agents for simple tasks
- Ignore integration requirements
- Skip security/compliance considerations
- Modify core agent definitions directly

### Extending Agents

✅ **DO**:
- Inherit from existing agents
- Add domain-specific logic in subclasses
- Document your extensions
- Contribute back to the community

❌ **DON'T**:
- Modify agent core behavior
- Remove required capabilities
- Break backward compatibility

### Security

✅ **DO**:
- Use environment variables for credentials
- Implement proper authentication
- Follow security best practices
- Regular security audits

❌ **DON'T**:
- Hardcode API keys
- Disable security features
- Ignore compliance requirements

---

## Troubleshooting

### Agent Not Found

**Problem**: Can't find the right agent for your use case

**Solution**:
1. Review collection READMEs
2. Search by tags: `grep -r "your-tag" agent-directory/`
3. Check categorization guide
4. Consider combining multiple agents

### Integration Failures

**Problem**: Agent fails to connect to external service

**Solution**:
1. Verify credentials/API keys
2. Check network connectivity
3. Review service documentation
4. Check rate limits

### Categorization Confusion

**Problem**: Unsure which collection an agent belongs to

**Solution**:
1. Review categorization rules
2. Score the agent across all dimensions
3. Apply the decision tree
4. Consult examples

### Missing Dependencies

**Problem**: Agent requires tools not installed

**Solution**:
1. Check agent's `required_tools` field
2. Install dependencies
3. Verify versions
4. Check example workflow setup

---

## Additional Resources

- **Agent Directory**: `/agent-directory/`
- **Collections**: `/workspaces/collections/`
- **Categorization Rules**: `/workspaces/collections/categorization_rules.md`
- **Examples**: `/examples/`
- **Schema Reference**: `/agent-directory/schema.yaml`

---

## Support

For questions or issues:
1. Check this guide
2. Review collection READMEs
3. Check example workflows
4. Consult fastWorkflow documentation
5. Join our Discord community

---

*Last Updated: 2026-02-27*
*Version: 1.0*

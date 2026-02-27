# FastWorkflow Workspace Collections

## Overview
This document defines the four workspace collections used to organize and categorize fastWorkflow agents and workflows. Each collection represents a logical grouping based on purpose, complexity, and use case.

## The Four Collections

### 1. Core & Foundation (`core`)
**Purpose**: Essential workflows and fundamental system agents that form the foundation of fastWorkflow.

**Characteristics**:
- Basic, reusable workflows
- Core system functionality
- Minimal dependencies
- Educational examples for new users
- Template workflows

**Examples**:
- hello_world workflow
- simple_workflow_template
- Basic command examples
- Core system agents

---

### 2. Business & Domain (`business`)
**Purpose**: Domain-specific workflows for real-world business applications.

**Characteristics**:
- Industry-specific implementations
- Production-ready workflows
- Complex business logic
- Real-world use cases
- Domain models (retail, airline, finance, etc.)

**Examples**:
- Retail workflow (Tau Bench)
- Airline workflow (Tau Bench)
- Messaging applications
- E-commerce workflows
- Customer service agents

---

### 3. Development & Tools (`development`)
**Purpose**: Workflows and agents for development, testing, and tooling purposes.

**Characteristics**:
- Testing frameworks
- Development utilities
- Build and deployment tools
- Code generation workflows
- Testing examples

**Examples**:
- todo_list_workflow (testing)
- Command router tests
- Build tool workflows
- Development examples
- Testing utilities

---

### 4. Advanced & Specialized (`advanced`)
**Purpose**: Complex, specialized workflows with advanced features or experimental capabilities.

**Characteristics**:
- Multi-agent systems
- Advanced DSPy integration
- Experimental features
- High complexity workflows
- Research and innovation

**Examples**:
- Agentic workflows with ReAct
- Multi-context navigation
- Advanced planning agents
- Research prototypes
- Experimental integrations

---

## Categorization Rules

### Priority-Based Assignment
1. **Core**: If the workflow is a template, example, or fundamental building block
2. **Business**: If the workflow solves a specific business domain problem
3. **Development**: If the workflow is primarily for testing, development, or tooling
4. **Advanced**: If the workflow demonstrates advanced capabilities or experimental features

### Complexity Indicators
- **Core**: Low to medium complexity, well-documented, beginner-friendly
- **Business**: Medium to high complexity, production-focused
- **Development**: Variable complexity, developer-focused
- **Advanced**: High complexity, requires deep understanding

### Audience
- **Core**: New users, learners, template seekers
- **Business**: Production developers, domain experts
- **Development**: Contributors, testers, framework developers
- **Advanced**: Researchers, advanced users, innovators

---

## Directory Structure

```
workspaces/
├── collections/
│   ├── core/
│   │   ├── README.md
│   │   ├── hello_world/
│   │   ├── simple_workflow_template/
│   │   └── ...
│   ├── business/
│   │   ├── README.md
│   │   ├── retail/
│   │   ├── airline/
│   │   └── ...
│   ├── development/
│   │   ├── README.md
│   │   ├── todo_list_workflow/
│   │   ├── test_workflows/
│   │   └── ...
│   └── advanced/
│       ├── README.md
│       ├── multi_agent_systems/
│       ├── experimental/
│       └── ...
└── agent-directory/
    └── inventory.json
```

---

## Usage

### For Users
Browse collections to find workflows that match your needs:
- Start with **Core** for learning and templates
- Use **Business** for production applications
- Reference **Development** for testing patterns
- Explore **Advanced** for cutting-edge features

### For Contributors
When adding a new workflow:
1. Review the categorization rules
2. Assess complexity and audience
3. Place in the appropriate collection
4. Update the agent inventory
5. Document in the collection README

---

## Maintenance

### Regular Reviews
- Quarterly review of agent categorization
- Migrate agents between collections as they mature
- Archive deprecated workflows
- Update inventory and documentation

### Version Control
- All collection changes tracked in git
- Agent inventory maintained in `agent-directory/inventory.json`
- Collection READMEs updated with each addition/removal

---

*Last updated: 2026-02-27*

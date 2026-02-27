# Agent Categorization Rules

## Overview
This document defines the rules and criteria for categorizing fastWorkflow agents and workflows into the four workspace collections.

## Quick Decision Tree

```
Is it a template or basic example?
├─ YES → CORE
└─ NO
   └─ Is it for testing/development?
      ├─ YES → DEVELOPMENT
      └─ NO
         └─ Is it domain-specific for business use?
            ├─ YES → BUSINESS
            └─ NO → ADVANCED
```

## Detailed Categorization Criteria

### 1. Core & Foundation

**Primary Indicators:**
- Named with "template", "hello", "simple", or "basic"
- Explicitly designed for learning or starting new projects
- Minimal external dependencies
- Well-documented with tutorials
- Low complexity (< 5 commands typically)

**Secondary Indicators:**
- Referenced in quickstart documentation
- Used as examples in README
- Frequently copied by users
- Foundation for other workflows

**Examples:**
- hello_world
- simple_workflow_template  
- messaging_app_1
- extended_workflow_example

**Command Count:** Typically 1-5 commands

---

### 2. Business & Domain

**Primary Indicators:**
- Implements specific business domain logic
- Named after business domains (retail, airline, finance, etc.)
- Production-ready with error handling
- Real-world use case implementation
- Medium to high complexity (5-20 commands)

**Secondary Indicators:**
- Includes domain models and data structures
- Has comprehensive validation
- Designed for production deployment
- Solves specific industry problems

**Examples:**
- retail_workflow (Tau Bench)
- messaging_app_2, messaging_app_3, messaging_app_4
- Industry-specific implementations

**Command Count:** Typically 5-20 commands

---

### 3. Development & Tools

**Primary Indicators:**
- Located in `tests/` directory
- Named with "test_" prefix
- Used for testing framework features
- Development utilities or build tools
- Variable complexity

**Secondary Indicators:**
- Used by contributors/developers
- Part of CI/CD pipeline
- Demonstrates testing patterns
- Framework validation

**Examples:**
- todo_list_workflow (test)
- hello_world_test
- example_workflow_test
- All test_* workflows

**Command Count:** Varies widely (1-25+ commands)

---

### 4. Advanced & Specialized

**Primary Indicators:**
- Multi-agent orchestration
- Experimental features
- Research prototypes
- Complex DSPy integration
- High complexity (20+ commands or sophisticated logic)

**Secondary Indicators:**
- Requires deep framework knowledge
- Uses cutting-edge features
- Not production-ready
- Innovation/research focus

**Examples:**
- ReAct agent implementations
- Multi-context navigation systems
- Experimental integrations
- Complex planning agents

**Command Count:** Typically 20+ commands or complex logic

---

## Step-by-Step Categorization Process

### Step 1: Identify the Workflow
- Locate the workflow directory
- Review README or documentation
- Examine command structure

### Step 2: Gather Metrics
- Count commands (`_commands` directory)
- Assess complexity
- Identify dependencies
- Review domain/purpose

### Step 3: Apply Primary Rules
1. Check if it's a template or basic example → CORE
2. Check if it's in `tests/` or for development → DEVELOPMENT
3. Check if it implements business domain → BUSINESS
4. Check if it's experimental/advanced → ADVANCED

### Step 4: Apply Secondary Rules
- Review naming conventions
- Assess production readiness
- Consider target audience
- Evaluate complexity level

### Step 5: Assign Collection
- Place in primary collection based on rules
- Update inventory.json
- Update collection README

### Step 6: Document
- Add entry to collection README
- Update agent directory inventory
- Note any special considerations

---

## Special Cases

### Workflows with Multiple Characteristics
**Rule:** Use the PRIMARY purpose/audience
- A complex test workflow → DEVELOPMENT (testing is primary)
- A business template → CORE (template is primary)
- An advanced business app → BUSINESS (production use is primary)

### Evolving Workflows
**Rule:** Workflows can migrate between collections
- Experimental → Advanced → Business (as it matures)
- Example → Template → Core (as it's adopted)

**Process:**
1. Document reason for migration
2. Update inventory.json
3. Move/link in collections
4. Update READMEs

### Ambiguous Cases
**Rule:** Choose based on INTENDED audience
- If uncertain between two collections, pick the one with lower barrier to entry
- When in doubt, prefer CORE or DEVELOPMENT for accessibility

---

## Validation Checklist

Before finalizing categorization:
- [ ] Workflow purpose clearly understood
- [ ] Command count verified
- [ ] Complexity assessed (low/medium/high)
- [ ] Target audience identified
- [ ] Primary vs secondary rules applied
- [ ] inventory.json updated
- [ ] Collection README updated
- [ ] Documentation reviewed

---

## Maintenance

### Quarterly Review
- Review all categorizations
- Migrate evolved workflows
- Archive deprecated workflows
- Update rules based on patterns

### When to Recategorize
- Workflow significantly changes
- Production readiness achieved
- Complexity increases/decreases
- New features added
- Use case shifts

---

## Metrics and Reporting

### Collection Health Metrics
- **Balance:** Each collection should have meaningful content
- **Growth:** Track additions/removals over time
- **Usage:** Monitor which collections are most accessed

### Target Distribution (approximate)
- Core: 20-30% (foundational, should be stable)
- Business: 30-40% (largest, production use cases)
- Development: 25-35% (testing and tools)
- Advanced: 5-15% (experimental, smaller set)

---

*Last updated: 2026-02-27*

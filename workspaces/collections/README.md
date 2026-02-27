# Agent Collections

This directory organizes the 43 agents into 4 categorized collections based on their characteristics and use cases.

## The 4 Collection Buckets

### 1. Basic Workflow (`basic_workflow/`)

**Purpose**: Simple, foundational agents that demonstrate core fastWorkflow capabilities

**Characteristics**:
- Low to medium complexity
- Minimal external dependencies
- Single or simple multi-step workflows
- Great for learning and prototyping

**Agent Count**: 10 agents

**Typical Use Cases**: Learning, demos, simple automation tasks

---

### 2. Integration (`integration/`)

**Purpose**: Agents that connect fastWorkflow to external systems and services

**Characteristics**:
- Medium complexity
- External API/service integration
- Authentication and connection management
- Data transformation between systems

**Agent Count**: 10 agents

**Typical Use Cases**: Third-party integrations, system connectivity, data exchange

---

### 3. Complex Workflow (`complex_workflow/`)

**Purpose**: Advanced agents with sophisticated state management and multi-step processes

**Characteristics**:
- High complexity
- Multi-context workflows
- State machine implementations
- Advanced error handling and orchestration

**Agent Count**: 10 agents

**Typical Use Cases**: Business process automation, enterprise workflows, orchestration

---

### 4. Specialized Domain (`specialized_domain/`)

**Purpose**: Industry or domain-specific agents tailored for particular use cases

**Characteristics**:
- Medium to high complexity
- Domain expertise embedded
- Industry-specific workflows
- Specialized integrations and compliance

**Agent Count**: 13 agents

**Typical Use Cases**: Industry-specific automation, vertical solutions, specialized processes

---

## Categorization Criteria

Agents are assigned to collections based on:

1. **Complexity Score** (Low/Medium/High)
   - Logic complexity
   - State management requirements
   - Error handling sophistication

2. **Integration Level** (Simple/Moderate/Complex)
   - Number of external systems
   - Authentication complexity
   - Data transformation needs

3. **Domain Specificity**
   - General-purpose vs. industry-specific
   - Required domain knowledge
   - Regulatory or compliance requirements

4. **Workflow Scope**
   - Single-purpose vs. multi-purpose
   - Workflow stages
   - Orchestration requirements

See `categorization_rules.md` for the complete decision matrix.

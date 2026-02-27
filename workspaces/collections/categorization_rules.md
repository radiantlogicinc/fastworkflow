# Agent Categorization Rules

This document defines the comprehensive rules and criteria for categorizing agents into the 4 collection buckets.

## Overview

Agents are categorized based on a multi-dimensional scoring system that evaluates:
1. Complexity
2. Integration Level
3. Domain Specificity
4. Workflow Scope

## The 4 Collection Buckets

### 1. Basic Workflow
### 2. Integration
### 3. Complex Workflow
### 4. Specialized Domain

---

## Categorization Decision Matrix

### Dimension 1: Complexity Score

**Low Complexity (Score: 1-3)**
- Single-purpose functionality
- Simple parameter extraction
- Minimal state management
- Straightforward error handling
- Direct execution flow

**Medium Complexity (Score: 4-6)**
- Multi-purpose functionality
- Moderate parameter validation
- Some state management
- Standard error handling
- Conditional execution flows

**High Complexity (Score: 7-10)**
- Complex multi-step workflows
- Advanced state machines
- Sophisticated error recovery
- Context hierarchies
- Orchestration of multiple workflows

### Dimension 2: Integration Level

**Simple (Score: 1-3)**
- No external dependencies or 1-2 simple APIs
- Basic authentication (API keys)
- Minimal data transformation
- Direct API calls

**Moderate (Score: 4-6)**
- 2-4 external systems
- OAuth or token-based auth
- Data transformation required
- Rate limiting needed
- Retry logic implemented

**Complex (Score: 7-10)**
- 5+ external systems
- Multiple authentication methods
- Complex data transformation
- Advanced error handling
- Circuit breakers and fallbacks

### Dimension 3: Domain Specificity

**General Purpose (Score: 1-3)**
- Applicable across industries
- No domain knowledge required
- Standard workflows
- Common use cases

**Moderate Domain Focus (Score: 4-6)**
- Some industry preference
- Basic domain knowledge helpful
- Adaptable workflows
- Industry-influenced design

**Highly Specialized (Score: 7-10)**
- Industry-specific
- Deep domain knowledge required
- Compliance requirements
- Industry-standard workflows
- Specialized terminology

### Dimension 4: Workflow Scope

**Single Purpose (Score: 1-3)**
- One primary function
- Linear execution
- Single context
- Minimal dependencies

**Multi-Purpose (Score: 4-6)**
- 2-4 related functions
- Some branching logic
- Few contexts
- Some dependencies

**Broad Scope (Score: 7-10)**
- Many related functions
- Complex branching
- Multiple contexts
- Many dependencies

---

## Collection Assignment Rules

### Rule 1: Basic Workflow Assignment

**Criteria**:
- Complexity Score: 1-4
- Integration Level: 1-3
- Domain Specificity: 1-4
- Workflow Scope: 1-4
- **Total Score**: < 15

**Examples**:
- Hello World Agent
- Calculator Agent
- File Manager Agent
- Todo Manager Agent

**Purpose**: Learning, prototyping, simple automation

---

### Rule 2: Integration Assignment

**Criteria**:
- Integration Level: 4+ (PRIMARY FACTOR)
- Complexity Score: 3-7
- Domain Specificity: 1-6
- Workflow Scope: 2-6
- **Total Score**: 15-25
- **Key Indicator**: Primarily focused on connecting to external systems

**Examples**:
- Database Connector Agent
- REST API Agent
- Slack Integration Agent
- AWS Integration Agent

**Purpose**: System connectivity, data exchange, third-party services

---

### Rule 3: Complex Workflow Assignment

**Criteria**:
- Complexity Score: 7+ (PRIMARY FACTOR)
- Integration Level: 3-8
- Domain Specificity: 1-7
- Workflow Scope: 6+ (SECONDARY FACTOR)
- **Total Score**: 20-30
- **Key Indicators**: 
  - Multi-step processes
  - State machine implementation
  - Orchestration requirements
  - NOT domain-specific

**Examples**:
- Retail Workflow Agent
- Approval Workflow Agent
- ETL Pipeline Agent
- Incident Management Agent

**Purpose**: Business process automation, enterprise workflows

---

### Rule 4: Specialized Domain Assignment

**Criteria**:
- Domain Specificity: 7+ (PRIMARY FACTOR)
- Complexity Score: 5-9
- Integration Level: 3-8
- Workflow Scope: 4-8
- **Total Score**: 19-30
- **Key Indicators**:
  - Industry-specific workflows
  - Compliance requirements
  - Specialized terminology
  - Domain expertise needed

**Examples**:
- Healthcare Patient Agent
- Financial Trading Agent
- Legal Document Agent
- Insurance Claims Agent

**Purpose**: Industry-specific automation, vertical solutions

---

## Decision Tree

```
START
  |
  ├─ Domain Specificity >= 7?
  │   ├─ YES → Specialized Domain
  │   └─ NO → Continue
  │
  ├─ Total Score < 15?
  │   ├─ YES → Basic Workflow
  │   └─ NO → Continue
  │
  ├─ Integration Level >= 5 AND primary purpose is integration?
  │   ├─ YES → Integration
  │   └─ NO → Continue
  │
  ├─ Complexity >= 7 AND Workflow Scope >= 6?
  │   ├─ YES → Complex Workflow
  │   └─ NO → Review manually
```

---

## Edge Cases and Special Rules

### Edge Case 1: High Integration + High Complexity
**Resolution**: If the agent's **primary purpose** is integration → Integration collection
Otherwise → Complex Workflow collection

**Example**: AWS Integration Agent (high complexity, but primary purpose is AWS integration) → Integration

### Edge Case 2: Moderate Domain Specificity
**Resolution**: If Domain Score is 5-6, categorize based on other factors first

**Example**: HR Recruitment Agent (moderately domain-specific) → Check complexity and workflow scope

### Edge Case 3: Mixed Characteristics
**Resolution**: Apply weighted scoring:
- Domain Specificity: 40%
- Complexity: 30%
- Integration Level: 20%
- Workflow Scope: 10%

---

## Scoring Worksheet

Use this worksheet to score a new agent:

| Dimension | Score (1-10) | Notes |
|-----------|--------------|-------|
| Complexity | __ | |
| Integration Level | __ | |
| Domain Specificity | __ | |
| Workflow Scope | __ | |
| **Total** | __ | |

**Decision**:
1. Domain Specificity >= 7? → Specialized Domain
2. Total < 15? → Basic Workflow
3. Integration Level >= 5 AND primary purpose is integration? → Integration
4. Complexity >= 7 AND Workflow Scope >= 6? → Complex Workflow

---

## Examples by Collection

### Basic Workflow (10 agents)
1. hello_world_agent - Simple demo agent
2. messaging_agent - Basic messaging
3. todo_manager_agent - Simple CRUD
4. calculator_agent - Math operations
5. file_manager_agent - File operations
6. data_validator_agent - Data validation
7. text_processor_agent - Text processing
8. logger_agent - Logging
9. notification_agent - Notifications
10. scheduler_agent - Task scheduling

### Integration (10 agents)
11. database_connector_agent - Database integration
12. rest_api_agent - REST APIs
13. email_integration_agent - Email services
14. slack_integration_agent - Slack
15. github_integration_agent - GitHub
16. aws_integration_agent - AWS
17. google_cloud_integration_agent - GCP
18. stripe_payment_agent - Stripe
19. twilio_sms_agent - Twilio
20. sendgrid_email_agent - SendGrid

### Complex Workflow (10 agents)
21. retail_workflow_agent - E-commerce workflows
22. chat_room_agent - Multi-user chat
23. work_item_manager_agent - Project management
24. approval_workflow_agent - Approval processes
25. document_processing_agent - Document workflows
26. etl_pipeline_agent - Data pipelines
27. customer_onboarding_agent - Onboarding
28. incident_management_agent - Incident handling
29. cicd_pipeline_agent - CI/CD
30. lead_management_agent - Lead workflows

### Specialized Domain (13 agents)
31. healthcare_patient_agent - Healthcare
32. financial_trading_agent - Finance
33. legal_document_agent - Legal
34. hr_recruitment_agent - HR
35. real_estate_listing_agent - Real Estate
36. education_course_agent - Education
37. insurance_claims_agent - Insurance
38. restaurant_ordering_agent - Hospitality
39. logistics_shipping_agent - Logistics
40. manufacturing_production_agent - Manufacturing
41. event_management_agent - Events
42. booking_reservation_agent - Hospitality
43. social_media_management_agent - Marketing

---

## Validation Checklist

Before finalizing categorization:

- [ ] Reviewed agent capabilities
- [ ] Evaluated all 4 dimensions
- [ ] Calculated total score
- [ ] Applied decision tree
- [ ] Checked for edge cases
- [ ] Confirmed with examples
- [ ] Verified collection balance
- [ ] Documented decision rationale

---

## Updates and Maintenance

### When to Recategorize
- Agent capabilities significantly change
- New dimensions identified
- Collection imbalance detected
- User feedback indicates miscategorization

### Process for Adding New Agents
1. Create agent definition using template
2. Score across all 4 dimensions
3. Apply decision tree
4. Assign to collection
5. Update collection manifest
6. Update documentation
7. Add to tests

---

## References

- Agent Definition Schema: `/agent-directory/schema.yaml`
- Collection READMEs: `/workspaces/collections/*/README.md`
- Agent Examples: `/examples/*`

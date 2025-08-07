# Extended Workflow Example

This example demonstrates fastWorkflow's workflow inheritance functionality by extending the `simple_workflow_template`.

## How it works

This workflow inherits from `fastworkflow.examples.simple_workflow_template` as declared in `workflow_inheritance_model.json`:

```json
{
    "base": [
        "fastworkflow.examples.simple_workflow_template"
    ]
}
```

## Inheritance Features Demonstrated

### 1. Command Override (`startup.py`)
The `startup.py` command overrides the base template's startup command while still calling the original functionality:
- Imports and calls the base `ResponseGenerator` 
- Adds custom initialization logic
- Demonstrates how to extend rather than completely replace functionality

### 2. New Command (`generate_report.py`)
Adds a completely new command not present in the base template:
- Provides reporting functionality specific to this extended workflow
- Shows how extending workflows can add new capabilities
- Includes proper signature with utterances and parameter extraction

### 3. Wrapper Command (`WorkItem/get_status.py`)
Extends an existing WorkItem command with enhanced functionality:
- Inherits base status functionality from the template
- Adds analytics and enhanced reporting
- Demonstrates selective override of command components

## Command Precedence

The inheritance system follows this precedence order (last occurrence wins):

1. Base template commands (`fastworkflow.examples.simple_workflow_template/_commands/`)
2. **This workflow's commands** (`_commands/` - highest precedence)
3. Core commands (`fastworkflow/_workflows/command_metadata_extraction`)

## Commands Available

### Inherited from Base Template
All commands from `simple_workflow_template` are automatically available:
- All WorkItem commands (add_child_workitem, go_to_workitem, mark_as_complete, etc.)
- Base startup command (overridden but still accessible)

### Extended/New Commands
- `startup` - Enhanced initialization with custom features
- `generate_report` - New reporting functionality
- `WorkItem/get_status` - Enhanced status with analytics

## Usage

Build and train this workflow like any other:

```bash
cd fastworkflow/examples/extended_workflow_example
fastworkflow train
fastworkflow run
```

The system will automatically merge commands from the base template with this workflow's commands, giving this workflow's commands precedence when there are naming conflicts.

## Key Benefits

1. **Code Reuse**: Inherits all functionality from the base template without copying code
2. **Selective Override**: Only override the commands you need to change
3. **Incremental Enhancement**: Add new features while maintaining base functionality
4. **Maintainability**: Changes to base template are automatically inherited
5. **Modularity**: Build complex workflows by composing simpler templates

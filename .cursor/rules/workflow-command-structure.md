# Workflow Command Structure Guide

## Overview
FastWorkflow commands follow a consistent directory structure pattern with automatic MCP tool integration. Each command has separate modules for parameter extraction, response generation, and utterance generation, with Pydantic models driving automatic schema generation.

## Command Directory Structure
```
_commands/
  <command_name>/
    parameter_extraction/
      __init__.py
      signatures.py          # CommandParameters and InputForParamExtraction classes
    response_generation/
      __init__.py
      command_implementation.py  # Core business logic  
      inference.py          # ResponseGenerator class
    utterances/
      __init__.py
      generate_utterances.py    # Utterance generation logic
      plain_utterances.json    # Static utterance examples
      template_utterances.json # Template patterns
```

## Key Components

### Parameter Extraction (signatures.py)
- `CommandParameters` Pydantic class defines expected inputs and automatically generates MCP tool schemas
- Uses `Annotated` fields with validation patterns, descriptions, and examples
- `InputForParamExtraction` provides workflow context for parameter validation
- Automatic JSON Schema generation for MCP compliance

### Response Generation
- `command_implementation.py` contains the core business logic
- `process_command()` function takes Session and CommandParameters
- Returns `CommandProcessorOutput` with structured results
- `inference.py` wraps implementation in `ResponseGenerator` class for framework integration

### Utterances
- `plain_utterances.json` contains example user inputs for training
- `template_utterances.json` contains parameterized patterns  
- `generate_utterances.py` creates training data variations

## Example Command: get_user_details
Located in [examples/retail_workflow/_commands/get_user_details](mdc:examples/retail_workflow/_commands/get_user_details)

### Parameter Pattern with MCP Integration
```python
class CommandParameters(BaseModel):
    """Returns user's first and last name, address, email, payment methods, and order id's"""
    user_id: Annotated[
        str,
        Field(
            default="NOT_FOUND",
            description="The user ID to get details for",
            pattern=r"^([a-z]+_[a-z]+_\d+|NOT_FOUND)$",
            examples=["sara_doe_496"]
        )
    ]
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
```

### Implementation Pattern
```python
def process_command(session: Session, input: CommandParameters) -> CommandProcessorOutput:
    data = load_data()
    result = GetUserDetails.invoke(data=data, user_id=input.user_id)
    return CommandProcessorOutput(status=result)
```

### ResponseGenerator Integration
```python
class ResponseGenerator:
    def __call__(self, session: Session, command: str, command_parameters: CommandParameters) -> CommandOutput:
        output = process_command(session, command_parameters)
        return CommandOutput(
            command_responses=[
                CommandResponse(response=f"User details: {output.status}")
            ]
        )
```

## Tool Integration Patterns

### Automatic MCP Tool Creation
- Commands automatically become MCP tools via `FastWorkflowMCPServer.list_tools()`
- JSON Schema generated from Pydantic `CommandParameters` class
- Tool descriptions generated from class docstrings and field descriptions  
- Field validation patterns, examples, and requirements preserved in schema

### Agent Integration
- Commands integrate with delegated agent architecture
- Individual tools created via `_create_individual_query_tool()` and `_create_individual_mcp_tool()`
- Tool documentation includes parameter schemas and usage examples
- Direct integration with workflow session message queues

### External Tool Integration
- Commands integrate with tools in workflow `tools/` directory
- Tools follow the `Tool` base class pattern with `invoke()` methods
- Data loaded from workflow `data/` directories (e.g., `retail_data/`)
- Business logic separated from command interface

## Validation and Schema Patterns

### Field Validation
- Use regex patterns for structured IDs (order_ids, user_ids, etc.)
- Default to "NOT_FOUND" for missing parameters to handle extraction failures
- Include clear examples in Field descriptions for better schema generation
- Use List[str] for multiple items (item_ids, new_item_ids)

### MCP Schema Generation
- Field descriptions become JSON Schema property descriptions
- Validation patterns become JSON Schema pattern constraints
- Examples become JSON Schema examples for tool documentation
- Required vs optional determined by `field_info.is_required()`

### Parameter Validation
```python
class InputForParamExtraction(BaseModel):
    command: str
    session: fastworkflow.Session
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    @classmethod
    def create(cls, session: fastworkflow.Session, _: str, command: str):
        return cls(command=command, session=session)
```

## Naming Conventions

### File and Directory Structure
- Command names use snake_case (e.g., `get_user_details`)
- Directory structure follows consistent pattern across all commands
- File names follow the module structure pattern

### Class and Method Naming
- Tool classes use PascalCase (e.g., `GetUserDetails`)
- Parameter fields match tool method parameters
- Pydantic models use descriptive names (`CommandParameters`, `InputForParamExtraction`)

### Integration Points
- `process_command()` function as standard entry point
- `CommandProcessorOutput` as standard return type
- `ResponseGenerator.__call__()` as framework integration point

## Current Examples in Retail Workflow

### Available Commands with MCP Integration
- `get_user_details` - User information lookup with user_id parameter
- `get_order_details` - Order status retrieval with order_id parameter
- `cancel_pending_order` - Order cancellation with order_id and reason parameters
- `list_all_product_types` - Product catalog browsing (no parameters required)
- `modify_pending_order_address` - Address updates with order_id and address parameters
- `find_user_id_by_email` - User lookup by email address
- `find_user_id_by_name_zip` - User lookup by name and zip code

### Tool Data Integration
- Commands use tools from [examples/retail_workflow/tools](mdc:examples/retail_workflow/tools)
- Data loaded from [examples/retail_workflow/retail_data](mdc:examples/retail_workflow/retail_data)
- Business logic encapsulated in tool classes with clear interfaces

## Development Best Practices

### Adding New Commands
1. Follow the standard directory structure pattern
2. Define comprehensive Pydantic `CommandParameters` class with full Field annotations
3. Implement business logic in `command_implementation.py` with tool integration
4. Create `ResponseGenerator` wrapper for framework integration
5. Add utterance examples for training data generation

### MCP Integration Considerations
- Command automatically becomes MCP tool - no additional configuration needed
- Parameter schemas generated automatically from Pydantic models
- Test both direct command execution and MCP tool call formats
- Verify schema generation includes proper validation and examples

### Testing Patterns
- Test command implementation with various parameter combinations
- Test MCP tool integration via `tests/test_mcp_server_integration.py` patterns
- Use real workflow data for realistic testing scenarios
- Test parameter validation and error handling 
# FastWorkflow Commands

This directory contains FastWorkflow command files generated from the Python application in `./examples/hello_world/application`.

## Overview

The generated code includes:
- Command files for all public methods and properties in the application
- A command context model mapping classes to commands

## Usage

These commands can be used with FastWorkflow's orchestration and agent frameworks to enable chat-based and programmatic interaction with the application.

## Available Commands

### Global Commands

- **add_two_numbers**
  - Example utterances:
    - `add two numbers`
    - `add two numbers {a} {b}`
    - `call add_two_numbers with {a} {b}`
  - Input model: `Input`
  - Output model: `Output`

## Context Model

The `context_inheritance_model.json` file maps application classes to command contexts, organizing commands by their class.

Structure example:

```json
{
  "ClassA": {"base": ["BaseClass1", ...]},
  "ClassB": {"base": []}
}
```

### Contexts and Commands

## Extending

To add new commands:

1. Add new public methods or properties to your application classes
2. Run the FastWorkflow build tool again to regenerate the command files and documentation

## Testing

You can test these commands using FastWorkflow's MCP server and agent interfaces.

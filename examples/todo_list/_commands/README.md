# FastWorkflow Commands

This directory contains FastWorkflow command files generated from the Python application in `./examples/todo_list/application`.

## Overview

The generated code includes:
- Command files for all public methods and properties in the application
- A command context model mapping classes to commands

## Usage

These commands can be used with FastWorkflow's orchestration and agent frameworks to enable chat-based and programmatic interaction with the application.

## Available Commands

### inheritance Context

No commands in this context.



## Context Model

The `context_inheritance_model.json` file maps application classes to command contexts, organizing commands by their class.

Structure example:

```json
{
  "context_name": {
    "/": ["command1", "command2", ...],
    "base": ["BaseClass1", ...]
  },
  ...
}
```

### Contexts and Commands

#### inheritance Context
No commands in this context.

## Extending

To add new commands:

1. Add new public methods or properties to your application classes
2. Run the FastWorkflow build tool again to regenerate the command files and documentation

## Testing

You can test these commands using FastWorkflow's MCP server and agent interfaces.

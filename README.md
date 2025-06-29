# `fastWorkflow`

[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![CI](https://img.shields.io/badge/ci-passing-brightgreen)](<LINK_TO_CI>)

Build agents and assistants for complex workflows and large-scale Python applications, with deterministic or AI-powered business logic.

---

### Why fastWorkflow?

- ✅ **Unlimited Tool Scaling**: fastworkflow can scale to an unlimited number of tools
- ✅ **Cost-Effective Performance**: fastWorkFlow with small, free models can match the quality of large expensive models
- ✅ **Reliable Tool Execution**: fastworkflow validation pipeline virtually eliminates incorrect tool calling or parameter extraction, ensuring a reliable tool response
- ✅ **Adaptive Learning**: 1-shot learning from intent detection mistakes. It learns your conversational vocabulary as you interact with it
- ✅ **Interface Flexibility**: Support programmatic, assistant-driven and agent-driven interfaces with the same codebase
- ✅ **Deep Code Understanding**: fastworkflow understands classes, methods, inheritance and aggregation so you can quickly 'AI-enable' large-scale Python applications

---

### Key Concepts

**Adaptive Intent Understanding**: Misunderstandings are a given in any conversation, no matter how intelligent the participants. Natural language applications should have intent clarification and parameter validation built-in. We have the ability to 1-shot adapt our semantic understanding of words and sentences based on the context of the conversation and clarifications of intent. Applications should also be able to do the same.

**Contextual Hierarchies**: Communication is always within a context. And not just one concept but layers of contexts. Interpretation starts with the narrowest context and expands to larger contexts if the narrow context does not 'fit' the interpretation. In programming languages, we express contexts as classes, tools as methods and context hierarchies using inheritance and aggregation. Natural language applications should understand classes, methods, inheritance and aggregation out-of-the-box.

**Signatures**: Signatures (ALA Pydantic and DSPy) are the most efficient way of mapping natural language commands to tool implementations, whether programmatic or GenAI. We use signatures as a backbone for implementing commands, enabling seamless integration with DSPy for producing LLM-content within a deterministic programming framework.

**Code Generation**: AI-enabling large-scale, complex applications is non-trivial. Build tools that can quickly map natural language commands to application classes and methods are critical if we are to build more than prototypes and demos.

**Context Navigation at Runtime**: Classes maintain state, not just methods. Method behaviors can change based on state. These capabilities are the building blocks for creating complex finite-state-machines on which non-trivial workflows are built. We need to support dynamically enabling/disabling methods along with the ability to navigate object instance hierarchies at run-time, if we want to build complex workflows.

---

## Architecture Overview

`fastWorkflow` separates the build-time, train-time, and run-time concerns. The `build` tool creates a command interface from your code, the `train` tool builds NLP models to understand commands, and the `run` scripts execute the workflow.

```mermaid
graph LR
    subgraph A[Build-Time]
        A1(Your Python App Source) --> A2{fastworkflow.build};
        A2 --> A3(Generated _commands);
        A3 --> A4(context_inheritance_model.json);
    end

    subgraph B[Train-Time]
        B1(Generated _commands) --> B2{fastworkflow.train};
        B2 --> B3(Trained Models in ___command_info);
    end

    subgraph C[Run-Time]
        C1(User/Agent Input) --> C2{Intent Detection};
        C2 --> C3(CommandExecutor);
        C3 --> C4(Your Application Logic);
        C4 --> C5(Response);
    end

    A --> B;
    B --> C;
```

---

## Installation

To get started, install `fastWorkflow` from PyPI using pip:

```sh
pip install fastworkflow
# Or with uv
uv pip install fastworkflow
```

**Note:** `fastWorkflow` installs PyTorch as a dependency. If you don't already have PyTorch installed, this could take 20-30 minutes depending on your internet connection and system.

---

## Quick Start: Running an Example in 5 Minutes

This is the fastest way to see `fastWorkflow` in action.

### Step 1: Fetch the `hello_world` Example

The `fastworkflow` command-line tool can fetch bundled examples:

```sh
fastworkflow examples fetch hello_world
```
This command will:
1. Copy the `hello_world` example into a new `./examples/hello_world/` directory.
2. Copy the environment files to `./examples/fastworkflow.env` and `./examples/fastworkflow.passwords.env`.

### Step 2: Add Your API Keys

The example workflows require API keys for the LLM models. Edit the `./examples/fastworkflow.passwords.env` file:

```sh
# Edit the passwords file to add your API keys
nano ./examples/fastworkflow.passwords.env
```

You'll need to add at least:
```
LITELLM_API_KEY_SYNDATA_GEN=your-mistral-api-key
LITELLM_API_KEY_PARAM_EXTRACTION=your-mistral-api-key
LITELLM_API_KEY_RESPONSE_GEN=your-mistral-api-key
LITELLM_API_KEY_AGENT=your-mistral-api-key
```

You can get a free API key from [Mistral AI](https://mistral.ai) - the example is configured to use the `mistral-small-latest` model which is available on their free tier.

### Step 3: Train the Example

Train the intent-detection models for the workflow:

```sh
fastworkflow examples train hello_world
```

This step builds the NLP models that help the workflow understand user commands.

### Step 4: Run the Example

Once training is complete, run the interactive assistant:

```sh
fastworkflow examples run hello_world
```

You will be greeted with a `User >` prompt. Try it out by asking "what can you do?" or "add 49 + 51"!

To see other available examples, run `fastworkflow examples list`.

---

## CLI Command Reference

The `fastworkflow` CLI provides several commands to help you work with workflows:

### Examples Management

```sh
# List available examples
fastworkflow examples list

# Fetch an example to your local directory
fastworkflow examples fetch <example_name>

# Train an example workflow
fastworkflow examples train <example_name>

# Run an example workflow
fastworkflow examples run <example_name>
```

### Workflow Operations

```sh
# Build a workflow from your Python application
fastworkflow build --source-dir <app_dir> --output-dir <workflow_dir>/_commands --context-model-dir <workflow_dir>

# Train a workflow's intent detection models
fastworkflow train <workflow_dir> <env_file> <passwords_file>

# Run a workflow
fastworkflow run <workflow_dir> <env_file> <passwords_file>
```

Each command has additional options that can be viewed with the `--help` flag:

```sh
fastworkflow examples --help
fastworkflow build --help
fastworkflow train --help
fastworkflow run --help
```

---

## Understanding the Directory Structure

A key concept in `fastWorkflow` is the separation of your application's logic from the workflow UI definition.

```
my-project/
├── my_app_source/            # <-- Your application's Python package
│   ├── __init__.py
│   └── models.py
│
└── my_workflow_ui/             # <-- The fastWorkflow definition
    ├── .env
    ├── passwords.env
    ├── _commands/              # <-- Generated by the build tool
    │   ├── Greeter/
    │   │   └── greet.py
    │   └── startup.py
    ├── ___command_info/        # <-- Generated by the train tool
    │   ├── Greeter/
    │   │   └── tinymodel.pth
    │   └── ...
    └── context_inheritance_model.json
```
-   Your application code (`my_app_source/`) remains untouched.
-   The `fastWorkflow` definition (`my_workflow_ui/`) contains all the generated files and trained models. The build tool (`--source-dir`) points to your app code, while the output flags (`--output-dir`, `--context-model-dir`) point inside the workflow directory.

---

## Building Your First Workflow: The Manual Approach

Before using the build tool, it's helpful to understand what it does by creating a simple workflow by hand. This will teach you the core concepts.

### Step 1: Design Your Application

Create a simple Python class.

```python
# my_app/greeter.py
class Greeter:
    """A simple class to greet someone."""
    def greet(self, name: str) -> str:
        """Greets the given name."""
        return f"Hello, {name}!"
```

### Step 2: Create the Workflow Directory

Set up the directory structure for your workflow UI.

```sh
mkdir -p my_workflow_ui/_commands/Greeter
touch my_workflow_ui/__init__.py
touch my_workflow_ui/_commands/__init__.py
touch my_workflow_ui/_commands/Greeter/__init__.py
```

### Step 3: Write the Command File

Create a file named `my_workflow_ui/_commands/Greeter/greet.py`. This file tells `fastWorkflow` how to handle the `greet` command for the `Greeter` context.

```python
# my_workflow_ui/_commands/Greeter/greet.py
import fastworkflow
from pydantic import BaseModel, Field

# The Signature defines the command's interface
class Signature:
    # The Input model defines the parameters the command accepts.
    class Input(BaseModel):
        name: str = Field(description="The name of the person to greet.")

    # The Output model defines the structure of the command's result.
    class Output(BaseModel):
        result: str = Field(description="The complete greeting.")

    # Plain utterances are used to train the intent detection model.
    plain_utterances = [
        "greet {name}",
        "say hello to {name}"
    ]

# The ResponseGenerator contains the logic to execute the command.
class ResponseGenerator:
    def __call__(self, workflow: fastworkflow.Workflow, command_parameters: Signature.Input) -> fastworkflow.CommandOutput:
        # Get the instance of your application class from the workflow
        app_instance: Greeter = workflow.command_context_for_response_generation
        
        # Call your application's method
        greeting_result = app_instance.greet(name=command_parameters.name)
        
        # Format the output
        output = Signature.Output(result=greeting_result)
        
        return fastworkflow.CommandOutput(
            command_responses=[
                fastworkflow.CommandResponse(response=output.model_dump_json())
            ]
        )
```

### Step 4: Create the Context Model

Create `my_workflow_ui/context_inheritance_model.json`. This file defines the contexts and their inheritance. For our simple case, it's just the `Greeter`.

```json
{
  "Greeter": {
    "base": []
  }
}
```

### Step 5: Train and Run

Your manual workflow is ready!
```sh
# Train the workflow
fastworkflow train my_workflow_ui/ .env passwords.env

# Run the workflow
fastworkflow run my_workflow_ui/ .env passwords.env --startup_command startup
```
*(Note: You would need to create a `startup.py` command to initialize the `Greeter` instance in a real scenario).*

---

## Refining Your Workflow

### Adding Inheritance

Let's add a new class that inherits from `Greeter`.

```python
# my_app/greeter.py
class LoudGreeter(Greeter):
    def greet(self, name: str) -> str:
        return f"HELLO, {name.upper()}!"
```

Update `context_inheritance_model.json` to reflect the inheritance:
```json
{
  "Greeter": {
    "base": []
  },
  "LoudGreeter": {
    "base": ["Greeter"]
  }
}
```
Now, if you retrain the model, the `LoudGreeter` context will automatically have access to the `greet` command.

### Adding Context Hierarchies

For applications where objects contain other objects (e.g., a `Project` containing `TodoItem`s), you need to tell `fastWorkflow` how to navigate the hierarchy.

Create a file named `my_workflow_ui/_commands/Greeter/_Greeter.py`:
```python
# my_workflow_ui/_commands/Greeter/_Greeter.py
from typing import Optional

class Context:
    @classmethod
    def get_parent(cls, command_context_object: "Greeter") -> Optional[object]:
        # Return the parent object if it exists, otherwise None.
        return getattr(command_context_object, 'parent', None)
```
This `get_parent` method provides the hook `fastWorkflow` needs to navigate up from a child context to its parent, enabling command resolution in nested contexts.

---

## Rapidly Building Workflows with the Build Tool

After understanding the manual process, you can use the `fastworkflow build` command to automate everything. It introspects your code and generates all the necessary files.

Delete your manually created `_commands` directory and run:
```sh
fastworkflow build \
  --source-dir my_app/ \
  --output-dir my_workflow_ui/_commands/ \
  --context-model-dir my_workflow_ui/ \
  --overwrite
```
This single command will generate the `greet.py` command, `get_properties` and `set_properties` for any properties, the `context_inheritance_model.json`, and more, accomplishing in seconds what we did manually.

---

## Environment Variables Reference

| Variable | Purpose | When Needed | Default |
|:---|:---|:---|:---|
| `SPEEDDICT_FOLDERNAME` | Directory name for workflow contexts | Always | `___workflow_contexts` |
| `LLM_SYNDATA_GEN` | LiteLLM model string for synthetic utterance generation | `train` | `mistral/mistral-small-latest` |
| `LLM_PARAM_EXTRACTION` | LiteLLM model string for parameter extraction | `train`, `run` | `mistral/mistral-small-latest` |
| `LLM_RESPONSE_GEN` | LiteLLM model string for response generation | `run` | `mistral/mistral-small-latest` |
| `LLM_AGENT` | LiteLLM model string for the DSPy agent | `run_agent` | `mistral/mistral-small-latest` |
| `LITELLM_API_KEY_SYNDATA_GEN`| API key for the `LLM_SYNDATA_GEN` model | `train` | *required* |
| `LITELLM_API_KEY_PARAM_EXTRACTION`| API key for the `LLM_PARAM_EXTRACTION` model | `train`, `run` | *required* |
| `LITELLM_API_KEY_RESPONSE_GEN`| API key for the `LLM_RESPONSE_GEN` model | `run` | *required* |
| `LITELLM_API_KEY_AGENT`| API key for the `LLM_AGENT` model | `run_agent` | *required* |
| `NOT_FOUND` | Placeholder value for missing parameters during extraction | Always | `"NOT_FOUND"` |
| `MISSING_INFORMATION_ERRMSG` | Error message prefix for missing parameters | Always | `"Missing required..."` |
| `INVALID_INFORMATION_ERRMSG` | Error message prefix for invalid parameters | Always | `"Invalid information..."` |

The example workflows are configured to use Mistral's models by default. You can get a free API key from [Mistral AI](https://mistral.ai) that works with the `mistral-small-latest` model.

---

## Troubleshooting / FAQ

> **`PARAMETER EXTRACTION ERROR`**
> This means the LLM failed to extract the required parameters from your command. The error message will list the missing or invalid fields. Rephrase your command to be more specific.

> **Slow Training**
> Training involves generating synthetic utterances, which requires multiple LLM calls, making it inherently time-consuming. The first run may also be slow due to model downloads from Hugging Face. Subsequent runs will be faster. Set `export HF_HOME=/path/to/cache` to control where models are stored. Training a small workflow takes ~5-8 minutes on a modern CPU.

> **Missing API Keys**
> If you see errors about missing environment variables or API keys, make sure you've added your API keys to the `fastworkflow.passwords.env` file as described in the Quick Start guide.

---

## For Contributors

Interested in contributing to `fastWorkflow` itself? Great!

1.  **Clone the repository:** `git clone https://github.com/your-repo/fastworkflow.git`
2.  **Set up the environment:** Create a virtual environment using your preferred tool (venv, uv, conda, poetry, etc.) with Python 3.11+
3.  **Install in editable mode with dev dependencies:** `pip install -e ".[dev]"`

Please see `CONTRIBUTING.md` for our contribution guidelines and code of conduct.

---

## License

`fastWorkflow` is released under the Apache License 2.0. See [LICENSE](LICENSE) for details.
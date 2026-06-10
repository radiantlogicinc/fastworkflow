# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`fastWorkflow` (v2.20.0) is a Python framework for building NLP-driven workflows and AI agents with deterministic or LLM-powered business logic. It enables "AI-enabling" existing Python applications by wrapping their classes and methods with an intent-detection and parameter-extraction pipeline built on DSPy, scikit-learn, and Pydantic.

## Development Setup

```sh
# Install with all dev/test extras (Poetry)
poetry install --with dev,test

# Or with pip in editable mode
pip install -e ".[server,training]"

# For AWS Bedrock LLM support
poetry install --with aws
```

Python 3.11–3.13 required.

## Key Commands

### Tests
```sh
# Run all tests
pytest tests/

# Run a single test file
pytest tests/test_command_executor.py

# Run a specific test class or function
pytest tests/test_command_executor.py::TestClassName::test_method

# Run with coverage
pytest --cov=fastworkflow tests/

# Skip slow tests
pytest -m "not slow" tests/
```

**Testing philosophy** (from `.cursor/rules/testing_rules.mdc`):
- Don't use Mock fixtures — all tests are integration tests against real components
- Use the real test workflows in `tests/example_workflow/` and `tests/hello_world_workflow/`
- Do NOT remove pytest tests without explicit user approval

### Linting
```sh
make lint   # Runs: isort, black, flake8, pylint, mypy, bandit
```

Individual tools use these flags: `flake8 . --ignore E501,E122,W503,E402,F401`

### Security Audit
```sh
make audit       # Scan full locked dependency graph for CVEs
make audit-json  # Same, outputs to audit-report.json
```

### fastworkflow CLI (for workflow management)
```sh
fastworkflow examples list
fastworkflow examples fetch hello_world

fastworkflow train <workflow_dir> <env_file> <passwords_file>
fastworkflow run   <workflow_dir> <env_file> <passwords_file>          # agent mode
fastworkflow run   <workflow_dir> <env_file> <passwords_file> --assistant  # deterministic

fastworkflow build --app-dir <app_dir> --workflow-folderpath <workflow_dir> --overwrite
fastworkflow refine <workflow_dir> ...                                  # Refine generated commands
fastworkflow run_fastapi_mcp <workflow_dir> <env_file> <passwords_file> # FastAPI/MCP server
```

Bundled example workflows live in `fastworkflow/examples/` (`hello_world`, `retail_workflow`, `messaging_app_*`, `simple_workflow_template`, etc.).

Prefix a command with `/` at the interactive prompt to force deterministic (non-agentic) execution.

## Architecture: Three Phases

```
Build-Time → Train-Time → Run-Time
```

**Build-time** (`fastworkflow/build/`): AST-introspects your Python application and generates `_commands/*.py` files plus `context_inheritance_model.json`.

**Train-time** (`fastworkflow/train/`): Generates synthetic utterances (via LLM + HuggingFace `datasets`) and trains intent-detection models (DistilBERT/BERT via scikit-learn). Outputs go to `___command_info/` inside the workflow directory.

**Run-time**: A three-stage pipeline for every user turn:
1. **Intent Detection** – sklearn classifier identifies the target command
2. **Parameter Extraction** – DSPy + Pydantic validates and extracts inputs
3. **Command Execution** – runs your business logic and generates a response

## Core Runtime Classes

| Class | File | Role |
|---|---|---|
| `Workflow` | `workflow.py` | Singleton per `workflow_id`; holds `Rdict` persistent state |
| `ChatSession` | `chat_session.py` | CLI wrapper; owns `user_message_queue` / `command_output_queue` threads |
| `WorkflowExecutionContext` | `workflow_execution_context.py` | Transport-free core; `process_message(user_input) → CommandOutput` |
| `CommandExecutor` | `command_executor.py` | NLU → routing → execution; also `perform_action()` for direct calls |
| `CommandContextModel` | `command_context_model.py` | Loads `context_inheritance_model.json`; lazy resolution of context/command hierarchy |
| `RoutingDefinition` / `RoutingRegistry` | `command_routing.py` | Per-workflow cache mapping contexts ↔ commands; avoids redundant module imports |
| `CommandDirectory` | `command_directory.py` | Scans `_commands/` for `*.json` metadata |

**Topology B** (current): `WorkflowExecutionContext` is synchronous and transport-free. FastAPI embeds it per-request by calling `process_message` directly. `ask_user` suspends trajectory via `CommandCancelledError` and resumes on next message. `ChatSession` adds optional queues for the CLI `keep_alive` loop.

## Workflow Directory Structure

```
my_workflow/
├── application/                     # Your app code (untouched)
├── _commands/                       # Command implementations (generated + edited)
│   ├── context_inheritance_model.json  # Context/command hierarchy
│   └── <command_name>.py            # Single-file command (preferred)
├── ___command_info/                 # Generated at train-time (gitignore)
├── ___workflow_contexts/            # Session state at run-time (gitignore)
└── ___convo_info/                   # Conversation logs (gitignore)
```

Add to `.gitignore`: `___workflow_contexts`, `___command_info`, `___convo_info`

## Command File Structure (Single-File Pattern)

New commands use a **single `.py` file** in `_commands/`. Legacy commands with subdirectories (`parameter_extraction/`, `response_generation/`, `utterances/`) are being migrated to this pattern. When you encounter both, use the single file.

```python
# _commands/<command_name>.py

class Signature:
    plain_utterances: list[str] = [...]   # Seed utterances for training
    template_utterances: list[str] = [...] # Optional parameterized patterns

    class Input(BaseModel):               # Pydantic params; use NOT_FOUND default
        model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)
        some_param: Annotated[str, Field(default="NOT_FOUND", description="...", examples=[...])]

    class Output(BaseModel):              # Structured result
        ...

    @staticmethod
    def generate_utterances(...): ...     # Optional: diverse utterance generation

    @staticmethod
    def db_lookup(workflow_snapshot, command) -> list[str]: ...     # Optional
    @staticmethod
    def process_extracted_parameters(...): ...  # Optional post-extraction hook


class ResponseGenerator:
    def __call__(self, workflow_snapshot, command: str, cmd_parameters: Signature.Input) -> fastworkflow.CommandOutput:
        return self._process_command(workflow_snapshot, command, cmd_parameters)

    @staticmethod
    def _process_command(workflow_snapshot, command, cmd_parameters) -> fastworkflow.CommandOutput:
        # Your business logic here
        ...
```

## Context Model (`context_inheritance_model.json`)

Each context entry has exactly two possible keys:
- `/` — list of command names available in that context
- `base` — list of parent context names whose commands are inherited

To add a new command: update the JSON, then create `_commands/<command_name>.py`. `CommandRoutingDefinition` validates that every declared command has an implementation.

## FastAPI Service (`fastworkflow/run_fastapi_mcp/`)

Exposes workflows as HTTP with JWT auth, SSE streaming, MCP, and Kubernetes probes.

Key endpoints: `/initialize`, `/invoke_agent`, `/invoke_agent_stream`, `/invoke_assistant`, `/perform_action`, `/new_conversation`, `/probes/healthz`, `/probes/readyz`

Install server extra: `pip install "fastworkflow[server]"` or `poetry install` (it's already in the venv).

## Environment Variables

Two env files per workflow (see `fastworkflow/examples/fastworkflow.env` for a template):

- `fastworkflow.env` — model strings (`LLM_AGENT`, `LLM_PARAM_EXTRACTION`, etc.), logging, intent model IDs
- `fastworkflow.passwords.env` — API keys (`LITELLM_API_KEY_AGENT`, etc.)

Key models (all default to `mistral/mistral-small-latest`): `LLM_SYNDATA_GEN`, `LLM_PARAM_EXTRACTION`, `LLM_RESPONSE_GEN`, `LLM_PLANNER`, `LLM_AGENT`, `LLM_CONVERSATION_STORE`.

LiteLLM Proxy: prefix model names with `litellm_proxy/` and set `LITELLM_PROXY_API_BASE`.

## Issue Tracking

Use **`bd` (beads)** for all task tracking — not markdown TODOs. See `AGENTS.md` for full `bd` command reference.

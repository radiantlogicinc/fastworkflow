# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`fastWorkflow` is a Python framework for building NLP-driven workflows and AI agents with deterministic or LLM-powered business logic. It enables "AI-enabling" existing Python applications by wrapping their classes and methods with an intent-detection and parameter-extraction pipeline built on DSPy, scikit-learn, and Pydantic.

## Testing Philosophy

(from `.cursor/rules/testing_rules.mdc`):
- Don't use Mock fixtures — all tests are integration tests against real components
- Use the real test workflows in `tests/example_workflow/` and `tests/hello_world_workflow/`
- Do NOT remove pytest tests without explicit user approval

## fastworkflow CLI

Run `fastworkflow --help` for the full command list (`examples`, `train`, `run`, `build`, `refine`, `run_fastapi_mcp`). Non-obvious behavior:
- `run` defaults to agent mode; `--assistant` runs deterministic mode.
- Prefix a command with `/` at the interactive prompt to force deterministic (non-agentic) execution.
- `train` validates seed coverage and duplicate capabilities, reuses generated-data caches,
  and automatically chooses selective or full retraining. `--regenerate-utterances` is the
  only training-policy override.

## Architecture: Three Phases

```
Build-Time → Train-Time → Run-Time
```

**Build-time** (`fastworkflow/build/`): AST-introspects your Python application and generates `_commands/*.py` files plus `context_inheritance_model.json`.

**Train-time** (`fastworkflow/train/`): Generates synthetic utterances (via LLM + HuggingFace `datasets`) and fine-tunes BERT-family intent classifiers with PyTorch/Transformers. Outputs go to `___command_info/` inside the workflow directory; publication is atomic and retains only current plus one previous recovery point.

**Run-time**: A three-stage pipeline for every user turn:
1. **Intent Detection** – fine-tuned BERT-family classifier identifies the target command
2. **Parameter Extraction** – DSPy + Pydantic validates and extracts inputs
3. **Command Execution** – runs your business logic and generates a response

**Topology B** (current): `WorkflowExecutionContext` is synchronous and transport-free. FastAPI embeds it per-request by calling `process_message` directly. `ask_user` suspends trajectory via `CommandCancelledError` and resumes on next message. `ChatSession` adds optional queues for the CLI `keep_alive` loop.

## Environment Variables

Two env files per workflow (see `fastworkflow/examples/fastworkflow.env` for a template):

- `fastworkflow.env` — model strings (`LLM_AGENT`, `LLM_PARAM_EXTRACTION`, etc.), logging, intent model IDs
- `fastworkflow.passwords.env` — API keys (`LITELLM_API_KEY_AGENT`, etc.)

Key models (all default to `mistral/mistral-small-latest`): `LLM_SYNDATA_GEN`, `LLM_PARAM_EXTRACTION`, `LLM_RESPONSE_GEN`, `LLM_PLANNER`, `LLM_AGENT`, `LLM_CONVERSATION_STORE`.

LiteLLM Proxy: prefix model names with `litellm_proxy/` and set `LITELLM_PROXY_API_BASE`.

## Issue Tracking

Use **`bd` (beads)** for all task tracking — not markdown TODOs. See `AGENTS.md` for full `bd` command reference.

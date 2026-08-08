# Agent Instructions

This file provides guidance to AI coding agents when working with code in this repository.

Activate the local `.venv` Python environment before running tests or scripts.

**On a CUDA machine, `poetry install` alone gives you CPU torch.** The default is
pinned to CPU-only wheels, because Poetry cannot lock torch from the CPU and CUDA
indexes at once. GPU users need both steps:

```bash
poetry install --with gpu && make install-gpu-torch
```

There is no warning if you skip it — training silently runs on CPU, which is much
slower and easy to mistake for the suite or the model being slow. `make
install-gpu-torch` verifies CUDA is usable before reporting success, so run
`python -c "import torch; print(torch.cuda.is_available())"` if you are unsure
which you have.

## Project Overview

`fastWorkflow` is a Python framework for building NLP-driven workflows and AI agents with deterministic or LLM-powered business logic. It enables "AI-enabling" existing Python applications by wrapping their classes and methods with an intent-detection and parameter-extraction pipeline built on DSPy, PyTorch/Transformers, scikit-learn utilities, and Pydantic.

## Testing Philosophy

(from `.cursor/rules/testing_rules.mdc`):
- Don't use Mock fixtures — all tests are integration tests against real components
- Use the real test workflows in `tests/example_workflow/`, `tests/hello_world_workflow/`, and `tests/todo_list_workflow/`
- Do NOT remove pytest tests without explicit user approval

### Never run two full suites at once

The full suite loads BERT-family models, and two concurrent runs exceed this box's
31 GB. The second run dies to the OOM killer partway through — observed twice at
~69%, reported as `PYTEST_EXIT=137`, which reads like a crash in whatever test was
running rather than a resource problem. If you are working alongside another agent,
**serialise the suite runs**; a run measured against a tree another agent is still
editing is worthless anyway, so waiting costs nothing.

The suite takes ~24 minutes. Budget for it rather than backgrounding it and hoping.

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

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:970c3bf2 -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

**Architecture in one line:** issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export. See https://github.com/gastownhall/beads/blob/main/docs/SYNC_CONCEPTS.md for details and anti-patterns.

## Agent Context Profiles

The managed Beads block is task-tracking guidance, not permission to override repository, user, or orchestrator instructions.

- **Conservative (default)**: Use `bd` for task tracking. Do not run git commits, git pushes, or Dolt remote sync unless explicitly asked. At handoff, report changed files, validation, and suggested next commands.
- **Minimal**: Keep tool instruction files as pointers to `bd prime`; use the same conservative git policy unless active instructions say otherwise.
- **Team-maintainer**: Only when the repository explicitly opts in, agents may close beads, run quality gates, commit, and push as part of session close. A current "do not commit" or "do not push" instruction still wins.

## Session Completion

This protocol applies when ending a Beads implementation workflow. It is subordinate to explicit user, repository, and orchestrator instructions.

1. **File issues for remaining work** - Create beads for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **Handle git/sync by active profile**:
   ```bash
   # Conservative/minimal/default: report status and proposed commands; wait for approval.
   git status

   # Team-maintainer opt-in only, unless current instructions forbid it:
   git pull --rebase
   bd dolt push
   git push
   git status
   ```
5. **Hand off** - Summarize changes, validation, issue status, and any blocked sync/commit/push step

**Critical rules:**
- Explicit user or orchestrator instructions override this Beads block.
- Do not commit or push without clear authority from the active profile or the current user request.
- If a required sync or push is blocked, stop and report the exact command and error.
<!-- END BEADS INTEGRATION -->

## bd: closes are invisible to git until you export with `-o`

Outside the managed block above, because it is repo-specific and must survive that
block being regenerated.

`bd` writes to a local Dolt database; `.beads/issues.jsonl` is a passive export and
is the file git tracks. `bd close` updates the database, **not** the JSONL. Since bd
1.1.2, plain `bd export` writes to stdout — the file needs `-o`:

```bash
bd export -o .beads/issues.jsonl
```

Omitting it does not error. `bd close` reports success, `git status` shows nothing,
and the JSONL still says `open` — which looks exactly like bd silently losing writes,
and was filed twice as that bug (`fix-46c`, `fix-fyw`) before the cause was found.

**A second, separate failure mode, which this habit does not catch.** Eight issues
were found finished but open in one session, including an epic sized at a month of
work. Their `closed_at` timestamps show nobody had ever run `bd close` on them — so
the export bug is not what hid them. The work simply shipped under neighbouring
issues and nobody went back for the parent. The export bug explains why closes stay
invisible *once someone tries*; it does not explain issues nobody closed.

So the habit is two things, not one: export after writing, **and** close the parent
when the last piece of its work lands. Verifying before scheduling is the only thing
that catches the second — one of those eight was the largest item on the board.

Verify by reading the file, not by trusting the command's output. Full detail in
`.cursor/skills/beads-workflow/SKILL.md`.

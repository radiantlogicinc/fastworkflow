# Specification for Evaluating FastWorkflow Agent Against Tau Bench Retail Workflow Challenge

## 1. Overview

### 1.1 Purpose
This specification outlines the requirements and design for a Python-based evaluation tool that assesses the performance of the FastWorkflow agent (located at `/run_agent`) on the Tau Bench retail workflow benchmark (from https://github.com/sierra-research/tau-bench, specifically the retail domain). The tool will simulate multi-turn interactions between the agent and a Tau Bench-compatible environment, compute success metrics (e.g., Pass^1 to Pass^4 rates), accumulate costs (e.g., API token usage), and generate an overall score. This will enable benchmarking FastWorkflow against Tau Bench leaderboards, identifying strengths/weaknesses in tool-calling, reasoning, and task completion for real-world retail scenarios.

The evaluation will focus on:
- **Task Completion**: How effectively the agent resolves user instructions by invoking the correct sequence of tools (e.g., `get_user_details`, `exchange_delivered_order_items`).
- **Multi-Turn Efficiency**: Measuring success across progressive "passes" (e.g., Pass^1 for single-turn success, up to Pass^4 for success within 4 turns/retries).
- **Cost and Reward**: Tracking API costs and final rewards as defined in Tau Bench.

This is not an implementation; it is a blueprint for developers to build the evaluation code.

### 1.2 Scope
- **In Scope**: Loading Tau Bench retail tasks, integrating with FastWorkflow agent, simulating Tau Bench environment, running evaluations, computing metrics (Pass^1 to Pass^4, average reward, total cost), and generating reports.
- **Out of Scope**: Modifying the FastWorkflow agent or Tau Bench codebase; supporting non-retail domains (e.g., airline); real-time API integrations beyond simulation; UI/dashboard for results (console/JSON output only).

### 1.3 Assumptions
- Tau Bench repo is cloned locally or accessible via pip (e.g., `pip install tau-bench` if available; otherwise, use the GitHub source).
- FastWorkflow agent supports tool-calling and multi-turn conversations as per `/run_agent/agent_module.py`.
- Environment variables for API keys (e.g., OpenAI, Anthropic) are set for the agent.
- Tasks are static JSON-like structures as shown in the user query (e.g., with `user_id`, `instruction`, `actions`).

### 1.4 Key Metrics (Based on Tau Bench Leaderboard)
- **Pass^k (k=1 to 4)**: Success rate where the agent achieves a reward of 1.0 (full task completion) within exactly k "passes" or turns. A "pass" is defined as a single agent-environment interaction cycle (query -> action -> response). If success occurs on turn m <= k, it counts for Pass^k only if m == k (strict per leaderboard; confirm with Tau Bench paper if needed). Aggregate as percentage over all tasks.
- **Average Reward**: Mean reward across all tasks (reward = 1.0 for success, 0.0 for failure, partial if Tau Bench supports it).
- **Total Cost**: Sum of API call costs (e.g., in USD or tokens) across all evaluations.
- **Overall Score**: Weighted average of Pass^1 to Pass^4 (e.g., (Pass^1 + Pass^2 + Pass^3 + Pass^4) / 4), or as per Tau Bench's primary metric (e.g., average Pass rate).
- **Error Breakdown**: Categorize failures (e.g., wrong tool, invalid args, max steps exceeded) using Tau Bench's auto_error_identification.py if integrated.

## 2. Requirements

### 2.1 Functional Requirements
1. **Task Loading**:
   - Load retail tasks from Tau Bench's data files (e.g., `tau_bench/tasks/retail.json` or equivalent; parse into a list of dicts with keys: `annotator`, `user_id`, `instruction`, `actions`).
   - Support filtering by task IDs or subsets (e.g., CLI flag `--task-ids 1,2,3`).
   - Validate tasks: Ensure each has a valid `instruction` and list of expected `actions` for ground-truth comparison.

2. **Environment Simulation/Integration**:
   - Use Tau Bench's `Env` class (from `tau_bench.envs.base`) for the retail domain, initialized with retail data (e.g., users.json, orders.json, products.json from `/retail_workflow/retail_data`).
   - Simulate user behavior: The agent acts as the "assistant"; the environment provides observations based on tool calls.
   - Handle resets: For each task, call `env.reset(task_index)` to start with the task's `instruction` as the initial user query.
   - Support multi-turn: Loop until `response.done` or max steps reached, passing agent actions to `env.step(action)`.

3. **Agent Integration**:
   - Initialize FastWorkflow agent from `/run_agent` (e.g., using `initialize_dspy_agent` or equivalent entrypoint).
   - Map Tau Bench tools to FastWorkflow commands (e.g., Tau's `get_user_details` -> FastWorkflow's `_commands/get_user_details.py`).
   - Adapt agent's input: Feed environment observations (e.g., user instructions, tool outputs) as agent queries.
   - Extract actions: Parse agent's output into Tau Bench's `Action` format (e.g., `{"name": "get_order_details", "arguments": {"order_id": "#W123"}}`).
   - Handle reasoning: If agent uses ReAct-style (Thought/Action), extract only the Action for env.step.

4. **Evaluation Loop** (Adapted from Tau Bench's solve()):
   - For each task:
     - Reset environment with task.
     - Initialize message history with system prompt (e.g., wiki/policy from Tau Bench) + initial observation.
     - Loop (up to max_num_steps=30):
       - Generate agent's next step (message, action, cost).
       - Step environment with action -> get observation, reward, done, info.
       - Append to history.
       - Accumulate cost.
     - Record final reward, steps taken, success (reward==1.0), and pass level (e.g., if succeeded on turn 2, counts for Pass^2+).
   - Ground-Truth Validation: Optionally compare agent's action sequence to task's expected `actions` for accuracy.

5. **Metrics Computation**:
   - Per-task: Reward, steps to success, cost, success turn (if any).
   - Aggregate:
     - Pass^1: % of tasks with success on exactly turn 1.
     - Pass^2: % of tasks with success within exactly 2 turns (or cumulative; align with Tau Bench paper).
     - Similarly for Pass^3 and Pass^4.
     - Average Reward: Mean over tasks.
     - Total/Avg Cost: Sum/mean API costs.
     - Overall Score: Arithmetic mean of Pass^1 to Pass^4.
   - Error Analysis: Use Tau Bench's auto_error_identification.py to classify failures (e.g., fault type: wrong tool, partial completion).

6. **Reporting**:
   - Console output: Table of per-task results + aggregates.
   - JSON export: Full results (tasks, histories, metrics) for reproducibility.
   - Leaderboard Comparison: Print metrics in format matching Tau Bench (e.g., table with Strategy | Pass^1 | Pass^2 | ...).

### 2.2 Non-Functional Requirements
- **Performance**: Support parallel evaluation (e.g., `--max-concurrency 10`) to handle API rate limits.
- **Configurability**: CLI flags for model (e.g., `--model gpt-4o`), provider (e.g., `--provider openai`), max steps, temperature, use_reasoning (ReAct vs. Act).
- **Error Handling**: Retry on API failures (up to 3x); log invalid actions; handle task validation errors.
- **Dependencies**: Python 3.10+, litellm (for completions), tau-bench (pip or local), rich (for console tables), pydantic (for parsing).
- **Reproducibility**: Seed random elements; cache results if `--clear-cache false`.
- **Security**: Do not hardcode API keys; use env vars.

## 3. Design

### 3.1 Architecture
- **Main Script**: `evaluate_tau_bench.py` (in `/run_agent` or new dir `/evaluation`).
- **Modules**:
  - `task_loader.py`: Load/parse Tau Bench retail tasks.
  - `env_adapter.py`: Wrapper for Tau Bench Env, mapping to FastWorkflow data/tools.
  - `agent_runner.py`: Interface to run FastWorkflow agent, parse outputs to Actions.
  - `evaluator.py`: Core loop (like solve()), metrics computation.
  - `reporter.py`: Generate console/JSON outputs.

### 3.2 Data Flow
1. Load tasks -> For each: Reset Env -> Init messages with instruction.
2. Agent generates action -> Env steps -> Update messages with obs.
3. Repeat until done or max steps -> Compute per-task metrics.
4. Aggregate and report.

### 3.3 Edge Cases
- Agent fails to parse (invalid JSON): Treat as error, assign reward 0.
- Max steps exceeded: Failure (reward 0).
- Partial success: If Tau Bench supports fractional rewards, use them.
- Tool Mismatch: If FastWorkflow lacks a tool, log and skip task.

### 3.4 Extensibility
- Add flags for other domains (e.g., `--env airline`).
- Support custom agents (e.g., via abstract Agent class).

This spec provides a complete blueprint; implementation should follow Tau Bench's MIT license and cite the repo/paper. If clarifications needed (e.g., exact Pass^k definition), refer to Tau Bench paper or run their examples.
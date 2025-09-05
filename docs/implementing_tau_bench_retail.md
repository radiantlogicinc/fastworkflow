# Specification for Evaluating FastWorkflow Agent Against Tau Bench Retail Workflow Challenge

## 1. Overview

### 1.1 Purpose
This specification outlines the requirements and design for evaluating the FastWorkflow agent via a programmatic agent session API (non-interactive; not the CLI) against the Tau Bench retail workflow benchmark (from https://github.com/sierra-research/tau-bench, specifically the retail domain). The integration is implemented in the Tau Bench fork as a thin adapter plus a runtime bridge that monkey-patches FastWorkflow's agent tool execution to perform agent-internal stepping with the Tau Bench environment (no fastworkflow code changes). The adapter is registered with Tau Bench's existing harness (`run.py`), minimizing changes to both repositories and requiring no modifications to Tau Bench tests. The bridge simulates multi-turn interactions, computes success metrics (e.g., Pass^1 to Pass^4 rates), accumulates costs (e.g., API token usage), and generates an overall score.

The evaluation will focus on:
- **Task Completion**: How effectively the agent resolves user instructions by invoking the correct sequence of tools (e.g., `get_user_details`, `exchange_delivered_order_items`).
- **Multi-Turn Efficiency**: Measuring success across progressive "passes" (e.g., Pass^1 for single-turn success, up to Pass^4 for success within 4 turns/retries).
- **Cost and Reward**: Tracking API costs and final rewards as defined in Tau Bench.

This is not an implementation; it is a blueprint for developers to build the evaluation code.

### 1.2 Scope
- **In Scope**: Using Tau Bench's native task loading and environment; integrating FastWorkflow as an agent through a thin adapter within the Tau Bench fork; registering the adapter with the existing `run.py` harness (e.g., `--agent-strategy fastworkflow`); running evaluations; computing metrics (Pass^1 to Pass^4, average reward, total cost); and generating reports in Tau Bench's expected formats.
- **Out of Scope**: Changing Tau Bench tests; large-scale refactors in either repo; supporting non-retail domains (e.g., airline); real-time API integrations beyond simulation; UI/dashboard for results (console/JSON output only).

### 1.3 Assumptions
- Tau Bench repo is forked/cloned locally and will install the latest `fastworkflow` (e.g., `pip install -U fastworkflow`) into its environment.
- FastWorkflow exposes a programmatic API to start and step a workflow chat session as an agent (e.g., `create_agent_session` / `start_agent_session`) without using the interactive CLI. If such an API is not yet public, only minimal, additive changes are made to expose it.
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

3. **Agent Integration (Option A: agent-internal stepping via runtime bridge)**:
   - Implement a thin adapter inside the Tau Bench fork (e.g., `tau_bench/agents/fastworkflow_adapter.py`) that conforms to Tau Bench's agent interface and is selectable via the existing harness (no test edits).
   - At adapter initialization, install a runtime bridge (monkey patch) that overrides FastWorkflow's internal tool execution entrypoint used by the agent (e.g., patch `fastworkflow.workflow_agent._execute_workflow_query`). The bridge must:
     - Plan-only: Resolve the exact command name and validated parameters without executing tools by invoking the CME “wildcard” route (extract `command_name`, `cmd_parameters`) and serializing parameters.
     - Translate the plan into Tau Bench `Action` ({`name`, `arguments`}).
     - Call `env.step(action)` directly (agent-internal stepping), obtain observation/reward/done.
     - Format the observation (deterministic `render_obs_for_agent`) and return it as the tool result to the agent so the agent can use it in subsequent turns.
   - Provide a minimal tool-name mapping layer to translate Tau Bench tool names to FastWorkflow tool identifiers if they differ (prefer 1:1 naming to avoid churn).
   - Avoid modifying tests: All formatting and structures must exactly match what the Tau Bench environment expects.
   - No changes to `fastworkflow` source are required; the bridge is installed only when `--agent-strategy fastworkflow` is selected.

4. **Evaluation Loop** (Reuse Tau Bench's `solve()`/runner with agent-internal stepping):
   - Use Tau Bench's existing loop/infrastructure; register/select the FastWorkflow adapter as the agent under test (e.g., via CLI flag or agent registry) without altering test cases.
   - Expose the agent via Tau Bench's CLI as: `python run.py --agent-strategy fastworkflow --env retail --model <m> --model-provider <p> --user-model <m> --user-model-provider <p> --user-strategy llm --max-concurrency <n>`.
   - For each task:
     - Reset environment with task.
     - Initialize message history with system prompt (e.g., wiki/policy from Tau Bench) + initial observation.
     - Loop (up to max_num_steps=30):
       - The agent, via the installed bridge, plans the next action, internally executes `env.step(action)`, and receives the observation text as the tool result.
       - The adapter simply advances the conversation (no external `env.step` call by the harness); append the observation to history.
       - Accumulate cost and step metrics via a callback from the bridge.
       - Break if `done` or max steps.
     - Record final reward, steps taken, success (reward==1.0), and pass level (e.g., if succeeded on turn 2, counts for Pass^2+).
   - Ground-Truth Validation: Optionally record the planned action sequence for comparison to task `actions`.

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
- **Configurability**: CLI flags for model (e.g., `--model gpt-4o`), provider (e.g., `--provider openai`), max steps, temperature, use_reasoning (ReAct vs. Act). Map these to FastWorkflow session configuration.
- **Compatibility**: Zero changes to Tau Bench tests; adapter strictly conforms to expected action/result formats.
- **Programmatic API Only**: Use FastWorkflow's non-interactive, programmatic agent session API (no CLI, no streaming I/O). Disable streaming; return structured steps.
- **Error Handling**: Retry on API failures (up to 3x); log invalid actions; handle task validation errors.
- **Dependencies**: Python 3.10+, litellm (for completions), tau-bench (pip or local), fastworkflow (latest), rich (for console tables), pydantic (for parsing).
- **Reproducibility**: Seed random elements; cache results if `--clear-cache false`.
- **Security**: Do not hardcode API keys; use env vars.

## 3. Design

### 3.1 Architecture
- **Integration Location (Tau Bench fork)**:
  - `tau_bench/agents/fastworkflow_adapter.py`: Thin adapter implementing Tau Bench's agent interface; constructs the FastWorkflow session, installs the runtime bridge, and orchestrates the conversation.
  - `tau_bench/agents/fastworkflow_bridge.py`: Runtime monkey-patch that overrides FastWorkflow's tool execution entrypoint used by the agent to:
    1) plan-only (extract `name`, `arguments`), 2) call `env.step(action)`, 3) format and return observation text to the agent, and 4) stream step metrics to the adapter.
  - `tau_bench/agents/fastworkflow_tool_map.py` (optional): Centralized mapping from Tau Bench tool names to FastWorkflow tool identifiers, if names differ.
  - Registration: Add the adapter to the existing agent factory/registry so it is selectable via `--agent-strategy fastworkflow` in `run.py`.
- **FastWorkflow**:
  - No code changes required. The bridge is installed at runtime only for the FastWorkflow strategy.
- **Reuse Existing Tau Bench Components**:
  - Use Tau Bench's native task loaders, environment, evaluator/solver, and reporting. Do not duplicate these in FastWorkflow.

### 3.2 Data Flow
1. Tau Bench loads tasks and environment as usual.
2. For each task: Reset Env -> Initialize messages with instruction.
3. Agent, via the bridge, plans an action, executes `env.step(action)` internally, and returns observation text to itself.
4. Adapter advances the agent turn, accumulating cost/metrics via bridge callbacks.
5. Repeat until done or max steps -> Compute per-task metrics via Tau Bench's existing logic.
6. Aggregate and report using Tau Bench's reporting.

### 3.3 Edge Cases
- Agent fails to produce a valid `Action`: Treat as error, assign reward 0.
- Max steps exceeded: Failure (reward 0).
- Partial success: If Tau Bench supports fractional rewards, use them.
- Tool mismatch or missing tool: Log clearly and skip task or return a no-op action per Tau Bench conventions; prefer aligning tool names to avoid mapping where possible.
- Optional toggle (debug): The bridge can be switched to “decision-only” to return planned actions as JSON with external stepping (for debugging) without changing tests.

### 3.4 Extensibility
- Add flags for other domains (e.g., `--env airline`) by setting appropriate `run_as` domain values.
- Support custom agents by swapping adapters via Tau Bench's agent registry without test changes.

### 3.5 Runtime Bridge Implementation Plan (Option A)

This section specifies the concrete design and implementation plan for the runtime bridge that enables agent-internal stepping without modifying the `fastworkflow` package.

#### 3.5.1 Files to add (Tau Bench fork)
- `tau_bench/agents/fastworkflow_adapter.py`: Adapter wiring to Tau Bench harness (agent factory) and lifecycle.
- `tau_bench/agents/fastworkflow_bridge.py`: Runtime monkey-patch that intercepts FastWorkflow agent tool execution and performs env stepping.
- `tau_bench/agents/fastworkflow_tool_map.py` (optional): Stable mapping of Tau Bench tool names to FastWorkflow tool identifiers.

#### 3.5.2 Runtime bridge (monkey-patch)
- Intercept the agent’s synchronous tool execution entrypoint (preferred target: `fastworkflow.workflow_agent._execute_workflow_query`).
- For each call:
  1) Plan-only using the CME wildcard to resolve the final command name and validated parameters (no tool side effects).
  2) Build a Tau Bench `Action` dict: `{ "name": <command_name>, "arguments": <validated params dict> }`.
  3) Call `env.step(action)` (agent-internal stepping) and obtain `(obs, reward, done, info)`.
  4) Format a deterministic observation string and return it as the tool result to the agent.
  5) Emit a per-step callback for metrics (action, reward, done), accumulated by the adapter.

Example (pseudocode):

```python
# tau_bench/agents/fastworkflow_bridge.py
import json
import fastworkflow
from fastworkflow import Action
from fastworkflow.command_executor import CommandExecutor

_SESSION_ENV: dict[int, any] = {}
_STEP_CB = None  # optional callback(dict)

def set_step_callback(fn):
    global _STEP_CB
    _STEP_CB = fn

def render_obs_for_agent(obs, info) -> str:
    # Implement deterministic, concise formatting (e.g., JSON)
    return json.dumps({"observation": obs, "info": info}, ensure_ascii=False)

def install_bridge(env) -> None:
    import fastworkflow.workflow_agent as wa

    def _execute_workflow_query_bridge(command: str,
                                       chat_session_obj: fastworkflow.ChatSession) -> str:
        _SESSION_ENV[id(chat_session_obj)] = env

        # Plan-only via CME wildcard
        cme_out = CommandExecutor.perform_action(
            chat_session_obj.cme_workflow,
            Action(command_name="wildcard", command=command)
        )
        resp = cme_out.command_responses[0]
        artifacts = resp.artifacts
        name = artifacts["command_name"]
        params = artifacts.get("cmd_parameters")

        # Best-effort parameter serialization
        if hasattr(params, "model_dump"):
            args = params.model_dump()
        elif hasattr(params, "dict"):
            args = params.dict()
        elif params is None:
            args = {}
        else:
            args = dict(params)

        tau_action = {"name": name, "arguments": args}
        obs, reward, done, info = _SESSION_ENV[id(chat_session_obj)].step(tau_action)

        if _STEP_CB:
            _STEP_CB({"action": tau_action, "reward": reward, "done": done, "info": info})

        return render_obs_for_agent(obs, info)

    # Install monkey patch
    wa._execute_workflow_query = _execute_workflow_query_bridge
```

Notes:
- Maintain a per-session Env registry (`_SESSION_ENV`) to ensure isolation in concurrent runs.
- Provide a stable observation formatter (`render_obs_for_agent`) for agent consumption.
- Optionally expose `set_step_callback(fn)` for the adapter to collect metrics.
- Fallback: if a future FastWorkflow version changes `_execute_workflow_query`, patch `CommandExecutor.invoke_command` with the same plan→env.step→return-text logic.

#### 3.5.3 Adapter wiring (`fastworkflow_adapter.py`)
- Responsibilities:
  - Construct `env` from Tau Bench context.
  - Install the runtime bridge: `install_bridge(env)`.
  - Initialize the FastWorkflow session/agent as usual.
  - Provide an agent wrapper for the harness that starts the conversation and reads step-level metrics from the bridge’s callback.

Adapter sketch:

```python
# tau_bench/agents/fastworkflow_adapter.py
from .fastworkflow_bridge import install_bridge, set_step_callback

class FastWorkflowAgentAdapter:
    def __init__(self, env, model_cfg):
        install_bridge(env)
        set_step_callback(self._on_step)
        # create/start fastworkflow session/agent here (programmatic API)
        self.metrics = []

    def _on_step(self, step_info: dict):
        self.metrics.append(step_info)

    def run_episode(self, task):
        # Initialize conversation with task instruction
        # Drive the agent until it finishes internally (bridge performs env.step)
        # Return per-episode results (reward, pass-level, etc.)
        ...
```

#### 3.5.4 Metrics & cost
- The bridge’s step callback reports `{action, reward, done, info}` per turn.
- The adapter accumulates these to compute: steps-to-success, Pass^k, average reward, costs (if available from model provider logs), and total tokens.
- If provider cost/tokens are tracked elsewhere, the adapter can merge them with step metrics on a per-turn basis.

#### 3.5.5 Concurrency & safety
- Use a per-session map (`_SESSION_ENV[id(chat_session)]`) to avoid cross-talk in concurrent evaluations.
- Ensure the bridge is installed only for `--agent-strategy fastworkflow` and removed/ignored otherwise.

#### 3.5.6 Debugging: decision-only mode
- Add a non-default debug toggle in the bridge to return planned action JSON without calling `env.step`. The harness can then step externally.
- Keep default as agent-internal stepping.

#### 3.5.7 Error handling
- Parameter resolution failures: return a crisp agent-readable error string so the agent can recover or re-plan.
- Env rejects action: format the error in the observation string; still record the step in metrics.
- Max steps exceeded: adapter marks failure and stops.

#### 3.5.8 Testing plan
- Unit: Parameter serialization (Pydantic/dataclass/dict) → dict.
- Unit: CME wildcard result contains `command_name` and `cmd_parameters`.
- Unit: Bridge returns deterministic observation text.
- Integration: End-to-end task with known ground truth; verify action sequence, rewards, and Pass^k.

#### 3.5.9 Integration steps
- Register adapter in Tau Bench agent factory so it is selectable via `--agent-strategy fastworkflow`.
- On adapter init: install the bridge, configure the model/provider, and create the FastWorkflow session.
- The harness remains unchanged for this strategy path; the agent executes internally and reports metrics via the adapter.

#### 3.5.10 Fallback strategy
- If `_execute_workflow_query` becomes unavailable, patch `CommandExecutor.invoke_command` with an equivalent bridge:

```python
from fastworkflow.command_executor import CommandExecutor as CE
_orig_invoke = CE.invoke_command

def invoke_command_bridge(chat_session, command):
    # Same plan-only extraction using chat_session.cme_workflow
    # Then env.step and wrap the observation as a CommandOutput text response
    ...

CE.invoke_command = invoke_command_bridge
```

This fallback keeps the design resilient across FastWorkflow versions without changing either repository’s tests.

This spec provides a complete blueprint; implementation should follow Tau Bench's MIT license and cite the repo/paper. If clarifications needed (e.g., exact Pass^k definition), refer to Tau Bench paper or run their examples.
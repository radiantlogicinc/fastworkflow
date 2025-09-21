## Agent Mode Live Command Traces — Design Specification

### 1. Overview

We will surface real-time command execution traces in agentic mode so users can see what the agent is doing as it plans and calls workflow commands. Traces will stream to the CLI while the agent runs, without changing the existing behavior of writing traces to disk. This keeps developer observability high and preserves current offline logs.

Key properties:
- Keep writing per-step records to the existing `action.jsonl` file (unchanged behavior).
- Add an in-memory streaming channel so the CLI (`run` tool) can render live traces as they occur.
- Render using the existing color scheme: command information in yellow and responses in green.
- Only enabled for agentic mode (`--run_as_agent`). Assistant mode remains unchanged.

### 2. Goals and Non‑Goals

- Goals
  - Show a live, readable sequence of calls the agent makes into the workflow.
  - Format each step exactly as:
    - Agent -> Workflow: <command or failing command text>
    - Workflow -> Agent: <command_name>, <parameters>: <response text>
  - Preserve the on-disk trace file `action.jsonl` as-is.
  - Avoid impacting the agent’s planning or execution performance.

- Non‑Goals
  - No new persistence format beyond the existing `action.jsonl`.
  - No UI overhaul; only incremental updates to the current `run` output.
  - Do not display internal “meta” tools (e.g., guidance or discovery) unless they translate to a workflow command execution via `execute_workflow_query`.
  - No automated tests in this change (manual verification only).

### 3. User Experience

When running with `--run_as_agent`, the CLI will show live trace lines while the spinner is active. For each agent tool call into the workflow:

```
Agent -> Workflow: <raw command text the agent sent>
Workflow -> Agent: <resolved_command_name>, <parameters>: <response text>
```

- The “Agent -> Workflow” line is considered command info and uses dim yellow styling.
- The “Workflow -> Agent” line prints the command name and parameters in dim yellow (command info) and the response text in dim green (responses), matching the existing color scheme except the colors are dim.
- If the command fails (see success semantics), both lines for that step render in dim orange to quickly distinguish failing traces.
- These lines appear incrementally as the agent progresses, followed later by the final panel rendered by existing logic.

### 4. High-Level Design

We will introduce a lightweight event stream for trace messages, emitted synchronously during agent tool execution and consumed by the CLI in real time.

- Producer: Agent execution path in `execute_workflow_query` (agent tool) emits events before and after the workflow command is executed.
- Transport: A new `Queue` on `ChatSession` dedicated to trace events. This avoids coupling to the existing `command_output_queue` used for final outputs and structured command results.
- Consumer: The `run` tool’s spinner loop drains and renders trace events as they arrive.

### 5. Components and Integration Points

- `fastworkflow/chat_session.py` ([chat_session.py](mdc:fastworkflow/chat_session.py))
  - Add a new `Queue` attribute (e.g., `_command_trace_queue`) and an accessor property `command_trace_queue`.
  - The queue is created per `ChatSession` and cleared when the session stops.

- `fastworkflow/workflow_agent.py` ([workflow_agent.py](mdc:fastworkflow/workflow_agent.py))
  - In `_execute_workflow_query(...)`:
    - Emit an “Agent -> Workflow” event before calling `CommandExecutor.invoke_command(...)` containing the raw `command` string.
    - After receiving `command_output`, emit a “Workflow -> Agent” event with `command_name`, `parameters` (best-effort serialized), `response text`, and `success`.
  - Do not change existing file logging behavior; this function should only add streaming events to the queue.

- `fastworkflow/command_executor.py` ([command_executor.py](mdc:fastworkflow/command_executor.py))
  - No required functional change for streaming. Continue writing `action.jsonl` records (already implemented in two code paths: early/handled and resolved execution).
  - Optional: expose utility helpers to normalize/serialize parameters for consistent rendering, but this can also live inside `workflow_agent.py` to keep scope tight.

- `fastworkflow/run/__main__.py` ([__main__.py](mdc:fastworkflow/run/__main__.py))
  - During the “Processing command...” spinner loop, poll `fastworkflow.chat_session.command_trace_queue` non‑blocking and render any events using `rich`.
  - Maintain the current panel printing for the final `CommandOutput`. The trace lines should appear above while the spinner runs; when the spinner completes, the final answer panel prints as it does today.

### 6. Event Model

Define a simple, structured event object for trace lines. This is an internal in‑memory contract only.

```python
class CommandTraceEventDirection(str, Enum):
    AGENT_TO_WORKFLOW = "agent_to_workflow"
    WORKFLOW_TO_AGENT = "workflow_to_agent"

@dataclass
class CommandTraceEvent:
    direction: CommandTraceEventDirection
    raw_command: str | None               # for AGENT_TO_WORKFLOW
    command_name: str | None              # for WORKFLOW_TO_AGENT
    parameters: dict | str | None
    response_text: str | None
    success: bool | None
    timestamp_ms: int
```

Notes:
- `parameters` should be a dict where available; otherwise a best-effort `str` is fine.
- Timestamps help with ordering if the UI ever needs to buffer.
- The producer should fill the appropriate fields based on direction.

Success semantics:
- `success` is derived directly from `CommandOutput.success` (see `fastworkflow/__init__.py`).
- Interpretation: `False` indicates a failing command (e.g., parameter extraction error, misunderstood intent, validation error, or any path where `CommandOutput.success` is false). `True` indicates a successful command. `None` is reserved for the pre-execution event (AGENT_TO_WORKFLOW) or unexpected exceptions before a `CommandOutput` exists.
- Rendering: when `success is False`, render the associated trace lines (both Agent->Workflow and Workflow->Agent for that step) in dim orange; otherwise use dim yellow (info) and dim green (response).

### 7. Emission Details (Agent Path)

Target hook: `_execute_workflow_query(command: str, chat_session_obj: ChatSession) -> str` in [workflow_agent.py](mdc:fastworkflow/workflow_agent.py).

1) Before invoking the command:
```python
chat_session_obj.command_trace_queue.put(CommandTraceEvent(
    direction=AGENT_TO_WORKFLOW,
    raw_command=command,
    timestamp_ms=now_ms(),
))
```

2) After invoking the command:
```python
# After CommandExecutor.invoke_command(...)
name = command_output.command_name or artifacts.get("command_name")
params = command_output.command_parameters or artifacts.get("cmd_parameters")
params_dict = params.model_dump() if hasattr(params, "model_dump") else (params if isinstance(params, dict) else str(params) if params is not None else None)
resp_text = "\n".join([r.response for r in command_output.command_responses if r.response])

chat_session_obj.command_trace_queue.put(CommandTraceEvent(
    direction=WORKFLOW_TO_AGENT,
    command_name=name,
    parameters=params_dict,
    response_text=resp_text or "",
    success=bool(command_output.success),
    timestamp_ms=now_ms(),
))
```

Behavioral notes:
- This works for both the “early/handled or error” path and the standard “resolved command” path because `command_output` is consistently populated by the wildcard route and/or response generator.
- No disk writes are added here. The existing writes in `command_executor.py` remain the single source of persisted traces.

### 8. Consumption and Rendering (CLI)

Target hook: the spinner loop inside [run/__main__.py](mdc:fastworkflow/run/__main__.py) where we currently sleep and update a status message.

Enhancement:
- While the wait thread is alive, poll `fastworkflow.chat_session.command_trace_queue` in a non‑blocking loop, rendering any available events immediately.
- Use `rich` to format lines with the current scheme:
  - Yellow for command information (labels and command name/parameters).
  - Green for response text.

Pseudocode sketch inside the spinner loop:
```python
while wait_thread.is_alive():
    try:
        while True:
            evt = fastworkflow.chat_session.command_trace_queue.get_nowait()
            # choose styles based on success
            info_style = "dim orange3" if (evt.success is False) else "dim yellow"
            resp_style = "dim orange3" if (evt.success is False) else "dim green"

            if evt.direction == AGENT_TO_WORKFLOW:
                console.print(Text("Agent -> Workflow: ", style=info_style), end="")
                console.print(Text(str(evt.raw_command or ""), style=info_style))
            else:
                # command info (dim yellow or dim orange3)
                info = f"{evt.command_name or ''}, {evt.parameters}: "
                console.print(Text("Workflow -> Agent: ", style=info_style), end="")
                console.print(Text(info, style=info_style), end="")
                # response (dim green or dim orange3)
                console.print(Text(str(evt.response_text or ""), style=resp_style))
    except queue.Empty:
        pass

    time.sleep(0.5)
    # existing status.update(...) remains
```

Rendering rules:
- Keep lines concise and fold/wrap naturally (Rich handles wrapping; we can also set `overflow="fold"` when using tables, but simple `Text` printing is fine for lines).
- Normal traces: dim yellow for command info and dim green for the response text.
- Failing command traces (success == False): dim orange for both the info and the response lines of that step.
- These lines are additive to the final panel output already printed after the spinner.

### 9. Configuration

- Default: live traces ON when `--run_as_agent` is set.
- Optional flag (non‑breaking): `--no_agent_traces` to disable streaming traces for quieter output.
- Optional environment override: `FASTWORKFLOW_SHOW_AGENT_TRACES=0/1` (CLI flag wins).

CLI wiring suggestion in [__main__.py](mdc:fastworkflow/run/__main__.py):
- Add an optional `--no_agent_traces` (default False). Effective only if `--run_as_agent` is True.
- Compute `show_agent_traces = args.run_as_agent and not args.no_agent_traces` and guard the render loop drain accordingly.

### 10. Backward Compatibility and Safety

- Assistant mode (non‑agent) is unchanged.
- Disk logging to `action.jsonl` remains unchanged in both early/handled and resolved paths.
- The new queue is per session and resides in memory only; it should be drained continuously to avoid build‑up. The spinner loop already ticks every 0.5s.
- If an exception occurs when producing trace events, agent execution still proceeds; the feature should be best‑effort and never block the agent.

### 11. Edge Cases and Details

- Multi‑response commands: if `command_output.command_responses` has multiple entries, concatenate using newlines for `response_text` in the event. The final panel continues to render each response as today.
- Parameter serialization: prefer `model_dump()` for Pydantic models; fallback to dict or string. If too large, render the `str` form (to avoid verbose output). No truncation is required initially; we can revisit if outputs become too long.
- Early exit and errors: the early “handled” path (e.g., parameter extraction error) still produces a post‑execution event. The response text is printed in green to be consistent with existing “response” styling. The final panel will also show the message as it does today.
- Workflow switching: when a new workflow starts via `start_workflow`, the queue remains attached to the same `ChatSession`. If identifying multiple workflows is needed later, the event shape can be extended without affecting this design.
- Keep‑alive sessions: the queue is drained on every user turn. Residual events (if any) should be read prior to returning to the prompt (the same spinner loop already drains until agent completion).

### 12. Implementation Plan (Edits Summary)

1) `fastworkflow/chat_session.py`
   - Add `self._command_trace_queue = Queue()` in `__init__`.
   - Add property:
     ```python
     @property
     def command_trace_queue(self) -> Queue:
         return self._command_trace_queue
     ```

2) `fastworkflow/workflow_agent.py`
   - In `_execute_workflow_query(...)`, emit two events:
     - Before execution: AGENT_TO_WORKFLOW (raw command).
     - After execution: WORKFLOW_TO_AGENT (name, parameters, response, success).

3) `fastworkflow/run/__main__.py`
   - In the spinner loop, while waiting for the agent’s result, non‑blocking drain of `fastworkflow.chat_session.command_trace_queue` and render each event.
   - Optional flag: add `--no_agent_traces` and compute `show_agent_traces` to guard rendering.

4) Optional utilities
   - If helpful, add a small serialization helper for parameters inside `workflow_agent.py` to keep the event construction clean and consistent with `action.jsonl`.

### 13. Example Output (Colors implied, not shown here)

```
Agent -> Workflow: get_user_details <user_id>sara_doe_496</user_id>
Workflow -> Agent: get_user_details, {"user_id": "sara_doe_496"}: Found user Sara Doe (premium)

Agent -> Workflow: exchange_delivered_order_items <order_id>o-14567</order_id> <sku>sku-778</sku>
Workflow -> Agent: exchange_delivered_order_items, {"order_id": "o-14567", "sku": "sku-778"}: Exchange initiated. RMA #RMA-91823
```

These lines stream during the spinner. After the agent finishes, the existing summary panel prints as usual.

### 14. Future Enhancements (Non‑Blocking)

- Toggle to include/exclude internal meta tools (e.g., discovery/guidance) for deep debugging.
- Rich “Live” container to group trace lines under a titled panel during processing.
- Compact parameter formatting with truncation and tooltips.
- Persist streaming traces into a rolling session log (separate from `action.jsonl`).



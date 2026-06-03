# WorkflowExecutionContext — embedder migration

Use `WorkflowExecutionContext` instead of `ChatSession` when you already own HTTP/session
transport (e.g. FastAPI) and need concurrent sessions without a global active workflow.

## Pattern

```python
import fastworkflow
from fastworkflow import WorkflowExecutionContext

fastworkflow.init(env_vars)

ctx = WorkflowExecutionContext(
    run_as_agent=True,
    ask_user_timeout=120.0,  # Topology A only; safety net when user_message_queue is set
)

app = fastworkflow.Workflow.create(workflow_path, workflow_id_str=session_id)
ctx.bind_app_workflow(app)

# Run each request in its own thread or asyncio task (ContextVar isolation)
output = ctx.process_message(user_query)

ctx.close()  # drops cme_workflow speedict store for this session
```

## Agent mode + ask_user: choose a topology FIRST

`run_as_agent=True` runs the whole ReAct loop *inside* `process_message`. If the agent
calls the `ask_user` tool, behavior depends on whether a `user_message_queue` is wired.
There are two valid topologies — do not mix them.

### Topology A — persistent worker thread (CLI / FastAPI today)

This is exactly what `ChatSession(keep_alive=True)` + `ChatWorker` provides, and what
`run_fastapi_mcp` uses today.

- One long-lived thread per session runs the agent loop and owns the queues.
- HTTP request threads are short-lived: they `put(user_query)` into `user_message_queue`
  and wait on `command_output_queue`. They do NOT call `process_message`.
- On `ask_user`: the clarifying question goes out via `command_output_queue` (request #1
  returns it), and the worker thread blocks on `user_message_queue.get()`.
- The answer arrives as request #2 on a different thread, which only `put()`s into the
  queue. The SAME worker thread resumes inside the same `process_message` call, so the
  full ReAct trajectory is preserved.

**`user_message_queue` is CLI/Topology-A only.** If you need true blocking,
human-in-the-loop `ask_user`, use `ChatSession(keep_alive=True)` or replicate its worker
loop around `WorkflowExecutionContext` with transport queues injected.

### Topology B — per-request suspend/resume (bare core)

Each HTTP request calls `process_message` on its own thread/task. When the agent calls
`ask_user` and no `user_message_queue` is set:

1. The ReAct trajectory is suspended in-memory on the `WorkflowExecutionContext`.
2. `process_message` returns immediately with a normal `CommandOutput` whose first response
   has `artifacts["awaiting_user"] = True` and `response` set to the clarification question.
3. No conversation-history turn is appended yet.
4. The next `process_message(user_answer)` resumes the same trajectory (full parity with
   Topology A: iteration reset, `action.jsonl` append, `raw_user_message`, replan).
5. On final completion, conversation history is appended once, keyed on the **original**
   user message from step 1.

```python
output = ctx.process_message("delete my task")
if output.command_responses[0].artifacts.get("awaiting_user"):
    question = output.command_responses[0].response
    # show question to user, then:
    output = ctx.process_message(user_answer)

# Abandon a pending clarification (timeout between HTTP requests, user navigates away):
ctx.cancel_pending()
```

**Do NOT set `user_message_queue` in Topology B** — it is unused for suspend/resume and
was only needed for the old blocking model.

Requirements:

- Reuse the **same** `WorkflowExecutionContext` instance across suspend and resume
  (in-memory trajectory; disk serialization is deferred).
- Optional `ask_user_timeout` applies only when a queue *is* present (Topology A). In
  Topology B, timeouts between turns are embedder-owned; call `cancel_pending()` to clear
  a wedged `awaiting_user` state.

### Nested intent-clarification `ask_user` (Topology B)

The intent-clarification sub-agent resolves ambiguity inside `_execute_workflow_query` and
normally never surfaces to the embedder. Its last-resort `ask_user` tool still requires a
`user_message_queue`; without one it raises `CommandCancelledError`, aborting the turn
cleanly (same as any other cancellation — suspension state is reset). v1 does **not**
support nested suspend/resume for that path.

## Pattern: setting transport queues

Queues are optional and default to `None`. **You do not need to specify the queue that
delivers the user query** — the query is passed directly as the `process_message(user_query)`
argument, not pulled from `user_message_queue`. For a plain synchronous request/response
embedder (Topology B), you can skip all three queues entirely (the first pattern above).

Specify a queue only when you need its specific channel:

| Queue | When to set it |
| --- | --- |
| `user_message_queue` | ONLY in Topology A (a persistent worker loop you run), to deliver `ask_user` replies. Useless/deadlock-prone in Topology B. NOT used to deliver the initial query. |
| `command_trace_queue` | If you want to stream/collect per-step trace events on a separate consumer. |
| `command_output_queue` | Rarely needed in Topology B — `process_message` already returns the `CommandOutput`. Used by Topology A to deliver outputs (incl. the `ask_user` question) back to request threads. |

```python
from queue import Queue

# Topology B example: trace streaming only; NO user_message_queue (suspend/resume).
ctx = WorkflowExecutionContext(run_as_agent=True)
ctx.bind_app_workflow(app)

command_trace_queue = Queue()  # drained by a separate consumer for live traces
ctx.set_transport_queues(
    command_trace_queue=command_trace_queue,
    # user_message_queue + command_output_queue omitted (default None)
)

output = ctx.process_message(user_query)
if output.command_responses[0].artifacts.get("awaiting_user"):
    output = ctx.process_message(user_follow_up_answer)
```

When `ask_user` fires without a `user_message_queue`, the turn suspends (does not hang or
fail). Resume with the next `process_message(answer)`. Use `cancel_pending()` to abandon.

## Notes

- One `WorkflowExecutionContext` per logical session; each gets its own `cme_workflow` and
  `cme_workflow.context["app_workflow"]` binding.
- `get_active_workflow()` is set only for the duration of `process_message` / `process_action`.
- Topology A (blocking `ask_user`): `ChatSession(keep_alive=True)` or a persistent worker
  with `user_message_queue` injected.
- Topology B (suspend/resume): bare `process_message` per request; check
  `artifacts["awaiting_user"]` or `ctx.awaiting_user`; resume with the next
  `process_message(answer)`; `cancel_pending()` to abandon.
- `ChatSession` remains the CLI/persistent-worker driver: queues, `ChatWorker`, `keep_alive`,
  `start_workflow`.
- `dspy.settings.lm` is process-global; agent/planner calls already use `dspy.context(...)`.

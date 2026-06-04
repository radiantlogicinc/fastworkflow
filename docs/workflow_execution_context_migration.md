# WorkflowExecutionContext — embedder migration

Use `WorkflowExecutionContext` instead of `ChatSession` when you already own HTTP/session
transport (e.g. FastAPI) and need concurrent sessions without a global active workflow.

## Pattern

```python
import fastworkflow
from fastworkflow import WorkflowExecutionContext

fastworkflow.init(env_vars)

channel_id = "my-channel"
ctx = WorkflowExecutionContext(run_as_agent=True, session_key=channel_id)

app = fastworkflow.Workflow.create(workflow_path, workflow_id_str=channel_id)
ctx.bind_app_workflow(app)

output = ctx.process_message(user_query)

ctx.close()  # drops cme_workflow speedict store for this session
```

Use a **stable `session_key`** (e.g. JWT `channel_id`) so `cme_<session_key>` and the app
workflow id reload the same speeddict session across process restarts.

## Agent mode + ask_user: choose a topology FIRST

`run_as_agent=True` runs the whole ReAct loop *inside* `process_message`. If the agent
calls the `ask_user` tool, behavior depends on whether a `user_message_queue` is wired.
There are two valid topologies — do not mix them.

### Topology A — persistent worker thread (CLI)

`ChatSession(keep_alive=True)` + `ChatWorker`:

- Worker thread blocks on `user_message_queue.get()` during `ask_user`.
- HTTP/CLI request threads only enqueue messages.

### Topology B — per-request suspend/resume (FastAPI / `run_fastapi_mcp`)

Each HTTP request calls `process_message` in a thread pool. When `ask_user` fires without
a `user_message_queue`:

1. ReAct state is suspended in memory (`fastWorkflowReAct._suspended`).
2. `process_message` returns with `artifacts["awaiting_user"]=True`.
3. The next `process_message(answer)` resumes via `resume()`.
4. `cancel_pending()` or `POST /cancel_pending` abandons the turn.

`run_fastapi_mcp` uses Topology B exclusively (no `ChatWorker`, no queue handoff).

## Durable trajectory serialization (horizontal scale)

Suspended state can be persisted so another worker can resume after cache eviction or restart.

### Serialize / restore API

```python
blob = ctx.serialize_state(channel_id=channel_id)
# ... store.save(channel_id, blob) ...

ctx.apply_serialized_state(blob)
output = ctx.process_message(user_answer)
```

Serialized fields include: `react` (trajectory, idx, input_args, iteration_counter),
`awaiting_user`, `action_log` (replaces per-process `action.jsonl`), `conversation_history_turns`,
`nlu_stage`, and `current_command_context_name` (best-effort navigation hint).

### SessionStateStore backends

Factory: `fastworkflow.session_state_store.get_session_state_store()`

| Env | Backend |
| --- | --- |
| `SESSION_STATE_STORE=disk` (default) | `DiskSessionStateStore` under `SPEEDDICT_FOLDERNAME/channel_session_state` |
| `SESSION_STATE_STORE=redis` | `RedisSessionStateStore` via `SESSION_STATE_REDIS_URL` or `REDIS_URL` |

### Scaling model

- **Sticky routing per `channel_id`** is required for workflow RocksDB (one writer per channel).
- **SessionStateStore** provides durability and rebalancing: any worker can cold-rehydrate
  a live `WorkflowExecutionContext`, re-run startup to rebuild command-context objects,
  then `apply_serialized_state()` before resuming.
- Process-local LRU (`ChannelSessionManager`, default 2000 channels) holds warm contexts;
  evicted awaiting sessions are auto-saved to the store.

### FastAPI endpoints

- `POST /invoke_agent`, `/invoke_assistant`, `/perform_action` — synchronous Topology B turns.
- `POST /invoke_agent_stream` — executor + concurrent `command_trace_queue` drain.
- `POST /cancel_pending` — `cancel_pending()` + clear store.

## Action log (replaces `action.jsonl`)

Agent turns append to `WorkflowExecutionContext._action_log` (in-memory, serialized with
pending state). CLI `ChatSession` sets `mirror_action_log_to_file=True` for optional
debug mirroring to cwd `action.jsonl`.

## Nested intent clarification (Topology B)

The intent-clarification predictor runs inside `_execute_workflow_query`. When it cannot
resolve from metadata/trajectory, it sets `needs_human=True` and the outer workflow agent
receives a directive to call its own `ask_user` (which suspends in Topology B).

## Transport queues

| Queue | When to set it |
| --- | --- |
| `user_message_queue` | Topology A only |
| `command_trace_queue` | Live trace streaming (FastAPI sets this) |
| `command_output_queue` | Topology A only |

## Notes

- `ChatSession` remains the CLI driver (Topology A, `mirror_action_log_to_file`).
- `dspy.settings.lm` is process-global; agent calls use `dspy.context(...)`.

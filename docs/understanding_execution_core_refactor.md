# Understanding the Transport-Free Execution Core Refactor

A learning checklist. We work through this top to bottom. Nothing gets checked
off until you can explain it in your own words, including the *why behind the why*.

Legend: [ ] not yet · [~] in progress · [x] mastered (you explained it back)

---

## Stage 1 — The Problem (most important)

- [x] 1.1 What three responsibilities `ChatSession` was conflating (NLU, active-workflow, transport)
- [x] 1.2 What "active-workflow resolution" means and why a stack exists at all
- [x] 1.3 Why the OLD active-workflow stack was unsafe for concurrent sessions
- [x] 1.4 The two distinct resolution paths (NLU path vs execution path) and why BOTH had to be fixed
- [x] 1.5 Why queues/threads were "dead weight" for a request/response server
- [x] 1.6 Why an embedder had to serialize behind a global lock today (the misuse)

## Stage 2 — The Solution

- [x] 2.1 What `WorkflowExecutionContext` owns and why it is "transport-free"
- [x] 2.2 How `contextvars` isolates the active workflow per thread/task
- [x] 2.3 How the single core feeds BOTH resolution paths
- [x] 2.4 Why `ChatSession` was kept (composition, not rename) and what it still owns
- [x] 2.5 The backward-compat shims and why each was needed
- [x] 2.6 The MCP resolution fallback (`get_active_workflow() or app_workflow`)

## Stage 3 — Edge Cases & Design Decisions

- [x] 3.1 The `ask_user` deadlock problem and the timeout + `CommandCancelledError` fix
- [x] 3.2 Why `CommandCancelledError` subclasses `BaseException`, not `Exception`
- [x] 3.3 Why queues stayed optional/injected rather than owned by the core
- [x] 3.4 The cross-thread `get_active_workflow()` fallback (the regression we caught)
- [x] 3.5 `close()` and the per-session cme_workflow store leak

## Stage 4 — Broader Context & Impact

- [x] 4.1 What this unlocks for a FastAPI / multi-session embedder
- [x] 4.2 What did NOT change and why that matters (wildcard/IntentDetection/ErrorCorrection)
- [x] 4.3 The concurrency guarantee and its limits (threads/tasks vs one sync thread)

---

## Notes / corrections captured during the session

Key takeaways (verified by quiz + capstone):

- The THREE conflated jobs in old ChatSession: (1) NLU/intent extraction via
  cme_workflow, (2) active-workflow resolution (stack), (3) message transport
  (queues + ChatWorker + keep_alive). The NLU one is the easiest to forget.
- TWO independent "find the app workflow" paths:
  - Path 1 (NLU): wildcard/IntentDetection/ErrorCorrection read
    `cme_workflow.context["app_workflow"]`.
  - Path 2 (execution): CommandExecutor / mcp_server call `get_active_workflow()`.
- TWO isolation mechanisms (the crux):
  - Path 1 isolated by each WorkflowExecutionContext owning its OWN cme_workflow.
  - Path 2 isolated by a ContextVar stack (per OS thread / asyncio task), no lock.
  - One core feeds both: `bind_app_workflow` writes Path 1; `process_message`
    pushes the contextvar for Path 2.
- ContextVar = "magic whiteboard": one name, a private copy per thread/task.
- ask_user deadlock (the original problem): in a server, the request thread blocks on
  `user_message_queue.get()` and the answer can only arrive in a FUTURE request, so nothing
  feeds the blocked call. Fix (Topology B): no queue — `ask_user` is non-blocking; it
  suspends the ReAct trajectory and `process_message` returns an `awaiting_user`
  CommandOutput; the next `process_message(answer)` resumes it. No timeout needed; abandon
  via `cancel_pending()`.
- CommandCancelledError subclasses BaseException specifically to slip past ReAct's
  `except Exception` (react.py:136), which would otherwise swallow it into an
  observation and keep looping.
- Queues stay optional/injected (ChatSession owns them) to keep the core
  transport-free; core-owned queues would re-couple transport, leak outputs, and
  fake-support ask_user.
- Regression caught: contextvar push happens on the ChatWorker thread, so
  main/test threads saw an empty stack. Fix: ChatSession.get_active_workflow()
  (and mcp_server) fall back to the bound app_workflow when the stack is empty.
- close() deletes the per-session cme_workflow speedict store (uuid id) to avoid
  accumulating orphaned dirs in long-running embedders.
- Safety preconditions: (1) ONE core per session, (2) each concurrent turn on its
  OWN thread/task. A single shared core re-shares Path 1; same-thread multiplexing
  defeats the ContextVar.
- Why wildcard/IntentDetection/ErrorCorrection were untouched: their contract
  (read cme_workflow.context["app_workflow"]) was preserved; only ownership moved.

"""
Context-scoped active workflow stack (per OS thread / asyncio task).

Replaces the per-ChatSession deque for resolving the current app workflow during
command execution and child-workflow nesting.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Optional

import fastworkflow
from fastworkflow.utils.logging import logger

# Immutable tuple used as a stack; each push/pop replaces the ContextVar value.
_workflow_stack_var: ContextVar[tuple[fastworkflow.Workflow, ...]] = ContextVar(
    "active_workflow_stack",
    default=(),
)


def get_active_workflow() -> Optional[fastworkflow.Workflow]:
    """Get the currently active workflow (top of stack)."""
    stack = _workflow_stack_var.get()
    return stack[-1] if stack else None


def push_active_workflow(workflow: fastworkflow.Workflow) -> None:
    """Push a workflow onto the context-local stack."""
    stack = _workflow_stack_var.get()
    new_stack = stack + (workflow,)
    _workflow_stack_var.set(new_stack)
    logger.debug(f"Workflow stack: {[w.id for w in new_stack]}")


def pop_active_workflow() -> Optional[fastworkflow.Workflow]:
    """Pop a workflow from the context-local stack."""
    stack = _workflow_stack_var.get()
    if not stack:
        return None
    workflow = stack[-1]
    new_stack = stack[:-1]
    _workflow_stack_var.set(new_stack)
    logger.debug(f"Workflow stack after pop: {[w.id for w in new_stack]}")
    return workflow


def clear_workflow_stack() -> None:
    """Clear the entire workflow stack for this context."""
    _workflow_stack_var.set(())
    logger.debug("Workflow stack cleared")
